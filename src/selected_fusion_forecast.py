"""Build the selected heterogeneous forecast after ensemble backtesting.

Selection is frozen from the leakage-safe search results:
DA = XGBoost, direct RT = equal LightGBM/XGBoost/Ridge, spread = Ridge,
coherent RT = DA(XGBoost) + spread(Ridge). Prediction intervals are calibrated
on the preceding 14 days with finite-sample Conformal residual quantiles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import integrated_price_forecast as base
from ensemble_price_forecast import COMPONENTS, fit_component


SELECTED = {
    "da": "xgboost",
    "rt_direct": "equal",
    "spread": "ridge",
    "rt_coherent": "da_xgboost_plus_spread_ridge",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Selected heterogeneous forecast")
    parser.add_argument("--price", type=Path, default=base.PRICE_DEFAULT)
    parser.add_argument("--weather", type=Path, default=base.WEATHER_DEFAULT)
    parser.add_argument("--power-dir", type=Path, default=base.ROOT)
    parser.add_argument("--forecast-date", default="2026-07-01")
    parser.add_argument("--calibration-days", type=int, default=14)
    parser.add_argument("--output-dir", type=Path, default=base.ROOT / "outputs" / "selected_fusion_forecast_20260826")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.power_dir.glob(base.POWER_GLOB))
    frame, coverage = base.load_price_weather(args.price, args.weather, paths)
    frame = base.add_all_feature_tables(frame)
    target_date = pd.Timestamp(args.forecast_date)
    date = frame["market_date"]
    complete = frame["weather_complete"] & frame["power_complete"]
    calibration_start = target_date - pd.Timedelta(days=args.calibration_days)
    cal_mask = (date >= calibration_start) & (date < target_date) & complete
    cal_train_mask = (date < calibration_start) & complete
    train_mask = (date < target_date) & complete
    predict_mask = date.eq(target_date) & complete
    if int(predict_mask.sum()) != 24:
        raise ValueError(f"forecast date needs 24 complete periods, got {int(predict_mask.sum())}")

    final: dict[str, dict[str, np.ndarray]] = {}
    calibration: dict[str, dict[str, np.ndarray]] = {}
    for target in base.TARGETS:
        final[target] = {}
        calibration[target] = {}
        for component in COMPONENTS:
            calibration[target][component] = fit_component(frame, target, cal_train_mask, cal_mask, component)
            final[target][component] = fit_component(frame, target, train_mask, predict_mask, component)

    actual_cal = {target: frame.loc[cal_mask, target].to_numpy(float) for target in base.TARGETS}
    cal_selected = {
        "da": calibration["da"]["xgboost"],
        "rt_direct": np.column_stack([calibration["rt"][c] for c in COMPONENTS]).mean(axis=1),
        "spread": calibration["spread"]["ridge"],
        "rt_coherent": calibration["da"]["xgboost"] + calibration["spread"]["ridge"],
    }
    q = {
        key: base.conformal_quantile(np.abs(actual - cal_selected[key]), 0.1)
        for key, actual in {
            "da": actual_cal["da"],
            "rt_direct": actual_cal["rt"],
            "spread": actual_cal["spread"],
            "rt_coherent": actual_cal["rt"],
        }.items()
    }

    # Keep the classification output from the integrated framework alongside
    # the selected point forecasts. It is advisory and does not change price
    # blending weights.
    spread_columns = base.feature_columns("spread")
    spread_x, _ = base.clean_matrix(frame, spread_columns, train_mask)
    spread_valid = train_mask.to_numpy(bool)
    classifier, classifier_backend = base.make_classifier("lightgbm")
    classifier.fit(
        pd.DataFrame(spread_x[spread_valid], columns=spread_columns),
        (frame.loc[train_mask, "spread"].to_numpy(float) >= 0).astype(int),
    )
    spread_probability = classifier.predict_proba(
        pd.DataFrame(spread_x[predict_mask.to_numpy(bool)], columns=spread_columns)
    )[:, 1]

    target_rows = frame.loc[predict_mask].sort_values("period").reset_index(drop=True)
    rows = []
    for i, row in target_rows.iterrows():
        da_pred = float(final["da"]["xgboost"][i])
        rt_direct = float(np.mean([final["rt"][c][i] for c in COMPONENTS]))
        spread_pred = float(final["spread"]["ridge"][i])
        rt_coherent = da_pred + spread_pred
        rows.append(
            {
                "market_date": target_date.date().isoformat(),
                "period": int(row["period"]),
                "day_ahead_p50": da_pred,
                "day_ahead_p10": da_pred - q["da"],
                "day_ahead_p90": da_pred + q["da"],
                "real_time_direct_p50": rt_direct,
                "real_time_direct_p10": rt_direct - q["rt_direct"],
                "real_time_direct_p90": rt_direct + q["rt_direct"],
                "spread_p50": spread_pred,
                "spread_p10": spread_pred - q["spread"],
                "spread_p90": spread_pred + q["spread"],
                "real_time_coherent_p50": rt_coherent,
                "real_time_coherent_p10": rt_coherent - q["rt_coherent"],
                "real_time_coherent_p90": rt_coherent + q["rt_coherent"],
                "spread_positive_probability": float(spread_probability[i]),
                "spread_direction": "positive" if spread_probability[i] >= 0.5 else "negative",
                "negative_price_risk": bool(da_pred - q["da"] < 0 or rt_coherent - q["rt_coherent"] < 0),
                "high_price_risk": bool(da_pred + q["da"] > 500 or rt_coherent + q["rt_coherent"] > 500),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "forecast.csv", index=False, encoding="utf-8-sig")
    payload = {
        "model_version": "selected-fusion-v1.0.0",
        "forecast_date": args.forecast_date,
        "selection": SELECTED,
        "components": list(COMPONENTS),
        "interval": {"method": "90% Conformal residual interval", "calibration_days": args.calibration_days, "q90": q},
        "forecast": rows,
        "data_coverage": coverage,
        "assumptions": [
            "Weather and power values on the target day are treated as pre-market forecasts.",
            "Direct RT is a benchmark; coherent RT is DA XGBoost plus Ridge spread.",
            "The spread direction classifier is not included in the point-price selection; it remains an advisory risk output.",
            f"Spread direction classifier backend: {classifier_backend}.",
        ],
    }
    (args.output_dir / "forecast.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "model_card.json").write_text(json.dumps({"selection": SELECTED, "interval": payload["interval"], "components": list(COMPONENTS)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "forecast_rows": len(rows), "selection": SELECTED, "q90": q}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
