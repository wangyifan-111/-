"""Autoregressive baselines inspired by AR/ARX/SARIMA EPF literature."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

import integrated_price_forecast as base


def forecast_ar(values: np.ndarray, lags: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= lags + 10:
        return np.repeat(values[-1] if len(values) else 0.0, 24)
    try:
        model = AutoReg(values, lags=lags, trend="ct", old_names=False, period=24).fit()
        pred = np.asarray(model.predict(start=len(values), end=len(values) + 23, dynamic=False), dtype=float)
        if np.isfinite(pred).all():
            return pred
    except Exception:
        pass
    return np.repeat(values[-1], 24)


def run_model(frame: pd.DataFrame, name: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    lags = 24 if name == "ar24" else 168
    days = sorted(pd.to_datetime(frame.loc[frame["weather_complete"] & frame["power_complete"], "market_date"].unique()))
    days = [d for d in days if start <= d <= end]
    rows: list[dict[str, Any]] = []
    date = frame["market_date"]
    complete = frame["weather_complete"] & frame["power_complete"]
    ordered = frame.sort_values("datetime")
    for fold, day in enumerate(days):
        train_rows = ordered.loc[(ordered["market_date"] < day) & complete.loc[ordered.index]]
        pred_rows = ordered.loc[(ordered["market_date"] == day) & complete.loc[ordered.index]].sort_values("period")
        pred: dict[str, np.ndarray] = {}
        for target in ("da", "spread", "rt"):
            pred[target] = forecast_ar(train_rows[target].to_numpy(float), lags)
        for h in range(min(24, len(pred_rows))):
            actual = pred_rows.iloc[h]
            rows.append({"market_date": day.date().isoformat(), "period": h + 1, "da_actual": float(actual["da"]), "spread_actual": float(actual["spread"]), "rt_actual": float(actual["rt"]), f"da_{name}_pred": float(pred["da"][h]), f"spread_{name}_pred": float(pred["spread"][h]), f"rt_{name}_direct_pred": float(pred["rt"][h]), f"rt_{name}_coherent_pred": float(pred["da"][h] + pred["spread"][h])})
        print(f"completed {name.upper()} fold {fold + 1}/{len(days)}: {day.date().isoformat()}", flush=True)
    result = pd.DataFrame(rows)
    scores = {"day_ahead": base.metric(result["da_actual"], result[f"da_{name}_pred"]), "spread": base.metric(result["spread_actual"], result[f"spread_{name}_pred"]), "real_time_direct": base.metric(result["rt_actual"], result[f"rt_{name}_direct_pred"]), "real_time_coherent": base.metric(result["rt_actual"], result[f"rt_{name}_coherent_pred"]), "spread_direction_accuracy": float(((result[f"spread_{name}_pred"] >= 0) == (result["spread_actual"] >= 0)).mean())}
    return result, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="AR-24 and AR-168 rolling trials")
    parser.add_argument("--backtest-start", default="2026-06-15")
    parser.add_argument("--backtest-end", default="2026-06-30")
    parser.add_argument("--output-dir", type=Path, default=base.ROOT / "outputs" / "ar_models_trial_20260831")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, coverage = base.load_price_weather(base.PRICE_DEFAULT, base.WEATHER_DEFAULT, sorted(base.ROOT.glob(base.POWER_GLOB)))
    frame = base.add_all_feature_tables(frame)
    summary: dict[str, Any] = {"models": {}, "backtest_period": {"start": args.backtest_start, "end": args.backtest_end}, "data_coverage": coverage}
    for name in ("ar24", "ar168"):
        result, scores = run_model(frame, name, pd.Timestamp(args.backtest_start), pd.Timestamp(args.backtest_end))
        result.to_csv(args.output_dir / f"{name}_walk_forward_predictions.csv", index=False, encoding="utf-8-sig")
        summary["models"][name] = {"backtest": scores}
    summary["leakage_controls"] = ["Each AR fold uses only target values before the target day.", "Forecast horizon is the next 24 market periods.", "No target-day realized weather or price is used."]
    (args.output_dir / "ar_models_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
