"""Post-day-ahead Shandong real-time price forecast and rolling ensemble.

This forecast is intended to run after the 24-period day-ahead clearing curve
is published.  The observed target-day DA curve is therefore an allowed input,
while target-day RT prices and spreads remain strictly unavailable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor

import integrated_price_forecast as base


MODEL_NAMES = tuple(
    name
    for name in ("lightgbm_l1", "xgboost_l1", "xgboost_l2", "extra_trees", "ridge")
    if not name.startswith("lightgbm") or base.HAS_LIGHTGBM
)
SEGMENTS = {
    "night": tuple(range(1, 7)),
    "solar": tuple(range(7, 17)),
    "evening": tuple(range(17, 25)),
}


def metric(actual: np.ndarray | pd.Series, prediction: np.ndarray | pd.Series) -> dict[str, float]:
    return base.metric(np.asarray(actual, dtype=float), np.asarray(prediction, dtype=float))


def add_known_da_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach features available once the complete target-day DA curve is known."""
    out = frame.copy()
    grouped = out.groupby("market_date", sort=False)["da"]
    out["known_da"] = out["da"]
    out["known_da_day_mean"] = grouped.transform("mean")
    out["known_da_day_std"] = grouped.transform("std").fillna(0.0)
    out["known_da_day_min"] = grouped.transform("min")
    out["known_da_day_max"] = grouped.transform("max")
    out["known_da_day_range"] = out["known_da_day_max"] - out["known_da_day_min"]
    out["known_da_day_median"] = grouped.transform("median")
    out["known_da_centered"] = out["known_da"] - out["known_da_day_mean"]
    out["known_da_zscore"] = out["known_da_centered"] / out["known_da_day_std"].replace(0.0, np.nan)
    out["known_da_zscore"] = out["known_da_zscore"].fillna(0.0)
    out["known_da_rank_pct"] = grouped.rank(method="average", pct=True)

    previous = grouped.shift(1)
    following = grouped.shift(-1)
    first = grouped.transform("first")
    last = grouped.transform("last")
    out["known_da_prev_period"] = previous.where(out["period"].ne(1), first)
    out["known_da_next_period"] = following.where(out["period"].ne(24), last)
    out["known_da_ramp_prev"] = out["known_da"] - out["known_da_prev_period"]
    out["known_da_ramp_next"] = out["known_da_next_period"] - out["known_da"]
    out["known_da_abs_ramp"] = out[["known_da_ramp_prev", "known_da_ramp_next"]].abs().max(axis=1)
    out["known_da_vs_yesterday"] = out["known_da"] - out["da_lag_24"]
    out["known_da_solar_interaction"] = out["known_da"] * out["ghi"]
    out["known_da_power_interaction"] = out["known_da"] * out["net_load_proxy"]
    return out


def feature_columns(target: str) -> list[str]:
    known_da = [
        "known_da", "known_da_day_mean", "known_da_day_std", "known_da_day_min",
        "known_da_day_max", "known_da_day_range", "known_da_day_median",
        "known_da_centered", "known_da_zscore", "known_da_rank_pct",
        "known_da_prev_period", "known_da_next_period", "known_da_ramp_prev",
        "known_da_ramp_next", "known_da_abs_ramp", "known_da_vs_yesterday",
        "known_da_solar_interaction", "known_da_power_interaction",
    ]
    return base.feature_columns(target) + known_da


def make_model(name: str) -> Any:
    if name == "ridge":
        return base.NumpyRidge(alpha=10.0)
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=350,
            min_samples_leaf=3,
            max_features=0.8,
            n_jobs=-1,
            random_state=42,
        )
    if name.startswith("lightgbm"):
        if not base.HAS_LIGHTGBM:
            raise RuntimeError("LightGBM is unavailable; repair its DLL installation before running this model")
        return base.LGBMRegressor(
            objective="regression_l1",
            n_estimators=500,
            learning_rate=0.025,
            num_leaves=25,
            max_depth=-1,
            min_child_samples=24,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=3.0,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )
    if name.startswith("xgboost"):
        if not base.HAS_XGBOOST:
            raise RuntimeError("XGBoost is unavailable")
        objective = "reg:absoluteerror" if name.endswith("l1") else "reg:squarederror"
        return base.XGBRegressor(
            objective=objective,
            n_estimators=500,
            max_depth=5,
            learning_rate=0.025,
            min_child_weight=8,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=3.0,
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"unknown model: {name}")


def fit_predict(
    frame: pd.DataFrame,
    target: str,
    train_mask: pd.Series,
    predict_mask: pd.Series,
    model_name: str,
) -> np.ndarray:
    columns = feature_columns(target)
    values, _ = base.clean_matrix(frame, columns, train_mask)
    valid = train_mask.to_numpy(bool) & frame[target].notna().to_numpy(bool)
    train_x = pd.DataFrame(values[valid], columns=columns)
    predict_x = pd.DataFrame(values[predict_mask.to_numpy(bool)], columns=columns)
    model = make_model(model_name)
    model.fit(train_x, frame.loc[valid, target].to_numpy(float))
    return np.asarray(model.predict(predict_x), dtype=float)


def simplex_weights(actual: np.ndarray, predictions: np.ndarray, min_samples: int = 24) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predictions).all(axis=1)
    if int(valid.sum()) < min_samples:
        return np.full(predictions.shape[1], 1.0 / predictions.shape[1])
    y = actual[valid]
    x = predictions[valid]

    def objective(weights: np.ndarray) -> float:
        return float(np.mean(np.abs(x @ weights - y)))

    initial = np.full(x.shape[1], 1.0 / x.shape[1])
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * x.shape[1],
        constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success or not np.isfinite(result.x).all():
        return initial
    weights = np.clip(result.x, 0.0, 1.0)
    return weights / weights.sum()


def segment_name(period: int) -> str:
    for name, periods in SEGMENTS.items():
        if period in periods:
            return name
    raise ValueError(f"invalid period: {period}")


def generate_oof_predictions(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    complete = frame["weather_complete"] & frame["power_complete"]
    market_date = frame["market_date"]
    dates = sorted(pd.to_datetime(frame.loc[complete, "market_date"].unique()))
    dates = [day for day in dates if start <= day <= end]
    rows: list[dict[str, Any]] = []
    for day_index, day in enumerate(dates, start=1):
        train_mask = (market_date < day) & complete
        predict_mask = market_date.eq(day) & complete
        if int(predict_mask.sum()) != 24:
            continue
        predictions: dict[str, np.ndarray] = {}
        for target in ("spread", "rt"):
            for model_name in MODEL_NAMES:
                predictions[f"{target}_{model_name}"] = fit_predict(
                    frame, target, train_mask, predict_mask, model_name
                )
        day_rows = frame.loc[predict_mask].sort_values("period").reset_index(drop=True)
        for index, row in day_rows.iterrows():
            output: dict[str, Any] = {
                "market_date": day.date().isoformat(),
                "period": int(row["period"]),
                "segment": segment_name(int(row["period"])),
                "da_actual": float(row["da"]),
                "spread_actual": float(row["spread"]),
                "rt_actual": float(row["rt"]),
            }
            for model_name in MODEL_NAMES:
                output[f"spread_{model_name}_pred"] = float(predictions[f"spread_{model_name}"][index])
                output[f"rt_from_spread_{model_name}_pred"] = (
                    output["da_actual"] + output[f"spread_{model_name}_pred"]
                )
                output[f"rt_direct_{model_name}_pred"] = float(predictions[f"rt_{model_name}"][index])
            rows.append(output)
        print(f"completed post-DA fold {day_index}/{len(dates)}: {day.date().isoformat()}", flush=True)
    return pd.DataFrame(rows)


def rolling_meta_ensemble(
    predictions: pd.DataFrame,
    test_start: pd.Timestamp,
    calibration_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = predictions.copy()
    result["market_date"] = pd.to_datetime(result["market_date"])
    candidate_columns = [
        *[f"rt_from_spread_{name}_pred" for name in MODEL_NAMES],
        *[f"rt_direct_{name}_pred" for name in MODEL_NAMES if name != "ridge"],
    ]
    test_days = sorted(result.loc[result["market_date"].ge(test_start), "market_date"].unique())
    weight_rows: list[dict[str, Any]] = []
    output_rows: list[pd.DataFrame] = []
    for day in test_days:
        day = pd.Timestamp(day)
        calibration_start = day - pd.Timedelta(days=calibration_days)
        calibration = result.loc[
            result["market_date"].ge(calibration_start) & result["market_date"].lt(day)
        ]
        target = result.loc[result["market_date"].eq(day)].copy()
        global_weights = simplex_weights(
            calibration["rt_actual"].to_numpy(float),
            calibration[candidate_columns].to_numpy(float),
        )
        target["rt_global_ensemble_pred"] = target[candidate_columns].to_numpy(float) @ global_weights
        global_bias = float(np.median(
            calibration["rt_actual"].to_numpy(float)
            - calibration[candidate_columns].to_numpy(float) @ global_weights
        ))
        target["rt_global_bias_corrected_pred"] = target["rt_global_ensemble_pred"] + 0.5 * global_bias
        weight_rows.append({
            "market_date": day.date().isoformat(),
            "segment": "global",
            "calibration_mae": metric(
                calibration["rt_actual"], calibration[candidate_columns].to_numpy(float) @ global_weights
            )["mae_yuan_per_mwh"],
            "bias": global_bias,
            **{f"weight_{column}": float(weight) for column, weight in zip(candidate_columns, global_weights)},
        })
        target["rt_segment_ensemble_pred"] = np.nan
        target["rt_segment_bias_corrected_pred"] = np.nan
        for segment in SEGMENTS:
            cal_segment = calibration.loc[calibration["segment"].eq(segment)]
            target_segment = target["segment"].eq(segment)
            segment_weights = simplex_weights(
                cal_segment["rt_actual"].to_numpy(float),
                cal_segment[candidate_columns].to_numpy(float),
                min_samples=36,
            )
            cal_pred = cal_segment[candidate_columns].to_numpy(float) @ segment_weights
            target_pred = target.loc[target_segment, candidate_columns].to_numpy(float) @ segment_weights
            segment_bias = float(np.median(cal_segment["rt_actual"].to_numpy(float) - cal_pred))
            target.loc[target_segment, "rt_segment_ensemble_pred"] = target_pred
            target.loc[target_segment, "rt_segment_bias_corrected_pred"] = target_pred + 0.5 * segment_bias
            weight_rows.append({
                "market_date": day.date().isoformat(),
                "segment": segment,
                "calibration_mae": metric(cal_segment["rt_actual"], cal_pred)["mae_yuan_per_mwh"],
                "bias": segment_bias,
                **{f"weight_{column}": float(weight) for column, weight in zip(candidate_columns, segment_weights)},
            })
        output_rows.append(target)
    return pd.concat(output_rows, ignore_index=True), pd.DataFrame(weight_rows)


def summarize(backtest: pd.DataFrame, calibration_days: int) -> dict[str, Any]:
    prediction_columns = [
        *[f"rt_from_spread_{name}_pred" for name in MODEL_NAMES],
        *[f"rt_direct_{name}_pred" for name in MODEL_NAMES],
        "rt_global_ensemble_pred",
        "rt_global_bias_corrected_pred",
        "rt_segment_ensemble_pred",
        "rt_segment_bias_corrected_pred",
    ]
    scores = {column: metric(backtest["rt_actual"], backtest[column]) for column in prediction_columns}
    segment_scores = {
        segment: {
            column: metric(group["rt_actual"], group[column])
            for column in prediction_columns
        }
        for segment, group in backtest.groupby("segment", sort=False)
    }
    return {
        "scenario": "post-day-ahead clearing real-time forecast",
        "known_at_prediction_time": [
            "complete target-day day-ahead clearing price curve",
            "target-day weather forecast",
            "target-day power output forecast",
            "historical DA, RT and RT-minus-DA spread through the previous market day",
        ],
        "models": list(MODEL_NAMES),
        "rolling_meta_window_days": calibration_days,
        "backtest": scores,
        "by_segment": segment_scores,
        "leakage_controls": [
            "Every base-model fold trains only on market dates before its target day.",
            "Target-day DA is used only in the explicitly post-clearing scenario.",
            "Rolling ensemble weights and bias use only the preceding calibration window.",
            "No target-day RT price or spread is used as an input.",
        ],
        "status": "exploratory on a previously inspected test interval; confirm on later unseen dates",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-DA real-time price forecast")
    parser.add_argument("--price", type=Path, default=base.PRICE_DEFAULT)
    parser.add_argument("--weather", type=Path, default=base.WEATHER_DEFAULT)
    parser.add_argument("--power-dir", type=Path, default=base.ROOT)
    parser.add_argument("--calibration-start", default="2026-06-08")
    parser.add_argument("--backtest-start", default="2026-06-15")
    parser.add_argument("--backtest-end", default="2026-06-30")
    parser.add_argument("--calibration-days", type=int, default=7)
    parser.add_argument(
        "--output-dir", type=Path, default=base.ROOT / "outputs" / "realtime_post_da_20260831"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, coverage = base.load_price_weather(
        args.price, args.weather, sorted(args.power_dir.glob(base.POWER_GLOB))
    )
    frame = add_known_da_features(base.add_all_feature_tables(frame))
    predictions = generate_oof_predictions(
        frame, pd.Timestamp(args.calibration_start), pd.Timestamp(args.backtest_end)
    )
    predictions.to_csv(args.output_dir / "all_oof_predictions.csv", index=False, encoding="utf-8-sig")
    backtest, weights = rolling_meta_ensemble(
        predictions, pd.Timestamp(args.backtest_start), args.calibration_days
    )
    backtest.to_csv(args.output_dir / "backtest_predictions.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(args.output_dir / "rolling_weights.csv", index=False, encoding="utf-8-sig")
    summary = summarize(backtest, args.calibration_days)
    summary["period"] = {"start": args.backtest_start, "end": args.backtest_end}
    summary["sample_count"] = int(len(backtest))
    summary["data_coverage"] = coverage
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
