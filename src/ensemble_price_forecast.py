"""Leakage-safe ensemble search for the Shandong price pipeline.

The script evaluates LightGBM, XGBoost and a standardized ridge model under
the same date-based walk-forward protocol. For each target day, blend weights
are learned only on the preceding calibration window, then the component
models are refit on all data before the target day. This makes the ensemble
comparison usable for a pre-market forecast rather than an in-sample blend.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import integrated_price_forecast as base


COMPONENTS = ("lightgbm", "xgboost", "ridge")


def fit_component(
    frame: pd.DataFrame,
    target_name: str,
    train_mask: pd.Series,
    predict_mask: pd.Series,
    component: str,
) -> np.ndarray:
    target = frame[target_name].astype(float)
    columns = base.feature_columns(target_name)
    x, _ = base.clean_matrix(frame, columns, train_mask)
    valid = train_mask.to_numpy(bool) & target.notna().to_numpy(bool)
    train_x = pd.DataFrame(x[valid], columns=columns)
    predict_x = pd.DataFrame(x[predict_mask.to_numpy(bool)], columns=columns)
    if component == "ridge":
        model = base.NumpyRidge(alpha=10.0)
    else:
        model, _ = base.make_regressor(backend_preference=component)
    model.fit(train_x, target.to_numpy(float)[valid])
    return np.asarray(model.predict(predict_x), dtype=float)


def simplex_weights(actual: np.ndarray, predictions: np.ndarray, objective: str = "mae") -> np.ndarray:
    """Fit nonnegative weights summing to one, with an equal-weight fallback."""
    actual = np.asarray(actual, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predictions).all(axis=1)
    if valid.sum() < 24:
        return np.full(predictions.shape[1], 1.0 / predictions.shape[1])
    y = actual[valid]
    p = predictions[valid]

    def loss(weights: np.ndarray) -> float:
        error = p @ weights - y
        if objective == "mae":
            return float(np.mean(np.abs(error)))
        return float(np.mean(error**2))

    result = minimize(
        loss,
        x0=np.full(p.shape[1], 1.0 / p.shape[1]),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * p.shape[1],
        constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        options={"maxiter": 300, "ftol": 1e-9},
    )
    if not result.success or not np.isfinite(result.x).all():
        return np.full(p.shape[1], 1.0 / p.shape[1])
    weights = np.clip(result.x, 0.0, 1.0)
    return weights / weights.sum()


def pred_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return base.metric(np.asarray(actual, dtype=float), np.asarray(prediction, dtype=float))


def run_search(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    calibration_days: int,
    objective: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    complete = frame["weather_complete"] & frame["power_complete"]
    date = frame["market_date"]
    complete_days = sorted(pd.to_datetime(frame.loc[complete, "market_date"].unique()))
    dates = [
        day for day in complete_days
        if start <= day <= end
        and frame.loc[date.eq(day), "da"].notna().all()
        and frame.loc[date.eq(day), "rt"].notna().all()
    ]
    rows: list[dict[str, Any]] = []
    weights_rows: list[dict[str, Any]] = []

    for day in dates:
        calibration_start = day - pd.Timedelta(days=calibration_days)
        cal_mask = (date >= calibration_start) & (date < day) & complete
        cal_train_mask = (date < calibration_start) & complete
        train_mask = (date < day) & complete
        predict_mask = date.eq(day) & complete
        target_predictions: dict[str, dict[str, np.ndarray]] = {}
        target_weights: dict[str, np.ndarray] = {}

        for target in base.TARGETS:
            target_predictions[target] = {}
            for component in COMPONENTS:
                cal_pred = fit_component(frame, target, cal_train_mask, cal_mask, component)
                final_pred = fit_component(frame, target, train_mask, predict_mask, component)
                target_predictions[target][component] = final_pred
                target_predictions[target][f"{component}_cal"] = cal_pred
            actual_cal = frame.loc[cal_mask, target].to_numpy(float)
            cal_matrix = np.column_stack([
                target_predictions[target][f"{component}_cal"] for component in COMPONENTS
            ])
            weights = simplex_weights(actual_cal, cal_matrix, objective)
            target_weights[target] = weights
            weights_rows.append(
                {
                    "market_date": day.date().isoformat(),
                    "target": target,
                    **{f"weight_{component}": float(weight) for component, weight in zip(COMPONENTS, weights)},
                    "calibration_mae": pred_metrics(actual_cal, cal_matrix @ weights)["mae_yuan_per_mwh"],
                    "calibration_samples": int(len(actual_cal)),
                }
            )

        target_rows = frame.loc[predict_mask].sort_values("period").reset_index(drop=True)
        for index, row in target_rows.iterrows():
            output: dict[str, Any] = {
                "market_date": day.date().isoformat(),
                "period": int(row["period"]),
            }
            for target in base.TARGETS:
                for component in COMPONENTS:
                    output[f"{target}_{component}_pred"] = float(target_predictions[target][component][index])
                weights = target_weights[target]
                output[f"{target}_ensemble_pred"] = float(
                    sum(weights[i] * target_predictions[target][component][index] for i, component in enumerate(COMPONENTS))
                )
            output["rt_coherent_ensemble_pred"] = output["da_ensemble_pred"] + output["spread_ensemble_pred"]
            output["da_actual"] = float(row["da"])
            output["rt_actual"] = float(row["rt"])
            output["spread_actual"] = float(row["spread"])
            rows.append(output)

    result = pd.DataFrame(rows)
    weights_frame = pd.DataFrame(weights_rows)
    scores: dict[str, Any] = {}
    if not result.empty:
        score_targets = {
            "day_ahead": ("da_actual", "da_ensemble_pred"),
            "real_time_direct": ("rt_actual", "rt_ensemble_pred"),
            "real_time_coherent": ("rt_actual", "rt_coherent_ensemble_pred"),
            "spread": ("spread_actual", "spread_ensemble_pred"),
        }
        for name, (actual_column, _) in score_targets.items():
            actual = result[actual_column].to_numpy(float)
            candidates = {}
            for component in COMPONENTS:
                pred_column = {
                    "da_actual": f"da_{component}_pred",
                    "rt_actual": f"rt_{component}_pred",
                    "spread_actual": f"spread_{component}_pred",
                }[actual_column]
                candidates[component] = pred_metrics(actual, result[pred_column].to_numpy(float))
            ensemble_column = score_targets[name][1]
            candidates["adaptive_ensemble"] = pred_metrics(actual, result[ensemble_column].to_numpy(float))
            equal_column = f"{actual_column.split('_')[0]}_equal_pred"
            equal_prediction = result[[f"{actual_column.split('_')[0]}_{component}_pred" for component in COMPONENTS]].mean(axis=1)
            candidates["equal_ensemble"] = pred_metrics(actual, equal_prediction.to_numpy(float))
            scores[name] = candidates
        result["spread_direction_correct"] = (
            (result["spread_ensemble_pred"] >= 0) == (result["spread_actual"] >= 0)
        ).astype(int)
        scores["spread_direction_accuracy"] = float(result["spread_direction_correct"].mean())

    metadata = {
        "components": list(COMPONENTS),
        "weight_method": f"nonnegative simplex optimization on prior {calibration_days}-day calibration window",
        "weight_objective": objective,
        "dates": [day.date().isoformat() for day in dates],
        "sample_count": int(len(result)),
        "data_assumptions": [
            "Target-day weather and power values are treated as pre-market inputs.",
            "Weights are learned before each target date; target-day actual prices are never used.",
            "RT coherent output is DA ensemble plus spread ensemble.",
        ],
    }
    return result, scores, {"weights": weights_frame, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward LightGBM/XGBoost/Ridge ensemble search")
    parser.add_argument("--price", type=Path, default=base.PRICE_DEFAULT)
    parser.add_argument("--weather", type=Path, default=base.WEATHER_DEFAULT)
    parser.add_argument("--power-dir", type=Path, default=base.ROOT)
    parser.add_argument("--backtest-start", default="2026-06-15")
    parser.add_argument("--backtest-end", default="2026-06-30")
    parser.add_argument("--calibration-days", type=int, default=7)
    parser.add_argument("--objective", choices=["mae", "rmse"], default="mae")
    parser.add_argument("--output-dir", type=Path, default=base.ROOT / "outputs" / "ensemble_search_20260826")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.power_dir.glob(base.POWER_GLOB))
    frame, coverage = base.load_price_weather(args.price, args.weather, paths)
    frame = base.add_all_feature_tables(frame)
    result, scores, details = run_search(
        frame,
        pd.Timestamp(args.backtest_start),
        pd.Timestamp(args.backtest_end),
        args.calibration_days,
        args.objective,
    )
    result.to_csv(args.output_dir / "walk_forward_predictions.csv", index=False, encoding="utf-8-sig")
    details["weights"].to_csv(args.output_dir / "adaptive_weights.csv", index=False, encoding="utf-8-sig")
    summary = {
        "output_dir": str(args.output_dir),
        "backtest": scores,
        "coverage": coverage,
        **details["metadata"],
    }
    (args.output_dir / "ensemble_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
