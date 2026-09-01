"""Calibrate fixed Transformer/tree blend weights before the reported test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import integrated_price_forecast as base
from ensemble_price_forecast import fit_component
from transformer_price_forecast import (
    TransformerConfig,
    complete_market_days,
    daily_arrays,
    fit_scales,
    predict,
    train_model,
)


def mae(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(prediction))))


def best_weight(actual: np.ndarray, first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    candidates = np.linspace(0.0, 1.0, 101)
    scores = [mae(actual, weight * first + (1.0 - weight) * second) for weight in candidates]
    index = int(np.argmin(scores))
    return float(candidates[index]), float(scores[index])


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-backtest Transformer blend calibration")
    parser.add_argument("--transformer-dir", type=Path, default=base.ROOT / "outputs" / "transformer_trial_20260827")
    parser.add_argument("--ensemble-predictions", type=Path, default=base.ROOT / "outputs" / "ensemble_search_20260826" / "walk_forward_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=base.ROOT / "outputs" / "transformer_trial_20260827")
    args = parser.parse_args()
    summary = json.loads((args.transformer_dir / "transformer_summary.json").read_text(encoding="utf-8"))
    tune_start = pd.Timestamp(summary["tuning_period"]["start"])
    tune_end = pd.Timestamp(summary["tuning_period"]["end"])
    selected = summary["selected_config"]
    config = TransformerConfig(**selected)
    candidate = next(item for item in summary["tuning_period"]["candidates"] if item["config"]["name"] == config.name)

    frame, _ = base.load_price_weather(base.PRICE_DEFAULT, base.WEATHER_DEFAULT, sorted(base.ROOT.glob(base.POWER_GLOB)))
    frame = base.add_all_feature_tables(frame)
    all_days = complete_market_days(frame)
    train_days = [day for day in all_days if day < tune_start]
    validation_days = [day for day in all_days if tune_start <= day <= tune_end]
    train_past, train_future, train_target, _ = daily_arrays(frame, train_days)
    val_past, val_future, val_target, val_days = daily_arrays(frame, validation_days)
    past_scale, future_scale, target_scale = fit_scales(train_past, train_future, train_target)
    train_scaled = (
        past_scale.transform(train_past), future_scale.transform(train_future), target_scale.transform(train_target)
    )
    val_scaled = (
        past_scale.transform(val_past), future_scale.transform(val_future), target_scale.transform(val_target)
    )
    model, _, _ = train_model(
        *train_scaled,
        config=config,
        epochs=80,
        seed=2401,
        validation=val_scaled,
    )
    transformer_val = target_scale.inverse(predict(model, val_scaled[0], val_scaled[1]))
    transformer_da = transformer_val[:, :, 0].reshape(-1)
    transformer_spread = transformer_val[:, :, 1].reshape(-1)

    date = frame["market_date"]
    complete = frame["weather_complete"] & frame["power_complete"]
    train_mask = (date < tune_start) & complete
    validation_mask = (date >= tune_start) & (date <= tune_end) & complete
    xgb_da = fit_component(frame, "da", train_mask, validation_mask, "xgboost")
    ridge_spread = fit_component(frame, "spread", train_mask, validation_mask, "ridge")
    actual_da = frame.loc[validation_mask, "da"].to_numpy(float)
    actual_spread = frame.loc[validation_mask, "spread"].to_numpy(float)
    actual_rt = actual_da + actual_spread

    validation_index = frame.loc[validation_mask, ["market_date", "period"]].reset_index(drop=True)
    validation_predictions = validation_index.assign(
        da_actual=actual_da,
        spread_actual=actual_spread,
        rt_actual=actual_rt,
        da_xgboost_pred=xgb_da,
        da_transformer_pred=transformer_da,
        spread_ridge_pred=ridge_spread,
        spread_transformer_pred=transformer_spread,
    )

    weight_da, tune_da_mae = best_weight(actual_da, xgb_da, transformer_da)
    weight_spread, tune_spread_mae = best_weight(actual_spread, ridge_spread, transformer_spread)
    weights = np.linspace(0.0, 1.0, 101)
    best_joint: tuple[float, float, float] | None = None
    for da_weight in weights:
        da_prediction = da_weight * xgb_da + (1.0 - da_weight) * transformer_da
        for spread_weight in weights:
            rt_prediction = da_prediction + spread_weight * ridge_spread + (1.0 - spread_weight) * transformer_spread
            score = mae(actual_rt, rt_prediction)
            if best_joint is None or score < best_joint[0]:
                best_joint = (score, float(da_weight), float(spread_weight))
    assert best_joint is not None

    validation_predictions["da_precalibrated_pred"] = (
        weight_da * validation_predictions["da_xgboost_pred"]
        + (1.0 - weight_da) * validation_predictions["da_transformer_pred"]
    )
    validation_predictions["spread_precalibrated_pred"] = (
        weight_spread * validation_predictions["spread_ridge_pred"]
        + (1.0 - weight_spread) * validation_predictions["spread_transformer_pred"]
    )
    validation_predictions["rt_precalibrated_pred"] = (
        validation_predictions["da_precalibrated_pred"]
        + validation_predictions["spread_precalibrated_pred"]
    )
    _, joint_da_weight, joint_spread_weight = best_joint
    validation_predictions["rt_joint_precalibrated_pred"] = (
        joint_da_weight * validation_predictions["da_xgboost_pred"]
        + (1.0 - joint_da_weight) * validation_predictions["da_transformer_pred"]
        + joint_spread_weight * validation_predictions["spread_ridge_pred"]
        + (1.0 - joint_spread_weight) * validation_predictions["spread_transformer_pred"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_predictions.to_csv(
        args.output_dir / "precalibrated_transformer_validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test = pd.read_csv(args.transformer_dir / "transformer_blend_predictions.csv")
    test["da_precalibrated_pred"] = weight_da * test["da_xgboost_pred"] + (1 - weight_da) * test["da_transformer_pred"]
    test["spread_precalibrated_pred"] = weight_spread * test["spread_ridge_pred"] + (1 - weight_spread) * test["spread_transformer_pred"]
    test["rt_precalibrated_pred"] = test["da_precalibrated_pred"] + test["spread_precalibrated_pred"]
    test["rt_joint_precalibrated_pred"] = (
        joint_da_weight * test["da_xgboost_pred"] + (1 - joint_da_weight) * test["da_transformer_pred"]
        + joint_spread_weight * test["spread_ridge_pred"] + (1 - joint_spread_weight) * test["spread_transformer_pred"]
    )
    test.to_csv(args.output_dir / "precalibrated_transformer_blend_predictions.csv", index=False, encoding="utf-8-sig")
    test_results: dict[str, Any] = {
        "day_ahead": base.metric(test["da_actual"], test["da_precalibrated_pred"]),
        "spread": base.metric(test["spread_actual"], test["spread_precalibrated_pred"]),
        "real_time_from_target_weights": base.metric(test["rt_actual"], test["rt_precalibrated_pred"]),
        "real_time_from_joint_weights": base.metric(test["rt_actual"], test["rt_joint_precalibrated_pred"]),
        "spread_direction_accuracy": float(
            ((test["spread_precalibrated_pred"] >= 0) == (test["spread_actual"] >= 0)).mean()
        ),
    }
    output = {
        "calibration_period": {"start": tune_start.date().isoformat(), "end": tune_end.date().isoformat(), "days": [d.date().isoformat() for d in val_days]},
        "selected_transformer_config": config.__dict__,
        "transformer_tuning_best_epoch": candidate["best_epoch"],
        "weights": {
            "day_ahead": {"xgboost": weight_da, "transformer": 1 - weight_da, "calibration_mae": tune_da_mae},
            "spread": {"ridge": weight_spread, "transformer": 1 - weight_spread, "calibration_mae": tune_spread_mae},
            "joint_real_time": {
                "day_ahead_xgboost": joint_da_weight,
                "day_ahead_transformer": 1 - joint_da_weight,
                "spread_ridge": joint_spread_weight,
                "spread_transformer": 1 - joint_spread_weight,
                "calibration_mae": best_joint[0],
            },
        },
        "reported_backtest": test_results,
        "leakage_control": "All weights were fixed on 2026-06-08 through 2026-06-14 before the 2026-06-15 through 2026-06-30 reported backtest.",
    }
    (args.output_dir / "precalibrated_blend_summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
