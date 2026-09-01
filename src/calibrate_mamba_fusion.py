"""Pre-backtest calibration of XGBoost/Ridge/Transformer/Mamba blends."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import integrated_price_forecast as base
from ensemble_price_forecast import fit_component, simplex_weights
from mamba_price_forecast import TrialConfig, predict as mamba_predict, train_model as train_mamba
from transformer_price_forecast import (
    TransformerConfig,
    complete_market_days,
    daily_arrays,
    fit_scales,
    predict as transformer_predict,
    train_model as train_transformer,
)


ROOT = Path(__file__).resolve().parent
MAMBA_DIR = ROOT / "outputs" / "mamba_trial_20260827"
TRANSFORMER_DIR = ROOT / "outputs" / "transformer_trial_20260827"


def main() -> None:
    mamba_summary = json.loads((MAMBA_DIR / "mamba_summary.json").read_text(encoding="utf-8"))
    transformer_summary = json.loads((TRANSFORMER_DIR / "transformer_summary.json").read_text(encoding="utf-8"))
    tune_start = pd.Timestamp(mamba_summary["tuning_period"]["start"])
    tune_end = pd.Timestamp(mamba_summary["tuning_period"]["end"])
    mamba_config = TrialConfig(**mamba_summary["selected_config"])
    transformer_config = TransformerConfig(**transformer_summary["selected_config"])

    frame, _ = base.load_price_weather(base.PRICE_DEFAULT, base.WEATHER_DEFAULT, sorted(base.ROOT.glob(base.POWER_GLOB)))
    frame = base.add_all_feature_tables(frame)
    days = complete_market_days(frame)
    train_days = [day for day in days if day < tune_start]
    validation_days = [day for day in days if tune_start <= day <= tune_end]
    train_past, train_future, train_target, _ = daily_arrays(frame, train_days)
    val_past, val_future, val_target, accepted = daily_arrays(frame, validation_days)
    past_scale, future_scale, target_scale = fit_scales(train_past, train_future, train_target)
    train_scaled = (
        past_scale.transform(train_past), future_scale.transform(train_future), target_scale.transform(train_target)
    )
    val_scaled = (
        past_scale.transform(val_past), future_scale.transform(val_future), target_scale.transform(val_target)
    )
    mamba, _, _ = train_mamba(
        *train_scaled, config=mamba_config, epochs=80, seed=3402, validation=val_scaled
    )
    transformer, _, _ = train_transformer(
        *train_scaled, config=transformer_config, epochs=80, seed=2401, validation=val_scaled
    )
    mamba_val = target_scale.inverse(mamba_predict(mamba, val_scaled[0], val_scaled[1]))
    transformer_val = target_scale.inverse(transformer_predict(transformer, val_scaled[0], val_scaled[1]))

    date = frame["market_date"]
    complete = frame["weather_complete"] & frame["power_complete"]
    train_mask = (date < tune_start) & complete
    validation_mask = (date >= tune_start) & (date <= tune_end) & complete
    xgb_da = fit_component(frame, "da", train_mask, validation_mask, "xgboost")
    ridge_spread = fit_component(frame, "spread", train_mask, validation_mask, "ridge")
    actual_da = frame.loc[validation_mask, "da"].to_numpy(float)
    actual_spread = frame.loc[validation_mask, "spread"].to_numpy(float)
    da_calibration = np.column_stack(
        [xgb_da, transformer_val[:, :, 0].reshape(-1), mamba_val[:, :, 0].reshape(-1)]
    )
    spread_calibration = np.column_stack(
        [ridge_spread, transformer_val[:, :, 1].reshape(-1), mamba_val[:, :, 1].reshape(-1)]
    )
    da_weights = simplex_weights(actual_da, da_calibration, "mae")
    spread_weights = simplex_weights(actual_spread, spread_calibration, "mae")
    actual_rt = actual_da + actual_spread

    def joint_loss(weights: np.ndarray) -> float:
        spread_prediction = spread_calibration @ weights
        rt_prediction = xgb_da + spread_prediction
        return float(
            0.5 * np.mean(np.abs(actual_spread - spread_prediction))
            + 0.5 * np.mean(np.abs(actual_rt - rt_prediction))
        )

    joint_result = minimize(
        joint_loss,
        x0=np.full(3, 1.0 / 3.0),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 3,
        constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        options={"maxiter": 300, "ftol": 1e-9},
    )
    joint_spread_weights = np.clip(joint_result.x, 0.0, 1.0)
    joint_spread_weights /= joint_spread_weights.sum()

    test = pd.read_csv(MAMBA_DIR / "mamba_blend_predictions.csv")
    test["da_precalibrated_four_model_pred"] = (
        da_weights[0] * test["da_xgboost_pred"]
        + da_weights[1] * test["da_transformer_pred"]
        + da_weights[2] * test["da_mamba_pred"]
    )
    test["spread_precalibrated_four_model_pred"] = (
        spread_weights[0] * test["spread_ridge_pred"]
        + spread_weights[1] * test["spread_transformer_pred"]
        + spread_weights[2] * test["spread_mamba_pred"]
    )
    test["rt_precalibrated_four_model_pred"] = (
        test["da_precalibrated_four_model_pred"] + test["spread_precalibrated_four_model_pred"]
    )
    test["rt_xgb_plus_precalibrated_spread_pred"] = (
        test["da_xgboost_pred"] + test["spread_precalibrated_four_model_pred"]
    )
    test["spread_precalibrated_joint_pred"] = (
        joint_spread_weights[0] * test["spread_ridge_pred"]
        + joint_spread_weights[1] * test["spread_transformer_pred"]
        + joint_spread_weights[2] * test["spread_mamba_pred"]
    )
    test["rt_precalibrated_joint_pred"] = test["da_xgboost_pred"] + test["spread_precalibrated_joint_pred"]
    test.to_csv(MAMBA_DIR / "precalibrated_mamba_blend_predictions.csv", index=False, encoding="utf-8-sig")
    output = {
        "calibration_period": {
            "start": tune_start.date().isoformat(), "end": tune_end.date().isoformat(),
            "days": [day.date().isoformat() for day in accepted],
        },
        "weights": {
            "day_ahead": dict(zip(["xgboost", "transformer", "mamba"], map(float, da_weights))),
            "spread": dict(zip(["ridge", "transformer", "mamba"], map(float, spread_weights))),
            "spread_joint_objective": dict(
                zip(["ridge", "transformer", "mamba"], map(float, joint_spread_weights))
            ),
        },
        "calibration_metrics": {
            "day_ahead": base.metric(actual_da, da_calibration @ da_weights),
            "spread": base.metric(actual_spread, spread_calibration @ spread_weights),
        },
        "reported_backtest": {
            "day_ahead_four_model": base.metric(test["da_actual"], test["da_precalibrated_four_model_pred"]),
            "spread_four_model": base.metric(test["spread_actual"], test["spread_precalibrated_four_model_pred"]),
            "spread_joint_objective": base.metric(test["spread_actual"], test["spread_precalibrated_joint_pred"]),
            "real_time_four_model": base.metric(test["rt_actual"], test["rt_precalibrated_four_model_pred"]),
            "real_time_xgboost_plus_spread": base.metric(test["rt_actual"], test["rt_xgb_plus_precalibrated_spread_pred"]),
            "real_time_joint_objective": base.metric(test["rt_actual"], test["rt_precalibrated_joint_pred"]),
            "spread_direction_accuracy": float(
                ((test["spread_precalibrated_four_model_pred"] >= 0) == (test["spread_actual"] >= 0)).mean()
            ),
        },
        "leakage_control": "All weights were fitted on June 8-14 before the June 15-30 reported backtest.",
    }
    (MAMBA_DIR / "precalibrated_mamba_blend_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
