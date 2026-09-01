"""Pre-backtest calibration and statistical evaluation for CNN/LSTM fusion.

Architectures and blend weights are fixed on 2026-06-08 through 2026-06-14.
The reported 2026-06-15 through 2026-06-30 walk-forward predictions are never
used to fit a weight.  Real-time price is always kept coherent as DA + spread.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import integrated_price_forecast as base
from cnn_lstm_price_forecast import TrialConfig, predict as sequence_predict, train_model as train_sequence
from ensemble_price_forecast import fit_component, simplex_weights
from mamba_price_forecast import TrialConfig as MambaConfig
from mamba_price_forecast import predict as mamba_predict
from mamba_price_forecast import train_model as train_mamba
from transformer_price_forecast import (
    TransformerConfig,
    complete_market_days,
    daily_arrays,
    fit_scales,
    predict as transformer_predict,
    train_model as train_transformer,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "cnn_lstm_trial_20260827"
TRANSFORMER_DIR = ROOT / "outputs" / "transformer_trial_20260827"
MAMBA_DIR = ROOT / "outputs" / "mamba_trial_20260827"


def mae(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual, dtype=float) - np.asarray(prediction, dtype=float))))


def fit_weights(actual: np.ndarray, components: dict[str, np.ndarray]) -> dict[str, float]:
    matrix = np.column_stack(list(components.values()))
    weights = simplex_weights(actual, matrix, "mae")
    return dict(zip(components, map(float, weights)))


def combine(components: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(weights[name] * components[name] for name in weights)


def paired_day_bootstrap(
    frame: pd.DataFrame,
    actual_column: str,
    old_column: str,
    new_column: str,
    draws: int = 10_000,
    seed: int = 20260827,
) -> dict[str, Any]:
    work = frame[["market_date", actual_column, old_column, new_column]].copy()
    work["improvement"] = (
        (work[actual_column] - work[old_column]).abs()
        - (work[actual_column] - work[new_column]).abs()
    )
    daily = work.groupby("market_date", sort=True)["improvement"].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(daily, size=(draws, len(daily)), replace=True).mean(axis=1)
    old_score = mae(work[actual_column], work[old_column])
    new_score = mae(work[actual_column], work[new_column])
    improvement = old_score - new_score
    return {
        "old_mae": old_score,
        "new_mae": new_score,
        "mae_improvement": improvement,
        "relative_improvement_pct": 100.0 * improvement / old_score,
        "day_block_bootstrap_95_ci": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "bootstrap_probability_improved": float(np.mean(samples > 0.0)),
        "days_improved": int(np.sum(daily > 0.0)),
        "days_total": int(len(daily)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cnn_lstm_summary = json.loads((OUTPUT_DIR / "cnn_lstm_summary.json").read_text(encoding="utf-8"))
    transformer_summary = json.loads((TRANSFORMER_DIR / "transformer_summary.json").read_text(encoding="utf-8"))
    mamba_summary = json.loads((MAMBA_DIR / "mamba_summary.json").read_text(encoding="utf-8"))
    tune_start = pd.Timestamp(cnn_lstm_summary["tuning_period"]["start"])
    tune_end = pd.Timestamp(cnn_lstm_summary["tuning_period"]["end"])

    frame, _ = base.load_price_weather(base.PRICE_DEFAULT, base.WEATHER_DEFAULT, sorted(base.ROOT.glob(base.POWER_GLOB)))
    frame = base.add_all_feature_tables(frame)
    days = complete_market_days(frame)
    train_days = [day for day in days if day < tune_start]
    validation_days = [day for day in days if tune_start <= day <= tune_end]
    train_past, train_future, train_target, _ = daily_arrays(frame, train_days)
    val_past, val_future, val_target, accepted_days = daily_arrays(frame, validation_days)
    past_scale, future_scale, target_scale = fit_scales(train_past, train_future, train_target)
    train_scaled = (
        past_scale.transform(train_past), future_scale.transform(train_future), target_scale.transform(train_target)
    )
    val_scaled = (
        past_scale.transform(val_past), future_scale.transform(val_future), target_scale.transform(val_target)
    )

    deep_predictions: dict[str, np.ndarray] = {}
    cnn_lstm_configs: dict[str, TrialConfig] = {}
    for index, family in enumerate(("cnn", "lstm")):
        details = cnn_lstm_summary["models"][family]
        config = TrialConfig(**details["selected_config"])
        cnn_lstm_configs[family] = config
        model, _, _ = train_sequence(
            *train_scaled,
            config=config,
            epochs=int(details["selected_epochs"]),
            seed=12400 + index,
            validation=val_scaled,
        )
        deep_predictions[family] = target_scale.inverse(sequence_predict(model, val_scaled[0], val_scaled[1]))

    transformer_config = TransformerConfig(**transformer_summary["selected_config"])
    transformer, _, _ = train_transformer(
        *train_scaled, config=transformer_config, epochs=80, seed=2401, validation=val_scaled
    )
    deep_predictions["transformer"] = target_scale.inverse(
        transformer_predict(transformer, val_scaled[0], val_scaled[1])
    )

    mamba_config = MambaConfig(**mamba_summary["selected_config"])
    mamba, _, _ = train_mamba(*train_scaled, config=mamba_config, epochs=80, seed=3402, validation=val_scaled)
    deep_predictions["mamba"] = target_scale.inverse(mamba_predict(mamba, val_scaled[0], val_scaled[1]))

    date = frame["market_date"]
    complete = frame["weather_complete"] & frame["power_complete"]
    train_mask = (date < tune_start) & complete
    validation_mask = (date >= tune_start) & (date <= tune_end) & complete
    actual_da = frame.loc[validation_mask, "da"].to_numpy(float)
    actual_spread = frame.loc[validation_mask, "spread"].to_numpy(float)
    da_components = {
        "xgboost": fit_component(frame, "da", train_mask, validation_mask, "xgboost"),
        **{name: values[:, :, 0].reshape(-1) for name, values in deep_predictions.items()},
    }
    spread_components = {
        "ridge": fit_component(frame, "spread", train_mask, validation_mask, "ridge"),
        **{name: values[:, :, 1].reshape(-1) for name, values in deep_predictions.items()},
    }

    da_sets = {
        "xgboost_only": ("xgboost",),
        "xgboost_cnn": ("xgboost", "cnn"),
        "xgboost_lstm": ("xgboost", "lstm"),
        "all_five": ("xgboost", "transformer", "mamba", "cnn", "lstm"),
    }
    spread_sets = {
        "ridge_only": ("ridge",),
        "ridge_cnn": ("ridge", "cnn"),
        "ridge_lstm": ("ridge", "lstm"),
        "ridge_cnn_lstm": ("ridge", "cnn", "lstm"),
        "existing_three": ("ridge", "transformer", "mamba"),
        "all_five": ("ridge", "transformer", "mamba", "cnn", "lstm"),
    }

    da_calibration: dict[str, Any] = {}
    for candidate, names in da_sets.items():
        components = {name: da_components[name] for name in names}
        weights = fit_weights(actual_da, components)
        prediction = combine(components, weights)
        da_calibration[candidate] = {"weights": weights, "metrics": base.metric(actual_da, prediction)}

    spread_calibration: dict[str, Any] = {}
    for candidate, names in spread_sets.items():
        components = {name: spread_components[name] for name in names}
        weights = fit_weights(actual_spread, components)
        prediction = combine(components, weights)
        spread_calibration[candidate] = {"weights": weights, "metrics": base.metric(actual_spread, prediction)}

    selected_da = min(da_calibration, key=lambda name: da_calibration[name]["metrics"]["mae_yuan_per_mwh"])
    selected_spread = min(
        spread_calibration, key=lambda name: spread_calibration[name]["metrics"]["mae_yuan_per_mwh"]
    )

    test = pd.read_csv(OUTPUT_DIR / "cnn_lstm_blend_predictions.csv")
    test_components_da = {
        "xgboost": test["da_xgboost_pred"].to_numpy(float),
        "transformer": test["da_transformer_pred"].to_numpy(float),
        "mamba": test["da_mamba_pred"].to_numpy(float),
        "cnn": test["da_cnn_pred"].to_numpy(float),
        "lstm": test["da_lstm_pred"].to_numpy(float),
    }
    test_components_spread = {
        "ridge": test["spread_ridge_pred"].to_numpy(float),
        "transformer": test["spread_transformer_pred"].to_numpy(float),
        "mamba": test["spread_mamba_pred"].to_numpy(float),
        "cnn": test["spread_cnn_pred"].to_numpy(float),
        "lstm": test["spread_lstm_pred"].to_numpy(float),
    }
    for candidate, details in da_calibration.items():
        test[f"da_calibrated_{candidate}_pred"] = combine(test_components_da, details["weights"])
    for candidate, details in spread_calibration.items():
        test[f"spread_calibrated_{candidate}_pred"] = combine(test_components_spread, details["weights"])
    test["da_selected_pred"] = test[f"da_calibrated_{selected_da}_pred"]
    test["spread_selected_pred"] = test[f"spread_calibrated_{selected_spread}_pred"]
    test["rt_selected_pred"] = test["da_selected_pred"] + test["spread_selected_pred"]

    transformer_test = pd.read_csv(TRANSFORMER_DIR / "precalibrated_transformer_blend_predictions.csv")
    mamba_test = pd.read_csv(MAMBA_DIR / "precalibrated_mamba_blend_predictions.csv")
    keys = ["market_date", "period"]
    test = test.merge(
        transformer_test[keys + ["spread_precalibrated_pred", "rt_precalibrated_pred"]],
        on=keys,
        validate="one_to_one",
    )
    test = test.merge(
        mamba_test[keys + ["spread_precalibrated_four_model_pred", "rt_xgb_plus_precalibrated_spread_pred"]],
        on=keys,
        validate="one_to_one",
    )
    test.to_csv(OUTPUT_DIR / "precalibrated_cnn_lstm_fusion_predictions.csv", index=False, encoding="utf-8-sig")

    da_test_results = {
        candidate: base.metric(test["da_actual"], test[f"da_calibrated_{candidate}_pred"])
        for candidate in da_calibration
    }
    spread_test_results = {
        candidate: base.metric(test["spread_actual"], test[f"spread_calibrated_{candidate}_pred"])
        for candidate in spread_calibration
    }
    selected_results = {
        "day_ahead": base.metric(test["da_actual"], test["da_selected_pred"]),
        "spread": base.metric(test["spread_actual"], test["spread_selected_pred"]),
        "real_time_coherent": base.metric(test["rt_actual"], test["rt_selected_pred"]),
        "spread_direction_accuracy": float(
            ((test["spread_selected_pred"] >= 0) == (test["spread_actual"] >= 0)).mean()
        ),
    }
    statistical = {
        "selected_spread_vs_ridge": paired_day_bootstrap(
            test, "spread_actual", "spread_ridge_pred", "spread_selected_pred", seed=20260827
        ),
        "selected_spread_vs_transformer_fusion": paired_day_bootstrap(
            test, "spread_actual", "spread_precalibrated_pred", "spread_selected_pred", seed=20260828
        ),
        "selected_spread_vs_mamba_fusion": paired_day_bootstrap(
            test,
            "spread_actual",
            "spread_precalibrated_four_model_pred",
            "spread_selected_pred",
            seed=20260829,
        ),
        "selected_rt_vs_transformer_fusion": paired_day_bootstrap(
            test, "rt_actual", "rt_precalibrated_pred", "rt_selected_pred", seed=20260830
        ),
        "selected_rt_vs_mamba_fusion": paired_day_bootstrap(
            test,
            "rt_actual",
            "rt_xgb_plus_precalibrated_spread_pred",
            "rt_selected_pred",
            seed=20260831,
        ),
    }
    output = {
        "calibration_period": {
            "start": tune_start.date().isoformat(),
            "end": tune_end.date().isoformat(),
            "days": [day.date().isoformat() for day in accepted_days],
        },
        "architecture_note": (
            "CNN/LSTM architectures and blend weights use the same pre-backtest calibration week; "
            "the reported 16-day test remains untouched, but the short calibration window limits weight stability."
        ),
        "calibration": {"day_ahead": da_calibration, "spread": spread_calibration},
        "selected_by_calibration": {"day_ahead": selected_da, "spread": selected_spread},
        "reported_backtest": {
            "all_day_ahead_candidates": da_test_results,
            "all_spread_candidates": spread_test_results,
            "selected": selected_results,
        },
        "statistical_comparison": statistical,
        "leakage_control": (
            "All component weights were fixed on June 8-14 before evaluation on June 15-30. "
            "Each component test prediction is from the same strict daily walk-forward protocol."
        ),
    }
    (OUTPUT_DIR / "precalibrated_cnn_lstm_fusion_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
