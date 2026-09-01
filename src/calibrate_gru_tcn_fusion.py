"""Pre-backtest calibration of GRU/TCN against the existing model ensemble."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import integrated_price_forecast as base
from calibrate_cnn_lstm_fusion import paired_day_bootstrap
from cnn_lstm_price_forecast import TrialConfig as CNNLSTMConfig
from cnn_lstm_price_forecast import predict as cnn_lstm_predict
from cnn_lstm_price_forecast import train_model as train_cnn_lstm
from ensemble_price_forecast import fit_component, simplex_weights
from gru_tcn_price_forecast import TrialConfig as GRUTCNConfig
from gru_tcn_price_forecast import predict as gru_tcn_predict
from gru_tcn_price_forecast import train_model as train_gru_tcn
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
OUTPUT_DIR = ROOT / "outputs" / "gru_tcn_trial_20260827"
CNN_LSTM_DIR = ROOT / "outputs" / "cnn_lstm_trial_20260827"
TRANSFORMER_DIR = ROOT / "outputs" / "transformer_trial_20260827"
MAMBA_DIR = ROOT / "outputs" / "mamba_trial_20260827"


def fit_weights(actual: np.ndarray, components: dict[str, np.ndarray]) -> dict[str, float]:
    weights = simplex_weights(actual, np.column_stack(list(components.values())), "mae")
    return dict(zip(components, map(float, weights)))


def combine(components: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(weights[name] * components[name] for name in weights)


def main() -> None:
    summary = json.loads((OUTPUT_DIR / "gru_tcn_summary.json").read_text(encoding="utf-8"))
    cnn_lstm_summary = json.loads((CNN_LSTM_DIR / "cnn_lstm_summary.json").read_text(encoding="utf-8"))
    transformer_summary = json.loads((TRANSFORMER_DIR / "transformer_summary.json").read_text(encoding="utf-8"))
    mamba_summary = json.loads((MAMBA_DIR / "mamba_summary.json").read_text(encoding="utf-8"))
    tune_start = pd.Timestamp(summary["tuning_period"]["start"])
    tune_end = pd.Timestamp(summary["tuning_period"]["end"])

    frame, _ = base.load_price_weather(base.PRICE_DEFAULT, base.WEATHER_DEFAULT, sorted(base.ROOT.glob(base.POWER_GLOB)))
    frame = base.add_all_feature_tables(frame)
    days = complete_market_days(frame)
    train_days = [day for day in days if day < tune_start]
    validation_days = [day for day in days if tune_start <= day <= tune_end]
    train_past, train_future, train_target, _ = daily_arrays(frame, train_days)
    val_past, val_future, val_target, accepted_days = daily_arrays(frame, validation_days)
    past_scale, future_scale, target_scale = fit_scales(train_past, train_future, train_target)
    train_scaled = (past_scale.transform(train_past), future_scale.transform(train_future), target_scale.transform(train_target))
    val_scaled = (past_scale.transform(val_past), future_scale.transform(val_future), target_scale.transform(val_target))

    deep: dict[str, np.ndarray] = {}
    for index, family in enumerate(("gru", "tcn")):
        details = summary["models"][family]
        config = GRUTCNConfig(**details["selected_config"])
        model, _, _ = train_gru_tcn(*train_scaled, config=config, epochs=int(details["selected_epochs"]), seed=13100 + index, validation=val_scaled)
        deep[family] = target_scale.inverse(gru_tcn_predict(model, val_scaled[0], val_scaled[1]))
    for index, family in enumerate(("cnn", "lstm")):
        details = cnn_lstm_summary["models"][family]
        config = CNNLSTMConfig(**details["selected_config"])
        model, _, _ = train_cnn_lstm(*train_scaled, config=config, epochs=int(details["selected_epochs"]), seed=13200 + index, validation=val_scaled)
        deep[family] = target_scale.inverse(cnn_lstm_predict(model, val_scaled[0], val_scaled[1]))

    transformer_config = TransformerConfig(**transformer_summary["selected_config"])
    transformer, _, _ = train_transformer(*train_scaled, config=transformer_config, epochs=80, seed=2401, validation=val_scaled)
    deep["transformer"] = target_scale.inverse(transformer_predict(transformer, val_scaled[0], val_scaled[1]))
    mamba_config = MambaConfig(**mamba_summary["selected_config"])
    mamba, _, _ = train_mamba(*train_scaled, config=mamba_config, epochs=80, seed=3402, validation=val_scaled)
    deep["mamba"] = target_scale.inverse(mamba_predict(mamba, val_scaled[0], val_scaled[1]))

    date = frame["market_date"]
    complete = frame["weather_complete"] & frame["power_complete"]
    train_mask = (date < tune_start) & complete
    val_mask = (date >= tune_start) & (date <= tune_end) & complete
    actual_da = frame.loc[val_mask, "da"].to_numpy(float)
    actual_spread = frame.loc[val_mask, "spread"].to_numpy(float)
    da_components = {"xgboost": fit_component(frame, "da", train_mask, val_mask, "xgboost"), **{name: values[:, :, 0].reshape(-1) for name, values in deep.items()}}
    spread_components = {"ridge": fit_component(frame, "spread", train_mask, val_mask, "ridge"), **{name: values[:, :, 1].reshape(-1) for name, values in deep.items()}}

    da_sets = {
        "xgboost_only": ("xgboost",),
        "xgboost_gru": ("xgboost", "gru"),
        "xgboost_tcn": ("xgboost", "tcn"),
        "xgboost_gru_tcn": ("xgboost", "gru", "tcn"),
        "all_deep": ("xgboost", "transformer", "mamba", "cnn", "lstm", "gru", "tcn"),
    }
    spread_sets = {
        "ridge_only": ("ridge",),
        "ridge_gru": ("ridge", "gru"),
        "ridge_tcn": ("ridge", "tcn"),
        "ridge_gru_tcn": ("ridge", "gru", "tcn"),
        "existing_mamba": ("ridge", "transformer", "mamba"),
        "all_deep": ("ridge", "transformer", "mamba", "cnn", "lstm", "gru", "tcn"),
    }

    calibration: dict[str, dict[str, Any]] = {"day_ahead": {}, "spread": {}}
    for candidate, names in da_sets.items():
        selected = {name: da_components[name] for name in names}
        weights = fit_weights(actual_da, selected)
        calibration["day_ahead"][candidate] = {"weights": weights, "metrics": base.metric(actual_da, combine(selected, weights))}
    for candidate, names in spread_sets.items():
        selected = {name: spread_components[name] for name in names}
        weights = fit_weights(actual_spread, selected)
        calibration["spread"][candidate] = {"weights": weights, "metrics": base.metric(actual_spread, combine(selected, weights))}

    selected_da = min(calibration["day_ahead"], key=lambda name: calibration["day_ahead"][name]["metrics"]["mae_yuan_per_mwh"])
    selected_spread = min(calibration["spread"], key=lambda name: calibration["spread"][name]["metrics"]["mae_yuan_per_mwh"])

    baseline = pd.read_csv(CNN_LSTM_DIR / "cnn_lstm_blend_predictions.csv")
    gru = pd.read_csv(OUTPUT_DIR / "gru_walk_forward_predictions.csv")
    tcn = pd.read_csv(OUTPUT_DIR / "tcn_walk_forward_predictions.csv")
    keys = ["market_date", "period"]
    test = baseline.merge(gru[keys + ["da_gru_pred", "spread_gru_pred", "rt_gru_pred"]], on=keys, validate="one_to_one")
    test = test.merge(tcn[keys + ["da_tcn_pred", "spread_tcn_pred", "rt_tcn_pred"]], on=keys, validate="one_to_one")
    test_da = {name: test[f"da_{name}_pred"].to_numpy(float) for name in ("xgboost", "transformer", "mamba", "cnn", "lstm", "gru", "tcn")}
    test_spread = {name: test[f"spread_{name}_pred"].to_numpy(float) for name in ("ridge", "transformer", "mamba", "cnn", "lstm", "gru", "tcn")}
    for candidate, details in calibration["day_ahead"].items():
        test[f"da_calibrated_{candidate}_pred"] = combine(test_da, details["weights"])
    for candidate, details in calibration["spread"].items():
        test[f"spread_calibrated_{candidate}_pred"] = combine(test_spread, details["weights"])
    test["da_selected_pred"] = test[f"da_calibrated_{selected_da}_pred"]
    test["spread_selected_pred"] = test[f"spread_calibrated_{selected_spread}_pred"]
    test["rt_selected_pred"] = test["da_selected_pred"] + test["spread_selected_pred"]

    transformer_test = pd.read_csv(TRANSFORMER_DIR / "precalibrated_transformer_blend_predictions.csv")
    mamba_test = pd.read_csv(MAMBA_DIR / "precalibrated_mamba_blend_predictions.csv")
    test = test.merge(
        transformer_test[keys + ["da_precalibrated_pred", "spread_precalibrated_pred", "rt_precalibrated_pred"]],
        on=keys,
        validate="one_to_one",
    )
    test = test.merge(mamba_test[keys + ["spread_precalibrated_four_model_pred", "rt_xgb_plus_precalibrated_spread_pred"]], on=keys, validate="one_to_one")
    test["rt_xgb_tcn_plus_mamba_spread_pred"] = (
        test["da_calibrated_xgboost_tcn_pred"] + test["spread_precalibrated_four_model_pred"]
    )
    test.to_csv(OUTPUT_DIR / "precalibrated_gru_tcn_fusion_predictions.csv", index=False, encoding="utf-8-sig")

    backtest_da = {candidate: base.metric(test["da_actual"], test[f"da_calibrated_{candidate}_pred"]) for candidate in calibration["day_ahead"]}
    backtest_spread = {candidate: base.metric(test["spread_actual"], test[f"spread_calibrated_{candidate}_pred"]) for candidate in calibration["spread"]}
    selected_metrics = {
        "day_ahead": base.metric(test["da_actual"], test["da_selected_pred"]),
        "spread": base.metric(test["spread_actual"], test["spread_selected_pred"]),
        "real_time_coherent": base.metric(test["rt_actual"], test["rt_selected_pred"]),
        "spread_direction_accuracy": float(((test["spread_selected_pred"] >= 0) == (test["spread_actual"] >= 0)).mean()),
    }
    statistical = {
        "xgboost_tcn_da_vs_xgboost": paired_day_bootstrap(
            test, "da_actual", "da_xgboost_pred", "da_calibrated_xgboost_tcn_pred", seed=20260839
        ),
        "xgboost_tcn_da_vs_transformer_fusion": paired_day_bootstrap(
            test, "da_actual", "da_precalibrated_pred", "da_calibrated_xgboost_tcn_pred", seed=20260840
        ),
        "selected_spread_vs_ridge": paired_day_bootstrap(test, "spread_actual", "spread_ridge_pred", "spread_selected_pred", seed=20260841),
        "selected_spread_vs_mamba_fusion": paired_day_bootstrap(test, "spread_actual", "spread_precalibrated_four_model_pred", "spread_selected_pred", seed=20260842),
        "selected_rt_vs_transformer_fusion": paired_day_bootstrap(test, "rt_actual", "rt_precalibrated_pred", "rt_selected_pred", seed=20260843),
        "selected_rt_vs_mamba_fusion": paired_day_bootstrap(test, "rt_actual", "rt_xgb_plus_precalibrated_spread_pred", "rt_selected_pred", seed=20260844),
        "xgboost_tcn_mamba_rt_vs_transformer_fusion": paired_day_bootstrap(
            test, "rt_actual", "rt_precalibrated_pred", "rt_xgb_tcn_plus_mamba_spread_pred", seed=20260845
        ),
    }
    output = {
        "calibration_period": {"start": tune_start.date().isoformat(), "end": tune_end.date().isoformat(), "days": [day.date().isoformat() for day in accepted_days]},
        "candidate_models": ["XGBoost", "Ridge", "Transformer", "MambaPy", "CNN", "LSTM", "GRU", "residual TCN"],
        "calibration": calibration,
        "selected_by_calibration": {"day_ahead": selected_da, "spread": selected_spread},
        "reported_backtest": {
            "all_day_ahead_candidates": backtest_da,
            "all_spread_candidates": backtest_spread,
            "selected": selected_metrics,
            "xgboost_tcn_plus_mamba_spread_real_time": base.metric(
                test["rt_actual"], test["rt_xgb_tcn_plus_mamba_spread_pred"]
            ),
        },
        "statistical_comparison": statistical,
        "leakage_control": "All architectures and weights were fixed before the June 15-30 test; test folds train only on earlier dates.",
    }
    (OUTPUT_DIR / "precalibrated_gru_tcn_fusion_summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
