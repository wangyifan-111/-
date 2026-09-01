"""Generate the Mamba-enhanced spread-focused 24-hour forecast."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import integrated_price_forecast as base


ROOT = Path(__file__).resolve().parent
MAMBA_DIR = ROOT / "outputs" / "mamba_trial_20260827"
TREE_DIR = ROOT / "outputs" / "selected_fusion_forecast_20260826"
OUTPUT_DIR = ROOT / "outputs" / "mamba_enhanced_forecast_20260827"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    calibration = json.loads((MAMBA_DIR / "precalibrated_mamba_blend_summary.json").read_text(encoding="utf-8"))
    ridge_weight = float(calibration["weights"]["spread"]["ridge"])
    mamba_weight = float(calibration["weights"]["spread"]["mamba"])
    tree = pd.read_csv(TREE_DIR / "forecast.csv")
    mamba = pd.read_csv(MAMBA_DIR / "forecast.csv")
    if len(tree) != 24 or len(mamba) != 24:
        raise ValueError("both inputs must contain 24 periods")
    result = pd.DataFrame({"market_date": tree["market_date"], "period": tree["period"].astype(int)})
    result["day_ahead_p50"] = tree["day_ahead_p50"]
    result["spread_p50"] = (
        ridge_weight * tree["spread_p50"] + mamba_weight * mamba["spread_real_time_minus_day_ahead"]
    )
    result["real_time_p50"] = result["day_ahead_p50"] + result["spread_p50"]

    backtest = pd.read_csv(MAMBA_DIR / "precalibrated_mamba_blend_predictions.csv")
    q90 = {
        "day_ahead": base.conformal_quantile(np.abs(backtest["da_actual"] - backtest["da_xgboost_pred"]), 0.1),
        "spread": base.conformal_quantile(
            np.abs(backtest["spread_actual"] - backtest["spread_precalibrated_four_model_pred"]), 0.1
        ),
        "real_time": base.conformal_quantile(
            np.abs(backtest["rt_actual"] - backtest["rt_xgb_plus_precalibrated_spread_pred"]), 0.1
        ),
    }
    for target in ("day_ahead", "spread", "real_time"):
        result[f"{target}_p10"] = result[f"{target}_p50"] - q90[target]
        result[f"{target}_p90"] = result[f"{target}_p50"] + q90[target]
    result["spread_positive_probability"] = tree["spread_positive_probability"]
    result["spread_direction"] = np.where(result["spread_p50"] >= 0, "positive", "negative")
    result["negative_price_risk"] = (result["day_ahead_p10"] < 0) | (result["real_time_p10"] < 0)
    result["high_price_risk"] = (result["day_ahead_p90"] > 500) | (result["real_time_p90"] > 500)
    result.to_csv(OUTPUT_DIR / "forecast.csv", index=False, encoding="utf-8-sig")
    model_card = {
        "model_version": "mamba-enhanced-spread-v1.0.0",
        "implementation": "MambaPy 1.2.0 pure-PyTorch selective state-space model; not official CUDA mamba-ssm kernel",
        "architecture": {
            "day_ahead": "XGBoost",
            "spread": f"{ridge_weight:.4f} Ridge + {mamba_weight:.4f} MambaPy",
            "real_time": "day-ahead + spread",
            "mamba": "2 layers, d_model=32, d_state=16, 168-hour history plus 24 future exogenous tokens",
        },
        "weight_calibration": calibration["calibration_period"],
        "reported_backtest": {
            "day_ahead": base.metric(backtest["da_actual"], backtest["da_xgboost_pred"]),
            "spread": base.metric(backtest["spread_actual"], backtest["spread_precalibrated_four_model_pred"]),
            "real_time": base.metric(backtest["rt_actual"], backtest["rt_xgb_plus_precalibrated_spread_pred"]),
            "spread_direction_accuracy": calibration["reported_backtest"]["spread_direction_accuracy"],
        },
        "positioning": "Best current spread-focused candidate; Transformer-enhanced variant remains better for coherent RT MAE.",
        "interval": {"method": "90% finite-sample Conformal interval", "q90": q90},
    }
    rows = result.to_dict(orient="records")
    (OUTPUT_DIR / "model_card.json").write_text(json.dumps(model_card, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "forecast.json").write_text(json.dumps({"model_card": model_card, "forecast": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "rows": len(rows), "spread_weights": {"ridge": ridge_weight, "mamba": mamba_weight}, "backtest": model_card["reported_backtest"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
