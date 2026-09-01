"""Create the production-style Transformer-enhanced forecast artifact."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import integrated_price_forecast as base


ROOT = Path(__file__).resolve().parent
TRIAL_DIR = ROOT / "outputs" / "transformer_trial_20260827"
TREE_DIR = ROOT / "outputs" / "selected_fusion_forecast_20260826"
OUTPUT_DIR = ROOT / "outputs" / "transformer_enhanced_forecast_20260827"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    calibration = json.loads((TRIAL_DIR / "precalibrated_blend_summary.json").read_text(encoding="utf-8"))
    da_weight = float(calibration["weights"]["day_ahead"]["xgboost"])
    spread_weight = float(calibration["weights"]["spread"]["ridge"])
    transformer = pd.read_csv(TRIAL_DIR / "forecast.csv")
    trees = pd.read_csv(TREE_DIR / "forecast.csv")
    if len(transformer) != 24 or len(trees) != 24:
        raise ValueError("both forecast inputs must contain 24 periods")
    result = pd.DataFrame(
        {
            "market_date": transformer["market_date"],
            "period": transformer["period"].astype(int),
        }
    )
    # The DA blend improved the short backtest but its day-block confidence
    # interval crossed zero. Keep XGBoost as the operational DA model and
    # expose the calibrated Transformer blend only as a benchmark.
    result["day_ahead_p50"] = trees["day_ahead_p50"]
    result["day_ahead_transformer_blend_benchmark_p50"] = (
        da_weight * trees["day_ahead_p50"] + (1.0 - da_weight) * transformer["day_ahead_price"]
    )
    result["spread_p50"] = (
        spread_weight * trees["spread_p50"]
        + (1.0 - spread_weight) * transformer["spread_real_time_minus_day_ahead"]
    )
    result["real_time_p50"] = result["day_ahead_p50"] + result["spread_p50"]

    backtest = pd.read_csv(TRIAL_DIR / "precalibrated_transformer_blend_predictions.csv")
    q90 = {
        "day_ahead": base.conformal_quantile(np.abs(backtest["da_actual"] - backtest["da_xgboost_pred"]), 0.1),
        "spread": base.conformal_quantile(np.abs(backtest["spread_actual"] - backtest["spread_precalibrated_pred"]), 0.1),
        "real_time": base.conformal_quantile(
            np.abs(backtest["rt_actual"] - (backtest["da_xgboost_pred"] + backtest["spread_precalibrated_pred"])), 0.1
        ),
    }
    for prefix in ("day_ahead", "spread", "real_time"):
        result[f"{prefix}_p10"] = result[f"{prefix}_p50"] - q90[prefix]
        result[f"{prefix}_p90"] = result[f"{prefix}_p50"] + q90[prefix]
    result["spread_positive_probability"] = trees["spread_positive_probability"]
    result["spread_direction"] = np.where(result["spread_p50"] >= 0, "positive", "negative")
    result["negative_price_risk"] = (result["day_ahead_p10"] < 0) | (result["real_time_p10"] < 0)
    result["high_price_risk"] = (result["day_ahead_p90"] > 500) | (result["real_time_p90"] > 500)
    result.to_csv(OUTPUT_DIR / "forecast.csv", index=False, encoding="utf-8-sig")
    rows = result.to_dict(orient="records")
    model_card = {
        "model_version": "transformer-enhanced-selective-fusion-v1.0.0",
        "architecture": {
            "day_ahead": "XGBoost (Transformer blend retained as benchmark only)",
            "spread": f"{spread_weight:.2f} Ridge + {1-spread_weight:.2f} Transformer",
            "real_time": "blended day-ahead + blended spread",
            "transformer": "one-layer encoder-decoder, d_model=32, 4 heads, 168-hour encoder, 24-hour decoder",
        },
        "weight_calibration": calibration["calibration_period"],
        "reported_backtest": {
            "day_ahead": base.metric(backtest["da_actual"], backtest["da_xgboost_pred"]),
            "spread": base.metric(backtest["spread_actual"], backtest["spread_precalibrated_pred"]),
            "real_time": base.metric(
                backtest["rt_actual"], backtest["da_xgboost_pred"] + backtest["spread_precalibrated_pred"]
            ),
        },
        "interval": {
            "method": "finite-sample Conformal absolute residual interval",
            "calibration_period": "2026-06-15 through 2026-06-30",
            "nominal_coverage": 0.90,
            "q90": q90,
        },
        "leakage_controls": [
            "Architecture and spread blend weight were fixed before the reported backtest.",
            "Transformer is operational only in the spread model because its DA gain was not statistically stable.",
            "The July 1 interval uses only residuals observed through June 30.",
            "Real-time price is coherent by construction: RT = DA + spread.",
        ],
    }
    (OUTPUT_DIR / "forecast.json").write_text(json.dumps({"model_card": model_card, "forecast": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "model_card.json").write_text(json.dumps(model_card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "rows": len(rows), "operational_weights": {"da_xgboost": 1.0, "spread_ridge": spread_weight, "spread_transformer": 1.0 - spread_weight}, "q90": q90}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
