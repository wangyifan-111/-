"""Paired day-block bootstrap for the Transformer-enhanced forecast."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "transformer_trial_20260827" / "precalibrated_transformer_blend_predictions.csv"
OUTPUT = ROOT / "outputs" / "transformer_trial_20260827" / "statistical_comparison.json"


def compare(frame: pd.DataFrame, actual: str, baseline: np.ndarray, enhanced: str, rng: np.random.Generator) -> dict:
    work = pd.DataFrame(
        {
            "market_date": frame["market_date"],
            "baseline_abs_error": np.abs(frame[actual].to_numpy(float) - baseline),
            "enhanced_abs_error": np.abs(frame[actual].to_numpy(float) - frame[enhanced].to_numpy(float)),
        }
    )
    daily = work.groupby("market_date")[["baseline_abs_error", "enhanced_abs_error"]].mean()
    daily["improvement"] = daily["baseline_abs_error"] - daily["enhanced_abs_error"]
    values = daily["improvement"].to_numpy(float)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return {
        "baseline_mae": float(work["baseline_abs_error"].mean()),
        "enhanced_mae": float(work["enhanced_abs_error"].mean()),
        "mae_improvement": float(values.mean()),
        "relative_improvement_pct": float(100 * values.mean() / work["baseline_abs_error"].mean()),
        "day_block_bootstrap_95_ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "bootstrap_probability_improved": float(np.mean(draws > 0)),
        "days_improved": int((values > 0).sum()),
        "days_total": int(len(values)),
    }


def main() -> None:
    frame = pd.read_csv(INPUT)
    frame["rt_spread_only_enhanced_pred"] = frame["da_xgboost_pred"] + frame["spread_precalibrated_pred"]
    rng = np.random.default_rng(20260827)
    output = {
        "method": "paired nonparametric bootstrap over 16 daily error blocks, 10000 draws",
        "day_ahead": compare(frame, "da_actual", frame["da_xgboost_pred"].to_numpy(float), "da_precalibrated_pred", rng),
        "spread": compare(frame, "spread_actual", frame["spread_ridge_pred"].to_numpy(float), "spread_precalibrated_pred", rng),
        "real_time_coherent": compare(
            frame,
            "rt_actual",
            (frame["da_xgboost_pred"] + frame["spread_ridge_pred"]).to_numpy(float),
            "rt_precalibrated_pred",
            rng,
        ),
        "real_time_with_transformer_spread_only": compare(
            frame,
            "rt_actual",
            (frame["da_xgboost_pred"] + frame["spread_ridge_pred"]).to_numpy(float),
            "rt_spread_only_enhanced_pred",
            rng,
        ),
        "interpretation": "A 95% interval crossing zero means the observed gain is not yet statistically stable over this short backtest.",
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
