"""Day-block bootstrap for the calibrated Ridge/Mamba spread blend."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "mamba_trial_20260827" / "precalibrated_mamba_blend_predictions.csv"
OUTPUT = ROOT / "outputs" / "mamba_trial_20260827" / "statistical_comparison.json"


def paired(frame: pd.DataFrame, actual: str, old: np.ndarray, new: np.ndarray, rng: np.random.Generator) -> dict:
    work = pd.DataFrame(
        {
            "market_date": frame["market_date"],
            "old": np.abs(frame[actual].to_numpy(float) - old),
            "new": np.abs(frame[actual].to_numpy(float) - new),
        }
    )
    daily = work.groupby("market_date")[["old", "new"]].mean()
    improvement = (daily["old"] - daily["new"]).to_numpy(float)
    draws = rng.choice(improvement, size=(10000, len(improvement)), replace=True).mean(axis=1)
    return {
        "old_mae": float(work["old"].mean()),
        "new_mae": float(work["new"].mean()),
        "mae_improvement": float(improvement.mean()),
        "relative_improvement_pct": float(100 * improvement.mean() / work["old"].mean()),
        "day_block_bootstrap_95_ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "bootstrap_probability_improved": float(np.mean(draws > 0)),
        "days_improved": int((improvement > 0).sum()),
        "days_total": int(len(improvement)),
    }


def main() -> None:
    frame = pd.read_csv(INPUT)
    rng = np.random.default_rng(20260827)
    mamba_spread = frame["spread_precalibrated_four_model_pred"].to_numpy(float)
    transformer_spread = 0.79 * frame["spread_ridge_pred"].to_numpy(float) + 0.21 * frame["spread_transformer_pred"].to_numpy(float)
    ridge_spread = frame["spread_ridge_pred"].to_numpy(float)
    xgb_da = frame["da_xgboost_pred"].to_numpy(float)
    output = {
        "method": "paired nonparametric bootstrap over 16 daily error blocks, 10000 draws",
        "spread_vs_ridge": paired(frame, "spread_actual", ridge_spread, mamba_spread, rng),
        "spread_vs_transformer_fusion": paired(frame, "spread_actual", transformer_spread, mamba_spread, rng),
        "real_time_vs_ridge": paired(frame, "rt_actual", xgb_da + ridge_spread, xgb_da + mamba_spread, rng),
        "real_time_vs_transformer_fusion": paired(
            frame, "rt_actual", xgb_da + transformer_spread, xgb_da + mamba_spread, rng
        ),
        "interpretation": "Intervals above zero support Mamba inclusion; intervals crossing zero indicate insufficient evidence over 16 days.",
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
