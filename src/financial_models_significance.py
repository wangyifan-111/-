"""Day-block bootstrap for financial-property model candidates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REGIME = ROOT / "outputs" / "financial_regime_trial_20260831"


def bootstrap(df: pd.DataFrame, actual: str, baseline: str, candidate: str, seed: int, reps: int = 5000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    groups = {day: group for day, group in df.groupby("market_date")}
    days = np.asarray(sorted(groups))
    differences = []
    for _ in range(reps):
        sample = rng.choice(days, len(days), replace=True)
        base_error = np.concatenate([np.abs(groups[d][baseline].to_numpy(float) - groups[d][actual].to_numpy(float)) for d in sample])
        candidate_error = np.concatenate([np.abs(groups[d][candidate].to_numpy(float) - groups[d][actual].to_numpy(float)) for d in sample])
        differences.append(float(base_error.mean() - candidate_error.mean()))
    return {"improvement_candidate_over_baseline": float(np.mean(differences)), "ci95_low": float(np.quantile(differences, 0.025)), "ci95_high": float(np.quantile(differences, 0.975)), "days": int(len(days)), "repetitions": reps}


def main() -> None:
    tree = pd.read_csv(ROOT / "outputs" / "ensemble_search_20260826" / "walk_forward_predictions.csv")
    transformer = pd.read_csv(ROOT / "outputs" / "transformer_trial_20260827" / "precalibrated_transformer_blend_predictions.csv")
    keys = ["market_date", "period"]
    merged = tree[keys + ["da_actual", "spread_actual", "rt_actual", "da_xgboost_pred", "spread_ridge_pred"]].copy()
    merged = merged.merge(transformer[keys + ["rt_precalibrated_pred"]], on=keys, validate="one_to_one")
    for name in ("huber", "median_gbdt", "setar", "regime_moe"):
        candidate = pd.read_csv(REGIME / f"{name}_walk_forward_predictions.csv")
        merged = merged.merge(candidate[keys + [f"da_{name}_pred", f"spread_{name}_pred", f"rt_{name}_direct_pred", f"rt_{name}_coherent_pred"]], on=keys, validate="one_to_one")
    comparisons = {
        "da_regime_moe_vs_xgboost": bootstrap(merged, "da_actual", "da_xgboost_pred", "da_regime_moe_pred", 41001),
        "da_median_gbdt_vs_xgboost": bootstrap(merged, "da_actual", "da_xgboost_pred", "da_median_gbdt_pred", 41002),
        "spread_huber_vs_ridge": bootstrap(merged, "spread_actual", "spread_ridge_pred", "spread_huber_pred", 41003),
        "spread_median_gbdt_vs_ridge": bootstrap(merged, "spread_actual", "spread_ridge_pred", "spread_median_gbdt_pred", 41004),
        "rt_setar_direct_vs_transformer_fusion": bootstrap(merged, "rt_actual", "rt_precalibrated_pred", "rt_setar_direct_pred", 41005),
        "rt_median_direct_vs_transformer_fusion": bootstrap(merged, "rt_actual", "rt_precalibrated_pred", "rt_median_gbdt_direct_pred", 41006),
    }
    payload = {"comparisons": comparisons, "interpretation": "Positive values favor the candidate. A 95% interval crossing zero is inconclusive.", "status": "post-hoc exploratory; later unseen dates are required for confirmation"}
    (REGIME / "financial_models_significance.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
