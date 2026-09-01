"""Paired day-block bootstrap for classical model backtests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import integrated_price_forecast as base


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "classical_models_trial_20260831"


def bootstrap(df: pd.DataFrame, actual: str, a: str, b: str, seed: int, reps: int = 3000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    groups = {day: g for day, g in df.groupby("market_date")}
    days = np.array(sorted(groups))
    diffs = []
    for _ in range(reps):
        sample = rng.choice(days, size=len(days), replace=True)
        ea = np.concatenate([np.abs(groups[d][a].to_numpy(float) - groups[d][actual].to_numpy(float)) for d in sample])
        eb = np.concatenate([np.abs(groups[d][b].to_numpy(float) - groups[d][actual].to_numpy(float)) for d in sample])
        diffs.append(float(ea.mean() - eb.mean()))
    return {"improvement_b_over_a_mean": float(np.mean(diffs)), "ci95_low": float(np.quantile(diffs, .025)), "ci95_high": float(np.quantile(diffs, .975)), "repetitions": reps, "days": int(len(days))}


def main() -> None:
    baseline = pd.read_csv(ROOT / "outputs" / "ensemble_search_20260826" / "walk_forward_predictions.csv")
    merged = baseline[["market_date", "period", "da_actual", "spread_actual", "rt_actual", "da_xgboost_pred", "spread_ridge_pred"]].copy()
    for name in ("seasonal_naive", "lasso", "elasticnet", "random_forest", "extra_trees", "svr", "mlp"):
        p = pd.read_csv(OUT / f"{name}_walk_forward_predictions.csv")
        merged = merged.merge(p[["market_date", "period", f"da_{name}_pred", f"spread_{name}_pred", f"rt_{name}_coherent_pred"]], on=["market_date", "period"], validate="one_to_one")
    # Existing XGBoost baseline names are normalized for concise comparisons.
    merged["da_xgb_pred"] = merged["da_xgboost_pred"]
    merged["spread_ridge_pred"] = merged["spread_ridge_pred"]
    comparisons = {}
    for name in ("seasonal_naive", "lasso", "elasticnet", "random_forest", "extra_trees", "svr", "mlp"):
        comparisons[f"da_{name}_vs_xgb"] = bootstrap(merged, "da_actual", "da_xgb_pred", f"da_{name}_pred", 31000 + len(comparisons))
        comparisons[f"spread_{name}_vs_ridge"] = bootstrap(merged, "spread_actual", "spread_ridge_pred", f"spread_{name}_pred", 32000 + len(comparisons))
        comparisons[f"rt_{name}_vs_xgb_ridge"] = bootstrap(merged.assign(rt_xgb_ridge_pred=merged["da_xgb_pred"] + merged["spread_ridge_pred"]), "rt_actual", "rt_xgb_ridge_pred", f"rt_{name}_coherent_pred", 33000 + len(comparisons))
    payload = {"comparisons": comparisons, "note": "Positive improvement means candidate b has lower MAE than baseline a. Intervals crossing zero are inconclusive.", "backtest_period": {"start": str(merged.market_date.min()), "end": str(merged.market_date.max())}}
    (OUT / "classical_models_significance.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
