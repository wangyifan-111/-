"""Pre-backtest calibration and strict test evaluation for DLinear/NLinear blends."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import integrated_price_forecast as base
from ensemble_price_forecast import fit_component, simplex_weights
from linear_ts_price_forecast import LinearConfig, fit_predict_day, run_walk_forward
from transformer_price_forecast import complete_market_days


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "linear_ts_trial_20260829"
TUNE_START = pd.Timestamp("2026-06-08")
TUNE_END = pd.Timestamp("2026-06-14")
TEST_START = pd.Timestamp("2026-06-15")
TEST_END = pd.Timestamp("2026-06-30")


def day_predictions(frame: pd.DataFrame, days: list[pd.Timestamp], configs: dict[str, LinearConfig], epochs: dict[str, int]) -> pd.DataFrame:
    date = frame["market_date"]
    complete = frame["weather_complete"] & frame["power_complete"]
    rows: list[dict[str, float | str | int]] = []
    for fold, day in enumerate(days):
        train_mask = (date < day) & complete
        predict_mask = date.eq(day) & complete
        da_xgb = fit_component(frame, "da", train_mask, predict_mask, "xgboost")
        spread_ridge = fit_component(frame, "spread", train_mask, predict_mask, "ridge")
        preds = {name: fit_predict_day(frame, day, cfg, epochs[name], 28000 + fold * 20 + i)[0] for i, (name, cfg) in enumerate(configs.items())}
        actual = frame.loc[predict_mask].sort_values("period").reset_index(drop=True)
        for h in range(24):
            row: dict[str, float | str | int] = {"market_date": day.date().isoformat(), "period": h + 1, "da_actual": float(actual.iloc[h]["da"]), "spread_actual": float(actual.iloc[h]["spread"]), "rt_actual": float(actual.iloc[h]["rt"]), "da_xgb_pred": float(da_xgb[h]), "spread_ridge_pred": float(spread_ridge[h])}
            for name, prediction in preds.items():
                row[f"da_{name}_pred"] = float(prediction[h, 0])
                row[f"spread_{name}_pred"] = float(prediction[h, 1])
            rows.append(row)
        print(f"calibration fold {fold + 1}/{len(days)}: {day.date().isoformat()}", flush=True)
    return pd.DataFrame(rows)


def combine(frame: pd.DataFrame, actual_col: str, columns: list[str], weights: np.ndarray) -> np.ndarray:
    return frame[columns].to_numpy(float) @ weights


def day_bootstrap(df: pd.DataFrame, actual: str, a: str, b: str, seed: int = 20260829, reps: int = 10000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    days = np.array(sorted(df["market_date"].unique()))
    diff: list[float] = []
    grouped = {day: g for day, g in df.groupby("market_date")}
    for _ in range(reps):
        sampled = rng.choice(days, size=len(days), replace=True)
        ea = np.concatenate([np.abs(grouped[d][a].to_numpy(float) - grouped[d][actual].to_numpy(float)) for d in sampled])
        eb = np.concatenate([np.abs(grouped[d][b].to_numpy(float) - grouped[d][actual].to_numpy(float)) for d in sampled])
        diff.append(float(ea.mean() - eb.mean()))
    return {"improvement_b_over_a_mean": float(np.mean(diff)), "ci95_low": float(np.quantile(diff, .025)), "ci95_high": float(np.quantile(diff, .975)), "repetitions": reps, "days": int(len(days))}


def main() -> None:
    frame, coverage = base.load_price_weather(base.PRICE_DEFAULT, base.WEATHER_DEFAULT, sorted(ROOT.glob(base.POWER_GLOB)))
    frame = base.add_all_feature_tables(frame)
    summaries = json.loads((OUT / "linear_ts_summary.json").read_text(encoding="utf-8"))
    configs = {name: LinearConfig(**details["selected_config"]) for name, details in summaries["models"].items()}
    epochs = {name: int(details["selected_epochs"]) for name, details in summaries["models"].items()}
    tune_days = [d for d in complete_market_days(frame) if TUNE_START <= d <= TUNE_END]
    test_days = [d for d in complete_market_days(frame) if TEST_START <= d <= TEST_END]
    calibration = day_predictions(frame, tune_days, configs, epochs)
    calibration.to_csv(OUT / "linear_calibration_predictions.csv", index=False, encoding="utf-8-sig")
    # Candidate sets are kept small to limit overfitting on seven calibration days.
    candidates = {
        "da_xgb_dlinear": (["da_xgb_pred", "da_dlinear_pred"], "da"),
        "da_xgb_nlinear": (["da_xgb_pred", "da_nlinear_pred"], "da"),
        "spread_ridge_dlinear": (["spread_ridge_pred", "spread_dlinear_pred"], "spread"),
        "spread_ridge_nlinear": (["spread_ridge_pred", "spread_nlinear_pred"], "spread"),
    }
    calibration_results: dict[str, dict[str, object]] = {}
    for name, (cols, target) in candidates.items():
        actual = calibration[f"{target}_actual"].to_numpy(float)
        weights = simplex_weights(actual, calibration[cols].to_numpy(float), "mae")
        calibration_results[name] = {"weights": {c: float(w) for c, w in zip(cols, weights)}, "calibration_mae": base.metric(actual, calibration[cols].to_numpy(float) @ weights)}

    for name in configs:
        test_path = OUT / f"{name}_walk_forward_predictions.csv"
        if not test_path.exists():
            _, _ = run_walk_forward(frame, name, configs[name], epochs[name], TEST_START, TEST_END)
    dlinear = pd.read_csv(OUT / "dlinear_walk_forward_predictions.csv")
    nlinear = pd.read_csv(OUT / "nlinear_walk_forward_predictions.csv")
    # Existing tree baseline has the same 16-day folds and is already frozen.
    tree = pd.read_csv(ROOT / "outputs" / "ensemble_search_20260826" / "walk_forward_predictions.csv")
    keys = ["market_date", "period"]
    test = tree[keys + ["da_actual", "spread_actual", "rt_actual", "da_xgboost_pred", "spread_ridge_pred"]].copy()
    test["da_xgb_pred"] = test["da_xgboost_pred"]
    for name, src in (("dlinear", dlinear), ("nlinear", nlinear)):
        test = test.merge(src[keys + [f"da_{name}_pred", f"spread_{name}_pred"]], on=keys, validate="one_to_one")
    for name, details in calibration_results.items():
        cols = list(details["weights"])
        weights = np.array(list(details["weights"].values()), dtype=float)
        target = "da" if name.startswith("da_") else "spread"
        test[f"{name}_pred"] = combine(test, f"{target}_actual", [c for c in cols], weights)
    # Evaluate coherent real time for the best DA and spread candidates.
    scores: dict[str, object] = {}
    for name in calibration_results:
        target = "da" if name.startswith("da_") else "spread"
        scores[name] = base.metric(test[f"{target}_actual"], test[f"{name}_pred"])
    test["rt_xgb_ridge_pred"] = test["da_xgb_pred"] + test["spread_ridge_pred"]
    for da_name in ("da_xgb_dlinear", "da_xgb_nlinear"):
        for spread_name in ("spread_ridge_dlinear", "spread_ridge_nlinear"):
            combo = f"rt_{da_name}_{spread_name}"
            test[combo] = test[f"{da_name}_pred"] + test[f"{spread_name}_pred"]
            scores[combo] = base.metric(test["rt_actual"], test[combo])
    statistical = {
        "da_xgb_dlinear_vs_xgb": day_bootstrap(test, "da_actual", "da_xgb_pred", "da_xgb_dlinear_pred", 20260891),
        "da_xgb_nlinear_vs_xgb": day_bootstrap(test, "da_actual", "da_xgb_pred", "da_xgb_nlinear_pred", 20260892),
        "spread_ridge_nlinear_vs_ridge": day_bootstrap(test, "spread_actual", "spread_ridge_pred", "spread_ridge_nlinear_pred", 20260893),
        "rt_best_linear_vs_xgb_ridge": day_bootstrap(test, "rt_actual", "rt_xgb_ridge_pred", "rt_da_xgb_nlinear_spread_ridge_nlinear", 20260894),
    }
    test.to_csv(OUT / "linear_fusion_backtest_predictions.csv", index=False, encoding="utf-8-sig")
    payload = {"calibration_period": {"start": str(TUNE_START.date()), "end": str(TUNE_END.date())}, "backtest_period": {"start": str(TEST_START.date()), "end": str(TEST_END.date())}, "calibration": calibration_results, "backtest_scores": scores, "statistical_comparison": statistical, "data_coverage": coverage, "recommendation": "Keep DLinear/NLinear as benchmark/challenger; do not replace the current XGBoost/Ridge/Transformer production blend unless a longer history confirms a stable gain.", "leakage_control": "Weights are fitted on June 8-14 before the June 15-30 test; each model fold uses only dates before the target day."}
    (OUT / "linear_fusion_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
