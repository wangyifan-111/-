"""Exploratory models motivated by financial properties of electricity prices.

The candidates address heavy tails and regime changes: Huber regression,
median gradient boosting, a SETAR-style threshold regression, and a soft
regime-switching mixture of experts. Evaluation is daily walk-forward.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import integrated_price_forecast as base


MODELS = ("huber", "median_gbdt", "setar", "regime_moe")


def matrix(frame: pd.DataFrame, target: str, train_mask: pd.Series) -> tuple[np.ndarray, list[str]]:
    columns = base.feature_columns(target)
    raw = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    median = np.nanmedian(raw[train_mask.to_numpy(bool)], axis=0)
    median[~np.isfinite(median)] = 0.0
    return np.where(np.isfinite(raw), raw, median), columns


def robust_linear():
    return make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=0.10, max_iter=1200))


def ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def median_gbdt():
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=220, learning_rate=0.045, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=3.0, random_state=42)


def fit_setar(x: np.ndarray, y: np.ndarray, lag24: np.ndarray, xp: np.ndarray, lag24p: np.ndarray) -> np.ndarray:
    low, high = np.quantile(lag24[np.isfinite(lag24)], [0.25, 0.75])
    train_regime = np.where(lag24 < low, 0, np.where(lag24 > high, 2, 1))
    pred_regime = np.where(lag24p < low, 0, np.where(lag24p > high, 2, 1))
    fallback = ridge().fit(x, y)
    prediction = np.asarray(fallback.predict(xp), dtype=float)
    for regime in range(3):
        selected = train_regime == regime
        if int(selected.sum()) < 80:
            continue
        model = ridge().fit(x[selected], y[selected])
        mask = pred_regime == regime
        if mask.any():
            prediction[mask] = model.predict(xp[mask])
    return prediction


def fit_regime_moe(x: np.ndarray, y: np.ndarray, xp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Soft mixture: classify low/normal/high target regime, then mix experts."""
    low, high = np.quantile(y, [0.20, 0.80])
    labels = np.where(y < low, 0, np.where(y > high, 2, 1))
    classifier = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=3.0, random_state=42)
    classifier.fit(x, labels)
    probability = classifier.predict_proba(xp)
    expert_predictions = []
    fallback = median_gbdt().fit(x, y)
    for regime in range(3):
        selected = labels == regime
        if int(selected.sum()) < 80:
            expert_predictions.append(np.asarray(fallback.predict(xp), dtype=float))
        else:
            expert = HistGradientBoostingRegressor(loss="absolute_error", max_iter=180, learning_rate=0.05, max_leaf_nodes=11, min_samples_leaf=15, l2_regularization=4.0, random_state=100 + regime)
            expert.fit(x[selected], y[selected])
            expert_predictions.append(np.asarray(expert.predict(xp), dtype=float))
    prediction = np.sum(np.column_stack(expert_predictions) * probability, axis=1)
    return prediction, probability


def fit_predict(frame: pd.DataFrame, target: str, train_mask: pd.Series, predict_mask: pd.Series, name: str) -> tuple[np.ndarray, np.ndarray | None]:
    x_all, columns = matrix(frame, target, train_mask)
    y_all = frame[target].to_numpy(float)
    valid = train_mask.to_numpy(bool) & np.isfinite(y_all)
    x, y, xp = x_all[valid], y_all[valid], x_all[predict_mask.to_numpy(bool)]
    if name == "huber":
        model = robust_linear().fit(x, y)
        return np.asarray(model.predict(xp), dtype=float), None
    if name == "median_gbdt":
        model = median_gbdt().fit(x, y)
        return np.asarray(model.predict(xp), dtype=float), None
    lag_column = columns.index(f"{target}_lag_24")
    if name == "setar":
        return fit_setar(x, y, x[:, lag_column], xp, xp[:, lag_column]), None
    if name == "regime_moe":
        return fit_regime_moe(x, y, xp)
    raise ValueError(name)


def run(frame: pd.DataFrame, name: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    complete = frame["weather_complete"] & frame["power_complete"]
    date = frame["market_date"]
    days = sorted(pd.to_datetime(frame.loc[complete, "market_date"].unique()))
    days = [day for day in days if start <= day <= end]
    rows: list[dict[str, Any]] = []
    for fold, day in enumerate(days):
        train_mask, predict_mask = (date < day) & complete, date.eq(day) & complete
        forecasts, probabilities = {}, {}
        for target in ("da", "spread", "rt"):
            forecasts[target], probabilities[target] = fit_predict(frame, target, train_mask, predict_mask, name)
        actual = frame.loc[predict_mask].sort_values("period").reset_index(drop=True)
        for hour in range(24):
            row: dict[str, Any] = {"market_date": day.date().isoformat(), "period": hour + 1, "da_actual": float(actual.iloc[hour]["da"]), "spread_actual": float(actual.iloc[hour]["spread"]), "rt_actual": float(actual.iloc[hour]["rt"]), f"da_{name}_pred": float(forecasts["da"][hour]), f"spread_{name}_pred": float(forecasts["spread"][hour]), f"rt_{name}_direct_pred": float(forecasts["rt"][hour]), f"rt_{name}_coherent_pred": float(forecasts["da"][hour] + forecasts["spread"][hour])}
            if name == "regime_moe":
                for target in ("da", "spread", "rt"):
                    for regime, label in enumerate(("low", "normal", "high")):
                        row[f"{target}_{label}_probability"] = float(probabilities[target][hour, regime])
            rows.append(row)
        print(f"completed {name} fold {fold + 1}/{len(days)}: {day.date().isoformat()}", flush=True)
    result = pd.DataFrame(rows)
    scores = {"day_ahead": base.metric(result["da_actual"], result[f"da_{name}_pred"]), "spread": base.metric(result["spread_actual"], result[f"spread_{name}_pred"]), "real_time_direct": base.metric(result["rt_actual"], result[f"rt_{name}_direct_pred"]), "real_time_coherent": base.metric(result["rt_actual"], result[f"rt_{name}_coherent_pred"]), "spread_direction_accuracy": float(((result[f"spread_{name}_pred"] >= 0) == (result["spread_actual"] >= 0)).mean())}
    return result, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial-property motivated electricity models")
    parser.add_argument("--backtest-start", default="2026-06-15")
    parser.add_argument("--backtest-end", default="2026-06-30")
    parser.add_argument("--output-dir", type=Path, default=base.ROOT / "outputs" / "financial_regime_trial_20260831")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, coverage = base.load_price_weather(base.PRICE_DEFAULT, base.WEATHER_DEFAULT, sorted(base.ROOT.glob(base.POWER_GLOB)))
    frame = base.add_all_feature_tables(frame)
    summary: dict[str, Any] = {"models": {}, "backtest_period": {"start": args.backtest_start, "end": args.backtest_end}, "data_coverage": coverage, "status": "post-hoc exploratory: requires confirmation on later unseen dates"}
    for name in MODELS:
        result, scores = run(frame, name, pd.Timestamp(args.backtest_start), pd.Timestamp(args.backtest_end))
        result.to_csv(args.output_dir / f"{name}_walk_forward_predictions.csv", index=False, encoding="utf-8-sig")
        summary["models"][name] = {"backtest": scores}
    summary["leakage_controls"] = ["Each fold trains only on rows before the target day.", "Imputation and regime thresholds use the training fold only.", "This test interval has already been inspected in earlier work, so results are exploratory rather than a new untouched holdout."]
    (args.output_dir / "financial_regime_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
