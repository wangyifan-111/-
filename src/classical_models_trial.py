"""Strict rolling comparison of classical EPF models from the literature.

Candidates: Lasso/ElasticNet (LEAR-style regularization), Random Forest,
ExtraTrees, SVR, and a seasonal naive lag baseline. All models use the same
pre-market feature tables as the existing XGBoost/Ridge system.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import integrated_price_forecast as base


def make_model(name: str):
    if name == "lasso":
        return make_pipeline(StandardScaler(), Lasso(alpha=0.04, max_iter=12000, random_state=42))
    if name == "elasticnet":
        return make_pipeline(StandardScaler(), ElasticNet(alpha=0.04, l1_ratio=0.25, max_iter=12000, random_state=42))
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=240, max_depth=14, min_samples_leaf=2, max_features=0.8, random_state=42, n_jobs=-1)
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=240, max_depth=14, min_samples_leaf=2, max_features=0.8, random_state=42, n_jobs=-1)
    if name == "svr":
        return make_pipeline(StandardScaler(), SVR(C=35.0, epsilon=0.10, gamma="scale", kernel="rbf"))
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.01, learning_rate_init=0.001, max_iter=600, early_stopping=False, random_state=42))
    raise ValueError(name)


def clean(frame: pd.DataFrame, cols: list[str], train_mask: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(x[train_mask.to_numpy(bool)], axis=0)
    med[~np.isfinite(med)] = 0.0
    return np.where(np.isfinite(x), x, med), med


def fit_predict(frame: pd.DataFrame, target: str, train_mask: pd.Series, predict_mask: pd.Series, name: str) -> np.ndarray:
    cols = base.feature_columns(target)
    x, _ = clean(frame, cols, train_mask)
    y = frame[target].to_numpy(float)
    valid = train_mask.to_numpy(bool) & np.isfinite(y)
    model = make_model(name)
    model.fit(x[valid], y[valid])
    return np.asarray(model.predict(x[predict_mask.to_numpy(bool)]), dtype=float)


def seasonal_naive(frame: pd.DataFrame, target: str, train_mask: pd.Series, predict_mask: pd.Series) -> np.ndarray:
    """Use only lagged target values; average yesterday and last-week analogues."""
    rows = frame.loc[predict_mask].sort_values("datetime")
    out = []
    for _, row in rows.iterrows():
        vals = [row.get(f"{target}_lag_24", np.nan), row.get(f"{target}_lag_168", np.nan)]
        vals = [float(v) for v in vals if np.isfinite(v)]
        out.append(float(np.mean(vals)) if vals else float(frame.loc[train_mask, target].median()))
    return np.asarray(out, dtype=float)


def run_model(frame: pd.DataFrame, name: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    days = sorted(pd.to_datetime(frame.loc[frame["weather_complete"] & frame["power_complete"], "market_date"].unique()))
    days = [d for d in days if start <= d <= end]
    rows: list[dict[str, Any]] = []
    date = frame["market_date"]
    complete = frame["weather_complete"] & frame["power_complete"]
    for fold, day in enumerate(days):
        train_mask = (date < day) & complete
        predict_mask = date.eq(day) & complete
        pred: dict[str, np.ndarray] = {}
        for target in ("da", "spread", "rt"):
            pred[target] = seasonal_naive(frame, target, train_mask, predict_mask) if name == "seasonal_naive" else fit_predict(frame, target, train_mask, predict_mask, name)
        actual = frame.loc[predict_mask].sort_values("period").reset_index(drop=True)
        for h in range(24):
            rows.append({"market_date": day.date().isoformat(), "period": h + 1, "da_actual": float(actual.iloc[h]["da"]), "spread_actual": float(actual.iloc[h]["spread"]), "rt_actual": float(actual.iloc[h]["rt"]), f"da_{name}_pred": float(pred["da"][h]), f"spread_{name}_pred": float(pred["spread"][h]), f"rt_{name}_direct_pred": float(pred["rt"][h]), f"rt_{name}_coherent_pred": float(pred["da"][h] + pred["spread"][h])})
        print(f"completed {name} fold {fold + 1}/{len(days)}: {day.date().isoformat()}", flush=True)
    result = pd.DataFrame(rows)
    scores = {
        "day_ahead": base.metric(result["da_actual"], result[f"da_{name}_pred"]),
        "spread": base.metric(result["spread_actual"], result[f"spread_{name}_pred"]),
        "real_time_direct": base.metric(result["rt_actual"], result[f"rt_{name}_direct_pred"]),
        "real_time_coherent": base.metric(result["rt_actual"], result[f"rt_{name}_coherent_pred"]),
        "spread_direction_accuracy": float(((result[f"spread_{name}_pred"] >= 0) == (result["spread_actual"] >= 0)).mean()),
    }
    return result, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical electricity price forecasting trials")
    parser.add_argument("--backtest-start", default="2026-06-15")
    parser.add_argument("--backtest-end", default="2026-06-30")
    parser.add_argument("--output-dir", type=Path, default=base.ROOT / "outputs" / "classical_models_trial_20260831")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, coverage = base.load_price_weather(base.PRICE_DEFAULT, base.WEATHER_DEFAULT, sorted(base.ROOT.glob(base.POWER_GLOB)))
    frame = base.add_all_feature_tables(frame)
    models = ("seasonal_naive", "lasso", "elasticnet", "random_forest", "extra_trees", "svr", "mlp")
    summary: dict[str, Any] = {"models": {}, "backtest_period": {"start": args.backtest_start, "end": args.backtest_end}, "data_coverage": coverage}
    for name in models:
        result, scores = run_model(frame, name, pd.Timestamp(args.backtest_start), pd.Timestamp(args.backtest_end))
        result.to_csv(args.output_dir / f"{name}_walk_forward_predictions.csv", index=False, encoding="utf-8-sig")
        summary["models"][name] = {"backtest": scores}
    summary["leakage_controls"] = ["Each fold fits only on dates before its target date.", "Feature medians are computed from the training fold only.", "No target-day realized price is used in prediction."]
    (args.output_dir / "classical_models_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
