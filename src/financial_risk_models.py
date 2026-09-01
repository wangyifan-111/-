"""Probabilistic risk models for heavy-tailed, clustered electricity prices.

Implements direct quantile boosting, a GARCH(1,1) volatility overlay with
empirical asymmetric innovations, and binary event probabilities for negative
and extreme prices. All folds use only information preceding the target day.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import integrated_price_forecast as base


QUANTILES = (0.05, 0.50, 0.95)
EVENTS = {
    "da_negative": ("da", lambda y: y < 0),
    "da_high": ("da", lambda y: y > 500),
    "rt_negative": ("rt", lambda y: y < 0),
    "rt_high": ("rt", lambda y: y > 600),
    "spread_up_tail": ("spread", lambda y: y > 200),
    "spread_down_tail": ("spread", lambda y: y < -200),
}


def prepare(frame: pd.DataFrame, target: str, train_mask: pd.Series) -> tuple[np.ndarray, np.ndarray, list[str]]:
    columns = base.feature_columns(target)
    raw = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    median = np.nanmedian(raw[train_mask.to_numpy(bool)], axis=0)
    median[~np.isfinite(median)] = 0.0
    return np.where(np.isfinite(raw), raw, median), frame[target].to_numpy(float), columns


def quantile_model(q: float):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=240, learning_rate=0.045, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=3.0, random_state=42 + int(q * 100))


def pinball(actual: np.ndarray, prediction: np.ndarray, q: float) -> float:
    error = actual - prediction
    return float(np.mean(np.maximum(q * error, (q - 1) * error)))


def garch_variance(errors: np.ndarray, horizon: int = 24) -> tuple[np.ndarray, float, float, dict[str, float]]:
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    errors = errors - np.mean(errors)
    variance = max(float(np.var(errors)), 1e-6)

    def recurse(params: np.ndarray) -> tuple[np.ndarray, float]:
        omega, alpha, beta = params
        h = np.empty(len(errors), dtype=float)
        h[0] = variance
        for i in range(1, len(errors)):
            h[i] = max(omega + alpha * errors[i - 1] ** 2 + beta * h[i - 1], 1e-8)
        likelihood = 0.5 * float(np.sum(np.log(h) + errors**2 / h))
        return h, likelihood

    initial = np.array([variance * 0.05, 0.10, 0.85])
    fitted = minimize(lambda p: recurse(p)[1], initial, method="SLSQP", bounds=((1e-8, variance * 10 + 1), (1e-6, 0.98), (1e-6, 0.98)), constraints={"type": "ineq", "fun": lambda p: 0.995 - p[1] - p[2]}, options={"maxiter": 180, "ftol": 1e-8})
    params = fitted.x if fitted.success else initial
    h, _ = recurse(params)
    standardized = errors / np.sqrt(h)
    q05, q95 = map(float, np.quantile(standardized, [0.05, 0.95]))
    future = np.empty(horizon, dtype=float)
    future[0] = max(params[0] + params[1] * errors[-1] ** 2 + params[2] * h[-1], 1e-8)
    for i in range(1, horizon):
        future[i] = max(params[0] + (params[1] + params[2]) * future[i - 1], 1e-8)
    return future, q05, q95, {"omega": float(params[0]), "alpha": float(params[1]), "beta": float(params[2]), "persistence": float(params[1] + params[2]), "optimization_success": bool(fitted.success)}


def fit_event_probability(x: np.ndarray, label: np.ndarray, xp: np.ndarray) -> np.ndarray:
    if np.unique(label).size < 2:
        return np.repeat(float(label.mean()), len(xp))
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=4.0, random_state=88)
    model.fit(x, label)
    return model.predict_proba(xp)[:, 1]


def safe_event_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, float | None]:
    actual = np.asarray(actual, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 0, 1)
    return {"event_rate": float(actual.mean()), "brier_score": float(brier_score_loss(actual, probability)), "roc_auc": float(roc_auc_score(actual, probability)) if np.unique(actual).size > 1 else None, "average_precision": float(average_precision_score(actual, probability)) if actual.sum() else None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantile, GARCH and event-risk electricity models")
    parser.add_argument("--backtest-start", default="2026-06-15")
    parser.add_argument("--backtest-end", default="2026-06-30")
    parser.add_argument("--output-dir", type=Path, default=base.ROOT / "outputs" / "financial_risk_trial_20260831")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, coverage = base.load_price_weather(base.PRICE_DEFAULT, base.WEATHER_DEFAULT, sorted(base.ROOT.glob(base.POWER_GLOB)))
    frame = base.add_all_feature_tables(frame)
    complete = frame["weather_complete"] & frame["power_complete"]
    date = frame["market_date"]
    days = sorted(pd.to_datetime(frame.loc[complete, "market_date"].unique()))
    days = [day for day in days if pd.Timestamp(args.backtest_start) <= day <= pd.Timestamp(args.backtest_end)]
    rows: list[dict[str, Any]] = []
    garch_parameters: list[dict[str, Any]] = []
    for fold, day in enumerate(days):
        train_mask, predict_mask = (date < day) & complete, date.eq(day) & complete
        forecasts: dict[str, dict[float, np.ndarray]] = {}
        garch_intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        event_probabilities: dict[str, np.ndarray] = {}
        cache: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
        for target in ("da", "spread", "rt"):
            x_all, y_all, columns = prepare(frame, target, train_mask)
            cache[target] = (x_all, y_all, columns)
            valid = train_mask.to_numpy(bool) & np.isfinite(y_all)
            x, y, xp = x_all[valid], y_all[valid], x_all[predict_mask.to_numpy(bool)]
            forecasts[target] = {}
            for q in QUANTILES:
                model = quantile_model(q).fit(x, y)
                forecasts[target][q] = np.asarray(model.predict(xp), dtype=float)
            ordered = np.sort(np.column_stack([forecasts[target][q] for q in QUANTILES]), axis=1)
            for i, q in enumerate(QUANTILES):
                forecasts[target][q] = ordered[:, i]
            lag24 = x_all[:, columns.index(f"{target}_lag_24")]
            seasonal_error = y_all[valid] - lag24[valid]
            variance, z05, z95, params = garch_variance(seasonal_error)
            scale = np.sqrt(variance)
            center = forecasts[target][0.50]
            garch_intervals[target] = (center + z05 * scale, center + z95 * scale)
            garch_parameters.append({"market_date": day.date().isoformat(), "target": target, "z05": z05, "z95": z95, **params})
        for event_name, (target, rule) in EVENTS.items():
            x_all, y_all, _ = cache[target]
            valid = train_mask.to_numpy(bool) & np.isfinite(y_all)
            event_probabilities[event_name] = fit_event_probability(x_all[valid], rule(y_all[valid]).astype(int), x_all[predict_mask.to_numpy(bool)])
        actual = frame.loc[predict_mask].sort_values("period").reset_index(drop=True)
        for hour in range(24):
            row: dict[str, Any] = {"market_date": day.date().isoformat(), "period": hour + 1, "da_actual": float(actual.iloc[hour]["da"]), "spread_actual": float(actual.iloc[hour]["spread"]), "rt_actual": float(actual.iloc[hour]["rt"])}
            for target in ("da", "spread", "rt"):
                for q, label in ((0.05, "p05"), (0.50, "p50"), (0.95, "p95")):
                    row[f"{target}_quantile_{label}"] = float(forecasts[target][q][hour])
                row[f"{target}_garch_p05"] = float(garch_intervals[target][0][hour])
                row[f"{target}_garch_p95"] = float(garch_intervals[target][1][hour])
            for event_name in EVENTS:
                row[f"{event_name}_probability"] = float(event_probabilities[event_name][hour])
            rows.append(row)
        print(f"completed risk fold {fold + 1}/{len(days)}: {day.date().isoformat()}", flush=True)
    result = pd.DataFrame(rows)
    target_scores: dict[str, Any] = {}
    for target in ("da", "spread", "rt"):
        actual = result[f"{target}_actual"].to_numpy(float)
        q05, q50, q95 = (result[f"{target}_quantile_{label}"].to_numpy(float) for label in ("p05", "p50", "p95"))
        g05, g95 = result[f"{target}_garch_p05"].to_numpy(float), result[f"{target}_garch_p95"].to_numpy(float)
        target_scores[target] = {"median_point": base.metric(actual, q50), "quantile_90_interval": {"coverage": float(((actual >= q05) & (actual <= q95)).mean()), "mean_width": float(np.mean(q95 - q05)), "pinball_p05": pinball(actual, q05, 0.05), "pinball_p50": pinball(actual, q50, 0.50), "pinball_p95": pinball(actual, q95, 0.95)}, "garch_90_interval": {"coverage": float(((actual >= g05) & (actual <= g95)).mean()), "mean_width": float(np.mean(g95 - g05)), "lower_exceedance": float((actual < g05).mean()), "upper_exceedance": float((actual > g95).mean())}}
    event_scores = {}
    for event_name, (target, rule) in EVENTS.items():
        event_scores[event_name] = safe_event_metrics(rule(result[f"{target}_actual"].to_numpy(float)).astype(int), result[f"{event_name}_probability"].to_numpy(float))
    result.to_csv(args.output_dir / "financial_risk_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(garch_parameters).to_csv(args.output_dir / "garch_parameters.csv", index=False, encoding="utf-8-sig")
    summary = {"backtest_period": {"start": args.backtest_start, "end": args.backtest_end}, "targets": target_scores, "events": event_scores, "event_definitions": {"da_negative": "DA < 0", "da_high": "DA > 500", "rt_negative": "RT < 0", "rt_high": "RT > 600", "spread_up_tail": "RT-DA > 200", "spread_down_tail": "RT-DA < -200"}, "data_coverage": coverage, "status": "post-hoc exploratory: requires confirmation on later unseen dates", "leakage_controls": ["Every fold uses only dates before the target day.", "Quantile models, event classifiers, GARCH parameters and innovation quantiles are refitted from each training fold.", "The June test interval has been inspected previously, so these results are not a new untouched holdout."]}
    (args.output_dir / "financial_risk_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
