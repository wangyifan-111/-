"""Real-time price forecast derived from a real-time-minus-day-ahead spread model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from weather_price_forecast_optimized import (
    ALL_POWER_FEATURES,
    _make_model,
    add_features,
    load_data,
)


SPREAD_MODEL = "random_forest_weather_power"


def _fit_predict(
    features: pd.DataFrame,
    target: pd.Series,
    train_mask: pd.Series,
    predict_mask: pd.Series,
    model_name: str = SPREAD_MODEL,
) -> tuple[np.ndarray, object, dict[str, float]]:
    columns = list(ALL_POWER_FEATURES)
    x = features[columns].copy()
    median = x.loc[train_mask, columns].median(numeric_only=True)
    x = x.fillna(median).fillna(0.0)
    valid_train = train_mask & target.notna()
    if int(valid_train.sum()) < 24 * 14:
        raise ValueError("实时价差模型可用训练样本不足，至少需要14天")
    model = _make_model(model_name)
    model.fit(x.loc[valid_train], target.loc[valid_train].astype(float))
    return model.predict(x.loc[predict_mask]), model, {k: float(v) for k, v in median.items()}


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {
        "mae_yuan_per_mwh": float(np.mean(np.abs(error))),
        "rmse_yuan_per_mwh": float(np.sqrt(np.mean(error**2))),
        "bias_yuan_per_mwh": float(np.mean(error)),
        "sample_count": int(len(error)),
    }


def run_spread_forecast(
    price_path: Path,
    weather_path: Path,
    forecast_date: str,
    power_paths: list[Path],
) -> dict[str, Any]:
    """Forecast 24 real-time-minus-day-ahead spread points for one market day."""
    df, coverage = load_data(price_path, weather_path, power_paths)
    target_date = pd.Timestamp(forecast_date).normalize()
    spread = (df["rt"] - df["da"]).astype(float)
    features = add_features(df, spread.to_numpy(float))

    complete = features["weather_complete"] & features["power_complete"]
    train_mask = (df["market_date"] < target_date) & complete & spread.notna()
    target_mask = (df["market_date"] == target_date) & complete
    if int(target_mask.sum()) != 24:
        raise ValueError(f"实时价差目标日应有24个时段，实际为{int(target_mask.sum())}")

    prediction, _, median = _fit_predict(features, spread, train_mask, target_mask)
    target_rows = df.loc[target_mask, ["market_date", "period", "datetime"]].copy().reset_index(drop=True)

    # Use a final 14-day historical holdout to estimate spread error width and
    # report a transparent historical metric. This is separate from the
    # day-ahead model's final-test score.
    validation_start = target_date - pd.Timedelta(days=14)
    validation_mask = (df["market_date"] >= validation_start) & (df["market_date"] < target_date) & complete & spread.notna()
    validation_train_mask = (df["market_date"] < validation_start) & complete & spread.notna()
    validation_prediction, _, _ = _fit_predict(features, spread, validation_train_mask, validation_mask)
    validation_actual = spread.loc[validation_mask].to_numpy(float)
    validation_score = _metrics(validation_actual, validation_prediction)
    residuals = validation_actual - validation_prediction
    if len(residuals) < 24:
        residuals = spread.loc[train_mask].to_numpy(float) - float(spread.loc[train_mask].mean())
    q10, q90 = np.quantile(residuals, [0.1, 0.9])

    rows = []
    for index, row in target_rows.iterrows():
        p50 = float(prediction[index])
        rows.append(
            {
                "market_date": row["market_date"].date().isoformat(),
                "period": int(row["period"]),
                "datetime": row["datetime"].isoformat(),
                "spread_p10": float(p50 + q10),
                "spread_p50": p50,
                "spread_p90": float(p50 + q90),
            }
        )
    return {
        "model_version": "realtime-spread-random-forest-v1.0.0",
        "selected_model": SPREAD_MODEL,
        "target": "real_time_minus_day_ahead_yuan_per_mwh",
        "forecast": rows,
        "summary": {
            "validation_start": validation_start.date().isoformat(),
            "validation_end": (target_date - pd.Timedelta(days=1)).date().isoformat(),
            "validation": validation_score,
            "interval_q10_residual": float(q10),
            "interval_q90_residual": float(q90),
            "feature_count": len(ALL_POWER_FEATURES),
        },
        "coverage": coverage,
        "feature_median": median,
    }
