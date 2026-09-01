"""Weather and dispatch-aware Shandong day-ahead price forecast.

The weather workbook is treated as archived day-ahead forecast data because it has
no forecast-publication timestamp. The assumption is recorded in every output.
The backtest is daily rolling-origin: each target day is forecast using only price
history through the previous market day and the weather forecast for the target day.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PRICE_DEFAULT = "山东省-现货价格-数据明细（2026-01-01_2026-06-30.xlsx"
WEATHER_DEFAULT = "分时天气预报-自定义-山东省-2026-01-01-2026-07-01.xlsx"
POWER_GLOB = "山东省-电源出力*.xlsx"
TARGET = "da"
LAGS = (24, 48, 72, 168, 336)
WEATHER_COLS = (
    "temperature",
    "wind10",
    "wind100",
    "ghi",
    "cloud",
    "precipitation",
    "humidity",
)
BASE_FEATURES = (
    "period",
    "clock_hour",
    "weekday",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "lag_24",
    "lag_48",
    "lag_72",
    "lag_168",
    "lag_336",
    "roll24_mean",
    "roll24_std",
    "roll168_mean",
    "roll168_std",
)
WEATHER_FEATURES = WEATHER_COLS + (
    "heating_cooling_degree",
    "ghi_hour_interaction",
    "cloud_ghi_interaction",
    "wind100_sq",
)
ALL_FEATURES = BASE_FEATURES + WEATHER_FEATURES
POWER_FEATURES = (
    "direct_load",
    "tie_line",
    "wind_power",
    "pv_power",
    "local_power",
    "self_power",
    "nuclear_power",
    "renewable_power",
    "renewable_share",
    "net_load_proxy",
    "power_forecast_flag",
)
WEATHER_POWER_FEATURES = WEATHER_FEATURES + POWER_FEATURES
ALL_POWER_FEATURES = BASE_FEATURES + WEATHER_POWER_FEATURES


def _market_datetime(date: pd.Series, period: pd.Series) -> pd.Series:
    date = pd.to_datetime(date.astype(str), errors="raise")
    period_text = period.astype(str).str.extract(r"(\d{1,2})", expand=False).astype(int)
    # Market period 24:00 is the 24th period of the market date, represented by
    # the following calendar date at 00:00 for chronological lag construction.
    return date + pd.to_timedelta(period_text % 24, unit="h") + pd.to_timedelta((period_text == 24).astype(int), unit="D")


def _market_period(period: pd.Series) -> pd.Series:
    """Map 15-minute dispatch points to the 24 market periods of their date."""
    text = period.astype(str).str.extract(r"(\d{1,2}):(\d{2})", expand=True)
    hour = text[0].astype(int)
    minute = text[1].astype(int)
    return np.where(hour == 24, 24, hour + (minute > 0).astype(int)).astype(int)


def _load_power(power_paths: list[Path]) -> tuple[pd.DataFrame, dict[str, object]]:
    frames = []
    source_counts: dict[str, int] = {}
    for path in sorted(power_paths):
        raw = pd.read_excel(path)
        raw.columns = [str(c).strip() for c in raw.columns]
        source = "forecast" if "预测出力" in path.name else "actual"
        source_counts[source] = source_counts.get(source, 0) + 1
        date = pd.to_datetime(raw.iloc[:, 0].astype(str), errors="raise").dt.normalize()
        period = pd.Series(_market_period(raw.iloc[:, 1]), index=raw.index)
        # The files share the same first two columns and dispatch columns in a
        # stable order; naming by position avoids locale/encoding issues.
        q = pd.DataFrame({
            "market_date": date,
            "period": period,
            "direct_load": pd.to_numeric(raw.iloc[:, 2], errors="coerce"),
            "tie_line": pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
            "wind_power": pd.to_numeric(raw.iloc[:, 4], errors="coerce"),
            "pv_power": pd.to_numeric(raw.iloc[:, 5], errors="coerce"),
            "local_power": pd.to_numeric(raw.iloc[:, 6], errors="coerce"),
            "self_power": pd.to_numeric(raw.iloc[:, 7], errors="coerce"),
            "nuclear_power": pd.to_numeric(raw.iloc[:, 10], errors="coerce"),
            "power_source": source,
        })
        frames.append(q)
    if not frames:
        return pd.DataFrame(columns=["market_date", "period", *POWER_FEATURES, "power_source"]), {
            "power_files": [], "power_rows": 0, "power_complete_days": 0,
        }
    raw_power = pd.concat(frames, ignore_index=True)
    numeric = list(POWER_FEATURES[:7])
    # Average the four 15-minute observations in each market period. If both
    # actual and forecast files ever overlap, prefer the forecast source for a
    # target day and retain the source label for auditing.
    raw_power["source_priority"] = (raw_power["power_source"] == "forecast").astype(int)
    # The supplied files do not overlap by month, but this source-priority
    # rule makes the merge deterministic if an overlapping forecast is added.
    source_by_period = raw_power.sort_values(["market_date", "period", "source_priority"]).groupby(
        ["market_date", "period"], as_index=False
    ).tail(1)[["market_date", "period", "power_source"]]
    grouped = raw_power.groupby(["market_date", "period"], as_index=False)[numeric].mean()
    grouped["renewable_power"] = grouped["wind_power"] + grouped["pv_power"]
    grouped["renewable_share"] = grouped["renewable_power"] / grouped["direct_load"].replace(0, np.nan)
    grouped["net_load_proxy"] = grouped["direct_load"] - grouped["renewable_power"]
    grouped = grouped.merge(source_by_period, on=["market_date", "period"], how="left")
    grouped["power_forecast_flag"] = (grouped["power_source"] == "forecast").astype(int)
    complete_days = grouped.groupby("market_date").size()
    meta = {
        "power_files": [str(p) for p in sorted(power_paths)],
        "power_rows_raw": int(len(raw_power)),
        "power_rows_market_period": int(len(grouped)),
        "power_start": grouped.market_date.min().date().isoformat(),
        "power_end": grouped.market_date.max().date().isoformat(),
        "power_complete_days": int((complete_days == 24).sum()),
        "power_source_file_counts": source_counts,
        "power_missing_cells_after_aggregation": int(grouped[list(POWER_FEATURES)].isna().sum().sum()),
    }
    return grouped, meta


def load_data(price_path: Path, weather_path: Path, power_paths: list[Path] | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    price_raw = pd.read_excel(price_path)
    price_raw.columns = [str(c).strip() for c in price_raw.columns]
    pdate = pd.to_datetime(price_raw.iloc[:, 0].astype(str), errors="raise")
    pperiod = price_raw.iloc[:, 1].astype(str).str.extract(r"(\d{1,2})", expand=False).astype(int)
    price = pd.DataFrame(
        {
            "market_date": pdate.dt.normalize(),
            "period": pperiod,
            "datetime": _market_datetime(price_raw.iloc[:, 0], price_raw.iloc[:, 1]),
            "da": pd.to_numeric(price_raw.iloc[:, 2], errors="coerce"),
            "rt": pd.to_numeric(price_raw.iloc[:, 3], errors="coerce"),
        }
    ).sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    # A missing historical price is filled only for the observed history; no
    # target value is used to construct weather features.
    price[["da", "rt"]] = price[["da", "rt"]].interpolate(limit_direction="both")

    weather_raw = pd.read_excel(weather_path)
    weather_raw.columns = [str(c).strip() for c in weather_raw.columns]
    weather_cols = {
        weather_raw.columns[3]: "temperature",
        weather_raw.columns[4]: "wind10",
        weather_raw.columns[5]: "wind100",
        weather_raw.columns[6]: "ghi",
        weather_raw.columns[7]: "cloud",
        weather_raw.columns[8]: "precipitation",
        weather_raw.columns[9]: "humidity",
    }
    wperiod = weather_raw.iloc[:, 1].astype(str).str.extract(r"(\d{1,2})", expand=False).astype(int)
    weather = pd.DataFrame(
        {
            "market_date": pd.to_datetime(weather_raw.iloc[:, 0].astype(str), errors="raise").dt.normalize(),
            "period": wperiod,
            "datetime": _market_datetime(weather_raw.iloc[:, 0], weather_raw.iloc[:, 1]),
            **{new: pd.to_numeric(weather_raw[old], errors="coerce") for old, new in weather_cols.items()},
        }
    ).sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    merged = price.merge(weather.drop(columns=["datetime"]), on=["market_date", "period"], how="left")
    # Keep weather-only rows after the price history so a production run can
    # generate the next market day's forecast from the supplied weather file.
    future_weather = weather.merge(price[["market_date", "period"]], on=["market_date", "period"], how="left", indicator=True)
    future_weather = future_weather[future_weather["_merge"] == "left_only"].drop(columns=["_merge"])
    if not future_weather.empty:
        future_weather = future_weather[["market_date", "period", "datetime", *WEATHER_COLS]]
        future_weather["da"] = np.nan
        future_weather["rt"] = np.nan
        future_weather = future_weather[["market_date", "period", "datetime", "da", "rt", *WEATHER_COLS]]
        merged = pd.concat([merged, future_weather], ignore_index=True, sort=False)
    merged = merged.sort_values("datetime").reset_index(drop=True)
    power_meta = {"power_files": [], "power_rows_raw": 0, "power_rows_market_period": 0, "power_complete_days": 0}
    if power_paths:
        power, power_meta = _load_power(power_paths)
        merged = merged.merge(power, on=["market_date", "period"], how="left")
        # The supplied dispatch forecast ends on 2026-06-30. For a later
        # production horizon, use the previous week's same market date/period
        # as an explicit proxy rather than silently using training medians.
        # Backtest rows are complete and are unaffected by this fallback.
        proxy_rows = 0
        power_lookup = power.set_index(["market_date", "period"])
        missing_power = merged["market_date"].notna() & ~merged[list(POWER_FEATURES)].notna().all(axis=1)
        for index in merged.index[missing_power]:
            target_date = merged.at[index, "market_date"]
            period = merged.at[index, "period"]
            key = (target_date - pd.Timedelta(days=7), period)
            if key not in power_lookup.index:
                continue
            source_row = power_lookup.loc[key]
            for column in POWER_FEATURES:
                merged.at[index, column] = source_row[column]
            merged.at[index, "power_source"] = "lag7_proxy"
            merged.at[index, "power_forecast_flag"] = 1
            proxy_rows += 1
        power_meta["power_proxy_rows"] = int(proxy_rows)
    weather_complete = merged[list(WEATHER_COLS)].notna().all(axis=1)
    merged["weather_complete"] = weather_complete
    merged["weather_day_complete"] = merged.groupby("market_date")["weather_complete"].transform("all")
    merged["power_complete"] = merged[list(POWER_FEATURES)].notna().all(axis=1)
    merged["power_day_complete"] = merged.groupby("market_date")["power_complete"].transform("all")
    coverage = {
        "price_rows": int(len(price)),
        "price_start": price.market_date.min().date().isoformat(),
        "price_end": price.market_date.max().date().isoformat(),
        "weather_rows": int(len(weather)),
        "weather_start": weather.market_date.min().date().isoformat(),
        "weather_end": weather.market_date.max().date().isoformat(),
        "weather_complete_start": (
            merged.loc[merged.weather_day_complete, "market_date"].min().date().isoformat()
            if merged.weather_day_complete.any()
            else None
        ),
        "weather_complete_end": (
            merged.loc[merged.weather_day_complete, "market_date"].max().date().isoformat()
            if merged.weather_day_complete.any()
            else None
        ),
        "weather_complete_days": int(merged.loc[merged.weather_day_complete, "market_date"].nunique()),
        "weather_only_rows_available_for_forecast": int(len(future_weather)),
        "weather_missing_cells_in_price_history": int(price.merge(weather.drop(columns=["market_date", "period"]), on="datetime", how="left")[list(WEATHER_COLS)].isna().sum().sum()),
        "weather_assumption": (
            "Workbook contains GFS forecast values but no forecast publication timestamp; "
            "the supplied values are treated as available before the target market day."
        ),
        **power_meta,
    }
    return merged, coverage


def add_features(df: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["clock_hour"] = out["period"] % 24
    out["weekday"] = out["market_date"].dt.dayofweek
    out["month"] = out["market_date"].dt.month
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["clock_hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["clock_hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["weekday"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["weekday"] / 7)
    series = np.asarray(values, dtype=float)
    for lag in LAGS:
        out[f"lag_{lag}"] = [series[i - lag] if i >= lag else np.nan for i in range(len(out))]
    out["roll24_mean"] = [np.mean(series[i - 24 : i]) if i >= 24 else np.nan for i in range(len(out))]
    out["roll24_std"] = [np.std(series[i - 24 : i]) if i >= 24 else np.nan for i in range(len(out))]
    out["roll168_mean"] = [np.mean(series[i - 168 : i]) if i >= 168 else np.nan for i in range(len(out))]
    out["roll168_std"] = [np.std(series[i - 168 : i]) if i >= 168 else np.nan for i in range(len(out))]
    temp = out["temperature"].astype(float)
    out["heating_cooling_degree"] = (18 - temp).clip(lower=0) + (temp - 24).clip(lower=0)
    out["ghi_hour_interaction"] = out["ghi"].astype(float) * np.sin(np.pi * out["clock_hour"] / 24).clip(lower=0)
    out["cloud_ghi_interaction"] = out["cloud"].astype(float) * out["ghi"].astype(float) / 100.0
    out["wind100_sq"] = out["wind100"].astype(float) ** 2
    return out


def _make_model(name: str):
    if name in {"ridge_base", "ridge_weather", "ridge_weather_power"}:
        alpha = {"ridge_base": 40.0, "ridge_weather": 70.0, "ridge_weather_power": 90.0}[name]
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])
    if name in {"extra_trees_weather", "extra_trees_weather_power"}:
        return ExtraTreesRegressor(
            n_estimators=350,
            max_features=0.75,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
    if name in {"random_forest_weather", "random_forest_weather_power"}:
        return RandomForestRegressor(
            n_estimators=300,
            max_features=0.75,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
    if name in {"hist_gradient_weather", "hist_gradient_weather_power"}:
        return HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            l2_regularization=8.0,
            random_state=42,
        )
    raise ValueError(f"unknown model: {name}")


def _prepare_xy(features: pd.DataFrame, target: pd.Series, columns: Iterable[str], train_mask: pd.Series):
    x = features[list(columns)].copy()
    median = x.loc[train_mask, list(columns)].median(numeric_only=True)
    x = x.fillna(median).fillna(0.0)
    valid = train_mask & target.notna()
    return x, target.astype(float), valid, median


def metric_dict(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted) - np.asarray(actual)
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    negative_actual = actual < 0
    high_actual = actual > 500
    return {
        "mae_yuan_per_mwh": float(np.mean(np.abs(error))),
        "rmse_yuan_per_mwh": float(np.sqrt(np.mean(error ** 2))),
        "bias_yuan_per_mwh": float(np.mean(error)),
        "negative_price_recall": float(np.sum((predicted < 0) & negative_actual) / max(1, np.sum(negative_actual))),
        "negative_price_precision": float(np.sum((predicted < 0) & negative_actual) / max(1, np.sum(predicted < 0))),
        "high_price_recall_gt_500": float(np.sum((predicted > 500) & high_actual) / max(1, np.sum(high_actual))),
    }


def _date_rows(df: pd.DataFrame, dates: list[pd.Timestamp]) -> list[int]:
    wanted = set(pd.to_datetime(dates).normalize())
    return [i for i, d in enumerate(df["market_date"]) if d in wanted]


def daily_forecast(
    df: pd.DataFrame,
    feature_table: pd.DataFrame,
    model_name: str,
    feature_columns: tuple[str, ...],
    train_end: pd.Timestamp,
    target_dates: list[pd.Timestamp],
) -> tuple[pd.DataFrame, object]:
    train_mask = (df["market_date"] <= train_end)
    if "weather" in model_name:
        train_mask &= feature_table["weather_complete"]
    if "power" in model_name:
        train_mask &= feature_table["power_complete"]
    valid_train = train_mask & feature_table[list(feature_columns)].notna().any(axis=1)
    x, y, valid_train, median = _prepare_xy(feature_table, df[TARGET], feature_columns, valid_train)
    model = _make_model(model_name)
    model.fit(x.loc[valid_train], y.loc[valid_train])
    test_indices = _date_rows(df, target_dates)
    prediction = model.predict(x.iloc[test_indices])
    rows = df.iloc[test_indices][["market_date", "period", "datetime", TARGET]].copy()
    rows["prediction"] = prediction
    rows["model"] = model_name
    rows["feature_median"] = [json.dumps({k: float(v) for k, v in median.items()}, ensure_ascii=False)] * len(rows)
    return rows.reset_index(drop=True), model


def evaluate_candidates(
    df: pd.DataFrame,
    features: pd.DataFrame,
    final_test_start: pd.Timestamp,
    include_power: bool = False,
) -> tuple[pd.DataFrame, dict[str, object], str, tuple[str, ...]]:
    all_candidates: dict[str, tuple[str, ...]] = {
        "ridge_base": BASE_FEATURES,
        "ridge_weather": ALL_FEATURES,
        "extra_trees_weather": ALL_FEATURES,
        "random_forest_weather": ALL_FEATURES,
        "hist_gradient_weather": ALL_FEATURES,
    }
    if include_power:
        all_candidates.update({
            "ridge_weather_power": ALL_POWER_FEATURES,
            "extra_trees_weather_power": ALL_POWER_FEATURES,
            "random_forest_weather_power": ALL_POWER_FEATURES,
            "hist_gradient_weather_power": ALL_POWER_FEATURES,
        })
    complete_mask = df["weather_day_complete"] & df[TARGET].notna()
    if include_power:
        complete_mask &= df["power_day_complete"]
    complete_days = sorted(pd.to_datetime(df.loc[complete_mask, "market_date"].unique()))
    tuning_days = [d for d in complete_days if pd.Timestamp("2026-05-15") <= d < final_test_start]
    # Expanding daily folds; the last two weeks are held out once for final reporting.
    fold_ranges = [(tuning_days[i], tuning_days[min(i + 6, len(tuning_days) - 1)]) for i in range(0, max(0, len(tuning_days) - 6), 7)]
    fold_rows: list[pd.DataFrame] = []
    fold_scores: dict[str, list[dict[str, float]]] = {name: [] for name in all_candidates}
    for fold_start, fold_end in fold_ranges:
        train_end = fold_start - pd.Timedelta(days=1)
        test_dates = [d for d in tuning_days if fold_start <= d <= fold_end]
        for name, columns in all_candidates.items():
            try:
                pred_rows, _ = daily_forecast(df, features, name, columns, train_end, test_dates)
            except ValueError:
                continue
            score = metric_dict(pred_rows[TARGET].to_numpy(float), pred_rows.prediction.to_numpy(float))
            score.update({"model": name, "fold_start": fold_start.date().isoformat(), "fold_end": fold_end.date().isoformat(), "train_end": train_end.date().isoformat(), "test_rows": int(len(pred_rows))})
            fold_scores[name].append(score)
            fold_rows.append(pred_rows.assign(fold_start=fold_start, fold_end=fold_end, phase="validation"))
    summary_rows = []
    for name, scores in fold_scores.items():
        if not scores:
            continue
        summary_rows.append({
            "model": name,
            "folds": len(scores),
            "validation_mae": float(np.mean([s["mae_yuan_per_mwh"] for s in scores])),
            "validation_rmse": float(np.mean([s["rmse_yuan_per_mwh"] for s in scores])),
            "validation_bias": float(np.mean([s["bias_yuan_per_mwh"] for s in scores])),
            "validation_negative_recall": float(np.mean([s["negative_price_recall"] for s in scores])),
            "validation_high_recall": float(np.mean([s["high_price_recall_gt_500"] for s in scores])),
        })
    if not summary_rows:
        raise RuntimeError("No valid weather folds were available for model selection")
    summary_rows.sort(key=lambda r: (r["validation_mae"], r["validation_rmse"]))
    ranking = pd.DataFrame(summary_rows)
    best_model = str(ranking.iloc[0]["model"])
    validation_predictions = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()

    final_dates = [d for d in complete_days if d >= final_test_start]
    final_rows = []
    final_scores = {}
    for name, columns in all_candidates.items():
        pred_rows, model = daily_forecast(df, features, name, columns, final_test_start - pd.Timedelta(days=1), final_dates)
        pred_rows = pred_rows.assign(phase="final_test")
        final_rows.append(pred_rows)
        final_scores[name] = metric_dict(pred_rows[TARGET].to_numpy(float), pred_rows.prediction.to_numpy(float))
        if name == best_model:
            selected_model = model
            selected_columns = columns
    combined_predictions = pd.concat([validation_predictions, *final_rows], ignore_index=True)
    detail = {
        "validation_ranking": ranking.to_dict(orient="records"),
        "validation_fold_scores": fold_scores,
        "final_test_start": final_test_start.date().isoformat(),
        "final_test_end": (max(final_dates).date().isoformat() if final_dates else None),
        "final_test_scores": final_scores,
        "selected_model": best_model,
        "selected_features": list(selected_columns),
        "protocol": {
            "validation": "expanding daily rolling-origin folds beginning 2026-05-15",
            "final_test": "held-out 2026-06-15 onward; no model selection uses these rows",
            "price_information": "only price history through the previous market day is used for each target day",
            "weather_information": "target-day weather workbook values are treated as pre-market forecasts per user-provided data description",
            "power_information": "target-day April-June dispatch values are treated as pre-market forecasts; January-March values are actual historical values",
        },
    }
    return combined_predictions, detail, best_model, selected_columns


def train_final_and_forecast(
    df: pd.DataFrame,
    features: pd.DataFrame,
    model_name: str,
    columns: tuple[str, ...],
    horizon_start: pd.Timestamp,
    horizon_end: pd.Timestamp,
):
    # Train on all rows available before the production horizon; only complete
    # weather rows are used for weather candidates.
    train_end = horizon_start - pd.Timedelta(days=1)
    target_dates = sorted(pd.to_datetime(df.loc[(df.market_date >= horizon_start) & (df.market_date <= horizon_end), "market_date"].unique()))
    rows, model = daily_forecast(df, features, model_name, columns, train_end, target_dates)
    residuals = rows[TARGET].to_numpy(float) - rows.prediction.to_numpy(float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 8:
        # For a genuinely future day there is no observed target to estimate
        # forecast error. Use recent hourly price volatility as a conservative
        # width and label the interval as an operational estimate.
        recent = df.loc[df[TARGET].notna(), TARGET].to_numpy(float)[-24 * 14 :]
        spread = float(np.std(np.diff(recent))) if len(recent) > 2 else 50.0
        residuals = np.asarray([-1.28 * spread, 1.28 * spread])
    rows["p10"] = np.clip(rows.prediction.to_numpy(float) + np.quantile(residuals, 0.1), -100, 1300)
    rows["p50"] = rows.prediction
    rows["p90"] = np.clip(rows.prediction.to_numpy(float) + np.quantile(residuals, 0.9), -100, 1300)
    rows["negative_risk"] = rows.p10 < 0
    rows["high_price_risk"] = rows.p90 > 500
    return rows, model


def run_weather_forecast(
    price_path: Path,
    weather_path: Path,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
    model_card_path: Path | None = None,
    power_paths: list[Path] | None = None,
) -> dict[str, object]:
    """Train the selected model and return a platform-friendly forecast payload."""
    df, coverage = load_data(price_path, weather_path, power_paths)
    feature_table = add_features(df, df[TARGET].to_numpy(float))
    card = json.loads(model_card_path.read_text(encoding="utf-8")) if model_card_path and model_card_path.exists() else {}
    selected_model = str(card.get("selected_model", "random_forest_weather"))
    selected_columns = tuple(card.get("features", ALL_FEATURES))
    rows, _ = train_final_and_forecast(df, feature_table, selected_model, selected_columns, forecast_start, forecast_end)
    rows = rows.replace({np.nan: None})
    return {
        "model_version": str(card.get("model_version", "weather-price-optimized-v1.0.0")),
        "selected_model": selected_model,
        "coverage": coverage,
        "forecast": rows.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--price", type=Path, default=Path(PRICE_DEFAULT))
    parser.add_argument("--weather", type=Path, default=Path(WEATHER_DEFAULT))
    parser.add_argument("--power-dir", type=Path, default=Path("."))
    parser.add_argument("--no-power", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/weather_power_price_optimized_202608"))
    parser.add_argument("--forecast-start", default="2026-07-01")
    parser.add_argument("--forecast-end", default="2026-07-01")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    power_paths = [] if args.no_power else sorted(args.power_dir.glob(POWER_GLOB))
    df, coverage = load_data(args.price, args.weather, power_paths)
    base_features = add_features(df, df[TARGET].to_numpy(float))
    predictions, comparison, selected_model, selected_columns = evaluate_candidates(
        df, base_features, pd.Timestamp("2026-06-15"), include_power=bool(power_paths)
    )
    final_forecast, fitted_model = train_final_and_forecast(
        df,
        base_features,
        selected_model,
        selected_columns,
        pd.Timestamp(args.forecast_start),
        pd.Timestamp(args.forecast_end),
    )
    predictions.to_csv(args.output_dir / "rolling_backtest_predictions.csv", index=False, encoding="utf-8-sig")
    final_forecast.to_csv(args.output_dir / "forecast.csv", index=False, encoding="utf-8-sig")
    if power_paths:
        power_export_columns = ["market_date", "period", *POWER_FEATURES, "power_source"]
        df[power_export_columns].drop_duplicates(["market_date", "period"]).to_csv(
            args.output_dir / "power_features_aggregated.csv", index=False, encoding="utf-8-sig"
        )
    comparison["coverage"] = coverage
    comparison["model_version"] = "weather-power-price-optimized-v1.0.0"
    comparison["target"] = "山东日前电价（元/MWh）"
    comparison["source_files"] = {"price": str(args.price), "weather": str(args.weather), "power": [str(p) for p in power_paths]}
    (args.output_dir / "model_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    artifact = {
        "model_version": "weather-power-price-optimized-v1.0.0",
        "selected_model": selected_model,
        "features": list(selected_columns),
        "coverage": coverage,
        "weather_assumption": coverage["weather_assumption"],
        "power_assumption": "April-June power files are treated as day-ahead forecasts; their publication timestamp is not present in the workbook.",
        "train_end_for_forecast": (pd.Timestamp(args.forecast_start) - pd.Timedelta(days=1)).isoformat(),
        "notes": [
            "Final test is held out from model selection.",
            "No automatic trading decision is produced by this artifact.",
            "When a target date has no power forecast file, power features use the previous week's same market period as an explicit proxy.",
        ],
    }
    (args.output_dir / "model_card.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"model": fitted_model, "features": list(selected_columns), "model_card": artifact}, args.output_dir / "weather_price_forecast_optimized.joblib")
    print(json.dumps({"selected_model": selected_model, "final_test_scores": comparison["final_test_scores"], "forecast_rows": int(len(final_forecast)), "power_files": len(power_paths), "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
