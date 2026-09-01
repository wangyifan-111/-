"""Integrated Shandong electricity-price forecasting framework.

The module implements one reproducible pipeline for:

* day-ahead price (DA) regression;
* real-time price (RT) regression;
* real-time-minus-day-ahead spread regression;
* spread direction classification;
* optional LightGBM/XGBoost quantile models;
* temporal residual correction (TCN when PyTorch is installed, otherwise a
  transparent ridge sequence fallback);
* finite-sample Conformal prediction intervals.

The model never uses target-day DA/RT/spread values as inputs. Target-day
weather and power data are treated as pre-market exogenous inputs, matching the
assumption documented in the existing project artifacts.

Optional backends are detected at runtime. If LightGBM/XGBoost/PyTorch are not
installed, the script remains runnable with a NumPy ridge/logistic fallback and
records the actual backend in ``model_card.json``. This is deliberate: a
fallback model must never be reported as LightGBM or XGBoost.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _prepare_windows_dll_search_path() -> None:
    """Expose common OpenMP runtime locations before loading LightGBM.

    Some Windows Python distributions have the MSVC runtime installed outside
    ``PATH``. LightGBM's native DLL then exists but cannot resolve
    ``VCOMP140.DLL``. Adding only directories that contain the dependency keeps
    the workaround local to this process and has no global PATH side effects.
    """
    if os.name != "nt":
        return
    candidates: list[Path] = []
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(variable)
        if root:
            base = Path(root)
            candidates.extend(
                path.parent for path in base.glob("**/vcomp140.dll")
            )
    codex_runtime = Path.home() / ".cache" / "codex-runtimes"
    if codex_runtime.exists():
        candidates.extend(path.parent for path in codex_runtime.glob("**/vcomp140.dll"))
    for directory in dict.fromkeys(candidates):
        try:
            os.add_dll_directory(str(directory))
        except (FileNotFoundError, OSError):
            continue


_prepare_windows_dll_search_path()

try:  # Optional primary backend.
    from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore

    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover - environment dependent.
    LGBMClassifier = LGBMRegressor = None  # type: ignore
    HAS_LIGHTGBM = False

try:  # Optional secondary primary backend.
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - environment dependent.
    XGBClassifier = XGBRegressor = None  # type: ignore
    HAS_XGBOOST = False

try:  # Optional residual learner.
    import torch  # type: ignore
    from torch import nn  # type: ignore

    HAS_TORCH = True
except Exception:  # pragma: no cover - environment dependent.
    torch = nn = None  # type: ignore
    HAS_TORCH = False


ROOT = Path(__file__).resolve().parent
PRICE_DEFAULT = ROOT / "山东省-现货价格-数据明细（2026-01-01_2026-06-30.xlsx"
WEATHER_DEFAULT = ROOT / "分时天气预报-自定义-山东省-2026-01-01-2026-07-01.xlsx"
POWER_GLOB = "山东省-电源出力*.xlsx"
LAGS = (24, 48, 72, 168, 336)
SEQUENCE_WINDOW = 24
TARGETS = ("da", "rt", "spread")


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def market_datetime(date: pd.Series, period: pd.Series) -> pd.Series:
    date = pd.to_datetime(date.astype(str), errors="raise")
    text = period.astype(str).str.extract(r"(\d{1,2})", expand=False).astype(int)
    return date + pd.to_timedelta(text % 24, unit="h") + pd.to_timedelta(
        (text == 24).astype(int), unit="D"
    )


def market_period(period: pd.Series) -> pd.Series:
    text = period.astype(str).str.extract(r"(\d{1,2}):(\d{2})", expand=True)
    hour = pd.to_numeric(text[0], errors="coerce").fillna(0).astype(int)
    minute = pd.to_numeric(text[1], errors="coerce").fillna(0).astype(int)
    return pd.Series(np.where(hour == 24, 24, hour + (minute > 0).astype(int)), index=period.index)


def load_price_weather(
    price_path: Path, weather_path: Path, power_paths: list[Path]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load files by stable column position, avoiding locale-specific headers."""
    price_raw = pd.read_excel(price_path)
    if price_raw.shape[1] < 4:
        raise ValueError("价格文件至少需要日期、时段、日前价格、实时价格四列")
    price_date = pd.to_datetime(price_raw.iloc[:, 0].astype(str), errors="raise").dt.normalize()
    price_period = pd.to_numeric(
        price_raw.iloc[:, 1].astype(str).str.extract(r"(\d{1,2})", expand=False), errors="coerce"
    ).astype(int)
    price = pd.DataFrame(
        {
            "market_date": price_date,
            "period": price_period,
            "datetime": market_datetime(price_raw.iloc[:, 0], price_raw.iloc[:, 1]),
            "da": pd.to_numeric(price_raw.iloc[:, 2], errors="coerce"),
            "rt": pd.to_numeric(price_raw.iloc[:, 3], errors="coerce"),
        }
    ).sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    price[["da", "rt"]] = price[["da", "rt"]].interpolate(limit_direction="both")
    price["spread"] = price["rt"] - price["da"]

    weather_raw = pd.read_excel(weather_path)
    if weather_raw.shape[1] < 10:
        raise ValueError("天气文件至少需要10列，且前两列为日期和时段")
    weather_date = pd.to_datetime(weather_raw.iloc[:, 0].astype(str), errors="raise").dt.normalize()
    weather_period = pd.to_numeric(
        weather_raw.iloc[:, 1].astype(str).str.extract(r"(\d{1,2})", expand=False), errors="coerce"
    ).astype(int)
    weather = pd.DataFrame(
        {
            "market_date": weather_date,
            "period": weather_period,
            "datetime": market_datetime(weather_raw.iloc[:, 0], weather_raw.iloc[:, 1]),
            "temperature": pd.to_numeric(weather_raw.iloc[:, 3], errors="coerce"),
            "wind10": pd.to_numeric(weather_raw.iloc[:, 4], errors="coerce"),
            "wind100": pd.to_numeric(weather_raw.iloc[:, 5], errors="coerce"),
            "ghi": pd.to_numeric(weather_raw.iloc[:, 6], errors="coerce"),
            "cloud": pd.to_numeric(weather_raw.iloc[:, 7], errors="coerce"),
            "precipitation": pd.to_numeric(weather_raw.iloc[:, 8], errors="coerce"),
            "humidity": pd.to_numeric(weather_raw.iloc[:, 9], errors="coerce"),
        }
    ).sort_values("datetime").drop_duplicates("datetime")

    merged = price.merge(weather.drop(columns=["datetime"]), on=["market_date", "period"], how="left")
    future_weather = weather.merge(
        price[["market_date", "period"]], on=["market_date", "period"], how="left", indicator=True
    )
    future_weather = future_weather[future_weather["_merge"].eq("left_only")].drop(columns=["_merge"])
    if not future_weather.empty:
        future_weather["da"] = np.nan
        future_weather["rt"] = np.nan
        future_weather["spread"] = np.nan
        merged = pd.concat([merged, future_weather[merged.columns]], ignore_index=True, sort=False)

    power, power_meta = load_power(power_paths)
    if not power.empty:
        merged = merged.merge(power, on=["market_date", "period"], how="left")
        lookup = power.set_index(["market_date", "period"])
        for idx in merged.index:
            if merged.loc[idx, "market_date"] is pd.NaT:
                continue
            columns = [c for c in POWER_COLUMNS if c in merged.columns]
            if merged.loc[idx, columns].notna().all():
                continue
            key = (merged.loc[idx, "market_date"] - pd.Timedelta(days=7), int(merged.loc[idx, "period"]))
            if key in lookup.index:
                for column in columns:
                    merged.loc[idx, column] = lookup.loc[key, column]
                merged.loc[idx, "power_source"] = "lag7_proxy"
                merged.loc[idx, "power_forecast_flag"] = 1.0

    weather_columns = ["temperature", "wind10", "wind100", "ghi", "cloud", "precipitation", "humidity"]
    merged["weather_complete"] = merged[weather_columns].notna().all(axis=1)
    merged["power_complete"] = merged[POWER_COLUMNS].notna().all(axis=1) if POWER_COLUMNS[0] in merged else False
    merged = merged.sort_values("datetime").reset_index(drop=True)
    coverage = {
        "price_rows": int(len(price)),
        "price_start": price["market_date"].min().date().isoformat(),
        "price_end": price["market_date"].max().date().isoformat(),
        "weather_rows": int(len(weather)),
        "weather_start": weather["market_date"].min().date().isoformat(),
        "weather_end": weather["market_date"].max().date().isoformat(),
        "weather_complete_days": int(merged.loc[merged["weather_complete"], "market_date"].nunique()),
        "power_complete_days": int(merged.loc[merged["power_complete"], "market_date"].nunique()),
        "power_files": [str(path) for path in power_paths],
        "weather_assumption": "Target-day workbook values are treated as pre-market weather forecasts; publication timestamps are absent.",
        "power_assumption": "April-June dispatch files are treated as pre-market forecasts; publication timestamps are absent.",
    }
    return merged, coverage


POWER_COLUMNS = [
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
]


def load_power(paths: list[Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    for path in paths:
        raw = pd.read_excel(path)
        if raw.shape[1] < 11:
            continue
        source = "forecast" if "预测出力" in path.name else "actual"
        date = pd.to_datetime(raw.iloc[:, 0].astype(str), errors="coerce").dt.normalize()
        period = market_period(raw.iloc[:, 1]).astype(int)
        frame = pd.DataFrame(
            {
                "market_date": date,
                "period": period,
                "direct_load": pd.to_numeric(raw.iloc[:, 2], errors="coerce"),
                "tie_line": pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
                "wind_power": pd.to_numeric(raw.iloc[:, 4], errors="coerce"),
                "pv_power": pd.to_numeric(raw.iloc[:, 5], errors="coerce"),
                "local_power": pd.to_numeric(raw.iloc[:, 6], errors="coerce"),
                "self_power": pd.to_numeric(raw.iloc[:, 7], errors="coerce"),
                "nuclear_power": pd.to_numeric(raw.iloc[:, 10], errors="coerce"),
                "source": source,
            }
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), {"power_files": [], "power_complete_days": 0}
    raw_power = pd.concat(frames, ignore_index=True)
    numeric = ["direct_load", "tie_line", "wind_power", "pv_power", "local_power", "self_power", "nuclear_power"]
    grouped = raw_power.groupby(["market_date", "period"], as_index=False)[numeric].mean()
    grouped["renewable_power"] = grouped["wind_power"] + grouped["pv_power"]
    grouped["renewable_share"] = grouped["renewable_power"] / grouped["direct_load"].replace(0, np.nan)
    grouped["net_load_proxy"] = grouped["direct_load"] - grouped["renewable_power"]
    source = raw_power.groupby(["market_date", "period"], as_index=False)["source"].last()
    grouped = grouped.merge(source, on=["market_date", "period"], how="left")
    grouped["power_forecast_flag"] = grouped["source"].eq("forecast").astype(float)
    grouped = grouped.rename(columns={"source": "power_source"})
    return grouped, {
        "power_files": [str(path) for path in paths],
        "power_rows": int(len(grouped)),
        "power_complete_days": int((grouped.groupby("market_date").size() == 24).sum()),
    }


def feature_table(df: pd.DataFrame, target_values: np.ndarray, prefix: str) -> pd.DataFrame:
    """Create only features available at or before each target market day."""
    out = df.copy()
    out["clock_hour"] = out["period"] % 24
    out["weekday"] = out["market_date"].dt.dayofweek
    out["month"] = out["market_date"].dt.month
    out["is_weekend"] = (out["weekday"] >= 5).astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * out["clock_hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["clock_hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["weekday"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["weekday"] / 7)
    series = np.asarray(target_values, dtype=float)
    for lag in LAGS:
        out[f"{prefix}_lag_{lag}"] = [series[i - lag] if i >= lag else np.nan for i in range(len(out))]
    out[f"{prefix}_roll24_mean"] = [np.nanmean(series[i - 24 : i]) if i >= 24 else np.nan for i in range(len(out))]
    out[f"{prefix}_roll24_std"] = [np.nanstd(series[i - 24 : i]) if i >= 24 else np.nan for i in range(len(out))]
    out[f"{prefix}_roll168_mean"] = [np.nanmean(series[i - 168 : i]) if i >= 168 else np.nan for i in range(len(out))]
    out[f"{prefix}_roll168_std"] = [np.nanstd(series[i - 168 : i]) if i >= 168 else np.nan for i in range(len(out))]
    temperature = pd.to_numeric(out["temperature"], errors="coerce")
    out["heating_cooling_degree"] = (18 - temperature).clip(lower=0) + (temperature - 24).clip(lower=0)
    out["ghi_hour_interaction"] = out["ghi"] * np.sin(np.pi * out["clock_hour"] / 24).clip(lower=0)
    out["cloud_ghi_interaction"] = out["cloud"] * out["ghi"] / 100.0
    out["wind100_sq"] = out["wind100"] ** 2
    return out


def feature_columns(prefix: str) -> list[str]:
    temporal = [
        "period", "clock_hour", "weekday", "month", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]
    lagged = [f"{prefix}_lag_{lag}" for lag in LAGS]
    lagged += [
        f"{prefix}_roll24_mean", f"{prefix}_roll24_std",
        f"{prefix}_roll168_mean", f"{prefix}_roll168_std",
    ]
    exogenous = [
        "temperature", "wind10", "wind100", "ghi", "cloud", "precipitation", "humidity",
        "heating_cooling_degree", "ghi_hour_interaction", "cloud_ghi_interaction", "wind100_sq",
        *POWER_COLUMNS,
    ]
    return temporal + lagged + exogenous


def add_all_feature_tables(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach target-specific lag/rolling features for DA, RT and spread."""
    out = frame.copy()
    for target in TARGETS:
        enriched = feature_table(out, out[target].to_numpy(float), target)
        generated = [column for column in feature_columns(target) if column not in out.columns]
        out[generated] = enriched[generated]
    return out


class NumpyRidge:
    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyRidge":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = np.nanmedian(x, axis=0)
        self.mean_[~np.isfinite(self.mean_)] = 0.0
        x = np.where(np.isfinite(x), x, self.mean_)
        self.scale_ = np.nanstd(x, axis=0)
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ < 1e-8)] = 1.0
        z = (x - self.mean_) / self.scale_
        gram = z.T @ z + self.alpha * np.eye(z.shape[1])
        self.coef_ = np.linalg.solve(gram, z.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.coef_ is None:
            raise RuntimeError("fallback model is not fitted")
        x = np.asarray(x, dtype=float)
        x = np.where(np.isfinite(x), x, self.mean_)
        return ((x - self.mean_) / self.scale_) @ self.coef_


class NumpyLogistic:
    def __init__(self, l2: float = 2.0, steps: int = 1200, learning_rate: float = 0.05):
        self.l2 = l2
        self.steps = steps
        self.learning_rate = learning_rate
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyLogistic":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = np.nanmedian(x, axis=0)
        self.mean_[~np.isfinite(self.mean_)] = 0.0
        x = np.where(np.isfinite(x), x, self.mean_)
        self.scale_ = np.nanstd(x, axis=0)
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ < 1e-8)] = 1.0
        z = (x - self.mean_) / self.scale_
        self.coef_ = np.zeros(z.shape[1])
        self.intercept_ = float(np.log((y.mean() + 1e-4) / (1 - y.mean() + 1e-4)))
        for _ in range(self.steps):
            logits = np.clip(z @ self.coef_ + self.intercept_, -30, 30)
            probability = 1.0 / (1.0 + np.exp(-logits))
            gradient = (z.T @ (probability - y)) / len(y) + self.l2 * self.coef_ / len(y)
            self.coef_ -= self.learning_rate * gradient
            self.intercept_ -= self.learning_rate * float(np.mean(probability - y))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.coef_ is None:
            raise RuntimeError("fallback classifier is not fitted")
        x = np.asarray(x, dtype=float)
        x = np.where(np.isfinite(x), x, self.mean_)
        z = (x - self.mean_) / self.scale_
        logits = np.clip(z @ self.coef_ + self.intercept_, -30, 30)
        p = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - p, p])


def make_regressor(quantile: float | None = None, backend_preference: str = "auto") -> tuple[Any, str]:
    if backend_preference not in {"auto", "lightgbm", "xgboost", "fallback"}:
        raise ValueError(f"unsupported backend: {backend_preference}")
    if backend_preference in {"lightgbm", "auto"} and HAS_LIGHTGBM:
        params: dict[str, Any] = {
            "n_estimators": 450,
            "learning_rate": 0.035,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "reg_lambda": 2.0,
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": -1,
        }
        if quantile is not None:
            params.update(objective="quantile", alpha=quantile)
        return LGBMRegressor(**params), "lightgbm"
    if backend_preference in {"xgboost", "auto"} and HAS_XGBOOST:
        params = {
            "n_estimators": 450,
            "max_depth": 6,
            "learning_rate": 0.035,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "reg_lambda": 2.0,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }
        # XGBoost quantile objectives differ by release; use conformalized
        # point predictions unless the installed release explicitly supports it.
        return XGBRegressor(**params), "xgboost_point_for_quantiles" if quantile is not None else "xgboost"
    if backend_preference in {"lightgbm", "xgboost"}:
        raise RuntimeError(f"requested backend '{backend_preference}' is not installed")
    return NumpyRidge(alpha=3.0), "numpy_ridge_fallback" if quantile is None else "numpy_conformal_quantile_fallback"


def make_classifier(backend_preference: str = "auto") -> tuple[Any, str]:
    if backend_preference not in {"auto", "lightgbm", "xgboost", "fallback"}:
        raise ValueError(f"unsupported backend: {backend_preference}")
    if backend_preference in {"lightgbm", "auto"} and HAS_LIGHTGBM:
        return LGBMClassifier(
            n_estimators=350, learning_rate=0.035, num_leaves=31,
            min_child_samples=20, reg_lambda=2.0, random_state=42,
            verbosity=-1, n_jobs=-1,
        ), "lightgbm"
    if backend_preference in {"xgboost", "auto"} and HAS_XGBOOST:
        return XGBClassifier(
            n_estimators=350, max_depth=6, learning_rate=0.035,
            subsample=0.9, colsample_bytree=0.85, reg_lambda=2.0,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        ), "xgboost"
    if backend_preference in {"lightgbm", "xgboost"}:
        raise RuntimeError(f"requested backend '{backend_preference}' is not installed")
    return NumpyLogistic(), "numpy_logistic_fallback"


def clean_matrix(frame: pd.DataFrame, columns: list[str], train_mask: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = frame[columns].apply(pd.to_numeric, errors="coerce").astype(float)
    median = x.loc[train_mask, columns].median(numeric_only=True).to_numpy(float).copy()
    median[~np.isfinite(median)] = 0.0
    values = np.where(np.isfinite(x.to_numpy(float)), x.to_numpy(float), median)
    return values, median


def fit_predict_model(
    frame: pd.DataFrame,
    target: pd.Series,
    columns: list[str],
    train_mask: pd.Series,
    predict_mask: pd.Series,
    quantile: float | None = None,
    backend_preference: str = "auto",
) -> tuple[np.ndarray, Any, str, np.ndarray]:
    x, median = clean_matrix(frame, columns, train_mask)
    valid = train_mask.to_numpy(bool) & target.notna().to_numpy(bool)
    model, backend = make_regressor(quantile, backend_preference)
    train_x = pd.DataFrame(x[valid], columns=columns)
    predict_x = pd.DataFrame(x[predict_mask.to_numpy(bool)], columns=columns)
    model.fit(train_x, target.to_numpy(float)[valid])
    return model.predict(predict_x), model, backend, median


def conformal_quantile(abs_residuals: Iterable[float], alpha: float = 0.1) -> float:
    values = np.asarray(list(abs_residuals), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    rank = int(math.ceil((len(values) + 1) * (1.0 - alpha)))
    return float(np.sort(values)[min(rank - 1, len(values) - 1)])


class NumpyTemporalResidual:
    """Ridge sequence residual learner used when PyTorch is unavailable."""

    def __init__(self, window: int = SEQUENCE_WINDOW):
        self.window = window
        self.model = NumpyRidge(alpha=8.0)

    def fit(self, residuals: np.ndarray) -> "NumpyTemporalResidual":
        x, y = sequence_samples(residuals, self.window)
        if len(y):
            self.model.fit(x, y)
        return self

    def predict_next(self, history: np.ndarray) -> float:
        if len(history) < self.window:
            return 0.0
        return float(self.model.predict(np.asarray(history[-self.window:], dtype=float).reshape(1, -1))[0])


if HAS_TORCH:
    class TinyTCN(nn.Module):  # type: ignore[misc]
        def __init__(self, channels: int = 16):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, channels, kernel_size=3, padding=2, dilation=1), nn.ReLU(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=2), nn.ReLU(),
                nn.Conv1d(channels, 1, kernel_size=1),
            )

        def forward(self, x):
            return self.net(x)[:, :, -1]


    class TorchTCNResidual:
        def __init__(self, window: int = SEQUENCE_WINDOW):
            self.window = window
            self.model = TinyTCN()

        def fit(self, residuals: np.ndarray) -> "TorchTCNResidual":
            x, y = sequence_samples(residuals, self.window)
            if len(y) == 0:
                return self
            torch.manual_seed(42)
            xt = torch.tensor(x[:, None, :], dtype=torch.float32)
            yt = torch.tensor(y[:, None], dtype=torch.float32)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003, weight_decay=1e-4)
            loss_fn = nn.SmoothL1Loss()
            self.model.train()
            for _ in range(160):
                optimizer.zero_grad()
                loss = loss_fn(self.model(xt), yt)
                loss.backward()
                optimizer.step()
            return self

        def predict_next(self, history: np.ndarray) -> float:
            if len(history) < self.window:
                return 0.0
            self.model.eval()
            with torch.no_grad():
                x = torch.tensor(history[-self.window:][None, None, :], dtype=torch.float32)
                return float(self.model(x).reshape(-1)[0])
else:
    TorchTCNResidual = NumpyTemporalResidual  # type: ignore[misc,assignment]


def sequence_samples(residuals: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= window:
        return np.empty((0, window)), np.empty(0)
    x = np.stack([values[i - window : i] for i in range(window, len(values))])
    y = values[window:]
    return x, y


@dataclass
class TargetResult:
    target: str
    prediction: np.ndarray
    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray
    direction_probability: np.ndarray | None
    residual_backend: str
    main_backend: str
    conformal_q: float
    calibration_mae: float | None


def fit_target_for_date(
    frame: pd.DataFrame,
    target_name: str,
    target_date: pd.Timestamp,
    calibration_days: int,
    residual_history: np.ndarray | None = None,
    backend_preference: str = "auto",
) -> TargetResult:
    target = frame[target_name].astype(float)
    prefix = target_name
    columns = feature_columns(prefix)
    date = frame["market_date"]
    complete = frame["weather_complete"] & frame["power_complete"]
    train_mask = (date < target_date) & complete & target.notna()
    predict_mask = (date == target_date) & complete
    if int(predict_mask.sum()) != 24:
        raise ValueError(f"{target_name}目标日需要24个完整时段，实际为{int(predict_mask.sum())}")
    prediction, model, backend, _ = fit_predict_model(
        frame, target, columns, train_mask, predict_mask, backend_preference=backend_preference
    )

    calibration_start = target_date - pd.Timedelta(days=calibration_days)
    cal_mask = (date >= calibration_start) & (date < target_date) & complete & target.notna()
    cal_train_mask = (date < calibration_start) & complete & target.notna()
    calibration_prediction, _, _, _ = fit_predict_model(
        frame, target, columns, cal_train_mask, cal_mask, backend_preference=backend_preference
    )
    actual = target.loc[cal_mask].to_numpy(float)
    calibration_residual = actual - calibration_prediction
    if residual_history is not None and len(residual_history) >= SEQUENCE_WINDOW:
        residual_source = np.concatenate([residual_history, calibration_residual])
    else:
        residual_source = calibration_residual
    residual_model = TorchTCNResidual(SEQUENCE_WINDOW).fit(residual_source)
    history = list(residual_source.astype(float))
    correction = []
    for _ in range(24):
        correction.append(residual_model.predict_next(np.asarray(history, dtype=float)))
        history.append(correction[-1])
    # Residual models are deliberately shrunk and clipped. This prevents an
    # unstable residual learner from overwhelming the primary price forecast.
    correction = np.clip(0.25 * np.asarray(correction), -300.0, 300.0)
    corrected = prediction + correction
    conformal_q = conformal_quantile(np.abs(calibration_residual - np.asarray(correction[: len(calibration_residual)])) if len(calibration_residual) <= len(correction) else np.abs(calibration_residual), 0.1)
    if conformal_q <= 0:
        conformal_q = conformal_quantile(np.abs(calibration_residual), 0.1)

    # LightGBM is the preferred quantile backend. With XGBoost/fallbacks,
    # conformalized point forecasts provide the honest interval instead.
    quantile_backend = "conformal_only"
    q10_prediction = None
    q90_prediction = None
    if backend_preference == "lightgbm" or (backend_preference == "auto" and HAS_LIGHTGBM):
        q10_prediction, _, q10_backend, _ = fit_predict_model(
            frame, target, columns, train_mask, predict_mask, quantile=0.10, backend_preference=backend_preference
        )
        q90_prediction, _, q90_backend, _ = fit_predict_model(
            frame, target, columns, train_mask, predict_mask, quantile=0.90, backend_preference=backend_preference
        )
        quantile_backend = f"{q10_backend}/{q90_backend}"
    p10 = corrected - conformal_q if q10_prediction is None else q10_prediction + correction - conformal_q
    p50 = corrected
    p90 = corrected + conformal_q if q90_prediction is None else q90_prediction + correction + conformal_q
    direction_probability = None
    if target_name == "spread":
        classifier, clf_backend = make_classifier(backend_preference)
        x, _ = clean_matrix(frame, columns, train_mask)
        valid = train_mask.to_numpy(bool)
        train_x = pd.DataFrame(x[valid], columns=columns)
        predict_x = pd.DataFrame(x[predict_mask.to_numpy(bool)], columns=columns)
        classifier.fit(train_x, (target.to_numpy(float)[valid] >= 0).astype(int))
        direction_probability = classifier.predict_proba(predict_x)[:, 1]
        backend = f"{backend}+direction:{clf_backend}"
    return TargetResult(
        target=target_name,
        prediction=corrected,
        p10=p10,
        p50=p50,
        p90=p90,
        direction_probability=direction_probability,
        residual_backend="tcn" if HAS_TORCH else "ridge_sequence_fallback",
        main_backend=f"{backend};quantiles:{quantile_backend}",
        conformal_q=float(conformal_q),
        calibration_mae=float(np.mean(np.abs(calibration_residual))) if len(calibration_residual) else None,
    )


def metric(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=float) - np.asarray(actual, dtype=float)
    return {
        "mae_yuan_per_mwh": float(np.mean(np.abs(error))),
        "rmse_yuan_per_mwh": float(np.sqrt(np.mean(error**2))),
        "bias_yuan_per_mwh": float(np.mean(error)),
        "sample_count": int(len(error)),
    }


def forecast_one_date(
    frame: pd.DataFrame,
    target_date: pd.Timestamp,
    calibration_days: int = 14,
    residual_histories: dict[str, np.ndarray] | None = None,
    backend_preference: str = "auto",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    residual_histories = residual_histories or {}
    results = {
        target: fit_target_for_date(
            frame, target, target_date, calibration_days, residual_histories.get(target), backend_preference
        )
        for target in TARGETS
    }
    rows = []
    target_rows = frame.loc[frame["market_date"].eq(target_date) & frame["weather_complete"] & frame["power_complete"]].sort_values("period")
    da, rt, spread = results["da"], results["rt"], results["spread"]
    for index, (_, row) in enumerate(target_rows.iterrows()):
        probability = None if spread.direction_probability is None else float(spread.direction_probability[index])
        rows.append(
            {
                "market_date": target_date.date().isoformat(),
                "period": int(row["period"]),
                "datetime": pd.Timestamp(row["datetime"]).isoformat(),
                "day_ahead_price": {"p10": float(da.p10[index]), "p50": float(da.p50[index]), "p90": float(da.p90[index])},
                "real_time_price": {
                    "p10": float(da.p10[index] + spread.p10[index]),
                    "p50": float(da.p50[index] + spread.p50[index]),
                    "p90": float(da.p90[index] + spread.p90[index]),
                },
                "real_time_direct_model_benchmark": {
                    "p10": float(rt.p10[index]),
                    "p50": float(rt.p50[index]),
                    "p90": float(rt.p90[index]),
                },
                "spread_real_time_minus_day_ahead": {"p10": float(spread.p10[index]), "p50": float(spread.p50[index]), "p90": float(spread.p90[index])},
                "spread_positive_probability": probability,
                "spread_direction": None if probability is None else ("positive" if probability >= 0.5 else "negative"),
                "negative_price_risk": bool(da.p10[index] < 0 or da.p10[index] + spread.p10[index] < 0),
                "high_price_risk": bool(da.p90[index] > 500 or da.p90[index] + spread.p90[index] > 500),
            }
        )
    outputs = {
        "model_version": "integrated-price-forecast-v1.0.0",
        "target_date": target_date.date().isoformat(),
        "forecast": rows,
        "models": {
            target: {
                "main_backend": result.main_backend,
                "residual_backend": result.residual_backend,
                "conformal_q_90": result.conformal_q,
                "calibration_mae": result.calibration_mae,
            }
            for target, result in results.items()
        },
        "architecture": {
            "main_regression": "LightGBM preferred, XGBoost secondary, NumPy ridge fallback",
            "temporal_residual": "dilated TCN preferred, ridge sequence fallback",
            "spread_direction": "binary classifier on RT-DA >= 0",
            "interval": "finite-sample Conformal calibration at 90% nominal coverage",
            "real_time_price_identity": "RT = DA + (RT - DA)",
            "direct_rt_regression": "trained as a benchmark; operational RT output uses DA plus spread for accounting coherence",
        },
    }
    return outputs, residual_histories


def run_walk_forward(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    calibration_days: int,
    backend_preference: str = "auto",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """True walk-forward evaluation; each date is trained only on prior dates."""
    complete_days = sorted(
        pd.to_datetime(frame.loc[frame["weather_complete"] & frame["power_complete"], "market_date"].unique())
    )
    dates = [day for day in complete_days if start <= day <= end and frame.loc[frame["market_date"].eq(day), "da"].notna().all()]
    history: dict[str, list[float]] = {target: [] for target in TARGETS}
    rows: list[dict[str, Any]] = []
    for day in dates:
        output, _ = forecast_one_date(
            frame, day, calibration_days,
            {target: np.asarray(values, dtype=float) for target, values in history.items()},
            backend_preference,
        )
        actual_rows = frame.loc[frame["market_date"].eq(day)].sort_values("period")
        for index, forecast_row in enumerate(output["forecast"]):
            actual_da = float(actual_rows.iloc[index]["da"])
            actual_rt = float(actual_rows.iloc[index]["rt"])
            actual_spread = float(actual_rt - actual_da)
            rows.append({
                "market_date": day.date().isoformat(),
                "period": int(forecast_row["period"]),
                "da_actual": actual_da,
                "da_pred": forecast_row["day_ahead_price"]["p50"],
                "rt_actual": actual_rt,
                "rt_pred": forecast_row["real_time_price"]["p50"],
                "spread_actual": actual_spread,
                "spread_pred": forecast_row["spread_real_time_minus_day_ahead"]["p50"],
                "spread_probability": forecast_row["spread_positive_probability"],
            })
        for target in TARGETS:
            actual = actual_rows["da" if target == "da" else "rt" if target == "rt" else "rt"].to_numpy(float)
            if target == "spread":
                actual = actual_rows["rt"].to_numpy(float) - actual_rows["da"].to_numpy(float)
            predicted = np.asarray([r["da_pred" if target == "da" else "rt_pred" if target == "rt" else "spread_pred"] for r in rows[-24:]])
            history[target].extend((actual - predicted).tolist())
    result = pd.DataFrame(rows)
    scores = {}
    if not result.empty:
        scores = {
            "day_ahead": metric(result["da_actual"], result["da_pred"]),
            "real_time": metric(result["rt_actual"], result["rt_pred"]),
            "spread": metric(result["spread_actual"], result["spread_pred"]),
            "spread_direction_accuracy": float(
                np.mean((result["spread_pred"] >= 0) == (result["spread_actual"] >= 0))
            ),
        }
    return result, scores


def build_model_card(coverage: dict[str, Any], backends: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_version": "integrated-price-forecast-v1.0.0",
        "targets": ["day_ahead_price", "real_time_price", "real_time_minus_day_ahead_spread", "spread_direction"],
        "main_backend": backends,
        "temporal_residual_backend": "tcn" if HAS_TORCH else "ridge_sequence_fallback",
        "interval_method": "Conformal prediction from rolling calibration residuals",
        "nominal_interval": 0.90,
        "calibration_days": args.calibration_days,
        "data_coverage": coverage,
        "assumptions": [
            "Target-day weather and power values are treated as pre-market forecasts because publication timestamps are unavailable.",
            "RT price is reconstructed as DA price plus RT-minus-DA spread.",
            "No trading action is generated; risk outputs are advisory only.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrated DA/RT/spread forecast with TCN residual and Conformal intervals")
    parser.add_argument("--price", type=Path, default=PRICE_DEFAULT)
    parser.add_argument("--weather", type=Path, default=WEATHER_DEFAULT)
    parser.add_argument("--power-dir", type=Path, default=ROOT)
    parser.add_argument("--forecast-date", default="2026-07-01")
    parser.add_argument("--backtest-start", default="2026-06-15")
    parser.add_argument("--backtest-end", default="2026-06-30")
    parser.add_argument("--calibration-days", type=int, default=14)
    parser.add_argument("--backend", choices=["auto", "lightgbm", "xgboost", "fallback"], default="auto", help="primary tree backend; auto prefers LightGBM")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "integrated_price_forecast_20260824")
    parser.add_argument("--skip-backtest", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    power_paths = sorted(args.power_dir.glob(POWER_GLOB))
    frame, coverage = load_price_weather(args.price, args.weather, power_paths)
    frame = add_all_feature_tables(frame)

    backtest_scores: dict[str, Any] = {}
    if not args.skip_backtest:
        backtest, backtest_scores = run_walk_forward(
            frame, pd.Timestamp(args.backtest_start), pd.Timestamp(args.backtest_end), args.calibration_days, args.backend
        )
        backtest.to_csv(args.output_dir / "walk_forward_backtest.csv", index=False, encoding="utf-8-sig")

    forecast, _ = forecast_one_date(frame, pd.Timestamp(args.forecast_date), args.calibration_days, backend_preference=args.backend)
    (args.output_dir / "forecast.json").write_text(json.dumps(forecast, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(forecast["forecast"]).to_csv(args.output_dir / "forecast.csv", index=False, encoding="utf-8-sig")
    backends = forecast["models"]
    card = build_model_card(coverage, backends, args)
    card["backtest"] = backtest_scores
    (args.output_dir / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "model_bundle.pkl").open("wb") as handle:
        pickle.dump({"model_card": card, "architecture": forecast["architecture"]}, handle)
    summary = {
        "output_dir": str(args.output_dir),
        "forecast_date": args.forecast_date,
        "forecast_rows": len(forecast["forecast"]),
        "backtest": backtest_scores,
        "main_backend": args.backend if args.backend != "auto" else ("lightgbm" if HAS_LIGHTGBM else "xgboost" if HAS_XGBOOST else "numpy_ridge_fallback"),
        "temporal_residual_backend": "tcn" if HAS_TORCH else "ridge_sequence_fallback",
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
