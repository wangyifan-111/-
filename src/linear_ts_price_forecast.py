"""Leakage-safe DLinear/NLinear trials for the Shandong electricity forecast.

The two models are compact long-sequence forecasting baselines inspired by
LTSF-Linear (Zeng et al., AAAI 2023).  They forecast DA and RT-DA spread
together from the preceding 168 hours and known target-day variables.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import integrated_price_forecast as base
from transformer_price_forecast import (
    FUTURE_COLUMNS,
    PAST_COLUMNS,
    complete_market_days,
    daily_arrays,
    fit_scales,
)

CONTEXT_HOURS = 168
HORIZON = 24


@dataclass(frozen=True)
class LinearConfig:
    name: str
    family: str
    moving_window: int = 25
    dropout: float = 0.05


class MovingAverage(nn.Module):
    def __init__(self, kernel: int):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("moving_window must be odd")
        self.kernel = kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, channels]. Replication avoids losing edge hours.
        pad = (self.kernel - 1) // 2
        y = x.transpose(1, 2)
        y = torch.nn.functional.pad(y, (pad, pad), mode="replicate")
        y = torch.nn.functional.avg_pool1d(y, kernel_size=self.kernel, stride=1)
        return y.transpose(1, 2)


class DLinearForecast(nn.Module):
    """Trend/seasonal decomposition followed by direct multi-step linear maps."""

    def __init__(self, past_dim: int, future_dim: int, window: int, dropout: float):
        super().__init__()
        self.decompose = MovingAverage(window)
        # A separate 168 -> 24 map per historical variable, followed by a
        # small cross-variable projection to the two requested targets.
        self.seasonal = nn.Linear(CONTEXT_HOURS, HORIZON)
        self.trend = nn.Linear(CONTEXT_HOURS, HORIZON)
        self.channel_head = nn.Linear(past_dim, 2)
        self.future_head = nn.Sequential(nn.Linear(future_dim, 16), nn.GELU(), nn.Dropout(dropout), nn.Linear(16, 2))

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        trend = self.decompose(past)
        seasonal = past - trend
        seasonal_out = self.seasonal(seasonal.transpose(1, 2)).transpose(1, 2)
        trend_out = self.trend(trend.transpose(1, 2)).transpose(1, 2)
        linear_out = self.channel_head(seasonal_out + trend_out)
        return linear_out + self.future_head(future)


class NLinearForecast(nn.Module):
    """Normalized linear model: subtract the last value before forecasting."""

    def __init__(self, past_dim: int, future_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(CONTEXT_HOURS, HORIZON)
        self.channel_head = nn.Linear(past_dim, 2)
        self.future_head = nn.Sequential(nn.Linear(future_dim, 16), nn.GELU(), nn.Dropout(dropout), nn.Linear(16, 2))

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        last = past[:, -1:, :]
        normalized = past - last
        out = self.linear(normalized.transpose(1, 2)).transpose(1, 2)
        out = self.channel_head(out)
        # The target and input tensors use separate feature scalers, so the
        # historical last value cannot be added directly in normalized space.
        # The head learns this level shift from the normalized sequence instead.
        return out + self.future_head(future)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model(past_dim: int, future_dim: int, config: LinearConfig) -> nn.Module:
    if config.family == "dlinear":
        return DLinearForecast(past_dim, future_dim, config.moving_window, config.dropout)
    if config.family == "nlinear":
        return NLinearForecast(past_dim, future_dim, config.dropout)
    raise ValueError(config.family)


def train_model(
    past: np.ndarray,
    future: np.ndarray,
    target: np.ndarray,
    config: LinearConfig,
    epochs: int,
    seed: int,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    patience: int = 15,
) -> tuple[nn.Module, int, float]:
    seed_everything(seed)
    model = make_model(past.shape[2], future.shape[2], config)
    dataset = TensorDataset(torch.tensor(past, dtype=torch.float32), torch.tensor(future, dtype=torch.float32), torch.tensor(target, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-3)
    loss_fn = nn.SmoothL1Loss(beta=0.75)
    best_state, best_loss, best_epoch, wait = copy.deepcopy(model.state_dict()), float("inf"), 1, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_past, batch_future, batch_target in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_past, batch_future), batch_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            if validation is None:
                current = float(loss.detach())
            else:
                vp, vf, vt = validation
                current = float(loss_fn(model(torch.tensor(vp, dtype=torch.float32), torch.tensor(vf, dtype=torch.float32)), torch.tensor(vt, dtype=torch.float32)))
        if current < best_loss - 1e-5:
            best_state, best_loss, best_epoch, wait = copy.deepcopy(model.state_dict()), current, epoch, 0
        else:
            wait += 1
        if validation is not None and wait >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_epoch, best_loss


def predict(model: nn.Module, past: np.ndarray, future: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(past, dtype=torch.float32), torch.tensor(future, dtype=torch.float32)).cpu().numpy()


def candidate_configs(family: str) -> list[LinearConfig]:
    if family == "dlinear":
        return [LinearConfig("dlinear_13", family, 13), LinearConfig("dlinear_25", family, 25), LinearConfig("dlinear_49", family, 49)]
    if family == "nlinear":
        return [LinearConfig("nlinear_small", family, 1), LinearConfig("nlinear_dropout", family, 1, 0.10)]
    raise ValueError(family)


def tune_architecture(frame: pd.DataFrame, family: str, tune_start: pd.Timestamp, tune_end: pd.Timestamp, max_epochs: int) -> tuple[LinearConfig, int, list[dict[str, Any]]]:
    days = complete_market_days(frame)
    train_days = [day for day in days if day < tune_start]
    val_days = [day for day in days if tune_start <= day <= tune_end]
    train_past, train_future, train_target, _ = daily_arrays(frame, train_days)
    val_past, val_future, val_target, _ = daily_arrays(frame, val_days)
    ps, fs, ts = fit_scales(train_past, train_future, train_target)
    train_scaled = (ps.transform(train_past), fs.transform(train_future), ts.transform(train_target))
    val_scaled = (ps.transform(val_past), fs.transform(val_future), ts.transform(val_target))
    results: list[dict[str, Any]] = []
    best: tuple[float, LinearConfig, int] | None = None
    for i, config in enumerate(candidate_configs(family)):
        model, epoch, loss = train_model(*train_scaled, config, max_epochs, 22000 + i, validation=val_scaled)
        pred = ts.inverse(predict(model, val_scaled[0], val_scaled[1]))
        da = base.metric(val_target[:, :, 0].ravel(), pred[:, :, 0].ravel())
        spread = base.metric(val_target[:, :, 1].ravel(), pred[:, :, 1].ravel())
        rt = base.metric((val_target[:, :, 0] + val_target[:, :, 1]).ravel(), (pred[:, :, 0] + pred[:, :, 1]).ravel())
        composite = float(np.mean([da["mae_yuan_per_mwh"], spread["mae_yuan_per_mwh"], rt["mae_yuan_per_mwh"]]))
        result = {"config": config.__dict__, "best_epoch": epoch, "validation_loss_scaled": loss, "day_ahead": da, "spread": spread, "real_time_coherent": rt, "composite_mae": composite}
        results.append(result)
        if best is None or composite < best[0]:
            best = (composite, config, max(8, epoch))
    assert best is not None
    return best[1], best[2], results


def fit_predict_day(frame: pd.DataFrame, target_date: pd.Timestamp, config: LinearConfig, epochs: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    train_days = [day for day in complete_market_days(frame) if day < target_date]
    tp, tf, tt, used = daily_arrays(frame, train_days)
    fp, ff, _, _ = daily_arrays(frame, [target_date], require_targets=False)
    ps, fs, ts = fit_scales(tp, tf, tt)
    model, _, loss = train_model(ps.transform(tp), fs.transform(tf), ts.transform(tt), config, epochs, seed)
    pred = ts.inverse(predict(model, ps.transform(fp), fs.transform(ff)))[0]
    return pred, {"train_days": len(used), "train_start": used[0].date().isoformat(), "train_end": used[-1].date().isoformat(), "epochs": epochs, "final_scaled_loss": loss}


def run_walk_forward(frame: pd.DataFrame, family: str, config: LinearConfig, epochs: int, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    days = [day for day in complete_market_days(frame) if start <= day <= end]
    rows: list[dict[str, Any]] = []
    for fold, day in enumerate(days):
        forecast, _ = fit_predict_day(frame, day, config, epochs, 23000 + fold)
        actual = frame.loc[frame["market_date"].eq(day)].sort_values("period")
        for h in range(24):
            da_pred, spread_pred = map(float, forecast[h])
            rows.append({"market_date": day.date().isoformat(), "period": h + 1, "da_actual": float(actual.iloc[h]["da"]), f"da_{family}_pred": da_pred, "spread_actual": float(actual.iloc[h]["spread"]), f"spread_{family}_pred": spread_pred, "rt_actual": float(actual.iloc[h]["rt"]), f"rt_{family}_pred": da_pred + spread_pred})
        print(f"completed {family.upper()} fold {fold + 1}/{len(days)}: {day.date().isoformat()}", flush=True)
    result = pd.DataFrame(rows)
    scores = {"day_ahead": base.metric(result["da_actual"], result[f"da_{family}_pred"]), "spread": base.metric(result["spread_actual"], result[f"spread_{family}_pred"]), "real_time_coherent": base.metric(result["rt_actual"], result[f"rt_{family}_pred"]), "spread_direction_accuracy": float(((result[f"spread_{family}_pred"] >= 0) == (result["spread_actual"] >= 0)).mean())}
    return result, {"scores": scores}


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe DLinear/NLinear electricity price trials")
    parser.add_argument("--tune-start", default="2026-06-08")
    parser.add_argument("--tune-end", default="2026-06-14")
    parser.add_argument("--backtest-start", default="2026-06-15")
    parser.add_argument("--backtest-end", default="2026-06-30")
    parser.add_argument("--forecast-date", default="2026-07-01")
    parser.add_argument("--max-tune-epochs", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=base.ROOT / "outputs" / "linear_ts_trial_20260829")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    frame, coverage = base.load_price_weather(base.PRICE_DEFAULT, base.WEATHER_DEFAULT, sorted(base.ROOT.glob(base.POWER_GLOB)))
    frame = base.add_all_feature_tables(frame)
    summary: dict[str, Any] = {"models": {}, "tuning_period": {"start": args.tune_start, "end": args.tune_end}, "backtest_period": {"start": args.backtest_start, "end": args.backtest_end}, "data_coverage": coverage}
    for family in ("dlinear", "nlinear"):
        config, epochs, tuning = tune_architecture(frame, family, pd.Timestamp(args.tune_start), pd.Timestamp(args.tune_end), args.max_tune_epochs)
        print(f"selected {config.name} with {epochs} epochs", flush=True)
        backtest, details = run_walk_forward(frame, family, config, epochs, pd.Timestamp(args.backtest_start), pd.Timestamp(args.backtest_end))
        backtest.to_csv(args.output_dir / f"{family}_walk_forward_predictions.csv", index=False, encoding="utf-8-sig")
        forecast, info = fit_predict_day(frame, pd.Timestamp(args.forecast_date), config, epochs, 24000 + (family == "nlinear"))
        pd.DataFrame([{"market_date": args.forecast_date, "period": h + 1, "day_ahead_price": float(forecast[h, 0]), "spread_real_time_minus_day_ahead": float(forecast[h, 1]), "real_time_price": float(forecast[h, 0] + forecast[h, 1])} for h in range(24)]).to_csv(args.output_dir / f"{family}_forecast.csv", index=False, encoding="utf-8-sig")
        summary["models"][family] = {"selected_config": config.__dict__, "selected_epochs": epochs, "tuning": tuning, "backtest": details["scores"], "forecast_training": info}
    summary["leakage_controls"] = ["Architecture selection ends before the reported backtest.", "Each fold trains only on dates before its target date.", "Target-day inputs are limited to known calendar, weather and power variables."]
    (args.output_dir / "linear_ts_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "models": {k: v["backtest"] for k, v in summary["models"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
