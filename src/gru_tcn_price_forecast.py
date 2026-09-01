"""Comparable GRU and residual TCN trials for the daily electricity forecast."""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import integrated_price_forecast as base
from transformer_price_forecast import complete_market_days, daily_arrays, fit_scales, seed_everything


@dataclass(frozen=True)
class TrialConfig:
    name: str
    family: str
    hidden: int
    layers: int
    kernel: int = 3
    dropout: float = 0.10


class GRUForecast(nn.Module):
    """GRU encoder over the historical sequence plus known target-day inputs."""

    def __init__(self, past_dim: int, future_dim: int, config: TrialConfig):
        super().__init__()
        self.encoder = nn.GRU(
            input_size=past_dim,
            hidden_size=config.hidden,
            num_layers=config.layers,
            batch_first=True,
            dropout=config.dropout if config.layers > 1 else 0.0,
        )
        self.future_projection = nn.Sequential(nn.Linear(future_dim, config.hidden), nn.GELU())
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden * 2, config.hidden), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.hidden, 2)
        )

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(past)
        context = hidden[-1].unsqueeze(1).expand(-1, future.shape[1], -1)
        return self.decoder(torch.cat([context, self.future_projection(future)], dim=2))


class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        y = self.dropout(self.activation(self.norm1(self.conv1(x))))
        y = self.dropout(self.activation(self.norm2(self.conv2(y))))
        return self.activation(y + residual)


class TCNForecast(nn.Module):
    """Residual dilated TCN encoder with a known-exogenous 24-hour decoder."""

    def __init__(self, past_dim: int, future_dim: int, config: TrialConfig):
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = past_dim
        for layer in range(config.layers):
            blocks.append(TCNResidualBlock(in_channels, config.hidden, config.kernel, 2**layer, config.dropout))
            in_channels = config.hidden
        self.encoder = nn.Sequential(*blocks)
        self.future_projection = nn.Sequential(nn.Linear(future_dim, config.hidden), nn.GELU())
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden * 3, config.hidden), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.hidden, 2)
        )

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(past.transpose(1, 2))
        context = torch.cat([encoded.mean(dim=2), encoded[:, :, -1]], dim=1)
        context = context.unsqueeze(1).expand(-1, future.shape[1], -1)
        return self.decoder(torch.cat([context, self.future_projection(future)], dim=2))


def make_model(past_dim: int, future_dim: int, config: TrialConfig) -> nn.Module:
    if config.family == "gru":
        return GRUForecast(past_dim, future_dim, config)
    if config.family == "tcn":
        return TCNForecast(past_dim, future_dim, config)
    raise ValueError(config.family)


def train_model(
    past: np.ndarray, future: np.ndarray, target: np.ndarray, config: TrialConfig,
    epochs: int, seed: int, validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    patience: int = 15,
) -> tuple[nn.Module, int, float]:
    seed_everything(seed)
    model = make_model(past.shape[2], future.shape[2], config)
    dataset = TensorDataset(
        torch.tensor(past, dtype=torch.float32), torch.tensor(future, dtype=torch.float32),
        torch.tensor(target, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
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
                val_past, val_future, val_target = validation
                current = float(loss_fn(
                    model(torch.tensor(val_past, dtype=torch.float32), torch.tensor(val_future, dtype=torch.float32)),
                    torch.tensor(val_target, dtype=torch.float32)
                ))
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


def candidate_configs(family: str) -> list[TrialConfig]:
    if family == "gru":
        return [TrialConfig("gru_small", family, 24, 1), TrialConfig("gru_medium", family, 40, 1), TrialConfig("gru_deep", family, 32, 2)]
    if family == "tcn":
        return [TrialConfig("tcn_small", family, 24, 2, 3), TrialConfig("tcn_wide", family, 40, 2, 5), TrialConfig("tcn_deep", family, 32, 3, 3)]
    raise ValueError(family)


def tune_architecture(frame: pd.DataFrame, family: str, tune_start: pd.Timestamp, tune_end: pd.Timestamp, max_epochs: int) -> tuple[TrialConfig, int, list[dict[str, Any]]]:
    days = complete_market_days(frame)
    train_days = [day for day in days if day < tune_start]
    val_days = [day for day in days if tune_start <= day <= tune_end]
    train_past, train_future, train_target, _ = daily_arrays(frame, train_days)
    val_past, val_future, val_target, _ = daily_arrays(frame, val_days)
    past_scale, future_scale, target_scale = fit_scales(train_past, train_future, train_target)
    train_scaled = (past_scale.transform(train_past), future_scale.transform(train_future), target_scale.transform(train_target))
    val_scaled = (past_scale.transform(val_past), future_scale.transform(val_future), target_scale.transform(val_target))
    results: list[dict[str, Any]] = []
    best: tuple[float, TrialConfig, int] | None = None
    for index, config in enumerate(candidate_configs(family)):
        model, best_epoch, val_loss = train_model(*train_scaled, config=config, epochs=max_epochs, seed=12500 + index, validation=val_scaled)
        forecast = target_scale.inverse(predict(model, val_scaled[0], val_scaled[1]))
        da = base.metric(val_target[:, :, 0].reshape(-1), forecast[:, :, 0].reshape(-1))
        spread = base.metric(val_target[:, :, 1].reshape(-1), forecast[:, :, 1].reshape(-1))
        rt = base.metric((val_target[:, :, 0] + val_target[:, :, 1]).reshape(-1), (forecast[:, :, 0] + forecast[:, :, 1]).reshape(-1))
        composite = float(np.mean([da["mae_yuan_per_mwh"], spread["mae_yuan_per_mwh"], rt["mae_yuan_per_mwh"]]))
        result = {"config": config.__dict__, "best_epoch": best_epoch, "validation_loss_scaled": val_loss, "day_ahead": da, "spread": spread, "real_time_coherent": rt, "composite_mae": composite}
        results.append(result)
        if best is None or composite < best[0]:
            best = (composite, config, best_epoch)
    assert best is not None
    return best[1], max(10, best[2]), results


def fit_predict_day(frame: pd.DataFrame, target_date: pd.Timestamp, config: TrialConfig, epochs: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    train_days = [day for day in complete_market_days(frame) if day < target_date]
    train_past, train_future, train_target, used_days = daily_arrays(frame, train_days)
    forecast_past, forecast_future, _, _ = daily_arrays(frame, [target_date], require_targets=False)
    past_scale, future_scale, target_scale = fit_scales(train_past, train_future, train_target)
    model, _, train_loss = train_model(past_scale.transform(train_past), future_scale.transform(train_future), target_scale.transform(train_target), config=config, epochs=epochs, seed=seed)
    forecast = target_scale.inverse(predict(model, past_scale.transform(forecast_past), future_scale.transform(forecast_future)))[0]
    return forecast, {"train_days": len(used_days), "train_start": used_days[0].date().isoformat(), "train_end": used_days[-1].date().isoformat(), "epochs": epochs, "final_scaled_loss": train_loss}


def run_walk_forward(frame: pd.DataFrame, family: str, config: TrialConfig, epochs: int, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    days = [day for day in complete_market_days(frame) if start <= day <= end]
    rows: list[dict[str, Any]] = []
    for fold, day in enumerate(days):
        forecast, info = fit_predict_day(frame, day, config, epochs, 12600 + 100 * (family == "tcn") + fold)
        actual = frame.loc[frame["market_date"].eq(day)].sort_values("period")
        for hour in range(24):
            da_pred, spread_pred = float(forecast[hour, 0]), float(forecast[hour, 1])
            rows.append({"market_date": day.date().isoformat(), "period": hour + 1, "da_actual": float(actual.iloc[hour]["da"]), f"da_{family}_pred": da_pred, "spread_actual": float(actual.iloc[hour]["spread"]), f"spread_{family}_pred": spread_pred, "rt_actual": float(actual.iloc[hour]["rt"]), f"rt_{family}_pred": da_pred + spread_pred})
        print(f"completed {family.upper()} fold {fold + 1}/{len(days)}: {day.date().isoformat()}", flush=True)
    result = pd.DataFrame(rows)
    scores = {"day_ahead": base.metric(result["da_actual"], result[f"da_{family}_pred"]), "spread": base.metric(result["spread_actual"], result[f"spread_{family}_pred"]), "real_time_coherent": base.metric(result["rt_actual"], result[f"rt_{family}_pred"]), "spread_direction_accuracy": float(((result[f"spread_{family}_pred"] >= 0) == (result["spread_actual"] >= 0)).mean())}
    return result, {"scores": scores}


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparable GRU and residual TCN trials")
    parser.add_argument("--tune-start", default="2026-06-08")
    parser.add_argument("--tune-end", default="2026-06-14")
    parser.add_argument("--backtest-start", default="2026-06-15")
    parser.add_argument("--backtest-end", default="2026-06-30")
    parser.add_argument("--forecast-date", default="2026-07-01")
    parser.add_argument("--max-tune-epochs", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=base.ROOT / "outputs" / "gru_tcn_trial_20260827")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    frame, coverage = base.load_price_weather(base.PRICE_DEFAULT, base.WEATHER_DEFAULT, sorted(base.ROOT.glob(base.POWER_GLOB)))
    frame = base.add_all_feature_tables(frame)
    summary: dict[str, Any] = {"models": {}, "tuning_period": {"start": args.tune_start, "end": args.tune_end}, "backtest_period": {"start": args.backtest_start, "end": args.backtest_end}, "data_coverage": coverage}
    for family in ("gru", "tcn"):
        config, epochs, tuning = tune_architecture(frame, family, pd.Timestamp(args.tune_start), pd.Timestamp(args.tune_end), args.max_tune_epochs)
        print(f"selected {config.name} with {epochs} epochs", flush=True)
        backtest, details = run_walk_forward(frame, family, config, epochs, pd.Timestamp(args.backtest_start), pd.Timestamp(args.backtest_end))
        backtest.to_csv(args.output_dir / f"{family}_walk_forward_predictions.csv", index=False, encoding="utf-8-sig")
        forecast, info = fit_predict_day(frame, pd.Timestamp(args.forecast_date), config, epochs, 12700 + 100 * (family == "tcn"))
        pd.DataFrame([{"market_date": args.forecast_date, "period": h + 1, "day_ahead_price": float(forecast[h, 0]), "spread_real_time_minus_day_ahead": float(forecast[h, 1]), "real_time_price": float(forecast[h, 0] + forecast[h, 1])} for h in range(24)]).to_csv(args.output_dir / f"{family}_forecast.csv", index=False, encoding="utf-8-sig")
        summary["models"][family] = {"selected_config": config.__dict__, "selected_epochs": epochs, "tuning": tuning, "backtest": details["scores"], "forecast_training": info}
    summary["leakage_controls"] = ["Architecture selection completed before the reported backtest.", "Each fold trains only on dates before its target date.", "Decoder inputs contain only calendar, weather and power features available before market clearing."]
    (args.output_dir / "gru_tcn_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "models": {name: data["backtest"] for name, data in summary["models"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
