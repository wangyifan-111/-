"""Leakage-safe multivariate Transformer trial for day-ahead electricity prices.

The model forecasts a complete 24-hour market day in one pass. Its encoder
reads only the preceding seven days (168 hours); the decoder receives only
target-day variables assumed available before trading: weather, power forecast
proxies and calendar features. It jointly predicts DA and RT-DA spread, then
derives coherent RT as DA + spread.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
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


CONTEXT_HOURS = 168
TARGET_COLUMNS = ["da", "spread"]
PAST_COLUMNS = [
    "da", "rt", "spread",
    "temperature", "wind10", "wind100", "ghi", "cloud", "precipitation", "humidity",
    "direct_load", "tie_line", "wind_power", "pv_power", "local_power", "self_power",
    "nuclear_power", "renewable_power", "renewable_share", "net_load_proxy",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
FUTURE_COLUMNS = [
    "period", "clock_hour", "weekday", "month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "temperature", "wind10", "wind100", "ghi", "cloud", "precipitation", "humidity",
    "heating_cooling_degree", "ghi_hour_interaction", "cloud_ghi_interaction", "wind100_sq",
    "direct_load", "tie_line", "wind_power", "pv_power", "local_power", "self_power",
    "nuclear_power", "renewable_power", "renewable_share", "net_load_proxy", "power_forecast_flag",
]


@dataclass(frozen=True)
class TransformerConfig:
    name: str
    d_model: int
    nhead: int
    layers: int
    dim_feedforward: int
    dropout: float = 0.10


@dataclass
class Scale:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, axes: tuple[int, ...]) -> "Scale":
        mean = np.nanmean(values, axis=axes, keepdims=True)
        std = np.nanstd(values, axis=axes, keepdims=True)
        mean[~np.isfinite(mean)] = 0.0
        std[~np.isfinite(std) | (std < 1e-6)] = 1.0
        return cls(mean, std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        clean = np.where(np.isfinite(values), values, self.mean)
        return (clean - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


class DailyPriceTransformer(nn.Module):
    def __init__(self, past_dim: int, future_dim: int, config: TransformerConfig):
        super().__init__()
        self.past_projection = nn.Linear(past_dim, config.d_model)
        self.future_projection = nn.Linear(future_dim, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.layers)
        self.output = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 2))
        self.register_buffer("past_position", sinusoidal_position(CONTEXT_HOURS, config.d_model))
        self.register_buffer("future_position", sinusoidal_position(24, config.d_model))

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(self.past_projection(past) + self.past_position)
        decoded = self.decoder(self.future_projection(future) + self.future_position, memory)
        return self.output(decoded)


def sinusoidal_position(length: int, d_model: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    encoding = torch.zeros(1, length, d_model, dtype=torch.float32)
    encoding[0, :, 0::2] = torch.sin(position * divisor)
    encoding[0, :, 1::2] = torch.cos(position * divisor[: encoding[0, :, 1::2].shape[1]])
    return encoding


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def complete_market_days(frame: pd.DataFrame, require_targets: bool = True) -> list[pd.Timestamp]:
    mask = frame["weather_complete"] & frame["power_complete"]
    if require_targets:
        mask &= frame[TARGET_COLUMNS].notna().all(axis=1)
    counts = frame.loc[mask].groupby("market_date").size()
    return sorted(pd.to_datetime(counts[counts.eq(24)].index))


def daily_arrays(
    frame: pd.DataFrame,
    days: list[pd.Timestamp],
    require_targets: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[pd.Timestamp]]:
    past_values: list[np.ndarray] = []
    future_values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
    accepted: list[pd.Timestamp] = []
    ordered = frame.sort_values("datetime").reset_index(drop=True)
    for day in days:
        future = ordered.loc[ordered["market_date"].eq(day)].sort_values("period")
        history = ordered.loc[ordered["market_date"].lt(day)].tail(CONTEXT_HOURS)
        if len(future) != 24 or len(history) != CONTEXT_HOURS:
            continue
        if not future["weather_complete"].all() or not future["power_complete"].all():
            continue
        if history[["da", "rt", "spread"]].isna().any().any():
            continue
        if require_targets and future[TARGET_COLUMNS].isna().any().any():
            continue
        past_values.append(history[PAST_COLUMNS].to_numpy(float))
        future_values.append(future[FUTURE_COLUMNS].to_numpy(float))
        target_values.append(future[TARGET_COLUMNS].to_numpy(float))
        accepted.append(day)
    if not accepted:
        raise ValueError("no complete daily Transformer samples were available")
    return (
        np.stack(past_values),
        np.stack(future_values),
        np.stack(target_values),
        accepted,
    )


def fit_scales(past: np.ndarray, future: np.ndarray, target: np.ndarray) -> tuple[Scale, Scale, Scale]:
    return (
        Scale.fit(past, axes=(0, 1)),
        Scale.fit(future, axes=(0, 1)),
        Scale.fit(target, axes=(0, 1)),
    )


def train_model(
    past: np.ndarray,
    future: np.ndarray,
    target: np.ndarray,
    config: TransformerConfig,
    epochs: int,
    seed: int,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    patience: int = 15,
) -> tuple[DailyPriceTransformer, int, float]:
    seed_everything(seed)
    model = DailyPriceTransformer(past.shape[2], future.shape[2], config)
    dataset = TensorDataset(
        torch.tensor(past, dtype=torch.float32),
        torch.tensor(future, dtype=torch.float32),
        torch.tensor(target, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-3)
    loss_fn = nn.SmoothL1Loss(beta=0.75)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 1
    wait = 0
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
                current = float(
                    loss_fn(
                        model(torch.tensor(val_past, dtype=torch.float32), torch.tensor(val_future, dtype=torch.float32)),
                        torch.tensor(val_target, dtype=torch.float32),
                    )
                )
        if current < best_loss - 1e-5:
            best_loss = current
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if validation is not None and wait >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_epoch, best_loss


def predict(model: DailyPriceTransformer, past: np.ndarray, future: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(
            torch.tensor(past, dtype=torch.float32),
            torch.tensor(future, dtype=torch.float32),
        ).cpu().numpy()


def tune_architecture(
    frame: pd.DataFrame,
    tune_start: pd.Timestamp,
    tune_end: pd.Timestamp,
    max_epochs: int,
) -> tuple[TransformerConfig, int, list[dict[str, Any]]]:
    all_days = complete_market_days(frame)
    train_days = [day for day in all_days if day < tune_start]
    validation_days = [day for day in all_days if tune_start <= day <= tune_end]
    train_past, train_future, train_target, _ = daily_arrays(frame, train_days)
    val_past, val_future, val_target, _ = daily_arrays(frame, validation_days)
    past_scale, future_scale, target_scale = fit_scales(train_past, train_future, train_target)
    train_scaled = (
        past_scale.transform(train_past), future_scale.transform(train_future), target_scale.transform(train_target)
    )
    val_scaled = (
        past_scale.transform(val_past), future_scale.transform(val_future), target_scale.transform(val_target)
    )
    candidates = [
        TransformerConfig("tiny_1layer", 16, 2, 1, 64),
        TransformerConfig("small_1layer", 32, 4, 1, 128),
        TransformerConfig("small_2layer", 32, 4, 2, 128),
    ]
    results: list[dict[str, Any]] = []
    best: tuple[float, TransformerConfig, int] | None = None
    for index, config in enumerate(candidates):
        model, best_epoch, val_loss = train_model(
            *train_scaled,
            config=config,
            epochs=max_epochs,
            seed=2400 + index,
            validation=val_scaled,
        )
        prediction = target_scale.inverse(predict(model, val_scaled[0], val_scaled[1]))
        da = base.metric(val_target[:, :, 0].reshape(-1), prediction[:, :, 0].reshape(-1))
        spread = base.metric(val_target[:, :, 1].reshape(-1), prediction[:, :, 1].reshape(-1))
        actual_rt = val_target[:, :, 0] + val_target[:, :, 1]
        predicted_rt = prediction[:, :, 0] + prediction[:, :, 1]
        rt = base.metric(actual_rt.reshape(-1), predicted_rt.reshape(-1))
        composite = float(np.mean([da["mae_yuan_per_mwh"], rt["mae_yuan_per_mwh"], spread["mae_yuan_per_mwh"]]))
        item = {
            "config": config.__dict__, "best_epoch": best_epoch, "validation_loss_scaled": val_loss,
            "day_ahead": da, "real_time_coherent": rt, "spread": spread, "composite_mae": composite,
        }
        results.append(item)
        if best is None or composite < best[0]:
            best = (composite, config, best_epoch)
    assert best is not None
    return best[1], max(10, best[2]), results


def fit_predict_day(
    frame: pd.DataFrame,
    target_date: pd.Timestamp,
    config: TransformerConfig,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    available_days = [day for day in complete_market_days(frame) if day < target_date]
    train_past, train_future, train_target, used_days = daily_arrays(frame, available_days)
    forecast_past, forecast_future, _, _ = daily_arrays(frame, [target_date], require_targets=False)
    past_scale, future_scale, target_scale = fit_scales(train_past, train_future, train_target)
    model, _, train_loss = train_model(
        past_scale.transform(train_past),
        future_scale.transform(train_future),
        target_scale.transform(train_target),
        config=config,
        epochs=epochs,
        seed=seed,
        validation=None,
    )
    prediction = target_scale.inverse(
        predict(model, past_scale.transform(forecast_past), future_scale.transform(forecast_future))
    )[0]
    return prediction, {
        "train_days": len(used_days),
        "train_start": used_days[0].date().isoformat(),
        "train_end": used_days[-1].date().isoformat(),
        "epochs": epochs,
        "final_scaled_loss": train_loss,
    }


def run_walk_forward(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: TransformerConfig,
    epochs: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    days = [day for day in complete_market_days(frame) if start <= day <= end]
    rows: list[dict[str, Any]] = []
    fold_info: list[dict[str, Any]] = []
    for fold, day in enumerate(days):
        prediction, info = fit_predict_day(frame, day, config, epochs, seed=4200 + fold)
        actual = frame.loc[frame["market_date"].eq(day)].sort_values("period")
        for hour in range(24):
            da_pred = float(prediction[hour, 0])
            spread_pred = float(prediction[hour, 1])
            rows.append(
                {
                    "market_date": day.date().isoformat(), "period": hour + 1,
                    "da_actual": float(actual.iloc[hour]["da"]), "da_transformer_pred": da_pred,
                    "spread_actual": float(actual.iloc[hour]["spread"]), "spread_transformer_pred": spread_pred,
                    "rt_actual": float(actual.iloc[hour]["rt"]), "rt_transformer_pred": da_pred + spread_pred,
                }
            )
        fold_info.append({"market_date": day.date().isoformat(), **info})
        print(f"completed Transformer fold {fold + 1}/{len(days)}: {day.date().isoformat()}", flush=True)
    result = pd.DataFrame(rows)
    scores = {
        "day_ahead": base.metric(result["da_actual"], result["da_transformer_pred"]),
        "real_time_coherent": base.metric(result["rt_actual"], result["rt_transformer_pred"]),
        "spread": base.metric(result["spread_actual"], result["spread_transformer_pred"]),
        "spread_direction_accuracy": float(
            ((result["spread_transformer_pred"] >= 0) == (result["spread_actual"] >= 0)).mean()
        ),
    }
    return result, {"scores": scores, "folds": fold_info}


def compare_and_blend(transformer: pd.DataFrame, ensemble_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    trees = pd.read_csv(ensemble_path)
    merged = trees.merge(transformer, on=["market_date", "period"], suffixes=("_tree", ""), validate="one_to_one")
    merged["da_xgb_transformer_equal_pred"] = (merged["da_xgboost_pred"] + merged["da_transformer_pred"]) / 2
    merged["spread_ridge_transformer_equal_pred"] = (merged["spread_ridge_pred"] + merged["spread_transformer_pred"]) / 2
    merged["rt_selective_transformer_equal_pred"] = (
        merged["da_xgb_transformer_equal_pred"] + merged["spread_ridge_transformer_equal_pred"]
    )
    scores = {
        "da_xgboost_transformer_equal": base.metric(merged["da_actual"], merged["da_xgb_transformer_equal_pred"]),
        "spread_ridge_transformer_equal": base.metric(merged["spread_actual"], merged["spread_ridge_transformer_equal_pred"]),
        "rt_coherent_selective_transformer_equal": base.metric(
            merged["rt_actual"], merged["rt_selective_transformer_equal_pred"]
        ),
    }
    return merged, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Multivariate sequence-to-sequence Transformer trial")
    parser.add_argument("--price", type=Path, default=base.PRICE_DEFAULT)
    parser.add_argument("--weather", type=Path, default=base.WEATHER_DEFAULT)
    parser.add_argument("--power-dir", type=Path, default=base.ROOT)
    parser.add_argument("--tune-start", default="2026-06-08")
    parser.add_argument("--tune-end", default="2026-06-14")
    parser.add_argument("--backtest-start", default="2026-06-15")
    parser.add_argument("--backtest-end", default="2026-06-30")
    parser.add_argument("--forecast-date", default="2026-07-01")
    parser.add_argument("--max-tune-epochs", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=base.ROOT / "outputs" / "transformer_trial_20260827")
    parser.add_argument("--ensemble-predictions", type=Path, default=base.ROOT / "outputs" / "ensemble_search_20260826" / "walk_forward_predictions.csv")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    frame, coverage = base.load_price_weather(args.price, args.weather, sorted(args.power_dir.glob(base.POWER_GLOB)))
    frame = base.add_all_feature_tables(frame)
    config, epochs, tuning = tune_architecture(
        frame, pd.Timestamp(args.tune_start), pd.Timestamp(args.tune_end), args.max_tune_epochs
    )
    print(f"selected {config.name} with {epochs} epochs", flush=True)
    backtest, details = run_walk_forward(
        frame, pd.Timestamp(args.backtest_start), pd.Timestamp(args.backtest_end), config, epochs
    )
    backtest.to_csv(args.output_dir / "walk_forward_predictions.csv", index=False, encoding="utf-8-sig")
    blend_scores: dict[str, Any] = {}
    if args.ensemble_predictions.exists():
        blended, blend_scores = compare_and_blend(backtest, args.ensemble_predictions)
        blended.to_csv(args.output_dir / "transformer_blend_predictions.csv", index=False, encoding="utf-8-sig")
    forecast, forecast_info = fit_predict_day(frame, pd.Timestamp(args.forecast_date), config, epochs, seed=5200)
    forecast_rows = [
        {
            "market_date": args.forecast_date, "period": hour + 1,
            "day_ahead_price": float(forecast[hour, 0]),
            "spread_real_time_minus_day_ahead": float(forecast[hour, 1]),
            "real_time_price": float(forecast[hour, 0] + forecast[hour, 1]),
        }
        for hour in range(24)
    ]
    pd.DataFrame(forecast_rows).to_csv(args.output_dir / "forecast.csv", index=False, encoding="utf-8-sig")
    summary = {
        "model": "sequence-to-sequence Transformer encoder-decoder",
        "selected_config": config.__dict__, "selected_epochs": epochs,
        "tuning_period": {"start": args.tune_start, "end": args.tune_end, "candidates": tuning},
        "backtest_period": {"start": args.backtest_start, "end": args.backtest_end},
        "backtest": details["scores"], "predefined_equal_blends": blend_scores,
        "forecast_date": args.forecast_date, "forecast_training": forecast_info,
        "data_coverage": coverage,
        "leakage_controls": [
            "Architecture tuning ends before the reported backtest starts.",
            "Each fold is fitted only on complete market days before its target day.",
            "All 24 target hours use a history ending before the target market day.",
            "Decoder inputs contain only calendar, weather and power variables assumed available pre-market.",
        ],
    }
    (args.output_dir / "transformer_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "forecast.json").write_text(json.dumps({"model": summary["model"], "forecast": forecast_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "selected_config": config.name, "epochs": epochs, "backtest": details["scores"], "blends": blend_scores}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
