"""CPU-reproducible MambaPy state-space trial for electricity prices.

This uses the pure-PyTorch Mamba implementation from ``mambapy``. It is a
genuine selective state-space model, but not the official CUDA/Triton
``mamba-ssm`` kernel. The forecasting protocol matches the Transformer trial:
168 historical hours, 24 target-day exogenous tokens, and no target-day price
observations in the input.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from mambapy.mamba import Mamba, MambaConfig
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import integrated_price_forecast as base
from transformer_price_forecast import (
    FUTURE_COLUMNS,
    PAST_COLUMNS,
    Scale,
    complete_market_days,
    daily_arrays,
    fit_scales,
    seed_everything,
)


@dataclass(frozen=True)
class TrialConfig:
    name: str
    d_model: int
    n_layers: int
    d_state: int
    expand_factor: int = 2
    d_conv: int = 4
    dropout: float = 0.05


class DailyPriceMamba(nn.Module):
    def __init__(self, past_dim: int, future_dim: int, config: TrialConfig):
        super().__init__()
        self.past_projection = nn.Linear(past_dim, config.d_model)
        self.future_projection = nn.Linear(future_dim, config.d_model)
        self.past_segment = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.future_segment = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.dropout = nn.Dropout(config.dropout)
        self.backbone = Mamba(
            MambaConfig(
                d_model=config.d_model,
                n_layers=config.n_layers,
                d_state=config.d_state,
                expand_factor=config.expand_factor,
                d_conv=config.d_conv,
                pscan=True,
                use_cuda=False,
            )
        )
        self.output = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 2))

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        past_token = self.past_projection(past) + self.past_segment
        future_token = self.future_projection(future) + self.future_segment
        sequence = torch.cat([past_token, future_token], dim=1)
        hidden = self.backbone(self.dropout(sequence))
        return self.output(hidden[:, -24:, :])


def train_model(
    past: np.ndarray,
    future: np.ndarray,
    target: np.ndarray,
    config: TrialConfig,
    epochs: int,
    seed: int,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    patience: int = 15,
) -> tuple[DailyPriceMamba, int, float]:
    seed_everything(seed)
    model = DailyPriceMamba(past.shape[2], future.shape[2], config)
    dataset = TensorDataset(
        torch.tensor(past, dtype=torch.float32),
        torch.tensor(future, dtype=torch.float32),
        torch.tensor(target, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(8, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1e-3)
    loss_fn = nn.SmoothL1Loss(beta=0.75)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 1
    wait = 0
    for epoch in range(1, epochs + 1):
        model.train()
        last_loss = None
        for batch_past, batch_future, batch_target in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_past, batch_future), batch_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            last_loss = loss
        model.eval()
        with torch.no_grad():
            if validation is None:
                assert last_loss is not None
                current = float(last_loss.detach())
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


def predict(model: DailyPriceMamba, past: np.ndarray, future: np.ndarray) -> np.ndarray:
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
) -> tuple[TrialConfig, int, list[dict[str, Any]]]:
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
        TrialConfig("mamba_tiny", 16, 1, 8),
        TrialConfig("mamba_small", 32, 1, 16),
        TrialConfig("mamba_small_2layer", 32, 2, 16),
    ]
    results: list[dict[str, Any]] = []
    best: tuple[float, TrialConfig, int] | None = None
    for index, config in enumerate(candidates):
        model, best_epoch, val_loss = train_model(
            *train_scaled,
            config=config,
            epochs=max_epochs,
            seed=3400 + index,
            validation=val_scaled,
        )
        forecast = target_scale.inverse(predict(model, val_scaled[0], val_scaled[1]))
        da = base.metric(val_target[:, :, 0].reshape(-1), forecast[:, :, 0].reshape(-1))
        spread = base.metric(val_target[:, :, 1].reshape(-1), forecast[:, :, 1].reshape(-1))
        rt = base.metric(
            (val_target[:, :, 0] + val_target[:, :, 1]).reshape(-1),
            (forecast[:, :, 0] + forecast[:, :, 1]).reshape(-1),
        )
        composite = float(np.mean([da["mae_yuan_per_mwh"], spread["mae_yuan_per_mwh"], rt["mae_yuan_per_mwh"]]))
        results.append(
            {
                "config": config.__dict__, "best_epoch": best_epoch, "validation_loss_scaled": val_loss,
                "day_ahead": da, "spread": spread, "real_time_coherent": rt, "composite_mae": composite,
            }
        )
        if best is None or composite < best[0]:
            best = (composite, config, best_epoch)
    assert best is not None
    return best[1], max(10, best[2]), results


def fit_predict_day(
    frame: pd.DataFrame,
    target_date: pd.Timestamp,
    config: TrialConfig,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_days = [day for day in complete_market_days(frame) if day < target_date]
    train_past, train_future, train_target, used_days = daily_arrays(frame, train_days)
    forecast_past, forecast_future, _, _ = daily_arrays(frame, [target_date], require_targets=False)
    past_scale, future_scale, target_scale = fit_scales(train_past, train_future, train_target)
    model, _, train_loss = train_model(
        past_scale.transform(train_past),
        future_scale.transform(train_future),
        target_scale.transform(train_target),
        config=config,
        epochs=epochs,
        seed=seed,
    )
    forecast = target_scale.inverse(
        predict(model, past_scale.transform(forecast_past), future_scale.transform(forecast_future))
    )[0]
    return forecast, {
        "train_days": len(used_days), "train_start": used_days[0].date().isoformat(),
        "train_end": used_days[-1].date().isoformat(), "epochs": epochs, "final_scaled_loss": train_loss,
    }


def run_walk_forward(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: TrialConfig,
    epochs: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    days = [day for day in complete_market_days(frame) if start <= day <= end]
    rows: list[dict[str, Any]] = []
    folds = []
    for fold, day in enumerate(days):
        forecast, info = fit_predict_day(frame, day, config, epochs, seed=6200 + fold)
        actual = frame.loc[frame["market_date"].eq(day)].sort_values("period")
        for hour in range(24):
            da_pred = float(forecast[hour, 0])
            spread_pred = float(forecast[hour, 1])
            rows.append(
                {
                    "market_date": day.date().isoformat(), "period": hour + 1,
                    "da_actual": float(actual.iloc[hour]["da"]), "da_mamba_pred": da_pred,
                    "spread_actual": float(actual.iloc[hour]["spread"]), "spread_mamba_pred": spread_pred,
                    "rt_actual": float(actual.iloc[hour]["rt"]), "rt_mamba_pred": da_pred + spread_pred,
                }
            )
        folds.append({"market_date": day.date().isoformat(), **info})
        print(f"completed Mamba fold {fold + 1}/{len(days)}: {day.date().isoformat()}", flush=True)
    result = pd.DataFrame(rows)
    scores = {
        "day_ahead": base.metric(result["da_actual"], result["da_mamba_pred"]),
        "spread": base.metric(result["spread_actual"], result["spread_mamba_pred"]),
        "real_time_coherent": base.metric(result["rt_actual"], result["rt_mamba_pred"]),
        "spread_direction_accuracy": float(((result["spread_mamba_pred"] >= 0) == (result["spread_actual"] >= 0)).mean()),
    }
    return result, {"scores": scores, "folds": folds}


def compare_blends(mamba: pd.DataFrame, tree_path: Path, transformer_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    trees = pd.read_csv(tree_path)
    transformer = pd.read_csv(transformer_path)[
        ["market_date", "period", "da_transformer_pred", "spread_transformer_pred"]
    ]
    merged = trees.merge(mamba, on=["market_date", "period"], suffixes=("_tree", ""), validate="one_to_one")
    merged = merged.merge(transformer, on=["market_date", "period"], validate="one_to_one")
    merged["da_xgb_mamba_equal_pred"] = (merged["da_xgboost_pred"] + merged["da_mamba_pred"]) / 2
    merged["spread_ridge_mamba_equal_pred"] = (merged["spread_ridge_pred"] + merged["spread_mamba_pred"]) / 2
    merged["spread_three_model_equal_pred"] = (
        merged["spread_ridge_pred"] + merged["spread_transformer_pred"] + merged["spread_mamba_pred"]
    ) / 3
    merged["rt_ridge_mamba_equal_pred"] = merged["da_xgboost_pred"] + merged["spread_ridge_mamba_equal_pred"]
    merged["rt_three_model_equal_pred"] = merged["da_xgboost_pred"] + merged["spread_three_model_equal_pred"]
    scores = {
        "da_xgboost_mamba_equal": base.metric(merged["da_actual"], merged["da_xgb_mamba_equal_pred"]),
        "spread_ridge_mamba_equal": base.metric(merged["spread_actual"], merged["spread_ridge_mamba_equal_pred"]),
        "spread_ridge_transformer_mamba_equal": base.metric(merged["spread_actual"], merged["spread_three_model_equal_pred"]),
        "rt_xgboost_plus_ridge_mamba_equal": base.metric(merged["rt_actual"], merged["rt_ridge_mamba_equal_pred"]),
        "rt_xgboost_plus_three_model_equal": base.metric(merged["rt_actual"], merged["rt_three_model_equal_pred"]),
    }
    return merged, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure-PyTorch Mamba state-space forecasting trial")
    parser.add_argument("--tune-start", default="2026-06-08")
    parser.add_argument("--tune-end", default="2026-06-14")
    parser.add_argument("--backtest-start", default="2026-06-15")
    parser.add_argument("--backtest-end", default="2026-06-30")
    parser.add_argument("--forecast-date", default="2026-07-01")
    parser.add_argument("--max-tune-epochs", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=base.ROOT / "outputs" / "mamba_trial_20260827")
    parser.add_argument("--tree-predictions", type=Path, default=base.ROOT / "outputs" / "ensemble_search_20260826" / "walk_forward_predictions.csv")
    parser.add_argument("--transformer-predictions", type=Path, default=base.ROOT / "outputs" / "transformer_trial_20260827" / "walk_forward_predictions.csv")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    frame, coverage = base.load_price_weather(base.PRICE_DEFAULT, base.WEATHER_DEFAULT, sorted(base.ROOT.glob(base.POWER_GLOB)))
    frame = base.add_all_feature_tables(frame)
    config, epochs, tuning = tune_architecture(frame, pd.Timestamp(args.tune_start), pd.Timestamp(args.tune_end), args.max_tune_epochs)
    print(f"selected {config.name} with {epochs} epochs", flush=True)
    backtest, details = run_walk_forward(frame, pd.Timestamp(args.backtest_start), pd.Timestamp(args.backtest_end), config, epochs)
    backtest.to_csv(args.output_dir / "walk_forward_predictions.csv", index=False, encoding="utf-8-sig")
    blend_scores: dict[str, Any] = {}
    if args.tree_predictions.exists() and args.transformer_predictions.exists():
        blended, blend_scores = compare_blends(backtest, args.tree_predictions, args.transformer_predictions)
        blended.to_csv(args.output_dir / "mamba_blend_predictions.csv", index=False, encoding="utf-8-sig")
    forecast, forecast_info = fit_predict_day(frame, pd.Timestamp(args.forecast_date), config, epochs, seed=7200)
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
        "model": "MambaPy selective state-space sequence model",
        "implementation": {"package": "mambapy", "version": importlib.metadata.version("mambapy"), "official_mamba_ssm_cuda_kernel": False},
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
            "The final 24 tokens contain only pre-market exogenous variables, not target-day prices.",
        ],
    }
    (args.output_dir / "mamba_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "forecast.json").write_text(json.dumps({"model": summary["model"], "forecast": forecast_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "selected_config": config.name, "epochs": epochs, "backtest": details["scores"], "blends": blend_scores}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
