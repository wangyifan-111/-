"""Leakage-safe rolling correction for the post-DA real-time forecast."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import integrated_price_forecast as base
import realtime_post_da_forecast as post_da


def metric(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    return base.metric(actual.to_numpy(float), pred.to_numpy(float))


def run_base(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, model_name: str) -> pd.DataFrame:
    complete = frame["weather_complete"] & frame["power_complete"]
    dates = sorted(pd.to_datetime(frame.loc[complete, "market_date"].unique()))
    dates = [day for day in dates if start <= day <= end]
    rows: list[pd.DataFrame] = []
    for index, day in enumerate(dates, start=1):
        train_mask = frame["market_date"].lt(day) & complete
        predict_mask = frame["market_date"].eq(day) & complete
        if int(predict_mask.sum()) != 24:
            continue
        prediction = post_da.fit_predict(frame, "rt", train_mask, predict_mask, model_name)
        target = frame.loc[predict_mask, ["market_date", "period", "da", "rt"]].sort_values("period").reset_index(drop=True)
        target = target.rename(columns={"da": "da_actual", "rt": "rt_actual"})
        target["base_pred"] = prediction
        rows.append(target)
        print(f"completed correction base fold {index}/{len(dates)}: {day.date().isoformat()}", flush=True)
    return pd.concat(rows, ignore_index=True)


def correction_for_day(
    history: pd.DataFrame,
    target: pd.DataFrame,
    window_days: int,
    level: str,
    statistic: str,
    alpha: float,
) -> pd.Series:
    day = pd.Timestamp(target["market_date"].iloc[0])
    prior = history.loc[
        history["market_date"].lt(day)
        & history["market_date"].ge(day - pd.Timedelta(days=window_days))
    ].copy()
    prior["error"] = prior["rt_actual"] - prior["base_pred"]
    if prior.empty:
        return target["base_pred"].copy()
    if level == "global":
        correction = getattr(prior["error"], statistic)()
        return target["base_pred"] + alpha * float(correction)
    if level == "period":
        correction = prior.groupby("period")["error"].agg(statistic)
        return target["base_pred"] + alpha * target["period"].map(correction).fillna(0.0)
    if level == "segment":
        correction = prior.groupby("segment")["error"].agg(statistic)
        return target["base_pred"] + alpha * target["segment"].map(correction).fillna(0.0)
    raise ValueError(level)


def score_configuration(
    predictions: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    window_days: int,
    level: str,
    statistic: str,
    alpha: float,
) -> float:
    test = predictions.loc[predictions["market_date"].between(start, end)].copy()
    output: list[pd.Series] = []
    for day, target in test.groupby("market_date", sort=True):
        # score_configuration is only called after the full OOF table exists;
        # the correction slice itself ends strictly before the target day.
        output.append(correction_for_day(predictions, target, window_days, level, statistic, alpha))
    if not output:
        return float("inf")
    corrected = pd.concat(output).sort_index()
    return float(np.mean(np.abs(test.loc[corrected.index, "rt_actual"].to_numpy(float) - corrected.to_numpy(float))))


def apply_configuration(
    predictions: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    window_days: int,
    level: str,
    statistic: str,
    alpha: float,
) -> pd.DataFrame:
    test = predictions.loc[predictions["market_date"].between(start, end)].copy()
    corrected: list[pd.DataFrame] = []
    for _, target in test.groupby("market_date", sort=True):
        target = target.copy()
        target["corrected_pred"] = correction_for_day(
            predictions, target, window_days, level, statistic, alpha
        ).to_numpy(float)
        corrected.append(target)
    return pd.concat(corrected, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling post-DA RT error correction")
    parser.add_argument("--start", default="2026-06-01")
    parser.add_argument("--tune-start", default="2026-06-08")
    parser.add_argument("--tune-end", default="2026-06-14")
    parser.add_argument("--backtest-start", default="2026-06-15")
    parser.add_argument("--backtest-end", default="2026-06-30")
    parser.add_argument("--base-model", default="lightgbm_l1")
    parser.add_argument(
        "--output-dir", type=Path, default=base.ROOT / "outputs" / "realtime_post_da_correction_20260831"
    )
    args = parser.parse_args()
    if not base.HAS_LIGHTGBM:
        raise RuntimeError("LightGBM is not available through the project DLL bootstrap")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, coverage = base.load_price_weather(
        base.PRICE_DEFAULT, base.WEATHER_DEFAULT, sorted(base.ROOT.glob(base.POWER_GLOB))
    )
    frame = post_da.add_known_da_features(base.add_all_feature_tables(frame))
    predictions = run_base(frame, pd.Timestamp(args.start), pd.Timestamp(args.backtest_end), args.base_model)
    predictions["market_date"] = pd.to_datetime(predictions["market_date"])
    predictions["segment"] = np.select(
        [predictions["period"].le(6), predictions["period"].le(16)],
        ["night", "solar"],
        default="evening",
    )
    predictions.to_csv(args.output_dir / "base_predictions.csv", index=False, encoding="utf-8-sig")

    tune_start = pd.Timestamp(args.tune_start)
    tune_end = pd.Timestamp(args.tune_end)
    configurations = [
        {"window_days": window, "level": level, "statistic": statistic, "alpha": alpha}
        for window in (3, 5, 7, 10)
        for level in ("global", "period", "segment")
        for statistic in ("mean", "median")
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    tune_scores: list[dict[str, Any]] = []
    for configuration in configurations:
        score = score_configuration(predictions, tune_start, tune_end, **configuration)
        tune_scores.append({**configuration, "tune_mae_yuan_per_mwh": score})
    tune_scores.sort(key=lambda item: item["tune_mae_yuan_per_mwh"])
    selected = tune_scores[0]
    backtest = apply_configuration(
        predictions,
        pd.Timestamp(args.backtest_start),
        pd.Timestamp(args.backtest_end),
        selected["window_days"],
        selected["level"],
        selected["statistic"],
        selected["alpha"],
    )
    backtest["base_abs_error"] = (backtest["base_pred"] - backtest["rt_actual"]).abs()
    backtest["corrected_abs_error"] = (backtest["corrected_pred"] - backtest["rt_actual"]).abs()
    backtest.to_csv(args.output_dir / "backtest_predictions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "base_model": args.base_model,
        "tuning_period": {"start": args.tune_start, "end": args.tune_end},
        "backtest_period": {"start": args.backtest_start, "end": args.backtest_end},
        "selected_configuration": selected,
        "tuning_baseline_mae_yuan_per_mwh": metric(
            predictions.loc[predictions["market_date"].between(tune_start, tune_end), "rt_actual"],
            predictions.loc[predictions["market_date"].between(tune_start, tune_end), "base_pred"],
        ),
        "backtest_base": metric(backtest["rt_actual"], backtest["base_pred"]),
        "backtest_corrected": metric(backtest["rt_actual"], backtest["corrected_pred"]),
        "by_segment": {
            segment: {
                "base": metric(group["rt_actual"], group["base_pred"]),
                "corrected": metric(group["rt_actual"], group["corrected_pred"]),
            }
            for segment, group in backtest.groupby("segment", sort=False)
        },
        "top_tuning_candidates": tune_scores[:15],
        "data_coverage": coverage,
        "leakage_controls": [
            "Base models train only on dates before each target day.",
            "Tuning uses only June 8-14; the chosen correction is then frozen for June 15-30.",
            "Each correction uses only the preceding window and never target-day RT.",
        ],
        "status": "exploratory on a previously inspected interval; confirm on later unseen dates",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
