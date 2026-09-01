"""Pre-test tuning of post-DA LightGBM real-time forecasts."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import integrated_price_forecast as base
import realtime_post_da_forecast as post_da


@dataclass(frozen=True)
class Candidate:
    name: str
    target: str
    objective: str
    window_days: int | None
    num_leaves: int = 25
    min_child_samples: int = 24


def candidates() -> list[Candidate]:
    result: list[Candidate] = []
    for target in ("spread", "rt"):
        for objective in ("regression_l1", "huber", "regression_l2"):
            for window in (None, 90, 60):
                window_name = "all" if window is None else f"{window}d"
                result.append(Candidate(f"{target}_{objective}_{window_name}", target, objective, window))
    result.extend([
        Candidate("spread_l1_90d_flexible", "spread", "regression_l1", 90, 31, 15),
        Candidate("rt_l1_90d_flexible", "rt", "regression_l1", 90, 31, 15),
    ])
    return result


def make_model(config: Candidate) -> Any:
    return base.LGBMRegressor(
        objective=config.objective,
        alpha=0.85,
        n_estimators=600,
        learning_rate=0.025,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=3.0,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )


def fit_day(frame: pd.DataFrame, day: pd.Timestamp, config: Candidate) -> pd.DataFrame:
    complete = frame["weather_complete"] & frame["power_complete"]
    date = frame["market_date"]
    train_mask = date.lt(day) & complete
    if config.window_days is not None:
        train_mask &= date.ge(day - pd.Timedelta(days=config.window_days))
    predict_mask = date.eq(day) & complete
    columns = post_da.feature_columns(config.target)
    values, _ = base.clean_matrix(frame, columns, train_mask)
    valid = train_mask.to_numpy(bool) & frame[config.target].notna().to_numpy(bool)
    train_x = pd.DataFrame(values[valid], columns=columns)
    predict_x = pd.DataFrame(values[predict_mask.to_numpy(bool)], columns=columns)
    model = make_model(config).fit(train_x, frame.loc[valid, config.target].to_numpy(float))
    raw_prediction = np.asarray(model.predict(predict_x), dtype=float)
    target = frame.loc[predict_mask, ["market_date", "period", "da", "rt"]].sort_values("period").reset_index(drop=True)
    target = target.rename(columns={"da": "da_actual", "rt": "rt_actual"})
    target[config.name] = raw_prediction + target["da_actual"].to_numpy(float) if config.target == "spread" else raw_prediction
    return target


def run_candidate(frame: pd.DataFrame, dates: pd.DatetimeIndex, config: Candidate) -> pd.DataFrame:
    rows = [fit_day(frame, day, config) for day in dates]
    return pd.concat(rows, ignore_index=True)


def best_blend_weight(actual: np.ndarray, first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    weights = np.linspace(0.0, 1.0, 101)
    scores = [np.mean(np.abs(actual - (weight * first + (1.0 - weight) * second))) for weight in weights]
    index = int(np.argmin(scores))
    return float(weights[index]), float(scores[index])


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune post-DA LightGBM on a pre-test week")
    parser.add_argument("--tune-start", default="2026-06-08")
    parser.add_argument("--tune-end", default="2026-06-14")
    parser.add_argument("--backtest-start", default="2026-06-15")
    parser.add_argument("--backtest-end", default="2026-06-30")
    parser.add_argument(
        "--output-dir", type=Path, default=base.ROOT / "outputs" / "realtime_lightgbm_tuning_20260831"
    )
    args = parser.parse_args()
    if not base.HAS_LIGHTGBM:
        raise RuntimeError("LightGBM is not available through the project DLL bootstrap")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, coverage = base.load_price_weather(
        base.PRICE_DEFAULT, base.WEATHER_DEFAULT, sorted(base.ROOT.glob(base.POWER_GLOB))
    )
    frame = post_da.add_known_da_features(base.add_all_feature_tables(frame))
    tune_dates = pd.date_range(args.tune_start, args.tune_end, freq="D")
    configs = candidates()
    tune_table: pd.DataFrame | None = None
    tune_scores: list[dict[str, Any]] = []
    for index, config in enumerate(configs, start=1):
        prediction = run_candidate(frame, tune_dates, config)
        if tune_table is None:
            tune_table = prediction
        else:
            tune_table = tune_table.merge(
                prediction[["market_date", "period", config.name]],
                on=["market_date", "period"],
                how="inner",
            )
        score = base.metric(prediction["rt_actual"], prediction[config.name])
        tune_scores.append({"config": asdict(config), **score})
        print(f"completed tuning candidate {index}/{len(configs)}: {config.name} MAE={score['mae_yuan_per_mwh']:.3f}", flush=True)
    assert tune_table is not None
    ranked = sorted(tune_scores, key=lambda item: item["mae_yuan_per_mwh"])
    selected_configs = [
        next(config for config in configs if config.name == ranked[0]["config"]["name"]),
        next(config for config in configs if config.name == ranked[1]["config"]["name"]),
    ]
    actual_tune = tune_table["rt_actual"].to_numpy(float)
    blend_weight, blend_tune_mae = best_blend_weight(
        actual_tune,
        tune_table[selected_configs[0].name].to_numpy(float),
        tune_table[selected_configs[1].name].to_numpy(float),
    )
    tune_table.to_csv(args.output_dir / "tuning_predictions.csv", index=False, encoding="utf-8-sig")

    test_dates = pd.date_range(args.backtest_start, args.backtest_end, freq="D")
    test_predictions = [run_candidate(frame, test_dates, config) for config in selected_configs]
    test = test_predictions[0].merge(
        test_predictions[1][["market_date", "period", selected_configs[1].name]],
        on=["market_date", "period"],
        how="inner",
    )
    first_name, second_name = selected_configs[0].name, selected_configs[1].name
    test["precalibrated_top2_blend"] = (
        blend_weight * test[first_name] + (1.0 - blend_weight) * test[second_name]
    )
    test.to_csv(args.output_dir / "backtest_predictions.csv", index=False, encoding="utf-8-sig")
    backtest_scores = {
        first_name: base.metric(test["rt_actual"], test[first_name]),
        second_name: base.metric(test["rt_actual"], test[second_name]),
        "precalibrated_top2_blend": base.metric(test["rt_actual"], test["precalibrated_top2_blend"]),
    }
    summary = {
        "tuning_period": {"start": args.tune_start, "end": args.tune_end},
        "backtest_period": {"start": args.backtest_start, "end": args.backtest_end},
        "candidate_ranking": ranked,
        "selected_top2": [asdict(config) for config in selected_configs],
        "precalibrated_blend": {
            "weight_first": blend_weight,
            "weight_second": 1.0 - blend_weight,
            "tuning_mae": blend_tune_mae,
        },
        "backtest": backtest_scores,
        "data_coverage": coverage,
        "leakage_controls": [
            "Configuration and blend weight are selected only on 2026-06-08 through 2026-06-14.",
            "Each tuning and test fold trains only on dates before its target day.",
            "Target-day DA is allowed only because this is a post-clearing RT forecast.",
            "No target-day RT price or spread is used as a feature.",
        ],
        "status": "exploratory because the reported test interval has been inspected in prior experiments",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
