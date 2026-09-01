"""Solar-period regime specialists for the post-DA real-time forecast."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import integrated_price_forecast as base
import realtime_post_da_forecast as post_da


REGIME_NAMES = ("negative", "normal", "high")


def regime_label(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(values < 0.0, 0, np.where(values > 300.0, 2, 1)).astype(int)


def make_lgbm_regressor() -> Any:
    return base.LGBMRegressor(
        objective="regression_l1",
        n_estimators=550,
        learning_rate=0.025,
        num_leaves=21,
        min_child_samples=18,
        colsample_bytree=0.9,
        reg_alpha=0.3,
        reg_lambda=3.0,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )


def make_classifier() -> Any:
    return base.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        class_weight="balanced",
        n_estimators=450,
        learning_rate=0.025,
        num_leaves=17,
        min_child_samples=20,
        colsample_bytree=0.9,
        reg_alpha=0.5,
        reg_lambda=4.0,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )


def clean_xy(
    frame: pd.DataFrame,
    train_mask: pd.Series,
    predict_mask: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    columns = post_da.feature_columns("rt")
    values, _ = base.clean_matrix(frame, columns, train_mask)
    train_valid = train_mask.to_numpy(bool) & frame["rt"].notna().to_numpy(bool)
    return (
        pd.DataFrame(values[train_valid], columns=columns),
        pd.DataFrame(values[predict_mask.to_numpy(bool)], columns=columns),
        frame.loc[train_valid, "rt"].to_numpy(float),
    )


def fit_solar_models(
    frame: pd.DataFrame,
    day: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    complete = frame["weather_complete"] & frame["power_complete"]
    solar = frame["period"].between(7, 16)
    train_mask = frame["market_date"].lt(day) & complete & solar
    predict_mask = frame["market_date"].eq(day) & complete & solar
    train_x, predict_x, train_y = clean_xy(frame, train_mask, predict_mask)

    segment_model = make_lgbm_regressor().fit(train_x, train_y)
    segment_prediction = np.asarray(segment_model.predict(predict_x), dtype=float)

    labels = regime_label(train_y)
    classifier = make_classifier().fit(train_x, labels)
    probability = np.asarray(classifier.predict_proba(predict_x), dtype=float)
    expert_predictions: list[np.ndarray] = []
    regime_counts: dict[str, int] = {}
    for regime_index, regime_name in enumerate(REGIME_NAMES):
        selected = labels == regime_index
        regime_counts[regime_name] = int(selected.sum())
        if int(selected.sum()) < 48:
            expert = make_lgbm_regressor().fit(train_x, train_y)
        else:
            expert = make_lgbm_regressor().fit(train_x.loc[selected], train_y[selected])
        expert_predictions.append(np.asarray(expert.predict(predict_x), dtype=float))
    expert_matrix = np.column_stack(expert_predictions)
    hard_regime = np.argmax(probability, axis=1)
    hard_prediction = expert_matrix[np.arange(len(expert_matrix)), hard_regime]
    soft_prediction = np.sum(expert_matrix * probability, axis=1)
    target = frame.loc[predict_mask, ["market_date", "period", "rt"]].sort_values("period").reset_index(drop=True)
    target = target.rename(columns={"rt": "rt_actual"})
    target["rt_solar_segment_lgbm_pred"] = segment_prediction
    target["rt_solar_moe_hard_pred"] = hard_prediction
    target["rt_solar_moe_soft_pred"] = soft_prediction
    target["predicted_regime"] = [REGIME_NAMES[index] for index in hard_regime]
    target["actual_regime"] = [REGIME_NAMES[index] for index in regime_label(target["rt_actual"].to_numpy(float))]
    for regime_index, regime_name in enumerate(REGIME_NAMES):
        target[f"prob_{regime_name}"] = probability[:, regime_index]
        target[f"expert_{regime_name}_pred"] = expert_matrix[:, regime_index]
    return target, regime_counts


def run(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[pd.DataFrame] = []
    counts: list[dict[str, Any]] = []
    dates = pd.date_range(start, end, freq="D")
    for index, day in enumerate(dates, start=1):
        result, regime_counts = fit_solar_models(frame, day)
        rows.append(result)
        counts.append({"market_date": day.date().isoformat(), **regime_counts})
        print(f"completed solar regime fold {index}/{len(dates)}: {day.date().isoformat()}", flush=True)
    return pd.concat(rows, ignore_index=True), counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Solar regime trial for post-DA real-time prediction")
    parser.add_argument("--start", default="2026-06-08")
    parser.add_argument("--backtest-start", default="2026-06-15")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=base.ROOT / "outputs" / "realtime_post_da_xgb_20260831",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base.ROOT / "outputs" / "realtime_solar_regime_20260831",
    )
    args = parser.parse_args()
    if not base.HAS_LIGHTGBM:
        raise RuntimeError("LightGBM is required; import integrated_price_forecast first to register the DLL path")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, coverage = base.load_price_weather(
        base.PRICE_DEFAULT, base.WEATHER_DEFAULT, sorted(base.ROOT.glob(base.POWER_GLOB))
    )
    frame = post_da.add_known_da_features(base.add_all_feature_tables(frame))
    solar, counts = run(frame, pd.Timestamp(args.start), pd.Timestamp(args.end))
    solar.to_csv(args.output_dir / "solar_oof_predictions.csv", index=False, encoding="utf-8-sig")

    baseline = pd.read_csv(args.baseline_dir / "all_oof_predictions.csv")
    baseline["market_date"] = pd.to_datetime(baseline["market_date"])
    solar["market_date"] = pd.to_datetime(solar["market_date"])
    merged = baseline.merge(solar, on=["market_date", "period"], how="left", suffixes=("", "_solar"))
    baseline_column = "rt_direct_lightgbm_l1_pred"
    for candidate in ("rt_solar_segment_lgbm_pred", "rt_solar_moe_hard_pred", "rt_solar_moe_soft_pred"):
        combined = candidate.replace("rt_solar", "rt_combined")
        merged[combined] = merged[candidate].fillna(merged[baseline_column])
    test = merged.loc[merged["market_date"].ge(pd.Timestamp(args.backtest_start))].copy()
    candidates = [
        baseline_column,
        "rt_combined_segment_lgbm_pred",
        "rt_combined_moe_hard_pred",
        "rt_combined_moe_soft_pred",
    ]
    scores = {candidate: base.metric(test["rt_actual"], test[candidate]) for candidate in candidates}
    solar_test = test.loc[test["period"].between(7, 16)]
    solar_scores = {
        candidate: base.metric(solar_test["rt_actual"], solar_test[candidate])
        for candidate in candidates
    }
    confusion = pd.crosstab(
        solar_test["actual_regime"], solar_test["predicted_regime"], normalize="index"
    ).fillna(0.0)
    merged.to_csv(args.output_dir / "combined_oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "backtest_period": {"start": args.backtest_start, "end": args.end},
        "overall": scores,
        "solar_period_7_to_16": solar_scores,
        "solar_regime_recall": confusion.to_dict(orient="index"),
        "training_regime_counts_by_day": counts,
        "data_coverage": coverage,
        "leakage_controls": [
            "Each solar specialist and classifier uses only dates before the target day.",
            "Target-day DA curve is allowed because this is a post-day-ahead-clearing forecast.",
            "No target-day RT price or spread is used by the models.",
        ],
        "status": "exploratory on a previously inspected interval; confirm on later unseen dates",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
