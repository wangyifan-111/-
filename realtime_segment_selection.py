"""Pre-test segment-specific selector for post-DA real-time predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import integrated_price_forecast as base


SEGMENTS = {"night": (1, 6), "solar": (7, 16), "evening": (17, 24)}


def select_segment(calibration: pd.DataFrame, candidates: list[str]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for segment, (lo, hi) in SEGMENTS.items():
        group = calibration[calibration["period"].between(lo, hi)]
        scores = {candidate: float(np.mean(np.abs(group["rt_actual"] - group[candidate]))) for candidate in candidates}
        selected[segment] = min(scores, key=scores.get)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Select post-DA RT candidate by segment")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=base.ROOT / "outputs" / "realtime_post_da_xgb_20260831" / "all_oof_predictions.csv",
    )
    parser.add_argument("--calibration-start", default="2026-06-08")
    parser.add_argument("--calibration-end", default="2026-06-14")
    parser.add_argument("--backtest-start", default="2026-06-15")
    parser.add_argument("--backtest-end", default="2026-06-30")
    parser.add_argument(
        "--output-dir", type=Path, default=base.ROOT / "outputs" / "realtime_segment_selection_20260831"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.predictions)
    frame["market_date"] = pd.to_datetime(frame["market_date"])
    candidates = [column for column in frame if column.startswith("rt_") and column.endswith("_pred")]
    calibration = frame[frame["market_date"].between(args.calibration_start, args.calibration_end)]
    selected = select_segment(calibration, candidates)
    test = frame[frame["market_date"].between(args.backtest_start, args.backtest_end)].copy()
    test["rt_segment_selected_pred"] = np.nan
    for segment, (lo, hi) in SEGMENTS.items():
        mask = test["period"].between(lo, hi)
        test.loc[mask, "rt_segment_selected_pred"] = test.loc[mask, selected[segment]]
    baseline = "rt_direct_lightgbm_l1_pred"
    scores = {
        "baseline": base.metric(test["rt_actual"], test[baseline]),
        "segment_selected": base.metric(test["rt_actual"], test["rt_segment_selected_pred"]),
    }
    by_segment = {
        segment: {
            "selected_candidate": selected[segment],
            "baseline": base.metric(
                test.loc[test["period"].between(lo, hi), "rt_actual"],
                test.loc[test["period"].between(lo, hi), baseline],
            ),
            "selected": base.metric(
                test.loc[test["period"].between(lo, hi), "rt_actual"],
                test.loc[test["period"].between(lo, hi), "rt_segment_selected_pred"],
            ),
        }
        for segment, (lo, hi) in SEGMENTS.items()
    }
    test.to_csv(args.output_dir / "backtest_predictions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "calibration_period": {"start": args.calibration_start, "end": args.calibration_end},
        "backtest_period": {"start": args.backtest_start, "end": args.backtest_end},
        "selected_candidates": selected,
        "overall": scores,
        "by_segment": by_segment,
        "selection_rule": "lowest MAE on the pre-test calibration week, one candidate per fixed time segment",
        "status": "exploratory on a previously inspected interval; confirm on later unseen dates",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
