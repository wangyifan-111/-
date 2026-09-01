"""Bridge the local price model to the platform's /api/v1 provider contract.

The default mode is dry-run. Use --submit only after the platform owner provides
the reachable test URL and confirms the run/review workflow.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "power-trading-platform" / "backend"))
from model_adapter import find_default_source, run_price_forecast  # noqa: E402


MISSING_DOMAINS = [
    "clearing_results", "complete_settlement", "congestion", "day_ahead_declarations",
    "deviation_assessment", "load_forecast", "manual_adjustments", "market_rules",
    "medium_long_term_positions", "outages", "real_time_declarations", "renewable_forecast",
    "retail_contracts", "trading_limits", "unit_status", "weather",
]


def level(probability: float) -> str:
    if probability >= 0.5:
        return "HIGH"
    if probability >= 0.2:
        return "MEDIUM"
    return "LOW"


def build_result(raw: dict[str, Any], request_id: str, run_id: str, market_date: str) -> dict[str, Any]:
    rows = raw["forecast"][:24]
    if len(rows) != 24:
        raise ValueError(f"model returned {len(rows)} points; v1 requires exactly 24")
    # The current dataset has prices and actual load only. The platform contract
    # therefore requires partial completeness and HOLD-only suggestions.
    periods = []
    for i, row in enumerate(rows, 1):
        da = {"p10": row["da_p10"], "p50": row["da_p50"], "p90": row["da_p90"]}
        rt = {"p10": row["rt_p10"], "p50": row["rt_p50"], "p90": row["rt_p90"]}
        negative = float(row["negative_risk"])
        spike = float(row["spike_risk"])
        reasons = []
        if negative:
            reasons.append("NEGATIVE_INTERVAL_CROSSED")
        if spike:
            reasons.append("HIGH_PRICE_INTERVAL_CROSSED")
        if not reasons:
            reasons.append("NO_THRESHOLD_RISK")
        periods.append({
            "period": i,
            "datetime": datetime.fromisoformat(row["datetime"]).replace(tzinfo=ZoneInfo("Asia/Shanghai")).isoformat(),
            "day_ahead_price_yuan_per_mwh": da,
            "real_time_price_yuan_per_mwh": rt,
            "spread_day_ahead_minus_real_time_yuan_per_mwh": round(da["p50"] - rt["p50"], 6),
            "negative_price_risk": {"probability": negative, "level": level(negative)},
            "high_price_risk": {"probability": spike, "level": level(spike)},
            "risk_reason_codes": reasons,
            "confidence": 0.5,
            "data_completeness": "PARTIAL",
            "strategy_suggestion": {
                "action": "HOLD", "volume_mwh": 0, "price_yuan_per_mwh": None,
                "confidence": 0.5,
                "reason_codes": ["STRATEGY_INPUTS_INCOMPLETE"],
            },
        })
    da_scores = raw["summary"]["da_metrics"]
    rt_scores = raw["summary"]["rt_metrics"]
    return {
        "schema_version": "1.0.0",
        "request_id": request_id,
        "run_id": run_id,
        "market_code": "SD",
        "market_date": market_date,
        "timezone": "Asia/Shanghai",
        "model": {"id": "price-forecast", "name": "山东现货价格组合预测", "version": "price-forecast-v1.0.0"},
        "data_snapshot": {
            "version": "sd-hourly-snapshot-2026h1-v1",
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "source_versions": {"prices": "sd-spot-2026h1-v1", "portfolio_load": "sd-portfolio-load-2026h1-v1"},
            "available_domains": ["load_actual", "prices"],
            "missing_domains": MISSING_DOMAINS,
        },
        "backtest": {
            "window_start": raw["summary"]["forecast_start"][:10],
            "window_end": raw["summary"]["forecast_end"][:10],
            "sample_count": 0,
            "mae_yuan_per_mwh": da_scores["mae"],
            "rmse_yuan_per_mwh": da_scores["rmse"],
            "bias_yuan_per_mwh": da_scores["bias"],
            "negative_price_direction_accuracy": da_scores["negative_accuracy"],
            "extreme_high_price_recall": da_scores["high_price_recall"],
        },
        "periods": periods,
        "strategy_ready": False,
        "warnings": ["STRATEGY_INPUTS_INCOMPLETE_HOLD_ONLY", "EXECUTION_DISABLED"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--market-date", default="2026-07-01")
    parser.add_argument("--api-key")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    request_id = f"sd-{args.market_date}-price-forecast-{uuid.uuid4().hex[:8]}"
    headers = {"Content-Type": "application/json", "X-Request-ID": request_id}
    if args.api_key:
        headers["X-API-Key"] = args.api_key
    raw = run_price_forecast(find_default_source(), horizon_hours=24, backtest_days=30)
    forecast_date = raw["forecast"][0]["datetime"][:10]
    if args.market_date != forecast_date:
        raise ValueError(f"market date {args.market_date} does not match model horizon date {forecast_date}; use a matching data snapshot")
    run_id = "dry-run-" + uuid.uuid4().hex[:12]
    if args.submit:
        payload = {
            "request_id": request_id, "market_code": "SD", "market_date": args.market_date,
            "model_id": "price-forecast", "model_version": "price-forecast-v1.0.0",
            "data_version": "sd-hourly-snapshot-2026h1-v1",
            "parameters": {"quantiles": [0.1, 0.5, 0.9]},
            "input_summary": {"available_domains": ["prices", "load_actual"], "hourly_point_count": 24, "customer_scope": "portfolio"},
            "timeout_seconds": 120,
        }
        req = Request(f"{args.base_url}/api/v1/models/price-forecast/runs", data=json.dumps(payload).encode(), headers=headers, method="POST")
        with urlopen(req, timeout=30) as response:
            run_id = json.load(response)["run_id"]
    result = build_result(raw, request_id, run_id, args.market_date)
    if args.submit:
        req = Request(f"{args.base_url}/api/v1/model-runs/{run_id}/results", data=json.dumps(result).encode(), headers=headers, method="PUT")
        with urlopen(req, timeout=30) as response:
            response.read()
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"submitted": args.submit, "run_id": run_id, "periods": len(result["periods"]), "strategy_ready": result["strategy_ready"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
