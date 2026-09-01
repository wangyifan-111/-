"""Bridge the day-ahead and real-time spread models to the /api/v1 contract.

The real-time forecast is produced as ``day-ahead forecast + forecasted
real-time-minus-day-ahead spread``. Trading actions remain HOLD-only.
"""
from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd

from realtime_spread_forecast import run_spread_forecast
from weather_price_forecast_optimized import run_weather_forecast


ROOT = Path(__file__).resolve().parent
PRICE_GLOB = "山东省-现货价格-*.xlsx"
WEATHER_GLOB = "分时天气预报-*.xlsx"
POWER_GLOB = "山东省-电源出力*.xlsx"


MISSING_DOMAINS = [
    "clearing_results",
    "complete_settlement",
    "congestion",
    "day_ahead_declarations",
    "deviation_assessment",
    "manual_adjustments",
    "market_rules",
    "medium_long_term_positions",
    "outages",
    "real_time_declarations",
    "retail_contracts",
    "trading_limits",
    "unit_status",
]


def _level(flag: bool) -> str:
    return "HIGH" if flag else "LOW"


def _first_matching(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"未找到输入文件: {directory / pattern}")
    return matches[0]


def _read_model_card(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    card = json.loads(path.read_text(encoding="utf-8"))
    comparison_path = path.with_name("model_comparison.json")
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        for key in ("selected_model", "final_test_start", "final_test_end", "final_test_scores"):
            if key not in card and key in comparison:
                card[key] = comparison[key]
    return card


def _score_from_card(card: dict[str, Any], selected_model: str) -> dict[str, Any]:
    score = card.get("final_test_scores", {}).get(selected_model, {})
    return {
        "window_start": card.get("final_test_start"),
        "window_end": card.get("final_test_end"),
        "sample_count": int(card.get("coverage", {}).get("price_rows", 0)),
        "mae_yuan_per_mwh": score.get("mae_yuan_per_mwh"),
        "rmse_yuan_per_mwh": score.get("rmse_yuan_per_mwh"),
        "bias_yuan_per_mwh": score.get("bias_yuan_per_mwh"),
        "negative_price_recall": score.get("negative_price_recall"),
        "extreme_high_price_recall": score.get("high_price_recall_gt_500"),
    }


def build_result(
    raw: dict[str, Any],
    spread_raw: dict[str, Any],
    card: dict[str, Any],
    request_id: str,
    run_id: str,
    market_date: str,
) -> dict[str, Any]:
    rows = raw.get("forecast", [])
    if len(rows) != 24:
        raise ValueError(f"模型返回 {len(rows)} 个时段；v1接口要求固定24个时段")
    spread_rows = spread_raw.get("forecast", [])
    if len(spread_rows) != 24:
        raise ValueError(f"实时价差模型返回 {len(spread_rows)} 个时段；v1接口要求固定24个时段")
    dates = {pd.Timestamp(row.get("market_date")).date().isoformat() for row in rows}
    if dates != {market_date}:
        raise ValueError(f"模型预测日期为 {sorted(dates)}，与请求日期 {market_date} 不一致")

    periods = []
    for row, spread_row in zip(rows, spread_rows):
        da = {
            "p10": float(row["p10"]),
            "p50": float(row["p50"]),
            "p90": float(row["p90"]),
        }
        rt = {
            "p10": da["p10"] + float(spread_row["spread_p10"]),
            "p50": da["p50"] + float(spread_row["spread_p50"]),
            "p90": da["p90"] + float(spread_row["spread_p90"]),
        }
        negative = da["p10"] < 0 or rt["p10"] < 0
        high = da["p90"] > 500 or rt["p90"] > 500
        reasons = []
        if da["p10"] < 0:
            reasons.append("DAY_AHEAD_P10_CROSSED_ZERO")
        if rt["p10"] < 0:
            reasons.append("REAL_TIME_P10_CROSSED_ZERO")
        if da["p90"] > 500:
            reasons.append("DAY_AHEAD_P90_EXCEEDED_500")
        if rt["p90"] > 500:
            reasons.append("REAL_TIME_P90_EXCEEDED_500")
        if not reasons:
            reasons.append("NO_THRESHOLD_RISK")
        periods.append(
            {
                "period": int(row["period"]),
                "datetime": pd.Timestamp(row["datetime"]).tz_localize("Asia/Shanghai").isoformat()
                if pd.Timestamp(row["datetime"]).tzinfo is None
                else pd.Timestamp(row["datetime"]).isoformat(),
                "day_ahead_price_yuan_per_mwh": da,
                "real_time_price_yuan_per_mwh": rt,
                "spread_day_ahead_minus_real_time_yuan_per_mwh": round(da["p50"] - rt["p50"], 6),
                "negative_price_risk": {
                    "flagged": negative,
                    "probability": None,
                    "level": _level(negative),
                    "method": "P10_CROSSED_ZERO",
                },
                "high_price_risk": {
                    "flagged": high,
                    "probability": None,
                    "level": _level(high),
                    "method": "P90_EXCEEDED_500",
                },
                "risk_reason_codes": reasons,
                "confidence": None,
                "data_completeness": "PARTIAL",
                "strategy_suggestion": {
                    "action": "HOLD",
                    "volume_mwh": 0,
                    "price_yuan_per_mwh": None,
                    "confidence": None,
                    "reason_codes": ["STRATEGY_INPUTS_INCOMPLETE"],
                },
            }
        )

    selected_model = str(raw.get("selected_model", card.get("selected_model", "unknown")))
    coverage = raw.get("coverage", {})
    return {
        "schema_version": "1.0.0",
        "request_id": request_id,
        "run_id": run_id,
        "market_code": "SD",
        "market_date": market_date,
        "timezone": "Asia/Shanghai",
        "model": {
            "id": "price-forecast",
            "name": "山东日前电价预测（天气+电源出力增强）",
            "version": str(raw.get("model_version", "weather-power-price-optimized-v1.0.0")),
            "selected_model": selected_model,
            "realtime_model": {
                "target": "real_time_minus_day_ahead_yuan_per_mwh",
                "version": str(spread_raw.get("model_version", "realtime-spread-random-forest-v1.0.0")),
                "selected_model": str(spread_raw.get("selected_model", "random_forest_weather_power")),
            },
        },
        "data_snapshot": {
            "version": "sd-weather-power-2026h1-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_versions": {
                "prices": "sd-spot-2026h1-v1",
                "weather_forecast": "sd-gfs-weather-2026h1-v1",
                "power_forecast": "sd-power-output-2026h1-v1",
            },
            "available_domains": [
                "prices",
                "weather_forecast",
                "load_forecast_proxy",
                "wind_forecast_proxy",
                "pv_forecast_proxy",
                "day_ahead_price",
                "real_time_price_history",
                "real_time_spread_forecast",
                "real_time_price_forecast",
            ],
            "missing_domains": MISSING_DOMAINS,
            "coverage": coverage,
        },
        "backtest": {
            **_score_from_card(card, selected_model),
            "real_time_spread": spread_raw.get("summary", {}).get("validation", {}),
        },
        "periods": periods,
        "strategy_ready": False,
        "warnings": [
            "REAL_TIME_PRICE_DERIVED_FROM_SPREAD_MODEL",
            "WEATHER_FORECAST_PUBLICATION_TIMESTAMP_MISSING",
            "POWER_FORECAST_PUBLICATION_TIMESTAMP_MISSING",
            "STRATEGY_INPUTS_INCOMPLETE_HOLD_ONLY",
            "EXECUTION_DISABLED",
        ],
    }


def _submit_run(base_url: str, headers: dict[str, str], payload: dict[str, Any]) -> str:
    request = Request(
        f"{base_url.rstrip('/')}/api/v1/models/price-forecast/runs",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return str(json.load(response)["run_id"])


def _submit_result(base_url: str, headers: dict[str, str], run_id: str, result: dict[str, Any]) -> None:
    request = Request(
        f"{base_url.rstrip('/')}/api/v1/model-runs/{run_id}/results",
        data=json.dumps(result, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="PUT",
    )
    with urlopen(request, timeout=30) as response:
        response.read()


def run_platform_forecast(
    price_path: Path,
    weather_path: Path,
    power_paths: list[Path],
    model_card_path: Path,
    market_date: str,
    request_id: str = "local-request",
    run_id: str = "local-run",
) -> dict[str, Any]:
    """Run the local model and return the same result payload used by /api/v1."""
    raw = run_weather_forecast(
        price_path=price_path,
        weather_path=weather_path,
        forecast_start=pd.Timestamp(market_date),
        forecast_end=pd.Timestamp(market_date),
        model_card_path=model_card_path,
        power_paths=power_paths,
    )
    spread_raw = run_spread_forecast(price_path, weather_path, market_date, power_paths)
    return build_result(raw, spread_raw, _read_model_card(model_card_path), request_id, run_id, market_date)


def main() -> None:
    parser = argparse.ArgumentParser(description="天气+电源出力增强模型 /api/v1 适配器")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--api-key")
    parser.add_argument("--market-date", required=True, help="目标市场日期，例如 2026-07-01")
    parser.add_argument("--price", type=Path)
    parser.add_argument("--weather", type=Path)
    parser.add_argument("--power-dir", type=Path)
    parser.add_argument("--model-card", type=Path)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data_dir = ROOT / "data"
    price = args.price or _first_matching(data_dir, PRICE_GLOB)
    weather = args.weather or _first_matching(data_dir, WEATHER_GLOB)
    power_dir = args.power_dir or data_dir
    power_paths = sorted(power_dir.glob(POWER_GLOB))
    if not power_paths:
        raise FileNotFoundError(f"未找到电源出力文件: {power_dir / POWER_GLOB}")
    model_card = args.model_card or ROOT / "model_card.json"

    raw = run_weather_forecast(
        price_path=price,
        weather_path=weather,
        forecast_start=pd.Timestamp(args.market_date),
        forecast_end=pd.Timestamp(args.market_date),
        model_card_path=model_card,
        power_paths=power_paths,
    )
    spread_raw = run_spread_forecast(price, weather, args.market_date, power_paths)
    request_id = f"sd-{args.market_date}-price-forecast-{uuid.uuid4().hex[:8]}"
    run_id = "dry-run-" + uuid.uuid4().hex[:12]
    headers = {"Content-Type": "application/json", "X-Request-ID": request_id}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    if args.submit:
        run_id = _submit_run(
            args.base_url,
            headers,
            {
                "asset_id": "default-price-history",
                "request_id": request_id,
                "market_code": "SD",
                "market_date": args.market_date,
                "model_id": "price-forecast",
                "model_version": raw["model_version"],
                "data_version": "sd-weather-power-2026h1-v1",
                "parameters": {
                    "quantiles": [0.1, 0.5, 0.9],
                    "horizon_hours": 24,
                    "realtime_model_version": spread_raw["model_version"],
                    "spread_target": "real_time_minus_day_ahead_yuan_per_mwh",
                },
                "input_summary": {
                    "available_domains": ["prices", "real_time_price_history", "weather_forecast", "load_forecast_proxy", "wind_forecast_proxy", "pv_forecast_proxy"],
                    "hourly_point_count": 24,
                    "customer_scope": "portfolio",
                },
                "timeout_seconds": 120,
            },
        )

    result = build_result(raw, spread_raw, _read_model_card(model_card), request_id, run_id, args.market_date)
    if args.submit:
        _submit_result(args.base_url, headers, run_id, result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"submitted": args.submit, "run_id": run_id, "periods": len(result["periods"]), "strategy_ready": result["strategy_ready"], "model_version": result["model"]["version"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
