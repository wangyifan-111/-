import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_prices(path):
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    dates = pd.to_datetime(df.iloc[:, 0].astype(str))
    hours = df.iloc[:, 1].astype(str).str.slice(0, 2).astype(int)
    dt = dates + pd.to_timedelta(hours, unit="h")
    out = pd.DataFrame({"datetime": dt, "da": pd.to_numeric(df.iloc[:, 2]), "rt": pd.to_numeric(df.iloc[:, 3])})
    out = out.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    out[["da", "rt"]] = out[["da", "rt"]].interpolate(limit_direction="both")
    return out


def calendar_features(ts):
    hour = ts.hour
    dow = ts.dayofweek
    doy = ts.dayofyear
    feats = [1.0]
    feats += [float(hour == h) for h in range(1, 24)]
    feats += [float(dow == d) for d in range(1, 7)]
    feats += [math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24)]
    feats += [math.sin(2 * math.pi * dow / 7), math.cos(2 * math.pi * dow / 7)]
    feats += [math.sin(2 * math.pi * doy / 365.25), math.cos(2 * math.pi * doy / 365.25)]
    return feats


LAGS = [1, 2, 3, 24, 25, 48, 72, 168, 336]


def row_features(ts, history):
    if len(history) < max(LAGS):
        return None
    x = calendar_features(ts)
    x += [history[-lag] for lag in LAGS]
    a24 = np.asarray(history[-24:], dtype=float)
    a168 = np.asarray(history[-168:], dtype=float)
    x += [a24.mean(), a24.std(), a168.mean(), a168.std(), np.median(a168)]
    return np.asarray(x, dtype=float)


def fit_ridge(times, y, train_end, alpha=35.0):
    xs, ys, history = [], [], []
    for ts, value in zip(times, y):
        x = row_features(ts, history)
        if x is not None and ts <= train_end:
            xs.append(x)
            ys.append(value)
        history.append(value)
    X = np.vstack(xs)
    target = np.asarray(ys)
    mean = X[:, 1:].mean(axis=0)
    scale = X[:, 1:].std(axis=0)
    scale[scale < 1e-9] = 1.0
    Xs = X.copy()
    Xs[:, 1:] = (Xs[:, 1:] - mean) / scale
    penalty = np.eye(Xs.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(Xs.T @ Xs + penalty, Xs.T @ target)
    return beta, mean, scale


def ridge_forecast(times, values, train_end, future_times, alpha=35.0):
    beta, mean, scale = fit_ridge(times, values, train_end, alpha)
    history = [v for t, v in zip(times, values) if t <= train_end]
    preds = []
    for ts in future_times:
        x = row_features(ts, history)
        xs = x.copy()
        xs[1:] = (xs[1:] - mean) / scale
        pred = float(xs @ beta)
        pred = float(np.clip(pred, -100, 1300))
        preds.append(pred)
        history.append(pred)
    return np.asarray(preds)


def weekly_forecast(times, values, train_end, future_times):
    history = [v for t, v in zip(times, values) if t <= train_end]
    out = []
    for _ in future_times:
        pred = history[-168]
        out.append(pred)
        history.append(pred)
    return np.asarray(out)


def climatology_forecast(times, values, train_end, future_times):
    hist = pd.DataFrame({"t": times, "y": values})
    hist = hist[hist.t <= train_end].copy()
    hist["hour"] = hist.t.dt.hour
    hist["dow"] = hist.t.dt.dayofweek
    by_hd = hist.groupby(["hour", "dow"]).y.mean()
    by_h = hist.groupby("hour").y.mean()
    return np.asarray([by_hd.get((t.hour, t.dayofweek), by_h[t.hour]) for t in future_times])


def metrics(actual, pred):
    err = actual - pred
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(pred - actual)),
        "negative_accuracy": float(np.mean((pred < 0) == (actual < 0))),
        "high_price_recall": float(np.sum((pred > 500) & (actual > 500)) / max(1, np.sum(actual > 500))),
    }


def target_run(df, col):
    times = df.datetime
    values = df[col].to_numpy(float)
    train_end = pd.Timestamp("2026-05-31 23:00:00")
    test = df[df.datetime > train_end]
    test_times = list(test.datetime)
    actual = test[col].to_numpy(float)
    candidates = {
        "岭回归-日历滞后": ridge_forecast(times, values, train_end, test_times),
        "重复上周": weekly_forecast(times, values, train_end, test_times),
        "小时星期均值": climatology_forecast(times, values, train_end, test_times),
    }
    scores = {name: metrics(actual, pred) for name, pred in candidates.items()}
    best = min(scores, key=lambda k: scores[k]["mae"])
    # Blend the two strongest models when their errors are close; it is more stable for a full-month horizon.
    ranked = sorted(scores, key=lambda k: scores[k]["mae"])
    blend_name = f"组合({ranked[0]}+{ranked[1]})"
    blend_pred = 0.65 * candidates[ranked[0]] + 0.35 * candidates[ranked[1]]
    blend_score = metrics(actual, blend_pred)
    if scores[ranked[1]]["mae"] <= scores[ranked[0]]["mae"] * 1.12 and blend_score["mae"] < scores[ranked[0]]["mae"]:
        selected = blend_name
        test_pred = blend_pred
    else:
        selected = best
        test_pred = candidates[best]
    scores[selected] = metrics(actual, test_pred)

    full_end = df.datetime.iloc[-1]
    future = list(pd.date_range(full_end + pd.Timedelta(hours=1), "2026-08-01 00:00:00", freq="h"))
    full_candidates = {
        "岭回归-日历滞后": ridge_forecast(times, values, full_end, future),
        "重复上周": weekly_forecast(times, values, full_end, future),
        "小时星期均值": climatology_forecast(times, values, full_end, future),
    }
    if selected.startswith("组合"):
        forecast = 0.65 * full_candidates[ranked[0]] + 0.35 * full_candidates[ranked[1]]
    else:
        forecast = full_candidates[selected]

    residual = actual - test_pred
    test_hours = np.asarray([t.hour for t in test_times])
    lower, upper = [], []
    for ts, pred in zip(future, forecast):
        same = residual[test_hours == ts.hour]
        lo, hi = np.quantile(same, [0.1, 0.9])
        lower.append(max(-100.0, pred + lo))
        upper.append(min(1300.0, pred + hi))
    return {
        "scores": scores,
        "selected": selected,
        "test_metrics": scores[selected],
        "test_actual": actual,
        "test_pred": test_pred,
        "future": future,
        "forecast": forecast,
        "lower": np.asarray(lower),
        "upper": np.asarray(upper),
    }


def main():
    source, output_dir = Path(sys.argv[1]), Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_prices(source)
    da = target_run(df, "da")
    rt = target_run(df, "rt")
    forecast = pd.DataFrame({
        "datetime": da["future"],
        "da_p10": da["lower"], "da_p50": da["forecast"], "da_p90": da["upper"],
        "rt_p10": rt["lower"], "rt_p50": rt["forecast"], "rt_p90": rt["upper"],
    })
    forecast["spread_p50"] = forecast.da_p50 - forecast.rt_p50
    forecast["negative_risk"] = (forecast.da_p10 < 0) | (forecast.rt_p10 < 0)
    forecast["spike_risk"] = (forecast.da_p90 > 500) | (forecast.rt_p90 > 500)
    forecast["market_date"] = forecast.datetime.dt.normalize()
    forecast.loc[forecast.datetime.dt.hour == 0, "market_date"] -= pd.Timedelta(days=1)
    forecast["period"] = forecast.datetime.dt.hour.replace(0, 24)
    forecast = forecast[["market_date", "period", "datetime", "da_p10", "da_p50", "da_p90", "rt_p10", "rt_p50", "rt_p90", "spread_p50", "negative_risk", "spike_risk"]]
    forecast.to_csv(output_dir / "forecast.csv", index=False, encoding="utf-8-sig")

    daily = forecast.groupby("market_date").agg(
        da_mean=("da_p50", "mean"), rt_mean=("rt_p50", "mean"),
        da_min=("da_p50", "min"), da_max=("da_p50", "max"),
        rt_min=("rt_p50", "min"), rt_max=("rt_p50", "max"),
        negative_risk_hours=("negative_risk", "sum"), spike_risk_hours=("spike_risk", "sum"),
    ).reset_index()
    daily.to_csv(output_dir / "daily.csv", index=False, encoding="utf-8-sig")
    hourly = forecast.groupby("period").agg(
        da_p50=("da_p50", "mean"), rt_p50=("rt_p50", "mean"),
        da_p10=("da_p10", "mean"), da_p90=("da_p90", "mean"),
        rt_p10=("rt_p10", "mean"), rt_p90=("rt_p90", "mean"),
    ).reset_index()
    hourly.to_csv(output_dir / "hourly.csv", index=False, encoding="utf-8-sig")

    backtest = pd.DataFrame({
        "datetime": df[df.datetime >= "2026-06-01"].datetime,
        "da_actual": da["test_actual"], "da_pred": da["test_pred"],
        "rt_actual": rt["test_actual"], "rt_pred": rt["test_pred"],
    })
    backtest.to_csv(output_dir / "backtest.csv", index=False, encoding="utf-8-sig")
    summary = {
        "source_rows": len(df), "source_start": str(df.datetime.min()), "source_end": str(df.datetime.max()),
        "forecast_start": str(forecast.datetime.min()), "forecast_end": str(forecast.datetime.max()),
        "da_selected": da["selected"], "rt_selected": rt["selected"],
        "da_scores": da["scores"], "rt_scores": rt["scores"],
        "forecast_summary": {
            "da_mean": float(forecast.da_p50.mean()), "rt_mean": float(forecast.rt_p50.mean()),
            "da_min": float(forecast.da_p50.min()), "da_max": float(forecast.da_p50.max()),
            "rt_min": float(forecast.rt_p50.min()), "rt_max": float(forecast.rt_p50.max()),
            "negative_risk_hours": int(forecast.negative_risk.sum()),
            "spike_risk_hours": int(forecast.spike_risk.sum()),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
