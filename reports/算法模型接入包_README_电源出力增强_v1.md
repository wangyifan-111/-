# 山东日前电价预测模型接入包（天气 + 电源出力增强版）

本包按照之前约定的 `/api/v1` 模型运行与结果协议整理，替换旧适配器后可以使用同样的接口路径：

- `POST /api/v1/models/price-forecast/runs`
- `PUT /api/v1/model-runs/{run_id}/results`

## 本次更新

当前模型为 `weather-power-price-optimized-v1.0.0`，具体形式是天气与电源出力增强的随机森林回归模型。输入包括：

- 历史日前电价滞后与滚动统计；
- 山东分时天气预报；
- 直调负荷、风电、光伏、可再生能源占比和净负荷代理值。

回测结果：滚动验证 MAE 为 `55.98 元/MWh`；最终留出测试集 MAE 为 `62.02 元/MWh`，RMSE 为 `77.46 元/MWh`。

实时价格部分单独训练 `realtime-spread-random-forest-v1.0.0`，目标是 `实时价格 − 日前价格`。接口最终使用“日前价格预测 + 实时价差预测”得到实时价格预测区间；实时价差模型在2026年6月17—30日历史窗口上的MAE约为 `84.48 元/MWh`。

## 本地运行

环境：Python 3.10+。

```powershell
pip install -r requirements.txt
python platform_v1_weather_power_adapter.py `
  --market-date 2026-07-01 `
  --output examples/forecast-strategy-result-v1.weather-power.dry-run.json
```

命令会生成固定24个市场时段的结果。每个时段包含日前电价、实时电价和日前/实时价差的预测区间、低价/高价风险标识以及 `HOLD` 策略建议。

## 提交测试平台

拿到同事平台的测试地址和鉴权信息后，再使用：

```powershell
python platform_v1_weather_power_adapter.py `
  --base-url http://<测试地址> `
  --api-key <测试密钥> `
  --market-date 2026-07-01 `
  --submit `
  --output examples/forecast-strategy-result-v1.weather-power.submitted.json
```

适配器会先创建模型运行，再回传结果。提交地址和密钥不要写进代码或压缩包。

## 在平台后端直接调用

如果同事的平台后端希望直接调用函数，而不是启动命令行进程，可以导入：

```python
from pathlib import Path
from platform_v1_weather_power_adapter import run_platform_forecast

result = run_platform_forecast(
    price_path=Path("data/山东省-现货价格-数据明细（2026-01-01_2026-06-30.xlsx"),
    weather_path=Path("data/分时天气预报-自定义-山东省-2026-01-01-2026-07-01.xlsx"),
    power_paths=sorted(Path("data").glob("山东省-电源出力*.xlsx")),
    model_card_path=Path("model_card.json"),
    market_date="2026-07-01",
    request_id="sd-2026-07-01-price-forecast-demo",
    run_id="local-run-demo",
)
```

将返回的 `result` 直接作为原 `/api/v1/model-runs/{run_id}/results` 的结果对象保存即可。平台端只需要把原来的旧模型调用替换为这个函数，并传入同一数据快照路径。

## 接口说明

请求仍使用 `price-forecast` 模型标识和 `/api/v1` 路径。结果协议保持原字段结构：

- `model.version`：当前模型版本；
- `periods`：固定24个市场时段；
- `day_ahead_price_yuan_per_mwh`：日前价格预测区间；
- `real_time_price_yuan_per_mwh`：由日前预测与实时价差预测相加得到 `p10/p50/p90`；
- `spread_day_ahead_minus_real_time_yuan_per_mwh`：日前预测中位数减去实时预测中位数；
- `strategy_suggestion.action`：固定为 `HOLD`；
- `strategy_ready`：固定为 `false`。

实时价不是直接复制日前价，而是由独立的实时价差模型推导。该模型的输入仍然只使用目标日前可获得的历史价差、天气和电源出力特征。

## 数据口径与上线边界

- 天气和电源出力文件没有预测发布时间，当前按目标日前可获得处理；
- 目标日没有对应电源出力文件时，模型使用上一周同日同市场时段作为显式代理，并在模型卡中记录；
- 当前只支持预测与风险提示，不支持自动申报、自动交易或结算；
- 正式上线前需要补充数据快照时间、版本和远程测试数据，完成无泄漏回测。
