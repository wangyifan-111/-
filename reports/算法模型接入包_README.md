# 王伊梵价格预测模型接入包

## 内容

- `price_forecast.py`：价格预测模型
- `weather_price_forecast_optimized.py`：天气 + 电源出力增强日前电价模型；使用山东 GFS 预报、价格滞后、统调负荷、风电、光伏和净负荷代理特征
- `platform_v1_adapter.py`：转换为陆璟行平台 `/api/v1` 正式结果协议并回传
- `power-trading-platform/backend/weather_model_adapter.py`：天气模型的可选平台适配器，默认仍为 HOLD
- `forecast-strategy-result-v1.dry-run.json`：24点 dry-run 示例

## 环境

- Python 3.10+
- `numpy`、`pandas`、`openpyxl`、`scikit-learn`、`joblib`
- 山东历史价格 Excel 文件放在项目根目录，文件名匹配 `山东省-现货价格-*.xlsx`

## 本地验证

```powershell
python platform_v1_adapter.py --market-date 2026-07-01 --output result.json
```

默认只生成本地结果，不写入平台。

天气模型本地回测与 24 点预测：

```powershell
$env:PYTHONPATH='.codex_deps'
py weather_price_forecast_optimized.py `
  --price '山东省-现货价格-数据明细（2026-01-01_2026-06-30.xlsx' `
  --weather '分时天气预报-自定义-山东省-2026-01-01-2026-07-01.xlsx' `
  --output-dir 'outputs/weather_price_optimized_202608' `
  --forecast-start 2026-07-01 `
  --forecast-end 2026-07-01
```

脚本默认自动读取项目根目录中 6 个 `山东省-电源出力*.xlsx` 文件：1–3 月实际出力、4–6 月预测出力。15 分钟数据会先聚合到第 1–24 市场时段，回测时只使用目标日前的历史行和目标日的预测出力行。

如果只需要复现不含电源出力的天气模型，可加 `--no-power`。

## 提交测试平台

拿到可访问的测试地址和 API Key 后运行：

```powershell
python platform_v1_adapter.py `
  --base-url http://<远程测试地址> `
  --api-key <测试密钥> `
  --market-date 2026-07-01 `
  --submit `
  --output result.json
```

脚本会：

1. 创建 `price-forecast` 模型运行；
2. 在本地执行模型；
3. 回传固定24点正式结果；
4. 输出 P10/P50/P90、价差、风险、回测指标和数据快照；
5. 在缺少持仓、合同、申报、成交、结算等数据时强制输出 `HOLD`。

## 当前限制

- 当前模型数据只覆盖到 2026-07-01，不能直接代表当前日期预测。
- 天气文件没有预报发布时间；上线前应补充预报快照时间和提前量，重新验证日前可用性。
- 电源出力文件没有预测发布时间；4–6 月数据按“日前预测出力”处理，正式上线前应补充发布时间和数据版本。
- 当未来目标日没有对应电源出力文件时，模型使用上一周同日同市场时段作为显式代理，并在模型卡中记录 `power_proxy_rows`。
- 天气模型的最终留出测试窗口为 2026-06-15 至 2026-06-30，正式切换前应在新的时间窗口持续回测。
- 当前策略输出固定为 `HOLD`，`strategy_ready=false`，`execution_allowed=false`。
- 真实平台联调前必须由平台方提供远程测试地址、鉴权方式和测试数据版本。
