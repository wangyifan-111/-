# Transformer 模型试验与融合结论

## 模型形式

本次测试的不是表格模型换名，而是基于 PyTorch 的多变量序列到序列 Transformer：

- 编码器输入：目标日前 168 小时的日前价格、实时价格、价差、天气和电源出力序列；
- 解码器输入：目标日 24 小时可提前获得的时间、天气预报和电源出力特征；
- 输出：未来 24 小时日前价格和实时价差；
- 实时价格：`日前价格 + 实时价差`；
- 最终结构：1 层编码器、1 层解码器、32 维隐层、4 个注意力头。

网络结构在 2026-06-08 至 2026-06-14 上选择，正式结果在随后 2026-06-15 至 2026-06-30 的 384 个小时上滚动回测。每个目标日只使用该日前数据。

## Transformer 单模型结果

| 目标 | MAE（元/MWh） | RMSE（元/MWh） |
|---|---:|---:|
| 日前价格 | 57.38 | 78.20 |
| 实时价格 | 100.76 | 138.30 |
| 实时价差 | 82.10 | 118.54 |

Transformer 单独预测日前和实时价格没有超过树模型，但价差 MAE 优于此前 Ridge 的 83.54，说明它对价差的动态变化有补充信息。

## 回测前固定权重的融合结果

权重完全使用 6 月 8–14 日确定：

```text
实验性日前融合 = 73% XGBoost + 27% Transformer
价差融合       = 79% Ridge + 21% Transformer
```

| 目标 | 原方案 MAE | Transformer 融合 MAE | 改善 |
|---|---:|---:|---:|
| 日前价格 | 47.12 | 46.22 | 1.91% |
| 实时价差 | 83.54 | 81.71 | 2.20% |
| 一致实时价格 | 87.93 | 86.42 | 1.72% |

## 统计检验与最终选择

使用 16 个交易日作为配对区块，进行 10,000 次 Bootstrap：

- 价差 MAE 改善的 95% 区间为 `0.43 至 3.35 元/MWh`，改善概率 99.62%；
- 日前 MAE 改善区间为 `-2.19 至 4.04 元/MWh`，区间跨 0；
- 一致实时价格改善区间同样跨 0。

因此最终生产方案只在统计证据较明确的价差环节引入 Transformer：

```text
日前价格 = XGBoost
实时价差 = 79% Ridge + 21% Transformer
实时价格 = XGBoost 日前价格 + 融合价差
```

该生产方案回测指标为：日前 MAE 47.12、价差 MAE 81.71、实时价格 MAE 86.80 元/MWh。实验性的日前融合仍保留在输出中作为 benchmark，但不作为正式结果。

## 文件

- [Transformer 代码](/D:/电力实习/transformer_price_forecast.py)
- [回测及结构选择摘要](/D:/电力实习/outputs/transformer_trial_20260827/transformer_summary.json)
- [回测预测明细](/D:/电力实习/outputs/transformer_trial_20260827/walk_forward_predictions.csv)
- [回测前权重校准](/D:/电力实习/outputs/transformer_trial_20260827/precalibrated_blend_summary.json)
- [统计检验](/D:/电力实习/outputs/transformer_trial_20260827/statistical_comparison.json)
- [最终 24 时段预测](/D:/电力实习/outputs/transformer_enhanced_forecast_20260827/forecast.csv)
- [最终模型卡片](/D:/电力实习/outputs/transformer_enhanced_forecast_20260827/model_card.json)
