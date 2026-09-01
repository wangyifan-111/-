# 山东价格预测

山东电力日前价格与实时价格预测模型及回测代码。

## 内容

- `integrated_price_forecast.py`：日前价格、实时价格、价差、风险区间与一致性预测框架。
- `realtime_post_da_forecast.py`：日前出清后，使用已公布日前曲线预测实时价格。
- `realtime_lightgbm_tuning.py`：LightGBM 目标函数、训练窗口和树参数调优。
- `realtime_post_da_correction.py`：滚动误差校正试验。
- `realtime_segment_selection.py`：分时段候选模型选择试验。
- `realtime_solar_regime_trial.py`：光伏时段状态专家试验。
- `requirements_integrated_price_forecast.txt`：Python 依赖。
- `实时价格预测调优结果_20260901.md`：当前回测结论。

## 重要说明

本仓库现为 Private，并包含本项目复现所需的价格、天气和电源出力数据。数据仅限获授权的协作者使用，不应再次公开或转发。

## 目录

- `data/`：山东现货价格、天气预测、风光出力和客户负荷数据。
- `src/`：传统模型、LightGBM/XGBoost、融合模型及 Transformer/CNN/LSTM/Mamba 试验代码。
- `reports/`：模型对比、误差、金融属性和接口接入记录。
- `examples/`：接口请求、预测结果和复现样例。

## 复现

先安装依赖，再按报告中的数据路径运行 `src/` 下的脚本。部分深度学习模型需要 PyTorch；LightGBM/XGBoost 版本应与 `requirements_integrated_price_forecast.txt` 保持一致。

从仓库根目录复现主结果（固定测试区间、随机种子和数据目录）：

```powershell
python integrated_price_forecast.py --backend xgboost --backtest-start 2026-06-15 --backtest-end 2026-06-30 --forecast-date 2026-07-01 --output-dir outputs/repro_xgboost
```

结果写入 `outputs/repro_xgboost/`，其中 `run_summary.json` 保存 MAE/RMSE 等指标，`walk_forward_backtest.csv` 保存逐时预测和真实值。由于训练包含滚动窗口，首次运行可能需要数分钟。

上述命令复现的是盘前一体化预测口径。本仓库此前记录的较低实时误差（约 `77.59` 元/MWh）属于“日前出清曲线已公布后”的实时预测口径，允许把当日完整日前曲线作为输入；它不能与盘前口径直接比较。相应复现命令为：

```powershell
python realtime_post_da_forecast.py --backtest-start 2026-06-15 --backtest-end 2026-06-30 --output-dir outputs/repro_post_da
```

已于 2026-09-01 用本仓库数据完整验证：该命令在 384 个时段上的最佳单模型为 `rt_direct_lightgbm_l1_pred`，实时价格 MAE 为 `77.593652 元/MWh`，RMSE 为 `115.810585 元/MWh`。完整指标和逐时预测分别见 `outputs/repro_post_da/summary.json` 与 `outputs/repro_post_da/backtest_predictions.csv`。

测试区间为 2026-06-15 至 2026-06-30。日前出清后 LightGBM-L1 实时价格预测 MAE 为 77.59 元/MWh；该结果属于探索性回测，上线前应使用后续未见日期复核。
