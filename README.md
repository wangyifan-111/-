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

测试区间为 2026-06-15 至 2026-06-30。日前出清后 LightGBM-L1 实时价格预测 MAE 为 77.59 元/MWh；该结果属于探索性回测，上线前应使用后续未见日期复核。
