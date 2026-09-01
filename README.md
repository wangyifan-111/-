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

本仓库不包含原始价格、天气、电源出力或客户用电数据。运行代码时，请将数据文件放在本地目录，并根据代码中的路径配置使用。

测试区间为 2026-06-15 至 2026-06-30。日前出清后 LightGBM-L1 实时价格预测 MAE 为 77.59 元/MWh；该结果属于探索性回测，上线前应使用后续未见日期复核。
