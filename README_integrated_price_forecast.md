# 一体化电价预测模型

入口脚本：`integrated_price_forecast.py`

该脚本把以下任务放在同一个、可审计的预测协议中：

1. 日前价格 `DA` 回归；
2. 实时价格 `RT` 回归；
3. 实时价差 `spread = RT - DA` 回归；
4. 价差方向分类：`spread >= 0` 为正方向；
5. 日前/实时/价差的P10、P50、P90预测区间；
6. TCN时序残差修正；
7. 90%名义覆盖率的Conformal Prediction区间。

## 模型形式

主回归器按优先级自动选择：

1. LightGBM：主点预测和分位数回归；
2. XGBoost：主点预测；分位数使用Conformal校准；
3. NumPy ridge回退：仅用于没有安装上述依赖的环境。

残差学习按优先级自动选择：

1. PyTorch实现的两层膨胀卷积TCN；
2. 无PyTorch时使用24小时残差窗口的ridge序列回退。

实时价格不直接把日前价格当作实时价格，而是使用明确恒等式：

```text
RT预测 = DA预测 + (RT - DA)价差预测
```

脚本同时训练一个直接RT回归器作为对照，但接口中的正式RT结果采用上述恒等式，保证日前、实时和价差三者在数值上保持一致。

## 运行

建议在同一Python环境安装：

```powershell
pip install lightgbm xgboost
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Windows 如果出现 `lib_lightgbm.dll` 找不到或 `VCOMP140.DLL` 缺失，需要安装 Microsoft Visual C++ 2015–2022 Redistributable（x64）。脚本会自动加入当前机器上可发现的 `vcomp140.dll` 所在目录。

然后运行：

```powershell
python integrated_price_forecast.py `
  --forecast-date 2026-07-01 `
  --backtest-start 2026-06-15 `
  --backtest-end 2026-06-30 `
  --calibration-days 14 `
  --output-dir outputs/integrated_price_forecast_20260824
```

固定使用 XGBoost（用于模型对比或上线复现）：

```powershell
python integrated_price_forecast.py `
  --backend xgboost `
  --forecast-date 2026-07-01 `
  --backtest-start 2026-06-15 `
  --backtest-end 2026-06-30 `
  --output-dir outputs/integrated_price_forecast_xgboost_20260824
```

`--backend` 可选 `auto`、`lightgbm`、`xgboost`、`fallback`。请求已安装但无法加载的后端时，脚本会直接报错，不会静默把结果冒充成目标模型。

## 融合模型搜索

融合搜索入口为 `ensemble_price_forecast.py`。它在相同的滚动回测协议下比较 LightGBM、XGBoost、Ridge、等权融合和前 7 天校准的自适应凸组合：

```powershell
python ensemble_price_forecast.py `
  --backtest-start 2026-06-15 `
  --backtest-end 2026-06-30 `
  --calibration-days 7 `
  --output-dir outputs/ensemble_search_20260826
```

根据当前 16 天回测，选择性融合方案为：日前 XGBoost、实时直接预测等权融合、价差 Ridge；正式账务一致实时价格使用“日前 XGBoost + 价差 Ridge”。预测入口为 `selected_fusion_forecast.py`，输出在 `outputs/selected_fusion_forecast_20260826`。完整比较见 `融合模型优化报告_20260826.md`。

## Transformer 扩展

`transformer_price_forecast.py` 实现了 168 小时编码、24 小时解码的多变量 Transformer。结构选择在正式回测开始前完成，随后按日滚动回测。当前统计上最可靠的用法是把 Transformer 作为价差模型的补充：

```text
日前价格 = XGBoost
实时价差 = 79% Ridge + 21% Transformer
实时价格 = 日前价格 + 实时价差
```

最终预测在 `outputs/transformer_enhanced_forecast_20260827`，详细结果和区块 Bootstrap 检验见 `Transformer模型试验与融合结论_20260827.md`。

## Mamba 扩展

`mamba_price_forecast.py` 使用 MambaPy 的纯 PyTorch 选择性状态空间模型，采用与 Transformer 相同的调参段和滚动回测口径。当前结果表明 Mamba 不适合日前主模型，但 `69.7% Ridge + 30.3% Mamba` 的价差 MAE 为 80.70 元/MWh，是当前价差点预测的最佳校准方案。Mamba 增强预测位于 `outputs/mamba_enhanced_forecast_20260827`，完整结论见 `Mamba模型试验与融合结论_20260827.md`。

如果目标日没有真实价格，脚本仍会使用目标日前的历史价格构造滞后特征，并使用目标日天气/电源出力输入生成24个时段预测。若目标日没有电源出力文件，脚本会显式使用前一周同一时段作为代理，并在覆盖信息中记录。

## 输出文件

- `forecast.json`：完整接口结果，包括日前价格、实时价格、价差、方向概率和风险标识；
- `forecast.csv`：24时段的扁平表；
- `walk_forward_backtest.csv`：严格按日期滚动的回测明细；
- `model_card.json`：真实使用的主模型、残差模型、区间方法和数据假设；
- `run_summary.json`：本次运行摘要；
- `model_bundle.pkl`：模型协议和卡片元数据。

## 统计口径

回测每个目标日只使用该日前的数据训练，随后预测目标日；TCN只学习此前滚动预测留下的残差。Conformal区间使用历史校准残差的有限样本分位点，不把测试日真实价格用于预测。风险标识仅用于辅助判断，不自动生成交易动作。

每次运行都会在 `model_card.json` 和 `run_summary.json` 中记录真实后端。当前环境已验证：LightGBM 4.6.0、XGBoost 3.4.1、PyTorch 2.13.0+cpu 均可加载；完整回测使用了真实 LightGBM 和 TCN。若部署到其他 Windows 机器，应先安装对应的 MSVC 运行库，并用模型卡片核对后端状态。
