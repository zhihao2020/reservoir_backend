# 活油脱气尺子：断层 + 通道，井底低于泡点

从 `cmg_fault_channel_gas` 克隆几何和渗透率。初值 \(S_o=0.80\)、\(S_w=0.20\)、\(S_g=0\)。生产井底 **1800 psi**，泡点 **2500 psi**。气只应来自 \(R_s(p)\) 放气。

## 运行

```bash
python black_oil/validation/cmg_fault_channel_lib/build_fault_channel_lib.py
cd black_oil/validation/cmg_fault_channel_lib
D:\Tool\CMG\IMEX\2024.20\Win_x64\EXE\mx202420.exe -f fault_channel_lib.dat
python black_oil/validation/cmg_fault_channel_lib/run_lib_smoke.py
python black_oil/validation/cmg_fault_channel_lib/run_invert_eval.py
python black_oil/validation/cmg_fault_channel_lib/plot_cmg_zh.py
```

## 本次实测

IMEX 正常结束。场均压约 2514 psi（注入 3200 顶着），生产井附近最低 **1808 psi**，低于泡点。全场平均 \(S_g\) 从 0 长到 **0.016**。

同岩石 \(F(K_{\mathrm{CMG}})\)，1 天（hybrid 迎风 + 井冻 \(q_T\)）：

| | 压力 | \(S_w\) | \(S_g\) | 均 \(S_g\) | 最低压 | 最高 \(S_g\) |
|--|------|---------|---------|-----------|--------|-------------|
| IMEX | — | — | — | 0.016 | 1808 psi | 0.074 |
| **F** | **13.9 psi** | **0.055** | **0.015** | **0.019** | **1815 psi** | **0.126** |

\(E_g\) 改成和 \(R_s\) 同一套 SI 之后，气会变轻、会升、会堆。均 \(S_g\) 从偏少 0.010 收到 0.019（IMEX 0.016）。压力 RMSE 从 10 到 14，是气动起来以后的场形态差，不是没放气。

反演修正：丢掉 0.25 天前瞬态；**只同化压力 + 含气**（含水 extras≈0.04 是 F≠IMEX，不能进 \(K\)）；\(R\) 封顶改为 \(1\times\sigma\)。

| | 基质 / md | 通道 / md | 对比度 | log K RMSE | 留出 |
|--|-----------|-----------|--------|------------|------|
| **真值** | 50 | 2000 | 40 | — | — |
| 旧 F（\(E_g\) 单位错） | 41 | 1494 | 36.1 | 0.21 | 1.07 |
| **现在（\(E_g\) 已改 SI）** | **54** | **2236** | **41.4** | **0.084** | **0.77** |

对比度 41（真 40）。通道略高，和均 \(S_g\) 略多于 IMEX 一致。2 天预报：\(F(K_{\mathrm{true}})\) 与 \(F(k_{\mathrm{post}})\) 都能跑完。图：`figures/zh_真值对反演_*.png`。

## 验什么

同岩石 \(F(K_{\mathrm{CMG}})\) 对 IMEX：压力掉到泡点以下后，\(S_g\) 必须从 0 长出来，并且场 RMSE 可报。
