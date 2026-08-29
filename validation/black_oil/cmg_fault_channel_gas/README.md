# 含气尺子：断层 + 通道 + 游离气

CMG 是虚拟实验。初始 \(S_w=0.20\)、\(S_o=0.70\)、\(S_g=0.10\)（IMEX 用 \(S_g=1-S_o-S_w\)，没有 `*SG` 关键字）。井底 3200 / 2800 psi，高于泡点 2500 psi，**不额外脱气**，气是放进去的游离气。

几何、渗透率、断层与 `cmg_fault_channel` 相同。正演 \(F\) 开三相（Corey，对应牌组 `*SLT`），测点含压力、含水、含气。图只做真值对反演。

## 运行

```bash
python validation/black_oil/cmg_fault_channel/build_fault_channel.py
python validation/black_oil/cmg_fault_channel_gas/build_fault_channel_gas.py
cd validation/black_oil/cmg_fault_channel_gas
D:\Tool\CMG\IMEX\2024.20\Win_x64\EXE\mx202420.exe -f fault_channel_gas.dat
python validation/black_oil/cmg_fault_channel_gas/run_invert_eval.py
python validation/black_oil/cmg_fault_channel_gas/plot_cmg_zh.py
```

## 本次实测

IMEX 正常结束。全场初始 \(S_g\approx 0.10\)，1 天均值约 0.05（气被采出/驱走）。

同岩石 \(F\)（相势 + 井筒水头 + 活油）对 IMEX：1 天压力 **2.04 psi**，含水 **0.025**，含油 0.023，含气 **0.009**。

反演用压力 + 含水 + 含气：

| | 基质 / md | 通道 / md | 对比度 | log K RMSE |
|--|-----------|-----------|--------|------------|
| **真值** | 50 | 2000 | 40 | — |
| Corey 三相 | 81 | 2564 | 31.5 | 0.46 |
| 表 × 乘积 | 55 | 1258 | 22.9 | 0.19 |
| Stone II | 58 | 1468 | 25.3 | 0.18 |
| Stone II + \(R\) 封顶 | 57 | 1617 | 28.2 | 0.15 |
| **+ 气相重力迎风（现在）** | **47** | **1899** | **40.1** | **0.055** |

通道收到 **1899**，对比度 **40.1**（真 40）。缺的是压力方程里气相势迎风，不是再拧 ES-MDA。

图：`figures/zh_真值对反演_*.png`。

## 说明

F 的三相已带活油 \(R_s\)（地面气 \(b_g S_g+R_s b_o S_o\)，表黏度，Peaceman 用 \(\lambda_t\)）。本尺子井底 2800 > 泡点 2500，闪蒸不动作。验的是：**场里已经有气时，反演能不能看见含气、能不能收回通道对比度。** 要验放气，把井底降到泡点以下。
