# CMG 两层尺子：同一工况，实验室反演

开源油藏程序（OPM、MRST…）是正演 \(F\)。这里要测的是数字孪生：

\[
x=F_{\mathrm{lab}}(m,u),\quad d=H(x),\quad d_{\mathrm{obs}}\to m_{\mathrm{post}}
\]

和 IMEX「工况一样」= **同一套井控 \(u(t)\)** + **同一批 \((x,y,z)\) 测点**。  
不是把 \(F\) 改成黑油再去收回格子 \(K_{\mathrm{CMG}}\)（那是调参）。

## 模型（从通道样例克隆后改）

| 项 | 原通道样例 | 本尺子 |
|----|------------|--------|
| 网格 | 7×7×5 VARI | **12×8×6 CART** |
| 渗透率 | 山脊 50/2000 md | **顶 500 md / 底 50 md** |
| 井控 | 大油田 STW/STO | **INJ 3200 psi / PROD 2800 psi**（定压对，流量当结果） |
| 时间 | 年 | 0.25–8 day |

定压对是为了让窗口里还有 \(\Delta p\)（旧案 PROD=1500 psi 会把全场压到一口井上，\(p\) 标准差只剩约 11 psi）。

## 测点

真实井/探针深度各不相同。尺子用 `column_sensors`：同一 \((x,y)\) 柱面，顶层和底层都有 \(p\) 和 \(S_w\)。  
评测里对比 **稀疏中面** vs **多深度+多类型**。种类和深度越多，层状 K 越好认。

跨模拟器（协议 B）**不**把井流量当观测：Peaceman 和实验室 `FlowPort` 的差会被拧进 K。

## 协议

| | 数据 | 通过标准 |
|--|------|----------|
| **A 自洽** | \(d=H(F_{\mathrm{lab}}(m_{\mathrm{true}}))\) | 收回层对比度，misfit/hold-out 变好 |
| **B 跨模拟器** | \(d\) 来自 CMG 内部测点 | 观测/hold-out/预报变好。后验 K 是等效实验室 K，不是 \(K_{\mathrm{CMG}}\) |

## 运行

```bash
python black_oil/validation/cmg_lab_layers/build_lab_layers.py
cd black_oil/validation/cmg_lab_layers
D:\Tool\CMG\IMEX\2024.20\Win_x64\EXE\mx202420.exe -f lab_layers.dat
python black_oil/validation/cmg_lab_layers/run_invert_eval.py
```

报告：`invert_eval_report.json`。

多案、探针剪枝、旋钮回溯见 `reservoir harness suite`（`black_oil/validation/cmg_harness/README.md`）。

场图（CMG 仿真 vs 反演正演）：

- `figures/cmg_vs_inv_fields_xz.png` — 中间 y 切片 \(p,S_w\)，0.25/0.5/1 天
- `figures/cmg_vs_inv_k_field.png` — \(K\) 场
- `figures/cmg_vs_inv_sw_xy.png` — 顶/底平面 \(S_w\)（1 天）

```bash
python black_oil/validation/cmg_lab_layers/plot_cmg_inv_fields.py
```

## 本次实测（IMEX 2024.20，定压对 3200/2800 psi）

全场压力标准差一直约 64–70 psi。同网格、同井控、多深度测点：

时序：A 用 0.125/0.25/0.375/0.50 天；B 同化 **0.25/0.50/1.0 天**，预报 **2 天**。

| | 层对比度（真 10） | 后验 md | log K RMSE | hold-out | 预报 |
|--|-------------------|---------|------------|----------|------|
| **A 自洽** | **9.56** | 54 / 520 | **0.065** | 1.59 | 1.12 |
| **B CMG 测点 + \(R\) 膨胀** | 3.10（等效，不是 10） | 42 / 130 | 不收 \(K_{\mathrm{CMG}}\) | **0.62** | **0.48** |

对策：先算 \(F(K_{\mathrm{CMG}})\) 对 CMG 测点的残差（压力约 180 psi，\(S_w\) 约 0.3），写进 \(R\)。集成就不再把模型差拧进层状 \(K\)。B 的 hold-out 从 4.3 降到 **0.74**（在模型误差尺度上拟合），对比度不再反转。

不要为了对齐 \(K_{\mathrm{CMG}}\) 去改 \(F\)。

## 用 CMG 测点反演后，和 IMEX 全场差多少

`run_cmg_gap.py` → `figures/cmg_gap_fields.png`、`cmg_gap_report.json`。

中间列是 **同一套 50/500 md** 用我们的 \(F\) 正演，右边是反演后的 \(F(k_{\mathrm{post}})\)。

| t (d) | \(p\) RMSE 同岩石 | \(p\) RMSE 反演后 | \(S_w\) RMSE 同岩石 | \(S_w\) RMSE 反演后 | \(S_w\) 均值 CMG / \(F(K)\) |
|-------|-------------------|-------------------|---------------------|---------------------|------------------------------|
| 0.25 | **56 psi** | 31 psi | 0.11 | 0.13 | 0.40 / 0.46 |
| 0.50 | **30 psi** | 28 psi | **0.08** | 0.11 | 0.49 / 0.51 |
| 1.00 | **17 psi** | 20 psi | **0.09** | **0.08** | 0.55 / 0.58 |

上一版压力差 150 psi，是因为我们只开了中间 1–2 格，IMEX 是 INJ K=3–6、PROD 全柱。工况没对齐，不是反演不会做。对齐射孔后，同岩石压力 1 天 RMSE **17 psi**，含水均值差约 0.03。  
前缘仍比 IMEX 陡（不可压 Corey、无重力）。B 后验 42/130 md 是等效实验室 \(K\)，不是 50/500。不要为了格子 \(K_{\mathrm{CMG}}\) 去改 \(F\)。

## 同一模型、三套网格

盒子、层、井控、测点坐标不变，只改 \(\Delta x\)。脚本：`run_grid_compare.py`。图：`figures/grid_compare_fields.png`、`figures/grid_compare_metrics.png`。

| 网格 | 格子数 | 对比度（真 10） | log K RMSE | hold-out |
|------|--------|-----------------|------------|----------|
| 8×6×4 | 192 | 11.09 | 0.182 | 1.38 |
| 12×8×6 | 576 | 11.08 | 0.191 | 1.36 |
| 16×10×8 | 1280 | 11.08 | 0.201 | 1.34 |

层都能收回，加密网格几乎不改反演结论（参数化是 2 个区域，不是每格一个 K）。
