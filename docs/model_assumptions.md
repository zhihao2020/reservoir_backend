# 模型假设与适用边界

## V1 产品范围

V1 assumes saturation observations are already provided by upstream sensing/inversion systems. Raw electrical, electromagnetic and acoustic inversion is outside the reservoir-core scope.

The lab backend ingests \(Q_{inj}(t)\), \(P_{prod}(t)\), \(P_{obs}\), \(S_{obs}(\sigma,x,y,z,t)\) and reconstructs \(p\), \(S_w,S_o,S_g\), \(z_i\), while estimating \(\theta=(\log C_f,\log\beta_{mf})\) with \(T_{mf}=\beta_{mf}T_{mf}^{ref}\). \(k_m\) and shape factor stay in \(T_{mf}^{ref}\).

M1 used self-consistent synthetic truth to prove the inverse machinery. **M2 product acceptance is a CMG-GEM cross-simulator field reconstruction** (`examples/lab_v1/cmg_gem/`). Inversion must not see the CMG 3-D field. Default invert is LM (`algorithm: auto`); ES-MDA runs only if identifiability or hold-out is weak, or `uq: true`. Parameter EnKF / UDP wait for M3.

V1 明确不做：Archie / EM / acoustic inversion、PINN、SRV、DFM/EDFM、AMR、thermal、zonal \(C_f\)、逐格 \(K\)、裂缝半长反演。产品 Case 是 `examples/lab_v1/`。

## 当前相位

- 正演 \(F\) 默认是组分 DPDP 全隐式（`physics.model: compositional_dpdp`）：两套 TPFA + 组分 transfer，主变量 \((n_f,p_f,n_m,p_m)\)。单孔组分 `compositional` 可选。黑油已删除。
- 反演 \(\theta=(\log C_f,\log T_{mf})\)。\(C_f=k_f b_f\) 管裂缝沿程传输，\(T_{mf}=\beta_{mf} T_{mf}^{ref}\) 管基质补给。\(k_m,\varphi_m,\varphi_f\) 和 PVT 固定，不进 \(\theta\)。产品 invert 默认 ES-MDA。
- 工作流：用 \([0, T_{\mathrm{hist}}]\) 观测反演，冻结 \(\theta\)，用整段井控做一次组分正演。
- 配置合同：`case.yaml` + `pvt.yaml` + `wells.yaml` + `controls.csv` + `observations.csv`。
- 黑油已删除。可选单孔组分：`physics.model: compositional`，入口 `examples/compositional/comp_example.yaml`。公开 PR 牌：`physics.fluid.file`；缺文件或缺临界量拒绝。

## 开源改编（FIM）

已获许可可改编 OPM/GEOS 算法进 `reservoir_backend/solver/fi_comp.py` / `fi_comp_dual.py`；**禁止同名**。对照表：`docs/fim_name_map.md`。`references/` 只读，产品不 `import` 上游。

## 可压缩性 / PVT

- 产品流体是组分 EOS 牌：`physics.fluid.file`（如 `pvt.yaml`）。Tc/Pc/ω/Mw/kij 从文件读；缺文件或缺临界量拒绝。
- 黏度、相对渗透率写在同一份 PVT sidecar，**不进** \(\theta\)。
- \(\theta=(\log C_f,\log T_{mf})\)。\(k_m,\varphi_m,\varphi_f\) 在 `rock` / `physics` 里固定。

## 毛管

- 产品 DPDP case 显式写 `capillary: none`
- YAML 必须写 `capillary: brooks_corey | van_genuchten | none`，禁止静默省略

## 重力

- 实验室默认关（\(\Delta\rho g H\sim 0.1\,\mathrm{psi}\)）
- CMG 矿场尺子开 \(g=9.81\)。相势 \(\Phi_\alpha=p_\alpha+\rho_\alpha g z\)（\(z\) 向上），迎风取各相自己的势
- 显式输运用相通量 \(v_\alpha=T\lambda_\alpha\Delta\Phi_\alpha\)，不是 \(f_w v_T\)。静水总速度接近 0 时仍会重力分异
- 默认初值是均匀 \(p_{\mathrm{init}}\)（对齐 IMEX `*PRES *CON`）。`hydrostatic_init: true` 才做静水修正。重力密度用格子 \(\rho_\alpha=\rho_{\alpha,\mathrm{sc}}\,b_\alpha(p)\)，面上算术平均

## 裂缝

- P0 不用 DFM/EDFM
- 已知厚构造用区域 K 表示

## 相对渗透率

- 实验室 YAML：Corey
- CMG 虚拟实验：牌组 `*SWT` 表（`TableTwoPhase.cmg_seawater`），和 PVT 一样是已知流体，不进 \(\theta\)
- 三相：`TableThreePhase.cmg_seawater`（SWT+SLT，默认 Stone II）。重力迎风含水/油/气三相势

## 离散

- Cartesian、对角渗透率、K-orthogonal 上的 TPFA
- **不保证**斜网格或旋转全张量上的结果
- 无 MPFA

## 网格

- 推荐 baseline：300 mm / 10 mm / \(30^3\)
- Variable Cartesian layering: `grid.dx`/`dy`/`dz` (aliases `DX`/`DY`/`DZ`) scalar or 1-D list along that axis; `geometry.size_m` must equal the axis sums. Optional `grid.file` relative to the case: YAML/JSON sidecar, or a CMG/Eclipse `*GRID` snippet `.grdecl` / `.dat`. Keywords (with or without `*`): CART/CARTESIAN, CORNER/CORNER-POINT, GRID, SPECGRID/DIMENS, NX/NY/NZ, DX/DY/DZ (aliases DI/DJ/DK; scalar, `*CON`, or n-vector along that axis), COORD, ZCORN (Eclipse order: COORD (nx+1)*(ny+1)*6, ZCORN 8*nx*ny*nz), ACTNUM (0 = inactive). DX/DY/DZ and no COORD/ZCORN builds CartesianGrid; COORD+ZCORN builds CornerPointGrid. Inactive cells get volume 0 and T=0 (same hook as zero-volume CPG cells). No NNC, PINCHOUT, faults, `*PVTO`, or wells.
- 场主表示 `(n_cells,)`
- 传感器坐标不必落在节点上
- 探头直径 6 mm；H 在插值场上做球平均
- 三维 p/S 是 F(m_post) 重建。产品尺子是自洽反演（贴回本正演），不是场 Dice 对 CMG

## 反演假设

- 控制量与观测分离：一口端口同一时刻不能既定流量又定压并两边都当数据
- 实验室过渡路径：2-region log K，不是粗网格 / 逐格 K。V1 目标是 \(\theta=(\log C_f,\log\beta_{mf})\)。`make_lab_v1_face_twin` 是 M1a 诊断夹具（0.30×0.20×0.10），不是 30 cm 产品的网格粗化；M1b 是 `examples/lab_v1/case_dev.yaml`
- 层状用 `region_axis: z` 和 `n_regions`。给了 `region_map` 就用图。标量 \(C_f\) 用 `parameterization: log_conductivity`
- 已知高渗体（层、通道）用对比度 \(\theta=(\log k_{\mathrm{lo}},\log(k_{\mathrm{hi}}/k_{\mathrm{lo}}))\)，\(k_{\mathrm{hi}}\ge k_{\mathrm{lo}}\)。符号是构造，数值才反演
- 正演默认隐式输运（两相和三相）。YAML `physics.transport: explicit` 可关。不是为了把场贴成 IMEX
- LM 在 θ 空间更新，输出点估计 \(\hat\theta\) 和 Hessian 对角 \(\sigma_\theta\)。不是逐格真值图
- 产品 invert 支持 hold-out 测点与 history/forecast 时间切开
- 测点是任意 \((x,y,z)\)。同一柱面上不同深度用 `column_sensors`。种类/深度越多，层状 K 越好认；单平面几个压力点不够
- 井指数和相对渗透率差不要靠拧 K 去吸收（那是调参）。跨模拟器时流量观测要谨慎，优先用内部 \(p,S_w\)
参考实现只放在 `references/methods/`（Equinor IES、pyesmda、dass、Emerick 2013），产品代码不 import。

```text
python scripts/lab_v1_generate_truth.py --dev --case B
python scripts/lab_v1_offline.py --dev
reservoir invert examples/lab/lab_cf.yaml --self-check --output results/cf
```

## 井 / 端口

- 实验室入口出口是 `FlowPort`
- Optional `wells.file` (relative to the case YAML) loads a CMG/IMEX `*WELL` / `*INJECTOR` / `*PRODUCER` / `*PERF` / `*OPERATE` / `*GEOMETRY` snippet onto the same `FlowPort` objects YAML `ports:` already builds. I/J/K are 1-based. BHP maps to `control=pressure`, STW/STO to `control=rate`. `*GEOMETRY` rw/geofac/skin map to `rw_m`/`geofac`/`skin` and set `use_productivity`. YAML `ports:` is unchanged. No group wells, VFP, workovers, time-varying history, `*WELLHYD`, or multilateral.
- 定压注入：出流面带走井筒组成（`composition` / `sw_inj`），不是井格 \(f_w(S_{wi})\approx 0\)
- 定压采出：按井格分流把净流入抽走，避免井格攒到 \(S_w=1\) 后时间步崩溃
- 实验室默认格子 Dirichlet / 半格 WI（半格也乘总流度 \(\lambda_t\)）。CMG 虚拟实验复制牌组 `*GEOMETRY` 的 Peaceman（\(r_w\)、geofac），\(q=\mathrm{WI}\,\lambda_t\,(p_{\mathrm{conn}}-p)\)。`*K` 井底压钉在最上射孔，往下加井筒水头 \(\rho_{\mathrm{wb}} g\Delta z\)。不拧 `wi_multiplier` 去贴 IMEX 流量
- 和 IMEX 比的是同一套射孔层位和 \(u(t)\)，不是同一套井指数公式
- CMG 虚拟实验 \(\varphi=0.30\)、海水 \(S_w^{\mathrm{inj}}=1\)、PVT 取牌组 `*BWI/*CW/*CO/*CPOR`。\(F(K_{\mathrm{CMG}})\) 场尺子均值压已落到几 psi（五点/断层 242 d）；剩下的是空间形态和相对渗透率表，不是再拧 \(c_t\)
- 跨模拟器时 \(R\) 加上 \(F(K_{\mathrm{CMG}})\) 对 CMG 测点的残差，避免把模型差拧进 \(K\)。协议 A 不吃井流量（定压流量对 \(K\) 太陡）

## 概念实验室 30 cm 水驱：反演对比

产品 invert 对比是 \(F(m_{\mathrm{post}})\) 对 \(F(m_{\mathrm{true}})\) 的饱和度场与压力场 nRMSE。**不是** CMG 格子场。尺度放大不进 inversion core。

测点坐标由 case `sensors.csv` 给出。声波 / 电磁原始反演不在本仓库。

页岩 IMEX 缝长/SRV/\(k_m\) 反演与济阳矿场吞吐已移出产品。

## 和开源仿真 / CMG / 生产的关系

开源油藏程序是正演 \(F\)。本仓库是数字孪生：\(F+H+\) LM。

要和 CMG（或以后的矿场）「工况一样」：

- 复制 **控制** \(u(t)\) 和 **测点坐标**，不复制 IMEX 黑油方程
- 井控设计必须让窗口里还有 \(\Delta p/\Delta S_w\)（当前尺子：INJ 3200 psi / PROD 2800 psi）
- 算法关：\(d=H(F_{\mathrm{lab}}(m_{\mathrm{true}}))\)，必须收回层状 K
- 稳健关：\(d\) 来自 CMG 或实验，通过标准是观测/hold-out/预报，不是 \(K=K_{\mathrm{CMG}}\)

真实生产走同一条链。\(F\) 已是黑油油水（实验室取 \(B=1\)）。后验 \(K\) 是该 \(F\) 下的等效渗透率。和 CMG 比的是同一套 \(u(t)\) 与测点，不是 \(K=K_{\mathrm{CMG}}\)，也不是把 Peaceman/井指数拧到和 IMEX 一样。

## 和 IMEX 还差什么（尺子上已经不大）

同岩石 1 天：油水约 1.6 psi / \(S_w\) 0.03，含气约 2 psi / \(S_g\) 0.009。相对 400 psi 压差，不是「差一个数量级」。

还能补、且补了会进方程的：

- 采出井筒密度跟 IMEX 管柱混合物还不完全一样（已有 `crossFlowMixtureDensity`，多段摩阻仍没有）
- 脱气均 \(S_g\) 已贴近（0.019 vs 0.016）；压力/含水形态还有顺序对全隐式的差

不会、也不该补成「变成 IMEX」的：

- VARI、断层落差、MPFA、斜网格
- 全隐式组分闪蒸、相对渗透率滞后、井筒摩阻多段井
- 用逐格 \(K\) 去吸收上述模型差
