# 模型假设与适用边界

## V1 产品范围

V1 assumes saturation observations are already provided by upstream sensing/inversion systems. Raw electrical, electromagnetic and acoustic inversion is outside the reservoir-core scope.

The lab backend ingests \(Q_{inj}(t)\), \(P_{prod}(t)\), \(P_{obs}\), \(S_{obs}(\sigma,x,y,z,t)\) and reconstructs \(p\), \(S_w,S_o,S_g\), \(z_i\), while estimating \(\theta=(\log C_f,\log\beta_{mf})\) with \(T_{mf}=\beta_{mf}T_{mf}^{ref}\). \(k_m\) and shape factor stay in \(T_{mf}^{ref}\).

M1 used self-consistent synthetic truth to prove the inverse machinery. **M2 product acceptance is a CMG-GEM cross-simulator field reconstruction** (`examples/lab_v1/cmg_gem/`). Inversion must not see the CMG 3-D field. Parameter EnKF / UDP wait for M3.

V1 明确不做：Archie / EM / acoustic inversion、PINN、SRV、DFM/EDFM、AMR、thermal、zonal \(C_f\)、逐格 \(K\)、裂缝半长反演。产品 Case 是 `examples/lab_v1/`，不是 `lab_apply.yaml`。

## 当前相位

- 正演 \(F\) 默认仍是顺序黑油：TPFA 压力 + 后向 Euler 隐式饱和度。守恒仍是 \(\partial_t(\varphi b_\alpha S_\alpha)+\nabla\cdot(b_\alpha v_\alpha)=q_\alpha^s\)
- 实验室物理实验默认 \(B=1,c=0\)（同一套方程的不可压特例）
- CMG 虚拟实验用牌组同款 PVT（`BlackOilPVT.cmg_seawater`：`*BWI/*CW/*CO/*CPOR`，泡点下未饱和 \(B_o\)）
- 三相：`*SWT`+`*SLT` 表 + Stone II。默认是顺序黑油：冻 \(v_T\)，耦合隐式 \((S_w,S_g)\)，守恒 **油 + 地面气**；油表面通量用面上迎风 \(b_o\)。输运 extras 默认 **势迎风**（Brenier–Jaffré 含 \(v_T\)，顺序黑油默认）；`upwind_type=hybrid` 才把粘性和重力拆开。牛顿过残余饱和度截断。活油在闪蒸后和步末更新压力时按增量容差迭代。步末 P→T→P。`fully_implicit` 才走耦合牛顿，活油主变量是 \((p,S_w,x)\)：无游离气时 \(x=R_s\)，油气两相时 \(x=S_g\)（一次切换 `switch_live_oil_unknown` + 步末可选 `liberate_excess_gas`，见 `docs/fim_name_map.md`）。失败砍步，不退回顺序。放气尺子闸门（约 ≤6.5 psi 且均 \(S_g\) 贴近顺序）未过前，产品默认仍关 FIM。格子级 AIM 还没接。
- 活油：`BlackOilPVT.cmg_seawater` 带牌组 \(R_s,B_o,E_g,\mu_o,\mu_g\) 表。地面气 \(G^s=\varphi(b_g S_g+R_s b_o S_o)\)，通量带油相溶解气。\(p\ge p_b\) 时 \(R_s\) 封顶、\(B_o\) 用 `*CO`、\(dR_s/dp=0\)；\(p<p_b\) 时饱和插值，压力存储加 \(S_o(b_o/b_g)\,dR_s/dp\)。压力步后先闪蒸再算相通量，输运后再按总气量闪蒸。不是全隐式组分闪蒸。
- \(\theta\) 只有岩石（log \(K\)）。PVT 是实验已知流体，不反演。Case 入口：`physics.pvt` → `io.pvt_cfg.pvt_from_cfg`；相对渗透率 \(\mu\) 从同一份盖章
- 组分 EXAMPLE（可选 \(F\)，`physics.model: compositional`）：等温两相气–油，Peng–Robinson + PT 闪蒸，主变量 \((n_i,p)\)。流体是公开 C1–nC10（Reid/Prausnitz/Poling 临界参数，Katz–Firoozabadi \(k_{ij}\)），常数 \(\mu\)。无水、无热、不编造济阳 Tc/Pc。饱和度由闪蒸摩尔体积得到，不走黑油 \(S_g\leftrightarrow R_s\)。接线在 `solver/fi_comp.py`，不改 `solver/fi.py`。入口 `examples/compositional/comp_example.yaml`。定流量井的井底压 \(p_{\mathrm{wf}}\) 是观测（控制是率）。有 \(H=p_{\mathrm{wf}}\) 后高渗带 \(K\) 可收回；低渗带对比度有阻尼。无井底压时并联两带的压力场几乎看不见绝对 \(K\)。Jacobian 的 coloring FD 不算观测用 \(p_{\mathrm{wf}}\)；报告时刻取最近接受步。Immiscible 水相可选：`physics.has_water: true`，未知量多 \(n_w\)，水不进 PR 闪蒸；EXAMPLE 孪生可带 \(S_w\) 观测做 2-region LM（`tests/cases/test_comp_water.py`）。公开 PR 牌：`physics.fluid.file`（YAML 或 Eclipse `TCRIT`/`PCRIT`/`ACF`/`MW`/`BIC`），缺文件或缺临界量则拒绝，不编造济阳 Tc/Pc。

## 开源改编（FIM）

已获许可可改编 OPM/GEOS 算法进 `reservoir_backend/solver/fi.py`；**禁止同名**。对照表：`docs/fim_name_map.md`。`references/` 只读，产品不 `import` 上游。

## 可压缩性 / PVT

- `physics.pvt: incompressible | slightly_compressible | cmg_seawater`（别名 `cmg` / `black_oil`）。也可写成 mapping：`pvt: {preset: cmg_seawater, mu_w: ...}`——标量 \(\mu\) 仅覆盖死油/不可压常数场；活油表 \(\mu(p)\) 仍优先。用户可在同一 mapping 里给 SI 表 `p`/`p_tab`、`rs`/`rs_tab`、`bo`/`bo_tab`、`eg`/`eg_tab`（或 `bg`/`bg_tab`，写入 \(E_g=1/B_g\)）、`muo`/`muo_tab`、`mug`/`mug_tab`；有表的列覆盖 preset。可选 `file` / `pvto` 指向同格式 YAML/JSON sidecar（相对 case YAML 目录），或 CMG/IMEX `*PVTO`/`*PVTW`/`*PVDG`/`*PVT` 文本（默认 field：psi、scf/stb、cP，与 `cmg_seawater` 相同，换算进 SI；`*INUNIT *SI` 为 kPa + sm3/sm3 + cP）。水默认仍是线性 \(b_w\)；给了 `p_w`/`bw` 才按表插值 \(B_w\)。YAML 里 `compressibility: <ct>` 仍可生成均匀岩石 \(c_r\)（无 `pvt` 时等价 `slightly_compressible`）
- 工厂：`io.pvt_cfg.pvt_from_cfg`。相对渗透率 \(\mu\) 从同一份 PVT 盖章。\(\theta\) 不含黏度
- \(b_W=(1+c_w(p-p_{\mathrm{ref}}))/B_{W,\mathrm{ref}}\)，油相同；\(\varphi(p)=\varphi_{\mathrm{ref}}(1+c_r(p-p_{\mathrm{ref}}))\)
- 定流量是地面流量。压力方程右端是 \(q^s/b\)。不可压全流量系统才钉压力基准；有存储项时不钉（否则均值压升被抽掉）
- 活油守恒 + 表黏度已进三相 \(F\)。`*PVT` 的 \(E_g\)（scf/RB）和 \(R_s\) 一样乘 \(0.178\) 进 SI，否则气密度会被算成原油量级、放气偏少。气相压缩用 \(E_g\) 表导数。压力步闪蒸后用新 \(\lambda,c_t\) 再解一次压力。

## 毛管

- 实验室 30 cm case 默认 Brooks–Corey
- 必须在配置里写 `capillary: brooks_corey | van_genuchten | none`
- 禁止静默 Pc=0 还声称实验室物理完整
- 毛管进水相势：\(\Phi_w=p-P_{cow}(S_w)+\rho_w g z\)。压力方程和相通量都用它。隐式输运冻住 \(v_T\)，并加上旧步分异 \(v_w-f_w v_T\)

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

测点坐标来自 `examples/lab/concept_probes.csv`（从 `测点.xlsx` 抄入 SI 米）：电阻率 75（底面/界面/顶面）+ 新增 7.5 cm 共 16。声波 12 / 电磁 8 在 `测点位置.pptx` 只有个数与 75 mm 间距，没有 xyz 表，不编造。

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
