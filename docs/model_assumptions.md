# 模型假设与适用边界

## 当前相位

- 正演 \(F\) 默认仍是顺序黑油：TPFA 压力 + 后向 Euler 隐式饱和度。守恒仍是 \(\partial_t(\varphi b_\alpha S_\alpha)+\nabla\cdot(b_\alpha v_\alpha)=q_\alpha^s\)
- 实验室物理实验默认 \(B=1,c=0\)（同一套方程的不可压特例）
- CMG 虚拟实验用牌组同款 PVT（`BlackOilPVT.cmg_seawater`：`*BWI/*CW/*CO/*CPOR`，泡点下未饱和 \(B_o\)）
- 三相：`*SWT`+`*SLT` 表 + Stone II。默认是顺序黑油：冻 \(v_T\)，耦合隐式 \((S_w,S_g)\)，守恒 **油 + 地面气**；油表面通量用面上迎风 \(b_o\)。输运 extras 默认 **势迎风**（Brenier–Jaffré 含 \(v_T\)，顺序黑油默认）；`upwind_type=hybrid` 才把粘性和重力拆开。牛顿过残余饱和度截断。活油在闪蒸后和步末更新压力时按增量容差迭代。步末 P→T→P。`fully_implicit` 才走耦合牛顿，活油主变量是 \((p,S_w,x)\)：无游离气时 \(x=R_s\)，油气两相时 \(x=S_g\)（`switch_live_oil_unknown` / `liberate_excess_gas`，见 `docs/fim_name_map.md`）。失败砍步，不退回顺序。放气尺子闸门（约 ≤6.5 psi 且均 \(S_g\) 贴近顺序）未过前，产品默认仍关 FIM。格子级 AIM 还没接。
- 活油：`BlackOilPVT.cmg_seawater` 带牌组 \(R_s,B_o,E_g,\mu_o,\mu_g\) 表。地面气 \(G^s=\varphi(b_g S_g+R_s b_o S_o)\)，通量带油相溶解气。\(p\ge p_b\) 时 \(R_s\) 封顶、\(B_o\) 用 `*CO`、\(dR_s/dp=0\)；\(p<p_b\) 时饱和插值，压力存储加 \(S_o(b_o/b_g)\,dR_s/dp\)。压力步后先闪蒸再算相通量，输运后再按总气量闪蒸。不是全隐式组分闪蒸。
- \(\theta\) 只有岩石（log \(K\)）。PVT 是实验已知流体，不反演

## 开源改编（FIM）

已获许可可改编 OPM/GEOS 算法进 `reservoir_backend/solver/fi.py`；**禁止同名**。对照表：`docs/fim_name_map.md`。`references/` 只读，产品不 `import` 上游。

## 可压缩性 / PVT

- `physics.pvt: incompressible | cmg_seawater`；YAML 里 `compressibility: <ct>` 仍可生成均匀岩石 \(c_r\)
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
- 场主表示 `(n_cells,)`
- 传感器坐标不必落在节点上
- 探头直径 6 mm；H 在插值场上做球平均
- 三维 p/S 是 F(m_post) 重建。产品尺子是自洽反演（贴回本正演），不是场 Dice 对 CMG

## 反演假设

- 控制量与观测分离：一口端口同一时刻不能既定流量又定压并两边都当数据
- 默认参数化是 2-region log K，不是粗网格 6³ / 逐格 K
- 层数不必写死为 2。`--auto` 在离散目录（均匀 / 2 层 / 3 层 / 顶高或底高 contrast）上按 hold-out 选。给了 `region_map` 就用图，不猜
- 已知高渗体（层、通道）用对比度 \(\theta=(\log k_{\mathrm{lo}},\log(k_{\mathrm{hi}}/k_{\mathrm{lo}}))\)，\(k_{\mathrm{hi}}\ge k_{\mathrm{lo}}\)。符号是构造，数值才反演。对比度先验必须能覆盖几十倍，否则 ensemble 会停在「高基质 + 弱通道」的等效模态上
- 正演默认隐式输运（两相和三相）。YAML `physics.transport: explicit` 可关。不是为了把场贴成 IMEX
- ES-MDA 在 θ 空间更新，输出均值/方差，不是唯一真值图
- 产品 invert 支持 hold-out 测点与 history/forecast 时间切开
- 测点是任意 \((x,y,z)\)。同一柱面上不同深度用 `column_sensors`。种类/深度越多，层状 K 越好认；单平面几个压力点不够
- 井指数和相对渗透率差不要靠拧 K 去吸收（那是调参）。跨模拟器时流量观测要谨慎，优先用内部 \(p,S_w\)

## Ensemble 和 HPO

有的是 **ES-MDA 家族 + 限时搜旋钮**。没有 Optuna/CMOST 那种搜格子 \(K\)。

已接线：`n_ensemble`、`n_assimilations`（\(\sum 1/\alpha_i=1\)）、log K 先验、同化后 `inflation=1.02`、失败成员回退、identifiability \(=\sigma_{\mathrm{post}}/\sigma_{\mathrm{prior}}\)。粗网格先验带一点空间光滑。

写了但默认关掉：Gaspari–Cohn 局部化（`md_localization`）。\(n_\theta=2\) 的层状尺子用不上。

参考实现只放在 `references/methods/`（Equinor IES、pyesmda、dass、Emerick 2013），产品代码不 import。

没有：外层 HPO、自适应 \(N_a\)、IES、CMOST、AutoGluon 依赖。用它们去贴 CMG 的 K 就是调参。

从 AutoGluon **只迁设计**：预设档、时限、hold-out 排行榜、赢家混合。不迁表格模型，不搜 \(K\)。

`--auto` 默认做限时随机 HPO：搜的是算法和旋钮（\(N_e,N_a,\sigma_{\mathrm{prior}}\), inflation），目标是 hold-out，不是格子 \(K\)。算法：ES、ES-MDA、几何 ES-MDA、ES-MDA-RS、IES（Chen–Oliver 阻尼迭代）。赢家可 hold-out 混合，变差丢掉。Equinor Localized ESMDA 等 \(n_\theta\) 大再开。

Ensemble 正演默认可并行（线程池，`inverse.n_workers`，`null` 为自动、最多 8）。不用进程池：正演是孪生上的闭包，Windows `spawn` 编不了。`n_workers: 1` 强制串行。

```text
reservoir invert config/lab_30cm.yaml --preset balanced
reservoir invert config/lab_30cm.yaml --auto --time-limit 120
```

## 井 / 端口

- 实验室入口出口是 `FlowPort`
- 定压注入：出流面带走井筒组成（`composition` / `sw_inj`），不是井格 \(f_w(S_{wi})\approx 0\)
- 定压采出：按井格分流把净流入抽走，避免井格攒到 \(S_w=1\) 后时间步崩溃
- 实验室默认格子 Dirichlet / 半格 WI（半格也乘总流度 \(\lambda_t\)）。CMG 虚拟实验复制牌组 `*GEOMETRY` 的 Peaceman（\(r_w\)、geofac），\(q=\mathrm{WI}\,\lambda_t\,(p_{\mathrm{conn}}-p)\)。`*K` 井底压钉在最上射孔，往下加井筒水头 \(\rho_{\mathrm{wb}} g\Delta z\)。不拧 `wi_multiplier` 去贴 IMEX 流量
- 和 IMEX 比的是同一套射孔层位和 \(u(t)\)，不是同一套井指数公式
- CMG 虚拟实验 \(\varphi=0.30\)、海水 \(S_w^{\mathrm{inj}}=1\)、PVT 取牌组 `*BWI/*CW/*CO/*CPOR`。\(F(K_{\mathrm{CMG}})\) 场尺子均值压已落到几 psi（五点/断层 242 d）；剩下的是空间形态和相对渗透率表，不是再拧 \(c_t\)
- 跨模拟器时 \(R\) 加上 \(F(K_{\mathrm{CMG}})\) 对 CMG 测点的残差，避免把模型差拧进 \(K\)。协议 A 不吃井流量（定压流量对 \(K\) 太陡）

## 和开源仿真 / CMG / 生产的关系

开源油藏程序是正演 \(F\)。本仓库是实验室数字孪生：\(F_{\mathrm{lab}}+H+\) ES-MDA。

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
