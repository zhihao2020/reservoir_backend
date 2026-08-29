# 审查报告 audit_2026

日期：2026-08-15  
范围：完整阅读 `reservoir_backend/`、`tests/`、`config/`、`STATUS.md`、`README.md`、`references/软件要求.txt`、`docs/check.txt`，并抽查 `validation/black_oil/`、`validation/shale_oil/`。  
原则：不以文件名、STATUS「已验证」、或旧测试通过为正确性证据。以 `docs/check.txt` 的物理问题定义为准。

---

## 一句话结论

当前仓库**不是**「实验控制 → 多相正演 → 观测算子 → ensemble 后验 → 预测」的数字孪生反演后端。

它是两套互相污染的系统叠在一起：

1. **产品主线（`pipeline/` + `references/软件要求.txt`）**：传感器四场重建。每个时刻把测点 p、S 插值/硬钉到网格，再用 Darcy 代数估 k、φ。饱和度被当成可空间插值的场。
2. **半成品正演/反演核（`solver/` + `forward_gm` + `ensemble_math`）**：Cartesian TPFA、显式两相输运、多步 ES-MDA 数学。物理上更接近规格，但**没有接到产品路径**，且 inversion wrapper 用插值、指示先验、后验混合把它冲掉了。

因此多数 `check.txt` §83 问题当前无法诚实回答。项目还没有形成实验室多相反演数字孪生。

---

## A. Current architecture

### A.1 真实结构（不是建议结构）

```
config YAML / CSV
        │
        ▼
pipeline.run / time_series          ← 产品编排
        │
        ├─ mesh_builder             ← Cartesian (nz,ny,nx)，传感器吸附到 cell
        ├─ point_workflow           ← 默认「四场」
        │     ├─ reconstruct_pressure     测点 p = cell Dirichlet
        │     ├─ reconstruct_saturation   测点 S = IDW/克里金
        │     ├─ invert_rock_properties   k ~ μ|u|/|∇p|（通量来自 k_prior）
        │     └─ IDW/克里金 → 全场 k,φ
        │
        ├─ transport_saturation     ← 可选迎风 fw；再与插值 S 混合
        │
        ├─ invert_rock / auto_inversion
        │     ├─ 用 nowcast 做指示先验
        │     ├─ 通道 θ / 裂缝 θ 门控（启发式）
        │     └─ 默认：全网格 log k 的 ES-MDA
        │
        └─ run_esmda_permeability
              ├─ G(m) = forward_gm（较干净）
              └─ 结束后再用 point-first 出 history，并 0.7/0.3 混合 k
```

`solver/` 另有一套更完整的物理碎片（毛管通量、重力、三相、CFL、饱和度求解器），**产品路径不调用它们**。`io/config_loader.py` 仍保留 Archie/EM/acoustic 旧配置入口。

### A.2 数据模型

`SensorSample` 把控制量和观测塞进同一个袋子：

- `well_pressure`
- `well_saturation`
- `well_rate`
- `boundary.pressure` / `boundary.flux`

没有 `Controls`、`Observations`、`FlowPort`、`ObservationOperator`。  
井和测点通过 `well_cell_id` 吸附到网格单元。场数组一律 `(nz, ny, nx)`。

### A.3 两套需求互相打架

| 来源 | 定义的「反演」 |
|------|----------------|
| `references/软件要求.txt` | 每个时刻：插值 p → 插值 S → 由 p、S、流量代数得到每个格子的 k、φ |
| `docs/check.txt` | 静态参数 \(m=\{K,\phi,\ldots\}\)，动态状态 \(x=\{p,S\}\)，\(x=F(m,u)\)，\(d=H(x,m)\)，\(d^{obs}\to m^{posterior}\) |

当前代码、测试、STATUS、README 跟随**软件要求**。`check.txt` 明确：与新物理定义冲突时以新定义为准。

### A.4 默认算例尺度

`config/sensor_case.yaml`、`config/sensor_series_case.yaml` 是 **100 × 80 × 30 m**、`dx=10 m` 的矿场盒子，不是 300 mm × 10 mm / 30³。实验室 30 cm 只存在于 `lab_horizon` 与 `validation/black_oil/lab_box_30cm/`，测试用 10³–15³，反演报告用 12³。

---

## B. Existing capabilities

只记**读过源码后确认存在**的能力，不按文件名推断。

### B.1 可用且数学上基本站住的

| 能力 | 位置 | 边界 |
|------|------|------|
| Cartesian 正交网格，非均匀间距，cell volume，邻居 | `core/grid.py` | 场是 `(nz,ny,nx)`；无 faces/normals 一级 API |
| SI 单位换算（Pa/bar、mD、cP） | `core/units.py` | 无 psi、mm、mL/min；YAML 不走该边界 |
| 单相稳态 TPFA 1D/2D/3D | `solver/pressure_solver.py` | 无 \(\lambda_t(S)\)、无重力势、无毛管 |
| 可选 \(\phi c_t V/\Delta t\) 存储项 | 同上 | 不是 compressibility 模型，只是裸数组 |
| 谐波传导率 | `solver/transmissibility.py` | 仅对角 \(k_x,k_y,k_z\) |
| Corey 两相 kr / \(f_w\) | `solver/relperm.py` | 产品路径默认用另一套 IMEX SWT 表 |
| Brooks–Corey / van Genuchten / Pc=0 求值 | `solver/capillary_pressure.py` | **未接入产品输运** |
| 显式毛管/重力水通量（可选） | `solver/capillary_flux.py`, `gravity_flux.py` | 仅 `saturation_solver`；默认关闭 |
| CFL 计算与违规抛错 | `solver/cfl.py` | 产品输运另写了一套，且 cap 子步 |
| ES-MDA 更新核：\(\sum 1/\alpha_i=1\)、扰动观测、对角 R、inflation、Gaspari–Cohn | `pipeline/ensemble_math.py` | 完整度够；被 wrapper 污染 |
| 物理正演 G(m)：不钉测点 Sw | `pipeline/forward_gm.py` | 时间单位启发式；Peaceman BHP |
| 空间留出工具（exclusive probe） | `pipeline/probe_split.py` | **产品 invert 不用** |
| 6 维通道管 / 裂缝条带参数化 | `k_param.py`, `frac_param.py` | 不是默认产品路径 |
| 实验室层理几何 | `pipeline/lab_horizon.py` | 网格阶梯文档有 15/30/50，测试不是 30³ |

### B.2 看起来完整、实际未接入产品或未完成

- `solver/saturation_solver.py`：更正规的显式两相，含可选 Pc/重力；产品用 `transport_saturation.py`。
- `solver/three_phase_*`：不可压三相核；pipeline 从不 import。
- `solver/tvd_transport.py`：1D TVD + 自适应 dt 报告；基线行为不变。
- `core/case.py`、`core/state.py`：空壳。
- BHP 井控：`Well(control=bhp)` 直接 `NotImplementedError`。
- `io/config_loader.py`：Archie/EM/acoustic 旧入口。

### B.3 产品声称有、物理上不是那回事

| 声称 | 实际 |
|------|------|
| 饱和度场 | 测点 IDW/克里金，再与输运场按测点个数加权混合 |
| 物性反演 | 用 \(k_{prior}\) 算通量，再 \(k\sim\mu\|u\|/\|\nabla p\|\)；循环依赖 |
| ES-MDA 反演 | 默认全网格 log k；事后与指示先验 / point-first k 混合 |
| 井点压力误差 ≈ 0 | 因为 Dirichlet 硬钉，不是 \(H(F(m))\) 拟合 |
| 三相 sw/so/sg | 输运只有水；so=1-sw，sg=0，测点三相再钉回去 |
| 数字孪生 | 没有 Controls/Observations 分离，没有 forecast CLI，没有状态 UQ |

---

## C. Requirement matrix

标记：`PASS` / `PARTIAL` / `FAIL` / `MISSING` / `OBSOLETE`

| § | 要求 | 状态 | 证据 |
|---|------|------|------|
| 1 | 动态 p,S vs 静态 K,φ；300 mm 立方 | **FAIL** | 产品把 S 当插值场；默认 YAML 是 100 m 盒子 |
| 2 | \(m,u,x,F,H\)；后验均值/方差 | **PARTIAL** | ES-MDA 写 `k_mean/k_std`；无 p/S ensemble；单图当重建 |
| 3 | Control ≠ Observation | **FAIL** | 同一井同时有 p、q、S；Dirichlet + 软观测；qw 用控制量 q |
| 4.1 | 空间留出验证 | **PARTIAL** | `probe_split` 存在；`invert_rock` 明确「产品用全部测点」 |
| 4.2 | 时间外推 | **MISSING** | 无 forecast 冻结 m；`max_times` 是整段抽样 |
| 4.3 | Synthetic truth = \(F(K_{true})\) | **FAIL** | `synthetic_twin` 沿通道**手绘** Sw，再钉井压 |
| 5 | 30³ / 10 mm baseline | **MISSING** | 无测试检查 \(0.3^3\) m³；lab 测试 10³–15³ |
| 6 | Parameter / Simulation / Observation 分离 | **FAIL** | 同一套 `(nz,ny,nx)` |
| 7 | ObservationOperator；传感器不决定网格 | **FAIL** | 吸附到 cell；无点插值/体积平均 |
| 8 | Grid 不绑死 nx/ny/nz；(n_cells,) 主表示 | **FAIL** | `Field3D` 强制 `grid.shape` |
| 9 | V1 静态局部加密 | **PARTIAL** | `mesh_refine` bbox 加密，不是井/缝/界面驱动 |
| 10 | 固定参数空间 + ParameterMapper | **MISSING** | 全网格成员维数=格子数 |
| 11 | 先 Region/CoarseField，禁止默认每 cell 独立 K | **FAIL** | 产品默认全网格 log k |
| 12 | \(\theta_K=\log K\)；S 约束 | **PARTIAL** | log k 有；φ 不反演；S clip 破坏守恒 |
| 13 | 正演不能停在稳态 Darcy | **FAIL**（产品） | nowcast 是稳态 TPFA + 插值 S |
| 14–15 | Model A/B/C 分层；C 为实验 baseline | **PARTIAL** | A 的压力核在；B 产品不完整；C 未组装 |
| 16 | Relperm 模块化 | **PARTIAL** | solver Corey vs pipeline 硬编码 SWT |
| 17 | 30 cm 不得默认 Pc=0 | **FAIL** | 产品输运无 Pc |
| 18 | 可压缩性配置 | **FAIL** | 硬编码 \(c_t=1.5\times10^{-9}\) |
| 19–20 | 裂缝架构可选，不过度实现 | **PARTIAL** | 高渗条带 + 6 维 θ；非 DFM |
| 21 | TPFA 仅 K-orthogonal；预留 MPFA | **PARTIAL** | 仅 TPFA；无接口预留 |
| 22 | Physics / Discretization / Solver 分离 | **FAIL** | pipeline 大函数；solver 碎片未编排 |
| 23 | 自适应 dt；失败不得 NaN 继续 | **FAIL** | 产品 cap 40 子步；`Field3D` 允许 NaN |
| 24 | 第一阶段 sequential/IMPES | **FAIL** | 无 IMPES 循环；压力不用 \(\lambda_t\)（solver 核）；pipeline 用缩放 k 凑 |
| 25 | 实验室 FlowPort，不默认 Peaceman | **FAIL** | `well_index.py` Peaceman；BHP 井控未实现 |
| 26 | ObservationModel 一级模块 | **MISSING** | |
| 27 | \(r^T R^{-1} r\) | **PARTIAL** | 更新用对角 R；诊断是白化 nRMSE |
| 28–29 | 完整 ES-MDA | **PARTIAL** | 数学核完整；wrapper 混合/伪造 ensemble |
| 30–32 | IES/EnKF 接口；offline/online | **MISSING** | 无空壳（好）；也无路由 |
| 33 | 更新后状态投影 | **PARTIAL** | clip S；无物理 restart |
| 34–36 | collapse / UQ / identifiability | **PARTIAL** | 有 inflation、k_std；无 posterior≈prior |
| 38 | 禁止 NN 代替物理 | **PASS** | 无 PINN；但启发式覆盖 G(m) |
| 39 | 目标目录 | **FAIL** | `pipeline/` 上帝模块 |
| 41 | 内部 SI；IO 边界换算 | **FAIL** | `dt>=0.5` 当天 |
| 42 | 场约定统一 | **FAIL** | `(nz,ny,nx)` 为主 |
| 43 | 无隐式全局状态 | **FAIL** | ES-MDA worker 模块级全局 |
| 44 | ensemble 串行/进程池 | **PARTIAL** | 有 process pool；失败回串行 |
| 46 | 单相解析/守恒验证 | **PARTIAL** | 1D/2D/3D 压力测试在；未绑 300 mm |
| 47 | Buckley–Leverett | **MISSING** | 产品无 BL 测试 |
| 48 | 毛管验证 | **OBSOLETE** | 测试已删（pycache 残留） |
| 49 | 三相验证 | **OBSOLETE** | 核还在，产品测试已删 |
| 50 | 300 mm / 10 mm / 0.027 m³ | **MISSING** | |
| 51 | 任意坐标观测算子 | **MISSING** | 只测 cell center |
| 52 | 线性高斯反演 benchmark | **MISSING** | |
| 53 | ES-MDA synthetic（data/K/hold-out/UQ） | **FAIL** | 测试只要求 nRMSE<50、k 有空间变化、对比度>1.15 |
| 54–55 | hold-out + forecast example | **MISSING** | |
| 56 | 每次模拟质量守恒报告 | **FAIL** | CLI 不输出相质量衡算 |
| 57 | 每次反演 diagnostics | **PARTIAL** | 有 nRMSE、α；无 update norm / failed members |
| 58 | ensemble 失败成员策略 | **MISSING** | 无 NaN 成员剔除 |
| 59–60 | 工程 YAML + simulate/invert/forecast | **FAIL** | `--mode slice/series/discovery/esmda` |
| 63 | 删除错误测试 | **FAIL** | 错误测试留下；正确物理测试删了 |
| 66 | 删除 Archie/EM/acoustic 通用反演 | **OBSOLETE** | `io/config_loader.py` |
| 67 | 反演 = history matching | **FAIL** | 产品「反演」仍是公式/插值 |
| 68 | 薄 DigitalTwin | **MISSING** | |
| 71–72 | model_assumptions / 适用边界 | **MISSING** | 旧 docs 已删，新的未写 |
| 74 P0 | 见规格清单 | **FAIL** | 未完成 |
| 75 P1 | 实验室多相反演 baseline | **FAIL** | 不能自称已达到 |

---

## D. Physics errors

### D.1 问题定义错误（最严重）

`references/软件要求.txt` 第 10 行把饱和度写成物性场。`point_workflow` / `reconstruct_saturation` / `invert_rock_properties` 按此实现：

- 每个时刻独立插值 S；
- 用该 S 和 p 估点 k、φ；
- 再空间插值到全网格。

这把**动态状态**当成**静态参数**。与 `check.txt` §1、§67 直接冲突。

### D.2 控制量与观测混用

水驱产品路径：

- 注入/采出压力进入 `cell_dirichlet`（`pressure_field.py`）；
- 同时 `well_rate` 作为源项（同一 cell 上 Dirichlet 赢）；
- ES-MDA 再把井压当软观测；
- 再把 `qw = |q_control| · f_w(S)` 当第三条观测。

衰竭路径：生产者 BHP 是 Dirichlet，默认仍进观测向量，残差对任何 k 都约 0。

`qw` 不是独立产水观测，只是已经同化过的 Sw 乘上控制流量。

### D.3 产品饱和度不是 PDE 解

`blend_recon_transport_sw`（`point_workflow.py`）按测点个数把 IDW 饱和度与输运饱和度加权。`inversion._sw_tongue_along_k` 沿高渗走廊涂 ΔSw。这是为 Dice 调图，不是 \(S=F(m,u)\)。

### D.4 毛管默认关闭

30 cm 实验室尺度规格禁止默认 Pc=0。产品输运不含毛管。solver 有 Brooks–Corey / VG，但 `capillary_pressure.py` 写明不耦合；产品从不调用。

### D.5 重力不在压力方程里

Darcy 应为 \(u_\alpha=-K\lambda_\alpha(\nabla p_\alpha-\rho_\alpha g)\)。压力 TPFA 与 `compute_face_fluxes` 无重力势。重力只作为输运附加项且默认关。

### D.6 三相未进入产品

`phases_from_sw`：`so=1-sw`，`sg=0`。`invert_rock_properties` 忽略 so、sg。三相模块孤立。

### D.7 可压缩性硬编码

`forward_gm.DEFAULT_CT = 1.5e-9`。无 incompressible / slightly_compressible / black-oil 配置。输运按不可压孔隙体积更新。

### D.8 实验室井模型用 Peaceman

`peaceman_well_index` 默认 `rw = min(0.10, 0.08·min(dx,dy))`。10 mm 格子上 rw 接近格子尺度。规格要求先做 `FlowPort`（面片/孔），不是油藏井。

### D.9 指示先验把动态信号写成地质

`auto_inversion` 用 ΔSw、Δp、通量做 shape indicator，再 `enhance_permeability_from_indicator` 当 k 先验。这把一次水驱足迹固化成静态通道，对衰竭/多层/裂缝会系统性偏。

### D.10 合成真值不是正演

`synthetic_twin.py`：单相 TPFA 后**覆盖**井压；Sw 按通道进度手绘。后续「反演恢复通道」测的是能否找回自己画的舌头，不是能否恢复 \(K_{true}\)。

---

## E. Numerical errors

### E.1 边界通量丢失

`compute_face_fluxes` 只填内部面，外边界通量恒 0。压力 Dirichlet 实际有边界质量进出，输运看不见入口注水，除非调用方另写边界通量。

### E.2 时间单位启发式

`forward_gm.dt_to_seconds` 与 `transport_saturation`：`dt >= 0.5` 当天，通量乘 86400。YAML 时刻 `0,30,60,90` 无单位。同一数字可被当成 30 秒或 30 天。违反内部 SI。

### E.3 CFL 被硬上限截断

产品输运：`n_eff = min(ceil(dt/dt_cfl), 40)`。大 dt 时仍可能超 CFL，然后 `clip(Sw)`。solver 路径 clip 后再报质量守恒，clip 已破坏 \(\sum\phi V\Delta S\)。

### E.4 CFL 未含 \(|f'(S)|\)

`cfl.py` 用 \(\mathrm{dt}\sum|q| / (\phi V)\)。Corey \(f_w\) 的 \(|f'|\) 可大于 1。

### E.5 active_mask 不进求解器

网格存 inactive，组装与输运仍扫全部 cell。

### E.6 压力质量衡算不完整

3D 报告只比 Dirichlet 边界流出与井源。Neumann、storage、Dirichlet cell 上的井被漏掉。

### E.7 循环 Darcy「反演」

通量由 \(k_{prior}\) 计算，再 \(k\leftarrow \mu|u|/|\nabla p|\)。一致时收回先验；不一致时得到先验的重缩放，不是独立观测反演。

### E.8 Ensemble 污染

- 无 NaN 成员剔除。
- Worker 用模块全局缓存 mesh/samples。
- Frac 路径把 `k_ensemble` 做成 shape `(1, nz, ny, nx)` 的假 ensemble。
- `run_esmda_permeability` 结束后 `k ← 0.7 k_mean + 0.3 k_{point-first}`。

### E.9 自适应时间步进不存在于产品

solver 超 CFL 就抛错；产品自己 cap 子步。无非线性失败回退、无 ΔS 控制器。

---

## F. Inversion errors

现有「反演」大部分**不是** history matching。

### F.1 三条互相矛盾的 invert

| 入口 | 实际做的事 |
|------|------------|
| `invert_rock_properties` | 单时刻代数 k、φ |
| `run_sensor_inversion` | 6 维井间通道管（自己 warn 非默认） |
| `invert_rock` / `run_esmda_permeability` | 指示先验 + **全网格** log k ES-MDA |

产品默认是第三条。规格禁止把每 cell 独立 K 当第一版。

### F.2 ES-MDA 核可用，产品包装不可用

`ensemble_math.esmda_update_step` 是真的多步 MDA。产品包装：

- 参数维 = 格子数（欠定）；
- qw 观测非法；
- 衰竭 BHP 双重使用；
- 后验与先验/插值混合；
- 发布场用 Dirichlet nowcast，不是 G(m) ensemble；
- 测试 `observation_rmse[-1] < 50` 几乎总能过。

### F.3 没有参数化层级

缺 Region / CoarseField / ParameterMapper。6 维通道管假设「一口注一口采一条管子」，实验室层理/隔层/裂缝都会错。lab README 自己写：井点 p 能对上，层理 Dice 低。

### F.4 验证测错了东西

- 井点 p 误差 0 = 硬钉成功。
- Dice > 0.10、对比度 > 1.15 = 比 IDW 稍有结构。
- CMG gap：全场 Sw L2 ≈ 0.39，井点 p = 0。报告把井点 0 误差当成「传感器硬约束工作正常」——对 check.txt 这是缺陷不是成绩。

### F.5 无 identifiability

低指示区把后验拉回先验（`_blend_posterior`），而不是报告「数据没约束住」。无 prior/posterior 方差比。

### F.6 Offline / online 未分开

只有参数 smoother。无 EnKF。新数据到来只能重跑整段。

---

## G. Architecture debt

1. **上帝模块 `pipeline/`**：网格、插值、正演、反演、CLI、孪生全在一起。
2. **双物理栈**：`solver/` 与 `pipeline/transport_*` 重复且不一致（两套 relperm、两套 CFL、两套输运）。
3. **场布局绑死 `(nz,ny,nx)`**：后续局部加密/非结构必须重写。
4. **兼容 shim**：未使用 kwargs、`mode` 只有 point_first、legacy `observer`、忽略的 `n_outer_loops`。
5. **测试锁死错误行为**：`test_pipeline_fields.py` 要求测点 p/S 机器精度吻合。
6. **正确测试被删**：pycache 里有 capillary、gravity、IMPES、three-phase、BL 相关旧测试，源文件已不在。
7. **文档断裂**：README 链到已删除的 `docs/ARCHITECTURE.md` 等；`docs/` 只剩 `check.txt`（本审查前）。
8. **STATUS 过称**：把插值/硬钉/烟雾测试标成「已验证」。
9. **隐式全局状态**：multiprocess worker globals。
10. **配置不是实验描述**：无 `physics.model`、`experiment.controls`、`inverse.parameterization`。

---

## H. Delete list

删除或降级为「诊断/非产品」，不要为保 pytest 数量留下。

### H.1 产品路径中应降级或删除

| 模块 | 原因 |
|------|------|
| `pipeline/property_field.invert_rock_properties` 作为「反演」 | 循环 Darcy，违反 §67 |
| `pipeline/saturation_field.reconstruct_saturation` 作为产品 S | 把状态当静态场 |
| `pipeline/point_workflow` 作为主 workflow | 实现的是软件要求，不是 check.txt |
| `inversion._sw_tongue_along_k` / `_sw_fill_corridor_1d` | 涂饱和度 |
| `esmda.py` 0.7/0.3 k 混合 | 污染后验 |
| `auto_inversion._blend_posterior` | 用启发式覆盖分析步 |
| `io/config_loader.py` Archie/EM/acoustic | §66 旧通用反演 |
| 未使用 kwargs / `mode` 兼容 | 绿地禁止 shim |
| `solver/*enhancement_report.py` | 包装未实现能力的成功报告 |

### H.2 应删除或改写的测试

| 测试 | 原因 |
|------|------|
| `tests/test_pipeline_fields.py` 中「测点必须机器精度吻合」 | 把 Dirichlet/pin 锁成合同 |
| `tests/test_sensor_series_inversion.py` 同上 | |
| `tests/test_pipeline_fields.py::test_property_inversion_positive` | 只测 k>0 |
| `tests/test_sensor_io_esmda.py::test_esmda_reduces_well_pressure_misfit` 的 `nRMSE<50` | 无物理内容 |
| `tests/test_auto_inversion.py` 仅对比度>1.15 | 不测 hold-out / \(K_{true}\) |
| `tests/test_shape_discovery.py` Dice>0.10 | 过软 |

### H.3 不要当产品留下的叙事

- README / STATUS 的「四场重建」作为最终产品定义。
- 「井点压力误差必须为 0」。
- 「同一测点永远不能同时测 p 和 S」作为物理定律（可作实验设计约定，不应写进求解器）。

---

## I. Reuse list

绿地 ≠ 全部重写。下列实现数学正确或接近，应迁到新 domain，而不是扔掉。

| 实现 | 为什么留 |
|------|----------|
| `core/grid.py` 几何（volume、spacing、locate、neighbors） | Cartesian baseline 需要 |
| `core/units.py` | 扩展后作 IO 边界 |
| `core/exceptions.py` | 扩展物理错误类型 |
| `solver/transmissibility.py` | K-orthogonal TPFA |
| `solver/pressure_solver.py` 1D/2D/3D | Model A |
| `solver/relperm.py` Corey | Model B |
| `solver/capillary_pressure.py` + `capillary_flux.py` | 接入 IMPES |
| `solver/cfl.py` + `estimate_stable_dt` | 自适应 dt 的材料 |
| `solver/velocity.py`（修边界通量后） | 守恒通量 |
| `solver/well_source.py` 定流量源 | 可演化成 FlowPort |
| `solver/three_phase_*` | P1 材料，不进 P0 产品 |
| `pipeline/ensemble_math.py` | ES-MDA 分析步 |
| `pipeline/forward_gm.py` 的「不钉 Sw」原则 | 反演正演必须如此 |
| `pipeline/probe_split.py` | **仅评估**，不进产品 invert |
| `pipeline/lab_horizon.py` | 300 mm 层理几何 |
| `pipeline/sensor_io.py` CSV 读写 | 改成 Control/Observation 后复用解析 |
| `tests/test_pressure_solver_{1,2,3}d.py` | 单相验证 |
| `tests/test_transmissibility.py` / `test_relperm.py` / `test_cfl.py` / `test_core_grid*.py` | 数值单元测试 |

`solver/saturation_solver.py` 比 `transport_saturation.py` 更接近 Model B，应作为输运基线（去掉 clip-当成功、补边界通量、接 Pc、自适应 dt），而不是继续发展 pipeline 那份循环 Python。

---

## 模块分类（§64）

| 模块 | 分类 | 原因 |
|------|------|------|
| `core/grid.py` | **REFACTOR** | 留几何；加 (n_cells,) helper；不要再让 solver 写死 `[k,j,i]` |
| `core/field.py` | **REWRITE** | 主表示改为 (n_cells,)；拒 NaN |
| `core/wells.py` | **REWRITE** | 拆成 FlowPort；Peaceman 降为可选 WellModel |
| `core/units.py` | **KEEP** | 补 mm / mL/min / psi |
| `core/case.py`, `core/state.py` | **REWRITE** | 空壳改为 Experiment / State |
| `solver/pressure_solver.py` | **REFACTOR** | Model A；IMPES 里用 \(\lambda_t\) |
| `solver/transmissibility.py` | **KEEP** | |
| `solver/relperm.py` | **REFACTOR** | 收成 RelpermModel |
| `solver/capillary_*` | **REFACTOR** | 默认对 30 cm 可开 |
| `solver/saturation_solver.py` | **REFACTOR** | 产品 Model B |
| `solver/three_phase_*` | **KEEP** | P1，不接 P0 |
| `solver/tvd_transport.py` | **KEEP** 低优先级 | 1D 研究代码 |
| `solver/*enhancement_report.py` | **DELETE** | |
| `solver/linear_solver_backend.py` | **DELETE 或 KEEP 诊断** | 真求解器用 spsolve |
| `pipeline/ensemble_math.py` | **KEEP** | 迁到 `inverse/` |
| `pipeline/forward_gm.py` | **REFACTOR** | 正式 Forward |
| `pipeline/esmda.py` | **REFACTOR** | 去掉 history 混合与全网格默认 |
| `pipeline/probe_split.py` | **KEEP** | 仅 validation |
| `pipeline/lab_horizon.py` | **KEEP** | |
| `pipeline/k_param.py`, `frac_param.py` | **REFACTOR** | 变成 Parameterization 实现，非自动门控主路径 |
| `pipeline/state.py` | **REWRITE** | Controls / Observations |
| `pipeline/run.py`, `time_series.py` | **REWRITE** | DigitalTwin CLI |
| `pipeline/auto_inversion.py`, `inversion.py` | **REWRITE** | 去掉启发式主路径 |
| `pipeline/pressure_field.py`, `saturation_field.py`, `property_field.py`, `point_workflow.py` | **DELETE 产品职责** | 可留 `diagnostics/nowcast.py` |
| `pipeline/spatial_interp.py` | **KEEP** | 仅先验/可视化，不作状态 |
| `pipeline/shape_indicator.py` | **KEEP** 后期 | 实验设计，不作 k 真值 |
| `pipeline/synthetic_twin.py` | **REWRITE** | 必须 \(d=H(F(m_{true}))\) |
| `io/config_loader.py` | **DELETE** | Archie 栈 |
| `io/structured_deck.py` | **KEEP** | 外部 deck 读入，非 runtime |

---

## STATUS / README 过称（摘录）

- 「网格划分 已验证」：测的是 100×50×20 m，不是 30³。
- 「饱和度场 MVP」：IDW，不是输运状态。
- 「物性 invert_rock_properties MVP」：循环 Darcy。
- 「ES-MDA 已接到产品」：全网格 log k + 假观测 qw + 后验混合。
- 「实验室 30 cm MVP」：验收网格 12³；井点 p=0 因硬钉；Dice 0.12–0.15。
- README 仍指向已删除的 `docs/ARCHITECTURE.md` 等四个文件。

---

## 对 §83 十二问的当前答案

1. 当前物理假设？**文档说不清。** 产品是稳态 TPFA + 插值 S + 可选无毛管迎风；solver 另有未接线的 Pc/重力/三相。
2. 注入采出条件？**未建模为 Controls。** p 和 q 同时出现。
3. 哪些数据参与反演？**产品：全部测点。** 井压既是 BC 又是数据。
4. 哪些数据留作验证？**产品不用 hold-out。**
5. K 如何参数化？**默认每 cell 一个 log k。**
6. 数据能否识别这些参数？**系统不回答。** 低敏感区被拉回先验。
7. 正演是否质量守恒？**产品不报告；clip 破坏守恒；边界通量丢失。**
8. posterior 是否更能预测未参与反演的数据？**产品未测。**
9. 哪些区域被约束？**无 identifiability 图。**
10. 哪些区域 uncertainty 仍大？**只有 k_std，且被混合污染。**
11. 新数据如何更新 twin？**只能重跑。无 online。**
12. 预测失败如何归因？**不能。** 插值、硬钉、启发式和 G(m) 缠在一起。

---

## 下一步（本文件之后）

1. 已写 `docs/target_architecture.md`。
2. **不要先搬目录。** 先改问题定义：Controls/Observations、状态/参数分离、观测算子、IMPES Model B、粗参数化 ES-MDA。
3. 用户确认目标架构后再做绿地重构。
