# 项目状态

主线：济阳页岩油组分 CO2 吞吐（一注四采）数字孪生；反演是低维 θ 上的 LM。

| 状态 | 含义 |
|------|------|
| **已验证** | 有实现与针对性测试，且测试测的是规格要求的行为 |
| **MVP** | 可用，假设写在 `docs/model_assumptions.md` |
| **不做** | P0 明确排除 |

## 能力表

| 能力 | 状态 | 入口 | 证据 |
|------|------|------|------|
| 300 mm / 10 mm → 30³，体积 0.027 m³ | 已验证 | `CartesianGrid.uniform` | `tests/grid/test_grid_lab.py` |
| 控制 / 观测分离 | 已验证 | `ControlSeries` / `ObservationSeries` | `tests/physics/test_pressure_analytical.py` |
| 非格点观测算子 | 已验证 | `ObservationOperator` | `tests/observation/test_observation_operator.py` |
| 单相 1D 压力 | 已验证 | IMPES `single_phase` | `tests/physics/test_pressure_analytical.py` |
| 两相 IMPES + CFL | MVP | `solver.impes.simulate` | `tests/solver/test_buckley_leverett.py` |
| 黑油表面体积 \(F\) | 已验证 | `physics.pvt.BlackOilPVT` | `tests/physics/test_black_oil.py` |
| TPFA 离散（迎风 \(\lambda\)、重力、\(k_z\)、TRANSI、SWT） | 已验证 | `discretization.tpfa`、`TableTwoPhase` | 五点/断层 \(F(K)\) p RMSE 21–23 psi，Sw 0.018–0.030 |
| 隐式输运（后向 Euler + Newton） | 已验证 | `solver.transport.implicit_water` | `tests/physics/test_black_oil.py`；CMG 尺子默认开 |
| 质量守恒报告 | MVP | `MassBalance`（地面水体积） | `tests/physics/test_mass_balance.py` |
| 毛管模型（显式选择） | 已验证 | `BrooksCorey` / `NoCapillary` | `tests/physics/test_capillary.py` |
| 相势通量（\(P_c\) + 重力分异） | 已验证 | `tpfa._phase_face_ops` | `tests/physics/test_capillary.py` |
| 井筒水头 + 隐式分异通量 | 已验证 | `impes._connection_bhp` | `tests/solver/test_well_index.py`、`test_capillary.py` |
| 均匀 `*PRES` 初值 + 格子 \(\rho(p)\) | 已验证 | `DigitalTwin.initial_state` | `tests/physics/test_capillary.py` |
| 活油脱气尺子（井底 < 泡点） | MVP | `cmg_fault_channel_lib` | PVT=`cmg_seawater`（`pvt_from_cfg`）；1 天 p RMSE **5.8 psi**、均 \(S_g\) 0.017 vs 0.016；反演对比度 **39.6**（真 40）、log \(K\) RMSE **0.055**，pass；FIM 闸门未过默认关 |
| 表导数 \(c_g\) + 闪蒸后二次压力 | 已验证 | `BlackOilPVT.cg_of` | `tests/physics/test_pvt_live_oil.py` |
| 线性高斯 LM | 已验证 | `inverse.lm.run_lm` | `tests/inverse/test_lm_linear.py` |
| Synthetic \(H(F(m_{true}))\) + hold-out | MVP | `synthetic` | `tests/inverse/test_synthetic_twin.py` |
| 冻结 m 的 forecast | MVP | `DigitalTwin.forecast` | `tests/inverse/test_forecast.py` |
| CLI validate/simulate/invert/forecast/synthetic | MVP | `reservoir` | `tests/cli/test_cli.py` |
| 实验室 apply 交付门闩（6 mm、2-region、预报） | 已验证 | `reservoir apply` | `tests/cli/test_apply.py` |
| 隐式输运（两相和三相默认，YAML `transport`） | 已验证 | `solver.transport.implicit_water` | `tests/physics/test_black_oil.py`、`tests/physics/test_three_phase.py`、`tests/inverse/test_structure.py` |
| 构造目录 hold-out 自选（1/2/3 层、contrast） | MVP | `inverse.structure`、`apply --auto` | `tests/inverse/test_structure.py` |
| 测点 CSV（SI / 分钟·kPa、hold-out、无 --demo） | 已验证 | `io.case` / `apply` | `tests/cli/test_apply.py`、`tests/io/test_case_csv.py` |
| 已知通道 region_map + 对比度 | 已验证 | `make_channel_waterflood` | `tests/inverse/test_synthetic_twin.py` |
| 三相顺序隐式 + hybrid 迎风 + 冻 \(q_T\) 井分流 | MVP | `implicit_blackoil` / `_well_transport_sources` | `tests/physics/test_three_phase.py` |
| 步末更新压力（P→T→P）+ 按 \(\Delta S/\Delta p\) 选步 | 已验证 | `PhysicsSpec.reupdate_pressure`、`state_change_timestep` | `tests/physics/test_three_phase.py` |
| 顺序输运守恒油+气 + Brenier 重力 extras | 已验证 | `implicit_blackoil(conserve=oil_gas)`、`sequential_gravity_face` | 放气 1 天 **5.6 psi** / 均 \(S_g\) **0.0165** vs 0.0162 |
| 油通量面迎风 \(b_o\) + 活油压力增量迭代 | 已验证 | `implicit_blackoil`、`_sfi_pressure_flux`、闪蒸后 `_picard_pressure` | 放气 1 天 **5.4 psi** / 均 \(S_g\) 0.017 |
| 顺序势迎风（Brenier 含 \(v_T\)，默认） | 已验证 | `upwind_type=potential`、`sequential_phase_fluxes` | 放气 1 天 **6.2 psi**；`hybrid` 仍可回退 |
| 两相 / 临界饱和度截断 | 已验证 | `critical_point_chop` | 两层 1 天 **7.0 psi** / \(S_w\) 0.035 |
| 井筒混合 / CNV / 牛顿松弛 / 按迭代选步 | 已验证 | `solver.seqtools` | `tests/solver/test_seqtools.py` |
| 全隐式黑油牛顿（\((p,S_w,x)\)，\(x=R_s\) 或 \(S_g\)） | MVP | `solver.fi.solve_fi_step`、`solver.adnum.CellAD`、`run_fim_ladder.py` | **PVT=`cmg_seawater`**。阶梯：死油 ~0.43；无放气活油 ~0.73 / dsg≈0；放气仍 ~10 psi / Sg 0.013。**本轮**：活油 Newton 初值改 \(p^n\)（IMPES 猜压导致线搜索全失败）；FIM Δt 按 Newton 次数 chop/grow（不再用显式 CFL 或放气阶梯 300 s 帽）；disappear 折回 \(R_s\)。步末全闪蒸会 underflow。vs CMG ~**11.5 psi**（闸门未过）；默认关 |
| 活油 \(R_s\) 守恒 + 表黏度 | 已验证 | `BlackOilPVT.flash_from_total` | `tests/physics/test_pvt_live_oil.py` |
| 点估计场 \(F(\hat\theta)\) | MVP | `DigitalTwin.reconstruct` | `tests/inverse/test_reconstruct_uq.py` |
| CSV 控制/观测 IO | 已验证 | `io.case` | `tests/io/test_case_csv.py` |
| 任意深度柱面测点 | 已验证 | `column_sensors` | `tests/observation/test_observation_operator.py` |
| CMG 测点诊断（非产品尺子） | MVP | `validation/black_oil/cmg_lab_layers/run_invert_eval.py` | `invert_eval_report.json` |
| 自洽两层 K 收回 | 已验证 | `make_two_layer_waterflood` | `tests/inverse/test_synthetic_twin.py` |
| 组分 EXAMPLE 孪生（等温气–油，C1–nC10） | MVP | `solver.fi_comp`、`eos/`、`comp/` | `tests/cases/test_comp_twin.py`；`examples/compositional/comp_example.yaml`。定流量井 \(p_{\mathrm{wf}}\) 进 \(H\)。2-region LM invert（数据 nRMSE 下降、对比度方向对）。不是济阳 GEM |
| 组分 immiscible 水相 | MVP | `CompSpec.has_water` | `tests/cases/test_comp_water.py`、`examples/compositional/comp_example_water.yaml`。水进 \(F\) 和 \(H\)（\(S_w\)+率井 BHP）；2-region LM invert，对比度方向对。水不进 PR |
| 公开 PR 牌加载 | 已验证 | `io.eos_load.load_eos_card` | `tests/physics/test_eos_load.py`；fixture 抄 OPM `1D_COMP` 数字。缺文件拒绝 |
| 济阳井网 GEM CO2 吞吐尺子（离线） | MVP | `validation/jiyang/cmg_co2_hnp` | 1 注 4 采水平井；EXAMPLE 三组分牌；克隆 `gmspr003`。井史 truth，不是现场 `.gem`，产品不调 CMG |
| 济阳井网组分孪生入口 | MVP | `examples/jiyang/jiyang_co2_hnp.yaml` | 同一井网 + 公开牌 + GEM BHP 观测。contrast θ + LM。`run_compare.py` 画 GEM vs \(F(m_{\mathrm{true}})\)。21×21×5 正演约 50 min。生产者 BHP nRMSE 未降前不跑全井网 invert |
| 页岩衰竭 frac θ LM | MVP | `inverse.frac`、`io.shale_case` | 默认 **4 维** θ + 顺序两相 + **MIN BHP=1500 psi**；合成 nRMSE ~2–2.7；跨 IMEX S1 nRMSE **~7.4**、`dp_ratio≈0.26`（`s1_inversion_report.json`）。`tests/cases/test_shale_synthetic.py`、`test_well_index.py::test_rate_producer_min_bhp_floor` |
| 统一 invert run report | MVP | `twin.run_report`、`cli.reporting` | `invert.json` + `residuals.csv`；页岩 suite 写 `run_reports[]` |
| check83 十二问验收 | MVP | `twin.acceptance` | `check83.json`；`docs/check83_acceptance.md`；`tests/twin/test_check83_report.py` |
| LM 后验小 ensemble Ne=8 | MVP | `inverse.post_ensemble` | `k_mean.npy` / `k_std.npy`；`tests/inverse/test_post_ensemble.py` |
| Forecast 时间外推尺子 | MVP | `synthetic.make_forecast_split_case` | `tests/inverse/test_forecast.py`；页岩 S5 `run_forecast_validate.py`（slow） |
| 页岩 YAML invert | MVP | `examples/shale_oil/s1–s5.yaml` | `load_case` → `truth_json` + `imex_out` |

## 排除

- 四场插值「反演」（已删除）
- 每 cell 独立反演 27k 个 K（非默认）
- Archie / EM / acoustic 通用反演（已删除）
- EnKF、ES-MDA、逐格 \(K\)、MPFA、动态 AMR、PINN。济阳现场 GEM 牌（没有 `.gem` 就不编造）、热。活油黑油仍是 \(R_s\) 表闪蒸，不是组分核
- 旧 `pipeline/` 产品路径（已删除）

## 实验室默认（2026-08 重构）

- 30 cm 立方，10 mm 网格，探头直径 6 mm（`H` 在插值场上做球平均）
- 反演默认 2 region log K，不是粗网格 6³ / 逐格 K
- 三维 p/S 是 F(m_post) 重建；产品尺子是自洽正演（F(m_true) → 反演 → 贴回 F），不是场 Dice 对 CMG
- `reservoir harness` 已删除
- 概念实验室 30 cm：`examples/lab/lab_concept.yaml` + `concept_probes.csv`（75 电阻率 + 16 新增 7.5 cm）。invert 对比 = 水驱相似准则 + \(F(m_post)\)/\(F(m_true)\) 场 nRMSE，不是 CMG。`tests/cases/test_lab_concept.py`。
- 30 cm 产品开发计划（活文档）：`docs/lab_product.qmd`
