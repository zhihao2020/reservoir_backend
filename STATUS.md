# 项目状态

主线：实验室 300 mm 立方试块的多相正演 + ensemble 反演。

| 状态 | 含义 |
|------|------|
| **已验证** | 有实现与针对性测试，且测试测的是规格要求的行为 |
| **MVP** | 可用，假设写在 `docs/model_assumptions.md` |
| **不做** | P0 明确排除 |

## 能力表

| 能力 | 状态 | 入口 | 证据 |
|------|------|------|------|
| 300 mm / 10 mm → 30³，体积 0.027 m³ | 已验证 | `CartesianGrid.uniform` | `tests/test_grid_lab.py` |
| 控制 / 观测分离 | 已验证 | `ControlSeries` / `ObservationSeries` | `tests/test_pressure_analytical.py` |
| 非格点观测算子 | 已验证 | `ObservationOperator` | `tests/test_observation_operator.py` |
| 单相 1D 压力 | 已验证 | IMPES `single_phase` | `tests/test_pressure_analytical.py` |
| 两相 IMPES + CFL | MVP | `solver.impes.simulate` | `tests/test_buckley_leverett.py` |
| 黑油表面体积 \(F\) | 已验证 | `physics.pvt.BlackOilPVT` | `tests/test_black_oil.py` |
| TPFA 离散（迎风 \(\lambda\)、重力、\(k_z\)、TRANSI、SWT） | 已验证 | `discretization.tpfa`、`TableTwoPhase` | 五点/断层 \(F(K)\) p RMSE 21–23 psi，Sw 0.018–0.030 |
| 隐式输运（后向 Euler + Newton） | 已验证 | `solver.transport.implicit_water` | `tests/test_black_oil.py`；CMG 尺子默认开 |
| 质量守恒报告 | MVP | `MassBalance`（地面水体积） | `tests/test_mass_balance.py` |
| 毛管模型（显式选择） | 已验证 | `BrooksCorey` / `NoCapillary` | `tests/test_capillary.py` |
| 相势通量（\(P_c\) + 重力分异） | 已验证 | `tpfa._phase_face_ops` | `tests/test_capillary.py` |
| 井筒水头 + 隐式分异通量 | 已验证 | `impes._connection_bhp` | `tests/test_well_index.py`、`test_capillary.py` |
| 均匀 `*PRES` 初值 + 格子 \(\rho(p)\) | 已验证 | `DigitalTwin.initial_state` | `tests/test_capillary.py` |
| 活油脱气尺子（井底 < 泡点） | MVP | `cmg_fault_channel_lib` | hybrid + 冻 \(q_T\) 1 天 p RMSE **13.9 psi**、均 \(S_g\) 0.019 vs 0.016；VO 主变量已实现但默认关 |
| 表导数 \(c_g\) + 闪蒸后二次压力 | 已验证 | `BlackOilPVT.cg_of` | `tests/test_pvt_live_oil.py` |
| 线性高斯 ES-MDA | 已验证 | `inverse.esmda.run_esmda` | `tests/test_esmda_linear.py` |
| Synthetic \(H(F(m_{true}))\) + hold-out | MVP | `validation.synthetic` | `tests/test_synthetic_twin.py` |
| 冻结 m 的 forecast | MVP | `DigitalTwin.forecast` | `tests/test_forecast.py` |
| CLI validate/simulate/invert/forecast/synthetic | MVP | `reservoir` | `tests/test_cli.py` |
| 实验室 apply 交付门闩（6 mm、2-region、预报） | 已验证 | `reservoir apply` | `tests/test_apply.py` |
| 隐式输运（两相和三相默认，YAML `transport`） | 已验证 | `solver.transport.implicit_water` | `tests/test_black_oil.py`、`tests/test_three_phase.py`、`tests/test_structure.py` |
| 构造目录 hold-out 自选（1/2/3 层、contrast） | MVP | `inverse.structure`、`apply --auto` | `tests/test_structure.py` |
| 测点 CSV（SI / 分钟·kPa、hold-out、无 --demo） | 已验证 | `io.case` / `apply` | `tests/test_apply.py`、`tests/test_case_csv.py` |
| 已知通道 region_map + 对比度 | 已验证 | `make_channel_waterflood` | `tests/test_synthetic_twin.py` |
| 三相顺序隐式 + hybrid 迎风 + 冻 \(q_T\) 井分流 | MVP | `implicit_blackoil` / `_well_transport_sources` | `tests/test_three_phase.py` |
| 步末更新压力（P→T→P）+ 按 \(\Delta S/\Delta p\) 选步 | 已验证 | `PhysicsSpec.reupdate_pressure`、`state_change_timestep` | `tests/test_three_phase.py` |
| 顺序输运守恒油+气 + Brenier 重力 extras | 已验证 | `implicit_blackoil(conserve=oil_gas)`、`sequential_gravity_face` | 放气 1 天 **5.6 psi** / 均 \(S_g\) **0.0165** vs 0.0162 |
| 油通量面迎风 \(b_o\) + 活油压力增量迭代 | 已验证 | `implicit_blackoil`、`_sfi_pressure_flux`、闪蒸后 `_picard_pressure` | 放气 1 天 **5.4 psi** / 均 \(S_g\) 0.017 |
| 顺序势迎风（Brenier 含 \(v_T\)，默认） | 已验证 | `upwind_type=potential`、`sequential_phase_fluxes` | 放气 1 天 **6.2 psi**；`hybrid` 仍可回退 |
| 两相 / 临界饱和度截断 | 已验证 | `critical_point_chop` | 两层 1 天 **7.0 psi** / \(S_w\) 0.035 |
| 井筒混合 / CNV / 牛顿松弛 / 按迭代选步 | 已验证 | `solver.seqtools` | `tests/test_seqtools.py` |
| 全隐式黑油牛顿（\((p,S_w,x)\)，\(x=R_s\) 或 \(S_g\)） | MVP | `solver.fi.solve_fi_step`（`FiStepResult`、本地命名辅助见 `docs/fim_name_map.md`） | 放气尺子约 **8.98 psi** / 均 \(S_g\) 0.014（闸门 ≤6.5 psi 未过）；默认关；`tests/test_fim_helpers.py`、`test_fim_liberation_gate.py` |
| 活油 \(R_s\) 守恒 + 表黏度 | 已验证 | `BlackOilPVT.flash_from_total` | `tests/test_pvt_live_oil.py` |
| 后验 p/S/K 分位数与标准差 | MVP | `DigitalTwin.reconstruct` | `tests/test_reconstruct_uq.py` |
| CSV 控制/观测 IO | 已验证 | `io.case` | `tests/test_case_csv.py` |
| 任意深度柱面测点 | 已验证 | `column_sensors` | `tests/test_observation_operator.py` |
| CMG 测点诊断（非产品尺子） | MVP | `cmg_lab_layers/run_invert_eval.py` | `invert_eval_report.json` |
| 反演预设 / 时限 / hold-out 排行 | MVP | `calibrate_auto`、`inverse.presets` | `tests/test_portfolio.py` |
| ES / ES-MDA / ES-MDA-RS + hold-out 混合 | MVP | `inverse.algorithms`、`greedy_holdout_blend` | `tests/test_esmda_linear.py`、`test_portfolio.py` |
| Ensemble 并行正演 | MVP | `inverse.parallel.map_members` | `tests/test_parallel.py` |
| 限时 HPO（搜算法旋钮） | MVP | `inverse.hpo.run_hpo` | `tests/test_hpo.py` |
| 自洽两层 K 收回 | 已验证 | `make_two_layer_waterflood` | `tests/test_synthetic_twin.py` |

## 排除

- 四场插值「反演」（已删除）
- 每 cell 独立反演 27k 个 K（非默认）
- Archie / EM / acoustic 通用反演（已删除）
- EnKF、MPFA、动态 AMR、PINN。全隐式组分闪蒸（活油是顺序 IMPES + 总气量闪蒸）
- 旧 `pipeline/` 产品路径（已删除）

## 实验室默认（2026-08 重构）

- 30 cm 立方，10 mm 网格，探头直径 6 mm（`H` 在插值场上做球平均）
- 反演默认 2 region log K，不是粗网格 6³ / 逐格 K
- 三维 p/S 是 F(m_post) 重建；产品尺子是自洽正演（F(m_true) → 反演 → 贴回 F），不是场 Dice 对 CMG
- `reservoir harness` 已删除
