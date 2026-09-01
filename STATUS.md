# 项目状态

主线：30 cm 页岩油实验数字孪生。V1 产品 Case 是 `examples/lab_v1/`（30³ 组分 DPDP + 面注采 + \(\theta=(\log C_f,\log\beta_{mf})\) + ES-MDA + Parameter EnKF）。饱和度由上游给出 \(S,\sigma\)；原始电/磁/声反演不在核心范围。`examples/lab/lab_cf.yaml` 是粗网格开发夹具；`lab_apply.yaml` 是遗留两区水驱演示。

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
| 隐式输运（两相和三相默认，YAML `transport`） | 已验证 | `solver.transport.implicit_water` | `tests/physics/test_black_oil.py`、`tests/physics/test_three_phase.py` |
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
| 自洽两层 K 收回 | 已验证 | `make_two_layer_waterflood` | `tests/inverse/test_synthetic_twin.py` |
| 组分 EXAMPLE 孪生（等温气–油，C1–nC10） | MVP | `solver.fi_comp`、`eos/`、`comp/` | `tests/cases/test_comp_twin.py`；`examples/compositional/comp_example.yaml`。定流量井 \(p_{\mathrm{wf}}\) 进 \(H\)。2-region LM invert（数据 nRMSE 下降、对比度方向对）。不是济阳 GEM |
| 组分 immiscible 水相 | MVP | `CompSpec.has_water` | `tests/cases/test_comp_water.py`、`examples/compositional/comp_example_water.yaml`。水进 \(F\) 和 \(H\)（\(S_w\)+率井 BHP）；2-region LM invert，对比度方向对。水不进 PR |
| 公开 PR 牌加载 | 已验证 | `io.eos_load.load_eos_card` | `tests/physics/test_eos_load.py`；fixture 抄 OPM `1D_COMP` 数字。缺文件拒绝 |
| 统一 invert run report | MVP | `twin.run_report`、`cli.reporting` | `invert.json` + `residuals.csv` |
| check83 十二问验收 | MVP | `twin.acceptance` | `check83.json`；`docs/check83_acceptance.md`；`tests/twin/test_check83_report.py` |
| LM 后验小 ensemble Ne=8 | MVP | `inverse.post_ensemble` | `k_mean.npy` / `k_std.npy`；`tests/inverse/test_post_ensemble.py` |
| Forecast 时间外推尺子 | MVP | `synthetic.make_forecast_split_case` | `tests/inverse/test_forecast.py` |
| Scalar \(C_f\) log 参数化 | 已验证 | `LogConductivityParameterization` | `tests/inverse/test_log_conductivity.py` |
| 联合 \(\log C_f,\log\beta_{mf}\) | MVP | `LogCfTmfParameterization` + 分层 ES-MDA | **M1a PASS** tiny。**M1b Case B PASS**（seed=3）：Cf 0.89%、Tmf 0.69%、holdout 0.0069、\(D_{C_f}=3.78\)、fail_rate 0。Case A/C PASS。四真值 T1/T3/T4 PASS，T2 Cf 5.37%。Seed sweep 5 个：2/5 过 5% 门闩，fail_rate 全 0，多数后验未覆盖真值（过自信）。**M1c 未过**：仪器 2 kPa 时 \(D_{C_f}=0.09\sim0.14\)（加流量/加测点后仍 <2），5% Cf 在 60 s 立方体上不可辨识 |
| ES-MDA（log \(C_f\)） | 已验证 | `inverse.esmda`、`twin.history_match` | `tests/inverse/test_esmda.py`、`test_esmda_cf.py`。线性高斯收回；合成无噪声 \(C_f\) 向真值靠近；后验 P05/P50/P95 |
| Parameter EnKF（在线一步） | MVP | `inverse.parameter_enkf` | `tests/inverse/test_parameter_enkf.py`。α=1，不改状态场 |
| DualContinuumState / transfer / ForwardModel adapter | MVP | `domain.state`、`physics.transfer`、`solver.forward_adapter` | `tests/domain/test_dual_state.py`、`tests/solver/test_forward_adapter.py` |
| DPDP DualRock + 组分 transfer | 已验证 | `physics.dual_rock`、`physics.transfer.ComponentTransfer` | `tests/physics/test_dual_rock.py`、`test_component_transfer.py` |
| DPDP compositional FIM D0–D4 | 已验证 | `comp.dual_residual`、`solver.fi_comp_dual` | `tests/comp/test_dual_d0.py`、`test_dual_d1234.py`：守恒相对误差 < 1e-4 |
| Sparse DPDP Jacobian + context | 已验证 | `solver.dpdp_jacobian`、`solver.dpdp_context`、`solver.linear` | `tests/comp/test_dpdp_sparse.py`、`test_dpdp_scale.py`（5³） |
| Scalar \(C_f\) ES-MDA on DPDP | 已验证 | `LogConductivityParameterization`、`HistoryMatchWorkflow` | `tests/inverse/test_esmda_cf.py`；`m=\log(C_f/C_{\mathrm{ref}})` |
| Observation QC | 已验证 | `observation.qc` | `tests/observation/test_qc.py` |
| Online checkpoint / UDP | MVP | `twin.online`、`io.udp_api` | `tests/twin/test_online_checkpoint.py` |
| 面注采 `make_face_port` | 已验证 | `ports.flow.make_face_port` | `tests/ports/test_face_port.py` |
| V1 产品 Case `examples/lab_v1/` | MVP | `case.yaml` 30³ + `case_dev.yaml` | `tests/io/test_lab_v1.py` |
| 传感器 CSV `sensor_id,…,sigma` | 已验证 | `io.case._read_sensors_csv` | `tests/io/test_lab_v1.py` |
| 饱和度默认 bulk | 已验证 | `ObservationOperator` | `tests/observation/test_sensor_medium.py` |
| Innovation trigger + \(E_p\) | MVP | `twin.loops.TwinLoops` | `tests/twin/test_loops.py` |
| LinearSolveResult 诊断 + Schur CPR | MVP | `solver.linear` | `tests/solver/test_linear_result.py` |
| Lab Gate（面 BC，非 scale gate） | MVP | `scripts/lab_v1_gate.py` | `tests/scripts/test_lab_v1_gate_schema.py` |
| TwinRuntime / FieldStore | MVP | `runtime/` | `tests/runtime/test_runtime.py` |
| 实验数据集 `experiments/EXP001` | MVP | `runtime.replay` | `reservoir replay experiments/EXP001` |
| 真实 PVT YAML + 3–6 lumping | MVP | `examples/lab_v1/pvt.yaml`、`eos.pvt_ingest` | `tests/physics/test_realfluid_flash.py` |

## 排除

- 四场插值「反演」（已删除）
- 每 cell 独立反演 27k 个 K（非默认）
- Archie / EM / acoustic 通用反演（已删除）
- 逐格 \(K\)、coarse-field、缝长/SRV/基质渗透率反演、济阳矿场吞吐、IMEX 页岩 suite、黑油 CMG 尺子（已删除）。PINN、MPFA、动态 AMR、热尚未做。活油黑油仍是 \(R_s\) 表闪蒸。实验室 `apply` 默认两区 log K + LM；`log_conductivity` 走组分 DPDP + ES-MDA
- 旧 `pipeline/` 产品路径（已删除）

## 实验室默认（2026-08 重构）

- 30 cm 立方，10 mm 网格，探头直径 6 mm（`H` 在插值场上做球平均）
- V1 产品 Case：`examples/lab_v1/`（组分 DPDP + 面注采 + log \(C_f\) + ES-MDA）。`lab_cf.yaml` 仅粗网格夹具
- 反演默认标量 log \(C_f\)（`ensemble_size=12`）。遗留水驱演示才是 2 region log K
- 三维 p/S 是 F(m_post) 重建；产品尺子是自洽正演（F(m_true) → 反演 → 贴回 F），不是场 Dice 对 CMG
- `reservoir harness` 已删除
- 概念实验室 30 cm：`examples/lab/lab_concept.yaml` + `concept_probes.csv`（75 电阻率 + 16 新增 7.5 cm）。invert 对比 = \(F(m_{\mathrm{post}})\)/\(F(m_{\mathrm{true}})\) 场 nRMSE，不是 CMG。`tests/cases/test_lab_concept.py`。
- 30 cm 产品开发计划（活文档）：`docs/lab_product.qmd`
