# 项目状态

主线：30 cm 页岩油实验数字孪生。V1 产品 Case 是 `examples/lab_v1/`（组分 DPDP FIM + 面注采 + \(\theta=(\log C_f,\log T_{mf})\) + ES-MDA）。历史窗反演后冻结 \(\theta\)，再正演整段井控。\(k_m,\varphi_m,\varphi_f\)、PVT 固定。**M2 是 CMG-GEM 尺子**，不是用户入口。online / UDP / 30³ ensemble 冻结到 M3。`examples/lab/lab_cf.yaml` 是粗网格夹具。黑油 IMPES/FIM 已删除。

| 状态 | 含义 |
|------|------|
| **已验证** | 有实现与针对性测试，且测试测的是规格要求的行为 |
| **MVP** | 可用，假设写在 `docs/model_assumptions.md` |
| **不做** | P0 明确排除 |

## 能力表

| 能力 | 状态 | 入口 | 证据 |
|------|------|------|------|
| 300 mm / 10 mm → 30³，体积 0.027 m³ | 已验证 | `CartesianGrid.uniform` | `tests/grid/test_grid_lab.py` |
| 控制 / 观测分离 | 已验证 | `ControlSeries` / `ObservationSeries` | `tests/observation/test_observation_operator.py` |
| 非格点观测算子 | 已验证 | `ObservationOperator` | `tests/observation/test_observation_operator.py` |
| 黑油 IMPES / 顺序隐式 / 黑油 FIM | **已删除** | — | 产品正演是组分 DPDP FIM |
| 线性高斯 LM | 已验证 | `inverse.lm.run_lm` | `tests/inverse/test_lm_linear.py` |
| 冻结 θ 的正演 | MVP | `DigitalTwin.forward_from_posterior` | `tests/inverse/test_forecast.py` |
| CLI validate/simulate/invert/forecast/apply | MVP | `reservoir` | `tests/cli/test_cli.py` |
| 实验室 apply（历史反演 → 组分正演） | 已验证 | `reservoir apply` | `tests/cli/test_apply.py` |
| 测点 CSV（SI / 分钟·kPa、hold-out、无 --demo） | 已验证 | `io.case` / `apply` | `tests/cli/test_apply.py`、`tests/io/test_case_csv.py` |
| 点估计场 \(F(\hat\theta)\) | MVP | `DigitalTwin.reconstruct` | `tests/inverse/test_reconstruct_uq.py` |
| CSV 控制/观测 IO | 已验证 | `io.case` | `tests/io/test_case_csv.py` |
| 任意深度柱面测点 | 已验证 | `column_sensors` | `tests/observation/test_observation_operator.py` |
| 自洽 \(\theta=(C_f,T_{mf})\) 收回 | MVP | `make_lab_v1_face_twin` | `tests/inverse/test_log_cf_tmf.py` |
| 组分 EXAMPLE 孪生（等温气–油，C1–nC10） | MVP | `solver.fi_comp`、`eos/`、`comp/` | `tests/cases/test_comp_twin.py`；`examples/compositional/comp_example.yaml`。定流量井 \(p_{\mathrm{wf}}\) 进 \(H\)。2-region LM invert（数据 nRMSE 下降、对比度方向对）。不是济阳 GEM。单孔 FIM Jacobian：局部闪蒸差分 + 解析 TPFA；线性解 SuperLU/ILU-GMRES/CPR（`tests/solver/test_comp_analytic_jac.py`） |
| 组分 immiscible 水相 | MVP | `CompSpec.has_water` | `tests/cases/test_comp_water.py`、`examples/compositional/comp_example_water.yaml`。水进 \(F\) 和 \(H\)（\(S_w\)+率井 BHP）；2-region LM invert，对比度方向对。水不进 PR |
| 公开 PR 牌加载 | 已验证 | `io.eos_load.load_eos_card` | `tests/physics/test_eos_load.py`；fixture 抄 OPM `1D_COMP` 数字。缺文件拒绝 |
| 统一 invert run report | MVP | `twin.run_report`、`cli.reporting` | `invert.json` + `residuals.csv` |
| check83 十二问验收 | MVP | `twin.acceptance` | `check83.json`；`docs/check83_acceptance.md`；`tests/twin/test_check83_report.py` |
| LM 后验小 ensemble Ne=8 | MVP | `inverse.post_ensemble` | `k_mean.npy` / `k_std.npy`；`tests/inverse/test_post_ensemble.py` |
| Forecast 时间外推尺子 | MVP | `synthetic.make_forecast_split_case` | `tests/inverse/test_forecast.py` |
| Scalar \(C_f\) log 参数化 | 已验证 | `LogConductivityParameterization` | `tests/inverse/test_log_conductivity.py` |
| 联合 \(\log C_f,\log\beta_{mf}\) | MVP | `LogCfTmfParameterization` + 分层 ES-MDA | **M1a PASS**。**M1b 主 Case B PASS**（Cf 0.89% / Tmf 0.69%）；T2 与 seed 稳健性未过。**M1c FAIL（已接受）**：实验室可行方案 0 个 \(D_{C_f,5\%}>2\)。不要再调 ES-MDA。 |
| ES-MDA（log \(C_f\)） | 已验证 | `inverse.esmda`、`twin.history_match` | `tests/inverse/test_esmda.py`、`test_esmda_cf.py`。线性高斯收回；合成无噪声 \(C_f\) 向真值靠近；后验 P05/P50/P95 |
| Parameter EnKF（在线一步） | MVP | `inverse.parameter_enkf` | `tests/inverse/test_parameter_enkf.py`。**M3 才解冻**；当前 M2 不是这条路 |
| CMG-GEM 交叉验证流水线 | MVP | `twin.cmg_benchmark`、`examples/lab_v1/cmg_gem/` | **M2a–d PASS**（Case B + 四真值）。通用 `--case` 入口。`physical_3d` 15³ 单孔 7 组分：当前 `sanwei_co2.out` 重打包 hidden（时刻 0 / 0.01 / 0.14 / 1 / 3 d，无 0.0001 d 幽灵帧）。t=8.64 s \(F_\mathrm{ours}(k_\mathrm{GEM})\) 压力 RMSE **1.6 Pa**（GEM 在 0.01 d 全场 span ~1.5 kPa；旧 pack 的 0.45 MPa「采井漏斗」是坏图）。**不是 M2a PASS**。15³ 正演在 ~433 s 切步 underflow，864/12096 s 未对齐。`tests/twin/test_cmg_benchmark.py` |
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
- 逐格 \(K\)、coarse-field、缝长/SRV、济阳矿场吞吐、黑油 IMPES/FIM（已删除）。PINN、MPFA、动态 AMR、热尚未做。产品 `apply`：历史窗反演 \(\theta=(\log C_f,\log T_{mf})\)，再组分 DPDP 正演。
- 旧 `pipeline/` 产品路径（已删除）

## 实验室默认（2026-08 重构）

- 30 cm 立方，10 mm 网格，探头直径 6 mm（`H` 在插值场上做球平均）
- V1 产品 Case：`examples/lab_v1/`（组分 DPDP FIM + 面注采 + \(\theta=(\log C_f,\log T_{mf})\) + ES-MDA）。`lab_cf.yaml` 仅粗网格夹具
- 产品 invert 默认 ES-MDA。`case_dev.yaml` 可用 `algorithm: auto`（先 LM）。\(k_m,\varphi_m,\varphi_f\)、PVT 固定
- 三维 p/S 是 \(F(\hat\theta)\) 重建。M2 尺子是 CMG-GEM 交叉验证，不是用户入口
- `reservoir harness` 已删除
- 30 cm 产品开发计划（活文档）：`docs/lab_product.qmd`

## physical_3d 尺子：已测 / 未关

**已测（2026-09-07）**

- GEM `.out` 官方：t=0.01 d 压力 span ~1.5 kPa（max 在注入井 8,8,11）；t=3 d span ~1.6 kPa；Sg=0；INJ ~1e-7 m³/day，PROD1–3 流量 0。
- 当前 `parse_gem_out_maps` 重打包 hidden：时刻 0 / 864 / 12096 / 86400 / 259200 s。旧 pack 的 8.64 s + 0.45 MPa「采井漏斗」是坏图，不是物理。
- `F_ours(k_GEM)` vs 插值 GEM，t=8.64 s：压力 RMSE **1.6 Pa**，本模型全场 50.000000 MPa。图：`results/lab_v1/cmg_gem_physical_3d_compare/`（gitignored）。**不是 M2a PASS。**
- 解析回归：`test_parse_physical_3d_gem_out_if_present`、`test_parse_gem_out_ignores_timestep_table_times` 过。

**未关（交给后续精度轮）**

1. 15³ 组分 FIM 约 433 s `TimeStepUnderflow`，对不上 GEM 第一个报告时刻 864 s（0.01 d），更对不上 0.14 / 1 / 3 d。
2. GEM `*GEOMECH`（`*NOCOUPERM`）在 F 外；864 s 的 ~1.5 kPa 鼓包本模型没有。
3. 黏度：GEM `*VISCOR *HZYT`，我们是 LBC / 常数 μ。
4. `tests/solver/test_comp_analytic_jac.py` Jacobian 误差约 1（已知）。
5. `tests/comp/test_dual_d1234.py` D2 underflow（已知，产品路径外）。
6. nrmse_p 在 GEM span 只有几十 Pa 时会被印出噪声放大；看 RMSE_Pa / `nrmse_p_sigma`。
7. 流量一旦离开 50/50 近零区，要再对 Peaceman WI vs GEM `*GEOMETRY *K 0.003 0.34`。

约束：不重跑 GEM、不调 ES-MDA、不改 50/50 井控去造漏斗、力学不进 F、不宣称 M2a PASS。
