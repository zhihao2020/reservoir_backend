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
| MRST 离散（迎风 \(\lambda\)、重力、\(k_z\)、TRANSI、SWT） | 已验证 | `discretization.tpfa`、`TableTwoPhase` | 五点/断层 \(F(K)\) p RMSE 21–23 psi，Sw 0.018–0.030 |
| MRST 隐式输运（后向 Euler + Newton） | 已验证 | `solver.transport.implicit_water` | `tests/test_black_oil.py`；CMG 尺子默认开 |
| 质量守恒报告 | MVP | `MassBalance`（地面水体积） | `tests/test_mass_balance.py` |
| 毛管模型（显式选择） | 已验证 | `BrooksCorey` / `NoCapillary` | `tests/test_capillary.py` |
| 线性高斯 ES-MDA | 已验证 | `inverse.esmda.run_esmda` | `tests/test_esmda_linear.py` |
| Synthetic \(H(F(m_{true}))\) + hold-out | MVP | `validation.synthetic` | `tests/test_synthetic_twin.py` |
| 冻结 m 的 forecast | MVP | `DigitalTwin.forecast` | `tests/test_forecast.py` |
| CLI validate/simulate/invert/forecast/synthetic | MVP | `reservoir` | `tests/test_cli.py` |
| 三相不混溶 IMPES | MVP | `CoreyThreePhase` | `tests/test_three_phase.py` |
| 后验 p/S/K 分位数与标准差 | MVP | `DigitalTwin.reconstruct` | `tests/test_reconstruct_uq.py` |
| CSV 控制/观测 IO | 已验证 | `io.case` | `tests/test_case_csv.py` |
| 任意深度柱面测点 | 已验证 | `column_sensors` | `tests/test_observation_operator.py` |
| 反演尺子 A/B（自洽 vs CMG 观测） | MVP | `cmg_lab_layers/run_invert_eval.py` | `invert_eval_report.json` |
| 多 CMG harness（探针/日记/回溯） | MVP | `reservoir harness` | `tests/test_cmg_harness_*.py`；lab+五点+断层+通道 |
| 反演预设 / 时限 / hold-out 排行 | MVP | `calibrate_auto`、`inverse.presets` | `tests/test_portfolio.py` |
| ES / ES-MDA / ES-MDA-RS + hold-out 混合 | MVP | `inverse.algorithms`、`greedy_holdout_blend` | `tests/test_esmda_linear.py`、`test_portfolio.py` |
| Ensemble 并行正演 | MVP | `inverse.parallel.map_members` | `tests/test_parallel.py` |
| 限时 HPO（搜算法旋钮） | MVP | `inverse.hpo.run_hpo` | `tests/test_hpo.py` |
| 自洽两层 K 收回 | 已验证 | `make_two_layer_waterflood` | `tests/test_synthetic_twin.py` |

## 排除

- 四场插值「反演」（已删除）
- 每 cell 独立反演 27k 个 K（非默认）
- Archie / EM / acoustic 通用反演（已删除）
- 溶气/放气 \(R_s(p)\) 自由气、EnKF、MPFA、动态 AMR、PINN
- 旧 `pipeline/` 产品路径（已删除）
