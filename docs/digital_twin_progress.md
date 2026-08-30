# V1 progress

## 已完成

- 删除 coarse-field / 缝长-SRV-\(k_m\) 反演 / 济阳矿场吞吐 / 黑油 CMG 尺子 / IMEX 页岩 suite
- invert / apply 不再写入相似准则
- DualContinuumState、LogConductivityParameterization、FractureConductivityModel、WarrenRootTransfer、FluidModel、ForwardModel adapter
- 现有 solver 通过 `TwinForwardAdapter` 接入 ForwardModel
- ES-MDA（Emerick 2013；Cholesky/SVD，禁止 `np.linalg.inv`；\(\sum 1/\alpha_k=1\)；失败 member 邻域替换）
- `HistoryMatchWorkflow`；YAML `parameterization: log_conductivity` 默认 `algorithm: esmda`
- 在线 Parameter EnKF 一步（α=1 + 小 random walk），不覆盖压力/饱和度场
- Synthetic Case A：无噪声 \(H(F(C_f^{\mathrm{true}}))\) 收回标量 \(C_f\)

## 未完成

- 完整基质–裂缝耦合时间步（当前用 `FractureConductivityModel` 把 \(C_f\) 映射到裂缝格子 \(k_f^{\mathrm{eff}}\)，基质 \(k_m\) 固定）
- 在线 checkpoint / rollback / UDP
- 实验室 `apply` 默认仍是两区 log K + LM（YAML `log_conductivity` 才走 ES-MDA）

## 接口变化

- `load_case` 拒绝 `inverse.truth_json`
- YAML `inverse.parameterization` 只接受 `region` / `contrast`
- `accept_demo` 不再含 `similarity`
- 删除 `twin.similarity`、`inverse.structure`（`--auto`）、`io.cmg_out`；历史窗拆分共用 `split_history_observations`

## 新增模块

见 `docs/digital_twin_repository_audit.md` 中 NEW 行。

## 已知问题

- 实验室产品路径仍是两区 log K + LM
- adapter 的 `step` 用 `simulate` 推到 \(t+\Delta t\)，内部仍可 substep
- `test_1d_dirichlet_linear_pressure`（`tests/physics/test_pressure_analytical.py`）在空 ports + face Dirichlet 上 TimeStepUnderflow；未改 IMPES

## 下一阶段

双重介质耦合时间步（Phase 2）。在线 checkpoint / UDP。
