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
- Gate 1：`DualRock`、`DualCompositionalState`（主变量 moles+pressure）
- Gate 2：组分 transfer（matrix→fracture > 0，上游 λ，反对称）
- Gate 3–4：D0–D4（单格、两格三种通量、\(\sigma=0\)、\(k_m^{\mathrm{intercell}}\to 0\)、\(4\times3\times2\) 井控质量守恒）
- Gate 5：\(m=\log(C_f/C_{\mathrm{ref}})\)；\(C_f\) 只改 DualRock 裂缝连续体；ES-MDA 走 `fi_comp_dual`
- ES-MDA 默认不再 clip innovation；观测 QC 在 smoother 之前过滤
- Parameter EnKF 拆成 forecast_parameters / analysis_parameters；`OnlineAssimilationWorkflow` + checkpoint/rollback
- 外围 UDP JSON 协议与跨尺度 nRMSE；不进入 solver
- DPDP restart 保存完整 DualCompositionalState（含 matrix moles）
- 7-color Cartesian Jacobian + sparse-vs-brute FD 测试
- Flash cache：全场残差复用两相 \(K\)；Jacobian FD 仍走 Wilson（否则 J 与 R 不一致）
- frozen-λ 快环接入 ports/controls 与 linearized \(c_t\)；饱和度 held
- Online slow loop：Parameter EnKF（上一 posterior ensemble），不再 `calibrate()`
- ProcessPool initializer：worker 只传 \(\theta\)
- Online EnKF：增量窗 \((t_{k-1},t_k]\)、QC、failed-member 替换；`TwinLoops.from_posterior`
- \(t>0\) 且缺 `moles_matrix` 时拒绝 lossless restart
- Jacobian 分计 `flash_main_s` / `flash_jacobian_s`；Newton 主路径改为 local thermo + TPFA/transfer 块装配（`dpdp_blocks.py`）；井源仍局部 FD

## 未完成

- 30³ 标准步（`dpdp_scale_gate`）约 85 s（原 223 s）；Jacobian 8.9 s 已低于 20–30 s 目标，线性求解约 65 s 仍是 60 s 总门槛的主因
- 真实实验 PVT / 电阻率声波反演尚未接入实测数据

## 规模（FastPR 后，见 `docs/bench/dpdp_scale_fastpr.json`）

| 网格 | 墙钟 | 说明 |
|------|------|------|
| 5³ | ~2.5 s | 标准短步 |
| 10³ | ~6.4 s | 已低于 60 s 门槛（原 coloring 约 757 s / 12.6 min） |
| 20³ | ~717 s | 5 个接受步 + 3 次拒绝，**不可与单步 30³ 直接比** |
| 30³ gate | **85 s** | 统一 `max_steps=1`；原 223 s。Jacobian 9 s，solve 65 s |
| 30³ 快环 reuse | ~0.093 s | 首次分解 ~28 s |

统一比较请跑 `python scripts/dpdp_scale_gate.py`。V1 apply：`examples/lab/lab_cf.yaml`。

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

## 下一阶段

10³ one-step \(<60\,\mathrm{s}\) 未达则不上 \(20^3\)。Flash cache 跨 Newton 仍未开（与 Wilson 残差不一致）。实验室 `apply` 仍是两区 log K + LM。
