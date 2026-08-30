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

## 未完成

- 实验室 `apply` 默认仍是两区 log K + LM（计划允许）
- 30³ 是 milestone（`scripts/dpdp_scale_bench.py --max-n 30`），默认 `pytest -m "not slow"` 不跑 ES-MDA/\(10^3\)

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

冻结 \(\lambda\) 快环求压（`solver/frozen_pressure.py`）。Flash 单相 bypass 仅全场残差启用。\(5^3/10^3\) 标 `slow`。
