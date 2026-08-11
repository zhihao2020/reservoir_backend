# 项目状态

## 状态分级

本文件是模块成熟度的**唯一维护来源**。

| 状态 | 含义 |
|------|------|
| **已验证** | 实现完备，具备针对性测试及基准/报告证据，可作为主线假设。 |
| **测试中** | 实现存在，但接口、示例或验证覆盖仍需扩展后方可视为主线能力。 |
| **开发中** | 实现进行中，或接口尚未稳定。 |
| **延后** | 超出当前 MVP 范围，有意推迟或仅记录未来方向。 |

## 模块状态表

### 已验证

| 领域 | 范围 | 证据 | 边界说明 |
|------|------|------|----------|
| 核心网格与场模型 | 结构化正交网格（均匀或轴向非均匀间距）、场容器、单位、井记录 | `tests/test_core_grid.py`、`tests/test_core_grid_tensor.py`、`tests/test_core_field.py`、`tests/test_units.py`、`tests/test_wells.py`、`tests/test_structured_deck.py` | 正交网格；无角点（corner-point）几何。 |
| 饱和度反演 | Archie 与多信号反演工具 | `tests/test_saturation_inversion_hardening.py`、`accuracy_reports/saturation_inversion_benchmark_summary.*` | 电磁/声学路径仍为经验模型。 |
| 压力重建 | TPFA 压力求解、井源项、边界工具、求解器统计 | `tests/test_pressure_solver_benchmark_hardening.py`、`tests/test_pressure_solver_enhancement.py`、`accuracy_reports/pressure_solver_*summary.*` | 无有限元求解器与工业井模型。 |
| 达西通量与速度 | 面通量、速度诊断、传导率辅助 | `tests/test_velocity.py`、`tests/test_transmissibility.py`、压力基准报告 | 仅结构化网格达西流。 |
| 油水饱和度输运 | 显式有限体积输运、CFL 诊断、可选 TVD/MUSCL | `tests/test_saturation_transport_benchmark_hardening.py`、`tests/test_saturation_transport_enhancement.py`、`accuracy_reports/saturation_transport_*summary.*` | 未实现全隐式模拟器。 |
| 毛细管与重力输运 | 毛细管压力/通量、重力通量、组合水相通量诊断 | `tests/test_capillary_gravity_benchmark_hardening.py`、毛细管/重力报告产物 | 仅显式诊断与基准案例。 |

### 测试中

| 领域 | 范围 | 证据 | 边界说明 |
|------|------|------|----------|
| 实验数据入口 | CSV/JSON/NPZ 读取、模式检查、QC、样例 | `tests/test_experimental_data_pipeline.py`、`tests/test_experimental_data_fixtures.py`、`accuracy_reports/experimental_data_qc_summary.*` | 需更多真实实验室数据集。 |
| 现场数据摄入 | 井表、生产历史、压力历史、调度 CSV、属性场 | `tests/test_field_data_ingestion.py`、`accuracy_reports/field_data_ingestion_summary.*` | 仅文件输入，无数据库服务。 |
| 多井调度 v0 | 调度元数据、产量/BHP 控制接口、开关状态、报告步 | `tests/test_well_schedule_model.py`、`accuracy_reports/well_schedule_model_summary.*` | 无 Peaceman 模型、复杂井筒网络或黑油井控。 |
| 简化 IMPES 循环 | 压力→通量→饱和度耦合的小规模合成水驱 | `tests/test_impes_loop.py`、`accuracy_reports/impes_loop_summary.*` | 仅合成示例，非完整储层模拟器。 |
| 简化三相 WOG | 相渗、相通量、输运检查、生产汇总 | `tests/test_three_phase_benchmark_hardening.py`、`accuracy_reports/three_phase_benchmark_summary.*` | 不可压缩 WOG，非黑油 PVT。 |
| 参数融合与不确定性 | 场融合、不确定性加权、轻量空间回退、合成孪生汇总 | `tests/test_parameter_fusion_benchmark_hardening.py`、`tests/test_parameter_fusion_uncertainty.py`、`tests/test_fusion_synthetic_twin.py` | 无历史拟合或集合同化工作流。 |
| 合成孪生历史拟合原型 | 已知真值 k/phi、生成观测、噪声与轻量更新基线 | `tests/test_synthetic_twin_history_matching.py`、`accuracy_reports/synthetic_twin_history_matching_summary.*` | 仅合成场景，非真实油田历史拟合或完整 EnKF/ES-MDA。 |
| 跨尺度分析 | 相似性准则、尺度效应、实验室-油田曲线验证、升尺度报告 | `tests/test_cross_scale_benchmark_cli.py`、`tests/test_cross_scale_upscaling_report.py`、跨尺度报告产物 | 仅报告层，无多尺度 FV 求解器。 |
| 结果与项目管理 | 结果清单、报告索引、项目/案例/运行注册 | `tests/test_result_export_contract.py`、`tests/test_project_case_management.py` | 基于文件的注册表，无数据库服务。 |
| 工业案例工作流 v0 | 配置→Project/Case/Run→IMPES→生产汇总与工程报告 | `tests/test_industrial_case_workflow.py`、`accuracy_reports/industrial_case_workflow_summary.*` | 仅合成结构化网格工作流。 |
| 基准注册与性能基线 | 报告聚合、基准索引、运行时与内存汇总 | `tests/test_benchmark_registry_hardening.py`、`tests/test_performance_baseline.py` | 注册表读取已有报告，非求解器。 |
| CLI 与案例配置 | 轻量脚本入口与 YAML 案例示例 | `tests/test_cli_run_case.py`、配置加载器测试 | CLI 表面有意保持精简。 |

### 延后

| 领域 | 范围 | 证据 | 边界说明 |
|------|------|------|----------|
| UDP、REST API 与前端集成 | 产品 API 与 UI 集成路线图 | `docs/api_frontend_integration_roadmap.md`、`tests/test_api_frontend_roadmap_docs.py`、前端字段契约文档 | 当前 MVP 无 REST 服务、UI 或 UDP 运行时。 |
| 黑油 / PVT 架构 | Bo/Bw/Bg、Rs/Rv、泡点、相行为、地面产量、井控、调度/重启/报告步设计 | `docs/black_oil_pvt_architecture.md`、`tests/test_black_oil_architecture_docs.py` | 仅设计文档，无黑油求解器或 PVT 表解析器。 |
| 完整 SPE 复现与 C++ 内核 | 高级模拟器与加速方向 | 路线图与限制文档 | 不在当前已验证 Python 后端范围内。 |
