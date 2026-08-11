# 文档地图

本页是 `docs/` 目录的导航索引。按阅读顺序与主题分类，避免在 redirect stub 页面间反复跳转。

## 必读（中文）

| 文档 | 一句话说明 |
|------|------------|
| [../README.md](../README.md) | 项目入口：安装、快速开始、能力概览 |
| [../STATUS.md](../STATUS.md) | 模块成熟度唯一权威表 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 仓库结构、数据流、模块关系 |
| [API_AND_DATA_CONTRACT.md](API_AND_DATA_CONTRACT.md) | CLI、YAML 配置、实验数据与结果契约 |
| [VALIDATION.md](VALIDATION.md) | 测试分层、基准命令、报告索引与文档一致性检查 |
| [ROADMAP.md](ROADMAP.md) | 当前限制、近期优先级与未来排除项 |

包源码均在 `reservoir_backend/` 下（含 `solver/`、`results/`）；根目录 `results/` 仅为运行时产物。文档路径漂移用 `python scripts/check_doc_code_consistency.py` 检查，语义检索用本地 QMD 集合 `reservoir-docs` / `reservoir-code`。

## 专题（英文技术深读）

### 数值方法

| 文档 | 一句话说明 |
|------|------------|
| [numerical_methods.md](numerical_methods.md) | 网格、反演、TPFA 压力、输运、三相公式与代码锚点 |
| [numerical_accuracy.md](numerical_accuracy.md) | → 重定向至 [VALIDATION.md](VALIDATION.md) |
| [pressure_solver_validation.md](pressure_solver_validation.md) | → 重定向 stub，压力求解验证细节见 numerical_methods |
| [pressure_solver_enhancement.md](pressure_solver_enhancement.md) | 压力求解增强报告说明 |
| [saturation_transport_validation.md](saturation_transport_validation.md) | → 重定向 stub |
| [saturation_transport_enhancement.md](saturation_transport_enhancement.md) | 输运增强报告说明 |
| [capillary_gravity_validation.md](capillary_gravity_validation.md) | → 重定向 stub |
| [three_phase_validation.md](three_phase_validation.md) | → 重定向 stub |
| [impes_sequential_loop.md](impes_sequential_loop.md) | IMPES 顺序耦合循环设计 |

### 数据与接口

| 文档 | 一句话说明 |
|------|------------|
| [experimental_data_pipeline.md](experimental_data_pipeline.md) | 实验 CSV/JSON/NPZ 读取与 QC 管线 |
| [data_schema.md](data_schema.md) | 实验数据字段模式定义 |
| [data_contract.md](data_contract.md) | → 重定向至 [API_AND_DATA_CONTRACT.md](API_AND_DATA_CONTRACT.md) |
| [case_configuration.md](case_configuration.md) | → 重定向 stub，配置细节见 API 契约 |
| [cli_usage.md](cli_usage.md) | → 重定向 stub，CLI 见 API 契约 |
| [frontend_field_contract.md](frontend_field_contract.md) | → 重定向 stub，字段契约见 API 契约 |
| [interface_contract.md](interface_contract.md) | → 重定向 stub |
| 现场数据摄入 | 井表、生产/压力历史（见 `tests/test_field_data_ingestion.py`、`accuracy_reports/field_data_ingestion_summary.*`） |

### 融合与跨尺度

| 文档 | 一句话说明 |
|------|------------|
| [fusion_synthetic_twin.md](fusion_synthetic_twin.md) | 合成孪生体场融合与报告 |
| [parameter_fusion_validation.md](parameter_fusion_validation.md) | → 重定向 stub |
| [parameter_fusion_uncertainty.md](parameter_fusion_uncertainty.md) | 参数融合不确定性加权 |
| [cross_scale_validation.md](cross_scale_validation.md) | 跨尺度验证概述 |
| [cross_scale_cli.md](cross_scale_cli.md) | 跨尺度 YAML/JSON 运行器配置 |
| [cross_scale_upscaling_report.md](cross_scale_upscaling_report.md) | 升尺度报告层设计 |

### 结果与工作流

| 文档 | 一句话说明 |
|------|------------|
| [result_manifest.md](result_manifest.md) | → 重定向 stub，清单契约见 API 契约 |
| [result_export_pipeline.md](result_export_pipeline.md) | 结果导出管线 |
| [project_case_management.md](project_case_management.md) | 项目/案例/运行文件注册 |

### 基准与性能

| 文档 | 一句话说明 |
|------|------------|
| [benchmark_registry.md](benchmark_registry.md) | → 重定向 stub，策略见 VALIDATION |
| [benchmark_selection_policy.md](benchmark_selection_policy.md) | 基准选取政策 |
| [function_benchmark_matrix.md](function_benchmark_matrix.md) | → 重定向 stub，矩阵见 `specs/14_function_benchmark_matrix.md` |
| [open_source_benchmark_references.md](open_source_benchmark_references.md) | 开源参考改编说明 |
| [performance_baseline.md](performance_baseline.md) | 性能基线报告 |

### 未来范围（设计文档）

| 文档 | 一句话说明 |
|------|------------|
| [black_oil_pvt_architecture.md](black_oil_pvt_architecture.md) | 黑油/PVT 架构路线图（无求解器实现） |
| [api_frontend_integration_roadmap.md](api_frontend_integration_roadmap.md) | REST/前端集成路线图（无服务实现） |
| [limitations_and_roadmap.md](limitations_and_roadmap.md) | → 重定向至 [ROADMAP.md](ROADMAP.md) |

### 流程与发布

| 文档 | 一句话说明 |
|------|------------|
| [validation_and_profiling.md](validation_and_profiling.md) | → 重定向 stub |
| [release_checklist.md](release_checklist.md) | → 重定向 stub |
| [module_matrix.md](module_matrix.md) | → 兼容性 stub，状态见 STATUS.md |

## 设计规格（`specs/`）

| 文档 | 一句话说明 |
|------|------------|
| [../specs/10_requirement_traceability.md](../specs/10_requirement_traceability.md) | 需求与测试追溯矩阵 |
| [../specs/11_combined_capillary_gravity_design.md](../specs/11_combined_capillary_gravity_design.md) | 毛细管+重力组合输运设计 |
| [../specs/12_three_phase_flow_design.md](../specs/12_three_phase_flow_design.md) | 三相流设计 |
| [../specs/13_cross_scale_analysis_design.md](../specs/13_cross_scale_analysis_design.md) | 跨尺度分析架构决策 |
| [../specs/14_function_benchmark_matrix.md](../specs/14_function_benchmark_matrix.md) | 功能基准矩阵 |
| [../specs/09_cpp_migration_spec.md](../specs/09_cpp_migration_spec.md) | C++ 迁移规格（延后） |

## 架构决策记录（ADR）

| 文档 | 说明 |
|------|------|
| [adr/001-git-as-engineering-source.md](adr/001-git-as-engineering-source.md) | Git 为工程真源 |
| [adr/002-notion-as-project-dashboard.md](adr/002-notion-as-project-dashboard.md) | Notion 为项目看板 |
| [adr/003-python-backend-and-udp-interface.md](adr/003-python-backend-and-udp-interface.md) | Python 后端与 UDP 接口 |
| [adr/004-numerical-solver-scope.md](adr/004-numerical-solver-scope.md) | 数值求解器范围边界 |

## 归档（非权威）

`docs/archive/doc_consolidation/` 保留历史文档快照，**不作为当前状态来源**。如需追溯旧版矩阵或检查清单，可在此查阅，但以 [../STATUS.md](../STATUS.md) 与 [ROADMAP.md](ROADMAP.md) 为准。

## 生成的报告产物

基准与验证摘要位于 `accuracy_reports/`（约 50 个 JSON/MD 文件）。完整对照表见 [VALIDATION.md](VALIDATION.md)。

## 阅读建议

1. 新开发者：README → ARCHITECTURE → API_AND_DATA_CONTRACT → 跑 `demo_case.yaml --dry-run`
2. 数值验证：VALIDATION → numerical_methods → 对应 `*_benchmark_hardening.py` 测试
3. 了解边界：STATUS → ROADMAP → 相关 specs 设计文档
