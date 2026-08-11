# 验证与基准

本文说明测试结构、基准报告位置与回归策略。读完本文你将知道如何运行测试、生成报告，以及各报告产物对应哪个模块。

**下一步阅读：** [benchmark_registry.md](benchmark_registry.md)（基准注册表）、[../STATUS.md](../STATUS.md)（模块证据）。

## 测试分层

| 层级 | 内容 | 示例 |
|------|------|------|
| 单元测试 | 网格、场、单位、井、底层数值辅助 | `tests/test_core_grid.py` |
| 模块测试 | 压力、速度、输运、相渗、反演、融合、跨尺度 | `tests/test_velocity.py` |
| 基准加固测试 | 报告模式、数值合理性、非过度声明措辞检查 | `tests/test_*_benchmark_hardening.py` |
| 集成测试 | 数据摄入、结果导出、项目注册、性能报告、顺序模拟 | `tests/test_industrial_case_workflow.py` |

### 全量运行

```bash
pytest -q
```

### 单模块运行

```bash
pytest tests/test_pressure_solver_benchmark_hardening.py -q
pytest tests/test_saturation_transport_benchmark_hardening.py -q
pytest tests/test_result_export_contract.py -q
```

编辑单个子系统时用针对性测试；声称仓库级回归健康前，应运行 `pytest -q`。

## 常用基准命令

| 命令 | 输出报告 |
|------|----------|
| `python benchmarks/three_phase_benchmark.py` | `accuracy_reports/three_phase_benchmark_summary.*` |
| `python -m reservoir_backend.cross_scale.runner` | `accuracy_reports/cross_scale_benchmark_summary.*` |
| `python -m reservoir_backend.cross_scale.upscaling_report` | `accuracy_reports/cross_scale_upscaling_summary.*` |
| `python -m reservoir_backend.performance.performance_report` | `accuracy_reports/performance_baseline_summary.*` |
| `python -m reservoir_backend.simulation.impes_report` | `accuracy_reports/impes_loop_summary.*` |
| `python -m reservoir_backend.fusion.synthetic_twin_report` | `accuracy_reports/fusion_synthetic_twin_summary.*` |
| `python scripts/run_accuracy_benchmarks.py` | 多模块基准摘要 |

## 报告产物对照表

| 报告文件名模式 | 对应模块 | 生成方式 |
|----------------|----------|----------|
| `pressure_solver_*summary.*` | 压力重建 | 基准加固测试 / 增强报告运行器 |
| `saturation_transport_*summary.*` | 油水输运 | 基准加固测试 / 增强报告运行器 |
| `capillary_gravity_benchmark_summary.*` | 毛细管与重力 | 基准加固测试 |
| `three_phase_benchmark_summary.*` | 简化三相 WOG | `benchmarks/three_phase_benchmark.py` |
| `saturation_inversion_benchmark_summary.*` | 饱和度反演 | 反演加固测试 |
| `parameter_fusion_*summary.*` | 参数融合 | 融合基准与不确定性测试 |
| `fusion_synthetic_twin_summary.*` | 合成孪生 | `synthetic_twin_report` 模块 |
| `cross_scale_benchmark_summary.*` | 跨尺度分析 | `cross_scale.runner` |
| `cross_scale_upscaling_summary.*` | 升尺度报告 | `cross_scale.upscaling_report` |
| `impes_loop_summary.*` | IMPES 顺序循环 | `impes_report` 模块 |
| `benchmark_registry_summary.*` | 基准注册表 | 注册表加固测试 |
| `performance_baseline_summary.*` | 性能基线 | `performance_report` 模块 |
| `project_case_management_summary.*` | 项目/案例管理 | `case_report` 模块 |
| `industrial_case_workflow_summary.*` | 工业案例工作流 | 集成测试 |
| `experimental_data_qc_summary.*` | 实验数据 QC | 数据管线测试 |
| `field_data_ingestion_summary.*` | 现场数据摄入 | 现场数据测试 |
| `result_manifest_summary.*` | 结果清单 | 结果导出契约测试 |
| `api_frontend_integration_roadmap_summary.*` | API/前端路线图 | 路线图文档测试 |

报告存储位置：

- `accuracy_reports/` — 基准、增强、融合、跨尺度、性能、项目/案例与模拟报告运行器生成的 JSON 与 Markdown 摘要。
- `validation_reports/` — 可选的本地验证输出目录（检出中可能不存在）。

## 基准注册表

基准注册表读取已有摘要文件并索引：基准名称、模块关联、验证类别、参考类型、关键指标、报告路径、限制与非过度声明检查。

**不重新运行物理案例**，也不替代模块级报告。详见 [benchmark_registry.md](benchmark_registry.md)。

## 参考数据政策

`references/` 下的开源参考材料用作改编元数据或方法上下文，**非运行时依赖**，也不构成与 OPM Flow、MRST、SPE 算例或商业模拟器的等价声明。

## 回归策略

1. 编辑单模块 → 运行对应 `tests/test_*.py`。
2. 合并或发布前 → 运行 `pytest -q`。
3. 基准措辞变更 → 运行相关 `*_benchmark_hardening.py` 测试。
4. 仅文档清理且未重跑测试时，应在变更说明中明确标注。
5. 改包路径或文档锚点后 → 运行 `python scripts/check_doc_code_consistency.py`，并用 QMD 检索活跃文档（见下）。

验证原则：**Function hardening first** — 以模块级 benchmark validation 报告为证据，workflow 设计在契约确认之后推进。

## 文档与代码一致性

路径级硬检查（排除 `docs/archive/`）：

```bash
python scripts/check_doc_code_consistency.py
```

语义检索（本地 [QMD](https://www.npmjs.com/package/@tobilu/qmd) 集合，开发机已索引时）：

```bash
qmd update
qmd search "reservoir_backend/solver" -c reservoir-docs -n 10
qmd search "results 包迁移" -c reservoir-docs -n 5
```

`STATUS.md` 仍是模块成熟度唯一权威表；本文只约定如何验证文档路径与报告产物未漂移。
