# API 与数据契约

本文汇总轻量 CLI、案例配置、实验数据、结果清单与前端字段契约。读完本文你将知道如何调用入口、编写 YAML 案例，以及数据字段的约束。

**下一步阅读：** [VALIDATION.md](VALIDATION.md)（如何验证）、[experimental_data_pipeline.md](experimental_data_pipeline.md)（实验数据管线细节）。

模块状态见 [../STATUS.md](../STATUS.md)。

## CLI 速查表

当前 CLI 表面有意保持精简，均为开发者工具，**非产品 API**。

| 命令 | 用途 | 典型输出 |
|------|------|----------|
| `python scripts/run_case.py --config config/demo_case.yaml --dry-run` | 校验 YAML 配置是否可加载 | 控制台配置摘要 |
| `python scripts/run_case.py --config config/demo_case.yaml` | 运行完整演示管线 | `results/` 下字段与报告 |
| `python -m reservoir_backend.simulation.impes_report` | 生成 IMPES 循环报告 | `accuracy_reports/impes_loop_summary.*` |
| `python -m reservoir_backend.fusion.synthetic_twin_report` | 生成合成孪生报告 | `accuracy_reports/fusion_synthetic_twin_summary.*` |
| `python -m reservoir_backend.performance.performance_report` | 性能基线报告 | `accuracy_reports/performance_baseline_summary.*` |
| `python -m reservoir_backend.project.case_report` | 项目/案例管理报告 | `accuracy_reports/project_case_management_summary.*` |
| `python -m reservoir_backend.cross_scale.runner` | 跨尺度基准运行 | `accuracy_reports/cross_scale_benchmark_summary.*` |
| `python -m reservoir_backend.cross_scale.upscaling_report` | 升尺度报告 | `accuracy_reports/cross_scale_upscaling_summary.*` |

## 案例配置（YAML）

案例文件位于 `config/`。典型结构（以 `demo_case.yaml` 为例）：

```yaml
case:
  case_id: demo_case
  output_dir: results
  mode: archie_only

grid:
  nx: 6
  ny: 5
  nz: 3
  dx: 1.0
  dy: 1.0
  dz: 1.0

rock:
  porosity: 0.2
  permeability_md: 100.0

fluid:
  mu_w: 1.0e-3
  mu_o: 5.0e-3

pressure:
  boundary_type: left_right_dirichlet
  left_pressure: 10.0
  right_pressure: 9.0
  pressure_unit: MPa

saturation:
  dt: 1000.0
  steps: 3
  max_cfl: 1.0
  use_capillary: false
  use_gravity: false
```

### 配置块说明

| 块 | 说明 |
|----|------|
| `case` | 案例 ID、输出目录、运行模式 |
| `grid` | 结构化网格尺寸与单元长度 |
| `rock` | 孔隙度、渗透率 |
| `fluid` | 油水黏度 |
| `archie` / `electromagnetic` / `acoustic` | 反演模型参数 |
| `pressure` | 边界类型与参考压力 |
| `saturation` | 时间步、CFL 限制、相渗与毛细管/重力开关 |
| `capillary` / `gravity` / `three_phase` | 扩展物理案例（见各 `config/*_case.yaml`） |

配置范围限于现有 Python 后端，**不包含**黑油 PVT、工业井控或前端工作流。

## 实验数据契约

数据管线将 CSV、JSON、NPZ 读入标准内部 `ExperimentalDataset`。

### 支持字段

| 字段名 | 中文说明 |
|--------|----------|
| `resistivity` | 电阻率 |
| `electromagnetic_response` | 电磁响应 |
| `acoustic_response` | 声学响应 |
| `pressure` | 压力 |
| `saturation` | 饱和度 |
| `porosity` | 孔隙度 |
| `permeability` | 渗透率 |
| `temperature` | 温度 |
| `time` | 时间 |
| `x`, `y`, `z` | 坐标 |
| `confidence` | 置信度 |
| `variance` | 方差 |
| `metadata` | 元数据 |
| `unit` | 单位 |
| `source_name` | 数据源名称 |

### 物理约束

| 检查项 | 规则 |
|--------|------|
| 孔隙度 | `[0, 1]` |
| 饱和度 | `[0, 1]` |
| 渗透率 | 大于 0 |
| 压力 | 有限值 |
| 电阻率 | 大于 0 |
| 置信度 | 存在时在 `[0, 1]` |
| 方差 | 存在时非负 |

单位归一化覆盖压力、渗透率、分数/百分比、时间、坐标与温度。QC 管线报告缺失单位、NaN/Inf、缺失值、重复时间或坐标、越界与离群标记。

样例文件见 `tests/fixtures/experimental_data/`。

## 结果清单契约

每条导出结果清单条目包含：

| 字段 | 说明 |
|------|------|
| `result_id` | 结果唯一标识 |
| `case_id` | 案例 ID |
| `run_id` | 运行 ID |
| `module` | 来源模块 |
| `result_type` | 结果类型 |
| `field_name` | 场名称 |
| `shape` | 数组形状 |
| `dtype` | 数据类型 |
| `unit` | 单位 |
| `path` | 文件路径 |
| `format` | 文件格式 |
| `created_at` | 创建时间 |
| `source_task` | 来源任务 |
| `source_report` | 来源报告 |
| `metadata` | 元数据 |
| `warnings` | 警告 |
| `limitations` | 限制说明 |

大数组应导出为 NPZ；CSV 导出仅含元数据与汇总行，不含完整 3D 场数据。

## 报告索引

报告索引可注册 JSON 与 Markdown 报告路径，例如：

- `accuracy_reports/experimental_data_qc_summary.json`
- `accuracy_reports/pressure_solver_benchmark_summary.json`
- `accuracy_reports/saturation_transport_benchmark_summary.json`
- `accuracy_reports/benchmark_registry_summary.json`
- `accuracy_reports/result_manifest_summary.json`

缺失路径应报告为警告，**不得伪造**。

## 前端字段契约

仓库包含面向未来前端或报告消费者的数据契约，定义：

- 压力场（pressure fields）
- 饱和度场（saturation fields）
- 融合场（fusion fields）
- 基准报告字段（benchmark report fields）
- QC 报告字段（QC report fields）
- 警告与错误字段
- 单位与形状约定

**仅为数据契约**，本仓库无前端实现、无 REST API、无数据库服务。详见 [frontend_field_contract.md](frontend_field_contract.md)。

## 限制

- 无商业数据管理平台。
- 无 Petrel 类工作流产品。
- 不通过本接口层重写求解器。
- 无黑油 PVT 契约。
- 归档历史文档可能与当前契约不一致，以本文与 [STATUS.md](../STATUS.md) 为准。

本工作流完全基于**文件**，无数据库服务。
