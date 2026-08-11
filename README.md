# Reservoir Backend

储层实验数据处理与结构化网格数值验证的 Python 后端原型。

## 项目定位

本仓库用于：

- 实验室或合成储层实验数据的可重复处理与质控；
- 结构化笛卡尔网格上的有限体积压力重建与饱和度输运验证；
- 参数融合、跨尺度分析与基准报告生成。

**本项目不是**商业储层模拟器、黑油模拟器或产品前端。模块成熟度以 [STATUS.md](STATUS.md) 为唯一权威来源。

## 快速开始

创建环境并以可编辑模式安装：

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e .
```

运行测试：

```bash
pytest -q
```

通过 CLI 做最小案例检查：

```bash
python scripts/run_case.py --config config/demo_case.yaml --dry-run
```

读取实验数据：

```python
from reservoir_backend.data.reader import read_experimental_data
from reservoir_backend.data.qc import run_qc_pipeline

dataset = read_experimental_data("tests/fixtures/experimental_data/valid_csv_core_fields.csv")
qc_report = run_qc_pipeline(dataset)
print(qc_report["success"])
```

生成报告（按需）：

```bash
python -m reservoir_backend.simulation.impes_report
python -m reservoir_backend.fusion.synthetic_twin_report
```

## 处理流程

```mermaid
flowchart LR
  input[实验数据或YAML配置]
  qc[数据质控与单位归一化]
  inv[饱和度反演]
  press[压力重建]
  flux[达西通量与速度]
  trans[饱和度输运]
  fuse[参数场融合]
  out[报告与结果清单]

  input --> qc --> inv --> press --> flux --> trans --> fuse --> out
```

跨尺度分析、IMPES 顺序耦合与工业案例工作流作为并行子系统运行，详见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 能力概览

详细状态与证据见 [STATUS.md](STATUS.md)。

| 领域 | 主要内容 | 状态来源 |
|------|----------|----------|
| 数据入口 | 实验 CSV/JSON/NPZ、现场井表与生产历史、QC 报告 | [STATUS.md](STATUS.md) |
| 反演 | Archie 与多信号饱和度反演 | [STATUS.md](STATUS.md) |
| 数值核心 | finite-volume 压力求解、达西通量、油水输运、毛细管/重力、简化三相 WOG | [STATUS.md](STATUS.md) |
| 耦合与融合 | IMPES 顺序循环、参数融合与不确定性、合成孪生体 | [STATUS.md](STATUS.md) |
| 跨尺度 | 相似性准则、尺度效应、实验室-油田曲线对比 | [STATUS.md](STATUS.md) |
| 工作流 | 项目/案例/运行注册、结果清单、工业案例 v0 | [STATUS.md](STATUS.md) |
| 延后范围 | 黑油/PVT 架构设计、REST/前端集成、UDP 服务 | [STATUS.md](STATUS.md) |

## 验证

主验证命令：

```bash
pytest -q
```

基准报告生成命令、报告路径与回归策略见 [docs/VALIDATION.md](docs/VALIDATION.md)。验证原则：**Function hardening first**，以 benchmark validation 报告为证据，而非对外部商业模拟器做等价声明。

## 文档导航

| 文档 | 说明 |
|------|------|
| [STATUS.md](STATUS.md) | 模块成熟度（唯一权威） |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 仓库结构、数据流与模块关系 |
| [docs/API_AND_DATA_CONTRACT.md](docs/API_AND_DATA_CONTRACT.md) | CLI、YAML 配置、实验数据与结果契约 |
| [docs/VALIDATION.md](docs/VALIDATION.md) | 测试分层、基准命令与报告索引 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 当前限制与未来范围 |
| [docs/README.md](docs/README.md) | 完整文档地图 |

实验数据样例位于 `tests/fixtures/experimental_data`。

<!-- doc-anchors:
CLI Usage,
finite-volume,
combined capillary + gravity transport,
pressure solver benchmark,
pressure solver enhancement,
saturation transport benchmark,
saturation transport enhancement,
saturation inversion benchmark,
capillary / gravity benchmark,
parameter fusion benchmark,
parameter fusion uncertainty,
benchmark registry,
performance baseline,
performance_baseline_summary,
IMPES,
synthetic twin,
result manifest,
frontend field contract,
project / case management,
project_case_management_summary,
lab-field validation,
curve-to-curve comparison,
similarity criteria,
scale-effect analysis,
cross-scale analysis design,
one backend with two first-level modules,
cross-scale implementation is not yet complete,
reservoir_backend.cross_scale.runner,
cross_scale_benchmark_summary,
reservoir_backend.cross_scale.upscaling_report,
cross_scale_upscaling_summary,
python benchmarks/three_phase_benchmark.py,
Three-phase WOG benchmark hardening,
black-oil simulator,
udp development is still deferred
-->
