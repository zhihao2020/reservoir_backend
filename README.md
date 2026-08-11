# Reservoir Backend

传感器驱动的储层 **四场重建** Python 内核：网格 → 压力场 → 饱和度场 → 物性场（k、φ）。

需求说明见 [references/软件要求.txt](references/软件要求.txt)。

## 做什么

| 步骤 | 输入 | 输出 |
|------|------|------|
| 网格 | 边界坐标、井坐标、dx/dy/dz | 单元序号与中心坐标 |
| 压力 | 井点压力、边界压力/流量、k 先验 | 全网格压力 p |
| 饱和度 | 井点油气水饱和度 | 全网格 sw, so, sg |
| 物性 | p、饱和度、流量尺度 + 渗流关系 | 全网格 k、φ |

**不是**商业模拟器、黑油引擎或前端产品。

## 安装

```bash
python -m pip install -e ".[dev]"
# 或
python -m pip install -r requirements.txt
```

## 快速开始

```bash
python -m reservoir_backend.pipeline.run --config config/sensor_case.yaml --output results/sensor_run
pytest tests/test_pipeline_mesh.py tests/test_pipeline_fields.py tests/test_pipeline_e2e_cli.py -q
```

```python
from reservoir_backend.pipeline import (
    AxisAlignedBounds,
    WellPoint,
    SensorSample,
    BoundaryConditions,
    build_mesh,
    run_time_slice,
)

mesh = build_mesh(
    AxisAlignedBounds(0, 100, 0, 80, 0, 30),
    dx=10, dy=10, dz=10,
    wells=[WellPoint("INJ", 10, 40, 15), WellPoint("PROD", 90, 40, 15)],
)
sample = SensorSample(
    time=0.0,
    well_pressure={"INJ": 1.2e7, "PROD": 1.0e7},
    well_saturation={"INJ": (0.8, 0.2, 0.0), "PROD": (0.25, 0.75, 0.0)},
    boundary=BoundaryConditions(pressure={"left": 1.2e7, "right": 1.0e7}),
)
fields = run_time_slice(mesh, sample)
```

## 文档

| 文档 | 说明 |
|------|------|
| [STATUS.md](STATUS.md) | 能力与证据 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 结构 |
| [docs/API_AND_DATA_CONTRACT.md](docs/API_AND_DATA_CONTRACT.md) | 契约 |
| [docs/VALIDATION.md](docs/VALIDATION.md) | 测试 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 限制与排除项 |

## 合规

`references/upstream` 仅为只读参考（git submodule），**禁止 import 上游代码**。
