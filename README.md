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
python -m reservoir_backend.pipeline.run --config config/sensor_series_case.yaml --mode series --output results/series
python -m reservoir_backend.pipeline.run --config config/sensor_series_case.yaml --mode esmda --output results/esmda
pytest tests/test_pipeline_mesh.py tests/test_pipeline_fields.py tests/test_pipeline_e2e_cli.py tests/test_shape_discovery.py tests/test_sensor_io_esmda.py -q
```

```python
from reservoir_backend.pipeline import (
    AxisAlignedBounds,
    WellPoint,
    SensorSample,
    BoundaryConditions,
    build_mesh,
    run_time_slice,
    build_channel_twin,
    run_shape_discovery,
    mask_overlap,
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

# multi-time shape discovery (no geometric mountain input)
twin = build_channel_twin(n_times=4)
disc = run_shape_discovery(twin.mesh, twin.samples)
print(mask_overlap(disc.active_mask, twin.true_channel_mask))
```

## 两套软件 / 两篇论文

| 目录 | 内容 |
|------|------|
| [black_oil/](black_oil/README.md) | 黑油水驱反演 + 论文 A + IMEX 尺子 |
| [shale_oil/](shale_oil/README.md) | 页岩油/裂缝反演 + 论文 B（独立主张） |

内核库 `reservoir_backend/` 供黑油主线使用；页岩油将换参数化与正演，不要混用 mxspr006 海水驱当页岩尺子。

## 文档

| 文档 | 说明 |
|------|------|
| [STATUS.md](STATUS.md) | 能力与证据 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 结构 |
| [docs/API_AND_DATA_CONTRACT.md](docs/API_AND_DATA_CONTRACT.md) | 契约 |
| [docs/VALIDATION.md](docs/VALIDATION.md) | 测试 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 限制与排除项 |
| [black_oil/README.md](black_oil/README.md) | 黑油论文 A 与 IMEX 验证 |
| [shale_oil/README.md](shale_oil/README.md) | 页岩论文 B 与裂缝孪生 |

## 合规

`references/upstream` 仅为只读参考（git submodule），**禁止 import 上游代码**。
