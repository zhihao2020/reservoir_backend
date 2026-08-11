# 架构

## 产品主线

```text
build_mesh → reconstruct_pressure → reconstruct_saturation → invert_rock_properties
```

包路径：`reservoir_backend/pipeline/`。

## 目录

| 路径 | 职责 |
|------|------|
| `reservoir_backend/core/` | Grid3D、Field3D、井、单位 |
| `reservoir_backend/solver/` | TPFA 压力、传导率、速度、输运辅助 |
| `reservoir_backend/pipeline/` | 四场产品 API |
| `reservoir_backend/data/` | 实验数据读取（可选） |
| `reservoir_backend/field_data/` | 井/生产表读取（可选） |
| `reservoir_backend/results/` | 结果清单/导出 |
| `reservoir_backend/io/` | 配置与结构化 deck 子集 |
| `config/sensor_case.yaml` | 传感器案例 |
| `references/` | 需求与只读上游 submodule |

已删除：cross_scale、workflow、project、performance、schedule、history_matching、api、inversion、fusion、simulation、cli。

## 数值依赖

压力重建可调用 `solver.pressure_solver.solve_steady_state_pressure_3d`（需 SciPy），井点压力在求解后钉扎到传感器值。
