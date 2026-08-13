# 架构

## 产品主线

```text
软件要求 1–4（每时刻 t）——**点优先**默认路径：

```text
1 build_mesh(边界, 井/测点, dx/dy/dz)
2 仅压力硬点(井压+observer_p) → 全网格 p
   （observer_s 无测压，从 p 场取值）
3 仅饱和度硬点(井S+observer_s) → 全网格 sw,so,sg
   （observer_p 无测 S，从 S 场取值）
4 在各硬点用互补后的 (p,S) 估算点 k,φ
   → 自动空间插值（IDW / 普通克里金 / 堆叠，LOO-CV 选择；log-k）
   → 全网格 k,φ
```

入口：`run_time_slice` → `run_point_first_slice`（`mode=grid_invert` 为旧全网反演）。

多时刻高精度：`run_time_series(assimilate_k=True)` 走**自动堆叠反演**（点优先 / 通量增强等多个物理成员，按留出测点加权）。无通道/层理模板。6 维通道管仅 `k_prior="channel_tube"`。

空间插值规则内置（**无 YAML/CLI method 开关**）：`N_MIN_KRIGING=8`，`CV_MARGIN=0.05`；压力场仍走 TPFA，不走本模块。

测点推荐（可选 API）：`recommend_probes` / `place_uniform_probes`（均匀或自适应 hybrid DOE）；验证见 `validation/cmg_probe_study`（CMG 虚拟测点扫 N，不改 .dat）。

CMG 仅作**非均质**正演对照，不进产品内核。
```

## 保留的包

| 路径 | 职责 |
|------|------|
| `reservoir_backend/core/` | Grid3D、Field3D、井、单位 |
| `reservoir_backend/solver/` | TPFA 压力、传导率、速度、输运辅助 |
| `reservoir_backend/pipeline/` | 四场产品 API |
| `reservoir_backend/io/` | 配置加载、结构化 deck 子集、场 NPZ |
| `config/sensor_case.yaml` | 传感器案例 |
| `references/` | 需求与只读上游 submodule |
| `tests/` | pipeline + 核心求解器测试 |
| `scripts/check_doc_code_consistency.py` | 文档路径检查 |

## 已删除（无用）

`harness`、`benchmarks`、`examples`、`accuracy_reports`、`validation_reports`、`profiling_reports`、`requirements`、  
`data`、`field_data`、`results`（包）、`utils`、跨尺度/融合/反演/工作流等旧包。

运行时输出目录：`results/`（仅产物）。
