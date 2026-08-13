# 路线图

## 当前

实现 `references/软件要求.txt` 四场 MVP：网格、压力、饱和度、k/φ。

已接入多时刻形态发现：`run_time_series` → `infer_shape_indicator` → `refine_mesh_by_indicator` → 再重建。

验证路径：

- 合成通道孪生：`pipeline.build_channel_twin` + `tests/test_shape_discovery.py`
- CMG IMEX 起伏通道：`validation/cmg_channel_3d/`（`*DTOP` 山脊）
- CMG IMEX 断层通道：`validation/cmg_fault_3d/`（`*FAULT` + `*TRANSI` 狗腿通道）
- 实验室 30 cm 山形层理：`validation/lab_box_30cm/`（平 DTOP，k 沿 `z_horizon` 铺；模具取出后全砂）
- 页岩油合成孪生：`shale_oil/validation/shale_frac/`（水平井 + 裂缝条带衰竭）
- 页岩油 IMEX 类比尺子：`shale_oil/validation/cmg_s1_hw5frac` … `cmg_s5_shutin`（5 工况，离线，非 GEM）

## 排除

- 角点网格、LGR、NNC
- 黑油 PVT、工业井控
- OPM Flow / MRST 运行时
- REST / 前端 / 产品 UDP
- 完整历史拟合 / EnKF

## 已完成的正确性升级

- 井点压力 **矩阵 Dirichlet**（`cell_dirichlet` in TPFA）
- k 场 **数组先验 + 固定点迭代**（跨时刻与加密后映射）
- φ **双时刻连续/物质平衡代理**（`φ ≈ -div(u)/(∂Sw/∂t)`）

## 已完成的数据/集成升级

- CSV 多时刻井/边界传感器（`sensor_io` + CLI `--mode series|discovery`）
- 轻量 ES-MDA log-k 集成（`run_esmda_permeability` + CLI `--mode esmda`）
- `assimilate_k=True` 接到自动反演（通量指标先验 + ES-MDA）；跨工况验收：`validation/inversion_generality/`

## 后续可选（未排期）

- EnKF 时序滤波 / 相关局部化加强
- CMG .out 自动导出 wells CSV
- 面传导倍率 / 断层进求解器
- 饱和度观测进入 ES-MDA 数据向量
