# 路线图

## 当前

实现 `references/软件要求.txt` 四场 MVP：网格、压力、饱和度、k/φ。

已接入多时刻形态发现：`run_time_series` → `infer_shape_indicator` → `refine_mesh_by_indicator` → 再重建。

验证路径：

- 合成通道孪生：`pipeline.build_channel_twin` + `tests/test_shape_discovery.py`
- CMG IMEX 起伏通道：`validation/cmg_channel_3d/`（`*DTOP` 山脊）
- CMG IMEX 断层通道：`validation/cmg_fault_3d/`（`*FAULT` + `*TRANSI` 狗腿通道）

## 排除

- 角点网格、LGR、NNC
- 黑油 PVT、工业井控
- OPM Flow / MRST 运行时
- REST / 前端 / 产品 UDP
- 完整历史拟合 / EnKF

## 后续可选（未排期）

- 井点压力参与矩阵钉扎（真 Dirichlet 组装）
- 两时刻物质平衡强化 φ 反演
- 传感器 CSV 批量时间序列驱动
- ES-MDA / EnKF 升级路径
- CMG .out 解析增强（完整井表 / SR3）
