# CMG 虚拟测点学习曲线

从 IMEX `.out` 全场 p/S **虚拟抽样** exclusive 测点（不改 CMG 井网）。

| case | layout | N | n_p/n_s | ES-MDA | Sw rel L2 ↓ | ΔSw Dice ↑ | k_ch/k_out | p hold-out RMSE (Pa) |
|------|--------|---|---------|--------|-------------|------------|------------|----------------------|
| cmg_undulating_channel | uniform | 8 | 4/4 | Y | 0.3251 | 0.656 | 8.644 | 4.09e+05 |
| cmg_undulating_channel | uniform | 12 | 6/6 | Y | 0.4563 | 0.624 | 8.068 | 1.46e+05 |
| cmg_faulted_dogleg | uniform | 8 | 4/4 | Y | 0.4950 | 0.494 | 3.770 | 1.31e+06 |
| cmg_faulted_dogleg | uniform | 12 | 6/6 | Y | 0.3659 | 0.580 | 7.744 | 1.11e+06 |

## 读法

- 反演是**同一条路径**（封闭油藏 + 6 维 θ ES-MDA + 走廊 1-D Sw），不按工况/N 切算法。
- **N↑ 后**压力 hold-out 应下降；Sw L2 / Dice / k 对比在断层算例上随 N 明显变好。
- 粗网格通道上 Sw L2 未必严格单调（多测点会抽到基质，IDW 可能抹形），但 p 与 k 仍随 N 变稳。
- **ES-MDA=Y**：井+测点软同化 θ，再点优先时序。CMG 只做对照。
- 井压硬约束误差应接近 0。

