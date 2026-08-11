# CMG 虚拟测点学习曲线

从 IMEX `.out` 全场 p/S **虚拟抽样** exclusive 测点（不改 CMG 井网）。

| case | layout | N | n_p/n_s | Sw rel L2 ↓ | ΔSw Dice ↑ | k_ch/k_out | p hold-out RMSE (Pa) |
|------|--------|---|---------|-------------|------------|------------|----------------------|
| cmg_undulating_channel | wells_only | 0 | 0/0 | 0.6006 | 0.464 | 1.000 | 1.2e+07 |
| cmg_undulating_channel | uniform | 4 | 2/2 | 0.7701 | 0.416 | 1.718 | 6.56e+06 |
| cmg_undulating_channel | adaptive | 4 | 2/2 | 0.9899 | 0.144 | 1.151 | 1.19e+07 |
| cmg_undulating_channel | uniform | 8 | 4/4 | 0.4895 | 0.416 | 0.005 | 5.82e+06 |
| cmg_undulating_channel | adaptive | 8 | 4/4 | 1.0170 | 0.160 | 1.805 | 1.27e+07 |
| cmg_undulating_channel | uniform | 16 | 8/8 | 1.0439 | 0.288 | 1.371 | 6.47e+06 |
| cmg_undulating_channel | adaptive | 16 | 8/8 | 1.0102 | 0.304 | 34.909 | 1.11e+07 |
| cmg_faulted_dogleg | wells_only | 0 | 0/0 | 0.8114 | 0.469 | 0.995 | 8.77e+06 |
| cmg_faulted_dogleg | uniform | 4 | 2/2 | 0.6749 | 0.444 | 0.000 | 6.87e+06 |
| cmg_faulted_dogleg | adaptive | 4 | 2/2 | 0.9573 | 0.383 | 1.220 | 6.78e+06 |
| cmg_faulted_dogleg | uniform | 8 | 4/4 | 0.5534 | 0.259 | 0.922 | 4.66e+06 |
| cmg_faulted_dogleg | adaptive | 8 | 4/4 | 0.9705 | 0.395 | 1.373 | 7.38e+06 |
| cmg_faulted_dogleg | uniform | 16 | 8/8 | 0.6864 | 0.420 | 0.025 | 4.09e+06 |
| cmg_faulted_dogleg | adaptive | 16 | 8/8 | 1.1011 | 0.235 | 2.170 | 5.69e+06 |

## 读法

- **N↑ 后 Sw L2 下降或 Dice 上升** → 更多虚拟测点改善重建。
- **adaptive vs uniform**：同 N 对比；非均质通道上 hybrid 应用 CMG 多时刻方差。
- 井压误差应接近 0（硬约束）。

