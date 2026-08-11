# CMG 虚拟测点学习曲线

从 IMEX `.out` 全场 p/S **虚拟抽样** exclusive 测点（不改 CMG 井网）。

| case | layout | N | n_p/n_s | ES-MDA | Sw rel L2 ↓ | ΔSw Dice ↑ | k_ch/k_out | p hold-out RMSE (Pa) |
|------|--------|---|---------|--------|-------------|------------|------------|----------------------|
| cmg_undulating_channel | wells_only | 0 | 0/0 | Y | 0.7030 | 0.256 | 1.024 | 1.35e+07 |
| cmg_undulating_channel | uniform | 4 | 2/2 | Y | 0.5665 | 0.448 | 0.866 | 7.88e+06 |
| cmg_undulating_channel | adaptive | 4 | 2/2 | Y | 0.6846 | 0.368 | 0.084 | 1.03e+07 |
| cmg_undulating_channel | uniform | 8 | 4/4 | Y | 0.3571 | 0.608 | 0.553 | 8.23e+06 |
| cmg_undulating_channel | adaptive | 8 | 4/4 | Y | 0.6380 | 0.560 | 0.722 | 1.11e+07 |
| cmg_undulating_channel | uniform | 12 | 6/6 | Y | 0.4196 | 0.528 | 0.478 | 4.37e+06 |
| cmg_undulating_channel | adaptive | 12 | 6/6 | Y | 0.6633 | 0.576 | 0.553 | 5.95e+06 |
| cmg_faulted_dogleg | wells_only | 0 | 0/0 | Y | 0.6858 | 0.432 | 1.110 | 9.1e+06 |
| cmg_faulted_dogleg | uniform | 4 | 2/2 | Y | 0.4754 | 0.543 | 0.135 | 8.51e+06 |
| cmg_faulted_dogleg | adaptive | 4 | 2/2 | Y | 0.6590 | 0.506 | 0.181 | 8.06e+06 |
| cmg_faulted_dogleg | uniform | 8 | 4/4 | Y | 0.5149 | 0.333 | 1.180 | 5.63e+06 |
| cmg_faulted_dogleg | adaptive | 8 | 4/4 | Y | 0.6284 | 0.481 | 0.234 | 5.94e+06 |
| cmg_faulted_dogleg | uniform | 12 | 6/6 | Y | 0.4649 | 0.407 | 0.995 | 5.03e+06 |
| cmg_faulted_dogleg | adaptive | 12 | 6/6 | Y | 0.7413 | 0.407 | 0.336 | 5.55e+06 |

## 读法

- **N↑ 后 Sw L2 下降或 Dice 上升** → 测点改善重建（反演未必严格单调）。
- **ES-MDA=Y**：先用井+压力测点软同化 log(k)，再点优先时序。
- **uniform / adaptive**：几何均匀 vs hybrid DOE；粗网格上均匀常更稳。
- **p hold-out**：未硬约束格点压力相对 CMG；越低越好。
- 井压硬约束误差应接近 0。

