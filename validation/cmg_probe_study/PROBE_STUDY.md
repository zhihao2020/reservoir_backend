# CMG 虚拟测点学习曲线

从 IMEX `.out` 全场 p/S **虚拟抽样** exclusive 测点（不改 CMG 井网）。

| case | layout | N | n_p/n_s | ES-MDA | Sw rel L2 ↓ | ΔSw Dice ↑ | k_ch/k_out | p hold-out RMSE (Pa) |
|------|--------|---|---------|--------|-------------|------------|------------|----------------------|
| cmg_undulating_channel | wells_only | 0 | 0/0 | Y | 0.6640 | 0.304 | 1.016 | 1.23e+07 |
| cmg_undulating_channel | uniform | 4 | 2/2 | Y | 0.5891 | 0.464 | 1.462 | 7.41e+06 |
| cmg_undulating_channel | adaptive | 4 | 2/2 | Y | 0.6961 | 0.464 | 0.445 | 9.79e+06 |
| cmg_undulating_channel | uniform | 8 | 4/4 | Y | 0.3355 | 0.608 | 0.255 | 6.21e+06 |
| cmg_undulating_channel | adaptive | 8 | 4/4 | Y | 0.7463 | 0.496 | 0.293 | 9.14e+06 |
| cmg_undulating_channel | uniform | 12 | 6/6 | Y | 0.4042 | 0.528 | 0.499 | 4.34e+06 |
| cmg_undulating_channel | adaptive | 12 | 6/6 | Y | 0.7681 | 0.512 | 0.457 | 6.62e+06 |
| cmg_faulted_dogleg | wells_only | 0 | 0/0 | Y | 0.6950 | 0.481 | 1.087 | 7.97e+06 |
| cmg_faulted_dogleg | uniform | 4 | 2/2 | Y | 0.5220 | 0.519 | 0.067 | 7.27e+06 |
| cmg_faulted_dogleg | adaptive | 4 | 2/2 | Y | 0.6101 | 0.593 | 0.529 | 6.27e+06 |
| cmg_faulted_dogleg | uniform | 8 | 4/4 | Y | 0.4903 | 0.395 | 0.992 | 5.32e+06 |
| cmg_faulted_dogleg | adaptive | 8 | 4/4 | Y | 0.7123 | 0.519 | 0.497 | 6.36e+06 |
| cmg_faulted_dogleg | uniform | 12 | 6/6 | Y | 0.4240 | 0.432 | 0.816 | 4.88e+06 |
| cmg_faulted_dogleg | adaptive | 12 | 6/6 | Y | 0.7634 | 0.543 | 0.386 | 5.4e+06 |

## 读法

- **N↑ 后 Sw L2 下降或 Dice 上升** → 测点改善重建（反演未必严格单调）。
- **ES-MDA=Y**：先用井+压力测点软同化 log(k)，再点优先时序。
- **uniform / adaptive**：几何均匀 vs hybrid DOE；粗网格上均匀常更稳。
- **p hold-out**：未硬约束格点压力相对 CMG；越低越好。
- 井压硬约束误差应接近 0。

