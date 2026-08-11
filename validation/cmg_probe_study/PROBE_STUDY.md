# CMG 虚拟测点学习曲线

从 IMEX `.out` 全场 p/S **虚拟抽样** exclusive 测点（不改 CMG 井网）。

| case | layout | N | n_p/n_s | ES-MDA | Sw rel L2 ↓ | ΔSw Dice ↑ | k_ch/k_out | p hold-out RMSE (Pa) |
|------|--------|---|---------|--------|-------------|------------|------------|----------------------|
| cmg_undulating_channel | wells_only | 0 | 0/0 | Y | 0.7355 | 0.320 | 1.228 | 1.2e+07 |
| cmg_undulating_channel | uniform | 4 | 2/2 | Y | 0.6439 | 0.512 | 2.132 | 7.68e+06 |
| cmg_undulating_channel | adaptive | 4 | 2/2 | Y | 0.7582 | 0.480 | 0.707 | 9.46e+06 |
| cmg_undulating_channel | uniform | 8 | 4/4 | Y | 0.3373 | 0.608 | 0.900 | 6.57e+06 |
| cmg_undulating_channel | adaptive | 8 | 4/4 | Y | 0.7746 | 0.496 | 0.593 | 8.44e+06 |
| cmg_undulating_channel | uniform | 12 | 6/6 | Y | 0.4116 | 0.720 | 1.305 | 4.07e+06 |
| cmg_undulating_channel | adaptive | 12 | 6/6 | Y | 0.8103 | 0.576 | 1.453 | 7.97e+06 |
| cmg_faulted_dogleg | wells_only | 0 | 0/0 | Y | 0.7331 | 0.407 | 1.305 | 7.64e+06 |
| cmg_faulted_dogleg | uniform | 4 | 2/2 | Y | 0.5487 | 0.568 | 0.226 | 6.39e+06 |
| cmg_faulted_dogleg | adaptive | 4 | 2/2 | Y | 0.6767 | 0.568 | 0.483 | 5.18e+06 |
| cmg_faulted_dogleg | uniform | 8 | 4/4 | Y | 0.4623 | 0.395 | 1.094 | 5.12e+06 |
| cmg_faulted_dogleg | adaptive | 8 | 4/4 | Y | 0.7269 | 0.519 | 0.420 | 6.15e+06 |
| cmg_faulted_dogleg | uniform | 12 | 6/6 | Y | 0.3988 | 0.457 | 1.509 | 5.05e+06 |
| cmg_faulted_dogleg | adaptive | 12 | 6/6 | Y | 0.7714 | 0.543 | 0.713 | 5.07e+06 |

## 读法

- **N↑ 后 Sw L2 下降或 Dice 上升** → 测点改善重建（反演未必严格单调）。
- **ES-MDA=Y**：先用井+压力测点软同化 log(k)，再点优先时序。
- **uniform / adaptive**：几何均匀 vs hybrid DOE；粗网格上均匀常更稳。
- **p hold-out**：未硬约束格点压力相对 CMG；越低越好。
- 井压硬约束误差应接近 0。

