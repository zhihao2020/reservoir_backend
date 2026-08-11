# CMG 虚拟测点学习曲线

从 IMEX `.out` 全场 p/S **虚拟抽样** exclusive 测点（不改 CMG 井网）。

| case | layout | N | n_p/n_s | ES-MDA | Sw rel L2 ↓ | ΔSw Dice ↑ | k_ch/k_out | p hold-out RMSE (Pa) |
|------|--------|---|---------|--------|-------------|------------|------------|----------------------|
| cmg_undulating_channel | wells_only | 0 | 0/0 | Y | 0.6382 | 0.384 | 0.824 | 1.38e+07 |
| cmg_undulating_channel | uniform | 8 | 4/4 | Y | 0.4857 | 0.400 | 7.881 | 9.62e+06 |
| cmg_undulating_channel | uniform | 12 | 6/6 | Y | 0.4535 | 0.544 | 9.677 | 5.28e+06 |
| cmg_faulted_dogleg | wells_only | 0 | 0/0 | Y | 0.8782 | 0.407 | 8.045 | 8.74e+06 |
| cmg_faulted_dogleg | uniform | 8 | 4/4 | Y | 0.5026 | 0.346 | 7.910 | 6.27e+06 |
| cmg_faulted_dogleg | uniform | 12 | 6/6 | Y | 0.5013 | 0.333 | 14.554 | 6.01e+06 |

## 读法

- **N↑ 后 Sw L2 下降或 Dice 上升** → 测点改善重建（反演未必严格单调）。
- **ES-MDA=Y**：先用井+压力测点软同化 log(k)，再点优先时序。
- **uniform / adaptive**：几何均匀 vs hybrid DOE；粗网格上均匀常更稳。
- **p hold-out**：未硬约束格点压力相对 CMG；越低越好。
- 井压硬约束误差应接近 0。

