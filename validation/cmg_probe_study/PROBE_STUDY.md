# CMG 虚拟测点学习曲线

从 IMEX `.out` 全场 p/S **虚拟抽样** exclusive 测点（不改 CMG 井网）。

| case | layout | N | n_p/n_s | ES-MDA | Sw rel L2 ↓ | ΔSw Dice ↑ | k_ch/k_out | p hold-out RMSE (Pa) |
|------|--------|---|---------|--------|-------------|------------|------------|----------------------|
| cmg_undulating_channel | wells_only | 0 | 0/0 | Y | 0.6424 | 0.224 | 0.830 | 1.2e+07 |
| cmg_undulating_channel | uniform | 8 | 4/4 | Y | 0.3386 | 0.608 | 8.716 | 6.25e+06 |
| cmg_undulating_channel | uniform | 12 | 6/6 | Y | 0.4079 | 0.656 | 7.090 | 4.16e+06 |
| cmg_faulted_dogleg | wells_only | 0 | 0/0 | Y | 0.7348 | 0.494 | 7.689 | 8.85e+06 |
| cmg_faulted_dogleg | uniform | 8 | 4/4 | Y | 0.4864 | 0.395 | 8.978 | 5.19e+06 |
| cmg_faulted_dogleg | uniform | 12 | 6/6 | Y | 0.3972 | 0.457 | 10.426 | 5.27e+06 |

## 读法

- **N↑ 后 Sw L2 下降或 Dice 上升** → 测点改善重建（反演未必严格单调）。
- **ES-MDA=Y**：先用井+压力测点软同化 log(k)，再点优先时序。
- **uniform / adaptive**：几何均匀 vs hybrid DOE；粗网格上均匀常更稳。
- **p hold-out**：未硬约束格点压力相对 CMG；越低越好。
- 井压硬约束误差应接近 0。

