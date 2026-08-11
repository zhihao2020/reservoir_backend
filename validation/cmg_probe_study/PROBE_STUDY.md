# CMG 虚拟测点学习曲线

从 IMEX `.out` 全场 p/S **虚拟抽样** exclusive 测点（不改 CMG 井网）。

| case | layout | N | n_p/n_s | ES-MDA | Sw rel L2 ↓ | ΔSw Dice ↑ | k_ch/k_out | p hold-out RMSE (Pa) |
|------|--------|---|---------|--------|-------------|------------|------------|----------------------|
| cmg_undulating_channel | wells_only | 0 | 0/0 | Y | 0.6281 | 0.368 | 8.471 | 1.2e+07 |
| cmg_undulating_channel | uniform | 8 | 4/4 | Y | 0.3565 | 0.560 | 3.290 | 6.96e+06 |
| cmg_undulating_channel | uniform | 12 | 6/6 | Y | 0.4052 | 0.656 | 1.113 | 4.15e+06 |
| cmg_faulted_dogleg | wells_only | 0 | 0/0 | Y | 0.7369 | 0.506 | 10.897 | 8.84e+06 |
| cmg_faulted_dogleg | uniform | 8 | 4/4 | Y | 0.4845 | 0.432 | 3.201 | 5.08e+06 |
| cmg_faulted_dogleg | uniform | 12 | 6/6 | Y | 0.3964 | 0.457 | 1.208 | 5.26e+06 |

## 读法

- **N↑ 后 Sw L2 下降或 Dice 上升** → 测点改善重建（反演未必严格单调）。
- **ES-MDA=Y**：先用井+压力测点软同化 log(k)，再点优先时序。
- **uniform / adaptive**：几何均匀 vs hybrid DOE；粗网格上均匀常更稳。
- **p hold-out**：未硬约束格点压力相对 CMG；越低越好。
- 井压硬约束误差应接近 0。

