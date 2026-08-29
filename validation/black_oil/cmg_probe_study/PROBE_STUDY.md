# CMG 虚拟测点学习曲线

从 IMEX `.out` 全场 p/S **虚拟抽样** exclusive 测点（不改 CMG 井网）。

| case | layout | N | n_p/n_s | ES-MDA | Sw rel L2 ↓ | ΔSw Dice ↑ | k_ch/k_out | p hold-out RMSE (Pa) |
|------|--------|---|---------|--------|-------------|------------|------------|----------------------|
| cmg_undulating_channel | uniform | 8 | 4/4 | Y | 0.3815 | 0.512 | 5.348 | 6.88e+05 |
| cmg_undulating_channel | uniform | 12 | 6/6 | Y | 0.3731 | 0.512 | 2.683 | 4.42e+05 |
| cmg_faulted_dogleg | uniform | 8 | 4/4 | Y | 0.4054 | 0.543 | 2.227 | 2.7e+06 |
| cmg_faulted_dogleg | uniform | 12 | 6/6 | Y | 0.4249 | 0.469 | 3.161 | 2.58e+06 |

## 读法

- **N↑ 后 Sw L2 下降或 Dice 上升** → 测点改善重建（反演未必严格单调）。
- **ES-MDA=Y**：`invert_rock`（指示先验 + ES-MDA 更新 log k，再锁 k 正演）。
- **uniform / adaptive**：几何均匀 vs hybrid DOE；粗网格上均匀常更稳。
- **p hold-out**：未硬约束格点压力相对 CMG；越低越好。
- 井压硬约束误差应接近 0。

