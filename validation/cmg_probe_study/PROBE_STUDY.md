# CMG 虚拟测点学习曲线

从 IMEX `.out` 全场 p/S **虚拟抽样** exclusive 测点（不改 CMG 井网）。

| case | layout | N | n_p/n_s | ES-MDA | Sw rel L2 ↓ | ΔSw Dice ↑ | k_ch/k_out | p hold-out RMSE (Pa) |
|------|--------|---|---------|--------|-------------|------------|------------|----------------------|
| cmg_undulating_channel | wells_only | 0 | 0/0 | Y | 0.7803 | 0.192 | 1.000 | 1.2e+07 |
| cmg_undulating_channel | uniform | 4 | 2/2 | Y | 0.7359 | 0.352 | 1.517 | 7.2e+06 |
| cmg_undulating_channel | adaptive | 4 | 2/2 | Y | 0.8199 | 0.288 | 1.443 | 1.05e+07 |
| cmg_undulating_channel | uniform | 8 | 4/4 | Y | 0.5597 | 0.352 | 0.204 | 6.34e+06 |
| cmg_undulating_channel | adaptive | 8 | 4/4 | Y | 0.7797 | 0.400 | 1.354 | 8.26e+06 |
| cmg_undulating_channel | uniform | 12 | 6/6 | Y | 0.7593 | 0.240 | 0.180 | 4.16e+06 |
| cmg_undulating_channel | adaptive | 12 | 6/6 | Y | 0.6254 | 0.304 | 1.208 | 6.48e+06 |
| cmg_faulted_dogleg | wells_only | 0 | 0/0 | Y | 0.8214 | 0.383 | 0.999 | 8.86e+06 |
| cmg_faulted_dogleg | uniform | 4 | 2/2 | Y | 0.5978 | 0.506 | 0.018 | 7.35e+06 |
| cmg_faulted_dogleg | adaptive | 4 | 2/2 | Y | 0.7336 | 0.494 | 1.164 | 6.36e+06 |
| cmg_faulted_dogleg | uniform | 8 | 4/4 | Y | 0.6686 | 0.210 | 0.873 | 5.11e+06 |
| cmg_faulted_dogleg | adaptive | 8 | 4/4 | Y | 0.7171 | 0.469 | 1.219 | 6.54e+06 |
| cmg_faulted_dogleg | uniform | 12 | 6/6 | Y | 0.6755 | 0.210 | 0.381 | 5.19e+06 |
| cmg_faulted_dogleg | adaptive | 12 | 6/6 | Y | 0.8228 | 0.506 | 1.184 | 5.38e+06 |

## 读法

- **N↑ 后 Sw L2 下降或 Dice 上升** → 测点改善重建（反演未必严格单调）。
- **ES-MDA=Y**：先用井+压力测点软同化 log(k)，再点优先时序。
- **uniform / adaptive**：几何均匀 vs hybrid DOE；粗网格上均匀常更稳。
- **p hold-out**：未硬约束格点压力相对 CMG；越低越好。
- 井压硬约束误差应接近 0。

