# CMG 正演 vs 反演对比图

由 `plot_cmg_vs_inv.py` 生成。测点仅 wells + virtual exclusive probes。

| figure | case | N | notes |
|--------|------|---|-------|
| `channel_N8_uniform_cmg_vs_inv.png` | cmg_undulating_channel | 8 | SwL2=0.326, Dice=0.592, k=8.55 |
| `channel_N12_uniform_cmg_vs_inv.png` | cmg_undulating_channel | 12 | SwL2=0.424, Dice=0.592, k=6.86 |

## 读图

- **第1行**：末时刻含水饱和度 Sw（CMG / 反演 / 绝对误差）
- **第2行**：多时刻 |ΔSw| 足迹与重叠（绿=一致，红=仅CMG，蓝=仅反演）
- **第3行**：压力场 + 反演渗透率（白线为 CMG deck 真通道轮廓）
- 标记：▲注入井 ▼生产井 ○ exclusive 测点
