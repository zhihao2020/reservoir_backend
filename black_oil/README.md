# 黑油水驱反演（软件 + 论文 A）

常规黑油 / 河道砂岩 **注水** 工况。运行时只用注采井 + exclusive 测点；CMG IMEX 只做离线尺子。

内核库仍是仓库根目录的 [`reservoir_backend/`](../reservoir_backend/)（四场重建 + 低维 θ ES-MDA）。本目录放 **论文 A 的验证、文档与实验入口**。

## 论文主张

稀疏测点下黑油水驱四场反演：封闭油藏、p/S 互斥观测、通道参数化 + 集合同化，与 IMEX 全场对照。

详见 [PAPER.md](PAPER.md)。

## 验证尺子（IMEX 海水驱）

| 算例 | 路径 | 角色 |
|------|------|------|
| 粗通道 7×7×5 | [validation/cmg_channel_3d](validation/cmg_channel_3d/README.md) | 冒烟 |
| 细通道 21×21×8 | [validation/cmg_channel_fine](validation/cmg_channel_fine/README.md) | **主尺子** |
| 断层狗腿 | [validation/cmg_fault_3d](validation/cmg_fault_3d/README.md) | 构造工况 |
| 五点井网 | [validation/cmg_fivespot](validation/cmg_fivespot/README.md) | 井网工况 |
| 测点扫 N | [validation/cmg_probe_study](validation/cmg_probe_study/README.md) | 学习曲线 |

```bash
# 从仓库根目录
python black_oil/validation/cmg_channel_fine/run_imex.py
python black_oil/validation/cmg_probe_study/run_probe_study.py --cases channel_fine --n-list 8,12,24 --layouts uniform
python black_oil/validation/cmg_probe_study/plot_cmg_vs_inv.py --cases channel --n-list 8,12
```

## 不是什么

- 不是页岩油（见 [`../shale_oil/`](../shale_oil/)）
- 运行时不调用 CMG
