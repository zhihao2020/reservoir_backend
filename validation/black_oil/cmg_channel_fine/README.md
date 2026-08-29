# CMG 加密通道尺子（主对照）

从 `cmg_channel_3d/mxspr006_channel.dat` 克隆。**只做验证尺子**，不进反演核。

| 项 | 值 |
|----|-----|
| 网格 | 21×21×8 VARI（3528 块） |
| 基质 | ~50 md 对数正态杂音 |
| 通道 | 不规则宽度，k≈1200–2800 md |
| 井 | INJ (1,1) / PROD (21,21) 多层射孔 |
| 流体 | 继承 mxspr006 海水驱 |

```bash
python validation/black_oil/cmg_channel_fine/build_channel_fine.py
python validation/black_oil/cmg_channel_fine/run_imex.py
python validation/black_oil/cmg_probe_study/run_probe_study.py --cases channel_fine --n-list 0,8,12,24 --layouts uniform
```
