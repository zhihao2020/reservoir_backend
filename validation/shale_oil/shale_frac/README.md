# 页岩油反演（水平井 + 裂缝条带）

本软件不做 GEM 组分闪蒸。页岩油在正交网格上先做成：

- 一口水平井（多段射孔）
- 垂直高渗条带 = 水力裂缝
- 条带周围略高渗 = 改造区
- 基质超低渗
- **衰竭**（无注水），封闭边界

反演走同一条 `invert_rock`：无注水井且多段生产完井时自动开 **缝条带 θ**（基质 / 缝 / SRV / 半长 / 缝距 / 相位），压力主导同化。不是注采通道软管，也不是 GEM。

```bash
python validation/shale_oil/shale_frac/run_validate.py
```

现场 30 cm 箱子可按同一几何填高渗条带。CMG 离线尺子已建成 5 套 IMEX 衰竭类比，见 [../cmg_shale_suite](../cmg_shale_suite/README.md)。
