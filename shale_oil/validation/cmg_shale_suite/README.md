# IMEX 页岩油类比尺子（S1–S5）

五套 **IMEX 单孔黑油衰竭** 算例：水平井 + 高渗裂缝条带 + SRV。  
克隆官方 `mxspr006` 的 PVT，去掉注水井。**不是 GEM 组分、不是双孔、不是吸附。**

| 编号 | 目录 | 工况 |
|------|------|------|
| S1 | [cmg_s1_hw5frac](../cmg_s1_hw5frac/) | 单水平井，5 条水力缝，衰竭 |
| S2 | [cmg_s2_hw9frac](../cmg_s2_hw9frac/) | 同井场，9 条更密缝 |
| S3 | [cmg_s3_twohw](../cmg_s3_twohw/) | 两口平行水平井，t=0 同时开（干扰） |
| S4 | [cmg_s4_parent_child](../cmg_s4_parent_child/) | 父井 HW1 先开；子井 HW2 约 1 年后开 |
| S5 | [cmg_s5_shutin](../cmg_s5_shutin/) | 同 S1，中期关井再开井 |

网格统一 `21×31×5` CART；基质 ~0.001 md，缝 8000 md，SRV 0.4 md。

```bash
python shale_oil/validation/cmg_shale_suite/build_shale_suite.py
python shale_oil/validation/cmg_shale_suite/run_imex.py --case all
python shale_oil/validation/cmg_shale_suite/smoke_parse.py
pytest tests/test_shale_cmg_suite.py -q
```

产品路径不调用 CMG。这些 `.out` 只当离线尺子。不要接到 `black_oil/validation/cmg_probe_study`。
