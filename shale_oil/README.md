# 页岩油 / 致密油反演（软件 + 论文 B）

与黑油水驱 **不是同一套软件主张**。页岩油是衰竭 + 水平井 + 高渗裂缝条带；基质超低渗，没有注水通道软管。

## CMG 尺子（IMEX 类比，离线）

五套工况已建成并跑通 IMEX `Normal Termination`。克隆官方 `mxspr006` 的黑油 PVT，去掉注水井；**单孔、无吸附、不是 GEM**。产品路径不调用 CMG。

| 编号 | 目录 | 工况 | 井 | 缝面 I | 缝块 |
|------|------|------|----|--------|------|
| S1 | [validation/cmg_s1_hw5frac](validation/cmg_s1_hw5frac/) | 单水平井，5 条水力缝，衰竭 | HW1 | 4,8,11,14,18 | 255 |
| S2 | [validation/cmg_s2_hw9frac](validation/cmg_s2_hw9frac/) | 同井场，9 条更密缝 | HW1 | 3,5,…,19 | 459 |
| S3 | [validation/cmg_s3_twohw](validation/cmg_s3_twohw/) | 两口平行水平井，t=0 同时开 | HW1+HW2 | 同 S1 | 345 |
| S4 | [validation/cmg_s4_parent_child](validation/cmg_s4_parent_child/) | 父井先开；子井约 1 年后开 | HW1；HW2@365 d | 同 S1 | 345 |
| S5 | [validation/cmg_s5_shutin](validation/cmg_s5_shutin/) | 同 S1，中期关井再开井 | HW1 | 同 S1 | 255 |

统一网格 `21×31×5` CART（3255 块）；基质 ~0.001 md，缝 8000 md，SRV 0.4 md；`*MIN BHP 1500 psi`。  
末时刻 `.out` 抽检：PRES/SW 全有限、非常数；缝平均压力比远场基质低 **约 965–1160 psi**；ΔSw 仅 ~0.002（衰竭而非水驱，尺子主信号是压降）。

生成 / 重跑 / 抽检：

```bash
python shale_oil/validation/cmg_shale_suite/build_shale_suite.py
python shale_oil/validation/cmg_shale_suite/run_imex.py --case all
python shale_oil/validation/cmg_shale_suite/smoke_parse.py
pytest tests/test_shale_cmg_suite.py -q
```

不要接到 `black_oil/validation/cmg_probe_study`，也不要用 mxspr006 海水驱冒充页岩。

## 合成孪生（无 CMG）

- 验证：[validation/shale_frac](validation/shale_frac/README.md)
- 测试：仓库根 `tests/test_shale_fracture.py`
- 内核仍暂时复用 `reservoir_backend` 的自动反演堆叠；**裂缝专用 θ / GEM 尺子尚未建立**

## 论文主张

见 [PAPER.md](PAPER.md)。

## 下一步（软件）

1. 独立参数化：缝长 / 导流 / SRV，而不是 inj–prod 6 维软管
2. 观测算子：多段水平井压力与产量（本套尺子已提供 PRES 场）
3. 可选：官方 `mxfrr023` 页岩气双孔 / PLNRFRAC，或 GEM 组分

```bash
python shale_oil/validation/shale_frac/run_validate.py
pytest tests/test_shale_fracture.py tests/test_shale_cmg_suite.py -q
```
