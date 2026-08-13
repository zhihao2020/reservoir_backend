# 页岩油 / 致密油反演（软件 + 论文 B）

与黑油水驱 **不是同一套软件主张**。页岩油是衰竭 + 水平井 + 高渗裂缝条带；基质超低渗，没有注水通道软管。

当前仓库里已有合成孪生入口：

- 验证：[validation/shale_frac](validation/shale_frac/README.md)
- 测试：仓库根 `tests/test_shale_fracture.py`
- 内核仍暂时复用 `reservoir_backend` 的自动反演堆叠；**裂缝专用 θ / GEM 尺子尚未建立**

## 论文主张

见 [PAPER.md](PAPER.md)。不要用 `black_oil/validation` 里的 mxspr006 海水驱算例冒充页岩。

## 下一步（软件）

1. 独立参数化：缝长 / 导流 / SRV，而不是 inj–prod 6 维软管
2. IMEX 衰竭或 GEM 组分尺子（nD 基质 + 裂缝）
3. 观测算子：多段水平井压力与产量，而不是密井网 Sw

```bash
python shale_oil/validation/shale_frac/run_validate.py
pytest tests/test_shale_fracture.py -q
```
