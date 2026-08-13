# 验证

## 主测试

```bash
pytest tests/test_pipeline_mesh.py tests/test_pipeline_fields.py tests/test_pipeline_e2e_cli.py tests/test_shape_discovery.py tests/test_sensor_io_esmda.py -q
```

可选保留的单元测试：`tests/test_core_*.py`、`tests/test_pressure_solver_*.py`、`tests/test_lab_horizon.py`、`tests/test_probe_split.py` 等核心求解器测试。

页岩油裂缝孪生：

```bash
python validation/shale_frac/run_validate.py
```

跨工况反演（同一旋钮，验证探点）：

```bash
python validation/inversion_generality/run_eval.py --methods point_first --n 12
```

## 形态发现 / CMG

```bash
# 合成：起伏通道 / 断层狗腿通道 / 30 cm 实验室层理
python black_oil/validation/cmg_channel_3d/run_imex_and_validate.py --synthetic
python black_oil/validation/cmg_fault_3d/run_imex_and_validate.py --synthetic
python validation/lab_box_30cm/run_validate.py --synthetic --n 15

# IMEX（需本机 CMG）
python black_oil/validation/cmg_channel_3d/run_imex_and_validate.py --execute
python black_oil/validation/cmg_fault_3d/run_imex_and_validate.py --execute
python black_oil/validation/cmg_fault_3d/run_imex_and_validate.py --from-out black_oil/validation/cmg_fault_3d/mxspr006_fault.out
```

CMG 用于提供**已知高渗通道 / 断层**的三维正演；本仓库算法从井点传感器反推形态指标，**不宣称与 IMEX 数值等价**。

## 原则

- 井点传感器匹配优先
- 饱和度闭合 sw+so+sg=1
- 不宣称与 OPM/CMG 等价
- 路径检查：`python scripts/check_doc_code_consistency.py`（活跃文档）
