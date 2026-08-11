# 验证

## 主测试

```bash
pytest tests/test_pipeline_mesh.py tests/test_pipeline_fields.py tests/test_pipeline_e2e_cli.py -q
```

可选保留的单元测试：`tests/test_core_*.py`、`tests/test_pressure_solver_*.py` 等核心求解器测试。

## 原则

- 井点传感器匹配优先
- 饱和度闭合 sw+so+sg=1
- 不宣称与 OPM/CMG 等价
- 路径检查：`python scripts/check_doc_code_consistency.py`（活跃文档）
