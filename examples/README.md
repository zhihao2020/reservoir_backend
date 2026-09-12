# 示例：实验室 300 mm 试块怎么跑

正演：`python -m reservoir_backend run <case.yaml>`（YAML 带着 grid / wells / fluid / schedule）。
历史反演再正演整段井控：`reservoir apply`。

| 目录 | 用途 |
|------|------|
| `run/` | 最短正向入口：CMG 习惯字段 + `wells.file` 片段 |
| `lab_v1/` | **产品 Case**：30 cm 组分 DPDP FIM + 面注采 + \(\theta=(\log C_f,\log T_{mf})\) |
| `lab/` | 粗网格夹具 `lab_cf.yaml` |
| `compositional/` | 可选单孔组分孪生；`literature_tiny.yaml` / `literature_hnp_1inj4prod.yaml` 是公开 EOS 的 LITERATURE EXAMPLE（后者是 CO2 吞吐演示），不是济阳 GEM |
| `lab_v1/cmg_gem/` | CMG-GEM 交叉验证尺子，不是用户入口 |

配置合同：`case.yaml` + `pvt.yaml` + `wells.yaml` + `controls.csv` + 观测 CSV。

```bash
python -m reservoir_backend run examples/run/case.yaml
python -m reservoir_backend run examples/compositional/literature_tiny.yaml
python -m reservoir_backend run examples/compositional/literature_hnp_1inj4prod.yaml
python -m reservoir_backend run examples/lab/lab_cf.yaml
reservoir apply examples/lab_v1/case_dev.yaml --demo --output results/lab_v1_demo
reservoir apply examples/lab/lab_cf.yaml --demo --output results/lab_cf
```

`--demo` 不是正式用法。没有探头读数，\(C_f,T_{mf}\) 反不出来。
