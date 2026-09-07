# 示例：实验室 300 mm 试块怎么跑

用户入口是 `reservoir apply`。一次历史反演，再组分正演整段井控。

| 目录 | 用途 |
|------|------|
| `lab_v1/` | **产品 Case**：30 cm 组分 DPDP FIM + 面注采 + \(\theta=(\log C_f,\log T_{mf})\) |
| `lab/` | 粗网格夹具 `lab_cf.yaml` |
| `compositional/` | 可选单孔组分孪生 |
| `lab_v1/cmg_gem/` | CMG-GEM 交叉验证尺子，不是用户入口 |

配置合同：`case.yaml` + `pvt.yaml` + `wells.yaml` + `controls.csv` + 观测 CSV。

```bash
reservoir apply examples/lab_v1/case_dev.yaml --demo --output results/lab_v1_demo
reservoir apply examples/lab/lab_cf.yaml --demo --output results/lab_cf
```

`--demo` 不是正式用法。没有探头读数，\(C_f,T_{mf}\) 反不出来。
