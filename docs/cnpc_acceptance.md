# 中石油实验室数字孪生 交付验收

合同对象：300 mm 立方试块多相渗流数字孪生反演后端。
不是页岩油全场模拟器，不是 CMG/IMEX 替代品。

## 1. 交付物

- 源代码与可安装 Python 包（`reservoir` CLI）
- 30 cm 算例：`examples/lab/lab_30cm.yaml`、`examples/lab/lab_apply.yaml`
- 测点 CSV 模板：`examples/lab/observations_template.csv`
- 已知通道填砂算例：`examples/lab/lab_channel.yaml`
- 本文件：验收口径（签字用）

## 2. 验收通过（必须同时满足）

在交付机执行：

```text
python -m pip install -e .
reservoir validate examples/lab/lab_30cm.yaml
reservoir apply examples/lab/lab_apply.yaml --demo --output results/lab
pytest -q
```

自洽演示（`--demo`，观测由本正演 F 生成）须达到：

- `n_theta = 2`（两区渗透率，不是逐格 K）
- `forward_match_nrmse` = nRMSE(F(m_post), F(m_true)) 小于 0.30
- 后验对比度与真值同量级（真值 10 时，后验约 8–12）
- 报告的 K 等于 expand(theta_mean)，不得另掺格子场
- 三维 p、Sw、So 来自 F(m_post)，不是独立反演的饱和度场
- pytest 反演相关用例通过

## 3. 有实测点时怎么交

1. 按 `examples/lab/observations_template.csv` 填时间、传感器、压力(Pa)或饱和度(0–1)、sigma
2. 在 YAML 的 `experiment.observations` 指向该 CSV
3. 去掉 `--demo`，执行 `reservoir apply examples/lab/lab_apply.yaml --output results/lab`
4. 分区与岩样一致：层状用 `region_axis: z`；已知通道用 `lab_channel.yaml` 的 `region_map`
5. 探头直径默认 6 mm；sigma 用该探头重复性，不要沿用模板里的 2 kPa / 0.04

## 4. 明确不验收

- 三维 p/S/K 与 CMG/IMEX 全场逐格相等或 Dice
- 逐单元格渗透率反演
- 页岩油裂缝参数（xf、Fcd、缝数）全场反演
- 用 HPO/堆叠把 K 调到看起来像 CMG
- 溶气 Rs、Stone 三相、神经网络代替正演

对 CMG 虚拟实验是后续课题：把 F 升级到与 IMEX 同原理（隐式输运、同一套 SWT/井模型），而不是在错误的 F 上加密测点或堆叠。

## 5. 甲方能拿到什么图/数

`results/lab/` 下：

- `apply.json`：theta、identifiability、assimilate/hold-out/forecast、质量守恒
- `k.npy`：拟合渗透率
- `pressure.npy`、`sw.npy`、`so.npy`：F(m_hat) 重建场
- `figures/posterior_fields_xz.png`：剖面图

## 6. 交付前还差（内部）

- 本工作区改动尚未提交到 origin/main
- 尚未用真实岩样 CSV 跑通 apply（无 `--demo`）
- 30 cm 全网格 `--self-check` 尚未作为签字算例跑完
- 探头 sigma 尚未换成该仪器重复性
