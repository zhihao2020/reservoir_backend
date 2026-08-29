# 实验室 30 cm 山形模具填砂孪生（L1）

模具取出后箱子**全是砂**、箱盖是平的。山只存在于层理：高渗砂沿 `z_horizon(x,y)` 铺。  
CMG 用**平 `*DTOP`** + 同一函数上色的 `*PERMI`，不用死格子、不用起伏顶面。

真值源：`reservoir_backend/pipeline/lab_horizon.py`（软件孪生与 CMG 共用）。

## 网格阶梯

| 用途 | n | 边长 |
|------|---|------|
| 合成 / 单测 | 15 | 20 mm |
| 反演验收 / 默认 CMG | 30 | 10 mm |
| CMG 真值 | 50 | 6 mm |

## 运行

```bash
# 导出 15/30/50 真值数组 + 填砂高程 CSV
python validation/black_oil/lab_box_30cm/export_truth.py

# 合成孪生（无需 CMG）
python validation/black_oil/lab_box_30cm/run_validate.py --synthetic --n 15

# 生成 IMEX 算例（默认 30^3；真值用 --n 50）
python validation/black_oil/lab_box_30cm/build_lab_case.py --n 30
```

`lab_box_30cm.dat` 从 `validation/cmg_channel_3d/mxspr006_channel.dat` 克隆后只改几何、PERM、井位与流量。PVT 仍是样例黑油，**不是**实验室流体相似。

## 反演试跑

```bash
python validation/black_oil/lab_box_30cm/run_inversion_eval.py --n 12 --ne 12 --na 3
```

报告：`inversion_eval_report.json`。现有 6 维井间软管先验对薄层理会偏胖，井点 p 可对上，层理形态 Dice 低。

## 验收

- 井点压力误差 ≈ 0
- 高 Sw 足迹落在高渗层上，不穿层走盒子对角线
- 形态发现 recall 软阈值（欠定）

矿场 `cmg_channel_3d` / `cmg_fault_3d` 仍作回归，不再加构造变体。页岩油裂缝条带是 L2，另开算例。
