# 论文 B：页岩油缝网 / 水平井反演

## 主张

非常规油藏流动由裂缝与改造区主导，可辨识参数和黑油河道水驱不同，因此单独成文。

## 与论文 A 的差别（必须写清）

| | 论文 A 黑油 | 本文 |
|--|-------------|------|
| 正演 | IMEX 黑油注水 | 衰竭 + 裂缝（或 GEM） |
| 参数 | 通道 θ | 缝网 / SRV（6 维 frac θ） |
| 观测 | 注采井 + p/S 探针 | 水平井段压力（ΔSw≈0，不进 misfit） |
| 反演 | LM | LM（不恢复 ES-MDA） |
| 尺子 | `validation/black_oil` 禁用 | `validation/shale_oil` IMEX 类比 |

## 现状

- 合成孪生：`synthetic.make_shale_depletion` + `validation/shale_oil/shale_frac`。
- IMEX 离线尺子 5 套工况（`validation/shale_oil/cmg_shale_suite`）。仍是 **单孔黑油衰竭类比**，不是 GEM。
- 产品：`FractureStripParameterization` + `io.shale_case` + `DigitalTwin.calibrate()`（LM）。
- 井轨迹固定；θ 改条带几何与 K。田间先验按完井/改造典型比给出（\(k_f/k_m\sim10^6\)、半长 ~35% 井段等）。

| 尺子 | 要检验的机理 |
|------|----------------|
| S1 5 缝单井 | 条带导流 + 基质供液 |
| S2 9 缝加密 | 缝密度 / SRV 连通 |
| S3 双水平井 | 井间干扰 |
| S4 父子井 | 时滞射孔 / 新井打开 |
| S5 关井再开 | 压降恢复，非稳态 |

抽检：791 d 末，缝–基质压差约 1.0–1.2×10³ psi，PRES/SW 全有限；ΔSw 可忽略（无注水）。

## 不写进本文

把 mxspr006 海水驱结果改称页岩油。
