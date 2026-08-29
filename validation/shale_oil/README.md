# 页岩油 / 致密油（历史 IMEX 尺子）

> 2026-08 内核按 `docs/target_architecture.md` 绿地重构。页岩反演走 **LM + `FractureStripParameterization`**（默认 **4 维**连续 θ；缝面数/相位按完井冻结），不再依赖已删除的 `pipeline/` / ES-MDA。

与黑油水驱 **不是同一套软件主张**。页岩油是衰竭 + 水平井 + 高渗裂缝条带；基质超低渗。

## CMG 尺子（IMEX 类比，离线）

五套工况已建成并跑通 IMEX `Normal Termination`。克隆官方 `mxspr006` 的黑油 PVT，去掉注水井；**单孔、无吸附、不是 GEM**。产品路径不调用 CMG。

| 编号 | 目录 | 工况 | 井 | 缝面 I | 缝块 |
|------|------|------|----|--------|------|
| S1 | [cmg_s1_hw5frac](cmg_s1_hw5frac/) | 单水平井，5 条水力缝，衰竭 | HW1 | 4,8,11,14,18 | 255 |
| S2 | [cmg_s2_hw9frac](cmg_s2_hw9frac/) | 同井场，9 条更密缝 | HW1 | 3,5,…,19 | 459 |
| S3 | [cmg_s3_twohw](cmg_s3_twohw/) | 两口平行水平井，t=0 同时开 | HW1+HW2 | 同 S1 | 345 |
| S4 | [cmg_s4_parent_child](cmg_s4_parent_child/) | 父井先开；子井约 1 年后开 | HW1；HW2@365 d | 同 S1 | 345 |
| S5 | [cmg_s5_shutin](cmg_s5_shutin/) | 同 S1，中期关井再开井 | HW1 | 同 S1 | 255 |

统一网格 `21×31×5` CART（3255 块）；基质 ~0.001 md，缝 8000 md，SRV 0.4 md；`*MIN BHP 1500 psi`。  
末时刻 `.out` 抽检：PRES/SW 全有限、非常数；缝平均压力比远场基质低 **约 965–1160 psi**；ΔSw 仅 ~0.002（衰竭而非水驱，尺子主信号是压降）。

生成 / 重跑 / 抽检：

```bash
python validation/shale_oil/cmg_shale_suite/build_shale_suite.py
python validation/shale_oil/cmg_shale_suite/run_imex.py --case all
python validation/shale_oil/cmg_shale_suite/smoke_parse.py
pytest tests/cases/test_shale_cmg_suite.py -q
```

不要接到 `validation/black_oil/cmg_probe_study`，也不要用 mxspr006 海水驱冒充页岩。

## 产品反演（LM）

- **参数化**：`FractureStripParameterization` — 默认 \(\theta=[\log k_m,\log k_f,\log k_{\mathrm{srv}},\log x_f]\)；\(n_{\mathrm{frac}}\)、相位按完井固定（BHP  alone 对整数缝面数雅可比≈0）。`free_geometry: true` 恢复 6 维。
- **正演**：默认顺序黑油（`fully_implicit: false`）；FIM 可选，全网格 S1 很慢。
- **Case 组装**：`reservoir_backend.io.shale_case.twin_from_shale_truth` — truth JSON + IMEX `.out` → `DigitalTwin`
- **尺子脚本**：`validation/shale_oil/cmg_shale_suite/run_suite_inversion.py`（S1–S5，LM + 统一 `run_report` schema）
- **预报尺子**：`validation/shale_oil/cmg_shale_suite/run_forecast_validate.py`（S5 关井段，slow）
- **YAML 入口**：`examples/shale_oil/s1.yaml` … `s5.yaml` → `reservoir invert`
- **合成孪生**（无 CMG）：`synthetic.make_shale_depletion` + `validation/shale_oil/shale_frac/run_validate.py`

控制进 \(F\)（定产 rate），观测进 misfit（完井 cell 压力）；不把观测井压钉成 Dirichlet。

首轮闸门（`run_suite_inversion.py` 内 `gates_pass`，跨 IMEX MVP）：

- `dp_sign_match`
- `dp_ratio` ≥ **0.2**（仍低于理想 0.60–0.95）
- \(|n_{\mathrm{frac,inv}} - n_{\mathrm{frac,true}}| \le 1\)（默认冻结为完井缝面数）
- `assimilate_nrmse` < **10**（cheap σ×5）
- `k_frac_over_matrix` > **100**

工况对齐：定产 + **MIN BHP=1500 psi**（与 IMEX `*OPERATE *MIN *BHP` 同语义）；产率对齐 `scale_min=0.5`（不再缩到 0.2）。

S1 现状（顺序两相 + 4 维 θ + MIN BHP）：~130 s；LM 10.2→7.4；`dp_ratio≈0.26`；`rate_scale≈2.0`；图见 `cmg_shale_suite/figures/s1_cmg_vs_fpost.png`。

## 论文主张

见 [PAPER.md](PAPER.md)。

```bash
python validation/shale_oil/shale_frac/run_validate.py
python validation/shale_oil/cmg_shale_suite/run_suite_inversion.py
python validation/shale_oil/cmg_shale_suite/run_forecast_validate.py  # 需 S5 .out，slow
pytest tests/inverse/test_frac_parameterization.py tests/cases/test_shale_synthetic.py tests/cases/test_shale_cmg_suite.py -q
```

`suite_inversion_report.json` 现标注 `algorithm: LM` 并含 `run_reports[]`（与 CLI `invert.json` 同 schema）。归档的 ES-MDA 报告已被 LM 流程取代。
