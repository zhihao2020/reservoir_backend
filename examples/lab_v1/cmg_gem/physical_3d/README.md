# 30 cm 页岩立方体 CO2 驱 GEM 牌

根据 `三维.docx` 和五井示意图。**不是** M2a 对齐牌 `../lab_v1_dev.dat`。

## 和 M2a 的差别

| | M2a `lab_v1_dev.dat` | 本牌 |
|--|----------------------|------|
| 井 | 左右面注采 | 中心注入 + 四角分层采出 |
| 流体 | C1–nC10 | WinProp CO2-flood 7 伪组分 EXAMPLE |
| T, P | 77 °C, 12 MPa | 120 °C, 50 MPa |
| 岩石 | φ=0.08, k=1 md, DPDP | φ=3.67%, k=0.018 md, 单孔 |
| 围压 | 无 | 60 MPa 总应力 → `*STRESS3D` 有效应力 10 MPa |

## 围压怎么进 GEM

实验室：围压 60 MPa，孔隙压 50 MPa。GEM `*STRESS3D` 是**初始有效应力**。`*BIOTSCOEF 1`（Terzaghi）时 σ′=10 MPa，总应力 60 MPa。`*GCFACTOR 0` 是无约束边界：增量面力为 0，总应力保持初始 60 MPa（恒围压、样品可变形）。`*NOCOUPERM`：文档渗透率已经是该应力下测的，不再用力学去改 k。

杨氏模量 20 GPa、泊松比 0.22：石英 55.7% 的黑色页岩，落在 Lucaogou 静力学 Es 3–21 GPa、ν 0.13–0.36（Xiong et al. 2023）。孔隙压缩系数 `*CPOR 1.2e-6 1/kPa` 由 α²/(φ K_dry) 估。细节在 `spec.yaml`。

原油 PVT 仍是公开 EXAMPLE，不是实验油。有化验再换。

## 井（图上估的深度，三层各 10 cm）

- 注入井：顶 → 监测层2（k=1–11）
- 采出井1：只打上层（k=1–5）
- 采出井2、3：通天（k=1–15）
- 采出井4：只打下层（k=11–15）

注入和回压都是 **50 MPa**（文档）。5 mL/min 只是泵上限（`*BHF 0.0072`）。基质 0.018 mD 上流量靠溶胀/扩散，不是五点驱。

`*PERMK = 0.1 * PERMI`（层理占位）。`*SGT` Corey n=2，Sorg=0.25。报告时刻 0.01 / 0.14 / 1 / 3 day。

GEM 2024.20 对单孔网格会忽略 `*DIFFUSION`（只用于裂缝–基质）。牌上数字保留，这次计算没有分子扩散。

## 我们的 case（通用入口）

`case.yaml` 是单孔组分 + 标量 \(\log k\)（`parameterization: log_permeability`），**不是** `examples/lab_v1/case.yaml` 的 30³ DPDP。
`grid.kdir: down` 把 GEM `*KDIR DOWN`（k=1 为顶）翻到我们的 z 向上网格。
电极行在 `sensors.csv` 里会被 loader 跳过（不做 Archie）。

```bash
python scripts/lab_v1_cmg_run_gem.py --deck examples/lab_v1/cmg_gem/physical_3d/sanwei_co2.dat --case examples/lab_v1/cmg_gem/physical_3d/case.yaml --work results/lab_v1/cmg_gem_physical_3d --timeout 1800
python scripts/lab_v1_cmg_pack_obs.py --case examples/lab_v1/cmg_gem/physical_3d/case.yaml --hidden results/lab_v1/cmg_gem_physical_3d/hidden --export examples/lab_v1/cmg_gem/physical_3d/export
python scripts/lab_v1_cmg_forward_gate.py --case examples/lab_v1/cmg_gem/physical_3d/case.yaml --export examples/lab_v1/cmg_gem/physical_3d/export
python scripts/lab_v1_cmg_invert.py --case examples/lab_v1/cmg_gem/physical_3d/case.yaml --export examples/lab_v1/cmg_gem/physical_3d/export --score --workers 4
```

先过正演等价再反演。15³×7 组分 ensemble 很重。力学不进 \(F\)（对齐 GEM `*NOCOUPERM`）。
定压注入井的源项用 `z_inj`（CO2），不用格子里的原油组成。`phaseid: crit` 对齐 GEM `*PHASEID *CRIT`。

## 跑 GEM

默认 `lab_v1_cmg_run_gem.py` 仍是 M2 4×4×2。这副牌要带 `--case`：

```bash
python -c "from pathlib import Path; from reservoir_backend.twin.cmg_benchmark import run_gem; print(run_gem(Path('examples/lab_v1/cmg_gem/physical_3d/sanwei_co2.dat'), Path('results/lab_v1/cmg_gem_physical_3d'), timeout_s=1800))"
```

不要写进 `results/lab_v1/cmg_gem_run`（那是 M2a）。

Phase 1 `validate_gem_preflight.py` 目前只认 `gem_ccs_family`，这个新牌会 gated。

## GEM 结果

跑次输出：`results/lab_v1/cmg_gem_physical_3d/`（`sanwei_co2.out` / `.sr3` / `figures/`）。

更正后 GEM 2024.20 跑到 3 day：Normal Termination，309 步，0 cut，约 104 s。

- 注采都是 50 MPa。t=3 d 注入约 1.3×10⁻⁷ m³/day（≪ 5 mL/min），烃采收率 **0.034%**
- 注入井格子 z_CO2 约 0.21（k=1），CO2 几乎没离开井格
- `*DIFFUSION` 被 GEM 忽略（单孔不能用该关键字）
