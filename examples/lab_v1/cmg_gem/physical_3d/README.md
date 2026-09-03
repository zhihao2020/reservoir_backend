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

注入目标 5 mL/min（`*BHF 0.0072` m³/day），注入 BHP ≤ 50 MPa，采出井 49.5 MPa。

## 我们的 case（通用入口）

`case.yaml` 是单孔组分 + 标量 \(\log k\)，**不是** `examples/lab_v1/case.yaml` 的 30³ DPDP。
电极行在 `sensors.csv` 里会被 loader 跳过（不做 Archie）。

```bash
python scripts/lab_v1_cmg_run_gem.py --deck examples/lab_v1/cmg_gem/physical_3d/sanwei_co2.dat --case examples/lab_v1/cmg_gem/physical_3d/case.yaml --work results/lab_v1/cmg_gem_physical_3d --timeout 1800
python scripts/lab_v1_cmg_pack_obs.py --case examples/lab_v1/cmg_gem/physical_3d/case.yaml --hidden results/lab_v1/cmg_gem_physical_3d/hidden --export examples/lab_v1/cmg_gem/physical_3d/export
python scripts/lab_v1_cmg_forward_gate.py --case examples/lab_v1/cmg_gem/physical_3d/case.yaml --export examples/lab_v1/cmg_gem/physical_3d/export
python scripts/lab_v1_cmg_invert.py --case examples/lab_v1/cmg_gem/physical_3d/case.yaml --export examples/lab_v1/cmg_gem/physical_3d/export --score --workers 4
```

先过正演等价再反演。15³×7 组分 ensemble 很重。力学不进 \(F\)（对齐 GEM `*NOCOUPERM`）。

## 跑 GEM

默认 `lab_v1_cmg_run_gem.py` 仍是 M2 4×4×2。这副牌要带 `--case`：

```bash
python -c "from pathlib import Path; from reservoir_backend.twin.cmg_benchmark import run_gem; print(run_gem(Path('examples/lab_v1/cmg_gem/physical_3d/sanwei_co2.dat'), Path('results/lab_v1/cmg_gem_physical_3d'), timeout_s=1800))"
```

不要写进 `results/lab_v1/cmg_gem_run`（那是 M2a）。

Phase 1 `validate_gem_preflight.py` 目前只认 `gem_ccs_family`，这个新牌会 gated。

## 第一次 GEM 2024.20 结果

`results/lab_v1/cmg_gem_physical_3d/sanwei_co2.out`：Normal Termination，155 步，0 cut，力学 155 步，约 52 s。

- 五口井全部 open
- 孔隙压 50 MPa，采出井 49.5 MPa（注采都钉 50 MPa 时注入量≈0，按计划把回压降了 500 kPa）
- t=0.14 d 注入井储层气速率 5.1×10⁻⁴ m³/day ≈ **0.35 mL/min**（BHP 卡住，不是 5 mL/min）。0.018 mD 上要打到 5 mL/min 大约需要 14 MPa 压差，不改 k
- 注入井格子 z_CO2 到 0.90，CO2 已进网格；四口采出井都出油
- 累注 CO2 ≈ 1.00 mol，物质平衡误差 4×10⁻³ %
