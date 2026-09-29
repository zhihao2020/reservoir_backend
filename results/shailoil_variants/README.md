# shailoil GEM 变体算例

由 `scripts/shailoil_variants.py` 从 `examples/shailoil.dat` 生成并运行（GEM 2024.20，`gm202420.exe -f`，6 个算例并行，单线程）。

- deck：`examples/shailoil_variants/shailoil_<变体>.dat`（改动写在文件头注释），运行目录另有一份拷贝
- het_k 的 k/phi：`examples/shailoil_variants/shailoil_het_k_field.csv`（i,j,k,phi,k_md；GEM 1 起始，K=1 为顶层，KDIR DOWN）
- 每个变体目录 `results/shailoil_variants/<变体>/`：`.dat`、`.out`、`.sr3`、`.geo`（地质力学）、`run.log`（GEM 标准输出）、`run.json`（墙钟时间/返回码）
- 汇总数据：`results/shailoil_variants/summary.json`

## 所有变体共同的修改

- *OUTPRN/*OUTSRF *GRID add *Z 'CO2' (global CO2 mole fraction)
- *WSRF *GRID / *GRIDDEFORM changed from 1 (every step) to *TIME (report times only)
- *WPRN *GRID *TIME and *WPRN *WELL *TIME stated explicitly
- schedule: *TIME 0.01 0.1 1 3 7 15 30 45 60 75 90 120 ... (every 30 d) ... 1800, then *STOP
- *DTMAX 0.01 up to day 15, *DTMAX 1.0 afterwards (base deck used 30.0)
- grid, EOS, initial conditions, well locations/perfs, KDIR DOWN unchanged unless noted
- 井输出沿用 base 的 `*OUTPRN *WELL *ALL` / `*OUTSRF *WELL *PAVG`：SR3 中每口井每步都有 BHP、地面条件量（`OILRATSC/GASRATSC/WATRATSC`，累计 `*VOLSC`）和储层条件量（`OILRATRC/GASRATRC/WATRATRC/BHFRATRC`，累计 `*VOLRC`）；.out 井报告同时打印 Surface 与 Reservoir 的瞬时/累计注采量
- 网格输出：`*OUTPRN/*OUTSRF *GRID *PRES *SO *SG *SW *POROS *PERM`（与 base 相同），只在报告时刻写出
- base 中的 `*OUTSRF *SPECIAL` 测点保留未动（后处理不依赖）

## 各变体结果

sg 均值：`mean` 为 3375 格算术平均，`PV` 为按孔隙度（体积相同）加权；自由气格数为 sg > 0.01 的网格数。

| 变体 | 改动 | 墙钟时间 (s) | 正常结束 | 步数 | 时间步截断 | 求解器失败 | 物质平衡误差 % | 第30天 sg mean / PV | 第30天自由气格数 | 最后一步 (天) sg mean / PV | 最后一步自由气格数 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 参考算例：仅做所有变体共同的输出/时间表修改（非任务要求，供对照） | 3370.5 | 是 | 3345 | 5 | 0 | 3.0453E-02 | 0.1401 / 0.1401 | 511 | (1800) 0.0000 / 0.0000 | 0 |
| het_k | PERMI 对数正态随机场（几何均值 0.03 mD，σ_lnk=1.0，Lh=0.08 m，Lv=0.04 m，指数协方差，seed=20260929），POR=0.05·(k/0.03)^0.1 截断 [0.02,0.10]，*PERMI/*POR *ALL 逐格写出 | 3480.4 | 是 | 3364 | 12 | 0 | 1.7550E-02 | 0.0901 / 0.0869 | 334 | (1800) 0.0000 / 0.0000 | 0 |
| layered_k | PERMI *KVAR：K=1-3/4-6/7-9/10-12/13-15 = 0.1/0.01/0.05/0.005/0.03 mD | 3530.4 | 是 | 3358 | 4 | 0 | 2.6494E-02 | 0.3744 / 0.3744 | 1389 | (1800) 0.0008 / 0.0008 | 3 |
| rate_x3 | INJ OPERATE MAX BHF 0.0072 → 0.0216 | 3200.4 | 是 | 3348 | 2 | 0 | 1.0229E-01 | 0.0152 / 0.0152 | 54 | (1800) 0.0000 / 0.0000 | 0 |
| rate_div3 | INJ OPERATE MAX BHF 0.0072 → 0.0024 | 3480.3 | 是 | 3336 | 2 | 0 | 1.8671E-02 | 0.4016 / 0.4016 | 1501 | (1800) 0.0000 / 0.0000 | 0 |
| inj_bottom | INJ 射孔 8,8,K=1-11 → 8,8,K=11-15（REFLAYER 在 K=11） | 3480.3 | 是 | 3350 | 6 | 0 | 5.4436E-02 | 0.3032 / 0.3032 | 1110 | (1800) 0.0000 / 0.0000 | 0 |

sg 均值 / 自由气格数随时间（算术平均 / sg>0.01 格数）：

| 变体 | 1 d | 7 d | 15 d | 30 d | 60 d | 90 d | 150 d | 300 d | 600 d | 1200 d | 1800 d |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.699 / 3013 | 0.490 / 1853 | 0.292 / 1083 | 0.140 / 511 | 0.046 / 165 | 0.017 / 59 | 0.005 / 17 | 0.001 / 3 | 0.000 / 0 | 0.000 / 0 | 0.000 / 0 |
| het_k | 0.722 / 3075 | 0.502 / 1885 | 0.254 / 943 | 0.090 / 334 | 0.030 / 110 | 0.016 / 57 | 0.007 / 26 | 0.001 / 5 | 0.000 / 1 | 0.000 / 0 | 0.000 / 0 |
| layered_k | 0.514 / 2476 | 0.577 / 2336 | 0.499 / 1904 | 0.374 / 1389 | 0.234 / 857 | 0.151 / 550 | 0.065 / 235 | 0.009 / 33 | 0.003 / 10 | 0.001 / 5 | 0.001 / 3 |
| rate_x3 | 0.653 / 2588 | 0.202 / 744 | 0.065 / 235 | 0.015 / 54 | 0.002 / 7 | 0.001 / 3 | 0.000 / 0 | 0.000 / 0 | 0.000 / 0 | 0.000 / 0 | 0.000 / 0 |
| rate_div3 | 0.556 / 2771 | 0.703 / 2821 | 0.560 / 2148 | 0.402 / 1501 | 0.224 / 824 | 0.143 / 521 | 0.059 / 213 | 0.012 / 43 | 0.002 / 6 | 0.000 / 1 | 0.000 / 0 |
| inj_bottom | 0.579 / 2610 | 0.549 / 2136 | 0.436 / 1628 | 0.303 / 1110 | 0.161 / 587 | 0.095 / 344 | 0.041 / 142 | 0.001 / 2 | 0.000 / 0 | 0.000 / 0 | 0.000 / 0 |

注意：自由气在前 1-7 天迅速出现（生产井 19 MPa 低于初始 20 MPa，加上注入 CO2），随后逐步消失；除 layered_k 外，最后一步全场 SO=1、SG=0。原因是 CO2 充满 30 cm 立方体后（.out 中 HCPVIN：rate_div3 约 3.2e5 %，base 约 9.6e5 %，rate_x3 约 2.9e6 %）流体变为单一烃相，而 deck 使用 `*PHASEID *OIL`，单相一律标为油相。因此后期 sg≈0 是相标识造成的，并不代表 CO2 前缘消失；后期 CO2 分布请用全局摩尔分数 z_CO2（下表）。

z_CO2 均值 / z_CO2>0.5 格数随时间（初始 z_CO2=0.03；SR3 变量 `Z(5)`，.out 中为 'CO2' 全局摩尔分数图）：

| 变体 | 1 d | 7 d | 15 d | 30 d | 60 d | 90 d | 150 d | 300 d | 600 d | 1200 d | 1800 d |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.876 / 3111 | 0.987 / 3371 | 0.995 / 3375 | 0.998 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 |
| het_k | 0.893 / 3186 | 0.986 / 3365 | 0.995 / 3374 | 0.999 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 |
| layered_k | 0.688 / 2409 | 0.956 / 3339 | 0.984 / 3367 | 0.992 / 3371 | 0.996 / 3373 | 0.998 / 3374 | 0.999 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 |
| rate_x3 | 0.961 / 3345 | 0.997 / 3375 | 0.999 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 |
| rate_div3 | 0.701 / 2498 | 0.947 / 3314 | 0.981 / 3368 | 0.992 / 3375 | 0.997 / 3375 | 0.998 / 3375 | 0.999 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 |
| inj_bottom | 0.753 / 2639 | 0.977 / 3371 | 0.991 / 3375 | 0.996 / 3375 | 0.998 / 3375 | 0.999 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 | 1.000 / 3375 |

结论：CO2 前缘在 0.01–3 天内扫过整个立方体（base 在 0.1 d 时 z_CO2 均值 0.46、标准差 0.37，7 d 时已为 0.99），rate_div3 / layered_k / inj_bottom 稍慢，也在 7–30 d 内基本扫完；压力场从约 1 d 起即为稳态（全场压力标准差不再变化）。因此对场重构有信息量的快照主要是 0.01、0.1、1、3（至多 7、15）天，30 d 以后的 PRES/SG/z_CO2 场几乎不随时间变化。

SR3 中 PERMI/J/K 以 darcy 存储（UnitsTable：Permeability internal=darcy, output=md），读取时乘 1000 得 mD。

## 收敛/警告

- **base**：初始化 0 警告 / 0 错误；"Repeat time step: program failed to converge" 2 次（汇总时间步截断 5 次），求解器失败 0 次，运行期 WARNING 行 0；GEM CPU 3300.25 s
  - 井控信息：`**WMLPOC: Set new operating constraint of type BHP for well 1 ('INJ') .`
  - 井控信息：`**WMLPOC: Switch operating constraint of well 1 ('INJ') to type BHF and advance to next time step.`
  - 井控信息：`**WMMAIN: Repeat (1) for this time step due to constraint violation.`
  - 井控信息：`**WMMRCH: Switch well 1 ('INJ') to BHP constraint for changes entered under OPERATE, ALTER or TARGET`
  - 井控信息：`**WMOCNI: Maximum BHF constraint violated for well 1 ('INJ') Max:  .007200,Actual:  .007820  m3/day`
  - 井控信息：`**WMOCNI: Maximum BHF constraint violated for well 1 ('INJ') Max:  .007200,Actual:  .007956  m3/day`
  - 井控信息：`**WMOCNI: Maximum BHP constraint violated for well 1 ('INJ') Max:  20000.0,Actual:  20047.2 kPa`
- **het_k**：初始化 0 警告 / 0 错误；"Repeat time step: program failed to converge" 6 次（汇总时间步截断 12 次），求解器失败 0 次，运行期 WARNING 行 0；GEM CPU 3414 s
  - 井控信息：`**WMLPOC: Switch operating constraint of well 1 ('INJ') to type BHF and advance to next time step.`
  - 井控信息：`**WMMRCH: Switch well 1 ('INJ') to BHP constraint for changes entered under OPERATE, ALTER or TARGET`
  - 井控信息：`**WMOCNI: Maximum BHF constraint violated for well 1 ('INJ') Max:  .007200,Actual:  .007200  m3/day`
- **layered_k**：初始化 0 警告 / 0 错误；"Repeat time step: program failed to converge" 1 次（汇总时间步截断 4 次），求解器失败 0 次，运行期 WARNING 行 0；GEM CPU 3457.4375 s
  - 井控信息：`**WMLPOC: Switch operating constraint of well 1 ('INJ') to type BHF and advance to next time step.`
  - 井控信息：`**WMMRCH: Switch well 1 ('INJ') to BHP constraint for changes entered under OPERATE, ALTER or TARGET`
  - 井控信息：`**WMOCNI: Maximum BHF constraint violated for well 1 ('INJ') Max:  .007200,Actual:  .007633  m3/day`
- **rate_x3**：初始化 0 警告 / 0 错误；"Repeat time step: program failed to converge" 0 次（汇总时间步截断 2 次），求解器失败 0 次，运行期 WARNING 行 0；GEM CPU 3133.734375 s
  - 井控信息：`**WMMRCH: Switch well 1 ('INJ') to BHP constraint for changes entered under OPERATE, ALTER or TARGET`
- **rate_div3**：初始化 0 警告 / 0 错误；"Repeat time step: program failed to converge" 0 次（汇总时间步截断 2 次），求解器失败 0 次，运行期 WARNING 行 0；GEM CPU 3415.390625 s
  - 井控信息：`**WMLPOC: Switch operating constraint of well 1 ('INJ') to type BHF and advance to next time step.`
  - 井控信息：`**WMMRCH: Switch well 1 ('INJ') to BHP constraint for changes entered under OPERATE, ALTER or TARGET`
  - 井控信息：`**WMOCNI: Maximum BHF constraint violated for well 1 ('INJ') Max:  .002400,Actual:  .002477  m3/day`
- **inj_bottom**：初始化 0 警告 / 0 错误；"Repeat time step: program failed to converge" 4 次（汇总时间步截断 6 次），求解器失败 0 次，运行期 WARNING 行 0；GEM CPU 3414.65625 s
  - 井控信息：`**WMLPOC: Switch operating constraint of well 1 ('INJ') to type BHF and advance to next time step.`
  - 井控信息：`**WMMRCH: Switch well 1 ('INJ') to BHP constraint for changes entered under OPERATE, ALTER or TARGET`
  - 井控信息：`**WMOCNI: Maximum BHF constraint violated for well 1 ('INJ') Max:  .007200,Actual:  .007365  m3/day`

## 文件大小 (MB)

- base：`run.json` 0.0，`run.log` 0.5，`shailoil_base.dat` 0.0，`shailoil_base.geo` 2.6，`shailoil_base.out` 12.7，`shailoil_base.sr3` 50.6
- het_k：`run.json` 0.0，`run.log` 0.5，`shailoil_het_k.dat` 0.1，`shailoil_het_k.geo` 2.7，`shailoil_het_k.out` 25.0，`shailoil_het_k.sr3` 56.1
- layered_k：`run.json` 0.0，`run.log` 0.5，`shailoil_layered_k.dat` 0.0，`shailoil_layered_k.geo` 2.7，`shailoil_layered_k.out` 15.1，`shailoil_layered_k.sr3` 54.0
- rate_x3：`run.json` 0.0，`run.log` 0.5，`shailoil_rate_x3.dat` 0.0，`shailoil_rate_x3.geo` 2.6，`shailoil_rate_x3.out` 12.1，`shailoil_rate_x3.sr3` 50.0
- rate_div3：`run.json` 0.0，`run.log` 0.5，`shailoil_rate_div3.dat` 0.0，`shailoil_rate_div3.geo` 2.6，`shailoil_rate_div3.out` 14.1，`shailoil_rate_div3.sr3` 52.2
- inj_bottom：`run.json` 0.0，`run.log` 0.5，`shailoil_inj_bottom.dat` 0.0，`shailoil_inj_bottom.geo` 2.7，`shailoil_inj_bottom.out` 11.9，`shailoil_inj_bottom.sr3` 50.7

## het_k 输入核对

SR3 初始时刻 PERMI / POROS / PERMK 与 CSV 对比（3375 格）：PERMI 最大相对误差 5.31e-07，POROS 1.51e-07，PERMK 相对 0.2·PERMI 5.36e-07。
