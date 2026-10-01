# 自然变量 FIM 移植记录（Step 1 已完成）

> 目标：把组分前向模型从「整体组分 z + 闪蒸 V(z)」改成 MRST 的**自然变量（natural variables）**形式，
> 用**逸度相等方程**替代闪蒸、饱和度直接做主变量，消除泡点处的雅可比奇异（这是冻压力输运/假收敛一直绕不过去的根子）。
> 本文档记录进度与关键约定，供后续会话继续。

## 为什么自然变量能消除泡点奇异

- 当前形式：主变量是整体组分 `z`，闪蒸 `V(z)` 在泡点处**不连续**（V 从 0 跳到 >0），数值雅可比在泡点处奇异 → 牛顿失速。
- 自然变量形式（MRST `NaturalVariablesCompositionalModel`）：主变量是 `(p, sO/sG, x, y)`（饱和度 + 相组分），
  用**逸度相等** `ln(x_i·φ_i^L) = ln(y_i·φ_i^V)` 作为显式方程。逸度相等是**光滑**的，泡点穿越变成「主变量切换 + 饱和度 chopping」。

## 关键约定（之前卡住、现已搞清）

**逸度系数导数必须用「摩尔数导数」 `d lnφ_i / d ln n_j`，不是摩尔分数扰动。**

- 正确：相对扰动摩尔数 `n_j → n_j(1+h)`，再归一化成摩尔分数，除以 `ln(1+h)`。
  这满足 **Euler 关系 `Σ_j d lnφ_i/d ln n_j = 0`**（逸度系数对摩尔数是 0 次齐次），验证精确到 1e-6。
- 错误（曾踩坑）：直接扰动摩尔分数 `x_j += h`、减 `x_末 -= h`，或扰动后不归一化——破坏了 sum=1 约束，导致雅可比错 4×。

**但闪蒸内部 Newton 的雅可比要的不是摩尔数导数，而是「无约束摩尔分数导数」 `∂lnφ_i/∂x_j`。** 两者是两件事：

- **摩尔数导数** `d lnφ_i/d ln n_j`（`fugacity_mole_deriv`）是「保持 Σx=1」的单纯形导数 → 用于 **FIM 外层雅可比**（x 是单纯形上的主变量，Step 5）。
- **无约束摩尔分数导数** `∂lnφ_i/∂x_j`（扰动 `x_j += h`、不归一化）→ 用于**闪蒸内部 Newton**（x/y 是自由向量，闭合方程 Σx=Σy 显式约束，正是 MRST `getPhaseFractionDerivativesPTZ` 用 AD 求导得到的量）。
- 两者差一个径向规范项 `G_i = Σ_k x_k ∂lnφ_i/∂x_k`：未归一化的 EOS 逸度系数对 x **不是 0 次齐次**（A、B 随 x 非线性缩放），故 `G_i ≠ 0`。误把 `∂lnφ_i/∂x_j` 当 `D[i,j]/x_j` 会漏掉 `G_i`，雅可比错 ~1（FD 检验抓到）。

## Step 1 已完成（可独立验证）

`src/core/pr_eos.py` 新增两个函数：

1. `fugacity_mole_deriv(x, P, T, phase, h)` → `(n, ncomp, ncomp)`，`out[i,j] = d lnφ_i / d ln n_j`。
   - 测试 `test_fugacity_mole_deriv_satisfies_euler`：Euler 关系 < 1e-5。

2. `natural_variables_residual(L, x, y, z, P, T)` → `(n, 2·ncomp+1)` 残差：
   - 质量平衡 `z_i − L·x_i − (1−L)·y_i`（ncomp）
   - 逸度相等 `ln(y_i·φ_i^V) − ln(x_i·φ_i^L)`（ncomp）
   - 闭合 `Σx − Σy`（1）
   - 测试 `test_natural_variables_residual_vanishes_at_flash`：在 SSI 闪蒸解处残差 ≈ 0（atol 1e-4）。

这两个是自然变量 Newton 的「积木」：残差就是 Newton 要驱动的量，摩尔数导数就是雅可比的核心块。

## Step 2 已完成

`src/core/pr_eos.py` 新增自然变量 Newton 闪蒸（已验证，**暂未替换生产闪蒸 `_flash_vec`**，原因见下）：

- `_fugacity_frac_deriv(x, aij, b, P, T, phase, h)` → 无约束摩尔分数导数 `∂lnφ_i/∂x_j`（扰动 `x_j += h`，不归一化）。
- `_fugacity_frac_deriv_analytic(x, aij, b, P, T, phase)` → **PR 逸度解析导数**（闭式，中心差分对照到 ~1e-8）。`∂lnφ_i/∂x_j = g_ij + (∂lnφ_i/∂Z)(∂Z/∂x_j)`，显式项（固定 Z 对 a/b/ψ 求导）+ 隐式项（立方根 Z 随 A/B 移动，`∂Z/∂x_j = −(∂F/∂x_j)/(∂F/∂Z)`）。
- `_fugacity_mole_deriv_analytic` → 解析摩尔数导数 `d lnφ_i/d ln n_j = x_j·(∂lnφ_i/∂x_j − G_i)`，`G_i = Σ_k x_k ∂lnφ_i/∂x_k`（径向规范项），精确满足 Euler `Σ_j D = 0`。
- `natural_variables_jacobian(L, x, y, P, T)` → `(n, 2ncomp+1, 2ncomp+1)` 雅可比，逸度块用解析导数。块结构（变量序 `[L, x, y]`，方程序 `[mass, fug, close]`）：
  - `∂mass/∂L = x−y`、`∂mass/∂x = L·I`、`∂mass/∂y = (1−L)·I`（线性）
  - `∂fug/∂x = −diag(1/x) − ∂lnφ^L/∂x`、`∂fug/∂y = +diag(1/y) + ∂lnφ^V/∂y`
  - `∂close/∂x = 1ᵀ`、`∂close/∂y = −1ᵀ`
- `_flash_natural_variables(z, P, T, aij, b)` → `(V, x, y)`：Wilson K + Rachford-Rice 初值 → Newton 解 `(L, x, y)`，二次收敛、不收敛回退 `_flash_vec`。

**解析导数带来的提速**：Newton 闪蒸从 4.8× 慢降到 3.27× 慢（仍比 SSI 慢，张量运算为主）；但解析导数（`_fugacity_mole_deriv`/`natural_variables_jacobian`）的真正价值在 Step 5——FIM 全局雅可比嵌入逸度导数、逐格闪蒸被逸度方程取代，单格闪蒸速度不再是瓶颈。

**关键修正**：Step 1 里「摩尔数导数就是雅可比核心块」不准确——摩尔数导数 `d lnφ/d ln n_j` 是 FIM 外层（Step 5）单纯形导数；闪蒸内部 Newton 需要**无约束摩尔分数导数** `∂lnφ_i/∂x_j`（含径向规范项 `G_i=Σ_k x_k ∂lnφ_i/∂x_k`，未归一化逸度系数非 0 次齐次，G_i≠0）。误用 `D/x` 漏掉 G_i，FD 检验抓到 ~1 的误差。

测试（`tests/test_compositional_fim.py`）：`test_natural_variables_jacobian_matches_finite_difference`、`test_analytic_fugacity_derivatives_match_finite_difference`、`test_natural_variables_flash_matches_ssi`、`test_natural_variables_newton_quadratic`、`test_natural_variables_state_consistent_with_flash`。

## 下一步（后续会话）

**Step 3 — 相态标志 + 主变量切换**（MRST `setFlag`/`getFlag`，`EquationOfStateModel.m:1156`）：
- `flag = 1·pureLiquid + 2·pureVapor`，`pureLiquid = (L==1)`、`pureVapor = (L==0)`，其余两相。容差 `saturationEpsilon=1e-6`。
- 两相格主变量 `(p, sw, sO, x_1..x_13, y_1..y_13)`；纯液 `y=x`（y 非主变量）、纯气 `x=y`（x 非主变量）、`sO/sG` 由 `sw` 定 → 主变量数随相态变（MRST `equationsNaturalVariables.m:160-179` 的 `~pureLiquid`/`pureVapor` 掩码）。

**Step 4 — 饱和度 chopping**（MRST `getSaturationIncrements`/`updateState`，`NaturalVariablesCompositionalModel.m:49-196`）：
- `w_sat = min(dsMax / max|ds|, 1)`、`w_x = min(dxMax / max|dx|, 1)`，取 `w = min(w_sat, w_x, w_y)` 松弛饱和度/组分更新，`capunit` + 归一化。
- 相变逻辑 `flashPhases`（`NaturalVariablesCompositionalModel.m:198`）：单相格做相稳定测试（`performPhaseStabilityTest`）决定是否转入两相。

**Step 5 — 接入 FIM**（MRST `equationsNaturalVariables.m`，主变量 `(p, sw, sO, x, y)` 替换 `(p, sw, z)`）：
- 残差：组分摩尔平衡 `accum·(m_c − m_c0) − q_c + div(y_c·λg/v_g·∇p_g + x_c·λl/v_l·∇p)`（`m_c = sO·x_c/v_l + sG·y_c/v_g`，`sG=1−sw−sO`）；逸度 `ln(y_c·φ^V) − ln(x_c·φ^L)`；水 `accum·(sw−sw0) − qw + div(λw·∇p)`。
- 雅可比：逸度块用 `natural_variables_jacobian`（`∂fug/∂x = −diag(1/x) − ∂lnφ^L/∂x` 等），其余（迁移率/体积对 sO、p 的导数）沿用现有 FD；质量平衡对 x/y 的导数含 `∂m_c/∂x_j = sO·δ_cj/v_l + sO·x_c·∂(1/v_l)/∂x_j`。
- 主变量从 `(p, sw, z)` 换到 `(p, sw, sO, x, y)`，`z` 退化为派生量 `z_c = m_c/Σm`。
- 关键：**闪蒸被逸度方程取代**，泡点奇异消失（这就是本系列改动的最终目的）。

## Step 5 已完成：稀疏块组装（`natural_variables.py`）

把 `natural_variables_step` 的雅可比从「混合 FD + 解析 flux-p + 解析逸度」升级为**完整解析稀疏块组装**
（照 MRST `equationsNaturalVariables` 的 AD 结构，`sparse_jacobian_full`），接入 Newton 循环。块清单：

- **flux-p**：冻结迎风 Laplacian + 密度 p 导数 `A·diag(λ·∂ρ/∂p)`。
- **flux-s/x/y**：`A_g·diag(y_c ρg ∂λg/∂s_k) + A·diag(x_c ρl ∂λl/∂s_k)` 及密度-组分导数；迁移率用
  `_mobility_derivs_full` 的 3×3 全偏导（tabular Stone-I 的 `kro=krog(sg)·krow(sw)` 使 λo 同时依赖 sw、sg，
  不能用 `_corey_mobilities_derivs` 的自饱和度捷径）。
- **accumulation / fugacity / closure / 井项**：块对角 + Peaceman 乘积法则解析偏导。
- **rate 控制**：注入井 Schur 耦合（`∂r_state/∂bhp`、`∂rate/∂state`、`∂rate/∂bhp`）解析组装。
- **隐含变量链式折叠**（照 MRST AD，`:170-179`）：纯液 `y=x, sO=1−sw, sG=0`、纯气 `x=y, sG=1−sw, sO=0`，
  用线性 fill 矩阵 F 做 `J = J_raw @ F` 再 `em/vm` 掩码。

**两个关键修复**（否则多格发散/错解）：

1. **逸度残差形式**：残差是线性型 `x·φL − y·φV`（MRST `f_V−f_L` 一致），旧 `fugacity_jacobian_full`
   误用对数型导数 `−I/x − ∂lnφ`（FD 差 1e3–1e4）。改为线性型 `φL(I + x·∂lnφL/∂x)`（FD 差 ~1e-5）。
2. **`unpack_full` 布局 bug（存量）**：`.reshape(ncomp−1, n).T` 是组分主序，与 `pack_full` 的 `.ravel()`
   （格主序）及所有掩码不一致——仅 `n==1` 时等价，故单格冒烟通过、多格全错。改 `.reshape(n, ncomp−1)`。

**验证**：`test_natural_variables_step_matches_reference`（单格）由失败转通过；
新增 `test_natural_variables_multi_cell_matches_reference`（纯液+两相两格，覆盖折叠+flux 块）与参考一致；
稀疏解析雅可比对照稠密 FD 一致到 ~2.7e-5（多格两相）。

## 相变切换已完成（照 MRST `flashPhases` + `performPhaseStabilityTest`）

补上了最后一块：**单相↔两相切换**（此前自然变量卡在「纯液格永远纯液」，羽流从死油起注发散）。

- **`phase_stability_test`**（`pr_eos.py`）：Michelsen 切平面距离（TPD）相稳定测试。两轮逐次代入（汽型/液型），
  `Y_i = z_i·φ^L_i(z)/φ^V_i(w)` / `X_i = z_i·φ^V_i(z)/φ^L_i(w)`，`S = ΣY > 1` 判该萌芽相存在。验证：单相
  z_CO2=0.0–0.3 → stable，两相 0.4–0.99 → unstable，与闪蒸相态一致。
- **`flash_phases`**（`natural_variables.py`）：纯液格稳定测试不稳 → 插入 `saturationEpsilon` 的萌芽汽
  （`y=y_inc`）；纯气格不稳 → 插入萌芽液（`x=x_inc`）；两相格 `sO≤0`/`sG≤0` → 收成单相。接入牛顿循环
  `fill_inactive` 之后，下一次迭代用新相态掩码。
- **验证**：死油起注 CO₂ 8 步——前 6 步 CO₂ 溶解（sg=0、z_CO₂ 0→0.38），第 7 步过泡点（z_CO₂≈0.42）析出
  自由气 sg=0.077→0.119，牛顿收敛不再发散（此前 npl=3 卡死、z_CO₂ 爆 1e163）。新增
  `test_natural_variables_phase_transition`。

## Schur 约简已完成（照 MRST `ReducedLinearizedSystem`/`keepNum`）

把自然变量两相格的 **30n 系统压到 15n**：保留组分摩尔平衡+水（`keep_eq_mask`）和
`(p, sw, sO, x[1..12])`（`keep_var_mask`），消去逸度+闭合和 `(sG, x[0], y)`。`x[0]` 放进被消
块是让逸度块（14×14）非奇异的必要之举（MRST 同）。`schur_solve` 做 `A_red = B − C·E⁻¹·D`，
`du_k = A_red⁻¹(−f + C·E⁻¹h)`，再恢复 `du_e = E⁻¹(−h − D·du_k)`；E 块格内局部（两相格 15×15
块对角），LU 便宜。单相格无逸度/闭合可消，`ke.size==0` 时退回全解。8 个自然变量测试全过。

## 参考（照抄，不要自造）

- `reference_projects/MRST-main/autodiff/compositional/models/natvars/equationsNaturalVariables.m` — 残差组装。
- `reference_projects/MRST-main/autodiff/compositional/models/eos/EquationOfStateModel.m` — `equationsEquilibrium`（质量平衡 + 逸度 + 闭合）、`getPhaseFractionDerivativesPTZ`（解析闪蒸雅可比）、`solveCubicEOS`（立方根，已移植）。
- `reference_projects/MRST-main/autodiff/compositional/utils/cubicPositive.m` — 立方根（已移植到 `_cubic_roots_vec`）。

## 已提交

- `849c165` — 立方根移植（修闪蒸 NaN）+ 冻压力输运诚实收敛。
- 本次 Step 1（`fugacity_mole_deriv` + `natural_variables_residual` + 测试）与 Step 2（`natural_variables_jacobian` + `_flash_natural_variables` + 测试）待提交。
