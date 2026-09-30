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

## 下一步（后续会话）

**Step 2 — 自然变量 Newton 闪蒸**：用 `fugacity_mole_deriv` 组装雅可比，Newton 解 `(L, x, y)`（替代 `_flash_vec` 的逐次替换，二次收敛、泡点光滑）。
- 雅可比块：`dF_fug/dx = −(δ_ij/x_i + d lnφ_i^L/d ln n_j 经链式) `，`dF_fug/dy = +(δ_ij/y_i + …)`；质量平衡/闭合是线性的。

**Step 3 — 相态标志 + 主变量切换**：每格 `[pureLiquid, pureVapor, twoPhase]`（`L==1 / L==0 / 0<L<1`），两相区主变量用饱和度。

**Step 4 — 饱和度 chopping**：限制牛顿步的饱和度更新量（MRST `getSaturationIncrements`）。

**Step 5 — 接入 FIM**：把 `_implicit_compositional_full_step` 的主变量从 `(p, sw, z)` 换成 `(p, sw, sO/sG, x, y)`，残差换成质量平衡 + 逸度相等 + 闭合。

## 参考（照抄，不要自造）

- `reference_projects/MRST-main/autodiff/compositional/models/natvars/equationsNaturalVariables.m` — 残差组装。
- `reference_projects/MRST-main/autodiff/compositional/models/eos/EquationOfStateModel.m` — `equationsEquilibrium`（质量平衡 + 逸度 + 闭合）、`getPhaseFractionDerivativesPTZ`（解析闪蒸雅可比）、`solveCubicEOS`（立方根，已移植）。
- `reference_projects/MRST-main/autodiff/compositional/utils/cubicPositive.m` — 立方根（已移植到 `_cubic_roots_vec`）。

## 已提交

- `849c165` — 立方根移植（修闪蒸 NaN）+ 冻压力输运诚实收敛。
- 本次 Step 1（`fugacity_mole_deriv` + `natural_variables_residual` + 测试）待提交。
