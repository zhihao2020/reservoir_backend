# 07b 自然变量FIM：用逸度相等消除泡点奇异

> 一句话本质：07 章的组分前向在**泡点**反复不收敛，是因为闪蒸 \(V(z)\) 在泡点处有个**尖角**（导数爆炸）；
> 自然变量法把主变量从 \((p,sw,z)\) 换成 \((p,sw,sO,sG,x,y)\)，用**光滑的逸度相等**方程取代闪蒸，
> 泡点就从「拐角」变成「坡」，牛顿可以一步跨过去。本质没变——相平衡仍是相平衡，只是从「算出来的函数」
> 变成了「要满足的方程」。

**本章概念脉络**（每个概念都在解决一个问题，按出现顺序）：

| 概念 | 它解决的问题 |
|---|---|
| 泡点奇异 | 07 章前向为什么在泡点反复不收敛 |
| 换主变量 + 逸度相等 | 把相平衡从「拐角的闪蒸函数」变成「光滑的方程」 |
| 变量切换 + 相态标志 + 掩码 | 每个格子的相态不同，怎么只保留它真正有的变量 |
| 相稳定测试 | 单相格该不该「长出」另一相 |
| 相切换 | 长出新相 / 收掉消失相，怎么在牛顿循环里做 |
| chopping（截步） | 牛顿一步别跨过相边界 |
| Schur 约简 | 逸度+闭合方程太占地方，怎么压掉一半 |
| 解析稀疏 Jacobian | 逸度导数两种、Euler 关系、\(G_i\) 径向项 |
| 接入生产前向 | 怎么变成 `forward.model: natural_variables` |

## 零、问题：07 章前向为什么在泡点反复不收敛？

07 章的组分前向用主变量 \((p,sw,z)\)：\(z\) 是**总**摩尔分数（整体组分），每个牛顿迭代里要跑一次
**闪蒸** \(V(z)\)——回答「这个整体组分 z，在压力 p 下，会劈成几相、气相占多少摩尔分数 V」。

这套做法在**泡点**附近反复出问题：不收敛、假收敛、牛顿卡死。这是「冻压力输运 / 假收敛一直绕不过去的根子」。
本章先搞清楚**为什么闪蒸在泡点会卡**，再给出解法。

> **先复习两个词**（06 章讲过，这里精确化）：
> **闪蒸 flash**＝给定 p、T、总组成 z，算混合物分成几相、每相占多少、组成和密度。
> **泡点 bubble point**＝液相刚要析出第一个气泡的边界（总 CO₂ 分数刚超过某阈值，\(V\) 从 0 跳到 >0）。

## 一、第一个问题：闪蒸 \(V(z)\) 在泡点有个「拐角」

把气相摩尔分数 \(V\) 看成总 CO₂ 分数 \(z\) 的函数 \(V(z)\)，画出来是：

- 在泡点**左侧**（CO₂ 少）：全溶，\(V=0\)，平平的一条线；
- 越过泡点：\(V\) **突然**从 0 翘起来（出现自由气）。

**问题在哪？** 在泡点这一点，\(V(z)\) 有一个**尖角**（像 \(f(x)=\max(0,x)\) 在 \(x=0\) 处）：左边的导数是 0，
右边一过泡点导数就急剧变大，\(\partial V/\partial z\) 在这个点上**趋于无穷/不连续**。

07 章对闪蒸项的 Jacobian 用的是**数值差分**（第 00c：把 z 扰动一点点看 V 变多少）。在尖角处，这个
「导数」要么算出来是无穷、要么剧烈跳变——**雅可比奇异 → 牛顿一步跨出去就错 → 失速、假收敛**。

> **形象化**：闪蒸 \(V(z)\) 像一条平路突然接上一段陡坡。牛顿法靠「切线」指路，在平路尽头（尖角）切线的
> 坡度一会儿是 0、一会儿是无穷大，指针乱指——牛顿就卡死在这里。泡点不是「难算」，是「切线根本不存在」。

## 二、第二个工具：换主变量，用逸度相等替代闪蒸

**解法思路**：既然闪蒸 \(V(z)\) 这个「拐角函数」是病灶，就**别再用它当主变量的函数**，改成直接携带
「相平衡的结果」本身。

**主变量 primary variables**（牛顿法直接求的那组未知量）从 \((p,sw,z)\) 换成：

\[
(p,\ sw,\ sO,\ sG,\ x,\ y).
\]

其中：\(p\) 压力；\(sw\) 水饱和度；\(sO,sG\) 油（液）/气饱和度（**独立主变量**，不再由闪蒸派生）；
\(x,y\) 液相/气相摩尔组成（各 ncomp−1 个独立分量，最后一个由「和为 1」隐含）。

**关键**：不再用闪蒸算「怎么劈分」，而是把热力学平衡写成一条**方程**——每个组分的**逸度**在气液两相
相等：

\[
\ln(x_i\varphi_i^L) = \ln(y_i\varphi_i^V),
\qquad i=1,\ldots,\mathrm{ncomp}.
\]

其中：\(x_i,y_i\) 组分 i 的液/气摩尔分数；\(\varphi_i^L,\varphi_i^V\) 液/气逸度系数（PR 方程解析给，
06 章已定义）。**逸度 fugacity**＝组分「想逃离这一相」的倾向；平衡时两相谁也不比谁更想逃，故相等。

**为什么这就光滑了？** 逸度相等是一条**处处光滑**的方程（不像 \(V(z)\) 在泡点有尖角）。泡点穿越从
「跨过一个拐角」变成「解一条光滑方程」——**拐角变坡**，牛顿的切线重新有意义。

同时，总组分 \(z\) **退化为派生量**：

\[
z_c = \frac{m_c}{\sum_j m_j},\qquad m_c = \frac{sO\cdot x_c}{v_l} + \frac{sG\cdot y_c}{v_g}.
\]

其中：\(m_c\) 组分 c 的总摩尔密度；\(v_l,v_g\) 液/气摩尔体积（EOS）。**主变量不再含 z，z 只是事后算出来看。**

> **形象化**：旧做法每次问「这团东西会怎么分家」（闪蒸，答案在泡点处突变）；新做法直接把「各家各有多少、
> 各家成分是什么」当作变量，然后写一条「两家谁也不想再搬」的平衡方程。前者是查一个带拐角的表，后者是解一条光滑的方程。

## 三、第三个工具：变量切换 + 相态标志 + 掩码

每个格子的相态不同（纯液 / 纯气 / 两相），主变量个数也不同。代码用三样东西管这个：

1. **相态标志 phase flag**：给每格贴一个标签。`flags_from_state(sO,sG)` 按饱和度和容差判定：
   - 纯液（pure liquid）：\(sG\le\) 容差；
   - 纯气（pure vapor）：\(sO\le\) 容差且非纯液；
   - 两相（two-phase）：其余。

2. **约简变量集 reduced variables**：单相格里，消失的那一相的变量**根本不进系统**，而不是弱耦合着。
   - 纯液格：\(y=x\)、\(sO=1-sw\)、\(sG=0\)，所以 y 不是主变量，只剩 \((p,sw,x)\)；
   - 纯气格：\(x=y\)、\(sG=1-sw\)、\(sO=0\)，只剩 \((p,sw,y)\)。
   - **隐含变量 implied variables / 折叠 fold**：这些「= 别的变量」的量不独立，被「折叠」进保留的变量里。

3. **掩码 mask**：一张只含 True/False 的「开关表」。`var_mask` 决定哪些主变量参与本次求解，
   `eq_mask` 决定哪些方程参与。全局雅可比只对「开着的变量 × 开着的方程」组装。

> **形象化**：掩码是教室的点名册，只勾「今天来了的人」。纯液格没气相，就把 y、sG 从名单上划掉——
> 不求解根本不存在的量，这是自然变量法「变量数随相态变」的核心。

## 四、第四个工具：相稳定测试（Michelsen 切平面距离 TPD）

**问题**：一个单相格（比如全油、\(sG=0\)），会不会其实**该分相**？不能永远当它单相——注入 CO₂ 后它可能
要析出气泡。

**方法**：**相稳定测试 phase stability test**——严格判断「混合物该稳定成单相，还是该分相」。用的是
Michelsen 的**切平面距离 tangent-plane distance (TPD)**：衡量「把这一相切成两相，能降低多少自由能」。
若存在一个「萌芽相」（无限小的一丁点另一相）能让自由能下降，则单相**不稳定**，该分相。

`phase_stability_test(z, p)` 做两轮逐次代入（汽型萌芽、液型萌芽），各收敛到一个驻点，判据：

\[
S = \sum_i w_i \ \begin{cases}\le 1 & \text{稳定，保持单相}\\ > 1 & \text{不稳定，存在萌芽相}\end{cases}
\]

其中：\(w_i\) 萌芽相的组成（由 \(w_i = z_i\varphi_i(z)/\varphi_i(w)\) 迭代得到）。返回 `(stable, x, y)`：
稳定的格子 \(x=y=z\)；不稳定的返回萌芽相组成。

> **形象化**：相稳定测试像「试一下」——往一杯纯水里「试放」一粒冰晶，看它会不会长大。会，说明该结冰
> （分相）；不会，说明水能一直稳定是单相。Michelsen 就是那个「试放一粒」的数学步骤。

## 五、第五个工具：相切换（插萌芽相 / 收掉消失相）

相稳定测试给出「该不该变」，**相切换 flash_phases** 执行「怎么变」，在牛顿循环每次迭代后做：

- **纯液格不稳定** → 插入 `saturationEpsilon` 的**萌芽汽**（\(y\) 取稳定性测试给出的萌芽组成），转两相；
- **纯气格不稳定** → 插入萌芽液；
- **两相格**里 \(sO\le0\) 或 \(sG\le0\) → 某一相消失了，**收成**单相。

其中：**萌芽相 incipient phase**＝刚要出现的那一丁点相（无限小，用一个极小饱和度代表）。

> **形象化**：相切换是「把该长出来的相，先播一颗芝麻大的种子进去」，让牛顿迭代有东西可算；
> 反过来，某相萎缩到 0 就把它「擦掉」，恢复单相。这一增一删，让变量集合始终跟着真实相态走。

## 六、第六个工具：chopping（截步）—— 牛顿一步别跨过相边界

牛顿法一步可能跨得太大，直接从「纯液」跳到「两相还带负饱和度」。**chopping / 截步（松弛 damping）**就是
给每步更新量设上限，逐格把变化砍到安全范围内：

\[
|\Delta s| \le \texttt{\_DS\_MAX}=0.2,\qquad |\Delta x|,|\Delta y|\le\texttt{\_DX\_MAX}=0.1,\qquad |\Delta p|\le\texttt{\_DP\_MAX}=2\times10^6.
\]

超过上限就整体按比例缩小这一步，再重新归一化（保证 \(\sum x=1\)、饱和度非负且和约束成立）。

> **形象化**：牛顿法像一脚油门踩到底，chopping 是装了个限速器——每步最多变这么多，防止一步冲出相边界
> 翻车。代价是可能要多踩几脚，但每脚都稳。

## 七、第七个工具：Schur 约简 30n → 15n

两相格的主变量是 4 + 2(ncomp−1)：`(p, sw, sO, sG)` 加 `x`、`y` 各 13 个独立分量 = **30 个/格**。
整个系统 30n 阶，太肥。但**逸度相等 + 闭合**这些方程，可以和一批变量**先消掉**。

**Schur 补 / 约简 Schur complement / reduction**（07 章 §2.6 对井 BHP 用过同一个思想，这里用到相平衡上）：
把一个线性系统的一部分未知先消掉，只解更小的那部分。代码 `keep_var_mask` / `keep_eq_mask` 决定谁留谁走：

- **保留**：组分摩尔平衡 + 水方程，以及变量 \((p, sw, sO, x[1..12])\) —— 每格 **15 个**；
- **消去**：逸度 + 闭合方程，以及变量 \((sG, x[0], y)\)。

`schur_solve` 做 \(A_{\mathrm{red}} = B - C\,E^{-1}D\)，先解 15n 的保留系统，再**恢复**被消的变量。因为
被消的 E 块是**格内局部**的（两相格 15×15 块对角），求逆便宜，所以整体又快又不破坏稀疏性。

> **为什么要把 \(x[0]\) 放进被消块？** 逸度块（14×14）在平衡系统里是奇异的（组分有一个不独立），把
> \(x[0]\) 一并消掉，逸度块才非奇异——这是 MRST 同款做法，不是随手选的。
>
> **工程直觉**：30n → 15n，矩阵阶数减半。稀疏求解的代价大致随非零数增长，减半阶数通常能换来接近
> 一个数量级的提速。Schur 约简就是「花一点代数，把大问题压成小问题」。

## 八、第八个工具：解析稀疏 Jacobian —— 逸度导数两种、Euler 关系、\(G_i\) 径向项

自然变量的全局雅可比 `sparse_jacobian_full` 是**解析**组装的（不像 07 章对闪蒸项做数值差分）。这里藏着
一个最容易踩的坑：**逸度系数的导数有两种，别混**。

先分清两个对象：

1. **无约束摩尔分数导数** \(\partial\ln\varphi_i/\partial x_j\)（`_fugacity_frac_deriv_analytic`）：
   把每个 \(x_j\) 当独立变量（\(\sum x\) 自由）。用于**闪蒸内部牛顿**（x/y 是自由向量，闭合方程显式约束）。

2. **摩尔数导数** \(d\ln\varphi_i/d\ln n_j\)（`fugacity_mole_deriv`）：
   保持 \(\sum x=1\) 时，逸度系数对「摩尔数」的导数。用于 **FIM 外层雅可比**（x 是单纯形上的主变量）。

两者差一个**径向规范项 radial gauge**：

\[
G_i = \sum_k x_k\,\frac{\partial\ln\varphi_i}{\partial x_k},
\qquad
\frac{d\ln\varphi_i}{d\ln n_j} = x_j\Bigl(\frac{\partial\ln\varphi_i}{\partial x_j} - G_i\Bigr).
\]

其中：\(G_i\) 之所以非零，是因为**未归一化**的逸度系数对 x **不是 0 次齐次**（A、B 随 x 非线性缩放）。

**Euler 关系（0 次齐次）Euler identity**：把**所有**摩尔数同比例放大，逸度系数不变，数学上就是

\[
\sum_j \frac{d\ln\varphi_i}{d\ln n_j} = 0.
\]

这是摩尔数导数最硬的一条自检：`test_fugacity_mole_deriv_satisfies_euler` 就是断言它 < 1e-5。
**误把 \(\partial\ln\varphi_i/\partial x_j\) 当 \(D[i,j]/x_j\) 用，会漏掉 \(G_i\)，雅可比错约 1**（FD 检验能抓到）。

> **形象化**：想象调一杯糖水。摩尔数导数是「按比例加水加糖，浓度不变」的视角（0 次齐次，和为 0）；
> 摩尔分数导数是「单独多撒一把糖」的视角。两者差的那点「径向」项 \(G_i\)，就是「单独加料会改变整体浓度」
> 的那部分。闪蒸内部用后者，FIM 外层用前者，拿错视角，雅可比就错一个量级。

**逸度残差用线性型，不用 log 型**：FIM 里逸度相等残差写成 \(x\cdot\varphi^L - y\cdot\varphi^V\)（线性型），
而不是 \(\ln(x\varphi^L)-\ln(y\varphi^V)\)（log 型）。两者根相同，但 log 型的导数含 \(1/x\)、\(1/y\)，
在微量相（\(y\to0\)，纯液格边界）会**爆炸**；线性型的导数 \(\varphi^L + x\cdot\partial\varphi^L/\partial x\)
处处有限。**换形式是为了 Jacobian 在相边界处不炸。**

## 九、接入生产前向：`forward.model: natural_variables`

`forward_saturations` 里 `name == "natural_variables"` 分发到 `_forward_natural_variables_saturations`
（照 `_forward_compositional_full_saturations`，**耦合**——压力也作主变量一起解，无冻压力模式），再套
`_natural_variables_adaptive` 自适应子步（照 `_implicit_compositional_full_adaptive`，lazy import
`natural_variables_step` 避免循环依赖）。输出同 `(sw, so, sg, z_co2)`。

- **耦合 coupled**：压力、饱和度、组成一起解（自然变量是耦合模型，天然如此）。
- **冻压力 frozen pressure**：07 章的组分输运有个「压力固定成给定场」的模式；自然变量**没有**这个模式
  ——它就是要靠压力一起解，才能跨过泡点。

CLI：`python -m src … --model natural_variables`（或 `forward.model: natural_variables`）。

## 十、回头看：old `(p,sw,z)+flash` vs new `(p,sw,sO,sG,x,y)+fugacity`

| 情形 | 主变量 | 相平衡 | 泡点处 |
|---|---|---|---|
| 07 章 old | \((p,sw,z)\) | 闪蒸 \(V(z)\)（拐角函数） | \(\partial V/\partial z\) 奇异 → 失速 |
| 07b new | \((p,sw,sO,sG,x,y)\) | 逸度相等（光滑方程） | 主变量切换 + 截步，平滑跨越 |

**一条本质**：相平衡这件事没变——都是「组分在气液两相间达到逸度相等」。变的只是**把它当「算出来的函数」
还是「要满足的方程」**。前者在相边界处有拐角（闪蒸），后者光滑（逸度相等），所以后者能让牛顿跨过泡点。

## 十一、代码对照

| 概念 | 函数（文件） |
|---|---|
| 主变量布局 pack/unpack | `pack_full` / `unpack_full`（`natural_variables.py` 闭包内） |
| 相态标志 | `phase_flags`（`pr_eos.py:847`）、`flags_from_state`（`natural_variables.py`） |
| 变量/方程掩码 | `var_mask` / `eq_mask`（`natural_variables.py`） |
| 隐含变量折叠 | `fill_inactive` / `full_comps`（`natural_variables.py`） |
| 逸度相等残差（FIM 用线性型 \(x\varphi^L-y\varphi^V\)） | `residual_full` 内 `r_fug_`（`natural_variables.py`） |
| 相稳定测试 | `phase_stability_test` / `_stability_check`（`pr_eos.py:683/651`） |
| 相切换 | `flash_phases`（`natural_variables.py`） |
| chopping 常量 | `_DS_MAX/_DX_MAX/_DP_MAX`（`natural_variables.py:44–46`） |
| Schur 约简 | `schur_solve` / `keep_var_mask` / `keep_eq_mask`（`natural_variables.py`） |
| 解析稀疏全局 Jacobian | `sparse_jacobian_full`（`natural_variables.py`） |
| 无约束摩尔分数导数 | `_fugacity_frac_deriv_analytic`（`pr_eos.py:717`） |
| 摩尔数导数（Euler 关系） | `fugacity_mole_deriv`（`pr_eos.py:770`） |
| 格内平衡残差 / Jacobian | `natural_variables_residual` / `natural_variables_jacobian`（`pr_eos.py:812/881`） |
| 自然变量牛顿闪蒸 | `_flash_natural_variables`（`pr_eos.py:1015`） |
| 闪蒸解析导数 ∂(V,x,y)/∂(p,z) | `flash_derivatives_ptz`（`pr_eos.py:939`） |
| \((sw,L,x,y)\to(sO,sG,\rho)\) 桥接 | `natural_variables_state`（`pr_eos.py:860`） |
| 生产分发 | `forward_saturations` `name=="natural_variables"`（`forward.py:425`）→ `_forward_natural_variables_saturations` / `_natural_variables_adaptive`（`forward.py:1597/1538`） |

## 十二、小练习

本章的「可观察检查点」就是仓库里的回归测试——每一条都是本章某个概念的直接落地：

```python
# 1) 摩尔数导数满足 Euler 关系 Σ_j D = 0（0 次齐次）
import numpy as np
from src.core.pr_eos import _Z_OIL_DEAD, fugacity_mole_deriv

x = _Z_OIL_DEAD.copy()[None, :]                    # 死油基底，14 组分 (1,14)
D = fugacity_mole_deriv(x, np.array([20e6]), phase="liq")   # (1, 14, 14)，D[i,j]=d lnφ_i/d ln n_j
print("max |Σ_j D| =", float(np.abs(D.sum(axis=2).max())))   # 应 ≈ 0（<1e-5）
```

```python
# 2) 残差在闪蒸解处≈0：质量平衡 + 逸度相等 + 闭合
from src.core.pr_eos import flash_direct_full, natural_variables_residual, _Z_OIL_DEAD, _CO2_IDX
import numpy as np

zs = np.array([0.6])                               # 总 CO₂ 摩尔分数，越过泡点（≈0.38）
z = np.outer(1.0 - zs, _Z_OIL_DEAD)                # 死油按比例缩小
z[:, _CO2_IDX] += zs                               # 加入 CO₂（_CO2_IDX=4），z 和为 1
p = np.full(zs.size, 20e6)
V, x, y = flash_direct_full(z, p)
L = 1.0 - V                                        # 液相摩尔分数
res = natural_variables_residual(L, x, y, z, p)
print("|质量平衡| =", float(np.abs(res[:, :14]).max()),
      " |逸度相等| =", float(np.abs(res[:, 14:28]).max()),
      " |闭合| =", float(np.abs(res[:, 28]).max()))
```

对应的 pytest 检查点（`python -m pytest tests/test_compositional_fim.py -m "not slow"`）：

| 想验证的概念 | 测试 |
|---|---|
| Euler 关系 | `test_fugacity_mole_deriv_satisfies_euler` |
| 残差在闪蒸解处≈0 | `test_natural_variables_residual_vanishes_at_flash` |
| 两套模型对同一单格一致 | `test_natural_variables_step_matches_reference`、`test_natural_variables_multi_cell_matches_reference` |
| 死油起注、泡点长出自由气 | `test_natural_variables_phase_transition` |
| 羽流 | `test_natural_variables_plume`（`test_bottom_plume_sinks` 是 slow 的 30 天验收） |

## 十三、章末测试题

**1.** 为什么 \(V(z)\) 在泡点处会让牛顿失速？自然变量法如何把「拐角」变「坡」？

<details><summary>答案与解析</summary>

\(V\) 作为 \(z\) 的函数在泡点从 0 跳到 >0，有尖角，\(\partial V/\partial z\) 趋于无穷/不连续，
数值雅可比奇异 → 牛顿一步失速。自然变量把主变量换成 \((p,sw,sO,sG,x,y)\)，用**光滑的逸度相等**方程
替代闪蒸，泡点穿越变成「解一条光滑方程 + 主变量切换」。

</details>

**2.** 纯液格的主变量是哪几个？\(y\) 和 \(sG\) 去哪了？

<details><summary>答案与解析</summary>

纯液格只剩 \((p,sw,x)\)。\(y=x\)、\(sO=1-sw\)、\(sG=0\) 都是隐含变量，被「折叠」进保留变量里，
不再独立求解（变量切换 + 掩码把它们从系统里划掉）。

</details>

**3.** Michelsen 稳定性判据 \(S=\sum w_i\) 的含义？\(S>1\) 说明什么？

<details><summary>答案与解析</summary>

\(S\) 是萌芽相组成之和，等于切平面距离的一个标度。\(S>1\) 表示存在一个萌芽相能让自由能下降，
单相不稳定、该分相；\(S\le1\) 则稳定保持单相。

</details>

**4.** Schur 约简消掉了哪些方程和变量？为什么必须把 \(x[0]\) 放进被消块？

<details><summary>答案与解析</summary>

消掉逸度 + 闭合方程，以及变量 \((sG, x[0], y)\)；保留组分摩尔平衡 + 水方程和 \((p,sw,sO,x[1..12])\)。
把 \(x[0]\) 放进被消块是为了让逸度块（14×14）非奇异——平衡系统里组分有一个不独立，消掉一个才可解。

</details>

**5.** 摩尔数导数 \(d\ln\varphi_i/d\ln n_j\) 与摩尔分数导数 \(\partial\ln\varphi_i/\partial x_j\) 差什么？
\(\sum_j D=0\) 从哪来？

<details><summary>答案与解析</summary>

差径向规范项 \(G_i=\sum_k x_k\partial\ln\varphi_i/\partial x_k\)：\(d\ln\varphi_i/d\ln n_j=x_j(\partial\ln\varphi_i/\partial x_j-G_i)\)。
\(\sum_j D=0\) 是 Euler 关系——把所有摩尔数同比例放大，逸度系数不变（0 次齐次）。误用两者，雅可比错约 1。

</details>

**6.** 为什么逸度残差用线性型 \(x\varphi^L-y\varphi^V\)，而不是 log 型？

<details><summary>答案与解析</summary>

两者根相同，但 log 型的导数含 \(1/x\)、\(1/y\)，在微量相（\(y\to0\)，纯液边界）爆炸；
线性型的导数处处有限。换形式是为了 Jacobian 在相边界处不炸。

</details>

## 十四、延伸阅读

- `reference_projects/MRST-main`：`equationsNaturalVariables.m`（残差组装）、`EquationOfStateModel.m`
  （`equationsEquilibrium`、`performPhaseStabilityTest`、`getPhaseFractionDerivativesPTZ`）
- Michelsen (1982), *The isothermal flash problem. Part II: Phase-split calculation*（切平面距离 TPD）
- Voskov & Tchelepi, natural-variables compositional 系列论文
- 下一篇：[08-管线与在线会话](08-管线与在线会话.md)
