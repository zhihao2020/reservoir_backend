# 00d Python与NumPy速成：够用就好

> 一句话本质：本系列所有「小练习」只需要 NumPy 的**四件武器**——数组、索引/切片、`@` 矩阵乘、
> `np.linalg` 求解。学会这四件，就能跑通每一章的代码块、读懂源码里每个 `phi`、`lam`、`rho_g` 在干什么。

**本章概念脉络**（每个概念都在解决一个问题）：

| 概念 | 它解决的问题 |
|---|---|
| 最小 Python 语法 | 怎么读懂 `import`/`def`/`for`/`if` |
| NumPy 数组 ndarray | 怎么把「一整个场」装进一个东西 |
| 索引与平坦下标 | 三维格子怎么和代码里的下标对应 |
| `@` vs `*` | 矩阵乘和逐元素乘，新手第一大坑 |
| `np.linalg` | 解方程、最小二乘、算残差大小 |
| ASCII 命名约定 | 希腊字母 \(\phi\)、\(\lambda\) 在代码里叫什么 |

## 零、问题：怎么把公式变成能跑的代码

前面三章是数学。要把它们跑起来，需要一点 Python。本章只讲**够读懂、够跑小练习**的最小集，
不追求系统。若你已会 Python，直接看第四节 `@` vs `*` 那一段即可。

## 一、第一个工具：最小 Python 语法

```python
import numpy as np            # 引入 NumPy，起个别名 np（几乎每章都写这行）

x = 3                         # 变量：把 3 存进名字 x
def f(t):                     # 函数：给一段计算起个名字，t 是输入
    return 2.0 * t + 1.0      # return 是输出

for k in range(4):            # 循环：k 依次取 0,1,2,3
    print(k, f(k))            # 打印

if x > 2:                     # 判断
    print("x 大于 2")
```

其中：`import` 引入别人写好的工具；`def` 定义函数；`for`/`if` 控制流程；`#` 后面是注释。

## 二、第二个工具：NumPy 数组 ndarray = 一个「场」

油藏里的压力场、饱和度场，都是一大堆数。NumPy 的**数组（ndarray）**就是用来装它们的：

```python
import numpy as np
p = np.array([10.0, 11.0, 12.0])      # 一维数组，3 个数
print(p.shape)                          # (3,)

phi = np.full((3, 3), 0.05)             # 3×3 的场，每个格子都是 0.05（孔隙度）
z   = np.linspace(0.0, 0.30, 16)        # 0 到 0.30 等距 16 个点
grid = np.zeros((2, 2))                 # 2×2 全 0

print(phi.sum(), phi.mean())            # 求和、平均——把整个场当整体操作
```

> **形象化**：数组就是一个「格子柜」。`shape` 告诉你柜子是几层几行几列；`.sum()` 是把所有抽屉里的数
> 加总——你不用一个个开抽屉，一句话就对整柜操作。这正是油藏模拟「向量化」的来源。

## 三、第三个工具：索引与平坦下标

场是三维的 `(nz, ny, nx)`（层、行、列），但代码里经常**拉平**成一维。这个拉平的规则全仓库统一：

\[
\mathrm{cell} = k\cdot ny\cdot nx + j\cdot nx + i.
\]

其中：\(i,j,k\) 分别是 x、y、z 方向的下标（都从 0 起）；`cell` 是拉平后的一维下标。

```python
grid = np.zeros((2, 3, 4))            # (nz, ny, nx) = (2,3,4)
grid[1, 2, 3] = 5.0                   # 第 k=1 层、j=2 行、i=3 列

flat = grid.ravel()                   # 拉平成一维
nz, ny, nx = 2, 3, 4
cell = 1 * ny * nx + 2 * nx + 3       # k*ny*nx + j*nx + i = 1*12+2*4+3 = 23
print(flat[cell])                     # 5.0 —— 和 grid[1,2,3] 是同一个抽屉
```

**轴顺序是 `(nz, ny, nx)`**：最外层、变化最慢的是 k（层号），`k=0` 是底层（z 最小）。别写成 `(nx,ny,nz)`，
否则仪表盘上的图和真实岩心会上下颠倒、行列对调（00 章测试题 1 就是考这个）。

## 四、第四个工具：`@` vs `*`（新手第一大坑）

NumPy 里 `*` 是**逐元素乘**，`@` 才是**矩阵乘**。两者天差地别：

```python
A = np.array([[2.0, 1.0], [1.0, 2.0]])
b = np.array([3.0, 3.0])

print(A * b)      # 逐元素：[[6,3],[3,6]]（形状不一致时报错或广播，几乎不是你要的）
print(A @ b)      # 矩阵乘：[9, 9]（第 00b 的 A@x = 每行加权求和）
```

> **形象化**：`*` 是「两个柜子抽屉一一对应相乘」；`@` 是「按第 00b 的电话簿规则，做加权求和」。
> 解方程组、算散度，全用 `@`。写错这个，结果会差出十万八千里，且不报错——排查第一反应就是看有没有把 `@` 写成 `*`。

## 五、第五个工具：`np.linalg` = 解方程、最小二乘、算大小

```python
import numpy as np

A = np.array([[2.0, 1.0], [1.0, 2.0]])
b = np.array([3.0, 3.0])
x = np.linalg.solve(A, b)          # 解 A x = b
print(x)                            # [1. 1.]

x, *_ = np.linalg.lstsq(A, b, rcond=None)   # 最小二乘（第 00b）
n = np.linalg.norm(A @ x - b)      # 残差的 2-范数（第 00c 的收敛判据）
print(n)                            # 0.0
```

其中：`solve` 解方程；`lstsq` 最小二乘；`norm` 算向量「大小」（残差多小）。

## 六、第六个工具：希腊字母去哪了 → ASCII 命名约定

代码里不能直接写 \(\phi\)、\(\lambda\)，本仓库用它们的英文拼写：

| 数学 | 代码 | 含义 |
|---|---|---|
| \(\phi\) | `phi` | 孔隙度 |
| \(\lambda_\alpha\) | `lam_w / lam_o / lam_g` | 水/油/气相流度 |
| \(\mu\) | `mu_w / mu_o / mu_g` | 相粘度 |
| \(\rho\) | `rho` | 密度 |
| \(\Delta\rho\,g\) | `rho_g` | 气相对液相的浮力头（**注意**：不是气体密度，见 01 章） |
| \(S_w,S_o,S_g\) | `sw / so / sg` | 三相饱和度 |
| \(k\) | `k` | 渗透率 |
| \(x,y\) | `x, y` | 液相/气相摩尔组成 |

还有两个常用操作：`.ravel()` 把多维拉成一维（配 `reshape` 还原），`np.maximum(a, b)` 逐元素取较大值
（用于把负饱和度夹回 0）。看到不认识的 `np.xxx`，去 NumPy 文档搜一下即可。

## 七、回头看：四件武器 + 命名，够跑全系列

| 需求 | 武器 |
|---|---|
| 装一个场 | `np.array / linspace / zeros / full` |
| 取某个格子 | 索引 `field[k,j,i]` + 平坦下标 `k*ny*nx+j*nx+i` |
| 算散度/解方程 | `@`、`np.linalg.solve`、`scipy.sparse.linalg.spsolve` |
| 最小二乘/残差 | `np.linalg.lstsq / norm` |
| 读懂希腊字母 | `phi/lam/rho_g/…` 命名表 |

## 八、小练习

```python
import numpy as np

# 1) 造 2×3×4 的场，取 k=1,j=2,i=3，验证拉平下标
field = np.arange(2 * 3 * 4).reshape(2, 3, 4)   # (nz,ny,nx)
i, j, k = 3, 2, 1
cell = k * 3 * 4 + j * 4 + i                     # k*ny*nx + j*nx + i
print("field[k,j,i] =", field[k, j, i], " flat[cell] =", field.ravel()[cell])

# 2) @ vs *
A = np.array([[2.0, 1.0], [1.0, 2.0]])
b = np.array([3.0, 3.0])
print("A @ b =", A @ b, "   A * b =", A * b)

# 3) 解方程 + 残差
x = np.linalg.solve(A, b)
print("解 x =", x, "  残差 =", np.linalg.norm(A @ x - b))

# 4) reshape 轴顺序的坑：写成 (nx,ny,nz) 会取错
wrong = np.arange(2 * 3 * 4).reshape(4, 3, 2)    # 错误顺序
print("正确 field[k,j,i] =", field[k, j, i], "  错误 reshape 同下标 =", wrong[i, j, k])
```

第 4 步两者不同——这就是 00 章反复提醒「reshape 用 `(nz,ny,nx)`」的原因。

## 九、章末测试题

**1.** `A * B`（NumPy 数组）和 `A @ B` 什么时候结果一样？

<details><summary>答案与解析</summary>

只有当 A、B 都是标量（单个数字），或一维向量「逐元素乘再求和」恰好等于点积时。对一般矩阵，`*` 是逐元素乘、
`@` 是矩阵乘，结果完全不同。解方程组、算散度一律用 `@`。

</details>

**2.** `reshape((nz, ny, nx))` 里，变化最慢（最外层）的是哪个轴？

<details><summary>答案与解析</summary>

`nz`（层号 k）。拉平下标 `cell = k*ny*nx + j*nx + i` 中 k 的系数最大，所以 k 变化最慢。

</details>

**3.** 判断：`field[k, j, i]` 的 `k=0` 是底层。

<details><summary>答案与解析</summary>

对。z 向上为正、`k=0` 对应 z 最小，是底层。所以羽流「沉到底部」在数组里是「sg 在 k=0 那层最大」。

</details>

**4.** 代码里 `rho_g` 是「气体密度」吗？

<details><summary>答案与解析</summary>

不是（这是个误导人的名字）。`rho_g` 是气相对液相的**浮力头** \((\rho_{\mathrm{CO_2}}-\rho_{\mathrm{oil}})g\)，
单位 Pa/m，>0 表示气比油重、下沉。详见 01 章。

</details>

## 十、延伸阅读

- NumPy 官方快速入门（`numpy.org` 的 Quickstart）
- 下一篇（零基础读完这四篇后）：[01-油藏渗流基础](01-油藏渗流基础.md)
