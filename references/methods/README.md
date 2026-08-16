# 方法学参考（只读，禁止运行时 import）

本目录存放 GitHub / 论文中的 **ES-MDA / 集成平滑 / 历史拟合** 开源实现与文献副本，用于对照与抽取算法思想。

## 合规

| 规则 | 说明 |
|------|------|
| **禁止** `import` 本目录或 `references/upstream` | 与 `软件要求` / 项目合规一致 |
| **禁止** 原样拷贝上游文件进 `reservoir_backend/` | 可改写思想，用自有命名与接口 |
| 本目录为 **研究参考** | 可删可更新；产品代码不依赖此路径 |

## 已下载

| 路径 | 来源 | 许可（以仓库为准） | 可抽取点 |
|------|------|-------------------|----------|
| `iterative_ensemble_smoother/` | [equinor/iterative_ensemble_smoother](https://github.com/equinor/iterative_ensemble_smoother) | 见上游 | α 归一化 \(\sum 1/\alpha_i=1\)；分批参数；子空间/Cholesky 求逆；观测扰动 |
| `pyesmda/` | [gitlab antoinecollet5/pyesmda](https://gitlab.com/antoinecollet5/pyesmda) | MIT | 面向对象 ES-MDA；MD/DD 局部化相关阵；参数边界；集成膨胀 |
| `dass/` | [equinor/dass](https://github.com/equinor/dass) | 见上游 | 教学向 ES；taper/localization 思路；与 Emerick 2013 衔接说明 |
| `esmda-seismic/` | [rodrigoext/esmda](https://github.com/rodrigoext/esmda) | 见上游 | 地震相 + ES-MDA 应用示例（非核心求解） |
| `genES-MDA-author-copy.pdf` | genES-MDA 作者稿 | 文献 | 通用 ES-MDA 软件包设计与自适应 MDA |
| Emerick & Reynolds 2013, *Computers & Geosciences* | ES-MDA 原始论文 | 文献 | 本地可放 `references/`，不进 git |

### 克隆命令（可复现）

```bash
cd references
mkdir -p methods && cd methods
git clone --depth 1 https://github.com/equinor/iterative_ensemble_smoother.git
git clone --depth 1 https://github.com/equinor/dass.git
git clone --depth 1 https://github.com/rodrigoext/esmda.git esmda-seismic
git clone --depth 1 https://gitlab.com/antoinecollet5/pyesmda.git
```

## 抽取进本仓库产品的逻辑（自研实现）

落地文件：`reservoir_backend/inverse/esmda.py`、`inverse/ensemble.py`。旧 `pipeline/` 已删除。

| 实践 | 文献/上游 | 本仓库实现 |
|------|-----------|------------|
| ES-MDA 多步同化 + \(\alpha\) 膨胀观测噪声 | Emerick & Reynolds 2013 | `normalize_alpha_weights` + 逐步更新 |
| \(\sum_i 1/\alpha_i = 1\) | equinor / 论文 | 默认等权 \(\alpha_i=N_a\) 归一化 |
| log 参数空间 + 上下界裁剪 | 储层 HM 常规 / pyesmda bounds | `log(k)` 更新后 `clip` |
| 对角 \(R\) 预条件改善条件数 | equinor esmda_inversion 讨论 | `solve_obs_system` 用 \(\mathrm{diag}(R)^{-1/2}\) |
| 单步 ES（α=1） | dass / Evensen | `algorithm: es` |
| ES-MDA 等权 α | Emerick 2013 / Equinor IES | `algorithm: esmda` |
| 几何 α（先稳后狠） | Emerick 变体 | `algorithm: esmda_geo` |
| 受限步自适应 α、自动停 | pyesmda ES-MDA-RS / Le 2016 | `algorithm: esmda_rs` |
| 阻尼迭代 ES | Chen & Oliver LM-EnRML / IES | `algorithm: ies` |
| 限时随机搜旋钮 | AutoGluon fit(time_limit) | `inverse.hpo.run_hpo` |
| 更新后集成轻微膨胀 | Anderson 2007 / pyesmda | `inflate_ensemble` |
| hold-out 加权混合、变差丢掉 | AutoGluon ensemble selection | `calibrate_auto(blend=True)` |
| 井–参数距离相关局部化 | Gaspari–Cohn / Equinor LocalizedESMDA | 写了，n_θ 大时再开 |

## 未抽取（有意）

- 上游包名、类名、文件布局
- 地震 VAE、ERT 工程栈
- 百万参数磁盘分批（网格规模小时不需要）

## 建议阅读顺序

1. Emerick & Reynolds 2013 PDF（本仓库 references 根目录）
2. equinor `esmda.py` 文档字符串 + `normalize_alpha`
3. pyesmda `ESMDA` 类 API 与 localization
4. 对照 `reservoir_backend/inverse/esmda.py` 的自研实现

没有 Optuna / CMOST / 网格搜索 K。那些是正演调参。要学的是先验、MDA 日程、膨胀、局部化，以及测点设计，不是外层 HPO。
