# Reservoir Backend

300 mm 立方试块的**实验室多相渗流数字孪生反演后端**。

```text
探头 CSV → 控制 u(t) → 先验 m → 正演 F → 观测算子 H → LM(θ) → \(\hat K\) → F(\(\hat m\))
```

饱和度是动态状态，不是物性场。反演默认 2-region log K，不是逐格 \(K\)。一次实验只交一份后验场，不要和 CMG 全场逐格去对。

设计：[docs/target_architecture.md](docs/target_architecture.md)。假设：[docs/model_assumptions.md](docs/model_assumptions.md)。验收：[docs/cnpc_acceptance.md](docs/cnpc_acceptance.md)。

## 安装

```bash
python -m pip install -e ".[dev]"
```

或 `python -m reservoir_backend ...`。

## 用户怎么用

入口是 `reservoir apply`。可运行算例在 [examples/](examples/README.md)。

### 有探头读数（正式）

1. 选算例：层状用 `examples/two_layer/`，已知通道填砂用 `examples/channel/`。
2. 按 `observations_template.csv` 填测点。探头名必须和 YAML 里 `sensors` 一致。
3. 在 YAML 写 `experiment.observations: observations.csv`（`case_from_csv.yaml` 已写好）。
4. **不要**加 `--demo`：

```bash
reservoir apply examples/two_layer/case_from_csv.yaml --output results/examples/two_layer
```

仓库里的 `observations.csv` 是自洽正演造的示例读数，用来先跑通。换岩样后用自己的表覆盖；`sigma` 用该探头重复性，不要照抄 2 kPa / 0.04。

层数不确定、又没有通道图时加 `--auto`：在均匀 / 2 层 / 3 层里按 hold-out 选构造，不搜格子 \(K\)。给了 `region_map` 就用图，不要猜。

### 没有测点：`--demo` 是自检，不是正式用法

没有 CSV 时直接 `apply` 会报错。加上 `--demo`，程序会在这块网格上埋一个已知真值（默认两层对比度 10；通道算例用通道图），用本正演算假测点，再走同一套反演，检查能不能收回。

```bash
reservoir apply examples/two_layer/case.yaml --demo --output results/examples/two_layer_demo
```

这是装机和验收用的（`docs/cnpc_acceptance.md`）。`--demo` 不会猜你那块石头。

| 场合 | 命令 |
|------|------|
| 实验室有探头 CSV | `apply <case_from_csv.yaml>` |
| 装好软件、验收、看输出 | `apply <case.yaml> --demo` |
| 既没 CSV 也没 `--demo` | 报错 |

### 跑完看什么

`results/examples/two_layer/`（或你给的 `--output`）里：

| 文件 | 含义 |
|------|------|
| `k.npy` | 拟合后的渗透率 |
| `pressure.npy`、`sw.npy`、`so.npy` | \(F(\hat m)\) 重建的三维场 |
| `apply.json` | \(\theta\)、拟合 / hold-out / 预报 |
| `figures/posterior_fields_xz.png` | 剖面图 |

有 `--demo` 时报告里多 `acceptance`（对比度是否大约 8–12）。有实测 CSV 时只看测点拟合和预报；后验 \(K\) 是本正演 \(F\) 下的等效渗透率。

### 测点 CSV

SI（秒、Pa、饱和度 0–1）：

```text
time_s,sensor,kind,value,sigma,holdout
100,P_in_bot,pressure,1.61e5,2000,0
100,S_mid_top,saturation,0.24,0.04,0
```

实验室单位（分钟、kPa）见 `examples/two_layer/observations_kpa_min.csv`。也可在 YAML 设 `observation_time_unit` / `observation_pressure_unit`。

不要把定压井的流量当观测。

**能用在哪：** 300 mm 试块、和配置一致的入口/出口与内部测点。探头 6 mm，\(H\) 在插值场上做球平均。  
**不能用在哪：** 要求 \(p/S/K\) 和 CMG 全场逐格相等。

## 其他命令

`apply` 不够用时再碰这些。`invert` 没有观测时用 `--self-check`，和 `apply --demo` 是同一类自检。

```bash
reservoir validate examples/lab/lab_30cm.yaml
reservoir simulate examples/lab/lab_30cm.yaml --output results/sim
reservoir invert   examples/lab/lab_30cm.yaml --self-check --output results/inv
reservoir invert   examples/lab/lab_30cm.yaml --auto --time-limit 120
reservoir forecast examples/lab/lab_30cm.yaml --output results/fc
reservoir synthetic --output results/syn
```

验收签字仍可用 `examples/lab/lab_apply.yaml --demo`。全网格 10 mm / \(30^3\) 是 `examples/lab/lab_30cm.yaml`，比示例慢。

## 当前物理（P0）

- Model A：单相 Cartesian TPFA
- Model B / D：黑油油水 IMPES（表面体积 \(\varphi b_\alpha S_\alpha\)）。实验室 \(B=1,c=0\)
- Model C：三相 IMPES（`*SWT`+`*SLT` + Stone II；活油 \(G^s=\varphi(b_g S_g+R_s b_o S_o)\)）
- 毛管：Brooks–Corey / van Genuchten / none，由 case 显式选择（实验室默认 Brooks–Corey）
- 端口：定流量（地面）或定压，不是默认 Peaceman 油藏井
- 反演：默认 2-region log K 或已知图的 contrast，LM 拟合井史。PVT 不进 \(\theta\)
- 后验：\(K\) 均值/标准差，以及 \(F(m_{\mathrm{post}})\) 的 \(p,S_w,S_o\)
- 时间：内部一律秒

## 明确未做

济阳 GEM 组分牌、水相组分、热、EnKF、MPFA、动态 AMR、神经网络代替正演。等温 EXAMPLE 组分（C1–nC10）是可选 \(F\)：`physics.model: compositional`，`examples/compositional/comp_example.yaml`。实验室三相仍可用独立 Corey。

`validation/` 下的离线 CMG/GEM 尺子不再接入本内核。

## 测试

```bash
pytest -q
```
