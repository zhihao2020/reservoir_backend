# 示例：实验室 300 mm 试块怎么跑

用户入口是 `reservoir apply`。一次反演，一份 \(F(m_{\mathrm{post}})\)。

算例都在 `examples/`：

| 目录 | 用途 |
|------|------|
| `two_layer/`、`channel/` | 用户交付默认路径 |
| `lab/` | 正式 10 mm / 30³、概念实验室、通道图、标量 \(C_f\) ES-MDA（`lab_cf.yaml`） |
| `compositional/` | EXAMPLE 组分孪生 |

| 场合 | 命令 | 测点从哪来 |
|------|------|------------|
| 实验室做完一块样，有探头读数 | `apply <case.yaml>`（不要 `--demo`） | 你填的 CSV |
| 装好软件、验收、看输出长什么样 | `apply <case.yaml> --demo` | 程序用已知两层/通道自己算出来 |
| 既没 CSV 也没 `--demo` | 直接报错 | — |

`--demo` 不是「没测到数时的正式用法」。没有探头读数，渗透率反不出来；`--demo` 反演的是它刚埋进去的那两层（或通道），不是你那块石头。

## 1. 两层填砂（默认产品路径）

配置：`examples/two_layer/case.yaml`  
网格 25 mm，大约一分钟。

**自检（无 CSV）：**

```bash
reservoir apply examples/two_layer/case.yaml --demo --output results/examples/two_layer_demo
```

会写出 `results/examples/two_layer_demo/observations.csv`（假测点）和验收字段 `acceptance`。对比度真值是 10，通过时后验大约 8–12。

**有实测点：**

1. 复制 `observations_template.csv` 为 `observations.csv`（已附一份填好的示例，可先直接跑）。
2. 按探头名填时间、压力或含水、`sigma`。探头名必须和 YAML 里 `sensors` 一致。
3. 用挂了 CSV 的算例，**不要**加 `--demo`：

```bash
reservoir apply examples/two_layer/case_from_csv.yaml --output results/examples/two_layer
```

`case_from_csv.yaml` 只比 `case.yaml` 多一行 `experiment.observations: observations.csv`。

实验室常用分钟和 kPa 时，用 `observations_kpa_min.csv` 那种表头（`time` + `time_unit` + `unit`），或在 YAML 里设 `observation_time_unit` / `observation_pressure_unit`。内部一律转成秒和 Pa。

## 2. 已知通道填砂

通道形状事先画在 `regions.npy`，只反演高低渗数值，不猜通道位置。

```bash
reservoir apply examples/channel/case.yaml --demo --output results/examples/channel_demo
reservoir apply examples/channel/case_from_csv.yaml --output results/examples/channel
```

## 3. 跑完看什么

输出目录里：

| 文件 | 含义 |
|------|------|
| `k.npy` | 拟合后的渗透率（该正演 \(F\) 下的等效 \(K\)） |
| `pressure.npy`、`sw.npy`、`so.npy` | \(F(\hat m)\) 重建的三维场 |
| `apply.json` | 拟合、hold-out、预报、\(\theta\) |
| `invert.json` | 统一 run report（`reservoir invert` / `apply`） |
| `check83.json` | check.txt §83 十二问结构化答案 |
| `residuals.csv` | 白化观测残差明细 |
| `k_std.npy` | post_ensemble 开启时的 \(K\) 标准差场 |
| `figures/posterior_fields_xz.png` | 剖面图 |

三维 \(p/S\) 不是单独反演出来的饱和度图。有 `--demo` 时才会和已知真值比；有实测 CSV 时只看测点拟合和预报。

## 4. 测点 CSV

SI：

```text
time_s,sensor,kind,value,sigma,holdout
100,P_in_bot,pressure,1.61e5,2000,0
100,S_mid_top,saturation,0.24,0.04,0
```

实验室单位：

```text
time,time_unit,sensor,kind,value,unit,sigma,holdout
1.667,min,P_in_bot,pressure,161,kPa,2,0
```

- `sensor` 必须是 YAML 里声明过的探头名。
- `kind`：`pressure` 或 `saturation`。
- `sigma` 用该探头重复性，不要照抄模板里的 2 kPa / 0.04。
- `holdout=1` 的探头不参与同化，只用来打分。也可在 YAML 的 `holdout_sensors` 里点名。
- 不要把定压井的流量当观测。

层状用 `region_axis: z`。已知通道用 `region_map`。标量裂缝导流用 `examples/lab/lab_cf.yaml`（ES-MDA）。

示例 CSV 由 `examples/_make_observations.py` 用本正演生成；改了网格或探头后重新跑一遍即可。
