# Reservoir Backend

300 mm 立方试块的**页岩油实验数字孪生**：先用一段观测反演 \(\theta=(\log C_f,\log T_{mf})\)，再冻结 \(\theta\) 做组分双重介质（DPDP）全隐式正演。

```text
观测 CSV [0, T_hist] + 井控 → ES-MDA 更新 (C_f, T_mf) → 冻结 θ → 组分 DPDP FIM 正演整段井控
```

- \(C_f = k_f b_f\)：裂缝网络沿程传输能力
- \(T_{mf}\)：基质 \(\leftrightarrow\) 裂缝补给
- 固定不反演：\(k_m,\varphi_m,\varphi_f\)、EOS/PVT、黏度、相对渗透率

产品 Case：[`examples/lab_v1/`](examples/lab_v1/)。假设：[docs/model_assumptions.md](docs/model_assumptions.md)。

## 安装

```bash
python -m pip install -e ".[dev]"
```

## 用户怎么用

主命令是 `reservoir apply`：历史窗反演，然后组分正演整段井控。日常和 CI 用 `case_dev.yaml`，不要在 CI 里跑 30³ ES-MDA。

配置五件套（路径写在 `case.yaml` 里）：

| 文件 | 内容 |
|------|------|
| `case.yaml` | 网格、物理开关、反演、引用 |
| `pvt.yaml` | EOS / 黏度 / 相对渗透率（不反演） |
| `wells.yaml` | 井几何与完井 |
| `controls.csv` | 井控时间序列 |
| `observations.csv` | 反演用的那段数据 |

```bash
reservoir apply examples/lab_v1/case_dev.yaml --demo --output results/lab_v1_demo
reservoir apply examples/lab_v1/case_dev.yaml --output results/lab_v1
```

有探头 CSV 时在 `experiment.observations` 写上路径，**不要** `--demo`。`--demo` 只是自检。粗网格夹具：`examples/lab/lab_cf.yaml`。

### 跑完看什么

| 文件 | 含义 |
|------|------|
| `k.npy` | 后验裂缝连续体渗透率（由 \(C_f\) 展开） |
| `pressure.npy`、`sw.npy`、`so.npy`、`sg.npy` | \(F(\hat\theta)\) 场 |
| `apply.json` / `invert.json` | \(\theta\)、拟合 / hold-out |

### 测点 CSV

```text
time_s,sensor,kind,value,sigma,holdout
100,P_in,pressure,1.20e7,2000,0
```

不要把定压井的流量当观测。CMG-GEM 只是交叉验证尺子，用户入口不必先跑 GEM。

## 其他命令

`invert` / `simulate` 仅调试。`forecast` **不再反演**，必须带 `--posterior invert.json`。

```bash
reservoir validate examples/lab_v1/case_dev.yaml
reservoir simulate examples/lab/lab_cf.yaml --output results/sim
reservoir invert   examples/lab/lab_cf.yaml --self-check --output results/inv
reservoir forecast examples/lab/lab_cf.yaml --posterior results/inv/invert.json --output results/fc
```

## 当前物理

- 正演：等温组分 DPDP FIM（Peng–Robinson + PT 闪蒸）。单孔组分 `physics.model: compositional` 可选。
- 反演：\(\theta=(\log C_f,\log T_{mf})\)，产品默认 ES-MDA。
- 黑油 IMPES / 顺序隐式 / 黑油 FIM 已删除。

## 明确未做

逐格 \(K\)、Archie/EM、热、MPFA、PINN。Online Parameter EnKF / UDP 冻到 M3。`references/` 只读对照。

## 测试

```bash
pytest -q
```
