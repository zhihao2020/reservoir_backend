# Reservoir Backend

300 mm 立方试块的**实验室多相渗流数字孪生反演后端**。

核心链：

```text
Experiment → Controls → Prior m → IMPES Forward F
         → ObservationOperator H → ES-MDA → Posterior → Forecast
```

饱和度是动态状态，不是物性场。渗透率在固定维数的区域 / 粗网格上反演，默认不是每格一个独立 K。

设计见 [docs/target_architecture.md](docs/target_architecture.md)。审查见 [docs/audit_2026.md](docs/audit_2026.md)。假设见 [docs/model_assumptions.md](docs/model_assumptions.md)。

## 安装

```bash
python -m pip install -e ".[dev]"
```

## CLI

```bash
reservoir validate config/lab_30cm.yaml
reservoir simulate config/lab_30cm.yaml --output results/sim
reservoir invert   config/lab_30cm.yaml --output results/inv
reservoir invert   config/lab_30cm.yaml --preset balanced
reservoir invert   config/lab_30cm.yaml --auto --time-limit 120
reservoir forecast config/lab_30cm.yaml --output results/fc
reservoir synthetic --output results/syn

# 实验室可用路径（不要拿去对 CMG 全场）
reservoir apply config/lab_apply.yaml --demo --output results/lab
# 有测点时：在 yaml 里写 experiment.observations: observations.csv，去掉 --demo
```

或 `python -m reservoir_backend ...`。

**能用在哪：** 300 mm 试块、和配置一致的入口/出口与内部测点时序。输出是后验 \(K\) 和 \(F(m_{\mathrm{post}})\) 的 \(p,S_w,S_o\)，不是 IMEX 格子场。  
**不能用在哪：** 要求 \(p/S/K\) 和 CMG 全场逐格相等。那是另一套正演，不是这个产品。

## 当前物理（P0）

- Model A：单相 Cartesian TPFA
- Model B / D：黑油油水 IMPES（MRST 表面体积 \(\varphi b_\alpha S_\alpha\)）。实验室 \(B=1,c=0\)；CMG 虚拟实验用牌组 PVT
- Model C：三相不混溶 IMPES（独立 Corey，闭合 \(S_w+S_o+S_g=1\)）
- 毛管：Brooks–Corey / van Genuchten / none，由 case 显式选择（实验室默认 Brooks–Corey）
- 端口：定流量（地面）或定压，不是默认 Peaceman 油藏井
- 观测：点传感器三线性插值；体积平均；端口相流量。测点 \((x,y,z)\) 不必共面，一口井不同深度用 `column_sensors`
- 反演：Region 或 CoarseField 上的 log K + ES-MDA。PVT 不进 \(\theta\)。和 CMG 对齐的是井控与测点，不是 \(K=K_{\mathrm{CMG}}\)
- 并行：ensemble 成员默认线程池（`n_workers`），不是进程池
- 后验：\(K\) 均值/标准差/分位数，以及 ensemble 正演得到的 \(p,S_w,S_o,S_g\) 统计
- 时间：内部一律秒

## 明确未做

溶气/放气 \(R_s(p)\) 自由气、EnKF、MPFA、动态 AMR、神经网络代替正演。三相相对渗透率是独立 Corey，不是 Stone。

`black_oil/` 与 `shale_oil/` 下的旧四场/CMG 脚本不再接入本内核。

## 测试

```bash
pytest -q
```
