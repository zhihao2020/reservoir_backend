# check.txt §83 十二问验收表

本表把 `docs/check.txt` §83 的 12 个问题映射到产品字段与 CLI 产物。运行 `reservoir invert` 或 `reservoir apply` 后，输出目录含 `check83.json`（结构化答案）与 `invert.json`（统一 run report）。

## 目标链

```
Experiment YAML → invert (LM) → post_ensemble (Ne=8, optional) → forecast → run_report + check83
```

正演始终是产品求解器 \(F\)。V1 尺子是自洽 synthetic truth，不对照 CMG/IMEX。runtime 不 import `references/**`。

## 十二问映射

| # | check.txt 问题 | 代码 / 字段 | CLI 产物 |
|---|----------------|-------------|----------|
| 1 | 物理假设是什么？ | `PhysicsSpec`：`model`、`fully_implicit`、`capillary`、`p_init`、`sw_init` | `check83.json` → `q01_physics_assumptions` |
| 2 | 井控如何设定？ | `experiment.controls`：端口、kind、时间点数 | `q02_controls` |
| 3 | 同化用了哪些传感器？ | `experiment.assimilate_observations()` | `q03_assimilating_data` |
| 4 | hold-out / 预报段观测？ | `observation.holdout`、`history_end_s` | `q04_holdout_data` |
| 5 | 参数化形式？ | `parameterization` 类名、`n_params` | `q05_parameterization`；`invert.json` → `parameterization` |
| 6 | 可识别性？ | LM 后验 `theta_std`、prior/posterior 比 `identifiability` | `q06_identifiability`；`invert.json` → `posterior.identifiability` |
| 7 | 质量守恒？ | `mass_report` → `relative_balance_error` | `q07_mass_balance`；`invert.json` → `mass_balance` |
| 8 | hold-out / forecast 是否改善？ | `assimilate_rmse`、`holdout_rmse`、`forecast_rmse` | `q08_holdout_forecast_improvement`；`invert.json` → `metrics` |
| 9 | K 在测点邻域是否被约束？ | `post_ensemble` → `k_std` + sensor cell mask | `q09_k_constrained_regions`；`k_std.npy` |
| 10 | 高不确定区在哪？ | `max(k_std)`、远离测点均值 | `q10_k_high_uncertainty` |
| 11 | 能否增量更新？ | `DigitalTwin.assimilate(posterior, new_obs)`：LM warm-start | `q11_incremental_update` |
| 12 | 失败如何归因？ | RMSE、mass balance、`notes` | `q12_failure_attribution` |

## 实现入口

- `reservoir_backend/twin/acceptance.py` — `build_check83_report()`、`write_check83_report()`
- `reservoir_backend/twin/run_report.py` — `build_invert_report()`、`write_run_report()` → `invert.json` + `residuals.csv`
- `reservoir_backend/cli/reporting.py` — `emit_invert_artifacts()` 统一写盘

## 与 check.txt §28 的偏离

| check.txt | 本产品 | 理由 |
|-----------|--------|------|
| ES-MDA ensemble | 过渡仍是 LM + 可选 **Ne=8 后验局部 ensemble**；V1 目标为 Scalar \(C_f\) + ES-MDA | 实验室路径尚未切换 |
| 全序贯 DA | `assimilate()` stub（1–2 步 LM） | 在线 Parameter EnKF 尚未接入 CLI |

## 验收命令

```bash
pytest tests/inverse/test_post_ensemble.py tests/twin/test_check83_report.py tests/inverse/test_forecast.py -q

reservoir apply examples/lab/lab_apply.yaml --demo --output results/lab
# 期望：invert.json, check83.json, residuals.csv；post_ensemble 开启时有 k_std.npy
```

## pass 语义

每题 `{answer, evidence, pass}`：`pass=true/false/null`。`null` 表示不适用（如无 hold-out 传感器）。`summary.n_pass` / `n_fail` / `n_na` 汇总。
