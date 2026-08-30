# V1 发布门槛

普通 CI（`.github/workflows/ci.yml`）跑 `pytest -m "not slow"` 和 `-m dpdp`。30³ 不进普通 CI，手工 / nightly：

```text
python scripts/dpdp_scale_gate.py --n 30 --t-end 0.05 --json-out docs/bench/dpdp_scale_gate.json
```

切产品版本前必须同时满足：

| 门槛 | 证据 |
|------|------|
| correctness 全绿 | CI / `pytest -q` |
| FastPR parity + 包络 | `tests/physics/test_flash_backend.py`, `test_flash_envelope.py` |
| D0–D4、Jacobian、restart | `tests/comp/` |
| EnKF 增量窗、无观测复用 | `tests/twin/test_loops.py`, `test_member_checkpoint.py` |
| 质量守恒 | DPDP 步报告 `mass_rel`；gate JSON |
| 标准 30³ 一步 | `docs/bench/dpdp_scale_gate.json`，目标 \(T<60\,\mathrm{s}\) |

性能结论只认 `dpdp_scale_gate` JSON（同一 `t_end`、`max_steps=1`、无线控、同一 Flash backend）。
