# 案例库

**完整联调说明见 [`docs/联调指南.md`](../docs/联调指南.md)**（从零到跑通的 step-by-step）。
协议细节见 [`docs/接口协议.md`](../docs/接口协议.md)。

## 一键自检

```bash
python scripts/verify_examples.py     # 应打印 "all examples verified"
```

## 案例一览

| 案例 | 目录 | 网格 | 模型 | 用途 |
|---|---|---|---|---|
| 秒级联调 | `small/` | 8×8×8 | black_oil_3phase | **首次跑通、TCP/UDP 联调** |
| 2D | `twod/` | 30×30×1 | black_oil_3phase | 单层切片调试 |
| 多模型对照 | `model_compare/` | 10×10×10 | total_mobility + 3phase | 验证两条反演路径 |
| 离线文件 | `offline/` | 20×20×20 | black_oil_3phase | `-o` 输出格式 |
| 在线对接 | `online/` | 60×60×60 | black_oil_3phase | GEM shailoil、16 测点 5 井 12 月 |

## 每个案例怎么跑

```bash
# 离线（文件 → 文件）
python -m src examples/<name>/case.yaml -o results/<name>

# 在线（TCP 入 + UDP 出），四终端见 docs/联调指南.md §5
python -m src examples/<name>/case.yaml --tcp-port 9000 --ip 127.0.0.1 --lab-port 9001 --field-port 9002
```

## 案例文件与生成

除 `online/` 外，其余由 `scripts/gen_synthetic_case.py` 生成（确定性、无 GEM）：

```bash
python scripts/gen_synthetic_case.py    # 重新生成 small/twod/model_compare/offline
```

每个案例目录含：`case.yaml` + `probes.csv` + `wells.csv` + `observations.csv` + `series.csv`。
`online/` 的 `send_steps.py` / `recv_fields.py` 是协议 v2 的参考实现，可对任意案例目录复用
（用 `--probes/--wells/--observations/--series` 指定路径）。

`case.yaml` 字段说明见 [`docs/案例库.md`](../docs/案例库.md)。
