# 非均质四场验证（对应软件要求 2–4）

**禁止**用均质 CMG/合成模型做验收。本目录与 `cmg_channel_3d`、`cmg_fault_3d` 一律使用 **非均质 k**（通道脊 / 断层狗腿）。

## 软件要求映射

| 步骤 | 输入 | 输出 | 入口 |
|------|------|------|------|
| 1 网格 | 边界、井、dx/dy/dz | 网格序号与坐标 | `build_mesh` |
| 2 压力 | 时刻 t 井压、边界压力/流量 | 全网格 p | `reconstruct_pressure` |
| 3 饱和度 | 时刻 t 井饱和度、边界流量线索 | 全网格 sw,so,sg | `reconstruct_saturation` |
| 4 物性 | 网格 p,S,流量 | 全网格 k, φ（可非均质） | `invert_rock_properties` |

端到端：`run_time_slice` / `run_time_series`。

## 运行

```bash
python validation/black_oil/heterogeneous_four_field/run_validate.py
```

合成通道 + 断层非均质孪生；报告 `validation_report.json`。

CMG 非均质正演（对照，非产品内核）：

```bash
python validation/black_oil/cmg_channel_3d/run_imex_and_validate.py --execute
python validation/black_oil/cmg_fault_3d/run_imex_and_validate.py --execute
```

## 验收要点

1. 真值 `k_contrast > 2`（明确非均质）
2. 井点压力重建误差 ≈ 0（传感器硬数据）
3. 全场 p/Sw 相对 L2 有限；log-k RMSE 用于物性诊断（欠定，不要求与 CMG 数值完全相等）
4. 不使用全场常数 k 的 CMG 基例作为“通过”标准
