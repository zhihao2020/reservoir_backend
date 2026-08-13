# CMG 断层 + 偏移通道验证算例

验证传感器四场 / 形态发现在**有断层**时的行为：构造落差 + 封闭性 + 高渗通道狗腿绕断层窗。

## 来源

| 文件 | 角色 |
|------|------|
| 基例 `IMEX/TPL/spr/mxspr006.dat` | 海水驱黑油（可跑通注水） |
| 参考 `IMEX/TPL/geo/mxgeo002.dat` | 官方 *Fault Barrier*（`*FAULT` throw 语法） |
| 参考 hrw/wwm | `*TRANSI *MOD = 0` 封闭断层 |
| 本算例 `mxspr006_fault.dat` | 克隆 mxspr006 后最小改造 |
| `reference_mxgeo002_fault_barrier.dat` | 只读参考副本（不改） |

再生：

```bash
python validation/cmg_fault_3d/build_faulted_case.py
```

## 模型特征

| 项目 | 设置 |
|------|------|
| 网格 | **9×9×4** `*CART`（`*FAULT` throw 在 CART 上支持） |
| 构造断层 | `*FAULT` **25 ft throw**，平面迹线台阶状（仿 mxgeo002） |
| 水力封闭 | `*TRANSI`：j=1–6 **密封 (0)**；j=7–9 **泄漏窗 (0.05)** |
| 高渗通道 | 断层西侧 j≈5，东侧 **偏移到 j≈8**，经泄漏窗狗腿连通 |
| 井 | INJ (1,5,k=2–4)，PROD (9,8,k=2–4) |
| 渗透率 | 基质 40 md；通道 2000 md；窗内仍可窜流 |

真值：`truth_fault.json`。

> 注：`*DTOP` 起伏与 `*FAULT` throw 不宜硬叠在同一 CART 模型上；起伏山脊见 `validation/cmg_channel_3d/`，断层见本目录。需要时可用两套算例分别验收。

## 运行

```bash
# 合成断层孪生（无 CMG）
python validation/cmg_fault_3d/run_imex_and_validate.py --synthetic

# IMEX
python validation/cmg_fault_3d/run_imex_and_validate.py --execute

# 从 .out 回灌
python validation/cmg_fault_3d/run_imex_and_validate.py --from-out validation/cmg_fault_3d/mxspr006_fault.out
```

## 验收要点

1. IMEX **Normal Termination**
2. 水沿狗腿通道 + 泄漏窗前进（ΔSw 足迹贴近通道 mask）
3. 形态发现能覆盖偏移通道走廊（允许 Dice 低于无断层算例）
4. 合成孪生中断层低渗带不应被大量标成“通道”

## 与算法关系

CMG 提供带断层的三维正演；本仓库仍只从**井点 p/Sw 时间序列**做重建与形态指示，不声称与 IMEX 数值等价。
