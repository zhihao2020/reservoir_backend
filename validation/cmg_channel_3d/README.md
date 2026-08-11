# CMG 三维通道验证算例

用于验证传感器四场 + 多时刻形态发现算法能否从井点数据推断出高渗通道（“山体/窜流通道”）。

## 来源

- 基例：`D:\Tool\CMG\IMEX\2024.20\TPL\spr\mxspr006.dat`（7×7×3 海水驱）
- 克隆并最小补丁：`mxspr006_channel.dat`
- 补丁规格：`patch_channel.json`（经 `cmg-suite/scripts/patch_dat.py` 应用，带审计回滚）

## 真值通道

注入井 `(1,1,1)` → 生产井 `(7,7,3)` 对角线高渗带：

- 基质：~50 md
- 通道：~2000 md（水平），~200 md（垂向）

真值块列表见 `truth_channel.json`。

## 运行

### 1. 算法侧合成孪生（无需 CMG 许可证）

```bash
python validation/cmg_channel_3d/run_imex_and_validate.py --synthetic
```

使用仓库内 `build_channel_twin` 生成已知通道 + 多时刻井点传感器，跑 `run_shape_discovery`，输出 Dice/precision/recall。

### 2. 运行 IMEX（需本机许可证）

```bash
python validation/cmg_channel_3d/run_imex_and_validate.py --execute
```

或手动：

```text
cd validation/cmg_channel_3d
D:\Tool\CMG\IMEX\2024.20\Win_x64\EXE\mx202420.exe -f mxspr006_channel.dat
```

### 3. 从 .out 回灌传感器并评估

```bash
python validation/cmg_channel_3d/run_imex_and_validate.py --from-out mxspr006_channel.out
```

若 `.out` 不存在或井表解析不全，脚本会用与水驱趋势一致的 **proxy 井点时间序列** 在 CMG 几何上做发现评估。

## 产出

| 文件 | 含义 |
|------|------|
| `synthetic_validation_report.json` | 合成孪生 Dice 等 |
| `validation_report.json` | CMG 几何上的发现指标 |
| `truth_active_mask.npy` | 真值通道 mask |
| `discovered_active_mask.npy` | 算法发现 mask |
| `.cmg_patch_audit/` | 补丁审计与回滚 |
| `mxspr006_channel.out` / `.sr3` | IMEX 运行产物（本地生成，默认不入库） |

## 实测结果（本机 2026-08-11）

| 项目 | 结果 |
|------|------|
| IMEX | **Normal Termination**，约 791 天，0.48 s |
| 井点时间序列 | 从 `.out` 解析 9 个时刻 Sw + BHP |
| 末时刻井 Sw | INJ 0.718，PROD 0.275（已见突破） |
| 发现 mask vs 真值通道 Dice | **~0.61**（recall 1.0） |
| CMG 全域 ΔSw 足迹 vs 通道 Dice | **~0.80**（说明水沿高渗通道前进） |
| 加密后网格 | 147 → 588 cells |

说明：反问题欠定，Dice 不必接近 1；关键是通道走廊被高指示区覆盖，且 CMG 正演足迹与真值通道一致。

## 与主线流水线关系

```text
多时刻井点 p / Sw
    → run_time_series（四场重建）
    → infer_shape_indicator（形态指标）
    → refine_mesh_by_indicator（正交加密）
    → 再跑时间序列
```

CMG 只提供 **带真值非均质的三维正演**，不替代本仓库的反演/传感器重建实现。
