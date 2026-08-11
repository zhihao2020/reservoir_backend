# CMG 三维通道验证算例

用于验证传感器四场 + 多时刻形态发现算法能否从井点数据推断出**起伏高渗通道/山体脊**（非水平分层）。

## 来源

- 基例：`D:\Tool\CMG\IMEX\2024.20\TPL\spr\mxspr006.dat`（海水驱）
- 工作算例：`mxspr006_channel.dat`
- 再生脚本：`build_undulating_case.py`（`*GRID *VARI` + `*DTOP` 起伏）

> IMEX 在 `*GRID *CART` 上禁止变深度；起伏构造必须用 `*GRID *VARI` + `*DTOP`（与官方 drm/geo 样例一致）。

## 构造起伏（关键）

| 项目 | 设置 |
|------|------|
| 网格 | **7×7×5** `*VARI`，`*KDIR *DOWN` |
| 层厚 `DK` | 40 / 35 / 30 / 25 / 20 ft（底→顶） |
| 顶面 `DTOP` | 对角山脊 + 正弦波，**起伏约 104 ft**（~1986–2089 ft） |
| 高渗体 | 沿 I≈J 走廊；**k 层随构造抬升**：翼部 k=1–3，脊部 k=3–5 |
| 渗透率 | 基质 50 md；通道 2000 md（水平）/ 200 md（垂向） |
| 井完井 | INJ (1,1) 与 PROD (7,7) 各射开 k=2,3,4 |

真值：`truth_channel.json`（含 `structure.dtop_ft` 与 `channel_blocks_ijk`）。

## 运行

### 0. 再生起伏算例

```bash
python validation/cmg_channel_3d/build_undulating_case.py
```

### 1. 算法侧合成孪生（无需 CMG 许可证）

```bash
python validation/cmg_channel_3d/run_imex_and_validate.py --synthetic
```

合成孪生含 **蜿蜒 + 垂向起伏** 的山体通道（非水平板状）。

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

## 产出

| 文件 | 含义 |
|------|------|
| `synthetic_validation_report.json` | 合成孪生 Dice 等 |
| `validation_report.json` | CMG 几何上的发现指标 |
| `truth_active_mask.npy` | 真值通道 mask |
| `discovered_active_mask.npy` | 算法发现 mask |
| `.cmg_patch_audit/` | 补丁审计与回滚 |
| `mxspr006_channel.out` / `.sr3` | IMEX 运行产物（本地生成，默认不入库） |

## 实测结果（起伏模型，本机 2026-08-11）

| 项目 | 结果 |
|------|------|
| 构造起伏 | **~104 ft** DTOP 山脊 + 波状起伏 |
| IMEX | **Normal Termination**，7×7×5=245 块，~791 天 |
| 井点时间序列 | 从 `.out` 解析 9 个时刻 Sw + BHP |
| 末时刻井 Sw | INJ ~0.67，PROD ~0.23 |
| 发现 mask vs 起伏通道 Dice | **~0.43**（recall ~0.89） |
| CMG 全域 ΔSw 足迹 vs 通道 Dice | **~0.68**（水沿起伏高渗体前进） |
| 加密后网格 | 245 → 980 cells |

说明：相对水平通道，起伏模型更难；Dice 下降属预期。关键验收：高 recall（脊被覆盖）+ CMG ΔSw 足迹与真值山脊一致。

## 与主线流水线关系

```text
多时刻井点 p / Sw
    → run_time_series（四场重建）
    → infer_shape_indicator（形态指标）
    → refine_mesh_by_indicator（正交加密）
    → 再跑时间序列
```

CMG 只提供 **带真值非均质的三维正演**，不替代本仓库的反演/传感器重建实现。
