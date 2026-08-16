# CMG 反演 harness

同一套实验室 \(F+H+\) ES-MDA，对多套 IMEX 算例打分。

**通过** = hold-out / 预报 / 场 \(p,S_w\) / 见水时间。  
**不是** \(K=K_{\mathrm{CMG}}\)。不搜格子 \(K\)，不把 WI 拧到贴 CMG 流量。

## 命令

```bash
reservoir harness suite --fast
reservoir harness suite --cases lab_layers
reservoir harness search --case lab_layers --time-limit 300
reservoir harness journal --threshold 1.0
```

`--fast`（默认）：探针 + 对 `invert_in_fast` 案做 2 区 ensemble。  
`--all-invert`：每个 ready 案都反演。  
`--no-invert`：只探针。

正演是黑油油水（地面流量 + \(B_\alpha,c_r\)）。\(\theta\) 只有 log \(K\)，不再反演 \(c_t\)。

\(F(K_{\mathrm{CMG}})\) 物理地板（MRST 相势迎风 + 隐式输运 + SWT + \(k_z\) + 断层 TRANSI）：

| 案 | 均值 p F / CMG | p RMSE 原始 / 去均值 | Sw RMSE |
|----|----------------|----------------------|---------|
| lab_layers | 3002 / 2974 | 32 / 16 psi | **0.050** |
| fivespot | **3641 / 3645** | **21 / 20 psi** | **0.018** |
| fault | **3489 / 3494** | **23 / 23 psi** | **0.030** |

以前场尺度原始 p RMSE 400–560 psi，几乎全是均值涨压对不上。

同一套 ES-MDA（\(N_e=12,N_a=3\)）+ 对比度参数化后再比 CMG 场图：

| 案 | 改 F 前 p / hold / J | 现在 p / hold / J | \(k_{\mathrm{lo}}\) md / 对比（真值） |
|----|----------------------|-------------------|----------------------------------------|
| fivespot | 422 / 0.67 / 1.71 | **22 / 0.67 / 1.05** | 69 / 21（~60 / 15） |
| fault | 233 / 0.70 / 1.91 | **33 / 0.68 / 1.28** | 70 / **30**（40 / 50）；通道 \(k\approx 2100\) md |
| channel | 335 / 0.80 / 2.06 | **28 / 0.66 / 1.23** | 79 / **38**（50 / 40） |

场图：`black_oil/validation/cmg_harness/figures/{fivespot,fault,channel}_cmg_vs_inv.png`

## 目录

| id | 类型 | 时间窗 | v1 |
|----|------|--------|----|
| `lab_layers` | 两层 CART | 0.25–2 d | ready |
| `fivespot` | 五点井网 | **1–607 d**（IMEX 矿场尺） | ready，通道/基质 2 区 |
| `fault` | 断层+窗 | 1–607 d | ready，通道/基质 2 区 |
| `channel` | 通道（CART 代理 VARI） | 1–607 d | ready |
| `lab_box` | 30 cm 山形层理 | — | `need_imex` |
| `shale_s1` | 页岩水平井+缝 | — | unsupported（不是水驱 F） |

射孔从 `truth.wells` 的 `k_cmg` / `k_perfs` 抄 IMEX，不再猜中间一格。

## 剪枝

一次 \(F(K_{\mathrm{CMG\ or\ prior}})\) 到第一个报告时刻：

- 时间步塌 → `prune:underflow`
- \(S_w\) 几乎不动 → `prune:no_flood`
- 压力场没有 \(\Delta p\) → `prune:no_dp`
- 见水时间差一个数量级 → `prune:bt`

被剪的尝试进日记，不跑 \(N_e\times N_a\)。

## 日记 / 回溯 / 突破点

`black_oil/validation/cmg_harness/journal/attempts.jsonl`

- 子节点 \(J\) 比父节点差 → `backtrack`
- `journal --breakthrough`：第一次 \(J\) 低于门槛，以及相对父节点 \(\Delta J\) 最大的 keep

搜索只动同化旋钮（算法、\(N_e,N_a,\sigma,\mathrm{inflation}\)）。

代码迭代环（读 harness → 只改一处允许的 \(F/H/\)invert → 复测 → 留或回退）写在 `.grok/workflows/cmg-invert-iterate.rhai`。用 `/workflow cmg-invert-iterate` 跑；需要项目 workflow 目录信任。

## 和 lab_layers 旧脚本

`run_invert_eval.py` / `run_cmg_gap.py` 仍可单独跑。新入口是 `reservoir harness`。
