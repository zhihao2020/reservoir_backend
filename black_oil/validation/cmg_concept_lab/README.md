# Concecpt 300 mm 单层 IMEX 测例

从 `black_oil/validation/cmg_channel_3d/mxspr006_channel.dat` 克隆，不改原牌。
几何与测点对齐 `references/concecpt`（`条件.txt`、山形公式）。电阻率点按棋盘拆成压力或饱和度，声波 10 点只饱和度；同一 \((x,y,z)\) 只有一种量。

不接 Archie。PVT/`*SWT` 保持 MXSPR006 海水黑油。

## 建牌

```bash
python black_oil/validation/cmg_concept_lab/build_concept_case.py --execute
```

写出 `mxspr006_concept.dat`、`truth_concept.json`、三份 `patch_*.json`。补丁走 cmg-suite `clone_case.py` / `patch_dat.py`。

## 预检与运行

脚本目录：`C:\Users\xuzhihao\.codex\skills\cmg-suite\scripts`

```bash
python C:\Users\xuzhihao\.codex\skills\cmg-suite\scripts\controlled_imex_run.py --case black_oil/validation/cmg_concept_lab/mxspr006_concept.dat --workspace D:\Tool\CMG\_cmg_suite_runs\concept_lab --cmg-home D:\Tool\CMG --execute --materialize-results --overwrite --pretty
```

文件名带 `spr`，预检 pattern 为 `spr_six_layer_kvar`。

## 结果

```bash
python C:\Users\xuzhihao\.codex\skills\cmg-suite\scripts\collect_metrics.py --path D:\Tool\CMG\_cmg_suite_runs\concept_lab --pretty > black_oil/validation/cmg_concept_lab/metrics.json
python C:\Users\xuzhihao\.codex\skills\cmg-suite\scripts\summarize_run_health.py --input black_oil/validation/cmg_concept_lab/metrics.json --pretty > black_oil/validation/cmg_concept_lab/run_health.json
python black_oil/validation/cmg_concept_lab/extract_probes.py --out D:\Tool\CMG\_cmg_suite_runs\concept_lab\case_clone\mxspr006_concept.out
```

`run_health.json`：`healthy_runnable`（Normal Termination，0 error，5 warning）。终了平均压力 1501.47 psi，121 步，约 18 s。

`probe_timeseries.json`：85 条时序，39 压力（Pa）+ 46 饱和度，无重复 xyz；时刻 0.1 / 0.5 / 1 / 2 / 5 day。
