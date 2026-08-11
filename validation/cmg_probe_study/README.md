# CMG 虚拟测点精度研究

## 目的

在**不修改 CMG 井网、不重跑 IMEX** 的前提下，从已有 `.out` 全场压力/饱和度中**虚拟抽样** exclusive 测点（`observer_p` / `observer_s`），扫测点数量 `N` 与布点策略，评估本软件点优先四场重建相对 CMG 的精度曲线。

## 测点推荐（产品 API）

| API | 说明 |
|-----|------|
| `place_uniform_probes` | 几何均匀 / 空间填充基线 |
| `recommend_probes` | 自适应 DOE：`maximin` / `variance` / `hybrid` |
| `split_n_probes` | N → (n_p, n_s)，奇数时 p 多 1 |

业务传感器 YAML **不**需要配置推荐算法；验证脚本与 API 直接调用。

## 运行

```bash
# 默认：channel+fault，N=0,4,8,16，uniform+adaptive
python validation/cmg_probe_study/run_probe_study.py

python validation/cmg_probe_study/run_probe_study.py --cases channel --n-list 0,4,8 --layouts adaptive
```

输出：

- `probe_study_report.json`
- `PROBE_STUDY.md`

## 依赖

- `validation/cmg_channel_3d/*.out` + `truth_*.json`
- `validation/cmg_fault_3d/*.out` + `truth_*.json`

无需本机 CMG 可执行文件（仅读已有结果）。
