# CMG 五点井网尺子（不同工况）

4 角注 + 中心采，缓变高渗条带 + 基质杂音。流体继承 mxspr006。

```bash
python validation/cmg_fivespot/build_fivespot.py
python validation/cmg_fivespot/run_imex.py
python validation/cmg_probe_study/run_probe_study.py --cases fivespot --n-list 8,16 --layouts uniform
```
