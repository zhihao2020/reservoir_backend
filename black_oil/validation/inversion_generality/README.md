# 跨工况反演验收

同一套旋钮（点优先 ± 全网格 ES-MDA）在通道 / 断层 / 实验室山形层理上扫测点数。  
探头按身份划分训练/验证；主指标是**验证探点** p/S RMSE，不是为某一套真值特化的 Dice。

```bash
python validation/inversion_generality/run_eval.py --methods point_first --n 12
python validation/inversion_generality/run_eval.py --methods point_first,grid_esmda --n 10 --ne 8 --na 2
```

现场出四场时全部探头都用；本目录的 train/val 只用于评估与一次性锁常数。

## 点优先复测（同一旋钮，n=12）

验证探点 RMSE（N=0 没有验证探头）。井点 p 误差均为 0。

| 孪生 | N | 验证 p RMSE (Pa) | 验证 Sw RMSE | 真值层 k 比 |
|------|---|------------------|--------------|-------------|
| 通道 | 8 | 8.9e4 | 0.33 | 1.10 |
| 通道 | 16 | 6.1e4 | 0.064 | 1.06 |
| 断层 | 8 | 3.8e4 | 0.30 | 1.17 |
| 断层 | 16 | 3.5e4 | 0.091 | 1.38 |
| 实验室层理 | 8 | 4.0e4 | 0.34 | 1.77 |
| 实验室层理 | 16 | 1.2e4 | 0.20 | 1.68 |

三套几何上 N 增大时验证误差都下降，没有为某一套改先验。k 对比仍然弱，高精度路应走全网格 ES-MDA，而不是换形状 θ。
