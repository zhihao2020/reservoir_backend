# 联调包（shale_oil）

本目录是完整联调交付：配置、数据、调用脚本、接收脚本、使用说明、requirements。
Linux `src.so` 由 GitHub Actions **pack-linux** 打进本目录（解压 zip 后应能看到 `src.so`）。
协议 v2 冻结不变（wire 接口未动）。

| 文件 | 作用 |
|---|---|
| `src.so` | 后端动态库（Linux / CPython 3.12） |
| `case.yaml` | 网格 / 反演 / 相似比 |
| `probes.csv` `wells.csv` `observations.csv` `series.csv` | 27 测点 / 5 井 / 120 拍 |
| `reservoir.py` | 后端启动器（源码和 src.so 都用它） |
| `send_steps.py` | 调用端：TCP 逐拍发 STEP |
| `receive_fields.py` | 接收端：UDP 收 lab/field 场 |
| `requirements.txt` | numpy / scipy / PyYAML |
| `VERSION` | 打包时写入的库版本 |

不要在 `src.so` 旁边再放 `src/` 目录，否则 Python 会加载源码包而不是这个扩展。

## 安装

```bash
pip install -r requirements.txt
```

## 联调命令（先 `cd` 到本目录）

启动顺序：**接收 → 后端 → 调用**。

```bash
# 终端 1：接收端（仪表盘参考实现，绑 9001/9002）
python receive_fields.py --lab-port 9001 --field-port 9002

# 终端 2：后端（真实 GEM 数据，27 测点 / 5 井 / 120 拍）
python reservoir.py case.yaml --tcp-port 9000 --ip 127.0.0.1 \
    --lab-port 9001 --field-port 9002

# 终端 3：调用端（读本目录时序，逐拍发 STEP）
python send_steps.py --host 127.0.0.1 --port 9000
```

`receive_fields.py` 每收到一拍打印一行 JSON（`n_cells`、各场均值）。加 `-o out` 会把每拍存成 `lab_XXXX.npz` / `field_XXXX.npz`。

`--model` 只开放 `black_oil`。不传则用 `case.yaml` 里的 `forward.model: none`。
