# reservoir 三维重建 —— 联调演示包（线性插值版）

这是一个**仅用于合同签订之前联调**的自包含演示包，走与正式版**完全相同、已冻结**的通讯协议 v2（TCP 入 / UDP 出），但把正式版的核心反演替换成了**简单的三维线性插值**，用于演示「采集端 → 后端」的调用/接收数据流。**大屏（仪表盘显示）不在本包内，由其他团队按协议文档实现。**

## 包内容

```
example_demo/
├── demo_server.py      # 后端（接收端）：监听 TCP、逐拍线性插值、UDP 发场
├── send_steps.py       # 采集端（调用端）：读 CSV、逐拍发 STEP
├── protocol.py         # 冻结 wire 协议 v2 帧封装（自包含，逐字节与规范一致）
├── case.yaml           # 演示配置：网格 / 测点 / 井 / 初值 / 矿场尺度
├── probes.csv          # 测点坐标
├── wells.csv           # 井坐标
├── observations.csv    # 测点观测时间序列
├── series.csv          # 井注采时间序列
├── requirements.txt    # 依赖
└── README.md           # 本文档
```

全部文件自包含：`send_steps.py` / `demo_server.py` 只引用本包内的 `protocol.py`，**不依赖、也不需要任何其它源码目录**。

## 环境要求

- Python ≥ 3.11
- 依赖：`numpy`、`scipy`、`PyYAML`

```bash
pip install -r requirements.txt
```

## 快速开始（2 个终端）

> **先进入本目录再跑**（下面所有命令都以 `example_demo/` 为当前目录）：
> ```bash
> cd example_demo
> ```

两个角色。**启动顺序：先起后端，再发采集端**。

```bash
# 终端 1：后端（接收端，监听 TCP，UDP 发场到 --lab-port / --field-port）
python demo_server.py case.yaml --tcp-port 9000 --ip 127.0.0.1 --lab-port 9001 --field-port 9002

# 终端 2：采集端（调用端，读 CSV、逐拍发 STEP）
python send_steps.py --host 127.0.0.1 --port 9000
```

> 大屏（仪表盘）由其他团队实现：它在 `--lab-port` / `--field-port` 收后端 UDP 发来的完整场。协议逐字节规范见正式版的 `docs/接口协议.md`。

## 预期输出

**终端 1（后端）**：

```json
{"mode": "tcp", "protocol_version": 2, "tcp_port": 9000, "demo": true, "core": "linear interpolation"}
{"tcp": "connected", "peer": "127.0.0.1:xxxxx"}
{"tcp": "disconnected"}
```

**终端 2（采集端）**：

```
HELLO v2: n_probes=4 n_wells=2 grid=8x8x8
ACK seq=0 time_s=2592000.0
ACK seq=1 time_s=5184000.0
ACK seq=2 time_s=7776000.0
```

## 关于本演示

- **核心是线性插值**（`scipy.interpolate.griddata`）：把测点/井的观测值线性插到全网格；`k` / `phi` 直接填常数（`k0_md` / `phi0`）。演示重点是**通讯链路与数据流**，不是反演精度。
- **通讯协议与正式版完全一致**（协议 v2，已冻结）：TCP 帧、STEP/记录编码、UDP 头、FIELD 块 CRC、END 整体 CRC、单元排序 `cell = k*ny*nx + j*nx + i`、reshape `(nz, ny, nx)`。
- **不含核心源码**：本包只包含冻结的公开 wire 协议与线性插值演示逻辑，不包含正式版的反演算法（克里金、岩石 k/φ 反演、会话管理等）。

## 换自己的数据

只需替换同格式的 `probes.csv` / `wells.csv` / `observations.csv` / `series.csv`，并同步修改 `case.yaml` 里的网格 `nx/ny/nz` 与 `extent_m` 即可。CSV 列格式：

| 文件 | 列 |
|---|---|
| `probes.csv` | `id,x_m,y_m,z_m` |
| `wells.csv` | `id,kind,x_m,y_m,z_m,x2_m,y2_m,z2_m` |
| `observations.csv` | `time_s,probe,quantity,value,unit`（quantity: pressure/sw/so/sg） |
| `series.csv` | `time_s,well,pw_pa,q_m3s,qw_m3s,qo_m3s,qg_m3s` |
