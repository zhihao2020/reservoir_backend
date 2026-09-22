# reservoir_backend 0.4.0

Laboratory 3D reconstruction: probe/well series → full-grid pressure, saturation, porosity, permeability.

库版本为 `src.version.__version__`（`0.4.0`）：

```bash
python -c "import src; print(src.__version__)"
python -m src --version
python -m src --help
```

## 安装

```bash
pip install -r requirements.txt
```

联调包最小依赖见 [`example/requirements.txt`](example/requirements.txt)。

## 从源码运行

```bash
python -m src path/to/case.yaml                          # 离线：文件 → 文件
python -m src path/to/case.yaml -o results/run           # 落盘（fields.npz / *.npy / *.csv / *.json）
python -m src path/to/case.yaml --tcp-port 9000 --ip 127.0.0.1 --lab-port 9001 --field-port 9002
```

`--tcp-port` 是测点/注采的入站 TCP（无默认值）；`--lab-port` / `--field-port` 是反演
结果的两路 UDP 出站（实验 / 矿场尺度）；`--control-port` 是可选的控制端口（应答
RESEND 重发请求）。不写 `-o` 不落盘。`--model` 只开放 `black_oil`（覆盖 YAML 里的
`forward.model`）；不传则沿用 case 配置。

在线时 `case.yaml` 只做 init（网格、测点坐标、井轨迹、初值、相似比）；`observations` /
`series` 可以没有（即使写了也会被忽略，观测一律走 TCP）。采集端连上 `--tcp-port` 后这条
连接一直复用，每一拍一个长度前缀帧。进程在 `0.0.0.0` 上监听，`--ip` 只用于把结果发到
仪表盘。

二氧化碳吞吐在 YAML 里设 `inversion.model: black_oil_3phase`。注入走种类 2，采出
水/油/气走种类 3/4/5。

## 文档

| 文档 | 内容 |
|---|---|
| [example/README.md](example/README.md) | 联调包使用说明 |
| [docs/接口协议.md](docs/接口协议.md) | 协议 v2 逐字节规范 |
| [docs/联调指南.md](docs/联调指南.md) | 三角色三端口拓扑、启动顺序、故障排查 |
| [docs/案例库.md](docs/案例库.md) | `example/case.yaml` 字段参考 |
| [docs/相似换算.md](docs/相似换算.md) | 几何-运动相似准则与跨尺度换算 |

## 联调命令（`example/`）

`example/` 里是完整联调包：配置、数据、调用脚本、接收脚本、使用说明、requirements。
Linux `src.so` 由 GitHub **pack-linux** 打进该目录。协议 v2 冻结不变。

```bash
# 终端 1：接收端
python example/receive_fields.py --lab-port 9001 --field-port 9002

# 终端 2：后端（真实 GEM 数据，27 测点 / 5 井 / 120 拍）
python -m src example/case.yaml --tcp-port 9000 --ip 127.0.0.1 \
    --lab-port 9001 --field-port 9002

# 终端 3：调用端（读 example 时序，逐拍发 STEP）
python example/send_steps.py --host 127.0.0.1 --port 9000
```

## 协议 v2 速览

- 字节序小端，`PROTOCOL_VERSION = 2`，魔术 `"RB"`。
- **TCP 入站**：长度前缀帧 `[u32 len][u8 ver][u8 type][payload]`，消息 HELLO / READY /
  STEP / ACK / NAK / BYE。后端发 HELLO（网格/测点/井/枚举/单位），采集端逐拍发 STEP，
  后端回 ACK/NAK（含错误码）。控制消息 JSON，STEP 二进制。
- **UDP 出站**：报头 `[2s"RB"][u8 ver][u8 type][u32 stream_id][u32 seq]`，消息 MANIFEST /
  FIELD / RESULTS / END。FIELD 带 CRC32，END 带总报数与整体 CRC，接收端可检测丢包/损坏；
  新增 RESULTS（井底压力、测点拟合误差等井级/测点级结果）。
- 全场 flat 数组按 `cell = k*ny*nx + j*nx + i` 排序，reshape 成 `(nz, ny, nx)`。

## 长会话性能（在线）

在线模式用**滑动窗口 + 后台反演**，长时间跑不会越来越慢、也不会因反演慢阻塞采集。窗口大小（2 拍，只看上下相邻时刻）已内置为常量，无需配置：

- 每拍只插值最新一拍（O(网格数)），结果立即发 UDP；
- 岩石反演（k/φ）放**后台线程**、每拍触发，k/φ 在下一拍随流更新；
- 只保留最近 2 拍（当前 + 上一拍）。

详见 [`docs/联调指南.md`](docs/联调指南.md)。

## 测试与自检

```bash
python scripts/verify_examples.py   # 一键校验 example/case.yaml
python -m pytest                    # 单元/接口测试
```

**完整使用说明见 [example/README.md](example/README.md) 与 [docs/联调指南.md](docs/联调指南.md)**。

## Linux `.so` (GitHub)

pack-linux 把 `src.so` 和 `example/` 打进同一个 zip。

- 手动：GitHub → Actions → **pack-linux** → Run workflow
- 发版：`git tag v0.4.1 && git push origin v0.4.1`（附件 `reservoir-backend-*-linux-so.zip`）
- 本机 Linux：`./build_linux.sh`

解压后进入 `example/`：

```bash
cd example
pip install -r requirements.txt
python receive_fields.py --lab-port 9001 --field-port 9002
python reservoir.py case.yaml --tcp-port 9000 --ip 127.0.0.1 \
    --lab-port 9001 --field-port 9002
python send_steps.py --host 127.0.0.1 --port 9000
```

`src.so` 旁边不要再放 `src/` 目录，否则 Python 会加载源码包而不是这个扩展。
