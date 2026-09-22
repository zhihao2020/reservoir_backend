# reservoir_backend 0.4.0

Laboratory 3D reconstruction: probe/well series → full-grid pressure, saturation, porosity, permeability.

库版本为 `src.version.__version__`（`0.4.0`）：

```bash
python -c "import src; print(src.__version__)"
python -m src --version
python -m src --help
```

## 从源码运行

```bash
pip install -e ".[dev]"
python -m src path/to/case.yaml                          # 离线：文件 → 文件
python -m src path/to/case.yaml -o results/run           # 落盘（fields.npz / *.npy / *.csv / *.json）
python -m src path/to/case.yaml --tcp-port 9000 --ip 127.0.0.1 --lab-port 9001 --field-port 9002
```

`--tcp-port` 是测点/注采的入站 TCP（无默认值）；`--lab-port` / `--field-port` 是反演
结果的两路 UDP 出站（实验 / 矿场尺度）；`--control-port` 是可选的控制端口（应答
RESEND 重发请求）。`--model {none,black_oil,compositional}` 选前向饱和度模型（覆盖
YAML 里的 `forward.model`），默认 `none`（克里金，联调默认）。`compositional` 是溶液气
（solution-gas）组分模型：CO₂ 溶解进油（`Rs = rs_slope·p`，亨利定律），组分
`C = sg/Bg + Rs·so`（自由气 + 溶解气），过饱和时析出自由气。不写 `-o` 不落盘。

在线时 `case.yaml` 只做 init（网格、测点坐标、井轨迹、初值、相似比）；`observations` /
`series` 可以没有（即使写了也会被忽略，观测一律走 TCP）。采集端连上 `--tcp-port` 后这条
连接一直复用，每一拍一个长度前缀帧。进程在 `0.0.0.0` 上监听，`--ip` 只用于把结果发到
仪表盘。

二氧化碳吞吐在 YAML 里设 `inversion.model: black_oil_3phase`。注入走种类 2，采出
水/油/气走种类 3/4/5。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/接口协议.md](docs/接口协议.md) | 协议 v2 逐字节规范（TCP/UDP 帧、枚举、manifest/results schema、单元排序、离线格式） |
| [docs/联调指南.md](docs/联调指南.md) | 三角色三端口拓扑、启动顺序、分步走查、故障排查 |
| [docs/案例库.md](docs/案例库.md) | 各案例用途/网格/模型/耗时 + `case.yaml` 字段参考 |
| [docs/相似换算.md](docs/相似换算.md) | 几何-运动相似准则与跨尺度换算 |

## 案例库

`examples/` 下 6 类案例（`small` / `twod` / `model_compare` / `offline` / `online` /
`shale_oil`），全部可直接运行，不依赖 GEM。发射脚本 `examples/online/send_steps.py` 是协议
v2 的参考实现，可对任意案例目录复用。仪表盘（UDP 接收端）由对接方按协议实现。

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

详见 [`docs/联调指南.md` §7](docs/联调指南.md)。

## 测试与自检

```bash
python scripts/verify_examples.py   # 一键校验示例库（应打印 "all examples verified"）
python -m pytest                    # 单元/接口测试
```

**完整使用说明见 [docs/联调指南.md](docs/联调指南.md)**（从零到跑通的 step-by-step，
含离线/在线/换自己采集端与大屏）。

## Linux `.so` (GitHub)

产物就是一个 `src.so`。numpy / scipy / PyYAML 仍用 pip 安装。

- 手动：GitHub → Actions → **pack-linux** → Run workflow
- 发版：`git tag v0.4.0 && git push origin v0.4.0`（附件 `reservoir-backend-*-linux-so.zip`）
- 本机 Linux：`./build_linux.sh`

解压后目录里只有 `src.so`（另有 `VERSION` / `README.txt`）：

```bash
pip install numpy scipy pyyaml
PYTHONPATH=/path/with/src.so python3 -m src --version
PYTHONPATH=/path/with/src.so python3 -m src case.yaml --tcp-port 9000 --ip 127.0.0.1 --lab-port 9001 --field-port 9002
```

`src.so` 旁边不要再放 `src/` 目录，否则 Python 会加载源码包而不是这个扩展。
