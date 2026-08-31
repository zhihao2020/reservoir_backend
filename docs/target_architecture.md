# 目标架构

日期：2026-08-15  
对应：`docs/check.txt`、`docs/digital_twin_repository_audit.md`  
原则：简单、显式、可测、物理正确。不为旧 API / 旧测试加 shim。仓库目录按角色分层（`examples/` 算例、`validation/` 离线尺子、`reservoir_backend/` 产品包）；搬家不改正演公式。

---

## 1. 系统真正要算的问题

300 mm 立方试块的实验室数字孪生：

\[
x_{0:T}=F(m,u_{0:T}),\quad
d^{sim}=H(x,m),\quad
d^{obs}\rightarrow m^{posterior}
\]

- \(m\)：静态参数。V1 产品默认标量 \(\log(C_f/C_{\mathrm{ref}})\)（`examples/lab_v1/`）。两区 log \(K\) 仅遗留水驱演示。粗网格 6³ / 逐格 / zonal \(C_f\) 不是默认。
- 饱和度观测由上游传感反演给出 \(S_\alpha,\sigma,x,y,z,t\)。电阻率 / 电磁 / 声学原始反演不在 reservoir-core 范围。
- \(u(t)\)：控制。定流量注入、定压采出、注入组成等。同一端口同一时刻，压力和流量不能同时当严格 BC 又当反演数据。
- \(x_t=(p,S_w[,S_g])\)：动态状态。\(S_o=1-S_w-S_g\)。饱和度不是物性场。
- \(H\)：观测算子。传感器是空间中的 \((x,y,z)\)，不必落在网格节点或同一深度。一口井/一根探针上不同深度就是多个 `Sensor`（`column_sensors`）。
- 外部油藏正演引擎只是某种 \(F\)。本产品是 \(F+H+\) 反演。和 CMG「工况一样」指同一套 \(u(t)\) 和同一批测点坐标，**不是** \(F\equiv\) IMEX。

最终输出必须是后验统计，不是一张「真实 K」图：

- \(E[K],\sigma_K\)
- \(E[p],\sigma_p\)，\(E[S_w],\sigma_{S_w}\)
- 同化残差、hold-out 残差、ensemble 诊断

---

## 2. 数据流

```text
Experiment
    |
    +---- Controls u(t)          进入正演，不进 misfit
    |
    +---- Observations d_obs     进 misfit；含 hold-out 标记
    |
Prior Parameters m_prior
    |
ParameterMapper                  固定维数的 θ → 计算网格上的 K,φ
    |
Forward Simulator F              V1 Cf 路径：组分 DPDP（两套 TPFA + transfer）
    |
State Field x(t)                 观测面看裂缝连续体 p_f, S；主变量是 (n_f, p_f, n_m, p_m)
    |
ObservationOperator H            点 / 体积 / 端口
    |
Predicted Observation d_sim
    |
Assimilator (ES-MDA / Parameter EnKF)  在 θ=log Cf 空间更新
    |
Posterior Parameters m_post
    |
Forecast                         冻结 m，只用未来 Controls
```

禁止的回流：

- 测点 p 写入压力 Dirichlet，同时又当 \(d^{obs}\)。
- 测点 S 写入输运钉死，同时又当 \(d^{obs}\)。
- 控制流量 q 乘 \(f_w(S)\) 假装产水观测。
- 插值 S 或手绘舌头覆盖 \(F(m)\)。
- 后验与 nowcast k 做 0.7/0.3 混合。

---

## 3. Domain 依赖（单向）

```text
domain          实验、控制、观测、状态、参数（无 PDE）
    ↑
grid            Cartesian 拓扑；fields 主表示 (n_cells,)
    ↑
rock / fluids   K,φ 变换；相对渗透率；毛管；压缩性
    ↑
discretization  TPFA（P0）；MPFA 仅接口注释，不写空类
    ↑
ports           FlowPort：rate 或 pressure，作用在 cell/face 集合
    ↑
solver          压力、输运、时间步、线性求解
    ↑
observation     H：与网格解耦
    ↑
inverse         Parameterization、LM、诊断
    ↑
twin            calibrate / forecast / reconstruct（薄编排）
    ↑
io / cli        单位换算、YAML、结果
```

`validation/` 只依赖上面的公共 API，不反向改物理。

旧 `pipeline/` 上帝模块拆完即删除产品职责，不留兼容包装。

---

## 4. 模块边界

只在真正有第二种实现时才抽象。P0 用 dataclass + 函数。

### 4.1 domain

```text
Experiment
    geometry: 0.30 × 0.30 × 0.30 m
    controls: list[ControlSeries]
    observations: list[ObservationSeries]
    sensors: list[Sensor]

Control          # 进入 F
    port_id, kind ∈ {rate, pressure, composition}, times, values

Observation      # 进入 misfit
    sensor_id, kind ∈ {pressure, saturation, phase_rate},
    x_m, y_m, z_m, continuum, sigma   # CSV 合同；σ 必填
    times, values, sigma, holdout: bool
    饱和度默认 continuum=bulk（孔隙度加权裂缝/基质）

State            # x
    pressure: (n_cells,)
    sw: (n_cells,)
    sg: (n_cells,) | None

Parameters       # m，固定维数
    theta: (n_theta,)
    names, transforms, bounds
```

### 4.2 grid

P0 只实现 Cartesian。接口先留齐，避免全库写死 `field[k,j,i]`：

- `n_cells`, `cell_centers (n_cells, 3)`, `cell_volumes (n_cells,)`
- `neighbors(cell) -> list[cell]`
- `face_area`, `center_distance`（结构化内部实现即可）
- `reshape_ijk` 仅用于可视化 / IO

三种空间：

| 名字 | P0 |
|------|-----|
| Simulation grid | 30 × 30 × 30，dx = 10 mm |
| Parameter grid | 过渡：2-region log K；V1 目标为标量 \(C_f\)。无 coarse-field / 逐格 K |
| Observation geometry | 探头 6 mm；H 在插值场上做球平均。坐标不必落在节点上 |

### 4.3 physics（P0 = Model A + Model B）

| 模型 | 用途 | 内容 |
|------|------|------|
| A 单相 | 数值验证、压力 benchmark | TPFA，不可压或微可压 |
| B 两相不混溶 | 实验室 \(B=1\) 特例 | 与 D 同一套守恒，PVT 取单位体积系数 |
| C 三相 | 可选 | `*SWT`+`*SLT` Stone II；活油 \(R_s\) 进气守恒 |
| D 黑油油水 | 默认 \(F\)（CMG 虚拟实验） | \(b_\alpha(p)\)、岩石压缩、地面流量；油水不跟踪溶解气 |

30 cm 默认：重力可关（水平驱）；**毛管默认开 Brooks–Corey 或由 case 显式关**，禁止静默 Pc=0 还声称实验室物理完整。

### 4.4 ports

```text
FlowPort
    cells or faces
    control: rate | pressure
    composition (injector)

WellModel / Peaceman   # 可选，默认不用在 30 cm 入口
```

定流量：q 是 Control，端口压力是 Observation。  
定压：p 是 Control，端口流量是 Observation。

### 4.5 observation

```text
ObservationOperator
    PointSensor      三线性 / 体积加权 cell
    VolumeSensor     有限尺寸平均
    PortObservation  端口压力、总流量、相流量
```

正演**禁止**读实验 CSV。IO 在边界把 CSV 变成 Control/Observation。

### 4.6 inverse

P0 只做低维 θ 上的 Levenberg–Marquardt：

```text
Parameterization
    RegionParameterization / ContrastParameterization   实验室过渡路径（2-region log K）
    LogConductivityParameterization                     V1：m = log C_f（标量）

Assimilator
    LM                          过渡：白化 misfit + Tikhonov；FD Jacobian
    ES-MDA                      V1 目标（接口未接到 invert CLI）
    post_ensemble (optional)    Ne=8 高斯采样 around \(\hat\theta\)
```

V1 只反演等效裂缝导流能力 \(C_f\)，不反演 \(k_m\)、缝长、SRV。实验室 `apply` 在 ES-MDA 落地前仍可用两区 log K。

默认**禁止**每 cell 独立反演 27k 个 K。coarse-field 已删除。

### 4.7 twin

```python
twin.calibrate(history)   # LM on θ
twin.assimilate(post, new_obs)  # LM warm-start stub（增量更新 MVP）
twin.forecast(controls)   # freeze m
twin.reconstruct(time)    # F(θ̂) at one time
```

不在 twin 里写 PDE 或 Kalman 公式。

---

## 5. Forward 流程（P0 IMPES）

```text
for n in timesteps:                          # 自适应 Δt
    λ = relperm(S^n)
    T = tpfa(K, λ, μ)                        # 若 K 变才重装
    p^{n+1} = pressure_solve(T, ports, φ c_t V/Δt, p^n)
    q_faces = conservative_flux(p, T)        # 含边界面
    若 CFL 或 ΔS 过大: Δt ← Δt/2; continue
    S^{n+1} = explicit_upwind(S^n, q_faces, fw, ports)
    clip 前检查守恒；越界则缩小 Δt，禁止 clip-当成功
    报告 mass balance
```

复用：

- 压力：现有 `solve_steady_state_pressure_3d` + mobility
- 输运：以 `solver/saturation_solver.py` 为底，不用 `transport_saturation.py` 的 Python 三重循环当产品
- 时间：`cfl.estimate_stable_dt`，失败减步，禁止 `min(n_sub, 40)` 硬截断

时间一律秒。CSV 里 `30 min` 在 IO 换成 1800 s。删除 `dt >= 0.5 ⇒ days`。

---

## 6. Inverse 流程（P0 offline）

```text
1. 读 Experiment：划分 control / assimilate / hold-out / forecast 时段
2. 选 Parameterization（默认 2-region log K 或已知图 contrast）
3. 从先验均值起步，LM + FD Jacobian 更新 θ
4. 输出 \(\hat\theta\)、\(K=\mathrm{expand}(\hat\theta)\)、Hessian 对角 σ
5. 打印 assimilation RMSE 与 hold-out RMSE
6. 冻结 m，用未来 controls 做 forecast，评分
```

产品 invert **必须**能留出测点。`probe_split` 从「可选评估」改成 inversion API 的一等参数。

### 6.1 孪生 ≠ 再写一个 CMG / 开源仿真

| 做法 | 实际在做什么 | 本项目 |
|------|----------------|--------|
| 把 \(F\) 改到和 IMEX 一样，再收回 \(K_{\mathrm{CMG}}\) | 调参 / history match | **禁止**当主线目标 |
| 同一套井控 \(u(t)\)，用实验室 \(F\) 解释测点 | 数字孪生反演 | **要做** |
| 换另一套正演引擎当 \(F\) | 换正演引擎 | 以后可插，不改变 \(H\) 和 LM |

工况对齐清单（和 CMG 或真实实验）：

1. 控制 \(u(t)\) 相同：定压对、定流量、注入组成、开关井时刻。
2. 测点 \(H\) 相同：真实 \((x,y,z)\)，深度可以一口一个样。
3. 初值已知：\(p_0,S_{w0}\)。
4. 数据还要有信息：井控必须让 \(\Delta p\)、\(\Delta S_w\) 留在窗口里。

测点越多、种类越杂（不同深度的 \(p\) 和 \(S_w\)，再加**不是控制量**的流量），可辨识性越好。稀疏单平面压力几乎看不见层。

产品尺子是自洽反演（贴回本正演），不是场 Dice 对 CMG。三维 p/S 是 F(m_post) 重建。

跨模拟器 / 真实生产的通过标准是观测、hold-out、预报变好。后验 \(K\) 是「我们的 \(F\) 下能解释数据的等效渗透率」，不是 CMG 格子 \(K\)。

---

## 7. Online twin 流程（P2，现在只定边界）

```text
calibrate → log C_f posterior
实验继续 → 新 d_{t+1}
forecast parameters (random walk)
y = H(F(m))
parameter EnKF on log C_f only
physical rerun of F
```

禁止把压力/饱和度写进 EnKF 状态向量。P0 的 API 把 `calibrate` 和 `forecast` 分开。

---

## 8. 配置（只暴露工程量）

```yaml
geometry:
  size_m: [0.3, 0.3, 0.3]

grid:
  type: cartesian
  spacing_m: 0.01

physics:
  model: two_phase_immiscible
  capillary: brooks_corey      # 或 none（必须显式）
  gravity: false
  pvt: incompressible          # 或 cmg_seawater（矿场牌组）；mapping 预留标量 μ
  # pvt:
  #   preset: cmg_seawater
  #   mu_w: 1.1e-3             # 死油/不可压覆盖；活油表优先
  #   mu_o: 0.64e-3

inverse:
  parameterization: region   # default 2-region log K
  n_regions: 2
  max_iter: 8

experiment:
  controls: controls.csv
  observations: observations.csv
  holdout_sensors: [P05, S03]
  history_end_s: 3600
```

高级调参进 expert 段，不进默认 case。

CLI：

```text
reservoir validate case.yaml
reservoir simulate case.yaml
reservoir invert case.yaml
reservoir forecast case.yaml
reservoir synthetic bench.yaml
```

---

## 9. 验证门槛（重构时同步建立）

没有这些测试，不得把能力标成「已验证」。

| 级 | 测试 | 通过标准 |
|----|------|----------|
| 网格 | 0.30 m / 10 mm → 30³，总体积 \(0.3^3\) | 相对误差 < 1e-12 |
| 观测 | \(p=x+2y+3z\) 在非格点采样 | 插值误差达离散阶 |
| 单相 | 1D 解析压力与通量 | 已有压力测试迁过来 |
| 两相 | Buckley–Leverett 无毛管 | 锋面位置、剖面、含水率 |
| 守恒 | 每步 mass balance report | 相对误差阈值；clip 不得冒充守恒 |
| 毛管 | Pc 单调、毛细平衡 | solver 旧测试恢复并接线 |
| 反演数学 | 线性高斯 LM | \(\hat\theta\) 靠近真值，misfit 下降 |
| Synthetic | \(K_{true}\) 分层/通道；噪声已知 | data misfit 降；hold-out 降；真值在合理后验带 |
| 时间外推 | 前 60% 同化，后 40% 冻结预测 | 预测误差写入报告 |

旧测试若断言「测点 p 必须机器精度相等」，删除，不改正演去迎合。

---

## 10. P0 实施顺序（先物理，后目录）

不要第一周把文件搬进 `src/reservoir_backend/{domain,grid,...}` 而不改行为。

1. **Domain 切开**：`SensorSample` 拆成 Control / Observation；禁止双用。
2. **ObservationOperator**：非格点采样；正演不再 `well_cell_id` 取值当 H。
3. **时间与单位**：内部秒；删 day 启发式。
4. **IMPES Model B**：接线 saturation_solver + 边界通量 + 自适应 dt + 质量报告。
5. **毛管**：30 cm case 显式选择模型。
6. **Parameterization**：默认 2-region log K；LM 只更新 θ。
7. **产品 invert**：用 G(m)+H，去掉指示混合、舌头、0.7/0.3。
8. **Synthetic truth**：观测必须来自 \(H(F(m_{true}))\)。
9. **Hold-out + forecast** 测试与 example。
10. **目录**：`pipeline` / Archie / 济阳矿场 / IMEX 页岩 suite / 黑油 CMG 尺子已删。算例在 `examples/{lab,two_layer,channel,compositional}`。

---

## 11. 明确不做（本轮）

- compositional / 热 / THM / 化学反应
- GPU、分布式
- PINN / 深度代理 / VAE
- 动态 AMR、工业 ECLIPSE 兼容
- 同时写第二套工业级 Fully Implicit（FIM 为 MVP opt-in；默认仍顺序，直到放气闸门）
- 空的 IES/EnKF/MPFA 类
- 为旧四场 API 写 compatibility shim
- 运行时 import OPM/GEOS；产品符号与上游同名

---

## 12. 重构后允许保留的「诊断」

点插值 nowcast（旧四场）可以留在 `diagnostics/nowcast.py`，名称必须是 nowcast / interpolation，**不得再叫 inversion**。STATUS 不得把它标成主线已验证反演。
