# 项目状态

主线：30 cm 页岩油实验数字孪生。V1 产品 Case 是 `examples/lab_v1/`（组分 DPDP FIM + 面注采 + \(\theta=(\log C_f,\log T_{mf})\) + ES-MDA）。历史窗反演后冻结 \(\theta\)，再正演整段井控。\(k_m,\varphi_m,\varphi_f\)、PVT 固定。**M2 是 CMG-GEM 尺子**，不是用户入口。online / UDP / 30³ ensemble 冻结到 M3。`examples/lab/lab_cf.yaml` 是粗网格夹具。黑油 IMPES/FIM 已删除。

| 状态 | 含义 |
|------|------|
| **已验证** | 有实现与针对性测试，且测试测的是规格要求的行为 |
| **MVP** | 可用，假设写在 `docs/model_assumptions.md` |
| **不做** | P0 明确排除 |

## 能力表

| 能力 | 状态 | 入口 | 证据 |
|------|------|------|------|
| 300 mm / 10 mm → 30³，体积 0.027 m³ | 已验证 | `CartesianGrid.uniform` | `tests/grid/test_grid_lab.py` |
| 控制 / 观测分离 | 已验证 | `ControlSeries` / `ObservationSeries` | `tests/observation/test_observation_operator.py` |
| 非格点观测算子 | 已验证 | `ObservationOperator` | `tests/observation/test_observation_operator.py` |
| 黑油 IMPES / 顺序隐式 / 黑油 FIM | **已删除** | — | 产品正演是组分 DPDP FIM |
| 线性高斯 LM | 已验证 | `inverse.lm.run_lm` | `tests/inverse/test_lm_linear.py` |
| 冻结 θ 的正演 | MVP | `DigitalTwin.forward_from_posterior` | `tests/inverse/test_forecast.py` |
| CLI validate/simulate/invert/forecast/apply | MVP | `reservoir` | `tests/cli/test_cli.py` |
| 实验室 apply（历史反演 → 组分正演） | 已验证 | `reservoir apply` | `tests/cli/test_apply.py` |
| 测点 CSV（SI / 分钟·kPa、hold-out、无 --demo） | 已验证 | `io.case` / `apply` | `tests/cli/test_apply.py`、`tests/io/test_case_csv.py` |
| 点估计场 \(F(\hat\theta)\) | MVP | `DigitalTwin.reconstruct` | `tests/inverse/test_reconstruct_uq.py` |
| CSV 控制/观测 IO | 已验证 | `io.case` | `tests/io/test_case_csv.py` |
| 任意深度柱面测点 | 已验证 | `column_sensors` | `tests/observation/test_observation_operator.py` |
| 自洽 \(\theta=(C_f,T_{mf})\) 收回 | MVP | `make_lab_v1_face_twin` | `tests/inverse/test_log_cf_tmf.py` |
| 组分 EXAMPLE 孪生（等温气–油，C1–nC10） | MVP | `solver.fi_comp`、`eos/`、`comp/` | `tests/cases/test_comp_twin.py`；`examples/compositional/comp_example.yaml`。定流量井 \(p_{\mathrm{wf}}\) 进 \(H\)。2-region LM invert（数据 nRMSE 下降、对比度方向对）。不是济阳 GEM。单孔 FIM Jacobian：局部闪蒸差分 + 解析 TPFA；线性解 SuperLU/ILU-GMRES/CPR（`tests/solver/test_comp_analytic_jac.py`） |
| 组分 immiscible 水相 | MVP | `CompSpec.has_water` | `tests/cases/test_comp_water.py`、`examples/compositional/comp_example_water.yaml`。水进 \(F\) 和 \(H\)（\(S_w\)+率井 BHP）；2-region LM invert，对比度方向对。水不进 PR |
| 公开 PR 牌加载 | 已验证 | `io.eos_load.load_eos_card` | `tests/physics/test_eos_load.py`；fixture 抄 OPM `1D_COMP` 数字。缺文件拒绝 |
| 统一 invert run report | MVP | `twin.run_report`、`cli.reporting` | `invert.json` + `residuals.csv` |
| check83 十二问验收 | MVP | `twin.acceptance` | `check83.json`；`docs/check83_acceptance.md`；`tests/twin/test_check83_report.py` |
| LM 后验小 ensemble Ne=8 | MVP | `inverse.post_ensemble` | `k_mean.npy` / `k_std.npy`；`tests/inverse/test_post_ensemble.py` |
| Forecast 时间外推尺子 | MVP | `synthetic.make_forecast_split_case` | `tests/inverse/test_forecast.py` |
| Scalar \(C_f\) log 参数化 | 已验证 | `LogConductivityParameterization` | `tests/inverse/test_log_conductivity.py` |
| 联合 \(\log C_f,\log\beta_{mf}\) | MVP | `LogCfTmfParameterization` + 分层 ES-MDA | **M1a PASS**。**M1b 主 Case B PASS**（Cf 0.89% / Tmf 0.69%）；T2 与 seed 稳健性未过。**M1c FAIL（已接受）**：实验室可行方案 0 个 \(D_{C_f,5\%}>2\)。不要再调 ES-MDA。 |
| ES-MDA（log \(C_f\)） | 已验证 | `inverse.esmda`、`twin.history_match` | `tests/inverse/test_esmda.py`、`test_esmda_cf.py`。线性高斯收回；合成无噪声 \(C_f\) 向真值靠近；后验 P05/P50/P95 |
| Parameter EnKF（在线一步） | MVP | `inverse.parameter_enkf` | `tests/inverse/test_parameter_enkf.py`。**M3 才解冻**；当前 M2 不是这条路 |
| CMG-GEM 交叉验证流水线 | MVP | `twin.cmg_benchmark`、`examples/lab_v1/cmg_gem/` | Case B + 四真值历史合成尺子。`physical_3d` 15³ 单孔 7 组分：逐连接井筒静压 + HZYT/LBC 接线后真实到达 864 / 12096 s；压力 RMSE **527.00 / 233.13 Pa**。864 s 比恒压基线 163.24 Pa 差，**physical_3d 未通过 M2a**。详见本页最新精度轮。 |
| DualContinuumState / transfer / ForwardModel adapter | MVP | `domain.state`、`physics.transfer`、`solver.forward_adapter` | `tests/domain/test_dual_state.py`、`tests/solver/test_forward_adapter.py` |
| DPDP DualRock + 组分 transfer | 已验证 | `physics.dual_rock`、`physics.transfer.ComponentTransfer` | `tests/physics/test_dual_rock.py`、`test_component_transfer.py` |
| DPDP compositional FIM D0–D4 | 已验证 | `comp.dual_residual`、`solver.fi_comp_dual` | `tests/comp/test_dual_d0.py`、`test_dual_d1234.py`：守恒相对误差 < 1e-4 |
| Sparse DPDP Jacobian + context | 已验证 | `solver.dpdp_jacobian`、`solver.dpdp_context`、`solver.linear` | `tests/comp/test_dpdp_sparse.py`、`test_dpdp_scale.py`（5³） |
| Scalar \(C_f\) ES-MDA on DPDP | 已验证 | `LogConductivityParameterization`、`HistoryMatchWorkflow` | `tests/inverse/test_esmda_cf.py`；`m=\log(C_f/C_{\mathrm{ref}})` |
| Observation QC | 已验证 | `observation.qc` | `tests/observation/test_qc.py` |
| Online checkpoint / UDP | MVP | `twin.online`、`io.udp_api` | `tests/twin/test_online_checkpoint.py` |
| 面注采 `make_face_port` | 已验证 | `ports.flow.make_face_port` | `tests/ports/test_face_port.py` |
| V1 产品 Case `examples/lab_v1/` | MVP | `case.yaml` 30³ + `case_dev.yaml` | `tests/io/test_lab_v1.py` |
| 传感器 CSV `sensor_id,…,sigma` | 已验证 | `io.case._read_sensors_csv` | `tests/io/test_lab_v1.py` |
| 饱和度默认 bulk | 已验证 | `ObservationOperator` | `tests/observation/test_sensor_medium.py` |
| Innovation trigger + \(E_p\) | MVP | `twin.loops.TwinLoops` | `tests/twin/test_loops.py` |
| LinearSolveResult 诊断 + Schur CPR | MVP | `solver.linear` | `tests/solver/test_linear_result.py` |
| Lab Gate（面 BC，非 scale gate） | MVP | `scripts/lab_v1_gate.py` | `tests/scripts/test_lab_v1_gate_schema.py` |
| TwinRuntime / FieldStore | MVP | `runtime/` | `tests/runtime/test_runtime.py` |
| 实验数据集 `experiments/EXP001` | MVP | `runtime.replay` | `reservoir replay experiments/EXP001` |
| 真实 PVT YAML + 3–6 lumping | MVP | `examples/lab_v1/pvt.yaml`、`eos.pvt_ingest` | `tests/physics/test_realfluid_flash.py` |

## 排除

- 四场插值「反演」（已删除）
- 每 cell 独立反演 27k 个 K（非默认）
- Archie / EM / acoustic 通用反演（已删除）
- 逐格 \(K\)、coarse-field、缝长/SRV、济阳矿场吞吐、黑油 IMPES/FIM（已删除）。PINN、MPFA、动态 AMR、热尚未做。产品 `apply`：历史窗反演 \(\theta=(\log C_f,\log T_{mf})\)，再组分 DPDP 正演。
- 旧 `pipeline/` 产品路径（已删除）

## 实验室默认（2026-08 重构）

- 30 cm 立方，10 mm 网格，探头直径 6 mm（`H` 在插值场上做球平均）
- V1 产品 Case：`examples/lab_v1/`（组分 DPDP FIM + 面注采 + \(\theta=(\log C_f,\log T_{mf})\) + ES-MDA）。`lab_cf.yaml` 仅粗网格夹具
- 产品 invert 默认 ES-MDA。`case_dev.yaml` 可用 `algorithm: auto`（先 LM）。\(k_m,\varphi_m,\varphi_f\)、PVT 固定
- 三维 p/S 是 \(F(\hat\theta)\) 重建。M2 尺子是 CMG-GEM 交叉验证，不是用户入口
- `reservoir harness` 已删除
- 30 cm 产品开发计划（活文档）：`docs/lab_product.qmd`

## physical_3d 尺子：已测 / 未关

**已测（2026-09-07）**

- GEM `.out` 官方：t=0.01 d 压力 span ~1.5 kPa（max 在注入井 8,8,11）；t=3 d span ~1.6 kPa；Sg=0；INJ ~1e-7 m³/day，PROD1–3 流量 0。
- 当前 `parse_gem_out_maps` 重打包 hidden：时刻 0 / 864 / 12096 / 86400 / 259200 s。旧 pack 的 8.64 s + 0.45 MPa「采井漏斗」是坏图，不是物理。
- `F_ours(k_GEM)` vs 插值 GEM，t=8.64 s：压力 RMSE **1.6 Pa**，本模型全场 50.000000 MPa。图：`results/lab_v1/cmg_gem_physical_3d_compare/`（gitignored）。**不是 M2a PASS。**
- 解析回归：`test_parse_physical_3d_gem_out_if_present`、`test_parse_gem_out_ignores_timestep_table_times` 过。

**未关（交给后续精度轮）**

1. 历史 433 s underflow：当前 864 / 12096 s 已真实到达，见最新精度轮；不再列为当前阻塞。
2. GEM `*GEOMECH` 现由 `physical_3d/case.yaml` 的 `geomech:` 打开 Cartesian 线弹性（`nocouperm: true`，不改 k）。产品 DPDP 默认关。**停止加大 cpor。** 全场 RMSE 与井柱 RMSE 分开报。
3. 黏度已接线：physical_3d `hzyt` → 现有 LBC/Jossi；GEM 完整 HZYT/PVC3 等价性仍未验证。
4. 单孔 Jacobian 历史约 1 的误差已在 D0/D2 混跑复现：CSC 缓存拓扑混用；现已修复，列 FD 误差 1.11e-9。
5. D2 属于产品 DPDP invert 所用 F。underflow 的缓存碰撞根因已修复；原测试保留，20 s 完成。
6. nrmse_p 在 GEM span 只有几十 Pa 时会被印出噪声放大；看 RMSE_Pa / `nrmse_p_sigma`。
7. 流量一旦离开 50/50 近零区，要再对 Peaceman WI vs GEM `*GEOMETRY *K 0.003 0.34`。

约束：不重跑 GEM、不调 ES-MDA、不改 50/50 井控去造漏斗、不宣称 M2a PASS。产品 DPDP 力学关；physical_3d 由 YAML `geomech` 打开，`*NOCOUPERM` 不改 k。


### physical_3d 精度复核（2026-09-07，当前 main 工作环境）

- 原代码真实运行到 **864.0 s**：9 个接受步，末步 dt=157.49794238683137 s，Newton=1，scaled residual ratio=0；50/50 BHP、k/PVT/井控未变。加 8.64 s 报告点时 10 步。历史 ~433 s underflow 本环境未复现，因此没有盲改 dt/Newton、linear.py 或放宽残差，历史原因仍未解释。未重跑 GEM、未调 ES-MDA、未处理 Dual D2；**不是 M2a PASS**。
- 删除旧 ours.npz 后按请求命令重算 864 s：**RMSE_p=163.2357985428623 Pa**，GEM min/max=49,999,800 / 50,001,300 Pa，ours=50,000,000 / 50,000,000 Pa，Sg RMSE=0。8.64 s RMSE=1.632357985428623 Pa（GEM 地图由 0/864 s 插值，不是 GEM 报告）。中文标签及相对 50 MPa 的 Pa 色标保留。
- 比较默认取首个实际 GEM 报告，underflow 不再自动回退到 8.64 s；缓存时刻不匹配或标记 truncated 则重算。forward_at_theta 检查真实接受时刻，禁止 nearest-state 冒充目标快照。

井诊断：两时刻 ours 所有井 BHP 和全部完井块压力均为 50,000,000 Pa；储层率均为 0 m³/day，well_molar_sources 各组分及总量均为 0 mol/s。下表 GEM 块压来自 hidden 的完井区 min/max，列值为 p−50 MPa（Pa）；864 s 总率来自 sanwei_co2.out 首个 Well Summary at Reservoir Conditions，保留原符号。

|井 (i,j)|连接数|ours ΣWI (m³)|GEM 块压差 8.64 s（插值）|GEM 块压差 864 s|GEM 储层总率 864 s (m³/day)|
|---|---:|---:|---:|---:|---:|
|INJ (8,8)|11|3.00004300652e-17|0 … 13|0 … 1300|3.6903e-7|
|PROD1 (3,13)|5|1.36365591205e-17|−2 … 1|−200 … 100|0|
|PROD2 (13,13)|15|4.09096773616e-17|−2 … 2|−200 … 200|0|
|PROD3 (3,3)|15|4.09096773616e-17|−2 … 2|−200 … 200|0|
|PROD4 (13,3)|5|1.36365591205e-17|0 … 2|0 … 200|−6.1186e-9|

- GEM 各井首连接 BHP 打印 5.0000e4 kPa（参考 BHP=50 MPa）；深部连接 BHP 打印可达 5.0002e4 kPa，不能把其有限位数表格反推成精确块压。GEM 8.64 s 无井报表，BHP/率不插值捏造。GEM INJ 地面气量 1.46264e-7 **M m³/day**，与储层总率不是同一单位/口径；不与 mol/s 直接相等比较。
- `.out` 五井回显 `*GEOMETRY *K 0.003 0.34 1.0 0.0`。ours peaceman_wi 用 dz=0.02 m、re=0.34√(dx dy)=0.0068 m、rw=0.003 m、skin=0、k=1.776e-17 m²，WI=2πk dz/ln(re/rw)=**2.727311824109158e-18 m³/连接**。这是本模型计算值，不是 GEM 打印 WI；GEM 等效 WI/非零率行为尚未验证。力学仍在 F 外，HZYT/LBC 差异保留。
- 可复现诊断：`python scripts/lab_v1_physical_3d_diagnostics.py`，逐连接 WI、两时刻块压/源项及来源保存到 `results/lab_v1/cmg_gem_physical_3d_compare/well_diagnostics.json`。baseline.log、progress.json、compare.json、pytest.log 同目录（gitignored）；本节为持久摘要。
- Jacobian 原测试本环境先有 6 passed；扩展 ±200,000 Pa 压力梯度后 8 passed，三个列 FD 最大误差均 **1.11119469e-9 < 0.005**，仍用原物理单位缩放和容差，未删测试。历史 ~1 误差未复现，不归因于已修复的解析导数。
- 验证命令：`python -m pytest tests/twin/test_physical_3d_precision.py tests/solver/test_comp_analytic_jac.py tests/twin/test_cmg_benchmark.py tests/comp/test_bhp_injector.py tests/physics/test_lbc.py -q -s -p no:cacheprovider`。输出：**50 passed in 3.30s**（包括 test_parse_physical_3d_gem_out_if_present 与 test_parse_gem_out_ignores_timestep_table_times）。
- 重算命令：`python scripts/lab_v1_cmg_compare_plot.py --case examples/lab_v1/cmg_gem/physical_3d/case.yaml --hidden examples/lab_v1/cmg_gem/physical_3d/export/hidden --out results/lab_v1/cmg_gem_physical_3d_compare --t-end 864`。

### Forward / invert 精度轮（2026-09-07，f8dce74 工作树）

**结果与边界**：井筒静压 + 注入井交叉流 + `*CPOR` + 油格子流度后，注入井柱跟上 GEM 水头；全场 RMSE 442 Pa（仍不是 M2a）。无 GEM 重跑、无储层重力/力学进 F、无 49.5 MPa、无 ES-MDA 调参。M1b Case B 未重跑认证。

井模型：Peaceman BHP 使用 `p_conn=p_ref+rho_wb*g*(z_ref-z_conn)`，z 向上；默认最高连接为 BHP 参考，physical_3d YAML 显式写出 INJ/PROD1–3 的 z_ref=0.29 m、PROD4 的 0.09 m。注井用注入组成，采井用本连接流体组成，在中点井筒压力、T 下闪蒸密度；没有拟合密度。Peaceman 连接默认不允许反向流动，可用 `allow_crossflow` 显式开启；面端口不变。井筒静压独立于 `physics.gravity: false`。BHP 源项的局部导数按组分批处理，保留列 FD 对照。

- 864 s 尺子拆开：`rmse_p_pa` 全场，`rmse_p_inj_column_pa` 注入井柱。加牌上 Biot 体积储集 α²/K_dry=1/11.9 GPa（不加大 *CPOR）后：**全场 437 Pa**，**注入井柱 41 Pa**。远场均值 ours +524 Pa vs GEM +98 Pa，与加储集前几乎相同（α²/K_dry 只比 *CPOR 多 ~7%）。缺口交给 YAML `geomech` 线弹性，停止加大 cpor。图标题同时写两项。不是 M2a PASS。
- GEM 原始井报表来自 `results/lab_v1/cmg_gem_physical_3d/sanwei_co2.out`，在 `Well Summary at Reservoir Conditions at 1.0000E-02 days` 直接读取 **BHP-Pblock 最后一列**（kPa ×1000），共 51 条连接。没有从 5 位有效数字 BHP 反推压差。原文摘录持久保存于 `tests/fixtures/gem_physical_3d_wells_864.txt`，解析器拒绝不存在的 8.64 s 井报表。

下表全为 Pa；“井模型@GEM块压”是隔离测试：仅测试井方程，使用 hidden 的实际 GEM 块压与固定初始组成，不进入正演或反演。GEM 地图打印分辨率 100 Pa，五口井最深连接的这一隔离误差均 <50 Pa；完整 F 的块压仍不匹配。

|井 / GEM I,J,K|GEM BHP-Pblock|井模型@GEM块压|完整 F_ours BHP-Pblock，864 s|
|---|---:|---:|---:|
|INJ 8,8,1|14.460|—|−280.264|
|INJ 8,8,11|126.780|142.487|12.863|
|PROD1 3,13,1|171.770|—|−50.050|
|PROD1 3,13,5|598.900|557.327|119.419|
|PROD2 13,13,1|171.690|—|−52.895|
|PROD2 13,13,15|2109.900|2100.646|1602.889|
|PROD3 3,3,1|171.870|—|−52.895|
|PROD3 3,3,15|2110.000|2100.646|1602.889|
|PROD4 13,3,11|−10.544|—|−142.979|
|PROD4 13,3,15|446.310|457.327|46.492|

- 864 s ours INJ 储层率 **1.34692e-7 m³/day**，GEM **3.6903e-7**；ours PROD1/2/3 分别 −1.70131e-8 / −1.85167e-8 / −1.85167e-8，GEM 均 0；PROD4 ours −6.54269e-8，GEM −6.1186e-9。全连接数值在 `well_diagnostics.json`；复算诊断命令：`python scripts/lab_v1_physical_3d_diagnostics.py --cache results/lab_v1/cmg_gem_physical_3d_compare/ours.npz`。缓存的 p/z 仅用于诊断源项，不重建总在位摩尔数。
- 12096 s 已运行相同 compare 命令，仅 `--out results/lab_v1/cmg_gem_physical_3d_compare_12096 --t-end 12096`：**74 个接受步**，实际 t=12096.000000000002 s，无截断；RMSE_p **233.13450624912585 Pa**，ours 范围 50,000,050.0755 … 50,001,429.6129 Pa，GEM 49,999,800 … 50,001,400 Pa，Sg RMSE=0，缓存报告摩尔守恒相对误差 1.39666e-10。

黏度：`pvt_co2.yaml` 的 `visc_model: hzyt` 选择已有 LBC/Jossi；卡的 MIXVC=1、VISVC=VCRIT、VISCOEFF 多项式与实现对应。体相不再硬编码常 μ，注入井流体沿用密度相关 μ；带 VCRIT 的未指定模型卡默认 LBC，缺 VCRIT 卡才默认常数。不改 Tc/Pc/Mw/VCRIT，不捏造济阳参数。这是模型接线验证，尚非 GEM 全 HZYT/PVC3 内部行为的等价证明。物理假设见 `docs/model_assumptions.md`。

DPDP 修复证据：

- 原 D2 单独跑通过；原 `D0 -> D1–D4 -> single Jacobian` 合跑为 **5 failed, 15 passed in 35.71s**（D2 underflow、三项单孔列 FD、单孔 Newton）。根因 `_csc_cached` 仅以 `(n_u, COO条数)` 命中缓存，不同网格/单双孔布局复用了错误的 COO→CSC 映射。现逐次核验 rows/cols 拓扑，维数相等不再意味着拓扑相同；没有删 D2、减小时窗或放宽 Newton 容差。
- `ComponentTransfer.compute` 残差乘 `transfer_multiplier`，原 `_transfer_coo` 导数漏乘；现一致地乘一次。新增异组成、双向 transfer、倍率 0.25 / 1 / 4 的所有列 FD 检查，最大相对误差分别 **5.45e-7 / 1.18e-7 / 2.71e-7**（门槛 5e-4）。单孔三个压力偏置的列 FD 误差均 **1.11119469e-9**（原门槛 0.005）。
- 修复后 D0/D1–D4 + sparse + 单孔/DPDP Jacobian：**24 passed in 32.30s**，D2 原 20 s 时窗真实完成，守恒/分离极限断言保留。

反演（Ne / Na / inflation / prior_std / seed 全部保持既有测试值）：

|夹具 / 指标|本轮修复前|修复后|
|---|---:|---:|
|scalar Cf，log Cf 真值 / 先验|1.609438 / 2.809438|相同|
|scalar Cf，log Cf 后验|2.727992428|2.727992428|
|scalar Cf，misfit 首→末|0.01482475→0.01489512|相同|
|joint 4×2×1，Cf 相对误差|0.005727%|2.162019%|
|joint 4×2×1，Tmf 相对误差|3.943207%|3.882916%|
|joint Cf P50 (m²)|9.99942731e-13|9.78379807e-13|
|joint Tmf P50（真值 2）|2.078864137|2.077658327|
|joint holdout 白化 RMSE|0.091671083|0.100564417|
|joint holdout 后验/先验 RMSE 比|0.04005016|0.04398016|

联合夹具仍通过原 Cf<5%、Tmf<10%、holdout 改善的门槛，但 **Cf 与数据 RMSE 回退，未证明整体恢复精度提高**；Tmf 略改善、scalar Cf 不变。正确 Jacobian 改变自适应步序，恢复精度仍需用时间离散误差收敛检验，不能靠 ES-MDA 调参掩盖。`test_lab_cf_yaml_is_dpdp` 另有预存的过期断言：已是 `log_cf_tmf` 的 YAML 仍断言 n_params=1；现断言类型与 n_params=2，未改配置或恢复门槛。

验证日志都位于 `results/lab_v1/cmg_gem_physical_3d_compare/`（gitignored）：

- `python -m pytest tests/twin/test_physical_3d_precision.py tests/solver/test_comp_analytic_jac.py tests/twin/test_cmg_benchmark.py tests/comp/test_bhp_injector.py tests/physics/test_lbc.py tests/comp/test_well_hydrostatic.py tests/solver/test_dpdp_jacobian_precision.py tests/comp/test_dual_d0.py tests/comp/test_dual_d1234.py tests/comp/test_dpdp_sparse.py -q -s -o addopts='' -p no:cacheprovider` → **73 passed in 204.40s**（`pytest_verified.log`，包括实际 864 s）。随后新增的 GEM 块压隔离测试在 `tests/comp/test_well_hydrostatic.py` 单独执行 → **5 passed in 1.37s**。
- 井加载/黏度/井源项扩展组合 → **30 passed in 11.35s**（`well_visc_tests.log`）；产品 PVT/io/5³ DPDP → **14 passed in 2.81s**（`product_pvt_tests.log`）。
- 完整反演复验命令：`python -m pytest tests/inverse/test_esmda_cf.py tests/inverse/test_log_cf_tmf.py tests/inverse/test_auto_invert.py tests/inverse/test_lab_v1_recovery.py -q -s -o addopts='' -p no:cacheprovider`。`-o addopts=''` 必须保留，仓库默认跳过 slow 恢复测试。最终结果见本节收尾记录。

剩余阻塞：864 s 场误差与顶部井压差符号仍不匹配；井筒隔离测试通过不能代替储层 F 验证。GEM 完整黏度细节、WI/多连接约束与其他未建模项仍需独立证据。YAML 线弹性已进 physical_3d，远场仍是 +525 Pa vs GEM +98 Pa；停止加大 cpor，不加拟合系数。反演门槛通过但数值精度未整体提高。

### YAML 线弹性（2026-09-07）

`geomech.E_GPa` / `nu` / `biot` 是用户 case 输入，不是代码常量，不进 θ。physical_3d 默认 20 GPa 只对齐该副 GEM `*ELASTMOD`。产品 `examples/lab_v1/case.yaml` 写出这些键，`enabled: false`；DPDP 若打开会拒绝。Hex8 线弹性，\(V_p=\varphi V\exp(c_{\mathrm{por}}\Delta p)(1+\alpha\theta)\)，\(\theta=\nabla\cdot u\)，k 不改。打开后不再叠标量 α²/K_dry。

删除 `ours.npz` 后 864 s 重算（147 s，未截断）：**全场 RMSE 437.54 Pa**，**注入井柱 41.08 Pa**，远场均值 ours **+524.67 Pa** vs GEM **+98.10 Pa**。注入井柱 GEM k=1…11：ours 46/166/295/425/555/683/809/932/1050/1159/1247，GEM 0/100/300/400/500/700/800/900/1100/1200/1300。与标量 Biot 一轮几乎相同：E=20 GPa 无约束样品上 θ~10⁻⁸，不能把压力钉在井旁。**停止加大 cpor，不加拟合系数。** 剩余不是再拧力学模量。不是 M2a PASS。

### Biot ΔV_p=α ΔV_bulk + 注入井气体流度（2026-09-07）

孔隙体积原先写成 `(1+αθ)`，相对孔隙体积少了 `1/φ`（φ=0.0367 约 27 倍）。改为 Biot `dV_p=α θ V_cell`，即 `(1+(α/φ)θ)`。注入井对齐 GEM `Mob Wtd Gas Inj`：正向用注入 CO2 流度，反向仍用格子油流度。

864 s 重算（291 s，未截断）：**全场 RMSE 400.06 Pa**（此前 437.5），**注入井柱 69.60 Pa**（此前 41.1），远场均值 ours **+487.2 Pa** vs GEM **+98.1 Pa**。井底 ours 1427 vs GEM 1300；角井底 ours 454 vs 井报表 191。最上 3 段注入井仍有反流（GEM 11 段全正）。径向仍偏平。不是 M2a PASS。不加大 cpor。

随后：注入井在 sg=0 时改回油流度（GEM BHP-Pblock 127 Pa 对齐油 λ，不是 16 Pa 的 CO2 λ）。`geofac` 改为 Peaceman `0.14√2≈0.198`（牌上 0.34 把 \(r_e\) 算大，WI 约为 GEM 连接 PI 的 1/3）。864 s：**全场 RMSE 373.0 Pa**，**注入井柱 41.9 Pa**，井底 1355 vs GEM 1300，远场均值 +460 vs GEM +98。不是 M2a PASS。

复算：`python scripts/lab_v1_cmg_compare_plot.py --case examples/lab_v1/cmg_gem/physical_3d/case.yaml --hidden examples/lab_v1/cmg_gem/physical_3d/export/hidden --out results/lab_v1/cmg_gem_physical_3d_compare --t-end 864`。测试：`tests/physics/test_geomech.py`。
