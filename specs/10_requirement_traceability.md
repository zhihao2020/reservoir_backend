# Requirement Traceability

This document tracks original backend requirements against the current Python prototype.

| 原始需求 | 当前实现模块 | 对应测试 | 状态 | 备注 |
| ---- | ------ | ---- | -- | -- |
| 饱和度计算：电阻率反演 | `reservoir_backend.inversion.resistivity_archie.ArchieInverter` | `tests/test_archie_inversion.py` | Done | Supports scalar, ndarray, and `Field3D`; clips saturation and reports confidence. |
| 饱和度计算：电磁信号反演 | `reservoir_backend.inversion.electromagnetic.ElectromagneticInverter` | `tests/test_electromagnetic_inversion.py` | Partial | 已实现 linear / polynomial 经验标定反演；不是 Maxwell 方程物理反演。 |
| 饱和度计算：声波信号反演 | `reservoir_backend.inversion.acoustic.AcousticInverter` | `tests/test_acoustic_inversion.py` | Partial | 已实现 linear / polynomial 经验标定反演；不是完整 Gassmann / 全波形反演。 |
| 三维压力场重构 | `reservoir_backend.solver.pressure_solver` | `tests/test_pressure_solver_1d.py`, `tests/test_pressure_solver_2d.py`, `tests/test_pressure_solver_3d.py` | Done | Supports 1D/2D/3D Cartesian steady single-phase Darcy pressure, Dirichlet/no-flow, wells, mass balance. |
| 三维饱和度场计算 | `reservoir_backend.solver.saturation_solver` | `tests/test_saturation_solver_1d.py`, `tests/test_saturation_solver_3d.py` | Done | Explicit oil-water transport with face flux, upwind fractional flow, CFL checks, material balance. |
| 毛管压力模型 | `reservoir_backend.solver.capillary_pressure` | `tests/test_capillary_pressure.py` | Done | 已实现独立 Pc(Sw) 模型、数值导数和配置校验。 |
| 毛管压力通量 | `reservoir_backend.solver.capillary_flux` | `tests/test_capillary_flux.py` | Done | 已实现 Pc field -> capillary mobility field -> x/y/z face capillary flux。 |
| 1D 毛管输运耦合 | `reservoir_backend.solver.saturation_solver.advance_saturation_1d_with_capillary` | `tests/test_saturation_capillary_1d.py` | Done | 已完成 1D capillary water flux 与显式饱和度推进的可选耦合；capillary disabled 时保持原 1D 行为。 |
| 3D 毛管输运耦合 | `reservoir_backend.solver.saturation_solver.advance_saturation_3d_with_capillary` | `tests/test_saturation_capillary_3d.py` | Done | 已完成 x/y/z capillary water flux 与 3D 显式饱和度推进的可选耦合；默认 full pipeline 仍保持 capillary disabled。 |
| 配置驱动毛管 pipeline | `config/capillary_case.yaml`, `examples/run_full_pipeline_demo.py`, CLI runner | `tests/test_capillary_pipeline.py` | Done | 毛管压力可通过 YAML/CLI 可选启用；enabled case 输出 Pc、capillary flux 和 capillary report，默认 case 仍 disabled。 |
| 非均匀毛管验证 case | `config/capillary_gradient_case.yaml`, `initial_saturation` pipeline support | `tests/test_capillary_gradient_case.py` | Done | 已新增 step_x 非均匀初始饱和度算例，用于验证非零 Pc 梯度、非零 capillary flux 和 saturation front smoothing。 |
| capillary profiling | `scripts/profile_capillary_pipeline.py` | `tests/test_capillary_gradient_case.py` | Done | 记录 demo、capillary 和 capillary-gradient cases 的运行时间、CFL、毛管通量和物质平衡指标。 |
| 重力通量模型 | `reservoir_backend.solver.gravity_flux` | `tests/test_gravity_flux.py` | Done | 已实现独立 gravity potential / mobility / x-y-z face gravity flux 和配置校验。 |
| 竖直 1D 重力输运耦合 | `reservoir_backend.solver.saturation_solver.advance_saturation_1d_vertical_with_gravity` | `tests/test_saturation_gravity_1d.py` | Done | 已完成竖直 1D gravity water flux 与显式饱和度推进的可选耦合；gravity disabled 和零密度差保持无重力行为。 |
| 3D 重力输运耦合 | `reservoir_backend.solver.saturation_solver.advance_saturation_3d_with_gravity` | `tests/test_saturation_gravity_3d.py` | Done | 已完成 x/y/z gravity water flux 与 3D 显式饱和度推进的可选耦合。 |
| 配置驱动 gravity pipeline | `config/gravity_case.yaml`, `examples/run_full_pipeline_demo.py`, CLI runner | `tests/test_gravity_pipeline.py` | Done | gravity transport 已支持通过 YAML / CLI 可选启用；enabled case 输出 gravity flux 和 gravity report。 |
| combined capillary + gravity design | `specs/11_combined_capillary_gravity_design.md` | `tests/test_combined_transport_design.py` | Done | 已完成 combined flux、CFL、material balance、report schema、YAML 行为和后续任务设计。 |
| combined flux composer | `reservoir_backend.solver.water_flux_composer` | `tests/test_water_flux_composer.py` | Done | 已实现独立 Fw_adv / Fw_cap / Fw_grav 组合、CFL effective flux 和 combined report；尚未接入 saturation solver。 |
| combined capillary + gravity solver | `reservoir_backend.solver.saturation_solver.advance_saturation_3d_with_capillary_and_gravity` | `tests/test_saturation_combined_capillary_gravity_3d.py` | Done | 已实现函数级 combined 3D saturation transport，使用 water flux composer 组合 advective/capillary/gravity flux。 |
| combined pipeline case | `config/combined_case.yaml`, `examples/run_full_pipeline_demo.py`, CLI runner | `tests/test_combined_pipeline.py` | Done | 已接入配置驱动 full pipeline / CLI；同时输出 capillary flux、gravity flux 和 combined_report。 |
| combined profiling and validation | `scripts/validate_combined_pipeline.py`, `scripts/profile_combined_pipeline.py` | `tests/test_combined_validation.py`, `tests/test_combined_profiling.py` | Done | 已实现 combined_case 文件/物理范围/通量非零/material balance/dt sensitivity 验证，以及 demo/capillary/gravity/combined runtime profiling。 |
| release candidate documentation | `README.md`, `docs/*.md` | `tests/test_release_documentation.py` | Done | 已整理项目结构、模块能力矩阵、case 配置、CLI、validation/profiling、数值方法、限制/路线图和 release checklist。 |
| three-phase flow design | `specs/12_three_phase_flow_design.md` | `tests/test_three_phase_design.py` | Done | 已完成简化 water-oil-gas transport 的 scope、状态变量、bounds、相渗、fractional flow、CFL、material balance、YAML 和测试计划设计；未实现三相 solver。 |
| three-phase relperm | `reservoir_backend.solver.three_phase_relperm` | `tests/test_three_phase_relperm.py` | Done | 已实现独立三相 Corey-style relperm / mobility / fractional flow；未接入 transport。 |
| three-phase fractional flow | `reservoir_backend.solver.three_phase_relperm.fractional_flow_three_phase` | `tests/test_three_phase_relperm.py` | Done | 计算 `fw`, `fo`, `fg` 并验证 `fw + fo + fg = 1`。 |
| three-phase phase flux | `reservoir_backend.solver.three_phase_flux` | `tests/test_three_phase_flux.py` | Done | 已实现独立三相 advective phase flux，使用 upwind `fw/fo/fg` 分解 total Darcy face flux；未推进饱和度。 |
| three-phase 1D transport | `reservoir_backend.solver.three_phase_transport` | `tests/test_three_phase_transport_1d.py` | Done | 已实现独立 1D incompressible water-oil-gas 显式输运；未接入 CLI/YAML。 |
| three-phase 3D transport | `reservoir_backend.solver.three_phase_transport` | `tests/test_three_phase_transport_3d.py` | Done | 已实现独立 3D incompressible water-oil-gas 显式输运；未接入 CLI/YAML。 |
| three-phase pipeline | `config/three_phase_case.yaml`, `examples/run_full_pipeline_demo.py`, CLI runner | `tests/test_three_phase_pipeline.py` | Done | 已接入 YAML/CLI，运行简化 incompressible WOG 三相 advective transport，输出 Sw/Sg/So 和 three_phase_report；不是 black-oil。 |
| three-phase validation/profiling | `scripts/validate_three_phase_pipeline.py`, `scripts/profile_three_phase_pipeline.py` | `tests/test_three_phase_validation.py`, `tests/test_three_phase_profiling.py` | Done | 已验证 three_phase_case 输出完整、closure/bounds/NaN/CFL/material balance 合理，并记录 dt sensitivity 与 runtime profiling。 |
| semi-implicit capillary diffusion | Not implemented | Not applicable | Planned | 显式 combined transport 当前通过小模型验证；强毛管、细网格或 dt sensitivity 明显恶化时再启动 semi-implicit capillary diffusion。 |
| 参数场融合 | `reservoir_backend.fusion.field_fusion`, `field_mapper`, `confidence` | `tests/test_field_fusion.py`, `tests/test_field_mapper.py` | Done | Same-grid weighted fusion, confidence weighting, saturation clipping, point mapping, NaN reporting. |
| 多源信号融合 demo | `examples/run_multisignal_inversion_demo.py`, `examples/run_full_pipeline_demo.py` | `tests/test_multisignal_pipeline.py` | Done | Resistivity / electromagnetic / acoustic saturation inversions enter confidence-weighted field fusion. |
| Python 与前端 UDP 交互 | `reservoir_backend.api.udp_server` | Existing lightweight regression only | Deferred | 前端通讯协议未知；本阶段不继续开发 UDP API。 |
| 结果输出 | `reservoir_backend.io.result_manager`, `writer` | `tests/test_result_manager.py` | Done | Saves `.npy`, `.json`, `.csv`; validates required outputs. |
| full pipeline demo | `examples/run_full_pipeline_demo.py` | `tests/test_full_pipeline.py` | Done | Runs Archie -> pressure -> flux/velocity -> saturation -> fusion -> result export. |
| 后续 C++ 性能迁移 | `specs/09_cpp_migration_spec.md`, profiling harness | `tests/test_requirement_traceability.py`, `tests/test_combined_profiling.py` | Planned | 等待 validation 和 profiling 明确瓶颈后再启动局部 C++ kernel；combined profiling 已输出判断依据。 |
| 三相流 | `specs/12_three_phase_flow_design.md`, `reservoir_backend.solver.three_phase_transport`, `config/three_phase_case.yaml` | `tests/test_three_phase_design.py`, `tests/test_three_phase_transport_3d.py`, `tests/test_three_phase_pipeline.py` | Done | 设计、独立相渗、phase flux、1D/3D transport 和简化 YAML/CLI pipeline 已完成；仍不是 black-oil。 |
| 黑油模型 | Not implemented | Not applicable | Planned | 当前不实现黑油 PVT / 组分耦合；后续单独立项。 |
| cross-scale analysis design | `specs/13_cross_scale_analysis_design.md` | `tests/test_cross_scale_design.py` | Done | 已明确需求 1 和需求 2 不拆成两个独立软件，采用 one backend with two first-level modules；已设计 cross_scale 边界、similarity criteria、scale-effect、lab-field mapping、curve validation、report schema 和后续阶段。 |
| similarity criteria module | Not implemented | Not applicable | Planned | 待 `042_similarity_criteria_module` 实现 dimensionless numbers 和 similarity score。 |
| scale-effect analysis module | Not implemented | Not applicable | Planned | 待 `043_scale_effect_analysis_module` 实现 scale ratios、regime detection 和 warnings。 |
| lab-field validation module | Not implemented | Not applicable | Planned | 待 `044_lab_field_validation_module` 实现实验/矿场曲线对比和 mismatch metrics。 |
| UDP minimal API | Not implemented | Not applicable | Planned | 待 `045_udp_api_minimal`；当前 UDP 仍 deferred，前端协议未知。 |
| software requirement acceptance report | Not implemented | Not applicable | Planned | 待 `046_software_requirement_acceptance_report` 输出最终验收报告。 |

## Deferred / Planned Items

- UDP API is deferred because the frontend communication protocol is not known.
- Electromagnetic and acoustic inversion are Partial: lightweight empirical interfaces are implemented, complex physics remains out of scope.
- Capillary pressure models, capillary face fluxes, opt-in 1D/3D capillary transport integration, and YAML/CLI-controlled capillary pipeline execution are implemented. Default demo and multisignal cases remain capillary disabled.
- Nonuniform initial saturation validation is implemented with `capillary_gradient_case.yaml`; default `capillary_case.yaml` remains a uniform-initial-Sw smoke case and may produce zero capillary flux.
- Gravity face flux, optional vertical 1D / 3D gravity transport coupling, and YAML/CLI-controlled gravity pipeline execution are implemented. Combined capillary+gravity transport design, standalone flux composer, function-level 3D combined solver, YAML/CLI combined pipeline case, combined validation/profiling harnesses, and release-candidate documentation are implemented.
- Semi-implicit capillary diffusion is Planned/Future. It should start only if dt sensitivity or larger cases show explicit-step instability, severe CFL restriction, or unacceptable material-balance/runtime behavior.
- Three-phase flow design, independent three-phase relperm / fractional flow, three-phase phase flux, 1D/3D three-phase transport, simplified YAML/CLI three-phase pipeline, and three-phase validation/profiling are Done.
- Cross-scale analysis design is Done. Similarity criteria, scale-effect analysis, lab-field validation, UDP minimal API, and the final software requirement acceptance report are Planned. Cross-scale analysis remains inside one backend as a second first-level module, not a separate software product.
- Black-oil remains Planned; three-phase transport design is not equivalent to black-oil PVT behavior.
- C++ migration is planned only after Python validation and profiling show a concrete bottleneck.
