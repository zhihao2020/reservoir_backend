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
| 参数场融合 | `reservoir_backend.fusion.field_fusion`, `field_mapper`, `confidence` | `tests/test_field_fusion.py`, `tests/test_field_mapper.py` | Done | Same-grid weighted fusion, confidence weighting, saturation clipping, point mapping, NaN reporting. |
| 多源信号融合 demo | `examples/run_multisignal_inversion_demo.py`, `examples/run_full_pipeline_demo.py` | `tests/test_multisignal_pipeline.py` | Done | Resistivity / electromagnetic / acoustic saturation inversions enter confidence-weighted field fusion. |
| Python 与前端 UDP 交互 | `reservoir_backend.api.udp_server` | Existing lightweight regression only | Deferred | 前端通讯协议未知；本阶段不继续开发 UDP API。 |
| 结果输出 | `reservoir_backend.io.result_manager`, `writer` | `tests/test_result_manager.py` | Done | Saves `.npy`, `.json`, `.csv`; validates required outputs. |
| full pipeline demo | `examples/run_full_pipeline_demo.py` | `tests/test_full_pipeline.py` | Done | Runs Archie -> pressure -> flux/velocity -> saturation -> fusion -> result export. |
| 后续 C++ 性能迁移 | `specs/09_cpp_migration_spec.md`, profiling harness | `tests/test_requirement_traceability.py` | Planned | 等待 validation 和 profiling 明确瓶颈后再启动局部 C++ kernel。 |
| 三相流 | Not implemented | Not applicable | Planned | 当前模型仍限定油水两相；三相流留作后续需求。 |
| 黑油模型 | Not implemented | Not applicable | Planned | 当前不实现黑油 PVT / 组分耦合；后续单独立项。 |

## Deferred / Planned Items

- UDP API is deferred because the frontend communication protocol is not known.
- Electromagnetic and acoustic inversion are Partial: lightweight empirical interfaces are implemented, complex physics remains out of scope.
- Capillary pressure models, capillary face fluxes, opt-in 1D/3D capillary transport integration, and YAML/CLI-controlled capillary pipeline execution are implemented. Default demo and multisignal cases remain capillary disabled.
- Nonuniform initial saturation validation is implemented with `capillary_gradient_case.yaml`; default `capillary_case.yaml` remains a uniform-initial-Sw smoke case and may produce zero capillary flux.
- Gravity face flux, optional vertical 1D / 3D gravity transport coupling, and YAML/CLI-controlled gravity pipeline execution are implemented. Combined capillary+gravity transport design, standalone flux composer, function-level 3D combined solver, and YAML/CLI combined pipeline case are implemented.
- Three-phase flow and black-oil models are Planned and not implemented in the current prototype.
- C++ migration is planned only after Python validation and profiling show a concrete bottleneck.
