# 路线图与限制

本文记录当前限制、近期维护优先级与未来范围。读完本文你将清楚本后端能做什么、不能做什么，以及文档维护政策。

**下一步阅读：** [../STATUS.md](../STATUS.md)（模块级证据）、[black_oil_pvt_architecture.md](black_oil_pvt_architecture.md)（黑油架构设计，仅文档）。

## 当前限制

当前后端面向结构化网格 Python 工作流与小型验证案例。已知限制包括：

- 结构化笛卡尔网格是主要数值目标；
- 显式输运仍是主要输运路径；
- 简化三相 WOG 工具为不可压缩模型，**未实现黑油 PVT**；
- 跨尺度模块提供报告与诊断，**非多尺度有限体积求解器**；
- 融合工具**不执行历史拟合、自动校准或集合同化**；
- 结果与项目管理基于文件，**无数据库服务**；
- 当前后端**不包含**前端、UDP 服务、REST API 或 C++ 加速层。

## 近期维护优先级

- 保持文档入口精简且一致；
- 保持 `STATUS.md` 为唯一维护的状态表；
- 加强数据管线对真实实验数据集的覆盖；
- 仅在现有结构化网格范围内扩展压力与输运回归案例；
- 当下游消费者需要稳定字段时改进报告模式；
- 保持基准可复现，不增加对外部模拟器的运行时依赖。

## 未来范围（明确排除）

以下领域超出当前 MVP：

- 黑油与组分 PVT；
- 广泛的工业井控与井筒网络；
- 全隐式储层模拟；
- 复杂角点点地质模型；
- 完整 SPE 复现声明；
- 与 OPM Flow 或 MRST 的等价声明；
- 生产前端、数据库服务、UDP 服务或 REST API；
- 无实测瓶颈时的 C++ 或 pybind11 内核迁移；
- 历史拟合、自动校准、EnKF、ES-MDA 或贝叶斯反演工作流。

黑油模拟器相关能力仅存在于架构设计文档中，**非已实现功能**。UDP 开发仍处于延后状态（udp development is still deferred）。

## 文档政策

- 历史矩阵与检查清单已移至 `docs/archive/doc_consolidation/`，保留用于追溯，**非活跃状态来源**。
- 新文档应链接本路线图与 `STATUS.md`，而非创建另一份完成度矩阵。
- 架构与数据流见 [ARCHITECTURE.md](ARCHITECTURE.md)；接口契约见 [API_AND_DATA_CONTRACT.md](API_AND_DATA_CONTRACT.md)。
- 完整文档索引见 [README.md](README.md)。
- **文档与代码一致性**：活跃文档中的路径声明应与仓库树一致；本地用 [QMD](https://www.npmjs.com/package/@tobilu/qmd) 索引 `docs/`、`specs/`、`STATUS.md` 等做语义检索，并用 `python scripts/check_doc_code_consistency.py` 做路径级硬检查。
