---
pageClass: plain-doc
---

# 华为 CANN 计算架构

对标华为昇腾 CANN 官方开发指南，系统掌握从昇腾 AI 处理器达芬奇架构、软件栈分层、AscendCL 编程接口、图引擎与算子开发到 HCCL 集合通信与性能调优的完整计算栈。学完这些章节，就写完了华为昇腾 AI 计算平台从芯片到应用开发的核心内容。

## 对标教材

- 华为昇腾 CANN 开发指南（华为 CANN 官方文档）
- 昇腾 AI 处理器架构与软件栈白皮书（华为昇腾官方文档）
- AscendCL / GE / HCCL 编程开发指南（华为 CANN 官方文档）

## 主题规划

<ProgressGrid cat="advanced/huawei-cann" />

### 第1篇

- [x] [昇腾 AI 处理器与达芬奇架构概述](./ascend-ai-processor-davinci-overview)（昇腾白皮书）
- [x] [AI Core 计算单元：Cube、Vector 与 Scalar](./ai-core-compute-units-cube-vector-scalar)（昇腾白皮书）
- [x] [昇腾芯片存储层级与片上内存](./ascend-memory-hierarchy-on-chip)（昇腾白皮书）
- [x] [CANN 软件栈总体架构与分层设计](./cann-software-stack-architecture)（CANN 开发指南）
- [x] [CANN 安装部署与开发环境搭建](./cann-installation-environment-setup)（CANN 开发指南 安装章节）
- [x] [MindSpore 与 CANN 的关系及运行框架](./mindspore-cann-relationship)（CANN 开发指南）

### 第2篇

- [x] [AscendCL 编程模型与开发流程](./ascendcl-programming-model)（CANN 开发指南 AscendCL 章节）
- [x] [运行管理：设备、上下文与流](./ascendcl-device-context-stream)（CANN 开发指南）
- [x] [内存管理：申请、释放与缓存策略](./ascendcl-memory-management)（CANN 开发指南）
- [x] [模型加载与推理执行](./ascendcl-model-loading-inference)（CANN 开发指南）
- [x] [AIPP 图像预处理与数据搬移](./aipp-image-preprocessing)（CANN 开发指南）
- [x] [同步/异步执行与事件同步机制](./ascendcl-sync-async-event)（CANN 开发指南）

### 第3篇

- [x] [GE 图引擎：构图与图优化](./ge-graph-engine-computation-optimization)（CANN 图引擎文档）
- [x] [ATC 模型转换工具与离线模型](./atc-model-conversion-tool)（CANN 开发指南 ATC 章节）
- [x] [TBE 算子开发框架与 DSL 编程](./tbe-operator-development-dsl)（CANN 算子开发文档）
- [x] [算子原型定义、注册与自动生成](./tbe-operator-proto-registration)（CANN 算子开发文档）
- [x] [自定义算子开发与调试验证](./custom-operator-development-debug)（CANN 算子开发文档）
- [x] [算子融合与计算图编译优化](./operator-fusion-graph-optimization)（CANN 图引擎文档）

### 第4篇

- [x] [HCCL 集合通信库与通信原语](./hccl-collective-communication)（CANN HCCL 文档）
- [x] [多卡训练拓扑与集群通信配置](./multi-card-training-topology)（CANN HCCL 文档）
- [x] [性能剖析与 Profiling 工具](./ascend-profiling-msprof)（Ascend msprof 文档）
- [x] [内存复用与算子执行性能优化](./memory-reuse-performance-optimization)（CANN 开发指南）
- [x] [大模型训练在昇腾平台的工程实践](./llm-training-ascend-practice)（CANN 官方文档）
