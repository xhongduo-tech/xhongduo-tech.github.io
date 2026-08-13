---
pageClass: plain-doc
---

# 华为 τ 架构

华为 τ 架构是面向大模型训练与推理的新一代 AI 处理器计算架构。学完这些章节，就写完了从芯片硬件设计、指令体系到软件栈生态的完整内容。

## 对标教材

- 华为，τ 架构技术报告（华为官方技术文档）
- 华为，AI 芯片生态文档（CANN / MindSpore 官方文档）

## 主题规划

<ProgressGrid cat="advanced/huawei-tau-architecture" />

### 第1篇

- [x] [τ 架构概览：产品定位与设计目标（τ 架构技术报告 第1章）](./tau-architecture-overview)
- [x] [从昇腾到 τ：架构演进与代际对比（华为 AI 芯片生态文档）](./ascend-to-tau-evolution)
- [x] [制程工艺与先进封装（τ 架构技术报告 第2章）](./process-and-advanced-packaging)
- [x] [片内系统总览：AI Core 与 SoC 组成（τ 架构技术报告 第2章）](./soc-overview-ai-core)
- [x] [内存系统：HBM 与多级缓存（τ 架构技术报告 第3章）](./memory-system-hbm-cache)

### 第2篇

- [x] [AI Core 微架构：Cube、Vector、Scalar 单元（τ 架构技术报告 第4章）](./ai-core-microarchitecture)
- [x] [矩阵乘与张量运算单元（τ 架构技术报告 第4章）](./matrix-multiply-tensor-unit)
- [x] [低精度计算：FP16 / BF16 / INT8 支持（τ 架构技术报告 第4章）](./low-precision-compute)
- [x] [指令集体系与指令流水线（τ 架构技术报告 第5章）](./instruction-set-and-pipeline)
- [x] [张量指令与数据流控制（τ 架构技术报告 第5章）](./tensor-instructions-dataflow)

### 第3篇

- [x] [片上网络（NoC）与带宽设计（τ 架构技术报告 第3章）](./noc-and-bandwidth)
- [x] [缓存一致性协议（华为 AI 芯片生态文档）](./cache-coherence)
- [x] [算子间数据复用与张量搬运（τ 架构技术报告 第6章）](./data-reuse-and-tensor-movement)
- [x] [多核并行与任务分配（华为 AI 芯片生态文档）](./multicore-parallelism)

### 第4篇

- [x] [CANN 计算架构与算子编程（华为 AI 芯片生态文档）](./cann-architecture-operator-programming)
- [x] [编译栈：图编译与算子融合（华为 AI 芯片生态文档）](./compilation-graph-fusion)
- [x] [MindSpore 与 τ 架构适配（华为 AI 芯片生态文档）](./mindspore-adaptation)
- [x] [大模型训练与推理优化实践（华为 AI 芯片生态文档）](./training-inference-optimization)
