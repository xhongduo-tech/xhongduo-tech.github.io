---
pageClass: plain-doc
---

# MTP 多 Token 预测

对标 DeepSeek-V2/V3 技术报告与 Gloeckle 等人的多 Token 预测论文，系统掌握让语言模型同时预测多个未来 Token 的训练范式、架构设计与推理加速技术。学完这些章节，就写完了多 Token 预测从动机、训练到投机解码应用的核心内容。

## 对标教材

- Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction"（Meta AI, 2024）
- DeepSeek-AI, "DeepSeek-V3 Technical Report"（2024, §MTP）
- DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"（2024）

## 主题规划

<ProgressGrid cat="advanced/llm-mtp-multi-token-prediction" />

### 第1篇

- [x] [单 Token 预测范式的效率局限（Gloeckle 2024 §1）](./single-token-paradigm-limits)
- [x] [多 Token 预测：定义与核心思想（Gloeckle 2024 §1）](./multi-token-prediction-definition)
- [x] [MTP 训练目标与损失函数设计（DeepSeek-V3 技术报告 §MTP）](./mtp-training-objective-loss)
- [x] [共享主干 + 多预测头的整体架构（Gloeckle 2024 §2）](./shared-trunk-multi-heads)
- [x] [MTP 与单 Token 预测的本质差异（Gloeckle 2024 §1）](./mtp-vs-single-token)
- [x] [从 DeepSeek-V2 到 V3 的 MTP 演进（DeepSeek-V2/V3 技术报告）](./deepseek-v2-to-v3-mtp-evolution)

### 第2篇

- [x] [并行 MTP 模块：注意力与因果掩码（DeepSeek-V3 技术报告 §MTP）](./parallel-mtp-attention-causal-mask)
- [x] [MTP 模块的逐层深度集成（DeepSeek-V3 技术报告 §MTP）](./mtp-layer-deep-integration)
- [x] [预测头参数与梯度流设计（DeepSeek-V3 技术报告 §MTP）](./mtp-head-parameters-gradient-flow)
- [x] [MTP 训练数据与样本构造（Gloeckle 2024 §3）](./mtp-training-data-samples)
- [x] [训练稳定性与损失权重调度（Gloeckle 2024 §4）](./mtp-training-stability-loss-weighting)
- [x] [模型容量与训练时长的权衡（Gloeckle 2024 §5）](./mtp-capacity-training-compute-tradeoff)

### 第3篇

- [x] [MTP 驱动的投机解码机制（DeepSeek-V3 技术报告 §MTP）](./mtp-driven-speculative-decoding)
- [x] [推理时复用 MTP 头的加速技巧（DeepSeek-V3 技术报告 §MTP）](./mtp-head-inference-acceleration)
- [x] [接收率提升与延迟收益分析（DeepSeek-V3 技术报告）](./mtp-acceptance-rate-latency-gains)
- [x] [与标准投机采样的对比（DeepSeek-V3 技术报告 §2.4）](./mtp-vs-standard-speculative-sampling)
- [x] [采样效率与数据利用率的提升（Gloeckle 2024 §3）](./mtp-sampling-efficiency-data-utilization)
- [x] [长上下文与多轮场景适配（DeepSeek-V3 技术报告）](./mtp-long-context-multiturn-adaptation)

### 第4篇

- [x] [多 Token 预测的因果估计（Causal Estimation 论文）](./mtp-causal-estimation)
- [x] [为什么有效：表征学习与隐式正则化视角（Gloeckle 2024 §5）](./mtp-why-it-works-representation-regularization)
- [x] [MTP 在推理与代码生成任务上的增益（Gloeckle 2024 §5）](./mtp-reasoning-code-gains)
- [x] [超越 Next-token：更长视野预测的前沿（Gloeckle 2024 §6）](./beyond-next-token-long-horizon-frontier)
