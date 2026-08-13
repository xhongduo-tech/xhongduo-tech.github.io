---
pageClass: plain-doc
---

# ktransformers（消费级 MoE 推理引擎）

ktransformers 是清华大学 MADSys 实验室与趋境科技开源的 CPU-GPU 异构 MoE 推理框架，通过把稀疏专家卸载到 CPU/DRAM、让 GPU 专注高算术强度算子的方式，使 DeepSeek-V3 等 671B 级超大 MoE 模型在单张 24GB 消费级显卡上即可满血推理。它是理解「算力普惠」时代消费级大模型推理与微调全链路的核心主题。

## 对标教材

- Hongtao Chen, Weiyu Xie et al., "KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models"（SOSP 2025）
- KVCache.AI, ktransformers 官方仓库与文档（kvcache-ai/ktransformers）
- DeepSeek-AI, "DeepSeek-V3 Technical Report"（2024，MLA 与 MoE 背景）

## 主题规划

<ProgressGrid cat="advanced/ktransformers-engine" />

### 第1篇 MoE 稀疏性与异构推理动机

- [x] [MoE 稀疏专家架构与激活原理](./moe-sparse-experts-activation)
- [x] [专家激活幂律分布：热专家与冷专家](./expert-activation-power-law)
- [x] [算术强度与 Roofline 模型](./arithmetic-intensity-roofline)
- [x] [消费级硬件运行超大 MoE 的显存与带宽挑战](./consumer-hardware-memory-challenges)
- [x] [全量卸载方案（llama.cpp/DeepSpeed）的瓶颈分析](./full-offload-bottlenecks)
- [x] [异构推理的机会窗口与设计目标](./hetero-inference-opportunity)

### 第2篇 核心架构：CPU-GPU 异构专家调度

- [x] [总体架构：GPU 算稠密、CPU 算稀疏的异构协同](./hybrid-architecture-overview)
- [x] [算术强度感知的算子放置与推理内核调度](./arithmetic-intensity-aware-scheduling)
- [x] [频率感知专家放置策略与 GPU Expert Mask](./frequency-aware-expert-placement)
- [x] [num_gpu_experts 参数与动态专家更新](./num-gpu-experts-dynamic)
- [x] [异步 CPU-GPU 任务调度与 CUDA Graph 流水线](./async-scheduling-cuda-graph)
- [x] [NUMA 感知内存分配与 Pinned Memory 缓冲](./numa-pinned-memory)

### 第3篇 内核优化与量化

- [x] [GPU 端 Marlin 量化矩阵内核](./gpu-marlin-kernel)
- [x] [CPU 端 llamafile 内核与多线程任务调度](./cpu-llamafile-kernel)
- [x] [Intel AMX-BF16 与 AVX512-BF16/AVX-VNNI 指令集](./amx-avx-instructions)
- [x] [预填充阶段的块量化与内存布局优化](./prefill-block-quantization)
- [x] [原生 BF16 与 FP8 per-channel 精度支持](./native-bf16-fp8-precision)
- [x] [INT4/INT8 量化权重在 CPU 端的推理](./int4-int8-cpu-inference)

### 第4篇 KV Cache 与上下文扩展

- [x] [三层 GPU-CPU-Disk 前缀缓存架构](./three-level-prefix-cache)
- [x] [KV Cache 复用与 139K 上下文扩展](./kv-cache-reuse-139k)
- [x] [与 SGLang 集成：Continuous Batching 与多并发](./sglang-integration-batching)
- [x] [balance-serve 多并发推理服务机制](./balance-serve-concurrency)
- [x] [权重注入与模型适配（injection 教程）](./weight-injection-tutorial)

### 第5篇 微调与部署实践

- [x] [环境搭建：install.sh 与 pip 安装流程](./environment-setup)
- [x] [消费级部署案例：24GB 单卡运行 DeepSeek-V3/R1](./deepseek-24gb-deployment)
- [x] [LLaMA-Factory 集成与 CPU/GPU 异构 LoRA 微调](./llama-factory-lora-finetune)
- [x] [超大 MoE 模型在消费级硬件上的 SFT（FSDP2）](./sft-fsdp2-consumer)
- [x] [RL-DPO 微调实践](./rl-dpo-finetune)
- [x] [多硬件后端：ROCm、Intel Arc、昇腾 NPU](./multi-backend-hardware)
