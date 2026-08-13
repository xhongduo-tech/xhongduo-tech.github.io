---
pageClass: plain-doc
---

# AI 硬件：NVIDIA H100/Hopper

对标 NVIDIA H100 白皮书、Hopper 架构论文与 CUDA 官方文档，系统掌握 Hopper 一代 GPU 的硬件架构、编程模型与大规模 AI 集群部署实践。学完这些章节，就写完了 NVIDIA 新一代 AI 硬件平台从芯片到系统部署的核心内容。

## 对标教材

- NVIDIA, "NVIDIA H100 Tensor Core GPU Architecture"（H100 白皮书, 2022）
- J. Choquette et al., "NVIDIA Hopper H100 GPU: Scaling Performance"（IEEE Micro, 2023）
- NVIDIA, "CUDA C++ Programming Guide"（NVIDIA 官方文档）

## 主题规划

<ProgressGrid cat="advanced/nvidia-h100-hopper" />

### 第1篇

- [x] [H100 概览：产品定位与 A100 对比](./h100-overview-a100-comparison)
- [x] [Hopper 架构设计理念与关键技术清单](./hopper-architecture-design-philosophy)
- [x] [TSMC 4N 制程与 2.5D 封装](./tsmc-4n-25d-packaging)
- [x] [HBM3 高带宽内存与缓存层级](./hbm3-memory-hierarchy)
- [x] [NVLink 4 与 NVSwitch 互联](./nvlink4-nvswitch)
- [x] [DGX H100 与数据中心系统](./dgx-h100-datacenter-system)

### 第2篇

- [x] [流式多处理器（SM）微架构](./sm-microarchitecture)
- [x] [第四代 Tensor Core 与 FP8 精度](./tensor-core-fp8)
- [x] [Transformer Engine：动态精度选择](./transformer-engine)
- [x] [Tensor Memory Accelerator（TMA）](./tensor-memory-accelerator)
- [x] [线程块簇与分布式共享内存](./thread-block-clusters-dsmem)
- [x] [Warpgroup 异步执行与流水线](./warpgroup-async-execution)

### 第3篇

- [x] [CUDA 线程层次与执行模型](./cuda-thread-hierarchy)
- [x] [Hopper 内存模型与一致性](./hopper-memory-model)
- [x] [PTX 指令集与 Hopper 新指令](./ptx-hopper-instructions)
- [x] [CUDA Graphs 与程序化依赖](./cuda-graphs)
- [x] [cuBLAS/cuDNN 性能库](./cublas-cudnn-libraries)
- [x] [统一内存与页迁移](./unified-memory-page-migration)

### 第4篇

- [x] [Roofline 性能分析与 Profiling](./roofline-profiling)
- [x] [大规模 LLM 训练与 H100 实践](./llm-training-h100)
- [x] [推理优化：KV Cache 与 FlashAttention](./inference-optimization-kv-cache)
- [x] [多节点集群：Scale-up 与 Scale-out](./scale-up-scale-out)
- [x] [能效、TCO 与数据中心部署](./power-efficiency-tco)
