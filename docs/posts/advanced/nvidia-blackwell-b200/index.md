---
pageClass: plain-doc
---

# AI 硬件：NVIDIA Blackwell/B200

以 NVIDIA Blackwell 平台白皮书与 B200 官方文档为纲，系统拆解从芯片微架构、张量核心、Transformer 引擎到 NVLink 互联、整柜液冷系统与配套软件栈的完整 AI 计算硬件栈。学完这些章节，就掌握了 Blackwell/B200 从晶体管到集群的完整知识地图。

## 对标教材

- NVIDIA Blackwell Platform Whitepaper（2024）
- NVIDIA B200 GPU 官方技术文档 / Datasheet
- NVIDIA GB200 NVL72 系统与 DGX 平台文档

## 主题规划

<ProgressGrid cat="advanced/nvidia-blackwell-b200" />

### 第1篇

- [x] [平台动机与新计算范式](./platform-motivation-paradigm)
- [x] [Blackwell 架构总览与关键创新](./architecture-overview)
- [x] [GB200 Grace Blackwell 超级芯片](./gb200-grace-blackwell-superchip)
- [x] [DGX GB200 NVL72 整柜系统](./dgx-gb200-nvl72-rack)
- [x] [液冷与机柜级供电设计](./liquid-cooling-power)
- [x] [B200 规格与 HBM3e 显存子系统](./b200-specs-hbm3e)

### 第2篇

- [x] [台积电工艺与裸片设计](./tsmc-process-dual-die)
- [x] [新一代张量核心与稀疏计算](./tensor-core-sparsity)
- [x] [第二代 Transformer 引擎](./transformer-engine-v2)
- [x] [FP4/FP8 混合精度与量化推理](./fp4-fp8-mixed-precision)
- [x] [CUDA 核心与通用计算能力](./cuda-core-general-compute)
- [x] [能效与每瓦性能指标](./energy-efficiency-perf-watt)

### 第3篇

- [x] [第五代 NVLink 互联](./nvlink-5)
- [x] [NVLink 域与 NVSwitch 拓扑](./nvlink-domain-nvswitch)
- [x] [网络 Quantum-X800 InfiniBand](./quantum-x800-infiniband)
- [x] [网络 Spectrum-X 以太网](./spectrum-x-ethernet)
- [x] [集群级扩展与多节点训练](./cluster-scale-multinode-training)
- [x] [MGX 模块化参考架构](./mgx-modular-reference)

### 第4篇

- [x] [保密计算与安全 AI](./confidential-computing-security)
- [x] [CUDA 12.8 与 Blackwell 软件栈](./cuda-12-8-software-stack)
- [x] [大规模 LLM 推理与部署](./llm-inference-deployment)
- [x] [训练与推理基准性能](./training-inference-benchmarks)
- [x] [Hopper H100 与 Blackwell B200 演进对比](./h100-vs-b200-evolution)
