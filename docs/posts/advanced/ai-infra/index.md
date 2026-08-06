---
pageClass: plain-doc
---

# AI 基础设施

大规模 AI 训练与推理基础设施的完整技术栈：从 GPU 体系结构与 CUDA 编程，到集合通信、并行策略、训练框架、显存优化、训练稳定性、集群调度、数据管线、推理架构、性能剖析、国产芯片生态与成本工程。

## 主题规划

<ProgressGrid cat="advanced/ai-infra" />


### 第一篇 GPU 体系结构与 CUDA 编程

- [x] [GPU 与 CPU 的设计哲学差异：吞吐优先 vs 延迟优先](./gpu-vs-cpu)
- [x] [流式多处理器（SM）的内部结构：CUDA Core、Tensor Core、寄存器与调度器](./sm-internal-structure)
- [x] [SIMT 执行模型与 warp（线程束）的工作原理](./simt-warp)
- [x] [线程层次：Grid、Block、Thread 的组织与索引](./thread-hierarchy)
- [x] [warp 调度与分支分化（divergence）的代价](./warp-scheduling-divergence)
- [x] [GPU 内存层次：寄存器、Shared Memory、L1/L2 缓存、全局内存](./gpu-memory-hierarchy)
- [x] [全局内存合并访存（coalescing）与 bank conflict](./memory-coalescing-bank-conflict)
- [x] [Shared Memory 编程与 `__syncthreads()` 同步语义](./shared-memory-syncthreads)
- [x] [Occupancy（占用率）的计算与调优](./occupancy-tuning)
- [x] [CUDA Stream 与异步执行、事件计时](./cuda-stream-async-events)
- [x] [kernel 启动开销与 kernel fusion（算子融合）的收益](./kernel-launch-overhead-fusion)
- [x] [矩阵乘法 kernel 优化实战：从 naive 到 tiling](./matmul-kernel-optimization)
- [x] [Tensor Core 与 WMMA/mma 指令编程](./tensor-core-wmma)
- [x] [Roofline 模型：判断 kernel 是计算瓶颈还是访存瓶颈](./roofline-model)
- [x] [FlashAttention 的 IO 感知设计思想解析](./flashattention-io-aware)

### 第二篇 集合通信

- [x] [集合通信原语总览：Broadcast、Reduce、AllReduce、AllGather、ReduceScatter、AllToAll](./collective-communication-primitives)
- [x] [Ring AllReduce 的算法推导与带宽最优性](./ring-allreduce)
- [x] [Tree AllReduce 与 Double Binary Tree：延迟与带宽的权衡](./tree-allreduce-double-binary-tree)
- [x] [NCCL 架构：拓扑检测、通道（Channel）与协议选择](./nccl-architecture)
- [x] [NCCL 调优：环境变量、拓扑感知与常见性能陷阱](./nccl-tuning)
- [x] [PCIe、NVLink、NVSwitch 的带宽层级与拓扑](./pcie-nvlink-nvswitch)
- [x] [RDMA 原理：内核旁路、零拷贝与队列对（QP）](./rdma-principle)
- [x] [RoCE v2 与 InfiniBand：无损网络、PFC 与拥塞控制（DCQCN）](./roce-v2-infiniband)
- [x] [通信计算重叠（overlap）的实现机制](./comm-compute-overlap)
- [x] [AllToAll 在 MoE 场景下的通信模式与优化](./alltoall-moe)

### 第三篇 并行策略

- [ ] 数据并行（DP）原理：梯度同步的实现与开销分析
- [ ] 张量并行（TP）：按行/按列切分矩阵乘的推导
- [ ] 张量并行的通信量分析与 Megatron 的 1D 切分方案
- [ ] 流水线并行（PP）：micro-batch、GPipe 调度与气泡率
- [ ] 1F1B 调度：减少流水线气泡的显存均衡策略
- [ ] 序列并行（SP）与 Context Parallel：长序列训练的切分
- [ ] 专家并行（EP）：MoE 路由与负载均衡问题
- [ ] ZeRO-1/2/3：优化器状态、梯度、参数的三级切分
- [ ] FSDP 的语义：AllGather 参数 + ReduceScatter 梯度
- [ ] 3D 混合并行（TP×PP×DP）的切分策略与通信组设计
- [ ] 并行策略选型实战：给定模型规模与集群拓扑如何配比

### 第四篇 训练框架

- [ ] Megatron-LM 整体架构与并行切分的代码结构
- [ ] Megatron-Core 与 Megatron-LM 的关系及新特性
- [ ] DeepSpeed 架构：ZeRO 实现、配置文件与训练流程
- [ ] PyTorch DDP 内部机制：bucket 分桶与梯度 allreduce 调度
- [ ] PyTorch FSDP2（fully_shard）的设计：per-parameter sharding
- [ ] 框架选型对比：Megatron-LM vs DeepSpeed vs FSDP2
- [ ] 混合精度训练：FP16/BF16、loss scaling 与主权重机制
- [ ] FP8 训练：E4M3/E5M2 格式、逐张量缩放与精度补偿

### 第五篇 显存优化

- [ ] 训练显存的构成拆解：参数、梯度、优化器状态、激活值
- [ ] 激活重计算（activation checkpointing）：选择性重算与全量重算
- [ ] 重计算策略的显存-算力权衡定量分析
- [ ] ZeRO-Offload：优化器状态与梯度卸载到 CPU/NVMe
- [ ] 统一内存（Unified Memory）与显存超订（oversubscription）
- [ ] 显存碎片与 PyTorch 的 caching allocator 机制
- [ ] 估算训练一个模型的显存需求：实战演练

### 第六篇 训练稳定性

- [ ] 大规模训练中的损失尖刺（loss spike）现象与成因
- [ ] 损失尖刺的应对：跳过 batch、回滚检查点与学习率调整
- [ ] 数值稳定性问题：溢出、下溢与 attention softmax 的精度
- [ ] 断点续训（checkpointing）：保存什么、保存频率与一致性
- [ ] 异步与分布式 checkpoint 的工程实现
- [ ] 容错训练：节点故障检测、挂起诊断与自动恢复
- [ ] 静默数据损坏（SDC）的检测与应对

### 第七篇 集群调度与资源管理

- [ ] 训练集群的网络拓扑：胖树（Fat-Tree）、轨式（Rail-optimized）设计
- [ ] Kubernetes 在 AI 训练中的角色：GPU 调度与 Volcano/ gang scheduling
- [ ] Slurm 作业调度：分区、优先级与多节点任务提交
- [ ] 弹性训练（elastic training）：TorchElastic 的动态扩缩容
- [ ] 训练任务的排队、抢占与配额管理实践
- [ ] 多租户集群的隔离与利用率优化

### 第八篇 数据管线与存储

- [ ] 训练数据管线的整体架构：存储、预处理、加载、喂卡
- [ ] 对象存储（S3/OSS）在训练场景下的性能特征
- [ ] 数据加载加速：WebDataset、FSSpec 与流式读取
- [ ] DataLoader 瓶颈诊断：num_workers、pin_memory 与预取
- [ ] 数据预处理下沉：CPU 预处理 vs GPU 预处理（DALI）
- [ ] 大规模语料的去重、清洗与 tokenize 流水线
- [ ] 检查点与训练数据的存储成本管理

### 第九篇 推理基础设施

- [ ] 推理与训练的差异：延迟敏感、无梯度、KV Cache
- [ ] 推理的两个阶段：prefill 与 decode 的资源特征
- [ ] 推理集群架构：路由层、推理实例与扩缩容策略
- [ ] 推理引擎的调度：continuous batching 与请求队列
- [ ] 推理的 GPU 选型：显存带宽为何比算力更重要
- [ ] PD 分离（prefill/decode disaggregation）架构

### 第十篇 监控与性能剖析

- [ ] 训练任务监控体系：GPU 利用率、显存、网络带宽与功耗
- [ ] PyTorch Profiler：算子耗时分析与 trace 解读
- [ ] Nsight Systems（nsys）：时间线分析与 CPU-GPU 协同诊断
- [ ] Nsight Compute（ncu）：kernel 级瓶颈定位
- [ ] NCCL 通信的观测与通信计算重叠的效果验证
- [ ] 分布式训练性能剖析的标准流程：先通信还是先算子

### 第十一篇 国产芯片与异构生态

- [ ] 国产 AI 芯片格局：训练与推理两条产品线的现状
- [ ] 华为 Ascend 生态：CANN、昇思 MindSpore 与 torch_npu
- [ ] 寒武纪 MLU：Cambricon Neuware 软件栈概览
- [ ] CUDA 生态的护城河与国产软件栈的差距分析
- [ ] CUDA 程序向国产芯片迁移的路径与坑点
- [ ] 异构混部与跨芯片训练的现实约束

### 第十二篇 成本工程

- [ ] MFU（Model FLOPs Utilization）的定义与计算方法
- [ ] 影响 MFU 的因素全景：并行策略、通信、重计算与数据加载
- [ ] 业界典型 MFU 水平与提升 MFU 的实战手段
- [ ] 训练成本核算：用 Scaling Law 反推一个模型需要多少 GPU·小时
- [ ] 自建集群 vs 云上租用的成本对比框架
- [ ] 推理成本核算：每百万 token 的成本构成与优化方向

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
