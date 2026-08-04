---
pageClass: plain-doc
---

# AI Infrastructure

The complete technology stack of large-scale AI training and inference infrastructure: from GPU architecture and CUDA programming, to collective communication, parallelism strategies, training frameworks, memory optimization, training stability, cluster scheduling, data pipelines, inference architectures, performance profiling, domestic chip ecosystem, and cost engineering.

## Topic Planning

<ProgressGrid cat="advanced/ai-infra" />


### Part 1 GPU Architecture and CUDA Programming

- [ ] GPU vs CPU design philosophy: throughput-first vs latency-first
- [ ] Internal structure of streaming multiprocessors (SMs): CUDA Cores, Tensor Cores, registers and schedulers
- [ ] The SIMT execution model and how warps work
- [ ] Thread hierarchy: organization and indexing of Grid, Block, and Thread
- [ ] Warp scheduling and the cost of branch divergence
- [ ] GPU memory hierarchy: registers, shared memory, L1/L2 caches, global memory
- [ ] Global memory coalescing and bank conflicts
- [ ] Shared memory programming and `__syncthreads()` synchronization semantics
- [ ] Computing and tuning occupancy
- [ ] CUDA streams, asynchronous execution, and event timing
- [ ] Kernel launch overhead and the benefits of kernel fusion
- [ ] Matrix multiplication kernel optimization in practice: from naive to tiling
- [ ] Tensor Cores and WMMA/mma instruction programming
- [ ] The Roofline model: determining whether a kernel is compute-bound or memory-bound
- [ ] Understanding the IO-aware design of FlashAttention

### Part 2 Collective Communication

- [ ] Overview of collective communication primitives: Broadcast, Reduce, AllReduce, AllGather, ReduceScatter, AllToAll
- [ ] Algorithm derivation of Ring AllReduce and bandwidth optimality
- [ ] Tree AllReduce and Double Binary Tree: the latency-bandwidth trade-off
- [ ] NCCL architecture: topology detection, channels, and protocol selection
- [ ] NCCL tuning: environment variables, topology awareness, and common performance pitfalls
- [ ] Bandwidth hierarchy and topology of PCIe, NVLink, and NVSwitch
- [ ] RDMA fundamentals: kernel bypass, zero-copy, and queue pairs (QP)
- [ ] RoCE v2 and InfiniBand: lossless networks, PFC, and congestion control (DCQCN)
- [ ] Implementation mechanics of communication-computation overlap
- [ ] AllToAll communication patterns and optimization in MoE scenarios

### Part 3 Parallelism Strategies

- [ ] Data parallelism (DP) fundamentals: implementation of gradient synchronization and its overhead analysis
- [ ] Tensor parallelism (TP): deriving row-wise/column-wise matrix multiplication splitting
- [ ] Communication volume analysis of tensor parallelism and Megatron's 1D splitting scheme
- [ ] Pipeline parallelism (PP): micro-batches, GPipe scheduling, and bubble ratio
- [ ] 1F1B scheduling: a memory-balanced strategy for reducing pipeline bubbles
- [ ] Sequence parallelism (SP) and Context Parallel: splitting long-sequence training
- [ ] Expert parallelism (EP): MoE routing and load-balancing problems
- [ ] ZeRO-1/2/3: three-level partitioning of optimizer states, gradients, and parameters
- [ ] FSDP semantics: AllGather parameters + ReduceScatter gradients
- [ ] 3D hybrid parallelism (TP×PP×DP): splitting strategies and communication-group design
- [ ] Parallelism strategy selection in practice: how to configure given model size and cluster topology

### Part 4 Training Frameworks

- [ ] Megatron-LM overall architecture and the code structure of parallel splitting
- [ ] The relationship between Megatron-Core and Megatron-LM, and new features
- [ ] DeepSpeed architecture: ZeRO implementation, config files, and the training workflow
- [ ] PyTorch DDP internals: gradient bucketing and allreduce scheduling
- [ ] PyTorch FSDP2 (`fully_shard`) design: per-parameter sharding
- [ ] Framework comparison: Megatron-LM vs DeepSpeed vs FSDP2
- [ ] Mixed-precision training: FP16/BF16, loss scaling, and the master weights mechanism
- [ ] FP8 training: E4M3/E5M2 formats, per-tensor scaling, and precision compensation

### Part 5 Memory Optimization

- [ ] Breaking down training memory usage: parameters, gradients, optimizer states, and activations
- [ ] Activation checkpointing: selective recomputation vs full recomputation
- [ ] Quantitative analysis of the memory-compute trade-off in recomputation strategies
- [ ] ZeRO-Offload: offloading optimizer states and gradients to CPU/NVMe
- [ ] Unified memory and memory oversubscription
- [ ] Memory fragmentation and PyTorch's caching allocator mechanism
- [ ] Estimating the memory required to train a model: a hands-on exercise

### Part 6 Training Stability

- [ ] The loss spike phenomenon in large-scale training and its causes
- [ ] Responding to loss spikes: skipping batches, rolling back checkpoints, and adjusting the learning rate
- [ ] Numerical stability issues: overflow, underflow, and attention softmax precision
- [ ] Checkpointing for resuming training: what to save, how often, and consistency
- [ ] Engineering implementation of asynchronous and distributed checkpoints
- [ ] Fault-tolerant training: node failure detection, hang diagnosis, and automatic recovery
- [ ] Detecting and handling silent data corruption (SDC)

### Part 7 Cluster Scheduling and Resource Management

- [ ] Training cluster network topologies: Fat-Tree and rail-optimized designs
- [ ] Kubernetes' role in AI training: GPU scheduling and Volcano/gang scheduling
- [ ] Slurm job scheduling: partitions, priorities, and multi-node job submission
- [ ] Elastic training: dynamic scaling with TorchElastic
- [ ] Practices for training job queuing, preemption, and quota management
- [ ] Isolation and utilization optimization in multi-tenant clusters

### Part 8 Data Pipelines and Storage

- [ ] Overall architecture of the training data pipeline: storage, preprocessing, loading, and feeding GPUs
- [ ] Performance characteristics of object storage (S3/OSS) in training scenarios
- [ ] Data loading acceleration: WebDataset, FSSpec, and streaming reads
- [ ] DataLoader bottleneck diagnosis: num_workers, pin_memory, and prefetching
- [ ] Pushing preprocessing downstream: CPU preprocessing vs GPU preprocessing (DALI)
- [ ] Deduplication, cleaning, and tokenization pipelines for large-scale corpora
- [ ] Managing the storage cost of checkpoints and training data

### Part 9 Inference Infrastructure

- [ ] Inference vs training: latency-sensitive, no gradients, and the KV cache
- [ ] The two phases of inference: resource characteristics of prefill and decode
- [ ] Inference cluster architecture: the routing layer, inference instances, and scaling strategies
- [ ] Inference engine scheduling: continuous batching and the request queue
- [ ] GPU selection for inference: why memory bandwidth matters more than compute
- [ ] Prefill/decode disaggregation (PD separation) architecture

### Part 10 Monitoring and Performance Profiling

- [ ] Training task monitoring: GPU utilization, memory, network bandwidth, and power consumption
- [ ] PyTorch Profiler: operator timing analysis and trace interpretation
- [ ] Nsight Systems (nsys): timeline analysis and CPU-GPU co-diagnosis
- [ ] Nsight Compute (ncu): locating kernel-level bottlenecks
- [ ] Observing NCCL communication and verifying the effectiveness of communication-computation overlap
- [ ] Standard workflow for profiling distributed training: communication first or operators first

### Part 11 Domestic Chips and the Heterogeneous Ecosystem

- [ ] The landscape of domestic AI chips: the current state of training and inference product lines
- [ ] Huawei Ascend ecosystem: CANN, MindSpore, and torch_npu
- [ ] Cambricon MLU: an overview of the Cambricon Neuware software stack
- [ ] The moat of the CUDA ecosystem and analysis of the gap in domestic software stacks
- [ ] The path and pitfalls of porting CUDA programs to domestic chips
- [ ] Real-world constraints of heterogeneous co-location and cross-chip training

### Part 12 Cost Engineering

- [ ] Definition and calculation of MFU (Model FLOPs Utilization)
- [ ] A full picture of factors affecting MFU: parallelism strategies, communication, recomputation, and data loading
- [ ] Typical MFU levels in industry and practical means of improving MFU
- [ ] Training cost estimation: using Scaling Laws to work backward to how many GPU-hours a model needs
- [ ] A cost comparison framework for building your own cluster vs renting from the cloud
- [ ] Inference cost estimation: the cost breakdown per million tokens and optimization directions

> After writing: create `xxx.md` in this directory, then change the corresponding item above to `- [x] [Title](./xxx)`.
