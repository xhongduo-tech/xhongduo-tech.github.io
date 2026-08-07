---
pageClass: plain-doc
---

# 计算机体系结构

以量化研究方法（Quantitative Approach）系统进阶计算机体系结构，对标 Hennessy & Patterson《Computer Architecture: A Quantitative Approach》章节体系，覆盖从指令级并行到仓库级计算机的完整内容。本科「计算机组成原理」基础见 [/posts/cs/computer-organization/](/posts/cs/computer-organization/)，本分类聚焦进阶量化视角。

## 主题规划

<ProgressGrid cat="cs/computer-architecture" />


### 第一篇 体系结构基础与量化分析

- [x] [计算机体系结构的定义：指令集、组成与实现的分工](./architecture-definition)
- [x] [计算机的分类与 Flynn 分类法](./computer-classification-flynn)
- [x] [量化分析的基本原则：以大概率事件为快](./quantitative-principles)
- [x] [性能度量：CPU 时间、CPI 与 MIPS 的陷阱](./performance-metrics-cpu-cpi)
- [x] [基准测试程序：SPEC、TPC 与基准测试的选择](./benchmarks-spec-tpc)
- [x] [Amdahl 定律：加速比的量化推导与应用](./amdahls-law)
- [x] [局部性原理与常见场景的加速比估算](./locality-principle)
- [x] [功耗墙（Power Wall）：动态功耗与静态功耗](./power-wall)
- [x] [Dennard 缩放的终结与「暗硅」（Dark Silicon）问题](./dark-silicon)
- [x] [从单核到多核：功耗约束下的设计转向](./single-core-to-multicore)
- [x] [成本、可靠性与可用性的量化度量](./cost-reliability-availability-metrics)
- [x] [体系结构的谬误与易犯的错误（Pitfalls and Fallacies）](./architecture-pitfalls-fallacies)

### 第二篇 指令级并行

- [x] [指令级并行（ILP）的概念与限制因素](./ilp-concepts-limits)
- [x] [数据相关、名相关与控制相关的辨析](./data-name-control-dependencies)
- [x] [动态调度基础：记分板（Scoreboard）算法](./scoreboard-dynamic-scheduling)
- [x] [Tomasulo 算法：保留站与公共数据总线（CDB）](./tomasulo-algorithm)
- [x] [寄存器重命名：消除 WAR 与 WAW 相关](./register-renaming)
- [x] [硬件推测（Speculation）：重排序缓冲区（ROB）](./hardware-speculation-rob)
- [x] [精确异常与推测的恢复机制](./precise-exceptions-speculation-recovery)
- [x] [分支预测：1 位/2 位预测器与相关分支预测器](./branch-prediction-1bit-2bit)
- [x] [锦标赛预测器（Tournament Predictor）与 TAGE](./tournament-tage-predictors)
- [x] [分支目标缓冲区（BTB）与返回地址栈](./branch-target-buffer-ras)
- [x] [多发射处理器：超标量（Superscalar）与超流水线](./superscalar-superpipelined)
- [x] [静态多发射：VLIW 与显式并行指令计算（EPIC）](./vliw-epic)
- [x] [动态调度与多发射的结合：现代超标量流水线](./modern-superscalar-pipeline)
- [x] [推测的限制与 ILP 的天花板研究](./speculation-limits-ilp-ceiling)
- [x] [实例剖析：Intel Core 与 ARM Cortex 的微架构](./intel-core-arm-cortex-microrch)

### 第三篇 指令集架构进阶

- [x] [指令集架构的分类：栈、累加器与寄存器型](./isa-classification-stack-accumulator-register)
- [x] [RISC 设计哲学：精简指令集的历史演进](./risc-design-philosophy)
- [x] [RISC-V 指令集概览：模块化与可扩展设计](./riscv-overview)
- [x] [RISC-V 整数指令集：RV32I/RV64I 的编码与语义](./riscv-integer-instruction-set)
- [x] [RISC-V 特权架构与异常处理机制](./riscv-privileged-architecture)
- [x] [RISC-V 压缩指令扩展（C 扩展）](./riscv-compressed-instructions)
- [x] [向量指令体系结构：向量长度无关编程模型](./vector-isa-programming-model)
- [x] [RISC-V 向量扩展（RVV）：配置指令与掩码操作](./riscv-vector-extension)
- [x] [向量体系结构的优势：与 SIMD 的本质对比](./vector-vs-simd)
- [x] [指令集设计的量化评估：代码密度与性能权衡](./isa-quantitative-evaluation)

### 第四篇 存储层次深入

- [x] [存储层次的量化回顾：命中时间与平均访存时间（AMAT）](./memory-hierarchy-amat)
- [x] [高级 Cache 优化（一）：小而简单的第一级 Cache](./cache-optimization-small-simple)
- [x] [高级 Cache 优化（二）：路预测与伪相联 Cache](./cache-optimization-way-prediction)
- [x] [高级 Cache 优化（三）：非阻塞 Cache 与缺失数并行](./cache-optimization-nonblocking)
- [x] [高级 Cache 优化（四）：硬件预取与编译器预取](./cache-optimization-prefetching)
- [x] [高级 Cache 优化（五）：编译器优化（循环交换、分块）](./cache-optimization-compiler)
- [x] [高级 Cache 优化（六）：关键字优先与早重启](./cache-optimization-critical-word)
- [x] [高级 Cache 优化（七）：合并写缓冲区与流水化访问](./cache-optimization-write-buffer)
- [x] [Cache 优化技术总结：对命中时间、缺失率与缺失代价的影响](./cache-optimization-summary)
- [x] [虚拟存储器深入：TLB 优化与多级页表](./tlb-optimization-multilevel-paging)
- [x] [虚拟机的存储保护与地址转换加速](./virtual-machines-memory-virtualization)
- [x] [主存技术：DRAM 的内部组织（Bank、行缓冲）](./dram-internal-organization)
- [x] [SDRAM、DDR 系列的演进与带宽提升机制](./sdram-ddr-evolution)
- [x] [存储控制器与调度策略](./memory-controller-scheduling)
- [x] [闪存（Flash）与新型非易失存储（PCM、3D XPoint）](./flash-emerging-nvm)
- [x] [存储可靠性：RAID 级别与纠错码（ECC）](./raid-ecc-reliability)
- [x] [实例剖析：现代服务器的存储层次结构](./server-memory-hierarchy-case)

### 第五篇 线程级并行

- [x] [线程级并行（TLP）与多处理器体系结构概述](./thread-level-parallelism-overview)
- [x] [对称多处理器（SMP）与分布式存储多处理器（DSM）](./smp-vs-dsm)
- [x] [Cache 一致性问题：监听协议（Snooping Protocol）](./cache-coherence-snooping)
- [x] [MESI/MOESI 一致性协议的状态转换](./mesi-moesi-protocols)
- [x] [目录式一致性协议（Directory-based Protocol）](./directory-based-coherence)
- [x] [同步原语：原子操作、锁与栅栏（Barrier）](./synchronization-primitives)
- [x] [一致性模型（Memory Consistency Model）：顺序一致性](./memory-consistency-sequential)
- [x] [弱一致性模型与 Release Consistency](./weak-release-consistency)
- [x] [多核处理器的性能建模与扩展性限制](./multicore-performance-modeling)
- [x] [同时多线程（SMT/Hyper-Threading）的原理与权衡](./simultaneous-multithreading)

### 第六篇 数据级并行

- [x] [数据级并行（DLP）概述：SIMD 与向量体系结构的再比较](./data-level-parallelism-overview)
- [x] [SIMD 指令扩展：从 MMX 到 AVX-512 的演进](./simd-instruction-evolution)
- [x] [SIMD 编程模型与自动向量化](./simd-programming-autovectorization)
- [x] [GPU 体系结构：从图形流水线到通用计算（GPGPU）](./gpu-architecture-overview)
- [x] [GPU 的 SIMT 执行模型：Warp 与线程束调度](./gpu-simt-execution)
- [x] [GPU 存储层次：共享存储、寄存器堆与全局存储](./gpu-memory-hierarchy)
- [x] [GPU 分支分歧（Divergence）与掩码执行](./gpu-branch-divergence)
- [x] [张量核心（Tensor Core）与矩阵运算加速](./tensor-core-matrix-acceleration)
- [x] [循环级并行与依赖分析](./loop-level-parallelism-dependency)

### 第七篇 领域专用体系结构

- [x] [领域专用体系结构（DSA）兴起的背景：摩尔定律的终结](./dsa-rise-background)
- [x] [DSA 的设计原则：专用存储、简化控制与领域匹配](./dsa-design-principles)
- [x] [Google TPU：脉动阵列（Systolic Array）的原理](./google-tpu-systolic-array)
- [x] [TPU 的演进：从推理芯片到训练集群](./tpu-evolution)
- [x] [NPU 与边缘 AI 加速器的设计权衡](./npu-edge-ai-accelerators)
- [x] [神经网络加速中的数据流：Weight/Output/Row Stationary](./nn-dataflow-stationary)
- [x] [稀疏性与量化在 DSA 中的利用](./sparsity-quantization-dsa)
- [x] [DSA 的编程模型与软件栈挑战](./dsa-programming-model-software)

### 第八篇 仓库级计算机与互连

- [x] [仓库级计算机（WSC）：作为一台计算机的数据中心](./warehouse-scale-computer)
- [x] [WSC 的体系结构：服务器、存储与网络组织](./wsc-architecture)
- [x] [延迟与吞吐的权衡：在线服务的尾延迟（Tail Latency）](./tail-latency)
- [x] [WSC 的能效：PUE 与整体成本模型（TCO）](./wsc-energy-pue-tco)
- [x] [MapReduce、Spark 等集群计算框架的体系结构支撑](./mapreduce-spark-cluster-computing)
- [x] [云计算与 WSC 的经济学](./cloud-economics-wsc)
- [x] [互连网络基础：拓扑结构（网格、环面、胖树）](./interconnection-topologies)
- [x] [交换技术：虫孔路由（Wormhole）与虚通道（Virtual Channel）](./wormhole-virtual-channel)
- [x] [集群互连实例：InfiniBand 与以太网的竞争](./infiniband-ethernet)
- [x] [片上网络（NoC）的设计要点](./network-on-chip)

### 第九篇 新兴专题

- [x] [存算一体（Processing-in-Memory）：近数据计算的复兴](./processing-in-memory)
- [x] [基于新型存储的存算一体架构（ReRAM 存算）](./reram-computing-in-memory)
- [x] [Chiplet 与先进封装：超越单芯片缩放](./chiplet-advanced-packaging)
- [x] [Chiplet 互连标准：UCIe 与异构集成](./ucie-heterogeneous-integration)
- [x] [RISC-V 生态：开源指令集对产业格局的影响](./riscv-ecosystem)
- [x] [CXL 互连协议：缓存一致性的设备互连与内存池化](./cxl-interconnect)
- [x] [量子计算体系结构初探](./quantum-computing-architecture)
- [x] [体系结构安全：Spectre、Meltdown 与微架构侧信道](./spectre-meltdown-side-channel)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
