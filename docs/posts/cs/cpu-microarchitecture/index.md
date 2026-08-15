---
pageClass: plain-doc
---

# CPU 微架构（乱序/分支预测/存储一致性）

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Hennessy, Patterson, "Computer Architecture: A Quantitative Approach" (6th ed., 2017)
- Shen, Lipasti, "Modern Processor Design: Fundamentals of Superscalar Processors" (2005)
- Patterson, Hennessy, "Computer Organization and Design" (RISC-V ed., 2020)

## 主题规划

<ProgressGrid cat="cs/cpu-microarchitecture" />

### 第1篇

- [x] [ISA 与微架构的分工、流水线基础（五级流水线）](./isa-vs-microarchitecture-pipelining)
- [x] [冒险与 forwarding、流水线控制](./pipeline-hazards-forwarding)
- [x] [分支预测（两位饱和计数器、TAGE、感知机预测器）](./branch-prediction-tage-perceptron)
- [x] [超标量发射与动态调度（记分板、Tomasulo 算法）](./superscalar-dynamic-scheduling)
- [x] [寄存器重命名与重排序缓冲（ROB）、精确异常](./register-renaming-rob-precise-exceptions)
- [x] [推测执行与安全侧信道（Spectre/Meltdown 及缓解）](./speculation-side-channel-spectre-meltdown)
- [x] [存储层次（缓存组织、映射、替换策略、写策略）](./memory-hierarchy-cache-design)
- [x] [预取（硬件预取器、软件预取）](./prefetching-hardware-software)

### 第2篇

- [x] [存储一致性模型（SC/TSO/弱模型）与缓存一致性协议（MESI/MOESI/目录）](./memory-consistency-cache-coherence)
- [x] [多核与片上互连（环形/Mesh/NoC）](./multicore-on-chip-interconnect)
- [x] [SIMD 与向量扩展（SSE/AVX/RVV/SVE）](./simd-vector-extensions)
- [x] [性能建模与评估（IPC 分析、Roofline、微基准测试）](./performance-modeling-evaluation)
