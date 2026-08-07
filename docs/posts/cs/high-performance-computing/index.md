---
pageClass: plain-doc
---

# 高性能计算

陈国良《并行计算》、HPC 体系（Top500）。按照「学完一个学科 = 写完该学科权威教材对应的全部博文」的标准，每写完一篇勾掉一条。

## 主题规划

<ProgressGrid cat="cs/high-performance-computing" />

### 第一篇 并行计算基础

- [x] [高性能计算的演进与 Top500](./evolution-and-top500)
- [x] [并行计算机体系结构（SMP/MPP/集群）](./parallel-architectures)
- [x] [并行性度量（加速比/效率/Amdahl）](./parallel-metrics-amdahl)
- [x] [并行计算模型（PRAM/BSP）](./parallel-computing-models)
- [x] [任务分解与映射](./task-decomposition-mapping)
- [x] [负载均衡策略](./load-balancing)
- [x] [通信与同步机制](./communication-synchronization)

### 第二篇 并行编程模型

- [x] [MPI 基础（进程/通信子）](./mpi-basics)
- [x] [MPI 集体通信](./mpi-collective-communication)
- [x] [MPI 点对点与非阻塞通信](./mpi-point-to-point-nonblocking)
- [x] [OpenMP 基础（并行区/工作共享）](./openmp-basics)
- [x] [OpenMP 同步与归约](./openmp-synchronization-reduction)
- [x] [CUDA 与 GPU 并行编程](./cuda-gpu-programming)
- [x] [混合并行（MPI+OpenMP/CUDA）](./hybrid-parallelism)
- [x] [分区全局地址空间模型](./partitioned-global-address-space)

### 第三篇 并行算法

- [x] [并行排序算法](./parallel-sorting)
- [x] [并行前缀和与归约](./parallel-prefix-sum-reduction)
- [x] [并行搜索](./parallel-search)
- [x] [并行图算法](./parallel-graph-algorithms)
- [x] [并行矩阵乘法](./parallel-matrix-multiplication)
- [x] [并行线性方程组求解](./parallel-linear-solvers)
- [x] [并行 FFT](./parallel-fft)
- [x] [并行 PDE 求解](./parallel-pde-solvers)
- [x] [检查点与容错](./checkpointing-fault-tolerance)
- [x] [性能剖析与优化](./performance-profiling-optimization)
- [x] [领域应用（气候/材料/流体/生物）](./domain-applications)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
