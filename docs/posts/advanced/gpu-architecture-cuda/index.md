---
pageClass: plain-doc
---

# GPU 架构与 CUDA 并行编程

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Kirk, Hwu, "Programming Massively Parallel Processors" (4th ed., 2022)
- NVIDIA, "CUDA C++ Programming Guide"（随 CUDA 版本更新的官方指南）
- Hennessy, Patterson, "Computer Architecture: A Quantitative Approach" (6th ed., §4 GPU 章)

## 主题规划

<ProgressGrid cat="advanced/gpu-architecture-cuda" />

### 第1篇

- [ ] GPU 简史：从图形管线到通用计算（GPGPU）
- [ ] SIMT 执行模型与硬件层次（SM/warp/线程束调度）
- [ ] CUDA 编程模型（grid/block/thread、kernel 启动）
- [ ] 内存层次（全局/共享/常量/纹理内存、合并访问）
- [ ] 占用率与延迟隐藏（并行度量化分析）
- [ ] 共享内存与 bank conflict、同步原语
- [ ] Tensor Core 与矩阵运算（WMMA/MMA、与 H100/B200 博文联动）
- [ ] 流、事件与并发执行（计算/传输重叠）

### 第2篇

- [ ] 统一内存与新特性（页迁移、动态并行、协作组）
- [ ] 性能分析与调优（Nsight、Roofline、内存/计算受限判定）
- [ ] 多 GPU 编程（NVLink/NVSwitch、NCCL、与集群博文衔接）
- [ ] 图形管线架构概述（光栅化、光线追踪核心 RT Core）
