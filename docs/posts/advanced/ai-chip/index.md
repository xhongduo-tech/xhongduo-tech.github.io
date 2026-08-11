---
pageClass: plain-doc
---

# AI 芯片

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- John L. Hennessy & David A. Patterson, "Computer Architecture: A Quantitative Approach" (6th, 2017)
- Vivienne Sze et al., "Efficient Processing of Deep Neural Networks" (Synthesis Lectures 2020)
- Karl G. H. J. et al., "In-Memory Computing" (2019)

## 主题规划

<ProgressGrid cat="advanced/ai-chip" />

### 第1篇

- [x] [GPU 架构与 Tensor Core (Hennessy & Patterson §4)](./gpu-architecture-tensor-core)
- [x] [TPU 脉动阵列 (Jouppi et al., TPU 2017; Hennessy & Patterson §7)](./tpu-systolic-array)
- [x] [AI 加速器数据流 (Sze et al., 2020 §3)](./ai-accelerator-dataflow)
- [x] [硬件加速器设计与编译器映射 (Sze et al., 2020 §6)](./hardware-compiler-mapping)
- [x] [量化与稀疏计算 (Sze et al., 2020 §5)](./quantization-sparse-computing)
- [x] [内存层次与带宽 (Sze et al., 2020 §4)](./memory-hierarchy-bandwidth)
- [x] [近存计算与存内计算 (Karl et al., 2019 §3)](./near-memory-computing)
- [x] [能效评估与 Roofline (Hennessy & Patterson §4.6)](./roofline-model)

### 第2篇

- [x] [AI 芯片评测基准（MLPerf） (Reddi et al., MLPerf 2020)](./mlperf-benchmark)
- [x] [代表性 AI 加速器案例（昇腾/寒武纪/Groq/Habana） (厂商技术白皮书; Jouppi et al., TPU 2017)](./representative-ai-accelerators)
- [x] [边缘 AI 芯片设计 (Sze et al., 2020 §7)](./edge-ai-chip-design)
