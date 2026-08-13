---
pageClass: plain-doc
---

# AI 硬件：华为昇腾 910B/910C/950

对标华为昇腾官方白皮书与 CANN 编程指南，系统重建从达芬奇（DaVinci）架构、910B/910C/950 芯片平台到软件开发全栈的国产 AI 硬件知识。学完这些章节，就写完了昇腾计算平台从硅片到算力服务的完整内容。

## 对标教材

- 华为昇腾 910B 技术白皮书
- 华为昇腾 910C 技术白皮书
- 昇腾 AI 处理器架构白皮书
- 昇腾 CANN 编程指南
- Atlas 系列硬件技术白皮书

## 主题规划

<ProgressGrid cat="advanced/huawei-ascend-910" />

### 第1篇

- [x] [达芬奇架构总览](./da-vinci-architecture-overview)
- [x] [AI Core 计算单元：Cube、Vector、Scalar](./ai-core-compute-units)
- [x] [AI Core 指令集与流水线设计](./ai-core-instruction-pipeline)
- [x] [片上缓存层次：L0/L1/L2 与数据流](./on-chip-memory-hierarchy)
- [x] [HBM 高带宽内存与带宽设计](./hbm-bandwidth-design)
- [x] [昇腾与 GPU 的架构设计哲学对比](./ascend-vs-gpu-philosophy)

### 第2篇

- [x] [昇腾 910B 规格与产品定位](./ascend-910b-specs)
- [x] [910B 训练/推理变体：910B2/B3/B4](./ascend-910b-variants)
- [x] [昇腾 910C：双 die 封装与超大显存](./ascend-910c-dual-die)
- [x] [昇腾 950 新架构与性能展望](./ascend-950-roadmap)
- [x] [Atlas 硬件形态：Atlas 800T/900 服务器](./atlas-800t-900-server)
- [x] [AI 加速卡接口与服务器集成](./accelerator-interface-server-integration)
- [x] [超节点集群与机间互联拓扑](./supernode-cluster-interconnect)

### 第3篇

- [x] [CANN 软件栈总览](./cann-software-stack)
- [x] [图编译器与算子融合](./graph-compiler-op-fusion)
- [x] [Ascend C 算子编程模型](./ascend-c-programming-model)
- [x] [Cube 矩阵乘算子开发实战](./cube-matmul-development)
- [x] [HCCL 集合通信与多卡并行](./hccl-collective-communication)
- [x] [MindSpore 在昇腾上的运行适配](./mindspore-ascend-adaptation)
- [x] [大模型训练与推理部署实践](./llm-training-inference-deployment)
