---
pageClass: plain-doc
---

# 华为 vllm-ascend

对标 vllm-ascend 官方仓库与华为昇腾推理文档，按章节逐节写成博文，系统重建从昇腾 NPU 软硬件栈、vLLM 硬件可插拔架构、Ascend 算子适配到并行部署与性能调优的完整推理引擎知识。学完这个专题，就写完了在华为昇腾 NPU 上运行 vLLM 大模型推理的全貌。

## 对标教材

- vllm-project, "vllm-ascend"（官方仓库与文档, github.com/vllm-project/vllm-ascend）
- 华为, "昇腾 CANN 推理与应用开发指南"（官方文档, Ascend 社区）
- vLLM, "vLLM 官方文档：Ascend Backend"（vllm.ai / docs.vllm.ai/projects/ascend）

## 主题规划

<ProgressGrid cat="advanced/huawei-vllm-ascend" />

### 第1篇

- [x] [vllm-ascend 概览：定位、特性与支持矩阵（vllm-ascend 官方文档 QuickStart）](./vllm-ascend-overview)
- [x] [昇腾 NPU 硬件与软件栈（Atlas 910B/310P、CANN、torch_npu）（华为昇腾推理文档 第1章）](./ascend-hardware-software-stack)
- [x] [环境安装与版本匹配（CANN/PyTorch/torch_npu 组合）（vllm-ascend 官方文档 Installation）](./installation-version-matrix)
- [x] [vllm serve 快速启动与在线推理（vllm-ascend 官方文档 QuickStart）](./vllm-serve-quickstart)
- [x] [支持模型矩阵与模型加载流程（vllm-ascend 官方文档 Support Matrix）](./supported-models)
- [x] [Docker 镜像与多形态部署（Atlas 310P/A3 镜像、openEuler）（vllm-ascend 官方仓库 Dockerfile）](./docker-deployment)

### 第2篇

- [x] [vLLM 硬件可插拔架构与 Ascend 后端（vLLM RFC Hardware Pluggable）](./vllm-pluggable-backend)
- [x] [vllm_ascend 包结构：平台层、算子层与执行器（vllm-ascend 官方仓库 vllm_ascend/）](./vllm-ascend-package-structure)
- [x] [PagedAttention 在昇腾上的实现与 KV Cache 管理（vllm-ascend 官方仓库 csrc/）](./paged-attention-ascend)
- [x] [自定义融合算子与 Ascend C 内核开发（华为昇腾 CANN 算子开发文档）](./ascend-custom-kernels)
- [x] [FlashAttention 昇腾实现与张量指令优化（华为昇腾推理文档 / vllm-ascend csrc/）](./flash-attention-ascend)
- [x] [ATC 图编译与算子图优化（华为昇腾 CANN 推理文档）](./atc-graph-compilation)

### 第3篇

- [x] [连续批处理与 PagedAttention 显存管理（vllm-ascend 官方文档 推理特性）](./continuous-batching)
- [x] [Prefix Caching 前缀缓存与复用（vllm-ascend 官方文档 推理特性）](./prefix-caching)
- [x] [Chunked Prefill 分块预填充与首字延迟（vllm-ascend 官方文档 推理特性）](./chunked-prefill)
- [x] [FP16/BF16 与 INT8 量化精度策略（华为昇腾量化文档 / vllm-ascend）](./quantization-precision)
- [x] [投机解码与并行采样（vllm-ascend 官方文档 推理特性）](./speculative-decoding)
- [x] [多模态大模型与 Embedding 模型推理支持（vllm-ascend 官方文档 Support Matrix）](./multimodal-embedding)

### 第4篇

- [x] [张量并行与多卡推理（vllm-ascend 官方文档）](./tensor-parallelism)
- [x] [专家并行 EP 大规模部署实战（vllm-ascend 官方文档 Tutorials）](./expert-parallel-deployment)
- [x] [性能基准测试与吞吐优化（vllm-ascend 官方仓库 benchmarks/）](./performance-benchmark)
- [x] [显存优化与长序列推理内存管理（华为昇腾推理文档）](./memory-optimization)
- [x] [msprof 性能剖析与瓶颈定位（华为昇腾 CANN 工具文档）](./msprof-profiling)
- [x] [常见问题排查与 FAQ（vllm-ascend 官方文档 / 社区）](./troubleshooting-faq)
