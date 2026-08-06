---
pageClass: plain-doc
---

# 大模型部署

大模型部署是将训练好的 LLM 高效转化为可用服务的完整工程体系。本篇以推理原理为根基，以 vLLM、SGLang、TensorRT-LLM 等主流引擎为主线，系统梳理从内核优化、量化、解码加速到分布式与服务化的全链路知识。

## 主题规划

<ProgressGrid cat="advanced/llm-deployment" />


### 第一篇 推理基础

- [x] [自回归生成的基本原理](./autoregressive-generation)
- [x] [Prefill 与 Decode 两阶段的计算特征](./prefill-decode-compute)
- [x] [算术强度与 Roofline 模型](./arithmetic-intensity-roofline)
- [x] [访存瓶颈：为什么 Decode 是 Memory-Bound](./decode-memory-bound)
- [x] [KV Cache 的原理与数据结构](./kv-cache-data-structure)
- [x] [KV Cache 显存占用估算与数值实例](./kv-cache-memory-estimation)
- [x] [推理吞吐、延迟与批大小的基本关系](./throughput-latency-batch)
- [x] [模型加载与权重显存布局](./model-loading-weight-memory-layout)

### 第二篇 推理引擎总览

- [x] [LLM 推理引擎要解决的核心问题](./llm-inference-engine-core-problems)
- [x] [主流引擎对比：vLLM、SGLang、TensorRT-LLM、TGI](./engine-comparison)
- [x] [推理框架与训练框架的本质区别](./inference-vs-training-framework)
- [x] [如何阅读和评估一个推理引擎的源码](./how-to-read-engine-source-code)

### 第三篇 vLLM

- [x] [PagedAttention：KV Cache 的页式内存管理](./paged-attention)
- [ ] 块表（Block Table）与逻辑物理块映射
- [ ] Continuous Batching 的原理与实现
- [ ] Chunked Prefill：长输入的分块调度
- [ ] Prefix Caching：共享前缀的缓存复用
- [ ] vLLM 调度器源码分析（一）：请求生命周期
- [ ] vLLM 调度器源码分析（二）：抢占与换入换出
- [ ] vLLM V0 到 V1 架构演进
- [ ] vLLM 的采样、停止条件与后处理
- [ ] vLLM 多 LoRA 服务原理

### 第四篇 SGLang

- [ ] RadixAttention：前缀树的 KV Cache 共享
- [ ] 结构化生成与有限状态机约束解码
- [ ] 前后端分离架构：Router 与 Runtime
- [ ] Cache-aware 路由与负载感知调度
- [ ] SGLang 程序式多轮对话原语

### 第五篇 TensorRT-LLM

- [ ] TensorRT 图优化与算子重写
- [ ] Kernel 融合与自定义 CUDA Kernel
- [ ] In-flight Batching 的原理
- [ ] 量化感知与 TensorRT-LLM 的低精度支持
- [ ] 引擎构建、序列化与部署流程

### 第六篇 量化

- [ ] 量化的基本原理：对称与非对称量化
- [ ] GPTQ：基于 Hessian 的逐层权重量化
- [ ] AWQ：激活感知的权重量化
- [ ] SmoothQuant：权重与激活的困难迁移
- [ ] FP8 量化：E4M3 与 E5M2 格式
- [ ] INT4 权重量化的工程实践
- [ ] KV Cache 量化的收益与精度损失
- [ ] 量化模型的精度评测方法

### 第七篇 解码优化

- [ ] 投机解码的原理与接受率分析
- [ ] 草稿模型的选择与训练
- [ ] Medusa：多头并行投机解码
- [ ] EAGLE：特征级外推的投机采样
- [ ] FlashAttention-1：IO 感知的精确注意力
- [ ] FlashAttention-2：更好的并行与工作分配
- [ ] FlashAttention-3：Hopper 架构的异步与 FP8
- [ ] FlashDecoding 与长序列 Decode 加速

### 第八篇 分布式推理

- [ ] 张量并行（TP）在推理中的实现
- [ ] 流水线并行（PP）与微批调度
- [ ] 专家并行（EP）与 MoE 模型推理
- [ ] PD 分离：Prefill 与 Decode 的解耦部署
- [ ] Mooncake：以 KV Cache 为中心的分离式架构
- [ ] 跨节点 KV Cache 传输与 RDMA
- [ ] 多机推理的通信开销分析

### 第九篇 服务化

- [ ] OpenAI 兼容 API 的设计与实现
- [ ] 流式输出：SSE 与 Token 流协议
- [ ] 推理服务的负载均衡策略
- [ ] API 网关：鉴权、限流与计费
- [ ] 多模型路由与版本管理
- [ ] 服务监控指标与告警体系

### 第十篇 压测与调优

- [ ] TTFT、TPOT、端到端延迟的定义与测量
- [ ] 吞吐、QPS 与并发的关系曲线
- [ ] 压测工具实践：vLLM bench、genai-perf
- [ ] 并发实验：找到服务的拐点与饱和点
- [ ] max-num-seqs 与 max-num-batched-tokens 调优
- [ ] 显存利用率与显存碎片问题排查
- [ ] 延迟异常的定位：从日志到 Profile

### 第十一篇 端侧部署

- [ ] llama.cpp：GGUF 格式与 CPU 推理
- [ ] llama.cpp 的量化方案（Q4_K、Q8_0 等）
- [ ] MLC LLM：面向移动端与浏览器的编译部署
- [ ] 端侧部署的内存与功耗约束
- [ ] 端侧与云端协同推理

### 第十二篇 硬件与成本

- [ ] GPU 推理的关键指标：显存容量、带宽、算力
- [ ] A100、H100、4090 推理性能对比
- [ ] Hopper 架构新特性：HBM3、NVLink、Transformer Engine
- [ ] 消费级显卡部署大模型的可行性分析
- [ ] 单 token 成本的估算模型
- [ ] 自建推理集群与调用 API 的成本权衡

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
