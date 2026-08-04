---
pageClass: plain-doc
---

# LLM Deployment

LLM deployment is the complete engineering system for efficiently turning a trained LLM into a usable service. Taking inference fundamentals as its foundation and mainstream engines such as vLLM, SGLang, and TensorRT-LLM as its main thread, this series systematically covers the full pipeline from kernel optimization, quantization, and decoding acceleration to distributed and service-oriented deployment.

## Topic Plan

<ProgressGrid cat="advanced/llm-deployment" />


### Part 1 · Inference Fundamentals

- [ ] Fundamentals of autoregressive generation
- [ ] Computational characteristics of the Prefill and Decode stages
- [ ] Arithmetic intensity and the Roofline model
- [ ] Memory access bottleneck: why Decode is memory-bound
- [ ] KV Cache principles and data structures
- [ ] Estimating KV Cache GPU memory usage with worked examples
- [ ] Basic relationships among inference throughput, latency, and batch size
- [ ] Model loading and weight memory layout

### Part 2 · Inference Engine Overview

- [ ] Core problems LLM inference engines must solve
- [ ] Comparing mainstream engines: vLLM, SGLang, TensorRT-LLM, TGI
- [ ] Fundamental differences between inference and training frameworks
- [ ] How to read and evaluate an inference engine's source code

### Part 3 · vLLM

- [ ] PagedAttention: paged memory management for the KV Cache
- [ ] Block tables and logical-to-physical block mapping
- [ ] Principles and implementation of Continuous Batching
- [ ] Chunked Prefill: chunked scheduling for long inputs
- [ ] Prefix Caching: cache reuse for shared prefixes
- [ ] vLLM scheduler source analysis (1): request lifecycle
- [ ] vLLM scheduler source analysis (2): preemption and swapping
- [ ] vLLM's architecture evolution from V0 to V1
- [ ] Sampling, stopping conditions, and post-processing in vLLM
- [ ] Multi-LoRA serving in vLLM

### Part 4 · SGLang

- [ ] RadixAttention: prefix-tree-based KV Cache sharing
- [ ] Structured generation and FSM-constrained decoding
- [ ] Decoupled frontend/backend architecture: Router and Runtime
- [ ] Cache-aware routing and load-aware scheduling
- [ ] SGLang's programmatic multi-turn dialogue primitives

### Part 5 · TensorRT-LLM

- [ ] TensorRT graph optimization and operator rewriting
- [ ] Kernel fusion and custom CUDA kernels
- [ ] Principles of in-flight batching
- [ ] Quantization awareness and TensorRT-LLM's low-precision support
- [ ] Engine building, serialization, and deployment workflow

### Part 6 · Quantization

- [ ] Fundamentals of quantization: symmetric and asymmetric quantization
- [ ] GPTQ: Hessian-based layer-wise weight quantization
- [ ] AWQ: activation-aware weight quantization
- [ ] SmoothQuant: migrating difficulty between weights and activations
- [ ] FP8 quantization: E4M3 and E5M2 formats
- [ ] Engineering practice of INT4 weight quantization
- [ ] Benefits and accuracy loss of KV Cache quantization
- [ ] Accuracy evaluation methods for quantized models

### Part 7 · Decoding Optimization

- [ ] Principles of speculative decoding and acceptance-rate analysis
- [ ] Choosing and training draft models
- [ ] Medusa: multi-head parallel speculative decoding
- [ ] EAGLE: feature-level extrapolation for speculative sampling
- [ ] FlashAttention-1: IO-aware exact attention
- [ ] FlashAttention-2: better parallelism and work partitioning
- [ ] FlashAttention-3: asynchrony and FP8 on Hopper
- [ ] FlashDecoding and long-sequence Decode acceleration

### Part 8 · Distributed Inference

- [ ] Implementing tensor parallelism (TP) for inference
- [ ] Pipeline parallelism (PP) and micro-batch scheduling
- [ ] Expert parallelism (EP) and MoE model inference
- [ ] PD separation: decoupled deployment of Prefill and Decode
- [ ] Mooncake: a KV Cache-centric disaggregated architecture
- [ ] Cross-node KV Cache transfer and RDMA
- [ ] Communication overhead analysis for multi-machine inference

### Part 9 · Serving

- [ ] Designing and implementing an OpenAI-compatible API
- [ ] Streaming output: SSE and token-stream protocols
- [ ] Load-balancing strategies for inference services
- [ ] API gateway: authentication, rate limiting, and billing
- [ ] Multi-model routing and version management
- [ ] Service monitoring metrics and alerting

### Part 10 · Benchmarking and Tuning

- [ ] Defining and measuring TTFT, TPOT, and end-to-end latency
- [ ] Throughput, QPS, and concurrency relationship curves
- [ ] Benchmarking tools in practice: vLLM bench, genai-perf
- [ ] Concurrency experiments: finding the service's inflection and saturation points
- [ ] Tuning max-num-seqs and max-num-batched-tokens
- [ ] Troubleshooting GPU memory utilization and fragmentation
- [ ] Locating latency anomalies: from logs to profiling

### Part 11 · On-device Deployment

- [ ] llama.cpp: the GGUF format and CPU inference
- [ ] llama.cpp quantization schemes (Q4_K, Q8_0, etc.)
- [ ] MLC LLM: compiled deployment for mobile and browsers
- [ ] Memory and power constraints of on-device deployment
- [ ] Collaborative on-device and cloud inference

### Part 12 · Hardware and Cost

- [ ] Key GPU inference metrics: memory capacity, bandwidth, compute
- [ ] Inference performance comparison: A100, H100, 4090
- [ ] New Hopper features: HBM3, NVLink, Transformer Engine
- [ ] Feasibility of deploying large models on consumer GPUs
- [ ] A cost-estimation model for per-token cost
- [ ] Cost trade-offs between self-hosted inference clusters and API calls

> After finishing writing: create `xxx.md` in this directory, then change the corresponding item above to `- [x] [标题](./xxx)`.
