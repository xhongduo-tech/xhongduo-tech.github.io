# 大模型部署

把模型变成服务：推理引擎、量化与性能优化的工程全貌。

## 主题规划

- [ ] LLM 推理的特殊性：自回归、Prefill 与 Decode 阶段
- [ ] KV Cache：原理、显存估算与管理
- [ ] 推理引擎全景：vLLM、SGLang、TensorRT-LLM、llama.cpp
- [ ] vLLM 原理：PagedAttention 与 Continuous Batching
- [ ] vLLM 进阶：Chunked Prefill、Prefix Caching 与调度器
- [ ] SGLang 原理：RadixAttention 与结构化生成
- [ ] SGLang 进阶：前后端分离与 Cache-aware 路由
- [ ] TensorRT-LLM：图优化、Kernel 融合与 In-flight Batching
- [ ] 权重量化：GPTQ、AWQ 与 SmoothQuant
- [ ] 低精度推理：FP8、INT4 与 KV Cache 量化
- [ ] 投机解码（Speculative Decoding）与 Medusa/EAGLE
- [ ] 注意力算子优化：FlashAttention 1/2/3
- [ ] 分布式推理：张量并行、流水线并行与专家并行
- [ ] PD 分离（Prefill/Decode Disaggregation）架构
- [ ] 服务化工程：OpenAI 兼容 API、流式输出与负载均衡
- [ ] 推理性能压测：TTFT、TPOT、吞吐量与并发调优
- [ ] 端侧部署：llama.cpp、MLC 与移动芯片适配
- [ ] GPU 选型与成本核算：A100/H100/消费级卡对比
