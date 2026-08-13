---
title: vllm serve 快速启动与在线推理
date: 2026-08-07
---

# vllm serve 快速启动与在线推理

<div class="epigraph">
<p>先让它跑起来，再让它跑得快。</p>
<footer>—— 陈皓（左耳朵耗子）</footer>
</div>

<div class="article-byline">
<p>第四级 · 华为 vllm-ascend ｜ vllm-ascend 官方文档 QuickStart ｜ 2026-08-07</p>
</div>

## 为什么从快速启动开始

环境装好后，第一件有成就感的事，就是让模型真正「开口说话」。`vllm serve` 一条命令，把权重加载、计算图编译、KV Cache 初始化、调度器上线、HTTP 服务监听全部串起来。但这条命令不是黑盒：**每多懂一个启动参数，你就多控制了一分内存占用与延迟。** 这一篇以昇腾后端为背景，把「一条命令起服务」拆成「服务启动时发生了什么」与「服务起来后怎么调」，为后面所有推理特性打好使用层面的基础。

## 1 一条命令的解剖

先看一条典型的昇腾启动命令：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 1
```

四个参数各管一件事：

**`--dtype bfloat16`**：推理权重与激活的默认精度。昇腾 910B 对 BF16 友好，多数模型官方推荐 BF16；FP16 也可用，但 BF16 在长尾数值上有更宽的指数范围。

**`--max-model-len 8192`**：模型服务的**上下文窗口上限**。它决定 KV Cache 预分配规模的上界，8192 表示单个序列最长允许 8192 个 token（含输入与输出）。

**`--gpu-memory-utilization 0.9`**：允许引擎占用 NPU 显存的**比例上限**。0.9 意味着最多用 90% 显存，留一点余量给运行时；不是设越高越好，设太高可能导致加载或换页时显存不足。

**`--tensor-parallel-size 1`**：单卡张量并行度。用几张卡切分同一个模型就设几；单卡推理保持 1。多卡并行的细节见第 4 篇《张量并行与多卡推理》。<span class="marginnote">启动参数远不止这四个，`--max-num-seqs` 控制一个批里最多几个序列、`--enable-prefix-caching` 打开前缀缓存、`--quantization` 指定量化方式。<strong>先记住一个原则：能显式写的都显式写，别依赖默认值</strong>——默认值是按通用场景调的，不一定适合你的显存与并发。</span>

## 2 启动时发生了什么

从敲下回车到出现 `Uvicorn running on http://0.0.0.0:8000`，昇腾后端大致经历五步：

**第一步，初始化设备与后端**：检测 `npu` 设备、申请 CANN 上下文、注册 vllm-ascend 的算子，日志里会出现昇腾后端初始化信息。

**第二步，加载模型权重**：从 Hugging Face 缓存或本地目录读取权重，切分到目标设备。模型较大时这一步耗时最长。

**第三步，构造计算图并编译**：vLLM 把模型前向组装成执行图。昇腾端会做算子融合与图优化（第 2 篇《ATC 图编译与算子图优化》），首次启动的编译时间明显长于纯 CUDA 场景，属于正常现象。

**第四步，初始化 KV Cache 管理器**：根据 `max-model-len`、`gpu-memory-utilization` 与批量上限，计算 KV Cache 总预算并切成页块，交给 PagedAttention 管理器（第 3 篇《连续批处理与 PagedAttention 显存管理》）。

**第五步，拉起 HTTP 服务**：以 OpenAI 兼容的接口监听 `0.0.0.0:8000`，调度器线程就绪，开始接收请求。

**辨析｜易错点：** 首启编译时间长 ≠ 卡死。昇腾的图编译在**每次进程启动时都会发生**，可能耗时几分钟；如果反复重启服务，这段等待会非常磨人。生产环境通常用「先预热再对外」或更长驻进程的方式规避，而不是把启动慢当成故障去查。

## 3 OpenAI 兼容接口与在线推理

服务起来后，一切交互都走 HTTP。vLLM 提供 OpenAI 兼容的接口，意味着 **OpenAI SDK、LangChain 等工具几乎零改造即可对接**。

一个最小请求：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "用一句话解释什么是 KV Cache。"}],
    "max_tokens": 128
  }'
```

返回体里最重要的字段是 `choices[0].message.content`（回答文本）与 `usage`（`prompt_tokens`、`completion_tokens` 计数）。

如果希望**流式输出**，在请求里加 `"stream": true`——服务端会把回答按 token 分片以 `data: {...}` 的 SSE 格式推回，用户在 UI 上看到的就是「一个字一个字蹦出来」的效果。流式不仅改善体验，也降低首字延迟的感知。

<span class="marginnote">除了 `/v1/chat/completions`，vLLM 还提供 `/v1/completions`（纯文本续写）、`/v1/embeddings`（向量化）、`/v1/models`（列出已加载模型）等端点。Embedding 与多模态的推理支持见第 3 篇《多模态大模型与 Embedding 模型推理支持》。</span>

## 4 公式解析：TTFT 与 TPOT

在线推理的体验，本质上由两个数字刻画。设第 $i$ 个 token 的解码耗时约等于两次相邻 token 产生的间隔：

**首字延迟（TTFT，Time To First Token）**：从请求到达服务端，到客户端收到**第一个**输出 token 的时间。它由 prefill（处理输入）主导：

$$
T_{\text{TTFT}} \approx T_{\text{prefill}} + T_{\text{传输}}
$$

**每个输出 token 的间隔（TPOT，Time Per Output Token）**：相邻两个输出 token 的时间差。decode 阶段一次前向只产出若干个 token，TPOT 由解码步耗时主导：

$$
T_{\text{TPOT}} \approx \frac{\text{一次 decode 步耗时}}{\text{该步产出的 token 数}}
$$