## 核心结论

如果团队只需要一个结论，可以先看这一句：`vLLM` 更适合通用开源模型的高吞吐服务，`TGI` 更适合已经深度依赖 Hugging Face 生态、想快速上线并直接使用 OpenAI 兼容接口与 SSE 流式输出的场景，`TensorRT-LLM` 更适合 NVIDIA GPU 数据中心里追求极致延迟和极致吞吐的部署。

这里的“吞吐”是单位时间内处理的 token 数，可以粗略理解成“同样的机器，一秒能产出多少有效回答”。三者的核心差异不在“能不能跑模型”，而在“如何处理 KV cache、如何做批处理、如何利用硬件”。

vLLM 的关键优势是 `PagedAttention`。它把 `KV cache` 做成分页结构。KV cache 可以白话理解为“模型为了继续往后生成，需要把前面已经算过的上下文中间结果存起来”。分页后不要求显存连续分配，能减少碎片，让更大的批次在同样显存里运行。[NVIDIA 的 vLLM 概览](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/overview.html?utm_source=openai)明确把它描述为高吞吐、内存高效的推理引擎，并强调简单 Python API 与 continuous batching。

TGI 的优势是产品化路径短。它直接提供 OpenAI 兼容的 `/v1/chat` 和 `/v1/completions`，支持连续批处理与 SSE 流式输出，适合“先把服务挂起来，再逐步调参”的团队。[Hugging Face Inference Endpoints 文档](https://huggingface.co/docs/inference-endpoints/engines/tgi?utm_source=openai)也明确说明了这点。但要注意，TGI 已在 **2025 年 12 月 11 日**进入维护模式，新生产流量官方更推荐 vLLM 或 SGLang。

TensorRT-LLM 的优势是把 NVIDIA GPU 的能力吃得更深。它提供 in-flight batching、paged KV cache、多 GPU/多节点并行，以及 FP8、INT4 等低精度能力。白话解释，低精度就是“用更小的数据格式存权重或缓存，换更低显存占用和更快速度”。它通常不是“最容易上手”的那个，但经常是“性能天花板最高”的那个。[TensorRT-LLM 官方文档](https://nvidia.github.io/TensorRT-LLM/latest/overview.html)和 [NVIDIA 官方博客](https://developer.nvidia.com/blog/nvidia-tensorrt-llm-now-accelerates-encoder-decoder-models-with-in-flight-batching/?utm_source=openai)都把 in-flight batching、paged attention、量化和 Triton 集成放在核心位置。

| 框架 | 核心定位 | 最适合的团队状态 | 典型交付接口 |
| --- | --- | --- | --- |
| vLLM | 通用开源模型高吞吐服务 | 需要自己掌控服务层、追求吞吐与成本平衡 | Python API、HTTP 服务、OpenAI 兼容接口 |
| TGI | Hugging Face 生态内快速上线 | 想低门槛部署、直接用流式输出和端点控制台 | `/v1/chat`、`/v1/completions`、SSE |
| TensorRT-LLM | NVIDIA 硬件极致优化 | 有 GPU 平台经验、愿意为性能投入工程复杂度 | `trtllm-serve`、Triton、LLM API |

对新手最实用的落地建议是：如果你已经在 Hugging Face Inference Endpoints 上工作，可以先用 TGI 端点把业务跑通，确认接口、日志、流式输出都满足要求；等流量稳定后，再新建 vLLM 端点迁移生产流量。这样能把“先上线”和“后续扩展”拆开处理。

---

## 问题定义与边界

模型服务框架对比，真正要回答的问题不是“谁更快”，而是“在我的模型、我的 GPU、我的上下文长度、我的并发目标下，谁更合适”。

先定义三个量：

- $B$：并发序列数，也就是同一时刻同时处理多少条请求。
- $S$：每条请求涉及的 token 数，可以近似看成输入 token 加输出 token。
- $L$：平均每个 token 的生成延迟。

一个常见的近似关系是：

$$
Q \approx \frac{B \times S}{L}
$$

这里的 $Q$ 表示总体吞吐。这个公式不是严格性能模型，但足够指导工程判断。想提升吞吐，通常有三条路：

1. 提高并发 $B$
2. 允许更长的请求 $S$
3. 降低单 token 延迟 $L$

问题在于，这三者互相挤占资源。更长上下文会占更多 KV cache；更高并发会让显存更紧；更激进的量化会提升速度，但可能带来精度或兼容性问题。

一个典型玩具例子是这样的：假设一张 GPU 在你的配置下，最多容纳 120 万个“活动 token 位点”。如果每个请求上限是 20 万 token，那么理论上最多同时放 6 条；如果每个请求上限改成 10 万 token，就能放 12 条。这不是模型数学本身变化，而是内存预算变化。

Hugging Face 的 TGI 文档给了一个真实工程例子：部署 `meta-llama/Llama-3.3-70B-Instruct` 时，如果 GPU 只能把 128k 上下文装下 3 份，那么保留完整 128k 上下文时只能服务 3 个并发请求；如果把最大输入降到 64k，就可能把并发提升到 6。这个例子非常重要，因为它说明“上下文长度”不是一个免费参数，而是直接影响批处理能力和吞吐的显存预算参数。

边界还包括平台约束。TGI 在 2025 年 12 月 11 日后是维护模式，意味着它仍然可用，但不再是官方重点演进方向；TensorRT-LLM 则强依赖 NVIDIA GPU 与对应的软件栈，不存在“顺手换到 AMD 再看看”的路线；vLLM 的适用面更宽，但不同模型的支持状态、KV cache 策略和并行配置仍需验证。

| 边界条件 | vLLM | TGI | TensorRT-LLM |
| --- | --- | --- | --- |
| 长上下文高并发 | 强，PagedAttention 友好 | 可做，但需仔细调 token 上限 | 强，但通常要结合更复杂的调优 |
| Hugging Face 端点快速上线 | 可选 | 最顺手 | 通常不是第一选择 |
| 维护与演进预期 | 活跃 | 维护模式 | 活跃，但偏 NVIDIA 专用 |
| 硬件限制 | 多数主流 GPU 场景 | 多后端支持，但以常见部署为主 | 明确面向 NVIDIA GPU |
| 多 GPU / 多节点深度优化 | 有 | 有限 | 强项 |

所以，本文的边界很明确：讨论的是开源大模型在线推理服务，不讨论训练，不讨论微调，也不讨论离线批量 embedding 任务。

---

## 核心机制与推导

三者的核心差异可以统一到一个问题：怎样在有限显存下，让更多请求一起跑，并且别让单请求延迟失控。

### 1. vLLM：PagedAttention 把 KV cache 从“整块预留”改成“按页分配”

`PagedAttention` 可以白话理解为“把原本需要整块连续内存的缓存，改成像操作系统分页那样按块管理”。根据 [Hugging Face 对 PagedAttention 的说明](https://huggingface.co/docs/text-generation-inference/conceptual/paged_attention?utm_source=openai)，KV cache 被拆成多个 block，通过查找表访问，不要求连续分配，并按需申请。这样做的直接效果有两个：

- 减少显存碎片和浪费
- 允许更多 batch 共存

如果把显存想成停车场，传统做法像是“每辆车预留一整排连续车位”，很容易留下碎片；PagedAttention 更像“每辆车分散停在多个编号车位上，地图记录在哪里”。车位不需要连续，但仍能找到。

这就是 vLLM 更容易把 $B$ 做大的原因。NVIDIA 文档明确写到，vLLM 借助 PagedAttention 与 continuous batching 提供高吞吐与近零内存浪费。

### 2. TGI：连续批处理 + SSE 流式输出，优先解决“快速上线的服务体验”

`continuous batching` 可以白话理解为“请求不是等一整批凑齐再统一执行，而是运行中持续插入新的请求”。这样可以减少 GPU 空转时间。TGI 文档明确列出 continuous batching、streaming、Paged Attention、KV cache、OpenAI 兼容 API 作为核心特性。

对使用者来说，TGI 的价值常常不是绝对性能第一，而是工程闭环短：

- 端点能直接对接应用层
- 可以开 SSE 流式返回
- 常见参数可直接在控制台或启动参数调整

在公式 $Q \approx B \times S / L$ 里，TGI 的策略更像是通过连续批处理和良好的服务化接口，把 $L$ 和实际可用 $B$ 调到一个稳定的业务区间。

### 3. TensorRT-LLM：把 NVIDIA 的底层优化能力尽量兑现成更小的 $L$

`in-flight batching` 可以白话理解为“请求在飞行途中就能并入批次，不需要等完整批次重新组织”。[NVIDIA 官方博客](https://developer.nvidia.com/blog/nvidia-tensorrt-llm-now-accelerates-encoder-decoder-models-with-in-flight-batching/?utm_source=openai)在 encoder-decoder 支持介绍中明确提到，它通过 in-flight batching、KV cache 管理、dual-paged KV cache 等机制实现高吞吐和低延迟。

TensorRT-LLM 的另一个关键点是量化。根据 [TensorRT-LLM 官方文档](https://nvidia.github.io/TensorRT-LLM/latest/overview.html) 和 [precision 文档](https://nvidia.github.io/TensorRT-LLM/reference/precision.html)，它支持 FP8、INT4 等量化路径。量化的推导逻辑很直接：

- 权重和缓存占用更小
- 显存压力下降
- 在同样显存里可装下更多请求，或减少访存成本
- 于是 $B$ 更容易提升，$L$ 也可能下降

但这不是无条件成立的。量化收益依赖 GPU 架构、模型支持和内核实现。例如官方文档把 H100、B200 等架构的能力差异写得很清楚，不能把“支持 FP8”当成所有卡都同等受益。

### 4. 从 128k 改到 64k，为什么并发能翻倍

还是用 TGI 的真实例子。设显存里可支撑的 KV cache 总预算是 $M$，每条请求在长上下文下大致需要 $m$，那么并发上限近似是：

$$
B_{\max} \approx \left\lfloor \frac{M}{m} \right\rfloor
$$

当你把最大上下文从 128k 降到 64k，单请求的 KV cache 需求近似减半，于是 $m$ 变小，$B_{\max}$ 上升。并发翻倍并不神秘，本质上是“每条请求占的缓存页数少了”。

这也是三种框架都绕不开的工程现实：再好的调度器，也不能突破物理显存预算。框架的价值在于减少浪费，而不是凭空制造内存。

---

## 代码实现

下面先给一个可运行的玩具模型，用来理解“上下文长度下降后并发为什么能上升”。它不是框架源码，只是把上面的预算关系写成最小程序。

```python
def estimate_concurrency(total_token_budget: int, max_tokens_per_request: int) -> int:
    assert total_token_budget > 0
    assert max_tokens_per_request > 0
    return total_token_budget // max_tokens_per_request

# 玩具例子：同一张 GPU，活动 token 总预算固定
budget = 384_000

# 如果每条请求最多 128k token，只能同时服务 3 条
b1 = estimate_concurrency(budget, 128_000)
assert b1 == 3

# 如果每条请求最多 64k token，可同时服务 6 条
b2 = estimate_concurrency(budget, 64_000)
assert b2 == 6

# 简化吞吐模型：Q ≈ B * S / L
def throughput(concurrency: int, tokens_per_request: int, latency_per_token: float) -> float:
    assert concurrency > 0
    assert tokens_per_request > 0
    assert latency_per_token > 0
    return concurrency * tokens_per_request / latency_per_token

q1 = throughput(3, 128_000, 10.0)
q2 = throughput(6, 64_000, 10.0)

# 这个玩具例子里，两者总 token 吞吐相同，但第二种配置能服务更多同时在线用户
assert q1 == q2 == 38400.0

print("128k上下文并发:", b1)
print("64k上下文并发:", b2)
print("吞吐近似:", q1, q2)
```

这个例子说明两点：

- 降上下文上限不一定提高“总 token 吞吐”，但可能提高“同时服务多少用户”
- 业务侧往往更关心并发能力和尾延迟，而不是单一 benchmark 数字

### TGI 的最小部署命令

下面这个例子适合“先跑起来再理解参数”：

```bash
text-generation-launcher \
  --model-id=meta-llama/Llama-3.3-70B-Instruct \
  --port=80 \
  --quantize=awq \
  --max-input-tokens=64000 \
  --max-batch-total-tokens=128000
```

这里的 `--max-input-tokens` 是单请求最大输入长度，`--max-batch-total-tokens` 是整个批次可容纳的 token 总量。两者一起决定并发上限。Hugging Face 的文档和 cookbook 都强调，这类参数直接影响显存占用与批处理效率。

### vLLM 的最小 Python 例子

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
params = SamplingParams(temperature=0.7, max_tokens=128)

outputs = llm.generate(
    ["解释什么是 KV cache，并给一个简短例子。"],
    params
)

text = outputs[0].outputs[0].text
assert isinstance(text, str)
assert len(text) > 0
print(text)
```

这个例子体现的是 vLLM 的上手路径：直接从 Python API 进入，不必先理解整套网关、路由和多进程布局。

### 真实工程例子：先用 TGI 上线，再迁到 vLLM

假设你在做一个面向内部客服的问答系统，第一阶段目标不是压榨极限性能，而是：

- 接口兼容现有 OpenAI SDK
- 支持打字机式流式输出
- 可以在控制台里改模型与 token 上限

这时先在 Hugging Face Inference Endpoints 上建 TGI 端点，能最快接业务。等到你发现主要瓶颈是高峰时段并发不够，或者模型切换更频繁，再按 Hugging Face 的迁移建议新建 vLLM 端点，把流量切过去。这就是“先求交付闭环，再做性能迁移”。

---

## 工程权衡与常见坑

先给结论：部署问题多数不是“框架装不上”，而是“参数、硬件、依赖、预期目标没有对齐”。

| 风险点 | 典型表现 | 更常见于 | 规避方式 |
| --- | --- | --- | --- |
| 上下文设太大 | 并发骤降、显存打满 | 三者都有 | 先按真实请求分布设上限，不按模型理论上限设 |
| 量化兼容性 | 启动失败、精度异常、性能不升反降 | TGI、TensorRT-LLM | 先确认模型和量化格式是否官方支持 |
| 把 benchmark 当生产结果 | 压测数字好看，线上尾延迟差 | 三者都有 | 用真实 prompt 长度和并发分布压测 |
| 驱动/SDK 不匹配 | 内核报错、engine 构建失败 | TensorRT-LLM | 先核对 GPU 架构、驱动、CUDA、TensorRT-LLM 版本 |
| 维护策略误判 | 新需求落不到框架里 | TGI | 新项目优先考虑 vLLM 或 SGLang |

### 1. TGI 的坑：维护模式不是“不能用”，但意味着长期演进预期变弱

这是最容易被忽略的一点。TGI 文档明确说明，自 **2025 年 12 月 11 日**起进入维护模式，只接受小型 bug 修复、文档改进和轻量维护任务。对已经在线的服务，这不意味着必须立刻下线；但对新项目，这意味着你不应该把未来两年的核心推理平台赌在它身上。

如果你还依赖 bitsandbytes、GPTQ 等量化路径，更要把依赖版本锁清楚。因为这类问题一旦出现，往往不是“改一行业务代码”能解决，而是容器版本、模型权重格式、CUDA 内核三方联动。

### 2. TensorRT-LLM 的坑：性能高，环境门槛也高

TensorRT-LLM 的问题通常不在概念，而在环境。比如：

- 你是否真的运行在 NVIDIA GPU 上
- GPU 架构是否支持你要用的精度路径
- 驱动、CUDA、TensorRT、TensorRT-LLM 版本是否匹配
- 模型是否已有成熟 recipe

官方文档明确区分了 Hopper、Ada、Ampere 等架构对 FP8、INT4 的支持情况。不能把“官方支持 INT4”理解成“任意模型、任意卡、任意命令都能稳定吃到收益”。

### 3. vLLM 的坑：吞吐强，不代表参数可以随便开

很多人第一次用 vLLM，会因为它吞吐好，就把并发、上下文、采样参数一起拉高。结果是：

- 显存水位过高
- 长尾请求把批处理拖慢
- 线上表现与单轮压测差距很大

vLLM 的优势是调度和缓存管理，不是替你自动完成所有容量规划。真实生产里，仍要先回答三个问题：

- 你的 prompt 长度分布是什么
- 你的输出长度分布是什么
- 你要优先保吞吐，还是优先保单请求时延

---

## 替代方案与适用边界

如果把三者放进一个决策树里，可以简单判断。

第一类团队：要最快上线、已经在 Hugging Face 体系里工作。  
这时 TGI 仍然有现实价值，尤其适合已有端点、已有 OpenAI SDK 接入、已有 SSE 流式返回的业务。但新建生产服务时，官方已经更推荐 vLLM 或 SGLang。也就是说，TGI 更像“现有资产可延续”，而不是“未来主力方向”。

第二类团队：主要服务开源 decoder-only 模型，希望在常规 GPU 集群上拿到高吞吐和较低开发复杂度。  
这时 vLLM 往往是更稳的默认选项。它的 PagedAttention、continuous batching 和 Python 友好性，使它在“自己掌控服务层”的场景里非常强。

第三类团队：明确在 NVIDIA 数据中心环境里，愿意投入更多部署和调优工作，以换取更低延迟、更高吞吐、更多多 GPU/多节点能力。  
这时 TensorRT-LLM 更合适，尤其在需要 Triton 集成、复杂并行、encoder-decoder 或多模态组合部署时更有优势。NVIDIA 官方博客已经把 encoder-decoder、dual-paged KV cache、Triton backend 支持列为重点能力。

| 方案 | 适合模型类型 | 适合硬件 | 运维负担 | 适用边界 |
| --- | --- | --- | --- | --- |
| vLLM | 开源 decoder-only 为主，也覆盖更多通用场景 | 主流 GPU 集群 | 中 | 追求吞吐与灵活性平衡 |
| TGI | Hugging Face 支持良好的开源模型 | 常见云 GPU / HF 端点 | 低到中 | 适合快速上线，但新项目要考虑维护模式 |
| TensorRT-LLM | decoder-only、encoder-decoder、多模态高性能场景 | NVIDIA GPU，尤其数据中心卡 | 高 | 适合性能优先、平台工程能力较强的团队 |
| SGLang | 新一代高性能服务候选 | 主流 GPU 集群 | 中 | 当你要替代 TGI 且关注活跃演进时可评估 |

所以最终不是“谁最好”，而是“谁最符合你的约束”。对初级工程师来说，最有价值的判断标准不是 benchmark 排名，而是下面这句：

- 先看你的硬件约束
- 再看你的服务接口要求
- 最后看你的长期维护路线

---

## 参考资料

- [NVIDIA vLLM Overview](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/overview.html?utm_source=openai)
- [Hugging Face Inference Endpoints: Text Generation Inference (TGI)](https://huggingface.co/docs/inference-endpoints/engines/tgi?utm_source=openai)
- [Hugging Face: PagedAttention](https://huggingface.co/docs/text-generation-inference/conceptual/paged_attention?utm_source=openai)
- [Hugging Face Cookbook: Benchmarking TGI](https://huggingface.co/learn/cookbook/benchmarking_tgi?utm_source=openai)
- [NVIDIA Technical Blog: TensorRT-LLM Now Accelerates Encoder-Decoder Models with In-Flight Batching](https://developer.nvidia.com/blog/nvidia-tensorrt-llm-now-accelerates-encoder-decoder-models-with-in-flight-batching/?utm_source=openai)
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/latest/overview.html)
- [TensorRT-LLM Numerical Precision](https://nvidia.github.io/TensorRT-LLM/reference/precision.html)
- [TensorRT-LLM Support Matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html)
