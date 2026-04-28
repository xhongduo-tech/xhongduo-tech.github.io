## 核心结论

模型推理优化迭代，可以先用一句话定义：**在正确性可控前提下，减少重复计算、减少调度等待、降低单 token 成本。**

这里的“推理”就是模型已经训练完成后，真正对外回答问题的过程；“优化”不是单指某个开关，而是一轮轮定位瓶颈、验证收益、再继续收敛参数的工程过程。对零基础读者最重要的一点是：**推理慢，往往不是因为乘法不够快，而是因为系统反复做了不该做的事。**

一个玩具例子：同一套问答机器人，每次都要先读 1500 个 token 的系统提示词和知识库摘要，再开始回答。如果这 1500 个 token 在很多请求里都相同，那么反复重算它们就是浪费。把这一段的中间结果记住，下次直接复用，速度通常就会明显变好。

一个真实工程例子：RAG 服务里常见“系统提示词 + 安全策略 + 检索出的长文档”作为前缀。这类前缀在同一个会话、同一个租户、同一个知识库模板下高度重复。此时 `prefix caching` 往往先带来最直接的收益：降低 TTFT，也就是首个 token 延迟，并减少总体 GPU 计算量。

先看总览：

| 优化手段 | 主要解决的问题 | 主要收益指标 | 典型适用场景 |
|---|---|---|---|
| prefix caching | 重复前缀被反复 prefill | TTFT、成本 | RAG、客服、多轮会话 |
| chunked prefill | prefill 过长阻塞 decode | 尾延迟、资源利用率 | 长输入混合短输出 |
| disaggregated prefill | prefill 与 decode 互相干扰 | p95/p99 延迟 | 强交互、低尾延迟要求 |
| speculative decoding | decode 串行步数太多 | ITL、tokens/s | draft 模型质量较高 |
| quantization | 单步算得贵、显存紧 | 成本、吞吐、显存 | 显存敏感、硬件支持好 |
| efficient attention kernel | 注意力计算访存重 | tokens/s、吞吐 | 长上下文、高性能 GPU |

所以，推理优化不是“堆概念”，而是围绕四个核心指标做判断：`TTFT`、`ITL`、`tokens/s`、单位 token 成本。

---

## 问题定义与边界

要优化，先要把问题定义准确。

`prefill` 是模型先把输入上下文整体读一遍并建立内部状态，可以白话理解为“先把题目看完”；`decode` 是模型开始一个 token 一个 token 地往外生成，可以理解为“开始逐字作答”。这两个阶段都慢，但慢的原因不同。

注意力机制的基本形式是：

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt{d})V
$$

这里的 $Q,K,V$ 分别是 query、key、value，可以白话理解为“当前 token 去对照历史 token 时所用的三组向量”。在自回归生成里，历史越长，模型就越依赖保存和访问过去的状态。

这就引出 `KV cache`。`KV cache` 是保存历史 token 的 key 和 value，避免每生成一个新 token 都把过去整段上下文重算一遍。它的近似内存规模常写成：

$$
M_{KV}\approx 2\cdot L\cdot d\cdot b\cdot T
$$

其中：
- $L$ 是层数
- $d$ 是隐藏维度
- $b$ 是每个元素所占字节数
- $T$ 是缓存的 token 数
- 前面的 `2` 表示 K 和 V 各存一份

这条式子说明一个基础事实：**上下文越长，KV cache 越占显存；显存越紧，调度越难。**

必要定义可以压成一张表：

| 术语 | 定义 | 新手白话解释 |
|---|---|---|
| prefill | 处理输入上下文 | 先把整段输入读完 |
| decode | 逐 token 生成输出 | 一个字一个字往外写 |
| KV cache | 缓存历史 K/V | 把读过的内容做成可复用笔记 |
| TTFT | Time To First Token | 用户等到第一个字出现的时间 |
| ITL | Inter-Token Latency | 连续两个 token 之间的间隔 |

边界同样要说清楚。不同负载，瓶颈不同：

| 负载类型 | 主要瓶颈 | 更适合的优化方向 |
|---|---|---|
| 长输入、短输出 | prefill 重 | prefix caching、chunked prefill |
| 短输入、长输出 | decode 串行 | speculative decoding、kernel 优化 |
| 显存紧张 | KV cache、权重驻留 | quantization、分页缓存 |
| 强交互低延迟 | 尾延迟波动 | disaggregated prefill、细粒度调度 |

一个反例很重要：**speculative decoding 不是总能更快。**如果 draft 模型提议的 token 经常被 target 模型否决，那么你只是多跑了一套预测，再多付一次验证成本，最终可能更慢。

---

## 核心机制与推导

推理优化是否成立，可以统一成一句判断：**额外开销必须小于节省下来的基线成本。**

### 1. prefix caching 为什么有效

玩具例子：有 100 个请求都共享前 1536 个 token 的系统前缀。如果不用缓存，每个请求都要把这 1536 个 token 重新 prefill 一遍；如果用了缓存，只需要第一次算，后面直接复用对应的 KV block。

这相当于把“每个人都重新抄一遍公式”变成“大家共享同一份板书”。

它成立的原因很直接：共享前缀的历史状态完全相同，复用不会改变结果，只是跳过了重复计算。它的适用边界也同样直接：**没有重复前缀，就没有复用收益。**

### 2. chunked prefill 为什么能改善调度

`chunked prefill` 的意思是把很长的一次性 prefill 切成更小的块，让它和 decode 更容易交错执行。白话解释：不要让一个超长输入一次占满整条流水线，而是分段进入队列。

如果不切块，一个 16k 输入的请求可能长时间占住资源，后面的短请求和正在 decode 的会话都要等。切块后，调度器更容易在 prefill 和 decode 之间做平衡，因此 p95、p99 延迟往往更稳定。

但它不是白赚。块切得太细，会引入更多调度和切换开销；块太大，又接近没切。

### 3. speculative decoding 为什么能减少串行步数

`speculative decoding` 可以白话理解为“小模型先打草稿，大模型再批量审稿”。

设 target 模型的条件分布是 $p_i$，draft 模型的条件分布是 $q_i$，第 `i` 个提议 token 的接受概率可写作：

$$
a_i=\min(1, p_i/q_i)
$$

你不必把这条式子理解成背概率论公式，只要抓住工程意义：**draft 和 target 越一致，接受率越高；接受率越高，一次验证能确认的 token 越多。**

于是有一个非常实用的成本判断：

$$
\frac{\text{draft成本}+\text{verify成本}}{\mathbb{E}[\text{一次验证拿到的token数}]}
<
\text{基线逐token解码成本}
$$

只有在这条不等式大致成立时，speculative decoding 才值得开。否则就是多做了一轮无效工作。

### 4. 量化与高效 kernel 为什么常常最后上

`quantization` 是把权重或激活从高精度换成低精度，例如 FP16 到 FP8 或 INT4。白话解释：用更省空间、搬运更快的数字表示法，换取更低的内存压力和更高的算力利用率。

`efficient attention kernel` 是更贴近 GPU 访存特点的专用实现，例如 FlashAttention。白话解释：同样的数学运算，换一种更少搬内存的数据路径来做。

这两类优化通常都有效，但比 prefix caching 更“硬件相关”，也更容易引入精度、兼容性、实现复杂度问题。所以在真实系统里，常见顺序是：

1. 先消除重复计算
2. 再改善 prefill/decode 调度
3. 再考虑 decode 串行优化
4. 最后叠加低精度和 kernel 优化

机制对照如下：

| 机制 | 解决什么 | 代价是什么 | 适合什么场景 |
|---|---|---|---|
| prefix caching | 重复前缀重算 | 需要缓存管理 | 重复模板、重复文档前缀 |
| chunked prefill | 超长 prefill 阻塞 | 参数调优复杂 | 长输入混跑 |
| disaggregated prefill | prefill/decode 资源争用 | 架构更复杂 | 追求低尾延迟 |
| speculative decoding | decode 串行过强 | 依赖接受率 | 小草稿模型足够准 |
| quantization | 显存和单步成本高 | 可能掉精度 | 成本敏感、硬件支持足 |
| efficient kernel | 注意力访存重 | 依赖平台实现 | 长上下文高吞吐 |

---

## 代码实现

代码实现的正确顺序不是“把所有开关都打开”，而是：**先接入，再调参，再验证。**

对新手最稳的接入路径通常是：
1. 先开 `prefix caching`
2. 再开 `chunked prefill`
3. 观察 `TTFT`、`p95 ITL`
4. 只有在 decode 仍明显成为瓶颈时，再试 `speculative decoding`
5. 显存仍紧或成本压力明显时，再做量化

一个简化的配置示意：

```python
class LLMEngine:
    def __init__(
        self,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_num_batched_tokens=4096,
        speculative_tokens=0,
        quantization=None,
    ):
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_chunked_prefill = enable_chunked_prefill
        self.max_num_batched_tokens = max_num_batched_tokens
        self.speculative_tokens = speculative_tokens
        self.quantization = quantization


engine = LLMEngine(
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    max_num_batched_tokens=8192,
    speculative_tokens=0,
    quantization=None,
)
```

下面给一个可运行的 Python 玩具脚本，用来模拟“什么时候 speculative decoding 可能有收益”。它不是框架源码，但足够表达成本判断逻辑。

```python
def speculative_benefit(
    baseline_cost_per_token: float,
    draft_cost: float,
    verify_cost: float,
    accepted_tokens_per_round: float,
) -> bool:
    assert baseline_cost_per_token > 0
    assert draft_cost >= 0
    assert verify_cost >= 0
    assert accepted_tokens_per_round > 0

    speculative_cost_per_token = (
        draft_cost + verify_cost
    ) / accepted_tokens_per_round

    return speculative_cost_per_token < baseline_cost_per_token


# 玩具例子：一次验证平均拿到 3 个 token，收益成立
assert speculative_benefit(
    baseline_cost_per_token=1.0,
    draft_cost=0.6,
    verify_cost=1.8,
    accepted_tokens_per_round=3.0,
) is True

# 反例：接受率差，一次只拿到 1 个 token，收益不成立
assert speculative_benefit(
    baseline_cost_per_token=1.0,
    draft_cost=0.6,
    verify_cost=1.8,
    accepted_tokens_per_round=1.0,
) is False
```

真实工程例子可以这样落地：一个企业知识库 RAG 服务，在同一租户下，大部分请求都共享系统提示词、安全规则、检索模板与一部分文档前缀。此时配置上优先打开前缀缓存，并采集前缀命中率、TTFT、显存占用。若高峰期发现长文档请求让交互延迟抖动，再引入 chunked prefill 或将 prefill 与 decode 拆开部署。

参数理解建议用表格记：

| 参数名 | 含义 | 调大效果 | 调小效果 | 常见副作用 |
|---|---|---|---|---|
| `enable_prefix_caching` | 是否复用共享前缀 KV | 降低重复 prefill | 无缓存收益 | 需要缓存命中管理 |
| `max_num_batched_tokens` | 单批次 token 上限 | 吞吐可能更高 | TTFT 可能更稳 | 过大可能压制 decode |
| `speculative_tokens` | 每轮草稿 token 数 | 理论并行更多 | 额外开销更小 | 盲目调大易失败 |
| `quantization` | 低精度模式 | 更省显存 | 精度更保守 | 可能带来质量回退 |

验证指标也必须固定下来：

| 指标 | 看什么 | 为什么重要 |
|---|---|---|
| `TTFT` | 第一个 token 出现的时间 | 最接近用户“卡不卡”感知 |
| `p50/p95 ITL` | 输出过程是否稳定 | 体现平均体验和尾延迟 |
| `throughput` | 总吞吐量 | 体现系统利用率 |
| `GPU memory` | 显存占用曲线 | 决定是否能放大 batch |
| `acceptance rate` | speculative 的接受率 | 决定它到底赚不赚钱 |

---

## 工程权衡与常见坑

工程里最常见的错误，不是技术不会用，而是**指标看错了**。

只看 `tokens/s` 就下结论，是非常典型的误判。因为一个系统可能总体吞吐变高，但 `TTFT` 明显恶化，用户仍会觉得“它变慢了”。对聊天类产品，首 token 往往比总吞吐更敏感；对离线批处理，吞吐反而更重要。

常见坑可以直接列出来：

| 坑位 | 问题本质 | 规避方式 |
|---|---|---|
| 只看 `tokens/s` | 局部指标代替全局体验 | 同时看 `TTFT`、`p95 ITL` |
| speculative 一味增大 token 数 | 忽略接受率 | 先测 `acceptance rate` |
| 以为 prefix caching 总有收益 | 没有重复前缀就无收益 | 先统计前缀复用率 |
| chunked prefill 参数拍脑袋 | 调度平衡被打破 | 做压测扫参数 |
| 全层量化一步到位 | 敏感层可能失真 | 做 A/B 和质量回归 |
| 把 disaggregated prefill 当吞吐手段 | 它更偏尾延迟治理 | 明确优化目标再上 |

一个真实工程坑：`max_num_batched_tokens` 设太大，prefill 吞掉太多资源，正在生成中的 decode 请求就会被饿住，结果是平均吞吐看起来不错，但 p95 ITL 变差，用户看到输出一顿一顿。反过来，设太小又会让长前缀请求排太久，TTFT 上升。

所以排障顺序最好固定：
1. 先看是否存在重复前缀
2. 再看 prefill 和 decode 是否互相阻塞
3. 再看显存是否真的是瓶颈
4. 最后才考虑更激进的量化或 speculative decoding

这套顺序的价值在于：越往后，收益越依赖实现细节，调试和回归成本也越高。

---

## 替代方案与适用边界

不同优化不是互相替代，而是处理不同类型的问题。

如果请求几乎没有重复前缀，`prefix caching` 的收益会很有限。这时更应该优先考虑 attention kernel、量化、batch 调度，或者直接缩短无效上下文。因为没有复用对象，再好的缓存策略也无从发挥。

如果系统最在意的是尾延迟，而不是总吞吐，那么 `disaggregated prefill` 往往比 speculative decoding 更对症。原因很简单：前者直接处理 prefill 对 decode 的资源干扰，后者主要处理 decode 自身的串行性。

可以用一张替代方案表来做决策：

| 问题类型 | 首选方案 | 备选方案 | 不适用条件 |
|---|---|---|---|
| 重复前缀多 | prefix caching | prompt 归一化 | 前缀几乎不重复 |
| 长输入拖慢响应 | chunked prefill | disaggregated prefill | 输入普遍很短 |
| decode 太慢 | speculative decoding | efficient kernel | 接受率过低 |
| 显存紧张 | quantization | 减少上下文、减 batch | 硬件不支持低精度 |
| 尾延迟不稳 | disaggregated prefill | 更细粒度调度 | 系统复杂度预算很低 |

可以把选择逻辑理解成一个简化决策树：
1. 是否有大量重复前缀？
2. 是否是 prefill 过重而不是 decode 过慢？
3. 是否显存先到上限？
4. 是否真的需要更低成本，而不是更低延迟？
5. 当前硬件是否支持对应优化路径？

因此，所谓“模型推理优化迭代”，真正的“迭代”二字体现在：每一轮只解决当前最主要的瓶颈，不把别人的最佳实践当成自己的默认答案。

---

## 参考资料

1. [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention and Continuous Batching](https://huggingface.co/papers/2309.06180)
2. [vLLM 文档：Performance and Tuning / Chunked Prefill](https://docs.vllm.ai/en/v0.4.2/models/performance.html)
3. [vLLM 文档：Automatic Prefix Caching](https://docs.vllm.ai/en/v0.11.2/features/automatic_prefix_caching/)
4. [vLLM 文档：Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
5. [Speculative Sampling 论文](https://huggingface.co/papers/2302.01318)
6. [FlashAttention 论文](https://huggingface.co/papers/2205.14135)
7. [NVIDIA TensorRT-LLM 官方文档](https://docs.nvidia.com/tensorrt-llm/index.html)
8. [vLLM 文档：FP8 W8A8 Quantization](https://docs.vllm.ai/en/latest/features/quantization/fp8/)
