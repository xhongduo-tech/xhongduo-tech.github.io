## 核心结论

推理服务降本的目标，不是把某一条请求“跑得更快”，而是在给定延迟约束下，让同一份 GPU 资源产出更多**可接受 token**。可接受 token 的白话解释是：这些输出不只要生成出来，还要满足质量、格式、工具调用等业务要求，能真正被下游系统使用。

如果把问题压缩成一个公式，可以写成：

$$
\text{成本} \approx \frac{\text{GPU 秒}}{\text{可接受 token}}
$$

这里最容易犯的错，是把“单卡 `tok/s` 很高”误当成“服务成本很低”。前者只描述裸吞吐，后者还受到排队、长上下文占显存、批处理空转、输出质量回退等因素影响。真正接近业务目标的指标是 **goodput**，也就是在 SLO 约束下，系统稳定交付的有效产出。

下面这张表先把几个常见指标放在一起看：

| 指标 | 它描述什么 | 为什么不能单独看 |
| --- | --- | --- |
| 单卡 `tok/s` | 单张卡每秒生成多少 token | 可能忽略排队和长请求拖慢全局 |
| `p95 latency` | 95% 请求在多久内完成 | 只看延迟会忽略资源利用率 |
| 并发承载数 | 同时能挂住多少请求 | 受 KV cache 和碎片强约束 |
| 显存占用 | 模型权重和 KV 是否装得下 | 装得下不等于跑得稳 |
| `goodput` | 在目标延迟内交付的有效 token | 最接近真实成本收益 |

再看一张“错误认知 vs 正确目标”表：

| 错误认知 | 正确目标 |
| --- | --- |
| 只要提升解码速度，成本自然下降 | 需要提升单位 GPU 秒对应的有效产出 |
| 批越大越省钱 | 批要在 `p95 latency` 可控前提下变大 |
| 长上下文只是慢一点 | 长上下文会显著抬高 KV cache，占掉并发 |
| 量化一定免费 | 量化可能引入精度和格式稳定性回退 |

一个真实工程例子是客服机器人。单卡 benchmark 可能显示 `tok/s` 很高，但白天并发一上来，长问题和短问题混在一起，`p95 latency` 迅速恶化。此时 GPU 看似忙碌，实际很多时间消耗在等待批次边界、持有长上下文 KV、处理低价值排队请求上。吞吐数字没有转化为更低成本，因为系统层面的浪费没有解决。

---

## 问题定义与边界

本文讨论的是**在线推理服务**的成本优化，不讨论训练成本，不讨论模型蒸馏本身，也不展开多机容灾和跨地域调度。这里的“成本”主要指部署后持续发生的 GPU 资源成本，包括显存占用导致的并发上限、调度方式导致的空转、以及解码机制导致的每 token 开销。

先统一几个术语：

| 术语 | 定义 | 白话解释 |
| --- | --- | --- |
| GPU 秒 | GPU 被占用的时间总量 | 云账单上最直接的资源消耗 |
| token | 模型处理和生成的最小文本单位 | 可以粗看成“字词碎片” |
| goodput | 在服务目标内完成的有效输出量 | 真正有业务价值的吞吐 |
| 延迟 | 请求从进入到完成的时间 | 用户要等多久 |
| 并发 | 同时在系统中处理的请求数 | 一次能挂多少单 |
| KV cache | 解码时保存历史注意力状态的缓存 | 为了不重复算旧上下文而存下来的中间结果 |

推理服务成本高，通常集中来自三件事：

1. `KV cache` 占显存，而且序列越长，占用越大。
2. 请求调度不合理，GPU 在等待批次、等待慢请求、等待固定轮次结束时空转。
3. 每轮生成的有效 token 太少，导致目标模型被频繁调用。

玩具例子很适合说明边界。假设同时来了两个请求，一个 prompt 长度 200 token，一个 4000 token。如果系统用固定上限做连续内存预留，比如每个请求都按 4096 token 留空间，那么短请求会浪费大块显存。如果系统又采用“静态批处理”，必须等当前整批执行完才能收新请求，那么新请求即使很短，也只能排队。这个例子说明，问题不是单一的“模型慢”，而是**显存管理**和**调度策略**同时失配。

因此本文的边界很明确：讨论的是服务系统层面的降本，也就是如何让同一张卡在可接受质量和延迟下，承载更多请求、交付更多有效 token。

---

## 核心机制与推导

推理服务里最核心的资源约束，通常不是算术单元，而是显存中的 `KV cache`。KV cache 的白话解释是：模型为了继续往后生成，不想每次都把前文重新算一遍，于是把每层注意力需要的历史表示保存下来。保存越多，后续越省重复计算；但保存越多，也越占显存。

单条序列的 KV cache 显存可写为：

$$
M_{kv} = T \cdot L \cdot 2 \cdot n_{kv} \cdot d_h \cdot b
$$

其中：

- $T$ 是当前序列长度
- $L$ 是层数
- $n_{kv}$ 是 KV heads 数
- $d_h$ 是每个 head 的维度
- $b$ 是每个元素的字节数
- 系数 2 表示同时存 K 和 V

用一个玩具数值例子代入：

- $L = 32$
- $n_{kv} = 32$
- $d_h = 128$
- $b = 2$ 字节，表示 FP16/BF16 量级
- $T = 2048$

则：

$$
M_{kv} = 2048 \cdot 32 \cdot 2 \cdot 32 \cdot 128 \cdot 2
$$

约等于：

$$
1{,}073{,}741{,}824 \text{ bytes} \approx 1.0 \text{ GiB}
$$

这意味着，一条 2048 token 的序列，光 KV cache 就大约吃掉 1 GiB 显存。接下来单卡并发上界可以粗略写为：

$$
N_{max} \approx \left\lfloor \frac{M_{gpu} - M_{model} - M_{ovh}}{M_{kv}} \right\rfloor
$$

这里：

- $M_{gpu}$ 是总显存
- $M_{model}$ 是模型权重占用
- $M_{ovh}$ 是运行时额外开销，例如 workspace、框架缓存、调度结构

如果一张 80GB GPU 上，模型权重占 40GB，其他开销先忽略为小量，那么理论上还能放约 40 条这样的序列。但这是理想值，真实工程里通常达不到，因为还会遇到**连续分配浪费**和**内存碎片**。

假设系统为每个请求按最大长度 $A$ 连续预留显存，而真实长度是 $T$，浪费比例是：

$$
w_{contig} = \frac{A - T}{A}
$$

如果 $A=2048$，但平均真实长度只有 $T=1300$，则：

$$
w_{contig} = \frac{2048 - 1300}{2048} \approx 36.5\%
$$

这不是小损耗，而是会直接吞掉本来能承载的并发。

Paged KV cache 的思路是把 KV 按页管理，不要求一整段连续。若分页块大小为 $B$，那么浪费上界可写为：

$$
w_{page} \le \frac{B - 1}{T}
$$

比如 $B=16$，$T=1300$，则：

$$
w_{page} \le \frac{15}{1300} \approx 1.2\%
$$

这说明 paged KV 主要解决的是**显存碎片和不规则长度下的浪费**，不是直接减少 attention 的数学计算量。

从公式往工程结论推，可以得到一个很重要的判断：单卡并发上界，先被 KV cache 的长度增长压低，再被内存分配策略进一步拉低。也就是说，长上下文不只是“更慢”，而是会让系统容纳请求的能力下降，进而抬高排队延迟和单 token 成本。

四类常见优化可以按它们主要作用的资源位置来理解：

| 优化手段 | 主要解决什么 | 资源作用点 |
| --- | --- | --- |
| `continuous batching` | 减少调度空转 | 调度 |
| `paged KV cache` | 减少显存碎片 | 存储 |
| `speculative decoding` | 减少目标模型调用次数 | 生成步数 |
| `quantization` | 减少显存和带宽压力 | 权重与访存 |

真实工程例子可以看企业知识库问答。白天高并发，问题长度波动大，有的请求只问一句短定义，有的会塞入多个检索片段。此时如果没有 continuous batching，短请求会被长请求绑定在静态批里一起等待；如果没有 paged KV，短请求又会因为预留长上下文而浪费显存。两者叠加后，系统看似 GPU 利用率不低，但 goodput 不高，因为很多资源没有转化为按时交付的结果。

---

## 代码实现

实现上不需要先写完整生产系统，先抓三个最小闭环就够了：请求调度、KV 分配、生成执行。对应的数据结构可以简化为：

| 组件 | 责任 |
| --- | --- |
| `request_queue` | 接收新请求，保存等待态 |
| `batch_scheduler` | 每个解码 step 重组 batch |
| `kv_allocator` | 按页分配和释放 KV |
| `decoder` | 执行 prefill 和 decode |
| `result_committer` | 提交输出并结束请求 |

下面先给一个可运行的 Python 玩具例子，用来验证上面的 KV 估算和分页浪费结论：

```python
from math import floor

def kv_bytes(T, L, n_kv, d_h, b):
    return T * L * 2 * n_kv * d_h * b

def gib(x):
    return x / (1024 ** 3)

def max_concurrency(M_gpu_gib, M_model_gib, M_ovh_gib, per_seq_gib):
    usable = M_gpu_gib - M_model_gib - M_ovh_gib
    return floor(usable / per_seq_gib)

def contiguous_waste(A, T):
    return (A - T) / A

def paged_waste_upper(B, T):
    return (B - 1) / T

per_seq = kv_bytes(T=2048, L=32, n_kv=32, d_h=128, b=2)
per_seq_gib = gib(per_seq)

assert round(per_seq_gib, 2) == 1.00

n = max_concurrency(M_gpu_gib=80, M_model_gib=40, M_ovh_gib=0, per_seq_gib=per_seq_gib)
assert n == 40

w1 = contiguous_waste(A=2048, T=1300)
w2 = paged_waste_upper(B=16, T=1300)

assert round(w1, 3) == 0.365
assert round(w2, 3) == 0.012
assert w2 < w1

print("per_seq_gib =", round(per_seq_gib, 3))
print("max_concurrency =", n)
print("contiguous_waste =", round(w1, 4))
print("paged_waste_upper =", round(w2, 4))
```

这个代码没有实现推理，但它把“为什么 paged KV 能把并发拉回去”这个结论变成了可验证的数值。

接着看一个简化版的 `continuous batching + paged KV` 主循环伪代码：

```python
def serve_loop():
    while True:
        new_reqs = dequeue_arrivals()
        for req in new_reqs:
            request_queue.append(req)

        active_batch = schedule_batch(
            waiting=request_queue,
            running=running_requests,
            token_budget=max_batch_tokens,
            memory_budget=free_kv_pages(),
        )

        for req in active_batch.new_prefill:
            req.pages = allocate_kv_pages(req.prompt_len)
            prefill(req)

        for req in active_batch.decode_set:
            next_token = decode_step(req)
            append_token(req, next_token)

            if req.finished:
                release_kv(req.pages)
                finalize(req)
```

这里最关键的点不是函数名，而是“**每个解码 step 重新组 batch**”。静态批处理的做法是把一整批固定住，必须等最慢请求拖完；continuous batching 则允许每轮都把新请求插进来，让 GPU 尽量持续有活干。

再看 speculative decoding 的“提案-验证”流程：

```python
def speculative_decode(req, draft_model, target_model, k):
    proposal = draft_model.generate(req.state, max_tokens=k)
    accepted = 0

    for token in proposal:
        ok = target_model.verify_next(req.state, token)
        if not ok:
            break
        commit(req, token)
        accepted += 1

    if accepted < len(proposal):
        true_token = target_model.decode_one(req.state)
        commit(req, true_token)

    return accepted
```

这个机制的本质是：让一个更便宜的小模型先提案，再让大模型批量验证。如果接受率高，大模型等于“少走了几步”；如果接受率低，大模型还是要回退补算，收益就会消失。

一个真实工程映射可以这样理解：

- `enqueue_request()`：接网关请求，带上 prompt 长度、优先级、SLO。
- `schedule_batch()`：依据 token budget 和显存余量，决定谁能进入当前步。
- `allocate_kv_pages()`：为请求按页申请 KV，而不是整段连续申请。
- `decode_step()`：执行一轮解码，支持普通解码或 speculative 验证。
- `release_kv()`：请求结束后及时释放页，避免“死占显存”。

---

## 工程权衡与常见坑

推理服务降本不能只看一个指标。工程里至少要同时盯住 `p50/p95 latency`、goodput、显存水位、GPU 利用率，以及某些特定机制的内部指标，例如 speculative decoding 的接受率。

先给一个指标优先级表：

| 指标 | 优先级 | 原因 |
| --- | --- | --- |
| `p95 latency` | 高 | 直接决定服务是否违约 |
| `goodput` | 高 | 直接反映成本是否真的下降 |
| 显存水位 | 高 | 决定并发上限和 OOM 风险 |
| GPU 利用率 | 中 | 高利用率不一定代表高价值产出 |
| 接受率 | 中 | speculative decoding 是否成立的关键 |

常见坑可以总结成下面这张表：

| 坑 | 为什么会错 | 规避策略 |
| --- | --- | --- |
| 只看 `tok/s` | 吞吐高不等于排队少 | 联合看 `p95 latency` 和 goodput |
| 把量化当成免费降本 | 可能损伤格式稳定性和任务正确率 | 做任务级回归，不只跑 perplexity |
| 误以为 `paged KV` 能减少计算量 | 它主要减少碎片，不直接减少 attention FLOPs | 把收益预期放在并发和显存上 |
| 忽略 workload 形态 | 不同请求长度分布决定优化收益 | 先做流量画像再选方案 |

这里最值得展开的是 speculative decoding 的失败模式。它听起来非常合理：先让小模型猜，再让大模型确认，似乎一定更省。但真实系统里，只有当 draft model 足够快、而且提案 token 的接受率足够高时，这条链路才成立。否则你付出了 draft 模型的额外执行成本，又没有换来足够多的大模型步数减少，最终可能比直接解码还慢。

这就是“看起来更聪明的方法反而失败”的典型例子。原因不神秘，还是资源账没算清：任何优化都不是免费插件，而是把开销从一个位置搬到另一个位置。

可以用一个简化决策树来判断：

| 判断问题 | 如果答案是“是” | 倾向方案 |
| --- | --- | --- |
| 是否 memory-bound | 长上下文把显存压满 | 优先 `paged KV`、量化 |
| 上下文长度是否波动大 | 长短请求混杂严重 | 优先 `paged KV` + `continuous batching` |
| 是否高 QPS 批量请求 | GPU 常常有足够待处理请求 | 优先调度优化 |
| draft 是否足够快且接受率高 | 验证成本能被摊薄 | 再考虑 speculative decoding |

工程上还有一个常见误判：把“GPU 利用率很高”理解为“优化做得很好”。这并不成立。GPU 也可能在高利用率地执行低价值工作，例如长请求拖住整批、无效 padding、频繁回退验证。这些都会让资源看起来很忙，但单位 GPU 秒交付的可接受 token 并不高。

---

## 替代方案与适用边界

没有一种优化能覆盖所有 workload。判断是否值得做某项优化，核心不是“它先进不先进”，而是“它解决的是否正好是当前瓶颈”。

先看三类常见场景：

| 场景 | 特征 | 更合适的方案 |
| --- | --- | --- |
| 长上下文、高波动客服系统 | prompt 长度差异大，白天并发高 | `continuous batching + paged KV + quantization` |
| 固定模板、短输出摘要服务 | 输入输出都较短，QPS 高 | 优先调度优化、模型裁剪、合理批处理 |
| 强工具调用的 agent 服务 | 输出格式严格，回退成本高 | 谨慎使用 speculative decoding，先保一致性 |

第一类是真实工程里最常见的降本主战场。因为长上下文本身就让 KV cache 成本很高，而长度波动又让连续分配浪费严重，所以 paged KV 的收益通常非常直接。再叠加 continuous batching，能把长短请求混合流量下的空转进一步压低。

第二类场景则不同。短上下文、高 QPS、输出固定时，系统可能更接近 compute-bound，也就是主要卡在解码算力而不是显存。这时一味追求 paged KV，收益可能有限；更实用的往往是模型裁剪、服务调度和批尺寸控制。

第三类是 agent 服务。它常常带工具调用、JSON 格式约束、函数参数严格性等要求。这里“可接受 token”的标准比普通聊天更苛刻，哪怕看起来吞吐更高，如果输出格式更不稳定，也会让重试率升高，最终把成本吃回去。因此 speculative decoding 要更谨慎，因为它对接受率和输出一致性的要求更高。

再给一张方案适配矩阵：

| 优化手段 | 长上下文 | 高波动长度 | 高 QPS | 低延迟强一致性 |
| --- | --- | --- | --- | --- |
| `continuous batching` | 适合 | 适合 | 很适合 | 适合 |
| `paged KV cache` | 很适合 | 很适合 | 一般 | 适合 |
| `quantization` | 适合 | 适合 | 适合 | 需验证质量 |
| `speculative decoding` | 视接受率而定 | 视接受率而定 | 适合部分场景 | 谨慎 |

边界也要说清：

- 量化不解决糟糕的调度策略。
- speculative decoding 不解决显存碎片。
- 调度优化不直接减少模型权重占用。
- paged KV 不直接减少 attention 本身的计算复杂度。

所以更准确的做法不是“选一个最强技术”，而是先识别当前系统到底是 **memory-bound** 还是 **compute-bound**。白话解释是：到底是显存先不够，还是计算先跑不动。这个判断一旦错了，优化方向大概率也会错。

---

## 参考资料

1. [PagedAttention: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://huggingface.co/papers/2309.06180)
2. [ORCA: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
3. [vLLM Speculative Decoding 文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
4. [Speculative Decoding: Fast Inference from Transformers via Speculative Decoding](https://huggingface.co/papers/2211.17192)
5. [TensorRT-LLM Quantization 文档](https://nvidia.github.io/TensorRT-LLM/torch/features/quantization.html)
