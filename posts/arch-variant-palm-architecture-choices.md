## 核心结论

PaLM 的关键架构选择，不是“发明了一个全新 Transformer”，而是把几处在超大规模训练里最影响吞吐、显存和稳定性的环节同时改了：并行 FFN+Attention、MQA、SwiGLU、RoPE、去偏置、256K 词表。这里的“并行”指同一个 block 里 Attention 和前馈网络 FFN 不再前后串行，而是从同一个归一化后的输入同时计算，再一起加回残差；“MQA”指多查询注意力，只有 Query 仍保留多头，Key/Value 改成各头共享。结果是训练更快，推理 KV Cache 更小，而且基本不牺牲质量。

PaLM 540B 是一个 decoder-only Transformer，也就是只看左侧上下文、按下一个 token 预测来训练的自回归模型。它在 6144 个 TPU v4 芯片上训练，数据规模为 780B tokens。论文里最值得工程师记住的不是“540B”这个数字，而是：当模型已经大到跨 pod 训练时，架构里的每个小低效都会被放大成真实成本。PaLM 的设计本质上是在回答一个问题：怎样让超大模型既能扩展，又不在训练和推理时被系统瓶颈拖垮。

| 模型 | 最高密集参数规模 | 已公开预训练 token |
| --- | --- | --- |
| PaLM (2022) | 540B | 780B |
| PaLM 2 (2023) | 官方报告未公开总参数 | 官方报告未公开总 token |

上表故意把 PaLM 2 留空，因为公开技术报告强调的是 compute-optimal 缩放和数据配比，没有在正文里直接给出外界常引用的总参数与总 token 数。写技术文章时，这类二手数字最好和“一手报告口径”分开。

---

## 问题定义与边界

标准 Transformer block 的常见写法是“先 Attention，再 FFN”。这在中小模型里问题不大，但在超大模型里有两个明显代价。

第一，训练时存在串行依赖。Attention 的输出要先算完，FFN 才能继续，这会让两个大矩阵乘不能很好融合。PaLM 论文明确给出结论：把 block 改成并行形式后，大规模训练速度大约快 15%。这里的“wall-time”就是实际时钟时间，不是理论 FLOPs。

第二，自回归推理必须缓存历史 Key/Value。KV Cache 可以理解成“为后续 token 复用的历史记忆”。标准多头注意力里，每个头都有自己的一组 K/V，head 数越多，缓存和带宽就越线性膨胀。对于长上下文、多 batch、低延迟服务，这不是小优化，而是能不能上线的问题。

玩具例子：假设一个 64 头模型，序列长度 4096，单头维度 128。标准多头注意力要缓存 $2 \times 4096 \times 64 \times 128$ 个数；MQA 只要缓存 $2 \times 4096 \times 1 \times 128$ 个数。理论上，KV 缓存缩小 64 倍。这个结论不依赖 540B 才成立，任何自回归模型都成立。

真实工程例子：如果你做一个代码补全服务，瓶颈通常不是单次前向算不动，而是并发用户一多，KV Cache 吃掉了大部分显存，导致 batch 上不去、首 token 延迟变差。PaLM 选择 MQA，就是在模型质量和服务成本之间，明确偏向“把推理系统做得可扩展”。

| 设计 | 训练路径 | 推理 KV 规模 | 主要影响 |
| --- | --- | --- | --- |
| 串行 block | Attention 后接 FFN | 每个 head 一组 K/V | 更易形成训练串行瓶颈 |
| 并行 block + MQA | Attention 与 FFN 同级计算 | 共享一组 K/V | 训练更快，推理更省缓存 |

---

## 核心机制与推导

PaLM 的并行 block 可以写成：

$$
y = x + \mathrm{Attention}(\mathrm{LN}(x)) + \mathrm{FFN}(\mathrm{LN}(x))
$$

其中 LN 是 LayerNorm，白话说就是把每层输入先做尺度标准化，减少训练发散。这个公式和传统写法的本质区别，不是“多加了一项”，而是把原来串行的两条子路径改成并列。这样做的直接收益是 Attention 和 FFN 的输入矩阵乘可以更好地融合，减少等待链路。

MQA 的推导更直接。标准多头注意力中：

$$
Q \in \mathbb{R}^{T \times h \times d},\quad
K,V \in \mathbb{R}^{T \times h \times d}
$$

这里 $T$ 是序列长度，$h$ 是头数，$d$ 是单头维度。MQA 改成：

$$
Q \in \mathbb{R}^{T \times h \times d},\quad
K,V \in \mathbb{R}^{T \times 1 \times d}
$$

也就是只共享 K/V，不共享 Q。原因很实际：不同 head 继续保留不同的查询视角，但“历史记忆”不再重复存 $h$ 份。于是 KV Cache 从 $O(T h d)$ 下降到 $O(T d)$。对增量解码来说，这相当于把最贵的状态存储项按头数折叠掉。

SwiGLU 是 PaLM 在 FFN 里的门控激活。门控的意思是“不是所有信息都直通，而是先算一个开关强度再放行”。它比传统 ReLU/GeLU 多一组线性变换，但在同等计算预算下质量更好。RoPE 是旋转位置编码，白话说是把位置信息写进向量旋转关系里，而不是简单加一个位置向量。这样在长序列上更稳，也更适合缓存式推理。去偏置则是把 dense 和 LayerNorm 里的 bias 去掉，论文报告它提升了大模型训练稳定性。256K 词表则是为了减少多语言和代码场景里的过度切分，尤其是空白符、Unicode 字节和数字被专门处理，这对源码建模很重要。

---

## 代码实现

下面先用一个纯 Python 的玩具实现，验证 MQA 为什么能把 KV Cache 从“按头增长”改成“常数头数”。

```python
def kv_cache_elements(seq_len: int, num_heads: int, head_dim: int, use_mqa: bool) -> int:
    kv_heads = 1 if use_mqa else num_heads
    # K 和 V 各存一份
    return 2 * seq_len * kv_heads * head_dim

def cache_reduction_ratio(seq_len: int, num_heads: int, head_dim: int) -> int:
    mha = kv_cache_elements(seq_len, num_heads, head_dim, use_mqa=False)
    mqa = kv_cache_elements(seq_len, num_heads, head_dim, use_mqa=True)
    return mha // mqa

assert kv_cache_elements(4096, 64, 128, False) == 2 * 4096 * 64 * 128
assert kv_cache_elements(4096, 64, 128, True) == 2 * 4096 * 1 * 128
assert cache_reduction_ratio(4096, 64, 128) == 64

def palm_parallel_block(x, attention_fn, ffn_fn):
    # x 是标量玩具输入；真实模型里会是向量或张量
    normed = x
    attn = attention_fn(normed)
    ffn = ffn_fn(normed)
    return x + attn + ffn

assert palm_parallel_block(2, lambda t: t * 10, lambda t: t + 3) == 27
```

真实实现时，核心结构通常是下面这样：

```python
def palm_parallel_block(x):
    normed = layer_norm(x)
    attn = attention(normed)      # Q 多头，K/V 共享
    ffn = swiglu_ffn(normed)
    return x + attn + ffn
```

如果把这个结构放到工程里，真正重要的是两点。第一，Attention 和 FFN 都从同一个 `normed` 读输入，所以算子融合空间更大。第二，KV Cache 的接口要按“KV 头数”和“Q 头数”分开设计，否则代码层面很容易又按多头分配了一遍缓存，把 MQA 的收益抵消掉。

---

## 工程权衡与常见坑

PaLM 的这些选择不是“只有收益，没有代价”。

并行 block 的风险在于小模型上未必总是质量完全不变。PaLM 论文给出的消融结论是：8B 规模有轻微退化，62B 没看到质量退化，因此他们推断 540B 可以视为质量中性。这说明一个常见坑：你不能把超大模型的经验，机械搬到 1B 以下模型。

MQA 的风险在于表达能力边界。它通过共享 K/V 省下了大量缓存，但本质上减少了每个头独立存储历史信息的自由度。所以后来的很多模型会选 GQA，也就是分组查询注意力，作为 MHA 和 MQA 之间的折中。

PaLM 2 给出的工程启发则更多落在数据侧。公开技术报告强调三件事：数据和模型应 roughly 1:1 地扩展；多语言混合不应只靠“多抓数据”，还要控制质量；去重和 canary 监测要前置。这里的 canary 可以理解成“人为埋入、用于检测记忆化的标记串”。如果没有这类监测，你很难知道模型是在泛化还是在背数据。

部署侧还有一个常见坑：很多团队只盯模型结构，却忽略推理安全策略。PaLM 2 证明控制 token 可以在推理时显著改变毒性概率，而不用重训整个模型。

| 控制 token | 非毒 prompt 下毒性延续概率 |
| --- | --- |
| 无 token | 0.075 |
| low toxicity | 0.033 |
| medium toxicity | 0.116 |
| high toxicity | 0.203 |

这张表的意义不是“加 token 就安全了”，而是说明部署系统本身也属于模型设计的一部分。真实工程里，一个教育助手、代码助手、搜索问答助手，对安全阈值的要求并不相同。

---

## 替代方案与适用边界

如果你的目标是理解 PaLM 的架构决策，那么一个结论很明确：它解决的是“超大规模 dense decoder-only 模型如何高效训练和可服务化”，不是所有场景的统一最优解。

没有 Pathways 和数千 TPU 时，最可迁移的经验通常只有三条。第一，优先考虑并行 block，因为它改动小、系统收益直接。第二，长上下文推理优先考虑 MQA 或 GQA，因为缓存压力往往先于算力成为瓶颈。第三，如果数据规模跟不上，不要盲目增参，PaLM 2 已经明确表明数据配比和混合质量同样关键。

真实工程上，若你只做 7B 到 30B 的开源部署，常见替代方案是 GQA。它不像 MQA 那样把 K/V 压到 1 组，而是若干头共享 1 组，精度和缓存之间更平衡。若你做的是超长上下文推理，还可能继续配合 KV 分页、量化缓存、分块注意力等系统优化。也就是说，PaLM 的架构选择是“一个很强的基线配方”，但不是终点。

最后需要校正一个常见误区：PaLM 的成功不能简单概括成“参数越大越强”。PaLM 展示了 BIG-bench 上的明显跃迁；PaLM 2 又进一步说明，更好的数据配比、训练目标和部署策略，可以让更小的模型在很多任务上超过前代更大的模型。对初级工程师来说，这比记住具体参数数字更重要。

---

## 参考资料

- Chowdhery et al., “PaLM: Scaling Language Modeling with Pathways”, JMLR 2023: [https://jmlr.org/papers/volume24/22-1144/22-1144.pdf](https://jmlr.org/papers/volume24/22-1144/22-1144.pdf)
- Google Research 页面，PaLM 摘要与项目入口: [https://research.google/pubs/palm-scaling-language-modeling-with-pathways/](https://research.google/pubs/palm-scaling-language-modeling-with-pathways/)
- PaLM 2 Technical Report, Google 2023: [https://ai.google/static/documents/palm2techreport.pdf](https://ai.google/static/documents/palm2techreport.pdf)
- Google Research Blog，PaLM 训练系统与效率说明: [https://research.google/blog/pathways-language-model-palm-scaling-to-540-billion-parameters-for-breakthrough-performance/](https://research.google/blog/pathways-language-model-palm-scaling-to-540-billion-parameters-for-breakthrough-performance/)
