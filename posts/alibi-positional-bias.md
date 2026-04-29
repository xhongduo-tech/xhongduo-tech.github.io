## 核心结论

ALiBi（Attention with Linear Biases，在线性偏置下做注意力）不是把“位置向量”加到 token 表示里，而是直接在注意力分数上加入一个“距离越远、惩罚越大”的线性项。白话说，它不是教模型记住“第 37 个位置长什么样”，而是直接规定“越远的内容默认越不重要，除非内容分数足够强把这个惩罚抵消掉”。

它最有价值的地方不是增加模型容量，而是改善长度外推。长度外推指模型训练时只见过较短序列，推理时却要处理更长序列。传统绝对位置嵌入通常把“位置”做成一张有限表，训练到 2k 长度，推理突然喂 8k，模型很容易在超出训练长度的位置上失稳；ALiBi 不依赖这张表，而是用一条固定的距离规则继续工作，所以改造成本低，外推更自然。

对 decoder-only 因果自注意力，核心公式是：

$$
e_{ij}=\frac{q_i k_j^\top}{\sqrt{d_h}}-m_h(i-j),\quad j\le i
$$

$$
a_{ij}=\mathrm{softmax}_j(e_{ij})
$$

其中 $d_h$ 是每个注意力头的维度，$m_h$ 是第 $h$ 个 head 的固定斜率。斜率可以理解为“这个 head 对远距离信息有多不耐烦”。

下面这张表先把它和最常见的绝对位置嵌入放在一起看：

| 方案 | 位置如何进入模型 | 是否依赖位置参数表 | 超训练长度时是否自然可用 | 改造成本 |
|---|---|---:|---:|---:|
| 绝对位置嵌入 | 给 token 表示加位置向量 | 是 | 通常较差 | 低 |
| ALiBi | 在 attention logits 上加线性距离偏置 | 否 | 通常较好 | 很低 |

玩具例子很直接：训练时模型只看过 2k token，线上要接 8k token。绝对位置嵌入常常在 2k 之后没有可靠的“位置经验”；ALiBi 只要还能算出距离 $i-j$，就能继续加偏置，不需要为新位置单独学习参数。

---

## 问题定义与边界

先把问题说清楚。Transformer 的自注意力本身只处理“谁和谁相关”，但如果没有额外机制，它并不知道两个 token 的先后和距离。位置编码的作用就是把“顺序”和“距离”塞进模型里。

难点出在“训练短，测试长”。假设模型训练时上下文长度是 1024，推理时输入变成 4096。这里至少有两层风险：

1. 模型没学过这么远的依赖关系。
2. 如果位置表示本身是有上界的，超出上界后甚至没有对应表示。

对 learned absolute embedding（可学习绝对位置嵌入）来说，第二条尤其明显：位置 1025 可能根本没有现成向量。对 sinusoidal 这类可计算位置编码来说，虽然位置值能继续算，但模型是否能稳定泛化到更长区间，依然不保证。ALiBi 试图解决的是这个“长度外推时的位置机制不稳定”问题。

在因果语言模型里，第 $i$ 个位置只能看见不超过它自己的历史位置，所以距离一般写成 $i-j$，其中 $j\le i$。白话说，当前 token 看前面第几个 token，距离就是“隔了多少步”。

边界也要划清。ALiBi 不是“所有位置编码问题的通解”，它最自然的场景是 decoder-only causal self-attention。因为它最初就是为这种结构设计的。下面这张边界表更实用：

| 场景 | 是否适合直接使用 ALiBi | 说明 |
|---|---|---|
| decoder-only causal LM | 是 | 原始设计目标，最自然 |
| encoder-only 双向注意力 | 需改造 | 一般要基于 $|i-j|$ 或重新设计偏置 |
| encoder-decoder cross-attention | 需改造 | query 和 key 不在同一时间轴 |
| 强依赖绝对位置的任务 | 不宜直接照搬 | ALiBi 更强调相对距离，不显式表达“绝对第几位” |

真实工程例子是代码助手或长文生成。训练时为了省算力，可能只用 2k 或 4k 上下文；上线后用户会粘贴 16k 甚至 32k 的代码、日志、文档。如果位置机制依赖训练长度内的位置表，超长推理就容易退化。ALiBi 的吸引力就在这里：它不是承诺“长上下文下更强”，而是承诺“用更少的机制改动，让长上下文更不容易崩”。

---

## 核心机制与推导

理解 ALiBi，要把注意力拆成三件事：内容分数、距离偏置、softmax 重分配。

第一步，原始内容分数：

$$
s_{ij}=\frac{q_i k_j^\top}{\sqrt{d_h}}
$$

这是标准缩放点积注意力，表示“位置 $i$ 对位置 $j$ 的内容相关性”。

第二步，加上线性距离偏置：

$$
e_{ij}=s_{ij}-m_h(i-j)
$$

这里的 $m_h(i-j)$ 是惩罚项。距离越远，减得越多。于是模型就被先验地推向“更关注近邻”。

第三步，做 softmax：

$$
a_{ij}=\frac{\exp(e_{ij})}{\sum_{t\le i}\exp(e_{it})}
$$

softmax 可以理解为“把一排分数变成概率分布”。因为远位置被减分，近位置通常会在归一化后获得更高权重。

一个最小玩具例子：

- 某个 query 对 4 个历史 token 的内容分数是 `[2.0, 1.0, 0.0, -1.0]`
- 当前 query 在位置 `i=4`
- 对应 key 在位置 `j=1,2,3,4`
- 设该 head 的斜率 $m_h=0.5$

那么距离分别是 `[3,2,1,0]`，偏置是 `[-1.5,-1.0,-0.5,0]`，最终 logits 变成：

| 距离 $i-j$ | 内容分数 | ALiBi 偏置 | 最终 logits | softmax 后趋势 |
|---:|---:|---:|---:|---|
| 3 | 2.0 | -1.5 | 0.5 | 降低 |
| 2 | 1.0 | -1.0 | 0.0 | 中等 |
| 1 | 0.0 | -0.5 | -0.5 | 偏低 |
| 0 | -1.0 | 0.0 | -1.0 | 最低 |

这个表有个细节值得强调：ALiBi 不是简单保证“最近 token 权重最大”，而是让“远距离需要更高内容分数才能赢”。在上面的例子里，最远位置原始内容分数最高，所以即使被惩罚，最后仍可能保留较大权重。这正是 ALiBi 的合理性：距离是先验，不是硬约束。

再往前一步，为什么要给不同 head 不同斜率 $m_h$？因为多头注意力本来就希望不同 head 关注不同模式。ALiBi 让一部分 head 有更陡的斜率，强烈偏好局部信息；另一部分 head 斜率更平，保留远距离通路。于是模型形成“多尺度距离偏好”。

可以把流程压缩成四步：

1. 先算原始内容相似度 $qk^\top$
2. 再扣掉距离惩罚 $m_h(i-j)$
3. 然后做 softmax
4. 最终让近距离更容易被分到权重，但远距离仍可凭内容胜出

如果换成双向注意力，通常不能直接照搬上面的因果版本。原因是双向注意力里既能看左边也能看右边，距离更自然写成 $|i-j|$。但这只是思路，不等于“把公式里改个绝对值就一定正确”，因为 encoder、cross-attention 的时间轴和语义边界都不同。

---

## 代码实现

工程上接入 ALiBi 的关键点很少，难点不在代码量，而在插入位置要对。最小改法通常是：

1. 去掉绝对位置嵌入
2. 为每个 head 生成一个固定斜率
3. 构造距离偏置矩阵
4. 在 `qk^T` 之后、`softmax` 之前把偏置加到 logits 上
5. 再与 causal mask 一起生效

最小张量维度通常如下：

| 张量 | 形状 | 含义 |
|---|---|---|
| `q` | `[B, H, Lq, Dh]` | query |
| `k` | `[B, H, Lk, Dh]` | key |
| `scores` | `[B, H, Lq, Lk]` | 原始注意力分数 |
| `bias` | `[H, Lq, Lk]` | ALiBi 偏置 |

下面给出一个可运行的 Python 玩具实现，直接展示 logits、mask 和偏置怎么组合。代码不依赖深度学习框架，方便先验证机制：

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

def build_alibi_bias(num_heads, q_len, k_len, slopes):
    bias = []
    for h in range(num_heads):
        head_bias = []
        m_h = slopes[h]
        for i in range(q_len):
            row = []
            for j in range(k_len):
                if j <= i:
                    row.append(-m_h * (i - j))
                else:
                    row.append(float("-inf"))  # causal mask
            head_bias.append(row)
        bias.append(head_bias)
    return bias

def attention_row(content_scores, bias_row):
    logits = [s + b for s, b in zip(content_scores, bias_row)]
    valid_logits = [x for x in logits if x != float("-inf")]
    probs = softmax(valid_logits)
    return logits, probs

# 玩具例子：单个 head，第 4 个位置看前 4 个位置
content_scores = [2.0, 1.0, 0.0, -1.0]
slopes = [0.5]
bias = build_alibi_bias(num_heads=1, q_len=4, k_len=4, slopes=slopes)

# 取最后一行，对应 i=3（从 0 开始）
logits, probs = attention_row(content_scores, bias[0][3])

expected_logits = [0.5, 0.0, -0.5, -1.0]
for a, b in zip(logits, expected_logits):
    assert abs(a - b) < 1e-9

assert probs[0] > probs[1] > probs[2] > probs[3]
assert abs(sum(probs) - 1.0) < 1e-9

print("logits:", logits)
print("probs:", [round(x, 3) for x in probs])
```

如果接入现有 Transformer，伪代码通常就是下面这样：

```python
scores = (q @ k.transpose(-2, -1)) / sqrt(dh)
scores = scores + alibi_bias
scores = scores.masked_fill(causal_mask == 0, -inf)
attn = softmax(scores, dim=-1)
out = attn @ v
```

这里有两个实现细节不能搞反。

第一，ALiBi 偏置是加到 logits 上，不是加到 token embedding 上。它作用于“注意力打分”，不作用于“词向量表示”。

第二，mask 与 bias 的顺序要保证因果约束不被破坏。最安全的方式是把未来位置直接填成 `-inf`，这样 softmax 后概率严格为 0。ALiBi 只是重排可见历史位置之间的相对分数，不能替代 causal mask。

真实工程例子是把一个已有 decoder-only 语言模型改造成 ALiBi 版本。常见操作是删除 learned positional embedding，保留 token embedding 和原注意力层结构，只在每层 attention forward 里额外注入一个可广播的 `[H, Lq, Lk]` 偏置。这样改动小，不需要重写整个模型架构。

---

## 工程权衡与常见坑

ALiBi 的优势很明确：简单、参数少、外推友好、改造成本低。它最大的代价也同样明确：它不显式表示绝对位置。如果任务对“第几位”本身非常敏感，ALiBi 可能不是最佳选择。

一个经常被忽略的权衡是：训练长度可以缩短，但不能短到脱离任务分布。比如业务真实请求常常有 8k 上下文，而训练永远只给 512，模型即使有 ALiBi，也未必能稳定学会长程依赖。ALiBi 解决的是“位置机制外推更自然”，不是“数据分布差异自动消失”。

下面这张“坑与规避”表最常见：

| 错误理解 | 真实风险 | 规避方式 |
|---|---|---|
| 把 ALiBi 当成通用位置编码最优解 | 在需要绝对位置信息的任务上退化 | 先确认任务更依赖相对距离还是绝对位置 |
| 同时保留强 absolute embedding 再指望 ALiBi 外推 | 模型仍可能依赖训练长度内的位置表 | 外推主路径上避免强绝对位置依赖 |
| 只改 attention，不检查 mask 逻辑 | 未来 token 泄漏，训练目标被破坏 | 明确区分 bias 和 causal mask |
| 训练长度太短却期待超长泛化 | 长程依赖学不出来 | 训练长度至少覆盖核心依赖跨度 |
| 不统一长上下文评测口径 | 困惑度对比失真 | 明确训练长度、测试长度、评测窗口 |

这里再给一个真实工程坑。做长上下文评测时，如果一个实验按 `sliding window` 评测，另一个按 `non-overlapping` 评测，困惑度通常不能直接横比。因为前者给了更多历史上下文，后者更苛刻。很多“ALiBi 明显更好”或“收益不明显”的结论，问题不在模型，而在评测口径没对齐。

另一个常见误区是把“训练短，测试长”理解成只和位置编码有关。实际上它还和优化、数据长度分布、KV cache 策略、注意力实现细节有关。ALiBi 只是这一链条中最容易改、回报也常常不错的一环。

---

## 替代方案与适用边界

把 ALiBi 放回位置编码家族里看，会更容易做选择。它不是替代所有方案，而是在“想要长上下文泛化、又不想引入复杂机制”的条件下很有吸引力。

| 方法 | 是否显式位置参数 | 是否容易外推 | 实现复杂度 | 适用架构 |
|---|---:|---:|---:|---|
| Learned Absolute Embedding | 是 | 较差 | 低 | 通用，但超长不稳 |
| Sinusoidal | 否 | 一般 | 低 | 通用 |
| RoPE | 否 | 较好 | 中 | 现代 LLM 常用 |
| Transformer-XL 类相对位置 | 通常有专门机制 | 较好 | 较高 | 相对位置建模较强 |
| Position Interpolation | 依赖原方案 | 作为补救可用 | 中 | 常用于延长已有模型上下文 |
| ALiBi | 否 | 较好 | 很低 | 特别适合 decoder-only causal LM |

可以把选择逻辑压缩成三类：

| 结论 | 场景 |
|---|---|
| 推荐使用 | decoder-only、长文生成、代码补全、日志分析、希望低成本做长度外推 |
| 需要改造后使用 | encoder-only、encoder-decoder、cross-attention |
| 不推荐直接使用 | 强依赖绝对位置、需要精细双向位置结构、已有成熟 RoPE 生态且不想改训练范式 |

新手版对比可以记成三句话：

- learned absolute embedding：位置靠表，超出训练长度后最容易出硬缺口。
- sinusoidal：位置能继续算，但“能算”不等于“外推稳定”。
- ALiBi：不学位置表，直接把距离惩罚写进注意力分数。

如果你的模型是代码助手、长文续写、对话生成这类 decoder-only 系统，ALiBi 很适合做一个“低摩擦升级”。如果你的模型是 encoder-only 检索编码器、分类器，或者 encoder-decoder 翻译模型，直接照搬 causal ALiBi 就不严谨了，至少要先重新定义距离偏置如何作用在对应注意力结构里。

---

## 参考资料

1. [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://openreview.net/forum?id=R8sQPpGCv0)  
   论文原文，给出 ALiBi 的方法定义、实验设置与“训练短、测试长”的核心论证。

2. [attention_with_linear_biases](https://github.com/ofirpress/attention_with_linear_biases)  
   原始代码仓库 README，适合理解作者给出的最小实现思路与斜率设计方式。

3. [MosaicML Composer ALiBi 文档](https://docs.mosaicml.com/projects/composer/en/v0.3.1/method_cards/alibi.html)  
   工程接入说明，重点是如何把 ALiBi 作为训练管线中的一种改造方法使用。

4. [Hugging Face Falcon 文档](https://huggingface.co/docs/transformers/model_doc/falcon)  
   真实模型文档，可用来确认 ALiBi 已进入实际大模型实现，而不只是论文概念。

5. [MosaicML LLM Foundry README](https://github.com/mosaicml/llm-foundry)  
   工程化训练框架与模型实践参考，适合看 ALiBi 在长上下文语言模型中的落地方式。
