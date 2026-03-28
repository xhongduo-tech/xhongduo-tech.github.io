## 核心结论

ALiBi 是一种把“相对距离惩罚”直接加到注意力分数里的位置方法。注意力分数指 softmax 之前的 logits，也就是模型决定“看谁更多一点”的原始分值。它不再给 token 额外加位置向量，而是在每个注意力头上加入固定线性偏置：

$$
\text{score}_{h}(i,j)=\frac{Q_iK_j^\top}{\sqrt{d_k}}-m_h\cdot |i-j|
$$

这里 $i,j$ 是两个 token 的位置，$m_h$ 是第 $h$ 个注意力头的固定斜率。白话说，距离越远，先扣分，再做 softmax。

这件事的重要性在于：模型训练时只见过 1024 token，不代表推理时也必须卡死在 1024。ALiBi 把“位置”改写成“距离惩罚”后，模型学到的是一个更稳定的规则：近处通常更重要，远处通常更难直接依赖。这个规则和序列绝对长度无关，所以更容易从训练长度外推到更长长度。

一个简短结论是：如果你的目标是“低改动、低成本、可直接外推到 2K 左右”，ALiBi 是最简单有效的位置方案之一；如果你的目标是 32K、64K、128K 这种超长上下文，ALiBi 通常不是最强选择。

| 方法 | 位置表示放在哪里 | 是否新增可学习参数 | 长度外推能力 | 实现复杂度 |
| --- | --- | --- | --- | --- |
| 绝对位置编码 / sin-cos | 输入 embedding 侧 | 通常有或隐式固定 | 弱到中 | 低 |
| ALiBi | attention logits 侧 | 无 | 中，2K 附近很实用 | 很低 |
| RoPE | Q/K 旋转变换 | 无 | 原生外推有限，常需缩放/插值 | 中 |

玩具例子：假设一句话只有 6 个 token，当前 token 想看前面的信息。传统位置编码会告诉模型“这是第 1、2、3、4、5、6 个位置”。ALiBi 不强调绝对编号，只强调“离我近还是远”。这意味着无论句子长度是 6 还是 600，只要“附近 token 往往更相关”这个规律成立，模型就更容易迁移。

---

## 问题定义与边界

问题本质不是“模型能不能处理长输入”，而是“模型有没有学到一种可外推的位置规律”。外推，白话说，就是训练时没见过这么长，推理时还能正常工作。

典型场景如下：

| 训练长度 | 推理长度 | 做法 | 结果风险 |
| --- | --- | --- | --- |
| 1024 | 1024 | 常规训练 | 风险最低 |
| 1024 | 2048 | 直接外推 | 位置方法决定成败 |
| 1024 | 8192 | 强行外推 | 多数方法明显退化 |
| 4096 | 4096 | 直接训练更长 | 成本显著增加 |

为什么训练更长会贵？因为自注意力复杂度和序列长度近似平方相关，即 $O(n^2)$。从 1024 拉到 2048，不只是“多一倍 token”，而是 attention 相关算子接近“四倍工作量”。所以工程上常见诉求不是“无限长”，而是“训练先短一点，线上先能顶住 2K 或 4K”。

ALiBi 的边界也要说清楚。它编码的是一种非常强的先验：越远越不该被轻易关注。这对语言建模通常有效，因为局部语法、最近实体、最近约束常常最重要。但当任务要求在超长上下文里精准捞一个很远的关键信息时，这个先验可能反而过强。

真实工程例子：你训练一个内部文档问答模型，训练窗口只有 1024。上线后用户贴进来 1800 token 的技术文档，希望模型仍然能理解前后约束。ALiBi 常常能比普通绝对位置编码更稳定，因为它没有把“1025 之后的位置”当成未见过的新世界。但如果用户塞进 50K token 的日志并问“第 41000 行那个错误码和第 900 行的配置有什么关系”，ALiBi 通常不够强。

---

## 核心机制与推导

标准自注意力先算内容相关性：

$$
S(i,j)=\frac{Q_iK_j^\top}{\sqrt{d_k}}
$$

ALiBi 把它改成：

$$
S_h(i,j)=\frac{Q_iK_j^\top}{\sqrt{d_k}}-m_h\cdot |i-j|
$$

其中注意力头是 multi-head attention 里的一个子通道，可以理解为“不同头看不同类型关系”。ALiBi 给每个头分配不同斜率 $m_h$，常用等比序列，让有的头强烈偏好局部，有的头只轻微惩罚长距。这样一组头合起来，既保留局部建模能力，也不完全放弃远距离依赖。

一个最小数值例子：

- 设某个头的斜率 $m_h=0.5$
- 当前 query 位置 $i=5$
- key 位置 $j=9$
- 距离为 $|5-9|=4$

则偏置为：

$$
-0.5 \times 4 = -2
$$

如果原始 logits 是 3.1，那么加完 ALiBi 后变成 1.1；如果另一个更近的 token 原始 logits 也是 3.1，但距离只有 1，则只会减成 2.6。softmax 后，近处 token 的注意力权重会显著更大。

这解释了 ALiBi 为什么容易外推。它没有把“第 1537 个位置”编码成某个训练时未充分覆盖的绝对向量，而是继续沿用同一条规则：距离从 100 变到 101，只是再多扣一点分。规则没变，所以长度扩展时分布变化更平滑。

可以把不同头理解成不同“视野”：

| Head | 斜率 $m_h$ | 偏好 |
| --- | --- | --- |
| Head 1 | 大 | 强局部，几乎只看附近 |
| Head 2 | 中 | 兼顾局部与中距 |
| Head 3 | 小 | 可以保留较长依赖 |
| Head 4 | 更小 | 最愿意看远处 |

论文和官方实现里，斜率不是学习出来的，而是固定初始化，常按 2 的幂附近的等比形式生成。这样做的好处是稳定、简单、无额外训练参数，也不会引入新的位置 embedding 表。

---

## 代码实现

工程实现非常直接。核心不是改 Transformer 结构，而是在 attention score 上加一个可广播的 bias 矩阵。

```python
import math

def build_alibi_bias(seq_len: int, slope: float):
    bias = []
    for i in range(seq_len):
        row = []
        for j in range(seq_len):
            row.append(-slope * abs(i - j))
        bias.append(row)
    return bias

def add_scores(raw_scores, bias):
    out = []
    for r, b in zip(raw_scores, bias):
        out.append([x + y for x, y in zip(r, b)])
    return out

seq_len = 5
slope = 0.5
bias = build_alibi_bias(seq_len, slope)

assert bias[2][2] == 0.0
assert bias[0][4] == -2.0
assert bias[4][0] == -2.0

raw_scores = [
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 3.0, 1.0, 3.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],
]

new_scores = add_scores(raw_scores, bias)

# 对位置 2 来说，看自己不扣分，看位置 4 会扣 1 分
assert new_scores[2][2] == 3.0
assert new_scores[2][4] == 2.0

print("ALiBi bias works.")
```

在真实模型里，通常写成下面这种形态：

```python
scores = (Q @ K.transpose(-2, -1)) / math.sqrt(dk)
scores = scores + alibi_bias + causal_mask
attn = softmax(scores, dim=-1)
```

其中 `alibi_bias` 的形状通常可广播为 `[1, num_heads, seq_len, seq_len]`。它是固定张量，不需要梯度。

一个常见初始化思路如下：

| head | $m_h$ 示例 |
| --- | --- |
| 0 | 1.0 |
| 1 | 0.5 |
| 2 | 0.25 |
| 3 | 0.125 |

真实实现一般会用更细致的等比公式，而不是手填这四个数，但思想一致：斜率越大，越偏局部。

---

## 工程权衡与常见坑

ALiBi 的工程价值主要来自三点。

第一，实现成本低。它不改 token embedding，不引入新参数，不需要复杂插值逻辑。对已有 decoder-only Transformer，通常只要删掉位置 embedding，并在 attention logits 上加 bias。

第二，训练和显存更省。原始论文报告了一个很实用的结果：1.3B 模型在 1024 token 上训练，推理到 2048 时，能达到与“用 sin/cos 并直接训练 2048”相当的 perplexity，同时训练更快、显存更省，量级约 11%。

第三，短到中距离依赖通常足够强。很多生成任务真正依赖的是最近几百个 token，而不是超远位置。ALiBi 的“近处优先”先验和这类任务相匹配。

但坑也非常明确。

| 优点 | 代价 |
| --- | --- |
| 外推到 2K 左右很省事 | 对极长距离依赖惩罚可能过重 |
| 无位置参数，改动小 | 超 8K 往往不如专门长上下文方法 |
| 训练速度和内存友好 | 对 needle-in-haystack 类任务不占优 |

needle-in-haystack 的白话解释是：在很长文本里找一根针，即从海量无关上下文中精确找到一个远处关键信息。ALiBi 的问题在于，它先验上就不鼓励看太远；如果关键答案恰好埋在很远的位置，模型可能在 logits 阶段就把那部分压得太低。

真实工程例子：BLOOM 采用了 ALiBi，这说明它在大模型训练里是可落地的，不是论文玩具。MPT 系列也使用过 ALiBi 或相关 attention bias 设计，原因同样偏工程化：实现简单、推理稳定、长一点的上下文能直接工作。但这类选择通常对应的是“成本优先、实用优先”，不是“128K 长上下文 SOTA 优先”。

常见坑还有两个：

1. 把 ALiBi 当成“万能长上下文方案”。它不是。2K、4K 以内它很有价值，64K 以上通常要看别的方法。
2. 误以为所有头都应该一样强。恰恰相反，不同头需要不同斜率，否则多头的分工会变差。

---

## 替代方案与适用边界

如果你的目标是更长上下文，主流替代路线是 RoPE 及其扩展，如位置插值、NTK-aware 缩放、YaRN。RoPE 的白话解释是：它不直接给 token 一个位置向量，而是把位置信息编码进 Q/K 的旋转角度。这样内容相关性和相对位置信息会一起进入点积。

但原生 RoPE 也有边界。训练长度之外直接外推时，效果通常明显退化。YaRN 的价值就在这里：它建立在 RoPE 扩窗上，用更高效的插值与缩放方法，把 LLaMA 一类模型扩展到 64K、128K，并保持较低 perplexity 和较好的利用率。

| 方法 | 主要思路 | 适合长度 | 远距依赖能力 | 实现复杂度 |
| --- | --- | --- | --- | --- |
| ALiBi | logits 加线性距离惩罚 | 1K 到 4K 左右较实用 | 中 | 低 |
| RoPE | Q/K 旋转编码位置 | 训练长度附近 | 中 | 中 |
| YaRN | 基于 RoPE 的扩窗插值/缩放 | 32K 到 128K | 强 | 中到高 |

怎么选可以直接按目标场景判断：

- 如果你是中小模型、训练预算紧、目标上下文 2K 左右，ALiBi 很合适。
- 如果你是通用大模型、明确要支持几十 K 到上百 K，上来就看 RoPE 扩窗系方法，尤其是 YaRN。
- 如果你的任务核心是局部一致性、邻近约束、短中程推理，ALiBi 往往足够。
- 如果你的任务核心是跨超长文档检索、跨章节精确引用、超远依赖恢复，ALiBi 往往吃亏。

一句话概括边界：ALiBi 解决的是“训练短、推理稍长”的问题，不是“无限长上下文”问题。

---

## 参考资料

1. Press, Smith, Lewis. *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*. ICLR 2022. 原始论文，ALiBi 的主要依据。链接：https://openreview.net/forum?id=R8sQPpGCv0
2. 官方实现仓库：`ofirpress/attention_with_linear_biases`。包含 slope 生成与 attention bias 的参考实现。链接：https://github.com/ofirpress/attention_with_linear_biases
3. Peng, Quesnelle, Fan, Shippole. *YaRN: Efficient Context Window Extension of Large Language Models*. ICLR 2024. 用于说明超长上下文扩展的主流替代路线。链接：https://openreview.net/forum?id=wHBfxhZu1u
4. BLOOM 模型卡与训练资料。可见其采用 ALiBi 作为位置方案，说明该方法已进入真实大模型工程实践。链接：https://huggingface.co/bigscience/bloom-560m
5. Hugging Face 论文页：*Train Short, Test Long*. 便于快速查看摘要与元信息。链接：https://huggingface.co/papers/2108.12409
