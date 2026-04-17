## 核心结论

GPT 的核心结构是 `Decoder-only Transformer`。`Decoder-only` 的意思是：模型只保留 Transformer 里的解码器堆叠，不使用单独的编码器。它的工作方式很单一也很强：给定前面的 token，预测下一个 token。

这里的 token 可以理解为“模型处理文本时使用的最小离散单位”，它不一定等于一个汉字或一个单词，也可能是子词片段。GPT 预训练时优化的是自回归语言模型，也就是让模型最大化每个位置条件概率的对数和：

$$
\max \sum_{t=1}^{T}\log P(x_t \mid x_1,\dots,x_{t-1})
$$

等价地，训练时最小化负对数似然损失：

$$
L = -\sum_{t=1}^{T}\log P(x_t \mid x_1,\dots,x_{t-1})
$$

这套目标和生成过程是严格对齐的。训练时模型学习“看到前文，预测后文”；推理时模型也正是“读取已有上下文，继续往后写”。这就是 GPT 在文本续写、对话生成、代码补全上表现自然的原因。

保证这种对齐的关键机制是 `causal mask`，中文通常叫“因果掩码”。“掩码”可以白话理解成“强行禁止看某些位置的规则”。在 GPT 中，位置 $t$ 只能看见自己和自己之前的位置，不能看未来位置，否则训练时就等于偷看答案。

GPT-1、GPT-2、GPT-3 的演进，本质上不是把架构换掉，而是在同一条技术路线下持续放大三件事：参数量、训练数据量、训练算力。随着规模增大，模型从“能续写”逐步发展到“能零样本或少样本完成任务”，也就是常说的 `in-context learning`，它的白话解释是“模型不用改参数，只靠提示词上下文就能临时学会任务格式”。

---

## 问题定义与边界

这篇文章讨论的是 GPT 这一类 `decoder-only + causal language modeling` 的模型，不讨论 BERT 这类双向编码器，也不展开 encoder-decoder 结构。

问题可以定义为一句话：为什么 GPT 必须使用 decoder-only 结构和因果掩码，它到底在优化什么，这样的设计为什么能直接用于生成？

边界先讲清楚。

第一，GPT 处理的是顺序生成问题。顺序生成的意思是：第 $t$ 个位置只能依赖 $1 \sim t-1$ 的历史，不能依赖未来。比如句子“今天下雨了”，当模型在生成“雨”之前，不能先偷看后面的“了”。如果训练时允许偷看，推理时却不允许，训练目标和真实使用场景就不一致。

第二，GPT 的预训练目标不是“理解整句后再分类”，而是“在每个位置上预测下一个 token”。这意味着它天然擅长生成，不天然擅长同时看左右上下文的精细判别任务。

第三，decoder-only 的注意力权限是下三角结构。可以用一个最小表格看清楚：

| 位置 | 可见位置 | 含义 |
|------|-----------|------|
| $t=1$ | 1 | 第一个 token 只能看自己 |
| $t=2$ | 1,2 | 第二个 token 只能看前两个 |
| $t=3$ | 1,2,3 | 第三个 token 不能看第 4 个及以后 |
| $t=4$ | 1,2,3,4 | 始终只能看当前位置及其左侧 |

这张表本质上就是“下三角注意力矩阵”的文字版。

玩具例子最容易说明边界。假设输入是“`The cat sat`”。训练时如果当前目标是预测第三个词 `sat`，模型只能使用前面的 `The cat` 作为条件，也就是学习：

$$
\log P(\text{sat} \mid \text{The cat})
$$

不能把 `sat` 自己右边的词，或者整句答案先读完再反推。否则这不是自回归建模，而是作弊。

因此，GPT 的边界也很明确：它对生成任务很自然，但对“必须同时依赖左边和右边上下文”的任务并不是最合适的原生结构。

---

## 核心机制与推导

Transformer 的核心算子是注意力。注意力可以白话理解成“当前 token 在读历史时，给每个历史位置分多少权重”。标准缩放点积注意力写成公式是：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

这里有几个术语需要先解释。

`Q`、`K`、`V` 分别是 Query、Key、Value。白话地说，Query 是“我现在想找什么”，Key 是“每个位置能提供什么索引信息”，Value 是“每个位置真正提供的内容”。$QK^T$ 先算出相似度分数，再经过 softmax 变成权重，最后对 $V$ 做加权求和。

GPT 的关键不是注意力本身，而是额外加上的掩码矩阵 $M$。定义如下：

$$
M_{ij} =
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
$$

意思是：如果列位置 $j$ 在行位置 $i$ 的未来，就把那个分数强行设成负无穷。由于 softmax 会把极小值压成 0，所以未来位置最终权重就是 0。

这一步非常关键。因为如果没有它，训练时第 3 个位置完全可以把第 4 个位置的信息也读进来，损失看起来会很低，但模型学到的是错误能力。

看一个玩具数值例子。假设某一层某个头在序列“今 天 下”上的原始注意力分数是：

$$
S=
\begin{bmatrix}
2.0 & 1.0 & 0.5 \\
1.2 & 2.5 & 0.3 \\
0.7 & 1.1 & 2.2
\end{bmatrix}
$$

如果我们给它加上因果掩码，第 1 行不能看第 2、3 列，第 2 行不能看第 3 列，第 3 行本来就在最后，因此矩阵变成：

$$
S + M=
\begin{bmatrix}
2.0 & -\infty & -\infty \\
1.2 & 2.5 & -\infty \\
0.7 & 1.1 & 2.2
\end{bmatrix}
$$

softmax 之后：

- 第 1 行权重只能落在第 1 列
- 第 2 行权重只能分给第 1、2 列
- 第 3 行权重才能分给第 1、2、3 列

这就实现了“信息只能从左往右流动，但训练时所有位置仍可并行计算”。

这里很多初学者会有一个误区：既然生成是一步一步做的，为什么训练能并行？答案是，掩码保证了每个位置的计算都只依赖合法历史，所以虽然我们一次把整句送进去，但每个位置实际上都在做自己的“局部下一词预测”。并行的是计算，不是信息泄露。

从概率分解角度看，一整个序列的联合概率可以按链式法则拆开：

$$
P(x_1,\dots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_1,\dots,x_{t-1})
$$

取对数后就变成求和：

$$
\log P(x_1,\dots,x_T)=\sum_{t=1}^{T}\log P(x_t\mid x_{<t})
$$

这就是 GPT 训练目标的来源。它不是拍脑袋选出来的，而是对序列联合概率做了标准概率分解。也因此，causal mask 和自回归损失是成套出现的：一个负责约束“能看什么”，一个负责优化“该预测什么”。

---

## 代码实现

先看一个最小可运行的 Python 例子，演示因果掩码如何把未来位置权重压成 0。这个例子不依赖深度学习框架，只用 `numpy`。

```python
import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def causal_mask(seq_len):
    mask = np.zeros((seq_len, seq_len), dtype=np.float64)
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                mask[i, j] = -1e9
    return mask

scores = np.array([
    [2.0, 1.0, 0.5],
    [1.2, 2.5, 0.3],
    [0.7, 1.1, 2.2],
], dtype=np.float64)

masked_scores = scores + causal_mask(3)
weights = softmax(masked_scores)

# 第 1 个位置不能看未来
assert np.allclose(weights[0, 1:], [0.0, 0.0], atol=1e-8)

# 第 2 个位置不能看第 3 个位置
assert np.allclose(weights[1, 2], 0.0, atol=1e-8)

# 每一行仍然是合法概率分布
assert np.allclose(weights.sum(axis=1), [1.0, 1.0, 1.0], atol=1e-8)

print(weights)
```

这个代码体现了 GPT 里最关键的约束，而不是完整模型。完整 GPT 通常还包括下面这些部分：

| 组件 | 作用 | 白话解释 |
|------|------|---------|
| Token Embedding | 把 token id 映射成向量 | 把离散编号变成可计算表示 |
| Position Embedding | 注入位置信息 | 告诉模型谁在前谁在后 |
| Multi-Head Attention | 多头注意力 | 用多个子空间同时观察依赖关系 |
| FFN | 前馈网络 | 对每个位置做非线性变换 |
| Residual | 残差连接 | 保留原信息，便于训练深层网络 |
| LayerNorm | 层归一化 | 稳定数值和梯度 |
| LM Head | 词表投影层 | 把隐藏状态变成下一个 token 概率 |

一个简化版伪代码如下：

```python
x = token_embedding(input_ids) + position_embedding(positions)

for block in decoder_blocks:
    x = x + causal_multi_head_attention(layer_norm(x))
    x = x + ffn(layer_norm(x))

x = final_layer_norm(x)
logits = x @ W_vocab.T
```

训练阶段一般采用 `teacher forcing`。这个术语的白话解释是“训练时把正确前文直接喂给模型，而不是让模型自己生成再接着喂”。例如输入是：

- 输入序列：`[BOS, 我, 喜欢, 学习]`
- 目标序列：`[我, 喜欢, 学习, AI]`

模型一次前向就能得到每个位置的 logits，然后与目标对齐计算交叉熵损失。由于有因果掩码，虽然整段一起算，但每个位置都没有越权。

真实工程例子是代码补全。假设 IDE 把下面的上下文送入模型：

```python
def add(a, b):
    return
```

decoder-only 模型会根据左侧上下文预测下一个 token，可能先生成空格，再生成 `a`，再生成 `+`，再生成 `b`。这不是“整句一起想好再吐出来”，而是一次一个 token 地向前滚动。工程上通常会缓存每层历史的 `K/V cache`，也就是“把过去 token 的 key 和 value 存起来，下次生成时不重复计算”，这样单步解码成本才不会随着历史长度重复爆炸。

---

## 工程权衡与常见坑

decoder-only GPT 的最大工程优势是统一。训练目标统一，模型结构统一，生成方式统一，预训练和下游生成天然一致。这使它非常适合构建通用文本生成系统。

但它也有很明确的代价和坑。

| 问题 | 症状 | 当前做法 |
|------|------|---------|
| 忘记加 causal mask | 训练损失异常好看，生成效果却很差 | 明确构造下三角 mask，单测检查未来权重为 0 |
| mask 方向写反 | 模型只能看未来或看不到历史 | 用小矩阵打印 attention 权重做人工校验 |
| 位置编码处理错误 | 长文本顺序感混乱 | 检查 position id 是否与缓存逻辑一致 |
| 深层网络训练不稳 | 梯度爆炸、loss 发散 | 使用 pre-layernorm、合适初始化、学习率预热 |
| 推理太慢 | 每生成一个 token 都全量重算 | 使用 KV cache |
| 长上下文退化 | 前文越长越难利用 | 引入更长上下文训练、分块注意力或检索增强 |

先说最重要的坑：训练和推理不一致。GPT 之所以有效，依赖的是“训练时只看左边，推理时也只看左边”。如果训练时允许看未来，模型会学到一种推理阶段根本不可用的策略。这种错误在 loss 曲线上不一定立刻暴露，因为它会让训练看起来更容易。

第二个工程问题是深层稳定性。随着 GPT 从 GPT-1 到 GPT-2、GPT-3 规模增大，网络层数和宽度都增加，残差路径变长，优化难度急剧上升。`pre-layernorm` 的意思是“先做层归一化，再做注意力或前馈”，白话地说就是先把数值范围整理一下，再送入大模块计算。相比早期的 post-norm，pre-norm 更容易稳定训练深层模型。

第三个问题是推理复杂度。训练时整个序列可以并行，所以 GPU 吞吐高；推理时必须逐 token 解码，所以天然更慢。KV cache 是工业系统的标配，否则用户每多生成一个 token，前面所有历史都要再算一遍，代价会非常高。

第四个问题是长上下文幻觉。很多人以为“上下文窗口更长”就等于“模型必然更会用长上下文”。这是两回事。窗口长度是硬容量，是否真能利用远距离信息还取决于训练分布、位置编码设计、注意力模式以及数据质量。

---

## 替代方案与适用边界

如果任务目标不是生成，而是判别，decoder-only 往往不是最优原生结构。

`Encoder-only` 的代表是 BERT。它可以双向看上下文，白话解释是“当前词左边右边都能一起看”，所以更适合分类、序列标注、抽取这类理解型任务。比如情感分类里，句尾的否定词会反过来改变前文含义，双向编码通常更自然。

`Encoder-decoder` 的代表是 T5、BART。这类结构先用编码器读完整输入，再由解码器自回归生成输出，适合机器翻译、摘要、问答这类“输入和输出职责明确分开”的任务。

对比可以压缩成下表：

| 架构 | 上下文访问 | 生成能力 | 更适合的任务 |
|------|------------|---------|-------------|
| Decoder-only | 只能看左侧历史 | 强 | 续写、对话、代码补全、通用生成 |
| Encoder-only | 可同时看左右上下文 | 弱 | 分类、抽取、检索表示学习 |
| Encoder-decoder | 编码端全局可见，解码端因果生成 | 强 | 翻译、摘要、结构化生成 |

还有一些折中路线，比如 `Prefix LM`。它允许前缀区域内部双向可见，而生成区域继续使用因果掩码。白话地说，就是“先把一段上下文完整读懂，再从后面开始续写”。这种设计试图兼顾理解和生成，但实现复杂度和训练设定都更细。

真实工程里怎么选，取决于目标是否以“开放式生成”为中心。

- 如果你做聊天、长文续写、代码助手，decoder-only 是自然选择。
- 如果你做文本分类、实体识别、句向量检索，encoder-only 往往更高效。
- 如果你做文档到摘要、问题到答案、源语言到目标语言，encoder-decoder 更符合输入输出结构。

GPT 的边界并不是“不能做理解任务”，而是“它做理解任务时通常是通过 prompt 把任务改写成生成问题来做”。这在通用性上很强，但在专用性和效率上不一定占优。

---

## 参考资料

- Michael Brenndoerfer, “Decoder Architecture: Causal Masking & Autoregressive Generation”, https://mbrenndoerfer.com/writing/decoder-architecture-causal-masking-autoregressive-transformers
- Emberverse, “Language Model Architecture Comparison”, https://emberverse.ai/piece/language_model_architecture_comparison
- Emberverse, “Architecture Variants”, https://emberverse.ai/piece/architecture_variants
- Michael Brenndoerfer, “GPT-2: Scaling Language Models for Zero-Shot Learning”, https://mbrenndoerfer.com/writing/gpt-2-scaling-language-models-zero-shot-learning
- Vaswani et al., “Attention Is All You Need”, https://arxiv.org/abs/1706.03762
- Radford et al., “Improving Language Understanding by Generative Pre-Training”, https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
- Radford et al., “Language Models are Unsupervised Multitask Learners”, https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- Brown et al., “Language Models are Few-Shot Learners”, https://arxiv.org/abs/2005.14165
