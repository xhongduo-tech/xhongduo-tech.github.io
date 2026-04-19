## 核心结论

HAN，Hierarchical Attention Network，中文常译为“层次注意力网络”，是一种面向文档分类的神经网络：先把词聚合成句子表示，再把句子聚合成文档表示，最后预测类别。

它的结构可以写成：

```text
词 -> 句子 -> 文档 -> 分类
```

HAN 不是把整篇文档直接压成一个长序列，而是显式模拟文档的自然层次：

```text
Embedding
  -> Word BiGRU
  -> Word Attention
  -> Sentence Vector
  -> Sentence BiGRU
  -> Sentence Attention
  -> Doc Vector
  -> Classifier
```

核心价值不是“模型更复杂”，而是“结构更适合长文本”。一篇长文档里，并不是每个词都同等重要，也不是每个句子都同等重要。HAN 先在句子内部找关键词，再在文档内部找关键句，因此能同时保留句内信息和句间结构。

新手版解释：一篇客服工单里，“退款失败”“重复扣费”这种词决定某个句子的含义；而“问题描述”“处理结果”“用户投诉”这些句子决定整篇工单的类别。HAN 会先找词，再找句。

| 方法 | 基本思路 | 是否利用句子结构 | 长文本适配 | 可解释线索 |
|---|---|---:|---:|---:|
| HAN | 词级注意力 + 句级注意力 | 是 | 较好 | 较强 |
| 直接 RNN | 把全文当成一个长序列 | 否 | 一般 | 弱 |
| 平均池化 | 对词向量或句向量求平均 | 弱 | 一般 | 弱 |
| Transformer 长文本方案 | 用自注意力或稀疏注意力建模长上下文 | 视模型而定 | 强 | 中等 |

HAN 是结构清晰、可解释性较强的经典长文本分类方案，但不是当前所有场景的最优解。

---

## 问题定义与边界

任务定义：给定一个由多个句子组成的文档，预测它所属的类别标签。输入既包含词序信息，也包含句子之间的结构信息。

设文档为：

$$D = \{S_1, S_2, ..., S_L\}$$

其中 $L$ 表示文档中的句子数。第 $i$ 个句子为：

$$S_i = \{w_{i1}, w_{i2}, ..., w_{iT_i}\}$$

其中 $T_i$ 表示第 $i$ 个句子的词数，$w_{it}$ 表示第 $i$ 个句子里的第 $t$ 个词。

分类目标是学习一个函数：

$$f(D) \rightarrow y$$

其中 $y$ 是类别标签，例如“退款问题”“物流问题”“正面评论”“负面评论”“法律风险”等。

新手版解释：如果是“商品评论分类”，一条评论可能有 5 到 20 个句子。HAN 能先判断每句在说什么，再判断整条评论是好评还是差评。如果只是“单句情感分类”，例如“这手机很好用”，HAN 的层次优势就不明显，因为它没有太多句间结构可利用。

| 场景 | 是否适合 HAN | 原因 |
|---|---:|---|
| 长文档分类 | 适合 | 有多句结构，信息分布不均 |
| 客服工单 | 适合 | 关键投诉点通常只出现在少数句子里 |
| 法律文本 | 适合 | 条款、事实、结论常分布在不同句子中 |
| 论文摘要 | 适合 | 背景、方法、结论有自然层次 |
| 舆情分析 | 适合 | 关键情绪或事件常集中在部分句子 |
| 超短文本 | 不适合 | 层次结构不足，额外复杂度收益低 |
| 无清晰分句文本 | 不适合 | 分句错误会破坏模型输入结构 |
| 强依赖跨句细粒度交互任务 | 不适合 | HAN 的句级聚合可能丢失复杂跨句关系 |

边界要明确：HAN 适合“有自然句子边界的长文本分类”，不适合句子切分很差、文本极短，或者需要精细跨句推理的任务。例如多轮对话中的指代消解、法律条款之间的复杂引用、论文段落之间的逻辑推断，通常需要更强的交互建模能力。

---

## 核心机制与推导

HAN 有两层核心结构：词级编码和句级编码。

第一层是词级编码。每个句子先经过 BiGRU 得到上下文词表示，再通过注意力机制挑出句内重点词，汇聚成句向量。

BiGRU，Bidirectional GRU，中文是“双向门控循环单元”。白话解释：它会同时从左到右、从右到左读一个序列，让每个词的表示同时包含前文和后文信息。

注意力机制，Attention，白话解释：它给不同输入分配不同权重，让模型更关注对当前任务更重要的部分。

词级公式如下：

```text
h_it = [\overrightarrow{h}_{it};\overleftarrow{h}_{it}]
u_it = tanh(W_w h_it + b_w)
α_it = exp(u_it^T u_w) / sum_t exp(u_it^T u_w)
s_i  = sum_t α_it h_it
```

其中 $h_{it}$ 是第 $i$ 个句子中第 $t$ 个词的上下文表示，$u_{it}$ 是它经过非线性变换后的注意力中间表示，$\alpha_{it}$ 是词级注意力权重，$s_i$ 是第 $i$ 个句子的向量表示。

$u_w$ 是可学习的词级上下文向量。白话解释：它像一个“什么词重要”的查询向量，会在训练中自动调整。

第二层是句级编码。模型把所有句向量组成一个句子序列，再送入另一个 BiGRU，得到上下文句表示，然后通过句级注意力聚合成文档向量。

句级公式如下：

```text
h_i  = [\overrightarrow{h}_i;\overleftarrow{h}_i]
u_i  = tanh(W_s h_i + b_s)
α_i  = exp(u_i^T u_s) / sum_i exp(u_i^T u_s)
v    = sum_i α_i h_i

p = softmax(W_c v + b_c)
```

其中 $h_i$ 是第 $i$ 个句子的上下文表示，$\alpha_i$ 是句级注意力权重，$v$ 是文档向量，$p$ 是类别概率分布。

$u_s$ 是可学习的句级上下文向量。白话解释：它像一个“什么句子重要”的查询向量。

新手版解释：句子里每个词都像一票，词级注意力决定哪些词投票更重；整篇文章里的每个句子再投一次票，句级注意力决定哪些句子更重要。最后不是“所有词平均一下”，而是“关键词先选，关键句再选”。

玩具例子：设有 2 个句子，每句 2 个词。句 1 的词级打分是 $[2,0]$，softmax 后权重约为 $[0.881,0.119]$。若 $h_{11}=[1,0]$，$h_{12}=[0,1]$，则：

$$s_1=0.881[1,0]+0.119[0,1]=[0.881,0.119]$$

句 2 的词级打分是 $[0,0]$，权重为 $[0.5,0.5]$。若 $h_{21}=[1,1]$，$h_{22}=[0,2]$，则：

$$s_2=0.5[1,1]+0.5[0,2]=[0.5,1.5]$$

句级打分设为 $[1.2,0.2]$，权重约为 $[0.731,0.269]$，文档向量为：

$$v=0.731s_1+0.269s_2\approx[0.778,0.492]$$

这表示模型更偏向句 1，也更偏向句 1 里的第 1 个词。

真实工程例子：客服工单分类。一条工单可能包含用户描述、客服追问、历史订单、处理结果和补充说明。真正决定类别的可能只有一句“用户反馈重复扣费，退款失败”。HAN 可以先在句内关注“重复扣费”“退款失败”，再在文档层面关注这句投诉描述，从而把工单分到“支付退款问题”。

---

## 代码实现

代码实现重点不是堆模块，而是清楚表达两次编码、两次注意力、两次加权求和。实际工程中通常会把一个 batch 的文档整理成形状类似 `[batch, sentences, words]` 的输入。模型先逐句处理每一行，再把所有句子的结果当成新的序列继续处理。

常见结构如下：

| 模块 | 作用 |
|---|---|
| `Dataset / CollateFn` | 分句、分词、截断、padding，构造层次张量 |
| `WordEncoder` | `Embedding + BiGRU + WordAttention` |
| `SentenceEncoder` | `BiGRU + SentenceAttention` |
| `Classifier` | `Linear + Softmax` 或训练时直接输出 logits |

伪代码：

```python
for doc in batch:
    sent_vecs = []
    for sent in doc:
        h_words = bigru_words(emb(sent))
        s_vec = word_attention(h_words)
        sent_vecs.append(s_vec)
    h_sents = bigru_sents(sent_vecs)
    v = sentence_attention(h_sents)
    logits = classifier(v)
```

输入和输出形状可以按下面理解：

| 阶段 | 张量形状示例 | 含义 |
|---|---|---|
| 输入 token id | `[B, L, T]` | B 个文档，每个最多 L 句，每句最多 T 个词 |
| Embedding | `[B, L, T, E]` | 每个词变成 E 维向量 |
| Word BiGRU 输出 | `[B, L, T, 2H]` | 每个词有双向上下文表示 |
| Word Attention 输出 | `[B, L, 2H]` | 每个句子变成一个向量 |
| Sentence BiGRU 输出 | `[B, L, 2H]` | 每个句子有上下文表示 |
| Sentence Attention 输出 | `[B, 2H]` | 每篇文档变成一个向量 |
| Classifier 输出 | `[B, C]` | 每篇文档对 C 个类别的打分 |

mask，中文可理解为“有效位置标记”。它的作用是告诉模型哪些 token 是真实输入，哪些只是 padding。没有 mask 时，padding 也会参与 softmax，注意力权重会被污染。

下面是一个可运行的最小 Python 例子，只实现 mask attention 和两级聚合，不依赖深度学习框架：

```python
import numpy as np

def masked_softmax(scores, mask):
    scores = np.array(scores, dtype=float)
    mask = np.array(mask, dtype=bool)
    masked = np.where(mask, scores, -1e9)
    exps = np.exp(masked - np.max(masked))
    exps = np.where(mask, exps, 0.0)
    return exps / exps.sum()

def attention_pool(vectors, scores, mask):
    vectors = np.array(vectors, dtype=float)
    weights = masked_softmax(scores, mask)
    return (weights[:, None] * vectors).sum(axis=0), weights

# 词级：两个句子，每句最多三个词，第三个位置可能是 padding
sent1_vectors = np.array([[1, 0], [0, 1], [9, 9]])
sent1_scores = [2.0, 0.0, 100.0]
sent1_mask = [1, 1, 0]

sent2_vectors = np.array([[1, 1], [0, 2], [9, 9]])
sent2_scores = [0.0, 0.0, 100.0]
sent2_mask = [1, 1, 0]

s1, w1 = attention_pool(sent1_vectors, sent1_scores, sent1_mask)
s2, w2 = attention_pool(sent2_vectors, sent2_scores, sent2_mask)

# 句级：两句都有效
doc_vector, sent_weights = attention_pool(
    vectors=np.array([s1, s2]),
    scores=[1.2, 0.2],
    mask=[1, 1],
)

assert np.allclose(w1.sum(), 1.0)
assert w1[2] == 0.0
assert np.allclose(w2, [0.5, 0.5, 0.0])
assert doc_vector.shape == (2,)
assert sent_weights[0] > sent_weights[1]

print("word weights sent1:", w1.round(3))
print("word weights sent2:", w2.round(3))
print("sentence weights:", sent_weights.round(3))
print("doc vector:", doc_vector.round(3))
```

这个例子里，padding 位置的打分故意设得很高。如果没有 mask，padding 会获得极高注意力权重，输出会被 `[9,9]` 污染。加入 mask 后，padding 权重固定为 0。

---

## 工程权衡与常见坑

HAN 的效果高度依赖分句和截断策略。句子切分错了，层次结构就被破坏了。新手版解释：如果一条工单被错误切成很多碎句，模型会把本来属于同一句的话拆开理解，注意力就会学偏；如果文本太长但又不截断，显存会爆，训练也会被大量无关内容干扰。

| 常见坑 | 后果 | 处理建议 |
|---|---|---|
| 句子切分错误 | 词级和句级边界都不可靠 | 使用稳定分句规则，抽样检查切分结果 |
| 文本过长 | 显存上升，噪声增加 | 设置 `max_sentences` 和 `max_words_per_sentence` |
| 注意力权重被误解为因果解释 | 错把相关线索当成严格原因 | 只把注意力可视化当作分析线索 |
| 小数据过拟合 | 训练集好、验证集差 | dropout、早停、词表裁剪、正则化 |
| 领域短文本收益有限 | 复杂度增加但效果不升 | 改用 TextCNN、BiGRU 或小型 Transformer |

工程参数建议：

| 参数 | 作用 | 建议 |
|---|---|---|
| `max_sentences` | 每篇文档最多保留多少句 | 根据长度分布选 80% 到 95% 分位点 |
| `max_words_per_sentence` | 每句最多保留多少词 | 常见可从 30、50、80 试起 |
| `dropout` | 降低过拟合 | 小数据可适当提高 |
| `early stopping` | 验证集不再提升时停止训练 | 监控验证集 F1 或 loss |
| `vocab pruning` | 控制词表大小 | 过滤极低频词，保留领域关键词 |

注意力可视化只能做线索，不等于解释。比如模型把高权重给了“退款失败”，可以说明这个词对当前预测有较大贡献线索，但不能直接证明“因为这个词，所以模型一定预测为退款类”。注意力权重受训练数据、上下文、参数初始化和相邻词共同影响。

另一个常见问题是截断方向。很多人默认保留文档开头，但客服工单、医疗记录、审计日志的关键信息可能出现在结尾。更稳的做法是先分析数据：标签相关信息更常出现在开头、结尾，还是中间。必要时可以保留开头若干句和结尾若干句，而不是简单截断后半部分。

---

## 替代方案与适用边界

如果任务是短文本分类，或者文本没有清晰分句，直接使用 Transformer、TextCNN、BiGRU 可能更简单、更稳。如果任务需要更强的全局交互或更长上下文建模，HAN 的表达能力通常不如现代长文本 Transformer 方案。

新手版解释：如果你要分类的是一条很短的用户评价，例如“发货快，质量不错”，HAN 多出来的层次结构不一定带来收益；但如果是一份多段客服工单，HAN 往往比“整篇平均池化”更容易抓住关键投诉点。

| 方案 | 是否利用句子结构 | 是否易解释 | 是否适合长文本 | 训练成本 | 对分句依赖 |
|---|---:|---:|---:|---:|---:|
| HAN | 是 | 较强 | 较好 | 中等 | 高 |
| 平均池化 RNN | 弱 | 弱 | 一般 | 低到中 | 低 |
| CNN/TextCNN | 否 | 中等 | 一般 | 低 | 低 |
| Transformer 长文本模型 | 视设计而定 | 中等 | 强 | 较高 | 低到中 |

平均池化 RNN 的优点是实现简单，缺点是容易把关键信息冲淡。TextCNN 的优点是训练快，适合短文本和局部模式明显的任务，例如标题分类、短评论分类。Transformer 长文本模型的优点是表达能力强，可以建模更复杂的跨句关系，缺点是训练和推理成本通常更高。

HAN 的适用边界可以概括为：文档足够长、有自然句子结构、分类信号集中在部分词和部分句子中，并且工程上希望获得一定程度的注意力可视化线索。在这些条件下，HAN 仍然是一个值得理解和复现的经典基线。

最终结论：HAN 是结构清晰、可解释性较强的经典长文本分类方案，但不是当前所有场景的最优解。它适合用来理解“层次建模 + 注意力聚合”的基本思想，也适合作为长文本分类任务的强基线之一。

---

## 参考资料

1. [Hierarchical Attention Networks for Document Classification - ACL Anthology](https://aclanthology.org/N16-1174/)
2. [Hierarchical Attention Networks for Document Classification - Microsoft Research PDF](https://www.microsoft.com/en-us/research/uploads/prod/2017/06/Hierarchical-Attention-Networks-for-Document-Classification.pdf)
3. [Hierarchical Attention Networks for Document Classification - Microsoft Research](https://www.microsoft.com/en-us/research/publication/hierarchical-attention-networks-document-classification/)
4. [GitHub - arunarn2/HierarchicalAttentionNetworks](https://github.com/arunarn2/HierarchicalAttentionNetworks)

本文依据论文原文整理，公式与结构以原文为准。建议阅读顺序：先看 ACL Anthology 页面确认论文信息，再看 PDF 理解公式和结构，最后看 GitHub 参考实现理解工程落地。
