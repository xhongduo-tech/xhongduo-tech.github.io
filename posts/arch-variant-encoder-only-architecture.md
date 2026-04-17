## 核心结论

Encoder-only 架构是指只保留 Transformer 的 encoder 堆栈。它的核心特征是**双向自注意力**，也就是序列里每个 token 都能同时看到左边和右边的上下文，不使用生成模型里的因果掩码。

这类模型的输出不是“下一个 token”，而是整段输入的**上下文表示**。白话说，模型读完一句话后，会给句中每个词都生成一个“结合了全文语境的新向量”。如果输入形状是 $[\text{batch}, \text{seq}]$，经过 encoder 后常见输出形状是：

$$
H \in \mathbb{R}^{\text{batch} \times \text{seq} \times \text{hidden}}
$$

其中第 0 个位置常放 `[CLS]`，它是一个专门用于聚合全句语义的占位 token。最后一层的 `[CLS]` 向量常被拿去做分类、排序、检索打分。

新手可先记住一句话：**把整段文本一次送进去，一次前向计算就拿到整句和每个词的表示，所以它非常适合理解类任务，不适合直接写长文章。**

| 架构 | 主要输出 | 推理方式 | 典型任务 |
| --- | --- | --- | --- |
| Encoder-only | 全序列表示、`[CLS]` 向量 | 单次 forward | 分类、检索、embedding、rerank、NER |
| Decoder-only | 下一个 token 概率 | 逐 token 自回归 | 对话、续写、代码生成 |
| Encoder-decoder | 条件生成序列 | encoder 一次 + decoder 自回归 | 翻译、摘要、问答生成 |

代表模型里，BERT-base 约 110M 参数，BERT-large 约 340M。RoBERTa 沿用 encoder-only 主体，但通过更长训练、更多数据、去掉 NSP、动态 mask 等训练策略，通常比 BERT 更强。DeBERTa 进一步把“内容”和“位置”拆开建模，在多项理解任务上继续推进。到今天，BGE、E5、GTE 这类嵌入模型，以及大量 reranker，仍然广泛采用 encoder-only，因为它们**非自回归、一次前向、吞吐高、部署简单**。

---

## 问题定义与边界

要理解 Encoder-only，先把问题边界划清楚。

它解决的是**文本理解**问题。文本理解指的是：给定一句或一段文本，输出一个标签、一个分数、一个向量，或者给每个 token 打标签。它不擅长做的是**长文本逐词生成**。

一个最小流程可以写成：

`输入 token 序列 -> embedding -> 多层 encoder -> 每个 token 的上下文向量 + [CLS] 聚合向量`

玩具例子：

输入句子：

`[CLS] 今天 上海 下雨 了 [SEP]`

经过 encoder 后：
- “上海”这个 token 的表示，不再只是“上海”这个词本身，而是“在今天、下雨、了”这些上下文条件下的“上海”
- `[CLS]` 会变成一个浓缩整句语义的向量，可用于“天气类句子分类”之类任务

这和 Decoder-only 不同。Decoder-only 的目标是“看到前文，预测后文”；Encoder-only 的目标是“看到全文，得到表示”。

| 维度 | Encoder-only | Decoder-only |
| --- | --- | --- |
| 注意力范围 | 左右都看 | 只能看左边已出现内容 |
| 训练目标 | MLM、对比学习、分类等 | 下一个 token 预测 |
| 单次输入后输出 | 全序列表示 | 当前步 token 分布 |
| 是否适合长文本生成 | 否 | 是 |
| 是否适合 embedding/检索 | 是 | 可做，但通常不如专用 encoder 高效 |

所以“通勤路上一次性把整篇文章交给模型，让每个词互相看一遍，再输出一个整段表示”，这是 encoder-only。  
“先给几个词，再一个一个往后吐词”，这是 decoder-only。

边界上还要注意一点：Encoder-only 不是不能输出文本，而是**不是为文本生成这个目标设计的**。你当然可以在顶层接一个分类头、span 预测头、MLM 头，但它不会像 GPT 那样天然适合在线连续生成。

---

## 核心机制与推导

### 1. 双向自注意力为什么适合理解

自注意力可以理解成“句中每个位置都去看别的位置，并计算应该看多少”。双向的意思是位置 $i$ 在计算时能访问全序列，而不是只访问 $1 \dots i$。

对第 $i$ 个 token，输出向量可写成：

$$
h_i' = \sum_{j=1}^{n} \alpha_{ij} v_j
$$

其中 $\alpha_{ij}$ 是第 $i$ 个位置对第 $j$ 个位置的注意力权重。因为 $j$ 可以遍历整句，所以“苹果”在“苹果发布新芯片”和“我买了苹果”里会得到不同表示。这就是上下文化。

### 2. MLM：不是生成下一个词，而是填空

MLM 是 **Masked Language Modeling**，白话说就是“随机挖空，再让模型猜回去”。BERT 的经典做法是随机选 15% token 参与预测：

- 80% 替换成 `[MASK]`
- 10% 保持原词不变
- 10% 替换成随机词

若序列长度为 $n$，掩码集合为 $M$，则训练目标可写成：

$$
\mathcal{L}_{MLM} = - \sum_{i \in M} \log P(x_i \mid x_{\setminus M})
$$

这里的 $x_i$ 是原始 token，$x_{\setminus M}$ 表示带掩码后的上下文。

玩具例子：

原句：
`[CLS] 猫 坐 在 垫子 上 [SEP]`

如果抽中了“坐”和“上”，一种可能的训练输入是：
`[CLS] 猫 [MASK] 在 垫子 随机词 [SEP]`

模型在**一次 forward**里同时预测多个被选中的位置，而不是像 GPT 那样先预测“坐”，再预测后面的词。

### 3. `[CLS]` 为什么能代表整句

`[CLS]` 是一个特殊 token。它一开始没有“天然意义”，但在预训练和微调过程中，会不断被任务头读取，所以最后会学成一个“整句摘要位”。

分类时常用：

$$
z = H_{[:,0,:]}
$$

也就是取最后一层第一个位置的向量作为句向量，再接线性层：

$$
\hat y = W z + b
$$

这就是很多情感分类、主题分类、query 分类的基本做法。

### 4. RoBERTa 为什么比 BERT 更强

RoBERTa 的核心不是发明了新骨架，而是说明：**BERT 很大程度上是没训够。**

它的关键训练策略包括：
- 去掉 NSP（Next Sentence Prediction，判断两句是否相邻的任务）
- 使用更大数据和更大 batch
- 训练更久
- 使用动态 mask，而不是把同一位置永远固定成掩码

动态 mask 的直觉很简单：如果每次都盯着同一套填空题，模型会记套路；如果每个 epoch 被遮住的位置都变，模型更容易学到普适规律。

### 5. DeBERTa：把“内容”和“位置”拆开算

DeBERTa 的关键改动是 **disentangled attention**，白话说就是把“这个词是什么”和“这个词在相对什么位置”分开建模。

BERT 里，词向量和位置向量通常直接相加；DeBERTa 则显式区分内容向量 $c_i$ 和相对位置向量 $p_{ij}$，再把注意力拆成多项。简化理解下，可以把未归一化分数写成三部分之和：

$$
\tilde A_{ij}
=
\langle Q^c_i, K^c_j \rangle
+
\langle Q^c_i, K^p_{ij} \rangle
+
\langle Q^p_{ij}, K^c_j \rangle
$$

再归一化为：

$$
A_{ij} = \mathrm{softmax}_j \left(\frac{\tilde A_{ij}}{\sqrt{3d}}\right)
$$

它的直觉是：  
“北京 在 上海 北边”与“上海 在 北京 北边”这两句，词基本一样，但相对位置关系完全不同。把内容和位置拆开后，模型更容易精确表达这类差异。

真实工程上，这种改进对 NLU、排序、抽取类任务常常有价值，因为这些任务很依赖细粒度语义关系，而不是只要会续写。

---

## 代码实现

下面用一个极简、可运行的 Python 例子模拟 encoder-only 的核心形态：一次 forward 同时得到 token 级输出、MLM logits 和 `[CLS]` 分类 logits。这里不依赖深度学习框架，只用 `numpy` 演示形状和流程。

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

class TinyEncoderOnly:
    def __init__(self, vocab_size=20, hidden=8, num_labels=3, seed=0):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.num_labels = num_labels

        self.token_emb = rng.normal(scale=0.1, size=(vocab_size, hidden))
        self.pos_emb = rng.normal(scale=0.1, size=(32, hidden))

        self.Wq = rng.normal(scale=0.1, size=(hidden, hidden))
        self.Wk = rng.normal(scale=0.1, size=(hidden, hidden))
        self.Wv = rng.normal(scale=0.1, size=(hidden, hidden))

        self.mlm_head = rng.normal(scale=0.1, size=(hidden, vocab_size))
        self.cls_head = rng.normal(scale=0.1, size=(hidden, num_labels))

    def encode(self, input_ids):
        # input_ids: [batch, seq]
        batch, seq = input_ids.shape
        x = self.token_emb[input_ids] + self.pos_emb[np.arange(seq)][None, :, :]  # [B, S, H]

        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv

        scores = q @ np.transpose(k, (0, 2, 1)) / np.sqrt(self.hidden)  # [B, S, S]
        attn = softmax(scores, axis=-1)
        h = attn @ v  # [B, S, H]
        return h

    def forward(self, input_ids):
        h = self.encode(input_ids)                      # [B, S, H]
        mlm_logits = h @ self.mlm_head                 # [B, S, V]
        cls_vec = h[:, 0, :]                           # 取 [CLS]
        cls_logits = cls_vec @ self.cls_head           # [B, C]
        return h, mlm_logits, cls_logits

# 假设 0=[CLS], 1=[MASK], 2=[SEP]
model = TinyEncoderOnly(vocab_size=30, hidden=12, num_labels=2, seed=42)
input_ids = np.array([
    [0, 5, 1, 9, 2],   # 第 3 个位置是 [MASK]
    [0, 7, 8, 1, 2],   # 第 4 个位置是 [MASK]
])

h, mlm_logits, cls_logits = model.forward(input_ids)

assert h.shape == (2, 5, 12)
assert mlm_logits.shape == (2, 5, 30)
assert cls_logits.shape == (2, 2)

# 一次 forward 同时取出多个掩码位置的预测
mask_positions = np.argwhere(input_ids == 1)
masked_scores = np.array([mlm_logits[b, s] for b, s in mask_positions])
assert masked_scores.shape == (2, 30)

# [CLS] 可直接用于句级分类
pred_labels = np.argmax(cls_logits, axis=-1)
assert pred_labels.shape == (2,)
```

上面这段代码对应的工程结构就是：

`input_ids -> token/position embedding -> encoder ->`
- `mlm_head`：对每个位置输出词表 logits，形状 `[batch, seq, vocab]`
- `cls_head`：对 `[CLS]` 输出分类 logits，形状 `[batch, num_labels]`

| 张量 | 含义 | 形状 |
| --- | --- | --- |
| `input_ids` | 输入 token id | `[batch, seq]` |
| `h` | encoder 输出 | `[batch, seq, hidden]` |
| `mlm_logits` | MLM 预测 | `[batch, seq, vocab]` |
| `cls_logits` | 句级分类预测 | `[batch, num_labels]` |

真实工程例子是检索系统：
1. 用 encoder-only embedding 模型把 query 和文档都编码成向量
2. 先做大规模向量召回
3. 再把 query 和候选文档拼接后送入 cross-encoder reranker 精排

这里第一阶段和第二阶段都常见 encoder-only 变体，因为它们不需要在线逐词生成，只需要快速计算表示或相关性分数。

---

## 工程权衡与常见坑

Encoder-only 的优势很明确，但工程上也有典型误区。

| 常见坑 | 问题本质 | 规避方式 |
| --- | --- | --- |
| 静态 mask | 训练时总见同一套填空位置，泛化差 | 用动态 mask，RoBERTa 的经验非常关键 |
| 迷信原始 `[CLS]` | 预训练得到的 `[CLS]` 不一定直接适合业务标签 | 做任务微调，或用专门训练过的 embedding 模型 |
| 误把生成模型当 embedding 主力 | 可用，但吞吐和延迟通常不占优 | 检索优先考虑 encoder-only embedding |
| 只看召回不看精排 | embedding 擅长粗筛，不等于最终排序最准 | 大规模检索用 embedding，小集合重排用 reranker |
| 忽略长度成本 | self-attention 对序列长度是二次复杂度 | 长文要切块、分段、做层级聚合 |

RoBERTa 给出的经验很重要：去掉 NSP、增大数据规模、加动态 mask，并不是“训练细节可有可无”，而是会直接影响泛化性能。很多团队复现 BERT 类模型效果差，问题不在结构，而在训练 recipe 太弱。

关于 throughput，可以直接这样理解：

- `embedding`：一次编码一个文本，适合海量库离线建索引和在线粗召回
- `reranker`：一次同时看 query 和候选文本，准确率更高，但吞吐明显低
- 常见组合：`embedding recall -> topK -> reranker`

对零基础读者，一个实用判断标准是：  
如果你要“从 100 万文档里先找 100 个可能相关的”，先用 embedding。  
如果你要“把这 100 个候选排出前 5 名”，再上 reranker。  
不要一上来就拿 cross-encoder 跑全库，也不要拿未经任务对齐的 `[CLS]` 直接替代成熟 embedding 模型。

---

## 替代方案与适用边界

Encoder-only 最适合的场景是：**分类、匹配、相似度、检索召回、rerank、序列标注、抽取**。这些任务都属于“读懂输入，然后输出结构化结果”。

如果任务变成“根据提示持续生成一大段文本”，应切换到别的架构。

| 方案 | 最适合任务 | 在线代价 | 吞吐特征 | 备注 |
| --- | --- | --- | --- | --- |
| Encoder-only | embedding、分类、rerank、NER | 低到中 | 高，单次 forward | 理解强，不直接生成 |
| Decoder-only | 对话、续写、代码生成 | 中到高 | 低于 encoder-only，逐步生成 | 生成能力最强 |
| Encoder-decoder | 翻译、摘要、受控生成 | 中到高 | 需解码生成 | 输入理解和输出生成都强 |

这里还要区分“能不能做”和“是不是合适做”。

有些大模型也能产出 embedding，但如果你的线上目标是低延迟、高吞吐、稳定批处理，encoder-only 往往更自然。原因不是它“更先进”，而是任务形式更匹配：输入一次给全，输出一次拿全，不需要 token-by-token 解码。

真实工程里，一个非常常见的边界方案是：

- 检索系统：encoder-only embedding + encoder-only reranker
- 问答系统：先检索，再把结果交给 decoder-only 生成回答
- 摘要/翻译：直接用 encoder-decoder 或 decoder-only

所以“LLM 时代 encoder-only 是否过时”的准确答案是：**在生成任务里不是主角，但在表示学习和排序任务里仍是高性价比主力。**

---

## 参考资料

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://research.google/pubs/bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding/)  
用途：BERT 原始定义、双向预训练思想、BERT-base 110M 与 BERT-large 340M 的来源。

2. [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692)  
用途：确认 RoBERTa 的关键训练改动，包括去掉 NSP、动态 masking、更多数据和更长训练。

3. [DeBERTa: Decoding-Enhanced BERT with Disentangled Attention](https://www.microsoft.com/en-us/research/publication/deberta-decoding-enhanced-bert-with-disentangled-attention-2/)  
用途：理解 disentangled attention、增强 mask decoder，以及相对 RoBERTa 的性能提升来源。

4. [Microsoft Research: DeBERTa surpasses human performance on the SuperGLUE benchmark](https://www.microsoft.com/en-us/research/blog/microsoft-deberta-surpasses-human-performance-on-the-superglue-benchmark/)  
用途：补充 DeBERTa 的工程解释，尤其是内容/位置分离的直观描述。

5. [Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5)](https://www.microsoft.com/en-us/research/publication/text-embeddings-by-weakly-supervised-contrastive-pre-training/)  
用途：说明为什么 encoder-only 仍然是现代 embedding 主力之一。

6. [Multilingual E5 Text Embeddings: A Technical Report](https://www.microsoft.com/en-us/research/?p=1100727)  
用途：补充多语言 embedding 的训练方法与模型尺寸权衡。

7. [Towards General Text Embeddings with Multi-stage Contrastive Learning (GTE)](https://huggingface.co/papers/2308.03281)  
用途：说明 GTE 如何用多阶段对比学习强化 encoder-only 文本表示。

8. [BGE Reranker 文档](https://bge-model.com/tutorial/5_Reranking/5.2.html)  
用途：查看 encoder-only/cross-encoder reranker 的实战接口与模型规格。

9. [BGE Reranker 总览](https://bge-model.com/tutorial/5_Reranking/5.1.html)  
用途：理解两阶段检索里“先召回再重排”的标准工程流程。
