## 核心结论

Transformer 是一种用于机器翻译的序列到序列模型。序列到序列模型，指输入是一段序列，输出也是一段序列，例如输入英文句子，输出中文句子。

它的关键变化是：不再用 RNN 按时间步递归处理，也不再主要依赖 CNN 的局部卷积窗口，而是完全基于注意力机制。注意力机制，白话说，就是模型在处理某个词时，计算它应该重点参考句子里的哪些词。

新手版理解：把一句英文翻译成中文时，Transformer 不是按词一个个顺着读完再翻译，而是让每个词先“看全句”，判断自己和其他词的关系，再由解码器逐步生成译文。

输入第 $i$ 个词时，Transformer 不只使用词本身的向量，还会加入位置编码：

$$
x_i^0 = E(w_i) + PE(i)
$$

其中 $E(w_i)$ 是词 $w_i$ 的词向量，$PE(i)$ 是第 $i$ 个位置的位置编码。位置编码，白话说，就是告诉模型“这个词出现在第几个位置”。

核心注意力公式是：

$$
Attn(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

其中 query 表示“当前词想查什么”，key 表示“每个词可被匹配的索引”，value 表示“真正被取走的信息”。

RNN、CNN、Transformer 的核心差异如下：

| 模型 | 处理方式 | 长距离依赖 | 并行能力 | 机器翻译中的典型问题 |
|---|---|---:|---:|---|
| RNN | 按词递归处理 | 弱，长句容易衰减 | 弱 | 训练慢，长句信息传递困难 |
| CNN | 固定窗口卷积堆叠 | 中，需要多层扩大感受野 | 强 | 远距离词关系不够直接 |
| Transformer | 全局自注意力 | 强，任意位置可直接交互 | 强 | 注意力显存开销随序列长度平方增长 |

Transformer 彻底改变机器翻译领域的原因，不是“结构更复杂”，而是它把“全局依赖建模”和“并行训练”同时做到了工程可用。

---

## 问题定义与边界

机器翻译的目标是把源语言句子映射为目标语言句子，同时尽量保持语义、语法、语气和上下文约束。

| 概念 | 英文术语 | 白话解释 | 例子 |
|---|---|---|---|
| 源句 | source sentence | 被翻译的输入句子 | `I love machine translation.` |
| 目标句 | target sentence | 希望生成的译文 | `我喜欢机器翻译。` |
| 序列到序列 | sequence-to-sequence | 输入和输出都是序列 | 英文词序列到中文词序列 |
| 自回归 | autoregressive | 生成下一个词时依赖已经生成的词 | 先生成“我”，再生成“喜欢” |

形式化地说，给定源句 $x=(x_1,\dots,x_m)$，模型要生成目标句 $y=(y_1,\dots,y_n)$。自回归翻译通常建模为：

$$
P(y|x)=\prod_{t=1}^{n}P(y_t|y_{<t},x)
$$

含义是：第 $t$ 个目标词的生成，依赖源句 $x$ 和前面已经生成的目标词 $y_{<t}$。

玩具例子：英文 `I saw her duck` 可能表示“我看见了她的鸭子”，也可能表示“我看见她低头躲开”。这里 `duck` 是名词还是动词，不能只看单词本身，必须结合上下文判断。Transformer 的注意力机制可以让 `duck` 与 `saw`、`her` 等词交互，形成上下文相关的表示。

Transformer 解决的是“如何在可并行的前提下建模句子内部依赖”，但它不是完整翻译系统的全部。

| 能解决什么 | 不能单独解决什么 |
|---|---|
| 建模句子中任意两个位置的关系 | 低质量训练数据导致的错误翻译 |
| 并行计算编码器和训练阶段解码器 | 业务术语表、风格规范、敏感词策略 |
| 通过 mask 控制生成时的信息可见范围 | 所有长文档上下文一致性问题 |
| 支持大规模语料训练 | 评测集泄漏、线上指标失真 |

真实工程例子：一个在线 `en->zh` 翻译系统，主干模型可以是 Transformer。系统还需要分词、术语替换、批量推理、beam search、置信度估计、缓存、监控和人工评测。模型结构只是其中一层。

---

## 核心机制与推导

Transformer 翻译模型通常由编码器和解码器组成。编码器读取源句，生成上下文表示；解码器读取已经生成的目标词，并参考编码器输出，逐步生成下一个目标词。

| 阶段 | 输入 | 主要模块 | 输出 |
|---|---|---|---|
| 编码器输入 | 源语言 token | 词嵌入 + 位置编码 | 带位置的源句表示 |
| 编码器层 | 源句表示 | 自注意力 + 前馈网络 | 上下文表示 |
| 解码器输入 | 已生成目标 token | masked 自注意力 | 目标端历史表示 |
| 编码器-解码器注意力 | 目标端表示 + 源句表示 | cross-attention | 对源句对齐后的表示 |
| 输出层 | 解码器隐藏状态 | 线性层 + softmax | 下一个词概率 |

自注意力，指同一个序列内部的词相互计算注意力。交叉注意力，指目标端表示去关注源句表示。

缩放点积注意力的公式是：

$$
Attn(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

$QK^T$ 得到相关性分数，$\sqrt{d_k}$ 用来缩放分数，避免维度较大时 softmax 过于尖锐。softmax，白话说，就是把一组分数转换成和为 1 的权重。

一个最小数值例子：假设查询 $q=1$，键 $k=[1,2]$，值 $v=[10,20]$，且 $d_k=1$。分数是 $[1,2]$，softmax 约为 $[0.269,0.731]$，输出约为：

$$
0.269 \times 10 + 0.731 \times 20 = 17.31
$$

这说明注意力不是只选一个词，而是按相关性对多个词的信息做加权汇聚。

多头注意力让模型从多个子空间同时观察关系：

$$
head_j = Attn(XW_Q^j, XW_K^j, XW_V^j)
$$

$$
MHA = Concat(head_1, ..., head_h) W_O
$$

多头，白话说，就是让模型用多组不同的参数同时看句子关系。有的头可能关注主谓关系，有的头可能关注否定词，有的头可能关注短语边界。

位置前馈网络对每个位置单独做非线性变换：

$$
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
$$

它不负责词与词之间的信息交换，词间交互主要由注意力完成；它负责增强每个位置表示的表达能力。

位置编码常用正余弦形式：

$$
PE(pos, 2i) = sin(pos / 10000^{2i / d_{model}})
$$

$$
PE(pos, 2i+1) = cos(pos / 10000^{2i / d_{model}})
$$

如果没有位置编码，`dog bites man` 和 `man bites dog` 会包含相同词集合，模型难以区分词序差异。机器翻译中词序非常关键，所以位置编码不是装饰项。

解码器还必须使用因果掩码。因果掩码，白话说，就是生成第 $t$ 个词时，只允许看第 $t$ 个词之前的信息，不能偷看未来答案。

| 当前生成位置 | 可看位置 1 | 可看位置 2 | 可看位置 3 | 可看位置 4 |
|---|---:|---:|---:|---:|
| 1 | yes | no | no | no |
| 2 | yes | yes | no | no |
| 3 | yes | yes | yes | no |
| 4 | yes | yes | yes | yes |

新手版例子：句子里有 `not` 时，模型需要知道它会影响后面的词义。注意力机制负责找到 `not` 和相关词之间的联系；因果掩码负责保证解码器训练时不会提前看到标准译文后面的词。

---

## 代码实现

实现 Transformer 时，不应该把它当成一个黑盒。最小理解路径是：输入嵌入、位置编码、注意力层、多头注意力层、前馈层、编码器块、解码器块、生成流程。

常见张量形状如下：

| 名称 | 形状 | 含义 |
|---|---|---|
| token ids | `batch x seq_len` | 每个词的整数编号 |
| embedding | `batch x seq_len x d_model` | 词向量 |
| Q/K/V | `batch x heads x seq_len x d_k` | 多头注意力输入 |
| attention score | `batch x heads x seq_len x seq_len` | 每个位置对其他位置的分数 |
| mask | `1 x 1 x seq_len x seq_len` | 控制哪些位置不可见 |
| logits | `batch x seq_len x vocab_size` | 每个位置预测词表概率前的分数 |

下面是一段可运行的简化 PyTorch 代码，重点展示 attention、mask 和形状，不是完整论文实现：

```python
import math
import torch
import torch.nn.functional as F

def attention(q, k, v, mask=None):
    # q, k, v: batch x heads x seq_len x d_k
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v), weights

def causal_mask(seq_len):
    # 1 表示可见，0 表示不可见
    return torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)

batch, heads, seq_len, d_k = 2, 4, 5, 8
q = torch.randn(batch, heads, seq_len, d_k)
k = torch.randn(batch, heads, seq_len, d_k)
v = torch.randn(batch, heads, seq_len, d_k)

mask = causal_mask(seq_len)
out, weights = attention(q, k, v, mask)

assert out.shape == (batch, heads, seq_len, d_k)
assert weights.shape == (batch, heads, seq_len, seq_len)
assert torch.all(mask[0, 0].triu(1) == 0)

# 第 0 个位置不能关注未来位置，所以未来权重应接近 0
assert torch.allclose(weights[0, 0, 0, 1:], torch.zeros(seq_len - 1), atol=1e-6)

print("attention output shape:", out.shape)
```

query 像“当前词想问的问题”，key 像“每个词的索引卡”，value 像“真正要取走的内容”。模型先算 query 和 key 的相关性，再按权重汇总 value。

多头注意力在实现上通常会把 `d_model` 拆成多个头：

```text
输入: batch x seq_len x d_model
线性映射: Q, K, V
拆头: batch x heads x seq_len x d_k
注意力: batch x heads x seq_len x d_k
合并头: batch x seq_len x d_model
输出映射: batch x seq_len x d_model
```

训练阶段，目标句整体已知，但必须加 causal mask，防止模型看到未来 token。推理阶段，目标句未知，只能从 `<bos>` 开始一步步生成。

beam search 是机器翻译中常见的生成策略。beam，白话说，就是每一步不只保留一个候选译文，而是保留若干个分数最高的候选。

```text
beam_search(source, beam_size):
    encoder_output = encode(source)
    beams = [("<bos>", score=0)]

    while not all beams end with <eos>:
        candidates = []
        for text, score in beams:
            next_token_scores = decode_next(text, encoder_output)
            for token, token_score in top_k(next_token_scores, beam_size):
                candidates.append((text + token, score + token_score))

        beams = top_k(candidates, beam_size)

    return best beam without <bos>/<eos>
```

真实工程中，beam size 不是越大越好。更大的 beam 可能提高离线指标，但会增加延迟，也可能生成过短、过保守的译文。

---

## 工程权衡与常见坑

工程上最重要的不是“用了 Transformer”这个标签，而是位置编码、mask、数据、训练稳定性和评测是否正确。

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| 去掉位置编码 | 模型难以理解词序 | 保留绝对或相对位置编码 |
| 解码器不加 causal mask | 训练时偷看答案，上线性能明显下降 | 严格检查 mask 方向和形状 |
| padding mask 错误 | 模型关注 `<pad>` | 对 padding 位置置不可见 |
| 训练集混入测试集 | BLEU 虚高 | 做数据去重和泄漏检测 |
| 只看 loss | 翻译质量判断不完整 | 同时看 BLEU、COMET、人评 |
| 长句 batch 太大 | 显存爆炸 | 动态 batch、梯度累积、高效 attention |

新手版例子：如果去掉 causal mask，解码器在训练时可能“看到标准答案”。比如预测第 3 个中文词时，它已经看到了第 4、第 5 个词。训练 loss 会很好看，但上线时没有未来答案可看，效果会明显下降。

评测指标也要分层看：

| 评测方式 | 白话解释 | 优点 | 局限 |
|---|---|---|---|
| BLEU | 看候选译文和参考译文的 n-gram 重合度 | 快、标准化、历史可比 | 对语义等价但表达不同的译文不够友好 |
| COMET | 用神经模型评估译文质量 | 与人工判断相关性通常更好 | 依赖评测模型本身 |
| 人工评测 | 人直接判断准确性和流畅度 | 最贴近业务目标 | 成本高、速度慢、一致性要控制 |

显存和速度是 Transformer 工程落地的核心约束：

| 手段 | 解决什么 | 代价 |
|---|---|---|
| 减小 batch size | 降低显存 | 梯度估计更不稳定 |
| 梯度累积 | 模拟大 batch | 训练步耗时增加 |
| 混合精度 | 降低显存、提升吞吐 | 需要处理数值稳定性 |
| 高效 attention | 降低注意力计算和显存开销 | 实现和兼容性更复杂 |
| 动态长度分桶 | 减少 padding 浪费 | 数据加载逻辑更复杂 |

机器翻译训练不能只看单句表现。真实系统里，还要检查术语一致性、长句截断、标点处理、数字单位、人名地名、敏感内容和线上延迟。

WMT 是机器翻译的主流评估基准。WMT，指 Workshop on Machine Translation 相关共享任务，长期提供机器翻译数据、任务定义和评测集合。使用 WMT 测试集时，必须保证测试数据没有进入训练语料，否则指标没有可信度。

---

## 替代方案与适用边界

Transformer 不是所有场景下的最优方案。它的标准自注意力复杂度通常是 $O(n^2)$，其中 $n$ 是序列长度。句子越长，注意力矩阵越大，显存和计算成本增长越快。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 标准 Transformer | 质量强、并行好、生态成熟 | 长序列成本高 | 常规句子级翻译 |
| RNN | 结构简单、流式友好 | 并行差、长依赖弱 | 低资源、小模型、教学场景 |
| CNN | 并行好、局部模式强 | 长距离关系需堆很多层 | 固定窗口特征明显的任务 |
| 稀疏注意力 | 降低长序列成本 | 注意力模式设计复杂 | 长文档翻译 |
| 线性注意力 | 理论上更适合长序列 | 质量和稳定性依任务而定 | 超长上下文、资源受限场景 |

新手版例子：如果你的场景是超长文档翻译，标准 Transformer 可能因为注意力开销太大而不合适。此时可以考虑长上下文优化版本、分段翻译加上下文缓存、稀疏注意力，或检索增强方案。

不同约束下的选择如下：

| 场景 | 主要约束 | 更适合的方向 |
|---|---|---|
| 长句或长文档 | 上下文长度、显存 | 稀疏注意力、长上下文 Transformer、分块策略 |
| 低延迟在线翻译 | 推理速度 | 小型 Transformer、蒸馏模型、量化、缓存 |
| 低资源语言 | 平行语料少 | 多语言预训练、迁移学习、数据增强 |
| 小数据业务场景 | 过拟合风险 | 预训练模型微调、术语表、检索增强 |
| 强可解释要求 | 对齐和错误分析 | 注意力分析、人评流程、规则后处理 |

真实工程例子：企业内部技术文档翻译，如果术语固定且数据不大，直接从零训练 Transformer 往往不划算。更稳妥的方案是使用预训练翻译模型微调，再结合术语表和人工抽检。模型负责通用语言能力，工程系统负责业务约束。

选择模型时要看训练资源、推理时延、上下文长度、可解释性要求和部署复杂度，而不是只看论文最高分。Transformer 是现代机器翻译的主干结构，但完整翻译系统依赖模型、数据、解码、评测和工程控制共同成立。

---

## 参考资料

| 类别 | 资料 | 适合用途 |
|---|---|---|
| 原始论文 | Vaswani et al., *Attention Is All You Need* | 确认 Transformer 结构和公式来源 |
| 在线论文版 | ar5iv HTML 版 | 逐节阅读论文内容 |
| 工程实现 | Harvard NLP Annotated Transformer | 对照代码理解模块实现 |
| 评测标准 | WMT Translation Task | 理解机器翻译评测任务和数据集 |

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原始论文，适合确认模型结构、注意力公式和实验设置。
2. [Attention Is All You Need ar5iv HTML](https://ar5iv.labs.arxiv.org/html/1706.03762) - 在线论文版，适合按章节阅读公式和结构。
3. [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) - 工程实现讲解，适合把论文模块对应到代码。
4. [Annotated Transformer GitHub Repository](https://github.com/harvardnlp/annotated-transformer) - 参考代码仓库，适合查看可运行实现。
5. [WMT24 General Machine Translation Task](https://www2.statmt.org/wmt24/translation-task.html) - 官方评测任务页，适合理解机器翻译基准和测试集设置。
