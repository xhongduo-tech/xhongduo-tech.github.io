## 核心结论

生成式文本摘要是一个条件序列生成问题：给定源文档 $x$，模型逐 token 生成摘要 $y$，目标是学习：

$$
p(y|x)=\prod_{t=1}^{T}p(y_t|y_{<t},x)
$$

这里的 token 是模型处理文本的最小单位，可以是字、词，也可以是子词。生成式摘要不是从原文里挑几句话，而是根据原文重新写出更短的表达。

新手版可以先这样理解：输入一篇很长的文章，模型先读完整篇内容，再写出一段更短的话。它有时会改写原文，例如把“用户多次尝试支付但均失败”写成“用户支付失败”；有时会把订单号、药名、模型名、错误码这种不能乱改的词原样抄回来。

| 维度 | 抽取式摘要 | 生成式摘要 |
|---|---|---|
| 基本做法 | 从原文中选择句子或片段 | 逐 token 生成新摘要 |
| 是否重写 | 通常不重写 | 可以重写、压缩、融合 |
| 是否能引入新句子 | 不能，只能选原文内容 | 能生成原文中没有的句式 |
| 优点 | 可解释、事实风险低 | 表达自然、压缩能力强 |
| 主要风险 | 摘要可能生硬、冗余 | 可能幻觉、改错实体 |
| 典型模型 | BERTSUM | Pointer-Generator、BART、PEGASUS |

核心判断很简单：如果任务只需要“选出重要原句”，抽取式更稳；如果任务需要“改写、概括和重组信息”，生成式更合适。`BART`、`PEGASUS` 属于典型生成式路线，`BERTSUM` 主要是抽取式对照模型。

---

## 问题定义与边界

设源文档为：

$$
x=(x_1,x_2,\dots,x_n)
$$

摘要为：

$$
y=(y_1,y_2,\dots,y_T)
$$

其中 $n$ 是原文长度，$T$ 是摘要长度。生成式摘要的输出不是一次性得到整段文字，而是在第 $t$ 步根据原文和已经生成的前缀 $y_{<t}$，预测下一个 token $y_t$。

| 符号 | 含义 |
|---|---|
| $x$ | 源文档 |
| $x_i$ | 源文档第 $i$ 个 token |
| $n$ | 源文档 token 数 |
| $y$ | 摘要 |
| $y_t$ | 摘要第 $t$ 个 token |
| $T$ | 摘要 token 数 |
| $h_i$ | 编码器对 $x_i$ 产生的上下文表示 |
| $s_t$ | 解码器第 $t$ 步的隐藏状态 |

编码器是把输入文本转换成向量表示的模块；解码器是根据这些向量一步步生成输出文本的模块。这个定义决定了训练、解码和评测都要按序列生成问题处理。

任务边界必须先说清楚。生成式摘要追求流畅、压缩和信息融合，但它不天然保证事实正确。模型可能把“未退款”写成“已退款”，也可能把型号、金额、日期改错。因此，生成式摘要适合“表达可以重写，但事实不能乱变”的任务，不适合没有校验机制的高风险决策。

| 场景 | 是否适合生成式摘要 | 原因 |
|---|---|---|
| 新闻标题式摘要 | 适合 | 需要压缩、改写、突出主旨 |
| 客服工单摘要 | 适合，但要保留实体 | 多轮对话需要压成一行，订单号和错误码不能错 |
| 产品评论总结 | 适合 | 多条评论可以融合成观点摘要 |
| 法律证据摘要 | 更适合抽取式或混合式 | 需要可追溯、可引用原文 |
| 医疗诊断摘要 | 不适合完全自动生成 | 事实风险高，必须有人工复核 |
| 逐句对齐报告 | 更适合抽取式 | 强调来源定位和解释性 |

玩具例子：原文是“张三在 3 月 1 日提交退款申请，客服在 3 月 2 日审核通过，退款金额为 128 元。”生成式摘要可以写成“张三的 128 元退款已于 3 月 2 日审核通过。”这句话不完全来自原文，但保留了关键事实。

真实工程例子：客服工单系统里，输入可能是几十轮聊天记录，输出是一行摘要给下一位坐席，例如“用户订单 A1024 支付失败，错误码 E503，多次重试无效，需排查支付网关。”这里“订单 A1024”和“E503”必须复制原文，不能自由改写。

---

## 核心机制与推导

生成式摘要的基本结构是编码器-解码器。编码器读取全文，得到每个位置的表示 $h_i$；解码器在第 $t$ 步生成摘要 token，并通过注意力机制查看原文重点位置。

注意力机制是一种“给输入不同位置分配权重”的方法。权重越大，表示当前生成步骤越关注该位置：

$$
c_t=\sum_i \alpha_{t,i}h_i
$$

其中 $\alpha_{t,i}$ 是第 $t$ 步对第 $i$ 个源 token 的注意力权重，$c_t$ 是上下文向量。

生成式摘要的关键不只在“会生成”，还在“会复制、会控重复、会保真”。指针网络允许模型从原文复制词，适合处理罕见词、专有名词、数字和代码。复制门 $p_{gen}$ 决定当前更偏向从词表生成，还是从原文复制：

$$
p_{gen}=\sigma(w_c^Tc_t+w_s^Ts_t+w_y^Te(y_{t-1})+b)
$$

最终词分布为：

$$
P(w)=p_{gen}P_{vocab}(w)+(1-p_{gen})\sum_{i:x_i=w}\alpha_{t,i}
$$

这里 $P_{vocab}(w)$ 是从固定词表生成词 $w$ 的概率；后半部分是从原文中所有等于 $w$ 的位置复制的概率总和。

玩具数值例子：假设某一步 $p_{gen}=0.7$，词表里 $P_{vocab}(\text{"总结"})=0.25$，则：

$$
P(\text{"总结"})=0.7\times0.25=0.175
$$

如果原文中专有名词 `"BERTSUM"` 当前注意力总和为 $0.30$，则复制概率为：

$$
P(\text{"BERTSUM"})=(1-0.7)\times0.30=0.09
$$

覆盖机制用于记录模型已经关注过哪些原文位置，避免反复写同一段信息。覆盖向量定义为：

$$
cov_{t,i}=\sum_{\tau<t}\alpha_{\tau,i}
$$

覆盖损失可以写成：

$$
L=-\sum_t\log P(y_t^\*)+\lambda\sum_t\sum_i\min(\alpha_{t,i},cov_{t,i})
$$

如果上一时刻注意力 $\alpha_1=(0.6,0.3,0.1)$，当前注意力 $\alpha_2=(0.2,0.5,0.3)$，则 $cov_2=(0.6,0.3,0.1)$，覆盖损失项为：

$$
\min(0.2,0.6)+\min(0.5,0.3)+\min(0.3,0.1)=0.6
$$

机制图可以写成：

```text
源文档 x
  |
  v
编码器 Encoder -> h_1, h_2, ..., h_n
  |
  v
注意力 Attention -> alpha_t -> 上下文向量 c_t
  |                              |
  |                              v
  |                        复制分布 Copy
  v                              |
解码器 Decoder -> s_t -> 生成分布 P_vocab
  |                              |
  +---------- p_gen 复制门 -------+
                 |
                 v
       最终分布 P(w) -> y_t
                 |
                 v
       覆盖向量 cov_t 抑制重复
```

预训练模型把这种结构做得更强。`BART` 通过“加噪再复原”学习重写能力：

$$
L_{BART}=-\log p(x|\tilde{x})
$$

其中 $\tilde{x}$ 是被打乱、遮盖或删除部分内容后的文本。`PEGASUS` 通过“挖空关键句再生成”学习摘要能力：

$$
L_{PEGASUS}=-\log p(G|x\setminus G)
$$

其中 $G$ 是从文档中抽出的关键句集合。这个目标更接近摘要任务，因为模型要根据剩余文本生成被挖掉的重要句子。

---

## 代码实现

实现生成式摘要时，数据流通常拆成七步：读入原文、编码、解码、计算词表分布、计算复制分布、计算损失、执行解码策略。训练阶段和推理阶段要分开：训练常用 teacher forcing，也就是把真实摘要前缀喂给解码器；推理阶段没有真实前缀，只能用模型上一步生成的结果继续生成。

下面是一个可运行的最小 Python 例子，用纯标准库模拟 `p_gen`、复制分布、覆盖更新和 coverage loss。它不是完整神经网络，但保留了指针生成摘要的核心计算形状。

```python
from math import exp, log

def sigmoid(x):
    return 1 / (1 + exp(-x))

def encode(source_tokens):
    return [{"token": tok, "pos": i} for i, tok in enumerate(source_tokens)]

def attention(step, encoded):
    if step == 1:
        weights = [0.6, 0.3, 0.1]
    else:
        weights = [0.2, 0.5, 0.3]
    assert len(weights) == len(encoded)
    assert abs(sum(weights) - 1.0) < 1e-9
    return weights

def copy_distribution(source_tokens, attn):
    dist = {}
    for tok, weight in zip(source_tokens, attn):
        dist[tok] = dist.get(tok, 0.0) + weight
    return dist

def coverage_update(prev_coverage, attn):
    return [c + a for c, a in zip(prev_coverage, attn)]

def decode_step(source_tokens, vocab_dist, attn, p_gen):
    copy_dist = copy_distribution(source_tokens, attn)
    final = {}

    for word, prob in vocab_dist.items():
        final[word] = final.get(word, 0.0) + p_gen * prob

    for word, prob in copy_dist.items():
        final[word] = final.get(word, 0.0) + (1 - p_gen) * prob

    return final

def loss_fn(target_word, final_dist, attn, coverage, lam=1.0):
    nll = -log(final_dist[target_word])
    cov_loss = sum(min(a, c) for a, c in zip(attn, coverage))
    return nll + lam * cov_loss

source = ["用户", "BERTSUM", "失败"]
encoded = encode(source)

coverage = [0.0, 0.0, 0.0]
attn1 = attention(1, encoded)
coverage = coverage_update(coverage, attn1)

attn2 = attention(2, encoded)
p_gen = sigmoid(0.8473)  # 约等于 0.7
vocab_dist = {"总结": 0.25, "失败": 0.10}

final_dist = decode_step(source, vocab_dist, attn2, p_gen)
loss = loss_fn("BERTSUM", final_dist, attn2, coverage)

assert round(p_gen, 1) == 0.7
assert round(final_dist["总结"], 3) == 0.175
assert round(final_dist["BERTSUM"], 3) == 0.09
assert round(sum(min(a, c) for a, c in zip(attn2, coverage)), 1) == 0.6
assert loss > 0
```

| 阶段 | 输入 | 输出 | 是否使用真值前缀 | 是否更新覆盖 |
|---|---|---|---|---|
| 训练 | 原文 + 真实摘要前缀 | 每步目标 token 概率 | 是 | 是 |
| 推理 | 原文 + 已生成摘要 | 下一个 token | 否 | 是 |
| 贪心解码 | 当前概率最大 token | 单条摘要 | 否 | 是 |
| Beam Search | 多个候选前缀 | 得分最高摘要 | 否 | 是 |

工程实现里最容易出问题的是维度和词表外词。注意力维度必须和源文档长度对齐；覆盖向量必须按解码步累计；复制分布要能把同一个词在原文多个位置的注意力相加；中文还要处理分词和子词切分，否则模型可能把专有名词切碎后改错。

---

## 工程权衡与常见坑

第一个权衡是流畅性和忠实度。$p_{gen}$ 高时，模型更像自由改写，摘要会更自然，但幻觉风险更高；$p_{gen}$ 低时，模型更像复述原文，实体更稳，但摘要可能不够简洁。第二个权衡是长度和信息密度。长度约束太松会啰嗦，太紧会截断关键事实。

真实工程例子：客服摘要中，原文可能包含“用户订单 A1024 在 iOS 17.2 下支付失败，错误码 E503，用户重试三次仍失败”。合格摘要可以是“订单 A1024 在 iOS 17.2 支付失败，错误码 E503，多次重试无效。”如果模型把“重试失败”写两遍，说明覆盖机制或解码约束不够；如果把 E503 改成 E530，说明实体保护失败。

| 问题现象 | 根因 | 规避方法 | 常用手段 |
|---|---|---|---|
| 幻觉 | 模型自由生成了原文没有的事实 | 限制生成自由度，做事实校验 | 复制机制、检索增强、一致性检查 |
| 重复 | 多次关注同一原文位置 | 惩罚重复注意力或重复片段 | coverage loss、trigram blocking |
| 专有名词拼错 | 子词切分或生成分布不稳定 | 让关键实体走复制路径 | 实体识别、约束解码 |
| 摘要太长 | 长度惩罚不足 | 明确最大长度和压缩率 | max length、length penalty |
| 摘要太短 | 解码过早输出结束符 | 调整最小长度 | min length、结束符惩罚 |
| 只看 ROUGE | n-gram 重合不等于事实正确 | 加字段级和人工评测 | factuality、人审、业务指标 |
| 把 BERTSUM 当生成式 | 混淆抽取式和生成式 | 按任务目标选模型 | BERTSUM 用于选句，BART 用于生成 |

| 手段 | 主要作用 | 不能解决的问题 |
|---|---|---|
| coverage loss | 减少重复关注 | 不能保证事实正确 |
| trigram blocking | 阻止重复三元片段 | 可能误伤合理重复 |
| 长度惩罚 | 控制摘要长短 | 不能判断信息是否完整 |
| 实体约束解码 | 保留订单号、药名、型号 | 需要额外实体识别或规则 |

一个常见误区是认为“生成式模型越大，事实越可靠”。这不成立。更大的模型通常语言能力更强，但只要目标仍是生成下一个 token，它就可能在不确定时补全一个看似合理的事实。业务落地时，摘要模型后面通常要接事实一致性检查、实体校验、人工抽检或置信度策略。

---

## 替代方案与适用边界

如果任务只要求选出原文中最重要的句子，抽取式方法更简单、可控、可解释，`BERTSUM` 这类方法就足够。如果任务要求改写、压缩、融合多句信息，生成式模型更合适，例如 `BART`、`PEGASUS` 或带复制机制的 Seq2Seq 模型。

| 方案 | 可解释性 | 事实一致性 | 压缩率 | 改写自由度 | 开发成本 | 适用边界 |
|---|---|---|---|---|---|---|
| 抽取式 | 高 | 较高 | 中 | 低 | 低到中 | 选新闻关键句、法务材料摘录 |
| 生成式 | 中到低 | 中 | 高 | 高 | 中到高 | 客服摘要、标题生成、评论总结 |
| 混合式 | 中到高 | 较高 | 中到高 | 中 | 高 | 先抽关键证据，再生成短摘要 |
| 模板式 | 高 | 高 | 低到中 | 低 | 中 | 工单字段固定、报告结构稳定 |
| 检索增强摘要 | 中 | 较高 | 中到高 | 中 | 高 | 需要引用来源和减少幻觉 |

新手可以按三个问题决策：

| 问题 | 更推荐的方案 |
|---|---|
| 摘要必须能追溯到原文句子吗 | 抽取式或混合式 |
| 摘要需要把多句话合成一句吗 | 生成式 |
| 错一个数字就会产生严重后果吗 | 模板式、抽取式或人工复核 |
| 是否有大量训练数据 | 有数据可用生成式，数据少先用抽取式或模板式 |
| 是否需要保留订单号、药名、型号 | 生成式加复制机制或约束解码 |

`BART` 的适用边界是通用生成和改写任务，它通过加噪复原学到了较强的语言重建能力。`PEGASUS` 的适用边界更偏摘要，因为它的预训练目标直接模拟“根据上下文生成关键句”。`BERTSUM` 的适用边界是抽取式摘要，它擅长判断哪些句子重要，但不是逐 token 写出新摘要。

新闻报道里，“挑两句关键原文”适合抽取式；客服工单里，“把多轮对话压成一行”适合生成式；医疗和法务场景通常需要更强约束或人工复核。工程上不必把所有问题都交给全生成模型。很多稳定系统会采用“抽取关键证据 + 生成短摘要 + 规则校验”的混合方案。

---

## 参考资料

1. [Get To The Point: Summarization with Pointer-Generator Networks](https://aclanthology.org/P17-1099/)
2. [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://aclanthology.org/2020.acl-main.703/)
3. [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://research.google/pubs/pegasus-pretraining-with-extracted-gap-sentences-for-abstractive-summarization-by-sequence-to-sequence-models/)
4. [Fine-tune BERT for Extractive Summarization](https://huggingface.co/papers/1903.10318)
5. [Pointer-generator 代码仓库](https://github.com/abisee/pointer-generator)
6. [BERTSUM 代码仓库](https://github.com/nlpyang/BertSum)
