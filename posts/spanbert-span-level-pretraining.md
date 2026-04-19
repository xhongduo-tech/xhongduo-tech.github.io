## 核心结论

SpanBERT 是一种跨度级预训练方法：它随机遮蔽连续词元跨度，并让模型只靠跨度两侧的边界表示恢复整段被遮蔽内容。

普通 BERT 的 masked language modeling，简称 MLM，核心目标是“补被遮住的 token”。token 是模型处理文本的最小单位，可以是一个词，也可以是一个子词片段。SpanBERT 的重点不是简单把多个 `[MASK]` 连在一起，而是把“连续文本片段”作为训练对象。普通 MLM 像补单词，SpanBERT 像补整段短语。前者更关心词和词之间的局部关系，后者更关心一段连续文本整体是什么。

SpanBERT 的核心由两部分组成：

| 机制 | 作用 | 关键变化 |
|---|---|---|
| span masking | 按连续跨度遮蔽文本 | 不再只随机遮蔽独立 token |
| SBO | 用边界表示预测跨度内部内容 | 迫使模型从外部上下文恢复整段语义 |
| 去掉 NSP | 不再训练下一句预测 | 采用单段文本预训练设置 |

SBO 是 Span Boundary Objective，中文可译为“跨度边界目标”。它的白话解释是：模型不能直接依赖被遮蔽位置自己的表示，而要根据左边界、右边界和相对位置，推断中间到底是什么。

这使 SpanBERT 更适合 span-centric 任务。span-centric 任务是指答案、实体、指代对象本身就是一段连续文本的任务，例如抽取式问答、共指消解和短语级表示学习。

| 任务类型 | 是否适合 SpanBERT | 原因 |
|---|---:|---|
| 抽取式问答 | 高 | 答案通常是连续文本片段 |
| 共指消解 | 高 | 需要判断多个 mention span 是否指向同一对象 |
| 实体识别 | 中到高 | 实体常是短语或连续词组 |
| 句子分类 | 中到低 | 主要依赖整体句向量，不一定需要显式 span 表示 |
| 生成任务 | 低到中 | SpanBERT 是编码器式预训练，不是生成式模型 |

SpanBERT 的准确理解是：它不是“BERT 加连续 `[MASK]`”，而是“通过跨度遮蔽和边界预测，让编码器学习更强的跨度表示”。

---

## 问题定义与边界

先定义输入序列：

$$
X = (x_1, x_2, ..., x_n)
$$

其中 $x_i$ 是第 $i$ 个 token。一个被遮蔽的跨度可以写成：

$$
Y = (x_s, x_{s+1}, ..., x_e)
$$

这里 $s$ 是跨度起点，$e$ 是跨度终点。span 的白话解释是：原文本里一段连续出现的词或 token。例如在句子 `The neural network machine` 中，`neural network` 就是一个跨度。

SpanBERT 关心的是“词级跨度”。词级跨度是先在自然语言词层面决定遮蔽哪几个词，再映射到 tokenizer 产生的子词 token。subword token 是分词器为了处理生僻词和词形变化而切出来的子词片段，例如 `playing` 可能被切成 `play` 和 `##ing`。SpanBERT 不希望随机只遮住一个词的一半，因为那会让任务变成补词形碎片，而不是理解完整短语。

| 对比项 | 词级跨度 | 子词级 token |
|---|---|---|
| 基本单位 | 自然词 | tokenizer 输出片段 |
| 示例 | `neural network` | `ne`, `##ural`, `network` 等可能片段 |
| 训练目标 | 恢复完整短语结构 | 恢复局部 token |
| 主要风险 | 需要词到 token 的对齐 | 容易遮住半个词，任务变形 |

玩具例子：句子 `The neural network machine`，如果遮蔽的是 `neural network`，SpanBERT 的目标不是分别猜 `neural` 和 `network` 两个独立词，而是根据 `The` 与 `machine` 判断中间大概率是一个描述机器类型的名词短语。

真实工程例子：抽取式问答系统中，问题是 `Who won Super Bowl 50?`，文章里答案是 `Denver Broncos`。答案不是一个分类标签，而是原文中的连续 span。模型需要判断哪个起点和终点组成最合适的答案片段。SpanBERT 的训练目标与这种下游任务更一致。

SpanBERT 的边界也要说清楚。它解决的是“如何让编码器更懂连续片段”，不是保证所有 NLP 任务都提升。

| 任务 | SpanBERT 的价值 | 边界 |
|---|---|---|
| 抽取式问答 | 学到更强答案 span 表示 | 仍需要 start/end 打分头 |
| 共指消解 | 更好表示 mention span | 仍依赖候选生成和聚类策略 |
| NER | 有助于实体短语表示 | 短实体任务收益可能有限 |
| 文本分类 | 可能有帮助 | 不一定优于普通强编码器 |
| 检索召回 | 不一定明显 | 通常更依赖句向量或双塔训练 |

因此，SpanBERT 的问题定义不是“改进 BERT 的所有能力”，而是“在预训练阶段显式学习跨度表示”。

---

## 核心机制与推导

SpanBERT 的训练可以拆成两步：先采样连续跨度并遮蔽，再用 SBO 从边界恢复跨度内部内容。

第一步是 span masking。SpanBERT 不是独立采样若干 token，而是采样若干连续词跨度。跨度长度近似来自几何分布：

$$
l \sim Geo(p)
$$

论文中常用 $p = 0.2$，并把最大跨度长度截断到 10。几何分布的白话解释是：它会产生较多短跨度，也保留一部分较长跨度。这样训练数据里既有短实体，也有多词短语。

例如原句：

```text
The neural network machine works well.
```

采样跨度：

```text
neural network
```

mask 后输入：

```text
The [MASK] [MASK] machine works well.
```

真实监督目标是：

```text
neural network
```

第二步是 SBO。SpanBERT 不让模型直接使用跨度内部每个 `[MASK]` 位置的最终表示去预测对应词，而是只使用跨度外部边界表示。设被遮蔽跨度是 $(x_s, ..., x_e)$，左边界是 $x_{s-1}$，右边界是 $x_{e+1}$。对跨度内部第 $i$ 个位置，构造：

$$
y_i = f([x_{s-1}; x_{e+1}; p_i])
$$

其中 $p_i$ 是相对位置嵌入，表示当前位置在跨度内部的第几个位置；$f$ 是两层前馈网络；`[;]` 表示向量拼接。相对位置嵌入的白话解释是：同样在 `The ___ ___ machine` 中，第一个空和第二个空角色不同，模型需要知道当前要预测的是第几个空。

SBO 损失可以写成：

$$
L_{SBO} = \sum_{i=s}^{e} CE(softmax(Wy_i), x_i)
$$

其中 CE 是交叉熵损失。交叉熵的白话解释是：真实词概率越高，损失越低；真实词概率越低，损失越高。总损失可写成：

$$
L = L_{MLM} + L_{SBO}
$$

一个最小数值例子：句子是 `The [MASK] [MASK] machine`，真实跨度是 `neural network`。边界 token 是 `The` 和 `machine`。假设模型对第一个位置预测 $P(neural)=0.8$，第二个位置预测 $P(network)=0.6$，则：

$$
L_{SBO} = -\ln(0.8) - \ln(0.6) \approx 0.223 + 0.511 = 0.734
$$

这个损失表达的是：如果边界表示足够理解中间短语，模型就能给真实词更高概率，损失就更低。

SpanBERT 还去掉了 BERT 的 NSP。NSP 是 Next Sentence Prediction，中文可译为“下一句预测”，目标是判断两个句子是否相邻。SpanBERT 采用单段输入训练，避免把训练能力分散到句间二分类任务上。对 span-centric 任务来说，跨度内部语义和边界上下文通常比“这两句是否相邻”更直接。

机制有效的原因可以概括为一句话：span masking 增加了被恢复对象的结构性，SBO 限制了可用信息来源，迫使模型把边界和内部短语之间的关系学进表示里。

---

## 代码实现

实现 SpanBERT 思路时，可以拆成三层：跨度采样、输入构造、SBO 预测头。

预训练流程伪代码如下：

```text
for sentence in corpus:
    words = split_to_words(sentence)
    spans = sample_word_spans(words)
    tokens, align = tokenize_and_align(words)
    masked_tokens = replace_span_tokens_with_mask(tokens, spans, align)

    hidden = encoder(masked_tokens)

    for each masked span:
        left = hidden[start - 1]
        right = hidden[end + 1]
        for position inside span:
            y = FFN(concat(left, right, relative_position_embedding))
            loss += CE(vocab_projection(y), original_token)
```

一个数据处理表：

| 原句 | 采样跨度 | mask 后输入 | 边界 token | 监督目标 |
|---|---|---|---|---|
| `The neural network machine works` | `neural network` | `The [MASK] [MASK] machine works` | `The`, `machine` | `neural`, `network` |
| `Denver Broncos won the game` | `Denver Broncos` | `[MASK] [MASK] won the game` | `[BOS]`, `won` | `Denver`, `Broncos` |

下面是一个简化但可运行的 Python 代码块，用来演示 span 采样、SBO 输入构造和损失计算。它不是完整神经网络训练代码，但保留了关键数据结构和损失逻辑。

```python
import math
import random

def sample_span(words, max_len=10, p=0.2, rng=None):
    rng = rng or random.Random(0)
    start = rng.randrange(0, len(words))
    length = 1
    while length < max_len and start + length < len(words) and rng.random() > p:
        length += 1
    end = start + length - 1
    return start, end

def build_sbo_input(words, start, end, mask_token="[MASK]"):
    masked = list(words)
    targets = words[start:end + 1]
    for i in range(start, end + 1):
        masked[i] = mask_token

    left_boundary = words[start - 1] if start > 0 else "[BOS]"
    right_boundary = words[end + 1] if end + 1 < len(words) else "[EOS]"
    relative_positions = list(range(1, len(targets) + 1))

    return {
        "masked": masked,
        "left_boundary": left_boundary,
        "right_boundary": right_boundary,
        "relative_positions": relative_positions,
        "targets": targets,
    }

def compute_sbo_loss(targets, predicted_probs):
    loss = 0.0
    for target in targets:
        prob = predicted_probs[target]
        loss += -math.log(prob)
    return loss

words = "The neural network machine works".split()
start, end = 1, 2
example = build_sbo_input(words, start, end)

assert example["masked"] == ["The", "[MASK]", "[MASK]", "machine", "works"]
assert example["left_boundary"] == "The"
assert example["right_boundary"] == "machine"
assert example["targets"] == ["neural", "network"]

loss = compute_sbo_loss(
    example["targets"],
    {"neural": 0.8, "network": 0.6}
)

assert round(loss, 3) == 0.734
```

在真实模型里，`left_boundary` 和 `right_boundary` 不是字符串，而是 Transformer 编码后的向量。Transformer 是一种基于注意力机制的神经网络结构，白话解释是：它能让每个 token 根据整段上下文更新自己的表示。SBO 预测头通常接收三个向量：左边界表示、右边界表示、相对位置嵌入，然后经过前馈网络和词表投影得到每个词的概率分布。

下游问答通常还要设计 span scoring。一个简化接口可以写成：

```python
def score_answer_span(hidden, start, end, scorer):
    start_vec = hidden[start]
    end_vec = hidden[end]
    span_vec = start_vec + end_vec
    return scorer(span_vec)
```

这个接口表达的是：候选答案不是单个 token，而是由 start 和 end 组成的连续片段。SpanBERT 的预训练刚好强化了边界表示对内部短语的概括能力，因此更容易迁移到这类任务。

---

## 工程权衡与常见坑

SpanBERT 的收益集中在 span-centric 任务，不能默认迁移到所有任务。复现时最常见的错误，是只做连续 mask，但没有实现 SBO，也没有调整训练设置。

| 错误做法 | 正确做法 | 影响 |
|---|---|---|
| 只把连续 token 换成 `[MASK]` | 同时实现 span masking 和 SBO | 否则只是普通 MLM 的轻微变体 |
| 按 subword 随机采样跨度 | 先按词级采样，再映射到 subword | 避免遮住半个词 |
| 继续使用 NSP | 使用单段训练设置 | 减少与跨度目标无关的训练信号 |
| 下游仍只用 `[CLS]` | 问答和共指中显式建模 span | 预训练收益无法充分释放 |
| 期待分类任务大幅提升 | 根据任务是否 span-centric 判断 | 避免错误选型 |

一个典型失败案例是：把 `The neural network machine` 中的 `neural network` 换成两个 `[MASK]`，但仍然让每个 `[MASK]` 位置用自己的最终 hidden state 做 MLM 预测。这通常无法复现 SpanBERT 的主要收益，因为模型仍可以依赖 `[MASK]` 位置的上下文混合表示，而不是被迫从边界压缩出整段信息。

另一个坑是“span”和“token”混用。假设一个词被 tokenizer 切成多个 subword，如果直接在 subword 级别随机选跨度，可能只遮住词的一部分。这样模型学到的可能是词形补全，而不是短语恢复。SpanBERT 的重点是让模型恢复自然语言中的连续片段，因此采样粒度要尽量贴近词和短语。

训练配置至少要检查这些项：

| 配置项 | 建议 |
|---|---|
| mask 粒度 | 词级 span，而不是独立 token |
| span 长度 | 近似几何分布，常用 $p=0.2$，最大长度 10 |
| SBO | 使用边界表示和相对位置预测内部 token |
| NSP | 关闭 |
| 下游任务 | 问答、共指、实体任务优先验证 |
| 评估方式 | 不只看预训练 loss，要看下游 span 任务指标 |

真实工程里还要考虑成本。SpanBERT 的训练复杂度和实现复杂度都高于直接使用现成 BERT。若团队只是做评论情感分类、主题分类或粗粒度检索，普通预训练编码器可能已经足够。若系统核心指标来自抽取式问答、文档理解、实体链接或共指消解，SpanBERT 的机制才更值得投入。

---

## 替代方案与适用边界

SpanBERT 不是唯一的 span 表示方法。它的优势在于：在预训练阶段就显式学习“用边界恢复跨度”。如果下游任务不依赖连续片段，其他方法可能更简单。

| 方法 | 核心目标 | 优势 | 适用边界 |
|---|---|---|---|
| BERT MLM | 随机预测被遮蔽 token | 简单、通用、生态成熟 | span 表示不是训练重点 |
| SpanBERT | 从边界恢复被遮蔽跨度 | 对问答和共指更直接 | 实现和复现更复杂 |
| RoBERTa 类方法 | 强化 MLM 训练策略 | 通用性能强 | 不专门建模 span 边界 |
| T5/BART 类去噪模型 | 重建被破坏文本 | 更适合生成和序列到序列任务 | 架构目标不同 |
| 任务内 span pooling | 下游阶段聚合 span | 实现成本低 | 缺少预训练阶段的显式约束 |

适用任务矩阵如下：

| 任务 | 推荐程度 | 说明 |
|---|---:|---|
| 抽取式问答 | 高 | 答案天然是 span |
| 共指消解 | 高 | mention 表示依赖边界和内部内容 |
| NER | 中到高 | 实体 span 有明确边界 |
| 文本分类 | 中低 | `[CLS]` 或句向量通常更关键 |
| 粗粒度语义检索 | 中低 | 双塔、对比学习可能更直接 |
| 摘要/翻译 | 低 | 更适合生成式或 encoder-decoder 模型 |

选择建议可以直接写成三条：

第一，如果任务输出就是原文中的连续片段，优先考虑 SpanBERT 或至少使用 span-aware 表示方式。

第二，如果任务只是句子级分类，不要为了 SBO 引入额外复杂度，先用普通强编码器建立基线。

第三，如果任务需要生成新文本，SpanBERT 不是最直接选择，T5、BART 或其他生成式预训练模型通常更合适。

SpanBERT 的价值不是取代所有预训练方法，而是把“跨度理解”这个目标提前放进预训练阶段。它适合的问题越接近“从上下文中识别、恢复、比较连续文本片段”，收益越可能稳定。

---

## 参考资料

1. [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://a11y2.apps.allenai.org/paper?id=81f5810fbbab9b7203b9556f4ce3c741875407bc)：论文 HTML，用于核对 span masking、SBO、去掉 NSP 等机制。
2. [facebookresearch/SpanBERT](https://github.com/facebookresearch/SpanBERT)：官方代码仓库，用于核对模型发布、任务脚本和复现入口。
3. [Princeton publication page: SpanBERT](https://collaborate.princeton.edu/en/publications/spanbert-improving-pre-training-by-representing-and-predicting-sp)：论文发布页，用于核对论文元信息和发表信息。
4. [Hugging Face Papers: 1907.10529](https://huggingface.co/papers/1907.10529)：论文索引页，用于快速查看摘要、引用入口和社区元信息。
