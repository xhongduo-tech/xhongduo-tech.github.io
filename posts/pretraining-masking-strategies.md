## 核心结论

预训练语言模型的掩码策略，指的是训练时故意删掉输入中的一部分内容，再让模型恢复原文。它决定了模型到底在学什么单位：是单个 token，还是连续短语、实体片段、整段局部结构。

BERT 的典型做法是随机挑出约 15% 的 token 做单点扰动。这个策略简单、稳定，适合 encoder-only 模型，也适合分类、检索、序列标注这类“理解优先”的任务。SpanBERT、MASS、T5 则把掩码单位扩展到 span，也就是连续的一段 token。白话解释：不是挖掉几个零散单词，而是直接挖掉一个短语或局部片段，让模型练习“补整段”。

这种差异会直接影响模型能力。单 token 掩码更像局部填空，模型容易依赖邻近词完成预测；span 掩码要求模型跨更长上下文恢复语义关系，更适合学习实体边界、短语结构和生成式重建。掩码率、span 长度分布、是否动态重采样，会共同决定训练难度、样本效率和下游迁移效果。

一个最直接的判断是：如果目标模型主要做判别任务，小模型或中等模型常常用 BERT 风格就够；如果模型要处理摘要、翻译、问答生成、长实体建模，span 级策略通常更合适。

| 方法 | 掩码粒度 | 采样方式 | 预测目标 | 更擅长的能力 |
|---|---|---|---|---|
| BERT | 单 token | 随机独立采样 | 被掩码 token 分类 | 词级语义、分类表征 |
| SpanBERT | 连续 span | span 长度分布采样 | span 内 token 重建 | 实体、短语、边界信息 |
| MASS | 连续片段 | 单段或局部连续段 | decoder 预测缺失片段 | seq2seq 生成、翻译 |
| T5 | span corruption | 多个 span + sentinel | decoder 只生成缺失 span 序列 | 通用文本到文本生成 |

---

## 问题定义与边界

掩码策略的核心问题不是“要不要遮住内容”，而是“遮住什么、遮住多少、以什么方式恢复”。这里至少有三个设计轴。

第一，掩码粒度。token 是分词后的最小训练单位，可能是词、子词或字符片段。单 token 掩码假设局部词级预测足够训练语言理解；span 掩码则假设很多真实语言现象是以连续片段出现的，比如人名、术语、动词短语、固定搭配。

第二，掩码预算。预算就是本轮训练允许破坏多少输入信息，常用公式是：

$$
\text{mask\_rate}=\frac{\text{masked tokens}}{\text{sequence length}}
$$

如果一个样本长度是 500，掩码率设为 15%，则预算是：

$$
500 \times 0.15 = 75
$$

也就是总共要扰动 75 个 token。

第三，span 切分方式。如果决定用 span 掩码，还要回答“这些被掩码 token 分成多少段”。平均 span 长度可以写成：

$$
L_{\text{avg}}=\frac{\text{masked tokens}}{\text{span count}}
$$

如果 75 个 token 被分成 25 个 span，则平均长度是 3。白话解释：总共要挖掉 75 个词，如果平均每段 3 个词，就大约要挖 25 个坑。

这里的边界也要说清楚。

BERT 的 15% 和 80/10/10 不是数学定律，而是早期工程经验。80/10/10 指被选中的 token 中，80% 真替换成 `[MASK]`，10% 换成随机词，10% 保持原样。这样做是为了减轻预训练和下游微调之间的输入分布差异，因为推理时通常不会出现 `[MASK]`。

但一旦模型变大、任务变复杂，这个预算可能太保守。掩码太少，模型看到的上下文过完整，恢复任务过于容易，训练信号不够强。反过来，掩码过多也会让输入信息塌缩，模型难以定位语义。掩码策略本质上是在“信息破坏强度”和“可恢复性”之间找平衡。

一个玩具例子：

原句：`我 昨天 在 北京 大学 参加 机器学习 讲座`

如果只做单 token 掩码，可能变成：

`我 昨天 在 [MASK] 大学 参加 机器学习 讲座`

这时模型主要学会“北京”这个单点恢复。

如果做 span 掩码，可能变成：

`我 昨天 在 [MASK] [MASK] 参加 机器学习 讲座`

或者 T5 风格：

`我 昨天 在 <extra_id_0> 参加 机器学习 讲座`

目标输出是：`<extra_id_0> 北京 大学`

两者训练难度不同。前者更像词汇猜测，后者更像短语恢复。

---

## 核心机制与推导

从目标函数看，单 token 掩码和 span 掩码都属于去噪学习，也就是把原始序列破坏后再恢复。区别在于恢复对象的结构。

设原序列为 $s$，被掩码的连续片段为 $z$，那么 span 去噪目标可以写成：

$$
\mathcal{L}(s,z)=-\log p_\theta(z\mid s\setminus z)
$$

其中 $s\setminus z$ 表示把 span 去掉后的剩余上下文。白话解释：模型只看没被删掉的部分，然后尽量把被删掉的那一段补回来。

如果 $z$ 只包含一个 token，这就退化成 BERT 式的 token 级恢复。如果 $z$ 是连续短语，模型必须同时回答三个问题：

1. 缺失内容大概属于什么语义单元。
2. 这个单元和左右边界如何对齐。
3. 片段内部多个 token 之间如何组合。

这就是 span 掩码更适合实体和短语建模的原因。因为现实语言中的很多关键语义本来就是连续出现的，例如“纽约证券交易所”“卷积神经网络”“低资源机器翻译”。

SpanBERT 在这个方向上的价值，是把预训练目标从“离散单词预测”推向“连续语义片段恢复”。T5 更进一步，不要求 encoder 在原位逐个输出被掩码 token，而是把每个被删掉的 span 替换成一个 sentinel token。sentinel token 可以理解为“占位编号”，例如 `<extra_id_0>`、`<extra_id_1>`。decoder 只生成这些缺失 span 的串联结果。

例如原句：

`The cat sat on the red mat near the window`

T5 可能构造为：

输入：`The cat sat <extra_id_0> mat <extra_id_1> window`

目标：`<extra_id_0> on the red <extra_id_1> near the`

好处有两个。

第一，decoder 不需要重新生成整句，只生成缺失内容，计算更集中。

第二，多个 span 可以共享同一个 seq2seq 框架，天然适合后续摘要、翻译、问答生成这类任务。

MASS 的思路和它接近，但更偏向 seq2seq 预训练：encoder 看带缺口的源序列，decoder 专门预测那段缺失片段。这对机器翻译尤其自然，因为翻译本来就要求在条件上下文下生成连续序列。

真实工程例子可以看无监督或低资源翻译。假设语料里有大量法语和英语单语文本，但没有足够平行语料。MASS 先让模型在法语句子内部练习“挖掉连续片段再补全”，再在英语句子上做同样训练。这样 encoder-decoder 会学会连续片段的编码和解码，迁移到翻译时，decoder 已经具备“根据上下文生成一段连续文本”的能力，因此比只做 token 级 MLM 更贴近下游目标。

从信息论角度理解，掩码率和 span 长度共同决定条件熵。条件熵可以粗略理解为“剩余上下文对缺失内容的不确定程度”。单 token 掩码通常保留了大量局部邻接词，条件熵较低；连续 span 会删掉一段内部结构，条件熵更高，因此任务更难，也更可能迫使模型学到高阶关系。

---

## 代码实现

下面给出一个简化的 span 掩码实现。它做三件事：

1. 根据 `mask_rate` 计算总预算。
2. 按随机长度采样多个不重叠 span。
3. 生成 BERT 风格或 T5 风格的替换结果。

```python
import random

def sample_spans(seq_len, mask_rate=0.15, max_span_len=5, seed=7):
    random.seed(seed)
    budget = max(1, int(seq_len * mask_rate))
    masked = set()
    spans = []

    while len(masked) < budget:
        start = random.randint(0, seq_len - 1)
        if start in masked:
            continue

        span_len = random.randint(1, max_span_len)
        end = min(seq_len, start + span_len)

        current = [i for i in range(start, end) if i not in masked]
        if not current:
            continue

        for i in current:
            masked.add(i)
        spans.append((current[0], current[-1] + 1))

        if len(masked) >= budget:
            break

    spans.sort()
    return spans, masked

def apply_t5_span_corruption(tokens, spans):
    output = []
    targets = []
    cursor = 0

    for idx, (start, end) in enumerate(spans):
        sentinel = f"<extra_id_{idx}>"
        output.extend(tokens[cursor:start])
        output.append(sentinel)

        targets.append(sentinel)
        targets.extend(tokens[start:end])
        cursor = end

    output.extend(tokens[cursor:])
    targets.append(f"<extra_id_{len(spans)}>")

    return output, targets

def apply_bert_mask(tokens, masked_positions):
    out = tokens[:]
    for i in masked_positions:
        out[i] = "[MASK]"
    return out

tokens = "我 昨天 在 北京 大学 参加 机器 学习 讲座".split()
spans, masked_positions = sample_spans(len(tokens), mask_rate=0.3, max_span_len=2, seed=3)

bert_view = apply_bert_mask(tokens, masked_positions)
t5_input, t5_target = apply_t5_span_corruption(tokens, spans)

assert len(masked_positions) >= 1
assert all(0 <= s < e <= len(tokens) for s, e in spans)
assert any(tok == "[MASK]" for tok in bert_view)
assert t5_input != tokens
assert t5_target[0].startswith("<extra_id_")

print("original:", tokens)
print("spans:", spans)
print("bert_view:", bert_view)
print("t5_input:", t5_input)
print("t5_target:", t5_target)
```

这个实现是教学版，不是生产版。生产训练里通常还会加上以下约束：

- span 长度不直接均匀采样，而是按几何分布或截断泊松分布采样，让短 span 更多、长 span 更少。
- 要避免特殊 token、padding、句边界被错误掩码。
- 要记录 label、loss mask、span 边界和 sentinel 映射，便于 decoder 对齐。
- 动态掩码通常在每个 epoch 重新采样，而不是一次写死。动态掩码就是同一条样本在不同轮次看到不同的洞位，能减少训练过拟合到固定缺口。

玩具例子可以直接看上面的输出结构。真实工程里，假设一个 batch 的长度是 512、掩码率是 40%、平均 span 长度是 3，那么每条样本大约有 204 个 token 被遮住、约 68 个 span。对 T5 类模型来说，encoder 输入更短、更稀疏，decoder 只需要输出缺失内容，而不是完整 512 token，这就是它在大规模生成预训练中常见的原因之一。

---

## 工程权衡与常见坑

最常见的误区，是把 BERT 的 15% 当成固定标准。它只是当年的有效默认值，不代表对所有模型都最优。模型越大、训练数据越多，通常越能承受更高掩码率。原因很简单：大模型容量更高，低难度恢复任务容易让训练信号偏弱。

下面是一个工程上常见的粗略对比：

| 策略 | 掩码率 | 平均 span 长度 | 预测对象数量 | 训练难度 | 常见用途 |
|---|---:|---:|---:|---|---|
| BERT 基线 | 15% | 1 | 高 | 低到中 | 分类、检索 |
| 高掩码单 token | 40% | 1 | 很高 | 中到高 | 大模型 encoder 预训练 |
| 中掩码 span | 30% | 3 | 中 | 中到高 | 实体、短语理解 |
| 高掩码 span corruption | 40%-60% | 3-10 | 相对更少 | 高 | 生成、摘要、翻译 |

这里“预测对象数量”要分开理解。被遮住的 token 总数可能很多，但如果采用 sentinel 压缩成少量 span，decoder 的步骤数可能反而更少。这就是“信息破坏更强，但解码目标更集中”的 trade-off。

常见坑主要有五类。

第一，span 太短，退化成 token masking。这样会失去连续结构训练的价值，模型还是主要在做局部填空。

第二，span 太长，剩余上下文不足。特别是在短文本上，如果一次删掉整段核心谓词或实体，样本会变成不可恢复噪声。

第三，掩码采样和分词边界不一致。中文、英文子词模型里，一个实体可能被拆成多个 subword。如果 span 设计只按 token 索引随机切，不考虑词边界，可能切碎命名实体，降低训练信号质量。

第四，静态掩码导致记忆洞位。训练集固定时，如果每条样本总在相同位置被挖洞，模型会部分记住模式，而不是学习可泛化的恢复能力。

第五，下游目标和预训练目标错位。比如你最终要做摘要或翻译，却仍只用低掩码单 token MLM；这不是不能用，而是目标偏差较大，迁移效率通常不如 span corruption 或 seq2seq 去噪。

一个真实工程例子：做企业知识库问答时，语料大量包含产品名、版本号、接口路径、错误码。如果预训练阶段只做随机单 token 掩码，模型更容易学到“某个局部词和周边上下文共现”；如果改成覆盖术语片段或接口片段的 span 掩码，模型会更常练习“跨边界恢复整段结构”，对实体识别、术语对齐、长答案抽取更有帮助。

---

## 替代方案与适用边界

不是所有模型都该上 span 掩码，也不是 span 掩码一定优于 token 掩码。选择策略要看模型结构和下游任务。

| 模型类型 | 主要任务 | 更常用策略 | 原因 |
|---|---|---|---|
| encoder-only | 分类、检索、NER | 单 token 或短 span | 关注表征质量，训练实现简单 |
| encoder-only 大模型 | 通用理解 | 更高掩码率 + 动态掩码 | 提高任务难度，避免训练过易 |
| encoder-decoder | 摘要、翻译、生成 | span corruption / MASS | 预训练目标更接近生成任务 |
| 实体密集文本 | 医疗、法律、知识库 | span 掩码 | 强化短语与边界建模 |
| 极小模型或低算力 | 轻量部署 | 低成本 token 掩码 | 训练稳定、实现便宜 |

还有几种替代思路值得区分。

第一，纯自回归语言模型，不掩码，只预测下一个 token。它最适合生成，但双向上下文利用较弱，不等价于 MLM。

第二，置换或删除式去噪，不一定显式插入 `[MASK]`。这类方法也能做去噪学习，但训练信号和恢复路径不同。

第三，动态课程式掩码。课程学习就是训练前期简单、后期变难。比如先用 15% 的短 span，后面逐步提高到 40% 并拉长 span。这个方案适合大模型，但实现更复杂，调参空间也更大。

一个初学者可以直接使用的经验规则是：

1. 只做文本分类、模型不大，先从 BERT 风格 15% 起步。
2. 做实体、关系抽取、长短语理解，尝试 20% 到 30% 的短 span 掩码。
3. 做摘要、翻译、对话生成，优先考虑 T5/MASS 一类的 span corruption。
4. 模型变大后，不要默认 15%，需要重新搜索更高掩码率和更合理的 span 分布。

适用边界也要明确。span 掩码提升的是“连续缺失恢复能力”，不是保证所有任务都变好。若数据本身极短、句法简单、标签任务强依赖局部触发词，span 掩码的收益可能有限。反过来，若文本长、结构复杂、术语密集，单 token 掩码往往不够。

---

## 参考资料

1. Devlin, Chang, Lee, Toutanova. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL 2019.  
   说明：定义了经典 MLM 与 15% 掩码、80/10/10 替换规则，是 token 级掩码的基线。  
   链接：https://aclanthology.org/N19-1423/

2. Joshi et al. *SpanBERT: Improving Pre-training by Representing and Predicting Spans*. TACL 2020.  
   说明：系统讨论连续 span 掩码对实体和短语建模的价值，是 span 级 encoder 预训练的重要代表。  
   链接：https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00300/43539/SpanBERT-Improving-Pre-training-by-Representing

3. Song et al. *MASS: Masked Sequence to Sequence Pre-training for Language Generation*. ICML 2019.  
   说明：提出面向 seq2seq 的连续片段掩码预训练，对机器翻译和摘要很有代表性。  
   链接：https://proceedings.mlr.press/v97/song19d.html

4. Raffel et al. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. JMLR 2020.  
   说明：T5 的核心论文，介绍了 span corruption 与 sentinel token 设计。  
   链接：https://www.jmlr.org/papers/v21/20-074.html

5. Wettig, Gao, Zhong, Chen. *Should You Mask 15% in Masked Language Modeling?* EACL 2023.  
   说明：实证分析更高掩码率在不同模型规模下的效果，是调参时最值得看的参考之一。  
   链接：https://aclanthology.org/2023.eacl-main.217/

6. Lewis et al. *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*. ACL 2020.  
   说明：虽然不只研究 span 掩码，但对去噪预训练与生成任务之间的关系有很好的工程参考价值。  
   链接：https://aclanthology.org/2020.acl-main.703/
