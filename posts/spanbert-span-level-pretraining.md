## 核心结论

SpanBERT 是一种跨度级预训练方法：它把随机遮蔽从“单个词元”改成“连续文本片段”，再用跨度边界目标只依赖左右边界表示来恢复整段被遮蔽内容。

普通 BERT 的核心训练信号是 Masked Language Modeling，简称 MLM，白话说就是“把某些位置盖住，让模型猜原来的词”。SpanBERT 保留了这种补全思想，但改变了补全单位。它不再只关注某个孤立位置像什么词，而是让模型学习“这一段连续文本整体是什么语义片段”。

关键变化有两部分：

| 对比项 | BERT | SpanBERT |
|---|---|---|
| 遮蔽单位 | 独立 token | 连续 span |
| 预测目标 | 单 token 预测 | span 内 token 预测 |
| 主要依赖 | 被遮蔽位置自身的上下文表示 | span 外侧左右边界表示 |
| 训练重点 | 位置级补全 | 跨度级语义恢复 |
| 更适合任务 | 通用语言理解、分类 | 问答、共指消解、实体抽取、片段匹配 |

玩具例子：句子“我 喜欢 北京 烤鸭 和 啤酒”中，如果遮蔽“北京 烤鸭”，普通 MLM 更像是在两个位置上分别猜“北京”和“烤鸭”。SpanBERT 的目标更严格：模型要根据左边界“喜欢”和右边界“和”，恢复中间整段 span。这里 span 是“连续文本片段”，白话说就是句子里相邻的一串词。

SpanBERT 的核心不是“把 BERT 变强一点”，而是把预训练目标改成更贴近下游任务的数据结构。抽取式问答里的答案通常是一段连续文本，共指消解里的 mention 也是一段连续文本。SpanBERT 直接在预训练阶段强化这种结构，因此在 span-centric 任务上更有优势。

---

## 问题定义与边界

SpanBERT 要解决的问题是：普通 BERT 虽然能做遮蔽词预测，但它对“连续片段”的整体建模不够直接。很多 NLP 任务的输出不是一个孤立词，而是一段连续文本。span-centric 任务就是这类以文本片段为核心输入或输出的任务，白话说就是“答案、实体、指代对象都要按一段文本来处理”。

抽取式问答是典型场景。给定问题“这道菜叫什么？”和上下文“我喜欢北京烤鸭和啤酒”，答案是“北京烤鸭”，不是单独的“北京”或“烤鸭”。模型需要判断答案 span 的起点和终点，还要理解这两个词合在一起表示一道菜。

任务边界可以这样看：

| 适合任务 | 不一定适合任务 |
|---|---|
| 抽取式问答：答案通常是连续 span | 句级分类：只需要判断整句类别 |
| 共指消解：mention 是连续片段 | 情感分类：多数情况下关注整体极性 |
| 命名实体识别：实体通常是连续词组 | 短文本意图识别：常由关键词或句级语义决定 |
| 片段匹配：需要比较两个 span 的语义 | 单词级词性标注：主要依赖局部 token 判断 |

真实工程例子：在一个客服知识库问答系统中，用户问“退款多久到账”，系统需要从文档中抽取“3 到 5 个工作日”作为答案。这个答案天然是连续 span。SpanBERT 的预训练方式会让模型更关注边界和片段内部的关系，因此更适合这种答案抽取任务。

但 SpanBERT 不是所有任务的固定最优解。如果任务只是判断“这句话是正面还是负面”，或者只需要短上下文内的单词级分类，SpanBERT 的额外训练复杂度未必带来明显收益。工程上不能把“在问答、共指上更强”误读成“所有任务都更强”。

---

## 核心机制与推导

SpanBERT 的第一步是 span masking。masking 是“遮蔽”，白话说就是把原文的一部分替换成特殊符号，让模型根据剩余上下文恢复原文。Span masking 不是随机打散 token，而是先采样连续跨度，再整体遮蔽。论文中 span 长度通常按几何分布采样，默认参数约为 $p=0.2$，并截断到 $l_{\max}=10$，总遮蔽预算约为 15%。

几何分布的直观含义是：短 span 更常见，长 span 也可能出现，但概率逐渐下降。可以写成：

$$
P(l=k)=(1-p)^{k-1}p
$$

其中 $l$ 是 span 长度，$k$ 是具体长度。截断到 $l_{\max}=10$ 表示最长只取 10 个词左右，避免训练目标过难。

第二步是 SBO，Span Boundary Objective，中文可译为“跨度边界目标”。它的白话解释是：预测被遮蔽内容时，不直接使用 span 内部位置的最终表示，而是只使用 span 外侧两个边界的表示，再加上相对位置嵌入。相对位置嵌入表示“当前要预测的是 span 内第几个 token”。

设输入序列为 $X=(x_1,\dots,x_n)$，被遮蔽跨度为 $(x_s,\dots,x_e)$。对跨度内第 $i$ 个 token，SpanBERT 构造：

$$
h_0=[c_{s-1};c_{e+1};p_{i-s+1}]
$$

$$
h_1=\mathrm{LayerNorm}(\mathrm{GeLU}(W_1h_0))
$$

$$
y_i=\mathrm{LayerNorm}(\mathrm{GeLU}(W_2h_1))
$$

其中 $c_{s-1}$ 是左边界 token 的 Transformer 输出表示，$c_{e+1}$ 是右边界 token 的 Transformer 输出表示，$p_{i-s+1}$ 是相对位置嵌入。GeLU 是一种神经网络激活函数，白话说就是给线性变换后的数值加上非线性能力。LayerNorm 是层归一化，白话说就是让表示的数值范围更稳定。

总损失可以写成：

$$
L_i=L_{\mathrm{MLM}}(x_i)+L_{\mathrm{SBO}}(x_i)
=-\log P(x_i\mid c_i)-\log P(x_i\mid y_i)
$$

直观上就是两条训练信号同时存在：MLM 让模型会做上下文补全，SBO 让模型必须从边界恢复整段内容。

流程图可以写成：

```text
输入句子
  ↓
连续 span 遮蔽
  ↓
Transformer 编码
  ↓
取左右边界表示 c_{s-1}, c_{e+1}
  ↓
结合相对位置 p_{i-s+1}
  ↓
预测被遮蔽 token
  ↓
计算 MLM loss + SBO loss
```

最小数值例子：句子“我 喜欢 北京 烤鸭 和 啤酒”，遮蔽跨度是“北京 烤鸭”。边界是“喜欢”和“和”。假设模型给出：

| token | $P(x_i\mid c_i)$ | $P(x_i\mid y_i)$ | 损失 |
|---|---:|---:|---:|
| 北京 | 0.6 | 0.5 | $-\log 0.6-\log 0.5\approx1.204$ |
| 烤鸭 | 0.2 | 0.4 | $-\log 0.2-\log 0.4\approx2.525$ |

该 span 总损失约为 $3.729$。这个例子说明，SpanBERT 不是只在空位上分别猜词，而是在训练模型使用边界语境恢复连续片段。

---

## 代码实现

实现 SpanBERT 思路时，关键顺序是：先按完整词采样连续 span，再遮蔽这些 span，然后编码，最后对每个 span 使用左右边界表示计算 SBO。完整词是指不把一个词随意拆开处理；在 WordPiece 或 BPE 这类 subword 分词里，一个词可能被切成多个子词，采样时要按原始词边界组织。

PyTorch 风格伪代码如下：

```python
# 1) 采样连续 span 并构造 mask
spans = sample_spans(tokens, mask_budget=0.15, geometric_p=0.2, max_len=10)

# 2) 送入 Transformer
hidden = encoder(masked_tokens)

# 3) 对每个被遮蔽 span 取左右边界
left = hidden[s - 1]
right = hidden[e + 1]

# 4) 结合相对位置，构造 SBO 表示
pos = position_embedding(i - s + 1)
h0 = torch.cat([left, right, pos], dim=-1)
y = sbo_mlp(h0)

# 5) 同时计算 MLM loss 和 SBO loss
loss = mlm_loss(hidden, labels) + sbo_loss(y, labels)
```

索引示意图：

```text
位置:   0    1     2     3    4    5
token: 我  喜欢  北京  烤鸭  和  啤酒
             ↑     ↑    ↑    ↑
           s-1     s    e   e+1

span = [s, e] = [2, 3]
预测“北京”时: i=2, i-s+1=1
预测“烤鸭”时: i=3, i-s+1=2
```

下面是一段可运行的 Python 代码，用最小逻辑演示 span 采样、边界索引和损失计算。它不是完整训练代码，但能验证核心索引是否正确。

```python
import math
import random

def sample_one_span(tokens, start, length):
    assert 0 < start
    assert start + length < len(tokens)
    s = start
    e = start + length - 1
    return s, e

def sbo_boundaries(tokens, s, e):
    left = tokens[s - 1]
    right = tokens[e + 1]
    rel_positions = [i - s + 1 for i in range(s, e + 1)]
    return left, right, rel_positions

def token_loss(p_mlm, p_sbo):
    return -math.log(p_mlm) - math.log(p_sbo)

tokens = ["我", "喜欢", "北京", "烤鸭", "和", "啤酒"]
s, e = sample_one_span(tokens, start=2, length=2)
left, right, rel = sbo_boundaries(tokens, s, e)

assert (s, e) == (2, 3)
assert left == "喜欢"
assert right == "和"
assert rel == [1, 2]

loss_beijing = token_loss(0.6, 0.5)
loss_kaoya = token_loss(0.2, 0.4)
span_loss = loss_beijing + loss_kaoya

assert round(loss_beijing, 3) == 1.204
assert round(loss_kaoya, 3) == 2.526
assert round(span_loss, 3) == 3.730
```

训练流程图：

```text
原始 tokens
  ↓
按完整词边界采样 span
  ↓
替换 span 为 [MASK] 或其他遮蔽形式
  ↓
Transformer 得到每个位置的 hidden state
  ↓
MLM: 用被遮蔽位置表示预测原 token
  ↓
SBO: 用 hidden[s-1]、hidden[e+1]、相对位置预测原 token
  ↓
合并损失并反向传播
```

这里最容易写错的是边界。SBO 使用的是 span 外侧的 $x_{s-1}$ 和 $x_{e+1}$，不是 $x_s$ 和 $x_e$。如果拿 span 内部 token 的表示做边界，模型就可能直接从被遮蔽区域泄漏信息，训练目标会偏离 SpanBERT 的设计。

---

## 工程权衡与常见坑

SpanBERT 的工程价值来自目标设计，不只是遮蔽形式。只做 span masking 但不做 SBO，模型更像在做“连续版 mask 填空”，没有被强制学习“边界表示如何概括中间片段”。这会削弱 SpanBERT 对抽取式问答和共指消解的优势。

常见坑如下：

| 错误做法 | 正确做法 | 会导致什么问题 |
|---|---|---|
| 把 span 按 subword 随意切碎 | 按完整词组成连续 span | 破坏片段语义一致性 |
| 使用 $x_s$、$x_e$ 当 SBO 边界 | 使用外侧 $x_{s-1}$、$x_{e+1}$ | 边界目标不成立，可能泄漏内部信息 |
| 只做 span masking，忽略 SBO | 同时计算 MLM loss 和 SBO loss | 退化成弱化版遮蔽训练 |
| 把 SpanBERT 当成所有任务必胜方案 | 根据任务是否 span-centric 选择 | 在分类任务上收益可能有限 |
| 遮蔽预算过大 | 保持约 15% 的遮蔽比例 | 上下文过少，训练过难 |
| span 长度无限制 | 使用最大长度截断，例如 $l_{\max}=10$ | 长 span 太难恢复，训练不稳定 |

仍以“我 喜欢 北京 烤鸭 和 啤酒”为例。错误做法是把“北京 烤鸭”拆成两个无关片段，分别遮蔽“北京”和“烤鸭”，甚至在 subword 级别只遮住某个碎片。这样模型学到的是局部补全，不是“北京烤鸭”作为一个菜名 span 的整体语义。正确做法是把“北京 烤鸭”视为连续片段整体遮蔽，再用“喜欢”和“和”两个外侧边界恢复内部内容。

真实工程中还要考虑成本。SpanBERT 需要修改数据预处理、mask 策略和训练头。如果团队只是微调现成模型，使用公开的 SpanBERT checkpoint 成本较低；如果要从头预训练，则需要额外实现和验证。对资源有限的团队，除非下游任务明显依赖 span，否则优先使用成熟 BERT 或 RoBERTa 系列模型通常更稳。

---

## 替代方案与适用边界

SpanBERT 不是唯一的 span 建模路线。可以把它放在 BERT、Whole Word Masking 和其他实体增强预训练方法之间理解。Whole Word Masking 是“整词遮蔽”，白话说就是如果一个词被拆成多个 subword，要一起遮住这个词的所有子词。它比普通 BERT 更尊重词边界，但仍不等于 SpanBERT，因为它没有 SBO，也不一定强调连续多词 span 的边界恢复。

| 方法 | 核心目标 | 优势 | 局限 | 适用任务 |
|---|---|---|---|---|
| BERT | 随机 token MLM + 下一句预测 | 通用、实现成熟、生态完整 | 对连续 span 的目标设计不直接 | 分类、匹配、序列标注、通用理解 |
| Whole Word Masking | 按完整词遮蔽 | 避免 subword 碎片化 | 仍主要是词级补全 | 中文/英文词级语义增强、通用微调 |
| SpanBERT | 连续 span masking + SBO | 强化跨度边界和片段恢复 | 实现更复杂，非 span 任务收益不稳定 | 抽取式问答、共指消解、实体抽取 |
| 实体增强预训练 | 引入实体链接或知识库信号 | 对实体密集任务有帮助 | 依赖额外标注或知识资源 | 知识问答、实体链接、领域搜索 |

如果你做的是句子级情感分类，比如判断“这家店服务不错”是正面还是负面，SpanBERT 的优势通常不会像在抽取式问答中那么明显。因为分类任务最终只需要一个句级标签，不一定要求模型精确恢复连续文本片段。

如果你做的是 SQuAD、MRQA 或企业文档问答，答案天然是连续 span，SpanBERT 的目标与任务结构更一致。模型在预训练阶段已经练习过“根据边界恢复内部片段”，微调时再学习“从上下文里找答案起止位置”，目标之间的差距更小。

工程选择可以按三条原则判断：

| 判断问题 | 选择倾向 |
|---|---|
| 下游输出是否经常是连续文本片段？ | 是：优先考虑 SpanBERT |
| 是否只做句级分类或短文本判断？ | 是：BERT 或 RoBERTa 通常足够 |
| 是否有能力维护自定义预训练流程？ | 否：优先使用现成 checkpoint |
| 是否需要按实体、答案、mention 精确定位？ | 是：SpanBERT 更值得评估 |

结论是：SpanBERT 的适用边界很清楚。它不是替代所有 BERT 类模型的通用答案，而是当任务核心对象从“单个位置”变成“连续片段”时，更匹配问题结构的预训练方法。

---

## 参考资料

1. [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://aclanthology.org/2020.tacl-1.5/)
2. [facebookresearch/SpanBERT](https://github.com/facebookresearch/SpanBERT)
3. [AllenAI paper-to-html](https://a11y2.apps.allenai.org/paper?id=81f5810fbbab9b7203b9556f4ce3c741875407bc)
4. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/)
5. [The Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
