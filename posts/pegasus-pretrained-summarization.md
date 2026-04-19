## 核心结论

PEGASUS 是面向抽象式摘要的 `Transformer 编码器-解码器` 预训练模型。抽象式摘要指模型不是简单复制原文句子，而是生成一段新的、更短的概括文本。

它的核心预训练任务是 `GSG（Gap Sentence Generation，间隔句子生成）`：先从文档中选出若干重要句子，把这些句子从原文里拿掉，再让模型根据剩余句子生成被拿掉的句子。

新手版理解：普通预训练常见做法是“随机遮几个词，让模型补空”；PEGASUS 更像“直接拿掉几句最重要的话，让模型把这些关键句补回来”。摘要任务本质上也是从长文中提炼关键信息，因此 GSG 和摘要目标高度对齐。

| 训练或任务形式 | 遮蔽粒度 | 模型要做什么 | 和摘要任务的关系 |
|---|---:|---|---|
| 随机 mask | 词或 token | 根据上下文补局部缺失词 | 学语言理解有用，但不直接等于摘要 |
| 句子级遮盖 | 整句 | 根据剩余文档恢复缺失句子 | 更接近从全文提炼关键内容 |
| 摘要任务目标 | 文档级信息 | 根据长文生成短摘要 | 直接要求压缩、选择、改写信息 |

为什么 PEGASUS 适合摘要：它在预训练阶段就让模型练习“从剩余正文生成关键句”，这个目标比随机遮词更接近摘要生成，所以在标注摘要数据较少时尤其有优势。

---

## 问题定义与边界

PEGASUS 解决的是抽象式文本摘要问题。输入是一篇或一段较长文本，输出是一段较短的摘要。它不是分类模型、检索模型，也不是只预测下一个词的普通语言模型。

场景示例：一篇长新闻有 20 句话，模型要生成 2 到 3 句摘要。它不应该把原文逐句复制一遍，而应该保留事件主体、关键动作、结果和必要背景。

邮件线程也是类似场景。把 PEGASUS 用在邮件总结里时，目标不是找出几个关键词，而是生成一段能概括整段对话的短文本，例如“客户询问退款进度，客服确认订单已进入财务审核，预计两个工作日内完成处理”。

| 问题项 | PEGASUS 中的含义 |
|---|---|
| 输入是什么 | 一段文档、新闻、邮件、工单、论文摘要前的正文等 |
| 输出是什么 | 一段较短的自然语言摘要 |
| 依赖什么数据 | 预训练阶段依赖大量文档；微调阶段最好有正文-摘要配对数据 |
| 适合什么任务 | 抽象式摘要、新闻摘要、领域文本摘要微调 |
| 不适合什么场景 | 精确检索、分类打标、需要逐字引用的摘要、任意超长文本无损总结 |

边界条件需要明确。

第一，PEGASUS 不是零样本通用摘要器。低资源表现好，意思是在少量标注摘要数据上微调后效果较强，不等于完全不需要任务数据。

第二，PEGASUS 不是 encoder-only 模型。encoder-only 模型指只有编码器，常用于分类、检索、序列标注等任务；PEGASUS 是编码器-解码器结构，编码器理解输入，解码器生成摘要。

第三，PEGASUS 不能直接对任意超长文本无损工作。Transformer 模型有最大输入长度限制，超过长度的文档需要截断、分段、层级摘要或换成长上下文方案。

---

## 核心机制与推导

设一篇文档由多个句子组成：

$$
D = \{s_1, s_2, ..., s_n\}
$$

PEGASUS 先从文档中选出 gap 句集合：

$$
G \subset D
$$

这里的 gap 句指被挖掉的句子。剩余内容作为输入：

$$
X = D \setminus G
$$

被挖掉的句子按原顺序拼接成生成目标：

$$
Y = concat(G)
$$

模型训练目标是根据 $X$ 生成 $Y$。损失函数是标准的序列到序列自回归生成损失：

$$
L(\theta) = -\sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}, X)
$$

其中 $y_t$ 是目标摘要序列中的第 $t$ 个 token，$y_{<t}$ 表示它之前已经生成的 token。自回归生成指模型每次生成下一个 token 时，都依赖已经生成的前文。准确说，PEGASUS 的编码器使用双向注意力理解输入，真正自回归的是解码器，不是编码器。

gap 句如何选择是关键。PEGASUS 不是随便删句子，而是倾向选择“像摘要”的重要句。一个常见近似是用 ROUGE1-F1 衡量某个句子和剩余文档的重合程度：

$$
r_i = ROUGE1\text{-}F1(s_i, D \setminus \{s_i\})
$$

ROUGE 是摘要评估指标，白话说就是看生成文本和参考文本在词语上重合多少。这里用它近似句子的重要性：如果某个句子和文档其他部分共享很多关键词，它可能包含主题信息，适合作为 gap 句。

玩具例子：文档有 4 句话，重要性分数是：

$$
[0.12, 0.71, 0.10, 0.55]
$$

取 `top-2` 后，gap 句为：

$$
G = \{s_2, s_4\}
$$

输入变成：

$$
X = \{s_1, s_3\}
$$

模型目标是生成：

$$
Y = concat(s_2, s_4)
$$

如果目标序列只有 3 个 token，模型对正确 token 给出的概率分别是 $[0.8, 0.5, 0.25]$，则损失约为：

$$
L = -\ln 0.8 - \ln 0.5 - \ln 0.25 \approx 2.30
$$

机制流程如下：

| 步骤 | 操作 | 产物 |
|---:|---|---|
| 1 | 文档切句 | 得到 $D=\{s_1,...,s_n\}$ |
| 2 | 计算句子重要性 | 得到每句分数 $r_i$ |
| 3 | 选出 gap 句 | 得到 $G$ |
| 4 | 构造输入和目标 | 得到 $X=D\setminus G$ 与 $Y=concat(G)$ |
| 5 | 训练编码器-解码器 | 最小化生成损失 $L(\theta)$ |

真实工程例子：在新闻摘要系统中，原文可能包含事件背景、人物表态、过程细节和结果。GSG 会倾向拿掉能代表主线的信息句，例如“监管部门宣布对某平台展开调查”或“公司表示将暂停相关业务”。模型在训练中不断学习：怎样从上下文恢复这些关键句。微调到新闻摘要数据后，它更容易生成“谁做了什么、结果是什么、影响是什么”这类摘要结构。

---

## 代码实现

实现 PEGASUS 思路时，关键不在重新写 Transformer，而在构造 GSG 数据。模型结构可以使用现成框架，数据构造却必须理解清楚。

| 模块 | 作用 |
|---|---|
| 文本预处理 | 清洗空白字符、统一标点、去掉异常内容 |
| 句子切分 | 把文档拆成句子列表 |
| GSG 样本构造 | 选择 gap 句，构造 source 和 target |
| tokenizer 编码 | 把文本转成 token id |
| 模型训练 | 用 seq2seq loss 训练模型 |
| 摘要生成 | 输入正文，解码生成摘要 |

最小流程是：

原文 → 切句 → 选 gap → 构造输入与目标 → 编码 → 训练 → 生成摘要

下面是一个可运行的玩具实现。它没有实现完整 ROUGE，而是用词集合 F1 近似句子重要性，目的是展示 GSG 数据构造的核心逻辑。

```python
import math
import re

def split_to_sentences(doc):
    parts = re.split(r"(?<=[。！？.!?])\s*", doc.strip())
    return [p for p in parts if p]

def tokenize(text):
    return re.findall(r"\w+|[\u4e00-\u9fff]", text.lower())

def f1_overlap(a, b):
    a_tokens = tokenize(a)
    b_tokens = tokenize(b)
    if not a_tokens or not b_tokens:
        return 0.0

    a_set = set(a_tokens)
    b_set = set(b_tokens)
    overlap = len(a_set & b_set)
    precision = overlap / len(a_set)
    recall = overlap / len(b_set)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def build_gsg_example(doc, k=1):
    sentences = split_to_sentences(doc)
    assert len(sentences) >= k + 1

    scores = []
    for i, sentence in enumerate(sentences):
        rest = " ".join(s for j, s in enumerate(sentences) if j != i)
        scores.append(f1_overlap(sentence, rest))

    gap_idx = sorted(
        sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    )

    source = " ".join(s for i, s in enumerate(sentences) if i not in gap_idx)
    target = " ".join(sentences[i] for i in gap_idx)

    return source, target, gap_idx, scores

doc = (
    "PEGASUS 是一个面向摘要任务的预训练模型。"
    "它使用间隔句子生成任务。"
    "模型根据剩余句子生成被遮蔽的重要句子。"
    "这种目标和摘要生成高度一致。"
)

source, target, gap_idx, scores = build_gsg_example(doc, k=1)

assert len(source) > 0
assert len(target) > 0
assert len(gap_idx) == 1
assert target in doc
assert source != target

probs = [0.8, 0.5, 0.25]
loss = -sum(math.log(p) for p in probs)
assert round(loss, 2) == 2.30

print("source:", source)
print("target:", target)
print("gap_idx:", gap_idx)
print("loss:", round(loss, 2))
```

新手版伪代码可以进一步压缩成：

```python
sentences = split_to_sentences(doc)
scores = [rouge1_f1(s, doc_without(s)) for s in sentences]
gap_idx = top_k(scores, k=2)
source = join([s for i, s in enumerate(sentences) if i not in gap_idx])
target = join([sentences[i] for i in gap_idx])
loss = seq2seq_loss(model(source), target)
```

推理阶段不再挖 gap 句。输入是一篇长新闻正文，模型直接生成摘要。例如：

| 阶段 | 输入 | 输出 |
|---|---|---|
| 预训练 | 去掉关键句后的文档 | 被去掉的关键句 |
| 微调 | 新闻全文 | 人工摘要 |
| 推理 | 新新闻全文 | 模型生成的 2 句摘要 |

实际工程中可以使用 Hugging Face 的 `PegasusTokenizer` 和 `PegasusForConditionalGeneration` 加载模型，但不要把 API 当作重点。真正影响效果的是：输入是否切对、训练样本是否匹配任务、摘要长度是否符合业务目标。

---

## 工程权衡与常见坑

PEGASUS 的效果强依赖句子切分质量。句子切分指把一整段文本拆成独立句子的过程。英文里通常依赖句号、问号和缩写规则；中文里还要处理“。！？；”、列表、标题、括号和口语聊天断句。

如果句子都切错了，模型学到的“重要句”就不是人类认为的摘要信息。例如中文客服工单里经常有短句、编号、时间戳和多轮对话。如果把“退款原因：商品破损。用户要求加急处理。”切成混乱片段，GSG 选择的 gap 句就会失真。

| 常见坑 | 影响 | 规避方法 |
|---|---|---|
| 把 PEGASUS 当成 encoder-only 模型 | 错用模型结构，无法正确生成摘要 | 明确它是编码器-解码器模型 |
| 把随机 mask 当成 GSG | 训练目标和摘要不对齐 | 按句子级别选择并生成 gap 句 |
| 中文文本没有正确分句 | 重要句选择失真 | 使用适配中文标点和业务格式的分句器 |
| 输入过长直接截断 | 关键信息可能被截掉 | 先分段、排序、检索或层级摘要 |
| 领域数据不足 | 生成风格不符合业务 | 用少量高质量领域摘要继续微调 |
| 把低资源理解成零监督 | 线上效果不可控 | 至少准备验证集和少量任务样本 |

长文本处理尤其常见。PEGASUS 不能天然读完整本手册或数小时会议记录。需要先判断文本长度和信息分布。

| 情况 | 建议策略 |
|---|---|
| 文本略超最大长度 | 保留标题、开头、结尾和高相关段落 |
| 文本很长但结构清晰 | 按章节分段摘要，再汇总 |
| 文本很长且查询明确 | 先检索相关段落，再摘要 |
| 会议纪要类多话题文本 | 先按话题切分，再分别生成摘要 |
| 法务或合规文本 | 避免自由改写，保留引用和出处 |

真实工程例子：客服工单摘要通常包括“用户问题、处理动作、当前状态、下一步”。直接用新闻摘要模型可能会生成流畅但不符合工单格式的文本。更稳的做法是先准备几百到几千条领域样本微调，再限制输出模板，例如“问题：...；处理：...；状态：...”。PEGASUS 可以作为基础模型，但业务格式仍要靠数据和约束补齐。

---

## 替代方案与适用边界

PEGASUS 适合抽象式摘要，但不是所有摘要任务的最优解。选模型时要看文本长度、摘要风格、领域差异、延迟预算和训练数据规模。

| 方案 | 预训练目标 | 是否适合摘要 | 长文本能力 | 低资源表现 | 工程复杂度 |
|---|---|---|---|---|---|
| PEGASUS | GSG，生成被遮蔽的重要句 | 很适合抽象式摘要 | 受最大长度限制 | 强，尤其适合少量摘要数据微调 | 中等 |
| BART | 文本去噪，重建被破坏文本 | 适合摘要和生成任务 | 受最大长度限制 | 强，但预训练目标更通用 | 中等 |
| T5 | text-to-text，多任务文本转换 | 适合多类 NLP 任务 | 受具体版本限制 | 取决于模型和数据 | 中等 |
| 长上下文摘要方案 | 长上下文建模或分段层级处理 | 适合超长文档 | 强 | 取决于模型和标注 | 较高 |

场景对比：

| 场景 | 推荐判断 |
|---|---|
| 新闻摘要 | PEGASUS 很适合，任务形式和 GSG 接近 |
| 邮件线程总结 | 可以用 PEGASUS，但通常需要领域微调 |
| 超长会议纪要 | 更适合分段摘要或长上下文模型 |
| 强术语技术文档 | 需要领域继续训练或微调 |
| 必须逐字引用的摘要 | PEGASUS 不一定合适，抽取式方案更稳 |
| 低延迟线上接口 | 需要评估模型大小、解码长度和缓存策略 |

新手版理解：不是所有“总结文本”的任务都必须用 PEGASUS。新闻、短文档、少量标注摘要数据这些条件越明显，PEGASUS 越合适；文本越长、格式越强、事实精确性要求越高，就越需要额外工程设计。

PEGASUS 的适用边界可以概括为四点：数据少但仍有摘要任务样本；目标是抽象式摘要；可以接受微调成本；原文具有可用的句子级结构。只要这些条件不满足，就应该认真比较 BART、T5、长上下文模型、抽取式摘要或检索增强摘要方案。

---

## 参考资料

1. [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777)：原始论文，用于确认 GSG 定义、训练目标和实验结论。
2. [Google Research: PEGASUS](https://research.google/pubs/pegasus-pretraining-with-extracted-gap-sentences-for-abstractive-summarization-by-sequence-to-sequence-models/)：官方研究页，用于了解方法背景和论文信息。
3. [google-research/pegasus](https://github.com/google-research/pegasus)：官方代码仓库，用于查看原始实现和训练配置。
4. [Hugging Face Transformers PEGASUS Documentation](https://huggingface.co/docs/transformers/model_doc/pegasus)：框架文档，用于快速上手 tokenizer、模型加载和摘要生成。
5. [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/)：ROUGE 指标来源，用于理解摘要评估和句子重要性近似。
