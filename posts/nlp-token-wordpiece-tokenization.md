## 核心结论

WordPiece 是一种**子词分词算法**。子词可以理解为“介于单个字符和完整单词之间的片段”。它的目标不是单纯把最常见的两个片段拼起来，而是优先选择“合并后最能提高训练语料解释能力”的片段对。

它常用一个近似分数来做贪心选择：

$$
\text{score}(A,B)=\frac{\text{freq}(AB)}{\text{freq}(A)\times \text{freq}(B)}
$$

这里的直觉是：如果 $A$ 和 $B$ 单独都很常见，但它们经常**一起出现**，那么把它们合成一个新子词是有价值的；如果它们只是因为各自太常见而碰巧频繁相邻，这个分数就不会太高。这个准则和 BPE 不同。BPE 只看“谁出现次数最多”，WordPiece 更接近“谁的结合更有信息量”。

BERT 采用的就是 WordPiece。它的词表大小大约是 3 万级，典型例子是 `bert-base-uncased` 的 30,522 个 token。对不在词表中的词，WordPiece 会拆成多个子词，非词首子词用 `##` 表示“这是接在前面后面的片段”。例如 `playing` 可以被拆成 `["play", "##ing"]`。这样模型即使没见过完整单词，也能利用已有子词理解含义，而不是直接退化成 `[UNK]`。

---

## 问题定义与边界

分词的核心问题很简单：神经网络不能直接处理原始文本，它需要把文本映射成离散 token，再查 embedding。**embedding** 可以理解为“每个 token 对应的一行可训练向量”。如果词表太大，embedding 矩阵会很大，训练和推理成本都会上升；如果词表太小，又会有大量未登录词，也就是 **OOV**，可以理解为“词表里没有的词”。

WordPiece 处理的正是这个矛盾：

| 目标 | 为什么重要 | WordPiece 的做法 |
| --- | --- | --- |
| 控制词表大小 | embedding 参数量和显存成本随词表增大 | 不为所有词保留整词 |
| 减少 `[UNK]` | 新词、罕见词、复合词很多 | 用子词回退表示 |
| 保留语义单元 | 过细拆分会损失常见结构 | 优先学习高价值子词 |

一个最容易理解的玩具例子是英文后缀。假设词表里已经有 `play`，但没有 `playing`。如果采用“整词词表”方案，`playing` 可能直接变成 `[UNK]`。而 WordPiece 会尽量把它表示成：

`playing -> ["play", "##ing"]`

这里 `##ing` 的意思不是“单独的 ing”，而是“接在前面词干后面的 ing”。这让模型能共享 `play`、`##ing` 在不同词里的统计规律。

它也有明确边界。WordPiece 解决的是“固定词表下如何兼顾覆盖率与参数规模”，不是“语言理解”的全部问题。分词再合理，也不能替代模型结构、预训练数据和下游任务标注质量。另一个边界是：WordPiece 最适合像 BERT 这样以编码为主的模型，即需要稳定、确定性切分结果的场景；它不是所有语言、所有任务下的唯一最优方案。

---

## 核心机制与推导

WordPiece 训练通常从很细的符号开始。对英文这类空格分词语言，一种常见做法是先把每个词拆成字符级单位，首字符保留原样，后续字符加 `##` 前缀。例如：

`word -> ["w", "##o", "##r", "##d"]`

这样设计的目的，是区分“词首出现的片段”和“词中续接片段”。这一步很关键，因为 `ing` 出现在词尾和词首，统计意义完全不同。

接下来，算法反复做三件事：

1. 统计当前语料中相邻片段对的频次。
2. 计算每个候选 pair 的分数。
3. 选择分数最大的 pair 合并成新 token。

为什么分数写成

$$
\frac{\text{freq}(AB)}{\text{freq}(A)\times \text{freq}(B)}
$$

而不是直接看 $\text{freq}(AB)$？

因为单纯频次会偏向“各自都很常见”的片段。例如 `##u` 和 `##g` 都可能在很多单词里出现，所以它们相邻的次数可能不低；但这不代表它们组合成一个子词就一定值钱。WordPiece 用分母对常见单片段做归一化，相当于在问：

“这两个片段一起出现的程度，是否显著高于它们各自独立出现时的预期？”

这个思路和 **PMI** 很接近。PMI 可以理解为“两个事件一起出现时比独立出现更紧密多少”。

看一个玩具例子。假设训练集中有这些词和频次：

- `hug` 出现 10 次
- `hugs` 出现 5 次
- `bug` 出现 8 次
- `bugs` 出现 4 次

初始切分后，`##g` 很常见，`##s` 也不算少，但 `##g` 和 `##s` 连在一起主要集中出现在复数形式里。于是 `("##g", "##s")` 的联合统计相对更“专一”，它的 score 可能高于某些绝对频次更大的 pair。这样，WordPiece 可能优先学出 `##gs`，而不是只按频率先学出别的组合。

这个例子说明一件事：WordPiece 在贪心地扩张词表，但它贪心的对象不是“最多见”，而是“最值得合并”。

再看一个真实工程例子。BERT 预训练面对海量开放域文本，里面会出现人名、术语、派生词、拼写变体。如果词表只保留完整单词，词表要膨胀得非常大；如果退回字符级，序列会过长，语义单元也太碎。WordPiece 的中间策略是：

- 高频整词直接保留，如 `play`
- 高频后缀或构词片段保留，如 `##ing`、`##ed`
- 低频复合词按已有子词组合表示

所以像 `pretraining` 这类词，即使不在整词词表中，也可能被编码为：

`["pre", "##train", "##ing"]`

这就是 WordPiece 的工程价值：在有限词表内，尽量把“常见语义块”留下来。

---

## 代码实现

下面给出一个可运行的简化版 Python 实现。它不追求完整工业级 tokenizer 功能，只演示 WordPiece 训练时的核心思想：统计词频、拆成初始子词、计算 pair score、做一次贪心合并。

```python
from collections import Counter, defaultdict

def initial_split(word):
    pieces = []
    for i, ch in enumerate(word):
        pieces.append(ch if i == 0 else "##" + ch)
    return pieces

def build_split_corpus(word_freqs):
    return {word: initial_split(word) for word in word_freqs}

def compute_piece_freqs(splits, word_freqs):
    piece_freq = Counter()
    pair_freq = Counter()

    for word, pieces in splits.items():
        freq = word_freqs[word]
        for p in pieces:
            piece_freq[p] += freq
        for i in range(len(pieces) - 1):
            pair_freq[(pieces[i], pieces[i + 1])] += freq

    return piece_freq, pair_freq

def score_pairs(splits, word_freqs):
    piece_freq, pair_freq = compute_piece_freqs(splits, word_freqs)
    scores = {}
    for (a, b), ab_freq in pair_freq.items():
        scores[(a, b)] = ab_freq / (piece_freq[a] * piece_freq[b])
    return scores

def merge_pair(a, b):
    if b.startswith("##"):
        return a + b[2:]
    return a + b

def apply_merge(splits, target_pair):
    a, b = target_pair
    new_splits = {}
    merged = merge_pair(a, b)

    for word, pieces in splits.items():
        new_pieces = []
        i = 0
        while i < len(pieces):
            if i < len(pieces) - 1 and pieces[i] == a and pieces[i + 1] == b:
                new_pieces.append(merged)
                i += 2
            else:
                new_pieces.append(pieces[i])
                i += 1
        new_splits[word] = new_pieces
    return new_splits, merged

word_freqs = {
    "hug": 10,
    "hugs": 5,
    "bug": 8,
    "bugs": 4,
}

splits = build_split_corpus(word_freqs)
scores = score_pairs(splits, word_freqs)
best_pair = max(scores, key=scores.get)
new_splits, new_token = apply_merge(splits, best_pair)

assert isinstance(best_pair, tuple)
assert len(best_pair) == 2
assert isinstance(new_token, str)
assert all(isinstance(v, list) for v in new_splits.values())

print("best_pair =", best_pair)
print("new_token =", new_token)
print("new_splits =", new_splits)
```

这段代码里最重要的是两点。

第一，`piece_freq` 和 `pair_freq` 都要按词频累加，而不是只看“词是否出现过”。因为训练语料里的出现次数决定了统计强度。

第二，`merge_pair(a, b)` 处理了 `##` 前缀。比如 `a="play"`、`b="##ing"` 时，合并结果应该是 `playing`；如果简单字符串拼接成 `play##ing`，就错了。

在真实工程中，一个完整 WordPiece 训练器通常还需要补足这些环节：

| 模块 | 作用 | 简化版是否覆盖 |
| --- | --- | --- |
| 预分词 | 先把原始文本切成词级单元 | 未覆盖 |
| 规范化 | 小写化、Unicode 归一化、清洗噪声 | 未覆盖 |
| 目标词表控制 | 达到指定词表大小后停止 | 可扩展 |
| 特殊 token 注入 | `[PAD]`、`[CLS]`、`[SEP]` 等 | 未覆盖 |
| 推理阶段最长匹配 | 新词编码时从最长子词开始回退 | 未覆盖 |

推理时常用的是**最长匹配**。最长匹配可以理解为“优先尝试最长能命中的词片段，命不中再继续拆短”。例如词表里有 `play` 和 `##ing`，没有 `playing`，那么编码 `playing` 时会先找 `play`，剩余部分再找 `##ing`，得到 `["play", "##ing"]`。如果剩余部分继续找不到，才进一步拆成更短子词，直到字符级或 `[UNK]`。

---

## 工程权衡与常见坑

WordPiece 的第一个工程权衡是“词表大小”和“序列长度”的平衡。词表越大，embedding 越重，但单词越可能整词命中，序列越短；词表越小，参数越省，但切分更碎，序列更长。Transformer 的自注意力复杂度和序列长度强相关，所以“过碎分词”会直接抬高计算成本。

第二个权衡是“统计稳定性”。WordPiece 依赖频次比值，语料太小时，score 很容易被偶然共现放大，导致词表学出很多只在小样本里看起来合理、实际泛化很差的子词。换句话说，WordPiece 的评分比 BPE 更聪明，但也更依赖足够稳定的统计量。

常见坑可以直接列出来：

| 问题 | 表现 | 后果 | 处理方式 |
| --- | --- | --- | --- |
| 把 WordPiece 当成 BPE | 只按 pair 频率排序 | 学不到高信息量组合 | 明确按 score 排序 |
| 忽略 `##` 前缀 | 词中片段和词首片段混在一起统计 | 推理时切分异常 | 训练和编码阶段统一规则 |
| 词表设得过小 | 大量词被拆成很短片段 | 序列变长，语义变碎 | 结合任务和显存调词表 |
| 直接迁移英文规则到中文 | 中文不一定适合同一套边界假设 | 切分收益有限 | 结合语言特性重设计 |
| 重训词表但不重训模型 | token id 含义变了 | 模型效果断崖下降 | 词表和 embedding 必须配套 |

一个很典型的误区是：有人以为“频率最高的 pair 就最该合并”。这在 BPE 里成立，在 WordPiece 里不成立。比如 `##u` 和 `##g` 都很常见，它们的 pair 频率可能比 `##g` 和 `##s` 更高；但如果 `##u`、`##g` 到处都能单独出现，而 `##g`、`##s` 更集中地共同出现，那么 WordPiece 会更偏向后者。这个差别不只是数学形式不同，而是直接决定词表是否能保留“真正稳定的子结构”。

真实工程里还要注意一个问题：分词器不是孤立模块。比如你要给 BERT 做领域微调，发现医学术语切得太碎，于是重训了一个领域 WordPiece 词表。这样做没问题，但如果你还想复用原始 BERT 权重，就要知道：新的 token id 与原 embedding 行已经不一一对应。通常需要重新初始化新 token 的 embedding，甚至重新做较充分的继续预训练，而不是只换个词表文件就结束。

---

## 替代方案与适用边界

WordPiece 不是唯一的子词方法。最常拿来比较的是 BPE 和 Unigram。

**BPE** 是“字节对编码”，这里可以理解为“每轮把出现次数最多的片段对直接合并”。它实现简单、训练快、直觉也清楚，很多生成式模型会采用类似思想。缺点是它只看频率，不主动惩罚“本来就很常见”的基础片段，所以不一定能优先学到最有区分度的组合。

**Unigram** 是另一条路线。它先准备一个较大的候选词表，再反复删除那些对整体概率贡献较小的 token。它本质上是“从大集合里剪枝”，而不是“从小集合里逐步合并”。优点是概率模型更完整，常能支持多种切分候选；代价是训练更复杂。

三者可以放在一起看：

| 特性 | WordPiece | BPE | Unigram |
| --- | --- | --- | --- |
| 基本思路 | 从小词表出发，按似然增益贪心合并 | 从小词表出发，按频率贪心合并 | 从大词表出发，按概率贡献剪枝 |
| 评分重点 | 联合出现是否“特别紧” | 出现次数是否“最多” | 删除后整体概率损失多大 |
| 典型标记 | 非词首常带 `##` | 通常不靠 `##` 表示续接 | 实现可变 |
| 优势 | 词表更偏信息量 | 简单高效 | 灵活、概率解释强 |
| 代价 | 统计和实现比 BPE 稍复杂 | 可能忽略高信息组合 | 训练最复杂 |

适用边界也很清楚：

- 如果你做的是 BERT 类编码模型，且希望词表稳定、切分确定，WordPiece 是非常自然的选择。
- 如果你更看重实现简单、训练速度快，BPE 往往更直接。
- 如果你需要更强的概率建模能力，或者希望一个词有多种切分候选，Unigram 更有优势。

最后要强调一点：WordPiece 很适合“固定词表的编码器”。如果模型架构本身已经转向字节级、字符级，或者采用完全不同的 tokenizer 设计，WordPiece 的优势会减弱。方法没有绝对优劣，只有是否符合系统目标。

---

## 参考资料

- [Hugging Face Course: WordPiece tokenization](https://huggingface.co/docs/course/en/chapter6/6)
- [Hugging Face Transformers: Tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary)
- [Hugging Face Transformers v5 RC: Tokenizer summary](https://huggingface.co/docs/transformers/v5.0.0rc0/en/tokenizer_summary)
- [Michael Brenndoerfer: WordPiece Tokenization, BERT’s Subword Algorithm Explained](https://mbrenndoerfer.com/writing/wordpiece-tokenization-bert-subword-algorithm)
- [Mue AI: WordPiece Tokenizer](https://muegenai.com/docs/dsa/section-1/chapter-2-understanding-the-bert-model/wordpiece-tokenization/)
- [Google Research BERT issue discussing vocab size](https://github.com/google-research/bert/issues/1225)
