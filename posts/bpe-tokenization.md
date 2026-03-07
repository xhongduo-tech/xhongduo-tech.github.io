## 核心结论

BPE（Byte Pair Encoding，字节对编码）本质上不是先为 NLP 发明的分词算法，而是一种压缩策略。它的核心动作只有一句话：从最小符号集合出发，反复把“出现次数最多的相邻符号对”合并成一个新符号。Sennrich 等人在 2016 年把这个思路引入机器翻译，用它构造子词表，从而把“全词词表太大”和“字符级序列太长”之间的矛盾压到一个可控区间。

在分词场景里，BPE 的价值不是“理解语言”，而是“把高频模式固化为短编码”。常见词或常见词片段会逐步变成单个子词，低频词则继续由已有子词拼出来，因此能显著减弱 OOV（Out-of-Vocabulary，词表外词，指训练时没见过、推理时无法直接映射的词）问题。

玩具例子最容易看出这一点。语料只有三词：`low lower lowest`。如果从字符开始，第一轮常见会合并 `l o -> lo`，第二轮再合并 `lo w -> low`。结果是 `low` 这个高频片段直接进入词表，而 `lower`、`lowest` 仍然可以由 `low + er`、`low + est` 这类子词组合表示。

合并轮数是可计算的。设初始字符词表为 $V_0$，目标词表大小为 $N$，每次合并只新增 1 个符号，则总合并次数是：

$$
T = N - |V_0|
$$

这也是 BPE 的一个工程优点：词表规模是显式可控参数，而不是训练后才知道结果。

下表展示同一语料在字符级与子词级下的差异：

| 词 | 原始字符序列 | 经过若干轮 BPE 后的子词序列 |
|---|---|---|
| low | `l o w </w>` | `low </w>` |
| lower | `l o w e r </w>` | `low er </w>` |
| lowest | `l o w e s t </w>` | `low est </w>` |

这里的 `</w>` 是词尾标记，用来保留单词边界，避免把一个词尾和下一个词首错误合并。

---

## 问题定义与边界

问题可以定义得很直接：给定语料，如何在不保存所有完整单词的前提下，构造一个大小受控、覆盖率足够高的子词词表？

这里“覆盖率”可以白话理解为：语料里有多少高频模式能被较短的 token 序列表示。BPE 不是在寻找语言学上最完美的词素边界，而是在寻找对压缩最有利的高频相邻片段。

以语料 `low lower lowest` 为例，先定义初始词表：

$$
V_0=\{l,o,w,e,r,s,t,</w>\}
$$

如果不设目标词表大小 $N$，算法会不断继续合并，最后甚至可能把整个高频词直接当成一个 token。这样虽然序列更短，但泛化会变差，因为新词、变形词、拼写变化都更难共享参数。反过来，如果 $N$ 太小，很多常见片段无法进入词表，推理时序列会过长。

因此典型停止条件是：

$$
|V| = N
$$

或者更工程化一点：继续合并后，验证集上的平均 token 数下降已不明显，说明再加词表收益很小。

一个简化流程图如下：

```text
字符初始化
   |
统计相邻符号对频率
   |
选择最高频 pair
   |
执行全局合并
   |
更新序列与词表
   |
检查 |V| == N 或收益停止
```

边界也要说清楚：

| 边界问题 | BPE 的处理方式 | 不负责的部分 |
|---|---|---|
| 未登录词 | 通过拆成已有子词处理 | 不保证语义正确 |
| 词表大小控制 | 直接通过目标 $N$ 控制 | 不自动找到“最佳” $N$ |
| 多语言/Unicode | 字符级 BPE 容易受字符集影响 | 纯字符方案可能仍需 `<unk>` |
| 形态学边界 | 可能部分对齐 | 不保证和语言学词素一致 |

所以 BPE 的问题边界很明确：它解决的是“编码与覆盖率”的工程问题，不是“语言结构解析”的理论问题。

---

## 核心机制与推导

BPE 的每一步都是贪心的。贪心的意思是：当前轮只选最能立刻提高压缩率的 pair，而不回看全局最优。

设当前语料被写成一组符号序列，某个相邻对 $(a,b)$ 的频率定义为：

$$
\text{freq}(a,b)=\sum_i \mathbf{1}\big((a,b)\text{ 出现在第 }i\text{ 个序列中}\big)
$$

更准确地说，实际实现通常统计所有出现次数，而不只是出现过一次与否。然后选择频率最大的 pair：

$$
(a^\*, b^\*) = \arg\max_{(a,b)} \text{freq}(a,b)
$$

再把它加入词表作为新符号 $ab$，并在所有序列中做替换。

继续用玩具例子 `low lower lowest`。加入词尾后，初始序列是：

- `l o w </w>`
- `l o w e r </w>`
- `l o w e s t </w>`

前三轮合并可以推成下面这样：

| 轮次 | 主要高频 pair | 频率 | 合并后新增符号 | 序列变化摘要 |
|---|---|---:|---|---|
| 0 | `(l,o)` | 3 | `lo` | 三个词都从 `l o ...` 变成 `lo ...` |
| 1 | `(lo,w)` | 3 | `low` | 三个词都从 `lo w ...` 变成 `low ...` |
| 2 | `(e,r)` 或 `(e,s)` | 1 | `er` 或 `es` | 取决于并列频率时的 tie-break 规则 |

注意第三轮有一个常被忽略的细节：很多教材为了演示方便直接写 `e s -> es`，但在真实实现里，如果多个 pair 同频，谁先合并取决于实现中的排序规则。所以“第三轮一定是 `e s`”不是理论保证，而是某个具体示例里的选择。

把前三轮按一种常见 tie-break 写出来，可以得到：

| 轮次 | 当前词表示例 | 当前词表增量 |
|---|---|---|
| 初始 | `l o w </w>` / `l o w e r </w>` / `l o w e s t </w>` | `{l,o,w,e,r,s,t,</w>}` |
| 合并 `l o` | `lo w </w>` / `lo w e r </w>` / `lo w e s t </w>` | `+ lo` |
| 合并 `lo w` | `low </w>` / `low e r </w>` / `low e s t </w>` | `+ low` |
| 合并 `e s` | `low </w>` / `low e r </w>` / `low es t </w>` | `+ es` |

如果继续下去，`es t` 可能合并成 `est`，于是 `lowest` 会被表示成 `low est </w>`。这正是 BPE 的典型结果：高频前缀和高频后缀各自稳定下来。

伪代码可以压缩成四步：

```text
初始化字符词表 V
while |V| < N:
    统计所有相邻 pair 的频率
    选择最高频 pair
    在语料中全局替换该 pair
    把新符号加入 V
```

从覆盖率角度看，合并次数越多，平均每个词需要的 token 数通常越少，但下降趋势会逐步变缓。可以把它理解成一条“前期下降快、后期趋平”的曲线：

```text
平均 token 数
^
|\
| \
|  \
|   \__
|      \____
+------------> 合并次数
```

前几轮合并会抓住极高频模式，收益最大；后面继续扩词表，收益越来越小。这就是为什么工程上通常不会把词表无限做大。

---

## 代码实现

下面给一个可运行的 Python 版本，只保留 BPE 的核心训练逻辑。它支持：

- 目标词表大小 `target_vocab_size`
- 是否保留词尾边界 `</w>`
- 以字符为基础单位的 BPE 训练

```python
from collections import Counter

END = "</w>"

def tokenize_word(word: str, use_end_of_word: bool = True):
    chars = list(word)
    if use_end_of_word:
        chars.append(END)
    return chars

def build_corpus(words, use_end_of_word: bool = True):
    return [tokenize_word(w, use_end_of_word) for w in words]

def get_vocab(corpus):
    vocab = set()
    for seq in corpus:
        vocab.update(seq)
    return vocab

def count_adjacent_pairs(corpus):
    pairs = Counter()
    for seq in corpus:
        for i in range(len(seq) - 1):
            pairs[(seq[i], seq[i + 1])] += 1
    return pairs

def merge_pair_in_sequence(seq, pair):
    a, b = pair
    merged = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
            merged.append(a + b)
            i += 2
        else:
            merged.append(seq[i])
            i += 1
    return merged

def merge_pair(corpus, pair):
    return [merge_pair_in_sequence(seq, pair) for seq in corpus]

def train_bpe(words, target_vocab_size, use_end_of_word=True):
    corpus = build_corpus(words, use_end_of_word)
    vocab = get_vocab(corpus)
    merges = []

    while len(vocab) < target_vocab_size:
        pairs = count_adjacent_pairs(corpus)
        if not pairs:
            break
        best_pair, best_freq = max(pairs.items(), key=lambda x: (x[1], x[0]))
        if best_freq < 1:
            break
        corpus = merge_pair(corpus, best_pair)
        new_symbol = best_pair[0] + best_pair[1]
        vocab.add(new_symbol)
        merges.append((best_pair, new_symbol, best_freq))

    return corpus, vocab, merges

words = ["low", "lower", "lowest"]
corpus, vocab, merges = train_bpe(words, target_vocab_size=11)

# 前两轮应先得到 lo 和 low
assert merges[0][1] == "lo"
assert merges[1][1] == "low"

# 合并后 low 应该已经成为一个子词
flat = [" ".join(seq) for seq in corpus]
assert any("low" in s for s in flat)

print("merges =", merges[:3])
print("corpus =", corpus)
```

核心函数职责可以用表概括：

| 函数 | 作用 | 新手应关注的点 |
|---|---|---|
| `tokenize_word` | 把单词拆成字符并追加 `</w>` | 边界符决定是否允许跨词尾误合并 |
| `count_adjacent_pairs` | 统计所有相邻 pair 频率 | 这是每轮决策依据 |
| `merge_pair` | 在整个语料中替换目标 pair | 替换必须是全局的，不是只替换一次 |
| `train_bpe` | 控制循环直到达到目标词表大小 | `target_vocab_size` 是主调参项 |

真实工程例子可以看大模型 tokenizer。GPT-2 使用的是字节级 BPE。字节级的白话解释是：不是从“字符”开始，而是从 UTF-8 编码后的单字节开始。这样初始基础单元固定为 256 个字节值，不依赖语言种类。

例如字符 `𝄞`（高音谱号，Unicode U+1D11E）在 UTF-8 下会编码成 4 个字节。字节级 BPE 不需要把它当“未知字符”，而是先把这 4 个字节映射为可打印符号，再按已有 merge 规则继续合并。因此理论上任意 Unicode 字符串都能被分解，不需要 `<unk>`。

这个过程可以写成：

$$
\text{text} \rightarrow \text{UTF-8 bytes} \rightarrow \text{printable symbols} \rightarrow \text{BPE merges}
$$

这也是 GPT-2 一类 tokenizer 能做到“词表固定，但几乎没有 OOV”的基础原因。

---

## 工程权衡与常见坑

BPE 最大的优点是稳定、简单、可控；最大的问题也是简单，因为它的合并规则是贪心的。

第一个权衡是词表大小。假设同一个系统分别训练 `1000` 和 `32000` 的词表，对词 `uncommonword` 的切分可能接近这样：

| 词表大小 | 可能切分 | 平均序列长度 | 参数压力 | 泛化能力 |
|---|---|---:|---:|---|
| 1000 | `u n com mon w ord` | 长 | 小 | 强 |
| 32000 | `un common word` 或 `uncommon word` | 短 | 大 | 较弱 |

这里“参数压力”指嵌入矩阵和输出层都会随着词表变大而变大。词表越大，单 token 表达力越强，但模型参数和稀疏性问题也越明显。

第二个权衡是形态边界。BPE 只知道频率，不知道词法结构。比如英语里某些后缀、德语复合词、土耳其语黏着结构，频率最高的 pair 不一定对应合理词素边界。Bostrom 与 Durrett 2020 的结果说明，BPE 的贪心构造会带来形态切分质量问题，而 unigram LM 在一些设定下更接近形态结构，也可能带来更好的预训练效果。

第三个坑是 tie-break。很多教程默认“第三高频就是某某 pair”，但真实实现里同频 pair 的先后顺序会影响后续全部 merge 链。也就是说：

- 训练代码不同，merge 表可能不同
- merge 表不同，tokenizer 结果会不同
- tokenizer 结果不同，模型输入分布也会不同

第四个坑是边界符。若不保留 `</w>`，某些实现容易把词尾与下一词词首在拼接语料中错误统计为同一 pair，导致训练出的 merge 不稳定。即使最终在线编码不用 `</w>` 显式输出，训练时也常要保留边界信息。

第五个坑是把“无 OOV”误解成“无信息损失”。字节级 BPE 确实能表示任何字符串，但表示得出来，不等于模型容易学会。对非常罕见的字节组合，模型仍然可能缺乏统计强度。

可以把常见经验总结成下面这张表：

| 常见坑 | 现象 | 应对方式 |
|---|---|---|
| 词表过小 | 序列太长，推理慢 | 增大词表或改用更强 tokenizer |
| 词表过大 | 参数增加，长尾泛化差 | 在验证集上比较长度与效果 |
| 贪心误合并 | 子词不贴近词素 | 对形态丰富语言评估替代方案 |
| 不处理边界 | 合并规则不稳定 | 训练时保留词边界标记 |
| 只看压缩率 | token 变短但任务效果不升 | 以任务指标而非长度单独决策 |

---

## 替代方案与适用边界

如果你的目标是“简单、成熟、实现成本低”，BPE 仍然是非常实用的默认选项。但它不是唯一选择。

先看字节级 BPE 和字符级 BPE 的差别：

| 方案 | 基础单位 | 是否天然支持任意 Unicode | 是否容易出现 `<unk>` | 典型代表 |
|---|---|---|---|---|
| 字符级 BPE | 字符 | 依赖字符表覆盖 | 可能 | 早期子词分词器 |
| 字节级 BPE | 256 个字节 | 是 | 基本不会 | GPT-2 |
| WordPiece | 子词 + 概率/启发式选择 | 通常支持 | 依实现而定 | BERT 系 |
| Unigram LM | 候选子词集合上的概率模型 | 支持较好 | 依实现而定 | SentencePiece |

字节级 BPE 特别适合开放文本环境。原因是互联网文本里会混合：

- 多语言字符
- emoji
- 数学符号
- 稀有标点
- 编码脏数据

纯字符 BPE 如果初始字符表不完整，就必须引入未知符号；字节级 BPE 则退回到字节层面，总能编码。

再看与 unigram LM 的差别。unigram LM 的白话解释是：先准备一大批候选子词，再通过概率模型删除不重要的候选，最终保留一组整体解释概率更好的子词表。它不是像 BPE 那样一步步只做局部最优合并，因此在形态边界更复杂的语言里，常比 BPE 更稳。

适用边界可以概括为：

| 场景 | 更合适的方案 | 原因 |
|---|---|---|
| 通用 LLM、网页语料、多语言脏文本 | 字节级 BPE | 无 OOV，工程实现成熟 |
| 形态丰富语言、重视词素边界 | unigram LM | 不受贪心合并强约束 |
| 兼容既有 BERT 生态 | WordPiece | 与预训练模型体系一致 |
| 教学、原理展示、小实验 | 字符级 BPE | 最容易讲清楚 |

因此“BPE 是否最好”的正确说法不是绝对判断，而是条件判断：

- 如果你优先要稳定和工业兼容性，选字节级 BPE。
- 如果你优先要更合理的子词边界，尤其是形态复杂语言，unigram LM 往往值得比较。
- 如果你只是想理解现代 tokenizer 的基础机制，先学字符级 BPE 最合适。

---

## 参考资料

| 资料 | 作者/站点 | 主要贡献 | 适用章节 |
|---|---|---|---|
| [A New Algorithm for Data Compression](https://dl.acm.org/doi/10.5555/177910.177914) | Philip Gage, 1994 | BPE 的压缩算法来源 | 核心结论、核心机制与推导 |
| [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/) | Rico Sennrich, Barry Haddow, Alexandra Birch, 2016 | 把 BPE 引入 NLP，说明如何缓解稀有词和 OOV | 核心结论、问题定义与边界、核心机制与推导 |
| [Language Models are Unsupervised Multitask Learners](https://github.com/openai/gpt-2) | Radford 等，OpenAI，2019 | GPT-2 实践中使用字节级 BPE；配套代码展示 bytes-to-unicode 设计 | 代码实现、替代方案与适用边界 |
| [Better language models and their implications](https://openai.com/blog/better-language-models/) | OpenAI | GPT-2 的工程背景与 tokenizer 使用场景 | 替代方案与适用边界 |
| [Byte Pair Encoding is Suboptimal for Language Model Pretraining](https://aclanthology.org/2020.findings-emnlp.414/) | Kaj Bostrom, Greg Durrett, 2020 | 讨论 BPE 贪心构造的局限，并比较 unigram LM | 工程权衡与常见坑、替代方案与适用边界 |
| [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://aclanthology.org/D18-2012/) | Taku Kudo, John Richardson, 2018 | 提供 unigram LM 与独立于空格语言的训练框架 | 替代方案与适用边界 |
| [subword-nmt](https://github.com/rsennrich/subword-nmt) | Sennrich 团队代码 | 经典 BPE 训练与编码实现 | 代码实现 |
