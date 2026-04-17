## 核心结论

BPE，Byte Pair Encoding，中文常译为“字节对编码”，本质上是一种**频率驱动的贪心子词构造算法**。贪心的意思是每一轮都只做当前看起来最划算的一次合并，不回头整体重算最优解。它从最小的符号集合开始，通常是字符集合；如果做字节级 BPE，起点就是 256 个字节。然后反复统计语料里所有相邻符号对的出现频率，选出频率最高的一对，把它合并成一个新符号，并把这个新符号加入词表。

形式化地说，第 $t$ 轮的选择是：

$$
x^*, y^* = \arg\max_{x,y} f_t(x,y)
$$

其中 $f_t(x,y)$ 表示当前切分状态下，相邻符号对 $(x,y)$ 在整个训练语料中的总频次。完成合并后，词表更新为：

$$
V_t = V_{t-1} \cup \{x^* \circ y^*\}
$$

这里 $\circ$ 表示把两个旧符号拼成一个新符号。

这套机制的直接结果有三点：

| 结论 | 含义 | 工程价值 |
|---|---|---|
| 词表从小到大逐步长出 | 先有字符/字节，再有高频子词 | 避免直接维护超大词表 |
| 高频片段会被压缩成单个 token | 常见前后缀、词干、空格前缀都容易形成稳定子词 | 降低序列长度，提高模型效率 |
| 训练后得到的是“合并顺序” | 推理时按训练好的合并规则切词 | 训练与推理逻辑统一 |

玩具例子可以先看最小版本。假设初始词表只有 `{a, b, c, d}`，语料里 `"ab"` 出现 5 次，`"bc"` 出现 3 次，`"cd"` 出现 1 次，那么第一轮应该合并的是频率最高的 `"ab"`，而不是 `"bc"`。如果把例子改成 `"bc"` 出现 6 次，那么第一轮就合并 `"bc"`，随后所有包含 `b c` 的地方都改写成 `bc`。BPE 的“学习结果”不是一句规则，而是一串合并历史，比如：

`(b, c) -> bc`  
`(a, bc) -> abc`  
`(d, e) -> de`

真实工程里，GPT-2、RoBERTa、BART 一类模型都用过 BPE 或其近亲变体。原因很直接：它兼顾了开放词表和有限词表。开放词表的意思是未登录词也能拆成更小单元；有限词表的意思是 embedding 表大小可控，通常控制在 32K 到 50K 左右。

---

## 问题定义与边界

BPE 要解决的问题不是“找词典”，而是：**在给定词表大小预算的前提下，找到一套足够稳定的子词单元，让任意输入都能被拆成有限 token 序列**。

这里的“子词”可以理解为“比字符大、比完整单词小的可复用片段”。例如英文里的 `ing`、`tion`、`pre`，中文里如果做字节级训练，也会落到字节片段而不是汉字词典。

它的输入边界主要有三类。

第一类是**初始符号边界**。标准 BPE 常从字符开始，字节级 BPE 从 256 个字节开始。两者差别不是思想，而是输入单位。字节级的好处是任何文本都能编码，不会因为字符集不全而出现未知符号。

第二类是**词边界或空格边界**。边界的意思是“某个位置不能被当成普通字符随意吞并”。如果不保留边界信息，BPE 会把空格附近的模式也学进去，产生不稳定甚至不可控的 token。比如 `"Hello world"` 如果把空格当普通字符处理，可能得到 `"o "` 这样的合并；这在某些实现里是允许的，在另一些实现里会破坏预期。很多实现会显式加入 `</w>` 这样的词尾标记，或者像 GPT-2 那样把空格编码成前缀标记。

第三类是**预分词边界**。预分词可以理解为在做 BPE 之前，先用正则把文本切成较粗的块，例如单词、数字、标点。这一步不是 BPE 的数学定义必需，但工程上非常常见，因为它能防止一些无意义拼接。

下面这张表把“边界符号”和“普通字符”放在一起看，会更清楚：

| 类型 | 例子 | 是否表示内容本身 | 是否参与普通合并 | 主要作用 |
|---|---|---|---|---|
| 普通字符 | `a`、`b` | 是 | 是 | 构成子词主体 |
| 词尾标记 | `</w>` | 否 | 通常受限制 | 防止跨词误合并 |
| 空格前缀 | `Ġhello` 中的 `Ġ` | 否 | 允许但有明确语义 | 表示 token 前面有空格 |
| 预分词边界 | regex 切出的块 | 否 | 块内合并，块间不合并 | 控制搜索空间 |

对零基础读者，一个容易混淆的点是：BPE 并不保证“语言学上合理”。它只保证“统计上高频”。所以它可能学出 `tion` 这种看起来合理的后缀，也可能学出 `##ing`、`Ġthe`、`abcd` 这种纯粹由频率驱动的片段。只要能稳定压缩语料并支持推理阶段复现，这就是合格的 BPE 词表。

---

## 核心机制与推导

BPE 的核心状态通常由两份数据结构维护。

第一份是 `word_frequencies`，记录“每个词当前如何被拆分，以及它在语料中出现多少次”。这里的“词”是训练时的基本样本，可以是预分词后的单词，也可以是字节块。  
第二份是 `pair_frequencies`，记录“所有相邻符号对在当前状态下的总频次”。

如果把一个词 $w$ 当前的分词结果写成 $[s_1, s_2, \dots, s_k]$，那么某一对相邻符号 $(x,y)$ 的总频次可以写成：

$$
f_t(x,y) = \sum_w \text{count}_w(x,y)
$$

其中 $\text{count}_w(x,y)$ 表示这对相邻符号在词 $w$ 的当前切分里出现了几次，再乘上该词本身的词频。于是每一轮都做同一件事：

1. 在 `pair_frequencies` 里找频率最大的 pair。
2. 生成新符号 $z = x^* \circ y^*$。
3. 把所有词里出现的 `x^* y^*` 改写成 `z`。
4. 只更新受影响词周围的 pair 频率。

这个“只更新受影响词”是实现 BPE 的关键。否则每一轮都重新扫描全部语料，复杂度会非常高。

看一个玩具例子。假设当前有三个词：

- `b c d`，频次 7
- `a b c`，频次 5
- `c d e`，频次 2

此时 pair 频率是：

- `(b, c)` 出现 $7 + 5 = 12$
- `(c, d)` 出现 $7 + 2 = 9$
- `(a, b)` 出现 $5$
- `(d, e)` 出现 $2$

所以第一轮合并 `(b, c)`。合并后：

- `b c d` 变成 `bc d`
- `a b c` 变成 `a bc`
- `c d e` 不变

更新时不用重扫全部 pair，只需要关心被改写的词。因为 `(b, c)` 消失了，新的相邻关系变成：

- 在 `bc d` 中新增 `(bc, d)`
- 在 `a bc` 中新增 `(a, bc)`

可以把局部更新的思想写成伪代码：

```python
for pair in pairs_removed_from_old_word:
    pair_frequencies[pair] -= word_freq

for pair in pairs_added_from_new_word:
    pair_frequencies[pair] += word_freq
```

改写规则可以写成：

$$
w \rightarrow w|_{x^*y^* \rightarrow z}
$$

意思是：把词 $w$ 的当前切分中所有相邻的 $x^*y^*$ 替换成新符号 $z$。

真实工程例子里，假设你训练英文技术博客的 tokenizer，语料中 `"token"`, `"tokenizer"`, `"tokenization"` 都很多，那么很可能先学出 `to`、`ken`、`token`，再进一步学出 `ization`。这是因为这些片段在多词之间可复用，频率累计很快。BPE 不需要知道 `ization` 是后缀，它只会看到高频相邻片段。

---

## 代码实现

下面给出一个可运行的简化 Python 实现。它保留了 BPE 的核心结构：`word_frequencies`、pair 统计、按最高频 pair 合并，并带有 `assert` 做基本验证。为了让逻辑清楚，这里用字符级样本，并用 `</w>` 表示词尾边界。

```python
from collections import Counter

def tokenize_word(word):
    return tuple(list(word) + ["</w>"])

def get_pair_frequencies(vocab):
    pair_freq = Counter()
    for symbols, word_freq in vocab.items():
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_freq[pair] += word_freq
    return pair_freq

def merge_vocab(pair, vocab):
    merged = {}
    left, right = pair
    new_symbol = left + right

    for symbols, word_freq in vocab.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == left and symbols[i + 1] == right:
                new_symbols.append(new_symbol)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        merged[tuple(new_symbols)] = merged.get(tuple(new_symbols), 0) + word_freq
    return merged

def train_bpe(word_frequencies, num_merges):
    vocab = {tokenize_word(word): freq for word, freq in word_frequencies.items()}
    merges = []

    for _ in range(num_merges):
        pair_freq = get_pair_frequencies(vocab)
        if not pair_freq:
            break
        best_pair, best_freq = pair_freq.most_common(1)[0]
        if best_freq <= 0:
            break
        merges.append(best_pair)
        vocab = merge_vocab(best_pair, vocab)

    return vocab, merges

def encode_word(word, merges):
    symbols = list(word) + ["</w>"]
    for left, right in merges:
        new_symbol = left + right
        i = 0
        merged = []
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == left and symbols[i + 1] == right:
                merged.append(new_symbol)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        symbols = merged
    return symbols

corpus = {
    "low": 5,
    "lower": 2,
    "newest": 6,
    "widest": 3,
}

final_vocab, merges = train_bpe(corpus, num_merges=10)

encoded_low = encode_word("low", merges)
encoded_newest = encode_word("newest", merges)

assert encoded_low[-1].endswith("</w>") or encoded_low[-1] == "</w>"
assert len(merges) > 0
assert isinstance(final_vocab, dict)
assert all(freq > 0 for freq in final_vocab.values())
```

这个版本不是最快，但结构是对的。训练流程可以概括成：

| 步骤 | 数据结构 | 作用 |
|---|---|---|
| 初始化 | `word_frequencies` | 把每个词拆成字符序列加边界 |
| 统计 pair | `Counter` | 找到当前最频繁的相邻对 |
| 执行合并 | 新旧词表示映射 | 把 `x y` 改成 `xy` |
| 记录 merge list | 列表 | 推理阶段复现训练顺序 |

真正的工程实现会进一步优化两件事。

第一，维护“哪些词包含某个 pair”的倒排索引。倒排索引就是从 pair 反查到词集合，这样合并 `(a, b)` 时，只处理含有 `(a, b)` 的词。  
第二，用增量更新而不是每轮全量 `get_pair_frequencies`。这也是为什么很多高性能实现会同时维护“词当前切分”和“pair 到频次”的双向状态。

真实工程例子可以看 GPT 类 tokenizer 的训练逻辑。它通常不会直接对原始整句做字符级遍历，而是先正则预分块，再做字节级编码，然后训练 BPE merge 表。推理时，只要把输入做同样的预处理，再按 merge 顺序逐步合并，就能得到与训练一致的 token 序列。

---

## 工程权衡与常见坑

BPE 看起来简单，但真正写到可用版本，坑主要集中在复杂度和边界处理。

最常见的问题是**重复扫描全部语料**。如果每一轮都重新遍历所有词并重建全部 pair 频率，复杂度近似会落到 $O(m \times W \times L)$，其中 $m$ 是合并轮数，$W$ 是词数，$L$ 是平均词长。词表要长到几万时，这个代价很快不可接受。更合理的做法是只更新受影响词，把每轮成本压到与 `affected_words` 和局部 pair 数相关的范围。

第二个问题是**边界丢失**。如果不保留 `</w>`、空格前缀或预分词边界，模型会学出跨边界的 token，导致编码结果不稳定。对新手来说，一个直观错误是把 `"hello world"` 当成纯字符流训练，结果 `"o "` 这种带空格尾巴的片段大量出现，而你又没有定义解码语义，后续就会混乱。

第三个问题是**零频 pair 不删除**。如果 `pair_frequencies` 里保留了已经被减到 0 或负数的条目，后续选最大值时容易出现脏状态。脏状态的意思是数据结构里残留了逻辑上已经失效的数据。

第四个问题是**训练和推理的预处理不一致**。训练时如果用了小写化、正则切分、字节编码，推理时漏掉其中一步，token 序列就会完全不同。

下面用表格列出常见坑和缓解方式：

| 常见坑 | 后果 | 缓解方式 |
|---|---|---|
| 每轮全量重扫所有词 | 训练时间爆炸 | 维护 `affected_words` 与增量 pair 更新 |
| 不保留边界信息 | 学出跨词或跨空格异常 token | 使用 `</w>`、空格前缀或 regex 预分词 |
| 不删除 0 频条目 | 选最大 pair 时出现脏数据 | 频率减到 0 立即删除 |
| 合并逻辑只替换首个匹配 | 同一词内多处 pair 漏合并 | 顺序扫描整词，处理所有非重叠匹配 |
| 训练/推理预处理不一致 | 同一文本编码不一致 | 固化同一套标准化与分块规则 |

再看一个新手版例子。假设语料里有 10 万个词条，当前要合并 `"ab"`。如果只有 `"abc"`、`"abd"`、`"zabx"` 三种词包含 `"ab"`，那就只改这三种词，不需要重新扫剩下 99997 个词条。这正是高性能 BPE 的基本思路。

---

## 替代方案与适用边界

BPE 不是唯一的子词算法。最常被拿来比较的是 WordPiece 和 SentencePiece。

WordPiece 的核心差别不在“也会合并子词”，而在**合并决策规则**。它不是单纯看绝对频率，而是看一种近似的联合强度：

$$
score(AB) = \frac{f(AB)}{f(A)f(B)}
$$

直观理解是：如果 `A` 和 `B` 单独都很常见，但它们组合起来并不特别稳定，那么这个分数不会高。这样做的目的，是避免总被高频单字符误导，更偏向挑选“粘得很紧”的组合。

举个新手版例子，`t` 和 `h` 都很常见，但只有当 `th` 的联合出现相对足够强时，WordPiece 才会优先合并它。BPE 则可能因为 `th` 的绝对频次高而较早合并。

SentencePiece 是一个更广的工具框架，既支持 BPE，也支持 Unigram。Unigram 可以理解为“先准备一批候选子词，再通过概率模型删掉不划算的子词”。它不是逐轮只加一个，而是从一个较大的候选集合里优化保留子集。一个简化表达是：

$$
P(x) = \prod_{i=1}^{n} P(s_i)
$$

其中 $x$ 被切成一串子词 $s_1, s_2, \dots, s_n$，目标是找到整体概率更高的切分，并通过 EM 一类方法调整子词集合。对无空格语言、多语种混合、需要子词采样增强的场景，SentencePiece 往往比手写 BPE 更方便。

三者适用边界可以这样理解：

| 方法 | 合并/选择依据 | 优势 | 适用边界 |
|---|---|---|---|
| BPE | 绝对频率最高 pair | 简单、稳定、易实现 | 通用 tokenizer，工程可控 |
| WordPiece | $\frac{f(AB)}{f(A)f(B)}$ | 更强调组合紧密度 | 语言模型、需要更保守合并时 |
| SentencePiece Unigram | 概率模型选择子词集合 | 对无空格语言友好，支持采样 | 多语种、端到端训练工具链 |

所以结论不是“谁绝对更好”，而是“目标不同”。如果你的目标是理解原理并从零实现，BPE 是最合适的起点；如果目标是直接训练工业级 tokenizer，通常会优先考虑成熟库里的 WordPiece 或 SentencePiece。

---

## 参考资料

| 来源 | 重点 | 补充内容 |
|---|---|---|
| [Emergent Mind: Byte-Pair Encoding (BPE) Algorithm](https://www.emergentmind.com/topics/byte-pair-encoding-bpe-algorithm?utm_source=openai) | 给出 BPE 的频率迭代公式、词表扩展形式化定义 | 适合理清 $x^*, y^*=\arg\max f_t(x,y)$ 与 $V_t$ 更新 |
| [Jun Yu Tan: Building a Fast BPE Tokenizer from Scratch](https://jytan.net/blog/2025/bpe/) | 详细解释 `word_frequencies` 与 `pair_frequencies` 的 Python 维护方式 | 对“增量更新而不是全量扫描”讲得最实用 |
| [Hugging Face Course: BPE](https://huggingface.co/docs/course/en/chapter6/5?utm_source=openai) | 用教学例子展示 BPE 的训练与编码流程 | 适合理解实际 tokenizer 工具链中的位置 |
| [GeeksforGeeks: Byte Pair Encoding in NLP](https://www.geeksforgeeks.org/byte-pair-encoding-bpe-in-nlp/?utm_source=openai) | 提供直观小例子 | 适合快速建立第一层直觉 |

BPE 的关键不在“会不会写一个统计频次的脚本”，而在于你是否明确维护了三件事：初始符号边界、pair 频率状态、训练与推理一致的合并顺序。把这三件事写对，BPE 就不是黑盒，而是一套非常可解释的子词构造流程。
