## 核心结论

子词分词指把原始文本切成“比整词更小、比单字符更有信息量”的单位。它的目标不是“切得像人类理解的词”，而是在有限词表下，让模型尽量少遇到未登录词，同时不要把序列拉得过长。

BPE、WordPiece、Unigram 都在做同一件事：在“词表大小”和“语料覆盖率”之间找平衡，但三者的优化方向不同。

- BPE 可以直观理解为“把最常见的字符对粘在一起”。它按频次贪心合并，高频片段会很快变成稳定子词。
- WordPiece 可以直观理解为“只合并不会降低整段概率的片段”。它不是只看出现次数，而是看这次合并是否更有利于训练语料的解释。
- Unigram 可以直观理解为“先准备一个大词表，再从里面删掉小概率项”。它直接给子词建概率模型，保留对整体似然贡献大的词。

三者的差别会直接影响工程结果。词表越大，通常 OOV 率越低，序列越短，但 embedding 和 softmax 成本更高；词表越小，泛化更稳，但 token 数会增多，训练和推理都会变慢。对英语这类空格清晰、词形规律强的语料，BPE 常常足够高效；对需要稳健处理未见片段的预训练编码器，WordPiece 常见；对多语种、低资源和复杂脚本场景，Unigram 通常更灵活。

下表先给出横向对比：

| 算法 | 核心策略 | 训练迭代方式 | 优点 | 主要代价 | 常见场景 |
|---|---|---|---|---|---|
| BPE | 贪心合并最高频相邻子串 | 每轮选最高频 pair 合并 | 实现简单，训练快 | 容易被频次偏置误导 | GPT 类、通用生成模型 |
| WordPiece | 选择最有利于语料似然的合并 | 每轮重算候选得分 | 分词更稳，兼顾概率解释 | 训练更慢 | BERT 类编码器 |
| Unigram | 从大词表中按概率剪枝 | EM 重估 + 删除低贡献词 | 多语种适配强，支持采样分词 | 实现复杂，训练成本高 | SentencePiece、多语种模型 |

---

## 问题定义与边界

问题可以严格表述为：给定原始字符序列 $x$，构造一个有限词表 $V$，把 $x$ 映射成 token 序列 $t_1,t_2,\dots,t_n$，使模型既能覆盖训练和推理文本，又不因序列过长而导致计算成本失控。

这里有三个必须明确的边界。

第一，输入粒度。输入可以从字节开始，也可以从字符开始。字节级输入的白话解释是“任何文本最终都能拆成基础字节，所以理论上零 OOV”，代价是序列可能更长。字符级输入更符合很多语言的表面形式，但会受到字符集设计影响。

第二，资源边界。词表预算不是无限的。设词表大小为 $|V|$，embedding 参数量大致与 $|V| \times d$ 成正比，$d$ 是向量维度。词表从 30k 增到 100k，覆盖率会变好，但参数、显存和 softmax 代价都会上升。

第三，目标模型。编码器、解码器、Seq2Seq、多语种模型对 tokenizer 的偏好不同。生成模型通常更关心吞吐量和实现稳定性，编码器更关心对词形变化和未知词的处理，多语种模型更关心不同脚本之间的统一覆盖。

可以把约束整理成一个边界表：

| 维度 | 需要回答的问题 | 对算法选择的影响 |
|---|---|---|
| 输入类型 | 字符级还是字节级 | 决定是否天然零 OOV |
| 语料分布 | 单语还是多语，是否低资源 | 决定频次统计是否可靠 |
| 词表预算 | 8k、32k、100k 还是更大 | 决定序列长度与参数开销 |
| 模型架构 | 编码器、解码器还是 Seq2Seq | 决定对稳定性和效率的偏好 |
| 预处理能力 | 是否允许依赖空格、分词器、规则 | 决定能否用语言特定假设 |

玩具例子可以先看中文。假设语料很小，只出现了“机器学习”“机器视觉”“机器翻译”。如果只按频次做 BPE，模型可能很快学到“机器”这个长片段，却对“学习”“视觉”“翻译”之后的新组合不够稳健。更糟的是，如果初始词表没有保留所有基础字符，遇到新词时会直接失去可分解路径。所以中文或低资源语言里，通常要显式保留字符级初始词表，保证任何词都还能回退到字符。

真实工程里，多语种场景更明显。一个面向 50 种语言的模型，不可能依赖每种语言都先做好人工分词。此时 tokenizer 必须在不同脚本、不同词边界规则下都能工作，这会明显提升 Unigram 或 byte fallback 方案的价值。

---

## 核心机制与推导

### 1. BPE：频次驱动的贪心合并

BPE 的核心规则是：统计所有相邻子串对 $(A,B)$ 的频次，选择 $\mathrm{freq}(AB)$ 最大的 pair 合并。形式上就是反复执行：

$$
(A^\*, B^\*) = \arg\max_{(A,B)} \mathrm{freq}(AB)
$$

然后把所有相邻的 $A^\*B^\*$ 替换成一个新 token。

白话解释是：谁总是一起出现，就先把谁绑定成一个更大的块。

玩具例子用语料 `a b a b c`。初始 token 为单字符：

- 序列：`a | b | a | b | c`
- 相邻 pair：`ab, ba, ab, bc`
- 频次：`ab=2, ba=1, bc=1`

所以第一步合并 `ab`：

- 新序列：`ab | ab | c`
- 新词表：`{a, b, c, ab}`

如果词表预算很紧，BPE 到这里可能就停止；如果预算更大，还会继续看 `ab ab` 或 `ab c` 这类更长片段。

BPE 的问题也来自它的优点。它只看频次，不看“这个合并是否真的提高了整个语料的解释质量”。所以在分布偏斜、样本稀少或多语种混合时，它可能学到局部非常高频、但泛化并不好的长串。

### 2. WordPiece：面向似然的合并

WordPiece 仍然是“逐步构造词表”，但不直接按频次选 pair。常见直觉公式是：

$$
\mathrm{score}(A,B) = \frac{\mathrm{freq}(AB)}{\mathrm{freq}(A)\cdot \mathrm{freq}(B)}
$$

它的白话解释是：不是看 `AB` 出现得多不多，而是看“相对于 A 和 B 各自已经很常见这件事，`AB` 是否异常地紧密”。

这样做的效果是，某些虽然频次高、但只是因为 A 和 B 自己都很常见的组合，不会被过早合并；真正“绑定强”的组合得分更高。

还是看 `a b a b c`：

- `freq(a)=2, freq(b)=2, freq(c)=1`
- `freq(ab)=2, freq(ba)=1, freq(bc)=1`

则：

$$
\mathrm{score}(ab)=\frac{2}{2\cdot 2}=0.5
$$

$$
\mathrm{score}(ba)=\frac{1}{2\cdot 2}=0.25,\quad \mathrm{score}(bc)=\frac{1}{2\cdot 1}=0.5
$$

这说明 WordPiece 至少要比较“谁更有独立绑定价值”，而不是仅凭出现次数。真实实现里，它通常和训练语料概率最大化联系在一起，因此每轮候选评估比 BPE 更贵。

### 3. Unigram：直接对子词集合建概率模型

Unigram 的出发点不同。它不从小词表往上长，而是从一个较大的候选词表开始，然后通过概率重估和剪枝得到更优子集。

若一个切分结果是 $t_1,t_2,\dots,t_n$，其概率写成：

$$
P(x) = \prod_{i=1}^{n} P(t_i)
$$

整个语料 $C$ 的目标似然可以写成：

$$
L(C) = \prod_{x \in C} P(x)
$$

通常会取对数，变成更容易优化的对数似然。核心思想是：如果删掉某个 token 后，整体似然下降很小，说明它不是必要词，可以剪掉。

玩具例子仍然用候选集合 `{a,b,c,ab,bc}`，假设初始概率为：

| token | 概率 |
|---|---|
| a | 0.3 |
| b | 0.3 |
| c | 0.2 |
| ab | 0.1 |
| bc | 0.1 |

如果删掉 `bc` 后，所有语料仍能由 `b` 和 `c` 组合恢复，且整体似然下降很小，那么 `bc` 会被剪掉。最终词表不是“合并出来的”，而是“筛选留下的”。

这也是 Unigram 更适合多语种和不规则文本的原因。它天然允许一个字符串有多种切分方式，再通过概率来判断哪种切分更合理。

### 4. 一个统一视角

三者都在最优化一个目标，只是近似方式不同：

- BPE：用局部最高频作为全局收益的近似。
- WordPiece：用候选合并对整体概率的提升作为准则。
- Unigram：直接把“哪组子词最能解释语料”写成概率模型，再做删减。

可以把流程压缩成一张文字流程图：

| 算法 | 初始化 | 每轮动作 | 停止条件 |
|---|---|---|---|
| BPE | 单字符/字节词表 | 合并最高频 pair | 达到目标词表大小 |
| WordPiece | 基础词表 | 选最高 score 的 pair | 达到目标词表大小 |
| Unigram | 大候选词表 | EM 重估后剪枝 | 剪到目标词表大小 |

---

## 代码实现

下面给一个可运行的最小 Python 例子，演示三种策略的核心判断，不追求工业级性能，但能把训练循环讲清楚。

```python
from collections import Counter
from math import prod

def bpe_best_pair(tokens):
    pairs = Counter(zip(tokens, tokens[1:]))
    return max(pairs.items(), key=lambda x: x[1])[0]

def wordpiece_best_pair(tokens):
    token_freq = Counter(tokens)
    pair_freq = Counter(zip(tokens, tokens[1:]))
    def score(pair):
        a, b = pair
        return pair_freq[pair] / (token_freq[a] * token_freq[b])
    return max(pair_freq, key=score)

def unigram_sequence_prob(segmentation, probs):
    return prod(probs[t] for t in segmentation)

# 玩具语料
tokens = ["a", "b", "a", "b", "c"]

# BPE: 频次最高 pair 应该是 ("a", "b")
assert bpe_best_pair(tokens) == ("a", "b")

# WordPiece: 在这个例子里 ("a","b") 和 ("b","c") 分数都可能领先
wp = wordpiece_best_pair(tokens)
assert wp in {("a", "b"), ("b", "c")}

# Unigram: 比较两种切分概率
probs = {"a": 0.3, "b": 0.3, "c": 0.2, "ab": 0.1, "bc": 0.1}
p1 = unigram_sequence_prob(["ab", "ab", "c"], probs)
p2 = unigram_sequence_prob(["a", "b", "a", "bc"], probs)

assert abs(p1 - 0.002) < 1e-12
assert abs(p2 - 0.0027) < 1e-12
assert p2 > p1
```

上面这个例子说明两件事：

1. BPE 的核心就是找最高频 pair。
2. Unigram 的核心不是“最长匹配”，而是“哪种切分概率更高”。

如果写成更接近工程实现的伪代码，可以分别理解成下面三段。

```python
# BPE 训练主循环
vocab = init_char_vocab(corpus)
segments = split_to_chars(corpus)

while len(vocab) < target_vocab_size:
    pair_freq = count_adjacent_pairs(segments)
    best_pair = argmax(pair_freq)
    segments = merge_pair(segments, best_pair)
    vocab.add(concat(best_pair))
```

```python
# WordPiece 训练主循环
vocab = init_base_vocab(corpus)
segments = segment_with_vocab(corpus, vocab)

while len(vocab) < target_vocab_size:
    pair_freq = count_adjacent_pairs(segments)
    token_freq = count_tokens(segments)
    best_pair = argmax_score(pair_freq, token_freq)  # freq(ab)/(freq(a)*freq(b))
    vocab.add(concat(best_pair))
    segments = resegment(corpus, vocab)
```

```python
# Unigram 训练主循环
vocab = init_large_candidate_vocab(corpus)
probs = init_probs(vocab)

while len(vocab) > target_vocab_size:
    probs = em_update(corpus, vocab, probs)
    loss_if_removed = estimate_loss_for_each_token(corpus, vocab, probs)
    removable = select_low_impact_tokens(loss_if_removed)
    vocab = vocab - removable
```

真实工程例子里，Hugging Face `tokenizers` 或 SentencePiece 会把这些过程做得更复杂：

- BPE 常用堆、索引表或增量更新来避免每轮全量重算。
- WordPiece 需要反复重估切分和分数，训练明显更慢。
- Unigram 一般依赖 EM，EM 的白话解释是“先根据当前词表猜切分，再根据切分重新估概率，循环直到稳定”。

---

## 工程权衡与常见坑

真正上线时，选择 tokenizer 不是算法竞赛，而是系统设计问题。

第一类权衡是序列长度和词表大小。词表大，平均 token 数少；词表小，平均 token 数多。设一段文本长度固定，token 数从 200 增到 260，注意力计算、缓存和吞吐都会受影响。大模型里，这个差异很容易转化为实际成本。

第二类权衡是语言分布。BPE 在英语上常常效果很好，因为高频词缀、空格边界、复用片段都比较稳定；但在中文、日文、泰文或低资源语言里，单纯频次可能不够。比如中文小语料若高频出现“人工智能”，BPE 可能迅速把它学成整体 token，但遇到“人工治理”“智能制造”时，分解能力未必理想。

第三类权衡是训练成本。BPE 训练最省，WordPiece 更贵，Unigram 最复杂。很多团队不是“不知道哪个好”，而是“承受不起更复杂方案的训练和调参成本”。

下面列常见坑与规避方式：

| 风险/坑 | 现象 | 规避措施 |
|---|---|---|
| BPE 过度依赖频次 | 学到局部高频长串，泛化差 | 保留字符级初始词表，控制最大 token 长度 |
| 中文只做简单 BPE | 新词切分碎或长串不稳定 | 保留全部汉字覆盖，必要时加 byte fallback |
| WordPiece 训练耗时高 | 候选多，重算分词慢 | 做语料采样，限制候选长度 |
| Unigram EM 不稳定 | 早期概率震荡，剪枝过猛 | 分阶段剪枝，保留最小字符覆盖 |
| 多语种脚本差异大 | 某些语言 OOV 或 token 过长 | 用 SentencePiece + 原始字节/显式空格 |
| 只看 OOV 不看长度 | OOV 降了但推理更慢 | 同时监控平均 token 数和下游延迟 |

玩具例子里也能看到坑。若 Unigram 删除 `bc` 后，语料仍可用 `b` 和 `c` 恢复，那么这是合理剪枝；但如果删除后某些字符串只能用极长切分恢复，虽然“理论可覆盖”，实际序列长度会恶化。所以工程上不能只看可恢复性，还要看切分质量分布。

真实工程例子可以看两类模型偏好：

- GPT 类模型常用 BPE 或 byte-level BPE，因为实现成熟、吞吐稳定、高频英语片段压缩效果好。
- mBART、ALBERT 这类多语种或 SentencePiece 生态模型更常见 Unigram，因为它更容易在统一框架下处理多种语言和不规则边界。

---

## 替代方案与适用边界

BPE、WordPiece、Unigram 不是唯一选择。工程上常见的替代路线至少还有三类。

第一，byte-level BPE。它的白话解释是“先把所有文本都拆成字节，再在字节序列上做 BPE”。优势是理论上零 OOV，因为任何字符都能表示成字节。GPT 系列偏爱这一思路，核心考量不是“语义更好”，而是“覆盖稳定、预处理简单、跨文本来源鲁棒”。

第二，SentencePiece Unigram。它通常直接在原始文本上工作，不强依赖空格或预先分词。多语种模型偏爱它，是因为不同语言可以共享一套训练框架。把它类比成“先准备一个足够大的零件库，再删掉没用的零件”，而不是边看边拼。

第三，纯字符级 tokenizer。它最简单，也最稳，因为不存在 OOV；但序列往往太长。它更适合极小模型、极受限设备，或对字符级变化极敏感的任务，不适合大多数通用 LLM 预训练。

可以做一个适用边界对比：

| 方案 | 零 OOV 能力 | 训练成本 | 平均序列长度 | 适合场景 |
|---|---|---|---|---|
| BPE | 中 | 低 | 中 | 单语、高吞吐生成 |
| WordPiece | 中 | 中 | 中 | 编码器、语义稳健切分 |
| Unigram | 中到高 | 高 | 中 | 多语种、复杂脚本 |
| byte-level BPE | 高 | 中 | 偏长 | 通用生成、原始文本混杂 |
| 纯字符级 | 高 | 极低 | 高 | 极简系统、字符敏感任务 |

为什么常说“GPT 系列更像 BPE 路线，而 mBART 更像 Unigram 路线”？核心不是品牌差异，而是目标差异。

- GPT 类生成模型重视大规模训练下的稳定吞吐、统一覆盖和成熟实现，byte-level BPE 很符合这个目标。
- mBART 这类多语种 Seq2Seq 模型要同时处理多脚本、多边界语言，Unigram + SentencePiece 更容易统一管理。

如果是新项目，简单决策可以这样做：

- 单语、以生成效率为主：优先 BPE 或 byte-level BPE。
- 编码器任务、强调词形和概率解释：考虑 WordPiece。
- 多语种、低资源、语言边界复杂：优先 SentencePiece Unigram。
- 极端要求零 OOV：优先 byte fallback 或纯字节方案。

---

## 参考资料

1. Bomberbot, *How to Train BPE, WordPiece, and Unigram Tokenizers from Scratch using Hugging Face*  
   重点：对三种 tokenizer 的直觉解释很清楚，直接支撑“BPE 是粘高频对、WordPiece 看概率、Unigram 做剪枝”的入门表述。

2. Hugging Face, *Tokenizer Summary*  
   重点：对 BPE、WordPiece、Unigram 的核心公式和设计目标有较规范的总结，可直接支撑文中的频次、score 与似然函数部分。

3. Emergent Mind, *SentencePiece Unigram Model*  
   重点：适合理解 Unigram 的概率建模、EM 重估和剪枝逻辑，尤其能支撑“从大词表删掉低贡献项”的工程视角。

4. Skool, *Tokenization for LLMs: BPE, Unigram, SentencePiece, WordPiece*  
   重点：对多语种、低资源语言、中文等实际场景中的风险和规避措施有更工程化的讨论。

5. Hugging Face `tokenizers` / SentencePiece 官方实现文档  
   重点：如果需要从文章进一步落到代码，可直接对应 BPE 的合并循环、WordPiece 的评分过程和 Unigram 的训练接口。
