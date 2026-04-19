## 核心结论

子词分词算法把文本切成介于“词”和“字符”之间的 token。token 是模型实际处理的最小文本单位，可以是一个词、一个字母、一个汉字，也可以是词的一部分。

它要解决的核心问题不是“找到真正的词义边界”，而是在有限词表下覆盖开放词表。开放词表指现实文本中总会出现训练时没见过的新词、拼写变体、专有名词、数字组合和符号组合。有限词表指模型只能保存有限数量的 token，例如 32k、50k 或 100k 个。

词级分词容易遇到 `OOV`。`OOV` 是 out-of-vocabulary，意思是输入片段不在词表里。字符级分词几乎没有 `OOV`，但序列会变长，模型计算成本上升。子词分词在两者之间取平衡：常见片段用较大的 token 表示，罕见词拆成较小片段表示。

玩具例子：`unhappiness` 可以切成 `un + happy + ness`，也可以切成 `un + happi + ness` 或更细的字符片段。即使模型没有见过完整单词 `unhappiness`，只要词表中有这些常见片段，它仍然能编码这个词。

| 切分层级 | 示例：`I love tokenization` | OOV 风险 | 序列长度 | 词表大小 |
|---|---|---:|---:|---:|
| word-level | `I / love / tokenization` | 高 | 短 | 大 |
| character-level | `I / l / o / v / e / ...` | 低 | 长 | 小 |
| subword-level | `I / love / token / ization` | 低 | 中等 | 中等 |

BPE、WordPiece、Unigram 都在解决同一个问题，但优化目标不同。BPE 迭代合并最高频字符对；WordPiece 倾向选择能提升语料似然的合并；Unigram 从一个较大的候选子词集合开始，用概率模型反向筛掉不重要的子词。SentencePiece 不是另一种独立算法，而是训练和编码框架，常把 BPE 或 Unigram 做成可直接处理原始文本的工程工具。

---

## 问题定义与边界

子词分词的输入是原始文本或已经预分词的文本，输出是 token 序列或 token id 序列。token id 是词表中每个 token 对应的整数编号，模型最终处理的是这些整数。

例如同一句话 `I love tokenization` 可以有三种切法：

| 方法 | 输出 |
|---|---|
| 词级 | `["I", "love", "tokenization"]` |
| 字符级 | `["I", " ", "l", "o", "v", "e", " ", "t", ...]` |
| 子词级 | `["I", "love", "token", "ization"]` |

子词分词的约束主要有三个。

第一，词表有限。模型不能把所有可能出现的字符串都记进词表。第二，语料开放。线上输入会持续出现训练语料中没有的新内容，例如新产品名、用户名、日志字段、版本号。第三，线上离线必须一致。训练时怎么归一化、怎么处理空格、怎么处理特殊符号，推理时也必须完全一致。

本文只讨论“分词与编码”，不讨论 Transformer、RNN 或其他下游模型结构。子词分词也不负责语义理解。它可以把 `unhappiness` 切成更容易覆盖的片段，但不会自动理解“否定 + 快乐 + 名词化”的完整语义关系。语义由后续模型根据上下文学习。

| 边界项 | 说明 | 典型误区 |
|---|---|---|
| 适用输入 | 原始文本、预分词文本、多语言文本、日志文本 | 以为只能处理英文空格分词文本 |
| 输出对象 | token、token id、可反解码文本 | 把 token 当作自然语言中的“词” |
| 不解决的问题 | 语义理解、实体消歧、句法分析、事实判断 | 以为切得像词就等于理解了词 |
| 工程约束 | 归一化、空格标记、特殊符号、词表版本 | 训练一个 tokenizer，线上用另一个配置 |
| 评价指标 | OOV、平均 token 数、词表覆盖率、下游效果 | 只看切分结果是否“像人类分词” |

---

## 核心机制与推导

BPE 是 Byte Pair Encoding，直译是字节对编码。在 NLP 中，它通常从字符级表示开始，每轮统计相邻符号对的频次，然后合并最高频的那一对。它的核心规则可以写成：

$$
(a,b)^* = \arg\max_{(a,b)} count_C(a,b)
$$

其中 $C$ 是训练语料，$count_C(a,b)$ 表示相邻符号对 $(a,b)$ 在语料中出现的次数。BPE 的训练过程不是理解语义，而是做高频局部合并。

玩具例子：训练语料只有 `low low lower`，并在词尾加 `</w>` 表示单词结束。

初始状态：

```text
l o w </w>
l o w </w>
l o w e r </w>
```

相邻对计数为：

| 相邻对 | 频次 |
|---|---:|
| `l o` | 3 |
| `o w` | 3 |
| `w </w>` | 2 |
| `w e` | 1 |
| `e r` | 1 |
| `r </w>` | 1 |

第一轮可以合并最高频之一，例如 `l o -> lo`：

```text
lo w </w>
lo w </w>
lo w e r </w>
```

第二轮再合并 `lo w -> low`：

```text
low </w>
low </w>
low e r </w>
```

这说明 BPE 会把常一起出现的片段粘成更大的 token。若继续训练，它可能把 `low </w>`、`e r`、`er </w>` 等片段继续合并。

WordPiece 与 BPE 相似，也通过逐步增加子词来构造词表，但它的训练准则通常概括为选择能带来最大语言模型收益的候选合并：

$$
(a,b)^* = \arg\max_{(a,b)}
\left[
\log P(C \mid V \cup \{ab\}) - \log P(C \mid V)
\right]
$$

这里 $V$ 是当前词表，$ab$ 是把相邻片段 $a$ 和 $b$ 合并后的新 token。直观说，WordPiece 不只看“出现次数最高”，还看“加入这个新 token 后，整个语料用当前词表解释得是否更好”。在实际编码时，WordPiece 常使用最长匹配，即从当前位置开始尽量找词表里最长的可用片段。

Unigram 的方向相反。它先准备一个较大的候选子词集合，再用概率模型判断哪些子词重要，最后剪掉贡献较小的项。Unigram 把一个句子的切分当作隐变量。隐变量是没有直接观测到、但模型需要推断的变量。对句子 $x$，可能存在多种切分 $s=(t_1,\dots,t_m)$。训练目标是最大化所有可能切分的边际似然：

$$
L(\theta)=\sum_{x \in C}\log \sum_{s \in S(x)}\prod_i p_\theta(t_i)
$$

其中 $S(x)$ 表示句子 $x$ 的所有可行切分，$p_\theta(t_i)$ 是 token $t_i$ 的概率。解码时通常选择概率最高的切分：

$$
s^*=\arg\max_{s \in S(x)}\sum_i \log p_\theta(t_i)
$$

Unigram 的优势是可以自然支持多候选切分。多候选切分指同一句话不只固定切成一种结果，而是允许训练时采样不同切法，从而提升模型对噪声和变体的鲁棒性。

SentencePiece 是一个工程框架。它可以直接读取原始句子，不强制依赖外部分词器，并把空格显式编码为特殊符号，例如 `▁`。这对中文、日文、韩文，以及中英混写文本尤其重要，因为这些文本不一定有稳定的空格边界。

```text
训练语料 -> 词表学习 -> 编码 -> token ids -> 模型
                         |
                         v
                    解码/反解码
```

真实工程例子：多语种搜索系统会遇到 `iPhone15价格`, `GPU 4090 driver failed`, `東京 hotel booking` 这类混合输入。如果先用某种语言的词典做预分词，数字、英文、汉字、符号可能被切坏。SentencePiece 可以直接从原始文本学习子词，并保留空格信息，降低前置分词器带来的误差。

---

## 代码实现

工程实现可以拆成三步：训练词表、编码文本、解码 token。不同算法共享这三个接口，但训练逻辑不同。下面代码实现一个最小 BPE，用来说明机制，不是生产级 tokenizer。

```python
from collections import Counter

END = "</w>"
UNK = "<unk>"

def word_to_symbols(word):
    return tuple(list(word) + [END])

def get_pair_counts(corpus):
    counts = Counter()
    for symbols, freq in corpus.items():
        for a, b in zip(symbols, symbols[1:]):
            counts[(a, b)] += freq
    return counts

def merge_pair(corpus, pair):
    merged = {}
    a, b = pair
    new_symbol = a + b
    for symbols, freq in corpus.items():
        result = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                result.append(new_symbol)
                i += 2
            else:
                result.append(symbols[i])
                i += 1
        merged[tuple(result)] = freq
    return merged

def train_bpe(words, num_merges):
    corpus = Counter(word_to_symbols(w) for w in words)
    merges = []
    vocab = {UNK, END}
    for symbols in corpus:
        vocab.update(symbols)

    for _ in range(num_merges):
        pair_counts = get_pair_counts(corpus)
        if not pair_counts:
            break
        best_pair, _ = pair_counts.most_common(1)[0]
        merges.append(best_pair)
        corpus = merge_pair(corpus, best_pair)
        vocab.add(best_pair[0] + best_pair[1])

    token_to_id = {tok: i for i, tok in enumerate(sorted(vocab))}
    return merges, token_to_id

def encode_word(word, merges, token_to_id):
    symbols = list(word_to_symbols(word))
    for pair in merges:
        a, b = pair
        result = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                result.append(a + b)
                i += 2
            else:
                result.append(symbols[i])
                i += 1
        symbols = result
    return [token_to_id.get(s, token_to_id[UNK]) for s in symbols]

def decode_ids(ids, id_to_token):
    pieces = [id_to_token[i] for i in ids]
    text = "".join(piece for piece in pieces if piece != END)
    return text.replace(END, "")

merges, token_to_id = train_bpe(["low", "low", "lower"], num_merges=2)
id_to_token = {i: t for t, i in token_to_id.items()}

ids = encode_word("lower", merges, token_to_id)
decoded = decode_ids(ids, id_to_token)

assert decoded == "lower"
assert ("l", "o") in merges
assert ("lo", "w") in merges
print(merges, ids, decoded)
```

最小伪代码流程如下：

```text
train_tokenizer(corpus, config):
    normalize text
    handle whitespace
    learn vocabulary
    save vocab file
    save merge/probability model
    save special tokens

encode(sentence):
    normalize with the same rule
    split or directly process raw text
    map pieces to token ids
    add special tokens if needed

decode(ids):
    map ids back to pieces
    remove special tokens if needed
    restore whitespace markers
```

词表保存通常至少包含四类内容：普通 token、特殊符号、算法参数、归一化规则。特殊符号是具有固定用途的 token，例如 `<pad>` 用于补齐，`<unk>` 用于未知片段，`<bos>` 表示句子开始，`<eos>` 表示句子结束。生产环境不能只保存 `vocab.txt`，还要保存 tokenizer 的完整配置，否则同一个句子在训练和推理时可能得到不同 token id。

---

## 工程权衡与常见坑

词表大小是最直接的权衡。词表太小，很多词会被拆成字符或很短片段，序列变长，训练和推理成本上升。词表太大，稀有词会被硬记住，模型泛化能力可能变差，嵌入矩阵也会变大。嵌入矩阵是 token id 到向量的查表参数，词表越大，这部分参数越多。

可以用一个简单指标观察切分效果：

$$
\text{平均压缩率}=\frac{\text{原始字符数}}{\text{token 数}}
$$

这个值过低，说明切得太碎；过高，可能说明词表记住了太多长片段。它不是唯一指标，但能快速暴露词表规模问题。

真实工程例子：日志文本 `GPU 4090 在2025年很常见` 同时包含英文、数字、空格、中文和年份。如果先用只适合英文的空格分词，可能得到 `GPU`、`4090`、`在2025年很常见`。如果再把中文部分交给另一个分词器，年份和中文边界又可能被切出不稳定结果。更稳的方式通常是让 SentencePiece 直接处理原始文本，并统一学习数字、英文、中文和空格模式。

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| 训练/推理规则不一致 | 同一句话离线和线上 token id 不同 | 固定同一份 tokenizer 配置和词表版本 |
| 词表过小 | token 过碎，序列变长，计算成本上升 | 调大 `vocab_size`，观察平均 token 数 |
| 词表过大 | 稀有词被硬记住，参数增多，泛化变差 | 根据语料规模控制词表，避免盲目增大 |
| 归一化不一致 | 大小写、全半角、Unicode 形式导致漂移 | 明确 normalization 规则并写入配置 |
| WordPiece/BPE 工具链混用 | 训练产物与解码规则不匹配 | 训练、编码、解码使用同一工具链 |
| 中英混合或无空格语言处理错误 | 预分词引入额外边界错误 | 优先考虑 SentencePiece 直接处理原始文本 |
| 特殊符号顺序变化 | 模型加载后 token id 错位 | 固定 `<pad>`、`<unk>` 等 id，不随意重排 |

另一个常见问题是把“看起来合理的切分”当作唯一目标。对模型来说，分词结果不一定要符合人类词典边界。`tokenization` 被切成 `token + ization` 或 `token + iz + ation`，未必有绝对对错。真正要看的是 OOV、序列长度、训练稳定性、下游任务指标和线上一致性。

---

## 替代方案与适用边界

没有一种子词算法在所有场景中都是最优。选择时应先看文本形态、语料规模、推理速度、是否多语言、是否需要随机切分增强。

| 算法/工具 | 适用场景 | 优点 | 限制 |
|---|---|---|---|
| BPE | 封闭领域、文本格式稳定、需要简单快速实现 | 机制简单，训练和编码容易实现 | 偏频次驱动，不直接建模多候选概率 |
| WordPiece | 类 BERT 体系、需要最长匹配编码的场景 | 工程上成熟，编码稳定 | 训练细节常依赖具体实现，不能随意与 BPE 混用 |
| Unigram | 多语言、噪声文本、需要多候选切分 | 概率模型清晰，支持子词正则化 | 训练实现比 BPE 复杂 |
| SentencePiece | 搜索、翻译、ASR、多语混写、无空格语言 | 可直接处理原始文本，支持 BPE/Unigram | 需要严格管理模型文件和归一化配置 |
| 纯字符 | 小模型、极低资源、特殊字符覆盖优先 | 几乎无 OOV，词表很小 | 序列长，学习长片段模式更困难 |
| 词级分词 | 词表封闭、领域固定、解释性优先 | 序列短，结果直观 | OOV 高，对新词和变体不友好 |

搜索、机器翻译、ASR 和多语混写场景通常优先考虑 SentencePiece，尤其是 SentencePiece Unigram。ASR 是 automatic speech recognition，指自动语音识别，输入转写文本常有口语、噪声和多语言混合。Unigram 的多候选能力和 SentencePiece 的原始文本处理能力，在这些场景中更稳。

封闭领域可以先试 BPE。例如一个只处理固定格式设备日志的系统，字段名、错误码和单位都比较稳定，BPE 的高频合并足够有效，且实现和排查成本低。若系统已经基于 BERT 生态，使用与模型预训练一致的 WordPiece 更重要，因为 tokenizer 不一致会直接破坏预训练模型学到的 token 分布。

新手选型可以按一个简单顺序判断：如果文本复杂、多语言、没有稳定空格，先看 SentencePiece；如果需要实现简单和速度，先看 BPE；如果必须复用 BERT 类模型，跟随原模型的 WordPiece；如果需要训练时随机切分增强，再重点看 Unigram 和子词正则化。

---

## 参考资料

1. [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/)
2. [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://aclanthology.org/D18-2012/)
3. [Google SentencePiece GitHub README](https://github.com/google/sentencepiece)
4. [Subword Regularization: Improving Neural Machine Translation Models with Multiple Subword Candidates](https://aclanthology.org/P18-1007/)
5. [A Fast WordPiece Tokenization System](https://research.google/blog/a-fast-wordpiece-tokenization-system/)
6. [Japanese and Korean Voice Search](https://research.google/pubs/japanese-and-korean-voice-search/)
