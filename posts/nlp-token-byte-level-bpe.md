## 核心结论

字节级 BPE（Byte-level BPE）是一种分词方法。分词方法就是把原始文本切成模型能处理的最小输入单元。它的核心做法不是先假设“字”或“词”是什么，而是直接从 **256 个 UTF-8 字节值** 出发训练词表。

这件事带来两个直接结果：

1. 它几乎彻底消除了 OOV（out-of-vocabulary，词表外词）。词表外词就是“输入里出现了一个符号，但词表里没有对应项”。因为任何 Unicode 文本都能编码成 UTF-8 字节流，而 256 个字节值天然可表示所有 UTF-8 编码结果，所以任意字符最终都能被拆成已知字节。
2. 它的压缩效率并不均匀。英语这类以 ASCII 为主的文本通常压缩得更好；中文、泰语、印地语等多字节脚本往往需要更多 token，同样长度的内容会更贵、更慢，也更占上下文窗口。

GPT-2 是较早大规模采用字节级 BPE 的模型之一，词表大小为 50,257。后续 GPT-4 体系常用的 `cl100k_base` 把词表扩到约 100K，本质目标仍然是：在“通用可表示”和“序列压缩率”之间做更好的平衡。

一个新手版玩具例子是“你好”。它在 UTF-8 下不是两个简单字符，而是一串字节。字节级 BPE 先把它看成字节序列，再把训练中高频出现的相邻字节对逐步合并成更长 token。模型看到的不是“神秘的中文字符”，而是“可还原成中文字符的字节模式”。

下表可以先看出“可表示”和“压缩率”是两件不同的事：

| 编码方案 | 初始单位 | 是否有 OOV | 英文压缩率 | 中文压缩率 | 典型问题 |
|---|---|---:|---:|---:|---|
| 纯词级/词典式 | 词或子词 | 有 | 中 | 中 | 生僻词、混合脚本容易失效 |
| 纯字节编码 | 1 个字节 | 无 | 很差 | 差 | token 数爆炸 |
| 字节级 BPE | 256 字节 + 合并词元 | 基本无 | 好 | 一般 | 多字节语言成本偏高 |

如果只按字节输入，不做任何合并，序列会显著变长。研究摘要中的对比是：约 1,000 字英语文本，GPT-2 风格 BPE 大约 750 token，而完全按字节处理可能到 5,000+ token。这说明分词器不是“前处理细节”，它直接决定推理成本和上下文利用率。

---

## 问题定义与边界

要理解字节级 BPE，先要把问题说清楚：**模型并不直接理解字符，它只接收离散 token ID 序列。** 因此，分词器的任务不是“把句子切开”这么简单，而是要同时满足三件事：

1. 任意输入都能表示。
2. 序列尽量短。
3. 词表规模不能无限膨胀。

传统词表方案的根本问题在于覆盖率。只要词表是有限的，而世界上的文本不断出现新拼写、新符号、新语言混排、新表情和罕见字符，就会遇到 OOV。字节级 BPE 的边界条件很明确：**它解决的是可表示性问题，不保证所有语言都压缩得同样高效。**

评估一个 tokenizer，不能只看“能不能切”，还要看“切出来有多长”。一个常见指标是归一化序列长度：

$$
c_{\lambda/\beta}=\frac{\sum_i \text{length}(T_\lambda(D_i))}{\sum_i \text{length}(T_\beta(D_i))}
$$

这里：

- $D_i$ 是第 $i$ 条文本样本。
- $T_\lambda$ 是 tokenizer $\lambda$ 对样本的编码结果。
- $\text{length}(\cdot)$ 是 token 序列长度。
- $c_{\lambda/\beta}<1$ 表示 tokenizer $\lambda$ 比基准 tokenizer $\beta$ 更省 token。

这个指标的白话解释是：**同一批文本，谁切出来更短，谁就更“省上下文”。**

一个简化理解例子：

- 同一句英文，旧 GPT-2 tokenizer 可能切成 12 个 token。
- `cl100k_base` 可能切成 10 个 token。

那么只看这一句，归一化长度就是：

$$
c_{\text{cl100k}/\text{gpt2}}=\frac{10}{12}\approx 0.83
$$

这意味着 `cl100k_base` 在该样本上压缩得更好。但这个指标必须放到数据集层面看，因为某个 tokenizer 可能对英文更优，对中文改善有限。

问题的边界也要讲清：

| 问题 | 字节级 BPE 是否解决 | 说明 |
|---|---:|---|
| 任意 Unicode 输入可表示 | 是 | 都能退化成 UTF-8 字节 |
| 生僻符号 OOV | 是 | 最差也能拆到字节 |
| 所有语言 token 数都接近 | 否 | 多字节语言常更吃亏 |
| 语义切分天然最优 | 否 | BPE 只优化频率，不理解语言学语义 |
| 词表无限扩展问题 | 部分解决 | 通过固定词表大小折中 |

真实工程里，分词器不是独立存在的。它与 API 计费、KV Cache 占用、最大上下文长度、吞吐量都绑定。也就是说，分词器压缩率差，不只是“前处理不优雅”，而是**系统成本真的会上升**。

---

## 核心机制与推导

字节级 BPE 的机制可以拆成三步：**字节化、切片、合并。**

### 1. 字节化

先把原始文本编码成 UTF-8 字节流。UTF-8 是一种字符编码方式，意思是“把 Unicode 字符编码成 1 到 4 个字节”。

例如字符串 `Café`：

- `C` -> `0x43`
- `a` -> `0x61`
- `f` -> `0x66`
- `é` -> `0xC3 0xA9`

所以 `Café` 的字节序列是：

$$
[43, 61, 66, C3, A9]
$$

白话讲，模型训练时并不需要先知道 `é` 是什么字符；它只需要知道这两个字节经常一起出现。

### 2. 预切片

GPT-2 一类 tokenizer 通常不会直接对整段原始字节流做全局合并，而是先按正则规则切片，比如：

- 空格 + 单词
- 数字串
- 标点
- 缩写片段

这样做的原因不是“更懂语义”，而是减少一些明显不合理的跨边界合并。比如把英文单词结尾和下一个标点直接合并，往往不稳定。

### 3. BPE 频率合并

BPE（Byte Pair Encoding）的核心是：统计训练语料里最常见的相邻单元对，把它们合并成一个新单元，然后重复这个过程。

初始词表是 256 个单字节值。之后每轮：

1. 统计所有相邻 pair 的频次。
2. 找到最高频 pair。
3. 把这个 pair 合并成一个新 token。
4. 更新序列表示。
5. 重复，直到达到目标词表大小。

一个玩具例子，用“你好你好”理解更直观。假设它的 UTF-8 字节片段中，某两个相邻字节组合反复出现，那么训练时这对字节就更可能被合并。再往后，已经合并出的片段还可以继续与邻居合并，最终形成更长 token。

这个过程为什么能消除 OOV？原因很简单：

- 初始集合已经覆盖了所有可能字节。
- 所有更长 token 都只是字节序列的组合。
- 即使一个字符或词从未在训练中出现，也总能退回到若干字节 token。

所以“未知字符无法编码”的问题被消除了。代价是：**能编码，不等于编码得短。**

下面这段伪代码是核心循环：

```python
def train_bpe(byte_sequences, vocab_size):
    vocab = {bytes([i]) for i in range(256)}

    while len(vocab) < vocab_size:
        pair_count = count_adjacent_pairs(byte_sequences)
        best_pair = argmax(pair_count)
        new_token = merge(best_pair)
        vocab.add(new_token)
        byte_sequences = replace_pair(byte_sequences, best_pair, new_token)

    return vocab
```

这个循环隐含了一个关键事实：BPE 优化的是“统计共现频率”，不是字符边界、词法结构或语义边界。因此它对英语这类高频重复模式压缩很好，对中文这类每个字符本身信息密度高、但 UTF-8 字节模式不一定重复到足够程度的语言，就未必占优。

---

## 代码实现

下面用一个最小可运行的 Python 例子，演示三件事：

1. 把字符串转成 UTF-8 字节；
2. 统计相邻 pair 频次；
3. 执行一次合并迭代。

```python
from collections import Counter

def utf8_bytes(text: str) -> list[bytes]:
    return [bytes([b]) for b in text.encode("utf-8")]

def pair_stats(tokens: list[bytes]) -> Counter:
    return Counter((tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1))

def merge_once(tokens: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
    merged = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            merged.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged

text = "Café Café"
tokens = utf8_bytes(text)
stats = pair_stats(tokens)
best_pair, freq = stats.most_common(1)[0]
new_tokens = merge_once(tokens, best_pair)

assert len(tokens) == len(text.encode("utf-8"))
assert freq >= 1
assert sum(len(t) for t in new_tokens) == len(text.encode("utf-8"))

print("原始字节 token:", tokens)
print("最高频 pair:", best_pair, "频次:", freq)
print("合并一次后:", new_tokens)
```

这段代码不是完整 tokenizer，但足够说明训练阶段的最小机制。

如果拿“你好你好”做玩具例子，也能看到类似现象：

```python
from collections import Counter

def utf8_units(text: str):
    return [bytes([b]) for b in text.encode("utf-8")]

def count_pairs(units):
    return Counter((units[i], units[i + 1]) for i in range(len(units) - 1))

units = utf8_units("你好你好")
pairs = count_pairs(units)

assert len(units) == len("你好你好".encode("utf-8"))
assert max(pairs.values()) >= 2

print(units)
print(pairs.most_common(5))
```

白话解释：

- 中文字符先被拆成多个字节。
- 如果某些字节相邻关系重复出现，BPE 会优先把它们合起来。
- 经过很多轮后，常见片段会变成单个 token。

真实工程里的实现比这个复杂得多，通常包含：

| 组件 | 作用 | 工程说明 |
|---|---|---|
| UTF-8 编码层 | 把任意文本转成字节 | 保证可表示性 |
| 正则切片层 | 先做粗分段 | 降低不合理跨边界合并 |
| merges 表 | 记录训练出的合并顺序 | 推理时按优先级匹配 |
| vocab/id 映射 | 把 token 映射成整数 ID | 供模型嵌入层查表 |
| decode 层 | 把 token 序列还原回文本 | 调试、生成结果展示都依赖它 |

真实工程例子是 LLM API 计费。假设你向模型发送两段信息，字符数接近，但一段是英文、一段是中文。由于中文在字节级 BPE 下往往需要更多 token，同样 8K 上下文窗口，中文可容纳的语义内容更少；同样每百万 token 定价，中文工作负载的成本更高。这不是模型参数导致的，而是 tokenizer 在输入层已经决定了。

---

## 工程权衡与常见坑

字节级 BPE 的最大工程优势是稳定。任何文本都能进模型，不会因为出现冷门字符、混合编码、表情或罕见脚本而直接报废。这对通用大模型尤其重要。

但它的代价同样明确：**多字节语言压缩效率偏低。**

研究摘要给出的典型结论是，在 `cl100k_base` 下：

- 1,000 字符英文，约 185 token；
- 1,000 字符中文，约 1,000 token。

这个数字不必机械地当成所有样本都成立，但它说明一个趋势：**中文 token/字符比通常显著高于英文。**

可以用表看得更直观：

| 语种/文本类型 | UTF-8 特征 | 字节级 BPE 表现 | 结果 |
|---|---|---|---|
| 英文 | 大量 ASCII 单字节 | 高频片段容易合并 | token 少，压缩好 |
| 中文 | 常见字符多为 3 字节 | 合并空间受限 | token 多，压缩一般 |
| 泰语/印地语 | 多字节且词形变化复杂 | 频率模式分散 | token 偏多 |
| 表情/混合脚本 | 编码复杂、长尾多 | 可表示但不一定高效 | 稳定但不省 |

常见坑主要有四类。

第一类是“把可表示误认为高效表示”。  
字节级 BPE 确实没有 OOV，但这不代表对所有语言都友好。对中文业务，输入成本和上下文占用可能明显偏高。

第二类是“错误处理 UTF-8 边界”。  
理论上，token 可以是任意字节序列，只要最终解码一致。但工程实现如果在训练、解码、正则切片或后处理环节处理不一致，就可能出现中间 token 难以调试、日志不可读、甚至 decode 异常的问题。白话讲，就是“虽然数学上能拼回去，但工程链路未必每一步都拼得对”。

第三类是“忽略预切片规则的影响”。  
很多人只关注 merges，不关注 regex。实际上，预切片规则会强烈影响可合并的局部模式。换一套正则，最终词表分布和压缩表现都可能变化。

第四类是“低估 tokenizer 对系统指标的影响”。  
推理成本近似与 token 数成正相关。若同样内容中文 token 数是英文的 2 到 3 倍，那么：

$$
\text{Cost} \propto \text{Input Tokens} + \text{Output Tokens}
$$

token 变多，意味着：

- 计费上升；
- 上下文更快耗尽；
- Prefill 更慢；
- KV Cache 更大。

所以 tokenizer 不是“训练前的小工具”，而是系统设计的一部分。

---

## 替代方案与适用边界

字节级 BPE 不是唯一方案，它只是一个在“稳定性”和“压缩率”之间很实用的折中。

先看三类常见方案：

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 纯字节编码 | 完全无 OOV，实现简单 | 序列极长 | 极端鲁棒性实验、研究基线 |
| 字节级 BPE | 无 OOV，压缩显著优于纯字节 | 多语种不均衡 | 通用大模型 |
| 语言特化子词方案 | 某些语言压缩更好 | 覆盖范围差，跨语言弱 | 单语或窄域系统 |

为什么纯字节编码通常不适合真实产品？因为它太“安全”，但不“经济”。如果 1,000 字英语文本需要 5,000+ token，那么上下文窗口几乎被编码开销吞掉了，模型大部分算力花在“读入原文”，而不是“做推理”。

为什么 `cl100k_base` 相比 GPT-2 往往更优？可以这样理解：

- 词表更大，允许吸收更多高频模式；
- 合并策略和训练数据分布更现代；
- 对英文代码混合文本通常更省 token。

但它对中文的改善通常没有对英文那么显著，因为中文的瓶颈不只是“词表不够大”，还在于 UTF-8 多字节结构与语料统计模式本身。也就是说，扩词表能改善，但不能消除语种差异。

适用边界可以概括为：

1. 如果你做的是通用聊天模型、开放域文本输入、跨语言输入，字节级 BPE 很稳妥。
2. 如果你做的是高比例中文、且上下文成本极敏感的系统，应该认真评估 tokenizer 的语言压缩率，而不是默认沿用英文主导词表。
3. 如果你做的是特定行业、固定语种、固定术语集，语言特化或领域特化 tokenizer 可能更划算。

一个简单判断标准是：  
**当“可处理任意输入”比“对某个语种极致省 token”更重要时，字节级 BPE 通常值得选；反之，就要考虑特化方案。**

---

## 参考资料

- Hugging Face, Tokenizer summary: 说明 BPE、byte-level BPE 的基本机制，以及 GPT-2 采用字节级 BPE 的背景。  
  https://huggingface.co/docs/transformers/v4.45.2/tokenizer_summary

- JY Tan, BPE tutorial: 用较直观的方式解释从字节流、切片到 merge 的过程，适合理解训练循环。  
  https://jytan.net/blog/2025/bpe/

- Emergent Mind, GPT-4's tokenizer: 提供 `cl100k_base`、归一化序列长度等评估视角，适合比较不同 tokenizer 的压缩效果。  
  https://www.emergentmind.com/topics/gpt-4-s-tokenizer

- SOTAaz, BPE vs Byte-level Tokenization: 讨论纯字节编码与 BPE 压缩率差异，也提到非拉丁文字在 token 成本上的劣势。  
  https://blog.sotaaz.com/post/bpe-vs-byte-level-tokenization

- LLM Calculator, Tokenization Performance Benchmark: 给出多语种 token 数对比，适合理解 tokenizer 对计费和上下文窗口的直接影响。  
  https://llm-calculator.com/blog/tokenization-performance-benchmark/
