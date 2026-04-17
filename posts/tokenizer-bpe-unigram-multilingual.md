## 核心结论

Tokenizer 是把原始文本切成模型可处理离散符号序列的规则系统。对大模型来说，词表设计不是“预处理细节”，而是直接影响训练成本、上下文利用率、跨语言覆盖率和推理长度的基础设施。

BPE、Unigram LM 与 byte fallback 可以看成三层能力：

| 机制 | 输入 | 输出 | 主要收益 |
|---|---|---|---|
| BPE 合并 | 原始字符或初始符号流 | 高频子词词表 | 快速复用高频片段，训练和编码都高效 |
| Unigram LM 剪枝 | 一个偏大的候选子词集合 | 概率最优的精简词表 | 更容易保留有语言学意义的子词边界 |
| byte fallback | 词表中未覆盖的字符 | UTF-8 字节序列 | 避免 `<unk>`，保证任意文本可编码、可逆 |

核心判断可以直接记住：

1. BPE 适合“高频共现片段很多”的场景，尤其是网页、多语言混合文本、代码和标点混合文本。
2. Unigram LM 适合“一个字符串可能有多种合理切分”的场景，尤其是形态变化复杂的语言。
3. byte fallback 不是主力分词策略，而是安全网。它的价值是保证任何字符都能编码，而不是让所有稀有文本都走字节级表示。
4. 多语言词表设计的关键不是“覆盖越多越好”，而是“在固定词表容量下，降低主要语言的 fertility”。fertility 可以直白理解为“同样一句话要被切成多少个 token”；越高，训练和推理越贵。

一个最小直觉例子是：BPE 像“不断把经常连在一起出现的字符焊接起来”；Unigram LM 像“先准备一大堆候选词，再删掉对整体表达能力影响最小的那些”；byte fallback 像“词表没收录时，退回到最底层字节编码，至少不会报错或丢信息”。

---

## 问题定义与边界

问题定义很明确：给定一份训练语料，要构造一个固定大小的词表，使模型在尽量少的 token 数下表示尽量多的文本，并且在遇到未登录字符时仍然能可逆编码。

这里有三个约束不能同时无限满足：

1. 词表容量有限。Embedding 和输出层通常按词表大小线性增长，词表越大，参数和显存开销越大。
2. 语言分布不均衡。英文有天然空格边界，中文没有；日文有汉字、假名混写；emoji、控制字符、数学符号又是另一类分布。
3. 文本必须可逆。也就是编码后还能无损还原原文，否则训练数据和推理输入会失真。

因此词表设计不是“找最好算法”，而是在下面几个目标之间取平衡：

$$
\text{目标} \approx \min \ \mathbb{E}[\text{tokens per text}]
\quad \text{s.t.} \quad
|V| \le K,\ \text{decode(encode(x))}=x
$$

其中 $|V|$ 是词表大小，$K$ 是预算上限。

多语言场景下，常见指标是平均 token 数或 fertility。可以用一个简单定义帮助理解：

$$
\text{fertility}(x)=\frac{\#\text{tokens in }x}{\#\text{characters in }x}
$$

如果一句中文 10 个汉字被切成 20 个 token，那么 fertility 是 2；如果被切成 6 个 token，则是 0.6。对工程来说，主要语言的 fertility 越低越好，因为这直接决定上下文窗口里能装多少真实内容。

边界也要说清楚：

- Tokenizer 不能替代模型理解能力。它只能决定“怎么切”，不能决定“模型懂不懂”。
- 词表变大不一定更好。多加很多冷门 token，可能几乎不用，却要持续占 embedding 参数。
- byte fallback 能解决 OOV，但不能解决长度膨胀。OOV 是“词表外”，长度膨胀是“虽然能编码，但编码太贵”。

一个新手最容易忽略的边界是：`<unk>` 少，不等于 tokenizer 好。若大量中文都退回成字节，虽然 technically 没有 `<unk>`，但 token 数会显著上升，模型训练成本和推理延迟都会变差。

---

## 核心机制与推导

### 1. BPE：按频率合并相邻符号

BPE 的原始思想很直接：从字符级开始，每一轮统计所有相邻符号对的出现频次，选出最常见的一对合并成新 token，然后在语料中替换，重复直到达到目标词表大小。

形式化写法是：

$$
(x^*, y^*) = \arg\max_{x,y} f_t(xy)
$$

其中 $f_t(xy)$ 表示第 $t$ 轮语料中相邻对 $xy$ 的频次。找到最频繁的一对后，构造新符号：

$$
z = x^* \circ y^*
$$

把 $z$ 加入词表，并把语料中的 $x^*y^*$ 替换成 $z$。

玩具例子可以用 `Pen Penapple Apple Pen`。先把它看作字符流：

- `P e n`
- `P e n a p p l e`
- `A p p l e`
- `P e n`

假设最高频相邻对先后是：

1. `P e -> Pe`
2. `Pe n -> Pen`
3. `Pen a p p l e -> Penapple` 这一步实际会经过更多细粒度合并，这里只是简化展示

得到的结果是：高频片段 `Pen` 和 `Penapple` 被收进词表。这样测试时即使出现训练中没见过的组合，比如 `PenapplePen`，也可以拆成已有 token 序列，而不是退回完全字符级。

BPE 的优点是快、稳定、容易实现。缺点是它只关心“频率最高的相邻对”，不直接优化整句似然，所以有时会学到对统计高频有利、但对语言边界不自然的切分。

### 2. Unigram LM：先放大候选，再用似然剪枝

Unigram LM 的直觉与 BPE 相反。它不是一步步合并，而是先准备一个偏大的候选词表，然后假设每个子词 $u$ 有一个概率 $\theta_u$，一个词或句子的切分概率由子词概率连乘得到。训练目标是最大化整份语料的似然：

$$
\mathcal{L}(\theta)=\sum_{w\in D}\log\sum_{s\in\mathrm{Seg}(w;V)}\prod_{u\in s}\theta_u
$$

这里：

- $D$ 是语料集
- $V$ 是当前候选词表
- $\mathrm{Seg}(w;V)$ 表示词 $w$ 的所有合法切分
- $\theta_u$ 是 token $u$ 的概率

白话解释是：同一个字符串可能有多种切法，Unigram LM 不急着定死，而是计算“所有可能切法的总概率”，然后保留对整体语料解释能力最强的 token。

训练时常见流程是：

1. 初始化一个比目标大很多的候选词表。
2. 用 EM 或近似方法估计每个 token 的概率。
3. 评估删掉某个 token 后，整体似然下降多少。
4. 删掉影响最小的一批 token，继续迭代，直到达到目标大小。

这就是“概率剪枝”。与 BPE 相比，Unigram LM 更容易保留语素边界。语素可以直白理解为“最小但有意义的词形单位”。对黏着语、形态变化复杂语言，Unigram 常比纯频率合并更稳。

### 3. byte fallback：最后一层可逆保险

SentencePiece 常见实践里会启用 byte fallback。它的含义是：如果某个字符或片段无法被现有词表覆盖，就退回到 UTF-8 字节序列进行编码。

例如一个罕见字符若 UTF-8 编码是三个字节，那么 tokenizer 至少可以输出这三个字节对应的 token，而不是输出 `<unk>`。

这层机制的工程意义非常大：

- 保证任意输入都可编码
- 保证编码后可逆解码
- 允许词表把容量优先留给高频语言片段，而不是硬塞满所有稀有字符

但要注意，byte fallback 是兜底，不是常规路径。如果一个语种经常触发 fallback，说明主词表对它覆盖不足。

### 4. 多语言词表为什么难

多语言词表的难点在于“共享”与“专项”之间的冲突。

- 共享 token 的好处：不同语言可能共享标点、数字、URL 模式、代码片段，提升复用率。
- 专项 token 的好处：对中文、阿拉伯语、日文等高频语言，专项 token 能显著降低 fertility。

因此常见做法不是纯共享，也不是完全分语言，而是：

1. 保留全局共享高频片段
2. 为主要语种分配足够多的专属高频子词
3. 用 byte fallback 覆盖尾部稀有字符

真实工程里，Qwen 的大词表路线就是典型例子：给多语言和控制 token 更充裕的容量，尽量让中英文都保持较低 fertility。相对地，如果词表主要偏英文，中文就更容易被切碎，甚至频繁退回字节级，这会直接拉长上下文。

---

## 代码实现

下面先用一个最小可运行 Python 实现演示 BPE 的核心训练逻辑，再给出 SentencePiece 的工程配置。

```python
from collections import Counter

def get_stats(tokens):
    pairs = Counter()
    for word in tokens:
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += 1
    return pairs

def merge_pair(tokens, pair):
    merged = []
    a, b = pair
    new_token = a + b
    for word in tokens:
        i = 0
        out = []
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                out.append(new_token)
                i += 2
            else:
                out.append(word[i])
                i += 1
        merged.append(out)
    return merged

# 玩具语料：Pen Penapple Apple Pen
corpus = [
    list("Pen"),
    list("Penapple"),
    list("Apple"),
    list("Pen"),
]

pairs = get_stats(corpus)
best = max(pairs, key=pairs.get)
assert best == ("P", "e")  # 第一轮最高频对通常是 P e

corpus = merge_pair(corpus, best)
pairs = get_stats(corpus)
best2 = max(pairs, key=pairs.get)
assert best2 == ("Pe", "n")  # 第二轮把 Pe 和 n 合并成 Pen

corpus = merge_pair(corpus, best2)

# 编码结果里至少应该出现 Pen 这个更长的子词
flat = [token for word in corpus for token in word]
assert "Pen" in flat

print(corpus)
```

这段代码没有实现完整的工业级 BPE，只演示两件关键事实：

1. BPE 的训练本质是“统计相邻对频率 + 替换”。
2. 一旦高频片段进入词表，后续同类文本就会更短。

真实工程不会手写全套 tokenizer 训练，通常直接使用 SentencePiece：

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="tok",
    vocab_size=40000,
    model_type="bpe",          # 可改为 "unigram"
    character_coverage=0.9995, # 对 CJK 常用较高覆盖率
    byte_fallback=True,        # 遇到罕见字符时退回到 UTF-8 bytes
    hard_vocab_limit=True
)

sp = spm.SentencePieceProcessor(model_file="tok.model")

text = "中文English混合🙂"
ids = sp.encode(text, out_type=int)
pieces = sp.encode(text, out_type=str)
decoded = sp.decode(ids)

assert decoded == text
print(ids)
print(pieces)
```

如果切换到 `model_type="unigram"`，其余流程基本一致，但训练内部不再是“频率最高相邻对合并”，而是“候选子词概率建模 + 剪枝”。

一个真实工程例子是训练中文、英文、代码混合语料的 tokenizer：

- 如果只用偏英文的小词表，`HTTPResponse`、`机器学习`、emoji、数学符号会被切得很碎。
- 如果用 4 万到 15 万量级词表，并开启 byte fallback，中文常见词和英文常见词干可被直接编码，极少见字符才回退到字节。
- 这样可以在不丢失可逆性的前提下，明显降低平均 token 数。

---

## 工程权衡与常见坑

词表设计的主要权衡是“参数开销”对“序列长度开销”。

一个简化判断表如下：

| 方案 | 词表参数开销 | 平均序列长度 | OOV 风险 | 多语言适配 |
|---|---|---|---|---|
| 小词表，无 fallback | 低 | 高 | 高 | 差 |
| 小词表，有 fallback | 低 | 更高但可编码 | 低 | 一般 |
| 大词表，有 fallback | 高 | 低 | 很低 | 好 |

最常见的坑有五类。

第一类坑是把 byte fallback 当成功能完备的多语言方案。它只能保证“能编码”，不能保证“编码高效”。如果中文经常被拆成多个字节 token，那么模型上下文会被迅速吃掉。

第二类坑是只看英文效果定词表大小。英文有空格、词干复用高、统计规律更容易压缩。中文没有天然词边界，如果词表不给足容量，中文 fertility 往往明显高于英文。

第三类坑是把所有领域词都硬塞进词表。比如医疗、法律、代码、数学符号都有很多高频片段，但词表容量始终有限。应该优先加入“高频且能显著降长”的 token，而不是纯粹为了覆盖看起来专业的术语。

第四类坑是错误理解 `character_coverage`。这个参数不是“把所有字符都塞进词表”的开关，而是控制训练时直接纳入字符集的覆盖比例。真正的长尾稀有字符仍应交给 fallback，而不是挤占主词表容量。

第五类坑是增量扩词后忽略兼容性。词表变了，embedding 维度和 ID 映射也变了。已有模型若直接替换 tokenizer，旧权重可能对新 token 没有有效表示，需要重新初始化新增 embedding 并做继续训练。

可以看一个对比表，帮助理解“原始词表”和“扩充词表”的差异：

| 场景 | 原始 vocab | byte fallback 使用情况 | 结果 |
|---|---|---|---|
| 偏英文词表处理中文术语 | 中文覆盖不足 | 高频触发 | token 数显著上升 |
| 多语言大词表处理中英混合 | 中文和英文均有高频子词 | 仅长尾字符触发 | 长度较稳定 |
| 垂直领域增量扩词后继续训练 | 新增领域术语 token | fallback 明显减少 | 领域文本更省 token |

把这个问题落到真实模型上，常见现象是：

- 大词表路线，像多语言和控制 token 充足的设计，更容易保持中文 1.5 到 1.8 字符对应 1 个 token 的量级。
- 偏英文词表路线，中文更容易被切成单字、碎片甚至字节，导致相同语义内容需要更多 token 才能表达。

这也是为什么“中文支持”不能只看是否能输入中文，而要看同样一段中文需要多少 token。

---

## 替代方案与适用边界

除了 BPE 和 Unigram LM，工程上还会遇到 WordPiece 与纯字节级 tokenizer。

| Tokenizer | 合并/建模策略 | 训练成本 | 多语适配 | `<unk>` 风险 |
|---|---|---|---|---|
| BPE | 最高频相邻对迭代合并 | 低到中 | 好 | 依赖 fallback |
| Unigram LM | 候选子词概率建模与剪枝 | 中到高 | 很好 | 依赖 fallback |
| WordPiece | 基于得分选择合并 | 中 | 好 | 通常存在 |
| 纯字节级 | 直接按字节编码 | 低 | 极强 | 几乎无 |

它们的适用边界可以直接概括：

BPE 适合大规模预训练的通用场景。它实现成熟、训练稳定、编码快，尤其适合网页文本、代码、聊天语料这类高频重复片段多的语料。

Unigram LM 适合对切分边界更敏感的任务。比如小语种、形态复杂语言、语素保留很重要的应用，它常能给出更自然的子词切分。

WordPiece 常见于 BERT 系列。它与 BPE 相似，但评分方式不同，不是单纯按相邻对频次做 greedy 合并。它在判别式预训练时代很常见，但在超大规模生成式模型中，BPE 和 SentencePiece 生态更普遍。

纯字节级方案的优点是极简和稳健。任何输入都能表示，不需要关心字符集覆盖。但代价也明显：对于中文、emoji、稀有字符，长度通常会更大。它更像把稳定性放在第一位。

如果只给一个工程建议，可以这样选：

- 做通用多语言大模型，优先考虑 `SentencePiece BPE + byte fallback`。
- 做形态复杂语言或想更精细控制切分边界，优先考虑 `Unigram LM + byte fallback`。
- 做极端稳健、极端简单、对 token 数不那么敏感的系统，可以考虑纯字节级。
- 如果现有模型词表已经固定，新增语言支持通常不是“换算法”，而是“扩词表 + 继续训练”。

最终原则不是追求某个 tokenizer 名字，而是看它在目标语料上的三个结果：

1. 平均 token 长度是否足够低。
2. 是否能稳定覆盖长尾字符。
3. 是否与模型部署成本匹配。

---

## 参考资料

- SentencePiece、BPE、Unigram LM 与 byte fallback 的机制综述：<https://www.emergentmind.com/topics/sentencepiece-bpe-tokenizer>
- BPE 算法说明：<https://www.emergentmind.com/topics/byte-pair-encoding-bpe-algorithm>
- Hugging Face 对 tokenizer、WordPiece、SentencePiece/Unigram 的概览：<https://huggingface.co/docs/transformers/en/tokenizer_summary>
- fast.ai 关于 SentencePiece 与 byte fallback 的说明：<https://www.fast.ai/posts/2025-10-16-karpathy-tokenizers>
- Qwen 文档中关于大词表与中文 token 密度的说明：<https://qwen.readthedocs.io/en/latest/getting_started/concepts.html>
- 关于 LLaMA 词表扩展与中文 token 膨胀问题的实践讨论：<https://pmc.ncbi.nlm.nih.gov/articles/PMC12910058/>
- 一个便于理解的 BPE 示例讨论：<https://stackoverflow.com/questions/50583254/explain-bpe-byte-pair-encoding-with-examples>
