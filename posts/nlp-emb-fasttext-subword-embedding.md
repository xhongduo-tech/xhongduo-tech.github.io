## 核心结论

FastText 的核心做法不是“给每个词只学一个向量”，而是“给词内部的字符片段也学向量”。这里的“子词”先按最直接的方式理解即可：**单词内部连续的一段字符**，通常是字符 n-gram。

如果词 $w$ 的子词集合记为 $G_w$，子词 $g$ 的向量记为 $z_g$，那么词向量可以写成：

$$
v_w=\sum_{g\in G_w} z_g
$$

很多实现还会把“整词本身”也当作一个特殊子词加入求和，这样既保留词内部结构，又保留整词特有信息。写成更完整的形式是：

$$
v_w=z_{\langle w\rangle}+\sum_{g\in G_w} z_g
$$

其中：

- $G_w$：词 $w$ 的所有字符 n-gram 集合
- $z_g$：子词 $g$ 的输入向量
- $z_{\langle w\rangle}$：整词自身的向量
- $v_w$：最终参与训练和推理的词表示

这带来两个直接结果。

第一，FastText 对未登录词（OOV, out-of-vocabulary）更稳。未登录词指“测试或线上推理时出现，但训练时没见过的完整词”。例如 `running`、`runner`、`runs` 都共享 `run`、`unn`、`ing` 一类片段。即使训练时没见过完整的 `runner`，模型也能用它的子词构造出一个近似合理的向量。

第二，FastText 对形态丰富语言更有优势。所谓“形态丰富”，是指一个词会因为时态、性、数、格等语法变化衍生出很多不同词形。此时不同词形之间往往共享大量词干、前缀和后缀，子词共享就会明显提高数据利用率。Bojanowski 等人在 2017 年论文中报告，FastText 在 Czech syntactic 词类比任务上的准确率约为 77.8%，普通 skip-gram 约为 52.8%。

| 方法 | 词表示方式 | Czech syntactic 准确率 |
|---|---|---:|
| Word2Vec skip-gram | 每个词一个独立向量 | 52.8 |
| FastText | 词向量由多个字符 n-gram 向量组合 | 77.8 |

但这不等于“FastText 在所有任务都更强”。它的优势来自词内部结构，因此更适合：

- 形态变化多的语言
- 拼写变体多的场景
- 新词和低频词较多的语料
- 需要静态词向量、但又不希望 OOV 直接失效的系统

代价也很明确：

- 要额外维护大量子词参数
- 模型体积通常大于 Word2Vec
- 推理时需要在线拆分子词并聚合
- 它仍然是静态词向量，不能做上下文消歧

可以先把它理解成一句话：**FastText 解决的是“词内部结构可复用”的问题，不是“词义随上下文变化”的问题。**

---

## 问题定义与边界

词嵌入的目标，是把离散词符映射到连续向量空间，使语义和语法上的相似性能被向量距离、点积或余弦相似度表达。

传统 Word2Vec 的基本假设是：**词是最小建模单位**。模型只学习“完整词”对应的向量，不进一步分析词内部结构。

这个假设在高频词上通常够用，但一遇到下面几类情况就会暴露短板：

| 问题类型 | 现象 | Word2Vec 的典型问题 |
|---|---|---|
| 低频词 | 训练中只出现几次 | 向量估计不稳定 |
| 未登录词 OOV | 推理时出现训练没见过的新词 | 根本没有向量 |
| 形态变体 | `run`、`running`、`runner` | 相互之间共享信息不足 |
| 拼写变体 | `color` / `colour`、缩写、派生词 | 泛化能力弱 |
| 噪声文本 | 搜索词、商品标题、社媒文本 | 新词、错拼、混写很多 |

FastText 的边界要讲清楚。它解决的是：

- 低频词样本不足
- OOV 没有词向量
- 词形变化之间无法共享统计信息
- 拼写变体和派生词难以泛化

它**不**解决的是：

- 同一个词在不同上下文里的多义性
- 长距离上下文依赖
- 句法级、篇章级动态语义建模

例如 `bank` 在 “river bank” 和 “central bank” 中语义不同。FastText 默认仍会给它一个静态基础向量，不会因为上下文不同而动态变化。这一点与 BERT、GPT 这类上下文化模型不同。

### FastText 里的“子词”到底是什么

FastText 默认不是按“词根词缀规则”人工切分，而是按**字符连续子串**切分。做法通常分四步：

1. 给词加边界符 `<` 和 `>`
2. 从中抽取长度为 3 到 6 的字符 n-gram
3. 把这些 n-gram 映射到参数表
4. 把对应向量相加得到词向量

例如 `apple` 会先写成 `<apple>`。如果只看 3-gram 和 4-gram，则有：

| n | 抽取结果 |
|---|---|
| 3 | `<ap`、`app`、`ppl`、`ple`、`le>` |
| 4 | `<app`、`appl`、`pple`、`ple>` |

边界符的作用不是装饰，而是区分“词首”和“词尾”。例如：

- `ing`：只是一个普通的连续片段
- `ing>`：明确表示“词尾是 ing”
- `<un`：明确表示“词首是 un”

这使模型能学到更接近前缀、后缀的信息，而不是把所有相同字符串都当成同一种模式。

### 为什么这套边界重要

FastText 的基本判断标准不再是“这个完整词是否出现在词表里”，而是“这个词包含的字符片段是否能和已有模式共享”。因此它对新词的处理方式，与 Word2Vec 完全不同：

| 模型 | 推理遇到新词时会怎样 |
|---|---|
| Word2Vec | 新词不在词表，通常直接失败或映射到 `<unk>` |
| FastText | 仍然可以抽子词并构造向量 |
| 上下文化 Transformer | 先做子词分词，再结合上下文动态编码 |

所以，FastText 的本质不是“更大的词表”，而是**把参数共享粒度从整词降到了字符片段级别**。

---

## 核心机制与推导

FastText 可以看成对 skip-gram 的一次子词化改造。skip-gram 的核心目标是：给定中心词，预测它周围的上下文词。FastText 保留了这个目标，但不再把中心词表示成一个单独向量，而是改成“多个子词向量的和”。

### 1. 从 skip-gram 开始

普通 skip-gram 中，中心词 $w$ 和上下文词 $c$ 的打分通常写成：

$$
s(w,c)=v_w^\top u_c
$$

其中：

- $v_w$：中心词输入向量
- $u_c$：上下文词输出向量

如果用负采样（negative sampling）训练，一个正样本 $(w,c)$ 的损失常写成：

$$
\mathcal{L}(w,c)= -\log \sigma(s(w,c)) - \sum_{i=1}^{K}\log \sigma\big(-s(w,n_i)\big)
$$

其中：

- $\sigma(x)=\frac{1}{1+e^{-x}}$ 是 sigmoid
- $n_i$ 是采样得到的负样本词
- $K$ 是负样本数

直观含义是：

- 正样本的打分应该更高
- 随机负样本的打分应该更低

### 2. FastText 把词向量替换成子词组合

FastText 不直接存“词 $w$ 的唯一输入向量 $v_w$”，而是先定义它的子词集合 $G_w$，再把词表示成这些子词向量之和：

$$
v_w=\sum_{g\in G_w} z_g
$$

若把整词本身也算进去，则更常见的写法是：

$$
v_w=\sum_{g\in G_w \cup \{\langle w\rangle\}} z_g
$$

于是中心词与上下文词的打分变成：

$$
s(w,c)=\left(\sum_{g\in G_w \cup \{\langle w\rangle\}} z_g\right)^\top u_c
$$

这一步很关键，因为它决定了**参数更新会在共享子词之间传播**。例如 `running` 和 `runner` 都含有 `run` 一类片段，那么训练 `running` 时，更新不仅作用于 `running` 这个词本身，也会作用于共享的子词参数。之后再遇到 `runner`，它就能受益于这部分共享知识。

### 3. 为什么“求和”就够了

很多新手会问：为什么不是拼接、平均、加权平均，而是直接求和？

原因有三点。

第一，求和在训练目标下最直接。skip-gram 只需要一个最终词向量去和上下文向量做点积，求和实现最简单。

第二，求和能自然表达“共享贡献”。某个子词在很多词中出现得越稳定，它的参数就越可能学到可复用模式。

第三，平均与求和在很多场景里只差一个缩放常数。若后续训练会继续调整参数，这种缩放差异通常不是决定性问题。

### 4. 一个具体例子：`apple`

设只取 3-gram，并把整词本身也算进去。词 `<apple>` 的子词可以写成：

$$
G_{\text{apple}}=\{\langle ap,\ app,\ ppl,\ ple,\ le\rangle\}
$$

于是：

$$
v_{\text{apple}}=z_{\langle apple\rangle}+z_{\langle ap}+z_{app}+z_{ppl}+z_{ple}+z_{le\rangle}
$$

这个式子表达的不是“把字母拆开平均”，而是“把多个局部拼写模式合成一个词向量”。

其中整词项 $z_{\langle apple\rangle}$ 也很重要。因为如果完全只靠 n-gram，那么很多拼写相似但语义差异很大的词会被拉得过近。把整词自身加入表示，可以保留它的专有信息。

### 5. 再看共享是怎么发生的

下面用三个词说明：

| 词 | 典型共享片段 |
|---|---|
| `run` | `run`、`un>`、`<ru` |
| `running` | `run`、`unn`、`ing`、`ng>` |
| `runner` | `run`、`unn`、`ner`、`er>` |

虽然这三个词不是同一个完整词，但它们共享一部分字符片段。于是训练 `running` 时学到的信息，会通过 `run`、`unn` 这类共享子词部分流向 `runner`。这就是 FastText 泛化能力的来源。

### 6. 哈希桶为什么必要

真实语料里的子词数量非常大，不能为每个字符片段都维护一个无限扩张的字典。FastText 的常见做法是使用哈希桶（bucket）存储子词参数。

设桶数为 $B$，哈希函数为 $h(\cdot)$，则任意子词 $g$ 会映射到：

$$
h(g)\in \{0,1,\dots,B-1\}
$$

实际存储的向量是：

$$
z_{h(g)}
$$

而不是“用原始字符串作为键去查一个无限字典”。

这么做的好处是：

- 参数规模固定，可控
- 内存可提前规划
- 工程实现简单，速度稳定

代价是哈希冲突。也就是不同子词可能落到同一个桶里，被迫共享同一个参数。这会引入噪声。

### 7. 哈希冲突的直觉

假设只有 4 个桶，却有 8 个常见子词：

| 子词 | 哈希桶 |
|---|---:|
| `run` | 0 |
| `ing` | 2 |
| `er>` | 1 |
| `<un` | 3 |
| `ion` | 2 |
| `pre` | 1 |
| `able` | 0 |
| `ly>` | 3 |

此时 `run` 和 `able` 会被写到同一个桶，模型无法完全区分它们。这就是冲突噪声。桶越大，冲突通常越少；桶越小，模型越省内存，但表示容量越弱。

### 8. 推理时为什么能处理 OOV

对一个训练没见过的新词 $w^\*$，FastText 的推理流程仍然成立：

1. 给它加边界符 `<` 和 `>`
2. 抽取长度为 3 到 6 的 n-gram
3. 对每个子词做哈希映射
4. 取出桶向量并求和
5. 得到 $v_{w^\*}$

只要这些子词中的一部分在训练中出现过足够多次，新词就能得到一个不完全随机的向量。这就是 FastText 处理 OOV 的根本机制。

---

## 代码实现

下面给一个**可直接运行**的最小 FastText 风格训练脚本。它不依赖第三方库，只用 Python 标准库，展示四件事：

1. 如何抽取字符 n-gram
2. 如何把子词哈希到固定桶
3. 如何用 skip-gram + 负采样做一个极简训练
4. 为什么训练没见过的 `runner` 仍然能得到向量，并与 `run`、`running` 更接近

```python
import math
import random
import hashlib
from collections import Counter

random.seed(0)

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def add_inplace(a, b, scale=1.0):
    for i in range(len(a)):
        a[i] += scale * b[i]

def extract_ngrams(word: str, min_n: int = 3, max_n: int = 6, include_word: bool = True):
    wrapped = f"<{word}>"
    grams = []
    for n in range(min_n, max_n + 1):
        if n > len(wrapped):
            continue
        for i in range(len(wrapped) - n + 1):
            grams.append(wrapped[i:i + n])
    if include_word:
        grams.append(wrapped)
    return grams

def stable_hash(text: str, bucket_size: int) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h, 16) % bucket_size

def cosine(a, b):
    na = math.sqrt(dot(a, a))
    nb = math.sqrt(dot(b, b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)

class TinyFastText:
    def __init__(self, vocab, dim=16, bucket_size=200, min_n=3, max_n=4, lr=0.05):
        self.vocab = list(vocab)
        self.word_to_id = {w: i for i, w in enumerate(self.vocab)}
        self.dim = dim
        self.bucket_size = bucket_size
        self.min_n = min_n
        self.max_n = max_n
        self.lr = lr

        # 子词桶向量：输入侧参数
        self.subword_table = [
            [(random.random() - 0.5) / dim for _ in range(dim)]
            for _ in range(bucket_size)
        ]

        # 上下文词向量：输出侧参数
        self.output_table = {
            w: [(random.random() - 0.5) / dim for _ in range(dim)]
            for w in self.vocab
        }

    def subword_ids(self, word):
        grams = extract_ngrams(word, self.min_n, self.max_n, include_word=True)
        return [stable_hash(g, self.bucket_size) for g in grams]

    def word_vector(self, word):
        ids = self.subword_ids(word)
        vec = [0.0] * self.dim
        for idx in ids:
            add_inplace(vec, self.subword_table[idx], scale=1.0)
        return vec

    def update_pair(self, center_word, context_word, label):
        sub_ids = self.subword_ids(center_word)
        center_vec = self.word_vector(center_word)
        out_vec = self.output_table[context_word]

        score = dot(center_vec, out_vec)
        pred = sigmoid(score)
        grad = pred - label

        old_out = out_vec[:]

        # 更新上下文输出向量
        for i in range(self.dim):
            out_vec[i] -= self.lr * grad * center_vec[i]

        # 更新中心词的所有子词向量
        for idx in sub_ids:
            for i in range(self.dim):
                self.subword_table[idx][i] -= self.lr * grad * old_out[i]

    def train(self, pairs, epochs=50, negative_samples=2):
        vocab_list = self.vocab[:]
        for epoch in range(epochs):
            random.shuffle(pairs)
            for center_word, context_word in pairs:
                self.update_pair(center_word, context_word, 1)
                for _ in range(negative_samples):
                    neg = random.choice(vocab_list)
                    if neg == context_word:
                        continue
                    self.update_pair(center_word, neg, 0)

def build_pairs(sentences, window_size=1):
    pairs = []
    for sent in sentences:
        for i, center in enumerate(sent):
            left = max(0, i - window_size)
            right = min(len(sent), i + window_size + 1)
            for j in range(left, right):
                if i == j:
                    continue
                pairs.append((center, sent[j]))
    return pairs

if __name__ == "__main__":
    # 故意不把 runner 放进训练语料，后面拿它做 OOV 演示
    sentences = [
        ["run", "fast"],
        ["running", "fast"],
        ["walk", "slow"],
        ["walking", "slow"],
        ["run", "daily"],
        ["running", "daily"],
    ]

    vocab = sorted({w for sent in sentences for w in sent})
    pairs = build_pairs(sentences, window_size=1)

    model = TinyFastText(vocab, dim=24, bucket_size=500, min_n=3, max_n=4, lr=0.05)
    model.train(pairs, epochs=80, negative_samples=3)

    words = ["run", "running", "runner", "walk", "walking"]
    vecs = {w: model.word_vector(w) for w in words}

    print("Subwords of 'runner':")
    print(extract_ngrams("runner", min_n=3, max_n=4, include_word=True))
    print()

    print("Cosine similarities:")
    for a, b in [("runner", "run"), ("runner", "running"), ("runner", "walk"), ("runner", "walking")]:
        print(f"{a:8s} vs {b:8s} -> {cosine(vecs[a], vecs[b]):.4f}")
```

运行后，你通常会看到类似趋势：

- `runner` 与 `run`、`running` 的相似度较高
- `runner` 与 `walk`、`walking` 的相似度较低

这里的重点不是数值一定是多少，而是**`runner` 从未作为完整词参与训练，却仍能构造向量并与相关词更接近**。这正是 FastText 相对 Word2Vec 的核心优势。

### 这段代码和真实 FastText 的差别

上面的脚本是教学版，不是工业级实现。它省略了很多工程细节，例如：

| 维度 | 教学脚本 | 真实 fastText |
|---|---|---|
| 语料处理 | 手工分词小样本 | 大规模文本流式读取 |
| 优化 | 简单 SGD | 更完整的训练实现 |
| 负采样 | 直接随机采样 | 更高效的分布式采样 |
| 子词表 | Python 列表 | C++ 高效存储 |
| 训练目标 | 简化版 skip-gram | 支持 skip-gram / cbow / supervised |
| 压缩 | 无 | 支持量化 |

但表示构造的核心机制是一致的：**词向量由多个子词向量组合而成**。

### 一个更贴近论文的伪代码

```python
for center_word, context_word in corpus_windows:
    sub_ids = hash_all_ngrams(center_word)
    center_vec = sum(subword_table[idx] for idx in sub_ids)
    score = dot(center_vec, output_table[context_word])
    loss = negative_sampling_loss(score, negatives)
    backpropagate(loss)
```

### 为什么这对新手尤其重要

很多人第一次学词向量时，会默认以为“一个词 = 一个向量”。FastText 需要把这个直觉纠正成：

- 完整词不是唯一建模单位
- 词内部结构也可以成为参数共享单位
- 新词不一定要在训练中出现过，才可能有向量

一旦这个直觉建立起来，FastText 的很多现象就会自然许多，包括：

- 为什么 `running` 能帮助 `runner`
- 为什么 `colour` 和 `color` 可能更接近
- 为什么哈希桶太小会带来性能下降
- 为什么它能做 OOV，而 Word2Vec 做不到

---

## 工程权衡与常见坑

FastText 的工程代价主要不在“算法逻辑更难”，而在“参数规模和部署成本更高”。

Word2Vec 只需要维护完整词表向量；FastText 除了整词向量，还要维护大量子词桶向量。即使使用哈希桶，模型体积通常仍明显更大。

### 1. 体积与内存不是次要问题

可以把几类方案的工程特性先放在一起看：

| 方案 | 模型大小 | OOV 处理 | 形态泛化 | 部署特点 |
|---|---|---|---|---|
| Word2Vec | 小 | 差 | 弱 | 内存最省 |
| 原始 FastText | 大 | 强 | 强 | 线上内存压力更大 |
| 量化 FastText | 中 | 保留大部分能力 | 中到强 | 更适合端侧或低内存环境 |
| 上下文化模型 | 很大 | 强 | 强 | 推理延迟和成本更高 |

新手常见误区是：只看离线评测指标，不看线上预算。实际上线上是否适合用 FastText，取决于三件事：

- 内存能否承受更大的参数表
- 推理时能否接受子词拆分和聚合开销
- OOV 改善是否足以抵消部署成本增加

### 2. 桶数太小会直接伤效果

第二个常见坑是把 `bucket` 调得太小，只为了缩小模型体积。这样做往往会让哈希冲突显著增多，多个本该独立的子词被迫共用参数，结果是：

- 低频词表示更噪
- 相似词与不相似词更难拉开
- OOV 构造出来的向量更不稳定

这个问题的本质不是“训练轮数不够”，而是**表示容量不够**。

可以把 `bucket` 理解成“子词参数空间的容量上限”。容量太小，冲突就像多人共用一个记事本，信息互相覆盖；容量足够大，子词之间的干扰才会下降。

### 3. 不要把 FastText 当成上下文化模型

FastText 擅长的是词形泛化，不是上下文消歧。下面这个对比很重要：

| 问题 | FastText 是否擅长 | 原因 |
|---|---|---|
| `runner` 没见过怎么办 | 是 | 可由子词构造向量 |
| `running` 和 `runner` 是否接近 | 是 | 共享子词参数 |
| `bank` 在不同句子是否有不同含义 | 否 | 仍是静态词向量 |
| 长句中的语义依赖 | 否 | 不建模动态上下文 |

因此，FastText 解决的是“词内部结构”和“低频/OOV 泛化”，不是“上下文语义建模”。

### 4. 中文场景不能机械套用英语结论

这是第四个常见坑。FastText 的经典优势多来自英语、德语、捷克语、芬兰语等依赖词形变化的语言。但中文的情况不同。

中文里：

- 词形变化不像英语、捷克语那么显著
- 分词方案会影响词表和 OOV 比例
- 字符级、词级、短语级边界并不总是稳定
- 专有名词、品牌词、混合串会非常多

因此在中文中，FastText 的收益通常更依赖具体任务：

| 中文任务场景 | FastText 可能的收益 |
|---|---|
| 搜索 Query、商品标题、错拼文本 | 较明显 |
| 领域术语、新词频繁出现 | 较明显 |
| 规范新闻文本、OOV 很少 | 可能有限 |
| 需要上下文理解的复杂任务 | 通常不够 |

也就是说，中文不是不能用 FastText，而是**收益来源更多是字符片段和新词泛化，不是传统意义上的词形变化**。

### 5. 量化经常是上线前必做项

原始 FastText 模型常常偏大，尤其在端侧、移动端或内存敏感服务中。此时常见做法是量化（quantization），也就是把高精度参数压缩成更省空间的近似表示。

官方命令常见形式类似：

```bash
fasttext quantize -input model.bin -output model
```

量化后的工程含义很直接：

- 模型更小
- 常驻内存更低
- 推理更友好
- 精度通常会有一定损失

因此它适合：

- 检索召回
- 轻量文本分类
- 资源受限设备
- 对延迟和体积敏感的服务

但如果任务对每一点精度都很敏感，例如高风险排序或精度要求极严的决策系统，就需要重新做离线和线上评估，而不是默认量化一定划算。

### 6. 一个实际判断框架

如果你在做工程选型，可以直接按下面几个问题判断：

1. 新词和低频词是不是主要痛点？
2. 文本里拼写变体、派生词、混合写法是不是很多？
3. 你需要静态词向量，还是需要上下文化语义？
4. 你的内存预算能不能接受更大的模型？
5. 是否可以接受量化带来的小幅精度下降？

如果前 3 个问题里，前两个答案偏“是”、第三个答案偏“静态向量够用”，那么 FastText 往往是合适候选。

---

## 替代方案与适用边界

FastText 很有用，但它不是默认最优。更合理的做法是把它放回选型坐标系里讨论。

### 1. 和 Word2Vec 的边界

如果任务满足下面条件，Word2Vec 仍然完全合理：

- 语料大，且高频词覆盖已经很好
- OOV 比例低
- 语言词形变化不明显
- 模型体积和推理成本非常敏感

此时 FastText 的额外收益，未必能覆盖额外成本。

### 2. 和 Transformer 的边界

如果你的核心问题是：

- 同一个词在不同句子里意义不同
- 需要句级甚至篇章级理解
- 需要更强的上下文建模

那么 FastText 不够。此时应该考虑带子词分词器的上下文化模型，例如 BERT、RoBERTa、GPT 一类结构。它们能做到“同一个词在不同上下文中有不同表示”，而 FastText 做不到。

但代价也更高：

- 训练更贵
- 推理更慢
- 部署更复杂
- 端侧使用更困难

### 3. 一个压缩后的选型表

| 方案 | 适合场景 | 不适合场景 |
|---|---|---|
| Word2Vec skip-gram | 高频词多、预算紧、OOV 少 | 新词多、形态变化多 |
| FastText | OOV 多、拼写变体多、需要静态向量 | 极端内存受限、需要上下文消歧 |
| 子词感知 Transformer | 需要上下文化语义和复杂理解 | 预算紧、只需轻量静态表示 |

### 4. 一个简单决策规则

可以直接按下面三条判断：

1. 如果你主要担心“新词没有向量”，优先考虑 FastText。
2. 如果你主要担心“同一个词在不同句子里含义不同”，FastText 不够，应看上下文化模型。
3. 如果你主要担心“模型太大、部署太贵”，先看 Word2Vec 或量化后的 FastText。

### 5. 为什么 FastText 在某些语言里收益更稳定

语言越依赖词内部结构，FastText 的性价比通常越高。

| 语言特点 | FastText 的典型收益 |
|---|---|
| 词形变化丰富 | 高 |
| 词缀规则明显 | 高 |
| 派生词很多 | 高 |
| 规范固定写法、OOV 少 | 中或低 |
| 上下文多义远比词形更重要 | 有限 |

所以，对英语，FastText 往往主要改善低频词、派生词、拼写变体；对捷克语、芬兰语一类形态更丰富的语言，它的优势通常更稳定。

结论可以收束成一句话：**FastText 最适合“词内部结构信息本身就很重要”的场景。**

---

## 参考资料

| 资料 | 侧重点 | 对应内容 |
|---|---|---|
| Bojanowski, Grave, Joulin, Mikolov, 2017, *Enriching Word Vectors with Subword Information* | FastText 原始论文 | 子词建模公式、skip-gram 训练目标、Czech syntactic 结果 |
| Mikolov et al., 2013, *Efficient Estimation of Word Representations in Vector Space* | Word2Vec 原始论文 | skip-gram 基础目标、负采样背景 |
| fastText 官方文档与官方仓库 README | 工程实践 | `minn`/`maxn`、`bucket`、量化、训练命令 |
| Joulin et al., *FastText.zip: Compressing text classification models* | 压缩部署 | 量化与低内存部署思路 |
| 各类 fastText 教学文章与实现解读 | 入门直观解释 | `< >` 边界符、`apple` 示例、OOV 机制说明 |

如果只保留最该读的三份资料，可以按这个顺序：

1. Bojanowski et al., 2017：先建立正式公式和实验结论
2. fastText 官方文档：再看参数和工程用法
3. Word2Vec 原始论文或高质量讲解：补足 skip-gram 背景

这样理解会更顺：先知道 FastText 改了什么，再知道它是从哪里改过来的。
