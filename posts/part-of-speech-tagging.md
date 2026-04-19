## 核心结论

词性标注是序列标注任务：给句子中的每个词分配一个语法类别，例如名词、动词、形容词、代词、限定词。它的目标不是判断某个词“天生是什么词性”，而是在上下文中找到最合理的标签序列 $y_{1:T}$。

同一个词在不同句子里可能对应不同词性。`book` 在 `a book` 中通常是名词，在 `I book a flight` 中通常是动词。白话说法是：同一个词要看它周围的词，才能知道它在句子里扮演什么角色。

词性标注方法可以按建模方式分成三类：

| 方法 | 核心思想 | 优点 | 局限 |
|---|---|---|---|
| HMM | 用标签转移概率和词发射概率建模整句概率 | 简单、可解释、低资源可用 | 独立性假设强，长距离上下文弱 |
| CRF | 直接建模整句标签序列的条件概率 | 能利用人工特征，序列一致性好 | 特征工程成本高，训练较慢 |
| BiLSTM-CRF | 用双向 LSTM 编码上下文，再用 CRF 解码 | 比传统方法更能利用上下文 | 需要标注数据和训练成本 |
| Transformer tagger | 用自注意力编码整句上下文，逐位置或全局预测标签 | 长距离依赖强，迁移能力好 | 模型更大，部署成本更高 |

核心结论是：词性标注的本质是“在整句范围内做标签序列选择”。HMM 和 CRF 强调序列概率建模，BiLSTM 和 Transformer 强调上下文表示学习。工程上不能只看模型新旧，还要看数据规模、延迟预算、语言特点和标签体系。

---

## 问题定义与边界

输入是词序列 $x_{1:T}$，输出是长度相同的标签序列 $y_{1:T}$。其中 $T$ 表示句子中的词数，$x_t$ 表示第 $t$ 个词，$y_t$ 表示第 $t$ 个词的词性标签。标签来自一个预定义集合，例如 `NOUN`、`VERB`、`ADJ`、`PRON`、`DET`。

玩具例子：

| 词序列 | 候选标签 | 最终标签 |
|---|---|---|
| I | PRON / NOUN | PRON |
| book | NOUN / VERB | VERB |
| a | DET / NOUN | DET |
| flight | NOUN / VERB | NOUN |

对应输出可以写成：

```text
I/PRON book/V a/DET flight/NOUN
```

新手版解释是：系统给每个词贴上语法标签，每个标签表示这个词在句子里的语法功能。

标签集必须统一，不能混用 Penn Treebank、Universal POS、UD。Penn Treebank 是英语树库常用的细粒度标签体系，Universal POS 是跨语言的粗粒度词性集合，UD 是 Universal Dependencies，包含词性和依存句法标注规范。混用标签集会导致训练数据和评测结果不可比较。

词性标注的边界也要明确。它解决的是“词在句中是什么词类”，不直接解决以下问题：

| 任务 | 解决什么 | 与词性标注的关系 |
|---|---|---|
| 句法分析 | 词与词之间的依存或短语结构 | 常以词性作为输入特征 |
| 命名实体识别 | 找出人名、地名、机构名等实体 | 可能利用词性信息，但标签目标不同 |
| 语义角色标注 | 判断谁做了什么、对谁做 | 更偏语义层面 |
| 意图识别 | 判断用户想完成什么操作 | 常用于搜索、对话和推荐系统 |

真实工程例子是电商搜索查询 `苹果手机壳`。系统通常先分词，例如切成 `苹果 / 手机壳` 或 `苹果手机 / 壳`，再判断每个词更像品牌、商品名、属性词还是普通名词。词性标注只负责语法或浅层类别判断，不负责完整商品结构理解。前面的分词错了，后面的词性和意图判断通常会跟着错。

---

## 核心机制与推导

HMM 是隐马尔可夫模型，白话解释是：有一串看不见的状态，也就是词性标签；每个状态会生成一个看得见的词。它有两个核心假设。

第一，标签序列满足马尔可夫性。马尔可夫性是指当前标签主要依赖上一个标签，而不是依赖完整历史：

$$
P(y_t \mid y_{1:t-1}) \approx P(y_t \mid y_{t-1})
$$

第二，观测词只依赖当前标签：

$$
P(x_t \mid y_{1:T}, x_{1:t-1}) \approx P(x_t \mid y_t)
$$

因此 HMM 里的词性标注可以写成：

$$
\hat y=\arg\max_{y_{1:T}} P(y_{1:T},x_{1:T})
=\arg\max_{y_{1:T}}\prod_{t=1}^{T}P(y_t\mid y_{t-1})P(x_t\mid y_t)
$$

其中 $y_0=<s>$ 表示句子开始标签。$P(y_t\mid y_{t-1})$ 叫转移概率，表示某个词性后面接另一个词性的概率。$P(x_t\mid y_t)$ 叫发射概率，表示某个词性生成某个词的概率。

用 `I book` 做一个最小数值例子，候选标签只有 `N` 和 `V`，分别表示名词和动词。

设：

| 概率 | 数值 |
|---|---:|
| $P(N\mid <s>)$ | 0.4 |
| $P(V\mid <s>)$ | 0.6 |
| $P(I\mid N)$ | 0.2 |
| $P(I\mid V)$ | 0.05 |
| $P(book\mid N)$ | 0.7 |
| $P(book\mid V)$ | 0.9 |
| $P(V\mid N)$ | 0.7 |
| $P(N\mid N)$ | 0.3 |
| $P(V\mid V)$ | 0.6 |
| $P(N\mid V)$ | 0.4 |

四条路径的分数是：

| 路径 | 分数 |
|---|---:|
| N,N | $0.4 \times 0.2 \times 0.3 \times 0.7=0.0168$ |
| N,V | $0.4 \times 0.2 \times 0.7 \times 0.9=0.0504$ |
| V,N | $0.6 \times 0.05 \times 0.4 \times 0.7=0.0084$ |
| V,V | $0.6 \times 0.05 \times 0.6 \times 0.9=0.0162$ |

最大的是 `N,V`，所以 `I` 被标成 `N`，`book` 被标成 `V`。这里的标签集合很粗，只是为了演示机制。真实系统会把 `I` 标成 `PRON`，不会标成普通名词。

如果句子很长，直接枚举所有标签路径会爆炸。假设每个位置有 $K$ 个候选标签，句长为 $T$，路径数是 $K^T$。Viterbi 算法用动态规划避免完整枚举。动态规划是把大问题拆成可复用的小问题，只保留每个阶段的最优中间结果。

Viterbi 递推公式是：

$$
\delta_t(s)=P(x_t\mid s)\max_{s'}\delta_{t-1}(s')P(s\mid s')
$$

其中 $\delta_t(s)$ 表示“到第 $t$ 个词为止，并且第 $t$ 个标签是 $s$”的最优路径分数。算法还会保存回溯指针，记录这个最优分数来自哪个上一个标签。最后从终点最大分数开始回溯，得到整句最优标签序列。

| 对比项 | 局部最优 | 全局最优 |
|---|---|---|
| 决策方式 | 每个词单独选最高分标签 | 比较整条标签路径 |
| 是否考虑相邻标签 | 弱 | 强 |
| 典型问题 | `book` 可能只按词频被判成名词 | 会结合 `I` 后面接动词的概率 |
| 对应方法 | 逐位置 softmax | HMM Viterbi、CRF 解码 |

神经网络方法把重点从“手写概率表”转向“学习上下文表示”。表示是模型为每个词计算出的向量，向量里编码了它和上下文的关系。常见形式是：

$$
h_{1:T}=\text{Encoder}(x_{1:T}),\quad p(y_t\mid x_{1:T})=\text{softmax}(Wh_t+b)
$$

Encoder 可以是 BiLSTM，也可以是 Transformer。BiLSTM 是双向长短期记忆网络，会从左到右和从右到左各读一遍句子。Transformer 使用自注意力机制，让每个词直接关注句子中的其他词。

如果在神经网络后面接 CRF，就不是每个位置独立取最大标签，而是对整句标签序列做全局归一化。全局归一化是指所有可能标签路径一起参与概率归一化，因此解码时仍需要动态规划，常用 Viterbi 找最优路径。

---

## 代码实现

下面是最小可运行的 `HMM + Viterbi` 示例。它只覆盖两个标签和两个词，用来展示状态转移、回溯指针和最终解码。

```python
def viterbi(words, tags, start_p, trans_p, emit_p):
    scores = []
    backpointers = []

    first_scores = {}
    first_back = {}
    for tag in tags:
        first_scores[tag] = start_p[tag] * emit_p[tag].get(words[0], 0.0)
        first_back[tag] = None
    scores.append(first_scores)
    backpointers.append(first_back)

    for t in range(1, len(words)):
        current_scores = {}
        current_back = {}
        for tag in tags:
            best_prev = None
            best_score = -1.0
            for prev in tags:
                score = scores[t - 1][prev] * trans_p[prev][tag] * emit_p[tag].get(words[t], 0.0)
                if score > best_score:
                    best_score = score
                    best_prev = prev
            current_scores[tag] = best_score
            current_back[tag] = best_prev
        scores.append(current_scores)
        backpointers.append(current_back)

    last_tag = max(scores[-1], key=scores[-1].get)
    path = [last_tag]

    for t in range(len(words) - 1, 0, -1):
        path.append(backpointers[t][path[-1]])

    path.reverse()
    return path, scores[-1][last_tag]


tags = ["N", "V"]
words = ["I", "book"]

start_p = {"N": 0.4, "V": 0.6}
trans_p = {
    "N": {"N": 0.3, "V": 0.7},
    "V": {"N": 0.4, "V": 0.6},
}
emit_p = {
    "N": {"I": 0.2, "book": 0.7},
    "V": {"I": 0.05, "book": 0.9},
}

path, score = viterbi(words, tags, start_p, trans_p, emit_p)

assert path == ["N", "V"]
assert abs(score - 0.0504) < 1e-9
print(list(zip(words, path)), score)
```

变量含义如下：

| 变量 | 含义 |
|---|---|
| `scores` | 每个位置、每个标签的最优路径分数 |
| `backpointers` | 回溯指针，记录当前最优路径来自哪个上一个标签 |
| `emit_p` | 发射概率 $P(x_t \mid y_t)$ |
| `trans_p` | 转移概率 $P(y_t \mid y_{t-1})$ |
| `start_p` | 句首标签概率 $P(y_1 \mid <s>)$ |

神经网络版本通常不手写概率表，而是让模型从数据中学习。下面是 `BiLSTM-CRF` 的伪代码，展示工程结构，不依赖具体框架。

```python
class BiLSTMCRFTagger:
    def __init__(self, vocab_size, tag_size):
        self.embedding = Embedding(vocab_size, dim=128)
        self.encoder = BiLSTM(input_dim=128, hidden_dim=256)
        self.classifier = Linear(input_dim=512, output_dim=tag_size)
        self.crf = CRF(tag_size)

    def forward(self, token_ids, gold_tags=None):
        x = self.embedding(token_ids)
        h = self.encoder(x)
        emissions = self.classifier(h)

        if gold_tags is not None:
            loss = -self.crf.log_likelihood(emissions, gold_tags)
            return loss

        best_tag_path = self.crf.decode(emissions)
        return best_tag_path
```

这个结构可以理解成四步：

| 步骤 | 作用 |
|---|---|
| `Embedding` | 把词 ID 转成向量 |
| `BiLSTM` | 编码左右上下文 |
| `Linear` | 给每个位置产生各标签分数 |
| `CRF` | 按整句标签序列做训练和解码 |

如果只是快速试用，也可以直接调用现成工具。例如 NLTK 的接口形式通常是：

```python
import nltk

tokens = ["I", "book", "a", "flight"]
tags = nltk.pos_tag(tokens)

assert len(tags) == len(tokens)
print(tags)
```

真实项目里，现成库更适合作为基线。基线是第一个可运行、可比较的版本，用来判断后续自研模型是否真的带来收益。

---

## 工程权衡与常见坑

词性标注的工程效果通常不只由模型决定。分词、标签体系、领域数据、评测方式都会影响最终质量。

中文场景里，分词质量尤其关键。比如电商查询 `苹果手机壳`，如果分成 `苹果 / 手机壳`，系统可能把 `苹果` 识别为品牌或名词，把 `手机壳` 识别为商品名。如果错误分成 `苹果手机 / 壳`，后续词性和商品理解都会偏。白话说法是：前面的步骤错了，后面就会跟着错。

常见坑如下：

| 常见坑 | 表现 | 规避策略 |
|---|---|---|
| 分词错误 | 中文词边界错，词性跟着错 | 联合优化分词和标注，或使用字词混合模型 |
| 标签集混用 | 训练、预测、评测标签不一致 | 项目开始前固定 Penn Treebank、Universal POS 或 UD |
| OOV | 新词、品牌词、术语无法识别 | 补词表、用子词模型、加入领域语料 |
| 领域迁移 | 新闻上好，电商或医疗上差 | 做领域微调，单独构造领域验证集 |
| 只看 accuracy | 高频标签掩盖低频标签问题 | 同时看 macro F1、混淆矩阵、关键标签召回 |

OOV 是 out-of-vocabulary，意思是训练词表之外的新词。比如新品牌名、网络流行语、医学术语，传统 HMM 或 MaxEnt 容易处理不好。MaxEnt 是最大熵模型，白话解释是：它用人工特征直接预测标签，在传统 NLP 中常作为判别式分类器使用。

不同模型的工程取舍可以简化为：

| 模型 | 延迟 | 精度 | 可迁移性 | 典型适用场景 |
|---|---|---|---|---|
| HMM | 低 | 低到中 | 弱 | 低资源、规则稳定、可解释需求强 |
| MaxEnt | 低到中 | 中 | 中 | 有人工特征、需要轻量部署 |
| BiLSTM-CRF | 中 | 中到高 | 中到高 | 有标注数据、要求序列一致性 |
| Transformer | 高 | 高 | 高 | 长上下文、多领域迁移、高质量标注 |

低延迟 CPU 服务可以优先考虑 HMM、MaxEnt 或小型神经网络。高准确率、长上下文、多领域文本更适合 BiLSTM-CRF 或 Transformer。需要注意的是，Transformer 并不自动解决所有问题。如果训练数据的标签本身不一致，模型只会学到不一致。

评测时不要只看整体 accuracy。词性标注里名词、动词等高频标签很多，模型即使低频标签很差，整体准确率也可能好看。更可靠的做法是看每类标签的 precision、recall、F1，以及常见混淆。例如 `NOUN` 和 `PROPN`、`ADJ` 和 `VERB`、中文名词和量词之间的混淆。

---

## 替代方案与适用边界

词性标注没有唯一最优方案。传统统计方法、判别式方法和深度学习方法各有适用边界。不能只按“模型新旧”决策。

| 方案 | 适合场景 | 优点 | 缺点 | 是否易部署 |
|---|---|---|---|---|
| HMM | 数据少、标签少、规则稳定 | 简单、快、可解释 | 上下文弱，OOV 差 | 易 |
| CRF | 有人工特征、需要全局序列约束 | 序列一致性好 | 特征工程重 | 中 |
| BiLSTM-CRF | 中等规模标注数据，强调上下文 | 效果稳，序列解码强 | 训练和推理成本更高 | 中 |
| Transformer | 大规模数据、复杂领域、长上下文 | 表示能力强，可迁移 | 模型大，延迟和成本高 | 较难 |

同样是词性标注，新闻文本、社交媒体、行业术语、古文会有不同表现。新闻文本更规范，传统模型容易取得稳定效果。社交媒体有缩写、错别字和口语表达，需要更强的鲁棒性。行业文本包含大量术语，通常需要领域词表或领域微调。古文和现代汉语差异明显，不能直接套用现代语料训练出的标签器。

如果只需要粗粒度词性、数据少、预算低，传统方法足够。如果要求更高召回和上下文一致性，优先神经网络模型。如果还需要和下游任务联合优化，例如信息抽取、问答、搜索排序，可以考虑把词性标注作为多任务学习的一部分，而不是单独训练一个固定模块。

还有一种现实选择是不显式做词性标注。很多 Transformer 下游模型可以直接从原始文本学习任务表示，例如文本分类、实体识别、检索排序。此时词性标注不一定是必需步骤。但如果系统需要可解释的语言学特征、规则回退、轻量部署，或者下游模块明确依赖词性，单独的词性标注器仍然有价值。

工程决策可以按三个问题收敛：

| 问题 | 倾向方案 |
|---|---|
| 数据少、延迟敏感、标签简单 | HMM / MaxEnt |
| 有稳定标注数据，需要整句一致性 | CRF / BiLSTM-CRF |
| 领域复杂、上下文长、可接受成本 | Transformer tagger |

最强模型不一定最好。真正的选择标准是：在你的数据、标签集、部署环境和下游任务里，哪个方案用最小成本达到足够质量。

---

## 参考资料

理论/算法：

1. [TnT - A Statistical Part-of-Speech Tagger](https://aclanthology.org/A00-1031/)：用于解释 HMM、统计词性标注和传统低成本方案。
2. [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991)：用于解释 BiLSTM-CRF 的编码与全局解码结构。
3. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：用于解释 Transformer 和自注意力建模长距离上下文。

工程文档：

4. [Stanford CoreNLP POS Tagger](https://stanfordnlp.github.io/CoreNLP/tools_pos_tagger.html)：用于了解传统 NLP 工具中的词性标注器使用方式。
5. [spaCy Tagger](https://spacy.io/api/tagger/)：用于了解工业 NLP 管线中的 tagger 组件。
6. [NLTK pos_tag](https://www.nltk.org/api/nltk.tag.pos_tag.html)：用于最小工具调用示例和教学场景。

标签体系：

7. [Universal POS Tags](https://universaldependencies.org/u/pos/)：用于说明跨语言粗粒度词性标签体系。
8. [Universal Dependencies](https://universaldependencies.org/)：用于说明 UD 标注规范和标签体系边界。
