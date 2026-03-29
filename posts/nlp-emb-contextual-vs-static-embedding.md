## 核心结论

静态嵌入和上下文嵌入的根本差别，是“向量是否只由词本身决定”。

静态嵌入指给每个词型分配一个固定向量，词型就是词在词表中的写法，比如 `bank`、`apple`。Word2Vec、GloVe 都属于这一类。它们的优点是简单、便宜、可提前算好，但缺点也直接：同一个词在任何句子里都拿同一个向量，无法区分多义。

上下文嵌入指词向量不仅由词本身决定，还由所在句子的上下文决定。上下文可以理解为目标词前后出现的词，以及整句提供的语义约束。ELMo 用双向 LSTM，也就是从左到右和从右到左两次读句子的循环网络；BERT 用 Transformer，也就是靠自注意力同时看全句关系的编码器。它们能让同一个 `bank` 在“river bank”和“bank loan”里得到不同向量。

对初学者，最直观的判断标准只有一个：模型是否能让“同词不同义”变成“不同向量”。如果不能，它就是静态嵌入；如果能，它就是上下文嵌入。

在效果上，BERT 这类模型通常显著强于 Word2Vec/GloVe。一个常见观察是：同一词在不同上下文中的余弦相似度，功能词如 `the`、`and` 仍然可能很高，接近 0.9；而多义词如 `bank`、`cell` 在不同语境下可能低到 0.2 左右。这说明模型确实把语义差异编码进去了。对应到词义消歧任务，BERT 系方法相对 Word2Vec 类方法常见能提升接近 20 个百分点。

但工程上不能只看精度。静态嵌入可以离线索引，也就是提前给所有词或文档算好向量，查询时直接比对；上下文嵌入往往要在线计算，因为同一个词的向量依赖当前句子，甚至依赖当前会话状态、时间和用户信息。这一点决定了它更强，也更贵。

| 维度 | 静态嵌入 | 上下文嵌入 |
|---|---|---|
| 向量来源 | 只看词型 | 看整句上下文 |
| 多义词区分 | 基本不能 | 天然可以 |
| 离线预计算 | 容易 | 受限 |
| 查询延迟 | 低 | 高 |
| 典型模型 | Word2Vec、GloVe | ELMo、BERT |
| 适合任务 | 静态检索、聚类 | WSD、QA、对话 |

---

## 问题定义与边界

这篇文章讨论的问题很具体：同一个词在不同句子里，模型能不能给出不同的表示，并且这个差异是否真的对应词义差异。

这里的“表示”就是嵌入向量，也就是把词映射到一个高维实数数组。白话讲，它是一组数字，用来让机器判断“两个词或两个语境是不是像”。

玩具例子先看一个最小对比：

- 句子 A：`He sat on the bank of the river.`
- 句子 B：`She works at the bank downtown.`

人能立刻看出两个 `bank` 分别表示“河岸”和“银行”。如果模型无论在哪个句子里都给 `bank` 一个相同向量，那么这个模型不具备词义区分能力。如果它能让两个 `bank` 的向量明显分开，那么它具有上下文化能力。

这个问题通常可用 SelfSim 衡量。SelfSim 可以理解为“同一个词在不同上下文里，向量彼此有多像”。定义为：

$$
\text{SelfSim}_\ell(w)=\frac{1}{n(n-1)}\sum_{j\neq k}\cos\big(f_\ell(s_j,i_j), f_\ell(s_k,i_k)\big)
$$

其中：

- $w$ 是目标词。
- $s_j$ 是第 $j$ 个包含该词的句子。
- $i_j$ 是该词在句子中的位置。
- $f_\ell(s_j,i_j)$ 表示模型第 $\ell$ 层对该位置输出的向量。
- $\cos(\cdot,\cdot)$ 是余弦相似度，也就是衡量两个向量方向是否接近。

如果 $\text{SelfSim}_\ell(w)$ 接近 1，说明这个词在不同上下文里的表示几乎一样，模型没有充分利用上下文；如果明显下降，说明模型开始根据语境区分词义。

边界也要说清楚。

第一，这篇文章主要比较“词级表示”，不是比较完整句向量、段落向量或文档检索系统。虽然它们相关，但问题层次不同。

第二，上下文嵌入不等于“永远更好”。如果任务本身不依赖语义细分，比如大规模离线索引、简单主题聚类、资源极其受限的系统，静态嵌入可能更合适。

第三，“无法离线索引”要理解准确。不是说任何上下文模型都绝对不能预编码，而是说它的表示依赖上下文，一旦上下文在查询时变化，提前算好的向量就可能失效。对纯静态文档库，可以预编码文档；但对需要结合查询、会话历史、用户状态的场景，通常必须在线重新编码。

---

## 核心机制与推导

静态嵌入为什么天然分不清多义词，原因在训练目标。

以 Word2Vec 为例，它通过共现关系学习词向量。共现关系可以理解为“哪些词经常一起出现”。Skip-gram 目标是：给定中心词，预测周围词。于是 `bank` 会同时和 `river`、`loan`、`money`、`shore` 等上下文一起出现。训练结束后，模型只能得到一个折中的 `bank` 向量，把这些语义混在一起。

GloVe 本质也类似，只是更显式地利用全局共现矩阵。矩阵可以理解为“每对词在语料里共同出现了多少次”的大表。它仍然给每个词型一个唯一向量，因此多义问题不会消失。

所以静态嵌入的机制可以概括成一句话：它学习的是“平均语义”，不是“当前语义”。

ELMo 往前走了一步。它先用字符级输入和双向 LSTM 处理整句。LSTM 是一种按顺序读取序列的神经网络，擅长建模前后依赖。对于句子中某个词，ELMo 会拿不同层的隐藏状态做加权求和，形成最终表示。这个“加权求和”很关键，因为低层更像词形和句法特征，高层更像语义特征，模型可以按任务需要组合。

如果写成简化形式，可以表示为：

$$
\text{ELMo}_k = \gamma \sum_{j=0}^{L} s_j h_{k,j}
$$

其中 $h_{k,j}$ 是第 $j$ 层对第 $k$ 个词的隐藏状态，$s_j$ 是可学习权重，$\gamma$ 是缩放参数。直观上，它不是只拿一个层，而是把多层信息混合起来。

BERT 再往前一步。它用 Transformer 编码全句。Transformer 的核心是自注意力，自注意力可以理解为“句子里每个词都去看别的词，并为它们分配权重”。当模型处理 `bank` 时，它会同时看见 `river`、`loan`、`account`、`shore` 等上下文词，并按相关性调整表示。

玩具例子如下：

- `The fisherman waited on the bank near the river.`
- `The banker approved the bank loan quickly.`

在第一句里，`bank` 会对 `river`、`fisherman`、`near` 分配更高注意力；在第二句里，它会更多关注 `loan`、`banker`、`approved`。于是同一个表面词在高层会得到不同语义位置。

这也是为什么 BERT 的高层 SelfSim 会下降。一个常见观察是，BERT-base 的输入层或低层 SelfSim 还比较高，约在 0.82 左右；随着层数上升，到第 12 层可能降到约 0.28。静态嵌入则不管你怎么“看层”，本质都是同一个向量，因此等价于始终 1。

可以把这种趋势看成“从词型到词义”的逐层分化：

| 表示层 | 静态嵌入 SelfSim | BERT SelfSim（示意） | 含义 |
|---|---:|---:|---|
| 输入层/词表层 | 1.00 | 0.82 | 主要保留词型信息 |
| 中间层 | 1.00 | 0.55 | 开始结合句法和局部语义 |
| 高层 | 1.00 | 0.28 | 更强语义细分 |

这里有一个容易误解的点：SelfSim 下降不是越低越好。太低可能说明表示过度分散，泛化变差。真正重要的是，它能否在区分词义的同时保留稳定的语义结构。

真实工程例子可以看词义消歧。词义消歧就是判断一个多义词在当前句子里到底是哪一个义项，比如 `plant` 是“工厂”还是“植物”。传统 Word2Vec 方法通常是：先拿词向量，再和词典释义或样例句做余弦匹配。但因为 `plant` 本身只有一个静态向量，它很难在细粒度语义上做准。BERT 类方法则直接编码整句，再拿目标词的上下文向量去做 kNN 或分类，F1 往往能显著高出一截。

---

## 代码实现

下面先用一个可运行的玩具实现，把“静态嵌入”和“上下文嵌入”的差别做成最小版本。这里不用真实 BERT，而是用规则构造一个上下文向量，目的是让机制一眼可见。

```python
import math

def l2_normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]

def cosine(a, b):
    a = l2_normalize(a)
    b = l2_normalize(b)
    return sum(x * y for x, y in zip(a, b))

STATIC_EMBEDDINGS = {
    "bank": [1.0, 1.0, 1.0],
    "river": [0.0, 1.0, 0.0],
    "loan": [1.0, 0.0, 0.0],
    "money": [1.0, 0.1, 0.0],
    "shore": [0.0, 1.0, 0.1],
}

def get_static_vector(token):
    return STATIC_EMBEDDINGS[token]

def get_contextual_vector(sentence_tokens, token_index):
    token = sentence_tokens[token_index]
    base = get_static_vector(token)[:]

    # 用邻近词构造一个最小上下文偏移量
    for i, t in enumerate(sentence_tokens):
        if i == token_index or t not in STATIC_EMBEDDINGS:
            continue
        ctx = STATIC_EMBEDDINGS[t]
        weight = 0.7 if abs(i - token_index) <= 2 else 0.3
        base = [b + weight * c for b, c in zip(base, ctx)]

    # 去均值，模拟工程里常见的去中心化操作
    mean = sum(base) / len(base)
    base = [x - mean for x in base]
    return l2_normalize(base)

s1 = ["he", "sat", "on", "the", "bank", "near", "the", "river"]
s2 = ["she", "applied", "for", "a", "bank", "loan"]

static_sim = cosine(get_static_vector("bank"), get_static_vector("bank"))
contextual_sim = cosine(get_contextual_vector(s1, 4), get_contextual_vector(s2, 4))

assert round(static_sim, 6) == 1.0
assert contextual_sim < 0.95
print("static_sim =", static_sim)
print("contextual_sim =", contextual_sim)
```

这个例子表达两个事实：

- `get_static_vector("bank")` 与句子无关。
- `get_contextual_vector(sentence, token_index)` 必须先知道整句，再生成目标位置的向量。

如果换成真实工程中的写法，静态嵌入通常类似这样：

```python
def get_static_vector(token, word2vec):
    return word2vec[token]

def search_by_static_embedding(token, index_vectors, topk=5):
    q = normalize(get_static_vector(token))
    return knn_search(q, index_vectors, topk=topk)
```

它的核心优势是 `index_vectors` 可以预先离线构建。离线的意思是数据进入系统前就算好，不用等查询发生。

上下文嵌入的伪代码则是：

```python
def get_contextual_vector(sentence, token_index, tokenizer, bert_model, layer=-1):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = bert_model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer]  # [batch, seq_len, hidden]
    vec = hidden[0, token_index]
    vec = vec - vec.mean()                 # 去均值，减少各向异性影响
    vec = vec / vec.norm()
    return vec

def search_by_context(sentence, token_index, candidate_vectors):
    q = get_contextual_vector(sentence, token_index, tokenizer, bert_model)
    return knn_search(q, candidate_vectors)
```

这里有三个工程关键词必须理解：

- `tokenizer`：分词器，把文本切成模型能处理的 token。
- `hidden_states`：每一层输出的隐藏状态，也就是各层中间表示。
- `layer=-1`：取最后一层。实践中不一定总是最后一层最好，但这是常见起点。

如果做一个真实工程例子，比如在线客服问答：

- 用户问：`How do I close my account at the bank?`
- 系统希望判断 `account` 和 `bank` 的语义，再匹配知识库答案。

这时用静态嵌入可以把词表和文档库提前编码，但它无法精确利用“close + account + bank”这组局部关系。用 BERT 则能更准确地表示当前问句，但必须在请求到来时编码一遍，延迟和算力成本都会上升。

下面这个表格总结“能不能预计算”这件事：

| embedding_type | precompute | 说明 |
|---|---|---|
| Word2Vec/GloVe | `True` | 词向量固定，离线索引直接可用 |
| ELMo | `Partially` | 依赖句子，上下文变了就要重算 |
| BERT token embedding | `False` | 词位置表示依赖整句编码 |
| BERT doc embedding | `Partially` | 静态文档可预编码，但查询相关表示常需在线算 |

---

## 工程权衡与常见坑

第一类权衡是时延与精度。

静态嵌入的计算路径短，部署简单，适合高吞吐检索。高吞吐的意思是单位时间内要处理很多请求。上下文嵌入更准确，但往往要跑完整编码器。对单机 CPU、边缘设备、实时接口来说，这个成本很真实。

一个典型坑是误以为“我把所有词都提前用 BERT 编一遍，就同时拥有高精度和低时延”。这通常不成立。因为 BERT 的词向量不是词表常量，而是句内位置相关的结果。你提前给孤立的 `bank` 算一个向量，到了 `river bank` 或 `bank loan` 语境里并不能直接复用。

第二类坑是各向异性。

各向异性可以理解为“很多向量都挤在相似方向上”。这样即使两个词语义并不近，余弦相似度也可能虚高。BERT 高层表示常见这个问题，所以直接拿最后一层做余弦近邻，效果未必稳定。

常见规避策略有：

- 先做 L2 归一化。
- 先去均值，再算余弦。
- 选择中间层而不是最后一层。
- 做白化或主成分去除。

第三类坑是 token 对齐。

BERT 常用子词切分，也就是把一个词拆成多个更小片段，比如 `embedding` 可能切成 `em`、`##bed`、`##ding`。如果你要取“词”的表示，就必须定义聚合方式：取首子词、取平均、取最后子词，结果可能不同。很多初学者直接用字符位置当 token 位置，会得到错位向量。

第四类坑是缓存策略。

如果系统是对话式 QA，当前用户问题往往依赖历史轮次。此时可以缓存不变部分，比如知识库文档向量、历史消息编码片段，再在当前轮只增量计算新增上下文。增量计算就是只重算变化部分，不重算全部。否则每轮全量编码，成本会迅速上升。

| 常见坑 | 现象 | 原因 | 规避策略 |
|---|---|---|---|
| 上下文依赖被忽略 | 线上效果低于离线实验 | 推理时没带完整上下文 | 在线拼接会话或查询语境 |
| 各向异性 | 近邻全都很像 | 高层向量方向过于集中 | 去均值、归一化、选中间层 |
| 子词错位 | 目标词表示不稳定 | token 和词位置没对齐 | 显式记录子词映射 |
| 预编码误用 | 多义词判断错误 | 把上下文向量当静态向量用 | 区分“文档编码”和“词位编码” |
| 延迟过高 | QPS 上不去 | 在线全量编码过重 | 批处理、蒸馏、小模型、缓存 |

真实工程里，一个比较务实的做法是分层架构：

- 第一层用倒排索引或静态向量做粗召回。
- 第二层用 BERT 类模型做精排或精细判别。

粗召回就是先快速找一批可能相关的候选；精排就是再用更贵的模型把这批候选排序。这种两阶段方法能兼顾成本和效果。

---

## 替代方案与适用边界

如果任务主要关心“整体主题接近”而不是“词义是否精确区分”，静态嵌入仍然有价值。比如小型博客站内搜索、标签聚类、冷启动推荐、教学用最小 NLP 系统，Word2Vec/GloVe 的实现成本和运行成本都更低。

如果任务关心多义词、长距离依赖、句内细微关系，那么上下文嵌入通常更合适。典型场景包括：

- 词义消歧
- 阅读理解
- 问答系统
- 对话系统
- 实体链接
- 语义匹配

这里还要区分几种替代路线。

第一，ELMo 是过渡方案。它比静态嵌入强，因为引入了上下文；但它基于 LSTM，长距离建模和并行效率通常不如 Transformer。今天它更多是教学或历史脉络中的重要节点。

第二，轻量化 Transformer，比如 DistilBERT，是资源受限场景下的折中。折中就是在精度和成本之间找中间点。它不一定达到标准 BERT 的效果，但常能保留大部分收益。

第三，句向量模型和双塔模型适合检索。双塔模型可以理解为“查询一座塔、文档一座塔，分别编码后做向量匹配”。它们通常比逐 token 的上下文表示更适合大规模召回，因为文档可提前编码；但如果你的目标是精确判断某个词位的含义，它们并不是等价替代。

新手最容易记住的一条边界是：

- 要离线索引、便宜、简单，用静态嵌入。
- 要区分多义、理解上下文、做精细语义，用上下文嵌入。
- 要两者兼顾，通常做“静态粗召回 + 上下文精排”。

下面给出任务选择表：

| 任务 | 推荐类型 | 主要考虑因素 |
|---|---|---|
| 博客站内粗搜索 | 静态嵌入/倒排索引 | 延迟低、实现简单 |
| 词义消歧 WSD | 上下文嵌入 | 必须利用句内语义 |
| 智能客服精排 | 上下文嵌入 | 问句含义依赖当前语境 |
| 大规模离线聚类 | 静态嵌入 | 预计算友好、成本低 |
| 交互式 QA | 上下文嵌入 | 需要动态读句子和上下文 |
| 资源受限线上服务 | 轻量 Transformer | 在时延和精度间折中 |

最后用一句话收束：静态嵌入解决的是“这个词通常是什么意思”，上下文嵌入解决的是“这个词在这句话里是什么意思”。两者不是谁彻底淘汰谁，而是服务于不同约束。

---

## 参考资料

1. Ethayarajh, 2019, *How Contextual are Contextualized Word Representations?*  
核心内容：提出并系统分析 SelfSim，指出 BERT 等模型的不同层具有不同程度的上下文化，高层同词表示相似度明显下降。

2. Peters et al., 2018, *Deep Contextualized Word Representations*  
核心内容：提出 ELMo，用双向语言模型的多层隐藏状态加权组合得到词的上下文表示，证明上下文化表示可显著提升多项 NLP 任务。

3. Devlin et al., 2019, *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*  
核心内容：提出双向 Transformer 预训练框架，说明整句上下文编码如何提升下游任务效果，是上下文嵌入进入主流工程实践的关键工作。

4. Loureiro et al., 2021, *Analysis and Evaluation of Language Models for Word Sense Disambiguation*  
核心内容：分析语言模型在词义消歧中的表现，展示上下文模型相对传统静态表示方法的明显优势。

5. Mikolov et al., 2013, *Efficient Estimation of Word Representations in Vector Space*  
核心内容：Word2Vec 代表论文，说明基于局部共现训练静态词向量的方法与优势，也间接解释了其无法区分多义词的根本原因。

6. Pennington et al., 2014, *GloVe: Global Vectors for Word Representation*  
核心内容：提出基于全局共现统计的静态词向量方法，适合理解静态嵌入为何学习到的是词的平均语义。

7. Stanford HAI 相关博客与教程材料  
核心内容：对上下文嵌入给出面向初学者的直观解释，适合作为“固定标签 vs 读完整句再决定含义”的入门参考。

8. 工程实践类资料，如 DataOps School 对 contextual embedding 的介绍  
核心内容：总结上线时延、上下文依赖、增量计算和归一化等工程问题，适合理解理论之外的部署边界。
