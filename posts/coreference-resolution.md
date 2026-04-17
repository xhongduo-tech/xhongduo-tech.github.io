## 核心结论

实体指代消解（coreference resolution）要解决的，不是“某个词是不是实体”这一类局部判断，而是把文档中所有指向同一对象的提及（mention）组织成共指链（coreference chain）。

例如：

“乔布斯去世了，他的遗产由库克继承。”

这里“乔布斯”和“他”指向同一个人，因此应该进入同一条共指链；“库克”则属于另一条链。任务目标不是只把“他”识别成代词，而是判断它和前文哪个提及指向同一实体。

形式化地说，给定文档中的提及集合
$$
M=\{m_1,m_2,\dots,m_n\}
$$
共指消解要学习的是一个划分：
$$
C=\{c_1,c_2,\dots,c_k\}, \quad \bigcup_{i=1}^{k} c_i=M,\quad c_a\cap c_b=\varnothing\ (a\neq b)
$$
其中每个集合 $c_i$ 内的所有提及都指向同一个实体。

过去常见做法是 pipeline：先做提及检测，再生成候选先行词，再做两两分类或排序。现在主流方向更偏向端到端神经模型，即同时学习两件事：

1. 哪些文本片段本身是有效提及。
2. 这些提及之间应该如何连接成链。

这样做的核心收益是减少上游错误向下游传播。例如，如果 pipeline 的第一步漏掉了“苹果公司”，那么后面的“这家公司”“它”即使判断得再准，也失去了可连接的目标；端到端模型会把“提及发现”和“共指链接”联合优化，通常更稳。

在公开基准 CoNLL-2012 上，系统通常综合 MUC、B³、CEAF 三类指标进行评价。公开强模型的整体 F1 常被报告在约 83.5% 到 83.6% 这一量级。这说明共指消解不是一个“已经完全解决”的问题，但在新闻理解、长文档信息抽取、知识图谱构建、摘要生成等场景中，已经具有明确工程价值。

| 指标 | 它衡量什么 | 直观含义 |
|---|---|---|
| MUC | 链接恢复能力 | 看模型有没有把本应连在一起的提及连起来 |
| B³ | 单个提及的聚类质量 | 看每个提及被分到的链是否正确 |
| CEAF | 预测链与真实链的整体对齐质量 | 看整条链作为一个整体是否匹配 |

还可以把这三个指标理解为三个观察角度：

| 观察角度 | 更关注什么 | 容易忽略什么 |
|---|---|---|
| MUC | 链接是否打通 | 单个提及错分的细粒度问题 |
| B³ | 每个提及是否进对了链 | 整链级别的全局结构 |
| CEAF | 预测链和真实链是否整体对齐 | 某些局部链接的微小差异 |

所以，共指消解的本质不是“实体识别的附属功能”，而是一个文档级结构预测任务。

---

## 问题定义与边界

先把边界说清楚。实体指代消解处理的是“文档内部显式出现的提及之间，哪些指向同一实体”。这里有三个基本概念，必须区分。

第一，提及（mention）。它是文本里一次具体出现的表达，可以是：

- 专有名词，如“苹果公司”
- 代词，如“它”“他”“她”
- 名词短语，如“这家公司”“这位研究者”

第二，实体（entity）。它不是文本片段，而是这些片段背后共同指向的对象。例如“苹果公司”“这家公司”“它”在特定上下文中都可能指向同一个实体。

第三，共指链（coreference chain）。它是一组提及的集合，这组提及都指向同一实体。

看一个最小例子：

文本：
“苹果公司发布了新手机。这家公司表示，它今年将继续加大芯片投入。”

可以得到如下共指结构：

| 提及 | 类型 | 所属链 | 说明 |
|---|---|---|---|
| 苹果公司 | 专名 | 链 A | 公司实体的首次出现 |
| 这家公司 | 名词短语 | 链 A | 对前文公司的回指 |
| 它 | 代词 | 链 A | 再次回指同一公司 |
| 新手机 | 普通名词 | 链 B 或无关 | 这是另一个对象，不属于公司链 |

这里最容易混淆的，是共指消解与其他 NLP 任务的区别。

| 任务 | 回答的问题 | 输出形式 |
|---|---|---|
| 命名实体识别（NER） | “这段文本是不是人名、地名、机构名？” | 实体边界 + 类型 |
| 实体链接（EL） | “这段文本对应知识库中的哪个条目？” | 文本到知识库实体的映射 |
| 共指消解（CR） | “这些提及是不是同一个对象？” | 文档内的共指链 |

例如：

“苹果公司发布新品。它表示出货量会增长。”

- NER 关心“苹果公司”是不是机构名。
- 共指消解关心“它”是不是“苹果公司”。
- 实体链接关心“苹果公司”应不应该连到某个知识库条目，例如 Apple Inc.

这几个任务在工程里经常串联，但它们不是同一个问题。一个常见流水线是：

文本 → NER 找出显式实体 → 共指消解把“它”“该公司”等并回前文 → 实体链接把整条链对齐到知识库

边界也必须明确：

| 处理范围 | 是否属于典型共指消解 | 原因 |
|---|---|---|
| 文档中显式出现的代词回指 | 是 | 标准任务核心内容 |
| 文档中显式出现的同义名词短语 | 是 | 例如“苹果公司”与“这家公司” |
| 需要大量世界知识的隐含推理 | 通常不是强项 | 模型未必能稳定推断 |
| 省略主语、篇章外补全 | 常不在标准设定内 | 超出文档内显式提及范围 |
| 跨文档同名实体对齐 | 不属于标准文档内共指 | 更接近实体消歧或跨文档实体聚类 |

例如：

“总统发表讲话。白宫随后回应。”

这里“白宫”是否等价于“美国政府”或“总统所属政府班子”，已经涉及转喻和世界知识，不是标准共指系统最稳定的处理对象。

再看一个经典歧义例子：

“奖杯装不进手提箱，因为它太大了。”

“它”到底指“奖杯”还是“手提箱”，不能只靠最近距离判断，而需要结合事件语义和常识。Winograd Schema 这一类样本，正好说明了共指消解的边界：不是所有歧义都能通过表面形式解决。

对新手来说，最稳的理解方式是：

- 提及是文本里的说法
- 实体是这些说法背后的对象
- 共指消解是把“不同说法”归并到“同一对象”

---

## 核心机制与推导

主流神经共指模型可以粗分为三类思路：

1. Mention-Ranking
2. Span-BERT 或 BERT 类编码器上的联合建模
3. 端到端 span-ranking

三者的共同点是：它们都不把共指看成一个简单标签，而是看成“提及之间的结构连接问题”。

### 1. Mention-Ranking：对先行词做排序

Mention-Ranking 的核心想法很直接。对于当前提及 $m_i$，枚举它之前的所有候选先行词（antecedent），然后给每个候选打分，选得分最高的那个。

这里不是在做“这两者是否共指”的独立二分类，而是在做排序：

- 候选 A 是否比候选 B 更像正确先行词
- 当前提及是否应该放弃连接，选择空先行词

例如句子：

“乔布斯去世了，他的遗产由库克继承。”

当模型处理“他”时，它的候选先行词可能是：

- 乔布斯
- 库克
- 空先行词（不回指任何前文）

模型要做的不是分别输出三个“是/否”，而是比较三者谁最合理。

### 2. 端到端 span-ranking：同时学提及和链接

端到端 span-ranking 比 Mention-Ranking 更进一步。它不预先假定“哪些片段一定是提及”，而是先枚举一批候选 span，再联合学习：

1. 这个 span 像不像一个有效提及
2. 如果它是提及，它应该连接到哪个前文 span

经典打分函数通常写成：
$$
s(i,j)=s_m(i)+s_m(j)+s_a(i,j)
$$

其中：

- $s_m(i)$：当前 span $i$ 作为提及的分数
- $s_m(j)$：候选先行词 $j$ 作为提及的分数
- $s_a(i,j)$：$i$ 与 $j$ 的配对分数

这个公式很重要，因为它明确区分了三个判断来源：

| 项 | 作用 | 直观解释 |
|---|---|---|
| $s_m(i)$ | 判断当前 span 是否像提及 | 当前片段本身是否值得建模 |
| $s_m(j)$ | 判断候选先行词是否像提及 | 前文片段是否像可被回指的对象 |
| $s_a(i,j)$ | 判断两者是否匹配 | 即使两边都像提及，也不一定共指 |

所以，一个链接成立，不是因为两段文本都“像实体”，而是因为它们作为两个提及彼此兼容。

### 3. 候选集合与 Dummy antecedent

对每个候选提及 $i$，模型定义一个候选集合 $Y(i)$：
$$
Y(i)=\{\epsilon,1,2,\dots,i-1\}
$$
其中 $\epsilon$ 表示 Dummy antecedent，也就是“空先行词”。

Dummy 的作用是允许模型表达：

- 当前 span 是一个提及，但它不是回指，而是某条新链的开始
- 或者当前 span 不值得与前文任何提及连接

这一步非常关键，因为不是每个提及都应当回指前文。例如在文档开头第一次出现“乔布斯”时，它通常就没有先行词。

### 4. softmax 概率与训练目标

给定候选集合后，对每个 span $i$，模型定义：
$$
P(y_i \mid D)=\frac{\exp(s(i,y_i))}{\sum_{y' \in Y(i)} \exp(s(i,y'))}
$$

含义是：在整篇文档 $D$ 中，当前提及 $i$ 选择候选 $y_i$ 作为先行词的条件概率。

整篇文档的联合目标通常写成：
$$
P(y_1,\dots,y_N\mid D)=\prod_{i=1}^{N} P(y_i\mid D)
$$

训练时，一般最大化正确先行词的概率；等价地，也可以最小化负对数似然：
$$
\mathcal{L}=-\sum_{i=1}^{N}\log P(y_i^\ast \mid D)
$$
其中 $y_i^\ast$ 是 gold antecedent。若某个提及没有先行词，则令 $y_i^\ast=\epsilon$。

很多实现会把 Dummy 的分数设为常数 0：
$$
s(i,\epsilon)=0
$$
这相当于给“不连接”提供一个统一基线。其他候选只有在得分明显高于这个基线时，才会被选中。

### 5. 用具体例子推导一遍

仍然看这句话：

“乔布斯去世了，他的遗产由库克继承。”

设当前提及是“他”，候选集合为：

- “乔布斯”
- “库克”
- Dummy

假设模型给出的总分为：

| 候选 | 分数 $s(i,j)$ | 含义 |
|---|---:|---|
| 乔布斯 | 5.2 | 最可能的先行词 |
| 库克 | 3.1 | 也是人名，但语义和位置都较弱 |
| Dummy | 0.0 | 不回指任何前文 |

softmax 概率为：
$$
P(\text{乔布斯})=\frac{e^{5.2}}{e^{5.2}+e^{3.1}+e^0}
$$

近似值：

- $e^{5.2}\approx 181.27$
- $e^{3.1}\approx 22.20$
- $e^{0}=1$

因此：
$$
P(\text{乔布斯})\approx \frac{181.27}{181.27+22.20+1}\approx 0.886
$$

同理可得：
$$
P(\text{库克})\approx \frac{22.20}{204.47}\approx 0.109
$$

$$
P(\text{Dummy})\approx \frac{1}{204.47}\approx 0.005
$$

于是模型会把“他”连接到“乔布斯”，并把两者放入同一条共指链。

### 6. 为什么 SpanBERT 有帮助

SpanBERT 路线的关键改进不在任务定义，而在表示学习。BERT 关注上下文编码，而 SpanBERT 更强调 span 级表征，即让模型更擅长表示“一个片段”的边界和内部内容。

对于共指任务，这很关键，因为模型反复要回答两类问题：

- 这段 span 是否像一个独立提及
- 这个 span 与另一个 span 是否指向同一对象

如果片段表示能力更强，$s_m(i)$ 和 $s_a(i,j)$ 的质量通常也会提升。因此 SpanBERT 常被用作更强的底层编码器。

### 7. 从结构角度理解整条流水线

可以把主流端到端共指系统概括成下面这条结构链：

span 枚举 → 提及打分 → 剪枝保留高质量 span → 为每个 span 枚举前文候选 → 计算配对分数 → softmax 选先行词 → 合并得到共指链

更细一点，可以写成：

| 阶段 | 输入 | 输出 | 目的 |
|---|---|---|---|
| 编码 | token 序列 | 上下文化 token 表示 | 给每个词上下文信息 |
| span 枚举 | token 表示 | 候选 span 集合 | 构造潜在提及 |
| mention scoring | span 表示 | $s_m(i)$ | 判断哪些 span 值得保留 |
| antecedent scoring | span 对 | $s_a(i,j)$ | 判断 span 之间是否适合连接 |
| 归一化选择 | 候选得分 | 先行词分布 | 为每个提及选 antecedent |
| 链构建 | 局部连接结果 | 共指链 | 恢复文档级结构 |

对初学者来说，最重要的不是记住每个模块名称，而是抓住一个结构性事实：

共指消解不是“给一句话打标签”，而是在文档层面对多个 span 的连接关系做结构预测。

---

## 代码实现

下面给一个可运行的玩具实现。它不依赖深度学习框架，而是用 Python 标准库演示“对候选先行词打分、softmax 归一化、选择最优先行词、再组装共指链”的最小流程。

这个实现不是真实的 BERT/SpanBERT 模型，但它保留了共指模型最关键的接口设计：

- 每个提及有 mention score
- 每对提及有 pairwise score
- 每个提及在候选先行词集合上做 softmax
- 选择最佳先行词后，可把整篇文档恢复成共指链

```python
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


DUMMY = "DUMMY"


@dataclass(frozen=True)
class Mention:
    mid: str
    text: str
    start: int
    end: int


def stable_softmax(xs: List[float]) -> List[float]:
    """Numerically stable softmax."""
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]


def antecedent_score(
    current: Mention,
    candidate_id: str,
    mention_scores: Dict[str, float],
    pair_scores: Dict[Tuple[str, str], float],
) -> float:
    """s(i, j) = s_m(i) + s_m(j) + s_a(i, j), Dummy score = 0."""
    if candidate_id == DUMMY:
        return 0.0

    if current.mid not in mention_scores:
        raise KeyError(f"Missing mention score for current mention: {current.mid}")
    if candidate_id not in mention_scores:
        raise KeyError(f"Missing mention score for candidate mention: {candidate_id}")

    pair_key = (current.mid, candidate_id)
    pair_score = pair_scores.get(pair_key, float("-inf"))
    if pair_score == float("-inf"):
        raise KeyError(f"Missing pair score for {pair_key}")

    return (
        mention_scores[current.mid]
        + mention_scores[candidate_id]
        + pair_score
    )


def choose_antecedent(
    current: Mention,
    candidate_ids: List[str],
    mention_scores: Dict[str, float],
    pair_scores: Dict[Tuple[str, str], float],
) -> Tuple[str, List[float], List[float]]:
    """Return best antecedent, probabilities, and raw scores."""
    scores = [
        antecedent_score(current, cand_id, mention_scores, pair_scores)
        for cand_id in candidate_ids
    ]
    probs = stable_softmax(scores)
    best_idx = max(range(len(candidate_ids)), key=lambda i: probs[i])
    return candidate_ids[best_idx], probs, scores


def resolve_document(
    mentions: List[Mention],
    mention_scores: Dict[str, float],
    pair_scores: Dict[Tuple[str, str], float],
) -> Dict[str, Optional[str]]:
    """
    Resolve each mention to its best antecedent.
    Returns a mapping: mention_id -> antecedent_id or None.
    """
    antecedents: Dict[str, Optional[str]] = {}

    for i, current in enumerate(mentions):
        previous_ids = [m.mid for m in mentions[:i]]
        candidate_ids = previous_ids + [DUMMY]
        best, probs, scores = choose_antecedent(
            current=current,
            candidate_ids=candidate_ids,
            mention_scores=mention_scores,
            pair_scores=pair_scores,
        )

        antecedents[current.mid] = None if best == DUMMY else best

        print(f"\nCurrent mention: {current.text} ({current.mid})")
        for cand_id, score, prob in zip(candidate_ids, scores, probs):
            label = "DUMMY" if cand_id == DUMMY else cand_id
            print(f"  candidate={label:8s} score={score:6.2f} prob={prob:.4f}")
        print(f"  chosen antecedent: {antecedents[current.mid]}")

    return antecedents


def build_chains(
    mentions: List[Mention],
    antecedents: Dict[str, Optional[str]],
) -> List[List[str]]:
    """
    Convert antecedent links into chains.
    A simple union-find-free implementation is enough for this toy example.
    """
    chain_by_mention: Dict[str, int] = {}
    chains: List[List[str]] = []

    for mention in mentions:
        ant = antecedents[mention.mid]
        if ant is None:
            chain_id = len(chains)
            chains.append([mention.mid])
            chain_by_mention[mention.mid] = chain_id
        else:
            chain_id = chain_by_mention[ant]
            chains[chain_id].append(mention.mid)
            chain_by_mention[mention.mid] = chain_id

    return chains


if __name__ == "__main__":
    # Document: "乔布斯去世了，他的遗产由库克继承。"
    mentions = [
        Mention(mid="m1", text="乔布斯", start=0, end=3),
        Mention(mid="m2", text="他", start=6, end=7),
        Mention(mid="m3", text="库克", start=12, end=14),
    ]

    # Mention confidence scores
    mention_scores = {
        "m1": 1.9,  # 乔布斯
        "m2": 1.5,  # 他
        "m3": 1.2,  # 库克
    }

    # Pairwise antecedent scores: (current, candidate)
    pair_scores = {
        ("m2", "m1"): 1.8,   # 他 -> 乔布斯
        ("m3", "m1"): -0.5,  # 库克 -> 乔布斯（不共指）
        ("m3", "m2"): -1.0,  # 库克 -> 他（不共指）
    }

    antecedents = resolve_document(mentions, mention_scores, pair_scores)
    chains = build_chains(mentions, antecedents)

    print("\nAntecedent mapping:")
    print(antecedents)

    print("\nPredicted chains:")
    for idx, chain in enumerate(chains, start=1):
        print(f"  Chain {idx}: {chain}")

    # Expected:
    # m1 starts a new chain
    # m2 links back to m1
    # m3 starts another chain
    assert antecedents == {
        "m1": None,
        "m2": "m1",
        "m3": None,
    }
    assert chains == [
        ["m1", "m2"],
        ["m3"],
    ]
```

这段代码可以直接运行。它会打印每个提及的候选先行词、对应分数、softmax 概率，以及最终恢复出的共指链。

如果把运行逻辑翻译成自然语言，大致是这样：

| 步骤 | 发生了什么 | 对应代码 |
|---|---|---|
| 1 | 枚举文档中的提及 | `mentions` |
| 2 | 当前提及只看前文候选 | `previous_ids + [DUMMY]` |
| 3 | 对每个候选计算总分 | `antecedent_score()` |
| 4 | 对分数做 softmax | `stable_softmax()` |
| 5 | 选概率最大的候选 | `choose_antecedent()` |
| 6 | 把局部连接恢复成链 | `build_chains()` |

如果把这套玩具实现扩展到真实神经模型，典型流程会变成：

1. 对整篇文档分词。
2. 用编码器生成上下文化 token 表示。
3. 枚举长度不超过阈值的 span。
4. 为每个 span 构造表示，例如起点向量、终点向量、内部注意力汇总。
5. 用 mention scorer 计算 $s_m(i)$，并做剪枝。
6. 对剩余 span 枚举前文候选。
7. 用 antecedent scorer 计算 $s_a(i,j)$。
8. 在每个候选集合上做 softmax。
9. 用 gold antecedent 训练，推理时取最大概率候选。
10. 根据 antecedent 链接恢复整条共指链。

伪代码如下：

```python
# tokenization
tokens = tokenize(document)

# contextual encoding
token_vecs = encoder(tokens)

# candidate spans
spans = enumerate_spans(tokens, max_width=10)

# span representation
span_reprs = [build_span_repr(token_vecs, span) for span in spans]

# mention scoring
mention_scores = [mention_mlp(h) for h in span_reprs]

# pruning
top_spans = prune(spans, mention_scores, top_k_ratio=0.4)

loss = 0.0
for i in top_spans:
    candidates = previous_spans(i, top_spans) + [DUMMY]
    scores = []
    for j in candidates:
        if j == DUMMY:
            scores.append(0.0)
        else:
            scores.append(sm(i) + sm(j) + sa(i, j))
    probs = softmax(scores)
    loss += negative_log_likelihood(probs, gold_antecedent(i))
```

对初学者来说，真正该抓住的是这两个层次：

- 局部层次：每个提及都要从若干前文候选里选一个 antecedent
- 全局层次：这些局部选择最后会拼成整篇文档的共指链

这也是为什么共指消解本质上是“结构预测”，而不是一个普通分类器。

---

## 工程权衡与常见坑

共指消解在论文里看起来像“加一个模块”，但在真实系统里通常是高耦合组件，因为它一头依赖上游文本质量，另一头直接影响下游抽取、聚合、检索和生成。

先看几种主流方案的工程权衡：

| 方案 | 准确率 | 速度 | 资源需求 | 典型问题 |
|---|---|---|---|---|
| 规则/启发式 | 低到中 | 快 | 低 | 覆盖有限，迁移能力差 |
| Pipeline 排序/分类模型 | 中 | 中 | 中 | 上游错误累积 |
| 端到端神经模型 | 高 | 慢 | 高 | 训练和部署成本高 |
| SpanBERT 类预训练模型 | 较高 | 较慢 | 高 | 模型重，推理开销大 |

真正上线时，最典型的坑通常有五类。

### 1. 上游错误泄漏

传统 pipeline 先做 mention detection。如果第一步漏掉核心提及，后面就没有可连目标。

例如：

“苹果公司发布了财报。它表示明年继续扩产。”

如果 mention detection 没识别出“苹果公司”，那么“它”即使被判成代词，也无法正确回链。

这个问题在以下文本中特别明显：

- OCR 噪声文本
- ASR 转写文本
- 低资源语言
- 口语化、断裂式表达

### 2. 候选爆炸

如果文档长度为 $T$，枚举所有长度不超过 $L$ 的 span，候选数大致是：
$$
O(TL)
$$
如果不限制长度，最坏可达：
$$
O(T^2)
$$
而提及之间的配对又会进一步膨胀到近似平方级别。

所以长文档会带来两层成本：

- span 数量增长
- span 对数量更快增长

这就是为什么真实系统通常必须做剪枝（pruning），例如：

- 限制最大 span 长度
- 只保留 mention score 最高的一部分 span
- 每个提及只看最近的若干个 antecedent 候选

否则显存和时延都会不可接受。

### 3. 距离启发式经常不够

很多新手会默认“代词回指最近实体”，但这条规则很容易失效。

看两组对比：

例子 A：
“微软收购了一家公司。它随后宣布裁员。”

这里“它”更可能指“微软”。

例子 B：
“微软收购了一家公司。它随后进入破产程序。”

这里“它”更可能指“那家公司”。

两个句子的表面结构几乎一样，但事件语义不同。说明共指判断不仅依赖距离，还依赖：

- 谓词语义
- 施事/受事角色
- 常识约束

### 4. 错误共指比缺失共指更危险

工程上这是最常被低估的一点。

- 没有共指：信息碎片化
- 错误共指：把两个本来不同的实体强行合并

后者的危害更大，因为它会污染整个下游系统。

例如在知识图谱里，错误共指可能导致：

- 两个公司被合并成一个节点
- 两个人的属性混到一起
- 两起不同事件被错误拼接
- 摘要生成输出错误事实

所以很多系统宁可保守一些，也不会盲目追求更高召回。

### 5. 长文档和跨段落现象更难

句内代词通常比跨段落回指简单。真正困难的是这种情况：

“苹果公司在周二发布声明……  
（中间几段介绍供应链和市场背景）  
这家总部位于库比蒂诺的公司还表示……”

这里“这家总部位于库比蒂诺的公司”与前文“苹果公司”之间间隔很远，字符串表面也不同。模型必须同时保留长程上下文和实体语义。

给一个更完整的工程例子。假设你在做跨语言新闻知识图谱，正文里可能出现同一家公司的一组提及：

- “Apple”
- “the company”
- “it”
- “the Cupertino-based firm”

如果共指链没建好，下游常见后果如下：

| 下游模块 | 没有共指时的问题 | 错误共指时的问题 |
|---|---|---|
| 事件抽取 | 同一主体被拆成多个事件节点 | 两个不同主体被强行合并 |
| 情感分析 | 对同一实体的评价被分散 | 情感被记到错误实体上 |
| 实体链接 | 重复请求知识库，增加歧义 | 错链到完全错误的 KB 节点 |
| 摘要生成 | 输出“它”但缺少明确先行词 | 生成事实性错误 |

一个实用结论是：共指模块不是简单后处理，而是事实聚合层。如果它错了，后面所有“统一实体视图”的模块都会一起错。

---

## 替代方案与适用边界

不是所有项目都值得直接上端到端共指模型。选型时，应该先看四个条件：

- 文本类型是否稳定
- 是否有标注数据
- 时延预算是否严格
- 错误共指的业务代价是否很高

不同方案的适用边界如下：

| 方案 | 适用场景 | 资源需求 | 优势 | 劣势 |
|---|---|---|---|---|
| 规则法 | 短文本、固定模板、低资源语言 | 很低 | 快，易解释，易上线 | 泛化差，规则维护成本高 |
| Mention-Ranking | 有标注数据，希望可控建模 | 中 | 结构清晰，便于分析错误 | 候选和特征设计仍较重 |
| SpanBERT 微调 | 新闻、长文档、高精度需求 | 高 | 表示能力强，效果稳定 | 部署重，推理慢 |
| 端到端 span-ranking | 追求整体最优，允许较高训练成本 | 高 | 减少 pipeline 误差传播 | 调参复杂，计算昂贵 |

可以把这些方案理解成“从可控到高性能”的一条谱系。

### 1. 规则法适合先做 baseline

如果你的文本结构比较固定，例如新闻快讯、客服模板、财报摘要，那么高频模式往往有限：

- “他/她”通常回指最近的人名
- “该公司/这家公司”通常回指最近的机构名
- 性别、数、类型不一致就过滤

例如可写出这样一组规则：

| 规则 | 例子 | 作用 |
|---|---|---|
| 类型一致约束 | “他”不回指机构名 | 过滤明显错误候选 |
| 最近优先 | “该公司”优先找最近机构名 | 提升简单样本命中率 |
| 数一致约束 | “他们”不回指单数实体 | 限制候选空间 |
| 标题优先/主语优先 | 新闻首句主语优先作为后文中心实体 | 改善新闻场景效果 |

这种 baseline 的优点是：

- 快
- 成本低
- 容易解释
- 便于定位错误

如果业务里 80% 的样本都属于简单模式，规则法已经能吃掉大部分收益。

### 2. Mention-Ranking 适合做中间路线

当你已经有标注数据，希望比规则法更稳，但又不想一下子上很重的端到端系统时，Mention-Ranking 是很合适的折中。

它的优点是：

- 结构直观，容易分析错误来源
- 候选空间和特征设计可控
- 工程上比超大预训练模型更轻

它适合以下场景：

- 中等规模数据集
- 需要一定可解释性
- 希望先验证共指在业务中的真实收益

### 3. 端到端模型适合文档级实体聚合任务

如果你的下游目标是：

- 事件抽取
- 知识图谱
- 长文档摘要
- 法务/研报/新闻分析

并且文本中存在大量文档级、多次重复回指，那么端到端模型通常更值得投入。它特别适合“先统一实体视图，再做后续推理”的场景。

### 4. 但它的边界仍然要明确

即使最强的端到端模型，也仍然有清晰边界：

| 限制 | 表现形式 |
|---|---|
| 数据依赖 | 没有足够标注时，效果不稳定 |
| 领域迁移 | 新闻训练的模型迁到法务或医疗常常退化 |
| 长度限制 | 超长文档常需截断、滑窗或分块 |
| 常识依赖 | 强世界知识歧义仍可能失败 |
| 成本约束 | 推理时间和显存开销较大 |

因此一个实用工程原则是：

先做弱规则 baseline，再决定是否引入神经模型。

更实际一点的落地路径通常是：

1. 先用规则法解决高频简单样本。
2. 统计错误分布，确认难点是否集中在长距离回指、名词短语变体、语义歧义。
3. 如果这些难点显著影响业务，再上神经模型。
4. 对高风险场景加入置信度阈值，宁可少连，也不要乱连。

很多业务最终采用的是混合方案：

规则法处理高置信模式，神经模型处理复杂样本，低置信输出则交给下游保守使用。

这通常比“全量端到端替代一切”更稳。

---

## 参考资料

下面这些资料覆盖了从经典排序方法、端到端模型、预训练编码器到标准数据集与评测设定的核心脉络。

| 来源 | 重点贡献 | 用途 |
|---|---|---|
| Wiseman, Rush, Shieber. 2015. *Learning Global Features for Coreference Resolution* | 代表性 mention-ranking / 全局特征建模思路 | 理解排序式先行词选择与全局一致性 |
| Lee et al. 2017. *End-to-end Neural Coreference Resolution* | 给出端到端 span-ranking 框架与经典打分形式 | 理解主流公式、Dummy antecedent、训练目标 |
| Joshi et al. 2020. *SpanBERT: Improving Pre-training by Representing and Predicting Spans* | 提供更强的 span 级表示 | 理解为什么更好的 span 表征能提升共指效果 |
| Pradhan et al. 2012. *CoNLL-2012 Shared Task* / OntoNotes | 标准数据集与评测设置 | 理解 benchmark、任务边界与指标定义 |
| Moosavi and Strube. 2016. *Which Coreference Evaluation Metric Do You Trust?* | 讨论 MUC、B³、CEAF 等评测差异 | 理解为什么不同指标看起来会“结论不同” |
| 综述类资料：*Coreference Resolution: Toward End-to-End and Cross-Lingual Systems* | 总结从 pipeline 到端到端、从单语到跨语言的发展 | 建立整体技术地图 |

如果要按阅读顺序安排，最实用的是下面这个路径：

| 阅读顺序 | 建议先看什么 | 原因 |
|---|---|---|
| 第一步 | CoNLL-2012 / OntoNotes 简介 | 先明确任务定义和评测口径 |
| 第二步 | Lee et al. 2017 | 抓住现代端到端共指的核心公式 |
| 第三步 | Wiseman et al. 相关工作 | 理解 mention-ranking 和全局建模思想 |
| 第四步 | SpanBERT | 理解编码器升级为何能提升效果 |
| 第五步 | 指标分析与综述 | 建立更完整的研究与工程视角 |

对初学者来说，最需要优先吃透的其实只有三件事：

1. 共指消解的输出不是标签，而是链。
2. 主流模型核心在于“提及打分 + 先行词打分 + Dummy 选择”。
3. 工程上最重要的问题不是论文分数本身，而是错误共指会不会污染下游系统。
