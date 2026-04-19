## 核心结论

抽取式文本摘要是一种“从原文中选择关键句子”的摘要方法。它不生成新句子，不改写原文，只把原文里最重要的若干句挑出来，按一定顺序组成短摘要。

新手版理解：把一篇长文章想成一堆句子，系统先给每句打分，再挑出分数最高的几句拼成摘要。

对比版理解：抽取式摘要像“从原文里圈重点”，改写式摘要像“用自己的话重写重点”。

抽取式摘要的核心目标是：在有限长度内保留最重要、最可靠、最容易回溯的原文信息。它牺牲了一部分语言连贯性，换来更强的事实稳定性。因为输出句子直接来自原文，所以它通常不会凭空编造原文没有的信息。

| 类型 | 是否生成新句子 | 事实稳定性 | 语言自然度 |
|---|---:|---:|---:|
| 抽取式摘要 | 否 | 高 | 一般 |
| 改写式摘要 | 是 | 取决于模型 | 更自然 |

这里的“事实稳定性”指摘要内容是否容易保持原文事实，不额外引入错误信息。“语言自然度”指摘要读起来是否像人工重新写过的一段话。

抽取式摘要适合事实优先的场景，例如新闻要点、客服工单、会议纪要、法律材料、检索结果摘要。它不适合要求强改写、强概括、强叙事的场景，例如营销文案、标题生成、口语化总结。

---

## 问题定义与边界

抽取式文本摘要的输入是一篇或多篇文档，输出是若干原文句子组成的短摘要。这里的“文档”可以是文章、新闻、对话、工单、报告、论文段落或检索结果。

形式化地看，原文可以表示为句子序列：

$$D = [s_1, s_2, ..., s_N]$$

抽取式摘要要从中选出一个子集：

$$Y = [s_{i_1}, s_{i_2}, ..., s_{i_k}], \quad Y \subseteq D$$

其中 $k$ 是摘要句子数量，或者由字数预算、token 预算、阅读时长预算决定。

新手版例子：一篇 20 句的工单记录，系统只返回其中 2 到 3 句最关键的原文句子，不新增解释。

边界版例子：如果原文没写“故障原因”，抽取式摘要不能凭空补出原因，只能选出最接近原因描述的句子。

| 项目 | 说明 |
|---|---|
| 输入 | 文档、对话、工单、新闻等 |
| 输出 | 原文句子子集 |
| 约束 | 不生成新句子 |
| 目标 | 保留主题、关键信息、可追溯性 |

抽取式摘要只负责筛选，不负责改写、纠错、补全推理或重新组织事实。这个边界很重要。比如原文中有一句“用户反馈接口仍然超时”，抽取式摘要可以选择这句话；但如果原文没有说明“数据库连接池耗尽”，摘要系统不能自己推断并输出这个原因。

玩具例子：

原文有 4 句：

1. 今天系统发布了一个新版本。
2. 发布后部分用户反馈登录接口超时。
3. 工程师回滚了认证服务配置。
4. 晚上天气很好。

如果只选 2 句，合理摘要可能是：

- 发布后部分用户反馈登录接口超时。
- 工程师回滚了认证服务配置。

第 4 句虽然是原文句子，但和主题无关，不应该被选中。

---

## 核心机制与推导

抽取式摘要通常可以分成两条主线：图排序方法和神经抽取方法。

图排序方法的代表是 TextRank 和 LexRank。它们把句子看成图上的节点，把句子之间的相似度看成边，然后通过迭代传播重要性。简单说，一个句子如果和很多重要句子都相关，它自己也会变得重要。

神经抽取方法把每个句子编码成向量，再用分类器判断这句话是否应该进入摘要。这里的“编码器”是把文本转换成数字表示的模型，“分类器”是根据数字表示输出概率的模型。

新手版理解：先把每个句子当成候选项；和其他句子更相关、信息更集中、覆盖主题更强的句子，得分更高。

机制版理解：

| 方法 | 核心思想 | 优点 | 局限 |
|---|---|---|---|
| TextRank / LexRank | 句子图排序 | 简单、可解释 | 依赖相似度质量 |
| 神经抽取器 | 句子分类/打分 | 表达能力更强 | 依赖训练数据 |

先看图方法。设第 $i$ 个句子是 $s_i$，句子 $s_i$ 和 $s_j$ 的相似度是 $w_{ij}$。相似度可以来自 TF-IDF、词向量、句向量或其他文本表示。这里的“TF-IDF”是一种把词语重要性转成数值向量的传统方法，常用于衡量句子之间的词汇重合和主题接近程度。

TextRank 的核心迭代公式可以写成：

```text
r_i^(t+1) = (1-d)/N + d * Σ_{j≠i} [ w_ji / Σ_{k≠j} w_jk ] * r_j^(t)
```

其中：

| 符号 | 含义 |
|---|---|
| $r_i$ | 第 $i$ 个句子的重要性分数 |
| $N$ | 句子总数 |
| $d$ | 阻尼系数，常见取值接近 0.85 |
| $w_{ji}$ | 句子 $j$ 指向句子 $i$ 的相似度权重 |
| $t$ | 当前迭代轮次 |

推导顺序是：

1. 先把句子表示为 $s_i$。
2. 再计算句子之间的连接权重 $w_{ij}$。
3. 初始化每个句子的分数 $r_i$。
4. 通过公式反复更新 $r_i$。
5. 按 $r_i$ 排序，选择 Top-k 句子。

一个最小数值例子：设 3 句中，$s_1$ 与 $s_2$ 相似度是 0.8，$s_1$ 与 $s_3$ 相似度是 0.2，$s_2$ 与 $s_3$ 相似度是 0.4。取 $d=0.85$，初始 $r_1=r_2=r_3=1$，一轮更新后大致得到：

| 句子 | 分数 |
|---|---:|
| $s_1$ | 0.90 |
| $s_2$ | 1.30 |
| $s_3$ | 0.50 |

排名结果是 $s_2 > s_1 > s_3$。如果摘要预算是 2 句，就选 $s_2$ 和 $s_1$。

再看神经抽取方法。常见形式是：

```text
h_i = Encoder(s_i)
p_i = σ(W h_i + b)
select(s_i) = 1 if p_i ≥ τ else 0
```

其中 $h_i$ 是句子向量，$p_i$ 是句子被选入摘要的概率，$\sigma$ 是 sigmoid 函数，作用是把分数压到 0 到 1 之间，$\tau$ 是选择阈值。

神经方法的推导顺序是：

1. 输入句子 $s_i$。
2. 编码器输出句子表示 $h_i$。
3. 分类头输出概率 $p_i$。
4. 根据阈值 $\tau$ 或 Top-k 策略决定是否选句。
5. 把选出的句子恢复到原文顺序。

---

## 代码实现

最小伪代码如下：

```js
sentences = splitIntoSentences(text)
scores = scoreSentences(sentences)

selected = []
for sentence in sentencesSortedByScore:
  if not tooSimilar(sentence, selected):
    selected.push(sentence)
  if selected.length === k:
    break

summary = restoreOriginalOrder(selected)
```

代码模块可以拆成五部分：

| 模块 | 作用 |
|---|---|
| 分句 | 把文本切成候选句 |
| 编码 | 将句子转成向量 |
| 打分 | 计算重要性或概率 |
| 去重 | 避免重复句子 |
| 排序 | 保持摘要可读性 |

下面是一个最小可运行 Python 版本。它不是完整工业级 TextRank，但包含抽取式摘要落地所需的核心步骤：句子切分、相似度计算、图构建、迭代更新、排序与去冗余。

```python
import re
import math
from collections import Counter

def split_sentences(text):
    parts = re.split(r"[。！？.!?]\s*", text.strip())
    return [p.strip() for p in parts if p.strip()]

def tokenize(sentence):
    return re.findall(r"[\w\u4e00-\u9fff]+", sentence.lower())

def vectorize(sentence):
    return Counter(tokenize(sentence))

def cosine(a, b):
    va, vb = vectorize(a), vectorize(b)
    keys = set(va) | set(vb)
    dot = sum(va[k] * vb[k] for k in keys)
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def textrank_summarize(text, k=2, max_iter=30, d=0.85, redundancy_threshold=0.92):
    sentences = split_sentences(text)
    n = len(sentences)
    if n <= k:
        return sentences

    weights = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                weights[i][j] = cosine(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(max_iter):
        new_scores = [(1 - d) / n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                out_sum = sum(weights[j])
                if out_sum > 0:
                    new_scores[i] += d * (weights[j][i] / out_sum) * scores[j]
        scores = new_scores

    ranked = sorted(range(n), key=lambda idx: scores[idx], reverse=True)

    selected = []
    for idx in ranked:
        candidate = sentences[idx]
        if all(cosine(candidate, sentences[j]) < redundancy_threshold for j in selected):
            selected.append(idx)
        if len(selected) == k:
            break

    return [sentences[i] for i in sorted(selected)]

text = "今天系统发布了一个新版本。发布后部分用户反馈登录接口超时。工程师回滚了认证服务配置。晚上天气很好。"
summary = textrank_summarize(text, k=2)

assert len(summary) == 2
assert any("登录接口超时" in s for s in summary)
assert all(s in split_sentences(text) for s in summary)

print(summary)
```

这段代码有两个关键点。

第一，输出必须来自原文。`assert all(s in split_sentences(text) for s in summary)` 检查了这一点。

第二，最终摘要恢复了原文顺序。即使排序阶段先选中了第 3 句，再选中第 2 句，输出时也应该按原文顺序展示，否则摘要会读起来混乱。

真实工程例子：客服工单摘要。系统输入一长串用户对话、排障记录、人工备注，输出 2 到 3 句原文摘要。这样做适合人工接手、质检复盘和后续工单检索，因为每一句都能回溯到原始记录。工程实现中通常还会保留句子来源位置、说话人、时间戳和工单阶段，避免摘要脱离上下文。

---

## 工程权衡与常见坑

工程上最常见的问题不是“模型不会跑”，而是“选出来的句子不好用”。抽取式摘要只做选句，所以它高度依赖原文质量、分句质量、相似度质量和后处理策略。

| 问题 | 表现 | 规避方式 |
|---|---|---|
| 重复句 | 同义句被重复选中 | MMR、冗余惩罚 |
| 断指代 | “它/这个/该问题”缺少前文 | 保留句序、检查指代 |
| 固定阈值失效 | 不同数据集效果差 | 验证集调参 |
| 长句偏置 | 长句得分过高 | 长度归一化 |
| 相似度漂移 | 图排序不稳定 | 更稳的句向量或多特征融合 |

“MMR”是 Maximal Marginal Relevance，中文可理解为最大边际相关性。它的目标是在“重要”和“不重复”之间做平衡。新手版例子：两句意思几乎一样，模型都选出来了，结果摘要变啰嗦。这不是模型太笨，而是缺少去重机制。

必要伪代码片段如下：

```js
if (similarity(candidate, selected) < threshold) {
  keep(candidate)
}
```

这个逻辑表示：候选句和已选句太像，就不再选它。实际工程中，`similarity(candidate, selected)` 通常会取候选句与所有已选句的最大相似度。

断指代是另一个常见坑。客服工单里“它坏了”“这个问题还在”单独抽出来会失去指代对象。工程上可以采用三种方式缓解：保留原句顺序，给候选句加入前一句上下文，或者检测句首是否含有“它、这个、上述、该问题”等指代词。

固定阈值也容易失效。神经抽取器里常见写法是 $p_i \ge \tau$ 就选句，但 $\tau=0.5$ 不一定适合所有数据集。新闻摘要、客服工单、法律材料的句子密度不同，阈值需要在验证集上调整。

长句偏置也很常见。长句包含更多词，可能和更多句子相似，因此在图排序里得分偏高。但长句不一定更适合摘要，可能只是信息杂。解决方式包括长度归一化、限制最大句长、加入位置特征或人工规则。

相似度漂移指的是相似度定义不稳定导致排名不可靠。比如只用词面重合时，“登录失败”和“认证接口超时”可能相似度不高，但在业务上它们可能高度相关。真实系统中通常会使用更稳的句向量，或者融合关键词、位置、标题匹配、业务字段等多种特征。

---

## 替代方案与适用边界

抽取式摘要适合“事实优先、可追溯、低风险”的场景，不适合要求语言高度自然、强重写能力的场景。

新手版判断：新闻快讯、工单复盘、法律/合规材料更适合抽取式；营销文案、摘要标题、通顺叙述更适合改写式。

对比版判断：如果你要“保真”，选抽取式；如果你要“更像人写的短文”，选改写式。

| 方案 | 适用场景 | 优点 | 不足 |
|---|---|---|---|
| 抽取式摘要 | 事实记录、工单、法务、检索 | 稳、可回溯 | 不够自然 |
| 改写式摘要 | 阅读摘要、内容生成 | 更流畅 | 可能幻觉 |
| 规则/启发式摘要 | 小规模、强约束场景 | 简单可控 | 泛化差 |

改写式摘要也叫生成式摘要。它会生成原文中不一定存在的新句子，因此更灵活，但也更可能产生“幻觉”。这里的“幻觉”指模型输出看似合理、但原文没有支持的信息。

规则/启发式摘要适合强约束场景。例如工单系统中，只抽取包含“故障原因”“处理方案”“恢复时间”的句子。这种方法简单可控，但一旦文本格式变化，规则就容易失效。

抽取式摘要也有明确边界。

第一，当原文句子本身质量低时，抽取式摘要也会输出低质量句子集合。它不会自动把坏句子改成好句子。

第二，当任务需要跨句整合、概括推理或压缩重复信息时，抽取式方法往往不够。比如原文分三处描述“北京、上海、广州三个地区都出现延迟”，抽取式摘要只能选择某些原句，不能自然合成“一线城市多地出现延迟”。

第三，当摘要需要统一口径、统一风格或面向外部发布时，抽取式摘要通常只是中间结果。更完整的系统可能先抽取关键证据句，再交给人工或生成模型做受控改写。

---

## 参考资料

TextRank 和 LexRank 属于图排序路线；SummaRuNNer、联合打分与选择、预训练编码器属于神经抽取路线。想先理解经典图方法，可以先读 TextRank；想看神经方法，可以读 SummaRuNNer 和后续预训练编码器工作。

| 文献 | 方向 | 作用 |
|---|---|---|
| Mihalcea & Tarau, 2004, *TextRank* | 图排序 | 经典 PageRank-style 抽取摘要 |
| Erkan & Radev, 2004, *LexRank* | 图排序 | 基于词汇中心性的句子重要性估计 |
| Nallapati et al., 2017, *SummaRuNNer* | 神经抽取 | RNN 文档级抽取模型 |
| Zhou et al., 2018, *Jointly Learning to Score and Select Sentences* | 神经抽取 | 句子打分与选择联合学习 |
| Liu & Lapata, 2019, *Text Summarization with Pretrained Encoders* | 预训练抽取 | 预训练编码器提升抽取效果 |

1. [TextRank: Bringing Order into Text](https://aclanthology.org/W04-3252/)
2. [LexRank: Graph-based Lexical Centrality as Salience in Text Summarization](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html)
3. [SummaRuNNer: A Recurrent Neural Network Based Sequence Model for Extractive Summarization of Documents](https://research.ibm.com/publications/summarunner-a-recurrent-neural-network-based-sequence-model-for-extractive-summarization-of-documents)
4. [Neural Document Summarization by Jointly Learning to Score and Select Sentences](https://aclanthology.org/P18-1061/)
5. [Text Summarization with Pretrained Encoders](https://aclanthology.org/D19-1387/)
