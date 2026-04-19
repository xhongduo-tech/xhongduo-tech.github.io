## 核心结论

关键词提取是从文档中自动选出一小组最能代表内容的词或短语，并按代表性分数排序。它的目标不是找出所有“看起来重要”的词，而是输出能够帮助检索、打标、聚类、推荐或主题发现的压缩表示。

一篇关于“机器学习在医疗搜索中的应用”的文章，真正有代表性的关键词通常不是“机器”“学习”这种泛词，而更可能是“医疗搜索”“模型”“检索”“诊断”等更能区分主题的词。关键词提取本质上不是词频统计，而是对候选词做代表性排序。

主流方法可以分成四类：

| 方法 | 输入 | 核心思想 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|---|---|
| TF-IDF | 文档集合与当前文档 | 当前文档常见、全语料少见的词更重要 | 简单、快、可解释 | 依赖语料，缺少语义理解 | 批量文档、搜索索引、初筛 |
| TextRank | 单篇文档分词结果 | 词共现形成图，图中中心节点更重要 | 无监督，不依赖训练语料 | 对分词和窗口敏感 | 单文档关键词、长文本 |
| RAKE | 单篇文档与停用词表 | 停用词切分短语，再按词频和共现密度打分 | 轻量，适合短语抽取 | 停用词表质量影响大 | 英文短语、工单、评论 |
| KeyBERT | 文档、候选短语、嵌入模型 | 选出与文档语义向量最接近的候选词 | 语义效果好，支持同义表达 | 依赖模型，成本高 | 内容运营、多语言、语义检索 |

统一地说，关键词提取可以写成：

$$
keywords = top_k(score(c, D))
$$

其中 $D$ 是文档，$c$ 是候选词或候选短语，$score(c, D)$ 表示候选项对文档的代表性分数。

输出示意如下：

```python
text = "机器学习模型可以提升医疗搜索系统的诊断信息检索效果"
keywords = ["医疗搜索", "机器学习模型", "诊断信息", "检索效果"]
```

---

## 问题定义与边界

关键词提取处理的是“单文档或少量文档中的代表性词或短语排序”。输入是一段文本，输出是若干关键词。它不是完整理解文章，也不是自动写摘要，更不是判断文章属于哪个固定类别。

例如客服工单：

```text
打印机无法联网，提示固件版本过旧，升级后仍然连接失败。
```

合理关键词可能是：

```python
["打印机", "无法联网", "固件版本", "升级", "连接失败"]
```

这些词可以用于检索、路由和打标。它不需要输出“用户很着急”这种主观推断，也不需要生成完整处理方案。关键词提取是一种压缩表示：保留最有索引价值的信息，舍弃多数上下文细节。

几个相近任务的边界如下：

| 任务 | 目标 | 输出形式 | 典型方法 |
|---|---|---|---|
| 关键词提取 | 找代表性词或短语 | `["固件版本", "无法联网"]` | TF-IDF、TextRank、RAKE、KeyBERT |
| 文本摘要 | 压缩全文含义 | 一段摘要文本 | Seq2Seq、LLM、抽取式摘要 |
| 主题分类 | 判断预设类别 | `故障工单`、`产品咨询` | 朴素贝叶斯、BERT、分类模型 |
| 实体识别 | 找命名实体 | `打印机型号`、`版本号` | CRF、BiLSTM-CRF、NER 模型 |

后文统一使用这些符号：

| 符号 | 含义 |
|---|---|
| $D$ | 当前文档 |
| $c$ | 候选词或候选短语 |
| $w$ | 单个词 |
| $N$ | 语料库中文档总数 |
| $df(t)$ | 包含词 $t$ 的文档数量 |
| $tf(t, D)$ | 词 $t$ 在文档 $D$ 中的出现强度 |

最小接口形态可以写成：

```python
def extract_keywords(text, topk=5):
    candidates = extract_candidates(text)
    scores = rank_candidates(candidates, text)
    return sorted(scores, key=scores.get, reverse=True)[:topk]
```

这个接口隐藏了很多细节，但边界很清楚：输入文本，输出关键词列表。

---

## 核心机制与推导

四类方法的差异来自它们使用的信号不同。TF-IDF 看统计稀有性，TextRank 看图结构中心性，RAKE 看短语内部结构，KeyBERT 看语义向量相似度。

| 方法 | 信号来源 | 关注的问题 | 代表性解释 |
|---|---|---|---|
| TF-IDF | 统计 | 这个词是不是当前文档的专属词 | 当前文档常见，全语料少见 |
| TextRank | 图 | 这个词是否连接了很多重要词 | 与重要词共现越多越重要 |
| RAKE | 短语 | 这个短语内部是否稳定 | 高频且共现紧密的词组更重要 |
| KeyBERT | 语义 | 这个短语和全文语义是否接近 | 向量空间中越接近越相关 |

TF-IDF 的核心公式是：

$$
score(t, D) = tf(t, D) \cdot idf(t)
$$

$$
idf(t) = \log\frac{1 + N}{1 + df(t)} + 1
$$

其中 $idf(t)$ 表示逆文档频率，白话说就是“这个词在整个语料里有多稀有”。如果一个词在当前文档里出现多，但在其他文档很少出现，它的 TF-IDF 就会高。

玩具例子：语料只有两篇文档。

```text
D1 = "机器 学习 搜索"
D2 = "机器 学习 医疗"
```

对 `D1` 来说，`机器` 出现在两篇文档中，$df=2$；`搜索` 只出现在一篇文档中，$df=1$。当 $N=2$ 时：

$$
idf(机器)=\log\frac{3}{3}+1=1
$$

$$
idf(搜索)=\log\frac{3}{2}+1\approx1.405
$$

所以 `搜索` 的区分度高于 `机器`。这就是 TF-IDF 的基本判断逻辑。

TextRank 把词看成图中的节点，把一定窗口内共同出现的词连成边。图是由节点和边组成的结构，节点表示词，边表示词之间的共现关系。它的迭代公式是：

$$
s(v_i) = (1-d) + d \cdot \sum_{v_j \in In(v_i)} \frac{w_{ji}}{\sum_{v_k \in Out(v_j)}w_{jk}} \cdot s(v_j)
$$

其中 $d$ 是阻尼系数，常用值接近 0.85。白话解释：一个词的重要性来自指向它的其他词；如果那些词本身也重要，它的分数会更高。

TextRank 主流程：

```text
1. 对文档分词，去掉停用词
2. 在固定窗口内统计词共现
3. 构建词图 G=(V,E)
4. 初始化每个词的分数
5. 反复按 TextRank 公式更新分数
6. 输出分数最高的 top-k 词
```

RAKE 的全称是 Rapid Automatic Keyword Extraction。它先用停用词和标点切分候选短语，再根据词频和词的共现度计算分数。停用词是“的、了、是、and、the”这类语法功能强但主题信息弱的词。

RAKE 的词分数通常写成：

$$
score(w)=\frac{deg(w)}{freq(w)}
$$

其中 $freq(w)$ 是词频，$deg(w)$ 是词与其他词共同出现在候选短语中的连接程度。短语得分是词分数求和：

$$
score(p)=\sum_{w \in p} score(w)
$$

KeyBERT 使用预训练嵌入模型。嵌入是把文本转成向量的表示方式，语义相近的文本在向量空间里距离更近。它的核心公式是：

$$
score(c, D)=cos(e(D), e(c))
$$

其中 $e(D)$ 是文档向量，$e(c)$ 是候选短语向量，$cos$ 是余弦相似度。白话解释：候选短语和整篇文档在语义空间里越接近，越可能是关键词。

同一篇文档里，四种方法可能给出不同结果。这不是谁一定错，而是它们看的信号不同：TF-IDF 看“稀有”，TextRank 看“连接”，RAKE 看“短语结构”，KeyBERT 看“语义接近”。

---

## 代码实现

工程实现通常不应该一上来就纠结算法，而是先确定统一流程：

```python
text = "..."
candidates = extract_candidates(text)
scores = rank_candidates(candidates)
keywords = postprocess(scores, topk=10)
```

也就是：分词、候选生成、打分、排序、去重。算法只是 `rank_candidates` 的不同实现。

下面是一段不依赖第三方库的最小 TF-IDF 示例。为了让代码可直接运行，这里使用空格分词。真实中文工程中通常会把 `tokenize` 替换成 `jieba`、`pkuseg` 或业务词典分词。

```python
import math
from collections import Counter, defaultdict

def tokenize(text):
    return text.split()

def build_idf(corpus):
    n = len(corpus)
    df = defaultdict(int)
    for doc in corpus:
        for term in set(tokenize(doc)):
            df[term] += 1
    return {
        term: math.log((1 + n) / (1 + freq)) + 1
        for term, freq in df.items()
    }

def tfidf_keywords(doc, corpus, topk=3):
    idf = build_idf(corpus)
    tf = Counter(tokenize(doc))
    scores = {
        term: count * idf.get(term, 0.0)
        for term, count in tf.items()
    }
    return sorted(scores, key=scores.get, reverse=True)[:topk]

corpus = [
    "机器 学习 搜索",
    "机器 学习 医疗",
    "医疗 搜索 诊断",
]

result = tfidf_keywords("机器 学习 搜索", corpus, topk=2)
assert result == ["搜索", "机器"] or result == ["搜索", "学习"]
assert "搜索" in result
print(result)
```

这段代码体现了一个关键点：`搜索` 比 `机器` 更有区分度，因为它在语料中更少见。

TextRank 的最小伪代码如下：

```text
tokens = tokenize(text)
tokens = remove_stopwords(tokens)
graph = build_cooccurrence_graph(tokens, window_size=4)

scores = {word: 1.0 for word in graph.nodes}
for step in range(max_iter):
    new_scores = {}
    for word in graph.nodes:
        new_scores[word] = (1 - d) + d * sum(
            edge_weight(neighbor, word) / out_weight_sum(neighbor) * scores[neighbor]
            for neighbor in graph.in_neighbors(word)
        )
    scores = new_scores

keywords = top_k(scores)
```

如果使用中文文本，分词质量会直接影响结果。例如：

```python
text = "机器学习模型可以提升医疗搜索系统的诊断信息检索效果"
```

理想候选词应包含 `机器学习模型`、`医疗搜索`、`诊断信息`、`检索效果`。如果分词器把它切成单字或大量泛词，后面的 TF-IDF、TextRank、RAKE 都会受到影响。

依赖和复杂度可以粗略比较如下：

| 方法 | 主要依赖 | 计算成本 | 工程复杂度 | 备注 |
|---|---|---|---|---|
| TF-IDF | 分词器、语料统计 | 低 | 低 | 需要维护语料 IDF |
| TextRank | 分词器、图迭代 | 中 | 中 | 窗口大小影响明显 |
| RAKE | 停用词表、短语切分 | 低 | 低 | 短语质量依赖停用词 |
| KeyBERT | 嵌入模型、候选生成 | 高 | 中到高 | 需要模型加载和向量缓存 |

真实工程例子：企业客服工单自动打标。常见流程是：清洗文本，分词，抽取关键词，把关键词映射到“产品、故障、版本、设备”等标签，再用于工单检索、自动路由和聚类。TF-IDF 可以先做高召回候选，TextRank 或 RAKE 可以处理单篇工单，KeyBERT 适合对语义质量要求更高的内容运营或知识库场景。

---

## 工程权衡与常见坑

中文关键词提取里，分词、停用词表、领域词典通常比算法本身更影响结果。算法决定的是排序方式，预处理决定的是候选集合。如果候选集合里没有正确短语，再好的打分公式也选不出来。

例如“固件版本 1.2.3”如果被切成 `固件 / 版本 / 1.2.3`，结果还勉强可用；如果被切成 `固 / 件 / 版本 / 1 / 2 / 3`，关键词质量会明显下降。再比如停用词表没有去掉“的、了、是”，TextRank 和 RAKE 都可能把这些无主题信息的词排到前面。

常见问题如下：

| 常见坑 | 现象 | 规避手段 |
|---|---|---|
| 分词错误 | 专有名词被切碎 | 加领域词典，保留产品名、版本号、型号 |
| 停用词不全 | “的、问题、情况”排名靠前 | 维护业务停用词表 |
| 短文本不稳定 | 工单、标题结果波动大 | 合并同类文本，保留 1-3 gram 候选 |
| 嵌入模型不匹配 | KeyBERT 语义结果偏离领域 | 使用中文或领域模型，做样本评估 |
| 重复短语过多 | “医疗搜索”“搜索系统”同时出现 | 做去重、包含关系过滤或 MMR |
| TF-IDF 偏稀有词 | 版本号、乱码得分过高 | 加正则过滤、最大/最小文档频率限制 |

候选过滤可以写成一个简单规则：

$$
len(c) \in [1,3],\quad df(c) < threshold
$$

这里的 $len(c)$ 可以理解为候选短语包含的词数，不是字符串长度。这个规则的意思是：候选短语不要太短或太长，并过滤掉在过多文档里都出现的泛词。

一个简单后处理函数如下：

```python
def postprocess(scored_items, topk=5, min_len=2):
    result = []
    seen = set()

    for phrase, score in sorted(scored_items, key=lambda x: x[1], reverse=True):
        phrase = phrase.strip()
        if len(phrase) < min_len:
            continue
        if phrase in seen:
            continue
        if any(phrase in old or old in phrase for old in seen):
            continue
        seen.add(phrase)
        result.append(phrase)
        if len(result) == topk:
            break

    return result

items = [
    ("医疗搜索", 0.92),
    ("搜索", 0.88),
    ("诊断信息", 0.80),
    ("的", 0.70),
]
assert postprocess(items, topk=2) == ["医疗搜索", "诊断信息"]
```

工程排查顺序建议固定：先看分词结果，再看候选短语，再看停用词和过滤规则，最后才调整算法参数或更换模型。很多“算法效果不好”的问题，实际是候选生成阶段已经丢掉了正确答案。

---

## 替代方案与适用边界

没有一种关键词提取方法适合所有场景。选择方法时要看文本长度、语料规模、实时性、语言、是否允许依赖模型，以及结果是否需要强可解释性。

| 场景 | 推荐方法 | 原因 |
|---|---|---|
| 新闻标题批量处理 | TF-IDF | 快速、可批量统计、适合高吞吐 |
| 单篇客服工单 | TextRank 或 RAKE | 不强依赖大语料，能处理单文档 |
| 企业知识库文章 | TF-IDF + KeyBERT | 先召回候选，再做语义排序 |
| 内容运营选题 | KeyBERT | 更关注语义一致性 |
| 搜索索引构建 | TF-IDF | 可解释、可离线预计算 |
| 多语言文档 | KeyBERT | 可选择多语言嵌入模型 |

不同方法的边界如下：

| 方法 | 短文本 | 长文本 | 中文 | 多语言 | 离线 | 实时 |
|---|---|---|---|---|---|---|
| TF-IDF | 一般 | 好 | 依赖分词 | 需要对应分词与语料 | 好 | 好 |
| TextRank | 一般 | 好 | 依赖分词 | 可扩展 | 好 | 中等 |
| RAKE | 好 | 中等 | 依赖停用词和切分 | 好 | 好 | 好 |
| KeyBERT | 好 | 好 | 依赖中文模型 | 好 | 中等 | 成本较高 |

经验上，轻量场景优先统计方法，单文档无监督场景优先 TextRank 或 RAKE，语义要求高时再使用 KeyBERT。更稳妥的工程做法是组合方法：先用 TF-IDF、TextRank 或 RAKE 生成候选，再用 KeyBERT 或业务规则重排。

统一调用接口可以保持不变：

```python
def extract_keywords(text, method="tfidf", topk=10):
    candidates = extract_candidates(text)

    if method == "tfidf":
        scores = rank_by_tfidf(candidates)
    elif method == "textrank":
        scores = rank_by_textrank(text)
    elif method == "rake":
        scores = rank_by_rake(text)
    elif method == "keybert":
        scores = rank_by_embedding_similarity(text, candidates)
    else:
        raise ValueError(f"unknown method: {method}")

    return postprocess(scores, topk=topk)
```

这段接口表达了一个工程原则：不要把业务流程绑死在某一个算法上。候选生成、打分、后处理分开后，后续替换算法、做 A/B 测试、加领域词典都会更容易。

最终选择不是“哪个算法最好”，而是“哪个算法更匹配当前约束”。如果只有几千条工单，且要求毫秒级响应，TF-IDF 或 RAKE 通常更合适；如果是知识库文章推荐，允许离线计算，KeyBERT 或混合方案更值得尝试。

---

## 参考资料

| 类型 | 来源 | 用途 |
|---|---|---|
| 文档 | scikit-learn TF-IDF 文档 | 理解 TF-IDF 平滑公式和实现参数 |
| 论文 | TextRank 论文 | 理解图排序方法来源 |
| 论文 | RAKE 论文 | 理解短语切分和词得分机制 |
| 文档 / 仓库 | KeyBERT 官方资料 | 理解嵌入相似度实现 |

1. [scikit-learn TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)
2. [TextRank: Bringing Order into Text](https://aclanthology.org/W04-3252/)
3. [Automatic Keyword Extraction from Individual Documents](https://www.pnnl.gov/publications/automatic-keyword-extraction-individual-documents)
4. [KeyBERT Documentation](https://maartengr.github.io/KeyBERT/)
5. [KeyBERT GitHub Repository](https://github.com/MaartenGr/KeyBERT)

如果要继续验证某个方法，建议先看官方实现，再看原始论文，最后用自己的语料评估效果。公式只是理论入口，关键词提取的工程效果更依赖数据质量、分词、候选生成和后处理规则。
