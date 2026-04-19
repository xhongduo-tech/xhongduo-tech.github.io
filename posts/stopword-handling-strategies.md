## 核心结论

停用词处理策略 = 对高频低信息词进行删除、保留或降权的规则集合。

停用词不是一批“必须删除的词”。更准确的说法是：停用词处理要根据任务、特征表示和模型能力决定。对词袋模型，删除一部分停用词可以减少噪声；对 TF-IDF，很多高频词会被自动降权；对 BERT 这类上下文模型，通常应先保留原文，让模型自己学习词在上下文里的作用。

核心判断只有一个：这个词是否对当前任务的判别有贡献。

玩具例子：“可以退款”和“不能退款”只差一个“不”。如果把“不”当停用词删除，两句话会变成接近的表示，模型容易把拒绝退款误判成允许退款。这里“不”虽然高频，但不是低信息词。

| 任务类型 | 推荐策略 | 原因 |
|---|---|---|
| 词袋分类 | 可删除部分停用词 | 高频功能词会稀释有效特征 |
| 情感分析 | 保留否定词、程度词 | “不”“很”“太”会改变情绪方向或强度 |
| 搜索召回 | 优先降权，谨慎删除 | 过度删除可能损失查询意图 |
| 深度上下文模型 | 通常保留原文 | 模型能利用上下文判断词的贡献 |

---

## 问题定义与边界

停用词是高频但语义贡献低的词，例如“的、是、在、了”。这里的“语义贡献低”不是指它永远没用，而是指它在某个任务中通常不能帮助区分类别、主题或相关性。

噪声词是对当前任务没有帮助、甚至干扰模型的词。噪声词可能是停用词，也可能是乱码、模板词、HTML 残留、日志固定前缀。

领域关键词是在特定业务里有强判别作用的词。它可能很高频，但不能因为高频就删除。真实工程例子：在医疗文本分类中，“阳性”“阴性”可能频繁出现，但它们直接影响诊断含义，不能作为停用词处理。

| 概念 | 含义 | 是否可直接删除 |
|---|---|---|
| 停用词 | 高频、低区分度的功能词 | 不一定 |
| 噪声词 | 对当前任务无帮助的词 | 通常可处理 |
| 领域关键词 | 业务中决定含义的高频词 | 不能删 |
| 否定词 | 改变语义方向的词 | 通常不能删 |

判断边界可以按三个问题走：

1. 这个词是否会改变句子语义？
2. 这个词是否在当前任务中携带判别信息？
3. 这个词是否属于领域术语，而不是通用功能词？

中文里“的、了、在”常常更接近停用词；“不、没、无、未”虽然短且高频，但在客服意图、风控、情感分析中经常是关键信号。

---

## 核心机制与推导

停用词处理主要有三条路线：显式过滤、自动降权、上下文建模。

显式过滤是直接从分词结果中删除词。它简单、可解释，但依赖停用词表质量。自动降权是不删除词，而是在特征权重里降低高频词影响。上下文建模是保留原句，让模型根据上下文学习词的作用。

TF-IDF 是最常见的自动降权方法。TF 是词频，表示词在当前文档中出现得多不多；IDF 是逆文档频率，表示词在整个语料中稀不稀有。

$$
tfidf(t,d)=tf(t,d)\times idf(t)
$$

$$
idf(t)=\log\frac{N}{df(t)}
$$

其中 $N$ 是文档总数，$df(t)$ 是包含词 $t$ 的文档数量。

最小数值例子：语料共 3 篇文档。“的”出现在 3 篇里，$df=3$，所以 $idf=\log(3/3)=0$。“芯片”只出现在 1 篇里，$df=1$，所以 $idf=\log(3/1)\approx1.099$。

如果某篇文档中“的”的词频是 0.2，则 $tfidf=0.2\times0=0$；“芯片”的词频是 0.1，则 $tfidf\approx0.1\times1.099=0.110$。这说明：不一定要手动删除所有高频词，权重机制本身已经能压低一部分通用词。

| 策略 | 机制 | 代表方法 | 适合什么 |
|---|---|---|---|
| 显式过滤 | 直接删词 | 停用词表 | 词袋、传统分类 |
| 自动降权 | 降低高频词影响 | TF-IDF、BM25、`max_df` | 稀疏特征、检索 |
| 上下文建模 | 保留词，由模型学习 | BERT、RoBERTa | 深度语义任务 |

流程可以写成：

原文 → 分词 → 统计 `tf/df` → 计算 `idf` → 得到 `tfidf` → 决定删、留或降权。

---

## 代码实现

代码实现的重点不是写一个很长的停用词表，而是让策略可配置、可复现、可验证。至少要把输入文本、分词器、规范化器、停用词集合、保留词集合、过滤逻辑、向量化器和评估逻辑串起来。

```python
import math
from collections import Counter

def tokenize(text):
    return text.split()

def filter_tokens(tokens, stopwords, keep_words):
    return [t for t in tokens if t in keep_words or t not in stopwords]

def tf(tokens):
    total = len(tokens)
    counts = Counter(tokens)
    return {word: count / total for word, count in counts.items()}

def idf(corpus_tokens):
    n = len(corpus_tokens)
    vocab = set(word for doc in corpus_tokens for word in doc)
    result = {}
    for word in vocab:
        df = sum(1 for doc in corpus_tokens if word in doc)
        result[word] = math.log(n / df)
    return result

def tfidf(tokens, idf_map):
    tf_map = tf(tokens)
    return {word: tf_map[word] * idf_map[word] for word in tf_map}

stopwords = {"的", "了", "在", "是"}
keep_words = {"不", "没", "无", "未"}

docs = [
    "商品 是 可以 退款 的",
    "商品 是 不 可以 退款 的",
    "芯片 在 设备 中 工作"
]

filtered_docs = [
    filter_tokens(tokenize(doc), stopwords, keep_words)
    for doc in docs
]

idf_map = idf(filtered_docs)
vec0 = tfidf(filtered_docs[0], idf_map)
vec1 = tfidf(filtered_docs[1], idf_map)

assert "的" not in filtered_docs[0]
assert "不" in filtered_docs[1]
assert vec1["不"] > 0
assert "芯片" in idf_map
```

| 配置项 | 含义 | 示例 |
|---|---|---|
| `stopwords` | 通用停用词表 | `{"的", "了", "在"}` |
| `keep_words` | 必须保留词 | `{"不", "没", "无"}` |
| `max_df` | 过滤超高频词 | `0.8` |
| `tokenizer` | 分词器 | 中文分词器、空格分词器 |

真实工程例子：客服意图分类中，训练集里经常出现“我要、这个、一下、请问”。这些词对区分“退款”“换货”“催发货”帮助有限，可以删除或降权。但“不退款”“没收到”“无法登录”里的“不、没、无法”必须保留，否则会破坏意图。

训练和推理必须使用同一套分词与规范化逻辑。如果训练时按词切分，推理时按字符切分，停用词匹配会失效，线上特征分布也会偏离训练分布。

---

## 工程权衡与常见坑

停用词策略本质上是在召回、精度和语义保真度之间做权衡。

删除停用词通常能减少特征维度、降低噪声、加快训练。但删除也会损失上下文，尤其是短文本任务。短文本中每个词都可能影响判断，“不喜欢”和“喜欢”只差一个字，删除错误会造成语义反转。

| 问题 | 后果 | 规避方式 |
|---|---|---|
| 直接套通用英文停用词表 | 误删领域词或不适配中文 | 先人工抽样检查 |
| 把否定词当停用词 | 语义反转 | 单独维护 `keep_words` |
| 训练和推理分词不一致 | 停用词匹配失败 | 统一预处理流水线 |
| 所有高频词都删掉 | 召回下降 | 优先降权，不要一刀切 |
| 在上下文模型里硬删词 | 破坏语义结构 | 先做消融实验 |

处理建议的优先级：

| 优先级 | 策略 | 说明 |
|---|---|---|
| 1 | 先保留原文 | 建立不删词基线 |
| 2 | 再尝试降权 | 用 TF-IDF、BM25 或 `max_df` 控制高频词 |
| 3 | 最后显式删除 | 只删除确认无用的词 |

在搜索场景中，过度删除高频词可能让查询变窄。比如用户搜索“在北京可以办理居住证吗”，如果删除过多功能词，只剩“北京 办理 居住证”，系统可能还能召回结果；但如果任务需要识别“可以吗”这种咨询意图，删掉语气和助词会影响下游分类。

---

## 替代方案与适用边界

停用词表只是方案之一，不是默认最优方案。模型越依赖人工特征，停用词策略越重要；模型越擅长上下文建模，越应该谨慎删除原文信息。

如果使用 TF-IDF + 线性分类器，可以删除一部分稳定无用的功能词，同时保留否定词、程度词和领域词。如果使用 BERT，通常先不要删词，直接输入完整文本，再通过实验判断是否需要裁剪。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 停用词表过滤 | 简单直接、可解释 | 容易误删 | 词袋、传统分类 |
| TF-IDF/BM25 降权 | 自动降低高频词影响 | 仍依赖特征工程 | 检索、稀疏向量 |
| `max_df` 规则 | 实现简单 | 只能处理极高频词 | 快速基线 |
| 上下文模型 | 语义保真度高 | 计算成本高 | BERT 类任务 |

选型规则：

| 条件 | 推荐选择 |
|---|---|
| 任务需要可解释性 | 显式过滤或 TF-IDF 降权 |
| 任务依赖上下文 | 保留原文 |
| 数据量小、要快速基线 | `max_df` + 小停用词表 |
| 领域强约束 | 先构建领域词表，再决定删留 |
| 情感、风控、客服意图 | 保留否定词和程度词 |

“可以退款”和“不能退款”在显式过滤下容易混淆，因为“不”可能被误删；在上下文模型里，模型更容易利用“不”和“退款”的组合关系判断语义。因此，停用词策略不是独立步骤，而是文本表示方案的一部分。

---

## 参考资料

1. [scikit-learn: Working With Text Data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
2. [scikit-learn: TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)
3. [scikit-learn: TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
4. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://research.google/pubs/bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding/)
5. [An Evaluation of Stop Word Lists in Text Retrieval](https://dl.acm.org/doi/10.1145/1599081.1599086)
