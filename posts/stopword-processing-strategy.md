## 核心定义

停用词处理策略，是指在文本特征化或建模前，决定哪些高频低信息词要删除、保留或降权的一组规则。

---

## 直观解释

传统做法是直接维护停用词表，把“的、是、在、and、the”这类词过滤掉。

TF-IDF 不一定要显式删词，它会自动压低“几乎到处都出现”的词的权重。

BERT 这类上下文模型不依赖词表删除策略，而是把停用词放回上下文里一起建模；在情感分析里，“不”“没”这类词往往不能删。

---

## 关键公式/机制

统一记号：

- 语料文档数：$N$
- 词项：$t$
- 文档：$d$
- 词频：$tf(t,d)$
- 文档频次：$df(t)$

TF-IDF 的核心是：

$$
tfidf(t,d)=tf(t,d)\times idf(t)
$$

$$
idf(t)=\log\frac{N}{df(t)}
$$

机制对应三类策略：

| 策略 | 处理方式 | 适用场景 |
|---|---|---|
| 显式过滤 | 用停用词表直接删除 | 词袋、检索、传统分类 |
| 自动降权 | 用 TF-IDF、BM25、`max_df` 降低高频词影响 | 稀疏向量特征 |
| 上下文建模 | 保留原文，由预训练模型学习词的作用 | BERT、RoBERTa 等 |

---

## 最小数值例子

设语料有 3 篇文档，词“的”出现在 3 篇里，词“芯片”只出现在 1 篇里。

- “的”：$df=3$，所以 $idf=\log(3/3)=0$
- “芯片”：$df=1$，所以 $idf=\log(3/1)\approx 1.099$

若某篇文档里：

- “的”出现 2 次，总词数 10，$tf=0.2$，$tfidf=0.2\times 0=0$
- “芯片”出现 1 次，总词数 10，$tf=0.1$，$tfidf=0.1\times 1.099\approx 0.110$

结论很直接：高频词不一定被删掉，但会被压到几乎没有权重。

---

## 真实工程场景

中文客服意图分类里，常见做法是：

- 词袋/TF-IDF 阶段保留“不是、不、没、未”这类否定词
- 过滤“的、了、在、呢”这类纯功能词
- 对高频业务词用 `max_df` 或领域停用词表单独处理

原因很明确：如果把“没”“不”删掉，“可以退款”与“不能退款”会被错误压成接近的表示。

---

## 常见坑与规避

| 坑 | 后果 | 规避 |
|---|---|---|
| 直接套通用英文停用词表 | 误删领域关键词 | 先做小样本人工审查 |
| 把“否定词”当普通停用词 | 情感、意图误判 | 明确保留“不、没、无” |
| 训练和推理分词不一致 | 停用词匹配失败 | 统一分词器和规范化流程 |
| 把所有高频词都删掉 | 召回和分类都掉点 | 优先用 `max_df` 或降权，不要一刀切 |
| 在 BERT 上强行删词 | 破坏上下文信息 | 先保留原文，再做任务级消融实验 |

---

## 参考来源

1. [scikit-learn feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html)
2. [scikit-learn TfidfTransformer docs](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)
3. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/)
4. [Stop Word Lists in Free Open-source Software Packages](https://aclanthology.org/W18-2502/)
5. [On Stopwords, Filtering and Data Sparsity for Sentiment Analysis of Twitter](https://aclanthology.org/L14-1265/)

