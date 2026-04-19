## 核心结论

文本相似度是把两段文本映射成一个标量，用来衡量它们在词面、字符面或语义面上的接近程度。给定两段文本 $x, y$，相似度通常记为 $s(x,y)$，分数越高表示越相似；距离通常记为 $d(x,y)$，距离越小表示越相似。

不同算法衡量的不是同一种“相似”。

| 方法 | 输入表示 | 衡量对象 | 优点 | 局限 |
|---|---|---|---|---|
| Jaccard | 词集合 | 词是否重叠 | 简单、可解释、适合去重 | 忽略词频、顺序和语义 |
| Cosine | 词袋向量 / TF-IDF 向量 | 向量方向是否接近 | 适合检索和粗召回 | 依赖分词和特征设计 |
| Edit distance | 字符序列 / 词序列 | 修改成本 | 适合拼写纠错、近重复检测 | 不理解语义 |
| WMD | 词向量分布 | 词语语义搬运成本 | 能利用词向量语义 | 计算重，依赖词向量覆盖 |
| SBERT | 句向量 | 整句语义接近程度 | 适合语义匹配和重排 | 需要模型，阈值要校准 |

玩具例子：`"我 喜欢 苹果"` 和 `"我 喜欢 香蕉"` 只差一个词，因此词面相似度较高。从语义上看，两句都表达“喜欢某种水果”，相似度也不低，但具体数值取决于方法。`Jaccard` 只看词集合重叠，`SBERT` 会把整句编码成句向量后再比较语义接近程度。

---

## 问题定义与边界

文本相似度不是“是否同义”的唯一判定，而是一个连续分值。连续分值指结果不是简单的 0 或 1，而是类似 $0.73$ 这样的程度判断。它常用于排序、召回、去重、聚类和检索。

形式化定义如下：

$$
x, y \in \mathcal{T}
$$

其中 $x, y$ 是两段输入文本，$\mathcal{T}$ 表示文本空间。算法输出可以是相似度：

$$
s(x,y) \in [0,1]
$$

也可以是距离：

$$
d(x,y) \ge 0
$$

相似度越大越接近，距离越小越接近。两者可以互相转换，但转换方式依赖具体算法。例如可以用 $s = 1 / (1 + d)$ 把非负距离压到 $(0,1]$ 区间。

使用文本相似度前，需要先确定四个边界。

| 边界项 | 需要明确的问题 | 示例 |
|---|---|---|
| 任务场景 | 是去重、检索、纠错、聚类还是排序 | FAQ 去重与搜索排序目标不同 |
| 比较粒度 | 按字符、按词、按句子还是按文档 | 中文可按字，也可先分词 |
| 文本长度 | 短文本和长文本是否使用同一指标 | 问句与长文章摘要不应直接套同一阈值 |
| 相似类型 | 看词面、字符变化还是语义 | 抄袭检测和意图匹配关注点不同 |

真实工程例子：FAQ 去重时，`"怎么修改密码"` 和 `"如何重置登录密码"` 词面不完全相同，但意图一致，可以看作高相似。抄袭检测中，`"文本相似度用于衡量文本接近程度"` 和 `"文本相似度可用于判断文本是否接近"` 可能需要更严格的字符级或词级比较，因为任务关注的是表达复制和改写痕迹，而不只是语义一致。

中文场景还要特别明确“字符级”和“词级”。`"研究生命起源"` 如果按字符看，是 `研 / 究 / 生 / 命 / 起 / 源`；如果按词看，可能是 `研究 / 生命 / 起源`。粒度不同，集合、向量和编辑距离都会变化。

---

## 核心机制与推导

文本相似度的计算路径可以拆成两步：先选择表示方式，再选择相似度函数。表示方式是把文本变成算法能处理的结构，例如集合、向量、编辑序列或句向量；相似度函数是在这些结构上计算分数或距离。

表示层级演进图：

```text
原始文本
  -> 集合表示
  -> 词袋向量 / TF-IDF 向量
  -> 字符或词级编辑序列
  -> 词向量分布搬运
  -> 句向量余弦相似度
```

公式总览：

| 方法 | 公式 | 说明 |
|---|---|---|
| Jaccard | $J(x,y)=\frac{|S_x \cap S_y|}{|S_x \cup S_y|}$ | $S_x,S_y$ 是分词后的集合 |
| Cosine | $\cos(x,y)=\frac{v_x \cdot v_y}{\|v_x\|\|v_y\|}$ | $v_x,v_y$ 是词袋或 TF-IDF 向量 |
| Edit distance | $d_{edit}(x,y)=\min \sum c(op_i)$ | 通过插入、删除、替换把 $x$ 变成 $y$ 的最小代价 |
| WMD | $d_{wmd}(x,y)=\min_T \sum_{i,j}T_{ij}\|w_i-w_j\|$ | $T$ 是词质量运输矩阵，$w_i,w_j$ 是词向量 |
| SBERT | $sim(x,y)=\cos(f_\theta(x),f_\theta(y))$ | $f_\theta$ 是句子编码模型 |

从玩具例子开始：

```text
x = "我 喜欢 苹果"
y = "我 喜欢 香蕉"
Sx = {我, 喜欢, 苹果}
Sy = {我, 喜欢, 香蕉}
```

交集是 `{我, 喜欢}`，并集是 `{我, 喜欢, 苹果, 香蕉}`，所以：

$$
Jaccard = \frac{2}{4}=0.5
$$

如果构造二值词袋向量，词表为 `[我, 喜欢, 苹果, 香蕉]`：

```text
vx = [1, 1, 1, 0]
vy = [1, 1, 0, 1]
```

点积为 $2$，两个向量的模长都是 $\sqrt{3}$，因此：

$$
cosine = \frac{2}{\sqrt{3}\sqrt{3}}=\frac{2}{3}\approx0.667
$$

如果按词级编辑距离计算，`我` 和 `喜欢` 不变，只需要把 `苹果` 替换成 `香蕉`，所以：

$$
d_{edit}=1
$$

WMD，Word Mover's Distance，中文可理解为“词移动距离”。它先把每个词映射到词向量空间，再计算一段文本的词分布搬到另一段文本的词分布需要付出的最小成本。运输矩阵 $T_{ij}$ 表示从第 $i$ 个词向第 $j$ 个词搬运多少“词质量”。$\|w_i-w_j\|$ 表示两个词向量之间的距离。如果 `苹果` 和 `香蕉` 在词向量空间里很近，那么替换它们的语义成本会低于把 `苹果` 搬到 `汽车`。

SBERT，Sentence-BERT，中文可理解为“把整个句子编码成向量的 BERT 变体”。它的计算路径是：先用同一个编码器分别处理 $x$ 和 $y$，得到句向量 $e_x=f_\theta(x)$、$e_y=f_\theta(y)$，再计算两个句向量的余弦相似度。它不只比较词是否重叠，而是把训练中学到的语义信息压进向量空间。

---

## 代码实现

最小工程流程通常是：预处理、特征构造、相似度计算、阈值判断、结果排序。

预处理流程图：

```text
原始文本
  -> 清洗空白字符
  -> 中文分词
  -> 大小写统一
  -> 去停用词
  -> 数字和符号归一化
  -> 输出 token 列表
```

下面代码不依赖中文分词库，为了可运行，假设输入已经用空格分好词。真实项目中可以把 `tokenize` 替换成 `jieba.lcut`、自研词典分词或模型分词。

```python
from math import sqrt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOPWORDS = {"的", "了", "吗"}

def tokenize(text):
    return [t.strip().lower() for t in text.split() if t.strip() and t.strip() not in STOPWORDS]

def jaccard_similarity(tokens_a, tokens_b):
    sa, sb = set(tokens_a), set(tokens_b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)

def edit_distance(a, b):
    # a, b 可以是字符列表，也可以是词列表
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            replace_cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,                 # 删除
                dp[i][j - 1] + 1,                 # 插入
                dp[i - 1][j - 1] + replace_cost   # 替换
            )
    return dp[m][n]

x = "我 喜欢 苹果"
y = "我 喜欢 香蕉"

tx = tokenize(x)
ty = tokenize(y)

jac = jaccard_similarity(tx, ty)
dist = edit_distance(tx, ty)

vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None)
tfidf = vectorizer.fit_transform([" ".join(tx), " ".join(ty)])
cos = cosine_similarity(tfidf[0], tfidf[1])[0][0]

assert tx == ["我", "喜欢", "苹果"]
assert ty == ["我", "喜欢", "香蕉"]
assert abs(jac - 0.5) < 1e-9
assert dist == 1
assert 0 <= cos <= 1

is_near_duplicate = cos > 0.8
print({"jaccard": jac, "tfidf_cosine": round(cos, 3), "edit_distance": dist, "near_dup": is_near_duplicate})
```

`scikit-learn` 的 `cosine_similarity` 适合直接计算稀疏向量相似度。`scipy.spatial.distance.jaccard` 提供的是 Jaccard distance，注意它返回距离，不是相似度；如果要转成相似度，可用 `1 - distance`。`gensim` 的 `wmdistance` 用于在词向量上计算 WMD。`Sentence-BERT` 的推理接口通常是先 `model.encode([x, y])` 得到两个句向量，再做余弦相似度。

SBERT 的工程调用形态通常如下：

```python
# 需要安装 sentence-transformers
# from sentence_transformers import SentenceTransformer, util
#
# model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# emb = model.encode(["怎么修改密码", "如何重置登录密码"], normalize_embeddings=True)
# sim = float(emb[0] @ emb[1])
# if sim > 0.8:
#     print("近似重复")
```

阈值判断必须基于业务数据。`sim > 0.8` 只能作为示例，不是通用规则。

---

## 工程权衡与常见坑

文本相似度的核心问题不是“算出来”，而是“算得稳定、算得可解释、算得足够快”。生产系统通常不会只用一种指标，而是按阶段组合：粗召回用便宜方法，精排用昂贵模型。

真实工程例子：问句检索系统中，如果只用 `Jaccard`，`"如何重置登录密码"` 和 `"忘记密码怎么办"` 可能因为词面重叠少而漏召回。如果直接用 `SBERT` 对所有候选问题做全量比对，计算成本又可能过高。更常见的方案是先用 `TF-IDF cosine` 或 BM25 取前几百个候选，再用 `SBERT` 重排。

| 常见坑 | 影响 | 规避策略 |
|---|---|---|
| 分词不统一 | 同一句话在不同模块得到不同 token，结果漂移 | 统一分词器、词典、停用词表和版本 |
| Jaccard 忽略词频 | `重要 重要 重要` 和 `重要` 被看得很接近 | 需要词频时改用 TF、TF-IDF 或 BM25 |
| Cosine 依赖特征 | 分词差、停用词多会污染向量 | 做清洗、停用词过滤、领域词典维护 |
| Edit distance 对长句敏感 | 长文本中少量改写会被距离放大 | 对短字段使用，或做归一化距离 |
| WMD 计算重 | 候选量大时延迟高 | 限制候选数，先召回再计算 |
| WMD 受 OOV 影响 | 未登录词没有词向量，语义丢失 | 使用覆盖更好的词向量或子词模型 |
| SBERT 阈值未校准 | 不同业务下同一分数含义不同 | 用标注集画分布，按准确率和召回率选阈值 |
| 数字和英文混写 | `iPhone 15`、`iphone15`、`15` 可能被错分 | 做大小写、空格、单位和数字归一化 |

OOV，out-of-vocabulary，指模型词表里没有的词。停用词是高频但区分度低的词，例如“的”“了”“吗”。这些细节会显著影响表面相似度，也会间接影响语义模型输入。

中文字符级与词级的选择要按任务定。拼写纠错、OCR 纠错、短标题近重复可以用字符级，因为字符变化本身重要。意图匹配、FAQ 检索、文本分类更适合词级或句向量，因为它们更关注语义单位。

---

## 替代方案与适用边界

没有一种方法适合所有场景。方法选择应从任务目标倒推，而不是从算法复杂度出发。

| 方法 | 适用场景 | 优点 | 缺点 | 成本 |
|---|---|---|---|---|
| Jaccard | 标签、短句、近重复去重 | 快、解释性强 | 不看词频和语义 | 低 |
| Cosine | 检索粗召回、文本分类特征 | 支持 TF-IDF，工程成熟 | 依赖分词和向量化方式 | 低到中 |
| Edit distance | 拼写纠错、字段匹配、短文本改写检测 | 对局部编辑敏感 | 长文本效果差，不懂语义 | 中 |
| WMD | 小规模语义距离分析 | 利用词向量语义 | 慢，依赖词向量覆盖 | 高 |
| SBERT | 语义匹配、重排、聚类 | 句级语义表达强 | 需要模型推理和阈值标定 | 中到高 |

如果目标是近重复去重，`Jaccard` 或 `TF-IDF cosine` 往往足够。例如文章标题去重、FAQ 问法轻微变化检测，不一定需要上深度模型。如果目标是“表达同一个问题但措辞不同”的语义匹配，`SBERT` 更合适。例如 `"银行卡丢了怎么处理"` 和 `"卡片遗失后如何挂失"` 词面重叠不高，但意图接近。如果文本中拼写错误很多，应先加 `edit distance` 或纠错层，再做语义匹配。

还要区分 `cross-encoder` 和 `SBERT`。`SBERT` 是双塔结构：两段文本分别编码成向量，再做余弦相似度，适合提前向量化和大规模检索。`cross-encoder` 是把两段文本拼在一起输入模型，让模型直接输出匹配分数，通常精度更高，但无法提前为单条文本独立缓存向量，成本更高。粗召回一般用 SBERT 或向量检索，最后少量候选重排可以用 cross-encoder。

阈值不是通用常数。一个客服 FAQ 系统里 $0.82$ 可能表示高相似；换到法律条款检索，$0.82$ 可能仍然不够可靠。阈值必须用业务标注数据校准，至少观察正样本和负样本的分数分布，再结合误召回和漏召回成本决定。

当词向量覆盖不足时，WMD 的效果会下降。领域词、缩写、新产品名、代码标识符如果没有合适词向量，运输成本就不能正确反映语义距离。此时应优先考虑领域词向量、子词模型、SBERT 或直接使用业务检索模型。

---

## 参考资料

1. [scikit-learn: cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
2. [SciPy: scipy.spatial.distance.jaccard](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.spatial.distance.jaccard.html)
3. [NIST DADS: Levenshtein distance](https://xlinux.nist.gov/dads/HTML/Levenshtein.html)
4. [Kusner et al., 2015, From Word Embeddings To Document Distances](https://proceedings.mlr.press/v37/kusnerb15.html)
5. [ACL Anthology: Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/)
6. [gensim: KeyedVectors.wmdistance](https://radimrehurek.com/gensim/models/keyedvectors.html)
7. [Sentence Transformers Documentation](https://www.sbert.net/)
