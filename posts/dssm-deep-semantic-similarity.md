## 核心结论

DSSM（Deep Structured Semantic Model，深度结构化语义模型）是一种双塔语义匹配模型：它把 `Query` 和 `Doc` 分别编码成向量，再用向量相似度判断二者是否相关。

`Query` 是用户输入的检索词，例如“手机壳防摔”。`Doc` 是候选文档、商品、广告或内容，例如“耐冲击保护套”。双塔结构是指模型有两路编码器，一路处理 `Query`，一路处理 `Doc`，二者先独立生成语义向量，再计算匹配分。

DSSM 的核心结论很直接：

| 对象 | 作用 |
|---|---|
| Query | 用户输入 |
| Doc | 候选文档、商品或内容 |
| 双塔编码器 | 分别生成语义向量 |
| 余弦相似度 | 计算相关性 |
| 负采样 | 用少量负例近似完整排序训练 |

相似度通常写成：

$$
s(q,d)=cos(z_q,z_d)=\frac{z_q \cdot z_d}{||z_q||||z_d||}
$$

其中 `z_q` 是 `Query` 向量，`z_d` 是 `Doc` 向量。两个向量方向越一致，余弦相似度越高，模型越倾向判定它们相关。

玩具例子：用户搜“手机壳防摔”，商品标题是“耐冲击保护套”。两个文本字面重合很少，但都指向“保护手机、防摔抗冲击”。传统关键词匹配可能弱，DSSM 的目标是把它们映射到接近的向量空间里。

真实工程例子：电商搜索召回中，可以离线把所有商品标题编码成向量并建立向量索引。用户输入 query 后，系统在线编码 query，再从向量索引里找 topK 个最相似商品，交给后续精排模型继续排序。

---

## 问题定义与边界

DSSM 解决的是语义检索和召回问题。召回是推荐、搜索系统中的第一层候选筛选，目标是从海量候选里快速找出一批可能相关的结果，而不是直接给出最终排序。

输入可以定义为：

$$
x_q,x_d \text{ 为输入词袋特征}
$$

$$
z_q=f_q(x_q), \quad z_d=f_d(x_d)
$$

其中 `f_q(·)` 和 `f_d(·)` 是两路编码器。词袋特征是把文本表示成词或子词是否出现、出现多少次的向量表示，它不显式保留原始词序。

例如用户搜“ipad 保护膜”，候选商品有“平板钢化膜”“Apple 平板贴膜”“屏幕保护贴”。DSSM 关注的是这些表达背后的语义接近性，而不是逐字匹配“ipad”和“保护膜”。

它适合和不适合的场景如下：

| 适合 | 不适合 |
|---|---|
| 召回 | 最终精排 |
| 同义改写匹配 | 复杂 token 交互建模 |
| 长尾词泛化 | 强规则排序 |
| 大规模候选检索 | 需要逐词对齐的判断 |

`token` 是模型处理文本的基本单元，可以是词、字、子词或字符片段。DSSM 的一个重要边界是：`Query` 和 `Doc` 的编码是独立完成的，所以它没有显式建模“query 第 2 个词和 doc 第 5 个词是否强相关”这类逐 token 交互。

这个边界带来两个结果。第一，`Doc` 向量可以离线预计算，线上延迟低。第二，它不适合做细粒度排序。如果同一个 query 下两个商品都“差不多相关”，DSSM 能把它们都拉近，但很难像 cross-encoder 一样把 query 和 doc 拼在一起逐词比较后再打分。

---

## 核心机制与推导

DSSM 的完整链路可以概括为：

```text
Query/Doc -> Word Hashing -> DNN -> L2 Normalize -> Cosine Similarity -> Softmax Loss
```

`Word Hashing` 是把词拆成字符 n-gram 的方法。字符 n-gram 是连续的若干个字符片段，例如英文 trigram 表示连续 3 个字符。它的作用是降低词表维度，并缓解 OOV 问题。OOV 是 out-of-vocabulary，指测试时出现了训练词表里没有的词。

例如 `running` 可以拆成若干字符片段，如 `run`、`unn`、`nni`、`nin`、`ing`。即使模型没有见过完整的 `running`，也可能从这些局部片段里学到它和其他相近词的关系。

符号表如下：

| 符号 | 含义 |
|---|---|
| `x_q` | query 输入特征 |
| `x_d` | doc 输入特征 |
| `z_q` | query 向量 |
| `z_d` | doc 向量 |
| `s(q,d)` | query 与 doc 的余弦相似度 |
| `γ` | 相似度缩放系数 |
| `d+` | 正样本文档 |
| `d_j` | 候选文档，包含正样本和负样本 |

训练目标是让点击过、相关的 doc 得分高，让负样本得分低：

$$
p(d^+|q)=\frac{exp(\gamma s(q,d^+))}{\sum_j exp(\gamma s(q,d_j))}
$$

$$
L=-\sum log \ p(d^+|q)
$$

`γ` 是缩放系数，用来拉开相似度差异。分母是一个近似 softmax，通常包含一个正样本和多个负样本。softmax 是把多个得分转换成概率分布的方法，得分越高，对应概率越大。负采样是指不拿全量 doc 做训练，而是给每个正样本配若干负例来近似完整训练目标。原论文中常见做法是每个正样本配 4 个随机未点击文档。

数值玩具例子：设向量已经归一化，`z_q=(1,0)`，正样本 `z_d+=(1,0)`，两个负样本分别是 `z_d1=(0,1)`、`z_d2=(-1,0)`，取 `γ=1`。

```text
s+ = 1
s1 = 0
s2 = -1
p(d+|q) = e^1 / (e^1 + e^0 + e^-1) ≈ 2.718 / 4.086 ≈ 0.665
L ≈ -log(0.665) ≈ 0.408
```

这说明正样本方向越接近 query，概率越高，损失越小。训练过程就是不断调整编码器参数，让相关 pair 的向量更接近，不相关 pair 的向量更远。

---

## 代码实现

最小召回流程可以写成：

```python
q_vec = encode(query)
doc_vecs = load_precomputed_doc_vectors()
scores = cosine_similarity(q_vec, doc_vecs)
topk = argsort(scores)[-K:]
```

这段逻辑的含义是：先把商品都变成向量，再拿用户查询去找最像的商品。

下面是一个可运行的 Python 玩具实现，包含输入特征构造、向量编码、余弦相似度和 topK 召回。它不是完整神经网络训练代码，但保留了 DSSM 线上召回的核心形态。

```python
import math
from collections import Counter

def char_trigrams(text):
    text = f"#{text.lower()}#"
    return [text[i:i+3] for i in range(len(text) - 2)]

def featurize(text):
    return Counter(char_trigrams(text))

def cosine(a, b):
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def retrieve(query, docs, k=2):
    q_vec = featurize(query)
    scored = [(doc, cosine(q_vec, featurize(doc))) for doc in docs]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]

docs = [
    "耐冲击手机保护套",
    "蓝牙无线耳机",
    "防摔透明手机壳",
    "平板电脑钢化膜",
]

top2 = retrieve("手机壳防摔", docs, k=2)
assert top2[0][0] in {"防摔透明手机壳", "耐冲击手机保护套"}
assert len(top2) == 2
print(top2)
```

训练版伪代码如下：

```python
q = encoder_q(x_q)
d_pos = encoder_d(x_pos)
d_negs = encoder_d(x_negs)

score_pos = cos(q, d_pos)
score_negs = [cos(q, d) for d in d_negs]

loss = softmax_loss(score_pos, score_negs)
```

更接近深度学习实现时，双塔编码器通常是：

```python
def tower_forward(x, embedding, mlp):
    h = embedding(x)
    h = mlp(h)
    z = l2_normalize(h)
    return z
```

工程链路要拆成离线和在线两部分：

| 模块 | 离线/在线 | 作用 |
|---|---|---|
| Doc 编码 | 离线 | 预计算候选向量 |
| 向量索引 | 离线 | 建立 ANN 检索结构 |
| Query 编码 | 在线 | 实时生成检索向量 |
| 相似度计算 | 在线 | topK 召回 |
| 精排模型 | 在线 | 对召回结果做细粒度排序 |

ANN 是 approximate nearest neighbor，近似最近邻检索，用来在大量向量中快速找相似向量。真实系统不会对全部商品逐个算余弦相似度，而是用 Faiss、HNSW 等索引结构降低延迟。

---

## 工程权衡与常见坑

DSSM 的效果不只取决于模型结构，还取决于训练数据、负样本质量、日志偏差处理和评估指标选择。

| 常见坑 | 现象 | 规避方式 |
|---|---|---|
| 把 DSSM 当精排 | 排序不够细 | 后接 cross-encoder、GBDT 或 LTR |
| 负样本太简单 | 学不到难边界 | 加 hard negatives |
| 直接套中文 | 字符粒度不合适 | 改成中文字符级或分词后 n-gram |
| 点击日志偏差 | 学到位置偏好 | 去噪、校正、时间切分 |
| 只看 NDCG | 线上召回变差 | 同时看 Recall@K、覆盖率、P99 延迟 |

hard negatives 是难负例，指看起来像相关但实际不相关的样本。新手常见错误是只随机采负样本。这样模型很容易学会区分“手机壳”和“蓝牙耳机”，但分不清“防摔手机壳”和“普通手机壳”哪个更匹配“手机壳防摔”。

真实工程中，电商搜索可以加入 `曝光未点` 和 `BM25 hard negatives`。`曝光未点` 是用户看到了但没有点击的商品，通常比随机商品更接近真实排序边界。BM25 是传统关键词检索算法，它召回的高分但未点击商品可以作为难负例，帮助模型学习“字面相关但语义或意图不够匹配”的情况。

中文场景还要注意 Word Hashing 的迁移。英文 trigram 有明确字符拼写结构，但中文词边界不同，直接照搬英文方案不一定合适。常见做法是使用中文字符粒度、分词后的词 n-gram，或直接使用现代 tokenizer 和 embedding 表示。

评估时也不能只看 NDCG。NDCG 是排序指标，关注高位结果质量；召回模型还必须关注 `Recall@K`、覆盖率和 P99 延迟。P99 延迟是 99% 请求都能完成的延迟上界，能反映线上尾部性能。

---

## 替代方案与适用边界

DSSM 适合语义召回、粗匹配和候选生成。它的核心价值是快、可预计算、容易扩展到大规模候选集合。

它不适合直接做最终排序，原因是没有显式 token-level 交互。token-level 交互是指模型能直接比较 query 中某个 token 和 doc 中某个 token 的关系。DSSM 把两边先压缩成向量再比较，表达能力天然受限。

新手版对比可以这样理解：DSSM 像“先把标题和搜索词各自总结成摘要，再比较摘要相似度”；Cross-Encoder 像“把 query 和 doc 放在一起逐字比对后再打分”。这个说法只是帮助理解，不能替代定义：本质区别是独立编码和联合编码。

| 方法 | 优势 | 劣势 | 适用阶段 |
|---|---|---|---|
| DSSM | 快、可预计算 | 交互弱 | 召回 |
| Bi-Encoder | 通用、易扩展 | 仍是独立编码 | 召回/粗排 |
| Cross-Encoder | 精度高 | 慢、难扩展 | 精排 |
| GBDT/LTR | 特征灵活 | 依赖手工特征 | 精排 |
| BM25 | 简单稳定、可解释 | 语义泛化弱 | 召回基线 |

Bi-Encoder 是双编码器模型，和 DSSM 思路接近，通常用更现代的 embedding、Transformer 或预训练模型做编码器。Cross-Encoder 是交叉编码器，把 query 和 doc 拼接后一起输入模型，适合精排但成本高。LTR 是 learning to rank，学习排序模型，常结合点击、转化、价格、类目等特征。

工程上更稳妥的架构通常是两阶段：召回阶段使用 DSSM、Bi-Encoder、BM25 或向量检索扩大候选覆盖；精排阶段使用 Cross-Encoder、GBDT、LTR 或多任务排序模型提高排序精度。如果目标是高精度排序，优先考虑“召回双塔 + 精排交互模型”的组合，而不是让 DSSM 单独承担全部排序逻辑。

---

## 参考资料

如果要确认 DSSM 的结构、公式和训练方式，先看原论文；如果要了解项目背景和后续扩展，再看 Microsoft Research 项目页和相关报告。本文的公式与训练目标以原论文为准。

- *Learning Deep Structured Semantic Models for Web Search using Clickthrough Data*  
  https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf

- Microsoft Research 论文页  
  https://www.microsoft.com/en-us/research/?p=165215

- DSSM 官方项目页  
  https://www.microsoft.com/en-us/research/project/dssm/

- *Unsupervised Learning of Word Semantic Embedding using the Deep Structured Semantic Model*  
  https://www.microsoft.com/en-us/research/publication/unsupervised-learning-of-word-semantic-embedding-using-the-deep-structured-semantic-model/
