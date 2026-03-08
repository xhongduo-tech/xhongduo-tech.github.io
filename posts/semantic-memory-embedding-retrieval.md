## 核心结论

语义记忆的 Embedding 表示，本质上是把文本、事件和经验压缩成可比较的向量；检索阶段再把用户问题编码成同一类向量，按“谁更像”找回相关记忆。这里的“像”，通常指语义相近，而不是时间正确、编号精确或版本一致。

结论先给出三条。

第一，语义记忆的上限不只由 Embedding 模型决定，还由检索策略决定。`text-embedding-3-large` 作为通用 Embedding 模型，官方给出的表示维度上限是 3072；OpenAI 在发布页给出的公开基准是 MTEB 平均 64.6、MIRACL 平均 54.9。公开榜单里，它在若干通用检索任务上的平均 `nDCG@10` 大约可到 0.811，但在更具体的 ESG 检索任务里，ClimateNLP 2025 论文报告它在一个测试切分上的 `NDCG@50` 约为 0.602。含义很直接：模型本身很强，但真实效果仍由数据域、切分方式、过滤规则、召回策略和重排流程共同决定，不能把通用榜单成绩直接当成生产效果。

第二，纯向量检索擅长“语义相近”，不擅长“时间正确”与“术语精确”。例如“最近 ESG 报告”和“去年 ESG 报告”在语义上非常接近，向量也往往接近；但用户真正关心的是“最近”这个时间约束。再例如“GRI 2-6”“SKU-2847-B”“错误码 E1024”这类精确词，BM25 往往比纯 Embedding 更可靠。**BM25** 可以先理解成一种关键词检索方法：词在文档里出现得越重要、越稀有，分数通常越高。

第三，工程上更稳的方案通常是混合检索：向量召回 + BM25 召回 + reranker 再排序。**reranker** 可以理解为“对已经召回出来的少量候选再精排一次的模型”。它不负责全库搜索，而是负责把前几十条候选重新排得更准。公开生产经验里，混合检索相对纯向量检索常能明显提高 `Recall@10`；一些 RAG 案例还报告，在不改生成模型的前提下，仅通过 BM25 + vector + reranker 的检索改造，就能把答案准确度提升一截。这说明检索流程本身就是系统能力的一部分，不是“可有可无的前处理”。

| 策略 | 优点 | 主要缺点 | 典型表现 |
| --- | --- | --- | --- |
| Embedding-only | 语义扩展强，能覆盖近义表达 | 容易忽略时间、数字、术语精确匹配 | 通用榜单强，垂直领域效果波动大 |
| Embedding + BM25 | 同时覆盖语义和关键词 | 系统复杂度、维护成本和延迟上升 | 常见于企业知识库与 RAG |
| Embedding + BM25 + reranker | 相关性最稳，引用命中率更高 | 成本最高，需控制候选集规模 | 适合答错成本高的问答系统 |

先看一个最小例子。查询是“最近 ESG 报告”。纯向量系统可能把“2024 年 ESG 报告模板说明”和“2025 年最新 ESG 报告发布”都排得很靠前，因为两者主题都围绕 ESG 报告；加入 BM25 后，含“最近”“最新”“2025”这类词的文档更容易进入前列；再加时间过滤后，老文档可以直接不参与排序。对用户来说，系统从“主题相关”变成了“语义相关且约束正确”。

---

## 问题定义与边界

讨论“语义记忆”时，最稳妥的拆法是三步：编码、存储、检索。

编码，是把文本转成向量。向量不是“文本原文的压缩包”，而是一种机器可计算的表示。它保留的是统计意义上的语义模式，不保证保留每个数字、每个版本号、每个时间顺序。对新手来说，一个很重要的认识是：向量强，不等于什么都能记住；它擅长“意思像不像”，不天然擅长“是不是 2025 年版”。

存储，是把向量和原文片段一起放进可查询系统。常见做法是向量库存向量，普通数据库或文档索引存原文与 metadata。**metadata** 就是随文档一起保存的结构化字段，例如 `date`、`source`、`author`、`version`、`department`、`permission`。这些字段本身不一定参与语义编码，但会在过滤、排序、审计和引用追踪里起决定性作用。

检索，是把用户查询也转成向量，再和库里向量比较距离，同时再跑一次 BM25 或规则过滤，最后把多路结果融合。多数工程问题都不是出在“向量算不出来”，而是出在“该过滤的没过滤、该融合的没融合、该重排的没重排”。

| 步骤 | 输入 | 输出 | 核心任务 | 常见问题 |
| --- | --- | --- | --- | --- |
| 编码 | 查询或文档文本 | 向量表示 | 压缩语义信息 | 时间关系弱；数字、代码、专名可能不稳定 |
| 存储 | 向量 + 原文 + metadata | 可索引记忆库 | 让记忆可查、可过滤、可追踪 | 文档更新后旧向量未重建；版本混杂 |
| 检索 | 用户查询 | 候选文档列表 | 找回相关且满足约束的内容 | 只看语义会错过精确词；只看关键词会漏掉近义表达 |

问题边界也要说清楚，否则很容易把不同检索任务混为一类。

如果查询是“怎么写 ESG 报告”，这是典型语义问题。用户可能说“可持续披露”“非财务报告”“CSR 报告”，而文档里写的是“ESG disclosure”。这时 Embedding 的优势很明显，因为它能把近义表达压到相近区域。

如果查询是“最近政策”“2025 年 6 月版本”“GRI 2-6”，这已经不是单纯语义问题，而是“语义 + 时间 + 术语”的组合问题。纯余弦相似度没有显式时间轴，也不理解“2-6”是编号而不是普通词，因此它无法稳定保证结果正确。

可以把不同查询类型简单区分成下面四类。

| 查询类型 | 例子 | 主要需求 | 优先策略 |
| --- | --- | --- | --- |
| 语义型 | “怎么写 ESG 报告” | 找近义表达、概念解释 | Embedding 优先 |
| 精确型 | “GRI 2-6”“SKU-2847-B” | 精确词命中 | BM25 优先 |
| 时间型 | “最新数据跨境政策” | 时间正确、版本正确 | 过滤 + 时间重排 |
| 混合型 | “2025 年最新 ESG 披露政策” | 同时要语义、术语和时间 | Hybrid + reranker |

真实工程里，企业知识库是最典型的例子。法务团队问“最新数据跨境政策”，如果系统只做 Embedding 检索，2023 年和 2025 年的政策解读都可能被召回，因为它们主题高度接近；但法务不能接受的恰恰不是“语义略偏”，而是“时间错误”。这时至少要做两件事：一是 metadata 过滤，把发布日期限制在合理时间窗口；二是用 BM25 保证“最新”“跨境”“政策”这些关键词确实命中。

换句话说，语义记忆不是“把所有历史文本变成向量”就结束了。真正的问题是：系统能不能在用户给出约束时，找回“对的那段、对的版本、对的时间”。

---

## 核心机制与推导

语义检索最常见的打分方式是余弦相似度。它衡量两个向量方向有多接近，公式是：

$$
\mathrm{sim}(q,d)=\frac{e_q \cdot e_d}{\|e_q\|\|e_d\|}
$$

其中，$e_q$ 是查询向量，$e_d$ 是文档向量，$e_q \cdot e_d$ 是点积，$\|e_q\|$ 与 $\|e_d\|$ 是向量长度。白话解释是：如果两段文本表达的意思更接近，它们在高维空间里的方向通常更接近，余弦值也更高。

如果向量已经做了长度归一化，那么：

$$
\|e_q\|=\|e_d\|=1 \Rightarrow \mathrm{sim}(q,d)=e_q \cdot e_d
$$

这就是为什么一些 Embedding 实现里，余弦相似度和点积排序几乎等价。OpenAI 的 Embeddings FAQ 也明确提到，其 embeddings 已做归一化，因此工程实现可以直接使用余弦或点积做高效近邻搜索。

但余弦相似度只回答一个问题：语义像不像。它不回答下面这些问题。

| 问题 | 余弦相似度是否天然处理 | 原因 |
| --- | --- | --- |
| 是不是最新版本 | 否 | “最新”是时间约束，不是纯语义相似 |
| 是否精确命中编号 | 不稳定 | 编号、SKU、错误码常缺乏稳定语义模式 |
| 是否来自指定来源 | 否 | 来源属于 metadata，不在纯向量距离里 |
| 是否有权限访问 | 否 | 权限是业务规则，不是语义属性 |

所以，实际系统通常要把多种信号合在一起。最常见的第一步就是混合检索。

BM25 的核心思想可以写成下面这个常见形式：

$$
\mathrm{BM25}(q,d)=\sum_{t \in q} \mathrm{IDF}(t)\cdot
\frac{f(t,d)\cdot (k_1+1)}
{f(t,d)+k_1\cdot \left(1-b+b\cdot \frac{|d|}{\mathrm{avgdl}}\right)}
$$

其中：

- $t$ 是查询中的一个词
- $f(t,d)$ 是词 $t$ 在文档 $d$ 中出现的次数
- $\mathrm{IDF}(t)$ 表示词越稀有，权重越高
- $|d|$ 是文档长度
- $\mathrm{avgdl}$ 是语料平均文档长度
- $k_1,b$ 是调节参数

这套公式不要求理解推导细节，初学者先抓住一件事就够了：BM25 关注的是“关键词是否命中、命中多少次、这个词稀不稀有、文档是否过长”。因此它对编号、专名、法规条目、错误码通常更敏感。

一种简单而稳健的融合方法是 **RRF**，全称 Reciprocal Rank Fusion，可以理解为“把多个排序名单按名次倒数加权合并”。公式是：

$$
\mathrm{score}_{RRF}(d)=\sum_i \frac{1}{k+\mathrm{rank}_i(d)}
$$

其中 $\mathrm{rank}_i(d)$ 表示文档 $d$ 在第 $i$ 个检索器中的名次，$k$ 是平滑常数，常见取值是 60。名次越靠前，贡献越大；文档若没有进入某个检索器的候选名单，那一项贡献就是 0。

看一个玩具例子。

查询：`ESG 新政`

| 文档 | cosine 排名 | BM25 排名 | 解释 |
| --- | --- | --- | --- |
| A: “ESG 报告总体介绍” | 1 | 5 | 语义很近，但没有“新政”精确词 |
| B: “2025 ESG 新政解读” | 3 | 1 | 语义略远，但关键词和时间都对 |
| C: “CSR 合规综述” | 2 | 8 | 主题相关，但不够精确 |

若只看向量，A 可能排第一；若只看 BM25，B 会排第一但可能漏掉一些近义表达；RRF 融合后，B 通常会更稳，因为它在两个名单里都不差。这里的关键不是“谁永远第一”，而是“正确文档更不容易被单一路径漏掉”。

这也是为什么混合检索常能提升 `Recall@10`。**Recall@10** 可以先理解为：在前 10 个结果里，系统有没有把真正相关的文档找回来。BM25 把精确词命中的文档补进来，向量检索把近义表达补进来，前 10 条里“正确候选的覆盖率”就更高。

需要注意的是，Embedding 维度更高，不等于时间关系会自动出现。时间关系通常需要额外建模，常见做法有三类。

1. 元数据过滤：先按日期、版本筛选，再做向量排序。
2. 时间特征融合：把年份、版本、有效期等字段纳入最终打分。
3. 时间感知检索：把语义相关性和时间相关性拆开建模，例如 MRAG、TimeR4 这类方法。

可以把最终检索分数抽象成：

$$
\mathrm{score}(d)=\alpha \cdot \mathrm{semantic}(d)+\beta \cdot \mathrm{lexical}(d)+\gamma \cdot \mathrm{time}(d)+\delta \cdot \mathrm{policy}(d)
$$

其中：

- $\mathrm{semantic}(d)$ 表示向量相关性
- $\mathrm{lexical}(d)$ 表示关键词相关性
- $\mathrm{time}(d)$ 表示时间匹配程度
- $\mathrm{policy}(d)$ 表示权限、来源、业务规则等约束
- $\alpha,\beta,\gamma,\delta$ 是权重

这不是唯一公式，但它揭示了工程事实：生产检索从来不是一个“只靠余弦相似度”的单变量问题。

---

## 代码实现

下面给一个最小可运行的 Python 示例。它只用标准库，完整演示四件事：

1. 文档与查询如何生成一个玩具版“语义向量”
2. BM25 如何做关键词打分
3. RRF 如何融合两路召回
4. “最近/最新”这类查询如何用时间过滤兜底

代码不依赖真实向量库，也不调用外部 API，但流程和真实系统一致，适合先把机制跑通。

```python
import math
import re
from collections import Counter
from statistics import mean

TOKEN_RE = re.compile(r"[A-Za-z0-9\-]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec[:]
    return [x / norm for x in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(l2_normalize(a), l2_normalize(b)))


# 一个手工词向量表：只为演示语义相近的效果
TERM_VECS = {
    "esg": [1.0, 0.0, 0.0, 0.0],
    "csr": [0.9, 0.1, 0.0, 0.0],
    "report": [0.0, 1.0, 0.0, 0.0],
    "disclosure": [0.0, 0.9, 0.1, 0.0],
    "policy": [0.0, 0.0, 1.0, 0.0],
    "regulation": [0.0, 0.0, 0.95, 0.05],
    "latest": [0.0, 0.0, 0.1, 1.0],
    "recent": [0.0, 0.0, 0.1, 0.95],
    "2025": [0.0, 0.0, 0.05, 1.0],
    "2024": [0.0, 0.0, 0.05, 0.7],
}


def embed_text(text: str) -> list[float]:
    tokens = tokenize(text)
    dims = 4
    acc = [0.0] * dims
    matched = 0
    for token in tokens:
        if token in TERM_VECS:
            matched += 1
            vec = TERM_VECS[token]
            acc = [x + y for x, y in zip(acc, vec)]
    if matched == 0:
        return acc
    return [x / matched for x in acc]


class BM25Index:
    def __init__(self, documents: list[dict], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(doc["text"]) for doc in documents]
        self.doc_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = mean(self.doc_lens) if self.doc_lens else 0.0

        self.df = Counter()
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.df[term] += 1

    def idf(self, term: str) -> float:
        n_docs = len(self.documents)
        df = self.df.get(term, 0)
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_index: int) -> float:
        terms = tokenize(query)
        freqs = self.doc_freqs[doc_index]
        dl = self.doc_lens[doc_index]
        score = 0.0

        for term in terms:
            f = freqs.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf(term) * numerator / denominator

        return score

    def rank(self, query: str) -> list[tuple[str, float]]:
        scored = []
        for i, doc in enumerate(self.documents):
            scored.append((doc["id"], self.score(query, i)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


def vector_rank(query: str, documents: list[dict]) -> list[tuple[str, float]]:
    qvec = embed_text(query)
    scored = []
    for doc in documents:
        dvec = embed_text(doc["text"])
        scored.append((doc["id"], cosine_similarity(qvec, dvec)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def rrf_fuse(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    scores = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def detect_time_constraint(query: str) -> int | None:
    tokens = set(tokenize(query))
    if {"latest", "recent"} & tokens:
        return 2025
    for token in tokens:
        if token.isdigit() and len(token) == 4:
            return int(token)
    return None


docs = [
    {
        "id": "a",
        "year": 2024,
        "text": "ESG report guide disclosure framework 2024",
    },
    {
        "id": "b",
        "year": 2025,
        "text": "latest ESG policy update regulation 2025",
    },
    {
        "id": "c",
        "year": 2023,
        "text": "CSR compliance overview and annual report",
    },
]

query = "latest ESG policy"

bm25 = BM25Index(docs)
vec_hits = vector_rank(query, docs)
bm25_hits = bm25.rank(query)
fused_hits = rrf_fuse([vec_hits, bm25_hits])

year_floor = detect_time_constraint(query)
if year_floor is not None:
    fused_hits = [
        (doc_id, score)
        for doc_id, score in fused_hits
        if next(doc["year"] for doc in docs if doc["id"] == doc_id) >= year_floor
    ]

print("Vector ranking:", vec_hits)
print("BM25 ranking:", bm25_hits)
print("Fused ranking:", fused_hits)

assert vec_hits[0][0] in {"a", "b"}
assert bm25_hits[0][0] == "b"
assert fused_hits[0][0] == "b"
```

这段代码里有几个值得初学者注意的点。

第一，`embed_text` 只是一个玩具版向量器。它不是为了模拟真实模型精度，而是为了说明“语义相关词会被映射到相近向量”这个机制。真实系统里，这一步会由 `text-embedding-3-large` 之类的模型完成。

第二，BM25 与向量检索处理的是两类不同信号。向量更关心“latest ESG policy”和“recent regulation on ESG”是不是一回事；BM25 更关心文档里有没有明确出现 `latest`、`policy`、`2025` 这些词。

第三，RRF 不直接融合分数，而是融合名次。这样做的好处是不同检索器的原始分值尺度可以不同，但仍能稳定合并。

第四，时间约束最好显式处理。例子里只要查询包含 `latest` 或 `recent`，系统就加一道年份过滤。真实工程里，这一步通常会更严格，例如按发布日期、失效日期、版本号或生效区间做过滤。

如果放到真实系统里，流程通常会写成下面这样：

```python
query_vec = embed(query, model="text-embedding-3-large")

vec_hits = vector_index.search(
    query_vector=query_vec,
    top_k=50,
    filters={"department": "legal"}
)

bm25_hits = bm25_index.search(
    query=query,
    top_k=50
)

hybrid_hits = reciprocal_rank_fusion([vec_hits, bm25_hits])

if query_requires_freshness(query):
    hybrid_hits = [
        hit for hit in hybrid_hits
        if hit.metadata["date"] >= "2025-01-01"
    ]

reranked = reranker.rank(query=query, candidates=hybrid_hits[:20])
```

真实工程里的关键不在代码形式，而在数据结构是否完整。下面这些字段通常是必须的。

| 字段 | 作用 | 缺失后的后果 |
| --- | --- | --- |
| `date` | 做最新、历史时点、生效区间过滤 | 无法稳定回答“最新版” |
| `version` | 区分制度修订版、模板版本 | 新旧版本混杂 |
| `source` | 回答“这段话出自哪里” | 无法引用与审计 |
| `permission` | 控制谁能检索到什么 | 可能出现越权召回 |
| `doc_type` | 区分政策、报告、FAQ、会议纪要 | 排序混乱，重排难做 |

一个简单但常被忽略的经验是：检索系统不是只有“向量表”这一层。真正可用的系统至少要同时管理“文本、向量、元数据、索引更新时间、权限边界”这几部分，否则后期几乎一定会出现“模型看起来没问题，但结果不能上线”的情况。

---

## 工程权衡与常见坑

最常见的坑不是“模型太弱”，而是把检索问题误判成模型问题。很多团队一看到结果不准，第一反应是换更大的 Embedding；但如果问题根源是缺少时间过滤、文档切分错误或 metadata 不全，换模型往往只能带来边际改善。

| 常见坑 | 典型现象 | 根因 | 规避措施 |
| --- | --- | --- | --- |
| 时间敏感查询只做向量检索 | “最近政策”返回去年的文档 | 时间约束没有显式建模 | 加 `date` 过滤或时间重排 |
| 只依赖 BM25 | 用户换一种说法就搜不到 | 近义表达没有召回 | 增加 Embedding 召回 |
| 只依赖向量检索 | 编号、术语、产品名容易漏 | 精确词不稳定 | 用 BM25 补精确词 |
| 候选集太小 | reranker 提升有限 | 正确文档根本没召回 | 把召回 `top_k` 做大 |
| 文档切分过碎 | 一段话单独看没信息 | 语义上下文丢失 | 按标题、段落、表格边界切分 |
| 文档切分过粗 | 一段里混了太多主题 | 噪声过大 | 控制 chunk 长度与重叠 |
| 向量更新不及时 | 新文档入库后搜不到 | 增量索引链路缺失 | 建立重建与补偿机制 |
| 只看离线指标 | 线上答复仍常出错 | 评测集与真实查询分布不一致 | 增加真实查询回放评测 |

对初级工程师来说，最实用的判断标准不是“用了多少先进模型”，而是先问四个问题。

1. 用户的问题里有没有时间约束，例如“最新”“当时”“去年”“截至 2025 年 6 月”。
2. 用户的问题里有没有精确词，例如编号、SKU、制度条款、错误码、产品名。
3. 系统能不能在召回前先做必要过滤，例如权限、部门、来源、日期。
4. 候选集里有没有给 reranker 足够空间，如果只给 5 条候选，精排几乎无从发挥。

延迟和效果一定是 trade-off。BM25 + 向量 + reranker 通常比纯向量更慢，因为你做了多路召回和一次精排。可以用下面这张表理解取舍。

| 方案 | 延迟 | 准确性上限 | 成本 | 适用场景 |
| --- | --- | --- | --- | --- |
| 纯向量 | 低 | 中 | 低到中 | FAQ、相似问答、低延迟检索 |
| BM25 + 向量 | 中 | 较高 | 中 | 企业知识库、通用 RAG |
| BM25 + 向量 + reranker | 中到高 | 高 | 高 | 法务、政策、金融、医疗等高准确场景 |

对新手最有帮助的一条经验是：先把“过滤”放在“相似度排序”前面。比如查询是“2025 年 6 月最新政策”，先筛时间范围，再跑语义排序；不要指望余弦相似度自己理解“最新”。在高风险场景里，错误的时间比不那么相似的语义更危险。

还有一个常见误区是把 reranker 当成“万能补丁”。reranker 只能在候选集里做更细的判断，不能凭空创造没召回出来的文档。所以如果前面的召回阶段已经漏掉正确文档，reranker 再强也救不回来。

---

## 替代方案与适用边界

不存在一套对所有场景都最优的检索策略，只有更匹配问题边界的组合。选择策略时，先看用户问题里哪类约束最硬，再决定谁做主、谁做辅。

| 方案 | 适用场景 | 不适用场景 | 判断信号 |
| --- | --- | --- | --- |
| BM25-only | 法规编号、SKU、错误码、产品名 | 口语化提问、近义改写多 | 查询里有大量精确词、数字、编号 |
| Embedding-only | FAQ、知识解释、语义改写丰富 | 强时效、强版本控制 | 用户经常不会用标准术语 |
| Hybrid | 企业知识库、RAG、搜索问答 | 极端低延迟要求 | 既要语义扩展，又要关键词精确 |
| Temporal RAG / 时间感知检索 | 新闻、政策、财报、版本演化知识库 | 静态知识库 | 查询里高频出现“最新/当时/去年/某月” |

可以再用几个对比例子把边界看清楚。

| 查询 | 最优起点策略 | 原因 |
| --- | --- | --- |
| “年度 ESG 报告模板” | BM25 或 Hybrid | 关键词明确，精确命中很重要 |
| “怎样写一份面向投资人的可持续披露说明” | Embedding 或 Hybrid | 用户未必说出标准术语，语义扩展更重要 |
| “2025 年最新 ESG 披露政策” | Hybrid + 时间过滤 | 同时有语义、关键词、时间约束 |
| “2024 年 8 月生效的跨境数据制度” | 时间感知检索或强过滤 Hybrid | 时间点本身就是检索主约束 |

如果系统长期处理“最新政策”“某时点财报”“某版本制度”这类问题，只靠补丁式规则会越来越脆弱。原因很简单：你处理的不是普通语义搜索，而是时间敏感检索。此时更稳的路线通常是：

1. 给文档补齐时间元数据，例如 `publish_date`、`effective_date`、`expire_date`、`version`。
2. 在召回层显式支持时间过滤或时间重排。
3. 在评测集里加入带时间约束的真实查询，而不是只测普通语义相似度。
4. 必要时使用时间感知检索方法，例如 MRAG、TimeR4 这类把时间相关性单独建模的方案。

MRAG、TimeR4、ChronoQA 这类近年的研究共同指向一件事：当问题显式带时间约束时，传统“只比语义距离”的检索器会成为瓶颈。换句话说，语义记忆不只是“记住内容”，还要“记住内容在什么时间成立、在什么版本下成立、是否仍然有效”。

这也是语义记忆和普通向量搜索的根本区别。前者面向的是“可用于回答和行动的记忆”，后者往往只要求“找几个意思相近的文本”。一旦系统要用于企业问答、法务检索、政策分析、财报追溯，时间、版本和来源就不再是附属信息，而是检索正确性的组成部分。

---

## 参考资料

| 来源 | 内容摘要 | 章节关联 |
| --- | --- | --- |
| OpenAI, *New embedding models and API updates* https://openai.com/index/new-embedding-models-and-api-updates/ | 给出 `text-embedding-3-large`、3072 维、MTEB 64.6、MIRACL 54.9 等公开信息 | 核心结论、代码实现 |
| OpenAI Docs, *text-embedding-3-large* https://platform.openai.com/docs/models/text-embedding-3-large | 官方模型页，包含模型定位与接口信息 | 核心结论、工程权衡 |
| OpenAI Help, *Embeddings FAQ* https://help.openai.com/en/articles/6824809-embeddings-faq | 说明 embeddings 已归一化，实践中可直接用余弦相似度或点积 | 核心机制与推导 |
| Agentset, *OpenAI text-embedding-3-large* https://agentset.ai/embeddings/openai-text-embedding-3-large | 提供公开榜单中的平均 `nDCG@10` 等对比信息 | 核心结论 |
| Ahmed et al., ClimateNLP 2025, *Enhancing Retrieval for ESGLLM via ESG-CID* https://aclanthology.org/2025.climatenlp-1.pdf | ESG 检索场景下报告 `Recall/MRR/MAP/NDCG` 等结果，说明垂直领域与通用榜单存在差异 | 核心结论、问题定义 |
| 21medien, *Hybrid Retrieval* https://www.21medien.de/en/blog/hybrid-retrieval.html | 讨论混合检索相对纯语义检索的召回提升 | 核心结论、核心机制 |
| OptyxStack, *Fixing Low Recall in Production RAG* https://optyxstack.com/case-studies/rag-low-recall-retrieval-miss | 生产案例中通过 hybrid + reranker 改善答案准确度 | 核心结论、工程权衡 |
| Qian et al., EMNLP 2024, *TimeR4* https://aclanthology.org/2024.emnlp-main.394/ | 时间感知检索对时间知识问答有效，说明“时间相关性”需要单独建模 | 替代方案与适用边界 |
| Zhang et al., Findings of EMNLP 2025, *MRAG* https://aclanthology.org/2025.findings-emnlp.167/ | 将语义相关性与时间相关性拆开建模，适合时间敏感问答与检索 | 核心机制与推导、替代方案与适用边界 |
| Chen et al., *ChronoQA* https://www.nature.com/articles/s41597-025-06098-y | 时间敏感 RAG 场景中，检索失败是主要误差来源之一 | 工程权衡、替代方案 |
| Cormack et al., *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods* https://dl.acm.org/doi/10.1145/1571941.1572114 | RRF 的经典论文，说明为什么多路排序融合在工程上稳健 | 核心机制与推导 |
| Robertson and Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* https://dl.acm.org/doi/10.1561/1500000019 | BM25 的经典综述，适合理解关键词检索的公式和边界 | 核心机制与推导 |
