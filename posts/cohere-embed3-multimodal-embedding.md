## 核心结论

Cohere Embed 3 的关键价值，是把**文本和图像编码到同一个向量空间**。向量空间可以理解成“把内容压缩成一串可比较数字的坐标系”。一旦文本和图像落在同一套坐标里，检索系统就可以直接做“中文搜图”“图片反查文档”“跨语言搜同类内容”，而不必先把图片强行转成文字再检索。

这件事的工程意义，不是多了一个“会看图”的模型，而是**索引结构统一了**。过去常见做法是：文本走文本 embedding，图片走 OCR 或图像模型，最后靠业务逻辑把两条链路拼起来。问题是链路一长，召回边界就变模糊，评估也更难做。统一多模态嵌入的意义，是让召回层只有一套相似度逻辑。

对多数工程团队，更实用的结论有三个：

| 结论 | 直接含义 |
|---|---|
| `embed-multilingual-v3.0` 是 1024 维 | 更偏召回质量和跨语言鲁棒性 |
| `embed-multilingual-light-v3.0` 是 384 维 | 更偏速度、存储和成本 |
| 图像输入进入同一检索空间 | 可以做图文互搜，而不是只做图像分类 |

玩具例子：查询“红色圆形按钮的设置页”，系统未必返回像素最接近的截图，而更可能返回语义上同类的界面图。这说明它优化的是“语义邻近”，不是“图片长得像”。

真实工程例子：企业知识库里有 PDF 截图、产品图、英文说明页和中文 FAQ。用户输入“找蓝底、带折线图的季度营收页”，系统可以直接从不同模态、不同语言的资产里一起召回候选，再交给 reranker 精排。reranker 可以理解成“第二阶段排序器”，负责在召回结果里做更细的排序。

---

## 问题定义与边界

先定义问题。多模态检索的目标，不是回答“图里是什么”，而是回答：**给定一个文本或图像查询，怎样在混合资产库中找出语义最相关的内容**。这里的“资产库”通常包括文章、截图、报表页面、商品图、设计稿、帮助中心页面等。

传统方案有三类：

| 方案 | 输入 | 索引方式 | 优点 | 局限 |
|---|---|---|---|---|
| 文本检索 | 文本 | 文本向量 | 简单，成熟 | 不能直接检索图片 |
| 图文分离检索 | 图像 + 文本 | 两套索引 | 易于局部改造 | 跨模态链路断裂 |
| 统一多模态嵌入检索 | 图像 + 文本 | 单一向量空间 | 图文互搜一致 | 需要重新评估质量与成本 |

Cohere Embed 3 解决的是其中的**表示层问题**。表示层可以理解成“系统内部怎样把内容表示成向量，方便后续比较”。它不直接解决下面这些问题：

| 不解决的问题 | 为什么不属于 Embed 3 本身 |
|---|---|
| OCR 识别错误 | 图片里的文字能否被读出，取决于 OCR 或图像内容本身 |
| 文档切分不合理 | 一页 PDF 切太大或太碎，都会影响召回 |
| 精排错误 | 召回后的排序通常还要靠 reranker |
| 权限过滤 | 哪些结果能返回给谁，是业务逻辑问题 |
| 最终答案生成 | 生成回答要靠 LLM 或模板系统 |

边界要说清楚。假设用户上传一张产品包装图，问“这是哪个型号的说明书”。Embed 3 的工作，是把这张图和说明书页面都放进同一可比较空间；它**不保证**一定命中正确型号。最终效果还依赖：

1. 说明书是否被正确切分并入库。
2. 图片是否清晰，是否超出格式和大小限制。
3. 元数据是否足够，例如品牌、系列、语言。
4. 是否在召回后做了精排。

因此，Embed 3 更适合被理解成**多模态检索底座**，而不是端到端“识图问答”系统。

---

## 核心机制与推导

从工程角度，可以把 Cohere Embed 3 理解为“**双编码器 + 共享空间对齐**”的实现。双编码器指“文本一套编码器、图像一套编码器”；共享空间对齐指“最后都映射到同一维度的向量空间里”。这是一种工程化理解，不是官方逐字架构声明，但它足够解释检索行为。

设文本查询为 $x$，图像输入为 $y$。文本编码器把文本映射成向量 $u$，图像编码器把图像映射成向量 $v$：

$$
u=f_t(x),\quad v=f_i(y),\quad u,v\in \mathbb{R}^d
$$

然后用余弦相似度计算相关性。余弦相似度可以理解成“比较两个向量方向有多接近”，范围通常在 $[-1,1]$：

$$
s(x,y)=\cos(u,v)=\frac{u^\top v}{\|u\|\|v\|}
$$

检索时，从库中选出相似度最高的前 $k$ 个结果：

$$
\text{top-}k=\arg\max_{y\in\mathcal{D}} s(x,y)
$$

符号说明如下：

| 符号 | 含义 |
|---|---|
| $x$ | 文本查询 |
| $y$ | 图像输入 |
| $f_t$ | 文本编码器 |
| $f_i$ | 图像编码器 |
| $u, v$ | 向量表示 |
| $d$ | 向量维度 |
| $s(x,y)$ | 相似度分数 |

玩具例子可以直接算。假设一条中文查询“蓝底折线图营收页”编码后得到向量 $q$，两张图分别得到 $a$ 和 $b$。如果：

- $\cos(q,a)=0.92$
- $\cos(q,b)=0.61$

那么系统会优先召回图 $a$。注意这里高分不代表图 $a$ 的像素和查询“长得像”，而是它在训练好的共享空间里，与该语义更接近。也就是说，模型更在意“这是营收图表页面”而不是“这张图是蓝色像素比较多”。

维度是最直接的工程变量。官方文档给出的关键信息是：

| 版本 | 维度 | 适合场景 | 代价 |
|---|---:|---|---|
| `embed-multilingual-v3.0` | 1024 | 更重召回质量、跨语言鲁棒性 | 存储和检索成本更高 |
| `embed-multilingual-light-v3.0` | 384 | 海量资产、高 QPS、成本敏感 | 需要实测是否足够 |

为什么维度会影响工程表现？因为一个 float32 向量每维通常占 4 字节。若有 $N$ 条向量，维度为 $d$，仅原始向量存储大约是：

$$
\text{Bytes} \approx N \times d \times 4
$$

以 100 万条向量为例：

- 1024 维：$1{,}000{,}000 \times 1024 \times 4 \approx 4.1$ GB
- 384 维：$1{,}000{,}000 \times 384 \times 4 \approx 1.5$ GB

这还没算索引结构、元数据、副本和缓存。实际系统里，维度下降通常会同步降低存储、网络传输和近邻搜索成本，但质量是否可接受，必须靠真实数据评估，而不能只靠参数表。

---

## 代码实现

真正可落地的实现，不是“调一次 API 就结束”，而是把文本和图像**统一入库、统一召回、分阶段排序**。

一个最小流程通常是：

1. 文本和图像分别生成 embedding。
2. 写入同一个向量库。
3. 查询时生成查询向量。
4. 先召回 top-k。
5. 再用 reranker 或业务规则精排。

下面给一个可运行的 Python 玩具实现。它不调用 Cohere API，而是用手工向量模拟“统一向量空间检索”，方便理解机制。

```python
import math

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)

# 假设这已经是模型产出的统一嵌入：
# query: “找蓝底、带折线图的季度营收页”
query_vec = [0.92, 0.05, 0.31, 0.12]

items = [
    {
        "id": "img_report_q3",
        "type": "image",
        "title": "Q3 revenue chart page",
        "vector": [0.90, 0.02, 0.35, 0.10],
    },
    {
        "id": "img_product_box",
        "type": "image",
        "title": "product packaging",
        "vector": [0.18, 0.81, 0.06, 0.21],
    },
    {
        "id": "doc_finance_en",
        "type": "text",
        "title": "Quarterly revenue summary",
        "vector": [0.88, 0.04, 0.28, 0.15],
    },
]

scored = []
for item in items:
    score = cosine(query_vec, item["vector"])
    scored.append((item["id"], round(score, 4), item["type"], item["title"]))

scored.sort(key=lambda x: x[1], reverse=True)

# top-2
top2 = scored[:2]

assert top2[0][0] in {"img_report_q3", "doc_finance_en"}
assert top2[0][1] >= top2[1][1]
assert scored[-1][0] == "img_product_box"

print(top2)
```

这段代码要表达的不是“真实模型如何训练”，而是工程链路的本质：**文本查询和图像资产都被表示成同一类向量对象**，检索阶段根本不关心它原来是文本还是图像，只关心相似度分数。

如果换成真实工程实现，伪代码会更接近下面这样：

```python
# 1. 生成文本查询向量
query_vec = cohere_embed(
    model="embed-multilingual-v3.0",
    input_type="search_query",
    texts=["找蓝底、带折线图的季度营收页"]
)

# 2. 生成文档向量
doc_vecs = cohere_embed(
    model="embed-multilingual-v3.0",
    input_type="search_document",
    texts=[
        "Q3 revenue summary with line chart",
        "Product packaging specification"
    ]
)

# 3. 生成图片向量
img_vec = cohere_embed(
    model="embed-multilingual-v3.0",
    input_type="image",
    images=["data:image/png;base64,..."]
)

# 4. 统一写入向量库
vector_db.upsert([
    {"id": "doc_1", "vector": doc_vecs[0], "type": "text", "lang": "en"},
    {"id": "doc_2", "vector": doc_vecs[1], "type": "text", "lang": "en"},
    {"id": "img_7", "vector": img_vec, "type": "image", "source": "pdf_page"},
])

# 5. 统一召回
candidates = vector_db.search(query_vector=query_vec[0], top_k=100)

# 6. 精排
ranked = rerank(query="找蓝底、带折线图的季度营收页", candidates=candidates)

# 7. 返回最终结果
result = ranked[:10]
```

真实工程例子：一个跨国电商内容平台，把商品主图、海报、规格页截图、英文文案和中文运营说明都放进同一索引。中文客服输入“找带三段折线图的欧洲站销售分析页”，召回层先把图和文一起捞出来，再由 reranker 结合标题、语言、时间和品类做精排。这时，多模态 embedding 负责的是“把候选找全”，不是“一次性把答案断准”。

---

## 工程权衡与常见坑

多模态 embedding 的难点，几乎都不在 API 调用，而在系统设计。下面这些坑最常见：

| 坑点 | 后果 | 规避方式 |
|---|---|---|
| 图文分开建索引 | 文本搜图、图片搜文容易断链 | 文本和图像进入同一向量空间 |
| 直接认为 light 版等价 | 线上召回下降 | 用真实查询集评估 Recall@K、nDCG、MRR |
| 忽略图片输入限制 | 上传失败或吞吐异常 | 控制格式、大小，单图不超过 5MB |
| 只看维度不看业务指标 | 选型失真 | 先定义评估集，再比较 1024 和 384 |
| 把 embedding 当最终排序 | top-k 噪声偏多 | 召回与精排分离 |
| 以为启用图像后要重嵌全部文本 | 不必要的成本上升 | 复用既有文本向量，按需补图像索引 |

这里有几个官方能力边界很重要：

| 项目 | 官方信息 |
|---|---|
| 多语言 | `embed-multilingual-v3.0` 支持 100+ 语言 |
| 图片格式 | `png`、`jpeg`、`webp`、`gif` |
| 图片大小 | 单张最大 5MB |
| 图片批量 | v3 图像 embedding 当前单次最多 1 张 |
| 文本批量 | v2 Embed API 文本一次最多 96 条输入 |

为什么这些限制重要？因为它们会直接影响吞吐设计。比如图片当前不支持批量，你就不能简单照搬文本 embedding 的批处理管线。队列、重试、超时和缓存策略都要分开设计。

另一个常见误解是：维度更低就只是“轻微降级”。这不严谨。384 维确实更省，但它省掉的是表达容量。表达容量可以理解成“向量容纳语义细节的空间大小”。在简单任务上，384 维可能足够；在跨语言、多模态、细粒度区分场景里，差距可能会被放大。所以正确问法不是“384 行不行”，而是“在我的查询分布和标注集上，384 的 Recall@20 是否还满足 SLA”。

如果数据量上到百万级，维度差异就变得非常具体。仅原始 float32 向量，1024 维约 4.1GB，384 维约 1.5GB。再加上 ANN 索引、元数据、冷热分层和副本，差距会进一步放大。对高 QPS 在线检索，这常常意味着机器规格、缓存命中和响应时间的连锁变化。

---

## 替代方案与适用边界

Cohere Embed 3 适合的是“统一多模态检索”问题，不是所有视觉或检索任务都要用它。选型应该按任务拆开。

| 方案 | 适用场景 | 优点 | 不适合的场景 |
|---|---|---|---|
| Cohere Embed 3 | 图文互搜、企业知识库、跨语言多模态召回 | 一套空间统一检索 | 只做纯文本任务时可能偏重 |
| 纯文本 embedding | FAQ、文档问答、知识库检索 | 简单、便宜、成熟 | 不能直接检索图片 |
| OCR + 文本 embedding | 扫描件、票据、表单 | 对“图里主要是字”很有效 | 对视觉语义和布局理解有限 |
| 图像分类/检测模型 | 商品识别、目标检测、质检 | 对对象识别强 | 不适合开放语义检索 |
| Caption 后再文本检索 | 低频场景、快速原型 | 容易接入旧系统 | caption 质量决定上限 |

如果业务只是“PDF 文本问答”，那更便宜的路线通常是 OCR 或文本抽取，加普通文本 embedding。因为这类任务的主要信号本来就在文字里。反过来，如果业务是“图搜图、图搜文、文搜图”，尤其还带跨语言需求，那么统一多模态 embedding 才真正有结构性优势。

还要注意，Embed 3 不是视觉理解模型的替代品。如果你的问题是“图中有几个零件”“这是不是裂纹缺陷”，那更像检测或分类任务，不是语义检索任务。检索模型擅长的是找相关内容，不擅长给出严格的结构化视觉判断。

工程上最稳的策略通常是：

1. 先判断问题是不是“召回问题”。
2. 如果是，再判断是否真的需要跨模态统一空间。
3. 如果需要，先用 1024 维做基线。
4. 成本压力大时，再评估 384 维是否足够。
5. 无论选哪个版本，都用真实查询集验证，而不是看参数表拍板。

---

## 参考资料

1. Cohere Docs: Cohere’s Embed Models  
https://docs.cohere.com/docs/cohere-embed

2. Cohere Docs Changelog: Embed v3.0 Models are now Multimodal  
https://docs.cohere.com/v1/changelog/embed-v3-is-multimodal

3. Cohere Docs: Unlocking the Power of Multimodal Embeddings  
https://docs.cohere.com/v2/docs/multimodal-embeddings

4. Cohere Docs API Reference: Embed API (v2)  
https://docs.cohere.com/v2/reference/embed

5. Cohere Product Page: Embed  
https://cohere.com/embed

6. Microsoft Tech Community: Introducing Multimodal Embed 3  
https://techcommunity.microsoft.com/blog/machinelearningblog/introducing-multimodal-embed-3-powering-enterprise-search-across-images-and-text/4276660

7. 说明：文中“统一向量空间”“双编码器 + 共享空间对齐”“晚期融合”属于对官方能力描述的工程化解释，用于帮助理解检索机制，不应视为官方逐字架构承诺。
