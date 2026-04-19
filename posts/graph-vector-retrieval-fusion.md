## 核心结论

图召回与向量召回融合，本质是在召回层同时利用两类不同信号：图召回用关系结构发现候选，向量召回用 Embedding 相似度发现候选。Embedding 是把文本、图片、商品、用户等对象表示成一组数字向量，便于计算相似度。

图召回更擅长回答“哪些物品和当前对象有明确关系”。例如查询“手机壳”时，它可以从用户购买、同品牌、同类目、搭配购、共浏览等边出发，找到苹果手机壳、同型号保护膜、充电器套装。向量召回更擅长回答“哪些物品在语义上接近”。例如它可以找到“保护套”“防摔壳”“透明壳”“MagSafe 壳”等标题近义、描述相似、功能接近的商品。

融合的目标不是让其中一路替代另一路，而是让结构相关和语义相近都进入候选池。工程里最稳妥的做法通常是：先各自召回，再合并去重，再统一归一化，最后融合打分。

全文主线公式是：

$$
s(q,i)=\alpha \tilde s_g(q,i)+\beta \tilde s_v(q,i), \quad \alpha+\beta=1
$$

其中 $q$ 是查询对象，$i$ 是候选物品，$\tilde s_g$ 是归一化后的图分数，$\tilde s_v$ 是归一化后的向量分数，$\alpha$ 和 $\beta$ 是两路信号的权重。

| 方法 | 主要依据 | 擅长问题 | 典型输出 |
| --- | --- | --- | --- |
| 图召回 | 边、路径、邻居、共现 | 结构相关、可解释关系 | 同品牌、共购、搭配、同类目候选 |
| 向量召回 | Embedding 相似度 | 语义泛化、内容相似 | 标题相近、描述相似、图片风格接近候选 |
| 融合召回 | 多路候选与统一打分 | 兼顾覆盖率与相关性 | 合并后的候选集合及融合分数 |

---

## 问题定义与边界

图召回是从图结构中找候选。图是由点和边组成的数据结构，点可以表示用户、商品、品牌、类目，边可以表示点击、购买、收藏、同品牌、属于某类目。图召回的直观含义是：从当前查询对象出发，沿着边或路径找到相关对象。

向量召回是从向量索引中找候选。向量索引是为大量 Embedding 建立的检索结构，用来快速找到 top-k 相似对象。向量召回的直观含义是：把查询和候选都变成向量，然后找距离近或夹角小的对象。

本文讨论的是召回融合，不是完整排序系统。召回的职责是从海量物品中快速找出几百到几千个候选；排序的职责是对这些候选做更精细的点击率、转化率或满意度预估；重排的职责是处理多样性、业务规则、去重、打散等最终展示约束。

在电商场景里，图召回可能来自“用户-商品-品牌-类目-共购”图。向量召回可能来自商品标题、描述、图片 Embedding。新手可以理解为：图召回是在找“有路能走到的东西”，向量召回是在找“看起来像的东西”。

本文使用如下符号：

| 符号 | 含义 |
| --- | --- |
| $q$ | 查询对象，可以是用户、商品、搜索词或当前上下文 |
| $i$ | 候选物品 |
| $C_g$ | 图召回得到的候选集合 |
| $C_v$ | 向量召回得到的候选集合 |
| $C=C_g \cup C_v$ | 两路候选合并后的集合 |

边界表如下：

| 概念 | 输入 | 输出 | 优势 | 局限 | 适用阶段 |
| --- | --- | --- | --- | --- | --- |
| 图召回 | 图节点、边、路径、交互记录 | 结构相关候选 | 可解释，关系明确，适合共购和搭配 | 对新物品和稀疏图不稳定 | 召回 |
| 向量召回 | 文本、图片、行为序列 Embedding | 语义相似候选 | 泛化强，适合新品和长尾内容 | 可解释性弱，依赖模型质量 | 召回 |
| 分数融合 | 两路召回分数 | 融合后的候选排序 | 简单稳定，易调参 | 表达能力有限 | 召回后粗排 |
| 特征融合 | Embedding、图特征、召回分数 | 模型预测分数 | 可学习非线性关系 | 依赖训练数据和线上一致性 | 排序或粗排 |

---

## 核心机制与推导

图分数 $s_g(q,i)$ 可以有多种来源。最简单的是邻居命中次数，例如同一批用户共同购买过 $q$ 和 $i$ 的次数。更复杂的方式包括路径权重、共现次数、Personalized PageRank、Node Similarity、随机游走结果。Node Similarity 是衡量两个节点邻居集合是否相似的一类图算法。

向量分数 $s_v(q,i)$ 通常来自余弦相似度。余弦相似度是用两个向量夹角衡量相似程度的指标，值越大表示方向越接近：

$$
s_v(q,i)=\cos(e_q,e_i)=\frac{e_q \cdot e_i}{\|e_q\|\|e_i\|}
$$

其中 $e_q$ 是查询向量，$e_i$ 是候选向量，$e_q \cdot e_i$ 是点积，$\|e_q\|$ 是向量长度。

图分数和向量分数通常不能直接相加，因为它们的数值分布不同。图分数可能是共现次数，范围从 0 到几万；向量分数可能是余弦相似度，范围大多在 0 到 1。即使两者都在 0 到 1，也不代表同一个数值有同等意义。图分数 0.8 可能表示强共购关系，向量分数 0.8 可能只是标题都包含相同关键词。

因此需要先做归一化。归一化是把不同来源的数值转换到可比较尺度。常用方法包括 min-max、z-score、分位数归一化、按召回通道做校准等。记归一化后的分数为：

$$
\tilde s_g(q,i), \quad \tilde s_v(q,i)
$$

然后做分数融合：

$$
s(q,i)=\alpha \tilde s_g(q,i)+\beta \tilde s_v(q,i), \quad \alpha+\beta=1
$$

玩具例子：查询是“手机壳”，候选 A 是同品牌同型号手机壳，候选 B 是标题很像但型号不完全匹配的保护套。

| 候选 | 原始图分数 | 原始向量分数 | 归一化图分数 | 归一化向量分数 |
| --- | ---: | ---: | ---: | ---: |
| A | 920 | 0.40 | 0.92 | 0.40 |
| B | 450 | 0.88 | 0.45 | 0.88 |

取 $\alpha=0.6,\beta=0.4$：

$$
s(A)=0.6\times0.92+0.4\times0.40=0.712
$$

$$
s(B)=0.6\times0.45+0.4\times0.88=0.622
$$

A 排在 B 前，不是因为 A 在每一项都更强，而是因为当前业务更看重结构关系。新手可以理解为：不是某一门课最高就一定总分第一，而是看加权后的总分。

另一类融合是特征融合。特征是模型输入中的可计算字段，例如价格、类目、图路径长度、向量相似度。特征融合不直接手写加权公式，而是构造特征向量：

$$
x=[e_q;e_i;\tilde s_g;\tilde s_v;d_{\text{path}};\deg]
$$

其中 $d_{\text{path}}$ 是路径距离，$\deg$ 是节点度数，也就是一个节点连接了多少条边。然后用模型学习分数：

$$
s=f(x)
$$

分数融合适合早期系统、数据较少、需要可解释调参的阶段。特征融合适合有训练样本、有稳定特征平台、需要学习复杂关系的阶段。

---

## 代码实现

最小工程流程可以拆成四步：图召回、向量召回、候选合并去重、统一打分排序。下面的代码是可运行的简化版本，展示核心结构。

```python
from math import sqrt

alpha = 0.6
beta = 0.4

ITEMS = {
    "A": {"title": "iphone phone case", "vec": [1.0, 0.1], "graph": 920},
    "B": {"title": "transparent protective cover", "vec": [0.6, 0.8], "graph": 450},
    "C": {"title": "charging cable", "vec": [0.2, 0.9], "graph": 300},
}

QUERY_VEC = [1.0, 0.0]

def graph_recall(query):
    return ["A", "B"]

def vector_recall(query):
    return ["B", "C", "A"]

def dedupe(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def graph_score(query, item):
    return ITEMS[item]["graph"]

def vector_score(query, item):
    v = ITEMS[item]["vec"]
    dot = sum(a * b for a, b in zip(QUERY_VEC, v))
    q_norm = sqrt(sum(a * a for a in QUERY_VEC))
    v_norm = sqrt(sum(a * a for a in v))
    return dot / (q_norm * v_norm)

def normalize_graph(score):
    return score / 1000.0

def normalize_vector(score):
    return score

def sort_by_score(items):
    return sorted(items, key=lambda x: x["score"], reverse=True)

def retrieve(query):
    cg = graph_recall(query)
    cv = vector_recall(query)
    candidates = dedupe(cg + cv)

    scored = []
    for item in candidates:
        sg = graph_score(query, item)
        sv = vector_score(query, item)
        ng = normalize_graph(sg)
        nv = normalize_vector(sv)
        score = alpha * ng + beta * nv
        scored.append({"item": item, "score": score, "ng": ng, "nv": nv})

    return sort_by_score(scored)

result = retrieve("phone case")
assert len(result) == 3
assert result[0]["item"] == "A"
assert len({x["item"] for x in result}) == len(result)
```

这段代码对应的调用时序是：

| 步骤 | 函数 | 作用 |
| --- | --- | --- |
| 1 | `graph_recall` | 从图关系中找候选 |
| 2 | `vector_recall` | 从向量相似度中找候选 |
| 3 | `dedupe` | 对 $C_g \cup C_v$ 去重 |
| 4 | `normalize_graph` / `normalize_vector` | 把两路分数转换到统一尺度 |
| 5 | `sort_by_score` | 按融合分数排序 |

真实工程例子：电商推荐首页可以同时跑多路召回。图召回从用户最近购买商品出发，沿着“同品牌”“同类目”“共购”“一起加购”边找候选。向量召回把用户最近浏览序列编码成用户兴趣向量，再去商品向量索引里查 top-k。两路结果合并后，先按商品 ID 去重，再过滤缺货、低质量、不可售商品，最后进入粗排或精排。

工程上建议把召回逻辑和融合逻辑拆开。召回逻辑负责“拿到候选”，融合逻辑负责“给候选打统一分”。这样排查问题时可以分别看：某一路是否召回不足、是否重复过多、分数是否异常、融合权重是否偏向某一路。

---

## 工程权衡与常见坑

融合系统的主要风险通常不是公式写错，而是数据口径和系统行为不一致。图分数和向量分数来自不同机制，分布不一致是常态。直接相加会让数值范围更大的通道支配结果。

同一个商品也可能同时被图通道和向量通道召回。如果不先去重，列表里可能出现重复商品，或者在后续统计中被当成两个候选，放大曝光概率。新手可以理解为：两个入口都把同一件商品送来了，先去重再排队才合理。

图召回还要注意时间泄漏。时间泄漏是训练或评估时使用了未来才发生的信息。例如用 4 月 10 日之后的购买边去评估 4 月 1 日的推荐效果，就会高估模型能力。图边、共现窗口、训练样本必须按业务时间切开。

向量召回要单独调 ANN 参数。ANN 是近似最近邻检索，用更低延迟换取可接受的召回误差。HNSW 和 IVF 是常见 ANN 索引结构。HNSW 通常通过图结构近似找近邻，IVF 通常先聚类再在部分桶内搜索。参数过小会漏候选，参数过大会增加延迟和成本。

| 问题 | 表现 | 原因 | 规避方法 |
| --- | --- | --- | --- |
| 分数直接相加 | 某一路长期支配排序 | 分布、量纲、含义不同 | 分通道归一化或校准 |
| 只加权不去重 | 重复商品增多，多样性下降 | 多路召回命中同一候选 | 先 union，再按 ID 去重 |
| 图召回未来边泄漏 | 离线效果虚高，线上下降 | 使用了评估时间之后的边 | 按时间窗构图和抽样 |
| ANN 参数失衡 | 延迟高或召回不足 | HNSW / IVF 参数未单独调 | 分别压测召回率、延迟、成本 |
| 线上线下口径不一致 | 离线提升，线上无效 | 特征、候选、过滤规则不同 | 固化特征定义和候选生成版本 |

权重 $\alpha,\beta$ 不应该只凭感觉设置。早期可以用人工评估或小流量实验确定范围；有点击、收藏、购买等反馈后，可以用 A/B 实验比较不同权重。A/B 实验是把用户随机分成不同组，分别使用不同策略，再比较真实业务指标的方法。

---

## 替代方案与适用边界

不是所有场景都需要图召回和向量召回融合。融合会带来额外的索引、存储、调参、监控和排障成本。是否融合，要看业务目标更需要关系、相似、解释性、冷启动，还是低延迟。

如果业务非常强调明确关系，例如“同品牌配件推荐”“买手机后推荐保护膜和充电器”，图召回往往更直接。因为路径和边本身就是业务解释。如果业务更依赖文本和内容泛化，例如“新品相似推荐”“文章相关推荐”“图片风格检索”，向量召回更合适。新手可以理解为：看你更需要“关系”，还是更需要“相似”。

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 纯图召回 | 可解释，关系明确，适合搭配和共购 | 稀疏图、新品、冷启动较弱 | 配件推荐、关系链推荐、老用户行为充分场景 |
| 纯向量召回 | 泛化强，适合文本、图片、长尾内容 | 可解释性弱，依赖 Embedding 质量 | 新品相似、内容推荐、语义搜索 |
| 规则召回 | 简单可控，容易上线 | 覆盖有限，维护成本随规则增加 | 冷启动兜底、运营强约束场景 |
| 召回后学习排序 | 能学习多特征组合 | 依赖样本、特征平台和在线一致性 | 候选量可控、反馈数据充足场景 |
| 图嵌入 + 向量混合 | 同时编码结构和语义 | 训练复杂，解释和调试成本高 | 大规模推荐、图数据和内容数据都丰富的场景 |

一个常见演进路径是：先用规则和单路召回跑通业务，再加入图召回提高结构覆盖，随后加入向量召回提高语义泛化，最后用排序模型学习融合。不要在数据和监控还不稳定时直接上复杂模型，否则问题很难定位。

---

## 参考资料

参考资料对应关系如下：

| 资料 | 支撑内容 |
| --- | --- |
| [Neo4j Vector Index and Search](https://neo4j.com/developer/genai-ecosystem/vector-search/) | 向量索引、top-k、cosine / euclidean、HNSW |
| [Neo4j Graph Data Science Introduction](https://neo4j.com/docs/graph-data-science/current/introduction/) | 图算法、图嵌入、链路预测、图数据科学流程 |
| [Recommending on graphs: a comprehensive review from a data perspective](https://link.springer.com/article/10.1007/s11257-023-09359-w) | 图推荐综述、结构信息、random walk、network embedding |
| [A systematic review and research perspective on recommender systems](https://link.springer.com/article/10.1186/s40537-022-00592-5) | hybrid filtering、weighted hybridization、feature-combination |
| [Faiss Documentation](https://faiss.ai/) | dense vector search、ANN、MIPS、向量检索工程基础 |

1. [Neo4j Vector Index and Search](https://neo4j.com/developer/genai-ecosystem/vector-search/)
2. [Neo4j Graph Data Science Introduction](https://neo4j.com/docs/graph-data-science/current/introduction/)
3. [Recommending on graphs: a comprehensive review from a data perspective](https://link.springer.com/article/10.1007/s11257-023-09359-w)
4. [A systematic review and research perspective on recommender systems](https://link.springer.com/article/10.1186/s40537-022-00592-5)
5. [Faiss Documentation](https://faiss.ai/)
