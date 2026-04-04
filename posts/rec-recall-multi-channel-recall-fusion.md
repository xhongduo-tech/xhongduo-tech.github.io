## 核心结论

多路召回的融合策略，核心不是“把每一路结果求并集”，而是把不同召回通道产出的候选，变成一套可以公平竞争、可控分配、能稳定降级的统一输出。

“召回通道”可以理解为不同找候选的方法，比如关键词召回、向量召回、协同过滤召回、规则召回。它们各自擅长的方向不同，但原始分数通常不可直接比较。关键词通道的 `0.6`，不一定比向量通道的 `0.4` 更差；有时只是量纲不同，也就是“分数的刻度尺不同”。

因此，融合层通常要同时解决三类问题：

| 问题 | 本质 | 不处理会怎样 |
|---|---|---|
| 分数不可比 | 不同通道的打分尺度不同 | 某一路长期垄断排序 |
| 重复候选过多 | 同一 item 被多路同时召回 | 热门结果重复挤占曝光 |
| 流量分配失衡 | 强势通道吃掉大部分名额 | 多样性、新颖性下降 |

实践里常见两大路线：

| 路线 | 核心思想 | 适合场景 |
|---|---|---|
| 归一化 + 加权融合 | 先把分数投到近似可比空间，再按权重求和 | 各通道分数较稳定，可做质量校准 |
| Rank Fusion / RRF | 不直接用原始分数，只看名次和跨路一致性 | 分数体系差异很大，难做统一标定 |

一个最重要的工程结论是：融合层必须和 `dedup`、`quota`、`fallback` 一起设计。`dedup` 是去重，也就是同一内容只保留一次；`quota` 是配额，意思是给某些通道保留最少或最多名额；`fallback` 是降级，意思是某路超时或异常时，系统切到备选策略。没有这三件事，融合公式再漂亮，也会在真实流量里失效。

---

## 问题定义与边界

多路召回融合，指的是把多个召回通道返回的候选集合合并成一个统一候选池，并给出一个最终排序，用于进入粗排、精排，或者直接展示。

这里的边界要先说清楚：

1. 讨论的是“召回层后的融合”，不是精排模型本身。
2. 目标不是让每一路平均分流，而是在质量、多样性、覆盖率之间做平衡。
3. 融合解决的是“候选整合问题”，不替代后续更细粒度的排序模型。

对零基础读者，最容易误解的是“直接把所有结果拼起来，再按分数排”。这个做法通常不对，因为不同通道的分数来源不同。

玩具例子：

- 语义召回给文档 A 打分 `0.91`
- 关键词召回给文档 B 打分 `12.7`

如果直接比较，B 永远在 A 前面。但这里 `12.7` 可能是 BM25 分数，`0.91` 可能是余弦相似度，它们不是同一把尺子。白话说，两个分数都在表达“相关”，但表达方式不一样，不能直接相减比大小。

多路召回融合常见边界问题如下：

| 边界问题 | 说明 | 典型处理 |
|---|---|---|
| 跨通道分数不一致 | 分数范围、分布、含义都不同 | 归一化、校准、改用 RRF |
| 重复 item | 同一文档被多路召回 | 按 `item_id` 去重并聚合证据 |
| 通道流量倾斜 | 强通道长期占满 topN | 配额、权重约束、分层截断 |
| 通道超时 | 某一路没按时返回 | 超时降级、保底通道兜底 |
| 通道质量漂移 | 模型升级后分布变化 | 在线监控、重标定、动态权重 |
| 新内容冷启动 | 新 item 缺少行为特征 | 给规则路或内容路保底流量 |

真实工程里，一个常见组合是：

- 倒排/BM25 负责“精确命中”
- 向量召回负责“语义扩展”
- 热门召回负责“保底可展示”
- 规则召回负责“业务约束”

如果没有融合控制，BM25 可能因为词面精准占满前列，向量召回虽然带来新颖性，却进不了最终列表；反过来，如果向量通道权重过大，也可能把词面高度相关的结果挤掉，导致点击率下降。

所以问题定义不能只看“谁分高”，而要看“如何让不同能力的通道共同产出稳定结果”。

---

## 核心机制与推导

### 1. 分数归一化

“归一化”就是把不同通道的原始分数映射到更可比的区间。白话说，是先把不同单位的尺子换成接近同一单位。

常见方法有三类。

第一类，min-max 归一化：

$$
\hat s^{\text{minmax}}_{d,j}=\frac{s_{d,j}-\min_j}{\max_j-\min_j}
$$

它把通道 $j$ 内的分数压到 $[0,1]$。优点是直观，缺点是对极端值敏感。

第二类，z-score 标准化：

$$
\hat s^z_{d,j}=\frac{s_{d,j}-\mu_j}{\sigma_j}
$$

这里 $\mu_j$ 是均值，$\sigma_j$ 是标准差。它的意思是“这个候选高于本通道平均水平多少个标准差”。适合分布相对稳定的场景，但如果原始分布很偏，效果会抖。

第三类，分位数归一化：

$$
\hat s^q_{d,j}=\mathrm{Quantile}_j(s_{d,j})
$$

“分位数”可以理解为“这个分数在本通道排到前百分之多少”。它比 min-max 更稳，因为不太受单个极端大值影响。

归一化后，再做加权融合：

$$
S_d=\sum_j \alpha_j \hat s_{d,j}
$$

其中 $\alpha_j$ 是通道权重，可以理解为“这个通道有多可信”。权重可以人工设定，也可以通过历史点击率、转化率、覆盖率等指标学习得到。

### 2. Rank Fusion

如果分数实在不可比，或者每次分数分布都漂移很大，就不要强行比较原始值，而是直接比较排名。

最常见的是 RRF，Reciprocal Rank Fusion：

$$
S_d=\sum_j \frac{\alpha_j}{k+\mathrm{rank}_j(d)}
$$

这里：

- $\mathrm{rank}_j(d)$ 表示文档 $d$ 在通道 $j$ 的名次
- $k$ 是平滑常数，通常取几十
- $\alpha_j$ 是通道权重

这个公式的直觉很简单：在多个通道里都排得靠前的 item，会得到更高总分。它奖励“跨路一致性”，也就是多个通道都认为不错的候选。

### 3. 玩具例子

假设有两个通道 A 和 B，两个候选 X 和 Y。

原始分数如下：

| 候选 | A 原始分数 | B 原始分数 |
|---|---|---|
| X | 0.8 | 0.9 |
| Y | 0.5 | 0.4 |

对每个通道做 min-max：

- A 归一化后：X = 1，Y = 0
- B 归一化后：X = 1，Y = 0

如果两路权重相等，$\alpha_A=\alpha_B=1$，则：

- $S_X = 1 + 1 = 2$
- $S_Y = 0 + 0 = 0$

这个例子很简单，但它说明了一件关键事实：融合层不是为了改变一切排序，而是为了把不同来源的证据放进同一套决策框架。

再看一个更接近真实的例子。假设：

- 通道 A 是 BM25，返回 D1、D2、D3
- 通道 B 是向量召回，返回 D2、D4、D1

如果用 RRF，D1 和 D2 因为在两路都出现，会得到双重加分；D3、D4 只在单一路命中，得分较低。于是最终结果更偏向“多通道都认可”的候选。这种机制在分数不可比时，比直接加权原始分数稳得多。

---

## 代码实现

下面给一个可运行的 Python 简化实现，包含四件事：

1. 按通道做 min-max 归一化
2. 做加权融合
3. 按 `item_id` 去重聚合
4. 做简单配额控制

```python
from collections import defaultdict

def minmax_normalize(channel_items):
    scores = [x["score"] for x in channel_items]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return {x["item_id"]: 1.0 for x in channel_items}
    return {
        x["item_id"]: (x["score"] - lo) / (hi - lo)
        for x in channel_items
    }

def fuse_by_weight(channels, weight_map):
    # channels: {"bm25": [{"item_id": "A", "score": 12.0}, ...], ...}
    fused = defaultdict(lambda: {"score": 0.0, "hits": set(), "per_channel": {}})

    for channel_name, items in channels.items():
        norm_scores = minmax_normalize(items)
        w = weight_map.get(channel_name, 1.0)
        for item in items:
            item_id = item["item_id"]
            s = norm_scores[item_id]
            fused[item_id]["score"] += w * s
            fused[item_id]["hits"].add(channel_name)
            fused[item_id]["per_channel"][channel_name] = s

    result = []
    for item_id, info in fused.items():
        # 跨路一致性奖励：被多路召回的 item 额外加一点分
        bonus = 0.05 * (len(info["hits"]) - 1)
        result.append({
            "item_id": item_id,
            "score": info["score"] + bonus,
            "hits": sorted(info["hits"]),
            "per_channel": info["per_channel"],
        })

    result.sort(key=lambda x: (-x["score"], x["item_id"]))
    return result

def apply_quota(ranked_items, source_priority, min_quota_map, topn):
    # ranked_items 里假设每个 item 有 hits 字段，source_priority 决定该 item 归属哪一路
    chosen = []
    used = set()
    quota_count = {k: 0 for k in min_quota_map}

    # 先满足保底配额
    for channel in source_priority:
        for item in ranked_items:
            if item["item_id"] in used:
                continue
            if channel in item["hits"] and quota_count.get(channel, 0) < min_quota_map.get(channel, 0):
                chosen.append(item)
                used.add(item["item_id"])
                quota_count[channel] += 1
                if len(chosen) >= topn:
                    return chosen

    # 再按总分补齐
    for item in ranked_items:
        if item["item_id"] in used:
            continue
        chosen.append(item)
        used.add(item["item_id"])
        if len(chosen) >= topn:
            break

    return chosen

channels = {
    "bm25": [
        {"item_id": "D1", "score": 15.0},
        {"item_id": "D2", "score": 13.0},
        {"item_id": "D3", "score": 8.0},
    ],
    "vector": [
        {"item_id": "D2", "score": 0.92},
        {"item_id": "D4", "score": 0.88},
        {"item_id": "D1", "score": 0.80},
    ],
    "popular": [
        {"item_id": "D5", "score": 1000},
        {"item_id": "D1", "score": 900},
    ]
}

weights = {"bm25": 1.0, "vector": 1.1, "popular": 0.3}
ranked = fuse_by_weight(channels, weights)
final_items = apply_quota(
    ranked_items=ranked,
    source_priority=["bm25", "vector", "popular"],
    min_quota_map={"bm25": 1, "vector": 1},
    topn=4,
)

assert ranked[0]["item_id"] in {"D1", "D2"}
assert any("vector" in x["hits"] for x in final_items)
assert len({x["item_id"] for x in final_items}) == len(final_items)
assert len(final_items) == 4

print(final_items)
```

这段代码是简化版，但已经体现出真实系统的骨架：

- 每路先独立归一化
- 用权重聚合分数
- 同一 `item_id` 聚合为一条，避免重复
- 用最小配额保证某些通道不会被完全挤掉

如果你更偏向排名融合，可以把上面的加权分数换成 RRF。RRF 实现也很短：

```python
def rrf_fuse(channels, weight_map, k=60):
    fused = defaultdict(float)
    for channel_name, items in channels.items():
        w = weight_map.get(channel_name, 1.0)
        for rank, item in enumerate(items, start=1):
            fused[item["item_id"]] += w / (k + rank)
    result = [{"item_id": item_id, "score": score} for item_id, score in fused.items()]
    result.sort(key=lambda x: (-x["score"], x["item_id"]))
    return result

rrf_result = rrf_fuse(channels, {"bm25": 1.0, "vector": 1.0, "popular": 0.2}, k=10)
assert len(rrf_result) >= 5
assert rrf_result[0]["score"] >= rrf_result[-1]["score"]
```

真实工程例子：

在 RAG 检索增强生成里，经常会同时用 `BM25 + 向量召回`。BM25 擅长抓关键词强相关段落，向量召回擅长补充语义近邻段落。融合层如果只信 BM25，会丢掉语义扩展；只信向量，又会把一些关键词精确命中的法规、代码、接口名压下去。所以通常先用 RRF 或归一化加权做第一轮融合，再配合 `topK per channel` 和去重，最后把候选送进 reranker。

---

## 工程权衡与常见坑

融合策略真正难的地方，不在公式，而在工程约束。

第一类坑，是“高分路独占”。

这通常发生在某一路分数分布更陡，或者权重过大。结果是这一路几乎占满 topN，其它通道失去存在感。表面上看 CTR 可能短期上涨，但长期往往会损失覆盖率和新颖性。

第二类坑，是“去重太晚”。

如果先截断再去重，同一个热门内容可能在多个通道都占坑，最后实际有效候选变少。正确做法通常是先按统一主键聚合，再做全局排序和截断。

第三类坑，是“配额过死”。

配额能防止通道被挤掉，但如果把 quota 写得太硬，会让低质量通道强行占位。配额应该是软约束或最小保底，而不是无条件固定占比。

第四类坑，是“降级缺失”。

真实线上环境里，向量服务、图召回服务、特征服务都有可能超时。如果没有降级，最终列表可能严重抖动，甚至直接空结果。

常见坑与规避方式如下：

| 常见坑 | 现象 | 规避策略 |
|---|---|---|
| 直接求并集 | 强势通道天然占优 | 做归一化或 RRF |
| 只做静态加权 | 分布漂移后排序失真 | 定期重标定，在线监控 |
| 去重放在最后 | 热门 item 重复占坑 | 先聚合后排序 |
| 没有保底 quota | 某些通道长期没流量 | 给关键通道最小配额 |
| quota 太死 | 低质量通道硬塞结果 | 软配额或阈值配额 |
| 没有超时降级 | 某路异常导致整体抖动 | fallback 到稳态通道 |
| 只看点击率调权 | 越调越头部化 | 同时监控覆盖率和多样性 |

一个很典型的新手错误是：看到向量召回点击率不如 BM25，就直接把它权重调低。这样短期可能没问题，但长期会让系统越来越依赖头部 query 和词面匹配，导致尾部 query、同义表达、长文本理解能力下降。更稳妥的做法是：

- 保留向量路最小曝光
- 单独看尾部 query 指标
- 分 query 类别调权，而不是全局一刀切

再看一个真实工程场景。假设你做内容推荐：

- 协同过滤通道命中率高，但偏历史热门
- 内容向量通道能补新内容，但精度略低
- 编辑规则通道用于活动和运营位

如果没有配额和降级，协同过滤会因为历史点击数据强，长期吃掉大多数流量，新内容越来越难拿到反馈，系统进入“强者愈强”的闭环。这个问题不是排序模型能单独解决的，必须在融合层就做流量治理。

---

## 替代方案与适用边界

多路召回融合没有唯一正确答案，只有和业务条件匹配的方案。

几种常见方案可以这样理解：

| 方案 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|
| 归一化 + 加权 | 直观，可结合质量学习权重 | 依赖分数稳定性 | 各通道分数可标定 |
| RRF | 不依赖原始分值，稳健 | 丢掉部分分值细节 | 分数不可比或波动大 |
| 学习型融合 | 能自动学复杂关系 | 依赖标注和特征稳定 | 数据充分、迭代能力强 |
| 分层融合 | 先分组再合并，便于治理 | 规则复杂 | 业务约束多的场景 |

如果你刚起步，建议优先顺序通常是：

1. 先做 `dedup + quota + timeout fallback`
2. 分数实在不可比时，优先上 RRF
3. 分数分布稳定后，再尝试归一化 + 加权
4. 数据量足够，再做学习权重或轻量学习融合

这里还有一个常见边界：某一路是否应该“硬保留”。

答案是，不一定。若某个通道长期质量很差、命中极少、还经常超时，它更适合做“可选通道”，而不是“必保通道”。白话说，它可以在其它通道候选不足时补位，而不是正常情况下强占名额。

例如：

- 图谱召回只有在实体类 query 下价值高
- 热门召回只有在冷启动或空召回时价值高
- 规则召回只有在活动期才应该提升优先级

这说明融合策略最好不是全局固定一套，而是按场景分治。常见分法包括：

- 按 query 类型分
- 按用户冷启动状态分
- 按内容品类分
- 按线上服务健康度分

当系统成熟后，融合权重 $\alpha_j$ 与配额 `quota` 往往都不是常量，而是动态策略。例如新用户提高热门路和内容路配额，老用户提高个性化路权重；向量服务延迟升高时，自动降低其参与度。这才是“工程可用”的融合，而不是只在离线表格里好看。

---

## 参考资料

| 资源 | 主题 | 说明 |
|---|---|---|
| [cnblogs: 多路召回架构与融合](https://www.cnblogs.com/alohablogs/p/19815342?utm_source=openai) | 多路召回整体设计 | 适合理解多通道召回、融合、去重和工程流程 |
| [axi.moe: Rank Fusion 与归一化](https://axi.moe/article/rn9lt25e/?utm_source=openai) | 分数归一化、RRF | 给出 min-max、z-score、RRF 等核心公式 |
| [RAG Academy](https://www.ragacademy.space/?utm_source=openai) | RAG 检索工程 | 可用于理解 BM25、向量召回、多路融合在 RAG 中的工程落地 |
| Cormack et al., Reciprocal Rank Fusion | Rank Fusion | RRF 的经典论文来源，适合追溯方法原理 |
| Manning 等《Introduction to Information Retrieval》 | 检索基础 | 适合补 BM25、排序、评价指标等背景知识 |
