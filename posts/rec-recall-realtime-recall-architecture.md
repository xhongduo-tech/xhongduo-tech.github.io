## 核心结论

实时召回的目标，不是“把所有数据都做成实时”，而是把**对结果影响最大的最新变化**尽快变成可检索候选。召回可以先理解成“从海量内容里先捞出一小批可能相关的候选”，它是排序之前的第一层筛选。

工程上最常见、也最稳妥的方案，不是单一索引，而是**热索引 + 冷索引**双路架构：

- 热索引负责最近新增、最近活跃、最近高价值的数据，追求秒级甚至亚秒级可见。
- 冷索引负责全量历史数据，追求低成本、高容量和稳定吞吐。

它解决的是三个目标之间的冲突：

| 目标 | 白话解释 | 为什么冲突 |
|---|---|---|
| Freshness | 新数据多久能被搜到 | 更新越频繁，写入和索引维护越重 |
| 吞吐 | 单位时间能处理多少写入和查询 | 吞吐高时，索引更新和合并更容易拥塞 |
| 延迟 | 一次查询多久返回 | 为了低延迟，通常不能等待慢路径完成 |

一个常用的近似表达是：

$$
F = \frac{1}{\Delta t_{update}}
$$

其中 $F$ 表示 Freshness，$\Delta t_{update}$ 表示一次更新从产生到可查询的时间间隔。间隔越短，新鲜度越高。

玩具例子：用户刚点击了一篇文章，这个点击事件进入流系统，经过特征计算和向量编码后，被写入热索引。下一次这个用户再请求推荐时，系统同时查询热索引和冷索引，再把结果合并。如果热索引暂时超时，就先返回冷索引结果，避免整个接口被拖慢。

真实工程例子：电商推荐里，“刚刚爆发点击的商品”需要几秒内进入候选集，否则流量窗口会直接错过；但平台的历史商品库可能有数亿条，不可能全都靠高频动态维护一个高成本实时索引。因此通常把热门、最新、活跃商品放进热层，把全量商品保存在冷层，再做双路并发检索。

---

## 问题定义与边界

实时召回要解决的问题是：**行为流、内容变更、库存变化、价格变动、用户兴趣漂移，如何在秒级进入召回结果，而不是等下一次全量重建。**

这里的“实时”不是绝对零延迟，而是一个明确的端到端预算。总延迟通常可以拆成：

$$
L_{total} = L_{capture} + L_{encoding} + L_{index} + L_{merge}
$$

- $L_{capture}$：事件采集延迟，也就是日志从业务服务进入消息队列或流系统的时间。
- $L_{encoding}$：编码延迟，也就是把原始行为转成特征或向量的时间。
- $L_{index}$：索引写入延迟，也就是数据真正进入可检索结构的时间。
- $L_{merge}$：查询合并延迟，也就是热冷结果汇总、去重、截断的时间。

一个典型边界如下：

| 层级 | 典型规模 | 更新频率 | 查询延迟目标 | 适合结构 |
|---|---|---|---|---|
| 热索引 | 百万到千万级 | 秒级或更快 | 1-10 ms | HNSW、内存倒排、轻量向量图 |
| 冷索引 | 亿级到十亿级 | 分钟到小时级 | 15-30 ms | IVF-PQ、磁盘 ANN、批式倒排 |

这意味着实时召回不是“单机加缓存”问题，而是一个跨链路设计问题。它至少包含以下边界：

1. 热数据必须支持高频增量写入。
2. 冷数据必须支持大规模低成本存储。
3. 查询路径必须允许异步合并，不能因为一条慢链路拖垮主请求。
4. 读写一致性通常不是强一致，而是“写完后在可接受时间内可见”的工程一致性。

可以把它画成一个简单流程：

`行为事件 -> 流式采集 -> 特征/向量化 -> 热索引`
`历史/全量数据 -> 批处理 -> 冷索引`
`查询请求 -> 并发查询热/冷 -> 合并去重 -> 返回候选`

新手常见误解是：只要索引支持插入，就是实时召回。这个理解不完整。真正的实时召回，要求整条链路都可控：事件不能丢、编码不能太慢、索引更新不能阻塞、查询不能等最慢分支。

---

## 核心机制与推导

核心机制是**分层更新**与**异步合并**。

先看分层更新。设热层更新周期为 $\tau_1$，冷层更新周期为 $\tau_2$，通常有：

$$
\tau_1 \ll \tau_2
$$

也就是热层频繁更新，冷层低频批量更新。这样做的直觉很简单：越新的数据越少、越重要，适合高成本维护；越旧的数据越多、越稳定，适合低成本压缩。

如果把整体吞吐近似记成 $T \approx QPS$，那么双层设计的意义在于：

- 热层承担高 Freshness；
- 冷层承担高容量；
- 查询时并发发出，避免总吞吐被慢更新链路拖垮。

从工程角度看，热层和冷层不是“主备关系”，而是“职责分工关系”。

| 维度 | 热索引 | 冷索引 |
|---|---|---|
| 主要职责 | 保证最新变化可见 | 保证全量覆盖 |
| 数据特点 | 新、热、活跃 | 老、全、稳定 |
| 更新方式 | 流式增量 | 批量构建/合并 |
| 常见结构 | HNSW | IVF-PQ |
| 主要风险 | 动态维护复杂 | 召回率下降、更新慢 |

为什么很多系统让热层偏向 HNSW、冷层偏向 IVF-PQ？

- HNSW 可以理解成“多层近邻图”，查询速度快，适合低延迟高召回，但动态插入、删除和图维护更复杂，内存成本也高。
- IVF-PQ 可以理解成“先粗分桶，再做压缩编码”，更省内存，适合超大规模，但压缩会损失部分召回率，而且增量更新不如纯内存结构灵活。

因此常见组合是：**热层 HNSW，冷层 IVF-PQ**。

玩具例子：  
假设热数据 1000 万条，查询 1-10 ms；冷数据 5 亿条，用 IVF-PQ 压缩后查询 15-25 ms。路由器收到请求后同时发给两边，设置总等待上限 50 ms。如果热层 8 ms 返回、冷层 22 ms 返回，就合并；如果热层因写入抖动超过 50 ms，就直接返回冷层候选，保证接口稳定。

这里还有一个关键点：**合并不是简单拼接**。至少要做三件事：

1. 去重：同一个 item 可能同时存在热层和冷层。
2. 重排：热层结果不一定天然优先，要看业务打分。
3. 可见性控制：如果一个 item 已写热层但还没进入冷层，查询时不能因为冷层旧版本覆盖掉新版本。

所以“写完可见”通常需要版本号、时间戳或 sequence id。白话说，就是每条数据都要带一个“谁更新得更晚”的标记，合并时只保留最新版本。

---

## 代码实现

下面用一个可运行的 Python 玩具实现，展示热冷索引抽象、并发查询、超时降级和版本合并。这里不实现真实 ANN，只模拟实时召回里的控制逻辑。

```python
import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Item:
    item_id: str
    score: float
    version: int


class BaseIndex:
    def search(self, query: str, topk: int) -> List[Item]:
        raise NotImplementedError


class MockHotIndex(BaseIndex):
    def __init__(self, delay_ms: int, items: List[Item]):
        self.delay_ms = delay_ms
        self.items = items

    def search(self, query: str, topk: int) -> List[Item]:
        time.sleep(self.delay_ms / 1000.0)
        return sorted(self.items, key=lambda x: x.score, reverse=True)[:topk]


class MockColdIndex(BaseIndex):
    def __init__(self, delay_ms: int, items: List[Item]):
        self.delay_ms = delay_ms
        self.items = items

    def search(self, query: str, topk: int) -> List[Item]:
        time.sleep(self.delay_ms / 1000.0)
        return sorted(self.items, key=lambda x: x.score, reverse=True)[:topk]


def merge_results(hot_items: List[Item], cold_items: List[Item], topk: int) -> List[Item]:
    merged: Dict[str, Item] = {}
    for item in hot_items + cold_items:
        old = merged.get(item.item_id)
        if old is None or item.version > old.version or (
            item.version == old.version and item.score > old.score
        ):
            merged[item.item_id] = item
    return sorted(merged.values(), key=lambda x: x.score, reverse=True)[:topk]


def dual_search(hot_index: BaseIndex, cold_index: BaseIndex, query: str, topk: int, timeout_ms: int):
    results: Dict[str, List[Item]] = {"hot": [], "cold": []}
    threads = []

    def run_search(name: str, index: BaseIndex):
        try:
            results[name] = index.search(query, topk)
        except Exception:
            results[name] = []

    for name, index in [("hot", hot_index), ("cold", cold_index)]:
        t = threading.Thread(target=run_search, args=(name, index))
        t.start()
        threads.append(t)

    deadline = time.time() + timeout_ms / 1000.0
    for t in threads:
        remain = deadline - time.time()
        if remain > 0:
            t.join(remain)

    # 超时降级：谁先回来用谁；两边都回来则合并
    return merge_results(results["hot"], results["cold"], topk)


hot = MockHotIndex(
    delay_ms=8,
    items=[
        Item("A", 0.95, 3),
        Item("B", 0.90, 5),
    ],
)

cold = MockColdIndex(
    delay_ms=20,
    items=[
        Item("A", 0.80, 2),  # 旧版本，会被热索引覆盖
        Item("C", 0.85, 1),
    ],
)

res = dual_search(hot, cold, query="user_123", topk=3, timeout_ms=50)
assert [x.item_id for x in res] == ["A", "B", "C"]
assert res[0].version == 3

slow_hot = MockHotIndex(delay_ms=80, items=[Item("D", 0.99, 1)])
res2 = dual_search(slow_hot, cold, query="user_123", topk=2, timeout_ms=30)
assert [x.item_id for x in res2] == ["C", "A"]
```

这段代码表达了四个关键点：

1. 热冷索引都实现统一接口，方便路由层切换。
2. 查询并发发出，而不是串行等待。
3. 超时后不报错退出，而是降级为单路结果。
4. 合并时按 `version` 去重，避免旧数据覆盖新数据。

配置上，通常要把预算显式写出来，例如：

```python
CONFIG = {
    "hot_query_timeout_ms": 15,
    "cold_query_timeout_ms": 35,
    "router_total_timeout_ms": 50,
    "hot_update_interval_ms": 1000,
    "cold_batch_interval_sec": 1800,
    "topk_per_index": 200,
    "final_topk": 100,
}
```

真实工程例子里，链路会更长：

`stream_processor -> feature_store -> embedding_service -> hot_index_writer`

同时再跑一个离线任务：

`batch_job -> train/codebook -> rebuild_cold_index -> publish_snapshot`

也就是说，实时召回不是单个检索服务，而是一条**可观察的流水线**。最少要监控这些指标：

| 指标 | 含义 | 异常信号 |
|---|---|---|
| event lag | 事件积压时间 | 流消费跟不上 |
| encode latency | 向量编码耗时 | 模型服务过载 |
| hot insert lag | 热索引写入延迟 | 热层卡住 |
| merge timeout rate | 合并超时比例 | 某一路查询不稳定 |
| fresh hit ratio | 新数据命中率 | 实时链路形同虚设 |

---

## 工程权衡与常见坑

实时召回最容易失败的地方，不在“能不能查”，而在“能不能长期稳定查”。

第一类坑是 HNSW 动态维护。HNSW 查询快，但动态插入和删除并不便宜。很多系统采用软删，也就是先打删除标记，不立刻清理节点。如果长期不做 compaction，白话说就是“图里留下很多无效点”，搜索路径会越来越脏，尾延迟抬高。

第二类坑是热冷一致性。热层数据更新快，冷层更新慢，如果没有统一版本语义，查询时可能出现：

- 热层是新商品名，冷层还是旧商品名；
- 热层显示有库存，冷层显示无库存；
- 热层已删除，冷层还在返回。

这会直接造成“写完不可见”或“旧数据回流”。

第三类坑是错误的超时策略。很多人把热层当作强依赖，结果热层一抖动，整个召回接口就超时。正确做法是把超时当作设计的一部分，而不是异常路径。

| 常见坑 | 现象 | 缓解措施 |
|---|---|---|
| HNSW 软删积压 | 查询尾延迟升高 | 定期 compaction，重建前预热新图 |
| 热冷版本不一致 | 新旧数据混用 | 用版本号或 sequence 合并 |
| 热层超时拖垮主链路 | 接口 p99 激增 | 并发查询 + 总超时降级 |
| 冷层批更新过慢 | 全量覆盖不足 | 缩短批周期，分片发布快照 |
| 写入无背压 | 高峰期堆积 | 队列限流、批写、熔断 |

一个真实工程里的典型故障是：商品下架后，只在热层做了删除标记，但冷层快照要 30 分钟后才更新；查询合并时又没有版本过滤，于是已经下架的商品仍然会被冷层召回。这个问题不是 ANN 算法问题，而是索引发布协议问题。

因此上线前通常需要两步：

1. 新索引先预热。白话说，就是先让缓存、连接、图结构进入稳定状态，不要直接切线上流量。
2. 写入带同步标记。也就是保证每次更新都有统一版本，合并时能判断谁更新得更晚。

---

## 替代方案与适用边界

不是所有推荐系统都需要热冷双路。是否采用双路，主要看数据规模、实时性要求和团队运维能力。

| 方案 | 适用条件 | 延迟 | 成本 | 维护复杂度 |
|---|---|---|---|---|
| 热冷双路 | 数据大、实时性强 | 低到中 | 中到高 | 高 |
| 纯热单路 | 数据量不大、追求极低延迟 | 很低 | 中 | 中 |
| 纯冷单路 | 可容忍分钟级更新 | 中到高 | 低 | 低 |

纯热单路适合什么场景？  
小型推荐系统、内容库不到 2000 万、主要目标是快速上线，且机器内存预算还能覆盖全量向量。这时直接用 HNSW 或高性能内存索引就够了，架构更简单，排障也更容易。

纯冷单路适合什么场景？  
例如日报推荐、专题页推荐、知识库补充召回，这类任务可以接受分钟级甚至小时级更新，不需要为秒级新鲜度付出高复杂度。此时批量构建 IVF-PQ 或倒排索引更划算。

什么时候应该从单路升级到双路？

1. 新数据晚几分钟可见，已经明显影响业务指标。
2. 单热层内存成本持续上涨，开始压缩召回覆盖。
3. 高频写入导致查询抖动，单一索引无法同时兼顾写和读。

什么时候可以从双路回退到单路？

1. 数据规模明显下降。
2. 实时更新需求不再是核心目标。
3. 团队无法稳定维护双层发布、一致性和监控体系。

所以双路不是“高级架构标配”，而是当单路已经无法同时满足容量、延迟和新鲜度时的必要升级。

---

## 参考资料

1. What is Real-Time Incremental Indexing?  
重点：解释实时增量索引的定义，以及为什么“秒级可见”是一个链路目标，而不是单个索引能力。  
链接：https://www.systemoverflow.com/learn/ml-embeddings/realtime-embedding-updates/what-is-real-time-incremental-indexing

2. Near Real-Time Indexing Pipelines  
重点：介绍热/冷分层、近实时索引流水线，以及为什么常见系统会采用分层更新和异步合并。  
链接：https://apxml.com/courses/large-scale-distributed-rag/chapter-6-advanced-rag-architectures-techniques/rag-dynamic-streaming-data

3. Approximate Nearest Neighbor Search: HNSW vs IVF-PQ at Billion Scale  
重点：对比 HNSW 和 IVF-PQ 在大规模向量检索中的延迟、内存、召回率和工程适用边界。  
链接：https://www.systemoverflow.com/learn/ml-search-ranking/search-scalability/approximate-nearest-neighbor-search-hnsw-vs-ivf-pq-at-billion-scale
