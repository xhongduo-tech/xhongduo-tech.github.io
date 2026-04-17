## 核心结论

Elasticsearch 的搜索优化，本质上不是“把机器堆大”这么简单，而是把搜索链路拆成四件事分别优化：倒排索引、分词器、相关性打分、分片布局。倒排索引，就是“词 -> 出现过这些词的文档列表”的结构；分词器，就是“把一句话切成可搜索词元”的规则；相关性打分，通常指 BM25 这类“谁更像用户想找的内容”的评分算法；分片布局，就是把一个索引拆到多少个物理子索引上运行。

一个对新手最有用的判断标准是：先决定“哪些字段值得全文搜索”，再决定“哪些条件只是筛掉不相关文档”，最后才决定“剩下的文档怎么排分”。如果 mapping 乱、text 字段开太多、filter 和 query 混用、分片数量失控，那么再好的硬件也只是替错误设计买单。

下面这张表先给出优化主线：

| 优化维度 | 目标效果 | 典型手段 |
| --- | --- | --- |
| mapping 约束 | 避免冗余索引和无效分词 | 该用 `keyword` 的不用 `text` |
| 分词器选择 | 让词项切分符合语言特征 | 英文、中文、前缀搜索分别配置 |
| query/filter 分离 | 减少参与打分的文档量 | `filter` 先筛，`must` 再评分 |
| 相关性打分 | 提高搜索结果排序质量 | BM25、字段 boost、时间衰减 |
| 分片布局 | 平衡并发、扩容与协调成本 | 控制 shard 数与单分片大小 |

玩具例子可以这样理解：你有 100 篇文章。先把“数据库”这个词出现在哪些文章里记下来，这叫倒排索引；把“Elasticsearch 搜索优化怎么做”拆成几个关键词，这叫分词；用一个分数公式判断哪篇更相关，这叫 BM25；再把 `status=published`、`year>=2024` 这种不需要参与排名的条件先过滤掉，这就是 filter 的价值。

---

## 问题定义与边界

搜索优化的边界，是在有限的 CPU、内存、磁盘和网络下，同时满足两件事：

1. 查询延迟足够低。
2. 排名结果足够稳定、足够相关。

这意味着不是所有字段都该建全文索引。全文索引，就是字段内容会被分词后写入倒排索引，方便按词搜索；精确匹配字段只需要原值查找，不需要分词。典型地：

| 字段类型选择 | 适合内容 | 查询成本变化 | 常见风险 |
| --- | --- | --- | --- |
| `keyword` | 状态、ID、枚举、标签原值 | 成本低，适合 filter/聚合 | 不支持自然语言检索 |
| `text` | 标题、正文、摘要 | 成本高，要分词和打分 | 字段多时索引膨胀 |
| `text` + `keyword` 子字段 | 既要搜索又要聚合/排序 | 灵活，但占更多空间 | 滥用会加重写入与存储 |

新手版本的边界判断可以简单记成三问：

1. 这个字段要不要按“词”搜索？
2. 这个条件是不是只负责“筛掉不满足的文档”？
3. 这个字段值多不多，值长不长，会不会把索引撑大？

例如博客系统里：

- `title`、`summary`、`body` 适合 `text`
- `slug`、`author`、`status`、`category` 适合 `keyword`
- `published_at` 适合 `date`
- `view_count` 适合数值类型

分片也是边界的一部分。分片就是索引的物理拆分单元。分得太多，单次查询要协调更多 shard，合并 segment 的开销也会上升；分得太少，单分片过大，扩容和并发能力会受限。工程上常见经验值是单分片控制在 10 到 50GB，这不是硬规则，但很适合作为初始容量区间。

---

## 核心机制与推导

Elasticsearch 搜索大致分四步：

1. 分词，把查询字符串切成词项。
2. 在倒排索引里找到每个词项对应的 posting list。posting list 就是“某个词出现在哪些文档里”的列表。
3. 用评分公式给候选文档打分。
4. 各 shard 返回局部 TopN，再由协调节点合并成全局结果。

BM25 的简化形式通常写成：

$$
score(q,d)=\sum_{t\in q} IDF(t)\cdot \frac{tf(t,d)\cdot(k_1+1)}{tf(t,d)+k_1\cdot\left(1-b+b\cdot\frac{|d|}{avgdl}\right)}
$$

其中：

- $tf(t,d)$ 是词项在文档中的出现次数
- $IDF(t)$ 是逆文档频率，词越稀有，权重通常越高
- $|d|$ 是文档长度
- $avgdl$ 是平均文档长度
- $k_1,b$ 是控制词频饱和和长度归一化的参数

白话解释是：某个词在这篇文档里出现得多、在全库里又不常见，而且文档长度没有被无意义拉长，那么它的分数就更高。

玩具例子：假设搜索词是“vector database”。

- 文档 A 标题里有两次 `vector`，正文有一次 `database`
- 文档 B 正文里各出现一次
- 文档 C 很长，正文里出现很多次，但夹杂大量无关内容

通常 A 会更靠前，因为标题字段往往有更高 boost，且文档更短，长度归一化后得分更集中。

分片会影响打分的一致性。因为默认相关性统计常在 shard 级别先计算，再合并结果。同一批文档如果分到 1 个 shard 和 5 个 shard，某些词的局部 DF 不同，导致 $IDF$ 略有差异，最终排序可能不完全一样。

| 场景 | 统计基准 | 结果特点 |
| --- | --- | --- |
| 1 shard | 全量文档统一统计 | 排名更稳定 |
| 5 shards | 每个 shard 局部统计后再合并 | 并发更好，但分数可能轻微漂移 |

所以，shard 设计不是“越多越快”，而是“并发能力”和“分数稳定性”之间的平衡。

真实工程例子：一个内容平台有 4 个节点、数亿文档，后来给索引新增了一个高基数、多值、`edge_ngram` 的 text 字段。高基数，就是字段不同取值非常多；`edge_ngram` 是“按前缀切词”的分词方式。结果 segment 数量、词项数量和 posting 规模都迅速膨胀，查询从毫秒级上升到 10 秒以上。问题并不在“ES 不够快”，而在“把本该谨慎控制的字段变成了超重全文索引”。

---

## 代码实现

先看一个合理的 mapping 方向。核心思路是：只给需要全文检索的字段开 `text`，其余字段尽量走 `keyword` 或数值/日期类型。

```json
{
  "mappings": {
    "properties": {
      "title": {
        "type": "text",
        "analyzer": "standard",
        "fields": {
          "raw": { "type": "keyword" }
        }
      },
      "body": {
        "type": "text",
        "analyzer": "standard"
      },
      "status": {
        "type": "keyword"
      },
      "category": {
        "type": "keyword"
      },
      "published_at": {
        "type": "date"
      },
      "view_count": {
        "type": "integer"
      }
    }
  }
}
```

查询时要明确区分 `filter` 和 `must`。`filter` 不参与打分，适合状态、时间范围、分类等条件；`must` 参与打分，适合关键词匹配。

```json
{
  "track_total_hits": false,
  "terminate_after": 10000,
  "query": {
    "function_score": {
      "query": {
        "bool": {
          "filter": [
            { "term": { "status": "published" } },
            { "term": { "category": "database" } },
            { "range": { "published_at": { "gte": "2025-01-01" } } }
          ],
          "must": [
            {
              "multi_match": {
                "query": "elasticsearch 搜索优化",
                "fields": ["title^3", "body"]
              }
            }
          ]
        }
      },
      "functions": [
        {
          "gauss": {
            "published_at": {
              "origin": "now",
              "scale": "30d",
              "decay": 0.5
            }
          }
        }
      ],
      "score_mode": "sum",
      "boost_mode": "multiply"
    }
  },
  "sort": [
    "_score",
    { "published_at": "desc" }
  ]
}
```

这里的重点只有两条：

- `filter` 负责尽早缩小候选集，缓存友好。
- `must` 负责真正的相关性计算，`title^3` 表示标题命中权重大于正文。

下面用一个可运行的 Python 玩具程序模拟“先过滤再打分”的思想：

```python
from math import log

docs = [
    {"id": 1, "status": "published", "title_tf": 2, "body_tf": 1, "length": 120},
    {"id": 2, "status": "draft",     "title_tf": 3, "body_tf": 2, "length": 90},
    {"id": 3, "status": "published", "title_tf": 1, "body_tf": 4, "length": 300},
]

def bm25_like(tf, doc_len, avgdl=170, k1=1.2, b=0.75, idf=1.5):
    return idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl)))

filtered = [d for d in docs if d["status"] == "published"]
scored = []
for d in filtered:
    score = 3 * bm25_like(d["title_tf"], d["length"]) + bm25_like(d["body_tf"], d["length"])
    scored.append((d["id"], score))

scored.sort(key=lambda x: x[1], reverse=True)

assert len(filtered) == 2
assert scored[0][0] == 1
assert scored[0][1] > scored[1][1]
print(scored)
```

这个例子不等于 Elasticsearch 的完整实现，但它抓住了优化重点：`draft` 文档先被过滤掉，只有真正需要参与排名的文档才进入评分阶段。

真实工程里还会配合这些参数：

| 参数 | 作用 | 适用场景 |
| --- | --- | --- |
| `track_total_hits: false` | 不精确统计总命中数，减少额外工作 | 只关心前几页结果 |
| `terminate_after` | 每 shard 达到一定文档数后提前停止 | 粗略召回、低延迟优先 |
| `shard_size` | 控制每 shard 返回候选量 | 聚合或 TopN 合并优化 |
| `function_score` | 在 BM25 外叠加业务权重 | 新鲜度、热度、权威性排序 |

---

## 工程权衡与常见坑

搜索优化常见失败，不是因为不会写 DSL，而是没有处理好工程约束。

| 问题 | 征兆 | 解决方向 |
| --- | --- | --- |
| 分片过多 | CPU 忙于协调，merge time 高 | 减少 shard，按生命周期 rollover |
| 动态 mapping 失控 | 新字段自动变成 `text`，磁盘暴涨 | 关闭或限制动态映射，显式定义字段 |
| 高基数字段误用 `text` | 索引体积大，查询慢，heap 紧张 | 改成 `keyword` 或拆分子字段 |
| `filter` 和 `must` 混用 | 不必要文档参与打分 | 固定条件放 `filter` |
| 滥用 `edge_ngram` | segment 膨胀，响应时间突增 | 仅在前缀搜索必要字段上使用 |
| refresh 太频繁 | 写入吞吐下降，segment 零碎 | 写入高峰适度调大 `refresh_interval` |

一个典型坑是“为了省事，让动态 mapping 自动推断类型”。结果某个本来只是 ID 列表或标签枚举的字段，被自动识别成 `text`，不仅多了分词和倒排索引，还可能生成额外子字段。字段一多、值一散，索引膨胀会非常快。

另一个高频坑是“看起来都像字符串，就都设成 text”。这在小数据量时不明显，但一旦进入真实流量，会直接放大三类成本：

1. 写入成本：分词、建词典、写 posting。
2. 查询成本：更多字段参与 match 和 scoring。
3. 存储成本：segment 和 term dictionary 更大。

监控上至少要盯住这些指标：

- search latency：查询延迟是否持续上升
- merge time：segment 合并是否过重
- heap usage：堆内存是否被词典和缓存持续挤压
- segment count：碎片是否过多
- indexing throughput：写入速度是否因 refresh/merge 明显下降

---

## 替代方案与适用边界

并不是所有搜索都值得上完整 BM25。很多系统只需要“精确筛选 + 简单排序”，此时纯 `keyword` + filter 往往更稳、更便宜。

| 方案 | 资源消耗 | 结果准确率 | 适用场景 |
| --- | --- | --- | --- |
| 纯 `keyword` + filter | 低 | 对自然语言弱 | 状态筛选、订单查询、后台检索 |
| BM25 全文搜索 | 中到高 | 对文本相关性强 | 博客、文档、商品搜索 |
| BM25 + function_score | 更高 | 兼顾文本与业务排序 | 内容平台、推荐式搜索 |
| Hot/Warm 分层 | 中 | 与查询设计相关 | 热数据低延迟、冷数据低成本 |

如果你的查询像“查订单号”“查用户状态”“查某个 slug”，那其实根本不需要全文检索。`term` 查询加缓存友好的 `filter` 更直接。只有当用户输入自然语言，且确实希望系统判断“哪篇更相关”时，BM25 才有意义。

真实工程例子是冷热分层。Hot 节点就是放最近、最常查的数据；Warm 节点就是放历史、低频数据。这样做的目的不是“让所有数据都一样快”，而是把有限资源优先给高价值查询。常见策略是：

- 最近 7 到 30 天数据放 hot
- 老数据 rollover 后迁移到 warm
- 查询默认先打 hot，必要时再扩展到 warm

这类方案适合日志、内容平台、审计检索等时间分布明显的系统；如果数据集本身很小，或者查询必须跨全量数据做统一排名，那么收益会下降。

---

## 参考资料

- [Elastic Blog: Optimizing Elasticsearch Searches](https://www.elastic.co/blog/found-optimizing-elasticsearch-searches)  
  重点在 query/filter 分离、缓存友好查询、减少不必要评分。本文“先 filter 再打分”的工程实践主要参考这一思路。

- [Elastic Blog: Practical BM25 Part 1 - How Shards Affect Relevance Scoring](https://www.elastic.co/blog/practical-bm25-part-1-how-shards-affect-relevance-scoring-in-elasticsearch)  
  重点说明 shard 级统计会影响相关性分数，本文“1 shard 与 5 shards 得分可能不一致”的部分基于该文阐述。

- [OneRuby: How Elasticsearch Really Works Under the Hood](https://oneruby.dev/how-elasticsearch-really-works-under-the-hood/)  
  重点在倒排索引、分词、segment、分片等基础机制。本文“倒排索引是什么”“搜索链路怎么走”的解释可对照阅读。

- [Medium: Elasticsearch - A Guide to Conceptually Learn and Implement](https://medium.com/%40bhargav.maddikera/elasticsearch-a-guide-to-conceptually-learn-and-implement-e910beecb190)  
  重点在概念性梳理 BM25、倒排索引和查询实现，本文公式与玩具推导可作为入门补充。

- [Elastic Discuss: Severe Performance Degradation After Adding High-Cardinality Text Field to Large Index](https://discuss.elastic.co/t/severe-performance-degradation-after-adding-high-cardinality-text-field-to-large-index/378571)  
  重点是大规模真实案例：高基数、多值 text 字段会显著放大 segment 和查询成本。本文“400M 文档性能从毫秒级到 10+ 秒”的工程例子来自这一类社区案例。

- [Sachith: Elasticsearch/OpenSearch Sizing, Mappings & Performance Tuning Guide](https://www.sachith.co.uk/elasticsearch-opensearch-sizing-mappings-performance-tuning-guide-practical-guide-nov-2-2025/)  
  重点在分片大小、mapping 约束、ILM、容量规划。本文“10-50GB/分片”“mapping 先行”的经验值参考了这类实践总结。
