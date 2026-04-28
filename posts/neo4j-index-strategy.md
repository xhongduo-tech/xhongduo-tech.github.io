## 核心结论

Neo4j 的索引设计，核心不是“这个节点有哪些属性”，而是“这条查询到底怎么筛数据”。**谓词**，白话说就是 `WHERE` 和 `ORDER BY` 里真正参与筛选、排序的条件。索引如果不围绕这些条件设计，建得再多，也只是增加写入负担。

可以把问题压缩成一个公式：

$$
\text{查询成本} \approx \text{候选集大小} \times \text{后续遍历与过滤成本}
$$

索引最重要的作用，不是“让数据库更高级”，而是先把候选集尽量缩小。

| 设计原则 | 说明 |
|---|---|
| 先看 `WHERE` / `ORDER BY` | 索引服务查询，不服务模型装饰 |
| 优先高选择性字段 | 选择性就是“这个条件能过滤掉多少数据” |
| 复合索引按谓词顺序设计 | 等值条件在前，范围条件在后 |
| 约束优先解决唯一性 | 既保数据质量，也顺带提供索引能力 |
| 不给所有属性建索引 | 多余索引会拖慢写入和更新 |

玩具例子很直接。假设 `:Order` 有 1000 万条数据，`tenantId='A'` 只占 1%，最近 30 天只占 10%。  
如果你只按 `tenantId='A'` 过滤，候选集大约是 10 万。  
如果你用 `(tenantId, createdAt)` 复合索引，候选集可以直接压到 1 万量级。  
这就是索引设计的本质：**先缩小起点，再谈图遍历。**

---

## 问题定义与边界

本文回答的问题不是“Neo4j 有哪些索引”，而是：

$$
\text{什么查询} \rightarrow \text{该建什么索引} \rightarrow \text{为什么这样建}
$$

这里的“慢查询”通常不是单一概念。它至少有三种来源：

| 慢的来源 | 是否主要靠索引解决 | 说明 |
|---|---|---|
| 起始筛选范围太大 | 是 | 典型是先扫很多节点再过滤 |
| 图遍历路径太深 | 不完全是 | 可能是模式设计或查询写法问题 |
| 排序和分页代价高 | 部分是 | 如果索引能覆盖排序，会明显改善 |

本文覆盖的边界如下：

| 本文覆盖 | 本文不覆盖 |
|---|---|
| `WHERE` / `ORDER BY` 驱动索引 | 图模型怎么建 |
| `range` / `text` / `full-text` / `constraint` | 集群部署、分片、读写分离 |
| 查询计划与候选集缩小 | 跨库查询、分布式事务 |
| 写入成本与索引数量的权衡 | 硬件容量规划 |

一个常见误解是：“我已经建了索引，为什么还慢？”  
原因通常不是“Neo4j 不用索引”，而是“索引没有把候选集压到足够小”。比如只给 `status` 建索引，但 `status='PAID'` 占全部订单的 60%，这个条件过滤能力就很弱。索引存在，不等于索引有效。

---

## 核心机制与推导

先把常用索引分清。**范围索引**，白话说就是面向精确匹配、范围筛选、前缀匹配的通用主力索引。**文本索引**，白话说就是专门加速字符串包含、后缀、前缀查询。**全文索引**，白话说就是把文本拆成词做搜索，并给结果打相关度分数。

| 索引类型 | 适合谓词 | 是否被 Cypher 自动选用 | 典型用途 |
|---|---|---|---|
| `range index` | `=`、`IN`、`IS NOT NULL`、范围、常见前缀场景 | 是 | 主业务过滤、排序辅助 |
| `text index` | `CONTAINS`、`ENDS WITH`、`STARTS WITH` | 是 | 字符串检索 |
| `full-text index` | 分词搜索、模糊匹配、相关度排序 | 否，需要显式调用 procedure | 搜索框、备注检索 |
| 唯一约束 / key 约束 | 唯一性与存在性校验 | 约束生效时提供 backing index | 防脏数据、防重复 |

### 1. 为什么“查询谓词驱动”比“数据模型驱动”更重要

假设 `User` 有 `name`、`email`、`city`、`bio`、`avatarUrl` 五个属性。  
如果真实查询只有两类：

1. 按 `email` 精确查用户
2. 按 `bio CONTAINS 'graph'` 搜文章作者

那有效索引大概率只有两个：`email` 的范围索引或唯一约束，`bio` 的文本或全文索引。  
给 `avatarUrl`、`city` 都建索引，多半只是维护成本。

### 2. 复合索引为什么要讲顺序

**复合索引**，白话说就是把多个属性按顺序放进一条索引结构里。它不是“多个单列索引简单叠加”。

经验规则可以写成：

$$
(\text{等值条件}) \rightarrow (\text{范围或前缀条件}) \rightarrow (\text{剩余存在性约束})
$$

更具体一点：

| 顺序规则 | 原因 |
|---|---|
| 等值 / `IN` 列放前面 | 先做最强过滤 |
| 范围列放后面 | 范围一旦打开，后续列可利用度通常下降 |
| 排序列尽量接在过滤列后面 | 有机会减少额外排序 |
| 低选择性字段不要盲目前置 | 会让索引前缀过滤力变弱 |

玩具例子：查询总是这样写。

```cypher
MATCH (o:Order)
WHERE o.tenantId = $tenantId
  AND o.status IN $statusList
  AND o.createdAt >= $from
RETURN o
ORDER BY o.createdAt DESC
LIMIT 20
```

比起三个单列索引，更合理的是一个复合范围索引：

`(tenantId, status, createdAt)`

因为查询模式本身就是按这个顺序缩小数据集。

### 3. `text` 和 `full-text` 不是一回事

很多人把“字符串搜索”混成一类，这是错误来源之一。

| 需求 | 更合适的索引 | 原因 |
|---|---|---|
| `name STARTS WITH 'Al'` | `range` 或 `text` | 前缀筛选，计划器可自动使用 |
| `remark CONTAINS 'delay'` | `text` | 子串检索 |
| 搜索“payment timeout”并按相关度排序 | `full-text` | 需要分词和 score |
| 多标签、多属性统一搜索框 | `full-text` | 支持更像搜索引擎的体验 |

关键差异是：`full-text` 不会像普通搜索性能索引那样被 Cypher 计划器自动挑选，你要显式调用 `db.index.fulltext.queryNodes()` 或 `queryRelationships()`。

### 4. 约束为什么也是索引策略的一部分

**约束**，白话说就是强制数据满足某种规则。比如订单号不能重复。  
唯一约束和 key 约束的价值不只是“防错”，还在于它们会提供索引支撑路径。

真实工程里，很多所谓“性能问题”其实先是“数据脏了”。  
例如 `orderNo` 本来应唯一，但库里出现重复值，结果你以为 `MATCH (o:Order {orderNo: $x})` 应该只命中一条，实际上命中多条，后续逻辑和分页都变得不稳定。  
所以约束不是附属品，而是索引策略的一部分。

---

## 代码实现

先给一个可运行的 Python 玩具程序，模拟“单列索引”和“复合索引”对候选集大小的影响：

```python
from math import ceil

def candidate_rows(total_rows: int, selectivities: list[float]) -> int:
    """
    selectivities 中每个值表示某个条件保留下来的比例。
    比如 0.01 表示只保留 1%。
    """
    result = total_rows
    for s in selectivities:
        result *= s
    return ceil(result)

total = 10_000_000

tenant_only = candidate_rows(total, [0.01])          # tenantId='A'
tenant_and_date = candidate_rows(total, [0.01, 0.10]) # 再加最近30天
tenant_status_date = candidate_rows(total, [0.01, 0.20, 0.10]) # 再加状态过滤

assert tenant_only == 100000
assert tenant_and_date == 10000
assert tenant_status_date == 2000

# 多余索引不会缩小这条查询的候选集，只会增加写入维护数量
index_count_minimal = 2   # 例如：业务复合索引 + orderNo唯一约束
index_count_overbuilt = 7 # 再额外给5个很少查询的字段建索引

assert index_count_overbuilt > index_count_minimal

print({
    "tenant_only": tenant_only,
    "tenant_and_date": tenant_and_date,
    "tenant_status_date": tenant_status_date,
    "extra_indexes": index_count_overbuilt - index_count_minimal,
})
```

上面这个程序不能模拟 Neo4j 的真实执行器，但它足够说明一个事实：**真正影响查询起点的，是命中谓词的过滤能力；真正影响写入成本的，是索引数量。**

下面给出一个真实工程例子。场景是多租户订单系统，列表页按租户、状态、创建时间过滤，并按创建时间倒序分页。

先建复合范围索引：

```cypher
CREATE RANGE INDEX order_tenant_status_createdAt
FOR (o:Order)
ON (o.tenantId, o.status, o.createdAt);
```

给业务唯一键建约束：

```cypher
CREATE CONSTRAINT order_orderNo_unique
FOR (o:Order)
REQUIRE o.orderNo IS UNIQUE;
```

查询写法：

```cypher
MATCH (o:Order)
WHERE o.tenantId = $tenantId
  AND o.status IN $statusList
  AND o.createdAt >= $from
RETURN o
ORDER BY o.createdAt DESC
SKIP $offset
LIMIT $limit;
```

如果你还有备注搜索需求，例如“搜出备注里包含 delay 的订单”，不要复用上面的范围索引，而应单独设计字符串检索路径：

```cypher
CREATE TEXT INDEX order_remark_text
FOR (o:Order)
ON (o.remark);
```

如果需求升级成搜索框，要支持多个词和相关度排序，则改成全文索引：

```cypher
CREATE FULLTEXT INDEX order_search_ft
FOR (o:Order)
ON EACH [o.remark, o.customerName];
```

查询时显式调用：

```cypher
CALL db.index.fulltext.queryNodes("order_search_ft", $q) YIELD node, score
RETURN node, score
ORDER BY score DESC
LIMIT 20;
```

索引和查询模式可以直接这样对应：

| 查询模式 | 推荐方案 | 说明 |
|---|---|---|
| `orderNo = $x` | 唯一约束 | 查单快，也防重复 |
| `tenantId = ... AND status IN ... AND createdAt >= ...` | 复合 `range` 索引 | 典型列表页 |
| `remark CONTAINS 'delay'` | `text` 索引 | 子串匹配 |
| 搜索 `refund timeout` 并按相关度排序 | `full-text` 索引 | 显式调用 procedure |

---

## 工程权衡与常见坑

索引的收益和代价必须一起看。每增加一个索引，插入、更新、删除都要多做一次索引维护。对读多写少系统，这个代价可能可接受；对高写入订单流、日志流、事件流，索引膨胀会非常明显。

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| 给所有属性建索引 | 写入变慢，收益分散 | 只保留高频高价值谓词 |
| 只建单列索引，不看组合查询 | 候选集仍然太大 | 围绕真实组合谓词建复合索引 |
| 把 `full-text` 当普通索引 | 查询计划不会自动使用 | 显式调用 `db.index.fulltext.queryNodes()` |
| 复合索引顺序乱 | 前缀利用率低 | 等值在前，范围在后 |
| 忽略排序字段 | 过滤后仍需昂贵排序 | 把常见 `ORDER BY` 一并考虑 |
| 可空字段随意参与设计 | 执行路径不稳定 | 结合 `IS NOT NULL` 或约束固定数据质量 |

这里最值得强调的是 `PROFILE`。  
`PROFILE`，白话说就是让 Neo4j 展示查询的真实执行计划和每一步代价。你不看 `PROFILE`，很多索引设计都是猜的。

真实工程里，一个常见场景是：

1. 你看到查询条件里有 `tenantId`、`status`、`createdAt`
2. 于是给三列分别建单列索引
3. 查询仍然慢
4. `PROFILE` 一看，起点节点数依旧很大，后面还做了大量过滤和排序

这时问题不在“Neo4j 不智能”，而在“索引布局没有贴合查询前缀”。

---

## 替代方案与适用边界

不是所有搜索需求都该靠同一种索引解决。先分需求，再选方案。

| 需求类型 | 推荐方案 | 不适合的情况 |
|---|---|---|
| 精确匹配 | `range` 索引或唯一约束 | 需要分词和相关度时 |
| 范围过滤 | `range` 索引 | 纯全文搜索 |
| 字符串 `CONTAINS` / `ENDS WITH` | `text` 索引 | 需要搜索打分 |
| 分词搜索、模糊搜索、相关度排序 | `full-text` 索引 | 想让普通 `WHERE` 自动命中 |
| 几乎不参与筛选的字段 | 不建索引 | 高频写入系统里尤其如此 |

一个简单决策顺序可以写成：

```text
先判断谓词类型
→ 再判断是否高频
→ 再判断选择性是否足够
→ 再判断是否需要覆盖排序
→ 最后评估写入维护成本
```

适用边界也要说清楚。索引能解决的是“如何更快找到起点”，不是“图上的任何慢查询都能靠索引修好”。  
如果查询慢是因为：

- 可变长路径太深
- 模式设计导致热点超级节点
- 返回字段过多、分页方式不合理
- 应用层反复发小查询

那就不能只盯着索引。

所以更准确的说法是：**Neo4j 索引是查询入口优化器，不是全部性能问题的总开关。**

---

## 参考资料

1. [Neo4j Cypher Manual: Search-performance indexes](https://neo4j.com/docs/cypher-manual/current/indexes/search-performance-indexes/)
2. [Neo4j Cypher Manual: Create indexes](https://neo4j.com/docs/cypher-manual/current/indexes/search-performance-indexes/create-indexes/)
3. [Neo4j Cypher Manual: The impact of indexes on query performance](https://neo4j.com/docs/cypher-manual/current/indexes/search-performance-indexes/using-indexes/)
4. [Neo4j Cypher Manual: Full-text indexes](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/full-text-indexes/)
5. [Neo4j Cypher Manual: Constraints](https://neo4j.com/docs/cypher-manual/current/schema/constraints/)
