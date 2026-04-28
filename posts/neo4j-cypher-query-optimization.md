## 核心结论

Cypher 查询优化的核心，不是把语句写得更短，而是让执行计划尽早把候选集合缩小，尽量少做关系扩展，尽量少传递大中间结果。

一个实用经验公式是：

$$
总代价 \approx |S_0| \times \prod_{i=1}^{d}(b_i \times f_i)
$$

这里，$|S_0|$ 是起点候选数，白话说就是“第一步可能从多少个点开始算”；$b_i$ 是第 $i$ 步的平均分支数，白话说就是“每走一步平均会岔出多少条边”；$f_i$ 是该步过滤后的保留比例，白话说就是“扩展完以后还剩多少结果”。

这不是 Neo4j 官方公式，而是非常有用的工程近似。它直接说明一个事实：起点集合一旦过大，后面每一跳都会把成本乘上去。

下表可以先记住：

| 写法 | 起点是否容易收缩 | 常见计划倾向 | 风险 |
|---|---|---|---|
| `MATCH (u:User {email:$e})-[:FOLLOWS*1..3]->(v)` | 是 | `NodeIndexSeek -> Expand` | 低 |
| `MATCH (u:User)-[:FOLLOWS*1..3]->(v) WHERE u.email=$e` | 否 | `NodeByLabelScan -> Expand -> Filter` | 高 |
| `MATCH (a), (b)` | 否 | 两边分别扫描后做组合 | 极高 |
| 多次 `OPTIONAL MATCH` 叠加大结果集 | 否 | 中间结果持续放大 | 高 |

玩具例子可以直接看两句查询的区别：

```cypher
MATCH (u:User {email:$e})-[:FOLLOWS*1..3]->(v)
RETURN v
```

```cypher
MATCH (u:User)-[:FOLLOWS*1..3]->(v)
WHERE u.email = $e
RETURN v
```

两句在逻辑上可能等价，但执行代价通常完全不同。前者先把 `u` 缩到 1 个，再往外找；后者先把大量 `User` 当起点展开，最后才补过滤。

---

## 问题定义与边界

本文讨论的是 Cypher 的执行计划优化，不是图数据库建模教程，也不是 Cypher 语法入门。

重点范围如下：

| 适用场景 | 不讨论内容 | 需要注意的前提 |
|---|---|---|
| 读查询调优，尤其是 `MATCH`、`WHERE`、`OPTIONAL MATCH`、可变长路径 | 图模型怎么设计、集群部署、写入吞吐优化 | 已有基本索引、知道查询目标、能使用 `EXPLAIN`/`PROFILE` |
| 按唯一键或高选择性属性定位起点，再做 1 到 3 跳扩展 | 复杂图算法理论推导 | 数据量不是玩具级，查询需要上线 |
| 分析 `Rows`、`DB Hits`、`Estimated Rows` | 只看语法风格做“玄学优化” | 接受“更快的写法可能更长” |

“选择性”这个词第一次出现时可以这样理解：一个条件能把候选数据压得越小，选择性越高。`email` 唯一索引的选择性通常很高，`status='active'` 往往就没那么高。

真实工程例子是“推荐关注”。需求不是“在全图里找可能的人”，而是“先按 `userId` 找到当前用户，再找 2 跳内可能认识的人，并过滤掉已关注、已拉黑、已注销账号”。这种问题的关键永远是先锁定一个很窄的起点。

---

## 核心机制与推导

Neo4j 会先由 planner 生成执行计划。planner 可以理解为“查询路线规划器”，负责决定从哪里开始找、先做什么、后做什么。然后执行器按这个计划逐步运行。

对新手最重要的不是记住所有算子，而是先区分两类路径：

1. 高效路径：`NodeIndexSeek -> Expand -> Filter/Projection`
2. 低效路径：`NodeByLabelScan -> Expand -> Filter`

`NodeIndexSeek` 的意思是“通过索引精确找起点”；`NodeByLabelScan` 的意思是“先把某个标签下的节点扫出来再说”。前者是窄入口，后者通常是宽入口。

假设有 100 万个 `:User`，`email` 上有唯一索引，平均每个用户关注 5 个人，且我们最多扩展 3 跳。

变量含义如下：

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $|S_0|$ | 起点候选数 | 一开始有多少个点可能被当成起点 |
| $b_i$ | 第 $i$ 步平均分支数 | 第 $i$ 跳平均会扩出去多少条边 |
| $f_i$ | 第 $i$ 步过滤保留比例 | 这一跳之后还能剩多少比例 |
| `1..d` | 可变长路径上界 | 最多走多少跳 |

如果写成：

```cypher
MATCH (u:User {email:$e})-[:FOLLOWS*1..3]->(v)
RETURN v
```

那么通常 $|S_0| = 1$。若先忽略过滤，状态增长近似是：

| 跳数 | 新增状态数 | 累计状态数 |
|---|---:|---:|
| 1 跳 | $1 \times 5 = 5$ | 6 |
| 2 跳 | $5 \times 5 = 25$ | 31 |
| 3 跳 | $25 \times 5 = 125$ | 156 |

如果写成：

```cypher
MATCH (u:User)-[:FOLLOWS*1..3]->(v)
WHERE u.email = $e
RETURN v
```

那么 planner 很可能先把大量 `User` 纳入候选，再考虑过滤。即使最后只会命中 1 个邮箱，前面的扩展也可能已经发生在巨大集合上。这就是“最后能筛出来”不等于“前面没有白算”。

更直白地说，Cypher 优化不是比谁会写花哨语法，而是比谁更早把 $|S_0|$ 压小，把每一跳的 $b_i$ 压小，把每一步的结果尽量就地过滤掉。

---

## 代码实现

调优时先用 `EXPLAIN` 看计划形状，再用 `PROFILE` 看真实执行。`Rows` 可以理解为“有多少行结果经过这个算子”；`DB Hits` 可以理解为“这个算子为了取数据碰了多少次存储层”。

一个推荐模板：

```cypher
PROFILE
MATCH (u:User {email: $e})-[:FOLLOWS*1..3]->(v:User)
WHERE v.status = 'active'
RETURN v.userId
LIMIT 20
```

一个常见低效写法：

```cypher
PROFILE
MATCH (u:User)-[:FOLLOWS*1..3]->(v:User)
WHERE u.email = $e AND v.status = 'active'
RETURN v.userId
LIMIT 20
```

再看一个真实工程例子。需求：给某个用户推荐二跳关注对象，排除已经关注的人。

```cypher
PROFILE
MATCH (u:User {userId: $uid})-[:FOLLOWS]->(:User)-[:FOLLOWS]->(cand:User)
WHERE cand.status = 'active'
  AND cand.userId <> $uid
  AND NOT EXISTS {
    MATCH (u)-[:FOLLOWS]->(cand)
  }
RETURN cand.userId, count(*) AS score
ORDER BY score DESC
LIMIT 20
```

这里 `NOT EXISTS { ... }` 的作用是做相关子检查，白话说是“对每个候选人，确认当前用户还没有直接关注过他”。

`PROFILE` 解读可以按下面这个表走：

| Operator | 看什么 | 常见修正 |
|---|---|---|
| `NodeIndexSeek` | 是否作为起点出现 | 没出现就检查索引和谓词写法 |
| `NodeByLabelScan` | `Rows` 是否过大 | 改成更窄的起点条件 |
| `Expand(All)` / `Expand(Into)` | 扩展前后的 `Rows` 是否暴涨 | 限制跳数、提前过滤、拆分查询 |
| `Filter` | 是否在大扩展之后才出现 | 把条件贴近对应 `MATCH` |
| `Optional` 相关算子 | 是否在大结果集上重复执行 | 用 `WITH` 或子查询隔离 |
| `CartesianProduct` | 是否意外出现 | 检查是否写了无连接模式 |

“写法前后对比”通常比抽象原则更有用。

前：

```cypher
MATCH (u:User)-[:FOLLOWS]->(v:User)
WITH u, v
WHERE u.email = $e
RETURN v
```

后：

```cypher
MATCH (u:User {email:$e})-[:FOLLOWS]->(v:User)
RETURN v
```

前者把过滤拖过了一个 `WITH` 边界。`WITH` 可以理解为“把前一段结果先整理后再进入下一段”，它是很有用的隔离工具，但如果把应当早做的过滤拖到 `WITH` 后面，通常只会更慢。

下面给一个可运行的 Python 玩具程序，用来模拟起点规模和跳数如何放大代价：

```python
def estimate_states(start_size: int, branches: list[int]) -> int:
    total = start_size
    current = start_size
    for b in branches:
        current *= b
        total += current
    return total

good = estimate_states(1, [5, 5, 5])
bad = estimate_states(1_000_000, [5, 5, 5])

assert good == 156
assert bad == 156_000_000

def estimate_with_filter(start_size: int, branch_filter_pairs: list[tuple[int, float]]) -> float:
    total = float(start_size)
    current = float(start_size)
    for b, f in branch_filter_pairs:
        current = current * b * f
        total += current
    return total

filtered = estimate_with_filter(1, [(5, 0.4), (5, 0.2), (5, 0.5)])
assert round(filtered, 2) == 8.0
print(good, bad, filtered)
```

这个程序当然不是 Neo4j 执行器本身，但它足够说明一个核心事实：先把起点压到 1 个，再做扩展，和先从 100 万个起点扩展，数量级完全不同。

---

## 工程权衡与常见坑

优化不只是“建个索引就结束”。很多查询慢，不是因为没索引，而是因为中间结果在扩展时已经失控。

常见坑如下：

| 坑点 | 症状 | 规避方式 | 诊断手段 |
|---|---|---|---|
| 无选择性起点扫描 | 一上来就是 `NodeByLabelScan`，`Rows` 很大 | 给高选择性属性建索引，改写谓词让 planner 能用 seek | `EXPLAIN`/`PROFILE` 看起点算子 |
| 可变长路径滥用 | `Rows` 和 `DB Hits` 指数增长 | 必须写上界，如 `*1..3`，不要无界展开 | 看扩展算子前后 `Rows` |
| 先扩展后过滤 | `Filter` 出现在大扩展之后 | 把条件贴近对应 `MATCH` | 对比改写前后计划 |
| 重复 `OPTIONAL MATCH` | 一层层把中间结果放大 | 能合并就合并，不能合并就 `WITH` 隔离 | 看 `OPTIONAL` 后 `Rows` 是否连乘上涨 |
| 笛卡尔积 | 计划里出现 `CartesianProduct` | 避免 `MATCH (a), (b)` 这种无连接模式 | `PROFILE` 看算子名 |
| 统计信息过旧 | 明明有索引却计划异常 | 重采样索引统计，必要时触发重新规划 | 结合执行计划与统计配置检查 |

`OPTIONAL MATCH` 可以理解为“左连接式匹配”，即使右边找不到也保留左边结果。问题在于，如果左边已经是很大的结果集，再多来几次 `OPTIONAL MATCH`，就会变成对大结果集的重复补充和重复复制。

例如下面这种写法就要警惕：

```cypher
MATCH (u:User {status:'active'})
OPTIONAL MATCH (u)-[:HAS_PROFILE]->(p)
OPTIONAL MATCH (u)-[:LIVES_IN]->(c)
OPTIONAL MATCH (u)-[:WORKS_AT]->(co)
RETURN u, p, c, co
```

如果 `u` 本身就很多，这条查询会在很大的中间结果上反复做可选扩展。工程上更稳妥的做法，通常是先缩小 `u`，或者分段查询，或者只在真正需要时再查附加信息。

---

## 替代方案与适用边界

当 Cypher 查询已经开始接近“路径枚举问题”时，继续抠语法细节通常收益很小。路径枚举问题的白话解释是：系统不是在找少量命中，而是在尝试大量可能路径。

下面这张表更重要：

| 问题类型 | 优先方案 | 不适合继续用 Cypher 直接展开的信号 |
|---|---|---|
| 唯一键定位后的小范围多跳查询 | Cypher + 索引 + 有界路径 | `Rows` 可控，结果集天然不大 |
| 子串检索、前后缀检索 | text index | 还在靠 label scan + `CONTAINS` 硬扫 |
| 高频推荐结果 | 预计算关系或物化结果 | 同一推荐逻辑被高频重复计算 |
| 复杂最短路、中心性、社区发现 | GDS 或离线计算 | 查询已经像图算法，而不是业务检索 |
| 大范围统计聚合 | 分层缓存、离线作业 | 在线 Cypher 每次都扫大图 |

比如“查名字里包含 `alex` 的用户”，优先考虑 text index，而不是先扫 `:User` 再用 `WHERE u.name CONTAINS 'alex'` 硬筛。

再比如“推荐关注”如果规则稳定、流量高，完全可以提前物化 `(:User)-[:RECOMMENDED]->(:User)`，或者离线算好候选列表。因为问题的瓶颈已经不是单条 Cypher 是否优雅，而是在线路径展开本身太贵。

所以适用边界很明确：Cypher 很擅长做局部图模式匹配，不擅长无限制地替代全文检索、离线推荐系统和复杂图算法平台。

---

## 参考资料

1. [Execution plans and query tuning](https://neo4j.com/docs/cypher-manual/current/planning-and-tuning/)
2. [Query tuning](https://neo4j.com/docs/cypher-manual/current/planning-and-tuning/query-tuning/)
3. [Operators in detail](https://neo4j.com/docs/cypher-manual/current/planning-and-tuning/operators/operators-detail/)
4. [OPTIONAL MATCH](https://neo4j.com/docs/cypher-manual/current/clauses/optional-match/)
5. [Variable-length paths](https://neo4j.com/docs/cypher-manual/current/patterns/variable-length-paths/)
6. [Statistics and execution plans](https://neo4j.com/docs/operations-manual/current/performance/statistics-execution-plans/)
7. [Search-performance indexes](https://neo4j.com/docs/cypher-manual/current/indexes/search-performance-indexes/)
