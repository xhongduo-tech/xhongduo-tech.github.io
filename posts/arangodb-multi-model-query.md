## 核心结论

ArangoDB 的“多模型查询”不是把图数据库、文档数据库、KV 数据库简单堆在一起，而是让三类访问方式共享同一套存储引擎、索引体系和 AQL。AQL 是 ArangoDB 的查询语言，白话说就是“用一条语句把遍历、筛选、聚合一次写完”。

它真正有价值的地方，是把图谱场景里最常见的三步放进一个执行计划：先沿边找关系，再按文档属性过滤，最后直接统计或聚合。对初级工程师最重要的一句判断是：一条 AQL 往往比“先查图，再拉文档，再在应用层拼接”更稳，因为它少搬数据，少拼接，少回表。回表的意思是“先靠索引找到候选，再回到原始文档把完整字段取出来”。

| 对比维度 | 库内一次 AQL | 应用层多次拼接 |
|---|---|---|
| 网络往返 | 1 次 | 多次 |
| 中间数据搬运 | 少 | 多 |
| 索引利用 | 统一由优化器决定 | 往往被拆散 |
| 回表次数 | 可被延后甚至减少 | 常被放大 |
| 调试复杂度 | 看一份执行计划 | 看多段代码与多次查询 |

玩具例子：从用户 `u1` 出发，找两跳内关联设备，再筛选 `city = "Shanghai"` 且 `risk_level = "high"`，最后按设备类型计数。这个需求如果拆成三次查询，应用层必须保存中间 ID 列表；如果写成一条 AQL，数据库能在遍历时就剪枝。

---

## 问题定义与边界

本文里的“多模型查询”特指三类能力的组合：

| 能力 | 负责什么 | 在 AQL 里常见形式 |
|---|---|---|
| 图遍历 | 找关系路径和邻居 | `FOR v, e, p IN ...` |
| 文档过滤 | 按属性筛选候选点或边 | `FILTER v.city == ...` |
| KV 访问 | 按 `_key` / `_id` 精确命中 | `DOCUMENT()` 或主键条件 |

最小问题可以写成：从某个用户节点出发，沿关系图找 `1..2` 跳内的对象，再筛选 `city=Shanghai`、`risk_level=high` 的顶点，输出数量和样本。新手版理解就是：先找关系，再看属性，再做统计。

它不是万能方案，边界要先说清楚：

| 项目 | 内容 |
|---|---|
| 输入 | 起点节点、遍历方向、深度范围、属性条件 |
| 输出 | 顶点列表、聚合统计、风险分布、路径样本 |
| 依赖 | 边集合默认边索引；高选择性字段上的 persistent index；必要时 vertex-centric index |
| 不适用 | 纯主键查询、强全文搜索、超重 OLAP、结果可预计算的稳定拓扑 |

“选择性”这个词第一次看容易抽象，它的白话解释是：一个条件能删掉多少无关数据。`risk_level="high"` 如果全库 50% 都满足，选择性就不高；如果只有 1% 满足，选择性就高。多模型查询是否快，核心就在于高选择性条件能不能尽早生效。

---

## 核心机制与推导

把一条多模型查询拆开看，成本大致来自四部分：

$$
C_{total} \approx C_{traversal}(d,b) + C_{filter}(s) + C_{materialize}(r) + C_{join}
$$

其中，$d$ 是遍历深度，$b$ 是平均分支因子，$s$ 是过滤选择性，$r$ 是需要真正取出完整文档的结果数。`materialize` 可以理解为“把轻量候选还原成完整文档”。

候选规模上界常可近似为：

$$
R_{max} \approx \sum_{i=1}^{d} b^i
$$

玩具推导：起点 1 个，平均每个点出边 3 条，深度 `1..2`，则候选上界约为 $3 + 9 = 12$。如果属性过滤只保留 10%，理论上最后只剩 1 条左右。问题在于，昂贵的不一定是最后返回 1 条，而是中间 12 条里有多少条已经被回表、构造 path、做了无效搬运。

真实工程例子：风控图谱里，从“用户”出发找两跳内“设备、手机号、银行卡、收货地址”。这类查询常见写法是先图遍历拿到几千个顶点，再按城市、风险、活跃度过滤。慢的根因通常不是“图遍历本身”，而是遍历后把几千个候选都 materialize，再做低选择性 FILTER。

执行层面可以粗略记成：`TraversalNode` 负责走图，`IndexNode` 负责用索引缩小集合，`MaterializeNode` 负责真正取文档。理想顺序不是“先把所有文档拿全再筛”，而是“先依赖索引和遍历做尽可能多的剪枝，再晚一点取全文档”。

| 优化器规则 | 作用 | 工程含义 |
|---|---|---|
| `optimize-traversals` | 把可下推的过滤条件压进遍历 | 尽早剪枝，少扩展无效边 |
| `push-down-late-materialization` | 延后取完整文档 | 少做无效回表 |
| `optimize-traversal-last-element-access` | 把 `p.vertices[-1]` 改写成 `v` | 避免为拿最后一个点而构造整条路径 |

这里最容易忽视的一点是：多模型优势不是“查询语法更短”，而是“剪枝、索引、回表次序由同一个优化器统一安排”。

---

## 代码实现

先给一个新手可理解版本。目标是：从用户 `users/u1` 出发，找两跳内的高风险上海设备。

```aql
FOR v IN 1..2 OUTBOUND "users/u1" graph_edges
  FILTER v.type == "device"
  FILTER v.city == "Shanghai"
  FILTER v.risk_level == "high"
  RETURN { key: v._key, score: v.risk_score }
```

实际工程里更常见的是边和点一起参与过滤，并顺手聚合：

```aql
FOR v, e, p IN 1..2 OUTBOUND @startVertex graph_edges
  FILTER v.type == "device"
  FILTER v.city == @city
  FILTER v.risk_level == @risk
  FILTER e.relation_strength >= @minStrength
  COLLECT deviceType = v.device_type WITH COUNT INTO cnt
  SORT cnt DESC
  RETURN { deviceType, cnt }
```

如果要看执行计划，重点不是“能不能跑通”，而是有没有过早 materialize、有没有没必要的 path 构造。`EXPLAIN` 用来观察计划，`PROFILE` 用来观察每个节点的实际行数和耗时。

| 看到的现象 | 说明的问题 | 常见改法 |
|---|---|---|
| `TraversalNode` 输出行数很大 | 深度或分支膨胀 | 缩短深度，增加边或点过滤 |
| 早早出现 `MaterializeNode` | 太早回表 | 只返回必要字段，依赖索引覆盖 |
| 明明不用路径却声明了 `p` | 构造了多余 path | 去掉 `p`，只保留 `v` 或 `e` |
| `FilterNode` 在遍历后才大量过滤 | 条件未下推 | 改写为可由遍历和索引利用的条件 |

下面这个 Python 代码块不是在调用 ArangoDB，而是模拟“深度、分支因子、选择性”对候选规模的影响，方便先建立数量级直觉：

```python
def candidate_upper_bound(branch_factor: int, min_depth: int, max_depth: int) -> int:
    total = 0
    for d in range(min_depth, max_depth + 1):
        total += branch_factor ** d
    return total

def expected_after_filter(candidates: int, selectivity: float) -> float:
    return candidates * selectivity

candidates = candidate_upper_bound(branch_factor=3, min_depth=1, max_depth=2)
remaining = expected_after_filter(candidates, selectivity=0.10)

assert candidates == 12
assert abs(remaining - 1.2) < 1e-9
assert candidate_upper_bound(2, 1, 3) == 14
```

这个玩具例子对应的工程直觉是：最终只返回 1 条，不代表查询一定便宜；如果前面已经对 12 条甚至 12000 条做了回表和路径构造，慢点仍然会出现在中间阶段。

---

## 工程权衡与常见坑

很多团队第一次用 ArangoDB，结果正确就收工，慢了就说“图数据库不适合线上”。这通常不是准确结论。更常见的真实原因是查询写法把优化空间堵死了。

| 坑点 | 后果 | 规避方式 |
|---|---|---|
| 深度设太大 | 候选指数膨胀 | 先用 `1..2`、`1..3` 验证，再逐步放宽 |
| 低选择性 FILTER | 剪枝效果差 | 优先下推高选择性条件 |
| 过早 `RETURN v` 或全文档 | 回表成本高 | 先返回 `_key`、少数字段或聚合值 |
| 滥用 `p.vertices[-1]`、`p.edges[-1]` | 迫使构造 path | 能用 `v`、`e` 就不用 `p` |
| 嵌套子查询过多 | 中间结果 materialize 频繁 | 合并逻辑，减少无必要子查询 |

错误写法和改写思路可以非常直接：

错误写法：遍历后立刻 `RETURN v`，再由上层子查询继续过滤。  
改写后：先 `FILTER`，先 `COLLECT`，最后只返回必要字段。

还有一个常见坑是“图集合和文档集合之间频繁 materialize”。如果顶点文档很大，但查询只需要 `city`、`risk_level`、`device_type` 三四个字段，就应该尽量让索引和投影覆盖这些字段，而不是每次都把整份文档读出来。对超节点场景，还要考虑 vertex-centric index，也就是“以 `_from` 或 `_to` 开头，再接业务字段”的持久化索引，用来减少超级节点上的边扫描。

---

## 替代方案与适用边界

多模型查询适合的是“关系结构”和“属性过滤”同时重要的场景，不适合所有查询。

| 方案 | 适用场景 | 优势 | 代价 | 切换信号 |
|---|---|---|---|---|
| ArangoDB 多模型查询 | 图遍历 + 属性过滤 + 聚合 | 一条 AQL 完成闭环 | 依赖索引与计划调优 | 查询同时关心关系和属性 |
| 纯文档查询 | 列表页、详情页、主键查询 | 简单直接 | 关系表达弱 | 几乎不需要遍历 |
| 关系型数据库 | 强事务、多表规范化 | JOIN 语义成熟 | 深层图遍历不自然 | 关系深度通常固定且较浅 |
| 搜索引擎 | 全文检索、模糊匹配 | 检索能力强 | 图结构处理弱 | 主要需求变成搜索 |
| 预聚合/离线结果 | 稳定拓扑、高频报表 | 在线查询极快 | 结果有时效性滞后 | 路径结果重复率极高 |

简单决策例子：如果你只按 `_key` 或 `user_id` 查详情，没必要引入图遍历；如果你要做“找名字相近、内容相关、支持高亮和召回”的搜索，优先考虑搜索引擎；如果“某类两跳关系结果”每天离线就能算好，而且线上只是反复读，预计算通常比实时遍历更省。

所以，是否选择 ArangoDB 多模型查询，不看“模型数量”，看两个问题：第一，关系遍历是不是业务核心；第二，属性过滤能不能在库内与遍历一起高效完成。

---

## 参考资料

1. [Graph traversals in AQL](https://docs.arangodb.com/3.12/aql/graphs/traversals/)
2. [The AQL query optimizer](https://docs.arangodb.com/3.12/aql/execution-and-performance/query-optimization/)
3. [Explain AQL Queries](https://docs.arangodb.com/3.12/aql/execution-and-performance/explaining-queries/)
4. [Profiling and Hand-Optimizing AQL queries](https://docs.arangodb.com/3.12/aql/execution-and-performance/query-profiling/)
5. [Which Index to use when](https://docs.arangodb.com/3.12/index-and-search/indexing/which-index-to-use-when/)
6. [Vertex-Centric Indexes](https://docs.arangodb.com/3.11/index-and-search/indexing/working-with-indexes/vertex-centric-indexes/)
7. [arangodb/arangodb](https://github.com/arangodb/arangodb)
