## 核心结论

Cypher 是 Neo4j 的声明式查询语言。声明式的白话解释是：你描述“结果应该长什么样”，数据库自己决定“具体怎么找”。因此，Cypher 的核心不是写循环，不是手工控制遍历顺序，而是把节点、关系、方向、过滤条件直接写成图模式。

它最重要的骨架可以先记成一个公式：

$$
\texttt{MATCH pattern [WHERE predicate] RETURN projections}
$$

含义很直接：

| 子句 | 作用 | 白话解释 |
| --- | --- | --- |
| `MATCH` | 描述图模式 | 把你要找的节点和连线“画出来” |
| `WHERE` | 增加条件 | 只保留满足条件的那部分结果 |
| `RETURN` | 输出结果 | 决定最后把哪些字段拿出来 |

最小可理解例子如下：

```cypher
MATCH (movie:Movie)
RETURN movie.title
```

这里 `(movie:Movie)` 表示“找所有带 `Movie` 标签的节点”，`movie.title` 表示返回这些节点的 `title` 属性。

如果只记一个模式，应该记这个：

```cypher
MATCH (n:Person)-[:KNOWS]->(m:Person)
RETURN n.name, m.name
```

它表达的不是“从头扫描表，再 join，再过滤”，而是“找出所有 `Person` 节点之间，存在 `KNOWS` 关系的有向连接”。这正是图查询和关系型 SQL 思维最不同的地方。

---

## 问题定义与边界

图数据库先要理解三个基础对象。

| 对象 | Cypher 写法 | 白话解释 |
| --- | --- | --- |
| 节点 | `(p:Person)` | 图里的一个点，比如“某个人” |
| 关系 | `-[:ACTED_IN]->` | 点和点之间的边，比如“出演了” |
| 属性 | `{name: 'Tom Hanks'}` | 点或边身上的字段，比如名字、时间 |

术语首次出现时可以这样理解：

- 节点：图里的实体，白话就是“一个对象”。
- 关系：实体之间的连接，白话就是“对象之间的动作或联系”。
- 标签：节点的类型名，白话就是“这个对象属于哪一类”。
- 属性：节点或关系上的键值对，白话就是“这个对象带了哪些字段”。

入门时可以把 Neo4j 图模型理解成下面这张表：

| Label | 典型属性 | 连线示例 |
| --- | --- | --- |
| `Person` | `name` | `-[:ACTED_IN]-> Movie` |
| `Movie` | `title`, `released` | `<-[:ACTED_IN]- Person` |

所以，`(p:Person {name: 'Alice'})` 的意思是：找一个标签为 `Person`，并且 `name='Alice'` 的节点。

本篇文章的边界只覆盖 Cypher 入门阶段最常用的内容：

- 读查询：`MATCH`、`WHERE`、`RETURN`
- 写操作：`CREATE`、`MERGE`、`SET`、`DELETE`
- 基础优化：索引、锚点查询、避免全图扫描
- 常见业务查询：共同作品、二跳推荐、按属性查节点

不展开的内容包括：

- 复杂事务控制
- 多数据库集群部署
- 图算法库的完整使用
- APOC 扩展过程
- 大规模批量导入与离线 ETL

也就是说，本文解决的是“如何理解并写出能工作的 Cypher 查询”，而不是“如何把 Neo4j 运维成一个大规模生产平台”。

一个很小的玩具例子可以先建立直觉。假设图里有三个人：

- Alice
- Bob
- Carol

其中 Alice 认识 Bob，Bob 认识 Carol。那么图模式可以写成：

```cypher
MATCH (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person)
RETURN a.name, b.name
```

如果数据库里存在 `Alice -> Bob` 这条关系，就返回 `Alice, Bob`。这里的关键不是把三个人看成三行数据，而是把“谁连到谁”当作一等公民处理。

---

## 核心机制与推导

Cypher 的核心机制，是用模式匹配图结构。模式就是一段“图的形状描述”。

例如：

```cypher
(tom:Person)-[:ACTED_IN]->(movie:Movie)<-[:ACTED_IN]-(kevin:Person)
```

这段模式可以读成：

1. 找一个叫 `tom` 的 `Person` 节点。
2. 它通过 `ACTED_IN` 关系连接到某个 `Movie`。
3. 同一个 `Movie` 还被另一个叫 `kevin` 的 `Person` 节点通过 `ACTED_IN` 指向。

只要图里存在这样的结构，这个模式就能匹配出结果。

这就是声明式的关键点：你不写“先从 Tom 出发，再遍历边，再查倒排表，再 join Kevin”，而是直接声明一个目标结构。优化器负责决定执行计划。

这个过程可以抽象成：

$$
Result = \{(v_1, v_2, \dots) \mid Graph \models Pattern \land Predicate\}
$$

白话解释是：结果集由所有满足“图里存在这个模式，且条件成立”的变量绑定组成。

### 1. `MATCH` 如何工作

`MATCH` 会在图里寻找所有满足模式的绑定。所谓“绑定”，就是把变量名和实际节点/关系对应起来。

例如：

```cypher
MATCH (p:Person)-[:KNOWS]->(f:Person)
RETURN p.name, f.name
```

如果图里有：

- Alice KNOWS Bob
- Alice KNOWS Carol

那么结果会有两行：

| p.name | f.name |
| --- | --- |
| Alice | Bob |
| Alice | Carol |

因为 `p` 可以绑定到 Alice，而 `f` 可以分别绑定到 Bob 和 Carol。

### 2. `WHERE` 不是独立查询，而是模式约束

`WHERE` 的作用不是“查完以后再额外做一次筛选”这么简单，它更准确地说，是给当前模式增加条件。

```cypher
MATCH (p:Person)-[:KNOWS]->(f:Person)
WHERE p.name = 'Alice'
RETURN f.name
```

这意味着：只保留起点叫 Alice 的那些路径。

因此，新手常犯的一个错误是把 `WHERE` 当成“随便放的过滤器”，结果写出可读性很差的查询。更合理的思路是：先想清楚你的图模式，再把限制条件放到最贴近模式的位置。

### 3. `CREATE`、`MERGE`、`SET`、`DELETE` 的职责

图的 CRUD 可以按这个表记忆：

| 子句 | 作用 | 白话解释 |
| --- | --- | --- |
| `CREATE` | 创建新节点/关系 | 不管重复不重复，直接新建 |
| `MERGE` | 匹配或创建 | 有就复用，没有才创建 |
| `SET` | 更新属性或标签 | 给对象补字段或改字段 |
| `DELETE` | 删除节点或关系 | 把对象从图里删掉 |

其中最重要的是 `MERGE`。它的价值在于幂等。幂等的白话解释是：同一个操作重复执行多次，结果仍然一致。

例如：

```cypher
MERGE (ann:Person {name: 'Ann'})
ON CREATE SET ann.created = timestamp()
ON MATCH SET ann.lastSeen = timestamp()
RETURN ann.name, ann.created, ann.lastSeen
```

推导逻辑是：

- 如果不存在 `Person {name: 'Ann'}`，就创建它，并执行 `ON CREATE`
- 如果已经存在，就复用它，并执行 `ON MATCH`

所以，`MERGE` 不是“先查一次，再 if else”，而是图数据库层提供的“匹配或创建”语义。

可以把它理解成一个简化公式：

$$
\texttt{MERGE(pattern)} =
\begin{cases}
\text{bind existing pattern}, & \text{if matched}\\
\text{create pattern}, & \text{if not matched}
\end{cases}
$$

但要注意，`MERGE` 的匹配单位是“整个模式”。这点很关键。

例如：

```cypher
MERGE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})
```

它尝试匹配的是“从 Alice 到 Bob 的这整条关系模式”，而不只是两个节点。若节点存在但关系不存在，仍会创建关系。这是很多人第一次用 `MERGE` 时误判的地方。

### 4. 方向、标签、属性共同决定匹配范围

在 Cypher 中，模式越具体，匹配范围越小。

| 模式 | 含义 | 匹配范围 |
| --- | --- | --- |
| `(n)` | 任意节点 | 最大 |
| `(n:Person)` | 任意 `Person` 节点 | 更小 |
| `(n:Person {name:'Alice'})` | 名叫 Alice 的 `Person` | 最小 |
| `(a)-[:KNOWS]->(b)` | 任意 `KNOWS` 关系 | 取决于关系数量 |

因此，工程上常说“先找锚点节点，再扩展关系”。锚点的白话解释是：你能先精准锁定的那个起始点。比如先用 `name='Tom Hanks'` 定位演员节点，再从这个节点扩出去找电影。

---

## 代码实现

下面先给一个完整的玩具例子，再给一个真实工程例子。

### 玩具例子：三个人的认识关系

假设我们要构造如下图：

- Alice 认识 Bob
- Alice 认识 Carol

创建数据：

```cypher
CREATE (alice:Person {name: 'Alice'})
CREATE (bob:Person {name: 'Bob'})
CREATE (carol:Person {name: 'Carol'})
CREATE (alice)-[:KNOWS]->(bob)
CREATE (alice)-[:KNOWS]->(carol)
```

查询 Alice 认识谁：

```cypher
MATCH (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person)
RETURN a.name, b.name
```

结果应当是两行：`Alice, Bob` 和 `Alice, Carol`。

如果不想重复创建 Alice，可以改成：

```cypher
MERGE (alice:Person {name: 'Alice'})
MERGE (bob:Person {name: 'Bob'})
MERGE (alice)-[:KNOWS]->(bob)
```

这更接近工程场景，因为重复执行不会无限制造重复节点。

### 可运行 Python 示例：用邻接表模拟 Cypher 模式匹配

下面这段 Python 不会连接 Neo4j，但它能模拟“节点 + 有向边 + 模式查找”的核心思想。邻接表的白话解释是：用字典存“每个点连向哪些点”。

```python
graph = {
    "Alice": ["Bob", "Carol"],
    "Bob": ["Carol"],
    "Carol": []
}

def match_knows(graph, start_name):
    return [(start_name, target) for target in graph.get(start_name, [])]

def common_neighbors(graph, a, b):
    sa = set(graph.get(a, []))
    sb = set(graph.get(b, []))
    return sorted(sa & sb)

# 模拟:
# MATCH (a:Person {name:'Alice'})-[:KNOWS]->(b:Person) RETURN a.name, b.name
rows = match_knows(graph, "Alice")
assert rows == [("Alice", "Bob"), ("Alice", "Carol")]

# 模拟共同邻居
graph2 = {
    "Tom": ["MovieA", "MovieB"],
    "Kevin": ["MovieB", "MovieC"]
}
assert common_neighbors(graph2, "Tom", "Kevin") == ["MovieB"]
```

这个例子说明，Cypher 的本质确实和“从图里找满足结构的绑定”一致。真实数据库做得更多，比如索引、执行计划、去重、事务，但底层查询思想没有变。

### 真实工程例子：查询 Tom Hanks 和 Kevin Bacon 的共同作品

这是 Neo4j 教学中最经典的例子之一。目标是：已知两个演员，找他们共同出演的电影。

```cypher
MATCH (tom:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(movie:Movie)<-[:ACTED_IN]-(kevin:Person {name: 'Kevin Bacon'})
RETURN movie.title
```

这条查询可以拆成三步理解：

1. 用 `Person {name: ...}` 锁定第一个演员。
2. 从他出发，沿 `ACTED_IN` 找到电影。
3. 继续要求第二个演员也通过 `ACTED_IN` 指向同一部电影。

如果只返回 `movie.title`，就得到了共同作品列表。

### 建索引：让锚点查询更快

如果经常按 `Person.name` 或 `Movie.title` 查找节点，应先建立索引：

```cypher
CREATE INDEX person_name_index FOR (n:Person) ON (n.name);
CREATE INDEX movie_title_index FOR (n:Movie) ON (n.title);
```

索引的白话解释是：给常查的字段建一个“可快速定位的目录”，避免每次都把所有节点扫一遍。

这样，下面这类查询更容易被优化器高效执行：

```cypher
MATCH (p:Person {name: 'Tom Hanks'})
RETURN p
```

工程上推荐的顺序通常是：

1. 先根据有索引的属性找到锚点节点。
2. 再沿关系做扩展。
3. 最后只返回必要字段。

### 基础 CRUD 示例

创建节点：

```cypher
CREATE (:Person {name: 'Alice', born: 1990})
```

创建关系：

```cypher
MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
CREATE (a)-[:KNOWS]->(b)
```

更新属性：

```cypher
MATCH (a:Person {name: 'Alice'})
SET a.city = 'Shanghai'
RETURN a.name, a.city
```

删除关系：

```cypher
MATCH (:Person {name: 'Alice'})-[r:KNOWS]->(:Person {name: 'Bob'})
DELETE r
```

删除节点前，通常要先删除关系，否则会报错：

```cypher
MATCH (a:Person {name: 'Alice'})
DETACH DELETE a
```

`DETACH DELETE` 的白话解释是：把这个节点相关的边一起断开并删除。

---

## 工程权衡与常见坑

### 1. 没有锚点条件，`MATCH` 可能退化成全图扫描

下面这种写法在小图上能跑，在大图上可能非常慢：

```cypher
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
RETURN p.name, m.title
```

原因不是语法错，而是约束太弱。数据库可能需要从大量 `Person` 节点开始扩展，代价高。

更稳妥的方式是先锚定：

```cypher
MATCH (p:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie)
RETURN m.title
```

或者：

```cypher
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE p.name = 'Tom Hanks'
RETURN m.title
```

这两种逻辑接近，但工程上仍要结合索引和执行计划看实际效果。

### 2. 不要把 `id()` 当业务主键

Neo4j 内部有节点 ID，但内部 ID 的问题是可能被重用。也就是说，一个节点删除后，将来新节点可能拿到相同的内部 ID。

因此，下面这种设计不安全：

- 对外暴露 `id(n)` 给前端
- 在业务系统里长期依赖它做唯一标识

更稳妥的做法是：

- 自己维护业务主键
- 或使用 UUID
- 并建立唯一约束

例如：

```cypher
MERGE (p:Person {uuid: '4f2a...'}) 
SET p.name = 'Alice'
```

### 3. `MERGE` 过大，容易产生意外结果

很多人会一次性把一整串模式都放进 `MERGE`，以为这是“避免重复”的万能写法。实际上，`MERGE` 匹配的是整体模式，模式太大时，任何一部分不匹配都可能触发创建。

通常更稳的写法是分步：

```cypher
MERGE (a:Person {name: 'Alice'})
MERGE (b:Person {name: 'Bob'})
MERGE (a)-[:KNOWS]->(b)
```

这比一口气 `MERGE` 整个链条更容易推理，也更容易排查重复数据问题。

### 4. 返回整个节点，常常比返回字段更重

下面的查询：

```cypher
MATCH (p:Person {name: 'Tom Hanks'})
RETURN p
```

会返回整个节点对象。对调试有用，但对接口层未必合适。

若业务只需要名字和出生年份，应该显式返回：

```cypher
MATCH (p:Person {name: 'Tom Hanks'})
RETURN p.name, p.born
```

原则是：只返回需要的数据，避免传输和序列化冗余。

### 5. 常见可用索引的谓词要有基本概念

对新手来说，不必一开始就记全部索引细节，但至少要知道，某些谓词更容易利用索引。

| 谓词 | 例子 | 典型用途 |
| --- | --- | --- |
| `=` | `p.name = 'Alice'` | 精确查找 |
| `IN` | `p.name IN ['Alice', 'Bob']` | 多值匹配 |
| `STARTS WITH` | `p.name STARTS WITH 'Al'` | 前缀搜索 |
| 范围比较 | `m.released > 2010` | 数值或时间区间 |

这张表的意义不是让你背语法，而是帮助判断：哪些条件更像“可查目录”，哪些条件更像“只能扫一遍再算”。

---

## 替代方案与适用边界

Cypher 很适合以下场景：

- 实体与关系天然重要，比如社交网络、知识图谱、推荐链路、权限依赖
- 查询问题本质是“找路径”“找邻居”“找共同连接”
- 需要灵活表达多跳关系

但它也不是所有数据问题的默认答案。

### 1. 如果只是简单 CRUD，关系型数据库可能更直接

如果你的业务主要是：

- 用户表
- 订单表
- 支付表
- 按主键查、按时间筛、做报表聚合

而跨表关系并不复杂，那么传统关系型数据库通常更成熟，团队也更熟悉。

图数据库的价值，主要体现在“关系本身是核心数据”的场景，而不是“所有数据都适合画成图”。

### 2. 如果只是做二跳推荐，纯 Cypher 就够了

例如，找“与 Tom Hanks 合作过的人合作过、但 Tom 自己还没合作过的人”，可以写成二跳推荐：

```cypher
MATCH (tom:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(movie1:Movie)<-[:ACTED_IN]-(coActor:Person)-[:ACTED_IN]->(movie2:Movie)<-[:ACTED_IN]-(coCoActor:Person)
WHERE tom <> coCoActor
  AND NOT (tom)-[:ACTED_IN]->(:Movie)<-[:ACTED_IN]-(coCoActor)
RETURN coCoActor.name
```

这类查询的优点是：

- 逻辑透明
- 不需要额外算法库
- 对中小规模图足够实用

### 3. 如果要做更复杂推荐，可能需要 Graph Data Science

当问题升级为：

- 相似用户推荐
- 相似商品发现
- 高维图 embedding
- 社区发现、中心性分析

这时纯 Cypher 往往不够。需要进入 GDS，也就是 Graph Data Science，白话解释是：图上的算法工具箱。

典型流程可以理解成这条 pipeline：

| 阶段 | 作用 | 白话解释 |
| --- | --- | --- |
| 构建图投影 | 把业务图转成算法可处理结构 | 先准备算法输入 |
| 计算 embedding | 把节点变成向量 | 给节点生成可比较的数值表示 |
| 运行 kNN | 查相似节点 | 找“谁和谁最像” |
| 用 Cypher 回查 | 取业务字段和明细 | 把算法结果还原成可展示数据 |

因此，适用边界可以简单总结为：

- 关系查询、路径查询、二跳协同过滤：先用 Cypher
- 更复杂的相似度学习和推荐：考虑 GDS
- 纯表结构业务、弱关系场景：未必需要图数据库

---

## 参考资料

- Neo4j Cypher Manual “Patterns” 入门介绍图模式与 `MATCH` 表达方式：<https://neo4j.com/docs/cypher-manual/4.0/syntax/patterns/>
- Neo4j Cypher Manual “MATCH” 章节：<https://neo4j.com/docs/cypher-manual/current/clauses/match/>
- Neo4j Cypher Manual “MERGE” 章节：<https://neo4j.com/docs/cypher-manual/current/clauses/merge/>
- Neo4j Getting Started “Updating the graph”：<https://neo4j.com/docs/getting-started/cypher/updating/>
- Neo4j 搜索性能索引文档：<https://neo4j.com/docs/cypher-manual/current/indexes/search-performance-indexes/create-indexes/>
- Neo4j Knowledge Base “How deletes work in Neo4j”：<https://neo4j.com/developer/kb/how-deletes-workin-neo4j/>
- Neo4j GraphAcademy “Introduction to Cypher”：<https://graphacademy.neo4j.com/courses/cypher-fundamentals/1-reading/1-intro-cypher/>
- Neo4j Classroom / Intro Training 示例资料：<https://neo4j-contrib.github.io/asciidoc-slides/content/training/intro/classroom/>
- Neo4j Graph Data Science FastRP + kNN 示例：<https://neo4j.com/docs/graph-data-science/current/getting-started/fastrp-knn-example/>
- Neo4j 推荐引擎教程：<https://neo4j.com/docs/getting-started/appendix/tutorials/guide-build-a-recommendation-engine/>
