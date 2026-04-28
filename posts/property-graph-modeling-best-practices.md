## 核心结论

属性图建模的重点，不是“把数据画成点和线”，而是按照未来的查询方式，决定一个概念应该落在节点、边、属性，还是中间节点上。节点是“能单独存在并被反复引用的对象”，边是“两个对象之间的一次关系事实”，属性是“附着在对象或关系上的说明字段”，中间节点是“本身也需要被查询和承载上下文的事件或关系对象”。

可以把这个判断写成一个简化映射函数：

$$
m(x)\in \{\text{node},\text{edge},\text{prop},\text{intermediate\ node}\}
$$

其中最常用的规则是：

| 结构 | 适合表达什么 | 典型判断信号 | 常见误用 |
| --- | --- | --- | --- |
| 节点 | 独立实体、可复用概念 | 有独立身份、会被多次连接、会被多跳遍历 | 把一次性事件也建成普通实体 |
| 边 | 两个节点之间的单次关系事实 | 只涉及两个端点，语义明确 | 在边上堆过多上下文，导致关系过胖 |
| 属性 | 展示字段、低频过滤条件 | 不需要独立引用，不参与多跳 | 高基数值直接塞成大数组 |
| 中间节点 | 多实体参与的事件或超边 | 关系本身有身份、时间、金额、状态等 | 本该独立成节点的事件被硬塞成一条边 |

一个玩具例子最能说明问题。`User` 买 `Item`，如果业务只关心“买过没”，`(u)-[:BOUGHT]->(i)` 就够了；如果还要记录订单号、金额、时间、门店、支付方式，这次“购买”已经不只是连线，而是一件可以被查询、审计、回放的事情，应该提升为 `Purchase` 节点。

对初学者最实用的一句话是：先想“以后会怎么查它”，再决定“它应不应该单独成点”。

---

## 问题定义与边界

本文讨论的是属性图模型本身的建模原则，不绑定某一家图库或某一种查询语言。无论底层是 Neo4j、NebulaGraph、JanusGraph 还是兼容 TinkerPop 的系统，只要核心结构是“节点 + 有向边 + 属性”，这些规则都成立。

边界划分是新手最容易犯错的地方。很多人会把“所有名词”都建成节点，这通常会导致图结构膨胀、遍历放大、语义反而更模糊。判断边界时，先区分三类东西：

1. 业务实体：能独立存在，例如用户、商品、设备、商户。
2. 关系事实：连接实体的一次具体行为，例如购买、登录、转账、访问。
3. 字段信息：只是说明用的值，例如昵称、颜色、创建时间、状态码。

下面这张表适合做初步判断：

| 判断问题 | 是 | 否 |
| --- | --- | --- |
| 是否有独立身份，能被单独引用？ | 倾向节点 | 倾向边或属性 |
| 是否会被多跳遍历？ | 倾向节点 | 倾向属性 |
| 是否会被多个对象复用？ | 倾向节点 | 倾向属性 |
| 是否只是展示或低频过滤字段？ | 倾向属性 | 继续判断 |
| 是否涉及两个以上实体共同参与？ | 倾向中间节点 | 可先考虑普通边 |

例如 `国家`、`城市`、`用户标签` 这类概念，不一定都要建成节点。如果 `country="CN"` 只是注册信息里的一个筛选字段，而且很少沿着“国家 -> 用户 -> 订单”这样的路径遍历，那么它保留为属性通常更简单。反过来，如果“城市”会被多处复用，比如要分析“城市内商户聚集、跨城市迁移、同城配送网络”，那它就值得独立成节点。

这里有一个容易忽略的边界：高基数标签。高基数，白话说就是“不同取值特别多”，例如一百万种商品标签 ID、海量关键词、海量设备指纹。如果把它们全塞到属性数组里，索引、去重、更新和过滤都会变重；如果这些值经常被复用和关联，拆成节点往往更稳。

所以，建模不是词性判断，而是查询判断。不是“它是不是名词”，而是“它以后是不是要被反复连接、过滤、遍历、更新”。

---

## 核心机制与推导

属性图可以抽象为：

$$
G=(V,E,\phi_V,\phi_E)
$$

其中：

- $V$ 是节点集合，也就是图中的对象。
- $E$ 是边集合，也就是对象之间的有向关系。
- $\phi_V$ 是节点属性函数，白话说就是“每个节点带哪些字段”。
- $\phi_E$ 是边属性函数，白话说就是“每条边带哪些字段”。

这个公式很简单，但它直接对应建模决策。因为一个业务概念 $x$ 最终只能落在四种位置中的一种：节点、边、属性、中间节点。可以把判断规则整理成下面的决策表：

| 条件 | 建模建议 | 原因 |
| --- | --- | --- |
| 需要独立身份 | 节点 | 能单独被引用、更新、去重 |
| 只表示两个实体之间的一次事实 | 边 | 语义直接，遍历成本低 |
| 只是过滤或展示字段 | 属性 | 不必增加图结构复杂度 |
| 一个关系要挂多类上下文 | 中间节点 | 避免边过胖或语义不清 |
| 一个事实涉及 3 个及以上实体 | 中间节点 | 普通边本质上只连两个端点 |

这套规则背后的逻辑可以再推一步。

第一，节点代表“身份”。如果某个对象会被多次引用，或者以后可能挂更多关系，它就不应该只是某个属性值。比如 `Merchant`、`Device`、`IP` 都有自己的生命周期和关系网络，它们天然适合做节点。

第二，边代表“事实发生”。边最适合表达“谁和谁之间发生了什么”。例如 `(:User)-[:FOLLOWS]->(:User)`、`(:Author)-[:WROTE]->(:Post)`。这类语义只要端点明确、上下文不复杂，用边最自然。

第三，属性代表“说明”。例如用户昵称、商品价格区间、文章摘要。它们通常不需要继续往外连，也不值得为一个简单字段新建一个节点。

第四，中间节点代表“这件事本身也是对象”。这类情况最典型。比如支付风控里的 `Transaction`：它同时连接 `User`、`Device`、`IP`、`Merchant`，又承载 `amount`、`risk_score`、`ts`、`valid_from`、`valid_to`。如果硬把这些东西塞到某一条边上，就会出现两个问题：

1. 一条边只能自然描述两个端点，超出两个端点后语义开始扭曲。
2. 上下文字段会被复制到多条边，导致一致性变差。

所以 `Transaction` 应该是节点，不是线。白话说：不是“用户和商户之间有一条交易边”，而是“有一笔交易，这笔交易连到用户、商户、设备和 IP”。

玩具例子可以再具体一点：

- 简单建模：`(User1)-[:BOUGHT]->(ItemA)`
- 升级建模：`(User1)-[:MADE]->(Purchase1001)-[:OF]->(ItemA)`

如果还要记录门店，则继续：

- `(Purchase1001)-[:AT]->(Store7)`

这样以后就能直接回答：“User1 在 Store7 买过哪些 Item？”“某时间窗内 Store7 发生了哪些 Purchase？”“某支付方式关联了哪些高风险 Purchase？”这些问题都围绕 `Purchase` 展开，而不是在多条边上拼字段。

---

## 代码实现

写代码前，先写查询样例，再反推模型，这是工程上最稳的方法。因为图模型不是先建完再想怎么查，而是先明确“最贵的查询是什么”。

下面先给一个建模判定函数。它不是数学上完备的算法，而是把常见经验转成可以执行的规则。

```python
def choose_mapping(
    has_identity: bool,
    multi_hop: bool,
    reusable: bool,
    display_only: bool,
    multi_entity: bool,
    rich_context: bool,
) -> str:
    if display_only and not multi_hop and not reusable:
        return "property"
    if multi_entity or rich_context:
        return "intermediate_node"
    if has_identity or multi_hop or reusable:
        return "node"
    return "edge"


# 玩具例子：购买行为只有“买过”语义
assert choose_mapping(
    has_identity=False,
    multi_hop=False,
    reusable=False,
    display_only=False,
    multi_entity=False,
    rich_context=False,
) == "edge"

# 真实工程例子：交易事件要连接多个实体并承载上下文
assert choose_mapping(
    has_identity=True,
    multi_hop=True,
    reusable=True,
    display_only=False,
    multi_entity=True,
    rich_context=True,
) == "intermediate_node"

# 国家只用于展示和过滤
assert choose_mapping(
    has_identity=False,
    multi_hop=False,
    reusable=False,
    display_only=True,
    multi_entity=False,
    rich_context=False,
) == "property"
```

如果用 Cypher 表达“用户在某门店购买某商品”，推荐这样建：

```cypher
CREATE (u:User {id: 'User1'})
CREATE (s:Store {id: 'Store7'})
CREATE (i:Item {id: 'ItemA'})
CREATE (p:Purchase {
  id: 'Purchase#1001',
  amount: 59,
  payment_method: 'card',
  ts: date('2026-04-01')
})
CREATE (u)-[:MADE]->(p)
CREATE (p)-[:AT]->(s)
CREATE (p)-[:OF]->(i)
```

这个结构有两个优点。第一，同一事实只存一次，不需要一边写 `User -> Item`，另一边再写 `User -> Store`，然后靠订单号补逻辑。第二，购买事件上的时间、金额、支付方式都聚合在 `Purchase` 上，不会散落在多条边里。

查询时也更直接。比如查某设备在时间窗内关联的交易：

```cypher
MATCH (d:Device {id: $deviceId})<-[:USED_BY]-(t:Transaction)
WHERE t.ts >= datetime($start) AND t.ts < datetime($end)
RETURN t.id, t.amount, t.risk_score, t.ts
ORDER BY t.ts DESC
```

再看一个真实工程例子：支付风控。常见结构是：

- `(:User)-[:INITIATED]->(:Transaction)`
- `(:Transaction)-[:USING_DEVICE]->(:Device)`
- `(:Transaction)-[:FROM_IP]->(:IP)`
- `(:Transaction)-[:TO_MERCHANT]->(:Merchant)`

这样可以直接做团伙分析，例如“找出与高风险交易共享过设备和 IP 的账户集合”。如果把 `Device`、`IP` 仅仅塞成交易属性，这种跨交易的共享关系就很难做多跳遍历。

实现上还有一个重要原则：不要为对称语义默认存两条方向相反的边。比如“认识”“相邻”“同现”这类关系，若业务语义本身对称，只保留一个规范方向，查询时按需处理。重复存两份，更新成本和遍历分支都会翻倍。

---

## 工程权衡与常见坑

高质量建模不是把一切都拆得最细，而是在查询速度、写入复杂度、维护成本之间找平衡。下面这些坑最常见。

| 坑名 | 典型表现 | 后果 | 修正方式 |
| --- | --- | --- | --- |
| supernode | 所有用户都连到同一个大节点，如 `China`、`AllTags` | 扩展一步就爆炸，遍历成本高 | 只在确实需要遍历时建点；必要时分层、分桶、加索引 |
| 双向边冗余 | 同一事实同时存 `A->B` 和 `B->A` | 数据重复、更新不一致 | 约定单向存储，查询时控制方向 |
| 超边硬塞 | 一次事件涉及多实体，却只用一条边表示 | 语义扭曲，上下文无处安放 | 改为中间节点 |
| 时间重复建模 | 用多条重复边表达不同时间片 | 边数量膨胀，难维护 | 时间做边属性或事件节点属性 |
| 层级过深 | 为了“看起来完整”引入很多纯展示层 | 遍历链路变长，查询变慢 | 只保留参与查询的层级 |
| 属性爆炸 | 高基数标签、长列表都塞进数组属性 | 索引难做，过滤低效 | 复用频繁的值拆为节点或代码表 |

先看 `supernode`。这类节点的特点是扇入或扇出极高。扇出，白话说就是“从这个点能连出去很多边”；扇入则相反。比如一个“全站标签”节点连着上百万篇文章，理论上没错，但实际遍历时第一跳就会把候选集放大到不可控。解决方式不是永远不用这种节点，而是问清楚：这个中心点是不是查询的必要中转站。如果只是展示分类，不一定值得独立成图中的枢纽。

再看时间和权重。很多系统会把“2026-04-01 登录一次”和“2026-04-02 又登录一次”建成两条同类型边，甚至几十条。这会让“同一关系”被拆成很多重复记录。更稳的做法有两种：

1. 如果只是简单上下文，把 `ts`、`weight`、`status` 放到边属性上。
2. 如果事件本身重要，把它提升为事件节点。

这里的判断标准不是字段多少，而是事件是否值得单独查询。比如“关注关系的创建时间”通常放边属性就够；但“转账记录”涉及金额、通道、风控分、到账状态，就更适合做事件节点。

还有一个常见误区是“为了规范而过度分层”。例如 `国家 -> 省 -> 市 -> 区 -> 街道 -> 门店` 全部建全，但业务查询只关心门店和城市。这时多出来的层级只会增加跳数，不会增加业务价值。图模型不是行政区树的数字复刻，而是面向查询的关系抽象。

---

## 替代方案与适用边界

属性图不是默认最优解。只有当“关系本身是主角”，并且多跳遍历是核心诉求时，图模型的优势才真正明显。

如果你的数据主要是单表过滤、主键查询、报表统计，那么关系型数据库通常更简单。比如只存“用户姓名、手机号、注册时间、最近登录时间”，一张 `users` 表就够了。硬上图，只会让模型更绕。

如果你的重点是严格三元语义、本体约束、跨系统知识交换，那么 RDF 更合适。RDF 可以理解成“以主语-谓语-宾语三元组为基本单位的知识表示方式”，它在语义标准化和本体推理上更强，但工程门槛和建模约束也更高。

可以做一个直接对比：

| 方案 | 适合的查询模式 | 建模复杂度 | 维护成本 | 典型适用场景 |
| --- | --- | --- | --- | --- |
| 关系型数据库 | 单表过滤、聚合、事务写入 | 低 | 低到中 | 订单、账户、后台管理 |
| 属性图 | 多跳关系遍历、路径发现、关联分析 | 中 | 中 | 风控、推荐、知识图谱、调用链 |
| RDF | 三元语义、推理、本体驱动交换 | 高 | 高 | 语义互操作、标准知识库 |

一个真实对比很典型。

如果你只要回答：“这个用户什么时候注册、手机号是什么、最近一笔订单金额多少？”关系型数据库最直接。

如果你要回答：“某设备三跳内关联了哪些高风险账户，这些账户是否共享过 IP、商户和收货地址？”属性图明显更合适，因为查询核心不再是字段筛选，而是路径扩展。

如果你要回答：“某个概念在跨系统语义标准里如何与其他概念对齐，并支持本体级推理？”那 RDF 往往比属性图更自然。

所以适用边界可以总结成一句话：当关系需要被走出来，图才有价值；当关系只是外键，图未必值得。

---

## 参考资料

1. [Neo4j: Graph Database Concepts](https://neo4j.com/docs/getting-started/appendix/graphdb-concepts/)
2. [Neo4j: Graph Modeling Tips](https://neo4j.com/docs/getting-started/data-modeling/modeling-tips/)
3. [Neo4j: Modeling Designs - Intermediate Nodes](https://neo4j.com/docs/getting-started/data-modeling/modeling-designs/)
4. [Neo4j GraphAcademy: Graph Data Modeling Core Principles](https://neo4j.com/graphacademy/training-gdm-40/03-graph-data-modeling-core-principles/)
5. [Apache TinkerPop Reference Documentation](https://tinkerpop.apache.org/docs/current/reference/)
6. [Amazon Neptune: Prefer Directed to Bi-directional Edges in Queries](https://docs.aws.amazon.com/neptune/latest/userguide/best-practices-opencypher-directed-edges.html)
7. [A Model and Query Language for Temporal Graph Databases](https://link.springer.com/article/10.1007/s00778-021-00675-4)
