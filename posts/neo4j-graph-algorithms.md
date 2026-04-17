## 核心结论

Neo4j Graph Data Science，简称 GDS，就是把常见图算法封装成数据库里的可执行过程。对初学者最重要的结论是：在同一个图投影上，你可以连续完成三件事。

1. 用 PageRank 找“重要节点”。重要的意思不是“连得多”，而是“被重要节点指向得多”。
2. 用 Louvain 找“社区”。社区就是内部连接更密、外部连接更稀的节点群。
3. 用最短路径算法找“传播链”。传播链就是从起点到终点总代价最小的一条关系序列。

这三类算法组合起来，正好覆盖很多真实工程任务：知识图谱里的影响力分析、支付网络里的欺诈团伙识别、推荐系统里的关联路径查询。GraphAcademy 的欺诈检测流程就是先做社区发现，再在社区内部做中心性排序，最后把路径结果交给调查流程继续核验，这种串联方式比单独跑某一个算法更接近生产环境。

| 类别 | 典型算法 | 回答的问题 | 更像什么岗位动作 |
|---|---|---|---|
| 中心性 | PageRank、Degree、Eigenvector | 谁更关键 | 排名、筛选重点对象 |
| 社区发现 | Louvain、Leiden、WCC、Label Propagation | 哪些节点天然成团 | 切群、分层、找团伙 |
| 路径 | Dijkstra、A*、Yen | 两点之间怎么连过去 | 查传播链、推荐链、调查链 |

核心理解只有一句：图算法不是互斥选择，而是围绕同一张投影图做不同视角的计算。官方文档也把 PageRank 放在中心性分类、Louvain 放在社区发现分类、Dijkstra/A* 放在路径分类中，说明它们本来就是同一工作流里的不同环节。参考：[PageRank 文档](https://neo4j.com/docs/graph-data-science/current/algorithms/page-rank/)、[Community 文档](https://neo4j.com/docs/graph-data-science/current/algorithms/community/)、[Pathfinding 文档](https://neo4j.com/docs/graph-data-science/current/algorithms/pathfinding/)。

---

## 问题定义与边界

先把三个问题分开，否则很容易把算法用错。

PageRank 解决的是“重要性排名”。它把一条入边看成一次投票，但不是所有投票等价，高分节点投出来的票更值钱。其经典形式是：

$$
PR(v)=\frac{1-d}{N}+d\sum_{u\rightarrow v}\frac{PR(u)}{C(u)}
$$

其中：

- $PR(v)$ 是节点 $v$ 的分数。
- $d$ 是阻尼因子，白话说就是“随机继续沿边走下去”的概率，常用 $0.85$。
- $N$ 是图中节点总数。
- $C(u)$ 是节点 $u$ 的出度，也就是它往外连了多少条边。

如果图是加权的，GDS 会把“按出度平均分配”改成“按边权比例分配”。所以权重不是装饰字段，它会直接改写分数传播方式。

Louvain 解决的是“怎么把节点分群更合理”。它优化的是模块度 $Q$。模块度可以理解成“社区内部真实连接密度，比随机连接基线高多少”。常见定义是：

$$
Q=\frac{1}{2m}\sum_{i,j}\left(A_{ij}-\frac{k_i k_j}{2m}\right)\delta(c_i,c_j)
$$

其中：

- $A_{ij}$ 表示节点 $i,j$ 之间是否有边或边权大小。
- $k_i,k_j$ 是节点的度或加权度。
- $m$ 是图中总边权的一半。
- $\delta(c_i,c_j)$ 是指示函数，两个节点在同一社区时取 1，否则取 0。

最短路径解决的是“从 A 到 B 最便宜怎么走”。这里的“便宜”通常是边权和最小：

$$
\text{cost}(p)=\sum_{e\in p} w(e)
$$

如果权重表示距离，就找最短距离；如果权重表示风险或传播代价，就找最小风险链路。

一个玩具例子最容易看懂。设一个 8 页网站图：`home` 指向 `about`、`links`、`product`，其中到 `product` 的边权是 `0.6`，到 `about` 和 `links` 各是 `0.2`；而 `links` 再以 `0.8` 指回 `home`，并用 `0.05` 平均指向四个外部页面。这说明两件事：

1. PageRank 关心“高权重边把更多分数送给谁”。
2. Louvain 关心“哪些页面更像一个内聚子群”。

因此，任务边界应当这样判断：

- 你要做排序，用中心性。
- 你要做分团，用社区发现。
- 你要做链路追踪，用路径算法。
- 你要同时做这三件事，就在同一投影图上串起来做，而不是分别维护三套数据。

参考：[PageRank 介绍](https://neo4j.com/docs/graph-data-science/current/algorithms/page-rank/)、[Louvain 机制说明](https://www.puppygraph.com/blog/louvain)。

---

## 核心机制与推导

### 1. PageRank 为什么能识别“核心节点”

PageRank 本质上是一个迭代过程。开始时每个节点先分到一个初始分数，之后每轮都把当前分数沿出边分发出去，再收回别人传来的分数。迭代多轮后，分数会逐步稳定，这个稳定值就是节点重要性。

它比“只看入度”更强，因为它区分了两种入边：

- 来自普通节点的入边
- 来自本身就高分节点的入边

后者影响更大。所以“被谁指向”比“被多少节点指向”更重要。

继续看上面的玩具例子。`home -> product` 的权重是 `0.6`，而 `home -> links` 是 `0.2`。这意味着从 `home` 流出去的影响力里，`product` 拿到的份额比 `links` 更大。Neo4j 官方的加权 PageRank 示例正是用这个结构来展示“边权越大，贡献越大”的效果。

### 2. Louvain 为什么能找到社区

Louvain 分两步循环：

1. 初始时每个节点单独成社区，尝试把节点移动到邻居社区，看模块度能否上升。
2. 当当前层再也提不动模块度时，把每个社区压缩成一个“超级节点”，继续在更高层重复。

所以 Louvain 不是一次分群，而是“局部改进 + 层级压缩”的递归过程。GDS 的 `maxLevels` 控制最多跑多少层；`includeIntermediateCommunities` 则决定是否把每一层的社区 ID 都保留下来。对初学者来说，这个参数很重要，因为它能让你看到“社区是怎么从小团合并成大团”的，而不是只看最终答案。

### 3. 路径算法为什么通常放在最后

路径算法一般依赖前两步的结果来缩小搜索范围。

- 如果已经知道某个社区是高风险团伙，就只在该社区内做最短路径。
- 如果已经知道几个高 PageRank 节点是核心账户，就从这些节点出发找传播链。

真实工程例子可以用欺诈检测来理解。GraphAcademy 的做法是先根据交易、共享卡、共享身份等规则构图，再用社区检测找 fraud rings，也就是欺诈团伙；之后在团伙内部用中心性算法排名可疑节点；最后再根据关系链找调查路径。这样做的好处是，算法不再是“黑箱打分器”，而是形成“团伙 -> 核心人 -> 关键链路”的可解释输出。

---

## 代码实现

先给一个可运行的 Python 玩具实现，用来直观看 PageRank 的传播逻辑。它不是 Neo4j 代码，但足够帮助你理解“权重如何改变得分分配”。

```python
from math import isclose

graph = {
    "home": {"about": 0.2, "links": 0.2, "product": 0.6},
    "about": {"home": 1.0},
    "product": {"home": 1.0},
    "links": {"home": 0.8, "a": 0.05, "b": 0.05, "c": 0.05, "d": 0.05},
    "a": {"home": 1.0},
    "b": {"home": 1.0},
    "c": {"home": 1.0},
    "d": {"home": 1.0},
}

def pagerank_weighted(graph, d=0.85, steps=50):
    nodes = list(graph.keys())
    n = len(nodes)
    pr = {node: 1.0 / n for node in nodes}

    for _ in range(steps):
        new_pr = {node: (1 - d) / n for node in nodes}
        for src, outs in graph.items():
            total_weight = sum(outs.values())
            if total_weight == 0:
                continue
            for dst, w in outs.items():
                new_pr[dst] += d * pr[src] * (w / total_weight)
        pr = new_pr
    return pr

scores = pagerank_weighted(graph)
assert isclose(sum(scores.values()), 1.0, rel_tol=1e-9)
assert scores["home"] > scores["a"]
assert scores["product"] > scores["links"]  # 因为 home 给 product 的权重更高

top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
print(top3)
```

接着看 Neo4j/GDS 的做法。第一步是建样例数据并投影图。

```cypher
CREATE
  (home:Page {name: 'Home'}),
  (about:Page {name: 'About'}),
  (links:Page {name: 'Links'}),
  (product:Page {name: 'Product'}),
  (a:Page {name: 'Site A'}),
  (b:Page {name: 'Site B'}),
  (c:Page {name: 'Site C'}),
  (d:Page {name: 'Site D'}),
  (home)-[:LINKS {weight: 0.2}]->(about),
  (home)-[:LINKS {weight: 0.2}]->(links),
  (home)-[:LINKS {weight: 0.6}]->(product),
  (about)-[:LINKS {weight: 1.0}]->(home),
  (product)-[:LINKS {weight: 1.0}]->(home),
  (a)-[:LINKS {weight: 1.0}]->(home),
  (b)-[:LINKS {weight: 1.0}]->(home),
  (c)-[:LINKS {weight: 1.0}]->(home),
  (d)-[:LINKS {weight: 1.0}]->(home),
  (links)-[:LINKS {weight: 0.8}]->(home),
  (links)-[:LINKS {weight: 0.05}]->(a),
  (links)-[:LINKS {weight: 0.05}]->(b),
  (links)-[:LINKS {weight: 0.05}]->(c),
  (links)-[:LINKS {weight: 0.05}]->(d);

MATCH (source:Page)-[r:LINKS]->(target:Page)
RETURN gds.graph.project(
  'myGraph',
  source,
  target,
  { relationshipProperties: r { .weight } }
);
```

然后先做内存估算，再跑 PageRank。GDS 官方明确建议先估算内存，不够就会阻止执行。

```cypher
CALL gds.pageRank.write.estimate('myGraph', {
  writeProperty: 'pageRank',
  maxIterations: 20,
  dampingFactor: 0.85
})
YIELD nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory;

CALL gds.pageRank.stream('myGraph', {
  relationshipWeightProperty: 'weight',
  maxIterations: 20,
  dampingFactor: 0.85
})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC, name ASC;
```

如果你要做个性化推荐或指定源头传播，可以加入 `sourceNodes`：

```cypher
MATCH (siteA:Page {name: 'Site A'})
CALL gds.pageRank.stream('myGraph', {
  relationshipWeightProperty: 'weight',
  sourceNodes: [siteA],
  dampingFactor: 0.85
})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC;
```

社区发现写法类似：

```cypher
CALL gds.louvain.stats('myGraph', {
  relationshipWeightProperty: 'weight',
  maxLevels: 10
})
YIELD communityCount, modularity, modularities, ranLevels
RETURN communityCount, modularity, modularities, ranLevels;

CALL gds.louvain.stream('myGraph', {
  relationshipWeightProperty: 'weight',
  includeIntermediateCommunities: true
})
YIELD nodeId, communityId, intermediateCommunityIds
RETURN
  gds.util.asNode(nodeId).name AS name,
  communityId,
  intermediateCommunityIds
ORDER BY name ASC;
```

最后，如果你已经锁定某两个账户或页面之间的调查方向，就可以加最短路径：

```cypher
MATCH (source:Page {name: 'Site A'}), (target:Page {name: 'Product'})
CALL gds.shortestPath.dijkstra.stream('myGraph', {
  sourceNode: source,
  targetNode: target,
  relationshipWeightProperty: 'weight'
})
YIELD totalCost, nodeIds
RETURN totalCost, [nodeId IN nodeIds | gds.util.asNode(nodeId).name] AS path;
```

---

## 工程权衡与常见坑

### 1. 不做内存估算，作业可能直接失败

GDS 的算法并不是直接对库表扫一遍，它先在内存里操作投影图。图一大，中心性、社区发现、路径算法都可能吃掉大量内存。所以规范动作应该是：

| 检查项 | 为什么要看 | 常见风险 |
|---|---|---|
| `*.estimate` | 预估算法内存占用 | OOM、任务被系统阻止 |
| `relationshipWeightProperty` | 明确是否按权重计算 | 把加权图误当无权图 |
| `maxLevels` | 控制 Louvain 层数 | 层数太低分得过碎，太高合得过粗 |
| `includeIntermediateCommunities` | 保留中间层级 | 只看最终社区，丢失演化过程 |
| `sourceNodes` | 个性化 PageRank | 把全局影响力误当局部传播力 |

### 2. Louvain 不一定稳定到“唯一正确答案”

Louvain 常见两个问题。

第一，分辨率极限。小而紧密的团可能被更大的团吞掉。第二，退化问题。不同运行或不同参数下，多个方案都可能得到接近的模块度，因此结果并不总是唯一。

GraphAcademy 对 `maxLevels` 的解释很直接：低层数更容易得到很多小社区，高层数更容易合成更少、更大的社区。因此在大图上最好至少对比两组配置，例如 `maxLevels: 1` 和 `maxLevels: 10`，观察 `communityCount` 与 `modularity` 怎么变，再决定你要“更细”还是“更粗”的社区。

### 3. 关系设计比算法名更重要

真实工程里，很多失败不是因为 Louvain 不好，而是因为边定义错了。比如在欺诈图里，把“同一设备”“同一手机号”“同一银行卡”“短时间高频转账”全部等价处理，最后得到的社区往往会过大、过脏。GraphAcademy 明确强调，算法只是工具，真正决定结构含义的是你如何设计关系和投影。

### 4. 路径结果不等于因果关系

最短路径只表示“在当前权重定义下，这条链最便宜”。它不自动等于“真实传播路径”或“真实作案链路”。如果边权设计不合理，比如把高风险边设成低成本，算法就会非常自信地输出错误路径。

---

## 替代方案与适用边界

不是每个问题都要上 PageRank + Louvain。

如果只是想做一个轻量排名，Degree Centrality 就够了。它只数连接数量，速度快，但不关心“谁给你连过来”。如果你关心“被高价值节点连接”这件事，再考虑 PageRank 或 Eigenvector。

如果你想要确定性更强的群组结果，可以先用 WCC。WCC，弱连通分量，就是在忽略方向后看哪些节点至少能互相到达。它不会优化模块度，因此更稳定，但表达能力也更弱，适合先粗切块。GraphAcademy 的欺诈课程里就采用了 Degree + WCC 的流程，先编码领域规则，再用简单算法找连通群。

如果 Louvain 的结果抖动太大，可以考虑：

| 算法 | 输出特点 | 优点 | 局限 |
|---|---|---|---|
| WCC | 连通块 | 稳定、快、易解释 | 只能回答“是否连通” |
| Label Propagation | 社区标签 | 快，适合大图探索 | 结果可能不稳定 |
| Louvain | 层级社区 | 常用、效果好、支持中间社区 | 有分辨率极限与退化问题 |
| Leiden | 改进型层级社区 | 通常比 Louvain 更稳 | 需要理解更多参数 |
| Modularity Metric | 划分评估指标 | 适合评估已有分群 | 不是直接分群算法 |
| HDBSCAN | 密度聚类 | 适合嵌入空间聚类 | 往往配合向量而非原始图结构 |

所以选型原则可以压缩成三句：

1. 先问业务问题是排名、分团，还是链路。
2. 再问图是有向、加权，还是只是连通。
3. 最后问你要探索性结果，还是更稳定、可复现的结果。

---

## 参考资料

- Neo4j Graph Data Science PageRank 文档：<https://neo4j.com/docs/graph-data-science/current/algorithms/page-rank/>
- Neo4j Graph Data Science Louvain 文档：<https://neo4j.com/docs/graph-data-science/current/algorithms/louvain/>
- Neo4j Graph Data Science Community Detection 总览：<https://neo4j.com/docs/graph-data-science/current/algorithms/community/>
- Neo4j Graph Data Science Pathfinding 总览：<https://neo4j.com/docs/graph-data-science/current/algorithms/pathfinding/>
- GraphAcademy Fraud Detection：<https://graphacademy.neo4j.com/courses/workshop-gds/2-community-detection-fraud/1-fraud-problem/>
- GraphAcademy Louvain Deep Dive：<https://graphacademy.neo4j.com/courses/workshop-gds/2-community-detection-fraud/2-louvain-deep-dive/>
- GraphAcademy Building Fraud Communities：<https://graphacademy.neo4j.com/courses/workshop-gds/2-community-detection-fraud/5-finding-fraud/>
- Louvain 模块度讲解：<https://www.puppygraph.com/blog/louvain>
