## 核心结论

图论研究的不是“单个对象”，而是“对象之间如何连接”。形式化地说，图是一个二元组：

$$
G=(V,E)
$$

其中：

- $V$ 是顶点集合（vertex set），表示对象本身
- $E$ 是边集合（edge set），表示对象之间的关系

如果把顶点理解为“节点”，把边理解为“连接”，大多数图问题就会变得直观很多。社交网络、地图导航、网页链接、任务依赖、网络路由，本质上都可以抽象成图。

图算法并不是一堆互不相关的技巧，而是围绕三类核心目标展开：

| 问题类型 | 典型算法 | 解决什么问题 | 关键前提 |
|---|---|---|---|
| 遍历 | BFS、DFS | 把图走一遍，找可达点、连通块、层级关系、是否有环 | 一般不依赖权重 |
| 最短路径 | BFS、Dijkstra、Bellman-Ford | 找两点之间总代价最小的路径 | 是否有权、是否允许负权决定算法 |
| 最小生成树 | Prim、Kruskal | 用最小总代价把所有点连起来 | 主要针对无向连通带权图 |

最重要的工程结论是：**图的表示方式会直接决定算法性能**。

- 邻接矩阵适合稠密图，也就是边很多、接近“任意两点都可能相连”的图
- 邻接表适合稀疏图，也就是边远少于 $|V|^2$ 的图

设图有 $|V|$ 个点、$|E|$ 条边，则两种表示的空间复杂度分别是：

$$
\text{邻接矩阵} = O(|V|^2), \qquad \text{邻接表} = O(|V|+|E|)
$$

这不是实现细节，而是建模阶段就要做出的结构选择。

看一个最小玩具例子。设有无权社交关系图：

- A 与 B、C 相连
- C 与 D 相连

如果要找 A 到 D 的最短“跳数”，可以直接使用 BFS。原因不是“BFS 常用”，而是因为在无权图中，每条边代价都相同，按层扩展时，**第一次到达目标点的路径一定是最短路径**。

这个结论可以写成更明确的形式：

$$
\text{无权图最短路} \Longrightarrow \text{按边数最少} \Longrightarrow \text{BFS 分层最优}
$$

因此，学习图论时最该先掌握的不是某个算法模板，而是三件事：

1. 图怎么表示
2. 边具有什么性质
3. 当前问题属于“遍历”“最短路”还是“最小生成树”

这三件事一旦判断错，后面的算法往往会整体选错。

---

## 问题定义与边界

图算法的第一层边界，是“边有没有方向”。

- 无向图：边没有方向，$u-v$ 与 $v-u$ 是同一条边。适合表示双向关系，比如“是否互为好友”“两地有公路直接连通”。
- 有向图：边有方向，$u \to v$ 与 $v \to u$ 不等价。适合表示单向关系，比如“网页跳转”“用户关注”“课程先修依赖”。

这一区别会影响存储方式，也会影响问题定义。比如在有向图中，“从 A 能到 B”不代表“从 B 也能到 A”。

第二层边界，是“边有没有权重”。权重可以理解为通过一条边需要付出的代价，也可以理解为关系的强度。它常见的含义包括：

| 场景 | 边权含义 |
|---|---|
| 地图导航 | 距离、时间、通行费 |
| 网络路由 | 时延、带宽成本 |
| 任务调度 | 执行耗时、资源消耗 |
| 金融图模型 | 收益、成本、汇率变化 |

因此图可以分成两类：

- 无权图：可以把每条边都视为代价 1
- 带权图：每条边附带一个数值

第三层边界，是“图是否允许负权”。

- 非负权图：所有边权都满足 $w(u,v)\ge 0$
- 含负权图：某些边满足 $w(u,v)<0$

负权边并不常见，但一旦出现，就会直接改变算法选择。最典型的影响是：**Dijkstra 不能用于含负权边的图**。这不是实现限制，而是正确性前提被破坏。

下面用一个 4 个点、3 条边的最小图说明表示方式。设顶点集合为：

$$
V=\{0,1,2,3\}
$$

边集合为：

$$
E=\{\{0,1\},\{0,2\},\{1,2\}\}
$$

这是一个无向无权图，点 3 是孤立点。

邻接矩阵定义为：

$$
A[i][j] =
\begin{cases}
1, & \text{如果 } i \text{ 与 } j \text{ 相连} \\
0, & \text{否则}
\end{cases}
$$

对应矩阵为：

|   | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| 0 | 0 | 1 | 1 | 0 |
| 1 | 1 | 0 | 1 | 0 |
| 2 | 1 | 1 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 |

这个矩阵的读法很简单：

- 第 0 行的 `1,1` 表示点 0 连到点 1 和点 2
- 第 3 行全是 0，表示点 3 没有任何邻居
- 无向图的矩阵关于主对角线对称

同一张图也可以写成邻接表：

- 0: [1, 2]
- 1: [0, 2]
- 2: [0, 1]
- 3: []

两种表示都正确，但它们擅长的操作不同：

| 表示方式 | 空间复杂度 | 判断两点是否直接相连 | 遍历某点所有邻居 | 适用场景 |
|---|---|---|---|---|
| 邻接矩阵 | $O(|V|^2)$ | $O(1)$ | $O(|V|)$ | 稠密图、小规模图、频繁查边 |
| 邻接表 | $O(|V|+|E|)$ | 取决于实现，通常不如矩阵直接 | $O(\deg(v))$ | 稀疏图、大规模图、遍历型算法 |

这里的 $\deg(v)$ 表示顶点 $v$ 的度，也就是“与 $v$ 相连的边数”。在有向图中还会区分：

$$
\deg^-(v)=\text{入度}, \qquad \deg^+(v)=\text{出度}
$$

也就是说，真正进入算法之前，至少要先把下面三件事说清楚：

1. 有向还是无向
2. 无权、带权，还是允许负权
3. 用什么数据结构存图

很多“算法写错”的根源，其实不是代码能力问题，而是这些建模边界一开始就没定义清楚。

---

## 核心机制与推导

### 1. BFS 与 DFS：先解决“怎么走”

BFS（Breadth-First Search，广度优先搜索）和 DFS（Depth-First Search，深度优先搜索）是图算法最基础的两种遍历方式。

它们解决的第一个问题不是“最短路”或“最优性”，而是：**怎样系统地走完整张图，而不漏点、不重复失控**。

两者的核心区别在于“下一步先处理谁”：

| 算法 | 数据结构 | 走法特征 | 直观理解 |
|---|---|---|---|
| BFS | 队列 | 一层一层向外扩展 | 先近后远 |
| DFS | 栈或递归 | 沿一条路先走到底再回退 | 先深后广 |

如果图用邻接表表示，两者时间复杂度都为：

$$
O(|V|+|E|)
$$

原因并不复杂：

- 每个点最多被首次访问一次
- 每条边最多被检查常数次
- 因此总工作量与点数和边数线性相关

无向图里，一条边通常会在两个端点的邻接表中各出现一次；有向图里，一条边只记录在起点的邻接表中。但复杂度量级仍然一样。

看一个无权图最短路径的例子：

- A 连 B、C
- B 连 E
- C 连 D
- D 连 E

从 A 找到 E 的最短跳数，BFS 的分层过程是：

| 层数 | 本层节点 | 含义 |
|---|---|---|
| 0 | A | 距起点 0 条边 |
| 1 | B, C | 距起点 1 条边 |
| 2 | E, D | 距起点 2 条边 |

第一次遇到 E 时，路径长度就是 2。为什么这个结论一定成立？因为 BFS 的队列保证了“先处理距离近的点，再处理距离远的点”。如果一个点第一次在第 $k$ 层被访问到，就说明已经找到了一条用了 $k$ 条边的路径，而且不可能再存在更短路径。

形式化地说，在无权图中：

$$
dist(v)=\text{从源点到 }v\text{ 的最少边数}
$$

而 BFS 恰好按这个值从小到大访问顶点。

DFS 不擅长直接做无权最短路，但它在很多结构性问题里很重要，例如：

- 判断图中是否有环
- 找连通块
- 做拓扑排序的基础遍历
- 做回溯搜索
- 处理树和图的递归性质

对新手来说，一个常见误区是把“BFS 更短、DFS 更深”当成口号记忆。更准确的理解应当是：

- BFS 的本质是“按距离层级展开”
- DFS 的本质是“沿路径深入探索结构”

它们都在遍历图，但解决的重点不同。

### 2. Dijkstra：非负权图的最短路径

当图不再是无权图，而是“每条边都有不同代价”时，BFS 就不够用了。因为“边数最少”不再等于“总代价最小”。

例如：

- 0 → 1，权重 2
- 0 → 2，权重 5
- 1 → 2，权重 1

从 0 到 2：

- 直接走 `0 -> 2`，总代价是 5
- 走 `0 -> 1 -> 2`，总代价是 3

虽然第二条路径边更多，但代价更小。因此最短路径问题要比较的是：

$$
\text{路径总代价} = \sum w(u,v)
$$

而不是边数。

Dijkstra 的核心机制是“贪心 + 松弛”。

先解释“松弛”（relaxation）。它的含义是：如果经过当前点 $u$ 去到邻居 $v$，能得到一条更短路径，就更新 $v$ 的当前最优距离：

$$
dist[v] = \min(dist[v], dist[u] + w(u,v))
$$

其中：

- $dist[x]$ 表示从源点到顶点 $x$ 的当前已知最短距离
- $w(u,v)$ 表示边 $(u,v)$ 的权重

Dijkstra 的做法是：

1. 初始时，源点距离设为 0，其余点设为无穷大
2. 每次从“尚未最终确定的点”中，选出当前距离最小的那个点
3. 用这个点去松弛它的邻边
4. 重复直到所有可达点都被处理完

它的正确性依赖一个关键前提：**所有边权非负**。

为什么非负权这么关键？因为只有在边权都不小于 0 时，“当前最小的那个点”才不可能被未来路径再改得更小。否则，如果后面出现一条负权边，就可能把已经“看似最优”的结果继续拉低，贪心判断就失效。

还是看上面的例子：

- 初始：$dist[0]=0,\ dist[1]=\infty,\ dist[2]=\infty$
- 处理 0 后：$dist[1]=2,\ dist[2]=5$
- 当前最小的是 1，于是处理 1
- 松弛边 $(1,2)$ 后：

$$
dist[2] = \min(5, 2+1)=3
$$

因此最短路是：

$$
0 \to 1 \to 2
$$

总成本为 3。

使用最小堆优化时，Dijkstra 的常见复杂度为：

$$
O((|V|+|E|)\log |V|)
$$

这个复杂度对大多数工程场景已经足够实用，因此它是最常见的单源最短路径算法之一。

### 3. Bellman-Ford：允许负权，但代价更高

如果图中允许负权边，Dijkstra 的前提就不成立了，这时常见替代方案是 Bellman-Ford。

它与 Dijkstra 的差别，不是“更新公式不同”，而是“是否相信局部贪心已经最终正确”。

Bellman-Ford 不做“当前最小点已确定”的假设，而是采用更稳妥、也更慢的策略：**反复扫描所有边，持续做松弛**。

它的核心过程可以写成：

$$
\text{重复 } |V|-1 \text{ 轮：对所有边 } (u,v)\text{ 执行松弛}
$$

为什么是 $|V|-1$ 轮？因为一条简单路径最多经过 $|V|-1$ 条边。每做一轮松弛，可以理解为“允许最短路径再多使用一条边”。因此当轮数达到 $|V|-1$ 时，所有不含环的最短路径都已经被覆盖。

如果在第 $|V|$ 轮仍然可以继续松弛，就说明图中存在负权环。

负权环的含义不是“有一条边是负的”，而是：

$$
w(C)<0
$$

其中 $C$ 是某个环。也就是说，沿这个环绕一圈，总代价反而变小。这样一来，你可以无限重复绕圈，使路径代价不断下降，因此“最短路径”不再有定义。

Bellman-Ford 的复杂度为：

$$
O(|V|\cdot |E|)
$$

和 Dijkstra 相比，它明显更慢，但适用边界更宽。实际使用时可以这样理解：

- 如果边权都非负，优先用 Dijkstra
- 如果可能有负权，或需要检测负权环，用 Bellman-Ford

这里要特别区分两个概念：

- “有负权边”不等于“有负权环”
- “有负权环”才意味着最短路径可能失去定义

### 4. Prim 与 Kruskal：不是最短路，而是最小生成树

最小生成树（Minimum Spanning Tree, MST）解决的不是“从起点到终点最短”，而是另一类问题：

> 在无向连通带权图中，选出若干条边，把所有顶点连起来，要求总代价最小，而且不能形成环。

这一定义里有三个关键条件：

1. 要覆盖所有点
2. 要保持连通
3. 不能有环

生成树一共会有多少条边？如果图有 $|V|$ 个点，那么任何生成树都恰好有：

$$
|V|-1
$$

条边。少了不连通，多了就一定成环。

Prim 和 Kruskal 都能求 MST，但思路不同。

Prim 的思路是“从一个点出发，逐步向外扩展”：

- 初始时选择任意一个起点
- 每次挑选一条最小边，把一个新点接入当前连通块
- 重复直到所有点都被纳入

因此它更像“长树枝”：始终维护一个不断扩大的已选点集合。

Kruskal 的思路是“先看全局最小边，再判断能不能加”：

- 先把所有边按权重从小到大排序
- 依次考虑每条边
- 如果加上这条边不会成环，就加入答案
- 否则跳过

因此它更像“在全局边集里做筛选”。

Kruskal 为什么一定要判环？因为生成树必须无环。如果一条候选边的两个端点已经在同一个连通块中，再加它就会形成环。

这正是并查集（Union-Find）的用途。并查集维护“哪些点当前属于同一个连通块”。它支持两个基本操作：

- `find(x)`：查找点 $x$ 所在集合的代表元
- `union(a,b)`：把两个集合合并

如果 `find(a) == find(b)`，说明 $a$ 和 $b$ 已经连通，再加边 $(a,b)$ 就会成环。

常见复杂度如下：

| 算法 | 主要用途 | 常见实现复杂度 | 使用条件 |
|---|---|---|---|
| BFS | 无权最短路、层次遍历 | $O(|V|+|E|)$ | 无权图或等权图 |
| DFS | 遍历、环检测、拓扑相关基础 | $O(|V|+|E|)$ | 通用 |
| Dijkstra | 非负权最短路 | $O((|V|+|E|)\log |V|)$ | 不能有负权 |
| Bellman-Ford | 含负权最短路 | $O(|V|\cdot |E|)$ | 可检测负权环 |
| Prim | 最小生成树 | $O((|V|+|E|)\log |V|)$ | 无向连通带权图 |
| Kruskal | 最小生成树 | $O(|E|\log |E|)$ | 需要边排序与判环 |

这里最容易混淆的一点是：**最短路径和最小生成树完全不是同一个问题**。

- 最短路径关心的是“某两点之间最便宜”
- 最小生成树关心的是“整张图全部连通时总成本最小”

MST 上某两点之间的路径，未必是原图中的最短路径；反过来，把所有点到源点的最短路径拼在一起，也未必能得到总权重最小的生成树。

---

## 代码实现

初学阶段最稳妥的写法，是把“图的表示”和“图上的算法”拆开。

- 表示层：图如何存储，通常优先使用邻接表
- 算法层：BFS、Dijkstra、Bellman-Ford、Kruskal 等在这个表示上运行

下面给出一个完整、可直接运行的 Python 示例，包含：

- 无权图邻接表构造
- BFS 最短路径
- 带权图邻接表构造
- Dijkstra
- Bellman-Ford
- 并查集
- Kruskal 最小生成树

```python
from collections import deque
import heapq


def build_unweighted_graph(n, edges, directed=False):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        if not directed:
            graph[v].append(u)
    return graph


def bfs_shortest_path(graph, start, target):
    queue = deque([start])
    parent = [-1] * len(graph)
    parent[start] = start

    while queue:
        u = queue.popleft()
        if u == target:
            break
        for v in graph[u]:
            if parent[v] == -1:
                parent[v] = u
                queue.append(v)

    if parent[target] == -1:
        return None

    path = []
    cur = target
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    path.reverse()
    return path


def build_weighted_graph(n, edges, directed=False):
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
        if not directed:
            graph[v].append((u, w))
    return graph


def dijkstra(graph, start):
    dist = [float("inf")] * len(graph)
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        cur_dist, u = heapq.heappop(pq)
        if cur_dist != dist[u]:
            continue

        for v, w in graph[u]:
            if w < 0:
                raise ValueError("Dijkstra cannot handle negative weights")
            nd = cur_dist + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def bellman_ford(n, edges, start):
    dist = [float("inf")] * n
    dist[start] = 0

    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != float("inf") and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break

    for u, v, w in edges:
        if dist[u] != float("inf") and dist[u] + w < dist[v]:
            raise ValueError("Negative cycle detected")

    return dist


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def kruskal_mst(n, edges):
    uf = UnionFind(n)
    mst_edges = []
    total_weight = 0

    for u, v, w in sorted(edges, key=lambda item: item[2]):
        if uf.union(u, v):
            mst_edges.append((u, v, w))
            total_weight += w
            if len(mst_edges) == n - 1:
                break

    if len(mst_edges) != n - 1:
        raise ValueError("Graph is not connected, MST does not exist")

    return total_weight, mst_edges


if __name__ == "__main__":
    # 1) 无权图最短路径：0 -> 4
    g1 = build_unweighted_graph(
        5,
        [(0, 1), (0, 2), (2, 3), (3, 4)]
    )
    path = bfs_shortest_path(g1, 0, 4)
    assert path == [0, 2, 3, 4]

    # 2) 非负权最短路：Dijkstra
    g2 = build_weighted_graph(
        3,
        [(0, 1, 2), (0, 2, 5), (1, 2, 1)],
        directed=True
    )
    dist = dijkstra(g2, 0)
    assert dist[2] == 3

    # 3) 含负权但无负权环：Bellman-Ford
    edges_bf = [
        (0, 1, 4),
        (0, 2, 5),
        (1, 2, -2),
    ]
    dist_bf = bellman_ford(3, edges_bf, 0)
    assert dist_bf[2] == 2

    # 4) 最小生成树：Kruskal
    edges_mst = [
        (0, 1, 1),
        (1, 2, 2),
        (0, 2, 4),
        (2, 3, 3),
        (1, 3, 5),
    ]
    total_weight, mst = kruskal_mst(4, edges_mst)
    assert total_weight == 6
    assert len(mst) == 3

    print("all tests passed")
```

这段代码可以直接运行，输出：

```text
all tests passed
```

上面每部分对应的建模目标不同：

| 代码部分 | 解决的问题 | 输入要求 |
|---|---|---|
| `bfs_shortest_path` | 无权图最短路径 | 无权图或等权图 |
| `dijkstra` | 非负权单源最短路 | 所有边权非负 |
| `bellman_ford` | 含负权单源最短路 | 允许负权，可检测负权环 |
| `kruskal_mst` | 最小生成树 | 无向连通带权图 |

如果只看 BFS 的最小骨架，伪代码如下：

```text
BFS(graph, start):
    queue <- [start]
    visited[start] <- true

    while queue 非空:
        u <- queue.pop_front()

        for v in graph[u]:
            if visited[v] == false:
                visited[v] <- true
                queue.push_back(v)
```

这里“入队时立即标记 visited”非常关键。原因是如果等到出队时再标记，同一个点可能被多个前驱重复加入队列，导致大量重复工作。

再看 Dijkstra 的最小骨架：

```text
Dijkstra(graph, start):
    dist[start] <- 0
    其余点 <- inf
    最小堆 pq <- [(0, start)]

    while pq 非空:
        取出当前距离最小的点 u
        如果这个距离不是最新值，跳过

        for (v, w) in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] <- dist[u] + w
                把 (dist[v], v) 压入堆
```

它的关键不是“用堆”本身，而是：

- 每次优先扩展当前代价最小的候选点
- 不断做松弛
- 依赖非负权保证贪心正确

图算法与数据结构的关系可以整理成一张更完整的表：

| 数据结构 | 典型 Python 形式 | 主要支持操作 | 常见对应算法 |
|---|---|---|---|
| 邻接表 | `list[list]` 或 `dict[list]` | 枚举邻居 | BFS、DFS、Dijkstra、Prim |
| 邻接矩阵 | 二维数组 | $O(1)$ 判断边存在 | 稠密图、某些 DP/Floyd 类算法 |
| 队列 | `collections.deque` | 头删尾插 | BFS |
| 栈 | `list` 或递归调用栈 | 后进先出 | DFS |
| 最小堆 | `heapq` | 取最小值 | Dijkstra、Prim |
| 并查集 | `parent/rank` 数组 | 合并集合、查连通块 | Kruskal |

真实工程里，路由网络是典型例子：

- 路由器是顶点
- 物理链路是边
- 时延或成本是边权

如果所有代价都非负，Dijkstra 就很自然；如果模型中出现了“负代价”，通常要么换用 Bellman-Ford，要么反过来检查：这个建模本身是否合理。因为很多工程问题里，负权并不总有物理含义。

---

## 工程权衡与常见坑

图算法在工程里经常不是“不会写”，而是“条件判断错了”。下面先看最常见的坑。

| 常见坑 | 为什么会出错 | 规避手段 |
|---|---|---|
| 稀疏图误用邻接矩阵 | 空间膨胀到 $O(|V|^2)$ | 大图默认优先考虑邻接表 |
| Dijkstra 处理负权边 | 贪心前提失效，结果可能错误 | 先检查权重范围，必要时换 Bellman-Ford |
| DFS 直接递归跑深图 | 调用栈过深可能溢出 | 改为显式栈 |
| Kruskal 不判环 | 会得到带环结构，不是树 | 使用并查集 |
| 无向图只加一条边 | 把无向图误建模成有向图 | 插入边时同时加 `u->v` 和 `v->u` |
| visited 标记太晚 | 队列或栈中出现重复节点 | 入队/入栈时立即标记 |
| 把 MST 当最短路 | 目标函数完全不同 | 先分清“局部最短”还是“全局连通成本最小” |
| 忘记处理非连通图 | 某些点不可达，树也可能不存在 | 显式判断 `None`、`inf` 或 MST 边数是否足够 |

下面用几个工程化场景解释这些坑为什么真实存在。

社交图通常非常稀疏。设有一百万用户，每个用户平均只和 200 人有关系，那么：

$$
|V|=10^6,\qquad |E|\approx 2\times 10^8
$$

这个量级已经很大，但如果使用邻接矩阵，空间规模是：

$$
O(|V|^2)=O(10^{12})
$$

这在内存上几乎不可接受。而邻接表的空间更接近“节点数 + 实际边数”，才能落到可处理范围内。也就是说，很多时候不是算法慢，而是存图方式一开始就错了。

再看 BFS 的典型业务含义：

- “找某人 2 跳内的联系人”
- “找最短好友链”
- “按层扩展推荐候选人”

这些问题本质都是无权图上的层次搜索。只要边权没有差别，BFS 就比 Dijkstra 更直接、更便宜。

Kruskal 里的判环最容易被初学者忽略，因为“按边从小到大选”看上去已经很合理了。但如果不判环，算法就会持续加入小边，最终可能形成一个总代价小但不满足树结构的图。并查集正是为这个步骤服务的。

并查集的最小核心代码如下：

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True
```

这段代码里：

- `find(x)` 的意义是“找点 $x$ 当前属于哪个连通块”
- `union(a,b)` 的意义是“把两个连通块合并起来”
- 如果 `union` 返回 `False`，说明两点早已连通，再加边就会成环

还有一个容易被忽略的边界是“不可达”。例如最短路径问题里，如果源点到目标点根本不存在路径，那么：

- BFS 往往返回 `None`
- Dijkstra / Bellman-Ford 中相应距离仍是 `inf`

这不是异常情况，而是图问题的正常输出之一。工程实现里必须显式处理，不能默认“肯定能到”。

---

## 替代方案与适用边界

图算法选择不是背诵题，而是条件判断题。一个实用的判断顺序通常是：

1. 先问图怎么存
2. 再问边有没有权
3. 再问权重是否可能为负
4. 最后问目标是“最短路”“遍历”还是“最小生成树”

把这个过程压缩成表，会更容易在工程里快速判断：

| 目标 | 图特征 | 优先方案 | 复杂度 | 备注 |
|---|---|---|---|---|
| 遍历 / 连通性 | 通用 | BFS / DFS | $O(|V|+|E|)$ | BFS 有层次，DFS 更适合深搜与回溯 |
| 无权最短路 | 无权图 | BFS | $O(|V|+|E|)$ | 不必使用 Dijkstra |
| 最短路 | 非负权 | Dijkstra | $O((|V|+|E|)\log |V|)$ | 堆优化最常见 |
| 最短路 | 有负权 | Bellman-Ford | $O(|V|\cdot |E|)$ | 可检测负权环 |
| 最小生成树 | 边排序方便、全局边集明确 | Kruskal | $O(|E|\log |E|)$ | 需要并查集 |
| 最小生成树 | 从局部逐步扩展更自然 | Prim | $O((|V|+|E|)\log |V|)$ | 适合邻接表 + 堆 |

可以把选择逻辑再说得更直接一些：

- 如果问题问的是“能不能到”“哪些点可达”“分几层到达”，先想 BFS/DFS
- 如果问题问的是“从 A 到 B 代价最小”，先看有没有权、有没有负权
- 如果问题问的是“把所有点接起来，总代价最低”，那是 MST，不是最短路

关于 Prim 和 Kruskal，没有绝对的“谁更高级”，更像是建模偏好不同。

- Kruskal 更像“先看全局所有边，再挑不会成环的最小边”
- Prim 更像“从一个已有连通块开始，持续往外接最便宜的新边”

因此常见经验是：

| 场景倾向 | 更常见的选择 |
|---|---|
| 边集天然完整、排序方便 | Kruskal |
| 已经是邻接表、从局部扩展更自然 | Prim |
| 图很稠密，常用矩阵表示 | 某些实现下 Prim 更直接 |

还要特别强调几个适用边界，避免概念混淆：

- Bellman-Ford 不是“更高级的 Dijkstra”，只是输入条件更宽，代价也更高
- 最小生成树不能替代最短路径，因为两者优化目标不同
- BFS 只在无权图或等权图里能直接当最短路算法使用
- 邻接矩阵不是“落后表示”，它在稠密图和频繁查边场景下仍然合理
- 算法复杂度不是唯一指标，常数项、内存、可维护性也会影响工程选择

如果只记一个判断框架，可以记下面这句：

$$
\text{先判断图的性质，再判断问题目标，最后再选算法}
$$

顺序不能反。先背算法名字，再回头套场景，通常会错。

---

## 参考资料

1. Cormen, Leiserson, Rivest, Stein，《Introduction to Algorithms》（CLRS），第 20、21、22、23、24 章。适合系统查 BFS、DFS、最小生成树、单源最短路径、并查集的定义、证明与复杂度边界。
2. Sedgewick, Wayne，《Algorithms, 4th Edition》，Graphs 相关章节。适合理解图表示、遍历与最短路径的实现细节，代码风格也更接近工程实践。
3. GeeksforGeeks，《Graph and its Representations》。适合快速查邻接矩阵、邻接表的定义与示例。
4. GeeksforGeeks，《Breadth First Search or BFS for a Graph》。适合复习 BFS 的分层机制、队列写法与复杂度。
5. GeeksforGeeks，《Dijkstra’s Shortest Path Algorithm》。适合查 Dijkstra 的堆优化实现与非负权前提。
6. GeeksforGeeks，《Bellman-Ford Algorithm》。适合补充负权边、负权环检测与多轮松弛的直观解释。
7. GeeksforGeeks，《Kruskal’s Minimum Spanning Tree Algorithm》。适合配合并查集理解“排序 + 判环”的完整流程。
8. CP-Algorithms，Graph Theory 与 Shortest Paths 相关条目。适合查竞赛和工程中常见模板、复杂度、边界条件。
9. Skiena，《The Algorithm Design Manual》，Graph Problems 相关章节。适合把图问题映射到实际工程场景，理解“问题建模先于算法选择”。
