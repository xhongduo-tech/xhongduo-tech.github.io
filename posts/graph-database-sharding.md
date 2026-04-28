## 核心结论

图数据库分片，是把顶点和边分配到多个机器上，在“尽量少跨机器遍历”和“尽量平均分摊负载”之间做折中。

这里先定义两个基本对象。顶点（vertex）就是图里的点，可以理解为“一个实体”；边（edge）就是点和点之间的连接，可以理解为“实体关系”。分片（shard）就是把一份大图拆到多台机器上的一个局部子集。

关系型数据库常按主键做水平切分，因为很多查询可以先定位一行，再做局部过滤。但图查询经常不是“查一行”，而是“从一个点继续往外走”。如果每走一步都要跨机器，请求就会从一次本地内存访问，变成多次网络通信，延迟和消息量都会放大。

因此，图数据库分片没有一个放之四海而皆准的最优答案。常见策略是 `edge-cut`、`vertex-cut`、社区划分和业务域划分。它们分别在三件事上取不同平衡：

| 策略类型 | 适用图结构 | 优点 | 缺点 | 典型场景 |
| --- | --- | --- | --- | --- |
| `edge-cut` | 社区明显、局部连接强 | 顶点通常不复制，实现直观 | 跨片边可能多，遍历易远程跳转 | 社交圈层、局部强关联图 |
| `vertex-cut` | 幂律分布明显、超级节点多 | 可摊薄热点边负载 | 顶点复制增多，一致性更复杂 | 大规模关系图、自然图 |
| 社区划分 | 模块结构稳定 | 分片内局部性好 | 计算分区本身成本高，动态图易失效 | 静态分析图、推荐图 |
| 业务域划分 | 访问边界清楚 | 工程语义强，治理简单 | 公共节点容易变热点 | 多租户、风控、知识域图 |

一个新手容易忽略的点是：图数据库分片不是“把数据平均切开”这么简单，而是“让高频访问路径尽量少跨片”。如果高频查询总是跨片，那么本来一次局部遍历就会被放大成多机消息风暴。

---

## 问题定义与边界

要讨论分片，先把问题对象和指标说清楚。

顶点分片函数记作 $s(v)$，表示顶点 $v$ 属于哪个分片。顶点副本集合记作 $R(v)$，表示顶点 $v$ 被复制到了哪些分片。机器 $m$ 上承载的边集合记作 $E(m)$。

最常用的三个指标如下：

$$
E_{cut}=|\{(u,v)\in E \mid s(u)\neq s(v)\}|
$$

它表示跨分片边数量。直白地说，就是“需要跨机器才能连起来的边有多少”。

$$
RF=\frac{1}{|V|}\sum_{v\in V}|R(v)|
$$

它表示复制因子（replication factor），即“平均每个顶点被保存了几份副本”。

$$
LB=\frac{\max_m |E(m)|}{|E|/|M|}
$$

它表示负载均衡因子。若 $LB=1$，说明所有机器上的边数量完全均衡；越偏离 $1$，说明越倾斜。

这三个指标常常互相冲突。你压低 $E_{cut}$，可能要提高 $RF$；你强行追求 $LB$ 接近 $1$，可能会打碎天然社区，导致遍历跨片增加。

下面这张表可以帮助区分“分片问题本身”和“容易混淆但不是同一层的问题”：

| 目标 | 不属于分片问题的内容 | 容易混淆的概念 |
| --- | --- | --- |
| 减少跨片遍历 | 单机图算法优化 | 图压缩不等于图分片 |
| 保持负载均衡 | 单点缓存命中率调优 | 缓存副本不等于持久副本 |
| 控制复制开销 | 业务建模是否正确 | ER 建模和图分片不是一层 |
| 降低热点冲击 | 网卡/磁盘参数调优 | 系统调优不能替代错误分片 |

问题边界也要明确。图数据库分片面对的是“图遍历、图计算、图存储”三者耦合的问题，而不是简单的数据切表问题。尤其在知识图谱、风控图、供应链图中，查询往往表现为从一个实体出发，沿若干关系连续扩展。这意味着“切得均匀”不一定“查得更快”。

玩具例子可以说明这个边界。假设一个用户节点连接设备、手机号、银行卡、商户、黑名单。如果你只是把节点数量平均分到四台机器上，看起来很均衡；但若一次风控查询经常要在这几个关系间连续跳转，跨片次数仍然可能很多，实际延迟不会好。

---

## 核心机制与推导

`edge-cut` 的核心思想是：尽量让顶点只归属一个分片，边允许被切开。白话说，就是“节点少复制，但可能多走远路”。它适合社区结构明显的图，因为同一社区内部边多，跨社区边少，只要切得对，大多数遍历仍可在本地完成。

`vertex-cut` 的核心思想是：尽量把边均匀分散到各机器，允许高连接度顶点被复制。白话说，就是“边就近放，热点点多备几份”。它适合幂律分布图。幂律分布的意思是，少数点度数极高，大多数点度数很低，例如明星账号、热门商户、全局标签节点。

先看一个最小玩具例子。图中 $A$ 连 $B,C,D,E$ 四条边。

1. `edge-cut`
   把顶点分成两片：$\{A,B,C\}$ 和 $\{D,E\}$  
   那么边 $A-D$、$A-E$ 会跨片，所以 $E_{cut}=2$。

2. `vertex-cut`
   把边分成两片：片 1 存 $A-B,A-C$，片 2 存 $A-D,A-E$  
   这时边都落在本地分片里，但顶点 $A$ 被复制到两片。  
   共有顶点 $A,B,C,D,E$ 五个，其中 $A$ 两份，其余各一份，所以：
   $$
   RF=\frac{2+1+1+1+1}{5}=1.2
   $$

这个例子说明了本质：`vertex-cut` 往往能降低跨片边，但会增加副本和同步成本；`edge-cut` 更省副本，但可能带来更多远程访问。

为什么 $E_{cut}$ 越小越好？因为一次长度为 $k$ 的局部遍历，如果其中有 $r$ 次跨片，整体延迟可粗略写成：

$$
T \approx k \cdot t_{local} + r \cdot t_{remote}
$$

其中 $t_{remote} \gg t_{local}$。这说明远程跳转次数往往主导总耗时。

为什么 $RF$ 不能无限增大？因为副本不是免费。若每次写入要同步到所有副本，写放大可以近似看成：

$$
W \approx W_0 \cdot RF
$$

这里 $W_0$ 是单副本写入成本。`RF` 越大，写路径越长，一致性处理越复杂，恢复成本也更高。

为什么 $LB$ 不能明显偏离 $1$？因为集群吞吐通常由最忙的机器决定。即使平均负载不高，只要某一台机器明显更忙，整体吞吐就会被它卡住。所以实践里常希望 $LB$ 接近 $1$，但不会为了极限均衡去无脑打碎局部性。

下面给出一个机制对比表：

| 维度 | `edge-cut` | `vertex-cut` | 社区划分 | 业务域划分 |
| --- | --- | --- | --- | --- |
| 主要切分对象 | 顶点 | 边 | 社区子图 | 业务边界 |
| 跨片控制方式 | 减少跨社区边 | 复制热点顶点 | 提高社区内聚 | 把常共访实体放一起 |
| 复制压力 | 低 | 中到高 | 视实现而定 | 对公共实体较高 |
| 热点处理 | 弱 | 强 | 中 | 需手工增强 |
| 动态图适应性 | 中 | 较强 | 较弱 | 取决于业务稳定性 |

从查询流程看，可以把遍历理解成三步：

1. 在当前分片找到起点顶点的邻接边。
2. 若目标邻居仍在本片，继续本地扩展。
3. 若边或顶点只在远端存在，则发起远程请求，或者命中该顶点副本后继续扩展。

所以分片策略真正优化的是“第 3 步出现的频率，以及它出现时的代价”。

---

## 代码实现

代码层面最重要的不是造一个完整分布式图引擎，而是把分片指标和分片规则算清楚。下面用一个最小可运行的 Python 例子展示 `edge-cut`、`vertex-cut` 及指标统计。

```python
from collections import defaultdict

edges = [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E")]

def all_vertices(edges):
    vs = set()
    for u, v in edges:
        vs.add(u)
        vs.add(v)
    return vs

def metrics_edge_cut(edges, vertex_partition, machine_count):
    e_cut = 0
    edge_buckets = defaultdict(list)
    replica_sets = defaultdict(set)

    for u, v in edges:
        pu = vertex_partition[u]
        pv = vertex_partition[v]
        if pu != pv:
            e_cut += 1
        edge_buckets[pu].append((u, v))
        replica_sets[u].add(pu)
        replica_sets[v].add(pv)

    vertices = all_vertices(edges)
    rf = sum(len(replica_sets[v]) for v in vertices) / len(vertices)
    avg_edges = len(edges) / machine_count
    lb = max(len(edge_buckets[m]) for m in range(machine_count)) / avg_edges
    return e_cut, rf, lb

def metrics_vertex_cut(edges, edge_partition, machine_count):
    e_cut = 0  # 边按所在分片本地保存，这里把跨片边视作 0
    edge_buckets = defaultdict(list)
    replica_sets = defaultdict(set)

    for edge, m in edge_partition.items():
        u, v = edge
        edge_buckets[m].append(edge)
        replica_sets[u].add(m)
        replica_sets[v].add(m)

    vertices = all_vertices(edges)
    rf = sum(len(replica_sets[v]) for v in vertices) / len(vertices)
    avg_edges = len(edges) / machine_count
    lb = max(len(edge_buckets[m]) for m in range(machine_count)) / avg_edges
    return e_cut, rf, lb

vertex_partition = {"A": 0, "B": 0, "C": 0, "D": 1, "E": 1}
edge_cut_result = metrics_edge_cut(edges, vertex_partition, 2)

edge_partition = {
    ("A", "B"): 0,
    ("A", "C"): 0,
    ("A", "D"): 1,
    ("A", "E"): 1,
}
vertex_cut_result = metrics_vertex_cut(edges, edge_partition, 2)

assert edge_cut_result[0] == 2
assert round(edge_cut_result[1], 2) == 1.0
assert round(edge_cut_result[2], 2) == 1.5

assert vertex_cut_result[0] == 0
assert round(vertex_cut_result[1], 2) == 1.2
assert round(vertex_cut_result[2], 2) == 1.0

print("edge-cut:", edge_cut_result)
print("vertex-cut:", vertex_cut_result)
```

这个例子刻意做得很小，目的是把指标变化看清楚，而不是模拟完整系统。输出结果会体现两件事：`edge-cut` 跨片更多但副本更少，`vertex-cut` 更均衡但副本更高。

如果把它抽象成工程流程，最小步骤一般是：

| 输入 | 分片规则 | 输出结果 | 适用说明 |
| --- | --- | --- | --- |
| 边列表 | 顶点哈希/社区分配 | 顶点归属、跨片边统计 | 适合 `edge-cut` |
| 边列表 | 边哈希/二维边分区 | 边归属、顶点副本统计 | 适合 `vertex-cut` |
| 查询起点 | 本地扩展 + 远程跳转 | 访问路径、远程请求数 | 用于评估真实查询成本 |

查询时的跨片处理逻辑可以写成下面这样的伪代码思路：

```python
def expand(start_vertex, max_hops):
    frontier = [(start_vertex, 0)]
    visited = {start_vertex}

    while frontier:
        v, depth = frontier.pop(0)
        if depth == max_hops:
            continue

        for neighbor in local_or_remote_neighbors(v):
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append((neighbor, depth + 1))

    return visited
```

这里 `local_or_remote_neighbors(v)` 的实现取决于你的分片策略。若当前分片有顶点副本和邻接索引，查询可以本地命中；若没有，就需要远程拉取。真正的系统优化，往往不在遍历代码本身，而在“让这个函数尽量少发远程请求”。

真实工程例子是多租户风控图：`账号-设备-手机号-商户-黑名单`。如果多数查询都围绕单租户账户展开，那么按租户或业务域分片通常比纯哈希好，因为它更符合访问路径。但共享设备指纹、商户和黑名单又会变成跨租户公共节点，这时就要额外做热点副本，否则热点片会被打满。

---

## 工程权衡与常见坑

工程里最常见的错误不是“不会分片”，而是“只优化一个指标”。

第一类坑是只追求少跨片边。这样做常见结果是把很多强关联数据硬塞到同一分片，短期看 $E_{cut}$ 下降了，长期却可能把超级节点压成热点。超级节点（supernode）就是连接数极高的节点，例如一个全局黑名单实体、一个热门商户、一个知识图谱中的通用概念节点。

第二类坑是只追求均衡。把边或点均匀打散后，$LB$ 好看了，但查询路径被切碎，高频遍历会出现大量远程请求。系统表面“平均很均匀”，实际“每次查询都很贵”。

下面给出常见风险表：

| 常见坑 | 现象 | 成因 | 规避方法 |
| --- | --- | --- | --- |
| 超级节点热点 | 单片 CPU/网络飙高 | 公共节点集中落单片 | 对高阶节点做选择性复制 |
| 社区被打碎 | 查询延迟抖动大 | 只按哈希均匀切分 | 先分析访问路径和社区结构 |
| 副本一致性成本高 | 写入吞吐下降 | `RF` 过高 | 只复制热点，限制副本上界 |
| 分片倾斜 | 某些机器长期过载 | 图结构天然不均 | 定期重平衡，按边负载而非点数统计 |
| 动态图失配 | 初始分片效果快速变差 | 新边不断改写社区结构 | 做在线迁移或周期重分区 |

这里再回顾三个核心指标的工程含义：

- $E_{cut}$ 高，通常意味着读路径成本高。
- $RF$ 高，通常意味着写路径和一致性成本高。
- $LB$ 高于合理阈值，通常意味着吞吐被最忙机器限制。

真实工程里，一个可执行的决策清单通常比抽象原则更有用：

| 检查问题 | 若答案是“是” | 倾向策略 |
| --- | --- | --- |
| 是否有明显社区结构 | 社区内访问远多于社区间 | 社区划分或 `edge-cut` |
| 是否是幂律度分布 | 少数点连接极多 | `vertex-cut` |
| 是否多租户隔离明显 | 大部分查询在租户内闭合 | 业务域/租户分片 |
| 是否存在全局公共节点 | 热点节点被大量共享访问 | 选择性复制 |
| 是否能接受较高一致性成本 | 写少读多 | 可提高副本换取读局部性 |

一个务实结论是：分片设计必须绑定“真实访问模式”来看，而不是只看静态图结构。静态结构告诉你图长什么样，访问模式告诉你系统到底会怎么被打。

---

## 替代方案与适用边界

没有一种策略适合所有图，更准确的说法是：应当按“图结构 + 查询模式 + 写入特征 + 运维约束”联合选择。

如果图的社区结构稳定，而且查询多在社区内闭合，社区划分往往是优先项。社交图、局部推荐图常属于这种情况。若图的度分布极不均匀，少量节点连接海量边，那么 `vertex-cut` 更常见，因为它能把热点边负载拆开。

如果是知识图谱或风控图，常常不能只靠一种纯策略。比如多租户风控图，租户内实体天然应放一起，但全局黑名单、共享设备、共享商户又跨租户访问频繁，这时就更适合“按租户主分片 + 公共热点副本”的混合方案。`silo / pool / hybrid` 的思路本质上也是这个逻辑：有些数据要隔离，有些数据要共享，有些则要按访问热点做折中。

下面是一个对比表：

| 方案 | 优势 | 弱点 | 适用边界 |
| --- | --- | --- | --- |
| 社区划分 | 查询局部性强 | 动态重分区成本高 | 稳定社区图 |
| 业务域划分 | 贴近业务语义 | 公共节点容易成瓶颈 | 多租户、知识域 |
| `edge-cut` | 实现和一致性较简单 | 跨片边可能多 | 社区明显图 |
| `vertex-cut` | 对超级节点更友好 | 副本与同步更复杂 | 幂律图 |
| 混合方案 | 更贴近真实场景 | 设计和运维复杂 | 中大型生产系统 |

再从适用边界角度压缩成一张表：

| 图类型 | 查询模式 | 写入特征 | 热点节点 | 推荐策略 |
| --- | --- | --- | --- | --- |
| 社交关系图 | 多跳局部遍历 | 中等 | 有时明显 | 社区划分 / `edge-cut` |
| 自然关系图 | 边扩展密集 | 高 | 明显 | `vertex-cut` |
| 多租户风控图 | 单租户内遍历为主 | 高 | 明显 | 业务域 + 热点复制 |
| 知识图谱 | 主题子图查询 + 公共概念复用 | 中高 | 常见 | 混合方案 |

如果一定要给一个经验公式，可以写成：

$$
\text{总成本} \approx \alpha \cdot E_{cut} + \beta \cdot RF + \gamma \cdot (LB-1)
$$

这里的 $\alpha,\beta,\gamma$ 不是固定常数，而是由业务决定的权重。读多写少的系统，$\alpha$ 往往更大；写入频繁且一致性敏感的系统，$\beta$ 往往更大；资源成本敏感或 SLA 严格的系统，$\gamma$ 不能忽略。

所以最终结论不是“选哪种策略最好”，而是“你的系统最怕什么代价，然后围绕它做有约束的折中”。

---

## 参考资料

| 标题 | 主要贡献 | 对应正文章节 | 可借鉴点 |
| --- | --- | --- | --- |
| PowerGraph | 解释幂律图下 `vertex-cut` 动机 | 核心机制与推导 | 为什么热点顶点应允许复制 |
| Spark GraphX PartitionStrategy | 展示常见分区策略工程实现 | 代码实现 | 边分区与副本上界思路 |
| Graph Partitioning for Distributed Graph Processing | 总结图分区指标与方法 | 问题定义与边界 | `RF`、`LB` 等指标定义 |
| AWS SaaS data partitioning models | 说明按租户边界分区的工程依据 | 工程权衡与适用边界 | 多租户 `silo/pool/hybrid` 设计 |

1. [PowerGraph: Distributed Graph-Parallel Computation on Natural Graphs](https://www.usenix.org/conference/osdi12/technical-sessions/presentation/gonzalez)
2. [Apache Spark GraphX PartitionStrategy](https://spark.apache.org/docs/latest/api/java/org/apache/spark/graphx/PartitionStrategy.html)
3. [Graph Partitioning for Distributed Graph Processing](https://link.springer.com/article/10.1007/s41019-017-0034-4)
4. [AWS SaaS data-partitioning models](https://docs.aws.amazon.com/prescriptive-guidance/latest/multi-tenancy-amazon-neptune/data-partitioning-models.html)
