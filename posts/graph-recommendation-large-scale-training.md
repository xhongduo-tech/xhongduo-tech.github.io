## 核心结论

大规模图推荐训练的核心，不是把图神经网络做得更深，而是把“全图训练”改成“可控规模的局部训练”。图神经网络，简称 GNN，是一种让节点从邻居节点接收信息并更新表示的模型。推荐系统里的用户、商品、品牌、类目都可以看成节点，点击、购买、收藏、归属关系都可以看成边。

工业级图推荐常见规模是十亿级节点、百亿级边。直接全图训练会遇到三个瓶颈：邻居爆炸、显存压力、跨机通信。邻居爆炸指的是每层都向外展开邻居，层数稍微增加，参与计算的节点数就快速膨胀。

总成本可以粗略写成：

$$
T_{train} \approx T_{sample} + O(|V_b| + |E_b|) + T_{comm}
$$

其中 $|V_b|$ 是一个 batch 内的节点数，$|E_b|$ 是边数，$T_{comm}$ 是分布式训练中的通信时间。优化目标不是单独压低某一项，而是让采样、计算、通信同时可控。

| 方法 | 主要解决的问题 | 核心动作 |
|---|---|---|
| 邻居采样 GraphSAGE | 邻居爆炸、显存不可控 | 每层最多采样固定数量邻居 |
| Cluster-GCN | batch 内边太稀疏、全图切分低效 | 用图聚类构造稠密子图 batch |
| 分布式分区训练 | 单机放不下、跨机通信过大 | 减少切边和 halo 节点 |

一个最小流程可以概括为：

```text
sample -> build subgraph -> message passing -> communicate if distributed
```

玩具例子：1 个用户节点做 2 层 GNN，如果每层最多采样 4 个邻居，展开规模上界约为 $1 + 4 + 4 \times 4 = 21$。如果不采样，一个高活跃用户可能连接几百个商品，每个商品又连接大量品牌、类目和相似商品，2 层展开后可能达到几千个节点。

真实工程例子：电商推荐图中，一个用户连接点击商品，商品连接品牌、店铺、类目、共购商品。训练“用户是否会点击某商品”时，如果每个 batch 都加载完整两跳子图，GPU 很快会等待 CPU 采样、内存拷贝和跨机特征拉取。工程优化通常先限制 fanout，再按图结构做分区，最后监控通信和 GPU 利用率。

---

## 问题定义与边界

本文讨论的问题是：如何让十亿级节点、百亿级边的推荐图还能训练得动、训得稳、训得快。这里不讨论新的 GNN 模型结构，也不展开特征工程、召回、排序、重排的完整推荐链路。

推荐图通常是异构图。异构图是指图里有多种节点和边，例如用户、商品、品牌、类目是不同节点类型，点击、购买、属于、相似是不同边类型。异构图比普通同构图更容易膨胀，因为一次邻居展开可能跨越多种关系。

| 术语 | 白话解释 |
|---|---|
| seed node | 当前 batch 需要预测或训练的起始节点 |
| fanout | 每层最多采样多少个邻居 |
| subgraph | 从全图中抽出的局部子图 |
| halo node | 不属于本分区、但为了计算被临时拉进来的外部分区节点 |
| edge cut | 被分区切开的跨分区边 |
| batch | 一次训练迭代处理的一小组样本 |

训练复杂度会随层数 $L$ 和每层 fanout $k_l$ 增长。粗略上界是：

$$
N_{sample} \le B \prod_{l=1}^{L} k_l
$$

其中 $B$ 是 seed node 数量，$k_l$ 是第 $l$ 层的采样邻居数。这个式子说明，层数和 fanout 都不能随意放大。

| 覆盖内容 | 不覆盖内容 |
|---|---|
| 邻居采样 | 如何设计新的 GNN 层 |
| Cluster-GCN 子图训练 | CTR 排序模型结构 |
| 图分区与通信 | 特征工程全流程 |
| 分布式训练瓶颈 | 在线推理服务架构 |

新手可以把全图训练理解成每次都读完整本百科全书，采样训练只读和当前问题相关的几页；分区训练则是把百科全书拆成几册，减少来回翻找和借书成本。这个类比只用于理解规模控制，不能替代对节点、边、通信量的精确定义。

---

## 核心机制与推导

邻居采样的本质，是把每层展开的上界固定住。GraphSAGE 的思路是对每个节点采样邻居，再聚合邻居表示。聚合是指把多个邻居节点的向量合成当前节点的新向量，例如取平均、求和或用神经网络加权。

假设 batch 有 $B=128$ 个 seed node，两层 fanout 是 $[15, 10]$，那么采样节点数量的粗略上界是：

$$
N_{sample} \le 128 \times 15 \times 10 = 19200
$$

实际节点数可能更少，因为不同 seed node 会共享邻居，也可能采不到足够邻居。这个上界的价值在于：训练规模从不可预测变成可配置。

子图 batch 的计算成本可以写成：

$$
batch\ cost \approx O(|V_b| + |E_b|)
$$

其中 $|V_b|$ 是子图节点数，$|E_b|$ 是子图边数。GNN 的消息传递通常沿边发生，所以边数直接影响计算量。

Cluster-GCN 的本质，是先把全图聚成多个内部连接更密的簇，再从簇中构造 batch。簇是图中连接更紧密的一组节点。这样做可以减少无关邻居展开，提高 batch 内部边的利用率。

分布式分区训练的重点是减少切边。切边是两端节点被分到不同机器上的边。跨分区边越多，训练时需要从远端机器拉取的特征越多。通信量可以粗略写成：

$$
comm\ bytes \approx C_{cut} \times d \times s
$$

其中 $C_{cut}$ 是跨分区边或跨分区依赖数量，$d$ 是特征维度，$s$ 是每个特征值占用字节数。比如 float32 的 $s=4$。如果一批训练有 1000 条跨分区依赖，特征维度是 128，那么通信量约为 $1000 \times 128 \times 4 = 512000$ 字节。把切边降到 200，通信量也会接近同比下降。

| 机制 | 控制对象 | 适合解决 |
|---|---|---|
| GraphSAGE 邻居采样 | 每层邻居数量 | 节点展开失控 |
| Cluster-GCN | batch 子图结构 | 子图太稀疏、训练效率低 |
| 分布式分区 | 跨分区边和 halo 节点 | 多机通信等待 |

---

## 代码实现

实现层面可以拆成四步：采样、构图、前向传播、跨分区通信。前向传播是指模型从输入特征计算输出预测的过程。

```python
from collections import defaultdict
import random

graph = {
    "u1": ["i1", "i2", "i3", "i4", "i5"],
    "i1": ["b1", "c1"],
    "i2": ["b1", "c2"],
    "i3": ["b2", "c1"],
    "i4": ["b3", "c3"],
    "i5": ["b3", "c1"],
}

def sample_neighbors(graph, seeds, fanout):
    sampled = set(seeds)
    frontier = list(seeds)
    edges = []

    for k in fanout:
        next_frontier = []
        for node in frontier:
            nbrs = graph.get(node, [])
            picked = nbrs[:k]
            for nbr in picked:
                sampled.add(nbr)
                next_frontier.append(nbr)
                edges.append((node, nbr))
        frontier = next_frontier

    return sampled, edges

nodes, edges = sample_neighbors(graph, ["u1"], fanout=[4, 2])
assert "u1" in nodes
assert len(nodes) <= 1 + 4 + 4 * 2
assert len(edges) <= 4 + 4 * 2
```

这段代码是玩具例子，展示 fanout 如何限制展开规模。真实框架会处理张量、边重编号、负采样、异构节点类型和多进程加载。

训练伪代码如下：

```python
seeds = sample_seed_nodes(batch_size=1024)
subgraph = neighbor_sampler(seeds, fanout=[15, 10])
out = gnn(subgraph.x, subgraph.edge_index)
loss = compute_loss(out, labels)
loss.backward()
optimizer.step()
```

| 阶段 | 输入 | 处理 | 输出 |
|---|---|---|---|
| 采样 | seed node、fanout | 多层邻居采样 | 节点集合、边集合 |
| 构图 | 采样结果 | 重编号、取特征、取标签 | batch subgraph |
| 前向传播 | 特征、边 | 消息传递和聚合 | 节点表示或预测分数 |
| 分布式通信 | 跨分区依赖 | 拉取 halo 特征 | 本地可计算 batch |

工程版例子：在 PyG 中，常用 `NeighborLoader` 做邻居采样，用 `num_neighbors=[15, 10]` 控制两层 fanout；更进一步可以用分层邻居采样减少后续层的无效计算。在 DGL 中，超大图训练通常先 `partition_graph`，生成多个分区文件，再由分布式训练进程加载本地分区，并通过 partition book 定位远端节点和边。

实际落地时，采样通常在 CPU 上执行，GPU 负责模型计算；高性能系统会把特征拉取做成异步，避免 GPU 等待数据。如果采样器、特征存储和训练进程没有流水线化，GPU 利用率会很低。

---

## 工程权衡与常见坑

fanout 不是越大越好。fanout 太小会丢信息，fanout 太大会让 batch 失控。正确做法是从较小配置开始，例如 $[10, 5]$ 或 $[15, 10]$，再用离线指标和线上效果验证收益。

| 常见坑 | 结果 | 规避方式 |
|---|---|---|
| fanout 太大 | batch 节点数暴涨，显存溢出 | 限制分层 fanout，监控每 batch 节点数 |
| 只看节点不看边 | 分区节点均衡但通信很高 | 同时看 edge cut 和 halo 比例 |
| 只看采样不看通信 | 单机有效，多机变慢 | 监控每步通信耗时和 GPU 等待 |
| 训练/评估图版本不一致 | 离线指标不稳定 | 固定图快照和特征版本 |
| 忽略热门节点 | 头部商品、品牌造成局部爆炸 | 对高阶节点做截断或按权重采样 |

| 监控指标 | 说明 |
|---|---|
| 每 batch 采样节点数 | 判断 fanout 是否导致规模失控 |
| 每 batch 采样边数 | 判断消息传递计算量 |
| edge cut 比例 | 判断分区质量 |
| halo 节点占比 | 判断跨分区依赖 |
| 每步通信耗时 | 判断是否被网络拖慢 |
| GPU 利用率 | 判断计算是否吃满 |

节点均衡不等于通信均衡。一个真实工程场景是：4 台机器各分到 2.5 亿节点，看起来很均衡；但热门商品和类目连接大量用户，如果这些高连接节点被切散，训练时每台机器都会频繁拉远端特征。结果是节点数均衡，通信却不均衡。

---

## 替代方案与适用边界

没有一种方法能覆盖所有场景。选择方案要看图规模、图密度、硬件条件和训练目标。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 全图训练 | 实现简单，信息完整 | 显存和内存压力大 | 小图、单机可放下 |
| 邻居采样 | 通用，容易接入 mini-batch | 有采样方差，fanout 要调参 | 大多数大图推荐训练 |
| Cluster-GCN | batch 内部更稠密，减少无效边 | 依赖聚类质量，动态图维护成本高 | 簇结构明显的图 |
| 分布式分区训练 | 支持超大图和多机训练 | 系统复杂，通信调优成本高 | 单机放不下的工业图 |
| 分层训练 / HGAM | 减少深层无效计算 | 和框架实现耦合更强 | 较深 GNN 或采样开销明显的场景 |

边界判断标准很直接：如果一台机器能稳定训练，优先用邻居采样，不要过早引入复杂分布式系统；如果图很密且社区结构明显，可以考虑 Cluster-GCN；如果图、特征或 embedding 表单机放不下，才需要系统性处理分区、halo 缓存、异步特征拉取和通信压缩。

新手版本的选择规则是：图不算特别大，先用邻居采样；图很密，考虑 Cluster-GCN；一台机器放不下，必须考虑分区和分布式。显存不是主要瓶颈时，过度复杂的分区和通信优化未必划算。

---

## 参考资料

| 来源 | 用途 |
|---|---|
| GraphSAGE | 理解邻居采样和归纳式表示学习 |
| Cluster-GCN | 理解基于聚类的子图 batch 训练 |
| PyG Neighbor Sampling | 理解框架中的邻居采样实现 |
| PyG Hierarchical Neighborhood Sampling | 理解分层裁剪如何减少无效计算 |
| DGL Distributed Partition | 理解图分区、halo 节点和分布式加载 |
| GIST | 理解更广义的分布式 GCN 训练思路 |

1. [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)
2. [Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://www.kdd.org/kdd2019/accepted-papers/view/cluster-gcn-an-efficient-algorithm-for-training-deep-and-large-graph-convol)
3. [Scaling GNNs via Neighbor Sampling - PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/neighbor_loader.html)
4. [Hierarchical Neighborhood Sampling - PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/advanced/hgam.html)
5. [dgl.distributed.partition_graph - DGL Documentation](https://www.dgl.ai/dgl_docs/en/0.8.x/generated/dgl.distributed.partition_graph.html)
6. [GIST: distributed training for large-scale graph convolutional networks](https://link.springer.com/article/10.1007/s41468-023-00127-8)
