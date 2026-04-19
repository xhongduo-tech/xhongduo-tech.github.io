## 核心定义

图推荐的大规模训练优化，是把全图 GNN 训练改成“局部子图 + 有限邻居 + 分片通信”的方法集合。目标不是堆更深的模型，而是让十亿级节点、百亿级边还能稳定训练。

---

## 直观解释

全图训练会遇到邻居爆炸。GraphSAGE 用邻居采样把每层扩展数压住，Cluster-GCN 用聚类后的稠密子图做 batch，层次化训练则只算当前 batch 真正需要的节点，避免 later-hop 的无用计算。

分布式训练再加一层约束：图要先分区，跨分区边会带来通信。分区越好，halo 节点越少，通信越轻；分区越差，GPU 就会花更多时间等特征和邻居数据。

---

## 关键公式/机制

> 这里把“层次化训练”按 PyG 的 hierarchical neighborhood sampling 理解。

| 符号 | 含义 |
|---|---|
| $B$ | 训练种子节点数 |
| $k_\ell$ | 第 $\ell$ 层每个节点最多采样的邻居数 |
| $L$ | GNN 层数 |
| $V_b, E_b$ | 一个 batch 对应的子图节点和边 |
| $C_{\text{cut}}$ | 跨分区边数 |
| $d$ | 节点/边特征维度 |
| $s$ | 单个特征元素字节数 |

核心机制可以压成三句：

$$
N_{\text{sample}} \le B \prod_{\ell=1}^{L} k_\ell,\quad
\text{通常近似为 } O(Bk^L)
$$

采样训练把每层展开规模封顶。层数越深，fanout 需要越保守，否则邻居数会指数增长。

$$
\text{batch cost} \approx O(|V_b| + |E_b|)
$$

Cluster-GCN 的关键是让每个 batch 落在密集子图里，减少边切割和跨簇消息传递。

$$
\text{comm bytes} \approx C_{\text{cut}} \cdot d \cdot s
$$

分布式训练的通信量，主要跟跨分区边有关。切边越少，halo replication 和特征拉取越少。

---

## 最小数值例子

设 1 个 seed user，2 层 GNN，每层最多采样 4 个邻居。

| 方案 | 节点数上界 |
|---|---:|
| 全图展开 | $1 + 50 + 50 \times 50 = 2551$ |
| 邻居采样 | $1 + 4 + 4 \times 4 = 21$ |

如果一个 batch 有 1000 条跨分区边，特征维度 $d=128$，用 fp16 传输，通信量约为：

$$
1000 \times 128 \times 2 = 256000 \text{ bytes} \approx 256 \text{ KB}
$$

如果通过更好的分区把切边降到 200，通信量就降到约 51.2 KB。

---

## 真实工程场景

电商图推荐常见做法是：把用户、商品、品牌、类目和交互边建成异构图，离线按 METIS 或类似算法分区，线上/离线训练用 neighbor sampling 做 minibatch，重图块用 Cluster-GCN 风格的子图 batch，分布式训练时用 halo 节点缓存跨分区邻居。这样能把显存峰值、通信等待和训练吞吐同时压住。

---

## 常见坑与规避

| 常见坑 | 直接后果 | 规避方式 |
|---|---|---|
| fanout 设太大 | 邻居爆炸，batch 失控 | 分层设上限，深层采样更小 |
| 分区只看节点数不看边 | 切边多，通信重 | 优先最小化 edge cut，控制 halo 比例 |
| Cluster 过稀或过碎 | batch 噪声大，吞吐差 | 保持簇内稠密，避免过小簇 |
| 只看采样，不看通信 | GPU 算完还在等数据 | 同时看采样节点数、切边数、带宽 |
| 训练/评估图版本不一致 | 指标失真 | 固定时间切分和图快照 |

---

## 参考来源

1. GraphSAGE: [Inductive Representation Learning on Large Graphs](https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs)  
2. Cluster-GCN: [DOI 10.1145/3292500.3330925](https://doi.org/10.1145/3292500.3330925)  
3. PyG Neighbor Sampling: [Scaling GNNs via Neighbor Sampling](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/neighbor_loader.html)  
4. PyG 层次化采样: [Hierarchical Neighborhood Sampling](https://pytorch-geometric.readthedocs.io/en/latest/advanced/hgam.html)  
5. DGL 分布式分区: [dgl.distributed.partition_graph](https://www.dgl.ai/dgl_docs/en/0.8.x/generated/dgl.distributed.partition_graph.html)  
6. 分布式图训练综述式实现： [GIST: distributed training for large-scale graph convolutional networks](https://link.springer.com/article/10.1007/s41468-023-00127-8)
