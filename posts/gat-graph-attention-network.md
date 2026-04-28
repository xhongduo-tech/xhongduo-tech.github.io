## 核心结论

GAT，Graph Attention Network，中文常译为“图注意力网络”，本质是在图上的消息传递里，用**可学习的邻居权重**替代固定平均规则。

白话解释：同一个节点收到很多邻居的信息时，GAT 不假设“每个邻居都差不多重要”，而是先给每条边打分，再决定该重点听谁。

适用判断：如果你的图里“邻居贡献明显不均匀”，例如少数关键邻居决定结果，GAT 往往比 GCN 更合适；如果邻居大多同等重要，GAT 的额外成本可能不值。

核心公式是：

$$
z_i = W h_i
$$

$$
e_{ij} = \mathrm{LeakyReLU}\left(a^T[z_i \Vert z_j]\right)
$$

$$
\alpha_{ij}=\mathrm{softmax}_j(e_{ij})
$$

$$
h'_i=\sigma\left(\sum_j \alpha_{ij} z_j\right)
$$

其中 $\alpha_{ij}$ 是注意力系数，也就是“节点 $i$ 在聚合时给邻居 $j$ 分配的权重”。

| 维度 | GCN | GAT |
|---|---|---|
| 权重来源 | 固定归一化邻接矩阵 | 由注意力网络学习 |
| 聚合方式 | 近似固定平均 | 按邻居重要性加权 |
| 表达能力 | 中等 | 更强，适合非均匀邻居 |
| 计算开销 | 较低 | 较高，边上打分更贵 |
| 适合场景 | 同质图、快速基线 | 邻居重要性差异明显的图 |

---

## 问题定义与边界

GAT 要解决的问题很具体：**图结构已知时，节点聚合邻居信息，如何让不同邻居有不同权重**。

白话解释：它不是重新发明图，也不是学习“应该连哪些边”，而是在已有边上学习“这些边各自该有多大分量”。

输入通常有两部分：

| 输入/输出 | 内容 |
|---|---|
| 输入 1 | 节点特征 $h_i$，也就是每个节点的向量表示 |
| 输入 2 | 图结构 $A$ 或邻接表，也就是谁和谁相连 |
| 输出 | 更新后的节点表示 $h'_i$ |

这里的“节点特征”可以理解成节点自带的数值描述，比如论文的词向量、用户画像、分子原子属性。“图结构”就是节点之间的连接关系。

它常见于以下任务：

| 任务类型 | 是否适合 GAT | 原因 |
|---|---|---|
| 节点分类 | 是 | 直接学习邻居贡献 |
| 图分类的子模块 | 是 | 可作为局部编码层 |
| 归纳式图学习 | 是 | 对未见节点可迁移 |
| 分子图 | 常见 | 原子邻居重要性不同 |
| 推荐系统 | 常见 | 用户行为边强弱不同 |
| 风控图 | 常见 | 少数高风险邻居更关键 |

问题边界也要说清楚：

| 边界问题 | 适合 GAT 的情况 | 不适合 GAT 的情况 |
|---|---|---|
| 图结构是否已知 | 已知 | 边本身大量缺失或噪声极大 |
| 邻居权重是否不均匀 | 明显不均匀 | 大多邻居贡献接近 |
| 是否需要一定可解释性 | 需要查看相对权重 | 需要严格因果解释 |
| 是否能接受较高成本 | 可以 | 延迟和显存非常敏感 |

玩具例子可以先看一个引用图。假设某篇论文连接了 20 篇参考文献，但真正和它主题强相关的只有 2 篇。GCN 会把 20 个邻居按固定规则混在一起，GAT 则会尝试把权重集中到那 2 篇更相关的论文上。

但边界也很明确。如果图非常稠密，一个节点连了上万个邻居，而且这些邻居大多同等重要，那么 GAT 可能只是多花算力，收益并不明显。

---

## 核心机制与推导

GAT 的核心流程可以压缩成 5 步：

| 步骤 | 数学对象 | 作用 |
|---|---|---|
| 输入特征 | $h_i$ | 节点原始表示 |
| 线性映射 | $z_i = Wh_i$ | 投影到可学习子空间 |
| 边打分 | $e_{ij}$ | 估计邻居 $j$ 对节点 $i$ 的重要性 |
| 邻居内 softmax | $\alpha_{ij}$ | 在同一节点的邻居集合内部归一化 |
| 加权聚合 | $h'_i$ | 得到新的节点表示 |

“线性映射”就是用一个矩阵把原始特征变换到新的表示空间；“softmax”可以理解成把一组分数转成总和为 1 的权重；“LeakyReLU”是一种激活函数，这里主要用于让边打分更稳定。

单头 GAT 的标准形式是：

$$
z_i = W h_i
$$

$$
e_{ij}=\mathrm{LeakyReLU}(a^\top [z_i \Vert z_j])
$$

$$
\alpha_{ij}=\frac{\exp(e_{ij})}{\sum_{k\in \mathcal N(i)\cup\{i\}}\exp(e_{ik})}
$$

$$
h'_i=\sigma\left(\sum_{j\in \mathcal N(i)\cup\{i\}} \alpha_{ij} z_j\right)
$$

这里 $\Vert$ 表示拼接，也就是把两个向量首尾接起来；$\mathcal N(i)$ 表示节点 $i$ 的邻居集合；softmax 是在**同一个节点的邻居内部**归一化，不是在全图归一化。

一个最小数值例子最能说明问题。设中心节点 $i$ 有两个邻居 $j_1,j_2$，为方便演示取 $W=1$、$a=[1,1]$、$\mathrm{LeakyReLU}(x)=x$。如果：

$$
h_i=1,\quad h_{j_1}=2,\quad h_{j_2}=0
$$

那么：

$$
e_{ij_1}=1+2=3,\quad e_{ij_2}=1+0=1
$$

softmax 后：

$$
\alpha_{ij_1}=\frac{e^3}{e^3+e^1}\approx 0.881,\quad
\alpha_{ij_2}\approx 0.119
$$

于是输出更偏向 $j_1$：

$$
h'_i\approx 0.881\times 2+0.119\times 0=1.762
$$

如果做简单平均，结果会是 $(2+0)/2=1$。差异就在于：GAT 不是平均邻居，而是偏向更相关的邻居。

多头注意力是 GAT 的另一个关键机制：

$$
h'_i=\mathop{\Vert}_{k=1}^K \sigma\left(\sum_j \alpha_{ij}^{(k)} W^{(k)}h_j\right)
$$

“多头”可以理解成让多个独立的注意力子空间并行看同一批邻居。中间层常用 concat，也就是把多个头直接拼接；最后一层常用 average，也就是把多个头取平均，避免输出维度继续膨胀。

从消息传递视角看，GCN 可以视为“固定权重的邻居聚合”，GAT 可以视为“可学习权重的邻居聚合”。两者框架相似，但权重来源不同，这也是 GAT 表达能力更强的根本原因。

真实工程例子是知识图谱中的实体分类。一个实体节点可能同时连着“别名”“类型”“上下游关系”“文档来源”等不同边。固定平均会把这些关系一视同仁，而 GAT 更可能自动把高信息量邻居权重抬高，比如核心类型节点和高可信来源节点。

---

## 代码实现

先看一个最小可运行版本。这个版本不依赖深度学习框架，只演示单头 GAT 的核心计算逻辑，并显式加上 self-loop 来避免 0 入度问题。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

def gat_single_head_scalar(features, neighbors, add_self_loop=True):
    """
    features: dict[node] = scalar feature
    neighbors: dict[node] = list of incoming neighbor nodes
    简化版：W=1, attention score = h_i + h_j
    """
    out = {}
    for i, h_i in features.items():
        nbrs = list(neighbors.get(i, []))
        if add_self_loop and i not in nbrs:
            nbrs.append(i)
        assert len(nbrs) > 0, f"node {i} has no neighbors"

        scores = [h_i + features[j] for j in nbrs]
        alphas = softmax(scores)
        out[i] = sum(alpha * features[j] for alpha, j in zip(alphas, nbrs))
    return out

features = {"i": 1.0, "j1": 2.0, "j2": 0.0}
neighbors = {"i": ["j1", "j2"], "j1": [], "j2": []}

result = gat_single_head_scalar(features, neighbors)
assert result["i"] > 1.0
assert result["i"] < 2.0
assert abs(result["i"] - 1.6652409558) < 1e-6  # self-loop 后的数值
print(result)
```

这段代码里，公式和变量可以一一对应：

| 公式项 | 代码变量 | 作用 |
|---|---|---|
| $h_i$ | `h_i` | 中心节点特征 |
| $\mathcal N(i)$ | `nbrs` | 邻居集合 |
| $e_{ij}$ | `scores` | 未归一化打分 |
| $\alpha_{ij}$ | `alphas` | softmax 后权重 |
| $h'_i$ | `out[i]` | 聚合后的输出 |

如果写成伪代码，逻辑就是：

```python
for each node i:
    neighbors = incoming_neighbors(i)
    neighbors = add_self_loop_if_needed(neighbors, i)

    z_i = W(h_i)
    for each neighbor j in neighbors:
        z_j = W(h_j)
        e_ij = LeakyReLU(a^T concat(z_i, z_j))

    alpha_ij = softmax_over_neighbors(e_ij)
    h_i_new = sum(alpha_ij * z_j for j in neighbors)
```

真实工程里一般不会手写边级 softmax，而是直接用框架。下面是 PyTorch Geometric 的最小用法：

```python
import torch
from torch_geometric.nn import GATConv

x = torch.randn(4, 8)  # 4 个节点，每个节点 8 维特征
edge_index = torch.tensor([
    [0, 1, 2, 2, 3],
    [1, 0, 0, 3, 2]
], dtype=torch.long)

# 显式加 self-loop 通常更稳，很多实现也支持内部添加
conv1 = GATConv(
    in_channels=8,
    out_channels=16,
    heads=4,
    concat=True,
    dropout=0.6,
    add_self_loops=True
)

conv2 = GATConv(
    in_channels=16 * 4,
    out_channels=3,
    heads=1,
    concat=False,   # 最后一层常用平均/不拼接
    dropout=0.6,
    add_self_loops=True
)

h = conv1(x, edge_index)
logits = conv2(h, edge_index)

assert logits.shape == (4, 3)
```

这段框架代码里有几个工程重点：

| 配置项 | 含义 |
|---|---|
| `heads` | 多头数量 |
| `concat=True` | 多头拼接，维度会乘以头数 |
| `concat=False` | 多头结果做平均或等价合并 |
| `dropout` | 对注意力或特征做随机丢弃，抑制过拟合 |
| `add_self_loops=True` | 给每个节点加自环，缓解 0 入度问题 |

---

## 工程权衡与常见坑

GAT 的主要成本来自“每条边都要单独打分”。这意味着图越大、节点出度越高，时间和显存压力越明显。

白话解释：GCN 更像一次批量平均，GAT 更像先把每条边过一遍再汇总，步骤天然更重。

常见坑可以直接列出来：

| 问题 | 代价 | 规避手段 |
|---|---|---|
| 0 入度节点 | 无法聚合或输出异常 | 加 self-loop，或单独兜底 |
| 高出度节点开销大 | 时间慢、显存高 | 邻居采样、稀疏实现、减小 head 数 |
| 多头 concat 维度膨胀 | 下游层参数暴涨 | 提前规划 hidden size |
| attention 被误当解释 | 容易过度解读 | 只当模型内部权重，不当因果结论 |
| dropout / head 数过敏 | 小图训练不稳定 | 先调学习率、head 数，再调层数 |

一个典型实战例子是推荐系统中的用户行为图。某些活跃用户会连接成千上万条点击、收藏、购买边。GAT 理论上能区分哪些行为更关键，但边级注意力计算会非常贵。此时常见做法不是直接全量跑，而是先做邻居采样，只保留最近行为、强反馈行为或召回后的候选边。

配置上通常要关注这些参数：

| 参数 | 作用 | 常见建议 |
|---|---|---|
| `attn_drop` | 注意力权重 dropout | 图小可适当小一些 |
| `feat_drop` | 特征 dropout | 防止节点特征过拟合 |
| `num_heads` | 头数 | 先从 2 到 8 尝试 |
| `add_self_loop` | 加自环 | 通常建议开启 |
| hidden size | 隐层宽度 | 结合头数一起规划 |

训练不稳定时，优先检查三件事。第一，学习率是否过高。第二，head 数是否过多。第三，concat 后维度是否过大导致后续层难训。大图场景里，还要一起看精度、吞吐、显存，不要只盯住一个验证集指标。

---

## 替代方案与适用边界

GAT 不是默认更高级，而是默认更贵。只有在“邻居贡献不均匀”这个前提成立时，它的额外复杂度才更可能带来收益。

可以把几个常见模型放在一起看：

| 模型 | 核心思想 | 优点 | 缺点 | 更适合什么 |
|---|---|---|---|---|
| GCN | 固定平均聚合 | 快、简单、稳 | 对邻居差异不敏感 | 同质图、基线模型 |
| GraphSAGE | 采样后聚合 | 易扩展到大图 | 权重机制不如 GAT 细 | 大规模归纳学习 |
| GIN | 强化结构表达 | 图分类能力强 | 对节点级大图不一定划算 | 图分类、结构判别 |
| GAT | 学习邻居权重 | 对非均匀邻居更灵活 | 慢、吃显存、调参敏感 | 中等规模、邻居重要性差异明显 |

适用边界可以一句话概括：

适合：邻居重要性差异明显、希望保留一定注意力权重信息、图规模中等、能接受更高计算成本。

不适合：极大规模稀疏图、在线实时延迟严格受限、邻居同质性很强、边噪声极大且结构本身不可靠。

如果任务只是“把附近邻居大致混一下”，GCN 往往更划算。如果你更关心可扩展性和归纳能力，GraphSAGE 常常更容易工程落地。如果是图分类并且更重结构表达，GIN 可能更强。至于图 Transformer，它更偏全局建模，但计算通常更重，而且对位置编码和结构设计更依赖。

---

## 参考资料

1. [Graph Attention Networks](https://petar-v.com/GAT/)
2. [PetarV-/GAT](https://github.com/PetarV-/GAT)
3. [PyTorch Geometric: GATConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html)
4. [DGL: GATConv](https://www.dgl.ai/dgl_docs/en/0.8.x/generated/dgl.nn.pytorch.conv.GATConv.html)
5. [DGL Tutorial: Understand Graph Attention Network](https://www.dgl.ai/dgl_docs/tutorials/models/1_gnn/9_gat.html)
