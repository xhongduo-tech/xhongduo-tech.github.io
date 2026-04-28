## 核心结论

GCN（Graph Convolutional Network，图卷积网络，白话讲就是“让图上每个节点把邻居信息收进来再更新自己”的神经网络层）最经典的形式，本质上不是像图像卷积那样在规则网格上滑动卷积核，而是在图结构上做一次**归一化邻居聚合**，再接一个**共享线性变换**。

它的经典更新式是：

$$
\tilde A = A + I,\quad \tilde D_{ii} = \sum_j \tilde A_{ij}
$$

$$
H^{(l+1)} = \sigma(\tilde D^{-1/2}\tilde A\tilde D^{-1/2} H^{(l)} W^{(l)})
$$

这条公式可以直接记成三步：

1. 给每个节点加自环，也就是让节点在聚合时看见自己。
2. 对“自己 + 邻居”的信息做归一化，避免高出度节点数值过大。
3. 用同一个权重矩阵做线性变换，再过激活函数。

从理论上看，GCN 可以看作谱图卷积（spectral graph convolution，白话讲就是“从图拉普拉斯矩阵的频域角度定义卷积”）的一阶近似；从实现上看，它更像“消息传递 + 平均聚合”的一个极简基线。

它最适合的任务是**半监督节点分类**：图里只有少量节点有标签，模型通过图结构把局部信息传播到未标注节点。它之所以经典，是因为简单、稳定、容易复现；它之所以不是终局方案，是因为层数一深容易过平滑，大图训练代价也不低，对异配图的适应性也有限。

---

## 问题定义与边界

先把问题说清楚。GCN 解决的是**图上的表示学习**，尤其是**节点分类**。图由节点和边组成，节点有特征，边表示关系。模型要做的是：利用节点特征和图结构，为每个节点学出一个更有判别力的表示，再用于分类。

一个标准定义如下：

- 输入：
  - 图结构 $G=(V,E)$
  - 邻接矩阵 $A \in \mathbb R^{N\times N}$
  - 节点特征矩阵 $X \in \mathbb R^{N\times F}$
- 输出：
  - 节点表示 $H \in \mathbb R^{N\times d}$
  - 或最终分类概率 $Z \in \mathbb R^{N\times C}$

这里的关键前提不是“有图就能用”，而是**图结构本身必须有信息**。如果边表示的关系和目标标签相关，GCN 往往有效；如果边很嘈杂，或者邻居之间天然差异很大，简单邻居平均就可能伤害性能。

下表给出这个问题的边界：

| 项目 | 内容 |
| --- | --- |
| 输入是什么 | 节点特征矩阵 $X$、邻接矩阵 $A$、少量已标注节点 |
| 输出是什么 | 每个节点的分类结果或低维表示 |
| 典型任务 | 半监督节点分类、节点表征学习 |
| 最常见场景 | 引文网络、知识图谱的简化同构子任务、社交关系图、分子图中的节点任务 |
| 不适合什么 | 图结构噪声很大、异配强、需要复杂边类型建模、超大图实时推理 |
| 核心假设 | 邻近节点往往更相似，即局部同配性较强 |

“同配性”（homophily，白话讲就是“连在一起的节点往往更像”）非常关键。比如论文引用网络里，机器学习论文更容易引用机器学习论文；这种场景下，邻居平均通常是有帮助的。

一个**真实工程例子**是引文网络论文分类。节点是论文，边是引用关系，节点特征可以是词袋向量或文本编码，只有少量论文有学科标签。GCN 的目标是：让“已标注论文”的局部结构信息，沿着引用边传播到未标注论文。Kipf 和 Welling 的原论文就是在这种 citation network 上验证 GCN 的代表性效果。

相反，如果是在强异配社交图中，比如“骗子账号更常连接正常用户、正常用户也可能反向连接骗子账号”，那么“邻居像我”这个假设就变弱了。此时简单 GCN 往往不是最合适的选择。

---

## 核心机制与推导

GCN 公式看起来像矩阵乘法，真正理解时要拆成节点级更新。第 $l$ 层中，节点 $i$ 的更新可以写成：

$$
h_i^{(l+1)} = \sigma\!\left(\sum_{j\in \mathcal N(i)\cup\{i\}} \frac{1}{\sqrt{\tilde d_i\tilde d_j}}\, h_j^{(l)} W^{(l)}\right)
$$

这里：

- $\mathcal N(i)$ 是节点 $i$ 的邻居集合。
- $\tilde d_i$ 是加自环后的度。
- $W^{(l)}$ 是第 $l$ 层共享权重。
- $\sigma$ 是非线性激活函数，比如 ReLU。

这句话翻成白话就是：**把自己和邻居的特征按度数缩放后求和，再做一次线性映射**。

### 为什么要加自环

自环（self-loop，白话讲就是“让节点把自己也当作邻居”）对应 $\tilde A = A + I$。

如果不加自环，节点更新时只看邻居，不看自己。这样会导致两类问题：

1. 自身特征被过快稀释。
2. 孤立节点没有可传播信息，表示退化。

加自环后，节点至少能保留一部分原始信息。这也是为什么很多实现默认自动加自环。

### 为什么要归一化

如果只做 $AH$，高度节点会把大量邻居特征直接累加进来，数值尺度随度数变大，训练容易不稳定。归一化就是在做“不同度数节点之间的尺度对齐”。

GCN 使用的是对称归一化：

$$
\hat P = \tilde D^{-1/2}\tilde A\tilde D^{-1/2}
$$

它不是简单平均，但效果接近“按图结构校正后的加权平均”。这个矩阵有两个直观作用：

1. 控制数值规模，避免高出度节点主导。
2. 让无向图上的传播更对称，保留更稳定的谱性质。

### 为什么共享权重

共享权重（shared weights，白话讲就是“所有节点都用同一套线性变换参数”）让模型参数量与节点数无关，只与特征维度有关。否则每个节点一套参数，模型既学不动，也无法泛化到新图。

### 玩具例子：两节点图

看一个最小数值例子。图里只有两个节点，互相连边：

- 特征矩阵
  $$
  X = \begin{bmatrix}1 & 0 \\ 0 & 2\end{bmatrix}
  $$
- 权重矩阵 $W = I$
- 激活函数 $\sigma(x)=x$

原始邻接矩阵：

$$
A=\begin{bmatrix}0&1\\1&0\end{bmatrix}
$$

加自环后：

$$
\tilde A=\begin{bmatrix}1&1\\1&1\end{bmatrix},\quad
\tilde D=\begin{bmatrix}2&0\\0&2\end{bmatrix}
$$

于是传播矩阵：

$$
\hat P=\tilde D^{-1/2}\tilde A\tilde D^{-1/2}
=\frac{1}{2}\begin{bmatrix}1&1\\1&1\end{bmatrix}
$$

一层传播后：

$$
H^{(1)} = \hat P X
= \frac{1}{2}\begin{bmatrix}1&1\\1&1\end{bmatrix}
\begin{bmatrix}1&0\\0&2\end{bmatrix}
=
\begin{bmatrix}0.5&1\\0.5&1\end{bmatrix}
$$

两个节点第一次更新后变得完全一致。这说明 GCN 的核心效应就是**局部平滑**：邻居之间的信息被混合，表示逐渐靠近。

### 从谱图卷积到一阶近似

原始谱图卷积通常基于图拉普拉斯矩阵（graph Laplacian，白话讲就是“描述图结构平滑性的矩阵”）的特征分解定义滤波器。但直接特征分解代价高，难以扩展到大图。

Kipf 和 Welling 的关键简化是：

1. 用切比雪夫多项式近似谱滤波器。
2. 再把多项式截断到一阶邻域。
3. 通过参数合并与重参数化得到最终传播形式。

最终模型只保留一跳邻居信息，因此单层只做一次局部传播。堆叠两层，就是两跳感受野；堆叠三层，就是三跳感受野。

### 为什么复杂度按边线性增长

如果图是稀疏的，$\tilde A$ 里非零元素数量大约与边数 $|E|$ 同阶。矩阵乘法 $\tilde A H$ 本质上是在对每条边做一次消息传递，所以稀疏实现下复杂度近似是：

$$
O(|E|F)
$$

这里 $F$ 是特征维度。这也是 GCN 在中小规模稀疏图上能高效运行的原因。

---

## 代码实现

先给最小实现，再给框架版。新手最容易混淆的是：**加自环、归一化、先聚合还是先线性变换、矩阵形状是否对齐**。

### 从零写前向传播

下面这个版本只依赖 `numpy`，能直接运行，并用 `assert` 验证前面的玩具例子。

```python
import numpy as np

def gcn_layer(x, a, w):
    """
    x: [N, F_in]
    a: [N, N] adjacency matrix, 0/1
    w: [F_in, F_out]
    """
    n = a.shape[0]
    i = np.eye(n, dtype=np.float64)
    a_tilde = a + i

    degree = np.sum(a_tilde, axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree))

    p = d_inv_sqrt @ a_tilde @ d_inv_sqrt
    return p @ x @ w

# toy example
A = np.array([
    [0.0, 1.0],
    [1.0, 0.0],
])

X = np.array([
    [1.0, 0.0],
    [0.0, 2.0],
])

W = np.eye(2)

H1 = gcn_layer(X, A, W)
expected = np.array([
    [0.5, 1.0],
    [0.5, 1.0],
])

assert np.allclose(H1, expected), f"unexpected result: {H1}"
assert np.allclose(H1[0], H1[1]), "two nodes should become identical after one layer"

print(H1)
```

这段代码与公式是一一对应的：

- `a + i` 对应 $\tilde A=A+I$
- `degree` 对应 $\tilde D$
- `d_inv_sqrt @ a_tilde @ d_inv_sqrt` 对应对称归一化
- `p @ x @ w` 对应先传播再线性变换

如果你把 `a + i` 去掉，结果就不再包含自身信息；如果你把归一化去掉，高度节点会更容易放大数值。

### 输入输出维度说明

| 符号 | 含义 | 形状 |
| --- | --- | --- |
| $A$ | 邻接矩阵 | $N \times N$ |
| $X$ | 输入节点特征 | $N \times F_{in}$ |
| $H^{(l)}$ | 第 $l$ 层节点表示 | $N \times F_l$ |
| $W^{(l)}$ | 第 $l$ 层权重矩阵 | $F_l \times F_{l+1}$ |
| $\tilde A$ | 加自环后的邻接矩阵 | $N \times N$ |
| $\tilde D$ | 加自环后的度矩阵 | $N \times N$ |
| $Z$ | 最终分类 logits | $N \times C$ |

### 两层 GCN 的直观结构

一个两层节点分类模型通常写成：

$$
H^{(1)} = \mathrm{ReLU}(\hat P X W^{(0)})
$$

$$
Z = \hat P H^{(1)} W^{(1)}
$$

其中 $Z$ 的每一行对应一个节点的分类分数。

为什么两层最常见？因为第一层收集一跳邻居，第二层收集二跳邻居。对很多引文图和属性图来说，二跳信息已经足够形成强基线，再往深走就更容易过平滑。

### PyTorch Geometric 版本

下面是一个典型的 PyG 两层 GCN 节点分类例子，使用 Cora 数据集。它需要本地安装 `torch` 和 `torch_geometric`。

```python
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

dataset = Planetoid(root="/tmp/Cora", name="Cora")
data = dataset[0]

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)   # 等价于一层归一化图传播 + 线性变换
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(dataset.num_features, 16, dataset.num_classes).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    acc = (pred[mask] == data.y[mask]).float().mean().item()
    return acc

for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        val_acc = evaluate(data.val_mask)
        test_acc = evaluate(data.test_mask)
        print(f"epoch={epoch:03d} loss={loss:.4f} val={val_acc:.4f} test={test_acc:.4f}")
```

这个实现里，`GCNConv` 已经默认处理了自环和归一化逻辑。工程上这很方便，但也容易让新手“只会调包，不懂公式”。所以建议先理解前面的 `numpy` 版，再看框架版。

### 真实工程例子：知识图谱中的实体分类

标题里有“知识图谱 / 图神经网络”这个标签，但要注意，**经典 GCN 更适合处理同构图或简化后的无类型图**。真实知识图谱通常是多关系图，边有类型，比如：

- `(公司A, 投资, 公司B)`
- `(人物C, 任职于, 公司A)`
- `(产品D, 属于, 品类E)`

如果把这些边类型全部抹平，构成一个普通邻接图，GCN 可以先做一个快速基线任务，比如“实体类别预测”或“风险实体识别”。例如：

- 节点：公司、法人、产品、门店
- 边：投资、任职、控制、交易
- 任务：识别高风险公司节点

此时 GCN 的价值在于：快速验证“图结构是否有帮助”。如果连这个基线都没有提升，往往说明问题不在模型深度，而在图构造、特征质量或标签定义上。如果基线有效，再考虑升级到 R-GCN、HAN 或更复杂的异构图模型。

---

## 工程权衡与常见坑

GCN 很容易上手，但工程上最重要的不是“能不能跑”，而是“什么时候它已经不适合继续堆了”。

### 坑位对照表

| 现象 | 可能原因 | 处理建议 |
| --- | --- | --- |
| 训练集效果上涨，验证集掉得快 | 过拟合、标签少、图噪声大 | 加 Dropout、权重衰减、早停、检查图构造 |
| 多加几层后准确率反而下降 | 过平滑 | 先退回 2 层，尝试残差、初始残差、GCNII |
| 大图训练很慢或显存爆 | 全邻居传播代价高 | 邻居采样、子图训练、Cluster-GCN、GraphSAGE |
| 节点度差异极大时训练不稳 | 图结构长尾严重 | 检查归一化、裁剪异常边、改图构造 |
| 图上连边很多但收益不大 | 图结构噪声大 | 做边过滤、边加权、重新定义关系 |
| 邻居越多效果越差 | 异配强，平均聚合错误混合信息 | 尝试 GAT、MixHop、H2GCN、GPR-GNN 等 |

### 为什么 2 层 GCN 是强基线

很多公开数据集上，2 层 GCN 是默认起点，不是因为它最强，而是因为它满足几个现实条件：

1. 实现极简单。
2. 两跳感受野往往足够覆盖局部语义。
3. 参数少，调参成本低。
4. 结果稳定，便于判断图结构是否有效。

如果你一开始就上深层复杂模型，很可能分不清提升到底来自模型结构，还是来自数据处理、采样策略和超参数技巧。

### 过平滑是什么

过平滑（oversmoothing，白话讲就是“层数一深，节点表示越来越像，最后分不出谁是谁”）是 GCN 最经典的问题之一。因为每一层都在做局部平均，反复传播后，节点表示会逐渐趋向同一子空间，类间边界被抹平。

从矩阵角度看，反复乘 $\hat P$ 类似不断做平滑操作：

$$
H^{(L)} \approx \hat P^L X W
$$

当 $L$ 变大时，很多节点表示会收敛到越来越相似的模式。

实际表现通常是：

- 2 层效果最好
- 4 层开始下降
- 更深后训练还能收敛，但验证性能明显变差

缓解办法包括：

- 残差连接
- 初始残差
- Jumping Knowledge
- PairNorm
- GCNII

其中 GCNII 的思路很直接：深层传播时，不让初始特征完全消失，并显式控制平滑强度。

### 过压缩是什么

过压缩（over-squashing，白话讲就是“远处大量信息被压进固定维度向量，信息来不及表达”）更偏图结构瓶颈问题。即使没有明显过平滑，远距离依赖也可能在少数边上被挤压，导致模型接收不到足够上下文。

这在树状图、长链图、局部桥接很强的图里更明显。GCN 不是专门为解决这个问题设计的，所以如果任务依赖远程结构，单纯堆层通常效果有限。

### 大图上的传播成本

GCN 的标准训练是全图传播。对小图没问题，但对工业图，节点和边规模可能上亿。此时即使单层复杂度对边线性增长，整体代价仍然很高。

常见工程处理方式有：

- 邻居采样：每层只采部分邻居
- 子图训练：按社区或簇切子图
- 预计算传播特征：先做扩散，再训练 MLP
- 离线图编码 + 在线轻模型推理

因此，工业里“直接全图 GCN 端到端训练”并不总是主流，更常见的是把它当作离线表征模块或基线验证模块。

---

## 替代方案与适用边界

GCN 的地位更像“重要基线”，而不是“所有图任务的默认最优解”。你应该先判断任务的结构，再决定是否用它。

下表给出几个常见替代方案：

| 模型 | 核心思想 | 优点 | 局限 | 适用边界 |
| --- | --- | --- | --- | --- |
| GCN | 归一化邻居聚合 | 简单、稳定、强基线 | 深层易过平滑，对异配不友好 | 同配性较强的节点分类 |
| GraphSAGE | 邻居采样 + 聚合 | 更适合大图，支持归纳学习 | 采样带来方差 | 超大图、在线增量节点 |
| GAT | 用注意力学习邻居权重 | 聚合更灵活，可区分邻居重要性 | 计算更重，稳定性依赖实现 | 邻居质量差异较大时 |
| R-GCN | 面向多关系图建模 | 适合知识图谱、异构关系 | 参数和实现更复杂 | 多边类型图、关系推理 |
| GCNII | 深层 GCN 改进 | 缓解过平滑，可堆更深 | 仍需调参与结构判断 | 想保留 GCN 风格但加深层数 |
| APPNP | 预测与传播分离 | 保留 MLP 表达能力，传播可控 | 传播超参数要调 | 半监督节点分类 |
| H2GCN / 异配模型 | 针对异配图设计 | 异配场景更稳 | 不一定适合普通同配图 | 邻居不相似的图 |

### 引文网络为什么适合 GCN

引文网络里：

- 节点有文本特征
- 引用关系有明显主题相似性
- 标签少但结构可传播
- 局部邻域通常就有足够信号

这正好匹配 GCN 的机制：邻居平均通常不会把语义搞乱，反而能补足标注稀缺带来的监督不足。

### 强异配社交网络为什么可能不适合

设想一个反欺诈图：

- 欺诈账号常与正常账号交互
- 正常账号也可能被动连到异常节点
- 邻居标签并不与自己一致

这时“邻居越像我越好”不成立。GCN 的平均聚合可能把区分信号洗掉。更合适的方向通常是：

- 设计边类型和方向特征
- 使用注意力或关系建模
- 尝试异配友好模型
- 先做局部子图与标签传播模式分析

所以工程判断应该是：**先问图结构是否支持局部平滑，再决定是否用 GCN**，而不是“图任务默认先上 GCN 再说”。

---

## 参考资料

1. [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907)  
适合看理论来源、公式推导和 citation network 实验设定，是理解 GCN 的第一手论文。

2. [PyTorch Geometric `GCNConv` 文档](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.GCNConv.html)  
适合对照工程实现，确认 `GCNConv` 默认处理了哪些归一化和自环逻辑。

3. [DGL GCN Tutorial](https://www.dgl.ai/dgl_docs/tutorials/models/1_gnn/1_gcn.html)  
适合从教程角度看消息传递实现，帮助把公式映射到实际代码。

4. [PyG `GCN2Conv` / GCNII 文档](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCN2Conv.html)  
适合继续看深层 GCN 的改进方向，理解如何缓解过平滑问题。
