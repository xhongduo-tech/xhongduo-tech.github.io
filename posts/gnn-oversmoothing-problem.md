## 核心结论

过平滑是指随着图神经网络 Graph Neural Network，简称 GNN，层数增加，不同节点的 embedding 逐渐趋于相同，导致模型难以区分节点。

它的本质不是“模型拿到的信息太少”，而是“信息被反复平均后变得太像”。GNN 的每一层通常都会做两件事：从邻居节点收集信息，再把这些信息和自身表示混合。浅层时，这能补充局部上下文；深层时，节点经过多轮邻居混合，会吸收越来越相似的图结构，原本有用的差异被抹平。

| 层数 | 表示变化 | 常见效果 |
|---|---|---|
| 浅层 | 补充局部上下文 | 通常有益 |
| 深层 | 过度平均 | 节点趋同、性能下降 |

玩具例子：两个节点最开始分别是 `[1, 0]` 和 `[0, 1]`，差异很清楚。如果每一层都把自己和对方平均一次，一层后两个节点都会变成 `[0.5, 0.5]`。此时节点已经无法区分。

真实工程例子：在用户-物品二部图推荐系统中，节点包括用户和商品，边表示点击、购买、收藏等行为。GNN 层数加深后，活跃用户、热门物品以及大量与它们相连的节点会不断互相混合，embedding 更容易朝热门区域收缩。结果是召回列表更像热门榜，长尾物品覆盖下降，用户个性化变弱。

所以，GNN 不是越深越好。对多数图任务，尤其是推荐系统，2 到 3 层经常比 6 层、8 层更稳。过深模型可能把节点表示压到低维平滑子空间里，训练指标和线上效果反而下降。

---

## 问题定义与边界

设节点初始特征为 $X$，第 $l$ 层节点表示为 $H^{(l)}$。一类常见 GNN 层可以写成：

$$
H^{(0)} = X,\quad H^{(l+1)} = \sigma(SH^{(l)}W^{(l)})
$$

其中，$S$ 是传播矩阵，用来描述节点之间如何聚合邻居信息；$W^{(l)}$ 是第 $l$ 层的可学习参数；$\sigma$ 是非线性激活函数，例如 ReLU。

正常的信息传播是必要的。比如一个用户买过手机壳，他的一阶邻居商品和二阶邻居用户能提供偏好信息。模型只看用户自身 ID embedding，表达力很弱；加入邻居信息后，推荐效果通常会上升。

过平滑发生在传播过度时。一个节点看 1 层邻居，信息仍然局部；看 10 层邻居，很多节点会覆盖到相似的大范围子图，最后吸收了差不多的信息。此时问题不是“没有传播”，而是传播把差异消掉了。

还需要区分过平滑和 over-squashing。over-squashing 可以直译为“过压缩”，意思是远距离大量信息需要挤进有限维 embedding，导致关键信息传不过来。过平滑强调“节点越来越像”，over-squashing 强调“信息虽然存在，但通道太窄”。

| 概念 | 现象 | 核心问题 |
|---|---|---|
| 过平滑 | 节点表示趋同 | 区分能力下降 |
| over-squashing | 长距离信息被压扁 | 信息瓶颈 |
| 正常消息传递 | 引入局部上下文 | 提升表达 |

一个简单判断方式是：如果多层后节点 embedding 方差下降、两两距离下降、类别之间距离变小，更像过平滑；如果任务依赖远距离结构，但远端信息无法有效影响目标节点，更可能涉及 over-squashing。

---

## 核心机制与推导

先忽略非线性函数和参数矩阵，只看传播本身，GNN 可以近似成反复应用传播矩阵：

$$
H^{(l)} = S^lX
$$

常见的归一化传播矩阵是：

$$
S=\tilde D^{-1/2}\tilde A\tilde D^{-1/2},\quad \tilde A=A+I
$$

这里 $A$ 是邻接矩阵，表示图中哪些节点相连；$I$ 是单位矩阵，加上它表示给每个节点增加自环；$\tilde D$ 是 $\tilde A$ 对应的度矩阵，记录每个节点连接强度。归一化的作用是避免高度节点在聚合时数值过大。

从线性代数看，$S^lX$ 的含义是对图信号反复做平滑。图信号可以理解为“每个节点上都有一个向量值”。如果相邻节点的向量差异很大，就是高频成分；如果相邻节点的向量接近，就是低频成分。

当图连通，并且除主特征值外的其他特征值满足 $|\lambda_2|, |\lambda_3|, \dots < 1$ 时，反复乘以 $S$ 会让非主方向分量逐渐衰减：

$$
S^lX \approx \text{主特征空间中的投影}
$$

这意味着原本区分节点的高频差异会被过滤掉，表示向低频平滑子空间收缩。直观说，GNN 层数越多，越像一个低通滤波器：保留全图共同成分，削弱局部差异。

一个常用观测指标是节点间距离：

$$
\|h_i^{(l)}-h_j^{(l)}\|
$$

如果层数增加后，大量节点对距离都下降，说明表示正在趋同。

另一个指标是 Dirichlet energy：

$$
\sum_{(i,j)\in E}\|h_i^{(l)}-h_j^{(l)}\|^2
$$

它衡量相连节点之间的表示差异。能量越低，说明图上的表示越平滑。过平滑时，这个值通常会逐层下降，甚至接近 0。

可以把示意图理解成三段：浅层时，相邻节点互相补充信息，但不同社区仍然有差异；中层时，同一社区内部更平滑；深层时，不同社区也开始混在一起，节点表示整体趋同。

---

## 代码实现

下面用 NumPy 做一个最小实验。重点不是训练完整 GNN，而是验证“传播层数增加会让表示趋同”，并观察残差连接如何保留差异。残差连接 Residual connection，是指把输入表示按一定比例直接加回输出，避免每层完全依赖邻居平均后的结果。

最小数值例子：

$$
S=\begin{bmatrix}0.5&0.5\\0.5&0.5\end{bmatrix},\quad
X=\begin{bmatrix}1&0\\0&1\end{bmatrix}
$$

一次传播后：

$$
H^{(1)}=SX=\begin{bmatrix}0.5&0.5\\0.5&0.5\end{bmatrix}
$$

两个节点完全一样，原来距离是 $\sqrt2$，现在变成 0。若加残差：

$$
H' = 0.5X+0.5SX
$$

则得到：

$$
\begin{bmatrix}0.75&0.25\\0.25&0.75\end{bmatrix}
$$

两个节点仍然保留差异。

```python
import numpy as np

def normalized_adj(adj, add_self_loop=True):
    if add_self_loop:
        adj = adj + np.eye(adj.shape[0])
    degree = adj.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    return d_inv_sqrt @ adj @ d_inv_sqrt

def pairwise_mean_distance(h):
    distances = []
    for i in range(len(h)):
        for j in range(i + 1, len(h)):
            distances.append(np.linalg.norm(h[i] - h[j]))
    return float(np.mean(distances))

def embedding_variance(h):
    return float(np.mean(np.var(h, axis=0)))

def dirichlet_energy(adj, h):
    energy = 0.0
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i, j] > 0:
                energy += np.linalg.norm(h[i] - h[j]) ** 2
    return float(energy)

# 玩具例子：两个节点相连
adj2 = np.array([
    [0.0, 1.0],
    [1.0, 0.0],
])
x2 = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
])

s2 = normalized_adj(adj2)
h1 = s2 @ x2
h_res = 0.5 * x2 + 0.5 * h1

assert np.allclose(s2, np.array([[0.5, 0.5], [0.5, 0.5]]))
assert np.allclose(h1, np.array([[0.5, 0.5], [0.5, 0.5]]))
assert pairwise_mean_distance(x2) > pairwise_mean_distance(h1)
assert pairwise_mean_distance(h_res) > pairwise_mean_distance(h1)

# 稍大一点的链式图：观察多层传播后的距离和能量
adj = np.array([
    [0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0],
], dtype=float)

x = np.eye(5)
s = normalized_adj(adj)

h = x.copy()
rows = []
for layer in range(1, 9):
    h = s @ h
    rows.append((
        layer,
        pairwise_mean_distance(h),
        embedding_variance(h),
        dirichlet_energy(adj, h),
    ))

for layer, dist, var, energy in rows:
    print(f"layer={layer}, mean_dist={dist:.4f}, var={var:.4f}, energy={energy:.4f}")

assert rows[-1][1] < rows[0][1]
assert rows[-1][2] < rows[0][2]
assert rows[-1][3] < rows[0][3]
```

一组典型输出趋势如下：

| 层数 | 平均两两距离 | 方差 | Dirichlet energy |
|---|---:|---:|---:|
| 1 | 较大 | 较大 | 较高 |
| 2 | 下降 | 下降 | 下降 |
| 3+ | 明显下降 | 接近 0 | 接近 0 |

真实工程中可以把 `h` 换成用户和物品 embedding，把 `adj` 换成交互图邻接矩阵。训练不同 GNN 层数时，同时记录 Recall、NDCG、embedding 方差、节点对距离、热门物品召回占比，就能判断性能下降是否和过平滑有关。

---

## 工程权衡与常见坑

过平滑不能只靠最终 AUC、Recall 或 NDCG 判断。最终指标下降只说明模型变差，不说明原因。工程上需要同时做层数消融和表示诊断。

层数消融是指固定其他条件，只改 GNN 层数，例如比较 1、2、3、4、6 层。表示诊断是指检查 embedding 本身是否正在塌缩，例如方差是否下降、两两距离是否变小、类间距离是否缩短、Dirichlet energy 是否逐层降低。

| 坑 | 现象 | 误判风险 | 规避方式 |
|---|---|---|---|
| 把过平滑和 over-squashing 混淆 | 表示趋同 vs 信息传不远 | 诊断方向错 | 分别看距离和长程依赖 |
| 只加 Residual 不做消融 | 指标略回升 | 以为问题解决 | 做 1/2/3/4 层对比 |
| 只看最终 AUC/Recall | 不知道为什么变差 | 无法定位问题 | 加 embedding 诊断指标 |
| 盲目加深层数 | 训练更慢，效果更差 | 以为模型容量不足 | 先验证浅层基线 |
| 忽略热门节点影响 | 召回更集中 | 误判为模型学到主流偏好 | 监控长尾覆盖和热门占比 |

推荐系统里的一个常见现象是：从 2 层加到 6 层后，离线 Recall 可能短期变化不大，但长尾覆盖下降，热门商品曝光上升。原因是热门物品连接大量用户，多层传播后，它们的 embedding 会影响更大范围的节点。大量用户向量被拉向热门物品方向，召回结果自然更集中。

常用诊断指标包括：

| 指标 | 含义 | 过平滑时的变化 |
|---|---|---|
| embedding 方差 | 节点表示整体分散程度 | 下降 |
| 节点对距离 | 不同节点向量的平均距离 | 下降 |
| 类间距离 | 不同标签或群体之间的距离 | 下降 |
| Dirichlet energy | 相邻节点表示差异 | 下降 |
| 长尾覆盖 | 推荐结果覆盖非热门物品的能力 | 下降 |

Residual、LayerNorm、Jumping Knowledge 等方法通常是缓解，不是保证根治。LayerNorm 是层归一化，用来稳定每层表示的数值分布；Jumping Knowledge 是把不同层的表示聚合起来，让模型可以同时使用浅层和深层信息。它们能改善训练和表示保留，但是否有效取决于图结构、任务目标和层数。

---

## 替代方案与适用边界

如果任务对层数非常敏感，优先考虑浅层模型，而不是继续堆深。对推荐系统来说，如果目标是利用用户和物品的邻居协同信号，2 到 3 层通常已经覆盖了足够多的行为上下文。继续加深可能把个性化信号洗掉。

LightGCN 是推荐系统里常见的简化图卷积模型。它去掉了复杂的特征变换和非线性激活，主要保留 user-item 图上的线性传播，并把多层 embedding 做加权组合。它的工程价值在于：结构简单、训练稳定、容易控制传播深度。

| 方案 | 主要作用 | 优点 | 局限 |
|---|---|---|---|
| Residual | 保留原始表示 | 简单有效 | 不能保证根治 |
| LayerNorm | 稳定训练 | 降低数值波动 | 对趋同问题有限 |
| Jumping Knowledge | 聚合不同层信息 | 保留浅层差异 | 结构更复杂 |
| LightGCN | 去掉非线性和特征变换 | 简洁稳定 | 表达能力更依赖图结构 |
| 层数消融 | 找到合适传播深度 | 成本低，结论直接 | 需要重复训练 |
| 初始 embedding 加权和 | 保留个体差异 | 推荐场景常用 | 权重需要调参 |

适用边界需要明确。

图很稠密、邻域噪声大时，深层更容易出问题。因为每层都会引入更多邻居，噪声和热门节点影响会快速扩散。

任务依赖局部邻居时，浅层通常更合适。比如电商召回中，用户近期点击、购买、相似用户行为往往已经能提供主要信号，不一定需要很远的图距离。

如果任务确实依赖强长程关系，不应该默认靠加深 GNN 解决。需要先判断问题是过平滑、over-squashing，还是图构建本身缺边。可能更合适的方案包括改图结构、加入边特征、采样关键路径、使用位置编码，或把图模型和序列模型结合。

工程上更稳的流程是：先训练浅层强基线，再做 1 到 4 层消融；如果深层变差，检查 embedding 方差、节点距离、Dirichlet energy 和长尾覆盖；确认过平滑后，再尝试 Residual、LayerNorm、Jumping Knowledge 或 LightGCN 式加权聚合。

---

## 参考资料

1. [A Survey on Oversmoothing in Graph Neural Networks](https://arxiv.org/abs/2303.10993)
2. [A Note on Over-Smoothing for Graph Neural Networks](https://arxiv.org/abs/2006.13318)
3. [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153)
4. [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126)
5. [Residual Connections and Normalization Can Provably Prevent Oversmoothing in GNNs](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6c473e69ba261200dd595d07494c1a73-Abstract-Conference.html)

本文结论来自过平滑相关论文、图信号传播分析和推荐系统工程实践归纳，不是某一个模型的偶然现象。建议阅读顺序是：先看综述理解问题全貌，再看理论论文理解为什么表示会趋同，然后看 SGC 和 LightGCN 理解工程上为什么浅层、简化传播经常有效。
