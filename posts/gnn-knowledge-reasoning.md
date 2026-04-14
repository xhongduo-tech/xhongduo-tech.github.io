## 核心结论

R-GCN 和 CompGCN 都是在做一件事：给知识图谱里的实体学到更适合推理的新向量。知识图谱可以先理解成“节点表示实体，带标签的边表示关系”的图。两者的分歧不在于要不要做消息传递，而在于“关系”参与计算的方式。

R-GCN 的核心做法是：每种关系单独用一套变换矩阵，把不同关系的邻居消息分开处理，再聚合到目标节点上。这样能明确保留“导演于”“出生地”“属于公司”这些关系的语义边界，不容易把不同类型的边混在一起。它适合关系类型不算极端多、或者可以接受做参数约束的场景。

CompGCN 往前走了一步：不只更新实体向量，也让关系向量直接参与卷积，并且区分原向边、反向边、自环三种方向滤波器。直白说，R-GCN 是“先按关系分类再变换”，CompGCN 是“先把实体和关系组合成复合消息，再按方向卷积”。这通常带来更强表达能力和更好的参数效率。

在 FB15k-237 的 link prediction 常见对比里，经典 R-GCN 基线常被引用为 filtered MRR 约 24.8 到 24.9，而 CompGCN 常见结果约 35.5。也就是说，公开文献里更常见的差距接近 10.6 到 10.7 个百分点，而不是 2.1 个百分点。这个差距说明：把关系嵌入显式带进卷积，对多关系知识推理通常是有效的。

| 模型 | 关系如何进入更新 | 是否更新关系嵌入 | 参数规模趋势 | FB15k-237 常见 filtered MRR |
| --- | --- | --- | --- | --- |
| R-GCN | 每种关系对应一个 $W_r$ | 否 | 随关系数上升明显 | $\approx 24.8 \sim 24.9$ |
| CompGCN | 先做 $φ(\text{实体}, \text{关系})$ 组合，再方向卷积 | 是 | 更节制 | $\approx 35.5$ |

简化流程图可以写成：

关系 $r$ → 选择变换/滤波器 → 处理邻居消息 → 聚合到实体  
CompGCN 额外多一步：实体 $x_u$ + 关系 $z_r$ → 组合 $φ(x_u, z_r)$ → 再卷积

---

## 问题定义与边界

问题定义很具体：给定一个多关系知识图谱 $\mathcal{T}=\{(h,r,t)\}$，我们希望学到实体和关系的表示，用于回答“某个头实体 $h$ 是否能通过关系 $r$ 连到某个尾实体 $t$”这类问题。这叫 link prediction，白话解释就是“给缺失三元组补全排序”。

新手版理解可以这样定：图里每条边不只是“连着”，而是带有含义标签。我们的目标不是单纯做邻居平均，而是让节点更新后还能区分“朋友”“同事”“作者”“位于”这些关系的不同作用。

为了描述“同一种关系下的邻居”，通常定义：
$$
N_i^r=\{j \mid (j,r,i)\ \text{或}\ (i,r,j)\ \text{与建图方式一致地属于节点 } i \text{ 的关系 } r \text{ 邻居}\}
$$
其中 $N_i^r$ 表示节点 $i$ 在关系 $r$ 下的邻居集合。归一化系数 $c_{i,r}$ 则用来防止某一类边太多时把消息放大，常见做法是
$$
c_{i,r}=|N_i^r|
$$

这个边界很重要。因为知识图谱任务不是“谁离我近就重要”，而是“哪种关系传来的消息该如何解释”更重要。关系语义边界如果丢了，模型就会把“出生地”和“工作地”当成同一种邻接信号，推理质量会明显下降。

玩具例子：

- 三元组：`(张三, 就职于, OpenAI)`
- 三元组：`(张三, 毕业于, 清华大学)`
- 三元组：`(OpenAI, 位于, 旧金山)`

如果你要更新“张三”的表示，“就职于”和“毕业于”都连到他，但两者含义不同。R-GCN 会为这两种关系各自用不同矩阵；CompGCN 会把“OpenAI + 就职于”和“清华大学 + 毕业于”先做不同组合，再送入不同方向滤波器。

真实工程例子：

在 FB15k-237、WN18RR 这类知识图谱补全任务中，实体多、关系多、测试时要对大量候选尾实体排序。工程目标通常不是解释某一条边，而是在有限显存和训练预算下，把 MRR、Hits@10 这些排序指标做高。此时模型是否能稳定处理多关系语义、是否容易爆参数，直接决定是否可用。

---

## 核心机制与推导

R-GCN 的核心公式是：
$$
h_i^{(l+1)}=\sigma\Big(W_0^{(l)}h_i^{(l)}+\sum_{r\in R}\sum_{j\in N_i^r}\frac{1}{c_{i,r}}W_r^{(l)}h_j^{(l)}\Big)
$$

这里的 $h_i^{(l)}$ 是第 $l$ 层节点表示，$\sigma$ 是激活函数，白话就是“把线性结果再过一层非线性映射”。$W_0$ 处理自环，$W_r$ 处理关系 $r$ 下来的邻居消息。关键点是：每种关系各自有自己的投影矩阵，所以关系不会混用参数。

但这马上带来一个问题。如果关系数是 $|R|$，输入维度是 $d_{in}$，输出维度是 $d_{out}$，那么单层关系参数大约就是
$$
|R|\cdot d_{in}\cdot d_{out}
$$
当关系数上百甚至上千时，参数会快速膨胀。因此 R-GCN 常配合 basis decomposition：
$$
W_r^{(l)}=\sum_{b=1}^{B} a_{rb}^{(l)}V_b^{(l)}
$$
意思是：不直接给每个关系单独学完整矩阵，而是共享 $B$ 个基底矩阵，再给每个关系学组合系数。白话解释就是“先准备几块公共积木，再拼出每个关系自己的矩阵”。

CompGCN 的核心思想是把关系嵌入也纳入卷积：
$$
h_v=f\Big(\sum_{(u,r)\in N(v)}W_{\lambda(r)}\phi(x_u,z_r)\Big)
$$
其中 $x_u$ 是邻居实体表示，$z_r$ 是关系表示，$φ$ 是组合算子，白话就是“把实体和关系揉成一个复合向量”的规则。$W_{\lambda(r)}$ 按方向选择滤波器，通常分成原向、反向、自环三类。关系本身还会更新：
$$
h_r=W_{\text{rel}}z_r
$$

这比 R-GCN 多出的表达力在于：R-GCN 的关系主要体现在“选哪一个矩阵”，CompGCN 的关系直接进入消息内容本身。于是“邻居是谁”和“通过什么关系过来”被绑定得更紧。

新手可以把它看成下面这张文字版图：

实体嵌入 $x_u$  
关系嵌入 $z_r$  
$\downarrow$  
组合 $φ(x_u,z_r)$  
$\downarrow$  
按原向/反向/自环选择 $W_{\lambda(r)}$  
$\downarrow$  
聚合到目标实体  
$\downarrow$  
更新实体表示，同时更新关系表示

常见的 $φ$ 有逐元素乘法、减法、循环相关等。不同组合算子会改变模型偏好。例如逐元素乘法更轻量，循环相关表达更强但实现和调参更复杂。

---

## 代码实现

实现时最容易混淆的是三件事：

1. 节点嵌入和关系嵌入要分开初始化。
2. R-GCN 是 relation-specific weight，CompGCN 是 composition + direction-specific filter。
3. 反向边通常要显式加入，否则模型看不到“从尾到头”的传播。

下面先给一个最小可运行的 Python 玩具实现，用来说明消息传递，不依赖深度学习框架：

```python
import math

def relu(x):
    return max(0.0, x)

def rgcn_one_layer(node_feat, edges, rel_weight, self_weight):
    out = {k: self_weight * v for k, v in node_feat.items()}
    deg = {}
    for src, rel, dst in edges:
        deg[(dst, rel)] = deg.get((dst, rel), 0) + 1

    for src, rel, dst in edges:
        norm = 1.0 / deg[(dst, rel)]
        out[dst] += norm * rel_weight[rel] * node_feat[src]
    return {k: relu(v) for k, v in out.items()}

def comp(x_u, z_r):
    return x_u * z_r

def compgcn_one_layer(node_feat, rel_feat, edges, dir_weight, self_weight):
    out = {k: self_weight * v for k, v in node_feat.items()}
    for src, rel, dst, direction in edges:
        msg = comp(node_feat[src], rel_feat[rel])
        out[dst] += dir_weight[direction] * msg
    return {k: relu(v) for k, v in out.items()}

nodes = {"zhangsan": 1.0, "openai": 2.0, "tsinghua": 3.0}
rels = {"works_at": 0.5, "graduated_from": 1.5}

rgcn_edges = [
    ("openai", "works_at", "zhangsan"),
    ("tsinghua", "graduated_from", "zhangsan"),
]
rgcn_out = rgcn_one_layer(
    node_feat=nodes,
    edges=rgcn_edges,
    rel_weight={"works_at": 2.0, "graduated_from": 1.0},
    self_weight=0.1,
)
assert rgcn_out["zhangsan"] > 0
assert math.isclose(rgcn_out["zhangsan"], 7.1, rel_tol=1e-6)

compgcn_edges = [
    ("openai", "works_at", "zhangsan", "in"),
    ("tsinghua", "graduated_from", "zhangsan", "in"),
]
compgcn_out = compgcn_one_layer(
    node_feat=nodes,
    rel_feat=rels,
    edges=compgcn_edges,
    dir_weight={"in": 2.0, "out": 1.5, "loop": 0.1},
    self_weight=0.1,
)
assert compgcn_out["zhangsan"] > 0
assert math.isclose(compgcn_out["zhangsan"], 11.2, rel_tol=1e-6)
```

这段代码不追求性能，只演示结构差异。R-GCN 按关系选矩阵，CompGCN 先 `comp(node, relation)` 再按方向加权。

如果换成真实工程里的类设计，最简结构通常是：

- `EntityEmbedding`
- `RelationEmbedding`
- `RGCNLayer` 或 `CompGCNLayer`
- `ScoreDecoder`，如 DistMult、ConvE、ConvTransE
- `LinkPredictionLoss`

简化伪代码如下：

```python
for (u, r, v) in edges:
    comp_msg = combine(x[u], z[r])      # CompGCN
    out[v] += W_dir[direction(r)] @ comp_msg

for r in relations:                     # R-GCN 常见视角
    for (u, v) in edges_of_relation[r]:
        out[v] += W_rel[r] @ x[u]

x = activation(normalize(out + self_loop))
z = W_rel_update @ z
```

组合算子可以先这样理解：

| $φ(x_u, z_r)$ | 含义 | 参数影响 |
| --- | --- | --- |
| Hadamard 乘法 $x_u \odot z_r$ | 逐元素相互调制 | 最轻量 |
| 减法 $x_u - z_r$ | 强调差异 | 轻量 |
| 循环相关 | 建模更复杂交互 | 计算更重 |
| 旋转式组合 | 适合方向和相位语义 | 对实现要求更高 |

---

## 工程权衡与常见坑

R-GCN 最典型的问题是爆参数。假设有 237 种关系、隐藏维度 200，那么单层关系矩阵规模就是 $237 \times 200 \times 200$，这还没算自环、偏置和解码器。关系数一多，显存、训练时间、过拟合风险都会上升。所以真实项目里几乎不会裸用“每个关系一个完整矩阵”的版本，而会加 basis 或 block decomposition。

CompGCN 虽然更省参数，但不是没有代价。它的性能更依赖组合算子 $φ$、关系初始化以及方向边构造。如果关系嵌入初始化太差，或者反向边漏加，消息质量会直接下降。还有一个经常被忽略的点：CompGCN 编码器好，不代表任何解码器都能自动变强。论文中的高分通常和特定 decoder 搭配出现，不能把“编码器收益”与“整体系统收益”完全混为一谈。

下面这个表更适合工程判断：

| 维度 | R-GCN | CompGCN |
| --- | --- | --- |
| 参数风险 | 高，关系多时明显 | 中，通常更可控 |
| 过拟合风险 | 较高 | 中等 |
| 对关系语义建模 | 强 | 更强 |
| 对实现复杂度要求 | 中 | 较高 |
| 常见稳定化手段 | basis、dropout、L2 | dropout、边方向校验、组合算子搜索 |

常见坑可以直接列成检查单：

- 关系数多时没做 basis decomposition。
- 忘了使用 $c_{i,r}$ 做归一化，导致高频关系主导训练。
- 只建原向边，没建反向边和自环。
- 把实体嵌入和关系嵌入共用同一初始化逻辑，导致训练初期不稳定。
- 只比较 encoder，不固定 decoder 和负采样策略，结论失真。
- 拿不同论文中的 MRR 直接横比，却忽略 filtered/raw、数据切分、解码器差异。

真实工程例子：

如果你在一个企业知识图谱里做“设备 A 是否属于产线 B”“工程师 C 是否负责工单 D”的补全，关系类型可能很快到上百种。此时 R-GCN 往往先面临显存和过拟合问题；CompGCN 则更像一个折中方案，能在保留关系建模能力的同时，把参数控制在更现实的范围内。

---

## 替代方案与适用边界

如果任务只是做很大规模的知识图谱补全，且关系极多、样本稀疏，直接上图卷积不一定是第一选择。纯 KGE 模型如 TransE、RotatE、ComplEx 往往更便宜，吞吐更高。它们的缺点是对局部图结构的显式利用较弱。

一个实用判断逻辑可以写成：

任务主要依赖局部结构传播 → 优先看 R-GCN / CompGCN  
任务关系很多且预算紧 → 先看轻量 KGE 或 R-GCN+basis  
任务要求实体和关系共同演化 → 优先看 CompGCN  
任务更强调路径搜索或规则推理 → 把 R-GCN / CompGCN 当编码器，再接路径推理模块

对零基础到初级工程师，最值得记住的不是“哪个模型永远最好”，而是“模型强项对应什么任务”：

| 方案 | 适用边界 | 参数特征 | 适配任务 |
| --- | --- | --- | --- |
| Simple GCN | 忽略关系标签也能接受时 | 最省 | 普通同质图 |
| R-GCN | 关系明确且数量可控 | 偏大 | 实体分类、基础 KG 推理 |
| CompGCN | 关系必须强表达且想控制参数 | 较优 | Link prediction、多关系推理 |
| 纯 KGE | 超大图、追求吞吐 | 较省 | 大规模补全、召回基线 |

如果你要做路径强化学习类系统，例如把图编码器接到策略网络前面，那么 R-GCN 和 CompGCN 常被当作结构编码器基线。此时它们不是终点，而是“为后续决策提供状态表示”的前端模块。路径任务更关心“状态表达是否足够区分下一步动作”，CompGCN 往往更有优势，但实现和调试成本也更高。

---

## 参考资料

- Schlichtkrull et al., *Modeling Relational Data with Graph Convolutional Networks*  
  链接：https://arxiv.org/abs/1703.06103  
  阅读重点：R-GCN 的传播公式、basis decomposition、参数爆炸问题。对应正文中的机制推导与工程权衡。

- DGL 官方教程，*Relational Graph Convolutional Network*  
  链接：https://www.dgl.ai/dgl_docs/tutorials/models/1_gnn/4_rgcn.html  
  阅读重点：R-GCN 的实现结构、relation-specific linear layer、消息函数与聚合流程。对应正文中的代码实现部分。

- Vashishth et al., *Composition-based Multi-Relational Graph Convolutional Networks*  
  链接：https://openreview.net/forum?id=BylA_C4tPr  
  阅读重点：CompGCN 的组合算子 $φ$、方向滤波器、关系嵌入联合更新。对应正文中的核心机制与替代方案。

- Applied Sciences 2026, *Path and Structural Features Enhanced Reinforcement Learning for Knowledge Graph Completion*  
  链接：https://www.mdpi.com/2076-3417/16/7/3460  
  阅读重点：把结构编码器放进路径推理系统的工程上下文。对应正文中的真实工程例子与适用边界。

- CrossE 结果表中常引用的 FB15k-237 对比  
  链接：https://wencolani.github.io/presentation/2019-WSDM-CrossE.pdf  
  阅读重点：R-GCN 在 FB15k-237 上的常见 filtered MRR 引用值 24.8。对应正文中的实验对比基线。
