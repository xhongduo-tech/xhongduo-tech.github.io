## 核心结论

异构图神经网络是面向“多类型节点 + 多类型边”的图表示学习模型。表示学习可以先理解成“把对象压缩成可计算的向量表示”，后续分类、检索、推荐都依赖这个表示。它解决的问题不是“把图做得更大”，而是“把不同关系的信息分开建模，再按目标类型汇总”。

普通 GNN 往往默认所有边语义一致，只关心“谁和谁相连”。异构 GNN 会区分“作者写论文”和“论文引用论文”这两种关系，因为它们传递的信息价值不同。新手可以把它理解为：普通 GNN 把所有连线都当同一种信号，异构 GNN 会先问“这根线代表什么关系”，再决定如何传递信息。

一层异构消息传递通常写成：

$$
m_{u\to v}^{(l,r)}=\phi_r(h_u^{(l)},h_v^{(l)},x_{uv})
$$

$$
h_v^{(l+1)}=\psi_{t_v}\Big(h_v^{(l)},\operatorname{AGG}_{r:\,dst(r)=t_v}\sum_{u\in N_r(v)} m_{u\to v}^{(l,r)}\Big)
$$

这里 $r$ 是关系类型，$t_v$ 是节点 $v$ 的类型，$N_r(v)$ 是在关系 $r$ 下指向 $v$ 的邻居集合。核心含义只有两步：先按关系算消息，再按目标类型聚合更新。

| 对比项 | 普通 GNN | 异构 GNN |
| --- | --- | --- |
| 输入 | 单一节点类型、单一边类型为主 | 多节点类型、多边类型 |
| 参数共享 | 大多全图共享 | 常按关系或类型区分 |
| 聚合方式 | 邻居统一聚合 | 先分关系，再按目标类型聚合 |
| 适用任务 | 同构节点分类、同构链路预测 | 知识图谱、学术图、推荐、风控 |
| 风险点 | 关系语义被混淆 | 参数量、采样复杂度、工程口径更难控 |

如果只记一个实现思路，可以记下面这段极简伪代码：

```python
for relation in edge_types:
    src_type, rel, dst_type = relation
    msg = relation_transform[rel](x[src_type], edge_index[relation])
    inbox[dst_type].append(msg)

for dst_type in node_types:
    agg = aggregate(inbox[dst_type])      # sum / mean / attention
    x[dst_type] = update[dst_type](x[dst_type], agg)
```

---

## 问题定义与边界

异构图先看输入结构。图里不只有一种“点”和一种“边”，而是有明确 schema。schema 可以先理解成“数据结构说明书”。例如学术图里可以有 `author / paper / venue / institution` 四类节点，边可以有 `writes / cites / published_in / affiliated_with` 四类关系。白话讲，就是图里有不同角色的人和事，每种线代表不同关系。

更正式一点，可以把输入写成：

- 节点集合按类型拆分：$\mathcal{V}=\cup_t \mathcal{V}_t$
- 边集合按关系拆分：$\mathcal{E}=\cup_r \mathcal{E}_r$
- 节点 $v$ 的类型记为 $t_v$
- 关系 $r$ 的一条边记为 $(u,v,r)$
- 节点特征记为 $x_v$，边特征记为 $x_{uv}$

这类模型的输出不固定，常见有三种：
- 目标节点类型的分类结果，比如论文主题分类
- 目标边的打分，比如用户是否会点击商品
- 检索或推荐排序分数，比如 query-item 相关性

玩具例子可以先看一个最小学术图：

| 项目 | 内容 |
| --- | --- |
| 节点类型 | `author`, `paper`, `venue`, `institution` |
| 边类型 | `writes`, `cites`, `published_in`, `affiliated_with` |
| 特征来源 | 作者画像、论文文本 embedding、会议级别、机构属性 |
| 预测目标 | 论文分类、作者合作预测、论文引用预测 |
| 训练标签来源 | 人工标注、历史关系、公开数据库 |

代码层面通常先定义 schema，再装数据：

```python
node_types = ["author", "paper", "venue", "institution"]
edge_types = [
    ("author", "writes", "paper"),
    ("paper", "cites", "paper"),
    ("paper", "published_in", "venue"),
    ("author", "affiliated_with", "institution"),
]

hetero_data = {
    "author": {"x": "..."},
    "paper": {"x": "...", "y": "..."},
    ("author", "writes", "paper"): {"edge_index": "..."},
    ("paper", "cites", "paper"): {"edge_index": "..."},
}
```

边界也要说清楚。异构 GNN 不是默认最优方案。以下场景未必值得上：
- 关系语义单一，直接同构 GNN 就够用
- 标签很少且图极稀疏，复杂模型容易只学到噪声
- 线上延迟、显存、可解释性约束很强，特征模型更稳
- 节点类型多但有效特征弱，最后只是把复杂度堆高

判断标准不是“图里有多种对象”，而是“不同关系是否真的需要区别对待，且这种区别能带来稳定收益”。

---

## 核心机制与推导

异构 GNN 的主线很统一：消息函数按关系区分，更新函数按目标类型区分。消息函数可以理解成“邻居怎么影响我”，更新函数可以理解成“我如何吸收这些影响”。

先看公式中的符号含义：

| 符号 | 含义 | 作用 |
| --- | --- | --- |
| $h_u^{(l)}$ | 第 $l$ 层节点表示 | 当前层输入特征 |
| $r$ | 关系类型 | 决定用哪套参数 |
| $x_{uv}$ | 边特征 | 表示交互强度、时间、权重等 |
| $\phi_r$ | 关系级变换 | 对不同关系分别建模 |
| $N_r(v)$ | 关系 $r$ 下指向 $v$ 的邻居 | 限定消息来源 |
| $\operatorname{AGG}$ | 聚合函数 | 合并多关系消息 |
| $\psi_{t_v}$ | 类型级更新 | 按目标类型更新表示 |

最小数值例子最能说明“为什么不能混着算”。设目标节点是论文 $P_1$，初始表示 $h_{P_1}^{(0)}=3$。它收到两类消息：

- 作者 $A_0$ 通过 `writes` 关系传来特征 $1$，关系权重 $W_{writes}=2$，得到消息 $2\times1=2$
- 论文 $P_0$ 通过 `cites` 关系传来特征 $2$，关系权重 $W_{cites}=1$，得到消息 $1\times2=2$
- 自环项可以理解为“保留自己原有信息”，设系数为 $0.5$

于是：

$$
h_{P_1}^{(1)} = 2 + 2 + 0.5\times 3 = 5.5
$$

这个玩具例子说明，同一个节点收到的消息不一样，要先分别算，再合并，不能混成一团。

从这里可以一路推到几类常见模型：

1. 最简单的关系加权求和  
   每种关系一套线性变换，最后求和。

2. R-GCN  
   全称 Relational Graph Convolutional Network，可以理解成“给每种关系单独卷积核，但做参数共享以防参数爆炸”。

3. HGT  
   全称 Heterogeneous Graph Transformer，可以理解成“在异构图上做类型相关的注意力机制”。注意力可以先理解成“对不同邻居分配不同权重”。

高层伪代码如下：

```python
for r in relations:
    src_type, rel, dst_type = r
    h_src = x[src_type]
    edges = edge_index[r]

    # R-GCN 风格：关系专属线性变换
    msg_r = W[rel] @ gather_neighbors(h_src, edges)

    # HGT 风格会在这里加入 type-specific Q/K/V 和 attention
    relation_msgs[dst_type].append(msg_r)

for dst_type in node_types:
    merged = sum(relation_msgs[dst_type])
    x[dst_type] = update_fn[dst_type](x[dst_type], merged)
```

共同主线不是名字，而是“关系专属参数”。如果没有这条主线，异构图的关系差异就会在第一步被抹平。

真实工程例子可以看电商推荐。节点类型有 `user / item / brand / category / query`，边类型有 `click / cart / buy / co-view / belong-to`。`click` 和 `buy` 虽然都是用户到商品的边，但业务价值不同，训练时一般不会共用完全相同的消息变换，否则模型会把“看过”和“买过”的信号混在一起。

---

## 代码实现

落到工程里，异构图代码至少要回答四个问题：输入数据怎么装、模型按什么关系传播、只对谁算损失、评估口径是什么。很多失败案例不是模型层写错，而是这四件事没有讲清楚。

下面给一个可运行的 Python 玩具实现。它不依赖深度学习框架，只演示“按关系循环，再按目标类型聚合”的核心逻辑，并用 `assert` 固定数值结果。

```python
from collections import defaultdict

def hetero_message_passing(node_features, edges, relation_weight, self_loop_weight):
    inbox = defaultdict(float)
    for src, rel, dst in edges:
        inbox[dst] += relation_weight[rel] * node_features[src]
    out = {}
    for node, value in node_features.items():
        out[node] = inbox[node] + self_loop_weight.get(node, 0.0) * value
    return out

node_features = {
    "A0": 1.0,   # author
    "P0": 2.0,   # paper
    "P1": 3.0,   # target paper
}

edges = [
    ("A0", "writes", "P1"),
    ("P0", "cites", "P1"),
]

relation_weight = {
    "writes": 2.0,
    "cites": 1.0,
}

self_loop_weight = {
    "A0": 0.5,
    "P0": 0.5,
    "P1": 0.5,
}

updated = hetero_message_passing(node_features, edges, relation_weight, self_loop_weight)

assert abs(updated["P1"] - 5.5) < 1e-9
assert abs(updated["A0"] - 0.5) < 1e-9
assert abs(updated["P0"] - 1.0) < 1e-9
print(updated)
```

如果换成 PyG 或 DGL，代码会更像“容器 + 关系卷积层 + 目标头”。新手可以把它理解成：先把不同类型的点和边装进容器，再让模型分别处理不同关系，最后只对目标任务输出打分。

```python
# 伪代码：PyG / DGL 风格
data["author"].x = author_x
data["paper"].x = paper_x
data["paper"].y = paper_label
data["author", "writes", "paper"].edge_index = writes_edge_index
data["paper", "cites", "paper"].edge_index = cites_edge_index

class HeteroModel:
    def __init__(self):
        self.conv1 = HeteroConv({
            ("author", "writes", "paper"): RelationConv(...),
            ("paper", "cites", "paper"): RelationConv(...),
        })
        self.cls_head = Linear(...)

    def forward(self, data):
        x_dict = self.conv1(data.x_dict, data.edge_index_dict)
        logits = self.cls_head(x_dict["paper"])
        return logits

logits = model(data)
loss = cross_entropy(logits[train_mask], data["paper"].y[train_mask])
loss.backward()
optimizer.step()
```

常见框架概念对应关系如下：

| 目标 | DGL | PyG |
| --- | --- | --- |
| 异构图容器 | `DGLHeteroGraph` | `HeteroData` |
| 关系级消息传递 | relation-specific module | `HeteroConv` 中按 relation 注册 |
| 关系图卷积 | `HeteroGraphConv` / `RelGraphConv` | `RGCNConv` / `HGTConv` / 自定义 relation conv |
| 数据访问 | `g.nodes['type']` / `g.edges['rel']` | `data['type']` / `data[src, rel, dst]` |

损失函数也要和任务对齐。节点分类常用交叉熵：

$$
\mathcal{L}_{cls} = -\sum_{i \in \mathcal{Y}} y_i \log \hat{y}_i
$$

链路预测常用打分函数配合负采样，例如：

$$
s(u,v)=h_u^\top h_v,\qquad
\mathcal{L}_{link}=-\log \sigma(s(u,v^+))-\log \sigma(-s(u,v^-))
$$

如果不明确“目标节点类型、采样方式、损失函数、指标”，代码即使能跑，也很难复现实验，更难上线。

---

## 工程权衡与常见坑

异构 GNN 放进完整工程流程里，重点不是“用了 HGT 还是 R-GCN”，而是输入输出口径、评测规则、回归风险和可回滚条件。模型只是中间一段。

先看最常见的问题表：

| 问题现象 | 根因 | 规避方式 | 上线检查项 |
| --- | --- | --- | --- |
| 离线 AUC 很高，上线掉得快 | 时间泄漏，训练图混入未来边 | 严格按时间切分构图 | 未来时间戳边数必须为 0 |
| 某些关系几乎不起作用 | 关系极不均衡 | 按关系采样、关系归一化、参数共享 | 各关系 batch 覆盖率 |
| 某类节点效果异常差 | 类型缺特征 | 补类型 embedding 或结构特征 | 缺失特征比例阈值 |
| 训练稳定，上线波动大 | 采样分布与线上不一致 | 固定 sampler 配置，做口径对齐 | 线上线下分布偏差 |
| 延迟超 SLA | 关系过多、邻居采样过深 | 限制 hop、缓存 embedding、裁剪 schema | p95 延迟、显存峰值 |
| 线上指标连续下滑 | 模型回归或特征漂移 | 保留旧模型热切换 | 明确回滚开关和阈值 |

推荐系统里有一个典型坑：训练集里混入未来点击边。比如你在 4 月 1 日训练用于预测 4 月 2 日点击的模型，却把 4 月 3 日发生的点击边也加进图里。这样离线 AUC 会虚高，因为模型不是学会了预测，而是偷看了答案。

很多关系还需要单独归一化。因为不同关系的边数差异可能是数量级级别的，若直接求和，热门关系会吞掉梯度。常见做法是关系内归一化：

$$
\tilde{m}_{u\to v}^{(r)}=\frac{1}{c_{v,r}}\,m_{u\to v}^{(r)}
$$

其中 $c_{v,r}$ 可以取关系 $r$ 下目标节点 $v$ 的邻居数，目的是让不同关系的贡献处在可比较范围内。

下面这段数据检查代码比“换个更复杂模型”更重要：

```python
def check_time_split(edges, train_end_ts):
    leaked = [e for e in edges if e["timestamp"] > train_end_ts]
    assert len(leaked) == 0, f"time leakage: {len(leaked)} future edges found"

def check_relation_coverage(edges, all_relations, min_ratio=0.01):
    total = len(edges)
    counts = {r: 0 for r in all_relations}
    for e in edges:
        counts[e["relation"]] += 1
    ratios = {r: counts[r] / total for r in all_relations}
    assert all(v >= min_ratio for v in ratios.values()), ratios

def check_negative_sampling(pos_edges, neg_edges):
    pos_pairs = {(e["src"], e["dst"]) for e in pos_edges}
    neg_pairs = {(e["src"], e["dst"]) for e in neg_edges}
    assert pos_pairs.isdisjoint(neg_pairs), "negative samples overlap with positives"

# 示例
edges = [
    {"src": 1, "dst": 2, "relation": "click", "timestamp": 10},
    {"src": 2, "dst": 3, "relation": "buy", "timestamp": 9},
]
check_time_split(edges, train_end_ts=10)
check_relation_coverage(edges, all_relations=["click", "buy"], min_ratio=0.0)
check_negative_sampling(
    pos_edges=[{"src": 1, "dst": 2}],
    neg_edges=[{"src": 2, "dst": 1}],
)
```

上线门槛也要提前写死，不能靠感觉。一个可执行的门槛通常包括：
- 离线主指标不低于当前基线
- 核心节点类型覆盖率不下降
- p95 延迟、显存、错误率过阈值自动回滚
- Shadow 流量验证通过后再放量

---

## 替代方案与适用边界

不是所有图任务都需要异构 GNN。复杂结构只有在“结构复杂度真的能转化成收益”时才值得付成本。否则更简单的方法更稳。

三个常见场景可以直接对比：

1. 只有单一类型节点和边  
   例如社交好友推荐，用户和用户之间只有一种关系，同构 GNN 通常更直接。

2. 关系类型很少且业务解释强  
   例如知识图谱里只关心几条元路径。元路径可以理解成“预先规定的关系链路模板”，如 `Author -> Paper -> Venue`。这时基于元路径的聚合或规则系统可能更可控。

3. 标签稀少但关系复杂  
   例如多角色推荐、风控网络、学术图检索，异构 GNN 更有价值，因为它能利用关系结构补标签不足。

| 方法 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 特征模型 | 简单、稳定、易上线 | 难利用高阶图结构 | 强特征、低延迟场景 |
| 同构 GNN | 实现成熟 | 关系语义容易混淆 | 单一关系图 |
| 元路径方法 | 可解释性强 | 依赖人工设计路径 | 关系少、业务规则清晰 |
| 矩阵分解 | 适合稀疏交互 | 难融入复杂多类型结构 | 经典推荐基线 |
| 异构 GNN | 结构表达力强 | 工程复杂、训练成本高 | 多类型多关系任务 |

元路径方法和异构 GNN 的差别，可以简单写成：

$$
h_v = \operatorname{AGG}_{p \in \mathcal{P}} f_p(v)
$$

这里 $\mathcal{P}$ 是人工定义的元路径集合，$f_p(v)$ 表示沿某条路径抽取出的特征。它本质上更像“先指定哪些关系链有用，再做聚合”。异构 GNN 则是把“关系如何组合”更多交给训练过程学习。

做方案选择时，可以用一个简单决策逻辑：

```python
def choose_model(num_node_types, num_edge_types, label_density, latency_budget_ms):
    if num_node_types == 1 and num_edge_types == 1:
        return "homogeneous_gnn"
    if num_edge_types <= 3 and latency_budget_ms < 20:
        return "feature_model_or_metapath"
    if label_density < 0.01 and num_edge_types >= 4:
        return "heterogeneous_gnn"
    return "start_from_simple_baseline"
```

工程上更稳的做法不是直接问“能不能上异构 GNN”，而是先问：
- 现有特征模型的瓶颈是不是结构信息缺失
- 不同关系是否真有不同预测价值
- 离线收益能否覆盖训练、推理、维护成本
- 出问题时能否快速回滚

---

## 参考资料

下表给出一个适合新手的阅读顺序，先学接口，再学评测，最后学模型来源。

| 来源 | 用途 | 建议顺序 |
| --- | --- | --- |
| DGL / PyG 异构图文档 | 理解数据容器和消息传递接口 | 1 |
| OGB 评测文档 | 明确节点分类、链路预测评测口径 | 2 |
| R-GCN / HGT 论文 | 理解关系专属参数与模型演进 | 3 |

最小复现实验清单可以按下面执行：

```text
1. 明确图 schema：节点类型、边类型、目标任务
2. 做时间切分，冻结未来信息
3. 统计类型覆盖率、缺失特征率、关系分布
4. 训练简单基线：特征模型 / 同构 GNN
5. 训练异构 GNN：固定 sampler、固定指标
6. 比较离线收益、延迟、显存、覆盖率
7. 导出模型并准备回滚开关
```

1. [DGL Heterogeneous Graphs](https://www.dgl.ai/dgl_docs/en/2.2.x/guide/graph-heterogeneous.html)
2. [PyG Heterogeneous Graph Learning](https://pytorch-geometric.readthedocs.io/en/stable/notes/heterogeneous.html)
3. [OGB Link Property Prediction](https://ogb.stanford.edu/docs/linkprop/)
4. [Modeling Relational Data with Graph Convolutional Networks (R-GCN)](https://www.microsoft.com/en-us/research/publication/modeling-relational-data-with-graph-convolutional-networks/)
5. [Heterogeneous Graph Transformer (HGT)](https://www.microsoft.com/en-us/research/?p=643494)
