## 核心结论

TransE 是知识图谱嵌入里的经典入门模型。知识图谱嵌入，白话说，就是把“实体”和“关系”变成可以计算的向量。它的核心假设很直接：如果三元组 $(h,r,t)$ 是真事实，那么头实体向量 $h$ 加上关系向量 $r$，应该接近尾实体向量 $t$，也就是

$$
h + r \approx t
$$

因此，TransE 常用下面的评分函数衡量三元组是否可信：

$$
f(h,r,t)=\|h+r-t\|_1 \quad \text{或} \quad f(h,r,t)=\|h+r-t\|_2
$$

分数越小，说明这个三元组越像真事实。

它的优点是结构极简、参数少、训练快、容易实现，所以非常适合做知识图谱补全的 baseline。baseline，白话说，就是先用一个简单可靠的起点验证数据、训练流程和评估指标是否工作正常。它的缺点也很明确：平移假设对一对多、多对一、多对多以及对称关系的表达能力弱，容易在复杂关系上失真。

先看一个玩具例子。设：

- $h=(1,0)$
- $r=(0,1)$
- $t=(1,1)$

则 $h+r=(1,1)=t$，所以分数为 0，这表示它是一个非常理想的真三元组。若把尾实体改成 $t'=(2,1)$，则误差变成 1，可信度更低。

可以先用一张表建立整体印象：

| 维度 | TransE 的表现 |
|---|---|
| 核心思想 | 关系是向量平移 |
| 打分方向 | 分数越小越可信 |
| 训练成本 | 低 |
| 参数规模 | 小 |
| 可解释性 | 强，几何直观明显 |
| 擅长关系 | 一对一、模式较规整的关系 |
| 薄弱点 | 1-N、N-1、N-N、对称关系 |

---

## 问题定义与边界

知识图谱通常写成三元组集合：

$$
G \subseteq E \times R \times E
$$

其中：

- $E$ 是实体集合，白话说，就是“人、地点、组织、概念”这些对象。
- $R$ 是关系集合，白话说，就是“位于、作者、父子、属于”这些连接方式。
- $G$ 是已知事实集合，白话说，就是数据库里当前认为成立的三元组。

一个三元组写成 $(h,r,t)$：

| 符号 | 名称 | 白话解释 |
|---|---|---|
| $h$ | 头实体 | 关系的起点 |
| $r$ | 关系 | 从头到尾的连接方式 |
| $t$ | 尾实体 | 关系的终点 |

例如：

- $(\text{巴黎}, \text{capital\_of}, \text{法国})$ 更像真事实
- $(\text{巴黎}, \text{capital\_of}, \text{日本})$ 明显不像真事实

TransE 要做的不是“逻辑证明”，而是“几何打分”。逻辑证明，白话说，是基于明确规则推出结论；几何打分则是把事实变成向量距离问题。它并不会显式写出“如果 A 在 B，B 在 C，那么 A 在 C”这类规则，而是通过向量空间中的相对位置学习哪些三元组更可信。

这决定了 TransE 的边界：

| 适合的问题 | 不适合的问题 |
|---|---|
| 知识图谱补全 | 严格符号逻辑推理 |
| 候选实体排序 | 需要精确可解释规则链的场景 |
| 快速建立 baseline | 强依赖复杂关系模式的场景 |
| 数据管线和评估验证 | 仅靠平移无法表达的关系 |

为什么说它对复杂关系有边界？因为同一个关系向量 $r$ 对所有三元组共享。如果某个头实体对应很多尾实体，例如“国家-拥有城市”，那么同一个 $h+r$ 要同时贴近多个不同的 $t$，这在几何上天然冲突。向量可以靠近一片区域，但很难同时等于很多彼此分散的点。

---

## 核心机制与推导

TransE 的机制可以拆成三个部分：嵌入、打分、训练。

第一步是嵌入。嵌入，白话说，就是给每个实体和关系分配一个可训练的向量。若嵌入维度为 $d$，则：

- 每个实体 $e \in E$ 对应向量 $\mathbf{e} \in \mathbb{R}^d$
- 每个关系 $r \in R$ 对应向量 $\mathbf{r} \in \mathbb{R}^d$

第二步是打分。TransE 用距离来定义真伪程度：

$$
f(h,r,t)=\| \mathbf{h}+\mathbf{r}-\mathbf{t} \|_1
$$

或者

$$
f(h,r,t)=\| \mathbf{h}+\mathbf{r}-\mathbf{t} \|_2
$$

其中：

- $\|\cdot\|_1$ 是 L1 距离，白话说，就是各维绝对值之和。
- $\|\cdot\|_2$ 是 L2 距离，白话说，就是欧氏距离。

分数越小，表示 $\mathbf{h}+\mathbf{r}$ 越接近 $\mathbf{t}$，三元组越可信。

继续看玩具例子。设：

- $\mathbf{h}=(1,0)$
- $\mathbf{r}=(0,1)$
- $\mathbf{t}=(1,1)$
- $\mathbf{t}'=(2,1)$

则：

$$
f(h,r,t)=\|(1,0)+(0,1)-(1,1)\|_1=0
$$

而负样本 $(h,r,t')$ 的分数是：

$$
f(h,r,t')=\|(1,0)+(0,1)-(2,1)\|_1=1
$$

这里的负样本，白话说，就是人为构造的“假事实”，用来告诉模型什么是不对的。最常见的构造方式是替换头实体或尾实体：

- 替换头：$(h',r,t)$
- 替换尾：$(h,r,t')$

第三步是训练。训练目标不是单独让正例分数小，而是要求正例比分负例更小，且至少小一个间隔 $\gamma$。常见损失函数是 margin ranking loss：

$$
L = \max(0,\gamma + f(h,r,t) - f(h',r,t'))
$$

这里：

- $\gamma$ 是 margin，白话说，就是希望正例至少比负例好多少。
- 如果正例已经足够优于负例，损失就是 0。
- 如果差距不够，模型就继续更新参数。

这个设计很重要，因为它不是在追求“正例绝对分数一定是某个值”，而是在追求“相对排序正确”。知识图谱补全最终通常也是排序问题，例如给定 $(h,r,?)$，从所有实体里找最可能的尾实体。

再看真实工程里的含义。假设企业知识图谱里有如下事实：

- $(\text{杭州阿里云园区}, \text{located\_in}, \text{杭州})$
- $(\text{杭州}, \text{located\_in}, \text{浙江})$
- $(\text{浙江}, \text{located\_in}, \text{中国})$

如果 `located_in` 的模式较稳定，TransE 往往能较快学到这个关系对应的向量平移，用于做候选城市、省份、国家排序。但如果关系是：

- $(\text{中国}, \text{has\_city}, \text{北京})$
- $(\text{中国}, \text{has\_city}, \text{上海})$
- $(\text{中国}, \text{has\_city}, \text{杭州})$

那么同一个 $\mathbf{h}+\mathbf{r}$ 要接近很多不同城市向量，表达上就开始吃力。

为什么训练时常做实体归一化？原始论文里通常要求实体向量满足：

$$
\|\mathbf{e}\|_2 = 1
$$

归一化，白话说，就是把向量长度固定到一个稳定范围。这样做主要有两个作用：

1. 防止模型通过无限放大或缩小向量范数来“投机”降低损失。
2. 让训练更稳定，避免不同实体向量尺度漂移过大。

可以把机制总结成下表：

| 步骤 | 数学对象 | 作用 |
|---|---|---|
| 嵌入 | $\mathbf{e}, \mathbf{r}$ | 把符号对象变成可训练向量 |
| 打分 | $f(h,r,t)$ | 判断三元组可信度 |
| 负采样 | $(h',r,t)$ 或 $(h,r,t')$ | 构造对比样本 |
| 排序损失 | $\max(0,\gamma+f^+-f^-)$ | 让正例优于负例 |
| 归一化 | $\|\mathbf{e}\|_2=1$ | 稳定训练、防止范数作弊 |

---

## 代码实现

下面给一个最小可运行的 Python 实现。它不依赖深度学习框架，只用 `numpy` 演示 TransE 的核心流程：初始化、打分、负采样、损失、梯度更新、实体归一化。代码里的 `assert` 用来保证核心行为正确。

```python
import numpy as np

np.random.seed(7)

def l1_score(h, r, t):
    return np.abs(h + r - t).sum()

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# 一个最小知识图谱
entities = ["Paris", "France", "Japan", "Berlin", "Germany"]
relations = ["capital_of"]

eid = {e: i for i, e in enumerate(entities)}
rid = {r: i for i, r in enumerate(relations)}

triples = [
    ("Paris", "capital_of", "France"),
    ("Berlin", "capital_of", "Germany"),
]

dim = 8
gamma = 1.0
lr = 0.05

entity_emb = np.random.randn(len(entities), dim) * 0.1
relation_emb = np.random.randn(len(relations), dim) * 0.1

# 按论文习惯，对实体做 L2 归一化
for i in range(len(entities)):
    entity_emb[i] = normalize(entity_emb[i])

def score(triple):
    h, r, t = triple
    return l1_score(entity_emb[eid[h]], relation_emb[rid[r]], entity_emb[eid[t]])

def corrupt_tail(triple):
    h, r, t = triple
    # 简化版负采样：随机替换尾实体，且不等于原尾实体
    candidates = [e for e in entities if e != t]
    t_neg = np.random.choice(candidates)
    return (h, r, t_neg)

def train_step(triple):
    global entity_emb, relation_emb

    pos = triple
    neg = corrupt_tail(triple)

    h, r, t = pos
    h_neg, r_neg, t_neg = neg

    h_idx, r_idx, t_idx = eid[h], rid[r], eid[t]
    h2_idx, r2_idx, t2_idx = eid[h_neg], rid[r_neg], eid[t_neg]

    h_vec = entity_emb[h_idx].copy()
    r_vec = relation_emb[r_idx].copy()
    t_vec = entity_emb[t_idx].copy()

    h_neg_vec = entity_emb[h2_idx].copy()
    r_neg_vec = relation_emb[r2_idx].copy()
    t_neg_vec = entity_emb[t2_idx].copy()

    pos_diff = h_vec + r_vec - t_vec
    neg_diff = h_neg_vec + r_neg_vec - t_neg_vec

    pos_score = np.abs(pos_diff).sum()
    neg_score = np.abs(neg_diff).sum()

    loss = max(0.0, gamma + pos_score - neg_score)

    if loss > 0:
        # L1 距离的次梯度：sign
        pos_grad = np.sign(pos_diff)
        neg_grad = np.sign(neg_diff)

        # 正例：拉近 h + r 和 t
        entity_emb[h_idx] -= lr * pos_grad
        relation_emb[r_idx] -= lr * pos_grad
        entity_emb[t_idx] += lr * pos_grad

        # 负例：推远 h' + r 和 t'
        entity_emb[h2_idx] += lr * neg_grad
        relation_emb[r2_idx] += lr * neg_grad
        entity_emb[t2_idx] -= lr * neg_grad

        # 实体归一化
        entity_emb[h_idx] = normalize(entity_emb[h_idx])
        entity_emb[t_idx] = normalize(entity_emb[t_idx])
        entity_emb[h2_idx] = normalize(entity_emb[h2_idx])
        entity_emb[t2_idx] = normalize(entity_emb[t2_idx])

    return pos_score, neg_score, loss

# 先检查玩具公式是否成立
h = np.array([1.0, 0.0])
r = np.array([0.0, 1.0])
t = np.array([1.0, 1.0])
t_bad = np.array([2.0, 1.0])

assert l1_score(h, r, t) == 0.0
assert l1_score(h, r, t_bad) == 1.0

before_true = score(("Paris", "capital_of", "France"))
before_false = score(("Paris", "capital_of", "Japan"))

for epoch in range(400):
    triple = triples[epoch % len(triples)]
    train_step(triple)

after_true = score(("Paris", "capital_of", "France"))
after_false = score(("Paris", "capital_of", "Japan"))

print("before_true:", round(before_true, 4))
print("before_false:", round(before_false, 4))
print("after_true:", round(after_true, 4))
print("after_false:", round(after_false, 4))

# 正例分数应当倾向下降，且一般应优于明显假例
assert after_true < before_true
assert after_true < after_false
```

这段代码体现了几个关键实现点。

第一，分数方向必须一致。TransE 是“距离越小越真”，不是“分数越大越真”。因此损失应写成：

$$
\max(0,\gamma + f_{\text{pos}} - f_{\text{neg}})
$$

而不是反过来。

第二，负采样不能太随意。上面的实现是最简版，只替换尾实体，而且没有做“过滤已知真值”。过滤，白话说，就是如果替换后恰好变成另一个真实三元组，就不能把它当负例。真实工程中通常需要维护一个真三元组集合，采样后检查是否冲突。

第三，归一化不能忘。即便用 Adam 或其他优化器，实体向量不做约束也经常导致训练发散或效果飘忽。

可以把训练流程整理成步骤表：

| 步骤 | 内容 |
|---|---|
| 1 | 初始化实体嵌入和关系嵌入 |
| 2 | 采样一个正三元组 |
| 3 | 通过替换头或尾构造负三元组 |
| 4 | 计算正例分数和负例分数 |
| 5 | 用 margin ranking loss 计算损失 |
| 6 | 反向更新参数 |
| 7 | 对实体向量做 L2 归一化 |
| 8 | 在验证集上评估 MRR、Hits@K |

其中 MRR 和 Hits@K 是知识图谱补全里常见指标：

- MRR：Mean Reciprocal Rank，白话说，是正确答案平均排第几的倒数。
- Hits@K：白话说，是正确答案是否进入前 K 名的比例。

真实工程例子里，假设你在做企业主数据知识图谱补全，目标是补齐 `company_located_in_city`、`company_belongs_to_industry`、`office_in_country` 这类关系。最务实的做法通常不是一开始上最复杂模型，而是先用 TransE 跑通：

1. 三元组抽取与清洗
2. 实体 ID 映射
3. 负采样逻辑
4. 离线评估指标
5. Top-K 召回接口

只要这条链路没问题，再根据关系模式判断是否升级模型。这样可以把“数据问题”和“模型问题”分开。

---

## 工程权衡与常见坑

TransE 的工程价值不在“它一定最强”，而在“它能快速给出可信起点”。如果一个知识图谱项目连 TransE 都跑不稳定，优先怀疑数据、采样、评估和训练流程，而不是急着换更复杂模型。

它常见的工程权衡如下：

| 维度 | 选择 TransE 的收益 | 代价 |
|---|---|---|
| 实现复杂度 | 很低，几天内可跑通 | 表达能力有限 |
| 训练速度 | 快，适合大规模初筛 | 复杂关系精度可能差 |
| 调试难度 | 低，错误更容易定位 | 上限不高 |
| 可解释性 | 强，易做几何直观分析 | 对关系模式适配差 |

最常见的坑有四类。

| 问题表现 | 原因 | 处理办法 |
|---|---|---|
| 训练 loss 降了，但评估很差 | 负采样误伤真事实 | 采用 filtered negative sampling，过滤已知真三元组 |
| 分数整体越来越极端，结果不稳定 | 忘记实体归一化 | 每步或每轮后做 $\|\mathbf{e}\|_2=1$ 归一化 |
| 排序结果完全反了 | 把“分数小更真”写反 | 检查 loss、排序方向、评估代码 |
| 某些关系效果长期很差 | 关系本身是 1-N 或 N-N | 按关系类型拆模型，或直接换 TransH/TransR/RotatE |

这里重点展开两个坑。

第一个坑是负采样污染。比如知识库里同时存在：

- $(\text{北京大学}, \text{located\_in}, \text{北京})$
- $(\text{清华大学}, \text{located\_in}, \text{北京})$

若你随机替换头实体，某些“替换后”的三元组可能依然为真。模型会被迫把真事实往假方向推，长期会伤害表示空间。

第二个坑是关系类型不匹配。以“国家有城市”为例，这是典型一对多关系。对同一个国家实体，中国加上 `has_city` 后，不可能同时精确落到北京、上海、杭州三个不同位置。于是训练只能折中，把多个正确答案压到一个近似区域，导致 Top-1 质量差、MRR 下降。这不是训练不充分，而是模型归纳假设本身不适配。

真实工程里，常见的策略不是“全图统一一个模型”，而是按关系类型做分层方案。例如：

- 对 `located_in`、`part_of`、`capital_of` 先用 TransE 建 baseline
- 对 `has_department`、`author_of`、`has_city` 这类明显 1-N/N-N 的关系重点观察
- 若复杂关系占比高，再切换更强模型

这比一开始就上重模型更稳，因为你先验证了数据链路是否健康。

---

## 替代方案与适用边界

TransE 不够用的根本原因，不是参数太少，而是关系表达形式太单一。它只允许“同一关系 = 同一平移向量”，这对复杂关系模式约束过强。

后续模型基本都在补这个短板。

| 模型 | 核心思想 | 更适合的关系类型 | 表达能力 | 复杂度 |
|---|---|---|---|---|
| TransE | 关系是平移 | 一对一、结构规整关系 | 基础 | 低 |
| TransH | 关系定义超平面后再平移 | 1-N、N-1 较 TransE 更好 | 中等 | 中低 |
| TransR | 实体空间与关系空间分离 | 关系语义差异大的场景 | 更强 | 中高 |
| RotatE | 关系是复空间旋转 | 对称、反对称、组合模式更强 | 强 | 中高 |

这些模型的改进方向可以概括为：

- TransH：让同一实体在不同关系下投影到不同超平面。超平面，白话说，就是高维空间中的一个“切片平面”。
- TransR：把实体和关系放到不同空间中，允许“同一实体在不同关系下有不同表示”。
- RotatE：不再把关系当平移，而是当旋转，能更自然表达对称、反对称、逆关系等模式。

选型时可以用一个简单规则：

| 数据特征 | 更合适的选择 |
|---|---|
| 需要快速跑通系统、做 baseline | TransE |
| 复杂关系不多，优先低成本验证 | TransE |
| 1-N、N-1、N-N 关系占比较高 | TransH / TransR |
| 对称、反对称、逆关系明显 | RotatE |
| 关系语义差异很大 | TransR |
| 追求更强表达能力，允许更高训练成本 | RotatE 或更复杂模型 |

因此，TransE 的适用边界可以明确写成一句话：当你的目标是快速建立一个结构简单、训练高效、可解释的知识图谱嵌入 baseline 时，它通常是第一选择；当你的数据中复杂关系模式占主要部分时，它通常不应是最终方案。

这不是“谁更高级”的问题，而是“谁更匹配数据结构”的问题。模型选型的核心不是追新，而是看归纳假设是否和任务结构一致。

---

## 参考资料

1. [Translating Embeddings for Modeling Multi-relational Data](https://papers.nips.cc/paper_files/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)
2. [Knowledge Graph Embedding by Translating on Hyperplanes (TransH)](https://www.microsoft.com/en-us/research/publication/knowledge-graph-embedding-by-translating-on-hyperplanes/)
3. [Learning Entity and Relation Embeddings for Knowledge Graph Completion (TransR)](https://doi.org/10.1609/aaai.v29i1.9491)
4. [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://arxiv.org/abs/1902.10197)
