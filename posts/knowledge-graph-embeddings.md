## 核心结论

TransE、TransR、RotatE 都在做同一件事：把知识图谱里的三元组 $(h,r,t)$ 映射到向量空间里，让“真实三元组得分低，伪造三元组得分高”。这里的三元组就是“头实体、关系、尾实体”，例如“北京-属于-中国”。

三者的区别不在训练框架，而在“关系怎么作用到实体”：

| 模型 | 建模方式 | 几何解释 | 更适合的关系 |
|---|---|---|---|
| TransE | $\mathbf{h}+\mathbf{r}\approx\mathbf{t}$ | 向量平移 | 一对一、结构较简单 |
| TransR | $\mathbf{h}\mathbf{M}_r+\mathbf{r}\approx\mathbf{t}\mathbf{M}_r$ | 先投影到关系空间，再平移 | 一对多、多对一、多对多 |
| RotatE | $\mathbf{e}_h\circ\mathbf{r}\approx\mathbf{e}_t$ | 复数域旋转 | 对称、反对称、组合关系 |

统一训练目标通常是负采样加 Margin Ranking Loss。负采样就是把真实三元组中的头实体或尾实体替换掉，人工造出错误样本；Margin Ranking 的作用是强制正样本比分错样本更“接近正确”。

结论可以先记成一句话：

$$
\text{TransE 用平移，TransR 用关系专属空间，RotatE 用复数旋转。}
$$

如果数据里关系简单、参数预算紧，先试 TransE；如果多对多关系多，优先考虑 TransR 或其轻量变体；如果需要同时表达对称、反对称和关系组合，RotatE 往往更稳。

---

## 问题定义与边界

知识图谱嵌入的目标，是把离散符号变成可学习的连续向量，用于链接预测。链接预测可以理解为：给定 $(h,r,?)$ 或 $(?,r,t)$，预测缺失实体。

一个基础例子：

- 正例：`(北京, 属于, 中国)`
- 待预测：`(上海, 属于, ?)`，模型应给“中国”更高排名

这里的“嵌入”就是实体和关系的稠密向量表示；“打分函数”就是判断三元组是否合理的规则。

这类方法的边界主要由关系模式决定。关系模式就是同一种关系在图中的连接形态，例如：

| 关系模式 | 例子 | 对模型的挑战 |
|---|---|---|
| 一对一 | `国家-首都-城市` | 相对容易 |
| 一对多 | `国家-包含-城市` | 一个头实体对应多个尾实体 |
| 多对一 | `城市-属于-国家` | 多个头实体对应同一尾实体 |
| 多对多 | `演员-参演-电影` | 两边都可能有多个匹配 |
| 对称 | `相邻于` | $(a,r,b)$ 与 $(b,r,a)$ 同时成立 |
| 反对称 | `父亲` | $(a,r,b)$ 成立时，$(b,r,a)$ 通常不成立 |

TransE 最大的问题，是它默认一个关系像固定平移向量。若 `父亲` 是一对多关系，理论上要满足：

$$
\mathbf{h}+\mathbf{r}\approx\mathbf{t}_1,\quad
\mathbf{h}+\mathbf{r}\approx\mathbf{t}_2
$$

这意味着 $\mathbf{t}_1$ 和 $\mathbf{t}_2$ 需要彼此非常接近。若一个父亲有多个子女，尾实体会被强行挤在一起，表达能力就不够。

玩具例子：

- 正例 1：`(张三, 父亲, 儿子A)`
- 正例 2：`(张三, 父亲, 儿子B)`

若用 TransE，两条式子同时成立时，会逼近：

$$
\mathbf{儿子A}\approx\mathbf{儿子B}
$$

这显然不合理，因为两个实体不该仅因共享同一关系就变得几乎相同。

训练时常见损失函数是：

$$
\mathcal{L}=\sum_{(h,r,t)\in S}\sum_{(h',r,t')\in S'}
\left[\gamma + f(h,r,t)-f(h',r,t')\right]_+
$$

其中：

- $S$ 是正样本集合
- $S'$ 是负样本集合
- $\gamma$ 是 margin，表示正负样本之间至少拉开的间隔
- $[x]_+=\max(0,x)$

边界也在于负样本质量。如果负样本太容易，例如把 `(北京, 属于, 中国)` 改成 `(北京, 属于, 香蕉)`，模型很快就学会，训练信号会变弱。难例才真正决定效果。

---

## 核心机制与推导

先看 TransE。它最直接，假设真实三元组满足：

$$
\mathbf{h}+\mathbf{r}\approx\mathbf{t}
$$

因此分数函数定义为：

$$
f_{\text{TransE}}(h,r,t)=\|\mathbf{h}+\mathbf{r}-\mathbf{t}\|_p
$$

这里的 $p$ 一般取 1 或 2。分数越小，三元组越可能为真。

玩具例子可以直接算。设：

$$
\mathbf{h}=(1,0),\quad \mathbf{r}=(0,1),\quad \mathbf{t}=(1,1)
$$

那么：

$$
\mathbf{h}+\mathbf{r}-\mathbf{t}=(1,0)+(0,1)-(1,1)=(0,0)
$$

所以：

$$
f(h,r,t)=0
$$

若负例尾实体改成 $\mathbf{t}'=(0,0)$，则：

$$
f(h,r,t')=\|(1,0)+(0,1)-(0,0)\|_2=\|(1,1)\|_2=\sqrt{2}
$$

这就是平移模型的直观含义：真实尾实体应落在“头实体加关系”的附近。

TransR 在 TransE 基础上加了一层关系专属空间。所谓“关系空间”，就是每个关系不一定在同一语义维度上观察实体。做法是为每个关系引入投影矩阵 $\mathbf{M}_r$：

$$
\mathbf{h}_r=\mathbf{h}\mathbf{M}_r,\quad \mathbf{t}_r=\mathbf{t}\mathbf{M}_r
$$

然后在关系空间中再做平移：

$$
f_{\text{TransR}}(h,r,t)=\|\mathbf{h}\mathbf{M}_r+\mathbf{r}-\mathbf{t}\mathbf{M}_r\|_p
$$

它的直觉是：同一个实体在不同关系下，应该呈现不同特征。比如“苹果”在“生产于”关系里更像公司，在“属于水果类别”关系里又更像食物概念。TransR 通过 $\mathbf{M}_r$ 让同一实体在不同关系下投影成不同表示。

RotatE 再进一步，把实体放到复数空间 $\mathbb{C}^k$。复数可以理解为“带方向的二维数对”，很适合表达旋转。RotatE 令关系向量的每一维都是单位模复数：

$$
\mathbf{r}_i=e^{i\theta_{r,i}}
$$

然后要求：

$$
\mathbf{e}_h\circ \mathbf{r}\approx \mathbf{e}_t
$$

其中 $\circ$ 是逐维乘法。打分函数通常写为：

$$
f_{\text{RotatE}}(h,r,t)=\|\mathbf{e}_h\circ\mathbf{r}-\mathbf{e}_t\|
$$

它的重要优势是能自然表达关系模式：

- 对称关系：旋转后再旋转可回到原位
- 反对称关系：正向和反向不是同一变换
- 组合关系：多个关系旋转角度可以相加

旋转玩具例子如下。设头实体为复数 $1+0i$，关系是逆时针旋转 $\frac{\pi}{2}$：

$$
r=e^{i\pi/2}=i
$$

则：

$$
(1+0i)\cdot i=i
$$

结果正好落到 $(0,1)$ 方向，这对应二维平面中从 x 轴正方向旋转到 y 轴正方向。它比简单平移更容易编码方向性和组合性。

训练时三者都可以用 Margin Ranking Loss。RotatE 常额外使用自对抗负采样。所谓“自对抗”，就是让模型当前认为更像真的负样本获得更高权重，公式可简写为：

$$
p_i=\frac{\exp(\alpha f_i)}{\sum_j \exp(\alpha f_j)}
$$

其中 $f_i$ 可理解为某种“难度分数”，$\alpha$ 控制集中程度。直观上，越难区分的负例，训练时越该重点学。

真实工程例子：在一个电商知识图谱里，可能有关系 `品牌-生产-商品`、`商品-属于-类目`、`用户-浏览过-商品`。如果只用 TransE，`品牌-生产-商品` 这类一对多关系容易把同品牌商品挤得过近，导致召回时很多商品分数相似；RotatE 或 TransR 往往更容易拉开结构差异。

---

## 代码实现

下面给一个最小可运行版本，用 NumPy 展示三种打分函数和 margin loss。它不是完整训练器，但足以说明前向计算逻辑。

```python
import numpy as np

def l2(x):
    return np.sqrt(np.sum(x * x, axis=-1))

def score_transe(h, r, t):
    return l2(h + r - t)

def score_transr(h, mr, r, t):
    h_proj = h @ mr
    t_proj = t @ mr
    return l2(h_proj + r - t_proj)

def score_rotate(h_re, h_im, phase, t_re, t_im):
    # r = cos(theta) + i sin(theta)
    r_re = np.cos(phase)
    r_im = np.sin(phase)

    # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    rot_re = h_re * r_re - h_im * r_im
    rot_im = h_re * r_im + h_im * r_re

    return l2(rot_re - t_re) + l2(rot_im - t_im)

def margin_ranking_loss(pos_score, neg_score, gamma=1.0):
    return np.maximum(0.0, gamma + pos_score - neg_score)

# TransE toy example
h = np.array([1.0, 0.0])
r = np.array([0.0, 1.0])
t = np.array([1.0, 1.0])
t_neg = np.array([0.0, 0.0])

pos = score_transe(h, r, t)
neg = score_transe(h, r, t_neg)
loss = margin_ranking_loss(pos, neg, gamma=1.0)

assert np.isclose(pos, 0.0)
assert np.isclose(neg, np.sqrt(2.0))
assert np.isclose(loss, max(0.0, 1.0 - np.sqrt(2.0)))

# TransR toy example
mr = np.array([[1.0, 0.0], [0.0, 2.0]])
r_rel = np.array([0.0, 1.0])
transr_score = score_transr(h, mr, r_rel, t)
assert transr_score >= 0.0

# RotatE toy example: rotate (1,0) by pi/2 to (0,1)
h_re = np.array([1.0])
h_im = np.array([0.0])
phase = np.array([np.pi / 2])
t_re = np.array([0.0])
t_im = np.array([1.0])

rotate_score = score_rotate(h_re, h_im, phase, t_re, t_im)
assert np.isclose(rotate_score, 0.0, atol=1e-6)
```

如果写成一个 batch 的训练框架，核心步骤一般如下：

```python
def train_one_batch(model, pos_triples, entity_emb, relation_emb, relation_proj=None, gamma=1.0):
    neg_triples = negative_sample(pos_triples)  # 替换头或尾实体
    pos_score = model(pos_triples, entity_emb, relation_emb, relation_proj)
    neg_score = model(neg_triples, entity_emb, relation_emb, relation_proj)
    loss = np.maximum(0.0, gamma + pos_score - neg_score).mean()
    return loss
```

这里的 `negative_sample` 有两个基础策略：

1. 替换头实体：$(h,r,t)\rightarrow(h',r,t)$
2. 替换尾实体：$(h,r,t)\rightarrow(h,r,t')$

训练时通常还要做约束。例如 TransE 常把实体向量限制在单位球附近，避免向量无限增大后仅靠范数取巧。RotatE 则通常约束关系模长接近 1，因为它的设计本意就是“旋转而非缩放”。

真实工程例子可以更具体一点。假设你要做问答系统中的实体补全，输入是：

- `(阿司匹林, 治疗, ?)`
- `(北京大学, 位于, ?)`
- `(iPhone 15, 属于品牌, ?)`  

推理阶段通常不是输出单个分数，而是把所有候选尾实体都打分排序，然后取 top-k。此时模型打分函数的稳定性，比单个三元组能否分类更重要。TransE 排序快，但在复杂关系上误召回更多；RotatE 推理更重一些，但排序质量常更稳定。

---

## 工程权衡与常见坑

工程上最先要看的不是“论文指标最高”，而是关系分布、实体规模和算力预算。

| 维度 | TransE | TransR | RotatE |
|---|---|---|---|
| 参数量 | 低 | 高，额外有 $\mathbf{M}_r$ | 中等 |
| 训练速度 | 快 | 慢 | 中等 |
| 多对多关系 | 弱 | 强 | 强 |
| 对称/反对称 | 弱 | 一般 | 强 |
| 组合关系 | 一般 | 一般 | 强 |
| 部署难度 | 低 | 中 | 中 |

常见坑主要有五类。

第一，关系模式不匹配。拿 TransE 直接做多对多图谱，常见现象是很多候选尾实体得分扎堆，排名很难拉开。新手容易把这误判成“训练不充分”，实际上是模型假设太弱。

第二，负采样太容易。若负例明显错误，loss 很快变成 0，梯度消失，后期几乎不学。解决方式通常是提高难例比例，或采用自对抗负采样。

第三，TransR 参数爆炸。若实体维度 $d$、关系维度 $k$、关系数 $|\mathcal{R}|$ 很大，则投影矩阵参数量接近：

$$
|\mathcal{R}| \times d \times k
$$

关系一多，显存和训练时间都会明显上升。实际工程中经常要做低秩分解、共享参数或降维。

第四，评测泄漏。知识图谱补全常用 filtered setting，即评测负例时要去掉训练集里真实存在的三元组。否则你可能把另一个真三元组误当假样本，指标会失真。

第五，只看 loss 不看排序指标。知识图谱任务最终往往关心 Hits@K、MRR 这类排序指标，而不是 loss 本身。loss 下降不一定意味着 top-k 召回提升。

一个真实工程场景：在商品知识图谱里，关系 `品牌-生产-商品`、`商品-属于-类目`、`商品-适配-配件` 都是高频多值关系。只用 TransE 时，线上候选集经常出现“同一品牌商品大量重复挤进前 20”的现象，业务看起来像召回很多，实际上去重后有效信息很少。切换到 TransR 或 RotatE 后，候选尾实体间区分度通常更高。

规避策略可以直接列出来：

- 关系简单、图谱大、延迟敏感时，先用 TransE 做基线。
- 多对多关系多时，优先尝试 TransR、TransD 或 RotatE。
- 负采样不要只做纯随机，至少混入同类型实体替换。
- TransR 关系太多时，优先做低秩矩阵或共享投影。
- 训练和评测都要使用 filtered 协议。
- 线上部署关注 top-k 排名质量，不只看训练损失。

---

## 替代方案与适用边界

TransE、TransR、RotatE 很经典，但不是唯一选择。替代方案的核心思路，是在“关系如何作用于实体”上做不同折中。

| 替代方案 | 核心思想 | 优势 | 劣势 | 适用情形 |
|---|---|---|---|---|
| TransH | 关系定义超平面后再平移 | 比 TransE 更能区分关系语义 | 仍有限制 | 需要轻量改进时 |
| TransD | 实体和关系都有动态投影 | 比 TransR 更省参数 | 实现更复杂 | 关系多、资源有限 |
| DistMult | 三线性对角张量打分 | 快、简单 | 难表达反对称 | 关系偏对称 |
| ComplEx | 复数域双线性建模 | 能表示反对称 | 解释性不如平移族直观 | 追求表达力 |
| SimplE | 双向角色嵌入 | 参数较规整 | 调参依赖较强 | 需要平衡复杂度与效果 |

适用边界可以总结为：

- 如果图谱以一对一、层级关系为主，TransE 够用，且训练部署都便宜。
- 如果关系复杂且数量不是特别夸张，TransR 更稳，但要准备更多算力。
- 如果对称、反对称、组合关系都重要，RotatE 是优先级很高的候选。
- 如果关系很多、TransR 太重，可以先试 TransD。
- 如果更看重表达能力而不是几何直观性，ComplEx 往往是强基线。

举例来说，在一个半结构化企业知识图谱中，既有 `公司-收购-公司` 这类方向敏感关系，也有 `公司-合作-公司` 这类接近对称的关系，还有 `产品-属于-业务线` 这类多对一关系。这时直接从 RotatE 或 TransD 起步通常比从 TransE 起步更省试错成本。

需要强调一个边界：这些方法本质上都属于浅层嵌入模型，依赖预定义三元组，不直接理解长文本上下文。如果你的知识主要藏在文档里，或需要复杂多跳推理，仅靠 TransE/TransR/RotatE 往往不够，通常要结合文本编码器、GNN 或检索增强方案。

---

## 参考资料

1. Bordes 等，《Translating Embeddings for Modeling Multi-relational Data》。贡献：提出 TransE，用简单平移建模知识图谱三元组，是平移族方法的起点。来源：原始论文与相关综述页面。  
2. Lin 等，《Learning Entity and Relation Embeddings for Knowledge Graph Completion》。贡献：提出 TransR，将实体投影到关系空间后再平移，增强对复杂关系的表达。来源：原始论文与知识图谱嵌入文档页面。  
3. Sun 等，《RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space》。贡献：提出 RotatE，用复数旋转表达对称、反对称和组合关系，并结合自对抗负采样。来源：原始论文与综述页面。  
4. Emergent Mind 的 TransE/RotatE 主题综述。贡献：对平移模型家族、训练目标、适用关系模式和工程实践做了集中整理，适合快速回顾。来源：Emergent Mind 专题页面。  
5. Knowledge Graph Embedding 文档中的 TransR 条目。贡献：给出 TransR 的实现接口、矩阵投影形式和工程使用方式，适合作为实现参考。来源：Read the Docs 文档页面。
