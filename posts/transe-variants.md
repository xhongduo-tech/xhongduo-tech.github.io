## 核心结论

TransE 系列的核心思想可以压缩成一句话：把知识图谱里的事实三元组 $(h,r,t)$ 变成向量几何问题，让头实体 $h$ 加上关系 $r$ 之后尽量接近尾实体 $t$。这里的“实体”就是图谱里的对象，比如国家、城市、公司；“关系”就是对象之间的连接，比如“首都为”“隶属于”“生产于”。

这类方法被称为平移模型，因为它把关系看成一个“位移向量”。最基础的 TransE 假设：

$$
\mathbf{h} + \mathbf{r} \approx \mathbf{t}
$$

它简单、快、参数少，因此常被当作知识图谱嵌入的入门基线模型。问题也很直接：当一个头实体会连向多个尾实体，或者多个头实体会连向同一个尾实体时，单一平移会发生表示挤压。最典型的是 1-N 和 N-1 关系，也就是“一个头对应多个尾”或“多个头对应一个尾”。

TransE 的三个常见变体，正是在这个问题上逐步加自由度：

| 模型 | 空间处理方式 | 直观理解 | 主要改进点 |
| --- | --- | --- | --- |
| TransE | 单一共享空间中的平移 | 所有关系都在同一张地图里移动 | 简单高效，但容易退化 |
| TransH | 先投影到关系超平面，再平移 | 不同关系先在各自平面上看实体 | 缓解 1-N / N-1 |
| TransR | 投影到关系特定空间，再平移 | 每个关系都有自己的语义空间 | 分离实体语义与关系语义 |
| TransD | 根据实体和关系动态生成投影 | 每次交互时临时决定怎么映射 | 更灵活，参数效率更好 |

一个直观例子是“国家”和“首都”。如果 $h=$“法国”，$r=$“首都为”，$t=$“巴黎”，TransE 直接要求“法国向量 + 首都为向量”接近“巴黎向量”。但如果关系换成“下辖省份”，一个国家可能对应很多省份，这时单一平移容易把多个省份压到相近位置。TransH、TransR、TransD 通过“先投影再平移”或“动态投影再平移”给模型更多表达能力。

如果只看文献里常见基线，WN18RR 上 TransE 的 MRR 约为 22.6%，TransR 可到约 23.1% 左右。这里的 MRR 是平均倒数排名，可以粗略理解为“真实答案平均排得有多靠前”，数值越高越好。这个提升不大，但它说明：仅靠更细粒度的空间建模，确实可以修正一部分 TransE 的结构性缺陷。

---

## 问题定义与边界

知识图谱嵌入的目标，是把离散的三元组数据变成固定维度的连续向量，使模型可以用几何距离判断某个事实是否可信。所谓“嵌入”，就是把一个离散对象映射成一个可训练的向量。训练后，真实三元组的得分应更低，错误三元组的得分应更高。

更正式地说，给定实体集合 $\mathcal{E}$、关系集合 $\mathcal{R}$，希望学习映射：

$$
e \in \mathcal{E} \mapsto \mathbf{e} \in \mathbb{R}^d,\quad
r \in \mathcal{R} \mapsto \mathbf{r} \in \mathbb{R}^d
$$

然后定义评分函数 $f_r(h,t)$。在平移模型里，通常是“距离越小，三元组越真”。

这个问题有一个关键边界：不是所有关系都适合用“全局平移”表达。

以 1-N 关系为例。假设存在：

- （中国，包含省份，浙江）
- （中国，包含省份，江苏）
- （中国，包含省份，广东）

如果用 TransE，那么模型希望同时满足：

$$
\mathbf{China} + \mathbf{contains\_province} \approx \mathbf{Zhejiang}
$$

$$
\mathbf{China} + \mathbf{contains\_province} \approx \mathbf{Jiangsu}
$$

$$
\mathbf{China} + \mathbf{contains\_province} \approx \mathbf{Guangdong}
$$

这意味着这几个尾实体向量会被拉向同一个位置附近。结果不是知识正确被表达，而是多个答案在空间里被压扁了。N-1 关系也一样，例如多个学生对应同一个学校时，头实体会被拉到相近区域。

下面这个表能看出差异：

| 场景 | TransE 的空间效果 | 带投影的变体效果 |
| --- | --- | --- |
| 1-1，例如“国家-首都” | 往往足够好 | 也能工作，但未必必要 |
| 1-N，例如“国家-省份” | 多个尾实体容易重叠 | 可在关系子空间中拉开 |
| N-1，例如“学生-学校” | 多个头实体容易重叠 | 可在投影后保留差异 |
| N-N，例如“演员-参演电影” | 容易混杂 | 需要更多结构能力 |

所以边界很明确：

1. 如果关系大多接近 1-1，TransE 往往够用。
2. 如果关系强烈多值化，单一平移不够，必须引入额外自由度。
3. 这些额外自由度最常见的做法，不是换掉“平移”本身，而是在平移前增加投影。

玩具例子可以更直观。设二维空间里：

- “法国”在 $(0,0)$
- “首都为”在 $(1,0)$
- “巴黎”在 $(1,0)$

那么它们刚好满足一次平移。但若“下辖省份”也固定成同一个向量，浙江、江苏、广东都会被要求接近同一个终点，这就是 TransE 的退化来源。

真实工程里，这种问题经常出现在电商、医疗、金融图谱中。比如“药物-治疗-疾病”可能是一对多，“公司-投资-企业”可能是多对多。如果还坚持只用单一平移，模型在召回阶段往往会把一组候选挤在一起，排序能力下降。

---

## 核心机制与推导

TransE 的评分函数最常写成：

$$
f_r(h,t)=\|\mathbf{h}+\mathbf{r}-\mathbf{t}\|_p
$$

其中 $\|\cdot\|_p$ 一般取 $L_1$ 或 $L_2$ 范数。所谓“范数”，就是向量长度或距离的计算方式。分数越低，表示三元组越可能是真的。

训练时不会只看正样本，还会构造负样本。所谓“负采样”，就是把真实三元组里的头或尾随机替换成错误实体，比如把（法国，首都为，巴黎）替换成（法国，首都为，苹果）。然后用 margin ranking loss 拉开正负样本距离：

$$
\mathcal{L}=\sum_{(h,r,t)\in \mathcal{S}}\sum_{(h',r,t')\in \mathcal{S}'}
\max\left(0,\gamma + f_r(h,t)-f_r(h',t')\right)
$$

其中 $\gamma$ 是 margin，可以理解为“正负样本至少要拉开多少距离”。

### TransE

TransE 最直接：

- 输入：实体向量 $\mathbf{h}, \mathbf{t}$，关系向量 $\mathbf{r}$
- 操作：直接做 $\mathbf{h}+\mathbf{r}$
- 输出：与 $\mathbf{t}$ 比距离

它的问题不是公式错，而是自由度太少。一个关系只有一个平移向量，所有实体共享同一几何规则。

### TransH

TransH 的想法是：同一个实体在不同关系下，不一定应该看成同一个点。它给每个关系定义一个超平面。所谓“超平面”，就是高维空间中的一张平面。先把实体投影到这张平面，再做平移。

设关系 $r$ 的法向量为 $\mathbf{w}_r$，则：

$$
\mathbf{h}_\perp=\mathbf{h} - \mathbf{w}_r^\top \mathbf{h}\,\mathbf{w}_r
$$

$$
\mathbf{t}_\perp=\mathbf{t} - \mathbf{w}_r^\top \mathbf{t}\,\mathbf{w}_r
$$

然后评分函数变成：

$$
f_r(h,t)=\|\mathbf{h}_\perp+\mathbf{d}_r-\mathbf{t}_\perp\|_p
$$

这里 $\mathbf{d}_r$ 是关系在超平面内的平移向量。它的意义是：同一个实体，在“首都为”和“位于洲”这两个关系下，可以先投影到不同观察平面，再比较。

### TransR

TransR 再往前一步。它认为实体空间和关系空间本来就不该完全一样。比如“巴黎”作为实体，有人口、地理、文化等多种语义；但关系“首都为”只关心其中一部分。因此每个关系都应有自己的映射矩阵 $\mathbf{M}_r$，把实体从实体空间映射到关系空间：

$$
\mathbf{h}_r=\mathbf{M}_r\mathbf{h},\quad \mathbf{t}_r=\mathbf{M}_r\mathbf{t}
$$

评分函数是：

$$
f_r(h,t)=\|\mathbf{h}_r+\mathbf{r}-\mathbf{t}_r\|_p
$$

TransR 的能力更强，因为它不只是“换个平面看”，而是“换个关系专用空间看”。

### TransD

TransD 的关键是动态投影。它不再只给关系一个固定矩阵，而是让实体和关系各自带一个投影向量，再根据当前实体-关系对构造投影矩阵。直观上，模型会根据“谁和谁在交互”决定映射方式。

常见写法是：

$$
\mathbf{M}_{rh}=\tilde{\mathbf{r}}\tilde{\mathbf{h}}^\top+\mathbf{I}
$$

$$
\mathbf{M}_{rt}=\tilde{\mathbf{r}}\tilde{\mathbf{t}}^\top+\mathbf{I}
$$

然后：

$$
\mathbf{h}_r=\mathbf{M}_{rh}\mathbf{h},\quad \mathbf{t}_r=\mathbf{M}_{rt}\mathbf{t}
$$

最终仍然做平移评分：

$$
f_r(h,t)=\|\mathbf{h}_r+\mathbf{r}-\mathbf{t}_r\|_p
$$

它比 TransR 更灵活，因为不需要为每个关系显式保存一个大矩阵；但训练也更敏感，因为投影本身是动态生成的。

下面把四个模型放在一张流程表里：

| 模型 | 输入 | 投影步骤 | 平移步骤 | 输出评分 |
| --- | --- | --- | --- | --- |
| TransE | $\mathbf{h},\mathbf{r},\mathbf{t}$ | 无 | $\mathbf{h}+\mathbf{r}$ | 与 $\mathbf{t}$ 的距离 |
| TransH | $\mathbf{h},\mathbf{r},\mathbf{t}$ | 投影到关系超平面 | $\mathbf{h}_\perp+\mathbf{d}_r$ | 与 $\mathbf{t}_\perp$ 的距离 |
| TransR | $\mathbf{h},\mathbf{r},\mathbf{t}$ | 乘关系矩阵 $\mathbf{M}_r$ | $\mathbf{h}_r+\mathbf{r}$ | 与 $\mathbf{t}_r$ 的距离 |
| TransD | $\mathbf{h},\mathbf{r},\mathbf{t}$ | 动态构造 $\mathbf{M}_{rh},\mathbf{M}_{rt}$ | $\mathbf{h}_r+\mathbf{r}$ | 与 $\mathbf{t}_r$ 的距离 |

再看一个最小对齐例子。令：

$$
\mathbf{h}=[0,0],\quad \mathbf{r}=[1,0],\quad \mathbf{t}=[1,0]
$$

则 TransE 中：

$$
\|\mathbf{h}+\mathbf{r}-\mathbf{t}\|=\|[0,0]+[1,0]-[1,0]\|=0
$$

如果 TransH 取法向量 $\mathbf{w}_r=[0,1]$，那么点本来就在平面内，投影不变；如果 TransR 取 $\mathbf{M}_r=\mathbf{I}$，也不变；TransD 若投影向量使动态矩阵退化为单位阵，也不变。这个例子说明：这些变体本质上是在基础平移之外增加条件化表达，不会破坏最基本的“可平移对齐”行为。

---

## 代码实现

工程实现时，最值得抽象的不是训练循环，而是“投影模块”。因为 TransE、TransH、TransR、TransD 的差异，本质上都发生在 `project(entity, relation)` 这一步。

一个最小训练流程通常是：

1. 读取实体、关系和三元组，建立 id 映射。
2. 初始化实体向量与关系向量。
3. 为每个正样本构造负样本。
4. 计算正负样本分数。
5. 用 margin ranking loss 更新参数。
6. 在验证集上评估 MRR、Hits@K。

先看最简版伪代码：

```python
def project(entity_vec, relation_id):
    return entity_vec  # TransE: identity

def score(h, r, t):
    h_proj = project(h, r)
    t_proj = project(t, r)
    return norm(h_proj + rel_emb[r] - t_proj)

loss = max(0.0, margin + score(h, r, t) - score(h_neg, r, t_neg))
```

如果换成 TransH，`project` 里先减去法向量分量；如果换成 TransR，就乘以关系矩阵；如果换成 TransD，就根据实体投影向量和关系投影向量临时生成矩阵。

下面给一个可运行的 Python 玩具实现。它不是完整训练器，但能把平移评分、负样本损失和 TransE/TransH 的投影逻辑跑通。

```python
import math

def l2_norm(vec):
    return math.sqrt(sum(x * x for x in vec))

def vec_add(a, b):
    return [x + y for x, y in zip(a, b)]

def vec_sub(a, b):
    return [x - y for x, y in zip(a, b)]

def vec_dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def vec_scalar_mul(s, v):
    return [s * x for x in v]

def transe_project(entity_vec, relation):
    return entity_vec

def transh_project(entity_vec, relation):
    w = relation["normal"]
    # 投影到关系超平面：e - (w^T e) w
    coeff = vec_dot(w, entity_vec)
    return vec_sub(entity_vec, vec_scalar_mul(coeff, w))

def score(entity_h, relation_vec, entity_t, project_fn, relation_meta=None):
    h_proj = project_fn(entity_h, relation_meta or {})
    t_proj = project_fn(entity_t, relation_meta or {})
    return l2_norm(vec_sub(vec_add(h_proj, relation_vec), t_proj))

def margin_ranking_loss(pos_score, neg_score, margin=1.0):
    return max(0.0, margin + pos_score - neg_score)

# 玩具例子：法国 + 首都为 = 巴黎
france = [0.0, 0.0]
capital_of = [1.0, 0.0]
paris = [1.0, 0.0]
apple = [3.0, 3.0]

pos = score(france, capital_of, paris, transe_project)
neg = score(france, capital_of, apple, transe_project)
loss = margin_ranking_loss(pos, neg, margin=1.0)

assert abs(pos - 0.0) < 1e-9
assert neg > pos
assert loss == 0.0

# TransH 在简单情形下退化为同样结果
relation_meta = {"normal": [0.0, 1.0]}
pos_h = score(france, capital_of, paris, transh_project, relation_meta)
assert abs(pos_h - 0.0) < 1e-9

# 一个 1-N 风格的小例子：多个省份共享同一平移目标，会产生挤压趋势
china = [0.0, 0.0]
contains_province = [2.0, 0.0]
zhejiang = [2.1, 0.1]
jiangsu = [1.9, -0.1]

s1 = score(china, contains_province, zhejiang, transe_project)
s2 = score(china, contains_province, jiangsu, transe_project)

assert s1 < 0.2
assert s2 < 0.2
```

这个代码体现了两点：

1. TransE 的主体非常简单，适合做基线和教学演示。
2. 一旦把 `project` 抽象清楚，后续变体可以复用同一套训练循环。

真实工程例子通常不是手写向量，而是从数据集中读取三元组，例如：

- 实体：药物、疾病、靶点
- 关系：治疗、作用于、副作用
- 任务：给定（药物，治疗，?）预测疾病候选排序

在这种场景里，常见做法是先用 TransE 快速训练一个基线，确认负采样、评估指标、数据清洗链路没问题，再切换到 TransR 或 TransD。原因很简单：如果基础数据管道都不稳定，直接上更复杂模型，问题会被掩盖得更深。

---

## 工程权衡与常见坑

TransE 系列真正难的地方，不在公式，而在工程细节。新手最容易误以为“模型越复杂越好”，实际不是。复杂度、参数量、训练稳定性、负采样质量，这几项必须一起看。

先给一张工程对比表：

| 模型 | 参数规模 | 训练技巧 | 常见坑 | 适用场景 |
| --- | --- | --- | --- | --- |
| TransE | 最小 | 负采样、向量归一化、margin 调整 | 1-N/N-1 退化明显 | 快速基线、关系简单 |
| TransH | 小到中 | 约束法向量单位化、超平面正则 | 法向量训练不稳 | 多值关系较多但资源有限 |
| TransR | 中到大 | 矩阵正则、较小学习率、分空间训练 | 参数多，容易过拟合 | 关系语义差异明显 |
| TransD | 中等 | 投影向量初始化、warmup、L2 正则 | 动态投影敏感，收敛不稳 | 需要灵活投影且控制参数 |

### 1. 负采样质量往往比换模型更先决定效果

负采样就是造“假事实”给模型学。问题是，如果负样本太容易，模型很快就学会区分，训练损失会很好看，但排序效果不会真正提升。

例如把（阿司匹林，治疗，发热）替换成（阿司匹林，治疗，银河系），这种负样本几乎没有训练价值。更好的方式是替换成语义上接近但实际上错误的实体，比如（阿司匹林，治疗，高血压）这种 harder negative。

文献里 TransR 配合更强的负采样策略时，WN18RR 上能从 TransE 的约 22.6% MRR 提升到约 23.0% 或更高。这个数字说明一个现实：高质量负样本和更细粒度投影都有效，但前者常常更先决定模型下限。

### 2. TransR 的参数量会迅速膨胀

如果实体维度是 $d_e$，关系空间维度是 $d_r$，每个关系都要一个 $d_r \times d_e$ 矩阵，那么关系数一多，参数会迅速上涨。对于大规模工业图谱，这不只是显存问题，还会让训练变慢、正则化变难。

所以 TransR 适合“关系类型不算太多，但每种关系语义差异明显”的场景。比如企业关系图谱里，“投资”“收购”“供应”“任职”是不同的语义空间，TransR 通常比 TransE 更合理。

### 3. TransD 更省矩阵，但更依赖初始化

TransD 表面上不需要给每个关系存一个完整矩阵，这是优点。但它把投影矩阵的构造转移成了“由实体向量和关系向量共同生成”，这意味着初始化不好时，早期训练会很抖。

工程上常见处理是：

- 先训练一轮或几轮 TransE 作为预热
- 用得到的实体向量初始化更复杂模型
- 对投影向量使用较小学习率
- 加 warmup，避免一开始梯度过大

### 4. 归一化和约束不是可选项

TransE 系列很容易出现向量无限放大来“投机取巧”降低损失，因此常见做法是对实体嵌入做单位球归一化，或者加范数约束。TransH 还常要求法向量接近单位长度，否则“超平面”会失去稳定几何意义。

### 5. 指标不要只看 Hits@10

Hits@10 可以理解为“真实答案是否排进前十”。它容易显得好看，但不够敏感。MRR 更能反映排序质量。如果模型只是把正确答案从第 100 名提到第 9 名，Hits@10 会很高兴，但真实检索体验未必够好。

真实工程例子里，做医疗知识图谱补全时，通常不只是看“是否召回疾病候选”，还要看“能否把关键候选排到前几位”。这时 MRR 比 Hits@10 更有价值。

---

## 替代方案与适用边界

TransE 系列不是知识图谱嵌入的终点，而是一类结构非常清晰的基线。如果问题已经超出“单步平移 + 投影”能处理的范围，就该考虑替代方案。

| 方法 | 核心机制 | 优势 | 限制 | 比平移模型更适合的边界 |
| --- | --- | --- | --- | --- |
| ComplEx | 复数空间建模 | 擅长非对称关系 | 可解释性较弱 | 关系方向性强 |
| RotatE | 复数空间旋转 | 对 1-N/N-1/N-N 更友好 | 训练仍依赖采样质量 | 多种映射模式共存 |
| ConvE | 卷积式交互 | 建模能力强于简单平移 | 结构更重 | 需要更复杂实体-关系交互 |
| R-GCN | 图神经网络聚合 | 利用邻域结构 | 训练和部署更重 | 图结构丰富、局部依赖强 |

ComplEx 的“复数空间”可以白话理解为：每个维度不仅有大小，还有额外相位信息，因此能更自然表达非对称关系，例如“父亲是”与“儿子是”的方向不同。RotatE 把关系看成旋转而不是平移，对 1-N/N-1 通常更有辨识力。

R-GCN 则完全是另一条路。它不只是看一个三元组，而是聚合邻居信息。所谓“图神经网络聚合”，就是让一个节点表示从周围节点和边里吸收上下文。这对复杂图谱很有效，但代价是训练和在线部署都更重。

一个现实决策可以这样理解：

1. 数据刚起步，想先确认图谱质量和训练链路，先上 TransE。
2. 明显存在大量多值关系，且希望维持较低复杂度，试 TransH。
3. 关系语义差异明显，愿意为表达力付出更多参数，试 TransR。
4. 关系很多、希望投影更灵活但不想存大量矩阵，试 TransD。
5. 如果多步推理、结构依赖、非对称关系很强，直接考虑 RotatE、ComplEx 或 R-GCN。

再给一个真实工程例子。医疗图谱通常有“药物-靶点-通路-疾病”的长链关系。如果目标只是做初步候选召回，TransE 系列足够轻，部署方便；如果目标是结合上下文邻域做更精细推理，例如根据多个路径联合判断药物重定位，R-GCN 这类模型更合适。实践里常见做法是先用 TransE 系列预热实体嵌入，再把它作为图神经网络的初始化，这样训练更稳。

所以适用边界可以总结成一句话：平移模型擅长把局部事实压缩成统一几何规则，但当关系模式、邻域结构或推理链条明显复杂化时，就该换到更强表达范式。

---

## 参考资料

1. Emergent Mind, TransE 主题综述，涉及 margin ranking loss、1-N/N-1 退化讨论。  
2. Next Electronics, Knowledge Graph Completion Models，包含 TransE、TransH、TransR 的基础公式与比较。  
3. KGE 文档，TransD API 与动态投影公式说明。  
4. PMC 论文《A Novel Encoder-Decoder Knowledge Graph Completion Model for Robot Brain》，给出 WN18RR 上 TransE 的约 22.6% MRR 基线。  
5. Op-Trans 相关论文与实验表，说明带更强负采样的 TransR 可达到约 23.0% 级别 MRR。
