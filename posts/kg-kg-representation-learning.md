## 核心结论

知识图谱表示学习，白话说，就是把“实体”和“关系”都编码成向量，让向量之间的几何运算近似三元组语义。一个三元组写作 $(h,r,t)$，分别表示头实体、关系、尾实体，例如“Bill Gates - 创立 - 微软”。

它的核心目标不是“记住所有事实”，而是学习一个评分函数：真三元组分数高，假三元组分数低。于是即使某条边没有明确写进图谱，模型也能根据几何结构补全它。这类任务通常叫 link prediction，白话说，就是“补全缺失关系或缺失实体”。

最常见的三类思路分别是：

| 模型 | 核心几何操作 | 适合的直观理解 | 主要优点 | 主要短板 |
| --- | --- | --- | --- | --- |
| TransE | 平移 | 从头实体沿关系向量走到尾实体 | 简单、快、参数少 | 难处理复杂关系模式 |
| RotatE | 复数旋转 | 头实体在复平面按关系角度旋转到尾实体 | 能处理对称、反对称、组合关系 | 训练复杂度高于 TransE |
| ConvE | 卷积交互 | 把实体和关系拼起来做局部特征提取 | 表达力更强 | 训练和推理更重 |

玩具例子先看 TransE。若已知“Bill Gates 创立 微软”，模型希望满足：

$$
\mathbf{e}_{BillGates} + \mathbf{r}_{创立} \approx \mathbf{e}_{微软}
$$

其评分函数常写成：

$$
f(h,r,t)=-\|\mathbf{e}_h+\mathbf{r}-\mathbf{e}_t\|
$$

距离越小，分数越高，表示三元组越可信。对初学者来说，可以把关系向量看成一支箭头：它把头实体推向尾实体。

结论很直接：如果你的图谱关系比较简单、规模很大、推理速度重要，先看 TransE；如果关系里有明显的方向性、对称性、组合性，RotatE 更稳；如果你愿意用更高算力换更强交互表达，可以考虑 ConvE。

---

## 问题定义与边界

问题可以正式写成：给定知识图谱中的正例三元组集合 $\mathcal{G}=\{(h,r,t)\}$，学习实体嵌入 $\mathbf{e}$ 和关系嵌入 $\mathbf{r}$，使评分函数 $f(h,r,t)$ 对正例高、对负例低。

这里的 embedding，白话说，就是“把离散对象压缩成可计算的连续向量”。图谱中的实体本来只是字符串，如“微软”“比尔盖茨”“美国”，模型无法直接做几何运算，所以先映射到低维空间。

这个问题的边界要说清楚：

1. 它解决的是“结构化三元组补全”，不是完整自然语言理解。
2. 它更擅长利用图中的连接结构，不一定擅长处理长文本描述。
3. 它通常学习的是统计规律，不保证逻辑上绝对正确。
4. 它主要做单跳关系建模，多跳推理往往需要额外机制。

新手可以把知识图谱想成“点和箭头组成的有向图”。表示学习不是把图画出来，而是把每个点和每种箭头都放进一个向量空间里，让“应该相连的点”在几何上满足某种规则。

一个更具体的任务定义如下：

| 输入 | 输出 | 目标 |
| --- | --- | --- |
| 头实体 $h$、关系 $r$、尾实体 $t$ | 一个分数 $f(h,r,t)$ | 正例分数高于负例 |
| 已知部分图谱 | 缺失实体或缺失关系 | 做补全与排序 |
| 大量离散符号 | 稠密向量表示 | 提升泛化能力 |

三种模型在这个任务中的位置可以概括为：

| 模型 | 输入三元组 | 评分函数思路 | 优化目标 |
| --- | --- | --- | --- |
| TransE | $(h,r,t)$ | 看 $\mathbf{e}_h+\mathbf{r}$ 与 $\mathbf{e}_t$ 是否接近 | 拉近正例、推远负例 |
| RotatE | $(h,r,t)$ | 看 $\mathbf{e}_h$ 经旋转后是否到达 $\mathbf{e}_t$ | 利用相位建模关系模式 |
| ConvE | $(h,r,t)$ | 从 $h,r$ 的局部交互中提特征，再匹配 $t$ | 学更复杂非线性交互 |

边界还体现在关系类型上。若图谱主要是一对一或接近平移结构的关系，TransE 往往足够；若存在一对多、多对多、反向关系、组合路径，简单平移就容易失真。再往上，如果实体还有丰富文本属性、时间信息、上下文信息，仅靠这类静态 embedding 模型通常不够。

---

## 核心机制与推导

### 1. TransE：把关系当作位移

TransE 的假设最简单：关系就是向量空间中的平移。

$$
\mathbf{e}_h+\mathbf{r}\approx\mathbf{e}_t
$$

对应评分函数：

$$
f(h,r,t)=-\|\mathbf{e}_h+\mathbf{r}-\mathbf{e}_t\|_p
$$

其中 $\|\cdot\|_p$ 常取 $L_1$ 或 $L_2$ 范数。范数，白话说，就是“向量偏差有多大”的度量。

玩具例子：

设

$$
\mathbf{e}_h=[1,0],\quad \mathbf{r}=[0.5,0.5]
$$

那么理想尾实体应为：

$$
\mathbf{e}_t=[1.5,0.5]
$$

因为

$$
\|\mathbf{e}_h+\mathbf{r}-\mathbf{e}_t\|
= \|[1,0]+[0.5,0.5]-[1.5,0.5]\|
= 0
$$

这表示完全匹配。训练时，正样本希望接近 0，负样本则希望更远。

常用损失是 margin ranking loss：

$$
\mathcal{L}=\sum_{(h,r,t)\in \mathcal{G}}\sum_{(h',r,t')\in \mathcal{G}^-}
\max\big(0,\gamma + d(h,r,t)-d(h',r,t')\big)
$$

其中 $\gamma$ 是间隔，白话说，就是“正例至少要比负例好多少”；$\mathcal{G}^-$ 是负样本集合，通常通过替换头实体或尾实体构造。

TransE 的强项是简单，但它对复杂关系模式支持有限。比如对称关系“夫妻”，若 $(A,夫妻,B)$ 成立，则 $(B,夫妻,A)$ 也应成立。仅靠一个固定平移向量，很难同时满足两个方向都低距离。

### 2. RotatE：把关系当作复平面的旋转

RotatE 在复数空间里表示实体和关系。复数，白话说，就是带有实部和虚部的数，可以自然表示“长度和角度”。

核心假设是：

$$
\mathbf{e}_h \circ \mathbf{r} \approx \mathbf{e}_t,\quad |r_i|=1
$$

这里 $\circ$ 是逐元素乘法，$|r_i|=1$ 表示每一维关系只做旋转、不改长度。于是关系的本质变成“每一维转一个角度”。

评分函数常写为：

$$
f(h,r,t)=-\|\mathbf{e}_h\circ\mathbf{r}-\mathbf{e}_t\|_1
$$

为什么它能处理更多关系模式？因为角度运算天然适合表示方向和组合。

- 对称关系：旋转角度接近 $0$ 或 $\pi$ 时，前后关系更容易满足互换约束。
- 反对称关系：例如“父亲”与“孩子”不能互换，旋转方向可以区分顺序。
- 组合关系：若 $r_3=r_1\circ r_2$，则连续两次旋转可表示路径组合。

例如，“国家-首都-城市”和“城市-位于-国家”往往是反向但不等价的关系，RotatE 可以通过不同相位建模这种方向性；TransE 则容易把它们压成近似的平移模式。

### 3. ConvE：用卷积显式建交互

ConvE 的思路不是直接规定某种几何变换，而是学习“头实体和关系如何局部组合”。卷积，白话说，就是用一个小窗口在矩阵上滑动，提取局部模式。

它通常先把 $\mathbf{e}_h$ 和 $\mathbf{e}_r$ reshape 成二维张量，拼接后做 2D 卷积，再通过全连接层与尾实体做匹配。形式上可写为：

$$
f(h,r,t)=g(\text{vec}(g([\hat{\mathbf{e}}_h;\hat{\mathbf{e}}_r]*\omega))W)\cdot \mathbf{e}_t
$$

其中：

- $[\hat{\mathbf{e}}_h;\hat{\mathbf{e}}_r]$ 是拼接后的二维输入；
- $*$ 表示卷积；
- $\omega$ 是卷积核；
- $W$ 是线性变换；
- $g$ 是非线性激活函数。

它的价值在于：不再假设“关系一定是平移”或“关系一定是旋转”，而是通过卷积去学习更复杂的局部交互特征。因此表达力通常强于 TransE，但代价是训练更重、调参更难。

三者对关系模式的支持可以总结为：

| 关系模式 | 例子 | TransE | RotatE | ConvE |
| --- | --- | --- | --- | --- |
| 一对一 | 出生地 | 较好 | 较好 | 较好 |
| 对称 | 夫妻、相邻 | 差 | 好 | 通常可学到 |
| 反对称 | 父亲、雇佣 | 弱 | 好 | 通常可学到 |
| 组合关系 | 省份+国家 | 弱 | 好 | 可学但依赖数据 |
| 多对多 | 参与、合作 | 一般 | 较好 | 较好 |

真实工程例子：在企业问答系统中，图谱里可能有“产品-属于-品类”“品类-归属-事业部”“客户-购买-产品”等关系。若用户问“某客户主要关联哪个事业部”，系统往往先做实体识别，再用 KG embedding 补全候选边，最后排序输出。此时若关系包含明显方向性和组合性，RotatE 往往比 TransE 更稳；若还要吸收实体与关系的高阶交互，ConvE 常更强，但资源消耗更大。

---

## 代码实现

下面先给一个最小可运行的 Python 版本，用纯 `numpy` 演示 TransE 的评分和 margin loss。它不是完整训练器，但逻辑与实际框架一致。

```python
import numpy as np

def l2_score(h, r, t):
    # 分数越大越好，这里取负距离
    return -np.linalg.norm(h + r - t, ord=2)

def margin_ranking_loss(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t, margin=1.0):
    pos_dist = np.linalg.norm(pos_h + pos_r - pos_t, ord=2)
    neg_dist = np.linalg.norm(neg_h + neg_r - neg_t, ord=2)
    return max(0.0, margin + pos_dist - neg_dist)

# 玩具正样本: h + r == t
h = np.array([1.0, 0.0])
r = np.array([0.5, 0.5])
t = np.array([1.5, 0.5])

# 负样本: 换一个错误尾实体
t_neg = np.array([0.2, 1.8])

pos_score = l2_score(h, r, t)
neg_score = l2_score(h, r, t_neg)
loss = margin_ranking_loss(h, r, t, h, r, t_neg, margin=1.0)

assert np.isclose(pos_score, 0.0)
assert pos_score > neg_score
assert loss == 0.0 or loss > 0.0

print("positive score:", pos_score)
print("negative score:", neg_score)
print("loss:", loss)
```

这段代码说明三件事：

1. 正样本的 $h+r-t$ 越接近 0，分数越高。
2. 负样本通过替换头或尾实体构造。
3. 训练不是单看正样本距离，而是要求“正样本比负样本更好一截”。

如果换成 PyTorch 风格，核心结构通常如下：

```python
import torch
import torch.nn as nn

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.entity = nn.Embedding(num_entities, dim)
        self.relation = nn.Embedding(num_relations, dim)
        nn.init.uniform_(self.entity.weight, -0.1, 0.1)
        nn.init.uniform_(self.relation.weight, -0.1, 0.1)

    def score(self, h_idx, r_idx, t_idx):
        h = self.entity(h_idx)
        r = self.relation(r_idx)
        t = self.entity(t_idx)
        return -torch.norm(h + r - t, p=2, dim=-1)

    def forward(self, pos_triplets, neg_triplets, margin=1.0):
        pos_score = self.score(pos_triplets[:, 0], pos_triplets[:, 1], pos_triplets[:, 2])
        neg_score = self.score(neg_triplets[:, 0], neg_triplets[:, 1], neg_triplets[:, 2])
        loss = torch.relu(margin - pos_score + neg_score).mean()
        return loss

# 形状检查
model = TransE(num_entities=10, num_relations=3, dim=8)
pos = torch.tensor([[0, 1, 2]])
neg = torch.tensor([[0, 1, 3]])
loss = model(pos, neg)
assert loss.ndim == 0
```

如果从工程流程看，三类模型大致共享同一套训练骨架：

| 步骤 | 做什么 | 原因 |
| --- | --- | --- |
| 读取三元组 | 把 `posts` 类似的离散 ID 转为索引 | Embedding 层需要整数编号 |
| 初始化向量 | 实体表、关系表随机初始化 | 作为可训练参数 |
| 负采样 | 替换头或尾构造假样本 | 让模型学会区分真伪 |
| 计算评分 | TransE/RotatE/ConvE 各自不同 | 这是模型差异核心 |
| 计算损失 | margin loss 或 logistic loss | 优化排序能力 |
| 更新参数 | SGD/Adam | 迭代逼近更优几何结构 |

换模型时真正变化的只有“评分函数”。

- TransE：`h + r - t`
- RotatE：把实体拆成实部和虚部，关系变成单位模复数，相当于逐维旋转
- ConvE：先把 `h` 和 `r` reshape，再卷积、激活、线性投影，最后与 `t` 匹配

所以从代码维护角度看，最合理的做法通常是把“数据管道、负采样、训练循环”抽象为通用模块，把“score function”做成可插拔组件。

---

## 工程权衡与常见坑

选模型不能只看论文分数，要看关系模式、延迟预算、显存预算、训练集规模。

先看主要权衡：

| 模型 | 参数规模 | 训练速度 | 推理速度 | 表达能力 | 典型使用建议 |
| --- | --- | --- | --- | --- | --- |
| TransE | 低 | 快 | 快 | 中低 | 大规模、低延迟基线 |
| RotatE | 中 | 中 | 中 | 高 | 关系模式复杂时优先 |
| ConvE | 中高 | 较慢 | 较慢 | 高 | 追求精度且算力充足 |

常见坑主要有以下几类。

第一，TransE 对复杂关系会塌缩。  
例如一个头实体对应多个尾实体时，单一平移向量会把不同尾实体往同一位置推，导致表达冲突。典型表现是训练集损失下降，但验证集排名指标上不去。

第二，负采样做得太粗。  
如果随便替换尾实体，容易生成“假负样本其实是真样本”的情况。例如“北京-位于-中国”替成“上海-位于-中国”并不是负样本。更稳的做法是使用 filtered setting，白话说，就是“评估时把图谱里已知为真的候选排除掉”。

第三，RotatE 的约束没处理好。  
若关系向量不保持单位模，模型就不再是纯旋转，表达会漂移。实现中通常需要把关系参数转成相位或显式归一化。

第四，ConvE 不是“换个层就一定更强”。  
它对 batch 组织、1-N 评分、卷积输入形状、dropout 比例比较敏感。1-N scoring，白话说，就是“一次拿一个 $(h,r)$ 去匹配所有尾实体”，这样能提高训练效率，但也更吃显存。

第五，评估指标理解错误。  
知识图谱补全常看 Hits@K、MRR。MRR 是 mean reciprocal rank，白话说，就是“正确答案排得越靠前，分数越高”。仅看训练 loss 很容易误判模型质量。

真实工程例子：一个智能客服系统会把“用户-购买-产品”“产品-属于-品牌”“品牌-归属-事业群”等边组织成知识图谱。当用户输入“这个用户大概率属于哪个事业群”，系统会先做实体链接，再用 RotatE 或 ConvE 对候选事业群打分。若线上要求 20ms 内返回，ConvE 可能太重，实际会退回 RotatE 或经过蒸馏的轻量模型。也就是说，离线最优模型不一定适合在线服务。

规避建议可以整理成表：

| 坑 | 现象 | 规避方式 |
| --- | --- | --- |
| TransE 无法处理复杂模式 | 对称/反对称关系表现差 | 换 RotatE 或 ComplEx |
| 假负样本污染 | 训练不稳定、评估失真 | 使用更强负采样和 filtered evaluation |
| 向量无约束发散 | 分数不稳定 | 做归一化或相位参数化 |
| ConvE 调参困难 | 方差大、复现难 | 固定输入形状，逐步调 dropout 和 batch |
| 线上推理太慢 | 排序延迟高 | 做 ANN 检索、候选截断或模型蒸馏 |

---

## 替代方案与适用边界

TransE、RotatE、ConvE 都属于“基于打分函数的 embedding 模型”，但不是唯一选择。

### 1. DistMult 与 ComplEx

DistMult 是双线性模型，白话说，就是让头实体、关系、尾实体逐维相乘再求和。它简单有效，但天然偏向对称关系，因为交换头尾后形式接近不变。

ComplEx 把 DistMult 扩展到复数空间，可以更好处理反对称关系，是 RotatE 之外一个常见强基线。

举个简化对比：

- 对称关系“夫妻”：DistMult、RotatE 都可能较好处理。
- 反对称关系“父亲”：DistMult 往往吃亏，ComplEx、RotatE 更合适。
- 组合关系“省会所在省份 + 省份所在国家 = 省会所在国家”：RotatE 通常比 DistMult 更自然。

### 2. 图神经网络方法

R-GCN 这类方法属于图神经网络。图神经网络，白话说，就是“让节点反复聚合邻居信息后再做预测”。它不只依赖一个静态向量表，而是显式利用邻居传播。

当图谱里有丰富实体属性、多跳依赖、局部子图结构很重要时，GNN 方法往往优于纯 embedding 模型。但它的代价是训练更重、实现更复杂、在线部署更难。

### 3. 语言模型结合图谱

如果实体有大量文本描述，例如商品介绍、论文摘要、百科条目，纯图结构模型的信息不够。这时常见做法是把预训练语言模型和图谱表示结合起来，让文本语义参与建模。

但边界也很清楚：一旦引入大模型，成本、稳定性、可解释性都会变差。对于只需要做结构补全的传统场景，不一定划算。

可以用下表做决策：

| 方案 | 支持的关系类型 | 典型应用场景 | 何时使用 |
| --- | --- | --- | --- |
| TransE | 简单平移型关系 | 大规模图谱补全、召回基线 | 速度和规模优先 |
| RotatE | 对称、反对称、组合 | 问答补全、复杂关系预测 | 关系模式复杂 |
| ConvE | 高阶非线性交互 | 离线高精度排序 | 算力充足且追求效果 |
| ComplEx | 对称与反对称 | 通用关系建模基线 | 需要稳定强基线 |
| R-GCN | 多跳与邻域传播 | 属性丰富的图推理 | 结构上下文重要 |
| LM+KG | 文本与图联合 | 搜索、推荐、问答融合 | 文本信息占比高 |

因此，适用边界可以压缩成一句话：  
如果关系结构简单且图很大，先用轻量 embedding；如果关系模式复杂，优先 RotatE/ComplEx；如果需要多跳传播或实体属性，转向 GNN；如果文本本身是主要信息源，再考虑语言模型结合图谱。

---

## 参考资料

1. [Knowledge graph embedding - Wikipedia](https://en.wikipedia.org/wiki/Knowledge_graph_embedding)  
适合新手。用于建立整体概念：什么是知识图谱表示学习、常见任务是什么。

2. Bordes et al., *Translating Embeddings for Modeling Multi-relational Data (TransE)*  
适合进阶。用于理解 $h+r\approx t$、评分函数和 margin loss 的原始设计。

3. Sun et al., *RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space*  
适合进阶。用于理解复数旋转、对称/反对称/组合关系建模。

4. Dettmers et al., *Convolutional 2D Knowledge Graph Embeddings (ConvE)*  
适合进阶。用于理解卷积交互、1-N 训练与更强表达力的来源。
