## 核心结论

RotatE 是一种知识图谱嵌入模型。知识图谱嵌入，白话说，就是把“实体和关系”压缩成可计算的向量，方便模型做预测。它的核心不是把关系看成“位移”，而是把关系看成复数空间中的“逐维旋转”，公式写成：

$$
t \approx h \odot r,\quad |r_i| = 1
$$

这里 $h,t,r \in \mathbb{C}^d$，$\odot$ 是逐元素乘法，意思是每一维单独相乘；$|r_i|=1$ 表示关系向量的每一维都落在复平面的单位圆上，也就是“只转角度，不改长度”。

这件事的价值在于：很多知识图谱关系，本质上不是简单加减法，而是有方向、有可逆性、有组合结构。RotatE 用相位旋转来表达这些结构，比 TransE 这类平移模型更自然。尤其是下面四类关系模式，RotatE 通常更占优势：

| 关系模式 | 含义 | RotatE 是否天然适合 |
| --- | --- | --- |
| 对称关系 | `married_to(A,B)` 往往也有 `married_to(B,A)` | 是 |
| 反对称关系 | `parent_of(A,B)` 不应推出 `parent_of(B,A)` | 是 |
| 逆关系 | `parent_of` 与 `child_of` 互为逆 | 是 |
| 组合关系 | `located_in(city,state)` + `located_in(state,country)` | 是 |

和 TransE 的最简对比如下：

| 模型 | 关系形式 | 核心公式 | 结构表达力 |
| --- | --- | --- | --- |
| TransE | 平移 | $h + r \approx t$ | 对逆/对称/组合支持有限 |
| RotatE | 旋转 | $h \odot r \approx t$ | 对复杂关系模式更灵活 |

一个玩具例子可以直接看出它的几何含义。设：

$$
h = 1 + 0i,\quad r = i = e^{i\pi/2},\quad t = 0 + 1i
$$

那么：

$$
h \odot r = (1+0i)\cdot i = i = t
$$

也就是把点 $(1,0)$ 在复平面上旋转 $90^\circ$，刚好转到 $(0,1)$。如果逆关系用 $r^{-1}=-i$ 表示，那么：

$$
t \odot r^{-1} = i\cdot(-i)=1=h
$$

这就是“转过去”和“转回来”的关系。

---

## 问题定义与边界

RotatE 解决的是知识图谱补全问题。知识图谱补全，白话说，就是图里有些边缺了，模型要把缺失的边补出来。训练样本写成三元组：

$$
(h,r,t)
$$

其中 $h$ 是头实体，$r$ 是关系，$t$ 是尾实体。例如：

- `(Alice, works_for, OpenAI)`
- `(Shenzhen, located_in, China)`
- `(TensorFlow, developed_by, Google)`

在预测阶段，常见任务有两种：

1. 给定 $(h,r,?)$，预测最可能的尾实体。
2. 给定 $(?,r,t)$，预测最可能的头实体。

例如在企业知识图谱里：

- `works_for(Alice, ?)`
- `subsidiary_of(CompanyA, ?)`
- `located_in(DeptX, ?)`  

这些都是典型的链接预测任务。链接预测，白话说，就是预测图中哪条连接最可能存在。

RotatE 的问题边界也要讲清楚。它擅长的是“关系结构建模”，不是“所有知识推理问题”。下面这个表更直接：

| 能做什么 | 不能直接做好什么 |
| --- | --- |
| 链接预测 | 从长文本里抽事实 |
| 建模逆关系、对称关系、组合关系 | 处理明显时间变化的动态图谱 |
| 给缺失边打分排序 | 自动解释为什么这条边成立 |
| 作为召回模块提供候选事实 | 代替规则系统做严格逻辑证明 |

这意味着，如果你的输入已经是干净的三元组图，RotatE 很合适；如果你的问题是“先从文档抽实体关系，再做图推理”，那 RotatE 只负责后半段。

还要注意一点：RotatE 不是多跳推理系统本身。它能通过嵌入空间部分表达组合关系，但这不等于它能像符号逻辑那样显式地列出完整推理链。工程里常见做法是：先用 RotatE 做高效候选召回，再把候选结果交给规则引擎、重排序模型或人工审核。

---

## 核心机制与推导

RotatE 的关键设定是：实体和关系都表示为复数向量。复数，白话说，就是由实部和虚部组成的数，比如 $a+bi$。复数的一个重要特性是，它非常适合表示“旋转”。

根据欧拉公式：

$$
e^{i\theta} = \cos \theta + i\sin \theta
$$

如果某一维关系写成：

$$
r_i = e^{i\theta_i},\quad |r_i|=1
$$

那它的含义就是“在该维把实体旋转 $\theta_i$ 角度”。因此整体关系建模为：

$$
t \approx h \odot r
$$

展开到每一维，就是：

$$
t_i \approx h_i r_i
$$

评分时，RotatE 不直接判断“等不等”，而是计算距离：

$$
d(h,r,t)=\|h\odot r - t\|
$$

再把距离转成分数：

$$
score(h,r,t)=\gamma - d(h,r,t)
$$

这里 $\gamma$ 是 margin，白话说，就是人为设定的分数上界缓冲区。距离越小，分数越高，说明三元组越像真事实。

还是先看玩具例子。设一维情况：

$$
h = 1+0i,\quad r=e^{i\pi/2}=i,\quad t=i
$$

则：

$$
h\odot r = i
$$

所以：

$$
d(h,r,t)=|i-i|=0
$$

这说明这个三元组是一个理想正样本。

为什么 RotatE 能表达多种关系模式？关键在相位，也就是旋转角度。

| 模式 | 数学条件 | 直观解释 |
| --- | --- | --- |
| 对称关系 | $r_i \approx 1$ 或 $r_i \approx -1$ | 旋转后还能回到相同方向或镜像方向 |
| 反对称关系 | 相位不能同时满足双向一致 | 从 A 到 B 可行，不代表 B 到 A 也可行 |
| 逆关系 | $r^{-1}=\bar r$，即相位取反 | 正向转过去，反向再转回来 |
| 组合关系 | $r_3 \approx r_1 \odot r_2$ | 连续旋转等于一次合成旋转 |

这里 $\bar r$ 表示共轭。共轭，白话说，就是虚部变号。因为单位模复数的逆恰好等于共轭，所以逆关系表达会非常自然。

举个稍微真实一点的工程例子。假设企业图谱里有：

- `(Alice, works_in, TeamX)`
- `(TeamX, belongs_to, CompanyA)`

如果模型学到：

$$
r_{\text{works\_for}} \approx r_{\text{works\_in}} \odot r_{\text{belongs\_to}}
$$

那么即使训练集中没有显式出现 `(Alice, works_for, CompanyA)`，模型也可能给它较高分。这就是组合关系带来的补全能力。

论文里另一个关键点是 self-adversarial negative sampling。负采样，白话说，就是人为制造假三元组，让模型学会区分真和假。RotatE 不只是随机采负样本，而是给“更像真的假样本”更高权重。形式上可写成：

$$
p_i=\frac{\exp(\alpha s_i)}{\sum_j \exp(\alpha s_j)}
$$

其中 $s_i$ 是负样本分数，$\alpha$ 控制权重尖锐程度。它的直觉很简单：太假的负样本训练价值低，模型应该把更多精力花在“难负样本”上。

---

## 代码实现

实现 RotatE 时，通常不直接用复数张量，而是把每个向量拆成实部和虚部。这样做更稳定，也更方便和常规深度学习框架对接。

设：

- `h_re, h_im`：头实体实部与虚部
- `t_re, t_im`：尾实体实部与虚部
- `r_phase`：关系相位
- `r_re = cos(r_phase)`
- `r_im = sin(r_phase)`

根据复数乘法：

$$
(a+bi)(c+di)=(ac-bd)+(ad+bc)i
$$

可得：

$$
\text{Re}(h\odot r)=h_{re}\cdot r_{re}-h_{im}\cdot r_{im}
$$

$$
\text{Im}(h\odot r)=h_{re}\cdot r_{im}+h_{im}\cdot r_{re}
$$

下面给一个可运行的最小 Python 例子，不依赖深度学习框架，先把机制跑通：

```python
import math

def rotate_score(h_re, h_im, r_phase, t_re, t_im, gamma=6.0):
    assert len(h_re) == len(h_im) == len(r_phase) == len(t_re) == len(t_im)
    dist_sq = 0.0
    for a, b, theta, c, d in zip(h_re, h_im, r_phase, t_re, t_im):
        r_re = math.cos(theta)
        r_im = math.sin(theta)

        hr_re = a * r_re - b * r_im
        hr_im = a * r_im + b * r_re

        dist_sq += (hr_re - c) ** 2 + (hr_im - d) ** 2

    dist = math.sqrt(dist_sq)
    return gamma - dist

# 玩具例子：1 旋转 90 度得到 i
h_re = [1.0]
h_im = [0.0]
r_phase = [math.pi / 2]
t_re = [0.0]
t_im = [1.0]

score_pos = rotate_score(h_re, h_im, r_phase, t_re, t_im)
score_neg = rotate_score(h_re, h_im, r_phase, [1.0], [0.0])

assert score_pos > score_neg
assert abs(score_pos - 6.0) < 1e-8
print("RotatE toy example passed.")
```

如果用 PyTorch，核心计算通常长这样：

```python
import torch

def rotate_distance(h_re, h_im, r_phase, t_re, t_im):
    r_re = torch.cos(r_phase)
    r_im = torch.sin(r_phase)

    hr_re = h_re * r_re - h_im * r_im
    hr_im = h_re * r_im + h_im * r_re

    diff_re = hr_re - t_re
    diff_im = hr_im - t_im
    dist = torch.sqrt(diff_re.pow(2) + diff_im.pow(2)).sum(dim=-1)
    return dist

def loss_fn(pos_score, neg_score, neg_weight):
    pos_loss = -torch.log(torch.sigmoid(pos_score)).mean()
    neg_loss = -(neg_weight * torch.log(torch.sigmoid(-neg_score))).sum(dim=-1).mean()
    return pos_loss + neg_loss
```

一个典型训练流程可以压缩成下面几步：

| 步骤 | 内容 |
| --- | --- |
| 1 | 读取三元组 `(h,r,t)` |
| 2 | 查表取出实体实部/虚部与关系相位 |
| 3 | 计算正样本分数 |
| 4 | 构造负样本，通常替换头实体或尾实体 |
| 5 | 对负样本打分，并按 self-adversarial 权重加权 |
| 6 | 计算损失并反向传播 |
| 7 | 评估 `MRR`、`Hits@1/3/10` |

真实工程例子里，企业主数据图谱常有 `parent_company_of`、`subsidiary_of`、`employs`、`works_for` 等关系。此时 RotatE 可以作为候选补全模块：先从百万级实体中为每条缺边召回 top-K 候选，再交给规则校验，例如“公司状态必须为存续”“国家字段必须一致”。这样做的价值是，RotatE 负责结构召回，规则系统负责业务约束，职责清晰。

---

## 工程权衡与常见坑

RotatE 在论文和基准集上表现不错，但落地时坑不少。核心问题不是“公式会不会写”，而是“约束、采样、评估、数据集质量”是否处理对了。

先看最常见的问题表：

| 问题现象 | 原因 | 规避办法 |
| --- | --- | --- |
| 训练能收敛，但效果不稳 | 负样本太容易 | 用 self-adversarial negative sampling |
| 模型解释失真 | 没有保持 $|r_i|=1$ | 关系用相位参数化，而不是直接学复数值 |
| 线下指标高，线上泛化差 | 数据有逆关系泄漏 | 优先看 `WN18RR`、`FB15k-237` 这类去泄漏数据 |
| 命中率高但业务可用性低 | 只看 raw setting | 使用 filtered setting，并做人审抽检 |
| 参数量大、训练慢 | 维度和负采样数过高 | 从中等维度起步，逐步扩容 |
| 分数差异不明显 | margin 或初始化不合适 | 调整 `gamma`、初始化范围和学习率 |

这里有几个点必须展开。

第一，关系模长约束不能丢。RotatE 的解释成立，依赖于关系是“单位模复数”。如果你直接让关系实部虚部自由训练，模型会退化成“旋转加缩放”的混合形式。这样不是一定不能用，但那已经偏离了 RotatE 设计的核心优势，逆关系和组合关系的几何解释会变弱。

第二，评估一定要看 filtered setting。filtered，白话说，就是评估一个候选答案时，把训练集、验证集、测试集中已知的其他真答案过滤掉。原因很简单：知识图谱经常一问多答。比如 `(Beijing, located_in, China)` 和 `(Beijing, capital_of, China)` 不是同一关系，但有些关系本身就会出现多个真尾实体。如果不做过滤，模型可能被“明明答对了另一个真答案”却算错。

第三，基准集泄漏问题不能忽视。早期数据集如 `WN18`、`FB15k` 里存在明显 inverse leakage。leakage，白话说，就是测试答案在训练结构里被“暗中提示”了。比如训练集有 `(B, child_of, A)`，测试集问 `(A, parent_of, ?)`，模型几乎只要学会“翻边”就能答对。这会把很多模型的指标抬得很夸张，但不代表真实推理能力强。更可信的选择通常是 `WN18RR` 和 `FB15k-237`。

工程上可以先用一套保守配置起步：

| 项目 | 建议起点 |
| --- | --- |
| embedding 维度 | 256 或 512 |
| 负样本数 | 64 到 256 |
| 学习率 | `1e-4` 到 `5e-4` |
| `gamma` | 6 到 24，按数据集调 |
| batch size | 按显存压到稳定吞吐 |
| 评估指标 | `MRR`、`Hits@1/3/10`、filtered |

另一个常见误区是把 RotatE 当成“比 TransE 高级，所以一定更好”。这不成立。如果你的图谱关系很简单，主要就是层级和平移模式，RotatE 可能没有明显收益，反而训练更复杂。模型选择不能只看论文排名，要看你的关系模式分布。

---

## 替代方案与适用边界

RotatE 很强，但不是默认答案。选型时更重要的是“你的图里主要是什么关系结构”。

下面把几个经典模型放在一起比较：

| 模型 | 核心公式 | 逆关系 | 对称关系 | 组合关系 | 实现复杂度 | 更适合什么场景 |
| --- | --- | --- | --- | --- | --- | --- |
| TransE | $h+r\approx t$ | 一般 | 一般 | 一般 | 低 | 简单关系、快速基线 |
| DistMult | 三线性打分 | 弱 | 强 | 弱 | 低 | 对称关系较多的图 |
| ComplEx | 复数双线性 | 强 | 强 | 一般 | 中 | 需要逆关系表达且实现不能太复杂 |
| RotatE | $h\odot r\approx t$ | 强 | 强 | 强 | 中 | 复杂关系模式较多的图 |

TransE 的优点是直观、快、易调。对于“组织架构、地理层级、关系类型有限”的图，它经常已经够用。DistMult 最大的问题是对称性偏强，容易把本不对称的关系学坏。ComplEx 也在复数空间里工作，对逆关系支持不错，但它的关系含义更偏双线性交互，不像 RotatE 那样有明确的“旋转几何解释”。

真实工程里可以这么分：

- 如果你要先做一个强基线，选 TransE。
- 如果你确认图里逆关系、反对称关系、组合关系很多，优先试 RotatE。
- 如果你还要融合文本编码器、规则系统或图神经网络，RotatE 更适合作为其中一个结构模块，而不是整套系统。

再说适用边界。RotatE 不擅长的情况主要有三类：

1. 文本主导任务  
如果实体和关系本身语义主要来自长文本，例如法律条款、客服工单、论文摘要，单纯三元组结构不够，往往要结合 BERT 类编码器。

2. 强时序任务  
如果事实会随时间变化，比如“CEO 是谁”“某公司何时隶属谁”，静态 RotatE 不够，需要时序知识图谱模型。

3. 强规则约束任务  
如果你要的是可审计的严格推理，如金融合规、医学规则校验，仅靠嵌入分数不够，需要规则引擎或逻辑系统补充。

一句话概括：RotatE 最适合“结构模式复杂、数据是图、目标是补边排序”的任务；一旦任务重心转向文本、时间或强规则，它就不再是主角。

---

## 参考资料

1. [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://arxiv.org/abs/1902.10197)
2. [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space - OpenReview](https://openreview.net/forum?id=HkgEQnRqYQ)
3. [RotatE ar5iv HTML 版](https://ar5iv.labs.arxiv.org/html/1902.10197)
4. [DeepGraphLearning/KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)
5. [WN18RR 数据集说明](https://paperswithcode.com/dataset/wn18rr)
6. [FB15k-237 数据集说明](https://paperswithcode.com/dataset/fb15k-237)
