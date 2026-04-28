## 核心结论

ComplEx 是一种知识图谱嵌入模型。知识图谱嵌入，白话说，就是把“实体”和“关系”压成可计算的向量，让模型用分数判断一个三元组是否合理。它的关键做法不是把向量放在实数空间 $\mathbb{R}^d$，而是放在复数空间 $\mathbb{C}^d$，并用

$$
\phi(s,r,o)=\Re\!\left(\langle e_s,w_r,\bar e_o\rangle\right)
$$

给三元组 $(s,r,o)$ 打分。

这里的 $\Re(\cdot)$ 是“取实部”，$\bar e_o$ 是“共轭”，白话说就是把尾实体向量里的虚部符号翻转。这个小改动很关键，因为它让模型天然有能力表达方向性：`parent_of(A, B)` 和 `parent_of(B, A)` 可以得到不同分数。

如果只看直觉，可以把 DistMult 理解成“更擅长判断像不像”，把 ComplEx 理解成“既判断像不像，也判断方向对不对”。

| 模型 | 是否容易表达对称关系 | 是否容易表达反对称关系 |
|---|---:|---:|
| DistMult | 强 | 弱 |
| ComplEx | 强 | 强 |

ComplEx 的价值不在于“永远最好”，而在于它用相对简单的打分函数，同时覆盖了对称关系和反对称关系这两类常见模式。在 `friend_of`、`similar_to` 这类近似对称关系上，它不会明显吃亏；在 `parent_of`、`works_at`、`acquired_by` 这类强方向关系上，它比纯对称双线性模型更合适。

---

## 问题定义与边界

ComplEx 解决的是知识图谱补全。知识图谱补全，白话说，就是已知很多结构化事实后，让模型给“缺失的一格”做排序预测。

典型输入不是一句自然语言，而是三元组：

- `(张三, works_at, 字节跳动)`
- `(阿司匹林, has_active_ingredient, 乙酰水杨酸)`
- `(微软, acquired, GitHub)`

推理时，常见任务有两种：

$$
(s,r,?) \quad\text{或}\quad (?,r,o)
$$

模型并不是直接“生成答案”，而是对候选实体逐个算分：

$$
\text{score}(s,r,o)\rightarrow \text{用于排序}
$$

分数越高，表示这个候选实体越可能成立。

一个玩具例子：

- 已知 `(小王, likes, 游戏)` 成立
- 已知 `(游戏, related_to, 编程)` 成立
- 现在问 `(小王, likes, ?)`，候选有 `游戏 / 编程 / 数据库`

这时模型要做的不是“写一句解释”，而是给三个候选打分并排序。

一个真实工程例子：

- 企业 CRM 图谱中有 `(员工, works_at, 公司)`
- 供应链图谱中有 `(公司, acquired_by, 公司)`
- 医药图谱中有 `(药品, has_active_ingredient, 成分)`

这些任务都不是开放域问答，而是结构化关系预测。只要实体集合是封闭的、关系是明确定义的，ComplEx 就有发挥空间。

| 适合的问题 | 不适合的问题 |
|---|---|
| 链接预测、候选实体排序 | 自由文本生成 |
| 有明确实体和关系的结构化图谱 | 没有图结构的纯文本问答 |
| 需要表达关系方向的补全任务 | 完全依赖长上下文语义推理的任务 |

边界也要说清楚。ComplEx 不负责“发现世界规律”的全部过程，它只是把已有图谱中的模式压进参数里。图谱本身噪声很大、关系定义混乱、候选实体没有收敛到固定集合时，模型分数再漂亮也很难稳定落地。

---

## 核心机制与推导

设头实体、关系、尾实体的嵌入分别为：

$$
e_s=a_s+i b_s,\quad w_r=c_r+i d_r,\quad e_o=a_o+i b_o
$$

其中 $a_s,b_s,c_r,d_r,a_o,b_o\in\mathbb{R}^d$。这句话的白话是：每个向量都拆成两半，一半是实部，一半是虚部。

尾实体取共轭后得到：

$$
\bar e_o=a_o-i b_o
$$

代入打分函数：

$$
\phi(s,r,o)=\Re\!\left(\sum_{k=1}^d e_s^k w_r^k \bar e_o^k\right)
$$

展开可得：

$$
\phi(s,r,o)=\sum_{k=1}^d
\Big(
a_s^k c_r^k a_o^k
+a_s^k d_r^k b_o^k
+b_s^k c_r^k b_o^k
-b_s^k d_r^k a_o^k
\Big)
$$

这四项可以直接理解成“两套坐标之间的交叉匹配”：

- $a_s c_r a_o$：实部之间的稳定匹配
- $a_s d_r b_o$：头实体实部与尾实体虚部的交叉作用
- $b_s c_r b_o$：虚部之间的稳定匹配
- $-b_s d_r a_o$：带符号的方向性交互

方向性主要来自“虚部参与时的符号变化”。这也是为什么 ComplEx 可以表达反对称关系，而 DistMult 不行。DistMult 的打分本质上对头尾交换更接近对称；ComplEx 因为用了共轭和虚部，交换头尾后分数一般会变。

一个最小数值例子，取 $d=1$：

$$
e_s=1+2i,\quad w_r=3+4i,\quad e_o=5+6i
$$

则

$$
\phi=\Re((1+2i)(3+4i)(5-6i))=35
$$

按展开式计算也是：

$$
1\cdot3\cdot5 + 1\cdot4\cdot6 + 2\cdot3\cdot6 - 2\cdot4\cdot5=35
$$

这说明“复数写法”和“实虚分拆写法”是完全一致的。

还可以看关系向量的不同形式：

| 关系向量形式 | 主要效果 |
|---|---|
| $w_r$ 纯实 | 更偏对称关系 |
| $w_r$ 纯虚 | 更偏反对称关系 |
| 实部和虚部都有 | 可同时表达两类模式 |

这不是说“纯实就一定对称、纯虚就一定反对称”，而是说模型的表达倾向会往这个方向走。工程上真正有价值的是：同一个模型类，不必为不同关系类型单独换架构。

---

## 代码实现

工程里通常不直接把参数存成“复数张量黑盒”，而是把实部和虚部分成两组实数向量。这样更容易初始化、正则化和调试。

下面是一个可运行的最小 Python 实现。它同时验证了两件事：

1. 展开式和复数公式等价  
2. 交换头尾后，分数通常会变化，说明模型能表达方向性

```python
def complex_score(s, r, o):
    # s, r, o are tuples: (real_list, imag_list)
    a_s, b_s = s
    c_r, d_r = r
    a_o, b_o = o

    total = 0.0
    for as_k, bs_k, cr_k, dr_k, ao_k, bo_k in zip(a_s, b_s, c_r, d_r, a_o, b_o):
        total += (
            as_k * cr_k * ao_k +
            as_k * dr_k * bo_k +
            bs_k * cr_k * bo_k -
            bs_k * dr_k * ao_k
        )
    return total


def complex_score_via_python_complex(s, r, o):
    s_complex = [complex(re, im) for re, im in zip(*s)]
    r_complex = [complex(re, im) for re, im in zip(*r)]
    o_complex = [complex(re, im) for re, im in zip(*o)]

    total = 0j
    for es, wr, eo in zip(s_complex, r_complex, o_complex):
        total += es * wr * eo.conjugate()
    return total.real


# 玩具例子：d = 1
s = ([1.0], [2.0])
r = ([3.0], [4.0])
o = ([5.0], [6.0])

score1 = complex_score(s, r, o)
score2 = complex_score_via_python_complex(s, r, o)

assert abs(score1 - 35.0) < 1e-9
assert abs(score1 - score2) < 1e-9

# 方向性检查：交换头尾后分数变化
score_forward = complex_score(s, r, o)
score_reverse = complex_score(o, r, s)

assert score_forward != score_reverse
print(score_forward, score_reverse)
```

如果换成 PyTorch 风格，核心前向传播通常就是下面这样：

```python
import torch

def complex_score_torch(e_re, e_im, r_re, r_im, s_idx, r_idx, o_idx):
    a_s = e_re[s_idx]
    b_s = e_im[s_idx]
    c_r = r_re[r_idx]
    d_r = r_im[r_idx]
    a_o = e_re[o_idx]
    b_o = e_im[o_idx]

    return (
        a_s * c_r * a_o +
        a_s * d_r * b_o +
        b_s * c_r * b_o -
        b_s * d_r * a_o
    ).sum(dim=-1)
```

训练时一般有三类输入：

- 正样本三元组：图谱中真实存在的事实
- 负样本三元组：把头实体或尾实体替换成错误候选得到的伪事实
- 损失函数：常见是 logistic loss、margin ranking loss 或带负采样的 softplus 形式

一个常见流程如下：

| 步骤 | 内容 |
|---|---|
| 1 | 按索引查出头实体、关系、尾实体的实部与虚部 |
| 2 | 用 ComplEx 公式计算正样本分数 |
| 3 | 构造负样本并计算负样本分数 |
| 4 | 通过损失函数拉高正样本、压低负样本 |
| 5 | 评估时对候选实体全量排序，计算 MRR、Hits@K |

真实工程例子里，假设你在做企业并购图谱：

- 已知 `(微软, acquired, GitHub)` 为正样本
- 负样本可构造为 `(微软, acquired, OpenAI)`、`(微软, acquired, Adobe)` 等
- 模型训练后，在 `(微软, acquired, ?)` 上应把更合理的候选排得更前

这类任务里，关系方向非常关键。`(GitHub, acquired, 微软)` 不只是“不太好”，而是语义上就错了。ComplEx 的优势恰好在这里。

---

## 工程权衡与常见坑

ComplEx 的数学定义不难，真正容易出错的是实现和评估。

最常见的坑是把公式写坏。比如少了 $\bar e_o$ 的共轭，方向性就会变弱；再比如只保留实部，模型会明显向 DistMult 退化。

| 坑点 | 后果 | 规避方式 |
|---|---|---|
| 少了共轭 $\bar e_o$ | 头尾交换差异变弱，方向性下降 | 严格按公式实现 |
| 只保留实部 | 表达能力接近纯实双线性模型 | 保持实部和虚部双通道 |
| 负采样过弱 | 模型只会记住高频模式 | 采用更强的尾实体/头实体替换策略 |
| 只看 raw MRR | 与论文结果不可直接比较 | 报告 filtered MRR |
| 横比不同数据集 | 结论失真 | 只在同一基准和相近协议下比较 |

另一个常见误区是误读论文数字。FB15k 和 FB15k-237 不是一回事。后者去掉了大量容易“投机”的逆关系，因此更能检验模型的真实泛化能力。直接拿两个数据集上的 MRR 横比，没有意义。

还要注意：ComplEx 的价值主要是表达能力，不是“无条件碾压所有基线”。一些后续汇总表中，FB15k-237 上常见引用结果大约是 ComplEx `MRR 0.247`、DistMult `0.241`，而某些训练设定下 TransE 也可能达到更高数值。这说明你不能把“用了复数”直接等价成“结果一定最好”。更稳妥的说法是：ComplEx 在不显著增加机制复杂度的前提下，更系统地解决了方向性建模问题。

工程上还有两个实际权衡：

- 参数量会比纯实单通道模型更大，因为每个嵌入要存实部和虚部
- 调试会更麻烦，因为错误不一定体现在 shape，而可能体现在符号和共轭处理

所以，实部和虚部分离存储，通常比直接依赖框架的复数类型更稳。尤其在你要打印中间张量、单步校验公式、或与论文展开式逐项对齐时，这种写法更可控。

---

## 替代方案与适用边界

如果任务只需要很轻量的基线，DistMult 和 TransE 往往更容易上手。基线，白话说，就是一个简单但可靠的起点，用来判断更复杂模型是否真的带来增益。

| 方法 | 优点 | 局限 |
|---|---|---|
| DistMult | 简单、快、实现短 | 对方向性关系不友好 |
| TransE | 直观、易解释 | 对一对多、多对多关系表达有限 |
| ComplEx | 同时兼顾对称与反对称 | 实现细节更容易写错 |
| RotatE | 关系建模能力更强 | 训练与调参通常更复杂 |

可以这样判断是否该用 ComplEx：

- 适合：关系类型多样，且方向性明显，比如 `parent_of`、`works_at`、`acquired_by`
- 可选：关系中既有相似性关系，也有因果或从属关系，希望单模型统一处理
- 不一定适合：图谱很大、你只想先跑一个极简基线，或关系几乎全是近似对称

一个玩具判断法是：

- 如果题目更像“这个实体像不像另一个实体”，DistMult 往往够用
- 如果题目更像“顺序一换，语义就错”，ComplEx 更值得优先考虑

一个真实工程判断法是：

- 在医药知识图谱中，`interacts_with` 可能接近对称
- `has_active_ingredient`、`contraindicated_for`、`metabolized_by` 明显有方向
- 如果你想用同一模型统一覆盖这些关系，ComplEx 是很自然的选择

因此，ComplEx 的适用边界很清楚：它不是通用文本模型，也不是所有知识图谱任务的最终答案，但在“结构化三元组补全 + 关系方向性重要”的场景里，它是一个经典且仍然实用的中等复杂度方案。

---

## 参考资料

1. [Complex Embeddings for Simple Link Prediction](https://proceedings.mlr.press/v48/trouillon16.html)
2. [ttrouill/complex 官方实现](https://github.com/ttrouill/complex)
3. [PyKEEN 中的 ComplEx 实现文档](https://pykeen.readthedocs.io/en/stable/_modules/pykeen/models/unimodal/complex.html)
4. [Observed Versus Latent Features for Knowledge Base and Text Inference（FB15k-237 数据集来源）](https://aclanthology.org/W15-4007/)
5. [FB15k-237 数据集说明](https://paperswithcode.com/dataset/fb15k-237)
6. [Dual Quaternion Embeddings for Link Prediction（含 FB15k-237 上常见基线汇总）](https://www.mdpi.com/2076-3417/11/12/5572)
