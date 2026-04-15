## 核心结论

SigLIP 的关键改动只有一处，但影响很大：它把 CLIP 使用的 softmax 对比损失换成了 sigmoid 二分类损失。这里的“对比损失”可以先理解成一种训练目标，用来让匹配的图像和文本更接近，不匹配的更远。CLIP 的目标要求一个 batch 内的样本一起参与归一化竞争；SigLIP 则把每一对图文单独当成“是/否匹配”的判断题。

这带来三个直接结果。第一，梯度耦合更弱。所谓“梯度耦合”，就是一个样本的更新方向会不会被 batch 里其他样本显著牵连。第二，对 batch size 的依赖下降，尤其在中小 batch 下更稳。原论文报告了在较小 batch 下 SigLIP 往往优于 CLIP，同时把 batch 做到极大时收益会很快递减。第三，对假阴性更鲁棒。假阴性就是“标签上被当成负样本，但语义上其实接近正样本”的样本，例如同一张图的另一句合理描述。

| 方法 | 训练目标 | 是否依赖 batch 内全局归一化 | 样本间耦合 |
|---|---|---|---|
| CLIP | softmax / InfoNCE | 是 | 强 |
| SigLIP | pairwise sigmoid loss | 否 | 弱 |

如果只记一句话，可以记成：SigLIP 不是“给 CLIP 加个 sigmoid 激活”，而是把“整批排序竞争”改成“逐对二分类学习”。

---

## 问题定义与边界

讨论对象是图文对齐预训练。输入是一批图片和一批文本，模型分别编码成向量，再让正确配对的图文相似度高、错误配对的相似度低。这里的“向量”可以先理解成模型学到的数值表示；“相似度”通常就是两个向量的点积。

设图像编码为 $u_i \in \mathbb{R}^d$，文本编码为 $v_j \in \mathbb{R}^d$。目标是让匹配对 $(i=i)$ 的分数高于不匹配对 $(i \neq j)$。SigLIP 研究的是损失函数，不是 backbone 架构，也不是数据清洗流程本身。因此它能缓解训练目标对大 batch 和假阴性的敏感性，但不能替代数据去重、caption 过滤、采样策略设计。

一个玩具例子最容易看出边界。假设 batch 里有 3 张图片和 3 段文本，其中图片 A 的真实描述有两种写法：“一只黑狗在草地上奔跑”和“黑色小狗在草坪上冲刺”。如果其中一句是正样本，另一句碰巧作为别的图片的文本进入同一 batch，那么 CLIP 的 softmax 会把它当成强竞争对手压下去；SigLIP 仍然会把它当成负对处理，但影响主要局限在这一对上，不会通过整行或整列的归一化把别的分数一起拉偏。

| 讨论问题 | SigLIP 是否直接解决 |
|---|---|
| 损失函数对大 batch 的依赖 | 是 |
| 假阴性造成的全局耦合放大 | 部分缓解 |
| 数据重复、脏标签、错配文本 | 否 |
| 编码器架构是否更强 | 否 |
| 下游检索/分类表现 | 可能改善，但取决于整体训练配置 |

所以本文的边界很明确：分析的是 SigLIP 为什么在训练目标层面更稳，而不是声称它能单独解决多模态预训练的全部问题。

---

## 核心机制与推导

SigLIP 先构造图文对的打分：

$$
z_{ij}=\alpha \langle u_i, v_j \rangle + \beta
$$

其中 $\langle u_i, v_j \rangle$ 是点积，白话讲就是“这张图和这段文本有多对得上”；$\alpha$ 是 inverse temperature，可以理解成分数缩放系数；$\beta$ 是偏置，用来整体平移判别边界。标签定义为：

$$
y_{ij}=
\begin{cases}
1,& i=j \\
0,& i\neq j
\end{cases}
$$

SigLIP 的单对损失写成：

$$
\ell_{ij}=y_{ij}\,\mathrm{softplus}(-z_{ij})+(1-y_{ij})\,\mathrm{softplus}(z_{ij})
$$

这里的 $\mathrm{softplus}(x)=\log(1+e^x)$。它是 logistic loss 的平滑写法。展开后更直观：

- 正样本：$\ell^+_{ij}=\log(1+e^{-z_{ij}})$
- 负样本：$\ell^-_{ij}=\log(1+e^{z_{ij}})$

对 $z_{ij}$ 求导，有一个很干净的结果：

$$
\frac{\partial \ell_{ij}}{\partial z_{ij}}=\sigma(z_{ij})-y_{ij}
$$

其中 $\sigma(z)=\frac{1}{1+e^{-z}}$ 是 sigmoid。它的含义非常直接：

| 样本类型 | 当 $z$ 很小时 | 当 $z$ 很大时 | 直观作用 |
|---|---:|---:|---|
| 正样本 $y=1$ | 梯度接近 $-1$ | 梯度接近 $0$ | 分数太低时强力拉高 |
| 负样本 $y=0$ | 梯度接近 $0$ | 梯度接近 $1$ | 分数太高时强力压低 |

这和 CLIP 的差异在于，CLIP 的 softmax 梯度依赖一整行或一整列的归一化概率。也就是说，某个负样本的存在不仅影响它自己，还会通过分母影响同一行其他项。SigLIP 不需要这个全局分母，所以每一对的更新更像局部决策。

再看一个数值玩具例子。设一个正样本分数 $z_{pos}=2$，一个负样本分数 $z_{neg}=-1$：

- 正样本损失：$\log(1+e^{-2}) \approx 0.127$
- 正样本梯度：$\sigma(2)-1 \approx -0.119$
- 负样本损失：$\log(1+e^{-1}) \approx 0.313$
- 负样本梯度：$\sigma(-1) \approx 0.269$

如果再加一个很容易的负样本，设 $z=-6$，则梯度只有 $\sigma(-6)\approx 0.0025$，几乎不再贡献更新。这说明一个重要事实：负样本不是越多越好。很多“已经很容易分开”的负样本只会增加算力开销，不一定继续带来有效学习信号。原论文也观察到，把 batch 做到极大后收益会明显递减，约 32k 已经是较合理区间。

真实工程里，这个机制的价值更明显。网页图文、商品图文、视频封面标题这类数据常见“同图多 caption”“近重复图片”“语义近义标题”。在这类数据上，强依赖 batch 排名的目标更容易被假阴性扰动；SigLIP 虽然没有把假阴性变成正样本，但它减少了污染扩散的范围。

---

## 代码实现

最小实现思路是先算出 $N \times N$ 的 pairwise logits 矩阵，对角线是正样本，非对角线是负样本，然后逐项做 sigmoid logistic loss。下面这个例子可以直接运行：

```python
import math

def softplus(x: float) -> float:
    if x > 20:
        return x
    if x < -20:
        return math.exp(x)
    return math.log1p(math.exp(x))

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def siglip_pair_loss(logit: float, label: int) -> float:
    return softplus(-logit) if label == 1 else softplus(logit)

def siglip_grad_wrt_logit(logit: float, label: int) -> float:
    return sigmoid(logit) - label

image_emb = [
    [1.0, 0.0],
    [0.0, 1.0],
]
text_emb = [
    [0.9, 0.1],
    [0.2, 0.8],
]

alpha = 5.0
beta = -1.0

logits = [
    [alpha * dot(i, t) + beta for t in text_emb]
    for i in image_emb
]

labels = [
    [1, 0],
    [0, 1],
]

loss = 0.0
for i in range(2):
    for j in range(2):
        loss += siglip_pair_loss(logits[i][j], labels[i][j])
loss /= 4.0

assert logits[0][0] > logits[0][1]
assert logits[1][1] > logits[1][0]
assert siglip_grad_wrt_logit(2.0, 1) < 0
assert siglip_grad_wrt_logit(-1.0, 0) > 0
assert loss > 0

print("logits =", logits)
print("loss =", round(loss, 6))
```

这段代码展示了 SigLIP 的两个实现重点。第一，核心对象是 pairwise logits 矩阵，不是单独的一行 softmax。第二，`alpha` 和 `beta` 不能随手省略。它们决定了 logit 落在什么区间，进而影响梯度是否过早饱和。理论分析工作也指出，可训练的温度和偏置对 sigmoid contrastive loss 的几何性质很关键。

如果换成张量框架，伪代码通常是：

```python
logits = alpha * image_emb @ text_emb.T + beta
labels = eye(N)
loss = labels * softplus(-logits) + (1 - labels) * softplus(logits)
loss = loss.mean()
```

这里没有行 softmax，也没有列 softmax，这正是它和 CLIP 最本质的实现差异。

---

## 工程权衡与常见坑

SigLIP 的工程优势主要有两类。第一类是训练稳定性和扩展性。因为不依赖全局 softmax 分母，batch 变化时训练目标的统计性质更平滑，分布式训练时也减少了“必须为了归一化拿到整批结构”的强约束。第二类是噪声容忍度。它不是不怕脏数据，而是更不容易被单个假阴性通过归一化结构放大成整批污染。

但它也有常见误区：

| 常见坑 | 错误理解 | 正确做法 |
|---|---|---|
| 只把 softmax 改成 sigmoid | 以为只是换激活函数 | 本质是换成 pairwise logistic loss |
| 忽略 $\alpha,\beta$ | 默认设成 1 即可 | 让它们可学习，至少认真调参 |
| 迷信超大 batch | 负样本越多越强 | 负样本收益递减，算力和通信未必值回票价 |
| 把鲁棒性理解成免疫噪声 | 以为假阴性问题自动消失 | 它只是降低耦合扩散，不修复数据本身 |
| 只看 loss 不看采样 | 觉得目标函数会兜底 | 去重、近重复过滤、caption 质量仍是核心 |

真实工程例子可以看电商搜索或图文检索。商品图往往有多条标题、营销文案和属性描述，同一商品还会有多个角度图。若 batch 内恰好采到“同一商品不同标题”或“同款不同角度”作为负样本，CLIP 风格目标容易把这些其实相近的文本压得过狠；SigLIP 仍会惩罚这些对，但伤害主要局限在局部，不会因为 softmax 分母导致一整行概率重新分配。这也是它在噪声较多的网页级图文预训练中更有吸引力的原因。

---

## 替代方案与适用边界

SigLIP 不是对 CLIP 的完全替代，而是另一种偏向。CLIP 更强调 batch 内相对排序，适合“这一组候选里谁最匹配谁”这种竞争结构很明确的任务。SigLIP 更强调每一对是否匹配，适合假阴性更常见、数据噪声更高、batch 规模和通信成本更敏感的场景。

| 方法 | 更适合的场景 | 优点 | 局限 |
|---|---|---|---|
| CLIP / InfoNCE | 数据较干净、强调候选排序 | 排名目标明确，经典基线强 | batch 耦合强，假阴性更敏感 |
| SigLIP | 噪声较多、batch 灵活性重要 | pairwise 梯度独立，扩展性更好 | 不显式建模整批相对概率 |
| 其他 ranking loss | 特定检索排序任务 | 可直接优化排序目标 | 往往更依赖采样与训练技巧 |

因此适用边界可以概括成两句。若你的核心问题是“大规模图文预训练中 batch 太贵、假阴性太多、负样本继续增加收益有限”，SigLIP 通常是更稳的选择。若你的任务更像封闭候选集排序，并且数据质量高、batch 竞争本身就是你想要的监督信号，那么 CLIP 仍然值得保留为强基线。

---

## 参考资料

- Sigmoid Loss for Language Image Pre-Training, arXiv / ICCV 2023: https://arxiv.org/abs/2303.15343
- google-research/big_vision 官方实现与配置: https://github.com/google-research/big_vision
- SigLIP demo notebook: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/SigLIP_demo.ipynb
- Global Minimizers of Sigmoid Contrastive Loss, OpenReview / NeurIPS 2025: https://openreview.net/forum?id=IeM6Io4Rsh
