## 核心结论

InfoNCE 是图文对比学习里最常见的目标函数。它把“图像 $i$ 应该匹配文本 $i$”这件事写成一个 softmax 分类问题：正样本相似度进入分子，同 batch 里其他候选进入分母，目标是让正确配对的概率最大。

温度参数 $\tau$ 的作用可以先用一句白话理解：它决定模型打分时“下手有多重”。$\tau$ 越小，softmax 分布越尖，最像正样本的错误候选，也就是 hard negative，会被更强地惩罚；$\tau$ 越大，分布更平缓，训练更稳，但区分近似负样本的力度更弱。

对图文任务，最常见的结论不是“$\tau$ 越小越好”，而是“$\tau$ 决定激进程度，数据质量决定上限”。如果 batch 里存在大量重复样本、同一实体的多种描述、同类但不冲突的样本，那么把 $\tau$ 从 `0.07` 压到 `0.01`，往往不是在学更好的对齐，而是在放大错误梯度。

一个新手版例子：把一张商品图和 10 个标题放在一起，让模型选最匹配的标题。正确标题应该拿最高分，其他标题都算干扰项。InfoNCE 做的就是这件事。$\tau$ 小时，模型会特别盯住“几乎答对但其实答错”的那个标题；$\tau$ 大时，模型会更均匀地看待所有错误选项。

---

## 问题定义与边界

图文对比学习要解决的问题，是把图像表示和文本表示映射到同一个向量空间。向量空间可以白话理解为“能用距离和夹角比较相似度的坐标系”。如果一张图和一句话描述的是同一个对象，它们在这个空间里应该更接近；如果描述不匹配，就应该更远。

它通常服务于三类任务：

| 任务类型 | 目标 | 训练后怎么用 |
| --- | --- | --- |
| 图文检索 | 让匹配图文距离更近 | 图找文、文找图排序 |
| 表征学习 | 学到通用跨模态特征 | 迁移到分类、聚类、召回 |
| 零样本分类 | 用文本标签当分类器 | 不重新训练分类头直接预测 |

本文边界很明确：只讨论 CLIP 式的 batch 内图文匹配，不展开 memory bank、跨 batch 负样本、MoCo 式队列，也不讨论生成式目标如 captioning loss。默认设置如下：

| 默认假设 | 含义 |
| --- | --- |
| 图像向量、文本向量都做 `L2` 归一化 | 归一化就是把向量长度缩成 1，只比较方向，不比较长度 |
| 负样本来自同一个 batch | 一条样本的其他文本或其他图像都暂时被当成负样本 |
| 一张图通常对应一条主文本 | 即单正样本设定，便于解释 InfoNCE |

这里必须分清三个概念。

正样本：真实匹配的图文对，比如“白色运动鞋图片”对应“男款白色跑鞋 42 码”。

负样本：当前 batch 中与该图片不匹配的文本，比如“蓝色休闲鞋”“黑色背包”。

假阴性：被当成负样本，但语义上并不真负。例如同一款鞋有两个合法标题，“男款白色跑鞋 42 码”和“白色透气运动鞋男 42”，在单正样本训练里，第二个可能被误当作负样本。这就是假阴性，白话说就是“系统以为错了，其实也对”。

因此，InfoNCE 的适用前提不是“负样本越多越好”，而是“负样本大多数真的是负的”。一旦这一点被破坏，温度调得再漂亮，也是在用更大的力气优化一个有噪声的目标。

---

## 核心机制与推导

设归一化后的图像向量是 $v_i$，文本向量是 $t_j$，相似度定义为点积：

$$
s_{ij} = v_i^\top t_j
$$

因为已经做了 `L2` 归一化，所以这里的点积就是余弦相似度。余弦相似度可以白话理解为“两个方向有多一致”，范围通常在 $[-1,1]$。

单向 InfoNCE 写成：

$$
L_i = -\log \frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}}
$$

这表示：给定图像 $i$，把 batch 里的所有文本都看成候选类别，要求正确文本 $i$ 的概率最大。CLIP 通常不只做图像到文本一次，还会做文本到图像一次，然后取平均：

$$
L = \frac{1}{2}(L_{\text{image}\rightarrow \text{text}} + L_{\text{text}\rightarrow \text{image}})
$$

这就是双向损失。它的工程意义很直接：不仅要求“每张图能找到对的文本”，还要求“每条文本也能找回对的图”。

为什么 $\tau$ 会影响 hard negatives？关键在 softmax 前的缩放。把所有 logits 除以更小的 $\tau$，等价于把相似度差异放大。

看一个玩具例子。某张图对三个文本的相似度分别是：

- 正样本：`0.40`
- hard negative：`0.39`
- 明显负样本：`0.10`

这三个值已经说明，真正麻烦的不是 `0.10`，而是那个 `0.39`。它和正确答案只差 `0.01`，模型几乎分不出来。

当 $\tau = 0.01$ 时，softmax 看见的是 `[40, 39, 10]`；当 $\tau = 0.07$ 时，softmax 看见的是约 `[5.71, 5.57, 1.43]`。前者把 `0.40` 和 `0.39` 的差距放大得更厉害，因此梯度也更极端。

对负样本 $k \neq i$，梯度可写为：

$$
\frac{\partial L_i}{\partial s_{ik}} = \frac{p_{ik}}{\tau}
$$

其中 $p_{ik}$ 是 softmax 后分给负样本 $k$ 的概率。这个公式的直观含义是：某个负样本越像正样本，$p_{ik}$ 越大；$\tau$ 越小，同样的 $p_{ik}$ 会被更大的 $1/\tau$ 放大。因此，小温度会重点打 hardest negatives。

用上面的数字近似一下：

| 设置 | softmax 概率近似 | hard negative 梯度量级 |
| --- | --- | --- |
| $\tau=0.01$ | `[0.731, 0.269, ~0]` | $0.269 / 0.01 \approx 26.9$ |
| $\tau=0.07$ | `[0.532, 0.461, 0.007]` | $0.461 / 0.07 \approx 6.58$ |

这张表容易引起一个误解：为什么 $\tau=0.07$ 的 hard negative 概率反而更高？原因是 softmax 更平滑了，概率分布没那么尖；但真正决定梯度强度的是 $p_{ik}/\tau$，所以总体更新仍然是小 $\tau$ 更猛。

因此，小温度的本质不是“更懂语义”，而是“更愿意为细微分差付出更大更新”。如果 hard negative 真的是有价值的难例，这会帮助学习；如果它其实是假阴性，这个机制就会把错误放大。

很多代码里并不直接写 $\tau$，而是写 `logit_scale = exp(a)`。这时 logits 计算通常是：

$$
\text{logits} = \exp(a) \cdot (v t^\top)
$$

它与温度的关系是：

$$
\tau = \frac{1}{\exp(a)}
$$

也就是说，`logit_scale` 越大，真实温度越小。这个倒数关系是工程里最容易看反的地方。OpenAI CLIP 常见初始化等价于 $\tau \approx 0.07$，不是把 `0.07` 直接乘到 logits 上，而是乘上它的倒数。

---

## 代码实现

一个最小可运行实现只需要五步：特征归一化、相似度矩阵、乘以 `logit_scale`、做双向交叉熵、取平均。这里的交叉熵可以白话理解为“让正确类别的概率尽量接近 1 的损失函数”。

先看张量形状：

| 变量 | 形状 | 含义 |
| --- | --- | --- |
| `image_feats` | `[B, D]` | 一批图像特征 |
| `text_feats` | `[B, D]` | 一批文本特征 |
| `sim` | `[B, B]` | 图像和文本两两相似度矩阵 |
| `labels` | `[B]` | 正确匹配索引，通常是 `0..B-1` |

下面是一个纯 Python 的玩具实现，便于看清机制，不依赖深度学习框架：

```python
import math

def l2_normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    assert norm > 0
    return [x / norm for x in vec]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def cross_entropy_from_logits(logits, target_index):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    probs = [x / total for x in exps]
    loss = -math.log(probs[target_index])
    return loss, probs

def clip_loss(image_feats, text_feats, tau=0.07):
    images = [l2_normalize(v) for v in image_feats]
    texts = [l2_normalize(v) for v in text_feats]

    logit_scale = 1.0 / tau
    sim = [
        [dot(img, txt) * logit_scale for txt in texts]
        for img in images
    ]

    bsz = len(sim)
    labels = list(range(bsz))

    loss_i2t = 0.0
    loss_t2i = 0.0

    for i in range(bsz):
        loss, _ = cross_entropy_from_logits(sim[i], labels[i])
        loss_i2t += loss

    for j in range(bsz):
        col_logits = [sim[i][j] for i in range(bsz)]
        loss, _ = cross_entropy_from_logits(col_logits, labels[j])
        loss_t2i += loss

    return 0.5 * (loss_i2t / bsz + loss_t2i / bsz)

# 玩具例子：第 0 对和第 1 对是明确匹配
image_feats = [
    [1.0, 0.0],
    [0.0, 1.0],
]
text_feats = [
    [1.0, 0.0],
    [0.0, 1.0],
]

loss = clip_loss(image_feats, text_feats, tau=0.07)
assert loss < 1e-3, loss

# 构造一个 hard negative：第二条文本几乎和第一张图同方向
text_feats_hard = [
    [1.0, 0.0],
    [0.99, 0.1],
]
loss_hard_small_tau = clip_loss([image_feats[0]], [text_feats_hard[0]], tau=0.01)
loss_hard_large_tau = clip_loss([image_feats[0]], [text_feats_hard[0]], tau=0.07)
assert loss_hard_small_tau <= loss_hard_large_tau + 1e-9
```

上面代码里有三个位置最关键。

第一，`L2 normalize` 必须先做。否则相似度不仅反映方向，还会混进向量长度，温度的解释会被破坏。

第二，`logit_scale = 1.0 / tau`。如果你在框架代码里学到的是 `exp(a)`，要明确它乘的是 logits，而不是直接当温度。

第三，双向交叉熵要都算。只算图像到文本，和标准 CLIP 有差异；许多复现效果差，就差在这里。

真实工程例子可以看电商图文检索。假设 batch 大小 `B=256`，每张商品图配一条主标题。模型前向时先得到 `[256, D]` 的图像特征和 `[256, D]` 的文本特征，归一化后做矩阵乘法得到 `[256, 256]` 的相似度矩阵。矩阵第 `i` 行表示“第 `i` 张图对所有标题的打分”，第 `j` 列表示“第 `j` 条标题对所有图片的打分”。然后分别做行方向和列方向的交叉熵。这套结构简单、并行效率高，也是 CLIP 类方法流行的原因。

---

## 工程权衡与常见坑

$\tau=0.01$ 和 $\tau=0.07$ 的差别，不是某个玄学超参数“谁更先进”，而是训练风格不同。

`0.01` 的工程含义是：更激进。它会明显放大 hard negatives 的训练信号，适合负样本质量较高、需要强分离边界的场景。

`0.07` 的工程含义是：更稳健。它不会那么猛烈地惩罚近似负样本，更适合作为默认起点，尤其在数据清洗不充分时。

常见坑如下：

| 常见坑 | 结果 | 规避方式 |
| --- | --- | --- |
| 未归一化 | 温度和向量范数耦合，训练行为失真 | 先 `L2 normalize` 再算相似度 |
| 混淆 `logit_scale` 和 `τ` | 把参数调反，越调越差 | 记住关系：`τ = 1 / logit_scale` |
| 假阴性过多 | 模型被迫推开本不该推开的样本 | 先做去重、多正样本建模，再谈温度 |
| 一味压低 `τ` | 短期 loss 降得快，长期泛化变差 | 把 `τ` 视为放大器，不是修复器 |

新手最容易忽略的是 batch 质量。比如同一 batch 里出现两张同款不同标题的商品图，一张标题写“男款白色跑鞋 42 码”，另一张写“透气白色运动鞋男士 42”。在业务上它们都合理，但单正样本 InfoNCE 会把彼此当负样本。如果此时再把 $\tau$ 压得很低，模型会收到非常强的错误信号，学到“这些其实相近的样本必须分开”。这会伤害召回，尤其伤害同义描述和近重复样本的鲁棒性。

所以工程上处理优先级通常是：

1. 去重和 batch 采样优化。
2. 支持多正样本。
3. 引入去偏损失或标签结构。
4. 最后再细调 $\tau$。

这四步的顺序很重要。因为假阴性问题本质是目标定义错了，温度只能改变“错得多重”，不能把错目标变对目标。

---

## 替代方案与适用边界

InfoNCE 不是唯一选择，它适合的是“单正样本较清晰、batch 足够大、负样本大多数真负”的场景。只要这三个前提被明显破坏，就该考虑替代方案。

| 方案 | 适合的数据分布 | 优点 | 局限 |
| --- | --- | --- | --- |
| InfoNCE / CLIP 式对比学习 | 单正样本近似成立，负样本较干净 | 简单直接，工程成熟 | 假阴性多时容易误罚 |
| Supervised Contrastive Learning | 同类多正样本、标签可用 | 能显式利用同类信息 | 依赖标签或组信息 |
| Debiased Contrastive Learning | 假阴性比例较高 | 试图修正负样本偏差 | 实现更复杂，假设更强 |

Supervised Contrastive Learning 可以白话理解为“不是只认一个正确答案，而是允许一组样本都算正例”。如果一个商品有多个合法标题、一个视频有多句等价描述、一个实体有多个别名，这类方法通常比单正样本 InfoNCE 更合适。

Debiased Contrastive Learning 则是在承认“分母里有些所谓负样本其实不真负”的前提下，修改估计方式，减少假阴性带来的偏差。它不一定总比 InfoNCE 好，但在高重复、高同义数据里通常更合理。

一个新手版判断准则是：

- 如果问题是“模型分不清明显错的负样本”，可以先看 $\tau$。
- 如果问题是“模型把本来相近的样本硬推开”，先看数据和损失定义，不要先动 $\tau$。

换句话说，什么时候优先调数据，而不是调温度？答案是：当错误来自样本语义边界不清，而不是来自打分不够尖锐。电商、短视频、广告素材、多标题内容库，往往都属于前者。

因此，InfoNCE 的适用边界可以总结为：它非常适合做通用跨模态对齐的基线和主干目标；但当多正样本是常态、标签结构清晰、假阴性比例高时，更结构化的监督对比学习或去偏方法通常更合适。

---

## 参考资料

1. van den Oord et al., *Representation Learning with Contrastive Predictive Coding*  
用途：定义。InfoNCE 的经典来源，给出对比预测目标的基础形式。

2. Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*  
用途：实现与机制。CLIP 的双向图文对比学习定义、训练结构和大规模应用背景。

3. OpenAI CLIP 源码 `clip/model.py`  
用途：实现。用于理解 `logit_scale` 的参数化方式，以及它与温度的倒数关系。

4. Khosla et al., *Supervised Contrastive Learning*  
用途：扩展。说明多正样本、同类监督信息如何进入对比学习目标。

5. Chuang et al., *Debiased Contrastive Learning*  
用途：去偏。用于理解假阴性导致的估计偏差，以及如何在目标层面缓解。

6. 本文关于“`0.01` 更狠、`0.07` 更稳”的表述  
用途：工程归纳。它不是来自某一条单独论文定理，而是由 softmax 缩放、梯度形式和常见训练现象共同得出的工程结论。
