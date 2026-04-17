## 核心结论

CLIP 和 ALIGN 解决的是同一个核心问题：把图像和文本映射到同一个语义空间。语义空间可以先理解成统一坐标系，语义相近的图文对在坐标系中更接近，语义无关的图文对更远。训练完成后，模型不再依赖固定类别标签，而是通过向量相似度直接完成图文检索、零样本分类和跨模态召回。

两者的核心差异不在损失函数，而在数据策略和扩展路径：

| 维度 | CLIP | ALIGN |
| --- | --- | --- |
| 核心论文 | Radford et al. 2021 | Jia et al. 2021 |
| 训练范式 | 双塔对比学习 | 双塔对比学习 |
| 数据规模 | 约 4 亿图文对 | 18 亿以上图文对 |
| 数据质量策略 | 更强调清洗、过滤、语义匹配 | 更强调直接扩量，容忍网页噪声 |
| 编码器风格 | 图像编码器 + 文本编码器 | 更大容量图像/文本编码器 |
| 工程重点 | 数据过滤、温度参数、泛化 | 大批量训练、吞吐、噪声鲁棒性 |
| 更典型优势 | 语义更可控，零样本分类强 | 大规模检索召回强，扩展性高 |

一个最直观的玩具例子是：一张“狗在公园奔跑”的图片，和文本 `a dog playing in the park` 应该得到高相似度；和 `a cat sleeping on the sofa` 应该得到低相似度。CLIP 和 ALIGN 都是在大规模训练中反复强化这种“正确配对更近、错误配对更远”的关系。区别在于，CLIP 倾向于先尽量过滤掉低质量文本，ALIGN 倾向于接受更多噪声，再用更大的数据量、batch 和模型容量把平均效果拉上来。

因此工程判断可以先写清楚：

1. 如果更在意语义对齐质量、零样本分类和可控性，CLIP 路线通常更稳。
2. 如果有很强算力、目标是超大规模图文检索，ALIGN 的“噪声换规模”路线更有吸引力。
3. 两者共同说明一件事：多模态预训练的关键不是“把图像和文本拼起来”，而是“能否构造出足够强、足够稳定的对比信号”。

---

## 问题定义与边界

问题可以严格定义为：

给定一个 batch 的图像集合 $\{I_i\}_{i=1}^N$ 和文本集合 $\{T_i\}_{i=1}^N$，训练两个独立编码器：

- 图像编码器 $f(I)$：把图片编码成向量
- 文本编码器 $g(T)$：把文本编码成向量

这里的“编码器”可以先理解成特征提取网络。它不直接输出类别，而是把原始输入压缩成可以比较的向量表示。目标是让正确配对 $(I_i, T_i)$ 的相似度尽量大，让错误配对 $(I_i, T_j), i \ne j$ 的相似度尽量小。

典型流程可以写成：

`图像 -> 图像编码器 -> 图像向量`  
`文本 -> 文本编码器 -> 文本向量`  
`向量归一化 -> 相似度矩阵 -> 对称 softmax 对比损失`

边界也要先说清楚。CLIP 和 ALIGN 主要解决的是**图文表示对齐**，不是生成任务。它们擅长回答“这张图和这段文字是否匹配”，不直接负责“根据文字生成图片”或“根据图片生成完整描述”。

常见输入输出边界如下：

| 项目 | 定义 |
| --- | --- |
| 输入 | 图像与对应文本，常见是 image-caption 或 image-alt-text |
| 输出 | 共享语义空间中的 embedding，以及图文相似度分数 |
| 训练监督 | 弱监督，监督信号来自天然网页图文配对 |
| 典型评估 | 图文检索、零样本分类、跨数据集迁移 |
| 非主要目标 | 高质量文本生成、细粒度区域 grounding、复杂链式推理 |

embedding 可以先理解成“压缩后的语义坐标”。一旦图像和文本都落到同一坐标系中，分类就能被改写成相似度比较问题。比如零样本分类时，不训练固定分类头，而是把候选类别写成文本模板：

- `a photo of a dog`
- `a photo of a car`
- `a photo of a traffic light`

然后把图像向量与这些文本向量比较，分数最高的类别就是预测结果。形式化写法是：

$$
\hat{y}=\arg\max_k \; \mathrm{sim}(v, w_k)
$$

其中 $v$ 是图像向量，$w_k$ 是第 $k$ 个类别模板的文本向量。

损失函数通常是对称的 normalized softmax contrastive loss。设归一化后的图像向量为 $v_i$，文本向量为 $w_i$，温度参数为 $\tau$，则图到文方向损失为：

$$
L_{i \to t} = - \frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(v_i^\top w_i / \tau)}{\sum_{j=1}^{N}\exp(v_i^\top w_j / \tau)}
$$

文到图方向损失为：

$$
L_{t \to i} = - \frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(w_i^\top v_i / \tau)}{\sum_{j=1}^{N}\exp(w_i^\top v_j / \tau)}
$$

总损失通常写成：

$$
L = \frac{L_{i \to t} + L_{t \to i}}{2}
$$

温度参数可以理解成 softmax 的缩放系数。$\tau$ 越小，分布越尖锐，模型越强调把正样本和负样本明显拉开；$\tau$ 太小又会带来数值不稳定，这也是工程调参重点之一。

---

## 核心机制与推导

两者共享的核心机制是“双塔 + 批内负样本 + 对称对比损失”。

“双塔”指图像和文本各自使用一个编码器，分别提取特征；“批内负样本”指一个 batch 中，除了正确配对外，其余文本都能作为当前图片的负样本，反过来也成立。这样训练时不需要人工额外采样大量负例，batch 本身就是负样本池。

### 玩具例子：batch=2 的数值推导

设有两个图像向量和两个文本向量：

- $v_1 = (1,0)$
- $v_2 = (0,1)$
- $w_1 = (0.9,0.1)$
- $w_2 = (0.1,0.9)$

这表示图像 1 更接近文本 1，图像 2 更接近文本 2。取 $\tau = 0.1$，先算点积相似度：

| 配对 | 点积 |
| --- | --- |
| $v_1 \cdot w_1$ | 0.9 |
| $v_1 \cdot w_2$ | 0.1 |
| $v_2 \cdot w_1$ | 0.1 |
| $v_2 \cdot w_2$ | 0.9 |

缩放后得到：

| 配对 | 缩放后分数 $s/\tau$ | 指数值近似 |
| --- | ---: | ---: |
| $v_1 \cdot w_1$ | 9 | 8103.08 |
| $v_1 \cdot w_2$ | 1 | 2.72 |
| $v_2 \cdot w_1$ | 1 | 2.72 |
| $v_2 \cdot w_2$ | 9 | 8103.08 |

对图像 1 来说，正确文本 1 的条件概率为：

$$
P(T_1|I_1)=\frac{e^{9}}{e^{9}+e^{1}} \approx 0.9997
$$

因此对应损失

$$
-\log P(T_1|I_1) \approx 0.0003
$$

已经非常小。这个数值例子说明：当正样本明显高于负样本时，对比损失几乎不再惩罚；当模型把正负关系搞反时，这个概率会迅速下降，损失会立刻变大，梯度也会推动模型重新拉开间隔。

### 为什么一个 batch 就足够重要

设 batch size 为 $N$，则单个图像在图到文方向会同时面对 $N-1$ 个负文本；单个文本在文到图方向也会同时面对 $N-1$ 个负图像。因此一次前向传播里，模型实际比较的是一个 $N \times N$ 相似度矩阵：

$$
S_{ij} = v_i^\top w_j
$$

其对角线 $S_{ii}$ 是正样本，非对角线 $S_{ij}, i \ne j$ 是负样本。batch 越大，负样本越丰富，对比学习信号通常越强。这也是为什么 CLIP 和 ALIGN 都高度依赖大 batch 或等效的大规模分布式训练。

### CLIP 与 ALIGN 的差异从哪里来

从公式层面看，两者几乎一样；从工程层面看，差异很大。

CLIP 更像“先把数据尽量做干净，再训练一个足够大的模型”。它隐含的判断是：噪声文本会污染对齐信号，尤其在零样本分类场景里，文本语义一旦偏掉，泛化能力会明显下降。

ALIGN 更像“先接受网页海量弱监督数据，再用规模把噪声摊薄”。它隐含的判断是：只要数据规模足够大、batch 足够大、模型容量足够高，噪声的平均影响可以被规模优势部分抵消。

可以把这种差异概括成一个工程近似式：

$$
\text{有效监督强度} \approx \text{数据规模} \times \text{配对质量} \times \text{模型容量利用率}
$$

CLIP 主要提升“配对质量”，ALIGN 主要提升“数据规模”和“模型容量利用率”。

### 真实工程例子：跨模态检索服务

假设一个素材平台有 5 亿张图片，用户输入 `a red vintage car on the street`，系统希望在几百毫秒内返回结果。一个现实做法是：

1. 离线阶段用图像编码器预计算全部图片 embedding，并写入向量索引。
2. 在线阶段用文本编码器把查询编码成 embedding。
3. 在向量库里做最近邻搜索，返回相似度最高的图片。

这种服务更接近 ALIGN 的强项，因为它更强调海量数据上的召回能力和吞吐。而如果任务是“对工业零件图片做零样本细分类”，例如区分几十种外观接近的零件类型，那么 CLIP 路线通常更稳，因为类别文本必须更干净，语义边界必须更清晰。

---

## 代码实现

下面给一个新手可直接运行的简化版 `python` 实现。它只依赖标准库，不需要安装 `numpy`。代码演示四件事：

1. 向量归一化
2. 相似度矩阵计算
3. 双向对比损失
4. 用文本模板做最小化的零样本分类

```python
import math


def l2_normalize(batch, eps=1e-12):
    """
    batch: List[List[float]], shape = [batch_size, dim]
    """
    normalized = []
    for row in batch:
        norm = math.sqrt(sum(x * x for x in row))
        norm = max(norm, eps)
        normalized.append([x / norm for x in row])
    return normalized


def transpose(matrix):
    return [list(col) for col in zip(*matrix)]


def matmul(a, b):
    """
    a: [m, d]
    b: [d, n]
    return: [m, n]
    """
    b_t = transpose(b)
    out = []
    for row in a:
        out_row = []
        for col in b_t:
            out_row.append(sum(x * y for x, y in zip(row, col)))
        out.append(out_row)
    return out


def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [x / s for x in exps]


def contrastive_loss(image_emb, text_emb, temperature=0.1):
    """
    image_emb: [batch_size, dim]
    text_emb:  [batch_size, dim]
    """
    if len(image_emb) != len(text_emb):
        raise ValueError("image_emb and text_emb must have the same batch size")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    image_emb = l2_normalize(image_emb)
    text_emb = l2_normalize(text_emb)

    logits = matmul(image_emb, transpose(text_emb))
    logits = [[x / temperature for x in row] for row in logits]

    probs_i2t = [softmax(row) for row in logits]
    probs_t2i = [softmax(row) for row in transpose(logits)]

    n = len(image_emb)
    loss_i2t = -sum(math.log(probs_i2t[i][i]) for i in range(n)) / n
    loss_t2i = -sum(math.log(probs_t2i[i][i]) for i in range(n)) / n
    loss = 0.5 * (loss_i2t + loss_t2i)

    return loss, logits, probs_i2t, probs_t2i


def zero_shot_predict(image_emb, class_text_emb, class_names, temperature=0.1):
    """
    image_emb:      [num_images, dim]
    class_text_emb: [num_classes, dim]
    """
    image_emb = l2_normalize(image_emb)
    class_text_emb = l2_normalize(class_text_emb)

    logits = matmul(image_emb, transpose(class_text_emb))
    logits = [[x / temperature for x in row] for row in logits]
    probs = [softmax(row) for row in logits]

    predictions = []
    for row in probs:
        best_id = max(range(len(row)), key=lambda i: row[i])
        predictions.append(class_names[best_id])

    return predictions, probs


def main():
    # 玩具例子：两个图像向量、两个文本向量
    image_emb = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    text_emb = [
        [0.9, 0.1],
        [0.1, 0.9],
    ]

    loss, logits, probs_i2t, probs_t2i = contrastive_loss(
        image_emb, text_emb, temperature=0.1
    )

    assert probs_i2t[0][0] > probs_i2t[0][1]
    assert probs_i2t[1][1] > probs_i2t[1][0]
    assert probs_t2i[0][0] > probs_t2i[0][1]
    assert probs_t2i[1][1] > probs_t2i[1][0]

    print("contrastive loss =", round(loss, 6))
    print("image-to-text probabilities:")
    for row in probs_i2t:
        print([round(x, 6) for x in row])

    # 最小零样本分类例子
    # 假设我们已经有 3 个类别模板的文本向量
    class_text_emb = [
        [0.95, 0.05],  # "a photo of a dog"
        [0.05, 0.95],  # "a photo of a cat"
        [0.70, 0.70],  # "a photo of an animal"
    ]
    class_names = ["dog", "cat", "animal"]

    predictions, probs = zero_shot_predict(
        image_emb, class_text_emb, class_names, temperature=0.1
    )

    print("zero-shot predictions =", predictions)
    print("zero-shot class probabilities:")
    for row in probs:
        print([round(x, 6) for x in row])


if __name__ == "__main__":
    main()
```

一组可预期输出大致如下：

```text
contrastive loss = 0.000146
image-to-text probabilities:
[0.999927, 0.000073]
[0.000073, 0.999927]
zero-shot predictions = ['dog', 'cat']
```

如果换成深度学习框架，训练主干一般就是下面这个骨架：

```python
# pseudo code
image_features = image_encoder(images)      # [B, D]
text_features = text_encoder(texts)         # [B, D]

image_features = normalize(image_features)
text_features = normalize(text_features)

logits = image_features @ text_features.T / temperature
targets = arange(batch_size)

loss_i2t = cross_entropy(logits, targets)
loss_t2i = cross_entropy(logits.T, targets)
loss = (loss_i2t + loss_t2i) / 2

loss.backward()
optimizer.step()
```

CLIP 和 ALIGN 在代码骨架上非常像，真正拉开差距的是训练管线：

| 环节 | CLIP 风格 | ALIGN 风格 |
| --- | --- | --- |
| 数据输入 | 更重过滤规则和去重 | 更重吞吐和海量网页抓取 |
| 文本处理 | 更关注文本质量 | 更容忍 alt-text 噪声 |
| batch 策略 | 大 batch 重要 | 更大 batch 更关键 |
| 模型容量 | 大模型有效 | 更依赖更大模型和分布式训练 |
| 系统优化 | 稳定性与泛化 | 吞吐、缓存、IO、并行 |

对初学者来说，先记住三件事最重要：

1. 归一化后，点积才更接近“方向相似度”，否则向量长度会干扰语义比较。
2. 一个 batch 就能自动产生大量负样本，所以 batch size 在对比学习里不是普通超参数，而是核心训练条件。
3. 温度参数直接影响 softmax 的锐度，进而影响训练稳定性和检索质量。

---

## 工程权衡与常见坑

CLIP 和 ALIGN 的真正难点不在“会不会写损失函数”，而在“数据、算力、系统稳定性是否匹配”。

### 关键权衡

| 问题 | 本质 | 常见后果 | 规避策略 |
| --- | --- | --- | --- |
| 数据质量低 | 图文不真配对 | 学到错误语义 | 过滤、去重、质量评分 |
| 数据规模小 | 负样本多样性不足 | 泛化差 | 扩大数据源，增强 batch |
| batch 太小 | 对比学习信号弱 | 检索效果差 | 梯度累积或分布式训练 |
| 温度不合适 | softmax 过平或过尖 | 收敛慢或不稳定 | 学习式温度或系统调参 |
| 精度设置不当 | 半精度数值溢出 | loss 异常、梯度爆炸 | loss scaling、监控 NaN |
| 文本分布偏置 | 高频模板词过强 | 模型偏见放大 | 采样平衡、模板清洗 |

这里的“梯度爆炸”可以先理解成参数更新步长失控，导致训练突然偏离正常区间。对比学习里它常发生在“大 batch + 低温度 + 半精度”同时出现的组合下。

### CLIP 路线的坑

CLIP 的主要代价是数据清洗成本高。你需要决定哪些图文对保留、哪些删除。这一步做得差，后面模型再大也很难补救。另一个问题是过滤规则本身会引入选择偏置，例如过于偏向英文网页、过于偏向常见视觉实体，最后让模型在长尾概念或非主流文本分布上表现变差。

还有一个常见误区是“只看总数据量，不看有效数据量”。如果 4 亿图文对里有大量模板化垃圾文本，例如：

- `image may contain: outdoor`
- `click here for more`
- `copyright reserved`

那么名义规模很大，但有效监督强度并不高。

### ALIGN 路线的坑

ALIGN 的主要代价是系统成本高。海量网页数据意味着：

- 抓取和解析成本高
- 去重成本高
- 存储与流式读取压力大
- 分布式训练通信成本高
- 数据审计与版权风险更难控制

而且噪声不只是“caption 不太准”这么简单。真实网页里的 alt-text 经常混入站点名、SEO 文案、版权说明、模板词，甚至和图片内容几乎无关。如果完全不做最基本的过滤，模型可能学到的是“网页共现模式”，而不是“视觉对象和语言语义”的对应关系。

### 一个现实建议

如果要在公司里落地图文检索系统，不要一开始就追论文规模。更可行的顺序通常是：

1. 先用中等规模、相对干净的数据验证离线指标是否上升。
2. 再加入更大规模的弱监督数据，观察召回与误检的变化。
3. 最后再考虑 ALIGN 风格的极限扩容。

因为工业系统里，坏数据带来的线上问题往往不是“精度略低”，而是“检索结果明显不合理”。对于用户，这种错误比单纯 recall 下降更容易破坏信任。

### 检索评估时应该看什么

图文检索常见指标不是单一准确率，而是排序指标。最常见的是 Recall@K：

$$
\mathrm{Recall@K} =
\frac{\text{查询中正确结果出现在前 K 个返回中的次数}}{\text{总查询次数}}
$$

例如 `Recall@10 = 0.82` 表示 82% 的查询，其正确匹配结果至少出现在前 10 个候选里。对 CLIP 和 ALIGN 这类模型，Recall@K 往往比普通分类准确率更能反映真实检索能力。

---

## 替代方案与适用边界

CLIP 和 ALIGN 很重要，但不是唯一选择。后续很多工作都在改进数据效率、生成能力或多任务统一性。

| 方法 | 核心特点 | 数据需求 | 模型规模 | 推荐场景 |
| --- | --- | --- | --- | --- |
| CLIP | 干净图文对比学习 | 中到大 | 中到大 | 零样本分类、可控检索 |
| ALIGN | 超大规模噪声对比学习 | 极大 | 大到超大 | 海量检索、极限扩展 |
| Data-Efficient CLIP | 更强调数据效率 | 小到中 | 小到中 | 算力有限、快速验证 |
| BLIP | 对比学习结合生成任务 | 中到大 | 中到大 | 检索 + caption + VQA |
| FLAVA | 多模态统一预训练 | 大 | 大 | 想统一多种任务接口 |

可以直接给新人一个可执行判断：

1. 想快速上线一个小模型，数据不够大，先看 Data-Efficient CLIP。
2. 想做高质量零样本分类，优先看 CLIP 思路和更严格的数据过滤。
3. 想做超大规模跨模态检索，且有强算力和数据工程能力，ALIGN 更合适。
4. 想同时覆盖检索、描述生成、问答，不应只盯 CLIP/ALIGN，BLIP 一类方法更自然。

适用边界也要明确。CLIP/ALIGN 并不擅长细粒度定位，例如“图里左下角的红色螺丝刀旁边那个零件是什么”。因为双塔结构更重全局语义对齐，不直接建模图像局部区域与文本片段之间的精确对应关系。类似任务通常需要 region-level 对齐、cross-attention 融合，或者直接改用生成式视觉语言模型。

所以选方案时不要只问“谁的 benchmark 更高”，而要问三个更具体的问题：

1. 我的任务更像分类、检索，还是生成、定位、推理？
2. 我的数据是少而干净，还是多而脏？
3. 我的系统瓶颈在模型能力，还是在数据管线和算力？

这三个问题回答清楚后，CLIP 和 ALIGN 的选择通常不会太模糊。

---

## 参考资料

1. Radford, A. et al. 2021. *Learning Transferable Visual Models From Natural Language Supervision*.  
核心贡献：系统展示了大规模图文对比预训练如何支持零样本分类，并把“文本模板分类”做成通用范式。  
建议阅读方式：先看方法图、训练目标和 zero-shot 分类实验，再看数据构造与 prompt 模板细节。

2. Jia, C. et al. 2021. *Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision*.  
核心贡献：证明了在更大规模的噪声网页数据上，双塔对比学习仍然可以持续扩展。  
建议阅读方式：重点看数据规模、模型规模、batch 设计和检索类实验，不要只看最终分数。

3. Sohn, K. 2016. *Improved Deep Metric Learning with Multi-class N-pair Loss Objective*.  
核心贡献：帮助理解“一个样本同时面对多个负样本”的训练思想，是批内负样本机制的重要前置背景。  
建议阅读方式：把它当作对比学习损失的历史补充材料，不必纠结具体应用场景差异。

4. OpenCLIP / LAION 相关工程资料。  
核心贡献：提供了开放复现路径，包含公开数据、训练实现、去重与过滤经验。  
建议阅读方式：先看数据管线和训练脚本，再看不同数据清洗策略对结果的影响。

5. Data-Efficient Image Transformers / Data-Efficient CLIP 相关论文与实现。  
核心贡献：讨论在数据和算力不足时，如何提升训练效率与样本利用率。  
建议阅读方式：和原始 CLIP 对照阅读，重点看“同等资源下如何缩小差距”。

6. Flickr30K 与 MSCOCO 图文检索 benchmark 论文及评测说明。  
核心贡献：定义了图文检索常见评测协议，例如 image-to-text、text-to-image 和 Recall@K。  
建议阅读方式：先弄清评测协议，再看论文结果，否则很容易误读指标。

7. BERT、EfficientNet、ViT 等基础模型论文。  
核心贡献：分别对应文本编码器、卷积视觉编码器和 Transformer 视觉编码器的主流选择。  
建议阅读方式：不必全文细读，重点理解编码器容量、输入表示和扩展性差异。
