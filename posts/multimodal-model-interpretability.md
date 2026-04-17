## 核心结论

多模态模型的可解释性，研究的不是“模型答对了没有”，而是“文字、图像、表格、语音这些输入，分别通过什么路径共同影响了输出”。严格说，它要回答三个问题：模型看了什么、哪些证据真正影响了答案、不同模态之间的证据是否能闭合成一条一致的解释链。

单看注意力热力图通常不够。注意力描述的是“信息分配到哪里”，更接近相关性证据，不自动等于因果证据。更稳妥的做法，是把三类证据串起来看：

1. 注意力，回答“模型把哪一段文本和哪一块图像连起来了”。
2. 梯度或 Grad-CAM，回答“输出对哪些输入最敏感”。
3. 反事实干预，回答“如果只改一个关键因素，答案会不会随之改变”。

只有当这三类证据大体一致时，解释才更接近可信。否则很可能只是“图画得好看”，不代表模型真的依赖了那些证据。

跨模态交互通常用交叉注意力刻画。若文本 token $i$ 查询图像 patch $j$，一个标准写法是：

$$
w_{i,j}=\operatorname{softmax}\left(\frac{q_i\cdot k_j}{\sqrt d}\right)
\quad\text{即}\quad
w_{i,j}=\frac{\exp\left(q_i\cdot k_j/\sqrt d\right)}{\sum_{j'}\exp\left(q_i\cdot k_{j'}/\sqrt d\right)}
$$

其中：

- $q_i$ 是第 $i$ 个文本 token 的查询向量。
- $k_j$ 是第 $j$ 个图像 patch 的键向量。
- $d$ 是向量维度。
- $w_{i,j}$ 是归一化后的注意力权重。

它表示“这个词在多大程度上关注这块图像”，但它仍然只是相关性，不自动代表“这块图像导致了这个答案”。

一个最小例子可以说明三类证据为什么要联合使用。问题是“杯子是什么颜色？”，图中杯子是红色。若同时观察到：

- 文本中的 `cup` token 对答案 `red` 的梯度贡献较高；
- 图像中杯子区域被 CAM 或显著图高亮；
- 只把杯子颜色改成蓝色后，`red` 的输出概率从 0.92 降到 0.18；

那么可以较有把握地说，模型依赖的是“杯子区域的颜色”这一跨模态证据，而不是数据集中“杯子经常是红色”这样的语言捷径。

| 解释类型 | 关注点 | 典型证据 | 验证手段 | 信任等级 |
| --- | --- | --- | --- | --- |
| 关联式解释 | 哪些输入与输出同时出现 | 注意力、显著图、特征激活 | 可视化对照 | 中 |
| 因果式解释 | 哪些输入改变会导致输出改变 | 反事实、干预、因果分解 | 改输入测概率变化 | 高 |
| 多模态可信解释 | 跨模态链条是否闭合 | 注意力 + 梯度 + 反事实一致 | 联合验证 | 更高 |

---

## 问题定义与边界

多模态可解释性的目标，不是分别给文本和图像各画一张热图，而是解释整个跨模态推理过程。这个过程至少包含三个层次。

1. 模态对齐。对齐指“文本中的词和图像中的区域是否被正确对应上”。例如问题里出现 `cup`，模型是否真的把它对到了杯子区域，而不是桌布或背景。
2. 概念融合。融合指“模型是否把多个模态的信息合并成一个中间概念”。例如文本中的“杯子”和图像中的“红色区域”是否被组合成“红杯子”。
3. 因果干预。干预指“如果改掉某个关键因素，输出会不会跟着改”。例如只改杯子的颜色，不改物体类别，颜色答案是否翻转。

这三个层次缺一不可。只解释文本侧，容易掩盖视觉失效；只解释图像侧，容易掩盖语言先验。一个典型错误是：文本里出现“红色”，图像里却是蓝色杯子。如果解释只展示文本 token 权重，很可能会误导读者以为模型做了跨模态推理；但如果同时看图像高亮、文本梯度和反事实结果，就能更清楚地判断模型是否偷懒走了语言捷径。

梯度归因的基本形式是：

$$
\frac{\partial y}{\partial x}
$$

其中：

- $y$ 是某个答案的得分或 logit；
- $x$ 可以是 token embedding、图像 patch embedding，甚至原始像素。

它衡量的是“输入稍微变化一点，输出会变化多少”。如果某个输入分量的梯度绝对值很大，说明当前输出对它更敏感。对新手来说，可以把它先理解成“局部影响力分数”。

如果把完整解释流程写成一条顺序链，它大致是：

$$
(\text{文本输入},\text{图像输入})
\rightarrow
\text{模态对齐检查}
\rightarrow
\text{融合层证据检查}
\rightarrow
\text{输出答案}
\rightarrow
\text{反事实改写输入}
\rightarrow
\text{再次推理并比较差异}
$$

这条链路有一个重要边界：多模态解释不保证“模型像人一样理解”。它做的是更有限、更工程化的事情，即在现有网络结构里尽量证明“输出与某些跨模态内部路径一致”。因此它更接近证据审计，而不是心智读取。

为了避免把“解释”这个词说得过宽，可以把常见任务边界列出来：

| 问题 | 是否属于多模态可解释性核心任务 | 原因 |
| --- | --- | --- |
| 给出一张图像热图 | 不充分 | 只覆盖视觉侧，缺少跨模态关系 |
| 给出文本关键词高亮 | 不充分 | 只覆盖语言侧，不能说明是否看图 |
| 说明词和区域如何对应 | 属于 | 这是模态对齐 |
| 说明哪些证据真正改变答案 | 属于 | 这是因果干预 |
| 说明中间概念如何形成 | 属于 | 这是概念融合 |

---

## 核心机制与推导

第一类机制是注意力解释。它回答的是“哪个文本 token 在关注哪些图像区域，或者哪些图像区域在回看哪些文本位置”。它的优点是直接、可视化友好、容易接入现有 Transformer 模型；缺点是只能说明关联强弱，不能单独证明这些位置就是决策依据。

若把文本表示记为 $T\in\mathbb{R}^{n\times d}$，图像 patch 表示记为 $V\in\mathbb{R}^{m\times d}$，则一层交叉注意力可以写成：

$$
Q = T W_Q,\qquad K = V W_K,\qquad U = V W_V
$$

$$
A = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}\right),\qquad
H = AU
$$

其中：

- $A\in\mathbb{R}^{n\times m}$ 是文本到图像的注意力矩阵；
- 第 $i$ 行第 $j$ 列的值，表示第 $i$ 个词对第 $j$ 个 patch 的关注程度；
- $H$ 是融合后的跨模态表示。

这里容易让新手误解的一点是：$A$ 很大，不代表 patch $j$ 一定决定了答案，只代表在这一层里信息大量从那个位置流入。

第二类机制是梯度与特征解释。它回答“当前输出最依赖哪些输入或中间特征”。最常见的两个对象是：

1. 输入梯度。直接看答案分数对输入 embedding 的导数。
2. Grad-CAM。看答案分数对中间特征图通道的导数，再把导数加权回空间位置。

以 Grad-CAM 为例，若视觉特征图为 $F^k\in\mathbb{R}^{h\times w}$，目标类别分数为 $y^c$，则通道权重可写为：

$$
\alpha_k^c = \frac{1}{hw}\sum_{u=1}^{h}\sum_{v=1}^{w}\frac{\partial y^c}{\partial F_{u,v}^k}
$$

再得到空间热图：

$$
L_{\text{Grad-CAM}}^c = \operatorname{ReLU}\left(\sum_k \alpha_k^c F^k\right)
$$

直观上，它表示“哪些空间位置通过哪些通道，最强地支持了当前答案”。与注意力相比，Grad-CAM 更接近“敏感性证据”。

第三类机制是反事实解释。反事实的定义很简单：构造一个只改变最少因素的样本，观察输出是否随之改变。形式上，若原始输入为 $x$，最小改动后的输入为 $x'$，输出为 $f(x)$ 与 $f(x')$，那么一个基本判定量是：

$$
\Delta_y = f(x') - f(x)
$$

如果问题问的是颜色，而反事实只改颜色，则理想情况是：

- 颜色答案发生明显变化；
- 非颜色相关答案尽量不变；
- 模型关注区域仍然落在同一个物体上。

这说明模型依赖的是“该物体的颜色”，而不是别的偶然线索。

在研究层面，CMCR 可以看作较明确的因果式框架。它把 EVQA 中的语言偏置、视觉捷径和解释一致性放到结构因果模型里讨论，用背门调整削弱语言混杂，用前门调整削弱视觉捷径。可以抽象写成：

$$
L' = f_{\text{backdoor}}(L, C_L), \qquad
V' = f_{\text{frontdoor}}(V, M, C_V)
$$

其中：

- $L$ 是语言特征，$V$ 是视觉特征；
- $C_L, C_V$ 是混杂因素；
- $M$ 是中介变量；
- $L',V'$ 是去除偏置后更接近因果证据的特征。

它的核心思想不是“造一个更漂亮的解释”，而是尽量把“真正支持答案的证据”与“训练数据中碰巧共现的偏差信号”拆开。

为了让“答案正确”和“解释可信”同时成立，常见做法是加入联合目标。一个抽象写法是：

$$
\mathcal{L} = \mathcal{L}_{ans} + \lambda \mathcal{L}_{exp} + \beta \,\mathrm{KL}\big(q(z|A,E)\,\|\,p(z|L',V')\big)
$$

其中：

- $\mathcal{L}_{ans}$ 约束答案预测；
- $\mathcal{L}_{exp}$ 约束解释质量；
- KL 项约束答案与解释共享相近的潜在因果因素；
- $\lambda,\beta$ 控制权重。

这类目标的含义是：答案要对，解释也要能被同一组内部证据支持，不能“答案靠一条路径，解释靠另一条路径”。

如果把多模态过程放到机理分析框架里，常见拆法是三阶段：

$$
H_1 = \phi_{\text{extract}}(X_t, X_v),\quad
H_2 = \phi_{\text{align}}(H_1),\quad
H_3 = \phi_{\text{synthesis}}(H_2),\quad
y = g(H_3)
$$

即：

1. 特征提取：把文本和图像先编码成可计算表示。
2. 模态对齐：建立词与区域、句段与局部视觉概念之间的联系。
3. 概念合成：形成“红杯子”“左侧车辆”“异常区域”等中间概念。
4. 输出生成：由中间概念支持最终答案。

这类分阶段分析的价值在于，每一层都可以单独观察和干预。但它仍有局限：特征命名不总是稳定，“这一层就是红色概念”往往只是近似说法，不是严格定理。

真实工程里，医疗 EVQA、安全审核、文档图文理解等场景最需要这种全链路解释。系统不只要回答“图像中是否存在异常”，还要给出：

- 异常区域在哪里；
- 它对应了哪段文本描述；
- 如果遮掉该区域或替换该描述，结论是否变化。

这也是为什么高风险场景不能只交付一张热图，而要交付一整条证据链。

| 方法 | 在解释链中的位置 | 能回答的问题 | 局限 |
| --- | --- | --- | --- |
| 注意力可视化 | 对齐层 | 模型在看哪里 | 相关不等于因果 |
| 梯度/Grad-CAM | 决策敏感层 | 哪些输入最影响当前输出 | 易受尺度与噪声影响 |
| 特征图谱/概念激活 | 中间表征层 | 模型学到了哪些概念 | 概念命名常有主观性 |
| 反事实干预 | 验证层 | 改某因素后答案是否变化 | 构造样本成本高 |
| 因果分解如 CMCR | 全链路 | 是否存在捷径、解释是否一致 | 建模复杂、训练成本高 |

再把三类主流证据放到一张表里，更容易理解它们的关系：

| 证据类型 | 典型问题 | 输出形式 | 最常见误用 |
| --- | --- | --- | --- |
| 注意力 | 模型把词和区域连到哪里 | 权重矩阵、热图 | 把高权重直接当因果 |
| 梯度/CAM | 输出对哪里最敏感 | 显著图、梯度分数 | 忽略噪声与尺度依赖 |
| 反事实 | 改一点输入会不会翻转答案 | 概率差、标签变化 | 修改过多因素导致结论失真 |

---

## 代码实现

工程上更实用的落地方式，是把解释模块做成推理旁路。模型正常前向时，同时记录交叉注意力、保留关键张量梯度，并构造少量反事实样本，最后一起输出解释报告。这样做的好处是：排障时能直接看到“模型看了哪里、敏感在哪里、改了什么会翻转”。

下面先给出一个完全可运行、只依赖 Python 标准库的玩具示例。它不追求还原真实模型结构，只演示三件事如何同时成立：

1. `cup` 这个词是否主要关注杯子 patch；
2. 输出对“杯子红色程度”是否敏感；
3. 只改颜色后，`red` 的概率是否明显下降。

```python
import math


def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross_attention(query, keys):
    d = len(query)
    scores = [dot(query, key) / math.sqrt(d) for key in keys]
    return softmax(scores)


def answer_prob_red(attn_to_cup_patch, cup_redness, text_bias, alpha=6.0):
    # 一个最小玩具 logit:
    # 视觉证据 = cup token 对 cup patch 的关注 * cup patch 的红色程度
    # 文本偏置 = 问题或训练数据里“杯子常见颜色”的语言先验
    visual_evidence = attn_to_cup_patch * cup_redness
    logit = alpha * (visual_evidence + 0.2 * text_bias - 0.5)
    return sigmoid(logit)


def numerical_gradient(fn, x, eps=1e-5):
    return (fn(x + eps) - fn(x - eps)) / (2.0 * eps)


def main():
    # token "cup" 的查询向量
    query_cup = [1.0, 0.0]

    # 两个图像 patch:
    # 第 0 个是 cup patch
    # 第 1 个是 background patch
    keys = [
        [1.2, 0.1],
        [0.2, 1.1],
    ]

    attn = cross_attention(query_cup, keys)
    cup_patch_weight = attn[0]
    background_weight = attn[1]

    # 原图：杯子是红色
    p_red_before = answer_prob_red(
        attn_to_cup_patch=cup_patch_weight,
        cup_redness=1.0,
        text_bias=0.4,
    )

    # 反事实：只把杯子改成蓝色，相当于 cup_redness 从 1.0 改到 0.0
    p_red_after = answer_prob_red(
        attn_to_cup_patch=cup_patch_weight,
        cup_redness=0.0,
        text_bias=0.4,
    )

    # 在 redness = 0.5 处估计局部梯度，表示“红色程度变化对答案概率的影响”
    grad_redness = numerical_gradient(
        lambda redness: answer_prob_red(
            attn_to_cup_patch=cup_patch_weight,
            cup_redness=redness,
            text_bias=0.4,
        ),
        x=0.5,
    )

    delta_prob = p_red_after - p_red_before

    print("cross attention weights:", [round(x, 4) for x in attn])
    print("cup patch weight       :", round(cup_patch_weight, 4))
    print("background patch weight:", round(background_weight, 4))
    print("p(red) before edit     :", round(p_red_before, 4))
    print("p(red) after edit      :", round(p_red_after, 4))
    print("delta probability      :", round(delta_prob, 4))
    print("d p(red) / d redness   :", round(grad_redness, 4))

    assert cup_patch_weight > background_weight
    assert p_red_before > p_red_after
    assert grad_redness > 0

    print("\nInterpretation:")
    print("1. cup token mostly attends to the cup patch.")
    print("2. red answer probability is sensitive to cup redness.")
    print("3. changing only color reduces the red probability significantly.")


if __name__ == "__main__":
    main()
```

这段代码的可解释性结论是直接可读的：

- 如果 `cup patch weight > background patch weight`，说明对齐基本正确；
- 如果 `d p(red) / d redness > 0`，说明输出对颜色证据敏感；
- 如果 `p(red)` 在改色后明显下降，说明反事实验证支持“颜色”是有效证据。

为了让新手不把这个例子误解成“真实模型就是这么算的”，可以明确区分两层含义：

| 部分 | 玩具代码中的作用 | 对真实模型的对应物 |
| --- | --- | --- |
| `cross_attention` | 模拟词对区域的关注 | Transformer 交叉注意力层 |
| `cup_redness` | 人工定义的视觉属性 | patch 特征、视觉编码器输出 |
| `text_bias` | 模拟语言先验 | 数据偏置、问题模板偏差 |
| `numerical_gradient` | 近似局部敏感性 | 自动求导得到的梯度 |
| 改色反事实 | 最小干预 | 图像编辑、mask、属性替换 |

如果希望进一步接近真实深度学习实现，下面给出一个同样可运行的 PyTorch 最小示例。它依赖 `torch`，但不依赖任何训练好的大模型。代码演示如何：

- 做一次最小交叉注意力；
- 对目标答案概率反向传播；
- 读取图像 patch 特征梯度；
- 比较原图与反事实图的输出差异。

```python
import math
import torch
import torch.nn.functional as F


torch.manual_seed(0)


def cross_attention(query, keys, values):
    # query: [1, d]
    # keys : [m, d]
    # values: [m, d]
    scores = (query @ keys.T) / math.sqrt(query.shape[-1])   # [1, m]
    attn = F.softmax(scores, dim=-1)                         # [1, m]
    fused = attn @ values                                    # [1, d]
    return attn, fused


def forward_red_logit(text_query, image_keys, image_values, red_probe):
    attn, fused = cross_attention(text_query, image_keys, image_values)
    # 用一个线性 probe 从融合特征读出“red”分数
    red_logit = (fused * red_probe).sum(dim=-1)  # [1]
    return red_logit, attn


def main():
    d = 4

    text_query = torch.tensor([[1.0, 0.2, 0.0, 0.0]], requires_grad=True)

    # 两个 patch: cup patch 和 background patch
    image_keys = torch.tensor([
        [1.2, 0.1, 0.0, 0.0],
        [0.1, 1.1, 0.0, 0.0],
    ], requires_grad=True)

    # value 的第 3 维代表“红色强度”信号
    image_values_before = torch.tensor([
        [1.0, 0.0, 1.0, 0.2],  # cup patch: redness high
        [0.0, 1.0, 0.0, 0.1],  # background
    ], requires_grad=True)

    image_values_after = torch.tensor([
        [1.0, 0.0, 0.0, 0.2],  # counterfactual: only cup redness changed to low
        [0.0, 1.0, 0.0, 0.1],
    ])

    red_probe = torch.tensor([[0.5, 0.0, 1.2, 0.1]])

    red_logit_before, attn_before = forward_red_logit(
        text_query, image_keys, image_values_before, red_probe
    )
    p_red_before = torch.sigmoid(red_logit_before)

    p_red_before.backward()
    image_value_grad = image_values_before.grad.detach().clone()

    with torch.no_grad():
        red_logit_after, attn_after = forward_red_logit(
            text_query.detach(), image_keys.detach(), image_values_after, red_probe
        )
        p_red_after = torch.sigmoid(red_logit_after)
        delta_prob = p_red_after - p_red_before.detach()

    print("attention before:", attn_before.detach().numpy().round(4).tolist())
    print("p(red) before   :", round(float(p_red_before.item()), 4))
    print("p(red) after    :", round(float(p_red_after.item()), 4))
    print("delta prob      :", round(float(delta_prob.item()), 4))
    print("grad on values  :", image_value_grad.numpy().round(4).tolist())

    assert attn_before[0, 0] > attn_before[0, 1]
    assert p_red_before.item() > p_red_after.item()


if __name__ == "__main__":
    main()
```

这段代码虽然仍是最小示例，但它已经比伪代码更接近真实实现。它揭示了一个重要工程事实：解释并不是在模型外“额外画图”，而是要在前向和反向过程中保留可审计的中间量。

如果把它抽象成统一接口，真实系统的流程一般可以写成：

$$
\text{Report} = \Big(
A_{\text{cross}},
G_{\text{text}},
G_{\text{vision}},
\text{CAM},
\Delta p_{\text{cf}}
\Big)
$$

其中：

- $A_{\text{cross}}$ 是交叉注意力；
- $G_{\text{text}}$ 是文本侧梯度；
- $G_{\text{vision}}$ 是视觉侧梯度；
- $\text{CAM}$ 是空间热图；
- $\Delta p_{\text{cf}}$ 是反事实概率变化。

一个实用的工程建议是，把 `attention hook`、`feature hook`、`gradient capture`、`counterfactual runner` 封装成统一解释接口。否则一旦模型版本升级、层名变化、张量形状调整，解释模块会比主模型更难维护。

---

## 工程权衡与常见坑

最大的误区，是把“能画出来”误当成“解释成立”。很多系统上线后只保存一张 attention 热图，这对演示可能够用，但对定位错误不够。因为热图只能说明关注分布，不能说明这些位置是否真正驱动了答案。

第二个常见坑，是只做单模态解释。图文审核里尤其常见：文本写“危险”，图片却只是普通厨房。若只看文本 token 权重，系统会显得很“有理”；但一旦加入图像反事实测试，比如遮住刀具区域或替换图片，就会发现模型可能根本没认真看图。

第三个坑，是没有反事实样本。没有“只改一个因素”的样本，很多解释验证都无法成立。比如颜色问答任务里，如果训练集几乎没有“蓝色杯子”，模型很容易学到“cup -> red/mug-like color”这种弱规则，而不去读取真实视觉颜色。

第四个坑，是解释生成滞后于推理。很多团队把解释服务做成离线异步任务，结果线上错判发生时，对应层的 attention、梯度、特征缓存都没有保存，最后只能看到输入输出日志，证据链直接断掉。

第五个坑，是概念层漂移。今天某个通道看起来像“红色”，下周模型重训后它可能已经编码“高饱和边缘纹理”了。如果不做固定探针和定期重标定，概念解释会越来越不稳定。

这些坑可以用一个更工程化的视角归纳：解释系统本身也需要像主模型一样被评估。至少要评估三类指标：

1. 稳定性。相似输入的解释是否相似。
2. 忠实性。解释变化是否真的对应输出变化。
3. 可用性。报告能否帮助人类排障与审计。

一个实用判断方法是做删减实验。设解释选出的关键证据集合为 $S$，原始输出为 $f(x)$，删除关键证据后的输出为 $f(x\setminus S)$，则可以定义一个最简单的忠实性分数：

$$
\text{Faithfulness}(S) = f(x) - f(x\setminus S)
$$

如果删掉所谓“关键证据”后，模型输出几乎不变，那么解释很可能不忠实。

| 常见坑 | 典型表现 | 后果 | 规避策略 |
| --- | --- | --- | --- |
| 单模态偏差 | 只看文本或只看图像 | 掩盖捷径 | 强制做跨模态联合解释 |
| 注意力误读 | 热图好看但答案不受影响 | 过度信任相关性 | 联合梯度与反事实验证 |
| 缺少反事实 | 无法验证最小变化是否翻转结论 | 难判因果 | 训练和评估都加入干预样本 |
| 解释延迟 | 推理后拿不到中间状态 | 无法排障 | 推理时同步记录解释证据 |
| 概念层漂移 | 同一特征在不同批次语义变动 | 报告不稳定 | 固定探针集并定期重标定 |

再补一张更贴近落地的排障表：

| 线上现象 | 可能原因 | 应先看什么证据 |
| --- | --- | --- |
| 模型总回答某个高频答案 | 语言偏置过强 | 文本梯度、问题模板统计、反事实问题改写 |
| 模型看错区域 | 模态对齐失败 | 交叉注意力、区域 grounding、CAM |
| 图像改了但答案不变 | 视觉证据未生效 | 视觉梯度、遮挡实验、属性反事实 |
| 文本稍改答案大幅波动 | 语言捷径严重 | token 梯度、文本反事实、训练分布检查 |
| 同图同问解释不稳定 | 特征漂移或缓存错位 | 中间层特征、hook 位置、随机性控制 |

---

## 替代方案与适用边界

如果任务是普通 VQA、内容检索、商品图文匹配，轻量方案通常就够用：交叉注意力 + Grad-CAM + 少量反事实测试。它的优点是部署成本低，能快速回答“模型大致看了哪里、哪些证据更重要、有没有明显捷径”。

如果任务位于医疗、安防、金融风控等高风险场景，仅靠注意力可视化通常不够。更合适的是引入因果一致性思路，例如 CMCR 一类方法。原因不是它“更学术”，而是它会明确处理三个高风险问题：

1. 语言偏置；
2. 视觉捷径；
3. 答案与解释不一致。

如果目标不是生成用户可读报告，而是研究模型内部表征，则可以考虑机理级方法，例如稀疏自编码器、概念向量、特征干预和跨模态回路分析。这类方法的价值主要在研究和内省，不在低成本上线。

DeX 这类反事实解释方法的优势在于它直接回答用户最关心的问题：“改什么会让结论翻转？”在主观分类、审核、推荐解释和偏差审计中，这种问题比“看了哪里”更有业务价值。但它也有边界：如果任务要求精确区域 grounding、像素级定位或完整推理轨迹，单靠 DeX 不够。

可以把不同方案按四个维度比较：因果性、细粒度、部署成本、用户可读性。

| 方案 | 因果性 | 可视化细粒度 | 部署成本 | 适用场景 |
| --- | --- | --- | --- | --- |
| 注意力 + Grad-CAM | 中 | 高 | 低 | 通用 VQA、检索、调试 |
| CMCR | 高 | 中 | 高 | EVQA、医疗、安全 |
| DeX / 反事实解释 | 高 | 中 | 中 | 主观分类、审核、偏差分析 |
| 机理级特征分析 | 中到高 | 很高 | 很高 | 研究、模型内省、失效诊断 |

如果需要一个更直接的选型标准，可以按目标反推方案：

- 你要“看模型大致看了哪里”，优先注意力和 CAM。
- 你要“确认它是不是靠捷径”，必须加反事实。
- 你要“证明答案与解释来自同一因果链”，应考虑 CMCR 这类方法。
- 你要“研究模型内部概念电路”，再考虑机理级分析。

再进一步，可以用任务风险来约束解释深度：

| 任务风险级别 | 最低建议配置 | 不建议省略的环节 |
| --- | --- | --- |
| 低 | 注意力 + 基本梯度 | 交叉模态对齐检查 |
| 中 | 注意力 + 梯度 + 少量反事实 | 概率变化验证 |
| 高 | 因果一致性 + 系统化反事实 + 人工审计 | 解释与答案一致性约束 |

核心边界仍然要强调一次：解释方法并不会自动让模型更公平、更鲁棒或更安全。它只能让问题更容易被看见、定位和验证。真正的改进仍然需要数据、训练目标和评估体系一起改。

---

## 参考资料

下表保留原有资料，并补足“该文主要解决什么问题、读者应如何使用”的信息。这里的重点不是把论文堆在一起，而是明确它们在解释链中的位置。

| 资料 | 核心贡献 | 适用场景 |
| --- | --- | --- |
| *Rethinking Explainability in the Era of Multimodal AI* | 指出单模态解释会系统性误判多模态模型，强调必须解释跨模态影响，而不是分别解释各模态 | 多模态解释方法论综述 |
| *Towards explainable visual question answering via cross-modal causal reasoning* | 提出 CMCR，用背门/前门干预与一致性约束缓解语言偏置、视觉捷径，并绑定答案与解释 | EVQA、因果一致性解释 |
| *Beyond Spurious Signals: Debiasing Multimodal Large Language Models via Counterfactual Inference and Adaptive Expert Routing* | 通过反事实推断与专家路由削弱多模态捷径偏差，强调“偏差控制”也是解释可信性的前提 | MLLM 去偏、鲁棒性提升 |
| *Cross-modal Counterfactual Explanations: Uncovering Decision Factors and Dataset Biases in Subjective Classification* | 用跨模态反事实解释定位决策因素与数据集偏差，回答“改什么会翻转判断” | 主观分类、偏差审计 |
| *Deciphering Cross-Modal Feature Interactions in Multimodal AIGC Models: A Mechanistic Interpretability Approach* | 讨论用机理分析拆解跨模态特征交互，强调“特征提取-对齐-概念合成”分阶段理解 | 研究型模型内省，结论需谨慎使用 |

如果按阅读顺序组织，建议这样看：

1. 先读方法论综述，建立“多模态解释不是单模态热图拼接”的基本框架。
2. 再读 CMCR 这一类工作，理解为什么解释需要因果一致性。
3. 然后读反事实解释与去偏工作，理解“可信解释”和“偏差控制”之间的关系。
4. 最后再看机理分析方向，因为它更适合已经熟悉模型结构和中间表征的读者。

对初学者来说，读这些资料时最好带着三条问题线索：

1. 这篇工作解释的是“相关性”还是“因果性”？
2. 它解释的是输入、跨模态对齐，还是中间概念？
3. 它有没有验证解释是否真的影响输出？

如果一篇工作只展示热图，却没有干预实验或一致性验证，就要对它的解释力度保持谨慎。
