## 核心结论

LLaVA 的核心不是“让大语言模型直接看图”，而是先把图像编码成一串视觉 token，再用投影层把这些 token 映射到语言模型的向量空间，最后把它们当作一段可读的前缀送进 LLM。这里的“投影层”就是一个维度变换模块，白话说，它负责把“视觉特征的坐标系”翻译成“语言模型能接收的坐标系”。

可以把这条链路写成两步：

$$
Z_v = g_v(X), \quad Z_v \in \mathbb{R}^{N \times d_v}
$$

$$
H_v = P(Z_v), \quad H_v \in \mathbb{R}^{N \times d_l}
$$

其中，$X$ 是输入图像，$g_v$ 是视觉编码器，白话说就是“把图片切碎并提取特征的模型”；$P$ 是投影层；$N$ 是视觉 token 数；$d_v$ 和 $d_l$ 分别是视觉空间和语言空间的向量维度。

围绕 LLaVA 的几个常见误解需要先澄清。

第一，projector 不是越深越好。它的作用是对齐，不是单独完成复杂视觉理解。视觉编码器输出什么分布、LLM 接收什么分布、训练数据能否覆盖这种分布变化，往往比“线性层还是两层 MLP”更决定上限。

第二，视觉编码器不是随便换。CLIP 类编码器的优势不只是图像表征强，更重要的是它在预训练阶段已经和文本做过对齐。DINOv2 的“视觉表征”通常更强，白话说就是更擅长学图像内部结构，但它不天然懂语言，因此接到 LLM 前，通常需要更强的对齐过程。

第三，Q-Former 不是免费收益。Q-Former 是一种用可学习 query 从视觉 token 中主动读取信息的模块，白话说就是先派一小组“查询向量”去视觉特征里挑重点，再把结果给 LLM。它的好处是可以压缩 token、提高选择性，但代价是参数更多、训练更重、调参更难。

如果只给一个实用判断，可以总结成一句话：LLaVA 的效果，通常先由视觉编码器决定信息质量，再由投影层决定接口是否顺畅，最后由数据和冻结策略决定这条桥能不能稳定学会。

---

## 问题定义与边界

这篇文章讨论的问题很具体：如何让一个原本只会处理文本的 LLM 理解图像输入，而且尽量少改动已有模型。这里的“理解”不是指模型真的拥有视觉感知，而是指它能利用图像特征完成图像问答、图表理解、OCR 辅助、视觉指令跟随等任务。

边界也要说清楚。本文讨论的是 LLaVA 这一类桥接范式：

`图像 -> 视觉编码器 -> 投影层 -> LLM -> 文本输出`

不展开以下方向：

| 方向 | 是否讨论 | 原因 |
|---|---|---|
| 端到端从零训练多模态大模型 | 否 | 成本高，问题设定不同 |
| 先把图像转成自然语言再喂给 LLM | 否 | 那是纯文本中介路线，不是向量对齐 |
| 扩散模型或图像生成模型 | 否 | 输出目标不同，不属于本文主线 |
| 仅做 OCR 管线工程 | 否 | 本文重点是视觉 token 到语言空间的对齐 |

几个核心术语先定下来。

| 术语 | 定义 | 白话解释 |
|---|---|---|
| 视觉编码器 | 把图像转成特征 token 的模型 | 负责“看图并拆成向量” |
| 投影层 | 把视觉特征映射到语言空间的模块 | 负责“翻译向量格式” |
| LLM | 处理 token 并生成文本的模型 | 负责“基于图像和文本做推理与回答” |

这里还要区分“图像变文字”和“图像变向量”。很多初学者会觉得，多模态不就是先做一段图像描述，再把描述发给大模型吗？这不是 LLaVA 的主路径。LLaVA 不是把图片先翻译成一段自然语言，而是把图片变成与语言 token embedding 同类的向量，让 LLM 在统一上下文中同时处理“视觉前缀”和“文本提示”。

一个玩具例子能说明这个边界。假设图片里有一个红色正方形和一个蓝色圆形，问题是“蓝色物体在什么位置”。如果走纯 OCR 或图像描述路线，系统可能先生成一句“图中有红色正方形和蓝色圆形”，再交给 LLM。但如果描述里没有“左上角”这类位置信息，LLM 就无从判断。LLaVA 路线保留的是更细的视觉 token，而不是只保留一句描述，因此理论上能让下游语言模型访问更原始的视觉线索。

---

## 核心机制与推导

先看标准形式。设输入图像为 $X$，视觉编码器输出：

$$
Z_v = g_v(X) \in \mathbb{R}^{N \times d_v}
$$

这里的 $N$ 是视觉 token 数。以常见 ViT 为例，图像会被切成 patch，白话说就是切成一小块一小块，再为每块生成一个向量。如果分辨率和 patch 大小固定，$N$ 也基本固定。例如一张高分辨率图像经过编码后，可能得到数百个 token。

但 LLM 的输入 embedding 维度通常是 $d_l$，不等于 $d_v$。因此需要 projector。

线性投影最直接：

$$
H_v = Z_v W + b
$$

其中 $W \in \mathbb{R}^{d_v \times d_l}$。它的优点是简单、便宜、稳定，缺点是只能做线性映射，表达能力有限。

两层 MLP 则是：

$$
H_v = W_2 \, \sigma(W_1 Z_v)
$$

这里的 $\sigma$ 是非线性激活函数，白话说就是在两次线性变换之间加一个“拐弯”，让模型能学到更复杂的关系。LLaVA-1.5 的经验之一，就是在合适数据和训练流程下，两层 MLP 往往比单线性层更稳，尤其在视觉特征分布和语言空间差异更大时更有用。

先看一个最小数值玩具例子。设单个视觉 token 为：

$$
z = \begin{bmatrix}1\\2\end{bmatrix}
$$

线性投影矩阵为：

$$
W=\begin{bmatrix}
1&0&1\\
0&1&1
\end{bmatrix}
$$

则输出为：

$$
h = z^\top W = [1,2,3]
$$

这说明线性层本质是在固定规则下重新组合输入维度。若要表示更复杂关系，比如第三维不仅是求和，而是与输入分布、位置、上下文有关，线性层就不够灵活，MLP 更合适。

Q-Former 的机制不同。它不对每个视觉 token 直接逐一投影，而是引入 $M$ 个可学习 query，从 $N$ 个视觉 token 中通过交叉注意力读取信息，输出固定长度表示。可以粗略记成：

$$
Q' = \text{CrossAttn}(Q, Z_v)
$$

这里 $Q$ 是 learnable queries，白话说就是一组可训练的“提问向量”。它的直接收益有两个：一是把视觉信息压缩成固定长度，减少传给 LLM 的 token 数；二是让模型主动挑重点，而不是把所有 patch 一股脑送进去。代价是训练复杂度和参数量都更高。

下面这个对比最关键：

| 方案 | 优势 | 劣势 | 适合场景 |
|---|---|---|---|
| 线性投影 | 参数少，训练稳，推理便宜 | 表达力有限 | 基线验证、数据不大时 |
| 2-layer MLP | 非线性更强，对齐能力更好 | 参数略增，仍需好数据 | LLaVA 类通用方案 |
| Q-Former | 可压缩 token，可选择重点 | 更重，更难训 | 大规模训练、需强选择性时 |

再看视觉编码器选择。CLIP-ViT-L/14 的优势是“视觉-文本预训练历史”。白话说，它的特征天然更接近语言监督目标，因此接一个 projector 后更容易与 LLM 对齐。EVA-CLIP 仍属于 CLIP 系，但 backbone 和训练规模更强，通常可视作“更强的 CLIP 路线升级版”。DINOv2 则更偏纯视觉表征，它在区域、纹理、结构上常更强，但默认不具备同等程度的语言对齐能力。

一个真实工程例子是图表问答。假设输入是财报折线图，问题是“哪一季度增长最快”。如果使用 CLIP 系视觉编码器，模型往往更容易把图中的轴标签、线条趋势、图例与文本问题对齐，因为预训练目标里本来就有图文对齐因素。如果换成 DINOv2，模型可能在局部视觉结构上更敏感，但未必能稳定把“这段上升斜率”与“增长最快”对应起来，这时 projector、数据规模和 instruction tuning 质量都会变得更关键。

---

## 代码实现

实现上最重要的是先理解数据流，而不是先追求完整训练框架。最小路径只有四步：图像编码、投影、文本 embedding 拼接、送入 LLM。

```python
import numpy as np

def linear_project(visual_tokens, W, b=None):
    out = visual_tokens @ W
    if b is not None:
        out = out + b
    return out

def relu(x):
    return np.maximum(x, 0.0)

def mlp_project(visual_tokens, W1, W2, b1=None, b2=None):
    hidden = visual_tokens @ W1
    if b1 is not None:
        hidden = hidden + b1
    hidden = relu(hidden)
    out = hidden @ W2
    if b2 is not None:
        out = out + b2
    return out

# 一个玩具 batch：B=1, N=2, dv=2
visual_tokens = np.array([[[1.0, 2.0],
                           [0.0, 1.0]]])   # [1, 2, 2]

# 线性投影到 dl=3
W = np.array([[1.0, 0.0, 1.0],
              [0.0, 1.0, 1.0]])           # [2, 3]

projected = linear_project(visual_tokens, W)
assert projected.shape == (1, 2, 3)
assert np.allclose(projected[0, 0], np.array([1.0, 2.0, 3.0]))
assert np.allclose(projected[0, 1], np.array([0.0, 1.0, 1.0]))

# 两层 MLP 投影
W1 = np.array([[1.0, 1.0],
               [0.0, 1.0]])               # [2, 2]
W2 = np.array([[1.0, 0.0, 1.0],
               [0.0, 1.0, 2.0]])          # [2, 3]

projected_mlp = mlp_project(visual_tokens, W1, W2)
assert projected_mlp.shape == (1, 2, 3)
assert np.all(projected_mlp >= 0.0)

# 文本 embedding 通常是 [B, T, dl]，这里只做形状拼接示意
text_embeds = np.zeros((1, 4, 3))
inputs_embeds = np.concatenate([projected, text_embeds], axis=1)
assert inputs_embeds.shape == (1, 6, 3)
```

上面这段代码没有调用真实 LLM，但已经把核心机制演示清楚了：视觉 token 的最后一维从 `dv` 变成 `dl`，然后才能与文本 embedding 在 token 维度上拼接。

工程里通常会拆成下面几个模块：

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `vision_encoder` | 图像张量 | `[B, N, dv]` | 提取视觉 token |
| `projector` | `[B, N, dv]` | `[B, N, dl]` | 对齐到语言空间 |
| `tokenizer/embedding` | 文本 | `[B, T, dl]` | 得到文本 embedding |
| `llm` | `[B, N+T, dl]` | logits | 生成答案 |
| `loss` | logits + labels | 标量 | 做监督训练 |

最小伪代码如下：

```python
visual_tokens = vision_encoder(images)          # [B, N, dv]
image_embeds = projector(visual_tokens)         # [B, N, dl]
text_embeds = llm.embed_tokens(input_ids)       # [B, T, dl]
inputs_embeds = concat(image_embeds, text_embeds, dim=1)
logits = llm(inputs_embeds=inputs_embeds)
loss = cross_entropy(logits[:, :-1], labels[:, 1:])
```

训练时常见流程是：先冻结视觉编码器，先训 projector；如果数据足够、目标任务复杂，再解冻部分 LLM 或做更完整的 instruction tuning。冻结，白话说就是参数先不更新，目的是减少训练不稳定和成本。

---

## 工程权衡与常见坑

真正落地时，backbone、projector、数据、冻结策略必须一起看。只改一个模块，往往会把系统从“能训”改成“训不动”。

先看一个总表：

| 决策点 | 常见错误 | 结果 | 更稳妥做法 |
|---|---|---|---|
| 换视觉编码器 | 直接把 CLIP 换成 DINOv2 | 对齐失败或收益很小 | 同时调整 projector 和训练数据 |
| 加深 projector | 以为层数越深越强 | 更难训，可能过拟合 | 先验证两层 MLP，再评估收益 |
| 使用 Q-Former | 低估其训练成本 | 训练慢、调参复杂 | 只有在 token 压缩确实关键时再上 |
| 提高分辨率 | 只看视觉更细 | token 数暴涨，LLM 上下文变贵 | 先算清 $N$ 与推理成本 |
| 冻结策略 | 一次性全解冻 | 训练不稳、灾难性遗忘 | 先冻 backbone，再逐步放开 |

几个坑需要单独展开。

第一，DINOv2 不是 CLIP 的无缝替代。DINOv2 学到的是强视觉特征，不是强图文对齐特征。对于“找边界、看纹理、识别区域关系”这类任务，它可能更强；但对于“把图中的视觉线索稳定映射到语言问题”这件事，CLIP 系往往更省心。很多失败实验不是因为 DINOv2 差，而是因为沿用了 CLIP 路线的 projector 和数据规模。

第二，不要只盯 projector 深度。视觉 token 数 $N$ 对系统开销影响很大。若图像分辨率翻倍，patch 数可能接近平方级增长，LLM 处理的前缀长度也会变长，推理成本和注意力负担显著上升。很多时候，系统瓶颈不是“投影层太浅”，而是“输入 token 太多”。

第三，Q-Former 的成本经常被低估。它不是简单的插拔头，而是引入一套额外的 cross-attention 读写机制。如果你的目标只是做一个稳定的图像问答或图表理解系统，2-layer MLP 往往更有性价比。只有当你明确需要压缩视觉 token、做区域选择、或追求更复杂的视觉读写行为时，Q-Former 才更合理。

第四，数据比结构更容易成为上限。一个 projector 能不能学会，不只看参数形式，还看图文指令数据是否覆盖了目标任务。比如你希望模型处理 UI 截图、表格、公式、自然图像，但训练数据大多只是通用图文问答，那么换更强视觉编码器也未必带来真实收益。

工程检查清单可以压缩成四项：

| 检查项 | 要问的问题 |
|---|---|
| 模块冻结 | 哪些层在更新，哪些层完全冻结 |
| token 开销 | 图像分辨率和视觉 token 数是否过大 |
| 数据质量 | 是否真有目标任务对应的图文监督 |
| 对齐目标 | backbone 输出分布是否适合当前 projector 和 LLM |

---

## 替代方案与适用边界

如果目标是做一个通用、稳妥、成本可控的多模态系统，CLIP 或 EVA-CLIP 加两层 MLP projector，通常是最均衡的起点。它不是绝对最强，但在工程上最容易得到“可训练、可复现、可调优”的结果。

如果任务更偏 OCR、表格、图表问答，CLIP 系通常仍是更稳的接口型选择。原因不是它一定视觉最好，而是它与文本对齐历史更长，能更顺畅地把视觉线索送进语言模型。

如果任务更偏细粒度区域理解、局部结构分析、视觉表征本身，DINOv2 值得考虑。但要接受一个现实：你获得的是更强视觉能力，不是免费语言对齐。对应地，你要付出更强 projector、更细训练流程，或更多高质量图文数据。

下面给一个汇总表：

| 方案 | 优势 | 劣势 | 适用任务 | 训练/推理代价 |
|---|---|---|---|---|
| CLIP-ViT-L/14 + Linear | 简单稳，易复现 | 上限有限 | 基线、多数 VQA | 低 |
| CLIP-ViT-L/14 + 2-layer MLP | 对齐更强，通用性好 | 比线性略重 | 通用视觉指令跟随 | 低到中 |
| EVA-CLIP + 2-layer MLP | 保留图文对齐，视觉更强 | backbone 更大 | 更高上限的通用系统 | 中 |
| DINOv2 + 2-layer MLP | 视觉表征强 | 语言对齐弱，调参更难 | 区域理解、结构分析 | 中到高 |
| 任意 backbone + Q-Former | 可压缩 token，可选重点 | 模型更复杂 | 大规模或需 token 压缩 | 高 |

可以把适用边界压成一句判断规则：

1. 通用指令跟随，优先 CLIP 或 EVA-CLIP。
2. OCR、表格、图表任务，先从 CLIP 系基线开始，再看是否需要更强视觉 backbone。
3. 细粒度定位、区域理解、局部结构任务，可评估 DINOv2，但要额外预算对齐成本。
4. 上下文窗口紧张、视觉 token 太多时，再认真考虑 Q-Former 这类压缩方案。

换句话说，LLaVA 架构里的 projector 选择，更多是在决定“接口复杂度”；视觉编码器选择，更多是在决定“你到底给接口什么信息”。二者不能分开谈。

---

## 参考资料

下表给出“本文判断来自哪里”，避免只堆链接不说明用途。

| 资料 | 用途 |
|---|---|
| LLaVA: Visual Instruction Tuning | 定义 LLaVA 基本桥接架构与线性投影思路 |
| Improved Baselines with Visual Instruction Tuning | 说明改进训练流程与 MLP projector 的经验价值 |
| BLIP-2 | 说明 Q-Former 的基本思想与冻结式桥接范式 |
| EVA-CLIP | 说明更强 CLIP 系视觉编码器的训练路线 |
| DINOv2 official repo | 说明 DINOv2 的视觉表征定位 |
| From CLIP to DINO: Visual Encoders Shout in Multi-modal Large Language Models | 比较视觉编码器替换对 MLLM 的影响 |
| LLaVA-MORE official repo | 参考多视觉编码器与工程扩展实践 |

原始论文与实现：
- LLaVA: Visual Instruction Tuning. https://huggingface.co/papers/2304.08485
- Improved Baselines with Visual Instruction Tuning. https://huggingface.co/papers/2310.03744
- BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. https://arxiv.gg/abs/2301.12597
- EVA-CLIP: Improved Training Techniques for CLIP at Scale. https://arxiv.gg/abs/2303.15389
- DINOv2 official repo. https://github.com/facebookresearch/dinov2
- From CLIP to DINO: Visual Encoders Shout in Multi-modal Large Language Models. https://github.com/YuchenLiu98/COMM
- LLaVA-MORE official repo. https://github.com/aimagelab/LLaVA-MORE

如果只读三份，建议顺序是：
1. 先读 LLaVA 原论文，建立桥接范式。
2. 再读 LLaVA-1.5/Improved Baselines，理解 projector 与数据流程为什么能显著影响效果。
3. 最后对照 BLIP-2、EVA-CLIP、DINOv2，理解“视觉编码器强”与“语言对齐强”不是同一个维度。
