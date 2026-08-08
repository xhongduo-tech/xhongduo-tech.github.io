---
title: 批量归一化与层归一化
date: 2026-08-07
---

# 批量归一化与层归一化

<div class="epigraph">
<p>让每一层都活在一个好天气里，训练自然顺利。</p>
<footer>—— 依据 Sergey Ioffe 与 Christian Szegedy（2015）的精神改写</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ 花书《深度学习》§8.7.1、李沐《动手学深度学习》§5.6、§6.3 ｜ 2026-08-07</p>
</div>

## 为什么从批量归一化与层归一化开始

深度网络难训练的一个重要原因是**层间分布漂移**：每一层的输入是上一层输出的函数，上一层参数一变，这一层的输入分布就跟着变。训练越深，这种「连锁漂移」越剧烈，导致**饱和激活**（Sigmoid 被推到饱和区、梯度消失）、学习率难以调大、初始化异常敏感。**批量归一化（Batch Normalization, BatchNorm）**（Ioffe & Szegedy, 2015）横空出世，宣称解决了这个难题：它把每层的输入分布**标准化**（零均值、单位方差），让训练从「与分布较劲」变成「在稳定分布上学习」。

BatchNorm 几乎改写了深度学习工程：它让网络可以**用更大的学习率**、**对初始化不敏感**、**允许更深的网络**，甚至在不少场景**取代了 Dropout 的部分作用**。但它也有软肋：依赖**批量统计量**，在 batch 很小、或序列长度可变时失效。于是**层归一化（Layer Normalization, LayerNorm）**、组归一化、实例归一化等「不依赖 batch」的变体相继出现——其中 LayerNorm 成为 Transformer 与所有现代大模型的标配。本节把归一化家族一网打尽。<span class="marginnote">BatchNorm 论文最初把动机归结为「内部协变量偏移（internal covariate shift）」，后来 Santurkar 等（2018）通过实验与理论指出，它真正的收益来自「让损失曲面更平滑」（Lipschitz 常数更小）——即优化地形的改善，而非「分布固定」。这个「归因修正」是深度学习里「被广泛接受的直觉后来被推翻」的经典案例，值得记住：<strong>热门解释未必是真正机制</strong>。</span>

## 1 批量归一化的机制：训练时归一化

**批量归一化**：对每个 mini-batch，在**特征维**上计算均值与方差，把输入标准化，再做一个「重缩放 + 重平移」（仿射变换）以恢复表达能力。

设某一层输入为 $\boldsymbol{x} \in \mathbb{R}^{n \times d}$（$n$ 为批量大小，$d$ 为特征维），对每个特征 $j$：

$$
\hat{x}_j = \frac{x_j - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}, \qquad
\tilde{x}_j = \gamma_j \hat{x}_j + \beta_j
$$

其中 $\mu_j, \sigma_j^2$ 是**该 batch 内第 $j$ 维的均值与方差**；$\gamma_j, \beta_j$ 是**可学习参数**（初始化为 1 和 0）；$\epsilon$ 是防止除零的小常数。<span class="marginnote">为什么要保留可学习的 $\gamma, \beta$？因为「强制零均值单位方差」可能不是最优——网络可能希望保留某些层的原始分布（如把某层输出集中在 0.5 附近）。$\gamma,\beta$ 给了网络「归一化后还想怎么变」的自由：若学到 $\gamma=\sqrt{\sigma^2}$、$\beta=\mu$，则恒等映射、归一化形同虚设——<strong>网络有权选择「要不要归一化」</strong>。</span>

**训练 vs 推理的关键差异**：训练时 $\mu, \sigma^2$ 用**当前 batch** 的统计量；推理时没有「batch」概念，必须用**训练期间累计的全局统计量**（移动平均）。这个「训练用 batch 统计、推理用全局统计」的双轨制，是 BatchNorm 正确使用与踩坑的核心。

**易错点一：** 推理时若还在用 batch 统计量（batch size 设错或忘记 $\mu,\sigma^2$），单样本预测的 $\mu,\sigma^2$ 会随输入剧烈变化，预测完全失真。PyTorch 的 $\mu,\sigma^2$ / $\mu,\sigma^2$ 切换的正是这一行为。

## 2 批量归一化为什么有效：三个真正的原因

**允许大学习率**：归一化把各特征尺度统一，损失曲面更「圆」，梯度方向更一致——这是它最大的实际收益。尺度不一时，大学习率会在某些方向震荡、在另一些方向停滞；归一化后各方向步长均衡。
**缓解梯度问题**：归一化把激活拉回激活函数的「非饱和区」（Sigmoid 中间段），避免饱和导致的梯度消失。
**隐式正则**：batch 统计量带随机性（不同 batch 的 $\mu,\sigma^2$ 不同），相当于给网络注入「轻微噪声」，有类似 Dropout 的正则效果——这让 BatchNorm 有时可以替代或削弱 Dropout。

**易错点二：BatchNorm 的作用位置。** 通常放在「线性变换之后、激活函数之前」（`Linear → BatchNorm → ReLU`），而不是激活之后。放在激活前的理由：归一化线性输出 $\boldsymbol{W}\boldsymbol{x}+\boldsymbol{b}$，让激活函数输入分布稳定；若放在激活后，$\sigma$ 的分布更不可控（如 ReLU 输出半正定）。<span class="marginnote">BatchNorm 的位置之争有一个著名结论：把 BN 放激活后（预激活）在 ResNet 变体里有时更好（Pre-activation ResNet），但对普通 BN 而言，「线性层后激活前」是教科书默认。实践中以验证集为准——结构细节的「标准答案」经常在具体任务上失效。</span>

## 3 批量归一化的软肋：batch 依赖

BatchNorm 的致命弱点是**依赖批量统计量**，在三类场景下失效：

1. **batch 太小**：小 batch 的 $\mu,\sigma^2$ 噪声极大，归一化失真。微调、目标检测（每张图框少）、以及**单样本推理**都受影响。
2. **序列长度可变**：NLP 里每个样本长度不同，padding 让「特征维」的统计量被「空白」污染。
3. **训练/推理分布不一致**：生成模型、强化学习中数据分布剧烈变化，batch 统计量无法反映真实分布。

于是出现了**不依赖 batch 的归一化家族**——它们把「归一化的统计量」从「batch 维」换到「特征维」或「通道维」：

| 方法 | 统计量的计算范围 | 适用 |
| --- | --- | --- |
| BatchNorm | 跨 batch、跨空间 | CNN 大 batch |
| LayerNorm | 单个样本内、跨特征 | Transformer/NLP |
| InstanceNorm | 单个样本、单个通道 | 风格迁移 |
| GroupNorm | 单个样本、通道分组 | batch 小时替代 BN |

**层归一化（Layer Normalization, LayerNorm）**：对**每个样本**的所有特征做归一化，完全不依赖 batch。设单样本特征向量 $\boldsymbol{x} \in \mathbb{R}^d$：

$$
\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad \mu = \frac{1}{d}\sum_j x_j, \quad \sigma^2 = \frac{1}{d}\sum_j (x_j - \mu)^2
$$

**LayerNorm 是 Transformer 的标配**：它对 batch 大小免疫（batch=1 也能算）、对序列长度可变免疫、且训练/推理行为完全一致（无「移动平均」双轨制）。<span class="marginnote">「BatchNorm 横向归一化（跨样本）、LayerNorm 纵向归一化（跨特征）」是对比两者的最好记忆法：BN 看「这个特征在 batch 里的分布」，LN 看「这个样本的特征分布」。在 NLP 里「样本 = 一个 token 的向量」，「跨特征归一化」天然合理；在视觉里，LN 有时不如 BN 因为「跨通道归一化」丢失了通道间的尺度信息——所以 GroupNorm 折中：按通道分组归一化，batch 小时表现接近甚至优于 BN。</span>

## 4 公式解析：BatchNorm 与 LayerNorm 的统计量对比

把两种归一化写成统一的「归一化-仿射」框架，差别只在「统计量在哪算」。设输入张量为 $\boldsymbol{x} \in \mathbb{R}^{n \times d}$：

**BatchNorm**（对特征 $j$，跨 batch $i$ 算统计量）：

$$
\tilde{x}_{ij} = \gamma_j \frac{x_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}} + \beta_j, \qquad
\mu_j = \frac{1}{n}\sum_i x_{ij}, \quad \sigma_j^2 = \frac{1}{n}\sum_i (x_{ij} - \mu_j)^2
$$

**LayerNorm**（对样本 $i$，跨特征 $j$ 算统计量）：

$$
\tilde{x}_{ij} = \gamma_j \frac{x_{ij} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta_j, \qquad
\mu_i = \frac{1}{d}\sum_j x_{ij}, \quad \sigma_i^2 = \frac{1}{d}\sum_j (x_{ij} - \mu_i)^2
$$

- **第一步，看下标**：BN 的 $\mu_j$ 依赖 $j$（特征）——**每个特征一个统计量，跨样本共享**；LN 的 $\mu_i$ 依赖 $i$（样本）——**每个样本一个统计量，跨特征共享**。
- **第二步，看 $\gamma,\beta$ 的形状**：两者都是逐特征（$d$ 维）的可学习向量——无论统计量怎么算，「仿射重参数化」都作用在特征维。
- **第三步，看依赖关系**：BN 的统计量依赖整个 batch（样本间耦合）；LN 的统计量只依赖当前样本（样本独立）。**LN 的「样本独立」特性让它天然适配 batch=1、变长序列、在线推理**——这是它在大模型里胜出的结构性原因。<span class="marginnote">「统计量在哪算」这个视角还能解释归一化的「表达能力」：BN 的逐特征统计量保留了「特征间尺度的相对关系」吗？不——它把每个特征独立归一化，特征间尺度被抹平；LN 的逐样本统计量保留「样本内特征结构」。这解释了为什么 LN 在「特征语义可解释」的 NLP 里更自然。</span>

## 5 现代实践：归一化在大模型中的位置

归一化家族在现代架构中的「势力范围」已经相当清晰：

- **CNN 大 batch 训练**：BatchNorm 仍是主力（ResNet、EfficientNet）。
- **batch 小的视觉任务**：GroupNorm（其等价物 Weight Standardization + GroupNorm 是 NFNet 的核心）。
- **Transformer / LLM**：LayerNorm（或 RMSNorm 这个「只归一化不平移」的简化版）是标配，放在 attention/FFN **之前**（Pre-LN），残差结构让 LN 位置的选择成为设计细节（见第六篇《Pre-LN/Post-LN》）。
- **大规模预训练**：Post-LN 在很深的模型里训练不稳，Pre-LN + warmup 成为事实标准。

**RMSNorm（Root Mean Square Normalization）**：把 LayerNorm 的「减均值」去掉，只做「除以均方根」：

$$
\hat{x}_j = \frac{x_j}{\sqrt{\frac{1}{d}\sum_j x_j^2 + \epsilon}}
$$

它省去均值计算、训练更稳，是 LLaMA 等主流 LLM 的默认选择。「减均值」之所以可省，是因为残差连接与注意力机制已经天然提供平移不变性。<span class="marginnote">「归一化层的位置与形式」是 2020–2024 年大模型架构优化的核心战场之一：Post-LN → Pre-LN → Pre-LN + scale → RMSNorm 的演进，每次都带来训练稳定性或性能的改善。这提醒我们：<strong>「标准架构」是动态的，今日的标配（Pre-LN + RMSNorm）是过去几年反复迭代的结果</strong>——详见第六篇《Transformer 架构》。</span>

## 6 小结

- **BatchNorm**：跨 batch 逐特征归一化 + 可学习仿射，训练用 batch 统计、推理用移动平均。
- 真正收益：**平滑损失曲面、允许大学习率**，附带缓解饱和与隐式正则；「内部协变量偏移」的解释已被修正。
- **LayerNorm**：跨特征逐样本归一化，不依赖 batch，训练/推理一致——Transformer 与 LLM 标配。
- **GroupNorm/InstanceNorm** 按通道/样本细分统计量，batch 小时替代 BN。
- **RMSNorm**：去均值的 LayerNorm，LLM 的事实标准。
- 选型：大 batch CNN 用 BN，变长序列/小 batch 用 LN/GroupNorm。

在下一节，我们换个正则思路：与其让每个参数独立地学，不如**让不同位置的参数共享同一个值**——这就是**参数绑定与参数共享**。
