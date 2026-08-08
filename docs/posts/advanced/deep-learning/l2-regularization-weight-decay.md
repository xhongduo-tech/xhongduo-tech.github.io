---
title: 正则化：L2 参数惩罚与权重衰减
date: 2026-08-07
---

# 正则化：L2 参数惩罚与权重衰减

<div class="epigraph">
<p>克制是一种智慧：不想拥有的太多，就不会被太多所累。</p>
<footer>—— 依据老子《道德经》「少则得，多则惑」的精神改写</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ 花书《深度学习》§7.1 ｜ 2026-08-07</p>
</div>

## 为什么从 L2 正则化开始

第三篇的主题是「正则化与优化」，而**L2 参数惩罚（L2 parameter penalty）**——更常用的名字是**权重衰减（weight decay）**——是这一切的起点。它是深度学习里最古老、最普遍、也最容易被忽视的正则手段：一行 `weight_decay` 参数，把权重的平方和加进损失，训练时每个权重都在被「温和地拽向零」。它的数学干净、实现极简、效果稳定，几乎任何网络都值得默认开启。

理解 L2 正则化，等于理解了正则化的**原型**：它的高斯先验解释（第二节）、它的几何直觉（L2 球）、它与 SGD 的相互作用（权重衰减 vs L2 的微妙差异）、它在现代 AdamW 中的回归——这条线索从本节一直延伸到第三篇末尾。**L2 是正则化的「hello world」**，值得完整解剖。<span class="marginnote">权重衰减与 L2 正则化在现代框架里被当作同一件事，但在 Adam 等自适应优化器下两者并不完全等价（本节第六部分详述）。这个细节催生了 AdamW（2019），如今已是预训练大模型的事实标准——一条从「正则项系数」到「大模型训练配置」的完整因果链。</span>

## 1 定义：把权重的平方加进损失

**L2 正则化**：在原有损失 $J$ 上，加上所有权重的平方和（偏置通常不惩罚），再乘一个强度系数：

$$
\tilde{J}(\boldsymbol{\theta}) = J(\boldsymbol{\theta}) + \underbrace{\frac{\alpha}{2} \|\boldsymbol{w}\|_2^2}_{\text{惩罚项}}
$$

其中 $\|\boldsymbol{w}\|_2^2 = \sum_i w_i^2$ 是权重向量的 L2 范数平方，$\alpha$ 控制惩罚强度，$\frac{1}{2}$ 纯粹为求导方便。**为什么只惩罚权重不惩罚偏置？** 因为偏置只移动决策面、不放大输入尺度，对大权重导致的高方差问题没有贡献；且惩罚偏置会破坏平移不变性。

**为什么叫「权重衰减」？** 看它对梯度下降的影响。把惩罚项的梯度加入更新规则：

$$
\boldsymbol{w} \leftarrow \boldsymbol{w} - \eta \nabla_{\boldsymbol{w}} J - \eta\alpha\boldsymbol{w} = (1 - \eta\alpha)\,\boldsymbol{w} - \eta\nabla_{\boldsymbol{w}}J
$$

每一步更新前，权重先被乘以**衰减因子** $(1 - \eta\alpha)$。只要 $\eta\alpha < 1$，权重每步都被「缩水」一点——这就是「衰减」一词的来历。<span class="marginnote">衰减因子 $(1-\eta\alpha)$ 的直觉：如果数据梯度为零（损失平了），权重并不会停在原地，而是按几何级数 $w_t = (1-\eta\alpha)^t w_0$ 指数收缩到零。这个「无梯度也会收缩」的特性，让权重衰减成为隐式的「早停式」容量控制——它让训练过程本质上在「更小权重的模型」附近活动。</span>

## 2 高斯先验：L2 是「我相信权重小」

回看《没有免费午餐定理》里的等式：**正则项 = 负对数先验**。L2 惩罚对应**高斯先验**：

$$
p(\boldsymbol{w}) \propto \exp\Big(-\frac{\|\boldsymbol{w}\|_2^2}{2\sigma_w^2}\Big)
$$

代入最大后验（MAP）目标 $\log p(\mathcal{D}\mid\boldsymbol{w}) + \log p(\boldsymbol{w})$，正则强度 $\alpha = \frac{1}{\sigma_w^2}$ 就是先验方差的倒数。**先验方差越小（$\sigma_w^2 \to 0$），我们越坚信权重该靠近零，$\alpha$ 越大，惩罚越狠。**

这个视角把 L2 正则化从「防过拟合的技巧」升级为「可解释的信念」：**当数据稀少、证据不足时，模型应该「谦虚」——怀疑大权重，相信小权重**。它也是贝叶斯学派「先验正则化」的入口，与第一篇《最大似然估计与贝叶斯统计》直接打通。<span class="marginnote">贝叶斯视角还揭示了一个隐蔽事实：L2 正则化的最优解是「后验均值」的高斯近似——如果数据足够多、似然足够尖，后验会塌缩到最大似然解附近，先验的影响自然消退。所以正则强度 $\alpha$ 应当随数据量增大而减小，这与「数据越多越不需要正则」的经验完全一致。</span>

## 3 对线性回归的精确刻画：权重收缩

在上一节的《正则化如何改写最优解》里我们已经见过线性回归的解析解。用**特征值分解**能看清 L2 到底「怎么收缩」权重。设 $\boldsymbol{X}$ 的奇异值分解中 $\boldsymbol{X}^{\top}\boldsymbol{X}$ 的特征值为 $\{\lambda_i\}$，特征向量为 $\{\boldsymbol{v}_i\}$，则正则化后的最优权重为

$$
\boldsymbol{w}^* = \sum_i \frac{\lambda_i}{\lambda_i + \alpha} \cdot \frac{\boldsymbol{v}_i^{\top}\boldsymbol{X}^{\top}\boldsymbol{y}}{\lambda_i}\,\boldsymbol{v}_i
$$

对比无正则解 $\boldsymbol{w}_{\text{OLS}} = \sum_i \frac{\boldsymbol{v}_i^{\top}\boldsymbol{X}^{\top}\boldsymbol{y}}{\lambda_i}\boldsymbol{v}_i$，差别只在缩放因子 $\frac{\lambda_i}{\lambda_i + \alpha}$：

- **$\lambda_i \gg \alpha$ 的方向**（数据方差大的方向）：缩放因子接近 1，权重几乎不受影响。
- **$\lambda_i \ll \alpha$ 的方向**（数据方差小的方向）：缩放因子接近 0，权重被强烈压缩。

**结论**：L2 正则化**保留「数据支持充分」的方向，压缩「数据支持不足」的方向**——它不是在「均匀缩小所有权重」，而是在「按证据强度分别对待」。<span class="marginnote">这个「按方向区分收缩」的性质有深刻的统计含义：沿着低方差方向，无正则解的估计方差极大（除以小特征值 $\lambda_i$ 放大了噪声）；L2 把这类方向的估计往零收缩，正是「方差换偏差」的精确机制。它在第二级《线性代数》的 SVD 与《数理统计》的岭回归里都有对应。</span>

## 4 对神经网络的直觉：压平过拟合的「峰谷」

对深度网络，L2 正则化的效果无法像线性回归那样闭式刻画，但直觉一致：**它惩罚「大而锋利」的权重组合**。考虑损失曲面——过拟合往往对应「为了拟合个别样本，某些权重变得极大」。权重极大的区域，损失曲面陡峭、决策边界曲折；L2 把这些「尖刺」压平，换来更平滑、更抗扰动的决策面。

**与「平滑性先验」的联系**：小权重 → 输出对输入的敏感度低 → 决策函数更平滑。若 $|\hat{y}(\boldsymbol{x}) - \hat{y}(\boldsymbol{x}')| \le L\|\boldsymbol{x}-\boldsymbol{x}'\|$ 的 Lipschitz 常数正比于权重范数，则 L2 正则就是在压 Lipschitz 常数——**让模型「变化得慢一点」，从而对噪声不敏感**。

**易错点一：** 别把 L2 当「学习率的替代」。权重衰减与学习率是两个独立旋钮：学习率控制「学多快」，权重衰减控制「信多大」。调参时应分别搜索，不能混为一谈。

**易错点二：** 正则化**不是越强越好**。$\alpha$ 过大时，权重被压到接近零，模型退化为「几乎不学」——偏差主导，训练损失都降不下去。$\alpha$ 的正确值要靠在验证集上搜索（对数尺度，如 $10^{-4}$ 到 $10^{-1}$）。<span class="marginnote">一个实用的经验：<strong>权重衰减与「数据增强 + 更多数据」是互补的</strong>。数据不够时，权重衰减救场；数据够了，可以适当调小 $\alpha$。在 ImageNet 时代的 CNN 中，$5\times10^{-4}$ 左右的权重衰减几乎是标配——它让 AlexNet 到 ResNet 都用同一套默认值，说明这个值在「正则收益」与「拟合损失」之间出奇地稳。</span>

## 5 实现：权重衰减在框架里的两个入口

PyTorch 里权重衰减有两个入口，语义略不同：

```python
# 入口一：优化器参数自带 weight_decay（SGD 下等价于 L2）
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)

# 入口二：手动把惩罚项加进损失——L2 的真正定义
l2_reg = sum(p.pow(2).sum() for p in model.parameters())
loss = ce_loss + 0.5 * 1e-4 * l2_reg
```

对于 **SGD**，两种方式数学等价（都对应 $(1-\eta\alpha)$ 的衰减因子）。但对于 **Adam** 这类自适应方法，两者**不等价**：优化器里的 `weight_decay` 项在 Adam 里被「逐元素的学习率」缩放，等效惩罚随梯度大小变化；而手动加 L2 项才是真正的「平方和进损失」。这个差异在 Adam 下会让权重衰减失效一半，于是 AdamW 把衰减**从梯度路径中分离**、直接作用在权重上——这就是自适应优化器的「解耦权重衰减」。(详见《Adam 及其变体》。)

**易错点三：** 别对**偏置**和**归一化层参数**加权重衰减。BatchNorm 的缩放因子 $\gamma$、偏置 $\beta$ 不应被衰减——衰减它们会破坏归一化的表达能力。PyTorch 里用参数分组：

```python
decay, no_decay = [], []
for name, param in model.named_parameters():
    if param.ndim <= 1 or "bias" in name:   # 偏置与 LayerNorm/BatchNorm 参数
        no_decay.append(param)
    else:
        decay.append(param)

optimizer = torch.optim.AdamW([
    {"params": decay,    "weight_decay": 1e-2},
    {"params": no_decay, "weight_decay": 0.0},
], lr=1e-3)
```

<span class="marginnote">「哪些参数该衰减、哪些不该」是大模型训练里一个真实存在的工程细节：Transformer 里通常只衰减 attention 与 FFN 的权重矩阵，不衰减嵌入、偏置、LayerNorm 参数——这部分配置在第九篇《调参方法论》与第六篇《Transformer》里会反复出现。</span>

## 6 小结

- **L2 正则化**：$\tilde{J} = J + \frac{\alpha}{2}\|\boldsymbol{w}\|_2^2$，等价于**高斯先验**「相信权重小」。
- 梯度下降下每个权重被乘**衰减因子** $(1-\eta\alpha)$——「权重衰减」由此得名。
- 线性回归视角：L2 按特征方向**选择性收缩**——保留数据支持充分的方向，压缩支持不足的方向。
- 神经网络视角：压平「大而锋利」的权重，换来更平滑、更抗噪的决策面。
- **SGD 下 L2 ≡ 权重衰减**；**Adam 下两者不等价**，催生了 AdamW。
- 别衰减偏置与归一化参数；$\alpha$ 在对数尺度上搜索，不是越强越好。

在下一节，我们把「向零收缩」换成「推向零」，得到一个会让许多权重**精确等于零**的正则化——这就是**正则化：L1 正则化与稀疏表示**。
