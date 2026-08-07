---
title: 前馈网络、残差连接与归一化位置（Pre-LN/Post-LN）
date: 2026-08-07
---

# 前馈网络、残差连接与归一化位置（Pre-LN/Post-LN）

<div class="epigraph">
<p>零件之间的安装顺序，常常比零件本身更重要。</p>
<footer>—— 依据架构工程经验改写</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ 李沐《动手学深度学习》§10.7、Xiong 等（2020）《On Layer Normalization in the Transformer Architecture》 ｜ 2026-08-07</p>
</div>

## 为什么从 FFN、残差与归一化位置开始

Transformer 块的核心是「注意力」，但让整台机器平稳运转的，是三个「配角」：**前馈网络（FFN）**提供逐位置的记忆与非线性，**残差连接**保证梯度流通，**层归一化（LayerNorm）**稳定分布。这三个零件看似简单，却有一个「魔鬼细节」决定深层 Transformer 能不能训练：**归一化放在残差「之前」还是「之后」**——**Pre-LN**（先归一化后加残差）与 **Post-LN**（先残差后归一化）。

原版 Transformer 用 Post-LN，但它「在深层时训练不稳」（需要学习率预热等复杂技巧）；Pre-LN 在现代模型（GPT、BERT 的后续变体）里成为主流，因为它「天生稳定」。本节把这三个「配角」逐个讲透，重点是「Pre-LN vs Post-LN」的差异与演进——它是「深层 Transformer 为什么能训练」的关键工程决策。<span class="marginnote">「Pre-LN vs Post-LN」是 Transformer 工程里最著名的「顺序之争」之一：Post-LN 是原论文选择，但它在「12 层以上」时训练不稳（梯度信号被残差路径与归一化路径的相互作用削弱）；Pre-LN 把归一化挪到残差之前，让「残差主路径」畅通无阻——「<strong>一个位置的挪动，决定深层可不可训</strong>」，这是架构工程里「顺序即架构」的经典案例。</span>

## 1 前馈网络：逐位置的记忆仓库

**前馈网络（Feed-Forward Network, FFN）**：Transformer 块里的第二个子层，对**每个位置独立**做两层的 MLP：

$$
\text{FFN}(\boldsymbol{x}) = \boldsymbol{W}_2\,\sigma(\boldsymbol{W}_1\boldsymbol{x} + \boldsymbol{b}_1) + \boldsymbol{b}_2
$$

**关键特性：逐位置（position-wise）**——FFN 的权重**在所有位置共享**，但每个位置独立计算（不跨位置交互）。它是「通道维的 MLP」（与 1×1 卷积同构，见《多通道与 1×1 卷积》）。

**内部维度**：经典配置是「**先扩宽 4 倍、再缩回**」——$d_{\text{model}} \to 4d_{\text{model}} \to d_{\text{model}}$。中间激活（如 GPT-2 的 768 → 3072）远大于模型维——**FFN 的参数量占 Transformer 总参数的约 2/3**（是「最大的参数块」）。

**为什么需要 FFN？** 注意力是「线性混合」（加权平均）——**它不做逐位置的「非线性变换」**。FFN 补上这一环：每个位置在「注意力混合后的表示」上做独立的非线性加工。**「注意力管『位置间的混合』，FFN 管『位置内的加工』」**——两者缺一不可。<span class="marginnote">「FFN 是记忆仓库」的现代解读：研究者（如 Geva 等, 2021）发现 FFN 的第一层像是「键值记忆」——$\boldsymbol{W}_1$ 的行像「记忆的键」（模式），$\boldsymbol{W}_2$ 的列像「记忆的值」（该模式对应的输出）。注意力决定「调用哪些记忆」，FFN 存储并输出这些记忆——「<strong>注意力取用、FFN 存储</strong>」是大模型知识存储的流行框架，也解释了 FFN 为什么占 2/3 参数。</span>

**易错点：** FFN 的激活函数（ReLU 或 GELU）在 Transformer 里常用 **GELU**（光滑、梯度好，见《激活函数》）。「FFN 是两层 MLP」——不是「一层」，中间必须有非线性，否则两层线性塌缩成一层。

## 2 残差连接：梯度的高速公路

Transformer 块里的残差连接与 ResNet 完全同构：每个子层输出加上「子层的输入」：

$$
\text{输出} = \text{子层}(\text{输入}) + \text{输入}
$$

**作用**（回顾《ResNet》的梯度分析）：梯度沿残差捷径**直达**任意层——$\frac{\partial \text{输出}}{\partial \text{输入}} = 1 + \frac{\partial \text{子层}}{\partial \text{输入}}$，**至少是 1**。深层 Transformer（几十层）靠它「梯度不消失」。

**残差与归一化的交互**：残差让「主路径的梯度」畅通；归一化稳定「每层输入的分布」——两者配合，深层可训练。

**易错点：** 残差要求「子层输出」与「输入」**同维度**（$d_{\text{model}}$）——这是 Transformer 所有子层（注意力、FFN）都保持 $d_{\text{model}}$ 维的原因。**「残差 + 同维度」是 Transformer 块「任意堆叠」的前提**。

## 3 归一化位置之争：Pre-LN vs Post-LN

归一化（LayerNorm）放哪，有两种方案：

**Post-LN（原版 Transformer）**：先做子层 + 残差，再归一化：

$$
\boldsymbol{x}' = \text{LN}\big(\boldsymbol{x} + \text{SubLayer}(\boldsymbol{x})\big)
$$

**Pre-LN（现代主流）**：先归一化，再做子层，最后加残差：

$$
\boldsymbol{x}' = \boldsymbol{x} + \text{SubLayer}\big(\text{LN}(\boldsymbol{x})\big)
$$

**差异**：归一化作用于「残差之前」还是「残差之后」。

**Post-LN 的问题**：深层时训练不稳。分析（Xiong 等, 2020）指出：Post-LN 的残差路径要「穿过」归一化层，而归一化会「缩放」梯度——**深层叠加时，残差主路径的梯度被归一化「削弱」**，导致「训练震荡、需要复杂的预热与调参」。

**Pre-LN 的优势**：归一化在残差之前，**残差主路径不经过任何归一化**——梯度沿「恒等捷径」畅通无阻，**深层天生稳定**（不需要复杂预热）。**「Pre-LN 是『用一点表达力，换训练稳定性』」**。<span class="marginnote">「Pre-LN 的代价」：Pre-LN 的归一化在残差前，会让「残差路径」的输入分布「过度归一化」，理论上损失一点「表达能力」（原论文 Post-LN 在「浅层 + 充分训练」下略优）。但实践上「深层 + 稳定」的价值远大于「浅层的微小精度」——所以 Pre-LN 成为 GPT、LLaMA 等现代模型的主流。这个「稳定优先」的取舍，是「<strong>理论最优 vs 工程可行</strong>」的又一实例。</span>

**易错点：** 「Pre-LN」与「Post-LN」的名字容易记反——**「Pre」指「归一化在子层之前」**，不是「残差之前」。用「归一化作用的对象」来记：Pre-LN 是「先归一化输入，再子层，最后残差」；Post-LN 是「先子层，再残差，最后归一化」。

## 4 公式解析：为什么 Pre-LN 梯度更稳

把两种方案的梯度行为用数学对比。设子层为 $\mathcal{F}$，**Post-LN** 的输入输出关系：

$$
\boldsymbol{x}' = \text{LN}(\boldsymbol{x} + \mathcal{F}(\boldsymbol{x}))
$$

**Pre-LN**：

$$
\boldsymbol{x}' = \boldsymbol{x} + \mathcal{F}(\text{LN}(\boldsymbol{x}))
$$

- **第一步，看 Post-LN 的梯度**：$\frac{\partial \boldsymbol{x}'}{\partial \boldsymbol{x}}$ 要「穿过」归一化层——归一化的雅可比是「缩放矩阵」（对输入做逐维缩放），在深层连乘时，**残差路径的梯度被归一化的缩放因子反复压缩**——梯度信号随层数减弱。
- **第二步，看 Pre-LN 的梯度**：$\frac{\partial \boldsymbol{x}'}{\partial \boldsymbol{x}} = 1 + \frac{\partial \mathcal{F}}{\partial \text{LN}(\boldsymbol{x})}\frac{\partial \text{LN}}{\partial \boldsymbol{x}}$——**恒等项「1」不经过任何归一化**，梯度沿残差直达，至少保持「1 的强度」。
- **第三步，读结论**：**Pre-LN 把「归一化的梯度削弱」从残差主路径上移开**——深层叠加时，梯度不随层数衰减。这就是「Pre-LN 深层稳定」的数学根源。<span class="marginnote">「梯度恒等项 = 1」的威力在深层最明显：$L$ 层 Pre-LN 的浅层梯度 ≈ 上游梯度 ×（累乘的 1 + 小扰动）——几乎不衰减；$L$ 层 Post-LN 的浅层梯度要穿过 $L$ 个归一化缩放——衰减明显。这与《ResNet》的「恒等捷径」分析同构——「<strong>归一化的位置，决定残差是不是『纯』的恒等捷径</strong>」。</span>

## 5 现代实践：归一化的其他变体

归一化在现代 Transformer 里还有几个重要变体：

- **RMSNorm**：去掉 LayerNorm 的「减均值」，只做「除以均方根」——省计算、训练更稳（LLaMA 的标配，见《批量归一化与层归一化》）。
- **Sandwich-LN**：残差前与残差后都放归一化——「两头都稳」，但多一层计算。
- **ScaleNorm / DeepNorm**：对残差路径加「缩放系数」——Post-LN 的「深度友好」改造。

**「归一化是现代 Transformer 最活跃的『小改动』领域之一」**——同一个「稳定分布」的目标，各家模型用自己的变体（LLaMA 用 RMSNorm+Pre-LN、DeepNorm 用 Post-LN+缩放）——「实现细节」的差异，在实际训练稳定性上的影响远大于预期。<span class="marginnote">「为什么 LLaMA 用 RMSNorm」：RMSNorm 是 LayerNorm 的「减均值」版——去掉均值计算，省一次归约（对长序列是大节省），且实践证明「减均值对 Transformer 并非必要」（残差与注意力自带平移不变性）。「省计算 + 足够稳」让 RMSNorm 成为开源 LLM 的主流——「<strong>工程上，能省的都省，只要训练还稳</strong>」。</span>

**易错点：** 归一化层的参数（LayerNorm/RMSNorm 的 $\gamma$ 缩放）**通常不做权重衰减**（衰减会破坏归一化的表达能力）——「参数分组」时要把归一化层单独列出（见《L2 正则化与权重衰减》）。

## 6 小结

- **FFN**：逐位置的两层 MLP（$d\to 4d\to d$），管「位置内加工」，占约 2/3 参数（「记忆仓库」）。
- **残差连接**：每个子层的输出加输入——梯度高速公路，深层可训练。
- **Post-LN**：先残差后归一化——原版选择，浅层略优但深层训练不稳。
- **Pre-LN**：先归一化后子层再残差——残差主路径不经过归一化，**深层天生稳定**，现代主流。
- Pre-LN 的数学：恒等项「1」不被归一化削弱——深层梯度不衰减。
- 现代变体：RMSNorm（LLaMA）、DeepNorm——「稳定分布」的工程实现百花齐放。

在下一节，我们看 Transformer 的第一个「只用一半」的伟大应用——用编码器做「理解」，这就是 **BERT：掩码语言模型与预训练表征**。
