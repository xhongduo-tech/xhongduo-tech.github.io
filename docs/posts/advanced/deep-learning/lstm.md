---
title: LSTM：门控记忆与长依赖
date: 2026-08-07
---

# LSTM：门控记忆与长依赖

<div class="epigraph">
<p>真正的记忆，不是记住一切，而是决定忘掉什么。</p>
<footer>—— 依据记忆研究的精神改写</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ 花书《深度学习》§10.10、李沐《动手学深度学习》§8.8 ｜ 2026-08-07</p>
</div>

## 为什么从 LSTM 开始

RNN 的隐状态用「$\tanh$ 加权和」更新记忆，但梯度沿时间连乘 $\boldsymbol{W}_{hh}$——长程记忆几乎学不到（见 BPTT 的连乘分析）。1997 年 Hochreiter 与 Schmidhuber 提出**长短期记忆网络（Long Short-Term Memory, LSTM）**，从**结构上**根治这个问题：它给记忆单元装上**门控（gate）**，用「加法路径」替代「乘法路径」，让梯度可以「畅通无阻」地在长序列中传播。LSTM 一举把「可记忆的序列长度」从十几步推到上千步，统治了 1997–2017 年的序列建模（语音、机器翻译、手写识别）。

理解 LSTM 的关键不是背四个公式，而是理解它的**设计动机**：普通 RNN 的「隐状态被 $\boldsymbol{W}_{hh}$ 反复相乘」是梯度消失的根源；LSTM 引入一条**细胞状态（cell state）**的「传送带」——信息在这条传送带上**直线流动**，只有门控决定「写多少、忘多少、读多少」。本节把 LSTM 的细胞状态、三个门控、以及「为什么门控能救梯度」层层拆开。<span class="marginnote">LSTM 的灵感来自「记忆的读-写-遗忘」模型：细胞状态是「长期记忆本」，输入门决定「写什么进记忆本」，遗忘门决定「撕掉记忆本的哪页」，输出门决定「把记忆本的哪些内容说出去」。这个「人脑记忆」的隐喻让 LSTM 直观好懂——但要记住，<strong>门控是「可学习的软开关」</strong>，不是「硬决策」，网络自己学会「何时写、何时忘」。</span>

## 1 从 RNN 的问题到 LSTM 的答案

普通 RNN 的隐状态更新：

$$
\boldsymbol{h}_t = \tanh(\boldsymbol{W}_{hx}\boldsymbol{x}_t + \boldsymbol{W}_{hh}\boldsymbol{h}_{t-1} + \boldsymbol{b}_h)
$$

**问题**：$\boldsymbol{h}_t$ 对 $\boldsymbol{h}_{t-1}$ 的雅可比是 $\boldsymbol{W}_{hh}^{\top}\,\text{diag}(\tanh')$——梯度沿时间每走一步就乘一次，谱半径 $\gamma<1$ 则指数消失（见《BPTT》）。

**LSTM 的解法**：引入**细胞状态 $\boldsymbol{c}_t$**，它的更新用**加法**而非乘法：

$$
\boldsymbol{c}_t = \underbrace{\boldsymbol{f}_t}_{\text{遗忘门}} \odot \boldsymbol{c}_{t-1} + \underbrace{\boldsymbol{i}_t}_{\text{输入门}} \odot \tilde{\boldsymbol{c}}_t
$$

关键：**$\boldsymbol{c}_t$ 对 $\boldsymbol{c}_{t-1}$ 的雅可比是 $\text{diag}(\boldsymbol{f}_t)$**——不再是「权重矩阵连乘」，而是「逐元素的门控缩放」。若遗忘门 $\boldsymbol{f}_t$ 接近 1，梯度沿细胞状态**几乎无损传播**——这就是「长程记忆可行」的机制。<span class="marginnote">「乘法 → 加法」是 LSTM 的核心手术：RNN 的「信息传递」是矩阵乘法（连乘爆炸/消失），LSTM 的「记忆传递」是「$\boldsymbol{f}_t \odot \boldsymbol{c}_{t-1} + \cdots$」——当 $\boldsymbol{f}_t\approx 1$ 时，细胞状态像「传送带」一样把过去的信息原样搬到未来，梯度也随传送带直线回流。这个「加法路径 = 梯度高速路」的思想，与 ResNet 的恒等捷径异曲同工（ResNet 2015 年把它用在深度上，LSTM 1997 年把它用在时间上）。</span>

## 2 三个门控：遗忘、输入、输出

LSTM 有三个门控，每个都是「Sigmoid 输出 $(0,1)$」的软开关：

**遗忘门（forget gate）**——决定「忘掉多少旧记忆」：

$$
\boldsymbol{f}_t = \sigma(\boldsymbol{W}_{f}[\boldsymbol{h}_{t-1}, \boldsymbol{x}_t] + \boldsymbol{b}_f)
$$

**输入门（input gate）**——决定「写入多少新信息」：

$$
\boldsymbol{i}_t = \sigma(\boldsymbol{W}_{i}[\boldsymbol{h}_{t-1}, \boldsymbol{x}_t] + \boldsymbol{b}_i)
$$

**候选记忆（candidate）**——「要写入的新内容」（用 $\tanh$ 生成候选）：

$$
\tilde{\boldsymbol{c}}_t = \tanh(\boldsymbol{W}_{c}[\boldsymbol{h}_{t-1}, \boldsymbol{x}_t] + \boldsymbol{b}_c)
$$

**输出门（output gate）**——决定「读出多少记忆到隐状态」：

$$
\boldsymbol{o}_t = \sigma(\boldsymbol{W}_{o}[\boldsymbol{h}_{t-1}, \boldsymbol{x}_t] + \boldsymbol{b}_o)
$$

**完整的细胞状态与隐状态更新**：

$$
\boldsymbol{c}_t = \boldsymbol{f}_t \odot \boldsymbol{c}_{t-1} + \boldsymbol{i}_t \odot \tilde{\boldsymbol{c}}_t
$$

$$
\boldsymbol{h}_t = \boldsymbol{o}_t \odot \tanh(\boldsymbol{c}_t)
$$

**流程读法**：先由「旧隐状态 + 当前输入」算出三个门和候选记忆；然后用遗忘门「按比例保留旧记忆」、输入门「按比例写入新记忆」，合成新细胞状态；最后输出门决定「从新记忆里读出多少」作为隐状态。<span class="marginnote">「为什么候选记忆用 $\tanh$、门控用 Sigmoid」：门控需要「$(0,1)$ 的软开关」所以用 Sigmoid；记忆内容需要「可正可负、有界」所以用 $\tanh$。这个「门控 Sigmoid + 内容 Tanh」的分工是 LSTM 的经典约定——理解「每个激活函数在扮演什么角色」，比死记公式更有用。</span>

**易错点：** 三个门控**共享输入**（都是 $[\boldsymbol{h}_{t-1}, \boldsymbol{x}_t]$ 的函数），但**各自有自己的权重**（$\boldsymbol{W}_f, \boldsymbol{W}_i, \boldsymbol{W}_o$ 不同）。「门」不是「同一个门」——遗忘、输入、输出是**三个独立学习**的开关，网络自己学「何时该忘、何时该记、何时该说」。

## 3 为什么 LSTM 能学长程依赖：梯度视角

回到《BPTT》的连乘分析。普通 RNN 中，损失对 $\boldsymbol{c}_k$（$k$ 远离末端）的梯度为

$$
\frac{\partial L}{\partial \boldsymbol{c}_k} = \frac{\partial L}{\partial \boldsymbol{c}_T} \prod_{j=k+1}^{T} \underbrace{\text{diag}(\boldsymbol{f}_j)}_{\text{门控雅可比}}
$$

- **第一步，看雅可比**：LSTM 的细胞状态雅可比是 $\text{diag}(\boldsymbol{f}_j)$——**不再是权重矩阵，而是逐元素的遗忘门**。
- **第二步，看两种命运**：当 $\boldsymbol{f}_j \approx 1$（「别忘」），连乘 $\approx 1$，梯度**无损流过**——长程记忆可学；当 $\boldsymbol{f}_j \approx 0$（「忘掉」），梯度截断——该路径「主动放弃」。
- **第三步，读精髓**：**遗忘门是「可学习的梯度开关」**——网络自己决定「哪条路径保留梯度、哪条丢弃」。相比普通 RNN 的「被动消失」，LSTM 是「主动选择」——该记住的梯度畅通，该忘的梯度截断。<span class="marginnote">「门控可学习」让 LSTM 比「固定结构的 ResNet 恒等捷径」更灵活：LSTM 的遗忘门可以学成「始终接近 1」（长期记忆模式）或「动态切换」（需要忘时忘）。这个「可学习的梯度路径选择」，是 LSTM 长程能力的本质。实践中，遗忘门初始化为接近 1（偏置设大），让网络「默认不忘记、学需要时才忘」——这是 LSTM 训练的重要技巧。</span>

**易错点：** LSTM 不是「保证」长程记忆，而是「允许」——遗忘门若学到 $\approx 0$，照样忘得快。**「门控给了结构上的可能，训练才让可能变成现实」**——遗忘门偏置初始化为 1、配合梯度裁剪，是让 LSTM 真正学会长依赖的工程配方。

## 4 公式解析：LSTM 单步的完整走查

把 LSTM 单步前向从头到尾走一遍，看清每个量的形状与作用。设输入 $\boldsymbol{x}_t\in\mathbb{R}^d$、隐状态与细胞状态 $\boldsymbol{h}_{t-1}, \boldsymbol{c}_{t-1}\in\mathbb{R}^h$，拼接向量 $[\boldsymbol{h}_{t-1}; \boldsymbol{x}_t]\in\mathbb{R}^{h+d}$：

$$
\boldsymbol{f}_t = \sigma\big(\boldsymbol{W}_f[\boldsymbol{h}_{t-1}; \boldsymbol{x}_t] + \boldsymbol{b}_f\big) \in (0,1)^h
$$

$$
\boldsymbol{i}_t = \sigma\big(\boldsymbol{W}_i[\boldsymbol{h}_{t-1}; \boldsymbol{x}_t] + \boldsymbol{b}_i\big) \in (0,1)^h
$$

$$
\tilde{\boldsymbol{c}}_t = \tanh\big(\boldsymbol{W}_c[\boldsymbol{h}_{t-1}; \boldsymbol{x}_t] + \boldsymbol{b}_c\big) \in (-1,1)^h
$$

$$
\boldsymbol{c}_t = \boldsymbol{f}_t \odot \boldsymbol{c}_{t-1} + \boldsymbol{i}_t \odot \tilde{\boldsymbol{c}}_t \in \mathbb{R}^h
$$

$$
\boldsymbol{o}_t = \sigma\big(\boldsymbol{W}_o[\boldsymbol{h}_{t-1}; \boldsymbol{x}_t] + \boldsymbol{b}_o\big) \in (0,1)^h, \qquad
\boldsymbol{h}_t = \boldsymbol{o}_t \odot \tanh(\boldsymbol{c}_t) \in \mathbb{R}^h
$$

- **第一步，看维度**：四个 $\boldsymbol{W}_*$ 都是 $h\times(h+d)$，输出全是 $h$ 维——**「拼接输入 → 四个投影 → 三个门 + 一个候选」是 LSTM 的固定骨架**。
- **第二步，看信息流**：$\boldsymbol{c}_t$ 由「旧记忆 × 遗忘门 + 新候选 × 输入门」合成——**加法路径**是记忆的「传送带」；$\boldsymbol{h}_t$ 由「新记忆经 $\tanh$ 再 × 输出门」读出——**输出门是「记忆 → 对外可见」的闸门**。
- **第三步，看参数量**：LSTM 的参数量约是普通 RNN 的 **4 倍**（四组投影 $\boldsymbol{W}_f,\boldsymbol{W}_i,\boldsymbol{W}_c,\boldsymbol{W}_o$）——**「能力更强」的代价是「参数更多」**。<span class="marginnote">「4 倍参数」是 LSTM 的经典代价：四个投影矩阵让参数量从 $4h^2$（RNN 的 $h^2$ 量级）翻到 $16h^2$ 量级（实际 4 组 $h\times(h+d)$）。这个「能力 vs 参数」的权衡，是后来 GRU 把它压到 3 个门的动机（下一节）。</span>

## 5 LSTM 的变体与实践

**Peephole 连接（窥视孔）**：把细胞状态 $\boldsymbol{c}_{t-1}$ 也喂给门控（让门「看到」记忆本身）——原始 LSTM 的增强，现代实现大多省略（收益不显著）。

**双向 LSTM**：前向 + 后向两个 LSTM，拼接两个方向的隐状态——让「上下文」同时看到过去与未来（见《双向 RNN》）。

**LSTM 的实践配方**：

- 遗忘门偏置初始化为 1（默认不忘记）。
- 梯度裁剪阈值 5 左右（LSTM 仍会爆炸，只是不消失）。
- 隐状态维 $h$ 是主要容量旋钮；深层 LSTM（堆叠）用残差连接辅助。

**易错点：** LSTM 的「记忆」是**有界**的（细胞状态经过 $\tanh$ 压缩到 $(-1,1)$ 才对外），但**细胞状态本身无界**（$\boldsymbol{c}_t$ 是累加，可能很大）——这导致「记忆的数值范围」需要处理（通常用 $\tanh$ 压缩输出）。**「内部可大、对外受限」**是 LSTM 记忆的微妙设计。

## 6 小结

- **LSTM** 引入细胞状态 $\boldsymbol{c}_t$ 的「传送带」，用**门控**（遗忘/输入/输出）控制「写什么、忘什么、读什么」。
- 三个门都是「Sigmoid 软开关」+ 独立权重；候选记忆用 $\tanh$。
- **梯度救星**：细胞状态雅可比是 $\text{diag}(\boldsymbol{f}_t)$——遗忘门接近 1 时梯度无损流过，「乘法路径」变「加法路径」。
- 遗忘门是「**可学习的梯度开关**」——网络主动决定保留/截断哪些路径。
- 参数量约普通 RNN 的 4 倍（四组投影）——能力与参数的权衡。
- 实践：遗忘门偏置初始化 1、梯度裁剪、双向拼接。

在下一节，我们看 LSTM 的「轻量版」——把三个门精简成两个，参数更少、效果相近，这就是 **GRU：门控循环单元**。
