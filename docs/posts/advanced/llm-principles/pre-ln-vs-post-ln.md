---
title: 残差连接与归一化的位置：Pre-LN 与 Post-LN 之争
date: 2026-08-07
---

# 残差连接与归一化的位置：Pre-LN 与 Post-LN 之争

<div class="epigraph">
<p>稳定不是自然发生的，而是设计出来的。</p>
<footer>—— 佚名（工程谚语）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型原理 ｜ GPT-2 / LLaMA 技术报告 / Xiong et al. 2020 ｜ 2026-08-07</p>
</div>

## 为什么一个"括号位置"值得一篇

Transformer 里每个子层外面包着「残差连接 + 层归一化」，而**归一化放在残差之前还是之后**，构成了 Pre-LN 与 Post-LN 两种架构流派。这个看似只有几行代码的差异，直接决定了几千层网络能否稳定训练——GPT-2 用 Post-LN，LLaMA 用 Pre-LN，今天几乎所有新模型都站队 Pre-LN。<span class="marginnote">Post-LN 是原版 Transformer 的设计：先残差加和、再对整个结果归一化（$\text{LN}(x + \text{Sublayer}(x))$）；Pre-LN 是「把 LayerNorm 放进残差分支」：先归一化、再做子层、最后加回输入（$x + \text{Sublayer}(\text{LN}(x))$）。差别就在 LayerNorm 挪到了子层之前。Xiong et al. 2020 从理论到实验论证了 Pre-LN 更适合深层训练。</span>

## 1 为什么需要残差连接

**残差连接（residual connection）** 让每个子层的输出等于「输入 + 子层的增量」：

$$
x' = x + \text{Sublayer}(x)
$$

它的两大作用：

- **梯度高速公路**：反向传播时，$\frac{\partial x'}{\partial x} = I + \frac{\partial \text{Sublayer}}{\partial x}$。即使子层梯度很小或很大，恒等项 $I$ 保证梯度能无损流过整条链——这是深层网络能训练的根本保障。
- **增量式学习**：每个子层只需学习「对输入的修正」，而不是从头构造输出。这让训练更平滑、收敛更快。

残差把「深层网络」从理论变成工程可能。没有残差，20 层以上的 Transformer 几乎无法训练。<span class="marginnote">残差的思想在 ResNet（2015）中首次引爆，Transformer 直接继承。它的数学本质是「恒等映射 + 扰动」，让信号像高速公路一样穿越深网——这是「从极限到大模型」里理解深度网络训练稳定性的第一块基石。</span>

## 2 Post-LN：原版的"先残差后归一化"

Post-LN 的顺序是：

$$
x' = \text{LayerNorm}\big(x + \text{Sublayer}(x)\big)
$$

即：先做残差加和，再对整个结果归一化。

**优点**：归一化作用在「完整输出」上，每个子层的输出分布严格受控（均值为 0、方差为 1），对后续层更「友好」。

**缺点**：梯度问题。残差分支里，LayerNorm 的输出直接进入下一层；反向时，梯度要穿过 LayerNorm 的归一化除法，其缩放随输入方差变化，**导致梯度在深层链路上放大或缩小，训练不稳定**。Post-LN 通常需要 **warmup + 小学习率** 才能驯服——这正是原版 Transformer 训练「很讲究」的原因。

## 3 Pre-LN：把归一化挪进分支

Pre-LN 的顺序是：

$$
x' = x + \text{Sublayer}\big(\text{LayerNorm}(x)\big)
$$

即：先对输入归一化，再做子层计算，最后加回原始输入。

**优点**：

- **训练更稳定**：残差主路径上只剩恒等加和，没有归一化除法，梯度以「恒等 + 小扰动」的方式稳定流动。实验表明 Pre-LN 在深层（如 60 层以上）仍能稳定训练，且 **warmup 不再是必需**。
- **实现更简洁**：许多框架用「残差流（residual stream）」视角理解 Pre-LN：所有信息沿着主路径流动，归一化与子层只是「读取当前状态、产生增量」。

**缺点**：**Post-LN 在同等层数下的最终性能通常略优**。Pre-LN 用一点最终性能，换取了训练的鲁棒性——这是一个典型的「可训练性 vs 容量」权衡。<span class="marginnote">有一类观点认为 Post-LN 的「不稳定」反而是一种隐式正则，让模型在深层学到更「紧」的表征；Pre-LN 太稳，深层表征可能冗余。这个争议在 T5 时代（Post-LN）与 LLaMA 时代（Pre-LN）之间反复横跳，最终工程稳定性获胜。</span>

## 4 公式解析：两种顺序的梯度行为

用一层的视角看两种结构对反向传播的影响。设子层为 $F$，损失为 $\mathcal{L}$。

**Post-LN**：$y = \text{LN}(x + F(x))$，反向时：

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial \text{LN}}{\partial (x + F(x))} \cdot \left(I + \frac{\partial F}{\partial x}\right)
$$

**Pre-LN**：$y = x + F(\text{LN}(x))$，反向时：

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} + \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial F}{\partial \text{LN}(x)} \cdot \frac{\partial \text{LN}}{\partial x}
$$

对这两条式子做三步拆解：

- **第一步，对比第一条的 $\frac{\partial \text{LN}}{\partial \cdot}$**：Post-LN 的梯度**必须穿过** LayerNorm 的归一化除法——这个因子约等于「除以输入标准差」，当残差流的方差波动时，梯度被整体缩放，多层叠加后剧烈震荡。**归一化除法的位置是 Post-LN 不稳定的根因。**
- **第二步，对比第二条的「恒等项直接相加」**：Pre-LN 的梯度是「恒等 1 + 小扰动项」，主路径上根本没有归一化除法。即使 $F$ 的梯度很小或很大，恒等项保证梯度 $O(1)$ 地流过每一层——**这就是稳定性的来源。**
- **第三步，读出权衡**：Pre-LN 更稳但性能略低，Post-LN 更强但娇贵。工程上常先 Pre-LN 保训练，再在最后阶段尝试切 Post-LN 或者用「sandwich」混合（归一化前后都放）来补性能。

**辨析｜易错点：** Pre-LN 不等于「不归一化」。它只是把归一化从「输出侧」挪到「输入侧」。两者都要 LayerNorm，只是位置不同——区别如同「先洗手再吃饭」与「吃完再洗手」，流程不同，都要洗。

## 5 现代架构的位置选择

| 架构 | 归一化位置 | 备注 |
| --- | --- | --- |
| Transformer 原版 | Post-LN | 需 warmup，深度有限 |
| GPT-2 / GPT-3 | Post-LN + 缩放初始化 | 用特殊初始化缓解 |
| GPT-J / LLaMA / Mistral / Qwen | Pre-LN（LLaMA 用 RMSNorm 替代 LN） | 主流，稳定优先 |
| T5 | Pre-LN（decoder 段） | 经验转向 |
| Sandwich-LN | 前 + 后双归一化 | 极端稳定，性能待验证 |

**归一化后置到「最终输出」也是常态**：GPT 在最后一层之后还会加一个最终 LayerNorm，把 logits 之前的表示稳定住。所以「Pre-LN」准确说是「子层内部 Pre-LN + 全模型末尾一个 LN」的组合。

## 6 术语速查表

| 术语 | 英文 | 一句话定义 |
| --- | --- | --- |
| 残差连接 | residual connection | $x + \text{Sublayer}(x)$ |
| Pre-LN | Pre-LN | 归一化在子层之前 |
| Post-LN | Post-LN | 归一化在残差和之后 |
| 恒等项 | identity | 梯度的高速公路 |
| warmup | warmup | 学习率预热阶段 |
| 残差流 | residual stream | 主路径上累积的表示 |

## 7 数值算例：60 层的梯度差异

设每层 LayerNorm 对梯度的缩放因子为 $\lambda$，60 层后梯度放大/缩小为 $\lambda^{60}$：

| $\lambda$ | $\lambda^{60}$ | 后果 |
| --- | --- | --- |
| 0.83 | 约 $1.4\times10^{-5}$ | 梯度消失 |
| 0.95 | 约 0.046 | 明显缩小 |
| 1.05 | 约 18 | 明显放大 |

**读这张表**：Post-LN 的梯度要穿过「每层的 LN 除法」，缩放因子偏离 1 一点，60 层后就是几个数量级的偏差——**这就是 Post-LN 深层不稳定的数学本质**。Pre-LN 把 LN 从主路径移走，梯度以「恒等 + 小扰动」流动，不经历这个连乘。

**辨析｜易错点：** 「Post-LN 不稳定」不等于「Post-LN 一定训不出来」——用「特殊初始化 + warmup + 小学习率」可以缓解（GPT-3 就是这么干的）。**稳定性是「工程权衡」不是「绝对二选一」**——现代模型选 Pre-LN 只是因为「稳定更便宜」。

## 8 归一化位置选型速查

| 场景 | 选择 | 理由 |
| --- | --- | --- |
| 深层（60+ 层） | Pre-LN | 稳定压倒一切 |
| 中深层（12–32 层） | Pre-LN | 稳妥主流 |
| 追求上限性能 | Post-LN + 特殊初始化 | 需精细调参 |
| 超深（100+ 层） | Sandwich / DeepNorm | 极端稳定 |

**选型原则**：**先保证能训出来，再谈更好**——Pre-LN 是「保底」，Post-LN 是「上限但娇贵」。

## 9 小结

- **残差连接**保证梯度以「恒等 + 扰动」方式流动，是深层网络可训练的根本。
- **Post-LN**：先残差加和、再归一化，输出严格受控但梯度穿归一化除法，**不稳定、需 warmup**。
- **Pre-LN**：先归一化、再做子层、最后加回输入，梯度主路径无除法，**稳定、免 warmup**，但最终性能略低。
- 现代模型（LLaMA、Qwen、Mistral）几乎全用 **Pre-LN + RMSNorm**。
- 权衡本质：**可训练性 vs 容量**，工程稳定性胜出。

在下一节，我们把「归一化」单独放大——**LayerNorm 与 BatchNorm**：为什么 NLP 离不开前者。
