---
title: GRU：门控循环单元
date: 2026-08-07
---

# GRU：门控循环单元

<div class="epigraph">
<p>少即是多：把三个开关减成两个，常常一样好用。</p>
<footer>—— 依据「简洁」设计哲学的精神改写</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ 李沐《动手学深度学习》§8.9、Cho 等（2014） ｜ 2026-08-07</p>
</div>

## 为什么从 GRU 开始

LSTM 用三个门解决了长程记忆，但代价是复杂的结构与约 4 倍于 RNN 的参数。**门控循环单元（Gated Recurrent Unit, GRU）**（Cho 等, 2014）把三个门精简成**两个门**——**重置门（reset gate）**与**更新门（update gate）**——并且**去掉了独立的细胞状态**，直接用隐状态承担记忆。GRU 参数更少、结构更简，却常常达到与 LSTM 相当的效果，成为「轻量级 LSTM」的标准选择。

GRU 的「精简」不是简单删减，而是**重新设计信息流**：更新门同时扮演「遗忘」与「输入」两个角色（旧的忘多少 = 新的进多少），重置门控制「候选记忆如何利用旧状态」。这个「合二为一」的设计，让 GRU 用两个门实现了 LSTM 三个门的核心功能。本节把 GRU 的机制、与 LSTM 的精确对比、以及「什么时候选 GRU」讲透。<span class="marginnote">GRU 诞生于神经机器翻译（Neural Machine Translation）的早期——Cho 等人在 Seq2Seq 论文里首次提出 GRU，用更少的参数解决翻译的长依赖。它后来成为「轻量序列模型」的代表：参数量小、训练快、在小数据上不易过拟合，在语音、文本、时序预测里广泛使用。</span>

## 1 从三个门到两个门：精简的哲学

LSTM 的三个门各自「专司其职」：遗忘门管「忘多少旧记忆」、输入门管「写多少新信息」、输出门管「读出多少」。GRU 的观察是：**「忘多少旧」与「写多少新」本质上是一件事的两面**——新的进来了，旧的相对就少了。于是：

- **更新门（update gate）$\boldsymbol{z}_t$**：合并「遗忘 + 输入」——$\boldsymbol{z}_t$ 控制「保留多少旧状态」，同时「$1-\boldsymbol{z}_t$」就是「写入多少新信息」。
- **重置门（reset gate）$\boldsymbol{r}_t$**：控制「计算新候选时，利用多少旧状态」。

**「合二为一」**：GRU 用「保留旧状态的比例」这一个量，同时表达「忘与记」——这是它参数更少的根源。

**易错点：** 别把 GRU 的「更新门」误解为「就是遗忘门」。更新门 $\boldsymbol{z}_t$ 是「**旧状态与新状态的比例开关**」：$\boldsymbol{z}_t$ 大 → 保留多旧状态、少写新信息；$\boldsymbol{z}_t$ 小 → 少留旧、多写新。它同时是「遗忘门」与「输入门」——**「忘多少」与「记多少」绑定在一起**（$1-\boldsymbol{z}_t$）。

## 2 GRU 的公式：两个门 + 一个候选

GRU 的计算流程（隐状态 $\boldsymbol{h}_t\in\mathbb{R}^h$，输入 $\boldsymbol{x}_t\in\mathbb{R}^d$，拼接 $[\boldsymbol{h}_{t-1}; \boldsymbol{x}_t]$）：

**重置门**——决定「旧状态对候选记忆的影响」：

$$
\boldsymbol{r}_t = \sigma(\boldsymbol{W}_r[\boldsymbol{h}_{t-1}; \boldsymbol{x}_t] + \boldsymbol{b}_r)
$$

**更新门**——决定「新旧状态的比例」：

$$
\boldsymbol{z}_t = \sigma(\boldsymbol{W}_z[\boldsymbol{h}_{t-1}; \boldsymbol{x}_t] + \boldsymbol{b}_z)
$$

**候选隐状态**——用「重置后的旧状态」生成新候选：

$$
\tilde{\boldsymbol{h}}_t = \tanh\big(\boldsymbol{W}_h[\boldsymbol{r}_t \odot \boldsymbol{h}_{t-1}; \boldsymbol{x}_t] + \boldsymbol{b}_h\big)
$$

**最终隐状态**——更新门在旧状态与候选之间插值：

$$
\boldsymbol{h}_t = \boldsymbol{z}_t \odot \boldsymbol{h}_{t-1} + (1 - \boldsymbol{z}_t) \odot \tilde{\boldsymbol{h}}_t
$$

**读法**：先算两个门；用重置门「调节旧状态的利用程度」生成候选记忆；最后用更新门「在旧状态与候选之间取比例」。**整个单元没有独立的细胞状态**——隐状态自己既是「记忆」也是「输出」。<span class="marginnote">「重置门 $\boldsymbol{r}_t$ 的角色」值得细品：$\boldsymbol{r}_t \odot \boldsymbol{h}_{t-1}$ 是「把旧状态按比例重置后再进候选」。当 $\boldsymbol{r}_t\approx 0$ 时，候选「几乎不看旧状态」，只由当前输入生成——这允许网络「忘记前文、只看当前」（适合「句子开头的新主题」）；当 $\boldsymbol{r}_t\approx 1$ 时，候选充分利用旧状态——「延续前文」。<strong>重置门控制「候选记忆与前文的耦合度」</strong>。</span>

**易错点：** GRU 的更新门 $\boldsymbol{z}_t$ 与 LSTM 的遗忘门 $\boldsymbol{f}_t$ 虽然都在「旧记忆」前当系数，但语义不同：LSTM 的 $\boldsymbol{f}_t$ 只管「忘」，新信息由输入门独立控制；GRU 的 $\boldsymbol{z}_t$ 同时管「忘多少」与「记多少」（互补）。**「GRU 的参数更少，因为两个角色被绑定」**。

## 3 GRU 为什么能解决梯度问题

GRU 与 LSTM 共享同一个「梯度救星」机制：**隐状态的更新有加法路径**。

$$
\boldsymbol{h}_t = \boldsymbol{z}_t \odot \boldsymbol{h}_{t-1} + (1-\boldsymbol{z}_t) \odot \tilde{\boldsymbol{h}}_t
$$

对 $\boldsymbol{h}_{t-1}$ 的雅可比：

$$
\frac{\partial \boldsymbol{h}_t}{\partial \boldsymbol{h}_{t-1}} = \text{diag}(\boldsymbol{z}_t) + \text{（含 $\tilde{\boldsymbol{h}}_t$ 对 $\boldsymbol{h}_{t-1}$ 的项）}
$$

- **第一步，看主项**：第一项 $\text{diag}(\boldsymbol{z}_t)$ 是「旧状态直达新状态」的**加法路径**——当 $\boldsymbol{z}_t\approx 1$ 时，梯度几乎无损流过（与 LSTM 的遗忘门同效）。
- **第二步，看差异**：LSTM 的细胞状态路径是「纯」的（$\boldsymbol{c}_t = \boldsymbol{f}_t\odot\boldsymbol{c}_{t-1}+\cdots$，无 $\tanh$ 压缩）；GRU 的隐状态路径要经过「候选 $\tilde{\boldsymbol{h}}_t$ 里的 $\tanh$」——**GRU 的梯度路径比 LSTM 稍「脏」一点**。
- **第三步，读实践**：理论上 LSTM 的「纯加法路径」更利于超长记忆；实践上 GRU 与 LSTM 在大多数任务上**效果相当**——「理论优势」没有转化为「明显实践差距」。<span class="marginnote">「LSTM vs GRU 谁更好」是序列建模的经典问题，结论大致是：<strong>在大多数任务上两者相当，GRU 参数少、训练快；在「需要极长记忆」的任务上 LSTM 的纯加法路径略优</strong>。2018 年前后的大规模实证（如 Google 的 systematic evaluation）倾向 GRU「性价比更高」，但 LSTM 在「门控更丰富」的场景（如带 peephole 的变体）仍有拥护者——「选哪个」常由「参数量预算」决定。</span>

**易错点：** GRU 的「重置门」在计算候选时**逐元素**作用于旧状态（$\boldsymbol{r}_t\odot\boldsymbol{h}_{t-1}$），不是「整段重置」——它是「每个维度独立决定要不要看旧状态」。**「逐元素的门控」是 GRU/LSTM 的通用语言**。

## 4 公式解析：GRU 的信息流（对比 LSTM）

把 GRU 与 LSTM 的信息流并排，看清「简化」到底简化了什么：

| 步骤 | LSTM | GRU |
| --- | --- | --- |
| 门控数 | 3（遗忘/输入/输出） | 2（更新/重置） |
| 记忆载体 | 细胞状态 $\boldsymbol{c}_t$ + 隐状态 $\boldsymbol{h}_t$ | 只有隐状态 $\boldsymbol{h}_t$ |
| 旧记忆处理 | 遗忘门：$\boldsymbol{f}_t\odot\boldsymbol{c}_{t-1}$ | 更新门：$\boldsymbol{z}_t\odot\boldsymbol{h}_{t-1}$ |
| 新信息处理 | 输入门：$\boldsymbol{i}_t\odot\tilde{\boldsymbol{c}}_t$ | $(1-\boldsymbol{z}_t)\odot\tilde{\boldsymbol{h}}_t$ |
| 对外输出 | 输出门：$\boldsymbol{h}_t = \boldsymbol{o}_t\odot\tanh(\boldsymbol{c}_t)$ | 直接：$\boldsymbol{h}_t$ 即输出 |
| 参数量 | 4 组投影（约 4 倍 RNN） | 3 组投影（约 3 倍 RNN） |

- **第一步，看「记忆载体」的合并**：LSTM 用「细胞状态存记忆 + 隐状态做输出」两个角色；GRU **合并成一个隐状态**——少一组状态，少一组输出门。
- **第二步，看「忘与记」的合并**：LSTM 的「忘 $\boldsymbol{f}_t$」与「记 $\boldsymbol{i}_t$」是**两个独立**的开关；GRU 的「$\boldsymbol{z}_t$ 与 $1-\boldsymbol{z}_t$」是**互补**的——少一个门。
- **第三步，看参数**：三组投影（$\boldsymbol{W}_r,\boldsymbol{W}_z,\boldsymbol{W}_h$）对四组（$\boldsymbol{W}_f,\boldsymbol{W}_i,\boldsymbol{W}_o,\boldsymbol{W}_c$）——**GRU 省约 25% 参数**，训练更快、更不易过拟合。<span class="marginnote">「合并」的代价与收益：GRU 少了一个「独立输入门」——LSTM 可以「同时大量写入新信息 + 完全不遗忘旧信息」（$\boldsymbol{i}_t=1, \boldsymbol{f}_t=1$ 同时成立），GRU 做不到（$\boldsymbol{z}_t$ 决定了新旧比例，写新必忘旧）。这个「表达自由度」的差异，在「需要同时保留旧记忆 + 大量吸收新信息」的极端场景下，LSTM 略占优——但绝大多数任务用不到这种极端。</span>

**易错点：** GRU 没有「细胞状态」，所以也没有「输出门」——它的隐状态**直接对外**。这意味着「GRU 的隐状态既是记忆又是输出」，信息「藏不住」（不像 LSTM 可以把记忆藏在细胞状态里、输出门决定泄露多少）。**「LSTM 的记忆可以『私密』，GRU 的记忆是『公开』的」**——这个差异在极少数任务上可感知。

## 5 GRU 的实践与选型

**GRU 的实践配方**：

- 参数量比 LSTM 少约 25%，在小数据上更稳。
- 训练速度更快（少一组投影）。
- 与 LSTM 一样需要梯度裁剪、合适的初始化（遗忘/更新门偏置）。

**选型建议**：

| 场景 | 推荐 |
| --- | --- |
| 默认序列模型、参数量敏感 | GRU |
| 需要极长记忆（超长文档） | LSTM（纯加法路径） |
| 大数据 + 大模型（参数量不是瓶颈） | LSTM 或 GRU 均可 |
| 现代 LLM（Transformer 时代） | 都不用了，注意力取代 |

**「GRU 与 LSTM 的竞争，最终被 Transformer 终结」**——2017 年 Transformer 用注意力同时解决「长程记忆」与「并行计算」，RNN 家族（含 LSTM/GRU）退居次要。但 GRU 的「精简设计」思想——「用最少的门达到目标」——至今仍在轻量模型里延续。<span class="marginnote">「RNN 家族被 Transformer 取代」的完整叙事：LSTM 解决「长程记忆」（加法路径）、GRU 精简「门控」（两个门）、Transformer 解决「并行」与「任意位置访问」（注意力）——三个问题被逐一攻克，而注意力「一步到位」同时解决了后两者（长程 + 并行）。理解「每个结构在解决什么」，才能看懂「为什么最终是注意力胜出」（第六篇）。</span>

## 6 小结

- **GRU**：两个门（更新 $\boldsymbol{z}_t$、重置 $\boldsymbol{r}_t$），无独立细胞状态，隐状态即记忆。
- **更新门** 合并「忘与记」：$\boldsymbol{z}_t$ 保留旧、$1-\boldsymbol{z}_t$ 写入新；**重置门** 控制「候选与前文的耦合」。
- **梯度救星**：隐状态更新有加法路径 $\text{diag}(\boldsymbol{z}_t)$——长程梯度可无损流过。
- 与 LSTM 对比：3 组投影 vs 4 组，参数省约 25%；大多数任务效果相当。
- LSTM 的「纯加法路径 + 独立输入门」在极端长记忆场景略优。
- 选型：参数量敏感选 GRU；RNN 家族最终被 Transformer 的注意力取代。

在下一节，我们看 RNN 的「组合玩法」——堆叠变深、双向看上下文，这就是**深层循环神经网络与双向循环神经网络**。
