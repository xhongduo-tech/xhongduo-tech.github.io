---
title: 实数集的基数不变量
date: 2026-08-07
---

# 实数集的基数不变量

<div class="epigraph">
<p>连续统不是一个数，而是一族相互牵制的量：它们共享同一条边界，却不肯听同一个指令。</p>
<footer>—— 埃里克 · 范 · 道仁（Eric van Douwen）</footer>
</div>

<div class="article-byline">
<p>第二级 · 公理集合论与模型论 ｜ Jech, <em>Set Theory</em> 第19章；Kunen 第4章 ｜ 2026-08-07</p>
</div>

## 为什么从基数不变量开始

上一篇我们看到，实数集的「正则性」被 AC 撬动。但「实数集有多大」这个问题本身，还藏着更精细的结构：$2^{\aleph_0}$（连续统的势）只是一个数，而实数集的**各种子结构**——可数个贫集要多少个才能盖满、多少个零测集才能拼出全部——各自问出一个独立的基数，这些基数彼此由不等式牵制，构成一张「不变量之网」。<span class="marginnote">这些量叫<strong>基数不变量（cardinal invariants / characteristics of the continuum）</strong>。它们都夹在 $\aleph_0$ 与 $2^{\aleph_0}$ 之间，但互不相等（多数情况下），于是 CH 的成立让它们全部坍缩成同一个数——这是 CH 最惊人的后果之一：它让整张网消失。</span>

今天先立起四个最核心的量：**可加性 $\mathrm{add}$**、**覆盖数 $\mathrm{cov}$**、**非定数 $\mathrm{non}$** 与**共尾性 $\mathrm{cof}$**（对零测集与贫集各一套，合称 Cichoń 格局），再画出它们之间的 ZFC 不等式，最后看力迫如何让这些量取到不同值。这一节是下一篇（第3篇力迫法）最直接的动机：**力迫能独立地调节这些量**。

## 1 四个基本不变量

设 $\mathcal{I}$ 是 $\mathbb{R}$ 上的一个「理想」——直觉上是被判为「小」的集合族，典型例子是**零测集理想 $\mathcal{N}$** 与**贫集理想 $\mathcal{M}$**。定义四个基数：

$$
\mathrm{add}(\mathcal{I}) = \min\{|\mathcal{F}| : \mathcal{F} \subseteq \mathcal{I}, \; \bigcup \mathcal{F} \notin \mathcal{I}\}
$$

最小的「并集跳出 $\mathcal{I}$ 的族的大小」——需要多少个 $\mathcal{I}$-小集合才能拼出一个 $\mathcal{I}$-大集合。

$$
\mathrm{cov}(\mathcal{I}) = \min\{|\mathcal{F}| : \mathcal{F} \subseteq \mathcal{I}, \; \bigcup \mathcal{F} = \mathbb{R}\}
$$

最小的「盖满全实数集的 $\mathcal{I}$-族」的大小——多少个 $\mathcal{I}$-小集合能覆盖 $\mathbb{R}$。

$$
\mathrm{non}(\mathcal{I}) = \min\{|A| : A \subseteq \mathbb{R}, \; A \notin \mathcal{I}\}
$$

最小的「不是 $\mathcal{I}$-小」的集合的大小。

$$
\mathrm{cof}(\mathcal{I}) = \min\{|\mathcal{F}| : \mathcal{F} \subseteq \mathcal{I}, \; \forall B \in \mathcal{I} \;\exists A \in \mathcal{F}, \; B \subseteq A\}
$$

最小的「能包含一切 $\mathcal{I}$-小集合」的族的大小——$\mathcal{I}$ 的共尾性。<span class="marginnote">这四个量可以读成四问：要多少小集合拼成大？（add）要多少小集合盖满？（cov）最小的不小集合多大？（non）要用多少小集合才能包罗全部小集合？（cof）——四问分别抓住理想的「加法」「覆盖」「非小」「共尾」四个侧面。</span>

对 $\mathcal{N}$（零测集）与 $\mathcal{M}$（贫集）各取这四量，加上 $2^{\aleph_0}$、$\aleph_0$、$\aleph_1$、$\mathrm{cf}(2^{\aleph_0})$，共同构成**Cichoń 格局**。

## 2 基本不等式：不变量如何相互牵制

对任意理想 $\mathcal{I}$（含有限集、封闭于可数并的适当条件），ZFC 中成立

$$
\aleph_0 \lt  \mathrm{add}(\mathcal{I}) \le \mathrm{cov}(\mathcal{I}) \le \mathrm{cof}(\mathcal{I}), \qquad
\aleph_0 \lt  \mathrm{add}(\mathcal{I}) \le \mathrm{non}(\mathcal{I}) \le \mathrm{cof}(\mathcal{I})
$$

并通常还有 $\mathrm{non}(\mathcal{I}) \le 2^{\aleph_0}$、$\mathrm{cov}(\mathcal{I}) \le 2^{\aleph_0}$。直觉：

- $\mathrm{add} \le \mathrm{cov}$：若要 $\lt \mathrm{cov}$ 个集合盖满 $\mathbb{R}$，它们作为 $\mathcal{I}$-族「并成 $\mathbb{R}$」就跳出 $\mathcal{I}$，所以 add 被 cov 卡住上界。
- $\mathrm{non} \le \mathrm{cof}$：一个「非小」集合 $A$ 本身不可能是任何「共尾族」的成员之并，故 non 提供 cof 的下界。

**要点**：这些不等式的关键推论是「在 ZFC 中不可能让 add 大于 cov」等——任何力的迫模型都必须遵守这张网。**Cichoń 格局**把这些量排成一张图：$\aleph_0 \le \mathrm{add}(\mathcal{M}) \le \mathrm{cov}(\mathcal{M}) \le \cdots$，其中一些不等式（如 $\mathrm{cov}(\mathcal{N}) \le \mathrm{non}(\mathcal{M})$ 等）由测度与范畴的**对偶**（Lebesgue 测度与 Baire 范畴的「几乎处处」对偶）给出，形成一张高度缠结的网。<span class="marginnote">Cichoń 格局最著名的「缺口」：$|\mathrm{cov}(\mathcal{N})|$ 与 $|\mathrm{non}(\mathcal{M})|$ 之间，力迫可以自由选择让谁大谁小——它们之间没有 ZFC 的不等式。这正是力迫法「双调节」能力的窗口。</span>

**辨析｜易错点：** 四个量不必两两可比——$\mathrm{cov}(\mathcal{N})$ 与 $\mathrm{non}(\mathcal{N})$ 之间无 ZFC 不等式（可大、可小、可相等）。初学者常误以为「一张网必是全序」，但基数不变量往往只是偏序。

## 3 两个可判定的不变量：$\mathfrak{b}$ 与 $\mathfrak{d}$

除了理想诱导的四量，还有两个来自「函数序」的不变量，在 Cichoń 格局里特别有名：

$$
\mathfrak{b} = \min\{|F| : F \subseteq \omega^\omega, \; F \text{ 无界}\}, \qquad
\mathfrak{d} = \min\{|F| : F \subseteq \omega^\omega, \; F \text{ 支配}\}
$$

其中 $f \le^* g$ 若 $f(n) \le g(n)$ 对**所有充分大的 $n$** 成立（「最终」比较）；$F$ 无界若没有单个 $g$ 最终支配所有 $f \in F$；$F$ 支配若每个 $f$ 都被某个 $g \in F$ 最终支配。

直觉：$\mathfrak{b}$ 是「要多少函数才造不出上界」，$\mathfrak{d}$ 是「要多少函数才能罩住一切」。ZFC 中

$$
\aleph_1 \le \mathfrak{b} \le \mathfrak{d} \le 2^{\aleph_0}
$$

且 $\mathrm{cov}(\mathcal{M}) \le \mathfrak{d}$、$\mathfrak{b} \le \mathrm{non}(\mathcal{M})$。<span class="marginnote">$(\omega^\omega, \le^*)$ 是「模有限修正」的函数格。$\mathfrak{b}$ 与 $\mathfrak{d}$ 分别衡量这个格的「反链」与「共尾」；它们在力迫里的调节最直观——加 Cohen 实数会增大 $\mathfrak{d}$ 而保持 $\mathfrak{b}$ 可数大，加 Hechler 实数则反之。</span>

**辨析｜易错点：** 最终支配 $\le^*$ 与逐点支配 $\le$ 不同——逐点支配要求对**所有** $n$ 成立，最终支配只要求「在无穷尾部」。$\mathfrak{b}$、$\mathfrak{d}$ 用的是最终支配；混用会导致不变量出错。

## 4 公式解析：一张 Cichoń 格局

把核心不等式画成一张「格局表」：

$$
\begin{array}{c}
\aleph_0 \le \mathfrak{b} \le \mathrm{add}(\mathcal{M}) \le \mathrm{cov}(\mathcal{M}) \le \mathfrak{d} \le \mathrm{cof}(\mathcal{M}) \le 2^{\aleph_0} \\
\aleph_0 \le \mathrm{add}(\mathcal{N}) \le \mathrm{cov}(\mathcal{N}) \le 2^{\aleph_0} \\
\aleph_0 \le \mathrm{non}(\mathcal{M}) \le \mathrm{non}(\mathcal{N}) \le \mathrm{cof}(\mathcal{N}) \le 2^{\aleph_0}
\end{array}
$$

- **$\aleph_0 \le$ 一切**：理想必含可数集且不封闭于可数并之外，故 add、cov 等至少 $\aleph_1$（$\aleph_1 \le \aleph_0$ 不可能）。
- **$\mathfrak{b} \le \mathrm{add}(\mathcal{M})$**：贫集的可加性被「无界函数数」卡住——想拼出非贫集，至少要能造出足够多无界函数。
- **$\mathrm{cov}(\mathcal{M}) \le \mathfrak{d}$**：用支配族能盖满贫集——每个贫集都被某个支配族「罩住」。
- **缺口**：$\mathrm{cov}(\mathcal{N})$ 与 $\mathrm{non}(\mathcal{M})$ 之间（以及 $\mathrm{cov}(\mathcal{N})$ 与 $\mathrm{non}(\mathcal{N})$ 之间）**没有 ZFC 不等式**——力迫可让 $\mathrm{cov}(\mathcal{N}) \lt  \mathrm{non}(\mathcal{N})$ 或反之。

**要点**：格局不是一个线性链，而是一张「几乎链」——绝大多数量由不等式串起，但留下的缺口恰恰是力迫的自由度。CH 成立时全部坍缩为 $\aleph_1 = 2^{\aleph_0}$，网消失；力迫可让它们取不同的中间值。

## 6 动手推导：$\mathfrak{b}$ 与 $\mathfrak{d}$ 的最小约束

把「无界族」与「支配族」的定义落在一个具体例子上，感受 $\mathfrak{b} \le \mathfrak{d}$ 为什么总成立。

- **第一步，$\mathfrak{b} \le \mathfrak{d}$**：设 $F$ 是支配族（$|F| = \mathfrak{d}$）。若 $F$ 有界（被某个 $g$ 最终支配），则 $g$ 支配一切——这与「$F$ 是支配族」不矛盾，但它意味着……实际上要证 $\mathfrak{b} \le \mathfrak{d}$：任取一个最小支配族 $F$（大小 $\mathfrak{d}$），若 $F$ 本身无界，则 $F$ 也是无界族，故 $\mathfrak{b} \le |F| = \mathfrak{d}$；若 $F$ 有界，则全体 $\omega^\omega$ 被 $g$ 最终支配，此时 $\mathfrak{d}$ 退化到 $1$，平凡。所以总归 $\mathfrak{b} \le \mathfrak{d}$。
- **第二步，最小例子**：$F = \{f_n : f_n(k) = n \cdot k\}$ 是支配族吗？对任意 $h$，取 $n > \sup_{k\le m} h(k)/k$……——用最终支配检查，$f_n(k) = nk$ 对充分大的 $k$ 会超过 $h(k)$。这个族的无界性、支配性可直接算。
- **第三步，与 $\aleph_0$ 的关系**：$\mathfrak{b} \ge \aleph_1$ 是因为：可数多个函数 $\{f_n\}$ 总有共同上界（逐点取 $\sup$ 再对角化），故无界族至少不可数。$\mathfrak{d} \le 2^{\aleph_0}$ 因为全体函数只有 $2^{\aleph_0}$ 个。
- **第四步，要点**：$\mathfrak{b}$ 与 $\mathfrak{d}$ 是「函数空间的结构常数」——它们衡量「$\omega^\omega$ 模有限修正」这个格有多「宽」。加 Cohen 实数或 Hechler 实数能独立调节它们，是第3篇力迫在 Cichoń 格局上的具体操作。

**辨析｜易错点：** 无界族与支配族都要求「最终」而非「逐点」——$F$ 无界意味着「没有单个 $g$ 最终支配 $F$ 的一切」，不要求「对每个 $n$ 都超不过」。初学者常拿「逐点支配」检验，得出错误的无界/支配判断。

## 7 小结

- **四个理想不变量**：$\mathrm{add}$（小集合并出大的最少个数）、$\mathrm{cov}$（盖满的最少个数）、$\mathrm{non}$（最小非小集合）、$\mathrm{cof}$（共尾族的最小大小）。
- **Cichoń 格局**：对零测集理想 $\mathcal{N}$ 与贫集理想 $\mathcal{M}$ 各取四量，构成 ZFC 内的不等式网络。
- **函数序不变量**：$\mathfrak{b}$（无界族最小大小）、$\mathfrak{d}$（支配族最小大小），基于「最终支配」$\le^*$。
- 格局呈偏序而非全序：$\mathrm{cov}(\mathcal{N})$ 与 $\mathrm{non}(\mathcal{N})$ 等对之间无 ZFC 不等式。
- 力迫法（第3篇）能独立调节这些量，CH 使它们全部坍缩——这是连续统结构多样性的量化表述。

在下一节，我们转入本专题第3篇的开端：可构造宇宙 $L$。Gödel 如何把宇宙压成「可定义的最小模型」，又如何在其中证明 CH 与 AC——$L$