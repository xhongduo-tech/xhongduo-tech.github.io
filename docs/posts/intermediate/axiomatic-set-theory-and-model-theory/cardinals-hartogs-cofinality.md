---
title: 基数、Hartogs 数与共尾性
date: 2026-08-07
---

# 基数、Hartogs 数与共尾性

<div class="epigraph">
<p>势论中最先被问的问题是：一个集合比另一个集合大吗？答案写成一个数，就是基数。</p>
<footer>—— 格奥尔格 · 康托尔（Georg Cantor）</footer>
</div>

<div class="article-byline">
<p>第二级 · 公理集合论与模型论 ｜ Jech, <em>Set Theory</em> 第3章；Kunen 第3章 ｜ 2026-08-07</p>
</div>

## 为什么从基数开始

上一节我们有了序数——回答「第几」。可数学里更常用的问题是多大规模：**「有多少个」**。自然数里「第几个」和「多少个」是同一套数，可一旦踏入无穷，两者立刻分道扬镳：$\omega$ 与 $\omega+1$ 是**不同的序数**，却都有「可数无穷多个」元素。<span class="marginnote">基数（cardinal）回答「多少个」，序数（ordinal）回答「第几个」。康托尔 1874 年证明实数比自然数多，正是人类第一次证明「存在不同级别的无穷」——从此「无穷」不再是铁板一块。</span>

基数理论是理解 ZFC 计算力的入口：**Hartogs 数**说明「比给定基数更大的最小基数」无需选择公理就能造出来；**共尾性**则衡量「一个基数离它的下确界有多远」，它把基数划成正则与奇异两个物种。今天先立起基数的 von Neumann 定义，再拆开 Hartogs 数这个精巧构造，最后用共尾性为下一篇《正则/奇异基数与 König 定理》铺路。

## 1 基数：最小的那个序数

两个集合 $A, B$ **等势（equinumerous）**，记作 $|A| = |B|$，若存在双射 $A \to B$。等势是等价关系，而**基数（cardinal number）** 就是每个等价类里的「代表」——取哪个代表？von Neumann 的答案干净利落：

**基数**是一个序数 $\kappa$，满足：不存在比 $\kappa$ 更小的序数与 $\kappa$ 等势，即 $\forall \alpha \lt  \kappa$，$\alpha$ 与 $\kappa$ 不等势。

换句话说，**基数是「最早出现的那个规模」**。每个集合 $A$ 的基数 $|A|$ 定义为：与 $A$ 等势的最小的那个序数。<span class="marginnote">注意这里用「最小序数」而不必诉诸选择公理：若存在与 $A$ 等势的良序，其序型的最小者就是 $|A|$；Hartogs 数会告诉我们这种良序总是足够多（在 AC 之下，每个 $A$ 都能被良序化，见第2篇《选择公理》。</span>）有限基数就是自然数 $0,1,2,\dots$；第一个无限基数是 $\omega$，记作

$$
\aleph_0 = \omega
$$

$\aleph_0$ 之后，按大小把无限基数排成一串：

$$
\aleph_0 \lt  \aleph_1 \lt  \aleph_2 \lt  \cdots \lt  \aleph_\alpha \lt  \cdots
$$

其中 $\aleph_{\alpha+1}$ 被定义为「严格大于 $\aleph_\alpha$ 的最小基数」，极限处 $\aleph_\lambda = \sup_{\alpha\lt \lambda} \aleph_\alpha$。<span class="marginnote">$\aleph$（读作 aleph）是希伯来字母，康托尔用来表示无穷基数。$\aleph_1$ 是「比可数无穷大的最小无穷」。连续统假设（CH）断言 $\aleph_1 = 2^{\aleph_0}$，即实数集正好取这个最小的不可数势——它是本专题第3篇的主角。</span>

**要点**：基数自带「极简主义」——一个基数就是「它这一类等势序数中最小的那个」。$\omega$ 与 $\omega+1$ 等势（$n \mapsto n+1$ 再补 $0 \mapsto 0$ 的移位双射），所以 $\omega$ 是基数而 $\omega+1$ 不是。

## 2 Hartogs 数：不需要选择公理的「下一个基数」

从 $\aleph_0$ 到 $\aleph_1$ 有一件微妙的事：**凭什么存在「比可数无穷更大的最小基数」？** 幂集 $\mathcal{P}(\omega)$ 确实比 $\omega$ 大（康托尔定理），但它是否恰好是「最小的更大基数」无关紧要——我们要的是那个「最小的更大基数」，而它可以用一个纯组合的构造造出来：

**Hartogs 定理**：对任意集合 $X$，都存在一个基数 $\kappa$，使得不存在单射 $\kappa \to X$；且这样的基数中存在最小者，记作 $\aleph(X)$（Hartogs 数）。

证明的骨架：考虑所有「良序集 $(A, \prec)$ 满足 $A \subseteq X$」的序型。由替换公理，这些序型构成一个序数的集合 $W$，而 $W$ 是传递且按 $\in$ 良序的，故 $W$ 本身是一个序数——记作 $\aleph(X)$。若存在单射 $\aleph(X) \to X$，则 $\aleph(X)$ 能被良序化后嵌入 $X$，于是它自己也成了「序型属于 $W$」的序数，矛盾于 $W$ 是序数（因为 $\alpha \in W \Rightarrow \alpha \lt  W$ 而 $W \notin W$）。<span class="marginnote">Hartogs 数 $\aleph(X)$ 是「不可能嵌入 $X$ 的最小序数」。对 $X = \omega$，它就是 $\aleph_1$。关键点：构造只用了替换公理与良序化子集，<strong>完全不依赖选择公理</strong>——即使没有 AC，$\aleph_1$ 依然存在。</span>

**辨析｜易错点：** 别以为「$|\mathcal{P}(X)|$ 就是 $X$ 的下一个基数」。$|\mathcal{P}(X)|$ 是某个基数，但它未必是最小的更大基数；$X$ 与 $\mathcal{P}(X)$ 之间可能还隔着别的基数（CH 悬而未决正是因为它拒绝回答这个问题）。Hartogs 数才是「最小更大基数」的精确化身。

## 3 共尾性：从一个子集能不能赶超

设 $\kappa, \lambda$ 是基数（或极限序数）。一个映射 $f: \lambda \to \kappa$ 称为**无界（unbounded）**的，若 $f$ 的像在 $\kappa$ 中没有上界：$\forall \xi \lt  \kappa \;\exists \alpha \lt  \lambda$，$f(\alpha) \ge \xi$。

**共尾性（cofinality）** 定义为

$$
\mathrm{cf}(\kappa) = \min\{\lambda : \text{存在无界映射 } f: \lambda \to \kappa\}
$$

直觉：**一个基数能不能被更小的序数「追赶」到顶**。若 $\mathrm{cf}(\kappa) = \kappa$，称 $\kappa$ 为**正则（regular）**；若 $\mathrm{cf}(\kappa) \lt  \kappa$，称 $\kappa$ 为**奇异（singular）**。

几个基本事实：

- $\mathrm{cf}(\aleph_0) = \aleph_0$，$\aleph_0$ 正则。
- $\mathrm{cf}(\aleph_{\omega}) = \omega \lt  \aleph_\omega$，因为映射 $n \mapsto \aleph_n$ 在 $\aleph_\omega = \sup_n \aleph_n$ 中无界——$\aleph_\omega$ **奇异**。
- 对后继基数 $\aleph_{\alpha+1}$，恒有 $\mathrm{cf}(\aleph_{\alpha+1}) = \aleph_{\alpha+1}$（见 König 定理），**后继基数都正则**。<span class="marginnote">「极限基数里既有正则也有奇异」是基数世界的分水岭。不可达基数（inaccessible cardinal）就是「正则 + 极限基数 + 足够大」的极限基数——它比所有奇异基数更接近「算不动的顶端」，其存在性在 ZFC 中不可证（大基数公理，见 Jech 第12章）。</span>

**辨析｜易错点：** 共尾性不是「子集大小」，而是「追赶能力」。$\aleph_\omega$ 有无穷多个 $\aleph_n$ 垫在下面，所以只需 $\omega$ 步就能「逼近」它——尽管 $\aleph_\omega$ 本身不可数地大。「少但快」与「多但慢」是共尾性唯一关心的区分。

## 4 公式解析：Hartogs 数就是「装不进 $X$ 的最小良序型」

把 Hartogs 定理的核心写成一个链条：

$$
\aleph(X) = \text{所有} \; \{\mathrm{otp}(A, \prec) : A \subseteq X, \; \prec \text{ 良序 } A\}
$$

- **$\mathrm{otp}(A,\prec)$**：良序集 $(A,\prec)$ 的序型（上一篇的良序表示定理给出的唯一序数）。
- **$\{ \cdots : A \subseteq X \}$**：把 $X$ 的每个「可良序化的子集」都编成序数。这里需要替换公理：从一个集合（$\mathcal{P}(X)$）里的对象出发，把它们一一映射成序数，得到的仍是集合，不会变成真类。
- **取并 / 最小**：这些序数聚集成的集合 $W$ 本身是序数，取 $W$ 即 $\aleph(X)$。若 $W$ 能单射进 $X$，$W$ 就会自己成为这些序型之一，自相矛盾。

于是 $\aleph(X)$ 有两个身份：**它是最小的「塞不进 $X$ 的基数」**，也是 **「$X$ 的子集们能达到的全部良序型的上界」**。对 $X = \omega$：$\omega$ 的一切可数子集的良序型涵盖了所有可数序数，它们的上界正是 $\aleph_1$。

**辨析｜易错点：** Hartogs 定理**没有**断言 $X$ 本身能被良序化——那需要选择公理（第2篇）。它只说：无论如何，存在一个「比 $X$ 大」的基数。这提醒我们基数的基本算术在 ZF（无 AC）中依然运转良好，AC 影响的是「比较任意两个集合的大小」这件事，而不是「是否存在更大的基数」。

## 6 动手推导：$\aleph_1 = \aleph(\omega)$ 的一步步验算

把 Hartogs 构造套在 $X = \omega$ 上，看它为什么恰好给出 $\aleph_1$。

- **第一步，列出 $\omega$ 的可良序子集**：$\omega$ 的子集有可数多个（每子集是自然数集），它们能被良序化吗？——**能**，因为 $\omega$ 本身按自然序良序，任何子集继承良序。于是所有 $(A, \prec)$（$A \subseteq \omega$）的序型都可取。
- **第二步，这些序型涵盖哪些序数**：$A$ 可数，故其序型是可数序数——即一切 $\lt  \aleph_1$ 的序数都能作为某个 $A$ 的序型出现（用自然数集编码可数良序，是一个经典的可数构造）。
- **第三步，取并得 $\aleph(\omega)$**：全体可数序型的上界 = 最小的不可数序数 = $\aleph_1$。这就是 $\aleph(\omega) = \aleph_1$。
- **第四步，关键观察**：这个构造**没用到选择公理**——我们只是「收集序型」再取并，替换公理保证收集合法。所以「$\aleph_1$ 存在」在 ZF 里就成立，无需 AC。

**辨析｜易错点：** $\aleph(\omega) = \aleph_1$ 依赖「$\omega$ 的每个可数良序都能编码进自然数」——这本身是一个非平凡的构造（用配对函数把有限序列编码）。初学者常以为「可数序型的全体是可数的」，从而误推「$\aleph_1$ 可数」——不，可数序型的**集合**不可数（每个可数序数都是某良序的序型，而可数序数有 $\aleph_1$ 个）。

### 更进一步：$\beth$ 层级与基数的幂

Hartogs 数给了「下一个更大的基数」，但基数算术里更常用的「按幂生长」的序列是 **$\beth$（beth）层级**：

$$
\beth_0 = \aleph_0, \qquad \beth_{\alpha+1} = 2^{\beth_\alpha}, \qquad \beth_\lambda = \sup_{\alpha\lt \lambda} \beth_\alpha \;(\lambda \text{ 极限})
$$

$\beth_1 = 2^{\aleph_0}$ 就是连续统。**广义连续统假设（GCH）** 断言 $\aleph_\alpha = \beth_\alpha$ 对所有 $\alpha$ 成立——它把「$\aleph$ 序列」与「$\beth$ 序列」压成一条线。GCH 与 AC 都独立于 ZF（Gödel 在 $L$ 里证明 GCH 一致，Cohen 证明其否定一致），而 GCH 蕴含 AC（Sierpiński）——**幂的「规律性」本身就逼出选择**，这是基数算术里最令人惊讶的事实之一。

Hartogs 数在这种视角下有了新读法：$\aleph(X)$ 是「比 $X$ 大的最小基数」，而 $\beth$ 序列是「按幂不断膨胀的基数」——前者测量「$X$ 缺多少」，后者测量「$X$ 能长多大」。

## 8 小结

- **基数**是最小的代表序数；$|\cdot|$ 取等势类的最小序数；$\aleph_0 = \omega$，后继为 $\aleph_{\alpha+1}$。
- **Hartogs 数 $\aleph(X)$**：塞不进 $X$ 的最小基数，用替换公理构造，**不依赖选择公理**；$\aleph_1 = \aleph(\omega)$。
- **共尾性 $\mathrm{cf}(\kappa)$**：追赶 $\kappa$ 所需的最短长度；正则 = 自己追自己，奇异 = 被更小的长度追上。
- $\aleph_0$ 与所有后继基数正则；$\aleph_\omega$ 奇异（$\mathrm{cf}(\aleph_\omega)=\omega$）。
- 基数的 von Neumann 实现让「多少个」也落进集合论语言，是 König 定理与正则/奇异分类的出发点。

在下一节，我们将用共尾性这个放大镜看基数的**加法与幂**：为什么 $\aleph_\alpha + \aleph_\alpha = \aleph_\alpha$ 而无害，König 定理却逼着 $2^{\aleph_0}$