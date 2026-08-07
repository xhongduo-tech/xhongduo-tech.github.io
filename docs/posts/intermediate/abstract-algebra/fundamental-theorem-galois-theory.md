---
title: Galois 理论基本定理
date: 2026-08-07
---

# Galois 理论基本定理

<div class="epigraph">
<p>分裂域的子域格与 Galois 群的子群格同构——这是 19 世纪数学最深刻的一行对应。</p>
<footer>—— 自 题（Galois 理论笔记）</footer>
</div>

<div class="article-byline">
<p>第二级 · 抽象代数 ｜ 杨子胥《近世代数》§11.3 ｜ 2026-08-07</p>
</div>

## 为什么从 Galois 理论基本定理开始

前两节我们建立了伽罗瓦对应的两个方向，这一节把它们收拢成一条总纲——**Galois 理论基本定理（Fundamental Theorem of Galois Theory）**：对有限 Galois 扩张 $E/F$，中间子域与 Galois 群子群之间的反序双射，连同「正规子群 ↔ 正规子扩张」与「商群 ↔ 商扩张」，构成一套完备的字典。**域论问题被整本地翻译成群论问题**。

这条定理是抽象代数数百年的高峰：方程可解性（$G$ 可解）、尺规作图（$G$ 为 2-群）、代数数论（子域与子群的对应）全部从这张字典里读出来。本节把基本定理的完整陈述、证明要点与「字典用法」讲透。

## 1 基本定理的完整陈述

**定理（Galois 理论基本定理）：** 设 $E/F$ 是有限 Galois 扩张，$G = \operatorname{Gal}(E/F)$。则

1. **对应**：映射 $K \mapsto \operatorname{Gal}(E/K)$ 与 $H \mapsto E^H$ 互逆，建立「中间子域」与「子群」的反序双射；
2. **大小**：$[E : K] = |\operatorname{Gal}(E/K)|$，$[K : F] = [G : \operatorname{Gal}(E/K)]$；
3. **正规 ↔ 正规**：$K/F$ 是 Galois 扩张 ⟺ $\operatorname{Gal}(E/K) \trianglelefteq G$，此时
$$
\operatorname{Gal}(K/F) \cong G / \operatorname{Gal}(E/K)
$$
4. **交换性**：若 $K$ 对应 $H$，则 $K$ 是 $F$ 上某个可分多项式的分裂域 ⟺ $H \trianglelefteq G$。

**字典（翻译手册）：**

| 域论（$E/F$ 的中间子域） | 群论（$G$ 的子群） |
| --- | --- |
| $K$（中间子域） | $\operatorname{Gal}(E/K)$（子群） |
| $F \subseteq K$ | $G \supseteq \operatorname{Gal}(E/K)$（反序） |
| $[K : F]$ | $[G : \operatorname{Gal}(E/K)]$（指标） |
| $K/F$ 正规（Galois） | $\operatorname{Gal}(E/K) \trianglelefteq G$（正规） |
| $\operatorname{Gal}(K/F)$ | $G/\operatorname{Gal}(E/K)$（商群） |

<span class="marginnote">基本定理的「字典」是使用 Galois 理论的日常工具：遇到域论问题，查表翻成群论问题，解完再翻回来。伽罗瓦的伟大在于这字典是<strong>完备</strong>的——域的每一层结构都有群论的精确镜像。「研究分裂域的中间域」变成「研究有限群的子群」，后者有成熟工具（西罗定理、可解群理论）。</span>

## 2 证明要点：为什么对应是双射

基本定理的证明依赖两个引理，它们的叠加给出「对应是双射」。

**引理 A（Artin）：** 对有限群 $H \le \operatorname{Aut}(E)$，$[E : E^H] \le |H|$，且 $E/E^H$ 是 Galois 扩张（$H$ 是其 Galois 群的一部分）。

**引理 B（$E/K$ 仍 Galois）：** 若 $E/F$ Galois 且 $F \subseteq K \subseteq E$，则 $E/K$ 也是 Galois 扩张，且 $\operatorname{Gal}(E/K) \le G$。

**证明（$H \mapsto E^H \mapsto \operatorname{Gal}(E/E^H)$）：** 由引理 A，$[E : E^H] \le |H|$；由引理 B 与大小恒等式，$|\operatorname{Gal}(E/E^H)| = [E:E^H]$。而 $H \le \operatorname{Gal}(E/E^H)$（$H$ 的元素都固定 $E^H$）。于是 $|H| \le |\operatorname{Gal}(E/E^H)| = [E:E^H] \le |H|$，处处相等，$H = \operatorname{Gal}(E/E^H)$。**「不动域的子群还是原群」——$H \mapsto E^H$ 是单射。**$\blacksquare$<span class="marginnote">「$H \mapsto E^H$ 单射」的证明是「夹逼」：$|H| \le |\operatorname{Gal}(E/E^H)| = [E:E^H] \le |H|$ 三面夹出相等。类似的「$K \mapsto \operatorname{Gal}(E/K)$ 单射」由 $E^{\operatorname{Gal}(E/K)} = K$（不动域性质）给出。两个单射 + 大小配对 = 双射。</span>

## 3 正规 ↔ 正规的证明

**证明（$K/F$ 正规 ⟹ $\operatorname{Gal}(E/K) \trianglelefteq G$）：** $K/F$ 正规 ⟹ $K$ 是 $F$ 上某多项式 $g$ 的分裂域。对 $\sigma \in G$、$\tau \in \operatorname{Gal}(E/K)$：要证 $\sigma\tau\sigma^{-1} \in \operatorname{Gal}(E/K)$，即 $\sigma\tau\sigma^{-1}$ 固定 $K$。取 $x \in K$，$x$ 是 $g$ 的根（$K$ 是 $g$ 的分裂域），$\sigma^{-1}(x)$ 也是 $g$ 的根，故 $\sigma^{-1}(x) \in K$（$g$ 的根全在 $K$），$\tau$ 固定它，$\sigma\tau\sigma^{-1}(x) = \sigma(\tau(\sigma^{-1}(x))) = \sigma(\sigma^{-1}(x)) = x$。$\blacksquare$

**证明（⟸ 方向）：** 若 $\operatorname{Gal}(E/K) \trianglelefteq G$，取 $K = E^H$（$H$ 正规）。对 $x \in K$、$\sigma \in G$，$\sigma(x)$ 的「$H$-稳定性」由正规性保证（$\sigma H \sigma^{-1} = H$ 推出 $\sigma(x)$ 被 $H$ 固定，$\sigma(x) \in E^H = K$）。于是 $G$ 把 $K$ 映到 $K$，$K/F$ 是正规扩张（$K$ 中不可约多项式的根都被 $G$ 留在 $K$）。$\blacksquare$<span class="marginnote">「正规 ↔ 正规」的证明是伽罗瓦理论的「灵魂时刻」：正规性（分裂域性质）与正规子群（共轭不变）在对应下完美同构。$\sigma\tau\sigma^{-1}$ 的共轭动作对应「把域里的根搬到另一个根」——<strong>正规子群 = 「无论怎么搬都搬回自己的子域」</strong>。</span>

## 4 公式解析：Gal(K/F) ≅ G / Gal(E/K)

把「商群 ↔ 商扩张」这条最深刻的字典项拆透。

- **第一步，问题的几何。** $K/F$ 正规，$K$ 的对称 $\operatorname{Gal}(K/F)$ 与 $E$ 的对称 $G$ 什么关系？——$G$ 的对称「限制到 $K$」给出 $\operatorname{Gal}(K/F)$，而「限制」这个动作的核是「固定 $K$ 的对称」$\operatorname{Gal}(E/K)$。

- **第二步，限制映射。** $\rho : G \to \operatorname{Gal}(K/F)$，$\rho(\sigma) = \sigma|_K$（限制到 $K$）。$K/F$ 正规保证 $\sigma(K) = K$（$G$ 把正规扩张映到自身），故 $\sigma|_K$ 是 $K$ 的自同构。

- **第三步，算核与像。** $\ker \rho = \{ \sigma \mid \sigma|_K = \mathrm{id}_K \} = \operatorname{Gal}(E/K)$（固定 $K$ 的对称）。像 = 全体 $\sigma|_K$ = $\operatorname{Gal}(K/F)$（$K/F$ 是 Galois 扩张，大小 $[K:F]$，而 $|G|/|\operatorname{Gal}(E/K)| = [K:F]$ 由大小对应，故满射）。

- **第四步，第一同构定理。** $G/\ker\rho \cong \operatorname{Im}\rho$，即 $G/\operatorname{Gal}(E/K) \cong \operatorname{Gal}(K/F)$。$\blacksquare$ **外层对称 = 内层对称压掉内层固定**——商群的语言完美描述「剥掉内层后的剩余对称」。

## 5 例子：用字典读 x^3 - 2 与 x^4 - 4x^2 + 2

把基本定理的「字典」用在两个例子上，展示翻译的威力。

**$x^3 - 2$（$G = S_3$）：**
- 中间域 $\mathbb{Q}(\omega)$（$x^2+x+1$ 的分裂域，正规）对应子群 $A_3 \trianglelefteq S_3$；
- $\operatorname{Gal}(\mathbb{Q}(\omega)/\mathbb{Q}) \cong S_3/A_3 \cong \mathbb{Z}_2$——「$\mathbb{Q}(\omega)$ 的对称是 $\mathbb{Z}_2$（共轭 $\omega \leftrightarrow \omega^2$）」。
- $\mathbb{Q}(\sqrt[3]2)$ 对应 $\langle (12) \rangle$（不正规），因为 $\mathbb{Q}(\sqrt[3]2)/\mathbb{Q}$ 不正规。$\checkmark$

**$x^4 - 4x^2 + 2$（$G = \mathbb{Z}_2 \times \mathbb{Z}_2$）：**
- 三个 2 阶子群对应三个中间域 $\mathbb{Q}(\sqrt2)$、$\mathbb{Q}(\sqrt{2+\sqrt2})$、$\mathbb{Q}(\sqrt{2-\sqrt2})$（各自是分裂域）；
- 所有子群都正规（$G$ 交换），所有中间域都是 $\mathbb{Q}$ 的正规扩张——**交换 Galois 群 ⟹ 一切中间子域正规**。<span class="marginnote">「$G$ 交换 ⟹ 所有中间子域正规」是基本定理的直接推论：交换群里所有子群正规，对应所有中间子域正规。$x^4 - 4x^2 + 2$ 的 Galois 群 $\mathbb{Z}_2\times\mathbb{Z}_2$ 交换，所以它的每个中间子域都是 $\mathbb{Q}$ 的分裂域——这与「根由两个嵌套的平方根构造」（$\sqrt{2\pm\sqrt2}$）完美吻合。</span>

## 6 小结

- **基本定理**：中间子域 ↔ 子群反序双射；$[K:F] = [G:\operatorname{Gal}(E/K)]$；正规 ↔ 正规；$\operatorname{Gal}(K/F) \cong G/\operatorname{Gal}(E/K)$。
- **字典**：域 ↔ 群、正规 ↔ 正规子群、商域 ↔ 商群。
- **证明**：Artin 引理 + 大小夹逼 = 双射；共轭稳定性 = 正规对应；限制映射 + 第一同构定理 = 商群。
- **用法**：把域论问题翻译成群论问题求解；$x^3-2$ 与 $x^4-4x^2+2$ 是字典的两次现场演练。

在下一节，我们用 Galois 理论回答最古老的问题：**根式扩张与方程的根式可解性**。方程可用根式求解 ⟺ Galois 群可解——伽罗瓦的伟大判决。
