---
title: 伽罗瓦对应：子群与子域
date: 2026-08-07
---

# 伽罗瓦对应：子群与子域

<div class="epigraph">
<p>分裂域的中间子域，与 Galois 群的子群一一对应——对称的层级就是结构的层级。</p>
<footer>—— 自 题（伽罗瓦对应笔记）</footer>
</div>

<div class="article-byline">
<p>第二级 · 抽象代数 ｜ 杨子胥《近世代数》§11.2 ｜ 2026-08-07</p>
</div>

## 为什么从伽罗瓦对应开始

上一节我们有了 Galois 群 $\operatorname{Gal}(E/F)$。这一节建立 Galois 理论的核心：**伽罗瓦对应（Galois correspondence）**——扩张 $E/F$ 的**中间子域**与 Galois 群的**子群**之间存在一个反序的双射：

$$
\left\{ \text{$E/F$ 的中间子域 } K \right\} \ \longleftrightarrow\ \left\{ \operatorname{Gal}(E/F) \text{ 的子群 } H \right\}
$$

方向是「反序」的：$K_1 \subseteq K_2$ 对应 $H_1 \supseteq H_2$（域越大，固定它的对称越少）。这个对应让「域的层级」与「对称的层级」完全同步——**研究分裂域的内部结构 = 研究 Galois 群的子群**。伽罗瓦对应是理解「方程可解性」（下下节）与「尺规作图」（第六节）的全部杠杆。

## 1 两个方向的映射

设 $E/F$ 是有限 Galois 扩张，$G = \operatorname{Gal}(E/F)$。

**方向一（域 → 群）**：对中间子域 $K$（$F \subseteq K \subseteq E$），定义

$$
\operatorname{Gal}(E/K) = \{ \sigma \in G \mid \sigma \text{ 固定 } K \}
$$

即「$K$ 的对称子群」（保持 $K$ 的自同构）。$K$ 越大，固定它的 $\sigma$ 越少，子群越小。

**方向二（群 → 域）**：对子群 $H \le G$，定义

$$
E^H = \{ x \in E \mid \sigma(x) = x \ \forall \sigma \in H \}
$$

即「$H$ 的不动域（fixed field）」（被 $H$ 全体固定的元素）。$H$ 越大，不动域越小。<span class="marginnote">两个方向互为「镜像」：$\operatorname{Gal}(E/K)$ 问「哪些对称不碰 $K$」，$E^H$ 问「$H$ 的对称把哪些元素钉死」。直觉对照：对称越多的子结构越「对称中心」，对应越小的子域/子群。<strong>域与群在「层级」上反向同步</strong>——这就是「反序对应」的含义。</span>

**例：** $E = \mathbb{Q}(\sqrt2, \sqrt3)$，$G = \operatorname{Gal}(E/\mathbb{Q}) \cong \mathbb{Z}_2 \times \mathbb{Z}_2$（元素：$\mathrm{id}$、$\sigma_1$（$\sqrt2\mapsto-\sqrt2$）、$\sigma_2$（$\sqrt3\mapsto-\sqrt3$）、$\sigma_1\sigma_2$）。

$E^{\langle \sigma_1 \rangle} = \mathbb{Q}(\sqrt3)$（$\sigma_1$ 只动 $\sqrt2$，不动 $\sqrt3$）；
$E^{\langle \sigma_2 \rangle} = \mathbb{Q}(\sqrt2)$；
$E^{\langle \sigma_1\sigma_2 \rangle} = \mathbb{Q}(\sqrt6)$（$\sigma_1\sigma_2$ 同时翻转两个根，$\sqrt6 = \sqrt2\sqrt3$ 不动）；
$E^{\langle \mathrm{id}\rangle} = E$，$E^{G} = \mathbb{Q}$。

四个中间子域 $\mathbb{Q}, \mathbb{Q}(\sqrt2), \mathbb{Q}(\sqrt3), \mathbb{Q}(\sqrt6), E$ ↔ 五个子群——**子域格与子群格完全同构**。

## 2 伽罗瓦对应的基本性质

**定理（伽罗瓦对应的基本性质）：** 设 $E/F$ 有限 Galois，$G = \operatorname{Gal}(E/F)$。

1. **不动域回到基域**：$E^G = F$（被全体对称固定的是基域）；
2. **子群回到不动域**：$\operatorname{Gal}(E/E^H) = H$（不动域的子群还是 $H$）;
3. **对应是双射**：$K \mapsto \operatorname{Gal}(E/K)$ 与 $H \mapsto E^H$ 互逆，建立中间子域 ↔ 子群的双射；
4. **反序**：$K_1 \subseteq K_2 \iff \operatorname{Gal}(E/K_2) \subseteq \operatorname{Gal}(E/K_1)$。

**证明（性质 1 的要点）：** $F \subseteq E^G$ 显然。反方向用「$E^G$ 不含超越 $F$ 的元素」的 Artin 引理：若 $x \in E^G$ 且 $x \notin F$，则 $F(x)$ 是 $E$ 中比 $F$ 大的子域，$[F(x):F] \ge 2$，而「$x$ 被 $G$ 全体固定」会迫使 $[F(x):F] = 1$（用「自同构个数 ≤ 扩张次数」的线性无关论证），矛盾。$\blacksquare$<span class="marginnote">性质 1（$E^G = F$）是「对称的反面是平凡」：如果一个元素被 Galois 群<strong>所有</strong>对称固定，它一定落在基域里。这直觉上自然（基域是「完全对称中心」），证明需要 Artin 引理（$E^H$ 的扩张次数被 $H$ 大小控制）。性质 3（双射）是全部对应理论的地基。</span>

## 3 对应定理：子群 ↔ 子域的完整版本

**定理（伽罗瓦对应 / Galois Correspondence）：** 设 $E/F$ 有限 Galois，$G = \operatorname{Gal}(E/F)$。映射 $K \mapsto \operatorname{Gal}(E/K)$ 与 $H \mapsto E^H$ 互逆，且：

1. $[E : K] = |\operatorname{Gal}(E/K)|$，$[K : F] = [G : \operatorname{Gal}(E/K)]$（大小对应）；
2. **$K/F$ 正规 ⟺ $\operatorname{Gal}(E/K) \trianglelefteq G$**，此时 $\operatorname{Gal}(K/F) \cong G / \operatorname{Gal}(E/K)$（正规对应商）；
3. $E^H/E^H$ 的扩张是 Galois 的。

**证明（大小对应）：** $E/K$ 是 Galois 扩张（子扩张仍 Galois），由上一节 $|\operatorname{Gal}(E/K)| = [E:K]$；乘法塔 $[K:F] = [E:F]/[E:K] = |G|/|\operatorname{Gal}(E/K)| = [G : \operatorname{Gal}(E/K)]$。$\blacksquare$<span class="marginnote">对应定理的「大小对应」让子域格与子群格不仅同构还「保度量」：域的扩张次数 = 群的指标。而「正规子群 ↔ 正规子扩张」（性质 2）是伽罗瓦理论最深刻的洞察——<strong>「$K/F$ 是分裂域（正规）」恰好对应「$\operatorname{Gal}(E/K)$ 是正规子群」</strong>，且外层对称 $\operatorname{Gal}(K/F)$ 是商群 $G/\operatorname{Gal}(E/K)$。这根链条直接通向「可解群 = 方程可根式求解」。</span>

## 4 公式解析：E^G = F 与 Artin 引理

把「不动域回到基域」的证明核心拆透，它是对应定理成立的支点。

**第一步，问题。** 要证 $E^G = F$：被 Galois 群全体固定 ⟹ 属于基域。

**第二步，Artin 引理。** 一般地：对有限群 $H \le \operatorname{Aut}(E)$，$[E : E^H] \le |H|$。证明用「线性无关的自同构」：若 $x_1, \dots, x_n \in E$ 在 $E^H$ 上线性无关，则 $n \le |H|$（构造 Vandermonde 型矩阵，用 $H$ 元素作用出线性方程组，反证矛盾）。

**第三步，代入 $H = G$。** $[E : E^G] \le |G| = [E:F]$（Galois 扩张大小恒等式），而 $E^G \supseteq F$ 给 $[E:E^G] \le [E:F]$。两者合起来 $[E : E^G] = [E : F]$（Artin 给出 ≤，包含给 ≤……实际是 Artin 给 $[E:E^G] \le |G| = [E:F]$，且 $F \subseteq E^G$ 给 $[E:F] = [E:E^G][E^G:F] \ge [E:E^G]$，故相等且 $[E^G:F] = 1$），$E^G = F$。$\blacksquare$

**第四步，意义。** Artin 引理把「群大小」与「扩张次数」连起来：$H$ 越大，不动域越小（$[E:E^H]$ 越大）。它是「$E^H$ 有多大」的精确回答，也是对应定理的证明心脏。

## 5 例子：x^3 - 2 的完整对应

把伽罗瓦对应在 $E = \mathbb{Q}(\sqrt[3]2, \omega)$、$G = S_3$ 上完整画出，这是最经典的对应图。

$G = S_3$ 的子群：$\{e\}$、$A_3 = \langle (123) \rangle$（3 阶）、三个 $\langle (12) \rangle$ 型（2 阶）、$S_3$。
**不动域**：
$E^{\{e\}} = E$；
$E^{A_3} = \mathbb{Q}(\omega)$（$A_3$ 由三循环生成，保持 $\omega$ 不动；$\sqrt[3]2$ 被三循环搬动）；
$E^{\langle (12) \rangle} = \mathbb{Q}(\sqrt[3]2)$（固定 $A_3$ 的 2 阶子群对应固定 $\sqrt[3]2$ 的域）；
$E^{S_3} = \mathbb{Q}$。
**对应验证**：$\mathbb{Q} \subseteq \mathbb{Q}(\sqrt[3]2) \subseteq E$ 对应 $S_3 \supseteq \langle (12)\rangle \supseteq \{e\}$（反序）。
**正规子群**：$A_3 \trianglelefteq S_3$ 对应 $\mathbb{Q}(\omega)/\mathbb{Q}$ 正规（$x^2+x+1$ 分裂域）；$\langle (12)\rangle$ 不正规对应 $\mathbb{Q}(\sqrt[3]2)/\mathbb{Q}$ 不正规（缺另两根）。$\checkmark$<span class="marginnote">$x^3-2$ 的对应图是 Galois 理论的「hello world」：$S_3$ 的四个子群 ↔ 四个中间子域，反序对应，正规子群 ↔ 正规扩张。尤其「$\langle (12)\rangle \leftrightarrow \mathbb{Q}(\sqrt[3]2)$」与「$\langle (12)\rangle$ 不正规 ↔ $\mathbb{Q}(\sqrt[3]2)$ 不正规」——正规性在两边同步失败，正是对应定理性质 2 的现场演示。</span>

## 6 小结

- **两个映射**：$K \mapsto \operatorname{Gal}(E/K)$（对称子群）、$H \mapsto E^H$（不动域）。
- **伽罗瓦对应**：中间子域 ↔ 子群的双射，反序；$E^G = F$。
- **大小对应**：$[E:K] = |\operatorname{Gal}(E/K)|$、$[K:F] = [G:\operatorname{Gal}(E/K)]$。
- **正规性对应**：$K/F$ 正规 ⟺ $\operatorname{Gal}(E/K) \trianglelefteq G$，且 $\operatorname{Gal}(K/F) \cong G/\operatorname{Gal}(E/K)$。
- **例子**：$x^3-2$ 的 $S_3$ 子群 ↔ 四个中间子域；$x^4-4x^2+2$ 的 $\mathbb{Z}_2\times\mathbb{Z}_2$ 对应四子域。

在下一节，我们把对应定理收拢成总纲：**Galois 理论基本定理**。子群与子域的双射、正规性与可解性的桥梁，将在此完整陈述。
