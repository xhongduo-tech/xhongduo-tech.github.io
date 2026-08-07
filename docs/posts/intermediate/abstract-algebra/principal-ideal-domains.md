---
title: 主理想整环（PID）
date: 2026-08-07
---

# 主理想整环（PID）

<div class="epigraph">
<p>每个理想都单生成的整环，是最「线性」的整环——所有结构问题都变成单生成元的算术。</p>
<footer>—— 自 题（PID 笔记）</footer>
</div>

<div class="article-byline">
<p>第二级 · 抽象代数 ｜ 杨子胥《近世代数》§9.6 ｜ 2026-08-07</p>
</div>

## 为什么从主理想整环开始

第八篇我们见过「$\mathbb{Z}$ 与 $F[x]$ 的理想全是主理想」。把这条性质立为定义，得到**主理想整环（principal ideal domain, PID）**：每个理想都由单个元素生成的整环。PID 处在「欧氏环 ⊆ PID ⊆ UFD」黄金链的中段——它比欧氏环宽（不需带余除法），比 UFD 窄（要求理想单生成）。

PID 是整环分解理论的甜点：在 PID 里，gcd 不仅能算，还能**线性表出**（裴蜀定理的推广）、不可约 ⟺ 素、每个非零素理想都是极大。理解 PID，就理解了「为什么 $\mathbb{Z}$ 与 $F[x]$ 这么像」——它们都是 PID，共享一整座分解理论大厦。本节把 PID 的定义、关键性质与「PID 是 UFD」的证明讲透。

## 1 PID 的定义与例子

**主理想整环（PID）**：设 $D$ 是整环。若 $D$ 的每个理想都是主理想（由单个元素生成），则称 $D$ 是**PID**。

**例：**
- $\mathbb{Z}$ 是 PID（第八篇已证：$\mathbb{Z}$ 的理想全为 $n\mathbb{Z}$）；
- $F[x]$（$F$ 域）是 PID（带余除法证明，第九篇《带余除法》）；
- **一切域是 PID**（只有 $\langle 0 \rangle$ 与 $\langle 1\rangle$ 两个理想）；
- **$\mathbb{Z}[x]$ 不是 PID**（$\langle 2, x \rangle$ 不是主理想）；
- **$\mathbb{Z}[\sqrt{-5}]$ 不是 PID**（甚至不是 UFD）。<span class="marginnote">PID 的直觉：「每个理想 $I$ 都能被一个元素撑起来」。$\mathbb{Z}$ 里撑起 $n\mathbb{Z}$ 的是 $n$（最小正元素）；$F[x]$ 里撑起理想的是「最小次数多项式」。PID 让理想运算（和、交、积）全部降级成「单生成元的 gcd/lcm 运算」——这是 PID 一切顺滑的根源。</span>

## 2 PID 的核心性质

PID 的一连串性质，让它成为「理想世界最顺滑的整环」。

**定理（PID 的素理想结构）：** 设 $D$ 是 PID，则

1. 每个非零素理想是极大理想；
2. **不可约元 ⟺ 素元**（UFD 的关键性质成立）；
3. 每个非零元素可分解成不可约元之积（分解存在）。

**性质 1 证明：** 设 $\langle p \rangle$ 非零素理想，且 $\langle p \rangle \subsetneq \langle a \rangle$。则 $p \in \langle a \rangle$，$p = ab$。由 $p$ 素（素理想生成元是素元），$p \mid a$ 或 $p \mid b$。若 $p \mid a$，则 $\langle a \rangle = \langle p \rangle$，矛盾；故 $p \mid b$，$b = pc$，$p = apc$，消去 $p$ 得 $ac = 1$，$a$ 是单位，$\langle a \rangle = D$。故 $\langle p \rangle$ 之上无真中间理想，$\langle p \rangle$ 极大。$\blacksquare$<span class="marginnote">「PID 中非零素理想极大」的证明只用两招：$p = ab$ 的分解 + 素性二分 + 消去律。它把「素 ⟺ 极大」从 PID 推到全域。这条性质在 $\mathbb{Z}$ 上的化身：$\langle p\rangle$（素数）既素又极大；在 $F[x]$ 上的化身：$\langle f\rangle$（不可约）既素又极大。<strong>PID 里「素 / 不可约 / 极大」三者合一。</strong></span>

**性质 2 证明（不可约 ⟺ 素）：** 素 ⟹ 不可约恒成立。不可约 ⟹ 素：设 $p$ 不可约，$\langle p \rangle$ 是极大（由性质 1 的证明技巧可证「不可约生成元 ⟹ 极大」），极大是素，故 $p$ 素。$\blacksquare$——**PID 满足 UFD 的等价刻画**。

## 3 PID 是 UFD：完整证明

把「PID ⟹ UFD」的证明走完整，它是分解理论的核心推导。

**定理：** 每个 PID 是 UFD。

**证明分两步。**

**第一步（分解存在）：** 假设 $a$ 不能分解成不可约元之积，则 $a$ 可约：$a = a_1 b_1$（都非单位）。至少一个（比如 $a_1$）也不能分解；$a_1 = a_2 b_2$，……得到无限链 $\langle a \rangle \subsetneq \langle a_1 \rangle \subsetneq \langle a_2 \rangle \subsetneq \cdots$（每个都真包含，因为 $a_i$ 真除 $a_{i+1}$）。令 $I = \bigcup_i \langle a_i \rangle$，$I$ 是理想（理想并的链是理想），PID 性给 $I = \langle c \rangle$。则 $c \in \langle a_k \rangle$ 对某 $k$（因为 $c$ 在并里），于是 $I \subseteq \langle a_k \rangle$；但 $\langle a_k \rangle \subsetneq I$，矛盾。故分解存在。$\blacksquare$<span class="marginnote">分解存在的证明用「理想并链 + PID 单生成」制造矛盾：无限可约会产生严格递增的理想链，而链的并是理想、被单生成元捕获，链条被迫终止。这套「Noether 条件」论证是交换代数的标准武器，PID 只是它最简单的舞台。</span>

**第二步（唯一性）：** 由性质 2（不可约 ⟺ 素），UFD 等价刻画的条件②成立，套上节定理即得唯一性。$\blacksquare$

**推论：** $\mathbb{Z}$、$F[x]$、一切域都是 UFD——算术基本定理与多项式唯一分解定理统一进 PID 理论。

## 4 公式解析：gcd 的线性表出（裴蜀定理的推广）

PID 里 gcd 不仅能算，还能「线性表出」——这是 PID 胜过一般 UFD 的独家能力。

**定理（PID 中 gcd 线性表出）：** 设 $D$ 是 PID，$a, b \in D$。则存在 $s, t \in D$ 使

$$
\gcd(a, b) = sa + tb
$$

**证明：** 考虑理想 $\langle a, b \rangle = \{ ra + sb \}$。PID 性给 $\langle a, b \rangle = \langle d \rangle$。则 $d = sa + tb$（$d \in \langle a, b\rangle$）；且 $a, b \in \langle d \rangle$，故 $d \mid a$、$d \mid b$。若 $c \mid a$ 且 $c \mid b$，则 $c \mid sa + tb = d$。故 $d$ 是 $a, b$ 的最大公因子，且 $d = sa + tb$。$\blacksquare$

- **第一步，理想即 gcd。** $\langle a, b \rangle = \langle \gcd(a,b) \rangle$（第八篇的理想和定理，在 $\mathbb{Z}$ 上是 $\langle a\rangle + \langle b\rangle = \langle \gcd\rangle$）。
- **第二步，线性组合的「组合性」。** $d = sa + tb$ 是「$a, b$ 的线性组合」——这正是裴蜀定理在 $\mathbb{Z}$ 里的形态：$\gcd(a,b) = sa + tb$。
- **第三步，为什么 UFD 不够。** 在一般 UFD（如 $\mathbb{Z}[x]$）里，$\gcd(2, x) = 1$ 存在，但 $1$ **不能**写成 $2s + xt$（代入 $x = 0$ 得 $1 = 2s(0)$，$s(0)$ 须 $\frac12$，非整数系数）——**UFD 有 gcd 但不保证线性表出，PID 两者兼得**。
- **第四步，应用。** 线性表出是「裴蜀型」结论的源头：中国剩余定理的互素条件（$\langle a\rangle + \langle b\rangle = \langle 1\rangle$）、方程 $ax + by = c$ 的可解判定（$c$ 需被 gcd 整除）——全部在 PID 上成立。

## 5 例子：PID 与非 PID 的边界

用「理想是否单生成」这把尺子量量各种整环。

| 整环 | PID？ | 为什么 |
| --- | --- | --- |
| 域 $F$ | ✓ | 只有两个平凡理想 |
| $\mathbb{Z}$ | ✓ | 理想 $= n\mathbb{Z}$ |
| $F[x]$ | ✓ | 带余除法 |
| $\mathbb{Z}[i]$ | ✓ | 欧氏（范数做除法） |
| $\mathbb{Z}[x]$ | ✗ | $\langle 2, x \rangle$ 非主理想 |
| $F[x, y]$ | ✗ | $\langle x, y \rangle$ 非主理想 |
| $\mathbb{Z}[\sqrt{-5}]$ | ✗ | 非 UFD（更非 PID） |

**观察**：PID 的边界在「单变量」与「多变量/带系数限制」之间。单变量多项式（系数域上）是 PID，多变量不是；系数域（可除）时是 PID，系数整环（不可除）时不是。**「域上单变量多项式环」是 PID 的标准模特**，而 $\mathbb{Z}[x]$ 是「差一步」的反例。<span class="marginnote">$\mathbb{Z}[x]$ 不是 PID 但仍是 UFD（高斯引理，下一篇），所以「PID ⊆ UFD」是严格包含。$F[x, y]$ 是 UFD 但非 PID（高斯引理可推广到多变量）。PID 这条链：「欧氏环 ⊆ PID ⊆ UFD」，每一层都严格——$\mathbb{Z}$ 与 $F[x]$ 在欧氏层，$\mathbb{Z}[\tfrac{1+\sqrt{-19}}{2}]$ 是「PID 但非欧氏」的著名例子（严格性的见证）。</span>

## 6 小结

- **PID**：每个理想都单生成的整环；$\mathbb{Z}$、$F[x]$、域是 PID，$\mathbb{Z}[x]$ 不是。
- **核心性质**：非零素理想是极大；不可约 ⟺ 素；分解存在（Noether 论证）。
- **PID ⟹ UFD**：素性刻画 + 分解存在的理想链论证。
- **gcd 线性表出**：$d = sa + tb$（裴蜀定理推广）；UFD 保证 gcd 存在但不保证线性表出。
- **包含链**：欧氏环 ⊆ PID ⊆ UFD，均严格。

在下一节，我们走到链条的最左端：**欧氏整环（Euclidean Domain）**。带余除法的整环是「最可计算」的一族，也是 $\mathbb{Z}$ 与 $F[x]$ 真正的共同家园。
