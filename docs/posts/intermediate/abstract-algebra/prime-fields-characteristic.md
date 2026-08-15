---
title: 素域与域的特征
date: 2026-08-07
---

# 素域与域的特征

<div class="epigraph">
<p>每个域都藏着一个最原始的小域——素域，它是域扩张与有限域的第一块砖。</p>
<footer>—— 自 题（素域笔记）</footer>
</div>

<div class="article-byline">
<p>第二级 · 抽象代数 ｜ 杨子胥《近世代数》§10.1 ｜ 2026-08-07</p>
</div>

## 为什么从素域与域的特征开始

第七篇讲过环的特征：$1$ 相加多少次回到 0。对域来说，特征只有两种——0 或素数 $p$。而由此可以抽出每个域内部的「最小子域」：**素域（prime field）**。特征 0 的域含素域 $\mathbb{Q}$，特征 $p$ 的域含素域 $\mathbb{F}_p$。

素域是域理论的地基：它是「域的最小内核」，也是有限域 $\mathbb{F}_{p^n}$ 的一切起点（有限域是素域 $\mathbb{F}_p$ 的 $n$ 维扩张）。同时，「特征」还制造出有限域特有的算术现象（Frobenius 自同态），这些现象在密码学（AES、椭圆曲线）里至关重要。本节把特征的结构定理、素域的构造与「$\mathbb{Q}$ vs $\mathbb{F}_p$」的二分讲透。

## 1 特征的结构定理：0 或素数

**定理（域的特征）：** 设 $F$ 是域，则 $\operatorname{char}(F) = 0$ 或 $\operatorname{char}(F) = p$（$p$ 素数）。

**证明：** 第七篇已证整环的特征是 0 或素数；域是整环，结论成立。更直接地：若特征 $n$ 为合数 $n = ab$（$1 < a, b < n$），则 $0 = n \cdot 1 = (a \cdot 1)(b \cdot 1)$，域无零因子，矛盾。$\blacksquare$<span class="marginnote">「特征只能 0 或素数」是域论的第一个分类结果：域世界从特征上就被切成两半——特征 0 的「稠密世界」（$\mathbb{Q}, \mathbb{R}, \mathbb{C}$）与特征 $p$ 的「周期世界」（$\mathbb{F}_p, \mathbb{F}_{p^n}$）。这条二分将决定第十一篇 Galois 理论里「可分性」是否自动成立。</span>

**例：**
$\operatorname{char}(\mathbb{Q}) = \operatorname{char}(\mathbb{R}) = \operatorname{char}(\mathbb{C}) = 0$；
$\operatorname{char}(\mathbb{F}_p) = \operatorname{char}(\mathbb{F}_{p^n}) = p$；
$\operatorname{char}(\mathbb{F}_2) = 2$，$\operatorname{char}(\mathbb{F}_4) = 2$。

## 2 素域：域的最小内核

**素域（prime field）**：不含真子域的域，称为素域。

**定理（素域的分类）：** 域 $F$ 的素域要么是 $\mathbb{Q}$（当 $\operatorname{char}(F) = 0$），要么是 $\mathbb{F}_p$（当 $\operatorname{char}(F) = p$）。

**证明（特征 $p$ 情形）：** $F$ 含子集 $P = \{ 0, 1, 2\cdot 1, \dots, (p-1)\cdot 1 \}$。$P$ 对加乘封闭（特征 $p$ 的算术：$i\cdot 1 + j \cdot 1 = (i+j)\cdot 1$、$(i\cdot1)(j\cdot1) = ij \cdot 1$），且 $p$ 个元素互不相同（特征 $p$ 是最小的），$P$ 是 $F$ 的 $p$ 阶子域，即 $\mathbb{F}_p$。任何 $F$ 的子域都含 $1$，从而含 $P$，故 $P$ 是 $F$ 的素域。$\blacksquare$<span class="marginnote">素域「由 $1$ 生成」：从 $1$ 出发反复相加得到 $\{0, 1, 2, \dots\}$，特征 0 时这就是 $\mathbb{Z}$，再取「商域」得 $\mathbb{Q}$；特征 $p$ 时加 $p$ 次回到 0，得到 $\mathbb{F}_p$。<strong>素域 = 「$1$ 的加法所及 + 除法闭合」的最小域。</strong></span>

**证明（特征 0 情形）：** 考虑 $F$ 中 $\{ n \cdot 1 \mid n \in \mathbb{Z} \}$（$n \cdot 1$ 各不相同，因为特征 0），它是与 $\mathbb{Z}$ 同构的子环；再取商域（分数）$\{ \frac{n \cdot 1}{m \cdot 1} \}$，得与 $\mathbb{Q}$ 同构的子域。任何子域含 $1$ 故含全体分数，此即素域。$\blacksquare$

**推论：** 任何域都恰含一个素域（同构意义下为 $\mathbb{Q}$ 或 $\mathbb{F}_p$）。**「$1$ 的加乘组合」是每个域的不可约内核。**

## 3 素域与域扩张的关系

素域是域扩张的「底座」：任何域都是其素域的扩张。

**定理（域的层级）：** 设 $F$ 是域，$P$ 是 $F$ 的素域。则 $F$ 是 $P$ 上的扩张：

$\operatorname{char}(F) = 0$：$F$ 是 $\mathbb{Q}$ 的扩张（$\mathbb{Q} \subseteq F$）；
$\operatorname{char}(F) = p$：$F$ 是 $\mathbb{F}_p$ 的扩张（$\mathbb{F}_p \subseteq F$）。

于是「研究域」可以分层：先研究素域（$\mathbb{Q}$ 与 $\mathbb{F}_p$），再研究素域上的扩张（下一篇）。**域扩张理论 = 从素域出发的攀登。**

**例：**
$\mathbb{C}$ 是 $\mathbb{Q}$ 的扩张（$\mathbb{Q} \subset \mathbb{R} \subset \mathbb{C}$）；
$\mathbb{F}_4$ 是 $\mathbb{F}_2$ 的扩张（下一篇构造 $\mathbb{F}_4 = \mathbb{F}_2[x]/\langle x^2+x+1\rangle$）；
有限域 $\mathbb{F}_{p^n}$ 总是 $\mathbb{F}_p$ 的 $n$ 次扩张。<span class="marginnote">「素域 + 扩张」的视角让域论变成「从 $\mathbb{Q}$ 或 $\mathbb{F}_p$ 出发的扩张树」。特征 $p$ 的域全是 $\mathbb{F}_p$ 的扩张，特征 0 的域全是 $\mathbb{Q}$ 的扩张。素域是每棵树的根，域扩张（下一篇）是树干，分裂域与 Galois 群是树冠。</span>

## 4 公式解析：特征 p 里的 Frobenius 自同态

特征 $p$ 的域有一个标志性的「奇算术」，它是有限域理论的心脏。

**定理：** 设 $F$ 是特征 $p$ 的域，则映射 $\sigma : F \to F$，$\sigma(a) = a^p$ 是域同态；$F$ 有限时 $\sigma$ 是自同构。

**证明（$n = 1$ 的二项式）：** 对 $a, b \in F$：

$$
(a + b)^p = \sum_{k=0}^p \binom{p}{k} a^{p-k} b^k = a^p + b^p
$$

中间项系数 $\binom{p}{k}$ 对 $1 \le k \le p-1$ 被 $p$ 整除，特征 $p$ 下消失。故 $\sigma(a + b) = \sigma(a) + \sigma(b)$；$\sigma(ab) = (ab)^p = a^p b^p = \sigma(a)\sigma(b)$。$F$ 有限时，$\sigma$ 是单射（核是 $\{0\}$，因为 $a^p = 0 \Rightarrow a = 0$），有限集单射即满射，$\sigma$ 是自同构。$\blacksquare$<span class="marginnote">「$(a+b)^p = a^p + b^p$」在特征 $p$ 里成立（中间项消失）——这是有限域算术与通常算术最大的不同。$\sigma(a) = a^p$ 称为 Frobenius 自同构，它是 $\mathbb{F}_{p^n}$ 的核心对称（下一篇《有限域》将证明 $\operatorname{Gal}(\mathbb{F}_{p^n}/\mathbb{F}_p) = \langle \sigma \rangle$）。AES 的字节运算（$\mathbb{F}_{2^8}$）就生活在这种「特征 2 的算术」里。</span>

## 5 例：素域的结构一览

把素域的两个代表彻底看清。

**$\mathbb{Q}$（特征 0 的素域）：** 有理数域。它是「最小的特征 0 域」——任何特征 0 的域都含一个 $\mathbb{Q}$。$\mathbb{Q}$ 的自同构只有恒等（$\varphi(1) = 1$ 决定一切）。

**$\mathbb{F}_p$（特征 $p$ 的素域）：** $p$ 阶有限域，即 $\mathbb{Z}/p\mathbb{Z}$。它的元素是 $\{ 0, 1, \dots, p-1 \}$，算术按模 $p$。**$\mathbb{F}_p$ 没有真子域**（任何子域都含 $1$，而 $\langle 1 \rangle = \mathbb{F}_p$）——它是素域。

**为什么素域只有这两族？** 因为域的特征只有 0 或 $p$，而素域完全由特征决定（由 $1$ 生成）。**「域的分类」第一步就是「按特征分家」**：$\mathbb{Q}$ 一族（含 $\mathbb{R}, \mathbb{C}$ 等），$\mathbb{F}_p$ 一族（含 $\mathbb{F}_{p^n}$ 等）。<span class="marginnote">素域的「最小性」让它在域扩张里扮演「基域」：$\mathbb{F}_{p^n}$ 的所有算术都建立在 $\mathbb{F}_p$ 之上。第十一篇 Galois 理论处理「基域上的对称」，素域是不可再分的底座——它的自同构平凡（$\mathbb{Q}$、$\mathbb{F}_p$ 的 $\operatorname{Aut}$ 都是 $\{ \mathrm{id} \}$），一切对称都来自「扩张」本身。</span>

## 6 对照速查：特征 0 与特征 p 的世界

把「域被特征切成两半」的整条逻辑排成一张对照表，这是理解后续全部域论的坐标。

| 对比项 | 特征 0 的世界 | 特征 $p$ 的世界 |
| --- | --- | --- |
| 素域 | $\mathbb{Q}$ | $\mathbb{F}_p$ |
| 代表域 | $\mathbb{Q}, \mathbb{R}, \mathbb{C}$ | $\mathbb{F}_p, \mathbb{F}_{p^n}$ |
| $n \cdot 1$ 的加法 | 永不回到 0（无限长） | $p$ 次回到 0（周期） |
| Frobenius 映射 | 无（$a \mapsto a^p$ 平凡化后才有意义） | $\sigma(a) = a^p$ 是自同态/自同构 |
| $(a+b)^p$ | 展开有很多项 | $= a^p + b^p$（中间项消失） |
| 典型应用 | 分析、几何、代数数论 | 有限域、AES、椭圆曲线、编码 |

**为什么这一刀切得这么深？** 因为特征完全由「$1$ 的加法」决定，而 $1$ 是每个域不可少的元素：特征 0 意味着 $1$ 的加法永远不重复，特征 $p$ 意味着恰在 $p$ 处重复。素域由此确定，域的整个「加法骨架」由此确定，连 Frobenius 这样的乘法奇景也只出现在特征 $p$。<span class="marginnote">特征 $p$ 的「奇算术」不是病态，而是另一个完整自洽的世界：AES 的全部字节运算都活在特征 2 里，$(a+b)^2 = a^2 + b^2$ 在 $\mathbb{F}_2$ 中成立——<strong>「不熟悉的算术」恰恰是现代密码学的默认算术</strong>。</span>

**易错辨析｜$\mathbb{F}_p$ 与 $\mathbb{Z}/p\mathbb{Z}$。** $\mathbb{F}_p$ 就是 $\mathbb{Z}/p\mathbb{Z}$——同模 $p$ 的剩余类环在 $p$ 为素数时无零因子、每非零元可逆，于是升级为域。但注意：$\mathbb{Z}/n\mathbb{Z}$ 在 $n$ 为合数时只是环（有零因子），不是域。<strong>「$p$ 为素数」是剩余类环升级为域的全部代价</strong>，这也解释了为什么 RSA 要用合数模 $\mathbb{Z}_n$（只做环运算）而 AES 要用素数幂域 $\mathbb{F}_{2^8}$（需要求逆）。

**再往前看一步**：素域是「不可再分的底座」，一切对称都来自「扩张」本身。第十一篇里，$\operatorname{Gal}(F/\mathbb{F}_p)$ 的所有自同构都要固定素域 $\mathbb{F}_p$（Frobenius 保持 $\mathbb{F}_p$ 不变），这与 $\mathbb{Q}$ 被固定是同一件事——<strong>素域固定 = 对称的参考系固定</strong>，这是 Galois 理论从本节就开始埋下的伏笔。

**一句话总结**：每个域都含一个不可再分的素域——特征 0 是 $\mathbb{Q}$，特征 $p$ 是 $\mathbb{F}_p$；素域由「$1$ 的加乘组合」唯一决定，是一切域扩张的底座。

**数字再走一遍**：$\mathbb{F}_5$ 里 $5 \cdot 1 = 0$，故特征为 5；$\mathbb{F}_5$ 中 $3^{-1} = 2$（$3 \times 2 = 6 \equiv 1$）。落到具体数字上，「素数是域、合数是环」便十分直观：$\mathbb{Z}/4\mathbb{Z}$ 里 $2$ 无逆元（$2 \times 2 = 0$ 是零因子），而 $\mathbb{F}_5$ 的每个非零元都可逆——素性与可逆性的直接对话。

**再校正一次直觉**：$\mathbb{R}$ 的特征是 0（素域 $\mathbb{Q}$），但 $\mathbb{R}$ 远不止 $\mathbb{Q}$——素域是「最小内核」，内核小不等于域小。这就把「素域」与「域」两层概念彻底分开：素域管特征，扩张管大小。

**通向下一节**：域扩张理论的全部工作，都在素域这个底座之上进行。下一节我们将学习如何把 $\mathbb{F}_p$ 或 $\mathbb{Q}$ 扩大成更大的域，并用「扩张次数」这把尺子度量扩张的大小。

## 7 小结

- **域的特征**：0 或素数 $p$（域无零因子 ⟹ 特征无真因子）。
- **素域**：不含真子域的域；恰为 $\mathbb{Q}$（特征 0）或 $\mathbb{F}_p$（特征 $p$），由 $1$ 生成。
- **域的层级**：任何域都是其素域的扩张；特征把域分成 $\mathbb{Q}$-扩张与 $\mathbb{F}_p$-扩张两族。
- **Frobenius**：$\sigma(a) = a^p$ 是特征 $p$ 域的自同态，有限域上为自同构；$(a+b)^p = a^p + b^p$。
- 素域是域扩张与有限域的底座，也是 AES/椭圆曲线密码的算术背景。

在下一节，我们开始攀登素域之上：**域扩张与扩张次数**。把一个域扩大成更大的域，并用「扩张次数」度量扩大的维度——这是有限域与 Galois 理论的骨架。
