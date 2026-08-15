---
title: Künneth 公式：积空间的同调
date: 2026-08-07
---

# Künneth 公式：积空间的同调

<div class="epigraph">
<p>两个空间的乘积，其同调由每个空间的同调「配对」而成——但配对的方式，藏着张量代数最深的一课。</p>
<footer>—— 赫尔曼 · 库内特（Hermann Künneth）</footer>
</div>

<div class="article-byline">
<p>第二级 · 代数拓扑 ｜ Hatcher 第2.2、3.1章；Munkres 第12章 ｜ 2026-08-07</p>
</div>

## 为什么从 Künneth 公式开始

环面 $T^2 = S^1 \times S^1$ 的同调是 $H_0 = \mathbb{Z}$、$H_1 = \mathbb{Z}^2$、$H_2 =
\mathbb{Z}$。两个圆各自同调
$\mathbb{Z}$（$H_0$、$H_1$），配对出来的维度数恰好是「每个因子贡献一维」的直和：$H_1(T^2) = H_1(S^1)
\otimes H_0(S^1) \oplus H_0(S^1) \otimes H_1(S^1) = \mathbb{Z} \oplus
\mathbb{Z}$。但事情没那么简单——**当同调带挠时，直积的同调会多出一个「交叉项」**，朴素猜想 $H_n(X \times Y) =
\bigoplus_{i+j=n} H_i(X) \otimes H_j(Y)$ 会**失败**。

**Künneth 公式（Künneth formula）**精确回答了 $H_*(X \times Y)$
是什么：它是一条短正合序列，头是「张量积的直和」，尾是「$\operatorname{Tor}$
项」。这条公式是代数拓扑与同调代数（第二级《同调代数》专题）交汇的核心：**空间乘积的问题，被翻译成阿贝尔群张量积与
$\operatorname{Tor}$
函子的问题**。读完它，你会真正理解「乘积空间」为什么比「无交并」深刻得多——无交并的同调是直和（可加性公理），乘积的同调是张量积加 Tor。


Künneth
公式是「无交并直和」直觉的乘积版升级：两个空间的乘积，其同调不只是「配对」，还要多一个修正项。读这一节时请把握一条主线——**先记住无挠情形的漂亮公式，再理解
Tor 项为什么会在有挠时出现**。Tor 不是麻烦，而是「张量积丢掉的信息的记账员」。当你以后在第二级《同调代数》里正式认识 Tor/Ext
时，会发现它们早就在这里与你打过照面了：代数概念从几何问题中自然生长，是最好的记忆方式。

## 1 交叉积：把两个洞「乘」成一个洞

在写 Künneth 公式前，先认识它的主角：**交叉积（cross product）**。

$$H_i(X) \times H_j(Y) \longrightarrow H_{i+j}(X \times Y), \qquad (\alpha, \beta) \longmapsto \alpha \times \beta$$

构造：在 Δ-复形或胞腔模型下，取 $X$ 的一个 $i$-单形 $\sigma$ 与 $Y$ 的一个 $j$-单形
$\tau$，它们的「乘积」$\sigma \times \tau$ 是 $X \times Y$ 里的一个 $(i+j)$-维对象（$\Delta^i
\times \Delta^j$ 可分解为 $i+j \choose i$ 个 $(i+j)$-单形）。对边界算子有**乘法的莱布尼茨法则**：

$$\partial(\sigma \times \tau) = \partial \sigma \times \tau + (-1)^i\, \sigma \times \partial \tau$$

这保证交叉积把「循环 × 循环」送到「循环」，把「边界」送到「边界」，从而在同调层良定义，并诱导上面的双线性配对。<span class="marginnote">符号 $(-1)^i$ 又是定向的账本：$\partial(\sigma\times\tau)$
展开时，$\sigma$ 的边界穿过 $\tau$ 要「翻面」，于是多一个 $(-1)^i$。这条「广义莱布尼茨法则」在微分几何里也有对应——外微分
$\mathrm{d}(\omega \wedge \eta) = \mathrm{d}\omega \wedge \eta +
(-1)^{|\omega|} \omega \wedge \mathrm{d}\eta$。</span>

交叉积是**双线性**的、与映射复合相容的：$(f_\* \alpha) \times (g_\* \beta) = (f \times g)_\*(\alpha
\times \beta)$。它是 Künneth 公式里的主要映射。

## 2 Künneth 公式

**定理（Künneth 公式）。** 设 $X, Y$ 是 CW 复形（或更一般的合理空间），则存在**分裂**的短正合序列

$$0 \to \bigoplus_{i+j=n} H_i(X) \otimes_\mathbb{Z} H_j(Y) \xrightarrow{\;\times\;} H_n(X \times Y) \xrightarrow{\;} \bigoplus_{i+j=n-1} \operatorname{Tor}_1^\mathbb{Z}\big(H_i(X),\ H_j(Y)\big) \to 0$$

映射 $\times$ 是交叉积，右端项是 $\operatorname{Tor}$ 函子（张量积的「误差项」，见下节）。**分裂**意味着同构（不典范）：

$$H_n(X \times Y) \cong \Big(\bigoplus_{i+j=n} H_i(X) \otimes H_j(Y)\Big) \oplus \Big(\bigoplus_{i+j=n-1} \operatorname{Tor}\big(H_i(X), H_j(Y)\big)\Big)$$

**当每个 $H_i(X)$、$H_j(Y)$ 都是自由阿贝尔群时**（无挠），$\operatorname{Tor} = 0$，公式简化为漂亮的同构：

$$H_n(X \times Y) \cong \bigoplus_{i+j=n} H_i(X) \otimes H_j(Y)$$

环面、球面、$\mathbb{CP}^n$ 等无挠空间的乘积都归入这个简单情形。<span class="marginnote">Tor
的几何来源：$\mathbb{Z} \otimes \mathbb{Z}_2 = \mathbb{Z}_2$ 但 $\mathbb{Z}_2 \otimes
\mathbb{Z}_2$ 只有 $\mathbb{Z}_2$，挠与挠相乘「掉信息」；Tor
正是记录「张量积丢掉了什么」的修正项。它是同调代数的第一批主角之一，见第二级《同调代数》。</span>

## 3 Tor 项：为什么朴素公式会失败

**反例**：算 $H_*( \mathbb{RP}^2 \times S^1)$。$\mathbb{RP}^2$ 有 $H_1 =
\mathbb{Z}_2$（挠）。朴素直和项给出 $H_2 \supseteq H_1(\mathbb{RP}^2) \otimes H_1(S^1) =
\mathbb{Z}_2 \otimes \mathbb{Z} = \mathbb{Z}_2$；但 Künneth 公式还要求加入 Tor
项：$\operatorname{Tor}(\mathbb{Z}_2, H_0(S^1)) =
\operatorname{Tor}(\mathbb{Z}_2, \mathbb{Z}) = 0$……这里 Tor 恰好为
0，但换个组合就非零。真正的病案：$\mathbb{RP}^2 \times \mathbb{RP}^2$ 的 $H_2$ 包含
$\operatorname{Tor}(\mathbb{Z}_2, \mathbb{Z}_2) = \mathbb{Z}_2$——朴素张量积
$\mathbb{Z}_2 \otimes \mathbb{Z}_2 = \mathbb{Z}_2$ 之外又多了一个
$\mathbb{Z}_2$，$H_2$ 的秩翻倍。

**Tor 是什么**：$\operatorname{Tor}^\mathbb{Z}(A, B)$ 是「张量积的导出函子」——粗略说，它度量「把 $B$
的挠与 $A$ 的挠同时拉进 $A \otimes B$ 时多出来的部分」。对有限生成阿贝尔群，只需一个口诀：

$$\operatorname{Tor}(\mathbb{Z}, B) = 0, \qquad \operatorname{Tor}(\mathbb{Z}_m, \mathbb{Z}_n) = \mathbb{Z}_{\gcd(m,n)}, \qquad \operatorname{Tor}(A \oplus A', B) = \operatorname{Tor}(A,B) \oplus \operatorname{Tor}(A',B)$$

**辨析｜易错点：** Künneth 公式里 Tor 的下标是 $i + j = n-1$，比张量积项低**一维**。这是「交叉积的莱布尼茨法则」中
$(-1)^i$ 的直接后果——误差项永远在维度 $n-1$ 出现。写公式时务必对齐维数，这是初学者最常抄错的地方。

## 4 例子：积空间的同调一览

**例 1：$T^n = (S^1)^n$。** 每因子自由，朴素公式成立：$H_k(T^n) =
\mathbb{Z}^{\binom{n}{k}}$。$T^2$：$H_0 = \mathbb{Z}$，$H_1 = \mathbb{Z}^2$，$H_2
= \mathbb{Z}$。

**例 2：$S^p \times S^q$（$p, q \ge 1$）。** $H_0 = \mathbb{Z}$，$H_p =\mathbb{Z}$，$H_q = \mathbb{Z}$，$H_{p+q} = \mathbb{Z}$（生成元是基本类 $[S^p] \times[S^q]$），其余为 0。<span class="marginnote">$H_{p+q}(S^p \times S^q) = \mathbb{Z}$
来自「整个乘积」作为唯一 $(p+q)$-维洞；生成元 $\alpha \times \beta$ 正是两个因子基本类的交叉积。这为 Poincaré
对偶篇的「乘积 = 交叠」直觉埋下伏笔。</span>

**例 3：$\mathbb{CP}^m \times \mathbb{CP}^n$。** 自由，$H_k = \mathbb{Z}^{\#\{(i,j):
i+j=k,\ i \le 2m,\ j \le 2n,\ i,j \text{ 偶}\}}$——偶维生成元的张量积组合。

## 5 公式解析：Künneth 短正合序列

$$0 \to \bigoplus_{i+j=n} H_i(X) \otimes H_j(Y) \xrightarrow{\ \times\ } H_n(X \times Y) \to \bigoplus_{i+j=n-1} \operatorname{Tor}(H_i(X), H_j(Y)) \to 0$$

- **第一步，主项**：$\bigoplus_{i+j=n} H_i(X) \otimes H_j(Y)$。每个 $(i,j)$ 配对给出一份「$i$-维洞与 $j$-维洞的张量积」，交叉积把它们送入 $H_n(X\times Y)$。**无挠时这就是全部答案。**
- **第二步，尾项**：$\bigoplus_{i+j=n-1} \operatorname{Tor}(H_i(X), H_j(Y))$。它报告「张量积漏掉的挠信息」，维度比主项低一维。**有挠时它非零**，是朴素公式失败的原因。
- **第三步，正合性**：左端单射、右端满射、中段正合。整体**分裂**，所以可以直接写同构——但分裂同构不典范（依赖基），书写时短正合序列形式更「诚实」。


**例：$H_*(S^2 \times S^2)$。** 两个因子同调均为 $\mathbb{Z}$（维 0、2），无挠。Künneth
直和项给出：$H_0 = \mathbb{Z}$，$H_2 = H_2(S^2)\otimes H_0 \oplus H_0 \otimes
H_2(S^2) = \mathbb{Z}^2$，$H_4 = \mathbb{Z}$，其余为 0。$H_4$ 的生成元是基本类的交叉积 $[S^2]
\times [S^2]$。**对比**：$S^2 \times S^2$ 与 $S^2 \vee S^2 \vee S^4$ 的同调群相同（都是
$\mathbb{Z}$、$\mathbb{Z}^2$、$\mathbb{Z}$），却靠上同调环区分（第 13
篇）——同调群相同的两个空间，几何可以完全不同。这正是我们一路强调「同调群只是清单，环结构才是族谱」的实例。

**Tor 的一项计算**：$\operatorname{Tor}(\mathbb{Z}_2, \mathbb{Z}_2) = \mathbb{Z}_2$
从哪来？取 $\mathbb{Z}$ 的投射分解 $0 \to \mathbb{Z} \xrightarrow{\times 2} \mathbb{Z}
\to \mathbb{Z}_2 \to 0$，对 $\mathbb{Z}_2$ 张量后得 $0 \to \mathbb{Z}_2
\xrightarrow{0} \mathbb{Z}_2 \to \mathbb{Z}_2 \to 0$，其中间的同调 $\ker 0 /
\operatorname{im} 0 = \mathbb{Z}_2$ 就是 $\operatorname{Tor}(\mathbb{Z}_2,
\mathbb{Z}_2)$。**「用自由分解算 Tor」是标准程序**，本段只是预告——完整理论在第二级《同调代数》里。

## 6 小结

- **交叉积**：$\alpha \times \beta \in H_{i+j}(X\times Y)$，满足莱布尼茨法则 $\partial(\sigma\times\tau) = \partial\sigma\times\tau + (-1)^i \sigma\times\partial\tau$。
- **Künneth 公式**：$0 \to \bigoplus_{i+j=n} H_i \otimes H_j \to H_n(X\times Y) \to \bigoplus_{i+j=n-1} \operatorname{Tor}(H_i, H_j) \to 0$，分裂。
- **无挠情形**：$H_n(X\times Y) \cong \bigoplus_{i+j=n} H_i(X) \otimes H_j(Y)$。
- **Tor 项**：记录挠相乘时的误差，$\operatorname{Tor}(\mathbb{Z}_m,\mathbb{Z}_n) = \mathbb{Z}_{\gcd(m,n)}$。
- **例子**：$T^n$、$S^p\times S^q$、$\mathbb{CP}^m\times\mathbb{CP}^n$。

在下一节，我们将转向同调论的「对偶面」——**上同调**，并给出连接同调与上同调的**万有系数定理**：它用 $\operatorname{Tor}$ 与
$\operatorname{Ext}$