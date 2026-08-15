---
title: Bass–Heller–Swan 定理
date: 2026-08-07
---

# Bass–Heller–Swan 定理

<div class="epigraph">
<p>代数 K 理论是一门把「自由」变成「加法」的手艺。</p>
<footer>—— 海曼·巴斯（Hyman Bass）</footer>
</div>

<div class="article-byline">
<p>第二级 · 代数 K 理论 ｜ Weibel《The K-book》§9 ｜ 2026-08-07</p>
</div>

## 为什么研究多项式环的 K 理论

前八节建立了 K 理论的「语言」，但「怎么算」的大问题还悬着。Bass–Heller–Swan 定理给出了 K 理论最有力的一条**递推公式**：它把**Laurent 多项式环** $R[t, t^{-1}]$ 的 K 群，用 $R$ 的 K 群与一个「低一级」的 K 群写出来。这就像数列里的二阶递推：知道了前两项，整个序列都被锚定。<span class="marginnote">在拓扑里，Laurent 多项式环对应「乘圆」$X \times S^1$——$R[t,t^{-1}]$ 的几何是 $R$ 的几何「打一个圈」。Bass–Heller–Swan 的拓扑原型正是 Bott 周期性里「$K(X\times S^1) = K(X)\oplus K^{-1}(X)$」式的分裂。</span>

这条定理还是「从极限到大模型」中**递推结构**的数学化身：用一个「环上加个生成元」的机械操作（$R \mapsto R[t,t^{-1}]$），把高阶 K 群折算成低阶——知识树从第 $n$ 级长向第 $n+1$ 级，靠的就是这种「换一条轴再看」的规则。

## 1 多项式环的 K 理论：同伦不变性

先看「只加一个变量、不加逆」的 $R[t]$。一个直觉是：多项式环 $R[t]$ 与 $R$「差不多」——$t$ 不过是多出来的一个哑变量。K 理论把这条直觉变成定理：

$$
K_n\big(R[t]\big) \cong K_n(R) \qquad (\text{对所有 } n \ge 0)
$$

**为什么成立**：设 $\iota: R \hookrightarrow R[t]$ 与 $\varepsilon: R[t] \twoheadrightarrow R$（$t \mapsto 0$），$\varepsilon \circ \iota = \mathrm{id}$。K 理论是函子，故 $K_n(\iota) \circ K_n(\varepsilon) = \mathrm{id}$。需要证明的是另一方向 $K_n(\iota) \circ K_n(\varepsilon) = \mathrm{id}$——这要用「同伦」：$t$ 可以连续地「滑」到 $0$（通过 $t \mapsto st$），而 $K_n$ 是**同伦不变的**（对环的「多项式同伦」不变）。<span class="marginnote">这条「同伦不变性」$K_*(R[t]) = K_*(R)$ 是代数 K 理论的基本公理之一，由 Bass（对 $K_0,K_1$）与 Quillen（对一切 $n$）证明。它对应拓扑里的 $K^*(X \times D^1) = K^*(X)$——<strong>加一个可收缩方向，K 群不变</strong>。Swan 定理（第 5 篇）保证这两个世界说的是同一件事。</span>

于是「$R \to R[t]$」对 K 理论来说**看不见**。真正能改变 K 群的是「取逆」——$t$ 一旦变成可逆元 $t^{-1}$，环里就多了一个「循环」的方向，这正是 $K_{n-1}(R)$ 从地里冒出来的地方。

## 2 Bass–Heller–Swan 定理

> **Bass–Heller–Swan 定理（Fundamental Theorem，Bass 1964；Bass–Heller–Swan；Quillen 全部 $n$）。** 对任意环 $R$ 与 $n \ge 0$：
> $$
> K_n\big(R[t, t^{-1}]\big) \ \cong\ K_n(R)\ \oplus\ K_{n-1}(R)\ \oplus\ N K_n(R)\ \oplus\ N K_n(R)
> $$
> 其中 $N K_n(R) = \ker\big(K_n(R[t]) \to K_n(R)\big)$ 是**增广核（nil-K 理论）**。

**逐步读**：
- **$K_n(R)$**：常系数环 $R \subset R[t,t^{-1}]$ 的贡献——「不含 $t$」的模块信息。
- **$K_{n-1}(R)$**：多出来的那一级——「取 $t$ 的逆」带来的**悬垂（suspension）**贡献，就像 $S^1$ 的环路空间把同伦群降了一级：$K_n(X \times S^1) \supset K_{n-1}(X)$。
- **两份 $N K_n(R)$**：来自两个「无穷端」$t \to 0$ 与 $t \to \infty$ 各自的障碍——每一端贡献一份「$t$ 造成但不来自常数环」的扭曲信息。

**名字的来历**：$N$ 代表 **nilpotent（幂零）**。$N K_0(R)$ 由 $R[t]$ 上「被 $t$ 幂零化」的模块信息生成；对一般环，$N K_n(R)$ 通常是**巨大的 $\mathbb{Q}$-向量空间**（无穷维），它的消失与否是「环是否正则」的一块试金石。

## 3 Nil-K 理论：挠性的度量

$N K_n(R) = \ker\big(K_n(R[t]) \to K_n(R)\big)$ 度量「多项式环比原环多出来的那部分 K 群」。既然 $K_n(R[t]) \cong K_n(R)$，这个核不是「$R[t]$ 的 K 群比 $R$ 大」——它是**映射本身的核**：$R[t]$ 的 K 群里那些「$t \mapsto 0$ 后消失」的类。

**正则环**上 nil-K 全部消失：

$$
R \text{ 正则（正则 Noether 环）} \quad\Longrightarrow\quad N K_n(R) = 0 \ \ (\forall n)
$$

因此对正则环得到干净的分裂：

$$
K_n\big(R[t,t^{-1}]\big) = K_n(R) \oplus K_{n-1}(R)
$$

例如域 $k$（当然正则）：$K_1(k[t,t^{-1}]) = k^\times \oplus \mathbb{Z}$，而 $k[t,t^{-1}]$ 的单位群恰是 $k^\times \cdot t^{\mathbb{Z}}$——**公式与「单位群」直接对账**，是定理最好的体检。

**辨析｜易错点：** $N K_n(R)$ 可以非零且非常巨大。取 $R = \mathbb{Z}$，$N K_1(\mathbb{Z})$ 是无穷维 $\mathbb{Q}$-向量空间——所以 $K_1(\mathbb{Z}[t,t^{-1}])$ 比 $K_1(\mathbb{Z}) = \mathbb{Z}/2$ 庞大多了。**「多项式加一个逆」在非正则环上是会爆炸的**。另一个易错：$K_n(R[t]) \cong K_n(R)$ 与「$N K_n = 0$」不是一回事——前者对一切环成立，后者只在正则环成立。

## 4 公式解析：Laurent 多项式的加法分解

把定理写成「空间级」的语言，便于看清每块贡献的来源：

$$
K_*\big(R[t,t^{-1}]\big) = K_*(R) \oplus \Sigma K_*(R) \oplus N K_*(R)^{\oplus 2}
$$

**第一步，看 $K_*(R[t]) = K_*(R)$**：同伦不变性说明「$t$ 的幂（不取逆）」不产生新 K 类。所以 $R[t,t^{-1}]$ 的多余结构只能来自 $t^{-1}$ 的存在——取逆 = 加一个「循环」。

**第二步，看 $\Sigma K_*(R)$**：$t^{-1}$ 的存在给环增加一个「转一圈」的自由度，对应空间上 $X$ 乘一个 $S^1$：$\Sigma K_*(R)$ 就是「$K_*(X \times S^1)$ 里多出的 $K_{*-1}(X)$」——**圆周每给一次，K 群降一级**。这是 Bott 周期性里 $K^*(X\times S^1) = K^*(X) \oplus K^{*-1}(X)$ 的代数回响。

**第三步，看 $N K_*^{\oplus 2}$**：$R[t] \to R[t,t^{-1}]$ 是「取 $t$ 的可逆化」的局部化，由 Q-构造的 Localization 定理（第 7 篇），它诱导长正合列，其中「增广核」$N K_*$ 出现两次——分别来自 $t$ 的**正向端**与**反向端**两个方向的「被幂零元扭曲」的类。正则环没有幂零扭曲，两份就归零。

**第四步，对账**：对 $R = k$（域），$N K_* = 0$，公式给出 $K_n(k[t,t^{-1}]) = K_n(k) \oplus K_{n-1}(k)$。$n = 1$ 时 $K_1(k[t,t^{-1}]) = k^\times \oplus \mathbb{Z}$，与「Laurent 多项式的单位 = 非零标量 × $t$ 的整数次幂」**完全吻合**——一条定理，代数、拓扑、单位群三面对账。

## 5 应用：正则环、投射空间与递推

**递归计算**：定理允许「剥掉一个变量」。对 $k[x_1, \dots, x_m][t, t^{-1}]$ 反复使用，得到

$$
K_*(\mathbb{A}^r_k \times \mathbb{G}_m^{\,s}) \ = \ \bigoplus_j \binom{s}{j}\, K_{*-j}(k)
$$

结合第 8 篇的投射丛公式，可以一路算出 $K_*(\mathbb{P}^r_k \times \mathbb{G}_m^s)$——**高阶 K 群在「好」环上的计算，几乎全是 BHS 递推 + 投射丛公式的组合拳**。

**空间的分解**：正则环 $R$ 上有空间级分解

$$
BGL\big(R[t,t^{-1}]\big)^+ \ \simeq\ BGL(R)^+ \times \Omega\, BGL(R)^+
$$

即「Laurent 环的 K 空间」是同伦等价于「$R$ 的 K 空间乘上它自己的环路空间」——**代数 K 理论同伦不变量在 $S^1$ 上「打结」的精确公式**。这个分解在稳定同伦论、可逆模的自同构群计算中反复出现。

**通向数论**：把 $R[t,t^{-1}]$ 换成「有理函数域」$k(t)$，配合局部化序列，可以计算 $K_n(k(t))$ 并引向「Beilinson 的递推」——那是第 11 篇类群与调节子的入口。**BHS 是算术 K 理论的引擎之一**。

### 术语速查表：Bass–Heller–Swan

| 记号 | 名称 | 含义 |
| --- | --- | --- |
| $R[t]$ | 多项式环 | 加一个变量，K 群不变 |
| $R[t,t^{-1}]$ | Laurent 多项式环 | 加一个变量及其逆 |
| $NK_n(R)$ | Nil-K | $\ker(K_n(R[t])\to K_n(R))$ |
| $\Sigma K_*(R)$ | 悬垂 | $K_{*-1}$ 的平移 |
| 正则环 | regular | 所有有限生成模有有限投射维数 |
| 同伦不变性 | —— | $K_n(R[t]) \cong K_n(R)$ |

**辨析｜易错点：** 「$K_n(R[t]) = K_n(R)$」对**一切**环成立，而「$NK_n = 0$」只在**正则环**成立——这两条经常被混为一谈。非正则环（如 $R = \mathbb{Z}[x]/(x^2)$）上，$NK_n(R)$ 可以非零到成为无穷维 $\mathbb{Q}$-向量空间，BHS 公式里那两份 $NK$ 瞬间「充气」。区分「同伦不变」与「nil 消失」是读懂 BHS 的第一道门。

## 6 小结

- **同伦不变性**：$K_n(R[t]) \cong K_n(R)$（多项式环不改变 K 群），对应拓扑里「加可收缩方向」。
- **Bass–Heller–Swan**：$K_n(R[t,t^{-1}]) \cong K_n(R) \oplus K_{n-1}(R) \oplus N K_n(R) \oplus N K_n(R)$。
- **Nil-K**：$N K_n(R) = \ker(K_n(R[t]) \to K_n(R))$，正则环上为 0，一般环上往往是巨大的 $\mathbb{Q}$-向量空间。
- **正则分裂**：$K_n(R[t,t^{-1}]) = K_n(R) \oplus K_{n-1}(R)$；对域 $k$ 与单位群 $k^\times \cdot t^{\mathbb{Z}}$ 对账。
- **空间版本**：$BGL(R[t,t^{-1}])^+ \simeq BGL(R)^+ \times \Omega BGL(R)^+$（正则 $R$）。
- **用途**：射影空间、环面簇的 K 群递推；算术 K 理论的引擎。

在下一节，我们回到 K 理论最原始的应用——**几何拓扑**。Whitehead 挠率把 $K_1$