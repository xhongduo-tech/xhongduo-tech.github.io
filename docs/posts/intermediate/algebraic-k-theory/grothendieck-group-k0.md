---
title: 投射模与 Grothendieck 群 K₀
date: 2026-08-07
---

# 投射模与 Grothendieck 群 K₀

<div class="epigraph">
<p>关键是耐心。孵蛋而不是砸蛋，才能得到小鸡。</p>
<footer>—— 亚历山大·格罗滕迪克（Alexander Grothendieck，《收获与播种》Récoltes et Semailles）</footer>
</div>

<div class="article-byline">
<p>第二级 · 代数 K 理论 ｜ Weibel《The K-book》§2.1–2.2 ｜ 2026-08-07</p>
</div>

## 为什么从投射模与 K₀ 开始

线性代数有一个爽快人心的分类定理：**有限维向量空间在同构意义下完全由维数决定**。可一旦把系数环从域换成一般的环——比如整数环 $\mathbb{Z}$，或多项式环 $k[x_1,\dots,x_n]$——「维数」就不再是完备的不变量。$k[x_1,x_2]$ 上的模远不止「自由模」一种，可它们又常常「差一点就是自由的」。代数 K 理论回答的正是这个问题：**在稳定意义下，把环上的模如何分类**。Grothendieck 群 $K_0(R)$ 就是这个分类的第一级台阶，也是最朴素的一个。<span class="marginnote">K 字母来自德文 <strong>Klasse</strong>（类）。Grothendieck 在 1957 年的 Riemann–Roch 定理中首次构造它；两年后 Atiyah 与 Hirzebruch 把它搬到拓扑空间上，诞生了拓扑 K 理论——这棵树的根在这里，梢在第 12 篇。</span>

这一节也接上了整棵知识树的脉络：你已经在《线性代数》《抽象代数》里见过模与直和，在《范畴论》里见过「同构类」与「函子」；$K_0$ 恰恰是把这三样揉成一个对象。它还是「从极限到大模型」的隐喻：**K₀、K₁、K₂……构成一条无限向上延伸的塔**，正如我们从集合一步步爬向大模型——每一级都在回答上一级不够用的问题。

## 1 投射模：把「可裂」写成一条提升性质

先回到模论。设 $R$ 是环，$P$ 是一个左 $R$-模。**投射模（projective module）**的定义如下：<span class="marginnote">在范畴论语言里，投射模就是「$\mathrm{Hom}(P, -)$ 是正合函子」的那个对象。这里用的是更几何的提升性质——先给直觉，再给范畴。</span>

> **定义（提升性质）。** 对任意满射 $f: M \twoheadrightarrow N$ 与任意同态 $g: P \to N$，都存在同态 $h: P \to M$ 使得 $f \circ h = g$。

$$
\begin{aligned}
& M \xrightarrow{f} N \to 0 \\
& P \xrightarrow{g} N \quad \rightsquigarrow \quad \exists h: P \to M,\ f\circ h = g
\end{aligned}
$$

这句话读起来拗口，直觉却一句话：**只要目标是 $N$，从 $P$ 出发的箭头总能「穿过」任何满射抬升到 $M$**——即「从 $P$ 到商模的映射总能提升到整个模」。自由模显然满足：基可以逐个拉回。投射模就是「长得像自由模」的模。

投射模有三条等价的刻画，是做题时的工具包：

- **直和因子**：$P$ 投射 $\iff$ 存在 $Q$ 使 $P \oplus Q$ 是自由模（$P$ 是某个自由模的直和项）。
- **正合性**：$P$ 投射 $\iff$ $\mathrm{Hom}_R(P, -)$ 把短正合列变成短正合列（保持满射）。
- **矩阵刻画**：$P$ 有限生成投射 $\iff$ 存在幂等矩阵 $e \in M_n(R)$（$e^2 = e$）使得 $P \cong eR^n$，即 $P$ 同构于 $e$ 作用在 $R^n$ 上的像。

**辨析｜易错点：** 初学者常误以为「投射 = 自由」。$R = \mathbb{Z}/6\mathbb{Z}$ 时，$\mathbb{Z}/2\mathbb{Z}$ 是 $\mathbb{Z}/3\mathbb{Z} \oplus \mathbb{Z}/2\mathbb{Z} \cong \mathbb{Z}/6\mathbb{Z}$ 的直和项，因此它是投射的，却绝不是自由的——自由模的秩唯一性对它失效。投射与自由的差距，正是 $K_0$ 要测量的东西。

## 2 直和幺半群与稳定等价

把所有有限生成投射模的同构类收进一个集合，记作 $\mathrm{P}(R)$。直和 $\oplus$ 在同构类上诱导一个二元运算，它满足结合律、交换律，且平凡模 $0$ 是单位元——于是 $(\mathrm{P}(R), \oplus)$ 构成一个**交换幺半群（commutative monoid）**。<span class="marginnote">注意这里「没有减法」：直和没有逆运算。一个模加一个非零模只会「变大」，不会变回 0。这正是要引入 Grothendieck 群的原因——给幺半群人工补上减法。</span>

幺半群里最关键的等价关系叫**稳定等价（stable equivalence）**：

$$
P \sim Q \iff P \oplus F \cong Q \oplus F \quad \text{（对某个有限生成自由模 } F \text{）}
$$

**直觉**：如果两个模在「各垫上一块自由模」之后同构，就把它们看成同一个。为什么会需要垫自由模？因为**消去律（cancellation）一般会失败**：$P \oplus F \cong Q \oplus F$ 推不出 $P \cong Q$。

**辨析｜易错点：** 这不是矫情。取 $R = \mathbb{R}[x,y,z]/(x^2+y^2+z^2-1)$（实三维球面的坐标环），存在非自由模 $P$ 使得 $P \oplus R \cong R^3$——垫上一块 $R$ 就看不出差别了。这类「悄悄隐藏的怪模」正是拓扑现象在代数里的化身，第 5 篇 Swan 定理会把这层窗户纸捅破。

## 3 Grothendieck 群：把减法还给幺半群

现在面临一个普遍的代数问题：**给定一个交换幺半群，如何造出一个「最经济的」交换群，让它包含原来的幺半群？** 答案就是 Grothendieck 群构造。

**Grothendieck 群（Grothendieck group）。** 设 $(M, +)$ 是交换幺半群。在 $M \times M$ 上定义等价关系

$$
(a, b) \sim (a', b') \iff \exists\, c \in M,\ a + b' + c = a' + b + c
$$

把 $(a,b)$ 的等价类记作 $a - b$，则商 $G(M) = M \times M / \sim$ 在运算 $(a-b) + (a'-b') = (a+a') - (b+b')$ 下成为交换群，称为 $M$ 的 **Grothendieck 群**。

**这满足一个普适性质**：任意幺半群同态 $\varphi: M \to A$（$A$ 为交换群）都唯一地穿过 $G(M)$。换句话说，$G$ 是「从交换幺半群范畴到交换群范畴」这个遗忘函子的左伴随——$G(M)$ 是把 $M$ 改造成群**最省力**的办法。<span class="marginnote">「最省力」正是伴随函子的味道：在《范畴论》里你见过「自由构造是遗忘函子的左伴随」，Grothendieck 群是同族兄弟——自由群加的是「新元素」，Grothendieck 群加的是「负元素」。</span>

现在把它用到 $\mathrm{P}(R)$ 上：

$$
\boxed{\,K_0(R) = G(\mathrm{P}(R), \oplus)\,}
$$

$K_0(R)$ 的元素是形式差 $[P] - [Q]$，其中 $[P]$ 表示 $P$ 的同构类。**Grothendieck 群的正式名称也叫 $K_0$**，这里的下标 0 意味着「第零级」——后面还有 $K_1, K_2, \dots$，一层一层往上搭。

## 4 公式解析：K₀(R) 的生成元与关系

$K_0(R)$ 有一种更透明的「生成元与关系」写法，它把 Grothendieck 群构造的全部内容压缩成一行：

$$
K_0(R) = \frac{\displaystyle \mathbb{Z}\big\{[P] : [P] \in \mathrm{P}(R)\big\}}{\big\langle [P \oplus Q] - [P] - [Q] \big\rangle}
$$

**第一步，看分子**：$\mathbb{Z}\{[P]\}$ 是「以每个同构类 $[P]$ 为自由生成元的自由交换群」。它把每个模同构类当成一个独立的符号，先不规定任何关系——于是 $[P]$ 可以取任意整数倍、任意相加。

**第二步，看分母**：尖括号里是一个生成关系——「$[P\oplus Q]$ 应当等于 $[P] + [Q]$」。分母正是「由这些差张成的子群」，取商就是在自由群里**强制直和变成加法**。

**第三步，读结果**：商群里的元素都能写成 $[P] - [Q]$，而且两条性质同时成立——同构类仍是类，直和仍翻译成加；代价是**个别模的个体身份被抹平**，只留下稳定意义下的信息。

**第四步，验证判定规则**：在 $K_0(R)$ 里 $[P] = [Q]$ 当且仅当存在自由模 $F$ 使 $P \oplus F \cong Q \oplus F$——这正是上一节的稳定等价。可以拿「垫自由模」当**免费垫脚石**：为了比较 $P$ 与 $Q$，各自垫一块，垫完了看齐没有。这就是 Grothendieck 群的全部哲学。

## 5 K₀ 能记住什么：秩、约化 K₀ 与类群

$K_0(R)$ 是抽象的，但能算出具体结果。先看最常用的计算：

| 环 $R$ | $K_0(R)$ | 解释 |
| --- | --- | --- |
| 域 $k$ | $\mathbb{Z}$ | 向量空间由维数决定 |
| 主理想整环（含 $\mathbb{Z}$） | $\mathbb{Z}$ | 有限生成无挠模皆自由 |
| $k[x_1,\dots,x_n]$（域上多项式环） | $\mathbb{Z}$ | Quillen–Suslin：投射即自由 |
| 分式域不出现的 Dedekind 整环 $R$ | $\mathbb{Z} \oplus \mathrm{Pic}(R)$ | 理想类群钻进 $K_0$ |
| 乘积 $k_1 \times k_2$ | $\mathbb{Z}^2$ | 分量逐个取秩 |

<span class="marginnote">第三行是大定理：1976 年 Quillen 与 Suslin 独立证明多项式环上投射模必自由（Serre 猜想）。第 4 行则预告了第 11 篇：Dedekind 整环上 $K_0$ 多出来的 $\mathrm{Pic}(R)$ 恰是理想类群——代数数论因此与 K 理论水乳交融。</span>

对交换环 $R$，同态「取秩」$\mathrm{rank}: K_0(R) \to \mathbb{Z}$ 有分裂，于是有正合列

$$
0 \to \widetilde K_0(R) \to K_0(R) \xrightarrow{\mathrm{rank}} \mathbb{Z} \to 0
$$

**约化 Grothendieck 群（reduced K-group）** $\widetilde K_0(R) = \ker(\mathrm{rank})$ 专门度量「秩为 0 却不一定自由」的模——它非零，就意味着环上有真正的怪模。$K_0(R) \cong \widetilde K_0(R) \oplus \mathbb{Z}$，对许多环成立。

$K_0$ 还满足两条**稳定性**：$K_0(R) \cong K_0(M_n(R))$（矩阵环不改变 $K_0$），且任意环同态 $f: R \to S$ 都诱导群同态 $f_*: K_0(R) \to K_0(S)$——于是 **$K_0$ 是从环范畴到交换群范畴的一个协变函子**。这一「函子性」是全部 K 理论的命脉，后面每升高一级都靠它延续。

## 6 小结

- **投射模** = 自由模的直和因子 = $\mathrm{Hom}(P,-)$ 正合的模；有限生成投射模由幂等矩阵 $e^2=e$ 实现为 $eR^n$。
- **稳定等价**：$P \sim Q \iff P \oplus F \cong Q \oplus F$；消去律失败使得「垫自由模」成为必要。
- **Grothendieck 群** $G(M)$：给交换幺半群 $M$ 补减法的最省力交换群，满足普适性质。
- **$K_0(R) = G(\mathrm{P}(R), \oplus)$**：以同构类为生成元，以 $[P\oplus Q] = [P]+[Q]$ 为关系。
- **计算**：域、PID、多项式环的 $K_0 = \mathbb{Z}$；Dedekind 整环多出理想类群；$\widetilde K_0$ 度量「秩零的怪模」。
- **函子性**：$R \mapsto K_0(R)$ 是协变函子，矩阵环不改变 $K_0$。

在下一节，我们将沿着塔向上爬一级，研究 $K_1$