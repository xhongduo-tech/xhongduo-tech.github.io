---
title: 理想、模与环同态
date: 2026-08-11
---

# 理想、模与环同态

<div class="epigraph">
<p>若要证明两个数 $a$ 与 $b$ 相等，先用 $a \leq b$ 再证 $a \geq b$ 是不公道的；应当揭示它们相等的内在根据。</p>
<footer>—— 埃米 · 诺特（Emmy Noether）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 交换代数 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从理想开始

1840 年代，库默尔（Kummer）研究费马大定理，必须在分圆域 $\mathbb{Z}[\zeta_p]$ 中做因子分解，却发现那里**唯一分解定理不成立**——数论从此站在十字路口。库默尔凭空引入「理想数」（ideale Zahlen）救活了因子分解；二十年后戴德金（Dedekind）把它们重新定义为**集合**，理想（ideal）由此诞生。<span class="marginnote">库默尔（1810—1893）早年是军人，晚年回母校做教授，一生最好的成果大多在数学教育之余完成。他的一班学生里走出了戴德金、克罗内克与康托尔——一门课带出三位改写数学的人。</span>

今天看来，理想是交换代数（乃至整门代数几何）的**组织中枢**：它既是「能当作数的东西」（因子分解的替代品），又是「环作用的空间」（后面模的雏形）。在第二级《抽象代数》里你已见过环、域与群，这一篇要把视野收拢到**交换环**上，把三个最基础也最亲密的角色——理想、模、同态——一次讲清。它们将支撑起本专题从局部化到维数理论的全部大厦。

## 1 子环、理想与「吸收」

设 $A$ 是一个**含单位元的交换环**。<span class="marginnote">本专题一律默认「环」= 含 $1$ 的交换环，除非特别说明。非交换、不含 1 的变体留给其他专题。</span>回忆第二级内容：**子环**是继承运算且含同一个 $1$ 的子集。但代数结构真正赖以生长的，往往不是子环，而是另一个对象：

**理想（ideal）**：非空子集 $\mathfrak{a} \subseteq A$ 称为理想，若对任意 $x, y \in \mathfrak{a}$ 与 $a \in A$ 都有

$$x + y \in \mathfrak{a}, \qquad ax \in \mathfrak{a}.$$

第二条称为**吸收律**：把 $\mathfrak{a}$ 的元素乘以环中任何元素，仍留在 $\mathfrak{a}$ 里。理想是「被环整体吸收」的子加群，是环作用在其上的最小不变量。

**理想为什么不是「子环」**：在 $\mathbb{Z}$ 中，理想 $n\mathbb{Z}$ 恰好也就是子环，容易造成错觉。一般环里两者是不同物种——子环是「环的缩小版」（必须含 $1$），理想是「环作用的靶子」（被乘法吸收）。一旦理想含 $1$，由吸收律立刻推出它等于整个环。

**辨析｜易错点：** 判断「$\mathfrak{a}$ 是不是理想」只查两条：加法封闭、吸收律。初学者常把「乘法封闭」当成理想的定义——那是子环的条件。乘法封闭（$x,y\in\mathfrak{a} \Rightarrow xy\in\mathfrak{a}$）对理想自动成立（因为 $xy = x\cdot y$ 就是吸收律），但它**不是定义**；反过来说，子环不一定满足吸收律，例如 $\mathbb{Z} \subset \mathbb{Q}$ 中 $\mathbb{Z}$ 是子环，但对 $a=\tfrac12 \in \mathbb{Q}$、$x=1 \in \mathbb{Z}$ 有 $ax = \tfrac12 \notin \mathbb{Z}$。

典型的理想：
- $\mathbb{Z}$ 中全体形如 $n\mathbb{Z}$ 的子群都是理想；零理想 $(0)$ 与单位理想 $(1)=A$ 永远存在。
- 多项式环 $k[x]$ 中，每个 $f$ 生成主理想 $(f)$；$k[x,y]$ 中 $(x,y)$ 是「过原点的多项式全体」。
- 环同态的核。这一点值得单独展开。

## 2 理想的运算与商环

理想的「算术」在 $\mathbb{Z}$ 上就是数的公因子、公倍数运算的抽象：

$$\mathfrak{a} + \mathfrak{b} = \{x + y \mid x\in\mathfrak{a},\ y\in\mathfrak{b}\}, \qquad \mathfrak{a}\mathfrak{b} = \Big\{\sum_i x_i y_i \Big\},\qquad \mathfrak{a} \cap \mathfrak{b}$$

以及**根（radical）**：

$$\sqrt{\mathfrak{a}} = \{x \in A \mid x^n \in \mathfrak{a} \text{ 对某个 } n \geq 1\}.$$

**重点：商环是「把所有理想元压成零」得到的环。** 给定理想 $\mathfrak{a}$，等价关系「$x \sim y \iff x - y \in \mathfrak{a}$」把 $A$ 分成陪集，商集 $A/\mathfrak{a}$ 在继承的运算下构成环，投影 $\pi : A \to A/\mathfrak{a}$ 是同态，核恰为 $\mathfrak{a}$。反之，**任何环同态 $f: A \to B$ 的核 $\ker f$ 都是理想**，且诱导出单射 $\bar{f}: A/\ker f \hookrightarrow B$。理想与同态的核就这样一一对应——这正是上一节说「核是理想」是基本事实的原因。

在 $\mathbb{Z}$ 上，$A/\mathfrak{a}$ 就是模 $n$ 的剩余类环 $\mathbb{Z}/n\mathbb{Z}$；在几何上，$k[x_1,\dots,x_n]/\mathfrak{a}$ 是「限制在 $\mathfrak{a}$ 的零点集合上的函数环」——这个视角在《Hilbert 零点定理与 Zariski 拓扑》一篇将彻底展开。

## 3 模：把线性代数搬到环上

**模（module）**：$A$-模 $M$ 是一个交换加群 $(M, +)$，配以标量乘法 $A \times M \to M$，$(a, m) \mapsto am$，满足

$$a(m_1 + m_2) = am_1 + am_2, \qquad (a+b)m = am + bm, \qquad (ab)m = a(bm), \qquad 1m = m.$$

与向量空间逐条对照，唯一的差别是：**系数环 $A$ 不是域**。

- 交换群 = $\mathbb{Z}$-模（标量就是整数倍）。
- 每个理想 $\mathfrak{a}$ 是 $A$-模（吸收律正是标量乘法的相容性）。
- $A^n$ 是**自由模**；$\mathbb{Z}/2\mathbb{Z}$ 是 $\mathbb{Z}$-模但非自由——它是**挠元**：$2 \cdot \bar{1} = 0$。

**重点：线性代数「选基 → 坐标」在一般环上失效。** 向量空间总有基、维数、秩，因为域上可除；模上不可除，基的存在与否都成问题，维数被「秩」与「挠」取代，于是研究手段从「算维数」升级为「看结构」——正合列与同调工具就此登场。第一级《线性代数》是域上的故事，交换代数是环上的故事，**模就是这条分界线上的第一个主角**。

**辨析｜易错点：** 「$A$-模」不要求 $A$ 与 $M$ 有共同的加法——标量 $a$ 与向量 $m$ 根本不是同一类对象。初学者会把「$a m$ 仍在 $M$ 中」错读成「$M$ 是 $A$ 的某种扩张」；正确的是 $M$ 是「$A$ 作用的靶空间」，类似群的左作用。

## 4 同态与同构定理

环同态、模同态的定义与群情形完全平行。对模同态 $f: M \to N$：

**同态基本定理**：

$$M/\ker f \cong \operatorname{im} f.$$

模的**正合列（exact sequence）**：一串模同态 $\cdots \to M_{i+1} \to M_i \to M_{i-1} \to \cdots$，每处满足 $\ker(\text{出的箭头}) = \operatorname{im}(\text{进的箭头})$。**短正合列**形如

$$0 \longrightarrow M' \overset{f}{\longrightarrow} M \overset{g}{\longrightarrow} M'' \longrightarrow 0,$$

即 $f$ 单、$g$ 满、$\operatorname{im} f = \ker g$。它说的是「$M$ 由 $M'$ 与 $M''$ 拼成」，但并非总是直和——拼法与分裂信息全藏在这条列的边界处。

**重点：正合列是模论的基本语法。** 把「模 $M$」换成「短正合列」，几乎所有问题（秩、深度、平坦性）都会牵出第三条模。后续《张量积与平坦模》《深度与正则序列》的许多定理，本质都是「对短正合列做某个构造，看正合性丢失多少」。

## 5 公式解析：中国剩余定理

交换代数里第一个能「算出结构」的定理，是环论版的**中国剩余定理**。

设 $\mathfrak{a}_1, \dots, \mathfrak{a}_n$ 两两互素（$\mathfrak{a}_i + \mathfrak{a}_j = A$），则

$$\frac{A}{\mathfrak{a}_1 \cap \cdots \cap \mathfrak{a}_n} \;\cong\; \frac{A}{\mathfrak{a}_1} \times \cdots \times \frac{A}{\mathfrak{a}_n}.$$

三步拆解这条公式：

- **第一步，定映射**：自然投影 $\pi: A \to \prod_i A/\mathfrak{a}_i$，$x \mapsto (x \bmod \mathfrak{a}_1, \dots, x \bmod \mathfrak{a}_n)$。核就是 $\bigcap_i \mathfrak{a}_i$，所以诱导单射自动成立——商环再大也至多同构于乘积。
- **第二步，证满射（关键）**：需要造出「在 $\mathfrak{a}_1$ 上余 $0$、在其余各处余 $1$」的元素 $e_1$，以及对称的 $e_i$。用互素条件：$\mathfrak{a}_1 + \mathfrak{a}_2 = A$ 推出 $\mathfrak{a}_1 + \mathfrak{a}_2\cdots\mathfrak{a}_n = A$，故存在 $b_1 \in \mathfrak{a}_1$、$c \in \mathfrak{a}_2\cdots\mathfrak{a}_n$ 使 $b_1 + c = 1$；取 $e_1 = c$ 即得 $e_1 \equiv 1 \pmod{\mathfrak{a}_1}$ 且 $e_1 \equiv 0 \pmod{\mathfrak{a}_j}$（$j \geq 2$）。
- **第三步，拼出任意余数**：给定 $(r_1,\dots,r_n)$，令 $x = \sum_i r_i e_i$，则 $x \equiv r_j \pmod{\mathfrak{a}_j}$ 对每个 $j$ 成立——满射得证。

这组 $e_i$ 是**幂等元**（$e_i^2 = e_i$，且 $e_i e_j = 0$），它们是「把环劈成若干块」的最小单位，也是代数几何中「把空间分成不交开集」的语言预演。<span class="marginnote">回到 $\mathbb{Z}$：互素的 $n_1,\dots,n_k$ 给 $\mathbb{Z}/n_1\cdots n_k \cong \prod \mathbb{Z}/n_i$，就是小学「分而治之」的取模问题，初等数论见第一级《整数与整除》。</span>

## 6 小结

- **理想**是含吸收律的子加群，是「能当数用的」结构；判断理想只查加法封闭与吸收律。
- 理想与**同态的核**一一对应；商环 $A/\mathfrak{a}$ 是「把 $\mathfrak{a}$ 压成 0」的环，中国剩余定理把它拆成乘积。
- **模**是把系数域换成环的向量空间，交换群 = $\mathbb{Z}$-模、理想都是模；环上无基与维数，挠与正合列取而代之。
- **同态基本定理** $M/\ker f \cong \operatorname{im} f$ 与**短正合列**是模论的基本语法。

在下一节，我们给环一个「造分式」的操作：把 $\mathbb{Z} \to \mathbb{Q}$ 的做法推广到任意环与任意「可乘子集」，这就是**局部化**——它将把几何上「只看一点附近」的局部直觉精确化，也是本专题最常用的一把手术刀。
