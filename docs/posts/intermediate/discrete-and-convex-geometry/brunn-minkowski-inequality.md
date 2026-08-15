---
title: Brunn–Minkowski 不等式
date: 2026-08-07
---

# Brunn–Minkowski 不等式

<div class="epigraph">
<p>Brunn–Minkowski 不等式是数学中最深远、最有用的不等式之一。</p>
<footer>—— R. J. 加德纳（R. J. Gardner，《Brunn–Minkowski 不等式》，Bull. AMS, 2002）</footer>
</div>

<div class="article-byline">
<p>第二级 · 离散与凸几何 ｜ Schneider《凸体》第6章 ｜ 2026-08-07</p>
</div>

## 为什么从 Minkowski 和开始

前几篇里「加法」只发生在标量上：$nP$ 是放大，$\sum \lambda_i x_i$ 是凸组合。这一篇引入集合之间的加法——**Minkowski 和**——然后把一个朴素的问题摆上台面：两个凸体的「和」体积，与各自的体积有什么关系？答案是一条以对称之美著称的不等式：**Brunn–Minkowski 不等式**。它是整个凸体理论的发动机：下一节的等周不等式是它的推论，概率论里的熵幂不等式、高维几何里的测度集中，甚至大模型训练中「混合」与「加噪」的直觉，都能在它的框架里找到影子。Minkowski 和把「加法」从数推广到集合，而 Brunn–Minkowski 告诉你：**体积对集合加法的 $1/d$ 次根是凹的**。

## 1 Minkowski 和：集合之间的加法

**Minkowski 和（Minkowski sum）**：对集合 $A, B \subseteq \mathbb{R}^d$，定义

$$
A + B = \{ a + b : a \in A,\ b \in B \}, \qquad \lambda A = \{ \lambda a : a \in A \}
$$

两个集合相加，就是把每个 $a$ 与每个 $b$ 逐对相加再收集起来。<span class="marginnote">它不是「并」也不是「差」，而是<strong>点集卷积的支撑形式</strong>：支撑函数恰好可加 $h_{A+B} = h_A + h_B$。这条「加法 ↔ 支撑函数相加」的小规律，是凸分析里支撑函数好用之所在。</span>

几个把直觉养大的例子：

- **两个线段相加**：$[0,1] + [0,1] = [0,2]$，长度的加法。一般地 $[0,a] + [0,b] = [0, a+b]$——**区间加法就是长度加法**。
- **圆盘加圆盘**：半径 $r$ 与 $s$ 的圆盘之和是半径 $r+s$ 的圆盘，$\mathrm{vol}$ 是 $\pi(r+s)^2$，不等于 $\pi r^2 + \pi s^2$——体积不服从加法，这是 Brunn–Minkowski 要管的事。
- **正方形加正方形**：$[0,1]^2 + [0,1]^2 = [0,2]^2$，面积从 1 跳到 4。

**多面体的 Minkowski 和仍是多面体**：两个多面体的和，顶点是「顶点之和」的子集。这让我们可以把 Minkowski 和代数化——下一篇的混合体积与等周不等式都从这里长出来。

**例**：正三角形（顶点 $(1,0),(0,1),(0,0)$）与它自身的 Minkowski 和是边长翻倍的正三角形；正方形 $[0,1]^2 + [0,1]^2 = [0,2]^2$；但「正方形 + 三角形」则长出一个六边形——**两个多面体的和，顶点是两顶点之和中的一部分，具体取哪些由两个多面体的法向扇形决定**。这个「顶点求和但不全取」的现象，是理解混合体积为什么非线性的一把钥匙。

## 2 Brunn–Minkowski 不等式：凹性藏在 1/d 次根里

**定理（Brunn–Minkowski 不等式）**：设 $A, B \subseteq \mathbb{R}^d$ 是非空紧集，则

$$
\mathrm{vol}\left( (1-\lambda) A + \lambda B \right)^{\frac{1}{d}} \ge (1-\lambda)\, \mathrm{vol}(A)^{\frac{1}{d}} + \lambda\, \mathrm{vol}(B)^{\frac{1}{d}}, \qquad 0 \le \lambda \le 1
$$

取 $\lambda = 1/2$ 就得到常用的**无参数形式**：

$$
\mathrm{vol}(A + B)^{\frac{1}{d}} \ge \mathrm{vol}(A)^{\frac{1}{d}} + \mathrm{vol}(B)^{\frac{1}{d}}
$$

**辨析｜易错点：** 不等式里必须有 $1/d$ 次根，否则不成立。取 $d=2$ 的两个小正方形，$\mathrm{vol}(A+B)$ 是「边长之和的平方」，而 $\mathrm{vol}(A) + \mathrm{vol}(B)$ 只是两边长平方之和——前者大得多，方向也对；但若 $A$ 是边长 $10$ 的正方形、$B$ 是边长 $0.1$ 的正方形，$\mathrm{vol}(A+B) = (10.1)^2 = 102.01$，$\mathrm{vol}(A)+\mathrm{vol}(B) = 100 + 0.01 = 100.01$，不加根时仍成立……真正不成立的是取凹性方向。**没有 $1/d$ 次根，体积本身是凸的而非凹的**：$\mathrm{vol}((1-\lambda)A + \lambda B)$ 一般大于等于 $\mathrm{vol}(A)^{1-\lambda}\mathrm{vol}(B)^{\lambda}$（这是由 AM–GM 从加根版本推来的），但它不小于的是「凹组合的体积」这一更紧的命题。取根的操作，把「乘积型」换成「加和型」，正是凹性的来源。<span class="marginnote">一句话：<strong>$1/d$ 次根把体积从「超线性膨胀」拉回「线性可加」</strong>。$d$ 维体积按尺度 $d$ 次齐次，取 $1/d$ 次根后变成 1 次齐次——齐次性对上了，不等式才顺理成章。</span>

**直觉**：把 $A$ 与 $B$ 想象成两个「团」。$A + B$ 至少像把两个团拼起来那么「大」，但拼接中会有重合、有空隙，体积不可能是简单的和——不等式说，虽然体积不是可加的，但 $1/d$ 次根后可加，而且只能大不能小。**等号成立当且仅当 $A$ 与 $B$ 是同心的位似凸体**（相互平移缩放的凸体）——比如圆盘与圆盘、正方形与正方形。

Minkowski 和与 Brunn–Minkowski 在工程里有一个意想不到的分身：**数学形态学（mathematical morphology）**。图像处理里的「膨胀（dilation）」$A \oplus B = \bigcup_{b \in B}(A + b)$ 本质上就是 $A + B$，而「腐蚀」「开」「闭」运算全是 Minkowski 和的变体。Brunn–Minkowski 保证的「和体积不小」，在形态学里翻译成「膨胀后的面积有下界」——**一条几何不等式成为图像处理的底层原理**。<span class="marginnote">形态学的「结构元素」$B$ 就是这里的 $B$；选取不同形状的 $B$（圆盘、线段、十字），得到不同的图像变换。Brunn–Minkowski 给「膨胀后的面积」一个与形状无关的普适下界。</span>

## 3 公式解析：从测度集中看凹性

把 Brunn–Minkowski 换成功能更强的**函数形式（Prékopa–Leindler 不等式）**，凹性的机制会透明起来：

$$
\int_{\mathbb{R}^d} h(x)\,dx \ge \left( \int f(x)\,dx \right)^{1-\lambda} \left( \int g(x)\,dx \right)^{\lambda}, \qquad h(x) := \sup_{y} f(y)^{1-\lambda} g\left( \frac{x-(1-\lambda)y}{\lambda} \right)^{\lambda}
$$

四步拆解：

- **第一步，读懂 $h$ 的定义**：$h$ 在点 $x$ 的值是「把 $x$ 拆成 $(1-\lambda)y + \lambda z$，取 $f(y)^{1-\lambda} g(z)^{\lambda}$ 的上确界」。这是「函数版的 Minkowski 和」——集合版的 $A+B$ 换成函数版的卷积式组合。
- **第二步，集合版本是函数版本的特例**：取 $f = \mathbf{1}_A$、$g = \mathbf{1}_B$（指示函数），则 $h = \mathbf{1}_{(1-\lambda)A + \lambda B}$（忽略边界），Prékopa–Leindler 退化为 $\mathrm{vol}((1-\lambda)A + \lambda B) \ge \mathrm{vol}(A)^{1-\lambda} \mathrm{vol}(B)^{\lambda}$，再结合 AM–GM 即可推出 Brunn–Minkowski。**集合是函数的一层皮**。
- **第三步，看出凹性**：令 $\varphi = -\log f$，不等式变成 $\log\int e^{-\varphi} \ge (1-\lambda)\log\int e^{-\varphi_1} + \lambda \log \log\int e^{-\varphi_2}$ 型的不等式——**测度的 log-凹组合保持积分**。这解释了为什么高斯函数、指数的积分处处与 Brunn–Minkowski 同台。
- **第四步，为何重要**：Prékopa–Leindler 是「测度集中」的引擎——高维球体、高维立方体中「绝大多数体积集中在薄壳」的事实，都是它的一句话推论。**大模型训练里「批量采样」「噪声增广」背后正是这种集中现象在起作用**，而集中现象的起源就是 Brunn–Minkowski 式的凹性。

举一个能摸到的例子：$d$ 维单位立方体的体积是 1，但「去掉外皮 $\varepsilon$」后的核心体积是 $(1 - 2\varepsilon)^d$——当 $d$ 很大时迅速趋近 0。**几乎所有体积都集中在「边界薄壳」里**。这条「体积集中」的结论正是测度集中（concentration of measure）最简单的实例，而它的证明只需要立方体体积公式与 Brunn–Minkowski 的凹性直觉。**高维几何的「看似反直觉」，被 Brunn–Minkowski 变成了可以推导的定理**——这是它作为「发动机」最生动的一击。

**熵幂不等式（entropy power inequality）** 是 Brunn–Minkowski 在信息论里的分身：设 $X, Y$ 是独立连续随机向量，熵幂 $N(X) = \frac{1}{2\pi e} e^{\frac{2}{d} H(X)}$，则

$$
N(X + Y) \ge N(X) + N(Y)
$$

两个独立随机变量的「信息量」相加，不小于各自信息量之和——结构与 Brunn–Minkowski 完全同构，因为「熵」本质上是「测度的体积」，而「独立和的分布」就是「测度的 Minkowski 卷积」。**信道容量、数据压缩、大模型训练里对「加噪」的分析，都能追溯到这一条**。Brunn–Minkowski 的领地，从几何一路长进了信息论。

## 4 等周不等式：Brunn–Minkowski 的头号推论

Brunn–Minkowski 不等式之所以是「发动机」，因为它几行就能推出下一节的等周不等式。设 $K$ 是凸体，记 $K_\varepsilon = K + \varepsilon B_2^d$（$B_2^d$ 是单位球），则 Brunn–Minkowski 给出：

$$
\mathrm{vol}(K_\varepsilon)^{\frac{1}{d}} \ge \mathrm{vol}(K)^{\frac{1}{d}} + \varepsilon\, \mathrm{vol}(B_2^d)^{\frac{1}{d}}
$$

两边对 $\varepsilon$ 展开取一阶项，得表面积下界 $S(K) \ge d\, \mathrm{vol}(B_2^d)^{1/d}\, \mathrm{vol}(K)^{(d-1)/d}$，等号在 $K$ 为球时成立——**这就是等周不等式**：给定体积，球的表面积最小（或者说给定表面积，球体积最大）。<span class="marginnote">这条「Brunn–Minkowski ⟹ 等周」的推导只需要一行展开，却贯穿了凸体理论的整条主线。我们把严格的等周证明留到下一篇，今天先把「发动机」开起来。</span>

**应用一瞥**：熵幂不等式（$N(X+Y) \ge N(X) + N(Y)$，其中 $N$ 是熵幂）是信息论里的 Brunn–Minkowski；Minkowski 和的稳定版本支撑着凸优化的对偶间隙分析；而「体积的 $1/d$ 次根是凹函数」这一句，本身就是凸分析里最常被引用的凹性来源之一——它与第二级《凸分析》中支撑函数、次梯度的体系无缝衔接。

## 5 混合体积：把加法微分化

Brunn–Minkowski 不等式坐镇的是「比较」，而**混合体积（mixed volume）**是它的「微积分」。对凸体 $K_1, \dots, K_d$ 与参数 $t_1, \dots, t_d \ge 0$，体积 $V(t_1 K_1 + \cdots + t_d K_d)$ 是 $t_1, \dots, t_d$ 的 $d$ 次齐次多项式；$t_1 t_2 \cdots t_d$ 项的系数（乘上 $\binom{d}{t_1,\dots,t_d}$ 的倒数）称为混合体积 $V(K_1, \dots, K_d)$。取 $K_1 = \cdots = K_d = K$，它就是 $\mathrm{vol}(K)$。

**关键性质**：混合体积对每个变元都是**多重线性、非负且连续**的。于是 Brunn–Minkowski 蕴含的许多不等式可以微分化——例如对三个凸体 $A, B, C$：

$$
V(A, B, \dots, B) \ge 0, \qquad V(A, A, \dots, B) \ge V(B, B, \dots, B)^{\frac{1}{d}} \cdots
$$

这一族不等式统称 **Minkowski 不等式**，其中把「$d-1$ 个 $A$ 与 1 个 $B$」混合的版本正是上一节等周不等式的来源：表面积 $S(K) = d \cdot V(K, \dots, K, B_2^d)$，把「表面积」看成「一个变元是单位球的混合体积」，等周就变成「混合体积不等式」的特例。<span class="marginnote">混合体积是凸几何的「高阶导数」：体积在「沿 Minkowski 和方向」的方向导数就是混合体积。这也是为什么混合体积理论（Schneider 一书的精华）能与代数几何的相交理论、几何测度论的表面积测度互相翻译。</span>

对读者而言，混合体积的价值在于：它把「两个凸体相加」这种几何操作变成一个**可微的多项式对象**，于是几何问题可以求导、可以极值化——**Brunn–Minkowski 是定性不等式，混合体积是定量微积分**，两者合起来构成凸几何对「加法结构」的完整刻画。

## 6 小结

- **Minkowski 和**：$A + B = \{a+b\}$；支撑函数可加 $h_{A+B} = h_A + h_B$；多面体之和仍是多面体。
- **Brunn–Minkowski 不等式**：$\mathrm{vol}((1-\lambda)A + \lambda B)^{1/d} \ge (1-\lambda)\mathrm{vol}(A)^{1/d} + \lambda \mathrm{vol}(B)^{1/d}$；体积的 $1/d$ 次根是凹函数。
- **$1/d$ 次根**：把体积的 $d$ 次齐次性拉回 1 次齐次，是凹性的来源；去掉根号的体积本身是凸的，方向会反转。
- **等号条件**：同心的位似凸体（圆盘与圆盘、正方形与正方形）达到等号——「平移缩放」是等号的全部自由度。
- **应用版图**：熵幂不等式（信息论）、形态学膨胀（图像处理）、混合体积（凸几何的微积分）——同一条凹性，三片领地。

在下一节，我们把这个不等式的最强推论单独拎出来：**等周不等式**——给定体积，球的表面积最小。它的证明只需把今天的 Brunn–Minkowski 沿「加厚 + 一阶变分」推一步，而它自己又将通向高维的测度集中与离散世界的扩张器图。
