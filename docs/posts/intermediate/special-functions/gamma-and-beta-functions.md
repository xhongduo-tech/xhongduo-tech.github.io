---
title: Gamma 函数与 Beta 函数
date: 2026-08-07
---

# Gamma 函数与 Beta 函数

<div class="epigraph">
<p>上帝也许是一个不体面的数学家，但他肯定是一个数学的创造者。</p>
<footer>—— 卡洛斯 · 费德里科 · 高斯（Carl Friedrich Gauss）</footer>
</div>

<div class="article-byline">
<p>第二级 · 特殊函数 ｜ 王竹溪、郭敦仁《特殊函数概论》 第 1 章 ｜ 2026-08-07</p>
</div>

## 为什么从 Gamma 函数开始

阶乘 $n!$ 只对正整数有意义。若要问「$1/2$ 的阶乘是多少」，初等数学给不出答案——可偏偏在概率统计、量子力学、复变函数与计算数学里，到处都需要一个能把阶乘「连成一条光滑曲线」的推广。**Gamma 函数（Gamma function）正是把阶乘从离散点集延伸到整个复平面的解析函数**，它是全部特殊函数世界的入口：超几何函数、Bessel 函数、Legendre 函数几乎都以 Gamma 函数为原料，Beta 函数则是它的最亲密的姊妹。<span class="marginnote">「特殊函数」这个学科的名字来自这样一个事实：它们不像 $\sin x$、$e^x$ 那样是初等函数，却反复出现在微分方程、积分与物理问题里，以至于值得被逐一起名、研究。王竹溪、郭敦仁《特殊函数概论》开篇即从 Gamma 函数讲起，正是看中它地位的基础性。</span>学本专题，Gamma 函数是第一块必须踩实的基石。

## 1 从阶乘到 Gamma：一个积分式的诞生

阶乘的经典定义是 $n! = 1 \cdot 2 \cdots n$。数学家想把它推广，关键在于找到一种**与阶乘同样满足递推关系、却能连续变化**的对象。

考察积分

$$
\Gamma(z) = \int_{0}^{+\infty} t^{z-1} e^{-t}\, dt, \qquad \operatorname{Re} z > 0
$$

这个积分在 $\operatorname{Re} z > 0$ 时收敛（$t\to 0$ 处靠 $t^{z-1}$ 可积，$t\to+\infty$ 处靠 $e^{-t}$ 衰减），称为**欧拉第二类积分（Euler integral of the second kind）**，它所定义的函数就叫 Gamma 函数。<span class="marginnote">为什么积分里是 $t^{z-1}$ 而不是 $t^z$？这纯粹是历史约定：欧拉在 1729 年写给哥德巴赫的信里给出的积分原形如此，而 $t^{z-1}$ 的选择让 $\Gamma(1)=1$ 成立，恰好接上 $0! = 1$。</span>

做一次分部积分立刻看到它的身世：

$$
\Gamma(z+1) = \int_0^{+\infty} t^{z} e^{-t}\,dt
= \left[-t^{z}e^{-t}\right]_0^{+\infty} + z\int_0^{+\infty} t^{z-1}e^{-t}\,dt
= z\,\Gamma(z)
$$

也就是说 **Gamma 函数满足递推关系 $\Gamma(z+1) = z\,\Gamma(z)$**，与阶乘的 $n! = n \cdot (n-1)!$ 逐字对应。再叠加 $\Gamma(1) = 1$，对正整数就有

$$
\Gamma(n+1) = n!, \qquad n = 0, 1, 2, \dots
$$

**Gamma 函数就是阶乘的解析延拓**——这是理解它一切性质的第一句话。而「递推关系 + 端点值」这种定义函数的方式，也预告了后面几乎每一个特殊函数家族（Bessel、Legendre、正交多项式）都会有自己的递推关系。

## 2 复平面上的延伸：Weierstrass 乘积与解析延拓

欧拉积分只在右半平面 $\operatorname{Re} z > 0$ 有意义。但递推关系 $\Gamma(z+1) = z\,\Gamma(z)$ 可以反过来用：在 $\operatorname{Re} z > 0$ 上有了 $\Gamma(z)$，就令

$$
\Gamma(z) = \frac{\Gamma(z+1)}{z}
$$

把定义向左推到 $-1 \lt  \operatorname{Re} z \le 0$（除 $z=0$ 外），再继续推到 $-2 \lt  \operatorname{Re} z \le -1$……如此逐条向左，就得到**整个复平面（除非正整数点外）上的 Gamma 函数**。这个逐片延拓的结果是唯一的，且与复分析里「解析延拓」的唯一性定理一致。<span class="marginnote">解析延拓（analytic continuation）是复变函数论的王牌工具：一个区域上解析的函数，若在某个小片带上与另一函数一致，则全区域上被唯一决定。Gamma 函数是解析延拓最经典的教学案例。相关严格框架见第二级《复变函数与积分变换》。</span>

Gamma 函数在整个复平面上的严格描述由 **Weierstrass 乘积**给出，它把 $\Gamma(z)$ 的极点位置直接写了出来：

$$
\frac{1}{\Gamma(z)} = z\, e^{\gamma z} \prod_{n=1}^{\infty} \left(1 + \frac{z}{n}\right) e^{-z/n}
$$

其中 $\gamma \approx 0.57721$ 是**欧拉-马歇罗尼常数（Euler–Mascheroni constant）**。注意等号左边是 $1/\Gamma(z)$——**$1/\Gamma$ 才是整函数，Gamma 本身在所有非正整数处有一阶极点**。<span class="marginnote">把 $1/\Gamma(z)$ 写成全平面的解析函数（整函数），是 Gamma 理论里一个常被低估的转折：它让「取对数」「看零点」等操作都有了干净的对象。这与复分析中「亚纯函数的倒数变整函数」的一般原理一脉相承。</span>对 $z$ 取倒数即得

$$
\Gamma(z) = \frac{1}{z}\, e^{-\gamma z} \prod_{n=1}^{\infty} \frac{e^{z/n}}{1 + z/n}
$$

它在 $z = 0, -1, -2, \dots$ 处有一阶极点，留数分别是 $(-1)^n / n!$。

## 3 三条金律：反射公式、加倍公式与 $\Gamma(1/2)$

Gamma 函数有三条在计算里天天用到的关系式，值得单独记忆。

**反射公式（Euler 反射公式）** 沟通了 $z$ 与 $1-z$：

$$
\Gamma(z)\,\Gamma(1-z) = \frac{\pi}{\sin \pi z}
$$

这条公式从 Weierstrass 乘积可证，也可以由正弦函数的乘积展开导出。它立刻给出最重要的特殊值：取 $z = 1/2$，

$$
\Gamma\left(\frac{1}{2}\right) = \sqrt{\pi}
$$

**加倍公式（Legendre 加倍公式 / 倍乘公式）** 把 $\Gamma(z)$ 与 $\Gamma(2z)$ 连起来：

$$
\Gamma(z)\, \Gamma\!\left(z + \frac12\right) = 2^{1-2z} \sqrt{\pi}\; \Gamma(2z)
$$

这是一般「乘性公式」（Gauss 乘法公式）在 $m=2$ 的特例。它在计算半整数阶 Gamma、以及统计力学中 Gamma 与 Zeta 函数纠缠的积分时极其有用。

**对数微商与 Digamma 函数**：对 $\log \Gamma(z)$ 求导得到 $\psi(z) = \Gamma'(z)/\Gamma(z)$，称为 **Digamma 函数（双 Gamma 函数）**。它满足 $\psi(z+1) = \psi(z) + 1/z$，并可通过 Euler 常数表示成

$$
\psi(z) = -\gamma - \frac{1}{z} + \sum_{n=1}^{\infty}\left(\frac{1}{n} - \frac{1}{z+n}\right)
$$

Digamma 函数是 Gamma 家族衍生出的一系列函数（polygamma $\psi^{(m)}$）的第一员，在后面渐近展开那一篇里会反复出场。

## 4 公式解析：为什么 $\Gamma(1/2) = \sqrt{\pi}$

**「$1/2$ 的阶乘等于 $\sqrt{\pi}$」大概是特殊函数里最让人惊讶的一行等式。** 把 $\pi$ 和阶乘联系起来，靠的是换元与高斯积分。

- **第一步，代入定义**：在欧拉积分里取 $z = 1/2$，得

$$
\Gamma\left(\frac12\right) = \int_0^{+\infty} t^{-1/2} e^{-t}\, dt
$$

- **第二步，换元去根号**：令 $t = u^2$，则 $dt = 2u\, du$、$t^{-1/2} = u^{-1}$，于是

$$
\Gamma\left(\frac12\right) = \int_0^{+\infty} u^{-1} e^{-u^2} \cdot 2u\, du = 2\int_0^{+\infty} e^{-u^2}\, du
$$

- **第三步，认出高斯积分**：$I = \int_0^{+\infty} e^{-u^2} du$ 是半个高斯积分。利用对称性 $\int_{-\infty}^{+\infty} e^{-u^2}du = \sqrt{\pi}$，故 $I = \sqrt{\pi}/2$。

- **第四步，合并**：

$$
\Gamma\left(\frac12\right) = 2 \cdot \frac{\sqrt{\pi}}{2} = \sqrt{\pi}
$$

这四步里真正起作用的只有一件事：**换元把 Gamma 的积分翻译成了高斯积分的对称形式**。同样的技巧可以推广——计算 $\int_0^{+\infty} x^{s-1} e^{-x} dx$ 之外的一切带幂函数乘指数函数的积分，几乎都绕不开 Gamma 函数。

## 5 Beta 函数：Gamma 的姊妹

**Beta 函数（Beta function）** 由欧拉第一类积分定义：

$$
B(p, q) = \int_0^1 t^{\,p-1} (1-t)^{\,q-1}\, dt, \qquad \operatorname{Re} p > 0,\ \operatorname{Re} q > 0
$$

它与 Gamma 函数之间有一条漂亮的桥梁公式：

$$
B(p, q) = \frac{\Gamma(p)\,\Gamma(q)}{\Gamma(p+q)}
$$

这条公式可由「Gamma 积分乘积 + 极坐标换元」证明：把 $\Gamma(p)\Gamma(q)$ 写成二维积分，再用 $t$ 与 $s$ 的比值替换其中一个变量，就能把两个单变量积分压缩成一个 Beta 积分。<span class="marginnote">证明的骨架是：$\Gamma(p)\Gamma(q) = \int_0^\infty\int_0^\infty x^{p-1}y^{q-1}e^{-(x+y)}dx\,dy$，令 $x = st, y = s(1-t)$ 后雅可比行列式为 $s$，于是因子分解成 $\int_0^\infty s^{p+q-1}e^{-s}ds \cdot \int_0^1 t^{p-1}(1-t)^{q-1}dt$。这个「先乘积后换元」的套路在特殊函数里反复出现。</span>

Beta 函数的好处是把积分区间压到 $[0,1]$，且自带 $(1-t)^{q-1}$ 的权重。统计里的 Beta 分布、组合数学里的二项积分、物理里的若干归一化常数，都以它为背景。**用 Gamma 表示 Beta（或反之）是计算的第一步**：任何一个形如 $\int_0^1 x^{a}(1-x)^{b}dx$ 的积分，答案直接就是 $B(a+1, b+1) = \Gamma(a+1)\Gamma(b+1)/\Gamma(a+b+2)$。

## 6 Gamma 函数在统计、物理与计算中的出场

Gamma 函数绝不是教材里的孤例，它遍布整个数理谱系：

**概率统计**：Gamma 分布与卡方分布都以 Gamma 函数做归一化常数；Gamma 分布的矩、卡方分布的期望，全靠 $\Gamma$ 的递推关系推得。<span class="marginnote">卡方分布 $\chi^2_k$ 的密度含因子 $x^{k/2 - 1}e^{-x/2}$，归一化常数是 $2^{k/2}\Gamma(k/2)$。$k=2$ 时归一到指数分布——这正对应 $\Gamma(1)=1$。详见第二级《概率论与数理统计》。</span>
- **量子力学与统计物理**：$\int_0^\infty \frac{x^{s-1}}{e^x - 1}dx = \Gamma(s)\,\zeta(s)$ 把 Gamma 与黎曼 Zeta 函数连接起来，是普朗克黑体辐射、玻色-爱因斯坦凝聚积分的基本原料。
- **分数阶微积分**：分数阶导数与积分算子的定义直接建立在 $\Gamma$ 之上，例如 Riemann–Liouville 分数积分 $I^\alpha f$ 的核含 $1/\Gamma(\alpha)$。
- **数值计算**：计算 $\log\Gamma(z)$（Stirling 级数）是科学计算库的基础函数；Gamma 的连分数与渐近展开是数值逼近的经典案例，我们在《渐近展开与最速下降法》一篇中还会回来。

**辨析｜易错点：** 初学者常把「Gamma 函数在 $z=0$ 处发散」误读为「$\Gamma(0) = 1$」。实际上 $\Gamma(0)$ 是一阶极点，$1/\Gamma(0) = 0$；$\Gamma(n+1) = n!$ 只对非负整数 $n$ 成立。另一个常见误区是把 $\Gamma(1/2) = \sqrt{\pi}$ 当成「$\pi$ 与半阶乘的偶然巧合」——它其实是高斯积分与换元的必然产物，上节的四步拆解给出了全部理由。

## 7 小结

- **Gamma 函数** $\Gamma(z) = \int_0^\infty t^{z-1}e^{-t}dt$ 把阶乘延拓到全复平面（$\operatorname{Re} z > 0$ 用积分，向左用解析延拓）。
- **递推关系** $\Gamma(z+1) = z\,\Gamma(z)$ 与 $\Gamma(1)=1$ 给出 $\Gamma(n+1) = n!$，它是理解一切后续性质的引擎。
- **Weierstrass 乘积** $\dfrac{1}{\Gamma(z)} = z e^{\gamma z}\prod_{n=1}^\infty (1 + z/n) e^{-z/n}$ 说明 $1/\Gamma$ 是整函数，Gamma 在非正整数处有一阶极点。
- **反射公式** $\Gamma(z)\Gamma(1-z) = \pi/\sin\pi z$ 与 **加倍公式** $\Gamma(z)\Gamma(z+1/2) = 2^{1-2z}\sqrt{\pi}\,\Gamma(2z)$ 是两条计算金律；特别地 $\Gamma(1/2) = \sqrt{\pi}$。
- **Beta 函数** $B(p,q) = \int_0^1 t^{p-1}(1-t)^{q-1}dt = \Gamma(p)\Gamma(q)/\Gamma(p+q)$ 是 Gamma 的压缩到 $[0,1]$