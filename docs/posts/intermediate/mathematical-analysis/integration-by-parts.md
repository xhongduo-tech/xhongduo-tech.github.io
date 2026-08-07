---
title: 分部积分法
date: 2026-08-07
---

# 分部积分法

<div class="epigraph">
<p>乘积求导法则的逆，是积分学最灵活的武器之一——它把「两种函数相乘」的积分，转嫁给「其中一种求导、另一种积分」的姊妹积分。</p>
<footer>—— 布鲁克·泰勒（Brook Taylor），对积分方法的系统化（节意）</footer>
</div>

<div class="article-byline">
<p>第二级 · 数学分析 ｜ 华东师大《数学分析》§8.3 ｜ 2026-08-07</p>
</div>

## 为什么需要第三种方法

换元法处理「复合结构」与「根式」，但对**两类不同类型函数相乘**的积分——$x\sin x$、$x\ln x$、$e^x\cos x$——换元往往无从下手。这些积分的共性：被积函数是「乘积」，且两个因子**谁也凑不成谁的导数**。

**分部积分法**来自乘积求导法则的逆运算，它把「难积的乘积」转移到「另一个乘积」，选对了方向就能让积分越算越简单。它是积分学的「杠杆」：**用求导换积分，赌「求导会让某因子变简单」。**<span class="marginnote">分部积分的直觉像一个「交易」：把 $\int u\,dv$ 换成 $uv-\int v\,du$。赚不赚，取决于 $\int v\,du$ 是否比 $\int u\,dv$ 容易。选 $u$ 的原则是「求导后变简单」（多项式求导降幂、$\ln$ 求导变分式），选 $dv$ 的原则是「积分后不变复杂」（$e^x$、$\sin x$ 积分后还是同类）。这个「选谁当 $u$」的判断，是分部积分的全部艺术。</span>

## 1 分部积分公式

**定理（分部积分法 / Integration by Parts）：设 $u,v$ 都可导且 $uv'$、$u'v$ 都可积，则**

$$\int u(x)\,dv=u(x)v(x)-\int v(x)\,du,$$

其中 $dv=v'(x)dx$，$du=u'(x)dx$。展开写法：

$$\int u(x)v'(x)\,dx=u(x)v(x)-\int v(x)u'(x)\,dx.$$

**公式解析：分部积分公式从哪来**

从乘积求导法则倒着走。$(uv)'=u'v+uv'$ 两边积分：

$$\int(uv)'\,dx=uv=\int u'v\,dx+\int uv'\,dx,$$

移项即得 $\int uv'\,dx=uv-\int vu'\,dx$。

**三步拆解**：
- **第一步，选 $u$ 与 $dv$**：把被积函数拆成 $u$（要**求导**的部分）与 $dv$（要**积分**的部分）；
- **第二步，算 $du$ 与 $v$**：$du=u'dx$，$v=\int dv$；
- **第三步，代公式**：$\int u\,dv=uv-\int v\,du$，算新积分。

**示范一**：$\displaystyle\int x\sin x\,dx$。选 $u=x$（求导变 1）、$dv=\sin x\,dx$（积分得 $-\cos x$）：

$$\int x\sin x\,dx=-x\cos x+\int\cos x\,dx=-x\cos x+\sin x+C.$$

多项式因子被「求导降幂」吃掉，积分完成。**若反着选**（$u=\sin x$、$dv=x\,dx$），会得到 $\frac{x^2}{2}\sin x-\int\frac{x^2}{2}\cos x\,dx$——**指数不降反升，越积越糟**。选对方向是命门。

**示范二**：$\displaystyle\int\ln x\,dx$。被积函数只有一个函数？**凑一个 1 进去**：$u=\ln x$（求导变 $\frac1x$）、$dv=dx$（积分得 $x$）：

$$\int\ln x\,dx=x\ln x-\int x\cdot\frac1x\,dx=x\ln x-x+C.$$

**「只有一个函数时，把它看成该函数 × 1」**——这是分部积分的经典开场。

## 2 选择 u 与 dv 的优先序

选 $u$ 的优先序可以用「**LIATE 原则**」记忆：

**L**ogarithm（对数）→ **I**nverse trig（反三角）→ **A**lgebraic（代数/多项式）→ **T**rigonometric（三角）→ **E**xponential（指数）

优先序靠前的当 $u$（优先**求导**），靠后的当 $dv$（优先**积分**）。理由是：

| 因子类型 | 求导后 | 积分后 |
| --- | --- | --- |
| 对数 $\ln x$ | 变分式 $\frac1x$（更简单） | 更复杂 |
| 反三角 $\arctan x$ | 变分式 $\frac1{1+x^2}$（更简单） | 更复杂 |
| 多项式 $x^n$ | 降幂 | 升幂 |
| 三角 $\sin,\cos$ | 同类 | 同类 |
| 指数 $e^x$ | 同类 | 同类 |

**谁求导后变简单，谁当 $u$；谁积分后不变糟，谁当 $dv$。** LIATE 正是这两条原则的总结——越「难积」的类型越该当 $u$ 去求导，越「好积」的类型越该当 $dv$ 去积分。<span class="marginnote">LIATE 是记忆口诀，不是定理——遇到「多项式 × 指数」按 LIATE 应选多项式当 $u$，这几乎总是对的。但偶尔有反例（如「多项式 × $\frac1x$」选多项式当 $u$ 会降成常数反而好）。判断的第一原则永远是「求导后是否变简单」，LIATE 只是它的快捷方式。</span>

**示范三（对数当 $u$）**：$\displaystyle\int x^2\ln x\,dx$。按 LIATE，$u=\ln x$、$dv=x^2dx$：

$$\int x^2\ln x\,dx=\frac{x^3}{3}\ln x-\int\frac{x^3}{3}\cdot\frac1x\,dx=\frac{x^3}{3}\ln x-\frac{x^3}{9}+C.$$

## 3 分部积分的进阶套路

**套路一：循环相消。** $\displaystyle\int e^x\sin x\,dx$。两类函数都「积分后不变糟」，无论选哪个当 $u$，二次分部后会**回到自身**：

设 $I=\int e^x\sin x\,dx$。取 $u=e^x$、$dv=\sin x\,dx$：$I=e^x(-\cos x)+\int e^x\cos x\,dx$。再分部一次：$\int e^x\cos x\,dx=e^x\sin x-\int e^x\sin x\,dx=e^x\sin x-I$。代入得

$$I=-e^x\cos x+e^x\sin x-I\quad\Longrightarrow\quad I=\frac{e^x(\sin x-\cos x)}{2}+C.$$

**要点：出现 $I$ 的方程，解方程**——「自身项搬家、除以 2」的循环消去法是「同类函数相乘」的标准解法。

**套路二：降幂递推。** $\displaystyle\int\sin^n x\,dx$（$n\ge2$）。取 $u=\sin^{n-1}x$、$dv=\sin x\,dx$：

$$\int\sin^n x\,dx=-\sin^{n-1}x\cos x+(n-1)\int\sin^{n-2}x\cos^2x\,dx$$

$$=-\sin^{n-1}x\cos x+(n-1)\int\sin^{n-2}x(1-\sin^2x)\,dx,$$

整理出递推式：

$$\int\sin^n x\,dx=-\frac{\sin^{n-1}x\cos x}{n}+\frac{n-1}{n}\int\sin^{n-2}x\,dx.$$

**每个高阶积分降到低两阶**——从 $\int\sin^2x\,dx=\frac x2-\frac{\sin2x}{4}$ 出发，逐层上推。这个「降幂递推」模板对 $\cos^n,\tan^n,e^x x^n$ 等一打积分都适用。

> **辨析｜易错点：**分部积分最经典的错误是**选错 $u$**：把多项式当 $dv$ 会升幂，把 $\ln x$ 当 $dv$ 更是灾难（$\ln x$ 的原函数要再分部一次）。**判断标准不是「顺眼」，而是「算完 $\int v\,du$ 是否更简单」**。另一个陷阱是**忘记换元/分部的组合**——很多题要先换元再分部（如 $\int e^{\sqrt x}dx$ 先令 $t=\sqrt x$ 再分部），一条路走不通就换组合。

## 4 分部积分的地位

分部积分是积分学的第三块基石（直接积分、换元、分部），它的应用远超初等积分：

- **递推公式的源头**：$\int x^ne^x\,dx$、$\int\ln^nx\,dx$、$\int\sin^nx\,dx$ 的递推式全靠分部；
- **定积分版本**：$\int_a^b u\,dv=[uv]_a^b-\int_a^b v\,du$，带上下限后可直接算（§9.6 详论）；
- **微积分基本定理的证明**：$\int_a^b f'(x)dx=f(b)-f(a)$ 本质是「分部积分的极限形式」；
- **概率论与统计**：正态分布 $\int x^2e^{-x^2/2}dx$ 的方差计算，靠分部；Gamma 函数 $\Gamma(\alpha+1)=\alpha\Gamma(\alpha)$ 的递推，靠分部（§19.4）。<span class="marginnote">Gamma 函数 $\Gamma(\alpha)=\int_0^\infty t^{\alpha-1}e^{-t}dt$ 是分部积分的「长期客户」：一次分部给出 $\Gamma(\alpha+1)=\alpha\Gamma(\alpha)$，从 $\Gamma(1)=1$ 推出 $\Gamma(n)=(n-1)!$——<strong>阶乘函数从离散延伸到连续</strong>，靠的就是分部积分这个递推引擎。第十九章我们会系统见面。</span>

## 5 小结

- **公式**：$\int u\,dv=uv-\int v\,du$；来源是乘积求导法则的逆。
- **选 $u$ 原则**：LIATE 优先序——对数/反三角当 $u$ 求导、指数/三角当 $dv$ 积分；核心是「求导后变简单」。
- **经典开场**：只有一个函数时把它看成「该函数 × 1」（$\int\ln x\,dx=x\ln x-x+C$）。
- **进阶套路**：循环相消（$e^x\sin x$ 解方程）、降幂递推（$\sin^nx$ 每步降两阶）。
- **应用**：递推公式、定积分、Gamma 函数、正态分布——分部积分贯穿整个积分学。

在下一节，我们处理积分计算的「收官专题」：**有理函数的不定积分与可化为有理函数的积分**。部分分式分解 + 四类最简分式的积分，是「任何有理函数都能手算积分」的总保证——它也预告了「初等函数积分不一定能写成初等函数」的边界。
