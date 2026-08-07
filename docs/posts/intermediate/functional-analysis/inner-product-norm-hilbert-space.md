---
title: 由内积诱导的范数与 Hilbert 空间
date: 2026-08-07
---

# 由内积诱导的范数与 Hilbert 空间

<div class="epigraph">
<p>Hilbert 空间是唯一一个既有 Banach 空间的完备性、又有欧氏几何的正交性的舞台。</p>
<footer>—— 保罗 · 哈尔莫斯（Paul Halmos），意译</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛函分析 ｜ 程其襄《泛函分析》§4.2 ｜ 2026-08-07</p>
</div>

## 为什么 Hilbert 空间如此特别

前几节我们有了内积空间，内积又诱导出范数。现在把两件最强的装备拼在一起：**完备性（Banach 空间的遗产）+ 正交性（欧氏几何的遗产）**——得到的就是 **Hilbert 空间（Hilbert space）**。它是泛函分析中「待遇最好」的空间：有垂直、有投影、有正交分解、有 Riesz 表示定理，几乎所有几何直觉都在这里完好无损。量子力学用 $L^2$ 作态空间，信号处理用 $L^2$ 作能量空间，最小二乘用 Hilbert 空间的投影——它的应用几乎无处不在。<span class="marginnote">Hilbert 空间的名字来自大卫 · 希尔伯特：他在 1912 年前后研究积分方程时引入 $l^2$，后人把「完备内积空间」命名为 Hilbert 空间。注意<strong>不是所有 Banach 空间都是 Hilbert 空间</strong>——只有那些范数「来自内积」的才是，判别法在下一节。</span>

## 1 范数与内积的兼容：极化恒等式

内积诱导范数 $\|x\| = \sqrt{\langle x, x\rangle}$。反过来的问题：给定范数，能否找回内积？在复空间用**极化恒等式（polarization identity）**：

$$
\langle x, y\rangle = \frac{1}{4}\Big( \|x + y\|^2 - \|x - y\|^2 + i\|x + iy\|^2 - i\|x - iy\|^2 \Big)
$$

（实空间退化为 $\langle x,y\rangle = \frac14(\|x+y\|^2 - \|x-y\|^2)$。）极化恒等式说明：**内积完全由范数决定**——范数里藏着内积的全部信息。<span class="marginnote">「极化」这个名字来自物理：给定范数（模长），极化恒等式把「夹角」的信息也提取出来。它也是「平行四边形公式」等价性的另一半：范数能满足极化恒等式 ⟺ 范数来自内积（下一节详述）。</span>

**核心要点：范数与内积互为表里**——内积给出范数，极化恒等式从范数找回内积。但前提是范数满足「平行四边形公式」，否则找回来的「内积」不满足公理。

## 2 Hilbert 空间的定义

**定义**：完备的内积空间称为 **Hilbert 空间**。即：$H$ 是内积空间，且由内积诱导的范数使 $H$ 成为 Banach 空间（每个柯西列收敛）。

**关键例子**：

- **$\mathbb{C}^n$**：有限维 Hilbert 空间。
- **$l^2$**：平方可和数列，Hilbert 空间（完备性在第二章已证）。
- **$L^2[a,b]$**：平方可积函数，Hilbert 空间——这是应用最广的无穷维 Hilbert 空间。<span class="marginnote">$L^2$ 之所以是 Hilbert 空间，依赖勒贝格积分的 Riesz-Fischer 定理（$L^p$ 完备）。信号处理里「能量有限信号」的集合就是 $L^2$：信号的能量 $\int|f|^2$ 正是范数平方。傅里叶变换、小波、滤波器设计，都在这个空间里展开。</span>

**辨析｜易错点：** Hilbert 空间是 Banach 空间的一种，但反过来不对。$C[0,1]$（sup 范数）、$l^p$（$p \neq 2$）、$L^p$（$p \neq 2$）都是 Banach 空间但不是 Hilbert 空间——它们的范数不来自任何内积。**「范数来自内积」是 Hilbert 空间区别于一般 Banach 空间的关键属性**，下一节给出精确判别。

## 3 内积的连续性

内积作为「双变量函数」是连续的：若 $x_n \to x$、$y_n \to y$（范数收敛），则

$$
\langle x_n, y_n\rangle \to \langle x, y\rangle
$$

证明只需一步估计：

$$
|\langle x_n,y_n\rangle - \langle x,y\rangle| \le |\langle x_n - x, y_n\rangle| + |\langle x, y_n - y\rangle| \le \|x_n - x\|\|y_n\| + \|x\|\|y_n - y\| \to 0
$$

（中间用到 Cauchy-Schwarz）。<span class="marginnote">内积连续听起来平凡，却是「弱收敛理论」（第七章）的起点：弱收敛 $x_n \rightharpoonup x$ 定义为「对一切 $y$ 有 $\langle x_n, y\rangle \to \langle x, y\rangle$」。内积连续保证「强收敛 ⟹ 弱收敛」，而反过来一般不成立。</span>

**核心要点：内积是连续的二元映射**——它是「第一变量线性 + 第二变量共轭线性 + 范数控制」三者的综合体现。

## 4 例子：L^2 是 Hilbert 空间（结构验证）

以 $L^2[a,b]$ 为例，完整走一遍「它是 Hilbert 空间」的验证：

- **内积结构**：$\langle f,g\rangle = \int f\overline g$，三条公理由积分性质保证。
- **范数**：$\|f\|_2 = \sqrt{\int|f|^2}$，Cauchy-Schwarz 保证内积收敛。
- **完备性**：Riesz-Fischer 定理——$L^p$ 的柯西列在 $L^p$ 内收敛（勒贝格积分的成果）。

于是 $L^2$ 具备：垂直（$\int f\overline g = 0$）、正交分解、投影、傅里叶级数收敛（第四章后文）——**欧氏几何的全部工具在 $L^2$ 里重生**。<span class="marginnote">量子力学把它用到极致：波函数 $\psi \in L^2(\mathbb{R}^3)$，$\int|\psi|^2 = 1$ 是归一化条件，「测量」「观测」「坍缩」都是 $L^2$ 上的内积/投影操作。第十章我们会专门用 Hilbert 空间语言重写量子力学。</span>

## 5 公式解析：内积连续性证明

把「为什么 $x_n \to x$ 且 $y_n \to y$ 推出内积收敛」拆成三步：

$$
\langle x_n, y_n\rangle - \langle x, y\rangle = \underbrace{\langle x_n - x, y_n\rangle}_{(A)} + \underbrace{\langle x, y_n - y\rangle}_{(B)}
$$

- **第一步，拆分（加零项）**：$\langle x_n,y_n\rangle - \langle x,y\rangle = \langle x_n,y_n\rangle - \langle x,y_n\rangle + \langle x,y_n\rangle - \langle x,y\rangle = \langle x_n - x, y_n\rangle + \langle x, y_n - y\rangle$。这是「加一项再减一项」的经典技巧。
- **第二步，分别估计**：用 Cauchy-Schwarz，$|(A)| \le \|x_n - x\|\|y_n\|$、$|(B)| \le \|x\|\|y_n - y\|$。
- **第三步，取极限**：$\|x_n - x\| \to 0$、$\|y_n - y\| \to 0$；而 $\{y_n\}$ 收敛故有界 $\|y_n\| \le M$。于是 $|(A)| + |(B)| \to 0$。

**关键**：证明用到了「收敛列有界」这个分析基本功 + Cauchy-Schwarz 把内积化为范数。**内积连续性的实质是「内积被范数控制，范数收敛则内积收敛」**。

## 6 例题精讲：Hilbert 空间的三个典型验证

**例题一：$L^2[0,1]$ 是 Hilbert 空间**。

- 内积：$\langle f, g\rangle = \int_0^1 f\bar g$，三条公理由积分性质保证。
- 范数：$\|f\|_2 = \sqrt{\int|f|^2}$，Cauchy-Schwarz 保证内积收敛。
- 完备性：Riesz-Fischer 定理（$L^p$ 完备），故 $L^2$ 是 Hilbert 空间。

**例题二：$l^2$ 与 $L^2$ 的酉等价**。

- 取 $L^2[0,1]$ 的三角正交基 $e_n(t) = e^{2\pi int}$。
- 坐标映射 $f \mapsto (\langle f, e_n\rangle)$ 是等距：帕塞瓦尔 $\sum|\hat f_n|^2 = \|f\|^2$。
- 由 §4.9，$L^2$ 与 $l^2$ 酉同构——两个空间的几何完全相同。

**例题三：$C[0,1]$ 为什么不是 Hilbert 空间**。

- sup 范数 $\|f\|_\infty = \max|f|$ 不满足平行四边形公式。
- 取 $f \equiv 1$、$g = t$：$\|f+g\|_\infty^2 + \|f-g\|_\infty^2 = 4 + 1 = 5$，
- 而 $2\|f\|_\infty^2 + 2\|g\|_\infty^2 = 2 + 2 = 4$，不等。
- 故 $C[0,1]$（sup 范数）不是内积空间，更不是 Hilbert 空间。

**核心要点**：三个例题覆盖「是」「酉等价」「不是」三种情形——判断 Hilbert 空间的关键是平行四边形公式与完备性。

**辨析｜易错点：** $C[0,1]$ 配 $L^2$ 范数也不完备（$L^2$-极限可以是阶梯函数），所以「$C[0,1]$ 不是 Hilbert」的说法要说明用哪个范数。


## 7 小结

- **内积诱导范数**：$\|x\| = \sqrt{\langle x,x\rangle}$；**极化恒等式**从范数找回内积。
- **Hilbert 空间**：完备的内积空间 = Banach + 正交几何；$l^2$、$L^2$ 是代表。
- **不全是 Hilbert**：$C[0,1]$、$l^p$（$p\neq2$）是 Banach 而非 Hilbert。
- **内积连续**：$x_n\to x$、$y_n\to y$ 推出 $\langle x_n,y_n\rangle\to\langle x,y\rangle$，靠 Cauchy-Schwarz + 收敛列有界。
- **应用**：$L^2$ 承载信号处理、量子力学、傅里叶理论——「平方可积」的空间是自然的 Hilbert 舞台。

在下一节，我们回答「什么范数来自内积」——**平行四边形公式与内积空间的范数刻画**，给出 Hilbert 空间的精确判别法。
