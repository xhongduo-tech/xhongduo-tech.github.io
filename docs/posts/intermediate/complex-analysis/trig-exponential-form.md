---
title: 复数的三角表示与指数表示
date: 2026-08-07
---

# 复数的三角表示与指数表示

<div class="epigraph">
<p>欧拉计算起来毫不费力，就像人呼吸，或鹰搏击长空一样。</p>
<footer>—— 弗朗索瓦 · 阿拉戈（François Arago），悼欧拉</footer>
</div>

<div class="article-byline">
<p>第二级 · 复变函数与积分变换 ｜ 西交《复变函数》§1.2 ｜ 2026-08-07</p>
</div>

## 为什么从三角表示与指数表示开始

上一节我们学会用「长度 + 角度」描述复数：模 $r$ 是长度，辐角 $\theta$ 是方向。但写起来还是老样子 $z = a + bi$，长度和角度藏在坐标里，要用 $\sqrt{a^2+b^2}$ 和 $\arctan$ 去「挖」出来。这一节我们换一个姿势：**直接用长度和角度把复数写出来**，得到三角表示 $z = r(\cos\theta + i\sin\theta)$。

这还不算什么。真正改变世界的是欧拉公式 $e^{i\theta} = \cos\theta + i\sin\theta$——它把指数函数与三角函数缝在一起，让复数可以写成 $z = re^{i\theta}$。从这一刻起，**乘除变成旋转与缩放**，三角公式变成指数运算，傅里叶变换、信号处理、大模型的位置编码都站在了这个公式的肩膀上。

## 1 三角表示：用长度与角度重新编码复数

设 $z = a + bi$ 在复平面上对应的点到原点距离为 $r$，与实轴正方向夹角为 $\theta$。由直角坐标与极坐标的换算，$a = r\cos\theta$，$b = r\sin\theta$，代入得

**核心概念：复数的三角表示（trigonometric form）**：

$$z = r(\cos\theta + i\sin\theta)$$

其中 $r = |z| \ge 0$，$\theta = \arg z$ 取主辐角或任意一支都可以（相差 $2\pi$ 时同一式子）。把 $a+bi$ 换成 $r(\cos\theta+i\sin\theta)$，相当于把复数的「身份证」从 $(a, b)$ 换成了 $(r, \theta)$——同一个点，两套坐标。<span class="marginnote">这与第一级《微积分》里的<strong>极坐标</strong>完全同源：平面上的点既可以用 $(x, y)$ 写，也可以用 $(r, \theta)$ 写。复数的三角表示就是「点的极坐标」，只不过把数本身写了出来。</span>

已知 $z = a + bi$ 求三角式，关键是求 $r$ 和 $\theta$：

$$r = \sqrt{a^2 + b^2}, \qquad \theta = \arg z$$

**辨析｜易错点：求辐角时不能只写 $\theta = \arctan\frac{b}{a}$。** 反三角函数只给出 $(-\frac{\pi}{2}, \frac{\pi}{2})$ 内的角，而辐角需要根据点所在的**象限**修正。例如 $z = -1 + i$：$r = \sqrt{2}$，且点在第二象限，所以 $\theta = \frac{3\pi}{4}$，而不是 $\arctan(-1) = -\frac{\pi}{4}$。两个角相差 $\pi$，画出来完全相反。

**例题：把 $z = -\sqrt{3} + i$ 写成三角式。** 先求模 $r = \sqrt{(-\sqrt{3})^2 + 1^2} = 2$；点 $(-\sqrt{3}, 1)$ 在第二象限，所以主辐角 $\theta = \frac{5\pi}{6}$（注意 $\arctan\frac{1}{-\sqrt{3}} = -\frac{\pi}{6}$ 只在第四象限成立，必须补 $\pi$）。于是

$$z = 2\left(\cos\frac{5\pi}{6} + i\sin\frac{5\pi}{6}\right)$$

同一个数，写成三角式后，长度 $r=2$、方向 $\theta=\frac{5\pi}{6}$ 一眼可见。

**两个复数相等，当且仅当模相等且辐角相差 $2\pi$ 的整数倍：**

$$r_1(\cos\theta_1 + i\sin\theta_1) = r_2(\cos\theta_2 + i\sin\theta_2) \iff r_1 = r_2 \ \text{且}\ \theta_1 - \theta_2 = 2k\pi$$

「模相等 + 辐角相差 $2\pi$ 的整数倍」同时成立，才是同一个复数。这一点与直角坐标下「实部虚部分别相等」是同一件事的两种表述。

## 2 欧拉公式：指数与三角的联姻

三角表示让「长度 + 角度」一目了然，但乘除运算时括号里的 $\cos\theta + i\sin\theta$ 依然笨重。1748 年欧拉（Euler）在《无穷小分析引论》（Introductio in analysin infinitorum）中写下了一个把一切都变轻的公式。

**核心概念：欧拉公式（Euler's formula）**：

$$e^{i\theta} = \cos\theta + i\sin\theta$$

把它代入三角表示，就得到

**核心概念：复数的指数表示（exponential form）**：

$$z = r\,e^{i\theta}$$

欧拉公式的「含金量」在它把三类不同的事物连成一体：指数函数、三角函数、以及复数 $i$。把 $\theta = \pi$ 代入，得到数学中最著名的恒等式之一：

$$e^{i\pi} + 1 = 0$$

一个式子同时含纳 $0, 1, i, e, \pi$ 五个最基本的常数，常被称作**欧拉恒等式**。

欧拉公式还给出两个反向的「翻译」公式，它们把三角函数用复指数表示出来，是第七章傅里叶级数的引擎：

$$\cos\theta = \frac{e^{i\theta} + e^{-i\theta}}{2}, \qquad \sin\theta = \frac{e^{i\theta} - e^{-i\theta}}{2i}$$

只需把 $e^{\pm i\theta}$ 用欧拉公式展开，两式相加、相减即得。<span class="marginnote">当 $\theta = \pi$ 时欧拉公式给出 $e^{i\pi} = -1$，即「指数函数取纯虚数得到单位圆上的旋转」。第七章里任意信号 $f(t)$ 被拆成不同频率的 $e^{i\omega t}$ 之和，靠的就是这一句——每个 $e^{i\omega t}$ 都是「以角速度 $\omega$ 旋转的指针」。</span>

## 3 公式解析：从幂级数到欧拉公式

欧拉公式不是天外飞仙，它可以从第一级《微积分》学过的泰勒级数严格推出。我们把 $e^{i\theta}$、$\cos\theta$、$\sin\theta$ 都展开，分三步看：

- **第一步，展开 $e^{i\theta}$。** 对 $e^x$ 的泰勒展开 $e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!}$ 代入 $x = i\theta$：

$$e^{i\theta} = 1 + i\theta + \frac{(i\theta)^2}{2!} + \frac{(i\theta)^3}{3!} + \frac{(i\theta)^4}{4!} + \cdots$$

- **第二步，利用 $i$ 的幂循环。** $i^2 = -1$，$i^3 = -i$，$i^4 = 1$，$i$ 的幂按 $1, i, -1, -i$ 循环。于是偶数项都变成实数、奇数项都带着 $i$：

$$e^{i\theta} = \underbrace{\left(1 - \frac{\theta^2}{2!} + \frac{\theta^4}{4!} - \cdots\right)}_{\cos\theta} \;+\; i\,\underbrace{\left(\theta - \frac{\theta^3}{3!} + \frac{\theta^5}{5!} - \cdots\right)}_{\sin\theta}$$

- **第三步，认出两个熟悉的级数。** 实部括号里正是 $\cos\theta$ 的泰勒展开（正负交替、只含偶次幂），虚部括号里正是 $\sin\theta$ 的泰勒展开（正负交替、只含奇次幂）。实部加虚部，即得 $e^{i\theta} = \cos\theta + i\sin\theta$。

**为什么是「交替」的？** 关键在于 $i^2 = -1$：每乘一次 $i$，符号翻转一次。正是这一个「负号」，把指数函数的级数劈成实部、虚部两条流，恰好各自由 $\cos$、$\sin$ 掌管。这是复数最深的一次「双螺旋」，也是「$i^2 = -1$ 造就整个世界」的最直接例证。

## 4 指数表示：乘除即旋转与缩放

现在把两块拼图合起来。设

$$z_1 = r_1 e^{i\theta_1}, \qquad z_2 = r_2 e^{i\theta_2}$$

由指数运算律（同一个 $e$，底数相乘指数相加）：

$$z_1 z_2 = r_1 r_2\, e^{i(\theta_1 + \theta_2)}, \qquad \frac{z_1}{z_2} = \frac{r_1}{r_2}\, e^{i(\theta_1 - \theta_2)} \quad (z_2 \ne 0)$$

**重点：复数相乘，模相乘、辐角相加；复数相除，模相除、辐角相减。** 这正是第一节课后预告的「乘法即旋转」，现在用指数表示三行就写完了。$z_1 z_2$ 的几何动作是：把向量 $z_1$ 绕原点旋转 $\theta_2$ 角、并伸缩 $r_2$ 倍。

由此还立刻得到共轭的指数写法：$\bar{z} = \overline{re^{i\theta}} = r e^{-i\theta}$——镜像就是「反转旋转方向」。

三个基本例子让这套规则落地：

- $i = e^{i\pi/2}$，所以乘以 $i$ 就是逆时针旋转 $90°$；
- $z \cdot e^{i\theta}$ 就是把 $z$ 旋转 $\theta$ 角、长度不变；
- $z^n = r^n e^{in\theta}$，整数次幂的模取 $n$ 次方、辐角取 $n$ 倍——下一节《复数的乘幂与方根》将从这里出发。<span class="marginnote">「旋转的叠加是角度相加」，这一条在信号处理与深度学习中反复出现：大模型的旋转式位置编码（RoPE）把位置 $m$ 编码为旋转 $m\theta$，两个 token 的相对位置就是角度差——与这里 $e^{i\theta_1} e^{i\theta_2} = e^{i(\theta_1+\theta_2)}$ 是同一条律。</span>

**辨析｜易错点：$z = 0$ 不能写成指数式 $0 \cdot e^{i\theta}$。** 因为 $r = 0$ 时辐角无定义（上一节已辨析），$0$ 的指数式没有意义。遇到 $z = 0$ 直接写成 $0$，别硬套 $re^{i\theta}$。

## 5 小结

- **三角表示** $z = r(\cos\theta + i\sin\theta)$：用长度与角度编码复数；求辐角要按象限修正，不能只取 $\arctan$。
- **欧拉公式** $e^{i\theta} = \cos\theta + i\sin\theta$：由泰勒级数展开三步入手，$i^2=-1$ 造成实虚部正负交替。
- **指数表示** $z = re^{i\theta}$；欧拉恒等式 $e^{i\pi} + 1 = 0$；$\cos\theta = \frac{e^{i\theta}+e^{-i\theta}}{2}$、$\sin\theta = \frac{e^{i\theta}-e^{-i\theta}}{2i}$。
- **乘法与除法**：模相乘除、辐角相加减——旋转与缩放被指数运算吸收。
- **共轭**：$\bar{z} = re^{-i\theta}$，即反转旋转方向。
- **陷阱**：$z = 0$ 无指数式；三角式相等要求「模相等 + 辐角差 $2k\pi$」。

在下一节，我们拿着 $z = re^{i\theta}$ 这把钥匙去开两扇门：**乘幂**（$z^n$ 的旋转叠加）与**方根**（方程 $w^n = z$ 为什么总有 $n$ 个解，以及它们如何均匀分布在圆周上）。
