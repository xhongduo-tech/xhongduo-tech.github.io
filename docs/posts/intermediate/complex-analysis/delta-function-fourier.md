---
title: 单位脉冲函数（δ 函数）及其傅里叶变换
date: 2026-08-08
---

# 单位脉冲函数（δ 函数）及其傅里叶变换

<div class="epigraph">
<p>$\delta$ 函数是一个「几乎没有宽度的单位面积」——工程用它，数学替它圆场。</p>
<footer>—— 广义函数论</footer>
</div>

<div class="article-byline">
<p>第二级 · 复变函数与积分变换 ｜ 西交《复变函数与积分变换》§7.3 ｜ 2026-08-08</p>
</div>

## 为什么需要「不存在的函数」

理想冲击——锤子敲一下、雷达发一个脉冲、电路里一个瞬时短路——在数学上都是「持续时间为零、幅度无穷、面积有限」的对象。普通函数无法描述它（没有这样的普通函数），但它的变换（频谱平坦）又如此有用。狄拉克（Dirac）在 1930 年代引入 $\delta$ 函数，工程界立刻爱不释手；数学家后来用「广义函数」（施瓦兹分布）把它严格化。**$\delta$ 函数是本专题从「经典」走向「广义」的分水岭**，也是理解「常数、正弦为何有傅里叶变换」的钥匙。<span class="marginnote">$\delta$ 函数的严格定义属于广义函数论（分布论）：$\delta$ 不是「点处的值」，而是「作用在测试函数上的泛函」$\langle\delta,\phi\rangle=\phi(0)$。工程师把它当函数用，数学家保证它严谨——两种视角都值得保留。</span>

## 1 δ 函数的定义与性质

**核心概念：单位脉冲函数（Dirac delta function）$\delta(t)$**：由两条「理想化」性质定义：

$$\delta(t) = \begin{cases} 0, & t\ne 0,\\ \infty, & t=0 \end{cases}, \qquad \int_{-\infty}^{\infty}\delta(t)\,dt = 1$$

它不是普通函数，而是「广义函数」（分布）。实用中常把它看成「高而窄、面积为 1」的脉冲族的极限：$\delta(t)=\lim_{\varepsilon\to0}\frac{1}{\varepsilon}p_{\varepsilon/2}(t)$（高 $\frac1\varepsilon$、宽 $\varepsilon$、面积 $1$ 的矩形）。

**三条核心性质：**

1. **筛选性（sampling property）**：$\int_{-\infty}^{\infty}f(t)\delta(t-t_0)\,dt = f(t_0)$。
2. **偶函数**：$\delta(-t)=\delta(t)$（直观：脉冲不偏向任何一侧）。
3. **缩放**：$\delta(at)=\frac{1}{|a|}\delta(t)$（$a\ne0$）；特别地 $\delta(2t)=\frac12\delta(t)$。

**重点：筛选性是 $\delta$ 的第一用途。** 它与任何函数相乘再积分，就「挑出」那个函数在某一点的值——这让积分里的「抠出特定时刻」变得干净利落。<span class="marginnote">筛选性为什么成立：$\delta(t-t_0)$ 只在 $t=t_0$ 处「活着」，积分只留下 $f(t_0)\int\delta dt=f(t_0)$。这一条让 $\delta$ 成为「采样」的数学语言——数字信号处理里「采样」就是「乘 $\delta$ 梳再积分」。</span>

## 2 δ 函数的傅里叶变换

**重点：$\delta(t)$ 的傅里叶变换是常数 $1$：**

$$\mathcal{F}[\delta(t)] = \int_{-\infty}^{\infty}\delta(t)e^{-i\omega t}\,dt = e^{-i\omega\cdot0} = 1$$

（由筛选性，被积函数只在 $t=0$ 处有贡献，值为 $e^0=1$。）

**对偶地，常数 $1$ 的傅里叶变换是 $2\pi\delta(\omega)$：**

$$\mathcal{F}[1] = 2\pi\delta(\omega)$$

**物理含义：** 一个理想的瞬时冲击（$\delta$）包含**所有频率且强度相等**——频谱平坦（白噪声的频谱就是平的）。反过来说，「常数信号」（直流）只含零频——频谱是位于 $\omega=0$ 的一根 $\delta$ 谱线。<span class="marginnote">「冲击 ↔ 平坦谱」是傅里叶对偶的极端体现：时域最「尖锐」的对象（$\delta$），对应频域最「平坦」的对象（常数）。信号越尖锐，频谱越宽——这就是「测不准」在信号里的普遍规律。</span>

**更一般的位移公式：** 对 $\delta(t-t_0)$：

$$\mathcal{F}[\delta(t-t_0)] = \int\delta(t-t_0)e^{-i\omega t}dt = e^{-i\omega t_0}$$

**时域移位的冲击 = 频域只加相位（模仍为 1）。** 这预告了下一节的「位移定理」。

## 3 公式解析：$\delta$ 函数的筛选性怎么用

筛选性看似简单，实则是整个 $\delta$ 演算的引擎。拆开一道典型用法：

$$\int_{-\infty}^{\infty}f(t)\delta(t-t_0)\,dt = f(t_0)$$

- **第一步，识别「脉冲位置」。** $\delta(t-t_0)$ 只在 $t=t_0$ 处非零——其余所有 $t$ 处被积函数为零。
- **第二步，把 $f(t)$ 在 $t_0$ 处「冻结」。** 因为只有 $t=t_0$ 有贡献，$f(t)$ 在积分里等价于 $f(t_0)$（常数）。
- **第三步，提出常数，用面积性质。** $\int f(t_0)\delta(t-t_0)dt=f(t_0)\int\delta(t-t_0)dt=f(t_0)\cdot1$。
- **实战用法：** 求 $\int_{-\infty}^{\infty}e^{-t^2}\delta(t-3)dt=e^{-9}$——一秒得答案。**任何「$\delta$ 乘函数再积分」都能这样「抠值」。**

**直觉：** $\delta$ 像一把「采样探针」——放在哪里，就报告那个位置的函数值。**积分遇到 $\delta$，等于把被积函数「钉」在脉冲位置求值。**

## 4 广义函数的傅里叶变换观

$\delta$ 函数的引入让傅里叶变换的版图完整。**没有 $\delta$，常数、正弦、符号函数都「没有傅里叶变换」**（它们不绝对可积）；有了 $\delta$，它们都有——谱是集中在某些频率的 $\delta$ 谱线。

**核心概念：广义傅里叶变换（generalized Fourier transform）**：把 $\delta$ 函数纳入变换理论，使「不绝对可积但在分布意义下有意义的信号」（常数、$e^{i\omega_0t}$、$\cos\omega_0t$、$\mathrm{sgn}\,t$）也有傅里叶变换。

**例：** $e^{i\omega_0t}$ 的变换：由「$1\leftrightarrow2\pi\delta(\omega)$」与位移定理，得

$$\mathcal{F}[e^{i\omega_0t}] = 2\pi\delta(\omega-\omega_0)$$

**单频信号 ↔ 频域一根 $\delta$ 谱线**——这完美印证「时域一个频率、频域一根谱线」的直觉。$\cos\omega_0t=\frac{e^{i\omega_0t}+e^{-i\omega_0t}}2$ 则对应两根谱线（正负频率各一根）。<span class="marginnote">广义变换的正式框架是分布论：$\langle\hat f,\phi\rangle=\langle f,\hat\phi\rangle$ 把变换的定义从「积分」推广到「对偶配对」。工程课通常直接接受「$\delta$ 是合法对象」，数学课则用分布论严格化。两种态度各有价值，本专题按工程惯例使用。</span>

## 5 δ 函数在系统分析中的角色

**核心概念：单位冲激响应（impulse response）**：线性时不变系统对 $\delta(t)$ 输入的输出 $h(t)$，称为系统的**单位冲激响应**。

**核心概念：频率响应（frequency response）**：$H(\omega)=\mathcal{F}[h(t)]$，即冲激响应的傅里叶变换。

**为什么 $\delta$ 是「万能探针」：** 任意输入 $x(t)$ 都能写成「无穷多个 $\delta$ 的加权叠加」$x(t)=\int x(\tau)\delta(t-\tau)d\tau$。由系统的线性与时不变性，输出

$$y(t)=\int x(\tau)h(t-\tau)d\tau = (x*h)(t)$$

**输出 = 输入与冲激响应的卷积。** 在频域，卷积变乘法：$Y(\omega)=X(\omega)H(\omega)$——**这就是系统分析的「时域卷积、频域相乘」原则**，它的建立完全依赖 $\delta$ 的筛选性。<span class="marginnote">「$x(t)=\int x(\tau)\delta(t-\tau)d\tau$」是 $\delta$ 最深刻的应用：任意信号被分解成「无穷多个不同时刻的冲击」。每个冲击单独通过系统产生 $h(t-\tau)$，线性叠加出输出。这个「分解-响应-叠加」的思路是信号与系统课程的骨架。</span>

## 6 补充：$\delta$ 函数运算的常用恒等式

$\delta$ 函数的「运算规则」在实际计算中高频出现，整理成一组可查的小表。

**恒等式一（缩放）：** $\delta(at)=\frac1{|a|}\delta(t)$。**例：** $\delta(2t)=\frac12\delta(t)$——脉冲被「压缩两倍」，面积不变所以高度加倍。

**恒等式二（复合函数）：** 若 $g$ 只有简单零点 $t_k$（$g(t_k)=0$，$g'(t_k)\ne0$），则

$$\delta(g(t))=\sum_k\frac{\delta(t-t_k)}{|g'(t_k)|}$$

**例：** $\delta(t^2-1)=\frac{\delta(t-1)}{2}+\frac{\delta(t+1)}{2}$——$\delta$ 在复合函数的每个零点各「放一个脉冲」，强度按导数倒数。

**恒等式三（筛选推广）：** $\int f(t)\delta(g(t))dt=\sum_k\frac{f(t_k)}{|g'(t_k)|}$。

**恒等式四（与阶跃）：** $\delta(t)$ 是 $u(t)$ 的分布导数：$u'(t)=\delta(t)$。**推导：** 对 $u$ 求导，$t\ne0$ 处导数为 $0$，$t=0$ 处是「单位跳变」——分布意义下就是 $\delta$。**这条连接了「阶跃响应」与「冲激响应」：冲激响应是阶跃响应的导数。**

**例（用恒等式求变换）：** $\mathcal{F}[\delta(2t)]=\frac12\mathcal{F}[\delta(t)]=\frac12$。**缩放把变换「摊」一半。**

**重点：$\delta$ 的「值」无意义，「积分作用」才是一切。** 所有恒等式最终都是「在积分里如何使用」的规则——**别试图给 $\delta(t)$ 在 $t=0$ 赋值，只按「筛选/面积」两条公理操作。**

**辨析｜易错点：$\delta(g(t))$ 的复合要求 $g$ 的零点「简单」。** 若 $g$ 有重零点（$g'=0$），$\delta(g(t))$ 在分布意义下无定义（需更精细处理）。**先确认 $g'$ 在零点非零，再套复合公式。**

## 7 小结

- **$\delta(t)$**：面积 1、只在 $t=0$ 处非零的广义函数；筛选性 $\int f\delta=f(0)$、偶性、缩放性。
- **变换**：$\delta(t)\leftrightarrow1$；$1\leftrightarrow2\pi\delta(\omega)$；$\delta(t-t_0)\leftrightarrow e^{-i\omega t_0}$。
- **单频谱线**：$e^{i\omega_0t}\leftrightarrow2\pi\delta(\omega-\omega_0)$；$\cos\omega_0t$ 对应两根谱线。
- **广义变换**：$\delta$ 让常数、正弦、符号函数都有了傅里叶变换。
- **系统分析**：输出 $=$ 输入 $\times$ 冲激响应（卷积）；频域相乘 $Y=XH$。

在下一节，我们把 $\delta$ 的威力用于「补全」傅里叶变换的版图：**广义傅里叶变换：单位函数、正弦与余弦函数**。常数、符号、正余弦、阶跃——这些经典信号的变换谱线一网打尽。
