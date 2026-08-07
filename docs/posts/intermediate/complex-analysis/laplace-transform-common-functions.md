---
title: 常见函数的拉普拉斯变换
date: 2026-08-08
---

# 常见函数的拉普拉斯变换

<div class="epigraph">
<p>一张拉普拉斯变换表，就是信号处理的「对数表」——查表、组合、性质，三步拿下一个变换。</p>
<footer>—— 变换表的使用哲学</footer>
</div>

<div class="article-byline">
<p>第二级 · 复变函数与积分变换 ｜ 西交《复变函数与积分变换》§8.2 ｜ 2026-08-08</p>
</div>

## 为什么「变换表」是核心资产

拉普拉斯变换的应用中，**求变换**几乎从不从定义积分硬算——而是查表。一张「常见函数 ↔ 变换」的表 + 一组性质（下节），足以覆盖几乎所有工程信号。这一节系统推导并整理基本变换表：阶跃、幂函数、指数、正余弦、双曲、$\delta$。每个都给出推导（至少一次），让你既会查表、也会自己推。<span class="marginnote">「查表 + 性质」是积分变换工程化的关键：工程师不追求从定义积分，而是把新信号「拆」成表里信号的组合，再用线性与性质拼出答案。这就像微积分里「查导数表 + 链式法则」——先背工具，再练组合。</span>

## 1 单位阶跃与 $\delta$ 函数

**变换对：**

$$\mathcal{L}[u(t)] = \frac{1}{s}, \qquad \mathcal{L}[\delta(t)] = 1$$

**推导（阶跃）：** $\mathcal{L}[u(t)]=\int_0^{\infty}1\cdot e^{-st}dt=\frac{e^{-st}}{-s}\Big|_0^{\infty}=\frac1s$（$\mathrm{Re}\,s>0$）。

**推导（$\delta$）：** 由筛选性 $\int_0^{\infty}\delta(t)e^{-st}dt=e^{-s\cdot0}=1$。

**推广：** 位移脉冲 $\delta(t-t_0)$（$t_0>0$）：$\mathcal{L}[\delta(t-t_0)]=e^{-st_0}$。

**例：** 阶跃的「延迟版」$u(t-a)$（$a>0$）——下节延迟定理会给出通用公式 $e^{-as}/s$，先记住这个特例。<span class="marginnote">$\delta(t)$ 的拉普拉斯变换是 $1$，与傅里叶一致。而阶跃 $u(t)$ 的拉普拉斯是 $\frac1s$，ROC $\mathrm{Re}s>0$——注意它不含 $\pi\delta(\omega)$ 项（那是傅里叶的直流谱线），因为拉普拉斯只在 $t\ge0$ 积分，常数在 $t<0$ 的部分被「无视」了。</span>

## 2 幂函数 $t^n$

**变换对：**

$$\mathcal{L}[t^n] = \frac{n!}{s^{n+1}}, \qquad n = 0,1,2,\dots$$

**推导（用 $t$ 的递推）：** $\mathcal{L}[t]=\int_0^{\infty}te^{-st}dt$。用分部积分 $u=t$，$dv=e^{-st}dt$：

$$\mathcal{L}[t]=\left.\frac{-te^{-st}}{s}\right|_0^{\infty}+\frac1s\int_0^{\infty}e^{-st}dt=0+\frac1s\cdot\frac1s=\frac{1}{s^2}$$

**一般公式：** 由递推 $\mathcal{L}[t^n]=\frac{n}{s}\mathcal{L}[t^{n-1}]$（分部积分）归纳得 $\frac{n!}{s^{n+1}}$。

**例：** $\mathcal{L}[t^3]=\frac{3!}{s^4}=\frac6{s^4}$。

**辨析｜易错点：$t^0=1$ 时公式给出 $\frac{0!}{s^1}=\frac1s$——与阶跃一致。** 幂函数表以 $t^n$ 为基本行，阶跃是 $n=0$ 的特例。<span class="marginnote">「$\mathcal{L}[t^n]=n!/s^{n+1}$」的记忆法：分母次数比 $n$ 大一，分子是 $n!$。工程里最常用 $t$（一次幂）$\leftrightarrow\frac1{s^2}$ 与 $t^2\leftrightarrow\frac2{s^3}$——对应「斜坡」与「抛物线」输入信号。</span>

## 3 指数、正余弦与双曲函数

**变换对：**

$$\mathcal{L}[e^{at}] = \frac{1}{s-a}, \qquad \mathcal{L}[\sin\omega t] = \frac{\omega}{s^2+\omega^2}, \qquad \mathcal{L}[\cos\omega t] = \frac{s}{s^2+\omega^2}$$

$$\mathcal{L}[\sinh at] = \frac{a}{s^2-a^2}, \qquad \mathcal{L}[\cosh at] = \frac{s}{s^2-a^2}$$

**推导（正弦）：** 用欧拉公式 $\sin\omega t=\frac{e^{i\omega t}-e^{-i\omega t}}{2i}$，由指数公式：

$$\mathcal{L}[\sin\omega t]=\frac1{2i}\left(\frac1{s-i\omega}-\frac1{s+i\omega}\right)=\frac{1}{2i}\cdot\frac{2i\omega}{s^2+\omega^2}=\frac{\omega}{s^2+\omega^2}$$

**推导（双曲）：** $\sinh at=\frac{e^{at}-e^{-at}}2$，代入指数公式即得 $\frac{a}{s^2-a^2}$。

**重点：正余弦与双曲只差分母的符号（$+$ vs $-$）与分子（$\omega$/$s$ vs $a$/$s$）。** 并排记忆可互相校验——$s^2+\omega^2$ 管振荡，$s^2-a^2$ 管指数增长。<span class="marginnote">「$\sin$ 分子是 $\omega$、$\cos$ 分子是 $s$；$\sinh$ 分子是 $a$、$\cosh$ 分子是 $s$」——分子是「哪个量在分子」：正弦/双曲正弦的分母根在虚轴/实轴上，分子取「非 $s$」的那个量。这张表的对称性很强，值得品味。</span>

## 4 组合型：指数乘幂、指数乘三角

**指数位移公式（核心组合）：**

$$\mathcal{L}[e^{at}t^n] = \frac{n!}{(s-a)^{n+1}}, \qquad \mathcal{L}[e^{at}\sin\omega t] = \frac{\omega}{(s-a)^2+\omega^2}, \qquad \mathcal{L}[e^{at}\cos\omega t] = \frac{s-a}{(s-a)^2+\omega^2}$$

**推导思路：** 直接积分 $e^{at}$ 与幂/三角的乘积，或观察到「把 $s$ 换成 $s-a$」：

$$\mathcal{L}[e^{at}f(t)] = F(s-a)$$

**这条是下节「频域位移定理」**——$e^{at}$ 让变换里的 $s$ 平移 $a$。用它可从基本表瞬间写出大量组合变换。

**例：** $\mathcal{L}[e^{-2t}\cos 3t]$：由 $\cos3t\leftrightarrow\frac{s}{s^2+9}$，把 $s\to s+2$：

$$\mathcal{L}[e^{-2t}\cos3t] = \frac{s+2}{(s+2)^2+9}$$

**衰减振荡（$e^{-2t}\cos3t$）在频域 = 基本变换平移。**<span class="marginnote">「$e^{at}$ 乘信号 = $s$ 平移 $a$」是最常用的组合技巧：阻尼振荡 $e^{-\zeta\omega_nt}\sin\omega_dt$ 的变换，就是无阻尼的 $\frac{\omega_d}{(s+\zeta\omega_n)^2+\omega_d^2}$。自动控制里二阶系统的响应全靠这条。</span>

## 5 公式解析：$\mathcal{L}[t^n]$ 的分部积分递推

把幂函数变换的递推证明写成可复用的三步：

$$\mathcal{L}[t^n] = \int_0^{\infty}t^n e^{-st}dt = \frac{n}{s}\mathcal{L}[t^{n-1}]$$

- **第一步，分部积分。** 取 $u=t^n$，$dv=e^{-st}dt$，则 $du=nt^{n-1}dt$，$v=-\frac{e^{-st}}s$：
$$\int_0^{\infty}t^ne^{-st}dt=\left[-\frac{t^ne^{-st}}s\right]_0^{\infty}+\frac{n}{s}\int_0^{\infty}t^{n-1}e^{-st}dt$$
- **第二步，边界项消失。** $t^n e^{-st}\big|_{0}^{\infty}=0$（$\mathrm{Re}s>0$ 时 $t\to\infty$ 指数压过多项式；$t=0$ 时 $n\ge1$ 为零）。
- **第三步，递推。** 余项正是 $\frac ns\mathcal{L}[t^{n-1}]$。从 $\mathcal{L}[t^0]=\frac1s$ 出发归纳：$\mathcal{L}[t^n]=\frac{n}{s}\cdot\frac{n-1}{s}\cdots\frac1s\cdot\frac1s=\frac{n!}{s^{n+1}}$。

**直觉：** 分部积分把「幂降一阶」——每分部一次，$t^n$ 变成 $t^{n-1}$、多出一个 $\frac ns$ 因子。**$n$ 次分部积到底，累积出 $n!$ 与 $s^{n+1}$。** 这个「降阶递推」是拉普拉斯变换处理多项式的标准手法。

## 6 基本变换对总表

| $f(t)$（$t\ge0$） | $F(s)$ | 记忆点 |
| --- | --- | --- |
| $\delta(t)$ | $1$ | 冲击 |
| $u(t)$ | $\frac1s$ | 阶跃 |
| $t^n$ | $\frac{n!}{s^{n+1}}$ | 幂 |
| $e^{at}$ | $\frac1{s-a}$ | 指数 |
| $\sin\omega t$ | $\frac{\omega}{s^2+\omega^2}$ | 振荡 |
| $\cos\omega t$ | $\frac{s}{s^2+\omega^2}$ | 振荡 |
| $\sinh at$ | $\frac{a}{s^2-a^2}$ | 双曲 |
| $\cosh at$ | $\frac{s}{s^2-a^2}$ | 双曲 |
| $e^{at}f(t)$ | $F(s-a)$ | 频移 |
| $t^ne^{at}$ | $\frac{n!}{(s-a)^{n+1}}$ | 组合 |

**重点：背表时按「家族」记**——阶跃/幂一族、指数一族、振荡一族、双曲一族、频移组合一族。**配上下节的线性/微分/积分/延迟性质，这张表足以应付全部初等信号的变换。**<span class="marginnote">「按家族记表」比死背强得多：每族有一个「母变换」（如指数 $e^{at}\leftrightarrow\frac1{s-a}$），其余是它的变体（$t^ne^{at}$、$e^{at}\sin$）。理解「母变换 + 性质」= 掌握全族。</span>

## 7 小结

- **阶跃与冲击**：$u(t)\leftrightarrow\frac1s$；$\delta(t)\leftrightarrow1$；$\delta(t-t_0)\leftrightarrow e^{-st_0}$。
- **幂函数**：$t^n\leftrightarrow\frac{n!}{s^{n+1}}$，分部积分递推导出。
- **指数/三角/双曲**：$e^{at}$、$\sin/\cos$、$\sinh/\cosh$ 各一族，分母符号区分振荡与增长。
- **组合公式**：$e^{at}f(t)\leftrightarrow F(s-a)$（频移），$t^ne^{at}\leftrightarrow\frac{n!}{(s-a)^{n+1}}$。
- **查表哲学**：按家族记表 + 性质组合，不必每次从定义积分。

在下一节，我们系统整理**拉普拉斯变换的性质：线性、微分、积分**。微分性质把 $y'(t)$ 变成 $sY(s)-y(0)$——**初始条件从这里进场**，解初值问题的开关就此打开。
