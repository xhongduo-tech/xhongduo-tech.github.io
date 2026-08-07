---
title: 傅里叶变换的性质：线性、位移、微分、积分
date: 2026-08-08
---

# 傅里叶变换的性质：线性、位移、微分、积分

<div class="epigraph">
<p>傅里叶变换的性质是一套「翻译规则」：时域的加减、平移、求导，在频域里都有对应的简单操作。</p>
<footer>—— 傅里叶变换操作手册</footer>
</div>

<div class="article-byline">
<p>第二级 · 复变函数与积分变换 ｜ 西交《复变函数与积分变换》§7.5 ｜ 2026-08-08</p>
</div>

## 为什么「性质」比「硬积分」重要

算傅里叶变换，从头积分常常又难又繁。傅里叶变换的性质提供了一条捷径：**从少数基本变换对出发，用「时域操作 ↔ 频域操作」的对应规则，组合出大量新变换。** 更重要的是，这些性质本身就是物理：位移定理说「时域平移 ⇔ 频域乘相位」，微分定理说「时域求导 ⇔ 频域乘 $i\omega$」——后者正是「微分方程化代数方程」的全部依据。本节建立线性、位移、微分、积分四条核心性质。<span class="marginnote">「时域操作与频域操作互相对应」是傅里叶分析的思维主线：每一对性质都是一座「翻译桥」。到第八章拉普拉斯变换，你会看到同一套性质几乎原样重现——先把桥修好，两章通用。</span>

## 1 线性性

**性质：** 对常数 $\alpha,\beta$ 与信号 $f,g$，

$$\mathcal{F}[\alpha f(t) + \beta g(t)] = \alpha F(\omega) + \beta G(\omega)$$

**证明：** 直接代入变换定义，积分是线性的。

**例：** 求 $\mathcal{F}[3e^{-2|t|} + 5\delta(t)]$。由 $e^{-a|t|}\leftrightarrow\frac{2a}{a^2+\omega^2}$ 与 $\delta(t)\leftrightarrow1$：

$$\mathcal{F}[\cdots] = 3\cdot\frac{4}{4+\omega^2} + 5\cdot1 = \frac{12}{4+\omega^2}+5$$

**线性性让「信号的叠加」对应「频谱的叠加」**——这正是频谱分析的基石：混合信号的总频谱 = 各成分频谱之和。

## 2 位移定理（时移与频移）

**性质（时域位移）：** 设 $f(t)\leftrightarrow F(\omega)$，则

$$f(t-t_0) \longleftrightarrow e^{-i\omega t_0}F(\omega)$$

**性质（频域位移）：** 对偶地，

$$e^{i\omega_0 t}f(t) \longleftrightarrow F(\omega-\omega_0)$$

**证明（时移）：** $\int f(t-t_0)e^{-i\omega t}dt$ 令 $u=t-t_0$，得 $e^{-i\omega t_0}\int f(u)e^{-i\omega u}du=e^{-i\omega t_0}F(\omega)$。

**解读：** **时域平移不改变振幅谱，只改变相位谱**（乘 $e^{-i\omega t_0}$，模为 1）。**频域位移（调制）把整个频谱平移到 $\omega_0$**——这是「调制/解调」的数学基础：把低频信号乘以载波 $e^{i\omega_0t}$，频谱搬移到 $\omega_0$ 附近。<span class="marginnote">「时移不改变振幅谱」是雷达测距的原理：回波相对发射的时延 $t_0$ 只体现在相位 $e^{-i\omega t_0}$，测出相位差就能反推距离。而「频移=调制」支撑着无线电广播：音频信号乘以射频载波，频谱搬上高频段发射。</span>

**例：** $g(t)=e^{-2|t-1|}$（把 $e^{-2|t|}$ 右移 1）。由 $e^{-2|t|}\leftrightarrow\frac4{4+\omega^2}$：

$$g(t) \longleftrightarrow e^{-i\omega}\cdot\frac{4}{4+\omega^2}$$

## 3 微分性质

**性质（时域微分）：** 设 $f(t)\leftrightarrow F(\omega)$，$f'$ 可变换，则

$$f'(t) \longleftrightarrow i\omega F(\omega)$$

**一般地：** $f^{(n)}(t)\longleftrightarrow(i\omega)^n F(\omega)$。

**性质（频域微分）：** 对偶地，

$$(-it)f(t) \longleftrightarrow F'(\omega), \qquad (-it)^n f(t)\longleftrightarrow F^{(n)}(\omega)$$

**证明（时域）：** 对逆变换 $f(t)=\frac1{2\pi}\int F(\omega)e^{i\omega t}d\omega$ 关于 $t$ 求导（可交换积分求导），得 $f'(t)=\frac1{2\pi}\int i\omega F(\omega)e^{i\omega t}d\omega$——正是「$i\omega F(\omega)$ 的逆变换」。

**重点：微分方程化代数方程。** 对含导数的方程两边取傅里叶变换，$\frac{d^n}{dt^n}$ 变成 $(i\omega)^n$——**微分方程变成代数方程**。这是下一节「用傅里叶解微分方程」的直接依据。<span class="marginnote">「求导 = 乘 $i\omega$」还揭示高频放大：高频分量（$\omega$ 大）在求导后被放大得多。所以「微分」是高频增强器，「积分」是低频增强器——图像锐化（微分）与模糊（积分）的频域解释正在于此。</span>

## 4 积分性质

**性质：** 设 $f(t)\leftrightarrow F(\omega)$ 且 $F(0)=0$（即 $\int f\,dt=0$，保证 $\int_{-\infty}^{t}f(\tau)d\tau$ 可变换），则

$$\int_{-\infty}^{t} f(\tau)\,d\tau \longleftrightarrow \frac{F(\omega)}{i\omega}$$

**证明思路：** 令 $g(t)=\int_{-\infty}^{t}f(\tau)d\tau$，则 $g'(t)=f(t)$。对时域微分性质反过来用：$f(t)\leftrightarrow i\omega G(\omega)$，故 $G(\omega)=\frac{F(\omega)}{i\omega}$。

**注意：若 $F(0)\ne0$，积分结果还多一个 $\pi F(0)\delta(\omega)$ 项**（直流成分，因为积分会把直流「累积」）。完整公式：

$$\int_{-\infty}^{t}f(\tau)d\tau \longleftrightarrow \frac{F(\omega)}{i\omega} + \pi F(0)\delta(\omega)$$

**例：** 求单位阶跃的积分关系验证：$\frac{1}{i\omega}\cdot\frac{1}{i\omega}$…（略）。**积分性质让「累加」在频域变成「除以 $i\omega$」**——与微分性质互逆，合起来构成微积分的频域镜像。<span class="marginnote">「微分乘 $i\omega$、积分除 $i\omega$」的对称美是傅里叶分析的精髓之一：微积分在时域里的一对互逆操作，在频域里就是「乘除 $i\omega$」一对互逆操作。这个「运算符 ↔ 乘法因子」的对应，让线性微积分方程在频域变成多项式方程。</span>

## 5 公式解析：微分性质三步证明

把「$f'(t)\leftrightarrow i\omega F(\omega)$」的证明拆开，理解「为什么求导变乘法」：

$$f'(t) \leftrightarrow i\omega F(\omega)$$

- **第一步，从逆变换出发。** $f(t)=\frac1{2\pi}\int F(\omega)e^{i\omega t}d\omega$。这是「信号 = 频域分量的叠加」。
- **第二步，对 $t$ 求导，把求导送进积分。** $f'(t)=\frac1{2\pi}\int F(\omega)\cdot\frac{d}{dt}e^{i\omega t}d\omega=\frac1{2\pi}\int F(\omega)\cdot i\omega e^{i\omega t}d\omega$。**求导只作用在 $e^{i\omega t}$ 上，产出因子 $i\omega$。**
- **第三步，认出逆变换。** $\frac1{2\pi}\int[i\omega F(\omega)]e^{i\omega t}d\omega$ 正是「$i\omega F(\omega)$ 的逆变换」——故 $f'\leftrightarrow i\omega F$。**每求一次导，指数 $e^{i\omega t}$ 就「吐出」一个 $i\omega$，因此 $n$ 阶导对应 $(i\omega)^n$。**

**直觉：** 复指数 $e^{i\omega t}$ 是「求导的本征函数」——对它求导等于乘 $i\omega$。傅里叶变换把信号拆成这些本征函数的叠加，于是「对信号求导」在频域变成「对每个分量乘 $i\omega$」。**这就是微分变乘法的深层原因：傅里叶基是微分算子的本征基。**

## 6 补充：微分性质的应用：从微分方程到传递函数

微分性质「$f'\leftrightarrow i\omega F$」在工程里最耀眼的应用是「把微分方程变代数方程」，这里用一道完整题演示。

**例：** 求解 $y''+2y'+y=f(t)$（零初值），其中 $f(t)$ 是任意输入。取傅里叶变换：

$$(i\omega)^2Y+2(i\omega)Y+Y=F \quad\Longrightarrow\quad Y(\omega)=\frac{F(\omega)}{(i\omega)^2+2i\omega+1}=\frac{F(\omega)}{(1+i\omega)^2}$$

**「微分算子」$(\frac{d^2}{dt^2}+2\frac d{dt}+1)$ 在频域变成「多项式」$(1+i\omega)^2$**——微分方程变成一个代数除法。输出 $y=\mathcal{F}^{-1}[Y]$。

**传递函数视角：** $H(\omega)=\frac1{(1+i\omega)^2}$ 是系统的频率响应。**给输入频谱 $F$，输出频谱 $Y=HF$——「乘法」代替「解方程」。**

**例（频谱搬移实战）：** 调幅信号 $g(t)=f(t)\cos\omega_0t$，用频移性质：

$$G(\omega)=\frac12[F(\omega-\omega_0)+F(\omega+\omega_0)]$$

若 $f$ 带限于 $|\omega|<B$ 且 $B<\omega_0$，$G$ 在 $\pm\omega_0$ 处各有一份 $F$ 的一半——**频谱被搬到载频两侧，互不重叠**。接收端用 $\cos\omega_0t$ 再乘、低通滤波，频谱搬回基带——**调制与解调各用一次频移性质。**

**重点：微分性质 + 频移性质是「频域工程」的两大支柱。** 前者把微分方程代数化（系统分析），后者把信号搬频（通信系统）。**两条性质单独看是技巧，合起来是「频域世界观」。**

**辨析｜易错点：傅里叶微分性质不带初值，只适合零初值/稳态问题。** 带初始条件的瞬态问题须用拉普拉斯变换（第八章）。**「稳态用傅里叶、瞬态用拉普拉斯」是工程铁律。**

## 7 小结

- **线性**：$\mathcal{F}[af+bg]=aF+bG$，叠加与频谱对应。
- **时移**：$f(t-t_0)\leftrightarrow e^{-i\omega t_0}F(\omega)$，只改相位不改振幅谱。
- **频移**：$e^{i\omega_0t}f(t)\leftrightarrow F(\omega-\omega_0)$，调制与频谱搬移。
- **微分**：$f^{(n)}\leftrightarrow(i\omega)^nF$；**频域微分** $(-it)^n f\leftrightarrow F^{(n)}$。
- **积分**：$\int_{-\infty}^t f\leftrightarrow\frac{F}{i\omega}+\pi F(0)\delta(\omega)$，与微分互逆。

在下一节，我们继续性质清单的另一半：**乘积定理与能量积分（帕塞瓦尔等式）**。时域的乘积对应频域的卷积，而帕塞瓦尔等式说「能量在时域等于频域」——信号能量的两种守恒。
