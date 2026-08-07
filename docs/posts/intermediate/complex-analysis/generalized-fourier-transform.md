---
title: 广义傅里叶变换：单位函数、正弦与余弦函数
date: 2026-08-08
---

# 广义傅里叶变换：单位函数、正弦与余弦函数

<div class="epigraph">
<p>有了 $\delta$ 函数，常数、正余弦、符号函数这些「不绝对可积」的老顽固，也都有了频谱身份证。</p>
<footer>—— 广义傅里叶变换观</footer>
</div>

<div class="article-byline">
<p>第二级 · 复变函数与积分变换 ｜ 西交《复变函数与积分变换》§7.4 ｜ 2026-08-08</p>
</div>

## 为什么「经典定义」管不住这些信号

经典傅里叶变换要求 $f$ 绝对可积。但常数 $1$、$\sin\omega_0t$、$\mathrm{sgn}\,t$、单位阶跃 $u(t)$——这些信号都不绝对可积，却在实际中无处不在（直流偏置、载波、开关动作）。**广义傅里叶变换**用 $\delta$ 函数把这些信号的「频谱」也定义出来：它们的谱是集中在特定频率的 $\delta$ 谱线。这一节把这些「标准广义变换对」全部列出，并给出推导。这是第七章承上（$\delta$）启下（性质与应用）的补给站。<span class="marginnote">「不绝对可积却有变换」的直觉：$e^{i\omega_0t}$ 虽然不绝对可积，但它的「能量」无限集中在 $\omega_0$ 这一个频率——分布意义下这个「无限集中」恰是 $\delta$。广义变换就是把「集中到一点的能量」合法化。</span>

## 1 单位函数（常数）的傅里叶变换

**核心变换对：**

$$1 \longleftrightarrow 2\pi\delta(\omega)$$

**推导：** 由上一节 $\delta(t)\leftrightarrow1$ 与对称性（$F(t)\leftrightarrow2\pi f(-\omega)$），把「$f(t)=\delta(t)$、$F(\omega)=1$」代入对称性：$F(t)=1\leftrightarrow 2\pi f(-\omega)=2\pi\delta(-\omega)=2\pi\delta(\omega)$。**或直接验证逆变换：$\frac1{2\pi}\int 2\pi\delta(\omega)e^{i\omega t}d\omega=1$。**<span class="marginnote">「常数 ↔ 零频处一根 $\delta$」的物理直觉：直流信号不含任何「变化」，全部能量压在 $\omega=0$ 这一个点上。现实中「偏置电压」的频谱就是 $\omega=0$ 的一根尖峰。</span>

**推广：** 任意常数 $C$：$C\leftrightarrow 2\pi C\delta(\omega)$。

## 2 正弦与余弦的傅里叶变换

**核心变换对：**

$$\cos\omega_0 t \longleftrightarrow \pi\left[\delta(\omega-\omega_0)+\delta(\omega+\omega_0)\right]$$

$$\sin\omega_0 t \longleftrightarrow \frac{\pi}{i}\left[\delta(\omega-\omega_0)-\delta(\omega+\omega_0)\right] = i\pi\left[\delta(\omega+\omega_0)-\delta(\omega-\omega_0)\right]$$

**推导（余弦）：** $\cos\omega_0t=\frac{e^{i\omega_0t}+e^{-i\omega_0t}}2$，而 $e^{i\omega_0t}\leftrightarrow2\pi\delta(\omega-\omega_0)$（上一节），故

$$\mathcal{F}[\cos\omega_0t]=\frac12\cdot2\pi\delta(\omega-\omega_0)+\frac12\cdot2\pi\delta(\omega+\omega_0)=\pi[\delta(\omega-\omega_0)+\delta(\omega+\omega_0)]$$

**解读：** 余弦信号 = 两个单频（$\pm\omega_0$）的叠加，频域是**两根对称的谱线**，高度 $\pi$。**正弦同理，但两根谱线异号**（因为 $\sin$ 是奇函数，对应奇对称谱）。<span class="marginnote">「余弦两根谱线」与「单频 $e^{i\omega_0t}$ 一根谱线」的差别，正是「实信号 ↔ 复信号」的差别：实余弦含正负两个旋转方向，复指数只含一个。频谱分析里用「单边谱」（只取正频率）处理实信号，就是这个原因的工程化。</span>

## 3 单位阶跃函数与符号函数

**核心概念：单位阶跃函数（unit step function）**：$u(t)=\begin{cases}1,&t>0\\0,&t<0\end{cases}$（在 $t=0$ 处取值任定，通常取 $\frac12$ 或 $1$）。**核心概念：符号函数（sign function）**：$\mathrm{sgn}\,t=\begin{cases}1,&t>0\\-1,&t<0\end{cases}$。

**符号函数的变换：**

$$\mathrm{sgn}\,t \longleftrightarrow \frac{2}{i\omega}$$

**推导思路：** $\mathrm{sgn}\,t$ 是奇函数、不绝对可积。用「截断 + 极限」：$\mathrm{sgn}\,t=\lim_{a\to0^+}(e^{-at}u(t)-e^{at}u(-t))$，逐项变换（用到 $e^{-at}u(t)\leftrightarrow\frac1{a+i\omega}$）再取极限，得 $\frac{2}{i\omega}$。

**阶跃函数的变换：** 关键观察 $u(t)=\frac12(1+\mathrm{sgn}\,t)$，故

$$u(t) \longleftrightarrow \pi\delta(\omega) + \frac{1}{i\omega}$$

**阶跃的频谱 = 直流分量（$\pi\delta(\omega)$）+ 奇对称分量（$\frac1{i\omega}$）。** 这分解完美对应「阶跃 = 常数一半 + 符号函数一半」。<span class="marginnote">$u(t)$ 的变换在系统分析里极其常用：电路接通（阶跃输入）的响应、阶跃响应的频谱，都直接引用这个公式。记住「阶跃 = 常数/2 + sgn/2」的分解，比硬背公式更可靠。</span>

## 4 公式解析：$\mathrm{sgn}\,t$ 变换的极限推导

$\mathrm{sgn}\,t$ 的变换是广义变换的「标准推导」，拆开看极限技巧：

$$\mathrm{sgn}\,t = \lim_{a\to 0^+}\left(e^{-at}u(t) - e^{at}u(-t)\right)$$

- **第一步，构造衰减近似。** $e^{-at}u(t)$（$t>0$ 部分，$a>0$ 使其可积）与 $e^{at}u(-t)$（$t<0$ 部分）分别可积；$a\to0^+$ 时两者趋向 $u(t)$ 与 $u(-t)$，差趋向 $\mathrm{sgn}\,t$。
- **第二步，逐项变换。** 对 $a>0$，$\int_0^{\infty}e^{-at}e^{-i\omega t}dt=\frac1{a+i\omega}$；对 $t<0$ 项对称得 $\frac1{a-i\omega}$。相减：
$$\mathcal{F}[e^{-at}u(t)-e^{at}u(-t)]=\frac1{a+i\omega}-\frac1{a-i\omega}=\frac{-2i\omega}{a^2+\omega^2}$$
- **第三步，取极限 $a\to0^+$。** $\lim\frac{-2i\omega}{a^2+\omega^2}=\frac{-2i\omega}{\omega^2}=\frac{2}{i\omega}$（$\omega\ne0$ 处）；$\omega=0$ 处分布意义下取主值。故 $\mathrm{sgn}\,t\leftrightarrow\frac{2}{i\omega}$。

**直觉：** 「先加衰减因子使可积 → 逐项变换 → 极限还原」是广义变换的标准三步曲。**衰减因子 $e^{-at}$ 是「临时拐杖」**——用完（取极限）就丢，但帮助把不绝对可积的信号「扶进」傅里叶变换的框架。<span class="marginnote">这条「截断-变换-极限」的路线，和第八章拉普拉斯变换的动机完全一致：拉普拉斯变换正是「对 $f(t)u(t)$ 乘 $e^{-st}$（$s=\sigma+i\omega$，$\sigma$ 提供衰减）」——它把傅里叶变换的「收敛拐杖」制度化。学完第七章再看第八章，你会认出同一条思路。</span>

## 5 广义变换对总表

| $f(t)$ | $F(\omega)$ | 说明 |
| --- | --- | --- |
| $1$ | $2\pi\delta(\omega)$ | 直流 |
| $\delta(t)$ | $1$ | 冲击 |
| $e^{i\omega_0t}$ | $2\pi\delta(\omega-\omega_0)$ | 单频 |
| $\cos\omega_0t$ | $\pi[\delta(\omega-\omega_0)+\delta(\omega+\omega_0)]$ | 余弦 |
| $\sin\omega_0t$ | $\frac\pi i[\delta(\omega-\omega_0)-\delta(\omega+\omega_0)]$ | 正弦 |
| $\mathrm{sgn}\,t$ | $\frac{2}{i\omega}$ | 符号 |
| $u(t)$ | $\pi\delta(\omega)+\frac1{i\omega}$ | 阶跃 |

**重点：广义变换的谱都含 $\delta$ 谱线或奇点型项。** 凡是「不绝对可积但在物理中有意义」的信号，其频谱总落在「$\delta$ 谱线 + 代数型谱」的组合——这就是广义变换的特征「指纹」。<span class="marginnote">做题时先判断「信号是否绝对可积」：可积走经典变换，不可积走广义变换表。$\delta$ 谱线出现的位置（$\omega_0$）就是信号的「载频」，它的高度给出信号在那一频率的「强度」。</span>

## 6 补充：广义变换的实战用法

广义傅里叶变换在信号处理里几乎是「常数与正余弦」的标准答案，把三种典型用法梳理清楚。

**用法一：求「截断信号」的频谱。** 信号只在 $[0,T]$ 内非零时，频谱 $\int_0^T f(t)e^{-i\omega t}dt$ 直接算（无需广义变换）。**广义变换处理的是「全时域持续」的信号**——常数、周期信号。

**用法二：周期信号的谱线结构。** 周期信号 $f(t)=\sum_n c_ne^{in\omega_0t}$（傅里叶级数），逐项取广义变换：

$$F(\omega)=2\pi\sum_n c_n\delta(\omega-n\omega_0)$$

**周期信号的频谱是「离散谱线族」**——每根谱线在谐波频率 $n\omega_0$ 处、强度 $2\pi c_n$。**这是「周期 ⟺ 离散谱」对偶的严格表述**，也是频谱分析仪显示「一根根谱线」的数学原因。

**用法三：调制信号的频谱搬移。** $f(t)\cos\omega_0t$ 的频谱 $\frac12[F(\omega-\omega_0)+F(\omega+\omega_0)]$——即使 $f$ 是常数（$F=2\pi\delta$），结果也合法：$\cos\omega_0t$ 的频谱是两根 $\pi\delta$ 谱线，与第三节一致。

**例（综合）：** 求 $u(t)\cos\omega_0t$ 的频谱。$u(t)\leftrightarrow\pi\delta(\omega)+\frac1{i\omega}$，由频移定理：

$$\mathcal{F}[u(t)\cos\omega_0t]=\frac12\left[\pi\delta(\omega-\omega_0)+\frac1{i(\omega-\omega_0)}+\pi\delta(\omega+\omega_0)+\frac1{i(\omega+\omega_0)}\right]$$

**「阶跃 × 余弦」的频谱 = 谱线 + 代数尾巴**——这类「因果振荡」信号在系统分析里极常见。

**辨析｜易错点：广义变换的结果含 $\delta$，取值要谨慎。** $F(\omega)$ 在 $\omega=\omega_0$ 处是「无限高」的谱线，不能当作普通函数代入求值——**它只在「乘以别的函数再积分」时才有意义**（分布意义）。**工程上读谱线高度（系数），不读「点值」。**

**重点：广义变换的「身份」是分布。** 用的时候记住两件事——谱线位置（频率）、谱线强度（系数）；别试图「代入点值」。

## 7 小结

- **常数**：$1\leftrightarrow2\pi\delta(\omega)$；$C\leftrightarrow2\pi C\delta(\omega)$。
- **余弦**：$\cos\omega_0t\leftrightarrow\pi[\delta(\omega-\omega_0)+\delta(\omega+\omega_0)]$（两根谱线）。
- **正弦**：$\sin\omega_0t\leftrightarrow\frac\pi i[\delta(\omega-\omega_0)-\delta(\omega+\omega_0)]$（异号谱线）。
- **符号/阶跃**：$\mathrm{sgn}\,t\leftrightarrow\frac2{i\omega}$；$u(t)\leftrightarrow\pi\delta(\omega)+\frac1{i\omega}$。
- **推导法**：截断（加衰减）+ 逐项变换 + 极限，是广义变换的标准三步曲。

在下一节，我们系统整理**傅里叶变换的性质**：线性、位移、微分、积分。这些性质让「算变换」不必每次都积分——用已知变换对 + 性质规则，就能「变」出大量新变换。
