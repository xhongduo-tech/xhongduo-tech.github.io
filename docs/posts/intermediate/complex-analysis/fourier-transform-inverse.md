---
title: 傅里叶变换及其逆变换
date: 2026-08-08
---

# 傅里叶变换及其逆变换

<div class="epigraph">
<p>傅里叶变换是一扇双向门：时域里看信号，频域里看频谱——同一枚硬币的两面。</p>
<footer>—— 傅里叶变换观</footer>
</div>

<div class="article-byline">
<p>第二级 · 复变函数与积分变换 ｜ 西交《复变函数与积分变换》§7.2 ｜ 2026-08-08</p>
</div>

## 为什么把「内层积分」单独命名

上一节傅里叶积分里有个内层积分 $\int f(\tau)e^{-i\omega\tau}d\tau$——它把「时间的函数」变成「频率的函数」。这个映射太重要，值得单独命名：**傅里叶变换**。它的意义是「频域分析」：给定一个时域信号 $f(t)$，$F(\omega)$ 告诉你在角频率 $\omega$ 处有多少成分。而逆变换从频谱还原信号。这一节把定义、存在条件、基本例子与几何直觉立起来。<span class="marginnote">傅里叶变换把「分析信号」变成「分析频谱」。在工程里这相当于换了一副眼镜：时域里纠缠的信号（如一段和弦），在频域里是几根清晰的谱线。从音频均衡器到心电图分析，都是这副眼镜的用法。</span>

## 1 傅里叶变换与逆变换的定义

**核心概念：傅里叶变换（Fourier transform）**：设 $f(t)$ 在 $(-\infty,\infty)$ 上绝对可积，定义

$$F(\omega) = \mathcal{F}[f(t)] = \int_{-\infty}^{\infty}f(t)e^{-i\omega t}\,dt$$

$F(\omega)$ 称为 $f$ 的**傅里叶变换**（或频谱函数）。

**核心概念：傅里叶逆变换（inverse Fourier transform）**：

$$f(t) = \mathcal{F}^{-1}[F(\omega)] = \frac{1}{2\pi}\int_{-\infty}^{\infty}F(\omega)e^{i\omega t}\,d\omega$$

**记号约定：** 常用 $f(t)\leftrightarrow F(\omega)$ 表示「$f$ 与 $F$ 构成傅里叶变换对」。**注意逆变换前面的 $\frac1{2\pi}$ 因子，正反两式只差这个系数与指数符号。**

**重点：变换对的结构。** 正变换用 $e^{-i\omega t}$、逆变换用 $e^{i\omega t}$，符号相反保证「变过去再变回来」恰好还原。**时域与频域互为对方的「镜像世界」，两扇门对称。**<span class="marginnote">不同教材的常数约定略有差异：有的把 $\frac1{2\pi}$ 拆成 $\frac1{\sqrt{2\pi}}$ 分给两个方向（对称形式），有的用频率 $f$ 而非角频率 $\omega$。公式的物理内容不变，做题前先确认所用教材的约定。</span>

## 2 存在条件与基本性质

**存在条件：** 若 $f$ 在 $(-\infty,\infty)$ 上绝对可积且满足狄利克雷条件，则 $F(\omega)$ 存在且逆变换在连续点还原 $f$。

**基本性质（本节先用，下节系统证明）：**

1. **线性性**：$\mathcal{F}[af+bg]=a\mathcal{F}[f]+b\mathcal{F}[g]$。
2. **对称性**：若 $f(t)\leftrightarrow F(\omega)$，则 $F(t)\leftrightarrow 2\pi f(-\omega)$。
3. **奇偶性**：若 $f$ 是实偶函数，则 $F$ 是实偶函数；若 $f$ 是实奇函数，则 $F$ 是纯虚奇函数。

**例：** 门函数 $f(t)=\begin{cases}1,&|t|\le a\\0,&|t|>a\end{cases}$。变换：

$$F(\omega)=\int_{-a}^{a}e^{-i\omega t}dt=\frac{e^{-i\omega t}}{-i\omega}\Big|_{-a}^{a}=\frac{e^{i\omega a}-e^{-i\omega a}}{i\omega}=\frac{2\sin a\omega}{\omega}$$

**门函数 ↔ $2\frac{\sin a\omega}{\omega}$（sinc 型）。** 这个变换对是信号处理里最常引用的「模板」——门函数对应 sinc 频谱，反之亦然。<span class="marginnote">门函数 ↔ sinc 的变换对揭示了「时域压缩 ⇔ 频域展宽」的对偶：门越窄（$a$ 小），sinc 主瓣越宽（高频成分越多）。雷达要发短脉冲（时域窄），就必须占用宽频带——这是测不准原理在信号里的化身。</span>

## 3 公式解析：逆变换为什么能「还原」

逆变换 $f(t)=\frac1{2\pi}\int F(\omega)e^{i\omega t}d\omega$ 为何恰好还原 $f$？代入正变换拆解：

$$f(t) = \frac{1}{2\pi}\int_{-\infty}^{\infty}\left[\int_{-\infty}^{\infty}f(\tau)e^{-i\omega\tau}d\tau\right]e^{i\omega t}\,d\omega = \frac{1}{2\pi}\int_{-\infty}^{\infty}f(\tau)\left[\int_{-\infty}^{\infty}e^{i\omega(t-\tau)}d\omega\right]d\tau$$

- **第一步，交换积分次序。** 把 $f(\tau)$ 提出内层（内层只依赖 $\omega$ 与 $\tau$）。
- **第二步，认出内层积分是「$\delta$ 函数」。** 内层 $\int e^{i\omega(t-\tau)}d\omega$ 是狄利克雷核的极限，它在分布意义下等于 $2\pi\delta(t-\tau)$（下节讲 $\delta$ 函数）。
- **第三步，用 $\delta$ 的筛选性。** $f(t)=\frac1{2\pi}\int f(\tau)\cdot 2\pi\delta(t-\tau)d\tau=f(t)$——$\delta$ 函数「挑出」$\tau=t$ 那一项，完美还原。

**直觉：** 逆变换是「把无穷多个频域分量 $F(\omega)e^{i\omega t}$ 叠加起来」。每个频率贡献一个旋转指针 $e^{i\omega t}$，加权 $F(\omega)$；叠加时不同频率「互相抵消」只在 $t$ 处「共振」出原信号。**频域叠加 = 时域重构，这就是逆变换的物理含义。**

## 4 常用傅里叶变换对

| $f(t)$ | $F(\omega)$ | 备注 |
| --- | --- | --- |
| 门函数 $p_a(t)$ | $\frac{2\sin a\omega}{\omega}$ | sinc 型 |
| $e^{-a|t|}$（$a>0$） | $\frac{2a}{a^2+\omega^2}$ | 洛伦兹线型 |
| $e^{-at^2}$ | $\sqrt{\frac{\pi}{a}}e^{-\omega^2/(4a)}$ | 高斯 ↔ 高斯 |
| $\delta(t)$ | $1$ | 冲击 → 平坦谱 |
| $1$ | $2\pi\delta(\omega)$ | 常数 → 冲击（下节广义变换） |
| $e^{i\omega_0t}$ | $2\pi\delta(\omega-\omega_0)$ | 单频 → 谱线 |
| $\cos\omega_0t$ | $\pi[\delta(\omega-\omega_0)+\delta(\omega+\omega_0)]$ | 余弦 → 两根谱线 |
| $\mathrm{sgn}\,t$ | $\frac{2}{i\omega}$ | 符号函数 |

**重点：这些变换对是信号处理的「乘法口诀表」**，尤其是 $e^{i\omega_0t}\leftrightarrow 2\pi\delta(\omega-\omega_0)$——它说明「单频信号在频域是一根谱线」，是频谱分析最核心的图像。<span class="marginnote">「时域单频 ↔ 频域谱线」的严格表述需要 $\delta$ 函数（下一节），因为 $\cos\omega_0t$ 不绝对可积，经典定义下不存在傅里叶变换。引入广义函数后，「常数、正弦、符号函数」这些不绝对可积的信号也都有了变换——这就是「广义傅里叶变换」那一节要处理的。</span>

## 5 傅里叶变换的物理解读

**核心概念：频谱密度（spectral density）**：$F(\omega)$ 是**频谱密度**——$|F(\omega)|$ 度量「角频率 $\omega$ 附近单位频率宽度的信号强度」，$\arg F(\omega)$ 是该频率分量的相位。

**解读三件套：**

1. **振幅谱** $|F(\omega)|$：哪些频率强、哪些弱——信号的能量分布。
2. **相位谱** $\arg F(\omega)$：各频率分量的「对齐方式」——决定波形形状。
3. **频谱搬移**：乘以 $e^{i\omega_0t}$ 把整个频谱平移到 $\omega_0$——调制与解调的数学基础。

**例：** 一段声音信号 $f(t)$，其 $|F(\omega)|$ 在 $200\sim 4000$ Hz 区域高——说明语音能量集中在中频；高频衰减快——对应「语音平滑、无嘶嘶声」。**振幅谱把「听到什么」变成「看到什么」。**<span class="marginnote">相位谱常被忽略，但它绝非无关紧要：两张振幅谱相同、相位谱不同的图，一张是清晰照片，另一张可能是「相位乱序」的噪声。JPEG、MP3 等压缩能丢一部分相位信息而不被察觉，是因为人眼/人耳对相位的敏感度有限——这是感知编码的发现。</span>

## 6 补充：傅里叶变换的对称性之美

傅里叶变换有一组「对偶性质」，掌握它们能把「已知一个变换对」放大成「一串变换对」。

**对称性定理：** 若 $f(t)\leftrightarrow F(\omega)$，则 $F(t)\leftrightarrow 2\pi f(-\omega)$。

**推导：** 由逆变换 $f(t)=\frac1{2\pi}\int F(\omega)e^{i\omega t}d\omega$，把 $t$ 与 $\omega$ 的角色互换并整理即得。**「时域函数」与「频域函数」交换位置，结果自动配对。**

**应用一：从门函数得到 sinc 的对偶。** 已知 $p_a(t)\leftrightarrow\frac{2\sin a\omega}{\omega}$。由对称性，$\frac{2\sin at}{t}\leftrightarrow2\pi p_a(-\omega)=2\pi p_a(\omega)$。**时域的 sinc 对应频域的门函数**——「时域窄 ⟺ 频域宽」的对偶在此又一次应验。

**应用二：从 $\delta$ 得到常数。** $\delta(t)\leftrightarrow1$，对称性给出 $1\leftrightarrow2\pi\delta(\omega)$——**与广义变换一节完全一致**。对称性自动「免费」得到常数与 $\delta$ 的配对。

**应用三：从高斯得到高斯。** $e^{-at^2}\leftrightarrow\sqrt{\frac\pi a}e^{-\omega^2/(4a)}$ 的对称性保证「高斯变换仍是高斯」——这是唯一「形状不变」的变换对，也是测不准原理的数学根源（时域与频域都「紧」）。

**重点：对称性定理让「变换对表」翻倍。** 每背一个 $f\leftrightarrow F$，自动多一个 $F(t)\leftrightarrow2\pi f(-\omega)$。**查表时先想对称性，往往能省一次积分。**

**综合例：** 已知 $\frac{2a}{a^2+t^2}\leftrightarrow?$。由 $e^{-a|t|}\leftrightarrow\frac{2a}{a^2+\omega^2}$，对称性把「时域的 $\frac{2a}{a^2+t^2}$」映到「频域的 $2\pi e^{-a|\omega|}$」——**洛伦兹 ↔ 指数衰减**，又一个工程常用对。

**辨析｜易错点：对称性定理的因子 $2\pi$ 别忘了。** $F(t)\leftrightarrow2\pi f(-\omega)$ 的 $2\pi$ 来自逆变换的系数。**若采用对称归一化（正逆都带 $\frac1{\sqrt{2\pi}}$），对称性定理就没有 $2\pi$——因子随约定而变。**

## 7 小结

- **正变换** $F(\omega)=\int f(t)e^{-i\omega t}dt$，**逆变换** $f(t)=\frac1{2\pi}\int F(\omega)e^{i\omega t}d\omega$。
- **存在条件**：绝对可积 + 狄利克雷条件；逆变换在连续点还原。
- **基本性质**：线性、对称性（$F(t)\leftrightarrow2\pi f(-\omega)$）、奇偶性。
- **关键变换对**：门函数 ↔ sinc、高斯 ↔ 高斯、$e^{i\omega_0t}\leftrightarrow2\pi\delta(\omega-\omega_0)$。
- **物理解读**：$|F|$ 是振幅谱（能量分布），$\arg F$ 是相位谱（波形形状）。

在下一节，我们直面不绝对可积的信号：**单位脉冲函数（$\delta$ 函数）及其傅里叶变换**。$\delta$ 是「瞬时冲击」的理想化，它的变换是平坦谱 $1$——把「冲击响应」与「系统函数」连起来的桥梁。
