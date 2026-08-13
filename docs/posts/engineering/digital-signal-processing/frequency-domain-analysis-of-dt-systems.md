---
title: 离散时间系统的频域分析
date: 2026-08-07
---

# 离散时间系统的频域分析

<div class="epigraph">
<p>当系统对一切正弦都只能改变幅度与相位时，我们获得了描述它的最干净的语言。</p>
<footer>—— 让-巴蒂斯特 · 约瑟夫 · 傅里叶（Jean-Baptiste Joseph Fourier）</footer>
</div>

<div class="article-byline">
<p>第六级 · 数字信号处理 ｜ 程佩青 第2章 ｜ 2026-08-07</p>
</div>

## 为什么从频域入手

上一篇我们用脉冲响应 $h[n]$ 与卷积完整刻画了 LTI 系统，理论上已经「够了」。但卷积求和 $y[n] = \sum_k x[k]h[n-k]$ 计算繁琐，而且它给出的是一串时间上的数，看不出系统「放大了哪些成分、压掉了哪些成分」。频域分析换了一个坐标系：**把信号当作许多频率分量的叠加，把系统当作一个对每个频率分量的「缩放器与移相器」。** 在这个坐标系里，卷积变成乘法，滤波器设计变成「画一条频率响应的形状」，一切都变得干净。这一篇建立离散时间傅里叶变换（DTFT）与系统频率响应的完整框架——它是连接前面时域与后面 DFT/滤波器设计的中枢。

## 1 复指数是 LTI 系统的特征函数

频域分析的数学支点是一个简单却深刻的事实：**把复指数输入一个 LTI 系统，输出仍是同一频率的复指数，只是幅度变了、相位变了。**

设输入 $x[n] = \mathrm{e}^{\,\mathrm{j}\omega n}$，卷积求和给出

$$y[n] = \sum_{k=-\infty}^{\infty} h[k]\, \mathrm{e}^{\,\mathrm{j}\omega (n-k)} = \Big(\sum_{k=-\infty}^{\infty} h[k]\, \mathrm{e}^{-\mathrm{j}\omega k}\Big) \mathrm{e}^{\,\mathrm{j}\omega n} = H(\mathrm{e}^{\,\mathrm{j}\omega})\, \mathrm{e}^{\,\mathrm{j}\omega n}$$

其中

$$\boxed{\;H(\mathrm{e}^{\,\mathrm{j}\omega}) = \sum_{n=-\infty}^{\infty} h[n]\, \mathrm{e}^{-\mathrm{j}\omega n}\;}$$

就是系统的**频率响应（frequency response）**。<span class="marginnote">类比线性代数：复指数是 LTI 系统的「特征向量」，$H(\mathrm{e}^{\,\mathrm{j}\omega})$ 是它对应的「特征值」。同一个想法在连续世界表现为「$\mathrm{e}^{\mathrm{s}t}$ 是微分方程的特征函数」，在离散世界这里成立，在后面的 z 变换中则会推广到 $\mathrm{e}^{\mathrm{s}n}$。</span>

复指数 $\mathrm{e}^{\,\mathrm{j}\omega n}$ 因此被称为 LTI 系统的**特征函数（eigenfunction）**，频率响应 $H(\mathrm{e}^{\,\mathrm{j}\omega})$ 就是特征值。这个「特征值语言」带来两个推论：

- **正弦响应**：对实正弦输入 $x[n] = A\cos(\omega_0 n + \phi)$，输出为 $A|H(\mathrm{e}^{\,\mathrm{j}\omega_0})|\cos\big(\omega_0 n + \phi + \angle H(\mathrm{e}^{\,\mathrm{j}\omega_0})\big)$。系统只改变**幅度**（放大 $|H|$ 倍）与**相位**（平移 $\angle H$），不改变频率。
- **叠加**：任意信号是许多频率成分的叠加，每个成分独立地按自己的频率响应被处理，最后再加总——这正是「滤波」的全部含义。

**辨析｜易错点：** $H(\mathrm{e}^{\,\mathrm{j}\omega})$ 是 $\omega$ 的**周期函数，周期 $2\pi$**，因为 $\mathrm{e}^{-\mathrm{j}\omega n}$ 对 $\omega$ 以 $2\pi$ 为周期。所以「频率响应」完整信息只需画在一个长度 $2\pi$ 的区间上（通常取 $-\pi \le \omega \lt  \pi$）。把 $H(\mathrm{e}^{\,\mathrm{j}\omega})$ 画成随 $\omega$ 增长的直线，是初学者最常见的错误——它其实画在圆上。

## 2 离散时间傅里叶变换（DTFT）

特征函数的发现立刻引出反问题：**任意信号能拆成多少复指数？** 答案就是离散时间傅里叶变换。

**DTFT 正变换（analysis）**：

$$X(\mathrm{e}^{\,\mathrm{j}\omega}) = \sum_{n=-\infty}^{\infty} x[n]\, \mathrm{e}^{-\mathrm{j}\omega n}$$

**DTFT 反变换（synthesis）**：

$$x[n] = \frac{1}{2\pi} \int_{-\pi}^{\pi} X(\mathrm{e}^{\,\mathrm{j}\omega})\, \mathrm{e}^{\,\mathrm{j}\omega n}\,\mathrm{d}\omega$$

从反变换可以读出直觉：**$x[n]$ 是单位圆上一圈复指数的「加权平均」**，权重 $X(\mathrm{e}^{\,\mathrm{j}\omega})/2\pi$ 刻画了每个频率成分的大小与相位。<span class="marginnote">DTFT 把无穷长的序列 $x[n]$ 映射成一个连续周期函数 $X(\mathrm{e}^{\,\mathrm{j}\omega})$。注意角色的对称性：时域是离散的，频域就是周期的；时域无限长，频域就是连续函数。这个「离散 $\leftrightarrow$ 周期」的成对关系是整本书最核心的对称性，后面的 DFT 就是把它斩断到有限长。</span>

### 收敛条件

DTFT 不是对一切序列都有定义。**绝对可和是充分条件**：若 $\sum_n |x[n]| \lt  \infty$，则 $X(\mathrm{e}^{\,\mathrm{j}\omega})$ 存在且一致收敛。但很多有用的序列——如 $u[n]$、常数、正弦——并不绝对可和。它们仍可能有 DTFT，只是收敛是**均方意义**或**广义函数（包含 $\delta$ 冲激）意义**的：

$$x[n] = 1 \iff X(\mathrm{e}^{\,\mathrm{j}\omega}) = 2\pi \sum_{k} \delta(\omega + 2\pi k)$$

$$\cos(\omega_0 n) \iff X(\mathrm{e}^{\,\mathrm{j}\omega}) = \pi\sum_{k}\big[\delta(\omega-\omega_0+2\pi k) + \delta(\omega+\omega_0+2\pi k)\big]$$

频域的 $\delta$ 冲激表示「这个频率成分能量无穷集中」，与连续世界的傅里叶变换完全平行。<span class="marginnote">工程上很少纠结广义函数：DFT 处理的都是有限长序列，自动绝对可和。DTFT 的广义函数理论保证「正弦的频谱是一条线」这个直觉在数学上站得住。</span>

## 3 常用性质与对称性

DTFT 的性质与连续傅里叶变换几乎逐条对应，下面是使用频率最高的几条（记 $X(\mathrm{e}^{\,\mathrm{j}\omega}) = X$，$Y(\mathrm{e}^{\,\mathrm{j}\omega}) = Y$）：

| 性质 | 时域 | 频域 |
| --- | --- | --- |
| 线性 | $ax[n] + by[n]$ | $aX + bY$ |
| 时移 | $x[n-n_0]$ | $\mathrm{e}^{-\mathrm{j}\omega n_0} X$ |
| 频移 | $\mathrm{e}^{\,\mathrm{j}\omega_0 n} x[n]$ | $X(\mathrm{e}^{\,\mathrm{j}(\omega-\omega_0)})$ |
| 时间翻转 | $x[-n]$ | $X(\mathrm{e}^{-\mathrm{j}\omega})$ |
| 卷积 | $x[n] * h[n]$ | $X \cdot H$ |
| 相乘 | $x[n]\cdot y[n]$ | $\tfrac{1}{2\pi} X * Y$（圆周卷积） |
| 帕塞瓦尔 | $\sum_n |x[n]|^2$ | $\tfrac{1}{2\pi}\int_{-\pi}^{\pi}|X(\mathrm{e}^{\,\mathrm{j}\omega})|^2 \mathrm{d}\omega$ |

其中**卷积定理**是本篇最重要的一条：**时域卷积 = 频域相乘。** 它让「滤波」从一次逐点求和变成一次频谱逐点相乘——这正是后面 FFT 加速滤波（重叠相加/重叠保留）的理论基础。

**帕塞瓦尔定理**给出能量守恒：时域总能量等于频域能量除以 $2\pi$ 的积分，$|X(\mathrm{e}^{\,\mathrm{j}\omega})|^2$ 被称为**能量谱密度**。<span class="marginnote">帕塞瓦尔定理把「信号的强弱」翻译成「频谱下方区域的面积」。滤波器的通带/阻带设计，本质就是在安排这块面积的取舍。</span>

**共轭对称**：对实信号 $x[n]$，$X(\mathrm{e}^{-\mathrm{j}\omega}) = X^*(\mathrm{e}^{\,\mathrm{j}\omega})$，即幅度谱 $|X(\mathrm{e}^{\,\mathrm{j}\omega})|$ 是偶函数、相位谱是奇函数。**实信号只需看正频率**，这是所有频谱图只画一半的习惯来源。

## 4 公式解析：系统频率响应与差分方程的关系

对由差分方程描述的 LTI 系统，频率响应可以直接「读出」，无需先求 $h[n]$。

$$\sum_{k=0}^{N} a_k\, y[n-k] = \sum_{m=0}^{M} b_m\, x[n-m]$$

代入 $x[n] = \mathrm{e}^{\,\mathrm{j}\omega n}$、$y[n] = H(\mathrm{e}^{\,\mathrm{j}\omega})\mathrm{e}^{\,\mathrm{j}\omega n}$，利用时移性质逐项化简：

- **第一步**：左边 $y[n-k]$ 的每一项贡献 $a_k H(\mathrm{e}^{\,\mathrm{j}\omega}) \mathrm{e}^{-\mathrm{j}\omega k} \mathrm{e}^{\,\mathrm{j}\omega n}$；右边 $b_m \mathrm{e}^{-\mathrm{j}\omega m} \mathrm{e}^{\,\mathrm{j}\omega n}$。
- **第二步**：两边约去公因子 $\mathrm{e}^{\,\mathrm{j}\omega n}$，得

$$H(\mathrm{e}^{\,\mathrm{j}\omega}) \sum_{k=0}^{N} a_k \mathrm{e}^{-\mathrm{j}\omega k} = \sum_{m=0}^{M} b_m \mathrm{e}^{-\mathrm{j}\omega m}$$

- **第三步**：解出

$$\boxed{\;H(\mathrm{e}^{\,\mathrm{j}\omega}) = \dfrac{\sum_{m=0}^{M} b_m \mathrm{e}^{-\mathrm{j}\omega m}}{\sum_{k=0}^{N} a_k \mathrm{e}^{-\mathrm{j}\omega k}}\;}$$

**频率响应 = 分子多项式的 DTFT 除以分母多项式的 DTFT。** 分子对应前馈（零点），分母对应反馈（极点）。对 FIR（$N=0$，$a_0=1$），$H(\mathrm{e}^{\,\mathrm{j}\omega})$ 是一个纯多项式；对 IIR，它是两个多项式之比——这正是后面 z 变换一章「零极点图」与滤波器设计的技术基础。

**辨析｜易错点：** 求频率响应时常见的错误是把系数写反（$b_m$ 对应输入、$a_k$ 对应输出），或者在 IIR 情况下忘记分母必须有 $a_0=1$ 的归一化。核对方法很简单：**令 $\omega=0$（直流），$H(1) = \frac{\sum b_m}{\sum a_k}$**，可以快速验证一个滤波器对常数输入的总增益。

## 5 理想滤波器与实际的鸿沟

用频率响应的语言，「滤波器设计」就是塑造 $|H(\mathrm{e}^{\,\mathrm{j}\omega})|$ 的形状。**理想低通滤波器**的频率响应定义为

$$H_{\mathrm{lp}}(\mathrm{e}^{\,\mathrm{j}\omega}) = \begin{cases} 1, & |\omega| \lt  \omega_c \\ 0, & \omega_c \lt  |\omega| \le \pi \end{cases}$$

即在通带内增益为 1、相位为 0，阻带内完全截止。它的脉冲响应可以算出来：

$$h_{\mathrm{lp}}[n] = \frac{\sin(\omega_c n)}{\pi n}, \qquad -\infty \lt  n \lt  \infty$$

这个 $h_{\mathrm{lp}}[n]$ 有两点致命之处：**它是无限长的**（无法在现实中实现），而且**不是因果的**（$n\lt 0$ 时不为 0）。<span class="marginnote">「理想的滤波器不存在」是 DSP 的第一次现实教育：所有真实滤波器都是对理想的近似，代价是过渡带与通带/阻带波纹。用什么样的近似、付出多少阶数，正是第 7、8 章 Butterworth / Chebyshev / FIR 设计要系统回答的问题。</span>

因此理想滤波器只能被近似。工程上常用的妥协指标是：**通带波纹** $\delta_1$、**阻带衰减** $\delta_2$、**过渡带宽度**，以及**相位线性度**。哪一项可以放宽、哪一项必须死守，取决于应用——音频对相位不太敏感，图像和通信却极其敏感。这一章先把「理想的样子」立住，逼近它的工程手段留到滤波器设计两篇。

## 6 小结

- 复指数 $\mathrm{e}^{\,\mathrm{j}\omega n}$ 是 LTI 系统的**特征函数**，对应的特征值 $H(\mathrm{e}^{\,\mathrm{j}\omega}) = \sum_n h[n]\mathrm{e}^{-\mathrm{j}\omega n}$ 即**频率响应**。
- 系统对正弦输入只改变**幅度与相位**，不改变频率；任意输入的响应按频率成分独立处理后再叠加。
- **DTFT**：$X(\mathrm{e}^{\,\mathrm{j}\omega}) = \sum_n x[n]\mathrm{e}^{-\mathrm{j}\omega n}$，反变换在 $[-\pi,\pi]$ 上积分；离散时域对应**周期频域**。
- 卷积定理：**时域卷积 = 频域相乘**；帕塞瓦尔定理给出能量守恒。
- 差分方程系统的频率响应是**分子/分母多项式之比**，分子看零点、分母看极点。
- 理想低通滤波器**无限长且非因果**，无法实现，只能近似——这是滤波器设计的起点。

在下一节，我们将给频率响应穿上更强大的外衣：把 $\mathrm{e}^{\,\mathrm{j}\omega}$ 推广为一般复数 $z$