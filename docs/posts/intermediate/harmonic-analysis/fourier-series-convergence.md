---
title: Fourier 级数与收敛性
date: 2026-08-11
---

# Fourier 级数与收敛性

<div class="epigraph">
<p>对自然的深入研究，是数学发现最丰富的源泉。</p>
<footer>—— 约瑟夫 · 傅里叶（Jean-Baptiste Joseph Fourier, 1768–1830）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 调和分析 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 Fourier 级数开始

我们网站的主线是「从极限到大模型」。这条主线的中段有一座必须翻越的山：**一个任意形状的周期信号，能否被分解成无数个「纯正弦波」的叠加？** Fourier 级数就是这座山的入口，也是整个调和分析的第一块基石——它把「函数」这种抽象的连续对象，翻译成「一族系数」这种我们可以逐个计算、逐个检验的离散数据。

在真正的大模型时代，这套翻译依旧无处不在：Transformer 的位置编码借用了频率基底的思路，语音合成与图像压缩用正弦基做编解码，而「把一个复杂对象分解成简单成分之和」恰恰是深度学习中「表征学习」的数学祖先。<span class="marginnote">位置编码（positional encoding）里常见 $\text{PE}_{(pos, 2i)}=\sin(pos/10000^{2i/d})$，本质就是用一列不同频率的三角波给序列里的每个位置打上「频率指纹」。</span>学这一课，我们其实是在学：**为什么「波」可以成为描述一切的通用语言。**

## 1 周期函数与三角级数

先固定研究范围。称函数 $f$ 是 **$2\pi$-周期** 的，若对一切 $x$ 有 $f(x+2\pi)=f(x)$。所有 $2\pi$-周期、且在 $[0,2\pi]$ 上 Lebesgue 可积的复值函数构成空间 $L^1(\mathbb{T})$，其中 $\mathbb{T}$ 表示单位圆周<span class="marginnote">记号 $\mathbb{T}=\mathbb{R}/2\pi\mathbb{Z}$ 把「周期函数」与「圆周上的函数」视为同一件事：把实数轴按 $2\pi$ 卷成圈，每点的函数值自动复制到下一个周期。到本专题最后一讲，我们会发现这正是一个「商群」。</span>。为行文方便，我们常在 $[-\pi,\pi]$ 上研究 $f$。

**三角多项式（trigonometric polynomial）** 是形如
$$
P_N(x) = \sum_{n=-N}^{N} a_n e^{inx}
$$
的有限和。由 Euler 公式 $e^{inx}=\cos nx+i\sin nx$，它等价于有限个 $\cos$ 与 $\sin$ 的线性组合。三角多项式是「好」的：连续、可微、周期，怎么折腾都不出事。Fourier 的伟大设想是：**把任意周期函数「看成」无穷阶的三角多项式。**

**重点**：这里出现的是复指数 $e^{inx}$，而不是单纯的 $\cos$、$\sin$。复指数的好处在于它把「频率」写成单一的指标 $n$，并且把 $\cos$、$\sin$ 之间的相位关系藏进了指数里，一切代数运算（平移、微分、卷积）都因此变得透明。

## 2 Fourier 系数：把函数投影到「基本波」上

为什么偏偏选 $e^{inx}$ 做基本波？因为它们两两正交：

$$
\frac{1}{2\pi}\int_{0}^{2\pi} e^{inx}\,\overline{e^{imx}}\,dx =
\begin{cases}
1, & n=m,\\
0, & n\neq m.
\end{cases}
$$

这就像欧氏空间里的标准正交基：用「内积」一测，各个方向干干净净地分开。于是我们把 $f$ 「投影」到第 $n$ 个方向上，得到 **Fourier 系数（Fourier coefficient）**：

$$
\widehat f(n) = \frac{1}{2\pi}\int_{0}^{2\pi} f(\theta)\,e^{-in\theta}\,d\theta, \qquad n\in\mathbb{Z}.
$$

用系数拼回部分和：

$$
S_N(f)(x) = \sum_{n=-N}^{N} \widehat f(n)\,e^{inx}.
$$

**核心问题来了**：当 $N\to\infty$ 时，$S_N(f)$ 在什么意义下收敛？收敛到什么？「收敛」这个词在不同范数下意思完全不同，这是本讲最重要的辨析点。

**辨析｜易错点：** 三个收敛层次必须分清。**逐点收敛**（对每个 $x$，$S_N(f)(x)\to f(x)$）是最强的直观要求；**一致收敛**更强，要求整条曲线的逼近误差一致趋零；**$L^2$（均方）收敛** 最宽松，只要求
$\int_0^{2\pi}|S_N(f)-f|^2 dx \to 0$。Fourier 级数的奇妙之处正在于：它在 $L^2$ 意义下对几乎所有 $L^2$ 函数成立，但逐点收敛却极为挑剔——这是个一眼看不穿的深潭。

## 3 Dirichlet 核与逐点收敛的困难

部分和可以写成一个卷积。代入系数公式并交换求和与积分：

$$
S_N(f)(x) = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(x-\theta)\,D_N(\theta)\,d\theta,
\qquad
D_N(\theta)=\sum_{n=-N}^{N} e^{in\theta}
=\frac{\sin\left((N+\tfrac12)\theta\right)}{\sin(\tfrac12\theta)}.
$$

核函数 $D_N$ 称为 **Dirichlet 核（Dirichlet kernel）**。它「浓缩」了逼近的全部机制：$S_N(f)$ 是 $f$ 在 $x$ 附近按权重 $D_N$ 做的局部平均。<span class="marginnote">注意 $D_N$ 的积分恰好是 $1$（逐项积分即得），且当 $N$ 增大时它的主瓣越来越窄、越来越尖——直觉上它该是「好的近似恒等核」。但下面这条性质马上会推翻这个直觉。</span>

问题出在**绝对值**上。可以证明

$$
\frac{1}{2\pi}\int_{-\pi}^{\pi} \left|D_N(\theta)\right|\,d\theta
\;\ge\; c\log N \;\to\; \infty,
$$

即 $D_N$ 的 $L^1$ 范数无界增长。这意味着 Dirichlet 核是一个**坏核**：它带正负号、振荡剧烈，$S_N$ 不是正的局部平均，而会「放大」输入函数的振荡。<span class="marginnote">这里透出调和分析的第一条方法论：<strong>想证收敛，先造好核。</strong> 一个好的核 $K_N$ 要满足三条：$\int K_N=1$、$\int|K_N|\le C$（$L^1$ 范数有界）、以及质量集中（对任意 $\delta>0$，$\int_{|x|>\delta}|K_N(x)|dx\to 0$）。$D_N$ 只满足第一条，因此通往一致收敛的路被堵死。</span>

经典 **Dirichlet 定理** 在更强的正则性条件下救人：若 $f$ 在 $x$ 附近分段光滑（例如分段 $C^1$），则

$$
\lim_{N\to\infty} S_N(f)(x) = \frac{f(x^+)+f(x^-)}{2},
$$

即在连续点回到 $f(x)$，在跳跃点收敛到左右极限的平均。这条定理本身不难证（对 $D_N$ 做局部化 + 黎曼-勒贝格引理），但它把「逐点收敛」的责任全部推给了函数的光滑性——只要 $f$ 有一点不光滑，Fourier 级数就开始「造反」。

## 4 Fejér 核与 Cesàro 平均：一致收敛的突破口

1900 年，年仅 20 岁的匈牙利数学家 **Leopold Fejér** 打破了僵局。他不再看部分和，而是看部分和的算术平均——**Cesàro 平均**：

$$
\sigma_N(f)(x)=\frac{S_0(f)(x)+S_1(f)(x)+\cdots+S_N(f)(x)}{N+1}.
$$

对应地，卷积核换成了 Fejér 核：

$$
F_N(\theta)=\frac{1}{N+1}\sum_{n=0}^{N}D_n(\theta)
=\frac{1}{N+1}\left(\frac{\sin\left(\frac{N+1}{2}\theta\right)}{\sin\frac{\theta}{2}}\right)^{2}.
$$

**重点：** $F_N \ge 0$（它是个平方！），$\int F_N=1$，且对任意 $\delta>0$ 有 $\int_{\delta<|\theta|\le\pi}F_N(\theta)\,d\theta \to 0$。三条「好核」条件全部满足。

于是就有了调和分析历史上第一个重量级结果——**Fejér 定理**：若 $f$ 在 $\mathbb{T}$ 上连续，则 Cesàro 平均 $\sigma_N(f)$ 一致收敛到 $f$；若 $f\in L^1(\mathbb{T})$，则 $\sigma_N(f)$ 在 $L^1$ 意义下收敛到 $f$，且逐点收敛在 Lebesgue 点处成立。<span class="marginnote">「平均」这个平凡动作（把坏核 $D_0,\dots,D_N$ 的振荡抵消掉）换来了一致收敛，堪称化腐朽为神奇。它给后续整个理论定下基调：<strong>遇到收敛困难的算子，试试对部分和取平均。</strong> 同一个想法后来在 Littlewood–Paley 理论、拟微分算子里反复出现。</span>

一个立即的推论：连续周期函数可以被三角多项式一致逼近——这就是 **Weierstrass 逼近定理** 的三角版本，也是 $C(\mathbb{T})$ 可分的直接证明。

## 5 公式解析：Parseval 恒等式

Fourier 级数最美丽的守恒律如下：

$$
\boxed{\;\frac{1}{2\pi}\int_{0}^{2\pi}\left|f(\theta)\right|^{2}d\theta
\;=\;\sum_{n=-\infty}^{\infty}\left|\widehat f(n)\right|^{2}\;}
$$

对 $f\in L^2(\mathbb{T})$ 成立。逐项拆解这条公式：

- **第一步，左边是什么**：$\dfrac{1}{2\pi}\int|f|^2$ 是函数 $f$ 的「能量」的周期平均——把 $f$ 看作一段电压信号，它就是一周内的平均功率。Fourier 级数理论里习惯用 $\frac{1}{2\pi}$ 归一化，使常数函数 $f\equiv 1$ 的能量正好是 $1$。
- **第二步，右边是什么**：$\sum_n|\hat f(n)|^2$ 把同一份能量按频率重新记账：每个频率 $n$ 贡献 $|\hat f(n)|^2$。这里系数 $\hat f(n)$ 的模平方就是「第 $n$ 个频率上藏着多少功率」。
- **第三步，为什么成立**：设 $S_N$ 是部分和。由正交性，$f-S_N$ 与 $S_N$ 正交，于是
$$
\frac{1}{2\pi}\int|f|^2=\frac{1}{2\pi}\int|S_N|^2+\frac{1}{2\pi}\int|f-S_N|^2
=\sum_{|n|\le N}|\hat f(n)|^2+\frac{1}{2\pi}\int|f-S_N|^2.
$$
  Fejér 定理保证了 $S_N\to f$（在 $L^2$ 意义下，可由 Cesàro 平均先收敛再逼近部分和），故余项趋于 $0$。
- **第四步，它告诉我们的世界观**：能量不增不减，只是从「按位置分布」换成了「按频率分布」。这是 Fourier 分析宇宙观的浓缩——**同一个东西，位置域与频率域是等价的两种记账方式。**

Parseval 恒等式等价于映射 $f \mapsto (\widehat f(n))_{n\in\mathbb{Z}}$ 是 $L^2(\mathbb{T})$ 到 $\ell^2(\mathbb{Z})$ 的**等距同构**（乘以常数 $1/\sqrt{2\pi}$），这为后面 $L^2$ 理论的建立（即 Plancherel 定理）铺平了道路。

## 6 Gibbs 现象与「不可信」的直觉

即使 $f$ 只有一个跳跃间断点，Fourier 级数也会露出狰狞面目。取方波 $f(x)=1_{0<x<\pi}-1_{\pi<x<2\pi}$（周期化），部分和 $S_N$ 在间断点附近会出现**过冲**：曲线不是乖乖爬到 $\pm1$，而是冲过头，冲出约 **$9\%$** 再带着高频振荡回摆（见下图）。

![Gibbs 现象](/images/harmonic-analysis/fourier-series-gibbs-1.svg)

**Gibbs 现象（Gibbs phenomenon）** 由 J. Willard Gibbs 在 1898–1899 年通过书信公之于众：过冲幅度约为
$$
\frac{2}{\pi}\int_{0}^{\pi}\frac{\sin t}{t}\,dt-1 \approx 0.0895,
$$
即对单位跳跃约 $8.9\%$，与 $N$ 无关。<span class="marginnote">增大 $N$ 只会把过冲<strong>挤向</strong>间断点、压缩振荡的宽度，但过冲的高度纹丝不动。这条恒定的 $9\%$ 正是「逐点收敛」在非光滑点上失效的直观显影。</span>它给一切使用 Fourier 级数的工程实践敲响警钟：**用有限项近似一个带跳变的信号，永远补不上那道「耳朵」**，除非改用别的手段（例如事后在频域做衰减窗，正是现代信号处理里「振铃抑制」的由来）。

**辨析｜易错点：** 不要以为「$N$ 足够大，曲线就足够贴近」。在 $L^\infty$ 范数下，$S_N(f)$ 对不连续函数**不收敛**（Gibbs 现象正是反例）；但它在 $L^2$ 下收敛，在逐点意义下收敛到左右平均。三个收敛概念，对应三种截然不同的行为——这正是第 2 节强调的辨析点的具体兑现。

## 7 小结

- **Fourier 系数** $\widehat f(n)=\frac{1}{2\pi}\int_0^{2\pi} f(\theta)e^{-in\theta}d\theta$ 是把周期函数投影到正交基 $e^{inx}$ 上的「坐标」。
- **Dirichlet 核** $D_N$ 是部分和的卷积核，但因 $\int|D_N|\sim\log N$ 发散而**不是好核**，导致逐点/一致收敛困难。
- **Fejér 核** $F_N\ge 0$ 是好核，Cesàro 平均 $\sigma_N(f)$ 对连续函数一致收敛（Fejér 定理）。
- **Parseval 恒等式** $\frac{1}{2\pi}\int|f|^2=\sum_n|\hat f(n)|^2$：能量在位置域与频率域守恒，$L^2\cong\ell^2$。
- **Gibbs 现象**：跳跃点附近部分和过冲约 $9\%$，与截断阶数无关，是 $L^\infty$ 收敛失败的明证。

在下一节，我们将迈出从「一个周期函数」到「一族函数」的关键一步：研究**极大函数与 Hardy-Littlewood 定理**——那个能同时解释「Lebesgue 微分定理」与「Fourier 级数逐点收敛」的、无处不在的算子。
