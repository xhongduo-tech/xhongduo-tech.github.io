---
title: Riesz 变换、乘子理论与分数次积分
date: 2026-08-11
---

# Riesz 变换、乘子理论与分数次积分

<div class="epigraph">
<p>Riesz 变换把「一个方向」的 Hilbert 变换编成了 $\mathbb{R}^n$ 的管弦乐队，而分数次积分是谱理论里最温柔的一个和弦。</p>
<footer>—— 马塞尔 · 里斯（Marcel Riesz, 1886–1969）与马塞尔 · 里斯之后的整个学派</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 调和分析 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 Riesz 变换开始

上一讲我们驯服了实轴上的 Hilbert 变换。可真实世界是 $n$ 维的：图像有横纵两个方向，矢量场有 $n$ 个分量，PDE 在 $\mathbb{R}^n$ 上演算。把「一维的 90° 相移」推广成「$n$ 个方向的偏导算子」，得到的就是 **Riesz 变换**——调和分析在欧氏空间高维化的第一块拼图。

而高维化的路不止一条。另一条路是**乘子理论**：既然 Hilbert 变换就是「频域乘 $-i\,\mathrm{sgn}(\xi)$」，那「频域乘任意合适的函数」能走多远？第三条路最出乎意料：核去掉奇性，反而得到**分数次积分**（Riesz 位势），它把「积分」的阶数从 0 平滑地推进到 $\alpha$，并与 Sobolev 嵌入、分数阶拉普拉斯直接相连。<span class="marginnote">Marcel Riesz（Fejér 之后、也同属雅盖隆谱系）在 1927 年前后把 Hilbert 变换写成 $\mathbb{R}^n$ 的「向量版」；而分数次积分则早在 19 世纪黎曼、刘维尔时期就埋下伏笔——「积分一次是幂、$\alpha$ 次积分是分数幂」的直觉，等了一百年才被 Hardy–Littlewood–Sobolev 定理严格地安放。</span>

## 1 Riesz 变换：$n$ 个方向的 Hilbert 变换

对 $j=1,\dots,n$，定义第 $j$ 个 **Riesz 变换（Riesz transform）**：

$$
R_j f(x)=c_n\,\mathrm{p.v.}\!\int_{\mathbb{R}^n} f(y)\,\frac{x_j-y_j}{|x-y|^{n+1}}\,dy,
$$

其中 $c_n=\frac{\Gamma((n+1)/2)}{\pi^{(n+1)/2}}$ 是归一化常数。它把一维核 $1/(\pi x)$ 换成了 $n$ 维奇核 $K_j(x)=c_n\,x_j/|x|^{n+1}$。

**重点：** $R_j$ 是「方向 $j$ 上的 Hilbert 变换」的精确意义是**乘子身份**：

$$
\widehat{R_j f}(\xi) = -i\,\frac{\xi_j}{|\xi|}\,\widehat f(\xi).
$$

**公式解析（乘子拆解）：**

- **第一步，乘子 $-i\,\xi_j/|\xi|$ 是什么**：它是「单位圆上的坐标」——把频率向量 $\xi$ 归一化到球面上再取第 $j$ 个坐标，乘以 $-i$。模长恰为 $1$，所以每个 $R_j$ 在 $L^2$ 上都是等距（范数 $1$）。
- **第二步，为什么这自然推广了 Hilbert**：$n=1$ 时 $\xi_1/|\xi|=\mathrm{sgn}(\xi)$，$-i\,\mathrm{sgn}(\xi)$ 正是 Hilbert 变换的乘子——**Riesz 变换在 $n=1$ 时退化为 Hilbert 变换**。
- **第三步，$R_j$ 之间如何咬合**：乘子满足恒等式 $\sum_j(\xi_j/|\xi|)^2=1$，故
$$
\sum_{j=1}^{n} R_j^2 = -I,
$$
  以及 $R_j^*=-R_j$（反自伴）、$R_jR_k=R_kR_j$（交换）——$R_j$ 构成一个 $n$ 维「Clifford 代数」系的生成元，像一把分量的「旋转算子」。
- **第四步，旋转协变性**：$R_j$ 在旋转下像向量一样变换，即 $R_j(f\circ\rho)=(\rho^{-1})$ 与 $\sum_j \xi_j R_j$ 配合，谐调和分析里「标量 ↔ 向量」的结构在此定型。

**辨析｜易错点：** $R_j$ 不是「对第 $j$ 个坐标做 Hilbert 变换再对其它坐标恒等」——那会得到别的算子。$R_j$ 的核 $x_j/|x|^{n+1}$ 在所有方向上都有尾，它同时「看到」所有方向，只是偏爱第 $j$ 个方向。把它想象成「带方向的奇异积分」而非「分坐标的 Hilbert 变换」，理解才正确。

## 2 Riesz 变换的 $L^p$ 有界性与新能力

核 $K_j$ 满足上一讲的全部条件：奇性可积于无穷、**球面上均值零**（$\int_{S^{n-1}}x_j d\sigma=0$ 因为奇函数）、Hörmander 条件成立。于是 C–Z 流水线直接上岗：

$$
\left\|R_j f\right\|_{L^p}\le C_p\,\|f\|_{L^p}, \qquad 1<p<\infty,\ j=1,\dots,n.
$$

这是 Riesz 变换最基本的定理。<span class="marginnote">新的能力在于「梯度的替代」：$R_j$ 在 $L^p$ 上可以当「微分」用。事实上有恒等式 $\partial_j(-R_j)=\cdots$ 使 $|\nabla f|\approx|\sum_j R_j\partial_j f|$ 的 $L^p$ 范数等价成立，于是 Riesz 变换成为调和函数、Sobolev 空间与椭圆正则性里「可被积分代替的微分」的化身——这是它在 PDE 里如此常见的原因。</span>

一个重要应用：调和函数梯度估计。若 $u$ 在球内调和，$R_j$ 能把「边界值到内部」的正则性定量地搬进去，得到 Stein 那本著名的《奇异积分与函数的可微性质》里的核心不等式——Riesz 变换连接调和分析与 PDE 的接口即在于此。

## 3 乘子理论：把「频域乘法」推广到极致

Riesz 变换引出一个更一般的问题。设 $m:\mathbb{R}^n\to\mathbb{C}$，定义**乘子算子** $T_m$：

$$
\widehat{T_m f}(\xi)=m(\xi)\,\widehat f(\xi).
$$

何时 $T_m$ 在 $L^p$ 上有界？$L^2$ 条件很简单（$|m|$ 有界即可）；$L^p$（$p\ne2$）要复杂得多。

**Mikhlin–Hörmander 乘子定理（Hörmander multiplier theorem）**：若 $m$ 在 $\mathbb{R}^n\setminus\{0\}$ 上光滑，且对一切多重指标 $|\alpha|\le\lfloor n/2\rfloor+1$ 有

$$
\left|\partial^{\alpha}m(\xi)\right|\le C\,|\xi|^{-|\alpha|},
$$

则 $T_m$ 在 $L^p$ 上有界，$1<p<\infty$。<span class="marginnote">条件说的是：$m$ 的大小与<strong>逐阶导数的衰减</strong>都被 $|\xi|^{-k}$ 控制——「尺度的导数」不放大振荡。$-i\xi_j/|\xi|$ 满足（$|\partial^\alpha|\lesssim|\xi|^{-|\alpha|}$），所以 Riesz 变换是这条大定理的首个案例。Mikhlin 1948 年、Hörmander 1960 年各自独立得到；它是现代奇异积分、PDE 拟微分算子理论的地基。</span>

**重点：** 乘子理论把「算子是否 $L^p$ 有界」化约为「乘子是否满足尺度条件」——这是一次巨大的维数压缩。拟微分算子、Littlewood–Paley 分解、双线性乘子估计（Hörmander 乘积）、乃至深度学习里卷积层的谱设计，全在乘子语言下工作。

**Riesz 变换与梯度的一个同构**：把向量算子记为 $Rf=(R_1f,\dots,R_nf)$。由乘子 $-i\xi_j/|\xi|$ 与 $|\xi|=\sqrt{\sum_j\xi_j^2}$，可以验证恒等式
$$
\sum_{j=1}^{n}R_j(-\partial_j)f = \Lambda f,\qquad \Lambda=\text{与 }|\xi|\text{ 相乘}
$$
（$\Lambda=(-\Delta)^{1/2}$ 是「一阶微分」）。于是「求一阶导」$\nabla f$ 的 $L^p$ 范数，被「先 $\partial_j$ 再 $R_j$ 再求和」这个全程有界的复合算子等价地表示出来——**Riesz 变换给了 $L^p$ 世界里一个「没有极点、处处好使」的梯度替身**，这就是它能在 PDE 正则性证明里替代经典导数使用的原因。

**辨析｜易错点：** 「$|m|\le C$ 处处成立」**不足以保证** $L^p$（$p\ne2$）有界。反例是 $m(\xi)=\mathrm{sgn}$ 的「高频圆环」调制——点态有界却产生 $L^p$ 无界的乘子。光滑性条件不是装饰，是 $L^p$ 有界性的命门。

## 4 公式解析：分数次积分与 Hardy–Littlewood–Sobolev

现在把视线转向「不带奇性」的积分算子。对 $0<\alpha<n$，**分数次积分（Riesz 位势 / fractional integral）**：

$$
\boxed{\;I_\alpha f(x)=c_{n,\alpha}\int_{\mathbb{R}^n}\frac{f(y)}{|x-y|^{n-\alpha}}\,dy\;}
$$

核 $|x-y|^{\alpha-n}$ 在原点是**可积奇点**（$\alpha>0$），所以没有主值问题，是普通（广义）积分。它把「积一次」推广成「积 $\alpha$ 次」——分数次积分之名由此而来。

拆解：

- **第一步，为什么说它是「积 $\alpha$ 次」**：Fourier 变换后 $I_\alpha$ 是乘子 $c|\xi|^{-\alpha}$（配常数），而「积分一次」对应乘 $1/\xi$（微分 $\partial$ 对应乘 $2\pi i\xi$）——于是「乘 $|\xi|^{-\alpha}$」就是「$\alpha$ 阶积分」。
- **第二步，$I_\alpha$ 改善可积性（$L^p\to L^q$）**：积分「抹平」了函数，使可积性变好。**Hardy–Littlewood–Sobolev 定理（HLS）** 给出精确的改善幅度：
$$
\left\|I_\alpha f\right\|_{L^q}\le C\,\|f\|_{L^p}, \qquad \frac1q=\frac1p-\frac{\alpha}{n},\quad 1<p<\frac{n}{\alpha}.
$$
  即「$p$ 可积的函数，$\alpha$ 次积分后变成 $q=\frac{np}{n-\alpha p}$ 可积」。指数关系 $\frac1q=\frac1p-\frac\alpha n$ 是维数分析（scaling）唯一允许的形式。
- **第三步，临界指数为什么是 $p=n/\alpha$**：当 $p=n/\alpha$ 时 $q=\infty$，但此时定理只在**弱**意义下成立（$L^{p}\to L^{n/(n-\alpha p)}$ 的弱型，对 $p=n/\alpha$ 是弱 $L^\infty$ 即 BMO 前身）。等号处的「差一点」正是 Sobolev 嵌入理论里临界指数的原始形态。
- **第四步，它和前面一切如何连接**：$I_\alpha$ 的核不满足 C–Z 的 Hörmander 条件（核无奇性、不零均值），所以 C–Z 流水线不管它；它的 $L^p\to L^q$ 有界性要用**非对角 Marcinkiewicz 插值**（第 4 篇的升级版）——从弱 $(1, n/(n-\alpha))$ 型 + $L^\infty$ 有界插出全部中间指数。<span class="marginnote">HLS 定理与 Sobolev 嵌入 $W^{1,p}\hookrightarrow L^{p^*}$（$1/p^*=1/p-1/n$）互为表里：$\alpha=1$ 时 $I_1$ 几乎是「一阶积分」，HLS 就是分数阶 Sobolev 嵌入的原型。而分数阶拉普拉斯 $(-\Delta)^{\alpha/2}$ 与 $I_\alpha$ 互为逆算子——「分数次积分」与「分数阶导数」是同一枚硬币。</span>

## 5 小结

- **Riesz 变换**：$R_j$，乘子 $-i\xi_j/|\xi|$，$n=1$ 时退化为 Hilbert 变换；满足 $\sum_jR_j^2=-I$、反自伴、交换、旋转协变。
- **$L^p$ 有界**：核满足 size + 零均值 + Hörmander ⟹ C–Z 流水线 ⟹ $1<p<\infty$；可当「微分替代品」用于梯度估计与 Sobolev。
- **Mikhlin–Hörmander 乘子定理**：尺度条件 $\left|\partial^\alpha m(\xi)\right|\le C|\xi|^{-|\alpha|}$ ⟹ $L^p$ 有界；点态有界不够，光滑性是命门。
- **分数次积分** $I_\alpha$：可积奇核，Fourier 乘子 $|\xi|^{-\alpha}$，把积分推广到 $\alpha$ 阶。
- **HLS 定理**：$\|I_\alpha f\|_{L^q}\le C\|f\|_{L^p}$，$\frac1q=\frac1p-\frac\alpha n$；与 Sobolev 嵌入、分数阶拉普拉斯互为表里。

在下一节，我们把视角彻底拉高，用**特征标与 Pontryagin 对偶**把本专题全部篇章——Fourier 级数、Fourier 变换、Haar 测度——统一成一个普适框架，并为「从极限到大模型」的调和分析之旅画上句号。
