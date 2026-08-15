---
title: Fourier 逼近与 Gibbs 现象
date: 2026-08-07
---

# Fourier 逼近与 Gibbs 现象

<div class="epigraph">
<p>自然之书是用数学的语言写成的，其字母是三角形、圆形和其他几何图形。</p>
<footer>—— 伽利略 · 伽利雷（Galileo Galilei, 《试金者》1623）</footer>
</div>

<div class="article-byline">
<p>第二级 · 函数逼近论 ｜ E. Ward Cheney, Introduction to Approximation Theory, §5.1–5.3 ｜ 2026-08-07</p>
</div>

## 为什么从 Fourier 逼近开始

前面几篇研究的逼近函数是多项式。但有一类「天然的函数族」被多项式遗漏了：**周期函数**。声音是空气压力的周期振动，交流电是正弦波，信号处理里的频率分析建立在三角波上。多项式只能在有界区间上工作，面对周期现象则显得笨拙——而三角多项式 $\sum (a_k \cos kx + b_k \sin kx)$ 天生就是周期函数。Fourier 逼近用三角多项式近似周期函数，它的部分和正是傅里叶级数。这个主题既是逼近论的一章，也是整个调和分析与信号处理的入口——第二级《调和分析》、第六级《信号与系统》与一切频域方法都从这里出发；而它最著名的一道裂缝——**Gibbs 现象**——生动展示了「逐点收敛」与「一致收敛」的深刻差别。

## 1 三角多项式与函数空间

在 $2\pi$ 周期连续函数空间 $C_{2\pi}$ 中，候选子集取**三角多项式**：

$$
T_n(x) = \frac{a_0}{2} + \sum_{k=1}^{n} (a_k \cos kx + b_k \sin kx)
$$

$T_n$ 是 $n$ 阶三角多项式，由 $2n+1$ 个参数决定。为什么写 $a_0/2$ 而不是 $a_0$？为了让系数公式整齐划一（见下）。

关键的结构是**正交性**：在区间 $[-\pi, \pi]$ 上，函数系 $\{1, \cos kx, \sin kx\}_{k=1}^\infty$ 满足

$$
\int_{-\pi}^{\pi} \cos kx\, \cos \ell x\, dx = \pi \delta_{k\ell}, \quad
\int_{-\pi}^{\pi} \sin kx\, \sin \ell x\, dx = \pi \delta_{k\ell}, \quad
\int_{-\pi}^{\pi} \cos kx\, \sin \ell x\, dx = 0
$$

（以及 $\int_{-\pi}^{\pi} 1\, dx = 2\pi$）。<span class="marginnote">这套正交关系是上一篇文章「正交基 → 逐项投影」思想在三角世界的复现。$1, \cos kx, \sin kx$ 是 $L^2[-\pi,\pi]$ 的一组正交基，Fourier 系数就是 $f$ 在这组正交基上的投影分量——Pythagoras 定理同样保证部分和在 $L^2$ 意义下最优。</span>

## 2 Fourier 系数与部分和

利用正交性，$f \in L^2[-\pi,\pi]$ 的 **Fourier 系数（Fourier coefficients）** 是

$$
a_k = \frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\cos kx\, dx, \qquad
b_k = \frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\sin kx\, dx
$$

于是第 $n$ 个**部分和（partial sum）** $S_n f(x) = \frac{a_0}{2} + \sum_{k=1}^{n}(a_k\cos kx + b_k\sin kx)$。把系数代回去，可以写成与核的卷积：

$$
S_n f(x) = \frac{1}{\pi}\int_{-\pi}^{\pi} f(t)\, D_n(x - t)\, dt, \qquad
D_n(u) = \frac{1}{2} + \sum_{k=1}^{n} \cos ku = \frac{\sin\left((n+\frac12)u\right)}{2\sin(u/2)}
$$

$D_n$ 是 **Dirichlet 核**。它像一个「聚焦透镜」：卷积 $S_n f$ 是 $f$ 被 $D_n$ 模糊后的结果，$n$ 越大透镜越锐利。$S_n f$ 在 $L^2$ 意义下收敛到 $f$（Plancherel 定理），这正是正交投影的直接结论——但**逐点行为**要微妙得多。

### 一个可算的例子：锯齿波的 Fourier 系数

设 $f(x) = x$（$x \in (-\pi, \pi)$，再做周期延拓）。它是奇函数，故 $a_k = 0$；正弦系数用分部积分：

$$
b_k = \frac{1}{\pi}\int_{-\pi}^{\pi} x \sin kx\, dx = \frac{2}{\pi}\int_0^\pi x\sin kx\, dx = \frac{2(-1)^{k+1}}{k}
$$

于是 $f$ 的 Fourier 级数是 $2\sum_{k=1}^{\infty}\frac{(-1)^{k+1}}{k}\sin kx$。三个观察值得记住：

- **系数以 $1/k$ 衰减**：因为周期延拓后的 $f$ 在 $x = \pm\pi$ 处有跳跃（从 $-\pi$ 跳到 $\pi$），间断函数的系数只能代数衰减。对照之下，光滑函数的系数指数衰减。**系数衰减速度 = 函数隐藏光滑性的指纹**，这是谐波分析反复用到的判据。
- **$\pi$ 的级数白送**：令 $x = \pi/2$，级数变成 $1 - \frac13 + \frac15 - \cdots = \frac{\pi}{4}$——著名的 Leibniz 级数，分文不花地从系数里掉出来。
- **部分和的最大值超出锯齿上界**：在跳跃 $x = \pm\pi$ 附近，$S_n f$ 过冲约 $8.95\%$——这正是下一节要计算的 Gibbs 现象。

## 3 收敛定理：Dirichlet 条件与 Fejér 平均

逐点收敛需要比 $L^2$ 更强的条件。经典结果是 **Dirichlet 定理**：若 $f$ 在 $x$ 处可导，则 $S_n f(x) \to f(x)$；若 $f$ 在 $x$ 处有跳跃间断，则 $S_n f(x) \to \frac{f(x^+) + f(x^-)}{2}$——收敛到跳跃两侧的平均值。更一般地，分段光滑（除有限个跳跃点外连续可导）的周期函数的 Fourier 级数在每个点收敛，跳跃点收敛到左右极限的平均。<span class="marginnote">「跳跃点收敛到平均值」看似奇怪，实则自然：跳跃在分布意义下是对称的，级数分不清该偏向哪一侧，于是取了中值。这也解释了为什么 Fourier 重建方波在跃变处「骑」在中线上。</span>

但 Dirichlet 定理只给**逐点收敛**，不给一致收敛。若 $f$ 连续但不够光滑，$S_n f$ 的一致收敛不保证（甚至可能在某点发散——这是 20 世纪初的著名反例）。补救来自 **Fejér 平均**：把前 $n+1$ 个部分和做算术平均，

$$
\sigma_n f(x) = \frac{S_0 f(x) + S_1 f(x) + \cdots + S_n f(x)}{n+1}
$$

**Fejér 定理**：对连续周期函数，$\sigma_n f$ 一致收敛到 $f$。这等价于 Weierstrass 第二定理（三角多项式在 $C_{2\pi}$ 中稠密）。<span class="marginnote">Fejér 平均是「Cesàro 求和」在 Fourier 级数上的应用：把振荡的部分和「摊平」，牺牲一点锐利换取一致收敛。这是逼近论的一课——<strong>换个求和方法，收敛性会质变</strong>。同样的思想在分数阶 Fourier 平均（如 Lanczos sigma 因子）中仍是主角。</span>

## 4 公式解析：Gibbs 常数

**Gibbs 现象是 Fourier 部分和在跳跃处的「固执的过冲」。** 考察单位方波 $f(x) = \operatorname{sgn}(x)$（$x \in (-\pi, \pi)$ 上取 $\pm 1$），在 $x=0$ 处从 $-1$ 跳到 $+1$。$S_n f$ 在跳跃附近振荡，且随着 $n \to \infty$，过冲的**高度并不衰减**，而是收敛到一个严格大于 $1$ 的常数：

$$
\lim_{n \to \infty} S_n f\left(\frac{\pi}{n+1/2}\right) = \frac{2}{\pi}\int_{0}^{\pi} \frac{\sin t}{t}\, dt \approx 1.17898
$$

这个常数 $G = \frac{2}{\pi}\int_0^\pi \frac{\sin t}{t}\, dt$ 称为 **Gibbs / Wilbraham 常数**。拆解三步：

- **第一步，过冲来源是 Dirichlet 核的旁瓣**：$S_n f$ 是 $f$ 与 $D_n$ 的卷积。$f$ 的跃变把 $D_n$ 的主瓣与旁瓣一起卷进来，旁瓣积分累积出一个固定大小的「多余隆起」，与 $n$ 无关。
- **第二步，积分定标**：让 $n \to \infty$，$S_n f$ 在 $x = \pi/(n+1/2)$（第一个极值点）处的取值趋于 $\frac{2}{\pi}\int_0^\pi \frac{\sin t}{t}dt$。$\int_0^\pi \frac{\sin t}{t}dt$ 是 **正弦积分函数** $Si(\pi)$，数值约 $1.8519$，乘 $2/\pi$ 得 $G \approx 1.17898$。高出自适应上限 $1$ 约 $0.179$。
- **第三步，换算成跳跃高度**：方波跳跃高度是 $2$（从 $-1$ 到 $+1$），过冲 $0.179$ 正是跳跃高度的 $8.95\%$。一般地，**任一跳跃高度 $J$ 处，部分和过冲约 $0.0895\,J$**，欠冲同理，且不随 $n$ 消失。

Gibbs 常数 $G \approx 1.17898$ 是数学物理里的著名常数，也是「点态收敛却不一致收敛」最直观的样本。

## 5 Gibbs 现象的意义与应对

Gibbs 现象不止是趣味数学，它有真实的工程后果：心电图波形、音频瞬态、图像边缘，处处是「跳跃」。用 Fourier 部分和重建这些信号时，边缘附近会冒出不收敛的过冲条纹——这在图像里表现为边缘振铃（ringing artifact），在音频里表现为瞬态前的 pre-echo。<span class="marginnote">MRI 图像边缘的振铃、视频压缩在硬边缘处的鬼影，本质都是 Gibbs 现象。它是「用正弦波逼近尖锐特征」不可避免的代价——因为任何部分和都是连续的三角多项式，不可能真正复制一个跳跃，只能在附近拼命震荡。</span>

应对思路有四条：**一是接受并后处理**——在重建后对边缘做锐化或窗函数平滑；**二是换核**——用 Fejér 核或其他正核替换 Dirichlet 核，牺牲锐利换取无过冲（代价是分辨率降低）；**三是加窗**——对 Fourier 系数乘 sigma 因子（Lanczos、Hamming 窗），这是信号处理的常规操作；**四是换基**——用分片多项式（样条）或小波，让「跳跃」被局部基函数直接表达，而非靠全局正弦波拼凑。第四条正是下一篇（样条）的动机之一。

## 6 术语速查表

| 术语 | 英文 | 一句话定义 |
| --- | --- | --- |
| Fourier 系数 | Fourier coefficients | $a_k,b_k = \frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\cos kx,\sin kx\, dx$ |
| 部分和 | partial sum | $S_n f$，$f$ 的 Fourier 级数前 $n$ 阶截断 |
| Dirichlet 核 | Dirichlet kernel | $D_n(u)=\frac{\sin((n+1/2)u)}{2\sin(u/2)}$ |
| Plancherel 定理 | Plancherel theorem | 部分和 $L^2$ 收敛且保持能量 |
| Fejér 平均 | Fejér mean | $\sigma_n f = (S_0f+\cdots+S_nf)/(n+1)$ |
| Dirichlet 定理 | Dirichlet theorem | 逐点收敛；跳跃点收敛到左右极限平均 |
| Gibbs 常数 | Gibbs constant | $G = \frac{2}{\pi}\int_0^\pi\frac{\sin t}{t}dt \approx 1.17898$ |
| 过冲 / 欠冲 | overshoot / undershoot | 跳跃处部分和超出 / 低于极限的部分 |
| 振铃 | ringing artifact | 图像或信号边缘的 Gibbs 条纹 |

## 7 小结

- 三角多项式 $\frac{a_0}{2} + \sum(a_k\cos kx + b_k\sin kx)$ 是周期逼近的天然候选；$\{1,\cos kx,\sin kx\}$ 在 $L^2[-\pi,\pi]$ 正交。
- Fourier 系数是投影系数，部分和 $S_n f$ 是最佳 $L^2$ 三角逼近；**Plancherel 定理**保证 $L^2$ 收敛。
- 逐点收敛需要光滑性（Dirichlet 定理，跳跃点收敛到左右极限平均）；**Fejér 平均**对连续周期函数一致收敛。
- **Gibbs 现象**：跳跃处部分和过冲约跳跃高度的 $8.95\%$，不随 $n$ 消失；**Gibbs 常数** $G = \frac{2}{\pi}\int_0^\pi\frac{\sin t}{t}dt \approx 1.17898$