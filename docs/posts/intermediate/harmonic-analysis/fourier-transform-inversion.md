---
title: Fourier 变换与反演公式
date: 2026-08-11
---

# Fourier 变换与反演公式

<div class="epigraph">
<p>把一个函数变成它的 Fourier 变换，就像把它从一个剧场搬进另一个剧场：节目单换了，剧场里上演的是同一个戏。</p>
<footer>—— 依据斯坦与沙卡尔奇《Fourier 分析导论》的思路（E. M. Stein, R. Shakarchi）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 调和分析 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 Fourier 变换开始

第 1 讲我们处理周期函数——把 $2\pi$-周期信号分解成离散频率 $e^{inx}$ 的和。但真实世界几乎从不「恰好周期」：一段语音有头有尾，一张图像有边界，一个事件在时间轴上只出现一次。**Fourier 变换** 把级数里的离散指标 $n\in\mathbb{Z}$ 换成连续频率 $\xi\in\mathbb{R}$，把「和」换成「积分」，于是能处理一切「不太离谱」的函数。

这还不是全部。Fourier 变换最迷人的地方在于它**保结构**：微分变成乘一个数，卷积变成普通乘积，能量守恒（Plancherel）。这意味着一个看似高深的算子方程，经过 Fourier 变换常常变成一列普通的代数方程——**把问题搬到频率域，难的问题变简单了。** 后面 Poisson 求和、采样定理、Paley–Wiener，乃至整个信号处理与深度学习的频域操作，全都站在这条「搬剧场」的智慧之上。<span class="marginnote">在深度学习里，这个「搬剧场」被用到极致：卷积网络里昂贵的空域卷积，在频域就是逐点相乘；而 LLM 长上下文的注意力稀疏化、图像超分辨率里的频域先验，无不是「先变到频率域处理，再变回来」的现代回响。</span>

## 1 定义与两个记号约定

对 $f\in L^1(\mathbb{R}^d)$，定义 **Fourier 变换（Fourier transform）**：

$$
\widehat f(\xi)=\int_{\mathbb{R}^d} f(x)\,e^{-2\pi i x\cdot \xi}\,dx,
\qquad \xi\in\mathbb{R}^d.
$$

这里用了 Stein–Shakarchi 的**归一化约定**：指数里系数取 $2\pi$，不带任何额外常数。它带来的直接好处是反演公式没有常数因子（见第 3 节），Plancherel 也是等距。代价是「频率」按每 $1/2\pi$ 一个单位计量——习惯就好。<span class="marginnote">另一种通行约定（如许多物理书）把 $e^{-i\omega x}$ 写进定义、把 $1/2\pi$ 留给反演。约定不同，常数不同，但定理的<strong>内容</strong>一字不差。读任何文献前先查它的约定——这是傅里叶学子的第一戒律。</span>

**重点：** 对 $L^1$ 函数，$\widehat f$ 一定存在（被 $\|f\|_1$ 控制）且**连续、在无穷远处趋于零**（Riemann–Lebesgue 引理）。但若想「来去自如」——变换再反演——$L^1$ 不够用，我们需要一个更精致的舞台。

## 2 Schwartz 空间：Fourier 变换的天然家园

**Schwartz 函数空间（space of rapidly decreasing functions）** $\mathcal{S}(\mathbb{R}^d)$ 由所有这样的光滑函数组成：对一切多重指标 $\alpha,\beta$，
$$
\sup_{x}\left|x^{\alpha}\,\partial^{\beta} f(x)\right| < \infty.
$$

直观地说：$f$ 光滑，且无论求多少阶导、乘多少次多项式，都仍然快速衰减。<span class="marginnote">「快速衰减」的确切含义是：$|f(x)|\le C_{k}(1+|x|)^{-k}$ 对一切 $k$ 成立。比如 $e^{-x^2}$、$e^{-|x|^2}$、$\frac{1}{1+x^2}$ 的伙伴们都在 $\mathcal{S}$ 里；而 $e^{-x}$ 在 $+\infty$ 方向不够快，不是。</span>

**为什么非选它不可？** 因为 $\mathcal{S}$ 同时容纳了微分、乘法、积分三大动作，且 Fourier 变换把它们完美地互相交换：

$$
\widehat{(\partial^{\alpha} f)}(\xi) = (2\pi i\xi)^{\alpha}\,\widehat f(\xi), \qquad
\widehat{(x^{\alpha} f)}(\xi) = \left(\frac{1}{-2\pi i}\right)^{|\alpha|}\partial^{\alpha}\widehat f(\xi),
$$

以及平移/调制/伸缩：
$$
\widehat{f(\cdot-h)}(\xi)=e^{-2\pi i h\cdot\xi}\widehat f(\xi),\qquad
\widehat{e^{2\pi i x\cdot h}f(x)}(\xi)=\widehat f(\xi-h),\qquad
\widehat{f(\lambda \cdot)}(\xi)=\lambda^{-d}\widehat f(\xi/\lambda).
$$

**重点：** Fourier 变换把 $\mathcal{S}$ 映到自身（$\mathcal{S}\to\mathcal{S}$ 双射），把「微分算子」变成「乘多项式」，把「乘多项式」变成「微分算子」。这正是它作为「搬剧场」的数学保证——在 $\mathcal{S}$ 里，一切操作都闭环。

**辨析｜易错点：** 不要把「$\widehat f$ 存在」与「$\widehat f$ 可反演」混为一谈。$L^1$ 保证存在与连续性，但 $L^1$ 的 Fourier 变换不一定可积，反演公式需要额外的可积条件或 $\mathcal{S}$ 框架。这也是为什么现代理论先在 $\mathcal{S}$ 上建立反演，再用**稠密性 + 连续延拓**把它推广到 $L^2$——下一节的两个定理正是这个套路。

## 3 公式解析：Fourier 反演公式

$$
\boxed{\;f(x)=\int_{\mathbb{R}^d}\widehat f(\xi)\,e^{2\pi i x\cdot\xi}\,d\xi\;},\qquad f\in\mathcal{S}(\mathbb{R}^d).
$$

逐项拆解：

- **第一步，为什么右边的指数是正号**：$\widehat f$ 里是 $e^{-2\pi i x\cdot\xi}$（把函数「拆」到频率域），反演就要用 $e^{+2\pi i x\cdot\xi}$（把频率「拼」回位置域）。正负号是拆与拼的方向，就像「编码」与「解码」互逆一样自然。
- **第二步，对称性从哪来**：由定义，$f$ 的变换是 $\widehat f$，而 $\widehat f$ 再变换（把 $x$ 换成 $-x$）恰好回到 $f$：
$$
\widehat{\widehat f}(-x)=f(x).
$$
  这就是为什么反演公式里「几乎看不见常数」——正是 $2\pi$ 归一化的红利。
- **第三步，证明的抓手（用高斯函数暖场）**：设 $G(x)=e^{-\pi|x|^2}$。它是唯一「形状不变」的：$\widehat G(\xi)=e^{-\pi|\xi|^2}$（自对偶）。对高斯函数，反演公式可以直接算出。于是对一般 $f\in\mathcal{S}$，把它和高斯函数卷起来得到 $f*G_\varepsilon$（$G_\varepsilon$ 是高斯族的伸缩，是「好核」），再让 $\varepsilon\to0$——两个方向（$f*G_\varepsilon\to f$ 与 $\widehat{f*G_\varepsilon}$ 反演回来）同时收敛，用密度与极限交换把反演公式「顶」出来。<span class="marginnote">高斯的自对偶是 Fourier 分析里最珍贵的巧合之一：它在量子力学里对应「位置—动量」的最小不确定态，在信号处理里是「时频积最小」的 Gabor 原子，在深度学习的初始化里是权重分布的天然先验——同一张脸在不同剧场反复登场。</span>
- **第四步，它实际在说什么**：反演公式宣告 $f\mapsto\widehat f$ 是一一对应（对 $\mathcal{S}$ 与后来的 $L^2$），且给出了**显式逆**。从此「函数」与「它的谱」是同一个东西的两种形态，不存在信息丢失。

## 4 Plancherel 定理：能量守恒的连续版本

第 1 讲的 Parseval 恒等式在连续世界升级成 **Plancherel 定理**：

$$
\|f\|_{L^2(\mathbb{R}^d)} = \|\widehat f\|_{L^2(\mathbb{R}^d)}, \qquad f\in L^2(\mathbb{R}^d).
$$

Fourier 变换是 $L^2$ 上的**等距同构**（西算子，乘以一个相位常数）。证明路线教科书式经典：先在 $\mathcal{S}$ 上算
$$
\|f\|_2^2 = \int f(x)\overline{f(x)}dx = \int f(x)\overline{\widehat{\widehat f}(-x)}\,dx = \int \widehat f(\xi)\overline{\widehat f(\xi)}d\xi=\|\widehat f\|_2^2,
$$
再因为 $\mathcal{S}$ 在 $L^2$ 中稠密、线性算子连续，延拓到整个 $L^2$。<span class="marginnote">注意延拓步骤的微妙：$L^1\cap L^2$ 上的定义（积分）与 $L^2$ 极限的定义（用 $\mathcal{S}$ 逼近）必须相容，且最后一步需要对「$f$ 的变换定义」做解释——教科书把这叫「$\mathcal{S}$ 上的等距可唯一连续延拓」。理解延拓，比记住定理本身更值钱。</span>

Plancherel 是 Parseval 的升华，也是所有「$L^2$ 理论」的枢纽：它让「谱分解」成为一个有内积结构的酉几何，微分局、自伴算子、拟微分算子的谱理论全部从这里生长。

## 5 卷积与乘积：操作两地的「翻译表」

Fourier 变换把最常用的两个操作完美翻译：

$$
\widehat{f*g}(\xi)=\widehat f(\xi)\,\widehat g(\xi), \qquad
\widehat{f\cdot g}(\xi)=\widehat f*\widehat g(\xi).
$$

**卷积在时域、乘积在频域**——这是整个 Fourier 分析「方法论」的引擎。例如解常系数线性微分方程 $P(D)f=g$：两边取变换，微分算子 $P(D)$ 变成乘多项式 $P(2\pi i\xi)$，方程变成纯代数的 $P(2\pi i\xi)\widehat f(\xi)=\widehat g(\xi)$，解出 $\widehat f$ 再反演即得 $f$——整条 PDE 求解的「Green 函数法」，本质就是这张翻译表的连续使用。<span class="marginnote">这句话的后半句（乘积变卷积）也要记住：它说明「在频域相乘」的代价是「在时域做卷积」——深度学习里的「频域滤波」「谱图神经网络」「Winograd 卷积加速」无一不是在这张翻译表上做文章。</span>

加上前面的微分规则，我们就拥有了一张「时域 ⟷ 频域」的完整翻译表：微分 ⟷ 乘 $2\pi i\xi$；卷积 ⟷ 乘积；能量 ⟷ 能量。**凡是涉及微分、卷积、能量的难题，搬去频域处理，常常就是几步代数。**

一条备忘：这张表在 $\mathcal{S}$ 上验证是最省事的——所有对象可积、可微、快速衰减，交换积分与微分无需任何辩解。这也正是教科书「先在 $\mathcal{S}$ 上建立全套恒等式，再用稠密性延拓」策略的全部意义：**把最苛刻的正规性留给 $\mathcal{S}$，把最广泛的应用留给 $L^2$。**

## 6 小结

- **Fourier 变换** $\widehat f(\xi)=\int f(x)e^{-2\pi ix\cdot\xi}dx$，$L^1$ 上存在且连续、无穷远趋零；$\mathcal{S}$ 是其自足的家园。
- **Schwartz 空间** 同时兼容微分、乘法、积分，Fourier 变换在它上面是对合（$\widehat{\widehat f}(-x)=f(x)$）。
- **反演公式** $f(x)=\int\widehat f(\xi)e^{2\pi ix\cdot\xi}d\xi$：用高斯自对偶 + 好核密度逼近证明。
- **Plancherel**：$\|f\|_2=\|\widehat f\|_2$，$L^2$ 等距同构；由 $\mathcal{S}$ 稠密连续延拓。
- **翻译表**：微分↔乘子、卷积↔乘积、能量↔能量——这是 Fourier 方法论的引擎。

在下一节，我们把「周期世界」与「直线世界」用一座桥连起来：**Poisson 求和公式**——它说，把函数在所有整数点上加起来，等于把它频谱在所有整数点上加起来；而这座桥的另一端，站着香农的采样定理。

（补一条自测题：用第 5 讲的高斯自对偶验证 $\widehat{e^{-\pi x^2}}(\xi)=e^{-\pi\xi^2}$，再用它验算 $L^1$ 反演公式在 $x=0$ 处的数值——这能一次性检验你对「约定」与「归一化」的理解是否到位。）
