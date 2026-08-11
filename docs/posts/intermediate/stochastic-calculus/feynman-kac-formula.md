---
title: Feynman-Kac 公式
date: 2026-08-11
---

# Feynman-Kac 公式

<div class="epigraph">
<p>一个偏微分方程的解析解，竟能化作一群随机轨道上期望的极限——物理学与概率论在此合流。</p>
<footer>—— 马克 · 卡茨（Mark Kac）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 随机分析（Itô 微积分） ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 Feynman-Kac 开始

至此，我们已经拥有了完整的随机微积分工具箱。是时候展示它最漂亮的副产物了：**随机分析与偏微分方程（PDE）的等价性**。Feynman-Kac 公式断言：一类扩散型 PDE 的解，可以写成某个随机微分方程解的期望——「求期望」代替「求导数」，路径积分代替偏微分。

这条桥到底有多宽？它意味着：**每一个带扩散的二阶抛物型方程背后都藏着一个随机过程；每一个扩散的期望都对应一个 PDE。** 热方程的解是布朗运动的位置期望，Black-Scholes 方程的解是复制组合的价值，量子力学虚时间下的薛定谔方程是路径积分——它们共享同一个数学结构。<span class="marginnote">Kac 于 1949 年发表此公式时致敬了 Feynman 的路径积分思想；<strong>它把「在路径空间上求积分」翻译成「在概率空间上求期望」</strong>。这也是数值方法（蒙特卡洛、深度 BSDE）的理论源头。</span>

## 1 定理陈述：PDE 与期望的同一枚硬币

**Feynman-Kac 公式**：设 $X$ 满足 SDE $dX_t = b(X_t)\,dt + \sigma(X_t)\,dB_t$，令算子

$$\mathcal{L} = \frac12 \sigma(x)^2 \frac{\partial^2}{\partial x^2} + b(x)\frac{\partial}{\partial x}$$

为**无穷小生成元（infinitesimal generator）**。若 $u(t,x)$ 是终值问题

$$\frac{\partial u}{\partial t} + \mathcal{L} u - r u = 0, \qquad u(T, x) = g(x)$$

的足够光滑的解，则

$$u(t, x) = E\Big[e^{-\int_t^T r(X_s)\,ds} \, g(X_T) \;\Big|\; X_t = x\Big].$$

**重点：PDE 的解「就是」条件期望**——算 $u$ 可以不碰任何偏导，只需模拟扩散路径并取平均（这就是蒙特卡洛定价的合法依据）。<span class="marginnote">$r$ 是折现率。取 $r = 0$、$b = 0$、$\sigma = 1$ 时，$\mathcal{L} = \tfrac12\partial_{xx}$，公式变成<strong>热方程</strong>：$u(t,x) = E[g(B_T) \mid B_t = x]$——热传导的每一点温度，都是初始数据在随机轨道上的平均。</span>

## 2 公式解析：$e^{-r(T-t)}$ 从哪来，$\mathcal{L}$ 往哪去

把公式的证明拆成三步，你就看清了「为什么 PDE 的系数恰是 $\mathcal{L}$」：

**第一步，构造折现量**：设 $M_s = e^{-r(s-t)} u(s, X_s)$（设 $r$ 为常数简化），对它用 Itô 公式：
  $dM_s = e^{-r(s-t)}\big[u_s + \mathcal{L}u - r u\big]\,ds + e^{-r(s-t)} u_x \sigma\,dB_s$。
- **第二步，利用 PDE**：中括号里正是 $u_s + \mathcal{L}u - r u = 0$，于是漂移项消失，$M$ 成为局部鞅（鞅）。
- **第三步，取期望**：$E[M_T \mid \mathcal{F}_t] = M_t$，代入 $M_T = e^{-r(T-t)}g(X_T)$ 即得 $u(t,x) = E[e^{-r(T-t)}g(X_T)\mid X_t = x]$。

**PDE 的「偏导数系数」$\mathcal{L}$，恰好是 Itô 公式在漂移项里产生的系数**——二者互为对方的「期望」与「导数」形态。这是整座桥的承重原理。<span class="marginnote">反过来看，Feynman-Kac 也说明：<strong>任何随机微分方程的期望问题，都对应一个偏微分方程；偏导难解就换期望（蒙特卡洛），期望难算就换偏导（有限差分）</strong>——这是数值随机分析的基本策略。</span>

**辨析｜易错点：** 公式的方向。终值问题（给出 $t=T$ 的数据反推 $t$ 更早的值）对应「条件期望」；若改成初值问题（给出 $t=0$ 的初始条件正推），则是**热方程的正向**形式，对应「转移概率密度」而不是「条件期望」。方向搞反，$t$ 与 $T$ 就会写错。

## 3 核心推论：生成元是「期望的瞬时变化率」

对任意光滑 $f$，生成元有一个极其直观的刻画：

$$\mathcal{L} f(x) = \lim_{h \downarrow 0} \frac{E[f(X_{t+h}) \mid X_t = x] - f(x)}{h}.$$

**重点：生成元度量「过程从当前位置出发，期望值瞬时变化的速度」**——它就是随机世界的「导数的期望版」。伊藤公式、Kolmogorov 方程、HJB 方程全部从这里出发。<span class="marginnote">这也解释了扩散项的系数为什么是 $\tfrac12\sigma^2$ 而非 $\sigma^2$：<strong>二阶项在生成元里天然带 $1/2$，因为 $(dB)^2 = dt$ 的期望里已经有 $1/2$ 因子</strong>。</span>

## 4 应用：从热方程到量子物理到金融

**热方程**：$u_t = \tfrac12 u_{xx}$，$u(t,x) = E[g(B_T) \mid B_t = x]$——布朗运动的转移概率就是热核 $p(t,x) = \frac{1}{\sqrt{2\pi t}} e^{-x^2/(2t)}$。
- **量子力学（虚时间）**：$i\hbar \partial_t \psi = -\frac{\hbar^2}{2m}\partial_{xx}\psi + V\psi$ 取虚时间 $t \mapsto -it$ 后变成带势能项的扩散方程——**薛定谔方程是「虚时间」下的 Feynman-Kac**，这正是路径积分与随机分析共享的天鹅脖子。<span class="marginnote">费曼的路径积分把传播子写成「所有路径的振幅叠加」，Kac 证明「适当旋转到虚时间后」，振幅变成概率、叠加变成期望——<strong>二者不是比喻，是同一算式的两支</strong>。</span>
- **金融（折现期望）**：Black-Scholes 定价「风险中性折现期望」正是 $r$ 版 Feynman-Kac 的直接应用——下一节我们会亲手写下它。

## 5 从 Feynman-Kac 到 Kolmogorov 方程

Feynman-Kac 的桥有一个「逆向」版本。设 $p(t, x; T, y)$ 是扩散 $X$ 的**转移密度**（从 $t$ 时刻的 $x$ 转移到 $T$ 时刻的 $y$ 的概率密度），则 $p$ 关于「后变量」$(T,y)$ 满足**正向方程**（Fokker–Planck / Kolmogorov forward）

$$\partial_T p = \frac12 \partial_y^2(\sigma^2 p) - \partial_y(b p),$$

而关于「前变量」$(t,x)$ 满足**逆向方程**（Kolmogorov backward）

$$-\partial_t p = \frac12 \sigma^2 \partial_x^2 p + b\,\partial_x p.$$

**重点：转移密度就是「看得见边界的期望解」——正逆向方程是同一座 Feynman-Kac 桥的两个方向。** 物理里 $p$ 是扩散物质的浓度，金融里它是风险中性转移密度，统计力学里它是配分函数的局部化。<span class="marginnote">工程直觉：<strong>前向方程像「粒子扩散的流体力学」，逆向方程像「倒着追期望的倒向归纳」</strong>——扩散模型（DDPM）的加噪过程走前向，去噪估计走后向。</span>

## 6 例：热核就是高斯核

取 $b = 0, \sigma = 1$，逆向方程化为热方程 $u_t + \tfrac12 u_{xx} = 0$。由 Feynman-Kac：

$$u(t,x) = E[g(B_T) \mid B_t = x] = \int_{\mathbb{R}} g(y)\, \frac{1}{\sqrt{2\pi (T-t)}} e^{-(y-x)^2/(2(T-t))}\,dy.$$

看：**解积分里那个高斯核，正是布朗运动的转移密度 $p(t,x;T,y)$**。于是「用 Feynman-Kac 求期望」与「解热方程」是同一件事——前者靠模拟（蒙特卡洛），后者靠网格（有限差分），结果相同。

更进一步，**深度 BSDE 方法**把「反解 PDE 的初值」翻译成「训练神经网络拟合倒向随机微分方程的解」，本质仍是这一座桥：神经网络在路径空间上逼近条件期望，与 Itô 公式给出的漂移项严格对账。<span class="marginnote">这也是机器学习的路线图：<strong>凡是你想算的期望，都有一个 PDE 在等价地替你算；反之亦然</strong>——梯度流、生成模型、强化学习里的 Bellman 方程，全在这张对照表里。</span>

## 7 从抛物到椭圆：Dirichlet 问题的概率解法

把 $u_t$ 项去掉（时间不再出现），Feynman-Kac 的「兄弟定理」给出椭圆型 Dirichlet 问题的概率公式：设区域 $D$ 与边界函数 $g$，$\tau_D$ 是 $X$ 首次离开 $D$ 的时刻，则

$$\mathcal{L}u = 0 \text{ 于 } D, \qquad u = g \text{ 于 } \partial D$$

的解是 $u(x) = E[g(X_{\tau_D}) \mid X_0 = x]$。

直觉极美：**调和函数 = 「随机游走首次撞墙时的期望读数」。** 势论里的调和测度、Green 函数都从这个概率视角获得直觉；数值上的「随机游走法」（random walk on spheres）也由此而来——**偏微分方程的问题，交给随机模拟去解**。

**一个容易记住的口诀**：生成元 $\mathcal{L}$ 管「期望怎么演化」，PDE 系数管「概率密度怎么流动」——二者是同一个无穷小算子的两种视图。前者看期望，后者看质量；Feynman-Kac 只是把这两个视图的账目对在一起。

## 8 一座桥，三处用：把 Feynman-Kac 当接口

**应用一，解析**：PDE 有显式解时，期望也有了闭式；反之期望可算时，PDE 解也到手。热核、Black-Scholes 公式都走这条。

**应用二，数值**：PDE 难解就用蒙特卡洛（对 $g(X_T)$ 模拟取平均），期望难估就用有限差分（解 PDE）。两者由这座桥互为替身，互为检验。

**应用三，学习**：现代「神经 PDE 求解器」把 $u$ 参数化进神经网络，用「PDE 残差 + 边界数据 + 蒙特卡洛对账」联合训练——网络既学偏导（PDE 视角），也学期望（概率视角），Feynman-Kac 正是这两套损失的契约。<span class="marginnote">强化学习同样如此：<strong>Bellman 方程是离散时间的 Feynman-Kac，价值网络是它的函数近似</strong>——「期望与方程互译」的精神一以贯之。</span>

**重点：Feynman-Kac 是随机分析的「接口层」**——上游是 SDE 与 Itô 公式，下游是 PDE、数值与学习算法。理解了这座桥，随机分析就从「一个理论板块」变成了「一个工具平台」。

（尾声：Feynman-Kac 是随机分析与偏微分方程的「双边签证」——从随机到确定、从确定到随机，只需一张期望的票。它也是本专题通往 Black-Scholes 的必经之路。）

## 9 小结

- **Feynman-Kac 公式**：$u_t + \mathcal{L}u - ru = 0$ 的解 $=$ 折现条件期望 $E[e^{-\int r\,ds}g(X_T)]$。
- **无穷小生成元** $\mathcal{L} = \tfrac12\sigma^2 \partial_{xx} + b\partial_x$ 是「期望的瞬时变化率」，由 Itô 公式自然导出。
- 证明的骨架：构造 $M = e^{-r(t-s)}u(s, X_s)$ → 用 Itô 公式消漂移 → 鞅取期望。
- **热方程、虚时间薛定谔方程、Black-Scholes 方程**是同一座桥的三段路。
- 数值意义：PDE 与期望互替，蒙特卡洛与有限差分各取所需。

在下一节，我们将站在 Feynman-Kac 的桥头，写下随机分析最著名的应用——**Black-Scholes 模型与应用**。
