---
title: 热方程与热核（抛物正则性、热核渐近展开、Li–Yau 估计）
date: 2026-08-07
---

# 热方程与热核（抛物正则性、热核渐近展开、Li–Yau 估计）

<div class="epigraph">
<p>「万物皆流（πάντα ῥεῖ）。」</p>
<footer>—— 赫拉克利特（Heraclitus）</footer>
</div>

<div class="article-byline">
<p>第二级 · 几何分析 ｜ Peter Li《Geometric Analysis》Ch. 4（Heat Kernel）｜ Jost 热方程章 ｜ 2026-08-07</p>
</div>

## 为什么从热方程开始

流形上的 Laplace 算子是静态的椭圆对象，热方程则给它装上时间：$u(t,x)$ 描述「热量如何在流形上随时间扩散」。**热方程（heat equation）与热核（heat kernel）**是几何分析最丰产的沃土——它们提供谱几何的入场券（特征值来自热核的短时展开）、Li–Yau 估计这样的精致不等式、以及把椭圆理论「时间化」的模板（调和映射热流、Ricci 流都是热方程的远亲）。

从课程体系看，本篇把第二级《PDE 引论》的抛物方程理论放到流形上，并与前面的 Hodge 理论接榫：热方程解 $e^{-t\Delta}f$ 的短时行为由曲率控制，长时行为收敛到调和函数。它也是「分析读出几何」最集中的一篇：**热核的短时渐近展开系数是曲率的显式多项式**。

<span class="marginnote">赫拉克利特的「万物皆流」用来形容几何流再贴切不过。热方程让初始数据按扩散规则演化；几何流（Ricci 流、平均曲率流）则是让几何对象本身按类似规则流动。这一篇里学到的抛物技巧——极大值原理、梯度估计、Harnack——正是 Perelman 处理 Ricci 流的核心武器。</span>

## 1 热方程与热核的定义

设 $(M,g)$ 紧致（或完备且曲率有界），**热方程（heat equation）**为

$$\partial_t u = -\Delta u, \qquad u(0,x) = f(x)$$

其中 $\Delta$ 是上一篇的正 Laplace–Beltrami 算子。由 Hodge 理论，$e^{-t\Delta}$ 是一个有界半群，**热核（heat kernel）**$K(t,x,y)$ 是它的积分核：

$$u(t,x) = e^{-t\Delta}f(x) = \int_M K(t,x,y)\, f(y)\, dV(y)$$

热核是「点热源」的扩散轮廓，具有三条基本性质：**对称性** $K(t,x,y)=K(t,y,x)$；**半群性** $K(t+s,x,y) = \int K(t,x,z)K(s,z,y)\,dz$；**归一性** $\int_M K(t,x,y)\,dy = 1$。前两条来自 $e^{-t\Delta}$ 是自伴半群，第三条来自常数函数被 $\Delta$ 湮灭。<span class="marginnote">半群性 $K_{t+s} = K_t * K_s$ 就是「热量先扩散 $t$ 秒再扩散 $s$ 秒等于直接扩散 $t+s$ 秒」，它把热核的构造归结为小时间 $t\to 0$ 的行为——这正是渐近展开的用武之地。</span>

热方程的根本定理是**抛物正则性（parabolic regularity）**：初始数据只要可积，$t>0$ 后解立即变成光滑的；更精确地，热方程解在 $t>0$ 上是 $C^\infty$ 的，且满足**抛物极大值原理**——解的最大值随时间单调不增。

### 1.1 热方程的三条来源

热方程在几何分析中有三种出场方式，值得分别记住：

- **通向调和**：$\partial_t u = -\Delta u$ 的长时极限 $u_\infty = \lim_{t\to\infty}u(t,\cdot)$ 是调和函数——热方程是「通向调和函数的斜坡」。
- **编码谱**：$e^{-t\Delta}$ 的迹 $\sum e^{-\lambda_k t} = \int K(t,x,x)dV$ 即热核迹，直接连接谱几何（见《谱几何》篇）。
- **几何流之母**：把 $\Delta$ 换成曲率算子，就是 Ricci 流的抛物骨架；换成张力算子，就是调和映射热流。热方程是「几何流之母」。

## 2 热核的短时渐近展开

欧氏空间 $\mathbb{R}^n$ 上热核是显式的高斯核 $K(t,x,y) = (4\pi t)^{-n/2} e^{-|x-y|^2/(4t)}$。流形上不再有闭式，但有 **Minakshisundaram–Pleijel 渐近展开（short-time asymptotic expansion）**：

$$K(t,x,x) \sim \frac{1}{(4\pi t)^{n/2}}\sum_{k=0}^{\infty} a_k(x)\, t^k, \qquad t \to 0^+$$

系数 $a_k(x)$ 是 $x$ 处曲率张量及其协变导数的**显式多项式**：$a_0 = 1$，$a_1 = \frac16 R(x)$（标量曲率），$a_2$ 含 $|\operatorname{Ric}|^2, |R|^2, \Delta R$ 等的组合。<span class="marginnote">这个展开的证明依赖<strong>热核的 Hadamard 构造</strong>：在测地坐标里把热核写成「高斯 × 级数」，代入热方程逐阶求解输运方程。它是「短时热核近乎欧氏、曲率作为小修正」这一直觉的严格化。</span>

**球面上的热核算例**：$S^1$（圆）上热核可用 Fourier 级数显式写出 $K(t,\theta)=\frac{1}{2\pi}\sum_k e^{-k^2t}e^{ik\theta}$；$t\to0^+$ 时它逼近周期化的高斯核，$t\to\infty$ 时趋向常数（均匀分布）。这个例子直观展示了「短时近欧氏、长时近均匀」的普遍行为，也印证了渐近展开的前两项（$t^{-1/2}$ 主项与 $R$ 修正项）。

这个展开是谱几何的发动机：对 $t$ 积分即得热核的迹

$$\sum_k e^{-\lambda_k t} = \int_M K(t,x,x)\, dV(x) = \frac{1}{(4\pi t)^{n/2}}\Big(\operatorname{Vol}(M) + \frac{t}{6}\int_M R\, dV + \cdots\Big)$$

把左边关于 $t$ 在 $0$ 附近展开并与右边比对，就能逐个读出**特征值 $\lambda_k$ 的 Weyl 型信息**与**几何量的谱不变量**（详见《谱几何》篇）。

## 3 公式解析：Li–Yau 梯度估计

热核的定量上界依赖一个深刻的估计——**Li–Yau 梯度估计（Li–Yau gradient estimate，1986）**。设 $u$ 是热方程在 Ricci 曲率下界 $\operatorname{Ric}\ge -K$（$K\ge0$）的完备流形上的正解，则

$$|\nabla \log u|^2 - \partial_t \log u \le \frac{n}{2t} + nK$$

逐项拆解：

- **第一步，看懂条件**：只需 **Ricci 曲率下界**。这是几何分析「把曲率信息注入分析不等式」的典型形式——曲率欠到某个负常数也没关系，用 $K$ 把它兜住。
- **第二步，思路：对 $|\nabla\log u|^2 - \partial_t\log u$ 作微分计算**。设 $F = |\nabla \log u|^2 - \partial_t \log u$，直接对 $u$ 的热方程与 Ricci 恒等式做运算，得到 $F$ 的演化不等式

$$\partial_t F \le -\Delta F - \frac{2}{n}F^2 + 2K F + (\text{Ricci 项})$$

这一步把「曲率下界」翻译成「$F$ 满足一个非线性抛物微分不等式」。
- **第三步，最大化原理**：在适当初/边条件下对 $F\cdot\varphi$（配一个截断函数）用极大值原理，取出全局上界，得到 $F \le \frac{n}{2t} + nK$。
- **第四步，导出 Harnack 不等式**：对时间区间 $0 < t_1 < t_2$ 与两点 $x_1,x_2$，沿连接它们的测地线积分上式即得 **Li–Yau Harnack**：

$$u(t_1,x_1) \le u(t_2,x_2)\,\Big(\frac{t_2}{t_1}\Big)^{n/2} \exp\Big(\frac{d(x_1,x_2)^2}{4(t_2-t_1)} + nK(t_2-t_1)\Big)$$

**Harnack 不等式意味着：正解在一个时空点的值控制其他时空点的值**——这给热核提供了同时向上和向下的界，是热核估计、谱下界、以及后来 Hamilton 的 Harnack 估计（Ricci 流）的原型。

## 4 热核的全局估计与谱的联系

Li–Yau 梯度估计通过对测地线积分给出热核上界，结合 Bishop–Gromov 体积比较可进一步得到：

**定理（热核上界）**：若 $\operatorname{Ric}\ge -K$，则存在只依赖 $n, K$ 的常数 $C$，使得对所有 $t>0, x,y\in M$，

$$K(t,x,y) \le \frac{C}{\operatorname{Vol}(B(x,\sqrt{t}))}\, e^{-\frac{d(x,y)^2}{4t}} \cdot (\text{时间因子})$$

这里分母是「以 $\sqrt t$ 为半径的球体积」而非欧氏的 $t^{n/2}$——体积比较把曲率的影响吸收进来，这正是几何分析「曲率控制体积、体积控制热核」链条的完成。

**Li–Yau 估计的一个典型用途**：取 $u$ 为热核 $K(t,x,\cdot)$（固定 $x$），则对任意 $y$，$K$ 满足正解估计，由此推出热核的对数梯度界与「热核在时间上的不塌缩」。配合体积比较，可证明热核在「曲率下界 + 体积上界」下的一致上界——这是《谱几何》篇中 $\lambda_1$ 下界的核心输入。

**局部化技巧**：即使没有全局 Ricci 下界，也可以在局部测地球上用截断函数得到「局部极大值原理」——这是非完备、非紧情形处理热方程的标准手法，也是「局部化」这一几何分析通用技巧的首次露面。

热核还连接了谱与几何：**特征值的每个信息都能从热核读出**——$\sum e^{-\lambda_k t}$ 的 $t\to 0^+$ 展开给 Weyl 律，$t\to\infty$ 行为给第一个正特征值的下界（由 $\lim_{t\to\infty}$ 比值定义）。所以热核研究 = 谱研究 = 几何信息研究。<span class="marginnote">「听到鼓的形状」（Mark Kac 的著名问题）说的就是这个方向的反问：热核/谱能否完全决定几何？答案大体不能（Milnor 的例子），但短时展开与长时展开合起来仍给出惊人的谱不变量——详见《谱几何》篇。</span>

### 3.1 对数 Sobolev 不等式

**对数 Sobolev 不等式（logarithmic Sobolev inequality, Gross 1975）** 是 Li–Yau 的另一面：

$$\int u^2\log u^2 \le \frac{2n}{C}\int|\nabla u|^2 + \|u\|_2^2\log\|u\|_2^2 + (\text{常数})$$

它是「熵与 Dirichlet 能量」之间的不等式，与热核的高斯上界、谱隙（spectral gap）互相等价。在当代，对数 Sobolev 与最优传输、扩散模型、以及「大模型时代」的概率侧（采样与去噪的熵界）直接挂钩——几何分析又一次在数据科学中找到回响。

**辨析｜易错点：** Li–Yau 估计要求热解 **正**；一般符号（有正有负）的解没有对数梯度。非紧流形上还要「解在无穷远适当地衰减」以保证积分与极大值论证合法。

**术语速查**：

| 记号 / 术语 | 含义 | 要点 |
| --- | --- | --- |
| 热方程 | $\partial_t u = -\Delta u$ | 通向调和、编码谱、几何流之母 |
| 热核 $K(t,x,y)$ | $e^{-t\Delta}$ 的积分核 | 对称、半群、归一 |
| 抛物正则性 | $t>0$ 后解立即光滑 | 抛物极大值原理 |
| 短时渐近展开 | $K(t,x,x)\sim(4\pi t)^{-n/2}\sum a_k t^k$ | $a_1=\frac16 R$，系数是曲率多项式 |
| Li–Yau 梯度估计 | $|\nabla\log u|^2 - \partial_t\log u \le \frac{n}{2t}+nK$ | 需 $\operatorname{Ric}\ge-K$ 且 $u>0$ |
| Harnack 不等式 | 正解一点控制他点 | 热核上下界 |
| 对数 Sobolev | $\int u^2\log u^2 \lesssim \int\|\nabla u\|^2 + \cdots$ | 与谱隙、熵、最优传输相连 |

## 5 小结

- **热方程与热核**：$e^{-t\Delta}f = \int K(t,x,y)f(y)dy$，对称、半群、归一三性质。
- **抛物正则性 + 极大值原理**：$t>0$ 后解立即光滑，最大值单调不增。
- **短时渐近展开** $K(t,x,x)\sim(4\pi t)^{-n/2}\sum a_k(x)t^k$：系数是曲率多项式，驱动谱几何。
- **Li–Yau 梯度估计**：Ricci 下界 ⇒ $|\nabla\log u|^2 - \partial_t\log u \le \frac{n}{2t}+nK$ ⇒ Harnack 不等式。
- **热核上界** = 体积比 + 高斯核，由 Bishop–Gromov 与 Li–Yau 联手得出。

在下一节，我们把热方程背后的分析工具系统化——**Sobolev 空间与流形上的 PDE 工具**：从 Sobolev 嵌入、Moser 迭代到 De Giorgi–Nash 正则性，这些是「解有多好」的通用答案。
