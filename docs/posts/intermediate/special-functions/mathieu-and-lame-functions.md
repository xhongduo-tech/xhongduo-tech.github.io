---
title: Mathieu 函数与 Lamé 函数
date: 2026-08-07
---

# Mathieu 函数与 Lamé 函数

<div class="epigraph">
<p>椭圆坐标与椭球坐标下的分离变量，孕育了特殊函数家族里最精密的两个成员。</p>
<footer>—— 埃米尔 · 马蒂厄（Émile Mathieu）与加布里埃尔 · 拉梅（Gabriel Lamé）工作精神之写照</footer>
</div>

<div class="article-byline">
<p>第二级 · 特殊函数 ｜ 王竹溪、郭敦仁《特殊函数概论》 第 8 章 ｜ 2026-08-07</p>
</div>

## 为什么从 Mathieu 与 Lamé 函数开始

前面各章的分离变量，都发生在「对称得很完美」的坐标系里：球坐标给 Legendre，柱坐标给 Bessel。可现实中的边界往往没那么圆：**椭圆柱面**的导体、**椭圆**膜、**椭球**谐振腔、**椭球坐标**下的量子阱——在这些几何里，分离变量给出的方程不再是 Bessel 或 Legendre 方程，而是一族参数随时间（自变量）周期变化的方程，即 **Mathieu 方程**；以及定义在椭球坐标上的 **Lamé 方程**。<span class="marginnote">Mathieu 方程写出来是 $y'' + (a - 2q\cos 2z)y = 0$——它看起来像简谐振子 $y'' + ay = 0$ 加上一个周期调制项 $-2q\cos 2z$。正是这个「周期系数」让它的解理论（Floquet 理论）与前面所有「常系数」方程截然不同。马蒂厄 1868 年为研究椭圆膜振动而引入它，如今却成了量子力学、粒子加速器与波动学的公共财产。</span>这两族函数标志着「特殊函数」由「简单坐标系」走向「精细坐标系」的延伸，也是通往现代物理（周期结构、带隙、稳定性图）的一道大门。

## 1 椭圆坐标与 Mathieu 方程

**椭圆坐标（elliptic coordinates）** $(\xi, \eta)$ 由

$$
x = c\cosh\xi\cos\eta, \qquad y = c\sinh\xi\sin\eta, \qquad \xi \ge 0,\ 0 \le \eta \lt  2\pi
$$

定义，其中 $c$ 是焦距。等 $\xi$ 线是共焦椭圆，等 $\eta$ 线是共焦双曲线。<span class="marginnote">$\xi\to\infty$ 时椭圆坐标退化回极坐标（$\cosh\xi \approx \sinh\xi \approx e^\xi/2$），所以 Mathieu 函数的解在极限下应回到 Bessel 函数——这条「极限回归」既是检验公式正确性的好方法，也是理解 Mathieu 函数与 Bessel 函数亲缘关系的线索。</span>在椭圆坐标下分离变量 Laplace 方程 $\nabla^2 u = 0$，设 $u = f(\xi)g(\eta)$，两个方程都有同一种形式，即 **Mathieu 方程**：

$$
\frac{d^2 y}{dz^2} + \left(a - 2q\cos 2z\right)y = 0
$$

这里 $z$ 对应 $\xi$ 或 $\eta$，$a$ 是分离常数，$q = \frac{1}{4}k^2c^2$ 正比于波数平方与焦距平方。**同一个方程同时控制两个变量，且都要求周期解**——这个双重要求是理解 Mathieu 谱的关键。

## 2 Floquet 理论：周期系数的谱

Mathieu 方程与前面所有方程的本质区别在于系数周期变化。处理这类方程的框架是 **Floquet 理论**。

**Floquet 定理**：若 $q(z)$ 以 $\pi$ 为周期（Mathieu 方程里 $\cos 2z$ 以 $\pi$ 为周期），则方程 $y'' + q(z)y = 0$ 的任意解可以写成

$$
y(z) = e^{\mu z}\, p(z), \qquad p(z + \pi) = p(z)
$$

其中 $\mu$ 称为 **Floquet 指数（特征指数）**。<span class="marginnote">这个形式的含义是：解 = 周期包络 $p(z)$ × 指数因子 $e^{\mu z}$。若 $\mu$ 为纯虚数，解在整个实轴上振荡有界；若 $\mu$ 实部非零，解指数增长——<strong>这正是物理里「禁带 / 传导带」区分的数学根源</strong>。</span>

**Mathieu 方程的谱结构**：在参数平面 $(a, q)$ 上，只有在某些特定的特征值曲线（特征值随 $q$ 变化形成的曲线族）上，方程才存在 $\pi$-周期或 $2\pi$-周期解。这些特征值记为 $a_n(q)$（$\pi$-周期偶解）、$b_n(q)$（$\pi$-周期奇解）、$a_{n+1}(q)$（$2\pi$-周期偶解）、$b_{n+1}(q)$（$2\pi$-周期奇解）。当 $q = 0$ 时回到常系数情形：$a_n(0) = b_n(0) = n^2$。<span class="marginnote">在 $q=0$ 处 $a_0 = 0$、$a_1 = b_1 = 1$、$a_2 = b_2 = 4$…… 每个 $n^2$ 退化为两条曲线。随 $q$ 增大，$a_n(q)$ 与 $b_n(q)$ 分开，两者之间的区域（$\operatorname{ce}_n$ 与 $\operatorname{se}_n$ 的特征值之间）没有周期解——这些间隔正是能带结构里的<strong>禁带</strong>。</span>

## 3 Mathieu 函数：$\operatorname{ce}$ 与 $\operatorname{se}$

在特征值上，Mathieu 方程的周期解称为 **Mathieu 函数**，分偶、奇两类：

$$
\operatorname{ce}_n(z, q) \quad \text{（偶，$\pi$-或$2\pi$-周期）}, \qquad \operatorname{se}_n(z, q) \quad \text{（奇，$\pi$-或$2\pi$-周期）}
$$

它们展开为 Fourier 级数：

$$
\operatorname{ce}_{2n}(z,q) = \sum_{r=0}^{\infty} A_{2r}^{(2n)}\cos 2rz, \qquad \operatorname{se}_{2n+1}(z,q) = \sum_{r=0}^{\infty} B_{2r+1}^{(2n+1)}\sin(2r+1)z
$$

把这样的级数代入 Mathieu 方程，比较 $\cos mz$ 的系数，得到关于系数 $A$ 的**三对角的特征值问题**——这又是「特殊函数 ⇄ 三对角矩阵」交汇的一例（与正交多项式一章呼应）。<span class="marginnote">Fourier 系数满足的递推是 $-q A_{r-2} + (a - r^2)A_r - q A_{r+2} = 0$（同类项合并后的形式）——它恰好是一个无穷三对角矩阵的特征方程。数值上，截断到有限阶再求对称三对角阵的特征值，就能高精度地算出 $a_n(q)$ 与 Fourier 系数。这就是 `scipy.special.mathieu_*` 背后的算法。</span>

**第二类解**（Mathieu 函数 $f_n, g_n$，非周期）对应同一个 $a,q$ 但线性独立的另一个解，在 $z\to\infty$ 时指数增长，通常在物理有界问题中被排除。

**归一化与正交性**：Mathieu 函数像所有本征函数族一样带有自然的正交归一关系。周期解 $\operatorname{ce}_n,\operatorname{se}_n$ 在 $[0,2\pi]$ 上关于平凡权重正交：

$$
\int_0^{2\pi}\operatorname{ce}_m(z,q)\operatorname{ce}_n(z,q)\,dz = \pi\,\delta_{mn}, \qquad
\int_0^{2\pi}\operatorname{se}_m(z,q)\operatorname{se}_n(z,q)\,dz = \pi\,\delta_{mn}
$$

偶、奇两族之间互相正交（$\int \operatorname{ce}_m\operatorname{se}_n = 0$）。**这组正交性让「把任意周期函数展开成 Mathieu 级数」成为可能**——正如 Fourier 级数把周期函数展开成 $\cos,\sin$，Mathieu 级数把「受椭圆边界约束」的周期函数展开成 Mathieu 函数，是 Fourier 展开在椭圆几何下的推广。<span class="marginnote">当 $q=0$ 时，$\operatorname{ce}_n(z,0) = \cos nz$、$\operatorname{se}_n(z,0)=\sin nz$，这组正交性就退回 Fourier 级数的正交性——再次验证「Mathieu 函数是被椭圆边界扭曲的 Fourier 基」这条直觉。</span>

## 4 公式解析：稳定性图与禁带

Mathieu 方程最重要的工程对象是**稳定性图（stability chart）**——在 $(a,q)$ 平面上标出「解有界」与「解无界」的区域。逐步拆解：

**第一步，画特征曲线**：$a_0(q), a_1(q), b_1(q), a_2(q), b_2(q), \dots$ 把 $(a,q)$ 平面划分成交替的条带。曲线之间（例如 $b_1(q) \lt  a \lt  a_1(q)$ 的下带与 $a_1 \lt  a \lt  b_2$ 的上带交替）是「有界区」，即**稳定区**。
**第二步，读出 Floquet 指数的含义**：在稳定区 $\mu$ 纯虚，解振荡有界；在不稳定区 $\operatorname{Re}\mu \neq 0$，解指数增长。**稳定/不稳定边界的转变，就是物理里「共振 / 参数共振」的阈值**。
**第三步，看一个经典场景：参变谐振子**。Mathieu 方程在 $q \neq 0$ 时描述**参数激励系统**——摆长周期变化的单摆、荡秋千时重心周期性上下移动。秋千在特定频率比下会越荡越高（$a \approx n^2$ 处出现参数共振），这正是稳定性图上「不稳定楔形」的物理后果。<span class="marginnote">荡秋千的「站-蹲-站」周期如果与摆频成 2:1 关系，就进入 Mathieu 不稳定区，振幅指数增长——这是参数共振（parametric resonance）的日常例子。在工程上，这既是机械振动要避免的危险，也是某些能量收集装置的原理。</span>
**第四步，推广到周期结构物理**：Mathieu 方程是「一维周期势」问题（如电子在周期晶格、光在周期介质的传播）的最简模型。稳定区 ↔ 导带（传播态），不稳定区 ↔ 禁带（指数衰减）。**整个固体物理里的能带概念，在 Mathieu 方程里有一个完全解析的缩影**——这是它从数学结构通往量子力学与光子晶体的桥梁。<span class="marginnote">推广到更一般的周期势（如 Kronig–Penney 模型）时，Floquet 理论依然适用：Bloch 定理 $u_k(r) = e^{ik\cdot r}p(r)$ 就是 Floquet 定理在三维周期势里的化身。Mathieu 方程因此常被用作「可解析的 Bloch 波玩具模型」，见《固体物理》能带论一章。</span>

**数值算例：小 $q$ 下特征值如何劈裂。** 用微扰论把 $q=0$ 处的简并点 $a_1(0)=b_1(0)=1$ 展开到二阶，得到 $a_1(q) = 1 + q - \frac{q^2}{8} + \cdots$ 与 $b_1(q) = 1 - q - \frac{q^2}{8} + \cdots$：两条特征曲线以 $q$ 的一次方线性分开，间距约为 $2q$，夹出的正是第一段禁带。<span class="marginnote">这与量子力学里「微扰解除简并」是同一个画面：$q=0$ 时 $\cos z$ 与 $\sin z$ 在 $a=1$ 处简并，$q\neq 0$ 后对称性不同的两支各自挪动，劈裂量正比于微扰强度 $q$。四极质谱仪的质量分辨率，本质上取决于这个禁带宽度。</span>有趣的是，线性劈裂只出现在奇数阶：对 $n=2$ 一阶项抵消，$a_2(q)$ 与 $b_2(q)$ 到 $q^2$ 才分开，所以偶数特征曲线在小 $q$ 下明显靠得更近。

**径向方程：修正 Mathieu 函数。** 前面为省事，我们把 $\xi$ 与 $\eta$ 两个方向都写成同一个 Mathieu 方程。严格地说，$\eta$ 方向（角向）是 $y'' + (a - 2q\cos 2\eta)y = 0$，而 $\xi$ 方向（径向）把 $\cos$ 换成 $\cosh$，成为**修正 Mathieu 方程** $R'' - (a - 2q\cosh 2\xi)R = 0$——余弦变双曲余弦，解也从振荡变成单调。<span class="marginnote">修正 Mathieu 函数记作 $\operatorname{Ce}_n, \operatorname{Se}_n$（大写），在 $\xi\to\infty$ 的极限下回到第一类 Bessel 函数——这与 §1 边注「$\xi\to\infty$ 退化回极坐标」完全呼应：椭圆柱外场的径向衰减由 $K_\nu$ 类的修正函数描述。</span>对谐振腔与波导这类有界问题，角向取周期 Mathieu 函数、径向取有界修正函数，两层匹配才得到完整的本征模式。

## 5 Lamé 函数：椭球坐标下的特殊函数

当分离变量发生在**椭球坐标（ellipsoidal coordinates）** 时，得到的方程叫 **Lamé 方程**。椭球坐标 $(\lambda_1, \lambda_2, \lambda_3)$ 由三个共焦二次曲面的参数刻画，在它上面分离变量 Laplace 方程，会得到形如

$$
4\sqrt{\varphi(\lambda)}\,\frac{d}{d\lambda}\left(\sqrt{\varphi(\lambda)}\,\frac{d\Lambda}{d\lambda}\right) + \left[h - n(n+1)\lambda\right]\Lambda = 0
$$

的方程，其中 $\varphi(\lambda) = (\lambda - e_1)(\lambda - e_2)(\lambda - e_3)$ 是三次多项式。它的多项式解就是 **Lamé 函数** $E_n^m(\lambda)$。<span class="marginnote">Lamé 方程的特征在于它含「三次多项式开方」的权重——这让它的解不能写成普通的超几何级数，而需要更精细的代数构造。Lamé 函数按次数 $n$ 与「族」$(K,L,M,N)$ 分类，每个 $n$ 对应 $2n+1$ 个独立的 Lamé 多项式，这个计数与球谐函数在每个 $l$ 下 $2l+1$ 个 $Y_l^m$ 完全平行。</span>

Lamé 函数与椭圆函数关系密切：当椭球退化为球（$e_1 = e_2$）时，Lamé 函数退回球谐函数；Lamé 方程在退化极限下分别给出 Legendre、Bessel 等方程。这再次展示「精细坐标系 → 退化坐标系」的谱系观。

**椭球谐波（ellipsoidal harmonics）** 是 Lamé 函数在椭球坐标下的完整体：把三个坐标方向上的 Lamé 函数乘起来，再配上相应的径向因子，就得到椭球坐标下 Laplace 方程的分离解。它们与球谐函数一一对应，只是把「球面上的 $Y_l^m$」换成「椭球面上的 Lamé 积」。**椭球谐波最大的实用价值在于描述椭球体的外部场**——地球不是一个正球，而是略扁的椭球，用椭球谐波展开地球重力场，比球谐展开更贴合几何，收敛也更快。

Lamé 函数的数值构造通常走两条路：一是把方程写成「三次多项式开方」权重下的谱问题，用多项式展开 + 矩阵特征值求解；二是利用它与 Weierstrass $\wp$ 函数的关系，把解表示成椭圆函数的乘积。第一条路数值稳定、直接面向工程；第二条路则揭示了 Lamé 函数与《椭圆积分与椭圆函数》一章的深刻联系——**两条路线共享同一个对象，正是「表示的等价性」在这一族函数上的又一次体现**。<span class="marginnote">把 Lamé 函数写成 Weierstrass $\wp$ 函数的乘积形式，是 19 世纪经典分析的高潮之一：它把「三次多项式根号下的积分」翻译成「椭圆函数的代数组合」，从而把 Lamé 方程纳入椭圆函数的统一体系。今天这一步在量子可积系统（如椭球坐标下的分离变量）里仍被反复使用。</span>

## 6 应用地图与易错点

**Mathieu 函数**：椭圆膜与椭圆柱波导的本征模式；四极质谱仪中离子的稳定轨迹（Mathieu 稳定性图是四极质量分析器设计的基本工具）；粒子加速器中的横向聚焦（Floquet 指数 ↔ 工作点）；量子受限周期势、光子晶体。<span class="marginnote">四极质谱仪（quadrupole mass filter）里，射频电压 $V\cos\omega t$ 与直流电压叠加，离子在 Mathieu 方程描述的力场中运动；只有当 $(a,q)$ 落在稳定性图上特定区域时，离子才能稳定通过到达检测器。这是「特殊函数 → 仪器设计」最直接的工程案例。</span>
**Lamé 函数**：椭球谐振腔、椭球量子点、地球物理中椭球谐波（ellipsoidal harmonics）对椭球天体引力场的展开。
**数值**：Mathieu 特征值与函数已有成熟算法（Fourier 级数截断 + 对称三对角特征值）；Lamé 多项式可用代数方法或超几何表达。

一个值得一提的退化线索：当 $q \to 0$ 时，Mathieu 方程退化回简谐振子 $y'' + ay = 0$，Mathieu 函数 $\operatorname{ce}_n(z,0) = \cos nz$、$\operatorname{se}_n(z,0) = \sin nz$——**Fourier 三角基从 Mathieu 函数里自动浮现**。反过来看，Mathieu 函数就是「被椭圆边界扭曲的 Fourier 基」。这条「$q$ 从 0 慢慢打开」的视角，比直接记住定义更能建立直觉：$q$ 小则解接近 $\cos/\sin$，$q$ 大则解被压缩进椭圆坐标的焦线附近，稳定性图也随之展开出一层层的禁带结构。<span class="marginnote">「从退化情形起步，再让参数连续变化」是理解一切带参数特殊函数的好策略：先看 $q=0$（或 $k=0$、$\nu=1/2$ 等）的已知极限，再看参数如何连续地改变函数的形态。这个视角在本专题的渐近展开与 q-级数两章还会再次派上用场。</span>

**辨析｜易错点：** 第一，Mathieu 方程是「系数随自变量周期变化」，与「解周期」是两回事——解周期只在特征曲线上出现，不要默认所有解都周期。第二，特征值 $a_n, b_n$ 在 $q>0$ 与 $q\lt 0$ 的记号含义不同（$a$ 族对应偶、$b$ 族对应奇），查表前先看定义域。第三，$\operatorname{ce}_n, \operatorname{se}_n$ 的下标 $n$ 对应 $q=0$ 时的退化阶数 $n^2$，而不是振荡次数。

## 7 小结

- **Mathieu 方程** $y'' + (a - 2q\cos 2z)y = 0$ 来自椭圆坐标下分离变量，系数周期变化是其区别于常系数方程的本质。
- **Floquet 定理** $y = e^{\mu z}p(z)$ 给出「周期包络 × 指数因子」的解结构，$\mu$ 的虚实决定有界与否。
- **Mathieu 函数** $\operatorname{ce}_n, \operatorname{se}_n$ 是特征值上的周期解，Fourier 系数满足三对角特征问题。
- **稳定性图** 把 $(a,q)$ 平面划分成稳定区（$\mu$ 纯虚，解有界）与不稳定区（$\operatorname{Re}\mu\neq0$，解指数增长），边界对应参数共振阈值；小 $q$ 下特征值从 $n^2$ 线性劈裂，偶数阶劈裂退到二阶。

在下一节，我们将进入第 2 篇的《特殊函数的积分表示与围道积分》：Mathieu 函数与 Lamé 函数的许多恒等式和数值性质，都要靠把周期展开换成积分表示与围道积分才能看清——那正是把本章两族函数放回「多重表示」统一框架的一次收束。