---
title: 线性矩阵方程（Sylvester/Lyapunov）
date: 2026-08-11
---

# 线性矩阵方程（Sylvester/Lyapunov）

<div class="epigraph">
<p>控制论的稳定性问题，最终可以归结为一个矩阵方程是否有正定解——Lyapunov 方程是稳定性之门的钥匙。</p>
<footer>—— 化用自亚历山大·李雅普诺夫（Aleksandr Lyapunov）</footer>
</div>

<div class="article-byline">
<p>第二级 · 矩阵论 ｜ Horn & Johnson《Matrix Analysis》 ｜ 2026-08-11</p>
</div>

## 为什么从线性矩阵方程开始

解方程是线性代数的主业，但此前我们解的都是**向量方程** $Ax = b$。
现实中频繁出现**未知量是矩阵**的方程：线性系统理论里的 Sylvester 方程、控制与稳定性分析里的
Lyapunov 方程、以及信号处理中的离散 Riccati 的线性化。上一节的 Kronecker 积与 vec
算子给了我们一把「万能钥匙」：**把矩阵方程改写成向量方程**。这一节我们把理论、可解性判据、
数值方法与稳定性应用一次讲清——这也是从纯矩阵论到控制论、再到现代大模型分析（如状态空间模型 Mamba
的离散化）的桥梁<span class="marginnote">从极限到大模型的连接：状态空间模型（SSM）的核心是连续时间的
$\dot{x} = Ax + Bu$ 与其离散化，离散化过程要解 Lyapunov/指数型矩阵方程；
线性二次调节（LQR）与卡尔曼滤波的 Riccati 方程也以 Lyapunov 方程为胚胎。
矩阵方程是"动态系统分析"的代数内脏。</span>。

## 1 Sylvester 方程与可解性

**Sylvester 方程（Sylvester equation）**：

$$AX - XB = C$$

其中 $A \in \mathbb{C}^{m\times m}$、
$B \in \mathbb{C}^{n\times n}$、
$C, X \in \mathbb{C}^{m\times n}$。它推广了 $AX - XB$
这种「双面作用」的未知矩阵。$B = \lambda I$ 时退化为 $AX - \lambda X = C$，即
$(A - \lambda I)X = C$——与特征向量方程的「矩阵版」直接相关。<span class="marginnote">名字由来：Sylvester 在 1884 年研究这类方程，
它是「矩阵方程理论」的起点。当 $C = 0$ 时，非零解 $X$ 的存在性与 $A, B$ 的公共特征结构相关，
这连接到后面要谈的谱条件。</span>

**可解性判据（唯一性）**：对任意右端 $C$，方程有唯一解 $X$ **当且仅当 $A$ 与 $B$
没有公共特征值**，即

$$\sigma(A) \cap \sigma(B) = \varnothing$$

其中 $\sigma(\cdot)$ 表示谱。**直观**：$AX - XB$ 这个线性映射的「本征频率」是
$\lambda_i(A) - \mu_j(B)$（特征值之差），全不为零 ⇔ 映射可逆。<span class="marginnote">为什么是特征值之差：用 vec 恒等式改写后，线性映射
$X \mapsto AX - XB$ 的特征值是 $\lambda_i(A) - \mu_j(B)$（这是上一节
Kronecker 谱结论的直接推论）。可逆 ⇔ 无零特征值 ⇔ $A, B$ 无公共特征值。
这个"谱语言"让可解性一目了然。</span>

**一般情形**：若 $A, B$ 有公共特征值，方程可能有解但不唯一（解空间有正维度），或对某些 $C$ 无解。

## 2 Lyapunov 方程与稳定性

**连续 Lyapunov 方程（continuous Lyapunov equation）**：

$$A^{*}X + XA = Q$$

**离散 Lyapunov 方程（Stein 方程）**：

$$A^{*}XA - X = Q$$

这里 $Q$ 给定（常取 Hermite 正定），$X$ 未知。<span class="marginnote">连续方程
$A^{*}X + XA = Q$ 是 Sylvester 方程 $A^{*}X - X(-A) = Q$ 的特例（取
$B = -A$）。两者共享同一套谱语言：连续方程可解唯一当 $A$ 与 $-A$ 无公共特征值，即 $A$
的特征值实部全非零。</span>

**Lyapunov 定理（稳定性）**：设 $A$ 是 Hermite 稳定性关心的对象——若对任意 Hermite
正定 $Q$，Lyapunov 方程有 Hermite 正定解 $X$，则 $A$ 的所有特征值有负实部；反之若 $A$
特征值实部全负，则对每个正定 $Q$ 存在唯一的正定解

$$X = \int_{0}^{\infty} e^{A^{*}t} Q e^{At}\, dt$$

这个积分形式是「解的存在性 + 正定性」的一箭双雕：被积函数在 $t\to\infty$ 时指数衰减（因 $A$ 稳定），
积分收敛且正定。<span class="marginnote">物理直觉：$A^{*}X + XA = Q$
度量「能量函数 $V(z) = z^{*}Xz$ 沿系统 $\dot z = Az$
的导数」——$\dot V = z^{*}(A^{*}X + XA)z = z^{*}Qz$。$Q$ 正定意味着 $V$
沿轨道严格下降，系统必然收敛到原点。<strong>Lyapunov 方程把「稳定性」翻译成「找一个下降的能量函数」</strong>。</span>

**离散版定理**：离散 Lyapunov 方程有唯一正定解 ⇔ $\rho(A) < 1$（$A$ 的谱半径小于 1），
这正是上一组《非负矩阵》里强调的收敛条件在控制论中的回响。

## 3 求解：vec 展开与数值方法

**理论解法（vec 展开）**：对 Sylvester 方程取 vec，用上节恒等式：

$$(I_n \otimes A - B^{T} \otimes I_m)\,\operatorname{vec}(X) = \operatorname{vec}(C)$$

得到一个 $mn \times mn$ 的**标准线性方程组**，可用高斯消元。但 $mn$
可能极大（$m = n = 10^3$ 时矩阵有 $10^{12}$ 个元素），直接展开不现实。<span class="marginnote">复杂度警示：vec 展开把问题规模从「两个 $n\times n$」膨胀到「一个
$n^2\times n^2$」，存储与时间是灾难级的。数值上必须利用结构——Sylvester 方程有 $O(n^3)$
的 Bartels–Stewart 算法，把 $A, B$ 先 Schur 三角化再回代，是工程中的标准做法。</span>

**Bartels–Stewart 算法**（两步）：先对 $A, B$ 做 Schur 分解 $A = UTU^{*}$、
$B = VSV^{*}$，则方程化为 $TY - YS = \tilde C$（$Y = U^{*}XV$），其中
$T, S$ 上三角；再逐列解一个三角方程组（回代）。复杂度 $O(n^3)$，远优于 vec 展开。

**辨析｜易错点：** 三个方程别混淆——Sylvester $AX - XB = C$ 是「双面作用、可解条件看公共谱」；
连续 Lyapunov $A^{*}X + XA = Q$ 是「$A$ 与 $-A$ 配对、稳定看实部」；离散
Lyapunov $A^{*}XA - X = Q$ 是「稳定看 $\rho(A) < 1$」。三者右端、谱条件、
结论各不相同，套错公式是高频错误。

## 4 应用：LQR、Riccati 方程与控制理论

Lyapunov 方程在控制论中不只是理论判据，它是**最优控制与滤波**的核心构件。这一节给出两个最重要的应用场景。

**线性二次调节器（LQR）**：对线性系统 $\dot{x} = Ax + Bu$，要选择控制 $u(t)$ 使代价
$\int_0^\infty (x^{*}Qx + u^{*}Ru)\,dt$ 最小。最优控制为 $u = -Kx$，
其中增益 $K = R^{-1}B^{*}P$，而 $P$ 满足**代数 Riccati 方程**

$$A^{*}P + PA - PBR^{-1}B^{*}P + Q = 0$$

注意它的前两项 $A^{*}P + PA$ 正是 Lyapunov 方程的形式——**Riccati 方程是带二次项
$-PBR^{-1}B^{*}P$ 的 Lyapunov 方程**。$P$ 正定且使闭环 $A - BK$ 稳定，
最优代价恰为 $x_0^{*}Px_0$。<span class="marginnote">从 Lyapunov 到
Riccati：当 $B = 0$（没有控制）时，Riccati 方程退化为 Lyapunov 方程
$A^{*}P + PA = -Q$。可以说 Riccati 方程是"闭环反馈版"的 Lyapunov
方程——Lyapunov 是稳定的判据，Riccati 是最优的判据，两者在同一框架里前后相承。</span>

**卡尔曼滤波**：连续时间卡尔曼滤波的状态估计误差协方差 $P(t)$ 满足

$$\dot P = AP + PA^{T} + BQB^{T} - PC^{T}R^{-1}CP$$

稳态解满足的正是**Riccati 型方程**（带 $+BQB^{T}$ 与 $-PC^{T}R^{-1}CP$ 两项）。
卡尔曼滤波与 LQR 在数学上是**对偶**的：一个是最优控制、一个是最优估计，共用同一套 Riccati 机器。
<span class="marginnote">对偶性：LQR 的 $A, B, Q, R$ 对应滤波的
$A^{T}, C^{T}, Q, R$，最优控制问题与最优估计问题因此共享求解器——控制与滤波是同一枚硬币的两面。
这也是为什么现代强化学习（基于模型）中常用 Riccati/Lyapunov 类方程做局部值函数近似。</span>

**求解 Riccati 方程**：可用 Newton 迭代：固定二次项、迭代解线性化后的 Lyapunov 方程，
收敛到唯一的稳定正定解。这与数值代数中"把非线性方程逐步线性化"的通用策略一致。工程库（如
`scipy.linalg.solve_continuous_are`）常先用 Schur 分解构造不变子空间（Laub
方法），复杂度 $O(n^3)$ 且数值稳定。

**状态空间模型与深度学习**：现代状态空间模型（SSM、Mamba）的核心方程

$$\dot{x} = Ax + Bu, \qquad y = Cx$$

正是线性系统理论的基本对象。其离散化需要计算 $e^{A\Delta t}$ 与矩阵积分（本质是矩阵指数/对数问题），
而可学习参数的初始化、以及系统稳定性的分析，则大量借用
Lyapunov/谱的理论——**"从极限到大模型"的主线在这里与经典控制论正面相遇**。<span class="marginnote">SSM 的深层连接：Mamba 等模型要求 $A$ 的特征值实部为负（系统稳定），
这是 Lyapunov 理论直接给出的设计约束；离散化步长与 $e^{A\Delta t}$ 的计算精度决定长程记忆的质量。
控制论不是过时学问，它是现代序列模型的理论骨架。</span>

**辨析｜易错点：** Lyapunov 方程与 Riccati 方程**不要混用**：前者线性、后者带二次项；
前者判定稳定性、后者求最优控制/滤波。另一个高频错误：LQR 中要求 $Q \succeq 0$、
$R \succ 0$（控制代价必须正定），若 $R$ 奇异则增益 $R^{-1}B^{*}P$
无意义——"代价矩阵的正定性约束"是建模时必须先验检查的。

## 5 公式解析：Lyapunov 积分解 $X = \int_0^\infty e^{A^{*}t}Qe^{At}\,dt$

这条积分解公式把「矩阵方程」与「矩阵指数」缝在一起，拆四步：

- **第一步，验证它是解**：代入方程左边，利用 $\frac{d}{dt}e^{At} = Ae^{At}$：

$$A^{*}X + XA = \int_0^\infty \left(A^{*}e^{A^{*}t}Qe^{At} + e^{A^{*}t}Qe^{At}A\right) dt = \int_0^\infty \frac{d}{dt}\left(e^{A^{*}t}Qe^{At}\right)dt$$

- **第二步，微积分基本定理**：定积分等于端点差：$e^{A^{*}t}Qe^{At}\Big|_{0}^{\infty}$。$t=0$ 时等于 $Q$；$t\to\infty$ 时 $e^{At}\to 0$（$A$ 特征值实部为负），积分为 $0 - Q$ 的相反数。
- **第三步，符号对齐**：原式为 $A^{*}X + XA = Q$，而导数积分为 $\left[e^{A^{*}t}Qe^{At}\right]_0^\infty = 0 - Q = -Q$。**负号差提示方程右边应是 $-Q$**——这正是为什么有的教材写 $A^{*}X + XA = -Q$。约定不同，解的符号不同。
- **第四步，正定性从何而来**：对任意 $z \neq 0$，$z^{*}Xz = \int_0^\infty \|Q^{1/2}e^{At}z\|^2 dt$（$Q$ 正定可开方）。被积函数恒 ≥ 0 且不恒为零（$e^{At}$ 可逆），故积分 > 0，$X$ 正定。**稳定性自动把解"锻造成"正定的能量矩阵**。

## 6 小结

- **Sylvester 方程** $AX - XB = C$：唯一可解 ⇔ $\sigma(A)\cap\sigma(B)=\varnothing$（谱语言：特征值之差）。
- **连续 Lyapunov** $A^{*}X + XA = Q$：$A$ 特征值实部全负 ⇔ 对正定 $Q$ 有唯一正定解；解有积分公式。
- **离散 Lyapunov** $A^{*}XA - X = Q$：唯一正定解 ⇔ $\rho(A) < 1$。
- **数值求解**：vec 展开是理论工具（$O(n^6)$ 不可行），Bartels–Stewart 三角化回代才是 $O(n^3)$ 的标准做法。
- 易错点：三个方程的谱条件与右端各不相同；积分解的符号取决于方程约定。

在下一节，我们将解答「不可逆矩阵如何谈逆」——Moore-Penrose 伪逆，把逆的概念推广到任意矩阵，
为最小二乘与广义逆理论收官。
