---
title: Brenier 定理与凸函数梯度映射
date: 2026-08-07
---

# Brenier 定理与凸函数梯度映射

<div class="epigraph">
<p>在平方距离的代价下，最优传输映射总是某个凸函数的梯度——这是一个美丽而深刻的定理。</p>
<footer>—— 雅恩 · 布雷尼耶（Yann Brenier），《Polar Factorization and Monotone Rearrangement of Vector-Valued Functions》（1991，意译）</footer>
</div>

<div class="article-byline">
<p>第二级 · 最优传输理论 ｜ Villani《Optimal Transport: Old and New》第9章、第10章 ｜ 2026-08-07</p>
</div>

## 为什么从 Brenier 定理开始

前面的故事到"Kantorovich 松弛"时，蒙日的映射视角被暂时放弃了。**Brenier 定理**（1991）把映射视角夺了回来：在平方代价 $c(x,y) = \|x-y\|^2$ 下，只要 $\mu$ 绝对连续（没有原子），**Kantorovich 问题的最优解其实是一个确定性耦合，且最优映射具有极其具体的形态——它是某个凸函数的梯度**。这一定理同时回答了三个问题：最优映射何时存在（几乎总是）、长什么样（凸函数梯度）、怎么刻画（Monge–Ampère 方程）。它是整个最优传输里最深刻也最实用的一座里程碑，直接支撑着 Wasserstein 梯度流、Wasserstein GAN 与生成模型的几何。<span class="marginnote">Brenier 的工作源自流体力学的"极分解"问题，但它的数学内核是纯分析+凸几何的。1991 年之后，Caffarelli、McCann、Ambrosio 等人把这条线做成了整套理论——如今它被称为最优传输的"Brenier 学派"。</span>

## 1 平方代价的特殊之处

先看为什么 $c = \|x-y\|^2$ 如此特别。把平方展开：

$$
\|x - y\|^2 = \|x\|^2 + \|y\|^2 - 2\, x \cdot y
$$

前两项 $ \|x\|^2$ 与 $\|y\|^2$ 分别只依赖于 $x$、只依赖于 $y$，它们在优化中只是常数——对任意耦合，$\int \|x\|^2 d\pi = \int \|x\|^2 d\mu$ 被边际钉死，与 $\pi$ 无关。于是平方代价下的 Kantorovich 问题等价于

$$
\max_{\pi \in \Pi(\mu,\nu)} \; \int_{X \times Y} x \cdot y \; d\pi(x,y)
$$

即**最大化传输方向与位置的点积（相关性）**。<span class="marginnote">"最大化 $x\cdot y$"的直觉：尽量把质量从 $x$ 传到与它"同向"的 $y$。这正好解释了为什么 $W_2$ 最优传输是"单调"的——高位置的 $x$ 倾向于传给高位置的 $y$，它不做无谓的交叉搬运。这与我们第 3 篇讲的 c-单调性（不允许交叉）完全一致。</span>

## 2 公式解析：T = ∇φ 与 Monge–Ampère 方程

**Brenier 定理**（简版）：设 $\mu \in \mathcal{P}_2(\mathbb{R}^d)$ 关于 Lebesgue 测度绝对连续，$\nu \in \mathcal{P}_2(\mathbb{R}^d)$ 任意。则

1. 平方代价下**存在唯一**的最优传输映射 $T$；
2. 存在**凸函数** $\varphi: \mathbb{R}^d \to \mathbb{R}$，使得 $T = \nabla \varphi$；
3. $\varphi$ 是下列 **Monge–Ampère 方程**的唯一凸解（相差一个常数）：

$$
\det D^2 \varphi(x) \, \rho(x) = \eta\big(\nabla \varphi(x)\big), \qquad \mu = \rho \, dx,\; \nu = \eta \, dy
$$

拆成三步理解：

- **第一步，读懂 $T = \nabla \varphi$**：最优映射不是任意函数，而是某个**凸函数**的梯度场。凸函数的梯度映射一定是"单调算子"（$(T(x)-T(x'))\cdot(x-x') \ge 0$），这正是"不交叉运输"的解析表达。$\varphi$ 称为**传输势（transport potential）**。
- **第二步，读懂 Monge–Ampère 方程**：左边 $\det D^2\varphi$ 是 $\nabla\varphi$ 的 Jacobi 行列式（局部体积放大率），乘上初始密度 $\rho(x)$，给出"被 $\nabla\varphi$ 推过去的密度"；右边 $\eta(\nabla\varphi(x))$ 是目标密度在像点的值。**方程说：推过去的密度必须恰好等于目标密度**——这就是质量守恒 $(\nabla\varphi)_\#\mu = \nu$ 的微分写法。
- **第三步，读懂"唯一凸解"**：Monge–Ampère 方程是高度非线性的二阶 PDE，一般解不唯一；但**要求解是凸的**就唯一确定了它。凸性在这里是"挑选物理合理解"的选择准则。

**辨析｜易错点：** $T = \nabla\varphi$ 要求 $\mu$ **绝对连续**（无原子）。若 $\mu$ 有原子，最优传输映射可能不存在，最优耦合会真正"分裂"。这是 Brenier 定理与"总存在唯一映射"之间最常见的误用边界。<span class="marginnote">直觉：如果 $\mu$ 的某个点上有原子质量，它的像若落在 $\nu$ 的原子区域边缘，为了同时满足两个边际，质量必须被拆散——而这违背映射的确定性。绝对连续性把这种"选择困难"排除掉了。</span>

## 3 一维情形：单调重排

在 $d=1$ 时，Brenier 定理退化成任何人都能验证的结论。设 $\mu,\nu$ 是实直线上的概率测度，$F(x) = \mu((-\infty, x])$、$G(y) = \nu((-\infty, y])$ 为累积分布函数，则最优映射是

$$
T(x) = G^{-1}\big(F(x)\big)
$$

这是**分位数函数对分位数函数的复合**，称为**单调重排（monotone rearrangement）**：$x$ 对应的 $F(x)$ 分位，被送到 $\nu$ 的同一个分位。<span class="marginnote">一维最优传输 = "按排名对号入座"：把 $\mu$ 的样本按大小排好队，再把 $\nu$ 的样本按同一排名依次分给它们。这也是一维分位数变换、以及生成模型里"正态化流"能精确工作的根本原因。第二级《概率论与数理统计》里的分位数函数在这里获得了最优性解释。</span>例如 $\mu = U(0,1)$、$\nu = U(a,b)$ 时，$T(x) = a + (b-a)x$——线性拉伸，显然最省"平方代价"。

## 4 从存在性到构造性：定理为何重要

Brenier 定理的威力在于它把最优传输从"黑箱耦合"推进到"显式构造"。三个直接推论值得一提：

**推论一：$W_2$ 是"函数的距离"。** 因为最优映射是凸函数梯度，$W_2^2(\mu,\nu)$ 可以写成传输势的积分：$W_2^2 = \int \|x\|^2 d\mu + \int \|y\|^2 d\nu - 2\int x \cdot \nabla\varphi(x)\, d\mu$。这使 Wasserstein 距离可以绕过耦合直接对 $\varphi$ 优化。

**推论二：为梯度流铺路。** 若把 $W_2$ 当作位形空间上的距离，那么"在 $W_2$ 度量下沿某个泛函最速下降"就等价于解一个与 $\nabla\varphi$ 有关的 PDE——这正是第 6 篇 JKO 格式的出发点。

**推论三：生成模型的几何。** 从 $\mu$（噪声）到 $\nu$（数据）的最优映射是凸函数梯度这一事实，给"连续归一化流""最优传输生成模型"提供了理论保证：**最优变换是良态（单调、保定向）的**，不会被折叠成病态的奇异性。<span class="marginnote">把 Brenier 定理与第一级《凸分析》对照：凸函数的梯度是"单调图"，它的逆也是单调图，所以 $T$ 可逆且 $T^{-1}$ 也是某个凸函数的梯度——最优传输在 $W_2$ 下是对合式的，这保证了双向生成都良性。</span>

与梯度流相关的一个重要推论是 $W_2$ 空间的**测地线**结构：给定 $\mu, \nu$ 与最优映射 $T = \nabla\varphi$，中间时刻的测度

$$
\mu_t = \big( (1-t)\,\mathrm{id} + t\,T \big)_\# \mu, \qquad 0 \le t \le 1
$$

就是连接 $\mu$ 到 $\nu$ 的 $W_2$ 测地线——质量沿直线、以匀速从起点流向终点。这条"映射线性插值"的路径是第 6 篇梯度流与图像插值应用共同的地基。

## 5 一个二维手算：平移就是最优传输

理论配上一个能完全算清的二维例子。设 $\mu$ 是单位正方形 $[0,1]^2$ 上的均匀分布，$\nu$ 是正方形 $[a,a+1] \times [c,c+1]$ 上的均匀分布（$a,c > 0$，整体向右上平移）。

**直觉上的最优映射**是把每个点平移 $(a,c)$：

$$
T(x) = x + (a, c)
$$

它把单位正方形整个搬到目标正方形，Jacobi 行列式 $\det \nabla T = 1$，所以 $T_\#\mu = \nu$（均匀测度被保持）。代价

$$
\int \|T(x) - x\|^2 d\mu(x) = a^2 + c^2
$$

**验证 $T = \nabla\varphi$**：令

$$
\varphi(x) = \frac12 \|x\|^2 + (a,c) \cdot x
$$

则 $\nabla\varphi(x) = x + (a,c) = T(x)$，且 Hessian $D^2\varphi = I$（正定，故 $\varphi$ 凸）。于是 Brenier 定理的形态完全验证：**最优映射是凸函数的梯度**，而"平移"正是其中最平凡的一种。<span class="marginnote">这里 $\varphi$ 的 Hessian 是单位矩阵，对应 Monge–Ampère 方程里 $\det D^2\varphi = 1$：体积放大率处处为 1，均匀测度搬到均匀测度，方程两边都等于常数，平凡地成立。</span>

**反例对照**：若换成**非凸**的 $\varphi(x) = -\frac12\|x\|^2$，梯度为 $\nabla\varphi(x) = -x$——把正方形"翻个面"，几何上就是所有质点穿过原点反向而行，轨迹交叉、方向反转。这样的映射不但不保持 $\nu$，而且违背"不交叉"的经济直觉。**凸性不是技术细节，它就是"最优"二字的意义**：凸函数的梯度场天然无旋、无交叉。

**辨析｜易错点：** 平移是 $W_2$ 最优传输里最简单的情形，但别被它误导成"最优映射总是平移"。一般情形（$\mu$ 变密度、$\nu$ 变密度）下，$T$ 是**非线性**的凸函数梯度，Monge–Ampère 方程 $\det D^2\varphi \cdot \rho = \eta \circ \nabla\varphi$ 才真正发威——$\rho,\eta$ 的起伏决定了 $\varphi$ 的曲率起伏。

## 6 小结

- **Brenier 定理**：平方代价下，若 $\mu$ 绝对连续，则存在唯一最优映射 $T$，且 $T = \nabla\varphi$（$\varphi$ 凸）。
- **Monge–Ampère 方程**：$\det D^2\varphi \cdot \rho = \eta \circ \nabla\varphi$，凸解唯一——质量守恒的微分形式。
- 平方代价可化为**最大化 $\int x\cdot y\, d\pi$**，其最优性等价于"单调不交叉"。
- 一维退化：$T = G^{-1} \circ F$，即**单调重排**。
- 定理把最优传输从"存在性"推向"构造性"，是梯度流与生成模型的理论基石。

在下一节，我们把 $W_2$