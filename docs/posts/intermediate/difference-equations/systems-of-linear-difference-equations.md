---
title: 线性差分方程组
date: 2026-08-07
---

# 线性差分方程组

<div class="epigraph">
<p>把许多变量互相纠缠的演化写成一行 $x_{n+1} = A x_n$，世界瞬间就变得整齐了。</p>
<footer>—— 引意于 W. G. Kelley & A. C. Peterson（*Difference Equations\*）</footer>
</div>

<div class="article-byline">
<p>第二级 · 差分方程 ｜ Elaydi《An Introduction to Difference Equations》 第4章 §4.1–4.3 ｜ 2026-08-07</p>
</div>

## 为什么需要「组」

现实系统很少有单一变量：捕食者与猎物两个种群互相影响、一个国家里产出与物价互相纠缠、神经网络里上千个神经元彼此连接。把 $k$ 个标量差分方程联立，写成**线性差分方程组**，一个方程就装下全部动力学。它也是高阶标量方程的另一种视角——我们在《高阶线性差分方程》末尾预告过：$k$ 阶方程可化为 $k$ 个一阶方程。<span class="marginnote">这一节其实是《线性代数》与《差分方程》的第一次正面联姻：矩阵 $A$ 的幂、特征值与特征向量，直接决定解的形状。若你对 Jordan 标准形生疏，建议先回《线性代数》复习特征分解。</span>

## 1 向量形式的初值问题

**线性差分方程组**的标准形式为：

$$x_{n+1} = A x_n + b_n$$

其中 $x_n \in \mathbb{R}^k$ 是状态向量，$A$ 是 $k \times k$ 常数矩阵，$b_n$ 是外力项。当 $b_n \equiv 0$ 时称齐次：

$$x_{n+1} = A x_n$$

给定初值 $x_0$，解是矩阵的幂：

$$x_n = A^n x_0$$

**辨析｜易错点：** $A^n$ 是矩阵乘幂，不是逐元素幂。$A^n x_0$ 的含义是「反复作用 $n$ 次」，这与标量情形 $a^n x_0$ 一致，但计算方式完全不同——必须借助特征分解或 Jordan 标准形，绝不可把矩阵每个元素自乘 $n$ 次。

## 2 可对角化情形：特征值特征向量分解

若 $A$ 有 $k$ 个线性无关的特征向量 $v_1, \ldots, v_k$（特征值 $\lambda_1, \ldots, \lambda_k$），则 $A = P \Lambda P^{-1}$，于是：

$$A^n = P \Lambda^n P^{-1}, \qquad \Lambda^n = \operatorname{diag}(\lambda_1^n, \ldots, \lambda_k^n)$$

把初值按特征向量展开 $x_0 = c_1 v_1 + \cdots + c_k v_k$，解写为各模式的叠加：

$$
x_n = c_1 \lambda_1^n v_1 + c_2 \lambda_2^n v_2 + \cdots + c_k \lambda_k^n v_k
$$

这与标量高阶方程 $x_n = \sum c_i r_i^n$ 的结构完全平行，只是系数 $c_i v_i$ 是向量。<span class="marginnote">每个特征向量 $v_i$ 对应一个「独立模式」：系统沿 $v_i$ 方向以 $\lambda_i^n$ 的速率缩放。对角化就是把耦合的系统「解耦」成 $k$ 个互不相干的标量演化——这正是线性代数「选对坐标系」思想的动力学版本。</span>

## 3 重特征值与 Jordan 标准形

当特征值有重根、特征向量不足 $k$ 个时，$A$ 不可对角化，须用 **Jordan 标准形（Jordan canonical form）**。以 $2\times 2$ 的 Jordan 块 $J = \begin{pmatrix} \lambda & 1 \\ 0 & \lambda \end{pmatrix}$ 为例：

$$J^n = \begin{pmatrix} \lambda^n & n \lambda^{n-1} \\ 0 & \lambda^n \end{pmatrix}$$

对角线上是 $\lambda^n$，**上三角处出现 $n \lambda^{n-1}$ 的修正项**——这就是「重根乘 $n$」在矩阵世界的来源。<span class="marginnote">回想高阶方程里重特征根贡献 $n^m r^n$ 的模式，与这里的 $n\lambda^{n-1}$ 完全同源：Jordan 块越大，出现 $n^j \lambda^{n-j}$ 的高阶项越多。两个视角在此殊途同归。</span>

**易错点｜辨析：** 重根并不自动导致 $n$ 项。只有 Jordan 块（即特征向量缺失）才产生 $n$ 修正；若矩阵可对角化，即便特征值重复，模式仍是纯 $\lambda^n$。判断标准是**几何重数是否等于代数重数**，而不是「有没有重根」。

## 4 稳定性：特征值模长判据

解是各模式 $\lambda_i^n$ 的叠加，而 $|\lambda_i^n| = |\lambda_i|^n$。因此：

**系统 $x_{n+1} = A x_n$ 渐近稳定（对任意初值 $x_n \to 0$）当且仅当所有特征值满足 $|\lambda_i| \lt  1$，即全部落在复平面的单位圆内。**

若某个 $|\lambda_i| > 1$，对应模式爆炸，系统不稳定；若 $|\lambda_i| = 1$ 且其余在圆内，模式持久，系统 Lyapunov 稳定但不渐近稳定。<span class="marginnote">「单位圆」判据在离散系统里是绝对的：连续系统看特征值实部是否 < 0（左半平面），离散系统看模长是否 < 1（单位圆内）。两套语言在《微分方程离散化与差分格式》篇会再次相遇——数值稳定性正是在两套判据之间搭桥。</span>

## 5 公式解析：捕食—被捕食系统的稳定性

设两个种群 $p_n$（被捕食者）、$q_n$（捕食者）满足：

$$
\begin{pmatrix} p_{n+1} \\ q_{n+1} \end{pmatrix} = \begin{pmatrix} 1.1 & -0.2 \\ 0.1 & 0.9 \end{pmatrix} \begin{pmatrix} p_n \\ q_n \end{pmatrix}
$$

- **第一步，求特征值**：$\det(A - \lambda I) = (1.1 - \lambda)(0.9 - \lambda) + 0.02 = \lambda^2 - 2\lambda + 1.01 = 0$，判别式 $\Delta = 4 - 4.04 = -0.04$，得共轭复根 $\lambda = 1 \pm 0.1i$。
- **第二步，看模长**：$|\lambda| = \sqrt{1 + 0.01} = \sqrt{1.01} > 1$，在单位圆外。
- **第三步，下结论**：复特征值的辐角使两个种群**振荡**（此消彼长），模长略大于 1 使振幅**缓慢发散**——系统不稳定。若矩阵第二行改为 $0.08$，则 $|\lambda| = \sqrt{1.0016} \lt  1$，振荡且衰减，两物种共存于稳定振荡。

关键一步是**从复特征值的模长同时读出「振荡」与「发散」两个信息**：辐角定周期、模长定增幅。这与标量方程复根 $\rho^n \cos n\theta$ 的解读完全一致，只是换到了向量视角。

## 6 基础矩阵、相伴矩阵与 Casoratian

### 基础矩阵

若 $k$ 个解向量 $x_n^{(1)}, \ldots, x_n^{(k)}$ 在 $\mathbb{R}^k$ 中线性无关，则它们构成**基础解系**，排成 **基础矩阵（fundamental matrix）** $\Phi_n = \bigl[ x_n^{(1)} \; \cdots \; x_n^{(k)} \bigr]$，通解写作 $x_n = \Phi_n c$。基础矩阵满足 $\Phi_{n+1} = A \Phi_n$，且

$$
\Phi_n = A^n \Phi_0
$$

——矩阵幂 $A^n$ 正是「以单位阵为初值的基础矩阵」。这一视角把「求矩阵幂」与「求基础解系」统一起来。

### 线性无关的判定：Casoratian

向量解的线性无关性用 **Casoratian** 行列式判定：$\det \Phi_n \neq 0$。对高阶梯标量方程，Casoratian 即由 $x_n, x_{n+1}, \ldots, x_{n+k-1}$ 排成的行列式，是 Wronskian 的离散对应。它满足一阶演化律：

$$
\det \Phi_{n+1} = \det(A) \cdot \det \Phi_n
$$

**易错点｜辨析：** 与 Wronskian 类似，Casoratian 在某一 $n$ 非零 ⇔ 在所有 $n$ 非零（只要 $\det A \neq 0$）。判定线性无关只需查一个时间点的值，不必逐点验证。

### 相伴矩阵：标量方程 ⇔ 方程组

把 $k$ 阶标量方程 $x_{n+k} + p_1 x_{n+k-1} + \cdots + p_k x_n = 0$ 化成方程组：令 $y_n = (x_n, x_{n+1}, \ldots, x_{n+k-1})^T$，则 $y_{n+1} = C y_n$，其中

$$
C = \begin{pmatrix} 0 & 1 & \cdots & 0 \\ \vdots & \ddots & \ddots & \vdots \\ 0 & \cdots & 0 & 1 \\ -p_k & \cdots & -p_2 & -p_1 \end{pmatrix}
$$

$C$ 叫**相伴矩阵（companion matrix）**，其特征多项式 $\det(rI - C) = r^k + p_1 r^{k-1} + \cdots + p_k$ 恰好还原标量方程的特征方程——**标量方程的根 ⇔ 方程组矩阵的特征值**，两个视角彻底打通。Fibonacci 方程 $F_{n+2} = F_{n+1} + F_n$ 的相伴矩阵 $C = \begin{pmatrix} 0 & 1 \\ 1 & 1 \end{pmatrix}$，$\det C = -1$，故其 Casoratian 每步变号但永不消失，解恒线性无关。

## 7 小结

- 线性差分方程组 $x_{n+1} = A x_n$ 的解是 $x_n = A^n x_0$，矩阵幂必须靠特征分解计算。
- 可对角化时解为模式叠加 $x_n = \sum c_i \lambda_i^n v_i$；不可对角化时 Jordan 块带来 $n \lambda^{n-1}$ 修正项。
- **稳定性判据**：全部特征值 $|\lambda_i| \lt  1$