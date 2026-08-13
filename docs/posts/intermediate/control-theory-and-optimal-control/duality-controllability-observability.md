---
title: 能控性与能观性的对偶原理
date: 2026-08-07
---

# 能控性与能观性的对偶原理

<div class="epigraph">
<p>能控性与能观性之间的对称，是整个线性系统理论中最优美的结构之一。</p>
<footer>—— 卡尔曼（Rudolf E. Kalman，1960）</footer>
</div>

<div class="article-byline">
<p>第二级 · 控制论与最优控制 ｜ Ogata《现代控制工程》Ch. 9 ｜ 2026-08-07</p>
</div>

## 为什么对偶值得单独讲

前面两节分别讲了能控性与能观性，细心的读者一定已经察觉到它们之间惊人的对称：能控看 $[B, AB, \dots]$，能观看 $[C; CA; \dots]^T$；可控 Gramian 与可观 Gramian 互为转置镜像；秩判据一脉相承。**对偶原理（duality principle）**把这种「形似」升级为一条严格的数学定理，并给出一个可操作的规则：把 $A$ 换成 $A^T$、$B$ 换成 $C^T$，能控性的任何结论都会翻译成能观性的结论。

这条原理不是用来「省一篇证明」的装饰品，它有实实在在的工程回报：能观性问题的每一个算法都能通过对偶直接改造成能控性问题的算法，反之亦然。今天把这条「镜像规则」讲透。

## 1 对偶系统

给定原系统

$$
\Sigma: \quad \dot{x} = Ax + Bu, \qquad y = Cx,
$$

构造它的**对偶系统（dual system）**

$$
\Sigma^d: \quad \dot{z} = A^T z + C^T v, \qquad w = B^T z,
$$

其中 $z$ 是 $n$ 维对偶状态，$v$ 是对偶输入，$w$ 是对偶输出。<span class="marginnote">注意对偶系统的输入输出通道发生了「换位」：原系统的输入矩阵 $B$ 变成对偶系统的输出矩阵 $B^T$，原系统的输出矩阵 $C$ 变成对偶系统的输入矩阵 $C^T$。维度也互换：原系统 $u \in \mathbb{R}^m \to y \in \mathbb{R}^p$，对偶系统 $v \in \mathbb{R}^p \to w \in \mathbb{R}^m$。信息流的箭头完全反转。</span>这正是「对偶」的直观含义：把系统的因果关系整个倒过来，信号方向反向，输入变输出、输出变输入。

## 2 对偶原理：定理与证明思路

**对偶原理（duality principle）**：系统 $\Sigma$ 完全能控，当且仅当对偶系统 $\Sigma^d$ 完全能观；系统 $\Sigma$ 完全能观，当且仅当 $\Sigma^d$ 完全能控。

证明的思路极其干净，就藏在能控性矩阵与能观性矩阵的转置关系里。构造 $\Sigma^d$ 的能观性矩阵：

$$
\mathcal{O}^d = \begin{bmatrix} B^T \\ B^T A^T \\ \vdots \\ B^T (A^T)^{n-1} \end{bmatrix}
= \begin{bmatrix} B^T \\ (AB)^T \\ \vdots \\ (A^{n-1}B)^T \end{bmatrix}
= \big(\begin{bmatrix} B & AB & \cdots & A^{n-1}B \end{bmatrix}\big)^T = \mathcal{C}^T.
$$

即**对偶系统的能观性矩阵，恰是原系统能控性矩阵的转置**。由于转置不改变秩，$\operatorname{rank}\mathcal{O}^d = \operatorname{rank}\mathcal{C}$。于是：

$$
\Sigma^d \text{ 完全能观} \;\Longleftrightarrow\; \operatorname{rank}\mathcal{O}^d = n \;\Longleftrightarrow\; \operatorname{rank}\mathcal{C} = n \;\Longleftrightarrow\; \Sigma \text{ 完全能控}.
$$

一个等式的转置，就完成了整条对偶原理的证明。<span class="marginnote">类似地，$\Sigma^d$ 的能控性矩阵是 $\Sigma$ 能观性矩阵的转置，第二条对偶关系同样成立。整个对偶原理的本质就是一句话：<strong>转置保持秩</strong>，而能控性与能观性都归结为某个矩阵的秩条件。</span>

## 3 对偶原理的工程回报

对偶原理的最大价值在于**一通百通**：设计层面的每个操作都有对偶版本。

- **观测器 ↔ 反馈控制器**。第 3 篇会讲到，状态观测器的增益 $L$ 通过「对偶系统的极点配置」计算：原系统要设计观测器，等价于对偶系统要设计状态反馈。一套极点配置算法，同时解决两个问题。
- **Gramian 镜像**。$W_c$ 满足 $AW_c + W_cA^T = -BB^T$；把 $A \to A^T$、$B \to C^T$，就得到 $A^T W_o + W_o A = -C^TC$——两条 Lyapunov 方程互为对偶，解也互为转置对应。
- **Kalman 分解对称**（下一节）：能控子空间与能观子空间的分解结构完全对偶，一个系统的「不能控部分」对应于对偶系统的「不能观部分」。

**辨析｜易错点：** 对偶不是「同一个系统换个说法」，而是**另一个系统**。$\Sigma$ 与 $\Sigma^d$ 的状态空间维数相同，但输入输出维数互换、系统矩阵转置，物理意义完全不同。另一个易错点是：对偶原理说的是「能控性 ⇔ 能观性」的**等价**，不是「能控的数值」等于「能观的数值」——$W_c$ 与 $W_o$ 是两个不同矩阵，只是满足同构的方程。做习题时，先写清「原系统 vs 对偶系统」，再套用秩判据，就不容易张冠李戴。

## 4 公式解析：对偶 Gramian 的镜像结构

把两条 Lyapunov 方程并排写出来，对偶关系一目了然：

$$
\underbrace{A W_c + W_c A^T = -BB^T}_{\Sigma \text{ 的可控 Gramian}}, \qquad
\underbrace{A^T W_o + W_o A = -C^T C}_{\Sigma \text{ 的可观 Gramian}}.
$$

- **第一步，识别对偶映射**：第二条方程里出现的是 $A^T$、$C^TC$，恰好等于第一条方程做替换 $A \to A^T$、$B \to C^T$ 的结果——这就是「$\Sigma^d$ 的可控 Gramian 方程」。
- **第二步，解读 Gramian 角色**：$W_c$ 度量「输入推动状态的容易程度」，$W_o$ 度量「输出观察状态的容易程度」。对偶原理说：推动 $A$ 的容易程度 = 观察 $A^T$ 的容易程度。
- **第三步，最小能量 ↔ 最大信息**：最小控制能量 $x_0^T W_c^{-1}x_0$ 与重构误差（噪声下）$W_o^{-1}$ 的角色也互相对偶——控制要「花小力气」对应观测要「多信息量」，本质是同一个凸问题的两面。

这条镜像结构的实际用途是：你在文献里看到任何一个关于能控性的定理或算法，立刻写出它的对偶版本，就免费得到能观性的对应结果。科研与工程里这是最高效的「借力」方式之一。

### 对偶的一次完整旅行

把对偶原理在一对具体的系统上完整走一遍，比记定理更管用。取一个最简单的双积分器系统

$$
\Sigma: \quad A = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}, \quad
B = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad
C = \begin{bmatrix} 1 & 0 \end{bmatrix}.
$$

**能控性检查**：$\mathcal{C} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$，秩 2，完全能控。**能观性检查**：$\mathcal{O} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$，秩 2，完全能观——双积分器「既能控又能观」，是个幸运儿。

现在构造对偶系统 $\Sigma^d$：$A^T = \begin{bmatrix} 0 & 0 \\ 1 & 0 \end{bmatrix}$，$B^d = C^T = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$，$C^d = B^T = \begin{bmatrix} 0 & 1 \end{bmatrix}$。按对偶原理，$\Sigma^d$ 应当完全能观、完全能控。逐项验证：

$$
\mathcal{O}^d = \begin{bmatrix} B^T \\ B^TA^T \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad
\operatorname{rank} = 2 \quad (\Sigma \text{ 能控}), \qquad
\mathcal{C}^d = \begin{bmatrix} B^d & A^TB^d \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
\operatorname{rank} = 2 \quad (\Sigma \text{ 能观}).
$$

注意到 $\mathcal{O}^d = \mathcal{C}^T$、$\mathcal{C}^d = \mathcal{O}^T$，秩的等式一步不差。**这条验证路线的意义**：以后你想给某个系统设计观测器，不必重新推导，只需把「对偶系统能控」这个命题查一遍秩，就立刻知道观测器能不能配任意极点——对偶原理替你省掉一整轮重复劳动。

### 对偶的一处「物理」直觉

为什么能控性与能观性偏偏对偶，而不是「无关」？一个物理想象：**输入是「往状态里灌信息」，输出是「从状态里取信息」**。能控性问「灌得进去吗」（$B$ 的支配力），能观性问「取得出来吗」（$C$ 的感知力）。时间反演之下，系统的因果箭头反转，「灌进去」与「取出来」恰好互换——这就是对偶的物理根源。这条直觉还预告了后续一个更深的结果：**Kalman 分解里「不可控」与「不可观」的角色在时间反演下互换**，而最小实现「既可控又可观」之所以两边都要求，正因为它要在时间反演下保持自我。

## 5 小结

- **对偶系统**：$A \to A^T$、$B \to C^T$、$C \to B^T$，信息流方向反转。把 $(A, B, C)$ 换成 $(A^T, C^T, B^T)$ 即得对偶。
- **对偶原理**：$\Sigma$ 能控 ⇔ $\Sigma^d$ 能观；$\Sigma$ 能观 ⇔ $\Sigma^d$