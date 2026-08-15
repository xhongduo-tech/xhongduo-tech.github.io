---
title: 线性系统的 Lyapunov 方程与稳定性判定
date: 2026-08-07
---

# 线性系统的 Lyapunov 方程与稳定性判定

<div class="epigraph">
<p>对线性系统而言，Lyapunov 方法不再是一种「艺术」，而是一条可以用矩阵代数直接求解的方程。</p>
<footer>—— 爱德华多 · 松塔格（Eduardo D. Sontag，*Mathematical Control Theory\*，1998）</footer>
</div>

<div class="article-byline">
<p>第二级 · 控制论与最优控制 ｜ Sontag《Mathematical Control Theory》Ch. 5 ｜ 2026-08-07</p>
</div>

## 为什么需要「可计算的」稳定性判据

上一节的 Lyapunov 直接法优雅而普适，却有一个工程上的痛点：**找 Lyapunov 函数没有通法**。但对线性系统 $\dot{x} = Ax$，事情可以做到极致——把「找一个二次型 $V$」这件事翻译成**一条矩阵方程**，解方程就能判定稳定性，还能顺手算出 Lyapunov 函数。这条方程叫**Lyapunov 方程（Lyapunov equation）**，是线性系统理论里反复出现的「工作马」：它判稳定性、算 Gramian、也是下一节极点配置和后面 LQR 里 Riccati 方程的近亲。

## 1 二次型 Lyapunov 函数

对线性系统，自然的候选 $V$ 是**二次型**：

$$
V(x) = x^T P x, \qquad P = P^T > 0 \text{（正定对称矩阵）}.
$$

$V$ 的等值面 $x^TPx = c$ 是椭球面，正定保证「$V$ 越大离原点越远」。沿 $\dot{x} = Ax$ 求导：

$$
\dot{V}(x) = \dot{x}^TPx + x^TP\dot{x} = (Ax)^TPx + x^TP(Ax) = x^T(A^TP + PA)x.
$$

所以「$\dot V \lt  0$」等价于「$A^TP + PA$ 负定」。<span class="marginnote">二次型求导的规矩是<strong>对称</strong>地每一项各碰一次：$\dot{x}^TPx$ 和 $x^TP\dot{x}$ 各贡献一半，合起来正好是 $A^TP + PA$——不是 $A^TP$ 或 $PA$ 单边，而是两者的和。这个「对称化」结构贯穿整个线性稳定性理论。</span>

于是稳定性问题变成纯线性代数问题：**找一个对称正定 $P$，使得 $A^TP + PA$ 负定。** 而 Lyapunov 方程把它变得更直接——不是验证「存在 $P$」，而是解出 $P$。

## 2 Lyapunov 方程与稳定性定理

**Lyapunov 方程（Lyapunov equation）**：

$$
A^T P + P A = -Q,
$$

其中 $Q$ 是任意给定的对称正定矩阵（通常取 $Q = I$ 最简单）。核心定理：

**线性系统稳定性定理**：$A$ 的所有特征值都具有负实部（$A$ 是 **Hurwitz** 矩阵），当且仅当对任意给定的对称正定 $Q$，Lyapunov 方程存在**唯一**的对称正定解 $P$。

这条定理把「稳定性」与「矩阵方程有正定解」完全等价起来。用法非常简单：

1. 任选正定 $Q$（如 $Q = I$）；
2. 解线性方程 $A^TP + PA = -Q$（关于 $P$ 的未知元是线性的，可用向量化方法求解）；
3. 检查 $P$ 是否正定：是 ⇒ 稳定；不是 ⇒ 不稳定。<span class="marginnote">$P$ 的正定性是结论的「证书」：一旦解出正定的 $P$，$V = x^TPx$ 就是现成的 Lyapunov 函数，渐近稳定性随之自动成立——这正是「给出一个 $V$ 就给出保证」的充分性在工程中的落地。而且这个判据对<strong>数值病态</strong>特别有辨识力：特征值靠近虚轴时 $P$ 会很大，提示稳定性余量不足。</span>

**例**：$A = \begin{bmatrix} 0 & 1 \\ -2 & -3 \end{bmatrix}$，取 $Q = I$。设 $P = \begin{bmatrix} p_{11} & p_{12} \\ p_{12} & p_{22} \end{bmatrix}$，解 Lyapunov 方程得 $P = \begin{bmatrix} 5/4 & 1/4 \\ 1/4 & 1/4 \end{bmatrix}$，主元皆为正（$p_{11} > 0$，$\det P = 1/4 > 0$），故 $A$ 稳定。验证特征值：$\lambda = -1, -2$，与判据一致。

## 3 与特征值判据的联系

线性系统稳定性的「经典」判据是看特征值：$\operatorname{Re}\lambda_i(A) \lt  0$ 对所有 $i$。Lyapunov 方程判据与它**等价**，但各有不可替代的价值：

| 判据 | 优点 | 缺点 |
| --- | --- | --- |
| 特征值判据 | 直接、直观、计算便宜 | 只给「稳/不稳」，不给余量 |
| Lyapunov 方程 | 给出 Lyapunov 函数与**稳定余量**；是后续优化（LQR）的框架 | 计算稍贵；$Q$ 的选择影响 $P$ 的病态 |

Lyapunov 方程还有一个重要的「副产品」：它与能控能观 Gramian 直接衔接。回想第 2 篇：当 $A$ 稳定时，可控 Gramian $W_c$ 与可观 Gramian $W_o$ 分别满足

$$
A W_c + W_c A^T = -BB^T, \qquad A^T W_o + W_o A = -C^TC.
$$

这两条正是 Lyapunov 方程（对 $A$ 与 $A^T$）。所以**「稳定系统 + 能控能观性」的 Gramian 计算，本质上就是在解 Lyapunov 方程**——前面第 2 篇的 Gramian，如今可以系统性地求出来。<span class="marginnote">这个联系不是巧合：$W_c = \int_0^\infty e^{At}BB^Te^{A^Tt}\mathrm{d}t$ 直接代入 Lyapunov 方程即可验证。它告诉我们，能控能观的「能量度量」与稳定性天然同框——这也是后面<strong>平衡实现</strong>（balanced realization）把稳定性与能控能观统一排序的出发点。</span>

## 4 公式解析：Lyapunov 方程从哪来

把「为什么是 $A^TP + PA = -Q$」推一遍，你会发现它只是上一节直接法在二次型下的「显式化」：

$$
\dot{V} = x^T(A^TP + PA)x \;\xrightarrow{\;\text{令其等于}\; -x^TQx\;}\; A^TP + PA = -Q.
$$

- **第一步，写出 $\dot V$**：$V = x^TPx$ 沿 $\dot x = Ax$ 的导数是 $x^T(A^TP + PA)x$，上一节已算过。
- **第二步，指定下降速率**：我们希望 $\dot V$ 是「负的二次型」。任取一个正定 $Q$，令 $\dot V = -x^TQx$。这个 $Q$ 不是凭空出现——它像「预设的能量耗散速度表」，不同 $Q$ 对应不同的「耗散预算」，但稳定性结论与 $Q$ 的选取无关。
- **第三步，逐项对比**：$x^T(A^TP + PA)x = x^T(-Q)x$ 对任意 $x$ 成立，当且仅当矩阵相等 $A^TP + PA = -Q$。

之所以「对任意正定 $Q$ 都有唯一正定解」就能判定稳定，是因为可以反向构造：如果 $A$ 稳定，取 $P = \int_0^\infty e^{A^Tt}Qe^{At}\mathrm{d}t$（收敛），直接代入验证它是解且正定；如果方程有正定解，则 $V = x^TPx$ 是 Lyapunov 函数，由直接法立即得到渐近稳定。**两条方向都有干净的证明，这正是「充要」二字的来源。**

**辨析｜易错点：** 常见错误有三个。其一，**$Q$ 必须正定（或至少半正定且可观对保证收敛）**：若 $Q$ 不正定，解出的 $P$ 可能无法判定。其二，**方程是「$A^TP + PA$」不是「$A^TPA - P$」**——后者是离散时间系统的 Lyapunov 方程（$\dot{x} = Ax$ 对应连续，$x_{k+1} = Ax_k$ 对应离散），两者形式不同、判据也不同，混用会得出完全错误的结论。其三，**稳定性的充分性要靠 $P$ 正定**：即使方程有解，解不正定就说明 $A$ 不稳定或 $Q$ 选得不合适，需要重解。

### 用 Lyapunov 方程算可控 Gramian

稳定系统 + 能控能观的 Gramian 是 Lyapunov 方程的解——这句话用起来有多顺手，跑一个完整的例子就清楚。取 $A = \begin{bmatrix} 0 & 1 \\ -2 & -3 \end{bmatrix}$（已验稳定）、$B = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$，求稳态可控 Gramian $W_c$。

由 $AW_c + W_cA^T = -BB^T$，设 $W_c = \begin{bmatrix} w_{11} & w_{12} \\ w_{12} & w_{22} \end{bmatrix}$，逐项展开矩阵方程，得到三个标量方程：

$$
-4w_{12} = 0, \qquad w_{11} - 3w_{12} - 2w_{22} = 0, \qquad 2w_{12} - 6w_{22} = -1.
$$

解得 $w_{12} = 0$，$w_{22} = 1/6$，$w_{11} = 1/3$，故 $W_c = \begin{bmatrix} 1/3 & 0 \\ 0 & 1/6 \end{bmatrix}$，正定——与能控性秩判据互相印证。

这条例子的工程含义：$W_c$ 的对角元分别度量「位置状态」与「速度状态」能被输入驱动的容易程度。若某个对角元接近零，说明对应方向几乎推不动——即便秩判据「满秩」，实际控制也要花巨大的力气。**Lyapunov 方程给出的不只是稳定性结论，还是「能控性有多强」的能量图谱**，这正是后面平衡实现（balanced realization）把稳定性与能控能观排序统一起来的起点。

### 一条方程，两种读法

Lyapunov 方程值得记住它的「双重身份」。**分析视角**：给定 $A$，解方程判定稳定并取 Lyapunov 函数；**设计视角**：第 4 篇的代数 Riccati 方程 $A^TP + PA - PBR^{-1}B^TP + Q = 0$ 正是 Lyapunov 方程加了二次项，而 Riccati 方程在 $\|K\| \to 0$ 的极限下退化回 Lyapunov 方程——最优反馈越弱，两者越接近。所以可以说：**Lyapunov 方程是「无控制的稳定性」，Riccati 方程是「最优控制的稳定性」**，中间隔着一条 $PBR^{-1}B^TP$。理解了这层递进，再看 LQR 的推导就会觉得水到渠成，而不是又背了一条新方程。

## 5 小结

- 线性系统的候选 Lyapunov 函数取**二次型** $V = x^TPx$，稳定性化为矩阵不等式 $A^TP + PA \lt  0$；这正是最常用的「试凑」形式。
- **Lyapunov 方程** $A^TP + PA = -Q$：$A$ Hurwitz ⇔ 对任意正定 $Q$ 存在唯一正定解 $P$。
- 用法：任取 $Q$（常用 $I$）→ 解方程 → 检查 $P$ 正定。
- 与**特征值判据**等价，但额外给出 Lyapunov 函数、稳定余量，并为 LQR 提供框架。
- **Gramian 是 Lyapunov 方程的解**：$AW_c + W_cA^T = -BB^T$，$A^TW_o + W_oA = -C^TC$，稳定性能控能观在此统一。
- 注意连续与离散 Lyapunov 方程形式不同，别混用。
- 实用流程：从 $Q = I$ 起步 → 解方程 → 检查 $P$ 正定；特征值靠近虚轴时 $P$ 病态放大，提示稳定余量不足。
- 递进关系：**Lyapunov 方程 = 无控制的稳定性，Riccati 方程 = 最优控制的稳定性**，中间隔着一条 $PBR^{-1}B^TP$。
- 连续与离散要分清：$\dot{x}=Ax$ 用 $A^TP + PA = -Q$，$x_{k+1}=Ax_k$ 用 $A^TPA - P = -Q$，混用会得出完全错误的结论。

在下一节，我们把「稳定性」的讨论从开环拉回闭环：从给定极点到反推增益，看**状态反馈与极点配置**如何让设计者拥有极点主导权。