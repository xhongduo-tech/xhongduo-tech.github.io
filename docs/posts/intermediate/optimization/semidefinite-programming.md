---
title: 半定规划（SDP）初步
date: 2026-08-07
---

# 半定规划（SDP）初步

<div class="epigraph">
<p>只能使用一次的想法是雕虫小技；能够反复使用的想法才成为方法。</p>
<footer>—— 乔治 · 波利亚（George Pólya）</footer>
</div>

<div class="article-byline">
<p>第二级 · 最优化理论 ｜ Boyd《Convex Optimization》§4.6 ｜ 2026-08-07</p>
</div>

## 为什么最后的锥是「半正定矩阵锥」

SOCP 把约束推广到二阶锥，已经能表达二次与范数问题。但有一大类问题它仍表达不了——**矩阵变量的问题**：最小化最大特征值、找满足正定性的矩阵、组合优化的松弛，这些都要求约束写在矩阵空间里。**半定规划（Semidefinite Programming, SDP）**把锥换成**半正定矩阵锥** $\mathbb{S}^n_+$，是锥规划家族里表达力最强、也最优雅的一员。<span class="marginnote">半正定锥记作 $X \succeq 0$，即「$X$ 对称且所有特征值非负」。它是「非负实数」在矩阵世界的化身：$\mathbb{R}_+$ 上的一切优化直觉，在 $\mathbb{S}^n_+$ 上都有对应物，只是维度变高了。</span>

## 1 从标量到矩阵：半正定锥

回顾锥的概念：$\mathcal{K}$ 是凸锥，若 $x \in \mathcal{K}$ 且 $\lambda \ge 0$ 则 $\lambda x \in \mathcal{K}$。半正定矩阵锥

$$
\mathbb{S}^n_+ = \{ X \in \mathbb{S}^n \mid X \succeq 0 \}
$$

满足全部锥的公理：$X, Y \succeq 0$ 时 $X + Y \succeq 0$（特征值之和非负）、$\lambda X \succeq 0$（$\lambda \ge 0$）。它内部是**正定矩阵锥** $\mathbb{S}^n_{++}$（特征值全为正），边界是半正定但奇异的矩阵。<span class="marginnote">几何直觉：$\mathbb{S}^n_+$ 像一个倒扣的「雪糕筒」嵌在 $\mathbb{R}^{n(n+1)/2}$ 维空间中。n=1 时它退化为非负实数锥 $\mathbb{R}_+$——LP 的锥；n=2 时它是三维空间里的一个实心旋转圆锥——SOCP 的锥。层级关系因此是自然的：LP ⊂ SOCP ⊂ SDP。</span>

半正定锥有一个常用等价刻画，贯穿整个 SDP 理论：**$X \succeq 0$ 当且仅当对一切 $v$ 有 $v^T X v \ge 0$**。这个「对所有方向非负」的性质，是后续一切矩阵不等式验证的基石。

## 2 SDP 的两种标准形式

SDP 有两种等价写法，各有用处。

**向量形式（free-variable form）**：变量是普通向量 $x$，约束是「一个仿射矩阵函数半正定」：

$$
\min_x\ c^T x \quad \mathrm{s.t.}\quad F(x) := F_0 + x_1 F_1 + \cdots + x_n F_n \succeq 0
$$

其中 $F_i$ 是对称矩阵。$F(x) \succeq 0$ 是一条「矩阵不等式约束」——一个仿射函数取值于半正定锥。<span class="marginnote">这种形式最适合表达「矩阵特征值问题」：比如约束 $\lambda_{\max}(A + \mathrm{diag}(x)) \le t$ 可改写为 $tI - A - \mathrm{diag}(x) \succeq 0$，一条 SDP 约束。</span>

**矩阵形式（inequality form）**：变量是矩阵 $X$，目标与约束用迹（trace）：

$$
\min_X\ \mathrm{tr}(CX) \quad \mathrm{s.t.}\quad \mathrm{tr}(A_i X) = b_i,\ \ X \succeq 0
$$

两种形式通过「把对称矩阵展开成向量、把内积 $\mathrm{tr}(A^TB)$ 看成标准内积」互相转换。<span class="marginnote">矩阵内积 $\langle A, B \rangle = \mathrm{tr}(A^T B)$ 把 $\mathbb{S}^n$ 变成一个欧氏空间，SDP 于是可以看成「在矩阵欧氏空间里、半正定锥上」的线性规划——这正是「SDP = 矩阵世界的 LP」这一说法的来源。</span>

## 3 SDP 的表达力：Schur 补与特征值约束

SDP 之所以强大，是因为一大类「看起来不是矩阵」的约束能等价翻译成矩阵半正定不等式。最常用的工具是**Schur 补引理**：

$$
\begin{bmatrix} A & B \\ B^T & C \end{bmatrix} \succeq 0 \quad\iff\quad C \succ 0 \ \text{ 且 } \ A - B C^{-1} B^T \succeq 0
$$

这个等价关系把「分块矩阵半正定」翻译成「一个更小矩阵 $A - BC^{-1}B^T$ 半正定」，后者正是 Schur 补。<span class="marginnote">应用最广的场景是二次约束：$\|Ax + b\|_2^2 \le c^Tx + d$ 等价于 $\begin{bmatrix} (c^Tx+d)I & Ax+b \\ (Ax+b)^T & 1 \end{bmatrix} \succeq 0$——二次约束「开方」成了线性矩阵不等式（LMI）。</span>QCQP、范数约束、鲁棒约束的一大半都能用 Schur 补落进 SDP。

**特征值问题的 SDP 化**：约束「$X$ 的最大特征值不超过 $t$」写作 $tI - X \succeq 0$；「$X$ 的最小特征值至少 $t$」写作 $X - tI \succeq 0$。于是最小化 $\lambda_{\max}(X)$ 就成了 SDP：

$$
\min_{t, X}\ t \quad \mathrm{s.t.}\quad tI - X \succeq 0
$$

一个原本涉及「非线性特征值」的问题，变成纯线性目标 + 矩阵锥约束。

## 4 公式解析：最小化最大特征值

把「最小化最大特征值」完整推导一遍，看 SDP 如何把非线性目标线性化：

$$
\min_{X \in \mathbb{S}^n}\ \lambda_{\max}(X) \quad\Longleftrightarrow\quad \min_{X, t}\ t \quad \mathrm{s.t.}\quad tI - X \succeq 0
$$

- **第一步，理解 $\lambda_{\max}$ 的变分刻画**：$\lambda_{\max}(X) = \max_{\|v\|_2=1} v^T X v$——最大特征值就是「在所有单位方向里，二次型能取到的最大值」。它是一个凸函数（单位球上凸函数族的最大值）。
- **第二步，引入水位 $t$**：要求 $\lambda_{\max}(X) \le t$，等价于对所有单位 $v$ 有 $v^TXv \le t$，等价于 $tI - X \succeq 0$——**一条线性矩阵不等式**。
- **第三步，极小化 $t$**：目标从非线性的 $\lambda_{\max}$ 变成线性的 $t$，非线性全部被吸收进锥约束。

这套「用线性函数 + 锥约束取代非线性函数」的手法，是 SDP 建模的核心方法论：**任何能用矩阵不等式表达的凸约束，都能进入 SDP 的统一框架**。

## 5 SDP 的典型应用：从特征值到组合优化松弛

SDP 的应用横跨工程与理论：

- **最小化最大特征值**：结构力学里的柔度优化、通信里的功率分配，都化为「最小化最大奇异值/特征值」。
- **矩阵范数问题**：谱范数 $\|A\|_2$（最大奇异值）约束可写为 $\begin{bmatrix} tI & A \\ A^T & tI \end{bmatrix} \succeq 0$，SDP 约束。
- **SDP 松弛（relaxation）**：NP-hard 的组合优化问题（如 MAX-CUT）把 0-1 变量 $x_i \in \{\pm1\}$ 松弛成矩阵变量 $X$ 且 $X \succeq 0$，得到一个可以多项式求解的 SDP，再随机化取整恢复近似解——这就是 1995 年 Goemans–Williamson 的著名 0.878 近似算法。<span class="marginnote">SDP 松弛是「用凸松弛逼近组合困难」的典范：把离散的秩-1 约束 $X = xx^T$ 放宽成 $X \succeq 0$，丢掉非凸的秩约束，换来多项式时间。第八篇《整数规划》会用同类思想再做一次。</span>
- **控制与 LMI**：Lyapunov 稳定性条件 $A^TP + PA \prec 0$ 是典型 LMI，SDP 是求解这类矩阵不等式系统的标准工具。

**辨析｜易错点：**SDP 的规模增长很快——$n \times n$ 矩阵变量在「向量化」后有约 $n^2/2$ 个标量自由度，大规模 SDP 求解代价高，实践中常配合低秩结构或专用算法。另一个常见误区是把 $X \succeq 0$ 与「所有元素非负」混为一谈：$X \succeq 0$ 是特征值非负，与元素非负完全无关（比如 $\begin{bmatrix}1&2\\2&1\end{bmatrix}$ 元素全正但特征值为 $-1$ 与 $3$，不是半正定）。

## 6 数值算例：把一个二次约束变成 LMI

用 Schur 补把「看起来不是矩阵」的约束翻译成 SDP。设

$$
\min\ x_1 + x_2 \quad \mathrm{s.t.}\quad x_1^2 + x_2^2 \le 1
$$

**第一步，识别**：约束 $\|x\|_2^2 \le 1$ 是二次约束，用 Schur 补写成矩阵形式 $\begin{bmatrix} I & x \\ x^T & 1 \end{bmatrix} \succeq 0$（取 $A = I$、$B = x$、$C = 1$，Schur 补 $I - xx^T \succeq 0$ 等价于 $\|x\|_2 \le 1$）。
**第二步，问题成为 SDP**：$\min x_1 + x_2$ s.t. $\begin{bmatrix}1&0&x_1\\0&1&x_2\\x_1&x_2&1\end{bmatrix} \succeq 0$——一条线性矩阵不等式。
**第三步，几何解**：单位圆上最小化 $x_1 + x_2$，最优 $x = (-\frac{1}{\sqrt2}, -\frac{1}{\sqrt2})$，$p^* = -\sqrt2 \approx -1.414$。
**第四步，读出 SDP 的价值**：本例手工可解，但换成「$x_i$ 是矩阵、约束是 $X$ 的最大特征值 ≤ 1」之类的问题，手工就无能为力——SDP 的统一框架照样吃下。

**要点**：Schur 补是「二次 → 线性矩阵不等式」的万能钥匙；SDP 把一大类非线性约束线性化进锥，这就是它表达力冠绝锥规划家族的原因。

## 7 小结

- **半正定锥**：$\mathbb{S}^n_+ = \{X \mid X \succeq 0\}$，是矩阵世界的「非负实数」，满足 $v^TXv \ge 0$。
- **SDP 两种形式**：向量形式 $F_0 + \sum x_i F_i \succeq 0$；矩阵形式 $\min \mathrm{tr}(CX)$。
- **Schur 补**：分块矩阵半正定 ⇔ Schur 补半正定，把二次/范数约束翻译成 LMI。
- **特征值目标线性化**：$\min \lambda_{\max}(X)$ 化成线性目标 + $tI - X \succeq 0$。
- **应用**：矩阵范数、LMI 稳定性、MAX-CUT 的 SDP 松弛（GW 算法）。
- **易错点**：$X \succeq 0$ 是特征值非负，不是元素非负；SDP 规模增长快，需注意求解代价。
- **算例闭环**：$\min x_1+x_2$ s.t. $x_1^2+x_2^2\le1$ 经 Schur 补变 LMI、得 $p^*=-\sqrt2$——二次约束进锥全程手算。
- **变分刻画**：$\lambda_{\max}(X) = \max_{\|v\|=1} v^TXv$ 是凸函数——目标线性化的第一步。
- **GW 算法**：MAX-CUT 的 SDP 松弛 + 随机取整给出 0.878 近似——组合困难的凸松弛范本。

在下一节，我们把「幂函数 + 对数」的结构纳入优化——**几何规划（GP）**，它通过换元变成凸问题，是工程优化里另一件称手兵器。
