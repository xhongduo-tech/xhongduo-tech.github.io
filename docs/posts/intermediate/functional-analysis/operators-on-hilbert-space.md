---
title: 希尔伯特空间上的算子（伴随、自伴算子）
date: 2026-08-07
---

# 希尔伯特空间上的算子（伴随、自伴算子）

<div class="epigraph">
<p>无穷！没有其他问题如此深刻地触动人类的精神。</p>
<footer>—— 大卫 · 希尔伯特（David Hilbert）</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛函分析 ｜ 程其襄《泛函分析》第4章 ｜ 2026-08-07</p>
</div>

## 为什么希尔伯特空间上「算子」更丰富

在一般 Banach 空间里，算子之间的研究要靠对偶算子绕路。但希尔伯特空间有一件独占的法宝——**内积**。有了内积，每个算子都自动拥有一个「镜像」：**伴随算子**。这是 Banach 世界没有的奢侈品，它直接孕育了自伴算子、酉算子、投影算子这一整个家族。<span class="marginnote"><strong>伴随算子的物理意义在量子力学（第10篇）里就是「转置复共轭」</strong>：可观测量对应自伴算子，时间演化对应酉算子。当你听到「矩阵的共轭转置 $A^\dagger$」时，那正是有限维版的伴随。</span>

这一篇梳理：伴随如何定义、自伴与正规酉的区别、谱的开头（见第九篇完整谱理论）、以及投影算子这个自伴特例。

## 1 伴随算子：内积带来的镜像

**核心概念（伴随算子）**：设 $H_1, H_2$ 是希尔伯特空间，$T \in \mathcal{B}(H_1, H_2)$。由 Riesz 表示定理，对每个 $y \in H_2$，映射 $x \mapsto \langle Tx, y\rangle$ 是 $H_1$ 上的连续线性泛函，故存在唯一 $T^*y \in H_1$ 使

$$
\langle Tx, y\rangle = \langle x, T^*y\rangle \quad (\forall x \in H_1)
$$

$T^*: H_2 \to H_1$ 称为 $T$ 的**伴随算子（adjoint）**。

**基本性质**（与有限维共轭转置一一对应）：

1. $(T + S)^* = T^* + S^*$，$(\alpha T)^* = \overline{\alpha}T^*$；
2. $(TS)^* = S^*T^*$（**反序**，转置家族的传统）；
3. $(T^*)^* = T$（自反）；
4. $\|T^*\| = \|T\|$，且 $\|T^*T\| = \|T\|^2$（**C\* 恒等式**）；
5. $T$ 可逆 ⟺ $T^*$ 可逆，且 $(T^{-1})^* = (T^*)^{-1}$。

**公式解析：伴随算子的「唯一自然性」。** 为什么 $\langle Tx, y\rangle = \langle x, T^*y\rangle$ 是对的规则？

- **第一步**：固定 $y$，$x \mapsto \langle Tx, y\rangle$ 是线性泛函（$T$ 线性 + 内积线性）。
- **第二步**：它连续，因为 $|\langle Tx, y\rangle| \le \|T\|\|x\|\|y\|$。
- **第三步**：Riesz 表示定理给出唯一的「代表元」——把这个代表元记作 $T^*y$，就得到伴随。**伴随的存在完全免费：它是 Riesz 表示定理的自动产物。**

## 2 自伴、正规、酉：三个关键类

**核心概念**：设 $T \in \mathcal{B}(H, H)$。

- **自伴（self-adjoint）**：$T^* = T$，即 $\langle Tx, y\rangle = \langle x, Ty\rangle$。物理上对应**可观测量**（实数型特征值）。
- **正规（normal）**：$TT^* = T^*T$。自伴与酉都是正规特例。
- **酉（unitary）**：$T^*T = TT^* = I$，即 $\|Tx\| = \|x\|$ 且满射。物理上对应**时间演化/旋转**。

**重点：自伴 ⟺ $\langle Tx, x\rangle$ 恒为实数。** 且对复希尔伯特空间，$\langle Tx, x\rangle = 0$ 对所有 $x$ ⟹ $T = 0$（极化恒等式）——**自伴性的验证只需检查 $\langle Tx, x\rangle$**。<span class="marginnote"><strong>自伴 vs 对称的陷阱到无界算子（第九篇）才真正致命</strong>：有界时两者一致，无界时「对称（$\langle Tx,y\rangle = \langle x,Ty\rangle$，定义域同）」不再自动蕴含自伴（还需定义域相等）。自伴的谱落在实轴上——量子力学特征值实测必为实数的数学根据。</span>

**例：$L^2$ 上的乘算子**。$T_\phi f = \phi f$（$\phi$ 实值本质有界）：$T_\phi^* = T_{\overline\phi} = T_\phi$，自伴。**特征值问题化为「$\phi$ 取常数的点」**——谱理论的种子。

**例：$\mathbb{C}^2$ 上的矩阵**。$\begin{pmatrix}0&1\\1&0\end{pmatrix}$ 自伴；$\begin{pmatrix}0&-i\\i&0\end{pmatrix}$ 自伴；$\begin{pmatrix}0&1\\-1&0\end{pmatrix}$ 反对称、$\begin{pmatrix}0&1\\-1&0\end{pmatrix}$ 酉。

## 3 自伴算子的谱：实的、非空的

**定理（自伴算子的谱）**：设 $T$ 是希尔伯特空间上的**有界自伴算子**，则

1. 谱 $\sigma(T) \subset \mathbb{R}$；
2. $\sigma(T) \subset [m, M]$，其中 $m = \inf_{\|x\|=1}\langle Tx, x\rangle$、$M = \sup_{\|x\|=1}\langle Tx, x\rangle$，且 $m, M \in \sigma(T)$；
3. $\|T\| = \max(|m|, |M|)$（谱半径等于范数）。

**直觉**：自伴算子的「特征值候选」全落在实轴上，且「上界 $M$、下界 $m$」本身都在谱里——**谱不是空壳，它被 $m, M$ 钉住**。

**证明第 2 点的骨架（反证）**：若 $M \notin \sigma(T)$，则 $M - T$ 有界可逆；$M - T$ 是自伴（$M$ 是数）且满足 $\langle (M - T)x, x\rangle \ge 0$ 恒成立，加上有界可逆，可得 $\langle (M - T)x, x\rangle \ge \delta\|x\|^2$，从而 $M - T$ 下有正定下界——与 $\sup \langle Tx, x\rangle = M$ 矛盾。**「上下界被谱钉住」是自伴性特有的刚硬。**

**公式解析：为什么 $\|T\| = \sup_{\|x\|=1}\langle Tx, x\rangle$（自伴时）。**

$$
\sup_{\|x\|=1}\langle Tx, x\rangle \le \|T\|\quad\text{平凡；反向用极化恒等式}
$$

- **第一步**：$\langle Tx, y\rangle = \tfrac{1}{4}[\langle T(x+y), x+y\rangle - \langle T(x-y), x-y\rangle + i\langle T(x+iy),x+iy\rangle - i\langle T(x-iy),x-iy\rangle]$（自伴性使内积化简）。
- **第二步**：每一项都被 $M\|z\|^2$ 控制，代入 $\|x\|=\|y\|=1$ 得 $|\langle Tx, y\rangle| \le M$。
- **第三步**：$|\langle Tx, y\rangle| \le M$ 对一切单位向量 ⟹ $\|T\| \le M$。**「数值域」$\langle Tx, x\rangle$ 决定算子范数——自伴算子的全部信息浓缩在实值二次型里。**

## 4 投影算子：自伴幂等的典范

**核心概念（正交投影）**：$P$ 称为**正交投影**，若 $P^2 = P$ 且 $P^* = P$（幂等 + 自伴）。

**定理**：$P$ 是正交投影 ⟺ $P = P_M$（某闭子空间 $M = \operatorname{ran}P$ 的正交投影）。且 $P$ 是投影时 $M^\perp = \ker P$。

**互余性**：$P_M + P_{M^\perp} = I$；$P_M P_{M^\perp} = 0$。

**公式解析：$P^2 = P$ 与 $P^* = P$ 为何「恰好」是投影。**

- **第一步**：$P^2 = P$ 表示「再投影一次不变」——$\operatorname{ran}P$ 上的点不动，这正是投影的定义。
- **第二步**：$P^* = P$（自伴）保证「投影的方向是正交的」而非斜的——$\langle Px, (I-P)y\rangle = \langle x, P(I-P)y\rangle = 0$。
- **第三步**：两者结合 ⟹ $H = \operatorname{ran}P \oplus \ker P$ 正交直和（第六篇的分解定理），$P$ 就是取 $\operatorname{ran}P$ 分量。**幂等给「投影」的身份，自伴给「正交」的品格。**

## 5 紧自伴算子的谱定理（预告）

在一般 Banach 空间里，紧算子（第八篇）已经有很好的谱理论（Riesz-Schauder）。在希尔伯特空间上，**紧自伴算子**更进一步——有**完整的对角化**：

**定理（谱定理，紧自伴版）**：设 $T$ 是希尔伯特空间上的紧自伴算子，则 $H$ 有一组规范正交基 $\{e_n\}$，使 $T$ 在这组基下**对角**：

$$
Tx = \sum_n \lambda_n \langle x, e_n\rangle e_n, \qquad \lambda_n \in \mathbb{R}, \ \lambda_n \to 0
$$

**特征向量构成正交基——这是有限维「实对称矩阵正交对角化」的无穷维翻版**，也是积分方程、Sturm-Liouville 问题、物理中本征展开的全部理论根据。<span class="marginnote"><strong>紧自伴算子的谱定理是整个学科「最漂亮」的一个定理</strong>：无穷维对象突然变成「对角矩阵」。第九篇《谱理论初步》与《紧算子的谱理论》会给出完整证明链，这里先立住结论。</span>

**例子（积分算子的谱展开）**：$T f(s) = \int_a^b K(s,t) f(t)\,dt$，$K$ 连续对称（$K(s,t) = \overline{K(t,s)}$），则 $T$ 紧自伴，特征函数 $\{e_n\}$ 构成 $L^2(a,b)$ 的基，且 $K(s,t) = \sum \lambda_n e_n(s)\overline{e_n(t)}$（**Mercer 定理**，$K$ 半正定时一致收敛）——积分方程理论（第五级《积分方程》专篇）的地基。

## 6 小结

- **伴随**：$\langle Tx, y\rangle = \langle x, T^*y\rangle$ 唯一确定 $T^*$；Riesz 表示自动给出。
- **反序与范数**：$(TS)^* = S^*T^*$，$\|T^*T\| = \|T\|^2$。
- **三大类**：自伴（谱实、数值域实）、正规（$TT^* = T^*T$）、酉（保范且满）。
- **数值域**：自伴时 $\|T\| = \sup_{\|x\|=1}\langle Tx,x\rangle$——信息全在实二次型。
- **投影**：幂等 + 自伴 ⟺ 正交投影；$P_M + P_{M^\perp} = I$。
- **紧自伴谱定理**：特征向量成基，$T$ 对角化，$\lambda_n \to 0$——有限维正交对角化的无穷维继承。

在下一节，我们把「紧」与「谱」正式连接——**紧算子与谱理论（Fredholm 二择一、谱分解）**。
