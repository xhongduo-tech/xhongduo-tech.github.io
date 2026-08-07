---
title: 正交分解定理
date: 2026-08-07
---

# 正交分解定理

<div class="epigraph">
<p>把向量投影到子空间上，剩余的部分垂直落下——Hilbert 空间的一切几何都浓缩在这一分解里。</p>
<footer>—— 冯 · 诺伊曼（John von Neumann），意译</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛函分析 ｜ 程其襄《泛函分析》§4.5 ｜ 2026-08-07</p>
</div>

## 为什么正交分解是几何核心

在欧氏空间里，把一个向量分解为「沿平面的分量 + 垂直平面的分量」是理所当然的。Hilbert 空间的**正交分解定理（orthogonal decomposition theorem）**把这句话推广到无穷维：**给定一个闭子空间 $M$，空间里的每个向量 $x$ 都能唯一地写成 $x = m + z$，其中 $m \in M$、$z \perp M$。** 这个定理是整章几何的枢纽：它同时给出最佳逼近（$m$ 是 $x$ 在 $M$ 中最近的点）、投影算子（$m = P_M x$）、以及正交补（$z \in M^\perp$）。Riesz 表示定理、对偶理论、微分方程的变分方法，全都建立在这条分解上。<span class="marginnote">正交分解在有限维是「垂直投影」，在 $L^2$ 里是「把函数拆成『属于子空间的部分』与『与其垂直的部分』」——信号处理中把信号分解为「信号+噪声」，概率论中把随机变量分解为「关于子代数可测 + 鞅差」，都是这个定理的面目。它与概率论里的<strong>条件期望</strong>是同构的（第十章可链接）。</span>

## 1 正交补

**定义**：设 $M \subset H$ 是子集（不必是子空间）。定义 $M$ 的**正交补（orthogonal complement）**

$$
M^\perp = \{ x \in H \mid \langle x, m\rangle = 0,\ \forall m \in M \}
$$

**性质**：

- $M^\perp$ 是 $H$ 的**闭**线性子空间（用内积的连续性验证：$x_n \in M^\perp$ 且 $x_n \to x$，则 $\langle x, m\rangle = \lim\langle x_n, m\rangle = 0$）。
- $M \cap M^\perp = \{0\}$（$x \in M \cap M^\perp \Rightarrow \langle x,x\rangle = 0 \Rightarrow x = 0$）。
- $M \subset (M^\perp)^\perp$，且 $M$ 闭时 $M = (M^\perp)^\perp$（由正交分解定理）。

**核心要点：正交补把「垂直」变成「补空间」的构造工具**——它不依赖维数，在无穷维照样工作，这是 Hilbert 空间最优雅的机制之一。<span class="marginnote">$M^\perp$ 总是闭的，这解释了为什么正交分解要求 $M$ 闭：<strong>如果不要求 $M$ 闭，分解里的 $m$ 可能跑出 $M$</strong>。闭性 = 极限封闭 = 投影目标锁定在 $M$ 内。这是「闭子空间才能做投影」的全部理由。</span>

## 2 正交分解定理

**定理（正交分解 / 投影定理）：设 $H$ 是 Hilbert 空间，$M$ 是 $H$ 的闭线性子空间。则对每个 $x \in H$，存在唯一的分解**

$$
x = m + z, \qquad m \in M, \quad z \in M^\perp
$$

**即 $H = M \oplus M^\perp$（正交直和）。**

证明的存在性部分（这是整个定理的精髓，值得细读）：

- **第一步，取极小化序列**：设 $d = d(x, M) = \inf_{y \in M}\|x - y\|$，取 $y_n \in M$ 使 $\|x - y_n\| \to d$（下确界的逼近序列）。
- **第二步，平行四边形公式**：对 $y_n, y_m \in M$，它们的平均 $(y_n + y_m)/2 \in M$（$M$ 线性），故 $\|x - (y_n+y_m)/2\| \ge d$。用平行四边形公式：

$$
\|y_n - y_m\|^2 = 2\|x - y_n\|^2 + 2\|x - y_m\|^2 - 4\Big\|x - \frac{y_n+y_m}{2}\Big\|^2 \le 2\|x-y_n\|^2 + 2\|x-y_m\|^2 - 4d^2 \to 0
$$

故 $\{y_n\}$ 是柯西列，收敛到 $m \in M$（$M$ 闭）。
- **第三步，验证垂直**：对任意 $y \in M$，考虑 $f(t) = \|x - m - ty\|^2$，它在 $t = 0$ 处取最小值（因为 $m$ 是最近点），故 $f'(0) = -2\operatorname{Re}\langle x - m, y\rangle = 0$。由 $y$ 任意，$x - m \perp M$，即 $z = x - m \in M^\perp$。

**唯一性**：若 $x = m_1 + z_1 = m_2 + z_2$，则 $m_1 - m_2 = z_2 - z_1 \in M \cap M^\perp = \{0\}$。<span class="marginnote">证明里的第二步是「平行四边形公式的力量」：它把「$y_n$ 离 $x$ 都很近」转化为「$y_n$ 彼此很近」（柯西性）。这个技巧叫<strong>「平行四边形取中点」</strong>，在变分法、最佳逼近里反复出现——它告诉我们：极小化序列自动收敛，无需额外紧性假设。Hilbert 空间因此比一般 Banach 空间「软」得多。</span>

## 3 公式解析：极小化序列的柯西性

把第二步中最关键的不等式拆开：

$$
\|y_n - y_m\|^2 \le 2\|x - y_n\|^2 + 2\|x - y_m\|^2 - 4d^2
$$

- **第一步，平行四边形公式**：以 $a = x - y_n$、$b = x - y_m$ 为邻边：

$$
\|a - b\|^2 + \|a + b\|^2 = 2\|a\|^2 + 2\|b\|^2
$$

注意 $a - b = y_m - y_n$，$a + b = 2x - (y_n + y_m)$。
- **第二步，代入**：$\|y_m - y_n\|^2 = 2\|x - y_n\|^2 + 2\|x - y_m\|^2 - 4\|x - (y_n+y_m)/2\|^2$。
- **第三步，用 $d$ 控制**：因为 $(y_n + y_m)/2 \in M$，$\|x - (y_n+y_m)/2\| \ge d$，故 $-4\| \cdot \|^2 \le -4d^2$，得到不等式。
- **第四步，取极限**：$\|x - y_n\|^2 \to d^2$、$\|x - y_m\|^2 \to d^2$，右边 $\to 2d^2 + 2d^2 - 4d^2 = 0$。

**关键**：整个推导的支点是「$M$ 线性（中点还在 $M$）」+「平行四边形公式（把距离平方展开）」——**两个条件缺一不可**。这也从反面解释了：为什么非 Hilbert 的 Banach 空间里「最佳逼近存在」要额外依赖紧性或严格凸性。

## 4 投影算子

正交分解 $x = m + z$ 中的 $m$ 定义了**正交投影算子（orthogonal projection）**

$$
P_M : H \to M, \qquad P_M x = m
$$

它满足：**幂等（$P_M^2 = P_M$）、自伴（$\langle P_M x, y\rangle = \langle x, P_M y\rangle$）、范数 $\le 1$、$\ker P_M = M^\perp$、$\operatorname{ran} P_M = M$**。下一节将专门研究投影算子与最佳逼近。<span class="marginnote">投影算子是「把一切垂直分量的信息扔掉」的算子。它在最小二乘、信号滤波、量子测量（投影公设）中处处出现。<strong>「幂等 + 自伴」刻画正交投影</strong>，这个刻画在第十章「投影算子与谱分解」里会再次登场。</span>

## 5 正交分解的应用速写

**应用一（$M^\perp$ 的二次正交补）**：$M$ 闭时 $(M^\perp)^\perp = M$。分解 $H = M \oplus M^\perp$ 后，两边再取正交补即得。

**应用二（解线性方程）**：方程 $Ax = b$（$A$ 自伴有界）可先做正交分解——把 $b$ 分解为 $\operatorname{ran}A$ 与 $(\operatorname{ran}A)^\perp$ 两部分，落在后者上的部分无法由 $A$ 生成，说明方程无解；落在前者上的部分则对应解。**Fredholm 理论、变分法（第十章）都建立在这条观察上。**<span class="marginnote">「方程可解 ⟺ 右端正交于值域的正交补」这句话，是 <strong>Fredholm 二择一</strong>（第八章）的雏形：$Ax = b$ 有解当且仅当 $b \perp \ker A^*$。把一个解方程问题化成「正交性检验」，是 Hilbert 空间方法论的核心。</span>

**应用三（条件期望的抽象形式）**：设 $(\Omega, \mathcal{F}, P)$ 是概率空间，$\mathcal{G} \subset \mathcal{F}$ 是子 $\sigma$-代数，$X \in L^2$。则条件期望 $E[X|\mathcal{G}]$ 恰是 $X$ 在闭子空间 $L^2(\Omega, \mathcal{G}, P)$ 上的正交投影——正交分解给出条件期望的存在性，这是概率论与泛函分析最深层的连接之一。

## 6 例题精讲：正交分解的三个应用

**例题一：$L^2$ 中分解出常数函数**。

- 子空间 $M$ = 常函数（一维）。$M^\perp$ = 平均为零的函数。
- 对任意 $f$，分解 $f = c + (f - c)$，其中 $c = \frac{1}{b-a}\int f$（平均值）。
- $c \in M$，$f - c \in M^\perp$（$\int(f-c) = 0$）。这是「去均值」操作。

**例题二：条件期望 = 正交投影**。

- 概率空间上 $X \in L^2$，$\mathcal{G}$ 是子 $\sigma$-代数。
- $E[X|\mathcal{G}]$ 是 $X$ 在 $L^2(\Omega,\mathcal{G})$ 上的正交投影。
- 正交分解给出条件期望存在性——概率论与泛函分析最深层的连接。

**例题三：解方程的 Fredholm 视角**。

- 方程 $Tx = b$（$T$ 有界）。$b = m + z$（$m \in \overline{\operatorname{ran}T}$，$z \in (\operatorname{ran}T)^\perp = \ker T^*$）。
- $z$ 部分无法被 $T$ 生成：$b$ 必须正交于 $\ker T^*$ 才有解。
- 这是 Fredholm 二择一（第八章）的雏形。

**核心要点**：正交分解的三个应用——去均值、条件期望、解方程——都靠「$H = M \oplus M^\perp$」这一条分解。

**辨析｜易错点：** 正交分解要求 $M$ 闭；$\overline{\operatorname{ran}T}$ 自动闭，但 $\operatorname{ran}T$ 本身可能不闭——解方程时要分清。


## 7 小结

- **正交补** $M^\perp$：恒为闭子空间，$M \cap M^\perp = \{0\}$，闭 $M$ 时 $(M^\perp)^\perp = M$。
- **正交分解定理**：闭子空间 $M$ 上 $x = m + z$ 唯一分解（$m \in M$、$z \perp M$），即 $H = M \oplus M^\perp$。
- **证明三件套**：极小化序列 + 平行四边形公式（取中点得柯西性）+ 求导验证垂直。
- **投影算子** $P_M$：幂等、自伴、$\|P_M\| \le 1$，核与值域分别指向 $M^\perp$ 与 $M$。
- **应用**：方程可解性（正交于值域补）、条件期望 = $L^2$ 正交投影、最佳逼近。

在下一节，我们聚焦投影算子本身——**投影算子与最佳逼近**：证明「$P_M x$ 是 $x$ 在 $M$ 中的唯一最近点」，并给出最佳逼近的存在性、唯一性与刻画。
