---
title: 伴随算子与对偶算子
date: 2026-08-07
---

# 伴随算子与对偶算子

<div class="epigraph">
<p>每个算子都有一只「幽灵之手」伸向对偶空间——那就是对偶算子。</p>
<footer>—— 巴拿赫（Stefan Banach），意译</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛函分析 ｜ 程其襄《泛函分析》§7.7 ｜ 2026-08-07</p>
</div>

## 为什么算子也要「取对偶」

第四章我们定义了 Hilbert 空间上的伴随算子 $T^*$（满足 $\langle Tx, y\rangle = \langle x, T^*y\rangle$）。但对一般 Banach 空间没有内积，伴随该怎么定义？答案：**对偶算子（dual operator）** $T^*$——它把「作用在 $Y$ 上的泛函」拉回到「作用在 $X$ 上的泛函」：

$$
(T^* f)(x) = f(Tx), \qquad f \in Y^*,\ x \in X
$$

这是「$T$ 在对偶空间的镜像」。它的重要性在于：**算子的核与值域关系（值域闭、可解性）在对偶空间里看得更清楚**——第八章的 Fredholm 理论、第九章的谱理论都建立在对偶算子之上。<span class="marginnote">在 Hilbert 空间里，Banach 对偶算子与第四章的伴随算子通过 Riesz 表示定理互相转换：$T^*_{\mathrm{Banach}} = $ Riesz 之后再做 Hilbert 伴随。但 Banach 对偶算子不需要内积，适用于一切赋范空间——这是它更一般的地方。</span>

## 1 对偶算子的定义

**定义**：设 $X, Y$ 是赋范线性空间，$T \in \mathcal{B}(X, Y)$。定义**对偶算子（dual operator）**

$$
T^* : Y^* \to X^*, \qquad (T^* f)(x) = f(Tx)
$$

即对每个 $f \in Y^*$，$T^*f$ 是「先把 $x$ 送进 $T$，再让 $f$ 读」的复合泛函。

**验证线性与有界**：$T^*f$ 显然线性（复合线性映射）；有界性：

$$
|(T^*f)(x)| = |f(Tx)| \le \|f\|\|T\|\|x\| \Rightarrow \|T^*f\| \le \|T\|\|f\|
$$

故 $\|T^*\| \le \|T\|$。<span class="marginnote">直觉：$T^*$ 是「站在 $Y^*$ 里，把泛函沿 $T$ 拉回 $X^*$」——它是「$T$ 的箭头反过来」的唯一自然方式。记号里 $T^*$ 的方向从 $Y^*$ 到 $X^*$ 与 $T$ 相反，这与「转置矩阵 $A^T$ 的行列互换」一致。</span>

## 2 对偶算子的基本性质

**定理：对偶映射 $T \mapsto T^*$ 满足**：

1. $(T + S)^* = T^* + S^*$，$(\alpha T)^* = \alpha T^*$（线性）；
2. $(ST)^* = T^* S^*$（乘积反序）；
3. $\|T^*\| = \|T\|$（**范数保持**）；
4. $X$ 自反时 $(T^*)^* = T$（二次对偶回到自身）。

其中 $\|T^*\| = \|T\|$ 的证明用 Hahn-Banach：对每个 $x$，$\|Tx\| = \sup_{\|f\|\le1}|f(Tx)| \le \sup_{\|f\|\le1}\|T^*f\|\|x\| \le \|T^*\|\|x\|$，故 $\|T\| \le \|T^*\|$；反向 $\|T^*\| \le \|T\|$ 已证。<span class="marginnote">范数保持说明「取对偶」不损失算子的「大小」——$T$ 与 $T^*$ 是「一样大的镜像」。这条性质在谱理论里保证「$\sigma(T^*) = \sigma(T)$」类的谱对称性（第九章）。</span>

**核心要点：对偶算子是「算子世界的转置」**——线性、反序、保范，全部与矩阵转置的性质一一对应。

## 3 核与值域的对偶关系

对偶算子最重要的贡献，是核与值域的**正交补关系**：

$$
\ker T^* = (\operatorname{ran} T)^\perp, \qquad \overline{\operatorname{ran} T} = (\ker T^*)^\perp
$$

**证明第一式**：$f \in \ker T^* \iff T^*f = 0 \iff f(Tx) = 0\ \forall x \iff f \in (\operatorname{ran}T)^\perp$。

**证明第二式**：对第一式两边取正交补（在 $Y^{**}$ 中），用 $(M^\perp)^\perp = \overline M$，并注意 $Y$ 经 $\kappa$ 嵌入 $Y^{**}$。<span class="marginnote">这条关系是 Fredholm 理论的基石：<strong>方程 $Tx = b$ 可解（$b \in \overline{\operatorname{ran}T}$）当且仅当 $b$ 正交于 $\ker T^*$</strong>。它把「解方程」变成「正交性检验」——这是泛函分析解决线性方程的核心范式。</span>

**应用（可解性判据）**：若 $\operatorname{ran}T$ 闭，则 $Tx = b$ 有解 ⟺ $b \perp \ker T^*$。这就是 Fredholm 二择一的雏形（第八章将严格化）。

## 4 公式解析：对偶算子的作用方式

把 $T^*$ 的定义逐层拆开：

$$
(T^* f)(x) = f(Tx)
$$

- **第一步，识别方向**：$T: X \to Y$，$f \in Y^*$ 作用在 $Y$ 上。$T^*f$ 要作用在 $X$ 上，所以输入 $x \in X$。
- **第二步，唯一自然的定义**：$T^*f$ 在 $x$ 处的值，只能是「把 $x$ 送进 $T$ 得 $Tx \in Y$，再让 $f$ 读」——即 $f(Tx)$。这是唯一使「$(T^*f)(x) = f(Tx)$」成立的定义。
- **第三步，方向反了**：$T^* : Y^* \to X^*$（与 $T$ 反向）。$(ST)^* = T^*S^*$ 的「反序」正是方向反向的必然结果。
- **第四步，验证线性**：$T^*(\alpha f + \beta g)(x) = (\alpha f + \beta g)(Tx) = \alpha f(Tx) + \beta g(Tx)$——线性性逐点成立。

**关键**：对偶算子的定义没有任何选择余地——「箭头反向 + 复合」是唯一自然的构造。**$T^*$ 是 $T$ 的「必然镜像」**，这正是它在理论中反复出现的原因。

## 5 例题精讲：对偶算子的计算

**例题一：移位算子的对偶**。

- $S(x_1,x_2,\ldots) = (0,x_1,x_2,\ldots)$ 于 $l^1$。
- $(S^*f)(x) = f(Sx)$，$f \in (l^1)^* = l^\infty$。$S^*f = (f_2, f_3, \ldots)$（去掉首项）。
- $S^*$ 是前移位：$S^*f(x) = \sum_{n\ge1} f_{n+1} x_n$。

**例题二：积分算子的对偶**。

- $T_K f(s) = \int_0^1 K(s,t)f(t)\\,dt$ 于 $L^2$。
- 对偶算子 $T_K^*$（作为 Hilbert 伴随）是 $T_K^* g(t) = \int \overline{K(s,t)}g(s)\\,ds$。
- 自伴 ⟺ 核是 Hermite 的：$K(s,t) = \overline{K(t,s)}$。

**例题三：核与值域关系的应用**。

- $T: l^2 \to l^2$，$T(x_n) = (x_n/n)$。$\ker T = \{0\}$，$\operatorname{ran}T$ 稠密不闭。
- $\ker T^* = (\operatorname{ran}T)^\perp = \{0\}$（稠密子空间的正交补为零）。
- 但 $T^*$ 也不满——$\overline{\operatorname{ran}T} = l^2$，值域不闭，「可解性」只在闭包意义下成立。

**核心要点**：三个例题展示对偶算子的计算——移位（拉回泛函）、积分核（共轭交换）、核值域（正交补关系）——都来自「$T^*f = f \circ T$」这一条定义。

**辨析｜易错点：** 对偶算子与 Hilbert 伴随不同：Banach 对偶算子 $T^*: Y^* \to X^*$ 不需要内积；Hilbert 伴随 $T^*: H_2 \to H_1$ 通过 Riesz 表示把「对偶」翻译回「原空间」。两者记号相同但对象不同，阅读文献时要分清。

## 6 小结

- **对偶算子**：$(T^*f)(x) = f(Tx)$，$T^*: Y^* \to X^*$，是 $T$ 在对偶空间的镜像。
- **基本性质**：线性、反序 $(ST)^* = T^*S^*$、保范 $\|T^*\| = \|T\|$。
- **核值域关系**：$\ker T^* = (\operatorname{ran}T)^\perp$，$\overline{\operatorname{ran}T} = (\ker T^*)^\perp$。
- **可解性判据**：$\operatorname{ran}T$ 闭时 $Tx = b$ 有解 ⟺ $b \perp \ker T^*$。
- **与 Hilbert 伴随的区别**：对偶算子作用于对偶空间，不依赖内积。
- **定位**：Fredholm 理论（第八章）与谱理论（第九章）的共同地基。

在下一节，我们进入第八章——**紧算子的定义与基本性质**，看「有限秩算子的极限」如何把 Fredholm 理论与谱理论带上新的高度。
