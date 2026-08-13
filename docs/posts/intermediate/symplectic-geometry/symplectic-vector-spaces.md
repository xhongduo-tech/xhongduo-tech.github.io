---
title: 辛向量空间与辛线性代数
date: 2026-08-07
---

# 辛向量空间与辛线性代数

<div class="epigraph">
<p>哈密顿力学可以完全建立在辛几何之上；反之，辛几何的大部分概念都从哈密顿力学获得意义。</p>
<footer>—— 弗拉基米尔 · 阿诺尔德（V. I. Arnol'd）</footer>
</div>

<div class="article-byline">
<p>第二级 · 辛几何 ｜ McDuff & Salamon 第2章；Cannas 第2-3章 ｜ 2026-08-07</p>
</div>

## 为什么从辛线性代数开始

这个网站的主线是「从极限到大模型」。在第二级《理论力学》里你见过哈密顿方程，在《微分几何》里你接触过流形；而**辛几何**恰恰是这两者交汇的地方——它是「相空间」的几何学。任何几何研究都从无穷小开始：研究曲面先研究切平面，研究辛流形就先研究切空间上的**辛向量空间**。辛结构本质上是一个「反对称的内积」：它不像普通内积那样量出长度和夹角，而是量出「面积与定向」。整门辛几何可以看作一句话的展开：**在切空间上放一个非退化反对称双线性形式，然后看这个结构能走多远。** 这一篇我们只做线性代数，却是后面一切（Darboux 定理、Lagrangian 子流形、Floer 同调）的出发点。<span class="marginnote">从课程地图看，这一篇是辛几何全部二十四篇的地基：没有它，余切丛的辛结构、哈密顿流、moment map 都无从谈起。与《线性代数》里的欧氏内积对比着学，理解最深。</span>

## 1 辛向量空间的定义

**辛向量空间（symplectic vector space）**：设 $V$ 是实向量空间，$\omega: V \times V \to \mathbb{R}$ 是双线性形式，满足以下两条，则称 $\omega$ 为**辛形式**，称 $(V, \omega)$ 为辛向量空间：

1. **反对称性（skew-symmetry）**：$\omega(v, w) = -\omega(w, v)$，对所有 $v, w \in V$。特别地 $\omega(v, v) = 0$。<span class="marginnote">从 $\omega(v,v) = -\omega(v,v)$ 立即推出 $\omega(v,v)=0$。所以辛形式不允许「一个向量与自身的配对」非零——这与内积 $\langle v, v \rangle &gt; 0$ 截然不同。它天然带着「定向面积」的意味，就像平行四边形面积 $\det(u, v)$ 会随 $u, v$ 交换变号。</span>
2. **非退化性（nondegeneracy）**：若 $v \in V$ 满足 $\omega(v, w) = 0$ 对所有 $w \in V$ 成立，则 $v = 0$。

非退化性说的是：**每一个非零向量 $v$ 都能找到某个 $w$，使 $\omega(v, w) \neq 0$。** 换句话说，由 $\omega$ 诱导的线性映射

$$
V \longrightarrow V^*, \qquad v \mapsto \omega(v, \cdot)
$$

是一个同构。这个映射在辛几何里无处不在：它把「向量」翻译成「对偶空间的余向量」，也就是把切向量变成 1-形式——哈密顿方程里的「把梯度变成速度场」全靠它。

## 2 偶维数与辛基

一个深刻而简单的结论是：**辛向量空间必定是偶维数。** 因为反对称非退化矩阵是奇数阶行列式为 0 的，只有偶数阶才可能可逆。

**辛基（symplectic basis）**：设 $\dim V = 2n$，存在基 $e_1, \dots, e_n, f_1, \dots, f_n$ 满足

$$
\omega(e_i, e_j) = 0, \qquad \omega(f_i, f_j) = 0, \qquad \omega(e_i, f_j) = \delta_{ij}
$$

这样的基叫**辛基**，也叫 **Darboux 基**。<span class="marginnote">这个结果从线性代数看是「反对称双线性形式的规范形定理」：任何非退化反对称矩阵都能通过坐标变换化成标准分块形式。类比：对称正定矩阵可对角化到 $\delta_{ij}$，反对称非退化矩阵则化到 $\begin{pmatrix}0 & I \\ -I & 0\end{pmatrix}$。</span>

在辛基下，$\omega$ 的矩阵是

$$
J_0 = \begin{pmatrix} 0 & I_n \\ -I_n & 0 \end{pmatrix}, \qquad
\omega(x, y) = x^T J_0 y
$$

其中 $I_n$ 是 $n$ 阶单位矩阵。**任何辛向量空间都线性同构于 $(\mathbb{R}^{2n}, \omega_0)$**，这里 $\omega_0(x, y) = x^T J_0 y$ 叫**标准辛形式**。这就是线性层面的「齐次性」：辛结构没有曲率、没有形变余地，所有辛向量空间「长一个样」。

## 3 辛群与辛变换

保持辛结构的线性变换构成一个重要的李群。

**辛群（symplectic group）**：

$$
\mathrm{Sp}(2n, \mathbb{R}) = \{ A \in \mathrm{GL}(2n, \mathbb{R}) \mid A^T J_0 A = J_0 \}
$$

条件 $A^T J_0 A = J_0$ 等价于 $\omega_0(Ax, Ay) = \omega_0(x, y)$：**辛变换是「面积保持」的线性映射。** 类比记忆：正交群 $\mathrm{O}(2n)$ 保内积，辛群 $\mathrm{Sp}(2n,\mathbb{R})$ 保辛形式。<span class="marginnote">注意维数：$\dim \mathrm{Sp}(2n,\mathbb{R}) = n(2n+1)$，而 $\dim \mathrm{O}(2n) = n(2n-1)$。辛群比正交群「胖」一轮——因为它保的是反对称量，约束方程更宽松。$\mathrm{Sp}(2,\mathbb{R}) = \mathrm{SL}(2,\mathbb{R})$ 是唯一的例外（$n=1$ 时两者都是面积保持）。</span>

辛群的**李代数**是

$$
\mathfrak{sp}(2n, \mathbb{R}) = \{ X \in \mathfrak{gl}(2n,\mathbb{R}) \mid X^T J_0 + J_0 X = 0 \}
$$

它的元素叫**无穷小辛变换**。把指数映射 $\exp(tX)$ 代入 $A^T J_0 A = J_0$ 并在 $t=0$ 求导，就得到 $X^T J_0 + J_0 X = 0$。这些 $X$ 就是后面哈密顿向量场的线性雏形——它们的分量满足对称条件，恰好对应二阶常微分方程里的「本征值成对出现」。

**辨析｜易错点：** 初学时常把「辛」和「斜对称内积」混为一谈。辛形式是非退化的，但**反对称性意味着退化性很容易发生**：任何一维子空间在 $\omega$ 下都「退化」，因为 $\omega(v, v) = 0$。所以「非退化」是全局条件，不是逐点条件。一个向量与自身的辛配对永远为 0，这与正交归一完全不同——**辛正交与通常正交是两种正交，不要共用直觉**。

## 4 公式解析：非退化性的坐标翻译

**核心公式：**

$$
\omega(v, w) = \sum_{i=1}^{n} (x_i y_{n+i} - x_{n+i} y_i) = x^T J_0 y
$$

其中 $x = (x_1, \dots, x_{2n})$、$y = (y_1, \dots, y_{2n})$ 是在辛基下的坐标。逐项拆解：

- **第一步，看标准矩阵 $J_0$**：$J_0^T = -J_0$（反对称）、$J_0^2 = -I_{2n}$（这是「乘 $J_0$」作为复结构的关键性质）。$\omega_0(x, y) = x^T J_0 y$ 把矩阵乘法展开就是上面的求和式。
- **第二步，看求和结构**：和式只把「前 $n$ 个坐标」与「后 $n$ 个坐标」交叉配对，同半边配对项全部为零。这正是辛基条件的坐标版本：$e_i$ 与 $f_j$ 配对为 $\delta_{ij}$，$e_i$ 与 $e_j$、$f_i$ 与 $f_j$ 配对为零。
- **第三步，看非退化**：若 $x \neq 0$，比如 $x_i \neq 0$，取 $y = f_i$（即 $y_{n+i} = 1$，其余为 0），则 $\omega(x, y) = x_i \neq 0$。这就在坐标上证明了「非零向量必有非零配对」——非退化性在此表现为矩阵 $J_0$ 可逆。
- **第四步，与内积对比**：欧氏内积 $\langle x, y \rangle = \sum x_i y_i$ 是「对应分量相乘再求和」，而辛形式是「交叉分量相乘再求和」。这一字之差（$y_i \to y_{n+i}$）就是整个辛世界与黎曼世界的分水岭。

## 5 复结构与辛结构的联系

标准矩阵 $J_0$ 满足 $J_0^2 = -I$，因此它给 $\mathbb{R}^{2n}$ 一个**线性复结构**：定义 $i \cdot x := J_0 x$，就把 $\mathbb{R}^{2n}$ 变成 $\mathbb{C}^n$。事实上

$$
(z_1, \dots, z_n) = (x_1 + i x_{n+1}, \dots, x_n + i x_{2n})
$$

于是 $\omega_0$ 在这个复坐标下等于

$$
\omega_0 = \frac{i}{2} \sum_{k=1}^{n} dz_k \wedge d\bar{z}_k
$$

这是一个贯穿全篇的伏笔：**辛结构与复结构天然配对。** 在辛基下，$J_0$ 既是「乘 $i$」又保持 $\omega_0$，还给出正定内积 $g(x, y) = \omega_0(x, J_0 y)$。这个「辛形式 + 复结构 = 黎曼结构」的三角关系，正是后面《近复结构与相容三元组》的主角。<span class="marginnote">这里的 $dz_k \wedge d\bar{z}_k$ 已用到微分形式语言，属于《微分几何》的内容。现在只需记住结论：标准辛形式在复坐标下是对角的 $(1,1)$-形式。Gromov 的伪全纯曲线理论正是建立在这个恒等式之上。</span>

## 6 Lagrangian 子空间的预告

辛线性代数最特别的概念是**Lagrangian 子空间（Lagrangian subspace）**：$V$ 的 $n$ 维子空间 $L$，且 $\omega|_L \equiv 0$（即 $L$ 是**迷向（isotropic）**的）。迷向意味着「$L$ 里任意两个向量都不产生辛配对」；维数推到头（$n$ 维）就是 Lagrangian。

一个具体例子：在 $(\mathbb{R}^{2n}, \omega_0)$ 中，由 $e_1, \dots, e_n$ 张成的子空间 $\mathbb{R}^n \times \{0\}$ 是 Lagrangian 的（前半边自配对为零）；由 $f_1, \dots, f_n$ 张成的 $\{0\} \times \mathbb{R}^n$ 也是。Lagrangian 子空间全体构成一个流形，叫 **Lagrangian 格拉斯曼流形** $\Lambda(n)$，它是下一层几何（Lagrangian 子流形、Lagrangian 交点数）的种子。<span class="marginnote">从哈密顿力学看，位置坐标构成一个 Lagrangian 子空间，「位置-动量」则是一对互补的 Lagrangian 子空间。后面《Lagrangian 子流形与锥》会把它升级成流形版本。</span>

## 7 小结

- **辛向量空间** $(V, \omega)$：$\omega$ 是反对称、非退化的双线性形式；辛结构是「反对称内积」，度量面积与定向而非长度。
- **偶维数**与**辛基**：$\dim V = 2n$，存在辛基使 $\omega$ 化为标准形式 $\omega_0(x, y) = x^T J_0 y$；所有辛向量空间线性同构。
- **辛群** $\mathrm{Sp}(2n, \mathbb{R})$：保持 $\omega_0$ 的线性变换，是「面积保持群」，其李代数由 $X^T J_0 + J_0 X = 0$ 刻画。
- **复结构配对**：$J_0^2 = -I$ 给出复坐标，$\omega_0 = \tfrac{i}{2} \sum dz_k \wedge d\bar{z}_k$，辛—复—黎曼三方相容。
- **Lagrangian 子空间**：$n$