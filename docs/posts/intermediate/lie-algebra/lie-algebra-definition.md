---
title: 李代数的定义与基本性质
date: 2026-08-11
---

# 李代数的定义与基本性质

<div class="epigraph">
<p>在数学中，正如在其他地方一样，你并不会注意到世界的规律，除非你先把它们一一分解开来。</p>
<footer>—— 苏菲 · 热尔曼（Sophie Germain）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 李代数与李群 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从李代数开始

大模型的每一次参数更新、物理学的每一次对称性论证，背后都有一个共同动作：**在一族对象身上做「微小的」操作，观察线性化的效果**。李代数（Lie algebra）就是把这个动作抽出来单独研究的对象。群描述的是对称的「整体」，李代数描述的是对称在「无穷小」处的切线——它比群简单（它只是一个向量空间加上一个括号运算），却几乎完整地继承了群的全部信息。<span class="marginnote">后文第 14 篇《李群-李代数对应与指数映射》会证明：对连通李群，李代数同构于群的「无穷小邻域」，信息量几乎等价。这也是为什么 AI 与物理中大量工作直接操作李代数而不是李群——先线性化，再补全。</span>

## 1 从矩阵的交换子说起

矩阵乘法一般不满足交换律：$AB$ 与 $BA$ 未必相等。二者之差

$$[A, B] = AB - BA$$

叫作 $A$ 与 $B$ 的**交换子（commutator）**，它精确度量了「先做 $A$ 再做 $B$」与「先做 $B$ 再做 $A$」的差别。<span class="marginnote">量子力学中位置与动量的交换子 $[x, p] = i\hbar$ 是测不准原理的代数根源；交换子不恒为零，恰恰是「两个操作不可同时确定」的数学翻译。</span>

交换子自动满足两条恒等式：**反对称性** $[A, A] = 0$（因而 $[A, B] = -[B, A]$）与 **Jacobi 恒等式**：

$$[A, [B, C]] + [B, [C, A]] + [C, [A, B]] = 0$$

把「矩阵 + 交换子」这套结构抽象出来，就得到李代数的定义。

## 2 李代数的定义

**李代数（Lie algebra）**：设 $\mathbb{F}$ 是一个域，$L$ 是 $\mathbb{F}$ 上的向量空间，$L \times L \to L$ 上有一个双线性映射 $[\cdot, \cdot]$，满足：

1. **反对称**：$[x, x] = 0$ 对所有 $x \in L$；
2. **Jacobi 恒等式**：$[x, [y, z]] + [y, [z, x]] + [z, [x, y]] = 0$ 对所有 $x, y, z \in L$。

则称 $L$ 是 $\mathbb{F}$ 上的李代数，$[\cdot, \cdot]$ 叫**李括号（Lie bracket）**。<span class="marginnote">李括号不要求结合律。$[x, [y, z]] \neq [[x, y], z]$ 是常态——这正是「非结合代数」的典型：结合律让给了 Jacobi 恒等式。</span>

**辨析｜易错点：** 李括号的「乘法」与通常的乘法有本质区别：它**不满足结合律**，只满足更弱的 Jacobi 恒等式。初学者最容易犯的错误是把 Jacobi 恒等式当成结合律来记——它们是不同的东西，后者说的是 $[x,[y,z]] = [[x,y],z]$，而 Jacobi 恒等式说的是三个两两嵌套项之和为零。另外注意反对称在特征 $2$ 的域上需要单独要求 $[x,x]=0$（此时反对称与对称等价）。

## 3 子代数、理想、同态与同构

与群、环、向量空间一样，李代数也有完整的结构子对象体系。

**子代数（subalgebra）**：$L$ 的线性子空间 $K$，若对任意 $x, y \in K$ 有 $[x, y] \in K$，则 $K$ 是 $L$ 的子代数。此时 $K$ 自己也是李代数。

**理想（ideal）**：$L$ 的线性子空间 $I$，若对任意 $x \in I$ 与 $y \in L$ 有 $[x, y] \in I$，则称 $I$ 是 $L$ 的理想。理想的定义只要求「被外面的元素作用后仍留在里面」——方向性更强，因此理想是商结构的原料。

**商代数（quotient algebra）**：若 $I$ 是理想，则商空间 $L/I$ 上可定义 $[x + I, y + I] = [x, y] + I$，使其成为李代数，称为**商代数**。

**同态（homomorphism）**：$\phi: L \to L'$ 是线性映射且保持括号 $\phi([x, y]) = [\phi(x), \phi(y)]$，则称 $\phi$ 是李代数同态；双射同态叫**同构（isomorphism）**。<span class="marginnote">这条框架与第一级《抽象代数》中的群同态一一对应：同态像、同态核、第一同构定理 $L/\ker\phi \cong \operatorname{im}\phi$ 在李代数中都成立。学过抽象代数的读者可以把本章当成「同态套路的重演」。</span>

## 4 核心例子

有了定义，我们先认识五个反复出场的基本李代数。

**全体矩阵：$\mathfrak{gl}(n, \mathbb{F})$**。$n \times n$ 矩阵全体配上交换子 $[A, B] = AB - BA$，构成李代数，记作 $\mathfrak{gl}(n, \mathbb{F})$。它是所有矩阵型李代数的「母体」。

**迹为零：$\mathfrak{sl}(n, \mathbb{F})$**。由迹 $\operatorname{tr} A = 0$ 的矩阵组成。因为 $\operatorname{tr}(AB) = \operatorname{tr}(BA)$，所以 $\operatorname{tr}[A, B] = 0$，迹零矩阵在交换子下封闭，是 $\mathfrak{gl}(n,\mathbb{F})$ 的理想。这是后面**半单理论**的第一个例子。

**上三角矩阵：$\mathfrak{t}(n, \mathbb{F})$** 与**严格上三角矩阵：$\mathfrak{n}(n, \mathbb{F})$**。前者对角线以下全为零，后者对角线及以下全为零。二者都是 $\mathfrak{gl}(n,\mathbb{F})$ 的子代数，且 $\mathfrak{n}(n,\mathbb{F})$ 是 $\mathfrak{t}(n,\mathbb{F})$ 的理想。它们分别是最典型的**可解**与**幂零**李代数（见第 2 篇）。

**反称矩阵：$\mathfrak{so}(n, \mathbb{F})$**。满足 $A^T = -A$ 的矩阵。由 $\operatorname{tr}(A^T B) = \operatorname{tr}(A^T B)^T = -\operatorname{tr}(B^T A)$ 可证 $[A,B]^T = -[A,B]$，故反称矩阵封闭于交换子。它是旋转群的李代数（见第 14 篇）。

**求导算子：$\operatorname{Der}(A)$**。对一个结合代数 $A$，其**导子（derivation）**是满足莱布尼茨法则 $\delta(ab) = \delta(a)b + a\delta(b)$ 的线性映射，全体导子配上交换子构成李代数。这是「李代数从代数本身长出来」的最自然途径。

## 5 直和与理想结构

当 $I, J$ 都是 $L$ 的理想且 $L = I \oplus J$（向量空间直和）时，称 $L$ 是 $I$ 与 $J$ 的**理想直和（direct sum of ideals）**。此时 $[I, J] = 0$：因为 $[i, j] \in I \cap J = 0$。<span class="marginnote">这里「理想直和」比向量空间直和更强：它还要求两块的括号交互为零。后半专题「半单 = 不可分解理想的直和」正是以这个概念为底。</span>

交换性（$\mathbb{F}$ 上的平凡结构 $[x, y] = 0$）加上直和，给了我们最便宜的构造工具：**阿贝尔李代数**是最简单的块，任意线性空间配上零括号即得。

## 6 公式解析：伴随映射与 Jacobi 恒等式的真义

定义**伴随映射（adjoint map）**：对每个 $x \in L$，令

$$\operatorname{ad}x: L \to L, \qquad (\operatorname{ad}x)(y) = [x, y]$$

则 Jacobi 恒等式最深刻的等价表述是：**$\operatorname{ad}$ 是从 $L$ 到导子代数 $\operatorname{Der}(L)$ 的李代数同态**，即

$$\operatorname{ad}([x, y]) = [\operatorname{ad}x, \operatorname{ad}y] = \operatorname{ad}x \circ \operatorname{ad}y - \operatorname{ad}y \circ \operatorname{ad}x$$

三步拆解这条式子：

- **第一步，读左边**：$(\operatorname{ad}[x,y])(z) = [[x,y], z]$，即先用 $[x,y]$ 作用 $z$。
- **第二步，读右边**：$[(\operatorname{ad}x)(\operatorname{ad}y) - (\operatorname{ad}y)(\operatorname{ad}x)](z) = [x, [y, z]] - [y, [x, z]]$，即先 $y$ 后 $x$ 与先 $x$ 后 $y$ 之差。
- **第三步，两者相等**：用反对称把 $-[y,[x,z]]$ 写成 $+[y,[z,x]]$，再整理即得 Jacobi 恒等式。

换句话说，**Jacobi 恒等式正是「内自同构 $\operatorname{ad}x$ 是一个导子」这个事实的展开**。这把一个抽象的代数恒等式翻译成了可操作的计算法则：每个 $x$ 给出的「沿 $x$ 方向的微分」都满足莱布尼茨法则。后文根系理论中，$\operatorname{ad}$ 的幂零元将直接参与构造 $\mathfrak{sl}(2)$ 三元组，那是本专题第 6 篇的核心武器。

## 7 小结

- **李代数** = 向量空间 + 双线性反对称括号 + Jacobi 恒等式；括号**不满足结合律**。
- 结构对象：**子代数、理想、商代数、同态**，与群论/环论的框架平行（第一同构定理成立）。
- 五个基本例子：$\mathfrak{gl}(n,\mathbb{F})$、$\mathfrak{sl}(n,\mathbb{F})$、$\mathfrak{t}(n,\mathbb{F})$、$\mathfrak{n}(n,\mathbb{F})$、$\mathfrak{so}(n,\mathbb{F})$，各自是后续可解、幂零、半单、紧理论的原型。
- **伴随映射 $\operatorname{ad}$** 把 Jacobi 恒等式翻译成「$\operatorname{ad}$ 是导子代数间的同态」，是全书最常用的换算器。
- 理想直和 $\oplus$ 把大李代数拆成互不干扰的块，是半单结构定理的出发点。

在下一节，我们将用括号的嵌套长度定义两个「良性」类——**可解与幂零李代数**，并见识它们如何充当一切结构分解的「噪声层」。
