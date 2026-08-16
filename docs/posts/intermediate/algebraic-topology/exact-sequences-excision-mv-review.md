---
title: 正合列、切除定理与 Mayer–Vietoris 序列
date: 2026-08-07
---

# 正合列、切除定理与 Mayer–Vietoris 序列

<div class="epigraph">
<p>正合列是同调论的马达：它把碎片串成链条，把链条算成答案。</p>
<footer>—— 代数拓扑学习者的概括（出处佚名）</footer>
</div>

<div class="article-byline">
<p>第二级 · 代数拓扑 ｜ Hatcher 第2.1、2.2章 ｜ 2026-08-07</p>
</div>

## 为什么从复习这三条引擎开始

上一节复习奇异同调时提到：奇异同调「难算」，全靠三条定理救场——**长正合序列、切除定理、Mayer–Vietoris 序列**。本节把这三条引擎从「听说过」升级为「会组装」：它们共享同一个代数骨架（正合列），各自负责一个几何操作（空间↔子空间、挖内部、开覆盖拼接）。<span class="marginnote">复习这三条的意义：几乎一切同调计算，最终都归结为「构造一条正合列、填一个不变量、读出答案」三步。掌握了正合列的「算法」，就掌握了同调计算的全部套路。</span>

这三条引擎不仅用于奇异同调，还适用于一切满足 Eilenberg–Steenrod 公理的同调理论（已复习）——它们是公理的**推导引擎**。这也解释了为什么谱序列（第 12 篇复习）能成为它们的超集：谱序列就是「无穷多层正合列」。

**为什么放在一起复习？** 三条引擎其实是同一台机器在不同几何场景下的三个按钮：配对 LES 处理「空间与子空间」，切除处理「挖掉内部」，MV 处理「开覆盖拼接」。把三台引擎装进同一张正合列语法里，后续无论遇到相对同调、系数变化还是纤维化，都能立刻认出「哦，又是正合列 + 追图」。

## 1 复习：正合列——同调论的语法

**正合列（exact sequence）**：群同态链 $A \xrightarrow{\alpha} B \xrightarrow{\beta} C$ 在 $B$ 处**正合**，若 $\operatorname{im} \alpha = \ker \beta$。整条链处处正合，就称为正合列。**复习时把「正合」当成四字检验法**：核对「像等于核」，像太大漏信息、像太小多信息，二者都不算正合。核心小等式：

$$0 \to A \xrightarrow{\alpha} B \xrightarrow{\beta} C \to 0 \text{ 正合} \iff \alpha \text{ 单射},\ \beta \text{ 满射},\ \operatorname{im}\alpha = \ker\beta$$

即「$B$ 恰好由 $A$ 嵌进、多余部分由 $C$ 读出」——短正合序列（short exact sequence）是一种「加法分解」。<span class="marginnote">正合列是「账本对账」：$\ker\beta$（账上应为零者）恰好等于 $\operatorname{im}\alpha$（来自上一环节者）。链式对账是同调论的语法规则，后面的 Mayer–Vietoris 与长正合序列都是它的成品句。</span>

**空间配对（pair）的短正合序列**：对 $A \subset X$，含入诱导

$$0 \to C_n(A) \xrightarrow{i} C_n(X) \xrightarrow{j} C_n(X)/C_n(A) \to 0$$

各维拼接、用蛇形引理追图，得到**相对同调长正合序列（LES of pair）**：

$$\cdots \to H_n(A) \xrightarrow{i_*} H_n(X) \xrightarrow{j_*} H_n(X, A) \xrightarrow{\partial_*} H_{n-1}(A) \xrightarrow{i_*} \cdots \to H_0(X,A) \to 0$$

$\partial_*$ 叫**连接同态（connecting homomorphism）**——它把「相对循环」送到「其边界」，是正合列的接头处。

**例：$\mathbb{R}^n \setminus \mathbb{R}^k$ 的同调。** 取 $A = \mathbb{R}^n \setminus \mathbb{R}^k$、$X = \mathbb{R}^n$，用配对 LES 逐段读出：中间维度的洞由 $\mathbb{R}^n$ 相对 $A$ 的「包络」转移而来，最终得到 $H_i(\mathbb{R}^n\setminus\mathbb{R}^k)$ 仅在 $i = n-k-1$ 处非零（$\mathbb{Z}$）——「挖掉 $\mathbb{R}^k$ 留下一个 $n-k-1$ 维的洞」，这正是配对 LES + 连接同态的招牌应用。

## 2 复习：切除定理——「内部不算」

**切除定理（excision）**：设 $Z \subset A \subset X$ 且 $\overline Z \subset \operatorname{int} A$，则含入 $(X\setminus Z, A\setminus Z) \hookrightarrow (X, A)$ 诱导同构

$$H_n(X\setminus Z, A\setminus Z) \xrightarrow{\cong} H_n(X, A)$$

直觉：相对同调 $H_n(X,A)$ 只关心「$X$ 相对 $A$ 多出来的部分」的洞；$Z$ 被 $A$ 的**内部**包住，不参与边界，切掉它不影响「相对骨架」。**换个说法**：切除定理说「远离 $A$ 的无关内脏不影响 $X$ 相对 $A$ 的洞」——它把「切掉什么无所谓」变成了精确的同构。<span class="marginnote">「$\overline Z$ 在 $A$ 内部」的闭包条件是技术关键：它保证切除后不触碰「相对骨架」的边界。复习时把这个条件当作例行检查点。</span>

切除定理的两大用途：

- **证明同伦不变性的替代路线**：奇异同调的同伦不变性已有棱柱算子，但切除定理给出独立证明（Hatcher 第 2.1 章用重分 + 切除）。
- **推导 Mayer–Vietoris 序列的原料**：MV 序列的证明核心就是「把 $A$ 与 $B$ 的并切成两部分再切除」——切除是 MV 的发动机。

## 3 复习：Mayer–Vietoris 序列——同调版的 Van Kampen

**Mayer–Vietoris 序列**：设 $X = A \cup B$（$A, B$ 为开集，或满足 Hatcher 的条件），则

$$\cdots \to H_n(A\cap B) \xrightarrow{\Phi} H_n(A) \oplus H_n(B) \xrightarrow{\Psi} H_n(X) \xrightarrow{\Delta} H_{n-1}(A\cap B) \to \cdots$$

其中 $\Phi(z) = (i_* z, -j_* z)$、$\Psi(x, y) = k_* x + l_* y$，$\Delta$ 是连接同态。**直觉：$A\cap B$ 的洞决定「$A$ 与 $B$ 如何拼」，$X$ 的洞由 $A$、$B$ 的洞在交点对齐后得到。**<span class="marginnote">Van Kampen 处理非交换的 $\pi_1$，Mayer–Vietoris 处理交换的同调——两者是「拼接基本群」与「拼接同调」的同构思路在不同代数下的实现。</span>

**用法（三步算法）**：第一步，把 $X$ 切成 $A \cup B$ 使三者同调可算；第二步，写出 MV 序列；第三步，用已知项与正合性推出未知项。经典产物：

- **楔和**：$H_n(A\vee B) \cong H_n(A) \oplus H_n(B)$（$n \ge 1$），交点是 0-维修补项。
- **球面**：$H_k(S^n) = \mathbb{Z}$（$k = 0, n$）、否则 0——用 $S^n = D^n_+ \cup D^n_-$ 两块半球拼接。
- **去点空间**：$\mathbb{R}^n\setminus\{0\} \simeq S^{n-1}$，同调立即读出。

**例：把「球面的同调」用 MV 完整走一遍（$n \ge 2$）。** 取 $A = S^n \setminus \{南极点\}$、$B = S^n \setminus \{北极点\}$，则 $A \cong B \cong \mathbb{R}^n$（可缩，$H_k = 0$ 对 $k\ge1$），$A\cap B \cong S^{n-1} \times \mathbb{R}$（同伦等价于 $S^{n-1}$）。MV 序列在中间维度读出：

$$0 \to H_n(S^n) \to H_{n-1}(S^{n-1}) \to 0 \quad\Rightarrow\quad H_n(S^n) \cong H_{n-1}(S^{n-1})$$

逐维接力，加上 $H_0(S^n) = \mathbb{Z}$ 与 $H_1(S^1) = \mathbb{Z}$，归纳出 $H_n(S^n) = \mathbb{Z}$、中间维为零。**这套「同维数逐级传递」的归纳，正是第 2 篇复习「球面的同调」中归纳法的 MV 版本。**

## 4 公式解析：连接同态与 MV 的追图

以配对长正合序列的连接同态为例拆开追图法：

$$\cdots \to H_n(X,A) \xrightarrow{\partial_*} H_{n-1}(A) \to \cdots$$

- **第一步，取代表**：$\alpha \in H_n(X,A)$，选相对循环 $c$（$\partial c \in C_{n-1}(A)$）。
- **第二步，取边界**：$\partial c$ 是 $A$ 中的 $(n-1)$-循环（因为 $\partial\partial c = 0$），于是给出类 $[\partial c] \in H_{n-1}(A)$。
- **第三步，验证良定义**：换个相对代表 $c' = c + b$（$b$ 为边界）时，$[\partial c'] = [\partial c] + [\partial b] = [\partial c]$，因为 $\partial b$ 在 $A$ 中可缩——连接同态良定义。**「先取相对循环、再取它的边界」就是连接同态的全部操作。**

Mayer–Vietoris 的 $\Delta$ 是同一操作在「拼接」语境下的翻版：把 $X$ 中的循环分解成 $A$ 部分与 $B$ 部分，差落在 $A\cap B$ 中，取这个差即得 $\Delta$。

一句话：**正合列的三件套——核像链、连接同态、追图——是这套计算的统一语法；切除与 MV 只是把几何切法翻译成语法的两个范例。**补一条观察：**正合列每「断」一次，就对应一次「信息传递」**——$\partial_*$ 的存在意味着高维的相对信息降维进入低维，这正是「洞随维度迁移」的代数显形。

## 5 核心对比：三引擎的分工

| 引擎 | 输入 | 输出 | 几何操作 | 典型用途 |
| --- | --- | --- | --- | --- |
| 配对长正合序列 | $A \subset X$ | $H_*(A) \to H_*(X) \to H_*(X,A)$ | 模掉子空间 | 相对同调、$\mathbb{R}^n\setminus\mathbb{R}^k$ |
| 切除定理 | $Z \subset \operatorname{int}A \subset X$ | 同构 | 挖内部 | 证明工具、MV 的原料 |
| Mayer–Vietoris | $X = A \cup B$ | $H_*(A\cap B) \to H_*(A)\oplus H_*(B) \to H_*(X)$ | 开覆盖拼接 | 球面、楔和、去点 |

**辨析｜易错点：**

- **正合性只是「局部」**：正合列给的是「$\operatorname{im} = \ker$」，不是「群的直和分解」。短正合序列一般不分裂；只有带分裂条件（右逆/左逆存在）时才退化为直和——复习时别把「正合」与「可分解」混为一谈。
- **连接同态的方向**：$H_n(X,A) \to H_{n-1}(A)$ **降一个维度**，这是同调（相对）的固有记号；上同调的连接同态（第 9 篇复习）方向相反、升一个维度。
- **MV 的覆盖条件**：$A, B$ 通常要求开集（或 Hatcher 的更弱条件），否则边界情形可能漏项。切法越「软」越安全。
- **正合列的尾端**：配对 LES 到 $H_0(X,A)$ 为止；$H_0$ 处理「路径分量」的计数需要额外小心（约化同调可避免零维干扰，见第 2 篇复习的约化同调）。
- **MV 的维度陷阱**：MV 中 $\Psi$ 有「正负号」（$\Phi(z) = (i_* z, -j_* z)$），负号不是笔误而是为了追图对账；漏掉负号会让后续同态计算整体出错。写 MV 序列时把符号当作序列的一部分抄下来。

## 6 小结

- **正合列**：$\operatorname{im}\alpha = \ker\beta$ 的链式对账；短正合序列是「加法分解」的语法。
- **配对长正合序列**：$H_n(A) \to H_n(X) \to H_n(X,A) \to H_{n-1}(A)$，连接同态 $\partial_*$ 是接头。
- **切除定理**：$\overline Z \subset \operatorname{int} A$ 时 $H_n(X\setminus Z, A\setminus Z) \cong H_n(X,A)$——内部可挖。
- **Mayer–Vietoris**：$H_n(A\cap B) \to H_n(A)\oplus H_n(B) \to H_n(X)$——开覆盖拼接，同调版 Van Kampen。
- **三步算法**：切空间 → 写序列 → 追图求未知；几乎一切同调计算都长这样。
- 复习口诀：**模掉子空间用配对 LES、挖内部用切除、拼并集用 MV；追图找 $\partial_*$**。
- 与课程的连接：这三引擎将在第 9 篇复习（上同调）中原样复用（上同调也有配对 LES 与 MV），在第三级《同调代数》中成为「导函子」的几何温床，在《代数拓扑进阶》中被谱序列（无穷多正合列）推广。
- **一句话概括本节**：同调计算 = 构造正合列 + 追图；配对 LES 管「模掉」、切除管「挖内部」、MV 管「拼并集」，三按钮共用一张语法表。
- **与之前复习的对账**：本节用正合列重新「算」出了球面、楔和、去点空间的同调，与第 5 篇单纯同调的参考答案逐一吻合——这再次印证「用什么算法，答案都一样」的同调哲学。

在下一节，我们复习最「工程化」的同调——**胞腔同调**：在 CW 复形上把同调计算压缩成一张边界矩阵，并顺手拿到 Brouwer 不动点与映射度。
