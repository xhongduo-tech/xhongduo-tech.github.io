---
title: 切空间、切丛与映射的微分
date: 2026-08-07
---

# 切空间、切丛与映射的微分

<div class="epigraph">
<p>自然界没有飞跃。</p>
<footer>—— 戈特弗里德 · 莱布尼茨（Gottfried Wilhelm Leibniz）</footer>
</div>

<div class="article-byline">
<p>第二级 · 微分拓扑 ｜ Guillemin & Pollack《Differential Topology》Ch.1 §1–2 ｜ 2026-08-07</p>
</div>

## 为什么从切空间开始

上一讲立住了「光滑流形」与「光滑映射」，但只回答了一个存在性问题：光滑性在哪里可以谈。微积分的真正力量在于**线性化**——用切线逼近曲线，用切平面逼近曲面，用全微分逼近函数。切空间（tangent space）就是这一思想在流形上的严格版本：**在每一点 $p$ 处，把流形 $M$ 局部地看成 $p$ 处的欧氏线性空间 $T_p M$，一切方向与导数都装进这个线性空间里。** 映射的微分 $df_p$ 则是「光滑映射在 $p$ 处的线性替身」。几乎所有微分拓扑的概念——正则值、横截性、嵌入、向量场——都是「$T_p M$ 与 $df_p$ 的某种性质」，所以这一讲是整个学科的发动机。

在进入严格定义之前，先建立一个心理模型：$T_p M$ 就是「$p$ 点处那个无限小的平面」，$df_p$ 就是「把 $p$ 附近一小块流形拍平后看到的线性变换」。下面所有公式都在为这个画面补上精确的标签。<span class="marginnote">多元微积分里的 Jacobi 矩阵就是 $df_p$ 在坐标下的样子；线性代数（第二级《线性代数》）里的线性映射、矩阵表示、维数公式，从这里开始全部搬进流形的语境。流形上的微分不是新发明，而是「把微积分搬到没有全局坐标的曲面上」。</span>

## 1 切向量与切空间

光滑流形上的切空间有不止一种等价定义，它们各有所长。这里给出两种最常用的视角。

**导子视角（代数定义）**：设 $M$ 光滑，$p \in M$，$C^\infty_p(M)$ 是 $p$ 附近光滑函数的芽。一个 **$p$ 处的切向量**是线性映射

$$v: C^\infty_p(M) \longrightarrow \mathbb{R}$$

满足莱布尼茨律 $v(fg) = f(p)\, v(g) + v(f)\, g(p)$。全体这样的导子构成**切空间（tangent space）** $T_p M$，它自然是一个 $\mathbb{R}$-线性空间，维数等于 $\dim M$。

导子视角的妙处在于它**完全不依赖坐标**：向量就是「对函数求方向导数」的规则。它把「几何的箭头」翻译成「代数的算子」，凡是线性代数里对算子的结论，都对切向量成立。<span class="marginnote">莱布尼茨律正是乘积求导法则 $D(fg) = fDg + gDf$ 的代数化。在原点 $0 \in \mathbb{R}^n$，切向量 $v$ 与偏导算子 $\sum v^i \partial/\partial x^i \big|_0$ 一一对应——这就是「向量 = 方向导数」这句话的精确含义。</span>

**曲线等价类视角（几何定义）**：设 $\gamma, \eta: (-\varepsilon, \varepsilon) \to M$ 是两条过 $p$ 的光滑曲线（$\gamma(0) = \eta(0) = p$）。若在 $p$ 的某坐标卡 $(U, \varphi)$ 下

$$\frac{d}{dt}\big|_{t=0} \varphi \circ \gamma(t) = \frac{d}{dt}\big|_{t=0} \varphi \circ \eta(t),$$

则称 $\gamma$ 与 $\eta$ **在 $p$ 处有相同速度**。切向量定义为曲线在此等价关系下的等价类。

几何定义最直观：**切向量就是「以 $p$ 为起点的无穷小方向」**，两条曲线等价当且仅当它们以相同速度穿过 $p$。验证这个定义不依赖坐标卡，恰好用上上一讲的转换函数——两种定义的等价性本身，就是「坐标无关」精神的一次练习。

**局部坐标下的表示**：取坐标卡 $(U, \varphi)$，坐标函数 $(x^1, \dots, x^n)$，则 $T_p M$ 有一组自然的基

$$\left\{ \frac{\partial}{\partial x^1}\Big|_p, \dots, \frac{\partial}{\partial x^n}\Big|_p \right\},$$

任意切向量 $v$ 唯一写成 $v = \sum_{i=1}^n v^i \frac{\partial}{\partial x^i}\big|_p$，分量 $(v^1, \dots, v^n)$ 称为 $v$ 的**坐标分量**。

## 2 映射的微分：$df_p$ 与链式法则

光滑映射在切空间之间诱导一个线性映射，这是整个微分拓扑的「主力运算」。

**微分（differential / pushforward）**：设 $f: M \to N$ 光滑，$p \in M$。定义

$$df_p: T_p M \longrightarrow T_{f(p)} N, \qquad (df_p(v))(g) = v(g \circ f),$$

其中 $g \in C^\infty_{f(p)}(N)$。用曲线等价类表述更朴素：若切向量 $v$ 由曲线 $\gamma$ 表示，则 $df_p(v)$ 由复合曲线 $f \circ \gamma$ 表示。**微分把「$M$ 上的方向」推进（push forward）成「$N$ 上的方向」。**

在坐标下，$df_p$ 就是 Jacobi 矩阵：设 $f = (f^1, \dots, f^m)$，则

$$df_p\left( \frac{\partial}{\partial x^j} \Big|_p \right) = \sum_{i=1}^m \frac{\partial f^i}{\partial x^j}(p)\, \frac{\partial}{\partial y^i}\Big|_{f(p)}.$$

于是 $df_p$ 关于基的矩阵正是 $\big( \partial f^i / \partial x^j (p) \big)_{m \times n}$。<span class="marginnote">「映射的微分」在文献里有多个名字：differential、derivative、pushforward（推进）、或 $f_*$。它们都是同一个线性映射。$df_p$ 的像就是「$N$ 中沿 $f$ 的切方向」，$df_p$ 的核就是「$M$ 中被 $f$ 压扁的方向」——正则值理论的核心就是在问这两者的维数。</span>

**链式法则（函子性）**：设 $f: M \to N$，$g: N \to P$，则

$$d(g \circ f)_p = dg_{f(p)} \circ df_p.$$

这是多元微积分链式法则的坐标无关形式，也是「微分」作为从光滑映射到线性映射的**函子**（保持复合与恒等）这一性质的精确表述。**最容易出的错**：$df_p$ 的定义域是 $T_p M$，值域是 $T_{f(p)} N$——起点和终点都绑在具体点上，不能混用两个不同点的切空间。<span class="marginnote">函子性是范畴语言（上篇提到的「对象—态射」框架）的具体化：$f \mapsto df$ 把一个光滑映射送成一个线性映射，且保持复合。第二级《范畴论》里的协变函子 $T$（切函子）就是这个映射的抽象装束。</span>

**局部微分同胚**：若 $df_p$ 是线性同构，则 $f$ 在 $p$ 附近是微分同胚（逆映射定理）。这是反函数定理的流形版本，也是「浸入/浸没」定义的分水岭：$df_p$ 单射 ⇔ 浸入，$df_p$ 满射 ⇔ 浸没。

矩阵的秩在这里第一次显出几何意义。**秩（rank）**：$df_p$ 的秩定义为它作为线性映射的像的维数，在坐标下即 Jacobi 矩阵的秩。链式法则给出一个惊人的内蕴结论：**在 $f$ 的任意一点的邻域里，只要 $df$ 的秩处处为常数，则 $f$ 局部上就是一条「直射」**——存在局部坐标使 $f$ 写成 $(x^1, \dots, x^n) \mapsto (x^1, \dots, x^r, 0, \dots, 0)$。这就是**秩定理（rank theorem）**，它是正则值理论与横截性理论背后最底层的「扁平化」机器。初学阶段先记住一句话：秩有多大，$f$ 的像「伸直」后就有多少维。

## 3 切丛：把全部切空间拼起来

单个切空间只是「一点处的线性化」。要让「方向」作为流形上的合法对象（比如讨论向量场、积分曲线），需要把各点的切空间**光滑地粘成一个整体**。

**切丛（tangent bundle）**：集合 $TM = \bigsqcup_{p \in M} T_p M$（各点切空间的不交并），配以投影 $\pi: TM \to M$，$\pi(v) = p$ 若 $v \in T_p M$。$TM$ 带有自然的 $2n$ 维光滑流形结构：对坐标卡 $(U, \varphi)$，定义坐标

$$(p, v) \longmapsto (x^1(p), \dots, x^n(p), v^1, \dots, v^n),$$

其中 $(v^1, \dots, v^n)$ 是 $v$ 在基 $\{\partial/\partial x^i\}$ 下的分量。坐标卡 $(TU, d\varphi)$ 让 $TM$ 成为光滑流形，$\pi$ 成为光滑映射。<span class="marginnote">$TM$ 的「一半坐标是点、一半坐标是方向」这一结构，是<strong>向量丛</strong>（vector bundle）的原型：$M$ 是底空间，每一点上挂着一根 $n$ 维纤维 $T_p M$。下一讲的正则值、再后面的横截性，本质都在问「纤维之间的关系」。T 恤上印的「TM 是个流形」，微分拓扑的学生都能看懂。</span>

**局部平凡化**：在 $U$ 上，$\pi^{-1}(U) \cong U \times \mathbb{R}^n$——切丛局部上是平凡的乘积，全局未必（环面 $T^2$ 的切丛是 $T^2 \times \mathbb{R}^2$，球面 $S^2$ 的切丛不是）。「局部像乘积、全局有缠绕」正是纤维丛的核心体验。

**辨析｜易错点：切丛「局部平凡」≠「全局平凡」。** 一个常见的错觉是「切丛既然每点都是一根 $\mathbb{R}^n$，全局也该是个乘积」。$S^2$ 的切丛给出了最响亮的反例：若 $TS^2 \cong S^2 \times \mathbb{R}^2$，就能在 $S^2$ 上找到处处非零的光滑向量场（把 $\mathbb{R}^2$ 的固定基向量铺到每个点）；但「毛球定理」（hairy ball theorem）说这样的场不存在——风吹不动一个刺猬，$S^2$ 上每个向量场必在某点为零。$TS^2$ 是否平凡的问题，正是 Poincaré–Hopf 指标定理（后面专门有一讲）的引子。

**向量场（vector field）**：光滑截面 $X: M \to TM$，满足 $\pi \circ X = \mathrm{id}_M$，即在每点 $p$ 选一个切向量且光滑地变。向量场是后面「流与管状邻域」「Poincaré–Hopf 指标定理」的主角。这里先记住一个事实：$TM$ 总是可平行化的局部区域上，向量场可以写成 $X = \sum X^i \partial/\partial x^i$，光滑性等价于分量函数 $X^i$ 光滑。

一个值得暂停的观察：**$TM$ 永远比 $M$ 「平」一个档次。** $M$ 可以任意弯曲，但 $TM$ 作为 $2n$ 维流形，其纤维结构自带线性结构——在 $TM$ 上可以做向量加减，在 $M$ 上不行。这正是「丛」的价值：把流形上不能直接做的线性运算，通过升到切丛来间接完成。微分方程、变分、几何测度论里「升到切丛/余切丛再谈运算」是通行手法，这里埋下伏笔。

## 4 公式解析：坐标变换下的切向量分量

切空间是「坐标无关」的，但具体计算必须落到坐标。不同坐标卡下，同一个切向量的分量如何变？这是切空间理论里最核心、也最容易出错的一条公式。

设 $p \in U_\alpha \cap U_\beta$，坐标卡 $\varphi_\alpha$ 与 $\varphi_\beta$，转换函数

$$y^i = \psi^i(x^1, \dots, x^n), \qquad \psi = \varphi_\beta \circ \varphi_\alpha^{-1}.$$

同一向量 $v$ 在两组基下的分量 $v = \sum v^i_\alpha \partial/\partial x^i = \sum v^j_\beta \partial/\partial y^j$ 满足：

$$
v^j_\beta = \sum_{i=1}^n \frac{\partial \psi^j}{\partial x^i}\,\, v^i_\alpha.
$$

拆成三步理解：

- **第一步，看「新 = 旧 × 雅可比」**：新坐标分量 $v^j_\beta$ 等于旧分量 $v^i_\alpha$ 与转换函数偏导的线性组合。矩阵 $J = (\partial \psi^j / \partial x^i)$ 就是「从 $\alpha$ 坐标换到 $\beta$ 坐标」的速度变换矩阵。方向本身不变，只是**表达它用的基换了**，所以分量要按基变换的「逆方向」调整——这正是线性代数里坐标变换公式的翻版。
- **第二步，为什么是「同向」而非「反向」**：基 $\partial/\partial x^i$ 变换时，若坐标 $x$ 变为 $y$，则 $\partial/\partial x^i = \sum_j (\partial \psi^j/\partial x^i) \partial/\partial y^j$（链式法则）。切向量是线性组合，系数随基同向变换，所以分量矩阵也是 $J$。这与「余切向量」（1-形式）的分量按 $J^{-1}$ 变换形成对照——凡是在流形上谈「反变（contravariant）向量」，指的就是这种同向变换。
- **第三步，验证两义性（一致性条件）**：三张坐标卡交叠时，按「$\alpha \to \beta \to \gamma$」与「$\alpha \to \gamma$」两条路径算出的分量必须一致，即 $J^{\beta\gamma} \cdot J^{\alpha\beta} = J^{\alpha\gamma}$。这个乘法规则正好是转换函数复合的链式法则，它保证「切向量」是良定义的全局对象。凡是满足这套规则的量，都叫**张量**——$v^i$ 是一个 $(1,0)$ 型张量，而 $df_p$ 的矩阵分量 $(\partial f^i / \partial x^j)$ 在换坐标时会同时用到 $J$ 与 $J^{-1}$。

**一个数值算例**：设 $M = \mathbb{R}^2$，坐标卡 $\varphi_\alpha$ 为恒等、$\varphi_\beta(x, y) = (2x, 3y)$（把坐标「拉长」）。转换函数 $\psi(u, v) = (2u, 3v)$，$J = \mathrm{diag}(2, 3)$。向量 $v = \partial/\partial x$ 在 $\alpha$ 坐标下的分量是 $(1, 0)$，在 $\beta$ 坐标下变成 $(2, 0)$——**坐标基被压扁，分量就变大**；这个「反变」互反关系正是「$v^j_\beta = \sum_i (\partial \psi^j/\partial x^i) v^i_\alpha$」的具体面孔，也是张量语言里的第一道算术题。

## 5 小结

- **切空间** $T_p M$ 是 $p$ 处全体「无穷小方向」的 $n$ 维线性空间；有导子、曲线等价类、坐标分量三种等价刻画。
- **微分** $df_p: T_p M \to T_{f(p)} N$ 是光滑映射的局部线性化，坐标下即 Jacobi 矩阵；满足**链式法则** $d(g \circ f) = dg \circ df$。
- $df_p$ 的**单射/满射/同构**分别对应**浸入/浸没/局部微分同胚**——这是后面三讲的核心判据。
- **切丛** $TM$ 把各点切空间光滑粘成 $2n$ 维流形，是**向量丛**的原型；向量场是其光滑截面。
- 切向量分量的坐标变换公式 $v^j_\beta = \sum_i (\partial \psi^j/\partial x^i) v^i_\alpha$ 是张量语言的第一课。

在下一节，我们将问一个决定性的大小问题：给定光滑映射 $f: M \to N$，哪些值 $y \in N$ 是「好值」——即原像里每个点 $p \in f^{-1}(y)$ 处的微分 $df_p$ 都满射。这样的 $y$ 叫**正则值**，它的原像自动是子流形——这就是**正则值与原像定理**。$
