---
title: 约化同调与相对同调
date: 2026-08-07
---

# 约化同调与相对同调

<div class="epigraph">
<p>好的记号让难题变易，坏的记号让易题变难。</p>
<footer>—— 让 · 迪厄多内（Jean Dieudonné）</footer>
</div>

<div class="article-byline">
<p>第二级 · 代数拓扑 ｜ Hatcher 第2.1章 ｜ 2026-08-07</p>
</div>

## 为什么从变形开始

上一篇定义了奇异同调群 $H_n(X)$，它是完美的拓扑不变量。但直接用起来，有两个地方不够顺手。

**其一，$H_0$ 碍事。** $H_0(X)$ 是道路连通分量的自由阿贝尔群，连通空间总有 $H_0 \cong
\mathbb{Z}$。可很多公式（比如计算球面同调的归纳）会因为这一个多余的 $\mathbb{Z}$ 而处处要分情况讨论。**约化同调**
$\widetilde{H}_n$ 把它删掉：$\widetilde{H}_0 = 0$ 对一切连通的 $X$，而 $\widetilde{H}_n =
H_n$（$n \ge 1$）。多一个维数 -1 的记号，少一整类例外。

**其二，$H_n(X)$ 看不见「模掉子空间」。** 很多几何问题带着子结构：球体 $D^n$ 带着边界球面
$S^{n-1}$，带柄的曲面带着柄圈。我们常常想知道「$X$ 相对 $A$ 有什么新东西」——**相对同调** $H_n(X, A)$ 正是为此而生：它把
$A$ 的一切同调直接归零，只保留「$X$ 里不被 $A$ 看到的」信息。更重要的是，$H_n(A) \to H_n(X) \to H_n(X,A)$
之间有一条**长正合序列**，它是同调论计算的发动机，本专题后面几乎所有定理（切除、Mayer–Vietoris、胞腔同调）都从它出发。


为什么这两个「变形」值得单独成篇？因为它们各自解决一类真实问题，并且是后续一切大定理的零件。约化同调在计算球面同调时把归纳的边界条件统一成一句话——不管从
$S^0$ 出发还是从 $S^n$ 出发，公式都是「一个尖峰」。相对同调则是切除定理（下一篇）与 Mayer–Vietoris 序列（第 6
篇）的定义性原料：没有
$H_n(X,A)$，就没有「模掉子空间」的语言，切除与正合序列就无从谈起。**先在这里把两个变形和它们的长正合序列练熟，后面的每一次计算都会用到。**

## 1 约化同调：给维数 -1 一个位置

约化同调的构造极其优雅：给原链复形**追加一项**。

定义**增广映射（augmentation）** $\varepsilon \colon C_0(X) \to \mathbb{Z}$：把每个 0-单形
$\sigma \colon \Delta^0 \to X$ 送到 $1$，再线性扩张；对 $n \ge 1$ 令 $\varepsilon
\partial_1 = 0$（验证：$\varepsilon \partial_1(\sigma) =
\varepsilon(\sigma|_{\widehat{v_0}}) - \varepsilon(\sigma|_{\widehat{v_1}}) =
1 - 1 = 0$）。于是得到**增广链复形（augmented chain complex）**：

$$\cdots \to C_1(X) \xrightarrow{\partial_1} C_0(X) \xrightarrow{\ \varepsilon\ } \mathbb{Z} \to 0$$

**约化同调群（reduced homology group）** $\widetilde{H}_n(X)$ 就是取这个复形在 $n$ 处的同调。立即得到：

$$\widetilde{H}_n(X) = H_n(X) \quad (n \ge 1), \qquad H_0(X) \cong \widetilde{H}_0(X) \oplus \mathbb{Z}$$

第二个式子是因为 $\ker \varepsilon / \operatorname{im} \partial_1 = \widetilde{H}_0$，而
$H_0 = \ker \partial_0 / \operatorname{im} \partial_1$ 且 $\partial_0 = 0$，于是
$H_0 \cong \widetilde{H}_0 \oplus \mathbb{Z}$（分裂，因为 $\mathbb{Z}$ 自由）。<span class="marginnote">直觉：约化同调把「点」的贡献剥离。单点空间 $*$ 的一切 $\widetilde{H}_n(*) = 0$，而
$H_0(*) = \mathbb{Z}$——「比点多出来的形状」才是约化同调关心的。后面球面同调 $H_n(S^k)$
用约化版本写出来漂亮得多：$\widetilde{H}_i(S^k) = \mathbb{Z}$ 当且仅当 $i = k$。</span>

**辨析｜易错点：** 不要写成「$\widetilde{H}_0 = H_0$ 去掉一个生成元」。严格说 $\widetilde{H}_0$
是增广复形的同调，而不是「从 $H_0$ 里删除某个元」；$H_0 \cong \widetilde{H}_0 \oplus \mathbb{Z}$
是分裂直和同构，但这个同构需要选一个分量的基——同构本身是典范的，选基只在具体描述时出现。

## 2 相对同调：模掉子空间

设 $A \subseteq X$。$C_n(A)$ 通过包含映射成为 $C_n(X)$ 的子群，且 $\partial_n(C_n(A))
\subseteq C_{n-1}(A)$，于是可以取商。

**相对链复形**：$C_n(X, A) := C_n(X) / C_n(A)$，边界算子由 $\partial_n$
诱导（先验证良定义）。**相对同调群（relative homology group）**：

$$H_n(X, A) := H_n\big(C_\bullet(X, A)\big) = \frac{\ker \partial_n \text{ 在 } C_n(X,A) \text{ 中}}{\operatorname{im} \partial_{n+1} \text{ 在 } C_n(X,A) \text{ 中}}$$

**几何意义**：$H_n(X, A)$ 中的循环是「边界落在 $A$ 里的 $n$-链」——它可能在 $A$ 里开口，但其边缘被 $A$
「吃掉」；两个这样的链若差一个**完全落在 $X$ 里**的边界（不再允许差在 $A$ 里的边界）就视为相等。所以相对同调度量的是「$X$ 相对 $A$
额外长出的洞」。<span class="marginnote">例子：$H_n(D^n, S^{n-1}) = \mathbb{Z}$（$n$
处），因为圆盘比它的边界球面多出「实心部分」，这一维由一个「填满整个圆盘」的 $n$-链代表——它没有任何面跑出 $A$ 之外，但自身不是 $X$
里的边界（$D^n$ 没有 $(n+1)$-单形）。这就是后面切除定理与球面同调的伏笔。</span>

特例：$H_n(X, \varnothing) = H_n(X)$；$H_n(X, X) = 0$。这提示我们，**相对同调可以看作「$X$ 到 $A$
的商空间的同调」的替身**——当 $X/A$ 有良好的 CW 结构时，$H_n(X,A) \cong \widetilde{H}_n(X/A)$（CW
对的切除性质），但相对同调的定义完全不依赖 $X/A$ 的良结构，总能定义。

## 3 长正合序列：同调论的发动机

三个对象 $A$、$X$、$(X,A)$ 的同调由一条正合序列串起来。关键的额外环节是一个**连接同态（connecting homomorphism）**
$\partial_* \colon H_n(X,A) \to H_{n-1}(A)$：它把相对循环的边界（在 $A$ 里）「追」回成 $A$ 的循环。

**定理（长正合序列 of a pair）：** 对子空间 $A \subseteq X$，有下列正合序列：

$$\cdots \to H_n(A) \xrightarrow{\;i_*\;} H_n(X) \xrightarrow{\;j_*\;} H_n(X, A) \xrightarrow{\;\partial_*\;} H_{n-1}(A) \xrightarrow{\;i_*\;} \cdots \xrightarrow{\;} H_0(X,A) \to 0$$

其中 $i \colon A \hookrightarrow X$ 是包含，$j \colon X \to (X, A)$ 是「忘掉
$A$」的商映射。正合性在每个位置的意思是「像 = 核」。

**辨析｜易错点：** 序列的右端止于 $H_0(X, A)$，不是无穷延伸——因为 $H_{-1}(A) = 0$。初学者常把左端的 $\cdots$
与右端的 $\to 0$ 搞混；记住「左无穷、右止于 0」即可。另外 $\partial_*$ 把维数**降一**：$H_n(X,A) \to
H_{n-1}(A)$，这是同调（而非上同调）的特征，上同调版的连接同态会升维，那是第 4 篇的内容。

## 4 公式解析：长正合序列如何用

一条正合序列本身不能直接「算」，但它是**推导工具**。拆成三步看它怎么用：

$$\cdots \to H_n(A) \xrightarrow{i_*} H_n(X) \xrightarrow{j_*} H_n(X,A) \xrightarrow{\partial_*} H_{n-1}(A) \to \cdots$$

- **第一步，抓住 $j_*$ 的像**：由正合性，$\operatorname{im} j_* = \ker \partial_*$。想知道「哪些 $H_n(X,A)$ 里的类来自 $X$ 本身」，就看哪些在 $\partial_*$ 下归零。
- **第二步，用 $i_*$ 定位 $j_*$ 的核**：$\ker j_* = \operatorname{im} i_*$。想知道「$X$ 的哪些同调类在相对化后消失」，就看哪些来自 $A$ 的子群像。
- **第三步，追图（diagram chasing）**：正合序列的价值在于——只要知道三个位置中任意两个，就能推出第三个。典型场景：已知 $H_n(A)$ 与 $H_n(X)$，利用 $j_*$ 与 $\partial_*$ 的正合性推出 $H_n(X,A)$。这是所有同调计算的骨架，切除定理、Mayer–Vietoris 序列本质都是「把 $H_n(X,A)$ 换成更易算的同调群」。

**一个立即的应用**：取 $A = \{x_0\}$ 为基点。$H_n(\{x_0\}) = 0$（$n \ge 1$），长正合序列在 $n \ge 1$
处给出 $H_n(X) \cong H_n(X, x_0)$；而在 $n = 1$ 处，$\mathbb{Z} \to H_1(X) \to
H_1(X,x_0) \to \mathbb{Z} \to H_0(X) \to 0$ 中间的链完整地交代了基点如何影响低维同调。<span class="marginnote">这条「$(X, x_0)$ 对」的序列，与第 1
篇基本群里的「基点选取」问题遥相呼应：同调群对基点不敏感（$H_n(X) \cong H_n(X,
x_0)$），这正是同调优于基本群的又一证据。</span>


**例：$H_0$ 与增广到底在干什么。** 取 $X$ 为两个点 $\{a, b\}$。$C_0(X)$ 以 $\sigma_a, \sigma_b$
为基，$\partial_1 = 0$，故 $H_0 = \mathbb{Z}^2$。增广 $\varepsilon(\sigma_a) =
\varepsilon(\sigma_b) = 1$，$\ker \varepsilon = \{n\sigma_a + m\sigma_b \mid n
+ m = 0\}$ 是秩 1 自由群，于是 $\widetilde{H}_0 = \mathbb{Z}$。而分解 $H_0 \cong
\widetilde{H}_0 \oplus \mathbb{Z}$ 里那个多出的 $\mathbb{Z}$，由「总系数」$\sigma_a +
\sigma_b$
代表——它就是「两个点连成一条道路」这一连通性的代数影子。**增广把「总系数」这个冗余精确地分离出去**，让约化同调只保留「比点还多的形状」。

**例：$(D^n, S^{n-1})$ 的相对同调。** 用对序列：$\cdots \to H_i(D^n) \to H_i(D^n, S^{n-1})
\xrightarrow{\partial_\*} H_{i-1}(S^{n-1}) \to H_{i-1}(D^n) \to \cdots$。$D^n$
可缩，$H_i(D^n) = 0$（$i \ge 1$），于是中间夹出 $H_i(D^n, S^{n-1}) \cong
\widetilde{H}_{i-1}(S^{n-1})$。因此 $H_n(D^n, S^{n-1}) = \mathbb{Z}$，其余为
0。几何翻译：相对类由「整个圆盘」代表——它的边界 $S^{n-1}$ 落在子空间里（被吃掉），而圆盘自身不是任何
$(n+1)$-链的边界。**这个例子是球面同调、切除定理、胞腔同调三处反复出现的计算原型，值得完整走一遍。**

**为什么长正合序列是「发动机」**：正合序列本身不含新信息——它只是把已知的三块（$A$、$X$、$X/A$
的同调）用关系锁在一起。它的力量在于**追图**：只要知道任意两个位置，第三个位置常常被唯一确定。这种「知二求三」的推理在同调论里无处不在——下一篇的切除、第
6 篇的 Mayer–Vietoris、第 5 篇的胞腔同调，本质上都是把同一种正合性逻辑用在不同的三件套上。

## 5 小结

- **约化同调** $\widetilde{H}_n$：增广链复形 $\cdots \to C_0(X) \xrightarrow{\varepsilon} \mathbb{Z} \to 0$ 的同调；$\widetilde{H}_n = H_n$（$n \ge 1$），$H_0 \cong \widetilde{H}_0 \oplus \mathbb{Z}$。
- **相对同调** $H_n(X,A)$：商复形 $C_n(X)/C_n(A)$ 的同调，度量「$X$ 相对 $A$ 的新洞」；$H_n(X,\varnothing) = H_n(X)$。
- **长正合序列 of a pair**：$\cdots \to H_n(A) \xrightarrow{i_*} H_n(X) \xrightarrow{j_*} H_n(X,A) \xrightarrow{\partial_*} H_{n-1}(A) \to \cdots$，连接同态降一维。
- **追图**：正合序列是计算工具，「知二推三」。

在下一节，我们将把长正合序列与商结构结合，证明同调论的**切除定理**——它说明 $H_n(X,A)$ 可以无视 $A$ 内部的细节，只由「$X$ 挖掉
$A$