---
title: 帽积与交叉积
date: 2026-08-07
---

# 帽积与交叉积

<div class="epigraph">
<p>同调把空间切成循环，上同调在循环上赋值；帽子把两者接在一起，叉子把两个空间编在一起。</p>
<footer>—— 佚名（代数拓扑课堂流传）</footer>
</div>

<div class="article-byline">
<p>第二级 · 代数拓扑 ｜ Hatcher 第3.2、3.3章 ｜ 2026-08-07</p>
</div>

## 为什么从帽积与交叉积开始

杯积（第 13 篇）是「上同调 × 上同调 → 上同调」的乘法，它让上同调成环。但上同调真正接入几何，还需要一座「上同调 × 同调 →
同调」的桥——**帽积（cap product）**：

$$\frown \colon H^k(X;R) \times H_n(X;R) \longrightarrow H_{n-k}(X;R)$$

帽积把「$k$-维洞上的函数」作用到「$n$-维洞」上，切出一个 $(n-k)$-维洞。上一节 Poincaré 对偶的主角正是它：$\alpha
\mapsto \alpha \frown [M]$。没有帽积，对偶只能是一句漂亮的空话。

**交叉积（cross product）**则是杯积的「积空间」版本：把 $X$ 上的类与 $Y$ 上的类编成 $X \times Y$
上的类。对同调它出现在 Künneth 公式里（第 10 篇）；对上同调它是杯积的源头（杯积 =
拉回到对角线的交叉积）。帽积与交叉积、杯积三者之间存在一串**相容性公式**，把它们织成一张严密的代数网——这张网正是 Poincaré
对偶、以及对偶下「杯积变成相交」的完整证明骨架。<span class="marginnote">一个统一视角：三种乘积都在回答「怎么把两个东西结合成一个」。杯积 $\smile$ 在同一个空间上结合两个函数；交叉积
$\times$ 在两个空间上结合两个类；帽积 $\frown$
把一个函数与一个循环结合成一个循环。它们共享同一条「切面求值」的机器，只是切的位置与方向不同。</span>


帽积和交叉积是「乘法家族」里最后两位成员，它们的意义在于**连接**：帽积连接上同调与同调（$H^k \times H_n \to
H_{n-k}$），交叉积连接两个空间的（上）同调（$H^k(X) \times H^l(Y) \to H^{k+l}(X\times
Y)$）。读这一节时请把四个运算（$\smile, \times, \frown$）放进同一张表：谁和谁结合、谁和谁对偶、维数怎么走。**帽积是
Poincaré 对偶的构造工具，交叉积是 Künneth 公式的主角**——二者都已在前文露面，本节是把它们的细节补齐。

## 1 帽积定义：函数吃掉前半段，留下后半段

在链层，设 $\varphi \in C^k(X;R)$、$\sigma \colon \Delta^n \to X$ 是奇异 $n$-单形（$n \ge
k$），记 $\sigma$ 的前 $k+1$ 个顶点张成的面为 $\sigma|_{[v_0,\dots,v_k]}$，后 $n-k+1$
个顶点张成的面为 $\sigma|_{[v_k,\dots,v_n]}$。定义：

$$\varphi \frown \sigma := \varphi\big(\sigma|_{[v_0,\dots,v_k]}\big) \cdot \sigma|_{[v_k,\dots,v_n]}$$

即**用 $\varphi$ 在前半段求值（得到一个 $R$ 中数），把后半段作为系数保留**——函数「吃掉」了链的前半段，留下后半段作为低维链。<span class="marginnote">几何直觉：$\alpha \frown \sigma$ 是「$\sigma$ 中与 $\alpha$
对偶的那部分」。在微分流形上，若 $\alpha$ 是闭形式、代表链是 $S$，则 $\alpha \frown$ 相当于「与 $S$ 相交」，交出的
$(n-k)$-维链就是结果——帽积即「相交算子」。</span>

**关键验证**：$\partial(\varphi \frown \sigma) = \delta\varphi \frown \sigma +
(-1)^k\, \varphi \frown \partial\sigma$，由此帽积在同调层良定义：

$$H^k(X;R) \times H_n(X;R) \xrightarrow{\ \frown\ } H_{n-k}(X;R)$$

**辨析｜易错点：** 帽积的维数关系是「上标减下标得下标」：$n - k$。它与杯积的「下标加下标得下标」方向相反。另外帽积里 $\varphi$
求值的面（$[v_0,\dots,v_k]$）与保留下来的面（$[v_k,\dots,v_n]$）**共享顶点
$v_k$**——这与杯积共享顶点的机制一模一样，都是「两片拼一个单形」的标准切法。

## 2 交叉积：把两个空间编在一起

对同调，交叉积已出现在 Künneth 公式（第 10 篇）。对**上同调**，同样有交叉积：

$$\times \colon H^k(X;R) \times H^l(Y;R) \longrightarrow H^{k+l}(X \times Y;R)$$

链层定义：对奇异单形 $\sigma \colon \Delta^k \to X$、$\tau \colon \Delta^l \to Y$，以及上链
$\varphi, \psi$，在 $\Delta^k \times \Delta^l$ 的某个三角剖分上定义 $(\varphi \times
\psi)(\sigma \times \tau) := \varphi(\sigma) \cdot \psi(\tau)$，再线性扩张（严格版需用
Eilenberg–Zilber 洗牌映射处理 $\Delta^k \times \Delta^l$ 的剖分）。

**杯积 = 交叉积 + 对角映射**。设 $\Delta \colon X \to X \times X$，$x \mapsto (x,x)$
是对角映射。则对 $\alpha, \beta \in H^*(X)$：

$$\alpha \smile \beta = \Delta^*(\alpha \times \beta)$$

**这是理解杯积的最优雅方式**：两个函数在「同一个点」上的相乘，就是先交叉积到 $X \times X$，再拉回到对角线 $X$。<span class="marginnote">对角映射 $X \to X \times X$
是同调论里「乘法之源」：在代数拓扑中，许多乘法结构（杯积、Pontryagin
积、量子杯积）都是「拉回对角」这一操作的变体。对角映射在微分流形上与「交点」对应——杯积 = 相交的对偶，正是这条线的终点。</span>

## 3 相容性公式：一张代数网

四个操作（$\smile$、$\times$、$\frown$、求值
$\langle\cdot,\cdot\rangle$）之间的相容性公式（标准约定下）：

**求值–交叉**：$\langle \alpha \times \beta,\ x \times y \rangle = \langle \alpha, x\rangle\, \langle \beta, y\rangle$；
**交叉–帽积**：$(\alpha \times \beta) \frown (x \times y) = (-1)^{|\beta|\,|x|}\, (\alpha \frown x) \times (\beta \frown y)$；
- **帽积–杯积（结合）**：$(\alpha \smile \beta) \frown \gamma = \alpha \frown (\beta \frown \gamma)$。

**最后一条尤其重要**：它说「先乘后戴」等于「先戴再戴」。在 Poincaré 对偶下（$\alpha \mapsto \alpha \frown
[M]$），这条公式把「杯积对偶于帽积」翻译成：**对偶把杯积变成「先相交再相交」**——即杯积对应对偶类的交集。这正是上一节「杯积 ↔
相交」论断的代数骨架。<span class="marginnote">这些公式里的符号 $(-1)^{|\beta||x|}$
依旧是定向账本：两个「交换」的乘积，奇维数交互时变号。计算时最好整套沿用同一约定（Hatcher 或
Spanier），不同教材的符号惯例可以差一个全局符号。</span>

## 4 与 Poincaré 对偶的关系

回到上一节：对闭可定向 $n$-流形 $M$，帽积映射 $\alpha \mapsto \alpha \frown [M]$ 是同构 $H^k(M)
\cong H_{n-k}(M)$。现在补充它的三个精细层面。

**第一，交叉积 ↔ 相交（Künneth 视角）。** 若 $M, N$ 都是闭流形，Poincaré 对偶 + Künneth
公式给出：$H^{k}(M) \otimes H^{l}(N)$ 的类经交叉积后，对偶于 $H_{n-k}(M) \otimes H_{m-l}(N)$
的类经交叉积——**「乘积流形的对偶 = 对偶的乘积」**。这保证了「相交数」在乘积流形上的可乘性。

**第二，帽积在维数上的「穿行」。** 帽积是唯一同时「吃同调又吃上同调」的运算，正是它让对偶成为「跨维反射」。$H^k$ 与 $H_{n-k}$
之间没有同调自身的映射，也没有上同调自身的映射，**只有帽积这座桥**——这也是对偶定理必须等帽积登场才能证明的原因。

**第三，模 2 与不可定向。** 对 $\mathbb{RP}^{2m}$，$\mathbb{Z}$ 系数下 $[M]$ 不存在，整系数对偶失败；但
$\mathbb{Z}_2$ 系数下基本类总存在，帽积给出 $\mathbb{Z}_2$
对偶。**「基本类是否存在」=「在哪种系数下可定向」**，这是帽积构造对「定向」条件的精确承诺。<span class="marginnote">Poincaré 对偶定理与 de Rham 定理的合流：在光滑流形上，$\mathbb{R}$ 系数上同调 =
闭微分形式模恰当形式（de
Rham），帽积对应「把形式与链相交」。于是「相交」在代数与分析两条路上到达同一个数——这就是为什么「横截相交数」在几何中既是拓扑的也是分析的。</span>

## 5 公式解析：帽积与交叉积的「切–留」机器

$$\varphi \frown \sigma = \varphi\big(\sigma|_{[v_0,\dots,v_k]}\big)\, \sigma|_{[v_k,\dots,v_n]}, \qquad (\varphi \times \psi)(\sigma \times \tau) = \varphi(\sigma)\,\psi(\tau)$$

- **第一步，切**：把单形 $\sigma$ 沿顶点 $v_k$ 切成「前 $k$-面」与「后 $(n-k)$-面」；把 $\Delta^k \times \Delta^l$ 三角剖分后，$\sigma \times \tau$ 拆成 $\sigma$ 与 $\tau$ 的独立组合。
- **第二步，求值**：帽积在**前**面求 $\varphi$ 的值（得数），交叉积在**各自**的因子单形上分别求 $\varphi, \psi$ 的值（得两个数再乘）。
- **第三步，保留**：帽积把后面的 $(n-k)$-面（带系数）留下，维数 $n \to n-k$；交叉积把「两个单形的乘积」整个留下，维数 $k + l$。**「切–留」的方向决定了运算的维数律**：$\frown$ 降维、$\times$ 升维、$\smile$ 升维。

**辨析｜易错点：** 帽积里 $\varphi$ 的求值面是**前** $k+1$ 个顶点（含 $v_0$），保留面是**后** $n-k+1$ 个（含
$v_n$），二者共享 $v_k$。若把「前」「后」搞反，符号与维数都会错——建议用「$\frown$ 箭头朝后（保留后面）」记忆。


**例：$S^2$ 上的帽积。** 取 $M = S^2$，$H^2(S^2) = \mathbb{Z}\langle \varphi
\rangle$，$H_2(S^2) = \mathbb{Z}\langle [S^2] \rangle$，$\langle \varphi, [S^2]
\rangle = 1$。帽积 $\varphi \frown [S^2]$ 是「在基本类上求 $\varphi$ 的值、留下剩余部分」：因为
$\varphi$ 是 2-类而 $[S^2]$ 是 2-链，剩余部分是 0-链，$\varphi \frown [S^2] = 1 \in H_0 =
\mathbb{Z}$。**Poincaré 对偶说 $\varphi \mapsto \varphi \frown [M]$ 是同构**——这里 $H^2
\to H_0$ 恰好是「求值」：给基本类赋 1 的函数对应「单点」（$H_0$ 的生成元）。低维例子把对偶的机制摊开看，不神秘。

**符号约定的实用建议**：交叉积与帽积公式里的 $(-1)^{|\beta||x|}$ 等符号，不同教材约定可差一个全局符号。**学习时选定一套（如
Hatcher）并始终如一**，比「各教材各抄一遍」可靠得多；检验公式是否正确，用 $S^1 \times S^1$ 或 $S^2$
这种简单空间算一遍即可。

**帽积与对偶的完整链路**：$\alpha \smile \beta$ 的对偶 = $\alpha \frown (\beta \frown
[M])$（由结合律），即「先相交再相交」。所以**杯积的几何是「两次相交的复合」**，而帽积正是那台「相交机器」。Poincaré 对偶篇里「杯积 ↔
相交」的断言，至此有了逐条公式的支撑。

## 6 小结

- **帽积**：$\frown \colon H^k(X) \times H_n(X) \to H_{n-k}(X)$，「函数吃掉前半、留下后半」；$\partial$ 层满足莱布尼茨法则，同调层良定义。
- **交叉积**：$\times \colon H^k(X) \times H^l(Y) \to H^{k+l}(X\times Y)$；$\alpha \smile \beta = \Delta^*(\alpha \times \beta)$。
- **相容性**：求值–交叉、交叉–帽积、帽积–杯积三条公式织成网络；$(a \smile b) \frown c = a \frown (b \frown c)$ 支撑「杯积对偶于相交」。
- **与 Poincaré 对偶**：$\alpha \mapsto \alpha \frown [M]$