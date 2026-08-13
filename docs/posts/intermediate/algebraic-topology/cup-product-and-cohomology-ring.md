---
title: 杯积与上同调环结构
date: 2026-08-07
---

# 杯积与上同调环结构

<div class="epigraph">
<p>上同调之所以比同调强大，不是因为它们知道得更多，而是因为它们能相乘。</p>
<footer>—— 拉乌尔 · 博特（Raoul Bott）</footer>
</div>

<div class="article-byline">
<p>第二级 · 代数拓扑 ｜ Hatcher 第3.2章 ｜ 2026-08-07</p>
</div>

## 为什么从杯积开始

上同调群 $H^n(X)$
的元素是「同调类上的函数」。函数最自然的操作是什么？**相乘**。两个函数逐点相乘得到新函数——把这件事搬到上同调上，就是**杯积（cup
product）**：

$$\smile \colon H^k(X;R) \times H^l(X;R) \longrightarrow H^{k+l}(X;R), \qquad (\alpha, \beta) \longmapsto \alpha \smile \beta$$

杯积给上同调装上乘法，把各维上同调群拼成一个**分级环**：**上同调环（cohomology ring）** $H^*(X;R) =
\bigoplus_n
H^n(X;R)$。这是整个代数拓扑里最重要、也最漂亮的一件工具——**同调群本身没有乘法**，两个「洞」无法相乘；但两个「洞上的函数」可以相乘。这就是上一节预告的「上同调独有」的结构。

为什么乘法如此重要？因为**信息量暴增**。同一个空间，$H^*$
里的乘积关系编码了「洞之间的相互作用」：两个洞能不能「乘」出第三个，乘出来等于几，这些在同调群里完全看不见。最经典的例子：$S^2 \times S^2$
与 $S^2 \vee S^2 \vee S^4$ 具有**同构的同调群**，却由杯积结构区分——前者 $[\text{点}]$
类的平方非零，后者为零。<span class="marginnote">「同调群相同但上同调环不同」是代数拓扑教材最常引用的警示案例：同调群只是「每个维度有几个洞」的清单，上同调环是「洞之间怎么相互生成」的族谱——后者严格更强。这也解释了为什么上同调（而非同调）才是现代代数拓扑的主角。</span>


杯积是「上同调 =
函数」这一身份的第一次变现。两个洞上的函数相乘，得到更高维洞上的函数——这个「乘法」让上同调从一群孤立的群变成一座结构精密的环。读这一节时请抓住两个问题：**为什么同调没有乘法？**（循环之间没有自然的「切半拼接」）；**为什么上同调有？**（函数之间可以相乘，且「切半拼接」由
$\sigma$ 的前后半段给出）。理解这两个问题，你就理解了上同调环在整个代数拓扑中不可替代的地位。

## 1 杯积定义：函数相乘

先定义链层的杯积，再证它诱导上同调层的乘法。设 $R$ 是交换环（取 $R = \mathbb{Z}$ 或 $\mathbb{Z}_2$
理解即可），$\varphi \in C^k(X;R)$、$\psi \in C^l(X;R)$。对奇异 $(k+l)$-单形 $\sigma \colon
\Delta^{k+l} \to X$，记其前 $k+1$ 个顶点张成的面为 $\sigma|_{[v_0,\dots,v_k]}$，后 $l+1$
个顶点张成的面为 $\sigma|_{[v_k,\dots,v_{k+l}]}$。定义：

$$(\varphi \smile \psi)(\sigma) := \varphi\big(\sigma|_{[v_0,\dots,v_k]}\big) \cdot \psi\big(\sigma|_{[v_k,\dots,v_{k+l}]}\big)$$

即「在 $\sigma$ 的前半段取 $\varphi$ 的值，后半段取 $\psi$ 的值，相乘」。这是把「$X$ 上的两个函数」复合成一个
$(k+l)$-上链的方式——**杯积把链「切开」，两个因子各管一半**。<span class="marginnote">几何直觉：杯积不是「逐点相乘」（那需要链是同一个），而是「把一个 $(k+l)$-维对象的前 $k$ 维用
$\varphi$ 度量、后 $l$ 维用 $\psi$ 度量」。在流形上，这就是微分形式的<strong>外积</strong> $\omega
\wedge \eta$——测「两个方向的微分同时变化」。</span>

**关键验证**：$\delta(\varphi \smile \psi) = \delta\varphi \smile \psi + (-1)^k\,
\varphi \smile \delta\psi$（广义莱布尼茨法则，又是 $(-1)^k$
的定向账本）。由此推出：上循环的杯积是上循环，上边界（乘以任意上链）是上边界，于是杯积**良定义**在同调层，给出

$$\smile \colon H^k(X;R) \times H^l(X;R) \to H^{k+l}(X;R)$$

## 2 分级交换性

杯积在交换环 $R$ 上满足**分级交换律（graded commutativity）**：

$$\alpha \smile \beta = (-1)^{k \cdot l}\, \beta \smile \alpha, \qquad \alpha \in H^k,\ \beta \in H^l$$

**辨析｜易错点：** 这不意味着「完全交换」——当 $k, l$ 都是奇数时，$(-1)^{kl} = -1$，杯积是**反交换**的（与微分形式外积
$\omega \wedge \eta = -\eta \wedge \omega$
一致）。「奇维类与奇维类交换要变号」，这是上同调环最常见也最容易被忽略的性质。$R$ 含 $2$-挠（如 $\mathbb{Z}_2$）时 $-$ 与
$+$ 相同，反交换自动变为交换。

**证明思路**：交换 $\alpha$ 与 $\beta$ 相当于把 $\sigma$ 的「前半」与「后半」对调，这需要沿着 $\Delta^{k+l}$
的对角线「扭一下」——构造一个显式的**链同伦**（几何上是把单形沿中间翻折），它引入符号 $(-1)^{kl}$。<span class="marginnote">这个「扭」本质上来自 $S^1$ 上的对合换向（$t \mapsto 1-t$），是定向理论的基础操作。它也是第 4
篇 Poincaré 对偶里「对偶配对反对称性」的根源。</span>

## 3 上同调环：拼成环

把所有维数的上同调拼起来，连同杯积：

$$H^*(X;R) := \bigoplus_{n \ge 0} H^n(X;R), \qquad (x, y) \mapsto x \smile y$$

$H^*(X;R)$ 成为**分级交换环**：加法是直和，乘法是杯积，单位元是 $1 \in H^0(X;R)$（每个连通分量上的常值函数
$1$）。<span class="marginnote">$H^0$ 的单位元扮演「乘上 1
不变」的角色；整环性（没有零因子）之类问题在各维间交互，是上同调环理论的深水区。对连通空间 $H^0 = R$，环的「地基」就是系数环本身。</span>

**例：射影空间的上同调环。**

- $H^*(\mathbb{CP}^n;\mathbb{Z}) = \mathbb{Z}[\alpha] / (\alpha^{n+1})$，其中 $\deg \alpha = 2$，$\alpha^k \neq 0$（$0 \le k \le n$）——**每上升二维，「乘 $\alpha$」给出新的生成元**，到 $\alpha^{n+1}$ 才为零。
- $H^*(\mathbb{RP}^n;\mathbb{Z}_2) = \mathbb{Z}_2[\alpha] / (\alpha^{n+1})$，$\deg \alpha = 1$。

这里的核心信息是「**幂零高度**」：$\alpha$ 的多少次方为零，正好等于空间的「维数上限」。$\mathbb{CP}^n$ 里 $\alpha^n
\neq 0$ 编码了「$n$ 个超平面可以彼此横截相交于一点」——**几何相交的信息被完整封存在上同调环里**。这为 Poincaré 对偶篇「杯积 =
交」的等价埋下最重要的伏笔。

## 4 公式解析：杯积定义

$$(\varphi \smile \psi)(\sigma) = \varphi\big(\sigma|_{[v_0,\dots,v_k]}\big) \cdot \psi\big(\sigma|_{[v_k,\dots,v_{k+l}]}\big)$$

- **第一步，切开单形**：$\sigma \colon \Delta^{k+l} \to X$ 有顶点 $v_0, \dots, v_{k+l}$。前 $k+1$ 个顶点张出「前脸」，后 $l+1$ 个张出「后脸」，两片面共享顶点 $v_k$（在低维一侧）。$\Delta^{k+l}$ 就这样被分成「一个 $k$-面 + 一个 $l$-面」。
- **第二步，分别求值**：$\varphi$ 在 $k$-面上取值，$\psi$ 在 $l$-面上取值，两者都是 $R$ 中元素，在 $R$ 中相乘。**两个上链的维数之和 = 目标上链的维数**，这是杯积的维数加法律 $\deg(\alpha \smile \beta) = \deg \alpha + \deg \beta$。
- **第三步，共享顶点**：两片面共享 $v_k$，正是保证「合成后仍是合法上链」的黏合点；若没有这个共享，乘积无法在 $C^{k+l}$ 上良定义。

**辨析｜易错点：** 杯积不依赖「$\sigma$ 的具体顶点标号」吗？$\sigma|_{[v_0,\dots,v_k]}$
的定向（顶点顺序）会影响符号，因此杯积在链层不交换；但最终的上同调层公式是有典范性的。做计算时务必固定「前半 / 后半」的切法，否则符号会错。


**例：$T^2 = S^1 \times S^1$ 的上同调环。** $H^0 = \mathbb{Z}$，$H^1 =
\mathbb{Z}\langle a, b \rangle$（$a, b$ 分别是纬线与经线的对偶类），$H^2 = \mathbb{Z}\langle
a \smile b \rangle$。由 Künneth（无挠）+ 杯积 = 交叉积拉回对角：$a \smile a = 0$，$b \smile b =
0$，而 $a \smile b = - b \smile a$（两个 1 维类反交换），且 $a \smile b \neq 0$。**几何读音**：$a
\smile b$ 非零，因为纬线与经线横截相交于一点——杯积编码「相交」。$a \smile a = 0$ 因为一条纬线与它自己平行不相交。**「杯积 =
相交数」在 $T^2$ 上已经完整呈现**，Poincaré 对偶篇只是把它一般化。

**一个必须澄清的记号**：$H^*(X)$ 的「分级交换环」意味着乘法要遵守 $\alpha\beta =
(-1)^{|\alpha||\beta|}\beta\alpha$，因此对 $T^2$ 的 1-类 $a, b$ 有 $a \smile b = -b
\smile a$；只有在系数含 $2$-挠（如
$\mathbb{Z}_2$）时才没有符号。**计算时若忘记符号，环结构会整个判错**——这是杯积最常见的坑。

**分辨力的实证**：$S^2 \times S^2$ 与 $S^2 \vee S^2 \vee S^4$ 同调群完全相同，但上同调环不同——前者
$H^4$ 的生成元是「$x \smile y$」（$x, y$ 为两个 $S^2$
的对偶类），而后者杯积全为零。**「同调相同、环不同」是上同调环存在的全部理由**，也是判断两个空间「同伦型是否相同」的第一把尺。

## 5 小结

- **杯积**：$\smile \colon H^k \times H^l \to H^{k+l}$，定义是「在链的前半段取 $\varphi$、后半段取 $\psi$，值相乘」。
- **莱布尼茨法则**：$\delta(\varphi \smile \psi) = \delta\varphi \smile \psi + (-1)^k \varphi \smile \delta\psi$，保证上同调层良定义。
- **分级交换律**：$\alpha \smile \beta = (-1)^{kl} \beta \smile \alpha$；奇维类反交换。
- **上同调环**：$H^*(X;R) = \bigoplus H^n$，带杯积；$H^*(\mathbb{CP}^n) = \mathbb{Z}[\alpha]/(\alpha^{n+1})$。
- **分辨力**：$S^2 \times S^2$ 与 $S^2 \vee S^2 \vee S^4$ 同调相同、上同调环不同。

在下一节，我们将登上一座高峰——**Poincaré 对偶**。它说：紧定向流形 $M^n$ 的上同调与同调「对折镜像」：$H^k(M) \cong
H_{n-k}(M)$