---
title: 上链复形与上同调群
date: 2026-08-07
---

# 上链复形与上同调群

<div class="epigraph">
<p>同调数出的是洞，上同调数出的是洞上的函数——而函数可以相乘，洞不能。</p>
<footer>—— 迈克尔 · 阿蒂亚（Michael Atiyah）</footer>
</div>

<div class="article-byline">
<p>第二级 · 代数拓扑 ｜ Hatcher 第3.1章 ｜ 2026-08-07</p>
</div>

## 为什么从上链开始

前面 11 篇的整个世界都在「往下走」：$n$-链的边界是 $(n-1)$-链，$\partial_n$ 把维数降一。现在我们要**掉头**：把每个
$C_n(X)$ 换成它的对偶空间 $\operatorname{Hom}(C_n(X),
G)$，边界算子换成对偶的**上边界算子（coboundary）** $\delta$，让维数**升一**。由此得到的**上同调群（cohomology
groups）** $H^n(X;G)$ 不是同调的复制品——它记录了「同调类上的
$G$-值函数」，天然带有同调完全没有的**乘法结构**（下一节的杯积），并最终导向 Poincaré 对偶那类「同调与上同调在流形上对称」的深刻定理。

万有系数定理已经预告：$H^n(X;G)$ 由 $\operatorname{Ext}(H_{n-1}, G)$ 与
$\operatorname{Hom}(H_n, G)$
决定。但**直接定义**上同调仍然必要——它不依赖万有系数定理独立存在，而且它的元素是「链上的函数」，这种「可求值」的身份是杯积、帽积、特征类等一切后续构造的出发点。<span class="marginnote">方向反转的意义：$C^n \xrightarrow{\delta} C^{n+1}$
让「上同调维数越高」对应「链维数越高」——于是「$X$ 上的 $n$-维函数空间」自然地在 $n$
增加时变大，这与同调「维数越高洞越少」的直觉方向相反，恰是函数与几何的分工。</span>


上同调「反转方向」这件事，初看是纯记号，实则是一次视角革命。同调问「有哪些循环」，上同调问「循环上能定义哪些函数」——后者天然带有「函数环」的结构（下一节的杯积），并且天然是**反变**的（映射把函数拉回来）。读这一节时请带着对比的眼光：**同调与上同调的每一项几乎都是对偶的**，但方向、维数记号、变异性全部翻转。把这两套机器并排放在脑子里，后面读到万有系数定理、Poincaré
对偶时会非常顺畅。

## 1 上链群与上边界算子

**上链群（cochain group）**：

$$C^n(X; G) := \operatorname{Hom}_\mathbb{Z}\big(C_n(X),\ G\big)$$

即「从 $n$-链群到 $G$ 的群同态」。$C^n(X;G)$ 中的元素称为 **$n$-上链（cochain）**。它是「给每个 $n$-单形分配一个
$G$ 中的值」的规则——因为 $C_n(X)$ 由奇异 $n$-单形自由生成，一个同态完全由它在各单形上的取值决定。<span class="marginnote">当 $G$ 是环（如 $\mathbb{Z}$、$\mathbb{Q}$）时，$C^n$
还有更多的代数结构；但目前只需 $G$ 是阿贝尔群。直观上把 $n$-上链想成「$n$-维对象上的函数」即可，微分几何里的「微分 $n$-形式」就是 $G
= \mathbb{R}$ 情形下的连续版本。</span>

**上边界算子（coboundary）** $\delta^n \colon C^n(X;G) \to C^{n+1}(X;G)$ 定义为
$\partial_{n+1}$ 的**对偶/转置**：

$$\delta^n(\varphi) := \varphi \circ \partial_{n+1} \qquad \text{即} \qquad \big(\delta^n \varphi\big)(\sigma) = \varphi\big(\partial_{n+1}(\sigma)\big)$$

对每个奇异 $(n+1)$-单形 $\sigma$，$\delta^n\varphi$ 在 $\sigma$ 上的值 = $\varphi$ 在
$\sigma$ 的边界上的值。因为 $\partial^2 = 0$，对偶后同样 $\delta^2 = 0$：

$$\delta^{n+1} \delta^n = \big(\partial_{n+2}\big)^* \circ \big(\partial_{n+1}\big)^* = \big(\partial_{n+1} \partial_{n+2}\big)^* = 0$$

于是得到**上链复形（cochain complex）**，方向与链复形相反：

$$0 \to C^0(X;G) \xrightarrow{\;\delta^0\;} C^1(X;G) \xrightarrow{\;\delta^1\;} C^2(X;G) \to \cdots$$

**辨析｜易错点：** $\delta$ 的方向是「升维」：$\delta^n \colon C^n \to
C^{n+1}$。初学者常被「同调与上同调同名下标」搞晕——记法上 $H_n$ 用下标、$H^n$ 用上标，边界用 $\partial$、上边界用
$\delta$，方向恰好相反。**同调测「有多少洞」，上同调测「洞上有多少函数」。**

## 2 上同调群：核模像，方向向上

定义：

$$Z^n(X;G) := \ker \delta^n \quad (\text{$n$-上循环}), \qquad B^n(X;G) := \operatorname{im}\delta^{n-1} \quad (\text{$n$-上边界}),$$

$$H^n(X;G) := \frac{Z^n(X;G)}{B^n(X;G)} = \frac{\ker \delta^n}{\operatorname{im}\delta^{n-1}}$$

$H^n(X;G)$ 称为 $X$ 的 **$n$ 维上同调群**。

**直觉**：一个 $n$-上循环 $\varphi$ 是「在一切 $n$-边界上取值为 0」的链函数——它只依赖同调类，给每个 $n$-维洞赋一个
$G$-值。两个上循环若差一个上边界（差一个「拉自 $(n-1)$-链」的函数），就视为同一上同调类。所以 $H^n$ 的元素可理解为「$n$-维洞上的
$G$-值函数」，配对记为：

$$\langle \varphi,\ \alpha \rangle := \varphi(\alpha) \in G, \qquad \varphi \in H^n(X;G),\ \alpha \in H_n(X)$$

这是**求值配对（evaluation pairing）**，它是「上同调与同调互为对偶」的最初体现。<span class="marginnote">配对标 $\langle \varphi, \alpha \rangle$
看起来简单，却是代数拓扑最常用的语言之一：在流形上，$\langle \cdot, \cdot \rangle$
把上同调类（如「闭形式」）作用到同调类（如「循环」）上得到数——这正是「积分」的抽象版本，也是 Poincaré 对偶篇里「对偶配对」的原型。</span>

**例：$H^n(S^m)$。** 由万有系数定理或直接计算：$H^n(S^m;\mathbb{Z}) = \mathbb{Z}$（$n = 0,
m$），其余为 0。$H^m(S^m)$ 的生成元 $\varphi$ 满足 $\langle \varphi, [S^m] \rangle =
1$——「给基本类赋 1」的函数。

## 3 上同调的函子性：映射把函数拉回来

连续映射 $f \colon X \to Y$ 诱导链映射 $f_\# \colon C_n(X) \to C_n(Y)$，取对偶得到「反向」的映射
$f^* \colon C^n(Y;G) \to C^n(X;G)$：

$$f^*(\varphi) := \varphi \circ f_\#, \qquad \text{即} \quad \big(f^*\varphi\big)(\sigma) = \varphi\big(f \circ \sigma\big)$$

因为 $f^* \delta = \delta f^*$（转置与复合可交换），$f^*$ 诱导上同调同态 $f^* \colon H^n(Y;G) \to
H^n(X;G)$。**注意方向翻转**：空间映射 $f \colon X \to Y$ 在同调上 $f_\* \colon H_n(X) \to
H_n(Y)$（协变），在上同调上 $f^* \colon H^n(Y) \to H^n(X)$（**反变**）。<span class="marginnote">方向翻转是「函数拉回」的必然结果：$Y$ 上的函数通过 $f$ 拉回成 $X$
上的函数，自变量反着走。这让上同调成为一个<strong>反变函子</strong>，而同调是<strong>协变函子</strong>——这个区别在障碍理论篇会直接派上用场。</span>

由函子性与同伦不变性（$f \simeq g \Rightarrow f^* = g^*$），上同调同样是同伦不变量；Eilenberg–Steenrod
公理对上同调也有镜像版本，只是连接同态 $\delta_*$ 的方向变为升维。

## 4 公式解析：上边界算子的双重角色

$$\big(\delta^n \varphi\big)(\sigma) = \varphi\big(\partial_{n+1}(\sigma)\big), \qquad \delta^n = \partial_{n+1}^*$$

- **第一步，对偶**：$\delta^n$ 是 $\partial_{n+1} \colon C_{n+1}(X) \to C_n(X)$ 的转置。转置把「从 $C_n$ 出发」变为「从 $\operatorname{Hom}(C_n, G)$ 出发」，方向反转。
- **第二步，求值语义**：$\delta^n\varphi$ 在 $\sigma$ 上的值，等于 $\varphi$ 在 $\sigma$ 的**边界**上的值。这就是「Stokes 定理」的代数雏形：**边界的函数值 = 函数在上链上的「积分」**。$\delta^2 = 0$ 是 $\partial^2 = 0$ 的对偶，无需新证明。
- **第三步，核模像语义**：$\ker \delta^n$ = 在一切边界上为零的函数 = 只依赖同调类的函数；$\operatorname{im}\delta^{n-1}$ = 从低维「拉回」的函数 = 平凡函数。商群 $H^n$ 留下的是「本质上新的洞函数」。

**辨析｜易错点：** 求值 $\langle \varphi, \alpha \rangle$ 的定义域里 $\alpha$
是**同调类**，不是任意链。因为 $\varphi$
在边界上为零，它在同调类上的取值才良定义（同类不同代表元只差边界，函数值不变）——**这正是「上循环」三个字的意义**。


**例：直接算 $H^*(S^1)$，不看 UCT。** $C_0(S^1) = \mathbb{Z}\langle \text{点}
\rangle$，$C_1(S^1) = \mathbb{Z}\langle \text{绕圈} \rangle$，$\partial_1 =
0$。对偶：$C^0 = \operatorname{Hom}(\mathbb{Z},\mathbb{Z}) = \mathbb{Z}$，$C^1 =
\operatorname{Hom}(\mathbb{Z},\mathbb{Z}) = \mathbb{Z}$，$\delta^0 = 0$。于是 $H^0
= \mathbb{Z}$，$H^1 = \ker \delta^1 / \operatorname{im}\delta^0 = \mathbb{Z}/0
= \mathbb{Z}$。**求值配对** $\langle \varphi, \alpha \rangle$ 中，$\varphi$ 给「绕 $k$
圈」的循环赋 $k$ 的整数倍——「积分」的雏形。直接计算让你看到上同调不是神秘的「对偶群」，而是**「链上函数的核模像」**，与同调是同一台机器的镜像。

**为什么「函数」比「循环」多一层结构**：$H_n$ 里的类是几何对象（循环模边界），没有自然的乘法；$H^n$
里的类是函数，函数可以逐点乘——下一节杯积的存在，根源就在「上同调的元素是函数」。**这正是上同调在代数拓扑里后来居上的根本原因**：不是因为它知道得更多（万有系数定理说它由同调决定），而是因为它**能相乘**。

## 5 小结

- **上链群**：$C^n(X;G) = \operatorname{Hom}(C_n(X), G)$，元素是「$n$-链上的 $G$-值函数」。
- **上边界算子**：$\delta^n = \partial_{n+1}^*$，$(\delta\varphi)(\sigma) = \varphi(\partial\sigma)$，方向升维，$\delta^2 = 0$。
- **上同调群**：$H^n(X;G) = \ker\delta^n / \operatorname{im}\delta^{n-1}$；元素给同调类赋 $G$-值，配对 $\langle \varphi, \alpha \rangle$。
- **反变函子**：$f \colon X \to Y$ 诱导 $f^* \colon H^n(Y) \to H^n(X)$，方向翻转；同伦不变。
- **例**：$H^n(S^m;\mathbb{Z}) = \mathbb{Z}$（$n=0,m$），生成元对基本类赋 1。

在下一节，我们将开发上同调最核心的独门武器——**杯积**。它给上同调群配上乘法，把它升级成**上同调环**
$H^*(X)$