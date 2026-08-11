---
title: 万有包络代数与 PBW 定理
date: 2026-08-11
---

# 万有包络代数与 PBW 定理

<div class="epigraph">
<p>把非结合的括号翻译成结合律的乘法，是表示论最深的直觉之一。</p>
<footer>—— 让-皮埃尔 · 塞尔（Jean-Pierre Serre，意译）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 李代数与李群 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么需要结合代数

李括号不满足结合律，这让我们无法直接利用结合代数的庞大工具箱（单位元、模块、张量积、张量代数……）。**万有包络代数（universal enveloping algebra）** $U(L)$ 提供了一座桥梁：它是「把 $L$ 装入一个结合代数、同时保留括号信息」的泛函构造，使得 **$L$ 的表示 ⟺ $U(L)$ 的模**。<span class="marginnote">物理学中这几乎是默认事实：角动量算符（$\mathfrak{su}(2)$ 的李代数）对应的实际算子作用在希尔伯特空间上时，用的是它们的结合代数的乘法律——万有包络代数就是「把 $[\cdot,\cdot]$ 换成 $AB-BA$」的严格框架。</span>

## 1 构造：张量代数除以理想

对域 $\mathbb{F}$ 上的李代数 $L$，构造 $L$ 的**张量代数**：

$$T(L) = \bigoplus_{n \ge 0} L^{\otimes n}, \qquad L^{\otimes 0} = \mathbb{F}$$

张量代数是自由结合代数（每个元素是有限多个向量的张量积的线性组合），乘法是张量积。**万有包络代数（universal enveloping algebra）**定义为其商：

$$U(L) = T(L) / \langle x \otimes y - y \otimes x - [x, y] \mid x, y \in L \rangle$$

即把「$xy - yx - [x,y]$」这类元素全部压成零。<span class="marginnote">这个商的意思是：在 $U(L)$ 里，$L$ 的元素的乘法满足 $xy - yx = [x,y]$——李括号变成了交换子。$U(L)$ 的单位元来自 $L^{\otimes 0} = \mathbb{F}$。</span>

**万有性质（universal property）**：$U(L)$ 是满足如下性质的唯一（在同构意义下）结合代数：对任意结合代数 $A$ 与线性映射 $f: L \to A$（满足 $f([x,y]) = f(x)f(y) - f(y)f(x)$），存在唯一代数同态 $\tilde f: U(L) \to A$ 使 $\tilde f|_L = f$。

**核心事实（模-表示对应）**：$L$ 在 $V$ 上的表示 $\phi: L \to \mathfrak{gl}(V)$ ⟺ $V$ 是 $U(L)$-模（$U(L)$ 作用在 $V$ 上，$x \cdot v = \phi(x)v$）。这使「$L$ 的表示论」变成「$U(L)$ 的模论」，可以自由使用张量积、Hom、张量模等结合代数工具。

## 2 PBW 定理：把有序乘积当成基

$U(L)$ 作为无穷维代数，基是什么？Poincaré–Birkhoff–Witt 定理给出答案。

**PBW 定理（Poincaré–Birkhoff–Witt theorem）**：设 $L$ 有有序基 $x_1, x_2, \dots, x_n$。则 $U(L)$ 的一组基由**严格有序（标准）单项式**构成：

$$\{ x_1^{a_1} x_2^{a_2} \cdots x_n^{a_n} \mid a_1, a_2, \dots, a_n \ge 0 \text{（有限支撑）} \}$$

即「下指标不降」的乘积，如 $x_2 x_3 x_3 x_5$ 而不含 $x_3 x_2$。<span class="marginnote">直观：$xy$ 与 $yx$ 在 $U(L)$ 里差一个 $[x,y]$（在 $L$ 中），所以「乱序」的乘积总能通过交换子规整成有序乘积的线性组合。PBW 说：这些有序乘积线性无关，正好撑起整个 $U(L)$。</span>

**推论**：自然映射 $L \to U(L)$（把 $x$ 送进 $x$）是**单射**——李代数可以看成它的包络代数的子空间。且若 $L$ 可解/幂零，$U(L)$ 作为结合代数有相应结构（Harish-Chandra 理论的基础）。

## 3 公式解析：为什么 $x y - y x$ 有资格做基

用 $L = \mathfrak{sl}(2,\mathbb{C})$（基 $e, f, h$）检验 PBW。$U(L)$ 的标准基是

$$\{ e^a f^b h^c \mid a, b, c \ge 0 \}$$

但 $U(L)$ 的元素也可以写成交错次序，比如 $e h$ 和 $h e$ 的关系：

$$e h = h e + [e, h] = h e - 2e$$

三步拆解：

- **第一步，回忆交换子**：$[e, h] = eh - he = -2e$（即 $[h, e] = 2e$ 移项），所以 $eh = he - 2e$。
- **第二步，看成重排**：任意乘积都能用「把低指标挪到左边」的规则重排成 $e^a f^b h^c$ 的线性组合，系数来自交换子。
- **第三步，唯一性**：PBW 断言这样重排后得到的系数**唯一**——没有多余关系。否则 $U(L)$ 维数会更小，表示论就会「缺元」。

**核心要点**：PBW 定理是整个表示论「有东西可算」的地基：它告诉我们 $U(L)$ 不会太小，李代数元素在包络代数中彼此独立，因而最高权理论（第 11 篇）里的 Verma 模不会「意外坍塌」。<span class="marginnote">对比：对有限维李代数，$U(L)$ 是无穷维的（$e^a f^b h^c$ 有无穷多个）。这解释了为什么最高权表示（下一节）需要 Verma 模这种无穷维对象来兜底。</span>

## 4 应用：Casimir 元素与表示论的标量

$U(L)$ 中最重要的元素之一是 **Casimir 元素（Casimir element）**。对半单 $L$，取 Killing 型下 $L$ 的一组正交基 $x_1, \dots, x_n$（$\kappa(x_i, x_j) = \delta_{ij}$），定义

$$\Omega = \sum_i x_i x_i \in U(L)$$

**核心事实**：$\Omega$ 与 $U(L)$ 中所有元素交换（在 $L$ 的伴随作用下不变），因此在每个不可约表示中由 Schur 引理作用为**标量**——这个标量可以分离表示，是物理中「Casimir 算符 = 好量子数」的代数根源。<span class="marginnote">对 $\mathfrak{sl}(2)$，取归一化基可得 $\Omega = \tfrac12(ef + fe) + \tfrac14 h^2$，在表示 $V_n$ 上作用为 $\tfrac14 n(n+2)$——正是自旋 $j = n/2$ 时 $J^2$ 的本征值 $j(j+1)$ 的雏形（差一个归一化因子）。</span>

**辨析｜易错点：** Casimir 元素依赖 Killing 型的正交基选取，但作为 $U(L)$ 元素（在 $\mathbb{C}$ 上）是**唯一**的——只要 $L$ 半单。初学者常担心「换基会不会变」：答案是元素不变，只是表达式的系数变。这也是它被称为「Laplacian 的李代数类比」的原因（Laplacian 也不依赖坐标选取）。

## 5 小结

- **万有包络代数** $U(L) = T(L) / \langle x y - y x - [x,y]\rangle$，是「把括号换成交换子」的泛函结合代数，由万有性质唯一确定。
- **模-表示对应**：$L$ 的表示 ⟺ $U(L)$-模；表示论转化为模论，张量积、Hom 等工具全数开放。
- **PBW 定理**：$U(L)$ 的基是严格有序单项式 $\{x_1^{a_1}\cdots x_n^{a_n}\}$；自然映射 $L \hookrightarrow U(L)$ 单射。
- **Casimir 元素** $\Omega = \sum x_i x_i$：中心元素，不可约表示中为标量，可分离表示。
- PBW 保证包络代数「不小」，Verma 模理论（下一节）由此站得住脚。

在下一节，我们将用 $U(L)$ 的工具构造最重要的表示——**最高权表示与 Verma 模**，并品尝 Harish-Chandra 定理的味道。
