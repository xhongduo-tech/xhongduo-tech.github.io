---
title: 对偶码与 MacWilliams 恒等式
date: 2026-08-07
---

# 对偶码与 MacWilliams 恒等式

<div class="epigraph">
<p>对称性，无论你把它看作什么，都是某种秩序观念的体现。</p>
<footer>—— 外尔（Hermann Weyl），《对称》（1952）</footer>
</div>

<div class="article-byline">
<p>第二级 · 编码理论（纠错编码） ｜ Roth 第4章；MacWilliams & Sloane 第5章 ｜ 2026-08-07</p>
</div>

## 为什么从对偶码开始

在上一节我们认识了一个事实：校验矩阵 $H$ 的零空间是码 $\mathcal{C}$。那么 $H$ 的**行空间**是什么？它是一个新的码——**对偶码**。线性代数里「正交补」这个朴素概念，在编码理论里长出了一整套深刻结构：对偶码的尺寸、重量分布与原码之间存在精确的对称关系，其中最著名的就是 **MacWilliams 恒等式**——它让一个码的重量分布从「难以枚举」变成「通过对偶码轻松算出」。

对偶关系也是现代编码理论的枢纽：自对偶码（self-dual code）控制着 Golay 码、量子纠错码等深刻对象；校验矩阵的本质就是「把码的对偶当成校验方程」。<span class="marginnote">在第二级《线性代数》里，正交补 $\mathcal{C}^\perp$ 满足 $(\mathcal{C}^\perp)^\perp = \mathcal{C}$ 与 $\dim \mathcal{C} + \dim \mathcal{C}^\perp = n$——这两条将原封不动搬进编码理论。对偶码不是新对象，而是同一个子空间换了个视角。</span>

## 1 对偶码的定义与基本性质

**内积**：$\mathbb{F}_q^n$ 上的标准内积定义为

$$\boldsymbol{x} \cdot \boldsymbol{y} = x_1 y_1 + x_2 y_2 + \cdots + x_n y_n \pmod q$$

**对偶码（dual code）**：码 $\mathcal{C}$ 的对偶码是与 $\mathcal{C}$ 全部码字都正交的向量集合：

$$\mathcal{C}^\perp = \{\boldsymbol{x} \in \mathbb{F}_q^n \mid \boldsymbol{x} \cdot \boldsymbol{c} = 0, \; \forall \boldsymbol{c} \in \mathcal{C}\}$$

线性代数直接给出三条性质：

1. $\dim \mathcal{C}^\perp = n - k$（若 $\dim \mathcal{C} = k$）；
2. $(\mathcal{C}^\perp)^\perp = \mathcal{C}$（对偶的对偶是自身）；
3. **校验矩阵的行空间恰好是对偶码**：$H$ 的行张成 $\mathcal{C}^\perp$。这解释了为什么「$H \boldsymbol{c}^T = 0$」就是「$\boldsymbol{c}$ 与对偶码全部向量正交」。

**重点：一个码的校验矩阵，就是其对偶码的生成矩阵。** 这句双关语是编码理论的引理式开胃菜——$G_\mathcal{C} = H_{\mathcal{C}^\perp}$，$H_\mathcal{C} = G_{\mathcal{C}^\perp}$。<span class="marginnote">由此，一切「关于生成矩阵的结论」自动翻转为「关于校验矩阵的结论」。设计校验矩阵保证距离（见第3篇的「$d$ = 最小相关列数」定理），本质上就是在设计一个性能优良的对偶码。</span>

## 2 自正交码与自对偶码

当码与自己的对偶重合时，会出现非常特殊的对称结构。

**自正交码（self-orthogonal）**：$\mathcal{C} \subseteq \mathcal{C}^\perp$，即任意两个码字正交。此时 $\dim \mathcal{C} \le n/2$。

**自对偶码（self-dual）**：$\mathcal{C} = \mathcal{C}^\perp$。此时 $\dim \mathcal{C} = n/2$，且每个码字与自身正交：$\boldsymbol{c} \cdot \boldsymbol{c} = 0$。<span class="marginnote">在特征 2 的域上，「与自身正交」不意味着「本身为零」——$\boldsymbol{c} \cdot \boldsymbol{c} = \sum c_i^2 = \sum c_i$（因为 $0^2 = 0$、$1^2 = 1$），所以自正交等价于「每个码字的重量为偶数」。二元自对偶码必是偶重量码。</span>

一个著名的自对偶码例子是二元 **$[8, 4, 4]$ 扩展 Hamming 码**，也叫 Hamming 码的扩展：在 $[7,4,3]$ 码上补一个全码字奇偶校验位，得到 $[8,4,4]$ 码，它满足 $\mathcal{C} = \mathcal{C}^\perp$。<span class="marginnote">扩展 Hamming 码与后面的扩展 Golay 码、以及量子纠错里的 CSS 构造一脉相承。「补一个校验位让码自对偶」，是构造强结构码的经典手法。</span>

**辨析｜易错点：** 自正交 ≠ 码字两两不相关。内积为零只说明「模 $q$ 求和为零」，在 $\mathbb{F}_2$ 上它只是「重合的 1 的个数为偶数」。不要把有限域内积当作实数内积的直觉——$\boldsymbol{c} \cdot \boldsymbol{c} = 0$ 在实数里只可能 $\boldsymbol{c} = \boldsymbol{0}$，在 $\mathbb{F}_2$ 上却意味着偶重量。

## 3 MacWilliams 恒等式：对偶之间的桥梁

重量枚举器（见第3篇）是对偶理论的中心角色。

**MacWilliams 恒等式（二元情形）：** 设 $\mathcal{C}$ 是 $\mathbb{F}_2^n$ 上的 $[n, k]$ 线性码，则

$$W_{\mathcal{C}^\perp}(x, y) = \frac{1}{|\mathcal{C}|} W_{\mathcal{C}}(x + y, x - y)$$

其中 $|\mathcal{C}| = 2^k$。把 $W_{\mathcal{C}}(x, y) = \sum A_i x^i y^{n-i}$ 代进去，右边是 $(x+y)$、$(x-y)$ 的幂展开后按 $x$ 的指数重新收集系数。

**广义形式（$q$ 元）：** 对 $\mathbb{F}_q$ 上的 $[n, k]$ 码，

$$W_{\mathcal{C}^\perp}(x, y) = \frac{1}{|\mathcal{C}|} W_{\mathcal{C}}(x + (q-1)y, x - y)$$

这个公式的力量在于：**计算对偶码的重量分布不需要枚举对偶码的码字**，只需对原码的重量枚举器做一次代数代换。<span class="marginnote">例子：$[7,4]$ Hamming 码有重量分布 $(1, 0, 0, 7, 7, 0, 0, 1)$（3 个 1 的码字 7 个、4 个 1 的码字 7 个）。代入 MacWilliams 恒等式，立刻得到其对偶 $[7,3]$ 单纯形码的重量分布 $(1, 0, 0, 0, 0, 0, 7, 0)$——全部 7 个非零码字重量都是 4。若不借助恒等式，得枚举 $2^3 = 8$ 个码字；码长一长，这种「代数查重分布」的优势就成指数级放大。</span>

## 4 公式解析：MacWilliams 恒等式为什么成立

要理解恒等式，不能只背公式，得看它背后的「傅里叶视角」。这里以二元情形拆解。

- **第一步，写成指示函数**：重量枚举器可以写成对每个码字的求和 $W_\mathcal{C}(x, y) = \sum_{\boldsymbol{c} \in \mathcal{C}} f(\boldsymbol{c})$，其中 $f(\boldsymbol{c}) = x^{\mathrm{wt}(\boldsymbol{c})} y^{n - \mathrm{wt}(\boldsymbol{c})} = \prod_{i=1}^{n} \varphi(c_i)$，$\varphi(0) = y$、$\varphi(1) = x$。$f$ 是「逐位因子」的乘积。
- **第二步，换一个求和顺序**：$W_{\mathcal{C}^\perp}$ 是对 $\mathcal{C}^\perp$ 求和。关键技巧是把「对偶码求和」改写成「对全空间求和 + 正交投影」：利用正交补的测度性质，$\sum_{\boldsymbol{x} \in \mathcal{C}^\perp} g(\boldsymbol{x}) = \frac{1}{|\mathcal{C}|} \sum_{\boldsymbol{u} \in \mathbb{F}_2^n} \sum_{\boldsymbol{c} \in \mathcal{C}} (-1)^{\boldsymbol{u} \cdot \boldsymbol{c}} g(\boldsymbol{u})$。
- **第三步，逐位分离**：内层对 $\boldsymbol{c}$ 求和时，由于 $\boldsymbol{u} \cdot \boldsymbol{c} = \sum u_i c_i$，$\sum_{\boldsymbol{c}} (-1)^{\boldsymbol{u}\cdot\boldsymbol{c}} g(\boldsymbol{u})$ 可以逐位分解，最终每个坐标贡献一个「$u_i = 0$ 或 $1$ 时的替换因子」，合起来恰好把 $(x, y)$ 换成 $(x+y, x-y)$。

**直觉：** 恒等式本质上是对二元群 $\mathbb{F}_2^n$ 上的一个离散傅里叶变换。$(-1)^{\boldsymbol{u}\cdot\boldsymbol{c}}$ 是群特征（character），「对子群求和 + 平均」就是在做正交分解。你在第一级《信息论》和第三级《信号与系统》见过的傅里叶「时域 ↔ 频域」互逆，在有限域上以「码 ↔ 对偶码」的形式重现——同一个数学内核，三种外衣。

## 5 实战：用恒等式算一次重量分布

拿 $[7,4]$ Hamming 码 $\mathcal{C}$ 完整走一遍。它的重量分布是 $A = (1, 0, 0, 7, 7, 0, 0, 1)$（3 个 1 的码字 7 个、4 个 1 的码字 7 个），枚举器为

$$W_{\mathcal{C}}(x, y) = y^7 + 7x^3 y^4 + 7x^4 y^3 + x^7$$

代入二元 MacWilliams 恒等式（$|\mathcal{C}| = 16$）：

$$W_{\mathcal{C}^\perp}(x, y) = \frac{1}{16}\Big[(x-y)^7 + 7(x+y)^3(x-y)^4 + 7(x+y)^4(x-y)^3 + (x+y)^7\Big]$$

逐项展开、按 $x$ 的幂合并同类项。这个展开冗长，但关键只在一个现象：**奇数项全部抵消**。最终只剩

$$W_{\mathcal{C}^\perp}(x, y) = y^7 + 7x^4 y^3$$

即 $[7,3]$ 单纯形码的重量分布 $(1, 0, 0, 0, 0, 7, 0, 0)$——7 个非零码字全部重量为 4。<span class="marginnote">为什么非零码字全是重量 4？对偶码 $[7,3]$ 的校验矩阵是原码 $[7,4]$ 的生成矩阵，其列是全部非零 3 位向量（Hamming 码的对偶结构）。可以验证：任意非零 3 维消息经 $G$ 编码后恰有 4 个 1——这是「单纯形」名字的由来，对应几何里的 3 维单纯形顶点。</span>

**方法论收获**：要得到 $[7,3]$ 码的重量分布，直接枚举只需 $8$ 个码字，恒等式的优势还不明显；但对 $[2^{10}-1]$ 级的码，直接枚举对偶码是天文数字，而多项式代换在纸上就能完成——**代数换分布，指数换多项式**，这就是恒等式的工程价值。

## 6 对偶码理论的现代回声

MacWilliams 恒等式不是博物馆藏品，它是一系列深刻结果的入口：

**Gleason 定理**：对二元自对偶码，重量枚举器必须落在某个由特定多项式张成的环里——这给出了「哪些重量分布可能存在」的强约束，也解释了扩展 Golay 码重量分布的漂亮结构。
**覆盖半径**：对偶码还牵出「一个码离全空间最远有多远」的覆盖半径概念，它与 $t$、$d$ 一起构成「好码三围」；覆盖半径与对偶最小距离之间存在对偶不等式（Delsarte 界），是线性规划界的前奏。
- **量子纠错**：CSS 构造用一对「嵌套」的经典码（$\mathcal{C}_2 \subset \mathcal{C}_1$）造出量子码，对偶码理论是它的语言。
- **密码学**：线性码的对偶与「覆盖半径」「k-重量枚举器」一起，构成编码密码学（McEliece 体制）的分析工具。<span class="marginnote">McEliece 公钥密码用「带结构却看似随机的 Goppa 码」做陷门，其安全性分析高度依赖重量分布与对偶性质——这是编码理论进入后量子密码学的通道。</span>

## 7 小结

- **对偶码** $\mathcal{C}^\perp$ 是与全部码字正交的向量集合；$\dim \mathcal{C}^\perp = n-k$，$(\mathcal{C}^\perp)^\perp = \mathcal{C}$。
- **一句话双关**：码的校验矩阵 = 对偶码的生成矩阵。
- **自正交** $\subseteq$ 与**自对偶** $=$；二元自对偶码必为偶重量码，扩展 Hamming 码 $[8,4,4]$ 是典范。
- **MacWilliams 恒等式** $W_{\mathcal{C}^\perp}(x,y) = \frac{1}{|\mathcal{C}|} W_\mathcal{C}(x + (q-1)y, x-y)$：由原码重量分布直接算出对偶码分布，无需枚举。
- 恒等式的本质是有限群上的离散傅里叶变换，「码 ↔ 对偶码」对应「时域 ↔ 频域」。
- Gleason 定理、量子 CSS 构造、McEliece 密码学，都是对偶理论的现代延伸。
- 实战演示：$[7,4]$ Hamming 码经恒等式一算，对偶 $[7,3]$ 单纯形码的 7 个非零码字全是重量 4。
- 方法论口诀：「代数换分布、指数换多项式」——恒等式把枚举难题变成多项式代换。
- 理解恒等式靠「傅里叶视角」：$(-1)^{\boldsymbol{u}\cdot\boldsymbol{c}}$ 是群特征，子群求和 + 平均 = 正交分解。
- 自对偶码在量子纠错（CSS 构造）与后量子密码（McEliece）中继续发挥核心作用。
- 实战例子提示：对偶码的重量分布「代数可得」的前提是原码分布已知——两条路互为捷径。
- **一个易错点**：恒等式给的是「对偶码的重量分布」，不是「原码的译码方法」——分布能算 ≠ 译码可行，两者别混为一谈。

在下一节，我们回到最基本的问题：给定额外冗余，一个码的 $n, k, d$ 能怎样权衡——码参数界：Hamming、Singleton 与 Gilbert-Varshamov 界。