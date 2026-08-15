---
title: 单范畴与对称单范畴、辫结构
date: 2026-08-07
---

# 单范畴与对称单范畴、辫结构

<div class="epigraph">
<p>张量积不是一个二元运算，而是一种生活方式。</p>
<footer>—— 依据桑德斯 · 麦克莱恩（Saunders Mac Lane）</footer>
</div>

<div class="article-byline">
<p>第二级 · 范畴论 ｜ Mac Lane Ch. VII, XI ｜ 2026-08-07</p>
</div>

## 为什么从单范畴开始

到目前为止，我们的范畴里只有「纵向」的复合。但数学里大量结构是**横向**组合出来的：两个向量空间的张量积、两个集合的笛卡尔积、两个代数对象的外直和。范畴论把这种横向运算抽象出来，得到**单范畴（monoidal category）**——它同时带乘法（张量积）与单位对象，就像「范畴版本的幺半群」。<span class="marginnote">单范畴是张量网络、量子场论、表示论与机器学习张量运算的共同语言；从 $n$ 维张量到神经网络的层组合，都可以装进「单范畴 + 态射」的框架。</span>更妙的是，当你追问「$a \otimes b$ 与 $b \otimes a$ 如何互换」时，会得到对称结构与辫结构的区分——前者给出线性代数的平凡交换，后者给出辫群、纽结不变量等一整个拓扑世界。

## 1 单范畴的定义

**单范畴（monoidal category）** $(\mathbf{C}, \otimes, I, \alpha, \lambda, \rho)$ 包含：

双函子 $\otimes: \mathbf{C} \times \mathbf{C} \to \mathbf{C}$（**张量积**）；
- 单位对象 $I$；
- 自然同构**结合子** $\alpha_{a,b,c}: (a \otimes b) \otimes c \xrightarrow{\cong} a \otimes (b \otimes c)$；
- 自然同构**左单位子** $\lambda_a: I \otimes a \xrightarrow{\cong} a$ 与**右单位子** $\rho_a: a \otimes I \xrightarrow{\cong} a$。

它们必须满足两条**一致性（coherence）**条件：**五边形公理**（图内一切由 $\alpha$ 组成的路径都交换）与**三角形公理**（$\lambda$、$\rho$、$\alpha$ 相互协调）。<span class="marginnote">五边形公理的作用是保证「多层括号随便怎么结合都相同」——正是它在背后让「我们通常把 $a \otimes b \otimes c$ 直接写出来而不加括号」是安全的。</span>

**严格单范畴（strict monoidal category）**：三个同构都是恒等——括号与单位彻底消失。**一致性定理（coherence theorem）**说：任何单范畴都等价于一个严格单范畴，因此「在计算层面可以假装结合子不存在」——这是 Mac Lane 最漂亮的定理之一。

**算例：结合子的存在感。** 在 $\mathbf{Set}$ 中 $(A \times B) \times C$ 与 $A \times (B \times C)$ 严格同构（只是配对括号不同），$\alpha$ 可取恒等——$\mathbf{Set}$ 是严格单范畴。而 $\mathbf{Vect}$ 里 $(V \otimes W) \otimes U$ 与 $V \otimes (W \otimes U)$ 只是自然同构（张量积的基底重排），并非恒等：$\mathbf{Vect}$ 非严格。一致性定理说「非严格也不怕」——所有括号重排都换到同构。

**辨析｜易错点：** 单范畴不是「范畴化的幺半群」这么简单——幺半群的结合律是等式，单范畴的结合律只要求**自然同构**并满足一致性。区别在「同构 vs 相等」：我们从不要求 $a \otimes (b \otimes c)$ 严格等于 $(a \otimes b) \otimes c$，只要求它们之间有一个好的同构。

## 2 单范畴的实例

**$(\mathbf{Set}, \times, \{\ast\})$**：笛卡尔积 + 单点集。
- **$(\mathbf{Set}, \sqcup, \emptyset)$**：不交并 + 空集——同一个底层范畴可以有不同的单结构，且它们互不等价。
- **$(\mathbf{Vect}_k, \otimes_k, k)$**：张量积 + 基域；机器学习里把一批向量张成的高阶张量就是这里的东西。

**数值算例：张量积与直和的区别。** 在 $\mathbf{Vect}_{\mathbb{R}}$ 里，$\mathbb{R}^2 \otimes \mathbb{R}^3 \cong \mathbb{R}^{6}$（维数相乘 $2 \times 3 = 6$），而直和 $\mathbb{R}^2 \oplus \mathbb{R}^3 \cong \mathbb{R}^{5}$（维数相加）。同一个底层范畴上 $\otimes$ 与 $\oplus$ 两套单结构并存，维数的「乘 vs 加」是区分它们最省事的指标——也解释了为何 `einsum` 的维度按乘法扩张，而 batch 拼接按加法走。
- **$(\mathbf{Ab}, \otimes_{\mathbb{Z}}, \mathbb{Z})$**、链复形范畴上的张量积与微分——同调代数的标准舞台。<span class="marginnote">后文《加性范畴与 Abel 范畴》里直和也是单结构（$\oplus$）。注意：同一个范畴上 $\otimes$ 与 $\oplus$ 两套单结构并存，是代数里最常见也最容易混的现象。</span>

## 3 对称与辫：方向不同的交换

单范畴未必交换。给交换结构建模，Mac Lane 区分了两种概念：

**辫单范畴（braided monoidal category）**：带自然同构**辫子** $\gamma_{a,b}: a \otimes b \xrightarrow{\cong} b \otimes a$，满足**六边形公理**（与 $\alpha$ 相容）；但 $\gamma_{b,a} \circ \gamma_{a,b}$ 不必是恒等。
- **对称单范畴（symmetric monoidal category）**：额外要求 $\gamma_{b,a} \circ \gamma_{a,b} = 1$——「交换两次等于不换」。

**直觉：** 对称 = 像普通乘法那样彻底交换；辫 = 交换会留下「辫子」，两次交换 ≠ 恒等，而是换了个拓扑。<span class="marginnote">辫子的名字来自真实世界的辫子：把两根绳子交换再交换，得到的不是原样而是缠了两圈的辫子。辫单范畴精确模拟了这条直觉，并由此产生辫群、量子群与 Jones 多项式——纽结不变量就在这个框架里构造。</span>

**数值算例：超向量空间——带符号的对称。** 在 $\mathbb{Z}/2$-分次向量空间（超空间）里，交换同构取符号约定 $\gamma(x \otimes y) = (-1)^{\lvert x \rvert \lvert y \rvert}\, y \otimes x$（$\lvert x \rvert \in \{0, 1\}$ 是奇偶度）。对两个奇元素 $\lvert x \rvert = \lvert y \rvert = 1$，$\gamma(x \otimes y) = -y \otimes x$，但 $\gamma^2(x \otimes y) = x \otimes y$ 仍是恒等——所以超空间是「对称但带符号」，并未真正辫化。真正「两次交换 ≠ 恒等」的例子来自更精细的丛（如某些量子群范畴），那里 $\gamma^2 \neq 1$。

**辨析｜易错点：** 并非每个单范畴都是对称的，甚至不是每个都带辫子。而且同一个范畴可以有两个不同的辫结构。判断一个「交换规则」是否合法，必须检查六边形公理——只核对「$\gamma \gamma = 1$」或「$\gamma$ 是自然变换」都不够。

**辫与对称，一张对照表：**

| | 辫单范畴 | 对称单范畴 |
| --- | --- | --- |
| 数据 | $\gamma_{a,b}: a \otimes b \cong b \otimes a$ | 同左 |
| 公理 | 六边形公理 | 六边形 + $\gamma_{b,a}\circ\gamma_{a,b} = 1$ |
| 两次交换 | 不必恒等 | 恒等 |
| 经典来源 | 辫群、纽结、量子群 | 集合、向量空间、超空间 |
| 不变量 | Jones 多项式等 | 通常只需平凡对称 |

## 4 公式解析：结合子的五边形公理

五边形公理是单范畴「一致性」的骨架，写成方程是四个对象上的可结合性：

$$
\alpha_{a,b,c \otimes d} \circ \alpha_{a \otimes b,c,d} =
(a \otimes \alpha_{b,c,d}) \circ \alpha_{a,b \otimes c,d} \circ (\alpha_{a,b,c} \otimes d)
$$

- **第一步，看左边**：从 $((a \otimes b) \otimes c) \otimes d$ 出发，先把内层 $c \otimes d$ 结合，再把 $(a \otimes b)$ 与结果结合——即「从左往右挪括号」。
- **第二步，看右边**：同一对象出发，先结合 $b, c, d$，再结合 $a$ 与 $(b \otimes c)$，最后收尾——即「从右往左挪括号」。
- **第三步，公理的意义**：两条「挪括号」路线结果相同。四个对象的情形一旦成立，五个、六个乃至任意多个对象就都成立（Mac Lane 用归纳证明）。
- **第四步，直觉**：五边形公理 = 「括号怎么加都一样」的归纳基础。它把无穷多条一致性检查压缩成一条。

**五边形为何叫「五边形」。** 把四个对象的所有「括号化」画出来：$((ab)c)d$、$(a(bc))d$、$(ab)(cd)$、$a((bc)d)$、$a(b(cd))$——正好五个，构成一个五边形的顶点，边就是结合子。公理说这个五边形交换，等于说「五种括号化两两可通」。

## 5 张量网络与机器学习

单范畴不是纯数学的奢侈品。把态射画成「节点」，把张量积画成「连线」，就得到**弦图（string diagram）**——量子信息里张量网络的书面语言。<span class="marginnote">张量网络（MPS、PEPS）的收缩计算，正是单范畴里态射复合的图形演算；纠缠、偏迹、量子电路都在这套语言下变得可读。</span>在深度学习里：

一个张量形状 $(d_1, \dots, d_n)$ 就是 $n$ 个向量空间的张量积；
- 神经网络层 = 态射，层组合 = 复合，batch 的并行 = 张量积；
- `einsum` 的每个式子都是一条弦图，其合法性恰由五边形/六边形公理所保证的一致性来背书。<span class="marginnote">理解「一致性定理保证无歧义」，就能解释为什么 `einsum` 里张量索引的重新组合永远可以放心地进行——括号与顺序在一致的单结构下是自由的。</span>

**把 `einsum` 读成弦图。** 一条 `einsum('ij,jk->ik', A, B)` 是两个矩阵的乘积：张量积「并排」、复合「首尾相连」，索引 $j$ 在中间求和——这正是弦图里「公共连线收缩」的记号。五边形 / 六边形公理保证的「括号自由」，让索引重排永远可以放心进行。

## 6 术语速查表

| 术语 | 英文 | 一句解释 |
| --- | --- | --- |
| 单范畴 | monoidal category | 张量积 $\otimes$ + 单位 $I$ + 结合子 / 单位子 |
| 张量积 | tensor product | 横向组合的双函子 |
| 结合子 | associator $\alpha$ | $(a \otimes b) \otimes c \cong a \otimes (b \otimes c)$ |
| 五边形公理 | pentagon axiom | 四对象结合性的一致性 |
| 三角形公理 | triangle axiom | $\lambda$、$\rho$ 与 $\alpha$ 的协调 |
| 一致性定理 | coherence theorem | 任何单范畴等价于严格单范畴 |
| 辫子 | braiding $\gamma$ | $a \otimes b \cong b \otimes a$ |
| 六边形公理 | hexagon axiom | $\gamma$ 与 $\alpha$ 的相容性 |
| 对称单范畴 | symmetric monoidal | $\gamma^2 = 1$ 的单范畴 |
| 严格单范畴 | strict monoidal | 三个同构都是恒等 |
| 超向量空间 | super vector space | 带符号交换的对称单范畴 |
| 弦图 | string diagram | 张量运算的图形演算 |

## 7 小结

- **单范畴** = 张量积 $\otimes$ + 单位 $I$ + 结合子/单位子，满足五边形与三角形一致性。
- **一致性定理**：任何单范畴等价于严格单范畴——括号在计算中可安全省略。
- **辫单范畴**：带满足六边形公理的 $\gamma_{a,b}$；**对称单范畴**：再加 $\gamma^2 = 1$。
- 实例：$(\mathbf{Set}, \times)$、$(\mathbf{Vect}, \otimes)$、$(\mathbf{Ab}, \otimes)$；同一范畴可有多个单结构。
- 张量网络、`einsum`、量子电路都是单范畴的图形语言。

在下一节，我们回到「范畴里能不能做线性代数」这个问题：当每个 $\mathrm{Hom}$ 集都天然带加法与零、双积存在时，就得到**加性范畴与 Abel 范畴**——同调代数与层上同调的全部舞台。
