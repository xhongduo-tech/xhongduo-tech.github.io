---
title: 伴随函子
date: 2026-08-07
---

# 伴随函子

<div class="epigraph">
<p>伴随函子无处不在。</p>
<footer>—— 桑德斯 · 麦克莱恩（Saunders Mac Lane）</footer>
</div>

<div class="article-byline">
<p>第二级 · 范畴论 ｜ Mac Lane Ch. IV ｜ 2026-08-07</p>
</div>

## 为什么从伴随开始

上一节的极限与余极限已经给出了「普适性质」的语法，但你会发现一个更深的规律：**乘积、自由对象、张量积……这些看似无关的构造，全是成对出现的**。自由群与遗忘、张量积与内态、直积与对角——每一对之间都隔着一座「映射数目的桥」。**伴随函子（adjoint functor）**就是这座桥的正式名字。Mac Lane 的名言「伴随函子无处不在」不是修辞：微分几何里的张量密度、代数里的自由对象、程序语言里的柯里化、机器学习里的对偶优化，背后几乎都能找到一对伴随。<span class="marginnote">机器学习里一个典型例子：特征空间上的「嵌入函子」与「解码函子」经常是一对伴随——自编码器的瓶颈结构恰好对应伴随的「单元—余单元」分解。</span>这一节学完，你会拥有把整个学科「配对重排」的透视能力。

## 1 伴随的三副面孔

设 $F: \mathbf{C} \to \mathbf{D}$、$G: \mathbf{D} \to \mathbf{C}$。称 $F$ 是 $G$ 的**左伴随（left adjoint）**，$G$ 是 $F$ 的**右伴随（right adjoint）**，记作 $F \dashv G$，若以下任一（彼此等价）的条件成立：

- **同态双射**：对任意 $x \in \mathbf{C}$、$y \in \mathbf{D}$，有自然同构
$$\mathrm{Hom}_{\mathbf{D}}(F x, y) \cong \mathrm{Hom}_{\mathbf{C}}(x, G y)$$
- **单元与余单元**：存在自然变换 $\eta: 1_{\mathbf{C}} \Rightarrow G F$ 与 $\varepsilon: F G \Rightarrow 1_{\mathbf{D}}$，满足三角形恒等式。
- **普适箭头**：每个对象 $x$ 都存在一条「泛映射」$x \to G(F x)$，使得任何 $x \to G y$ 都唯一穿过它。

三副面孔各有用途：同态双射最适合计算，单元/余单元最适合推图，普适箭头最贴近「自由构造」的直觉。

**什么时候用哪副面孔？** 下表是快速指引：

| 面孔 | 核心数据 | 最佳用途 |
| --- | --- | --- |
| 同态双射 | $\mathrm{Hom}(Fx, y) \cong \mathrm{Hom}(x, Gy)$ | 计算、计数、构造双射 |
| 单元 / 余单元 | $\eta, \varepsilon$ + 三角恒等式 | 推图、化简复合、编程实现 |
| 普适箭头 | 泛映射 $x \to G(Fx)$ | 自由构造、存在性证明 |

**方向是语义的核心。** $F \dashv G$ 读作「$F$ 是 $G$ 的左伴随」。判断谁在左、谁在右，唯一标准是双射中 $\mathrm{Hom}$ 的位置：左边从 $Fx$ 出发，右边从 $Gy$ 出发，于是 $F$ 在左、$G$ 在右。把方向写反，整套符号都会连锁错位——这是几乎所有伴随习题的失分点。

## 2 单元与余单元、三角恒等式

给定 $F \dashv G$，**单元（unit）** $\eta$ 与**余单元（counit）** $\varepsilon$ 通过双射对恒等态射取像得到：

$$1_{F x} \in \mathrm{Hom}(F x, F x) \xmapsto{\cong} \eta_x: x \to G F x, \qquad 1_{G y} \xmapsto{\cong} \varepsilon_y: F G y \to y$$

它们必须满足**三角形恒等式（triangular identities）**——两条「走一半再折返」的路线都能化简为恒等：

$$G \varepsilon \circ \eta G = 1_G, \qquad \varepsilon F \circ F \eta = 1_F$$

**辨析｜易错点：** 单元与余单元的方向不同：$\eta$ 把对象「塞进 $G F$」，$\varepsilon$ 把对象「从 $F G$ 里取出来」。除非特别情况，它们**不是**同构——这正是「遗忘」与「自由」之间永远隔着一条缝隙的含义。

**算一个单元：自由群。** 取 $F(X) \dashv U$（自由 ⊣ 遗忘），$\eta_X: X \to U(F(X))$ 把每个生成元 $x$ 送去「长度 1 的字 $x$」；$\varepsilon_G: F(U(G)) \to G$ 把「由 $G$ 的元素拼成的自由群里的字」用 $G$ 的乘法约化回去——$x \cdot y$ 约成 $xy$。三角形恒等式 $G\varepsilon \circ \eta G = 1_G$ 在此说的是：先按生成元自由生成、再按乘法约化，等于原封不动。

## 3 伴随的实例

- **自由 ⊣ 遗忘**：自由群函子 $\mathbf{Set} \to \mathbf{Grp}$ 是遗忘函子 $\mathbf{Grp} \to \mathbf{Set}$ 的左伴随。一条从集合 $X$ 到群 $G$ 的平凡映射，等价于从自由群 $F(X)$ 到 $G$ 的一个同态——「生成元的赋值」就是同态。<span class="marginnote">这在机器学习里对应「参数化」：给模型参数 $X$（平凡集合）指定取值，等价于给出整个由参数生成的假设空间到具体模型的同态。</span>
- **乘积 ⊣ 对角**：对角函子 $\Delta: \mathbf{C} \to \mathbf{C} \times \mathbf{C}$ 的右伴随就是乘积 $(-) \times (-)$，左伴随是余积。这是把上节所有极限统一成「伴随的一侧」的钥匙。
- **张量 ⊣ 内态（柯里化）**：在 $\mathbf{Vect}$ 中 $\mathrm{Hom}(A \otimes B, C) \cong \mathrm{Hom}(A, \mathrm{Hom}(B, C))$——正是函数式编程里 `curry` / `uncurry`。**伴随函子在程序语言里就是柯里化**，这是范畴论与编程最漂亮的相遇之一。
- **Galois 连接**：设 $f: X \to Y$ 是偏序集间的单调映射，则直接像 $f_*: \mathcal{P}(X) \to \mathcal{P}(Y)$ 与逆像 $f^*: \mathcal{P}(Y) \to \mathcal{P}(X)$ 满足 $f_* \dashv f^*$。这是序论、抽象解释与形式概念分析里最常见的伴随——「左伴随保并、右伴随保交」在此退回成集合论的常识。

**一张伴随速查表。** 把数学里反复出现的伴随对并排看，方向感立刻清晰：

| 左伴随 $F$ | 右伴随 $G$ | 范畴 $\mathbf{C} \to \mathbf{D}$ |
| --- | --- | --- |
| 自由群 $F$ | 遗忘 $U$ | $\mathbf{Set} \to \mathbf{Grp}$ |
| 自由模 $F$ | 遗忘 $U$ | $\mathbf{Set} \to \mathbf{Mod}_R$ |
| 乘积 $(-) \times (-)$ | 对角 $\Delta$ | $\mathbf{C} \times \mathbf{C} \to \mathbf{C}$ |
| 张量 $-\otimes B$ | 内态 $\mathrm{Hom}(B, -)$ | $\mathbf{Vect} \to \mathbf{Vect}$ |
| 层化 $a$ | 遗忘 $\mathbf{Sh} \to \mathbf{Psh}$ | $\mathbf{Psh} \to \mathbf{Sh}$ |

注意「自由」总在左、「遗忘」总在右——这是记忆伴随方向最稳的锚。

为什么这些「对」如此之多？因为伴随的两侧互为「最优解」：$G$ 是「最便宜地把 $\mathbf{D}$ 塞进 $\mathbf{C}$ 的方式」，$F$ 是「最经济地把 $\mathbf{C}$ 铺到 $\mathbf{D}$ 的方式」。这个「最优」来自普适性质——伴随的每一侧都在普适意义下不可再改，这正是它无处不在的根源。

## 4 公式解析：柯里化双射

以集合为例，写出伴随的「同态双射」并在计算层面拆开：

$$
\mathrm{Hom}_{\mathbf{Set}}(A \times B, C) \cong \mathrm{Hom}_{\mathbf{Set}}(A, C^B)
$$

- **第一步，理解两边的对象**：左边是「接受一对 $(a, b)$ 返回 $c$」的函数 $f: A \times B \to C$；右边是「先接受 $a$，返回一个『接受 $b$ 返回 $c$』的函数」的 $g: A \to C^B$。
- **第二步，来向（curry）**：给定 $f$，定义 $g(a)(b) := f(a, b)$——把二元函数拆成「逐点返回一元函数」。
- **第三步，去向（uncurry）**：给定 $g$，定义 $f(a, b) := g(a)(b)$。
- **第四步，为什么是伴随**：两步互逆、且对 $A, B, C$ 的一切变化自然——这正是「同态双射」的定义。函数式语言里 `(a -> b -> c)` 与 `(a, b) -> c` 的类型等价，本质就是这条双射的编译期实例。

**数值算例：把伴随当「字典」数一遍。** 取 $A = \{a_1, a_2\}$、$B = \{b_1, b_2\}$、$C = \{0, 1\}$。左边 $A \times B$ 有 4 个元素，函数 $f: A \times B \to C$ 共有 $2^4 = 16$ 个；右边 $C^B$ 是「$B \to C$」的函数集，有 $2^2 = 4$ 个元素，于是「$A \to C^B$」的函数数是 $4^2 = 16$。两边都数出 16 不是巧合：柯里化双射给出 16 个函数之间的**一一对应**——$g(a_1)$ 就是「把 $b \mapsto f(a_1, b)$」的那张查表，$g(a_2)$ 同理。<span class="marginnote">数一数即知 $\lvert C^{A \times B} \rvert = \lvert (C^B)^A \rvert$，即 $2^{2 \times 2} = (2^2)^2$——指数记号的「换底」恰好就是伴随。</span>

## 5 伴随保持极限

伴随对普适构造有极强的「搬运」能力，这是它最重要的定理之一：

**右伴随保持极限，左伴随保持余极限。** 即若 $F \dashv G$，则 $G$ 把 $\mathbf{D}$ 中的任意极限锥映成 $\mathbf{C}$ 中的极限锥，$F$ 同理搬运余极限。<span class="marginnote">这立刻解释了许多「显然」现象：遗忘函子有左伴随，所以它保持乘积——群直积的底集就是底集之积；对角函子有左右伴随，所以乘积与余积天然存在。</span>反过来说，**一个保持极限的函子何时才真的有左伴随？** 这就是下一节《伴随函子定理》要回答的问题。

**数值确认「右伴随保极限」。** 遗忘函子 $\mathbf{Grp} \to \mathbf{Set}$ 保持乘积：$\mathbb{Z}/2 \times \mathbb{Z}/2$ 的底集就是 $\{0,1\} \times \{0,1\}$，恰有 4 个元素。而张量积 $-\otimes B$ 是左伴随（$-\otimes B \dashv \mathrm{Hom}(B,-)$），故保持**余**极限：$(A_1 \oplus A_2) \otimes B \cong (A_1 \otimes B) \oplus (A_2 \otimes B)$——张量积对直和的分配律，正是「左伴随保余极限」在 $\mathbf{Vect}$ 里的脸。

**辨析｜易错点：** 「左伴随保余极限、右伴随保极限」——两个方向刚好相反，是经典错题来源。记忆口诀：**左就是左，管拼不管取；右就是右，管取不管拼**。另外，左伴随与右伴随**不是**「一个在左、一个在上」的随意安排：$F \dashv G$ 的方向决定了一切符号。

## 7 术语速查表

| 术语 | 英文 | 一句解释 |
| --- | --- | --- |
| 左伴随 | left adjoint | 双射 $\mathrm{Hom}(Fx, y) \cong \mathrm{Hom}(x, Gy)$ 中的 $F$ |
| 右伴随 | right adjoint | 与 $F$ 配对、方向相反的一侧 $G$ |
| 单元 | unit $\eta$ | $1_{\mathbf{C}} \Rightarrow GF$，把对象塞进复合 |
| 余单元 | counit $\varepsilon$ | $FG \Rightarrow 1_{\mathbf{D}}$，把对象取出来 |
| 三角形恒等式 | triangular identities | $\eta$、$\varepsilon$ 满足的两条化简律 |
| 普适箭头 | universal arrow | 每个 $x$ 的泛映射 $x \to GFx$ |
| 柯里化 | currying | 二元函数 ⇔ 返回函数的函数 |

## 8 小结

- **伴随** $F \dashv G$ 由同态双射 $\mathrm{Hom}(Fx, y) \cong \mathrm{Hom}(x, Gy)$ 刻画，等价于单元/余单元与普适箭头。
- **单元 $\eta$ 与余单元 $\varepsilon$** 满足三角形恒等式，一般不是同构。
- 实例三连：自由 ⊣ 遗忘、对角 ⊣ 乘积/余积、张量 ⊣ 内态（柯里化）。
- **右伴随保极限，左伴随保余极限**——普适构造沿伴随自动搬运。

在下一节，我们将追问逆命题：给定一个「行为良好」的函子，它在什么条件下一定有左伴随或右伴随？——这就是**伴随函子定理**，也是完备范畴理论的高潮之一。
