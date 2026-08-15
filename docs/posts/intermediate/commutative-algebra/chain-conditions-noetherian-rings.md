---
title: 链条件与 Noether 环、Artin 环
date: 2026-08-07
---

# 链条件与 Noether 环、Artin 环

<div class="epigraph">
<p>先取合适的最大元素，再用升链条件。</p>
<footer>—— 埃米 · 诺特（Emmy Noether），哥廷根讲学时的口头禅</footer>
</div>

<div class="article-byline">
<p>第二级 · 交换代数 ｜ Atiyah–Macdonald Ch. 6-8 ｜ 2026-08-07</p>
</div>

## 为什么从链条件开始

上一节准素分解的关键一步，是「Noether 环的升链稳定」逼出分解存在。这一节我们停下脚步，把这台发动机本身拆开看：什么叫**链条件**，为什么它会成为整个学科的分水岭。**Noether 环**大概是交换代数里最重要的一类环——多项式环、代数簇的坐标环、代数整数环全是它；**Artin 环**则是它的「对偶」，短得可爱却结构惊人地简单。<span class="marginnote">「Noether 环」的名字来自 Emmy Noether；「Artin 环」来自 Emil Artin。两个概念由 Max Zorn 的引理——没错，就是 Zorn 引理的那位——在 1930 年代理清成「链条件」的统一表述。</span>

核心问题只有一句：**什么时候一个环/模可以「有限地」看透？** 链条件给出的答案，让 Hilbert 基定理、准素分解、维数理论全部有了共同的立足点。

## 1 升链条件与 Noether 模

**升链条件（ACC）**：一串子模 $M_1 \subseteq M_2 \subseteq M_3 \subseteq \cdots$ 从某一项起稳定，即存在 $n$ 使 $M_n = M_{n+1} = M_{n+2} = \cdots$。

满足 ACC 的模叫 **Noether 模**。链条件还有两条等价刻画，交换代数里反复换着用：

**重点：下列三条等价（对 $A$-模 $M$）：**

1. **ACC**：任何子模升链稳定；
2. **极大条件**：$M$ 的任何非空子模族都有极大元；
3. **有限生成**：$M$ 的每个子模都有限生成。

三者中「有限生成」最常用也最直观——Noether 模就是「子模全都能由有限个元素生成」的模。<span class="marginnote">等价性的证明套路固定：「极大元 + 反证」：若无极大元，用选择公理造一条严格升链。这是 Zorn 引理在环论里的第一次正经出场。</span>

**Noether 环**：作为自身的模是 Noether 的环，即其全体理想满足 ACC。注意区别——Noether 环的每个子模（理想）有限生成，但环本身作为环可以有无限多元素。

**辨析｜易错点：** 「Noether 环的子环」不一定是 Noether 环——子环可以大而无当（如 $k[x]$ 的子环 $k[x^2, x^3]$，生成元不满但有限）。真正继承 Noether 性的是**商环**与**局部化**：$A$ Noether 则 $A/\mathfrak{a}$ 与 $S^{-1}A$ 都 Noether。判断「类 Noether 环」先想继承关系，别默认子环也继承。

**核心对照表：Noether 与 Artin**

| 性质 | Noether | Artin |
| --- | --- | --- |
| 链条件 | 升链稳定（ACC） | 降链稳定（DCC） |
| 等价刻画 | 极大条件；子模全有限生成 | 极小条件；素理想全是极大 |
| 维度 | 一般任意 | 自动为 0（Hopkins–Levitzki） |
| 结构 | 可作准素分解 | 有限个局部 Artin 环的直积 |
| 例子 | $\mathbb{Z}$、$k[x_1,\dots,x_n]$ | $k[x]/(x^n)$、$\mathbb{Z}/n\mathbb{Z}$ |
| 商环 / 局部化继承 | 是 | 是 |

竖着看这表：**Noether 负责「能有限分解」，Artin 负责「结构特别小」。** Artin 环自动是 Noether 的，反之不然——$k[x]$ 永远不 Artin，因为降链 $(x) \supsetneq (x^2) \supsetneq (x^3) \supsetneq \cdots$ 永不停。

**辨析｜易错点（续）：** 别把「子环不继承 Noether 性」误当成「有限生成代数不 Noether」。$k$ 上有限生成代数必 Noether（Hilbert 基定理），但「子环」与「有限生成代数」是两码事——$k[x^2,x^3]$ 是 $k[x]$ 的子环，同时也是 $k$ 上有限生成代数，所以它 Noether；真正不 Noether 的子环要更刁钻：$k[x, xy, xy^2, xy^3, \dots] \subset k[x,y]$ 中，理想 $(x, xy, xy^2, \dots)$ 需要无限多个生成元。**「Noether 性」永远先问是「作为谁的子环/商环/代数」。**

## 2 Hilbert 基定理

交换代数最早也最著名的 Noether 结果，是 Hilbert 1890 年证明的**基定理**：

**Hilbert 基定理（Hilbert Basis Theorem）**：若 $A$ 是 Noether 环，则多项式环 $A[x_1, \dots, x_n]$ 也是 Noether 环。

关键在 $n=1$，用归纳即可：$A[x] = A[x_1]\cdots[x_n]$。<span class="marginnote">Hilbert 证明它是为了给不变量理论奠基，结论吓坏当时同行：无穷维对象居然总能有限生成。魏尔斯特拉斯曾讽刺「这和神学一样不可信」，结果 30 年后 Hilbert 用它统一了代数不变量理论。</span>

证明思想（$A[x]$ 情形）：取理想 $\mathfrak{a} \subseteq A[x]$，把每个多项式按首项系数分类，首项系数的理想 $L_i$ 是 $A$ 中升链，由 ACC 稳定；再把首项系数有限生成，回代系数与次数界，拼出 $\mathfrak{a}$ 的有限生成集。**「对次数取上界、对系数用环的 Noether 性」**——这个双射证明是基定理的标准模板。

**重点：多项式环保 Noether，是「有限生成」最可靠的来源。** 代数几何里坐标环 $k[x_1,\dots,x_n]/\mathfrak{a}$ 全是 Noether 的，这保证了上一节准素分解在其上处处可用。

对 $k[x]$ 实际数一遍：理想 $\mathfrak{a} = (x^2 - 1, x^3 - 1)$。首项系数理想先是 $(1) = k$（因为 $x^3-1$ 首项系数 $1$），升链 $(\text{首项系数})$ 在一步内稳定；于是 $1 \in k$ 给出……更直观地，$x^3 - 1 - x(x^2 - 1) = x - 1$，故 $(x^2-1, x^3-1) = (x-1)$——两个生成元归约成一个。基定理不负责找出这个归约（那是 Gröbner 基的活），只保证「一定有有限的生成元集」。

## 3 Artin 环：对称的另一极

## 3 Artin 环：对称的另一极

**降链条件（DCC）**：任何降链 $M_1 \supseteq M_2 \supseteq \cdots$ 稳定。满足 DCC 的模叫 **Artin 模**，DCC 的环叫 **Artin 环**。

Artin 环看起来只是 Noether 的对偶，实则「矮得多」：**Artin 环都是 Noether 的，且 Krull 维数为 0**（每个素理想都是极大的）——这个惊人结论叫 **Hopkins–Levitzki 定理**。<span class="marginnote">名字里藏着尴尬史：DCC 在 1930 年代被称作「descending chain condition」，有人把满足它的环戏称为「finite」环，直到 Emmy Noether 的学生们厘清后才冠名 Artin。</span>

**重点：Artin 环的结构定理**：Artin 环是有限个局部 Artin 环的直积，

$$A \cong A_{\mathfrak{m}_1} \times \cdots \times A_{\mathfrak{m}_r}$$

其中 $\mathfrak{m}_i$ 取遍 $A$ 的（有限个）极大理想。<span class="marginnote">这与「$\mathbb{Z}/n\mathbb{Z} \cong \prod \mathbb{Z}/p_i^{e_i}\mathbb{Z}$」完全同构——中国剩余定理在 Artin 环上的终极版。Artin 环的可理解性正是「分成小块、每块都很小」。</span>

典型例子：有限域 $k$ 上的环 $k[x]/(x^n)$ 是 Artin 的（也 Noether），它有唯一的素理想 $(x)$，维数 0。更一般地，域上的有限维代数都是 Artin 环。

**Noether / Artin 环的典型例子一览：**

| 环 | Noether? | Artin? | 说明 |
| --- | --- | --- | --- |
| $\mathbb{Z}$ | 是 | 否 | 升链稳定，降链 $(p) \supset (p^2) \supset \cdots$ 不停 |
| $k[x]$ | 是 | 否 | 同 $\mathbb{Z}$，维数 1 |
| $k[x]/(x^n)$ | 是 | 是 | 有限维 $k$-代数 |
| $k[x_1, x_2, \dots]$ | 否 | 否 | 无穷多个变量，理想升链不停 |
| $\mathbb{Z}/12\mathbb{Z}$ | 是 | 是 | 有限环必 Artin |
| $k[x]_{(x)}$ | 是 | 否 | 维数 1，非 Artin |

注意最后一行的 $k[x]_{(x)}$：它是局部环、Noether、一维，但**不是** Artin——Artin 要求维数 0。这提醒我们「局部 + Noether + 一维」是 DVR（第1篇《离散赋值环》），与 Artin 环是两个不同的极小世界。

**重点：Artin 环的结构定理证明思想是「逐点局部化 + CRT 拼回」。** $\operatorname{Spec} A$ 在 Artin 环上是有限离散点集（全是极大理想），对每个 $\mathfrak{m}$ 取 $A_{\mathfrak{m}}$ 得局部 Artin 环，再由中国剩余定理（第1篇《理想、模与环同态》）粘回直积。$k[x,y]/(x^2, y)$ 就是现成例子：有限维 $k$-代数，唯一极大理想 $(\bar{x}, \bar{y})$，幂零。

## 4 公式解析：降链为什么逼出「有限」

用 Hopkins–Levitzki 里最精彩的一步来拆公式。设 $R$ 是 Artin 局部环，$\mathfrak{m}$ 是其唯一极大理想。**要证 $\mathfrak{m}$ 幂零**，即存在 $n$ 使 $\mathfrak{m}^n = 0$。

- **第一步，构造降链**：$\mathfrak{m} \supseteq \mathfrak{m}^2 \supseteq \mathfrak{m}^3 \supseteq \cdots$。Artin 环给 DCC，故存在 $n$ 使 $\mathfrak{m}^n = \mathfrak{m}^{n+1} = \cdots$，设 $\mathfrak{m}^n = \mathfrak{c}$ 是「稳定尾」。
- **第二步，用 Nakayama 的思想**：若 $\mathfrak{c} \neq 0$，考虑满足 $\mathfrak{m}\mathfrak{c} \neq \mathfrak{c}$ 的所有理想中取极小者（用 Noether/极大条件）。可以取到 $\mathfrak{c}' = \mathfrak{c}$，但 $\mathfrak{m}\mathfrak{c} = \mathfrak{c}$ 与「取极小」矛盾——因为 $\mathfrak{m}\mathfrak{c} \subsetneq \mathfrak{c}$。
- **第三步，结论**：矛盾说明 $\mathfrak{c} = 0$，即 $\mathfrak{m}^n = 0$。于是剩余域 $k = R/\mathfrak{m}$ 的每个有限生成模都被 $\mathfrak{m}$ 的幂杀死——Artin 局部环「处处在有限步内归零」。

**辨析｜易错点：** 别把 Artin 环误当「有限环」。$k[x]/(x^n)$ 在 $k$ 无限时是无限环，却仍是 Artin 的。Artin 的本质不是「元素少」，而是「结构链短」：所有素理想都是极大的，且根幂零。

用 $\mathbb{Z}/n\mathbb{Z}$ 把 Artin 结构定理走到底。$n = 12$：$\mathbb{Z}/12\mathbb{Z} \cong \mathbb{Z}/4\mathbb{Z} \times \mathbb{Z}/3\mathbb{Z}$——两个局部 Artin 环的直积，各自唯一极大理想分别是 $(2)$ 与 $(3)$ 的像，都是幂零的。有限环都是 Artin 环，所以「$\mathbb{Z}/n\mathbb{Z}$ 由中国剩余定理拆成素幂环之积」正是 Artin 结构定理在 $\mathbb{Z}$ 上的具体形态——同一句话，第1篇看到的是数论，这里看到的是结构。

**术语速查表**

| 术语 | 一句话含义 |
| --- | --- |
| ACC | 任何子模升链稳定 |
| DCC | 任何子模降链稳定 |
| Noether 环 | ACC 成立；子模全有限生成 |
| Artin 环 | DCC 成立；自动 Noether、维数 0 |
| Hilbert 基定理 | $A$ Noether ⇒ $A[x_1,\dots,x_n]$ Noether |
| 局部 Artin 环 | 唯一极大理想且极大理想幂零 |

## 5 小结

- **升链条件（ACC）** ⇔ 极大条件 ⇔ 子模全有限生成，Noether 模/环由此定义。
- **Hilbert 基定理**：$A$ Noether ⇒ $A[x_1,\dots,x_n]$ Noether；有限生成坐标环全靠它。
- **Noether 性在商环、局部化中继承**，但**子环不一定继承**。
- **Artin 环**：DCC 成立，自动 Noether、维数 0、是有限个局部 Artin 环的直积；Hopkins–Levitzki 说明它「有限步归零」。

在下一节，链条件将开出几何之花：**Hilbert 零点定理**告诉你多项式的零点与理想如何互相确定，**Zariski 拓扑**则把 $\operatorname{Spec}$ 变成一张真正的地图。
