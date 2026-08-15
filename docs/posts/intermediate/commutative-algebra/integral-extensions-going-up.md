---
title: 整扩张：Going-up 与 Going-down
date: 2026-08-07
---

# 整扩张：Going-up 与 Going-down

<div class="epigraph">
<p>整扩张把「维数守恒」写进环的血管里——维数在扩张中既不增也不减。</p>
<footer>—— I. S. Cohen 与 A. Seidenberg（1946）</footer>
</div>

<div class="article-byline">
<p>第二级 · 交换代数 ｜ Atiyah–Macdonald Ch. 5 ｜ 2026-08-07</p>
</div>

## 为什么从整元开始

数论里「代数整数」是 $\mathbb{Q}$ 中满足首一多项式（整系数）的根。交换代数把这句话抽象成环论概念：**整元（integral element）**。研究「把一个环装进更大的环」，就是**整扩张**——它涵盖代数整数、坐标环的有限扩张、Noether 正规化，是一根贯穿数论与几何的轴线。<span class="marginnote">代数整数的原型：$\mathbb{Z}[\sqrt{d}]$ 是 $\mathbb{Q}(\sqrt{d})$ 中整元全体；高斯整数 $\mathbb{Z}[i]$ 是 $\mathbb{Z}$ 的整闭包。Dedekind 整环（第1篇《离散赋值环》）的「整闭」条件，正是「整闭包 = 自身」。</span>

整扩张最神奇的性质是**维数守恒**：$\dim A = \dim B$（$A \subset B$ 整）。这一篇先立整元与整闭包的基本机制，再讲两个定理——**Going-up**（素理想链可以「抬升」）与 **Going-down**（链还可以「落下来」）——它们共同把「整扩张」变成一张无瑕的 $\operatorname{Spec}$ 地图。

## 1 整元与整闭包

**整元（integral element）**：$x \in B$ 在 $A$ 上整（$A \subseteq B$ 是子环），若存在首一多项式

$$x^n + a_1 x^{n-1} + \cdots + a_n = 0, \qquad a_i \in A.$$

标准例子：$\sqrt[3]{2}$ 在 $\mathbb{Z}$ 上整（$x^3 - 2 = 0$）；$\tfrac12$ 在 $\mathbb{Z}$ 上**不**整（$2x - 1 = 0$ 不是首一）；$x$ 在 $k[x^2]$ 上整（$z^2 - x^2 = 0$ 首一）。**「首一」是关键：首项系数必须是 1，否则局部化、分式域之类会混进非整元。**

**核心对照表：整与非整**

| 元素 | 基环 | 整? | 首一多项式 |
| --- | --- | --- | --- |
| $\sqrt{2}$ | $\mathbb{Z}$ | 是 | $x^2 - 2$ |
| $\sqrt[3]{2}$ | $\mathbb{Z}$ | 是 | $x^3 - 2$ |
| $\tfrac12$ | $\mathbb{Z}$ | 否 | $2x - 1$ 不首一 |
| $x$ | $k[x^2]$ | 是 | $z^2 - x^2$ |
| $\sqrt{-5}$ | $\mathbb{Z}$ | 是 | $x^2 + 5$ |
| $\zeta_5$ | $\mathbb{Z}$ | 是 | 分圆多项式（首一） |

末行的 $\zeta_5$ 是「整」概念的数论原乡：五次单位根满足首一分圆多项式，故是代数整数——第1篇《离散赋值环》里 $\mathcal{O}_K$ 就是「整元全体」。

**整闭包（integral closure）**：$A$ 在 $B$ 中的整闭包 $\bar{A}^{B}$ = $B$ 中在 $A$ 上整的元素全体。$A$ 称为**整闭（integrally closed）**，若其整闭包等于自身（常见语境：在分式域中）。

**重点：整元素构成一个环，且「整性 = 有限生成模块性」。** 下面这条引理是全部理论的发动机：

$$x \text{ 在 } A \text{ 上整} \iff A[x] \text{ 是有限生成 } A\text{-模} \iff A[x] \subseteq B \text{ 含于某有限生成 } A\text{-子模}.$$

证明的核心是行列式技巧：由 $x^n = -(a_1 x^{n-1} + \cdots)$ 反复降幂得 $x^k \in A + Ax + \cdots + Ax^{n-1}$。<span class="marginnote">「整 ⇔ 有限生成模」使整闭包成为环：$x, y$ 整则 $A[x, y]$ 有限生成，故 $x + y, xy$ 也在有限生成 $A$-模里，由引理它们都整。</span>

**辨析｜易错点：** 「整」与「有限」是两回事。$\mathbb{Q}$ 中每个 $q \in \mathbb{Q}$ 在 $\mathbb{Z}$ 上都是「有限次」的吗？不：$q$ 整于 $\mathbb{Z}$ ⇔ $q \in \mathbb{Z}$（有理数中整元就是整数）。「整」对**首一**要求严格，别和「可分/代数」等弱概念混用——代数元允许任意多项式，整元强制首一。

算一个整闭包：$k[x^2, x^3] \subset k[x]$ 的整闭包是 $k[x]$。因为 $x$ 满足首一方程 $z^2 - x^2 = 0$（系数 $x^2 \in k[x^2,x^3]$），故 $x$ 整于 $k[x^2,x^3]$；而整元的全体构成环，恰好填满 $k[x]$。这也解释了「尖端坐标环 $k[x^2,x^3] \cong k[u,v]/(v^2-u^3)$ 非正规」：它的整闭包是 $k[x]$，第1篇《正则局部环》的辨析里见过它。

## 2 Going-up：素理想链的抬升

设 $A \subseteq B$，$B$ 在 $A$ 上整。

**Lying-over 定理**：每个 $\mathfrak{p} \in \operatorname{Spec} A$ 都「躺」在某个 $\mathfrak{q} \in \operatorname{Spec} B$ 之下：$\mathfrak{q} \cap A = \mathfrak{p}$。

**Going-up 定理**：若 $\mathfrak{p}_1 \subseteq \mathfrak{p}_2 \subseteq \cdots \subseteq \mathfrak{p}_n$ 是 $A$ 的素理想链，则存在 $B$ 的素理想链 $\mathfrak{q}_1 \subseteq \cdots \subseteq \mathfrak{q}_n$ 使 $\mathfrak{q}_i \cap A = \mathfrak{p}_i$——**链可以「抬升」。**

**重点：整扩张下，$\operatorname{Spec} B \to \operatorname{Spec} A$ 是满射，且每条闭链都能抬上去。** 几何直觉：整扩张 =「有限厚度的覆盖」，覆盖的纤维有限，任何下层点都能找到上层点压住。<span class="marginnote">Lying-over 的证明典型地用局部化：$A \to B_{\mathfrak{p}}$ 的构造 + Zorn 引理选极大理想。它比初看深刻——「每点有覆盖」正是「$\operatorname{Spec}$ 是满射」的代数事实。</span>

用具体数字看 Lying-over：$A = \mathbb{Z}$，$B = \mathbb{Z}[i]$（高斯整数），$B$ 在 $A$ 上整。素理想 $(5)$：$5 = (2+i)(2-i)$ 在 $B$ 中分裂，故 $(5)$ 被 $(2+i)$、$(2-i)$ 两个素理想覆盖；而 $(3)$ 是惯性素数（在 $\mathbb{Z}[i]$ 中仍为素理想），只被一个素理想覆盖。**Lying-over 只保证「至少一层」，分裂的算术细节全在 $(2+i)(2-i)$ 这类等式里。**

若换成 $(7)$：$7 \equiv 3 \pmod 4$，在 $\mathbb{Z}[i]$ 中仍是素元，$(7)$ 只被一个素理想覆盖——「分裂 / 惯性 / 分歧」三种算术行为，恰好对应覆盖层数的差别，这正是代数数论里「素理想分解」三个词的几何面貌。

**维数守恒定理**：$A \subseteq B$ 整 ⇒ $\dim A = \dim B$。

由 Going-up 得 $\dim B \geq \dim A$；反向用**不相容性（incomparability）**：整扩张里，同一条 $\mathfrak{q} \cap A = \mathfrak{p}$ 之上的素理想两两不可比较（若 $\mathfrak{q}_1 \subsetneq \mathfrak{q}_2$ 且 $\mathfrak{q}_1 \cap A = \mathfrak{q}_2 \cap A$ 则矛盾）——于是链长不增。**整扩张不改变维数，这是整性最优雅的算术果实。**<span class="marginnote">例：$k[x^2, x^3] \subset k[x]$ 整，两边 $\dim = 1$；$\mathbb{Z} \subset \mathcal{O}_K$ 整，$\dim \mathcal{O}_K = 1$——代数整数环永远是一维（Dedekind 整环），不随域扩张改变，这解释了第1篇《离散赋值环》里「数域整数环总是一维」。</span>

这条「维数守恒」有一个直接推论：代数整数环 $\mathcal{O}_K$ 永远一维，无论 $K/\mathbb{Q}$ 扩到多大。这也是整扩张在数论里最日常的用法：Kummer 在 $\mathbb{Z}[\zeta_p]$ 里做因子分解，维数恒为 1，变化的只有理想类群——第1篇《离散赋值环》与《局部化与分数理想》的算术，在这里找到了「维数不变」的地基。

## 3 Going-down：条件更强

Going-up 在一般整扩张中成立；**Going-down**（链往下抬）却需要额外条件。

**Going-down 定理（Cohen–Seidenberg）**：设 $A \subseteq B$ 整，$A$ 是整闭整环，$B$ 是整环，则对 $A$ 的素理想链 $\mathfrak{p}_1 \supseteq \mathfrak{p}_2 \supseteq \cdots$ 与 $\mathfrak{q}_1$（覆盖 $\mathfrak{p}_1$），存在 $B$ 的素理想链 $\mathfrak{q}_1 \supseteq \mathfrak{q}_2 \supseteq \cdots$ 覆盖之——**链可以「落下来」。**

**重点：Going-down 需要「$A$ 整闭」这个几何上极强的条件。** 几何翻译：正规簇的有限覆盖里，「一般点的分支」总能蔓延到更特殊点的分支——这本质是「正规环上主理想的局部化保持某些整除关系」。<span class="marginnote">Going-down 也可以从<strong>平坦</strong>推出（$A \to B$ 忠实平坦 ⇒ Going-down），这正是《张量积与平坦模》里平坦性的几何红利：正规簇上的有限平坦覆盖都满足 Going-down。</span>

**辨析｜易错点：** 整扩张一般**不**满足 Going-down。标准反例：$A = k[x^2, x^3]$（非整闭），$B = k[x]$ 整扩张。取 $\mathfrak{p}_2 = (x^3 - x^2)$？不必记细节，记住结构：Going-up 总是对的，Going-down 需要整闭或平坦——**看到「Going-down」先检查前提是否满足，别默认。** 很多教材习题故意把 Going-up 当成 Going-down 出题，属第一类易错。

Going-down 的「落下」直觉在几何上是「正规簇上主理想的可整除性可局部化」：若 $\mathfrak{p}_1 \supseteq \mathfrak{p}_2$ 且 $\mathfrak{q}_1$ 覆盖 $\mathfrak{p}_1$，则存在 $\mathfrak{q}_2 \subseteq \mathfrak{q}_1$ 覆盖 $\mathfrak{p}_2$。$A$ 整闭（正规）时，「下方点的分支总能蔓延到上方点的分支里」——这正是正规簇上「特殊化」可沿分支下降的代数版本。

## 4 公式解析：Noether 正规化

整扩张最有生产力的应用是 **Noether 正规化引理**：有限生成的 $k$-代数 $A$ 存在 $k$ 上的多项式子环，使 $A$ 是其**有限整扩张**。

**Noether 正规化引理**：设 $k$ 是域，$A = k[x_1, \dots, x_n]$ 有限生成，则存在代数独立的 $y_1, \dots, y_r \in A$ 使 $A$ 在 $B = k[y_1, \dots, y_r]$ 上整。

**第一步，看意义**：任何有限型 $k$-代数都是「多项式环（r 个自由变量）上的有限整扩张」——把任意的代数「压扁」成「有限厚度的多项式层」。
- **第二步，构造**：在 $x_i$ 间有代数关系时，用「一般线性变换 $x_i \mapsto x_i + c_i x_n$」把关系式的首项系数化为常数，逐项剥出 $y_1,\dots,y_r$（核心是 **Nagata 的「换元使首项常化」技巧**）。<span class="marginnote">几何视角：对仿射簇 $V = \operatorname{Spec} A$，正规化给出满射 $V \to k^r$（有限覆盖）——维数定理 $\dim A = r$ 立即成立。这是「维数 = 代数独立元素个数」的最短证明路径。</span>
- **第三步，推论**：$\dim A = r =$ 代数独立的极大个数。配上维数守恒定理，$\dim k[x_1,\dots,x_n]/\mathfrak{a}$ 被正规化一举算清。

**辨析｜易错点：** Noether 正规化里的 $y_i$ 不是 $x_i$ 本身——一般要做线性组合。初学者直接拿 $x_i$ 当 $y_i$，在 $A = k[x,y]/(xy-1)$ 时 $x$ 与 $y$ 都代数独立吗？不，$xy = 1$ 已给出关系，正规化要挑出 $y_1$ 使 $A$ 整于 $k[y_1]$——例如 $y_1 = x + y$。**「代数独立」要验证，不是拿来即用。**

把正规化用到坐标环上：$A = k[x,y]/(y^2-x^3)$，取 $y_1 = x$ 不行（$y^2 = x^3$ 给出关系），需要 $y_1 = y - x^2$ 之类的组合使 $A$ 整于 $k[y_1]$，于是 $\dim A = 1$——与「曲线是一维」的几何直觉吻合。**「维数 = 代数独立元素个数」在正规化后一目了然**，这也是《维数理论》与《正则局部环》里反复用到的落点。

**术语速查表**

| 术语 | 一句话含义 |
| --- | --- |
| 整元 | 首一多项式方程的根 |
| 整闭包 | 整元全体，构成环 |
| 整闭 | 整闭包等于自身 |
| Lying-over | 每个素理想被上层素理想覆盖 |
| Going-up | 素理想链可向上抬升 |
| Going-down | 素理想链可向下延伸（需整闭/平坦） |
| Noether 正规化 | 有限型 $k$-代数是多项式环的有限整扩张 |

## 5 小结

- **整元**：首一多项式根；「整 ⇔ 有限生成模」是发动机；整闭包是环，整闭环 = 闭包自身。
- **Lying-over 与 Going-up**：整扩张的 $\operatorname{Spec}$ 满射且链可抬升；维数守恒 $\dim A = \dim B$。
- **Going-down**：需 $A$ 整闭（或扩张平坦）；「落下链」是正规性/平坦性的红利。
- **Noether 正规化**：有限型 $k$-代数是多项式环的有限整扩张；维数 = 代数独立元素个数。

在下一节，整扩张与正则序列在「局部」合流：**深度与正则序列**——用 Ext、用 Koszul 同调衡量「模有多高」，把 $\operatorname{depth} \leq \dim$ 写成可计算的同调等式。
