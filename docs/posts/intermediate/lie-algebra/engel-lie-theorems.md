---
title: Engel 定理与 Lie 定理
date: 2026-08-11
---

# Engel 定理与 Lie 定理

<div class="epigraph">
<p>任何一个足够对称的结构，其表现出的特征往往归因于它内在的一个简单构件。</p>
<footer>—— 赫尔曼 · 外尔（Hermann Weyl）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 李代数与李群 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么需要两条结构定理

上一节的算子刻画给了我们两个条件：「每个 $\operatorname{ad}x$ 幂零」与「$\operatorname{ad}$ 的导出序列归零」。但它们只停留在抽象条件上——我们还需要知道：**这样的李代数长什么样？** Engel 定理与 Lie 定理回答了这个问题。前者断言：所有伴随算子幂零 ⟺ 存在一个基使全体元素都是**严格上三角**。后者断言：可解 ⟺ 存在基使全体元素都是**上三角**。把「上三角」落实为矩阵形式，两条定理就完成了从「代数条件」到「具体形状」的翻译。<span class="marginnote">这正是线性代数中「同时三角化」的推广：可解与幂零对应着整个李代数的表示可以同时上三角化/严格上三角化，与抽象代数中「表示论与结构论互相印照」的思路一脉相承。</span>

## 1 Engel 定理及其意义

**Engel 定理（Engel's Theorem）**：设 $V$ 是有限维向量空间，$\mathfrak{g} \subseteq \mathfrak{gl}(V)$ 是李代数。若**每个** $x \in \mathfrak{g}$ 都是 $V$ 上的幂零算子，则存在 $V$ 的基，使 $\mathfrak{g}$ 中所有矩阵同时为**严格上三角**。<span class="marginnote">关键在于「同时」二字：幂零性是一个个算子单独成立的事实，Engel 定理保证它们能共享同一个三角化基。这是从「逐点性质」推出「整体结构」的典范。</span>

对李代数本身（取伴随表示 $\operatorname{ad}: L \to \mathfrak{gl}(L)$）有一个对偶版本：

**Engel 定理（李代数版本）**：若 $L$ 中每个元素 $x$ 都满足 $\operatorname{ad}x$ 是幂零算子（指数统一），则 $L$ 是幂零李代数。

这两条是「幂零 ⇔ 所有 $\operatorname{ad}x$ 幂零」的严格化。证明的核心是 Engel 的**关键引理**：

> 若 $\phi: L \to \mathfrak{gl}(V)$ 是表示，且每个 $\phi(x)$ 幂零，则存在非零 $v \in V$ 被所有 $\phi(x)$ 零化：$\phi(x)v = 0$ 对所有 $x \in L$。

这引理保证了一个公共零向量，逐层取商就逐行「剥出」严格上三角结构。<span class="marginnote">严格上三角矩阵的幂零性指数不超过维数 $n$，因此「每个算子幂零且指数统一」在 $\mathfrak{gl}(V)$ 中是自动的——Engel 定理实际上讲的是：若每个元素幂零，则整个李代数幂零（对伴随表示）。</span>

## 2 Lie 定理：可解代数的三角化

**Lie 定理（Lie's Theorem）**：设 $\mathbb{F}$ 是代数闭域（最常用 $\mathbb{C}$），$\mathfrak{g} \subseteq \mathfrak{gl}(V)$ 是可解李代数，则存在 $V$ 的基使 $\mathfrak{g}$ 中所有矩阵**同时为上三角**。

**对偶版本**：有限维复数域可解李代数存在基，使所有 $\operatorname{ad}x$ 同时上三角化；即可解李代数可嵌入 $\mathfrak{t}(n,\mathbb{C})$（对上三角李代数）的同构像。

**关键结论**：Lie 定理推出——可解李代数的一个子代数、从而其所有元素在某个基下是上三角的；特别地，**可解李代数的表示都可由一维表示过滤**。用表示论的语言：可解群/代数的不可约表示只能是一维的。<span class="marginnote">这正是 Schur 引理与 Lie 定理结合的经典结论：可解李代数的所有不可约表示都是一维的。这在量子力学中意味着可解对称性只能给出单态的量子数，无法提供简并。</span>

**辨析｜易错点：** Lie 定理要求域是代数闭的。在 $\mathbb{R}$ 上可解李代数未必可三角化——旋转矩阵 $\begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$ 生成的二维实李代数可解，但没有实的公共特征向量。复数闭包是关键的「加上虚数单位」才能完成三角化。

## 3 从证明看思想：Engel 与 Lie 的公共钥匙

两条定理的证明共享一个「归纳骨架」：

**Engel 路径**：先用关键引理找到公共零向量 $v_1$；把 $V$ 商掉 $\mathbb{F}v_1$，在商空间上重复，得到递降旗 $V \supset V_1 \supset V_2 \supset \cdots \supset 0$，其中 $\mathfrak{g} V_i \subseteq V_{i+1}$。这样选基就得到严格上三角。
- **Lie 路径**：先证明可解李代数在**每个**不变子空间上有公共特征向量（利用「特征空间在可解作用下的不变性」这一引理）；再逐层取商、归纳，得到上三角旗。

两条路线的差别在于引理：Engel 用「幂零算子有公共零化向量」，Lie 用「可解作用下有公共特征向量」。<span class="marginnote">「旗（flag）」——一条全嵌套的线性子空间链——正是「同时三角化」的几何表述：存在旗稳定链即同时上三角。这个观点在 Schubert 簇、旗流形中有深远延伸（第三级《代数几何》）。</span>

**核心要点**（两条定理的对比表格）：

| 定理 | 条件 | 结论 | 域要求 |
| --- | --- | --- | --- |
| Engel | 所有 $\operatorname{ad}x$ 幂零 | 存在基使严格上三角 | 任意域 |
| Lie | 李代数可解 | 存在基使上三角 | 代数闭 |

## 4 推论与应用

两个直接推论值得单独列出：

**推论 A（可解代数的单性）**：若 $L$ 可解且 $\mathbb{C}$ 上，则 $L$ 有理想链 $0 = L_0 \subset L_1 \subset \cdots \subset L_n = L$ 使每个商 $L_i / L_{i-1}$ 一维。这正是「过滤」的语言。

**推论 B（Engel 的纯算子版本）**：一个由幂零算子生成的李代数必幂零——这直接说明可解/幂零的李代数理论其实都是「幂零算子的理论」的衍生物。<span class="marginnote">在数值分析里，幂零矩阵对应 Jordan 块全为 0 的矩阵；可解/幂零结构的「逐层消去」与 Jordan 标准形的「逐块简化」遥相呼应。</span>

## 5 公式解析：Engel 定理的构造性读法

以 $\mathfrak{sl}(2,\mathbb{C})$ 中的严格上三角子代数为例，理解 Engel 的结论。取

$$\mathfrak{n} = \left\{ \begin{pmatrix} 0 & a \\ 0 & 0 \end{pmatrix} : a \in \mathbb{C} \right\} = \mathbb{C} e$$

这是一个一维幂零李代数，$\operatorname{ad}e$ 在基 $\{e, h, f\}$ 下作用为 $\operatorname{ad}e(e)=0, \operatorname{ad}e(h) = -2e, \operatorname{ad}e(f) = -h$，矩阵形式

$$\operatorname{ad}e = \begin{pmatrix} 0 & -2 & 0 \\ 0 & 0 & -1 \\ 0 & 0 & 0 \end{pmatrix}$$

三步拆解：

- **第一步，验幂零**：$(\operatorname{ad}e)^2(f) = \operatorname{ad}e(-h) = 2e$，$(\operatorname{ad}e)^3(f) = 0$，且 $(\operatorname{ad}e)^2$ 作用在 $e, h$ 上已为零，故 $(\operatorname{ad}e)^3 = 0$——幂零。
- **第二步，找公共零化向量**：满足 $\operatorname{ad}e(v)=0$ 的是 $\mathbb{C}e$，取 $v_1 = e$ 作第一行。
- **第三步，商掉再看**：在 $L/\mathbb{C}e$ 上 $\operatorname{ad}e$ 仍是幂零（这是关键引理重演的微观样本），继续剥出第二行，最终得到严格上三角——正是一个旗稳定链 $L \supset \mathbb{C}e \supset 0$。

这演示了 Engel 定理「找到零化向量 → 商 → 再找」的机械流程，也正是下一节起 $\mathfrak{sl}(2)$ 表示论中反复出现的技巧原型。

## 6 小结

- **Engel 定理**：所有 $\operatorname{ad}x$ 幂零 ⟹ 存在基使全体严格上三角；等价于「幂零 ⇔ 幂零算子族」。
- **Lie 定理**：$\mathbb{C}$ 上可解李代数可同时上三角化；可解代数的不可约表示必一维。
- 两条定理证明共享「旗」归纳骨架，区别仅在公共特征向量/零化向量的引理；Lie 定理需要**代数闭**域。
- 推论：可解代数有「一维商」的过滤链；幂零算子族生成的李代数必幂零。
- 上三角/严格上三角是「可解/幂零」在具体基下的化身，为后续根空间分解提供了模板。

在下一节，我们将离开「退化」的一端，转向另一极端——**半单李代数与 Killing 型**，并证明不可约表示的完全可约性。
