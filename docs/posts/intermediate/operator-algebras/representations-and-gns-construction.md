---
title: 表示与 GNS 构造
date: 2026-08-07
---

# 表示与 GNS 构造

<div class="epigraph">
<p>数学是理性的音乐。</p>
<footer>—— 詹姆斯 · 约瑟夫 · 西尔维斯特（James Joseph Sylvester）</footer>
</div>

<div class="article-byline">
<p>第二级 · 算子代数 ｜ Murphy《C\*-Algebras and Operator Theory》第2章 ｜ 2026-08-07</p>
</div>

## 为什么从 GNS 构造开始

上一节说过，每个态都「长成」向量期望值的样子。把这个断言严格化并倒过来——**从任意态出发，精确造出一个带循环向量的表示**——就是 **GNS 构造**（Gelfand–Naimark–Segal）。它不只是证明工具：它是一台**从代数到 Hilbert 空间**的机器，把每个态 $\varphi$ 翻译成一个「舞台」$(\pi_\varphi,\mathcal{H}_\varphi,\xi_\varphi)$。

GNS 构造是 C\* 代数理论真正的中心舞台。表示论的全部问题——哪些表示等价、哪些不可约、表示如何分类——都以它为起点；量子场论的真空表示、非交换几何的谱三元组、以及第 21 篇 von Neumann 代数的表示生成，全部站在它的肩膀上。这一节我们要把它看穿。

## 1 表示与等价：舞台上的角色

回顾：**表示**是 $\ast$-同态 $\pi:A\to B(\mathcal{H})$。表示论的第一批问题都围绕「舞台之间的变换」：

**子表示（subrepresentation）**：$\mathcal{H}'$ 是 $\mathcal{H}$ 的闭子空间且对 $\pi(A)$ 不变，则 $\pi|_{\mathcal{H}'}$ 是子表示。

**不可约表示（irreducible representation）**：没有非平凡闭不变子空间的表示（即 $\pi(A)$ 在 $B(\mathcal{H})$ 中的交换子只有标量 $\mathbb{C}1$）。不可约 = 「最小的舞台」。

**酉等价（unitary equivalence）**：$\pi_1,\pi_2$ 酉等价，若存在酉 $U:\mathcal{H}_1\to\mathcal{H}_2$ 使 $U\pi_1(a)=\pi_2(a)U$ 对所有 $a$。酉等价的表示是「同一个舞台的不同座次」。

**Schur 引理**：$\pi$ 不可约当且仅当 $\pi(A)'=\mathbb{C}1$（交换子只有标量）；两个不可约表示酉等价当且仅当它们之间存在非零交织算子。这是表示论的第一条公理，也是物理里「超选择规则」的代数表达。<span class="marginnote">Schur 引理把「不可约」翻译成「交换子平凡」：不可约表示中能跟所有算子都交换的只有常数。量子力学里，一个表示若不可约，则不存在与所有可观测量都交换的非平凡守恒量——这就是为何不可约表示对应「纯」物理系统。</span>

## 2 GNS 构造：从态到表示的四步

设 $\varphi$ 是 $A$ 上的态。

**第一步，半内积**：在 $A$ 上定义 $\langle a,b\rangle_\varphi=\varphi(b^*a)$。它是半内积（可能退化）：$\langle a,a\rangle=\varphi(a^*a)\ge0$。

**第二步，商空间**：$N_\varphi=\{a:\varphi(a^*a)=0\}$ 是左理想（$N_\varphi$ 对左乘封闭），作商 $\mathcal{H}_0=A/N_\varphi$，内积正定，$\mathcal{H}_\varphi=\overline{\mathcal{H}_0}$ 完备化。

**第三步，左乘表示**：$\pi_\varphi(a)(b+N_\varphi)=(ab)+N_\varphi$，良定义（因 $N_\varphi$ 是左理想），且 $\pi_\varphi$ 是 $\ast$-表示。

**第四步，循环向量**：$\xi_\varphi=1+N_\varphi$，满足 $\pi_\varphi(A)\xi_\varphi$ 稠密于 $\mathcal{H}_\varphi$——$\xi_\varphi$ 是**循环向量**，且 $\varphi(a)=\langle\pi_\varphi(a)\xi_\varphi,\xi_\varphi\rangle$。<span class="marginnote">循环向量是「舞台的聚光灯」：$\xi_\varphi$ 通过表示张出整个 Hilbert 空间，所以 $\varphi$ 的信息被完整保留。物理里 $\xi_\varphi$ 就是「真空/基态」——真空不是没有粒子，而是「循环向量」。</span>

**定理（GNS 存在且唯一）**：对每个态 $\varphi$，存在循环表示 $(\pi_\varphi,\mathcal{H}_\varphi,\xi_\varphi)$ 使 $\varphi(a)=\langle\pi_\varphi(a)\xi_\varphi,\xi_\varphi\rangle$；若 $(\pi,\mathcal{H},\xi)$ 是另一个这样的循环表示，则 $(\pi,\mathcal{H},\xi)$ 与 $(\pi_\varphi,\mathcal{H}_\varphi,\xi_\varphi)$ **酉等价**（存在酉 $U$ 且 $U\xi_\varphi=\xi$）。**GNS 表示在同构意义下是唯一的**。

## 3 纯态与不可约表示

GNS 构造的一个最美妙的推论，把「态的几何」与「表示的代数」焊死：

**定理**：态 $\varphi$ 是**纯态**当且仅当 $\pi_\varphi$ 是**不可约表示**。

证明方向之一的直觉：若 $\mathcal{H}_\varphi=\mathcal{H}_1\oplus\mathcal{H}_2$ 是 $\pi_\varphi(A)$ 不变分解，则 $\varphi$ 能写成两个态的平均（分别用 $\xi_\varphi$ 在 $\mathcal{H}_1,\mathcal{H}_2$ 的分量定义），矛盾于纯性；反过来用 Schur 引理。<span class="marginnote">这条对应是整座理论的枢纽：<strong>不可约表示 = 纯态 = 物理上的「纯态」</strong>。量子系统的一个纯态（波函数）对应代数上的一个纯态、对应表示论里的一个不可约表示——三个世界，同一个概念。后面对因子分类（第 23 篇）里的「因子态」，就是这条线的 von Neumann 版。</span>

**推论（分离性精确版）**：对 $a\neq0$，存在**不可约表示** $\pi$ 与向量 $\xi$ 使 $\pi(a)\xi\neq0$；对 $a\neq b$ 存在态分离它们。所有不可约表示的直和（通用表示的一个子集）已经忠实。

## 4 公式解析：$\varphi(a)=\langle\pi_\varphi(a)\xi_\varphi,\xi_\varphi\rangle$

$$
\varphi(a) = \bigl\langle \pi_\varphi(a)\,\xi_\varphi,\ \xi_\varphi \bigr\rangle
$$

- **第一步，看左端**：$\varphi$ 是「抽象」的态——只对元素 $a$ 给数，不知道 Hilbert 空间。
- **第二步，看右端**：$\pi_\varphi$ 是「具体」的表示——$a$ 变成算子 $\pi_\varphi(a)$，$\xi_\varphi$ 是单位向量。右端是向量态 $\omega_{\xi_\varphi}$ 在表示上的拉回。
- **第三步，看等式为什么成立**：按定义 $\langle\pi_\varphi(a)\xi_\varphi,\xi_\varphi\rangle=\langle a\cdot 1,1\rangle_\varphi=\varphi(1^*a)=\varphi(a)$。符号主义在这里终于兑现：**态就是「对单位元做作用再取内积」**。
- **第四步，看它的意义**：等式把所有「线性阅读代数」的方式全部实现为「向量期望值」。它把态理论（分析）翻译成表示论（代数），也把量子力学的「期望值 = $\langle\xi,A\xi\rangle$」公理翻译成 C\* 语言。GNS 定理的意义在于：**每个抽象态都有一个具体的物理实现**，且实现唯一（酉等价）。

## 5 表示论的应用：从代数到量子场论

**例 1（通用表示）**：$\pi_u=\bigoplus_\varphi\pi_\varphi$（对所有态直和）是忠实的（第 9 篇）。若取**可数**个态且 $A$ 可分，$\mathcal{H}_u$ 可分。

**例 2（忠实态）**：$A$ 可分时存在**忠实态** $\varphi$（可数个分离点态平均），于是单个 GNS 表示已经忠实——**可分 C\* 代数用一个表示就能忠实实现**。<span class="marginnote">这是「一个舞台演全场」的定理：对可分代数，一个聪明的态就够。物理上它对应「存在一个真空态，其 GNS 表示忠实描述整个代数」。量子场论里不同真空（不同 GNS 表示）给出不等价的理论——正是表示等价性的物理威力。</span>

**例 3（不可约表示与谱）**：$a$ 正规时，$\lambda\in\sigma(a)$ 当且仅当存在不可约表示 $\pi$ 使 $\lambda$ 是 $\pi(a)$ 的近似特征值。谱 = 全体不可约表示上的「特征值云」——这是谱的表示论刻画，与第 3 篇的算子刻画、第 8 篇的 Gelfand 刻画三足鼎立。

**辨析｜易错点：**GNS 表示 $\pi_\varphi$ **未必单射**。$\ker\pi_\varphi=\{a:\varphi(b^*a)=0\ \forall b\}$ 非零时，表示「看不见」某些元素。只有取遍所有态（通用表示）或取忠实态（可分情形），忠实性才到手。所以「GNS = 忠实」是误区；正确的说法是「通用表示忠实」。

## 6 例：GNS 构造全流程

把 GNS 构造在具体例子上完整走一遍，四个步骤就不再抽象。

**例：$A=C(\mathbb{T})$，态 $\varphi=\delta_1$（在 1 处求值）**。半内积 $\langle f,g\rangle=f(1)\overline{g(1)}$，零空间 $N=\{f:f(1)=0\}$，商空间 $\cong\mathbb{C}$，$\mathcal{H}_\varphi=\mathbb{C}$。表示 $\pi_\varphi(f)=f(1)$，循环向量 $\xi_\varphi=1$。GNS 把「点」变成「一维表示」。

**例：$A=C(\mathbb{T})$，态 $\varphi=\mu$（Lebesgue 测度）**。半内积 $\langle f,g\rangle=\int f\overline g\,dm$，$N=\{0\}$，$\mathcal{H}_\varphi=L^2(\mathbb{T})$，表示是乘法算子，循环向量 $\xi_\varphi=1$。GNS 把「测度」变成「$L^2$ 空间」。

**例：$A=M_2$，$\varphi=\mathrm{Tr}(\rho\,\cdot)$**。若 $\rho=\mathrm{diag}(1,0)$，$N=\{X:\rho^{1/2}X=0\}$，$\mathcal{H}_\varphi\cong\mathbb{C}^2$，表示是恒等，循环向量是 $\rho^{1/2}$ 的像。GNS 忠实复现「密度矩阵的支撑」。

**第三步的验证**：$\pi_\varphi(a)(b+N)=(ab)+N$ 良定义，因为 $N$ 是左理想——若 $b\in N$，$\varphi((ab)^*ab)=\varphi(b^*a^*ab)\le\|a\|^2\varphi(b^*b)=0$，故 $ab\in N$。左理想性是关键。

**唯一性直觉**：带循环向量的表示在酉等价下唯一——「一个态只有一个舞台」。物理上这保证「真空态只有一个（模等价）」。

## 7 延伸：循环向量与真空态

循环向量（cyclic vector）是 GNS 构造的「聚光灯」，它的意义远超技术细节。

**循环的定义**：$\xi$ 循环 ⟺ $\pi(A)\xi$ 稠密 ⟺ $\xi$ 通过表示「看见」整个 Hilbert 空间。

**真空态的 GNS**：量子场论里真空态 $\omega_0$ 给出循环向量 $\xi_0$（真空）。「真空不是空无一物，而是循环向量」——它张出整个物理空间。

**分离向量**：$\xi$ 分离 ⟺ $a\mapsto\pi(a)\xi$ 单射 ⟺ 「不同元素给出不同向量」。循环与分离是一对互相对偶的概念（对交换子也成立），von Neumann 代数的忠实正常态给出分离向量（第 22 篇）。

**从循环到整个空间**：表示 $\pi$ 若非循环，可分解为循环子表示的直和（取 $\pi(A)x$ 的闭包）。「一切表示由循环表示拼成」——循环表示是表示论的「原子砖块」。

**物理意义**：真空、基态、参考态，都是「选一个循环向量」；不同的选择给出酉等价的表示。GNS 说：物理上「选态 = 选舞台」。

## 8 延伸：表示论的三大定理

GNS 之后，表示论的三条「宪法」把整个理论钉稳。

**第一条：GNS 存在且唯一**（本篇）：态 ↔ 带循环向量表示，双向对应，酉等价下唯一。

**第二条：纯态 ⟺ 不可约**（第 10 篇）：不可约表示 = 纯态的 GNS。表示论的「原子」被态空间的「极点」完整刻画。

**第三条：通用表示忠实**（第 9 篇）：$\pi_u=\bigoplus_\varphi\pi_\varphi$ 是忠实的。所有表示「同时上演」时，任何元素都被看见——C\* 代数 = 具体算子代数的保证。

**推论（表示的直积分）**：一般表示可沿中心分解为不可约表示的直积分（第 24 篇）。原子（不可约）→ 分子（直积分）→ 整体，表示论的分层完成。

**应用（调和分析）**：$C^*(G)$ 的表示 = $G$ 的酉表示（第 16 篇）；不可约表示的分类 = 群表示论的核心。GNS 把群表示论收纳进 C\* 代数表示论。

**一句话总结**：GNS 三大定理 = 「态造表示、纯态造原子、直和造全体」——表示论从此是一门完整的科学。

## 9 小结

- **表示与等价**：子表示、不可约表示、酉等价、Schur 引理（不可约 ⟺ 交换子平凡）。
- **GNS 构造**：态 $\varphi$ → 半内积 → 商空间 → 左乘表示 + 循环向量 $\xi_\varphi$，四步建起 $(\pi_\varphi,\mathcal{H}_\varphi,\xi_\varphi)$。
- **唯一性**：带循环向量的表示酉唯一；**纯态 ⟺ 不可约表示**。
- **等式** $\varphi(a)=\langle\pi_\varphi(a)\xi_\varphi,\xi_\varphi\rangle$