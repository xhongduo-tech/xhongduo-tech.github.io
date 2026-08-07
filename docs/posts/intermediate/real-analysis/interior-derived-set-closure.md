---
title: 开核、导集与闭包
date: 2026-08-07
---

# 开核、导集与闭包

<div class="epigraph">
<p>闭包是集合的完成：把一切极限都请进门，世界才完整。</p>
<footer>—— 库拉托夫斯基（Kazimierz Kuratowski）</footer>
</div>

<div class="article-byline">
<p>第二级 · 实变函数与测度论 ｜ 周民强《实变函数论》§2.4 ｜ 2026-08-07</p>
</div>

## 为什么从开核、导集、闭包开始

前三节我们有了三个「点」概念（内点、聚点、边界点）与两类「集」概念（开集、闭集）。本节把它们装配成三个**算子**：内部 $E^\circ$、导集 $E'$、闭包 $\overline{E}$。算子比概念高级之处在于：它们是**对每个集合都执行一遍的机械操作**，把「含于 $E$ 的最大开集」「含 $E$ 的最小闭集」这样的极值问题变成可计算的对象。

这三个算子回答了同一个问题的三个侧面：**$E$ 离「完整」还缺什么？** 内部补的是「漏洞」（把非内点删掉），闭包补的是「极限」（把聚点收进来），导集则给出「极限的来源」。它们之间的关系式 $\overline{E}=E\cup E'=E\cup\partial E$ 是本节的核心，也是后面测度逼近、可测性论证的常备工具。<span class="marginnote">这三个算子满足一套漂亮的代数规律（Kuratowski 闭包公理：$\overline{\varnothing}=\varnothing$、$E\subset\overline E$、$\overline{\overline E}=\overline E$、$\overline{E\cup F}=\overline E\cup\overline F$）。<strong>反过来，任何满足这四条公理的运算都可以定义出整个拓扑结构</strong>——拓扑学可以在「开集」与「闭包算子」两套公理间自由切换。</span>

## 1 开核：最大的开子集

**定义（开核 / 内部）**：集合 $E$ 的全体内点组成的集合称为 $E$ 的**开核（interior）**，记作 $E^\circ$。

$$E^\circ=\{x\in\mathbb{R}^n:\exists\,\delta>0,\ B(x,\delta)\subset E\}$$

**定理（开核是最大开子集）**：$E^\circ$ 是开集，且是**含于 $E$ 的最大开集**——若 $G$ 是含于 $E$ 的任一开集，则 $G\subset E^\circ$。

证明分两步。先证 $E^\circ$ 开：取 $x\in E^\circ$，则存在 $B(x,\delta)\subset E$；对任意 $y\in B(x,\delta)$，取 $\delta'=\delta-\|y-x\|>0$，有 $B(y,\delta')\subset B(x,\delta)\subset E$，故 $y\in E^\circ$。这推出 $B(x,\delta)\subset E^\circ$，即 $x$ 是 $E^\circ$ 的内点。再证最大性：$G\subset E$ 且 $G$ 开，则 $G$ 的每点都是 $E$ 的内点，故 $G\subset E^\circ$。

**重点：$E^\circ$ 同时回答了「内部」与「最大开子集」两个问题，它们是同一个对象。** 因此「$E$ 开 $\iff E=E^\circ$」；而一旦 $E$ 含有孤立的尖点（如 $E=\{0\}\cup(1,2)$），开核自动把这些尖点删去：$\{0\}^\circ=\varnothing$。

## 2 导集：聚点们

**定义（导集）**：集合 $E$ 的全体聚点组成的集合称为 $E$ 的**导集（derived set）**，记作 $E'$。

$$E'=\{x\in\mathbb{R}^n:\forall\,\delta>0,\ B(x,\delta)\cap(E\setminus\{x\})\neq\varnothing\}$$

导集是「$E$ 的极限所在」的集合。注意它与内部、闭包不同：$E'\not\subset E$ 不一定成立。例：$E=\{1,\tfrac12,\tfrac13,\dots\}$，则 $E'=\{0\}$，而 $0\notin E$。

导集最重要的性质是**二次导集给出聚点结构的层次**：$E''=(E')'$ 可能比 $E'$ 更小。康托尔-本迪克森定理（Cantor–Bendixson theorem）说，$\mathbb{R}$ 上的闭集 $F$ 可以分解为 $F=P\cup C$，其中 $P$ 是完备集（$P'=P$），$C$ 至多可数。**「反复取导集」是分析集合论深度的尺子**，康托尔当年正是用它研究三角级数收敛集时发明了集合论。<span class="marginnote">导集运算可以反复进行：$F\supset F'\supset F''\supset\cdots$。用序数可以继续往后取「第 $\omega$ 次导集」，这催生了超限归纳法。康托尔研究傅里叶级数唯一性时，正是这类「极限的极限」把他带进了无穷序数的世界。</span>

## 3 闭包：最小闭超集

**定义（闭包）**：设 $E\subset\mathbb{R}^n$。记

$$\overline{E}=E\cup E'$$

称 $\overline{E}$ 为 $E$ 的**闭包（closure）**。即闭包 = 集合本身 + 它的全部聚点。

闭包还有一种等价描述，在实际使用中往往更方便：**$\overline{E}$ 是包含 $E$ 的最小闭集**。

**定理**：$\overline{E}$ 是闭集；若 $F$ 闭且 $E\subset F$，则 $\overline{E}\subset F$。

证明：$\overline{E}$ 闭需要 $(\overline{E})'\subset\overline{E}$。取 $x\in(\overline{E})'$，可证 $x\in E'$（若 $x$ 的每个邻域都含 $\overline{E}$ 中异于 $x$ 的点，这些点若非 $E$ 的点则逼近 $E$ 的聚点，最终仍推出 $x\in E'$）。于是 $(\overline{E})'=E'\subset\overline{E}$，闭性得证；最小性由 $\overline{E}=E\cup E'$ 与「$F\supset E$ 闭 ⇒ $F'\subset F$ 且 $E\subset F$ ⇒ $\overline{E}\subset F$」立即得到。

**闭包的三张脸**，按使用场景选择：

$$\overline{E}=E\cup E'=E\cup\partial E=\bigcap\{F\ \text{闭}:E\subset F\}$$

第三条等式尤其深刻：**闭包是所有闭超集的交**，把「最小闭超集」的极值性质写得明明白白。$E=(0,1)$ 时，$\overline{E}=[0,1]$，与「把开区间补上端点」的直觉完全一致。

**辨析｜易错点：闭包与「加边界」是同义反复，但不可误记为「加内部」。** $\overline{E}$ 只补聚点，不补任何孤立点——$E=\{0\}\cup(1,2)$ 的闭包是 $\{0\}\cup[1,2]$，孤立点 $0$ 原地不动。而 $\partial E=\{0\}\cup\{1\}\cup\{2\}$ 包含三个点，但 $0$ 已被 $E$ 含有，闭包无需再补。<span class="marginnote">闭包、边界、内部的数值关系：$\overline{E}=E^\circ\cup\partial E$ 恒成立，且 $E^\circ\cap\partial E=\varnothing$。于是「闭包 = 内部 + 边界」是一层不重叠的划分——对区间 $[0,1)$，内部 $(0,1)$ 加上边界 $\{0,1\}$ 恰好拼成 $[0,1]$。</span>

## 4 公式解析：三个算子的极值本质

把三个算子并排写成极值形式，这是本节最值得背的一张表：

$$E^\circ=\bigcup\{G\ \text{开}:G\subset E\},\qquad \overline{E}=\bigcap\{F\ \text{闭}:E\subset F\},\qquad \partial E=\overline{E}\cap\overline{E^c}$$

- **第一步，读开核的「并」**：$E^\circ$ 是所有含于 $E$ 的开集的并。因为「最大开子集」唯一，这个并恰好等于开核本身——**「并」求最大，天然地筛选出极值**。
- **第二步，读闭包的「交」**：$\overline E$ 是所有含 $E$ 的闭集的交。**「交」求最小**，与开核的「并」形成完美的对偶——这正是「开」与「闭」互补关系的算子层面。
- **第三步，读边界的「双闭包之交」**：$\partial E=\overline E\cap\overline{E^c}$。$x\in\partial E$ 意味着 $x$ 同时被 $E$ 与 $E^c$ 的极限所夹——「两侧都有点」精确翻译为「同时属于两侧闭包」。

这三个式子一起回答了「一个集合如何被它的开近似与闭近似夹住」：$E^\circ\subset E\subset\overline E$，而 $\partial E$ 恰好是夹层两端的缝隙。**测度论下一章的逼近定理（用开集从外逼近、用闭集从内逼近）正是在这个夹层上展开的。**

## 5 小结

- **开核** $E^\circ$：含于 $E$ 的最大开集，$E$ 开 $\iff E=E^\circ$。
- **导集** $E'$：全体聚点；$E'\subset E$ 当且仅当 $E$ 闭；反复取导集给出集合的层次（Cantor–Bendixson）。
- **闭包** $\overline E=E\cup E'=E\cup\partial E$：包含 $E$ 的最小闭集，闭超集的交。
- **对偶**：$E^\circ$ 用「并」求最大开子集，$\overline E$ 用「交」求最小闭超集，$\partial E=\overline E\cap\overline{E^c}$。
- **夹层结构**：$E^\circ\subset E\subset\overline E$，为下一章测度逼近定理铺路。

在下一节，我们将构造实变函数论的第一个「奇怪集合」——**康托尔集**，并证明它同时拥有「零长度」与「不可数」两种看似矛盾的属性。
