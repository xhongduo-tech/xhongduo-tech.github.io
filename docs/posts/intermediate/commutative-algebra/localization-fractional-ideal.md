---
title: 局部化与分数理想
date: 2026-08-11
---

# 局部化与分数理想

<div class="epigraph">
<p>数学不是一门小心谨慎的科学……它敢于在直觉上走得更远。</p>
<footer>—— 安德烈 · 韦伊（André Weil）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 交换代数 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从局部化开始

你早就掌握了一种造环操作：从整数 $\mathbb{Z}$ 造有理数 $\mathbb{Q}$——把非零整数全部「放进来做分母」。局部化（localization）就是把这个操作推向一般环：给定环 $A$ 与一个**可乘子集** $S$，造一个新环 $S^{-1}A$，让 $S$ 里的元素全部可逆。名字来自代数几何：只看函数在某点附近的表现，就是「局部」；全局信息由所有局部拼回。<span class="marginnote">「局部与整体的对立统一」是代数几何的核心思想：一个空间 $X$ 上的结构常常由各点邻域决定，而交换代数正是用 $A_{\mathfrak{p}}$（在素理想处局部化）来刻画「点 $\mathfrak{p}$ 附近」。</span>

局部化值得一开始就学透，因为它是交换代数里**出场率最高的构造**：素理想对应（$\operatorname{Spec} S^{-1}A$）、平坦性、支集、相伴素、深度与维数，全都建立在它之上。这一篇讲清楚它的定义、几何含义，并顺手引入**分数理想**——它为下一阶段《离散赋值环与 Dedekind 整环》备好主角。

## 1 从 $\mathbb{Z} \to \mathbb{Q}$ 到一般局部化

$\mathbb{Q}$ 的构造可以抽象成三条规则：分数 $a/s$ 相等当且仅当存在 $t$ 使 $t(as' - a's) = 0$；分子分母按 `(a/s)(b/t) = ab/(st)`、`(a/s) + (b/t) = (at + bs)/(st)` 运算。对一般环，只要分母取出的集合 $S$ 满足**乘法封闭**（$s, t \in S \Rightarrow st \in S$）且含 $1$，同样的规则就给出一个环：

**局部化（localization）**：$S^{-1}A = \{a/s \mid a \in A,\ s \in S\}/\sim$，其中 $a/s \sim a'/s'$ 当且仅当存在 $t \in S$ 使 $t(as' - a's) = 0$。

标准的例子：
- $S = \{1, n, n^2, \dots\}$：$S^{-1}\mathbb{Z} = \mathbb{Z}[1/n]$，含分母为 $n$ 的幂的分数。
- $S = A \setminus \mathfrak{p}$（$\mathfrak{p}$ 是素理想）：记作 $A_{\mathfrak{p}}$，称为**在素理想 $\mathfrak{p}$ 处局部化**。几何上 $A_{\mathfrak{p}}$ 就是「只看 $\mathfrak{p}$ 这个点附近」的环。
- $S = $ 所有非零除子（整环时）：$A \to \operatorname{Frac}(A)$，即分式域。

**重点：局部化有万有性质。** 同态 $f: A \to S^{-1}A$ 把 $S$ 中元素映为可逆元，并且它是最小（万有）的具有此性质的同态：任何把 $S$ 映为可逆元的同态 $g: A \to B$，都唯一穿过 $S^{-1}A$。用图说话——「给 $S$ 造逆」这件事，局部化就是终对象。

**辨析｜易错点：** 局部化与「约分」的直观：$a/s$ 相等的判定引入了 $t$，是因为 $A$ 可能有零因子，$as' - a's$ 可能非零但被 $t$ 零化。整环上没有这个麻烦（取 $t=1$ 即可），所以初学者常忘掉 $t$——但正是这个 $t$ 让局部化在一般环上仍然良定义。

## 2 局部化下的理想：$\operatorname{Spec}$ 的对应

局部化为什么几何上等于「看 $\mathfrak{p}$ 附近」？答案是理想之间精确的对应关系。

**重点：$S^{-1}A$ 的素理想与 $A$ 中「不与 $S$ 相交」的素理想一一对应：**

$$\operatorname{Spec} S^{-1}A \;\cong\; \{\mathfrak{p} \in \operatorname{Spec} A \mid \mathfrak{p} \cap S = \emptyset\}, \qquad \mathfrak{q} \mapsto \mathfrak{q} \cap A.$$

特别地，$A_{\mathfrak{p}}$ 的素理想对应 $A$ 中包含于 $\mathfrak{p}$ 的素理想——所以「$\mathfrak{p}$ 附近」的谱正是所有「包含在 $\mathfrak{p}$ 里」的点，比 $\mathfrak{p}$ 大的点全被局部化「扔出窗外」。<span class="marginnote">把 $\operatorname{Spec}$ 想成一幅地图：局部化 $A_{\mathfrak{p}}$ 是「把镜头拉近到点 $\mathfrak{p}$」；拉近后能看见的只有原来就在 $\mathfrak{p}$ 下方（或等于 $\mathfrak{p}$）的点。Zariski 拓扑会在《零点定理与 Zariski 拓扑》一篇接管这张地图。</span>

**局部性质（local property）**：一个关于环 $A$ 的性质 $P$ 称为局部性质，若「$A$ 有 $P$」当且仅当「对所有素理想 $\mathfrak{p}$，$A_{\mathfrak{p}}$ 有 $P$」。例如「$A$ 是整环」不是局部性质（需所有局部化都是整环且条件加严），但「$A$ 的模是平坦的」是。局部性质是「用局部拼整体」这一思想的形式化。

## 3 局部环与极大理想

在 $\mathfrak{p}$ 处局部化得到的环 $A_{\mathfrak{p}}$ 是**局部环（local ring）**：有唯一的极大理想，即 $\mathfrak{p} A_{\mathfrak{p}}$，它由 $\mathfrak{p}$ 中元素充当分子组成。局部环记法常写作 $(R, \mathfrak{m}, k)$，其中 $k = R/\mathfrak{m}$ 称为**剩余域（residue field）**。<span class="marginnote">「局部环」一词在交换代数里指「唯一极大理想」的环，与拓扑中的局部环名字撞车但含义不同；代数几何里它正是「函数环在某点的茎」。</span>

局部化的巨大价值之一：很多环论问题的答案都藏在其所有局部化中。例如 $\mathfrak{a} = A$ 当且仅当对所有极大理想 $\mathfrak{m}$ 有 $\mathfrak{a}_{\mathfrak{m}} = A_{\mathfrak{m}}$——「不在这处也不在那处消失，就处处都在」。这是「局部判别全局」的范本，后续的支集理论会把它做成精确定理。

## 4 分数理想

设 $A$ 是整环，$K = \operatorname{Frac}(A)$。**分数理想（fractional ideal）**：$K$ 的一个 $A$-子模 $M$，且存在非零 $d \in A$ 使 $dM \subseteq A$（「通分后有界」）。

普通理想 $\mathfrak{a}$ 都是分数理想（取 $d = 1$）。
- $M = \tfrac12 \mathbb{Z} \subset \mathbb{Q}$ 是 $\mathbb{Z}$ 的分数理想：$2M = \mathbb{Z} \subseteq \mathbb{Z}$。
- 分数理想可以相乘：$MN = \{\sum m_i n_i\}$；可以取逆：$M^{-1} = \{x \in K \mid xM \subseteq A\}$。

**重点：全体非零分数理想构成一个群（Dedekind 意义下的理想乘法群），单位元是 $A$。** 这个群的大小是数域重要的不变量——**理想类群（ideal class group）**。理想类群平凡当且仅当 $A$ 是主理想整环，而它是否有限、阶多少，正是代数数论的核心问题之一。<span class="marginnote">数论史名场面：库默尔 1840 年代论证费马大定理的重要情形时，就是在「理想数的类」里转，他证明了分圆域的类数对「正则素数」不整除 $p$ 的情形——这是费马大定理最早的实质性进展。</span>

分数理想引入「逆」的概念后，理想运算第一次有了群结构：普通理想中「乘法的逆元」一般不存在（$(p)$ 的逆是 $\tfrac1p \mathbb{Z}$，不是理想），放宽到分数理想后逆元齐备。下一篇讲 Dedekind 整环时，正是靠这条「每个非零分数理想可逆」的性质，得到理想唯一分解为素理想的乘积。

## 5 公式解析：局部化保持正合性

局部化最重要的代数性质是**正合**：对 $A$-模的短正合列

$$0 \longrightarrow M' \longrightarrow M \longrightarrow M'' \longrightarrow 0,$$

作用 $S^{-1}(-)$ 后仍得短正合列

$$0 \longrightarrow S^{-1}M' \longrightarrow S^{-1}M \longrightarrow S^{-1}M'' \longrightarrow 0.$$

拆解这条公式：

- **第一步，明确对象**：$S^{-1}M = \{m/s \mid m \in M,\ s \in S\}$，把「模的元素」也允许除以 $S$ 中的元素，加法与标量乘法逐坐标继承。
- **第二步，直觉**：局部化是「造分式」，而分式运算只做加减乘除，不动「核/像」结构——就像 $\mathbb{Q}$ 里判断 $m/s = 0$ 等价于 $m$ 在 $M$ 中为零，单射与满射都被保留。
- **第三步，为什么需要 $S^{-1}(-)$ 是函子**：正合性对所有 $S$ 成立，等价于「$S^{-1}A$ 是平坦 $A$-模」；这正是下一阶段《张量积与平坦模》的入口——局部化 = 与 $S^{-1}A$ 做张量积，而张量积一般只保右正合，平坦性补上左正合。

**辨析｜易错点：** 「正合性保持」不等于「像不变」。局部化会改变模（元素多了分母），但**核与像的相对关系**不变。判断 $M = 0$ 时，若每个 $M_{\mathfrak{p}} = 0$，则 $M = 0$——「处处为零才是零」，这是局部判别整体在模上的直接推论。

## 6 小结

- **局部化** $S^{-1}A$ 把可乘子集 $S$ 的元素变成可逆元，$\mathbb{Z} \to \mathbb{Q}$ 是它的原型；整环时取 $S = A\setminus 0$ 得分式域。
- 素理想对应：$\operatorname{Spec} S^{-1}A \cong \{\mathfrak{p} \cap S = \emptyset\}$，局部化 = 看一点附近。
- $A_{\mathfrak{p}}$ 是**局部环**，唯一极大理想 $\mathfrak{p}A_{\mathfrak{p}}$；「局部性质」用所有局部化判别整体。
- **分数理想**允许分子分母都是元素，构成理想乘法群；理想类群平凡 ⇔ PID。

在下一节，我们回到 Noether 环境下的「分解」：把整数分解为素数的故事，推广为理想分解为**准素理想**的交——这是 Emmy Noether 与 Lasker 建立的准素分解理论。
