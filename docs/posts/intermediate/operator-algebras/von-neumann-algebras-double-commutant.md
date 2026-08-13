---
title: von Neumann 代数与双交换子定理
date: 2026-08-07
---

# von Neumann 代数与双交换子定理

<div class="epigraph">
<p>每个伟大的发现都始于一个令人困惑的观察。</p>
<footer>—— 约翰 · 冯 · 诺依曼（John von Neumann）</footer>
</div>

<div class="article-byline">
<p>第二级 · 算子代数 ｜ Kadison & Ringrose《Fundamentals of the Theory of Operator Algebras》第8章 ｜ 2026-08-07</p>
</div>

## 为什么从双交换子定理开始

C\* 代数是「范数闭」的算子代数，它们已经足够深刻。但量子力学的数学（冯·诺依曼 1930 年代建立）需要一种更「大」的闭性：**弱拓扑下的闭性**。由此诞生的对象叫 **von Neumann 代数**——它要求代数包含「所有可从上逼近的算子」，而它的定义竟能由一个纯代数的条件等价刻画，这就是本节的**双交换子定理（von Neumann bicommutant theorem）**。

为什么双交换子定理如此重要？因为它把**拓扑**（弱闭）与**代数**（交换子）缝合在一起：一个 $\ast$-子代数 $\mathcal{M}\subset B(\mathcal{H})$ 是弱闭的，当且仅当 $\mathcal{M}=\mathcal{M}''$（双交换子等于自己）。于是「闭性」变成「生成元与谁交换」的可计算问题。von Neumann 代数理论的一切——因子、类型、约化——都从这条定理的地基上长出来，而谱定理（第 13 篇）也在这里找到最终归宿。

## 1 von Neumann 代数：弱闭的算子代数

**von Neumann 代数（von Neumann algebra）**：$B(\mathcal{H})$ 中包含恒等算子、且对**强算子拓扑（SOT）**（或弱算子拓扑 WOT）闭的 $\ast$-子代数。

三条说明：

- **为何用弱拓扑**：范数闭的 C\* 代数「颗粒太粗」，单位球上弱收敛的算子列不一定范数收敛；物理里谱测度、投影的极限都是强/弱意义下的，只有弱闭才能装下它们。
- **两种等价闭性**：SOT 闭 ⟺ WOT 闭 ⟺ $\ast$-强闭（对 $\ast$ 运算也不怕），三者互相等价（因 $\ast$ 在单位球上连续）。
- **含幺是定义的一部分**：von Neumann 代数必含 $I$（这是它与一般 C\* 代数的关键区别之一）。<span class="marginnote">「von Neumann 代数」有时也叫「$W^*$-代数」；严格说，$W^*$-代数指「某个 von Neumann 代数的抽象同构类」。代数上 $W^*$ = 有预对偶的 C\* 代数；具体上，它总可表示成某个 $B(\mathcal{H})$ 的弱闭子代数。</span>

**例（最基本的 von Neumann 代数）**：$B(\mathcal{H})$ 本身；交换子 $\mathcal{M}'$（第 24 篇）；$L^\infty(X,\mu)$（作为 $L^2(X)$ 上的乘法算子代数）；以及第 13 篇谱定理的主角——单个自伴算子生成的 $W^*(T)$。每个都是弱闭。

## 2 交换子：与一切可交换的居民

**交换子（commutant）**：$\mathcal{S}'=\{T\in B(\mathcal{H}):TS=ST\ \forall S\in\mathcal{S}\}$，即「与 $\mathcal{S}$ 中每个算子都可交换」的算子全体。**双交换子** $\mathcal{S}''=(\mathcal{S}')'$。

交换子自动是弱闭的 $\ast$-子代数（交换性是闭条件），且 $S\subset S''$，$S'''=S'$。**交换子把「交换性」当成一个对象来研究**——von Neumann 代数的全部结构都编码在它的交换子代数里。<span class="marginnote">物理直觉：$\mathcal{M}'$ 是「与 $\mathcal{M}$ 中所有可观测量都交换的算子」，即「$\mathcal{M}$ 的守恒量」。量子力学里，$\mathcal{M}'$ 的非平凡元对应「超选择规则」——不可约表示 $\pi$ 的交换子只有标量（Schur 引理，第 11 篇），于是「无守恒量」正是「$\mathcal{M}$ 不可约」。双交换子定理让这条物理直觉变得精确可算。</span>

**例**：$\mathcal{M}=B(\mathcal{H})$ 时 $\mathcal{M}'=\mathbb{C}1$（中心平凡），$\mathcal{M}''=B(\mathcal{H})$。$S=\{T\}$（单个正规算子）时，$S''=W^*(T)$（第 13 篇：$T$ 生成的 von Neumann 代数 = $T$ 的所有 Borel 函数）。

## 3 双交换子定理

**定理（von Neumann 双交换子定理）**：设 $\mathcal{M}\subset B(\mathcal{H})$ 是含幺 $\ast$-子代数。则下列等价：

1. $\mathcal{M}=\mathcal{M}''$（双交换子等于自己）；
2. $\mathcal{M}$ 是弱闭的（von Neumann 代数）；
3. $\mathcal{M}$ 是强闭的。

**辨析｜易错点：**定理要求 $\mathcal{M}$ **含单位元**。不含 $I$ 的 $\ast$-子代数（如紧算子 $\mathcal{K}(\mathcal{H})$）的双交换子可以是 $B(\mathcal{H})$，却本身不弱闭。$\mathcal{K}(\mathcal{H})$ 不满足 $\mathcal{M}=\mathcal{M}''$，正因它缺单位——**单位元的存在，让「交换子」与「闭性」终于互通**。<span class="marginnote">证明的三步是：先用 $\mathcal{M}'\subset\mathcal{M}''$ 与闭性给出 $\mathcal{M}''\subset\overline{\mathcal{M}}^{\mathrm{SOT}}$ 方向的「稠密 + 交换性」论证，再用 Kaplansky 密度定理（第 22 篇）把强闭子代数的球内的点「拉回」$\mathcal{M}$ 本身，最后靠单位元的存在完成「$\mathcal{M}\supset\overline{\mathcal{M}}^{\mathrm{SOT}}$」的反向包含。每一步都微妙，环环相扣。</span>

**推论（代数 = 几何）**：von Neumann 代数的「代数身份」（交换子）完全由「拓扑身份」（弱闭）决定。于是**研究 von Neumann 代数 = 研究交换子的结构**——这是第 23 篇因子分类、第 24 篇约化理论的出发点。

## 4 公式解析：$\mathcal{M}=\mathcal{M}''$

$$
\mathcal{M}'' = (\mathcal{M}')', \qquad \mathcal{M}\ \text{von Neumann} \iff \mathcal{M}=\mathcal{M}''
$$

- **第一步，看左端**：$\mathcal{M}''$ 是「与 $\mathcal{M}'$ 可交换的一切算子」。因为 $\mathcal{M}\subset\mathcal{M}''$ 恒真（$\mathcal{M}$ 的元素当然与 $\mathcal{M}'$ 的元素可交换），问题只在反向包含：**$\mathcal{M}''$ 里的算子是否都被 $\mathcal{M}$ 包含**。
- **第二步，看证明的枢纽——有限秩投影**：对任意 $\mathcal{M}''$ 中的 $T$ 与向量 $\xi$，要证明 $T\xi\in\overline{\mathcal{M}\xi}$。取到 $\overline{\mathcal{M}\xi}$ 的正交投影 $P$，$P\in\mathcal{M}'$（因 $\mathcal{M}\xi$ 是 $\mathcal{M}$ 不变子空间），于是 $T$ 与 $P$ 交换，$T\xi=TP\xi=PT\xi\in\overline{\mathcal{M}\xi}$。**「先投影到轨道，再用交换性」**——这就是定理的代数核心。
- **第三步，看单位元为何必需**：上一步只给出 $T\xi\in\overline{\mathcal{M}\xi}$（稠密性），要升级成 $T\in\overline{\mathcal{M}}^{\mathrm{SOT}}$（弱闭）需要 $T$ 在球内被 $\mathcal{M}$ 强逼近，这要靠 Kaplansky 密度定理；而 Kaplansky 密度定理的前提正是「含幺」。
- **第四步，看意义**：等式 $\mathcal{M}=\mathcal{M}''$ 是「自反性」的终极形态——**von Neumann 代数恰好是那些「被自己的交换子重新确定」的子代数**。它把「弱闭」这个看似分析的条件，翻译成「双交换子」这个看似代数的条件，从此拓扑与分析问题都能用代数方法处理。

## 5 从谱定理到 von Neumann 代数

双交换子定理 + 谱定理（第 13 篇）给出一条黄金链：对正规算子 $T$，$W^*(T)=S''$（$S=\{T\}$）是交换 von Neumann 代数；反过来，**每个交换 von Neumann 代数都「来自」某个正规算子**（或一族）。于是：

**定理（交换 von Neumann 代数）**：$\mathcal{M}\subset B(\mathcal{H})$ 交换 von Neumann 代数 ⟺ $\mathcal{M}$ 弱闭交换 ⟺ 存在谱测度使 $\mathcal{M}$ 是它的像（的弱闭包）。交换 von Neumann 代数 ≅ $L^\infty(X,\mu)$（谱测度分解）——**谱定理在 von Neumann 层面 = 交换 von Neumann 代数的结构定理**。<span class="marginnote">第 13 篇谱定理的最终形态在这里：单个算子的对角化，其实是「它生成的交换 von Neumann 代数 ≅ $L^\infty$」的特例。von Neumann 代数理论把「单个算子的谱分解」升级为「整个代数的分解」，这正是第 24 篇约化（直接积分）的伏笔。</span>

**应用（表示论的 von Neumann 视角）**：$A$ 是 C\* 代数，$\pi$ 是表示，则 $\pi(A)''$ 是 von Neumann 代数——**每个表示都自然产生一个 von Neumann 代数**。GNS 构造（第 11 篇）的 $\pi_\varphi(A)''$ 承载着「态 $\varphi$ 的全部谱信息」，量子场论里「真空表示对应的 von Neumann 代数」正是这套语言的产物。

**辨析｜易错点：**不要把「von Neumann 代数」与「$W^*$-代数」用混。$W^*$-代数是可以**抽象**定义的（有预对偶的 C\* 代数），von Neumann 代数默认是 $B(\mathcal{H})$ 内的**具体**子代数。虽然每个 $W^*$-代数都有忠实表示成为 von Neumann 代数，但「表示」与「内在结构」之分（尤其类型、因子分类不依赖表示）在第 23 篇会格外重要。

## 6 例：交换子与双交换子的计算

把交换子在具体例子里算一遍，双交换子定理就不再神秘。

**$S=\{T\}$（单个正规算子）**：$S''=W^*(T)$——$T$ 生成的 von Neumann 代数 = $T$ 的一切 Borel 函数（第 13 篇）。这是双交换子定理与谱定理的接口。

**$S=\{S\}$（单个移位）**：$S''=B(\ell^2)$（移位是不可约的：无非平凡不变子空间……更精确地，$S'=\mathbb{C}1$，故 $S''=B(\ell^2)$）。「一个移位张出整个 $B(\mathcal{H})$」。

**$S=B(\mathcal{H})$**：$S'=\mathbb{C}1$，$S''=B(\mathcal{H})$。中心平凡——$B(\mathcal{H})$ 是因子（第 23 篇）。

**$S=\mathcal{K}(\mathcal{H})$（紧算子）**：$\mathcal{K}''=B(\mathcal{H})$（$\mathcal{K}$ 强闭包是 $B$），但 $\mathcal{K}$ 本身不弱闭——它缺单位元。双交换子定理对它「失效」的活例子。

**$S=\{P\}$（单个投影）**：$P''=\{aP+b(1-P):a,b\in\mathbb{C}\}$（交换 von Neumann 代数）。「$P$ 的 Borel 函数」= 把 $P$ 当开关的一切组合。

**一句话总结**：交换子是「与给定算子可交换的居民」；双交换子把它们「重新收编」，恰好得到弱闭包。

## 7 延伸：双交换子定理的证明脉络

把证明的「骨架」拆出来，定理就不再是魔法。

**核心引理（轨道稠密）**：对 $T\in\mathcal{M}''$ 与 $x\in\mathcal{H}$，$Tx\in\overline{\mathcal{M}x}$。取 $P$ 为向 $\overline{\mathcal{M}x}$ 的投影，$P\in\mathcal{M}'$，于是 $TPx=PTx$ 即 $Tx=P(Tx)\in\overline{\mathcal{M}x}$——「投影到轨道再用交换性」。

**从稠密到强闭**：$T\in\overline{\mathcal{M}}^{\mathrm{SOT}}$ 需要「$T$ 被 $\mathcal{M}$ 强逼近」；轨道稠密只给「$Tx$ 被 $\mathcal{M}x$ 逼近」。两者差一步，这一步由「有限组合 + 连续性」补齐。

**Kaplansky 密度定理的登场**：把「强逼近」升级为「单位球内的强逼近」——这是第 22 篇的内容。$\mathcal{M}$ 含幺是前提。

**反向包含**：$\mathcal{M}\subset\overline{\mathcal{M}}^{\mathrm{SOT}}$ 平凡；$\mathcal{M}''\supset\overline{\mathcal{M}}^{\mathrm{SOT}}$ 由「交换子自动弱闭 + $\mathcal{M}\subset\mathcal{M}''$」得到。两头一夹，$\mathcal{M}=\mathcal{M}''$。

**一句话总结**：双交换子定理的证明 = 「轨道稠密 + Kaplansky 密度」两记重拳，把弱闭性与交换子结构打成一体。

## 8 延伸：von Neumann 代数的地位

von Neumann 代数在数学物理中的位置，值得从高处看一眼。

**量子力学的代数公理**：von Neumann 1930 年代为量子力学建立代数公理化：可观测量 = 自伴算子、态 = 正常态、时间演化 = 自同构群。von Neumann 代数正是这套公理的「代数骨架」。

**与 C\* 代数的分工**：C\* 代数描述「代数结构」（表示无关），von Neumann 代数描述「表示后的闭包」（拓扑结构）。物理上，C\* 代数给「可观测量代数」，von Neumann 代数给「该表示下的全部极限」。

**超有限 von Neumann 代数**：$\mathcal{R}$（第 23 篇）是「连续维数」的化身——量子信息的 Jones 指数、子因素理论都在 $\mathcal{R}$ 的子代数的世界里。

**Connes 的贡献**：用模理论（Tomita–Takesaki）把 III 型因子细分（第 23 篇），并证明超有限因子分类——von Neumann 代数是当代数学物理（量子场论、拓扑相）的公共语言。

**一句话总结**：von Neumann 代数是「量子力学的代数宪法」——它把物理的极限、对称、统计全部收纳进一个弱闭算子代数。

## 9 小结

- **von Neumann 代数**：含幺、弱闭（SOT 闭）的 $\ast$-子代数；SOT 闭 ⟺ WOT 闭。
- **交换子** $\mathcal{M}'$：与 $\mathcal{M}$ 可交换的一切算子，自动弱闭；$\mathcal{M}'''=\mathcal{M}'$。
- **双交换子定理**：含幺 $\ast$-子代数 $\mathcal{M}=\mathcal{M}''$ ⟺ 弱闭 ⟺ 强闭；证明核心是「投影到轨道 + Kaplansky 密度」。
- **单位元是命门**：$\mathcal{K}(\mathcal{H})$ 缺单位，双交换子定理对它失效。
- **谱定理的归宿**：交换 von Neumann 代数 ≅ $L^\infty(X,\mu)$；正规算子生成的 von Neumann 代数 = 它的 Borel 函数全体。
- **表示 → von Neumann**：$\pi(A)''$ 把每个表示升级成 von Neumann 代数。

在下一节，我们解剖 von Neumann 代数上的拓扑与泛函——**超弱拓扑与正常态**：$B(\mathcal{H})$