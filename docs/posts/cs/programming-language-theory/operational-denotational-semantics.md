---
title: 操作语义与指称语义
date: 2026-08-07
---

# 操作语义与指称语义

<div class="epigraph">
<p>程序测试只能用来证明 bug 的存在，而永远无法证明 bug 的缺席！</p>
<footer>—— 艾兹格 · 迪科斯彻（Edsger W. Dijkstra）</footer>
</div>

<div class="article-byline">
<p>第三级 · 程序设计语言理论 ｜ Pierce《类型与程序设计语言》Ch.3 ｜ 2026-08-07</p>
</div>

## 为什么从语义开始

上一课的类型安全依赖一句断言：「良类型程序每一步都保持类型，且永远卡不住。」可「步」「卡住」这些词，在没给程序下精确定义前，都只是比喻。**语义（semantics）**就是给程序的「意义」一个数学上可操作的精确答案。测试可以暴露错误，却无法证明正确；要谈「正确」，必须先有「意义」的权威定义。本课介绍两种主流答案：**操作语义**——意义在于「程序如何一步步运行」；**指称语义**——意义在于「程序计算出的那个数学对象」。这是全书形式化地层的基石，也是《编译原理》《程序分析》等专题一切证明的起点。

## 1 操作语义：意义就是求值过程

**操作语义（operational semantics）**把程序的意义定义为它的**行为**——「运行起来会发生什么」。皮尔斯在《类型与程序设计语言》第 3 章用算术表达式系统演示了它：语法只有布尔值、条件、后继 $\texttt{succ}$、前驱 $\texttt{pred}$、判零 $\texttt{iszero}$，外加数值与真值。行为用**归约规则（reduction rules）**写，每条规则说明「什么样的项可以变成什么样的项」：

$$\frac{t_1 \longrightarrow t'_1}{\texttt{if}\;t_1\;\texttt{then}\;t_2\;\texttt{else}\;t_3 \;\longrightarrow\; \texttt{if}\;t'_1\;\texttt{then}\;t_2\;\texttt{else}\;t_3}$$

$$\texttt{if}\;\texttt{true}\;\texttt{then}\;t_2\;\texttt{else}\;t_3 \;\longrightarrow\; t_2, \qquad \texttt{if}\;\texttt{false}\;\texttt{then}\;t_2\;\texttt{else}\;t_3 \;\longrightarrow\; t_3$$

这种「一次只走一步、在项内部的某个子位置上进行」的规则，称为**小步语义（small-step semantics）**。把「零步或多步」记作 $\longrightarrow^{*}$，程序的运行就是一条归约链：

$$\texttt{if}\;\texttt{true}\;\texttt{then}\;\texttt{succ}\;\overline{0}\;\texttt{else}\;\overline{0} \;\longrightarrow\; \texttt{succ}\;\overline{0} \;\longrightarrow\; \overline{1}$$

把这条链拆开看，每个环节都是「在某个子位置走一步」：$\texttt{if true then succ 0 else 0}$ 的最外层条件已经是 $\texttt{true}$，第一步走的是**计算规则**；而 $\texttt{succ}\;(\texttt{if true then 0 else 1})$ 则要先按**搜索规则**走进参数 `if` 内部，等参数归约成值再回到外层。**小步语义把「整个程序在干什么」压成「某个最小子项走了哪一步」**，这种一次一步、位置精确的粒度，正是之后证明「每一步都保持类型」所依赖的最小单位。

**求值的最终产物是值（value）**：无法再走、也不该再走的项（这里是数值与真值）。走不动但又不是值的项称为**卡住（stuck）**——$\texttt{iszero}\;\texttt{true}$ 就卡住了，因为类型系统（上一课）正是要把这类项赶出良类型集合。小步语义还给每条规则配了**左右两条「搜索规则」**（$\texttt{succ}\;t \to \texttt{succ}\;t'$ 等），规定参数按什么顺序求值——我们上一课讲的按值/按名调用，正是这些搜索规则的裁剪。

**卡住的最小样本**：$\texttt{iszero}\;\texttt{true}$ 无法再走（$\texttt{iszero}$ 只接受数值），可它又不是值，于是成了「卡住」的项。类型系统的全部工作，就是让这类项**根本进不了良类型集合**——上一课《类型系统与类型规则》的进展引理保证：良类型的封闭项绝不会卡住。语义里的「卡住」与类型里的「被拒绝」，是一枚硬币的两面。

搜索规则的不同裁剪，直接对应不同的**求值策略（evaluation strategy）**：若先归约实参再归约函数，得到**按值求值（call-by-value）**；若函数归约到值之前不碰实参，得到**按名求值（call-by-name）**。同一个项 $(\lambda x.\,\lambda y.\,5)\;(1/0)$：按值求值会先算 $1/0$ 而出错（卡住），按名求值则从不触碰 $1/0$、径直返回 $\lambda y.\,5$——**参数从未被用，就一次也不求值**。这条「什么时候求值」的分野，正是《参数传递与作用域》一课的序曲。

## 2 大步语义：只看起点与终点

与「一步一步走」相对，**大步语义（big-step / natural semantics）**只记录「从整体项到最终值」的一次性关系 $t \Downarrow v$，读作「项 $t$ 求值到值 $v$」。它的规则直接写终态：

$$\frac{t_1 \Downarrow \texttt{true} \quad t_2 \Downarrow v}{\texttt{if}\;t_1\;\texttt{then}\;t_2\;\texttt{else}\;t_3 \;\Downarrow\; v}
\qquad
\frac{t \Downarrow \texttt{true}}{\texttt{if}\;t\;\texttt{then}\;t_2\;\texttt{else}\;t_3 \;\Downarrow\; v_2}
\quad \text{（另一分支同理）}$$

**一个小步 vs 大步的对照**：求值 $\texttt{if true then succ 0 else pred 0}$，小步语义给出两条链：$\texttt{if true then succ 0 else pred 0} \longrightarrow \texttt{succ 0} \longrightarrow \overline{1}$；大步语义则直接写 $\texttt{if true then succ 0 else pred 0} \Downarrow \overline{1}$。前者把每一步都陈列出来，后者只报结果——**一个回答「怎么算的」，一个回答「算出什么」**。

**两种语义是同一枚硬币的两面**：小步把过程拆成可见的原子动作，便于证明「中间状态的性质」（如类型保持）；大步省略中间态、直接给出结果，写解释器（interpreter）时更接近递归实现——Scheme、SML 的语义教材多用大步，而类型安全证明几乎总用小步<span class="marginnote">大步语义的一个局限：它描述「求值到值」，却无法表达「卡住」或「永不终止」的中间形态，因为关系 $t \Downarrow v$ 只存在或不存立。想讨论「到正规形的每一条路径」，还得回到小步。</span>。

大步语义之所以贴近解释器实现，是因为它把「求值」写成一次性的递归：`eval(if t1 then t2 else t3) = eval(t2)` 当 `eval(t1) = true`。在 ML 里定义 `datatype value = ...` 再加上递归的 `eval` 函数，几乎就是大步规则的直接翻译；而小步语义对应「显式状态机 + 循环」的实现，更接近汇编层的一次一步推进。**语义不是空谈——它直接规定了解释器与证明的结构**。

## 3 指称语义：意义是数学对象

**指称语义（denotational semantics）**给出了另一个答案：程序的意义不是它的运行过程，而是它**所指称（denote）的那个数学对象**。用一个语义函数 $\llbracket \cdot \rrbracket$ 把语法映射到某个数学论域——表达式映射到数，布尔表达式映射到真值，程序映射到「从输入到输出的偏函数」。它最受推崇的品质是**组合性（compositionality）**：

$$\llbracket t_1\;t_2 \rrbracket = \llbracket t_1 \rrbracket\,(\llbracket t_2 \rrbracket)$$

**一个复合短语的意义，是它的各部分的意义的函数**。这把「程序的意义」还原为数学里的函数复合——于是程序的行为可以像数学对象一样推演、证明等式，而无需想象一台机器在跑。代价是：要容纳递归（如 Y 组合子或递归函数），论域必须用**域论（domain theory）**构造（Scott 域、不动点语义），数学复杂度陡增<span class="marginnote">指称语义的奠基人是 Dana Scott 与 Christopher Strachey（1960 年代）。Scott 发现用「连续函数加上最小不动点」可以给递归一个干净的数学意义；他同时代的大步/小步语义则被 Plotkin 与 Kahn 系统化。三大语义学派——操作、指称、公理——在 1970 年代基本成形。</span>。

**一个最简指称语义**：给算术表达式定义语义函数 $\llbracket \cdot \rrbracket$，规定 $\llbracket \overline{n} \rrbracket = n$、$\llbracket t_1 + t_2 \rrbracket = \llbracket t_1 \rrbracket + \llbracket t_2 \rrbracket$。于是 $\llbracket \overline{2} + \overline{3} \rrbracket = 2 + 3 = 5$——**表达式的意义就是它算出的那个数，加法被「翻译」成数学里的加法**，全程不需要想象一台机器在跑。这种「语法 → 数学对象」的映射，就是指称语义的全部工作。

处理递归时，语义函数会撞上自我指涉：若 $\llbracket \texttt{fact} \rrbracket$ 的等式右边出现它自己，就必须借助**最小不动点（least fixed point）**——在一族偏序的「连续函数」里取最小解，数学上保证这样的解存在且唯一。Scott 域论正是为此而造：它把「程序的意义」放进一个可以取不动点的空间，让递归函数获得干净的意义。**递归 = 不动点**，这一主题在《递归类型》与《System F》两课还会以类型的面貌再次出现。

## 4 三种语义的对比

| 语义 | 意义 = | 代表符号 | 强项 | 弱点 |
| --- | --- | --- | --- | --- |
| 操作（小步） | 一步步的行为 | $t \longrightarrow t'$ | 适合证明类型安全、并发交错 | 状态细节冗长 |
| 操作（大步） | 起点→终值 | $t \Downarrow v$ | 简洁、贴近解释器 | 无法表达中间态/卡住 |
| 指称 | 所指数学对象 | $\llbracket t \rrbracket$ | 组合性、可做等式推理 | 需域论工具，递归处理困难 |

三者是**互补**而非竞争：指称语义回答「程序算什么」，操作语义回答「程序怎么算」，而证明「两者一致」本身就是一篇重要研究——例如证明某个指称模型对每一步操作归约都保持意义（soundness of denotational semantics wrt operational semantics）。<span class="marginnote">还有一个本课未展开的流派——<strong>公理语义（axiomatic semantics）</strong>：用「前置条件/后置条件」给语句写证明规则，霍式逻辑（Hoare logic）即其代表，是《程序验证》专题的主角。</span>

**辨析｜易错点：** 不要把「求值顺序」与「语义类型」混为一谈。小步/大步说的是「语义怎么呈现」，按值/按名说的是「参数何时求值」——前者是呈现方式的差异，后者是策略的差异，二者正交：一个按值求值的小步语义和一个按名求值的小步语义，规则形状不同，但「小步」这个身份不变。**判断一个语言用哪种语义，看它的规则怎么写；判断一个语言用哪种策略，看它的搜索规则怎么裁。**

## 5 公式解析：小步规则的三段式读法

以条件语句的搜索规则为例，把小步规则解剖开：

$$
\frac{t_1 \longrightarrow t'_1}{\texttt{if}\;t_1\;\texttt{then}\;t_2\;\texttt{else}\;t_3 \;\longrightarrow\; \texttt{if}\;t'_1\;\texttt{then}\;t_2\;\texttt{else}\;t_3}
$$

- **前提 $t_1 \longrightarrow t'_1$**：条件 $t_1$ 先走一步。它把「整个 if 能否走」这个整体问题，化归为「它的条件能否走」这个子问题——**归约总是发生在项内部某个子位置**，这正是「结构归约」的含义。
- **结论左侧**：$t_1$ 走了，两个分支 $t_2, t_3$ **原封不动**。这与按值调用一致：先求条件，再谈分支。
- **结论右侧**：整个 if 的下一步就是「条件已前进后的同一个 if」。于是 $t_1$ 一路归约到 $\texttt{true}$ 或 $\texttt{false}$ 后，再由计算规则 $\texttt{if true then } t_2 \dots \longrightarrow t_2$ 接手，完成跳转。
- **规则间的分工**：搜索规则管「哪里能走」，计算规则管「走到哪里去」。两者合起来定义了这个语言的全部求值策略。

这套「搜索 + 计算」的分工可以推广：任何语言的小步语义都由两类规则拼成——**计算规则描述「值上发生的本质归约」，搜索规则描述「按什么顺序走进子项」**。读懂一个语言的操作语义，就是读懂这两族规则的编排。

## 6 术语速查

| 术语 | 记号 | 一句话直觉 |
| --- | --- | --- |
| 小步语义 | $t \longrightarrow t'$ | 一次走一步 |
| 大步语义 | $t \Downarrow v$ | 直接报结果 |
| 计算规则 | 值上的归约 | 真正干活 |
| 搜索规则 | 决定先归约哪个子项 | 决定顺序 |
| 值（value） | 不能再走的项 | 求值终点 |
| 卡住（stuck） | 走不动却非值的项 | 病态终点 |
| 指称语义 | $\llbracket t \rrbracket$ | 意义 = 数学对象 |
| 组合性 | $\llbracket t_1 t_2 \rrbracket = \llbracket t_1 \rrbracket (\llbracket t_2 \rrbracket)$ | 意义可拆可合 |
| 求值策略 | 按值 / 按名 / 按需 | 参数何时求值 |
| 公理语义 | 前置/后置条件写证明规则 | 霍式逻辑的流派 |

**一句话记忆**：操作语义把程序当「过程」、指称语义把程序当「对象」，而二者的等价性证明（指称模型相对操作归约的 soundness）是形式化语义学的核心检验——「程序算什么」与「程序怎么算」最终必须对得上。这张表也将贯穿后续各课：讲类型规则时是「判定怎么证」，讲对象与递归时是「自指怎么写」，语义始终是那台总在背景里运行的解释器。

## 7 小结

- **操作语义**把意义定义为运行行为：小步语义 $t \longrightarrow t'$ 拆成原子步骤（配搜索规则 + 计算规则），大步语义 $t \Downarrow v$ 只记起点与终值。
- 求值到达**值**即完成；走不动却非值的项称为**卡住**，类型系统的目标正是排除卡住的程序。
- **指称语义**把意义定义为数学对象：语义函数 $\llbracket t\rrbracket$ 满足**组合性**，代价是需要域论处理递归。
- 三大语义学派互补，证明「操作与指称一致」是重要研究主题。
- 语义是所有形式化证明的地基，也是后续每章规则写法的模板。

在下一节，我们从「项如何求值」转向「名字如何解析」——真实语言里参数怎么传、变量在哪一层有效，这是**参数传递与作用域**。
