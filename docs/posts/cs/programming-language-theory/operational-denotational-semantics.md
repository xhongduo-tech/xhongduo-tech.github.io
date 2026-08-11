---
title: 操作语义与指称语义
date: 2026-08-11
---

# 操作语义与指称语义

<div class="epigraph">
<p>程序测试只能用来证明 bug 的存在，而永远无法证明 bug 的缺席！</p>
<footer>—— 艾兹格 · 迪科斯彻（Edsger W. Dijkstra）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机基础 · 程序设计语言理论 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从语义开始

上一课的类型安全依赖一句断言：「良类型程序每一步都保持类型，且永远卡不住。」可「步」「卡住」这些词，在没给程序下精确定义前，都只是比喻。**语义（semantics）**就是给程序的「意义」一个数学上可操作的精确答案。测试可以暴露错误，却无法证明正确；要谈「正确」，必须先有「意义」的权威定义。本课介绍两种主流答案：**操作语义**——意义在于「程序如何一步步运行」；**指称语义**——意义在于「程序计算出的那个数学对象」。这是全书形式化地层的基石，也是《编译原理》《程序分析》等专题一切证明的起点。

## 1 操作语义：意义就是求值过程

**操作语义（operational semantics）**把程序的意义定义为它的**行为**——「运行起来会发生什么」。皮尔斯在《类型与程序设计语言》第 3 章用算术表达式系统演示了它：语法只有布尔值、条件、后继 $\texttt{succ}$、前驱 $\texttt{pred}$、判零 $\texttt{iszero}$，外加数值与真值。行为用**归约规则（reduction rules）**写，每条规则说明「什么样的项可以变成什么样的项」：

$$\frac{t_1 \longrightarrow t'_1}{\texttt{if}\;t_1\;\texttt{then}\;t_2\;\texttt{else}\;t_3 \;\longrightarrow\; \texttt{if}\;t'_1\;\texttt{then}\;t_2\;\texttt{else}\;t_3}$$

$$\texttt{if}\;\texttt{true}\;\texttt{then}\;t_2\;\texttt{else}\;t_3 \;\longrightarrow\; t_2, \qquad \texttt{if}\;\texttt{false}\;\texttt{then}\;t_2\;\texttt{else}\;t_3 \;\longrightarrow\; t_3$$

这种「一次只走一步、在项内部的某个子位置上进行」的规则，称为**小步语义（small-step semantics）**。把「零步或多步」记作 $\longrightarrow^{*}$，程序的运行就是一条归约链：

$$\texttt{if}\;\texttt{true}\;\texttt{then}\;\texttt{succ}\;\overline{0}\;\texttt{else}\;\overline{0} \;\longrightarrow\; \texttt{succ}\;\overline{0} \;\longrightarrow\; \overline{1}$$

**求值的最终产物是值（value）**：无法再走、也不该再走的项（这里是数值与真值）。走不动但又不是值的项称为**卡住（stuck）**——$\texttt{iszero}\;\texttt{true}$ 就卡住了，因为类型系统（上一课）正是要把这类项赶出良类型集合。小步语义还给每条规则配了**左右两条「搜索规则」**（$\texttt{succ}\;t \to \texttt{succ}\;t'$ 等），规定参数按什么顺序求值——我们上一课讲的按值/按名调用，正是这些搜索规则的裁剪。

## 2 大步语义：只看起点与终点

与「一步一步走」相对，**大步语义（big-step / natural semantics）**只记录「从整体项到最终值」的一次性关系 $t \Downarrow v$，读作「项 $t$ 求值到值 $v$」。它的规则直接写终态：

$$\frac{t_1 \Downarrow \texttt{true} \quad t_2 \Downarrow v}{\texttt{if}\;t_1\;\texttt{then}\;t_2\;\texttt{else}\;t_3 \;\Downarrow\; v}
\qquad
\frac{t \Downarrow \texttt{true}}{\texttt{if}\;t\;\texttt{then}\;t_2\;\texttt{else}\;t_3 \;\Downarrow\; v_2}
\quad \text{（另一分支同理）}$$

**两种语义是同一枚硬币的两面**：小步把过程拆成可见的原子动作，便于证明「中间状态的性质」（如类型保持）；大步省略中间态、直接给出结果，写解释器（interpreter）时更接近递归实现——Scheme、SML 的语义教材多用大步，而类型安全证明几乎总用小步<span class="marginnote">大步语义的一个局限：它描述「求值到值」，却无法表达「卡住」或「永不终止」的中间形态，因为关系 $t \Downarrow v$ 只存在或不存立。想讨论「到正规形的每一条路径」，还得回到小步。</span>。

## 3 指称语义：意义是数学对象

**指称语义（denotational semantics）**给出了另一个答案：程序的意义不是它的运行过程，而是它**所指称（denote）的那个数学对象**。用一个语义函数 $\llbracket \cdot \rrbracket$ 把语法映射到某个数学论域——表达式映射到数，布尔表达式映射到真值，程序映射到「从输入到输出的偏函数」。它最受推崇的品质是**组合性（compositionality）**：

$$\llbracket t_1\;t_2 \rrbracket = \llbracket t_1 \rrbracket\,(\llbracket t_2 \rrbracket)$$

**一个复合短语的意义，是它的各部分的意义的函数**。这把「程序的意义」还原为数学里的函数复合——于是程序的行为可以像数学对象一样推演、证明等式，而无需想象一台机器在跑。代价是：要容纳递归（如 Y 组合子或递归函数），论域必须用**域论（domain theory）**构造（Scott 域、不动点语义），数学复杂度陡增<span class="marginnote">指称语义的奠基人是 Dana Scott 与 Christopher Strachey（1960 年代）。Scott 发现用「连续函数加上最小不动点」可以给递归一个干净的数学意义；他同时代的大步/小步语义则被 Plotkin 与 Kahn 系统化。三大语义学派——操作、指称、公理——在 1970 年代基本成形。</span>。

## 4 三种语义的对比

| 语义 | 意义 = | 代表符号 | 强项 | 弱点 |
| --- | --- | --- | --- | --- |
| 操作（小步） | 一步步的行为 | $t \longrightarrow t'$ | 适合证明类型安全、并发交错 | 状态细节冗长 |
| 操作（大步） | 起点→终值 | $t \Downarrow v$ | 简洁、贴近解释器 | 无法表达中间态/卡住 |
| 指称 | 所指数学对象 | $\llbracket t \rrbracket$ | 组合性、可做等式推理 | 需域论工具，递归处理困难 |

三者是**互补**而非竞争：指称语义回答「程序算什么」，操作语义回答「程序怎么算」，而证明「两者一致」本身就是一篇重要研究——例如证明某个指称模型对每一步操作归约都保持意义（soundness of denotational semantics wrt operational semantics）。<span class="marginnote">还有一个本课未展开的流派——<strong>公理语义（axiomatic semantics）</strong>：用「前置条件/后置条件」给语句写证明规则，霍式逻辑（Hoare logic）即其代表，是《程序验证》专题的主角。</span>

## 5 公式解析：小步规则的三段式读法

以条件语句的搜索规则为例，把小步规则解剖开：

$$
\frac{t_1 \longrightarrow t'_1}{\texttt{if}\;t_1\;\texttt{then}\;t_2\;\texttt{else}\;t_3 \;\longrightarrow\; \texttt{if}\;t'_1\;\texttt{then}\;t_2\;\texttt{else}\;t_3}
$$

- **前提 $t_1 \longrightarrow t'_1$**：条件 $t_1$ 先走一步。它把「整个 if 能否走」这个整体问题，化归为「它的条件能否走」这个子问题——**归约总是发生在项内部某个子位置**，这正是「结构归约」的含义。
- **结论左侧**：$t_1$ 走了，两个分支 $t_2, t_3$ **原封不动**。这与按值调用一致：先求条件，再谈分支。
- **结论右侧**：整个 if 的下一步就是「条件已前进后的同一个 if」。于是 $t_1$ 一路归约到 $\texttt{true}$ 或 $\texttt{false}$ 后，再由计算规则 $\texttt{if true then } t_2 \dots \longrightarrow t_2$ 接手，完成跳转。
- **规则间的分工**：搜索规则管「哪里能走」，计算规则管「走到哪里去」。两者合起来定义了这个语言的全部求值策略。

## 6 小结

- **操作语义**把意义定义为运行行为：小步语义 $t \longrightarrow t'$ 拆成原子步骤（配搜索规则 + 计算规则），大步语义 $t \Downarrow v$ 只记起点与终值。
- 求值到达**值**即完成；走不动却非值的项称为**卡住**，类型系统的目标正是排除卡住的程序。
- **指称语义**把意义定义为数学对象：语义函数 $\llbracket t\rrbracket$ 满足**组合性**，代价是需要域论处理递归。
- 三大语义学派互补，证明「操作与指称一致」是重要研究主题。
- 语义是所有形式化证明的地基，也是后续每章规则写法的模板。

在下一节，我们从「项如何求值」转向「名字如何解析」——真实语言里参数怎么传、变量在哪一层有效，这是**参数传递与作用域**。
