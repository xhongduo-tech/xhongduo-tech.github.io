---
title: ABM 范式与 NetLogo 基础
date: 2026-08-07
---

# ABM 范式与 NetLogo 基础

<div class="epigraph">
<p>我们倾向于用「静止的、可以被分析的对象」来思考，但世界是由「运动的、彼此作用的成分」构成的。</p>
<footer>—— 米切尔 · 雷斯尼克（Mitchel Resnick, <em>Turtles, Termites, and Traffic Jams</em>）</footer>
</div>

<div class="article-byline">
<p>第九级 · 基于主体的建模与复杂适应系统 ｜ Wilensky &amp; Rand《An Introduction to Agent-Based Modeling》第1章 ｜ 2026-08-07</p>
</div>

## 为什么从 NetLogo 开始

上一节我们立住了「涌现要从底部生长」的思想，但思想不变成能跑的模型就只是隐喻。这一节的任务是把范式落地：选择一门最小的建模语言，亲手把第一批主体放进虚拟世界。<span class="marginnote">NetLogo 由 Uri Wilensky 于 1999 年在西北大学开发，源自 1960 年代 Papert 的 LOGO 语言传统——「海龟」（turtle）一词正是从 LOGO 继承而来。它至今是 ABM 教学与科研引用量最高的平台之一。</span>选择 NetLogo 不是因为它是功能最强的 ABM 平台——功能最强的是用于大规模仿真的 Repast 与 Mason，本专题后面会专门讲到——而是因为它把 ABM 的「主体-规则-观察」三件事暴露得最干净，学它等于学 ABM 的共同语法。

## 1 ABM 范式：从写方程到写规则

在进入 NetLogo 前，先想清楚 ABM 建模者与常规建模者做的事有什么本质不同。常规模型（微分方程、统计模型）的建立方式是：**先确定感兴趣的宏观变量，再写这些变量之间如何变化的方程**。ABM 的方式正好颠倒：

**常规范式**：宏观变量 $\to$ 宏观方程 $\to$ 求解/模拟 $\to$ 得到宏观行为。

**ABM 范式**：定义主体 $\to$ 写个体规则 $\to$ 运行虚拟世界 $\to$