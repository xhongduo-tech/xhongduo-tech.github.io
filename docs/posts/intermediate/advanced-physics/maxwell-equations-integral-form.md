---
title: 麦克斯韦方程组的积分形式
date: 2026-08-07
---

# 麦克斯韦方程组的积分形式

<div class="epigraph">
<p>四个方程，把电、磁、光的全部秘密浓缩成一组数学。麦克斯韦方程组，是物理学的诗。</p>
<footer>—— 后世物理学家对麦克斯韦方程组的评价</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等物理 ｜ 程守洙《普通物理学》第十四章 §14-2 ｜ 2026-08-07</p>
</div>

## 为什么从麦克斯韦方程组开始

从第十章到第十四章，我们积累了静电学、静磁学、电磁感应、位移电流的全部知识。现在到了收网的时刻：把它们合并成四个方程——**麦克斯韦方程组（Maxwell's equations）**。这四行方程完成了物理学的第二次大统一：电、磁、光原来是同一件事。它们是经典电磁学的巅峰，也是相对论（第十七章）与量子电动力学的起点。这一节给出积分形式，逐条解读物理意义，并讨论它的完备性——给定电荷电流分布，方程组唯一确定电磁场。

## 1 麦克斯韦方程组的积分形式

**麦克斯韦方程组（积分形式）**：

**① 电场的高斯定理**（电场有源）：

$$\oint_S \boldsymbol{D}\cdot\mathrm{d}\boldsymbol{S} = \sum q_{\text{自由}}$$

**② 磁场的高斯定理**（磁场无源）：

$$\oint_S \boldsymbol{B}\cdot\mathrm{d}\boldsymbol{S} = 0$$

**③ 法拉第定律**（变化磁场产生电场）：

$$\oint_L \boldsymbol{E}\cdot\mathrm{d}\boldsymbol{l} = -\frac{\mathrm{d}}{\mathrm{d}t}\int_S\boldsymbol{B}\cdot\mathrm{d}\boldsymbol{S} = -\frac{\mathrm{d}\Phi_B}{\mathrm{d}t}$$

**④ 全电流安培环路定理**（电流与变化电场产生磁场）：

$$\oint_L \boldsymbol{H}\cdot\mathrm{d}\boldsymbol{l} = \sum I + \frac{\mathrm{d}}{\mathrm{d}t}\int_S\boldsymbol{D}\cdot\mathrm{d}\boldsymbol{S}$$

<span class="marginnote">四方程的记忆框架：①「电荷是电场之源」、②「磁场无单极」、③「变磁生电」、④「电流和变电生磁」。③与④的对称性——「变化磁场 ↔ 变化电场」——是电磁波的发动机。</span>

## 2 四个方程的物理意义

逐一解读：

**方程①（高斯定理）**：电场线从正电荷出发、终于负电荷；电位移通量由自由电荷决定。真空中退化为 $\oint\boldsymbol{E}\cdot\mathrm{d}\boldsymbol{S} = \sum q/\varepsilon_0$。

**方程②（磁高斯定理）**：磁力线永远闭合，不存在磁单极子——磁场无「源」无「汇」。

**方程③（法拉第定律）**：变化的磁场激发涡旋电场；是动生 + 感生电动势的统一，含负号（楞次定律）。

**方程④（安培-麦克斯韦定律）**：磁场由传导电流与位移电流（变化电场）共同激发。

**重点：麦克斯韦方程组描述了「场如何被激发」与「场如何互相转化」，是经典电磁学的完备定律。** 配合介质方程（$\boldsymbol{D} = \varepsilon\boldsymbol{E}$、$\boldsymbol{B} = \mu\boldsymbol{H}$）与洛伦兹力公式（$\boldsymbol{F} = q\boldsymbol{E} + q\boldsymbol{v}\times\boldsymbol{B}$），可以解决一切经典电磁问题。<span class="marginnote">麦克斯韦方程组的完备性：给定初始时刻的电磁场与边界上的电荷电流分布，四方程唯一确定此后任意时刻的电磁场。它们是决定论方程——拉普拉斯的「确定性」在电磁学中的体现，直到量子力学才被修正。</span>

## 3 麦克斯韦方程组与电磁波

在真空中（无电荷、无电流：$\rho = 0$、$\boldsymbol{j} = 0$），方程组退化为对称形式。由 ③ 与 ④ 联立，可以推出**波动方程**：

$$\nabla^2\boldsymbol{E} - \frac{1}{c^2}\frac{\partial^2\boldsymbol{E}}{\partial t^2} = 0, \qquad \nabla^2\boldsymbol{B} - \frac{1}{c^2}\frac{\partial^2\boldsymbol{B}}{\partial t^2} = 0$$

其中波速：

$$c = \frac{1}{\sqrt{\mu_0\varepsilon_0}} \approx 3\times10^8\ \text{m/s}$$

**重点：麦克斯韦方程组预言电磁波，且波速等于光速——「光是一种电磁波」。** 这是 19 世纪物理学的巅峰成就：光的本质（波动光学）与电磁学合流。赫兹 1888 年用实验产生并探测到电磁波，验证了麦克斯韦的理论。<span class="marginnote">「$c = 1/\sqrt{\mu_0\varepsilon_0}$」这个式子看起来平凡，却藏着一个惊人的事实：光速由两个电磁常数决定，与光源运动无关——这正是狭义相对论（第十七章）的出发点。麦克斯韦方程组已经埋下了相对论的种子。</span>

## 4 公式解析：麦克斯韦方程组与波动方程

简述从方程组到波动方程的关键一步：对真空中的法拉第定律取旋度，再用 ④ 代入。

$$
\nabla\times(\nabla\times\boldsymbol{E}) = -\mu_0\frac{\partial}{\partial t}(\nabla\times\boldsymbol{H}) = -\mu_0\varepsilon_0\frac{\partial^2\boldsymbol{E}}{\partial t^2}
$$

- **第一步，取旋度**：对 $\nabla\times\boldsymbol{E} = -\partial\boldsymbol{B}/\partial t$ 两边取旋度（$\nabla\times$）。
- **第二步，代入 ④**：$\nabla\times\boldsymbol{H} = \varepsilon_0\frac{\partial\boldsymbol{E}}{\partial t}$（真空中无电流），替换右边。
- **第三步，矢量恒等式**：$\nabla\times(\nabla\times\boldsymbol{E}) = \nabla(\nabla\cdot\boldsymbol{E}) - \nabla^2\boldsymbol{E}$，真空中 $\nabla\cdot\boldsymbol{E} = 0$，只剩 $-\nabla^2\boldsymbol{E}$。
- **第四步，得波动方程**：$-\nabla^2\boldsymbol{E} = -\mu_0\varepsilon_0\partial^2\boldsymbol{E}/\partial t^2$，即 $\nabla^2\boldsymbol{E} = \mu_0\varepsilon_0\partial^2\boldsymbol{E}/\partial t^2$，波速 $c = 1/\sqrt{\mu_0\varepsilon_0}$。

**辨析｜易错点：**麦克斯韦方程组是**经典**电磁学的完备描述，但它不包含量子效应。在微观尺度（原子内部）与极端条件（强场）下，需要量子电动力学（QED）。另外，方程组有积分与微分两种形式：积分形式看「通量/环流」（宏观、直观），微分形式看「散度/旋度」（局域、微分方程）——第二十一章将给出微分形式。

## 5 麦克斯韦方程组的地位

| 方程 | 来源 | 物理意义 |
| --- | --- | --- |
| ① $\oint\boldsymbol{D}\cdot\mathrm{d}\boldsymbol{S} = \sum q$ | 库仑定律 + 高斯 | 电荷激发电场 |
| ② $\oint\boldsymbol{B}\cdot\mathrm{d}\boldsymbol{S} = 0$ | 磁单极不存在 | 磁场无源 |
| ③ $\oint\boldsymbol{E}\cdot\mathrm{d}\boldsymbol{l} = -\mathrm{d}\Phi_B/\mathrm{d}t$ | 法拉第定律 | 变磁生电 |
| ④ $\oint\boldsymbol{H}\cdot\mathrm{d}\boldsymbol{l} = \sum I + \mathrm{d}\Phi_D/\mathrm{d}t$ | 安培 + 位移电流 | 变电生磁 |

<span class="marginnote">麦克斯韦方程组统一了电学、磁学与光学，是物理学的第二次大统一（第一次是牛顿力学统一天上地下，第三次是爱因斯坦统一时空，第四次是电弱统一……）。它启发了爱因斯坦的狭义相对论（光速不变）、以及后来的规范场论——「从极限到大模型」里这条「统一」的主线，麦克斯韦是承前启后的巨人。</span>

## 6 小结

- **麦克斯韦方程组（积分形式）**：四个方程——电场有源、磁场无源、变磁生电、变电生磁。
- ③与④的对称性：变化的磁场与电场互相激发，是电磁波的发动机。
- 真空中推出波动方程，波速 $c = 1/\sqrt{\mu_0\varepsilon_0} = $ 光速——光是电磁波。
- 方程组 + 介质方程 + 洛伦兹力 = 经典电磁学的完备描述。
- 预言电磁波（赫兹验证）、埋下相对论种子——19 世纪物理学的巅峰。

在下一节，我们研究电磁波的性质——**电磁波的产生与基本性质**。
