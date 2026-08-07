---
title: 平面向量的加、减运算
date: 2026-08-07
---

# 平面向量的加、减运算

<div class="epigraph">
<p>两条路在林中分叉，我选择了人迹更少的一条。</p>
<footer>—— 罗伯特 · 弗罗斯特（Robert Frost, The Road Not Taken）</footer>
</div>

<div class="article-byline">
<p>第一级 · 基础数学 ｜ 人教A版 必修第二册 §6.2 ｜ 2026-08-07</p>
</div>

## 为什么从向量加减开始

上一节定义了向量，但只定义了「它是什么」，还没回答「拿它做什么」。本节装上第一个运算：**加法**。向量的加法不是两个数的加法——两个位移「先向东 3 米再向北 4 米」的总效果，不是把 3 和 4 相加，而是「从起点直接连到终点」，结果是 5 米。**向量加法的几何直觉就是「位移的合成」**：先走一个位移，再走一个位移，合起来就是两次位移的接力。<span class="marginnote">注意 3 + 4 = 5 的三角形：两个相互垂直的向量相加，长度满足勾股定理而非算术加法。这说明「向量相加」不是「长度相加」——向量的运算规则由它的几何意义决定，不能拿数的运算律硬套。</span> 理解这一点，是后面理解力合成、速度叠加、以及大模型里向量加法的关键第一步。

## 1 加法：三角形法则与平行四边形法则

已知向量 $\vec{a}$、$\vec{b}$，定义它们的和 $\vec{a}+\vec{b}$ 如下：**三角形法则**——平移使 $\vec{b}$ 的起点与 $\vec{a}$ 的终点相接，则从 $\vec{a}$ 的起点指向 $\vec{b}$ 的终点的新向量就是 $\vec{a}+\vec{b}$。

$$
\vec{a}+\vec{b} = \overrightarrow{AC},\quad \text{其中 } \overrightarrow{AB}=\vec{a},\ \overrightarrow{BC}=\vec{b}
$$

**平行四边形法则**——平移使 $\vec{a}$、$\vec{b}$ 共起点，以它们为邻边作平行四边形，则从公共起点指向对角顶点的那条对角线就是 $\vec{a}+\vec{b}$。两种法则等价：三角形法则是「首尾相接」，平行四边形法则是「共起点作对角线」，画法不同，得到的和相同。<span class="marginnote">平行四边形法则的本质就是「同时发生」的合成：一条船同时被水流推、被桨推，实际运动方向是两者共同作用的方向——这就是对角线。而三角形法则对应「先后发生」的合成。同一道加法，两种叙事，殊途同归。</span>

### 运算律

向量加法满足**交换律**与**结合律**：

$$
\vec{a}+\vec{b}=\vec{b}+\vec{a}, \qquad (\vec{a}+\vec{b})+\vec{c}=\vec{a}+(\vec{b}+\vec{c})
$$

几何上，交换律说的就是平行四边形对角线只有一个；结合律说的是「多个向量首尾相接，和就是从第一个起点指向最后一个终点」，与加括号的方式无关。

**重点：向量加法满足交换律与结合律，这一点与数的加法完全一致**——因此多个向量求和时，可以任意调换次序、任意分组。同时 $\vec{a}+\vec{0}=\vec{a}$（零向量是加法单位元），$\vec{a}+(-\vec{a})=\vec{0}$（相反向量之和为零）。

## 2 减法：转化为加法

向量的减法**不另立新规**，而是用相反向量归约为加法：

$$
\vec{a}-\vec{b}=\vec{a}+(-\vec{b})
$$

**减一个向量，等于加上它的相反向量。** 几何上：把 $\vec{a}$、$\vec{b}$ 平移成共起点，则从 $\vec{b}$ 的终点指向 $\vec{a}$ 的终点（注意箭头方向！）的向量就是 $\vec{a}-\vec{b}$。<span class="marginnote">口诀：「共起点，连终点，箭头指向被减向量。」$\vec{a}-\vec{b}$ 的箭头永远指向 $\vec{a}$（被减数）那一边。凡是画错减法方向的人，九成是忘了「被减数」这三个字——被减的放前面，方向朝它。</span>

由此可得一个重要恒等式：$\overrightarrow{AB}=-\overrightarrow{BA}$，以及中点公式的雏形——若 $M$ 是线段 $AB$ 的中点，则 $\overrightarrow{OM}=\dfrac{1}{2}(\overrightarrow{OA}+\overrightarrow{OB})$。这条式子把「中点」翻译成了向量语言，是下一章《平面向量基本定理》与解析几何里反复使用的工具。

## 3 公式解析：位移合成的三角不等式

向量加法的长度有什么规律？设 $\vec{a}$、$\vec{b}$ 为任意向量，有**三角不等式**：

$$
\left|\vec{a}+\vec{b}\right| \le |\vec{a}| + |\vec{b}|
$$

拆三步理解：

- **第一步，看几何**：$\vec{a}$、$\vec{b}$ 首尾相接构成三角形，第三边正是 $|\vec{a}+\vec{b}|$。三角形两边之和大于第三边——这正是不等式名字的来历。
- **第二步，什么时候取等号**：当 $\vec{a}$、$\vec{b}$ **同向**时，三条边躺在一条直线上，$\left|\vec{a}+\vec{b}\right|=|\vec{a}|+|\vec{b}|$。直线是三角形的退化极限，取等号意味着「三边共线且同向」。
- **第三步，反向情形**：$\vec{a}$、$\vec{b}$ 反向时，和长 $=||\vec{a}|-|\vec{b}||$。合力的最小可能值就在反向时取到——「方向相反的两个力互相抵消」，这正解释了为什么拔河拉直线最省力。

三角不等式把「几何直觉」与「代数不等式」焊在一起，是后面学习绝对值不等式、向量模长、以及范数概念的第一课。<span class="marginnote">把三角不等式推广到 $n$ 个向量与高维空间，就是线性代数里的「范数三角不等式」，那是整个度量几何的地基。今天这条不等式，是那个宏大结构的一个点。</span>

## 4 向量加减在平面几何中的应用

向量加减不只是定义，更是**证明工具**。一个经典例子：证明「三角形两边中点的连线平行于第三边且等于第三边的一半」。设三角形顶点 $A,B,C$，$D,E$ 分别是 $AB,AC$ 的中点，则

$$
\overrightarrow{DE}=\overrightarrow{AE}-\overrightarrow{AD}=\frac{1}{2}\overrightarrow{AC}-\frac{1}{2}\overrightarrow{AB}=\frac{1}{2}\left(\overrightarrow{AC}-\overrightarrow{AB}\right)=\frac{1}{2}\overrightarrow{BC}
$$

于是 $\overrightarrow{DE}=\frac{1}{2}\overrightarrow{BC}$，即 $DE\parallel BC$ 且 $DE=\frac{1}{2}BC$。<span class="marginnote">这个证明的妙处在于：几何里要费力构造的「中位线定理」，用向量减法两行就写完了。<strong>把几何关系翻译成向量等式，用代数算，再把结果翻译回几何</strong>——这就是「向量法」的完整循环，也是后面解析几何的精神。</span> 请注意 $\overrightarrow{AE}-\overrightarrow{AD}$ 这一步：它正是「共起点、连终点、箭头指向被减数 $\overrightarrow{AE}$」。

## 5 例题精讲：向量加减的几何应用

向量加减在几何里的应用，集中在「用向量表示线段关系」。看两道题。

### 题一：用向量表示三角形的中线

在 $\triangle ABC$ 中，$D$ 为 $BC$ 的中点，用 $\overrightarrow{AB}=\vec a$、$\overrightarrow{AC}=\vec b$ 表示 $\overrightarrow{AD}$。

- **第一步，首尾相接**：$\overrightarrow{AD}=\overrightarrow{AB}+\overrightarrow{BD}$。
- **第二步，用中点条件**：$\overrightarrow{BD}=\frac12\overrightarrow{BC}=\frac12(\vec b-\vec a)$。
- **第三步，合并**：$\overrightarrow{AD}=\vec a+\frac12(\vec b-\vec a)=\frac12\vec a+\frac12\vec b=\frac12(\vec a+\vec b)$——中线向量等于两边向量和的二分之一。

<span class="marginnote">「首尾相接 + 中点条件」是向量表示线段的万能法：<strong>从起点出发，沿已知路线走到终点，每段用已知向量表示</strong>。$\overrightarrow{BD}=\frac12\overrightarrow{BC}$ 是中点条件的向量语言。这条「中线向量 = 两边向量和的一半」的结论，与「平行四边形对角线交点」的向量表示一脉相承，是向量法的基本工具。</span>

### 题二：用向量证明对角线关系

在平行四边形 $ABCD$ 中，用向量证明对角线 $AC$ 与 $BD$ 互相平分。

- **第一步，设向量**：$\vec a=\overrightarrow{AB}$，$\vec b=\overrightarrow{AD}$。$\overrightarrow{AC}=\vec a+\vec b$，$\overrightarrow{BD}=\overrightarrow{AD}-\overrightarrow{AB}=\vec b-\vec a$。
- **第二步，设交点**：设 $AC$ 与 $BD$ 交于 $O$，$\overrightarrow{AO}=t\overrightarrow{AC}=t(\vec a+\vec b)$。
- **第三步，两条路表示 $\overrightarrow{AO}$**：$\overrightarrow{AO}=\overrightarrow{AB}+\overrightarrow{BO}=\vec a+s(\vec b-\vec a)=(1-s)\vec a+s\vec b$。系数相等：$t=1-s$、$t=s$，解得 $t=s=\frac12$——$O$ 是 $AC$ 与 $BD$ 的中点，对角线互相平分。

<span class="marginnote">「同向量两种表示 + 系数唯一」是向量法证明的引擎：<strong>$\overrightarrow{AO}$ 从 $AC$ 走与从 $AB+BO$ 走，得到两个表达式，基本定理保证系数相同，于是列方程解出比例</strong>。本题 $t=\frac12$ 说明交点平分对角线。这类「设比例、列方程、得定比」的流程，是向量法证明几何命题的通用套路。</span>

**辨析｜易错点（补充）：** 一是**减法方向**——$\overrightarrow{BD}=\overrightarrow{AD}-\overrightarrow{AB}=\vec b-\vec a$，箭头指向被减向量 $\vec b$，别写反；二是**设分点漏系数**——$\overrightarrow{BO}=s\overrightarrow{BD}=s(\vec b-\vec a)$，$s$ 是未知比例，别漏乘；三是**系数对应错位**——$\vec a$ 与 $\vec b$ 的系数分别对齐，别交叉配。

## 6 小结

- 向量加法：**三角形法则**（首尾相接）与**平行四边形法则**（共起点作对角线）等价。
- 加法满足**交换律、结合律**；$\vec{a}+\vec{0}=\vec{a}$，$\vec{a}+(-\vec{a})=\vec{0}$。
- 减法归约为加法：$\vec{a}-\vec{b}=\vec{a}+(-\vec{b})$；几何上「共起点、连终点、箭头指向被减向量」。
- 三角不等式 $\left|\vec{a}+\vec{b}\right|\le|\vec{a}|+|\vec{b}|$，同向取等号。
- 向量法是「翻译 → 计算 → 翻译回」的循环，中位线定理是第一个范例。

在下一节，我们给向量配上「放大缩小」：**平面向量的数乘运算**——当一个数去乘一个向量时，方向与长度各发生什么。
