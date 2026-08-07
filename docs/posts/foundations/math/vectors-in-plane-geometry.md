---
title: 向量在平面几何中的应用
date: 2026-08-07
---

# 向量在平面几何中的应用

<div class="epigraph">
<p>每一种几何都可以代数化，每一种代数都可以几何化。</p>
<footer>—— 大卫 · 希尔伯特（David Hilbert）</footer>
</div>

<div class="article-byline">
<p>第一级 · 基础数学 ｜ 人教A版 必修第二册 §6.4.1 ｜ 2026-08-07</p>
</div>

## 为什么从向量与平面几何开始

平面几何有几千年的公理化传统：画辅助线、证全等、引垂线，靠的是巧思。向量方法则换了一条路——**不靠巧思，靠翻译**。把平行翻成「成比例」，把垂直翻成「点乘为零」，把中点翻成「坐标平均」，把共线翻成「系数和为 1」，然后放心大胆地算。本节把前面所有向量工具（加减、数乘、数量积、基本定理、坐标）集合起来，用它们系统证明一批平面几何命题。<span class="marginnote">希尔伯特断言几何可以代数化——向量正是这场代数化的第一件利器。学会向量法，等于给「证几何题」装上了一台通用机器：虽然不一定是最短的证法，但几乎一定是「不用想」的证法。</span> 这也是「从极限到大模型」里反复出现的思想：**换一套语言，难题的难度不变，但性质变了——从「找巧思」变成「走流程」。**

## 1 几何条件与向量条件的对照表

向量证明的第一步，是把几何语言翻译成向量语言。这一对照表是本节的地基：

| 几何条件 | 向量条件 |
| --- | --- |
| $P$ 在直线 $AB$ 上 | $\overrightarrow{AP}=\lambda\overrightarrow{AB}$（$\lambda\in\mathbb{R}$） |
| $A,B,C$ 三点共线 | $\overrightarrow{AB}=\lambda\overrightarrow{AC}$ |
| $M$ 是 $AB$ 的中点 | $\overrightarrow{OM}=\frac{1}{2}(\overrightarrow{OA}+\overrightarrow{OB})$ |
| $\ell_1\parallel\ell_2$ | 方向向量 $\vec{a}\parallel\vec{b}$，即 $\vec{a}=\lambda\vec{b}$ |
| $\ell_1\perp\ell_2$ | 方向向量 $\vec{a}\cdot\vec{b}=0$ |
| $P$ 分 $AB$ 为 $m:n$ | $\overrightarrow{AP}=\frac{m}{m+n}\overrightarrow{AB}$ |

<span class="marginnote">这张表是「翻译手册」：做题前先把题目里的每个几何词换成对应的向量式。多数学生证明不下去，不是向量运算不熟，而是<strong>几何条件没有及时翻译</strong>——漏译一步，后面就全卡住。</span> 熟记这张表，等于拿到了几何问题的「向量词典」。

## 2 经典证明一：平行四边形对角线互相平分

用向量法证明「平行四边形的对角线互相平分」。设平行四边形 $ABCD$，$O$ 为两条对角线 $AC$ 与 $BD$ 的交点。证明 $O$ 同时是 $AC$ 与 $BD$ 的中点。

- **第一步，翻译条件**：$O$ 在 $AC$ 上，设 $\overrightarrow{AO}=t\overrightarrow{AC}$；$O$ 在 $BD$ 上，设 $\overrightarrow{BO}=s\overrightarrow{BD}$。其中 $t,s$ 是待定比例。
- **第二步，用基底表示**：选 $\vec{a}=\overrightarrow{AB}$、$\vec{b}=\overrightarrow{AD}$ 为基底。由 $ABCD$ 是平行四边形，$\overrightarrow{AC}=\vec{a}+\vec{b}$，$\overrightarrow{BD}=\vec{b}-\vec{a}$。
- **第三步，列方程解出**：$\overrightarrow{AO}=t(\vec{a}+\vec{b})$；而 $\overrightarrow{AO}=\overrightarrow{AB}+\overrightarrow{BO}=\vec{a}+s(\vec{b}-\vec{a})=(1-s)\vec{a}+s\vec{b}$。两个表达式必须相等，由基本定理系数唯一：

$$
1-s=t, \qquad s=t
$$

解得 $t=s=\frac{1}{2}$。于是 $O$ 是 $AC$ 与 $BD$ 的中点，证毕。<span class="marginnote">注意第三步用的是「<strong>同一向量在基底下的表示唯一</strong>」——这是基本定理在证明里的标准出场方式。两条不同的路线表示同一个向量，系数必须对应相等，方程由此而来。几何题变成了解方程组。</span>

这个证明的意义在于：它完全没有画辅助线，全靠「设比例 → 表示 → 列方程」，并且**可以机械化地照搬到任意四边形、任意分点问题**。

## 3 公式解析：用数量积证「对角线垂直的平行四边形是菱形」

设平行四边形 $ABCD$ 的对角线 $AC\perp BD$，证明它是菱形（即邻边相等）。

- **第一步，翻译**：用向量表示两条对角线：$\overrightarrow{AC}=\vec{a}+\vec{b}$，$\overrightarrow{BD}=\vec{b}-\vec{a}$，其中 $\vec{a}=\overrightarrow{AB}$、$\vec{b}=\overrightarrow{AD}$。垂直条件翻译为 $\overrightarrow{AC}\cdot\overrightarrow{BD}=0$。
- **第二步，展开点乘**：$(\vec{a}+\vec{b})\cdot(\vec{b}-\vec{a})=|\vec{b}|^2-|\vec{a}|^2$。分配律展开时，$\vec{a}\cdot\vec{b}$ 与 $\vec{b}\cdot\vec{a}$ 恰好相消——**垂直条件等价于「两对角线向量点乘为零」**。
- **第三步，读出结论**：$|\vec{b}|^2-|\vec{a}|^2=0$，即 $|\vec{b}|=|\vec{a}|$，也就是 $AD=AB$，平行四边形邻边相等，故为菱形。证毕。

这条证明展示了数量积的绝妙之处：**它把「角度条件」变成了「模长条件」**——垂直本来是角度关系，一展开点乘，竟然直接吐出邻边相等的代数事实。<span class="marginnote">数量积像一座桥：桥的这一头是角度（$\cos\theta=0$），桥的那一头是长度（$|\vec{a}|=|\vec{b}|$）。很多看似八竿子打不着的几何量，经过点乘展开就勾连起来。这也是为什么数量积是「几何证明神器」。</span>

## 4 三线共点：向量的高光时刻

「三线共点」是平面几何里公认的难题，向量法却能一视同仁地处理。以「三角形的三条中线交于一点（重心）」为例：设 $G$ 是中线 $AD$ 与 $BE$ 的交点，证明 $G$ 也在中线 $CF$ 上，且 $\overrightarrow{AG}=\frac{2}{3}\overrightarrow{AD}$。

设 $\vec{a}=\overrightarrow{AB}$、$\vec{b}=\overrightarrow{AC}$，$D$ 是 $BC$ 中点，则 $\overrightarrow{AD}=\frac{1}{2}(\vec{a}+\vec{b})$。设 $\overrightarrow{AG}=t\overrightarrow{AD}=\frac{t}{2}(\vec{a}+\vec{b})$。同理可算出使 $G$ 落在 $BE$ 上的 $t$ 值，并与 $CF$ 上的表示比较——两种表示一致，即证三线交于同一点。<span class="marginnote">三线共点的向量证法套路统一：<strong>设交点为两线的交点，证明它满足第三条线的条件</strong>。几何里的巧思被替换成「设点、表示、验证」的三步流程，人人可学、不依赖灵感。</span>

**重点：向量法证明的通用流程**——(1) 选好基底；(2) 把目标点表示为基底的线性组合；(3) 用「同向量表示唯一」或「点乘为零」列方程；(4) 回译成几何结论。这条流程几乎覆盖平面几何里平行、垂直、共线、共点、等长、成比例的全部命题类型。

## 5 例题精讲：向量法证三点共线

向量法在「三点共线」的证明上极其干净。看一道题：在 $\triangle ABC$ 中，$D$ 是 $AB$ 的中点，$E$ 是 $AC$ 上且 $AE=\frac13 AC$ 的点，$BE$ 与 $CD$ 交于 $F$，证明 $F$ 分 $BE$ 的比例为定值。

- **第一步，选基底**：设 $\vec{a}=\overrightarrow{AB}$、$\vec{b}=\overrightarrow{AC}$，则 $\overrightarrow{AD}=\frac12\vec{a}$、$\overrightarrow{AE}=\frac13\vec{b}$。
- **第二步，设分点**：$F$ 在 $BE$ 上，设 $\overrightarrow{BF}=t\overrightarrow{BE}=t(\vec{b}\cdot\frac13-\vec{a})=\frac t3\vec{b}-t\vec{a}$。$F$ 在 $CD$ 上，设 $\overrightarrow{CF}=s\overrightarrow{CD}=s(\frac12\vec{a}-\vec{b})=\frac s2\vec{a}-s\vec{b}$。
- **第三步，同向量列方程**：$\overrightarrow{AF}=\overrightarrow{AB}+\overrightarrow{BF}=\vec{a}+\frac t3\vec{b}-t\vec{a}=(1-t)\vec{a}+\frac t3\vec{b}$；又 $\overrightarrow{AF}=\overrightarrow{AC}+\overrightarrow{CF}=\vec{b}+\frac s2\vec{a}-s\vec{b}=\frac s2\vec{a}+(1-s)\vec{b}$。系数相等：$1-t=\frac s2$，$\frac t3=1-s$，解得 $s=\frac45$，$t=\frac35$。
- **第四步，读结论**：$F$ 分 $BE$ 为 $BF:FE=t:(1-t)=\frac35:\frac25=3:2$。

<span class="marginnote">「设分点 + 同向量系数相等」是向量法证明共线/交比的通用框架：<strong>两条路表示同一向量，基本定理保证系数唯一，于是列方程组解出分点比例</strong>。本题 $F$ 同时落在 $BE$ 与 $CD$ 上，用两条路线表示 $\overrightarrow{AF}$，系数对齐即得 $s,t$。这类「交点定比」问题在几何里要费辅助线，向量法只需代数解方程——<strong>「以算代证」是向量法的灵魂</strong>。</span>

**辨析｜易错点：** 一是**设分点后漏方向**——$F$ 在 $BE$ 上应设 $\overrightarrow{BF}=t\overrightarrow{BE}$，不是 $\overrightarrow{BF}=t\vec{b}$；二是**两条路线表示的不是同一向量**——都要化成 $\overrightarrow{AF}$（或同一基准），别一边 $\overrightarrow{BF}$ 一边 $\overrightarrow{CF}$；三是**系数对应错位**——$\vec{a}$ 的系数与 $\vec{b}$ 的系数分别对齐，别交叉配对。

## 6 小结

- **翻译手册**：共线 $\Leftrightarrow$ 成比例，垂直 $\Leftrightarrow$ 点乘为零，中点 $\Leftrightarrow$ 向量平均，分点 $\Leftrightarrow$ 系数配比。
- 证明通用流程：**选基底 → 表示 → 列方程 → 回译**。
- 数量积把「角度条件」翻译成「模长条件」，是垂直、等长类命题的利器。
- 三线共点等难题可用「设交点、验第三线」统一处理，不依赖辅助线巧思。

在下一节，我们从平面进入新的数系：当方程 $x^2=-1$ 也需要解时，数学造出了新数——**数系的扩充和复数的概念**。
