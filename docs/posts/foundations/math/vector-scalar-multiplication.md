---
title: 平面向量的数乘运算
date: 2026-08-07
---

# 平面向量的数乘运算

<div class="epigraph">
<p>给我一个支点，我可以撬起整个地球。</p>
<footer>—— 阿基米德（Archimedes）</footer>
</div>

<div class="article-byline">
<p>第一级 · 基础数学 ｜ 人教A版 必修第二册 §6.2.3 ｜ 2026-08-07</p>
</div>

## 为什么从数乘向量开始

向量能做加法，那「一个向量乘以一个数」有意义吗？当然有：你拉着行李箱，把拉力 $F$ 加倍，就是 $2F$；把拉力反向，就是 $-F$。**数乘向量（scalar multiplication）**就是向量的「放大缩小与反向」，它把「长度」这个标量信息灌进向量——虽然只动大小与方向，却为向量世界装上了「比例」这把尺子。<span class="marginnote">阿基米德的杠杆名言藏着数乘的思想：力臂放大 $k$ 倍，效果就放大 $k$ 倍——方向没变，大小变了。数乘就是这种「纯比例变换」在向量上的化身。</span> 更关键的是，数乘与加法合在一起，构成向量运算的完整骨架：**向量的线性运算**。整章后面的基本定理、坐标表示、几何应用，全都建立在这两个运算之上。

## 1 数乘的定义与几何意义

**数乘向量（scalar multiplication）**：实数 $\lambda$ 与向量 $\vec{a}$ 的乘积是一个向量，记作 $\lambda\vec{a}$，满足：

**长度**：$|\lambda\vec{a}|=|\lambda||\vec{a}|$（长度放缩 $|\lambda|$ 倍）；
**方向**：当 $\lambda>0$ 时，$\lambda\vec{a}$ 与 $\vec{a}$ 同向；当 $\lambda<0$ 时，反向；当 $\lambda=0$ 时，$\lambda\vec{a}=\vec{0}$。

由此立刻得到两条极常用的结论：$(-1)\vec{a}=-\vec{a}$（相反向量就是数乘 $-1$），以及 $\dfrac{\vec{a}}{|\vec{a}|}$ 是与 $\vec{a}$ 同向的单位向量。<span class="marginnote">「把一个向量单位化」——除以自己的长度得到同向单位向量——这个操作在物理里（分解方向）、在深度学习的归一化里（L2 范数归一化）无处不在。单位化本质就是数乘 $\frac{1}{|\vec{a}|}$。</span>

### 运算律

数乘与数的乘法一样，满足三条律，其中 $\lambda,\mu\in\mathbb{R}$：

$$
\lambda(\mu\vec{a})=(\lambda\mu)\vec{a}, \qquad
(\lambda+\mu)\vec{a}=\lambda\vec{a}+\mu\vec{a}, \qquad
\lambda(\vec{a}+\vec{b})=\lambda\vec{a}+\lambda\vec{b}
$$

第一条是**结合律**（数与数先乘），后两条是**分配律**（一个是对向量分配、一个是对数分配）。有了这些律，含向量的代数式可以像普通多项式一样展开、合并同类项——**运算的代数化是向量能被「算」出来的前提**。

## 2 共线定理：数乘的反向解读

数乘回答了一个深刻的问题：**两个向量什么时候在同一条直线上？**

**平面向量共线定理**：向量 $\vec{a}$（$\vec{a}\neq\vec{0}$）与向量 $\vec{b}$ 共线，当且仅当存在唯一实数 $\lambda$，使得 $\vec{b}=\lambda\vec{a}$。<span class="marginnote">这个定理是「数乘」最漂亮的收获：<strong>共线 ⇔ 成比例</strong>。方向相同或相反的几何关系，被翻译成了「一个向量是另一个的数倍」的代数关系。几何问题代数化，从此有了第一个标准工具。</span>

**辨析｜易错点：** 定理里 $\vec{a}\neq\vec{0}$ 的条件绝不能省。如果 $\vec{a}=\vec{0}$，那么「存在 $\lambda$ 使 $\vec{b}=\lambda\vec{a}$」只能推出 $\vec{b}=\vec{0}$，无法描述非零向量与零向量的关系。而零向量与任意向量共线，这条要单独记忆。另一个常见错误是忽略「唯一」：一旦 $\vec{a}\neq\vec{0}$ 且 $\vec{b}$ 与 $\vec{a}$ 共线，比例 $\lambda$ 是**唯一的**，$\lambda=\dfrac{\vec{b}\text{ 的长度（带符号）}}{\vec{a}\text{ 的长度}}$，同向为正、反向为负。

共线定理的应用非常直接：判定三点 $A,B,C$ 共线，只需验证 $\overrightarrow{AB}=\lambda\overrightarrow{AC}$。例如在平行四边形 $ABCD$ 中，$M$ 为对角线交点，则 $\overrightarrow{AM}=\frac{1}{2}\overrightarrow{AC}$——一条数乘式子就点破了「中点」的全部几何信息。

## 3 公式解析：$\lambda\vec{a}$ 的三重读法

把数乘式子 $b=\lambda\vec{a}$ 做三步拆解，看清它承载的信息：

**第一步，读长度**：$|\vec{b}|=|\lambda||\vec{a}|$。$|\lambda|$ 是放缩倍数，与方向无关。若 $|\lambda|>1$ 则拉长，若 $|\lambda|<1$ 则压缩。
**第二步，读方向**：$\lambda$ 的正负决定方向同向或反向。**符号只告诉我们方向，绝对值只告诉我们大小**——两者完全解耦，互不干扰。
**第三步，读共线**：$\vec{b}=\lambda\vec{a}$ 一旦成立，$\vec{a}$、$\vec{b}$ 必共线。反过来说，共线是「成比例」的充要条件。这条「双向通道」使得几何中的平行关系可以被代数等式完全替代——这正是解析几何「以数代形」的核心理念。

结合 $\vec{a}\cdot|\vec{a}|$ 的例子再读一遍：把 $\vec{a}$ 写成「长度 × 同向单位向量」即 $\vec{a}=|\vec{a}|\cdot\dfrac{\vec{a}}{|\vec{a}|}$，这一分解把「大小」与「方向」两个信息拆开存放——**向量的长度信息住在数乘的系数里，方向信息住在单位向量里**，拆解之后，后续的数量积、坐标分解才有了抓手。<span class="marginnote">「大小与方向解耦」是向量最深刻的设计：同一个向量可以任你分解成「标量 × 单位向量」。这思想延伸出去，就是线性代数里「向量 = 基向量 × 坐标」的雏形——下一章的基本定理正是它的正式版。</span>

## 4 数乘在平面几何中的应用

数乘向量在几何证明里极其实用，因为它可以直接「表达位置关系」。看一个经典例子：在 $\triangle ABC$ 中，$D$ 是 $BC$ 上一点且 $\overrightarrow{BD}=2\overrightarrow{DC}$，用 $\vec{a}=\overrightarrow{AB}$、$\vec{b}=\overrightarrow{AC}$ 表示 $\overrightarrow{AD}$。

由 $\overrightarrow{BD}=2\overrightarrow{DC}$ 得 $\overrightarrow{BD}=\frac{2}{3}\overrightarrow{BC}$，于是

$$
\overrightarrow{AD}=\overrightarrow{AB}+\overrightarrow{BD}=\vec{a}+\frac{2}{3}(\vec{b}-\vec{a})=\frac{1}{3}\vec{a}+\frac{2}{3}\vec{b}
$$

这里 $\frac{1}{3}$ 与 $\frac{2}{3}$ 正是分点位置的体现：**「$\vec{a}$ 前系数小、$\vec{b}$ 前系数大」对应着 $D$ 更靠近 $C$ 这一几何事实**。一般地，若 $D$ 分 $BC$ 为 $\overrightarrow{BD}:\overrightarrow{DC}=m:n$，则 $\overrightarrow{AD}=\dfrac{n}{m+n}\vec{a}+\dfrac{m}{m+n}\vec{b}$——这个「加权平均」的结构在下一节的定比分点与后面平面向量基本定理中会再次出现。<span class="marginnote">注意两个系数之和 $=\frac{1}{3}+\frac{2}{3}=1$，且 $D$ 就在直线 $BC$ 上。<strong>「系数和为 1」的共线判别法</strong>：点 $P$ 在直线 $BC$ 上当且仅当 $\overrightarrow{AP}=(1-t)\vec{b}+t\vec{c}$——这是共线定理最常用的一种包装，几何里判定三点共线就靠它。</span>

## 5 例题精讲：数乘与共线定理的应用

数乘向量与共线定理的考题，集中在「三点共线」「定比分点」与「向量的拆分」。

### 应用一：三点共线的判定

$A,B,C$ 三点共线 $\iff \overrightarrow{AB}=\lambda\overrightarrow{AC}$（存在实数 $\lambda$）。更常用的形式：$\overrightarrow{OC}=(1-t)\overrightarrow{OA}+t\overrightarrow{OB}$——「系数和为 1」的共线判法。例：$P$ 在直线 $AB$ 上当且仅当 $\overrightarrow{OP}=x\overrightarrow{OA}+y\overrightarrow{OB}$ 且 $x+y=1$。

### 应用二：定比分点

若 $P$ 分 $\overrightarrow{AB}$ 为 $AP:PB=m:n$，则 $\overrightarrow{OP}=\dfrac{n}{m+n}\overrightarrow{OA}+\dfrac{m}{m+n}\overrightarrow{OB}$——「离谁近，谁的系数大」。中点取 $m=n$：$\overrightarrow{OP}=\frac12(\overrightarrow{OA}+\overrightarrow{OB})$。

### 应用三：向量的「长度 × 方向」拆分

$\vec{a}=|\vec{a}|\cdot\dfrac{\vec{a}}{|\vec{a}|}$——把向量拆成「标量（长度）× 单位向量（方向）」。这个拆分让「方向」与「大小」解耦，是物理里分解力、后续坐标化的基础。

<span class="marginnote">「系数和为 1」的共线判法是数乘最漂亮的推论：<strong>$P$ 在直线 $AB$ 上 ⇔ $\overrightarrow{OP}=x\overrightarrow{OA}+y\overrightarrow{OB}$ 且 $x+y=1$</strong>。它把「三点共线」的几何判定变成「系数和的代数检查」。定比分点公式则是它的定量版——分点比例直接对应系数配比。<strong>「离谁近谁的系数大」这个直觉，配合系数和为 1 的检查，定比分点题基本不会错</strong>。</span>

**辨析｜易错点（补充）：** 一是**共线判定漏「系数和为 1」**——若 $x+y\neq1$，$P$ 不一定在直线 $AB$ 上；二是**定比分点系数写反**——$AP:PB=m:n$ 时 $OA$ 的系数是 $\frac n{m+n}$（离 $B$ 的比例），别配反；三是**单位向量方向**——$\frac{\vec{a}}{|\vec{a}|}$ 与 $\vec{a}$ 同向，$\vec{a}$ 反向时是 $-\frac{\vec{a}}{|\vec{a}|}$。

## 7 自测与要点回顾

**自测**：已知 $A,B,C$ 三点满足 $\overrightarrow{OC}=3\overrightarrow{OA}-2\overrightarrow{OB}$，判断 $A,B,C$ 的位置关系。

**答案**：$\overrightarrow{OC}=3\overrightarrow{OA}-2\overrightarrow{OB}$ 可改写为 $\overrightarrow{OC}=3\overrightarrow{OA}+(-2)\overrightarrow{OB}$，系数和 $3+(-2)=1$——由「系数和为 1」的共线判法，$C$ 在直线 $AB$ 上，即 $A,B,C$ 三点共线。

**要点回顾**：共线 ⇔ 成比例（非零向量）；「系数和为 1」判三点共线；定比分点「离谁近谁的系数大」；数乘把「大小」与「方向」解耦——长度乘 $|\lambda|$、方向由 $\lambda$ 正负决定。

## 8 小结

- **数乘** $\lambda\vec{a}$：长度变 $|\lambda|$ 倍，方向由 $\lambda$ 正负决定，$\lambda=0$ 时为零向量。
- 运算律：结合律与两条分配律，使向量式子可以像多项式一样展开。
- **共线定理**：$\vec{a}\neq\vec{0}$ 时，$\vec{a},\vec{b}$ 共线 $\Leftrightarrow$ 存在唯一 $\lambda$ 使 $\vec{b}=\lambda\vec{a}$。
- 数乘把「大小」装进系数、「方向」装进单位向量；分点公式 $\overrightarrow{AD}=\dfrac{n}{m+n}\vec{a}+\dfrac{m}{m+n}\vec{b}$ 是「系数和为 1」共线判法的来源。

在下一节，我们引入向量的第二种乘法——它不再产出向量，而是产出一个数：**平面向量的数量积**。
