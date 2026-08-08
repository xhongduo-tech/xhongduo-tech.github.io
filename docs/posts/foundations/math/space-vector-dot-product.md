---
title: 空间向量的数量积运算
date: 2026-08-07
---

# 空间向量的数量积运算

<div class="epigraph">
<p>一个点乘，量出长度、角度与垂直。</p>
<footer>—— 向量方法的格言（Dot product in 3D）</footer>
</div>

<div class="article-byline">
<p>第一级 · 基础数学 ｜ 人教A版 选择性必修第一册 §1.1.2 ｜ 2026-08-07</p>
</div>

## 为什么从空间向量数量积开始

平面向量的数量积给了我们求夹角、长度、判定垂直的工具。在空间里，这些需求更迫切：求异面直线所成的角、求点到平面的距离、判定线线垂直——立体几何的三大度量问题，全都需要空间向量的数量积。好消息依然是：**数量积的定义、性质、运算律从平面到空间完全一致**，只是向量可以朝任何方向。<span class="marginnote">数量积是「维度无关」的又一次证明：$\vec{a}\cdot\vec{b}=|\vec{a}||\vec{b}|\cos\theta$ 在二维、三维乃至 $n$ 维都成立。这个「不变式」太重要了——它让空间中的长度、夹角、垂直问题统一成一个点乘公式，立体几何从此有了「代数化」的通用工具。</span> 学完本节，你就拥有了用向量法解立体几何度量问题的第一把钥匙。

## 1 空间向量的夹角与数量积

**空间向量的夹角**：已知非零向量 $\vec{a},\vec{b}$，作 $\overrightarrow{OA}=\vec{a}$、$\overrightarrow{OB}=\vec{b}$，则 $\angle AOB=\theta$（$0\le\theta\le\pi$）叫作 $\vec{a}$ 与 $\vec{b}$ 的夹角。当 $\theta=\frac{\pi}{2}$ 时，$\vec{a}\perp\vec{b}$。

**空间向量的数量积**：

$$
\vec{a}\cdot\vec{b}=|\vec{a}|\,|\vec{b}|\,\cos\theta
$$

结果是一个实数，读作「$\vec{a}$ 点乘 $\vec{b}$」。规定零向量与任何向量的数量积为 0。<span class="marginnote">定义与平面完全一样：数量积 = 长度 × 长度 × 夹角余弦。$|\vec{b}|\cos\theta$ 是 $\vec{b}$ 在 $\vec{a}$ 方向上的投影——「把 $\vec{b}$ 投影到 $\vec{a}$ 上再乘 $\vec{a}$ 的长度」。三维里投影的含义不变，只是向量可以从任何方向投影到另一条直线方向上。</span> 数量积的符号同样由 $\cos\theta$ 决定：锐角为正、钝角为负、垂直为零。

## 2 数量积的性质与运算律

空间数量积的**四条重要性质**（$\vec{a},\vec{b}$ 为非零向量）：

1. **垂直判定**：$\vec{a}\perp\vec{b}\iff\vec{a}\cdot\vec{b}=0$。
2. **同向与反向**：同向时 $\vec{a}\cdot\vec{b}=|\vec{a}||\vec{b}|$，反向时为 $-|\vec{a}||\vec{b}|$。
3. **模长公式**：$\vec{a}\cdot\vec{a}=|\vec{a}|^2$，即 $|\vec{a}|=\sqrt{\vec{a}\cdot\vec{a}}$。
4. **夹角公式**：$\cos\theta=\dfrac{\vec{a}\cdot\vec{b}}{|\vec{a}|\,|\vec{b}|}$。

**运算律**：满足交换律 $\vec{a}\cdot\vec{b}=\vec{b}\cdot\vec{a}$、数乘结合律 $(\lambda\vec{a})\cdot\vec{b}=\lambda(\vec{a}\cdot\vec{b})$、分配律 $(\vec{a}+\vec{b})\cdot\vec{c}=\vec{a}\cdot\vec{c}+\vec{b}\cdot\vec{c}$；**不满足消去律，没有结合律**。<span class="marginnote">与平面数量积完全一致：点乘把一个向量「压」成一个数，丢失方向信息，所以「$\vec{a}\cdot\vec{b}=\vec{a}\cdot\vec{c}$ 推不出 $\vec{b}=\vec{c}$」。这条「降维代价」的警告在空间里同样成立——凡是平面数量积不能做的事，空间里也不能做。</span> 由分配律可得**向量平方展开**：

$$
(\vec{a}+\vec{b})\cdot(\vec{a}+\vec{b})=|\vec{a}|^2+2\vec{a}\cdot\vec{b}+|\vec{b}|^2
$$

这是空间里求「和向量长度」的常用工具。

## 3 公式解析：用数量积求异面直线所成的角

数量积在立体几何里的第一个大用是**求异面直线所成的角**。设两条异面直线 $l_1,l_2$ 的方向向量分别为 $\vec{u},\vec{v}$，则它们所成的角 $\theta$ 满足

$$
\cos\theta=\frac{|\vec{u}\cdot\vec{v}|}{|\vec{u}|\,|\vec{v}|}
$$

拆三步理解：

- **第一步，识别角色**：异面直线本身不相交，但它们的方向向量可以「平移共起点」。$l_1,l_2$ 所成的角，等于方向向量 $\vec{u},\vec{v}$ 的夹角（或它的补角）。
- **第二步，为什么要加绝对值**：方向向量夹角 $\varphi\in[0,\pi]$，可能取到钝角；而异面直线所成的角定义为 $[0,\frac{\pi}{2}]$ 内的锐角（或直角）。所以取 $\cos$ 的**绝对值**，把钝角折回锐角：$\theta=\min(\varphi,\pi-\varphi)$。
- **第三步，落到公式**：由夹角公式 $\cos\varphi=\frac{\vec{u}\cdot\vec{v}}{|\vec{u}||\vec{v}|}$，加绝对值后即上式。**算出的 $\cos\theta$ 必非负，对应 $[0,\frac{\pi}{2}]$ 的角**——这就是所求的异面直线角。

<span class="marginnote">「取绝对值」是求异面角最关键的一步：忘记绝对值，可能得到钝角，那已经不是「异面直线所成的角」。反之，求二面角时<strong>不取绝对值</strong>，因为二面角可以是钝角。同一个点乘公式，加不加绝对值，取决于所求角度的范围——这是向量法求角最容易踩的坑。</span>

**辨析｜易错点：** 一是**忘记绝对值**——异面角取 $[0,\frac{\pi}{2}]$，必须把 $\cos$ 取绝对值；二是**把「直线垂直」当「方向向量夹角 $90^\circ$」**——两条异面直线垂直 $\iff \vec{u}\cdot\vec{v}=0$（不需要取绝对值的夹角恰好为直角）；三是**用点乘求长度时符号错误**——$|\vec{u}+\vec{v}|^2=|\vec{u}|^2+2\vec{u}\cdot\vec{v}+|\vec{v}|^2$，中间的 $2\vec{u}\cdot\vec{v}$ 可正可负，别当成恒为正。

## 4 应用：向量法求两点距离与垂直判定

数量积还统一了「距离」问题。设 $A,B$ 为空间两点，则

$$
|AB|=|\overrightarrow{AB}|=\sqrt{\overrightarrow{AB}\cdot\overrightarrow{AB}}
$$

任何「两点距离」都可以用「向量的自点乘开方」计算——这是空间直角坐标下距离公式的来源（下一节坐标表示将给出分量形式）。而「线线垂直」「线面垂直」「面面垂直」的向量判定都归结为点乘为零或方向向量关系：

- 线线垂直 $\iff$ 方向向量点乘为零。
- 线面垂直 $\iff$ 直线的方向向量与平面内**两条不共线**向量的点乘都为零。
- 面面垂直 $\iff$ 两平面的法向量点乘为零（法向量在 §1.4 引入）。

<span class="marginnote">「垂直问题全部化为点乘为零」——这是数量积对立体几何最彻底的贡献。有了它，判定垂直不再需要构造辅助线，只要算两个点乘。这种「把几何判定变成代数验证」的方法，正是整个《空间向量与立体几何》章的主旋律。</span>

## 5 例题精讲：数量积的坐标计算

空间数量积的考题，常与坐标结合求角与判定垂直。看一道题。

### 题目：求异面直线所成的角

正方体 $ABCD$-$A'B'C'D'$ 棱长为 1，求异面直线 $A'B$ 与 $B'C$ 所成的角。

**第一步，建系取方向向量**：以 $A$ 为原点，$A'(0,0,1)$，$B(1,0,0)$，$\overrightarrow{A'B}=B-A'=(1,0,-1)$；$B'(1,0,1)$，$C(1,1,0)$，$\overrightarrow{B'C}=C-B'=(0,1,-1)$。
**第二步，算点乘与模**：$\overrightarrow{A'B}\cdot\overrightarrow{B'C}=1\times0+0\times1+(-1)\times(-1)=1$；$|\overrightarrow{A'B}|=\sqrt2$，$|\overrightarrow{B'C}|=\sqrt2$。
**第三步，套公式**：$\cos\theta=\dfrac{|\overrightarrow{A'B}\cdot\overrightarrow{B'C}|}{|\overrightarrow{A'B}||\overrightarrow{B'C}|}=\dfrac{1}{\sqrt2\cdot\sqrt2}=\dfrac12$，$\theta=60^\circ$。

<span class="marginnote">求异面直线角的流程：<strong>建系 → 写两直线的方向向量 → 算点乘与模 → 套 $\cos\theta=\frac{|\vec u\cdot\vec v|}{|\vec u||\vec v|}$</strong>。注意<strong>取绝对值</strong>——异面直线所成的角在 $[0,\frac\pi2]$，算出的 $\cos$ 必须非负，对应锐角或直角。本题 $\cos\theta=\frac12$，$\theta=60^\circ$，是正方体里经典的异面角结论。</span>

**辨析｜易错点（补充）：** 一是**方向向量写反**——$\overrightarrow{A'B}=B-A'$，终点减起点，写成 $A'-B$ 得相反向量（点乘取绝对值后不变，但垂直判定会错）；二是**忘取绝对值**——异面角取 $[0,\frac\pi2]$，$\cos$ 必须取绝对值；三是**模长算错**——$(1,0,-1)$ 的模是 $\sqrt2$，别漏平方项或漏根号。

## 6 小结

- **空间数量积**：$\vec{a}\cdot\vec{b}=|\vec{a}||\vec{b}|\cos\theta$，定义与平面一致。
- 性质：垂直 $\iff$ 点乘为零；$|\vec{a}|=\sqrt{\vec{a}\cdot\vec{a}}$；$\cos\theta=\dfrac{\vec{a}\cdot\vec{b}}{|\vec{a}||\vec{b}|}$。
- 运算律：交换、数乘、分配；**无消去律、无结合律**。
- **求异面直线角**：$\cos\theta=\dfrac{|\vec{u}\cdot\vec{v}|}{|\vec{u}||\vec{v}|}$——**记得取绝对值**。
- 距离 $|AB|=\sqrt{\overrightarrow{AB}\cdot\overrightarrow{AB}}$；垂直全部化为点乘为零。

在下一节，我们找出空间向量世界的「最小拼图」：**空间向量基本定理**——空间向量的基底与坐标。
