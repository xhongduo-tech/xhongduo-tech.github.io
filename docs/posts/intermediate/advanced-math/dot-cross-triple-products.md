---
title: 数量积、向量积与混合积
date: 2026-08-07
---

# 数量积、向量积与混合积

<div class="epigraph">
<p>向量的乘法，给出两种答案：一个数，或一个向量。</p>
<footer>—— 威廉 · 罗恩 · 哈密顿（William Rowan Hamilton）</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等数学 ｜ 同济《高等数学》下册 §8.2 ｜ 2026-08-07</p>
</div>

## 为什么从数量积、向量积与混合积开始

上一节学了向量的加减与数乘，但「两个向量相乘」还没定义。实际需要两种截然不同的「乘法」：**数量积**（结果是一个数，衡量「同向程度」与功）、**向量积**（结果是一个向量，垂直且衡量「平行程度」与力矩）。再加上**混合积**（三个向量的「体积」），这三种运算构成了空间向量的全部「乘法」。它们不只是代数练习——功 $W = \mathbf{F}\cdot\mathbf{s}$、磁场力 $\mathbf{F} = q\mathbf{v}\times\mathbf{B}$、平行六面体体积，全是这三种运算的物理化身。<span class="marginnote">哈密顿发明四元数时引入了「向量」的雏形，他区分了「数量部分」与「向量部分」——这正是数量积与向量积的由来。<strong>点积回答「两向量有多同向」，叉积回答「两向量有多垂直（及转向）」</strong>，两种乘法互补地度量了向量的相对几何。</span>

## 1 数量积（点积）

**数量积（点积，dot product）**：

$$\mathbf{a} \cdot \mathbf{b} = |\mathbf{a}|\,|\mathbf{b}|\cos\theta$$

其中 $\theta$ 是两向量的夹角（$0 \le \theta \le \pi$）。结果是一个**数**。坐标形式：

$$\mathbf{a} \cdot \mathbf{b} = a_xb_x + a_yb_y + a_zb_z$$

**性质**：交换律、分配律；$\mathbf{a}\cdot\mathbf{a} = |\mathbf{a}|^2$；$\mathbf{a} \perp \mathbf{b} \iff \mathbf{a}\cdot\mathbf{b} = 0$（两非零向量）。

**重点：点积是「投影的放大」**——$\mathbf{a}\cdot\mathbf{b} = |\mathbf{a}| \cdot (|\mathbf{b}|\cos\theta)$ 是「$\mathbf{a}$ 的长度 × $\mathbf{b}$ 在 $\mathbf{a}$ 方向上的投影长度」。它把「同向程度」量化为一个数。<span class="marginnote">投影的直觉：$\mathbf{b}\cdot\mathbf{e}_a = |\mathbf{b}|\cos\theta$ 是 $\mathbf{b}$ 在 $\mathbf{a}$ 方向的投影。点积 = 投影 × 长度，这使它在「做功」（力沿位移方向的分量 × 位移）与「相似度」（向量内积越大越同向）中成为核心工具。余弦相似度 $\cos\theta = \frac{\mathbf{a}\cdot\mathbf{b}}{|\mathbf{a}||\mathbf{b}|}$ 是信息检索与机器学习里最常用的相似度度量。</span>

## 2 向量积（叉积）

**向量积（叉积，cross product）**：结果是一个**向量** $\mathbf{c} = \mathbf{a}\times\mathbf{b}$，满足

$$|\mathbf{c}| = |\mathbf{a}|\,|\mathbf{b}|\sin\theta$$

方向：$\mathbf{c}$ **垂直于** $\mathbf{a}$ 与 $\mathbf{b}$ 所在的平面，指向由**右手定则**确定（四指从 $\mathbf{a}$ 弯向 $\mathbf{b}$，拇指指向 $\mathbf{c}$）。

坐标形式（借助行列式）：

$$\mathbf{a}\times\mathbf{b} = \begin{vmatrix}\mathbf{i} & \mathbf{j} & \mathbf{k}\\ a_x & a_y & a_z\\ b_x & b_y & b_z\end{vmatrix} = (a_yb_z - a_zb_y,\ a_zb_x - a_xb_z,\ a_xb_y - a_yb_x)$$

**性质**：**反交换律** $\mathbf{a}\times\mathbf{b} = -\mathbf{b}\times\mathbf{a}$；$\mathbf{a}\times\mathbf{a} = \mathbf{0}$；$\mathbf{a}\parallel\mathbf{b} \iff \mathbf{a}\times\mathbf{b} = \mathbf{0}$；$|\mathbf{a}\times\mathbf{b}|$ = 以 $\mathbf{a},\mathbf{b}$ 为邻边的**平行四边形面积**。<span class="marginnote">叉积的几何意义是「面积向量」：模是平行四边形面积，方向垂直于平行四边形所在平面。右手定则的关键：$\mathbf{a}\times\mathbf{b}$ 与 $\mathbf{b}\times\mathbf{a}$ 方向相反（反交换律），顺序不能反。物理里力矩 $\mathbf{M} = \mathbf{r}\times\mathbf{F}$、磁场洛伦兹力都用叉积——「旋转效果」天然是叉积的语言。</span>

## 3 混合积

**混合积**：$(\mathbf{a}\times\mathbf{b})\cdot\mathbf{c}$，先叉积后点积，结果是一个数。坐标形式是三阶行列式：

$$(\mathbf{a}\times\mathbf{b})\cdot\mathbf{c} = \begin{vmatrix}a_x & a_y & a_z\\ b_x & b_y & b_z\\ c_x & c_y & c_z\end{vmatrix}$$

**几何意义**：$|(\mathbf{a}\times\mathbf{b})\cdot\mathbf{c}|$ = 以 $\mathbf{a},\mathbf{b},\mathbf{c}$ 为棱的**平行六面体体积**。<span class="marginnote">混合积「体积」的直觉：$|\mathbf{a}\times\mathbf{b}|$ 是底面积，$\mathbf{c}$ 在法向的投影 $|\mathbf{c}\cos\varphi|$ 是高，乘积即体积。若混合积为 0，则三向量共面（体积为零）——这是「共面判定」的代数判据，也是三阶行列式几何意义的来源。</span>

**三个向量共面** ⟺ $(\mathbf{a}\times\mathbf{b})\cdot\mathbf{c} = 0$。

## 4 公式解析：三种积的计算与几何

设 $\mathbf{a} = (1,0,0)$、$\mathbf{b} = (0,1,0)$、$\mathbf{c} = (0,0,1)$（坐标轴单位向量）：

**第一步，算点积**：$\mathbf{a}\cdot\mathbf{b} = 1\cdot0 + 0\cdot1 + 0\cdot0 = 0$——正交，夹角 $90°$。
**第二步，算叉积**：$\mathbf{a}\times\mathbf{b} = \begin{vmatrix}\mathbf{i}&\mathbf{j}&\mathbf{k}\\1&0&0\\0&1&0\end{vmatrix} = \mathbf{k}$——$\mathbf{i}\times\mathbf{j}=\mathbf{k}$，垂直且符合右手定则。
**第三步，算混合积**：$(\mathbf{a}\times\mathbf{b})\cdot\mathbf{c} = \mathbf{k}\cdot\mathbf{k} = 1$——单位立方体的体积。
**第四步，验证反交换**：$\mathbf{b}\times\mathbf{a} = -\mathbf{k}$，方向相反。

**关键**：三种积可以看成「从两个向量到数/向量」的三种投影：点积量「同向」，叉积量「垂直 + 面积」，混合积量「体积」。三个量合起来，完全刻画了三向量在空间的相对姿态。

## 5 三种积的应用

**功与功率**：$W = \mathbf{F}\cdot\mathbf{s}$、$P = \mathbf{F}\cdot\mathbf{v}$——点积把「力沿位移方向的投影」量化。<span class="marginnote">点积的机器学习用法：<strong>余弦相似度</strong> $\cos\theta$ 忽略模长、只比方向，是文本检索、嵌入向量相似度、推荐系统里最常用的度量。你在第二级《线性代数》与第三级《信息检索》都会与它重逢。</span>
- **力矩与角动量**：$\mathbf{M} = \mathbf{r}\times\mathbf{F}$、$\mathbf{L} = \mathbf{r}\times\mathbf{p}$——叉积描述「旋转效果」。
- **平行四边形与三角形面积**：$S = |\mathbf{a}\times\mathbf{b}|$。
- **共面与体积判定**：混合积为 0 判定共面，绝对值是平行六面体体积。
- **叉积与机器学习**：三维几何计算（法向量、旋转）在计算机图形学与机器人学里大量使用叉积。

## 7 数值算例：用混合积判定共面

判定四点 $A(1,0,0)$、$B(0,1,0)$、$C(0,0,1)$、$D(1,1,1)$ 是否共面。

**第一步，作三向量**：$\overrightarrow{AB} = (-1,1,0)$，$\overrightarrow{AC} = (-1,0,1)$，$\overrightarrow{AD} = (0,1,1)$。
**第二步，算混合积**：$(\overrightarrow{AB}\times\overrightarrow{AC})\cdot\overrightarrow{AD}$，先算 $\overrightarrow{AB}\times\overrightarrow{AC} = (1,1,1)$，再点积 $\overrightarrow{AD}$：$0\cdot1 + 1\cdot1 + 1\cdot1 = 2 \neq 0$。
**第三步，判定**：混合积非零 ⇒ 三向量不共面 ⇒ 四点不共面（$D$ 在 $\triangle ABC$ 平面外）。

**配套观察**：混合积 $\frac{1}{6}$ 的绝对值是四面体 $ABCD$ 的体积——当 $D$ 与 $A,B,C$ 共面时体积为零，混合积为零。**「混合积 = 0」既是共面判据，也是体积为零的翻译**。

## 8 对照表：三种向量的积

| 运算 | 结果 | 度量 | 物理例 |
| --- | --- | --- | --- |
| 数量积 $\mathbf{a}\cdot\mathbf{b}$ | 数 | 同向程度（投影） | 功 $W = \mathbf{F}\cdot\mathbf{s}$ |
| 向量积 $\mathbf{a}\times\mathbf{b}$ | 向量 | 垂直 + 面积 | 力矩 $\mathbf{M} = \mathbf{r}\times\mathbf{F}$ |
| 混合积 $(\mathbf{a}\times\mathbf{b})\cdot\mathbf{c}$ | 数 | 体积 | 四面体体积 |

## 9 常见错误自查清单

| 错误 | 正确做法 |
| --- | --- |
| 叉积交换顺序 | $\mathbf{a}\times\mathbf{b} = -\mathbf{b}\times\mathbf{a}$，顺序不可换 |
| 点积忘 $\cos\theta$ 或忘坐标公式 | 两种形式等价，任选其一 |
| 混合积顺序打乱 | 先叉后点，括号位置决定结果符号 |
| 共面判定只算一个叉积 | 用混合积完整判定，单看叉积不够 |

## 10 三种积与机器学习/计算机图形

三种向量积是三维计算的原子操作：

- **法向量计算**：平面 $\mathbf{a}\times\mathbf{b}$ 给出法向，是图形学光照、表面朝向的基础；
- **余弦相似度**：$\frac{\mathbf{a}\cdot\mathbf{b}}{|\mathbf{a}||\mathbf{b}|}$ 是文本嵌入、推荐系统的核心相似度；
- **几何变换**：旋转、反射在三维图形引擎里用叉积与点积实现。

你在本节学的「点积量同向、叉积量面积、混合积量体积」，是理解三维几何计算与向量检索系统的钥匙。

## 11 小结

- **数量积**：$\mathbf{a}\cdot\mathbf{b} = |\mathbf{a}||\mathbf{b}|\cos\theta$，结果是数，$\perp \iff$ 点积为 0。
- **向量积**：$\mathbf{a}\times\mathbf{b}$ 垂直两向量、右手定则定向，模 = 平行四边形面积，$\parallel \iff$ 叉积为 0。
- **混合积**：$(\mathbf{a}\times\mathbf{b})\cdot\mathbf{c}$ 是平行六面体体积，为 0 ⟺ 三向量共面。
- 点积量同向、叉积量垂直与面积、混合积量体积。
- 应用：功、力矩、面积、共面判定、余弦相似度、图形学法向量。

在下一节，我们将用向量工具研究空间中最基本的几何对象——**平面及其方程**。
