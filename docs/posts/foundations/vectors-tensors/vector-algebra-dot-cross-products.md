---
title: 向量代数：点积、叉积与多重积
date: 2026-08-11
---

# 向量代数：点积、叉积与多重积

<div class="epigraph">
<p>一切可计量的东西都可当作向量来处理。</p>
<footer>—— 詹姆斯 · 克拉克 · 麦克斯韦（James Clerk Maxwell）</footer>
</div>

<div class="article-byline">
<p>第一级 · 基础科学 · 向量与张量初步 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从向量代数开始

这个博客的终点是大模型。而大模型的第一性操作，是**把词、句子、图像都变成一串数字向量，然后用点积去比较它们像不像**。Transformer 的自注意力分数是 query 与 key 的向量内积，embedding 相似度常由余弦相似度（归一化点积）度量。可以说，**理解点积，就理解了深度学习一半的几何直觉**。

但向量远不止「给机器学习用的数据结构」。它是物理学家与工程师的母语：力、速度、电场强度、角动量……一切「既有大小又有方向」的量都是向量。本专题沿着 Arfken《Mathematical Methods for Physicists》第一章的路线，从三维空间里的向量代数讲起，一路走到张量。这一篇，我们先把三种最重要的运算——点积、叉积、多重积——彻底讲清楚。<span class="marginnote">Arfken, Weber &amp; Harris, "Mathematical Methods for Physicists" (7th ed.) §1.1–§1.7 覆盖本讲全部内容；Boas 第三版第 3 章与之平行。</span>

## 1 向量：从几何对象到坐标数组

**向量（vector）**：既有大小又有方向的量，用带箭头的线段 $\vec{a}$（或粗体 $\mathbf a$）表示。<span class="marginnote">「向量」一词源自拉丁语 vehere（搬运）。Hamilton 于 1843 年在都柏林桥上顿悟四元数（含虚部 i、j、k），Gibbs 与 Heaviside 随后从中提炼出现代向量分析体系。</span>

向量的第一个关键思想是**与起点无关**：平移不改变向量本身，只改变它的「位置」。因此平面向量 $\vec{a}$ 可以用它在基 $\{\mathbf e_1, \mathbf e_2\}$ 下的两个分量唯一表示，

$$
\mathbf a = a_1 \mathbf e_1 + a_2 \mathbf e_2 \quad\longleftrightarrow\quad \mathbf a = (a_1, a_2)
$$

分量表示把几何对象翻译成数字，一切运算都变成对数字的运算——这就是「线性代数」的起点。同一向量在不同坐标系下分量不同，但**向量本身不变**；分量如何随坐标变换而变，正是本专题后半部分「协变、逆变」两讲的伏笔。

两个基本运算随之而来：

- **加法** $\mathbf a + \mathbf b = (a_1+b_1, a_2+b_2)$：几何上遵循平行四边形法则。
- **数乘** $c\,\mathbf a = (c a_1, c a_2)$：拉伸（$c>1$）或压缩（$0<c<1$），$c<0$ 时反向。

**辨析｜易错点：** 向量加法满足交换律与结合律，这是它与「位移叠加」一一对应的基础；但向量与标量、向量与向量之间的「乘法」却有多种互不相容的版本——点积、叉积、外积、张量积。搞混它们，是初学向量最常见的灾难源头。

## 2 点积：把「同向程度」变成数字

**点积（dot product，内积）**：两个向量的模乘以夹角余弦，

$$
\mathbf a \cdot \mathbf b = |\mathbf a|\,|\mathbf b|\cos\theta
$$

其中 $\theta \in [0,\pi]$ 是两向量的夹角。在直角坐标基下，点积还有更便于计算的形式：

$$
\mathbf a \cdot \mathbf b = a_1 b_1 + a_2 b_2 + a_3 b_3
$$

两个定义殊途同归，后者是前者的分量翻译。<span class="marginnote">坐标形式的正确性仰仗基向量 $\mathbf e_i$ 互相垂直且长度为 1（正交归一基）。基若不正交，点积公式要引入度量矩阵——那是本专题《正交曲线坐标》一讲的内容。</span>

由点积立即可得三个常用推论：

- **求夹角**：$\cos\theta = \dfrac{\mathbf a\cdot\mathbf b}{|\mathbf a||\mathbf b|}$——自注意力、余弦相似度的数学内核。
- **垂直判据**：$\mathbf a \perp \mathbf b \iff \mathbf a \cdot \mathbf b = 0$。这是「正交」在解析几何里的精确翻译。
- **投影**：$\mathbf a$ 在 $\mathbf b$ 方向上的投影长度为 $\mathbf a \cdot \hat{\mathbf b} = \dfrac{\mathbf a \cdot \mathbf b}{|\mathbf b|}$——把向量拆解成「平行分量 + 垂直分量」的基础。

点积同时给出了重要不等式。**柯西-施瓦茨不等式（Cauchy–Schwarz inequality）**：

$$
|\mathbf a \cdot \mathbf b| \le |\mathbf a|\,|\mathbf b|
$$

它来自 $|\cos\theta|\le 1$。虽然证明只有一行，但它几乎撑起半边分析数学：向量的夹角概念、范数的三角不等式、乃至 Fourier 级数的收敛性都仰仗它。

## 3 叉积：创造「第三个方向」的乘法

叉积只定义在三维空间，它把一个平面区域的信息编码成法向量：

$$
\mathbf a \times \mathbf b = \mathbf n\, |\mathbf a|\,|\mathbf b|\sin\theta
$$

其中 $\mathbf n$ 是由右手定则确定方向的单位法向量。<span class="marginnote">叉积的发明动机很物理：力矩 $\boldsymbol\tau = \mathbf r\times\mathbf F$、角动量 $\mathbf L = \mathbf r\times\mathbf p$、洛伦兹力 $\mathbf F = q\mathbf v\times\mathbf B$ 都是两个向量的叉积。</span>右手定则——四指从 $\mathbf a$ 弯向 $\mathbf b$，拇指方向即 $\mathbf n$——意味着叉积是**反对称**的：

$$
\mathbf b \times \mathbf a = -\mathbf a \times \mathbf b
$$

分量形式是最实用的计算工具。若 $\mathbf a = (a_1,a_2,a_3)$，$\mathbf b = (b_1,b_2,b_3)$，则

$$
\mathbf a \times \mathbf b = (a_2 b_3 - a_3 b_2,\; a_3 b_1 - a_1 b_3,\; a_1 b_2 - a_2 b_1)
$$

几何上，$|\mathbf a \times \mathbf b| = |\mathbf a||\mathbf b|\sin\theta$ 恰是 $\mathbf a,\mathbf b$ 张成的**平行四边形面积**。平行判据随即而来：$\mathbf a \parallel \mathbf b \iff \mathbf a\times\mathbf b = \mathbf 0$。

**辨析｜易错点：** 点积是交换的（$\mathbf a\cdot\mathbf b=\mathbf b\cdot\mathbf a$），叉积是反对称的（$\mathbf a\times\mathbf b=-\mathbf b\times\mathbf a$）。对称性的天壤之别，根源在于点积是「标量结果」、叉积是「带手性的向量结果」。把一个记成另一个，后续物理公式会整体翻车。

## 4 多重积：三个向量的三幕剧

把三种「积」组合起来，得到三个向量的多重积，一共两种：

**标量三重积（scalar triple product）**——先叉后点：

$$
\mathbf a \cdot (\mathbf b \times \mathbf c) = \begin{vmatrix} a_1 & a_2 & a_3 \\ b_1 & b_2 & b_3 \\ c_1 & c_2 & c_3 \end{vmatrix}
$$

它的几何意义是**平行六面体体积**（可能带符号）。行列式的三条性质对应它的三条恒等式：任意交换两个向量变号；循环置换不变 $\mathbf a\cdot(\mathbf b\times\mathbf c)=\mathbf b\cdot(\mathbf c\times\mathbf a)=\mathbf c\cdot(\mathbf a\times\mathbf b)$；三向量共面 $\iff$ 三重积为零。

**向量三重积（vector triple product）**——先叉后叉：

$$
\mathbf a \times (\mathbf b \times \mathbf c) = \mathbf b\,(\mathbf a\cdot\mathbf c) - \mathbf c\,(\mathbf a\cdot\mathbf b)
$$

这就是著名的 **BAC–CAB 公式**。<span class="marginnote">记忆口诀：back cab——结果沿「外侧两向量」张成，系数分别是 a 与另一向量的点积。</span>它说明 $\mathbf a\times(\mathbf b\times\mathbf c)$ 落在 $\mathbf b,\mathbf c$ 张成的平面内，因此叉积**不满足结合律**：$(\mathbf a\times\mathbf b)\times\mathbf c$ 是另一回事。

## 5 公式解析：叉积分量公式为什么是那个样子

$$

(\mathbf a \times \mathbf b)_i = \sum_{j,k} \varepsilon_{ijk}\, a_j b_k

$$

这个式子用**Levi-Civita 符号** $\varepsilon_{ijk}$ 写出了叉积的每个分量。逐项拆解：

- **第一步，认识 $\varepsilon_{ijk}$**：当 $(i,j,k)$ 是 $(1,2,3)$ 的偶置换时取 $+1$，奇置换取 $-1$，任何两个指标相同取 $0$。它是「有序三指标」的方向记号。
- **第二步，取第一个分量 $i=1$**：只有 $j,k$ 满足 $\varepsilon_{1jk}\ne 0$ 的项存活，即 $(j,k)=(2,3)$ 与 $(3,2)$，分别给出 $+a_2 b_3$ 与 $-a_3 b_2$。于是 $(\mathbf a\times\mathbf b)_1 = a_2b_3 - a_3b_2$，与第 3 节分量公式一致。
- **第三步，几何直觉**：$\sin\theta$ 的符号需要「手性」编码，$\varepsilon_{ijk}$ 正是把右手定则数字化的最小装置。

这条公式的价值远超叉积本身：它是本专题后文 Levi-Civita 记号、赝张量、乃至行列式理论的**统一骨架**。记住它，等于为整条张量之路先铺了一块基石。<span class="marginnote">同一个 $\varepsilon$ 还能压缩标量三重积：$\mathbf a\cdot(\mathbf b\times\mathbf c) = \sum_i a_i (\mathbf b\times\mathbf c)_i = \sum_{ijk}\varepsilon_{ijk}a_i b_j c_k$，与行列式展开完全同构。</span>

## 6 小结

- **向量**是既有大小又有方向、且与起点无关的量；坐标分量是它在基下的数字表示。
- **点积** $\mathbf a\cdot\mathbf b = |\mathbf a||\mathbf b|\cos\theta = \sum a_i b_i$：度量同向程度，蕴含柯西-施瓦茨不等式，是余弦相似度与自注意力的数学内核。
- **叉积** $\mathbf a\times\mathbf b$：三维专属、右手定则、反对称，模长等于平行四边形面积，分量公式与 Levi-Civita 符号一一对应。
- **标量三重积** = 平行六面体体积 = 行列式；**向量三重积**满足 BAC–CAB 公式且不满足结合律。

在下一节，我们将让向量「流动」起来——把它变成空间中的函数，用三个微分算子梯度、散度、旋度去刻画场的局部行为，这就是**向量分析**。
