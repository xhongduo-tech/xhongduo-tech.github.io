---
title: 内积空间的定义与柯西-施瓦茨不等式
date: 2026-08-07
---

# 内积空间的定义与柯西-施瓦茨不等式

<div class="epigraph">
<p>分析学中两个向量之间的夹角，与欧几里得几何里的一样，由内积唯一决定。</p>
<footer>—— 约翰 · 冯 · 诺伊曼（John von Neumann），意译</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛函分析 ｜ 程其襄《泛函分析》§4.1 ｜ 2026-08-07</p>
</div>

## 为什么需要内积

前三章的空间只有「长度」没有「角度」：范数度量大小，却无法回答「两个函数是否垂直」。可「垂直」「投影」「正交分解」是几何学的灵魂——函数空间里也该有。答案是给空间装上**内积（inner product）**：一种「向量配对」的运算 $\langle x, y\rangle$，它同时携带**角度信息**（垂直、投影）与**长度信息**（$\langle x, x\rangle$ 给出范数）。**Hilbert 空间**（完备的内积空间）是泛函分析中结构最丰富、应用最广的空间：量子力学的态空间、$L^2$ 信号空间、最佳逼近与最小二乘，全都住在 Hilbert 空间里。<span class="marginnote">内积的历史源头是<strong>点积</strong> $\vec{a}\cdot\vec{b} = |a||b|\cos\theta$。冯 · 诺伊曼在 1929 年把「点积」抽象为公理，定义了抽象的 Hilbert 空间。今天你用 Python 算「两个向量的余弦相似度」，用的就是内积——只是坐标版本。</span>

## 1 内积的公理

设 $H$ 是复数域 $\mathbb{C}$ 上的线性空间。映射 $\langle \cdot, \cdot\rangle : H \times H \to \mathbb{C}$ 若满足：

1. **共轭对称性**：$\langle x, y\rangle = \overline{\langle y, x\rangle}$；
2. **第一变量线性**：$\langle \alpha x + \beta y, z\rangle = \alpha\langle x, z\rangle + \beta\langle y, z\rangle$；
3. **正定性**：$\langle x, x\rangle \ge 0$，且 $\langle x, x\rangle = 0 \iff x = 0$；

则称 $\langle \cdot, \cdot\rangle$ 为 $H$ 上的**内积（inner product）**，$(H, \langle\cdot,\cdot\rangle)$ 称为**内积空间（inner product space）**。

**辨析｜易错点：** 当数域是 $\mathbb{C}$ 时，内积对**第二变量**是**共轭线性**的（不是线性）：

$$
\langle x, \alpha y\rangle = \bar{\alpha}\langle x, y\rangle
$$

这是初学最易踩的坑。它由共轭对称性 + 第一变量线性推出。实空间（$\mathbb{R}$）里没有这个问题（$\bar\alpha = \alpha$），本书以复空间为准，实空间看作特例。<span class="marginnote">物理学家习惯另一种约定：内积对<strong>第二变量</strong>线性（线性是 $\langle x, \alpha y\rangle = \alpha\langle x,y\rangle$）。数学家的「共轭线性第二变量」与物理学家相反。约定不同不影响数学，但<strong>阅读任何文献前先确认它的约定</strong>——这是学术交流的潜规则。</span>

## 2 例子：从点积到积分

**例一（$\mathbb{C}^n$）**：$\langle x, y\rangle = \sum_{i=1}^n x_i \overline{y_i}$，标准内积。

**例二（$l^2$）**：$\langle x, y\rangle = \sum_{n=1}^\infty x_n \overline{y_n}$。需要证明级数收敛：由霍尔德不等式（$p = q = 2$），$\sum |x_n \overline{y_n}| \le \|x\|_2\|y\|_2 < \infty$。<span class="marginnote">$l^2$ 的内积存在性依赖<strong>Hölder 不等式在 $p=q=2$ 的特例</strong>，也就是后面的 Cauchy-Schwarz 不等式。所以 Cauchy-Schwarz 不只是一个「题」，它保证 $l^2$、$L^2$ 的内积<strong>良定义</strong>。</span>

**例三（$L^2[a,b]$）**：$\langle f, g\rangle = \int_a^b f(t)\overline{g(t)}\, dt$，同样由 Cauchy-Schwarz（积分版）保证收敛。

**例四（矩阵内积）**：$n\times n$ 复矩阵配 $\langle A, B\rangle = \operatorname{tr}(B^*A)$，构成内积空间。

## 3 Cauchy-Schwarz 不等式

**定理（Cauchy-Schwarz 不等式）：设 $H$ 是内积空间，则对一切 $x, y \in H$，**

$$
\big| \langle x, y\rangle \big|^2 \le \langle x, x\rangle \langle y, y\rangle
$$

且等号成立当且仅当 $x, y$ **线性相关**（一个向量是另一个的倍数）。

这是整个内积理论的地基：它把「两个向量的内积」用「各自的长度」控制住，等价于说「两个单位向量的内积绝对值不超过 1」——即「夹角余弦不超过 1」。<span class="marginnote">名字里有三个人（柯西、施瓦茨、布尼亚科夫斯基），因为它在不同领域被独立发现：数列求和版（柯西）、积分版（布尼亚科夫斯基）、一般抽象版（施瓦茨）。它们是同一个不等式的三种形态，都对应「$|\cos\theta| \le 1$」。</span>

**由 Cauchy-Schwarz 立即得到三角不等式**：

$$
\|x + y\| \le \|x\| + \|y\|, \qquad \text{其中 } \|x\| = \sqrt{\langle x, x\rangle}
$$

（平方展开后用 Cauchy-Schwarz 控制交叉项 $2\operatorname{Re}\langle x,y\rangle$。）这一步说明：**内积自动诱导一个范数**。

## 4 公式解析：Cauchy-Schwarz 不等式的证明

一个经典的「判别式」证明，它只用了正定性，非常优雅：

- **第一步，考察非负二次函数**：对任意数 $\lambda$，由正定性，

$$
0 \le \|x + \lambda y\|^2 = \langle x + \lambda y, x + \lambda y\rangle = \langle x,x\rangle + \lambda\langle y,x\rangle + \bar\lambda\langle x,y\rangle + |\lambda|^2\langle y,y\rangle
$$

- **第二步，选特殊的 $\lambda$**：令 $\lambda = -\frac{\langle x, y\rangle}{\langle y, y\rangle}$（若 $y \neq 0$）。代入得

$$
0 \le \langle x,x\rangle - \frac{|\langle x,y\rangle|^2}{\langle y,y\rangle}
$$

- **第三步，移项**：

$$
|\langle x, y\rangle|^2 \le \langle x,x\rangle\langle y,y\rangle
$$

- **等号情形**：等号成立当且仅当 $x + \lambda y = 0$，即线性相关。$y = 0$ 时两边均为 0，也线性相关。

**关键**：证明只用了「$\|x + \lambda y\|^2 \ge 0$」这个正定性公理，配合「挑一个让交叉项恰到好处的 $\lambda$」。**正定性是整个证明的唯一来源**——它揭示了 Cauchy-Schwarz 的本质是「长度平方非负」。

## 5 Cauchy-Schwarz 的三个著名形态

| 形态 | 不等式 | 几何含义 |
| --- | --- | --- |
| 有限和 | $(\sum x_i y_i)^2 \le (\sum x_i^2)(\sum y_i^2)$ | 点积版 |
| 无穷级数 | $\sum |x_n y_n| \le \sqrt{\sum x_n^2}\sqrt{\sum y_n^2}$ | 保证 $l^2$ 内积收敛 |
| 积分 | $\big|\int fg\big|^2 \le \int|f|^2\int|g|^2$ | 保证 $L^2$ 内积收敛 |

三种形态的唯一区别是把「求和」换成「积分」，证明同构。**这个不等式在信号处理里就是「互相关系数不超过 1」，在机器学习里就是「余弦相似度 $\in [-1,1]$」，在概率论里就是「$|\mathrm{Cov}(X,Y)| \le \sqrt{\mathrm{Var}(X)\mathrm{Var}(Y)}$」**——同一个不等式的不同面孔。<span class="marginnote">概率论中的相关系数 $\rho = \frac{\mathrm{Cov}(X,Y)}{\sqrt{\mathrm{Var}X\,\mathrm{Var}Y}}$ 的绝对值不超过 1，正是 Cauchy-Schwarz：随机变量构成内积空间 $\langle X,Y\rangle = E[XY]$，方差就是「长度平方」。这个链接能帮你把两门课串起来。</span>

## 6 例题精讲：Cauchy-Schwarz 的三个使用

**例题一：验证 $l^2$ 内积良定义**。

- $\langle x,y\rangle = \sum x_n\bar y_n$ 收敛需要 $\sum|x_n\bar y_n| < \infty$。
- Cauchy-Schwarz：$\sum|x_n y_n| \le \|x\|_2\|y\|_2 < \infty$。
- 没有 Cauchy-Schwarz，$l^2$ 的内积都不存在。

**例题二：三角不等式的推导**。

- $\|x+y\|^2 = \|x\|^2 + \|y\|^2 + 2\operatorname{Re}\langle x,y\rangle$。
- $|2\operatorname{Re}\langle x,y\rangle| \le 2\|x\|\|y\|$（Cauchy-Schwarz）。
- 故 $\|x+y\|^2 \le (\|x\| + \|y\|)^2$。

**例题三：相关系数 $|\rho| \le 1$**。

- 随机变量内积 $\langle X,Y\rangle = E[XY]$，$\rho = \frac{\operatorname{Cov}(X,Y)}{\sqrt{\operatorname{Var}X\operatorname{Var}Y}}$。
- $|\operatorname{Cov}(X,Y)| \le \sqrt{\operatorname{Var}X\operatorname{Var}Y}$ 正是 Cauchy-Schwarz。
- 概率论与泛函分析的第一个交汇点。

**核心要点**：Cauchy-Schwarz 的三个使用——良定义、三角不等式、相关系数——都是「$|\langle x,y\rangle| \le \|x\|\|y\|$」的化身。

**辨析｜易错点：** 等号成立 ⟺ $x, y$ 线性相关（一个向量是另一个的倍数）。相关系数 $|\rho| = 1$ 意味着线性相关。


## 7 小结

- **内积公理**：共轭对称、第一变量线性、正定；复内积对第二变量共轭线性。
- **内积空间例子**：$\mathbb{C}^n$、$l^2$、$L^2$、矩阵内积空间。
- **Cauchy-Schwarz**：$|\langle x,y\rangle|^2 \le \langle x,x\rangle\langle y,y\rangle$，等号 ⟺ 线性相关；证明全靠正定性。
- **内积诱导范数**：$\|x\| = \sqrt{\langle x,x\rangle}$，三角不等式由 Cauchy-Schwarz 推出。
- **三张面孔**：有限和、级数、积分——同一不等式的不同形态，串起分析、概率、ML。

在下一节，我们研究内积诱导的范数何时「名副其实」——**由内积诱导的范数与 Hilbert 空间**，并揭示「什么范数来自内积」的平行四边形判别法。
