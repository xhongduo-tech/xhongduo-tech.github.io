---
title: 常系数线性方程组
date: 2026-08-07
---

# 常系数线性方程组

<div class="epigraph">
<p>矩阵的指数，让线性流在时间轴上前进得如此优雅。</p>
<footer>—— 化用于苏菲 · 热尔曼（Sophie Germain）时代的分析传统</footer>
</div>

<div class="article-byline">
<p>第二级 · 常微分方程 ｜ 丁同仁《常微分方程》 第二篇 第五章 §2 ｜ 2026-08-07</p>
</div>

## 为什么矩阵指数是答案

上一节引入了 $\boldsymbol{x}' = A(t)\boldsymbol{x}$，但 $A$ 随时间变化时没有统一解法。当系数**是常数矩阵 $A$** 时，我们期待解是「指数的推广」——一维时 $x' = ax$ 的解是 $x = e^{at}x_0$，多维时自然猜想

$$\boldsymbol{x} = e^{At}\,\boldsymbol{x}_0$$

只要把「$e$ 的矩阵次幂」定义清楚，这条公式就成立。**矩阵指数（matrix exponential）**

$$e^{At} = \sum_{k=0}^{\infty}\frac{A^k t^k}{k!} = I + At + \frac{A^2 t^2}{2!} + \cdots$$

是本节的一号主角。它把线性代数（特征值、Jordan 形）与微分方程焊接在一起——这也解释了为什么这类方程与《线性代数》《矩阵论》如此亲密。<span class="marginnote">矩阵指数由弗罗贝尼乌斯等人系统发展，如今是量子力学时间演化算符 $e^{-iHt/\hbar}$、控制系统状态转移、以及一切线性动力学的共同语言。</span>

## 1 矩阵指数及其性质

**矩阵指数定义**：对 $n\times n$ 常矩阵 $A$，幂级数

$$e^{At} = I + At + \frac{A^2 t^2}{2!} + \frac{A^3 t^3}{3!} + \cdots$$

对一切 $t$ 绝对收敛。关键性质：

- $\dfrac{d}{dt} e^{At} = A\,e^{At} = e^{At}A$（与 $A$ 可交换）；
- $e^{A\cdot 0} = I$；
- 若 $AB = BA$，则 $e^{A+B} = e^A e^B$（**不可交换时不成立**）。

**定理**：初值问题 $\boldsymbol{x}' = A\boldsymbol{x},\ \boldsymbol{x}(0) = \boldsymbol{x}_0$ 的唯一解是

$$\boldsymbol{x}(t) = e^{At}\,\boldsymbol{x}_0$$

这就是常系数线性方程组的通解公式——$e^{At}$ 本身就是一个基解矩阵。<span class="marginnote">「$e^{A+B}=e^Ae^B$ 需要交换性」这条细节常被忽略：$A,B$ 不交换时只能用「BCH 公式」之类的修正。这提醒我们，矩阵世界里的「指数」比数世界更挑剔。</span>

## 2 特征值方法：三种情形

要实际算出 $e^{At}$，绕不开 $A$ 的特征结构。设 $A$ 有特征值 $\lambda$ 与特征向量 $\boldsymbol{v}$：

$$A\boldsymbol{v} = \lambda\boldsymbol{v} \quad\Longrightarrow\quad e^{At}\boldsymbol{v} = e^{\lambda t}\boldsymbol{v}$$

于是 $\boldsymbol{x} = e^{\lambda t}\boldsymbol{v}$ 是方程的解。

**情形一：$A$ 有 $n$ 个线性无关特征向量**（含实对称、对角化矩阵）。解

$$\boldsymbol{x}(t) = \sum_{k=1}^{n} C_k\,e^{\lambda_k t}\,\boldsymbol{v}_k$$

**情形二：共轭复特征值** $\lambda = \alpha + i\beta$。复解 $e^{\lambda t}\boldsymbol{v}$ 取实部、虚部得两个实解：

$$\boldsymbol{x}_1 = e^{\alpha t}\big(\operatorname{Re}(\boldsymbol{v})\cos\beta t - \operatorname{Im}(\boldsymbol{v})\sin\beta t\big), \qquad \boldsymbol{x}_2 = e^{\alpha t}\big(\operatorname{Re}(\boldsymbol{v})\sin\beta t + \operatorname{Im}(\boldsymbol{v})\cos\beta t\big)$$

**情形三：重特征值缺特征向量**。需要**广义特征向量（generalized eigenvector）**：对重数 $m$、代数重数与几何重数不同时，解含 $t^k e^{\lambda t}$ 多项式因子：

$$\boldsymbol{x} = e^{\lambda t}\Big(\boldsymbol{v}_0 + t\,\boldsymbol{v}_1 + \frac{t^2}{2!}\boldsymbol{v}_2 + \cdots\Big)$$

其中 $\boldsymbol{v}_0 = \boldsymbol{v}$（真特征向量），$\boldsymbol{v}_{k}$ 满足 $(A-\lambda I)\boldsymbol{v}_{k} = \boldsymbol{v}_{k-1}$。<span class="marginnote">这正是第二篇「重根补 $x^k e^{rx}$」的矩阵版：标量方程的二重根对应这里的 Jordan 块。用 Jordan 标准形看，$e^{At} = P e^{Jt} P^{-1}$，每个 Jordan 块给出 $e^{\lambda t}(1, t, t^2/2!, \dots)$ 型列。</span>

**例子**：$A = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$，特征值 $\pm i$。取 $\lambda = i$，特征向量 $\boldsymbol{v} = (1, i)^T$，解 $e^{it}(1,i)^T$。实部虚部：

$$\boldsymbol{x}_1 = \begin{pmatrix}\cos t \\ -\sin t\end{pmatrix}, \qquad \boldsymbol{x}_2 = \begin{pmatrix}\sin t \\ \cos t\end{pmatrix}$$

对应 $\boldsymbol{x}' = A\boldsymbol{x}$ 即 $x_1' = x_2,\ x_2' = -x_1$——圆周运动。

## 3 如何计算 $e^{At}$

三条实用路线：

**路线一：对角化**。$A = P\Lambda P^{-1}$（$\Lambda$ 对角），则 $e^{At} = P e^{\Lambda t}P^{-1}$，而 $e^{\Lambda t} = \operatorname{diag}(e^{\lambda_1 t}, \dots, e^{\lambda_n t})$。

**路线二：Jordan 标准形**。$A = PJP^{-1}$，$e^{Jt}$ 的每个 Jordan 块 $J(\lambda) = \lambda I + N$（$N$ 幂零）给出 $e^{\lambda t}\left(I + tN + \frac{t^2}{2!}N^2 + \cdots\right)$，幂零级数在有限项截断。

**路线三：化零多项式 / Putzer 算法**。不求 Jordan 形，用 Cayley–Hamilton 定理把 $e^{At}$ 表成 $I, A, \dots, A^{n-1}$ 的线性组合（系数是 $t$ 的初等函数）。<span class="marginnote">路线三适合手算：$e^{At} = \sum_{k=0}^{n-1} r_k(t) A^k$，其中 $r_k$ 满足与特征值关联的递推（Putzer 算法）。它避免了 Jordan 形的求逆，是数值上更稳的选择。</span>

**辨析｜易错点：** 特征值法只给出 $n$ 个解，缺特征向量的重特征值必须补广义特征向量，否则解数不足 $n$ 个、解空间「缺维」。验证解组完整与否的快捷方式：检查朗斯基 $W(t) = e^{(\operatorname{tr}A)t} W(0)$，在 $t=0$ 算一次行列式即可。

## 4 公式解析：$\boldsymbol{x}(t) = e^{At}\boldsymbol{x}_0$

这条公式是常系数方程组的总答案，逐层拆：

**第一步，定义如何给**：$e^{At}$ 由幂级数定义，不是「逐元素取指数」。所以 $\left(e^{At}\right)_{ij}$ 与 $e^{a_{ij}t}$ 毫无关系——这是初学者最大的误会。
**第二步，为什么它真是解**：对幂级数逐项求导得 $\frac{d}{dt}e^{At} = A + A^2t + \cdots = A e^{At}$。于是 $(\boldsymbol{x})' = A e^{At}\boldsymbol{x}_0 = A\boldsymbol{x}$ ✓，且 $\boldsymbol{x}(0) = I\boldsymbol{x}_0 = \boldsymbol{x}_0$ ✓。
**第三步，特征值怎么参与**：$e^{At}\boldsymbol{v} = e^{\lambda t}\boldsymbol{v}$ 把「矩阵指数作用在特征向量上」化为「标量指数乘特征向量」。这是特征值方法的一切起点。
**第四步，与矩阵指数的计算对接**：无论是 $P e^{\Lambda t}P^{-1}$ 还是 Jordan 块公式，最终都回到 $\boldsymbol{x} = e^{At}\boldsymbol{x}_0$。$A$ 的「增长方向」由特征值实部决定——$\operatorname{Re}\lambda < 0$ 的方向被压缩，$>0$ 被放大，这正是下一节稳定性讨论的前奏。<span class="marginnote">把 $e^{At}$ 想成「时间 $t$ 的流」：它是把初始状态 $\boldsymbol{x}_0$ 顺着系统动力学送到 $\boldsymbol{x}(t)$ 的映射。控制论里 $e^{A(t-t_0)}$ 正是状态转移矩阵。</span>

## 5 实例：双弹簧-质量系统

两个质量 $m$ 用弹簧 $k$ 串联、两端固定，设位移 $x_1, x_2$，运动方程

$$m x_1'' = -2kx_1 + kx_2, \qquad m x_2'' = kx_1 - 2kx_2$$

对位移部分作特征分析：$A = \begin{pmatrix} -2k/m & k/m \\ k/m & -2k/m \end{pmatrix}$，特征值 $\lambda_1 = -k/m$（向量 $(1,1)^T$，两质量同向振动）、$\lambda_2 = -3k/m$（向量 $(1,-1)^T$，反向振动）。

**特征模式（normal mode）**：系统的运动是这两个模式的叠加——同相慢、反相快。这是「特征向量分解」的物理意义：不是抽象代数游戏，而是**找到系统自然的振动方式**。<span class="marginnote">「模式分解」思想贯穿力学、电路、量子力学（本征态）与数据科学（主成分分析）。常微分方程里的特征值方法，是把「系统自然动作」找出来的通用语法。</span>

**辨析｜易错点：** 位移是二阶的，若想化成完整一阶系统要扩到四维（加速度分量）。直接对位移矩阵做特征分析虽快，但它的特征值与完整系统的关系是 $\pm\sqrt{-\lambda_i}$（成对出现），符号别弄丢。

**小窍门**：当 $A$ 是对角阵 $A = \operatorname{diag}(a_1, \dots, a_n)$ 时，$e^{At} = \operatorname{diag}(e^{a_1 t}, \dots, e^{a_n t})$——矩阵指数退化成逐元素指数。所以对角化 $A = P\Lambda P^{-1}$ 后 $e^{At} = P e^{\Lambda t}P^{-1}$ 的一切运算都变得平凡。把「对角化 → 逐元素指数 → 变换回去」这条流水线记住，矩阵指数不再神秘。

**再强调**：$e^{At}$ 是矩阵指数，不是逐元素指数——这是矩阵情形唯一需要时刻警惕的地方。算完随手用 $e^{A\cdot 0} = I$ 验一遍。

**一句话**：矩阵指数是「对角化 × 逐元素指数 × 还原」的三部曲。

## 6 小结

- **矩阵指数** $e^{At} = \sum A^k t^k/k!$：$\boldsymbol{x}' = A\boldsymbol{x}$ 的解是 $\boldsymbol{x} = e^{At}\boldsymbol{x}_0$。
- 特征值方法：相异实特征值 → $e^{\lambda_k t}\boldsymbol{v}_k$；共轭复特征值 → 取实部虚部得振荡解；重特征值 → **补广义特征向量**（$t^k e^{\lambda t}$ 因子）。
- 计算 $e^{At}$ 三条路：对角化、Jordan 形、Putzer 算法。
- $e^{A+B}=e^Ae^B$ 需要 $AB=BA$；$e^{At}$ 不是逐元素取指数。
- 特征值实部决定解的增长/衰减方向，是下一节稳定性理论的伏笔。

在下一节，我们从线性走进非线性——**非线性方程与线性化**，看奇点附近如何用「切线」理解「曲面」。
