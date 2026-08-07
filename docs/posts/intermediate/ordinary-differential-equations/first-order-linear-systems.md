---
title: 一阶线性方程组
date: 2026-08-07
---

# 一阶线性方程组

<div class="epigraph">
<p>把许多变量的演化放到一起看，单个方程只是它的一条坐标。</p>
<footer>—— 约瑟夫-路易 · 拉格朗日（Joseph-Louis Lagrange）</footer>
</div>

<div class="article-byline">
<p>第二级 · 常微分方程 ｜ 丁同仁《常微分方程》 第二篇 第五章 §1 ｜ 2026-08-07</p>
</div>

## 为什么从方程组开始

现实系统几乎都是「多变量同步演化」：捕食者与猎物的数量、多个质点的位置、电路里多个电容的电压。描述它们的是**方程组**——一阶线性方程组正是这些系统最直接的语言。<span class="marginnote">更妙的是，任何 $n$ 阶标量方程都可以通过「把 $y, y', \dots, y^{(n-1)}$ 设为新变量」化成一个 $n$ 维一阶方程组。所以方程组不是新麻烦，而是<strong>统摄全局的框架</strong>——单方程只是 $n=1$ 的特例。</span>

本篇（第三篇）的方法论也会在这里转折：前两篇我们追求「公式解」，而从本节起，「不求解也能读懂系统」的**定性理论**逐渐登场。矩阵符号将把我们带到与《线性代数》交汇的高地。

## 1 方程组的标准形与解

**一阶线性方程组（标准形）**：

$$\begin{cases} x_1' = a_{11}(t)x_1 + \cdots + a_{1n}(t)x_n + g_1(t) \\ \vdots \\ x_n' = a_{n1}(t)x_1 + \cdots + a_{nn}(t)x_n + g_n(t) \end{cases}$$

用向量与矩阵写成紧凑形式：

$$\boldsymbol{x}' = A(t)\,\boldsymbol{x} + \boldsymbol{g}(t), \qquad \boldsymbol{x} = \begin{pmatrix} x_1 \\ \vdots \\ x_n \end{pmatrix}, \quad A(t) = \big(a_{ij}(t)\big)$$

$\boldsymbol{g} \equiv \boldsymbol{0}$ 时为**齐次**方程组 $\boldsymbol{x}' = A(t)\boldsymbol{x}$。**解**是一个向量值函数 $\boldsymbol{x}(t) = \big(x_1(t), \dots, x_n(t)\big)^T$ 在区间 $I$ 上满足方程组。<span class="marginnote">注意 $\boldsymbol{x}$ 是「$t$ 的函数向量」，不是「$t$ 和 $x$」的双变量——习惯矩阵写法后，很多结论与一维情形只差一个「向量」前缀。</span>

**存在唯一性定理（方程组版）**：若 $A(t)$ 与 $\boldsymbol{g}(t)$ 的各分量在含 $t_0$ 的区间上连续，则初值问题 $\boldsymbol{x}' = A\boldsymbol{x} + \boldsymbol{g},\ \boldsymbol{x}(t_0) = \boldsymbol{x}_0$ 在 $t_0$ 附近存在唯一解。证明与标量版如出一辙（皮卡迭代逐分量进行）。

## 2 齐次方程组的解空间

齐次方程组 $\boldsymbol{x}' = A(t)\boldsymbol{x}$ 的解集仍是一个**线性空间**，维数为 $n$。用**矩阵**组织 $n$ 个线性无关解：

**基解矩阵（fundamental matrix）** $\Phi(t)$：把 $n$ 个线性无关解 $\boldsymbol{x}_1, \dots, \boldsymbol{x}_n$ 作为列向量排成矩阵：

$$\Phi(t) = \Big[\boldsymbol{x}_1(t)\ \ \boldsymbol{x}_2(t)\ \ \cdots\ \ \boldsymbol{x}_n(t)\Big]$$

则通解为

$$\boldsymbol{x}(t) = \Phi(t)\,\boldsymbol{C}, \qquad \boldsymbol{C} = (C_1, \dots, C_n)^T$$

**朗斯基行列式（方程组版）** $W(t) = \det\Phi(t)$，满足

$$W'(t) = \big(\operatorname{tr} A(t)\big)\,W(t)$$

其中 $\operatorname{tr} A = \sum_i a_{ii}(t)$ 是矩阵的迹。于是 $W(t) = W(t_0)\exp\!\left(\int_{t_0}^t \operatorname{tr}A(s)\,ds\right)$——**要么恒零、要么恒非零**，$W(t_0)\neq 0 \iff$ 解组线性无关。<span class="marginnote">这条 Abel 公式的矩阵版，把标量情形的 $p_1$ 换成了 $\operatorname{tr}A$。主对角线上的「总增益」控制着相空间体积的膨胀与收缩——这个想法在动力系统理论里叫「相体积」演化，与统计物理的刘维尔定理同源。</span>

**结构定理**：非齐次方程组的通解 = 对应齐次通解 + 一个特解：

$$\boldsymbol{x}(t) = \Phi(t)\,\boldsymbol{C} + \boldsymbol{x}^*(t)$$

## 3 高阶方程 ↔ 一阶方程组

**状态变量法**：把 $n$ 阶方程 $y^{(n)} + a_1 y^{(n-1)} + \cdots + a_n y = g(t)$ 化为方程组。设

$$x_1 = y, \qquad x_2 = y', \qquad \dots, \qquad x_n = y^{(n-1)}$$

则 $x_1' = x_2,\ x_2' = x_3,\ \dots,\ x_{n-1}' = x_n$，且

$$x_n' = y^{(n)} = -a_n x_1 - a_{n-1}x_2 - \cdots - a_1 x_n + g(t)$$

写成矩阵：

$$\boldsymbol{x}' = \begin{pmatrix} 0 & 1 & 0 & \cdots & 0 \\ 0 & 0 & 1 & \cdots & 0 \\ \vdots & & & \ddots & \vdots \\ -a_n & -a_{n-1} & \cdots & -a_1 \end{pmatrix}\boldsymbol{x} + \begin{pmatrix} 0 \\ \vdots \\ g(t) \end{pmatrix}$$

中间的矩阵叫**友矩阵（companion matrix）**。<span class="marginnote">这个转化是控制论的起点：把「高阶系统」变成「一阶状态方程」，状态向量里装的正是位置与速度等高阶导数的快照。现代控制（第三级《控制科学与工程》）的一切，都从这张友矩阵出发。</span>

**辨析｜易错点：** 反向也成立——方程组一般**不能**化成一个高阶标量方程（除非可解耦）。所以「方程组更一般」是单方向的包含关系。另外，$n$ 个初值条件对应 $\boldsymbol{x}(t_0)$ 的 $n$ 个分量，写初值别写丢。

## 4 公式解析：$\boldsymbol{x}' = A\boldsymbol{x}$ 的通解 $\boldsymbol{x} = \Phi(t)\boldsymbol{C}$

这条公式把「解方程组」压缩成「求基解矩阵」，逐层拆：

- **第一步，为什么 $\Phi$ 的各列是解**：$\Phi$ 的第 $k$ 列 $\boldsymbol{x}_k$ 满足 $\boldsymbol{x}_k' = A\boldsymbol{x}_k$。写成矩阵形式就是 $\Phi' = A\Phi$——**基解矩阵满足同一个矩阵微分方程**。
- **第二步，为什么 $\boldsymbol{C}$ 是任意向量**：通解 $\Phi\boldsymbol{C}$ 是 $n$ 个解向量的线性组合，$C_k$ 为任意常数。解空间是 $n$ 维，$\Phi$ 的列构成一组基。
- **第三步，初值怎么定 $\boldsymbol{C}$**：$\boldsymbol{x}(t_0) = \Phi(t_0)\boldsymbol{C} = \boldsymbol{x}_0$，故 $\boldsymbol{C} = \Phi(t_0)^{-1}\boldsymbol{x}_0$。$W(t_0)\neq 0$ 保证 $\Phi(t_0)$ 可逆。<span class="marginnote">有些教材喜欢用「标准基解矩阵」$\Phi(t)\Phi(t_0)^{-1}$，它在 $t_0$ 处等于单位阵，让初值问题变成 $\boldsymbol{x} = \Phi(t)\Phi(t_0)^{-1}\boldsymbol{x}_0$——比记 $\boldsymbol{C}$ 更机械。</span>
- **第四步，与标量情形的对偶**：一维时 $\Phi = e^{\int a dt}$，$\boldsymbol{C} = C$——一切齐次线性理论的公式，在向量化之后只是把「函数」换成「矩阵、向量」。这就是线性代数统一一切线性对象的威力。

## 5 建模例：两容器混合

两个盐水池通过管道相连，是方程组建模的经典范例。设池 1、池 2 的盐量分别为 $x_1(t), x_2(t)$，体积 $V_1, V_2$，池 1 以流量 $q$ 净流入清水、池 2 以流量 $q$ 净流出，两池间以流量 $q$ 互相交换。盐量方程：

$$x_1' = -q\frac{x_1}{V_1} + q\frac{x_2}{V_2}, \qquad x_2' = q\frac{x_1}{V_1} - q\frac{x_2}{V_2}$$

写成 $\boldsymbol{x}' = A\boldsymbol{x}$，$A = \begin{pmatrix} -q/V_1 & q/V_2 \\ q/V_1 & -q/V_2 \end{pmatrix}$。若 $V_1 = V_2 = V$，$A$ 的特征值为 $0$ 与 $-2q/V$：

- 特征值 $0$ 对应「总盐量守恒」方向 $\boldsymbol{v} = (1,1)^T$；
- 特征值 $-2q/V$ 对应「浓度差衰减」方向 $\boldsymbol{v} = (1,-1)^T$。

通解 $\boldsymbol{x}(t) = C_1\binom{1}{1} + C_2 e^{-2qt/V}\binom{1}{-1}$：**总盐量不变，浓度差按指数抹平**——系统趋于两池同浓度。<span class="marginnote">这是「守恒律 + 扩散」在最低维度的体现：守恒方向对应零特征值，耗散方向对应负特征值。把「零特征值 = 守恒量」记住，读矩阵系统就有了物理抓手。</span>

**辨析｜易错点：** 特征值 $0$ 不是「不稳定」——它对应守恒方向（中性），解在该方向不衰减也不增长。看到 $0$ 特征值先想「什么量守恒」，这是线性系统分析的高级直觉，也是后面李雅普诺夫稳定性的伏笔。

### 一阶方程组实例

解 $\boldsymbol{x}' = A\boldsymbol{x}$，$A = \begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix}$，$\boldsymbol{x}(0) = \binom{1}{2}$。$A$ 是对角阵，解直接分量解耦：

$$x_1' = 2x_1 \Rightarrow x_1 = e^{2t}, \qquad x_2' = 3x_2 \Rightarrow x_2 = 2e^{3t}$$

基解矩阵 $\Phi(t) = \operatorname{diag}(e^{2t}, e^{3t})$，通解 $\boldsymbol{x} = \Phi\boldsymbol{C}$，初值代入得 $\boldsymbol{C} = (1, 2)^T$。<span class="marginnote">这个例子看似平凡，却展示了方程组的核心逻辑：<strong>对角化之后，耦合系统退化成相互独立的单变量指数</strong>——而求 $A$ 的特征分解，正是把一般系统「对角化」的手段，下一节常系数方程组将彻底展开。</span>

**再提醒**：$\Phi$ 的各列是解，$\det\Phi \neq 0$ 保证解组独立。拿到一个基解矩阵，先算一次 $W(t_0)$ 验独立，再谈通解——三秒钟的检查，避免整篇白算。

**预告**：常数矩阵 $A$ 时，基解矩阵恰好是 $\Phi(t) = e^{At}$——下一节的主角。

## 6 小结

- **一阶线性方程组** $\boldsymbol{x}' = A(t)\boldsymbol{x} + \boldsymbol{g}(t)$：$n$ 维解空间，解是向量值函数。
- 存在唯一性定理对分量连续的情形成立；齐次解空间维数为 $n$。
- **基解矩阵** $\Phi$ 的列是 $n$ 个线性无关解，通解 $\boldsymbol{x} = \Phi\boldsymbol{C}$；朗斯基 $W = \det\Phi$ 满足 $W' = (\operatorname{tr}A)W$。
- **高阶方程 ↔ 方程组**：状态变量法把 $n$ 阶方程化为 $n$ 维一阶方程组（友矩阵）。
- 非齐次通解 = 齐次通解 + 特解，结构定理与标量版完全平行。

在下一节，当系数 $A$ 是常数矩阵时，方程组迎来它的显式解——**常系数线性方程组**，矩阵指数将大显身手。
