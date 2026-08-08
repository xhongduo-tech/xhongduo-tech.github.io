---
title: 线性代数：标量、向量、矩阵与张量
date: 2026-08-07
---

# 线性代数：标量、向量、矩阵与张量

<div class="epigraph">
<p>数学是给不同事物起同一个名字的艺术。</p>
<footer>—— 亨利 · 庞加莱（Henri Poincaré）</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ 花书《深度学习》§2.1 ｜ 2026-08-07</p>
</div>

## 为什么从线性代数开始

一个神经网络的「前向传播」，看起来是一长串神秘的矩阵、向量运算：$h = \sigma(\boldsymbol{W}\boldsymbol{x} + \boldsymbol{b})$、注意力里的 $\boldsymbol{Q}\boldsymbol{K}^{\top}$、嵌入表的一次「查表」。剥开所有的名词，底层的操作几乎全是**线性代数**——标量与向量相乘、向量与向量点积、矩阵与矩阵相乘。可以说，**线性代数是深度学习的语法本身**。

这个网站叫「从极限到大模型」。**极限**给了我们研究连续变化、求梯度所需的全部语言（那是微积分的事，见第一级《基础数学》与第三级《微积分》）；而**线性代数**给的，则是**高维空间的几何语言**——一个模型动辄几百万个参数，这些参数构成的空间就是一座几百万维的抽象几何宫殿，线性代数就是这座宫殿的地图与尺规。<span class="marginnote">花书把线性代数列为深度学习的第一块数学基石，放在概率论与数值计算之前：先学会「在数字组成的多维空间里做运算」，后面的一切（梯度、优化、表示）才有落脚的舞台。</span>

这一节对标花书 §2.1，我们先把四个最基本的概念——**标量、向量、矩阵、张量**——的定义与记号立起来，再展开向量与矩阵的运算，最后回答一个贯穿全站的问题：深度学习**为什么**偏偏用线性代数作为它的第一语言。同济《线性代数》是第二级的标准教材，本专题用「几何直觉 + 记法先行」的视角重讲，两套体系互补。

## 1 标量、向量、矩阵与张量

### 标量

**标量（scalar）**：一个单独的数，通常用小写斜体字母表示。比如学习率 $\eta = 0.01$、正则化系数 $\lambda = 10^{-4}$，都是标量。标量只属于一个数集，例如 $x \in \mathbb{R}$ 表示 $x$ 是实数。<span class="marginnote">$\mathbb{R}$ 是实数集记号，$\in$ 读作「属于」——这两个符号在第一级《集合的概念》里已经正式登场：$x \in \mathbb{R}$ 就是在用集合语言声明「$x$ 从实数里取值」。</span>

### 向量

**向量（vector）**：一列有序排列的数。记作

$$
\boldsymbol{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^{n}
$$

向量里的每个分量 $x_i$ 都是标量。数学里约定向量的下标从 1 开始；而 Python 的 NumPy 数组从 0 开始，$x_i$ 对应 $x_i$。<span class="marginnote"><strong>行向量与列向量</strong>：竖着写的叫列向量（如上），横着写的是行向量 $\boldsymbol{x}^{\top} = [x_1, x_2, \dots, x_n]$。深度学习默认用列向量，转置符号 $\top$ 负责在两者之间切换——这个约定在花书与 PyTorch 里都是一致的。</span>

向量是什么？它既是一组数，也是一个「点」或一条「从原点出发的有向线段」。把 $\mathbb{R}^2$ 里的向量想成平面上的箭头，把 $\mathbb{R}^{n}$ 想成 $n$ 维空间里的箭头——**维度高于 3 的图像画不出来，但代数照算不误**。这是线性代数最重要的心法：几何直觉负责想象低维，代数负责处理高维。

### 矩阵

**矩阵（matrix）**：一个二维数组，即排成矩形的数表。记作 $\boldsymbol{A} \in \mathbb{R}^{m \times n}$，表示 $\boldsymbol{A}$ 有 $m$ 行、$n$ 列。矩阵的每个元素用双下标表示：$A_{i,j}$ 是第 $i$ 行、第 $j$ 列的数（数学从 1 开始编号，NumPy 从 0 开始）。

$$
\boldsymbol{A} = \begin{bmatrix} A_{1,1} & A_{1,2} & \cdots & A_{1,n} \\ A_{2,1} & A_{2,2} & \cdots & A_{2,n} \\ \vdots & \vdots & \ddots & \vdots \\ A_{m,1} & A_{m,2} & \cdots & A_{m,n} \end{bmatrix}
$$

一个矩阵可以看作「$m$ 个行向量堆起来」或「$n$ 个列向量排成一排」——**矩阵是向量的容器，也是把向量映射成向量的机器**（后者的视角我们留到第 6 节）。

### 张量

**张量（tensor）**：轴数超过两个的数组。花书 §2.1 特意声明了本书的约定：**这里的「张量」泛指「在规则网格上、具有可变数量轴的任意数组」**，并不指物理学里那个要满足坐标变换定律的张量。<span class="marginnote">物理学里的张量有严格的坐标变换要求，机器学习把这个词借过来，只表示「多维数组」。PyTorch 的 `Tensor`、NumPy 的 `ndarray` 都是这个含义——两者只是披着不同外衣的同一件事。</span>

例如一张彩色图片可以编码为一个三阶张量，形状是「高 × 宽 × 通道」；一批这样的图片则是一个四阶张量，形状是「批大小 × 高 × 宽 × 通道」。轴（axis）的个数叫**阶数**，也叫 **rank** 或 **ndim**；每个轴的大小按顺序排成一个元组，就是**形状（shape）**。

![张量的阶数与形状：从标量到三阶张量](/images/deep-learning/linear-algebra-basics-1.svg)

这张图把四个对象排在同一张画布上：0 阶的标量只有一个格子，1 阶的向量是一串格子，2 阶的矩阵是一片格子，3 阶的张量是「许多片格子堆叠成块」。**阶数每加一，就多一个轴，形状就多一位**——这就是深度学习里最常用的那套形状语言的直观来源。

## 2 向量的运算：点积与外积

向量之间有两种最基本的乘法，结果截然不同。

**点积（dot product）**，也叫内积：两个同维向量逐元素相乘再求和，结果是一个**标量**。

$$
\boldsymbol{x}^{\top} \boldsymbol{y} = \sum_{i=1}^{n} x_i y_i
$$

点积的几何意义是**两个向量的对齐程度**：$\boldsymbol{x}^{\top}\boldsymbol{y} = \|\boldsymbol{x}\| \, \|\boldsymbol{y}\| \cos \theta$，其中 $\theta$ 是两向量之间的夹角。夹角越小，点积越大；两向量正交（夹角 90°）时点积为 0。这条「点积 ≈ 相似度」的直觉，是注意力机制 $\boldsymbol{Q}\boldsymbol{K}^{\top}$ 的胚胎——Transformer 里「查询与键越像，注意力权重越大」，底层就是点积在衡量相似度。

**外积（outer product）**：两个向量相乘得到一个**矩阵**。对 $\boldsymbol{x} \in \mathbb{R}^{m}$、$\boldsymbol{y} \in \mathbb{R}^{n}$：

$$
\boldsymbol{x} \boldsymbol{y}^{\top} = \begin{bmatrix} x_1 y_1 & x_1 y_2 & \cdots & x_1 y_n \\ x_2 y_1 & x_2 y_2 & \cdots & x_2 y_n \\ \vdots & \vdots & \ddots & \vdots \\ x_m y_1 & x_m y_2 & \cdots & x_m y_n \end{bmatrix}
$$

外积得到的是一个**秩为 1 的矩阵**——它的每一列都是同一向量 $\boldsymbol{x}$ 的缩放版本。理解「秩为 1」不需要着急，下一节我们把「秩」正式引入，届时你会发现外积是构造和理解低秩结构最直观的工具。

## 3 矩阵的运算：转置、乘法、秩与逆

### 转置

**转置（transpose）**：把矩阵的行与列对调，记作 $\boldsymbol{A}^{\top}$。按元素定义：$(\boldsymbol{A}^{\top})_{i,j} = A_{j,i}$。转置把 $m \times n$ 矩阵变成 $n \times m$ 矩阵，把行向量变成列向量。一个常用性质：**积的转置等于反序的转置之积**，

$$
(\boldsymbol{A}\boldsymbol{B})^{\top} = \boldsymbol{B}^{\top}\boldsymbol{A}^{\top}
$$

这条「反序律」后面在推导反向传播、求梯度时会反复用到。

### 矩阵乘法

**矩阵乘法**是两个矩阵「行与列对碰」的运算。设 $\boldsymbol{A}$ 是 $m \times n$、$\boldsymbol{B}$ 是 $n \times p$，则乘积 $\boldsymbol{C} = \boldsymbol{A}\boldsymbol{B}$ 是 $m \times p$，其第 $i$ 行第 $j$ 列元素为：

$$
C_{i,j} = \sum_{k=1}^{n} A_{i,k} B_{k,j}
$$

两条值得先记住的直觉：

- **形状条件**：$\boldsymbol{A}$ 的列数必须等于 $\boldsymbol{B}$ 的行数（都等于 $n$），乘法才有定义；结果的形状是「$\boldsymbol{A}$ 的行数 × $\boldsymbol{B}$ 的列数」。
- **交换律不成立**：一般地 $\boldsymbol{A}\boldsymbol{B} \neq \boldsymbol{B}\boldsymbol{A}$。顺序有讲究——「先做哪个变换」决定了一切，这一点在神经网络里对应着层的顺序。

### 单位矩阵与逆矩阵

**单位矩阵（identity matrix）** $\boldsymbol{I}_n$：$n \times n$ 对角线上全为 1、其余全为 0 的方阵，它乘任何矩阵都不改变对方：$\boldsymbol{I}_m \boldsymbol{A} = \boldsymbol{A} \boldsymbol{I}_n = \boldsymbol{A}$。单位矩阵在矩阵世界里扮演「1」的角色。

**逆矩阵（inverse）**：若方阵 $\boldsymbol{A}$ 存在 $\boldsymbol{A}^{-1}$ 使得

$$
\boldsymbol{A}^{-1}\boldsymbol{A} = \boldsymbol{A}\boldsymbol{A}^{-1} = \boldsymbol{I}
$$

则称 $\boldsymbol{A}$ 可逆（非奇异），$\boldsymbol{A}^{-1}$ 是它的逆。逆矩阵的意义在于**解线性方程组**：$\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ 两边左乘 $\boldsymbol{A}^{-1}$ 得 $\boldsymbol{x} = \boldsymbol{A}^{-1}\boldsymbol{b}$。<span class="marginnote">花书在这里特别提醒：数值计算里几乎从不显式求逆，因为既慢又不稳定；真正求解用的是高斯消元、LU 分解等数值方法。把「逆」当作概念理解，把「消元」当作工程手段——这条忠告在第二级《线性代数》第三篇会展开。</span>

### 秩

**秩（rank）**：矩阵中**线性无关**的行向量（或列向量）的最大个数，记作 $\text{rank}(\boldsymbol{A})$。直觉上，秩衡量的是**这个矩阵能表达多少「独立的信息」**：全零矩阵秩为 0；一个 $m \times n$ 矩阵的秩不超过 $\min(m, n)$；秩等于 $\min(m,n)$ 时称**满秩**。不满秩的矩阵会「压缩维度」——它把高维空间压扁到低维里，这正是后面 SVD 与低秩近似专题的起点。

## 4 公式解析：矩阵乘法 $\boldsymbol{C}=\boldsymbol{A}\boldsymbol{B}$ 的求和定义

矩阵乘法是深度学习里出现频率最高的一行公式，值得逐符号拆开。核心等式：

$$
C_{i,j} = \sum_{k=1}^{n} A_{i,k} B_{k,j}
$$

- **第一步，读下标**。$C_{i,j}$ 是结果矩阵第 $i$ 行第 $j$ 列的元素；$A_{i,k}$ 是 $\boldsymbol{A}$ 的第 $i$ 行、第 $k$ 列；$B_{k,j}$ 是 $\boldsymbol{B}$ 的第 $k$ 行、第 $j$ 列。注意三者下标的位置：$i$ 只出现在 $\boldsymbol{A}$ 与结果，$j$ 只出现在 $\boldsymbol{B}$ 与结果，$k$ 则同时出现在 $\boldsymbol{A}$ 与 $\boldsymbol{B}$——**$k$ 是被消掉的中间维**。
- **第二步，看求和**。$\sum_{k=1}^{n}$ 的意思是：把 $k$ 从 1 数到 $n$，把每一对 $A_{i,k}B_{k,j}$ 相乘后**累加起来**。几何直觉是：**结果的第 $i,j$ 格 = $\boldsymbol{A}$ 的第 $i$ 行与 $\boldsymbol{B}$ 的第 $j$ 列做一次点积**。这就是「行 × 列对碰」的来源，也解释了为什么要求 $\boldsymbol{A}$ 的列数等于 $\boldsymbol{B}$ 的行数——被求和的 $k$ 必须有一致的范围。
- **第三步，看形状**。对每个 $(i,j)$ 都要做一次上述求和，其中 $i$ 有 $m$ 种取法、$j$ 有 $p$ 种取法，所以结果恰是 $m \times p$ 矩阵。**两个形状在中间对齐（$n$ 相同），两端相乘得到最终形状（$m \times p$）**——这个「中间对齐」的口诀，往后看任何张量乘法都能一眼看出结果形状。

用一个具体数字例子收尾。设

$$
\boldsymbol{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \qquad
\boldsymbol{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
$$

则 $C_{1,1} = 1\times5 + 2\times7 = 19$（$\boldsymbol{A}$ 第 1 行 $\cdot$ $\boldsymbol{B}$ 第 1 列），$C_{1,2} = 1\times6 + 2\times8 = 22$，$C_{2,1} = 3\times5 + 4\times7 = 43$，$C_{2,2} = 3\times6 + 4\times8 = 50$，于是

$$
\boldsymbol{C} = \boldsymbol{A}\boldsymbol{B} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
$$

## 5 张量的轴与形状：NumPy 视角

数学记号讲清楚了，还要落进代码。NumPy 里一切数组都有两个属性：`ndim`（轴数）与 `shape`（形状），正对应我们上文的「阶数与形状」。

```python
import numpy as np

x = np.zeros((2, 3, 4))   # 一个三阶张量
print(x.ndim)             # 3（轴数 = 阶数）
print(x.shape)            # (2, 3, 4)（每个轴的大小）
```

数组里的每个轴都有语义：一个形状为 `(N, C, H, W)` 的四阶张量，`N` 是批（batch）、`C` 是通道、`H` 是高、`W` 是宽。**深度学习里「每个轴是谁」的约定，构成了模型输入、输出形状的全部语言**——读文档、写模型、调试维度不匹配，本质都是在和这套轴与形状的约定打交道。<span class="marginnote">后续所有深度学习博文都会反复用到「形状对齐」的语言：线性层把 (…, n) 变到 (…, m)，卷积把 (N, C, H, W) 变到 (N, C', H', W')。今天把轴与形状的直觉钉牢，往后看任何网络结构都不会迷失方向。</span>

## 6 第一性看深度学习：为什么是线性代数

把记号都立起来之后，回到最根本的问题：**深度学习为什么用线性代数？**

答案藏在「矩阵乘法 = 线性映射」这个等式中。一个 $m \times n$ 矩阵 $\boldsymbol{A}$ 定义了一个从 $\mathbb{R}^{n}$ 到 $\mathbb{R}^{m}$ 的**线性映射**：对任意向量 $\boldsymbol{x}$ 作用，得到 $\boldsymbol{A}\boldsymbol{x}$。线性映射的意义是它**保持加法和数乘**：$\boldsymbol{A}(\boldsymbol{u}+\boldsymbol{v}) = \boldsymbol{A}\boldsymbol{u} + \boldsymbol{A}\boldsymbol{v}$，$\boldsymbol{A}(c\boldsymbol{x}) = c(\boldsymbol{A}\boldsymbol{x})$——意思是「先变换再相加」等于「先相加再变换」，线性对象的结构被保留下来。

由此得到两条对深度学习至关重要的结论：

- **复合线性映射 = 矩阵乘法**。依次施加两次线性映射（先用 $\boldsymbol{B}$，再用 $\boldsymbol{A}$）等价于一次作用 $\boldsymbol{A}\boldsymbol{B}$，因为 $(\boldsymbol{A}\boldsymbol{B})\boldsymbol{x} = \boldsymbol{A}(\boldsymbol{B}\boldsymbol{x})$。深层网络的每一层 $\boldsymbol{W}^{[l]}\boldsymbol{x} + \boldsymbol{b}^{[l]}$ 就是一次（仿射的）线性映射，**几十层叠起来，若没有非线性，整条链仍然只是一个线性映射**——这正是「纯线性堆叠没有意义」的数学理由，也是激活函数 $\sigma(\cdot)$ 存在的第一性动机。
- **几乎所有前向计算都是张量收缩**。词嵌入是「查矩阵的一行」，注意力是矩阵乘，MLP 是矩阵乘，卷积也能写成矩阵乘。**学习一个模型，就是在学习这一串张量运算里的那些矩阵元素（参数）**；而「怎么更新这些参数」，又回到了梯度（微积分）在参数空间（线性代数）里的运动。

于是整条主线连起来了：**极限/微积分提供「怎么变」的律动，线性代数提供「在哪里变」的空间**。把空间和律动都握住，我们才有资格讨论下一节——矩阵的分解（特征分解与奇异值分解），那是把一个矩阵拆成「能看懂」的部件的放大镜。

## 7 小结

- **标量**是单个数（$x \in \mathbb{R}$），**向量**是一列有序数（$\boldsymbol{x} \in \mathbb{R}^{n}$），**矩阵**是二维数表（$\boldsymbol{A} \in \mathbb{R}^{m \times n}$），**张量**是轴数超过 2 的多维数组；深度学习里的「张量」仅指多维数组。
- **点积**把两个同维向量变成标量，衡量相似度（$\boldsymbol{x}^{\top}\boldsymbol{y} = \|\boldsymbol{x}\|\|\boldsymbol{y}\|\cos\theta$）；**外积**把两个向量变成秩为 1 的矩阵。
- **矩阵乘法** $C_{i,j}=\sum_k A_{i,k}B_{k,j}$：结果第 $i$ 行第 $j$ 列 = $\boldsymbol{A}$ 第 $i$ 行与 $\boldsymbol{B}$ 第 $j$ 列的点积，形状「中间对齐、两端相乘」。
- **逆矩阵**解方程 $\boldsymbol{x}=\boldsymbol{A}^{-1}\boldsymbol{b}$，但数值计算用消元而非显式求逆；**秩**衡量矩阵的独立信息量。
- **阶数 = 轴的个数，形状 = 每个轴大小的元组**；NumPy 的 `ndim` 与 `shape` 就是它们的代码形态。
- **矩阵乘法是线性映射**：复合线性映射仍是线性映射，因此深层网络必须引入非线性（激活函数）才能超越线性表达。

在下一节，我们将进入矩阵分解：**特征分解与奇异值分解**——把一个矩阵拆成特征向量与特征值的乘积，那是理解维度压缩、数据分布与表示学习的第一把钥匙。
