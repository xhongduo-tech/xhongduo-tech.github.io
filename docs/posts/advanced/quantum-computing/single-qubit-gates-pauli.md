---
title: 单比特门：X、Y、Z 与 Pauli 门
date: 2026-08-07
---

# 单比特门：X、Y、Z 与 Pauli 门

<div class="epigraph">
<p>我不能创造的东西，我就不理解。</p>
<footer>—— 理查德 · 费曼（Richard Feynman）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen & Chuang《量子计算》§1.3.1 单比特门 ｜ 2026-08-07</p>
</div>

## 为什么从 Pauli 门开始

上一节我们搭好了线路模型的语法：线、门、测量。现在开始填空——**门**到底是什么。量子门是作用在量子态上的幺正算符，而单比特门是最小的一格：$2 \times 2$ 的幺正矩阵。在无穷多个 $2\times 2$ 幺正矩阵里，有三个格外重要，它们就是 **X、Y、Z 门**，合称 **Pauli 门**。

它们重要有两个理由。第一，它们是**布洛赫球上的基础旋转**：任何单比特幺正门都能写成绕某个轴转某个角，而 Pauli 门正好是绕 $x, y, z$ 三个坐标轴的半圈（$\pi$ 弧度）旋转——理解它们就理解了单比特量子演化的全部。第二，它们还是量子纠错的「错误字母表」：**比特翻转、相位翻转**这些噪声，恰好就是 X、Z 门在作用。费曼那句「不能创造就不理解」用在这里正合适——把 Pauli 门亲手算透、摆进线路，你对量子门才算真正「创造」过。

## 1 单比特门是什么

**核心概念：** 一个**单比特门**是一个 $2\times2$ 幺正矩阵 $U$，作用在单比特态 $|\psi\rangle$ 上得到新态 $U|\psi\rangle$。幺正性要求

$$
U^\dagger U = U U^\dagger = I
$$

$U^\dagger$ 是 $U$ 的共轭转置。这个条件保证了门**保内积**（概率总和守恒、正交态仍正交），也保证了门**可逆**（$U^{-1} = U^\dagger$）。单比特门在布洛赫球上有一个优美的图像：$U$ 是球面上的一个**旋转**——保长度的线性变换在球面几何里就是旋转。<span class="marginnote">任意单比特幺正门可以分解为「绕某个轴转某个角」：$R_{\hat n}(\theta)$。这一点在《布洛赫球》与后续《旋转门 Rx、Ry、Rz》两篇会展开；本篇先看三个特例——绕 $x, y, z$ 轴转 $\pi$ 弧度，恰好就是 Pauli 门。</span>

一个重要的细节是**全局相位不可观测**：$U$ 与 $e^{i\varphi}U$ 作用在任何态上得到的统计分布完全相同（每个振幅都乘同一个相位因子，测量概率 $|\cdot|^2$ 不变）。因此我们说「门等价」时，通常忽略全局相位——这个约定在第三节就会用到。

## 2 三兄弟登场：X、Y、Z

Pauli 门的矩阵定义如下：

$$
X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \qquad
Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \qquad
Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
$$

把三个门对计算基态的作用写成表格：

| 门 | 作用在 $\lvert 0\rangle$ | 作用在 $\lvert 1\rangle$ | 布洛赫球解释 |
| :---: | :--- | :--- | :--- |
| $X$ | $\lvert 1\rangle$ | $\lvert 0\rangle$ | 绕 $x$ 轴转 $\pi$：**比特翻转** |
| $Y$ | $i\lvert 1\rangle$ | $-i\lvert 0\rangle$ | 绕 $y$ 轴转 $\pi$：翻转 + 相位 |
| $Z$ | $\lvert 0\rangle$ | $-\lvert 1\rangle$ | 绕 $z$ 轴转 $\pi$：**相位翻转** |

**重点：X 门在计算基下就是「量子 NOT」。** 它把 $|0\rangle \leftrightarrow |1\rangle$ 互换，逻辑上等同于经典 NOT 门。Z 门则不动 $|0\rangle$、只给 $|1\rangle$ 加一个负号——它翻转的是**相对相位**，在计算基的测量里看不出任何区别（$Z|0\rangle = |0\rangle$、$Z|1\rangle = -|1\rangle$ 测到的仍分别是 0 和 1），但对叠加态 $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$，它把 $\beta$ 的符号翻过来：$Z|\psi\rangle = \alpha|0\rangle - \beta|1\rangle$。

**辨析｜易错点：** X 门「等于 NOT」只在**计算基**下成立。对叠加态它就不是「取反」了——$X|+\rangle = |+\rangle$（$|+\rangle = \frac{|0\rangle+|1\rangle}{\sqrt2}$ 是 X 的本征态，特征值 $+1$），X 根本没改变 $|+\rangle$。把「X = NOT」这个速记用在叠加态上，是初学阶段最常见的方向性错误。

## 3 公式解析：为什么 Pauli 门是「半圈旋转」

这一节把「绕轴转 $\pi$」这句话用公式钉死。一切旋转门的母公式是：绕单位矢量 $\hat n$ 旋转角 $\theta$ 的幺正算符为

$$
R_{\hat n}(\theta) = \cos\frac{\theta}{2}\, I - i \sin\frac{\theta}{2}\,(\hat n \cdot \vec\sigma)
$$

其中 $\vec\sigma = (X, Y, Z)$ 是 Pauli 矩阵三元组，$\hat n \cdot \vec\sigma = n_x X + n_y Y + n_z Z$。这个公式怎么来的？先看一个关键性质：**$( \hat n \cdot \vec\sigma)^2 = I$**——因为 Pauli 矩阵两两反交换（$\{X, Y\} = XY + YX = 0$ 等），交叉项相消，只剩 $n_x^2 X^2 + n_y^2 Y^2 + n_z^2 Z^2 = (n_x^2 + n_y^2 + n_z^2) I = I$。

于是指数展开可以「奇偶劈开」。用 $A = -i\frac{\theta}{2}(\hat n \cdot \vec\sigma)$ 展开 $e^{A} = I + A + \frac{A^2}{2!} + \cdots$，偶数幂都回到 $(\hat n\cdot\vec\sigma)^{2k} = I$、奇数幂都带一个 $(\hat n\cdot\vec\sigma)$，合并正余弦级数就得到上面的母公式。

现在取 $\hat n = \hat x$、$\theta = \pi$，代入 $\cos\frac\pi2 = 0$、$\sin\frac\pi2 = 1$：

$$
R_{\hat x}(\pi) = -i\,X
$$

$R_{\hat x}(\pi)$ 与 $X$ 只差全局相位 $-i$，物理上完全等价——**X 门就是绕 $x$ 轴转 $\pi$ 弧度的旋转**。同样地，$R_{\hat y}(\pi) \equiv Y$、$R_{\hat z}(\pi) \equiv Z$（都是忽略全局相位后的等价）。这就是「半圈旋转」的精确含义：转半圈，把布洛赫球上的点送到关于该轴的对称点。<span class="marginnote">把 $\theta = \pi/2$ 代入就得到另一族熟悉的门：$R_{\hat x}(\pi/2)$ 正比于 $\frac{1}{\sqrt2}(I - iX)$，而 Hadamard 门 $H = \frac{X+Z}{\sqrt2}$ 正是绕「$x$ 与 $z$ 之间的对角轴」转 $\pi$——下一篇主角。所有单比特门都从这同一个母公式派生。</span>

再验证一个本征关系，把图像落到代数。因为 $X|+\rangle = |+\rangle$，所以 $|+\rangle$ 是 X 的特征值 $+1$ 的本征态，$|-\rangle = \frac{|0\rangle-|1\rangle}{\sqrt2}$ 是特征值 $-1$ 的本征态：

$$
X|+\rangle = |+\rangle, \qquad X|-\rangle = -|-\rangle
$$

对应布洛赫球：绕 $x$ 轴旋转，落在 $x$ 轴上那两个点（$|+\rangle$ 与 $|-\rangle$）当然不动——只是其中一个转完「绕一圈等于没转」的结论反映成特征值 $\pm 1$。**本征态就是旋转轴上不动的点**，这句话是贯穿后续一切门分析的直觉。

## 4 Pauli 门的代数：乘法表与对易

Pauli 门不只是三个孤立的门，它们构成一个小而美的代数结构。直接验证可得

$$
X^2 = Y^2 = Z^2 = I, \qquad XY = iZ, \quad YZ = iX, \quad ZX = iY
$$

三个门两两**反交换**：$XY = -YX$、$YZ = -ZY$、$ZX = -XZ$。这四个矩阵 $\{I, X, Y, Z\}$（连同相位因子 $\pm 1, \pm i$）组成所谓 **Pauli 群**；对量子纠错来说更重要的是下面这个事实：

**任何 $2\times2$ 矩阵都能写成 $I, X, Y, Z$ 的复线性组合。** 例如任意单比特密度算符可以展开为

$$
\rho = \frac{I + \vec r \cdot \vec\sigma}{2}, \qquad \vec r \in \mathbb{R}^3
$$

$\vec r$ 正是布洛赫球上的坐标矢量（见《布洛赫球》与《密度算符》两篇）。**Pauli 矩阵是单比特算符空间的完备基**——就像三维空间里的单位向量 $e_x, e_y, e_z$。这解释了一个看似巧合的事：为什么量子错误也被写成 $X, Y, Z$？因为任何可能的单比特误差，都能唯一地分解成这四个基的线性组合。<span class="marginnote">Pauli 矩阵最初不是为计算发明的：1925 年泡利（Wolfgang Pauli）为了描述电子自旋引入它们，与 Stern-Gerlach 实验的磁场劈裂直接相关。近百年后，它们成了量子门、量子纠错、量子机器学习的共同语言——「为自旋发明的矩阵」变成「通用量子计算的字母表」，这是物理学里漂亮的意外之一。</span>

**辨析｜易错点：** $Y = iXZ$，而且 $Y$ 的矩阵约定在不同书里有 $i$ 的正负号差异（取决于 $\vec\sigma$ 的定义）。一旦选定约定，$XY = iZ$ 等乘法表必须自洽；运算时别把 $i$ 丢了——它正是对易关系里「转动」的几何来源。

## 5 为什么 Pauli 门是错误字母表

把 Pauli 门当作「错误」来读，是进入量子纠错（第八篇）的钥匙。

**$X$ 错误 = 比特翻转（bit flip）**：$|0\rangle$ 变成 $|1\rangle$。就像经典通信里的 0/1 反了。
**$Z$ 错误 = 相位翻转（phase flip）**：$|1\rangle$ 的分量加负号。经典世界里没有对应的错误——经典比特只有 0/1，没有「相位」。这是量子信息特有的误差类型。
**$Y$ 错误 = 两者同时发生**（带上相位）。
**$I$ 对应「没出错」。**

**重点：** 量子纠错的关键预处理步骤——**误差离散化**——正是建立在 Pauli 基上的：在测量意义上，任意单比特噪声都等价于以一定概率施加 $I, X, Y, Z$ 之一。于是「连续的、无限多种的噪声」被归结为「四种离散的 Pauli 错误」，纠正它们就只需处理这四种情况。没有 Pauli 门，量子纠错连问题都定义不出来。<span class="marginnote">这个「把连续噪声离散成 Pauli 错误」的过程在 N&C §10.2「离散化量子错误」里有详细证明，核心是：校验子测量会强迫噪声「塌缩」成 Pauli 错误之一。这也是为什么「三比特比特翻转码」「相位翻转码」都直接用 Pauli 语言描述。</span>

## 6 一个可运行的示例（Qiskit）

用 Qiskit 亲手「创造」三个门，验证它们的矩阵作用：

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

zero = Statevector.from_label("0")
for name, gate in [("X", "x"), ("Y", "y"), ("Z", "z")]:
    qc = QuantumCircuit(1)
    getattr(qc, gate)(0)
    print(name, Operator(qc).data.round(4))      # X/Y/Z 的矩阵
    print(name + "|0⟩ ->", zero.evolve(qc).data)  # X|0⟩=|1⟩, Y|0⟩=i|1⟩, Z|0⟩=|0⟩
```

对照第 2 节的表格：$X|0\rangle = |1\rangle$、$Y|0\rangle = i|1\rangle$、$Z|0\rangle = |0\rangle$——程序输出与矩阵计算完全一致。这就是「不能创造就不理解」：亲手跑一遍，三个门就从符号变成了你拥有的对象。

## 7 小结

- **单比特门**是 $2\times2$ 幺正矩阵，$U^\dagger U = I$，可逆、保内积；布洛赫球上是旋转。
- **Pauli 门**：$X = \begin{pmatrix}0&1\\1&0\end{pmatrix}$、$Y = \begin{pmatrix}0&-i\\i&0\end{pmatrix}$、$Z = \begin{pmatrix}1&0\\0&-1\end{pmatrix}$；X 是比特翻转，Z 是相位翻转，Y 两者兼具。
- **半圈旋转**：母公式 $R_{\hat n}(\theta) = \cos\frac\theta2 I - i\sin\frac\theta2(\hat n\cdot\vec\sigma)$；$\theta = \pi$ 时 $R_{\hat x}(\pi) = -iX \equiv X$。
- **本征关系**：$X|\pm\rangle = \pm|\pm\rangle$，本征态就是旋转轴上不动的点；X 的「NOT」只在计算基下成立。
- **代数结构**：$X^2 = Y^2 = Z^2 = I$，两两反交换，$XY = iZ$ 等；$I, X, Y, Z$ 是单比特算符空间的完备基（$\rho = \frac{I + \vec r\cdot\vec\sigma}{2}$）。
- **错误字母表**：任何单比特噪声等价于 $I, X, Y, Z$ 之一——误差离散化是量子纠错的前提。

在下一节，我们继续填充门库：**Hadamard 门与相位门（S、T 门）**——$H$ 从 $|0\rangle$ 造出叠加、$S$ 与 $T$ 是比 $Z$ 更精细的相位旋转，它们共同构成构建任意单比特门所需的第三块拼图。
