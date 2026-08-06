---
title: 线性算符与矩阵表示
date: 2026-08-07
---

# 线性算符与矩阵表示

<div class="epigraph">
<p>在数学里，你并非理解了事物，而只是习惯了它们。</p>
<footer>—— 约翰 · 冯 · 诺伊曼（John von Neumann）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen & Chuang《量子计算》第2章 ｜ 2026-08-07</p>
</div>

## 为什么从线性算符开始

量子态是向量，那么「让量子态改变」的东西是什么？答案是**算符（operator）**：把向量变成向量的线性映射。量子计算里，一切「门」都是**幺正算符**（量子门），一切「测量」都由**厄米算符**描述——所以我们先要把「算符」这个对象本身吃透。

这一节回答三个问题：算符是什么、怎样把算符写成矩阵、怎样从矩阵读懂算符在做什么。学完之后，「量子门就是矩阵」这句话就不再是一句口号，而是一个随时可以落笔计算的事实。<span class="marginnote">线性是量子力学的一道铁律：态的演化是线性的。上一节《复数域上的向量空间》讲了线性结构，这一节我们把「线性映射」这个最核心的对象请上台。</span>

## 1 线性算符的定义

**线性算符（linear operator）**：从向量空间 $V$ 到自身（或另一空间）的映射 $A$，满足

$$
A\big(\alpha|v\rangle + \beta|w\rangle\big) = \alpha\,A|v\rangle + \beta\,A|w\rangle, \qquad \forall\, \alpha,\beta\in\mathbb{C},\ |v\rangle,|w\rangle\in V
$$

两个最基本的线性算符：

- **恒等算符** $I$：$I|v\rangle = |v\rangle$，对应单位矩阵。
- **Pauli-X 算符** $X$（比特翻转门）：$X|0\rangle = |1\rangle$，$X|1\rangle = |0\rangle$。<span class="marginnote">Pauli-X 的名字来自物理学家 Wolfgang Pauli 的泡利矩阵体系。它把 $|0\rangle$ 和 $|1\rangle$ 对调，就像经典逻辑里的 NOT 门——但它作用在叠加态上时还会引入相干性，这是经典 NOT 没有的。它和 Z、Y 合称 Pauli 门，是量子计算最基本的积木，见第三篇《单比特门：X、Y、Z 与 Pauli 门》。</span>

线性是「门」天然的性质：量子演化由薛定谔方程支配，方程的右端是线性的，所以任何合法的量子门都必须是线性映射——这正是上一节不可克隆定理证明里用到的那条铁律。

## 2 矩阵表示：选一组基

算符是抽象的，矩阵是具体的。把算符翻译成矩阵需要**选一组基** $\{|1\rangle,\dots,|n\rangle\}$（标准正交基）。定义矩阵元：

$$
A_{ij} = \langle i|\,A\,|j\rangle
$$

$A_{ij}$ 读作「$A$ 在基 $\{|i\rangle\}$ 下的 $i$ 行 $j$ 列元素」。整个矩阵写作：

$$
A \;=\; \begin{pmatrix}
A_{11} & A_{12} & \cdots & A_{1n}\\
A_{21} & A_{22} & \cdots & A_{2n}\\
\vdots & \vdots & \ddots & \vdots\\
A_{n1} & A_{n2} & \cdots & A_{nn}
\end{pmatrix}
$$

**算符作用在向量上，翻译成矩阵乘列向量**：设 $|v\rangle = \sum_j v_j|j\rangle$，则

$$
\big(A|v\rangle\big)_i \;=\; \sum_{j=1}^{n} A_{ij}\, v_j
$$

即「结果向量的第 $i$ 个分量 = 矩阵第 $i$ 行与列向量 $v$ 的内积」。<span class="marginnote">这里用到了上一节的完备性关系：$A|v\rangle = \sum_{ij}|i\rangle\langle i|A|j\rangle\langle j|v\rangle$，把 $A$ 夹在一串基之间，再读出第 $i$ 个分量就是 $\sum_j A_{ij}v_j$。矩阵乘法本质上就是「外积重构 + 完备性插入」的坐标语言。</span>

> 辨析｜易错点： **算符与矩阵不是一回事**。算符是「空间到空间的线性映射」，不依赖坐标；矩阵是算符在**某个选定基**下的坐标表示。同一个算符 $X$，在 $\{|0\rangle,|1\rangle\}$ 基下是 $\begin{pmatrix}0&1\\1&0\end{pmatrix}$，在 $\{|+\rangle,|-\rangle\}$ 基下是 $\begin{pmatrix}1&0\\0&-1\end{pmatrix}$——矩阵不同，算符相同。说「算符等于矩阵」时，永远默认了某组基。

## 3 公式解析：矩阵元与算符的重建

**「矩阵元是算符在基之间的夹心」这句话，是理解一切量子门矩阵的钥匙。** 核心公式有两条：取矩阵元、重建算符。

$$
A_{ij} = \langle i|\,A\,|j\rangle, \qquad
A = \sum_{i,j} A_{ij}\,|i\rangle\langle j|
$$

拆成三步看：

**第一步，取矩阵元 $A_{ij} = \langle i|A|j\rangle$**。$A|j\rangle$ 是把算符作用到第 $j$ 个基向量上，得到一个向量；再用 $\langle i|$ 去「读」它的第 $i$ 个分量。因为基是标准正交的，$\langle i|A|j\rangle$ 精确地给出「从 $|j\rangle$ 出发、到 $|i\rangle$ 的转移幅度」——**矩阵元是「跳变振幅」**，这正是它在量子算法里扮演的角色。

**第二步，重建公式 $A = \sum_{ij} A_{ij}|i\rangle\langle j|$**。这等于说「把矩阵元当成外积 $|i\rangle\langle j|$ 的系数，加权求和就还原出整个算符」。验证它：作用在任意基向量 $|k\rangle$ 上，

$$
\sum_{i,j} A_{ij}\,|i\rangle\langle j|k\rangle
= \sum_{i,j} A_{ij}\,|i\rangle\,\delta_{jk}
= \sum_i A_{ik}\,|i\rangle
= A|k\rangle
$$

（最后一步 $\sum_i A_{ik}|i\rangle$ 正是「$A|k\rangle$ 按基展开成第 $k$ 列的系数」。）它对每个基向量都成立，所以重建公式成立。

**第三步，把两步合起来看**：取矩阵元（$A_{ij} = \langle i|A|j\rangle$）与重建算符（$A = \sum_{ij}A_{ij}|i\rangle\langle j|$）是一对互逆的操作——前者是「投影到坐标」，后者是「从坐标拼回几何对象」。<span class="marginnote">这个「坐标 ↔ 几何对象」的双向通道，正是线性代数的核心哲学：算符本身不依赖坐标，但坐标让我们能算。量子线路图上的每个门，在计算机内部都是一张这样的矩阵。</span>

## 4 Pauli 矩阵：量子计算的三块积木

把上面的一般理论落到三个最常用的算符上。Pauli 算符在计算基 $\{|0\rangle,|1\rangle\}$ 下的矩阵是：

$$
I = \begin{pmatrix}1&0\\0&1\end{pmatrix}, \qquad
X = \begin{pmatrix}0&1\\1&0\end{pmatrix}, \qquad
Y = \begin{pmatrix}0&-i\\i&0\end{pmatrix}, \qquad
Z = \begin{pmatrix}1&0\\0&-1\end{pmatrix}
$$

它们的物理作用一目了然：

- **$X$（比特翻转）**：$X|0\rangle = |1\rangle$，$X|1\rangle = |0\rangle$——把 0 变 1、1 变 0。
- **$Z$（相位翻转）**：$Z|0\rangle = |0\rangle$，$Z|1\rangle = -|1\rangle$——只给 $|1\rangle$ 乘上 $-1$ 的相位，不改概率。
- **$Y$（两者结合）**：$Y = iXZ$，同时翻转比特并翻转相位。<span class="marginnote">$Y$ 里的虚数 $i$ 不是摆设：它是为了满足 $Y^2 = I$ 而必需的相位因子。四个算符 $I,X,Y,Z$ 构成一组完备的线性基——任何 $2\times2$ 矩阵都能唯一写成 $aI+bX+cY+dZ$ 的形式，这是后面积分量子过程、做噪声建模时的出发点。</span>

Pauli 门还满足漂亮的代数关系：$X^2 = Y^2 = Z^2 = I$，以及反对易关系 $XZ = -ZX$（后一篇学幺正算符时，我们会看到它们如何被用来构造所有单比特旋转）。

## 5 算符的合成与对易

算符可以合成：$(AB)|v\rangle = A(B|v\rangle)$，「先做 $B$ 再做 $A$」。矩阵表示下就是矩阵乘法：

$$
(AB)_{ik} = \sum_j A_{ij}\,B_{jk}
$$

**辨析｜易错点：算符合成不满足交换律。** $AB$ 与 $BA$ 通常不同，顺序就是一切。定义**对易子（commutator）**

$$
[A, B] = AB - BA
$$

若 $[A,B]=0$，称 $A,B$ **对易**；否则**不对易**。量子计算里到处是不对易的例子：$XZ \neq ZX$（事实上 $XZ = -ZX$）。不对易意味着「先翻转比特再翻转相位」与「先翻转相位再翻转比特」结果不同——这正是量子门顺序不能乱排的代数根源，也是海森堡不确定关系的数学表述（第四篇《厄米算符与幺正算符》会回到这一点）。

## 6 用 Qiskit 亲眼看见矩阵

理论讲完，让计算机把矩阵直接打出来。Qiskit 的 `Operator` 类可以把一个线路编译成它对应的矩阵：

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

# X 门：比特翻转
qc = QuantumCircuit(1)
qc.x(0)
print("X =", Operator(qc))

# Z 门：相位翻转
qc2 = QuantumCircuit(1)
qc2.z(0)
print("Z =", Operator(qc2))
```

输出正是第 4 节的矩阵：`X = [[0+0j, 1+0j], [1+0j, 0+0j]]`，`Z = [[1+0j, 0+0j], [0+0j, -1+0j]]`。<span class="marginnote">在真实实验里，我们永远看不到「矩阵」——硬件执行的是物理脉冲。但一切量子算法、噪声模型、编译优化，最终都要落到这套矩阵表示上才能被经典软件处理。矩阵表示就是量子计算里「经典软件与量子硬件」之间的共同语言。</span>

试着把 `qc.x(0)` 换成 `qc.h(0)`（Hadamard 门），你会看到

$$
H = \frac{1}{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix}
$$

它把 $|0\rangle$ 变成 $(|0\rangle+|1\rangle)/\sqrt2$——制造叠加态的最基本门，正是「一个矩阵」在起作用。这就是本节的收官之点：**每一个量子门，都是一张矩阵；每一张矩阵，都描述一次线性变换。**

## 7 小结

- **线性算符**是量子态改变的方式：$A(\alpha|v\rangle+\beta|w\rangle) = \alpha A|v\rangle + \beta A|w\rangle$，量子门必须线性。
- **矩阵表示**依赖选基：矩阵元 $A_{ij} = \langle i|A|j\rangle$ 是「从 $|j\rangle$ 跳到 $|i\rangle$ 的振幅」，算符作用 = 矩阵乘列向量。
- **重建公式** $A = \sum_{ij} A_{ij}|i\rangle\langle j|$ 与取矩阵元互逆，是「坐标 ↔ 几何」的双向通道。
- **Pauli 门** $I,X,Y,Z$ 是量子计算的基础积木：$X$ 翻转比特、$Z$ 翻转相位、$Y=iXZ$ 两者兼有。
- **合成不对易**：$[A,B]=AB-BA$，量子门顺序不能乱排，这是很多量子现象的代数根源。

在下一节，我们将学习量子力学最关心的两类特殊算符——**厄米算符与幺正算符**：厄米算符描述「可以测量」的量，幺正算符描述「可以演化」的门，两者正是量子计算这台机器「读出」与「运转」的两套核心零件。
