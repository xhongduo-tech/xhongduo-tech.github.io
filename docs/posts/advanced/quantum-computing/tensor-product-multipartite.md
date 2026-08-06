---
title: 张量积（tensor product）：多体系统的状态空间
date: 2026-08-07
---

# 张量积（tensor product）：多体系统的状态空间

<div class="epigraph">
<p>我不会称纠缠是量子力学的某一个方面，而宁愿说它是量子力学的典型特征——那个把量子力学与经典思路彻底区分开来的特征。</p>
<footer>—— 埃尔温 · 薛定谔（Erwin Schrödinger），《量子力学的现状》（1935）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen & Chuang《量子计算》§2.1 线性代数 ｜ 2026-08-07</p>
</div>

## 为什么从张量积开始

前几篇我们都在单系统里打转：一个量子比特的状态是 $\mathbb{C}^2$ 里的单位矢量，一个算符是 $\mathbb{C}^2$ 上的矩阵。但量子计算从来不止一个比特——Deutsch 算法两个比特，Shor 算法几百个比特。问题来了：**两个系统的状态空间该怎么拼？**

答案不是把维度相加，而是**相乘**。一个比特的态空间是二维，两个比特的态空间不是 $2+2=4$ 维的"平面"，而是 $2\times 2=4$ 维的"空间"。把两个系统拼起来的数学工具，就叫**张量积（tensor product）**。它看起来只是一条机械的构造规则，却在诞生处就埋下了量子计算最惊人的种子——**纠缠**。可以说，张量积是第二篇《Dirac 记号》之后，全书最重要的一个数学概念。

## 1 从「拼系统」到张量积

先看直观。系统 A 处于态 $|v\rangle$，系统 B 处于态 $|w\rangle$，两个系统彼此独立地拼成一个更大的系统，这个联合态写作

$$
|v\rangle \otimes |w\rangle
$$

读作「$|v\rangle$ 张量 $|w\rangle$」，也常简写为 $|v\rangle|w\rangle$ 甚至 $|vw\rangle$。**张量积是「并列组合」的记号**：它把「系统 A 处在这个态、系统 B 处在那一个态」的信息装进一个对象。

具体怎么算？设 $|v\rangle = \begin{pmatrix}v_1\\v_2\end{pmatrix}$，$|w\rangle = \begin{pmatrix}w_1\\w_2\end{pmatrix}$，则

$$
|v\rangle \otimes |w\rangle
= \begin{pmatrix} v_1 w_1 \\ v_1 w_2 \\ v_2 w_1 \\ v_2 w_2 \end{pmatrix}
$$

把第一个矢量的每个分量乘上整个第二个矢量。例如 $|0\rangle\otimes|1\rangle$：

$$
|0\rangle \otimes |1\rangle
= \begin{pmatrix}1\\0\end{pmatrix}\otimes\begin{pmatrix}0\\1\end{pmatrix}
= \begin{pmatrix}0\\1\\0\\0\end{pmatrix}
$$

<span class="marginnote">张量积与普通乘法最关键的差别是：它是<strong>双线性的</strong>——对加法和数乘都线性：$|v\rangle\otimes(|w\rangle+|w'\rangle) = |v\rangle\otimes|w\rangle + |v\rangle\otimes|w'\rangle$，且 $(\lambda|v\rangle)\otimes|w\rangle = \lambda(|v\rangle\otimes|w\rangle)$。这条性质保证叠加原理在多体系统里仍然成立。</span>

**辨析｜易错点：** 张量积不是点积、不是叉积，更不是逐个元素相乘。$|0\rangle\otimes|1\rangle$ 与 $|1\rangle\otimes|0\rangle$ 是两个**不同**的矢量——顺序至关重要，就像坐标 $(x,y)$ 与 $(y,x)$ 不同。

## 2 矩阵的 Kronecker 积

算符也要张量起来。两个矩阵 $A$（$m\times m$）与 $B$（$n\times n$）的张量积 $A\otimes B$ 是 $mn\times mn$ 的分块矩阵：

$$
A \otimes B
= \begin{pmatrix}
a_{11}B & a_{12}B & \cdots \\
a_{21}B & a_{22}B & \cdots \\
\vdots & \vdots & \ddots
\end{pmatrix}
$$

即把 $A$ 的每个元素 $a_{ij}$ 换成一个小块 $a_{ij}B$。<span class="marginnote">这个名字叫 <strong>Kronecker 积</strong>，以德国数学家利奥波德 · 克罗内克（Leopold Kronecker）命名。数学书里常写作 $A\otimes B$，矩阵计算的书里有时写作 `kron(A,B)`——线性代数与 MATLAB/NumPy 都这么叫。</span>

张量积有四条运算规则，几乎每个量子线路推导都会用到：

$$
(A\otimes B)(C\otimes D) = AC \otimes BD
$$

$$
(A\otimes B)^\dagger = A^\dagger \otimes B^\dagger
$$

$$
(A\otimes B)^{-1} = A^{-1} \otimes B^{-1}
$$

$$
(A\otimes B)(|v\rangle\otimes|w\rangle) = A|v\rangle \otimes B|w\rangle
$$

**第一条规则是最常用、也最容易被当成"元素相乘"而算错的**：它要求 $A$ 乘 $C$、$B$ 乘 $D$，各自独立地做矩阵乘法，**顺序不能乱**。第三条规则需要 $A, B$ 都可逆，幺正矩阵都满足。

## 3 双量子比特的计算基

两个量子比特的态空间是 $\mathbb{C}^2 \otimes \mathbb{C}^2 \cong \mathbb{C}^4$，一组标准正交基是四个「并列态」：

$$
|00\rangle,\quad |01\rangle,\quad |10\rangle,\quad |11\rangle
$$

它们分别是 $|0\rangle\otimes|0\rangle$、$|0\rangle\otimes|1\rangle$、$|1\rangle\otimes|0\rangle$、$|1\rangle\otimes|1\rangle$，在 $\mathbb{C}^4$ 里写成一排：

$$
|00\rangle = \begin{pmatrix}1\\0\\0\\0\end{pmatrix},\quad
|01\rangle = \begin{pmatrix}0\\1\\0\\0\end{pmatrix},\quad
|10\rangle = \begin{pmatrix}0\\0\\1\\0\end{pmatrix},\quad
|11\rangle = \begin{pmatrix}0\\0\\0\\1\end{pmatrix}
$$

于是任意双比特态都可以写成这组基的复线性组合：

$$
|\psi\rangle = \alpha|00\rangle + \beta|01\rangle + \gamma|10\rangle + \delta|11\rangle, \qquad |\alpha|^2 + |\beta|^2 + |\gamma|^2 + |\delta|^2 = 1
$$

<span class="marginnote"><strong>位序约定（endianness）</strong>：教材（Nielsen & Chuang）通常把 $|q_1\rangle|q_0\rangle$ 写成 $|q_1q_0\rangle$，第一位是最高位；而 Qiskit 使用小端序，`Circuit(2)` 的第 0 号量子比特在标签的<strong>最右</strong>。写程序时务必确认，否则张量积顺序会整体颠倒——这是 Qiskit 新手最常见的坑之一。</span>

维度相乘的规律在这里露了真容：$n$ 个量子比特的态空间是 $\mathbb{C}^{2^n}$。**每多一个比特，维度就翻一倍**——这就是量子计算机指数级状态空间的数学来源，也是第二篇《Dirac 记号》里「$2^n$ 个振幅」那句话的出处。

## 4 子系统上的算符与纠缠的诞生

联合系统上的算符不一定总是张量积形式，但「只动其中一个比特」的算符一定是。例如「在第一个比特上做 $X$、第二个比特不动」写成

$$
X \otimes I
$$

作用在基态上：$X\otimes I\,|01\rangle = (X|0\rangle)\otimes(I|1\rangle) = |1\rangle\otimes|1\rangle = |11\rangle$。这正是量子线路里一根线上画一个 $X$ 门、另一根线空着的数学版本。

现在到了本节最重要的观察。任意两个独立态的张量积 $|v\rangle\otimes|w\rangle$ 叫**乘积态（product state）**，它描述「两个系统各自有明确状态」的世界。但 $\mathbb{C}^4$ 里并非所有矢量都是乘积态。考虑

$$
|\Phi^+\rangle = \frac{1}{\sqrt2}\big(|00\rangle + |11\rangle\big)
$$

它能不能写成 $(\alpha|0\rangle+\beta|1\rangle)\otimes(\gamma|0\rangle+\delta|1\rangle)$？展开乘积态：$\alpha\gamma|00\rangle + \alpha\delta|01\rangle + \beta\gamma|10\rangle + \beta\delta|11\rangle$。要想不含 $|01\rangle, |10\rangle$，必须 $\alpha\delta = 0$ 且 $\beta\gamma = 0$；而四个系数同时非零（$|\Phi^+\rangle$ 里 $\alpha\gamma = \beta\delta = 1/\sqrt2 \neq 0$）时这两式不可能同时成立。**结论：$|\Phi^+\rangle$ 不是乘积态。**

这种「拆不成两个独立态」的态，就叫**纠缠态（entangled state）**。**纠缠的数学本质是：张量积空间比乘积态的集合大得多——绝大多数矢量都是纠缠的。** 纠缠正是薛定谔所说的「量子力学的典型特征」，也是量子并行、量子密钥分发和隐形传态（本专题第二篇《量子隐形传态》）的源头。

## 5 公式解析：$H\otimes H$ 作用于 $|00\rangle$

我们把张量积的规则串起来算一遍。设两个比特都处于 $|0\rangle$，各自过一个 Hadamard 门 $H = \frac{1}{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$。联合演化算符是 $H\otimes H$，求

$$
(H\otimes H)\,|0\rangle\otimes|0\rangle
$$

拆成三步：

- **第一步，用第三条规则拆开作用**。联合算符分别作用到各自的态上：

$$
(H\otimes H)\,|0\rangle\otimes|0\rangle = H|0\rangle \otimes H|0\rangle = |+\rangle \otimes |+\rangle
$$

其中 $H|0\rangle = \frac{1}{\sqrt2}(|0\rangle + |1\rangle) = |+\rangle$。

- **第二步，写出乘积态的显式形式**。代入 $|+\rangle = \frac{1}{\sqrt2}(|0\rangle + |1\rangle)$：

$$
|+\rangle\otimes|+\rangle
= \frac{1}{2}\Big(|0\rangle + |1\rangle\Big)\otimes\Big(|0\rangle + |1\rangle\Big)
$$

- **第三步，用双线性展开成计算基**。四对组合全部出现：

$$
= \frac12\Big(|0\rangle\otimes|0\rangle + |0\rangle\otimes|1\rangle + |1\rangle\otimes|0\rangle + |1\rangle\otimes|1\rangle\Big)
= \frac12\big(|00\rangle + |01\rangle + |10\rangle + |11\rangle\big)
$$

得到的是**等幅叠加的乘积态**——四个振幅各为 $1/2$，模方各为 $1/4$，恰好归一。注意它仍然是乘积态（等于 $|+\rangle|+\rangle$），**叠加 ≠ 纠缠**：一个态可以处处"糊在一起"却依然能拆成两个独立态的乘积。纠缠要求的是"拆不开"，这是比"叠加"强得多的性质。用 Qiskit 可以一步核对这个展开：

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

qc = QuantumCircuit(2)
qc.h(0)
qc.h(1)
print(Statevector(qc))   # 1/2 |00> + 1/2 |01> + 1/2 |10> + 1/2 |11>
```

## 6 小结

- **张量积**是把两个系统拼成一个的规则：$|v\rangle\otimes|w\rangle$ 每个分量与整个另一个矢量相乘；它是**双线性**的，且**顺序不能换**。
- 矩阵张量积（Kronecker 积）满足 $(A\otimes B)(C\otimes D) = AC\otimes BD$，$(A\otimes B)^\dagger = A^\dagger\otimes B^\dagger$。
- $n$ 个量子比特的态空间是 $\mathbb{C}^{2^n}$，基态是 $|00\cdots0\rangle$ 到 $|11\cdots1\rangle$；维度**相乘**而非相加。
- 子系统上的算符写成 $X\otimes I$ 的形式；张量积空间里**绝大多数矢量是纠缠态**，拆不成两个独立态。
- 叠加 ≠ 纠缠：$|+\rangle|+\rangle$ 是叠加也是乘积态；$|00\rangle+|11\rangle$ 才是纠缠。

在下一节，我们将把前面积累的全部工具——态矢量、厄米算符、幺正演化、张量积——收拢成量子力学的**四条基本假设**。那是量子计算从「数学」切换到「物理」的最后一步。
