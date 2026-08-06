---
title: Dirac 记号（bra-ket）：右矢、左矢与内外积
date: 2026-08-07
---

# Dirac 记号（bra-ket）：右矢、左矢与内外积

<div class="epigraph">
<p>一种好的记号具有一种微妙而引人遐思的力量，有时令人觉得它几乎像一位活的老师。</p>
<footer>—— 伯特兰 · 罗素（Bertrand Russell），《数学原理》谈符号</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen & Chuang《量子计算》第2章 ｜ 2026-08-07</p>
</div>

## 为什么从 Dirac 记号开始

上一节我们有了复数域上的内积空间，却还在用「$\sum_i v_i^* w_i$」这种笨拙的坐标语言描述向量。物理学家 Paul Dirac 在 1939 年的一篇论文 *A New Notation for Quantum Mechanics* 里发明了一套记号，把「向量、对偶、内积、外积」压缩成几个顺手的小符号，从此成为量子力学与量子计算的标准语言。<span class="marginnote">Dirac 本人把这套记号称作「bra-ket」，因为内积 $\langle\phi|\psi\rangle$ 的记号由「bra」$\langle\phi|$ 与「ket」$|\psi\rangle$ 拼合而成——他在论文里幽默地说，bra 与 ket 拼在一起（bracket，即「括号」）。出处：P. A. M. Dirac, <i>Math. Proc. Camb. Phil. Soc.</i> 35 (1939) 416–418。</span>

这套记号不是可有可无的缩写。它把「向量是空间里的元素」这一几何直觉原样保留下来，让推导不依赖坐标——这正是量子计算里「换基」「投影」「求振幅」等高频操作能行云流水的原因。

## 1 右矢与左矢

**右矢（ket）**，记作 $|v\rangle$，就是上一节说的那个向量本身——$\mathbb{C}^n$ 里的一列数。<span class="marginnote">ket 是 vector（向量）去掉 vi 再拆成两半……Dirac 的原意更朴素：ket 与 bra 合称 bracket。今天我们把 $|v\rangle$ 读作「ket-v」或「右矢 v」，$\langle v|$ 读作「bra-v」或「左矢 v」。</span>

**左矢（bra）**，记作 $\langle v|$，是与 $|v\rangle$ 配对的「共轭转置」。如果 $|v\rangle$ 写成一列：

$$
|v\rangle = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}, \qquad
\langle v| = \begin{pmatrix} v_1^* & v_2^* & \cdots & v_n^* \end{pmatrix}
$$

即 $\langle v| = (|v\rangle)^\dagger$（$\dagger$ 表示共轭转置）。注意**每个分量都取复共轭**，这是上一节「内积对第一个分量共轭线性」的记号化：$|\psi\rangle = \alpha|0\rangle+\beta|1\rangle$ 对应 $\langle\psi| = \alpha^*\langle 0|+\beta^*\langle 1|$。

> 辨析｜易错点： $|v\rangle$ 与 $\langle v|$ 不是「同一个向量的两种写法」，而是**两个不同空间里的对象**：$|v\rangle$ 住在原空间 $V$，$\langle v|$ 住在其对偶空间 $V^*$。把 bra 想成「行向量、取共轭」只是方便记忆；严格地说，bra 是一个「从 $V$ 射到 $\mathbb{C}$ 的函数」。

## 2 内积：bra 与 ket 的结合

bra 与 ket 结合得到**一个复数**：

$$
\langle v|w\rangle = \sum_i v_i^*\, w_i
$$

这就是上一节的标准内积，现在换成了记号。它满足：

- **共轭对称**：$\langle v|w\rangle = \langle w|v\rangle^*$；
- **对 ket 线性、对 bra 共轭线性**：$\langle v|\,(\alpha|w_1\rangle+\beta|w_2\rangle) = \alpha\langle v|w_1\rangle + \beta\langle v|w_2\rangle$；
- **正定**：$\langle v|v\rangle = \sum_i |v_i|^2 \ge 0$。

量子计算里内积最常见的三个用途：算**归一化** $\langle\psi|\psi\rangle=1$、算**重叠度（fidelity）** $|\langle\phi|\psi\rangle|^2$、算**投影系数** $c_i = \langle i|\psi\rangle$。<span class="marginnote">把 $\langle i|\psi\rangle$ 想成「$|\psi\rangle$ 在基 $|i\rangle$ 方向上的分量」——它正是展开式 $|\psi\rangle = \sum_i c_i|i\rangle$ 里那个 $c_i$。测量得到 $|i\rangle$ 的概率 $|\langle i|\psi\rangle|^2$，不过是「分量的模方」。Born 规则用内积一句话就说完了。</span>

## 3 对偶空间与 Riesz 表示定理

为什么要有 bra 这个「双胞胎」？因为量子力学里我们经常要「把一个向量线性地变成数」——内积、期望值、测量概率都是这种操作。**线性泛函（linear functional）**：从 $V$ 到 $\mathbb{C}$ 的线性映射。所有线性泛函构成 $V$ 的**对偶空间（dual space）$V^*$**。

关键定理是 **Riesz 表示定理（Riesz representation theorem）**：对任意线性泛函 $f$，都存在**唯一的**向量 $|v\rangle$，使得

$$
f(|w\rangle) = \langle v|w\rangle, \qquad \forall\, |w\rangle
$$

换句话说，**「线性地把向量变成数」这件事，与「用某个 bra 做内积」这件事是一一对应的**。<span class="marginnote">Riesz 表示定理的证明思路：泛函 $f$ 的核是一个超平面，取核的正交补里那个单位向量 $|v\rangle$，就能验证 $\langle v|\cdot\rangle = f(\cdot)$。这个定理保证了 bra 与 ket 的一一对应（对有限维空间它几乎是显然的，但对无穷维 Hilbert 空间它是整个理论的地基）。</span>

这个观点非常重要：**bra 不是一个「画成行的向量」，而是一个「函数」**。一旦接受这一点，你会自然理解为什么下面外积的定义是「把两个 ket 粘在一起」而不是「两个 ket 相加」。

## 4 外积：把两个 ket 粘成一个算符

如果 bra 是「函数」，那么**外积（outer product）**就是把这个函数塞进一个算符里。定义

$$
|w\rangle\langle v|
$$

是一个从 $V$ 到 $V$ 的线性算符，作用规则是

$$
\big(|w\rangle\langle v|\big)\,|u\rangle \;=\; |w\rangle\,\langle v|u\rangle \;=\; \langle v|u\rangle\,|w\rangle
$$

先让 bra 吃掉 $|u\rangle$ 得到复数 $\langle v|u\rangle$，再把它乘到 $|w\rangle$ 上。<span class="marginnote">外积可以类比成矩阵「列 × 行」：$|w\rangle$ 是列向量、$\langle v|$ 是行向量，外积 $\begin{pmatrix}w_1\\w_2\end{pmatrix}(v_1^*\;v_2^*)$ 得到 $2\times2$ 矩阵，而内积 $\langle v|w\rangle = v_1^*w_1+v_2^*w_2$ 得到标量。内积是「行乘列」，外积是「列乘行」——顺序不同，结果不同，这正是第 5 节公式解析要用到的直觉。</span>

最常用的外积是**投影算符（projector）**：

$$
P_v = |v\rangle\langle v|
$$

它把任意 $|u\rangle$ 投到 $|v\rangle$ 张成的直线上：$P_v|u\rangle = \langle v|u\rangle\,|v\rangle$。投影算符是幂等的：$P_v^2 = P_v$——投两次等于投一次。下一节讲测量时，投影算符是核心道具。

## 5 公式解析：完备性关系 ∑|i⟩⟨i| = I

**完备性关系（completeness relation）是 Dirac 记号最锋利的武器**：对任意标准正交基 $\{|i\rangle\}$，有

$$
\sum_{i} |i\rangle\langle i| \;=\; I
$$

其中 $I$ 是恒等算符。这条式子让「在任意位置插入一组基」成为合法操作，量子算法里几乎所有推导都用它。拆成三步看它为什么成立：

**第一步，把任意向量按基展开**：设 $|\psi\rangle = \sum_i c_i |i\rangle$，其中系数正是内积 $c_i = \langle i|\psi\rangle$（这是基的标准正交性直接给出的）。

**第二步，把算符作用上去**：

$$
\Big(\sum_i |i\rangle\langle i|\Big)|\psi\rangle
= \sum_i |i\rangle\langle i|\psi\rangle
= \sum_i c_i\,|i\rangle
= |\psi\rangle
$$

**第三步，结论**：这个算符把**每个**向量都送回它自己，所以它等于恒等算符 $I$。<span class="marginnote">「等于恒等算符」的判定标准：对所有向量作用结果相同。线性算符由它在基上的作用完全决定，所以只要验证它对任意 $|\psi\rangle$ 成立即可。这条式子也叫「插入单位分解（inserting a resolution of the identity）」，在推导量子门的矩阵元、概率幅、以及后面积分路径计算时会反复出场。</span>

把第 5 节的结论与第 4 节合起来，还能写出行之最频繁的一条变形——**用外积重构恒等算符的意义，是「坐标系」本身可以被当作算符**：

$$
|\psi\rangle \;=\; \sum_i |i\rangle\langle i|\psi\rangle, \qquad
\langle\phi|\psi\rangle \;=\; \sum_i \langle\phi|i\rangle\langle i|\psi\rangle
$$

第二条叫**内积的基展开**：把内积拆成「经过全部中间基」的路径之和，这是理解纠缠态、路径积分与量子干涉的统一视角。

## 6 用 Dirac 记号重写之前的一切

把前几篇的公式全部用 bra-ket 重写一遍，记号的力量立刻显现：

- **单量子比特**：$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$，归一化 $\langle\psi|\psi\rangle = 1$；另一组基 $|+\rangle = \frac{|0\rangle+|1\rangle}{\sqrt2}$、$|-\rangle = \frac{|0\rangle-|1\rangle}{\sqrt2}$。
- **张量积（多体态）**：$|00\rangle = |0\rangle\otimes|0\rangle$，一般地 $|ab\rangle = |a\rangle\otimes|b\rangle$；$n$ 个比特的基是 $|x_1x_2\cdots x_n\rangle$。
- **贝尔态**：$|\Phi^+\rangle = \frac{1}{\sqrt2}\big(|00\rangle + |11\rangle\big)$——一个「两个比特互相纠缠」的态，它的特点是不能被写成 $|a\rangle\otimes|b\rangle$ 的形式，这个性质在第 4 节学纠缠时会是主角。
- **投影到新基**：$\langle +|0\rangle = \frac{1}{\sqrt2}$，所以「在 $|+\rangle,|-\rangle$ 基下测 $|0\rangle$」得到 $|+\rangle$ 的概率是 $|\langle+|0\rangle|^2 = \frac12$——这是 Hadamard 门制造叠加的代数表达。

如果想亲手验证这些记号，Python 的 numpy 就能把 bra-ket 翻译成数组运算：

```python
import numpy as np

ket0 = np.array([1, 0]); ket1 = np.array([0, 1])
bra0 = ket0.conj()          # 共轭转置 = bra
psi = (ket0 + ket1) / np.sqrt(2)   # |+⟩

inner = bra0 @ psi          # ⟨0|ψ⟩ = 1/√2
outer = np.outer(psi, psi.conj())  # |ψ⟩⟨ψ| 投影算符
print(inner, outer, sep="\n")
```

输出里的 `outer` 矩阵正是投影算符 $|+\rangle\langle+|$ 的矩阵表示——这也是下一节「算符与矩阵」的预告：**每一个 Dirac 记号的式子，最终都能翻译成一串矩阵运算，而反过来，任何矩阵运算都能写成 bra-ket 的外积组合。**

## 7 小结

- **ket $|v\rangle$** 是向量（列），**bra $\langle v|$** 是其共轭转置，住在对偶空间；二者是不同空间的对象。
- **内积** $\langle v|w\rangle$ 是复数，对 ket 线性、对 bra 共轭线性；它给出归一化、重叠度与投影系数 $c_i = \langle i|\psi\rangle$。
- **Riesz 表示定理**：线性泛函与 bra 一一对应——bra 的本质是「把向量变成数的函数」。
- **外积** $|w\rangle\langle v|$ 是算符，投影算符 $P_v = |v\rangle\langle v|$ 幂等 $P_v^2 = P_v$。
- **完备性关系** $\sum_i |i\rangle\langle i| = I$ 允许在任何位置插入一组基，是所有推导的万能工具。

在下一节，我们将把「外积的组合」正式化——**线性算符与矩阵表示**：学习如何把一个算符写成矩阵、矩阵元 $\langle i|A|j\rangle$ 如何取，以及 Pauli 门为什么是量子计算的三块积木。
