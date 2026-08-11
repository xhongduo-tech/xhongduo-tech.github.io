---
title: 量子比特与量子门
date: 2026-08-11
---

# 量子比特与量子门

<div class="epigraph">
<p>我敢说，没有人真正理解量子力学。</p>
<footer>—— 理查德 · 费曼（Richard Feynman），The Character of Physical Law（1965）</footer>
</div>

<div class="article-byline">
<p>第四级 · 高阶专题 · 量子信息基础 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从量子比特开始

这门课叫《量子信息基础》，与经典信息学的分岔发生在「信息的最小单位」上：经典计算以**比特（bit）**为最小单位，量子计算以**量子比特（qubit）**为最小单位。表面上看只是把「0/1」换成「0 和 1 的叠加」，但这一处改动牵动全身——叠加带来的指数级并行、干涉带来的计算、纠缠带来的非局域关联，全部由此生长出来。<span class="marginnote">费曼在 1981 年「用计算机模拟物理」（Simulating Physics with Computers）的演讲中指出：想高效模拟量子世界，就必须用量子力学自己的计算机。这句话被公认为「量子计算」领域的发令枪。</span>

本专题后续每一课——测量、纠缠、算法、纠错、密码、信道、熵——都要反复使用本课的两件工具：**态矢量**与**幺正门**。把这两样读熟，整个专题就有了共同语言。这也是「从极限到大模型」主线里的一块新大陆：第一级《线性代数》里学过的矩阵、特征值与酉矩阵，将在这里获得它们在物理世界的第一个舞台。

## 1 从比特到量子比特

经典比特取两个确定值之一：$0$ 或 $1$。量子比特是一个**二维量子系统**，它的状态是复二维向量空间 $\mathbb{C}^2$ 中的一个单位向量：

$$
|\psi\rangle = \alpha|0\rangle + \beta|1\rangle, \qquad |\alpha|^2 + |\beta|^2 = 1
$$

其中 $|0\rangle$ 与 $|1\rangle$ 构成**计算基（computational basis）**，$\alpha, \beta \in \mathbb{C}$ 称为**概率幅（probability amplitude）**。测量这个量子比特时，读到 $|0\rangle$ 的概率是 $|\alpha|^2$，读到 $|1\rangle$ 的概率是 $|\beta|^2$，二者之和恒为 1，这正是归一化条件 $|\alpha|^2 + |\beta|^2 = 1$ 的物理含义。

**重点：量子比特不是「介于 0 与 1 之间的某个小数」，而是 0 与 1 两种成分按复系数叠加的状态。** 这就像一段音频不是「在某个音量上」，而是许多频率成分的叠加——只不过量子叠加的系数是复数，且叠加发生在测量之前。

**狄拉克符号（bra-ket notation）**是量子信息的地基语言：态 $|\psi\rangle$（ket，右矢）是列向量，其共轭转置写成 $\langle\psi|$（bra，左矢）是行向量，内积写作 $\langle\varphi|\psi\rangle$。物理上真实的例子比比皆是：电子的自旋上/自旋下、光子的偏振水平/竖直、原子核的自旋能级，都是天然的两能级量子比特<span class="marginnote">狄拉克符号由英国物理学家保罗 · 狄拉克（Paul Dirac）在 1939 年发明，本意是 bracket（括号）拆成 bra 与 ket。它把「向量 + 内积」压缩成两个符号，后文几乎所有公式都用它书写。</span>。

## 2 Bloch 球：一个量子比特的地理

叠加系数的模长与相位并非完全自由。任意单量子比特纯态总可以写成：

$$
|\psi\rangle = \cos\frac{\theta}{2}\,|0\rangle + e^{i\varphi}\sin\frac{\theta}{2}\,|1\rangle, \qquad 0 \le \theta \le \pi,\; 0 \le \varphi < 2\pi
$$

两个实数参数 $(\theta, \varphi)$ 恰好构成一个单位球面上的点 $(\sin\theta\cos\varphi,\, \sin\theta\sin\varphi,\, \cos\theta)$。这个球称为**Bloch 球（Bloch sphere）**：**纯态（pure state）对应球面上的点，球的北极是 $|0\rangle$，南极是 $|1\rangle$，赤道上的点都是「等概率叠加」但相位各异的态。**<span class="marginnote">为什么是 $\theta/2$ 而不是 $\theta$？因为 $|0\rangle$ 与 $|1\rangle$ 在态空间里正交（夹角 90°），却要映射到球面上一南一北（夹角 180°）。夹角被拉伸一倍，于是参数角要折半。这是「球面几何」与「态空间几何」之间一个极易混淆的换算。</span>

![单量子比特的 Bloch 球：纯态是球面上的点，北极 |0⟩、南极 |1⟩，|ψ⟩ 的位置由角度 θ 与相位 φ 决定](/images/quantum-information/qubits-and-quantum-gates-1.svg)

球心位置 $(\alpha=\beta=0)$ 对应**最大混合态** $\frac{1}{2}I$，它不是纯态，而是「一半概率处于 $|0\rangle$、一半概率处于 $|1\rangle$」的经典混和——一个「不知道是哪一个」的统计描述。

**辨析｜易错点：** 整体相位不可观测，相对相位才可观测。$|\psi\rangle$ 与 $e^{i\gamma}|\psi\rangle$（整体乘一个复数相位）描述同一个物理态，因为所有概率 $|\alpha|^2$ 不受影响；但 $|+\rangle = (|0\rangle+|1\rangle)/\sqrt2$ 与 $|-\rangle = (|0\rangle-|1\rangle)/\sqrt2$ 只差一个相对相位负号，却是完全不同的态——这正是干涉实验里「相消还是相长」的根源，我们会在量子算法一课再次遇到它。

## 3 单比特量子门：幺正旋转

在两次测量之间，封闭量子系统的演化是**确定性的、可逆的**，数学上由一个**幺正矩阵（unitary matrix）** $U$ 描述，满足 $U^\dagger U = I$（$U^\dagger$ 是 $U$ 的共轭转置）。幺正性保证概率守恒：$|\psi\rangle \to U|\psi\rangle$ 后范数不变，内积 $|\langle\varphi|\psi\rangle|$ 不变——量子门不做信息删除，它只是「旋转」。

几个最常用的单比特门（矩阵写作计算基 $\{|0\rangle,|1\rangle\}$）：

| 名称 | 矩阵 | 作用 |
| --- | --- | --- |
| Pauli-X（类比 NOT） | $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ | $X|0\rangle=|1\rangle$，$X|1\rangle=|0\rangle$，比特翻转 |
| Pauli-Z | $\begin{pmatrix}1&0\\0&-1\end{pmatrix}$ | $Z|0\rangle=|0\rangle$，$Z|1\rangle=-|1\rangle$，相位翻转 |
| Pauli-Y | $\begin{pmatrix}0&-i\\i&0\end{pmatrix}$ | $X$ 与 $Z$ 的组合：$Y=iXZ$ |
| Hadamard | $\frac{1}{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$ | 把基态送进均匀叠加 |
| 相位门 $S$ | $\begin{pmatrix}1&0\\0&i\end{pmatrix}$ | 只给 $|1\rangle$ 分量乘 $i$ |
| $\pi/8$ 门 $T$ | $\begin{pmatrix}1&0\\0&e^{i\pi/4}\end{pmatrix}$ | 细调相对相位 |

**重点：量子门全部可逆，这是它与经典门的本质区别。** 经典 AND、OR 都丢弃信息（输入不可由输出唯一还原），而量子演化保内积、可求逆。这也解释了为什么量子线路里没有经典的「COPY」（拷贝）指令——拷贝会重复信息，违背可逆性，这一点在第 5 节的不可克隆定理中会得到精确的表述<span class="marginnote">经典 NOT 门就是可逆的，而 AND 门不可逆。早在 1973 年，查尔斯 · 贝内特（Charles Bennett）就证明经典计算也能改造为可逆计算，只是要额外付出「垃圾位」的代价。量子计算只是把「可逆」从可选项变成了强制项。</span>。

## 4 多比特门与 CNOT

两个量子比特的态生活在 $\mathbb{C}^4$ 里，基向量是 $|00\rangle, |01\rangle, |10\rangle, |11\rangle$。最重要的两比特门是**受控非门（controlled-NOT，CNOT）**：

$$
\text{CNOT} = \begin{pmatrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 0 & 1\\
0 & 0 & 1 & 0
\end{pmatrix}, \qquad
\text{CNOT}|a,b\rangle = |a, b \oplus a\rangle
$$

它的语义是：第一个比特（控制位）为 $|1\rangle$ 时，翻转第二个比特（目标位）；为 $|0\rangle$ 时不动。$b \oplus a$ 是二进制加法模 2。CNOT 的威力在于它能把叠加「传递」过去：若控制位处于 $\frac{|0\rangle+|1\rangle}{\sqrt2}$，则

$$
\text{CNOT}\left(\frac{|0\rangle+|1\rangle}{\sqrt2}\otimes|0\rangle\right) = \frac{|00\rangle + |11\rangle}{\sqrt2}
$$

输入明明是各自独立的两个比特，输出却变成「要么全是 0、要么全是 1」的关联态——**CNOT 是制造纠缠的第一把扳手**。

**通用性定理**：任意 $n$ 比特的幺正变换，都可以由 CNOT 与单比特门组成的线路近似到任意精度。单比特门取一个非平凡集合（如 $\{H, S, T, \text{CNOT}\}$）就够了，这就是所谓「通用门集」。<span class="marginnote">Soloayv–Kitaev 定理给出逼近效率的保证：只需 $O(\log(1/\epsilon))$ 个门就能把误差压到 $\epsilon$ 以内。它让「有限几个离散门就能实现任意计算」从想法变成工程前提，量子纠错与容错计算都依赖它。</span>

## 5 量子线路与不可克隆定理

量子线路（quantum circuit）的画法约定：每条水平线代表一个量子比特（线从最左边的时间 $t_0$ 流向右），方框是单比特门，控制点加实心圆、目标位加 $\oplus$ 的是 CNOT。<span class="marginnote">线路图与经典逻辑门图表面相似，但含义完全不同：经典门图里的每条线携带确定比特，量子线路里的每条线携带叠加态，门是幺正的。</span>

**不可克隆定理（no-cloning theorem）**是量子信息最深刻的禁令：**不存在幺正变换 $U$，使得对任意未知态 $|\psi\rangle$ 都有 $U\,|\psi\rangle|0\rangle = |\psi\rangle|\psi\rangle$。** 证明只有三行：假设存在，对 $|\psi\rangle$ 与另一个态 $|\varphi\rangle$ 分别作用后取内积，左边得 $\langle\psi|\varphi\rangle$，右边得 $\langle\psi|\varphi\rangle^2$，于是 $\langle\psi|\varphi\rangle \in \{0,1\}$——即两个态要么正交、要么相同。换言之，只有正交态（其实是一类已知的特殊态）才可能被「完美复制」，一般量子态做不到。

这个定理是量子密码安全的根基（窃听者无法复制信道中的量子态），也是量子纠错难做的根源（无法简单地「备份」信息）。经典世界里「复印」「备份」天经地义，量子世界里线性代数一票否决——理解这一点，整个专题的许多反直觉结论都有了出发点。

## 6 公式解析：Hadamard 门

**为什么一个矩阵就能把确定态变成「同时是 0 也是 1」？** 我们从矩阵本身拆起。

$$H = \frac{1}{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$$

- **第一步，作用于 $|0\rangle$**：$H|0\rangle = \frac{1}{\sqrt2}(|0\rangle + |1\rangle)$。两个分量的模平方各是 $1/2$，测量时两种结果的概率各半——「确定」变成了「叠加」。
- **第二步，作用于 $|1\rangle$**：$H|1\rangle = \frac{1}{\sqrt2}(|0\rangle - |1\rangle)$。模平方同样各半，但 $|1\rangle$ 前的系数带负号。这个负号就是**相对相位**，它现在看不见，却在下一步干涉时出场。
- **第三步，再来一次 H**：由于 $H^2 = I$（H 自逆），$H(H|0\rangle) = |0\rangle$。把两次作用展开看：$(|0\rangle+|1\rangle)+(|0\rangle-|1\rangle) = 2|0\rangle$，两个分量里的 $|1\rangle$ 恰好**相消**，$|0\rangle$ 分量**相长**。于是叠加态又回到确定态。

第三步就是**量子干涉（interference）**的雏形：概率幅带相位，相位不同则加减抵消或加强，概率不是简单的叠加。量子算法的全部威力，都建立在这三个字上——让「好答案」的分量相长、让「坏答案」的分量相消。这也是对第 2 节「相对相位可观测」论断的第一次具体兑现。

## 7 小结

- **量子比特**是 $\mathbb{C}^2$ 中的单位向量 $\alpha|0\rangle+\beta|1\rangle$，测量读到 $|0\rangle$ 的概率为 $|\alpha|^2$；它不是一个介于 0 和 1 之间的数。
- Bloch 球用 $(\theta,\varphi)$ 直观表示单比特纯态；球心是最大混合态，纯态在球面上。
- **量子门是幺正变换** $U^\dagger U=I$，全部可逆；核心门有 Pauli-X/Y/Z、Hadamard、$S$、$T$。
- **CNOT** 是标准纠缠发生器：$\text{CNOT}\,\frac{|0\rangle+|1\rangle}{\sqrt2}|0\rangle = \frac{|00\rangle+|11\rangle}{\sqrt2}$；CNOT + 单比特门构成通用门集。
- **不可克隆定理**：未知量子态无法被完美复制，它是量子密码与量子纠错一切困难与安全性的共同源头。
- Hadamard 门自逆，其两次作用的干涉相消展示「量子干涉」的本质：概率幅带相位地加减。

在下一节，我们将回答本课反复回避的问题：**「读出」量子比特时究竟发生了什么？** 这就是量子测量——量子力学里最反直觉、也最需要精确语言的一环。
