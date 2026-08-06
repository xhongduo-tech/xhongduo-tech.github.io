---
title: 密度算符：混合态与部分迹
date: 2026-08-07
---

# 密度算符：混合态与部分迹

<div class="epigraph">
<p>量子力学令人印象深刻。但我内心的声音告诉我，这还不是真正的东西。这理论说得很多，却一点也没有让我们更接近老天的秘密。无论如何，我都确信，上帝不掷骰子。</p>
<footer>—— 阿尔伯特 · 爱因斯坦（Albert Einstein），致马克斯 · 玻恩的信（1926）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen & Chuang《量子计算》§2.4 密度算符 ｜ 2026-08-07</p>
</div>

## 为什么从密度算符开始

上一篇结尾我们把测量的问题逼到了墙角：态矢量 $|\psi\rangle$ 假定了「系统就是这一个纯态」，但现实有三个地方会破坏这个假定。第一，**我们可能根本不知道系统是哪个态**——只有一份「以概率 $p_i$ 处于态 $|\psi_i\rangle$」的清单；第二，系统可能只是**一个纠缠大系统里被拎出来的小碎片**——只看碎片时，它根本没有一个纯态可言（回想贝尔态的一半，纯态信息被纠缠「吃掉」了）；第三，系统可能与**环境**发生相互作用，演化不再是幺正的。

这三件事共享同一个数学工具：**密度算符（density operator / density matrix）**。它把「概率的不确定」与「纠缠造成的模糊」统一装进一个正半定、迹为 1 的算符里，也让量子测量、演化的公式全部以一种更普适的形式重写一遍。可以毫不夸张地说，**从本篇起，量子信息才真正开始**——量子信道、量子纠错、量子热力学，全部建立在密度算符的语言之上。它之于量子力学，就像**协方差矩阵之于统计**、**dropout 之于神经网络**：都是把「不确定性」系统化地记账。

## 1 从态矢量到密度算符

设系统处于态 $|\psi_i\rangle$ 的概率为 $p_i$（$\sum_i p_i = 1$，$p_i \geq 0$）。这样的系统用**密度算符**描述：

$$
\rho = \sum_i p_i\, |\psi_i\rangle\langle\psi_i|
$$

**纯态（pure state）**是它的特例：某 $p_k = 1$，其余为零，此时 $\rho = |\psi_k\rangle\langle\psi_k|$，即一个秩为 1 的投影算符。而一般的 $\rho$（秩大于 1）称为**混合态（mixed state）**。<span class="marginnote">密度矩阵最早由约翰 · 冯 · 诺伊曼（John von Neumann）在 1927 年引入。他当时关心的正是「量子统计力学」：一堆系统按概率分布处于不同纯态时，该怎么算平均值。今天这个词演化为「密度算符」——因为它是抽象的态空间上的算符，不依赖具体基的选择。</span>

为什么「不知道哪个态」非要用新的对象？因为态矢量 $\sum_i p_i|\psi_i\rangle$ 是**没有意义**的——把概率与态直接相加，既不是归一化态，也丢了概率的语义。正确的组合方式是**外积的加权和**。这一替换带来的最大好处，是把上一节的「测量概率」与「后测态」统一成干净的两行：

$$
p(m) = \mathrm{tr}\big(M_m \rho M_m^\dagger\big), \qquad \rho \to \frac{M_m \rho M_m^\dagger}{p(m)}
$$

而期望值变成一句漂亮的循环：**「观测均值」=「把可观测量与密度算符相乘再取迹」**：

$$
\langle M \rangle = \sum_i p_i \langle\psi_i|M|\psi_i\rangle = \mathrm{tr}(M\rho)
$$

推导只有两步：$\langle\psi_i|M|\psi_i\rangle = \mathrm{tr}(M|\psi_i\rangle\langle\psi_i|)$，再对 $i$ 求和，把 $\sum_i p_i$ 吸进 $\rho$。同样地，封闭系统的演化从 $|\psi\rangle \to U|\psi\rangle$ 变成

$$
\rho \to U\rho U^\dagger
$$

## 2 密度算符的三条基本性质

一个算符 $\rho$ 是合法密度算符，当且仅当它满足：

- **迹为 1**：$\mathrm{tr}(\rho) = 1$（总概率归一，由 $\sum_i p_i = 1$ 保证）；
- **厄米性**：$\rho^\dagger = \rho$（每个 $|\psi_i\rangle\langle \psi_i|$ 都厄米）；
- **非负性**：$\rho \geq 0$，即所有本征值非负（因为 $p_i \geq 0$）。

三条性质是测量概率 $p(m) = \mathrm{tr}(E_m\rho) \geq 0$ 与 $\sum_m p(m) = 1$ 的**充分必要**保障。还有一个区分纯态/混合态的量：**纯度（purity）** $\mathrm{tr}(\rho^2)$。

**重点：** 纯态与混合态的判据是 $\mathrm{tr}(\rho^2) = 1 \iff$ 纯态，$\mathrm{tr}(\rho^2) < 1 \iff$ 混合态。证明一蹴而就：$\rho$ 厄米，可在本征基下写成 $\rho = \sum_k \lambda_k |k\rangle\langle k|$，其中 $\lambda_k \geq 0$、$\sum_k\lambda_k = 1$；则 $\mathrm{tr}(\rho^2) = \sum_k \lambda_k^2$。而 $\sum_k \lambda_k^2 = 1$ 当且仅当某 $\lambda_k = 1$ 其余为 0——即纯态。纯度越小，系统越「混乱」；对 $d$ 维完全混合态 $\rho = I/d$，纯度取最小值 $1/d$。<span class="marginnote">纯度 $1 \to 1/d$ 像一个「有效维度的倒数」，与统计里的「有效自由度」、信息论里的「有效字母数」同构。最大混合态 $I/d$ 是「最无知」的状态：你只知道系统在 $d$ 维空间里，其余一概不知。</span>

密度算符还引出**冯 · 诺伊曼熵（von Neumann entropy）**，它是香农熵在量子世界的对应：

$$
S(\rho) = -\mathrm{tr}\big(\rho\log\rho\big) = -\sum_k \lambda_k\log\lambda_k
$$

$S(\rho) = 0$ 当且仅当纯态；完全混合态达到最大值 $\log d$。它将是第四篇《纠缠的度量》里度量纠缠的主角。

## 3 部分迹：从复合系统看子系统

现在处理第二种「模糊」：系统 AB 处于纯态 $|\psi\rangle_{AB}$，但我们只观察 A。A 的状态不是纯态，却也不是「不知道」——它是**约化密度算符（reduced density operator）**：

$$
\rho_A = \mathrm{tr}_B(\rho_{AB})
$$

其中 $\mathrm{tr}_B$ 叫**部分迹（partial trace）**，只对 B 的部分求迹。它的定义由线性与下面这条规则完全确定：对任意 $|a_1\rangle, |a_2\rangle \in \mathcal{H}_A$ 与 $|b_1\rangle, |b_2\rangle \in \mathcal{H}_B$，

$$
\mathrm{tr}_B\big(|a_1\rangle\langle a_2| \otimes |b_1\rangle\langle b_2|\big)
= |a_1\rangle\langle a_2| \cdot \langle b_2 | b_1 \rangle
$$

直观地读：**把 B 的两个「括号」闭合起来缩成一个数（内积），剩下的 A 部分原样保留**。$\mathrm{tr}_B$ 像一个「对 B 求和」的运算——这正是「把 B 丢掉」的数学翻译。<span class="marginnote">部分迹与求和的类比很贴切：如果 $\rho_{AB}$ 在基 $\{|a\rangle\otimes|b\rangle\}$ 下写成矩阵，那么 $\rho_A = \sum_b \langle b|\rho_{AB}|b\rangle$——就是把 B 的每个指标求和，像把二维数组沿某条轴约化。Qiskit 的 <code>partial_trace</code> 与 NumPy 的 <code>einsum('abij->ij', …)</code> 干的正是这件事。</span>

**辨析｜易错点：** 部分迹不是「取对角块」。$\rho_A$ 是 $\mathcal{H}_A$ 上的算符，它在 $A$ 的基下写作矩阵；而「取 $\rho_{AB}$ 在 $|a\rangle|b\rangle$ 下的对角块」只是在 $\mathcal{H}_B$ 固定一个基、对 B 求和的结果——两者在概念上等价，但 $\rho_A$ 本身不依赖 $B$ 的基的选择（可验证：换一个 $B$ 基，求和结果不变）。这正是一个好定义该有的样子。

## 4 公式解析：贝尔态的部分迹

把规则亲手用一遍。设 AB 处于最大纠缠态 $|\Phi^+\rangle = \frac1{\sqrt2}(|00\rangle + |11\rangle)$，求 A 的约化密度算符 $\rho_A$。

**第一步，写出 $\rho_{AB}$。** 外积展开，四个交叉项一个不少：

$$
\rho_{AB} = |\Phi^+\rangle\langle\Phi^+|
= \frac12\Big(|00\rangle\langle00| + |00\rangle\langle11| + |11\rangle\langle00| + |11\rangle\langle11|\Big)
$$

**第二步，对每一项做部分迹。** 用规则 $\mathrm{tr}_B(|a_1\rangle\langle a_2|\otimes|b_1\rangle\langle b_2|) = |a_1\rangle\langle a_2|\,\langle b_2|b_1\rangle$，逐项结算：

- $\mathrm{tr}_B(|00\rangle\langle00|) = |0\rangle\langle0|\cdot\underbrace{\langle0|0\rangle}_{1} = |0\rangle\langle0|$
- $\mathrm{tr}_B(|00\rangle\langle11|) = |0\rangle\langle1|\cdot\underbrace{\langle1|0\rangle}_{0} = 0$
- $\mathrm{tr}_B(|11\rangle\langle00|) = |1\rangle\langle0|\cdot\underbrace{\langle0|1\rangle}_{0} = 0$
- $\mathrm{tr}_B(|11\rangle\langle11|) = |1\rangle\langle1|\cdot\underbrace{\langle1|1\rangle}_{1} = |1\rangle\langle1|$

**第三步，相加。** 交叉项全被 $\langle b_2|b_1\rangle$ 杀死，剩下：

$$
\rho_A = \frac12\big(|0\rangle\langle0| + |1\rangle\langle1|\big) = \frac{I}{2}
$$

**这个结果值得停下来看三眼。** 第一眼：整个复合系统明明是纯态（$\mathrm{tr}(\rho_{AB}^2) = 1$），可单独看 A 却得到完全混合态 $I/2$（$\mathrm{tr}(\rho_A^2) = 1/2$）——**纠缠把子系统的「纯度」偷走了**，这正是「只看一半就丢失信息」的量化。第二眼：$\rho_A$ 与 B 的约化密度算符相同（$I/2$），对称地说明贝尔态里 A、B 地位平等。第三眼：如果我们**不知道** AB 是贝尔态、只知道 A 处于 $I/2$，那么「A 到底是贝尔态的一半，还是被人随机初始化成 $|0\rangle$ 或 $|1\rangle$？」——测量无法分辨，因为 $\rho_A$ 携带了 A 能提供的全部信息。用 Qiskit 可以一键验证：

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)                       # 构造贝尔态 |Φ⁺⟩
rho = DensityMatrix(qc)
rho_A = partial_trace(rho, [1])   # 丢掉第 1 个量子比特，留下第 0 个
print(rho_A)                      # [[0.5, 0], [0, 0.5]]，即 I/2
print(rho_A.purity())             # 0.5 < 1，混合态
```

## 5 辨析：叠加态 ≠ 混合态

初学者最容易犯的错误，是把「叠加」与「混合」混为一谈。它们都「既像 0 又像 1」，但本质不同：

- **叠加态** $|+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt2}$ 是一个**确定的纯态**，振幅之间存在确定的相对相位——它是**相干**的。
- **混合态** $\rho = \frac12 I = \frac12|0\rangle\langle0| + \frac12|1\rangle\langle1|$ 描述「有 50% 的概率是 $|0\rangle$、50% 是 $|1\rangle$」——相位关系被**抹掉了**。

两者能靠测量区分吗？能，关键在**基的选择**。在计算基 $\{|0\rangle,|1\rangle\}$ 下测，两者都各半概率，无法区分；但在 $X$ 基 $\{|+\rangle,|-\rangle\}$ 下测：

$$
\langle X \rangle_{|+\rangle} = \langle+|X|+\rangle = +1, \qquad
\langle X \rangle_{I/2} = \mathrm{tr}\Big(X\cdot\frac I2\Big) = \frac12 \mathrm{tr}(X) = 0
$$

$|+\rangle$ 在 $X$ 基下**每次都**给出 $+$，而 $I/2$ 各半。**$X$ 的期望值把叠加态与混合态彻底分开**。一句话总结：**叠加是「同一系统里两个振幅相干地并存」，混合是「不同系统（或不同时刻）按概率分布地各是各」**——前者需要相对相位，后者没有相位可言。

这条辨析在后面格外重要：**退相干（decoherence）的本质，就是相对相位被环境「平均掉」，把叠加态一点点磨成混合态**——那是第九篇《退相干、噪声与量子门保真度》的主线，也是 NISQ 时代所有量子计算机的头号敌人。

## 6 系综的歧义性：同一个 $\rho$，无数种出身

密度算符还有一个更微妙的性质：**同一个密度算符可以由完全不同的概率系综产生**。例如

$$
\frac12 I = \frac12|0\rangle\langle0| + \frac12|1\rangle\langle1|
= \frac12|+\rangle\langle+| + \frac12|-\rangle\langle-|
$$

左边说「50% 是 $|0\rangle$、50% 是 $|1\rangle$」，右边说「50% 是 $|+\rangle$、50% 是 $|-\rangle$」——两种制备方案，同一个 $\rho$。<span class="marginnote">这有点像统计里「两个不同的先验给出同一个边际分布」：先验之间的差别被「积分掉了」。信息论视角下，$\rho$ 记录了系统所有可观测量的期望，任何测量都无法区分两个给出同一 $\rho$ 的系综——密度算符是「可观测信息」的完备记账本，不多也不少。</span>

这个事实有两层含义。消极的一面：物理上「系统到底是什么系综」是一个无法用测量回答的问题，哲学上对应量子力学的**系综解释**之争。积极的一面：它给了我们**自由度**——在理论推导中，只要 $\rho$ 相同，我们可以**自由挑选最好算的系综**。隐形传态、量子信道容量的证明里，这种「换系综」的偷懒屡试不爽。

## 7 小结

- **密度算符** $\rho = \sum_i p_i|\psi_i\rangle\langle\psi_i|$ 统一描述：纯态、概率混合、子系统、开放系统；测量与演化公式为 $p(m)=\mathrm{tr}(M_m\rho M_m^\dagger)$、$\rho\to U\rho U^\dagger$。
- **三条性质**：$\mathrm{tr}\rho = 1$、$\rho^\dagger = \rho$、$\rho \geq 0$；**纯度** $\mathrm{tr}(\rho^2)$ 判定纯/混合，$S(\rho) = -\mathrm{tr}(\rho\log\rho)$ 是量子熵。
- **部分迹** $\rho_A = \mathrm{tr}_B(\rho_{AB})$ 用规则 $\mathrm{tr}_B(|a_1\rangle\langle a_2|\otimes|b_1\rangle\langle b_2|) = |a_1\rangle\langle a_2|\langle b_2|b_1\rangle$ 计算；贝尔态的一半是 $I/2$——纠缠偷走了子系统的纯度。
- **叠加 ≠ 混合**：相干相位是区别的关键，$X$ 基测量能分辨它们；退相干就是把相干磨掉的动态过程。
- **系综歧义**：同一 $\rho$ 可有无数种制备方式，测量不可分辨——这是自由度，也是哲学争论的源头。

在下一节，我们把「纠缠偷走纯度」这件事倒过来看：任意一个混合态 $\rho_A$，能不能把它「重新拼回」一个纯态 $|\psi\rangle_{AB}$，使得丢掉 B 恰好还原出 $\rho_A$？能——这就是 **Schmidt 分解与纯化**，也是理解纠缠的「标准形」工具。
