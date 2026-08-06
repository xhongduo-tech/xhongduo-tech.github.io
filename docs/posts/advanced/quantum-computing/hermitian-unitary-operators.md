---
title: 厄米算符与幺正算符
date: 2026-08-07
---

# 厄米算符与幺正算符

<div class="epigraph">
<p>数学语言在表述物理定律时的恰当性，是一个我们既不理解也不配拥有的奇妙礼物。</p>
<footer>—— 尤金 · 维格纳（Eugene Wigner），《数学在自然科学中不合理的有效性》（1960）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen & Chuang《量子计算》§2.1 线性代数 ｜ 2026-08-07</p>
</div>

## 为什么从厄米算符与幺正算符开始

上一篇《线性算符与矩阵表示》里，我们把线性算符请进了矩阵的框架：一个算符就是一架「对矢量做线性加工」的机器，有矩阵可写、有迹可循。但量子力学并不需要**所有**这种机器——它只认两类。<span class="marginnote">「只认两类」不是审美偏好，而是物理需要：测量的结果必须是实数，所以描写测量的算符必须保证实数的期望；概率在演化中必须守恒，所以演化算符必须保范数。这两条硬约束，就分别逼出了厄米与幺正。</span>

第一类叫**厄米算符（Hermitian operator）**，它描写**可观测的量**——能量、自旋、位置，凡是「可以测出实数结果的物理量」，都是厄米算符。第二类叫**幺正算符（unitary operator）**，它描写**态的演化**——一个量子比特从 $|0\rangle$ 变成 $|+\rangle$，一套线路从输入变成输出，凡是「封闭系统随时间的变化」，都是幺正算符。

换句话说：**厄米算符是量子力学的「测量之手」，幺正算符是量子力学的「演化之手」。** 而这两只手之间有一条隐秘的纽带——把厄米算符放进指数函数，就得到幺正算符。今天这篇，就是看清这两只手，以及它们之间那条纽带。

## 1 共轭转置：一切的定义原点

定义这两类算符，都需要同一个操作：**共轭转置（conjugate transpose）**，也叫**伴随（adjoint）**，记作 $A^\dagger$。对一个矩阵 $A$，先转置、再把每个元素取复共轭：

$$
A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}
\quad\Longrightarrow\quad
A^\dagger = \begin{pmatrix} a^* & c^* \\ b^* & d^* \end{pmatrix}
$$

符号 $\dagger$（读作 dagger，剑号）来自狄拉克（P. A. M. Dirac）。<span class="marginnote">这里最容易算错的一步是顺序：<strong>先转置，后共轭</strong>。先共轭后转置结果一样，所以顺序本身无所谓；真正要小心的是——<strong>乘积的共轭转置要把因子倒过来</strong>：$(AB)^\dagger = B^\dagger A^\dagger$。这和 $(AB)^{-1} = B^{-1}A^{-1}$ 是同一个「穿袜子要最后脱」的道理。</span>

共轭转置还有一个更本质的身份：它是「搬动内积」的规则。对任意两个态 $|\psi\rangle, |\phi\rangle$ 与算符 $A$，恒有

$$
\big(\langle\psi|\, A\,|\phi\rangle\big)^* = \langle\phi|\, A^\dagger\, |\psi\rangle
$$

左边的复数取共轭，等于把 $A$ 换成 $A^\dagger$、把两个态对调。这一条规则看似只是记号，却是后面证明「厄米算符本征值为实数」的出发点。

有了共轭转置，两类主角就可以一句话定义了：

- **厄米算符**：$A^\dagger = A$，算符等于自己的共轭转置；
- **幺正算符**：$U^\dagger U = UU^\dagger = I$，算符的共轭转置是自己的逆。

## 2 厄米算符：可观测量的化身

**厄米算符（Hermitian operator）**：满足 $A^\dagger = A$ 的算符。它的第一个立即推论是：对任何态 $|\psi\rangle$，对角元都是实数

$$
\langle\psi|\,A\,|\psi\rangle^* = \langle\psi|\,A^\dagger\,|\psi\rangle = \langle\psi|\,A\,|\psi\rangle
$$

一个复数等于自己的共轭，它只能是实数。这个「对角元实数」的性质，正是厄米算符能描写测量的全部秘密：**可观测量的期望值 $\langle A \rangle = \langle\psi|A|\psi\rangle$ 是一个实数，而物理测量只能给出实数。** 例如哈密顿量 $H$（能量算符）是厄米的，Pauli 矩阵 $\sigma_x, \sigma_y, \sigma_z$ 全是厄米的。

厄米算符还有两个在下一篇《本征值、本征向量与谱分解》里要系统展开的性质，这里先预告：

- **本征值是实数**——可观测量的可能取值（测量结果）因此都是实数；
- **不同本征值对应的本征向量相互正交**——不同的测量结果对应互不重叠的态。

## 3 幺正算符：演化的化身

**幺正算符（unitary operator）**：满足 $U^\dagger U = UU^\dagger = I$ 的算符。它的核心性质是**保内积**：对任意 $|\psi\rangle, |\phi\rangle$，

$$
\langle U\psi \,|\, U\phi\rangle = \langle\psi |\, U^\dagger U \,| \phi\rangle = \langle\psi|\phi\rangle
$$

内积守恒，范数自然守恒：$\langle U\psi|U\psi\rangle = \langle\psi|\psi\rangle$。这恰恰是概率要满足的性质——态 $|\psi\rangle$ 的范数平方是总概率，它必须恒等于 1；只有幺正演化能保证这一点。这也解释了为什么量子线路里的门全是幺正矩阵：$X$、$H$、$S$、$T$、CNOT，无一例外。<span class="marginnote">幺正算符在矩阵层面的形象是：<strong>列向量组成一组标准正交基</strong>。检查一个矩阵是否幺正，最直接的办法就是看它的列是否两两正交且模长为 1——这对接下来手算各种量子门非常实用。</span>

幺正算符的**本征值都在单位圆上**（模长为 1）。这和厄米算符形成对照：厄米的本征值是实数（在实轴上），幺正的本征值在单位圆上。**辨析｜易错点：** 很多人看到 $U^\dagger = U^{-1}$ 就以为幺正和「正交」是一回事。严格说，正交矩阵是实数域上的幺正矩阵（$O^TO = I$）；幺正矩阵允许复元素，是它的推广。另外，**一个矩阵可以既是厄米又是幺正**——当且仅当它的本征值全是 $\pm 1$，Pauli 门和 Hadamard 门正是这样的特例。

## 4 二者之间的桥：厄米指数化

厄米与幺正看起来是两路人，但有一条公式把它们焊在一起：

$$
A \text{ 是厄米的} \quad\Longrightarrow\quad e^{iA} \text{ 是幺正的}
$$

反过来也对：任何幺正算符 $U$ 都能写成 $U = e^{iH}$，其中 $H$ 是某个厄米算符。指数函数在这里是「从可测量到演化」的生成器。

这条桥在物理上就是量子力学的**薛定谔演化**：哈密顿量 $H$ 是厄米的（可测量——能量），时间演化算符

$$
U(t) = e^{-iHt/\hbar}
$$

是幺正的（演化）。**同一个厄米算符，取指数就生成演化**——这就是「观测」与「演化」在数学上最深刻的联系。量子线路里的旋转门也是这个模式：例如绕 $x$ 轴的旋转门 $R_x(\theta) = e^{-i\theta X/2}$，正是厄米算符 $X$ 的指数。

## 5 公式解析：为什么 $e^{i\theta \sigma_x}$ 是幺正的

我们用一个具体的矩阵，亲手验证「厄米指数化 → 幺正」这条桥。取 Pauli 矩阵 $\sigma_x = \begin{pmatrix}0&1\\1&0\end{pmatrix}$，构造 $e^{i\theta\sigma_x}$，证明它对任意实数 $\theta$ 都是幺正的。拆成三步：

- **第一步，利用 $\sigma_x^2 = I$ 化简指数**。对任何矩阵，$e^{A} = \sum_{k=0}^\infty A^k/k!$。由于 $\sigma_x^2 = I$，所有偶数次幂都是 $I$，奇数次幂都是 $\sigma_x$，于是偶数项求和得 $\cos\theta\,I$，奇数项求和得 $i\sin\theta\,\sigma_x$：

$$
e^{i\theta\sigma_x} = \cos\theta\, I + i\sin\theta\,\sigma_x
= \begin{pmatrix} \cos\theta & i\sin\theta \\ i\sin\theta & \cos\theta \end{pmatrix}
$$

- **第二步，对它取共轭转置**。$I^\dagger = I$，$\sigma_x^\dagger = \sigma_x$，但 $i$ 的共轭是 $-i$，所以

$$
\big(e^{i\theta\sigma_x}\big)^\dagger = \cos\theta\, I - i\sin\theta\,\sigma_x
$$

- **第三步，两者相乘**。因为 $I$ 与 $\sigma_x$ 对易，直接展开：

$$
\big(e^{i\theta\sigma_x}\big)^\dagger e^{i\theta\sigma_x}
= \cos^2\theta\, I + \sin^2\theta\, \sigma_x^2
= (\cos^2\theta + \sin^2\theta)\, I = I
$$

交叉项 $\pm i\sin\theta\cos\theta$ 恰好抵消，而 $\sigma_x^2 = I$。结果就是恒等算符，故 $e^{i\theta\sigma_x}$ 是幺正的。用 Qiskit 可以一秒验证这件事：

```python
from qiskit.quantum_info import Operator, rx

u = Operator(rx(0.5))        # R_x(θ) = e^{-iθX/2}，一个幺正旋转门
print(u.is_unitary())        # True
```

注意 Qiskit 里的 `rx(θ)` 是 $e^{-i\theta X/2}$，指数上的负号只是约定，并不影响「厄米指数生成幺正」的结论。

## 6 小结

- **共轭转置** $A^\dagger$ 是定义一切的基础：$(AB)^\dagger = B^\dagger A^\dagger$，且 $\langle\psi|A|\phi\rangle^* = \langle\phi|A^\dagger|\psi\rangle$。
- **厄米算符** $A^\dagger = A$：对角元（期望值）为实数，描写**可观测的量**。
- **幺正算符** $U^\dagger U = I$：保内积、保范数，描写**封闭系统的演化**，量子门都是幺正的。
- 厄米与幺正之间由**指数桥**连接：$A$ 厄米 $\Rightarrow e^{iA}$ 幺正；$e^{i\theta\sigma_x}$ 的幺正性可逐步验证。
- 厄米的本征值是实数，幺正的本征值在单位圆上；两者兼有当且仅当本征值为 $\pm 1$。

在下一节，我们将回答两个问题：厄米算符的实数本征值从哪来？一个算符如何被拆成「本征值 × 投影」之和？这就是**本征值、本征向量与谱分解**。
