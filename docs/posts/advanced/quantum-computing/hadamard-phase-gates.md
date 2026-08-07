---
title: Hadamard 门与相位门（S、T 门）
date: 2026-08-07
---

# Hadamard 门与相位门（S、T 门）

<div class="epigraph">
<p>如果你以为你懂了量子力学，那你一定没懂量子力学。</p>
<footer>—— 理查德 · 费曼（Richard Feynman）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen & Chuang《量子计算》§1.3.1、§4.3 单比特门 ｜ 2026-08-07</p>
</div>

## 为什么从 Hadamard 门开始

上一篇我们把 Pauli 门 X、Y、Z 摆上了桌。但若门库里只有 Pauli 门，量子计算和经典计算其实还看不出本质差别——因为 Pauli 门作用在计算基态上，只是把 $\lvert 0\rangle \leftrightarrow \lvert 1\rangle$ 换来换去、或加个相位，从不制造真正的新状态。<span class="marginnote">严格说：X、Y、Z 都把计算基态映射回计算基态，因此单独用它们跑任何线路，输出都和经典可逆电路一一对应。从「经典」到「量子」的跃迁，靠的是<strong>会制造叠加态</strong>的门——H 正是第一个。</span>

**真正让量子计算「量子」起来的第一个门，是 Hadamard 门 H。** 它从 $\lvert 0\rangle$ 出发造出叠加态 $\lvert +\rangle$，把一根「经典味」的线变成既能是 0、又能是 1 的量子线。几乎每一张量子算法线路图（Deutsch-Jozsa、QFT、Shor、Grover）都从一排 H 开始——H 是算法的「起跑器」：先把输入摊成均匀叠加，再让各个分支并行演化、相互干涉。

本篇还一并介绍两个更精细的相位门：**S 门**与**T 门**。它们看起来只是给 $\lvert 1\rangle$ 加了个复相位，却是「从离散砖块逼近任意单比特旋转」所必需的零件。在容错量子计算里，T 门的个数甚至被当作电路复杂度的度量（T-count）。费曼那句著名的自嘲放在这里恰到好处：H、S、T 行为如此简单，但把叠加与相位如何协同真正「算透」，正是量子计算的第一道坎。

## 1 Hadamard 门：制造叠加的门

**核心概念：** **Hadamard 门（H 门）**是如下的幺正矩阵：

$$
H = \frac{1}{\sqrt{2}}
\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}
$$

它对计算基态的作用是「各占一半」：

$$
H\lvert 0\rangle = \frac{\lvert 0\rangle + \lvert 1\rangle}{\sqrt{2}} = \lvert +\rangle, \qquad
H\lvert 1\rangle = \frac{\lvert 0\rangle - \lvert 1\rangle}{\sqrt{2}} = \lvert -\rangle
$$

$\lvert +\rangle$ 与 $\lvert -\rangle$ 合称 **X 基**（或对角基），因为它们是 X 门的两个本征态（特征值 $+1$ 与 $-1$，见上一篇）。**重点：H 把计算基「旋转」到 X 基，又因为 $H^2 = I$，它再作用一次就把自己变回去**——所以 H 既能把叠加态「制造」出来，也能把叠加态「拆」回计算基。这一来一回正是干涉实验的标准剧本：`H → 中间任意幺正门 → H` 的夹层结构，就是几乎所有量子算法的骨架。

在布洛赫球上，H 是绕「$x$ 轴与 $z$ 轴之间的对角线」转 $\pi$ 弧度的旋转：它把北极 $\lvert 0\rangle$ 送到赤道上的 $\lvert +\rangle$ 点，把南极 $\lvert 1\rangle$ 送到对面的 $\lvert -\rangle$ 点。<span class="marginnote">H 的旋转轴是 $\hat n = (\hat x + \hat z)/\sqrt{2}$，也就是布洛赫球上「东北方向」那条半径。为什么是 $\pi$？因为 $H^2 = I$ 而一次 H 显然不是恒等，所以它必然是「转半圈」——转半圈两次等于转一圈回到原处，这是对 $H^2 = I$ 最直接的几何理解。</span>

## 2 相位门 S 与 T：比 Z 更精细的旋转

上一篇讲过 Z 门 $\lvert 1\rangle \mapsto -\lvert 1\rangle$，即给 $\lvert 1\rangle$ 的分量乘上 $e^{i\pi} = -1$。**S 门与 T 门就是把这个相位旋转「切成更小的角度」**：

$$
S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}, \qquad
T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}
$$

S 给 $\lvert 1\rangle$ 乘 $e^{i\pi/2} = i$（转 $\pi/2$），T 给 $\lvert 1\rangle$ 乘 $e^{i\pi/4}$（转 $\pi/4$）。于是有一条漂亮的「相位标尺」：

$$
T \to T^2 = S \to S^2 = Z \to Z^2 = I
$$

**T 是相位旋转的「最小砖块」：$T^2 = S$、$S^2 = Z$。** 反过来读，Z 是四分之一转，S 是八分之一转，T 是十六分之一转——用足够多的 T，就能把相位旋转逼近到任意精度。这就是「任意单比特门可以只用 H、S、T 三兄弟逼近」这句话的直觉来源（严格的 Solovay-Kitaev 定理在第三篇后面专门讲）。

相位门对叠加态的作用值得亲手算一遍。设 $\lvert +\rangle = \frac{\lvert 0\rangle + \lvert 1\rangle}{\sqrt{2}}$，则

$$
S\lvert +\rangle = \frac{\lvert 0\rangle + i\lvert 1\rangle}{\sqrt{2}} = \lvert +i\rangle
$$

$\lvert +i\rangle$ 是 Y 门的 $+1$ 本征态（布洛赫球上 $+y$ 方向）。**辨析｜易错点：** S 不改变 $\lvert 0\rangle$、只给 $\lvert 1\rangle$ 加相位，所以在计算基测量里，$S\lvert 0\rangle$ 与 $\lvert 0\rangle$ 测出的都是 0——相位门对计算基「看不见」。但对叠加态，相对相位的变化会直接改变后续干涉的结果：$S\lvert +\rangle$ 与 $\lvert +\rangle$ 是**完全不同**的两个态（布洛赫球上一个在 $+x$、一个在 $+y$）。「对基态看不见 ≠ 没起作用」，这是相位门最常被低估的地方。

## 3 公式解析：为什么 $H = \frac{X + Z}{\sqrt{2}}$

这一节把 H 门拆成 Pauli 门的组合，这是理解 H 的代数本质、也是后面推导「$H$ 共轭 $X \leftrightarrow Z$」的钥匙。

**第一步，写出 $X$ 与 $Z$。** 上一篇的矩阵：

$$
X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \qquad
Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
$$

**第二步，相加再归一。** 直接做 $X + Z$：

$$
X + Z = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}
$$

除以 $\sqrt{2}$ 正是 H 矩阵。所以

$$
H = \frac{X + Z}{\sqrt{2}}
$$

这个式子说了三件事。第一，H 是「X 与 Z 各半」的混合，这对应它在布洛赫球上绕「$x$ 与 $z$ 之间」的轴旋转。第二，分母的 $\sqrt{2}$ 不是装饰——它保证列向量单位范数（$1^2 + 1^2 = 2$），从而 $H^\dagger H = I$。第三，**H 是厄米算符**：$H^\dagger = H$（矩阵等于自己的共轭转置），同时又是幺正的，于是 $H^\dagger H = H^2 = I$——这个「既厄米又幺正」的双重身份，让 H 同时扮演「变换」和「反变换」。

**第三步，推导共轭关系 $HXH = Z$。** 用上面三个事实：

$$
HXH = H \cdot X \cdot H
$$

代入 $H = \frac{X+Z}{\sqrt{2}}$，利用 $X^2 = I$、$Z^2 = I$、$XZ = -ZX$（Pauli 门两两反交换，见上一篇）：

$$
HXH = \frac{(X+Z)\,X\,(X+Z)}{2}
= \frac{(X+Z)(I + XZ)}{2}
= \frac{X + Z + XZX + ZXZ}{2}
$$

利用 $XZX = -Z$（因为 $XZ = -ZX$，两边左乘 $X$：$X X Z X = X(-ZX)$… 一步步代）与 $ZXZ = -X$，交叉项全部相消，剩

$$
HXH = \frac{X - X + Z - Z + 2Z}{2} = Z
$$

同理可得 $HZH = X$。**重点：H 门在共轭意义下把 X 与 Z 互换——「用 H 夹一下」，比特翻转变成相位翻转。** 这个 $HXH = Z$ 的「共轭引理」在后面构造 CZ 门、分析噪声、设计纠错码时会反复出现，值得现在就钉死。

## 4 相位标尺与 T-count

把相位门家族排成一列，你会看到一张「八分圆」：

$$
\underbrace{I}_{0°},\; \underbrace{T}_{\pi/4},\; \underbrace{S}_{\pi/2},\; \underbrace{Z}_{\pi},\; \underbrace{-I}_{2\pi}
$$

从 $I$ 到 $-I$ 是绕 $z$ 轴转一整圈，相位从 $0$ 走到 $2\pi$；S 是 Z 的平方根，T 是 Z 的四次方根。<span class="marginnote">「S = Z 的平方根」「T = Z 的四次方根」这种说法在文献里常见，记号是 $S = \sqrt{Z}$、$T = \sqrt[4]{Z}$。要注意这是<strong>门在代数上的根</strong>：$T^4 = Z$，而不是逐元素开方。</span>

为什么 T 格外重要？在容错量子计算里，只有 Clifford 门（H、S、CNOT、Pauli）能「廉价」地容错实现，而 T 门不是 Clifford 门——它无法被稳定子码的容错框架直接保护。因此一个算法的成本常常用 **T-count（T 门总个数）** 来估计。这解释了为什么 QFT、Shor 算法在文献里总是被反复优化「减少 T 门」——不是 T 门本身贵，而是它承载了「突破 Clifford 能力」的全部重担。

**辨析｜易错点：** 全局相位与相对相位不能混淆。$S$ 与 $e^{i\pi/4}S$ 在物理上等价（全局相位不可观测），但 $S$ 与「只给 $\lvert 1\rangle$ 加相位」的相对相位门完全不同。一个直观的判别法：**全局相位作用在每一个分量上、可以被整体提取；相对相位只作用在某些分量上、改变分量之间的干涉关系**。写代码时，Qiskit 的 `s` 门是相对相位门；若手滑写成了 `p(pi/2)`（作用于全局）再套测量，就会得到错误的统计分布。

## 5 一个可运行的示例（Qiskit）

用 Qiskit 验证 H、S、T 的矩阵作用，并检查 $H^2 = I$：

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

# H、S、T 对 |0⟩ 与 |1⟩ 的作用
for label, gate in [("H", "h"), ("S", "s"), ("T", "t")]:
    for ini in [0, 1]:
        qc = QuantumCircuit(1)
        if ini == 1:
            qc.x(0)
        getattr(qc, gate)(0)
        sv = Statevector(qc).data
        print(f"{label}|{ini}⟩ -> {sv}")

# H² = I 的检查
H = Operator(QuantumCircuit(1).h(0)).data
print("H² 与 I 一致：", (H @ H - np.eye(2) < 1e-12).all())
```

对照理论：$H\lvert 0\rangle = [1,1]/\sqrt{2}$、$H\lvert 1\rangle = [1,-1]/\sqrt{2}$、$S\lvert 1\rangle = [0,i]$、$T\lvert 1\rangle = [0, e^{i\pi/4}]$。这些数串就是「制造叠加」与「切分相位」的全部秘密——把 H 放上线路，`H` 后跟任意门、再跟 `H`，就构成了干涉的骨架。

## 6 小结

- **Hadamard 门** $H = \frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$：$H\lvert 0\rangle = \lvert +\rangle$、$H\lvert 1\rangle = \lvert -\rangle$，$H^2 = I$，是制造叠加态的第一块砖。
- 布洛赫球上 H 是绕「$x$ 与 $z$ 之间的对角轴」转 $\pi$ 的旋转；$H = \frac{X+Z}{\sqrt2}$ 是其代数本质。
- **共轭引理**：$HXH = Z$、$HZH = X$——H 在共轭下互换比特翻转与相位翻转。
- **相位门**：$S = \mathrm{diag}(1,i)$、$T = \mathrm{diag}(1,e^{i\pi/4})$；相位标尺 $T^2 = S$、$S^2 = Z$。
- 相对相位改变叠加态的干涉结果，但对计算基测量「看不见」；全局相位与相对相位不可混淆。
- **T-count** 是容错量子计算的主要成本指标，因为 T 门非 Clifford、难以廉价的容错实现。

在下一节，我们把 H、S、T 收进更一般的框架：**旋转门 Rx、Ry、Rz**——绕三个坐标轴的任意角度旋转，以及「任意单比特门 = 三次旋转之积」的分解定理。到那时，单比特门的图景才算完整。
