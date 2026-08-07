---
title: 受控门：CNOT、CZ 与受控-U
date: 2026-08-07
---

# 受控门：CNOT、CZ 与受控-U

<div class="epigraph">
<p>缠结并非量子力学中某一个可选的奇特特征，而正是那个迫使它彻底背离经典思维路线的特征。</p>
<footer>—— 埃尔温 · 薛定谔（Erwin Schrödinger）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen & Chuang《量子计算》§4.3 受控门 ｜ 2026-08-07</p>
</div>

## 为什么从受控门开始

单比特门再多，也只能让每个量子比特各自旋转——而量子计算的威力恰恰来自**比特之间的关联**。**受控门（controlled gate）**是「两个比特之间的条件操作」：当控制比特处于 $\lvert 1\rangle$ 时，才对目标比特施加某个门 $U$。别小看这个「如果」，它是量子线路里唯一能制造**纠缠**的引擎，也是从「并行的单比特旋转」到「量子并行」的关键一步。

薛定谔那句话用在这里再合适不过：CNOT 把一个处于叠加态的控制比特和一个目标比特耦合起来，得到的状态无法再写成两个单比特态的乘积——这就是缠结。上一轮我们已经在《多量子比特系统与纠缠态》《贝尔态》里反复用 CNOT 造过贝尔态；本篇要做的，是把「受控」这个操作本身彻底解剖：它的矩阵长什么样、CZ 与 CNOT 如何互相转化、任意受控-U 怎么由 CNOT 和单比特门搭出来。这些是通向 Toffoli 门、量子加法器乃至 Shor 算法中模幂运算的必经之路。

## 1 受控-U 的通用定义

**核心概念：** 设 $U$ 是一个单比特门。**受控-U（controlled-$U$）**是作用在两个量子比特上的双比特门：第一个比特叫**控制位（control）**，第二个叫**目标位（target）**。规则是——控制位为 $\lvert 1\rangle$ 时施加 $U$，控制位为 $\lvert 0\rangle$ 时不施加：

$$
\lvert c\rangle \lvert t\rangle \;\longmapsto\; \lvert c\rangle\, U^{c}\lvert t\rangle
$$

记号 $U^c$ 表示「$c = 0$ 时取 $I$、$c = 1$ 时取 $U$」。在计算基 $\{\lvert 00\rangle, \lvert 01\rangle, \lvert 10\rangle, \lvert 11\rangle\}$ 下，受控-U 的矩阵是分块对角形式：

$$
C(U) = \begin{pmatrix}
I & 0\\
0 & U
\end{pmatrix}
$$

左上块 $I$ 管「控制位是 0」的两行，右下块 $U$ 管「控制位是 1」的两行。**重点：受控门总是幺正的**——因为 $C(U)^\dagger C(U) = \mathrm{diag}(I^\dagger I, U^\dagger U) = \mathrm{diag}(I, I) = I$。所以受控门完全可逆：$C(U)^{-1} = C(U^\dagger)$。

当 $U = X$ 时，受控-U 就是 **CNOT（controlled-NOT）门**；当 $U = Z$ 时就是 **CZ 门**。它们是本篇的主角，也是后面所有多比特门（Toffoli、Fredkin、量子加法器）的积木。<span class="marginnote">线路图里，受控门的标准画法是：控制位画实心点 ●，从它引一根竖线到目标位；目标位上画 $U$ 方框（CNOT 通常画 $\oplus$）。控制位为 1 才触发，这个「1」的约定是整个受控家族的统一规则。</span>

## 2 CNOT：量子条件门

**核心概念：** **CNOT 门**（受控非门）是 $U = X$ 的受控门，它在计算基下的作用是

$$
\lvert c\rangle\lvert t\rangle \longmapsto \lvert c\rangle\lvert t \oplus c\rangle
$$

其中 $\oplus$ 是模 2 加法（XOR）。展开就是：$\lvert 00\rangle \mapsto \lvert 00\rangle$、$\lvert 01\rangle \mapsto \lvert 01\rangle$、$\lvert 10\rangle \mapsto \lvert 11\rangle$、$\lvert 11\rangle \mapsto \lvert 10\rangle$。矩阵（在 $\{\lvert 00\rangle,\lvert 01\rangle,\lvert 10\rangle,\lvert 11\rangle\}$ 基下）：

$$
\mathrm{CNOT} = \begin{pmatrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 0 & 1\\
0 & 0 & 1 & 0
\end{pmatrix}
$$

CNOT 与经典 XOR 的关系值得说清楚：**CNOT 把目标位变成「控制位 XOR 目标位」，但控制位原样保留**——所以它是「可逆的 XOR」。经典 XOR 是不可逆的（从输出推不回两个输入），而 CNOT 把一份输入留在控制位里，恰好补回了丢失的信息，因此可逆，且 $\mathrm{CNOT}^{-1} = \mathrm{CNOT}$（自己就是自己的逆）。<span class="marginnote">这正是可逆计算的起点：经典电路丢信息（AND 门从输出推不回输入），量子电路通过「保留一份副本」让所有操作可逆。兰道尔原理说丢 1 比特信息至少耗散 $k_B T\ln 2$ 能量，可逆计算可以绕开这个下限——详见第三级《计算机组成原理》关于 Landauer 原理的讨论。</span>

**重点：CNOT 是纠缠的引擎。** 让控制比特处于叠加态 $\lvert +\rangle = \frac{\lvert 0\rangle + \lvert 1\rangle}{\sqrt2}$、目标比特处于 $\lvert 0\rangle$，则

$$
\mathrm{CNOT}\left(\frac{\lvert 0\rangle + \lvert 1\rangle}{\sqrt2} \otimes \lvert 0\rangle\right)
= \frac{\lvert 00\rangle + \lvert 11\rangle}{\sqrt2} = \lvert \Phi^+\rangle
$$

这是第一个贝尔态。**注意：控制比特叠加的两支分别走了「不触发」与「触发」两条路，输出态的两项对应两个不同的控制位取值**——两支历史被目标比特「标记」了，从此无法再写成单个比特态的乘积。CNOT 一出手，纠缠就出现了；而没有纠缠，量子算法（隐形传态、超密编码、Shor）全都无从谈起。

## 3 CZ 门：受控相位翻转

**核心概念：** **CZ 门（受控-Z）**是 $U = Z$ 的受控门，它在计算基下的作用是「当且仅当两个比特都是 $\lvert 1\rangle$ 时，给状态加一个负号」：

$$
\mathrm{CZ}\lvert c\rangle\lvert t\rangle = (-1)^{c\cdot t}\lvert c\rangle\lvert t\rangle
$$

矩阵：

$$
\mathrm{CZ} = \begin{pmatrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & -1
\end{pmatrix}
$$

CZ 有两个和 CNOT 很不一样的性质。第一，**CZ 是「对称」的**——控制位与目标位完全对等，把两个比特互换，矩阵不变（$(-1)^{c\cdot t} = (-1)^{t\cdot c}$）。线路图里 CZ 的两个点都画成实心 ●，无法区分谁控制谁。第二，CZ 与 CNOT 只差一层 H：把目标比特先 H、过 CNOT、再 H，得到的正是 CZ。下一节我们把这个关系算到底。

**重点：CZ 在计算基下只对 $\lvert 11\rangle$ 起作用，看起来「温和」，但它同样能制造纠缠。** 例如对 $\lvert ++\rangle$ 施加 CZ：$\lvert ++ \rangle = \frac{1}{2}(\lvert 00\rangle + \lvert 01\rangle + \lvert 10\rangle + \lvert 11\rangle)$，CZ 把 $\lvert 11\rangle$ 一项变负，得到 $\lvert 00\rangle + \lvert 01\rangle + \lvert 10\rangle - \lvert 11\rangle$ 的归一化形式——这个态同样是纠缠态。CZ 与 CNOT 在「制造纠缠」这件事上完全等价，因为 H 可以来回切换它们。

## 4 公式解析：$\mathrm{CZ} = (I \otimes H)\,\mathrm{CNOT}\,(I \otimes H)$

这一节把「CZ 与 CNOT 只差一层 H」用矩阵算出来。选目标比特作第二比特，即 $H$ 作用在目标位上（写成 $I \otimes H$）。

**第一步，写出三个因子。** 在 $\{\lvert 00\rangle, \lvert 01\rangle, \lvert 10\rangle, \lvert 11\rangle\}$ 基下：

$$
I \otimes H = \frac{1}{\sqrt{2}}
\begin{pmatrix}
1 & 1 & 0 & 0\\
1 & -1 & 0 & 0\\
0 & 0 & 1 & 1\\
0 & 0 & 1 & -1
\end{pmatrix}, \qquad
\mathrm{CNOT} = \begin{pmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&0&1\\
0&0&1&0
\end{pmatrix}
$$

**第二步，先算 $\mathrm{CNOT}\,(I\otimes H)$。** CNOT 的作用是交换 $\lvert 10\rangle \leftrightarrow \lvert 11\rangle$，等价于交换整个矩阵的第三、四行。于是

$$
\mathrm{CNOT}\,(I\otimes H) = \frac{1}{\sqrt{2}}
\begin{pmatrix}
1 & 1 & 0 & 0\\
1 & -1 & 0 & 0\\
0 & 0 & 1 & -1\\
0 & 0 & 1 & 1
\end{pmatrix}
$$

**第三步，再左乘 $I\otimes H$。** 把它与上面矩阵相乘。按块看：前两行与前两列给出 $\frac{1}{2}\begin{pmatrix}2&0\\0&2\end{pmatrix} = I$（右上角块）；后两行后两列给出 $\frac{1}{2}\begin{pmatrix}1&-1\\-1&1\end{pmatrix}\begin{pmatrix}1&-1\\1&1\end{pmatrix} = \begin{pmatrix}0&0\\0&0\end{pmatrix}$…… 逐块相乘后得到

$$
(I \otimes H)\,\mathrm{CNOT}\,(I \otimes H) = \begin{pmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&1&0\\
0&0&0&-1
\end{pmatrix} = \mathrm{CZ}
$$

**验证直觉：** 这条等式的物理含义是「把目标比特的相位翻转换成比特翻转」。因为 $H X H = Z$（上上一篇的共轭引理），受控-X 外面套两层 H（各在目标位上），受控-X 就被共轭成了受控-Z。**共轭引理从单比特门升级成了双比特门的翻译工具**：想实现 CZ 但硬件只支持 CNOT？在目标位两边各加一个 H 即可；反过来，硬件只支持 CZ，也能用同样的办法造出 CNOT。

**辨析｜易错点：** H 要加在**目标位**上，不是控制位。把 $H$ 加在控制位两边得到的是「受控-H 的共轭变体」，那是另一个门。动手前先确认：目标位是 $U$ 作用的那根线，H 必须夹在目标位上。

## 5 辨析：控制比特并非「不受影响」

**辨析｜易错点（最常见的误解）：** 受控门是「条件执行」，很多人因此以为**控制比特始终不变、只有目标比特被操作**。这在计算基下成立，但在叠加态下**不成立**——控制比特会被纠缠「污染」。

看第二节的例子：$\mathrm{CNOT}\left(\lvert +\rangle\otimes\lvert 0\rangle\right) = \lvert \Phi^+\rangle$。此时你若问「控制比特是多少」，答案是**它既不是 0 也不是 1**——系统的两个比特已不可分离，谈论「控制比特单独的状态」在量子力学里根本没有定义（要用密度算符求部分迹，见《密度算符》一篇，此时控制比特的部分态是最大混合态）。**「控制位不变」只在两个比特都没有叠加时才为真。** 这是经典「if」与量子「受控」最根本的差别：经典 if 不改变条件的值，量子受控会把条件本身卷入纠缠。

另一个易错点是**测量破坏纠缠**：对 $\lvert \Phi^+\rangle$ 测量其中一个比特，会立即把另一个比特「定死」在相同的取值上——这是隐形传态的核心机制，也是「受控操作 + 测量 = 远距离关联」的来源。这部分在《量子隐形传态》已经讲过，本篇只是把它归因到受控门的头上。

## 6 从受控-U 到 CNOT + 单比特门

如果 $U$ 就是 X、Z 或 S、T，受控门直接可用。但任意的单比特 $U$ 怎么做成受控-U？答案是：**把它分解成「三个单比特门夹两个 CNOT」**。

N&C §4.3 给出标准构造：任意单比特 $U$ 可以写成 $U = e^{i\alpha} A X B X C$，其中 $ABC = I$（$A, B, C$ 都是单比特门）。于是受控-U 的线路是：目标位上依次放 $C$、过 CNOT、放 $B$、过 CNOT、放 $A$，控制位上补一个相位门 $P(\alpha) = \mathrm{diag}(1, e^{i\alpha})$（因为这个 $\alpha$ 在控制位为 1 时会变成条件相位）。<span class="marginnote">为什么需要两个 CNOT？因为一个 CNOT 只能「在控制位为 1 时把目标位翻转」，要在两条分支上施加不同的单比特变换（$U$ vs $I$），必须借助 CNOT 把分支差异「放大」出来再合并。直觉：$ABC = I$ 保证了控制位为 0 时三段变换恰好抵消为 $I$。</span>

**重点：受控-U 的通用代价是 2 个 CNOT + 若干单比特门。** 这个「1 个受控门 ≈ 2 个 CNOT」的成本公式，是后面估计量子算法门复杂度、以及设计容错线路时的基本换算单位。当 $U$ 退化为 X 时，这个构造自动退化回单个 CNOT，与直觉一致。

## 7 一个可运行的示例（Qiskit）

用 Qiskit 验证 CZ 与 CNOT 的关系，以及 CNOT 制造纠缠：

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

# 1) 验证 CZ = (I⊗H) · CNOT · (I⊗H)
qc2 = QuantumCircuit(2)
qc2.h(1)            # 目标位（第 2 根线）先 H
qc2.cx(0, 1)        # 再 CNOT
qc2.h(1)            # 最后 H（注意 H 夹在目标位上）
print("CZ 与 (I⊗H)CNOT(I⊗H) 是否一致：",
      np.allclose(Operator(qc2).data, np.diag([1, 1, 1, -1])))

# 2) CNOT 制造纠缠：|+0⟩ -> |Φ+⟩
qc3 = QuantumCircuit(2)
qc3.h(0)
qc3.cx(0, 1)
print("|Φ+⟩ =", Statevector(qc3).data)
```

第一段把三个门按 `H · CNOT · H` 顺序排好，验证整体矩阵等于 $\mathrm{diag}(1,1,1,-1)$——这正是 §4 推出来的结论。第二段复现 §2 的纠缠制造：输出态振幅为 $[1,0,0,1]/\sqrt{2}$，即 $\lvert 00\rangle + \lvert 11\rangle$。**CNOT 不再是「两个比特的运算」，而是「两个比特的婚姻」——从此它们只能一起被描述。**

## 8 小结

- **受控-U**：$\lvert c\rangle\lvert t\rangle \mapsto \lvert c\rangle U^c\lvert t\rangle$，矩阵分块 $\mathrm{diag}(I, U)$，恒幺正、可逆。
- **CNOT**：$\lvert c\rangle\lvert t\rangle \mapsto \lvert c\rangle\lvert t \oplus c\rangle$，可逆的 XOR，自己就是自己的逆；控制位为叠加态时制造纠缠（$\lvert +\rangle\lvert 0\rangle \to \lvert \Phi^+\rangle$）。
- **CZ**：$\lvert c\rangle\lvert t\rangle \mapsto (-1)^{c t}\lvert c\rangle\lvert t\rangle$，对称、与 CNOT 通过 $H$ 互换。
- **公式解析**：$\mathrm{CZ} = (I\otimes H)\,\mathrm{CNOT}\,(I\otimes H)$，是单比特共轭引理 $HXH = Z$ 向双比特的推广；H 夹在目标位。
- **辨析**：控制位在叠加态下会被纠缠「污染」，并非「不受影响」；这是经典 if 与量子受控的本质差别。
- **分解代价**：任意受控-U ≈ 2 个 CNOT + 若干单比特门 + 1 个条件相位。

在下一节，我们把这套「受控」哲学推广到三个比特：**Toffoli 门与 Fredkin 门**——受控-受控-NOT 与受控交换，它们让量子线路能够模拟任意经典可逆计算，是通往量子加法器与 Shor 算法中算术电路的桥梁。
