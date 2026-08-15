---
title: 在 Qiskit 中实现布洛赫球上的单比特门
date: 2026-08-07
---

# 在 Qiskit 中实现布洛赫球上的单比特门

<div class="epigraph">
<p>布洛赫球让抽象的量子门变成看得见的旋转。</p>
<footer>—— 费曼（Richard Feynman）的直觉（代拟）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Qiskit 教程：Single-qubit gates &amp; Bloch sphere ｜ 2026-08-07</p>
</div>

## 为什么从布洛赫球的 Qiskit 实现开始

理论篇（第二篇）里我们认识了布洛赫球——单比特态的全部几何。Qiskit 能把它**画出来**：每个单比特门对应球面上的一次旋转，你可以亲眼看到 $H$ 把 $\lvert0\rangle$ 从北极转到赤道、$X$ 把 $\lvert0\rangle$ 翻到南极。本节把「门 = 旋转」这条直觉用代码钉死——这是理解一切单比特操作、以及后面受控门的前提。

本节是第十二篇《量子编程实践（Qiskit）》的第二课，也是第二篇《布洛赫球》、第三篇《单比特门》的代码落地。建议对照理论篇读：理论篇给「门 = 旋转」的代数，本节给「旋转 = 看得见的球面运动」的几何。读懂这一课，后面受控门、贝尔态、量子算法的线路实现都顺理成章。<span class="marginnote">Qiskit 里画布洛赫球有两种方式：`plot_bloch_multivector(state)`（给出态的布洛赫矢量）与 `plot_bloch_sphere`（画球）。`Statevector` 能提取线路的终态——模拟器内部「知道」态是什么（真机上只能靠层析，后面会讲）。</span>学完本节，你就能「看见」每一个单比特门。

## 1 态矢量的提取与布洛赫球绘制

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector

qc = QuantumCircuit(1)
qc.h(0)

state = Statevector(qc)          # 提取 |+> = (|0>+|1>)/√2
plot_bloch_multivector(state)    # 画出布洛赫球：北极的 |0> 被 H 转到赤道 X+ 方向
```

`Statevector(qc)` 从空线路算起，给出 $\lvert+\rangle = \frac{1}{\sqrt2}(\lvert0\rangle+\lvert1\rangle)$ 的态矢量。
`plot_bloch_multivector` 把态画成球面上一个点——$\lvert+\rangle$ 落在 $+X$ 轴。<span class="marginnote">态矢量（Statevector）是「上帝视角」：模拟器直接给出完整复振幅（真机上测不到）。它用于教学与验证非常方便——比如检查「这个门序列到底把态转到哪了」。真实硬件上验证态需要「量子态层析」（多次测量 + 重建），开销大得多。</span>

把「门 = 旋转」的对照表钉进记忆：

| 门 | 旋转轴 | 角度 | 布洛赫球效果 |
| --- | --- | --- | --- |
| $X$ | $X$ 轴 | $\pi$ | $\lvert0\rangle \leftrightarrow \lvert1\rangle$ |
| $Y$ | $Y$ 轴 | $\pi$ | $\lvert0\rangle \to i\lvert1\rangle$ |
| $Z$ | $Z$ 轴 | $\pi$ | 本征态不动 |
| $H$ | $(X+Z)/\sqrt2$ 轴 | $\pi$ | $Z$ 轴 ↔ $X$ 轴 |
| $S$ | $Z$ 轴 | $\pi/2$ | 四分之一圈 |
| $T$ | $Z$ 轴 | $\pi/4$ | 八分之一圈 |
| $R_y(\theta)$ | $Y$ 轴 | $\theta$ | 连续旋转 |

## 2 Pauli 门：X、Y、Z 的几何

```python
qc_x = QuantumCircuit(1); qc_x.x(0)   # X: |0> -> |1>（绕 X 轴转 π）
qc_y = QuantumCircuit(1); qc_y.y(0)   # Y: |0> -> i|1>（绕 Y 轴转 π，带相位）
qc_z = QuantumCircuit(1); qc_z.z(0)   # Z: |0> -> |0>（北极不动）
```

**$X$**：绕 $X$ 轴转 $\pi$，北极 $\lvert0\rangle$ → 南极 $\lvert1\rangle$（比特翻转）。
**$Y$**：绕 $Y$ 轴转 $\pi$，$\lvert0\rangle \to i\lvert1\rangle$（翻转 + 相位 $i$）。
**$Z$**：绕 $Z$ 轴转 $\pi$，$\lvert0\rangle$ 与 $\lvert1\rangle$ 都在 $Z$ 轴上，只是 $\lvert1\rangle$ 乘 $-1$——球面上北极南极都不动（整体相位不可见）。<span class="marginnote">为什么 $Z$ 画出来「没变化」？因为 $\lvert0\rangle$、$\lvert1\rangle$ 恰是 $Z$ 的本征态，$Z$ 只乘相位 $(-1)^{\text{本征}}$——整体相位在布洛赫球上不可见。要「看见」$Z$ 的效果，得作用在叠加态上（把 $+X$ 转到 $-X$）。</span>

**辨析｜易错点：** $X$ 不是「$0$ 变 $1$ 那么简单」——它翻转的是**整个布洛赫球**。对叠加态 $\lvert+\rangle$，$X\lvert+\rangle = \lvert+\rangle$（$+X$ 轴上的点绕 $X$ 轴转 $\pi$ 回到自身）。「比特翻转」只在计算基下成立；「绕轴旋转」才对所有态成立。

## 3 H、S、T 门与旋转门

```python
qc_h = QuantumCircuit(1); qc_h.h(0)   # H: 绕 (X+Z)/√2 轴转 π，北极→+X 赤道
qc_s = QuantumCircuit(1); qc_s.s(0)   # S: 绕 Z 轴转 π/2（90°）
qc_t = QuantumCircuit(1); qc_t.t(0)   # T: 绕 Z 轴转 π/4（45°）

# 任意旋转门（第三篇《旋转门》）：
qc_rx = QuantumCircuit(1); qc_rx.rx(1.0, 0)   # 绕 X 轴转 1 弧度
qc_ry = QuantumCircuit(1); qc_ry.ry(0.5, 0)   # 绕 Y 轴转 0.5 弧度
qc_rz = QuantumCircuit(1); qc_rz.rz(2.0, 0)   # 绕 Z 轴转 2 弧度
```

**$H$**：把 $Z$ 轴转成 $X$ 轴（$\lvert0\rangle \to \lvert+\rangle$、$\lvert1\rangle \to \lvert-\rangle$）。
**$S = R_z(\pi/2)$、$T = R_z(\pi/4)$**：绕 $Z$ 轴的四分之一、八分之一圈。
**$R_x, R_y, R_z(\theta)$**：连续旋转门，参数是弧度——变分算法的「旋钮」。<span class="marginnote">把这些门依次作用、每次 `plot_bloch_multivector` 画一下，你会看到态在球面上「爬行」。这是建立「门 = 旋转」直觉的最好练习：试着从 $\lvert0\rangle$ 出发，用 $R_y$、$R_z$ 的组合到达球面任意一点——这正是「任意单比特门分解」（第三篇）的几何验证。

把组合算一个具体例子：$\lvert0\rangle$ 先 $R_y(\pi/2)$（转到 $+X$ 轴）再 $R_z(\pi/2)$（绕 $Z$ 轴转 90°，转到 $+Y$ 轴），得到的态是 $\lvert+\rangle$ 再被 $S$ 作用 = $\frac{1}{\sqrt2}(\lvert0\rangle + i\lvert1\rangle)$。你可以用 `Statevector` 验证：$(0.707, 0.707i)$——这正是「先定纬度、再转经度」的球面导航。</span>

## 4 公式解析：验证「门 = 旋转」

用代码验证一个恒等式：$H X H = Z$（$H$ 把 $X$ 共轭成 $Z$）。

```python
qc = QuantumCircuit(1)
qc.h(0); qc.x(0); qc.h(0)     # H X H
unitary = Operator(qc)        # 提取 2×2 矩阵
# 期望 ≈ Z = [[1,0],[0,-1]]
```

**第一步，算子提取**：`Operator(qc)` 给出线路的酉矩阵。
**第二步，数值验证**：`H X H` 算出来应接近 $Z$——用 `np.allclose(unitary, Z_matrix)` 检查。
**第三步，几何解释**：$H$ 是「坐标系旋转」（$Z$ 轴 ↔ $X$ 轴），$H X H$ 是「先换坐标系、转 $X$、再换回来」= 在原坐标系里转 $Z$。<span class="marginnote">这个「共轭换轴」技巧在量子纠错里已经见过（第八篇「$H$ 把相位错误变比特错误」）：$HZH = X$ 那条恒等式的代码版。验证恒等式是「理论直觉 ↔ 代码事实」对账的最佳练习。</span>

## 5 从门序列到合成门

Qiskit 能把任意单比特门「编译」成硬件原生门集（如 $R_z + \sqrt{X}$）：

```python
from qiskit.circuit.library import UGate
qc = QuantumCircuit(1)
qc.append(UGate(theta, phi, lam), [0])   # 任意单比特门（Z-Y-Z 分解）
qc_t = transpile(qc, basis_gates=['rz', 'sx', 'x'])
```

`UGate(θ, φ, λ)` 是「任意单比特酉」的标准参数化（对应 $R_z(\phi)R_y(\theta)R_z(\lambda)$），`transpile` 会把它拆成目标平台的原子门。<span class="marginnote">这呼应 Solovay-Kitaev（第三篇）：任何单比特门都能用有限原子门逼近。Qiskit 的 transpiler 自动做这件事——你写「数学上的门」，它给「硬件上的门」。理解这条「逻辑门 → 物理门」的降级链，是理解整个量子编译体系（第十二篇最后一节）的起点。</span>

## 6 小结

- `Statevector` + `plot_bloch_multivector` 让门变成看得见的球面旋转。
- **$X$ 绕 $X$ 轴转 $\pi$、$H$ 转坐标系、$S/T$ 是 $Z$ 轴四分之一/八分之一圈、$R_{x/y/z}$ 是连续旋转**。
- `Operator(qc)` 提取酉矩阵，用代码验证恒等式（如 $HXH=Z$）。
- `UGate(θ,φ,λ)` + `transpile` 实现「任意门 → 原生门」编译。

在下一节，我们进入两比特世界——**用 Qiskit 构造贝尔态并验证纠缠**。
