---
title: 用 Qiskit 实现 QFT 与相位估计
date: 2026-08-07
---

# 用 Qiskit 实现 QFT 与相位估计

<div class="epigraph">
<p>QFT 的代码比它的数学简单——这正是量子编程的乐趣。</p>
<footer>—— Qiskit 社区（代拟）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Qiskit 教程：Quantum Fourier Transform &amp; Phase Estimation ｜ 2026-08-07</p>
</div>

## 为什么从 QFT 的 Qiskit 实现开始

第五篇里 QFT 是「$O(n^2)$ 个受控相位门」的数学构造；本节把它写成可运行的 Qiskit 函数，并用它实现**相位估计**——Shor 算法的核心子程序。写 QFT 的代码会逼你把「受控 $R_k$ 门」和「比特反转」落到实处，而相位估计的代码会把「受控-$U^{2^j}$ + 逆 QFT」变成一个可验证的管线。<span class="marginnote">好消息：Qiskit 自带 `QFT` 与 `PhaseEstimation` 库（`qiskit.circuit.library`），但本节<strong>从零手写</strong>——因为手写一遍才能看懂 QFT 的线路结构。看懂后可以用库函数加速开发。学完本节，你就解锁了 Shor 之前最后一块拼图。</span>

## 1 手写 QFT

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PhaseGate

def qft_circuit(n):
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            angle = 2 * np.pi / 2 ** (k - j + 1)
            qc.cp(angle, k, j)      # 受控相位门（控制 k、目标 j）
    # 比特反转：交换 i 与 n-1-i
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc
```

- 外层循环每比特一个 $H$，内层循环施加受控相位 $R_{k-j+1}$——对应理论里的 $\frac{n(n-1)}{2}$ 个受控门。
- 末尾的 `swap` 做比特反转（QFT 输出是「反序」的）。<span class="marginnote">`qc.cp(angle, control, target)` 是受控相位门 $CR_k$。理论里 $R_k = \text{diag}(1, e^{2\pi i/2^k})$，这里的 `angle = 2π/2^(k-j+1)` 正是它。把理论的门索引对应到代码的循环变量，是「数学 → 代码」翻译的典型练习。</span>

验证：对 $\lvert j\rangle$ 作用 QFT，应该得到 $\frac{1}{\sqrt{2^n}}\sum_k e^{2\pi i jk/2^n}\lvert k\rangle$——用 `Operator` 与理论矩阵比对。

## 2 用 QFT 验证傅里叶对

直接验证「QFT 把计算基变到傅里叶基」：

```python
from qiskit.quantum_info import Statevector

n = 3
qc = QuantumCircuit(n)
qc.x(1)                      # 制备 |010> = |2>
qc.compose(qft_circuit(n), inplace=True)
sv = Statevector(qc)
print(sv)                    # 每个分量振幅 ≈ exp(2πi·2k/8)/√8
```

- 输入 $\lvert2\rangle = \lvert010\rangle$，QFT 后振幅是 $\frac{1}{\sqrt8}e^{2\pi i \cdot 2k/8}$。
- 与理论公式逐项比对——代码验证定理。<span class="marginnote">这个「输入一个计算基、看输出振幅」的验证是 QFT 调试的黄金标准：任何索引错位都会让振幅不匹配。算通一次 $\lvert2\rangle$ 的例子，你对 QFT 的「相位结构」就有了肌肉记忆。</span>

## 3 实现相位估计

相位估计（第五篇）：给定 $U$ 与 $\lvert u\rangle$，用「受控-$U^{2^j}$ + 逆 QFT」读出本征相位。以 $U = T = R_z(\pi/4)$（本征相位 $1/8$）为例：

```python
from qiskit.circuit.library import QFT, PhaseGate

def phase_estimation(U, n_control):
    qc = QuantumCircuit(n_control + 1)
    qc.h(range(n_control))           # 控制寄存器开叠加
    qc.x(n_control)                  # 本征态制备：|1>（T 的本征态）
    # 受控-U^(2^j)
    for j in range(n_control):
        for _ in range(2 ** j):
            qc.append(U.control(), [j, n_control])
    # 逆 QFT
    qc.compose(QFT(n_control, inverse=True), range(n_control), inplace=True)
    qc.measure(range(n_control), range(n_control))
    return qc
```

- 控制寄存器（$n$ 比特）先 $H$ 开叠加；本征态 $\lvert1\rangle$ 放在辅助位。
- 受控-$U^{2^j}$ 用「重复 $2^j$ 次 `U.control()`」实现。
- 逆 QFT 用 Qiskit 库 `QFT(n, inverse=True)`，测量读出相位的二进制近似。<span class="marginnote">`U.control()` 生成受控版本——Qiskit 自动把任意门升级成受控门。`QFT(n, inverse=True)` 一行搞定逆变换。对 $T$ 门，相位 $\theta = 1/8$，3 个控制比特的测量应读出 `001`（二进制 $1/8$）。</span>

## 4 公式解析：读出结果的判读

$T = \text{diag}(1, e^{i\pi/4})$，本征值 $e^{2\pi i \cdot \frac18}$，相位 $\theta = 1/8$。用 $n=3$ 个控制比特，相位估计应给出

$$
\frac{a}{2^3} \approx \frac18 \;\Rightarrow\; a = 1 \;\Rightarrow\; \text{测量结果 } \lvert001\rangle
$$

- **第一步，本征值**：$T\lvert1\rangle = e^{i\pi/4}\lvert1\rangle = e^{2\pi i\cdot\frac18}\lvert1\rangle$，所以 $\theta = 1/8$。
- **第二步，二进制近似**：$\theta = 1/8 = 0.001_2$，$n=3$ 时精确可表示，$a = 1$。
- **第三步，测量**：`counts` 应集中在 `'001'`——读出 $\theta$。<span class="marginnote">把测量结果 `a` 除以 $2^n$ 就得到相位估计 $\tilde\theta = a/2^n$。这个「测量整数 → 相位」的换算就是 Shor 里「读出周期」的机制（第六篇会细讲）。相位估计是 Shor、HHL、量子化学的能量计算共同的心脏——在代码里跑通它，你离 Shor 只差一个「模幂 oracle」。</span>

**辨析｜易错点：** `U.control()` 重复 $2^j$ 次时，$j$ 从 0 到 $n-1$——第 0 个控制比特施加 $U^1$、第 $n-1$ 个施加 $U^{2^{n-1}}$。搞错指数会得到错误的相位（差一个因子 2）。另一个易错点：本征态必须准备正确——用 $\lvert1\rangle$ 测的是 $T$ 的 $\lvert1\rangle$ 本征相位，若准备错态，结果是本征态的混合。

## 5 从相位估计到 Shor

相位估计在 Qiskit 里的骨架，换上「模幂 oracle」（$U\lvert y\rangle = \lvert ay \bmod N\rangle$）就是 Shor 的第二步。Qiskit 甚至提供 `Shor` 库类：

```python
from qiskit.algorithms import Shor

shor = Shor()
result = shor.factor(15)      # 分解 15 = 3 × 5
```

但「用库」不如「看懂骨架」：相位估计 = 控制寄存器 + 受控-$U^{2^j}$ + 逆 QFT + 读相位，这一套骨架在 Shor、HHL、量子化学里原样复用。<span class="marginnote">本节结尾的定位：<strong>相位估计是「量子算法工具箱」的中枢</strong>——QFT 是它的内核，Shor/HHL 是它的应用。在 Qiskit 里手写一遍相位估计，比读十遍理论更能建立「这些算法如何咬合」的整体感。</span>

## 6 小结

- **手写 QFT**：每比特 $H$ + 受控相位 + 末尾 swap 反转——$O(n^2)$ 门。
- **验证**：用 `Statevector` 把计算基的 QFT 输出与理论振幅逐项比对。
- **相位估计**：`h(range)` 开叠加 + 受控-$U^{2^j}$ + 逆 QFT + 测量。
- **判读**：测量 $a$ → 相位 $\tilde\theta = a/2^n$；$T$ 门相位 $1/8$ → 读出 `001`。
- **定位**：相位估计是 Shor、HHL、量子化学能量计算的中枢——下一步就是 Shor。

在下一节，我们实现最著名的搜索算法——**用 Qiskit 实现 Grover 搜索**。
