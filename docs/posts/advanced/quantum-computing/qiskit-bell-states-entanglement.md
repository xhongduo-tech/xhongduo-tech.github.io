---
title: 用 Qiskit 构造贝尔态并验证纠缠
date: 2026-08-07
---

# 用 Qiskit 构造贝尔态并验证纠缠

<div class="epigraph">
<p>量子计算的第一口「真味」是亲手造出纠缠。</p>
<footer>—— 尼尔森（Michael Nielsen）与庄（Isaac Chuang）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Qiskit 教程：Entanglement &amp; Bell state ｜ 2026-08-07</p>
</div>

## 为什么从贝尔态的 Qiskit 实现开始

第四篇理论里，贝尔态是纠缠的「标准资源」；现在用代码把它造出来，并**严格验证**它确实纠缠。Qiskit 能做的三件事：构造 $\lvert\Phi^+\rangle$（H + CNOT）、提取态矢量/密度矩阵、以及用量子态层析验证纠缠。<span class="marginnote">「构造」很容易（两行代码）；「验证纠缠」才是重点——如何确认造出来的态「写不成两个单比特态的张量积」？Qiskit 提供两条路：<strong>看密度矩阵的秩 / 约化态的纯度</strong>（理论判据的代码版），以及 <strong>CHSH 不等式的违背</strong>（第四篇的检验）。本节两条都做。</span>学完本节，你就有了一套「量子资源质检」的完整工具箱。

## 1 构造贝尔态

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

qc = QuantumCircuit(2)
qc.h(0)        # 第 0 比特：|0> -> |+>
qc.cx(0, 1)    # CNOT：控制 0、目标 1
print(Statevector(qc))
# 输出: Statevector([0.70710678, 0, 0, 0.70710678], dims=(2, 2))
#       即 (|00> + |11>)/√2 = |Φ+>
```

- `qc.h(0)` 制造单比特叠加；`qc.cx(0, 1)` 用 CNOT 把叠加「扩散」成两比特纠缠。
- 态矢量 `[1/√2, 0, 0, 1/√2]`（按 $00, 01, 10, 11$ 顺序）正是 $\lvert\Phi^+\rangle$。<span class="marginnote">构造其余三个贝尔态：在 CNOT 前对第 0 比特加不同门——$\lvert\Phi^-\rangle$ 用 $X$ 或调整相位、$\lvert\Psi^\pm\rangle$ 换用 $X$ 门作用于目标/控制。练一遍这四个态的线路，你就把第四篇《贝尔态》的生成线路全部内化了。</span>

## 2 验证纠缠：约化态纯度

理论判据（第四篇《纠缠的定义》）：纯态纠缠当且仅当子系统约化态是混合态（$S(\rho_A) > 0$）。代码实现：

```python
from qiskit.quantum_info import DensityMatrix, partial_trace

rho = DensityMatrix(qc)                    # 整体密度矩阵
rho_A = partial_trace(rho, [1])            # 对第 1 比特取部分迹
print(rho_A.purity())                      # 纯度 tr(ρ_A²)，约 0.5（<1 → 混合）
print(rho_A.entropy())                     # von Neumann 熵，约 1（>0 → 纠缠）
```

- `purity() < 1`：$\rho_A$ 是混合态——整体纠缠。
- `entropy() ≈ 1`：纠缠熵 1 ebit——最大纠缠（$\lvert\Phi^+\rangle$）。<span class="marginnote">对照：若线路只做 `qc.h(0)` 不做 CNOT，态是 $\lvert+\rangle\otimes\lvert0\rangle$（可分），`purity()` 返回 1、`entropy()` 返回 0——无纠缠。用代码「复现」理论判据，是最扎实的理解方式。</span>

**辨析｜易错点：** `partial_trace(rho, [1])` 里的 `[1]` 是**要丢掉的**比特索引（把第 1 比特约化掉，留下第 0 比特）。搞反索引会得到另一个子系统——对贝尔态两者熵相等，但对非对称态会得到错误结论。

## 3 验证纠缠：CHSH 不等式违背

更「物理」的验证：测量 CHSH 不等式，看 $S > 2$。在 Qiskit 里，用不同的测量基组合统计 $\langle A_aB_b\rangle$：

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile

sim = AerSimulator()

def chsh_circuit(a_angle, b_angle):
    qc = QuantumCircuit(2, 2)
    qc.h(0); qc.cx(0, 1)                    # 造 |Φ+>
    qc.ry(a_angle, 0); qc.ry(b_angle, 1)    # Alice/Bob 各自的测量基
    qc.measure([0, 1], [0, 1])
    return qc

# 用四组 (a, b) 方向算 S
# E(α,β) = P(同号) - P(异号)（用 counts 估计）
# S = E(0,π/4) + E(0,3π/4) + E(π/2,π/4) - E(π/2,3π/4)
```

- 每个 `(α, β)` 跑多次 shots，统计两比特结果的「同号/异号」比例，得 $E(α,β)$。
- 四组组合代入 CHSH 表达式，算 $S$——对 $\lvert\Phi^+\rangle$ 与合适的角度，$S$ 应逼近 $2\sqrt2 \approx 2.83 > 2$。<span class="marginnote">这段代码就是第四篇《CHSH 不等式》的实验版：理论算 $S = 2\sqrt2$，代码在理想模拟器上也会逼近这个值。把「理论期望」与「代码统计」对上，是理解「量子违背贝尔不等式」最直接的方式。真实硬件上会因噪声而 $S$ 略低于 $2\sqrt2$，但通常仍 > 2。</span>

## 4 公式解析：贝尔测量的 Qiskit 实现

贝尔测量（第四篇《贝尔态与贝尔测量》）的线路是「CNOT + H + 计算基测量」。在 Qiskit 里用「解纠缠 + 读基」实现：

```python
def bell_measurement(qc, a, b):
    qc.cx(a, b)     # 解纠缠（逆 CNOT，CNOT 自逆）
    qc.h(a)         # 逆 H（H 自逆）
    qc.measure([a, b], [0, 1])
```

- **第一步，逆 CNOT**：$CNOT^{-1} = CNOT$，把 $\lvert\Phi^+\rangle$ 拆回 $\lvert00\rangle$。
- **第二步，逆 H**：$H^{-1} = H$，把拆开的态转回计算基。
- **第三步，读基**：读出 $(0,0)$ 即「原态是 $\lvert\Phi^+\rangle$」，四个贝尔态对应四个 $(x,y)$。<span class="marginnote">这个「逆线路解纠缠 + 读基」的模式是量子算法的通用收尾技巧：把「纠缠态里的信息」先解出来再测量。贝尔测量、隐形传态、量子密钥分发里都用到它。</span>

## 5 把贝尔态用于隐形传态

在 Qiskit 里完整跑一遍量子隐形传态（理论见第二篇）——验证「纠缠 + 贝尔测量 + 经典通信」能传未知态：

```python
# Alice 的未知态（比如 |1>）在比特 0，共享纠缠对在比特 1,2
qc = QuantumCircuit(3, 1)
qc.x(0)                     # 要传的态：|1>
qc.h(1); qc.cx(1, 2)        # 制备共享贝尔态 |Φ+>_{1,2}
qc.cx(0, 1); qc.h(0)        # Alice：贝尔测量（CNOT + H）
qc.measure([0, 1], [0, 1])  # Alice 读 2 比特
qc.x(2).c_if(1, 1)          # Bob：若第1位=1 则 X
qc.z(2).c_if(0, 1)          # Bob：若第0位=1 则 Z
```

测量比特 2 应恒为 $\lvert1\rangle$——未知态被传到 Bob。<span class="marginnote">这段代码把「隐形传态协议」完整跑通，包括经典条件门（`c_if`）。它同时演示了第三篇《延迟测量原理》的实用反面：这里必须<strong>中间测量 + 经典控制</strong>，因为 Bob 的操作依赖 Alice 的测量结果——测量不能「推迟到最后」。这是「经典受控操作必须中途测量」的活教材。</span>

## 6 小结

- **构造**：`h(0)` + `cx(0,1)` 两行造出 $\lvert\Phi^+\rangle$。
- **验证（判据版）**：`partial_trace` + `purity()<1` / `entropy()>0` 确认纠缠。
- **验证（物理版）**：不同测量基统计 CHSH $S$，理想下逼近 $2\sqrt2$。
- **贝尔测量**：逆 CNOT + 逆 H + 读基，用于隐形传态等协议。
- **隐形传态**：`c_if` 经典条件门实现 Bob 的修正——经典受控操作必须中途测量。

在下一节，我们把算法搬上 Qiskit——**用 Qiskit 实现 Deutsch-Jozsa 算法**。
