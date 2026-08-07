---
title: 用 Qiskit 实现 VQE 与 QAOA 实例
date: 2026-08-07
---

# 用 Qiskit 实现 VQE 与 QAOA 实例

<div class="epigraph">
<p>变分算法是「量子出值、经典优化」的合唱——Qiskit 就是这台合唱的指挥。</p>
<footer>—— Qiskit 社区（代拟）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Qiskit 教程：VQE &amp; QAOA ｜ 2026-08-07</p>
</div>

## 为什么从 VQE/QAOA 的 Qiskit 实现开始

第十篇理论上讲清了 VQE 与 QAOA 的框架；本节把它们变成能跑的 Qiskit 程序——一个解「最小分子」$H_2$ 的基态能量，一个解「最小组合优化」MaxCut 的近似最优切分。<span class="marginnote">Qiskit 的 `qiskit.algorithms.minimum_eigensolvers` 提供 `VQE` 与 `QAOA` 现成类；但本节先手写最小骨架（参数化线路 + `Estimator` 算期望 + 优化器循环），再展示库用法。手写版让你看见「VQA 循环」的每个齿轮，库版让你知道生产环境长什么样。</span>学完本节，「变分」从抽象框架变成你亲手调过的程序。

## 1 最小 VQE：$H_2$ 的基态能量

$H_2$ 的哈密顿量映射到 2 比特后的 Pauli 形式（第十篇《VQE》）：

```python
from qiskit.quantum_info import SparsePauliOp

H2_op = SparsePauliOp(
    ['II', 'IZ', 'ZI', 'ZZ', 'XX', 'YY'],
    coeffs=[-1.0524, 0.3979, -0.3979, -0.0112, 0.1809, 0.1809]
)
```

参数化线路（硬件高效拟设：旋转 + 纠缠）：

```python
def h2_ansatz(theta):
    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0); qc.ry(theta[1], 1)
    qc.cx(0, 1)
    qc.ry(theta[2], 0); qc.ry(theta[3], 1)
    return qc
```

- `SparsePauliOp` 是 Qiskit 的稀疏 Pauli 算符——VQE 的目标哈密顿量。
- 拟设 = 两层 `ry` + 一个 `cx`，参数 $\vec\theta$ 四维。<span class="marginnote">注意：`H2_op` 的系数是经典预计算得到的（用 `PySCF` 之类的化学包），VQE 的「量子部分」只负责「在给定哈密顿量下找基态」。把化学与量子优化解耦，是 VQE 工程的标准分层。</span>

## 2 手写 VQA 循环

用 `Estimator`（Qiskit 1.0 的期望值计算器）手写 VQA 的「量子估计 + 经典优化」：

```python
from qiskit_aer.primitives import Estimator
from scipy.optimize import minimize

estimator = Estimator()

def cost(theta):
    pub = (h2_ansatz(theta), H2_op)
    return estimator.run([pub]).result().values[0]

res = minimize(cost, [0.1] * 4, method='COBYLA')
print(res.fun)     # 期望约 -1.857（H2 基态能量，Hartree）
```

- `Estimator` 把「线路 + 算符」转成期望值——VQA 的「量子出值」。
- `scipy.minimize` 做经典优化——「经典优化」。
- 收敛到 $H_2$ 基态能量约 $-1.857$ Hartree。<span class="marginnote">对照第十篇《VQE》：`cost(theta)` 就是 $E(\vec\theta) = \sum_i h_i\langle\hat P_i\rangle$ 的封装——`Estimator` 内部自动做 Pauli 分解 + 期望值求和。`COBYLA` 是无梯度优化器（VQA 常用，因为梯度带噪声），`SPSA` 也常见。这个「Estimator + scipy 优化」的循环就是 VQA 的最小范式。</span>

## 3 QAOA：MaxCut 实例

用 QAOA 解 3 比特 MaxCut（三角形图），成本哈密顿量 $C = \sum_{(i,j)}\frac{1-Z_iZ_j}{2}$：

```python
def maxcut_cost(n, edges):
    terms, coeffs = [], []
    for (i, j) in edges:
        zz = ['I'] * n; zz[i] = 'Z'; zz[j] = 'Z'
        terms.append(''.join(zz)); coeffs.append(-0.5)   # -ZZ/2
    # 常数项 (|E|/2) 在优化中忽略，加上无妨
    return SparsePauliOp(terms, coeffs)

edges = [(0, 1), (1, 2), (0, 2)]
cost_op = maxcut_cost(3, edges)
```

QAOA 线路（$p=1$：问题层 + 混合层）：

```python
from qiskit.circuit.library import RXGate

def qaoa_circuit(theta, n, edges):
    gamma, beta = theta[0], theta[1]
    qc = QuantumCircuit(n)
    qc.h(range(n))                     # 均匀叠加
    for (i, j) in edges:
        qc.cp(2 * gamma, i, j)         # 问题层 e^{-iγC}：受控 ZZ 相位
    qc.rx(2 * beta, range(n))          # 混合层 e^{-iβB}
    return qc
```

- 问题层用 `cp(2γ, i, j)` 实现 $e^{-i\gamma Z_iZ_j}$（ZZ 项 → 受控相位）。
- 混合层用 `rx(2β)` 实现 $e^{-i\beta X}$（每比特绕 $X$ 转）。<span class="marginnote">对照第十篇《QAOA》：$e^{-i\gamma Z_iZ_j}$ 的实现是「CNOT + $R_z$ + CNOT」或等效的受控相位；Qiskit 里 `cp(2γ)` 直接给出。优化目标是把「期望成本」$\langle C\rangle$ 最大化——经典优化器调 $(\gamma,\beta)$。这个最小 QAOA 能解 $K_3$ 的 MaxCut（最优切 2 条边）。</span>

## 4 公式解析：QAOA 期望成本与最优解

对 $K_3$（三角形），MaxCut 最优解切 2 条边（3 比特里两 0 一 1 或反之）。$p=1$ QAOA 的期望成本

$$
\langle C\rangle = \sum_{(i,j)\in E} \frac{1 - \langle Z_iZ_j\rangle}{2}
$$

- **第一步，逐边贡献**：每条边贡献 $\frac{1-\langle Z_iZ_j\rangle}{2}$，被切（两端不同）时 $\langle Z_iZ_j\rangle = -1$、贡献 1。
- **第二步，对称性**：$K_3$ 上 $p=1$ QAOA 的期望与 $(\gamma,\beta)$ 的关系对称，最优参数给出期望接近最优切分数（约 1.5–1.9，随参数质量）。
- **第三步，采样判读**：测量线路，取 counts 里「成本最高」的比特串作为解。<span class="marginnote">对 $K_3$，所有「两 0 一 1」的串（001、010、100）都是最优解。QAOA 的测量分布应集中在这些串附近。跑 `shots=1000`，数一下「最优串占比」，你就看到了 QAOA 的近似行为——不是「必中」，而是「大概率接近最优」。</span>

**辨析｜易错点：** QAOA 的 `cp(2γ, i, j)` 只实现 $ZZ$ 的**对角相位**，不改变计算基振幅——问题层的本质是「给每个计算基态按成本加相位」。混合层才负责「振幅扩散」。这个「相位编码 + 扩散」的分工（第七篇 Grover 同款）是理解 QAOA 的关键。

## 5 用 Qiskit 库：一行 VQE 与 QAOA

生产环境用库：

```python
from qiskit.algorithms.minimum_eigensolvers import VQE, QAOA
from qiskit_algorithms.optimizers import COBYLA

vqe = VQE(estimator, h2_ansatz, COBYLA())
result = vqe.compute_minimum_eigenvalue(H2_op)
print(result.eigenvalue)          # ≈ -1.857

qaoa = QAOA(estimator, COBYLA(), reps=1)
res_q = qaoa.compute_minimum_eigenvalue(cost_op)
```

- `VQE(estimator, ansatz, optimizer)`：三件套直接组装。
- `QAOA(estimator, optimizer, reps=1)`：问题哈密顿量喂进去，自动生成线路。<span class="marginnote">库版封装了「拟设 → 估计 → 优化」的全部循环，你只提供「哈密顿量 + 拟设 + 优化器」。但「看懂库在做什么」仍需要手写版的经验——这也是本节的顺序：先手写骨架，再换库。VQE/QAOA 是 NISQ 应用的两大支柱，学会「手写 + 库」双轨，就解锁了整个变分应用层。</span>

## 6 小结

- **VQE 最小实例**：`SparsePauliOp`（$H_2$ 哈密顿量）+ `ry`/`cx` 拟设 + `Estimator` + `scipy` 优化 → 基态能量约 $-1.857$。
- **QAOA 最小实例**：MaxCut 成本算符 + `cp`/`rx` 线路 + 优化 $(\gamma,\beta)$ → 近似最优切分。
- **VQA 循环范式**：`Estimator` 算期望（量子）+ `minimize` 优化（经典）。
- **分工**：问题层加相位、混合层扩振幅——Grover 同款的「相位编码 + 扩散」。
- **库用法**：`VQE`/`QAOA` 类一行组装；手写版是理解库的前提。

在下一节，我们把程序从模拟器搬到云端真机——**模拟器与真实后端：IBM Quantum 云端实验**。
