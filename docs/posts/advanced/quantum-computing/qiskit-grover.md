---
title: 用 Qiskit 实现 Grover 搜索
date: 2026-08-07
---

# 用 Qiskit 实现 Grover 搜索

<div class="epigraph">
<p>把「翻转 oracle」和「扩散算子」写成代码，Grover 的 $\sqrt N$ 就真的跑起来了。</p>
<footer>—— Qiskit 社区（代拟）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Qiskit 教程：Grover's algorithm ｜ 2026-08-07</p>
</div>

## 为什么从 Grover 的 Qiskit 实现开始

第七篇里 Grover 是两个反射（oracle + 扩散）的复合；本节把这两个反射写成代码，在 3–4 个比特上「眼见为实」地看到 $\sqrt N$ 次迭代把目标概率抬到接近 1。<span class="marginnote">Grover 的 Qiskit 实现是「算法骨架 + oracle 模块」的又一个样板：主线路固定（$H^{\otimes n}$ → 迭代 $G^k$ → 测量），oracle 随问题换。Qiskit 还提供 Grover 库类，但手写一遍才能理解「oracle 为什么是负号、扩散为什么是均值翻转」。</span>学完本节，你就掌握了「搜索类算法」的全部工程套路。

## 1 翻转 oracle：标记目标

以 $n=3$ 比特、目标 $\lvert101\rangle$ 为例。oracle 要「给目标加负号」：

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def oracle(target="101"):
    n = len(target)
    qc = QuantumCircuit(n)
    for i, b in enumerate(target):
        if b == "0":
            qc.x(i)                              # 对齐：目标里的 0 翻成 1
    qc.mcp(np.pi, list(range(n - 1)), n - 1)     # 全 1 时翻转相位（多控-Z）
    for i, b in enumerate(target):
        if b == "0":
            qc.x(i)                              # 还原对齐
    return qc

qc = QuantumCircuit(3)
qc.h(range(3))                                   # 均匀叠加
qc.compose(oracle(), inplace=True)               # 作用 oracle
print(Statevector(qc).data.round(4))             # |101⟩ 分量带负号
```

理论上的 $O = I - 2\lvert x^*\rangle\langle x^*\rvert$（目标反射）在代码里用「多控门 + 相位技巧」实现。
对三比特，多控-$Z$（$\lvert101\rangle$）实现「全 1 时翻转相位」——把非目标位用 $X$ 对齐到 1、翻转后再还原，就得到任意目标的 oracle。<span class="marginnote">标准做法：把目标模式「对齐」到 $\lvert111\rangle$ 再翻转。对目标 $\lvert101\rangle$：先对第 1 位作用 $X$（把 0 变 1），再 CCZ（全 1 翻转相位），再 $X$ 还原。这个「对齐-翻转-还原」是构建任意标记 oracle 的通用模板。</span>

## 2 扩散算子

扩散算子 $D = 2\lvert s\rangle\langle s\rvert - I = H^{\otimes n}(2\lvert0\rangle\langle0\rvert - I)H^{\otimes n}$：

```python
def diffusion(n):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.x(range(n))                               # |0⟩ ↔ |1⟩ 对齐
    qc.mcp(np.pi, list(range(n - 1)), n - 1)     # 全 1 翻转 → 关于 |0⟩ 反射
    qc.x(range(n))
    qc.h(range(n))
    return qc
```

先 $H$ 再「绕 $\lvert0\rangle$ 翻转」再 $H$——理论里的「先转基、翻转 $\lvert0\rangle$、转回」。
多控-$Z$ 实现「全 1 翻转」，配合两侧 $X$/$H$ 完成「关于均值反射」。<span class="marginnote">对照理论（第七篇《Grover 迭代》）：扩散 = 「关于均匀叠加 $\lvert s\rangle$ 反射」。代码里 $H$ 把 $\lvert s\rangle$ 映射到 $\lvert0\rangle$，翻转 $\lvert0\rangle$ 的相位再转回——「$H$-$X$-多控-$Z$-$X$-$H$」是它的标准实现。多控-$Z$ 就是「多控制相位门」，是扩散算子的核心积木。</span>

## 3 Grover 主线路

```python
from qiskit_aer import AerSimulator

n = 3
qc = QuantumCircuit(n, n)
qc.h(range(n))                                   # 均匀叠加 |s⟩
for _ in range(2):                               # k = ⌊π/4·√8⌋ = 2 次迭代
    qc.compose(oracle(), inplace=True)           # 翻转 oracle
    qc.compose(diffusion(n), inplace=True)       # 扩散算子
qc.measure(range(n), range(n))
print(AerSimulator().run(qc, shots=1024).result().get_counts())   # 大概率 '101'
```

迭代次数按 $k \approx \frac{\pi}{4}\sqrt{N}$ 取整——理论的最优值。
对 $N=8$，$k=2$，两次迭代后 $\lvert101\rangle$ 概率应很高。<span class="marginnote">「迭代次数」是 Grover 的命门（第七篇《几何解释》）：转少了概率不够、转多了过头。$k = \lfloor\frac{\pi}{4}\sqrt{N}\rfloor$ 是单解时的最优取整。若解个数未知，就需要量子计数（第七篇《多次解》）先估——工程上那才是标准流程。</span>

## 4 公式解析：为什么概率能到接近 1

理论（第七篇）：$k$ 次迭代后目标分量 $\sin\big((2k+1)\theta\big)$，$\sin\theta = 1/\sqrt N$。对 $N=8$：

$$
\theta = \arcsin\frac{1}{\sqrt8} \approx 0.361, \qquad (2k+1)\theta \big|_{k=2} = 5\theta \approx 1.807 \approx \frac{\pi}{2}
$$

- **第一步，角度**：$\theta \approx 0.361$（弧度）。
- **第二步，凑 $\pi/2$**：$k=2$ 时 $5\theta \approx 1.807$，与 $\frac\pi2 \approx 1.571$ 接近但略过——所以不是 100%。
- **第三步，概率**：$\sin^2(5\theta) \approx \sin^2(1.807) \approx 0.95$——约 95% 概率测到目标。<span class="marginnote">对 $N=8$，$k=2$ 给约 95%；$k=1$ 只有约 78%；$k=3$ 降到约 55%（过头了）。用代码把 $k=0,1,2,3$ 都跑一遍、画概率曲线，你会亲眼看到「转过了头」的回落——这是 Grover 几何最直观的代码验证。</span>

**辨析｜易错点：** oracle 内部的辅助/相位逻辑若写错，可能「标记」了错误的目标或完全失效。调试方法：用 Statevector 看 oracle 作用后目标分量是否真的带负号。另一个易错点：扩散算子的多控-Z 需要正确对齐控制/目标比特，索引错一位整个迭代就错了。

## 5 通用化：任意标记 oracle 与 Grover 库

**任意目标**：把「对齐-翻转-还原」模板参数化（输入目标二进制串，自动生成 $X$ 序列 + CCZ）。
**Qiskit 库**：Grover 提供现成的搜索器（自动决定迭代次数、处理多次解）。<span class="marginnote">手写 vs 库的取舍：<strong>手写一遍建立理解，库函数提升效率</strong>。Qiskit 的 Grover 甚至能自动用量子计数估计解个数、自适应迭代次数——工程上直接用它。但面试/教学/理解层面，手写骨架无可替代。</span>

## 6 小结

- **oracle**：「对齐-翻转-还原」模板给任意目标加负号（X + 多控-Z/CCZ + 相位技巧）。
- **扩散**：$H$-$X$-多控-$Z$-$X$-$H$ 实现「关于均匀叠加反射」。
- **主线路**：$H^{\otimes n}$ → 迭代 $G^k$（$k = \lfloor\frac\pi4\sqrt N\rfloor$）→ 测量。
- **概率**：$N=8$ 时 $k=2$ 约 95%；过度迭代会回落——迭代次数要精确。
- **工程**：手写骨架理解原理，Grover 库提升效率（自动计数 + 迭代）。

在下一节，我们实现变分家族——**用 Qiskit 实现 VQE 与 QAOA 实例**。
