---
title: 用 Qiskit 实现 Deutsch-Jozsa 算法
date: 2026-08-07
---

# 用 Qiskit 实现 Deutsch-Jozsa 算法

<div class="epigraph">
<p>第一次在代码里看到「一次查询判断平衡还是常数」，你会真正理解量子并行。</p>
<footer>—— IBM Quantum 团队（代拟）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Qiskit 教程：Deutsch-Jozsa algorithm ｜ 2026-08-07</p>
</div>

## 为什么从 DJ 的 Qiskit 实现开始

第五篇理论上，Deutsch-Jozsa 是「指数查询加速」的第一课；现在把它写成可运行的 Qiskit 程序。这个过程会逼你把三件事落到实处：**oracle 怎么用代码实现**、**$H^{\otimes n}$ 怎么铺**、**结果怎么判读**。<span class="marginnote">DJ 的 Qiskit 实现是「算法理论 → 工程代码」的迷你样板：理论里的「oracle $O_f$」在代码里是一个子线路（`QuantumCircuit` 的函数），「$H^{\otimes n}$」就是 `qc.h(range(n))`，「全 0 → 常数」就是 `counts` 的键名。把符号翻译成 API，是量子编程的第一道技能。</span>学完本节，你就能独立实现任意「黑盒相位」型算法。

## 1 一个平衡 oracle：例子

先实现一个 2 比特平衡函数 $f(x) = x_1 \oplus x_2$（恰好一半输出 1）：

```python
from qiskit import QuantumCircuit

def balanced_oracle(n):
    qc = QuantumCircuit(n + 1)      # n 个输入比特 + 1 个辅助比特
    # 用 CNOT 实现 f(x) = x1 ⊕ x2：每个 xi 为 1 就翻转辅助比特
    for i in range(n):
        qc.cx(i, n)
    return qc
```

这个 oracle 在叠加态上「同时」翻转辅助比特的相位——正是理论里的翻转查询。<span class="marginnote">对照理论：翻转查询 $\lvert x\rangle\lvert y\rangle \to \lvert x\rangle\lvert y\oplus f(x)\rangle$ 的代码实现就是「一串 CNOT」（每输入比特控制一次辅助比特）。常数 oracle 则是空线路（$f \equiv 0$）或全 $X$ 翻转辅助比特（$f\equiv1$）。</span>

## 2 DJ 算法的完整线路

理论线路：$H^{\otimes n}$ → oracle → $H^{\otimes n}$ → 测量，辅助比特先 $X$ 再 $H$（$\lvert-\rangle$）：

```python
def deutsch_jozsa(n, oracle):
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)                       # 辅助比特 |0> -> |1>
    qc.h(range(n + 1))            # 全部 H：输入比特开叠加、辅助比特 |+> -> 由 X 后成 |->? 见公式解析
    qc.compose(oracle, inplace=True)   # 施加 oracle
    qc.h(range(n))                # 输入比特再 H
    qc.measure(range(n), range(n))     # 测输入比特
    return qc
```

$X$ + $H$：辅助比特变 $\lvert-\rangle$（先 $X$ 再 $H$ 得到 $\frac{1}{\sqrt2}(\lvert0\rangle-\lvert1\rangle)$），oracle 因此等价于「相位查询」。
$H^{\otimes n}$：第一次开叠加，第二次收干涉。
测量结果若全 0 → 常数；否则 → 平衡。<span class="marginnote">注意 `compose` 把 oracle 子线路嵌入主线路——子线路的比特编号必须与主线路对齐（这里 oracle 用 0..n 号比特）。这是 Qiskit 里「模块化线路」的标准姿势：算法主骨架 + 可替换的 oracle 子模块。</span>

## 3 公式解析：辅助比特的 $\lvert-\rangle$ 技巧在代码里的体现

理论关键：翻转查询 + 辅助 $\lvert-\rangle$ = 相位查询。代码里的 $X$ + $H$ 正是制备 $\lvert-\rangle$：

$$
\lvert0\rangle \xrightarrow{X} \lvert1\rangle \xrightarrow{H} \frac{\lvert0\rangle - \lvert1\rangle}{\sqrt2} = \lvert-\rangle
$$

- **第一步，$X$**：$\lvert0\rangle \to \lvert1\rangle$。
- **第二步，$H$**：$\lvert1\rangle \to \lvert-\rangle = \frac{1}{\sqrt2}(\lvert0\rangle-\lvert1\rangle)$。
- **第三步，oracle 作用**：$\lvert x\rangle\lvert-\rangle \to (-1)^{f(x)}\lvert x\rangle\lvert-\rangle$——辅助比特始终是 $\lvert-\rangle$，相位被「复制」到输入寄存器。<span class="marginnote">为什么先 $X$ 再 $H$ 而不是直接 $H$？因为直接 $H$ 给的是 $\lvert+\rangle$，而 $\lvert+\rangle$ 在翻转查询下不会产生负相位（$f$ 翻转 $\lvert0\rangle/\lvert1\rangle$ 在 $\lvert+\rangle$ 上相互抵消）。$\lvert-\rangle$ 是「相位敏感」的辅助态——这是 DJ 线路「多一个 $X$」的原因，也是全部黑盒算法共用的「$X$-$H$ 小技巧」。</span>

## 4 完整运行与判读

```python
from qiskit_aer import AerSimulator

def run_dj(n, oracle):
    """在模拟器上跑 DJ 线路并返回 counts"""
    return AerSimulator().run(deutsch_jozsa(n, oracle), shots=1024).result().get_counts()

n = 3
# 测试常数 oracle（空线路）
const = QuantumCircuit(n + 1)
result_c = run_dj(n, const)
print(result_c)        # 期望 {'000': 1024} —— 全 0 → 常数

# 测试平衡 oracle
bal = balanced_oracle(n)
result_b = run_dj(n, bal)
print(result_b)        # 期望 {'111': 1024} 或类似非全 0 —— 平衡
```

常数 oracle → 测量必为**全 0**（概率 1）。
平衡 oracle → 测量必为**非全 0**（理论保证不是全 0）。<span class="marginnote">这个「全 0 / 非全 0」的二值判读完美体现 DJ 的语义：<strong>一次 oracle 调用 + 一次测量，区分常数与平衡</strong>。经典要 $2^{n-1}+1$ 次查询才能保证——在代码里跑通这个对比，指数加速就从「定理」变成「眼见为实」。</span>

**辨析｜易错点：** DJ 的「一次查询」指的是**一次 oracle 调用**，不是「一条线路」。线路里 `compose` 进去的 oracle 是一个整体模块——它的内部实现（几个 CNOT）是「oracle 的实现成本」，不计入查询复杂度。这是「查询复杂度 ≠ 线路复杂度」的代码版提醒（第五篇《查询复杂度》）。

用一个数字感受加速比。经典确定性算法保证区分常数与平衡，最坏要试 $2^{n-1}+1$ 个输入：

| 输入比特数 $n$ | 经典最坏查询数 $2^{n-1}+1$ | 量子查询数 |
| --- | --- | --- |
| 2 | 3 | 1 |
| 3 | 5 | 1 |
| 10 | 513 | 1 |
| 100 | $2^{99}+1 \approx 6.3\times10^{29}$ | 1 |

上面 3 比特的测试在 Qiskit 里一次 `run` 就完成，而经典要穷举 5 个输入才能下结论。**当 $n=100$ 时，经典那个「5」已经膨胀到宇宙年龄量级的操作数，量子的「1」纹丝不动**——这就是 DJ 想让你记住的「查询复杂度」意义上的指数加速。

## 5 扩展到 Bernstein-Vazirani

DJ 的代码骨架改三行就是 BV 算法（第五篇）：oracle 换成「内积相位」，测量结果直接读出隐藏串。

```python
def bv_oracle(s):                     # s 是隐藏比特串
    qc = QuantumCircuit(len(s) + 1)
    for i, bit in enumerate(s):
        if bit == '1':
            qc.cx(i, len(s))          # 只有 s_i=1 的位才翻转辅助
    return qc
```

测量输出 = 隐藏串 $s$。<span class="marginnote">这个「改 oracle 不改骨架」的练习极有价值：它让你看到 DJ、BV、Simon 共享同一套「$H$ → 相位 oracle → $H$ → 测量」引擎（第五篇的谱系图），只是 oracle 的相位函数不同。理解算法家族共用一个引擎，比逐个背线路重要得多。</span>

## 6 小结

- **oracle 实现**：平衡函数 = 一串 CNOT；常数 = 空线路/全 X——oracle 是「可替换子线路」。
- **$\lvert-\rangle$ 技巧**：$X$ + $H$ 制备相位敏感的辅助态，把翻转查询变相位查询。
- **DJ 骨架**：$H^{\otimes n}$ → oracle → $H^{\otimes n}$ → 测量；全 0 = 常数，非全 0 = 平衡。
- **查询 vs 线路**：oracle 内部实现成本不计入查询复杂度。
- **家族扩展**：改 oracle 就是 BV——DJ/BV/Simon 共享同一引擎。

在下一节，我们用 Qiskit 实现更重的算法——**用 Qiskit 实现 QFT 与相位估计**。
