---
title: Toffoli 门与 Fredkin 门
date: 2026-08-07
---

# Toffoli 门与 Fredkin 门

<div class="epigraph">
<p>信息是物理的。</p>
<footer>—— 罗夫 · 兰道尔（Rolf Landauer）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen & Chuang《量子计算》§4.3、§1.4 通用门集 ｜ 2026-08-07</p>
</div>

## 为什么从 Toffoli 门开始

前三篇讲完单比特门与受控门，门库还差最重要的一块拼图：**三个比特的门**。**Toffoli 门（受控-受控-NOT，CCNOT）**与 **Fredkin 门（受控交换，CSWAP）**，是量子线路从「物理上可逆的比特操作」走向「能跑经典算法」的桥梁。

为什么要专门为三比特门写一篇？因为 Toffoli 门有一个单比特门和 CNOT 都没有的身份：**它可以模拟经典计算里的一切**。用 Toffoli 加辅助比特，你能搭出与门、或门、加法器、乃至整个经典 CPU 的可逆版本——而且每一步都不丢信息。兰道尔那句「信息是物理的」正是这个故事的注脚：经典计算丢弃信息会耗散能量，可逆计算（量子计算是其中最彻底的一种）可以从原理上绕开这个下限。本篇之后，你就能回答「量子线路能不能做经典算术？」——答案是能，而且就是用 Toffoli 做的，Shor 算法里的模幂运算、Grover 里的 oracle 实现，全都建立在它上面。

## 1 Toffoli 门：受控-受控-NOT

**核心概念：** **Toffoli 门（CCNOT）**作用在三个比特上，前两个是**控制位**，第三个是**目标位**：当且仅当前两个控制位都为 $\lvert 1\rangle$ 时，翻转目标位。记作

$$
\lvert a\rangle\lvert b\rangle\lvert c\rangle \;\longmapsto\; \lvert a\rangle\lvert b\rangle\lvert c \oplus (a \cdot b)\rangle
$$

其中 $a\cdot b$ 是经典与（AND），$\oplus$ 是模 2 加法。展开来看：$\lvert 000\rangle, \lvert 001\rangle, \lvert 010\rangle, \lvert 011\rangle, \lvert 100\rangle, \lvert 101\rangle$ 都不动；只有 $\lvert 110\rangle \leftrightarrow \lvert 111\rangle$ 互换。<span class="marginnote">注意这里写成 $c \oplus (a\cdot b)$ 是一个<strong>可逆的与</strong>：两个控制位原样保留，目标位带上「两控制位都是 1」的信息。这个「保留输入 + 追加结果」的模式，正是把经典运算嵌入可逆计算的通用手法。</span>

**重点：Toffoli 是对称的。** 三个比特中任意两个都可以当控制位、剩下的当目标位，行为完全一样（因为 $a\cdot b = b\cdot a$）。在布洛赫/矩阵层面，Toffoli 就是「以 $a, b$ 为控制位的受控-X」——把上一篇的受控-U 再套一层控制。

## 2 Fredkin 门：受控交换

**核心概念：** **Fredkin 门（CSWAP）**是受控-交换门：控制位为 $\lvert 1\rangle$ 时，交换另外两个比特的内容；控制位为 $\lvert 0\rangle$ 时，什么都不做：

$$
\lvert c\rangle\lvert a\rangle\lvert b\rangle \;\longmapsto\;
\begin{cases}
\lvert c\rangle\lvert a\rangle\lvert b\rangle & c = 0\\
\lvert c\rangle\lvert b\rangle\lvert a\rangle & c = 1
\end{cases}
$$

Fredkin 门有一个特别的性质：**它守恒「1 的个数」**——无论控制位是 0 还是 1，三个比特中 1 的总数都不变（交换两个比特不改变总数）。因此 Fredkin 门被称为**守恒门（conservative gate）**，它对应物理上「粒子数守恒」的可逆运算。Toffoli 门则不守恒（$\lvert 111\rangle$ 翻转后 1 的个数从 3 变成 2）。

**重点：Toffoli 与 Fredkin 都是经典可逆计算的万能门**——只用 Toffoli（配辅助比特）可以模拟任意经典电路，只用 Fredkin 也可以。Fredkin 的「守恒性」使它天然适合模拟那些尊重某种守恒律的物理过程（比如粒子碰撞模型），这是 1980 年代可逆计算理论里的一支重要脉络。

## 3 公式解析：Toffoli 的矩阵与「受控-受控」结构

这一节把 Toffoli 拆到底。在八维计算基 $\{\lvert 000\rangle, \lvert 001\rangle, \ldots, \lvert 111\rangle\}$ 下，Toffoli 是 $8\times 8$ 矩阵，其中只有 $\lvert 110\rangle \leftrightarrow \lvert 111\rangle$ 被互换，其余对角元都是 1：

$$
\mathrm{Toffoli} = \begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
\end{pmatrix}
$$

**第一步，读出结构。** 把前两个比特视为「地址」，第三个比特为「数据」：Toffoli 只在前两位都是 1 的「最后两个地址」上做了一次 X。这正是「受控-受控-X」的字面意思——两层控制叠加。

**第二步，验证作用。** 对任意输入 $\lvert a b c\rangle$，输出 $\lvert a b,\, c\oplus (a\land b)\rangle$。取 $\lvert 110\rangle$：$a\land b = 1$，故 $c$ 从 0 翻到 1，得 $\lvert 111\rangle$；取 $\lvert 111\rangle$ 则回到 $\lvert 110\rangle$。其余六个基矢不动。与矩阵逐列对照，完全一致。

**第三步，为什么它可逆。** 观察：$c \oplus (a\land b)$ 再异或一次 $(a\land b)$ 就回到 $c$——Toffoli 作用两次等于恒等：$\mathrm{Toffoli}^2 = I$。所以 $\mathrm{Toffoli}^{-1} = \mathrm{Toffoli}$，自逆、幺正。**可逆性来自「异或一次再异或一次即还原」，这是所有 $\oplus$-型门（CNOT、Toffoli）共有的特征。**

**第四步，从 CNOT 造 Toffoli。** 代数上 Toffoli 不是 CNOT 的直接张量积，但它可以用 6 个 CNOT + 若干单比特门精确搭出（N&C §4.3 给出标准线路）。直觉是：两层控制无法一次完成，需要一个「临时位置」把两个控制位的与搬出来再搬回去。这预告了更一般的结论——**任何受控-受控-U 都能由 CNOT 与单比特门实现**，Toffoli 只是其中 $U = X$ 的特例。<span class="marginnote">工程上还有一个重要细节：物理硬件往往只有 CNOT，Toffoli 需要被<strong>编译</strong>成 CNOT + 单比特门序列。IBM 超导芯片上实现一次 Toffoli，大约要 6 个 CNOT；而若用 H、S、T 全集，则需 7 个 T 门（所谓「T-count = 7」）——这正是上一篇提到 T-count 重要的原因。</span>

## 4 Toffoli 是经典可逆计算的万能门

现在回答本篇的核心问题：**为什么 Toffoli 能模拟一切经典计算？**

经典计算的万能门是 NAND（与非）：任何布尔函数都能用 NAND 搭出来。Toffoli 可以构造 NAND：

固定第三个比特为 $\lvert 1\rangle$，则 Toffoli 输出 $\lvert a\rangle\lvert b\rangle\lvert 1 \oplus (a\land b)\rangle$。而 $1 \oplus (a\land b) = \neg(a\land b)$，正是 NAND。

所以：**输入 $a, b$（前两个控制位），辅助位放 $\lvert 1\rangle$，目标位读出的就是 $a$ 与 $b$ 的 NAND。** 再辅以复制线路（用 CNOT 把比特复制到辅助位），任何经典电路都能改写为可逆的 Toffoli 线路——这就是可逆计算的存在性证明。<span class="marginnote">为什么经典 NAND 不可逆而 Toffoli 版的 NAND 可逆？因为 Toffoli <strong>保留了输入副本</strong>（前两个控制位没丢），只是「额外」把结果写进目标位。可逆计算的全部窍门就是：别删信息，把结果追加在边上。</span>

**重点：量子线路是经典计算的严格超集。** 给一台量子计算机装上「只允许 Toffoli 和辅助比特」的编译器，它就能模拟任何经典算法；再加上 CNOT 与单比特门提供的叠加与干涉，它还能做经典算不了的事。这个「经典可嵌入量子」的结论（本内特在 1970 年代建立）让量子计算的安全性不依赖「比经典更强」，而依赖「包含经典且更强」——RSA 等经典密码在量子威胁下失守，正是因为量子计算机能完整地运行这些算法、再附赠 Shor 的加速。

**辨析｜易错点：** Toffoli 的「经典万能」与「量子通用」是两回事。Toffoli 单独使用（配辅助位）只能模拟经典可逆电路——输入输出都是计算基态，从不制造叠加。要让 Toffoli 参与真正的量子算法（叠加、干涉），还得配 H 门等制造叠加的门。判断标准：**如果一个门库只含 Toffoli 与经典初态，它跑出的永远是经典电路**；量子加速必须靠 H 等非经典门引入叠加。

## 5 辨析：三比特门与纠缠、与门控

本节集中澄清几个 Toffoli/Fredkin 相关的经典误区。

**辨析｜易错点一（「三比特 = 更强」）：** Toffoli 是双控门，但它**没有引入 CNOT 之外的新量子能力**——任何 Toffoli 都能用 CNOT + 单比特门实现。它的价值不在「更强的量子纠缠」，而在「更丰富的经典可逆逻辑」：用它可以写算术、写条件、写函数求值。量子算法的复杂度分析里，Toffoli 常被当作「理想化的成本单位」，因为实现它的 CNOT 数是一个已知常数（约 6）。

**辨析｜易错点二（控制位不受影响）：** 和 CNOT 一样，Toffoli 的两个控制位在**计算基**下不变，但在叠加态下也会被卷入纠缠。把控制位置于叠加态再施加 Toffoli，输出的三个比特通常不可分离——「控制位没被碰」只在经典输入时成立。

**辨析｜易错点三（Fredkin 与 Toffoli 的关系）：** Fredkin 可以看作「受控的交换」，而交换（SWAP）本身可由三个 CNOT 组成（$\mathrm{SWAP} = \mathrm{CNOT}_{12}\mathrm{CNOT}_{21}\mathrm{CNOT}_{12}$）。所以 Fredkin 也能由 CNOT 与单比特门实现，并非「不可约的新门」。它们共同的意义是给可逆计算提供了两套**守恒律不同的**万能构造（Toffoli 不守恒、Fredkin 守恒），对应物理中「信息守恒」与「粒子数守恒」两种直觉。

## 6 一个可运行的示例（Qiskit）

用 Qiskit 验证 Toffoli 的 NAND 能力，并搭一个可逆半加器：

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

sim = AerSimulator()

# 第一段：Toffoli 造 NAND（目标位预置 1，双控翻转后取反）
for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    qc = QuantumCircuit(3, 1)
    qc.x(2)                          # 目标位 c 预置 1
    if a: qc.x(0)
    if b: qc.x(1)
    qc.ccx(0, 1, 2)                  # a∧b=1 时翻转 → c = ¬(a∧b)
    qc.measure(2, 0)
    print(f"a={a} b={b} -> {sim.run(qc, shots=1).result().get_counts()}")  # 仅 (1,1) 输出 0

# 第二段：可逆半加器（和位用 CNOT 求 XOR，进位用 Toffoli 求 AND）
qc = QuantumCircuit(4, 2)            # a, b, 和位, 进位
qc.x(0)                              # 例：a=1, b=0
qc.cx(0, 2)                          # 和位 = a⊕b = 1
qc.ccx(0, 1, 3)                      # 进位 = a∧b = 0
qc.measure([2, 3], [0, 1])
print(sim.run(qc, shots=1).result().get_counts())   # '01'：和=1、进位=0
```

第一段展示 Toffoli 造 NAND 的四种真值表组合——这正是 §4 讲的「目标位置 1 再双控翻转」；第二段用 CNOT 加 Toffoli 搭出半加器：和位用 CNOT 求 XOR、进位用 Toffoli 求 AND。**看见没有：量子算术（Shor 模幂的细胞）就是这样从 Toffoli 一颗一颗长出来的。** 把这两行线路换成大数，你就有了 Shor 算法里算术引擎的原型。

## 7 小结

- **Toffoli（CCNOT）**：$\lvert a,b,c\rangle \mapsto \lvert a,b, c\oplus (a\land b)\rangle$，双控制、自逆、$8\times8$ 矩阵只在 $\lvert 110\rangle \leftrightarrow \lvert 111\rangle$ 上做 X。
- **Fredkin（CSWAP）**：控制位为 1 时交换两比特，**守恒 1 的个数**，是守恒门。
- **公式解析**：Toffoli 的矩阵结构 = 「前两位地址 + 末位数据」的两层受控-X；作用两次等于恒等，故可逆。
- **经典万能**：Toffoli 目标位置 1 即得 NAND，配辅助位可模拟任意经典电路；量子线路是经典计算的严格超集。
- **辨析**：Toffoli 的「经典万能」≠「量子通用」，要造叠加还需 H；三比特门未引入 CNOT 之外的新纠缠能力，价值在可逆逻辑丰富度。
- **工程**：Toffoli ≈ 6 CNOT、T-count = 7，是量子算术（半加器、模幂）与复杂度分析的基本积木。

在下一节，我们补上线路模型的最后一个动作：**量子测量的线路表示与延迟测量原理**——测量是线路里唯一不可逆的步骤，而「把测量推迟到末尾」这条原理将贯穿后续所有算法分析。
