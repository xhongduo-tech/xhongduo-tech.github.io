---
title: 贝尔态（Bell states）与贝尔测量
date: 2026-08-07
---

# 贝尔态（Bell states）与贝尔测量

<div class="epigraph">
<p>量子态是真实的，尽管它们是空间性的……当我们测量到其中一个的时候，另一个也会同时获得确定的值。</p>
<footer>—— 爱因斯坦、波多尔斯基与罗森（Einstein, Podolsky &amp; Rosen）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen &amp; Chuang《量子计算与量子信息》§1.3.6、§2.5 ｜ 2026-08-07</p>
</div>

## 为什么从贝尔态开始

上一节我们知道了「写不成积的态」就是纠缠态，并反复用 $\lvert\Phi^+\rangle = \frac{1}{\sqrt2}(\lvert00\rangle + \lvert11\rangle)$ 当例子。这个态并非孤例——它属于一族**最大纠缠态**，称为**贝尔态（Bell states）**。四个贝尔态构成两比特系统的一组正交基，而且每个都能由一条统一的小线路制造出来；更妙的是，对贝尔态的**测量**（贝尔测量）能把一个两比特系统的全部信息一次性读出——这正是隐形传态、超密编码的共同机关。<span class="marginnote">贝尔态以物理学家贝尔（John Stewart Bell）命名。1964 年他用量子纠缠构造不等式，直接撼动了爱因斯坦的隐变量世界观——下一节《EPR 佯谬与隐变量理论》就讲这件事。先在这里把四个态本身摸熟。</span>本节同时回答「如何制造」与「如何测量」。

## 1 四个贝尔态与它们的统一生成线路

两比特系统 $\lvert b_1 b_2\rangle$ 的四个贝尔态是

$$
\lvert\Phi^+\rangle = \frac{1}{\sqrt2}(\lvert00\rangle + \lvert11\rangle), \quad
\lvert\Phi^-\rangle = \frac{1}{\sqrt2}(\lvert00\rangle - \lvert11\rangle)
$$

$$
\lvert\Psi^+\rangle = \frac{1}{\sqrt2}(\lvert01\rangle + \lvert10\rangle), \quad
\lvert\Psi^-\rangle = \frac{1}{\sqrt2}(\lvert01\rangle - \lvert10\rangle)
$$

记忆方法：$\Phi$ 是「两比特相同」的叠加（00 与 11），$\Psi$ 是「两比特相反」的叠加（01 与 10），正负号区分相位。它们**共同的关键性质**是：每个态对任意一方的测量，结果都完全随机（各 $\tfrac12$），但两方结果**总是强相关**——测 $\lvert\Phi^+\rangle$ 只要一边得 0，另一边必得 0；测 $\lvert\Psi^-\rangle$ 两边结果必相反。<span class="marginnote">这种「单边随机、双边确定」的相关性，就是「最大纠缠」的操作定义。它也是量子密码学里检测窃听、量子隐形传态里传状态的基础——一个贝尔态 = 一个共享纠缠对（ebit）。</span>

四者还能由一条线路统一生成：先 $H$ 作用在第一个比特上，再 CNOT（第一个比特控制第二个）。以 $\lvert00\rangle$ 为输入得到 $\lvert\Phi^+\rangle$，以 $\lvert01\rangle$ 输入得到 $\lvert\Phi^-\rangle$，以此类推。把「输入 → 输出 → 贝尔测量结果」合成一张速查表：

| 输入 | 生成线路（$H$ + CNOT）输出 | 贝尔测量读出 $(x,y)$ |
| --- | --- | --- |
| $\lvert00\rangle$ | $\lvert\Phi^+\rangle$ | $(0,0)$ |
| $\lvert01\rangle$ | $\lvert\Phi^-\rangle$ | $(0,1)$ |
| $\lvert10\rangle$ | $\lvert\Psi^+\rangle$ | $(1,0)$ |
| $\lvert11\rangle$ | $\lvert\Psi^-\rangle$ | $(1,1)$ |

这张表把「生成」与「测量」两件事锁死在一起：制备哪个贝尔态，贝尔测量就必定读出对应的 $(x,y)$——不存在其他结果。

## 2 公式解析：$\lvert00\rangle \xrightarrow{H \otimes I} \xrightarrow{CNOT} \lvert\Phi^+\rangle$

把这条线路拆成三步，看纠缠是如何「被制造」出来的：

$$
\lvert00\rangle \xrightarrow{H\otimes I} \frac{1}{\sqrt2}(\lvert0\rangle+\lvert1\rangle) \otimes \lvert0\rangle \xrightarrow{CNOT} \frac{1}{\sqrt2}(\lvert00\rangle + \lvert11\rangle)
$$

- **第一步，Hadamard 制造叠加**：$H\lvert0\rangle = \frac{1}{\sqrt2}(\lvert0\rangle + \lvert1\rangle)$。第一个比特变成叠加态，两个比特仍可分离，尚未纠缠。
- **第二步，CNOT 制造纠缠**：CNOT 以第一个比特为控制位：控制位为 $\lvert0\rangle$ 时目标位不变，为 $\lvert1\rangle$ 时目标位翻转。于是 $\lvert0\rangle\otimes\lvert0\rangle \to \lvert00\rangle$、$\lvert1\rangle\otimes\lvert0\rangle \to \lvert11\rangle$。
- **第三步，线性性完成运算**：CNOT 是线性算符，直接作用在第一步的叠加态上，把两项分别映射，得到 $\frac{1}{\sqrt2}(\lvert00\rangle + \lvert11\rangle)$。<span class="marginnote">关键在第三步：CNOT 同时作用在两个分支上，把一个「比特叠加」放大成一个「比特对叠加」。纠缠不是「加上去」的，而是「通过受控门把叠加从单比特扩散到多比特」的——这是理解一切纠缠线路生成的心法。</span>

## 3 贝尔测量：在贝尔基下的投影测量

**贝尔测量（Bell measurement）**就是在贝尔基 $\{\lvert\Phi^\pm\rangle, \lvert\Psi^\pm\rangle\}$ 下做投影测量。它把任意两比特态投影到某个贝尔态上，返回四个可能结果之一（携带 2 比特经典信息）。<span class="marginnote">注意：贝尔测量不是「分别测两个比特再把结果拼起来」。标准基测量会丢失相位信息（把 $\lvert\Psi^-\rangle$ 与 $\lvert\Psi^+\rangle$ 混为一谈），贝尔测量则四个态完全分辨——这正是隐形传态能传「未知态」而经典测量不能的原因。</span>

贝尔测量的标准线路是**「CNOT + H + 计算基测量」**：

1. 先把 CNOT 作用在两个比特上（第一个为控制），
2. 再对第一个比特做 $H$ 门，
3. 最后对两个比特分别做标准基测量，读出结果 $x, y \in \{0,1\}$。

这个线路把「贝尔基下的测量」转换成「计算基下的测量」——四个结果 $(x,y)$ 一一对应四个贝尔态。例如输入 $\lvert\Phi^+\rangle$ 会读出 $(0,0)$，输入 $\lvert\Psi^-\rangle$ 会读出 $(1,1)$。

## 4 贝尔测量的数学：酉变换对角化

为什么「CNOT + H + 计算基」就等于贝尔测量？因为 $U_{B} = (H \otimes I)\, CNOT$ 是一个酉变换，它把贝尔基映到计算基：

$$
U_B \lvert\Phi^+\rangle = \lvert00\rangle, \quad U_B \lvert\Phi^-\rangle = \lvert01\rangle, \quad U_B \lvert\Psi^+\rangle = \lvert10\rangle, \quad U_B \lvert\Psi^-\rangle = \lvert11\rangle
$$

三步拆解这条公式的用意：

- **第一步，换基**：测量前的 $U_B$ 把「贝尔基」旋转成「计算基」。物理上，我们并不真的在贝尔基测量，而是先把系统转一个角度，让贝尔态变成计算基里的确定态。
- **第二步，标准基测量**：此时再测计算基，四个贝尔态分别读出唯一确定的结果 $(x,y)$，互不混淆。
- **第三步，逆推**：$U_B$ 是酉的、可逆的，从读出 $(x,y)$ 就能唯一反推输入是哪个贝尔态。<span class="marginnote">这套「先转基、再测量」的思路贯穿量子信息：测量不是只能钉死在一个固定基上，任何酉变换 + 计算基测量 ≡ 一个更一般的测量。第三篇的《量子测量的线路表示》里我们已经为这个结论铺过路。</span>

**辨析｜易错点：** 贝尔测量输出的 $(x,y)$ 是**经典信息**（读取结果），不是「测得哪个贝尔态后系统变成什么」的完整描述。测量后系统确实坍缩到对应的贝尔态，但在隐形传态场景里那个态属于遥远的 Bob，Alice 手里只剩 $(x,y)$——这正是「测量破坏纠缠、经典比特携带信息」的分工。

## 5 贝尔测量在 Qiskit 中的一步实现

生成与测量一对贝尔态，在 Qiskit 里各只有三步，正好对应本节的生成线路与贝尔测量线路：

```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(2, 2)
qc.h(0)             # H：制造叠加
qc.cx(0, 1)         # CNOT：制造纠缠 → 输入 |00⟩ 得 |Φ+⟩
qc.cx(0, 1)         # 贝尔测量第一步：再一个 CNOT
qc.h(0)             # 贝尔测量第二步：H
qc.measure([0, 1], [0, 1])   # 第三步：计算基测量
```

制备的是 $\lvert\Phi^+\rangle$：先 $H$、再 CNOT，得到 $\frac{1}{\sqrt2}(\lvert00\rangle + \lvert11\rangle)$；随后「CNOT + $H$ + 测量」正是贝尔测量的标准线路，结果必然读出 $(0,0)$。若把输入改成 $\lvert01\rangle$，制造出的是 $\lvert\Phi^-\rangle$，读出 $(0,1)$。**生成 + 贝尔测量两条线路拼在一起，就是隐形传态与超密编码的公共内核**——第十二篇《Qiskit 实践》会在这个内核上直接叠加传态协议。

## 6 贝尔态的资源角色

两个贝尔态相关的应用前已登场，这里点一下它们如何复用本节内容：

**量子隐形传态**：Alice 与 Bob 共享 $\lvert\Phi^+\rangle$，Alice 对「要传的态 + 自己那半」做贝尔测量，把 2 比特结果经典告诉 Bob，Bob 据此旋转即可还原。<span class="marginnote">全部机制在第二篇《量子隐形传态》里已展开；贝尔测量正是那一步「把纠缠变成经典 2 比特」的操作。</span>
- **超密编码**：Alice 通过操纵自己那半贝尔态，把 2 比特经典信息编码进 1 个量子比特的四种变换，Bob 做贝尔测量读回 2 比特——「1 个量子比特 ≅ 2 个经典比特」的兑换在此发生。

## 7 小结

- 四个**贝尔态**：$\lvert\Phi^\pm\rangle, \lvert\Psi^\pm\rangle$，是两比特最大纠缠态，构成一组正交基。
- 统一生成线路：**H + CNOT**；纠缠是「把单比特叠加通过受控门扩散成比特对叠加」。
- **贝尔测量** = 贝尔基投影测量，实现为「CNOT + H + 计算基测量」，四个结果一一对应四个贝尔态。
- **速查**：输入 $\lvert00\rangle/\lvert01\rangle/\lvert10\rangle/\lvert11\rangle$ → $\lvert\Phi^+\rangle/\lvert\Phi^-\rangle/\lvert\Psi^+\rangle/\lvert\Psi^-\rangle$ → 读出 $(0,0)/(0,1)/(1,0)/(1,1)$。
- 贝尔态是 ebit 资源：**1 个共享贝尔态 + 2 比特经典 = 传 1 个未知量子比特（隐形传态）**；1 个量子比特 + 1 个共享贝尔态 = 2 比特经典（超密编码）。

在下一节，我们将回到 1935 年那场著名的思想实验：EPR 佯谬如何从贝尔态推出「量子力学不完整」的结论，以及它如何意外地指向**隐变量理论**。
