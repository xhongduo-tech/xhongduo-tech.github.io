---
title: 多量子比特系统与纠缠态
date: 2026-08-07
---

# 多量子比特系统与纠缠态

<div class="epigraph">
<p>我称它不是量子力学的一个偶然特征，而是量子力学的典型特征——正是它使量子力学彻底背离了经典的思路。</p>
<footer>—— 埃尔温 · 薛定谔（Erwin Schrödinger，Entanglement 一词的命名者）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen & Chuang《量子计算》§1.3 多量子比特 ｜ 2026-08-07</p>
</div>

## 为什么从多量子比特开始

前面几篇我们反复说「布洛赫球只对单个量子比特成立，多比特是另一片天地」。现在正式踏入这片天地。**多量子比特系统是量子计算威力的真正来源**——单个量子比特只是「一个更好的随机数发生器」，而 $n$ 个量子比特的态空间是 $2^n$ 维，这里面冒出的**纠缠（entanglement）**，是经典世界完全没有的物理资源。

从「从极限到大模型」的主线看，这一步等于从「单个神经元」走向「神经网络」：单比特的叠加是「一」，多比特的纠缠才是「多」——而一切量子算法（Deutsch-Jozsa、Shor、Grover）的加速，本质都是在对纠缠做编排。今天我们要回答三个问题：多比特的态空间长什么样？什么样的态算纠缠？纠缠到底「怪」在哪？

## 1 多比特态空间：指数爆炸的起点

$n$ 个量子比特的态空间是**张量积空间**（见第一篇《张量积》）：

$$
(\mathbb{C}^2)^{\otimes n} = \mathbb{C}^2 \otimes \cdots \otimes \mathbb{C}^2, \qquad \dim = 2^n
$$

对 $n = 2$，基矢是 $\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$，任意两比特态写成

$$
|\psi\rangle = c_{00}|00\rangle + c_{01}|01\rangle + c_{10}|10\rangle + c_{11}|11\rangle, \qquad \sum_{ij}|c_{ij}|^2 = 1
$$

其中 $|ab\rangle$ 是 $|a\rangle \otimes |b\rangle$ 的简写。一般地，**$n$ 个量子比特的态由 $2^n$ 个复振幅描述**。这个数字的膨胀速度值得停下来体会：$n = 300$ 时，$2^{300} \approx 2\times10^{90}$——比可观测宇宙中的原子总数（约 $10^{80}$）还多出十亿亿亿倍。<span class="marginnote">这就是「量子并行」的字面来源：一个 300 比特的寄存器，一个态矢量里「同时装着」$2^{90}$ 个量级的振幅。但要提醒的是，这些振幅不是「并行计算的独立分支」——它们必须协同演化、干涉，最终被压缩成测量的一串比特，见《为什么更快》一文的误区辨析。</span>

**重点：** 多比特态空间是 $2^n$ 维，但一个量子线路里你只能施加 $n$ 个量子比特上的门——指数维数是你「隐形持有」的资源，你控制的是少数几个旋钮，状态却在这个巨大的空间里漂移。这正是量子算法的核心张力：如何用多项式多个门，把巨大的振幅空间里「对的答案」放大到可见。

## 2 张量积与可分离态

多比特态里最简单的一类叫**乘积态（product state）**，也叫**可分离态（separable state）**：它由每个比特各自的状态张量积而成，

$$
|\psi\rangle = |a\rangle \otimes |b\rangle, \qquad |a\rangle \in \mathbb{C}^2, \ |b\rangle \in \mathbb{C}^2
$$

例如 $|0\rangle \otimes |+\rangle = |0+\rangle$。对乘积态，每个比特「各管各的」：测量第一个比特不会透露第二个比特的任何信息，两个比特就像两个独立硬币。

**辨析｜易错点：** 不是所有「两比特态」都是乘积态。判断标准是：**能不能把它分解成两个单比特态的乘积**。比如 $\frac{|00\rangle + |01\rangle}{\sqrt2} = |0\rangle \otimes \frac{|0\rangle + |1\rangle}{\sqrt2}$ 是可分的；但 $\frac{|00\rangle + |11\rangle}{\sqrt2}$ 你试遍所有 $|a\rangle, |b\rangle$ 也拼不出来（下一节证明）。**可分离态只是两比特态空间的一个低维子集**——绝大多数两比特态都是纠缠态。

## 3 纠缠态：写不成乘积的态

**纠缠态（entangled state）**：不能写成乘积态的态。最重要的例子是四个**贝尔态（Bell states）**：

$$
|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt2}, \qquad |\Phi^-\rangle = \frac{|00\rangle - |11\rangle}{\sqrt2}
$$

$$
|\Psi^+\rangle = \frac{|01\rangle + |10\rangle}{\sqrt2}, \qquad |\Psi^-\rangle = \frac{|01\rangle - |10\rangle}{\sqrt2}
$$

贝尔态可以用一个极简单的线路造出来：先对第一个比特施加 $H$ 门，再以它为控制位、以第二个比特为目标位施加 CNOT 门：

![制备贝尔态 |Φ⁺⟩ 的量子线路](/images/quantum-computing/multiqubit-systems-entanglement-1.svg)

**重点：纠缠最直观的体现在于测量相关性。** 对 $|\Phi^+\rangle$ 测第一个比特：测到 0 的概率是 $1/2$，测到 1 的概率是 $1/2$，完全随机；但**测到 0 之后，第二个比特必然也是 0；测到 1 之后，第二个比特必然也是 1**。两个比特就像被一根看不见的线拴在一起——即使它们相隔几公里，测量结果也完美同步。这正是爱因斯坦 1935 年为之不安的「幽灵般的超距作用」（spooky action at a distance）。<span class="marginnote">「超距」的措辞要小心：纠缠不允许<strong>超光速传信息</strong>。Alice 在远端测自己的比特，结果仍是随机的（她无法控制读到 0 还是 1），Bob 单独看自己比特的统计分布也不会有任何改变——相关性只有把两侧的读数<strong>拿回来对比</strong>时才显现，而「拿回来对比」是经典步骤，跑不赢光速。这一条叫 no-signaling，是量子纠缠与科幻「心灵感应」的分界线。</span>

## 4 纠缠的本质：子系统没有独立状态

相关性只是表象。纠缠更本质的地方在于：**纠缠态的每个子系统，单独拿出来看，处于完全混合态**。

以 $|\Phi^+\rangle$ 为例，对它做部分迹（partial trace，见第一篇《密度算符》），丢掉第二个比特：

$$
\rho_A = \mathrm{Tr}_B\Big(|\Phi^+\rangle\langle\Phi^+|\Big) = \frac{I}{2} = \frac12|0\rangle\langle0| + \frac12|1\rangle\langle1|
$$

子系统 A 的密度算符是**完全混合态**——它没有任何确定性，就像一枚抛在空中的公平硬币。**纠缠态的全部信息不在任何一个子系统里，而在子系统之间的「关联」里。** 这解释了为什么纠缠这么难懂：你盯着其中一个比特看，看到的只有随机；必须把两个比特放在一起看，才看到结构。<span class="marginnote">这与经典直觉彻底相反：经典世界里，两个物体的「联合状态」的信息量 = 各自信息量之和；量子世界里，纠缠态的联合信息可以「大于」各部分之和——部分各自是纯随机，整体却高度确定。这是「整体大于部分之和」在物理里最精确的实例。</span>

用一个统一工具概括：**Schmidt 分解**。任意两比特纯态 $|\psi\rangle = \sum_i s_i\,|i_A\rangle|i_B\rangle$（$s_i \ge 0$，$\sum_i s_i^2 = 1$）。**纠缠的判据就是 Schmidt 系数 $s_i$ 的个数**：只有一个非零系数 $\Leftrightarrow$ 可分离；有两个 $\Leftrightarrow$ 纠缠。$|\Phi^+\rangle$ 的 Schmidt 系数是 $(1/\sqrt2, 1/\sqrt2)$——两个，纠缠。这是第一篇《Schmidt 分解与纯化》的结论在纠缠上的首次实战。

## 5 公式解析：证明 |Φ⁺⟩ 不可分 + 部分迹计算

把「纠缠 = 写不成乘积」用两段推导钉死。

**第一段：反证法证明 $|\Phi^+\rangle$ 不可分。** 假设它可以写成乘积 $(a|0\rangle + b|1\rangle)\otimes(c|0\rangle + d|1\rangle)$，展开：

$$
(a|0\rangle + b|1\rangle)\otimes(c|0\rangle + d|1\rangle) = ac|00\rangle + ad|01\rangle + bc|10\rangle + bd|11\rangle
$$

与 $|\Phi^+\rangle = \frac{1}{\sqrt2}|00\rangle + \frac{1}{\sqrt2}|11\rangle$ 逐系数比对，必须同时满足

$$
ac = \frac1{\sqrt2}, \quad ad = 0, \quad bc = 0, \quad bd = \frac1{\sqrt2}
$$

由 $ad = 0$ 得 $a = 0$ 或 $d = 0$。若 $a = 0$，则 $ac = 0 \neq 1/\sqrt2$，矛盾；若 $d = 0$，则 $bd = 0 \neq 1/\sqrt2$，矛盾。**所以不存在这样的乘积分解，$|\Phi^+\rangle$ 是纠缠态。** 看中间那一步：矛盾的本质是「两个振幅都要占满 $1/\sqrt2$，而乘积分解强迫四个系数两两相乘」——乘积态的结构根本放不下纠缠态的信息。

**第二段：部分迹算出 $\rho_A = I/2$。** 先展开 $\rho = |\Phi^+\rangle\langle\Phi^+|$：

$$
\rho = \frac12\Big(|00\rangle\langle00| + |00\rangle\langle11| + |11\rangle\langle00| + |11\rangle\langle11|\Big)
$$

对第二个比特求部分迹。规则是 $\mathrm{Tr}_B\big(|i\rangle\langle j|\otimes|k\rangle\langle l|\big) = |i\rangle\langle j|\cdot\langle l|k\rangle$。逐项看：

- $|00\rangle\langle00| = |0\rangle\langle0|\otimes|0\rangle\langle0| \longrightarrow |0\rangle\langle0|\cdot\langle0|0\rangle = |0\rangle\langle0|$
- $|00\rangle\langle11| = |0\rangle\langle0|\otimes|0\rangle\langle1| \longrightarrow |0\rangle\langle0|\cdot\langle1|0\rangle = 0$
- $|11\rangle\langle00| \longrightarrow 0$（同理）
- $|11\rangle\langle11| \longrightarrow |1\rangle\langle1|$

三项为零、两项保留，于是

$$
\rho_A = \frac12\Big(|0\rangle\langle0| + |1\rangle\langle1|\Big) = \frac{I}{2}
$$

**辨析｜易错点：** $\rho_A = I/2$ 是「完全混合态」，但**它绝不等于「$|\Phi^+\rangle$ 是随机二选一的状态」**。混合态的语言是「$|0\rangle$ 与 $|1\rangle$ 按概率各是各」（没有相干、没有关联），而 $|\Phi^+\rangle$ 的两个分量之间**相干且关联**——$I/2$ 只是「只盯着一个比特看」时信息丢失后的投影。这与单比特篇「叠加 ≠ 混合」是同一堂课，只是升级到了两比特版本。

## 6 小结

- $n$ 个量子比特的态空间是 $(\mathbb{C}^2)^{\otimes n}$，维数 $2^n$ 指数增长；$n=300$ 时振幅数远超宇宙原子数。
- **可分离态** = 乘积态 $|a\rangle\otimes|b\rangle$，各比特独立；**纠缠态** = 写不成乘积的态。
- 四个**贝尔态**是两比特纠缠态的代表，由 $H$ + CNOT 线路制备；对 $|\Phi^+\rangle$ 测一个比特，另一比特随之确定——**测量相关性**。
- 纠缠的**本质**：子系统的约化密度算符是完全混合态（$\rho_A = I/2$），信息全部藏在关联里；Schmidt 系数个数 ≥ 2 即纠缠。
- 纠缠**不能超光速传信息**（no-signaling）：相关性需要经典手段对照才显现。
- 反证法证明不可分：乘积分解强迫 $ac, ad, bc, bd$ 同时成立，与 $ad = 0$、$bd = 1/\sqrt2$ 矛盾。

在下一节，我们面对一个更尖锐的推论：**不可克隆定理**——为什么量子比特无法被「复制粘贴」，而经典比特可以。这个「做不到」的定理，恰好是量子密码安全的根基，也是量子隐形传态（再下一节）必须用「拆了重装」而不是「复制」来完成传输的原因。
