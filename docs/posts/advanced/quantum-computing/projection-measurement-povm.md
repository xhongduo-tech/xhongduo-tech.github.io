---
title: 投影测量与广义测量（POVM）
date: 2026-08-07
---

# 投影测量与广义测量（POVM）

<div class="epigraph">
<p>近年来我们经验的伟大扩展，揭示了简单力学概念的不足，并因此动摇了那种把观察建立在习惯性解释之上的基础。</p>
<footer>—— 尼尔斯 · 玻尔（Niels Bohr），《原子论与自然的描述》（Como 讲座，1927）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen & Chuang《量子计算》§2.2.4–2.2.5 ｜ 2026-08-07</p>
</div>

## 为什么从 POVM 开始

上一篇《量子力学的基本假设》在结尾留下了一个钩子：把「测量」这一条假设单独放大。今天就来兑现。上篇给出的测量算符 $\{M_m\}$ 是最一般的框架，但实际计算里我们几乎总落在两种特例上：**投影测量（projective measurement）** 与 **POVM（Positive Operator-Valued Measure，正算子值测度）**。

为什么需要两种？一句话：**投影测量回答「测量后系统变成什么」，POVM 回答「我们有多大概率看到什么」**。量子信息里大量问题只关心后者——两个态分不分得开、密钥有没有被窃听、相位估到第几位——测量后那个态怎样，往往不重要。把「后测态」这个包袱扔掉，测量会露出一个更自由、也更精巧的数学结构，这就是 POVM。它与机器学习里「软分类（soft classification）」、与统计里的「拒绝选项（rejection option）」在精神上一脉相承：实在分不清时，坦率地承认「不知道」。

## 1 投影测量：测量算符的特例

回顾测量公设：测量算符族 $\{M_m\}$ 给出概率 $p(m) = \langle\psi|M_m^\dagger M_m|\psi\rangle$ 与后测态 $M_m|\psi\rangle/\sqrt{p(m)}$。**投影测量**要求每个 $M_m$ 都是投影算符，即 $M_m^\dagger M_m = M_m$。此时算符自动厄米且幂等：$M_m^2 = M_m$，于是概率退化为玻恩定则 $p(m) = \langle\psi|M_m|\psi\rangle$。

教科书里投影测量常写成另一种等价的、更漂亮的形式——用**可观测量的谱分解**。设 $M$ 是一个厄米算符（可观测量），它的谱分解（见第一篇《本征值、本征向量与谱分解》）是

$$
M = \sum_m m\, P_m
$$

其中 $m$ 遍历 $M$ 的（可能重复的）本征值，$P_m$ 是本征值 $m$ 对应的本征空间上的投影：$P_m^2 = P_m$、$P_m = P_m^\dagger$、且 $\sum_m P_m = I$。测量「可观测量 $M$」的完整规则是：

$$
p(m) = \langle\psi|P_m|\psi\rangle, \qquad \text{测量后态} = \frac{P_m|\psi\rangle}{\sqrt{p(m)}}
$$

<span class="marginnote">当本征值 $m$ 简并（多个本征向量共享同一个本征值）时，$P_m$ 把整个简并子空间都投影出来，$M_m$ 在子空间内部取任何正交基都给出同样的统计——所以投影测量在简并情形下也只关心「子空间」，不关心「子空间里哪一支」。</span>

投影测量有两条可爱又危险的性质：**可重复性**——测完再测一次，得到同一结果的概率是 1（因为 $P_m^2 = P_m$）；**正交性**——$P_mP_{m'} = \delta_{mm'}P_m$，不同结果对应的后测态彼此正交。这两条性质让投影测量做不了两件事：测完东西后它把态「毁」成了基态，且它只在正交的备选之间做选择。

## 2 为什么还不够：区分两个非正交态

量子信息里最基本的任务之一：给一个未知态，它要么是 $|\psi_1\rangle$，要么是 $|\psi_2\rangle$，请分辨是谁。**如果 $|\psi_1\rangle$ 与 $|\psi_2\rangle$ 正交**，投影测量立刻完成：取 $P_1 = |\psi_1\rangle\langle\psi_1|$，测到结果 1 必是 $|\psi_1\rangle$，测到 0 必是 $|\psi_2\rangle$，零失误。

**如果两者不正交，任何测量都无法保证全对。** 直觉很直接：测量本质是在态上「拷问」出概率 $p(m) = |\langle\psi|\,m\rangle|^2$，而两个态在任意基下的概率分布总有重叠——就像两根几乎平行的筷子，无论从哪个角度看都有阴影交叠。这个直觉对应一个严格定理：**两个态可以被无损（零错误）区分，当且仅当它们正交**。<span class="marginnote">无损区分的严格表述用「迹距离」：两个态可完美区分当且仅当迹距离 $\|\rho_1 - \rho_2\|_1 = 2$，而纯态 $|\psi_1\rangle,|\psi_2\rangle$ 的迹距离 $= 2\sqrt{1-|\langle\psi_1|\psi_2\rangle|^2}$，取到 2 当且仅当 $\langle\psi_1|\psi_2\rangle = 0$。我们会在第四篇《纠缠的度量》里正式介绍迹距离。</span>

那不正交就彻底没戏了吗？不是。我们仍然可以要求：**凡是我们敢下结论的时候，结论一定是对的；拿不准的时候，就老实说「不确定」。** 这种「宁可拒绝也不出错」的策略叫**无损区分（unambiguous discrimination）**。它恰好是投影测量给不了、而 POVM 天生擅长的事——因为 POVM 允许 $E_m$ 彼此不正交，允许结果个数多于维度，允许有一支专门收集「无法判断」的概率。

## 3 广义测量与 POVM：扔掉后测态

从一般测量 $\{M_m\}$ 出发，观测者能看到的全部统计信息只有

$$
p(m) = \langle\psi|\,M_m^\dagger M_m\,|\psi\rangle
$$

它只依赖组合 $E_m \equiv M_m^\dagger M_m$。把这些 $E_m$ 单独拎出来定义：

**POVM（正算子值测度）**：一族满足以下两条的正算符 $E_m$（$E_m \geq 0$，即厄米且非负本征值）：

$$
\sum_m E_m = I
$$

对任意态 $\rho$，测量得到结果 $m$ 的概率为 $p(m) = \mathrm{tr}(E_m\rho)$（纯态时即 $\langle\psi|E_m|\psi\rangle$）。$E_m$ 称为 **POVM 元（POVM element）**。

**辨析｜易错点：** 三个地方和投影测量完全不同。第一，$E_m$ **不必是投影**——它只需非负，可以秩大于 1，甚至「胖乎乎」地占满整个空间；第二，$E_mE_{m'} = 0$ **不再成立**，$E_m$ 可以彼此「重叠」，这正是无损区分能工作的原因；第三，POVM **没有**后测态公式——因为同一个 $E_m$ 可以被不同的 $M_m$ 实现（例如 $M_m = U_m\sqrt{E_m}$ 对任意幺正 $U_m$ 都给同一个 $E_m$），后测态取决于你选的 $U_m$，而 POVM 对此沉默。

还有一个概念上的澄清：**POVM 不是「比投影测量弱」的测量，而是「更一般」的测量。** 任何投影测量都是一个 POVM（取 $E_m = P_m$），反过来却不然。而且著名的 **Naimark 扩张定理（Neumark dilation）** 保证：任何 POVM 都可以通过「引入一个辅助系统、在其上做一次投影测量、再把辅助系统丢弃」来实现<span class="marginnote">Naimark 扩张是「测量即扩展」思想的数学化：想要非正交、欠定、过完备的测量，就把系统放进更大的希尔伯特空间里做个正正经经的投影测量，然后把多余的子系统「部分迹」掉。而「部分迹」正是下一篇《密度算符：混合态与部分迹》的主角——两个主题在概念上是咬合的。</span>。所以 POVM 不是一个另起炉灶的测量理论，而是投影测量在更大空间里的投影。这个「借空间」的技巧，将反复出现在隐形传态、量子密钥分发与量子纠错里。

## 4 公式解析：用 POVM 无损区分 $|0\rangle$ 与 $|+\rangle$

把上面的抽象全部落到一个具体的算例上。设未知态是下面两个非正交态之一（$|\langle0|+\rangle| = 1/\sqrt2$）：

$$
|\psi_1\rangle = |0\rangle, \qquad |\psi_2\rangle = |+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt2}
$$

构造三个 POVM 元（其中 $\alpha = \frac{\sqrt2}{1+\sqrt2}$）：

$$
E_1 = \alpha\,|1\rangle\langle1|, \qquad
E_2 = \alpha\,|-\rangle\langle-| = \alpha\,\frac{(|0\rangle-|1\rangle)(\langle0|-\langle1|)}{2}, \qquad
E_3 = I - E_1 - E_2
$$

分三步拆解，看它为什么能「无损」区分。

- **第一步，验证它是合法的 POVM**。$E_1, E_2$ 是非负算符的倍数，显然 $E_1, E_2 \geq 0$；系数 $\alpha$ 是特意取的，使 $E_3 = I - E_1 - E_2$ 依旧非负（读者可验：$\alpha \leq 1$，且 $E_3$ 的两个本征值分别为 $1-\alpha$ 与 $1-\alpha\cdot\frac{1+\sqrt2}{2}\cdot\frac{\sqrt2}{1+\sqrt2} = 1-\frac{\sqrt2}{2} \geq 0$）。三者和为 $I$，合法性成立。

- **第二步，看「打不中」的方向**。对 $|\psi_1\rangle = |0\rangle$：

$$
\langle\psi_1|E_1|\psi_1\rangle = \alpha\,\langle0|1\rangle\langle1|0\rangle = 0
$$

对 $|\psi_2\rangle = |+\rangle$：

$$
\langle\psi_2|E_2|\psi_2\rangle = \alpha\,|\langle-|+\rangle|^2 = 0
$$

**这就是无损的机制**：$E_1$ 与 $|0\rangle$ 完全错开，$E_2$ 与 $|+\rangle$ 完全错开。于是「结果 1 出现」这一事件对 $|\psi_1\rangle$ 而言概率为零——一旦看到结果 1，就可以肯定态不是 $|\psi_1\rangle$，只能是 $|\psi_2\rangle$；反之，看到结果 2，肯定态是 $|\psi_1\rangle$。结论永远正确。

- **第三步，计算成功与失败的概率**。用 $|\langle0|-\rangle|^2 = |\langle1|+\rangle|^2 = 1/2$：

$$
\langle\psi_1|E_2|\psi_1\rangle = \alpha\cdot\frac12 = \frac{\alpha}{2}, \qquad
\langle\psi_2|E_1|\psi_2\rangle = \alpha\cdot\frac12 = \frac{\alpha}{2}
$$

两个态各自的**成功识别概率**都是 $\alpha/2 = \frac{\sqrt2}{2(1+\sqrt2)} \approx 0.293$；其余概率落在 $E_3$ 上，即「不确定，放弃」。而且可以证明这个 0.293 是**最优**的：无损区分两个纯态的成功率上界是 $1 - |\langle\psi_1|\psi_2\rangle| = 1 - 1/\sqrt2 \approx 0.293$，上面的构造恰好达到——POVM 不仅能用，还用得最省。

```python
# 用 NumPy 核对上述概率（E_m 在计算基 {|0⟩, |1⟩} 下）
import numpy as np

psi1 = np.array([1, 0])                     # |0⟩
psi2 = np.array([1, 1]) / np.sqrt(2)        # |+⟩
alpha = np.sqrt(2) / (1 + np.sqrt(2))

E1 = alpha * np.outer(np.array([0, 1]), np.array([0, 1]))   # α|1⟩⟨1|
minus = np.array([1, -1]) / np.sqrt(2)                      # |-⟩
E2 = alpha * np.outer(minus, minus)                         # α|-⟩⟨-|
E3 = np.eye(2) - E1 - E2

for name, s in [("|0⟩", psi1), ("|+⟩", psi2)]:
    ps = [float(s.conj() @ E @ s) for E in (E1, E2, E3)]
    print(name, [round(p, 4) for p in ps])
# 输出近似：
#   |0⟩ → [0.0, 0.2929, 0.7071]
#   |+⟩ → [0.2929, 0.0, 0.7071]
```

**辨析｜易错点：** 结果 1 对应的是「态是 $|\psi_2\rangle$」还是「态是 $|\psi_1\rangle$」？反直觉但很关键：因为 $E_1$ 与 $|\psi_1\rangle$ 错开，**看到结果 1 意味着态不是 $|\psi_1\rangle$，故是 $|\psi_2\rangle$**；同理看到结果 2 意味着态是 $|\psi_1\rangle$。编号与判断是「交叉」的，写程序或写文章时最容易搞反。

## 5 小结

- **投影测量**：由可观测量的谱分解 $M = \sum_m mP_m$ 描述，$P_m$ 是正交投影，满足可重复性与正交性；后测态是 $P_m|\psi\rangle/\sqrt{p(m)}$。
- **POVM**：只保留测量结果的统计规律，$E_m = M_m^\dagger M_m \geq 0$、$\sum_m E_m = I$，概率 $p(m) = \mathrm{tr}(E_m\rho)$；$E_m$ 不必正交、不必是投影、个数可多于维度。
- **无损区分的界线**：两个态可零错误区分当且仅当正交；非正交态最多做到「成功 $1 - |\langle\psi_1|\psi_2\rangle|$，其余诚实地说不知道」，且存在达到该界的 POVM。
- **Naimark 扩张**：任何 POVM 都能用「大空间上的投影测量 + 丢弃辅助系统」实现——测量问题的解决思路常常是借一个更大的空间。
- POVM 是「软测量」：允许不确定，但不允许错误——这个思想将在量子密钥分发（BB84 的窃听检测）与量子隐形传态里反复出现。

在下一节，我们将面对投影测量留下的一个隐患：**它默认观测者「知道」系统处于哪个纯态。** 可现实里系统可能处于「不知道哪个、只知道按概率分布」的混合态，也可能只是一个纠缠大系统里被我们单独拿出来的子系统。描述这两者的工具，就是**密度算符**。
