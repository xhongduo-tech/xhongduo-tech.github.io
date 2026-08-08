---
title: Bernstein-Vazirani 算法
date: 2026-08-07
---

# Bernstein-Vazirani 算法

<div class="epigraph">
<p>量子计算不仅能在某些问题上更快，还能以最优雅的方式揭示隐藏的结构。</p>
<footer>—— 伯恩斯坦（Ethan Bernstein）与瓦齐拉尼（Umesh Vazirani）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen &amp; Chuang《量子计算与量子信息》§1.4.1（BV 问题）｜ 2026-08-07</p>
</div>

## 为什么从 Bernstein-Vazirani 算法开始

Deutsch-Jozsa 让我们看到了「一次查询读出全局性质」的威力，但它的问题是刻意的、学术化的。**Bernstein-Vazirani（BV）问题**换了一种问法：给定一个隐藏的比特串 $s \in \{0,1\}^n$，函数被定义为内积 $f(x) = s \cdot x \pmod 2$，问 $s$ 是多少？<span class="marginnote">BV 问题出自 E. Bernstein &amp; U. Vazirani, "Quantum Complexity Theory," <i>SIAM J. Comput.</i> 26 (1997) 1411——就是这篇论文最早定义了 BQP 复杂度类（量子多项式时间），并提出「量子图灵机」的严格模型。</span>经典算法需要 $n$ 次查询（每次问一个基向量 $e_i$，得到 $s_i$），而 BV 算法**一次查询就把整个 $s$ 读出来**。虽然只是「线性加速」，但它的结构比 Deutsch-Jozsa 更本质——它把「干涉读出多个比特」的技巧用到了极致，也是理解 Simon、Shor 内积相位结构的台阶。

## 1 问题设定与经典做法

**BV 问题**：设隐藏串 $s \in \{0,1\}^n$，黑盒计算 $f_s(x) = s \cdot x \pmod 2 = \bigoplus_{i: s_i=1} x_i$。任务是确定 $s$。

经典确定性算法：依次查询 $x = e_1, e_2, \dots, e_n$（$e_i$ 是第 $i$ 位为 1 的单位向量）。$f_s(e_i) = s_i$，于是 $n$ 次查询后拼出完整 $s$。<span class="marginnote">能否更少？能证明下界是 $n$：每个查询只返回一个比特，而 $s$ 有 $n$ 个未知比特，信息论上至少需要 $n$ 个比特的回答。量子算法的惊人之处就在于绕过了这个「信息论下界」——因为一次量子查询并行返回的是「相位叠加」，信息藏在相位里。</span>这个「信息论下界」对比让 BV 加速的含义变得清晰：不是常数倍，而是**结构性的**。

## 2 BV 算法的线路

线路与 Deutsch-Jozsa 几乎一模一样，只差查询的语义：

1. 前 $n$ 个比特制备 $\lvert0\rangle^{\otimes n}$，全部作用 $H$；辅助比特制备 $\lvert-\rangle$。
2. 作用相位查询 $O_{f_s}$：$\lvert x\rangle \to (-1)^{s\cdot x}\lvert x\rangle$。
3. 前 $n$ 个比特全部作用 $H$，测量。

第 2 步后的态是 $\frac{1}{\sqrt{2^n}}\sum_x (-1)^{s\cdot x}\lvert x\rangle$——这正是 $s$ 的**Walsh-Hadamard 变换**：它把「内积相位」变成一个集中于 $s$ 的态。第 3 步的 $H^{\otimes n}$ 把它逆变换回来，测量直接得到 $s$。<span class="marginnote">记忆点：$H^{\otimes n}$ 作用在 $\lvert s\rangle$ 上会展开成「所有 $x$ 上的内积相位」$\frac{1}{\sqrt{2^n}}\sum_x(-1)^{s\cdot x}\lvert x\rangle$；因为 $H$ 自逆，再作用一次就回到 $\lvert s\rangle$。BV 算法等于「先展开、再收回」，相位编码在其中充当了「身份标签」。</span>

## 3 公式解析：为什么测量直接给出 $s$

核心恒等式是一条漂亮的酉变换恒等式：

$$
H^{\otimes n} \left( \frac{1}{\sqrt{2^n}}\sum_{x} (-1)^{s\cdot x}\lvert x\rangle \right) = \lvert s\rangle
$$

三步拆解：

- **第一步，把 $H$ 作用到 $\lvert x\rangle$**：$H^{\otimes n}\lvert x\rangle = \frac{1}{\sqrt{2^n}}\sum_y (-1)^{x\cdot y}\lvert y\rangle$。作用后总态为 $\frac{1}{2^n}\sum_{x,y}(-1)^{s\cdot x + x\cdot y}\lvert y\rangle$。
- **第二步，对 $x$ 求和**：固定 $y$，$\sum_x (-1)^{(s+y)\cdot x}$。若 $s + y \ne 0$，正负项等数相消得 0；若 $s + y = 0$（即 $y = s$），得 $2^n$。
- **第三步，归一**：只剩 $y=s$ 这一项，系数 $1$。所以测量必得 $s$，一次成功。<span class="marginnote">这就是「干涉读出内积」的定量机制：<strong>所有 $x$ 分量的相位在 $y=s$ 处相长、其余处相消</strong>。它和 Deutsch-Jozsa 的区别在于：那里只关心「$y=0$ 的振幅是否为 0」，这里要读出完整的 $s$——但数学结构是同一个（Hadamard 变换的完全正交性）。</span>

## 4 公式解析：相位查询的具体作用

再往前一步，看清楚查询到底做了什么。相位查询 $O_{f_s}$ 作用在叠加态上：

$$
O_{f_s}\left( \frac{1}{\sqrt{2^n}}\sum_x \lvert x\rangle \right) = \frac{1}{\sqrt{2^n}}\sum_x (-1)^{f_s(x)} \lvert x\rangle = \frac{1}{\sqrt{2^n}}\sum_x (-1)^{s\cdot x}\lvert x\rangle
$$

- **第一步，逐项作用**：$O_{f_s}$ 对每个 $\lvert x\rangle$ 独立作用，乘上相位 $(-1)^{s\cdot x}$。
- **第二步，线性性**：酉算符是线性的，叠加态上逐项作用、相位逐项附加，得到「内积相位叠加」。
- **第三步，结构观察**：这个态与 $s$ 之间是 Walsh-Hadamard 对偶关系——查询把「$s$」的信息从「相位里」编码进「振幅里」，下一步的 $H$ 再把它解出来。<span class="marginnote">整套 BV 机制可以用一句话概括：<strong>把「隐藏串 $s$」翻译成「对偶空间里的一个相位函数」，再用一次正交变换原样读回</strong>。这套「编码进相位 → 正交变换读出」的范式，将在 Simon（周期）、Shor（模周期）里一次次复用。</span>

**辨析｜易错点：** BV 算法没有「概率失败」——它一次查询、确定成功，因为 $H^{\otimes n}$ 变换是精确的、相位编码无损。这与 Grover（有概率）、Deutsch-Jozsa 平衡情形（也确定）都不同。另一个易错点：**BV 的加速是线性的（$n \to 1$），不是指数的**——不要把它与 Deutsch-Jozsa、Simon 的指数加速混为一谈。

## 5 BV 的意义：从算法到复杂度类

BV 的价值不只是「一个更快的读串算法」，它承载了三重意义：

**BQP 的奠基**：BV 论文引入了量子多项式时间的复杂度类 **BQP**，并讨论量子图灵机的合理性。量子计算理论从此有了「自己的 P」。
**查询优势的结构证据**：BV 表明量子能在「结构已知」的问题上提供确定性线性加速，是查询模型里「量子 > 经典」的干净样例。<span class="marginnote">BQP 与经典类的精确关系至今未定：已知 $\mathrm{BPP} \subseteq \mathrm{BQP}$，但是否严格包含未知。BV 是证明「$\mathrm{BQP}$ 可能大于 $\mathrm{BPP}$」的第一块拼图。</span>
- **教学价值**：它是从 Deutsch-Jozsa（读一个性质）过渡到 Simon（读周期）的完美中继——三者共享「Hadamard + 相位 + Hadamard」骨架，复杂度逐级升高。

## 6 小结

- **BV 问题**：已知 $f_s(x) = s\cdot x \pmod 2$，读出隐藏串 $s$；经典 $n$ 次查询，量子 **1 次**。
- 线路 = **$H^{\otimes n}$ → 相位查询 → $H^{\otimes n}$ → 测量**，直接得到 $s$，确定成功。
- 核心恒等式 $H^{\otimes n}\big(\frac{1}{\sqrt{2^n}}\sum_x(-1)^{s\cdot x}\lvert x\rangle\big) = \lvert s\rangle$：内积相位在 $s$ 处相长、其余处相消。
- **线性加速**（$n \to 1$），不是指数；同时是 **BQP** 复杂度类的奠基论文。

在下一节，我们把相位编码的内积推广成「周期性」——**Simon 算法**用一次查询指数级加速地发现隐藏周期，它比 BV 更接近 Shor 算法的心脏。
