---
title: CHSH 不等式及其量子违背
date: 2026-08-07
---

# CHSH 不等式及其量子违背

<div class="epigraph">
<p>任何定域隐变量理论都不可能复现量子力学的全部预测。</p>
<footer>—— 约翰 · 贝尔（John Bell）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen &amp; Chuang《量子计算与量子信息》§2.6 ｜ 2026-08-07</p>
</div>

## 为什么从 CHSH 不等式开始

上一节的贝尔不等式是开创性的，但它只适用于特定的测量设置，实验上难以直接操作。1969 年，Clauser、Horne、Shimony 与 Holt 给出了一个**实用化版本**——CHSH 不等式。它把「定域隐变量 vs 量子力学」的对立压缩成一条简洁的数值界线：**定域隐变量 ≤ 2，量子力学可达 $2\sqrt2$**。<span class="marginnote">CHSH 是四个作者姓氏的首字母：Clauser、Horne、Shimony、Holt（1969 年发表于 <i>Phys. Rev. Lett.</i> 23, 880）。2022 年诺贝尔物理学奖颁给了 Aspect、Clauser、Zeilinger，表彰他们用贝尔不等式实验确立了量子纠缠的实在性。</span>今天的量子信息实验中，CHSH 几乎是检验纠缠的「标配测试」，也是量子密码学 device-independent 协议的核心工具。

## 1 实验设置与关联量

设想 Alice 与 Bob 共享一个两比特纠缠态。Alice 可以从两个测量设置 $a, a'$ 中选一个，Bob 从 $b, b'$ 中选一个；每个测量的输出都是二值的，记作 $\pm1$。定义**关联期望值**

$$
\langle A_a B_b \rangle = \sum_{x,y \in \{\pm1\}} xy \, P(x, y \mid a, b)
$$

这是「Alice 与 Bob 结果乘积的平均」——同号贡献 +1，异号贡献 −1。CHSH 考察四个设置的组合

$$
S = \langle A_a B_b \rangle + \langle A_a B_{b'} \rangle + \langle A_{a'} B_b \rangle - \langle A_{a'} B_{b'} \rangle
$$

**定域隐变量理论保证 $\lvert S \rvert \le 2$，量子力学可达到 $2\sqrt2$。**<span class="marginnote">注意 $S$ 的定义里最后一项是减号。这个「三项加、一项减」的非对称组合正是 CHSH 的巧妙之处：它让定域隐变量的最大值恰好是 2，而量子关联能把它顶破。</span>

## 2 定域隐变量为什么 ≤ 2

关键假设是：测量结果由预先存在的隐变量 $\lambda$ 决定，且 Alice 的结果与 Bob 的设置无关（**定域性**）。于是

$$
A_a(\lambda), A_{a'}(\lambda), B_b(\lambda), B_{b'}(\lambda) \in \{\pm1\}, \qquad S(\lambda) = A_a B_b + A_a B_{b'} + A_{a'} B_b - A_{a'} B_{b'}
$$

对每个固定的 $\lambda$，四个数 $\pm1$ 代入 $S(\lambda)$ 只可能取 $+2$ 或 $-2$（逐项验证：$A_a(B_b + B_{b'}) + A_{a'}(B_b - B_{b'})$，其中 $B_b + B_{b'}$ 为 $\pm2$ 或 0，$B_b - B_{b'}$ 为 0 或 $\pm2$，两者不会同时非零）。<span class="marginnote">这一步是纯代数的：先固定 $\lambda$，把 $S$ 写成 $A_a(B_b+B_{b'}) + A_{a'}(B_b-B_{b'})$，因为 $B_b, B_{b'} \in \{\pm1\}$，两括号一为 $\pm2$ 一为 0，所以 $S(\lambda) = \pm2$。</span>再对 $\lambda$ 的分布取平均，$\lvert S\rvert = \lvert \int S(\lambda)p(\lambda)d\lambda \rvert \le \int \lvert S(\lambda)\rvert p(\lambda)d\lambda = 2$。

## 3 量子违背：算到 $2\sqrt2$

现在用量子力学算同一个 $S$。取贝尔态 $\lvert\Phi^-\rangle = \frac{1}{\sqrt2}(\lvert00\rangle - \lvert11\rangle)$，并选如下测量方向（把输出 $\pm1$ 编码为沿某个轴的自旋）：

$$
a: Z, \quad a': X, \quad b: \frac{-Z-X}{\sqrt2}, \quad b': \frac{Z-X}{\sqrt2}
$$

- **第一步，算关联**：由量子力学，$\langle A_a B_b\rangle = -\cos\theta_{ab}$（其中 $\theta_{ab}$ 是 $a$ 与 $b$ 两个方向的夹角）。这四个夹角分别算出来：$\theta_{ab} = \frac{3\pi}{4}$、$\theta_{ab'} = \frac{\pi}{4}$、$\theta_{a'b} = \frac{\pi}{4}$、$\theta_{a'b'} = \frac{3\pi}{4}$。
- **第二步，代入 $S$**：$\cos\frac{3\pi}{4} = -\frac{\sqrt2}{2}$，$\cos\frac{\pi}{4} = \frac{\sqrt2}{2}$。于是
$$
S = -(-\tfrac{\sqrt2}{2}) - (\tfrac{\sqrt2}{2}) - (\tfrac{\sqrt2}{2}) + (-\tfrac{\sqrt2}{2}) = \frac{\sqrt2}{2}+\frac{\sqrt2}{2}+\frac{\sqrt2}{2}+\frac{\sqrt2}{2} = 2\sqrt2
$$
- **第三步，结论**：$2\sqrt2 > 2$，量子力学与任何定域隐变量理论矛盾。<span class="marginnote">几何直觉：四个测量方向在单位球面上构成两个「菱形」，量子关联的余弦项让它们在 $S$ 里同向叠加。$2\sqrt2$ 也是 Tsirelson 界——量子理论允许的最大 CHSH 值，再大就违背量子本身。</span>

**辨析｜易错点：** 量子违背 CHSH **不**意味着测量前结果「还没确定、测了才确定」就完事了。要点是：对每个固定 $\lambda$，$S(\lambda)=\pm2$；量子力学预测的 $S=2\sqrt2 > 2$ 意味着**不存在能同时为四个设置分配确定结果的 $\lambda$**——「结果由 $\lambda$ 预先决定」这个前提本身就站不住。

## 4 公式解析：为什么方向组合能顶破 2

把「夹角余弦」与「$S$ 构造」拼在一起看：

$$
S = -\cos\theta_{ab} - \cos\theta_{ab'} - \cos\theta_{a'b} + \cos\theta_{a'b'}
$$

- **第一步，构造正弦差**：利用恒等式 $-\cos\theta_{ab} - \cos\theta_{a'b'} = -2\cos\frac{\theta_{ab}+\theta_{a'b'}}{2}\cos\frac{\theta_{ab}-\theta_{a'b'}}{2}$ 一类的化简，把四项归并。
- **第二步，几何选取**：让四个方向在球面上满足「夹角互补」，使两个余弦组合同号叠加。上例中前三个余弦都是 $-\frac{\sqrt2}{2}$、最后一个 $+\frac{\sqrt2}{2}$，四项绝对值相同、符号方向统一。
- **第三步，整体缩放**：四个 $\pm\frac{\sqrt2}{2}$ 加起来 = $2\sqrt2$。若夹角选得「均匀互补」，量子上限 $2\sqrt2$ 就被精确顶到。<span class="marginnote">一句话记忆：定域隐变量的 $S$ 是「单 $\lambda$ 下 $\pm2$ 的平均」，而量子的 $S$ 是「纠缠态四个关联的相干求和」——相干性让四个余弦项不必逐项抵消，反而同向叠加。</span>

## 5 CHSH 的当代角色

CHSH 不只是教科书定理，它是量子技术的一线工具：

- **纠缠验证**：实验中测到 $S > 2$ 即宣告「纠缠真实存在、非定域」，这是量子设备的出厂检验。
- **Device-independent 量子密码**：如果协议的安全性只依赖「$S$ 超过某个阈值」而不是对设备内部建模，就得到不信任设备的密码协议——抵抗侧信道攻击的天然后门。
- **量子优势的度量**：$S$ 的大小可当作「纠缠强度」的标尺，与第八篇的纠缠度量衔接。<span class="marginnote">2022 年诺奖实验普遍做到 $S > 2$ 而同时关闭「定域性漏洞」「自由选择漏洞」——量子力学与定域实在论的决斗，在实验上已尘埃落定。</span>

## 6 小结

- **CHSH 不等式**：$\lvert S\rvert \le 2$ 对所有定域隐变量理论成立，其中 $S = \langle A_a B_b\rangle + \langle A_a B_{b'}\rangle + \langle A_{a'} B_b\rangle - \langle A_{a'} B_{b'}\rangle$。
- 定域性 + 预定性 → 每个 $\lambda$ 给出 $S(\lambda)=\pm2$，平均后 $\le 2$。
- **量子违背**：贝尔态 + 特定方向组合给出 $S = 2\sqrt2$，顶破 2；$2\sqrt2$ 是 Tsirelson 界。
- 应用：纠缠验证、device-independent 量子密码、纠缠度量标尺。

在下一节，我们把「纠缠」从定性变成定量：如何给一个纠缠态打一个分数？这就是**纠缠的度量：并发度（concurrence）与纠缠熵**。
