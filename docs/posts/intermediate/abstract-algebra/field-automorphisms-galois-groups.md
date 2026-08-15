---
title: 域的自同构与 Galois 群
date: 2026-08-07
---

# 域的自同构与 Galois 群

<div class="epigraph">
<p>方程的对称性，藏在分裂域的自同构里——Galois 群把「解方程」翻译成「数对称」。</p>
<footer>—— 自 题（Galois 群笔记）</footer>
</div>

<div class="article-byline">
<p>第二级 · 抽象代数 ｜ 杨子胥《近世代数》§11.1 ｜ 2026-08-07</p>
</div>

## 为什么从域的自同构与 Galois 群开始

第十篇末我们看到：有限域 $\mathbb{F}_{p^n}$ 的自同构群（Galois 群）是 $n$ 阶循环群。现在把这个概念推广到一般扩张：**Galois 群** $\operatorname{Gal}(E/F)$ = 保持 $F$ 不变（固定 $F$ 逐点）的 $E$ 的自同构群。

Galois 群是整个 Galois 理论的主角：一个方程的可解性、分裂域的子域结构、尺规作图的可行性——全部由 Galois 群决定。伽罗瓦的伟大洞见：**「方程 $f(x) = 0$ 的解」的代数信息，完整编码在「$f$ 的分裂域的对称（Galois 群）」里。** 本节把自同构与 Galois 群的定义、基本性质与「根的置换」视角讲透。

## 1 域自同构的定义

**域自同构（field automorphism）**：设 $E$ 是域，双射 $\sigma : E \to E$ 若保持加法和乘法（$\sigma(a+b) = \sigma(a)+\sigma(b)$、$\sigma(ab) = \sigma(a)\sigma(b)$），则称 $\sigma$ 是 $E$ 的自同构。全体自同构成群，记作 $\operatorname{Aut}(E)$。

**自同构自动保持结构**：
- $\sigma(0) = 0$、$\sigma(1) = 1$；
- $\sigma(-a) = -\sigma(a)$、$\sigma(a^{-1}) = \sigma(a)^{-1}$（$a \ne 0$）；
- $\sigma(n \cdot 1) = n \cdot 1$（固定素域的元素）；
- 若 $f(a) = 0$（$f$ 系数在素域），则 $f(\sigma(a)) = 0$——**自同构把根送到根**。

**定理（自同构固定素域）：** 域 $E$ 的任何自同构固定其素域（$\mathbb{Q}$ 或 $\mathbb{F}_p$）逐点。<span class="marginnote">「自同构固定素域」是域自同构的第一约束：$\sigma(1) = 1$ 推出 $\sigma(n \cdot 1) = n \cdot 1$、$\sigma(\frac{m}{n}) = \frac{\sigma(m)}{\sigma(n)}$，所以 $\mathbb{Q}$（或 $\mathbb{F}_p$）的元素全被钉死。<strong>自同构只能「移动」素域之外的元素</strong>——这正是「扩张部分」才有对称可言的根源。</span>

**例：**
- $\operatorname{Aut}(\mathbb{Q}) = \{ \mathrm{id} \}$（素域无对称）；
- $\operatorname{Aut}(\mathbb{C})$（保持 $\mathbb{R}$ 的）$= \{ \mathrm{id}, \text{共轭} \}$（$a + bi \mapsto a \pm bi$，若要求保 $\mathbb{R}$；否则巨大）；
- $\operatorname{Aut}(\mathbb{Q}(\sqrt2)) = \{ \mathrm{id}, \sigma \}$，$\sigma(a + b\sqrt2) = a - b\sqrt2$（把 $\sqrt2$ 送 $\sqrt2$ 或 $-\sqrt2$）。

## 2 Galois 群的定义

**Galois 群（Galois group）**：设 $E/F$ 是域扩张，保持 $F$ 逐点不变的 $E$ 的自同构全体

$$
\operatorname{Gal}(E/F) = \{ \sigma \in \operatorname{Aut}(E) \mid \sigma(a) = a \ \forall a \in F \}
$$

构成群（复合运算），称为扩张 $E/F$ 的 **Galois 群**。

**Galois 群的作用**：若 $E$ 是 $f \in F[x]$ 的分裂域，则 $\operatorname{Gal}(E/F)$ 作用在 $f$ 的根集合上：$\sigma(\alpha_i) = \alpha_j$（根仍为根）。这个作用给出单射

$$
\operatorname{Gal}(E/F) \longrightarrow S_n, \qquad n = \deg f
$$

——**Galois 群嵌入根的置换群**。$E$ 的每个元素都被「根如何置换」决定（$E = F(\alpha_1, \dots, \alpha_n)$，$\sigma$ 由它在根上的作用确定）。<span class="marginnote">「$\sigma$ 由它在根上的作用完全决定」是 Galois 理论的关键观察：因为 $E = F(\alpha_1,\dots,\alpha_n)$ 且每个 $\alpha_i$ 由 $f$ 的多项式关系绑定，$\sigma$ 的像由根的像唯一确定。于是研究 Galois 群 = 研究「根集合上的合法置换」。<strong>「解方程的对称性」=「根的置换群」</strong>——这是伽罗瓦把代数问题翻译成群论问题的核心。</span>

**例：**
$\operatorname{Gal}(\mathbb{C}/\mathbb{R}) \cong \mathbb{Z}_2$（$\{ \mathrm{id}, \text{共轭} \}$），作用在 $x^2+1$ 的根 $\{i, -i\}$ 上为 $S_2$；
$\operatorname{Gal}(\mathbb{Q}(\sqrt2)/\mathbb{Q}) \cong \mathbb{Z}_2$（$\sqrt2 \mapsto \pm\sqrt2$）；
$\operatorname{Gal}(\mathbb{F}_{p^n}/\mathbb{F}_p) \cong \mathbb{Z}_n$（Frobenius 生成，第十篇）。

## 3 分裂域的 Galois 群：x^3 - 2 的完整计算

把 Galois 群在 $x^3 - 2$ 的分裂域上完整算一遍，感受「根的置换」如何生成群。

**例：$f = x^3 - 2$ 在 $\mathbb{Q}$ 上，分裂域 $E = \mathbb{Q}(\sqrt[3]2, \omega)$**（$\omega$ 本原三次单位根），根 $\alpha_1 = \sqrt[3]2$、$\alpha_2 = \omega\sqrt[3]2$、$\alpha_3 = \omega^2\sqrt[3]2$。

$[E : \mathbb{Q}] = 6$（$x^3-2$ 次数 3、$\omega$ 的 $x^2+x+1$ 次数 2）；
Galois 群大小 $= [E:\mathbb{Q}] = 6$（Galois 扩张，下节定理），故 $\operatorname{Gal}(E/\mathbb{Q})$ 是 $S_3$ 的 6 阶子群 = $S_3$ 本身；
**置换的实况**：$\sigma \in \operatorname{Gal}$ 必须把根送到根。$\omega = \frac{\alpha_2}{\alpha_1}$，$\sigma(\omega)$ 由 $\sigma(\alpha_i)$ 决定。可以验证：任意根的置换都合法（因为 $\mathbb{Q}$ 上 $x^3-2$ 的判别式为负、无二次子域约束），$\operatorname{Gal} \cong S_3$。<span class="marginnote">$x^3 - 2$ 的 Galois 群是 $S_3$（全部 6 个置换），因为它的分裂域「足够大」（次数 $6 = 3!$）。$S_3$ 可解（有正规子群 $A_3$ 且 $A_3$、$S_3/A_3$ 都循环），所以 $x^3 - 2$ 可根式求解——事实也如此（$\sqrt[3]2$ 是根式）。<strong>「Galois 群是否可解」与「方程可否根式求解」的对应，即将在下下节展开。</strong></span>

## 4 公式解析：|Gal(E/F)| = [E:F]（对 Galois 扩张）

Galois 群的大小与扩张次数的关系，是 Galois 理论最重要的「大小恒等式」。

**定理：** 若 $E/F$ 是有限 Galois 扩张（可分正规，即某个可分多项式的分裂域），则

$$
| \operatorname{Gal}(E/F) | = [E : F]
$$

**证明（思路）：** 用「自同构个数 = 扩张次数」的逐步论证：
- **第一步（单代数扩张）**：$E = F(\alpha)$，$\alpha$ 的最小多项式 $m$ 次数 $n$，根 $\alpha = \alpha_1, \alpha_2, \dots, \alpha_n$（可分 ⟹ $n$ 个互异根）。保持 $F$ 的自同构 $\sigma$ 由 $\sigma(\alpha) = \alpha_i$ 决定（$n$ 种选择），故 $|\operatorname{Gal}| = n = [E:F]$。
- **第二步（一般分裂域）**：$E = F(\alpha_1, \dots, \alpha_k)$，逐步添加根；每步「$\alpha_i$ 有 $[\text{下一步域} : \text{上一步域}]$ 个像」（可分保证根互异），乘积等于乘法塔。$\blacksquare$

**第四步，直觉**：可分性保证「每个根的像有足够多选择」（根互异），正规性保证「选择不跑出 $E$」，两者合起来让「自同构个数 = 维度」。**这个恒等式是 Galois 对应的「大小引擎」**——下一节的对应定理将把「子群与子域」按大小一一挂钩。<span class="marginnote">「$|\operatorname{Gal}(E/F)| = [E:F]$」在可分正规扩张里成立，缺任何一条都会失败：$\mathbb{Q}(\sqrt[3]2)/\mathbb{Q}$ 不是正规，其自同构只有 $\mathrm{id}$（$|\operatorname{Gal}| = 1 \ne 3 = [E:F]$）；$\mathbb{F}_p(t^{1/p})/\mathbb{F}_p(t)$ 不可分，自同构也少于扩张次数。<strong>Galois 扩张是「对称刚好等于维度」的扩张。</strong></span>

## 5 例子：小分裂域的 Galois 群

把几个经典分裂域的 Galois 群算出来，建立直觉库。

| 分裂域（$f$） | Galois 群 | 说明 |
| --- | --- | --- |
| $x^2 - 2$ / $\mathbb{Q}$ | $\mathbb{Z}_2$ | 根 $\pm\sqrt2$ 互换 |
| $x^2 + 1$ / $\mathbb{Q}$ | $\mathbb{Z}_2$ | 共轭 $i \mapsto -i$ |
| $x^3 - 2$ / $\mathbb{Q}$ | $S_3$ | 全部 6 个置换 |
| $x^4 - 4x^2 + 2$ / $\mathbb{Q}$ | $\mathbb{Z}_2 \times \mathbb{Z}_2$ | 根 $\pm\sqrt{2\pm\sqrt2}$，置换受限 |
| $x^{p^n} - x$ / $\mathbb{F}_p$ | $\mathbb{Z}_n$ | Frobenius 生成 |

**观察**：Galois 群从 $\mathbb{Z}_2$（两根互换）到 $S_3$（三根全置换）到 $\mathbb{Z}_2\times\mathbb{Z}_2$（受限的双重互换），刻画了「根的自由程度」。$x^4 - 4x^2 + 2$ 的根虽是四个，但置换只有 4 种而非 $4! = 24$ 种——**根的代数关系约束了对称**，这正是「方程可解性」的度量。<span class="marginnote">「根的代数关系约束对称」是 Galois 理论的最深直觉：根之间若有额外关系（如 $\alpha_1\alpha_2 = \alpha_3$），置换必须保持它，Galois 群就变小。$x^4 - 4x^2 + 2$ 的根成对 $\pm$ 且满足二次套二次的关系，Galois 群被压成 $\mathbb{Z}_2 \times \mathbb{Z}_2$——这也是它可根式求解的原因（每个 $\mathbb{Z}_2$ 对应一次开平方）。</span>

## 6 对照速查：自同构的个数与扩张次数

把「自同构个数 vs 扩张次数」的三种情形排成一张表，Galois 扩张的边界一目了然。

| 扩张 | 自同构个数 $|\operatorname{Gal}|$ | $[E:F]$ | 相等？ |
| --- | --- | --- | --- |
| $\mathbb{Q}(\sqrt2)/\mathbb{Q}$（Galois） | 2 | 2 | ✓ |
| $\mathbb{Q}(\sqrt[3]2)/\mathbb{Q}$（非正规） | 1 | 3 | ✗ |
| $\mathbb{F}_p(t^{1/p})/\mathbb{F}_p(t)$（不可分） | 1 | $p$ | ✗ |

**数值算例：为什么 $\mathbb{Q}(\sqrt[3]2)/\mathbb{Q}$ 的自同构只有恒等。** 保持 $\mathbb{Q}$ 的自同构 $\sigma$ 必须把 $\sqrt[3]2$ 送到 $x^3 - 2$ 的根，即 $\sqrt[3]2, \omega\sqrt[3]2, \omega^2\sqrt[3]2$。但后两个不在 $\mathbb{Q}(\sqrt[3]2)$ 里（含 $\omega$），所以 $\sigma(\sqrt[3]2)$ 只能是 $\sqrt[3]2$，$\sigma = \mathrm{id}$。$|\operatorname{Gal}| = 1 \ne 3 = [E:F]$——<strong>非正规 ⟹ 自同构少于维度</strong>。<span class="marginnote">「自同构把根送到根」在非正规扩张里立刻失效：$x^3 - 2$ 的另外两个根不回家，$\sigma$ 没有可送的对象。<strong>正规性（根全回来）是「自同构够多」的前提</strong>，可分性（根互异）是「每个根都有独立像」的前提——两者合起来才让 $|\operatorname{Gal}| = [E:F]$。Galois 扩张正是「对称刚好装满维度」的扩张。</span>

**易错辨析｜$\operatorname{Gal}(E/F)$ 是子群不是集合。** $\operatorname{Gal}(E/F) \le \operatorname{Aut}(E)$ 是自同构群下的子群（复合封闭、含恒等、闭逆），不是随便一批自同构。判定「$\sigma$ 是否属于 Galois 群」要看「是否固定 $F$ 逐点」与「是否自同构」两条，缺一不可。

**一句话记法**：Galois 群 = 固定基域的自同构群；自同构把根送到根；$|\operatorname{Gal}(E/F)| = [E:F]$ 只在 Galois 扩张成立——对称刚好等于维度的扩张。

## 7 小结

- **域自同构**：保加乘的双射；自动固定素域；把根送到根。
- **Galois 群** $\operatorname{Gal}(E/F)$：保持 $F$ 的自同构群；$E$ 是 $f$ 的分裂域时嵌入 $S_n$。
- **根的作用**：$\sigma$ 由根上的作用决定；研究 Galois 群 = 研究合法根置换。
- **大小恒等式**：Galois 扩张时 $|\operatorname{Gal}(E/F)| = [E:F]$。
- **例**：$x^3-2$ 的 Galois 群是 $S_3$；$x^4-4x^2+2$ 的是 $\mathbb{Z}_2\times\mathbb{Z}_2$；有限域的是 $\mathbb{Z}_n$。

在下一节，我们建立 Galois 理论的核心对应：**伽罗瓦对应：子群与子域**。Galois 群的中子群与分裂域的中子域一一对应，这是「对称决定结构」的精确形态。
