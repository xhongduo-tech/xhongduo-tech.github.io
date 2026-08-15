---
title: Weyl 特征标公式
date: 2026-08-07
---

# Weyl 特征标公式

<div class="epigraph">
<p>最好的公式，把对称群的全部动作浓缩成一个干净的商。</p>
<footer>—— 赫尔曼 · 外尔（Hermann Weyl）</footer>
</div>

<div class="article-byline">
<p>第二级 · 李代数与李群 ｜ Humphreys §22-24 ｜ 2026-08-07</p>
</div>

## 为什么需要特征标

最高权定理告诉我们「有哪些表示」，但没有告诉我们「每个表示里各权各出现多少次」。**特征标（character）**就是回答这个问题的函数：它把表示的「构成」编码成一个整数系数多项式。Weyl 特征标公式则给出这个多项式的**闭式解**——它像是表示论的「生成函数」，一切计数问题都在里面。<span class="marginnote">在量子场论与统计力学中，特征标（partition function 的推广）用来枚举态的重数；对仿射李代数，Weyl 特征标公式变成 Macdonald 恒等式与 theta 恒等式——从单条公式辐射出整片领域。</span>

为什么「知道各权出现多少次」如此重要？因为表示论里最常问的问题——**这个表示里有哪些态、每个态几重**——正是特征标的系数表。更高一层，特征标把「表示之间的运算」翻译成「多项式之间的运算」：直和变加法、张量积变乘法、对偶变指数反转。于是「分解一个复杂的表示」变成「做一次多项式运算」，而这个多项式还能被 Weyl 公式闭式算出。三条性质拼起来，特征标成了表示论的可计算引擎。

## 1 权重与特征标

设 $L$ 是复半单李代数，$V$ 是有限维表示。**形式特征标（formal character）**定义为：

$$\operatorname{ch} V = \sum_{\lambda \in \Lambda} \dim(V_\lambda)\, e^{\lambda}$$

其中 $e^{\lambda}$ 是权格 $\Lambda$ 上的形式指数（把「权 $\lambda$ 出现 $\dim V_\lambda$ 次」记成一条单项式）。<span class="marginnote">形式指数 $e^{\lambda}$ 满足 $e^{\lambda} e^{\mu} = e^{\lambda + \mu}$——这使「张量积的权谱 = 权谱的卷积」自动成立：$\operatorname{ch}(V \otimes W) = \operatorname{ch}V \cdot \operatorname{ch}W$。特征标因此把表示论运算翻译成多项式运算。</span>

**核心事实**：$\operatorname{ch} V = \operatorname{ch} W \iff V \cong W$。特征标完全决定表示（Brauer–Nesbitt 定理的雏形）。于是「算特征标」等于「完全理解表示」。

**术语速查表**：

| 术语 | 记号 | 含义 |
| --- | --- | --- |
| 权谱 | $V_\lambda$ | $V$ 的 $H$ 联合特征空间 |
| 权多重度 | $\dim V_\lambda$ | 权 $\lambda$ 在 $V$ 中出现的次数 |
| 形式特征标 | $\operatorname{ch}V = \sum \dim V_\lambda\, e^\lambda$ | 把多重度编码成多项式 |
| 形式指数 | $e^\lambda$ | 权格上的形式记号，服从 $e^\lambda e^\mu = e^{\lambda+\mu}$ |
| Weyl 向量 | $\rho$ | 正根和的一半 $\tfrac12\sum_{\alpha>0}\alpha$ |
| 长度 | $\ell(w)$ | $w$ 写成简单根反射之积的最少个数 |

**辨析｜易错点：** 形式指数 $e^{\lambda}$ 不是真正的指数函数——它是权格上的**形式记号**，只服从运算法则 $e^{\lambda}e^{\mu} = e^{\lambda+\mu}$，不能代入数值「求值」。因此特征标公式里的分子分母都是形式级数，约分、求和的合法性来自形式幂级数环而非数值分析。初学者若试图把 $e^{\lambda}$ 当实指数取值，会立刻得出荒谬结论。

## 2 Weyl 特征标公式

**Weyl 特征标公式（Weyl character formula）**：对支配整权 $\lambda$，不可约最高权表示 $L(\lambda)$ 的特征标为

$$\operatorname{ch} L(\lambda) = \frac{\sum_{w \in W} (-1)^{\ell(w)}\, e^{w(\lambda + \rho) - \rho}}{\prod_{\alpha > 0} \left( 1 - e^{-\alpha} \right)}$$

其中 $\rho = \tfrac12 \sum_{\alpha > 0} \alpha$ 是**Weyl 向量**（所有正根和的一半），$\ell(w)$ 是 $w$ 的**长度**（把 $w$ 写成简单根反射之积所需的最少反射数）。<span class="marginnote">分母是正根空间的「权谱生成函数」——因为 $L(\lambda)$ 作为 $U(n_-)$-模由负根生成，特征标必然带这个因子。分子则是分子里的 Weyl 群求和——「反对称化」把权按 $W$ 的符号对称地排列，保证结果是多项式而非级数。</span>

等价的形式（用分母 $\Delta$）:

$$\Delta = \prod_{\alpha > 0} \left( e^{\alpha/2} - e^{-\alpha/2} \right), \qquad \operatorname{ch} L(\lambda) = \frac{\sum_{w} (-1)^{\ell(w)} e^{w(\lambda+\rho)}}{\sum_{w} (-1)^{\ell(w)} e^{w(\rho)}}$$

**Weyl 维数公式（Weyl dimension formula）**是直接推论（令变量 → 1）：

$$\dim L(\lambda) = \prod_{\alpha > 0} \frac{\langle \lambda + \rho, \alpha\rangle}{\langle \rho, \alpha\rangle}$$

## 3 特征标的运算规则

特征标之所以好用，是因为它把表示论的四类基本运算全部翻译成多项式运算，且翻译规则极简：

| 表示运算 | 特征标运算 |
| --- | --- |
| 直和 $V \oplus W$ | 加法 $\operatorname{ch}V + \operatorname{ch}W$ |
| 张量积 $V \otimes W$ | 乘法 $\operatorname{ch}V \cdot \operatorname{ch}W$ |
| 对偶 $V^*$ | 反转 $\operatorname{ch}V(e^{-\lambda})$ |
| 对称/外幂 | 生成函数（Schur 多项式） |

**为什么张量积变乘法？** 因为 $V \otimes W$ 的权谱是两谱的「所有两两之和」：$(V \otimes W)_\mu = \bigoplus_{\lambda + \nu = \mu} V_\lambda \otimes W_\nu$，重数做卷积。而形式指数 $e^\lambda e^\nu = e^{\lambda+\nu}$ 恰好把这个卷积编码成乘积——这正是「形式指数」定义的精妙之处，一切都在乘法规则里。

**数值例**：对 $\mathfrak{sl}(2)$，$V_1 \otimes V_1 = V_2 \oplus V_0$（第 6 篇）。用特征标验证：$\operatorname{ch}V_1 = e + e^{-1}$，则 $\operatorname{ch}V_1^2 = e^2 + 2 + e^{-2} = \operatorname{ch}V_2 + 1$——恰为 $V_2$（权 $\pm 2, 0$）加平凡表示 $V_0$（权 $0$）。多项式乘法自动给出 Clebsch–Gordan 分解，无需逐项数态。

## 4 公式解析：用 $A_2$ 验证特征标公式

取 $\mathfrak{sl}(3,\mathbb{C})$（$A_2$ 根系），标准表示 $V = L(\lambda_1)$，其中最高权 $\lambda_1$（基本权 $\omega_1$，$\lambda_1(h_{\alpha_1})=1, \lambda_1(h_{\alpha_2})=0$）。正根：$\alpha_1, \alpha_2, \alpha_1 + \alpha_2$；$\rho = \alpha_1 + \alpha_2$。Weyl 群 $W \cong S_3$，六个元素。

- **第一步，读分子**：$\sum_w (-1)^{\ell(w)} e^{w(\lambda_1 + \rho)} = \sum_w (-1)^{\ell(w)} e^{w(2\alpha_1 + \alpha_2)}$。$S_3$ 六个元素分别把 $2\alpha_1 + \alpha_2$ 送成六个权：$2\alpha_1 + \alpha_2, 2\alpha_1 - \alpha_2, \alpha_2 - \alpha_1, \alpha_1 - \alpha_2, \alpha_1 + \alpha_2$（含符号翻转），排列为反对称和。
- **第二步，读分母**：$\prod_{\alpha>0}(1 - e^{-\alpha}) = (1 - e^{-\alpha_1})(1 - e^{-\alpha_2})(1 - e^{-\alpha_1 - \alpha_2})$。
- **第三步，约分**：分子分母的公共因子约去后，得到一个有限多项式：$\operatorname{ch}V = e^{\alpha_1} + e^{\alpha_2} + e^{-\alpha_1 - \alpha_2} + e^{-\alpha_1} + e^{-\alpha_2} + e^{\alpha_1 + \alpha_2}$（六个权各一维）——这正是 $\mathfrak{sl}(3)$ 标准表示的三维 + 三维共轭表示的结构。<span class="marginnote">这个六项多项式说明：标准表示有六个权，每个权一维。用 $\operatorname{ch}$ 的语言，张量积 $V \otimes V^*$ 的分解等一切运算都变成多项式乘除——表示论变成了「系数计数」。</span>

**核心要点**：Weyl 特征标公式的精髓是「用 Weyl 群反对称化消除分母」：分子（反对称和）与分母（正根积）的公共因子自动抵消，剩下的就是干净的多项式。它把无穷级数的比值变成有限和——这就是表示论可计算性的保证。<span class="marginnote">对 $\mathfrak{sl}(2)$，公式退化为 $e^{\lambda}$ 的等比求和，$V_n$ 的特征标是 $\sum_{k=0}^n e^{n-2k}$，与第 6 篇的权谱完全一致——读者可作为自检。</span>

## 5 Weyl 维数公式与多重度

维数公式给出一个可计算的整数：

$$\dim L(\lambda) = \prod_{\alpha > 0} \frac{\langle \lambda + \rho, \alpha\rangle}{\langle \rho, \alpha\rangle}$$

对 $\lambda = \omega_1$（$A_2$，标准三维表示的最高权）代入，取标准归一化 $\langle \alpha_1, \alpha_1\rangle = \langle\alpha_2,\alpha_2\rangle = 2$、$\langle\alpha_1,\alpha_2\rangle = -1$。先算分母：$\langle\rho,\alpha_1\rangle = 1$，$\langle\rho,\alpha_2\rangle = 1$，$\langle\rho,\alpha_1+\alpha_2\rangle = 2$。再算分子 $\lambda + \rho = \omega_1 + \alpha_1 + \alpha_2$：

- 对 $\alpha_1$：$\langle \omega_1+\rho, \alpha_1\rangle = 1 + 1 = 2$，比值为 $2/1 = 2$；
- 对 $\alpha_2$：$\langle \omega_1+\rho, \alpha_2\rangle = 0 + 1 = 1$，比值为 $1$；
- 对 $\alpha_1+\alpha_2$：$\langle \omega_1+\rho, \alpha_1+\alpha_2\rangle = 1 + 2 = 3$，比值为 $3/2$。

三者之积 $2 \times 1 \times \tfrac32 = 3$——恰为标准三维表示。<span class="marginnote">维数公式的比值不依赖内积归一化（分子分母同变），但中间数字依赖。初学者算错维数，十有八九是混用了不同教材的 $\langle\cdot,\cdot\rangle$ 标定——锁定同一套再算。</span>

**再算一个更高权的例子**：取 $\lambda = 2\omega_1$（对称平方表示）。$\lambda + \rho = 2\omega_1 + \alpha_1 + \alpha_2$，逐个正根：

- 对 $\alpha_1$：$\langle 2\omega_1+\rho, \alpha_1\rangle = 2 + 1 = 3$，比值 $3$；
- 对 $\alpha_2$：$\langle 2\omega_1+\rho, \alpha_2\rangle = 0 + 1 = 1$，比值 $1$；
- 对 $\alpha_1+\alpha_2$：$\langle 2\omega_1+\rho, \alpha_1+\alpha_2\rangle = 2 + 2 = 4$，比值 $2$。

积 $3 \times 1 \times 2 = 6$——正是 $\operatorname{Sym}^2(\mathbb{C}^3)$ 的维数 $\binom{3+2-1}{2} = 6$。两条独立路线（Weyl 公式与组合计数）给出同一答案，这是对公式可靠性的极佳自检：**维数公式不只是漂亮，它算得准**。

**多重度**：权谱内部的多重度（一个权在不可约表示里出现几次）由 Kostant 重数公式或 Freudenthal 递推给出，它们都是 Weyl 特征标公式的推论。粗略说，重数 = 分子分母在单项式层面的「系数比」——特征标公式把「求重数」这个看似困难的问题也归约成了多项式除法。

## 6 小结

- **形式特征标** $\operatorname{ch}V = \sum \dim V_\lambda\, e^{\lambda}$：表示 ⟺ 多项式，张量积 ⟺ 乘法。
- **Weyl 特征标公式**：$\operatorname{ch}L(\lambda) = \sum_w (-1)^{\ell(w)} e^{w(\lambda+\rho)} / \Delta$，把无穷级数比值化为有限和。
- **Weyl 向量** $\rho$ 与 **Weyl 群** 的反对称化是公式的心脏；分母是正根生成函数。
- **Weyl 维数公式** $\dim L(\lambda) = \prod_{\alpha>0} \langle\lambda+\rho,\alpha\rangle/\langle\rho,\alpha\rangle$：注意内积归一化一致性。
- 特征标运算规则：直和变加法、张量积变乘法、对偶变反转——多项式乘法自动给出 Clebsch–Gordan 分解。
- $\dim L(2\omega_1) = 6$ 与 $\operatorname{Sym}^2(\mathbb{C}^3)$ 的组合计数吻合，是公式可靠性的自检。
- 权多重度由 Kostant 公式/Freudenthal 递推给出，都是 Weyl 公式的推论。
- 公式对 $\mathfrak{sl}(2)$ 退化为等比求和，是自检的标准案例。

在下一节，我们将回到更具体的舞台：**SU(2) 与 SU(3) 的表示**——看这些抽象公式如何在物理中最常出现的紧李群上落地。
