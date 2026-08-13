---
title: 超限递归与序数算术
date: 2026-08-07
---

# 超限递归与序数算术

<div class="epigraph">
<p>无穷看起来是一道坚固的墙，可超限归纳告诉我们：它可以被一步一步地走过去。</p>
<footer>—— 汉斯 · 哈恩（Hans Hahn）</footer>
</div>

<div class="article-byline">
<p>第二级 · 公理集合论与模型论 ｜ Jech, <em>Set Theory</em> 第2章；Kunen 第2章 ｜ 2026-08-07</p>
</div>

## 为什么从超限递归开始

上一篇我们得到了序数，也隐约用到了一件事：**把良序集逐段数过去**。这个「逐段数过去」不是随便的措辞，而是一条严格的定理——**超限递归（transfinite recursion）**。它回答的问题是：当我们要按序数一步一步定义对象（加法、乘法、幂、$V_\alpha$、可构造层级 $L_\alpha$……）时，「每一步都只依赖前面所有的步」凭什么合法？<span class="marginnote">普通归纳法只覆盖 $\mathbb{N}$，超限递归把同一套思想推广到一切序数：分「零、后继、极限」三种情形走，极限处取并。第3篇力迫法里几乎所有构造都靠它立身。</span>

今天我们把序数的运算真正装进机器。加法与乘法都不是「一行定义」，而是**用超限递归逐段刻出来的**；它们因此长出了与自然数算术截然不同的怪癖——交换律失效，$1 + \omega \neq \omega + 1$。理解这些「怪癖」不是刁难，而是理解良序结构的关键：**序数算术编码的是「排列的形状」，不是「数量的多少」**。

## 1 超限归纳与超限递归

先看证明用的工具。**超限归纳原理（transfinite induction）**：设 $C$ 是序数的一个性质，若

- 基础：$C(0)$ 成立；
- 后继步：$C(\alpha) \Rightarrow C(\alpha+1)$；
- 极限步：对所有 $\alpha \lt  \lambda$ 有 $C(\alpha)$（$\lambda$ 为极限序数）$\Rightarrow C(\lambda)$；

则 $C(\alpha)$ 对所有序数成立。证明只需一句：若 $C$ 不普遍成立，取最小的 $\alpha$ 使 $\lnot C(\alpha)$，矛盾。

再看定义用的工具。**超限递归定理（transfinite recursion）**：若 $F$ 是一个「把序数之前的结果映射到下一步」的函数（对每个序数 $\alpha$，$F$ 在「一切 $\lt  \alpha$ 处的取值」上有定义），则存在唯一序列（类函数）$(a_\alpha)_{\alpha \in \mathrm{On}}$ 满足

$$
a_\alpha = F\bigl( (a_\xi)_{\xi \lt  \alpha} \bigr)
$$

也就是说，$a_\alpha$ 的值完全由它之前的所有值决定。定理本身由替换公理 + 归纳证明：任何 $\alpha$ 之前的过程都能在 $\alpha$ 处收束成一个集合，再把它们拼成 $\mathrm{On}$ 上的类函数。<span class="marginnote">递归与归纳是同一个硬币的两面：归纳是「证明性质对所有序数成立」，递归是「定义对象对所有序数成立」。递归定理的安全性依赖替换公理——它保证「把定义域是一段序数的对象收集起来」仍是集合，不会偷渡成真类。</span>

**辨析｜易错点：** 初学者常以为「递归 = 用一个公式直接写出 $a_n$」。超限递归比这更根本：它允许**极限处**的定义使用前面无穷多项的信息（如并集 $\bigcup_{\xi\lt \lambda} a_\xi$）。这正是 $\omega + \omega = \bigcup_{n\lt \omega} (\omega + n)$ 这类等式能成立的原因，普通归纳法做不到这一点。

## 2 序数加法与乘法：按良序的形状来定义

序数加法用超限递归定义：对任意序数 $\alpha$，定义 $\alpha + \beta$ 为

$$
\alpha + 0 = \alpha, \qquad \alpha + (\beta+1) = (\alpha + \beta) + 1, \qquad \alpha + \lambda = \sup_{\beta \lt  \lambda} (\alpha + \beta)
$$

其中 $\lambda$ 为极限序数，$+1$ 是后继 $\cup \{\cdot\}$，$\sup$ 在这里就是取并集。**直观**：$\alpha + \beta$ 是「先放一段长为 $\alpha$ 的良序，再接一段长为 $\beta$ 的良序」这个拼接结果的序型。<span class="marginnote">两个良序首位相接仍是良序；$\alpha + \beta$ 就是它的序型。拼接的方向不可颠倒——「先 $\alpha$ 后 $\beta$」与「先 $\beta$ 后 $\alpha$」形状不同，这正是交换律失效的根源。</span>

序数乘法同样递归定义：

$$
\alpha \cdot 0 = 0, \qquad \alpha \cdot (\beta+1) = \alpha \cdot \beta + \alpha, \qquad \alpha \cdot \lambda = \sup_{\beta\lt \lambda} \alpha \cdot \beta
$$

**直观**：$\alpha \cdot \beta$ 是「$\beta$ 份长度为 $\alpha$ 的段，按 $\beta$ 的顺序排好」——即 $\beta$ 个副本 $\alpha$ 依次相接，而不是「$\alpha$ 个 $\beta$」。所以 $\alpha \cdot \beta$ 通常不等于 $\beta \cdot \alpha$。

由此立刻看到两个反直觉的等式：

- $1 + \omega = \sup_{n\lt \omega}(1+n) = \omega$，而 $\omega + 1 > \omega$，故 $1 + \omega = \omega \neq \omega + 1$。**加法左单位仍成立，右单位失效**。
- $2 \cdot \omega = \sup_{n\lt \omega} 2n = \omega$，而 $\omega \cdot 2 = \omega + \omega > \omega$，故 $2 \cdot \omega = \omega \neq \omega \cdot 2$。

**要点**：序数运算把「加/乘」翻译成「良序拼接」。在拼接线段的直觉里，这些等式一目了然：无穷长线段前面补一段有限线段，还是无穷长；但先走有限步再走向无穷，与先到无穷再继续，是两种完全不同的行程。

## 3 序数幂与康托尔范式

序数幂递归定义：

$$
\alpha^0 = 1, \qquad \alpha^{\beta+1} = \alpha^\beta \cdot \alpha, \qquad \alpha^\lambda = \sup_{\beta\lt \lambda} \alpha^\beta \;(\lambda \text{ 极限})
$$

例如 $2^\omega = \sup_n 2^n = \omega$（注意这里不是「$2$ 的 $\omega$ 次方有多大」，而是「把所有 $2^n$ 的最小上界」）。序数幂同样是拼接直觉的产物，与基数幂完全不同——$2^{\aleph_0}$ 那个量级的故事属于第3篇。

**康托尔范式（Cantor normal form）** 断言：任何非零序数 $\alpha$ 都能唯一写成

$$
\alpha = \omega^{\beta_1} \cdot k_1 + \omega^{\beta_2} \cdot k_2 + \cdots + \omega^{\beta_n} \cdot k_n
$$

其中 $\beta_1 > \beta_2 > \cdots > \beta_n$ 是序数，$k_i$ 是正整数。<span class="marginnote">康托尔范式是「以 $\omega$ 为底的进位制展开」：序数像多项式一样被唯一拆解。它让序数变得可以「计算」——比较大小、判定等式都归结为比较这个范式的字典序。正则基数一章会用它处理 $\epsilon$ 数等特殊序数。</span>例如 $\omega^2 \cdot 3 + \omega \cdot 5 + 7$ 就是范式；而 $\omega^\omega$ 的范式是它自己。由范式可证一个漂亮结论：**对每个 $\alpha > 0$，都存在序数 $\epsilon$（$\epsilon$-数）满足 $\epsilon = \omega^\epsilon$**，最小的记作 $\epsilon_0 = \sup\{\omega, \omega^\omega, \omega^{\omega^\omega}, \dots\}$——它是「以 $\omega$ 为底迭代幂次」的极限。

## 4 公式解析：为什么 $1 + \omega = \omega$

用一个具体例子把极限情形拆开。序数加法的关键在第 2 式（后继步）与第 3 式（极限步）。看

$$
1 + \omega = \sup_{n \lt  \omega} (1 + n)
$$

- **第一步，识别极限步**：$\omega$ 是极限序数（不是任何 $\beta+1$），所以 $\alpha + \omega$ 必须用第三条递归式：$\alpha + \omega = \sup_{\beta \lt  \omega} (\alpha + \beta)$。这里 $\alpha = 1$。
- **第二步，逐段算**：$1 + 0 = 1$，$1 + 1 = 2$，$1 + 2 = 3$，…… 归纳得 $1 + n = n+1$（对有限 $n$，普通加法与序数加法一致）。
- **第三步，取并**：$\sup_{n\lt \omega} (n+1) = \bigcup_{n\lt \omega} (n+1) = \{0,1,2,\dots\} = \omega$。

而 $\omega + 1$ 走的是另一条路：$\omega + 1 = (\omega + 0) + 1 = \omega \cup \{\omega\}$，它比 $\omega$ 多一个「住在无穷之后」的新点。**差别在于后继与极限两种步法不可交换**：$1 + \omega$ 是「每段都比上段长一格的无穷接力」，收敛到 $\omega$；$\omega + 1$ 是「已经跑完无穷再迈一步」。

**辨析｜易错点：** 别把「$1 + \omega = \omega$」误读成「加法结合律破碎」——结合律 $(\alpha+\beta)+\gamma = \alpha+(\beta+\gamma)$ 对序数**仍然成立**。破碎的只有交换律。这也提醒我们：集合论里的「直觉」必须绑定到良序拼接的具体形状，而不是自然数的运算习惯。

## 6 动手推导：康托尔范式与 $\epsilon_0$

把序数算术的「以 $\omega$ 为底进位制」走一遍，看它如何把大序数「算」出来。

- **第一步，展开一个序数**：取 $\alpha = \omega^2 \cdot 3 + \omega \cdot 5 + 7$。康托尔范式说这就是唯一展开：指数 $\omega^2 > \omega > 1$ 递减，系数 $3, 5, 7$ 是正整数。
- **第二步，比较大小**：两个序数比大小 = 字典序比较范式：先比最高指数，再比该系数，再比下一指数……$\omega^2 \cdot 3 + 1$ 与 $\omega^2 \cdot 2 + \omega^{100}$ 谁大？最高指数都是 $\omega^2$，系数 $3 > 2$，故前者大。
- **第三步，$\epsilon_0$ 怎么来的**：$\epsilon_0 = \sup\{\omega, \omega^\omega, \omega^{\omega^\omega}, \dots\}$。因为 $\omega^{\epsilon_0} = \sup_n \omega^{\omega^{\omega^{\dots}}} = \epsilon_0$——它是「幂到不动点」的最小序数。范式里出现 $\epsilon_0$ 时，指数不再小于基数本身，范式展开才「稳定」。
- **第四步，为什么这有用**：康托尔范式把序数算术变成「可计算的」——判断相等、比较大小、做加乘都可归结为范式的机械操作。这是超限递归的「落地形态」：任何序数都能被有限数据表示（只要允许「取 sup」作为基本操作）。

**辨析｜易错点：** 康托尔范式的系数必须是**有限正整数**，指数必须**递减**。$1 + \omega$ 不能写成 $\omega^0 \cdot 1 + \omega$——范式的指数必须从左到右严格递减，$1$ 是 $\omega^0$，而 $\omega^0 = 1 \lt  \omega$，放不到前面。初学者常试图把「交换律失效」的式子强行按自然数直觉重排，从而写错范式。

## 7 小结

- **超限归纳**：零、后继、极限三步验证，覆盖全部序数；**超限递归**：每一步由此前一切步决定，可定义 $\mathrm{On}$ 上的类函数。
- **序数加法** = 良序首尾拼接；$1 + \omega = \omega \neq \omega + 1$，**交换律失效**。
- **序数乘法** = $\beta$ 份 $\alpha$ 依次拼接；$2 \cdot \omega = \omega \neq \omega \cdot 2$。
- **序数幂** $\alpha^\lambda = \sup_{\beta\lt \lambda}\alpha^\beta$；**康托尔范式**把每个序数唯一展开为 $\omega^{\beta_1}k_1 + \cdots + \omega^{\beta_n}k_n$。
- 最小 $\epsilon$-数 $\epsilon_0 = \sup\{\omega, \omega^\omega, \omega^{\omega^\omega}, \dots\}$ 满足 $\epsilon_0 = \omega^{\epsilon_0}$。

在下一节，我们转向「多少个」的问题：基数是怎样从序数里挑出来的？为什么 $\omega$ 与 $\omega+1$ 是同一个基数，而 $\aleph_1$