---
title: 中国剩余定理（Chinese Remainder Theorem）
date: 2026-08-07
---

# 中国剩余定理（Chinese Remainder Theorem）

<div class="epigraph">
<p>孙子算经问：「今有物不知其数，三三数之剩二，五五数之剩三，七七数之剩二。」——两千年前的算题，是环论直积分解的雏形。</p>
<footer>—— 自 题（中国剩余定理笔记）</footer>
</div>

<div class="article-byline">
<p>第二级 · 抽象代数 ｜ 杨子胥《近世代数》§8.6 ｜ 2026-08-07</p>
</div>

## 为什么从中国剩余定理开始

「物不知数」问题：求一个整数，除以 3 余 2、除以 5 余 3、除以 7 余 2。《孙子算经》给出了解法，而它的现代抽象形态就是**中国剩余定理（Chinese Remainder Theorem, CRT）**——一条横跨数论、环论与计算的伟大定理。环论版本说：

$$
R / (I_1 I_2 \cdots I_k) \ \cong \ (R / I_1) \times (R / I_2) \times \cdots \times (R / I_k)
$$

当理想 $I_i$ 两两互素时。CRT 的核心思想：**同时满足多个同余条件，等价于在多个商环的直积里选一个元素。** 每个分量独立、互不干扰——「分而治之」在代数里有了精确形式。

CRT 是数论（解同余方程组）、密码学（RSA 加速解密）、编码理论（第十二篇）与多项式插值（拉格朗日插值）的共同地基。本节从整数版本出发，建立环论版本并证明之。

## 1 整数版本：物不知数的代数化

**定理（整数中国剩余定理）：** 设 $m_1, \dots, m_k$ 两两互素，则对任意整数 $a_1, \dots, a_k$，同余方程组

$$
x \equiv a_1 \pmod{m_1}, \quad x \equiv a_2 \pmod{m_2}, \quad \dots, \quad x \equiv a_k \pmod{m_k}
$$

有解，且解模 $M = m_1 m_2 \cdots m_k$ 唯一。

**证明（构造解）：** 设 $M_i = M / m_i$。因为 $m_i$ 与 $M_i$ 互素，$M_i$ 模 $m_i$ 可逆，设 $M_i t_i \equiv 1 \pmod{m_i}$。则

$$
x = a_1 M_1 t_1 + a_2 M_2 t_2 + \cdots + a_k M_k t_k
$$

模 $m_i$ 时，除第 $i$ 项外所有项都含 $m_i$ 的因子（$M_j$ 含 $m_i$ 当 $j \ne i$），故 $x \equiv a_i M_i t_i \equiv a_i \pmod{m_i}$——每个同余都满足。唯一性：若 $x, x'$ 都满足，则 $x - x'$ 被每个 $m_i$ 整除，因 $m_i$ 两两互素，$x - x'$ 被 $M$ 整除。$\blacksquare$<span class="marginnote">构造 $x = \sum a_i M_i t_i$ 是「分治」的典范：每个分量 $a_i$ 只在自己的模 $m_i$ 下起作用（$M_i t_i \equiv 1$），在别的模下被 $m_j$ 整除成 0。这就像投票：每个「条件」把自己的候选人投进对应分量，互不干扰。孙子的「三三数之剩二」解得 $x = 23$（模 105 唯一）。</span>

**例（物不知数）**：$m_1 = 3, m_2 = 5, m_3 = 7$，$a = (2, 3, 2)$。$M = 105$，$M_1 = 35$，$t_1 = 2$（$35 \cdot 2 \equiv 1 \pmod 3$）；$M_2 = 21$，$t_2 = 1$（$21 \equiv 1 \pmod 5$）；$M_3 = 15$，$t_3 = 1$（$15 \equiv 1 \pmod 7$）。$x = 2 \cdot 35 \cdot 2 + 3 \cdot 21 \cdot 1 + 2 \cdot 15 \cdot 1 = 140 + 63 + 30 = 233 \equiv 23 \pmod{105}$。$\checkmark$——**答案是 23**。

## 2 环论版本：商环的直积分解

把整数版本抽象到任意环，得到最深刻的形态。

**定理（中国剩余定理，环论版本）：** 设 $R$ 是含幺交换环，$I_1, \dots, I_k$ 是两两**互素**的理想（$I_i + I_j = R$ 对 $i \ne j$）。则

$$
R \big/ (I_1 \cap I_2 \cap \cdots \cap I_k) \ \cong \ (R/I_1) \times (R/I_2) \times \cdots \times (R/I_k)
$$

映射为 $r + \bigcap I_i \mapsto (r + I_1, \dots, r + I_k)$。

**注**：两两互素时 $\bigcap I_i = I_1 I_2 \cdots I_k$（乘积 = 交），所以左边常写作 $R/(I_1 \cdots I_k)$。<span class="marginnote">环论版本的洞察：「模 $I_1 \cap \cdots \cap I_k$ 取同余」分解为「分别模每个 $I_i$」的直积。条件 $I_i + I_j = R$（互素）保证「每个分量可以独立指定」——若理想不互素，直积会多出「相容性」约束。$I_i + I_j = R$ 正是「$m_i$ 与 $m_j$ 互素」在理想语言里的化身。</span>

**证明（$k = 2$ 情形）：** 考虑 $\varphi : R \to (R/I_1) \times (R/I_2)$，$\varphi(r) = (r + I_1, r + I_2)$。
- **同态**：逐分量保持加乘；
- **核**：$\varphi(r) = (0, 0) \iff r \in I_1$ 且 $r \in I_2$，故 $\ker\varphi = I_1 \cap I_2$；
- **满射**：由 $I_1 + I_2 = R$，存在 $e_1 + e_2 = 1$（$e_1 \in I_1$、$e_2 \in I_2$）。取 $r = a_2 e_1 + a_1 e_2$，则 $r \equiv a_1 \pmod{I_1}$（因为 $a_2 e_1 \in I_1$、$a_1 e_2 \equiv a_1 \cdot 1$）且 $r \equiv a_2 \pmod{I_2}$——任意目标 $(a_1, a_2)$ 都被取到，满射。

套第一同构定理：$R/(I_1 \cap I_2) \cong (R/I_1) \times (R/I_2)$。$k$ 个理想的情形归纳即得。$\blacksquare$<span class="marginnote">满射性的构造 $r = a_2 e_1 + a_1 e_2$ 是「互素分解 1 = e_1 + e_2」的应用：$e_1 \in I_1$、$e_2 \in I_2$ 像「单位元的互素分裂」，让每个分量可以独立调整而不影响其他。这套「幂等元分解」的思想在代数、表示论、乃至信号处理的滤波器组里反复出现。</span>

## 3 例子：从 CRT 到整数直积

CRT 在整数上的应用是「交换直积」的转换器。

**例：$\mathbb{Z}_n$ 的直积分解。** $n = m_1 \cdots m_k$（$m_i$ 两两互素）时：

$$
\mathbb{Z}_n \ \cong \ \mathbb{Z}_{m_1} \times \mathbb{Z}_{m_2} \times \cdots \times \mathbb{Z}_{m_k}
$$

因为 $\mathbb{Z}_n = \mathbb{Z}/\langle n\rangle$、$\mathbb{Z}_{m_i} = \mathbb{Z}/\langle m_i \rangle$，且 $\langle m_i \rangle$ 两两互素（$\gcd(m_i, m_j) = 1$）。这正坐实第六篇的「互素合并」：$\mathbb{Z}_6 \cong \mathbb{Z}_2 \times \mathbb{Z}_3$、$\mathbb{Z}_{12} \cong \mathbb{Z}_4 \times \mathbb{Z}_3$。

**例（RSA 的解密加速）**：RSA 解密算 $m = c^d \bmod n$，$n = pq$。用 CRT 把 $n$ 拆成 $p, q$ 两部分分别算 $m_p = c^d \bmod p$、$m_q = c^d \bmod q$（指数可先按费马小定理缩小），再用 CRT 重组 $m$。**解密速度提升约 4 倍**——CRT 是 RSA 工程实现的标准加速器（第十二篇详述）。<span class="marginnote">CRT 在 RSA 里的角色：把「模大数 $n$」的慢运算拆成「模 $p$、模 $q$」的快运算再重组。$c^d \bmod p$ 比 $c^d \bmod n$ 快得多（数字小、且指数可按 $p-1$ 缩小）。「分治 + 重组」的 CRT 思想，从《孙子算经》一路用到现代密码学。</span>

**例（拉格朗日插值）**：$F[x]$ 中 $\langle x - a_i \rangle$ 两两互素（$a_i \ne a_j$），CRT 给出

$$
F[x] \big/ \langle (x - a_1)\cdots(x - a_k) \rangle \ \cong \ F[x]/\langle x - a_1 \rangle \times \cdots \times F[x]/\langle x - a_k \rangle \cong F^k
$$

「在 $k$ 个点取值」= 在 $k$ 个坐标独立取值——这就是拉格朗日插值多项式的代数本质：多项式 $f$ 由它在 $k$ 个点的值唯一决定（次数 < $k$）。

## 4 公式解析：R/(I₁⋯I_k) ≅ ∏(R/I_i) 的机制

把 CRT 环论版本的核心机制「互素 ⟹ 直积」拆透。

- **第一步，互素的意义。** $I_i + I_j = R$ 意味着存在 $e_{ij} \in I_i$、$e'_{ij} \in I_j$ 使 $e_{ij} + e'_{ij} = 1$。这是「$1$ 可以按理想分裂」。

- **第二步，为什么需要互素（满射性的关键）。** 要取到任意目标 $(a_1, \dots, a_k)$，需要一个「幂等元族」$e_1, \dots, e_k$：$e_i \equiv 1 \pmod{I_i}$、$e_i \equiv 0 \pmod{I_j}$（$j \ne i$）。互素保证这种「独立开关」存在。$k = 2$ 时 $e_1 = e_2$（$e_1 + e_2 = 1$ 中取 $e_1 \in I_2$ 的分量）即 $r = \sum a_i e_i$ 的实现。

- **第三步，核的形态。** 同时满足「每个分量都是 0」的元素正是 $\bigcap I_i$。两两互素时 $\bigcap I_i = I_1 I_2 \cdots I_k$（归纳证明：$(I_1 \cap I_2) = I_1 I_2$ 当互素），所以左边写成 $R/(I_1\cdots I_k)$。

- **第四步，意义。** CRT 说「取同余」可以被分解成「每个理想独立取同余」的直积——**约束的分治**。$R/I_1 \cap \cdots \cap I_k$ 里那个「同时满足全部约束」的元素，等价于直积里各分量「各管各的约束」。整个现代密码学与代数计算都站在这个「分治」上。

## 5 例：多项式版与「造结构的通用性」

CRT 在多项式环里还能造出「既有结构」的范例。

**例：$F[x]/\langle fg \rangle$ 的分解**（$f, g$ 互素多项式）。$\langle f\rangle + \langle g \rangle = F[x]$（互素 ⟺ $f, g$ 无公共根），CRT 给出

$$
F[x] / \langle fg \rangle \ \cong \ F[x]/\langle f \rangle \times F[x]/\langle g \rangle
$$

**例：用 CRT 解多项式方程。** 求满足「模 $x-1$ 余 $2$、模 $x-2$ 余 $3$」的多项式 $f$（次数 < 2）：$f(1) = 2$、$f(2) = 3$，直线 $f(x) = x + 1$——一次插值就是 CRT 的最简情形。<span class="marginnote">「CRT 无处不在」的根源：任何「取同余」结构（整数、多项式、函数环）在两两互素时都分裂成直积。第十二篇线性码与循环码会看到：CRT 把模多项式环拆成「子模」的直积，从而设计出有清晰纠错能力的码。</span>

## 6 小结

- **整数 CRT**：两两互素模数下，同余方程组有解且模 $M$ 唯一；构造解 $x = \sum a_i M_i t_i$。
- **环论 CRT**：$I_i$ 两两互素 ⟹ $R/\bigcap I_i \cong \prod (R/I_i)$；互素时 $\bigcap = $ 乘积。
- **满射性的关键**：互素给出「幂等元分解 $1 = \sum e_i$」，各分量独立可调。
- **应用**：$\mathbb{Z}_n$ 直积分解、RSA 解密加速（快 4 倍）、拉格朗日插值、$F[x]/\langle fg\rangle$ 分解。
- CRT = 约束的分治：多约束 ⟺ 直积各分量独立。

在下一节，我们离开环的一般理论，进入整环的分解王国：**整环上的一元多项式环**。多项式环是环论里最重要的具体舞台，也是第九篇分解理论的第一个主角。
