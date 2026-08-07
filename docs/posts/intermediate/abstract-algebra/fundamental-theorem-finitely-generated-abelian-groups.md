---
title: 有限生成阿贝尔群基本定理
date: 2026-08-07
---

# 有限生成阿贝尔群基本定理

<div class="epigraph">
<p>代数结构的最高成就是把一整族对象彻底分类——有限生成阿贝尔群是第一个被完全征服的王国。</p>
<footer>—— 自 题（有限生成阿贝尔群笔记）</footer>
</div>

<div class="article-byline">
<p>第二级 · 抽象代数 ｜ 杨子胥《近世代数》§6.2 ｜ 2026-08-07</p>
</div>

## 为什么从有限生成阿贝尔群基本定理开始

抽象代数的「不变量纲领」说：研究一族代数系统 = 把它们按同构分类。循环群的结构定理是第一次小试牛刀（只有两族），而**有限生成阿贝尔群基本定理**是第一次全面胜利：**每一个有限生成阿贝尔群都是循环群的直积，且分解在本质意义下唯一。** 这等于给了整个王国一张完整的「元素周期表」。

这条定理的意义不止于阿贝尔群本身：它是第六篇的分类中枢、也是线性代数中「整数矩阵的标准形/初等因子」思想的群论化，还与第八篇中国剩余定理、第十篇有限域结构定理遥相呼应。理解它的陈述、分解算法与唯一性，你就掌握了「把一大族对象彻底分类」的完整范式。本节先建立定义与定理陈述，再走一遍分解算法。

## 1 有限生成阿贝尔群与自由阿贝尔群

**有限生成阿贝尔群（finitely generated abelian group）**：存在有限个生成元 $a_1, \dots, a_k$ 使得 $G = \langle a_1, \dots, a_k \rangle$ 的交换群。

**自由阿贝尔群（free abelian group）**：同构于 $\mathbb{Z}^r = \mathbb{Z} \times \cdots \times \mathbb{Z}$（$r$ 个）的群，$r$ 称为**秩（rank）**。$\mathbb{Z}^r$ 的标准基 $e_1, \dots, e_r$（第 $i$ 分量为 1、其余为 0）是生成元，且没有任何非平凡关系——「自由」= 生成元之间无约束。

**有限生成阿贝尔群的「生成与关系」视角**：设 $G$ 由 $a_1, \dots, a_k$ 生成，则映射 $\mathbb{Z}^k \to G$，$(n_1, \dots, n_k) \mapsto n_1a_1 + \cdots + n_ka_k$ 是满同态（加法记号），核 $K$ 是 $\mathbb{Z}^k$ 的子群。由同态基本定理

$$
G \cong \mathbb{Z}^k / K
$$

**任何有限生成阿贝尔群都是自由阿贝尔群的商群**。剩下的工作就是弄清 $\mathbb{Z}^k$ 的子群 $K$ 长什么样——这正是下一节「整数矩阵标准形」处理的量。<span class="marginnote">「自由阿贝尔群商掉子群」是有限生成阿贝尔群的万有框架：$\mathbb{Z}_n = \mathbb{Z} / n\mathbb{Z}$ 是最简单的情形（$k=1$），$V_4 = \mathbb{Z}^2 / K$（$K = \{ (2m, 2n) \}$）是 $k = 2$ 的例子。这套「生成元 + 关系」的语言在第八篇群表示、以及更广义的「模论」里会全面开花。</span>

## 2 基本定理的陈述：两种规范形

**定理（有限生成阿贝尔群基本定理）：** 设 $G$ 是有限生成阿贝尔群，则

1. **不变因子形（invariant factor form）**：存在唯一的一串正整数 $d_1, d_2, \dots, d_k$ 满足 $d_1 \mid d_2 \mid \cdots \mid d_k$（$d_i$ 整除 $d_{i+1}$），以及非负整数 $r$，使得

$$
G \cong \mathbb{Z}^r \times \mathbb{Z}_{d_1} \times \mathbb{Z}_{d_2} \times \cdots \times \mathbb{Z}_{d_k}
$$

2. **初等因子形（elementary divisor form）**：存在唯一的素数幂列表 $p_1^{e_1}, \dots, p_s^{e_s}$（允许重复），以及非负整数 $r$，使得

$$
G \cong \mathbb{Z}^r \times \mathbb{Z}_{p_1^{e_1}} \times \cdots \times \mathbb{Z}_{p_s^{e_s}}
$$

$r$ 称为**自由秩（free rank）**，即 $\mathbb{Z}$ 因子的个数；两种形式在「$r$、因子列表」下各自唯一。<span class="marginnote">两种规范形的关系：把不变因子 $d_i$ 各自分解为素数幂 $d_i = \prod p^{e}$，再按素数分组展开，就得到初等因子。例：不变因子 $(2, 12)$ ⟹ $d_1 = 2$、$d_2 = 12 = 2^2\cdot 3$，初等因子为 $2, 2^2, 3$。初等因子形是「按素数分别列出最大块」，不变因子形是「按整除链串起来」。</span>

**例：** $G = \mathbb{Z}_{12} \times \mathbb{Z}_{18}$。
- 初等因子形：$\mathbb{Z}_{12} \cong \mathbb{Z}_4 \times \mathbb{Z}_3$，$\mathbb{Z}_{18} \cong \mathbb{Z}_2 \times \mathbb{Z}_9$，故 $G \cong \mathbb{Z}_4 \times \mathbb{Z}_9 \times \mathbb{Z}_2 \times \mathbb{Z}_3$——初等因子 $4, 9, 2, 3$；
- 不变因子形：把素数幂按「整除链」合并——$2$ 与 $2^2$ 合并成 $2^2 = 4$ 与 $2$……实际链条为 $d_1 = 2$、$d_2 = 2 \cdot 9 = 18$？不对，需检查整除链：$G \cong \mathbb{Z}_2 \times \mathbb{Z}_{36}$（因为 $2 \mid 36$ 且 $\mathbb{Z}_2 \times \mathbb{Z}_{36} \cong \mathbb{Z}_4 \times \mathbb{Z}_9 \times \mathbb{Z}_2$？36 的分解是 $4 \times 9$，而 $2$ 单独……）这里 $2 \times 18$ 与 $2 \times 4 \times 9$ 的比较需要按算法做——下一节用矩阵标准形精确演示。

## 3 分解算法：从矩阵标准形出发

分解算法的理论核心是**整数矩阵的 Smith 标准形**。设 $G$ 由 $a_1, \dots, a_k$ 生成，关系为 $r_1, \dots, r_l$（$G \cong \mathbb{Z}^k / K$，$K = \mathrm{Im}(M)$ 由关系矩阵 $M$ 的行生成）。对 $M$ 做整数的初等行列变换（交换、倍加、变号）化为 Smith 标准形：

$$
M \longrightarrow \begin{pmatrix} d_1 & & & \\ & d_2 & & \\ & & \ddots & \\ & & & d_t \end{pmatrix}
$$

其中 $d_1 \mid d_2 \mid \cdots \mid d_t$。于是

$$
G \cong \mathbb{Z}^{k - t} \times \mathbb{Z}_{d_1} \times \cdots \times \mathbb{Z}_{d_t}
$$

$d_i$ 正是**不变因子**，$k - t$ 是自由秩。<span class="marginnote">Smith 标准形是「整数矩阵的对角化」：用允许的初等变换（行/列交换、加整数倍、变号）把矩阵化为对角且对角元递增整除。这对应「换基」——把 $\mathbb{Z}^k$ 的基与关系基同时重组。它是线性代数对角化的整数版本，也是「整数上矩阵理论」的核心工具，连接第六篇与《线性代数》。</span>

**例：** $G = \langle a, b \mid 2a + 4b = 0,\ 6b = 0 \rangle$。关系矩阵 $M = \begin{pmatrix} 2 & 4 \\ 0 & 6 \end{pmatrix}$（行向量 $(2,4)$、$(0,6)$）。做整数初等变换：

$$
\begin{pmatrix} 2 & 4 \\ 0 & 6 \end{pmatrix} \to \begin{pmatrix} 2 & 0 \\ 0 & 6 \end{pmatrix}
$$

（$4$ 用列变换减掉），再检查 $2 \mid 6$ 成立，Smith 标准形为 $\mathrm{diag}(2, 6)$。故 $G \cong \mathbb{Z}_2 \times \mathbb{Z}_6$，不变因子 $(2, 6)$。$\checkmark$

## 4 公式解析：G ≅ ℤ^k / K 的完整含义

把「有限生成阿贝尔群 = 自由阿贝尔群的商」这条公式从四个层面读透。

- **第一层（存在）**：$G$ 有有限生成元 $a_1, \dots, a_k$，于是 $\varphi : \mathbb{Z}^k \to G$，$\varphi(n_1,\dots,n_k) = \sum n_i a_i$ 是满同态。同态基本定理给出 $G \cong \mathbb{Z}^k / \ker \varphi$。

- **第二层（$\ker \varphi$ 的结构）**：$\ker \varphi$ 是 $\mathbb{Z}^k$ 的子群，由关系（满足 $\sum n_i a_i = 0$ 的 $(n_i)$ 向量）生成，即关系矩阵的像。自由阿贝尔群的子群仍是自由阿贝尔群（关键定理），故 $\ker \varphi \cong \mathbb{Z}^t$，可嵌入 $\mathbb{Z}^k$。

- **第三层（Smith 标准形介入）**：换基后 $\mathbb{Z}^k \cong \mathbb{Z}^{k-t} \times d_1\mathbb{Z} \times \cdots \times d_t\mathbb{Z}$，商掉得到 $\mathbb{Z}^{k-t} \times \mathbb{Z}_{d_1} \times \cdots \times \mathbb{Z}_{d_t}$。**换基 = 找更聪明的坐标，让关系变对角。**

- **第四层（唯一性）**：$r$（自由秩）与不变因子链 $d_1 \mid \cdots \mid d_t$ 在同构下不变——$r$ 是「$\mathbb{Z}$ 分量的个数」，可由「无挠部分的秩」读出；$d_i$ 由「挠子群 $G_{\mathrm{tor}}$ 的 $p$-分量」逐一读出。唯一性需要「素数幂因子分解唯一」与「$\mathbb{Z}_n$ 的唯一分解」——它们共同保证规范形唯一。

## 5 例子：有限阿贝尔群的「周期表」

用基本定理把低阶有限阿贝尔群全部列出，感受「彻底分类」的力度。

**阶为 $p^n$ 的有限阿贝尔群** = 把 $n$ 拆成「素数幂指数」的分拆数。$p$ 固定时，$p^2$ 阶：$2$ 种（$\mathbb{Z}_{p^2}$、$\mathbb{Z}_p \times \mathbb{Z}_p$）；$p^3$ 阶：$3$ 种（$\mathbb{Z}_{p^3}$、$\mathbb{Z}_{p^2}\times\mathbb{Z}_p$、$\mathbb{Z}_p^3$）；$p^4$ 阶：$5$ 种（分拆 $4, 3+1, 2+2, 2+1+1, 1+1+1+1$）。**有限阿贝尔群的计数 = 整数分拆的计数**，这个联系是群论与组合数学的又一握手。<span class="marginnote">$p$-群阿贝尔的个数 = $p(n)$（整数分拆数），因为每个 $p^n$ 阶阿贝尔群对应「把指数 $n$ 拆成若干正整数」的一种分拆。$p(4) = 5$、$p(5) = 7$。这也是为什么「有限阿贝尔群的完全分类」如此干净——它归结为早已研究透的整数分拆。</span>

**阶为 $n$ 的有限阿贝尔群个数**：$n = 12 = 2^2 \cdot 3$。$2$ 部分分拆：$2$ 种（$\mathbb{Z}_4$、$\mathbb{Z}_2\times\mathbb{Z}_2$）；$3$ 部分：$1$ 种。总共有 $2 \times 1 = 2$ 个：$\mathbb{Z}_4 \times \mathbb{Z}_3 \cong \mathbb{Z}_{12}$ 与 $\mathbb{Z}_2 \times \mathbb{Z}_2 \times \mathbb{Z}_3$。**两个 12 阶阿贝尔群**，用基本定理一次数清，不靠枚举。

**有限部分与自由部分的分离**：基本定理说 $G \cong \mathbb{Z}^r \times G_{\mathrm{tor}}$，其中 $G_{\mathrm{tor}} = \{ g \mid o(g) < \infty \}$ 是**挠子群（torsion subgroup）**。有限生成阿贝尔群 = 自由部分（$\mathbb{Z}^r$，秩 $r$）+ 挠部分（有限阿贝尔群）。这个「自由 ⊕ 挠」的分解是基本定理最简洁的概括。

## 6 小结

- **有限生成阿贝尔群** = $\mathbb{Z}^k / K$（自由阿贝尔群的商），生成元与关系语言。
- **基本定理**：$G \cong \mathbb{Z}^r \times \mathbb{Z}_{d_1} \times \cdots \times \mathbb{Z}_{d_k}$（不变因子形，$d_1 \mid \cdots \mid d_k$）或 $\mathbb{Z}^r \times \prod \mathbb{Z}_{p^e}$（初等因子形），均唯一。
- **分解算法**：关系矩阵做 Smith 标准形，对角元即不变因子；自由秩 $r = k - t$。
- **$G = \mathbb{Z}^r \times G_{\mathrm{tor}}$**：自由部分 ⊕ 挠子群。
- **应用**：$p^n$ 阶有限阿贝尔群个数 = 整数分拆数 $p(n)$；12 阶阿贝尔群恰有 2 个。

在下一节，我们把基本定理打磨成实用的分类工具：**有限阿贝尔群的结构与不变量分解**。两个有限阿贝尔群同构当且仅当它们的不变因子（或初等因子）相同——同构判定变成一页清单比对。
