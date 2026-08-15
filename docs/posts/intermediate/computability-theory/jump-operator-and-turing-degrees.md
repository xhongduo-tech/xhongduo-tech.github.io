---
title: 跳跃算子与 Turing 度
date: 2026-08-07
---

# 跳跃算子与 Turing 度

<div class="epigraph">
<p>递归论的研究对象不是单独的集合，而是它们关于可计算归约所组成的等价类——这些等价类构成了一个丰富得惊人的数学结构。</p>
<footer>—— 约瑟夫 · 肖恩菲尔德（Joseph R. Shoenfield，*Degrees of Unsolvability\*, 1971）</footer>
</div>

<div class="article-byline">
<p>第二级 · 可计算性理论（递归论） ｜ R. I. Soare, *R. E. Sets and Degrees\*, §III.4–§IV.1 ｜ 2026-08-07</p>
</div>

## 为什么从 Turing 度开始

上一篇建立了两种归约，其中 Turing 归约 $\le_T$ 像一个"难度尺子"：$A \le_T B$ 表示 $A$ 不比 $B$ 难。把互相都能归约的集合归为一类，就得到**Turing 度**——可计算性意义上的"难度等级"。<span class="marginnote">Turing 度之于集合，如同等价类之于元素：集合只是"难度"的一个代表，度本身才是递归论真正研究的对象。这也呼应第一级《集合的概念》里"等价类划分"的思想，只是这次等价关系是"互为 Turing 可计算"。</span>有了度，自然的追问是：难度有没有顶？有没有底？从任一难度出发能否构造更难的？**跳跃算子（jump operator）** 回答第三个问题：它是"从任意集合生成一个严格更难的集合"的通用机器，也是本篇与下一篇 Post 问题共同的主角。

## 1 Turing 度：可计算性的等价类

**核心概念**：若 $A \le_T B$ 且 $B \le_T A$，称 $A$ 与 $B$ **Turing 等价**，记作 $A \equiv_T B$。$\equiv_T$ 是一个等价关系，它的每个等价类称为一个 **Turing 度（Turing degree）**，记作 $\deg(A)$（有时简写为 $\mathbf{a}, \mathbf{b}$）。

等价类的直觉：$A$ 与 $B$ 在同一条难度线上——任一方的神谕都能解出另一方。于是"以 $A$ 为神谕可算的一切"，与"以 $B$ 为神谕可算的一切"，是同一个集合。这个"以 $A$ 为神谕可计算的函数集合"也可以当作度的定义：

$$\deg(A) = \{ B \mid B \le_T A \text{ 且 } A \le_T B \}$$

两个度之间定义序关系：$\mathbf{a} \le \mathbf{b}$ 当且仅当存在 $A \in \mathbf{a}$、$B \in \mathbf{b}$ 使得 $A \le_T B$（这个定义与代表元的选择无关，是良定义的）。<span class="marginnote">把"集合"换成"度"，所有可判定性问题的表述都变得更干净：例如"停机问题不比 $K$ 难"变成"$\deg(K)$ 是 c.e. 度中的最大者"。第三级《算法设计与分析》里 NP 完备类也有同样的精神：不是研究单个问题，而是研究"难度等价类"。</span>

## 2 度的底部：可计算度 0

最简单的度是**可计算度**，记作 $\mathbf{0}$：它包含所有可判定集合（空集、偶数集、素数集……），它们是"难度为零"的一族——不需要任何神谕。

- $\mathbf{0}$ 是全部度结构中的**最小元**：对任何度 $\mathbf{a}$，$\mathbf{0} \le \mathbf{a}$。
- $\mathbf{0}$ 之上的下一个经典度是 $\mathbf{0}' = \deg(K)$，即停机问题（对角线集合 $K$）的度。

**定理（度的分层起点）**：$\mathbf{0} \lt  \mathbf{0}'$，且不存在"介于"$\mathbf{0}$ 与 $\mathbf{0}'$ 之间的 c.e. 度之外的阻塞——但我们暂时只知道这两层。**Post 问题**（下一篇的主角）问的就是：是否存在 c.e. 度严格介于 $\mathbf{0}$ 与 $\mathbf{0}'$ 之间。

## 3 跳跃算子：生成更难的集合

**核心概念**：给定集合 $A$，其**跳跃（jump）** 定义为

$$A' = \{ e \mid \Phi^A_e(e)\!\downarrow \}$$

即"$A$-神谕机 $\Phi^A_e$ 在输入 $e$ 上停机"的集合。$A'$ 是**相对化**的停机问题：以 $A$ 为神谕，问"第 $e$ 个 $A$-程序在输入 $e$ 上停不停机"。<span class="marginnote">跳跃把"停机问题不可判定"的构造相对化到任意集合 $A$ 上：对 $A$ 而言，$A'$ 正如 $K$ 对空神谕一样不可判定。这就是"相对化"的威力——把绝对结论推广为相对结论。</span>

跳跃的最基本定理：

**定理（跳跃严格增加）**：对任何 $A$，$A \lt _T A'$，即 $A \le_T A'$ 但 $A' \not\le_T A$。

证明用的是与停机问题完全相同的对角化：若 $A' \le_T A$，则存在 $A$-神谕机判定"$\Phi^A_e(e)$ 是否停机"，把它反转构造出一个"坏程序"，再让它问自己——矛盾。**跳跃是"任何集合都能构造出严格更难的集合"的保证。**

## 4 公式解析：$A'$ 的两种等价定义

跳跃有两种等价的写法，把它们并排拆开能看清本质：

$$A' = \{ e \mid \Phi^A_e(e)\!\downarrow \} = \{ \langle e, n \rangle \mid \Phi^A_e(n)\!\downarrow \}$$

- **第一步，看第一种写法**：$\{ e \mid \Phi^A_e(e)\!\downarrow \}$ 把 $A$-程序编号与输入叠在同一个 $e$ 上，是对角线形式（$e$ 同时在程序位置和输入位置）。这与 $K = \{e \mid \varphi_e(e)\downarrow\}$ 的构造逐字对应，只是把 $\varphi_e$ 换成 $\Phi^A_e$。
- **第二步，看第二种写法**：$\{ \langle e, n \rangle \mid \Phi^A_e(n)\!\downarrow \}$ 用配对编码 $\langle e, n \rangle$ 同时记录"程序编号 $e$"与"输入 $n$"，是"全停机问题"形式，与原始停机问题 $H$ 对应。
- **第三步，证明两者等价**：由 $s$-$m$-$n$ 定理，存在全可计算函数 $g$ 使 $\Phi^A_{g(e,n)}(x) = \Phi^A_e(n)$（对任意 $x$，尤其 $x = g(e,n)$）。于是 $g(e,n) \in \{e \mid \Phi^A_e(e)\downarrow\}$ 当且仅当 $\Phi^A_e(n)\downarrow$，当且仅当 $\langle e,n \rangle \in \{\langle e,n\rangle \mid \Phi^A_e(n)\downarrow\}$。两个集合互相 $\le_m$，故 Turing 等价。
- **第四步，度写法**：在度上，跳跃是良定义的运算：$\deg(A') = \mathbf{a}'$，且只依赖 $\mathbf{a}$ 不依赖代表元。于是 $A \le_T B \Rightarrow A' \le_T B'$，即跳跃在度的序上是**单调**的。

## 5 跳跃塔与度的图景

从任意集合出发，反复跳跃得到一条严格递增的"跳跃塔"：

$$A \lt _T A' \lt _T A'' \lt _T A''' \lt _T \cdots$$

在度上则是 $\mathbf{0} \lt  \mathbf{0}' \lt  \mathbf{0}'' \lt  \cdots$。这条塔没有顶：每一层都比上一层难，而且每层都有一个**相对化的停机问题**作为代表。下图是这条塔与前几篇概念的整合：

![Turing 度的跳跃层级](/images/computability-theory/jump-operator-and-turing-degrees-1.svg)

图中最底层的 $\mathbf{0}$ 是所有可判定集；$\mathbf{0}'$ 是停机问题 $K$ 的度；$\mathbf{0}''$ 是"$\mathbf{0}'$-停机问题"（与"是否存在无穷多停机程序"、"$\operatorname{Tot}$ 问题"等自然问题同度）……每上一层，不可判定性加深一层。<span class="marginnote">算术分层（第 8 篇）给这条塔以坐标：$\mathbf{0}^{(n)}$ 恰是算术分层中 $\Sigma_n$ 完备集的度。那时你会看到这条塔和"可以用一阶算术表达的命题"的层级完全重合。</span>

## 6 度结构的丰富性

跳跃塔只是度结构的冰山一角。围绕 $\le_T$ 的全体度组成一个巨大的偏序集 $\mathcal{D} = (D, \le)$，它有远超塔的丰富性：

- **最小元与最大元**：$\mathbf{0}$ 是最小元；**不存在最大元**（跳跃已保证每个度之上还有度）。
- **并运算**：任意两个度 $\mathbf{a}, \mathbf{b}$ 有**上确界（join）** $\mathbf{a} \vee \mathbf{b}$，由 $A \oplus B$（把 $A$ 与 $B$ 编码进一个集合）的度实现。
- **稠密性**（Sacks 稠密性定理）：对 c.e. 度 $\mathbf{a} \lt  \mathbf{b}$，存在 c.e. 度 $\mathbf{c}$ 严格介于其间——c.e. 度之间没有"相邻"。
- **嵌入**：$\mathcal{D}$ 能嵌入各种代数结构（格、偏序、甚至任意可数偏序），说明它极其复杂——递归论后期大量工作都在刻画"$\mathcal{D}$ 的一阶理论"。

**辨析｜易错点：** 不要以为"度越大越不可判定"就是"越坏"。很多度论定理关心的是**存在性与结构**：比如"是否存在既非可计算、又不与 $K$ 同度、还介于两者之间的 c.e. 度"——这是 Post 问题，下一篇会给出它的答案（是的，存在）。度结构不是一条链，而是处处有分支的丛林。

## 7 一个观察：$\mathbf{0}''$ 的日常面孔

跳跃塔的第二层 $\mathbf{0}''$ 并不抽象——它装着一大堆自然的"存在性"问题：

- **全函数问题** $\operatorname{Tot}$："程序 $e$ 是否对所有输入停机？"——$\Pi_2$ 完备，度为 $\mathbf{0}''$。
- **无穷停机问题** $\operatorname{Inf}$："程序 $e$ 是否在无穷多个输入上停机？"——$\Sigma_2$ 完备，度为 $\mathbf{0}''$。

这两个问题的共同点：它们都含有**两个交替的量词**（"对所有输入" + "存在步数"），对应算术分层的 $\Pi_2 / \Sigma_2$，也就是 $\mathbf{0}''$。而停机问题 $K$ 只有一个存在量词（$\Sigma_1$，度 $\mathbf{0}'$），所以它比这些问题"低一层"。

这个观察展示了跳跃与第 8 篇《算术分层》的接口：**量词交替一次，跳跃一级**。以后看到"对所有 $x$ 都存在 $y$"型的数学命题，你就能直觉地猜测它的难度位于 $\mathbf{0}''$ 附近——这种"从公式形状猜难度"的能力，是递归论给你的一把快尺。

## 8 术语速查表

| 术语 | 英文 | 含义 | 出处 |
| --- | --- | --- | --- |
| Turing 度 | Turing degree | 互为 Turing 可归约的集合等价类 | §1 |
| 可计算度 | computable degree $\mathbf{0}$ | 可判定集的度，度的最小元 | §2 |
| 跳跃 | jump $A'$ | $\{ e \mid \Phi^A_e(e)\!\downarrow \}$，$A$ 的"下一个难度" | §3 |
| 跳跃塔 | jump hierarchy | $\mathbf{0} \lt  \mathbf{0}' \lt  \mathbf{0}'' \lt  \cdots$ 严格递增 | §5 |
| join（上确界） | join | $\deg(A \oplus B) = \deg(A) \vee \deg(B)$ | §6 |
| 上确界 | supremum | 一族度的最小上界 | §6 |
| 稠密性 | density | 任意两度之间存在中间度（c.e. 情形） | §6 |
| 极小度 | minimal degree | 与 $\mathbf{0}$ 之间没有中间度的非零度 | §6 |
| Sacks 稠密性定理 | Sacks density theorem | 对 c.e. 度 $\mathbf{a} \lt  \mathbf{b}$，存在中间 c.e. 度 | §6 |
| 相对化 | relativization | 把定理原样搬到任意神谕上 | §3 |

## 9 小结

- **Turing 度**是"互相 Turing 可归约"的等价类；$\mathbf{0}$ 是最小元（可判定集），$\mathbf{0}' = \deg(K)$ 是停机问题的度。
- **跳跃算子** $A' = \{ e \mid \Phi^A_e(e)\downarrow \}$ 是相对化的停机问题，满足 $A \lt _T A'$——任何集合都有严格更难的集合。
- 跳跃在度上是单调良定义的运算，生成严格递增的**跳跃塔** $\mathbf{0} \lt  \mathbf{0}' \lt  \mathbf{0}'' \lt  \cdots$。
- 度结构 $\mathcal{D}$