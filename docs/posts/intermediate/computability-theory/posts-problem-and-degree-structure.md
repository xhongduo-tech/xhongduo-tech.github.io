---
title: Post 问题与度的结构
date: 2026-08-07
---

# Post 问题与度的结构

<div class="epigraph">
<p>递归可枚举度在序关系下构成了一个稠密、处处分支、丰富到难以想象的结构。</p>
<footer>—— 杰拉尔德 · 萨克斯（Gerald E. Sacks，*Degrees of Unsolvability\*, 1963）</footer>
</div>

<div class="article-byline">
<p>第二级 · 可计算性理论（递归论） ｜ R. I. Soare, *R. E. Sets and Degrees\*, §VII.4–§IX.3 ｜ 2026-08-07</p>
</div>

## 为什么从 Post 问题与度的结构开始

第 7 篇我们用优先方法构造了"中间度"，回答了 Post 问题。但 Post 问题的意义不止于"答案是是"——它打开了**度的结构理论**：全体 c.e. 度在 $\le_T$ 下组成什么形状的偏序？有没有稠密性？能否嵌入各种代数结构？这些结构性问题是递归论 1960–1980 年代的主战场。<span class="marginnote">Post 问题之所以重要，是因为它把递归论从"一个个不可判定性证明"推进到"一整族难度对象的结构分类"——正如群论把"一个个方程的可解性"推进到"群的结构理论"。</span>这一篇沿着结构理论的几条主干走一遍，并在结尾指出它通向更深的问题。

## 1 Post 问题：问题与答案

Post 问题由埃米尔 · 波斯特（Emil Post）在 1944 年明确提出，它问：

**Post 问题（Post's problem）**：是否存在 c.e. 集 $A$，使 $\mathbf{0} \lt  \deg(A) \lt  \mathbf{0}'$？换言之，c.e. 度在 $\mathbf{0}$ 与 $\mathbf{0}'$ 之间是否只有这两个端点？

答案（Friedberg 与 Muchnik，1957）：**存在**。事实上构造出的两个 c.e. 度甚至**互不可比较**——这比"介于中间"更强。因此 c.e. 度结构 $\mathcal{R}$ 不是一条从 $\mathbf{0}$ 到 $\mathbf{0}'$ 的链，而是从第一步起就四处开叉。<span class="marginnote">"存在互不可比较的 c.e. 度"的证明正是第 7 篇的有穷损害优先方法：构造两个集合 $A, B$，同时满足 $A \not\le_T B$ 与 $B \not\le_T A$ 两组要求。</span>

**核心概念**：**c.e. 度结构**，记作 $\mathcal{R} = (R, \le)$，是全体 c.e. 集合的 Turing 度在 $\le_T$ 下组成的偏序结构。它是递归论研究最深的结构之一。

## 2 c.e. 度的基本骨架

$\mathcal{R}$ 有哪些确定性的特征？

**最小元**：$\mathbf{0}$（可计算 c.e. 集）。
**最大元**：$\mathbf{0}'$（m-完备 c.e. 集的度，如 $K$）。
**上确界**：任意两个 c.e. 度 $\mathbf{a}, \mathbf{b}$ 有 join $\mathbf{a} \vee \mathbf{b}$（由 $A \oplus B$ 的度实现，仍是 c.e. 度）。
- **没有下确界（一般地）**：两个 c.e. 度的 meet（下确界）可能不存在——$\mathcal{R}$ 不是格。
- **不可比较**：存在 $\mathbf{a}, \mathbf{b}$ 互不可比较（Friedberg–Muchnik）。

**辨析｜易错点：** 有上确界未必有下确界，意味着 $\mathcal{R}$ 比"格"更野。很多人直觉上把偏序当成"树"或"格子"，但 $\mathcal{R}$ 中一对元素可能"没有公共下界"，这是结构理论一再强调的非直觉事实。

## 3 稠密性与嵌入：$\mathcal{R}$ 的丰满

$\mathcal{R}$ 远不止"有几根骨头"，它稠密得惊人：

**定理（Sacks 稠密性，1964）**：若 $\mathbf{a} \lt  \mathbf{b}$ 是 c.e. 度，则存在 c.e. 度 $\mathbf{c}$ 使得 $\mathbf{a} \lt  \mathbf{c} \lt  \mathbf{b}$。

这立刻推出：**c.e. 度之间没有"相邻"**——$\mathbf{0}$ 与 $\mathbf{0}'$ 之间塞得下无穷多个 c.e. 度，而且它们处处稠密，像有理数那样。<span class="marginnote">稠密性用无穷损害优先方法证明：为"在 $\mathbf{a}$ 与 $\mathbf{b}$ 之间插入 $\mathbf{c}$"构造一个高度精细的要求系统，每个要求都可能被损害无穷多次但仍收敛。这是第 7 篇"无穷损害"的第一个重大应用。</span>

**嵌入定理（Lachlan, Lerman 等）**：许多代数结构都能嵌入 $\mathcal{R}$：任意有限格、某些可数偏序、甚至 $\mathcal{R}$ 中每个度 $\mathbf{a}$ 之上的区间 $[\mathbf{a}, \mathbf{0}']$ 都可以复杂到嵌入各种结构。<span class="marginnote">"嵌入"在这里指：存在保序单射把给定结构搬进 $\mathcal{R}$。这类似于第三级《数据结构》里"图的同构嵌入"，只是这里的"图"换成了可数偏序。</span>

## 4 公式解析：join 与跳跃的代数

结构理论离不开代数运算。c.e. 度上的**join（上确界）**有精确的集合实现：

$$\deg(A \oplus B) = \deg(A) \vee \deg(B), \qquad A \oplus B = \{ 2n \mid n \in A \} \cup \{ 2n + 1 \mid n \in B \}$$

- **第一步，看编码**：$A \oplus B$ 把 $A$ 的元素编码成偶数、$B$ 的元素编码成奇数，塞进一个集合。给定 $\chi_{A \oplus B}$，只要看 $2n$ 与 $2n+1$ 两格，就能分别读出 $\chi_A(n)$ 与 $\chi_B(n)$。
- **第二步，证上界**：$A \le_T A \oplus B$（查偶数格），$B \le_T A \oplus B$（查奇数格），所以 $\deg(A \oplus B)$ 是 $\deg(A), \deg(B)$ 的一个上界。
- **第三步，证最小性**：若 $\deg(A), \deg(B) \le \mathbf{c}$，即 $A, B$ 都能以 $C$（$C \in \mathbf{c}$）为神谕判定，那么 $A \oplus B$ 也能以 $C$ 为神谕判定（分别算奇偶两路再合并）。故任何上界都 $\ge \deg(A \oplus B)$，它是**最小**上界。
- **第四步，推广**：跳跃也是"代数"运算：$\deg(A') = \mathbf{a}'$，且 $(\mathbf{a} \vee \mathbf{b})' = \mathbf{a}' \vee \mathbf{b}'$。c.e. 度的代数结构由 $\vee$ 与 $'$（和序 $\le$）共同生成，是结构理论的基本语言。

## 5 从 c.e. 度到全部度：极小度

结构理论还研究**全体 Turing 度**（不只 c.e. 度），其中最美也最神秘的对象是**极小度（minimal degree）**：

**核心概念**：一个非零度 $\mathbf{m}$ 称为**极小度**，若不存在度 $\mathbf{d}$ 使得 $\mathbf{0} \lt  \mathbf{d} \lt  \mathbf{m}$——即 $\mathbf{m}$ 与 $\mathbf{0}$ 之间没有"中间"。

极小度的存在性（Spector，1956；Sacks 用无穷损害简化）是无穷损害优先方法最早的胜利之一。极小度对应的集合"看不出任何中间复杂度"，它们像递归论中的"原子"。<span class="marginnote">极小度与第 6 篇"度结构处处稠密"并不矛盾：稠密性只对 c.e. 度成立，极小度不在 c.e. 度中。c.e. 度稠密、非 c.e. 度却可以孤立，这两幅图景拼出 $\mathcal{D}$ 的完整性格。</span>

**定理（结构大观）**：全体度结构 $\mathcal{D}$ 是：一个**不可数**的偏序（有 $2^{\aleph_0}$ 个度）、有最小元 $\mathbf{0}$、无最大元、每个度之上有度、任意对子有 join、某些对子没有 meet。

## 6 $\mathcal{D}$ 与 $\mathcal{R}$ 的一阶理论

结构理论的终极问题之一是：**$\mathcal{R}$ 与 $\mathcal{D}$ 的一阶理论可判定吗？** 换句话说，是否存在算法判定"关于度的任意一阶句子是否为真"？

**$\mathcal{D}$ 的一阶理论**：Harrington–Shelah 等人证明其极其复杂，**不可判定**（甚至互递归于真二阶算术的某个片段）。
**$\mathcal{R}$ 的一阶理论**：Lachlan 证明它不可判定；之后 Slaman 与 Woodin 进一步证明它等价于真算术（true arithmetic）——即"c.e. 度结构的可定义性"与"自然数的全部算术真理"具有相同的复杂度。<span class="marginnote">Slaman–Woodin 的结果把 $\mathcal{R}$ 抬到与"自然数结构"同等的复杂度：要完全描述 c.e. 度结构，至少要知道所有算术真理。这是结构理论对"$\mathcal{R}$ 有多复杂"的终极回答之一。</span>

把"结构有多复杂"落到一条具体的句子上，一切就直观了。**Sacks 稠密性用一阶句子写成**

$$\forall \mathbf{a}\, \forall \mathbf{b}\, \big(\mathbf{a} < \mathbf{b} \to \exists \mathbf{c}\, (\mathbf{a} < \mathbf{c} \land \mathbf{c} < \mathbf{b})\big)$$

判定"这样的句子在 $\mathcal{R}$ 中是否为真"正是 $\mathcal{R}$ 的一阶理论；而"$\mathcal{R}$ 中有不可比较的度"写成 $\exists \mathbf{a}\, \exists \mathbf{b}\, (\mathbf{a} \not\le \mathbf{b} \land \mathbf{b} \not\le \mathbf{a})$，同样是一条一阶句子。**一阶理论就是在问：所有能用变量句子写出的结构性质，哪些为真。** Slaman–Woodin 说，回答这个问题的算法不存在，而且它的复杂度和判定"自然数的全部算术真理"完全一样。

这些结果说明：**度的结构不是"一个小玩具"，而是一个能编码算术全部复杂性的对象**——递归论以"可计算"为工具，却挖出了一个"不可计算到极致"的结构。这种自指式的深度，正是递归论最迷人的地方。

## 7 里程碑一览：度的结构理论是如何长出来的

把结构理论的几个关键节点串起来，能看清这门学问的成长轨迹：

- **1944，Post 问题提出**：明确追问 $\mathbf{0}$ 与 $\mathbf{0}'$ 之间是否有 c.e. 度，此后十年递归论苦于没有构造工具。
- **1957，Friedberg–Muchnik**：有穷损害优先方法登场，Post 问题获肯定解决，同时证明互不可比较的 c.e. 度存在。
- **1964，Sacks 稠密性**：无穷损害优先方法成熟，证明 c.e. 度处处稠密——"两度之间总有中间度"。
- **1966，Lachlan 的不可嵌入结果**：随后发现 $\mathcal{R}$ 中有"嵌入障碍"，证明结构远比"稠密"复杂。
- **1970s，Lachlan–Lerman 等**：系统的嵌入理论成型，各种有限格、可数偏序被证实可嵌入。
- **1990s，Slaman–Woodin**：$\mathcal{R}$ 的一阶理论等价于真算术，结构复杂度登顶。

这条时间线展示的是一个范式跃迁：从"证明一个存在"（Post 问题）到"刻画整个结构"（$\mathcal{R}$ 的一阶理论）。递归论从未停止给"难度"画更细的地图。

## 8 术语速查表

| 术语 | 英文 | 含义 | 出处 |
| --- | --- | --- | --- |
| c.e. 度结构 | c.e. degree structure $\mathcal{R}$ | 全体 c.e. 度在 $\le_T$ 下的偏序 | §1 |
| Post 问题 | Post's problem | $\mathbf{0}$ 与 $\mathbf{0}'$ 之间是否存在 c.e. 度 | §1 |
| 上确界 | join | 两个度的最小上界，由 $A \oplus B$ 实现 | §4 |
| 下确界 | meet | 两个度的最大下界（$\mathcal{R}$ 中未必存在） | §2 |
| Sacks 稠密性 | Sacks density | c.e. 度 $\mathbf{a} \lt  \mathbf{b}$ 之间存在中间 c.e. 度 | §3 |
| 嵌入 | embedding | 保序单射把给定结构搬进 $\mathcal{R}$ | §3 |
| 极小度 | minimal degree | 与 $\mathbf{0}$ 之间没有中间度的非零度 | §5 |
| 一阶理论 | first-order theory | 关于一个结构的全体一阶真句子 | §6 |
| 真算术 | true arithmetic | 与 $\mathcal{R}$ 的一阶理论复杂度相当的对象 | §6 |
| 不可数度 | continuum many degrees | $\mathcal{D}$ 共有 $2^{\aleph_0}$ 个度 | §5 |

## 9 小结

- **Post 问题**（是否存在 $\mathbf{0}$ 与 $\mathbf{0}'$ 之间的 c.e. 度）已获肯定解决，且构造出互不可比较的 c.e. 度。
- **c.e. 度结构 $\mathcal{R}$**：有最小元 $\mathbf{0}$、最大元 $\mathbf{0}'$、任意对子有 join、一般无 meet，不是格。
- **Sacks 稠密性**：$\mathbf{a} \lt  \mathbf{b}$ 之间有 c.e. 度——c.e. 度处处稠密。
- 各种代数结构可**嵌入** $\mathcal{R}$，其复杂度随跳跃层级递增。
- **极小度**（非 c.e.）与 c.e. 稠密性共存，构成 $\mathcal{D}$ 的完整图景；$\mathcal{R}$ 的一阶理论等价于真算术，结构复杂度登顶。
- 结构理论从"证明一个存在"（Post 问题）走到"刻画整个结构"（一阶理论），方法论也从有穷损害升级到无穷损害与树论证。

在下一节，我们将换一个方向继续挖掘"可计算"的边界——**算法随机性**（Martin-Löf 随机性）：问一个无限序列何时才算"随机"，答案竟是它通过了所有可计算的统计检验。