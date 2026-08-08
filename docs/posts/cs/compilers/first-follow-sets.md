---
title: FIRST 集与 FOLLOW 集的计算
date: 2026-08-07
---

# FIRST 集与 FOLLOW 集的计算

<div class="epigraph">
<p>预测的本质，是知道一个短语可能以什么开头，以及它可能被什么跟随。</p>
<footer>—— 唐纳德 · 克努斯（Donald E. Knuth）谈自顶向下分析的前瞻思想</footer>
</div>

<div class="article-byline">
<p>第三级 · 编译原理 ｜ 《编译原理》（龙书）§4.4.2 ｜ 2026-08-07</p>
</div>

## 为什么从 FIRST/FOLLOW 集开始

上一节说，预测分析能「看一眼」就决定用哪条产生式，前提是两套集合算得清：一个串可能**以什么记号开头**，一个非终结符**后面可能跟什么记号**。这两套集合分别叫 FIRST 集与 FOLLOW 集——它们是自顶向下分析的「情报系统」，也是 LL(1) 判定、分析表构造的共同地基。

这两个名字值得先记一句话：**FIRST 管「开头」，FOLLOW 管「后面」**。FIRST 用来在「同一条产生式的多个候选」里挑，FOLLOW 用来在「这个非终结符可以放过去（ε）」时兜底。<span class="marginnote">FOLLOW 的一个冷门但重要的用途：它是「这个位置允许哪些记号」的合法性清单，错误恢复时能用来做同步点——分析器报错后跳到 FOLLOW 记号再继续，比胡乱跳过可靠得多。</span>

## 1 FIRST 集：一个串能以什么开头

对任意文法符号串 $\alpha$，**FIRST($\alpha$)** 是所有「$\alpha$ 能推导出的、以某个终结符开头的句子」的首终结符集合；若 $\alpha \Rightarrow^* \varepsilon$，则 $\varepsilon$ 也属于 FIRST($\alpha$)。

对单个符号递归定义：

若 $X$ 是终结符，$\text{FIRST}(X) = \{X\}$。
若 $X$ 是非终结符，对每条产生式 $X \to Y_1Y_2\cdots Y_k$：
先把 FIRST($Y_1$) 里除 $\varepsilon$ 外的元素加入 FIRST($X$)；
若 $Y_1 \Rightarrow^* \varepsilon$，再把 FIRST($Y_2$) 里除 $\varepsilon$ 外的加入；以此类推；
若 $Y_1\cdots Y_k$ 全部能导出 $\varepsilon$，则 $\varepsilon \in \text{FIRST}(X)$。

直观理解：**FIRST 是「第一个字符能是什么」**。终结符的 FIRST 是它自身（如 `+` 的 FIRST 是 {+}），非终结符的 FIRST 是它全部候选产生式导出的首终结符集合。<span class="marginnote">串的 FIRST 由「打头符号」一路推下去：从左往右扫，若 Y1 不能导出 ε，串的 FIRST 就是 FIRST(Y1)；Y1 能导出 ε 就接着看 Y2。这个「从左往右试、遇 ε 就下一位」的动作会反复出现。</span>

## 2 FOLLOW 集：一个非终结符后面可能跟什么

对非终结符 $A$，**FOLLOW($A$)** 是在**某些句型**中紧随 $A$ 之后可能出现**终结符**的集合；若 $A$ 出现在某句型末尾（可以是 $S$ 的位置），则 $A$（输入结束标记）属于 FOLLOW($A$)。

计算规则（对开始符号与每条产生式 $B \to \alpha A \beta$）：

1. 若 $A = S$（开始符号），则 $`$$` ∈ FOLLOW($S$)。
2. 对产生式 $B \to \alpha A \beta$：FIRST($\beta$) 中除 $\varepsilon$ 外的元素都属于 FOLLOW($A$)。
3. 若 $B \to \alpha A \beta$ 且 $\beta \Rightarrow^* \varepsilon$（即 $\beta$ 能消失），则 FOLLOW($B$) 全部并入 FOLLOW($A$)。

注意 FOLLOW 只对**非终结符**定义，且**不含 $\varepsilon$**——它回答「后面必须跟一个真实记号（或结束）」。<span class="marginnote">规则 3 是 FOLLOW 计算的「传染」关键：若 β 能消失且 β 在一串的末尾，那么 A 后面能跟什么，B 后面就能跟什么。读者可以在表达式文法上验算：FOLLOW(E') 会被 E 的位置传导。</span>

## 3 计算算法：不动点迭代

FIRST 与 FOLLOW 都是**最小不动点**：从空集出发，反复套用规则，直到集合不再增长。因为文法符号有限、每个集合的候选终结符有限，迭代必然终止。

以经典表达式文法为例手算 FOLLOW：

$$\begin{aligned} E &\to T\,E' & E' &\to +T\,E' \mid \varepsilon \\ T &\to F\,T' & T' &\to \times F\,T' \mid \varepsilon \\ F &\to (E) \mid \textbf{id} \end{aligned}$$

- $`$$` ∈ FOLLOW($E$)。
- 由 $F \to (E)$：`)` ∈ FOLLOW($E$)。
- 由 $E \to T\,E'$：FOLLOW($E$) ⊆ FOLLOW($E'$)，且 FIRST($E'$) ∖ {ε} = {`+`} ⊆ FOLLOW($T$)。
- 由 $E' \to +T\,E'$：FOLLOW($E'$) ⊆ FOLLOW($T$)，FOLLOW($T$) 传导给 $T'$，再传给 $F$。

不断迭代直到收敛，得到 FOLLOW($E$)=FOLLOW($E'$)={\$, )}，FOLLOW($T$)=FOLLOW($T'$)={\$, ), +}，FOLLOW($F$)={\$, ), +, ×}。<span class="marginnote">这套「从空集开始、规则不断补充、直到饱和」的算法，是编译里数据流分析的雏形——第八篇的到达定值、活跃变量分析，用的正是同一套不动点思想。先在这里种下这颗种子。</span>

## 4 公式解析：FIRST 的传递规则

把 FIRST 的计算浓缩成一条「决策链」。对 $X \to Y_1Y_2\cdots Y_k$：

$$\text{FIRST}(X) \supseteq \bigcup_{j=1}^{t} (\text{FIRST}(Y_j) \setminus \{\varepsilon\}), \quad t = \max\{j \mid Y_1,\ldots,Y_{j-1} \Rightarrow^* \varepsilon\}$$

- **第一步，从左扫**：先取 $Y_1$ 的 FIRST 去 ε。若 $Y_1$ 能导出 ε，就继续看 $Y_2$；否则停在这里。
- **第二步，延伸到最长前缀**：$t$ 是「连续能导出 ε」的最长前缀长度。把这些符号的 FIRST 全并进来，但每个都先挖掉 $\varepsilon$。
- **第三步，ε 的归属**：只有当前缀「一路通到底」（全部 $Y_1\cdots Y_k$ 都能导出 ε）时，$\varepsilon$ 才属于 FIRST($X$)。ε 不是随便给的，它代表「整个串可以消失」。

**这条链解释了「为什么 ε 这么麻烦」**：它让 FIRST 计算必须看一串符号而不只是一个，也让 LL(1) 判定多了「ε 候选对 FOLLOW 避让」这条规则。ε 是语言的「空」，是分析里反复要小心处理的特例。

## 5 FIRST/FOLLOW 的用途一览

这两套集合支撑起整个自顶向下技术栈：

| 用途 | 依赖 | 作用 |
| --- | --- | --- |
| LL(1) 文法判定 | FIRST、FOLLOW | 检查候选 FIRST 两两不相交、ε 候选对 FOLLOW 避让 |
| 预测分析表构造 | FIRST、FOLLOW | 表项 $M[A,\ a]$ 填哪条产生式 |
| 错误恢复同步 | FOLLOW | 非法记号后跳到「本非终结符之后该出现的记号」 |
| 自底向上分析的辅助 | FIRST | 计算 LR 项集时的先行记号 |

**辨析｜易错点：** FOLLOW 集合**不含 ε**，但它对**每个**非终结符都定义——包括那些从不出现在句子末尾的。ε 属于 FIRST 的可能，属于 FOLLOW 的永远不可能；把 ε 误放进 FOLLOW 是初学者的高频错误。

## 7 思考与练习

**练习 1 手算 FIRST**：对文法 $E \to T E'$、$E' \to +T E' \mid \varepsilon$、$T \to F T'$、$T' \to \times F T' \mid \varepsilon$、$F \to (E) \mid \textbf{id}$，手算每个非终结符的 FIRST 集，再算 $\text{FIRST}(T E')$ 与 $\text{FIRST}(+ T E')$——体会「串的 FIRST 由打头符号决定」。

**练习 2 手算 FOLLOW**：对同一文法手算 FOLLOW，特别注意「ε 传播」的规则——E′ 为什么继承 E？E′ 为什么含 `)`？

**练习 3 ε 的归属**：构造一个含 $\varepsilon$ 产生式的文法，手动走一遍 FIRST——验证「只有整个串都能导出 ε，ε 才属于 FIRST」这条规则。若中间有个符号不能导出 ε，ε 会不会「误入」？

**练习 4 交叉验证**：用练习 1、2 的结果，检查文法是否满足 LL(1) 条件（候选 FIRST 不相交、ε 候选对 FOLLOW 避让）——为第十二节做准备。<span class="marginnote">练习 4 是把 FIRST/FOLLOW 从「计算」接到「使用」：算出集合后，下一步就是用它判定 LL(1)、填预测分析表。先算好，下一节的表就「所见即所得」。</span>

**练习 5 不动点直觉**：解释为什么「空集起步、规则不断补充、直到饱和」的迭代必然终止——提示：文法符号有限、每个集合的元素有限，且只增不减。

**练习 6 FOLLOW 不含 ε**：论证为什么 FOLLOW 集合永远不含 ε——即使某非终结符总是「被放过去」，它后面也该跟一个真实记号（或 `$` 输入结束）。提示：FOLLOW 回答「后面必须跟什么」，ε 不是「后面跟的东西」。

**延伸**：FIRST/FOLLOW 的「不动点迭代」与第八篇「到达定值」的迭代是同一套数学——找两个分析的共同结构（空集起步、单调增、有界收敛），在第三级《离散数学》的「闭包」概念里找理论源头。

## 8 小结

- **FIRST($\alpha$)**：$\alpha$ 能推导出的句子以什么终结符开头；$\alpha \Rightarrow^* \varepsilon$ 时含 ε。
- **FOLLOW($A$)**：句型中紧随 $A$ 之后的终结符集合（含输入结束 `$`），**永远不含 ε**。
- 计算规则：FIRST 从左往右扫符号、遇 ε 下一位；FOLLOW 靠 `$` 起点、FIRST 喂入、ε 传播三条规则。
- 两者都是**不动点算法**：空集起步、迭代到饱和，与第八篇的数据流分析同源。
- 用途：LL(1) 判定、预测分析表构造、错误恢复同步、LR 先行记号。

在下一节，我们把 FIRST/FOLLOW 装进一张表：**LL(1) 文法与预测分析表**——预测分析由「能预测」到「照着表走」。
