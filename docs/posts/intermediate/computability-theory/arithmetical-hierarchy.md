---
title: 算术分层
date: 2026-08-07
---

# 算术分层

<div class="epigraph">
<p>一个集合属于算术分层的位置，刻画了认识它的算法代价。</p>
<footer>—— 斯蒂芬 · 克利尼（Stephen C. Kleene，*Recursive Predicates and Quantifiers\*, 1943）</footer>
</div>

<div class="article-byline">
<p>第二级 · 可计算性理论（递归论） ｜ R. I. Soare, *R. E. Sets and Degrees\*, §IV.1–§IV.3 ｜ 2026-08-07</p>
</div>

## 为什么从算术分层开始

前几篇里，"不可判定"似乎是一顶非黑即白的帽子：可判定、或不可判定。但第 6 篇的跳跃塔已经暗示，不可判定性**有程度**：$K$ 比可判定集难，$K'$ 比 $K$ 难。这一篇给出精确的坐标系统——**算术分层（arithmetical hierarchy）**：它用一阶算术公式的量词结构，把所有"可用自然数语言定义的集合"排成一个无限层谱。<span class="marginnote">算术分层把两件事统一起来：一方面，它是<strong>句法</strong>的（公式的量词层数）；另一方面，它精确对应<strong>计算</strong>的（跳跃塔的层数）。"句法即复杂度"这个主题贯穿整个逻辑与计算机科学的接口，也是第四级《大模型原理》里"表达力分层"思想的老祖宗。</span>看懂它，你就拥有了一把度量"不可判定程度"的标准尺。

## 1 用一阶算术说"集合"

**核心概念**：**算术公式（arithmetical formula）** 是只允许讨论自然数的公式，其中的量词只作用于自然数，谓词只含加法、乘法、等号与基本算术关系（$\lt $, $|$ 整除等）。例如

$$\varphi(n) :\equiv \exists k\ (n = 2k)$$

说"$n$ 是偶数"；$\exists k\ \exists j\ (n = kj \land k > 1 \land j > 1)$ 说"$n$ 是合数"。每个算术公式 $\varphi(n)$ 定义了一个集合 $\{ n \mid \varphi(n) \text{ 为真} \}$。

关键在于**量词的嵌套方式**。量词可以"交替"：$\forall x\ \exists y\ \cdots$ 是 $\forall$ 在外、$\exists$ 在内的一次交替。交替层数决定了集合在分层中的位置。**开头那个量词**与**交替次数**是分类的两个依据。

## 2 $\Sigma_n$ 与 $\Pi_n$：量词层谱

把算术公式按"开头的量词"与"交替次数"分类：

**核心概念**：一个公式若可化为**开头一个 $\exists$、共 $n$ 层量词交替**的形式（最外层是 $\exists$），则称它是 **$\Sigma_n$ 公式**；若最外层是 $\forall$，则称它是 **$\Pi_n$ 公式**。$\Sigma_n$ 公式所定义的自然数集合称为 $\Sigma_n$ 集；$\Pi_n$ 公式定义的集合称为 $\Pi_n$ 集。

$$\Sigma_1:\ \exists y_1\ R(y_1), \qquad \Pi_1:\ \forall y_1\ R(y_1)$$
$$\Sigma_2:\ \exists y_1\ \forall y_2\ R(y_1, y_2), \qquad \Pi_2:\ \forall y_1\ \exists y_2\ R(y_1, y_2)$$

其中 $R$ 是（可计算判定的）无界量词公式。这里有个很漂亮的对应：**"存在一个证据"是 $\Sigma$，"所有情况都成立"是 $\Pi$**。例如：

- "$n$ 是合数" = $\exists$ 分解——**$\Sigma_1$**。
- "$n$ 是素数" = $\forall$ 平凡分解——**$\Pi_1$**。
- "停机问题 $K$" = $\exists t$（程序在 $t$ 步内停机）——**$\Sigma_1$**。

$\Sigma_1$ 集正是 c.e. 集，$\Pi_1$ 集正是 c.e. 集的补集（co-c.e.）——上一节的内容在分层里找到了自己的位置。<span class="marginnote">"$\Sigma_1$ = c.e.、$\Pi_1$ = co-c.e."是最重要的一条对应：半可判定（$\exists$ 证据）在分层语言里就是"存在一个枚举步骤"。</span>

## 3 $\Delta_n$ 与分层定理

**核心概念**：若一个集合既是 $\Sigma_n$ 的又是 $\Pi_n$ 的，则称它是 **$\Delta_n$ 集**：

$$\Delta_n = \Sigma_n \cap \Pi_n$$

$\Delta_n$ 是"两侧都能定义"的集合。最重要的两条分层定理：

**定理（分层递进）**：对每个 $n$，
$$\Sigma_n \subsetneq \Sigma_{n+1}, \qquad \Pi_n \subsetneq \Pi_{n+1}, \qquad \Sigma_n \cup \Pi_n \subsetneq \Delta_{n+1}$$

即每一层都严格大于上一层，且 $\Sigma_n$ 与 $\Pi_n$ 的并集严格包含在 $\Delta_{n+1}$ 中。分层是**真分层**——每爬一层都有新的集合出现。<span class="marginnote">证明每层严格包含用对角化：构造一个"$\Sigma_{n+1}$ 中但不在 $\Pi_n$ 中的集合"，方法与停机问题的对角线一脉相承——先假设 $\Pi_n$ 枚举全了，再构造一个"反转自己"的集合。</span>

**定理（$\Delta_1$ = 可计算）**：一个集合是可判定的，当且仅当它是 $\Delta_1$ 集。

这后一条是递归论与算术分层之间的"焊接点"：$\Delta_1$ 精确等于可计算集。于是整个分层以"可计算"为底座，一层层堆向更高的不可判定性。

## 4 公式解析：$K$ 落在分层的哪里

用分层语言重写停机问题，能看出它的"难度指纹"。$K = \{ e \mid \varphi_e(e)\!\downarrow \}$：

$$e \in K \iff \exists t\ \big(\text{第 } e \text{ 个程序在输入 } e \text{ 上 } t \text{ 步内停机}\big)$$

- **第一步，去量词**：$e \in K$ 等价于"存在步数 $t$，使得程序 $e$ 在 $t$ 步内停机"。这里"$t$ 步内停机"是一个可计算的有限检查（跑 $t$ 步，看是否进入接受态），所以括号内是可判定谓词 $R(e, t)$。
- **第二步，数量词**：整个句子只有**一个存在量词** $\exists t$，前面没有 $\forall$——所以 $K$ 是 $\Sigma_1$ 集。这与第 3 篇"$K$ 是 c.e."完全吻合。
- **第三步，看补集**：$\overline{K}$：$e \in \overline{K} \iff \forall t\ \big(\text{第 } e \text{ 个程序在输入 } e \text{ 上 } t \text{ 步内不停机}\big)$。一个 $\forall$ 打头——$\Pi_1$ 集。
- **第四步，定位**：$K \in \Sigma_1 \setminus \Pi_1$（$\Sigma_1$ 中但非 $\Pi_1$，因为 $\Pi_1$ 是 co-c.e.，而 $K$ 的补不是 c.e.），$\overline{K} \in \Pi_1 \setminus \Sigma_1$。于是 $\Delta_1 = \Sigma_1 \cap \Pi_1$ 正好把 $K$ 排除在外——**$K$ 不可判定，在分层里被"卡"在 $\Sigma_1$ 与 $\Pi_1$ 之间**。

## 5 完备集定理：分层与跳跃塔对齐

分层不是孤立的句法游戏，它和第 6 篇的跳跃塔精确对接：

**定理（完备集定理）**：对每个 $n$，存在 $\Sigma_n$ 完备集 $S_n$——所有 $\Sigma_n$ 集都 $\le_m$ 归约到它——且 $\deg(S_n) = \mathbf{0}^{(n)}$。

第 $n$ 层"最难"的 $\Sigma_n$ 集，其 Turing 度恰是跳跃塔的第 $n$ 级 $\mathbf{0}^{(n)}$。于是：

| 分层位置 | 计算含义 | 代表集合 |
| --- | --- | --- |
| $\Sigma_1$ | c.e. | $K$（停机问题） |
| $\Pi_1$ | co-c.e. | $\overline{K}$ |
| $\Delta_2$ | 以 $K$ 为神谕可判定 | 各种 $\limsup$ 类集合 |
| $\Sigma_2$ | 相对 $K$ 的半判定 | $K'$、$\operatorname{Tot}$（全函数问题） |

**辨析｜易错点：** $\Sigma_n$ 完备集的度是 $\mathbf{0}^{(n)}$，但**不是每个** $\Sigma_n$ 集都有度 $\mathbf{0}^{(n)}$——完备集是"最难"，一般集合可能落在更低的度。<span class="marginnote">完备集定理是"从下往上看"的坐标：想判断一个集合有多难，就看它属于哪一层、是不是该层完备的。这与第三级《算法设计与分析》里"NP 完备是 NP 中最难"的定位思路完全同构。</span>

## 6 算术分层的地图

把分层画成一张地图，前几篇的所有主角都各就各位：

$$\Sigma_1 \cup \Pi_1 \subset \Delta_2 \subset \Sigma_2 \cup \Pi_2 \subset \Delta_3 \subset \cdots$$

- $\Delta_1$：可计算集（底座）。
- $\Sigma_1 \setminus \Pi_1$：c.e. 但不可判定（如 $K$）。
- $\Pi_1 \setminus \Sigma_1$：co-c.e. 但不可判定（如 $\overline{K}$）。
- $\Delta_2$：以停机问题为神谕可判定的集合（"半可判定 + 半可补判定"）。
- 每上一层，都需要再跳跃一次。

这条谱系还向两端延伸：向下，$\Delta_0$ 是"无界量词"以外的有界可计算；向上，越过全部有限层就进入**分析分层**（集合的量词），那是第二级《公理集合论与模型论》与数学基础的地盘。<span class="marginnote">算术分层与哥德尔不完备性一脉相承：哥德尔证明了"真"不是 $\Sigma_1$ 可枚举的——真算术（truth of arithmetic）甚至不在任何 $\Sigma_n$ 层里，它属于分层的"外部"。这是《数学哲学》与第四级逻辑相关章节的连接点。</span>

## 8 术语速查表

| 术语 | 英文 | 含义 | 出处 |
| --- | --- | --- | --- |
| 算术公式 | arithmetical formula | 只含自然数量词与算术谓词的公式 | §1 |
| $\Sigma_n$ 集 | $\Sigma_n$ set | 由 $\exists$ 开头、$n$ 层交替公式定义的集合 | §2 |
| $\Pi_n$ 集 | $\Pi_n$ set | 由 $\forall$ 开头、$n$ 层交替公式定义的集合 | §2 |
| $\Delta_n$ 集 | $\Delta_n$ set | 既是 $\Sigma_n$ 又是 $\Pi_n$ 的集合 | §3 |
| 量词交替 | quantifier alternation | 量词从 $\forall$ 到 $\exists$（或反向）的换层 | §2 |
| 半可判定 | semi-decidable | 存在量词 = 存在证据，$\Sigma_1$ = c.e. | §2 |
| 完备集定理 | completeness theorem | $\Sigma_n$ 完备集的度恰为 $\mathbf{0}^{(n)}$ | §5 |
| 真算术 | true arithmetic | 全体算术真命题，超越一切 $\Sigma_n$ 层 | §6 |
| 分析分层 | analytical hierarchy | 对集合量词的分层，算术分层的上方延伸 | §6 |

## 9 小结

- **算术公式**用自然数量词定义集合；量词的**开头**与**交替层数**决定集合的层次。
- $\Sigma_n$（$\exists$ 开头）与 $\Pi_n$（$\forall$ 开头）是分层的两翼；**$\Delta_n = \Sigma_n \cap \Pi_n$**。
- **$\Delta_1$ = 可计算集**；**$\Sigma_1$ = c.e. 集**；**$\Pi_1$ = co-c.e. 集**。
- 分层**严格递增**：$\Sigma_n \subsetneq \Sigma_{n+1}$ 等，每层都有新集合。
- **完备集定理**把分层与跳跃塔对齐：$\Sigma_n$ 完备集的度恰是 $\mathbf{0}^{(n)}$