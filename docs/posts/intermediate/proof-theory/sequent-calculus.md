---
title: Gentzen 序贯演算（LK / LJ）
date: 2026-08-07
---

# Gentzen 序贯演算（LK / LJ）

<div class="epigraph">
<p>关于逻辑推演，关键的事实是：每个证明都可以通过消除「截断」而化为一种范式。</p>
<footer>—— 格哈德 · 根岑（Gerhard Gentzen），《关于逻辑推演的研究》（1935）</footer>
</div>

<div class="article-byline">
<p>第二级 · 证明论 ｜ A. S. Troelstra & H. Schwichtenberg《Basic Proof Theory》第3章 ｜ 2026-08-07</p>
</div>

## 为什么从序贯演算开始

自然演绎贴近人的思维，但它在技术上有一个「不干净」的地方：**消去假设的动作藏在方括号里**，规则的真前提与假前提混在一起。根岑要证明他的主定理（截消定理），需要一种把一切都摊在明面上的形式系统——于是他发明了**序贯演算（sequent calculus）**。<span class="marginnote">序贯演算的原始设计目标不是「自然」，而是「适合证明关于证明的定理」。它像证明论的汇编语言：结构规则、逻辑规则、截规则各司其职，每个动作都可被精确计量。自然演绎与序贯演算的对照，正是下一篇截消定理的全部起点。</span>

如果把自然演绎比作高级语言，序贯演算就是它的中间表示。同一个证明，在自然演绎里是层层嵌套的推导树，在序贯演算里变成一行行清晰的「左假设右结论」推导；而自然演绎里隐含的「消去」，在序贯演算里成为一条显式的**截规则（cut rule）**。正是这条截规则可以被消掉，才带来证明论一系列深刻的结论。

## 1 序贯是什么

序贯演算的基本对象是**序贯（sequent）**：一种形如

$$
\Gamma \Rightarrow \Delta
$$

的记号，其中 $\Gamma$ 与 $\Delta$ 是**有限个公式**组成的列表，$\Rightarrow$ 读作「蕴涵于」或「推演出」。$\Gamma$ 叫**前件（antecedent）**，$\Delta$ 叫**后件（succedent）**。<span class="marginnote">注意区分两层记号：$\Rightarrow$ 是序贯内的元记号，属于「元语言」；$\to$ 是公式内的联结词，属于「对象语言」。这个区分在证明论里必须时刻清醒，否则就会混淆「推导」与「蕴含」——前者可被截断，后者只能被分离。</span>

**序贯的经典语义**：$\Gamma \Rightarrow \Delta$ 表示「若 $\Gamma$ 中所有公式都为真，则 $\Delta$ 中至少有一个公式为真」。当 $\Delta$ 为空时，它退化为「$\Gamma$ 不一致」；当 $\Gamma$ 为空时，它表示「$\Delta$ 中至少有一个可真」——特别地，$\Rightarrow A$ 就是「$A$ 可证」。语义上的这个定义把「推导」翻译成「真值条件下的必然性」，它正是完全性定理要来回校准的靶子。

## 2 结构规则

序贯演算的头一组规则与任何联结词无关，它们操纵的是**结构的形状**——公式的增减、交换、合并。这组规则叫**结构规则（structural rules）**，共四条：

| 规则 | 左式变形 | 右式变形 |
| --- | --- | --- |
| 弱化（weakening） | 前件可任意加公式：$\Gamma \Rightarrow \Delta$ 推出 $\Gamma, A \Rightarrow \Delta$ | 后件可任意加公式：$\Gamma \Rightarrow \Delta$ 推出 $\Gamma \Rightarrow \Delta, A$ |
| 收缩（contraction） | 前件重复可合并：$\Gamma, A, A \Rightarrow \Delta$ 推出 $\Gamma, A \Rightarrow \Delta$ | 后件重复可合并：$\Gamma \Rightarrow \Delta, A, A$ 推出 $\Gamma \Rightarrow \Delta, A$ |
| 交换（exchange） | 前件顺序无关：$\Gamma, A, B, \Delta \Rightarrow \Theta$ 推出 $\Gamma, B, A, \Delta \Rightarrow \Theta$ | 后件顺序无关：$\Gamma \Rightarrow \Delta, A, B, \Theta$ 推出 $\Gamma \Rightarrow \Delta, B, A, \Theta$ |
| 截断（cut） | 见第 5 节公式解析 | 见第 5 节公式解析 |

结构规则在自然演绎里是被「内化」的：假设可重复使用（对应收缩）、可闲置（对应弱化）、顺序可换（对应交换）。把它们单独拎出来，是序贯演算最关键的发明——**一旦某条结构规则被拿走，逻辑的性质就会剧烈改变**。<span class="marginnote">拿走收缩与弱化，得到的就是<strong>线性逻辑（linear logic）</strong>——公式像资源一样只能使用一次，对应函数式程序里「每次调用恰好消耗一次」的价值观。这个联系到第 9 篇 Curry–Howard 对应时会再次浮出水面。</span>

## 3 逻辑规则：左规则与右规则

每个联结词在序贯演算里都有两条规则：**右引入（right introduction）**在前件不动的情况下把公式放进后件，**左引入（left introduction）**把公式放进前件。右规则对应自然演绎的引入，左规则对应消去。以核心联结词为例：

| 联结词 | 右引入 | 左引入 |
| --- | --- | --- |
| 合取 | $\dfrac{\Gamma \Rightarrow \Delta, A \qquad \Gamma \Rightarrow \Delta, B}{\Gamma \Rightarrow \Delta, A \land B}$ | $\dfrac{\Gamma, A \Rightarrow \Delta}{\Gamma, A \land B \Rightarrow \Delta}$（及 $B$ 版） |
| 析取 | $\dfrac{\Gamma \Rightarrow \Delta, A}{\Gamma \Rightarrow \Delta, A \lor B}$（及 $B$ 版） | $\dfrac{\Gamma, A \Rightarrow \Delta \qquad \Gamma, B \Rightarrow \Delta}{\Gamma, A \lor B \Rightarrow \Delta}$ |
| 蕴含 | $\dfrac{\Gamma, A \Rightarrow \Delta, B}{\Gamma \Rightarrow \Delta, A \to B}$ | $\dfrac{\Gamma \Rightarrow \Delta, A \qquad \Gamma, B \Rightarrow \Delta}{\Gamma, A \to B \Rightarrow \Delta}$ |
| 否定 | $\dfrac{\Gamma, A \Rightarrow \Delta}{\Gamma \Rightarrow \Delta, \neg A}$ | $\dfrac{\Gamma \Rightarrow \Delta, A}{\Gamma, \neg A \Rightarrow \Delta}$ |

量词也各有左右规则，只是多了一条变量条件：全称右引入 $\dfrac{\Gamma \Rightarrow \Delta, A(x)}{\Gamma \Rightarrow \Delta, \forall x\,A(x)}$ 要求 $x$ 不出现在 $\Gamma \cup \Delta$ 的任何公式中；存在左引入 $\dfrac{A(y), \Gamma \Rightarrow \Delta}{\exists x\,A(x), \Gamma \Rightarrow \Delta}$ 要求 $y$ 是全新变量。这正是自然演绎里 $\forall$ 引入条件的序贯版——同一个约束，换了位置而已。

注意右引入的规则形状非常规整：**每个右引入都要「消耗」一个同位置的公式**（合取消耗后件里的 $A$ 与 $B$，蕴含把 $A$ 挪到前件、把 $B$ 留在后件）。这种「公式从下到上被逐层拆开」的性质，是截消定理与子公式性质的根源——推导的每一步都只引入子公式，不会凭空造出全新的公式。

## 4 从推导树到序贯推导

自然演绎的每个证明都能机械地翻译成序贯推导。以 $(A \land B) \to (B \land A)$ 为例，序贯版本的推导如下：

$$
\frac{\dfrac{A \land B \Rightarrow A \land B}{A \land B \Rightarrow B} \qquad \dfrac{A \land B \Rightarrow A \land B}{A \land B \Rightarrow A}}{A \land B \Rightarrow B \land A}
\;\Rightarrow\;
\frac{A \land B \Rightarrow B \land A}{\Rightarrow (A \land B) \to (B \land A)}
$$

关键转变在于：**自然演绎里的「假设」变成了序贯里的前件公式**。$A \land B \Rightarrow A \land B$ 是一个公理（同一序贯），之后用合取左规则把它拆成 $B$ 与 $A$，再用合取右规则拼回 $B \land A$，最后用蕴含右规则把 $A \land B$ 挪到前件变成结论。<span class="marginnote">翻译揭示了自然演绎与序贯演算的对应关系：假设列表 = 前件，待证结论 = 后件，消去假设 = 蕴含右引入（公式从前件挪走）。这也解释了为何 LJ 限制后件至多一个公式——它正好对应直觉主义自然演绎里「一次只证一个结论」的形态。</span>

## 5 公理与推导的阅读方向（辨析）

序贯演算里唯一的出发点是一组**公理**，最典型的是**同一公理（identity axiom）**：

$$
\Gamma, A \Rightarrow A, \Delta
$$

它读作「前件里的 $A$ 蕴含后件里的 $A$」，无论周围的 $\Gamma$、$\Delta$ 是什么都成立——因为若 $A$ 真则 $A$ 真，是平凡的同一性。整棵序贯推导树就是把公理反复向上「翻折」成结论的过程。

**辨析｜易错点：** 初学者常以为序贯推导像自然演绎一样「从上往下」构造——先写前提再得结论。实际在证明搜索里，序贯推导几乎总是**自底向上**构造的：从目标 $\Rightarrow A$ 出发，问「哪条规则能产生它」，于是逐步把 $A$ 拆成更小的子公式，直到撞上公理。这个「反向」的习惯至关重要：**右引入规则向上读是分解，向下读才是合成**。自动定理证明（一阶逻辑的 prover）偏爱序贯演算而非自然演绎，正是因为这种反向搜索可以被机械执行。

两种形式化的差别，可以用一张表收束：

| 维度 | 自然演绎 | 序贯演算 |
| --- | --- | --- |
| 证明形态 | 嵌套推导树 | 逐层展开的序贯树 |
| 假设处理 | 方括号内隐式消去 | 前件公式显式搬移 |
| 捷径机制 | 蕴含消去（隐含） | 截规则（显式） |
| 典型用途 | 人的数学证明 | 证明论与自动推理 |

## 6 公式解析：cut 规则

序贯演算里最核心的规则是**截规则（cut）**，它是自然演绎中「消去」动作在序贯里的显式化身：

$$
\frac{\Gamma \Rightarrow \Delta, A \qquad A, \Gamma' \Rightarrow \Delta'}{\Gamma, \Gamma' \Rightarrow \Delta, \Delta'}
$$

分四步拆解：

- **第一步，读左前提**：$\Gamma \Rightarrow \Delta, A$ 说「由 $\Gamma$ 可推出 $\Delta$ 或 $A$」。经典语义下这是「$\Delta$ 与 $A$ 至少一个成立」。
- **第二步，读右前提**：$A, \Gamma' \Rightarrow \Delta'$ 说「若 $A$ 与 $\Gamma'$ 都成立，则 $\Delta'$ 成立」。
- **第三步，拼接**：若 $A$ 成立，右前提给出 $\Delta'$；若 $A$ 不成立，左前提必须给出 $\Delta$。两种情况都保证「$\Gamma, \Gamma'$ 推出 $\Delta, \Delta'$」，于是截规则合法。
- **第四步，看代价**：结论里公式 $A$ 完全消失——它只在两个前提中出现，充当「拼接胶水」。$A$ 被称为**截公式（cut formula）**，可以非常复杂，甚至可以不在最终结论的子公式中出现。正是这条「凭空而来又凭空消失」的规则，破坏了子公式性质。

**cut 规则在演绎中扮演的是「捷径」角色**：先用一个引理 $A$，再用这个引理。截消定理将证明：凡是用截规则拼出的证明，都能改写成不用截规则的证明——即任何「借道引理」的证明都可以被「拉直」。拉直的过程不是免费的，它会付出推导变长的代价，但换来的是结构与可控性。

## 7 LK 与 LJ 的差别

根岑给出了两套序贯演算：**LK** 用于经典逻辑，**LJ** 用于直觉主义逻辑。两者的全部规则几乎相同，唯一的差别是：**LJ 限制后件至多包含一个公式**。<span class="marginnote">「至多一个」意味着后件列表要么空、要么恰好一个公式。由此，排中律 $A \lor \neg A$ 在 LJ 里推不出来——它的推导需要后件同时保留 $A$ 与 $\neg A$ 两个位置，被单后件限制直接堵死。这与自然演绎里「需要双重否定消去」是同一个现象的两种面孔。</span>

具体地，LJ 的蕴含右规则变成 $\dfrac{\Gamma, A \Rightarrow B}{\Gamma \Rightarrow A \to B}$（后件只留一个 $B$），收缩、弱化、交换在两侧都保留，但经典特有的多公式后件推理——比如「由 $\Gamma \Rightarrow \Delta, A \lor \neg A$ 用析取右规则」——在 LJ 中无路可走。**LK 与 LJ 共享几乎全部规则却相差一个微小的形状限制，这一对比是理解经典与直觉主义差异的最佳显微镜**。

## 8 小结

- **序贯**是「前件 $\Gamma$ 蕴涵后件 $\Delta$」的记号，$\Gamma \Rightarrow \Delta$ 在经典语义下表示「$\Gamma$ 全真则 $\Delta$ 至少一真」。
- **结构规则**（弱化、收缩、交换、截断）操纵证明的形状，与联结词无关；删去收缩与弱化就得到线性逻辑。
- 每个联结词各有**左引入与右引入**两条规则，右引入对应自然演绎的引入，左引入对应消去。
- **截规则**用中间公式 $A$ 拼接两个证明，结论里 $A$