---
title: 多项式层次 PH
date: 2026-08-07
---

# 多项式层次 PH

<div class="epigraph">
<p>人的手必须伸得比够得着更远——否则天堂何用？</p>
<footer>—— 罗伯特 · 勃朗宁（Robert Browning, "Andrea del Sarto"）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算理论（可计算性与计算复杂性） ｜ Arora & Barak《计算复杂性：一种现代方法》第5章 ｜ 2026-08-07</p>
</div>

## 为什么量词要堆叠

NP 的语言可以写成「$\exists$ 一个证书，验证器接受」；
coNP 是「$\forall$ 证书，验证器拒绝」。
但现实问题往往需要**交替**的量词：「是否存在一个策略，使得**无论**对手怎么走，我都能赢？」
——这就是「$\exists \forall$」问题，比 NP 更复杂。
**多项式层次（polynomial hierarchy, PH）** 就是按量词交替的层数，把「带交替量词的可判定问题」排成无限阶梯。
<span class="marginnote">多项式层次由 Meyer 与 Stockmeyer 在 1972 年提出。
直观地，$\Sigma_2$ 对应「两轮博弈：我先走（∃），你任意走（∀），检验胜负」；
$\Sigma_3$ 是三轮……每多一层，就多一轮「应对任意对手」。
</span>

在课程主线上，PH 是一个「脚手架」：它把 P、NP、coNP 挂到无限延伸的梯子上，并提供了绝佳的**塌缩语言**——「若 $P = NP$，则整座层次塌成一点」。
它是衡量「你比 NP 强多少」的标尺。

## 1 交替量词的定义

**$\Sigma_i$ 与 $\Pi_i$**（$i \ge 1$）：设 $M$ 是多项式时间判定器，定义

$$
L \in \Sigma_i \iff \exists \text{ 多项式 } q, \exists M:\ \ w \in L \iff \exists x_1\, \forall x_2\, \exists x_3 \cdots Q_i x_i:\ M(w, x_1, \dots, x_i) = accept
$$

其中 $|x_j| \le q(|w|)$，量词 $Q_i$ 依 $i$ 奇偶交替（$i$ 奇数为 $\exists$，偶数为 $\forall$）。
$\Pi_i$ 的定义把最外层的 $\exists$ 换成 $\forall$。

**基例**：$\Sigma_1 = NP$（一个 $\exists$ 证书），$\Pi_1 = coNP$（一个 $\forall$）。
<span class="marginnote">「量词的串长受多项式约束」很关键：$\exists x_1$ 中的 $x_1$ 不能长得离谱，否则「存在性」就失去多项式意义。
这与 NP 证书长度多项式有界一脉相承。
</span>

**多项式层次**：

$$
PH = \bigcup_{i \ge 1} \Sigma_i = \bigcup_{i \ge 1} \Pi_i
$$

**重点：PH 的每一层都在「上一层的答案上加一个量词」。**
 $\Sigma_2 = NP^{NP}$——带 NP oracle 的机器能解的东西，直觉上等于「$\exists\forall$」的布尔结构。

## 2 公式解析：Σ₂ 的博弈味道

看第二层 $\Sigma_2$，它的形状最常出现：

$$
w \in L \iff \exists x\, \forall y:\ M(w, x, y) = accept
$$

逐项拆解：

- **第一步，读「$\exists x$」**：存在一个「策略/承诺」$x$——比如「我有一着棋」。
- **第二步，读「$\forall y$」**：对**任何**对手回应 $y$——「无论你怎么走」。
- **第三步，读整体语义**：$w \in L$ 当且仅当「我有一招，让对手无论怎么应对都输」——这就是**两轮完美信息博弈**的判定语言。$\Sigma_2$ 恰是「存在一着必胜」的类，$\Pi_2$ 是「对任何一着都能应」的类。

**典型问题**：$\Sigma_2$ 完全问题如「是否存在集合 $X$，使对任意集合 $Y$，某个 SAT 公式在 $X \oplus Y$ 上可满足」——名字听着绕，本质都是「存在一个，使得任意一个，验证成立」。

## 3 PH 的内部结构：塌缩定理

PH 最优雅的性质是**塌缩（collapse）**：

**定理：** 若 $\Sigma_i = \Pi_i$ 对某个 $i \ge 1$ 成立，则 $PH = \Sigma_i$——整座层次从第 $i$ 层往下塌，不再增长。

**推论：若 $P = NP$，则 $PH = P$。**
 因为 $\Sigma_1 = \Pi_1 = P$ 立即触发塌缩。

**直观**：层次之所以可能无限延伸，是因为每一层都在「严格挑战」上一层；
一旦某一层「自我对称」（$\Sigma_i = \Pi_i$），挑战就失去了力量，上面的层全部落入第 $i$ 层。
<span class="marginnote">塌缩定理的意义在「反方向」：研究者常证明「假设 PH 不塌缩」，然后推出某结果。
若有人证明了 PH 塌缩，那将是仅次于 P = NP 的大新闻——因为它会连锁推翻一大批「以 PH 不塌缩为前提」的结论。
</span>

**包含关系**：

$$
\Sigma_1 \subseteq \Sigma_2 \subseteq \Sigma_3 \subseteq \cdots \subseteq PSPACE
$$

每一层都包含上一层；
且 $PH \subseteq PSPACE$（交替量词的暴力搜索用多项式空间即可）。

## 4 PH 与电路、随机性的纠缠

PH 虽然「看起来很大」，但有三条重要的「有限性」结果：

**定理（Karp–Lipton）：若 $NP \subseteq P/poly$，则 $PH = \Sigma_2$。**
 非一致电路若足够强，多项式层次就塌缩。
<span class="marginnote">这是「电路 vs 统一类」最重要的关联定理之一。
直觉：若 SAT 有多项式电路，验证者就能用电路「非一致地」回答问题，把 $\exists \forall$ 的嵌套结构压缩成两层。
</span>

**定理（Sipser–Gács–Lautemann）：$BPP \subseteq \Sigma_2 \cap \Pi_2$。**
 随机类 BPP 不会超过 PH 第二层——随机性再强，也强不过两层量词。
<span class="marginnote">这两条定理共同传达一个信息：PH 是「统一计算」的上限容器——电路（非一致）、随机（概率）、归约（逻辑）都逃不出它。
若能证明 PH 有完全问题，则 PH = PSPACE（因而塌缩）——所以「PH 有没有完全问题」也成了开放问题。
</span>

**重点：PH 的每一条结果几乎都是「若 X 太强，则 PH 塌缩」——塌缩成了理论界的「否决票」。**
 每当某假设太美好，塌缩定理就提醒我们：这假设恐怕不成立。

## 5 PH 的完全问题与开放问题

**$\Sigma_i$ 完全问题**：每层都有——「带 $i$ 层交替量词的量化布尔公式」在 $\Sigma_i$ 中完全（量词交替 $i$ 层、最外是 $\exists$）。而 TQBF（无交替限制）在 PSPACE 中完全。
**PH 自身是否有完全问题**：**开放**。若 PH 有完全问题，则 PH = PSPACE（且塌缩）——所以「PH = PSPACE 吗」「PH 有完全问题吗」都悬而未决。<span class="marginnote">对比：NP 有 SAT 作为完全问题，所以 NP 是「自足的」。PH 没有这样的「总代表」——若它有，层次就不可能是无限的。这从结构上解释了为什么 PH 研究如此依赖塌缩论证。</span>

## 6 小结

- **$\Sigma_i$ / $\Pi_i$**：$i$ 层交替量词（最外 $\exists$/$\forall$）的多项式时间语言；$\Sigma_1 = NP$、$\Pi_1 = coNP$。
- **PH** = $\bigcup_i \Sigma_i$；每层包含下层，$PH \subseteq PSPACE$。
- **塌缩定理**：$\Sigma_i = \Pi_i$ ⟹ $PH = \Sigma_i$；$P = NP$ ⟹ $PH = P$。
- **Karp–Lipton**：$NP \subseteq P/poly$ ⟹ $PH = \Sigma_2$；**Sipser–Gács–Lautemann**：$BPP \subseteq \Sigma_2 \cap \Pi_2$