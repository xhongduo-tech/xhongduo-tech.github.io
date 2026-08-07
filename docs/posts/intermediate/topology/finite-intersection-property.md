---
title: 紧空间的等价刻画：有限交性质
date: 2026-08-07
---

# 紧空间的等价刻画：有限交性质

<div class="epigraph">
<p>覆盖是加法，交是减法——紧致性在加法和减法下是同一回事。</p>
<footer>—— 对覆盖与有限交性质对偶性的概括</footer>
</div>

<div class="article-byline">
<p>第二级 · 拓扑学 ｜ 尤承业《基础拓扑学讲义》第四章 ｜ Munkres《Topology》§26 ｜ 2026-08-07</p>
</div>

## 为什么需要「交」的版本

上一课用「覆盖」（并的语言）定义了紧性，但许多证明中「交」的语言更顺：紧空间里，一族闭集如果「每有限个都相交」，那么「全体也相交」。这条**有限交性质（FIP）**是紧性的对偶刻画，它的威力在于把「无穷」的结论化约到「有限」的假设——这是紧致性反复使用的核心机制。Cantor 交集定理（递减闭集列的交非空）、「紧空间上连续函数达到最大值」的证明、以及 Tychonoff 定理的证明，全部依赖 FIP。<span class="marginnote">FIP 是「覆盖」的镜像：覆盖说「并盖得住」，FIP 说「交逃不掉」。取补集把两者互相翻译，是 De Morgan 律在拓扑证明中的标准动作。</span>

## 1 有限交性质的定义

**有限交性质（finite intersection property, FIP）**：设 $\mathcal{F}$ 是 $X$ 的子集族。若 $\mathcal{F}$ 的**每个有限子族**都有非空的交集，则称 $\mathcal{F}$ 具有**有限交性质**。

即：对任意 $F_1, \ldots, F_n \in \mathcal{F}$，

$$F_1 \cap F_2 \cap \cdots \cap F_n \neq \emptyset$$

注意「每个有限子族」而非「全体」——全体交集可能是空集，但只要每个有限子族都非空，就称 FIP 成立。<span class="marginnote">「每有限个都交」与「全体相交」的差别正是紧致性的全部工作：紧性承诺「每有限个交 ⟹ 全体交」。这个「有限→无穷」的推进是紧致性在分析里屡建奇功的原因。</span>

看例子：

- 在 $\mathbb{R}$ 中，$\{[0, 1/n] \mid n \ge 1\}$ 有 FIP（有限个 $[0,1/n_i]$ 的交是 $[0, 1/\max n_i]$，非空），且全体交集 $\bigcap [0,1/n] = \{0\}$ 非空——这里 FIP 的「全体」结论成立。
- 在 $\mathbb{R}$ 中，$\{(0, 1/n) \mid n \ge 1\}$ 有 FIP，但全体交集 $\bigcap (0,1/n) = \emptyset$——**开集的 FIP 不能推出全体交非空**！这正是「闭集」登场的原因。

## 2 主定理：紧 ⟺ 闭集族的 FIP

**定理（有限交性质刻画）**：拓扑空间 $X$ 是紧空间，当且仅当对 $X$ 的**每个由闭集组成**且具有有限交性质的子族 $\mathcal{F}$，都有

$$\bigcap_{F \in \mathcal{F}} F \neq \emptyset$$

证明用取补集把「覆盖」翻译成「交」：

- **（⟸ 紧 ⟹ FIP 成立）**：设 $\mathcal{F}$ 是闭集族且全体交为空：$\bigcap F = \emptyset$。取补集得 $\{X \setminus F\}$ 是 $X$ 的开覆盖（$X = X \setminus \emptyset = \bigcup (X \setminus F)$）。$X$ 紧 ⟹ 有有限子覆盖 $X = (X \setminus F_1) \cup \cdots \cup (X \setminus F_n)$，取补集得 $F_1 \cap \cdots \cap F_n = \emptyset$——这与 FIP 矛盾。故全体交非空。
- **（⟸ FIP ⟹ 紧）**：设 $\mathcal{A}$ 是 $X$ 的开覆盖，假设无有限子覆盖。则对任意有限个 $A_1, \ldots, A_n \in \mathcal{A}$，$\bigcup A_i \neq X$，故 $\bigcap (X \setminus A_i) \neq \emptyset$——闭集族 $\{X \setminus A \mid A \in \mathcal{A}\}$ 有 FIP。由假设其全体交非空：$\bigcap (X \setminus A) \neq \emptyset$，即 $\bigcup A \neq X$，与 $\mathcal{A}$ 是覆盖矛盾。<span class="marginnote">证明的两步完全镜像：开覆盖的「无有限子覆盖」被取补集翻译成闭集族的「每有限个交非空」；紧性把后者推向「全体交非空」，再取补集回推「覆盖被有限盖住」。De Morgan 律是这场镜像戏的导演。</span>

这条定理的价值：**紧性可以从「覆盖」的视角切换成「交」的视角**，两种视角各自擅长不同的证明。

## 3 Cantor 交集定理

FIP 最著名的应用是**嵌套闭集列的交**：

**定理（Cantor 交集定理）**：设 $X$ 紧，$F_1 \supset F_2 \supset F_3 \supset \cdots$ 是 $X$ 的**递减非空闭集列**，则

$$\bigcap_{n=1}^{\infty} F_n \neq \emptyset$$

证明：$\{F_n\}$ 是闭集族，且递减非空保证任意有限个 $F_{i_1} \cap \cdots \cap F_{i_k} = F_{\max i_j} \neq \emptyset$——FIP 成立。由主定理，全体交非空。∎

这个定理在 $\mathbb{R}^n$ 中是最重要的分析工具之一：**嵌套闭方体列必含公共点**，是「二分法找零点」「压缩映射定理」的根基。在紧空间中它不再需要度量、不需要「长度趋零」——纯拓扑的 FIP 就够了。<span class="marginnote">Cantor 交集定理是「$\mathbb{R}$ 的完备性」在紧空间里的推广形态：实数轴里的「嵌套闭区间必交」，在任意紧空间里都成立。学到《泛函分析》时，Banach 空间里「闭球嵌套」与完备性的关系正是它的同源兄弟。</span>

注意递减的**开集**列没有这个结论：$(0, 1/n)$ 递减但交为空。所以「闭」是 Cantor 定理不可少的条件。

## 4 公式解析：取补集的翻译

FIP 与覆盖的互译公式：

$$\bigcap_{\alpha} F_\alpha = \emptyset \iff X = \bigcup_{\alpha} (X \setminus F_\alpha)$$

- **第一步，读左边**：$\bigcap F_\alpha = \emptyset$ 说「没有点同时属于所有 $F_\alpha$」。
- **第二步，读右边**：$X = \bigcup (X \setminus F_\alpha)$ 说「每个点都至少不属于某个 $F_\alpha$」——这正是「没有点属于所有 $F_\alpha$」。
- **第三步，De Morgan**：两步是 De Morgan 律 $\bigcap F_\alpha = \emptyset \iff \bigcup (X \setminus F_\alpha) = X$ 的直接写法。这个翻译让「开覆盖 ↔ 闭集族」无缝切换，也解释了为什么 FIP 定理要闭集：$X \setminus A$（$A$ 开）自动是闭集。

## 5 辨析｜易错点

**辨析｜易错点：** FIP 有四个高频误区：

- **FIP 的「全体交非空」只对闭集族成立**：开集族有 FIP 可能全体交空（$(0,1/n)$）。闭是必要条件。
- **「每有限个交非空」≠「全体交非空」的普适断言**：只有紧空间才把 FIP 提升为「全体交非空」。非紧空间（$\mathbb{R}$ 的开区间族）FIP 失效。
- **主定理的证明依赖取补集**：补集的补集是原集（$X \setminus (X \setminus A) = A$），De Morgan 律方向要准，别把交并弄反。
- **Cantor 定理要求闭**：递减闭集列才交非空；递减开集列没有保证。$\bigcap (0,1/n) = \emptyset$ 是最小反例。

## 6 小结

- **FIP**：有限子族之交皆非空的性质。
- **主定理**：$X$ 紧 ⟺ 每个闭集族若 FIP 则全体交非空。
- **证明**：取补集 + De Morgan 律，把「覆盖」翻译成「交」。
- **Cantor 交集定理**：紧空间中递减非空闭集列交非空。
- **闭不可丢**：开集族的 FIP 不保证全体交非空。
- FIP 是「有限 → 无穷」的推进机制，Tychonoff 定理的证明将深度依赖它。

在下一节，我们将建立紧性的两条「遗传」性质：**紧空间的闭子集是紧的；Hausdorff 空间中紧集是闭的**。

### FIP 的补集视角再练一遍

把「覆盖 ↔ 交」的翻译练熟，是掌握 FIP 的捷径。三组对照：

| 覆盖语言 | 交的语言（取补集） |
| --- | --- |
| $\bigcup A_\alpha = X$ | $\bigcap (X \setminus A_\alpha) = \emptyset$ |
| 覆盖有有限子覆盖 | 闭集族有有限个交为空 |
| 覆盖无有限子覆盖 | 闭集族 FIP（每有限个交非空） |

第一行是 De Morgan 律本身；第二行是「有有限子覆盖」的翻译；第三行是「无有限子覆盖」⟺「FIP」——这正是 FIP 定理证明里反复用到的等价。把这张表背下来，任何「覆盖」问题都可以切到「交」的视角再想一遍。

### FIP 与 Tychonoff 的伏笔

FIP 是 Tychonoff 定理证明的入场券（上一课的预告），这里先埋一个关键直觉：

- 积空间 $\prod X_\alpha$ 的紧性，等价于「任何 FIP 闭集族交非空」。
- 对每个坐标 $\alpha$，「投影闭包的 FIP」由 $X_\alpha$ 的紧性给交点 $x_\alpha$；把所有坐标的交点 $(x_\alpha)$ 拼起来，就是整个积空间的交点。
- 这个「逐坐标取交」的图式，是 Tychonoff 证明的骨架。现在用「交」的语言记住它，学 Tychonoff 时就能直接对上。

### FIP 的实战判断

判断一个集族是否有 FIP，是使用定理前的第一步，三个实用技巧：

- **递减闭集列天然 FIP**：$F_1 \supset F_2 \supset \cdots$（非空）的任意有限个之交是最小的那个，非空——递减列自动 FIP。
- **「有限交非空」要逐个验**：不能只看「两两相交」就断言 FIP——三个集合两两相交但三者之交为空是可能的（$A=\{1,2\}, B=\{2,3\}, C=\{1,3\}$）。
- **取补集看覆盖**：要验证闭集族 FIP，可以看补集族「是否无有限子覆盖」——用上一课的覆盖直觉换过来。

这三招覆盖了 FIP 判断的大部分实操场景。
