---
title: 导出函子
date: 2026-08-11
---

# 导出函子

<div class="epigraph">
<p>在数学里，你并不是真的理解什么东西，你只是逐渐习惯了它们。</p>
<footer>—— 约翰 · 冯 · 诺依曼（John von Neumann）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 同调代数 ｜ 对标教材 Weibel Ch. 2 ｜ 2026-08-11</p>
</div>

## 为什么从导出函子开始

前两篇建立了两件武器：**复形与同调**给出「测量洞的代数」，**蛇引理**给出「精确拼接的长正合列」。现在要问一个更高层的问题：**如何把一个只「近似正合」的函子升级成一套精确的测量机器？** 答案就是**导出函子（derived functors）**。

几乎所有同调不变量——Ext、Tor、群上同调、Hochschild 同调——都是同一个配方的产物：**用射影（或内射）对象去「代换」输入，再把这个代换喂给函子，最后取同调。**

名字的由来值得点破：「导出」不是「推导」的意思，而是「从 $F$ 派生出的高阶序列」——$F$ 本身是第 0 阶，$L_1F, R^1F$ 是它的「导出后代」。到第十一篇《导出范畴简介》我们会看到，这套「后代」可以被封装进一个真正的函子。 这个配方如此万能，以至于同调代数本质上是「导出的学问」；到了最后一篇《导出范畴简介》，我们会看到「导出」二字还有更彻底的一层含义。

## 1 问题的提出：精确到哪一步为止？

回忆线性代数与群论里的教训：并不是所有函子都「完全正合」。设 $R$ 是环，考虑两个最基本的 $R$-模函子：

- **$\operatorname{Hom}_R(-, M)$**：把每个 SES $0 \to A \to B \to C \to 0$ 送到 $0 \to \operatorname{Hom}(C, M) \to \operatorname{Hom}(B, M) \to \operatorname{Hom}(A, M)$。右端一般**不满**——$\operatorname{Hom}$ 是**左正合（left exact）**的。
- **$-\otimes_R M$**：把 SES 送到 $A \otimes M \to B \otimes M \to C \otimes M \to 0$。左端一般不**单**——张量积是**右正合（right exact）**的。

**核心思想**：正合性在某一步断掉，断掉的地方不是「错误」，而是信息——我们把它编成高阶导出函子。

**定义（导出函子的精神）**：一个函子的「第 $i$ 个右导出函子」$R^iF$ 在输入 $M$ 上取值为 $H^i(F(I^\bullet))$，其中 $I^\bullet$ 是 $M$ 的某个内射解析；左导出函子 $L_iF$ 取值为 $H_i(F(P_\bullet))$，其中 $P_\bullet$ 是 $M$ 的某个射影解析。**「解析」就是前两篇学过的复形概念，只是把边界映射换成「约简的核」**——我们稍后逐个拆开。

## 2 射影与内射对象：解析的原材料

解析由「好对象」砌成。**射影对象（projective object）** $P$：$\operatorname{Hom}(P, -)$ 是正合函子，即任意满射 $B \twoheadrightarrow C$ 都能把 $P \to C$ 提升到 $B$。自由模是射影的；在 $R$-模范畴中**有足够多的射影对象**（每个模都满射于某个自由模）。

**内射对象（injective object）** $I$：$\operatorname{Hom}(-, I)$ 正合，即任意单射 $A \hookrightarrow B$ 都能把 $A \to I$ 延拓到 $B$。阿贝尔群里**可除群**（如 $\mathbb{Q}$、$\mathbb{Q}/\mathbb{Z}$）是内射的；$R$-模范畴**有足够多的内射对象**。

「足够多」这个词很关键：它保证每个对象都能被嵌入一个内射对象（手法是「嵌入可除包」），这是内射解析存在的充分条件，也是整个右导出理论的地基。

<span class="marginnote">射影与内射完全对偶：翻转箭头方向，角色互换。这种「箭头方向翻转」的对称性是范畴论的本能，第一级《集合的概念》里「属于」与「包含」的方向感，在这里升格为「提升」与「延拓」。记住一条口诀：<strong>射影向上提，内射向外延</strong>。</span>

## 3 解析：把 M 拆成标准零件

**射影解析（projective resolution）**：一个复形 $P_\bullet$（$P_n$ 射影，$n \ge 0$）连同满射 $\varepsilon : P_0 \to M$，使得

$$\cdots \to P_2 \to P_1 \to P_0 \xrightarrow{\;\varepsilon\;} M \to 0 \quad \text{精确}$$

（把 $M$ 看成 $P_{-1}$，整个序列就是「$P_\bullet$ 加尾巴」的正合复形。）**内射解析（injective resolution）**对称：$0 \to M \to I^0 \to I^1 \to I^2 \to \cdots$，各 $I^n$ 内射。

**解析为什么有用**：有足够多射影/内射对象保证解析总存在；而**比较定理**保证任意两个解析之间链同伦等价，因此「用哪个解析」不影响后续同调——这是导出函子良定义的第一道保险。

**辨析｜解析不是唯一的**：一个模通常有无穷多种射影解析——自由解析、最小的射影解析、还有「bar 解析」（见《群同调与 Lie 代数同调》）。比较定理保证它们在链同伦意义下彼此一致，于是「取哪套解析」不影响 $L_iF$。**解析是一种坐标系：同一个点在坐标变换下长相不同，导出的不变量却不变。**

<span class="marginnote">解析的直觉可以照搬做菜：你手里有一块「形状复杂的肉」$M$，解析就是把它切成一排「规整的肉丁」$P_n$，肉丁足够规整（射影），拼起来还是原来的肉（精确）。随后我们对「肉丁」做化验（作用函子），把结果拼回完整报告（取同调）。</span>

## 4 定义导出函子：M 的化验报告

现在给出正式定义。设 $F : R\text{-Mod} \to S\text{-Mod}$ 是**右正合**的加性函子（如 $-\otimes M$），$M$ 任取一个射影解析 $P_\bullet \to M \to 0$。作用 $F$ 得复形 $F(P_\bullet)$，取同调：

$$\boxed{\,L_iF(M) := H_i(F(P_\bullet))\,}$$

若 $G$ 是**左正合**函子（如 $\operatorname{Hom}(-, M)$），取 $M$ 的内射解析 $0 \to M \to I^\bullet$：

$$\boxed{\,R^iG(M) := H^i(G(I^\bullet))\,}$$

**性质**（都可由前两篇的机器证明）：

- **$L_0F = F$**、$R^0G = G$：第零个导出函子就是原来的函子；
- **短正合列给长正合列**：$0 \to A \to B \to C \to 0$ 诱导
$$\cdots \to L_1F(A) \to L_1F(B) \to L_1F(C) \to F(A) \to F(B) \to F(C) \to 0$$
——这里的连接同态正是《蛇引理与连接同态》里那条蛇；
- **消没**：若 $M$ 本身射影，则 $L_iF(M) = 0$（$i \ge 1$）；若 $M$ 内射，则 $R^iG(M) = 0$（$i \ge 1$）。

<span class="marginnote">这条长正合列把上一节的蛇引理变成了「生产力」：每一个 SES 都自动吐出一整套高阶修正项。代数几何里的 <strong>Grothendieck 谱序列</strong>（第六篇）研究「两个函子复合」时，用的正是 $R^iG \circ R^jF$ 的拼装，导出函子的长正合列是它的燃料。</span>

**约化到 acyclic 对象**：定义解析时不必苛求「射影/内射」，只需对象满足 $L_iF(P) = 0$（$i \ge 1$），即所谓 **$F$-acyclic**。同调结果不变。这给计算带来巨大自由：例如群同调里，用「自由 $\mathbb{Z}G$-模」当射影解析，但实践中常换成更小的 $F$-acyclic 解析来算。

「acyclic」一词与《群同调与 Lie 代数同调》直接相关：bar 解析往往只是「$\mathbb{Z}$ 的 $F$-acyclic 解析」——它比最小射影解析长，却胜在显式可算。**自由度来自「不必最优，只要 acyclic」的松弛。**

## 5 公式解析：R^iG(M) = H^i(G(I^\bullet))

把右导出函子这条公式按四个台阶拆开：

$$
R^iG(M) = H^i\bigl( G(I^\bullet) \bigr), \qquad 0 \to M \to I^0 \to I^1 \to I^2 \to \cdots
$$

- **第一步，造内射解析**：利用「有足够多内射对象」，把 $M$ 嵌入内射模 $I^0$，再把余核 $I^0/M$ 嵌入 $I^1$，如此递推。得到正合列 $0 \to M \to I^\bullet$。
- **第二步，作用 $G$**：$G$ 左正合，所以 $0 \to G(M) \to G(I^0) \to G(I^1) \to \cdots$ **可能失去右端精确性**——$G(I^0) \to G(I^1)$ 的像不再等于 $G(I^1)$ 中的核。失去的信息记入高阶项。
- **第三步，取上同调**：$R^iG(M) = \ker(G(I^i) \to G(I^{i+1})) / \operatorname{im}(G(I^{i-1}) \to G(I^i))$——这正是前两篇学过的「商 = 洞」。于是**「$G$ 在 $M$ 上丢了什么」被量化成 $R^iG(M)$ 的洞**。
- **第四步，验证不依赖解析**：任何两个内射解析链同伦等价，链同伦在同调上无差别，故 $R^iG(M)$ 良定义。

一个立刻能算的例子：对 $G = \operatorname{Hom}_\mathbb{Z}(-, \mathbb{Z})$ 与 $M = \mathbb{Z}/2$。取内射解析 $0 \to \mathbb{Z}/2 \to \mathbb{Q}/\mathbb{Z} \xrightarrow{2} \mathbb{Q}/\mathbb{Z} \to 0$（乘法 2 是满的，因为 $\mathbb{Q}/\mathbb{Z}$ 可除）。作用 $G = \operatorname{Hom}_\mathbb{Z}(-, \mathbb{Z})$ 后，注意 $\operatorname{Hom}_\mathbb{Z}(\mathbb{Q}/\mathbb{Z}, \mathbb{Z}) = 0$（$\mathbb{Q}/\mathbb{Z}$ 里没有有限阶整数同态到 $\mathbb{Z}$，只有零同态），于是整条复形变成 $0 \to \mathbb{Z} \xrightarrow{0} 0 \to \cdots$。故 $R^0G(\mathbb{Z}/2) = \mathbb{Z}$，$R^1G(\mathbb{Z}/2) = 0$。数字虽小，但「先解析、再作用、后取同调」的流水线完全走通。

再配一个**左导出**的可手算例子：取 $F = -\otimes_\mathbb{Z} \mathbb{Z}/2$（右正合），对 $M = \mathbb{Z}/2$ 计算 $L_iF$。射影解析取 $0 \to \mathbb{Z} \xrightarrow{\times 2} \mathbb{Z} \to \mathbb{Z}/2 \to 0$，作用 $F$ 后得到复形 $0 \to \mathbb{Z}/2 \xrightarrow{\;0\;} \mathbb{Z}/2 \to 0$——注意 $\times 2$ 被张量积「踩平」成零映射，这正是右正合性丢失单射性的现场。取同调：

$$L_0F(\mathbb{Z}/2) = \mathbb{Z}/2, \qquad L_1F(\mathbb{Z}/2) = \mathbb{Z}/2, \qquad L_iF(\mathbb{Z}/2) = 0 \ (i \ge 2)$$

再用长正合列交叉验证：SES $0 \to \mathbb{Z} \xrightarrow{\times 2} \mathbb{Z} \to \mathbb{Z}/2 \to 0$ 诱导 $L_1F(\mathbb{Z}) \to L_1F(\mathbb{Z}/2) \to F(\mathbb{Z}) \to F(\mathbb{Z}/2)$，而 $\mathbb{Z}$ 平坦故 $L_1F(\mathbb{Z}) = 0$，于是 $L_1F(\mathbb{Z}/2) \cong \ker(F(\mathbb{Z}) \to F(\mathbb{Z}/2)) = \ker(\mathbb{Z}/2 \xrightarrow{0} \mathbb{Z}/2) = \mathbb{Z}/2$——对上了。**这就是下一节 Tor 的原型**：$L_i(-\otimes_R M) = \operatorname{Tor}_i^R(-, M)$。

## 6 下一站：Ext 与 Tor

这套配方的两个明星产品很快登场：**$\operatorname{Ext}^i_R(M, N) = R^i\operatorname{Hom}_R(M, -)(N)$** 与 **$\operatorname{Tor}^R_i(M, N) = L_i(-\otimes_R N)(M)$**。它们分别编码「同态空间的洞」与「张量积的洞」，并同时具有「两头都能算」的**平衡性**——下一节我们会看到，导出函子框架最大的红利，是告诉我们同一组不变量可以走两条完全不同的路去计算。

## 7 小结

- $\operatorname{Hom}$ 左正合、$-\otimes$ 右正合；**正合性断掉处编码信息**。
- 射影对象「向上提」，内射对象「向外延」；两类对象在 $R$-模中都**足够多**。
- 解析是「切成规整零件」，比较定理保证解析选择不敏感。
- $L_iF = H_i(F(P_\bullet))$、$R^iF = H^i(F(I^\bullet))$；第零个导出函子还原为原函子。
- 短正合列自动给出**长正合列**，连接同态来自蛇引理。
- $F$-acyclic 解析可替代射影/内射解析，结果不变。
- 「足够多」射影/内射对象保证解析存在；解析是「坐标系」，不变量不因坐标而变。
- 导出函子的长正合列 = 蛇引理的自动化：一个 SES 自动吐出一整套高阶修正。

在下一节，我们将为两个最重要的导出函子验明正身：**Ext 与 Tor**——它们如何分类「模的扩张」，又如何测出「挠」。
