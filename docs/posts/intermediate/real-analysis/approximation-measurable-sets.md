---
title: 可测集的逼近：用开集、闭集与 Gδ、Fσ 集逼近
date: 2026-08-07
---

# 可测集的逼近：用开集、闭集与 Gδ、Fσ 集逼近

<div class="epigraph">
<p>可测集并不可怕——它总可以被开集从外面、闭集从里面夹住，夹缝的宽度可以小到任意程度。</p>
<footer>—— 斯坦尼斯瓦夫 · 萨克斯（Stanisław Saks）</footer>
</div>

<div class="article-byline">
<p>第二级 · 实变函数与测度论 ｜ 周民强《实变函数论》§3.7 ｜ 2026-08-07</p>
</div>

## 为什么从可测集的逼近开始

可测集由抽象的 Carathéodory 条件定义，但实际计算与证明中，我们总想把可测集「翻译」成熟悉的对象——开集、闭集、$G_\delta$、$F_\sigma$。**逼近定理（regularity）** 承诺这件事可行：任意可测集都能用开集从外面罩住、用闭集从里面撑住，误差（测度差）任意小；更强的版本甚至用 $G_\delta$（可数开交）与 $F_\sigma$（可数闭并）实现「测度零误差」的夹逼。

这条定理是测度论的「翻译手册」：证明可测集满足某性质时，先对开集/闭集验证（那里有紧致性、连通性等强工具），再通过逼近把结论传到可测集。**Egorov 定理、Luzin 定理、L^p 稠密性，全部依赖这本翻译手册。**<span class="marginnote">逼近定理即 Lebesgue 测度的<strong>内外正则性（inner/outer regularity）</strong>。它在抽象测度论中并不自动成立——需要 Radon 测度等额外条件。Lebesgue 测度是「正则性典范」：外正则（开集逼近）与内正则（紧集逼近）同时满足，这让实变函数论享有许多抽象理论没有的便利。</span>

## 1 外逼近：用开集

**定理（外逼近）**：设 $E$ 可测。则对任意 $\varepsilon>0$，存在开集 $G\supset E$，使

$$m(G\setminus E)<\varepsilon$$

证明（有限测度情形）：由外测度的正则性，存在开集 $G\supset E$ 使 $m^*(G)<m^*(E)+\varepsilon$。而 $E$ 可测，$m^*(G)=m(G)$，$m^*(E)=m(E)$，且 $m(G)=m(E)+m(G\setminus E)$（$E$ 可测故切割无损耗，$m(G)=m(G\cap E)+m(G\setminus E)=m(E)+m(G\setminus E)$）。代入得 $m(G\setminus E)<\varepsilon$。

**无穷测度情形**：$m(E)=+\infty$ 时，把 $E$ 分解成 $E=\bigcup_k(E\cap B(0,k))$，每块有限测度（可能仍无穷），对外逼近后取可数并。

**重点：外逼近的误差用「差集的测度」衡量**，$m(G\setminus E)<\varepsilon$ 比「$m(G)-m(E)<\varepsilon$」更强——它要求罩住后多出来的部分极小，而不仅是总测度接近。这保证了「$G$ 几乎就是 $E$」，而非「$G$ 与 $E$ 总大小接近但形状完全不同」。

## 2 内逼近：用闭集与紧集

**定理（内逼近）**：设 $E$ 可测。则对任意 $\varepsilon>0$，存在闭集 $F\subset E$，使

$$m(E\setminus F)<\varepsilon$$

证明：由外逼近应用于 $E^c$：存在开集 $G\supset E^c$ 使 $m(G\setminus E^c)<\varepsilon$。取 $F=G^c$，$F$ 闭、$F\subset E$，且 $E\setminus F=E\cap G=G\setminus E^c$，故 $m(E\setminus F)<\varepsilon$。**「外逼近补集」自动给出「内逼近」**——这是开闭对偶的又一次胜利。

**推论（紧集内逼近）**：若 $m(E)<+\infty$，则存在紧集 $K\subset E$ 使 $m(E\setminus K)<\varepsilon$。取闭集 $F\subset E$ 且 $m(E\setminus F)<\varepsilon/2$，再取有界闭集 $K=F\cap\overline{B(0,R)}$ 使 $m(F\setminus K)<\varepsilon/2$（取大 $R$ 即可）。

**辨析｜易错点：内逼近与外逼近不对称。** 外逼近对任意可测集成立；内逼近的「紧集版」需要 $m(E)<+\infty$（用有界性截断）。无界集 $E=\mathbb{R}$ 没有「紧子集撑住无穷测度」的可能——这是「有界性」在测度论里反复出现的原因。<span class="marginnote">内逼近需要 $m(E)<\infty$ 而外逼近不需要，这个不对称的本质是：<strong>「从外面罩」不怕罩出无穷，「从里面撑」撑不出无穷</strong>。在概率论里测度全有限，内外逼近无条件成立——这也是概率测度格外「顺手」的又一体现。</span>

## 3 Gδ 与 Fσ 的零误差逼近

把逼近推到极限：误差可以归零，代价是用「可数步」的结构。

**定理（$G_\delta$ 外逼近）**：设 $E$ 可测。则存在 $G_\delta$ 集 $G\supset E$，使

$$m(G\setminus E)=0$$

**定理（$F_\sigma$ 内逼近）**：设 $E$ 可测。则存在 $F_\sigma$ 集 $F\subset E$，使

$$m(E\setminus F)=0$$

证明思路：对 $\varepsilon=\tfrac1k$ 取外逼近开集 $G_k\supset E$ 使 $m(G_k\setminus E)<\tfrac1k$，令 $G=\bigcap_kG_k$。$G$ 是 $G_\delta$ 集、$G\supset E$，且 $m(G\setminus E)\le m(G_k\setminus E)<\tfrac1k$ 对一切 $k$，故 $m(G\setminus E)=0$。内逼近对称。

**重点：可测集与 Borel 集「只差零测集」。** 对任意可测集 $E$，存在 $G_\delta$ 集 $G$ 与 $F_\sigma$ 集 $F$，使得 $F\subset E\subset G$ 且 $m(G\setminus F)=0$。于是：**可测集 = Borel 集 ± 零测集**（这正是上节提到的 $\mathcal{M}=\mathcal{B}\oplus$（零测子集）的证明路径）。这个「夹逼归零」的结论，让几乎所有「对 Borel 集成立」的性质自动推广到可测集。

## 4 公式解析：零误差逼近的 $\tfrac1k$ 取极限

$G_\delta$ 逼近的证明展示了「$\varepsilon$ 归零」的标准手法：

$$G=\bigcap_{k=1}^{\infty}G_k,\qquad m(G_k\setminus E)<\frac1k$$

$$0\le m(G\setminus E)\le m(G_k\setminus E)<\frac1k\ \Longrightarrow\ m(G\setminus E)=0$$

- **第一步，读「对 $\varepsilon=\tfrac1k$ 逐个逼近」**：外逼近定理对任意 $\varepsilon>0$ 都给出开集。取 $\varepsilon_k=\tfrac1k$，得到一列开集 $G_k\supset E$，误差逐次缩小。
- **第二步，读「$G=\bigcap_k G_k$ 为何是 $G_\delta$」**：可数个开集的交是 $G_\delta$ 集。**「取交」把「逐个开集」压缩成「单个 $G_\delta$ 结构」**——代价是结构变复杂（$G_\delta$ 未必开），报酬是误差可以归零。
- **第三步，读「$m(G\setminus E)<\tfrac1k$ 对所有 $k$」**：$G\subset G_k$ 故 $G\setminus E\subset G_k\setminus E$，单调性给误差上界 $\tfrac1k$。**一个非负实数小于所有 $\tfrac1k$，只能是 $0$**——「归零」的论证只有这一句。

**这套「$\tfrac1k$ 收缩」手法**（先逐层逼近，再取极限归零）在 Egorov 定理、Luzin 定理、控制收敛定理中反复出现，是「从近似到精确」的测度论标准通道。

## 5 逼近定理的实例与直观

**实例一（有理数集的逼近）**：$E=\mathbb{Q}\cap[0,1]$，$m(E)=0$。外逼近：取 $G_k=\bigcup_{r\in\mathbb{Q}\cap[0,1]}(r-\tfrac{\varepsilon}{2^k},r+\tfrac{\varepsilon}{2^k})$，则 $G_k\supset E$ 且 $m(G_k)\le\sum_k\tfrac{2\varepsilon}{2^k}\to$（可压任意小），$m(G\setminus E)<\varepsilon$。**可数有理点被「薄薄的开区间」罩住，总长任意小**——零测集的典型外逼近。

**实例二（康托尔集的逼近）**：$C$ 零测，$C=\bigcap_kC_k$（$C_k$ 是 $2^k$ 个闭区间）。内逼近：$F_k=C_k$ 是闭集且 $C\subset F_k$——等等，$F_k\supset C$，不满足 $F\subset E$。正确方向：$m(C)=0$，内逼近平凡（$F=\varnothing$ 已足够）。真正的示范：$m(C)=0$ 时「$G_\delta$ 外逼近」$C\subset G$ 且 $m(G\setminus C)=0$——取 $G=C$ 自身（$C$ 已是 $G_\delta$：$C=\bigcap_kC_k$，闭集交）。**零测集自动是 $G_\delta$ 的极限**。

**实例三（正测度集的逼近）**：$E=[0,1]\cap(\mathbb{R}\setminus\mathbb{Q})$（无理数），$m(E)=1$。外逼近：$G=(0,1)$，$m(G\setminus E)=0$（补集 $\mathbb{Q}$ 零测）。内逼近：$F=[0,1]\setminus\bigcup_{r\in\mathbb{Q}\cap[0,1]}(r-\delta_r,r+\delta_r)$（挖去有理点的薄邻域），$F$ 闭、$F\subset E$、$m(E\setminus F)<\varepsilon$。**「挖开集补」构造内逼近闭集**——这是「补集挖空技术」的实际演练。

**重点：逼近定理的实用性在于「用开/闭集替换可测集」。** 证明「可测集有性质 $P$」时，先证「开集有 $P$」「闭集有 $P$」（它们有紧致性、连通性等强工具），再通过逼近把 $P$ 传到可测集（误差任意小 ⇒ 极限成立）。**「好集逼近坏集」是测度论证明的标准句式**——Luzin、Egorov、L^p 稠密性全部这样工作。

## 7 数值演练与逼近速查

**算例一（有理数集的零误差外逼近）**：$E=\mathbb{Q}\cap[0,1]$，$m(E)=0$。$G_\delta$ 外逼近：$G_k=\bigcup_r(r-\tfrac1{2^k},r+\tfrac1{2^k})$，$G=\bigcap_kG_k$，$m(G)\le\sum_k\tfrac{2}{2^k}\to$ 任意小，$m(G\setminus E)=0$——**零测集被 $G_\delta$ 罩到测度零**。

**算例二（无理数集的逼近）**：$E=[0,1]\setminus\mathbb{Q}$，$m(E)=1$。外逼近：$G=(0,1)$，$m(G\setminus E)=0$（补集 $\mathbb{Q}$ 零测）。内逼近：挖去有理点薄邻域得闭集 $F$，$m(E\setminus F)<\varepsilon$。**「补集挖空」构造内逼近闭集是标准动作。**

**对照表：逼近的四种强度**

| 逼近 | 结论 | 条件 |
| --- | --- | --- |
| 开集外 | $m(G\setminus E)<\varepsilon$ | 无 |
| 闭集内 | $m(E\setminus F)<\varepsilon$ | 无 |
| 紧集内 | $m(E\setminus K)<\varepsilon$ | $m(E)<\infty$ |
| $G_\delta$/$F_\sigma$ | 测度差为零 | 无 |

**术语速查**

| 记号 | 含义 |
| --- | --- |
| 正则性 | 开外闭内逼近 |
| 外正则 | 开集逼近 |
| 内正则 | 紧集逼近 |
| $G_\delta$ | 可数开交 |

**辨析｜易错点：内逼近的紧集版需要 $m(E)<\infty$，外逼近不需要。** 「从外罩」不怕罩出无穷，「从内撑」撑不出无穷——**内逼近的「有界性」条件是正则性不对称的本质**。

### 三步记住「$\tfrac1k$ 归零」

- **逐层逼近**：$\varepsilon_k=\tfrac1k$ 给开集 $G_k$。
- **取交归零**：$G=\bigcap_kG_k$，$m(G\setminus E)<\tfrac1k$ 对一切 $k$。
- **夹逼**：非负实数小于所有 $\tfrac1k$ ⇒ 为 $0$。

**延伸（与抽象测度论连接）**：逼近定理即 Lebesgue 测度的内外正则性——抽象测度论里需 Radon 条件才成立。**Lebesgue 测度是「正则性典范」**，这让实变函数论享有许多抽象理论没有的便利（Egorov、Luzin、L^p 稠密全依赖它）。

**一道收束练习**：证明「$E$ 可测 ⇔ 存在 $G_\delta\supset E$ 与 $F_\sigma\subset E$ 使 $m(G\setminus F)=0$」——它是「可测集 = Borel 集 ± 零测集」的完整表述，也是 $\mathcal{M}=\mathcal{B}$ 完备化的证明路径。

## 8 小结

- **外逼近**：任意可测集可用开集外罩，误差 $m(G\setminus E)<\varepsilon$。
- **内逼近**：任意可测集可用闭集内撑，误差 $m(E\setminus F)<\varepsilon$；紧集版需 $m(E)<\infty$。
- **$G_\delta$/ $F_\sigma$ 零误差逼近**：存在 $G_\delta\supset E$、$F_\sigma\subset E$ 使测度差为零。
- **核心结论**：可测集 = Borel 集 ± 零测集；$\mathcal{M}$ 是 $\mathcal{B}$ 的完备化。
- **方法模板**：「先 $\tfrac1k$ 近似，再取极限归零」，测度论极限论证的标准动作。
- **数值**：$\mathbb{Q}\cap[0,1]$ 被 $G_\delta$ 罩到零；$[0,1]\setminus\mathbb{Q}$ 被挖空闭集内逼近。

在下一节，我们面对不可测的深渊：构造 **Vitali 集**，证明「不可测集确实存在」，并考察选择公理在其中的角色。
