---
title: 环的同态与同构
date: 2026-08-07
---

# 环的同态与同构

<div class="epigraph">
<p>环同态同时看守两条运算——它把加法的桥梁与乘法的桥梁焊成一座。</p>
<footer>—— 自 题（环同态课堂笔记）</footer>
</div>

<div class="article-byline">
<p>第二级 · 抽象代数 ｜ 杨子胥《近世代数》§7.3 ｜ 2026-08-07</p>
</div>

## 为什么从环的同态与同构开始

群有同态，环自然也有：**环同态**是同时保持加法和乘法的映射。它是「两种运算的桥梁」，也是抽象代数「同态-核-商」三件套在环论里的第一个角色。与群论最大的不同在于：**环同态的核不再只是子环，而是第八篇的主角——理想**，这预示着商环理论的到来。

本节把环同态的定义、基本性质（保持零元、负元、幂；像的子环性）、同构与自同构讲透，并特别辨析「含幺环同态是否保幺」这个环论特有的约定问题。掌握了环同态，你就能把群论里熟悉的「同态语言」迁移到环的世界，而迁移过程中的「偏差」（核变成理想、单位元约定）正是环论独特的味道。

## 1 环同态的定义

**环同态（ring homomorphism）**：设 $R, S$ 是环，$\varphi : R \to S$ 满足对一切 $a, b \in R$：

1. $\varphi(a + b) = \varphi(a) + \varphi(b)$（保持加法）；
2. $\varphi(ab) = \varphi(a)\varphi(b)$（保持乘法）。

则称 $\varphi$ 是环同态。

若 $\varphi$ 是双射，则称**环同构**，记作 $R \cong S$；若 $R = S$ 且 $\varphi$ 是同构，则称**自同构**。<span class="marginnote">环同态定义只要求两条：加法与乘法。它自动保持加法单位元（$\varphi(0_R) = 0_S$）与负元（$\varphi(-a) = -\varphi(a)$），因为加法部分是群同态。注意定义里<strong>不要求</strong>保持乘法单位元——这是含幺环同态的附加约定，下文专门辨析。</span>

**例：**
**模 $n$ 同态**：$\varphi : \mathbb{Z} \to \mathbb{Z}_n$，$\varphi(k) = k \bmod n$——保持加法与乘法；
**包含映射**：$\iota : \mathbb{Z} \to \mathbb{Q}$，$\iota(k) = k$——子环的包含是同态；
**求值同态**：$\varphi_x : \mathbb{R}[x] \to \mathbb{R}$，$\varphi_x(f) = f(x)$——「代入 $x$」保持多项式加法与乘法；
**矩阵环上的转置**：$M_n(\mathbb{R}) \to M_n(\mathbb{R})$，$A \mapsto A^T$——同构（保持加法和乘法，虽然乘法顺序保持 $A^T B^T = (BA)^T$……需注意这是环同态，因为 $A^T B^T = (BA)^T$ 而环乘法顺序 $A^T B^T$ 正是 $(BA)^T$，故保持乘法）。

## 2 环同态的基本性质

环同态从「两种运算」继承了一串基本性质。

**定理（环同态基本性质）：** 设 $\varphi : R \to S$ 是环同态。

1. $\varphi(0_R) = 0_S$，$\varphi(-a) = -\varphi(a)$（加法同态自动给出）；
2. $\varphi(a^n) = \varphi(a)^n$ 对 $n \ge 1$（幂被保持）；
3. $\operatorname{Im}\varphi$ 是 $S$ 的子环（像仍是子环）；
4. $\ker \varphi = \{ a \in R \mid \varphi(a) = 0_S \}$ 是 $R$ 的**理想**（第八篇详证）。

**像的子环性**：$\varphi(a) - \varphi(b) = \varphi(a - b) \in \operatorname{Im}\varphi$，$\varphi(a)\varphi(b) = \varphi(ab) \in \operatorname{Im}\varphi$——减法与乘法封闭，像成子环。$\blacksquare$<span class="marginnote">「像」自动是子环（减法 + 乘法封闭）；「核」自动是理想（不但是子环，还吸收 $R$ 的任何乘积）。群论里「核是正规子群」在环论升级为「核是理想」——理想正是环论里「能充当核」的那类子环，第八篇会把这层对应彻底讲清。</span>

**辨析｜易错点：** 环同态 $\varphi$ 是单射 ⟺ $\ker \varphi = \{ 0_R \}$。这条在环论里照样成立（因为加法部分是群同态，单射判定看核；乘法保持不影响单射性）。注意「$\varphi$ 保持 $0$」与「$\ker \varphi = \{0\}$」是两回事——前者每时每刻成立，后者才是单射的判据。

## 3 含幺环同态与「保幺」的约定

环同态对单位元的处理是环论最微妙的约定问题。

**问题**：$R, S$ 都含幺，环同态 $\varphi : R \to S$ 是否必有 $\varphi(1_R) = 1_S$？

**答案**：**不一定**。例子：零同态 $\varphi : R \to S$，$\varphi(r) = 0_S$ 是环同态，但 $\varphi(1_R) = 0_S \ne 1_S$（当 $S \ne \{0\}$）。

**两种约定：**
**含幺环同态（unital homomorphism）**：额外要求 $\varphi(1_R) = 1_S$。这是「环同态」在现代代数里的默认约定（多数教材与论文）；
若不要求保幺，则「零同态」「到子环的包含」都算同态，但「环同构」的定义会失去「单位元唯一」的整洁性。

**本系列约定**：除非特别说明，「环同态」指**含幺环同态**（要求 $\varphi(1_R) = 1_S$）。<span class="marginnote">保幺约定的影响：$M_n(\mathbb{R}) \to M_m(\mathbb{R})$ 的环同态若存在，必须把 $I_n$ 映到 $I_m$。而「$\mathbb{Z} \to 2\mathbb{Z}$ 的包含」在保幺约定下<strong>不是</strong>同态（$1 \mapsto 1 \notin 2\mathbb{Z}$ 的乘法单位元）——所以把「子环」定义成不含幺的那类，会与「保幺同态」约定产生微妙张力。做题时先确认约定。</span>

**例（保幺约定的威力）：** 若 $\varphi : R \to S$ 是保幺环同态且 $u \in R$ 可逆，则 $\varphi(u)$ 在 $S$ 中可逆，且 $\varphi(u^{-1}) = \varphi(u)^{-1}$。证明：$\varphi(u)\varphi(u^{-1}) = \varphi(uu^{-1}) = \varphi(1_R) = 1_S$——保幺让「可逆性」被同态传递。这解释了为什么密码学与数论里的环同态（如 $\mathbb{Z} \to \mathbb{Z}_n$）如此常用：它们保持可逆性，单位信息不丢失。

## 4 公式解析：同构判定与保持运算

环同构是「结构相同」的精确表达，判定流程与群同构完全平行。

**判定环同构 $R \cong S$ 的三步：**
1. **构造映射** $\varphi : R \to S$（灵感通常来自「把 $R$ 的元素翻译成 $S$ 的元素」）；
2. **证双射**（单射：$\ker \varphi = \{0\}$ 或直接 $a \ne b \Rightarrow \varphi(a) \ne \varphi(b)$；满射：每个 $s$ 都有原像）；
3. **证保持双运算**：$\varphi(a+b) = \varphi(a)+\varphi(b)$ 且 $\varphi(ab) = \varphi(a)\varphi(b)$。

**例：$M_n(\mathbb{R}) \cong M_n(\mathbb{R})$ 由转置给出**——$\varphi(A) = A^T$：双射（$(A^T)^T = A$）、保加法（$(A+B)^T = A^T+B^T$）、保乘法（$(AB)^T = B^T A^T$，注意顺序反了！）。这里出现一个陷阱：$(AB)^T = B^T A^T$，要写成 $\varphi(AB) = \varphi(A)\varphi(B)$ 需要 $\varphi(AB) = (AB)^T = B^T A^T = \varphi(B)\varphi(A)$，即 $\varphi(AB) = \varphi(A)\varphi(B)$ 当且仅当 $\varphi(A)\varphi(B) = \varphi(B)\varphi(A)$……实际上转置保持乘法：$\varphi(AB) = (AB)^T = B^T A^T = \varphi(B)\varphi(A)$，而这正是「$\varphi(AB) = \varphi(A)\varphi(B)$」因为环乘法里 $\varphi(A)\varphi(B) = A^T B^T$ 而 $(AB)^T = B^T A^T$。**关键**：环同态要求 $\varphi(ab) = \varphi(a)\varphi(b)$（保持「先乘后映」），转置满足 $(AB)^T = B^TA^T$ 而右边在环的乘法顺序下等于 $\varphi(A)\varphi(B)$（$A^T B^T$）……这里需要小心：$(AB)^T = B^T A^T$，而 $\varphi(A)\varphi(B) = A^T B^T$。两者一般不等。**转置不是保持乘法的环同态，除非 $AB = BA$！**<span class="marginnote">转置的「反序」让它<strong>不是</strong>环同态（除非限制到交换子环）：$(AB)^T = B^TA^T \ne A^TB^T = \varphi(A)\varphi(B)$（一般）。这正说明环同态要求「先乘后映 = 先映后乘」，顺序必须一致；反序映射（反同态）在环论里是另一个概念。读教材时留意：$A \mapsto A^T$ 是 $M_n$ 的「反自同构」而非自同构。</span>

**例（正确的同构）：** $\mathbb{R}[x] / \langle x \rangle \cong \mathbb{R}$ 待第八篇商环后证明；更直接的同构：$\mathbb{Z} \cong 2\mathbb{Z}$？——不！作为环，$\mathbb{Z}$ 含幺而 $2\mathbb{Z}$ 不含幺（保幺约定下不同构）；作为加法群则 $\mathbb{Z} \cong 2\mathbb{Z}$（$n \mapsto 2n$）。**同一个对象，群语言与环语言给出不同答案**——环结构比群结构更精细，这是环论「比群论要求更多」的体现。

## 5 自同构与「结构自对称」

与群的自同构类似，环的自同构刻画环的「结构对称」。

**定理：** $R$ 的全部自同构在复合下构成群，记作 $\operatorname{Aut}(R)$。

**例：**
$\operatorname{Aut}(\mathbb{Q}) = \{ \mathrm{id} \}$（有理数域只有恒等自同构，因为 $\varphi(1) = 1$ 且 $\varphi(n/m) = n/m$）；
$\operatorname{Aut}(\mathbb{C}) $ 是巨大的群（选择公理下有无穷多自同构，可把 $\sqrt[4]{2}$ 送到 $i\sqrt[4]{2}$）；但**域自同构保持 $\mathbb{R}$** 的只有共轭 $z \mapsto \bar z$ 与恒等（连续性假设下）——拓扑结构会削减代数对称；
$\operatorname{Aut}(\mathbb{Z}) = \{ \mathrm{id} \}$（保幺同态从 $1$ 出发唯一确定）。<span class="marginnote">「$\operatorname{Aut}(\mathbb{C})$ 巨大而 $\operatorname{Aut}(\mathbb{R})$ 平凡」是抽象代数里著名的反直觉：越「连着」的域，自同构越受拓扑约束。第十一篇 Galois 理论里，域的自同构（Galois 群）成为分类方程可解性的核心工具——现在看到的「自同构稀少」正是那里「结构被钉死」的预兆。</span>

**自同构的保结构作用**：环自同构把可逆元映到可逆元、把理想映到理想、保持零因子与否。它是环的「内在对称」，在第八篇商环、第十一篇 Galois 理论中反复出场。

## 6 对照速查：群同态与环同态的同框

把群同态与环同态并排，看清「多一条运算」带来的全部差异。

| 对比项 | 群同态 | 环同态 |
| --- | --- | --- |
| 保持的运算 | 一种（乘法/加法） | 两种（加法 + 乘法） |
| 核的角色 | 正规子群 | 理想 |
| 像的角色 | 子群 | 子环 |
| 单射判定 | $\ker = \{e\}$ | $\ker = \{0\}$ |
| 保单位元 | 自动（$e \to e$） | 需约定（保幺 vs 不保） |

**数值算例：$\varphi : \mathbb{Z} \to \mathbb{Z}_6$，$\varphi(k) = k \bmod 6$ 是含幺环同态吗？** $\varphi(1) = 1 \bmod 6 = 1 = 1_{\mathbb{Z}_6}$，保幺 ✓；$\varphi(2) = 2$，$\varphi(3) = 3$，$\varphi(2 \cdot 3) = \varphi(6) = 0$，$\varphi(2)\varphi(3) = 2 \cdot 3 = 6 \bmod 6 = 0$ ✓——保乘法成立，虽然「$2 \cdot 3 = 0$」在 $\mathbb{Z}_6$ 里是零因子行为，但同态仍然忠实记录。<span class="marginnote">$\mathbb{Z} \to \mathbb{Z}_6$ 是最常用的环同态：它把整数的加法乘法「模 6 投影」。核是 $6\mathbb{Z}$（理想），像是整个 $\mathbb{Z}_6$。密码学里 RSA 的加密计算、离散对数的模运算，全部生活在这种「模 $n$ 环同态」的投影里——同态是「模算术」的代数本质。</span>

**易错辨析｜「保持乘法」与「保持乘法顺序」。** 环同态要求 $\varphi(ab) = \varphi(a)\varphi(b)$（保持「先乘后映」）。转置 $A \mapsto A^T$ 满足 $(AB)^T = B^TA^T$——顺序反了，除非 $\varphi(A)\varphi(B) = \varphi(B)\varphi(A)$（即交换子环），否则转置不是环同态。判定时务必逐条验证「先乘后映 = 先映后乘」，别被「看起来像」骗过。

**一句话记法**：环同态看两条运算；核升级为理想；保幺是约定；转置是反同态不是同态——「两条运算」让环同态比群同态多一层谨慎。

## 7 小结

- **环同态**：保持加法与乘法两条；同构 = 双射同态；自同构 = 到自身的同构。
- **基本性质**：$\varphi(0) = 0$、$\varphi(-a) = -\varphi(a)$；像为子环；核为理想（第八篇）。
- **单射判定**：$\ker \varphi = \{0\}$。
- **保幺约定**：环同态默认保 $1_R \to 1_S$；否则零同态等反例成群。
- **陷阱**：转置 $A \mapsto A^T$ 是反同态而非同态；$\mathbb{Z} \not\cong 2\mathbb{Z}$ 作为环（群意义下却同构）。

在下一节，我们研究环里的「可逆元素」与「碰壁元素」：**单位元、可逆元与零因子**。可逆元构成乘法群，零因子则让乘法消去律失效——环的「分子结构」由此分化。
