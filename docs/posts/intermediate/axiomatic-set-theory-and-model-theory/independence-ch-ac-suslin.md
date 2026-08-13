---
title: 力迫的应用：CH、AC 与 Suslin 假设的独立性
date: 2026-08-07
---

# 力迫的应用：CH、AC 与 Suslin 假设的独立性

<div class="epigraph">
<p>连续统假设既不能被证明也不能被否证——它悬在 ZFC 的边缘，由每一个扩张决定它的宿命。</p>
<footer>—— 保罗 · 科恩（Paul Cohen, 1966）</footer>
</div>

<div class="article-byline">
<p>第二级 · 公理集合论与模型论 ｜ Jech, <em>Set Theory</em> 第15章；Kunen 第10章 ｜ 2026-08-07</p>
</div>

## 为什么从独立性开始

前两节我们造好了力迫的机器：条件、通用滤、力迫关系、布尔值模型。现在收割战果——**三个教科书级的独立性证明**：连续统假设（CH）、选择公理（AC）与 Suslin 假设（SH）各自在 ZFC 中「既不能证明也不能否证」。这三个证明构成现代数学基础最壮观的图景：**ZFC 是一座地基，但它没有盖死天花板**。<span class="marginnote">Cohen 在 1963 年用可数链条件（ccc）偏序加上 $\aleph_2$ 个 Cohen 实数，证明了 $\neg$CH 的相对一致性，也证明 AC 的独立性（对称子模型法，与 Feferman 合作）。今天回头看，这三个结果各用一招：加实数改 CH、砍选择函数改 AC、迭代力迫改 SH。</span>

今天逐一拆解：CH 的独立性（Cohen 的 ccc 论证）、AC 的独立性（对称子模型）、Suslin 假设的独立性（Solovay-Tennenbaum 的迭代力迫 + Martin 公理）。三者合起来，展示了力迫的三大招式：**加对象、删对象、反复加**。

## 1 CH 的独立性：$\aleph_2$ 个 Cohen 实数

**目标**：证明 $\mathrm{Con}(\mathrm{ZFC}) \Rightarrow \mathrm{Con}(\mathrm{ZFC} + 2^{\aleph_0} = \aleph_2)$（从而 CH 不可证，因为 CH 断言 $2^{\aleph_0} = \aleph_1$）。

**武器**：Cohen 偏序 $\mathbb{P} = \mathrm{Fn}(\omega_2 \times \omega, 2)$——「有限部分函数 $\omega_2 \times \omega \to 2$」，序为延长。它同时添加 $\aleph_2$ 个 Cohen 实数（每个 $\alpha \lt  \omega_2$ 对应一条无限串）。

**论证骨架**：

1. **ccc**：$\mathrm{Fn}(\omega_2 \times \omega, 2)$ 满足可数链条件——任意反链至多可数。证明用「Δ-系统引理」（任意不可数族的有限集合族含不可数个两两相交于同一根部的子族），把反链里「互不相容（= 互斥的有限函数）」导出可数。
2. **基数保持**：由 ccc + 力迫定理，$\aleph_0, \aleph_1, \aleph_2, \dots$ 在 $V$ 与 $V[G]$ 里一致——新对象（实数）不改变基数。
3. **$2^{\aleph_0} = \aleph_2$**：每个 Cohen 实数不在 $V$ 里（分叉稠密集），且 $G$ 编码出 $2^{\aleph_0} = \aleph_2$ 个不同的新实数；由 ccc 保证恰有 $\aleph_2$ 个（可数链条件给出上界，Cohen 偏序给出下界）。<span class="marginnote">「恰有 $\aleph_2$ 个」的两边：下界是「每个 $\alpha$ 给一条新串」；上界是「$V[G]$ 里实数由名字 + $G$ 决定，而名字只有 $\aleph_2$ 多个（ccc 保证）」。Cohen 的洞见正是用 ccc 控制「新对象的数量不爆炸」。</span>

**结论**：$V[G] \vDash 2^{\aleph_0} = \aleph_2 \neq \aleph_1$，即 $\neg$CH。与 Gödel 的 $L \vDash \mathrm{CH}$ 合起来：**CH 独立于 ZFC**。

**辨析｜易错点：** ccc 只保证「基数不改变」，不保证「$2^{\aleph_0}$ 恰好 $\aleph_2$」——后者还要「名字计数」的一步。初学者常以为「加了 $\aleph_2$ 个实数就有 $2^{\aleph_0}=\aleph_2$」，其实还需排除「多出更多」：ccc 恰恰封死这条路。

## 2 AC 的独立性：对称子模型

**目标**：证明 $\mathrm{Con}(\mathrm{ZF}) \Rightarrow \mathrm{Con}(\mathrm{ZF} + \neg \mathrm{AC})$。

**武器**：**对称子模型（symmetric submodel）**——取力迫扩张 $V[G]$ 的一个「对某些名字对称不变」的子模型。思路：把「选择函数」这类全局对象在扩张里「砍掉」，但保留「没有 AC 时依然可定义的」结构。

**论证骨架**：

1. 取 Cohen 偏序添加**可数多个** Cohen 实数 $a_n$（$n \in \omega$），得 $V[G]$。
2. 定义「名字的置换」：把 $a_n$ 之间的有限置换自然扩展成名字的自同构（automorphism）。
3. **对称子模型** $\mathrm{HS} \subseteq V[G]$：只保留「名字在置换群下几乎不变（有有限支撑）」的集合。
4. **关键引理**：$\mathrm{HS} \vDash \mathrm{ZF}$（在 $V[G]$ 里可定义，且满足所有 ZF 公理——用对称性保证分离/替换不会「逃出」$\mathrm{HS}$）。
5. **AC 失效**：集合 $A = \{a_n : n \in \omega\}$ 在 $\mathrm{HS}$ 里，但它**没有选择函数**——任何「给每个 $a_n$ 选一个有限串」的函数都会被某个置换破坏（因为 $a_n$ 都是「彼此对称」的）。<span class="marginnote">对称子模型的核心思想：AC 需要「从每个袋子里选一个」，而若袋子的元素互相不可区分（可置换），就没有「不偏不倚」的选择函数。这是「破坏选择公理」的通用模板——Cohen 用它首次证明 AC 独立。</span>

**结论**：$\mathrm{HS} \vDash \mathrm{ZF} + \neg \mathrm{AC}$，AC 独立于 ZF。

**辨析｜易错点：** 对称子模型**不是** $L$ 那种「最小模型」，它是「介于 $V$ 与 $V[G]$ 之间的传递模型」，靠自同构的不动性定义。初学者常误以为「没有 AC 的模型都很小」；实际上它的大小由「对称部分」决定，可以容纳不可数结构。

## 3 SH 的独立性：迭代力迫与 Martin 公理

**目标**：证明 SH 与 $\neg$SH 都与 ZFC 相容。

**方向一（SH 一致）**：**Martin 公理（MA）** + $\neg$CH 推出 SH。Martin 公理断言：对每个满足 ccc 的偏序 $\mathbb{P}$ 和少于 $2^{\aleph_0}$ 个稠密集，存在滤与它们全相交。它把「通用滤」从「单个偏序」推广到「所有 ccc 偏序」。

**方向二（$\neg$SH 一致）**：在 $L$ 里用 $\Diamond$ 构造 Suslin 树（第2篇），从而 $L \vDash \neg \mathrm{SH}$。

**SH 的独立性论证**：

1. **Solovay-Tennenbaum（1971）**：从 $V = L$ 出发，用**迭代力迫（iterated forcing）**「反复加反例」——每步消灭一个「潜在的 Suslin 线」，迭代 $\aleph_2$ 步后得到 $V[G] \vDash \mathrm{MA} + \neg \mathrm{CH}$，而 MA 推出 SH。
2. **Jensen（1972）**：在 $L$ 中 $\Diamond$ 造出 Suslin 树，$L \vDash \neg \mathrm{SH}$。
3. 两边合起来：**SH 独立于 ZFC**（$V[G] \vDash \mathrm{SH}$，$L \vDash \neg \mathrm{SH}$）。<span class="marginnote">迭代力迫是力迫的「长跑版」：一个接一个的偏序 $(\mathbb{P}_\alpha)$ 用「有穷支撑（finite support）」连起来，在极限处取并。它把「解决一个问题」升级为「解决无穷多个问题」——MA 正是「迭代力迫后的极限公理」，在组合集合论里几乎取代 CH 的角色。</span>

**要点**：SH 的独立性是「力迫的三招式合体」：$L$ 提供 $\neg$SH（内模型），迭代力迫提供 SH（加对象 + 反复加），两边一夹，独立性成立。

**辨析｜易错点：** MA 不是「AC 的推广」，而是「通用滤的存在定理」——它只在 ccc 偏序上保证「足够小的稠密集族有交」。MA 与 CH 不相容（MA + $\neg$CH 才有趣），初学者常混淆「MA 推出 CH」与「MA 推出 $\neg$CH」——实际是 MA + $\neg$CH 一起成立（在迭代力迫模型里）。

## 4 公式解析：ccc 为什么能保持基数

把「ccc ⇒ 基数保持」的关键一步写成公式，拆开看：

$$
\text{ccc}(\mathbb{P}) \;\Longrightarrow\; \forall \kappa \text{ 基数},\; (\kappa \text{ 在 } V \text{ 中为基数} \Rightarrow \kappa \text{ 在 } V[G] \text{ 中为基数})
$$

- **反证**：若 $\kappa$ 在 $V[G]$ 里可数，则存在单射 $f: \omega \to \kappa$（在 $V[G]$ 中）。
- **名字化**：$f$ 有名字 $\dot f$；由力迫定理，存在 $p \in G$ 使 $p \Vdash \dot f: \omega \to \kappa$ 单射。
- **制造反链**：对每个 $n$，找 $p_n \le p$ 与值 $\alpha_n \lt  \kappa$ 使 $p_n \Vdash \dot f(n) = \check \alpha_n$。因为 $f$ 单射，不同 $n$ 的 $\alpha_n$ 互异，且 $p_n$ 两两不相容（否则同一条件强迫两个不同值）。
- **ccc 矛盾**：$\{p_n\}$ 是不可数反链（$\kappa$ 不可数时），违反 ccc。

**要点**：ccc 的关键作用是把「$V[G]$ 里的单射」翻译成「$V$ 里的不可数反链」，再用手头偏序的性质否定它。**基数保持 = 反链限制**——这是 ccc 一切应用的总根源。

**辨析｜易错点：** 「ccc」保证「$V$ 与 $V[G]$ 有相同的基数」，但**不**保证「共尾性不变量」全保持（如 $\mathrm{cf}$ 可能被某些非 ccc 偏序改变）。下一节我们会专门讨论「何时 ccc 不够、需要更强的守恒」——那正是《力迫与基数守恒、共尾性保持》的主题。

## 6 动手推导：为什么 $\mathrm{Fn}(\omega_2\times\omega,2)$ 的扩张有恰有 $\aleph_2$ 个新实数

把「$2^{\aleph_0} = \aleph_2$」的两个方向都验一遍，补上「名字计数」这一步。

- **第一步，下界（至少 $\aleph_2$ 个）**：对每个 $\alpha \lt  \omega_2$，Cohen 实数 $x_\alpha = \bigcup G \restriction \{\alpha\} \times \omega$（限制在坐标 $\alpha$ 上的串）不在 $V$ 里（分叉稠密集），且 $\alpha \neq \beta$ 时 $x_\alpha \neq x_\beta$。故 $V[G]$ 里有至少 $\aleph_2$ 个新实数。
- **第二步，上界（至多 $\aleph_2$ 个）**：$V[G]$ 里每个实数 $y \subseteq \omega$ 都有名字 $\tau$。名字是「$\omega$ 的元素配对条件的集合」——它的结构由「条件集合」决定。ccc 保证「条件可数多」不够，需要更细：实际用「名字的宽度 = 名字里序对的个数」递归计数，ccc + $\aleph_2$-cc 给出名字只有 $\aleph_2$ 多个（按「编码」计数）。
- **第三步，ccc 的作用**：$2^{\aleph_0}$ 在扩张里 = 「名字个数 × 每个名字的取值方式」。ccc 保证「取值方式不爆炸」（没有不可数多互斥条件），从而总数被钉在 $\aleph_2$。
- **第四步，要点**：加 $\aleph_2$ 个 Cohen 实数只是「下界」；「恰好 $\aleph_2$」要靠名字计数与 ccc 的「上界」收口。两者缺一不可。

**辨析｜易错点：** 初学者常以为「加了 $\aleph_2$ 个实数就有 $2^{\aleph_0} = \aleph_2$」——这只是下界。若偏序不 ccc（如可数支撑迭代），扩张可能悄悄多出更多实数，使 $2^{\aleph_0}$ 更大。ccc 的关键作用正是「封死上界」。

### 更进一步：独立性结果的时间线

把本专题反复出现的「独立性」串成一张时间表，理解它们如何逐层加深：

| 年份 | 结果 | 方法 |
| --- | --- | --- |
| 1938 | Gödel：$\mathrm{Con(ZF)} \Rightarrow \mathrm{Con(ZFC+GCH)}$ | 可构造宇宙 $L$ |
| 1963 | Cohen：$\mathrm{Con(ZF)} \Rightarrow \mathrm{Con(ZFC+\neg CH)}$ | 力迫法 |
| 1963 | Cohen-Feferman：AC 独立 | 对称子模型 |
| 1971 | Solovay-Tennenbaum：SH 与 ZFC 相容 | 迭代力迫 + MA |
| 1972 | Jensen：$L \vDash \Diamond$，$\neg$SH 与 ZFC 相容 | $\Diamond$ + 力迫 |
| 1980s | Shelah：proper 力迫，稳定性理论 | 迭代力迫工程化 |

**要点**：每一条独立性结果都是一次「对 ZFC 边缘的测绘」——$L$ 证明「ZFC 够用」，Cohen 证明「ZFC 不够用」，而 SH 的独立证明 ZFC「不表态」。这张表也预告了第4篇：模型论（紧致性、超积）会给这些独立性问题提供另一套「模型的视角」——把 ZFC 当作众多一阶理论之一来研究。

## 8 小结

- **CH 独立**：$\mathrm{Fn}(\omega_2\times\omega,2)$ 是 ccc，加 $\aleph_2$ 个 Cohen 实数得 $2^{\aleph_0}=\aleph_2$；配合 $L \vDash \mathrm{CH}$。
- **AC 独立**：对称子模型 $\mathrm{HS}$ 砍掉选择函数，保留 ZF；可数多个 Cohen 实数不可区分。
- **SH 独立**：$L \vDash \neg\mathrm{SH}$（$\Diamond$ 造 Suslin 树）；迭代力迫 + MA 推出 $\mathrm{SH}$（Solovay-Tennenbaum）。
- **ccc ⇒ 基数保持**：$V[G]$ 里的单射被翻译成 $V$