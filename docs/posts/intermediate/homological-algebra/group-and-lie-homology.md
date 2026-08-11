---
title: 群同调与 Lie 代数同调
date: 2026-08-11
---

# 群同调与 Lie 代数同调

<div class="epigraph">
<p>实数领域中连接两条真理的最短路径，往往要穿过复数领域。</p>
<footer>—— 雅克 · 阿达马（Jacques Hadamard）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 同调代数 ｜ 对标教材 Weibel Ch. 6–7 ｜ 2026-08-11</p>
</div>

## 为什么从群同调开始

前七篇建立了一套「代数显微镜」，现在轮到它**出诊**：把显微镜对准**群**与 **Lie 代数**——数学里两个最古老、最具体的代数结构。做法惊人地简单：把一个群 $G$ 编码进一个环 $\mathbb{Z}G$（**群环**），于是「群」就变成了「$\mathbb{Z}G$-模的世界」，而我们已经会算这个世界里的一切导出函子。

这条「绕路」正是阿达马那句话的翻版：**为了研究群，我们穿过了模的代数领域。** 回报极其丰厚：$H^2(G, M)$ 分类群的扩张、$H_1(G, \mathbb{Z})$ 恰是交换化 $G/[G,G]$、$H_2(G, \mathbb{Z})$ 是肖尔乘子……同一套配方搬到 Lie 代数，又得到 Chevalley-Eilenberg 复形与著名的 **Whitehead 引理**。这一节是把「导出函子」从定义变成「武器」的关键一跃。

## 1 群环：把群装进代数

**群环（group ring）** $\mathbb{Z}G$：由有限形式线性组合 $\sum_{g \in G} n_g \, g$（$n_g \in \mathbb{Z}$，几乎全为 0）构成的环，加法按项相加，乘法由群乘法线性扩张。$G$ 嵌进 $\mathbb{Z}G$ 作为「单位元素的陪集」。

**$\mathbb{Z}G$-模 = 带 $G$ 作用的阿贝尔群**：$\mathbb{Z}G$-模 $M$ 就是在阿贝尔群 $M$ 上规定一个**与加法兼容的 $G$-作用**（$g(x+y) = gx + gy$）。

为什么非要用 $\mathbb{Z}G$ 这么大的环？因为「群作用」被编码成环乘法后，「$G$-模的同调」就自动享受环上 Tor/Ext 的全部机器，无需为群另立法规。**把代数结构翻译成环，是同调代数最省事的通用接口。**

补充：$\mathbb{Z}G$ 不是交换环（除非 $G$ 交换），所以「射影解析」不能简化成「自由模解析」——这正是群环视角的难点，也是 bar 解析存在的意义：**它给出统一的、不依赖交换性的显式解析。**两个马上要用的特殊模：

- **不动点**：$M^G = \{m \mid gm = m,\ \forall g\}$；
- **轨道商（coinvariants）**：$M_G = M \big/ \langle gm - m \rangle \cong M \otimes_{\mathbb{Z}G} \mathbb{Z}$，其中 $\mathbb{Z}$ 带**平凡作用**。

<span class="marginnote">「平凡作用」是全场最不起眼却最关键的角色：它让 $\mathbb{Z}$ 成为 $\mathbb{Z}G$-模（$g \cdot 1 = 1$）。所有群同调定义里的「$Z$」都是这个平凡模——<strong>群同调 = 平凡模与 $M$ 之间的纠缠</strong>，纠缠得越深，群的复杂结构暴露得越多。</span>

## 2 定义：两个模板直接套用

一切在此落定：

$$\boxed{\,H_n(G, M) := \operatorname{Tor}_n^{\mathbb{Z}G}(\mathbb{Z}, M), \qquad H^n(G, M) := \operatorname{Ext}_{\mathbb{Z}G}^n(\mathbb{Z}, M)\,}$$

第零阶即不动点与轨道商：$H^0(G,M) = M^G$，$H_0(G,M) = M_G$。高阶项的全部性质（长正合列、消没、与解析无关）都已在《导出函子》与《Ext 与 Tor》里备好，无需重证——这就是「一次搭台，四处演出」。

**低阶含义**（都是几十年的经典，可直接引用）：

- **$H^1(G, M)$**：导子（crossed homomorphisms）商掉内导子；特别地 $H^1(G, \mathbb{Z}) = \operatorname{Hom}(G_{ab}, \mathbb{Z})$。
- **$H^2(G, M)$**：**$M$ 被看作阿贝尔核时，$G$ 被 $M$ 扩张的方式的分类**（与 $\operatorname{Ext}^1$ 的「扩张分类」遥相呼应，只是群的扩张更硬核）。
- **$H_1(G, \mathbb{Z}) = G_{ab} = G/[G,G]$**：一维同调把群「交换化」——这正是拓扑里「基本群的一维同调 = 基本群的交换化」的代数版本。
- **$H_2(G, \mathbb{Z})$**：**Schur 乘子**，度量「群与它的一个完美覆盖之间的差距」，在群表示论与中心的分类里到处露面。

**Shapiro 引理**是群同调最重要的一台「搬运机」：对子群 $H \le G$ 与 $H$-模 $M$，有 $H^n(G, \operatorname{Ind}^G_H M) \cong H^n(H, M)$——诱导模把「$H$ 的上同调」原样搬进「$G$ 的上同调」。几何上它对应「$G$-主丛的纤维同调」，是 Leray-Serre 谱序列（《谱序列》）在群论里的镜像。

<span class="marginnote">$H_2$ 与拓扑的联系值得记住：对每个群 $G$ 存在一个 CW 复形 $BG$（分类空间），使得 $H_* (BG; \mathbb{Z}) \cong H_*(G, \mathbb{Z})$。于是<strong>群同调 = 某个拓扑空间的奇异同调</strong>——代数与几何在 $H_2$ 处真正合流。这也是为什么「群的乘法表看起来很离散，却藏着连续拓扑的信息」。</span>

**一个手算的群上同调**：$G = \mathbb{Z}/2$，$M = \mathbb{Z}/2$（平凡作用），算 $H^2(G, M)$。用循环群的标准结论：对 $G = \langle g \mid g^n \rangle$ 有 $H^2(G, M) \cong M^G \big/ (1 + g + \cdots + g^{n-1})M$。这里 $n = 2$、$M^G = \mathbb{Z}/2$、$1 + g$ 在 $\mathbb{Z}/2$ 上是零映射，故

$$H^2(\mathbb{Z}/2, \mathbb{Z}/2) = \mathbb{Z}/2$$

**这恰好分类 $M$ 被 $G$ 的两种扩张**：分裂的 $\mathbb{Z}/2 \times \mathbb{Z}/2$ 与不可分的 $\mathbb{Z}/4$——与《Ext 与 Tor》里 $\operatorname{Ext}^1_\mathbb{Z}(\mathbb{Z}/2, \mathbb{Z}/2) = \mathbb{Z}/2$ 如出一辙，只是这次主角是群而非模。同调代数的「扩张分类」在模与群两个舞台上，各自上演了同一出戏。

## 3 标准（bar）解析：显式的计算引擎

定义是抽象的，计算却需要显式对象。**bar 解析**给出 $\mathbb{Z}$ 的一个标准自由 $\mathbb{Z}G$-解析：令 $P_n$ 是以 $n$ 元组 $[g_1 \mid g_2 \mid \cdots \mid g_n]$ 为基的自由 $\mathbb{Z}G$-模，微分

$$d[g_1 \mid \cdots \mid g_n] = g_1[g_2 \mid \cdots \mid g_n] + \sum_{i=1}^{n-1} (-1)^i [g_1 \mid \cdots \mid g_i g_{i+1} \mid \cdots \mid g_n] + (-1)^n [g_1 \mid \cdots \mid g_{n-1}]$$

<span class="marginnote">别被公式吓到：每一项就是把「相邻两个元素乘起来、当作一个」，符号按位置交替——和《复形与同调群》里「三角形边界的正负端抵消」是同一件事。<strong>bar 解析就是「把群元素写成词、再让词之间消长」的复形化</strong>，它让 $H^2$、$H_1$ 等低阶群可以被逐项写出，是计算与群论工作者的常备工具。</span>

再取 $H_1$ 验一条：$H_1(G, \mathbb{Z}) = G/[G, G]$。自由群 $F_2$ 的交换化是 $\mathbb{Z}^2$，故 $H_1(F_2, \mathbb{Z}) = \mathbb{Z}^2$——拓扑上 $F_2$ 对应「两点连成的八字形」，其 $H_1$ 恰有两个洞的生成元；而 $G = \mathbb{Z}/2$ 时 $H_1 = \mathbb{Z}/2$，与 $B\mathbb{Z}/2 = \mathbb{RP}^\infty$ 的 $H_1 = \mathbb{Z}/2$ 完全吻合。**「群同调 = 分类空间同调」在这里第二次得到验证。**

对 $S_n$（$n \ge 4$），$H_2(S_n, \mathbb{Z}) = \mathbb{Z}/2$——这正是「$S_n$ 存在唯一二重覆盖 $2\cdot S_n$」的同调回答。Schur 乘子把「群能否被完美群覆盖」与 $H_2$ 直接挂钩：覆盖的存在性等价于 $H_2(G, \mathbb{Z})$ 的某种消没。**$H_2$ 于是成了「群的隐藏结构」的保险柜。**

## 4 Lie 代数同调：包络代数出场

Lie 代数 $\mathfrak{g}$（域 $k$ 上）同构地用**普遍包络代数** $U(\mathfrak{g})$ 代替群环。定义完全平行：

$$H_n(\mathfrak{g}, M) := \operatorname{Tor}_n^{U(\mathfrak{g})}(k, M), \qquad H^n(\mathfrak{g}, M) := \operatorname{Ext}_{U(\mathfrak{g})}^n(k, M)$$

其中 $k$ 是平凡模（$\mathfrak{g}$ 作用为 0）。**Chevalley-Eilenberg 复形**给出显式模型：$H_*(\mathfrak{g}, M)$ 是外代数复形 $\Lambda^\bullet \mathfrak{g} \otimes M$ 的同调，微分来自 $\mathfrak{g}$ 的括号——**Lie 括号在复形里化身为微分**。

**CE 复形是 bar 解析的 Lie 模拟**：群用「词」作基底（bar），Lie 代数用「外幂」作基底（$\Lambda^\bullet \mathfrak{g}$），微分都来自「把相邻两个乘起来」——只是 Lie 的乘法换成了括号，符号从交换律换成了反对称。**同一种「递归乘、交替和」的语法，同时说活了群与 Lie 代数。**

低阶含义与群的情形逐字对应：$H^0 = \mathfrak{g}$-不变量、$H^1 = $ 导子商内导子、$H^2(\mathfrak{g}, M) = $ Lie 代数的扩张分类。

**Whitehead 引理**（半单 Lie 代数的同调消没）：域特征 0 上，$\mathfrak{g}$ 半单、$M$ 有限维时

$$H^1(\mathfrak{g}, M) = 0 = H^2(\mathfrak{g}, M)$$

**推论**：半单 Lie 代数上的每个导子都是内导子，每个扩张都分裂——**半单结构因此「刚性」到没有任何模糊地带**。这是同调代数回赠给 Lie 理论最漂亮的一记重拳：过去需要长篇计算的「Levi 分解」的刚性，如今只是 $H^1 = H^2 = 0$ 的一句话。

**同调维数登场**：本篇与《同调维数与整体维数》的接点在于——群 $G$ 的**上同调维数** $\operatorname{cd} G = \sup\{n \mid H^n(G, M) \ne 0\ \text{对某模 } M\}$ 测「$G$ 的模结构需要多长的解析」。自由群 $\operatorname{cd} = 1$，有限群 $\operatorname{cd} = \infty$。**一个群的「几何维度」，被它的模论维度精确量化。**

## 5 公式解析：H_n(G, M) = Tor_n^{ZG}(Z, M)

把群同调的「总公式」拆成四步，你就掌握了全部套路：

$$
H_n(G, M) = \operatorname{Tor}_n^{\mathbb{Z}G}(\mathbb{Z}, M)
$$

- **第一步，为什么是群环**：群没有天然的加法结构，但 $\mathbb{Z}G$ 有。把 $G$ 编码成环后，「$G$-模」与「$\mathbb{Z}G$-模」是同一回事，之前学的 Tor 原封可用。
- **第二步，为什么是平凡模 $\mathbb{Z}$**：同调要测的是「$G$ 在 $M$ 上留下的痕迹」，而「没有 $G$ 痕迹」的基准就是平凡作用。$\operatorname{Tor}(\mathbb{Z}, M)$ 比较「$M$ 实际携带的 $G$-结构」与「什么都没有的 $\mathbb{Z}$」，差值正是 $G$ 的信息。
- **第三步，为什么是 Tor**：Tor 是右正合张量积的左导出函子；$M_G = M \otimes_{\mathbb{Z}G} \mathbb{Z}$ 只保留了「轨道商」，而 $\operatorname{Tor}_n$ 把「取商时丢失的纠缠」逐层还原成 $H_n$。**群同调 = 轨道商的导出化。**
- **第四步，如何落地**：取 $\mathbb{Z}$ 的 $\mathbb{Z}G$-射影解析（bar 解析即标准选择），作用 $-\otimes_{\mathbb{Z}G} M$，再取同调。整套流水线你已走了八篇——现在只是换了一条流水线、贴上了「群」的标签。

如果你把这篇与前五篇对照着读，会发现**群同调、Lie 代数同调与普通模的 Tor/Ext 没有任何新招数**——「换环 + 换解析」而已。同调代数的统一性，正是它最大的优雅。

## 6 小结

- 群同调/上同调是 $\mathbb{Z}G$ 上的 $\operatorname{Tor}/\operatorname{Ext}$（平凡模 $\mathbb{Z}$）；**群环把群翻译成代数**。
- 低阶含义：$H^0 = $ 不动点，$H_1(G,\mathbb{Z}) = G_{ab}$，$H^1 = $ 导子/内导子，$H^2 = $ 群扩张分类，$H_2 = $ Schur 乘子。
- **bar 解析**给出显式自由解析，是低阶计算的引擎；$H_*(G) \cong H_*(BG)$ 接通拓扑。
- Lie 代数情形用 $U(\mathfrak{g})$ 与 Chevalley-Eilenberg 复形平行定义；**Whitehead 引理**：半单 Lie 代数 $H^1 = H^2 = 0$。
- 一次定义、两处演出：导出函子的框架让「群」与「Lie 代数」共享同一套机器。
- **Shapiro 引理**：诱导模把子群上同调原样搬进大群；bar 解析是显式计算引擎。
- **Whitehead 引理**把半单 Lie 代数的刚性归结为 $H^1 = H^2 = 0$ 一句话。
- 上同调维数 $\operatorname{cd} G$ 把「群的几何维度」翻译成模论的解析长度。

在下一节，我们将往更抽象、也更有几何血肉的方向推进：**单纯方法**——把离散的复形结构（单纯集合）与链复形之间的等价关系（Dold-Kan）搭起来，那是从「抽象同调」走向「空间同调」的桥梁。
