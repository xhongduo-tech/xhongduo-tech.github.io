---
title: 局部上同调
date: 2026-08-07
---

# 局部上同调

<div class="epigraph">
<p>局部上同调把「支集」「深度」「维数」与「对偶」融成一条同调之河。</p>
<footer>—— Alexander Grothendieck（1961 年巴黎 IHÉS 讲义精神）</footer>
</div>

<div class="article-byline">
<p>第二级 · 交换代数 ｜ Matsumura 补充章 / Eisenbud Ch. 18 ｜ 2026-08-07</p>
</div>

## 为什么从局部上同调开始

我们一路走来积累了大量「局部」工具：支集（哪里非零）、深度（多深）、维数（多宽）、相伴素（病历卡）、Gorenstein（自对偶）。**局部上同调（local cohomology）**是这些线索的最后收束：Grothendieck 1961 年把「只看支集在 $\{\mathfrak{m}\}$ 附近」的取节函子求右导出，得到一群 $H^i_{\mathfrak{m}}(M)$，然后发现——**深度、维数、对偶性全部排队来报到**。<span class="marginnote">「局部上同调」名字的来历：它把「支撑在闭集 $Z$ 内的截面」做成导出函子。Grothendieck 在 1961 年战后的巴黎讲了一学期的讲义（后由 Robin Hartshorne 整理成书），从此它成为现代交换代数与代数几何的标配语言。</span>

这一篇是**选读**：不求全证明，只求把「为什么一切汇聚于此」讲透。学完它，你会看到一个漂亮的闭环——深度、维数、CM、Gorenstein、局部对偶，原来是同一群上同调在不同阶数上的投影。

## 1 支撑在一点：$\Gamma_{\mathfrak{m}}$

设 $(R, \mathfrak{m})$ Noether 局部环，$M$ 是 $R$-模。

**$\mathfrak{m}$-挠子模**：

$$\Gamma_{\mathfrak{m}}(M) = \{m \in M \mid \mathfrak{m}^t m = 0 \text{ 对某个 } t \geq 1\} = \bigcup_t \operatorname{Ann}_M(\mathfrak{m}^t).$$

**重点：$\Gamma_{\mathfrak{m}}$ 是左正合函子，但不右正合，其右导出函子就是局部上同调：**

$$H^i_{\mathfrak{m}}(M) = R^i \Gamma_{\mathfrak{m}}(M), \qquad H^0_{\mathfrak{m}}(M) = \Gamma_{\mathfrak{m}}(M).$$

直观上 $H^i_{\mathfrak{m}}(M)$ 度量「$M$ 中支撑恰落在 $\{\mathfrak{m}\}$ 的第 $i$ 阶部分」。等价定义：$H^i_{\mathfrak{m}}(M) = \varinjlim_t \operatorname{Ext}^i_R(R/\mathfrak{m}^t, M)$（把《深度》的 $\operatorname{Ext}$ 判据「补」到所有幂）；也可用 Čech 复形计算。<span class="marginnote">为什么「导出」重要：$\Gamma_{\mathfrak{m}}$ 只抓「一步被 $\mathfrak{m}$ 幂杀死」的部分，直接取会丢信息；右导出把「被杀掉时损失的高阶信息」全部捡回。这正是上同调一贯的使命——<strong>在取不动点时把损失记成高阶项</strong>，与第一级《同调代数》的取不动点直觉一脉相承。</span>

标准例子（$R = k[x]$，$\mathfrak{m} = (x)$，$M = R$）：$H^0_{\mathfrak{m}}(R) = 0$（整环里没有非零元被 $x$ 幂杀死）；$H^1_{\mathfrak{m}}(R) = k[x, x^{-1}]/k[x]$，即「只在原点之外有极点的有理函数模掉多项式」——这是「挖掉原点」在代数里的回声，也解释了几何里的「洞」。

再算一个 $\mathbb{Z}$-模的例子：$M = \mathbb{Z}/p^n\mathbb{Z}$，$\mathfrak{m} = (p)$。$\Gamma_{(p)}(M) = M$（每个元素都被 $p^n$ 杀死），故 $H^0_{(p)}(M) = \mathbb{Z}/p^n$、$H^1_{(p)}(M) = 0$；反过来对 $M = \mathbb{Z}$：$\Gamma_{(p)}(\mathbb{Z}) = 0$，而 $H^1_{(p)}(\mathbb{Z}) = \mathbb{Z}[1/p]/\mathbb{Z}$。**「有挠的部分被 $H^0$ 抓住，无挠的部分溢出到 $H^1$」——上同调在给「挠」分级**，这正是第1篇《相伴素与支集》里「病历卡」的同调化。

**辨析｜易错点：** $\Gamma_{\mathfrak{m}}$ 抓的是「被 $\mathfrak{m}$ 的**某次幂**杀死」的元素，不是「被 $\mathfrak{m}$ 一步杀死」的 socle。$R = k[x,y]/(x^2, xy)$ 中 $\bar{x}$ 同时落在两者里，但「一步杀死」（$\operatorname{Hom}(k,R)$）与「幂杀死」（$\Gamma_{\mathfrak{m}}$）是两个不同的子模——深度 0 的两种判据分别对应它们（见《深度与正则序列》）。

**核心对照表：局部上同调收束了什么**

| 概念（先前章节） | 局部上同调身份 |
| --- | --- |
| 深度 | 首个非零 $H^i_{\mathfrak{m}}$ 的下标 |
| 维数 | 最后一个非零下标（完备情形） |
| CM 模 | 只有中间那个非零 |
| Gorenstein | 顶部 $H^d_{\mathfrak{m}}$ 与内射包相连 |
| 支集 | 支撑在 $\{\mathfrak{m}\}$ 的部分 |
| 相伴素 | 决定 $H^i$ 的非零位置 |

这张表是本文的路线图：前五篇各自独立的「深浅宽窄」，最后在一条 $H^i_{\mathfrak{m}}$ 序列上排好队。

## 2 深度与维数的上同调身份

**重点（深度定理）**：$M \neq 0$ 有限生成时，

$$\operatorname{depth} M = \min\{\, i \mid H^i_{\mathfrak{m}}(M) \neq 0\,\}.$$

用 $R = k[x]_{(x)}$ 落到具体：$H^0_{(x)}(R) = 0$（整环无挠），$H^1_{(x)}(R) = k[x,x^{-1}]/k[x] \neq 0$，首个非零下标是 1——深度 1，与正则序列 $\{x\}$ 的长度一致。$R$ 是 DVR（第1篇《离散赋值环》），也是 CM 环（深度 = 维数 = 1）。**两条计算路径——数正则序列、数非零上同调——在这里给出同一个答案。**

**重点（维数定理，Grothendieck 消失）**：$H^i_{\mathfrak{m}}(M) = 0$ 对 $i > \dim M$，且 $M \neq 0$ 有限生成于完备环上时 $H^{\dim M}_{\mathfrak{m}}(M) \neq 0$。于是

$$\dim M = \max\{\, i \mid H^i_{\mathfrak{m}}(M) \neq 0\,\} \qquad (M \text{ 完备情形}).$$

**于是深度与维数变成同一条序列的两端：** 深度 = 首个非零上同调的下标，维数 = 最后一个非零上同调的下标。**CM 模恰好是「中间全空」的模：$H^i_{\mathfrak{m}}(M) = 0$ 当且仅当 $i \neq \dim M$。**<span class="marginnote">这是《Cohen–Macaulay》一篇最漂亮的同调化：CM = 上同调只在一个下标非零。深度与维数从「数正则序列/链」变成「数非零上同调」，几何直觉（横截、无洞）彻底代数化。</span>

**辨析｜易错点：** 维数定理的「$H^{\dim M} \neq 0$」需要适当的有限生成与完备化前提；非完备环上顶部上同调可能消失（需取完备化恢复）。**用「最后一个非零下标 = 维数」时，先确认环完备或模有限生成。**

## 3 顶部上同调与局部对偶

顶部 $H^{d}_{\mathfrak{m}}(M)$（$d = \dim M$）信息量最大。先看最基本的模 $k = R/\mathfrak{m}$：

$$H^{d}_{\mathfrak{m}}(k) = \begin{cases} k & d = 0,\\ 0 & d > 0. \end{cases}$$

而顶部上同调与**内射包**相连：$H^d_{\mathfrak{m}}(R)$ 是 $k$ 的内射包 $E_R(k)$ 的「推广」——这是 Gorenstein 判据的同调形态（对比《Cohen–Macaulay》公式解析的 $\operatorname{Ext}^d(k, R) \cong k$）。

**局部对偶定理（Local Duality）**：设 $(R, \mathfrak{m})$ 是完备局部环（含剩余域 $k$），$\omega$ 是其规范模（$\omega = H^d_{\mathfrak{m}}(R)^\vee$，$\vee$ 表 **Matlis 对偶** $(-)^\vee = \operatorname{Hom}_R(-, E_R(k))$），则对有限生成 $M$：

$$H^i_{\mathfrak{m}}(M)^\vee \;\cong\; \operatorname{Ext}^{d-i}_R(M, \omega) \qquad (0 \leq i \leq d).$$

**重点：局部对偶把「支撑在一点的上同调」与「全局的 Ext」互相转换。** 顶部情形 $i = d$ 给出 $H^d_{\mathfrak{m}}(M)^\vee \cong \operatorname{Hom}_R(M, \omega)$——规范模就是「对偶空间的代表元」，Serre 对偶在完备局部环上的化身。<span class="marginnote">Matlis 对偶是「局部版本的向量空间对偶」：$E_R(k)$ 扮演「局部 $k$」的角色。Grothendieck 对偶（代数几何里对任意簇的 Serre 对偶）正是局部对偶在整体空间的纤维化——本专题的最后一颗珠，串起《张量积》《深度》《CM/Gorenstein》所有线索。</span>

**辨析｜易错点：** 局部对偶要求 $R$ **完备**。非完备环先取 $\widehat{R}$ 再张量（$H^i_{\mathfrak{m}}(M) \otimes_R \widehat{R} = H^i_{\widehat{\mathfrak{m}}}(M \otimes \widehat{R})$）——「完备化与上同调可交换」正是《完备化》一篇埋下的伏笔。

## 4 公式解析：深度 = 首个非零上同调

把「深度」这条最常用公式拆开：

$$\operatorname{depth} M = \min\{ i \mid H^i_{\mathfrak{m}}(M) \neq 0\}, \qquad H^i_{\mathfrak{m}}(M) = \varinjlim_t \operatorname{Ext}^i_R(R/\mathfrak{m}^t, M).$$

- **第一步，$i = 0$ 对照**：$H^0_{\mathfrak{m}}(M) = \Gamma_{\mathfrak{m}}(M)$ 是非零 ⇔ 有元素被 $\mathfrak{m}$ 的某幂杀死 ⇔ $\operatorname{depth} M = 0$。而 $\operatorname{Ext}^i(R/\mathfrak{m}, M)$（《深度》的 $i=0$）只查「$\mathfrak{m}$ 一步杀死」——极限 $\varinjlim_t$ 把「某幂」补全。
- **第二步，为什么是极限**：$R/\mathfrak{m}^t$ 随 $t$ 增大，「被 $\mathfrak{m}^t$ 杀死」的要求逐层放宽；$\operatorname{Ext}^i(R/\mathfrak{m}^t, M)$ 度量「第 $i$ 阶扩张被 $t$ 阶截断杀掉」的程度，$t \to \infty$ 把所有幂一网打尽。这个极限在 $i$ 上保持左正合，正是右导出函子 $R^i\Gamma_{\mathfrak{m}}$ 的化身。
- **第三步，与正则序列握手**：正则序列每延长一个元素，$\operatorname{Ext}^i(R/\mathfrak{m}^t, \cdot)$ 的消失起始下标就推后一位（长正合列 + 归纳），极限后就是「首个非零 $H^i$ 的下标 = 深度」。**「数正则序列」与「数非零上同调」在此完全统一。**

**辨析｜易错点：** $\operatorname{depth} M = \min\{i : H^i \neq 0\}$ 中「$H^i = 0$」对 $i < \operatorname{depth}$ 是对**所有**有限生成 $M$ 的普适事实；但「$H^{\dim M} \neq 0$」只在完备化/有限生成前提下稳定。**记住：深度管「第一个非零」，维数管「最后一个非零」，中间空 ⇔ CM。**

把「深度管第一个、维数管最后一个」用一句话串起来：深度是「上同调从哪个下标开始出现」，维数是「到哪个下标结束」。二者之间的空段恰是「CM 的空」——CM 模的上同调只有一个非零项。**局部上同调把整门交换代数的「深浅宽窄」压进了一条序列。**

**术语速查表**

| 术语 | 一句话含义 |
| --- | --- |
| $\Gamma_{\mathfrak{m}}(M)$ | 被 $\mathfrak{m}$ 幂杀死的子模 |
| 局部上同调 $H^i_{\mathfrak{m}}$ | $\Gamma_{\mathfrak{m}}$ 的右导出函子 |
| 深度 | 首个非零 $H^i$ 的下标 |
| 维数 | 最后一个非零下标（完备） |
| 顶部上同调 $H^d_{\mathfrak{m}}$ | 与内射包、Gorenstein 相连 |
| Matlis 对偶 | $\operatorname{Hom}_R(-, E_R(k))$ |
| 局部对偶 | $H^i_{\mathfrak{m}}(M)^\vee \cong \operatorname{Ext}^{d-i}(M,\omega)$ |

## 5 小结

- **$\Gamma_{\mathfrak{m}}$**（被 $\mathfrak{m}$ 幂杀死的子模）是左正合函子；其右导出 $H^i_{\mathfrak{m}}(M)$ 即**局部上同调**，等价于 $\varinjlim_t \operatorname{Ext}^i(R/\mathfrak{m}^t, M)$。
- **深度** = 首个非零 $H^i_{\mathfrak{m}}$ 的下标；**维数** = 最后一个非零下标（完备情形）；**CM ⇔ 只有中间那个非零**。
- 顶部上同调 $H^d_{\mathfrak{m}}(R)$ 联系内射包与 Gorenstein 判据；**局部对偶** $H^i_{\mathfrak{m}}(M)^\vee \cong \operatorname{Ext}^{d-i}(M, \omega)$ 把局部与全局对偶接通。
- 完备化与局部上同调可交换；局部对偶需完备前提。

至此，本专题从理想、模、同态出发，穿过局部化、准素分解、链条件、零点定理、Dedekind 环、完备化、维数、Koszul 复形、平坦性、整扩张、深度、CM/Gorenstein、相伴素，最终在局部上同调处汇成闭环。若你想继续，可以沿着 **Grothendieck 对偶**与**导出范畴**再上一个台阶——那是第二级《代数几何》的入口；也可以回到第一级《线性代数》与《同调代数》，把这里的「正合列」「导出函子」语言与基础接榫。交换代数的地图已经画完，但每条路都通向更远的地平线。
