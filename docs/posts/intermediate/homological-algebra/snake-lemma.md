---
title: 蛇引理与连接同态
date: 2026-08-11
---

# 蛇引理与连接同态

<div class="epigraph">
<p>一个好的证明，是那种让我们变得更聪明的证明。</p>
<footer>—— 尤里 · 马宁（Yuri I. Manin, "Mathematics as Metaphor"）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 同调代数 ｜ 对标教材 Weibel Ch. 1.3 ｜ 2026-08-11</p>
</div>

## 为什么从蛇引理开始

上一篇《复形与同调群》以「长正合列」收尾，却留下一个未解之谜：那个神秘的**连接同态** $\delta_n$ 到底长什么样？凭什么它是精确的？答案集中在一个以「蛇」命名的引理里——**蛇引理（snake lemma）**。它是同调代数中**引用率最高的单条引理**：长正合列、Mayer-Vietoris 序列、相对同调、Künneth 公式，乃至群同调里的 Shapiro 引理，全都是它的一次次「变奏」。

这一节我们把它彻底驯服：先看陈述，再做一次完整的「通勤图追踪」（diagram chasing）证明，最后体会它如何像一条装配线，把「核」与「余核」精确地焊接成长正合列。这与代数中的**群论**、**线性代数**一脉相承：求解的失败（核）与多解（余核）总是成对出现，蛇引理告诉我们对偶地度量它们。

## 1 蛇引理：通勤图里的精确链

先摆出主角。设有交换图，上下两行**精确**：

$$
\begin{array}{ccccccccc}
0 & \xrightarrow{} & A & \xrightarrow{\;f\;} & B & \xrightarrow{\;g\;} & C & \xrightarrow{} & 0 \\
 & & \Big\downarrow{\alpha} && \Big\downarrow{\beta} && \Big\downarrow{\gamma} \\
0 & \xrightarrow{} & A' & \xrightarrow{\;f'\;} & B' & \xrightarrow{\;g'\;} & C' & \xrightarrow{} & 0
\end{array}
$$

**蛇引理**：存在一条**精确**的长序列

$$
\ker\alpha \;\xrightarrow{\;}\; \ker\beta \;\xrightarrow{\;}\; \ker\gamma \;\xrightarrow{\;\delta\;}\; \operatorname{coker}\alpha \;\xrightarrow{\;}\; \operatorname{coker}\beta \;\xrightarrow{\;}\; \operatorname{coker}\gamma
$$

其中 $\ker\alpha \to \ker\beta$ 与 $\ker\beta \to \ker\gamma$ 由横向箭头诱导，$\operatorname{coker}\alpha \to \operatorname{coker}\beta$ 与 $\operatorname{coker}\beta \to \operatorname{coker}\gamma$ 由纵向箭头的余诱导，而**正中间的 $\delta : \ker\gamma \to \operatorname{coker}\alpha$ 就是连接同态**——它不属于图中任何现成的箭头，必须「绕」出来。

![蛇引理通勤图：横向两行精确，纵向箭头 α、β、γ，红色虚线 δ 即连接同态](/images/homological-algebra/snake-lemma-1.svg)

图中那条从 $C$ 蜿蜒回到 $A'$ 的曲线，正是「蛇」的名字由来：它盘在 $C$ 的核上，绕到 $B$ 上提起来，再落到 $A'$ 的余核里。

## 2 构造连接同态：一次完整的通勤图追踪

$\delta$ 的构造是理解整个引理的关键，我们把每一步走一遍。设 $c \in \ker\gamma \subseteq C$。

- **第一步（抬起）**：因为 $g$ 满射，存在 $b \in B$ 使 $g(b) = c$。$b$ 是 $c$ 的一个「提升」。
- **第二步（下落）**：把 $b$ 沿 $\beta$ 送到 $b' = \beta(b) \in B'$。由交换性 $g'\beta = \gamma g$，有 $g'(b') = \gamma(g(b)) = \gamma(c) = 0$，所以 $b' \in \ker g' = \operatorname{im} f'$。
- **第三步（再落）**：因为 $f'$ 单射限制为 $A' \to \operatorname{im} f'$ 的同构，存在唯一 $a' \in A'$ 使 $f'(a') = b'$。
- **第四步（商掉歧义）**：定义 $\delta(c) = a' + \operatorname{im}\alpha \in \operatorname{coker}\alpha$。

**辨析｜易错点**：前两步的「提升」$b$ 与「回拉」$a'$ 都**不唯一**，这正是为什么 $\delta$ 的取值落在**余核**里——若换了 $b$ 或 $a'$，差出来的部分恰好落在 $\operatorname{im}\alpha$ 中，商掉后类不变。**「不唯一的选择」被精确地商掉，是同调代数最常用的一招。**

<span class="marginnote">这条「下、左、下」的之字形路径处处都要用上行的满射性 $g$ 与下行的单射性 $f'$——丢掉任何一个，「蛇」就断了。所以蛇引理的前提（上行右端满、下行左端单）恰好是够用且必要的。</span>

## 3 核与余核：被蛇咬住的一对「测量器」

要真懂蛇引理，得先懂它的两个端点。**核（kernel）** $\ker\alpha = \{a \mid \alpha(a) = 0\}$ 测「丢失的信息」；**余核（cokernel）** $\operatorname{coker}\alpha = A'/\operatorname{im}\alpha$ 测「多出来的信息」。两者是同一个同态的对偶两端：

若 $\alpha$ **单**，则 $\ker\alpha = 0$；
- 若 $\alpha$ **满**，则 $\operatorname{coker}\alpha = 0$；
- **精确性** $\ker\alpha \to \ker\beta \to \ker\gamma \xrightarrow{\delta} \operatorname{coker}\alpha \to \cdots$ 把这些「测量器」首尾相接，任何一个环节不精确，就意味着我们漏掉了某种几何或代数信息。

由「核单、余核满」两个平凡方向还能得到一个常用判据：若 $\alpha$ 单且 $\gamma$ 满，则 $\delta = 0$ 当且仅当「$\ker\beta \to \ker\gamma$ 满」当且仅当「$\operatorname{coker}\alpha \to \operatorname{coker}\beta$ 单」——三个等价条件如同三根指针，齐齐指向「SES 是否分裂」的同一个答案。

<span class="marginnote">对偶性是本站课程的常客：线性代数里的「秩-零度定理」$\dim V = \operatorname{rank} + \operatorname{nullity}$，正是「满射部分 + 丢失部分」的加性表述；到了同调代数这里，核与余核不再是数字，而是<strong>对象</strong>，加性等式升级为<strong>精确列</strong>——这就是范畴论语言登场的时刻。</span>

## 4 蛇的威力：长正合列、Mayer-Vietoris 与相对同调

把蛇引理应用到**复形的短正合列**（每个指标 $n$ 一行图），就逐层得到上一篇的长正合列。$C_n$ 层的连接同态 $\delta_n$ 正是蛇引理中那条「蛇」在每个维度上的体现，而长序列的精确性一劳永逸地被蛇引理保证。

两个立即的应用值得点名：

- **相对同调**：对拓扑空间偶 $(X, A)$，$0 \to C_\bullet(A) \to C_\bullet(X) \to C_\bullet(X)/C_\bullet(A) \to 0$ 是复形的 SES，蛇引理给出
$$\cdots \to H_n(A) \to H_n(X) \to H_n(X, A) \xrightarrow{\partial} H_{n-1}(A) \to \cdots$$
这告诉我们：**把 $A$ 的部分挖掉后，「新长出来的洞」与「低一维的洞」之间始终守恒**——一种朴素的「拓扑守恒律」。

- **Mayer-Vietoris 序列**：把空间 $X = U \cup V$ 拆成两个开集，重叠部分的同调通过蛇引理被精确地嵌进 $H_n(U) \oplus H_n(V)$ 与 $H_n(X)$ 之间。**「整体洞 = 局部洞 + 重叠修正」**，是测洞学里的加法交换律。

**连接同态是自然的（natural）**：给定两个 SES 之间的一族纵向链映射 $\alpha, \beta, \gamma$，$\delta$ 与之可交换——「蛇」在整片通勤图里可以整体移动。这条自然性正是长正合列成为**函子性机器**（而非一次性计算）的保证：谱序列、导出函子与群上同调全部靠它才站得住脚。

**一个可手算的例子**：把蛇引理放到整数模上验算。取上、下两行同为 $0 \to \mathbb{Z} \xrightarrow{\times 2} \mathbb{Z} \xrightarrow{\mathrm{mod}\,2} \mathbb{Z}/2 \to 0$，纵向取 $\alpha = \beta = \gamma = \times 2$。先检查交换性：$g'\beta(n) = 2n \bmod 2 = 0 = \gamma g(n)$，$f'\alpha(n) = 4n = \alpha f(n)$，全图通勤。由于 $\gamma$ 在 $\mathbb{Z}/2$ 上是零映射，立即算出

$$\ker\alpha = \ker\beta = 0, \qquad \ker\gamma = \mathbb{Z}/2, \qquad \operatorname{coker}\alpha = \operatorname{coker}\beta = \operatorname{coker}\gamma = \mathbb{Z}/2$$

蛇引理断言 $0 \to \mathbb{Z}/2 \xrightarrow{\;\delta\;} \mathbb{Z}/2 \to \mathbb{Z}/2 \to \mathbb{Z}/2 \to 0$ 精确。逐项追踪 $\delta(1)$：把 $1 \in \ker\gamma$ 提升为 $b = 1 \in \mathbb{Z}$（$g(b) = 1 \bmod 2$），作 $\beta(b) = 2$，再由 $f' = \times 2$ 回拉得 $a' = 1$，于是

$$\delta(1) = [1] \in \operatorname{coker}\alpha = \mathbb{Z}/2$$

**$\delta$ 恰为恒等映射**——序列精确性一目了然：$\delta$ 满（核为 0），其后第一个映射是 0（核 $= \mathbb{Z}/2$），第二个映射是恒等（核 $= 0$）。「蛇」在这组小数字上完全显形：核列与余核列被一条非平凡的 $\delta$ 精确焊死。

## 5 公式解析：δ 的四步走

连接同态 $\delta : \ker\gamma \to \operatorname{coker}\alpha$ 的构造是整个学科最经典的公式化动作，拆成四步：

$$
c \in \ker\gamma \;\xrightarrow{\text{抬}}\; b \in B,\; g(b)=c \;\xrightarrow{\text{落}}\; \beta(b) \in B' \;\xrightarrow{\text{回拉}}\; a' \in A',\; f'(a')=\beta(b) \;\xrightarrow{\text{商}}\; \delta(c) = a' + \operatorname{im}\alpha
$$

- **为什么能抬**：$g$ 满射（上行右端精确 $B \twoheadrightarrow C$）。
- **为什么抬完必能落**：交换性 $g'\beta = \gamma g$，配合 $c \in \ker\gamma$，保证 $\beta(b) \in \ker g' = \operatorname{im} f'$。
- **为什么回拉唯一**：$f'$ 单射（下行左端精确 $0 \to A'$）。
- **为什么最后要商**：抬与回拉的**选择不唯一**，但任意两次选择的差都落在 $\operatorname{im}\alpha$，故余核类 $\delta(c)$ 良定义。

记牢这四步的节奏（**抬-落-回拉-商**），你就在同调代数里掌握了第一套肌肉记忆。之后 Ext、Tor 里的连接同态、谱序列里的边缘同态，全是它换了马甲的版本。

## 6 小结

- **蛇引理**把 SES 的核列与余核列焊接成一条精确列，中间唯一的「非现成」箭头是**连接同态** $\delta$。
- $\delta$ 的构造是「**抬-落-回拉-商**」四步走；选择的不唯一由余核商掉。
- 前提条件恰是**上行右端满、下行左端单**——每个假设都在构造里被用到。
- 复形 SES 的长正合列、相对同调、Mayer-Vietoris 序列都是蛇引理的应用。
- 连接同态是**自然的**：它与链映射可交换，这让长正合列成为可「整体移动」的机器。
- 若 $\alpha$ 单且 $\gamma$ 满，则「$\delta = 0$」「$\ker\beta \to \ker\gamma$ 满」「$\operatorname{coker}\alpha \to \operatorname{coker}\beta$ 单」三者等价，指向 SES 是否分裂的同一答案。

在下一节，我们将把「解析 + 商掉歧义」这套哲学从图论层面升级为**函子层面**：用「射影对象代替、再取同调」的方式定义一系列高阶函子——**导出函子**，它是 Ext 与 Tor 的母亲。
