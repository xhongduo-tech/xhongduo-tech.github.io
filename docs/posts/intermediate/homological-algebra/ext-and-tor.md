---
title: Ext 与 Tor
date: 2026-08-11
---

# Ext 与 Tor

<div class="epigraph">
<p>做数学的艺术，在于找到那个已经孕育了普遍性的一切胚芽的特例。</p>
<footer>—— 大卫 · 希尔伯特（David Hilbert）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 同调代数 ｜ 对标教材 Weibel Ch. 3 ｜ 2026-08-11</p>
</div>

## 为什么从 Ext 与 Tor 开始

上一篇《导出函子》给出了配方，这一篇用配方烘焙两块招牌点心：**Ext 与 Tor**。它们各自承载一项经典使命——

- **$\operatorname{Ext}$**：测「同态空间的洞」，并一举解决古典的**扩张问题**（extension problem）：把两个模「拼」成第三个，有多少种拼法；
- **$\operatorname{Tor}$**：测「张量积的洞」，并成为**挠（torsion）**的天然探测器。

更妙的是它们共享一条黄金性质：**平衡性（balancing）**——同一个量可以从「左变元」或「右变元」两个方向去算，殊途同归。这套「一个不变量、两条路」的模式，正是同调代数的优雅所在，也直接通向拓扑里的**万有系数定理**。

## 1 Ext：扩张的分类器

**定义**：设 $M, N$ 是 $R$-模，

$$\operatorname{Ext}^i_R(M, N) := R^i\operatorname{Hom}_R(M, -)(N) \cong R^i\operatorname{Hom}_R(-, N)(M)$$

第一个等号用 $N$ 的内射解析（Hom 的第一个变元…这里的记号要小心，见公式解析）；第二个等号是**平衡定理（balancing theorem）**：对 $M$ 取射影解析与对 $N$ 取内射解析，得到**同一组**上同调。低阶展开：

- $\operatorname{Ext}^0_R(M, N) = \operatorname{Hom}_R(M, N)$：第零个导出函子还原为 Hom；
- $\operatorname{Ext}^1_R(M, N)$ 精确分类 $M$ 被 $N$ 扩张的方式；
- 更高阶项 $\operatorname{Ext}^i$（$i \ge 2$）编码「扩张的扩张」之间更高阶的障碍。

<span class="marginnote">「扩张问题」是群论与模论最古老的追问：已知 N 与 M，想知道所有满足 $0 \to N \to E \to M \to 0$ 的中间对象 $E$。古典做法是逐个试（Schreier 定理），而 $\operatorname{Ext}^1$ 一次性给出全部答案，且带代数结构——这是「代数不变量终结手工枚举」的教科书级案例。</span>

**核心例子**：$\operatorname{Ext}^1_\mathbb{Z}(\mathbb{Z}/2, \mathbb{Z}/2) = \mathbb{Z}/2$，即 $\mathbb{Z}/2$ 被 $\mathbb{Z}/2$ 扩张，**恰好有两种**：分裂扩张 $\mathbb{Z}/2 \oplus \mathbb{Z}/2$（对应 $0$）与不可分扩张 $\mathbb{Z}/4$（对应非零类）。一个平凡到可以心算的环，藏着完全不同的两种结构——Ext 把它们一眼分开。

这个 $\mathbb{Z}/2$ 是怎么算出来的？取 $\mathbb{Z}/2$ 的射影解析 $0 \to \mathbb{Z} \xrightarrow{\times 2} \mathbb{Z} \to \mathbb{Z}/2 \to 0$，作用 $\operatorname{Hom}_\mathbb{Z}(-, \mathbb{Z}/2)$ 得复形 $0 \to \operatorname{Hom}(\mathbb{Z}/2, \mathbb{Z}/2) \to \operatorname{Hom}(\mathbb{Z}, \mathbb{Z}/2) \xrightarrow{0} \operatorname{Hom}(\mathbb{Z}, \mathbb{Z}/2) \to 0$。因为 $\operatorname{Hom}(\mathbb{Z}, \mathbb{Z}/2) = \mathbb{Z}/2$ 且 $\times 2$ 诱导的映射是零映射，故

$$H^0 = \operatorname{Hom}(\mathbb{Z}/2, \mathbb{Z}/2) = \mathbb{Z}/2, \qquad H^1 = \mathbb{Z}/2$$

**这个非零的一阶上同调，对应的正是不可分扩张 $\mathbb{Z}/4$**——「$H^1$ 与扩张一一对应」在最小例子上得到教科书式的验证。

## 2 Tor：挠的探测器

**定义**：

$$\operatorname{Tor}^R_i(M, N) := L_i(-\otimes_R N)(M) \cong L_i(M \otimes_R -)(N)$$

同样有**平衡性**。第零阶 $\operatorname{Tor}_0^R(M, N) = M \otimes_R N$。高阶 Tor 度量张量积「单射性失败」的量。

Tor 的名字来自 **torsion（挠）**。对整数环 $\mathbb{Z}$ 有一个极好的公式：

$$\operatorname{Tor}^\mathbb{Z}_1(\mathbb{Z}/m, \mathbb{Z}/n) = \mathbb{Z}/\gcd(m, n)$$

而更有启发性的对比是这个：

$$
\mathbb{Q} \otimes_\mathbb{Z} \mathbb{Z}/2 = 0, \qquad \operatorname{Tor}^\mathbb{Z}_1(\mathbb{Q},\, \mathbb{Z}/2) = \mathbb{Z}/2
$$

**张量积把 $\mathbb{Z}/2$ 的挠性彻底抹平（商掉后为 0），Tor 却精确地把它「留下」（$\mathbb{Z}/2$）**。直觉：$M \otimes N$ 里分母互相「踩平」，而 $\operatorname{Tor}_1$ 保存了「本应对齐却对不齐」的那部分纠缠——这正是挠在代数里的相貌。

<span class="marginnote">在拓扑里 Tor 的「挠」与几何的「挠」遥相呼应：万有系数定理里，$\operatorname{Tor}_1(H_{n-1}(X), \mathbb{Z}/2)$ 描述的正是「上一维的洞以二阶系数缠绕而成的零维影」。名字不是巧合。</span>

## 3 长正合列与计算路径

Ext 与 Tor 都对两个变元分别满足长正合列。例如固定 $N$，SES $0 \to A \to B \to C \to 0$ 给出

$$
\cdots \to \operatorname{Ext}^i_R(C, N) \to \operatorname{Ext}^i_R(B, N) \to \operatorname{Ext}^i_R(A, N) \to \operatorname{Ext}^{i+1}_R(C, N) \to \cdots
$$

**辨析｜易错点**：$\operatorname{Ext}$ 对第一个变元是**逆变**的，所以长正合列里箭头方向是 $C \to B \to A$；对第二个变元是**共变**的，方向是 $A \to B \to C$。Tor 则两个变元都共变。**动手写长正合列前，先确认变元的方向。** 把方向写反是初学阶段最高频的错误。

**计算路径**：平衡性让计算者有两条路可选——挑更容易的那个变元做解析。例如算 $\operatorname{Ext}^1_\mathbb{Z}(\mathbb{Z}/2, \mathbb{Q})$，对 $\mathbb{Z}/2$ 取射影解析 $0 \to \mathbb{Z} \xrightarrow{2} \mathbb{Z} \to \mathbb{Z}/2 \to 0$，作用 $\operatorname{Hom}(-, \mathbb{Q})$ 后每个 $\mathbb{Z} \to \mathbb{Q}$ 项全消失，直接得到 $\operatorname{Ext}^1 = 0$——这比直接解析 $\mathbb{Q}$ 快得多。

实战铁律：**先试「另一个变元的解析」**。平衡定理最大的红利不是理论，而是「哪边好算用哪边」的自由——遇到一个 Ext 算不动，翻转变元常常柳暗花明。

## 4 拓扑的回报：万有系数定理

Ext 与 Tor 最漂亮的「回本」，是给拓扑学一个免费的定理。设 $C_\bullet$ 是自由阿贝尔群构成的链复形（如拓扑空间的奇异链复形），$M$ 是任意阿贝尔群，则

$$0 \to H_n(C_\bullet) \otimes M \to H_n(C_\bullet \otimes M) \to \operatorname{Tor}^\mathbb{Z}_1(H_{n-1}(C_\bullet), M) \to 0$$

与对偶版本（用 Ext）：

$$0 \to \operatorname{Ext}^1_\mathbb{Z}(H_{n-1}(C_\bullet), M) \to H^n(C_\bullet; M) \to \operatorname{Hom}_\mathbb{Z}(H_n(C_\bullet), M) \to 0$$

<span class="marginnote">这两条叫<strong>万有系数定理（universal coefficient theorem）</strong>。它的意义在于「拆解」：想算带系数 $M$ 的同调，不必重新搭整个奇异复形，只要知道整数系数的 $H_n$ 与 $H_{n-1}$，再各补一个 Tor / Ext 修正项。<strong>Tor 修正「下探一维的挠」，Ext 修正「上探一维的核」</strong>——这就是标题里「万有」的含义。</span>

再回头品味：**Tor 之所以能修正 $H_{n-1}$，正是因为它右正合地保留了张量积丢掉的东西；Ext 之所以能修正 $H^{n-1}$，正是因为它左正合地保留了 Hom 丢掉的东西。** 一切环环相扣。

## 5 公式解析：Ext^1 与扩张的对应

把最富信息量的一条公式拆开：

$$
\operatorname{Ext}^1_R(M, N) \;\cong\; \{\, 0 \to N \to E \to M \to 0 \,\} \big/ \sim
$$

- **第一步，读懂右侧**：右端是所有以 $N$ 为子模、以 $M$ 为商模的短正合列，商掉等价关系 $\sim$：两个扩张等价，若存在 $E \to E'$ 使整个三层图交换。等价类 = 「本质不同的拼法」。
- **第二步，类到元素**：给定扩张 $0 \to N \to E \to M \to 0$，对它作用 $\operatorname{Hom}(-, N)$ 得长正合列，其中连接同态恰给一个 $\operatorname{Ext}^1$ 类——这是「蛇引理 → 导出函子 → Ext 分类」三级火箭的末端。
- **第三步，什么时候是零类**：扩张**分裂**（$E \cong N \oplus M$）当且仅当对应 $\operatorname{Ext}^1$ 类为零。于是「不可分扩张的个数」= $|\operatorname{Ext}^1_R(M, N)|$（有限时）。
- **第四步，Baer 和**：两类扩张之间还可「竖着拼接、横着取商」地定义加法，使这个集合成为真正的群——这正是 $\operatorname{Ext}^1$ 作为导出函子天然携带的群结构。**分类不只是一个集合，而是一个群**，这是抽象代数的极高回报。

末了提醒：$\operatorname{Ext}^1$ 分类扩张这件事，对「模」用 $\operatorname{Ext}^1_R$、对「群」用 $H^2(G, -)$（见《群同调与 Lie 代数同调》）、对「Lie 代数」用 $H^2(\mathfrak{g}, -)$——**「扩张分类」是同调代数送给全数学的统一礼物**，三场戏共用同一套剧本：一个二阶上同调群，一场分类的全部秘密。

## 6 小结

- $\operatorname{Ext}^i_R(M,N)$ 是 Hom 的右导出函子，$\operatorname{Tor}^R_i(M,N)$ 是张量积的左导出函子。
- **平衡定理**：两个变元任选一边解析，结果相同——计算有两条路。
- $\operatorname{Ext}^1$ 分类 $M$ 被 $N$ 的**扩张**（分裂 ⇔ 零类）；$\operatorname{Tor}^\mathbb{Z}_1(\mathbb{Q},\mathbb{Z}/2)=\mathbb{Z}/2$ 揭示张量积丢掉挠、Tor 保住挠。
- 长正合列对每个变元成立；**注意 Ext 第一变元的方向是逆变的**。
- **万有系数定理**：$H_n(C\otimes M)$ 由 $H_n \otimes M$ 与 $\operatorname{Tor}_1(H_{n-1}, M)$ 拼出（上同调版用 Ext）。

在下一节，我们将问一个「多少步才够」的问题：把每个模都切开需要多长的解析？答案通向**同调维数与整体维数**——以及希尔伯特那条震惊世界的合冲定理。
