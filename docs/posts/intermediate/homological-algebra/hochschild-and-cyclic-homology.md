---
title: Hochschild 与循环同调（选读）
date: 2026-08-11
---

# Hochschild 与循环同调（选读）

<div class="epigraph">
<p>那么时间是什么？没人问我时，我明白；一旦要向发问的人解释，我就不知道了。</p>
<footer>—— 奥古斯丁（Augustine of Hippo, "Confessions"）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 同调代数 ｜ 对标教材 Weibel Ch. 9 ｜ 2026-08-11</p>
</div>

## 为什么从 Hochschild 开始

选读不意味着冷门——**Hochschild 同调**是非交换代数版「微分形式」的第一套完整理论，而**循环同调**是它的「时间化」版本，是非交换几何的基石，也是代数 K 理论里 Chern 特征的归宿。它们把前九篇的工具最后拧成一股绳：既然空间可以翻译成代数（连续函数环），那么「代数的同调」就应该反过来读出「空间的几何」。

把这句话落到实处的是两个惊人的事实：$C^\infty(M)$ 的 Hochschild 同调就是 $M$ 上的微分形式 $\Omega^n(M)$；它的循环同调就是 $M$ 的 de Rham 上同调。**代数与分析（光滑流形）在此同框**——而这一切都从「导子」这个最古典的概念出发。

## 1 导子与 Hochschild 复形

设 $A$ 是 $k$-代数。**导子（derivation）** $D : A \to M$（$M$ 是 $A$-双模）满足莱布尼茨律 $D(ab) = aD(b) + D(a)b$。导子是微分的代数灵魂——但它们住在哪？答案是**双模**里。记 $A^e = A \otimes A^{\mathrm{op}}$，则 $A$-双模就是 $A^e$-模，而 $A$ 自身是 $A^e$-模（$a \cdot m \cdot b$）。于是定义

$$\boxed{\,HH_n(A, M) := \operatorname{Tor}_n^{A^e}(A, M)\,}$$

**Hochschild 复形**是显式模型：$C_n(A, M) = M \otimes A^{\otimes n}$，边界映射 $b = \sum_{i=0}^n (-1)^i d_i$，其中 $d_0(m \otimes a_1 \otimes \cdots \otimes a_n) = ma_1 \otimes a_2 \otimes \cdots$、$d_i$ 把相邻两个 $a$ 相乘、$d_n$ 用最右的 $a_n$ 从右边作用 $m$。**每一项都是「删掉一个点、合并两段」**，与单纯方法里的面映射同出一辙——Hochschild 复形本质上是某个「自由单纯双模解析」的产物。

**历史注记**：Hochschild 1945 年的论文原本研究「扩张与导子的分类」，其中一条引理（Hochschild-Konig-Smith）今天成了整个学科的地基。半个世纪后 Connes 给它装上 $B$ 算子，Hochschild 同调才从「交换情形的好工具」升级为「非交换情形的必需品」。

把 $\operatorname{Tor}_n^{A^e}(A, M)$ 再拆开读一遍：它说的是「以 $A^e$ 为环、以 $A$ 与 $M$ 为模取 Tor」。这里 $A$ 是**对角模**——左乘右乘都用 $A$ 自身，$M$ 则是任意双模。**Hochschild 同调 = 对角模与双模之间的纠缠**，与群同调里「平凡模与 $M$ 的纠缠」结构完全平行：同一行公式，换掉环、换掉第二个变元，就生成同调代数里一半的公式。

低阶读出：$HH_0(A, M) = M / [A, M]$（$[A,M]$ 由 $am - ma$ 张成）；$HH_1(A, M)$ 与导子商内导子有关。取 $M = A$ 时，$HH_0(A) = A/[A,A]$——**零阶 Hochschild 同调就是「交换化」**，与 $H_1(G,\mathbb{Z}) = G_{ab}$ 遥相呼应。

两个立刻能算的例子。**矩阵代数** $A = M_n(k)$：$[A, A] = \mathfrak{sl}_n(k)$（迹为零的矩阵），故 $HH_0(M_n(k)) = k$——**Hochschild 同调的第一课就是「迹」：零维类只认矩阵的迹**，这为下文 Chern 特征落在循环同调埋下伏笔。**多项式环** $A = k[x]$：$HH_0 = k[x]$、$HH_1 = k[x]\,dx$——一维 Hochschild 类就是「一个微分」，正是 HKR 定理的最初模样。

<span class="marginnote">记号 $A^e = A \otimes A^{\mathrm{op}}$ 是 Hochschild 理论的总开关：它把「双模」这个需要两只手的东西，压缩成「一只手的模」。<strong>$HH_n$ 定义中「对 $A^e$ 取 Tor、第二个变元用 $A$」</strong>——这句口诀背熟，就拿到了整套理论的钥匙。</span>

## 2 Hochschild 同调 = 非交换微分形式

**Hochschild-Kostant-Rosenberg 定理（HKR）**：对正则交换 $k$-代数 $A$（例如多项式环），

$$HH_n(A) \cong \Omega^n_{A/k} \quad \text{（Kähler 微分形式）}$$

于是多项式环 $k[x_1, \dots, x_d]$ 的 Hochschild 同调就是**外代数**：$HH_n = \Lambda^n(k^d)$ 的自由 $A$-模。特例一目了然：$HH_0(k[x]) = k[x]$，$HH_1(k[x]) = k[x]\,dx$——**一维 Hochschild 类就是「一个微分」**。

对光滑流形 $M$，取 $A = C^\infty(M)$，HKR 的几何形态为：$HH_n(A) \cong \Omega^n(M)$，且 Hochschild 边界 $b$ 对应 de Rham 微分 $d$。**换句话说：流形上的微分几何，在代数世界里就是 Hochschild 同调。** 这就是「非交换几何」的起点：把「空间 = 函数环」的信条反过来用——哪怕这个「环」对应不上任何几何空间（例如量子群的函数环），照样可以做「非交换的微分几何」。

当 $A$ 不是交换正则代数时，$HH_n$ 不再等于 $\Omega^n$，而变成「微分形式的非交换修正」——这正是非交换几何的原材料。例如 Weyl 代数 $k\langle x, y\rangle/(xy - yx - 1)$ 的 $HH_*$ 比多项式环丰富得多，因为它「记得」$xy \ne yx$。

还有一个经典的对照：对群代数 $A = k[G]$（$G$ 有限），$HH_*(A)$ 与「共轭类的循环」有关——$HH_0(k[G])$ 以共轭类为基。**Hochschild 同调把「群的结构」翻译成「迹与共轭」的语言。**

<span class="marginnote">HKR 定理的名字值得记住，它是「代数-几何字典」里最重要的一页：交换情形下 Hochschild = 微分形式。而一旦离开交换情形，$HH_n$ 就测量「非交换性造成的偏差」——这偏差正是 Connes 循环同调要追的「时间」。</span>

## 3 循环同调：给同调装上时间

Hochschild 同调忘了「循环对称」：$a_0 \otimes a_1 \otimes \cdots \otimes a_n$ 与「把下标循环移位」$a_1 \otimes \cdots \otimes a_n \otimes a_0$ 之间没有自然联系。**Connes 算子** $B$ 把循环移位补回来：

$$B : HH_n(A) \to HH_{n+1}(A), \qquad B(a_0 \otimes \cdots \otimes a_n) = \sum_{i=0}^n (-1)^{ni}\left(1 \otimes a_i \otimes \cdots \otimes a_n \otimes a_0 \otimes \cdots \otimes a_{i-1}\right)$$

并满足 $B^2 = 0$、$bB + Bb = 0$——于是 $(b, B)$ 张成一个**双复形**（循环双复形），其总复形的同调就是**循环同调（cyclic homology）** $HC_n(A)$。

**SBI 精确列**（Connes）：循环同调、Hochschild 同调与移位之间有精确列

$$\cdots \to HC_n(A) \to HH_n(A) \xrightarrow{\;B\;} HC_{n-2}(A) \to HC_{n-1}(A) \to \cdots$$

它把「时间」（$HC$）与「瞬间切片」（$HH$）按每两维周期性联系起来——这就是**Connes 周期性**：$HC_n$ 与 $HC_{n-2}$ 之间隔着一个「乘法」同构（对幂等元生成的情形）。**循环同调 = 带上 $S^1$-作用（时间流）的 Hochschild 同调**，几何对应物是自由环空间 $LM$ 的 $S^1$-等变同调。

一个能读到底的例子：取 $A = k$（单点代数）。$HH_0(k) = k$、$HH_n(k) = 0$（$n \ge 1$），而循环同调给出**周期性的双排**：$HC_{2r}(k) = k$、$HC_{2r+1}(k) = 0$。SBI 序列在这里化身为「每两维重现一次」的周期律——Connes 周期性的最简标本。几何上它对应「一个点的自由环空间 = $S^1$」，而 $S^1$ 的 $S^1$-等变上同调正是这样的周期双排。**代数与几何各自算出的结果，分毫不差。**

**Connes 周期性的几何意义**：$HC_{n+2}(A)$ 与 $HC_n(A)$ 的周期等同，是「$S^1$ 旋转两圈 = 恒等」的代数回声。物理上它对应「玻色子与费米子的统计互换」，KMS 态与量子统计的循环迹——**一个 $B$ 算子，把数学的周期性与物理的自旋统计接在一起**。

**SBI 序列的三件套**：$S$（遗忘时间）、$B$（转动时间）、$I$（注入）——三个算子各管一职，把「瞬间」与「时间」的账本互相换算。背下 SBI，就等于背下了循环同调的骨架。

## 4 应用：Chern 特征与代数 K 理论

循环同调最壮观的出场在**代数 K 理论**：对环 $R$，有**Chern 特征**

$$\operatorname{ch} : K_n(R) \longrightarrow HC_n(R)$$

它把「代数 K 群」（$GL_n(R)$ 的高阶同伦群，极难计算）映射进「循环同调」（相对可控）。这给 K 理论提供了同伦论之外第二条可算的路径，也是**非交换几何**（Connes）里「对没有点集的几何做积分」的基石。

**Morita 不变性**：Hochschild 同调与循环同调都是 Morita 不变量——$A$ 与 $M_n(A)$ 的 $HH_*$、$HC_*$ 完全相同。因此「取矩阵」不改变任何同调信息，$HH_0$ 只记迹的洞察在此升格为一般定理。**换壳的代数，测不出矩阵的壳。**

**与 de Rham 的终极对照**：流形 $M$ 上 $HC_* (C^\infty(M))$ 的周期化版本正是 $H_{dR}(M) \otimes k[u^{\pm 1}]$，且 $B$ 对应 $d$。**「循环同调 = 带时间参数的 de Rham」**——一句话概括本篇的几何归宿。

对光滑流形 $M$，循环同调与 de Rham 的对应关系最终落到：$HC_* (C^\infty(M))$ 的周期化版本与 $H^\bullet_{dR}(M) \otimes k[u^{\pm 1}]$ 同构，且 Connes 算子 $B$ 在微分形式上就是外微分 $d$。**一个抽象的代数不变量，精确地重放了流形的全部 de Rham 数据**——「从极限到大模型」的数学旅程，在这里与物理（量子场论里的正则化、迹映射）接上了头。

**辨析｜易错点**：$HH_*$ 与 $HC_*$ 只差一个算子 $B$，初学者常把两者混为一谈。记住三句话：$HH_n$ 是「切片同调」（瞬时），$HC_n$ 是「滚动同调」（含时间）；$HH$ 由 $b$ 单独定义，$HC$ 由 $(b, B)$ 双复形定义。**没有 $B$ 就没有周期性与 Chern 特征**——一个记号之差，隔着「切面」与「轨道」两重世界。

## 5 公式解析：边界算子 b 的四步

把 Hochschild 边界 $b : M \otimes A^{\otimes n} \to M \otimes A^{\otimes n-1}$ 拆开：

$$
b(m \otimes a_1 \otimes \cdots \otimes a_n) = ma_1 \otimes a_2 \otimes \cdots \otimes a_n + \sum_{i=1}^{n-1} (-1)^i\, m \otimes a_1 \otimes \cdots \otimes a_i a_{i+1} \otimes \cdots \otimes a_n + (-1)^n a_n m \otimes a_1 \otimes \cdots \otimes a_{n-1}
$$

- **第一步，三项各管一摊**：第一项（$d_0$）把最左的 $a_1$ 与模块结构合并；中间项（$d_i$）把相邻两因子乘起来（删掉一个因子）；最后一项（$d_n$）把最右的 $a_n$ 从右边作用到 $m$ 上。
- **第二步，交替符号**：$(-1)^i$ 保证「先删 $i$ 再删 $j$」与「先删 $j$ 再删 $i$」在求和后抵消——$b^2 = 0$ 的全部秘密在符号，与第一、九篇完全同一套原理。
- **第三步，为什么商掉 $[A, M]$**：$HH_0 = M/b(M \otimes A)$ 里的关系 $am - ma \in \operatorname{im} b$（来自 $d_0$ 与 $d_1$ 的配对），所以**零维 Hochschild 类 = 「可交换的痕迹」**——这正是迹映射 $\operatorname{Tr}$ 的本质，Chern 特征最终落在循环同调，也是因为「迹」天然循环。
- **第四步，与 $B$ 合流**：$B$ 在相邻的 $HH_*$ 之间穿行，与 $b$ 组成双复形。**$b$ 管「瞬间微分」，$B$ 管「循环滚动」；两者合起来，时间与空间一起被计入同调。** 这四步合起来，就是非交换微积分的「牛顿第二定律」。

一个快速验算：$A = k[x]$、$n = 1$、$M = A$，则 $C_1 = A \otimes A$，$b(a \otimes b) = ab \otimes 1 - a \otimes b$（中间项 $d_1$ 把 $a, b$ 乘起来）。逐项写开就能看到「首项与末项如何合并、中间项如何抵消」——**手工写一次 $b^2 = 0$ 的抵消过程，胜过读十遍定义**。

末了提醒：本篇的「选读」标签只表示它在经典同调代数教材里的排位靠后，**在当代（非交换几何、代数 K 理论、拓扑递归论）它已是标配**。读完这篇，你对「同调 = 测洞」的认知，将升级为「同调 = 测量一切可迹结构」。

## 6 小结

- **Hochschild 同调** $HH_n(A) = \operatorname{Tor}_n^{A^e}(A, A)$，Hochschild 复形 $C_n = M \otimes A^{\otimes n}$、$b = \sum (-1)^i d_i$。
- $HH_0 = $ 交换化 $A/[A,A]$；**HKR 定理**：交换正则代数上 $HH_n \cong \Omega^n$（微分形式）。
- **循环同调** $HC_n$ 由 Connes 算子 $B$ 与 $(b,B)$ 双复形定义，**SBI 精确列**与周期性刻画它与 $HH$ 的关系。
- $C^\infty(M)$ 的 $HH = \Omega^n(M)$、$HC = H_{dR}(M)$：**代数不变量重放流形几何**。
- Chern 特征 $K_n \to HC_n$ 把代数 K 理论接进循环同调，是非交换几何的地基。
- $HH_*$ 与 $HC_*$ 只差一个算子 $B$：$HH$ 是切片、$HC$ 是轨道，勿混。
- Morita 不变性、Connes 周期性（$HC_{n+2}$ 与 $HC_n$ 型同构）与 SBI 序列是循环同调的三根支柱。

在下一节，我们将完成这次远航的最后一跃：把十篇以来的「复形、同伦、导出函子」整体收进一个更高的范畴——**导出范畴**。三角结构会替我们回答：为什么同调不变量才是「真正的不变量」。
