---
title: 双复形与 Künneth 公式
date: 2026-08-11
---

# 双复形与 Künneth 公式

<div class="epigraph">
<p>数学的艺术，是给不同的东西取同一个名字。</p>
<footer>—— 亨利 · 庞加莱（Henri Poincaré）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 同调代数 ｜ 对标教材 Weibel Ch. 2.7, 5.6 ｜ 2026-08-11</p>
</div>

## 为什么从双复形开始

上一篇《谱序列》说「双复形有两个天然的过滤器」，本篇兑现承诺：把双复形这只装置彻底拆开，看它如何同时给出**两条通向同一个答案的路**，并顺手导出整个学科最重要的乘法公式——**Künneth 公式**。

为什么重要？因为数学里几乎所有的「乘积对象」——空间的乘积 $X \times Y$、复形的张量积 $C \otimes D$、群的直积——其同调都不是两个因子同调的平凡乘积。Künneth 公式精确地告诉你：**乘积的同调 = 因子同调的「卷积」+ 一个 Tor 修正**。而更深刻的是，之前讲过的**万有系数定理**只是 Künneth 公式的零维特例。一石三鸟，这节内容密度极高。

## 1 双复形：一张二维账本

**双复形（double complex / bicomplex）**：一族模 $C_{p,q}$（$p, q \in \mathbb{Z}$）连同两个微分——水平微分 $d^h : C_{p,q} \to C_{p-1,q}$ 与垂直微分 $d^v : C_{p,q} \to C_{p,q-1}$，满足 $d^h d^h = 0$、$d^v d^v = 0$，以及关键的**反交换律**

$$d^v d^h + d^h d^v = 0$$

两个微分「走了一个小方块回到原点，两次走法的差别为 0」——这是二维的同调代数：一条边界的边界为零，升级为一个方块的对边差为零。

**例（最重要的来源）**：两个复形 $C, D$ 的**张量积双复形**：$(C \otimes D)_{p,q} = C_p \otimes D_q$，水平微分取 $d^C$、垂直微分取 $d^D$。反交换律自动成立。**空间 $X \times Y$ 的奇异链双复形**通过 Eilenberg-Zilber 定理与 $C_*(X) \otimes C_*(Y)$ 链同伦等价，所以「乘积空间的几何」归结为「张量积双复形的代数」。

一个几何注脚：$C_*(X \times Y)$ 本身并不是严格意义的张量积复形，但 **Eilenberg-Zilber 定理**给出链同伦等价 $C_*(X\times Y) \simeq C_*(X) \otimes C_*(Y)$——用「洗牌」把矩形剖分重排成三角剖分。于是 Künneth 公式直接翻译成拓扑定理：$H_n(X \times Y)$ 由 $\oplus_{p+q=n} H_p(X) \otimes H_q(Y)$ 与一个 Tor 修正拼出（域上无修正）。「乘积空间测洞」从此退化为「因子空间的纯代数运算」。

**辨析｜分裂是意外不是常态**：$H_n = \bigoplus H_p \otimes H_q$ 直接成立需要额外条件（如 $R$ 是域、或 Tor 项为零）。整数环上 $0 \to \mathbb{Z} \xrightarrow{\times 2} \mathbb{Z} \to \mathbb{Z}/2 \to 0$ 不分裂，正是 Tor 项存在的日常形态——**不要默认「乘积的同调 = 同调的乘积」。**

**总复形（total complex）**：把二维账本折叠回一维：

$$(\operatorname{Tot} C)_n = \bigoplus_{p+q=n} C_{p,q}, \qquad d = d^h + d^v$$

<span class="marginnote">检查 $d^2 = 0$ 是全过程的唯一技术动作：$d^2 = (d^h+d^v)^2 = d^hd^h + d^hd^v + d^vd^h + d^vd^v = d^hd^v + d^vd^h$，而反交换律让这两项抵消。<strong>反交换律不是可有可无的装饰，它是 $d^2=0$ 的命门</strong>。</span>

## 2 两个过滤器：同一条河的两种渡法

总复形上天然有两个过滤器：

- **按列过滤**（保留 $p \le$ 常数）：先取垂直同调，再取水平同调；
- **按行过滤**（保留 $q \le$ 常数）：先取水平同调，再取垂直同调。

**双复形的两个标准谱序列**（Weibel 5.6）：两条谱序列都收敛到同一个 $H_*(\operatorname{Tot} C)$，但第 2 页长得很不一样：

$$
{}^I E^2_{p,q} = H^h_p\bigl(H^v_q(C)\bigr), \qquad {}^{II} E^2_{p,q} = H^v_q\bigl(H^h_p(C)\bigr)
$$

**「先横后竖」与「先竖后横」殊途同归**——庞加莱说的「给不同的东西取同一个名字」，在这里有了最硬核的体现。比较这两个谱序列，就是同调代数里一类重要论证（comparison theorems）的模板。

**两个谱序列怎么选**：实践中优先挑「$E^2$ 好算」的那个过滤器。若双复形来自张量积 $C \otimes D$ 且 $C$ 平坦，按列过滤的 $E^2 = H_p(C) \otimes H_q(D)$ 已接近答案；按行过滤的 $E^2 = H_q(\text{先水平取同调})$ 往往更难。**「先横后竖还是先竖后横」不是品味问题，是计算策略问题。**

<span class="marginnote">从计算角度：哪条路好走就走哪条。若其中一个「双复形的行（或列）都是单复形」，相应的谱序列会在第 2 页<strong>退化（collapse）</strong>——退化点就是公式丰收的时刻。Künneth 公式正是「退化 + 分裂」的产物。</span>

## 3 Künneth 公式：乘积的同调

**定理（Künneth）**：设 $R$ 是环，$C, D$ 是复形，且 $C$ 各项平坦（如自由）。则存在**分裂**短正合列

$$0 \to \bigoplus_{p+q=n} H_p(C) \otimes_R H_q(D) \to H_n(C \otimes_R D) \to \bigoplus_{p+q=n-1} \operatorname{Tor}^R_1(H_p(C), H_q(D)) \to 0$$

特别地，当 $R$ 是**域**时 Tor 项消失，得到教科书里最美的公式：

$$\boxed{\,H_n(C \otimes D) \cong \bigoplus_{p+q=n} H_p(C) \otimes H_q(D)\,}$$

**证明的一句话骨架**：在 $C$ 的列过滤谱序列里，第 1 页是 $C_p \otimes H_q(D)$，取 $H_p$ 后 $E^2_{p,q} = H_p(C) \otimes H_q(D)$；若 $C$ 平坦则 $E^2$ 即退化极限页，于是 $H_n = \oplus E^2$，而 Tor 项记录「$C$ 不射影时，扩展/合成中的纠缠」——它与上一篇万有系数定理里的 Tor 完全同源。

**一个把 Tor 项逼出来的例子**：取 $C = D = \{ \mathbb{Z} \xrightarrow{\times 2} \mathbb{Z} \}$（非零项集中在 0、1 维），有 $H_0(C) = H_0(D) = \mathbb{Z}/2$、$H_1(C) = H_1(D) = 0$。Künneth 公式直接读出

$$H_1\bigl(\operatorname{Tot}(C \otimes D)\bigr) = \operatorname{Tor}_1^\mathbb{Z}\bigl(H_0(C), H_0(D)\bigr) = \operatorname{Tor}_1^\mathbb{Z}(\mathbb{Z}/2, \mathbb{Z}/2) = \mathbb{Z}/2$$

**张量积项 $\oplus H_1\otimes H_0 \oplus H_0 \otimes H_1$ 全部为零，这个 $H_1$ 完全由「挠的纠缠」贡献。** 若换成域系数，同样的输入只给出平凡的 $\mathbb{Z}/2$，两者一对照，Tor 修正的存在感拉满——「乘积的同调不总是乘积」的最经济例证。

**拓扑例**：$H_*(S^m \times S^n; \mathbb{Q})$。域上 Künneth 给出 $H_k = \mathbb{Q}$ 当 $k \in \{0, m, n, m+n\}$，其余 0——两个环面的乘积 $T^2 = S^1 \times S^1$ 于是 $H_0 = H_2 = \mathbb{Q}$、$H_1 = \mathbb{Q}^2$，与「环面有两个洞」的几何直觉严丝合缝。

反过来，「乘积空间」的直观也回馈代数：$H_n(X \times Y)$ 里的每一个同调类都可分解为「$X$ 的类 × $Y$ 的类」的和——这是同调意义下的「张量分解」。

## 4 万有系数定理是 Künneth 的特例

把 $D$ 取成「只有零维、等于 $M$」的平凡复形：$D_0 = M$，$D_n = 0$（$n \ne 0$）。则 $C \otimes D$ 就是 $C \otimes M$，而 Künneth 公式退化为

$$0 \to H_n(C) \otimes M \to H_n(C \otimes M) \to \operatorname{Tor}_1(H_{n-1}(C), M) \to 0$$

——正是上一篇的**万有系数定理**。同理，对偶地取 $\operatorname{Hom}$ 版的 Künneth（注意方向反转），退化出 Ext 版 UCT。

**一条公式覆盖三个定理**（UCT、UCT 的对偶、Künneth），这正是双复形「二维视角」的回报：把 0 维的特例放进 2 维的框架里，所有「修正项」都现出 Tor/Ext 的原形。

这条「降维打击」的路径值得记住：**先把问题写成双复形，再让谱序列替你做苦力**。同调代数里大量「看似无关的公式」，都是同一张二维账本的不同折叠方式。

## 5 公式解析：分裂 SES 的四个符号

把 Künneth 的核心公式拆开：

$$
0 \to \bigoplus_{p+q=n} H_p(C)\otimes H_q(D) \xrightarrow{\;\alpha\;} H_n(C\otimes D) \xrightarrow{\;\beta\;} \bigoplus_{p+q=n-1} \operatorname{Tor}_1(H_p(C), H_q(D)) \to 0
$$

- **第一步，读卷积**：$\bigoplus_{p+q=n} H_p \otimes H_q$ 是把「$C$ 的第 $p$ 个洞」与「$D$ 的第 $q$ 个洞」配对，总维数 $p+q=n$。这恰是**柯西卷积** $\sum_{p+q=n} a_p b_q$ 的代数版：乘积的同调类 = 因子同调类的卷积。
- **第二步，读 $\alpha$**：$\alpha$ 把每个 $[c] \otimes [d]$ 送到张量积同调类 $[c \otimes d]$。$\alpha$ 总是**单**的（分裂 SES 的左端），直观上「卷积出来的类」不会互相冲突。
- **第三步，读 Tor 修正**：$\beta$ 的核是 $\alpha$ 的像；剩余部分由 $\operatorname{Tor}_1(H_p C, H_q D)$（$p+q=n-1$）填充。**Tor 项把「上一维的纠缠」投影进本维**——和万有系数定理中 $\operatorname{Tor}_1(H_{n-1}, M)$ 的位次完全一致。
- **第四步，何时无修正**：$R$ 为域，或一侧复形各维平坦且同调自由时，Tor 项为 0，SES 变为同构。**「干净域上：乘积同调 = 同调卷积」是默认心智模型，Tor 是环变复杂时的修正。**

**末了对照**：UCT 与 Künneth 的修正项为何都是 $\operatorname{Tor}_1$（而非更高阶）？因为「平凡复形 $M$」与「两个因子复形」的谱序列在第 2 页都已退化，$E^2$ 只剩一层，Tor 修正只可能出现在 $\operatorname{Tor}_1$。**谱序列退化的程度，决定了公式的简洁程度。** 一个活的例子：$\operatorname{Tor}^\mathbb{Z}_1(\mathbb{Z}/2, \mathbb{Z}/2) = \mathbb{Z}/2 \ne 0$，所以 $C = D = \{\mathbb{Z} \xrightarrow{2} \mathbb{Z}\}$（都是 $\mathbb{Z}/2$ 的解析）的张量积同调里，$H_1$ 会多出一个 $\mathbb{Z}/2$——这在域上不会发生，唯有整数环的挠能造出它。

## 6 小结

- **双复形**有两套微分 $d^h, d^v$ 与反交换律 $d^vd^h + d^hd^v = 0$；**总复形** $(\operatorname{Tot}C)_n = \oplus_{p+q=n} C_{p,q}$ 折叠回一维。
- 张量积 $C \otimes D$ 与乘积空间 $X \times Y$（经 Eilenberg-Zilber）都是双复形的来源。
- 两个过滤器给出**两个谱序列**，均收敛到 $H_*(\operatorname{Tot} C)$：先横后竖 = 先竖后横。
- **Künneth 公式**：$0 \to \oplus H_p C \otimes H_q D \to H_n \to \oplus \operatorname{Tor}_1(H_p C, H_q D) \to 0$；域上化为纯卷积同构。
- **万有系数定理是 Künneth 的特例**（$D = M$ 平凡复形）。
- 谱序列退化的程度决定公式的简洁程度：退化即 Künneth/UCT 式的显式公式；分裂不是常态。

在下一节，我们将转向导出函子最迷人的应用现场：给**群**与 **Lie 代数**装上同调——用 $\mathbb{Z}G$ 模与包络代数的语言，重写前五篇的全部机器。
