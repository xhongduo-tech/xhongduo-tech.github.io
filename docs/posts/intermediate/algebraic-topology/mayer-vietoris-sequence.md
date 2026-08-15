---
title: Mayer–Vietoris 序列及应用
date: 2026-08-07
---

# Mayer–Vietoris 序列及应用

<div class="epigraph">
<p>要理解一个复杂对象，先把它拆成两块简单的，再精确说明两块怎么接上。</p>
<footer>—— 沃尔夫冈 · 迈尔（Walther Mayer）与莱奥波德 · 维托里斯（Leopold Vietoris）</footer>
</div>

<div class="article-byline">
<p>第二级 · 代数拓扑 ｜ Hatcher 第2.2章 ｜ 2026-08-07</p>
</div>

## 为什么从 Mayer–Vietoris 序列开始

前几篇的计算都带一个共同模式：把空间**拆开**算。球面切成南北半球，$\mathbb{RP}^n$
逐层剥骨架。这个「分而治之」的思路值得被提炼成一条通用的机器。对基本群，它叫 Van Kampen 定理（本专题第 1
篇）；对同调，它就是**Mayer–Vietoris 序列**——代数拓扑里出现频率最高的单个计算工具。

它的设定极其简单：$X$ 被两个子空间 $A, B$ 的内部盖满（$X = \operatorname{int} A \cup
\operatorname{int} B$）。那么 $X$ 的同调由 $A$、$B$、$A \cap B$
的同调拼出来，拼合的规则是一条正合序列。**你只需要知道三块简单空间的同调，就能算出整块空间的同调**——这正是上一节胞腔同调之外的另一种「同调算法」，两者配合几乎能计算一切常见空间。


Mayer–Vietoris
序列的魅力在于它几乎「免费」：你只需要把空间拆成两块开集，剩下的全部自动化。它也因此成为同调论里出现频率最高的单个工具——从球面到射影空间、从图到流形，凡是「能拆成两块的」都归它管。读这一节时请建立两个习惯：**第一，遇到空间先想怎么拆**（哪两块开集能盖满它）；**第二，拆完先看交叠**（$A
\cap B$ 的同调越简单，结果越干净）。这两个习惯会陪伴你完成今后绝大多数同调计算。

## 1 定理陈述：同调版的 Van Kampen

**定理（Mayer–Vietoris 序列）。** 设 $X$ 是 $A$ 与 $B$ 的内部之并，$A, B \subseteq X$，则存在正合序列

$$\cdots \xrightarrow{\partial_*} H_n(A \cap B) \xrightarrow{\ \Phi\ } H_n(A) \oplus H_n(B) \xrightarrow{\ \Psi\ } H_n(X) \xrightarrow{\ \partial_*\ } H_{n-1}(A \cap B) \xrightarrow{\Phi} \cdots$$

$$\cdots \to H_0(A) \oplus H_0(B) \xrightarrow{\Psi} H_0(X) \to 0$$

其中三个映射是：

- $\Phi(x) = \big(i_*(x),\ -j_*(x)\big)$：把 $A \cap B$ 的类分别塞进 $A$ 与 $B$，且**第二个分量取负号**；
- $\Psi(u, v) = k_*(u) - l_*(v)$：把 $A$ 与 $B$ 的类分别塞进 $X$，**相减**；
- $\partial_*$ 是连接同态：把 $X$ 里的循环「切开」，沿着 $A \cap B$ 追回一个低一维的类。

**条件**是 $X = \operatorname{int} A \cup \operatorname{int} B$，不能只要求 $X = A \cup
B$。这个条件保证「$X$ 上的链」可以被细分到「每小段全落在 $A$ 里或全落在 $B$
里」，从而可以按落在哪块归组——这是序列成立的几何前提。<span class="marginnote">记号 $H_n(A) \oplus H_n(B)$
是直和：$A$ 与 $B$ 各自贡献的同调类<strong>独立</strong>地记录，同调类 $([\alpha], [\beta])$
表示同时携带两个空间的洞。$\Phi$ 里的负号是为了让 $\Phi$ 与 $\Psi$ 在直和中配平正合性——纯粹是符号技术，别被它吓住。</span>

**与 Van Kampen 的对照**：$\pi_1$
版本处理非交换的群，用的是「自由积模掉关系」；同调版本处理交换群，用的是「直和」加正合序列。同调不选基点、天然交换，所以 Mayer–Vietoris 比
Van Kampen 干净得多，这也再次体现同调群相对基本群的优势。

## 2 证明线索：一切从切除来

Mayer–Vietoris 序列不是新的公理，而是**切除定理的直接推论**。线索如下。

令 $Z$ 为 $A \setminus B$ 的一个「内部保护带」——更确切地说，构造两个同伦等价替换 $A' \hookrightarrow
A$、$B' \hookrightarrow B$，使得 $A' \cup B' = X$ 且可用切除。具体地，令 $A' = X -
\overline{C}$、$B' = X - \overline{D}$ 取合适的闭子集，使 $\{A', B'\}$ 覆盖 $X$ 的内部且交叠 $A'
\cap B'$ 同伦等价于 $A \cap B$。则包含映射 $(A', A' \cap B') \hookrightarrow (X, B)$
满足切除定理条件，给出同构：

$$H_n(A', A' \cap B') \cong H_n(X, B)$$

把 $H_n(X, B)$ 的长正合序列与 $H_n(A', A' \cap B')$ 的同构拼起来，就得到 Mayer–Vietoris 序列。<span class="marginnote">这个证明展示了「正合序列 + 切除」作为基本构件的威力：只要能把问题化为「某个相对同调的同构 +
两条长正合序列」，就能拼出新的正合序列。同调论的整个大厦就是这样一层层拼起来的。</span>

## 3 立即应用：楔和、去点与黏合

**例 1：楔和（wedge sum）。** 设 $X, Y$ 带基点，取 $X \vee Y$（在基点处粘成一点）。令 $A$ 为 $X$ 加上 $Y$
的一个小邻域（同伦等价于 $X$），$B$ 为 $Y$ 加上 $X$ 的小邻域（同伦等价于 $Y$），则 $A \cap B$
可缩。Mayer–Vietoris 序列中，$H_n(A \cap B) = 0$（$n \ge 1$）使连接同态消失，立即得到：

$$\widetilde{H}_n(X \vee Y) \cong \widetilde{H}_n(X) \oplus \widetilde{H}_n(Y) \qquad (n \ge 1)$$

「两空间粘在一个点上，同调直接相加」——这句话之前是靠直觉，现在是定理。

**例 2：去掉一个点的流形。** 设 $M$ 是 $n$-维闭流形，$M \setminus \{p\}$。把 $M$ 拆成 $A = M
\setminus \{p\}$（同伦等价于我们要算的）与 $B$ 为 $p$ 的闭圆盘邻域（可缩），$A \cap B$ 同伦等价于
$S^{n-1}$。Mayer–Vietoris 序列立即给出 $H_k(M) \cong H_k(M \setminus \{p\})$（$k \lt n-1$），而在 $k = n-1$ 处出现 $\mathbb{Z}$ 项，交代「挖掉一点留下一个 $n-1$-维洞」。<span class="marginnote">这解释了直觉：$S^n$ 去掉一点变成 $\mathbb{R}^n$（可缩），而 $S^n \setminus
\{p\} \simeq D^n \simeq *$ 的同调为 0；但 $M$ 挖点后保留的低维同调来自 $M$
自身的骨架，只有最高维被破坏。</span>

## 4 公式解析：Mayer–Vietoris 序列的用法

$$\cdots \to H_n(A \cap B) \xrightarrow{\Phi} H_n(A) \oplus H_n(B) \xrightarrow{\Psi} H_n(X) \xrightarrow{\partial_*} H_{n-1}(A \cap B) \to \cdots$$

- **第一步，找 $\ker \Psi$**：正合性给 $\ker \Psi = \operatorname{im} \Phi$。即「$A$、$B$ 的同调类拼成 $X$ 里零类的，恰是来自交叠的那些」。这告诉我们 $H_n(X)$ 的生成元是从 $A$ 或 $B$ 来的，且交叠部分的冗余被精确模掉。
- **第二步，找 $\operatorname{coker} \Phi$**：$\operatorname{coker} \Phi \cong \operatorname{im} \Psi \subseteq H_n(X)$。于是 $H_n(X) \cong (H_n(A) \oplus H_n(B))/\operatorname{im}\Phi$ **模掉**连接同态可能引入的低维修正——当 $H_{n-1}(A \cap B) = 0$ 或 $\partial_*$ 平凡时，$H_n(X)$ 就完全由直和商掉 $\operatorname{im}\Phi$ 给出。
- **第三步，处理连接同态**：$\partial_*$ 是唯一「跨维」的信息通道。计算中先猜直和商的形式，再检查 $\partial_*$ 是否非零——$\mathbb{RP}^n$ 的挠、$\mathbb{CP}^n$ 的偶维生成元，都来自对 $\partial_*$ 或粘合度的精细追踪。

**辨析｜易错点：** $\Phi$ 的第二分量带负号、$\Psi$ 是相减，二者配合才能保证「$\operatorname{im}\Phi
\subseteq \ker \Psi$」。初学时常把 $\Psi$ 写成相加 $k_\* + l_\*$——那样序列一般不正合。符号不是装饰，是正合性的命脉。


**例：用 Mayer–Vietoris 再算一遍 $H_*(S^n)$。** 拆成南北开圆盘 $U_+ \simeq *$、$U_- \simeq
*$，交叠 $U_+ \cap U_- \simeq S^{n-1}$。序列在 $k \ge 2$ 处：$0 \to H_k(S^n) \to
H_{k-1}(S^{n-1}) \to 0$，故 $H_k(S^n) \cong H_{k-1}(S^{n-1})$；$k = 1$ 处从 $H_0$
的序列给出 $\widetilde{H}_1(S^n) \cong \widetilde{H}_0(S^{n-1})$。逐层剥到
$S^0$，得到与对序列相同的答案。**两条路（对序列 /
Mayer–Vietoris）殊途同归**——它们共享同一套切除机制，只是「切」的位置不同，这本身就是对切除定理可靠性的双重确认。

**例：环带与去点平面。** 设 $X$ 是环带 $\{1 \le \lvert z \rvert \le 2\}$，拆成内半与外半两块，交叠可缩。得
$\widetilde{H}_1(X) = \mathbb{Z}$、其余约化同调为 0。**一个直观但值得验证的结论**：$X$ 与 $S^1$
同伦等价（径向收缩到内边界），Mayer–Vietoris 正确读出了这一点。计算工具能自动发现同伦等价，是「不变量忠于伦型」的活证据。

**与 Van Kampen 的再对照**：基本群版本需要「自由积模掉关系」，非交换让计算很容易陷入关系化简；同调版本是直和 +
正合，全程线性。**同一道「拆开再拼」的题，交换世界（同调）比非交换世界（$\pi_1$）容易得多**——这也是为什么许多几何结论先在同调层面被证明、再试图提升到基本群层面。

## 5 小结

- **Mayer–Vietoris 序列**：$X = \operatorname{int} A \cup \operatorname{int} B$ 时，$\cdots \to H_n(A \cap B) \to H_n(A) \oplus H_n(B) \to H_n(X) \to H_{n-1}(A \cap B) \to \cdots$ 正合。
- **映射**：$\Phi = (i_*, -j_*)$、$\Psi = (k_*, -l_*)$，连接同态 $\partial_*$ 降一维。
- **来源**：切除定理 + 两条长正合序列的合成，不是新公理。
- **应用**：$\widetilde{H}_n(X \vee Y) \cong \widetilde{H}_n(X) \oplus \widetilde{H}_n(Y)$