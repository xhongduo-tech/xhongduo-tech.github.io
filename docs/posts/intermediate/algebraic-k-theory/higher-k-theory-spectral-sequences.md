---
title: 高阶 K 群的谱序列
date: 2026-08-07
---

# 高阶 K 群的谱序列

<div class="epigraph">
<p>谱序列是算同伦群的十进制算法。</p>
<footer>—— 约翰·米尔诺（John Milnor）</footer>
</div>

<div class="article-byline">
<p>第二级 · 代数 K 理论 ｜ Weibel《The K-book》§8 ｜ 2026-08-07</p>
</div>

## 为什么需要谱序列

前面几节给了高阶 K 群的定义（+-构造、Q-构造）和理论武器（Resolution、Devissage、Localization），但要**真的算出** $K_n(X)$，还缺一台引擎。谱序列正是这台引擎：它把「一个复杂对象被滤过成一堆简单小块」这件事，翻译成「从 $E_1$ 页开始逐页取同调」的机械流程。<span class="marginnote">谱序列诞生于 Leray 与 Koszul，被 Serre 在谱序列的循环计算中发扬光大；在 K 理论里，它是 Quillen、Brown–Gersten、Atiyah–Hirzebruch 计算高阶群的共同母机。「从极限到大模型」的读者可以把谱序列想成<strong>逐层逼近</strong>：每一页 $E_r$ 是前页同调的新一轮修正，最终稳定在答案的「分次版」。</span>

本节聚焦两座最常用的谱序列：**余维滤过谱序列**（算概形的 K 群）与 **Atiyah–Hirzebruch 谱序列**（算拓扑 K 群）。它们一个从「点上的 K 群」拼出整体的 K 群，一个从「奇异上同调」拼出 K 群——K 理论的计算由此有了统一的手艺。

## 1 谱序列速览

**谱序列（spectral sequence）** 是一列「页」$(E_r, d_r)$：每一页是一个双分次对象，配有微分 $d_r$ 满足 $d_r^2 = 0$；下一页是上一页的同调：

$$
E_{r+1}^{p,q} = \frac{\ker\big(d_r: E_r^{p,q} \to E_r^{p+r,\, q-r+1}\big)}{\operatorname{im}\big(d_r: E_r^{p-r,\, q+r-1} \to E_r^{p,q}\big)}
$$

当 $r$ 足够大时 $d_r = 0$，页稳定在 $E_\infty$。**收敛** $E_2^{p,q} \Rightarrow H^{p+q}$ 的意思是：目标对象 $H^n$ 有一条滤过，其逐级商正是 $E_\infty^{p,\, n-p}$（把不同的 $p$ 加起来就是 $H^n$）。<span class="marginnote">记忆口诀：<strong>「$E_2$ 页是入口，微分是打磨，$E_\infty$ 是分次的答案。」</strong> 实际的 K 理论问题里，$E_2$ 往往是熟悉的上同调或 K 群，微分则是算术对象（tame 符号、Bockstein 同态）——困难集中在微分。</span>

## 2 余维滤过与 Brown–Gersten 谱序列

设 $X$ 是（拟紧、分离的）正则概形，$X^{(p)}$ 记余维为 $p$ 的点的集合，$k(x)$ 记点 $x$ 的**剩余域（residue field）**。把 $K_n(X)$ 按「支撑在余维 $\ge p$」滤过，Q-构造的 Localization 定理逐层给出长正合列，拼成：

> **Brown–Gersten–Quillen 谱序列（余维谱序列）。**
> $$
> E_1^{p,q} = \bigoplus_{x \in X^{(p)}} K_{-p-q}\big(k(x)\big) \quad\Rightarrow\quad K_{-p-q}(X)
> $$

**第一行读法**：$E_1^{p,q}$ 把 $X$ 上**余维恰好 $p$ 的所有点**的剩余域 K 群全部加起来。$q$ 是「偏移量」，总次数 $-p-q$ 才是真正的 K 群指标。**整体 = 各余维点的贡献之和**——这是「局部化」的终极形式：每个点提供一个「当地的 K 群」。

**第一微分**：$d_1: E_1^{p,q} \to E_1^{p+1,q}$ 是**余维 $p$ 的点**到**它闭包里的余维 $p+1$ 点**的算术映射——在 $p=1$ 的情形，它正是第 3、4 篇的 **tame 符号**（对离散赋值环取「主符号」）。**谱序列把「符号」升格成了「微分」**。

**Gersten 猜想**：对正则局部环 $X = \operatorname{Spec} R$，人们猜测这条谱序列在 $E_2$ 页就**退化**（所有高阶微分消失）——这叫 **Gersten 猜想**，已被 Kato、Bloch 等在多重情形证明，一般情形至今开放。<span class="marginnote">Gersten 猜想之所以迷人，是因为它说「正则环的 K 群可以仅由剩余域 K 群递归算出」——这是一个「从点上的算术到整体的算术」的纲领。它的动机上同调版本就是 Bloch 的高阶 Chow 群理论。</span>

## 3 Atiyah–Hirzebruch 谱序列：从奇异上同调算拓扑 K

对拓扑 K 理论（第 12 篇的主角），Atiyah–Hirzebruch 给出另一条谱序列：

$$
E_2^{p,q} = H^p\big(X;\ K^q(\mathrm{pt})\big) \ \Rightarrow\ K^{p+q}(X)
$$

对复拓扑 K 理论，$K^q(\mathrm{pt}) = \mathbb{Z}$（$q$ 偶）而 $= 0$（$q$ 奇），于是

$$
E_2^{p,q} = \begin{cases} H^p(X;\ \mathbb{Z}) & q \text{ 偶} \\ 0 & q \text{ 奇} \end{cases}
$$

**读法**：拓扑 K 群从奇异上同调 $H^p(X;\mathbb{Z})$「复制」出来——偶数次上同调原样进入 K 理论，奇数次则不贡献。$S^2$ 的情形立即可算：$H^0(S^2) = H^2(S^2) = \mathbb{Z}$，故 $K^0(S^2) = \mathbb{Z} \oplus \mathbb{Z}$（Hopf 线丛带来的那个 $\mathbb{Z}$），$K^1(S^2) = 0$——与第 5 篇 Swan 定理的结论吻合。<span class="marginnote">AHSS 是「广义上同调理论」的共同骨架：对任何广义上同调 $E^*$ 都有 $E_2^{p,q} = H^p(X; E^q(\mathrm{pt})) \Rightarrow E^{p+q}(X)$。K 理论只是它的一个实例，把系数环 $E^*(\mathrm{pt})$ 换成 K 群的「点谱」$\mathbb{Z}[b, b^{-1}]$。</span>

## 4 公式解析：E₁ 页、微分与收敛

把余维谱序列的四个构件逐个拧紧：

$$
E_1^{p,q} = \bigoplus_{x \in X^{(p)}} K_{-p-q}\big(k(x)\big) \quad\Rightarrow\quad K_{-p-q}(X)
$$

**第一步，看指标 $-p-q$**：$q$ 的范围是 $[-p, 0]$（因为 $K_{-p-q}$ 的指标非负要求 $-p - q \ge 0$）。所以 $E_1$ 页支持在「三角形」$0 \le -p-q$ 里——这不是巧合，而是「余维滤过」天然给出的**有限支撑**，正是它让谱序列能收敛。

**第二步，看 $d_1$ 的方向**：$d_1: E_1^{p,q} \to E_1^{p+1,q}$ 把「余维 $p$」推进到「余维 $p+1$」，即**深入闭包**。每个余维 $p$ 点 $x$ 贡献的 $K_{-p-q}(k(x))$，被 $d_1$ 拆散、求和、映射到它闭包里的低维点——「高余维点会收到来自低余维点的边界」。

**第三步，看下一页**：$E_2^{p,q} = \ker d_1 / \operatorname{im} d_1$ 是「余维 $p$ 层的上同调」。对 $X = \operatorname{Spec} F$（域），谱序列退化为 $E_1^{0,0} = K_0(F) = \mathbb{Z}$ 一个点——**域的情形平淡，正说明谱序列为「整体」而设**。

**第四步，看收敛**：$E_\infty$ 的 $K_{-p-q}(X)$ 的滤过商是「精确到余维 $p$ 的贡献」。若 Gersten 猜想成立，$E_2 = E_\infty$，答案直接是「各剩余域 K 群的某种上同调」——**从点到整体，一路由微分清算**。

## 5 应用：投射丛公式与数域的 K 群

**投射空间 $\mathbb{P}^1_k$**：$X^{(0)}$ 只有一个点（一般点），$X^{(1)}$ 是无穷多个闭点。Brown–Gersten 谱序列给出经典的**投射丛公式**：

$$
K_n(\mathbb{P}^1_k) \cong K_n(k) \oplus K_{n-1}(k)
$$

这正是「$\mathbb{P}^1$ 上每个向量丛 = 平凡丛在直和意义下的扭转」的 K 理论翻版——与第 5 篇 $S^2$ 的 $\mathbb{Z} \oplus \mathbb{Z}$ 遥相呼应（$k = \mathbb{C}$ 时 $\mathbb{P}^1_\mathbb{C} = S^2$）。<span class="marginnote">投射丛公式 $K_n(\mathbb{P}^r_k) \cong \bigoplus_{i=0}^{r} K_{n-i}(k)$ 是代数 K 理论的「切割原理」：把射影空间按维数切开，每一刀贡献一个平移的 K 群。拓扑里的对应物是第 3 节的 AHSS 在 $X = \mathbb{C}P^r$ 上的退化。</span>

**数域的 K 群**：设 $F$ 是数域、$\mathfrak{o}_F$ 是整数环。由局部化谱序列与 Borel 调节子得到**秩公式**：

$$
\operatorname{rank}_{\mathbb{Z}} K_{2n-1}(\mathfrak{o}_F) = \begin{cases} r_1 + r_2, & n \text{ 偶} \\ r_2, & n \text{ 奇} \end{cases}, \qquad
K_{2n}(\mathfrak{o}_F) \text{ 有限}
$$

$r_1, r_2$ 是 $F$ 的实嵌入与复嵌入对数。**K 群的秩只依赖嵌入计数，而挠部分依赖根的次数与 L 函数的值**——这正是第 11 篇「与代数数论的联系」的计算支柱，Borel 调节子也在那里再登场。高阶 K 群由此从「抽象同伦群」变成「可算的算术量」。

### 术语速查表：谱序列与 K 理论

| 术语 | 含义 |
| --- | --- |
| 页 $E_r$ | 谱序列的第 $r$ 轮，双分次对象 |
| 微分 $d_r$ | $E_r^{p,q} \to E_r^{p+r,\,q-r+1}$，满足 $d_r^2=0$ |
| 收敛 $E_2 \Rightarrow H$ | $H$ 的滤过商为 $E_\infty$ |
| 退化 | 某页之后 $d_r = 0$，$E_\infty$ 提前现身 |
| 余维滤过 | 按「支撑在余维 $\ge p$」滤过 K 群 |
| $X^{(p)}$ | $X$ 上余维恰为 $p$ 的点集 |
| Gersten 猜想 | 正则局部环上余维谱序列在 $E_2$ 退化 |
| AHSS | $E_2^{p,q} = H^p(X;K^q(\mathrm{pt})) \Rightarrow K^{p+q}(X)$ |

**辨析｜易错点：** 余维谱序列的 $E_1$ 页指标是「$K_{-p-q}$」——总次数带负号，因为它跟随的是**上同调式的余维滤过**；而 AHSS 的 $E_2$ 页总次数 $p+q$ 直接就是 K 群指标。两套索引约定方向相反，混用必乱。

## 6 小结

- **谱序列**：一列 $(E_r, d_r)$，$E_{r+1}$ 是 $E_r$ 的同调；收敛时 $E_\infty$ 给出目标的分次商。
- **Brown–Gersten 谱序列**：$E_1^{p,q} = \bigoplus_{x \in X^{(p)}} K_{-p-q}(k(x)) \Rightarrow K_{-p-q}(X)$，从剩余域 K 群拼出整体 K 群。
- **第一微分** $d_1$：余维推进的算术映射（$p=1$ 时即 tame 符号）；**Gersten 猜想**断言正则局部环上 $E_2 = E_\infty$。
- **Atiyah–Hirzebruch 谱序列**：$E_2^{p,q} = H^p(X; K^q(\mathrm{pt})) \Rightarrow K^{p+q}(X)$，偶数次上同调进入 K 理论。
- **应用**：投射丛公式 $K_n(\mathbb{P}^r_k) \cong \bigoplus K_{n-i}(k)$；数域 $K_{2n-1}(\mathfrak{o}_F)$ 的秩由嵌入计数 $r_1, r_2$ 决定。
- **易错**：索引约定（余维谱 $E_1$ 页 vs AHSS $E_2$ 页）、微分双次数 $(r, 1-r)$