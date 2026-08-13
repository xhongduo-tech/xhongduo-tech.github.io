---
title: BCH 码：根集、BCH 界与纠错能力
date: 2026-08-07
---

# BCH 码：根集、BCH 界与纠错能力

<div class="epigraph">
<p>发现问题的表述方式，往往比解决问题本身更重要。</p>
<footer>—— 约翰 · 冯 · 诺依曼（John von Neumann）</footer>
</div>

<div class="article-byline">
<p>第二级 · 编码理论（纠错编码） ｜ van Lint 第6章；Roth 第8章；MacWilliams & Sloane 第9章 ｜ 2026-08-07</p>
</div>

## 为什么从 BCH 码开始

循环码给了我们构造工具：挑 $x^n - 1$ 的因式相乘，就得到一个码。但此前我们挑因式靠的是「运气」——Hamming 码碰巧好，别的呢？我们需要一条**设计规则**：给定要纠的错数 $t$，直接造出一个最小距离 $\ge 2t+1$ 的码。

BCH 码（Bose–Chaudhuri–Hocquenghem，由三人于 1959–60 独立发现）给出了这条规则：**让生成多项式以一段连续的根为根**。这个「连续根串」极其强大，它保证最小距离至少等于根串的长度加一——这就是 **BCH 界**。<span class="marginnote">BCH 码是编码理论里第一批「指哪打哪」的码：说要纠 2 个错，就真造出 $d \ge 5$ 的码；纠 3 个错，就造出 $d \ge 7$ 的码。设计距离（designed distance）与真实距离往往相等，只偶尔差一点。工程上 CD、闪存、卫星通信里的主力纠错码，几乎全是 BCH 族。</span>

## 1 从根到码：BCH 码的定义

先准备好场地。设 $q$ 元域 $\mathbb{F}_q$，取正整数 $m$ 使 $n \mid q^m - 1$，于是 $\mathbb{F}_{q^m}$ 里有 $n$ 次本原单位根。设 $\alpha$ 是其中一个 $n$ 次本原单位根（$\alpha^n = 1$，$\alpha^i \neq 1$ for $0<i<n$）。<span class="marginnote">当 $n = q^m - 1$ 时 $\alpha$ 是 $\mathbb{F}_{q^m}$ 的本原元，此时称码为<strong>本原 BCH 码</strong>；$n$ 更小时是「缩短/非本原」情形。二元 $[2^m-1, \cdot, \cdot]$ BCH 码是工程上最常见的本原情形。</span>

**BCH 码的定义：** 设 $\delta$ 是设计距离。取 $b$ 为某个整数，令生成多项式 $g(x)$ 为「以 $\alpha^b, \alpha^{b+1}, \dots, \alpha^{b+\delta-2}$ 为根」的最小多项式的**最小公倍式**：

$$g(x) = \mathrm{lcm}\{M^{(b)}(x), M^{(b+1)}(x), \dots, M^{(b+\delta-2)}(x)\}$$

其中 $M^{(i)}(x)$ 是 $\alpha^i$ 在 $\mathbb{F}_q$ 上的最小多项式。由 $g$ 生成的循环码就是设计距离为 $\delta$ 的 BCH 码。<span class="marginnote"><strong>狭义 BCH 码（narrow-sense）</strong> 取 $b = 1$（根从 $\alpha$ 开始）；$b$ 任意的叫广义。二元码常用狭义本原 BCH。注意 $g$ 的根集合要「循环共轭封闭」——$\alpha^i$ 的共轭 $\alpha^{iq}, \alpha^{iq^2}, \dots$ 都自动成为 $g$ 的根，所以连续根串在共轭作用下会「外溢」，$g$ 的次数通常比 $\delta - 1$ 大。</span>

## 2 BCH 界：连续根串给出距离下界

**BCH 界（BCH bound）：** 设计距离为 $\delta$ 的 BCH 码，其最小距离至少为 $\delta$：

$$d_{\min} \ge \delta$$

即码能保证纠正至少 $t = \lfloor (\delta-1)/2 \rfloor$ 个错误。<span class="marginnote">这是「设计距离」一词的由来：$\delta$ 是设计时承诺的距离下界，真实距离可能更大。例：$[15,7]$ BCH 码设计距离 5，真实距离也是 5；$[15,5]$ BCH 码设计距离 7，真实距离 7；而 $[23,12]$ Golay 码的 $d=7$ 正是由 BCH 界与别的手段共同确定的。</span>

BCH 界的证明见下一节公式解析，它只用了两个工具：**多项式的根**与 **Vandermonde 矩阵的非奇异性**。这条界是 BCH 码全部价值的源泉——它把「要纠 $t$ 个错」翻译成「根串取 $2t$ 个连续幂」，构造规则由此完全机械化。

## 3 公式解析：BCH 界为什么成立

证明「没有重量小于 $\delta$ 的非零码字」。反设存在非零码字 $c(x)$ 重量 $w \lt  \delta$，即

$$c(x) = x^{i_1} + x^{i_2} + \cdots + x^{i_w}$$

（二元情形；$q$ 元时系数 $\neq 0$ 即可）。

- **第一步，根代入**：因为 $\alpha^b, \dots, \alpha^{b+\delta-2}$ 都是 $g(x)$ 的根，而 $c(x)$ 是 $g$ 的倍式，所以 $c(\alpha^j) = 0$ 对 $j = b, b+1, \dots, b+\delta-2$ 全部成立。展开得 $w$ 个方程：

$$\sum_{l=1}^{w} \alpha^{j i_l} = 0, \qquad j = b, b+1, \dots, b+\delta-2$$

- **第二步，写成矩阵**：把 $j$ 取 $b, \dots, b+w-1$ 这 $w$ 个方程（因为 $w \le \delta-1$，这些 $j$ 都在根串内），提出公共因子 $\alpha^{b i_l}$，得到 $w \times w$ 线性系统：

$$\begin{pmatrix} (\alpha^{i_1})^b & \cdots & (\alpha^{i_w})^b \\ (\alpha^{i_1})^{b+1} & \cdots & (\alpha^{i_w})^{b+1} \\ \vdots & & \vdots \\ (\alpha^{i_1})^{b+w-1} & \cdots & (\alpha^{i_w})^{b+w-1} \end{pmatrix} \begin{pmatrix} \alpha^{b i_1} c_{i_1} \\ \vdots \\ \alpha^{b i_w} c_{i_w} \end{pmatrix} = \boldsymbol{0}$$

- **第三步，Vandermonde 非奇异**：系数矩阵是 $\alpha^{i_1}, \dots, \alpha^{i_w}$ 上的 Vandermonde 矩阵。$i_l$ 互不相同（位置不同）且 $0 \le i_l \le n-1$，所以 $\alpha^{i_l}$ 两两不同；Vandermonde 行列式非零，矩阵可逆。于是唯一解是零向量——但每个分量 $\alpha^{b i_l} c_{i_l} \neq 0$（$c_{i_l} \neq 0$），矛盾！

**直觉：** 连续 $w$ 个「根条件」给出一组互相独立的方程，逼得码字必须消失——**根串越长，能容纳的非零位置越少**。这就是「连续根串 = 距离」的代数本质。回忆第 3 篇的「$d$ = 校验矩阵最小相关列数」：BCH 界正是从「根」这一侧重新证明了同一个道理。

## 4 例子：从 Hamming 码到 $[15, 7, 5]$ BCH 码

**Hamming 码 = 设计距离 3 的本原 BCH 码。** 二元 $[2^m-1, 2^m-1-m, 3]$ Hamming 码正是取 $g(x)$ = $\alpha$ 的最小多项式（本原多项式）的狭义本原 BCH 码：根串只有 $\{\alpha\}$ 一个（$\delta = 3$），BCH 界给出 $d \ge 3$，而它恰取等号。

**$[15, 7, 5]$ BCH 码**（$m = 4$，$n = 15$，纠 2 错）：设 $\alpha$ 是 $\mathbb{F}_{16}$ 的本原元（取本原多项式 $x^4 + x + 1$）。设计距离 5，根串取 $\{\alpha, \alpha^2, \alpha^3, \alpha^4\}$。

| 根 | 最小多项式（二元） |
| --- | --- |
| $\alpha$ | $x^4 + x + 1$ |
| $\alpha^3$ | $x^4 + x^3 + x^2 + x + 1$ |
| $\alpha^2, \alpha^4$ | $\alpha$ 的共轭，已含在 $x^4+x+1$ 里 |

$g(x) = (x^4+x+1)(x^4+x^3+x^2+x+1) = x^8 + x^7 + x^6 + x^4 + 1$，次数 8，故 $k = 15 - 8 = 7$，得 $[15, 7, 5]$ 码。<span class="marginnote">注意根串里 $\alpha^2, \alpha^4$ 没贡献新因式——它们与 $\alpha$ 共轭，最小多项式相同。共轭「外溢」让实际 $g$ 的次数（8）大于「根串长度减一」（4），这是 $k$ 偏小的代价，也解释了「$d \ge \delta$」而非等号：$[15,7]$ 的真实距离恰好 5，但共轭外溢有时会意外抬高距离。</span>

## 5 BCH 码的距离与界

BCH 码的真实最小距离通常比设计距离大或相等。已知的事实：

二元本原 BCH 码的真实距离几乎总是等于设计距离（除个别例外）；
**Hartmann-Tzeng 界**与 **Roqué 界**是 BCH 界的推广：当根串有「周期性的空隙」时也能给出更强的下界。<span class="marginnote">工程上不需要每次都算真实距离：设计距离 $\delta$ 已经是「保证值」，按 $\delta$ 设计的码天然满足纠 $t = \lfloor(\delta-1)/2\rfloor$ 错的承诺。这也是 BCH 码敢在规格书里写「纠正 2 个错误」的原因。</span>

**辨析｜易错点：** BCH 界保证的是「最小距离 $\ge \delta$」，不是「恰好 $\delta$」。说「$[15,7]$ 码能纠 2 个错」用的是下界；它可能实际能纠更多（比如意外变成 $d=6$，那就能纠 3 个错的部分模式）。规格书按设计距离写，是最保守也最诚实的承诺。

## 6 BCH 码参数速览

本原二元 BCH 码的参数由「取哪些共轭类做根」完全决定，下面是 $m = 4, 5$ 时的常见选择（设计距离 = 真实距离）：

| $n = 2^m-1$ | $k$ | $d$（设计/真实） | 纠错 $t$ | 根串 |
| --- | --- | --- | --- | --- |
| 15 | 11 | 3 | 1 | $\{\alpha\}$ |
| 15 | 7 | 5 | 2 | $\{\alpha, \alpha^2, \alpha^3, \alpha^4\}$ |
| 15 | 5 | 7 | 3 | $\{\alpha, \dots, \alpha^6\}$ |
| 31 | 26 | 3 | 1 | $\{\alpha\}$ |
| 31 | 21 | 5 | 2 | $\{\alpha, \dots, \alpha^4\}$ |
| 31 | 16 | 7 | 3 | $\{\alpha, \dots, \alpha^6\}$ |
| 31 | 11 | 11 | 5 | $\{\alpha, \dots, \alpha^{10}\}$ |
| 31 | 6 | 15 | 7 | $\{\alpha, \dots, \alpha^{14}\}$ |

读这张表要抓住两点。第一，**$k$ 随 $t$ 增加而减少的速度比「$\delta - 1$」快**：从 $t=1$ 到 $t=2$ 只损失 4 位信息（$26 \to 21$），到 $t=7$ 只剩 6 位——纠错能力用信息位「买」来，越贵越难。第二，**根串取偶数是浪费**：$\alpha^4$ 是 $\alpha^2$ 的共轭，加进根串不增维数损失，所以「设计距离 $2t+2$」与「$2t+1$」给出同一个码——这也是为什么 $d$ 总取奇数。<span class="marginnote">工程查表文化：芯片里的 BCH 编码器常把 $(n, k, t)$ 当规格参数直接查 ROM 表，而不是现场算 $g(x)$。这张表就是那种规格表的雏形。</span>

这些参数不是巧合——它们全部可由「BCH 界 + 共轭结构」机械推出，这正是 BCH 码「指哪打哪」的实证。

## 7 BCH 码与 Reed-Solomon 码的关系

BCH 码的设计语言——「根串」——在下一节会以另一种形式复活：**Reed-Solomon 码就是「$q$ 元本原 BCH 码」**，但它把符号域直接取成 $\mathbb{F}_{q^m}$，让每个根的最小多项式都是线性的，从而把「共轭外溢」消掉。结果：RS 码达到 Singleton 界（MDS），是 BCH 码在「符号数 = 域大小」时的最紧形态。<span class="marginnote">一句话记忆：<strong>BCH 码在 $\mathbb{F}_q$ 上、根在 $\mathbb{F}_{q^m}$ 里；RS 码在 $\mathbb{F}_{q^m}$ 上、根也在 $\mathbb{F}_{q^m}$ 里</strong>。前者适合纠正「比特突发错误」的二元场景，后者适合纠正「符号突发错误」的字节场景——CD、DVD、二维码里全是 RS 码。</span>

## 8 小结

- **BCH 码**：生成多项式以连续根串 $\alpha^b, \dots, \alpha^{b+\delta-2}$ 为根（取各最小多项式的 lcm），设计距离 $\delta$。
- **BCH 界**：$d_{\min} \ge \delta$，保证纠 $t = \lfloor(\delta-1)/2\rfloor$ 个错。
- 证明靠 Vandermonde 矩阵非奇异：连续根条件逼得重量 $\lt  \delta$ 的码字只能为零。
- Hamming 码 = 设计距离 3 的本原 BCH 码；$[15,7,5]$ 是设计距离 5 的典型。
- 共轭外溢让 $g$ 次数偏大、$k$ 偏小；真实距离可能 $\ge$ 设计距离。
- **BCH 在 $\mathbb{F}_q$、根在 $\mathbb{F}_{q^m}$；RS 把符号域提成 $\mathbb{F}_{q^m}$，达到 MDS**——下一节的主角。
- 参数速览：$n=31$ 时 $t=1,2,3,5,7$ 对应 $k=26,21,16,11,6$；根串取偶数幂不增距离、只增冗余。
- BCH 界是「下界」承诺：设计距离 $\delta$ 保证 $d \ge \delta$