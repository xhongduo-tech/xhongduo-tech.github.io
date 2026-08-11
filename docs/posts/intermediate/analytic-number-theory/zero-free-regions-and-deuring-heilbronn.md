---
title: 零区域与 Deuring-Heilbronn 现象
date: 2026-08-11
---

# 零区域与 Deuring-Heilbronn 现象

<div class="epigraph">
<p>数学是无穷的科学。</p>
<footer>—— 赫尔曼 · 外尔（Hermann Weyl）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 解析数论 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么误差项盯住零点不放

第四篇我们看到：素数定理等价于「$\zeta$ 在 $\sigma = 1$ 上无零点」，且误差项的大小取决于「零点离 $\sigma=1$ 能有多近」。零区域理论把这个「多近」变成精确的估计；而对 $L(s,\chi)$，同样的故事会更险——那里可能出现一个极其接近 $s=1$ 的实零点，就是臭名昭著的 **Siegel 零点**。本章的戏剧性在于一个悖论般的现象：**一个坏零点（若存在）反而会「赶走」其他零点**，这就是 Deuring–Heilbronn 现象。

## 1 $\zeta$ 的零区域：一条不断变薄的带子

**零区域（zero-free region）**：已知 $\zeta(\beta + it) \ne 0$ 的区域 $\beta \ge 1 - \eta(|t|)$，其中 $\eta$ 趋于 0。经典结果（de la Vallée Poussin）给出

$$
\beta \ge 1 - \frac{c}{\log(|t| + 2)}, \qquad |t| \ge 2
$$

而 1958 年 Vinogradov–Korobov 用指数和方法把带子削薄成

$$
\beta \ge 1 - \frac{c}{(\log |t|)^{2/3}(\log\log |t|)^{1/3}}
$$

**重点：零区域的形状由 $\log |t|$ 决定，而不是 $|t|$ 的幂次**——这是解析数论里「几乎零区域的量级」的标配：垂直方向越走越高，容许的零区域越缩越小，但缩得非常慢。<span class="marginnote">为什么形状是 $\log$？因为 $\zeta$ 的 Euler 乘积在高 $t$ 处几乎沿 $\sigma = 1$ 的垂线震荡，能用到的初等下界（比如 $\zeta$ 的三乘积恒等式）只给出 $\log |t|$ 量级的增长。想更深就得靠更长的乘积与指数和估计，这正是 Vinogradov–Korobov 的思路。</span>

零区域直接翻译成误差项：把 Perron 积分的积分线从 $\sigma=1$ 挪进这条带子，得到

$$
\psi(x) = x + O\!\left(x\, e^{-c\sqrt{\log x}}\right)
$$

经典带子给出指数 $e^{-c\sqrt{\log x}}$；换成 Vinogradov–Korobov 带子则得到 $e^{-c(\log x)^{3/5}(\log\log x)^{-1/5}}$——更薄、误差更好。

## 2 公式解析：零区域如何喂给误差项

把「零区域 → 误差项」的传递链拆成三步：

$$
\psi(x) = \frac{1}{2\pi i}\int_{\sigma=c} \left(-\frac{\zeta'}{\zeta}(s)\right)\frac{x^s}{s}\,ds \;\longrightarrow\;
\psi(x) = x + O\!\left(x e^{-c\sqrt{\log x}}\right)
$$

- **第一步，Perron 公式**：$\psi(x)$ 是 $-\zeta'/\zeta$ 的 Mellin 逆变换在 $x$ 处的值。积分线最初立在 $\sigma = 2$。
- **第二步，向左挪线**：把积分线移到 $\sigma = 1 - \delta$。由留数定理，$s=1$ 的极点贡献主项 $x$；而经过每一个 $\zeta$ 的零点时，$-\zeta'/\zeta$ 有极点，必须绕开。若零区域是 $\sigma \ge 1 - \eta(t)$，就可以把线推到 $\sigma = 1 - \eta(T)$（$T$ 是积分被截断的高度）。
- **第三步，截断误差**：挪线后竖直段上的被积函数带 $x^{1-\eta}$，截断 $|t| \le T$ 带来误差 $x/T \log x$ 量级。取平衡 $\eta(T) \approx 1/\sqrt{\log x}$（即 $T \approx e^{\sqrt{\log x}}$），两处误差都是 $x e^{-c\sqrt{\log x}}$。

**直觉**：每把零区域向前推进一点，积分线就能左移一点，$x$ 的指数就被压低一点——**零点与误差项是同一枚硬币的两面**。<span class="marginnote">若黎曼假设成立（零区域直接是整个半平面 $\sigma \ge 1/2$），同样的推导会给出 $\psi(x) = x + O(\sqrt{x}\log^2 x)$。所以「RH ⟹ 最好误差」不是玄学，只是把上面第三步的 $\eta$ 换成常数 $1/2$。</span>

## 3 $L$-函数与 Siegel 零点：唯一的例外

对 $L(s,\chi)$，零区域必须把 $q$ 写进去：当 $\chi$ 为非原特征模 $q$ 时，除了一个**可能的例外**，$L(s,\chi)$ 在 $\beta \ge 1 - c/\log(q(|t|+2))$ 内无零点。这个例外只对**实的非主特征**可能出现，且是一个**实、简单**的零点，落在线段 $1 - c/\log q \le \beta < 1$ 上，叫作 **Siegel 零点**（也叫 Landau–Siegel 零点）。

**关键事实：Siegel 零点是否真的存在，至今无人知道**——所有证明都只能处理「若存在」的情形。1935 年 Siegel 证明了：

$$
L(1, \chi) \gg_\varepsilon \frac{1}{q^{\varepsilon}} \qquad \text{（对任意 } \varepsilon > 0\text{）}
$$

也就是说 $L(1,\chi)$ 不会太小（等价地，Siegel 零点不会离 1 太近）。但这里的常数 $\gg_\varepsilon$ 是**无效的（ineffective）**：它依赖一个「若某个反例存在则……否则……」的二难论证，没有人能把它算出来。<span class="marginnote"><strong>辨析｜易错点：</strong> 无效常数不是「未知的大常数」，而是<strong>原则上算不出</strong>：Siegel 证明里两个分支各含一个无法控制的量。这与密码学里「可证安全的常数」形成鲜明对比——也是为什么数论家一边用它证明定理、一边拼命想绕过它的原因。大筛法（第十一篇）的 Bombieri–Vinogradov 定理正是「无例外地」绕过 Siegel 零点的模范。</span>

## 4 Deuring–Heilbronn 现象：坏零点赶走好零点

到这里出现最反直觉的一环。直觉说「存在一个零点会让事情更糟」，但解析数论里却有一条定理**反过来用**：

**Deuring–Heilbronn 现象**：若存在一个模 $q_1$ 的 Siegel 零点 $\beta_1$，则 $\zeta$ 函数（以及所有远离该模数的 $L$ 函数）在

$$
\beta \ge 1 - c\, \frac{\log q_1}{\log(q_1(|t| + 2))}
$$

内无零点——比常规零区域（$1 - c/\log(|t|+2)$）**宽得多**。

**重点：一个坏零点扮演了「排斥子」——它把临界带里别的零点都推到更右边**。<span class="marginnote">直觉类比：若 $L(s,\chi_1)$ 在 $\beta_1 \approx 1$ 有零点，那么在 $\beta_1$ 附近 $L'(\chi_1)$ 会很大；通过类数公式与某种「互斥」关系，这迫使 $\zeta$ 和其余 $L$ 函数在 $\sigma > \beta_1$ 保持正则。分析上这是「留数很大 ⟹ 更小区域内不能再有零点」的传播机制。它是解析数论里少有的「假设更坏，结论更好」的定理，证明用到了 Deuring 1930 年与 Heilbronn 1934 年关于类数的经典工作。</span>

这条现象的真正价值在于**构造矛盾**：在许多论证里，只需证明「若 Siegel 零点存在则能推出更强的均匀性，从而推出矛盾」，就能无条件地得到均匀性结论。例如素数在等差数列中的均匀分布：**对任何固定 $A$，要么对所有 $q \le x^A$ 成立 $\psi(x;q,a) \sim x/\varphi(q)$，要么某个 Siegel 零点在起作用**——D–H 现象正是那个「分叉证明」的枢纽。

## 5 小结

- **零区域**：$\zeta$ 在 $\beta \ge 1 - c/\log(|t|+2)$ 无零点；Vinogradov–Korobov 改进到 $(\log|t|)^{2/3}$ 形状，带来更优的误差项 $x e^{-c(\log x)^{3/5}}$ 量级。
- 误差项与零区域**一一对应**：零点离 $\sigma=1$ 越远（零区域越宽），Perron 积分挪线越深，误差越小。
- **Siegel 零点**：唯一可能的例外——实的、简单的、接近 $s=1$ 的零点，只出现在实特征 $L$ 函数中，**是否存在至今未知**。
- **Siegel 定理**：$L(1,\chi) \gg_\varepsilon q^{-\varepsilon}$，但常数**无效**、原则上不可计算。
- **Deuring–Heilbronn 现象**：坏零点会排斥其他零点——「假设更坏、结论更好」，是解析数论中最奇特的工具之一。

下一节，我们把目光从「零点离 1 多近」转向「零点分布的整体图景」——**黎曼假设与零点计数**。
