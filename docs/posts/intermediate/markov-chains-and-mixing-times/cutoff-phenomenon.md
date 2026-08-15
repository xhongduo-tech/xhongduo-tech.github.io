---
title: cutoff 现象
date: 2026-08-07
---

# cutoff 现象

<div class="epigraph">
<p>混沌并非突然降临，但有时它几乎如此。</p>
<footer>—— 雅克 · 阿达马（Jacques Hadamard）</footer>
</div>

<div class="article-byline">
<p>第二级 · 马尔可夫链与混合时间 ｜ Levin, Peres &amp; Wilmer《Markov Chains and Mixing Times》Ch. 18 ｜ 2026-08-07</p>
</div>

## 为什么从 cutoff 现象开始

洗牌一节我们看到一个引人入胜的现象：GSR 随机洗牌从第 6 次到第 7 次，全变差距离从 $0.61$ 陡降到 $0.33$——分布似乎**在一个很窄的时间窗口内「突然」变均匀**。这并非个例，而是一大类链的普遍特征，称作 **cutoff 现象（cutoff phenomenon）**：混合时间之前几乎完全不混，混合时间之后几乎完全均匀，中间隔着一条极窄的「悬崖」。这个现象 1990 年左右由 Aldous 与 Diaconis 提出并命名，如今是混合时间理论最活跃的研究方向之一。<span class="marginnote">cutoff 的重要应用含义：它意味着「跑到 $0.9 t_{\mathrm{mix}}$ 步几乎没有任何样本价值，跑到 $1.1 t_{\mathrm{mix}}$ 步就已经完全混匀」——实践上，知道 $t_{\mathrm{mix}}$ 是否发生 cutoff 会彻底改变「跑多少步」的策略。</span>

**名字的由来**：「cutoff」字面即「切断」——在临界时间处，收敛过程像是被一把刀切成「未混」与「已混」两段，中间没有过渡。这个词正描述了模拟中反复观察到的「突发混合」：研究者跑了几千步毫无动静，再多跑几十步突然就均匀了。

## 1 cutoff 的精确定义

考虑一族链（如「$n$ 张牌的洗牌」），混合时间 $t_{\mathrm{mix}}^{(n)}$ 随 $n \to \infty$。若对任意 $0 \lt  \varepsilon \lt  1$ 有

$$
\lim_{n \to \infty} \frac{t_{\mathrm{mix}}^{(n)}(\varepsilon)}{t_{\mathrm{mix}}^{(n)}(1-\varepsilon)} = 1
$$

则称这族链呈现 **cutoff**。<span class="marginnote">这个定义的要点：$t_{\mathrm{mix}}(\varepsilon)$ 随 $\varepsilon$ 变化在「非 cutoff」情形下差异巨大（比如 $t_{\mathrm{mix}}(\varepsilon) \sim \log(1/\varepsilon) \cdot t_{\mathrm{rel}}$），而 cutoff 情形下所有 $\varepsilon$ 都给出相同的量级——悬崖太陡，不同置信水平的混合时间几乎重合。</span>

等价地，用**窗口（window）**表述：cutoff 意味着存在 $t_n$ 与 $w_n = o(t_n)$（窗口远小于混合时间），使得

$$
\lim_{n\to\infty} \max_{\mu_0} \lVert \mu_0 P^{(n)\,t_n + c w_n} - \pi_n \rVert_{\mathrm{TV}} = \begin{cases} 1, & c \to -\infty \\ 0, & c \to +\infty \end{cases}
$$

即：**在 $t_n$ 之前距离趋于 1（完全没混），在 $t_n$ 之后距离趋于 0（完全混匀），窗口 $w_n$ 与 $t_n$ 相比可忽略**。<span class="marginnote">形象地：画出「距离-时间」曲线，非 cutoff 是缓慢的斜坡，cutoff 是垂直的悬崖。判定 cutoff 就是判定这条曲线是否在极限意义下变成阶跃函数。</span>

## 2 公式解析：为什么「几乎相同」意味着「悬崖」

cutoff 定义看似平淡，实则非常苛刻。逐项拆解：

$$
\frac{t_{\mathrm{mix}}(\varepsilon)}{t_{\mathrm{mix}}(1-\varepsilon)} \to 1
$$

- **第一步，读出两个时间**：$t_{\mathrm{mix}}(\varepsilon)$ 是压到 $\varepsilon$ 的时间（$\varepsilon$ 小），$t_{\mathrm{mix}}(1-\varepsilon)$ 是压到 $1-\varepsilon$ 的时间（几乎还没混，因为距离要小于 $1-\varepsilon$ 只需要很弱的要求）。通常 $t_{\mathrm{mix}}(\varepsilon)$ 远大于 $t_{\mathrm{mix}}(1-\varepsilon)$。
- **第二步，非 cutoff 的典型行为**：谱方法给出 $t_{\mathrm{mix}}(\varepsilon) \approx t_{\mathrm{mix}}(1-\varepsilon) + t_{\mathrm{rel}}\log\frac{1-\varepsilon}{\varepsilon}$，比值多出 $\log(1/\varepsilon)$ 的因子——**不趋于 1**，说明收敛是「渐进式」的。
- **第三步，cutoff 的语义**：比值趋于 1 意味着从「几乎没混」到「几乎混匀」之间几乎没有缓冲时间——**中间态几乎不存在**。
- **第四步，直觉来源**：如果距离函数 $d(t)$ 在极限下是阶跃函数，那么「第一次降到 $\varepsilon$ 以下」与「第一次降到 $1-\varepsilon$ 以下」会在同一点附近，比值趋于 1。cutoff 就是「距离曲线变成阶跃」的定量刻画。

## 3 谁有 cutoff，谁没有

**有 cutoff 的经典例子**：

- **GSR 随机洗牌**：$t_{\mathrm{mix}} \approx \tfrac32 \log_2 n$，存在 cutoff（Bayer–Diaconis 与 Diaconis–Shahshahani 的结论），窗口宽 $O(1)$——七次法则的严格版本。
- **随机对换**：$t_{\mathrm{mix}} \approx \tfrac12 n \log n$，存在 cutoff。
- **顶随机洗牌**：$t_{\mathrm{mix}} = n \log n$，存在 cutoff。
- **超立方体上的懒惰随机游走**：$t_{\mathrm{mix}} \approx \tfrac14 n \log n$，存在 cutoff。

**没有 cutoff 的例子**：

- **$n$-环上的简单随机游走**：$t_{\mathrm{mix}} \asymp n^2$，距离随 $\sqrt{t}/n$ 平滑衰减，无 cutoff。
- **一般扩散型链**：只要距离以 $1/\sqrt{t}$ 或缓幂衰减，就不会 cutoff。<span class="marginnote">区分两类链的粗判据：cutoff 常出现在「一步能打乱很多结构」的链（洗牌、高维随机游走），而扩散型链（环、图上的局部随机游走）平滑收敛。直觉：一步造成「全局混合」的链，混合就像「等待一个罕见事件」——等待期间几乎没进展，事件发生后瞬间完成。</span>

把上面的例子汇总成一张表：

| 链 | 混合时间量级 | cutoff? |
| --- | --- | --- |
| GSR 随机洗牌 | $\tfrac32\log_2 n$ | 有（窗口 $O(1)$） |
| 随机对换 | $\tfrac12 n\log n$ | 有 |
| 顶随机洗牌 | $n\log n$ | 有 |
| 超立方体懒惰游走 | $\tfrac14 n\log n$ | 有 |
| $n$-环简单游走 | $n^2$ | 无 |
| 平面上扩散 | $n^2$ 量级 | 无 |

**一个有趣的中庸例子**：固定 $n$ 时，$d$-维超立方体的随机游走在 $d \to \infty$ 时有 cutoff，$d$ 小时没有——cutoff 是**渐近**现象，小规模系统看不出悬崖。

## 4 判定 cutoff 的条件

研究者的目标是把「是否 cutoff」变成可判定的问题。已知的部分刻画：

**必要充分条件（对可逆链，Aldous–Diaconis / Chen–Saloff-Coste）**：设 $X_0 \sim \pi$（平稳初值），从平稳出发的链回到 $X_0$ 附近的时间记 $\tau$。cutoff 的一个必要条件是「$t_{\mathrm{rel}} = o(t_{\mathrm{mix}})$」——松弛时间远小于混合时间。<span class="marginnote">直觉：$t_{\mathrm{rel}}$ 是「局部松弛」的时间，$t_{\mathrm{mix}}$ 是「全局混合」的时间。若局部松弛本身就慢到与全局混合同阶，距离曲线必然平滑，不可能有悬崖。反之，局部快速松弛 + 全局靠「罕见长程事件」混合，是 cutoff 的配方。</span>

**Cutoff 与分离距离（separation distance）**：定义

$$
s(n) = \max_x \left(1 - \frac{P^n(x, \cdot)}{\pi}\right)
$$

separation cutoff 是更强、更易证明的版本：若 $s(n)$ 在窗口内从 1 跳到 0，则全变差 cutoff 也成立。很多链的 cutoff 证明都走「先证 separation cutoff」的路线（如顶随机洗牌）。

**一个值得记住的开放问题**：尽管 cutoff 已有许多充分条件与必要条件，对「一般可逆链何时有 cutoff」仍没有完整刻画——它被列为 Levin–Peres–Wilmer 书末开放问题清单的首要问题。已知的拼图包括瓶颈比下界、比较定理与「支撑分离」的刻画，但把它们统一成充要条件仍是未竟之业。

## 5 cutoff 与混合时间的「相变」

cutoff 最深刻的视角是把混合时间看成「相变」：把步数 $n$ 当「时间」，距离当「序参量」，cutoff 就是序参量在临界时间处的突变。<span class="marginnote">这个类比不是装饰：随机游走混合与热力学相变共享数学骨架——特征值谱、亚稳态、临界慢化。研究 cutoff 的数学工具（瓶颈比、比较定理、加权谱）与统计物理几乎同源。这也是「从极限到大模型」主线里概率论与物理交汇的缩影。</span>

**对 MCMC 实践的后果**：若链有 cutoff，那么「跑半步 $0.5 t_{\mathrm{mix}}$」与「跑 $0.9 t_{\mathrm{mix}}$」几乎没差别——都远未混匀；一旦超过 $t_{\mathrm{mix}}$，再多跑几步就完全混匀。**知道 cutoff 就能把计算预算花在刀刃上**；不知道 cutoff 而按「均匀衰减」的直觉设定步数，会严重低估或浪费。

**窗口宽度是另一个研究对象**。窗口 $w_n$ 的大小决定「悬崖」的陡峭程度：GSR 洗牌的窗口是 $O(1)$（一步之内的悬崖），而随机对换的窗口散布在 $O(n)$ 量级。「窗口比混合时间小多少」本身就是一个丰富的渐近问题，它把 cutoff 从「是/否」的判别细化成「多陡峭」的度量。

## 6 辨析｜易错点：cutoff 是渐近现象，不是小规模现象

**辨析｜易错点：** 对固定的小系统（如 $n = 10$ 的洗牌）谈「cutoff」没有严格意义。cutoff 定义中的极限 $n \to \infty$ 意味着它是**一族链的渐近行为**。小系统可能看起来「距离下降较快」，但那是有限大小的普通行为，不是 cutoff。判断一个实际应用是否「遇到 cutoff」，要看其背后的渐近族是否被证明有 cutoff，而非看单条曲线。

**辨析｜易错点：** 「cutoff」不等于「混合很快」。cutoff 是**距离曲线的形状**（悬崖），不保证混合时间小。超立方体随机游走的混合时间 $O(n\log n)$ 与 $n$-环的 $O(n^2)$ 都可能有或没有 cutoff——两个性质正交：**量级由谱/几何决定，形状由 cutoff 决定**。

## 7 小结

- **cutoff 现象**：混合时间前后距离从 1 陡降到 0，窗口远小于混合时间，极限下距离曲线变成阶跃。
- 形式定义：$t_{\mathrm{mix}}(\varepsilon)/t_{\mathrm{mix}}(1-\varepsilon) \to 1$；等价地用窗口 $w_n = o(t_n)$ 表述。
- 经典例子：**GSR 洗牌、随机对换、顶随机、超立方体**有 cutoff；**环、扩散型链**没有。
- 必要条件是 $t_{\mathrm{rel}} = o(t_{\mathrm{mix}})$