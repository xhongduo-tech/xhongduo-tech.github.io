---
title: 二次互反律
date: 2026-08-07
---

# 二次互反律

<div class="epigraph">
<p>二次互反律是数论的重锤；高斯为之呕心沥血，给出了八个不同的证明。</p>
<footer>—— 卡尔 · 弗里德里希 · 高斯（Carl Friedrich Gauss），称其为「黄金定理」</footer>
</div>

<div class="article-byline">
<p>第二级 · 数论 ｜ 潘承洞《初等数论》第三章 §4、Hardy &amp; Wright《数论导引》第九章 ｜ 2026-08-07</p>
</div>

## 为什么二次互反律被誉为「黄金定理」

勒让德符号回答「$a$ 模 $p$ 是不是平方」；但当 $a$ 本身是另一个素数 $q$ 时，出现了一个对称的谜题：

$$\left(\frac{q}{p}\right) \quad \text{与} \quad \left(\frac{p}{q}\right)$$

前者问「$q$ 模 $p$ 的身份」，后者问「$p$ 模 $q$ 的身份」。
两者之间有什么关系？
高斯发现的关系如此之简洁优美，以至于他称其为 **「黄金定理」（theorema aureum）**，一生给出八个证明，并在《算术研究》中以之作为全书的高潮：

$$\left(\frac{q}{p}\right)\left(\frac{p}{q}\right) = (-1)^{\frac{p-1}{2}\frac{q-1}{2}}$$

$p, q$ 为奇素数。
这条定理的价值不只是美学——**它把「大素数判断」转化为「小素数判断」**：想算 $\left(\frac{97}{1999}\right)$，只需算 $\left(\frac{1999 \bmod 97}{97}\right) = \left(\frac{59}{97}\right)$，递归下去，最后变成几个小素数的符号。
二次互反律是「用欧几里得算法式的递归计算勒让德符号」的总开关。<span class="marginnote">高斯 1796 年 4 月 8 日首次证明二次互反律，日记里记着「EYPHKA」（找到了）。
它把看似独立的素数两两「绑」在一起——素数之间不是孤岛，而是由同余类关系编织成的网。
希尔伯特后来把二次互反律推广到代数数论，成了「类域论」的种子。</span>

## 1 定理的陈述

**二次互反律（law of quadratic reciprocity）：** 设 $p, q$ 是互不相同的奇素数，则

$$\left(\frac{q}{p}\right) = \begin{cases} \left(\frac{p}{q}\right), & p \equiv 1 \pmod 4 \ \text{或} \ q \equiv 1 \pmod 4 \\ -\left(\frac{p}{q}\right), & p \equiv 3 \pmod 4 \ \text{且} \ q \equiv 3 \pmod 4 \end{cases}$$

换句话说，当 $p, q$ **至少一个** $1 \bmod 4$ 时，两个符号相等；**都**是 $3 \bmod 4$ 时，符号相反。
指数 $\frac{p-1}{2}\frac{q-1}{2}$ 正是这个「一个 3 还是两个 3」的编码：两个都 $3 \bmod 4$ 时它才为奇数。

验算：$p = 3, q = 5$。$\left(\frac{5}{3}\right)$：$5 \equiv 2$，$2$ 模 $3$ 非平方，值为 $-1$。$\left(\frac{3}{5}\right)$：$3$ 模 $5$ 的平方根？$3^2=9\equiv4$，$4^2\equiv1$，$2^2\equiv4$——$3$ 非平方，值为 $-1$。
两者相等 ✓（$3,5$ 都 $3 \bmod 4$？$3\equiv3$，$5\equiv1$，至少一个 $1 \bmod 4$，应相等 ✓）。

验算 $p = 3, q = 7$：都 $3 \bmod 4$。$\left(\frac{7}{3}\right)$：$7 \equiv 1$，$1^2=1$，是平方，值 $1$。$\left(\frac{3}{7}\right)$：前面算过 $-1$。
相反 ✓。

## 2 配套公式：-1 与 2 的符号

二次互反律只处理「两个不同的奇素数」。还有两个「边界」素数需要单独公式：

$$\left(\frac{-1}{p}\right) = (-1)^{\frac{p-1}{2}}, \qquad \left(\frac{2}{p}\right) = (-1)^{\frac{p^2-1}{8}}$$

- $\left(\frac{-1}{p}\right) = 1$ 当且仅当 $p \equiv 1 \pmod 4$：$-1$ 是二次剩余恰在 $p$ 为 $1 \bmod 4$ 时。
- $\left(\frac{2}{p}\right) = 1$ 当且仅当 $p \equiv \pm 1 \pmod 8$：$2$ 是二次剩余恰在 $p$ 为 $\pm1 \bmod 8$ 时。<span class="marginnote">这三条——互反律、$-1$ 的公式、$2$ 的公式——合起来构成「勒让德符号的完整计算法则」。任何 $\left(\frac{a}{p}\right)$ 都能分解后由这三条递归求出，无需再算任何高次幂。高斯本人把这套系统视为数论的珍宝。</span>

这三个公式加上勒让德符号的完全乘性，就给出了**任意**勒让德符号的机械计算流程。

## 3 公式解析：计算一个大符号

$$\left(\frac{97}{1999}\right) = -\left(\frac{1999 \bmod 97}{97}\right) = -\left(\frac{59}{97}\right)$$

- **第一步，翻面**：$97 \equiv 1 \pmod 4$，所以二次互反律给出 $\left(\frac{97}{1999}\right) = \left(\frac{1999}{97}\right)$（至少一个 $1 \bmod 4$，符号相等）。
- **第二步，约化分子**：勒让德符号只依赖分子的模 $p$ 剩余类，$1999 \bmod 97 = 59$，故 $\left(\frac{1999}{97}\right) = \left(\frac{59}{97}\right)$。
- **第三步，递归翻面**：$59 \equiv 3 \pmod 4$、$97 \equiv 1 \pmod 4$，至少一个 $1 \bmod 4$，$\left(\frac{59}{97}\right) = \left(\frac{97 \bmod 59}{59}\right) = \left(\frac{38}{59}\right)$。
- **第四步，分解再乘**：$38 = 2 \times 19$，$\left(\frac{38}{59}\right) = \left(\frac{2}{59}\right)\left(\frac{19}{59}\right)$。$59 \equiv 3 \bmod 8$，$\left(\frac{2}{59}\right) = -1$。再对 $\left(\frac{19}{59}\right)$ 翻面：$19 \equiv 3, 59 \equiv 3$ 都 $3 \bmod 4$，符号相反，$\left(\frac{19}{59}\right) = -\left(\frac{59 \bmod 19}{19}\right) = -\left(\frac{2}{19}\right)$；$19 \equiv 3 \bmod 8$，$\left(\frac{2}{19}\right) = -1$。所以 $\left(\frac{19}{59}\right) = -(-1) = 1$。最终 $\left(\frac{97}{1999}\right) = (-1) \times 1 = -1$。

每步都是「翻面 + 约化 + 分解」，规模不断缩小——**二次互反律让符号计算有了类似辗转相除法的收敛性**。<span class="marginnote">注意到步进完全对应欧几里得算法的模约化。
事实上，用互反律计算勒让德符号的复杂度正是 $O(\log)$ 量级，与辗转相除同阶。
这也是它被称为「计算数论的基石」的原因。</span>

## 4 证明的图景：为什么会有互反律

二次互反律的证明很多，但最「看得见」的直觉来自**高斯引理**与**数格点**。

**高斯引理**：对奇素数 $p$ 与 $p \nmid a$，设 $S = \{a, 2a, \ldots, \frac{p-1}{2}a\}$，把每个元素化为模 $p$ 的「绝对值最小剩余」（在 $\pm 1, \ldots, \pm\frac{p-1}{2}$ 中），则

$$\left(\frac{a}{p}\right) = (-1)^n$$

其中 $n$ 是 $S$ 中「取到负值」的个数。<span class="marginnote">高斯引理把符号和「负数出现的次数」挂钩，这是理解 $\frac{p-1}{2}\frac{q-1}{2}$ 这种指数的来源。
而「数格点」证明则数矩形 $\{(x,y): 1 \le x \le \frac{p-1}{2}, 1 \le y \le \frac{q-1}{2}\}$ 里位于直线 $y = \frac{q}{p}x$ 两侧的格点数之差——几何与算术在此交汇。</span>

最著名的「数格点」证明（高斯第三个证明）思路：画 $p \times q$ 的网格，数直线 $qx = py$ 上方的格点数与下方的格点数之差，这个差恰好编码了指数 $\frac{p-1}{2}\frac{q-1}{2}$ 的奇偶性。
两种计数方式——从 $p$ 方向看和从 $q$ 方向看——给出同一差值，而每条边各对应一个符号，于是互反律成立。

**辨析｜易错点：**二次互反律只对**不同的奇素数**成立。$p = q$ 时符号 $\left(\frac{p}{p}\right) = 0$；$p = 2$ 或 $q = 2$ 时用 $\left(\frac{2}{p}\right)$ 公式。
把互反律套到 $2$ 或相同素数上是常见错误。

## 5 例题演练：用互反律计算符号

**例一：完整计算 $\left(\frac{97}{1999}\right)$。** $97 \equiv 1 \pmod 4$，互反律给出 $\left(\frac{97}{1999}\right) = \left(\frac{1999 \bmod 97}{97}\right) = \left(\frac{59}{97}\right)$。$59 \equiv 3$、$97 \equiv 1 \pmod 4$，再翻面：$\left(\frac{59}{97}\right) = \left(\frac{97 \bmod 59}{59}\right) = \left(\frac{38}{59}\right)$。$38 = 2 \times 19$：$\left(\frac{2}{59}\right) = -1$（$59 \equiv 3 \pmod 8$）；$\left(\frac{19}{59}\right)$，$19 \equiv 3$、$59 \equiv 3$ 都 $3 \bmod 4$，符号相反：$\left(\frac{19}{59}\right) = -\left(\frac{59 \bmod 19}{19}\right) = -\left(\frac{2}{19}\right)$，而 $19 \equiv 3 \pmod 8$，$\left(\frac{2}{19}\right) = -1$，故 $\left(\frac{19}{59}\right) = 1$。
汇总：$\left(\frac{97}{1999}\right) = (-1) \times 1 = -1$。<span class="marginnote">每一步都是「翻面 + 模约化」，数值一路缩小，像辗转相除一样收敛。
整条计算没有出现任何超过 $1999$ 的中间量——这就是互反律作为「计算引擎」的全部意义。</span>

**例二：三步法。** 求 $\left(\frac{1009}{2011}\right)$。$1009 \equiv 1 \pmod 4$，$\left(\frac{1009}{2011}\right) = \left(\frac{2011 \bmod 1009}{1009}\right) = \left(\frac{2}{1009}\right)$。$1009 \bmod 8 = 1$，$\left(\frac{2}{1009}\right) = 1$。
整个计算两行完成——若用欧拉判据要算 $1009^{1005}$，高下立判。

**例三：判断 $-1$ 与 $2$ 的身份。** 求 $\left(\frac{-1}{17}\right)$：$17 \equiv 1 \pmod 4$，$\left(\frac{-1}{17}\right) = 1$，$-1$ 是模 $17$ 的平方根（$4^2 = 16 \equiv -1$ ✓）。
求 $\left(\frac{2}{41}\right)$：$41 \equiv 1 \pmod 8$，$\left(\frac{2}{41}\right) = 1$，$2$ 是模 $41$ 的平方剩余。

**例四：互反律只对奇素数。** $p = q$ 时 $\left(\frac{p}{p}\right) = 0$，互反律不适用；$q = 2$ 时用 $\left(\frac{2}{p}\right)$ 公式。
比如 $\left(\frac{2}{7}\right) = (-1)^{(49-1)/8} = (-1)^6 = 1$，$2$ 是模 $7$ 的平方剩余（$3^2 = 9 \equiv 2$ ✓）。

**辨析｜易错点：**翻面时分子要先取模，即用 $p \bmod q$ 替换 $p$ 再翻；
顺序不能颠倒——先翻面后约化也正确（符号只依赖剩余类），但先约化再翻面更省计算。
另外「符号相反」的判断要看**两个**素数是否都 $3 \bmod 4$，只看一个会错。

**辨析｜易错点：**互反律计算中「分子取模再翻面」与「先翻面再取模」结果相同，但「取模」必须用欧几里得式的余数（$0 \le r < p$），不能取「绝对值最小剩余」后再翻——两种剩余类的符号可能相差一个 $(-1)$。习惯上统一用标准余数，避免符号漂移。

## 6 小结

- **二次互反律**：$\left(\frac{q}{p}\right)\left(\frac{p}{q}\right) = (-1)^{\frac{p-1}{2}\frac{q-1}{2}}$；两者在「至少一个 $1 \bmod 4$」时相等，都在 $3 \bmod 4$ 时相反。
- **配套公式**：$\left(\frac{-1}{p}\right) = (-1)^{(p-1)/2}$，$\left(\frac{2}{p}\right) = (-1)^{(p^2-1)/8}$。
- **计算流程**：翻面 → 模约化 → 分解 → 递归，复杂度 $O(\log)$，堪比辗转相除。
- 高斯称其为「黄金定理」，给出了八个证明；「数格点」证明把几何与算术连在一起。
- 互反律是勒让德符号机械计算的开关，也是通向类域论的大门。

在下一节，我们转向乘法的「阶」：给定 $a$ 与模 $m$，$a$ 的哪些幂回到 $1$？
最小的那个幂次就是**阶**，而「生成元」的存在性把我们引向**原根与指数**。
