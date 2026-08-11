---
title: 大筛法与 Bombieri-Vinogradov 定理
date: 2026-08-11
---

# 大筛法与 Bombieri-Vinogradov 定理

<div class="epigraph">
<p>上帝也许不掷骰子，但素数身上一定发生了什么奇怪的事。</p>
<footer>—— 保罗 · 爱多士（Paul Erdős）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 解析数论 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么需要对「几乎所有模数」说话

第五篇的 Dirichlet 定理给出了每个互素余类里的素数均匀分布，但有一个致命限制：**它只能处理 $q \le (\log x)^A$ 这种很小的模数**。若想对 $q$ 大到手 $\sqrt{x}$ 也能保证 $\psi(x;q,a) \approx x/\varphi(q)$，那就等价于广义黎曼假设（GRH）——没有人会证。但 1965 年 Bombieri（1961 年 Vinogradov 也有雏形）证明了一个奇迹：**虽然不能对每个 $q$ 成立，但把所有 $q \le \sqrt{x}$ 的误差加起来，总误差仍然小得吓人**。这就是 Bombieri–Vinogradov 定理，而它的证明核心是一种全新的、纯「筛选」风格的不等式——**大筛不等式**。

## 1 大筛不等式：一个纯组合-分析估计

设 $a_1, \ldots, a_N$ 是任意复数，记指数和 $S(\alpha) = \sum_{n \le N} a_n e(n\alpha)$（$e(t) = e^{2\pi i t}$）。**大筛不等式**（Montgomery–Vaughan 形式）：

$$
\sum_{q \le Q} \sum_{\substack{a=1 \\ (a,q)=1}}^{q} \left|S\left(\frac{a}{q}\right)\right|^2 \le (N + Q^2)\, \sum_{n \le N} |a_n|^2
$$

**重点：左边把所有分母 $q \le Q$、所有互素分子 $a$ 对应的约 $Q^2$ 个点上的指数和模方全部加起来，右边却只付出「数据量 $N$ + 频率数 $Q^2$」的代价**。<span class="marginnote">直觉上，$S(\alpha)$ 在「与 $n$ 相关的频率」上集中，而 $\{a/q\}$ 这些有理点分布得足够均匀，两者「不打架」——把模方和限制在 $N \Sigma|a_n|^2$ 附近。这像极了信号处理里的<strong>测不准原理</strong>：信号与其傅里叶变换不可能同时集中，大筛不等式就是它的离散有限版。</span>

**辨析｜易错点：** 左边遍历的是**互素**的 $(a,q)$，共 $\sum_{q\le Q}\varphi(q) \approx 3Q^2/\pi^2$ 个点，而不是 $Q^2$ 附近所有网格点。若不加互素条件，点在 $a/q$ 上会重复坍缩，不等式就不对——「约化分数」是这个估计的天然坐标系。

## 2 公式解析：$(N + Q^2)$ 为什么是两个量

$(N + Q^2)$ 这个形状不是拼凑的，它是两种极端情形的逼和：

**极端一：$Q$ 很小（测试频率稀少）**。此时不等式右边主要由 $N\Sigma|a_n|^2$ 主导。想想单个点：$|S(\alpha)|^2 \le N \sum|a_n|^2$ 就是 Cauchy–Schwarz，把约 $Q^2$ 个点全加起来，需要 $Q^2 \le N$ 时系数才不过分——频率没有「淹没」信号。
- **极端二：$N$ 很小（信号本身短）**。此时 $|S(\alpha)| \le \sum|a_n|$，而 $Q^2$ 个点上的「功率」被 $Q^2\Sigma|a_n|^2$ 界住——是**频率点的正交性**在起作用：$\frac1q\sum_{a} e(n a/q)$ 只在 $n \equiv 0 \pmod q$ 时非零。

**直觉**：$N + Q^2$ 是「信号自由度」与「频率自由度」的和——**不等式把两边的信息量都算进去了，谁大谁出价**。这正是测不准原理的算术形态：短信号（小 $N$）测不准，就花 $Q^2$ 的价钱；频率稀疏（小 $Q^2$）就花 $N$ 的价钱。<span class="marginnote"><strong>证明骨架</strong>：对每个 $q$，左边内层 $\sum_{(a,q)=1}|S(a/q)|^2$ 用「开平方、换序、先对 $a$ 求和」得到对角项 $\sum_n |a_n|^2 \cdot \#\{(a,q): n a \equiv n a\}$ 与交叉项；交叉项里的内层和 $\sum_{(a,q)=1} e((n-m)a/q) = \sum_{d|(n-m,q)} d\,\mu(q/d)$ 是 Ramanujan 和，它只在 $n \equiv m \pmod q$ 时非零——正是这一步把所有非对角项杀掉。</span>

## 3 从指数和到特征和：大筛的第二张脸

大筛不等式本身只涉及指数和 $e(n\alpha)$，但数论要的是特征和 $\sum a_n \chi(n)$。两者通过 Gauss 和（第六篇）搭桥：对原特征 $\chi$ 模 $q$，

$$
\chi(n) = \frac{1}{\tau(\chi)}\sum_{a \bmod q} \overline{\chi(a)}\, e\!\left(\frac{an}{q}\right)
$$

把特征展开成指数和的线性组合，再代入大筛不等式，得到**特征和版本**：

$$
\sum_{q \le Q} \sum_{\chi \bmod q} \left|\sum_{n \le N} a_n \chi(n)\right|^2 \ll (N + Q^2)\, \sum_{n \le N} |a_n|^2
$$

其中 $\chi$ 取模 $q$ 的全部特征（原特征为主）。**重点：特征和版本把「对所有模数平均」变成了可行操作**——这正是 Bombieri–Vinogradov 的核心输入。<span class="marginnote">注意这里不用关心 Siegel 零点的「若存在」分支：大筛给的是<strong>对所有 $q$ 统一平均</strong>的硬不等式，没有任何例外模数。这就是「平均理论」之所以能绕过第七篇那个无效常数的根本原因。</span>

## 4 Bombieri–Vinogradov 定理：GRH 的平均版

现在把大筛喂给 $\psi(x; q, a)$ 的 Perron 表示（第五篇的 L-函数 + 反演公式），得到：

**Bombieri–Vinogradov 定理**：对任意 $A > 0$，存在 $B = B(A)$，使得当 $Q = x^{1/2}(\log x)^{-B}$ 时，

$$
\sum_{q \le Q}\ \max_{y \le x}\ \max_{(a,q)=1}\left|\psi(y;q,a) - \frac{y}{\varphi(q)}\right| \ll_A \frac{x}{(\log x)^A}
$$

**重点：GRH 对每个 $q$ 给出的误差 $O(\sqrt{x}\log^2 x)$，这里对全部 $q \le \sqrt{x}$ 平均求和后，总误差仍是 $x/\log^A x$**——平均意义上，素数在每个互素余类里的均匀性达到了 GRH 的水平。<span class="marginnote"><strong>辨析｜易错点：</strong> 三个「max」缺一不可。若把 $\max_y$ 拿掉（固定 $y = x$），定理退化成简单情形；$\max_{(a,q)=1}$ 更是关键——「对余类取最大」才保证每个余类都好，若只对 $a$ 求和，误差会被人为稀释。这正是大筛「对 $a$ 先取模方再求和」的精细之处。</span>

**意义**：Bombieri–Vinogradov 让大量「假设 GRH 才能做的定理」变成无条件成立——条件是结论对 $q \le \sqrt{x}$ 的**平均**成立即可。Hooley 证明三素数定理的奇偶部分、素数在短区间的密度估计、Goldston–Yıldırım 型工作，全都建立在这条定理上。

## 5 推广与前沿：Elliott–Halberstam 与素间隙

Bombieri–Vinogradov 的界 $Q = \sqrt{x}$ 是「无条件能到的极限」。若假设更强的 **Elliott–Halberstam（EH）猜想**——把 $Q$ 推到 $x^{1-\varepsilon}$——会发生什么？

**素数间隙**：Maynard 与张益唐的思路都依赖「在某个区间里找到若干素数」的筛法；对任意大的有界间隙，EH 型平均给出最强武器。2013 年张益唐在**未假设 EH**、仅用 Bombieri–Vinogradov 及其素数推进（PIN）的情况下证明了「存在无穷多对间隔 < 7000 万的素数」，随后 Polymath 与 Maynard 把间隔压到 246。<span class="marginnote">这段历史是「平均理论」的教科书级案例：一个看似抽象的平均均匀性定理，直接牵动「孪生素数猜想」这条最古老的直觉问题。细节见第三级《数论专题》的孪生素数条。</span>
- **三素数定理**：Vinogradov 1937 年已无条件证明「每个充分大的奇数都是三个素数之和」，其证明同样建立在特征和的估计上——大筛之后又多了 Bombieri–Vinogradov 这条更顺的路。

**重点：大筛与 Bombieri–Vinogradov 的价值不在单个定理，而在它把「平均化」确立为解析数论的核心策略**——凡是个别对象难啃的，先问「平均下来行不行」，平均可行就足够推进绝大多数应用。

## 6 小结

- **大筛不等式**：$\sum_{q\le Q}\sum_{(a,q)=1}|S(a/q)|^2 \le (N + Q^2)\sum|a_n|^2$——「信号自由度 + 频率自由度」的测不准原理，证明全靠 Ramanujan 和杀掉交叉项。
- **特征和版本**：借助 Gauss 和把特征展开成指数和，得到对所有模数与特征统一的平均估计——**没有 Siegel 零点例外**。
- **Bombieri–Vinogradov 定理**：$Q = \sqrt{x}/\log^B x$ 时，对 $q$ 平均、对 $y$ 与 $a$ 取 max 的误差总计 $\ll x/\log^A x$——GRH 的均匀性在平均意义下无条件成立。
- **Elliott–Halberstam 猜想**：把 $Q$ 推向 $x^{1-\varepsilon}$，是素间隙（246）、更精细素数分布研究的推进器。
- 「先平均、再应用」是解析数论的元方法：不可逐点证明的，往往可平均证明。

最后一节，我们把分析的工具对准一个「加法」问题：**Waring 问题与 Hardy–Littlewood 圆法**——整数如何表示成幂的和，以及那个把整数论变成傅里叶积分的宏伟蓝图。
