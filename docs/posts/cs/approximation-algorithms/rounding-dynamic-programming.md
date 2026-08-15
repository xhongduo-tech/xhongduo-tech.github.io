---
title: 数据舍入与动态规划
date: 2026-08-07
---

# 数据舍入与动态规划

<div class="epigraph">
<p>如果一个问题的精确解只需伪多项式时间，那么把数据舍入到合适的精度，就能把「伪多项式」变成「真多项式」——代价只是损失一点点最优性。</p>
<footer>—— 威廉森与施莫伊斯（Williamson & Shmoys）</footer>
</div>

<div class="article-byline">
<p>第三级 · 近似算法 ｜ Williamson & Shmoys, *The Design of Approximation Algorithms\*, Ch.3 ｜ 2026-08-07</p>
</div>

## 为什么从数据舍入开始

前两篇的近似比都是常数或对数——算法本身简单，难在证明。这一篇换一种思路：
**先把问题精确地解出来，再为「多项式时间」这个要求做出牺牲**。
许多 NP 困难问题存在**伪多项式时间**的动态规划：时间关于输入的数字大小（而不是位数）是多项式的。
只要这些数字不是天文数字，DP 就能跑完；可一旦数字变大，它就崩了。

数据舍入（rounding data）的妙处在于：
**把大的数字故意舍掉一些精度，让 DP 重新变成多项式时间**，同时用舍入误差精确地控制近似比。
这把「时间」与「精度」做了一次显式的交易——精度由你定（任何 $\varepsilon$），时间也随之确定。
由此引出近似算法最重要的两个"方案"概念：**PTAS** 与 **FPTAS**。
本专题后面的背包、装箱、调度都会反复用这套交易。

## 1 伪多项式 DP：背包问题

**背包问题（knapsack）**：给定 $n$ 个物品，物品 $i$ 有价值 $v_i$ 和体积 $s_i$，背包容量为 $B$，选一个物品子集使总体积 $\le B$ 且总价值最大。它 NP 困难，但有著名的伪多项式 DP。

令 $V = \sum_i v_i$ 为价值总和。
定义 $S(i, c)$ 为「只考虑前 $i$ 个物品、恰好达到价值 $c$」所需的最小总体积（不存在则为 $+\infty$）。递推：

$$
S(i,c) = \min\{ S(i-1, c),\ S(i-1, c-v_i) + s_i \}
$$

初始 $S(0,0)=0$，其余 $S(0,c)=+\infty$。
最终答案是 $\max\{ c : S(n,c) \le B \}$。
时间与空间都是 $O(nV)$。<span class="marginnote">这个 DP 的「坐标轴」选的是价值而非体积，是为了把「最大化价值」翻译成「最小化达到给定价值的体积」——最大化问题换一个视角就变成最小化问题，这是建模的常见手法。注意它是伪多项式：$O(nV)$ 关于价值的大小 $V$ 线性，但关于表示 $V$ 的位数 $\log V$ 是指数的。</span>

**辨析｜易错点：** 「多项式时间」的定义按输入**位数**衡量。$V$ 用二进制表示需要 $\log V$ 位，因此 $O(nV)$ 不是 $n,\log V$ 的多项式。伪多项式算法在 $V$ 小时实用，却躲不过 NP 困难性的本质——除非我们能牺牲精度。

## 2 公式解析：背包的 FPTAS

数据舍入的做法是：把价值**按同一尺度缩小后取整**，让 DP 的价值坐标轴变短。
设 $V_{\max} = \max_i v_i$，取缩放因子 $K = \frac{\varepsilon V_{\max}}{n}$，定义**舍入后的价值**

$$
v_i' = \left\lfloor \frac{v_i}{K} \right\rfloor
$$

对舍入后的价值跑同一个 DP，它的价值总和 $\le n \cdot \frac{V_{\max}}{K} = \frac{n^2}{\varepsilon}$，所以时间变成 $O\big(n \cdot \frac{n^2}{\varepsilon}\big) = O(n^3/\varepsilon)$——**关于 $n$ 和 $1/\varepsilon$ 都是多项式**。

现在分析精度损失。
设原问题最优解 $\mathcal{O}$ 的总价值为 $\mathrm{OPT}$，舍入版本里 DP 找到的解 $\mathcal{O}'$ 是「按舍入价值」最优的。因为 $v_i' \le v_i$，任意解在舍入尺度下的价值都 $\le$ 原价值；且对每个物品

$$
v_i - K \cdot v_i' = v_i - K\left\lfloor \frac{v_i}{K} \right\rfloor \le K
$$

于是任何解在两个尺度下的价值差不超过「物品数 × K」：

$$
\mathrm{OPT} - \sum_{i \in \mathcal{O}'} v_i \ \le\ \left(\sum_{i \in \mathcal{O}} v_i' - \sum_{i \in \mathcal{O}'} v_i'\right) + nK \ \le\ 0 + nK = nK = \varepsilon V_{\max}
$$

而 $\mathrm{OPT} \ge V_{\max}$（最优解至少拿走价值最大的那件物品），所以

$$
\mathrm{OPT} - \sum_{i \in \mathcal{O}'} v_i \le \varepsilon V_{\max} \le \varepsilon\,\mathrm{OPT}
$$

即算法的价值 $\ge (1-\varepsilon)\mathrm{OPT}$。<span class="marginnote">三步拆解：① 缩放让 DP 坐标轴缩短到 $n^2/\varepsilon$；② 舍入误差用「每个物品至多丢 $K$」累计成 $nK$；③ 用平凡下界 $\mathrm{OPT} \ge V_{\max}$ 把绝对误差换算成相对误差。这套「缩放 → 累计误差 → 平凡下界换算」是 FPTAS 的标准三段论。</span>

**重点：** 我们由此得到**多项式时间近似方案（PTAS）**与**完全多项式时间近似方案（FPTAS）**的定义：

**PTAS**：对每个固定的 $\varepsilon>0$，存在 $(1+\varepsilon)$-近似算法，时间关于 $n$ 多项式（但关于 $1/\varepsilon$ 可以任意，比如 $n^{1/\varepsilon}$）。
**FPTAS**：时间关于 $n$ 与 $1/\varepsilon$ 都是多项式。

背包有 FPTAS；而对一般的 NP 困难问题，多数只可能有 PTAS 甚至更弱——这是问题的固有结构决定的。

## 3 调度：把 DP 从「总时间」上舍入

数据舍入不只在价值上工作，也可以在**时间**上工作。
考虑 $m$ 台相同机器的**最小完工时间**问题（上一篇的负载均衡）：
存在伪多项式 DP 以总处理时间 $\sum p_j$ 为坐标轴。把处理时间舍入成「桶」：

- 固定 $\varepsilon$，把每个 $p_j$ 向上取整到最近的 $\delta = \frac{\varepsilon}{1+\varepsilon} \cdot \frac{P_{\max}}{m}$（$P_{\max} = \max_j p_j$）的倍数，其中只保留「大作业」（$p_j \ge \varepsilon P_{\max}$，个数 $\le m/\varepsilon$ 可控）。
- 大作业个数不多，可以对其用 DP 精确枚举分配给各机器的方案；小作业再贪心补上。

这一构造给出的时间关于 $n$ 多项式、关于 $1/\varepsilon$ 可以是指数（如 $m^{1/\varepsilon}$），因此是 **PTAS 而非 FPTAS**。<span class="marginnote">同一道调度题：价值轴上的背包给了 FPTAS，时间轴上的完工时间只给了 PTAS。差别来自舍入后的坐标轴大小：前者缩到 $n^2/\varepsilon$，后者缩到 $(m/\varepsilon)^{1/\varepsilon}$ 量级——后者的 $1/\varepsilon$ 进了指数，于是 FPTAS 无望。理解这个差别，就理解了为什么 PTAS 与 FPTAS 是两个不同世界。</span>

**重点：** PTAS 与 FPTAS 的差别不是技术细节，而是**「误差倒数的代价」**：FPTAS 意味着 $1/\varepsilon$ 只让时间多项式增长，这在实践里几乎等于「随便你要多准」；PTAS 则可能在 $1/\varepsilon$ 增大时时间爆炸。能证 FPTAS 的问题（背包、装箱近似）通常有「舍入后坐标轴变短」的温和结构。

## 4 泛化：舍入的两种坐标

把本专题前面两篇与这一篇放在一起，你会看到数据舍入其实是一族技术。
**凡是被「单个数的大小」卡住的多项式算法，都可以试试把那个数舍入**：

| 场景 | 伪多项式坐标 | 舍入后坐标 | 结果 |
| --- | --- | --- | --- |
| 背包 | 价值总和 $V$ | $n^2/\varepsilon$ | FPTAS |
| 完工时间调度 | 总处理时间 $\sum p_j$ | $(m/\varepsilon)^{1/\varepsilon}$ | PTAS |
| 装箱近似 | 最优箱数 $\mathrm{OPT}$ | $1/\varepsilon$ 量级 | PTAS |

共同的方法论是：
**先找一个「关于数字大小多项式」的精确算法，再把数字舍入到使时间回归「关于位数多项式」的精度，最后用舍入误差的累计把近似比锁住**。这三步里，前两步是工程，第三步是数学。<span class="marginnote">数据舍入也是本专题后面《线性规划舍入》的「离散表兄」：LP 松弛是把整数变量放成实数，数据舍入是把实数数据收成整数。两种"舍入"方向相反，却都在「松弛—逼近」的同一张地图上。</span>

**辨析｜易错点：** 舍入方向很有讲究——背包把价值**向下取整**（$\lfloor \cdot \rfloor$）会低估，但保留可行性；如果把价值向上取整，DP 找到的解可能超出真实价值、失去「可行解」的身份。**舍入必须保证输出解对原问题可行**，这是设计红线。

## 5 小结

- **伪多项式 DP**：关于数字大小多项式、关于位数指数；是数据舍入的原料。
- **背包 FPTAS**：价值按 $K=\varepsilon V_{\max}/n$ 缩放取整，时间 $O(n^3/\varepsilon)$，价值 $\ge (1-\varepsilon)\mathrm{OPT}$。
- **PTAS vs FPTAS**：PTAS 时间对 $1/\varepsilon$ 可任意增长；FPTAS 要求关于 $1/\varepsilon$ 也多项式。
- **完工时间调度**只有 PTAS：因为舍入坐标里有 $(1/\varepsilon)^{1/\varepsilon}$，$1/\varepsilon$ 进了指数。
- **方法论三件套**：先找伪多项式精确算法 → 把卡时间的数字舍到合适精度 → 用误差累计锁近似比；前两步是工程，第三步是数学。
- **舍入红线**：向下取整保可行性，向上取整可能破坏可行解身份——输出必须对原问题可行。
- **一条包含链**：$\mathrm{FPTAS} \subseteq \mathrm{PTAS}$；PTAS 的时间对 $1/\varepsilon$ 可任意增长，FPTAS 则要求多项式——两者的分界由「舍入后坐标轴的长度」决定。

在下一节，我们把「舍入数据」换成一个更强大的原料——**线性规划的确定性舍入**：把整数规划松弛成 LP、解出分数解、再按阈值弹回整数，并用比值链 $\mathrm{OPT} \ge \mathrm{LP}$ 完成桥接。