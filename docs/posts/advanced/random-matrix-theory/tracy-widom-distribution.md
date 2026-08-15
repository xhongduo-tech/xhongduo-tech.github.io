---
title: Tracy–Widom 分布
date: 2026-08-07
---

# Tracy–Widom 分布

<div class="epigraph">
<p>最大的特征值，比其余的都要「大」得多——但它的涨落有自己的定律。</p>
<footer>—— 克雷格 · 特蕾西与哈罗德 · 威多姆（Craig Tracy & Harold Widom, 1994）</footer>
</div>

<div class="article-byline">
<p>第四级 · 随机矩阵理论 ｜ Mehta Ch. 6；Tracy–Widom, CMP 159 (1994) ｜ 2026-08-07</p>
</div>

## 为什么从 Tracy–Widom 分布开始

半圆律回答的是「特征值整体怎么分布」，但物理学家和统计学家更常关心极值：**最大的那个特征值在哪？** 这正是随机矩阵理论二十世纪九十年代最漂亮的发现之一——**Tracy–Widom 分布**。它刻画谱边缘的涨落：最大特征值不落在半圆边缘的确定位置，而是围绕它做 $N^{-2/3}$ 量级的随机摆动，摆动的极限律由一个非线性微分方程（Painlevé II）定义。更惊人的是，这个分布到处出现：随机置换的最长递增子序列、随机增长界面的高度、Wishart 矩阵的最大奇异值、金融市场协方差阵的谱……它成了「极值统计」在关联体系中的通用语言，也是普适性（第 9 篇）最著名的招牌。

## 1 边缘涨落尺度：$N^{-2/3}$

先看清「最大特征值涨得多大」。半圆密度在边缘处像平方根一样消失：

$$\rho_{\text{sc}}(x) = \frac{1}{2\pi}\sqrt{4 - x^2} \sim \frac{1}{\pi}\sqrt{2}\sqrt{2 - x}, \qquad x \to 2^{-}$$

在边缘 $x = 2$ 附近，把坐标换成 $x = 2 + N^{-2/3} s$，密度的量级变成 $\sqrt{N^{-2/3}} = N^{-1/3}$。这意味着：**「一个特征值」对应的长度单元在边缘处是 $N^{-2/3}$**。半圆律只保证 $\lambda_{\max} \to 2$（一阶项），但二阶项如何？答案是

$$\lambda_{\max} = 2 + N^{-2/3} \, \zeta, \qquad \zeta \sim \text{Tracy–Widom}$$

$\zeta$ 是一个 $O(1)$ 的随机变量，其分布不随 $N$ 变化。<span class="marginnote">这个 $N^{-2/3}$ 标度是随机矩阵「软边缘」的指纹。对比：半圆内部特征值间距是 $N^{-1}$，而边缘间距是 $N^{-2/3}$——边缘比体内「稀」得多，因为密度在那里以平方根消失。这也解释了为什么边缘涨落比体内大一个数量级。</span>

这个 $N^{-2/3}$ 的重要性怎么强调都不过分：它独立于矩阵元素分布，是纯几何（密度平方根消失）的产物。所有「软边缘」现象共享它，而 Tracy–Widom 分布则是这个标度下最精细的统计。

## 2 Airy 核与 Fredholm 行列式

要得到最大特征值的极限分布，沿用第 4 篇的行列式点过程思路：$\lambda_{\max} \le s$ 当且仅当**没有特征值超过 $s$**，即

$$F_2(s) = \mathbb{P}(\lambda_{\max} \le s) = \mathbb{P}(\text{区间 } [s, \infty) \text{ 内特征值个数为 } 0)$$

对 DPP，「某个区间内没有点」的概率由 Fredholm 行列式给出。边缘极限下核变成 **Airy 核**：

$$K(x, y) = \frac{\operatorname{Ai}(x)\operatorname{Ai}'(y) - \operatorname{Ai}'(x)\operatorname{Ai}(y)}{x - y}$$

其中 $\operatorname{Ai}$ 是 **Airy 函数**——方程 $u'' = x u$ 的衰减解。<span class="marginnote">Airy 函数是「一维量子粒子在线性势阱中」的波函数（如均匀重力场中的冷原子、光的焦散线）。Airy 核于是可以读作：$K(x,y) = \sum_{\text{占据能级}} \phi(x)\phi(y)$，是软边缘处单粒子密度矩阵的极限——与第 4 篇「自由费米子 = DPP」的图景完全衔接。</span>

**Fredholm 行列式**：$F_2(s) = \det(I - K_s)$，其中 $K_s$ 是 Airy 核限制在 $[s, \infty)$ 上的积分算子，$\det(I - K_s) = \sum_{k\ge 0} \frac{(-1)^k}{k!}\int_{[s,\infty)^k} \det[K(x_i,x_j)] \, dx_1\cdots dx_k$。这是「区间 $[s,\infty)$ 内没有点的概率」的行列式表达，来自第 4 篇 $R_k = \det[K]$ 的幂级数求和。

Fredholm 行列式在 1990 年代初被 Tracy 与 Widom 化成了可计算的常微分方程——这是整个理论「从行列式到 ODE」的关键转折。

## 3 公式解析：Painlevé II 与 $F_2$

Tracy–Widom 的天才之处，是把那个貌似不可操作的 Fredholm 行列式，化简成**一个可数值求解的非线性 ODE**。

$$F_2(s) = \exp\!\left(-\int_{s}^{\infty} (x - s)\, q(x)^2 \, dx\right)$$

其中 $q$ 是 **Painlevé II 方程**的解：

$$q''(x) = x\, q(x) + 2\, q(x)^3, \qquad q(x) \sim \operatorname{Ai}(x) \ \text{当 } x \to +\infty$$

- **$q$ 是 Hastings–McLeod 解**：Painlevé II 有无数解，但加上「当 $x \to +\infty$ 时 $q(x) \sim \operatorname{Ai}(x)$（指数衰减）」这个边界条件后唯一确定。$q$ 在 $x \to -\infty$ 时像 $\sqrt{-x/2}$ 那样增长。
- **$(x - s) q(x)^2$**：来自「数 $[s,\infty)$ 内点的个数」的指数表达式——Fredholm 行列式取对数后，用 $q$ 的积分表示。直观上，$q(x)^2$ 是「在 $x$ 处存在边缘模式」的密度，$(x-s)$ 是「该模式到区间端点的距离」。
- **$F_2$ 的性质**：$F_2$ 是连续 CDF，右尾 $1 - F_2(s) \sim \frac{e^{-\frac{4}{3}s^{3/2}}}{16\pi s^{3/2}}$（$s \to +\infty$），左尾 $F_2(s) \sim e^{-\frac{|s|^3}{12}}$（$s \to -\infty$）。右尾是「被拉伸的指数」，左尾是「立方指数」——**强烈非对称**，向左拖尾。

**三种对称性**：对 $\beta = 1$（GOE）与 $\beta = 4$（GSE），Tracy–Widom 也用同一个 $q$ 表达：$F_1(s)^2 = F_2(s)\, e^{-\int_s^\infty q(x)\,dx}$，$F_4$ 则有类似公式（带 $2^{2/3}$ 的尺度因子）。三种分布形状相近，但位置与尺度不同，且 $\beta$ 越大分布越靠右。<span class="marginnote">这个「三个 $\beta$ 共享同一个 Painlevé II 解」的事实，是 $\beta$ 统一性的极致体现：排斥强度的差别只改变积分表达式的组合系数，不改变背后的可积结构。</span>

## 4 一个传奇应用：最长递增子序列

Tracy–Widom 分布的第一个「出圈」应用完全不在矩阵里。**Baik–Deift–Johansson（1999）**：随机置换 $\pi \in S_n$ 的**最长递增子序列（LIS）**长度 $L_n$ 满足

$$\frac{L_n - 2\sqrt{n}}{n^{1/6}} \xrightarrow{d} \text{TW}_2$$

这个结果震惊了组合学界——一个纯离散的组合量，其极限分布竟是随机矩阵的 Painlevé II。<span class="marginnote">BDJ 定理的证明用到了「增长型幼犬路径」与 Toeplitz 行列式：$L_n$ 的分布函数写成 Hankel/Toeplitz 行列式，而它的极限与 Airy 核的 Fredholm 行列式是同一个对象。这是「组合 → 行列式 → Painlevé」链条的第一次公开演出。</span>

此后 TW 分布如雨后春笋般出现：**随机增长模型**（TASEP、定向聚合物、KPZ 方程）的界面高度涨落、**Wishart 矩阵**的最大奇异值（多元统计、主成分分析）、**随机汉森森—O'Connell–Yor** 模型、**量子输运**中的电导涨落。它们背后的共同机制：**某种确定性/普适性极限过程（Airy 过程族）在不同系统中反复出现**。

## 5 数值与识别

数值上识别 TW 分布并不难：生成 $N = 1000$ 的 GUE 矩阵，取最大特征值，重复上千次得到 $(\lambda_{\max} - 2)N^{2/3}$ 的直方图，与 $F_2$ 的密度曲线叠放即可验证。分布的特征很醒目：**单峰、左尾长而缓、右尾短而急**，均值约为 $-1.77$，标准差约为 $0.90$。

实践中判断「这是不是 TW」有三个线索：其一，**尺度是 $N^{-2/3}$**（对谱边缘）或 $n^{-1/6}$（对 LIS 类问题）；其二，**左尾重、右尾轻**；其三，**无需任何参数拟合**——标准化后直接对标准 TW。<span class="marginnote">要注意与经典极值统计（Fisher–Tippett–Gnedenko）区分：独立同分布样本的极大值服从 Gumbel/Fréchet/Weibull，而 TW 描述<strong>强关联</strong>样本的极值。关联让极值涨落被压缩：Gumbel 的标准差是 $O(1)$ 常数，而 TW 的涨落随样本数按 $n^{-1/6}$ 缩小——关联体系的极值「更可预测」。</span>

## 6 术语速查表与数值识别要点

| 术语 | 记号 / 公式 | 一句话含义 |
| --- | --- | --- |
| 软边 | 谱边缘，密度平方根消失 | $N^{-2/3}$ 标度的来源 |
| 硬边 | 谱被 0 或有限端点顶住 | $N^{-2}$ 标度、Bessel 核 |
| Airy 函数 | $\operatorname{Ai}$，$u''=xu$ 的衰减解 | 软边缘极限的基函数 |
| Airy 核 | $K(x,y)=\frac{\operatorname{Ai}(x)\operatorname{Ai}'(y)-\operatorname{Ai}'(x)\operatorname{Ai}(y)}{x-y}$ | 边缘极限核 |
| Fredholm 行列式 | $\det(I-K_s)$ | 区间内无点的概率 |
| Painlevé II | $q''=xq+2q^3$ | 定义 $F_\beta$ 的非线性 ODE |
| Hastings–McLeod 解 | $q(x)\sim\operatorname{Ai}(x)$（$x\to+\infty$） | 唯一化的边界条件 |
| TW 右尾 | $1-F_2(s)\sim\frac{e^{-\frac43s^{3/2}}}{16\pi s^{3/2}}$ | 拉伸指数尾 |
| TW 左尾 | $F_2(s)\sim e^{-|s|^3/12}$ | 立方指数尾 |
| BDJ 定理 | $(L_n-2\sqrt n)/n^{1/6}\to\text{TW}_2$ | LIS 的极限分布 |

**数值识别要点**：识别「这是不是 TW」不需要拟合参数。标准流程：

- 生成 $N=1000$ 的 GUE 矩阵，取最大特征值 $\lambda_{\max}$，重复约 $10^4$ 次，做标准化 $(\lambda_{\max}-2)N^{2/3}$。
- 画直方图，与 $F_2$ 的密度曲线叠放。判断要点：**单峰、左尾长而缓、右尾短而急**。
- 对 Wishart 软边，用 $(\lambda_{\max}-b)N^{2/3}$（$b=(1+\sqrt c)^2$）标准化，同样应与 TW 叠合。
- 对 LIS，用 $(L_n-2\sqrt n)/n^{1/6}$ 与 TW$_2$ 对比。

**三个易错点**：

1. **标准化常数**：边缘涨落是 $N^{-2/3}$（方阵）或 $n^{-1/6}$（LIS），用错指数分布就发散。
2. **位置项**：中心项是 $2$（方阵、方差 $1/N$ 归一）或 $2\sqrt n$（LIS），不是 $0$。
3. **样本量**：$N$ 至少几百到上千，$N^{-2/3}$ 是慢收敛，样本太小会对不上。

**三种 $\beta$ 的位置差异**：数值上 $\beta$ 越大，TW 分布整体越靠右——强排斥把最大特征值推得更高：$F_4$ 相对 $F_2$ 右移约 $O(1)$，$F_1$ 相对 $F_2$ 略左移。三者形状相似，但位置与宽度不同，实际检验时须用对应 $\beta$ 的分位数。

**辨析｜易错点：** TW 的右尾 $1-F_2(s) \sim \frac{e^{-\frac{4}{3}s^{3/2}}}{16\pi s^{3/2}}$ 衰减极快，左尾 $F_2(s) \sim e^{-|s|^3/12}$ 更慢，因此分布强烈不对称。「超过 $b$ 才算显著」必须用**右尾分位数**；若误用对称分位点，会把大量噪声墙内的峰值误判为信号。实际 $p$ 值做法：算 $q = (\lambda_{\max} - b)N^{2/3}$，代入 $F_2$ 右尾即可。

## 7 小结

- **软边缘标度**：$\lambda_{\max} = 2 + N^{-2/3}\,\zeta$，标度来自半圆密度在边缘的平方根消失。
- **Airy 核**：边缘极限下 DPP 的核 $K(x,y) = \frac{\operatorname{Ai}(x)\operatorname{Ai}'(y) - \operatorname{Ai}'(x)\operatorname{Ai}(y)}{x-y}$。
- **Fredholm 行列式**：$\mathbb{P}(\lambda_{\max} \le s) = \det(I - K_s)$，即「$[s,\infty)$ 内无点」的概率。
- **Painlevé II**：$F_2(s) = e^{-\int_s^\infty (x-s)q(x)^2 dx}$，$q'' = xq + 2q^3$，$q \sim \operatorname{Ai}$——Fredholm 行列式被化为 ODE。
- **BDJ 定理**：LIS 长度 $(L_n - 2\sqrt{n})/n^{1/6} \to \text{TW}_2$——TW 跨出矩阵的第一个组合学应用。

在下一节，我们将离开方阵世界，看矩形矩阵样本协方差的谱——那里最大的特征值同样由 TW 控制，而整体谱由 Marchenko–Pastur 律描述。