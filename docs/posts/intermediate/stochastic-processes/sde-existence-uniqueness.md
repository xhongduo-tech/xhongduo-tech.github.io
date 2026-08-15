---
title: 随机微分方程（SDE）的解与存在唯一性
date: 2026-08-07
---

# 随机微分方程（SDE）的解与存在唯一性

<div class="epigraph">
<p>方程里住着随机，解便是「适应未来」的过程——存在与唯一，是随机分析给建模者的承诺。</p>
<footer>—— 斯特罗克 · 瓦拉德汉（Daniel W. Stroock）</footer>
</div>

<div class="article-byline">
<p>第二级 · 随机过程 ｜ 张波《应用随机过程》§8.4 ｜ 2026-08-07</p>
</div>

## 随机世界里的「常微分方程」

普通微分方程 $\dot x = f(x)$ 描述确定性演化；**随机微分方程（Stochastic Differential Equation, SDE）**把演化放进噪声里：
$$
dX(t) = \mu\big(X(t), t\big)\, dt + \sigma\big(X(t), t\big)\, dB(t).
$$
**漂移项 $\mu$ 管「趋势」，扩散项 $\sigma$ 管「噪声」**。它是金融（利率、股价）、物理（朗之万方程）、生物（随机种群）建模的标准语言。

但「方程有解吗？解唯一吗？」在随机世界里不是免费的——需要专门的**存在唯一性定理**。这节给出：SDE 的严格含义（积分形式）、解的定义、以及保证「有且仅有唯一解」的条件。**没有这条定理，一切建模都是空中楼阁。**

这节的价值在于「先证明地基再盖楼」：金融里每个利率模型、每个股价模型都是一个 SDE，如果它们连「解存在且唯一」都保证不了，后面的一切定价公式都是空的。**读完这节你会明白——Lipschitz 条件不是书斋里的教条，而是建模者的安全网**。<span class="marginnote">SDE 与 ODE 的关键差别：<strong>SDE 的解是「适应过程」——$X(t)$ 在时刻 $t$ 只依赖 $B(s)$（$s \le t$），不偷看未来</strong>。这个「适应性」要求让存在唯一性的证明比 ODE 复杂得多：Itô 积分本身就要被积函数适应。</span>

## 1 SDE 的严格形式

**SDE 不能逐点理解**（$dB$ 不存在），必须写成**积分形式**：
$$
X(t) = X(0) + \int_0^t \mu\big(X(s), s\big)\, ds + \int_0^t \sigma\big(X(s), s\big)\, dB(s).
$$
**第一个积分是普通（路径）积分，第二个积分是 Itô 积分。** 说「$X$ 是 SDE 的解」意思是：上式对几乎所有轨道成立，且 $X$ 是适应过程。

**强解 vs 弱解**：

- **强解（strong solution）**：在给定的布朗运动 $B$ 上，存在适应过程 $X$ 使积分方程成立——「用同一股噪声就能构造解」；
- **弱解（weak solution）**：可以重新选布朗运动 $B'$ 使方程成立——「换个噪声环境也能配出解」。<span class="marginnote">强弱之分的重要场景：<strong>金融建模通常要强解</strong>（你需要「给定市场的噪声」来定价）；而<strong>某些退化方程只有弱解没有强解</strong>。存在唯一性定理给的通常是「强解存在且唯一」。</span>

强弱之分还决定「模拟怎么做」：强解在给定噪声下逐路径可复现（蒙特卡洛对同一噪声能重放路径），弱解只在分布意义下确定（重放需要重新采样）。**工程上若只关心期望（定价、风险指标），弱解够用；若要路径级复现，必须强解**——这是两种解在实践中最直接的分野。

## 2 存在唯一性定理

**定理（Itô / Lipschitz 条件）**：设系数 $\mu$、$\sigma$ 满足：

1. **全局 Lipschitz**：存在 $K$ 使
$$
|\mu(x,t) - \mu(y,t)| + |\sigma(x,t) - \sigma(y,t)| \le K |x - y|;
$$
2. **线性增长**：存在 $K$ 使
$$
|\mu(x,t)| + |\sigma(x,t)| \le K(1 + |x|);
$$
3. $X(0)$ 与 $B$ 独立、$E[X(0)^2] < \infty$。

则 SDE 有**唯一强解**，且解连续依赖初值。<span class="marginnote">两个条件的角色：<strong>Lipschitz 保证「解的唯一性」（Picard 迭代的压缩性），线性增长保证「解不爆炸」（多项式时刻有界）</strong>。它们正是 ODE 的 Picard-Lindelöf 定理在随机世界的翻版，只是证明更复杂（要同时处理 Itô 积分的矩估计）。</span>

**证明骨架（Picard 迭代 + 鞅矩估计）**：定义迭代 $X^{n+1}(t) = X(0) + \int \mu(X^n)ds + \int \sigma(X^n)dB$。用 Itô 等距 + Grönwall 引理证明 $\{X^n\}$ 是 $L^2$ 柯西列，极限即唯一解。**Itô 等距在这里扮演「能量估计」的角色**——它是随机世界里的 Grönwall 引理原料。

## 3 公式解析：Lipschitz 条件为什么够用

**目标：理解 Lipschitz 条件如何让 Picard 迭代收敛——用 Itô 等距把「解的差距」变成可控制的递推。**

第一步，设两个候选解 $X^1$、$X^2$，记 $D(t) = X^1(t) - X^2(t)$。由 SDE 相减：
$$
D(t) = \int_0^t [\mu(X^1) - \mu(X^2)]ds + \int_0^t [\sigma(X^1) - \sigma(X^2)]dB.
$$
第二步，取平方期望。交叉项期望为 0（Itô 积分期望 0），只剩两项：
$$
E[D(t)^2] \le 2E\Big[\Big(\int [\Delta\mu] ds\Big)^2\Big] + 2E\Big[\Big(\int [\Delta\sigma] dB\Big)^2\Big].
$$
第三步，用 Lipschitz + 等距 + Cauchy-Schwarz：
$$
E[D(t)^2] \le 2K^2 t \int_0^t E[D(s)^2]ds + 2K^2 \int_0^t E[D(s)^2] ds.
$$
第四步，Grönwall 引理。$E[D(t)^2] \le C \int_0^t E[D(s)^2]ds$ 型不等式推出 $E[D(t)^2] = 0$——**两解几乎必然相同，唯一性成立**。

**这个推导为什么重要**：它示范了 SDE 理论的标准技巧——**Itô 等距把「解的随机差距」化成「确定性积分不等式」，再用 Grönwall 封死**。Lipschitz 条件不是摆设：它精确地让「差距的平方」能被自己的积分控制。

## 4 退化情形：Lipschitz 不满足时

当系数不满足 Lipschitz（比如平方根扩散 $\sigma(x) = \sqrt x$），强解可能不存在或非唯一：

- **CIR / Feller 条件**：$\sigma(x) = \sqrt x$ 的平方根扩散（利率建模）在 $\sigma(0) = 0$ 处不 Lipschitz，但仍有唯一**弱解**（且解保持非负）——这需要专门的「Yamada-Watanabe」类定理。
- **非唯一例子**：$\sigma(x) = \sqrt{|x|}$ 可能多解或爆解。<span class="marginnote">建模启示：<strong>真实的金融/生物模型常踩在 Lipschitz 边界上</strong>——CIR 利率、广义 OU、logistic 种群。处理它们要记住「全局 Lipschitz 只是充分条件」，别因系数不光滑就放弃建模，改用弱解理论或数值方法。</span>

把「CIR 在边界不 Lipschitz」想清楚：$\sigma(x) = \sqrt x$ 在 $x = 0$ 处斜率无穷（$\sqrt x$ 的导数在 0 发散），但 Yamada-Watanabe 条件（$|\sigma(x)-\sigma(y)| \le K\sqrt{|x-y|}$）仍然成立——**比 Lipschitz 更弱的 Hölder-$1/2$ 条件也足以保证唯一性**。这就是为什么 CIR 利率模型既能保持非负、又有唯一解：它的扩散项刚好踩在「够用」的临界上。

## 5 数值求解：欧拉-丸山方法

即使解存在，显式解通常没有闭式。**欧拉-丸山（Euler-Maruyama）方法**是 SDE 数值解的标准格式：
$$
X_{k+1} = X_k + \mu(X_k, t_k)\, \Delta t + \sigma(X_k, t_k)\, \sqrt{\Delta t}\, Z_k, \qquad Z_k \sim N(0,1).
$$
**「确定步长 $\mu\Delta t$ + 随机步长 $\sigma\sqrt{\Delta t}Z$」——这正是 Itô 增量结构**（$dB$ 是 $\sqrt{dt}$ 量级）。它的强收敛阶是 $1/2$（比 ODE 的欧拉法 $1$ 阶慢），弱收敛阶是 $1$——随机性让收敛变慢。<span class="marginnote">欧拉-丸山与蒙特卡洛的组合是金融工程的标配：<strong>「离散化 + 多条路径平均」估计期权价格、风险指标</strong>。离散化误差用弱收敛阶控制，统计误差用 $\sqrt{N}$ 控制——两个误差源的权衡是数值随机分析的日常。</span>

把「两个误差源的权衡」量化：设时间步 $\Delta t$、路径数 $N$，离散化误差 $\propto \Delta t$（弱阶 1），统计误差 $\propto 1/\sqrt N$。总误差最小化大致要求 $\Delta t \sim 1/\sqrt N$——**步长与路径数要一起调，而不是单方面加密**。这个权衡是金融蒙特卡洛里反复出现的工程判断。

## 6 SDE 术语速查

| 术语 | 记号 | 含义 | 要点 |
| --- | --- | --- | --- |
| SDE | $dX = \mu dt + \sigma dB$ | 随机微分方程 | 积分形式才严格 |
| 漂移 / 扩散 | $\mu$ / $\sigma$ | 趋势 / 噪声 | 两个角色分离 |
| 强解 | 给定噪声构造 | 逐路径可复现 | 金融定价常用 |
| 弱解 | 可换噪声 | 仅分布确定 | 退化方程可能只有弱解 |
| Lipschitz 条件 | $\|\mu(x)-\mu(y)\|+\|\sigma(x)-\sigma(y)\| \le K\|x-y\|$ | 唯一性 | Picard 压缩性 |
| 线性增长 | $\|\mu\|+\|\sigma\| \le K(1+\|x\|)$ | 不爆炸 | 多项式矩有界 |
| 欧拉-丸山 | $X_{k+1} = X_k + \mu\Delta t + \sigma\sqrt{\Delta t}Z$ | 数值求解 | 强收敛阶 $1/2$ |

## 7 小结

- **SDE**：$dX = \mu dt + \sigma dB$，严格含义是积分形式（第二个积分是 Itô 积分）。
- **强解 vs 弱解**：强解用给定噪声构造；弱解可换噪声。存在唯一性定理给强解。
- **存在唯一性**：全局 Lipschitz + 线性增长 + 平方可积初值 ⟹ 唯一强解。
- **证明工具**：Picard 迭代 + Itô 等距 + Grönwall——Lipschitz 让差距被自身积分控制。
- **数值**：欧拉-丸山，强收敛阶 $1/2$——随机让收敛变慢。
- **Yamada-Watanabe**：比 Lipschitz 更弱的 Hölder-$1/2$ 也保证唯一——CIR 等边界模型「刚好够用」。

在下一节，我们动手解几个经典 SDE：**常见 SDE 的求解：几何布朗运动与 OU 过程**——用 Itô 公式把方程变成可积形式。
