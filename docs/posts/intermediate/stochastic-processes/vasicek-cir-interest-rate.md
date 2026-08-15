---
title: 随机利率模型：Vasicek 与 CIR 模型
date: 2026-08-07
---

# 随机利率模型：Vasicek 与 CIR 模型

<div class="epigraph">
<p>利率不是常数，它绕着长期均值呼吸——而每一次呼吸，都决定着一张债券的价格。</p>
<footer>—— 约翰 · 考克斯（John C. Cox）</footer>
</div>

<div class="article-byline">
<p>第二级 · 随机过程 ｜ 张波《应用随机过程》§10.3 ｜ 2026-08-07</p>
</div>

## 让利率「活」起来

Black-Scholes 假设利率 $r$ 是常数——但现实里利率随时间随机波动（央行加息、经济周期）。**利率建模**把 $r(t)$ 本身变成一个随机过程，核心是「**均值回复**」：利率长期围绕某个水平波动，偏离后会被拉回。两个最经典的模型：

- **Vasicek 模型**（1977）：
$$
dr = \theta(\alpha - r)\, dt + \sigma\, dB;
$$
- **CIR 模型**（Cox-Ingersoll-Ross, 1985）：
$$
dr = \theta(\alpha - r)\, dt + \sigma \sqrt{r}\, dB.
$$

两者的差别只有一个 $\sqrt r$——但这点差别决定了**利率能否为负**，是「有没有 Feller 条件」的分水岭。它们都是第八篇 OU 过程的近亲：Vasicek 就是 OU，CIR 是「平方根扩散」版的 OU。<span class="marginnote">均值回复模型的家谱：<strong>Vasicek = OU（第八篇讲过解式）；CIR = OU + $\sqrt r$ 扩散</strong>。OU 的解式、平稳分布、积分因子全部可以直接借用——学习时把它们当「OU 的金融马甲」最省力。</span>

本节目标：对比两个模型、推导各自的解与平稳分布、并理解「为什么债券价格要解微分方程」。

## 1 Vasicek 模型

**Vasicek**：$dr = \theta(\alpha - r)dt + \sigma dB$——OU 过程。

**解式**（第八篇）：
$$
r(t) = r(0)e^{-\theta t} + \alpha(1 - e^{-\theta t}) + \sigma e^{-\theta t}\int_0^t e^{\theta s} dB(s).
$$

**性质**：
- **均值回复**：$E[r(t)] \to \alpha$，速度 $\theta$；
- **高斯**：$r(t)$ 正态分布，平稳分布 $N(\alpha, \sigma^2/2\theta)$；
- **致命缺陷：可以取负**——高斯分布允许 $r < 0$，现实中负利率罕见（但近年日本/欧洲确实出现过，反而成了它的「优点」）。<span class="marginnote">Vasicek 的优点与缺点：<strong>高斯可解（闭式债券价）、可负利率（近年反而写实）；缺点是无界性让「利率不能太负」等常识失效</strong>。它在教材里常被用作「均值回复 + 可解」的示范模型。</span>

**辨析｜易错点：** 别把「均值回复速度 $\theta$」与「长期均值 $\alpha$」混淆。$\theta$ 大 = 回复快（利率被拉回靶心的速度），$\alpha$ 是靶心本身——两者独立地由数据估计。**「回复得快」绝不意味着「回归到高」**，这是均值回复模型初学者的第一坑。

## 2 CIR 模型

**CIR**：$dr = \theta(\alpha - r)dt + \sigma\sqrt r\, dB$。

**非负性（Feller 条件）**：若 $2\theta\alpha \ge \sigma^2$，则 $r(t) > 0$ 几乎必然——**利率不会触碰 0**。$\sqrt r$ 扩散的妙处：利率越接近 0，波动 $\sigma\sqrt r$ 越小，把利率「推开」——**非线性波动保护了非负性**。

**平稳分布**：$r(t)$ 渐近服从**非中心卡方/伽马**分布：
$$
r_\infty \sim \mathrm{Gamma}\Big( \frac{2\theta\alpha}{\sigma^2},\; \frac{2\theta}{\sigma^2} \Big),
$$
均值 $\alpha$，方差 $\alpha\sigma^2/(2\theta)$。<span class="marginnote">CIR 是「Vasicek 的正值修正」：<strong>$\sqrt r$ 扩散让利率永不触零（Feller 条件），且平稳分布从高斯换成伽马</strong>。代价是转移密度复杂（非中心卡方），但债券价格仍有闭式——所以 CIR 是利率建模的标准选择。</span>

**数值例：Feller 条件的量级感。** 设 $\theta = 0.5$、$\alpha = 0.04$（长期利率 4%）、$\sigma = 0.15$，则 $2\theta\alpha = 0.04 > \sigma^2 = 0.0225$——条件宽裕满足，利率被强力推离 0，触零几乎不可能。若把 $\sigma$ 提到 $0.25$，$\sigma^2 = 0.0625 > 0.04$，条件破坏，利率可能触碰甚至越过 0——**波动率大到一定程度，连「非负」这个最基本的性质都保不住**。所以 $\sigma$ 不是随便调的：它直接决定模型是否「写实」。

## 3 公式解析：CIR 的 Feller 条件

**目标：理解「$2\theta\alpha \ge \sigma^2$ 保证利率非负」的机制——扩散在 0 处「推开」利率。**

第一步，写 CIR 在 $r = 0$ 附近的行为。$r$ 小时，漂移 $\theta(\alpha - r) \approx \theta\alpha$（正），扩散 $\sigma\sqrt r \approx 0$——**漂移把利率往上推，扩散几乎不拉它下来**。

第二步，比较漂移与扩散的边界行为。扩散项 $\sigma\sqrt r$ 在 0 处的「强度」与漂移项 $\theta\alpha$ 竞争。直观：$r$ 靠近 0 时，漂移是 $O(1)$ 量级、扩散是 $O(\sqrt r)$ 量级——**$\sqrt r$ 比线性更快消失**，所以漂移「赢」了，利率被推开。

第三步，Feller 条件的形式。严格判据 $2\theta\alpha \ge \sigma^2$ 来自 Feller 边界分类：当漂移系数足够强（$\theta\alpha$ 大）或扩散足够弱（$\sigma$ 小）时，0 是「自然边界」（不可达）。

第四步，直觉核对。$\sigma$ 固定时，$\alpha$ 或 $\theta$ 越大，$2\theta\alpha \ge \sigma^2$ 越容易满足——**长期均值越高或回复越快，利率越不可能触零**。

**这个推导为什么重要**：Feller 条件示范了「SDE 的边界行为」分析——**扩散在边界处如何决定「可达性」**。这种「漂移 vs 扩散在边界竞争」的分析，在金融、生态、排队模型里反复出现。

## 4 债券定价：从利率到价格

**零息债券**：到期日 $T$ 支付 1 元，当前价格 $P(t, T)$。在风险中性测度下（利率模型已经写成 $\mathbb{Q}$ 形式）：
$$
P(t, T) = E_{\mathbb{Q}}\Big[ e^{-\int_t^T r(u)du} \mid \mathcal{F}_t \Big].
$$
对 Vasicek 与 CIR，这个期望有**闭式**：
$$
P(t, T) = e^{A(t,T) - B(t,T)\, r(t)},
$$
其中 $A, B$ 是确定的函数（由模型参数决定）。**债券价格 = 「利率的负指数积分」在风险中性测度下的期望**——利率模型的一切用途最终都落到这个期望上。<span class="marginnote">「期限结构」（收益率曲线）就是从债券价格反解出的利率曲线：<strong>$P(t,T)$ 给出 $T$ 期限的即期收益率</strong>。利率模型的检验标准就是「能否拟合真实收益率曲线」——Vasicek/CIR 都能闭式拟合，所以它们长盛不衰。</span>

**数值例：价格对利率的敏感度（久期的雏形）。** 对 Vasicek，$B(t,T) = \frac{1}{\theta}\big(1 - e^{-\theta(T-t)}\big)$ 恰好是「价格对 $r$ 的敏感度」：$\partial \ln P / \partial r = -B$。它只依赖 $\theta$ 与剩余期限，与当前利率 $r$ 无关——这正是「久期不随利率水平变化」的高斯模型特性。取 $\theta = 0.3$、$T - t = 5$ 年：$B = \frac{1 - e^{-1.5}}{0.3} \approx 2.59$——利率上升 1%（0.01），债券价格大约下降 2.6%。**期限越长、$\theta$ 越小，$B$ 越大，价格对利率越敏感**——这正是债券久期管理的核心直觉，也解释了为什么均值回复参数 $\theta$ 会直接进入交易决策。

## 5 两个模型的对照

## 5 两个模型的对照

| 维度 | Vasicek | CIR |
| --- | --- | --- |
| 扩散项 | $\sigma$ 常数 | $\sigma\sqrt r$ |
| 利率符号 | 可负 | 非负（Feller） |
| 分布 | 高斯 | 伽马/非中心卡方 |
| 平稳分布 | $N(\alpha, \sigma^2/2\theta)$ | $\mathrm{Gamma}$ |
| 债券闭式 | ✅ | ✅ |
| 关系 | OU | OU + $\sqrt r$ |

**选择指南**：要可解+教学 → Vasicek；要非负+真实 → CIR。

**选择不只是学术偏好**：银行做利率敏感性测试时，Vasicek 便于解析推导（高斯线性），CIR 便于符合「利率非负」的监管直觉，尤其在低利率环境下——**Feller 条件近乎自动满足，正利率的承诺让 CIR 成为风险管理（如 Solvency II 的压力测试）的首选**。而在教学与快速原型里，Vasicek 的闭式优势无可替代：公式少、直觉清、一行代码可复现。<span class="marginnote">现代利率模型的军备竞赛：<strong>Vasicek/CIR 是「单因子」模型——一个随机源驱动整条曲线</strong>。真实曲线需要多因子（LMM、HJM 框架）、随机波动（Black-Karasinski）——但单因子模型的均值回复骨架，是一切升级的底座。</span>

## 6 小结

- **Vasicek**：$dr = \theta(\alpha - r)dt + \sigma dB$——OU，高斯可解，利率可负。
- **CIR**：$dr = \theta(\alpha - r)dt + \sigma\sqrt r\, dB$——非负（Feller 条件），伽马平稳分布。
- **Feller 条件** $2\theta\alpha \ge \sigma^2$：扩散在 0 处太弱，利率被推开——边界行为分析。
- **债券价格** $P = E_{\mathbb{Q}}[e^{-\int r}] = e^{A - Br}$——闭式可求，拟合收益率曲线。
- 关系：CIR = Vasicek + $\sqrt r$ 扩散；均值回复是共同骨架。

**风险中性测度的备注**：上述债券定价公式在风险中性测度 $\mathbb{Q}$ 下成立，模型参数（如 $\theta$）在 $\mathbb{P}$ 与 $\mathbb{Q}$ 下可能不同。**从真实测度到风险中性的变换，就是「市场风险溢价」被吸收进参数的过程**——这正是 Black-Scholes 那篇与第十篇风险中性定价的衔接点，也是随机过程在金融里最深的一处结合。

**一处对照**：Vasicek 与 CIR 的差别仅在扩散项的 $\sqrt r$，却带来「可负 vs 非负」「高斯 vs 伽马」的质变——**一个平方根，改变了模型的基本性质**。这是「扩散项形状决定过程性格」最简洁的教科书案例。

在下一节，我们把随机过程用到保险精算：**保险中的随机过程：Cramér-Lundberg 破产模型**——理赔流与盈余过程。
