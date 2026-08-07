---
title: 风险中性定价与鞅测度初步
date: 2026-08-07
---

# 风险中性定价与鞅测度初步

<div class="epigraph">
<p>换一个概率测度，漂移便改了方向——Girsanov 让「风险」从定价中蒸发。</p>
<footer>—— 伊戈尔 · 吉尔萨诺夫（Igor Girsanov）</footer>
</div>

<div class="article-byline">
<p>第二级 · 随机过程 ｜ 张波《应用随机过程》§10.2 ｜ 2026-08-07</p>
</div>

## 定价的「测度把戏」

上一节的 Black-Scholes 里冒出一个神秘测度 $\mathbb{Q}$——在它下面，股价漂移从 $\mu$ 变成 $r$，折现价格成了鞅。这一节把这个「测度」讲透：**等价鞅测度（equivalent martingale measure, EMM）**与**Girsanov 定理**——换测度如何「重写」随机过程。

核心思想一句话：**概率测度是可以换的，只要新的测度与原测度等价（零概率事件相同）**。在合适的新测度下，带漂移的布朗运动变成标准布朗运动（或反之）——**漂移不是过程的固有属性，而是「测度依赖」的**。金融定价的整个策略，就是选一个测度让资产价格变鞅，然后期望、折现、得价。<span class="marginnote">「漂移是测度依赖的」是本专题最反直觉也最重要的思想之一：<strong>同一个股价路径，在真实测度 $\mathbb{P}$ 下有漂移 $\mu$，在风险中性测度 $\mathbb{Q}$ 下漂移 $r$——路径没变，只是「权重」变了</strong>。定价不关心真实漂移，因为可对冲的风险不应索取回报。</span>

本节目标：定义等价鞅测度、陈述 Girsanov 定理、并给出资产定价基本定理。

## 1 等价鞅测度

**等价概率测度**：两个概率测度 $\mathbb{P}$ 与 $\mathbb{Q}$ 等价，若它们有相同的零概率事件集合（记为 $\mathbb{Q} \sim \mathbb{P}$）。

**等价鞅测度（EMM）**：$\mathbb{Q}$ 是 EMM，若 $\mathbb{Q} \sim \mathbb{P}$ 且**折现资产价格是 $\mathbb{Q}$-鞅**：
$$
e^{-rt} S(t) \;\text{是 } \mathbb{Q}\text{-鞅}.
$$

**换测度的数学工具——Radon-Nikodym 导数**：$\mathbb{Q}$ 与 $\mathbb{P}$ 等价 ⟹ 存在随机变量 $L$（密度）使
$$
\frac{d\mathbb{Q}}{d\mathbb{P}} = L, \qquad E_{\mathbb{Q}}[X] = E_{\mathbb{P}}[L X].
$$
**把 $\mathbb{Q}$-期望翻译成 $\mathbb{P}$-期望，乘上密度 $L$ 即可。**<span class="marginnote">Radon-Nikodym 导数的直觉：<strong>测度 $\mathbb{Q}$ 是测度 $\mathbb{P}$ 的「加权版」——权重函数就是 $L$</strong>。期望的换算公式 $E_{\mathbb{Q}}[X] = E_{\mathbb{P}}[LX]$ 是测度变换的全部运算基础，也通向贝叶斯（后验 = 先验 × 似然）的同构。</span>

## 2 Girsanov 定理

**Girsanov 定理**：设 $B(t)$ 是 $\mathbb{P}$-标准布朗运动，$\theta(t)$ 是适应过程，定义
$$
L(t) = \exp\!\Big( -\int_0^t \theta(s)\, dB(s) - \frac12 \int_0^t \theta(s)^2\, ds \Big).
$$
若 $E[L(t)] = 1$（Novikov 条件），则存在等价测度 $\mathbb{Q}$（$d\mathbb{Q}/d\mathbb{P} = L(T)$）使得
$$
\tilde B(t) = B(t) + \int_0^t \theta(s)\, ds
$$
是 $\mathbb{Q}$-标准布朗运动。

**含义**：**加上一个漂移的布朗运动，换个测度就变回标准布朗运动**。$L(t)$ 正是第八篇的**指数鞅（随机指数）**——它是测度变换的「引擎」。<span class="marginnote">Girsanov 的用途：<strong>把「带漂移的布朗」（真实世界）重写成「无漂移的布朗」（风险中性世界）</strong>——只需选 $\theta$ 匹配风险溢价。指数鞅 $e^{-\int\theta dB - \frac12\int\theta^2}$ 的每一项都有名字：$-\int\theta dB$ 是「漂移补偿」，$-\frac12\int\theta^2$ 是「凸性修正」。</span>

**在 Black-Scholes 中**：真实测度 $\mathbb{P}$ 下 $dB$ 有漂移 $(\mu - r)/\sigma$；取 $\theta = (\mu - r)/\sigma$（夏普比率），Girsanov 给 $\mathbb{Q}$ 使 $d\tilde B = dB + \theta dt$，股价变成
$$
dS = rS dt + \sigma S\, d\tilde B.
$$
**$\mu$ 消失了——风险中性测度把它「吸收」进测度变换。**

## 3 公式解析：指数鞅是测度变换的密度

**目标：验证 $L(t)$ 满足「密度」的三个要求——非负、期望 1、可作测度变换。**

第一步，写 $L(t)$ 并确认它是鞅。$L(t) = e^{-\int_0^t \theta dB - \frac12\int_0^t\theta^2 ds}$——这正是随机指数 $\mathcal{E}(-\int\theta dB)$（第八篇），它满足 $dL = -L\theta dB$，是鞅（$E[L(t)] = L(0) = 1$，Novikov 条件保证）。

第二步，确认非负。指数恒正，$L(t) > 0$——可以作概率密度。

第三步，定义新测度。对任意事件 $A$，$\mathbb{Q}(A) = E_{\mathbb{P}}[L(T)\mathbb{1}_A]$。$\mathbb{Q}$ 是概率测度（$\mathbb{Q}(\Omega) = E[L(T)] = 1$），且 $\mathbb{Q} \sim \mathbb{P}$（$L > 0$）。

第四步，验证 $\tilde B$ 是 $\mathbb{Q}$-布朗。用 Lévy 特征：$\tilde B$ 连续、是 $\mathbb{Q}$-鞅（$d\tilde B = dB + \theta dt$，在 $\mathbb{Q}$ 下 $dB$ 的漂移被 $L$ 抵消）、二次变差 $[\tilde B] = t$——满足布朗运动的鞅特征定理，故是 $\mathbb{Q}$-布朗。**Girsanov 成立。**

**这个推导为什么重要**：它展示了「测度变换」的完整工序——**构造指数鞅、确认密度合法、用鞅特征验证新布朗**。这套工序是金融数学里「换测度」的标准动作，也是统计里「加权似然」的连续时间版。

## 4 资产定价基本定理

**基本定理（一价定律的测度版本）**：市场无套利 ⟺ 存在等价鞅测度 $\mathbb{Q}$；市场完备（所有期权可复制）⟺ $\mathbb{Q}$ 唯一。

- **无套利 ⟺ EMM 存在**：没有免费午餐，当且仅当能找到让折现价格成鞅的测度；
- **完备 ⟺ EMM 唯一**：所有或有权益都可复制，当且仅当鞅测度只有一个。

**定价流程**：① 找到 EMM $\mathbb{Q}$；② 期权价格 $V_0 = E_{\mathbb{Q}}[e^{-rT}V_T]$；③ 若市场完备，价格唯一。**Black-Scholes 市场（GBM + 可连续交易）恰好完备**——所以公式唯一。<span class="marginnote">不完备市场的含义：<strong>存在不可复制的风险（如跳跃、随机波动）时，期权价格不再是唯一——不同投资者可对同一期权出不同价</strong>。真实市场不完备，这正是「模型风险」与「对冲不完美」的数学根源。</span>

## 5 应用一瞥

- **Black-Scholes**：Girsanov 找到 $\mathbb{Q}$，定价期望可算（上一节）。
- **外汇**：两种货币的风险中性测度通过 Girsanov 互变——远期汇率定价。
- **信用风险**：违约强度模型用测度变换处理「违约事件」。
- **统计**：Radon-Nikodym 导数是似然比的连续版——「重要性采样」「贝叶斯更新」都是测度变换。

## 6 小结

- **等价鞅测度**：$\mathbb{Q} \sim \mathbb{P}$ 且折现资产是 $\mathbb{Q}$-鞅。
- **Radon-Nikodym 导数** $L = d\mathbb{Q}/d\mathbb{P}$：$E_{\mathbb{Q}}[X] = E_{\mathbb{P}}[LX]$。
- **Girsanov 定理**：$L(t) = e^{-\int\theta dB - \frac12\int\theta^2}$ 换测度 ⟹ 漂移被吸收，布朗归零。
- **资产定价基本定理**：无套利 ⟺ EMM 存在；完备 ⟺ EMM 唯一。
- 应用：Black-Scholes、外汇、信用风险、统计似然比。

在下一节，我们把利率本身变成随机过程：**随机利率模型：Vasicek 与 CIR 模型**——OU 与平方根扩散的金融身份。
