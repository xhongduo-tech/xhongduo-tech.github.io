---
title: Black-Scholes 模型与应用
date: 2026-08-11
---

# Black-Scholes 模型与应用

<div class="epigraph">
<p>标的的价格可以被对冲掉——这是整个衍生品理论的根基：不确定性可以被复制，而不必被猜中。</p>
<footer>—— 菲舍尔 · 布莱克 与 迈伦 · 斯科尔斯（Fischer Black &amp; Myron Scholes）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 随机分析（Itô 微积分） ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 Black-Scholes 开始

随机分析至此终于走到了它最著名的应用现场。1997 年诺贝尔经济学奖所表彰的 **Black-Scholes 公式**，把期权定价从「经验与猜测」变成「可推导的数学定理」。它之所以值得单开一节，不是因为金融有多迷人，而是因为它把前面所有工具——布朗运动、Itô 积分、Itô 公式、Girsanov 测度变换、Feynman-Kac——**串成了一条完整的推理链**。

这堂课的戏剧性在于一个完全反直觉的结论：**期权价格不依赖股票的期望收益率 $\mu$**。为什么「多大概率会涨」不影响「涨了能赚多少钱」的定价？答案藏在两个字里：**对冲**。<span class="marginnote">对标 Øksendal 第十二章。1960 年代 Samuelson 与 Bachelier 的早期尝试都因 $\mu$ 残留而失败；Black–Scholes–Merton 在 1973 年的突破正是引入了<strong>无风险对冲 + 测度变换</strong>。</span>

## 1 市场模型：两只资产，一种风险

**风险资产（股票）**：几何布朗运动 $dS_t = \mu S_t\,dt + \sigma S_t\,dB_t$。

**无风险资产（货币市场）**：$dB_t^0 = r B_t^0\,dt$，即 $B_t^0 = e^{rt}$。

**重点：$\sigma$ 是波动率（客观，来自二次变差），$\mu$ 是预期收益（主观，来自测度）。** 在风险中性测度 $Q$ 下，$\mu$ 会被替换成 $r$——这正是 Girsanov 的工作，稍后细看。<span class="marginnote">投资组合 $V_t = \alpha_t S_t + \beta_t B_t^0$ 称为<strong>自融资</strong>（self-financing），如果资金只在组合内部流动，不注入不抽走：$dV_t = \alpha_t\,dS_t + \beta_t\,dB_t^0$。<strong>「复制」就是找一个自融资组合使 $V_T$ 恒等于期权收益。</strong></span>

## 2 对冲论证：漂移被消除的瞬间

设期权价值 $V(t, S_t) = f(t, S_t)$，用 Itô 公式展开（$S$ 的二次变差 $d\langle S\rangle = \sigma^2 S^2\,dt$）：

$$df = \Big(f_t + \mu S f_x + \frac12 \sigma^2 S^2 f_{xx}\Big)dt + \sigma S f_x\,dB.$$

若持有 $\Delta = f_x$ 份股票对冲（即组合 $V = f - f_x S$），则

$$dV = \Big(f_t + \frac12 \sigma^2 S^2 f_{xx}\Big)dt.$$

**重点：$\mu$ 消失了！** 因为对冲组合里股票项 $f_x\,dS$ 的漂移恰好与 $f$ 中 $\mu$ 项相抵。剩下的式子不含 $\mu$，只含 $r$ 与 $\sigma$——这正说明：**在一个可完美对冲的市场里，定价不取决于资产会涨多少，只取决于波动的结构与无风险利率**。<span class="marginnote">类比：<strong>保险定价取决于损失分布与无风险利率，而不取决于「保险公司赌标的会涨」</strong>。对冲把「预测未来」从定价里删除了。</span>

再由无套利：$dV = r V\,dt$（对冲组合只能获得无风险收益），推出 **Black-Scholes PDE**：

$$f_t + rS f_x + \frac12 \sigma^2 S^2 f_{xx} = r f.$$

## 3 风险中性测度：Girsanov 的第二条路

同样的 PDE 也可以用测度变换「免写偏导」地得到。取 $\theta = (\mu - r)/\sigma$，由 Girsanov：

$$\widetilde B_t = B_t + \frac{\mu - r}{\sigma} t \quad \text{是 } Q \text{ 下的标准布朗运动。}$$

代入股票 SDE：$dS_t = \sigma S_t\,d\widetilde B_t$——**在 $Q$ 下股票恰好是无漂移的鞅**（更精确地说，折现后的 $e^{-rt}S_t$ 是 $Q$-鞅）。于是由 Feynman-Kac，期权价格 = 风险中性折现期望：

$$V(t, S_t) = e^{-r(T-t)} E_Q\big[g(S_T) \;\big|\; \mathcal{F}_t\big],$$

其中 $g$ 是期权在到期的收益函数（如看涨期权 $g(S) = (S - K)^+$）。<span class="marginnote">这是 Girsanov 与 Feynman-Kac 的联姻：<strong>Girsanov 负责把 $\mu$ 换成 $r$，Feynman-Kac 负责把 PDE 换成期望</strong>。对欧式期权，$S_T$ 在 $Q$ 下的分布显式已知（对数正态），期望能直接算出来。</span>

## 4 公式解析：Black-Scholes 看涨期权公式

对看涨期权 $g(S_T) = (S_T - K)^+$，代入上述期望并利用 $S_T = S_t e^{(r - \sigma^2/2)(T-t) + \sigma\sqrt{T-t}\,Z}$，$Z\sim\mathcal{N}(0,1)$，可得

$$C = S_t\, \Phi(d_1) - K e^{-r(T-t)}\, \Phi(d_2),$$

$$d_1 = \frac{\ln(S_t/K) + (r + \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}, \qquad d_2 = d_1 - \sigma\sqrt{T-t},$$

其中 $\Phi$ 是标准正态分布函数。逐项拆解：

- **第一项 $S_t \Phi(d_1)$**：股票端价值。$\Phi(d_1)$ 可解释为**风险中性下 $S_T > K$（期权终将行权）的「含权概率」乘以 Delta 加权**——它恰好等于复制组合里所需的股票份额 $\Delta = \partial C / \partial S = \Phi(d_1)$。
- **第二项 $K e^{-r(T-t)}\Phi(d_2)$**：现金端价值。$\Phi(d_2)$ 是风险中性下**行权概率** $Q(S_T > K)$；把它折现回现在，就是到期时需支付的执行价 $K$ 的现值。
- **整体直觉**：$C = (\text{行权时拿到的股票现值}) - (\text{需支付执行价的现值})$，两项分别用「含权概率」与「行权概率」打折——公式正是**期望收益在风险中性世界里的显式积分**。

**辨析｜易错点：** 公式里的概率是**风险中性测度下的**，不是真实世界的。$\Phi(d_2) = Q(S_T > K)$ 而**不是** $P(S_T > K)$——两者可以相差很远。这正是上一节强调的「概率是测度的观点」在金融里的直接后果。<span class="marginnote">工程上，$d_1, d_2$ 的 $\sigma^2/2$ 项又是 Itô 凸性修正的化身——<strong>同一个 $- \sigma^2/2$，在解的对数里出现一次，在公式里再出现一次</strong>。</span>

## 5 应用与局限：从公式到市场

**Delta 对冲**：$\Delta = \Phi(d_1)$ 给出对冲比例，连续调整即可复制期权——这是公式最直接的工程产出。
- **隐含波动率**：把市场期权价格代回公式反解 $\sigma$，得到「市场先生」的波动率预期，衍生品市场的温度计。<span class="marginnote">隐含波动率曲面偏离常数假设，正说明真实市场的 $\sigma$ 并非固定——<strong>BS 模型是「第一个近似」，不是「最后真相」</strong>。</span>
- **局限**：常数波动率、常数利率、无交易成本、资产可连续对冲——这些假设在极端行情（如跳空、闪崩）下全部失守，催生了随机波动率模型（Heston）、跳扩散模型等现代改进。

**重点：Black-Scholes 公式的伟大不在于它是「正确的现实」，而在于它提供了一个可证伪的基准**——后来的全部模型，都是在同一个「无套利 + 复制」框架下向现实逼近。

## 6 例：看跌期权、平价关系与 Greeks

由恒等式 $S_T - K = (S_T - K)^+ - (K - S_T)^+$，代入风险中性期望立即得到 **看跌–看涨平价**：

$$C - P = S_t - K e^{-r(T-t)}.$$

对账：左边是「买看涨、卖看跌」，到期时无论 $S_T$ 如何都恰好兑现 $S_T - K$；右边是「现在买股票、卖出无风险债券」。两边现金流完全相同，只能同价——**平价关系不依赖任何分布假设，纯粹是无套利 + 复制**。

**Greeks 由公式直接求偏导得到：**

- **Delta**：$\Delta = \partial C/\partial S = \Phi(d_1)$——对冲比例，正是鞅表示定理里那个「噪声暴露系数」；
- **Gamma**：$\Gamma = \partial^2 C/\partial S^2 = \phi(d_1)/(S\sigma\sqrt{T-t})$——Delta 的变化速度，衡量凸性暴露；
- **Vega**：$\mathcal{V} = \partial C/\partial\sigma = S\sqrt{T-t}\,\phi(d_1)$——对波动率的敏感度，波动率交易的主角。

**重点：Vega 是波动率敏感度，它提醒我们——期权本质上是「波动率的杠杆」。** Black-Scholes 世界把波动率假设为常数，Vega 只能用来读市场（隐含波动率），不能用来对冲模型误差；真实市场的波动率曲面（vol smile）正是对「常数 $\sigma$」假设的直接反驳。

**一个数字直觉**：取 $S = 100, K = 100, r = 0.05, T = 1, \sigma = 0.2$，可得 $d_1 \approx 0.25$、$d_2 \approx 0.05$，于是 $C \approx 100 \times 0.599 - 100 e^{-0.05} \times 0.520 \approx 10.45$——平价期权的价格大约是标的的 10%，且几乎全来自波动率项。试试把 $\sigma$ 改成 0：$C$ 会立即塌向 $S - Ke^{-rT} \approx 4.88$，那正是「无风险世界」里行权收益的现值。

## 7 三个视角看同一件事

**视角一（PDE）**：对冲论证推出 Black-Scholes PDE，解出 $f(t,S)$——「解析视角」。
**视角二（期望）**：Girsanov 找到风险中性测度 $Q$，$f = e^{-r(T-t)}E_Q[g(S_T)]$——「概率视角」。
**视角三（复制）**：鞅表示定理保证 $f$ 可写成一个 Itô 积分，即一个自融资组合——「可操作视角」。

三者殊途同归：**同一个价格，三种算法**。PDE 有限差分、蒙特卡洛期望、Delta 复制，分别对应这三种视角的实现。Black-Scholes 公式的伟大不在于「算出一个数」，而在于**把三个视角焊在同一个数上**——后来的随机波动率、跳扩散模型，全都在这三视角框架内升级。<span class="marginnote">对机器学习学习者：<strong>「同一个对象，多视角表示」正是表示学习的纲领</strong>；Black-Scholes 是最早的多视角理论样板之一。</span>

（尾声：读完本篇，请记住一件事——Black-Scholes 世界的全部结论都在同一个前提下成立：波动率 $\sigma$ 是常数、市场可连续交易、无摩擦。真实市场把这三点一一破坏，于是有了随机波动率、跳扩散与交易成本模型。理解基准，才谈得上偏离；看懂理想，才看得懂现实。）

（再补一句：Black-Scholes 公式的推导链——Itô 公式 → 对冲 → PDE → Feynman-Kac → 期望 → 显式解——正是本专题从第一章到本章的完整回声。）

## 8 小结

- **市场模型**：$dS = \mu S\,dt + \sigma S\,dB$ 加无风险资产；自融资组合复制收益。
- **对冲论证**：用 $\Delta = f_x$ 份股票对冲，$\mu$ 从方程中消失，得到 Black-Scholes PDE。
- **风险中性测度**：Girsanov 取 $\theta = (\mu-r)/\sigma$，$Q$ 下 $S$ 变鞅，价格 = 折现期望（Feynman-Kac）。
- **看涨公式** $C = S\Phi(d_1) - Ke^{-r(T-t)}\Phi(d_2)$：含权概率与行权概率分别给两端打折。
- **工程产出**：Delta 对冲、隐含波动率；**理论基准**：所有后续模型都在同一无套利框架下修正假设。

在下一节，我们将回头审视定价所需的那个关键性质——为什么任何可复制收益都能被「表示」出来。这就是**鞅表示定理**。
