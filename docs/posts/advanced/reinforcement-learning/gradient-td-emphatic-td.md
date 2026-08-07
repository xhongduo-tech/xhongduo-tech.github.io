---
title: 梯度TD方法与强调TD方法（Emphatic TD）
date: 2026-08-07
---

# 梯度TD方法与强调TD方法（Emphatic TD）

<div class="epigraph">
<p>当目标本身不可学习时，聪明人不是硬着头皮优化它，而是换一个学得动的目标。</p>
<footer>—— 改编自理查德 · 萨顿（Richard S. Sutton）</footer>
</div>

<div class="article-byline">
<p>第四级 · 强化学习 ｜ Sutton & Barto《强化学习（第2版）》 第11章 §11.7–11.8 ｜ 2026-08-07</p>
</div>

## 为什么要在「不可能的悬崖」边另辟蹊径

上一课证明了贝尔曼误差不可学习。这一课介绍两条**绕开悬崖**的可行路径。第一条是**梯度 TD（gradient-TD）**：放弃「不可学的 $\overline{\text{BE}}$」，改学一个**可学的变体**——投影贝尔曼误差（MSPBE），并配一组辅助参数在单样本下无偏估计它。第二条是**强调 TD（emphatic-TD）**：不去换目标，而是**重新加权更新**——给「目标策略真正会走到的状态」更高权重，把离策略分布「掰回」接近 on-policy，从而稳住半梯度。<span class="marginnote">两条路的分工很清晰：梯度 TD 修的是「目标」（学一个可学的量），强调 TD 修的是「分布」（把权重调对）。它们共享同一个野心：<strong>在离策略 + 函数逼近 + 自举三要素齐备时，让 TD 依然稳定收敛</strong>——第11章的正面答案。</span>

## 1 投影贝尔曼误差：把不可学的目标「投影」成可学的

回顾 $\overline{\text{BE}}$ 不可学的原因：它需要对所有状态的真分布取期望。**投影贝尔曼误差（projected Bellman error，MSPBE）** 做了一处关键妥协——**先把贝尔曼目标投影到「特征张成的子空间」，再量误差**：

$$
\overline{\text{PBE}}(\mathbf w) \;=\; \Big\| \Pi\, T_\pi \hat v_\mathbf w - \hat v_\mathbf w \Big\|^2_\mu
$$

其中 $T_\pi$ 是贝尔曼算子，$\Pi$ 是**到特征空间的投影算子**（把「函数」压回「$\mathbf w$ 能表达的形态」），$\|\cdot\|_\mu$ 是 on-policy 分布加权的范数。**PBE = 0 的点恰是 TD 不动点 $\mathbf w_{\text{TD}}$**——所以最小化 PBE 与半梯度 TD 的收敛目标是同一个地方，却换了一个**单样本可估计**的目标函数。<span class="marginnote">「投影」的直观：$\Pi$ 把任何函数拍扁到「线性组合 $\mathbf w^\top\mathbf x$ 能表示的面」上。PBE 度量的是「贝尔曼目标被拍扁后，离当前 $\hat v$ 还有多远」——它不要求 $\hat v$ 满足贝尔曼方程（那不可能，因为特征不够表达真值），只要求 $\hat v$ 是「离贝尔曼目标最近的可表达点」。</span>

## 2 梯度TD：一组辅助参数换一次无偏估计

**梯度 TD（gradient-TD）** 家族的关键技巧：**引入第二组参数 $\mathbf v_t$，专门估计「梯度里那个没法单样本无偏的部分」**。最著名的 **TDC（TD with gradient correction）** 两步更新：

$$
\begin{aligned}
\delta_t &= R_{t+1} + \gamma\,\mathbf w_t^\top \mathbf x_{t+1} - \mathbf w_t^\top \mathbf x_t\\
\mathbf w_{t+1} &= \mathbf w_t + \alpha\big(\delta_t \mathbf x_t - \gamma \rho_{t+1} \mathbf x_{t+1}\,\mathbf x_t^\top \mathbf v_t\big)\\
\mathbf v_{t+1} &= \mathbf v_t + \beta\big(\delta_t \mathbf x_t - \gamma \rho_{t+1} \mathbf x_{t+1}\,\mathbf x_t^\top \mathbf v_t\big)
\end{aligned}
$$

看结构：**第二项 $-\gamma\rho_{t+1}\mathbf x_{t+1}\mathbf x_t^\top\mathbf v_t$ 就是「半梯度欠的那一针」**——上一课我们说过，完整梯度需要 $\mathbb{E}[\delta\nabla\hat v(S')]$，它不能单样本无偏估计；TDC 用辅助参数 $\mathbf v_t$ **在线估计这个量**，从而给半梯度补上「梯度修正项」。$\mathbf v_t$ 自己用一个更大的步长 $\beta$ 单独学习，收敛后修正项就精确了。<span class="marginnote">数学上，$\mathbf v_t$ 收敛到「把 TD 误差投影到特征空间」的估计，$x_t^\top\mathbf v_t$ 就是那条「欠的梯度」。这让 TDC 在<strong>离策略 + 线性</strong>下证明收敛到 $\mathbf w_{\text{TD}}$——半梯度做不到的事，它用「第二组参数」做到了。GTD 是它的近亲，只多一个「半投影」的中间态，思想同源。</span>

## 3 强调TD：给「重要状态」加权重

**强调 TD（emphatic-TD）** 换了个角度：不换目标，改**更新权重**。它维护一个标量**强调系数（emphasis）** $M_t$，把 TD 更新乘上它：

$$
\mathbf w_{t+1} \;=\; \mathbf w_t + \alpha\, M_t\, \rho_t\, \delta_t\, \mathbf x_t
$$

其中强调系数按递归更新（$I_t$ 是**兴趣（interest）**，衡量「当下状态有多值得学」）：

$$
M_t \;=\; \gamma\,\lambda\, \rho_{t-1}\, M_{t-1} + I_t
$$

$M_t$ 的语义是「从目标策略的视角，这条轨迹此刻还值不值得重点学习」。**当 $I_t$ 只落在「目标策略真正会走」的路径上时，$M_t$ 给这些状态大幅加权、给离策略的枝节近乎零权重——于是有效分布被「强调」回接近 on-policy**，半梯度的不稳定性被权重抹平。<span class="marginnote">直觉类比：老师在讲台上讲重点（$I_t$ 高的状态），学生（更新）应该记重点笔记（$M_t$ 大），而不是把课上的每一句闲话都抄一遍。强调 TD 让更新「只记重点」——把离策略数据的噪声权重压下去。</span>强调 TD 同样在离策略 + 线性下可证明收敛到 TD 不动点，且实现比梯度 TD 更简单（只需维护一个标量）。<span class="marginnote">注意强调 TD 的收敛目标是 $\mathbf w_{\text{TD}}$（同一不动点），但它的「强调」会让这个不动点<strong>更偏目标策略</strong>——因为加权分布被掰正了。梯度 TD 修目标、强调 TD 修分布，两条路殊途同归于「把离策略稳定住」。</span>

## 4 公式解析：TDC 的「修正项」从哪来

$$
\underbrace{\mathbf w_{t+1} = \mathbf w_t + \alpha\delta_t \mathbf x_t}_{\text{半梯度 Sarsa 式更新}} \;\;-\;\; \alpha\gamma\rho_{t+1}\,\mathbf x_{t+1}\underbrace{\big(\mathbf x_t^\top \mathbf v_t\big)}_{\text{辅助估计}}
$$

- **第一步，认半梯度主体**：$\alpha\delta_t\mathbf x_t$ 就是第9章的半梯度更新——它把「TD 误差 × 特征」作为参数修正。
- **第二步，认修正项**：$-\alpha\gamma\rho_{t+1}\mathbf x_{t+1}(\mathbf x_t^\top\mathbf v_t)$ 补的是「上一课欠的 $-\gamma\nabla\hat v(S')$ 那一项」——但**不是用它本身**（单样本有偏），而是用辅助参数 $\mathbf v_t$ 的**在线估计** $\mathbf x_t^\top\mathbf v_t$ 来替代。
- **第三步，认两时标**：$\mathbf v_t$ 以更快的步长 $\beta$ 学习、收敛得比 $\mathbf w_t$ 快，于是当 $\mathbf w$ 稳定时，修正项已经足够精确——**「辅助估计先稳定，主参数后稳定」的两时标结构，是梯度 TD 无偏性的工程内核**。<span class="marginnote">两时标（two time-scale）思想在 RL 里反复出现：Actor-Critic（第13章）、SAC 的温度参数，都用「快的一层估计辅助量、慢的一层做主体」的结构。TDC 是它的 TD 版本。</span>

## 5 易错点辨析

**辨析｜易错点：** 把「梯度 TD」当成「对 $\overline{\text{BE}}$ 做梯度下降」。**不是**——梯度 TD 优化的是 MSPBE（投影版），而不是不可学的 $\overline{\text{BE}}$。名字里的「梯度」指「这次是真的梯度下降」，但降的是**换了目标**后的损失。记错这一点，会以为 TDC 在解贝尔曼误差。

**另一个易错点**：以为强调 TD 需要额外模型。强调系数 $M_t$ 只用 $\rho$、$\gamma$、$\lambda$、$I_t$ 这些**样本内可得**的量，不需要模型——它是纯数据驱动地「重加权」，不是「用模型重规划」。

**第三个易错点**：忽略两时标的步长条件。TDC 的收敛要求 $\beta \to 0$ 比 $\alpha$ 快（$\beta$ 是快时标）。若把 $\beta$ 调得比 $\alpha$ 还慢，修正项追不上主体，TDC 退化成带噪声的半梯度——**「辅助参数要学得更快」是梯度 TD 的硬性实现要求**。

## 6 小结

- **MSPBE**：把贝尔曼目标投影到特征空间再量误差，**单样本可学**，零点恰是 TD 不动点。
- **梯度 TD / TDC**：用第二组参数 $\mathbf v_t$ 在线估计「欠的梯度」，给半梯度补修正项；离策略+线性下收敛到 $\mathbf w_{\text{TD}}$。
- **强调 TD**：用强调系数 $M_t = \gamma\lambda\rho_{t-1}M_{t-1} + I_t$ 重加权更新，把离策略分布「掰回」on-policy。
- 两条路：**梯度 TD 修目标、强调 TD 修分布**，殊途同归于「离策略下稳定」。
- 两时标结构（$\beta$ 快于 $\alpha$）是梯度 TD 无偏性的实现关键。

在下一节，我们离开第11章的「硬核稳定性」，进入第12章——**资格迹**：用 λ 把 n步方法的所有「步数」同时装进一次更新，让 TD(λ) 在「快」与「稳」之间做到兼得。
