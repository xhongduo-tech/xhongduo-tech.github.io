---
title: SAC：软Actor-Critic
date: 2026-08-07
---

# SAC：软Actor-Critic

<div class="epigraph">
<p>最优策略不是「只有一个最优动作」，而是在「好」与「别太死板」之间找到平衡。</p>
<footer>—— 改编自图奥马斯 · 哈尔诺亚（Tuomas Haarnoja）等，2018</footer>
</div>

<div class="article-byline">
<p>第四级 · 强化学习 ｜ 深度强化学习专题 ｜ 原文：Haarnoja et al. 2018 ｜ 2026-08-07</p>
</div>

## 为什么「确定性最优」不是最优的

TD3 学的是确定性策略——「每个状态一个动作」。但真实世界充满干扰：机器人关节有噪声、观测有误差，**一个「贴着最优轨迹走」的确定性策略，稍有扰动就崩**；而一个「在好动作附近保持随机」的策略反而更稳。**SAC（Soft Actor-Critic，软演员-评论家）** 把「最大化回报」与「最大化策略熵」并进同一个目标——**最大熵强化学习**。它学到的是**随机策略**：既拿到高回报，又保留「随机性」这个鲁棒性保险。它也因此成为连续控制领域最流行、最稳的离策略算法之一。<span class="marginnote">「软（soft）」来自「软贝尔曼算子」——把「取 max」换成「取 softmax（log-sum-exp）」，让「最优价值」不再是「最好的那个动作的值」，而是「所有动作按回报指数加权后的软最大值」——熵的贡献被折进价值里。</span>

## 1 最大熵目标：回报 + 熵

SAC 的优化目标是**最大熵强化学习**：

$$
J(\pi) \;=\; \mathbb{E}\Big[\sum_{t} \gamma^t \big(R_t + \alpha\, H(\pi(\cdot \mid S_t))\big)\Big]
$$

**比普通 RL 多了一项策略熵 $H(\pi)$**——「策略在该状态下的随机程度」被当成额外的「奖励」：$\alpha$（温度系数）控制「多想要随机性」。$\alpha \to 0$ 退化为普通 RL（只要回报）；$\alpha$ 大则策略偏向「均匀随机」。**最大熵策略是「既有好回报、又不轻易把话说死」的折中**。<span class="marginnote">熵项的两个实战收益：一是<strong>天然探索</strong>——策略随机即探索，不需要 $\varepsilon$-贪心或动作噪声；二是<strong>鲁棒性</strong>——随机策略对动作噪声、模型误差不敏感，迁移到真实系统更稳。这正是 SAC 在机器人控制里特别受欢迎的原因。</span>

## 2 软价值更新：把熵折进 Q

最大熵目标下，价值函数的贝尔曼方程变成**软贝尔曼方程**。critic 的目标是：

$$
y \;=\; r + \gamma\Big[ Q'(s', a') - \alpha \log \pi(a' \mid s') \Big], \qquad a' \sim \pi(\cdot \mid s')
$$

**关键改动：目标里减去 $\alpha\log\pi(a'|s')$**——这是「策略熵」在价值层面的投影。因为 $H(\pi) = -\mathbb{E}[\log\pi]$，把「熵奖励」折进目标，等价于在价值里加「$-\alpha\log\pi$」项。**SAC 的 critic 学的是「回报 + 熵」的软价值，actor 按它改进**。<span class="marginnote">softmax 视角：软价值 $V(s) = \alpha\log\int_a \exp(Q(s,a)/\alpha)\,da$ 是「软最大」——它把「最好的动作」与「次好动作们的价值」按指数权重融合。熵项 $\alpha\log\pi$ 恰好让「价值里体现『多一个选择的价值』」——这就是「软」字的数学含义。</span>

## 3 actor 更新：重参数化，让采样可微

SAC 的 actor 输出**随机策略**（高斯），但用**重参数化技巧（reparameterization trick）**让「采样」可微——把随机性挪到外部的噪声源：

$$
a \;=\; \mu(s) + \sigma(s)\odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

于是 actor 的损失（要最大化的「软价值期望」）可以直接对参数求导：

$$
\mathcal{L}_{\pi} \;=\; \mathbb{E}_{s,\epsilon}\Big[\alpha \log \pi\big(f_\theta(s,\epsilon)\mid s\big) - Q(s, f_\theta(s,\epsilon))\Big]
$$

**对比 REINFORCE 的采样梯度**：重参数化让梯度「穿透采样」直达 actor 参数，方差远小于「score function」方法——这是 SAC 训练稳定性的关键一环。<span class="marginnote">重参数化技巧是「从生成模型（VAE）借来的」：与其对「随机动作」求梯度（方差大），不如把随机源固定为 $\epsilon$、让「动作是参数的确定性函数」——梯度变得「直来直去」。这比 REINFORCE 的「对数概率 × 优势」低方差得多。</span>

## 4 公式解析：温度 α 与自动调温

**温度系数 $\alpha$** 控制「探索/随机性」的强度。SAC 的扩展版把它**自动调节**——约束「平均熵不低于目标熵 $\mathcal{H}$」：

$$
\mathcal{L}(\alpha) \;=\; \mathbb{E}\Big[-\alpha \log\pi(a\mid s) - \alpha \bar{\mathcal{H}}\Big]
$$

梯度上升更新 $\alpha$：**当策略熵低于目标（策略太确定）→ $\alpha$ 增大（多奖励随机）；熵高于目标 → $\alpha$ 减小**。这让「探索强度」自适应地匹配任务——早期多探索、后期保持最小随机性。<span class="marginnote">自动调温解决了「$\alpha$ 怎么选」这个麻烦：把「$\alpha$ 是超参」变成「$\alpha$ 是学习目标」。这是 SAC 相比其他「熵加权」方法（如带固定熵权的变体）的实用优势——少一个要调的旋钮。</span>

**SAC 与 TD3 的对照**：

| 维度 | TD3 | SAC |
| --- | --- | --- |
| 策略 | 确定性 | 随机（高斯） |
| 目标 | 回报 | 回报 + 熵 |
| 高估处理 | 双 critic 取 min | 双 critic 取 min（同） |
| 探索 | 动作噪声 | 策略自身的随机性 |
| 鲁棒性 | 较弱（确定性易受扰） | 强（随机性兜底） |

**SAC 与 TD3 共享「双 critic + 目标网络 + 回放」的骨架**，差别在「随机 vs 确定」与「熵目标」。SAC 通常更稳、样本效率更高，代价是实现略复杂（要重参数化与温度调节）。<span class="marginnote">一个实用建议：连续控制任务上，SAC 与 TD3 谁更好取决于「随机性是否重要」——任务有扰动/部分可观测，SAC 的随机策略更优；任务几乎确定、追求极致精度，TD3 的确定性策略可能更直接。两者都值得作为默认基线。</span>

## 5 易错点辨析

**辨析｜易错点：** 把 SAC 的熵项当成「探索噪声」。熵奖励不只是「让动作随机」——它是**目标函数的一部分**，改变的是「最优策略」的定义（不再唯一取 max，而是 softmax）。SAC 学的是「软最优策略」，不是「带噪声的最优策略」——区别在「随机性被纳入价值」，而不是「训练时的抖动」。

**另一个易错点**：把「重参数化」与「REINFORCE」混用。SAC 的 actor 用重参数化（$\epsilon$ 外部化、梯度穿透采样）；REINFORCE 用 score function（$\nabla\ln\pi$）。**两者都是「对随机策略求梯度」，但方差与适用场景不同**——在连续动作 + 可微 actor 下，重参数化全面占优。

**第三个易错点**：固定 $\alpha$ 而不调温。固定小 $\alpha$ 让 SAC 退化成「几乎只要回报」——失去最大熵的意义；固定大 $\alpha$ 让策略永远很随机、不收敛。**自动调温（约束熵）是 SAC 的标配，别手动钉死 $\alpha$**（除非有明确理由）。

## 6 小结

- **SAC**：最大熵 RL——目标 = 回报 + $\alpha\times$策略熵；学「软最优」随机策略。
- **软价值目标**：$y = r + \gamma(Q'(s',a') - \alpha\log\pi(a'|s'))$——熵折进价值。
- **重参数化 actor**：$a = \mu(s) + \sigma(s)\epsilon$——采样可微、梯度低方差。
- **自动调温**：约束平均熵，$\alpha$ 自适应——少一个超参、探索自动匹配。
- 与 TD3 同骨架（双 critic + 目标网络 + 回放），差别在随机策略 + 熵目标——更稳更鲁棒。

在下一节，我们看看如何把这种训练「放大」到大规模：**IMPALA**——大规模分布式 Actor-Learner 架构，兼顾离策略数据利用与策略保真。
