---
title: PPO：近端策略优化
date: 2026-08-07
---

# PPO：近端策略优化

<div class="epigraph">
<p>与其小心翼翼地解一个带约束的难题，不如把约束「烙」进目标里，一步到位。</p>
<footer>—— 改编自约翰 · 舒尔曼（John Schulman）等，2017</footer>
</div>

<div class="article-byline">
<p>第四级 · 强化学习 ｜ 深度强化学习专题 ｜ 原文：Schulman et al. 2017 ｜ 2026-08-07</p>
</div>

## 为什么 TRPO 的「严格约束」可以简化

TRPO 用「KL 约束 + 自然梯度 + 共轭梯度 + 线搜索」保证每次更新不越界——理论漂亮，但实现复杂、每步要解二阶系统，在大模型上尤其笨重。**PPO（Proximal Policy Optimization，近端策略优化）** 用一个**裁剪（clip）**的代理目标，把「约束」直接写进损失函数：**当新旧策略的比值 $\rho$ 偏离 1 太远时，目标被「剪断」，梯度不再鼓励继续偏离**。这一招把「带约束优化」变成「一次反向传播」，简单、稳、适配各种网络，成了**当今深度 RL 与 RLHF（大模型对齐）的默认策略优化器**。<span class="marginnote">PPO 论文的核心主张：「在数据效率与训练稳定性上，PPO 达到或超过 TRPO，但实现只需要一阶优化。」这个「一阶就能达到二阶效果」的简化，让它在工程上迅速取代 TRPO，也成为后来 RLHF 里 KL 约束的前身。</span>

## 1 裁剪代理目标：把「别走太远」写进损失

PPO 的损失基于 TRPO 的代理目标 $L = \mathbb{E}[\rho_t \hat A_t]$（$\rho_t = \pi_\theta(a|s)/\pi_{\theta_{\text{old}}}(a|s)$），但加了裁剪：

$$
L^{\text{CLIP}}(\boldsymbol{\theta}) \;=\; \mathbb{E}\Big[\min\big(\rho_t \hat A_t,\; \operatorname{clip}(\rho_t,\, 1-\varepsilon,\, 1+\varepsilon)\, \hat A_t\big)\Big]
$$

$\operatorname{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon)$ 把 $\rho$ 限制在 $[1-\varepsilon, 1+\varepsilon]$ 内（$\varepsilon$ 通常 0.2）。**$L^{\text{CLIP}}$ 是「未裁剪目标」与「裁剪后目标」的逐点取小**——它只在「$\rho$ 朝偏离 1 的方向改进」时起作用：如果 $\rho$ 已经越界，裁剪项把目标钉住、梯度归零，**策略不再被推着继续偏离**。<span class="marginnote">取 min 的细节很关键：当 $\hat A > 0$（好动作），$L$ 想增大 $\rho$（多选这个动作）；$\rho$ 一旦超过 $1+\varepsilon$，裁剪项 $\rho_{\text{clip}}\hat A$ 固定、min 取到它——梯度不再鼓励「多过头」。当 $\hat A < 0$（坏动作），对称地限制 $\rho$ 不能低于 $1-\varepsilon$。<strong>「好动作也别太贪、坏动作也别太绝」</strong>——这就是近端（proximal）的含义。</span>

## 2 PPO 的整体结构：on-policy + GAE + 裁剪

PPO 是**纯 on-policy** 的 AC 方法，循环分两步：

1. **采集**：用当前策略 $\pi_{\theta_{\text{old}}}$ 跑一批轨迹，用 **GAE** 算优势 $\hat A$。
2. **优化**：在**同一批数据**上，用裁剪损失做**多个 epoch** 的梯度上升（一次采集、多次利用），更新 $\theta$ 后丢弃数据、重新采集。

**为什么「多个 epoch」不违反 on-policy？** 因为裁剪目标里的 $\rho = \pi_\theta/\pi_{\theta_{\text{old}}}$ 用**旧策略做分母**——多个 epoch 都在「旧策略这批数据」上优化，$\rho$ 的偏离被裁剪限制住，不会把策略推离旧数据太远。**裁剪既是「复用数据」的许可，又是「不过度偏离」的闸门**。<span class="marginnote">对比 A2C：A2C 每批数据只更新一次；PPO 用裁剪允许同一批数据被「近似离策略」地多次利用——这让 PPO 的样本效率优于 A2C。裁剪半径 $\varepsilon$ 与 epoch 数是配套的超参：$\varepsilon$ 越小、epoch 越多，越接近「严格 on-policy」。</span>

**PPO 的完整损失**通常是裁剪目标 + 价值损失 + 熵奖励：

$$
L^{\text{PPO}} \;=\; \mathbb{E}\big[L^{\text{CLIP}} - c_1\,(V(s) - R^{\text{target}})^2 + c_2\, H(\pi)\big]
$$

## 3 为什么 PPO 这么好用：稳定 + 通用 + 简单

PPO 成为事实标准，靠的是三条「工程友好」特质：

1. **稳定**：裁剪限制了单次更新的「策略距离」，不会像普通策略梯度那样「一步崩盘」——对超参（学习率、batch）的敏感度大幅降低。
2. **通用**：一个实现（AC + GAE + 裁剪）横跨连续控制（MuJoCo）、游戏（Atari）、再到**大模型对齐（RLHF）**——裁剪目标对「策略分布的更新幅度」的软约束，在不同领域都管用。
3. **简单**：没有二阶优化、没有线搜索，就是「标准反向传播」——任何深度学习框架几小时就能写对。<span class="marginnote">PPO 的「通用性」在 RLHF 里尤其被放大：大模型作为策略、人类偏好作为奖励，PPO 的裁剪（配合 KL 惩罚）防止模型被「奖励黑客」一下推离自然语言分布——这份「防止过冲」的能力正是它成为 LLM 对齐标配的原因（详见本专题第19篇）。</span>

## 4 公式解析：min 如何实现「近端」

$$
L^{\text{CLIP}} = \mathbb{E}\Big[\underbrace{\min\big(\rho \hat A,\; \rho_{\text{clip}} \hat A\big)}_{\text{裁剪闸门}} \Big], \qquad \rho_{\text{clip}} = \operatorname{clip}(\rho,\,1-\varepsilon,\,1+\varepsilon)
$$

- **第一步，认 $\rho$**：$\rho = \pi_\theta/\pi_{\text{old}}$ 度量「新策略比旧策略更常/更少选这个动作」。$\rho=1$ 是新旧相同，$\rho>1$ 是新策略更偏爱。
- **第二步，认裁剪**：当 $\rho$ 落在 $[1-\varepsilon, 1+\varepsilon]$ 内，$\rho_{\text{clip}}=\rho$，min 取 $\rho\hat A$（正常梯度）；当 $\rho$ 越界，$\rho_{\text{clip}}$ 是边界值，min 取它——**越界部分的梯度为 0**。
- **第三步，认方向性**：min 只裁剪「对优化不利」的越界：好动作（$\hat A>0$）的 $\rho$ 不能无限涨、坏动作（$\hat A<0$）的 $\rho$ 不能无限跌。**策略被允许「温和改进」，不允许「激进跳变」**——这就是「近端」的精确机制。<span class="marginnote">对比 TRPO：TRPO 用显式的 KL 约束（硬边界）保证「每次更新 KL ≤ δ」；PPO 用裁剪（软边界）只对「越界的梯度」做惩罚。硬约束保证每一次都合规，软约束更简单但可能「踩线」——实践中 PPO 的简单性通常更值钱。</span>

## 5 易错点辨析

**辨析｜易错点：** 以为 PPO 是离策略方法（因为数据被多 epoch 复用）。PPO 是**近 on-policy**：它在「旧策略采集的一批数据」上做局部优化，靠裁剪控制「复用导致的偏离」。它不是「用历史任意数据学新策略」的通用离策略（那是 DQN 的领域）。**「近似离策略地复用一批数据」≠「离策略学习」**。

**另一个易错点**：把裁剪系数 $\varepsilon$ 当「学习率」。$\varepsilon$ 是**策略更新的信任半径**（允许 $\rho$ 偏离 1 的范围），学习率是参数的步长——两个维度都影响更新大小，但语义不同。$\varepsilon$ 太大裁剪失效（≈ 普通策略梯度），太小更新寸步难行。

**第三个易错点**：忽略价值损失与熵项的平衡。PPO 的裁剪只管 actor；critic 的价值损失与熵奖励仍需调系数。实践中「价值损失系数 $c_1$ 过大」会让训练被 critic 主导，「熵系数 $c_2$ 过大」会阻止收敛——**PPO 的稳定性是「裁剪 + 价值 + 熵」三件套的共同结果**。

## 6 小结

- **PPO**：裁剪代理目标 $L^{\text{CLIP}} = \mathbb{E}[\min(\rho\hat A, \operatorname{clip}(\rho)\hat A)]$——一阶优化实现 TRPO 的约束效果。
- **近端机制**：好动作不贪、坏动作不绝——$\rho$ 越界时梯度归零。
- **结构**：on-policy 采集 + GAE 优势 + 多 epoch 裁剪优化 + 价值/熵正则。
- **优点**：稳定、通用、简单——控制、游戏、大模型对齐通吃。
- PPO 是 RLHF 的默认策略优化器——第19篇会回到它。

在下一节，我们转向连续控制的价值方法：**DDPG**——确定性策略梯度，让「动作连续」也能用 Q 学习式的方法。
