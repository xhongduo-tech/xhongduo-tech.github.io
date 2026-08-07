---
title: TD3：双延迟深度确定性策略梯度
date: 2026-08-07
---

# TD3：双延迟深度确定性策略梯度

<div class="epigraph">
<p>别信一个评论家的甜言蜜语——让两个评论家互相把关，再给目标打上平滑剂。</p>
<footer>—— 改编自斯科特 · 藤本（Scott Fujimoto）等，2018</footer>
</div>

<div class="article-byline">
<p>第四级 · 强化学习 ｜ 深度强化学习专题 ｜ 原文：Fujimoto et al. 2018 ｜ 2026-08-07</p>
</div>

## 为什么 DDPG 的三个毛病要一起治

DDPG 在连续控制上开了路，但训练不稳的毛病众所周知。**TD3（Twin Delayed DDPG，双延迟深度确定性策略梯度）** 诊断出三个「病灶」，各开一味药：**critic 高估**（最大化偏差）→ 用**双 critic 取小**；**actor-critic 耦合震荡** → 用**延迟策略更新**；**Q 的尖锐尖峰被利用** → 用**目标策略平滑**（加噪声）。三味药合起来，让 TD3 在几乎所有连续控制基准上都稳定地超过 DDPG。<span class="marginnote">TD3 论文的全名把三味药写进标题：Twin（双 critic）、Delayed（延迟更新）、DDPG。它是「对 DDPG 的一次系统维修」——每一个技巧都有明确的病根对应，是「问题驱动设计算法」的教科书案例。</span>

## 1 味药一：裁剪双 Q——别被高估骗

第6章的最大化偏差与第42课的 Double DQN 都指向同一个敌人：**critic 高估**。DDPG 用单一 critic，actor 被「Q 的梯度」牵着走——critic 一旦高估某个动作，actor 就被骗往那里。

**TD3 维护两个 critic** $Q_{\boldsymbol{\theta}_1}, Q_{\boldsymbol{\theta}_2}$（独立初始化），目标取**两者取小**：

$$
y \;=\; r + \gamma\, \min_{i=1,2} Q'_{i}\big(s',\, \mu'(s')\big)
$$

**取 min 的原因**：两个 critic 的估计都带噪声，误差方向随机；取小后，**「高估的部分」被另一个 critic 的（不那么高的）估计压住**——系统性高估被抵消。<span class="marginnote">直觉：两个评论家独立打分，一个可能偏高、一个可能偏低；取 min 保守地相信「较低的那个」——宁可低估、不可高估，因为 actor 是「跟着 Q 上升走」的，低估只会让 actor 保守，高估会让 actor 被带偏。</span>

## 2 味药二：延迟更新——让 critic 先站稳

actor 与 critic 相互依赖：critic 要给 actor 指路，actor 改了、critic 的指路又过时。**两者同步更新会「互相追逐」、震荡**。TD3 的解法是**延迟（delayed）**：

- **critic 每步更新**（用回放数据拟合目标）；
- **actor 每 $d$ 步才更新一次**（如 $d=2$），且在 actor 更新前，**目标网络先同步一次**。

**逻辑**：actor 更新依赖 critic 的「指路」（$\nabla_a Q$）；让 critic 先多学几步、把 $Q$ 估得更准，actor 的「指路」才可信。**「让评论家先站稳，演员再迈步」**——这是两时标（two time-scale）思想的工程化。<span class="marginnote">延迟更新与第39课的「两时标」一脉相承：评论家（critic）以更快的时间尺度收敛，演员（actor）以较慢尺度跟随。TD3 把这个原则用「更新频率」显式实现，比「步长大小」更直接、更好调。</span>

## 3 味药三：目标策略平滑——别踩尖峰

Q 函数在动作空间里可能有**尖锐的尖峰**——某个动作的值被噪声抬高（critic 拟合出假峰）。actor 若被指到尖峰上，训练就崩。**目标策略平滑（target policy smoothing）** 在算目标时给下一动作加**裁剪后的噪声**：

$$
y \;=\; r + \gamma\, \min_{i=1,2} Q'_{i}\big(s',\, \mu'(s') + \epsilon\big), \qquad \epsilon \sim \operatorname{clip}(\mathcal{N}(0,\sigma), -c, c)
$$

噪声让「目标动作」不再是一个点，而是**一个邻域的平均**——Q 的尖峰被邻域平均「磨平」，actor 不再被单个假峰欺骗。<span class="marginnote">这味药在理论上对应「策略的平滑正则」：对目标动作加噪声等价于「用目标策略邻域的平均价值做目标」，让 critic 学到「不尖、不抖」的价值面——也顺便提高了 critic 对「actor 输出的小偏差」的鲁棒性（避免 actor 更新后目标动作突然离开 critic 熟悉的区域）。</span>

## 4 公式解析：TD3 目标的「三保险」

$$
y = r + \gamma \underbrace{\min_{i=1,2}}_{\text{① 双 critic 取小}} Q'_i\Big(s',\ \underbrace{\mu'(s') + \operatorname{clip}(\epsilon,-c,c)}_{\text{③ 目标平滑：加裁剪噪声}}\Big), \qquad \text{② actor 每 } d \text{ 步才更新}
$$

- **第一步，认取小**（①）：两个目标 critic 的值取 min——压住高估，保守目标。
- **第二步，认延迟**（②）：actor 的更新频率被降低，critic 先收敛——减少 actor-critic 耦合震荡。
- **第三步，认平滑**（③）：目标动作加裁剪噪声 $\epsilon$——Q 的尖峰被邻域平均磨平，critic 学「不尖」的价值面。**三味药各打一个病灶，合起来让连续控制的离策略训练既稳又准**。<span class="marginnote">TD3 的「三保险」互相独立、可单独拆用：只加双 critic 就够压高估；只延迟就能稳训练；只平滑就能防尖峰。它们组合时效果叠加，这也是为什么 TD3 几乎成为「离策略连续控制」的默认基线——SAC 则是它的「随机策略」表亲。</span>

## 5 易错点辨析

**辨析｜易错点：** 以为 TD3 的「双 critic」是「两个网络取平均」。**是取 min，不是取平均**——取 min 专门压「高估」（保守方向），取平均只是降方差、不防系统性高估。方向错了，药就白吃。

**另一个易错点**：把「目标平滑噪声」与「探索噪声」混为一谈。探索噪声加在**在线动作**上（采集数据时，训练用）；目标平滑噪声加在**目标动作**上（算 target 时，且被裁剪）。两者作用位置与目的都不同——一个是探索、一个是正则。

**第三个易错点**：忽略「actor 更新时目标网络同步」的顺序。TD3 的延迟更新里，**actor 每次更新前要先把目标网络同步到在线参数**——否则 actor 用的是「过期更久的目标」。这个「同步时机」是实现里最常见的 bug。

## 6 小结

- **TD3 三味药**：双 critic 取 min（压高估）、延迟 actor 更新（稳耦合）、目标平滑噪声（防尖峰）。
- **目标**：$y = r + \gamma\min_i Q'_i(s', \mu'(s') + \operatorname{clip}(\epsilon))$。
- 双 critic 取小是「保守」而非「平均」——方向专治高估。
- 延迟更新是「critic 先站稳、actor 再迈步」的两时标工程化。
- TD3 是离策略连续控制的默认基线；SAC 是它的随机策略升级版。

在下一节，我们把「确定性策略」升级成「最大熵随机策略」：**SAC**——软 Actor-Critic，用熵奖励换取更稳的探索与更强的鲁棒性。
