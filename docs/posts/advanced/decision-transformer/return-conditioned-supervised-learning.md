---
title: Return-conditioned 监督学习范式（行为克隆视角）
date: 2026-08-11
---

# Return-conditioned 监督学习范式（行为克隆视角）

<div class="epigraph">
<p>大道至简。</p>
<footer>—— 《道德经》（The great way is simple）</footer>
</div>

<div class="article-byline">
<p>第四级 · 高阶专题 · Decision Transformer（序列建模 RL） ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 Return-conditioned 监督学习开始

Decision Transformer 不是从天而降的孤例，它属于一个更大的家族——**Return-conditioned Supervised Learning（RvS）**。

这个家族的主张朴素到近乎挑衅：**离线强化学习，也许只需要「条件行为克隆 + 一个聪明的条件」就够了。**

2022 年 Emmons 等人的工作《RvS: What is Essential for Offline RL via Supervised Learning?》把这条思路推到极端：用最朴素的 MLP 策略、不碰 Transformer，也能在多个 D4RL 任务上追平甚至超过精心设计的离线算法。<span class="marginnote">Emmons, Eysenbach, Kostrikov, Levine, "RvS: What is Essential for Offline RL via Supervised Learning?"（ICLR 2022）。DT 正是这个家族里「用 Transformer + RTG 条件」的特例——理解了 RvS，DT 的每个零件都变成可替换的。</span>

这一课讲清楚 RvS 的定义、它与行为克隆的分界、以及它逼我们重新思考的问题：离线 RL 的「本质」到底是什么。

## 1 RvS：条件行为克隆

**Return-conditioned Supervised Learning（RvS）**：把离线 RL 表述为「在条件变量 $z$ 下，最大化数据动作的条件对数似然」的监督学习，其中 $z$ 携带期望结果的信息。

记学习目标为：

$$
\max_\theta \; \mathbb{E}_{(s, a, z) \sim \mathcal{D}} \left[ \log \pi_\theta(a \mid s, z) \right]
$$

条件 $z$ 有两种常见形态，对应 RvS 的两个实例。

**RvS-R（Return-conditioned）**：$z$ 就是返回目标（RTG），$z = \hat{R}_t$——这是 Decision Transformer 的选择。

**RvS-Go（Goal-conditioned）**：$z$ 是目标状态，$z = s_g$——当任务天然有「目标状态」的语义（如导航、摆放）时，目标条件更自然。

**关键结论**：Emmons 等人发现，目标条件在「稀疏奖励、目标语义明确」的任务上表现突出；返回条件在「稠密奖励、返回信号可靠」的任务上更稳。

**两者都没有显式优化回报——它们只做「照着条件复述数据」。** <span class="marginnote">把 DT 放回这个坐标系：DT = RvS 家族里「条件用 RTG + 结构用因果 Transformer + 决策用最近 $K$ 步」的一个具体配置。家族成员之间的差别，几乎都落在「条件选什么、结构选什么」这两个旋钮上。</span>

## 2 与行为克隆的分界：条件让它超越平均

朴素**行为克隆（Behavioral Cloning, BC）**学的是 $\max_\theta \mathbb{E}[\log \pi_\theta(a \mid s)]$——没有条件，模型只能拟合「数据的平均行为」。

BC 的软肋众所周知：数据里如果混合了平庸与高手的轨迹，BC 只能学出一个「平均水平的模仿者」。

RvS 用**条件**把 BC 升级了：

**BC 输出单一策略**；RvS 输出「一族策略」，由 $z$ 挑选。
- BC 无法区分「这条轨迹平庸、那条是高手」；RvS 用 $z$ 把高手与平庸的轨迹分开——高 $z$ 段学高手的动作，低 $z$ 段学平庸的动作。
- **推理时只需把 $z$ 拧到高位，就「在条件分布里选中了高手」**——这等于用条件密度估计实现了「选择超越平均的轨迹」，而不需要任何价值函数。

**辨析｜易错点：** 常有人说 RvS/DT「就是行为克隆」。

更准确的说法是：**RvS 是「条件化的行为克隆」，而「条件」是它从离线数据中提炼出优化信号的全部机制。**

没有条件的 BC 只能复述平均；有了条件的 BC 能按目标检索风格。

两者的关系不是「等价」，而是「BC 是 RvS 在条件恒定时（$z$ 与 $s$ 无关）的特例」。

## 3 简单性的辩护：What is essential?

Emmons 等人的论文标题直接问：「离线 RL 的监督学习中，什么是本质的？」他们的回答近乎激进。

1. **不需要价值函数**：RvS 与最强基线的差距在多数任务上可接受，而实现与调试成本低一个量级。
2. **不需要复杂的架构**：论文默认用 MLP，LSTM 可选；Transformer 不是必需品。
3. **超参极少**：没有折扣因子、没有 $\alpha$ 正则权重、没有 expectile 参数。
4. **稳定性好**：没有价值发散风险，训练曲线平稳。

这逼出一个更深的问题：**过去离线 RL 的复杂设计（保守化、约束、分位数），有多少是「问题必需的」，有多少是「价值自举这条路自带的包袱」？**

RvS 的实验暗示：当你绕过价值自举，很多包袱可以直接扔掉。<span class="marginnote">这是本专题反复出现的主题，也呼应博客主线《从极限到大模型》里对「简单有效」的追问——第三级《大模型原理》中「预训练 + 提示」的极简叙事，与 RvS 的「条件 + 监督」如出一辙。</span>

当然，简单性有它的代价，RvS 论文自己也承认：在部分任务上它的成绩仍低于精心调参的价值方法；它对条件的质量（返回信号是否可靠、目标语义是否清晰）非常敏感——**它把「优化」交给了「条件」，条件不给力，算法就不给力。**

## 4 RvS 的三个实验发现

Emmons 等人的实验有几条具体发现，值得逐条记下。

1. **架构几乎不重要**：在多数任务上，MLP 策略与 LSTM 策略性能相当——**序列记忆不是 RvS 取得成绩的必要条件**，条件才是。这一点直接冲击了「Transformer 是 DT 成功关键」的直觉。
2. **目标条件优于返回条件（在目标明确的任务上）**：在稀疏奖励、目标语义清晰的任务上，RvS-Go 显著优于 RvS-R；反过来在稠密奖励任务上 RvS-R 更稳。**「条件该选什么」是任务相关的设计决策**，不是通用默认。
3. **RTG 的上限由数据决定**：RvS-R 在高 RTG 条件下学到的策略，不会超过「数据里真正高返回轨迹」的质量——它的天花板是数据的，不是模型的。这条发现与《序列建模 RL 局限性》的「条件依赖」警告互为因果。<span class="marginnote">三连起来看，RvS 的讯息是：<strong>离线 RL 的监督学习路线，瓶颈不在模型，在「条件」与「数据」</strong>。这为后文评估 DT 的真正卖点（Transformer 到底贡献了什么）提供了标尺。</span>

## 5 公式解析：条件似然为什么能「挑选高手」

把 RvS 的目标函数拆开，看「挑选高手」的机制藏在哪个符号里：

$$
\max_\theta \; \mathbb{E}_{(s, a, z) \sim \mathcal{D}} \left[ \log \pi_\theta(a \mid s, z) \right] = \max_\theta \; \mathbb{E}_{s, z} \left[ \underbrace{\mathbb{E}_{a \sim \pi_{\mathcal{D}}(\cdot | s, z)} \left[ \log \pi_\theta(a \mid s, z) \right]}_{\text{对给定 (s,z) 的条件动作分布做 MLE}} \right]
$$

三步拆解。

**第一步，看 $z$ 的位置**：$z$ 出现在条件侧，不在被预测侧——它约束的是「在哪种意图下说话」，这正是「一族策略」的数学形态：$\pi_\theta(\cdot \mid s, z)$ 是 $z$ 的连续族。

**第二步，看期望内层**：对固定的 $(s, z)$，内层期望是在「数据里恰好拥有该 $(s, z)$ 组合的轨迹」上取平均——**如果数据里 $z$ 高的片段主要是高手动作，那么这个条件均值就被高手主导**。

条件变量把「数据的质量分层」翻译成了「概率分布的分层」。

**第三步，看推理**：$\pi_\theta(\cdot \mid s, z_{\text{高}})$ 就是「在高手行为上拟合出的条件分布」——采样它，约等于「请出数据里的高手来决策」。

**所谓「从离线数据中提炼超越平均的策略」，数学上就是「把条件拧到数据的高分位段，再条件采样」。**

这一句，是 RvS 全部朴素力量的来源。

## 6 小结

- **RvS** 把离线 RL 写成条件似然 $\log \pi_\theta(a \mid s, z)$；$z$ 取 RTG（RvS-R）或目标状态（RvS-Go）。
- **与 BC 的分界**：BC 是 $z$ 恒定的退化情形；RvS 靠条件把「高手段」与「平庸段」在概率上分开。
- **简单性主张**：无需价值函数、无需复杂架构、超参极少、训练稳定。
- **三条实验发现**：架构不重要、条件类型是任务相关的设计、RTG 上限由数据决定。
- **本质答案**：条件采样 = 条件化的行为克隆，用「拧条件」实现「选高手」。
- **代价**：条件质量决定上限，稀疏奖励或返回噪声大时乏力。

在下一节，我们见识 RvS 家族的另一种极端形态——用 Transformer 把整条轨迹都当作要预测的序列，这就是 **Trajectory Transformer**。
