## 核心结论

RLHF，直译是“基于人类反馈的强化学习”，可以理解为：不用人类逐字逐句写标准答案，而是让人类告诉模型“这两个答案哪个更好”，再把这种偏好变成训练信号。工程上，RLHF 通常不是一段训练，而是三段串联：

| 阶段 | 输入数据 | 目标模型 | 产物 | 核心目的 |
| --- | --- | --- | --- | --- |
| SFT | 指令-回答示例 | 初始策略模型 $\pi_0$ | 会基本听指令的助理 | 建立语言先验和基础行为 |
| RM | 成对偏好数据 $(y^+, y^-)$ | 奖励模型 $r_\phi$ | 能给回答打分的评委 | 把“人类更喜欢哪个”压成标量 |
| PPO | prompt、策略采样答案、RM 打分 | 策略 $\pi_\theta$ 与 value | 对齐后策略 | 提高高分答案概率，同时限制漂移 |

如果只记一句话，可以记成：

$$
\text{RLHF} = \text{SFT} \rightarrow \text{RM} \rightarrow \text{PPO}
$$

以及 PPO 阶段的目标：

$$
\max_{\theta}\ \mathbb{E}_{y\sim \pi_\theta(\cdot|x)}[r_\phi(x,y)] - \beta \cdot \mathrm{KL}(\pi_\theta \| \pi_0)
$$

这里的“策略”是模型对下一个 token 或整段回答给出的概率分布，白话说就是“模型更愿意说什么”。RLHF 做的事，不是让模型学会新知识，而是重新分配“愿意说什么”的概率，让更符合人类偏好的回答更容易被采样出来。

可以用一个日常化但不失真的三步理解：

1. SFT：先把模型调成“基础助理”，至少能像样地回答问题。
2. RM：再训练一个“评委”，让它知道两个答案里哪个更好。
3. PPO：最后让助理反复出答案，由评委打分，把高分答案的概率抬高，但不能偏离原始助理太远。

---

## 问题定义与边界

RLHF 要解决的问题不是“让模型会说话”，而是“在已有会说话的语言模型上，用有限的人类偏好数据，把回答风格和行为往人类希望的方向推”。

更形式化地说，给定输入 $x$，语言模型会定义一个条件分布：

$$
\pi_\theta(y|x)
$$

其中 $\pi_\theta$ 是当前策略，$\theta$ 是模型参数，$y$ 是回答。SFT 得到一个初始策略 $\pi_0$；RM 学到一个奖励函数 $r_\phi(x,y)$；PPO 则在 $\pi_0$ 的基础上优化出新的策略 $\pi_\theta$。

核心关系是：

$$
\pi_0 \xrightarrow{\text{偏好数据}} r_\phi \xrightarrow{\text{RL优化}} \pi_\theta
$$

目标不是让 $\pi_\theta$ 无限追求高奖励，而是在高奖励与小漂移之间折中：

$$
\max_{\theta}\ \mathbb{E}[r_\phi(x,y)] \quad \text{s.t.} \quad \pi_\theta \text{ 不要离 } \pi_0 \text{ 太远}
$$

这里“不要离太远”通常用 KL 散度衡量。KL，直译是“相对熵”，白话说就是“两个概率分布差得有多大”。

边界也很明确：

| 边界因素 | 含义 | 直接后果 |
| --- | --- | --- |
| 偏好数据质量 | 人类排序是否一致、是否覆盖关键场景 | RM 学到的分数可能偏 |
| RM 容量 | 评委模型是否足够强 | 太小会被策略钻空子 |
| KL 约束强度 | 限制策略漂移的力度 | 太强学不动，太弱容易跑偏 |
| 算力预算 | 每轮 PPO 需要多模型并行前向 | 训练成本最高、最慢 |

玩具例子先说明“策略更新的对象”是什么。假设 prompt 是“解释 HTTP 404 是什么”，模型有两个候选回答：

- $y_1$：定义清楚，术语准确
- $y_2$：语言流畅，但把 404 说成“服务器崩了”

如果当前策略给它们的概率分别是：

$$
\pi_0(y_1|x)=0.3,\quad \pi_0(y_2|x)=0.1
$$

RLHF 不是直接把 $y_2$ 删除，而是通过奖励模型和 PPO，把 $y_1$ 的概率继续抬高，把 $y_2$ 压低，同时保持整体语言能力不被破坏。

真实工程边界则更硬。比如一个已经能完成通用问答的 8B 模型，如果没有 SFT 直接做 PPO，那么策略起点太差，RM 打分的梯度会非常噪，训练会花很多步在修复基础格式问题，而不是优化偏好。

---

## 核心机制与推导

### 1. SFT：先得到可用的 $\pi_0$

SFT，监督微调，就是拿“输入-高质量回答”样本做标准语言模型训练。它优化的还是交叉熵，本质上是最大化示例答案的似然。白话说：让模型先像一个合格助理。

如果没有这一步，后面的 RM 和 PPO 都会不稳定，因为策略生成的样本质量太差，偏好训练很难成立。SFT 决定的是语言先验，也就是模型默认会怎么说话、怎么组织信息、是否遵守指令格式。

### 2. RM：把排序压成标量奖励

RM，奖励模型，就是把“成对偏好”训练成一个打分器。数据通常长这样：

- 同一个 prompt $x$
- 一个优选答案 $y^+$
- 一个劣选答案 $y^-$

RM 输出两个分数：

$$
r_\phi(x, y^+), \quad r_\phi(x, y^-)
$$

训练目标通常是 Bradley-Terry 风格的排序损失：

$$
\mathcal{L}_{RM} = -\log \sigma\left(r_\phi(x, y^+) - r_\phi(x, y^-)\right)
$$

$\sigma$ 是 sigmoid，白话说是把“优答案应该比分答案高”转成一个可优化概率。

这一步很关键，因为人类通常不擅长给一个绝对分数，比如“这段回答值 7.4 分”，但能较稳定地判断“两段里哪个更好”。RM 的作用就是把成对比较压缩成一个连续标量。

### 3. PPO：在高奖励与小漂移之间做折中

PPO，近端策略优化，是一种强化学习算法。它不是简单“看到高分就加大概率”，而是通过受控更新，避免一次改太多。这里的“近端”可以理解成“不要一步走太远”。

RLHF 中常见目标可以写成：

$$
J(\theta)=\mathbb{E}_{y\sim\pi_\theta(\cdot|x)}[r_\phi(x,y)]-\beta \cdot \mathrm{KL}(\pi_\theta(\cdot|x)\|\pi_0(\cdot|x))
$$

第一项追求高奖励，第二项惩罚偏离参考策略 $\pi_0$。$\beta$ 越大，越保守；$\beta$ 越小，越激进。

训练时通常不直接对整句概率做原始策略梯度，而是用 PPO 的 ratio clipping。定义：

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}
$$

其中 $a_t$ 是动作，白话说就是当前 token 选择；$s_t$ 是状态，白话说就是当前上下文。PPO 的核心 surrogate loss 通常写成：

$$
\mathcal{L}_{PPO} = -\mathbb{E}\left[\min\left(\rho_t A_t,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t\right)\right]
$$

这里的 $A_t$ 是 advantage，优势函数，白话说就是“这一步动作到底比平均水平好多少”。实际实现里还会加：

- value loss：训练价值网络，降低高方差
- entropy bonus：维持一定探索，避免策略过早塌缩

### 4. 最小数值例子：为什么奖励和 KL 会拉扯

假设同一个 prompt 下有两个候选：

| 候选 | 初始策略 $\pi_0$ | RM 分数 |
| --- | --- | --- |
| $y_1$ | 0.3 | 6 |
| $y_2$ | 0.1 | 3 |

如果 PPO 之后，策略把 $y_1$ 提到 0.4，那么它相对 $\pi_0$ 的对数比率是：

$$
\log \frac{0.4}{0.3} \approx 0.288
$$

如果只看 RM，显然应该继续把 $y_1$ 往上推；但如果推太猛，比如推到 0.9，模型整体分布会大幅偏离 $\pi_0$，语言质量、格式稳定性和泛化都可能退化。

可以把它看成两股力：

| 力量来源 | 希望发生什么 |
| --- | --- |
| RM 奖励 | 高分答案概率继续上升 |
| KL 惩罚 | 不要突然把分布改得面目全非 |

所以 PPO 更新后的目标不是“找到最高分答案”，而是“在参考策略附近，找到更高偏好得分的分布”。

玩具例子里，如果 $y_1$ 从 0.3 升到 0.4，奖励提升明显，KL 代价也还可控；如果升到 0.8，奖励可能继续变好，但 KL 成本可能已经盖过收益。RLHF 的稳定性就在这个折中点上。

---

## 代码实现

三段训练通常都会各自保存 checkpoint，因为它们是三个不同目标、不同数据、不同损失的阶段：

- SFT checkpoint：作为 $\pi_0$
- RM checkpoint：作为打分器 $r_\phi$
- PPO checkpoint：保存 policy、reference、value 相关状态

下面先给一个可运行的玩具 Python 代码，模拟“奖励减 KL”的选择逻辑。它不训练大模型，但能跑通核心公式。

```python
import math

# toy prompt 下两个候选答案
pi0 = {"y1": 0.3, "y2": 0.1}
pi_new = {"y1": 0.4, "y2": 0.08}
reward = {"y1": 6.0, "y2": 3.0}

beta = 0.5

def expected_reward(policy, reward_table):
    return sum(policy[k] * reward_table[k] for k in policy)

def kl_on_shared_support(p, q):
    total = 0.0
    for k in p:
        if p[k] > 0 and q[k] > 0:
            total += p[k] * math.log(p[k] / q[k])
    return total

er0 = expected_reward(pi0, reward)
er1 = expected_reward(pi_new, reward)
kl = kl_on_shared_support(pi_new, pi0)

objective0 = er0
objective1 = er1 - beta * kl

assert er1 > er0
assert kl > 0
assert objective1 > objective0 - 1e-9

print("baseline reward:", round(er0, 4))
print("new reward:", round(er1, 4))
print("KL penalty:", round(kl, 4))
print("new objective:", round(objective1, 4))
```

这段代码的意义很直接：

- `expected_reward` 表示当前策略在 RM 下的期望收益
- `kl_on_shared_support` 表示新策略相对旧策略的漂移
- `objective1 = er1 - beta * kl` 对应 RLHF 的核心目标

真实训练当然复杂得多，因为实际对象是整句 token 序列，而且每个 batch 都要同时经过多个模型。一个简化版 PPO 训练循环可以写成下面这样：

```python
for batch in prompts:
    responses = policy.sample(batch)
    with no_grad():
        ref_logprobs = ref_model.logprob(batch, responses)
        rewards = reward_model.score(batch, responses)
        values = value_model(batch, responses)

    logprobs = policy.logprob(batch, responses)
    kl = logprobs - ref_logprobs
    shaped_rewards = rewards - beta * kl

    advantages, returns = compute_gae(shaped_rewards, values)

    policy_loss = ppo_clipped_loss(logprobs, old_logprobs, advantages)
    value_loss = mse(value_model(batch, responses), returns)
    entropy_loss = entropy_bonus(policy)

    loss = policy_loss + c1 * value_loss - c2 * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(policy.parameters(), max_norm)
    optimizer.step()
```

这个循环要点是四步串联：

1. sampling：当前 policy 先出答案
2. scoring：ref、RM、value 同时打信息
3. loss：把 reward、KL、advantage 组合成 PPO 损失
4. optimizer：反传并更新

真实工程例子里，以 OpenRLHF 的 Llama-3-8B 流水为例，常见做法是：

- 先跑 SFT，得到基础助理 checkpoint
- 用 preference pair 训练 reward model checkpoint
- 用 policy checkpoint + ref checkpoint + reward checkpoint + value head 跑 PPO
- 每阶段单独落盘，便于回滚、复现实验和替换单个组件

---

## 工程权衡与常见坑

RLHF 的难点不在公式，而在工程稳定性。三段里最贵、最慢、最容易发散的通常是 PPO。

### 1. RM 往往比主模型小一档

这是因为 PPO 每个 batch 都要让 RM 参与打分。如果 RM 和主策略同尺寸，整体吞吐会显著下降。于是工程上常见配置是：

- 主策略 8B
- RM 7B、3B 或更小
- value 共享主干或挂轻量 head

这样能省算力，但代价是 RM 太弱时容易被策略“骗”。所谓“骗”，不是模型有意识作弊，而是策略学到一些表面特征，能让 RM 给高分，但对真实人类并不好。

典型例子：

- RM 偏爱“结构完整、礼貌、长度适中”
- policy 学到“多写免责声明、固定模板、堆安全措辞”
- 奖励上升，但用户觉得回答变空、变慢、信息密度下降

这就是 reward hacking，奖励作弊。白话说就是“模型在优化评分器，不是在优化真实需求”。

### 2. PPO 是算力瓶颈

PPO 一个 batch 常常需要：

| 组件 | 作用 | 成本特点 |
| --- | --- | --- |
| policy | 采样与更新主策略 | 成本最高 |
| ref model | 计算 KL 参考概率 | 只前向，但不可省 |
| reward model | 给整句打分 | 每条样本都要跑 |
| value model | 估计回报基线 | 增加额外前向/反向 |

所以同样一张 GPU，PPO 的有效训练速度通常明显慢于 SFT。SFT 主要是在固定标签上做监督学习；PPO 则需要“生成 + 多模型打分 + 多损失回传”。

真实工程里，OpenRLHF 的 8B 级流水通常会把这几类模型拆到多卡上，通过混合引擎或分布式方式并行，原因很简单：否则 rollout 和评分吞吐跟不上。

### 3. 常见异常与监控

| 异常现象 | 可能原因 | 监控手段 |
| --- | --- | --- |
| reward 爆炸 | RM 被钻空子、beta 太小 | 监控 reward 均值与人工抽样 |
| KL 急剧上升 | 学习率过大、更新 epoch 过多 | 监控 per-token KL、early stop |
| 回答变长但变空 | RM 偏爱形式特征 | 监控长度分布、人工对比评估 |
| 训练震荡 | advantage 方差高、batch 太小 | 监控 value loss、reward variance |
| 模型失去基础能力 | 过度偏离 $\pi_0$ | 保留强 ref，增大 KL 惩罚 |

新手最容易忽略的一点是：训练日志里的 reward 上升，不等于模型真正变好了。必须配合人工抽样。因为 RM 是近似的人类偏好，不是人类本身。

### 4. 超参敏感不是“调一调就行”

PPO 常见敏感项包括：

- learning rate
- rollout batch size
- PPO epoch 数
- KL target 或 beta
- reward normalization

这些参数耦合很强。比如学习率略大，再叠加 epoch 稍多，就可能让 KL 在几个 step 内快速失控。很多团队实际会先固定一套保守超参，让训练“慢但稳”，再逐步放开。

---

## 替代方案与适用边界

三段式 RLHF 不是唯一方案，它的优势是解释清晰、模块拆分清楚，但代价是流程长、算力贵、组件多。

### 1. 省略 SFT 的方案

理论上可以直接从已有强指令模型出发，拿偏好数据训练 RM，再跑 PPO。前提是这个起始模型本身已经是不错的助理，也就是“天然就是一个可用的 $\pi_0$”。

适用情景：

- 你拿到的基座已经做过高质量 instruction tuning
- 你手头几乎没有新的 SFT 数据
- 你真正想优化的是风格、偏好、拒答边界

问题是，如果起始策略不够稳，没有明确的 $\pi_0$ 参考，PPO 的漂移就很难控制。模型可能会把很多基础行为一起改坏。

### 2. 直接排序优化或监督替代

资源有限时，很多团队会退一步，不做完整 PPO，而采用更便宜的偏好优化方法，例如直接排序损失、pairwise ranking、甚至把优选答案当 SFT 样本继续监督训练。

它们的共同点是：

- 不需要在线 rollout 那么多次
- 不需要同时挂 policy/ref/RM/value 全套
- 训练更像常规微调，吞吐更高

但它们的边界也明确：当你需要显式控制“奖励提升”和“策略漂移”的平衡时，PPO 仍然更直接。

下面做一个简表：

| 方案 | 是否需要 RM | 是否需要 PPO | 优点 | 缺点 | 适用条件 |
| --- | --- | --- | --- | --- | --- |
| 标准三段 RLHF | 是 | 是 | 对齐控制最完整 | 成本最高，流程最复杂 | 预算足、目标明确 |
| 强 SFT + 轻偏好微调 | 可选 | 否 | 简单稳定 | 偏好信号利用不充分 | 资源有限、追求快迭代 |
| 直接从强模型做 PPO | 是 | 是 | 少一阶段 | 起始策略要求高 | 已有成熟指令模型 |
| 纯监督排序替代 | 否或弱化 | 否 | 成本低、易复现 | 难显式控漂移 | 小团队、实验早期 |

一个新手能理解的情景是：如果你没有大量“标准答案”数据，但有人类偏好对比对，那么可以先训练 RM，再在已有助理模型上做少量 PPO 微调。这是可行的。但如果这个已有模型本身不稳定，缺少可靠的 $\pi_0$，就会出现“奖励涨了，整体行为却散了”的问题。

所以三段式的真正价值，不是“流程多一步更高级”，而是每段都在解决一个独立问题：

- SFT 解决基础可用性
- RM 解决偏好标量化
- PPO 解决受约束的策略优化

---

## 参考资料

[1] DeepWiki / OpenRLHF 基础训练流程  
原始站点：https://deepwiki.com/OpenRLHF/OpenRLHF/9.1-basic-training-workflows

[2] Aman.ai, Preference Optimization Primer  
原始站点：https://aman.ai/primers/ai/preference-optimization/

[3] Suvash Sedhain, Reward Modeling in Practice  
原始站点：https://mesuvash.github.io/blog/2026/reward-modeling/

[4] QubitTool, RLHF Complete Guide  
原始站点：请按标题检索原始文章获取对应版本

[5] OpenRLHF 项目文档与示例代码  
原始站点：请按项目名检索 GitHub 与文档主页

建议阅读顺序：

1. 先看 DeepWiki，建立三阶段流水的整体图。
2. 再看 Aman.ai，理解 PPO 目标、KL 和 clipped ratio。
3. 然后看 Suvash 的奖励建模文章，理解 RM 在工程里为什么容易被“骗”。
4. 最后结合 OpenRLHF 示例，观察 checkpoint、分布式训练和多模型协同是怎么落地的。
