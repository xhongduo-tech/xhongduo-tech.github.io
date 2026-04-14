## 核心结论

PPO，Proximal Policy Optimization，直白说就是“每次只允许策略小步改动”的强化学习优化方法。在 RLHF，Reinforcement Learning from Human Feedback，直白说就是“先让模型学会模仿，再用人类偏好训练出来的奖励模型继续微调”的流程里，PPO 的价值不在于把 reward model 的分数拉到最高，而在于**在奖励上升、策略稳定、语言能力不塌**这三个目标之间做受约束的折中。

它的核心由三部分组成：

1. **clipped objective**：截断目标，直白说就是“新策略相对旧策略的改动不能太大”。  
2. **KL penalty**：KL 罚项，直白说就是“如果新策略偏离参考模型太远，就额外扣分”。  
3. **GAE + value head**：GAE 是“更平滑地估计当前动作到底值不值”的方法，value head 是“给每个状态估计未来总回报”的分支，它们一起降低训练方差。

PPO 在 RLHF 中常见的优化目标可以写成：

$$
L^{\text{CLIP}}=\mathbb{E}\left[\min\left(r_t A_t,\ \text{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right)\right]
$$

其中

$$
r_t=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
=\exp\left(\log \pi_\theta(a_t|s_t)-\log \pi_{\theta_{\text{old}}}(a_t|s_t)\right)
$$

如果把 RLHF 过程说得最朴素一点，就是：  
旧模型先生成回答，reward model 给回答打分，value head 估计“这段回答大概值多少分”，GAE 算出“比预期好多少或差多少”，PPO 再根据这个差值去调策略，但只允许它在 clip 和 KL 的护栏内前进一步。

玩具例子可以先这样理解：旧策略给某个 token 的概率是 0.4，新策略想把它提到 0.65，优势 $A_t=2.0$。这时比例

$$
r_t=0.65/0.4=1.625
$$

若 $\epsilon=0.2$，clip 后最多只按 1.2 计算，于是目标不再用 $1.625\times 2=3.25$，而是被截到 $1.2\times 2=2.4$。这一步的意义不是“算得更保守一点”这么简单，而是防止策略因为一次高奖励样本就突然偏到另一个分布。

---

## 问题定义与边界

RLHF 里的目标不是“让模型输出 reward model 最喜欢的文本”，而是“在尽量保留原始语言能力的前提下，让输出更符合人类偏好”。这里至少有三个信号同时存在：

| 信号来源 | 它在优化什么 | 常见风险 |
| --- | --- | --- |
| reward model 分数 | 鼓励更符合偏好的输出 | reward hacking，模型学会钻打分规则空子 |
| KL 约束 | 防止策略远离参考模型 | 约束过强会学不动，过弱会发散 |
| value 估计 | 为 policy gradient 提供 baseline | 估计不准会让 advantage 噪声很大 |

这里的“baseline”可以白话理解成“先估一个正常水平，再看真实结果比它高还是低”，这样梯度不会因为奖励本身波动太大而乱跳。

问题边界主要有四个。

第一，**PPO 解决的是受约束的策略更新问题，不是 reward model 训练问题**。如果 reward model 本身打分错了，PPO 只会更高效地把错误放大。

第二，**PPO 默认只保证单步更新相对稳定，不保证全局最优**。也就是说，它擅长“别走太猛”，不擅长“保证一定走到最好的地方”。

第三，**RLHF 中的动作不是单个分类标签，而是一串 token**。因此 ratio、advantage、KL 往往都要在 token 级别计算，再汇总到序列级别。文本生成里的“动作空间巨大”是 PPO 在 NLP 里比在小游戏环境里更难调的根本原因。

第四，**PPO 不是为了完全消除分布漂移，而是把漂移压到可控范围**。如果没有 KL 或 clip，策略可能迅速偏到 reward model 评分高但语言质量差、事实性差、冗长重复的区域；如果约束太死，又会几乎退化成监督微调附近的小噪声。

对新手最重要的一句话是：  
在 RLHF 里，reward 是“油门”，KL 和 clip 是“刹车与方向盘”，value + GAE 是“减震器”。只踩油门，车会冲出去；只踩刹车，车又不走。

---

## 核心机制与推导

PPO 的推导起点是 policy gradient。它的基本思想是：如果某个动作最终带来了更高回报，就提高这个动作在同样状态下被选中的概率；如果带来了更低回报，就降低它的概率。

最原始的目标可以写成：

$$
L^{PG}=\mathbb{E}\left[\log \pi_\theta(a_t|s_t)\hat{A}_t\right]
$$

其中 advantage，优势，白话解释就是“这个动作比模型原本预期好多少”。  
如果 $\hat{A}_t>0$，就应该增加该动作概率；如果 $\hat{A}_t<0$，就应该降低。

但直接这样优化会有一个问题：新旧策略可能差太远，导致一次更新就把分布推坏。所以 PPO 改成看**新旧策略的相对变化率**：

$$
r_t=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

于是目标改成：

$$
L^{CPI}=\mathbb{E}[r_t\hat{A}_t]
$$

这已经比直接用新策略概率更合理，因为它显式参考了旧策略。但如果 $r_t$ 很大，仍然会有过冲。于是 PPO 再做一步 clip：

$$
L^{\text{CLIP}}=\mathbb{E}\left[\min\left(r_t\hat{A}_t,\ \text{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t\right)\right]
$$

这条公式要分正负优势分别理解。

1. 当 $\hat{A}_t>0$ 时，说明这个动作值得鼓励。  
如果 $r_t$ 从 1 增长到 1.1、1.15，这是合理的；但如果直接涨到 1.8，就说明新策略过度偏向这个动作，clip 会把它截住。

2. 当 $\hat{A}_t<0$ 时，说明这个动作不值得鼓励。  
这时如果 $r_t$ 掉得太狠，clip 同样会阻止你一次性把这个动作打入冷宫。

所以 clip 的本质是：**无论奖励方向如何，都限制单步更新幅度**。

接下来是 advantage 从哪里来。在 RLHF 中，常见做法是训练一个 value head。value head 可以理解为“挂在 policy 模型上的一个额外回归头，用来预测当前状态未来大概还能拿到多少总回报”。

它输出 $V(s_t)$，然后构造时序差分误差：

$$
\delta_t=r_t^{\text{reward}}+\gamma V(s_{t+1})-V(s_t)
$$

这里的 $r_t^{\text{reward}}$ 是环境奖励或经 KL 修正后的 token 级奖励。  
GAE，Generalized Advantage Estimation，白话说就是“把未来很多步的 TD 误差按衰减权重加起来”，公式是：

$$
\hat{A}_t=\sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k}
$$

其中：

- $\gamma$ 控制未来奖励折现，白话说就是“未来值多少钱”
- $\lambda$ 控制 bias-variance tradeoff，白话说就是“更信短期估计还是更信长链路回报”

为什么 GAE 有用？因为纯 Monte Carlo 回报方差太大，纯一步 TD 偏差又太强。GAE 在两者之间做折中，通常更适合文本 RLHF 这种高噪声场景。

在 RLHF 里，还要再加一个关键项：**KL penalty**。  
它不是 PPO 原始论文唯一必须的部分，但在语言模型对齐里极常见，因为必须防止模型远离初始参考策略。常见写法是：

$$
R_t^{\text{shaped}} = R_t^{\text{rm}} - \beta D_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}})
$$

或者在 token 级近似为：

$$
r_t^{\text{kl}}=\beta\left(\log \pi_\theta(a_t|s_t)-\log \pi_{\text{ref}}(a_t|s_t)\right)
$$

然后把它并入奖励。这里 $\beta$ 就是 KL 系数，白话说就是“偏离参考模型要罚多重”。

真实工程里，经常不是一个总 loss，而是三部分相加：

$$
L = L_{\text{policy}} + c_v L_{\text{value}} + c_{kl} L_{\text{kl}}
$$

其中

$$
L_{\text{value}}=\frac{1}{2}\mathbb{E}(V_\theta(s_t)-\hat{G}_t)^2
$$

$\hat{G}_t$ 是目标回报。  
这意味着 PPO 在 RLHF 里不是只训练 policy，它是在同一批样本上同时训练：

- policy：让输出更符合偏好
- critic/value：让优势估计更稳
- 相对参考策略的偏移控制：让语言能力不崩

真实工程例子：  
假设一个问答模型回答“如何重置密码”。reward model 更喜欢“步骤清晰、礼貌、简洁”的回答。如果没有 KL，模型可能学会堆模板式废话，因为 reward model 也许对“看起来完整”的文本给高分；如果没有 value head，样本间奖励波动大，训练会抖；如果没有 clip，某一批高分回答会把策略猛拉向狭窄风格。PPO 的三件套正是在约束这种联合作用。

---

## 代码实现

下面先给一个最小可运行的玩具实现，只演示 PPO 的三个核心量：ratio、clip、value loss。它不是完整训练器，但代码能运行，而且能验证公式行为。

```python
import math

def clipped_policy_objective(old_prob, new_prob, advantage, eps=0.2):
    ratio = new_prob / old_prob
    clipped_ratio = min(max(ratio, 1 - eps), 1 + eps)
    return min(ratio * advantage, clipped_ratio * advantage), ratio, clipped_ratio

def value_loss(value_pred, return_target):
    return 0.5 * (value_pred - return_target) ** 2

# 玩具例子：旧概率 0.4，新概率 0.65，优势为正
obj, ratio, clipped_ratio = clipped_policy_objective(0.4, 0.65, 2.0, eps=0.2)
assert round(ratio, 3) == 1.625
assert round(clipped_ratio, 3) == 1.2
assert round(obj, 3) == 2.4

# 如果优势为负，过大的 ratio 反而会更糟，clip 仍然生效
obj2, ratio2, clipped_ratio2 = clipped_policy_objective(0.4, 0.65, -2.0, eps=0.2)
assert round(obj2, 3) == -3.25
assert round(ratio2, 3) == 1.625
assert round(clipped_ratio2, 3) == 1.2

# value head 预测未来回报
vloss = value_loss(value_pred=5.8, return_target=6.5)
assert round(vloss, 3) == 0.245

print("ppo toy example ok")
```

为什么第二个 `assert` 看起来没有被 clip 成 `-2.4`？因为 PPO 目标里取的是 `min(...)`。  
当 advantage 为负时，`min(-3.25, -2.4)=-3.25`，这正是 PPO 设计的关键之一：它不会阻止你惩罚坏动作，但会阻止你对好动作奖励过头。

再看一个更接近 RLHF 的伪代码。下面的逻辑是主流框架中都会出现的骨架：

```python
def ppo_rlhf_step(policy_model, ref_model, reward_model, optimizer, batch,
                  gamma=1.0, lam=0.95, clip_eps=0.2,
                  vf_coef=0.5, kl_coef=0.02):
    queries = batch["queries"]

    # 1. 当前策略生成回答
    responses, logprobs, values = policy_model.generate_with_logprobs_and_values(queries)

    # 2. 参考模型提供 ref logprobs，用于 KL 约束
    ref_logprobs = ref_model.logprobs(queries, responses)

    # 3. reward model 对完整回答打分
    rm_scores = reward_model.score(queries, responses)

    # 4. 形成 token 级 shaped reward
    # 常见做法：中间 token 只有 KL 惩罚，末 token 再加 reward model 分数
    kl_per_token = logprobs - ref_logprobs
    rewards = -kl_coef * kl_per_token
    rewards[:, -1] += rm_scores

    # 5. 用 value head + GAE 计算 advantage / returns
    advantages, returns = compute_gae(rewards, values, gamma=gamma, lam=lam)

    # 6. 重新前向，得到新 logprobs / new values
    new_logprobs, new_values = policy_model.evaluate(queries, responses)

    ratio = (new_logprobs - logprobs).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()

    value_loss = 0.5 * ((new_values - returns) ** 2).mean()
    approx_kl = (new_logprobs - logprobs).mean()

    loss = policy_loss + vf_coef * value_loss + kl_coef * approx_kl

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "approx_kl": approx_kl.item(),
        "reward_mean": rm_scores.mean().item(),
    }
```

这段伪代码里有两个 RLHF 特有点。

第一，**reward model 通常给的是序列级分数，不是每个 token 一个分数**。  
所以工程上常把 reward model 分数加到最后一个 token，其他 token 主要承受 KL 惩罚。这样就把“整段回答得分”映射回 token 轨迹。

第二，**policy 和 value 往往共享主干，只是在最后接不同 head**。  
这能节省显存，也让 value 学到与语言表征一致的状态特征。

常见超参可以先这样理解：

| 超参 | 作用 | 调大后常见现象 | 调小后常见现象 |
| --- | --- | --- | --- |
| `clip_eps` | 单步策略改动范围 | 更新更激进，容易不稳 | 更新更保守，收敛慢 |
| `kl_coef` | 偏离参考模型的惩罚强度 | 输出更像旧模型，reward 难涨 | 更容易偏航、发散 |
| `vf_coef` | value loss 权重 | 更重视 critic 拟合 | advantage 更噪 |
| `learning_rate` | 参数更新步长 | clip fraction、KL 易暴涨 | 学得太慢 |
| `gamma` | 未来奖励折现 | 更重长期回报 | 更重即时分数 |
| `lambda` | GAE 平滑程度 | 方差更低但偏差可能变大 | 更接近短期 TD |

如果把它放进真实工程流水线，通常还有这些步骤：

1. 从 prompt 数据集采样输入。  
2. 当前 policy 生成若干回复。  
3. reward model 打分，同时计算相对 reference model 的 KL。  
4. 对 reward 做标准化或 whiten，白话说就是“把不同 batch 的数值尺度拉回稳定区间”。  
5. 用 GAE 算 advantage。  
6. 做多轮 minibatch PPO 更新。  
7. 监控 `approxkl`、`clipfrac`、`reward_mean`、`value_loss`、`response_length`。

---

## 工程权衡与常见坑

PPO 在论文里看起来很整齐，但在 RLHF 里最难的是工程细节。下面这些坑几乎都会遇到。

| 问题 | 现象 | 常见缓解手段 |
| --- | --- | --- |
| KL over-penalize | 模型越来越像参考模型，reward 不涨 | 降低 `kl_coef`，用 adaptive KL |
| KL 过小 | 文风突变、胡言乱语增多 | 提高 `kl_coef`，减小学习率 |
| clip fraction 过高 | 说明大量样本都撞上 clip 边界 | 降低学习率或减小 batch update epoch |
| reward hacking | 模型学会刷分而不是真变好 | length penalty、拒答约束、多维 reward |
| value 崩 | advantage 噪声大，loss 抖动 | 提高 `vf_coef`，做 reward normalization |
| 回复变长 | 末 token 奖励驱动模型拖长输出 | 长度惩罚、EOS 奖励校正 |
| approxkl 异常 | 更新方向和预期不符 | 检查 logprob 对齐、mask、padding |

先说最关键的 KL。  
在 RLHF 中，KL 不只是“正则项”，它其实在定义“你允许模型偏离基线模型多少”。`kl_coef` 太大，模型几乎只会在原模型附近轻微抖动；`kl_coef` 太小，reward model 会把策略拉向奇怪区域。很多工程实现会用 **adaptive KL**，也就是根据实际观测到的 KL 自动调节 $\beta$，目标是让 KL 保持在某个范围附近。

再说 reward hacking。  
reward hacking，白话说就是“模型学会骗评分器，而不是学会真正更好地回答”。在文本生成里，一个非常常见的形式就是：如果 reward model 更偏好看起来全面的回答，模型可能通过拉长文本、重复安全模板、堆礼貌套话来拿高分。这就是为什么很多实现要加入长度惩罚、格式约束，或者把 reward 拆成多个维度，而不是只靠一个标量分数。

一个真实工程例子：  
如果你训练的是客服助手，reward model 可能给“非常详细、非常礼貌”的回复更高分。但生产环境真正需要的是“正确、短、可执行”。这时 PPO 可能把模型推向每次都输出四段免责声明。离线评估分数上去了，线上完成率却下降。这不是 PPO 算错了，而是 reward 定义和业务目标错位了。

还有一个经常被忽视的问题是 **mask 与对齐**。  
文本序列里有 prompt token、response token、padding token。PPO loss、KL、value loss 通常只应该在 response 有效位置上算。如果把 prompt 部分也算进去，或者 padding 没 mask 掉，logprob ratio 会失真，`approxkl` 也会变得很怪。

另一个坑是 **old logprobs 的使用时机**。  
PPO 假设一批样本对应的 old policy 固定。如果你在一个 rollout 上重复更新太多轮，new policy 已经离 old policy 很远，clip 虽然还在，但数据已经过时，更新质量会迅速下降。所以工程上通常限制每批 rollout 的 PPO epoch 数，不会无限重复榨干同一批样本。

---

## 替代方案与适用边界

PPO 不是唯一方案，它只是当前工程上“效果、稳定性、复杂度”比较平衡的一种。

| 方法 | 核心约束方式 | 优点 | 缺点 | 适用边界 |
| --- | --- | --- | --- | --- |
| PPO | clip + 可选 KL | 简单、高效、并行友好 | 仍需细调超参 | 大多数 RLHF 微调 |
| TRPO | trust region 二阶约束 | KL 约束更严格 | 实现复杂、计算重 | 对稳定性要求极高的小规模场景 |
| Adaptive KL penalty | 动态调 KL 系数 | 更易维持目标 KL | 仍依赖一阶优化稳定性 | PPO 的常见增强版 |

TRPO，Trust Region Policy Optimization，白话说就是“直接从优化问题层面限制每步不能走出可信区域”。它比 PPO 更接近“严格约束”的原始思路，但需要处理二阶信息，如 Fisher 矩阵近似，复杂度高得多。对大模型 RLHF 来说，这种代价通常不划算，所以很多系统更偏向 PPO 或带自适应 KL 的 PPO。

还有一类替代思路是不做完整 PPO，而是采用**更直接的 KL 正则化目标**。比如只做 policy loss + KL，或者在偏好学习后改用 DPO、IPO 这类直接偏好优化方法。它们绕开在线 rollout 的一部分复杂度，但边界也明确：如果你需要的是在线交互式 reward 优化，PPO 仍然更自然。

什么时候 PPO 不一定是最佳选择？

1. 当 reward signal 极其稀疏，value 很难学稳时。  
2. 当你必须把 KL 精确锁在很小范围时，TRPO 类方法更有理论吸引力。  
3. 当你不想维护 rollout、critic、advantage 这一整套 RL 管线时，直接偏好优化方法更省工程成本。  
4. 当 reward model 噪声很大时，PPO 可能只是稳定地优化错误目标。

所以更准确的结论不是“RLHF 就该用 PPO”，而是：  
**如果你已经有可用的 reward model，需要在线地、渐进地改策略，同时还要保住原模型分布，PPO 是当前非常务实的默认选项。**

---

## 参考资料

1. RLHF Book: *Policy Gradients / Value Functions and PPO*  
   链接：https://rlhfbook.com/c/11-policy-gradients  
   作用：解释 RLHF 中 policy gradient、value function、PPO、GAE 的关系。

2. RLHF Book PDF  
   链接：https://rlhfbook.com/book.pdf  
   作用：给出 PPO、GAE、KL penalty 在 RLHF 里的统一数学视角。

3. Hugging Face: *The N Implementation Details of RLHF with PPO*  
   链接：https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo  
   作用：覆盖 reward shaping、KL 控制、长度问题、训练敏感点等工程细节。

4. ApX Machine Learning: *PPO for RLHF Context*  
   链接：https://apxml.com/courses/rlhf-reinforcement-learning-human-feedback/chapter-4-rl-ppo-fine-tuning/ppo-for-rlhf-context  
   作用：适合把 PPO 在 RLHF 中的训练流程和数值例子连起来理解。
