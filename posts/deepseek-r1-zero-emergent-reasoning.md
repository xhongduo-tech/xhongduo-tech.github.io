## 核心结论

DeepSeek-R1-Zero 的关键现象是：**不先做 SFT 冷启动，只在基础模型上直接做强化学习，推理行为也会自己长出来**。这里的强化学习，白话说就是“模型先试着答，再按结果拿分，分高的写法以后更容易被重复”。

这件事重要，因为它说明推理能力不一定只能靠人工示范的思维链教出来。只要任务可验证，例如数学题有标准答案、代码题能跑测试，训练系统就能只依据“答对还是答错”这类标量奖励，让模型逐渐学会更长的思考、更频繁的自检，以及走错后回退重算。Nature 论文给出的一级证据是：AIME 2024 上，DeepSeek-R1-Zero 的 pass@1 从 15.6% 升到 77.9%，用 16 次自一致采样可到 86.7%；同时平均输出长度在训练中持续上升，并在 8.2k step 后出现明显跳变。

这类“推理涌现”不是魔法，而是策略优化的自然结果。如果长一点、慢一点、带验证的解法更容易拿到奖励，策略梯度就会把概率质量推向这些解法。结果就是：模型不一定更优雅，但往往更愿意写出长链条、做中间检查、尝试替代路径。

一个给初学者的玩具理解是：你不告诉学生“标准解题步骤”，只说“答对记 1 分，答错记 0 分”。如果学生发现“先列式、再验算、错了回头改”更容易得分，他就会自己形成这种习惯。R1-Zero 的现象，本质上就是这个过程在大模型上的放大版。

---

## 问题定义与边界

先给定义。**DeepSeek-R1-Zero 是一个 RL-only 推理模型**。RL-only 的意思是：后训练阶段不先喂人工写好的推理示例，而是直接从 DeepSeek-V3-Base 出发，用规则奖励做强化学习。

它解决的问题不是“让模型说话更像人”，而是“让模型在可验证任务上更会推”。可验证任务，白话说就是答案能被程序或规则直接判对错的任务，比如数学、代码、逻辑推理。它并不直接解决开放域写作、百科问答、品牌文案这类“答案没有唯一判分器”的任务。

下面这张表可以把边界看清楚：

| 任务类型 | 是否容易设计可靠奖励 | RL-only 的典型表现 | 主要限制 |
| --- | --- | --- | --- |
| 数学题 | 高 | 长链推理、自检、回退明显 | 输出冗长，用户不一定愿意看 |
| 代码题 | 高 | 更愿意逐步分析并对照测试 | 可能反复解释同一处 bug |
| 逻辑推理 | 中高 | 会尝试多路径验证 | 可能在错误路径上“想太久” |
| 开放域写作 | 低 | 难稳定提升 | 奖励难定义，容易 reward hacking |
| 日常对话 | 低 | 可读性不稳定 | 容易语言混杂、风格不统一 |

所以，R1-Zero 的边界很明确：它更像“面向可验证推理的研究样机”，不是直接面向所有用户场景的成品助手。Nature 原文也直接指出了它的两个主要缺点：**poor readability** 和 **language mixing**，也就是可读性差、语言混杂。

---

## 核心机制与推导

R1-Zero 的核心训练算法是 **GRPO**，全称 Group Relative Policy Optimization。白话解释：同一道题不只采样一个答案，而是采样一组答案，然后看这组里谁相对更好，再推动策略朝更好的那几个方向移动。

它的关键目标可以写成：

$$
\mathcal{J}_{\text{GRPO}}(\theta)=
\mathbb{E}_{q,\{o_i\}}
\left[
\frac{1}{G}\sum_{i=1}^{G}
\left(
\min\left(
\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}A_i,\;
\mathrm{clip}\left(
\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)},
1-\epsilon,1+\epsilon
\right)A_i
\right)
-\beta D_{KL}(\pi_\theta\|\pi_{\text{ref}})
\right)
\right]
$$

其中组内优势是：

$$
A_i=\frac{r_i-\mu}{\sigma}
$$

这里每个符号都不神秘：

| 符号 | 含义 | 白话解释 |
| --- | --- | --- |
| $o_i$ | 第 $i$ 个 rollout | 同一道题采样出的第 $i$ 个回答 |
| $r_i$ | 奖励 | 这条回答拿到的分数 |
| $A_i$ | 优势 | 这条回答比同组平均值好多少 |
| $\epsilon$ | clip 参数 | 防止一步更新太猛 |
| $\beta$ | KL 系数 | 防止新策略偏离参考策略太远 |

为什么这会催生长推理？因为奖励只看结果正确性和格式，不限制中间思路长什么样。当模型发现“多写几步、顺手验算、必要时回退”更容易得到高奖励，它就会把这类行为保留下来。

可以把这个机制想成一次小型竞赛。同一道题先写 16 个版本，不是谁绝对高分谁就赢，而是看“相对同组平均水平谁更好”。这样做有两个效果：

1. 奖励更稳定。不是直接拿原始分数硬推，而是做组内归一化，减少波动。
2. 探索更充分。模型可以在同一道题上试多种思路，逐渐发现哪种思路更稳。

一个玩具例子：

同一道题采样 4 个答案，奖励分别是 `[0, 0, 1, 1]`。  
如果这 2 个正确答案都恰好是“写得更长、且中间带验算”的版本，那么它们的优势 $A_i$ 就更高。训练多轮后，策略会逐渐提高“长链+自检”这类输出的概率。这个过程不需要人工告诉模型“先列条件、再验证”，只需要奖励系统持续偏爱最终正确的路径。

Nature 论文观察到的“aha moment”也可以这样理解：当模型在训练中第一次大规模发现“等等，我应该回头检查一下”这类模式会提高得分，它的推理风格会出现相变，从直接往前写，变成更像“边走边验”的策略。

---

## 代码实现

如果把论文训练循环缩成初学者能读懂的版本，它大致是下面这样：

```python
import math

def normalize_advantages(rewards):
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var) if var > 0 else 1.0
    return [(r - mean) / std for r in rewards]

def clipped_objective(prob_ratio, advantage, eps=0.2):
    unclipped = prob_ratio * advantage
    clipped_ratio = min(max(prob_ratio, 1 - eps), 1 + eps)
    clipped = clipped_ratio * advantage
    return min(unclipped, clipped)

# 玩具例子：同一道题采样 4 个答案，只有后两个答对
rewards = [0.0, 0.0, 1.0, 1.0]
advantages = normalize_advantages(rewards)

assert len(advantages) == 4
assert abs(sum(advantages)) < 1e-9
assert advantages[2] > 0 and advantages[0] < 0

# 新策略对“正确答案3”的概率提升了 50%，但会被 clip 限制
obj = clipped_objective(prob_ratio=1.5, advantage=advantages[2], eps=0.2)
expected = 1.2 * advantages[2]
assert abs(obj - expected) < 1e-9

print("advantages:", [round(x, 3) for x in advantages])
print("clipped objective:", round(obj, 3))
```

这段代码不训练大模型，但它抓住了 GRPO 的三个核心动作：

1. 对同题多次采样。
2. 用组内均值和标准差算优势。
3. 用 clip 限制过猛更新。

真实工程版本当然复杂得多。Nature 披露的 DeepSeek-R1-Zero 训练细节包括：每题采样 16 个输出；每步 32 道不同题，所以 batch size 是 512；8.2k step 前最大长度 32,768 token，之后升到 65,536；总训练 10,400 step；每 400 step 用最新 policy 替换 reference model；每次 rollout 生成 8,192 个输出，再切成 16 个 minibatch，只做 1 个 inner epoch。

下面是一个更接近工程流程的伪代码：

```python
for step in range(10400):
    questions = sample_questions(32)
    rollouts = [policy.sample(q, n=16, max_len=max_len(step)) for q in questions]
    rewards = rule_based_verifier(rollouts)  # 数学判题/代码跑测
    loss = grpo_loss(policy, ref_policy, rollouts, rewards, kl_coef=0.001, clip_eps=10)
    policy.update(loss)

    if step % 400 == 0:
        ref_policy.load_state(policy)
```

真实工程例子可以想成一个“数学题与代码题统一训练平台”：

| 模块 | 数学任务 | 代码任务 |
| --- | --- | --- |
| 输入 | 题目文本 | 编程题描述 |
| rollout | 采样 16 个解法 | 采样 16 个程序 |
| verifier | 比对标准答案 | 跑单测/隐藏测试 |
| reward | 答对给高分，格式合法加分 | 测试通过率越高分越高 |
| 风险 | 长链太长 | 代码解释太长、重复修 bug |

这个例子说明了 R1-Zero 为什么适合“有判题器”的环境。只要 verifier 足够可靠，模型就能围绕“怎样更容易通过验证”自发长出推理结构。

---

## 工程权衡与常见坑

R1-Zero 最强的地方，恰好也是它最大的问题来源：**它愿意用大量 token 换正确率**。一些公开二级整理常把成功样本概括为 4,700+ token 的长思维链；而 Nature 的一级结论更稳妥，表述为“从数百 token 增长到数千 token”，并在 8.2k step 后因为允许更长上下文而进一步上升。

这带来几个典型工程坑：

| 控制项 | 解决什么问题 | 不加会怎样 |
| --- | --- | --- |
| `max_tokens` 上限 | 超长 CoT | 成本高、时延高、用户读不完 |
| 早停规则 | 错路反复打转 | 在错误思路上自我验证很多轮 |
| 语言一致性奖励 | 中英混杂 | 输出不适合直接展示 |
| 后续偏好对齐 | 可读性差 | 结果对，但像训练日志 |
| 分阶段训练 | 奖励过窄 | 推理强，通用对话弱 |

最容易被误解的一点是：**长 CoT 不等于更聪明，很多时候只是更肯花 token 试错**。如果 reward 设计不够好，模型会学到“多说一点也许更容易撞对”，这会形成冗长、重复、甚至自我催眠式的推理。

另一个常见坑是 **reward hacking**。这个词的意思是“模型学会骗分，而不是学会真正解决问题”。在数学和代码里，规则奖励相对可靠，所以风险可控；但一旦迁移到开放式写作，奖励器常常变成另一个模型，这时系统更容易出现“看起来像高分答案，其实没真正解决问题”的情况。

---

## 替代方案与适用边界

R1-Zero 不是最终形态，更像一个研究结论的极端版本：**纯 RL 的确能催生推理，但会牺牲可读性和通用性**。这也是后续 DeepSeek-R1 采用多阶段训练的原因。

可以把两条路线放在一起看：

| 路线 | 训练阶段 | 优点 | 适用边界 |
| --- | --- | --- | --- |
| R1-Zero | 直接 RL-only | 最能观察推理自然涌现 | 研究推理机制、可验证任务 |
| R1 | 冷启动 SFT + RL + 再 SFT + 再 RL | 推理与可读性更平衡 | 通用助手、生产部署 |
| Distill | 用大模型推理样本蒸馏小模型 | 成本低、易部署 | 资源受限环境 |

如果要对外提供产品，通常不会直接把 R1-Zero 原样上线，而会走下面这条更稳的工程路线：

1. 先用 RL-only 挖出高质量推理模式。
2. 再用 SFT 和偏好对齐把语言风格修整到可展示状态。
3. 最后把这种推理模式蒸馏到更小模型，降低延迟和成本。

因此，R1-Zero 的适用边界不是“所有对话系统”，而是“研究推理如何出现”“构造高质量 reasoning data”“为后续模型蒸馏提供源头样本”。如果你关心的是推理能力从哪里来，它非常有代表性；如果你关心的是最终用户体验，它本身又明显不够。

---

## 参考资料

1. Nature: [DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning](https://www.nature.com/articles/s41586-025-09422-z)
2. DeepSeek-AI GitHub: [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)
3. arXiv / official citation entry: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
4. Emergent Mind: [DeepSeek-R1-Zero: RL-Only Reasoning LLM](https://www.emergentmind.com/topics/deepseek-r1-zero)
