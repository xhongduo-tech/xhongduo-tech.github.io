## 核心结论

推理强化训练里，奖励设计先回答一个根问题：**到底奖励“最后答对”，还是奖励“中间推得对”**。前者叫 ORM，Outcome Reward Model，白话就是“只看交卷分数”；后者叫 PRM，Process Reward Model，白话就是“每一步草稿都要判分”。

两类奖励没有谁绝对更先进，差别主要在信号密度和成本结构：

| 维度 | ORM | PRM |
|---|---|---|
| 奖励位置 | 只在最终答案处给分 | 在推理步骤上持续给分 |
| 信号密度 | 稀疏 | 密集 |
| 标注成本 | 低，常可规则化验证 | 高，常需人工或模型逐步标注 |
| 训练难点 | 探索难，容易只学常见套路 | 标注贵，且容易把“好过程”定义错 |
| 典型场景 | 数学判题、代码测评、确定性问答 | 复杂数学证明、形式化推理、长链规划 |

对新手可以用一个玩具比喻理解。想象写作文：

- ORM 像老师只看最后总分，不看你中间每段怎么写。
- PRM 像老师会逐段批注，指出“第二段论证跳步”“第三段例子不成立”。

这也是为什么 DeepSeek-R1 这条路线能用**纯 ORM + GRPO**做出推理能力，而 AlphaProof 这类高难度数学系统会走**PRM + MCTS**。前者证明“只看终点也能学会推理”，后者证明“高难题里过程反馈和搜索通常更强”。

GRPO，Group Relative Policy Optimization，白话就是“同一道题一次采样多份答案，让这些答案彼此比较，再决定该鼓励谁”。它的关键作用是：即使 ORM 很稀疏，只要一组候选答案里有人做对、有人做错，组内相对比较也能形成有效梯度。

一个常见简化写法是：

$$
\mathcal{J}_{\text{GRPO}}(\theta)
=
\mathbb{E}\left[
\frac{1}{G}\sum_{i=1}^{G}
\min\left(r_i A_i,\ \text{clip}(r_i,1-\epsilon,1+\epsilon)A_i\right)
-\beta D_{KL}(\pi_\theta\|\pi_{\text{ref}})
\right]
$$

其中：

- $G$ 是组大小，一次对同一题采样多少个答案。
- $A_i$ 是 advantage，优势，白话就是“这个答案比组内平均水平好多少”。
- $r_i=\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}$ 是新旧策略概率比。
- $\epsilon$ 是 clip 阈值，用来限制更新幅度。
- $\beta D_{KL}$ 是 KL 正则，白话就是“别为了拿奖励把分布改得太离谱”。

核心判断可以压缩成一句话：**资源紧、可验证任务多时先用 ORM；任务特别难、过程质量决定成败时才值得上 PRM。**

---

## 问题定义与边界

“推理强化训练中的奖励设计”不是泛泛讨论“奖励函数怎么写”，而是更具体地讨论：**在长链推理里，奖励信号应该落在结果，还是落在过程**。

设模型对一道题生成 $T$ 个步骤的推理链：

$$
s_1 \rightarrow s_2 \rightarrow \cdots \rightarrow s_T \rightarrow y
$$

其中 $y$ 是最终答案。

那么两种奖励边界分别是：

- ORM：只定义 $R(y)$，例如答对给 1，答错给 0。
- PRM：定义每一步的 $R_t(s_t)$，最终总奖励可写成 $\sum_{t=1}^{T} R_t$ 或带折扣聚合。

看一个三步推理链的玩具例子。题目是“$17 \times 3$ 等于多少”。

推理链可能是：

1. $17 \times 3 = (10+7)\times 3$
2. $= 30 + 21$
3. $= 51$

如果最终答案写成 51：

- ORM 只在最后给 1。
- PRM 可以给每步各 $0.33$，也可以给更细粒度，比如前两步各 $0.2$，最后一步 $0.6$。

如果第二步写成 $30+18$：

- ORM 可能直接给 0。
- PRM 则能定位“前缀是对的，第二步开始错了”。

这就是两者最本质的区别：**ORM 回答“你最后有没有做对”，PRM 回答“你是从哪一步开始做错的”。**

信息流可以用下面这个表理解：

| 阶段 | ORM 信息流 | PRM 信息流 |
|---|---|---|
| 输入题目 | 进入策略模型 | 进入策略模型 |
| 生成步骤 1 | 不打分 | 可打分 |
| 生成步骤 2 | 不打分 | 可打分 |
| 生成步骤 3 | 不打分 | 可打分 |
| 生成最终答案 | 打分 | 打分 |
| 反馈回训练 | 主要由终局奖励回传 | 由多步局部奖励回传 |

边界还不只在“结果 vs 过程”。工程上还要补两个现实条件。

第一，ORM 并不等于“完全没有过程信号”。在 GRPO 里，如果同一题采样多条答案，这些答案往往共享前缀、再在后面分叉。于是即便奖励只看终点，组内比较也会把“哪些前缀更常通向正确终点”隐式强化出来。这就是后来论文所说的“GRPO 隐式 PRM”。

第二，PRM 也不等于“肯定更稳定”。如果 PRM 标注质量差，或者训练时早停过猛，模型可能因为前几步分低而根本不继续探索完整解链。也就是说，**密集奖励不自动等于好奖励**。

因此本文讨论的边界是：

- 任务以可验证推理为主，如数学、代码、形式化证明。
- 奖励对象聚焦推理行为，而不是一般聊天偏好。
- 训练算法以 PPO/GRPO 一类策略优化为背景。
- 不展开多智能体信用分配，也不讨论开放式创作任务的主观奖励。

---

## 核心机制与推导

先看 ORM 怎么进入 GRPO。

对一道题 $q$，一次采样 $G=3$ 个候选输出，最终规则验证得到结果奖励：

| 样本 | 最终是否正确 | 原始奖励 |
|---|---:|---:|
| $o_1$ | 对 | 1 |
| $o_2$ | 错 | 0 |
| $o_3$ | 错 | 0 |

GRPO 常把组内奖励做标准化，形成 advantage。最简单的直观写法是：

$$
A_i = \frac{R_i - \mu}{\sigma + \delta}
$$

其中 $\mu$ 是组内均值，$\sigma$ 是组内标准差，$\delta$ 是防止除零的小常数。

上面这个例子里：

- $\mu = \frac{1+0+0}{3} = \frac{1}{3}$

若忽略 $\delta$，标准差约为 $0.471$，则：

- $A_1 \approx \frac{1-1/3}{0.471} \approx 1.414$
- $A_2 \approx \frac{0-1/3}{0.471} \approx -0.707$
- $A_3 \approx -0.707$

这一步非常关键。虽然 ORM 原始奖励只有 0 或 1，但经过组内相对化后，训练目标不再只是“答对加分”，而是“比同组其他答案更值得增加概率”。

再看 ratio 与 clip。假设三条输出在新旧策略下的概率比为：

- $r_1 = 1.30$
- $r_2 = 0.80$
- $r_3 = 1.50$

若 $\epsilon=0.2$，clip 后得到：

- $\text{clip}(r_1)=1.2$
- $\text{clip}(r_2)=0.8$
- $\text{clip}(r_3)=1.2$

逐项计算优化项：

1. 对正确样本 $o_1$，因为 $A_1>0$，目标倾向于增大概率，但被 clip 限制在 $1.2A_1$，防止更新过猛。
2. 对错误样本 $o_2$，因为 $A_2<0$，目标倾向于减小概率，$0.8A_2$ 仍是负值。
3. 对错误样本 $o_3$，虽然当前新策略把它概率拉到了 1.5 倍，但因为它是负 advantage，clip 后会更强地抑制“过度偏向错误答案”的更新。

把数值写出来更直观：

| 样本 | $A_i$ | $r_i$ | clip 后 | $\min(r_iA_i,\text{clip}\cdot A_i)$ |
|---|---:|---:|---:|---:|
| $o_1$ | 1.414 | 1.30 | 1.20 | 1.697 |
| $o_2$ | -0.707 | 0.80 | 0.80 | -0.566 |
| $o_3$ | -0.707 | 1.50 | 1.20 | -1.061 |

平均后，模型会被推动去提高 $o_1$ 类型答案的概率，降低 $o_2,o_3$ 类型答案的概率。

这里能看到一个关键事实：**即便奖励只在终点给，组内相对比较也把“哪些生成轨迹更像成功轨迹”传回去了。**

为什么有人说 GRPO “偷偷像 PRM”？原因在于长链输出并不是一个原子动作，而是一串 token。若同组样本共享前缀：

- 样本 A：前 20 个 token 合理，后面继续推对了。
- 样本 B：前 20 个 token 一样，但第 21 个 token 开始走偏。
- 样本 C：更早就走偏。

那么最终的 ORM 奖励在反向传播时，会让共享前缀上的 token 也受到正向强化。这相当于一种**Monte Carlo 式的隐式过程奖励**：不是显式给“第 12 步 0.2 分”，但训练效果会近似表达“这段前缀更有希望通向正确终点”。

这也是为什么只看 ORM 时，仍可能出现“反思、回溯、自检”一类推理行为。它不一定来自人类逐步标注，而可能来自组内竞争把这些中间模式筛出来。

但这个机制有两个限制：

- 如果所有样本都错，ORM 全是 0，组内 advantage 可能塌缩，探索仍然困难。
- 如果正确答案极少、分叉点很晚，隐式过程奖励会很弱，PRM 的优势才会明显放大。

真实工程例子是 AlphaProof。它在 Lean 形式化证明环境中不只是看“最终证明过没过”，还结合搜索树中的中间状态质量来引导探索。MCTS，蒙特卡洛树搜索，白话就是“在可能的推理分支上分配试错预算，优先深入更有希望的分支”。这类系统面对 IMO 级别难题时，比单纯终局奖励更容易利用计算资源。

---

## 代码实现

实现层面，最稳妥的做法不是把 ORM 和 PRM 写成两套训练器，而是把它们抽象成统一的 `reward_callback`。GRPO 主过程只关心“给我每个样本的奖励”，至于奖励来自终点还是步骤，由回调决定。

先看一个可运行的玩具实现。它不依赖深度学习框架，只演示 ORM、PRM 和组内 advantage 的核心结构。

```python
from math import sqrt

def orm_reward(final_answer, ground_truth):
    return 1.0 if final_answer == ground_truth else 0.0

def prm_reward(steps, expected_steps):
    # 每一步匹配给分，返回 [step_reward...]
    rewards = []
    for i, step in enumerate(steps):
        rewards.append(1.0 if i < len(expected_steps) and step == expected_steps[i] else 0.0)
    return rewards

def normalize(xs, eps=1e-8):
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    std = sqrt(var + eps)
    return [(x - mean) / std for x in xs]

def grpo_objective(ratios, advantages, clip_eps=0.2):
    assert len(ratios) == len(advantages)
    terms = []
    for r, a in zip(ratios, advantages):
        clipped = min(max(r, 1 - clip_eps), 1 + clip_eps)
        terms.append(min(r * a, clipped * a))
    return sum(terms) / len(terms)

# 玩具例子：三条候选输出的 ORM 奖励
rewards = [
    orm_reward(51, 51),  # correct
    orm_reward(48, 51),  # wrong
    orm_reward(45, 51),  # wrong
]
advantages = normalize(rewards)
ratios = [1.3, 0.8, 1.5]
obj = grpo_objective(ratios, advantages, clip_eps=0.2)

assert rewards == [1.0, 0.0, 0.0]
assert advantages[0] > 0 and advantages[1] < 0 and advantages[2] < 0
assert obj > 0

# PRM 例子：三步推理里前两步对、第三步错
steps = ["split 17 into 10+7", "30+21", "48"]
expected = ["split 17 into 10+7", "30+21", "51"]
step_rewards = prm_reward(steps, expected)

assert step_rewards == [1.0, 1.0, 0.0]
```

如果把这个结构映射到真实训练，伪代码通常是下面这样：

```python
def compute_reward(sample, mode, verifier=None, prm_scorer=None):
    if mode == "orm":
        return verifier(sample.final_answer)  # 只看终点
    if mode == "prm":
        step_scores = []
        for step_index, step_text in enumerate(sample.steps):
            step_scores.append(prm_scorer(step_text, step_index, sample))
        return sum(step_scores)
    raise ValueError("unknown mode")

def train_step(prompt_batch, policy, ref_policy, mode):
    grouped_samples = []
    for prompt in prompt_batch:
        samples = policy.sample_group(prompt, group_size=G)
        rewards = [compute_reward(s, mode, verifier, prm_scorer) for s in samples]
        advantages = group_normalize(rewards)
        grouped_samples.append((prompt, samples, advantages))

    loss_rl = 0.0
    for prompt, samples, advantages in grouped_samples:
        for sample, adv in zip(samples, advantages):
            ratio = policy.logprob(sample, prompt).exp() / sample.old_prob
            clipped_ratio = clip(ratio, 1 - eps, 1 + eps)
            loss_rl += -min(ratio * adv, clipped_ratio * adv)
            loss_rl += beta * kl_to_ref(policy, ref_policy, prompt, sample)

    # PRM 路线常额外混合语言建模损失，避免推理奖励把语言能力拉坏
    loss = loss_rl
    if mode == "prm":
        loss += alpha * language_modeling_loss(policy, prompt_batch)

    return loss
```

这个抽象有两个好处：

| 设计点 | 作用 |
|---|---|
| `verifier` 独立 | 适合数学判题、单元测试、编译器执行这类规则可验任务 |
| `prm_scorer` 独立 | 可接人工标注器、规则模板或另一个奖励模型 |
| GRPO 主循环不变 | 便于在同一框架下切换 ORM 与 PRM |
| 可混合 LM loss | 防止 RL 后只会“拿分”，不会正常表达 |

真实工程里，ORM 常见实现是：

- 数学题用精确答案比对。
- 代码题用测试集通过率。
- 形式化证明用 proof checker 是否接受。

PRM 常见实现则更复杂：

- 人工逐步标注“这一步是否正确、是否必要”。
- 用单独训练的过程评分模型打分。
- 在搜索系统中，把中间状态价值估计也纳入奖励。

---

## 工程权衡与常见坑

奖励设计真正难的地方，不在概念区分，而在成本和失真。

先看最常见的坑：

| 坑 | 发生原因 | 典型表现 | 常见缓解策略 |
|---|---|---|---|
| 稀疏奖励 | ORM 只在终点给分 | 训练初期全是 0，模型学不到 | 增大组采样、做规则校验、加入长度控制或课程学习 |
| 早停截断 | PRM 前几步低分 | 模型不愿继续展开推理 | 限制 early stop，保留最小展开长度 |
| 隐性过程奖励被忽略 | ORM+GRPO 已在奖励前缀 | 调参时误判“为什么模型开始写长链” | 观察组内共享前缀和长度分布 |
| 奖励黑客 | 模型学会骗 verifier | 格式对但内容假，或钻评测漏洞 | 多 verifier、对抗样本、人工抽检 |
| 长度偏置 | 长链更容易偶然撞对或多得分 | 模型无节制变长 | 长度惩罚、长度归一化、分段奖励 |
| PRM 标注漂移 | 标注标准不稳定 | 同类步骤分数不一致 | 统一 rubric，多人复核，抽样校正 |

对新手最重要的是理解两个直观失败模式。

第一，ORM 的失败模式是“冷门题全是 0”。如果任务分布里很多题一开始根本答不对，那么模型看到的几乎都是失败样本。它容易退回最常见套路，比如机械套模板、输出熟悉但不真正求解的格式。这时 GRPO 的组内比较能缓解，但前提是**至少偶尔能采到正确样本**。完全采不到时，组内比较也无从谈起。

第二，PRM 的失败模式是“前面低分，后面就不写了”。例如一个证明题需要先定义变量、再列出引理、最后合并。如果 PRM 对前两步过苛刻，模型会学到“展开越多，越容易暴露错误”，从而提前收缩回答。工程上常要混合语言建模损失，或规定最短推理长度，避免这种过早保守。

这里还要单独说长度问题。R1 一类系统里，人们观察到模型会自发拉长推理。这不一定总是“更会思考”，有时只是因为更长的链条给了更多修正机会，或者组内比较更容易让某些长链偶然胜出。因此长度惩罚不是装饰项，而是稳定训练的重要部件。目标不是盲目压短，而是**防止“越长越占便宜”这个偏差吞掉真实质量信号**。

真实工程里，代码题是个很典型的 ORM 优势场景。因为它天然有单元测试、编译器、运行结果这些 verifier。你不需要请人逐行评判“这段思路是否优雅”，只要测试能过，奖励就足够硬。相反，在形式化数学证明里，中间 lemma 的选择、状态展开顺序、搜索分支价值都很关键，PRM 或价值估计就更容易体现收益。

---

## 替代方案与适用边界

如果把任务难度和资源约束一起看，奖励策略大致可以这么选：

| 任务难度 | 可用验证器 | 标注资源 | 推荐策略 |
|---|---|---|---|
| 低到中 | 强，可自动判分 | 低 | ORM 为主 |
| 中到高 | 强，但终局很稀疏 | 中 | ORM + GRPO + 长度/采样改进 |
| 高 | 强，且可定义中间状态质量 | 中到高 | PRM 或隐式/显式过程奖励结合 |
| 极高 | 需要树搜索或证明搜索 | 高 | PRM + 搜索，如 MCTS |
| 主观开放任务 | 弱 | 高 | 不适合直接套本文框架 |

可以把选择原则压成三条。

1. **终极指标明确、验证器强、预算有限**：优先 ORM。  
   例如代码生成、标准数学题、规则明确的竞赛问答。这里最大收益来自扩大采样、提高 verifier 可靠性、调好 GRPO 和长度控制，而不是先上昂贵 PRM。

2. **题目极难、错误往往发生在中间步骤、终局成功率极低**：考虑 PRM。  
   例如形式化证明、长链规划、深层搜索。终局奖励过稀时，只靠 ORM 很可能永远等不到足够多成功样本。

3. **先评估 ORM+GRPO 的隐式过程奖励够不够**。  
   这是现在很容易被忽略的边界。并不是一旦关心过程就必须显式上 PRM。很多任务里，组采样已经提供了部分过程反馈。若这部分信号够用，再额外建设 PRM 的边际收益可能不高。

AlphaProof 是最好的边界案例。它在 Lean 数学环境下结合 RL 和搜索，目标不是“普通题答对率再涨一点”，而是进入 IMO 级别难度。此时只看最终是否证明完成，往往太晚了，因为大部分尝试在很早的中间状态就已经注定失败。搜索配合过程价值判断，才能更有效分配算力。

而 R1 路线的重要意义在于另一侧边界：它证明了**不必先拥有昂贵 PRM，纯结果奖励也可能催生强推理行为**。这对工程团队很现实，因为多数团队先能拿到的是答案判分器，不是高质量逐步标注数据。

所以更实际的决策顺序通常是：

1. 先问任务能否被可靠验证。
2. 能验证时，先做 ORM 基线。
3. 若终局奖励太稀疏，再用 GRPO 的组采样、长度控制、课程学习去补。
4. 仍不够时，再投入 PRM 或搜索系统。

---

## 参考资料

1. DeepSeek 团队，[DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning](https://www.nature.com/articles/s41586-025-09422-z)，Nature，2025。给出 GRPO 目标、纯规则奖励与 ORM 实践。
2. DeepMind 团队，[Olympiad-level formal mathematical reasoning with reinforcement learning](https://www.nature.com/articles/s41586-025-09833-y)，Nature，2025。介绍 AlphaProof、形式化数学、RL 与搜索结合。
3. Michael Sullivan，[GRPO is Secretly a Process Reward Model](https://arxiv.org/abs/2509.21154)，arXiv，2025。讨论 GRPO 的隐式过程奖励效应。
4. Wikipedia，[Reasoning model](https://en.wikipedia.org/wiki/Reasoning_model)。用于 ORM 与 PRM 的入门定义对照。
