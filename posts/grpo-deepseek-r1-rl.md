## 核心结论

GRPO（Group Relative Policy Optimization，组相对策略优化）可以理解为“不给大模型再配一个价值网络，只在同一题的多份答案之间做相对比较，然后直接更新生成模型本身”的强化学习方法。术语里的“策略”就是当前模型的输出分布，“价值网络”就是专门预测答案好坏的辅助模型。

它的核心价值有三点：

| 维度 | SFT | 标准 PPO/RLHF | GRPO |
|---|---|---|---|
| 训练信号 | 标准答案 | reward + value/critic | 组内 reward 相对差 |
| 是否需要 critic | 否 | 是 | 否 |
| 参数量变化 | 基本不变 | 增加 actor + critic | 基本不变 |
| 适合任务 | 模仿已有答案 | 通用 RLHF | 数学/推理/可验证任务 |
| 主要成本 | 标注数据 | 训练复杂、显存更高 | 多次采样与验证 |

对初级工程师最重要的判断是：GRPO 不是 SFT 的替代品，而是 SFT 之后的增强步骤。先用 SFT 或 PEFT 把模型训到“会按格式答题、能写出基本推理”，再用 GRPO 强化“同一题里更优的那种解法”。如果基础模型连题都读不懂，GRPO 往往只会放大噪声。

一个最直观的玩具例子是：同一个 prompt 采样 3 个回答，reward 分别是 0.8、0.2、0.0，组均值是 0.33。于是第一条答案相对组均值更好，第二和第三更差。GRPO 就让第一条答案中每个 token 的梯度更容易被保留，后两条更容易被压低。这样不需要额外训练 critic，也能得到“相对优胜劣汰”的更新信号。

最终要点可以压缩成一句话：GRPO 用组内平均 reward 代替 critic，沿着 actor-only 的更新路径，在可自动验证的推理任务上，常能用更低系统复杂度换到不错的效果提升，但代价是生成成本上升，而且长度偏差、难度偏差会非常容易把训练带偏。

---

## 问题定义与边界

GRPO 解决的问题不是“让模型学会知识”，而是“在已有能力上，把正确推理轨迹的概率再推高一点”。这里的“推理轨迹”就是模型一步步写出的中间 token 序列，比如数学解题过程、代码推导步骤、结构化分析链路。

它的边界必须讲清楚：

| 场景 | 是否适合 GRPO | 原因 |
|---|---|---|
| 数学题、代码题、规则明确问答 | 适合 | 有自动 verifier，可直接判分 |
| 长链路推理 | 适合 | 组内对比能放大优质轨迹 |
| 开放式写作、情感表达 | 不太适合 | 很难稳定自动打分 |
| 依赖人工偏好排序 | 更适合 DPO/RLHF | GRPO 不直接消费 preference pair |
| 基础模型很弱 | 不适合直接上 | 多采样出来的大多是垃圾答案 |
| 只做小参数适配 | 可以 | LoRA/PEFT 后仍可继续做 GRPO |

这里的“verifier”指自动验证器，也就是一个能判断答案对错或质量的程序。数学任务里它可能是符号比对器；代码任务里它可能是单元测试；结构化抽取任务里它可能是规则匹配器。

一个新手常见误区是把 GRPO 理解成“只要有 reward 就能训练”。这不准确。GRPO 对 reward 的稳定性要求很高，因为它不训练 critic 来平滑估计，而是直接用组内相对分数驱动 token 更新。如果 verifier 分数波动大、存在大量误判，GRPO 会很快学到错误偏好。

所以它最适合的路径通常是：

1. 先做 SFT，让模型具备基本格式与任务理解。
2. 如需省参数，SFT 阶段可用 LoRA 等 PEFT。
3. 再在可验证任务上做 GRPO，强化高质量输出。
4. 最后单独评估收益是否覆盖生成与验证成本。

真实工程例子可以这样理解：一个 7B 数学模型已经通过 SFT 学会“读题、列式、给出 boxed answer”。这时继续加更多 SFT 数据，提升会越来越慢；但如果对每道题采样多个解答，用 Math verifier 判分，再用 GRPO 强化相对更好的解法，往往还能继续抬高 GSM8K、MATH 这类指标。它优化的是“解题策略分布”，不是“知识注入”。

---

## 核心机制与推导

GRPO 的目标函数可写成：

$$
J_{\text{GRPO}}(\theta)=\mathbb{E}_{q,\{o_i\}}\frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left[
\min\left(
r_{i,t}(\theta)\hat A_i,\;
\text{clip}(r_{i,t}(\theta),1-\epsilon,1+\epsilon)\hat A_i
\right)
-\beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})
\right]
$$

其中：

- $q$ 是 prompt。
- $G$ 是同一 prompt 采样出的候选答案数。
- $o_i$ 是第 $i$ 个候选答案。
- $|o_i|$ 是答案长度，也就是 token 数。
- $r_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})}$，表示新旧策略在该 token 上的概率比。
- $\hat A_i$ 是 advantage，直白说就是“这条答案比组平均好多少”。

GRPO 最关键的变化在 advantage：

$$
\hat A_i = r_i - \bar r,\qquad
\bar r=\frac{1}{G}\sum_{j=1}^G r_j
$$

这里的 $r_i$ 是 verifier 给整条答案的分数，$\bar r$ 是同组平均分。也就是说，GRPO 不问“这条答案绝对有多好”，而问“它比同题其他答案更好吗”。

玩具例子如下：

| 候选答案 | reward | 组均 reward | advantage |
|---|---:|---:|---:|
| A | 0.8 | 0.33 | +0.47 |
| B | 0.2 | 0.33 | -0.13 |
| C | 0.0 | 0.33 | -0.33 |

如果某个 token 在新策略下的概率相对旧策略升高，而该答案的 advantage 为正，这个 token 更新方向就是被鼓励的；如果 advantage 为负，则会被抑制。`clip` 的作用是限制更新幅度，避免一步走太远，训练发散。KL penalty 则限制当前策略不要偏离 reference model 太多，避免模型为了追 reward 迅速失去可读性和泛化性。

从参数更新路径看，GRPO 很像 PPO，但少了 value head。路径可以概括为：

| 步骤 | 标准 PPO | GRPO |
|---|---|---|
| 生成候选 | 需要 | 需要 |
| 打分 | reward model / 环境 | verifier / reward |
| baseline 来源 | critic/value net | 组均 reward |
| 更新对象 | actor + critic | actor only |
| 稳定器 | clip + KL + value loss | clip + KL |

这就是为什么它常被描述为“actor-only 的 PPO 风格方法”。它没有把系统复杂度降到零，只是把“训练一个可靠 critic”的难题换成了“设计一个可靠 verifier，并承担多采样成本”。

---

## 代码实现

下面给一个可运行的极简 Python 版本，只演示 GRPO 的核心计算，不涉及真实大模型前向传播。代码中的“概率比”模拟 PPO-style clip，“奖励”模拟 verifier 输出。

```python
from math import isclose

def grpo_objective(rewards, ratios, lengths, eps=0.2, beta=0.0, kl_values=None):
    """
    rewards: 每个候选答案的整句 reward
    ratios:  每个候选答案在各 token 上的新旧策略概率比
    lengths: 每个候选答案的 token 长度
    kl_values: 每个候选答案的平均 KL 惩罚
    """
    assert len(rewards) == len(ratios) == len(lengths)
    G = len(rewards)
    baseline = sum(rewards) / G
    advantages = [r - baseline for r in rewards]

    if kl_values is None:
        kl_values = [0.0] * G

    total = 0.0
    for adv, token_ratios, length, kl in zip(advantages, ratios, lengths, kl_values):
        assert len(token_ratios) == length
        token_sum = 0.0
        for ratio in token_ratios:
            clipped = min(max(ratio, 1 - eps), 1 + eps)
            surrogate = min(ratio * adv, clipped * adv)
            token_sum += surrogate - beta * kl
        total += token_sum / length
    return total / G, advantages

# 玩具例子
rewards = [0.8, 0.2, 0.0]
lengths = [3, 2, 4]
ratios = [
    [1.10, 1.25, 0.95],   # 高分答案，大多数 token 被鼓励
    [0.90, 1.05],         # 中间答案，轻微压制
    [0.80, 0.85, 1.10, 1.00],  # 低分答案，整体应被压制
]

obj, adv = grpo_objective(rewards, ratios, lengths, eps=0.2)

assert len(adv) == 3
assert isclose(sum(adv), 0.0, abs_tol=1e-9)
assert adv[0] > 0 and adv[1] < 0 and adv[2] < 0
assert obj != 0.0

print("advantages:", adv)
print("objective:", round(obj, 6))
```

这段代码对应的训练逻辑可以概括成下面的伪代码：

```python
for prompt in batch:
    candidates = sample_model(prompt, num_samples=G)
    rewards = [verifier(prompt, c) for c in candidates]
    baseline = mean(rewards)
    advantages = [r - baseline for r in rewards]

    for candidate, adv in zip(candidates, advantages):
        for token in candidate.tokens:
            ratio = pi_theta(token) / pi_old(token)
            clipped_ratio = clip(ratio, 1 - eps, 1 + eps)
            token_loss += -min(ratio * adv, clipped_ratio * adv)

        token_loss += beta * kl_to_reference(candidate)

update_actor(token_loss)
```

真实工程里，数据格式通常至少需要以下字段：

| 字段 | 含义 | 是否必须 |
|---|---|---|
| `prompt` | 输入问题 | 必须 |
| `samples` | 同题采样得到的多个输出 | 训练时产生 |
| `reward` | 每个输出的 verifier 分数 | 必须 |
| `old_logprob` | 旧策略下 token 对数概率 | 必须 |
| `ref_logprob` | reference 模型对数概率 | 常用 |
| `attention_mask` | 有效 token 位置 | 必须 |
| `completion_length` | 输出长度 | 常用 |

如果是 PEFT 场景，比如 LoRA，更新的仍然只是 actor 上挂的低秩适配参数。也就是说，GRPO 不要求你再新增一套 critic 参数；参数规模通常仍接近 SFT 阶段，只是训练时的采样、缓存、打分流水线明显更重。

---

## 工程权衡与常见坑

GRPO 最容易被误判的地方是“省了 critic，所以更便宜”。这句话只对了一半。它省的是模型结构和训练链路复杂度，不是端到端算力。

真实成本主要来自三部分：

1. 同一 prompt 要生成 $G$ 个候选，推理成本近似放大为原来的 $G$ 倍。
2. 每个候选都要过 verifier，验证成本也随之放大。
3. 训练时还要保存 old policy、reference policy、reward、长度等信息，内存和 I/O 压力都更高。

所以工程上真正要问的不是“GRPO 能不能涨点”，而是“涨的点数是否值回额外生成和验证成本”。以数学推理为例，如果你只需要一个轻量 FAQ 模型，GRPO 往往不划算；如果你要冲榜、做高价值推理模块，且 verifier 很强，它才有意义。

最常见的性能退化来源有三类。

第一类是长度偏差。原始 GRPO 常按输出长度做归一化，即把整条答案的贡献除以 $|o_i|$。这会导致一个反直觉现象：很长但错误的答案，平均到每个 token 后，负信号被摊薄，惩罚反而不够。结果就是模型逐渐学会“写得更长，即使错也不太亏”。

| 问题 | 表现 | 后果 |
|---|---|---|
| 长度归一化过强 | 长错误答案惩罚变小 | 输出越来越长，准确率不稳 |
| reward 方差过大 | 组内 advantage 噪声大 | 训练震荡，loss 降但指标不涨 |
| verifier 不可靠 | 错答被打高分 | 学到错误偏好 |
| 基线模型太弱 | 候选都很差 | 相对比较失真，强化噪声 |

第二类是难度偏差。如果训练集里有很多极易题或极难题，组均 baseline 会变得不稳定。极易题里候选答案都高分，advantage 差异很小，训练信号弱；极难题里大家都低分，偶然噪声会被放大。后续一些改进方法会引入标准差归一化等处理，本质上是在做 reward 尺度校正。

第三类是 reference 约束过松或过紧。KL 系数太小，模型可能快速偏离 SFT 分布，出现格式崩坏、胡乱冗长、投机性模式；KL 系数太大，又会让 GRPO 形同虚设，几乎学不到东西。

原始 GRPO 与后续修正思路可以粗看成这样：

| 方案 | 长度处理 | 优点 | 风险 |
|---|---|---|---|
| 原始 GRPO | 常见做法是按长度平均 | 实现简单 | 长错误答案惩罚可能被稀释 |
| 带标准差规整的变体 | reward 做标准化 | 减少组间尺度波动 | 小 batch 时不稳定 |
| 去掉不合理长度缩放的修正 | 更强调真实 token 贡献 | 抑制“越错越长” | 需重新调学习率和 KL |

如果用一个真实工程例子来理解：某个数学模型在 SFT 后已经能做出不少正确答案。上线 GRPO 后，离线 loss 看起来持续下降，但线上发现模型回答变长、解释更多、最终答案反而更容易错。这通常不是“强化学习没用”，而是 length normalization、reward 设计或 KL 约束出了问题。看 loss 不够，必须同时看准确率、平均输出长度、拒答率、格式错误率。

---

## 替代方案与适用边界

GRPO 不是唯一选择。它和 SFT、DPO、标准 PPO、PEFT 的关系最好分开看。

| 方法 | 需要的信号 | 是否需要 critic | 参数变化 | 更适合什么 |
|---|---|---:|---:|---|
| SFT | 标准答案 | 否 | 小 | 基础指令跟随 |
| DPO | 偏好对 `(chosen, rejected)` | 否 | 小 | 人类偏好对齐 |
| 标准 PPO/RLHF | reward + value | 是 | 大 | 通用强化学习对齐 |
| PEFT | 不是训练目标，是参数方式 | 否 | 很小 | 低成本适配 |
| GRPO | 同题多样本 reward | 否 | 小 | 可验证推理任务 |

几个关键区别：

第一，GRPO 和 DPO 的差异在于监督信号来源。DPO 用的是“偏好对”，即人工或系统告诉你 A 比 B 好；GRPO 用的是“同题多采样 + verifier 打分”，更像真正的 RL。若你的任务天然有 preference pair，比如对话风格、摘要偏好，DPO 往往更直接。

第二，GRPO 和标准 PPO 的差异在于是否引入 critic。标准 PPO 通常更通用，但链路更复杂，训练更重；GRPO 牺牲了一部分估计能力，换来更轻的 actor-only 结构。前提是你的任务里组均 reward 足够当 baseline。

第三，PEFT 与 GRPO 不冲突。PEFT（参数高效微调）只是“怎么改参数”，比如只训练 LoRA；GRPO 是“用什么目标训练”。工程上常见组合是：`Base Model -> SFT/LoRA -> GRPO/LoRA`。这样参数不大，但训练目标从模仿切到相对奖励优化。

因此，适用边界可以简单归纳为：

- 如果你有高质量标准答案，先做 SFT。
- 如果你有偏好数据，优先看 DPO。
- 如果你有稳定 verifier，且任务是数学、代码、规则推理，GRPO 值得尝试。
- 如果你既没有 verifier，也没有偏好对，只能人工主观打分，GRPO 通常不稳。
- 如果模型还没学会基础任务，不要跳过 SFT 直接上 GRPO。

真正的判断标准不是“GRPO 先进不先进”，而是“你的任务能不能稳定地产生组内相对优劣”。没有这个前提，GRPO 就会退化成高成本噪声放大器。

---

## 参考资料

- DeepSeekMath 相关解析：重点看 GRPO 的目标函数、组均 reward baseline、SFT 到 RL 的连接方式。适合先建立整体框架。
- Aman 的 preference optimization primer：重点看 GRPO 与 PPO、DPO 的关系，以及长度偏差、难度偏差、训练配置等工程问题。
- Chan Kha Vu 关于 DeepSeek R1 改进的笔记：重点看长答案偏差、训练退化现象和修正思路，适合理解“为什么 loss 下降但效果变差”。
- 一些 SFT/PEFT/GRPO 对比文章：价值主要在于帮助区分“参数高效方法”和“训练目标方法”不是同一维度。
- 如果做代码或数学 verifier，建议同时查对应 benchmark 的官方评测规则，因为 reward 定义直接决定 GRPO 学到什么。
