## 核心结论

Nash Learning from Human Feedback，简称 NLHF，可以理解为“把对齐训练写成一个博弈问题”。博弈的意思是：不是只让模型去追一个固定评分器的高分，而是同时考虑“如果评分方式继续变化，当前策略还能不能站得住”。

传统 RLHF 的核心路径是“固定一个奖励模型，然后优化策略”。这类方法能把分数拉高，但有一个结构性风险：模型可能学会迎合当前 judge，而不是学会在更一般的偏好标准下稳定地做好任务。NLHF 的目标正是缓解这个问题。它把“当前策略 $p$”和“可能挑战它的候选策略 $q$”放进同一个偏好 oracle 里比较，再通过极小极大优化寻找一个稳定点：

$$
\min_{p}\max_{q}\ \mathbb{E}_{x \sim \mathcal{D}}\left[\ell_{\text{oracle}}(p,q\mid x) + \beta D_{\mathrm{KL}}(q\|p)\right]
$$

这里的 oracle 可以白话理解为“成对比较谁更好的裁判器”；$D_{\mathrm{KL}}$ 是 KL 散度，可以白话理解为“两个概率分布差多远的距离度量”。这个目标的含义不是“把某个答案打到最高分”，而是“让任何偏离当前策略的对手都很难在总体上击败它”。因此，NLHF 追求的是一个 $\beta$-正则化的纳什均衡，而不是单次评测上的局部最优。

一个玩具例子可以直接说明区别。假设同一个提示下，策略 $p_1$ 和 $p_2$ 的回答拿去给 judge 比较，$p_1$ 被判更好的概率是 0.7，$p_2$ 是 0.3。传统方法会想办法继续把 $p_1$ 推到更高分；NLHF 则进一步追问：如果有人构造一个略微不同的对手 $q$，它能不能专门利用 judge 的偏好漏洞反超 $p_1$？如果能，说明 $p_1$ 并不稳定；如果任何 $q$ 的优势都被 KL 代价抵消，才接近均衡。

---

## 问题定义与边界

NLHF 解决的问题，不是“如何找到一个高分回答”，而是“如何找到一个在评分器更新、候选策略变化、提示分布波动下仍然稳定的策略”。这里的“稳定”不是口头描述，而是有明确博弈含义：当前策略已经足够好，以至于别的策略即使尝试偏离，也很难在综合收益上占到便宜。

这个问题成立，需要先明确边界。

| 维度 | NLHF 关注什么 | 不解决什么 | 关键前提 |
|---|---|---|---|
| 奖励稳定性 | 减少对固定 judge 的过拟合 | 不保证绝对真实的人类价值 | oracle 至少能做相对可靠的 pairwise 比较 |
| 策略更新 | 关注策略与挑战者之间的对抗稳定 | 不直接等价于全局最优文本质量 | 候选策略采样要覆盖有代表性的对手 |
| 分布约束 | 用 KL 防止策略飘离参考模型 | 不自动解决数据分布外失真 | reference 模型和训练模板要一致 |
| 工程流程 | 适合成对比较式偏好学习 | 不适合完全没有偏好信号的场景 | prompt 模板、tokenizer、judge 输入格式要统一 |

这里的 pairwise preference oracle，白话说就是“它不一定会给一个绝对分，但它能比较 A 和 B 谁更好”。这和很多奖励模型只输出单个分数不同。NLHF 更依赖这种“你和对手相比怎么样”的信号，因为它本质上在求稳定的对抗平衡。

再看一个新手版直觉例子。把 judge 想成会逐渐更新口味的裁判。单向优化的目标是“让模型讨好当前版本的裁判”；NLHF 的目标是“即使裁判更新一点点、或者来了另一个风格接近的裁判，模型依然不容易被新的竞争者打败”。所以它优化的是动态稳定性，而不是静态高分。

它也有清晰边界。第一，oracle 如果本身经常判错，博弈求得再稳定，也只是“对错误裁判稳定”。第二，KL 约束如果太弱，策略会为了击败候选者而迅速偏离基座模型；如果太强，又会让策略几乎不动。第三，prompt template 不统一时，judge 比较的是不同格式下的回答，训练信号会直接失真。

---

## 核心机制与推导

NLHF 的关键转换，是把“最大化奖励”改写成“对任意候选对手保持优势”。

设 $x$ 是提示，$p(\cdot|x)$ 是当前策略，$q(\cdot|x)$ 是候选挑战策略。oracle 会比较从 $p$ 和 $q$ 采样出的回答，给出一个谁更优的偏好损失 $\ell_{\text{oracle}}(p,q\mid x)$。如果把这个量看成“当前策略面对挑战者时暴露出的弱点”，那么训练目标就是让这些弱点在所有可能的 $q$ 上都尽量小。

于是得到：

$$
\min_{p}\max_{q}\ \mathbb{E}_{x}\left[\ell_{\text{oracle}}(p,q\mid x)+\beta D_{\mathrm{KL}}(q\|p)\right]
$$

这个式子里，$\max_q$ 表示“替当前策略寻找最难对付的挑战者”；$\min_p$ 表示“更新当前策略，让这种最坏情况也尽量变小”。这正是极小极大问题。

为什么 KL 项写成 $D_{\mathrm{KL}}(q\|p)$ 很重要？直观上，它惩罚的是“对手 $q$ 为了赢你，要偏离你多远”。如果一个挑战者只靠非常激进、非常不自然的偏离才能赢，那这种胜利不值得被鼓励。KL 在这里相当于一个正则器，白话就是“给偏离行为加成本”，防止训练走向不受控的模式崩塌。

可以把机制写成一个简化流程：

1. 当前策略 $p$ 在一批 prompt 上生成回答。
2. 构造若干候选对手 $q$，也在同样 prompt 上生成回答。
3. oracle 对每组 $(p,q)$ 做成对比较，估计谁更受偏好。
4. 计算 oracle 损失加上 KL 代价。
5. 更新 $q$ 去寻找当前 $p$ 的弱点。
6. 更新 $p$ 去修补这些弱点。
7. 当任何合理的 $q$ 都难以获得净优势时，接近纳什均衡。

玩具例子可以算得更具体一些。假设提示集合只有一个 $x$，策略空间只有两个回答 A 和 B。当前策略 $p$ 以 0.8 概率输出 A，0.2 概率输出 B；对手 $q$ 以 0.4 概率输出 A，0.6 概率输出 B。oracle 判断 A 胜过 B 的概率是 0.7。此时 $q$ 想通过增加 B 的比例来利用某个 judge 偏好漏洞，但如果它离 $p$ 偏得太远，KL 项会迅速增大。训练的平衡点不是“把 A 概率推到 1”，而是找到一个分布，使任何替代分布的净收益都不再明显更高。

真实工程例子更接近 Hugging Face TRL 的 `Nash-MD Trainer`。在这个流程里，SFT 模型是初始策略，PairRMJudge 之类的成对偏好模型是 oracle，UltraFeedback 一类数据集提供 prompt。每轮训练不是只算“回答得了多少分”，而是要同时看“当前回答相对于候选回答赢多少”和“候选为了赢付出了多少分布偏移成本”。这比普通 PPO 式 RLHF 多了一层“显式对手”。

---

## 代码实现

下面先给一个最小可运行的玩具实现。它不训练大模型，只演示 NLHF 目标里的两部分：oracle 比较项和 KL 正则项。代码中的分布只有两个动作，但足以看到“挑战者是否值得偏离”的逻辑。

```python
import math

def kl_div(q, p):
    eps = 1e-12
    return sum(qi * math.log((qi + eps) / (pi + eps)) for qi, pi in zip(q, p))

def oracle_loss(p, q, pref_a_over_b=0.7):
    # 两个动作: A, B
    # 近似地把“对手赢当前策略”的概率写成一个简单比较项
    p_a, p_b = p
    q_a, q_b = q
    p_win = p_a * q_b * pref_a_over_b + p_b * q_a * (1 - pref_a_over_b)
    q_win = q_a * p_b * pref_a_over_b + q_b * p_a * (1 - pref_a_over_b)
    return q_win - p_win

def objective(p, q, beta=0.5):
    return oracle_loss(p, q) + beta * kl_div(q, p)

p = [0.8, 0.2]
q_close = [0.7, 0.3]
q_far = [0.2, 0.8]

obj_close = objective(p, q_close, beta=0.5)
obj_far = objective(p, q_far, beta=0.5)

# 远离 p 的挑战者虽然可能更会“投机”，但 KL 成本更大
assert obj_far > obj_close

# 概率分布应归一化
assert abs(sum(p) - 1.0) < 1e-9
assert abs(sum(q_close) - 1.0) < 1e-9
assert abs(sum(q_far) - 1.0) < 1e-9
```

这个例子里，`oracle_loss` 是一个极简近似，只为说明结构：对手 $q$ 的收益不只看它能不能赢，还要减去偏离当前策略的代价。真实系统里，oracle 往往是单独训练过的 preference model，输入是同一 prompt 下两份回答，输出偏好概率或 margin。

如果把它扩展到工程训练，伪代码通常长这样：

```python
# prompts 来自偏好数据集
for batch in prompts:
    # 1. 当前策略生成回答
    y_p = policy.generate(batch)

    # 2. 候选策略/混合策略生成挑战回答
    y_q = challenger.generate(batch)

    # 3. pairwise judge 比较 (x, y_p) 和 (x, y_q)
    margins = judge.compare(batch, y_p, y_q)   # 越大表示 p 越占优

    # 4. 计算 KL 惩罚
    kl_penalty = kl(challenger.logprobs(batch, y_q),
                    policy.logprobs(batch, y_q))

    # 5. 组合成博弈目标
    # mixture_coefficient 用于混合若干候选来源或奖励项
    reward = -margins + beta * kl_penalty
    loss_p = minimize_against_worst_q(reward)
    loss_q = maximize_attack_value(reward)

    # 6. 分别更新 p 和 q
    policy.step(loss_p)
    challenger.step(loss_q)
```

实际配置里，几个参数最关键：

| 配置项 | 作用 | 过小的风险 | 过大的风险 |
|---|---|---|---|
| `beta` | KL 正则强度，控制策略偏离成本 | 模型容易为赢 judge 而漂移 | 模型几乎不学习新偏好 |
| `mixture_coefficient` | 混合候选或奖励来源的权重 | 对手覆盖不足，博弈太弱 | 训练噪声变大，收敛变慢 |
| `missing_eos_penalty` | 对缺失 EOS 的输出加罚，控制回答收尾 | 输出拖尾、被 judge 错判 | 过早截断，信息不完整 |

真实工程例子可以这样理解。比如你用一个 0.5B 级别的 SFT 模型做起点，用 PairRMJudge 作为成对裁判，再用 UltraFeedback 风格的 prompt 数据训练。每个 step 都会抽 prompt、生成多组候选、让 judge 评估相对优劣，再通过 `beta` 和混合策略控制训练方向。和普通 RLHF 最大的差别，不在优化器名字，而在“是否显式建模挑战者”。

---

## 工程权衡与常见坑

NLHF 比单向 RLHF 更稳定，但代价也更直接：候选更多、比较更多、日志更复杂、配置更敏感。它不是“把 PPO 换个名字”，而是把训练目标从单边优化升级成双边博弈。

最常见的坑如下：

| 常见坑 | 现象 | 原因 | 规避方式 |
|---|---|---|---|
| 模板不一致 | judge 评分波动异常 | policy 与 reward model 的 chat template 不同 | 统一 prompt 包装、角色标签、EOS 规则 |
| tokenizer 不一致 | 同一句话评分失真 | token 切分不同，logprob 和文本边界不对应 | policy、reference、judge 尽量共享 tokenizer |
| 候选过弱 | 训练很快“稳定”但泛化差 | $q$ 不足以暴露真实弱点 | 增加候选来源或提高采样多样性 |
| KL 过弱 | 输出风格发散、偏离基座模型 | 只追求击败 judge | 提高 `beta`，并监控 reference 漂移 |
| KL 过强 | 分数几乎不涨 | 对手和当前策略都被锁死 | 逐步调参，不要一开始设太大 |
| EOS 控制差 | 回答过长或截断怪异 | judge 把长度问题误当质量问题 | 使用 `missing_eos_penalty` 并检查终止规则 |

对新手最容易踩的坑，是“用错尺子”。如果 SFT 模型输出的是一种聊天模板，而 reward model 或 judge 看到的是另一种模板，那么比较结果根本不是“回答质量比较”，而是“格式差异比较”。这会让训练方向从第一步就歪掉。

另一个常见误区，是只看单个 reward 均值，不看相对 margin。NLHF 的关键不只是“我得了几分”，而是“我相对挑战者是否保持优势”。因此日志里至少要盯这些指标：

- `loss/score`：总目标是否在改善。
- `rewards/margins`：当前策略相对候选策略的胜负边际。
- `beta`：KL 正则是否仍处在合理量级。
- `missing_eos_penalty`：长度控制是否影响评估。
- 候选多样性指标：不同来源的 $q$ 是否真的提供了挑战。

如果这些指标出现矛盾，比如 `loss/score` 上升但 `rewards/margins` 下降，就要怀疑模型是在“学会绕过打分形式”，而不是学会更稳定地完成任务。

---

## 替代方案与适用边界

NLHF 不是所有场景的默认答案。它的价值在于“对 judge drift 更稳”，代价在于“训练更重、实现更复杂”。

| 方案 | 对 judge drift 的稳健性 | 训练资源 | 样本效率 | 实现复杂度 | 适合场景 |
|---|---|---|---|---|---|
| 标准 RLHF / PPO | 低到中 | 中 | 中 | 中 | reward model 相对稳定、先追求可用结果 |
| RLHF + KL | 中 | 中 | 中 | 中 | 需要控制模型漂移，但不做显式博弈 |
| NLHF / Nash-MD | 高 | 高 | 视候选质量而定 | 高 | 担心过拟合 judge，需要动态稳定性 |

如果只有一个比较稳定的奖励模型，而且资源有限，标准 RLHF 加适度 KL 通常更实际。它的逻辑更简单，调试面也更窄。只有当你明确观察到“模型很会迎合 judge，但换个 judge 或换个提示分布就掉性能”时，NLHF 的额外成本才更容易被证明是值得的。

适用边界可以概括成几条：

- 适合有可靠 pairwise oracle 的场景。
- 适合 prompt 分布较广、judge 漂移真实存在的场景。
- 适合能承受多候选采样和多次 judge 调用的训练预算。
- 不适合完全没有偏好比较信号的任务。
- 不适合候选池极小、对手构造能力很弱的场景。
- 不适合 tokenizer、模板、reference 模型都还没对齐的早期粗糙流水线。

可以把三种方法的差别记成一句话：标准 RLHF 是“追分”，RLHF+KL 是“追分但别跑太偏”，NLHF 是“在允许对手挑战的前提下，找到一个不容易被击败的稳定策略”。

---

## 参考资料

1. Munos et al., *Nash Learning from Human Feedback*, ICML 2024, PMLR 235:36743-36768。重点：给出 NLHF 的定义、纳什均衡视角，以及 Nash-MD 类算法框架。
2. Emergent Mind 对 arXiv:2402.07314 的整理页面。重点：解释带 reverse-KL 正则的极小极大目标，以及在线/离线求解的直观推导。
3. Hugging Face TRL 文档 `Nash-MD Trainer`。重点：提供工程实现入口、训练配置、指标说明，以及与 PairRMJudge 等组件的衔接方式。
