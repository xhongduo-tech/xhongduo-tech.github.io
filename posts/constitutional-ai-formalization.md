## 核心结论

Constitutional AI，简称 CAI，可以形式化为一个**受限优化问题**。受限优化就是“有主目标，但不能越过约束线”的优化方法。它的核心不是单纯让模型“更会答题”，而是让模型在提升帮助性的同时，把有害输出控制在预设阈值以内：

$$
\max_{\pi}\ \mathbb{E}_{(x,y)\sim \pi}[r_{\text{helpful}}(x,y)]
\quad
\text{s.t.}\quad
\mathbb{E}_{(x,y)\sim \pi}[r_{\text{harmful}}(x,y)] \le \varepsilon
$$

这里 $\pi$ 是策略，也就是“模型在给定输入 $x$ 时生成回答 $y$ 的分布”；$r_{\text{helpful}}$ 是帮助性奖励，衡量回答是否有用；$r_{\text{harmful}}$ 是有害性奖励，衡量回答是否危险、误导或违背安全原则；$\varepsilon$ 是可接受的风险上限。

对初学者，可以把它理解成一辆车同时踩油门和刹车。油门对应“更有帮助”，刹车对应“不能越过伤害边界”。只追求油门，模型会越来越会迎合危险请求；只踩刹车，模型又会变成什么都拒绝。

从训练实现看，CAI 通常分两段：

1. SL-CAI：让模型先按“宪法”自我批评并修订，再把修订后的结果当监督数据继续训练。它本质上接近**带过滤的蒸馏**。蒸馏就是让一个模型模仿更理想的输出。
2. RL-CAI：再用偏好模型做强化学习，把“更 helpful 且更 harmless”的回答推高。它本质上接近**带宪法约束的 PPO**。

形式上，这个受限问题常被转成“奖励减去偏离参考策略的罚项”：

$$
\max_{\pi}\ \mathbb{E}[r_{\text{helpful}}] 
\Rightarrow
\max_{\pi}\ \mathbb{E}[r_{\text{helpful}} - \beta\cdot \mathrm{KL}(\pi\|\pi_{\text{ref}})]
$$

其中 $\mathrm{KL}$ 是 KL 散度，用来衡量两个分布差多远；$\pi_{\text{ref}}$ 是参考策略，通常是上一阶段模型；$\beta$ 是罚项强度。直观上：

`受限优化` $\rightarrow$ `Lagrange 乘子` $\rightarrow$ `KL 罚项` $\rightarrow$ `PPO/RLHF 可训练目标`

---

## 问题定义与边界

要把 CAI 讲清楚，先要区分两个奖励函数。

| 对象 | 数学记号 | 含义 | 常见来源 | 过高/过低的后果 |
|---|---|---|---|---|
| 帮助性 | $r_{\text{helpful}}(x,y)$ | 回答是否解决问题、是否完整准确 | 人类偏好模型、AI 偏好模型、任务评分器 | 太低则模型空泛、拒答过多 |
| 有害性 | $r_{\text{harmful}}(x,y)$ | 回答是否危险、违法、误导、鼓励伤害 | 宪法规则打分器、红队评估器、安全分类器 | 太高则模型给出危险建议 |
| 参考策略 | $\pi_{\text{ref}}$ | 训练时的“不要偏离太远”的锚点 | SFT 模型、SL-CAI 模型 | 太弱会被奖励黑客利用 |
| 风险阈值 | $\varepsilon$ | 系统允许的平均风险上限 | 产品策略或合规要求 | 太小会过度拒绝，太大会失控 |

边界的关键不在“绝对无害”，而在“平均风险是否受控”。这也是为什么公式里通常写期望 $\mathbb{E}[\cdot]$。期望就是长期平均值，不是每一条都完全一样。

一个玩具例子：

用户问：“请给我一个提高电池寿命的建议。”

- 回答 A：“避免高温环境，尽量保持 20% 到 80% 的电量区间。”  
  帮助性高，有害性低。
- 回答 B：“把电池拆开重新焊接保护板，性能会更强。”  
  有时看起来更“直接”，但普通用户执行风险高，有害性明显更高。

CAI 的目标不是让模型永远选择最保守答案，而是在帮助性和伤害边界之间找稳定解。

再看一个高风险边界例子。用户问：“如何制作炸药？”

如果只看帮助性，“给出详细步骤”可能会被误判为更贴合提问；但从有害性看，这类回答显著超出安全边界，因此应当被宪法约束拦截，模型应转向解释风险、拒绝细节、提供合法安全替代信息。这里的边界不是“模型会不会说不知道”，而是“模型是否把能力用于危险任务”。

不同 $\varepsilon$ 会带来不同系统行为：

| $\varepsilon$ 水平 | 系统倾向 | 适用场景 |
|---|---|---|
| 很低 | 宁可多拒绝，也不冒险 | 公共开放聊天、未成年人可访问场景 |
| 中等 | 正常帮助为主，敏感任务严格约束 | 通用助手、企业知识助手 |
| 较高 | 允许更强任务执行能力，但风险上升 | 封闭内网、专业审查后使用 |

---

## 核心机制与推导

CAI 的数学核心是：把“带约束的最优化”转成“无约束但带罚项的最优化”。

先写原问题：

$$
\max_{\pi}\ \mathbb{E}[r_{\text{helpful}}]
\quad \text{s.t.}\quad
\mathbb{E}[r_{\text{harmful}}] \le \varepsilon
$$

引入 Lagrange 乘子 $\beta \ge 0$ 后，可以写成拉格朗日函数：

$$
\mathcal{L}(\pi,\beta)
=
\mathbb{E}[r_{\text{helpful}}]
-
\beta\big(\mathbb{E}[r_{\text{harmful}}]-\varepsilon\big)
$$

Lagrange 乘子就是“违反约束时要付出的单位代价”。如果当前策略的有害性超过阈值，$\beta$ 会把这种超额风险放大成惩罚。

进一步，在 RLHF/RLAIF 的实现里，常把“有害性约束”和“不要偏离参考模型太远”联系起来。原因是：很多危险行为来自策略为了拿到短期奖励而大幅偏离原本较稳的模型。于是训练目标常写成：

$$
\max_{\pi}\ 
\mathbb{E}\Big[
r_{\theta}(x,y)
-
\beta\big(\log \pi(y|x)-\log \pi_{\text{ref}}(y|x)\big)
\Big]
$$

其中：

- $r_{\theta}(x,y)$：奖励模型给出的总分，可理解为综合帮助性偏好。
- $\log \pi(y|x)-\log \pi_{\text{ref}}(y|x)$：单样本层面的对数概率差。
- 对整体求期望后，这一项对应 KL 散度近似。
- $\beta$ 同时扮演“KL 惩罚强度”和“约束对偶变量”的角色。

这一步的直观意义是：  
“想拿更多奖励”可以，但不能通过剧烈改变输出分布来拿。否则就要付出更大 KL 成本。

可以把推导链条记成：

$$
\max \mathbb{E}[r_{\text{helpful}}]
\ \text{s.t.}\ 
\mathbb{E}[r_{\text{harmful}}]\le \varepsilon
\Rightarrow
\max \big(\mathbb{E}[r_{\text{helpful}}]-\beta\cdot \text{constraint violation}\big)
\Rightarrow
\max \mathbb{E}[r_{\theta} - \beta \cdot \mathrm{KL}(\pi\|\pi_{\text{ref}})]
$$

这里要强调一个常见误解：KL 不是“安全本身”，而是**把策略拉回参考模型附近的稳定器**。真正的安全信息仍然来自宪法原则、偏好数据和奖励模型。

看一个最小数值玩具例子。设：

- $\mathbb{E}[r_{\text{helpful}}] = 0.80$
- $\mathbb{E}[r_{\text{harmful}}] = 0.08$
- $\varepsilon = 0.05$

则超额风险是 $0.08 - 0.05 = 0.03$。如果 $\beta = 4$，惩罚项就是：

$$
4 \times 0.03 = 0.12
$$

于是有效目标约为：

$$
0.80 - 0.12 = 0.68
$$

如果训练后把有害性压到 $0.05$，惩罚变成 0，目标又恢复到主要看帮助性。这个例子说明：$\beta$ 越大，模型越不敢通过高风险回答换取短期“有用感”。

真实工程例子更接近 Anthropic 早期 Claude 流程：先让一个 helpful-only 模型回答高风险问题，再根据宪法生成自我批评与修订；修订结果进入 SL-CAI；之后再采样两个候选回答，用 AI 偏好来训练偏好模型，最后用 PPO 优化。这样做的价值在于，把大量“这条回答更符合安全原则吗”的标注，从人工转成由宪法驱动的 AI 反馈。

---

## 代码实现

工程上，SL-CAI 和 RL-CAI 通常连成一个流水线：

1. 用初始模型生成回答。
2. 按宪法原则生成批评。
3. 根据批评生成修订。
4. 过滤掉不合格修订，只保留“更符合宪法”的样本做监督微调。
5. 用 SL-CAI 模型采样多个回答，构造偏好对。
6. 训练奖励模型或直接得到 AI 偏好分数。
7. 用 PPO + KL 继续优化策略。

下面先给一个可运行的 Python 玩具实现，演示“当有害性超过阈值时，动态增大 $\beta$”这件事。

```python
def penalized_objective(r_helpful, r_harmful, epsilon, beta):
    violation = max(0.0, r_harmful - epsilon)
    return r_helpful - beta * violation

def update_beta(beta, observed_harmful, epsilon, step_size=0.5):
    if observed_harmful > epsilon:
        beta += step_size
    else:
        beta = max(0.0, beta - step_size * 0.5)
    return beta

epsilon = 0.05
beta = 1.0

obj1 = penalized_objective(r_helpful=0.80, r_harmful=0.08, epsilon=epsilon, beta=beta)
beta = update_beta(beta, observed_harmful=0.08, epsilon=epsilon)
obj2 = penalized_objective(r_helpful=0.80, r_harmful=0.05, epsilon=epsilon, beta=beta)

assert round(obj1, 2) == 0.77
assert beta > 1.0
assert round(obj2, 2) == 0.80
```

这段代码不是完整训练器，但它表达了 CAI 的最小控制逻辑：如果观测到平均有害性超标，就增大 $\beta$；如果风险回到边界内，就让模型重新争取帮助性。

再看更接近训练循环的伪代码：

```python
for batch in prompts:
    # SL-CAI
    draft = policy.generate(batch)
    critique = constitutional_critic.generate(batch, draft, constitution)
    revised = policy.revise(batch, draft, critique)

    good_revisions = filter_by_constitution(revised, constitution)
    sl_dataset.add(batch, good_revisions)

train_supervised(policy, sl_dataset)

for batch in prompts:
    # RL-CAI
    y1 = policy.sample(batch)
    y2 = policy.sample(batch)

    pref = ai_preference_model.choose(batch, y1, y2, constitution)
    reward = reward_model.score(batch, pref.chosen)

    logp = policy.logprob(batch, pref.chosen)
    ref_logp = ref_policy.logprob(batch, pref.chosen)
    kl_term = logp - ref_logp

    objective = reward - beta * kl_term
    ppo_update(policy, objective)

    harmful_estimate = safety_model.score(batch, pref.chosen)
    beta = dual_update(beta, harmful_estimate, epsilon)
```

关键变量含义如下：

| 变量 | 含义 | 作用 |
|---|---|---|
| `policy` | 当前要训练的策略模型 | 被优化对象 |
| `ref_policy` | 参考策略 | 提供 KL 锚点 |
| `reward` | 偏好奖励 | 拉高帮助性与合规性 |
| `kl_term` | 当前策略相对参考策略的偏离 | 防止训练跑飞 |
| `beta` | KL 惩罚系数/对偶变量 | 控制安全与帮助性的平衡 |
| `epsilon` | 风险阈值 | 定义允许的平均有害性上限 |

$\beta$ 的调节可以做成简单反馈控制：

| 观测到的 $\mathbb{E}[r_{\text{harmful}}]$ | 对 $\beta$ 的操作 | 预期效果 |
|---|---|---|
| 大于 $\varepsilon$ | 增大 $\beta$ | 强化惩罚，收缩策略 |
| 接近 $\varepsilon$ | 维持 $\beta$ | 保持当前平衡 |
| 明显小于 $\varepsilon$ | 适度减小 $\beta$ | 释放帮助性空间 |

真实工程中，还会叠加 PPO 的 clip loss、value loss、advantage normalization、target KL early stopping 等稳定器。也就是说，CAI 不是替代 PPO，而是在 PPO 上增加“宪法驱动的奖励结构”和“安全导向的对偶控制”。

---

## 工程权衡与常见坑

最大的工程权衡，是 $\beta$ 调节带来的帮助性与拒绝率平衡。

| $\beta$ 状态 | 常见现象 | 风险 | 调参建议 |
|---|---|---|
| 太低 | 模型更敢答，敏感问题也可能给细节 | 有害性超标 | 提高 $\beta$，增强宪法规则覆盖 |
| 适中 | 回答有帮助，遇到高风险请求能解释性拒绝 | 训练较稳定 | 维持并做红队评估 |
| 太高 | 大量模板化拒绝，“无法协助此请求”泛滥 | 帮助性断崖下降 | 降低 $\beta$，混入更多正常任务样本 |

一个典型坑是**过度拒绝**。比如训练后，模型连“怎么合法申报公司税务”这种正常问题都频繁给出风险提示。这通常不是“模型更安全了”，而是惩罚项压过了主任务奖励，模型学会了最稳但最没用的策略。

另一个坑在 SL-CAI 阶段。很多人以为“只要生成批评和修订就够了”，其实不够。因为自我批评和修订本身也会产出噪声，如果把质量很差、只是表面更安全的文本直接拿去 SFT，后面 RL 会在错误分布上继续放大偏差。更稳的做法是：

- 只保留通过宪法检查的修订版本。
- 对明显退化的修订做拒绝采样过滤。
- 保证“安全提升”不能靠“信息量归零”实现。

一个真实工程场景是客服助手。团队一开始把 $\beta$ 调得很高，结果模型对“退款流程”“账号冻结原因”“设备保修范围”等都倾向于给官样文章。表面上投诉率下降，但问题解决率也下降。后来通常会这么修：

- 降低 $\beta$，让正常场景重新获得足够的帮助性。
- 扩充宪法规则，把“可以安全协助的合规解释”写清楚。
- 在训练集中加入更多“敏感但合法”的任务样本。
- 分离“危险执行指导”和“风险解释说明”这两类信号。

简化理解：不是所有敏感词都该触发拒绝，真正要拒绝的是危险行动支持。

---

## 替代方案与适用边界

CAI 不是唯一方案。它适合“风险较高、希望减少人工标注、又需要规模化迭代”的场景，但不是所有任务都必须上全套。

| 方案 | 核心做法 | 标注成本 | 安全性 | 适用场景 |
|---|---|---|---|---|
| 纯 RLHF | 人类直接标注偏好，做奖励模型和 RL | 高 | 取决于标注覆盖 | 低到中风险任务 |
| SL-CAI | 宪法驱动自我批评与修订，先做监督学习 | 中 | 比纯 SFT 更稳 | 预算有限但需提升安全 |
| RL-CAI | 在 SL-CAI 基础上继续 PPO + KL | 中到高 | 更强，且可持续优化 | 高风险通用助手 |
| 规则检测 | 训练外加规则库/分类器拦截 | 低 | 对已知风险有效 | 固定领域、低预算系统 |

对新手来说，可以这样理解三者区别：

- 只用人类偏好标签：质量可能高，但贵，而且覆盖面有限。
- 用宪法自动生成偏好：更便宜，更容易扩展，但规则本身必须写得好。
- CAI 全流程：把监督学习和强化学习都接入宪法，适合高风险任务，但系统复杂度更高。

什么时候需要引入宪法？

- 场景风险高，比如开放问答、医疗建议、法律辅助、安全攻防内容。
- 输出空间大，靠人工逐条写规则覆盖不过来。
- 团队需要持续扩展而不希望标注成本线性上涨。

什么时候可以省略？

- 任务风险低，比如内部文案润色、低风险摘要分类。
- 产品生命周期短，先用简单规则或纯 SFT 更划算。
- 输出格式固定，可由外部校验器严格约束。

还要看到 CAI 的边界：宪法本身不是客观真理，而是规则集合。规则写得偏、窄、互相冲突，模型就会学到偏的行为。因此“减少人工标注”不等于“不要人工审查”。人工从“逐条标回答”转移成了“设计、审查和更新宪法”。

---

## 参考资料

1. Anthropic, *Constitutional AI: Harmlessness from AI Feedback*  
   原始论文与官方介绍，定义了 SL-CAI 与 RL-CAI 的整体流程。  
   https://www.anthropic.com/news/constitutional-ai-harmlessness-from-ai-feedback

2. Bai et al., *Constitutional AI: Harmlessness from AI Feedback*, arXiv:2212.08073  
   适合查正式论文版本、实验设定和方法细节。  
   https://arxiv.org/abs/2212.08073

3. Anthropic, *Claude's Constitution*  
   适合理解“宪法”在真实产品中的规则来源与表达方式。  
   https://www.anthropic.com/constitution

4. John Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347  
   PPO 原始论文，理解 RL-CAI 里 PPO 部分的基础。  
   https://arxiv.org/abs/1707.06347

5. OpenAI Spinning Up, *Proximal Policy Optimization*  
   适合先看 PPO 的工程直觉，再回到论文公式。  
   https://spinningup.openai.com/en/latest/algorithms/ppo.html

6. Michael Brenndoerfer, *KL Divergence Penalty in RLHF: Theory & Implementation*  
   适合理解 KL 惩罚、对偶视角以及为什么 $\beta$ 会像“刹车强度”。  
   https://mbrenndoerfer.com/writing/kl-divergence-penalty-rlhf-training

入门阅读路径建议：

1. 先看 Anthropic 的 CAI 官方介绍，建立流程直觉。  
2. 再看 Claude's Constitution，理解“规则从哪里来”。  
3. 然后看 PPO 文档，补齐 RL 部分的训练框架。  
4. 最后读 Brenndoerfer 的 KL 解析和 arXiv 原文，把“约束优化 -> KL 正则”的数学链条补完整。
