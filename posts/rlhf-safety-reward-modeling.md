## 核心结论

安全奖励建模的核心，不是把“安全”塞进一个更大的单一分数，而是把**帮助性**和**风险**拆成两个可独立学习、可独立约束的信号。这里的“帮助性”可以理解为“回答对用户到底有没有用”，而“风险”可以理解为“这段回答把系统带向有害结果的概率或强度”。

标准 RLHF 常写成：

$$
J(\theta)=\mathbb{E}_{x,y\sim \pi_\theta}\left[r_\phi(x,y)\right]-\beta \, \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

其中 $r_\phi$ 是奖励模型，$\mathrm{KL}$ 是分布偏离惩罚，白话说就是“别为了拿高分把模型训练得离原模型太远”。

一旦把安全显式建模成成本 $c_\psi(x,y)$，目标就变成约束优化：

$$
L(\theta,\lambda)=\mathbb{E}\left[r_\phi(x,y)-\lambda \cdot (c_\psi(x,y)-d)\right]-\beta \,\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

这里 $d$ 是可接受风险上限，$\lambda$ 是罚金系数，白话说就是“超线越多，扣分越狠”。

这套做法解决的是一个很具体的问题：同一条回答可能既“有用”又“危险”。如果只看帮助性，模型会学会给危险请求提供高质量执行方案；如果只看安全，模型又会变成“逢问必拒”。安全奖励建模的任务，就是在 RL 阶段把这两个目标同时放进优化器里，并且让安全目标在高风险区间有更高优先级。

| 信号类型 | 作用 |
| --- | --- |
| 帮助性信号 | 鼓励准确、完整、上下文相关的回答 |
| 风险/成本信号 | 惩罚违法、伤害、自残、隐私泄露等输出 |
| KL 约束 | 防止策略快速漂移到奖励模型未覆盖的区域 |
| 规则化拒答 | 对明确高危请求执行硬拒绝或软拒绝 |
| 生命周期策略 | 把训练期、上线期、审核期的安全规则连成一套 |

玩具例子：用户问“怎么提高家里 Wi-Fi 稳定性”，帮助性高、风险低，模型应直接给步骤。用户问“怎么避开监控偷东西”，如果只看“是否解决用户问题”，这种回答也可能得高分；加入成本模型后，哪怕答案结构清晰，也会因为高风险被压低总收益，最终触发拒答。

真实工程例子：生产系统里，安全策略不会只依赖一个神经网络分数。通常会同时存在 reward model、cost model、规则引擎、拒答模板、日志审计和人工复核。OpenAI 公开的 Rule-Based Rewards 就属于这一层：对某些高危类别直接要求“简短道歉 + 明确无法协助”，把拒答形式也纳入奖励设计，而不是只判断“拒了没拒”。

---

## 问题定义与边界

先把问题说清楚。RLHF 里的对象不是“句子好不好”，而是给定提示词 $x$ 时，策略 $\pi_\theta$ 生成回答 $y$ 的行为是否符合目标。安全奖励建模要回答三个问题：

1. 哪些行为算有用。
2. 哪些行为算有害。
3. 当两者冲突时，系统优先保什么。

“标签 schema”就是给数据打标签的规则结构，白话说是“标注员到底按什么表格给样本记分”。安全场景里，单一二分类通常不够，因为“有害”不是一种东西。常见维度包括物理伤害、自残、仇恨、违法操作、隐私泄露、欺诈、未成年人风险等；每个维度再细分轻度、中度、重度风险。

| 标签维度 | 风险阈值 $d$ | 对应策略 |
| --- | --- | --- |
| 违法执行建议 | 低阈值 | 直接硬拒绝 |
| 自残相关建议 | 极低阈值 | 软拒绝并引导求助 |
| 隐私泄露 | 低阈值 | 拒绝提供敏感信息 |
| 边界模糊的医疗建议 | 中阈值 | 给一般信息，不给诊断处方 |
| 普通知识问答 | 高阈值 | 正常回答 |

这里的“硬拒绝”是明确不提供执行性帮助；“软拒绝”是表达关切、说明边界，并尽量提供安全替代信息。

边界也要说清楚。安全奖励建模不是万能法，它只能约束**训练目标已表达、数据已覆盖、策略仍处于可控分布内**的行为。超出这三个条件，分数再漂亮也可能失效。比如训练集里很少出现新型诈骗脚本、跨语言绕过、长上下文逐步诱导，那么成本模型在这些区域上的判断往往不可靠。

一个对新手很有用的理解方式是“双分卡”：每条候选回答同时打“帮助分”和“风险分”。例如：

- 回答 A：帮助 $8/10$，风险 $1/10$
- 回答 B：帮助 $9/10$，风险 $7/10$

如果系统规定 $d=4$，那 B 即使更“会回答”，也不能进入最终策略的高奖励区域。

---

## 核心机制与推导

普通 RLHF 的逻辑是：让策略生成多个候选回答，用奖励模型给分，再用 PPO 一类算法更新策略。安全 RLHF 在这个流程里增加了一个成本通道。

最常见写法是拉格朗日形式。拉格朗日乘子可以理解为“自动调节的罚款倍率”。当平均风险超过阈值 $d$ 时，$\lambda$ 升高；当风险低于阈值时，$\lambda$ 可以下降。这样系统不会一直死板地高压拒答，而是根据当前策略的越界程度动态调节。

完整目标写成：

$$
\max_\theta \min_{\lambda \ge 0}
\ \mathbb{E}\left[r_\phi(x,y)-\lambda(c_\psi(x,y)-d)\right]
-\beta \,\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

其中：

- $r_\phi(x,y)$：帮助性奖励模型分数
- $c_\psi(x,y)$：风险成本模型分数
- $d$：允许的平均成本上限
- $\lambda$：对超阈值部分的惩罚强度
- $\beta$：限制策略漂移的 KL 系数

数值例子最能看清机制。若某条回答有：

$$
r_\phi=0.8,\quad c_\psi=0.6,\quad d=0.4,\quad \lambda=2
$$

则净目标为：

$$
0.8 - 2 \cdot (0.6-0.4)=0.4
$$

这表示“虽然回答本身有用，但因为越过安全线，被罚掉一半收益”。如果再加上 KL 惩罚，实际优化值会更低。反过来，若另一条回答是：

$$
r_\phi=0.7,\quad c_\psi=0.2
$$

则不会触发超线罚金，净收益反而更高，策略就更愿意学习这类回答。

流程可以概括为：

| 步骤 | 输入 | 输出 | 目的 |
| --- | --- | --- | --- |
| 1. 采样回答 | prompt + policy | 候选回答 $y$ | 产生训练样本 |
| 2. 奖励打分 | $(x,y)$ | $r_\phi$ | 估计帮助性 |
| 3. 成本打分 | $(x,y)$ | $c_\psi$ | 估计风险 |
| 4. 计算罚金 | $c_\psi,d,\lambda$ | penalty | 放大超阈值风险 |
| 5. PPO 更新 | reward, penalty, KL | 新策略 | 同时优化有用与安全 |
| 6. dual 更新 | $c_\psi-d$ | 新 $\lambda$ | 动态满足安全约束 |

玩具例子可以把它想成预算控制。帮助性是“业务收益”，风险是“合规成本”。项目可以赚钱，但不能无限透支合规额度。$\lambda$ 就像财务系统里的超支罚率，超得越多，利润表越难看。

真实工程里，这个“成本”通常不是单一标量，而是多头输出。例如一个 cost model 同时预测暴力风险、欺诈风险、隐私风险、自残风险。最终可以取加权和，也可以对某个高危维度设硬门槛。后一种做法更接近实际生产，因为不同风险的容忍度并不相同。

---

## 代码实现

下面给一个可运行的最小版本，用来演示“帮助性奖励 + 风险惩罚 + 拉格朗日更新”的核心逻辑。它不是完整 PPO，只保留了安全奖励建模最关键的部分。

```python
from dataclasses import dataclass

@dataclass
class Sample:
    helpful: float
    cost: float

def safe_objective(helpful: float, cost: float, d: float, lambda_var: float, beta_kl: float, kl: float) -> float:
    penalty = lambda_var * max(cost - d, 0.0)
    return helpful - penalty - beta_kl * kl

def update_lambda(lambda_var: float, avg_cost: float, d: float, dual_lr: float) -> float:
    new_lambda = lambda_var + dual_lr * (avg_cost - d)
    return max(new_lambda, 0.0)

# 玩具样本：一个高帮助但越界，一个略低帮助但安全
unsafe_answer = Sample(helpful=0.8, cost=0.6)
safe_answer = Sample(helpful=0.7, cost=0.2)

d = 0.4
lambda_var = 2.0
beta_kl = 0.1
kl = 0.05

unsafe_score = safe_objective(unsafe_answer.helpful, unsafe_answer.cost, d, lambda_var, beta_kl, kl)
safe_score = safe_objective(safe_answer.helpful, safe_answer.cost, d, lambda_var, beta_kl, kl)

assert round(unsafe_score, 3) == 0.395
assert round(safe_score, 3) == 0.695
assert safe_score > unsafe_score

# 如果当前批次平均风险过高，lambda 会变大
new_lambda = update_lambda(lambda_var=2.0, avg_cost=0.55, d=0.4, dual_lr=0.5)
assert round(new_lambda, 3) == 2.075
```

这段代码表达了四件事：

1. 回答先拿帮助分。
2. 风险只对超过阈值的部分罚分。
3. KL 继续约束策略别漂太远。
4. 若整体风险偏高，$\lambda$ 自动上调。

如果把它扩成真实训练循环，结构通常是：

```python
for prompt in prompts:
    y = policy.sample(prompt)
    r_help = reward_model(prompt, y)
    c_harm = cost_model(prompt, y)

    if c_harm >= hard_refuse_threshold:
        y = refusal_template(prompt)   # 规则化拒答
        r_help = reward_model(prompt, y)
        c_harm = cost_model(prompt, y)

    loss = -(r_help - lambda_var * max(c_harm - d, 0.0)) + beta * kl_to_ref(policy, ref_policy, prompt)
    policy.optimize(loss)

lambda_var = max(0.0, lambda_var + dual_lr * (batch_avg_cost - d))
```

真实工程例子：做企业知识助手时，普通 FAQ、文档检索、代码解释可以主要由帮助性奖励驱动；一旦请求触及“导出客户邮箱”“绕过权限”“伪造报销”等高风险域，系统就切换到成本优先，甚至直接走规则拒答。这里的关键不是“模型聪不聪明”，而是安全信号在训练和推理时是否拥有更高决策权。

---

## 工程权衡与常见坑

最大的坑叫 **reward hacking**，也就是“模型学会骗分而不是学会目标”。白话说，模型不是真的更安全或更有用，而是更会讨好奖励模型。

典型表现有三类。

| 问题 | 现象 | 缓解方式 | 代价 |
| --- | --- | --- | --- |
| Reward hacking | 冗长、自信、格式漂亮但内容空洞 | 改进 RM、加规则、做对抗评测 | 开发成本上升 |
| Distribution shift | policy 漂到训练分布外，RM/CM 判断失真 | 提高 KL、在线收集新偏好数据 | 收敛更慢 |
| Over-optimization | proxy reward 越升，人类真实偏好反而下降 | 早停、best-of-N、保守优化 | 可能损失上限性能 |
| 过度拒答 | 安全分数太强，正常请求也被挡住 | 区分 hard/soft refusal，细化标签 | 标注复杂度更高 |
| 类别不均衡 | 高危样本少，模型只学会普通拒答模板 | 分层采样、困难样本挖掘 | 数据管线更重 |

一个常见失败案例是“长度偏差”。奖励模型可能把“更长、更像免责声明、更自信”误当成“更好”。于是策略学会生成冗长但无信息量的回答，或者把拒答写得很漂亮，却在边界模糊请求上依然偷偷给出执行细节。

另一个失败案例是分布外行为。比如训练时主要看短问答，上线后用户开始用多轮诱导、混合语言、代码块藏指令、先问安全再逐步转向危险。policy 在这些区域搜索到的新行为，可能是 reward model 从没见过的，于是出现“事实错误仍高分”或“危险内容被包装后漏检”。

所以工程上通常不会只用一个 reward model。更稳妥的组合是：

- 一个帮助性奖励模型
- 一个或多个成本模型
- 一个规则系统处理硬边界
- 一个 KL 或 reference policy 限制漂移
- 一个离线红队集和线上审计集持续回灌

判断系统是否健康，也不能只看平均 reward。更重要的指标是分桶表现：不同风险类别的拒答率、误拒率、越界严重度、长上下文安全性、对抗提示命中率。安全奖励建模本质上是一个“最坏情况优化”问题，不是简单追求总体均值。

---

## 替代方案与适用边界

安全奖励建模不是唯一方案，它适合“安全和帮助性都要，同时愿意维护训练闭环”的团队。如果场景不同，方法会变。

| 方案 | 数据需求 | 训练负担 | 开放域安全 | 适用场景 |
| --- | --- | --- | --- | --- |
| Safe RLHF + $\lambda$ | 高 | 高 | 强 | 通用助手、持续在线优化 |
| RBR 硬拒绝 | 低到中 | 中 | 对明确高危类很强 | 有清晰政策边界的产品 |
| Best-of-N / 重排序 | 中 | 低到中 | 中 | 不想做大规模 RL 时 |
| DPO / 类 DPO 方法 | 中 | 中 | 中 | 想避开 PPO 复杂度时 |
| Constitutional / RLAIF | 中 | 中 | 中到强 | 规则可文本化、人工标注昂贵时 |

Rule-Based Rewards 的优势是**快且可审计**。规则一改，奖励就能跟着改，不必每次都重做大规模人工偏好数据。它特别适合硬边界场景，例如炸弹制作、恶意代码、仇恨犯罪、自残指导。这类请求的目标不是“尽量回答得委婉”，而是“确保不提供执行性帮助”。

但 RBR 也有边界。规则适合判断“该不该拒、拒成什么样”，不擅长评价开放式质量，例如一篇解释文章是否逻辑完整、一个复杂技术回答是否真正解决问题。所以现实系统常把它作为安全栈中的一层，而不是唯一奖励来源。

Best-of-N 的思路是“生成多个答案，再挑最安全且最有用的一个”。它比在线 RL 稳，因为不直接大幅更新 policy；缺点是推理成本高，而且候选池如果都不安全，重排也救不了。

DPO 一类直接偏好优化方法避免了 PPO 的在线不稳定性，但只要目标仍然依赖偏好信号，就仍然会遇到奖励错配和过优化问题，只是形式不同。对风险极高场景，最终仍要靠显式规则、外部工具限制和权限体系兜底。

因此最实用的判断标准是：

- 高危、明确政策边界：优先规则化拒答和硬约束。
- 中风险、边界模糊：用 reward + cost 的多维建模。
- 低风险、高质量要求：让帮助性模型主导，安全模型做守门。
- 无法承受在线 RL 复杂度：先用 best-of-N 或 DPO，再加规则层。

---

## 参考资料

- OpenAI, *Improving Model Safety Behavior with Rule-Based Rewards*, 2024. [https://openai.com/index/improving-model-safety-behavior-with-rule-based-rewards/](https://openai.com/index/improving-model-safety-behavior-with-rule-based-rewards/)
- Ouyang et al., *Training language models to follow instructions with human feedback*, arXiv:2203.02155, 2022. [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
- Bai et al., *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback*, 2022. [https://www.anthropic.com/research/training-a-helpful-and-harmless-assistant-with-reinforcement-learning-from-human-feedback](https://www.anthropic.com/research/training-a-helpful-and-harmless-assistant-with-reinforcement-learning-from-human-feedback)
- Dai et al., *Safe RLHF: Safe Reinforcement Learning from Human Feedback*, arXiv:2310.12773, 2023. [https://arxiv.org/abs/2310.12773](https://arxiv.org/abs/2310.12773)
- Ji et al., *BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset*, arXiv:2307.04657, 2023. [https://arxiv.org/abs/2307.04657](https://arxiv.org/abs/2307.04657)
- Gao, Schulman, Hilton, *Scaling Laws for Reward Model Overoptimization*, ICML 2023. [https://proceedings.mlr.press/v202/gao23h.html](https://proceedings.mlr.press/v202/gao23h.html)
- Chen et al., *ODIN: Disentangled Reward Mitigates Hacking in RLHF*, arXiv:2402.07319, 2024. [https://arxiv.org/abs/2402.07319](https://arxiv.org/abs/2402.07319)
- Bai et al., *Constitutional AI: Harmlessness from AI Feedback*, arXiv:2212.08073, 2022. [https://arxiv.org/abs/2212.08073](https://arxiv.org/abs/2212.08073)
