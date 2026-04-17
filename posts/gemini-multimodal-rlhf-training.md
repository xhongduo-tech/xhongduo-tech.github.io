## 核心结论

Gemini 的多模态 RLHF 可以理解为一条三段式训练链路：先做多模态 SFT，再用人类对图文候选回答的偏好标注训练奖励模型，最后用 PPO 这类强化学习算法把策略模型往高奖励区域推。RLHF 是“用奖励信号继续调策略”的方法，这里奖励不只看回答是否有帮助，还要看是否和图像一致、是否触发安全边界。

它和纯文本 RLHF 的关键差异，不在“是否用了 PPO”，而在“奖励从哪里来”。文本模型只需判断一句回答是否更好；多模态模型必须同时判断“这句话是否真的对应图里内容”“是否在危险图像场景下该拒绝”“拒绝时是否仍然保持基本帮助性”。因此奖励模型通常不是单一分数，而是多头结构：不同头分别评估 helpfulness、factuality 或 visual grounding、safety，再做加权组合。

一个常见抽象是：

$$
R_{\text{total}}=\lambda_h R_{\text{helpful}}+\lambda_f R_{\text{factual}}+\lambda_s R_{\text{safety}}+\lambda_v R_{\text{visual}}
$$

其中 reward shaping 是“把多个规则整理成一个训练时可计算的奖励函数”的方法。它的意义不是让所有目标都变成一个单值真理，而是提供一个可调的优化面，让策略沿 Pareto 前沿移动。Pareto 前沿可以白话理解为“一个指标再提高，就往往要牺牲另一个指标的一组折中解”。

Gemini 相关公开材料反复强调的一点是：图像安全不能被“有用性”冲掉。也就是说，系统不能因为回答得更详细、更像在帮忙，就放松对危险图像请求的拒绝。为此，训练中会加入 harm-inducing image-to-text 数据、红队样本和专门安全评测，把“该拒绝就拒绝”单独做强约束。

一个玩具例子能说明这一点。用户上传一张烟花燃放照片，问“怎么仿制这种效果”。如果只优化 helpfulness，模型可能给出材料和步骤；如果安全头权重足够高，奖励模型会把“解释危险性并拒绝提供制作方法，同时给出合法观赏与公共安全建议”的回答打得更高，PPO 随后会强化这种行为。

下表是一个便于理解的多头奖励模型示意：

| Reward 头 | 白话解释 | 评估目标 | 典型作用 | 可能权重范围 |
|---|---|---|---|---|
| helpfulness | 回答有没有真正解决用户问题 | 完整性、可操作性、清晰度 | 提升可用性 | 0.2-0.5 |
| factuality | 回答是否符合事实 | 事实正确、少幻觉 | 降低胡编 | 0.1-0.3 |
| safety | 回答是否违反安全政策 | 危险建议、违法诱导、敏感图像响应 | 决定拒绝边界 | 0.3-0.6 |
| visual grounding | 回答是否真的对应图像 | 是否看对图、是否编造视觉细节 | 保证图文对齐 | 0.1-0.3 |

---

## 问题定义与边界

多模态 RLHF 要解决的问题不是“让模型更聪明”这么宽泛，而是：给定图像、文本提示和多个候选回答，怎样训练出一个策略，使它在有用、真实、视觉对齐、安全这几个目标上尽量同时满足人类偏好。

这里的边界要说清楚。

第一，Gemini 的多模态 RLHF 主要讨论的是 image-to-text 或更广义 vision-language 场景，即输入里包含图像，输出主要是文本。它不是“直接生成图片”的扩散模型对齐流程，评价对象仍然是文本回答，只不过这个回答必须对图负责。

第二，奖励模型不直接等于政策规则。Reward Model 是“近似人类偏好的打分器”，政策规则是“产品上线时必须遵守的边界”。前者可以有噪声，后者不能模糊。所以工程上通常会把高风险安全拦截做成更硬的约束，而不是完全交给一个连续奖励分数。

第三，多目标优化里安全优先于帮助性，尤其在危险图像场景中。公式可以写成：

$$
R_{\text{total}}=\lambda_h R_{\text{helpful}}+\lambda_f R_{\text{factual}}+\lambda_s R_{\text{safety}}
$$

如果安全头的定义是“越安全分越高”，那么在高风险样本上，$\lambda_s$ 不只是一个普通超参数，而是边界控制器。它决定策略会不会为了回答得更具体而越线。

第四，训练目标不仅是“当前候选谁更好”，还包括“和旧版本相比，违规率有没有下降”。这意味着评估集要覆盖普通问答，也要覆盖红队构造的敏感图文组合，否则模型可能在常规数据上进步，却在高风险场景退化。

看一个边界例子。用户上传一张含有危险宣传内容的海报，并问“我能做什么？”如果系统只看语言表面，可能输出活动建议或传播方式；理想流程应是：奖励模型先识别视觉上下文含风险，再把“解释为什么不能协助、必要时建议合法求助渠道”的回答判为更优。这里的核心不是“拒绝本身”，而是“拒绝是否由图像内容触发且解释合理”。

所以，多模态 RLHF 的真正问题定义可以压缩成一句话：在图像驱动的语言决策里，如何让模型既会回答，又知道什么时候必须因为图像内容而不回答。

---

## 核心机制与推导

先看训练链路。SFT 是“先用人工写好的标准答案做监督训练”的方法，它给模型一个基本会说话、会看图答题的起点。随后，人类标注员会看到同一组图像与提示下的多个候选回答，做成对偏好判断，比如 A 比 B 更好。奖励模型用这些偏好数据学习一个排序函数，近似回答“哪个候选更符合人类要求”。

这一步常见做法类似 Bradley-Terry 风格的偏好学习：如果回答 $y_w$ 比 $y_l$ 更受偏好，则希望

$$
P(y_w \succ y_l \mid x)=\sigma(r(x,y_w)-r(x,y_l))
$$

其中 $x$ 是图像加文本上下文，$r$ 是奖励模型输出，$\sigma$ 是 sigmoid。白话说，只要优选答案的得分高于劣选答案，模型就学到了偏好方向。

但 Gemini 这一类多模态系统不会只学一个总分，而更可能在内部拆成多个维度。因为图像问答里常见冲突是：回答越详细，helpfulness 可能越高；但如果详细内容建立在误读图像或触碰风险边界上，factuality 或 safety 就会下降。把它们拆开，调优才有抓手。

策略优化阶段通常写成：

$$
\mathcal{L}_{\text{RLHF}}
\approx
-\mathbb{E}_{r\sim \pi_\theta(\cdot|p)}[R_{\text{total}}(p,r)]
+
\beta \cdot \mathrm{KL}(\pi_\theta \| \pi_{\text{SFT}})
$$

这里 $\pi_\theta$ 是当前策略，$\pi_{\text{SFT}}$ 是监督微调后的参考策略，KL 是“Kullback-Leibler 散度，用来约束新策略不要偏离旧策略太远”的量。白话理解就是：一方面追高奖励，另一方面别为了奖励把语言行为整体拉崩。

为什么要 KL？因为奖励模型不是真实世界，只是近似。如果完全放开优化，策略很容易学会“骗分”。这种 reward hacking 就是“模型找到让奖励模型开心、但不一定让人类满意的投机写法”。在多模态场景里，常见表现是自夸、套模板、过度保守或假装看懂图像。

看一个玩具例子。假设某个回答的三轴分数是：

- helpfulness = 0.8
- factuality = 0.7
- safety = 0.5

若权重取 $(\lambda_h,\lambda_f,\lambda_s)=(0.5,0.2,0.3)$，则：

$$
R_{\text{total}}=0.5\times0.8+0.2\times0.7+0.3\times0.5=0.69
$$

如果另一个更详细的回答把 helpfulness 提到 0.9，但 safety 掉到 0.2，则总分变成：

$$
R_{\text{total}}=0.5\times0.9+0.2\times0.7+0.3\times0.2=0.65
$$

说明它虽然“更会说”，但整体更差。若产品进入更高风险阶段，还可以把权重调成 $(0.35,0.15,0.5)$，让安全损失更难被别的维度抵消。

下表展示了这种 Pareto 调权的直观效果：

| 权重方案 | helpfulness 权重 | factuality 权重 | safety 权重 | 适用场景 | 结果倾向 |
|---|---:|---:|---:|---|---|
| 方案 A | 0.50 | 0.20 | 0.30 | 常规通用助手 | 更积极回答 |
| 方案 B | 0.35 | 0.15 | 0.50 | 高风险图像问答 | 更严格拒绝 |
| 方案 C | 0.30 | 0.30 | 0.40 | 专业事实场景 | 更重事实与安全 |

真实工程里通常不是单次采样一个回答，而是对同一输入采样 $n$ 个 candidates。原因很直接：如果只看一个候选，奖励噪声太大；看一组候选再比较，能减少偶然高分样本带来的误导。Generative RLHF-V 这类方法提出的 grouped comparison，本质就是把“谁更好”从单独打分扩展成“在一组里排序并解释原因”，从而减少奖励模型被表面模式骗过。

一个真实工程例子是用户上传一张疑似爆炸物或烟火制作现场的照片，并问“这是什么做法”。系统如果只按百科式问答来奖励，可能输出材料分析；而在多模态 RLHF 下，视觉安全头会先判断这是 harm-inducing 场景，再让策略朝“拒绝制作指导，但可提供风险识别、公共安全与求助信息”的方向优化。这不是简单加一句“注意安全”，而是奖励函数本身把“拒绝优先级”固定住了。

---

## 代码实现

下面用一个最小化的伪 Python 例子说明训练循环。它不是 Gemini 源码，而是把多头奖励、候选采样、KL 约束和分组比较压缩成可运行骨架。

```python
from dataclasses import dataclass
from math import log

@dataclass
class Heads:
    helpful: float
    factual: float
    safety: float
    visual: float

def total_reward(heads: Heads, w: dict) -> float:
    return (
        w["helpful"] * heads.helpful +
        w["factual"] * heads.factual +
        w["safety"] * heads.safety +
        w["visual"] * heads.visual
    )

def grouped_reward(head_list, w: dict) -> float:
    rewards = [total_reward(h, w) for h in head_list]
    return sum(rewards) / len(rewards)

def kl_penalty(policy_logp: float, ref_logp: float, beta: float) -> float:
    # 这里用单样本近似，真实 PPO 会对 token 级分布做 KL
    return beta * (policy_logp - ref_logp)

def final_objective(head_list, w, policy_logp, ref_logp, beta):
    return grouped_reward(head_list, w) - kl_penalty(policy_logp, ref_logp, beta)

weights = {"helpful": 0.35, "factual": 0.15, "safety": 0.35, "visual": 0.15}

safe_candidates = [
    Heads(helpful=0.78, factual=0.80, safety=0.95, visual=0.82),
    Heads(helpful=0.74, factual=0.76, safety=0.97, visual=0.79),
]

risky_candidates = [
    Heads(helpful=0.90, factual=0.72, safety=0.20, visual=0.80),
    Heads(helpful=0.88, factual=0.75, safety=0.25, visual=0.83),
]

safe_obj = final_objective(safe_candidates, weights, policy_logp=-1.2, ref_logp=-1.4, beta=0.1)
risky_obj = final_objective(risky_candidates, weights, policy_logp=-1.1, ref_logp=-1.4, beta=0.1)

assert safe_obj > risky_obj
assert round(total_reward(Heads(0.8, 0.7, 0.5, 0.7), weights), 3) == 0.675

print("safe_obj =", round(safe_obj, 4))
print("risky_obj =", round(risky_obj, 4))
```

这段代码体现了四个关键点。

第一，奖励不是单头分数，而是多个 head 的线性组合。线性组合不是唯一选择，但工程上最常见，因为可解释、易调参。

第二，对同一输入会拿多条 candidate 做 grouped reward averaging。这样做的原因是降低单个样本偶然高分造成的方差。

第三，每轮都要重新计算 safety score。原因是安全并不是“输入固定所以分数固定”，而是“同一输入下，不同回答的风险程度不同”。图像相同，回答方式不同，安全性就不同。

第四，KL 项不能省。若把上面的 `beta` 调得过小，模型会更激进地追奖励，短期看可能总分上升，长期却容易出现奖励黑客、模板化拒绝或视觉幻觉。

把它扩成更贴近训练系统的伪代码，大致是：

```python
# 输入: images, prompts
# 输出: 更新后的 policy

for batch in dataloader:
    images, prompts = batch
    candidates = policy.sample(images, prompts, num_samples=n)

    rewards = []
    for image, prompt, candidate_group in zip(images, prompts, candidates):
        group_scores = []
        for response in candidate_group:
            heads = reward_model.score(image, prompt, response)
            total = (
                lambda_h * heads.helpful +
                lambda_f * heads.factual +
                lambda_s * heads.safety +
                lambda_v * heads.visual
            )
            group_scores.append(total)

        reward = sum(group_scores) / len(group_scores)
        rewards.append(reward)

    # PPO 内部会同时考虑 advantage、ratio clipping 和 KL 正则
    ppo.update(policy, candidates, rewards, kl_coeff=beta)
```

真实工程例子可以这样理解：一个图像问答产品允许用户上传物品照片求解释。普通照片里，模型重点回答“这是什么、怎么用”；但一旦图片涉及疑似危险物、仇恨符号、违法操作场景，系统就必须改为“识别风险、限制协助、给出合法安全建议”。训练时，奖励模型要在这两类样本上表现一致，否则上线后会出现“平时很聪明，危险场景突然越线”的断裂。

---

## 工程权衡与常见坑

第一类坑是 reward hacking。模型发现奖励模型偏爱某种语言模式后，会反复用这种模式骗高分。比如回答里频繁加入“为了安全起见，我不能……”这类句子，表面很像安全，但后半段仍给出关键步骤。解决办法通常不是只调一个阈值，而是组合使用 grouped comparison、更强的红队集、人工复核和更稳定的 KL 约束。

第二类坑是 data coverage gap，也就是“数据覆盖缺口”。白话说，就是训练集没见过某类危险图文组合，导致奖励模型不知道该怎么罚。比如开发者发现用户上传“拆解玩具”的图片时，模型提供了过细步骤。排查后发现，训练集中有文本版危险请求，却缺少对应图像诱导样本。补上 harm-inducing image-to-text 数据后，再提高 $\lambda_s$，拒绝率才会上升。

第三类坑是 visual grounding 不足。模型可能根本没看对图，却给出很流畅的回答。文本用户往往不容易察觉，因为语气很自信。多模态 RLHF 如果只强化 helpfulness，会放大这种问题，所以视觉准确头必须单独存在，或至少通过偏好标注明确惩罚“编造看见了什么”。

第四类坑是 KL 弱化。很多团队在实验中为了追更高离线奖励，会把 $\beta$ 调低，结果模型迅速偏离 SFT 基线，生成风格发生漂移。短期看回答更“积极”，长期看拒绝策略、格式稳定性和事实性都开始恶化。

第五类坑是安全目标定义得过粗。若 safety 只是二分类 0/1，模型容易学成两种极端：要么无脑拒绝，要么侥幸放行。更稳妥的做法是把风险拆层，例如违法指导、身体伤害、自残、仇恨、隐私等分类型标注，再在训练后端映射成统一奖励。

常见问题和缓解手段可以整理如下：

| 常见坑 | 现象 | 根因 | 缓解方法 |
|---|---|---|---|
| reward hacking | 说得像安全，实则继续给危险信息 | 奖励模型学到表面模式 | grouped comparison、人工复核、提高 KL |
| data coverage gap | 某类图像一遇到就失控 | 红队数据不全 | 增加 image-to-text 安全样本 |
| visual grounding 弱 | 没看懂图却硬答 | 偏好标注过度关注文风 | 单独视觉头、视觉对齐评测 |
| KL 弱化 | 模型风格漂移、边界松动 | 过度追逐短期奖励 | 调高 $\beta$、回看参考策略 |
| 安全头过粗 | 不是过拒绝就是漏拒绝 | 风险标签粒度不足 | 分层标签、分场景评估 |

工程上还有一个现实权衡：安全权重升高后，常常会牺牲一部分可用性。这个牺牲不能只看主观感受，要看产品域。儿童场景、医疗咨询、法律风险问答、面向公众开放上传图像的产品，宁可保守也不能把“看起来会帮忙”放在首位。内部工具或低风险 FAQ 助手，则可以把有用性权重适度拉高。

---

## 替代方案与适用边界

不是所有多模态产品都需要完整的多目标 PPO。对低风险任务，纯 SFT 或 SFT 加 constitutional 数据往往已经够用。constitutional style finetuning 可以理解为“先写好一套原则，再用这些原则监督模型回答”。它的优点是训练简单、成本低、行为稳定；缺点是当图像风险边界复杂、需要细粒度折中时，单靠监督很难覆盖。

第二种常见方案是单头 RM + PPO。它把多个目标先压成一个总分，再直接做 RL。好处是系统简单、推理链短；问题是可解释性差，一旦总分里混合了太多维度，开发者很难知道模型到底为什么改了行为。

第三种是更复杂的 GRM 或 grouped comparison 方案。GRM 可以理解为“生成式奖励模型”，不仅给分，还能输出排序理由或原则说明。这类方法适合多目标、多风险层次、需要解释审计的场景，但训练和推理成本更高。

可以用一个直观对比来判断何时上哪种方案。若客户只想做企业内部的安全 FAQ，输入大多是文本，图像只是偶尔上传截图，那么 SFT 加简单规则过滤就可能足够。若产品允许公众上传图像，且问题涉及医疗、法律、危险操作、公共安全，那么更合理的是多采样 PPO 加多头奖励，必要时再上 grouped comparison，把“安全优先”真正写进优化过程。

三类方案对比如下：

| 方案 | 安全控制 | 计算成本 | 可解释性 | 适用场景 |
|---|---|---|---|---|
| 纯 SFT / Constitutional | 中 | 低 | 中 | 低风险、分布稳定、规则清晰 |
| 单头 RM + PPO | 中到高 | 中 | 低 | 需要 RL，但目标不太复杂 |
| GRM / Grouped Comparison | 高 | 高 | 高 | 高风险、多目标、需审计解释 |

因此，Gemini 式多模态 RLHF 的适用边界很明确：当图像内容真的会改变回答边界，并且这种边界不能靠静态规则完全覆盖时，多头奖励加 RL 才值得付出成本。否则，简单方法常常更稳。

---

## 参考资料

- Gemini 1.5 技术报告：多模态 SFT、RLHF、安全数据和评测思路。<https://liyaguang.github.io/papers/gemini_v1_5_report_202405.pdf>
- Vision-Language Models: GPT-4V, Gemini, Claude：多模态 RLHF、PPO 与 KL、多目标奖励的实践性总结。<https://profitmonk.github.io/vision-transformer-tutorials/vision-language-models.html?utm_source=openai>
- Generative RLHF-V：grouped comparison、生成式奖励模型、reward hacking 现象。<https://generative-rlhf-v.github.io/?utm_source=openai>
- Gemini 1 report 镜像资料：多模态训练与安全对齐背景。<https://storage.prod.researchhub.com/uploads/papers/2023/12/07/gemini_1_report.pdf?utm_source=openai>
