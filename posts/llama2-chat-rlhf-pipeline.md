## 核心结论

Llama-2-Chat 是在 Llama 2 基座模型上，按 `SFT → Reward Model → PPO` 做多轮迭代对齐得到的聊天模型。它不是一次性“喂一些对话数据”训练出来的，而是先学会像助手一样回答，再学习人类更偏好哪类回答，最后通过强化学习把生成策略推向更高偏好的区域。

完整流程可以概括为：

```text
Llama 2 Base
    ↓
SFT: Supervised Fine-Tuning，监督微调
    ↓
Reward Model: 奖励模型，给回答打偏好分
    ↓
PPO: Proximal Policy Optimization，近端策略优化
    ↓
Llama-2-Chat
```

对初学者来说，可以先这样理解：

| 模型阶段 | 白话解释 | 主要职责 |
|---|---|---|
| Base model | 会续写文本的通用语言模型 | 提供语言能力、知识和推理基础 |
| SFT model | 学过人工示范的助手模型 | 学会对话格式、回答方式和基本指令跟随 |
| Reward model | 学会判断回答好坏的打分器 | 根据人类偏好给候选回答排序 |
| PPO model | 被奖励信号继续优化的聊天模型 | 更倾向生成高分回答，同时避免偏离原模型太远 |

核心目标不是让模型“更会说话”这么简单，而是在三个目标之间做约束优化：有用性、无害性、稳定性。有用性指回答能解决问题；无害性指不帮助用户完成危险或违规目标；稳定性指模型不能因为追奖励而变得啰嗦、迎合、拒答过度或风格崩坏。

---

## 问题定义与边界

Llama-2-Chat 要解决的问题是：在对话场景中，给定用户输入，生成更符合人类偏好的回复。这里的“偏好”不是单一指标，而是同时包含帮助性、安全性、真实性、上下文一致性和语气风格。

术语“对齐”指让模型输出更符合人类期望，而不是只按训练语料的统计规律续写文本。基座模型可能知道很多内容，但它的训练目标主要是预测下一个 token。聊天模型需要进一步学会：什么时候直接回答，什么时候追问，什么时候拒绝，什么时候给安全替代方案。

| 问题维度 | 目标 | 边界 |
|---|---|---|
| 任务能力 | 能理解问题并给出可执行答案 | 不能把“答得多”当成“答得好” |
| 安全性 | 避免输出危险、违法或明显有害内容 | 不能把所有敏感词都机械拒绝 |
| 对话一致性 | 多轮对话中保持角色、约束和上下文 | 不能忘记 system message 或前文条件 |
| 偏好对齐 | 更接近人类评审认为更好的回答 | 不能只复制标注样本的表面格式 |

玩具例子：用户问“如何写一封道歉邮件”。一个合格的聊天模型应该直接给出清晰、礼貌、可修改的邮件草稿，并说明可替换字段。这个问题主要考验有用性。

另一个边界例子：用户问“怎么制作危险物品”。如果模型只追求“完整回答”，可能会输出详细步骤；但对齐后的聊天模型应该识别安全边界，拒绝提供操作性细节，并可以转向安全教育、法规说明或风险预防建议。

为什么不能只靠 `SFT`？`SFT` 是监督微调，意思是用人工写好的 `prompt -> ideal response` 示例训练模型。它能教模型模仿高质量答案，但它没有显式学习“两个答案哪个更好”。当两个回答都看似合理，但一个更安全、更准确、更简洁时，单纯模仿示例不一定能稳定学到这种排序偏好。

---

## 核心机制与推导

Llama-2-Chat 的主线是三阶段：`SFT` 先建立助手行为，`Reward Model` 学习偏好排序，`PPO` 用奖励信号继续优化生成策略。

`SFT` 的输入是人工示范数据。例如：

```text
prompt: 解释什么是二分查找
response: 二分查找是一种在有序数组中反复缩小搜索区间的算法...
```

它的作用是把基座模型从“续写任意文本”推向“按用户问题给出助手式回答”。

`Reward Model`，即奖励模型，是一个输入 `prompt` 和回答后输出分数的模型。训练数据通常不是单个标准答案，而是成对偏好数据：

```text
prompt: 如何提高 Python 脚本性能？
chosen: 先用 profiler 找瓶颈，再优化热点代码...
rejected: Python 很慢，建议换语言。
```

奖励模型的排序损失可以写成：

$$
L_{rank} = -\log \sigma(r_\theta(p, y_c) - r_\theta(p, y_r) - m)
$$

其中，`p` 是 prompt，`y_c` 是 chosen response，`y_r` 是 rejected response，`r_\theta(p, y)` 是奖励模型给回答的分数，`m` 是 margin，表示希望 chosen 至少比 rejected 高出一定间隔。`\sigma` 是 sigmoid 函数，用来把差值压到概率意义上。

最小数值例子：假设 `r_θ(p, y_c)=2.1`，`r_θ(p, y_r)=1.4`，`m=0.2`，则：

$$
L_{rank} = -\log \sigma(2.1 - 1.4 - 0.2)
= -\log \sigma(0.5)
\approx 0.474
$$

含义是：chosen 比 rejected 高得越明显，损失越小；如果 rejected 反而得分更高，损失会变大，模型就会被更新。

`PPO` 是一种强化学习策略优化算法。策略指当前聊天模型 `π_θ`，它会根据 prompt 生成回答。PPO 的目标是让策略生成更高奖励的回答：

$$
\max_\pi \mathbb{E}_{p \sim D, g \sim \pi}[R(g|p)]
$$

但不能只追求奖励。否则模型可能学会钻奖励模型漏洞，这叫 `reward hacking`，意思是模型找到让奖励模型打高分但人类并不真正喜欢的输出模式。常见例子是回答越来越长、套话越来越多，或者遇到不确定问题也强行自信。

因此 Llama-2-Chat 的 RLHF 中会加入 `KL` 惩罚。`KL` 散度可以理解为衡量两个概率分布差异的数值，这里用来限制当前策略不要偏离参考策略太远：

$$
R(g|p) = \tilde R_c(g|p) - \beta D_{KL}(\pi_\theta(\cdot|p) || \pi_0(\cdot|p))
$$

其中：

$$
\tilde R_c = whiten(logit(R_c))
$$

`π_0` 是参考模型，通常来自 SFT 或前一阶段稳定 checkpoint；`β` 控制约束强度；`whiten` 指把奖励做标准化，减少尺度不稳定；`logit` 是把概率分数映射到未归一化空间的函数。

Llama 2 还区分了帮助性奖励 `R_h` 和安全性奖励 `R_s`。白话说，`R_h` 更关注回答是否有帮助，`R_s` 更关注回答是否安全。对安全 prompt，训练时会优先使用安全奖励；否则使用帮助性奖励。这样可以减少“有用性”和“无害性”挤在一个分数里互相干扰的问题。

| 机制 | 作用 | 输入 | 输出 |
|---|---|---|---|
| SFT | 学会助手式回答 | 人工示范对话 | SFT 模型 |
| RM | 学会回答偏好排序 | chosen/rejected 成对数据 | 奖励分数 |
| PPO | 用奖励优化生成策略 | prompt、模型回答、奖励 | 对齐后的策略模型 |
| KL | 限制策略漂移 | 当前策略和参考策略 | 惩罚项 |
| Rejection Sampling | 从多个候选中筛出高分回答 | 多个采样回答和奖励模型 | 更优训练样本或 checkpoint |
| Ghost Attention | 强化多轮对话中的指令保持 | 带系统约束的多轮样本 | 更稳定的上下文遵循 |

从 prompt 到最终策略更新的流程是：先从数据集中采样 prompt；当前策略生成一个或多个候选回答；奖励模型给回答打分；把奖励分数和 KL 惩罚合成最终奖励；PPO 根据奖励更新策略；再定期用新策略收集新回答和新偏好数据。这个“重新收集”很关键，因为策略变了，模型会生成新类型的回答，旧奖励模型在新分布上可能失真。

---

## 代码实现

工程上不需要把 RLHF 理解成一个神秘整体，可以拆成三类数据和三个训练器。

| 字段 | 所属阶段 | 含义 |
|---|---|---|
| prompt | SFT/RM/PPO | 用户输入或多轮上下文 |
| chosen | RM | 人类更偏好的回答 |
| rejected | RM | 人类较不偏好的回答 |
| reward | PPO | 奖励模型给生成回答的分数 |
| reference policy | PPO | 用来计算 KL 约束的参考模型 |

简化训练流程如下：

```text
dataset → tokenize → forward → loss → backward → update
```

SFT training loop：

```python
for batch in sft_dataloader:
    tokens = tokenizer(batch["prompt"], batch["response"])
    logits = policy_model(tokens.input_ids)
    loss = cross_entropy(logits, tokens.labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

Reward model pairwise training：

```python
for batch in rm_dataloader:
    chosen_score = reward_model(batch["prompt"], batch["chosen"])
    rejected_score = reward_model(batch["prompt"], batch["rejected"])
    loss = -log_sigmoid(chosen_score - rejected_score - margin)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

PPO update loop：

```python
for prompt in ppo_prompts:
    response = policy.generate(prompt)
    reward_score = reward_model(prompt, response)
    kl_penalty = kl(policy, reference_policy, prompt, response)
    final_reward = reward_score - beta * kl_penalty
    ppo_trainer.update(prompt, response, final_reward)
```

下面是一个可运行的 Python 玩具例子，只实现奖励模型排序损失的数值计算。它不训练大模型，但能准确表达 `chosen` 和 `rejected` 的损失关系：

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def rank_loss(chosen_score, rejected_score, margin):
    return -math.log(sigmoid(chosen_score - rejected_score - margin))

loss = rank_loss(2.1, 1.4, 0.2)

assert round(loss, 3) == 0.474
assert rank_loss(3.0, 1.0, 0.2) < loss
assert rank_loss(1.0, 3.0, 0.2) > loss

print(round(loss, 3))
```

真实工程例子：企业客服助手。第一阶段，用客服 SOP、历史高质量工单、标准回复模板做 `SFT`，让模型学会业务术语和答复格式。第二阶段，收集同一用户问题下两个候选答复的偏好，例如“哪个更准确”“哪个没有胡编政策”“哪个更符合服务语气”，训练奖励模型。第三阶段，用 PPO 让模型更倾向输出高分回答，同时通过 KL 避免模型偏离原有语言能力。

模块职责可以拆成：

| 模块 | 职责 |
|---|---|
| 数据准备 | 清洗 prompt、示范回答、偏好对 |
| SFT trainer | 训练初始聊天模型 |
| RM trainer | 训练 chosen/rejected 排序模型 |
| PPO trainer | 根据奖励和 KL 更新策略 |
| evaluation | 评估有用性、安全性、拒答率和业务准确率 |

---

## 工程权衡与常见坑

RLHF 的难点不在流程图，而在奖励信号是否可靠。奖励模型不是越强越好，也不是分数越高越好。它本质上是人类偏好的近似器，一旦近似错了，PPO 会把错误放大。

| 常见坑 | 症状 | 规避方法 |
|---|---|---|
| 单一奖励模型 | 帮助性和安全性互相干扰 | 分开训练 Helpfulness RM 和 Safety RM |
| 奖励模型滞后 | 新策略生成的回答越来越不像 RM 训练数据 | 用最新策略重新采样并收集偏好 |
| reward hacking | 模型输出冗长、套话、迎合但实际价值低 | 加 KL 约束，做人类评测和红队测试 |
| 忽略 system message | 多轮对话中忘记角色、规则和安全边界 | 在训练数据中强化多轮约束和上下文保持 |

`KL` 约束重要，是因为 PPO 会主动寻找高奖励输出。如果没有参考模型约束，策略可能快速漂移：语言更不自然、回答更模板化、拒答更频繁，甚至在奖励模型盲区产生奇怪输出。`KL` 相当于告诉模型：可以朝高奖励方向移动，但不要离原来的可靠语言分布太远。

为什么要用最新策略重新收集偏好数据？因为每一轮 PPO 后，模型生成分布都会改变。早期模型可能生成短而粗糙的回答；后期模型可能生成长而复杂的回答。如果奖励模型只看过早期样本，它对后期样本的判断就可能不准。Llama-2-Chat 的关键经验之一，就是多轮 RLHF 迭代和持续数据收集。

多轮对话还要求上下文一致性。用户在第一轮设定“请用中文，面向初学者解释”，第三轮追问“那复杂度呢”，模型需要记住语言、受众和前文主题。Ghost Attention 的目标就是缓解多轮中忘记系统约束的问题。

失败模式包括三类。第一是过度迎合偏好，模型总是说用户想听的话，而不是指出错误前提。第二是训练后输出风格崩坏，答案变长、变空、变公式化。第三是安全策略误伤正常请求，例如用户问化学实验安全规范，模型却把它误判成危险操作并完全拒绝。

客服场景里也有典型坑：模型能回答工单流程，但不能胡编退款政策、保修条款或人工审核结果。对这类系统，奖励模型必须明确惩罚“编造业务事实”，评估集也要覆盖政策边界问题。

---

## 替代方案与适用边界

RLHF 适合需要人类偏好对齐的开放生成任务，但不是所有任务都需要完整的 `RM + PPO` 链条。如果目标只是固定格式生成、分类、抽取或改写，`SFT`、规则模板或偏好优化方法可能更合适。

| 方法 | 适用任务 | 标注成本 | 训练复杂度 | 对齐效果 |
|---|---|---|---|---|
| SFT | 固定格式回答、领域问答、风格迁移 | 中 | 低到中 | 能模仿示范，但偏好比较能力有限 |
| RLHF | 开放聊天、安全对齐、复杂助手 | 高 | 高 | 对人类偏好拟合强，但工程成本高 |
| RLAIF | 用 AI 反馈替代部分人工反馈 | 中 | 高 | 降低人工成本，但依赖评审模型质量 |
| DPO/偏好优化 | 直接用偏好对优化策略 | 中 | 中 | 比 PPO 简化，适合已有偏好数据 |
| 规则/模板 | 表单回复、合规话术、结构化输出 | 低 | 低 | 稳定可控，但泛化能力弱 |

术语 `RLAIF` 指 Reinforcement Learning from AI Feedback，也就是用 AI 模型提供部分偏好反馈。术语 `DPO` 指 Direct Preference Optimization，它直接利用偏好对优化模型，不单独训练奖励模型再跑 PPO，工程链路更短。

数据少时，优先考虑 SFT 或规则约束。因为 RLHF 需要偏好数据、奖励模型评估和多轮迭代，小数据下奖励模型容易过拟合。

安全要求高时，可以考虑 RLHF，但不能只依赖 RLHF。还需要安全数据、红队测试、规则拦截、人工审核和线上监控。RLHF 能改善模型倾向，但不能替代完整安全系统。

算力受限时，可以先做 SFT，再考虑 DPO 这类更轻量的偏好优化。PPO 需要 rollout、奖励模型推理、参考模型 KL 计算和策略更新，训练成本明显更高。

业务例子：搜索问答助手更依赖对齐，因为它要在有用性、引用可靠性、拒答和不确定性表达之间平衡。数据抽取器则可能不需要 RLHF，只要监督学习加 schema 校验就足够。前者是开放生成问题，后者是结构化预测问题，工程解法不应混用。

---

## 参考资料

1. [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
2. [Meta AI Research: Llama 2 Open Foundation and Fine-Tuned Chat Models](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
3. [Meta Llama 官方 GitHub 仓库](https://github.com/meta-llama/llama)
4. [Meta Llama MODEL_CARD.md](https://github.com/meta-llama/llama/blob/main/MODEL_CARD.md)
5. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
