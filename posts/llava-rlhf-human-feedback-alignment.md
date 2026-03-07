## 核心结论

LLaVA-RLHF 的关键，不是把文本领域的 RLHF 机械迁移到多模态，而是把“图像事实”显式接入奖励学习链路。RLHF 可以先粗略理解成两步：

1. 先让人类比较两个回答，训练一个“奖励模型”去模拟这种偏好。
2. 再让策略模型朝高奖励方向优化。

在纯文本任务里，奖励模型只看文本通常还能工作；但在视觉语言模型里，如果奖励模型不看图像，只看回答文本，它很容易把“像正确答案的表达方式”误判为“真正正确”。结果就是模型语言很顺，但事实不对，这就是视觉幻觉。

LLaVA-RLHF 中的 Fact-RLHF 解决的正是这个问题。它做了两件事：

1. 奖励模型不只看候选回答，还看图像及其事实辅助信息，例如 caption、候选选项、问题上下文。
2. 策略优化不再只追求“更像人类喜欢的语气”，而是追求“在图像条件下更符合人类偏好的事实表达”。

这件事的工程意义很直接。对于客服、质检、商品审核、图像问答这类场景，系统往往允许模型说“不确定”，但不能接受模型编造。SFT 只能让模型学会常见格式和常见答案分布，很难系统性抑制“看起来合理、实际上不在图里”的错误；Fact-RLHF 则更适合把这种错误压下去。

先看一个简化对比表：

| 方案 | 奖励是否看图像事实 | 主要改善目标 | 幻觉控制 | 额外成本 |
|---|---|---|---|---|
| 仅 SFT | 否 | 学会任务格式与常见回答 | 中 | 低到中 |
| 文本 RLHF | 弱 | 提升偏好一致性 | 容易被“话术”欺骗 | 中 |
| Fact-RLHF | 是 | 降低视觉事实错误 | 强 | 中到较高 |
| 更细粒度反馈方案（如 RLHF-V） | 是 | 纠正局部错误与细节事实 | 更强 | 更高 |

用一个真实业务场景看差异更直观。用户上传商品图，问：“这个杯子有没有盖子，材质像不像不锈钢？”  
如果模型没看清图，或者奖励模型根本不把图像事实当约束，它很可能输出：

- “该杯子带有盖子，整体为金属材质，适合保温使用。”

这类回答语言很完整，但很可能是模板化脑补。LLaVA-RLHF 的目标，是让模型更倾向于输出：

- “图中杯口上方未见盖子结构；杯体表面有金属反光，但仅凭图片无法完全确认材质是否为不锈钢。”

前者“像客服”，后者“像基于图像做判断的客服”。LLaVA-RLHF 想优化的是后者。

---

## 问题定义与边界

问题定义可以写得非常明确。给定图像 $x$、问题 $q$、回答 $y$，我们希望模型输出既流畅，又满足图像条件下的事实一致性。这里的“事实一致”不是泛泛而谈的“真理”，而是更窄、更工程化的定义：

$$
y \text{ should be grounded in } (x, q, f)
$$

其中 $f$ 表示额外事实辅助信息，例如 caption、候选选项、人工标注属性等。换成白话，就是：回答不能违背当前图片里能看到的内容；如果系统还给了 caption 或选项，回答也不能与这些已知事实冲突。

多模态场景比纯文本更难，至少有三类原因：

| 难点 | 白话解释 | 直接后果 |
|---|---|---|
| 图像信息稠密 | 一张图里同时包含对象、位置、颜色、材质、数量、关系 | 模型容易漏看、错看、脑补 |
| 偏好不等于事实 | 人类往往更喜欢完整、礼貌、像客服的话术 | 奖励模型可能学会迎合文风，而不是核对图像 |
| 标注成本高 | 标注者必须同时核对图和文，还要比较两个回答 | 高质量偏好数据更贵、更稀缺 |

最常见的失败模式是：两个候选回答里，语言更圆滑的那个，不一定更符合图片。

看一个玩具例子。图里只有一个红苹果和一个空盘子，问题是“盘子里有几种水果？”

- 回答 A：“图中只有一个苹果，因此水果种类为 1。”
- 回答 B：“图中可能还有未完全展示的水果，因此至少 2 种。”

如果只看语言，B 听起来更谨慎，甚至更像“考虑周全”的回答；但如果看图像事实，A 才是更好的答案。多模态偏好标注的价值，就在于让标注者在这种成对候选里选出“事实更对”的那个，而不是“措辞更像标准答案”的那个。

偏好建模通常写成 Bradley-Terry 形式：

$$
P(\text{human selects } s_1)=\sigma(r_\phi(s_1)-r_\phi(s_2))
$$

其中：

- $r_\phi$ 是奖励模型，对候选答案给一个标量分数；
- $\sigma(z)=\frac{1}{1+e^{-z}}$ 是 sigmoid；
- 分差越大，人类更可能偏好分数更高的答案。

同一个式子也常写成 softmax 形式：

$$
P(\text{choose } s_1)=\frac{e^{r_\phi(s_1)}}{e^{r_\phi(s_1)}+e^{r_\phi(s_2)}}
$$

这两个写法本质等价。因为：

$$
\frac{e^{r_1}}{e^{r_1}+e^{r_2}}
=
\frac{1}{1+e^{-(r_1-r_2)}}
=
\sigma(r_1-r_2)
$$

这个等价关系很重要，它说明偏好学习真正关心的是“两个候选的相对分差”，而不是某个答案的绝对分数。

边界也要说清楚。LLaVA-RLHF 解决的是“图像条件下的偏好对齐与事实约束”，它不等于下面三件事：

1. 不等于完整视觉推理保证。图片模糊、遮挡严重、角度不足时，模型仍可能无法判断。
2. 不等于外部世界知识校验。比如品牌年份、历史背景、法律规则，这些往往超出图像本身。
3. 不等于全量安全对齐。偏见、隐私、越权建议、攻击性内容仍需要独立的安全机制。

所以它最适合的任务边界是：图像是主要事实来源，业务对“不要编造”比对“回答尽量丰富”更敏感。

---

## 核心机制与推导

LLaVA-RLHF 可以拆成三层：候选生成、偏好学习、策略优化。按流水线看，会更容易理解。

| 层级 | 输入 | 输出 | 目标 |
|---|---|---|---|
| 候选生成 | 图像 + 问题 | 多个候选回答 | 产生可比较的答案对 |
| 奖励学习 | 图像 + 问题 + 候选 + 事实信息 | 奖励分数 | 学会区分“事实更好”的回答 |
| 策略优化 | 当前策略采样的回答 + 奖励 | 更新后的策略 | 提高高奖励回答的概率 |

### 1. 候选生成

第一层是候选生成。先用已有视觉语言模型，对同一组输入 $(x, q)$ 采样多个候选回答。为什么一定要多个？因为没有比较，就没有偏好数据。  
如果两个候选非常接近，人类标注几乎没有信息量；如果两个候选一好一坏差异足够明显，偏好数据才能告诉奖励模型“到底什么样的回答更值得奖励”。

常见做法包括：

- 用不同 temperature 采样；
- 用不同 decoding 随机种子；
- 混合“保守答案”和“详细答案”；
- 专门采样一些容易出错的负例。

这一步的目的不是追求最终答案质量，而是提高“可比较性”。

### 2. 奖励模型训练

第二层是奖励模型训练。文本 RLHF 里，奖励模型常常只看 prompt 和回答；但在 LLaVA-RLHF 里，奖励模型需要显式看到图像条件和事实辅助信息：

$$
r_\phi = r_\phi(v, q, y, f)
$$

其中：

- $v$：图像编码后的视觉表示；
- $q$：用户问题；
- $y$：候选回答；
- $f$：事实辅助信息，例如 caption、ground-truth choice、额外上下文。

Fact-RLHF 的核心，不是简单把更多输入塞进模型，而是改变奖励学习的判别依据。  
如果奖励模型只看文本，它很可能学到以下错误模式：

- 回答越长越像高分；
- 越保守、越礼貌越像高分；
- 越像标准客服模板越像高分。

这类现象就是 reward hacking 在多模态场景下的典型表现。所谓 reward hacking，就是策略模型发现了“怎么骗过评分规则”，而不是“怎么真正完成任务”。

在视觉问答里，reward hacking 通常表现为：

- 过度使用模糊限定词，如“根据图片推测”“大概率”“看起来像”；
- 用常识替代观察，如“杯子一般有盖”“运动鞋通常有网孔”；
- 用套话抬高流畅度，如“整体设计简洁、实用性强”。

这些表达不一定错，但如果它们脱离图像证据，奖励模型就会被“文本风格”劫持。

奖励模型训练常用的 pairwise loss 是：

$$
\mathcal{L}_{\text{RM}}
=
-\log \sigma\left(r_\phi(v,q,y^+,f)-r_\phi(v,q,y^-,f)\right)
$$

其中 $y^+$ 是被人类选中的回答，$y^-$ 是被拒绝的回答。  
这个损失函数的含义很直接：如果奖励模型给好答案的分数明显高于坏答案，loss 就小；反之就大。

再往前推一步，它相当于最大化以下对数似然：

$$
\log P(y^+ \succ y^- \mid x,q,f)
=
\log \sigma(r^+ - r^-)
$$

因此 RM 训练本质上是在拟合“人类在图像条件下更偏好哪一个回答”。

### 3. 策略优化

第三层是策略优化。策略模型参数记为 $\theta$，目标是最大化在当前图像条件下的期望奖励：

$$
\max_\theta \mathbb{E}_{y \sim \pi_\theta(\cdot|x,q)}[r_\phi(v,q,y,f)]
$$

LLaVA-RLHF 使用 PPO 进行优化。PPO 可以先直白理解成：每轮更新都往高奖励方向走，但不能一下走太猛，否则模型可能迅速崩成奇怪的模板化分布。

其常见剪切目标写成：

$$
L^{\text{PPO}}(\theta)=
\mathbb{E}\left[
\min\left(
\rho_t(\theta)A_t,\ 
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

其中：

$$
\rho_t(\theta)=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

含义分别是：

- $\rho_t(\theta)$：新旧策略在当前动作上的概率比；
- $A_t$：advantage，表示“这一步比平均水平好多少”；
- $\epsilon$：clip 范围，限制一次更新不能过大。

如果 $\rho_t$ 超出 $[1-\epsilon, 1+\epsilon]$，PPO 就把它截住。  
这能减少“为了追高奖励，一步把语言分布推坏”的风险。

实际训练里，目标函数通常还会加入 KL 惩罚，限制新策略离参考策略太远：

$$
\max_\theta\ 
\mathbb{E}[r_\phi(v,q,y,f)]
-
\beta\,\mathrm{KL}\!\left(\pi_\theta \,\|\, \pi_{\text{ref}}\right)
$$

这里的 $\beta$ 控制“追奖励”和“别偏离原模型太远”之间的平衡。  
如果没有这个约束，模型可能出现：

- 回答越来越短；
- 回答越来越模板化；
- 只学会一个“保守高分句式”。

### 最小数值例子

假设同一张图、同一个问题下，有两个候选答案 A 和 B：

- $r_\phi(A)=0.95$
- $r_\phi(B)=0.65$

那么人类更可能选择 A 的概率是：

$$
P(A)=\frac{e^{0.95}}{e^{0.95}+e^{0.65}} \approx 0.57
$$

也可以写成：

$$
P(A)=\sigma(0.95-0.65)=\sigma(0.30)\approx 0.57
$$

这说明只要奖励差是 0.30，就已经形成了可观测的偏好倾向。  
如果差值变成 2.0，则：

$$
P(A)=\sigma(2.0)\approx 0.88
$$

这时偏好已经非常明确。

从工程角度看，真正关键的不是某个数值大小，而是奖励模型是否能稳定地区分下面两类答案：

| 类型 | 表面特征 | 实际价值 |
|---|---|---|
| 看起来顺的答案 | 语言完整、礼貌、像模板 | 不一定符合图像 |
| 事实上对的答案 | 结论锚定在“图中可见” | 更可靠，可复核 |

真实场景可以再看一遍图文客服。用户上传一张鞋子图片，问：“鞋带是黑色还是深蓝色？鞋面有没有网孔？”  
如果奖励模型没做事实增强，它可能偏好这类回答：

- “根据图片和常见款式判断，鞋带偏深色，鞋面采用透气设计。”

这句话读起来很自然，但“常见款式判断”本身就暴露了它在用经验补事实。  
Fact-RLHF 更倾向于奖励这种回答：

- “图中鞋带更接近深蓝色；鞋面可见网孔结构。若需要，我可以继续描述鞋底和鞋帮细节。”

后者的优势不是更华丽，而是判断都有可见依据。

---

## 代码实现

工程流程可以概括为四步：

1. 用基座 VLM 对同一图文输入生成多个候选回答。
2. 让标注者从候选中选出更符合图像事实的答案，形成偏好对。
3. 用偏好对训练事实增强奖励模型。
4. 用 PPO 或近似策略优化方法根据奖励模型更新策略模型。

偏好数据可以用一个简单 JSON 结构表示：

| 字段 | 含义 | 说明 |
|---|---|---|
| `image_id` | 图像标识 | 指向图片或离线特征 |
| `question` | 用户问题 | 当前轮文本输入 |
| `fact` | 事实辅助信息 | caption、选项、属性标注等 |
| `answer_a` | 候选回答 A | 来自模型采样 |
| `answer_b` | 候选回答 B | 来自模型采样 |
| `chosen` | 人类偏好 | `a` 或 `b` |
| `note` | 标注备注 | 可选，记录为什么 A/B 更好 |

下面给出一个可直接运行的 Python 玩具实现。它不训练真实 VLM，但把三件关键事串了起来：

- 偏好概率如何计算；
- 奖励模型 pairwise loss 如何计算；
- PPO 裁剪目标如何限制更新幅度。

```python
import math
from dataclasses import dataclass


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def preference_prob(r1: float, r2: float) -> float:
    """Bradley-Terry / sigmoid preference."""
    return sigmoid(r1 - r2)


def rm_pairwise_loss(r_chosen: float, r_rejected: float) -> float:
    """Negative log-likelihood for pairwise preference learning."""
    p = preference_prob(r_chosen, r_rejected)
    return -math.log(p)


def ppo_clip_objective(
    old_logprob: float,
    new_logprob: float,
    advantage: float,
    eps: float = 0.2,
) -> float:
    """
    PPO clipped surrogate objective for one sample.
    Returns the contribution before the outer '-' sign in gradient descent code.
    """
    ratio = math.exp(new_logprob - old_logprob)
    unclipped = ratio * advantage
    clipped_ratio = min(max(ratio, 1.0 - eps), 1.0 + eps)
    clipped = clipped_ratio * advantage
    return min(unclipped, clipped)


@dataclass
class PreferenceSample:
    image_id: str
    question: str
    fact: str
    answer_a: str
    answer_b: str
    chosen: str  # 'a' or 'b'


def demo() -> None:
    sample = PreferenceSample(
        image_id="shoe-001",
        question="鞋带是黑色还是深蓝色？鞋面有没有网孔？",
        fact="caption: 一双深蓝鞋带的运动鞋，鞋面可见透气网眼。",
        answer_a="根据常见款式判断，鞋带偏深色，鞋面采用透气设计。",
        answer_b="图中鞋带更接近深蓝色；鞋面可见网孔结构。",
        chosen="b",
    )

    # 假设奖励模型给出的分数
    r_a = 0.65
    r_b = 0.95

    # 1) 偏好概率
    p_b = preference_prob(r_b, r_a)
    print(f"P(human prefers B) = {p_b:.6f}")

    # 2) 奖励模型损失
    loss = rm_pairwise_loss(r_b, r_a)
    print(f"Reward model pairwise loss = {loss:.6f}")

    # 3) PPO 裁剪目标
    old_logprob = math.log(0.40)
    new_logprob = math.log(0.48)
    advantage = 1.5

    obj = ppo_clip_objective(
        old_logprob=old_logprob,
        new_logprob=new_logprob,
        advantage=advantage,
        eps=0.2,
    )
    print(f"PPO clipped objective = {obj:.6f}")

    # 基本断言，保证示例数值正确
    assert 0.57 < p_b < 0.58
    assert abs(loss - (-math.log(p_b))) < 1e-12
    assert abs(obj - 1.8) < 1e-9

    # 如果新策略更新过猛，会被 clip
    too_new_logprob = math.log(0.70)
    obj2 = ppo_clip_objective(
        old_logprob=old_logprob,
        new_logprob=too_new_logprob,
        advantage=advantage,
        eps=0.2,
    )
    print(f"PPO clipped objective after too-large update = {obj2:.6f}")
    assert abs(obj2 - 1.8) < 1e-9


if __name__ == "__main__":
    demo()
```

如果运行这段代码，输出会体现三件事：

| 现象 | 含义 |
|---|---|
| `P(human prefers B)` 约为 `0.57` | 分差 `0.30` 已能形成偏好 |
| `Reward model pairwise loss` 为正数 | 奖励模型还在学习把好答案分得更高 |
| 第二次 PPO 目标仍是 `1.8` | 更新过猛时，PPO 会触发裁剪 |

如果写成训练伪代码，逻辑通常如下：

```python
# 1. 采样候选回答
responses = policy.generate(image, question, num_samples=4)

# 2. 标注者基于图像和 fact_context 选出更好的回答
chosen, rejected = human_label(responses, image=image, fact_context=fact_context)

# 3. 训练奖励模型
r_chosen = reward_model(image, question, chosen, fact_context)
r_rejected = reward_model(image, question, rejected, fact_context)
rm_loss = -log_sigmoid(r_chosen - r_rejected)

optimizer_rm.zero_grad()
rm_loss.backward()
optimizer_rm.step()

# 4. 用 PPO 更新策略
sampled_answer = policy.generate(image, question, num_samples=1)
old_logprob = ref_policy.logprob(image, question, sampled_answer)
new_logprob = policy.logprob(image, question, sampled_answer)

reward = reward_model(image, question, sampled_answer, fact_context)
value = value_fn(image, question, sampled_answer)
advantage = reward - value

policy_loss = -ppo_clip_objective(old_logprob, new_logprob, advantage, eps=0.2)
kl_penalty = beta * kl_divergence(policy, ref_policy, image, question)
total_loss = policy_loss + kl_penalty

optimizer_policy.zero_grad()
total_loss.backward()
optimizer_policy.step()
```

需要注意，这段伪代码省略了真实系统里的多个细节，但工程上有三个点不能省：

| 关键点 | 为什么重要 | 忽略后会怎样 |
|---|---|---|
| 视觉嵌入与文本对齐一致 | RM 和策略必须看见“同一种图文对应关系” | 奖励信号会失真 |
| KL 约束 | 防止模型为了追奖励而分布崩坏 | 输出可能突然变短或模板化 |
| 奖励归一化 | 不同 batch、不同任务的奖励尺度不一样 | PPO 容易不稳定 |

还可以补一个更贴近新手理解的直观说明：

- SFT 像“老师拿标准答案教学生背”；
- 奖励模型像“助教学会判断两份作业哪份更好”；
- PPO 像“根据助教评分，逐步调整学生的答题习惯”。

LLaVA-RLHF 的难点在于，这个“助教”不能只看作文写得像不像标准答案，还必须盯着题目配图本身。

---

## 工程权衡与常见坑

LLaVA-RLHF 最核心的工程权衡，不在“要不要做 RLHF”，而在“奖励模型到底看什么、看多少、看得是否稳定”。

先看常见坑：

| 坑 | 现象 | 根因 | 规避策略 |
|---|---|---|---|
| Reward hacking | 回答更圆滑但更不真实 | RM 学会偏好话术而非事实 | 给 RM 接入图像、caption、选项、属性约束 |
| 视觉偏好不足 | RM 区分不了颜色、数量、位置等细节错误 | 偏好数据覆盖太窄 | 专门补颜色/数量/材质/关系类难例 |
| 多模态错位 | 模型说的对象与图中区域对不上 | 视觉编码器、投影层、策略训练不同步 | 保持训练链路一致，避免特征漂移 |
| 奖励过拟合 | 验证集外表现明显下降 | 标注风格太单一 | 增加反例、跨场景验证、跨任务采样 |
| PPO 不稳定 | 回答突然极短、极保守、模板化 | 奖励尺度、KL、学习率设置不当 | 奖励归一化、控制 KL、缩小步长 |

一个典型对照是“只看回答文本的 RM”和“看图像事实增强信息的 RM”。

问题：图中是一只白猫躺在木椅上。  
候选回答 1：“图中是一只白猫躺在椅子上。”  
候选回答 2：“这是一只可爱的宠物，正在舒适地休息，看起来非常放松。”

如果 RM 只看文本，回答 2 很可能因为“描述更丰富、更有人味”而被打高分；但如果 RM 同时看到图像或 caption，它通常更容易判断回答 1 更好，因为回答 1 的内容更可验证，事实密度更高，修辞噪声更低。

这里有一个很实用的工程判断标准：  
当业务目标是“减少编造”时，奖励模型应该优先偏好“可验证性”，而不是“文学性”。

再看成本。偏好标注并不便宜，因为标注员需要同时完成三件事：

1. 看懂图片；
2. 理解问题；
3. 比较两个回答谁更贴图。

因此，多模态偏好数据的单位成本通常高于纯文本偏好数据。公开资料中常见的量级是几千美元做一万级别样本，这并不是一个可以随意试错的成本。更现实的做法通常不是一开始全量铺开，而是把预算集中在最危险的错误类型上。

推荐的落地顺序通常是：

1. 先用高质量多模态 SFT 把基础问答能力拉起来。
2. 再把偏好数据集中投在高风险幻觉点上，如颜色、数量、材质、位置关系、是否存在某对象。
3. 用独立验证集专门测“事实不符率”，而不是只看 BLEU、ROUGE、CIDEr 这类文本相似指标。

可以把评估指标拆成下面几类：

| 指标 | 是否足够 | 作用 |
|---|---|---|
| BLEU / ROUGE | 不够 | 只看文本表面相似性 |
| GPT 风格评分 | 不够 | 容易被流畅度影响 |
| 幻觉率 / 事实错误率 | 更关键 | 直接衡量是否编造 |
| 人工偏好胜率 | 关键 | 看实际使用体验 |
| 拒答合理率 | 关键 | 衡量“不确定时是否会克制” |

真实业务里，最常见的高价值收益并不是“回答更漂亮”，而是“错误更可控”。  
例如在商品审核系统里，运营会问：

- 主图里是否含有品牌 logo？
- 是否出现两件以上商品？
- 是否附带赠品配件？
- 颜色是否与标题宣称一致？

这些问题都有一个共同特征：错误比含糊更糟。  
在这种场景里，“不确定，请人工复核”通常比“凭经验猜一个”更安全。Fact-RLHF 的价值，就在于把奖励偏向“事实优先、宁缺毋滥”的输出风格。

---

## 替代方案与适用边界

不是所有团队都需要直接上 LLaVA-RLHF。更合理的选择，通常取决于预算、延迟要求、可解释性要求，以及业务对幻觉的容忍度。

先看一张分层表：

| 路线 | 成本 | 准确率上限 | 适合场景 |
|---|---|---|---|
| 仅 SFT | 最低 | 中 | 原型验证、内容生成、低风险问答 |
| SFT + 自动校验 | 低到中 | 中 | 轻量图文问答、容错较高场景 |
| Fact-RLHF | 中 | 高 | 客服、审核、事实敏感问答 |
| 更细粒度多模态 RLHF | 最高 | 更高 | 高风险业务、复杂细节校验 |

可以把它理解成一条成本与精度曲线：

$$
\text{SFT} \rightarrow \text{Fact-RLHF} \rightarrow \text{Fine-grained RLHF-V}
$$

越往右：

- 标注更贵；
- 管线更复杂；
- 奖励设计更难；
- 但对细粒度幻觉的压制通常更强。

### 低配可落地路线

如果预算不足，通常不必在第一天就完整实现 LLaVA-RLHF。更务实的路线是：

1. 先用 SFT 建立基础视觉问答能力。
2. 只收集 1k 到 2k 条最关键的视觉偏好样本。
3. 把偏好标注集中在事故代价最高的任务上。
4. 用自动 caption、规则模板、已有 QA 数据扩充负例，但保留人工偏好作高质量锚点。

这条路线的逻辑是：  
SFT 解决“会不会答”，少量 Fact-RLHF 解决“答的时候会不会编”。

### 为什么不能只靠自动生成数据

很多团队会问：既然人工偏好贵，能不能完全用模型自己生成偏好数据？  
答案通常是否定的。自动数据当然有价值，但更适合作为扩覆盖工具，而不是替代高质量人工偏好。原因很简单：

- 如果教师本身就会幻觉，它生成的“事实判断”也会带幻觉；
- 如果自动标注器本身偏好流畅文风，奖励模型会学到同样的偏差；
- 如果没有人工高质量样本作校准，系统很难知道自己是在“减少错误”还是“换一种方式犯错”。

### Fact-RLHF 的适用边界

Fact-RLHF 更适合“图像是主要证据源”的任务。例如：

- 商品图问答；
- 视觉客服；
- 图片审核；
- 场景描述；
- 简单属性识别。

如果任务大量依赖外部知识，例如：

- “这款相机属于哪一年发布的系列？”
- “这件文物出自哪个朝代？”
- “图中药品是否符合某项最新监管要求？”

那么单靠图像事实增强奖励通常不够，还需要把检索、知识库、规则系统、外部校验链路一起接进来。

因此，判断一个任务是否值得做 Fact-RLHF，可以先看三件事：

1. 业务是否对视觉事实错误高度敏感。
2. 团队是否能收集到一批高质量偏好对。
3. 是否有评测集专门测幻觉，而不是只测语言流畅度。

满足这三点时，Fact-RLHF 往往比“继续堆更多 SFT 数据”更有效。  
原因不复杂：SFT 主要教模型“该怎么说”，Fact-RLHF 更直接地教模型“什么时候不能乱说”。

---

## 参考资料

下面给出一组更完整的参考资料，按“原始论文/官方页面/扩展路线”三类组织：

| 文献/页面 | 类型 | 核心贡献 | 适合怎么读 |
|---|---|---|---|
| Aligning Large Multimodal Models with Factually Augmented RLHF | 论文 | 提出 Fact-RLHF，系统说明奖励模型为何要看事实辅助信息 | 先读摘要、方法、实验 |
| LLaVA-RLHF 官方页面 | 项目页 | 汇总模型、数据、评测、代码入口 | 适合快速建立整体认知 |
| LLaVA-RLHF GitHub | 代码仓库 | 给出 SFT/RLHF 训练入口与工程说明 | 适合看训练配置和数据组织 |
| MMHal-Bench | 评测基准 | 专门惩罚视觉幻觉 | 适合理解“如何测少编造” |
| RLHF-V | 扩展路线 | 用更细粒度纠错反馈做对齐 | 适合比较更高成本方案的收益 |

1. LLaVA-RLHF 官方页面：项目概览、Fact-RLHF、模型与数据入口。  
   https://llava-rlhf.github.io/

2. 官方论文（ACL Anthology）：**Aligning Large Multimodal Models with Factually Augmented RLHF**。这是最完整的一手来源，适合看方法、实验和 MMHal-Bench。  
   https://aclanthology.org/2024.findings-acl.775/

3. 项目代码仓库：包含训练说明、SFT/RLHF 目录、模型和数据链接。  
   https://github.com/llava-rlhf/LLaVA-RLHF

4. IBM Research 页面：对论文摘要和贡献点有一页式说明，适合快速复习。  
   https://research.ibm.com/publications/aligning-large-multimodal-models-with-factually-augmented-rlhf

5. RLHF-V 官方页面：更细粒度纠错式多模态对齐方案，适合与 Fact-RLHF 做成本-收益比较。  
   https://rlhf-v.github.io/

6. RLHF-V CVPR 2024 页面：包含摘要、poster、paper 入口，可用于理解“细粒度 correctional feedback”为何能进一步降低幻觉。  
   https://cvpr.thecvf.com/virtual/2024/poster/31610

7. 一篇偏工程理解的阅读笔记：重点讨论 reward hacking、事实增强奖励、标注成本等问题，适合作为辅助材料而不是一手来源。  
   https://zhangtemplar.github.io/align-fact/

从阅读顺序上，更推荐这样看：

1. 先看官方页面，建立整体框架。
2. 再读 ACL 论文的方法和实验部分。
3. 接着看代码仓库，理解训练管线如何落地。
4. 最后再看 RLHF-V，理解更细粒度反馈为什么可能进一步压低幻觉。
