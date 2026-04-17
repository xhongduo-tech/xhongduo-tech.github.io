## 核心结论

DPO，Direct Preference Optimization，直译是“直接偏好优化”。它做的事很直接：不给模型先单独训练一个奖励模型，再拿 PPO 去强化学习，而是直接拿“同一个问题下，回答 A 比回答 B 好”的偏好对训练策略模型。

它最常见的目标函数是：

$$
\mathcal{L}_{\text{DPO}}
=
-\mathbb{E}_{(x,y^+,y^-)\sim \mathcal D}
\left[
\log \sigma \left(
\beta \Big(
(\log \pi_\theta(y^+|x)-\log \pi_\theta(y^-|x))
-
(\log \pi_{\text{ref}}(y^+|x)-\log \pi_{\text{ref}}(y^-|x))
\Big)
\right)
\right]
$$

其中，$\pi_\theta$ 是训练中的策略模型，白话说就是“正在被更新的回答模型”；$\pi_{\text{ref}}$ 是参考模型，白话说就是“冻结不动、用来防止模型跑偏的旧模型”；$\beta$ 控制更新时离参考模型有多远。

一句话理解：DPO 把“人类更喜欢哪个回答”变成一个二分类问题，让新模型相对参考模型，更倾向于把概率分给被偏好的回答。

玩具例子是这样的：同一个提示词下，人类在两段回答里选了更好的那段。DPO 不关心绝对分数，只关心“选中的回答，是否比被拒绝的回答更值得模型生成”。它比较的是两段回答的对数概率差，再和参考模型做差，最后进 sigmoid。

一个简化流程可以写成：

```text
偏好对 (x, y+, y-)
      ↓
算 4 个 log prob:
logπ(y+), logπ(y-), logπref(y+), logπref(y-)
      ↓
margin = β * [ (logπ+ - logπ-) - (logπref+ - logπref-) ]
      ↓
loss = -log sigmoid(margin)
      ↓
反向传播，只更新 πθ
```

---

## 问题定义与边界

先把问题说清楚。DPO 的输入不是环境奖励，不是在线交互轨迹，而是离线偏好数据。离线，白话说就是“训练前已经收集好的数据，不在训练过程中边跑边采样”。一条数据通常长这样：

| 字段 | 含义 | 白话解释 |
|---|---|---|
| `x` | prompt | 用户问题或上下文 |
| `y+` | chosen | 人类选中的更好回答 |
| `y-` | rejected | 人类认为更差的回答 |
| `label` | preference | 谁赢谁输的标注 |

所以 DPO 解决的问题不是“模型怎么在环境里探索”，而是“给定一堆偏好对，怎么直接把模型调成更符合这些偏好”。

这决定了它的边界。

第一，DPO 只能覆盖偏好数据所在的分布。分布，白话说就是“训练里常见的问题和回答类型”。如果你的偏好数据全是客服问答，那 DPO 学到的主要就是客服风格、客服安全边界、客服回答长度。它不会自动长出医学问答、代码修复、长链推理这些新能力。

第二，DPO 不会凭空创造新偏好。它不是搜索算法，也不是在线强化学习。它不会在训练时主动探索新回答，再让人重新打分。数据里没出现过的偏好结构，它只能靠已有模型能力去外推，这种外推通常不可靠。

第三，数据覆盖度决定上限。覆盖度，白话说就是“你想让模型学会的情况，数据里到底出现了多少”。如果偏好集里只有“短回答更好”，模型就容易过度偏向短回答；如果偏好集里几乎没有多轮推理、代码解释、拒答边界，它在这些场景就学不稳。

把它和传统 RLHF 的数据路径放在一起看更清楚：

| 方案 | 训练输入 | 是否需要在线采样 | 是否单独训练 RM | 核心风险 |
|---|---|---:|---:|---|
| DPO | 离线偏好对 | 否 | 否 | 强依赖偏好覆盖 |
| RLHF(PPO) | 偏好数据 + rollout | 是 | 是 | 系统复杂、训练不稳 |
| 纯 SFT | 指令-答案对 | 否 | 否 | 学“模仿”，不直接学“偏好差异” |

一个简化案例：如果你的数据全是“客服回复要礼貌、简短、先道歉”，那 DPO 训练出来的模型，会在这个闭环里更稳定，但它对“复杂故障定位”“跨轮追问”“用户在抱怨时如何平衡解释与解决方案”这些没覆盖的细节，仍然可能表现差。

---

## 核心机制与推导

DPO 的关键点不是“它看起来像分类”，而是“它确实来自一个带 KL 约束的奖励最大化问题”。

KL 散度，白话说就是“新模型和旧模型差了多远”。在传统 RLHF 里，常见目标可以写成：最大化奖励，同时不要离参考模型太远。它的最优策略满足：

$$
\pi^*(y|x)\propto \pi_{\text{ref}}(y|x)\exp\left(\frac{r(x,y)}{\beta}\right)
$$

把它改写一下，可以得到隐式奖励：

$$
r(x,y)=\beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}+\beta \log Z(x)
$$

这里的 $Z(x)$ 只和 prompt 有关，不和回答 $y$ 有关。所以当我们比较同一个 prompt 下两个回答 $y^+$ 和 $y^-$ 时，这个常数会抵消。于是得到：

$$
r(x,y^+) - r(x,y^-)
=
\beta \left[
\log \frac{\pi_\theta(y^+|x)}{\pi_{\text{ref}}(y^+|x)}
-
\log \frac{\pi_\theta(y^-|x)}{\pi_{\text{ref}}(y^-|x)}
\right]
$$

如果再假设偏好服从 Bradley-Terry 模型，也就是“哪个回答被选中，取决于两个回答分数差”，那么：

$$
P(y^+ \succ y^-|x)
=
\sigma(r(x,y^+) - r(x,y^-))
$$

把上面的隐式奖励代进去，就得到 DPO 损失。

这一步的意义很大。它说明 DPO 不是拍脑袋把 sigmoid 套在偏好对上，而是把“奖励模型 + KL 约束策略优化”整体折叠成了一个直接可训练的目标。也因此，论文才会说“你的语言模型其实暗中就是一个奖励模型”。

看一个数值玩具例子。设 $\beta=0.1$，并且：

| 项 | 数值 |
|---|---:|
| $\log \pi_\theta(y^+|x)$ | -2.0 |
| $\log \pi_{\text{ref}}(y^+|x)$ | -2.5 |
| $\log \pi_\theta(y^-|x)$ | -3.0 |
| $\log \pi_{\text{ref}}(y^-|x)$ | -2.0 |

那么：

$$
\text{logit}
=
0.1 \times \big(({-2.0}+2.5)-({-3.0}+2.0)\big)
=
0.1\times(0.5-(-1.0))
=
0.15
$$

损失就是：

$$
\text{loss}=-\log \sigma(0.15)\approx 0.62
$$

这个值不是越小越“绝对正确”，而是表示当前模型还可以继续把偏好边界拉大。重点在于，整个样本只需要 4 个 log probability，不需要单独训练奖励模型，也不需要像 PPO 那样反复 rollout。

真实工程里，常见做法是：先拿一个 SFT 模型当参考模型，再用人工偏好对或规则打分生成的 chosen/rejected 数据做 DPO 微调。比如客服机器人项目中，团队会把“是否解决问题、是否礼貌、是否合规”合成偏好信号，形成离线偏好对，然后直接训练回答模型，而不是再搭一个 RM+PPO 的完整堆栈。

---

## 代码实现

实现 DPO，主流程并不复杂：

1. 准备 batch：每条样本都包含 `prompt`、`chosen`、`rejected`。
2. 用参考模型前向，得到 `chosen` 和 `rejected` 的 log probability。
3. 用当前策略模型前向，得到同样两组 log probability。
4. 计算 margin 和 DPO loss。
5. 只对策略模型反向传播，参考模型冻结。

下面先给一个可运行的 Python 玩具实现，目的是把公式和数值算清楚：

```python
import math

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def dpo_loss(logp_pos, logp_neg, logp_ref_pos, logp_ref_neg, beta=0.1):
    margin = beta * ((logp_pos - logp_neg) - (logp_ref_pos - logp_ref_neg))
    loss = -math.log(sigmoid(margin))
    return margin, loss

# 玩具例子
margin, loss = dpo_loss(
    logp_pos=-2.0,
    logp_neg=-3.0,
    logp_ref_pos=-2.5,
    logp_ref_neg=-2.0,
    beta=0.1,
)

assert round(margin, 2) == 0.15
assert abs(loss - 0.620957) < 1e-4

# 若模型更偏好 chosen，loss 应该更小
_, better_loss = dpo_loss(
    logp_pos=-1.5,
    logp_neg=-3.5,
    logp_ref_pos=-2.5,
    logp_ref_neg=-2.0,
    beta=0.1,
)
assert better_loss < loss
print("ok")
```

在 PyTorch 里，核心代码通常就是这一行：

```python
loss = -torch.log(torch.sigmoid(
    beta * ((logp_pos - logp_neg) - (logp_ref_pos - logp_ref_neg))
)).mean()
```

更接近真实训练的伪代码如下：

```python
# policy_model: 可训练
# ref_model: 冻结
# batch: prompt, chosen, rejected

with torch.no_grad():
    ref_logp_pos = sequence_logprob(ref_model, prompt, chosen)
    ref_logp_neg = sequence_logprob(ref_model, prompt, rejected)

logp_pos = sequence_logprob(policy_model, prompt, chosen)
logp_neg = sequence_logprob(policy_model, prompt, rejected)

margin = beta * ((logp_pos - logp_neg) - (ref_logp_pos - ref_logp_neg))
loss = -torch.nn.functional.logsigmoid(margin).mean()

loss.backward()
optimizer.step()
optimizer.zero_grad()
```

这里有两个实现细节必须说清楚。

第一，`sequence_logprob` 一般是“答案 token 的条件对数概率求和”或“求平均”。求和更接近原始序列似然；求平均会减弱长度影响。两者没有绝对对错，但训练和评估必须一致。

第二，reference model 通常是初始策略的冻结副本。冻结，白话说就是“它只参与计算，不更新参数”。这样做的目的是给策略一个稳定基准，避免模型为了赢偏好对而彻底偏离原有语言分布。

真实工程例子：在 Hugging Face TRL 或 NVIDIA NeMo 的 DPO 训练里，数据集字段通常就叫 `prompt`、`chosen`、`rejected`，训练器会同时算 policy 和 reference 的 log probs，再记录 reward margin、preference accuracy、KL 相关指标。整体比 PPO 训练短很多，也更容易在单机多卡或 LoRA 场景中落地。

---

## 工程权衡与常见坑

DPO 的优势很明确，但坑也很集中。

| 问题 | 原因 | 应对 |
|---|---|---|
| 输出风格越来越窄 | 偏好数据单一，模型持续压低 rejected 模式 | 扩充偏好多样性，混入 SFT loss 或保留多风格样本 |
| 训练后“更安全但更笨” | beta 太大，模型过度贴近参考模型 | 调小 $\beta$，观察 reward margin 和人工评测 |
| 训练不收敛 | 偏好对噪声高，chosen/rejected 本身差异不稳定 | 清洗低一致性样本，过滤质量差标注 |
| 长答案吃亏 | 序列 log prob 求和时受长度影响 | 试平均 log prob，或统一答案长度分布 |
| 只会压 rejected，不会真正提升 chosen | 数据太弱，reference 对 chosen 本来就不差 | 提升数据难度，加入更高质量 preferred 样本 |

一个常见误区是把 DPO 理解成“只要有偏好对就一定比 PPO 好”。这不成立。DPO 省掉了 rollout 和奖励模型，代价就是更依赖离线数据本身。如果数据覆盖窄、标注偏、chosen/rejected 差距太小，DPO 很容易学成“特定风格过滤器”，而不是“更强的对齐模型”。

另一个常见坑是只盯训练 loss，不看 reward margin 和 preference accuracy。margin，白话说就是“chosen 相对 rejected 的隐式奖励差”。如果 loss 在降，但 margin 长期不增，往往说明模型只是在吃简单样本，或者 beta 让更新太保守。

还有一个工程现实：DPO 往往更擅长“压低明显差的回答”，不一定自动产生“远超参考模型的新好回答”。原因很简单，优化信号来自相对比较，不是开放式探索。所以当参考模型已经把 chosen 的概率打得不低时，进一步抬高它的梯度可能并不强。

---

## 替代方案与适用边界

如果把 DPO 放回整个对齐方法族里，它最适合的是“离线偏好数据充足、工程资源有限、想快速把 SFT 模型往偏好方向推一段”的场景。

| 方法 | 适合什么情况 | 不适合什么情况 |
|---|---|---|
| DPO | 已有较高质量偏好对，想低成本对齐 | 需要在线探索、新域偏好缺失 |
| RLHF(PPO) | 需要更灵活的奖励设计和在线采样 | 团队小、训练预算紧、系统复杂度受限 |
| GRPO/同类变体 | 想保留采样式优化，又减少 PPO 负担 | 数据和评估链路还不稳定 |
| 纯 SFT | 有高质量标准答案，但缺偏好标注 | 需要区分“哪个好”而不是“像不像答案” |

一个小团队例子：如果只有 2000 条人工偏好对、GPU 预算也有限，优先做 DPO 往往最现实。因为它直接复用现有 SFT 模型和偏好数据，不需要再训练 RM，也不需要 PPO rollout 基础设施。

反过来，如果你要做的是一个新域助手，数据每天都在变，团队还想让模型在上线后持续从交互里学习，那么只靠 DPO 通常不够。你需要更完整的 RLHF 或其变体，至少要有一种能把新采样结果转成奖励信号的机制。

所以最实用的判断标准不是“DPO 先进还是 PPO 先进”，而是这三个问题：

1. 你有没有足够好的离线偏好对？
2. 你现在要的是低成本稳定优化，还是在线探索新行为？
3. 你的参考模型本身是否已经具备足够的基础能力？

如果前两问的答案分别是“有”和“前者”，DPO 通常就是更合适的起点。

---

## 参考资料

- 原始论文：[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://huggingface.co/papers/2305.18290)
- NVIDIA 工程文档：[Model Alignment by DPO, RPO, and IPO](https://docs.nvidia.com/nemo-framework/user-guide/25.02/modelalignment/dpo.html)
- NVIDIA 实战文档：[DPO with NeMo Customizer](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/finetuning/dpo_with_nemo_customizer.html)
- Hugging Face TRL 文档：[DPO Trainer](https://huggingface.co/docs/trl/main/dpo_trainer)
- 入门阅读顺序建议：
  1. 先看论文，理解“隐式奖励 + Bradley-Terry + KL 约束”三者关系。
  2. 再看 NeMo 或 TRL 文档，确认训练输入、reference model、beta 和指标监控方式。
  3. 最后跑一个最小 DPO 实验，先验证数据格式和 reward margin，再调 beta 与 batch 配置。
