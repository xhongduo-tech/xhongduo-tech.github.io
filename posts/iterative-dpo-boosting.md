## 核心结论

Iterative DPO，可以直译为“迭代式直接偏好优化”，本质上不是一种全新的损失函数，而是一种**训练流程**：每一轮都用当前模型生成新候选答案，给这些答案做偏好标注，再继续用 DPO 训练。这样做的作用，是把一次性静态偏好学习，变成持续补数据、持续纠偏的闭环。

它和普通 DPO 的关键区别不在“怎么求 loss”，而在“数据是不是跟着模型一起更新”。普通 DPO 往往只吃一批固定偏好数据；Iterative DPO 会让模型在第 $t$ 轮暴露出新的弱点，再把这些弱点转成第 $t+1$ 轮的训练样本，所以效果常常更强，尤其是在复杂指令、推理题、安全拒答这类“初始模型会不断犯新错”的任务上。

把它类比成 boosting 是可以的，但要加边界：boosting 是显式组合多个弱学习器，Iterative DPO **不是做模型集成**，而是让同一个策略在多轮“采样-标注-再训练”里不断吸收更难、更贴近当前错误分布的数据。更准确的说法是：它像一种面向偏好数据的在线 hard example mining，白话讲就是“模型哪里还做不好，就继续围着哪里刷题”。

| 对比项 | 一次性 DPO | Iterative DPO |
|---|---|---|
| 训练步骤 | 一次训练 | 多轮循环 |
| 数据来源 | 固定离线偏好集 | 当前模型持续生成的新偏好对 |
| 模型状态 | 基本只更新参数 | 参数更新后还会影响下一轮数据分布 |
| 适合问题 | 数据已较完整、预算有限 | 错误模式会随模型变强而变化的任务 |

---

## 问题定义与边界

先定义任务。偏好优化中的“偏好数据”，白话讲就是“同一个问题下，两份回答里哪份更好”。形式通常是三元组 $(x, y_w, y_l)$：$x$ 是 prompt，$y_w$ 是胜者回答，$y_l$ 是败者回答。目标不是让模型记住某个标准答案，而是让它**提高胜者相对败者的条件概率**。

DPO 的边界也要说清楚。它解决的是“相对偏好排序”问题，不直接保证事实正确、推理严密或绝对安全。Iterative DPO 只是把这个排序训练做成多轮迭代，因此它擅长的是：随着模型能力上升，不断补上新的偏好缺口；它不擅长的是：在没有高质量评审、奖励模型或人工标注的情况下，自动产生可靠监督。

玩具例子很简单。给定 prompt：“解释梯度下降。”  
当前模型可能采样出两版答案：

- $y_w$：定义准确，说明“沿损失函数下降方向更新参数”
- $y_l$：只说“不断试错直到变好”，过于简化

标注者选择前者为胜者。一次 DPO 会把这对样本用掉就结束；Iterative DPO 会在模型更新后，继续让**新模型**回答同类问题。此时新模型可能已经不会犯“过于简化”这种低级错误了，但会暴露新问题，比如公式对了、直觉不清，或者解释太长。于是下一轮数据会更难，也更贴近当前瓶颈。

数学上，一轮 DPO 的目标常写成：

$$
\mathcal L_{\text{DPO}}(\theta)
=
-\mathbb E_{(x,y_w,y_l)\sim D}
\left[
\log \sigma
\left(
\beta
\left(
\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}
\right)
\right)
\right]
$$

其中 $\pi_\theta$ 是当前策略，白话讲就是“现在这个模型怎么回答”；$\pi_{\mathrm{ref}}$ 是参考策略，白话讲就是“拿来做约束的旧模型”；$\beta$ 控制偏好拉开的力度。

Iterative DPO 只是把这个目标放进循环：

$$
D_{t+1} = D_t \cup \text{Label}\big(\text{Sample}(\pi_{\theta_t}, X_t)\big), \qquad
\theta_{t+1} = \arg\min_\theta \mathcal L_{\text{DPO}}(\theta; D_{t+1})
$$

也就是：采样 $\to$ 标注 $\to$ 再训练 $\to$ 更新 $\theta$。

---

## 核心机制与推导

DPO 的关键量其实是“胜者和败者的对数概率差”。设

$$
\Delta_\theta(x) = \log \pi_\theta(y_w|x) - \log \pi_\theta(y_l|x)
$$

参考模型也有同样的差值 $\Delta_{\mathrm{ref}}(x)$。那么 DPO 优化的是：

$$
-\log \sigma\big(\beta(\Delta_\theta(x)-\Delta_{\mathrm{ref}}(x))\big)
$$

这里的 sigmoid，白话讲就是“把任意实数压到 0 到 1 的平滑函数”。当 $\Delta_\theta - \Delta_{\mathrm{ref}}$ 越大，说明当前模型比参考模型更偏向胜者，loss 越小；反过来，loss 会推动模型继续提高胜者概率、降低败者概率。

研究摘要里的数值例子正好可以说明梯度方向。若当前模型的 logit 差是 $0.5$，参考模型差是 $0.2$，取 $\beta=5$，则输入为：

$$
z = 5\times(0.5-0.2)=1.5
$$

此时损失约为：

$$
-\log \sigma(1.5)\approx 0.20
$$

这不是 0，说明还有优化空间。梯度会继续把 $z$ 往更大方向推，也就是继续增加“胜者相对败者”的优势。

Iterative DPO 的增益来自两个层面。

第一，**数据分布跟着模型走**。如果只做一次 DPO，模型只会学会一批固定错误的修正；而迭代时，第 $t$ 轮更新后的模型会生成不同分布的候选，新的偏好对往往更难、更贴近当前能力边界。

第二，**隐式 KL 约束仍然存在**。DPO 的推导来自受 KL 约束的奖励最大化，白话讲就是“既要更符合偏好，也不要离参考模型飘太远”。所以它不像纯策略梯度那样显式跑 RL 回合，但依然保留了“别漂移过猛”的结构。需要注意的是，这个 KL 约束不是一个单独写出来的罚项，而是通过参考策略比值嵌进了目标函数。

这里也要纠正一个常见误解：Iterative DPO 不意味着每轮 margin 都会无上限累加。真实工程里，新增样本的质量、采样温度、参考模型是否固定、长度偏置是否受控，都会影响后续轮次是否继续收益。如果新数据开始变噪，迭代会把噪声也强化。

真实工程例子是 Liu 等人的 Iterative Length-Regularized DPO。它的问题很典型：模型在迭代偏好训练后，回答质量提升了，但也更容易变长，于是“看起来更强”部分其实混入了“更啰嗦”带来的评测偏置。论文的做法是在迭代 DPO 中加入长度正则，控制 verbosity，最终让 7B 模型在 AlpacaEval 2.0 上取得了对 GPT-4 Preview 的 50.5% 长度控制胜率。这个结果说明：迭代有效，但必须防止模型把“多写一点”学成“更优回答”的捷径。

---

## 代码实现

下面给一个可运行的玩具实现。它不训练神经网络，只模拟 DPO 的核心计算和多轮更新逻辑，目的是看清“当前模型采样、偏好标注、再训练”的闭环。

```python
import math

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def dpo_loss(samples, beta=5.0):
    """
    samples: list of (model_gap, ref_gap)
    model_gap = log pi(y_w|x) - log pi(y_l|x)
    ref_gap   = log pi_ref(y_w|x) - log pi_ref(y_l|x)
    """
    losses = []
    for model_gap, ref_gap in samples:
        z = beta * (model_gap - ref_gap)
        losses.append(-math.log(sigmoid(z)))
    return sum(losses) / len(losses)

def one_step_update(samples, lr=0.1):
    """
    用一个极简规则模拟“提高胜者相对败者的 gap”
    """
    updated = []
    for model_gap, ref_gap in samples:
        grad_signal = sigmoid(5.0 * (model_gap - ref_gap)) - 1.0
        new_gap = model_gap - lr * grad_signal
        updated.append((new_gap, ref_gap))
    return updated

# 第 0 轮：模型对胜者只有轻微偏好
round0 = [(0.5, 0.2), (0.1, 0.0), (0.3, 0.1)]
loss0 = dpo_loss(round0)

# 第 1 轮：在同一批偏好样本上更新
round1 = one_step_update(round0, lr=0.3)
loss1 = dpo_loss(round1)

# 第 2 轮：模拟用更新后模型重新采样到更难样本
round2 = round1 + [(0.05, 0.0), (0.12, 0.1)]
loss2 = dpo_loss(round2)

assert loss1 < loss0
assert loss2 > 0
assert round1[0][0] > round0[0][0]

print("round0 loss =", round(loss0, 4))
print("round1 loss =", round(loss1, 4))
print("round2 loss =", round(loss2, 4))
```

上面代码里：

- `model_gap` 表示当前模型更偏向胜者多少
- `ref_gap` 表示参考模型原本的偏好差
- `dpo_loss` 对应 DPO 公式
- `one_step_update` 用简化规则模拟梯度更新
- 第 2 轮多加了更难样本，表示 Iterative DPO 不只是在老题上刷分，而是在新模型暴露出的新边界上继续采样

工程里更接近下面这段伪代码：

```python
for round_id in range(R):
    samples = sample_with_current_model(model, prompts)
    filtered = filter_by_length_and_quality(samples)
    prefs = label_preferences(filtered, human_or_reward_model)
    loss = compute_dpo_loss(model, ref_model, prefs, beta=5.0)
    model.step(loss)
    log_round_metadata(round_id, source="model@t", date="2026-04-14")
```

每一步的作用分别是：

- 采样：必须用最新模型，否则就失去“迭代”意义
- 过滤：先清理极长、重复、明显低质量样本
- 标注：人工或奖励模型给出胜负
- 训练：按 DPO 目标拉开胜负差距
- 记录：保留轮次、来源、时间和 prompt 分布，防止后面查不清数据污染

---

## 工程权衡与常见坑

Iterative DPO 的真实难点不在公式，而在数据闭环质量。如果采样和标注管线不稳定，模型会越来越自信地学习错误偏好。

| 常见问题 | 为什么会发生 | 缓解策略 |
|---|---|---|
| 噪声偏好累积 | 新模型采样质量差，错误答案也可能被误标为胜者 | 提高评审质量，加入 reward model 或人工抽检 |
| 越训越长 | 更长回答在主观评测里常显得“更认真” | 加长度正则、长度桶采样、长度控制评测 |
| 多样性下降 | 每轮都从相近 prompt 上采样，分布越来越窄 | 每轮加入新 prompt、分层采样不同领域 |
| 过拟合评审器 | 奖励模型有固定偏见，模型学会“讨好评委” | 混合人工标注、对抗样本、不同评审源交叉验证 |
| 参考漂移失控 | 若参考策略更新过快，KL 约束会变弱或不稳定 | 常用固定 SFT 参考，或使用慢速更新参考 |

一个很典型的新手坑是“越长越好”陷阱。比如模型生成的胜者总是 800 字，败者总是 200 字。你以为模型学到了更完整的解释，实际上它可能只学到了“多写就更容易赢”。这也是长度正则在迭代偏好训练里经常出现的原因。

另一个坑是把 Iterative DPO 当成“永远多轮就更好”。不是。轮数增加意味着更多计算、更多标注成本、更多数据治理复杂度。通常只有当你已经看到模型在新一轮采样中持续暴露“可被偏好监督修复”的错误时，继续迭代才有意义。

---

## 替代方案与适用边界

如果预算有限，普通 DPO 往往是第一选择。它简单、稳定、不需要在线滚动采样，适合已经有一批质量不错的偏好数据集时直接训练。

如果任务需要复杂探索，尤其是长时序决策、工具使用、多轮交互，PPO 这类 RLHF 方法仍然有位置。PPO，白话讲就是“通过策略梯度直接优化长期回报的强化学习方法”，它更灵活，但工程复杂度和训练不稳定性都更高。

| 方法 | 数据量需求 | 工程复杂度 | 适合场景 |
|---|---|---|---|
| 一次性 DPO | 中 | 低 | 已有较成熟偏好数据，想快速对齐 |
| Iterative DPO | 中到高 | 中到高 | 模型错误会随能力提升而迁移，需要持续补样本 |
| PPO / 经典 RLHF | 高 | 高 | 需要更强探索能力或显式长期回报优化 |

再说一个边界。Iterative DPO 适合“已有基础模型、已有采样与标注流水线、还能承受多轮训练”的团队。它不太适合以下情况：

- 只有很小的标注预算
- 没有可靠评审器或人工审核
- 任务本身没有明确成对偏好结构
- 线上分布变化太快，来不及做稳定的数据治理

所以，新手版可以这样记：一次 DPO 像只上一个训练阶段；Iterative DPO 像每轮都先考试、找错题、再补课。它强在闭环，不强在名字。

---

## 参考资料

- Rafailov 等，《Direct Preference Optimization: Your Language Model is Secretly a Reward Model》([arXiv 2305.18290](https://huggingface.co/papers/2305.18290))。DPO 的基础论文，核心价值是把“带 KL 约束的偏好优化”改写成直接可训练的分类式目标。
- Liu 等，《Iterative Length-Regularized Direct Preference Optimization: A Case Study on Improving 7B Language Models to GPT-4 Level》([arXiv 2406.11817](https://huggingface.co/papers/2406.11817))。本文最相关的工程案例，重点说明迭代 DPO 的收益与“长度膨胀”问题，以及长度正则的必要性。
- Pang 等，《Iterative Reasoning Preference Optimization》([NeurIPS 2024](https://mlanthology.org/neurips/2024/pang2024neurips-iterative/))。聚焦推理任务，说明普通迭代偏好优化对 reasoning 未必天然有效，并展示了带额外 NLL 项的改法。
- Emergent Mind 的 “Iterative DPO” 主题综述([Emergent Mind](https://www.emergentmind.com/topics/iterative-direct-preference-optimization-dpo))。适合快速浏览概念脉络、常见变体与应用方向，但应以论文原文为准。
