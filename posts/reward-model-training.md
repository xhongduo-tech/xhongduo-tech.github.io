## 核心结论

奖励模型（Reward Model，RM）训练的目标，不是直接判断答案“是否客观正确”，而是学习“在人类做二选一偏好判断时，哪一个回答更可能被选中”。它把同一条 prompt 下两个候选回答之间的相对偏好，压缩为一个可微分的标量分数，再通过 Bradley-Terry 模型把“分数差”映射成“胜出概率”。

最核心的公式是：

$$
P(y_w \succ y_l \mid x)=\sigma\left(r_\theta(x,y_w)-r_\theta(x,y_l)\right)
$$

其中：

| 符号 | 含义 | 直白解释 |
|---|---|---|
| $x$ | prompt | 用户输入的问题或指令 |
| $y_w$ | winner | 在标注中胜出的回答 |
| $y_l$ | loser | 在标注中落败的回答 |
| $r_\theta(x,y)$ | 奖励分数 | 奖励模型给回答打的可比较分数 |
| $\sigma$ | sigmoid 函数 | 把任意实数压到 0 到 1 之间 |

直白解释：如果回答 A 比回答 B 高 1 分，那么模型认为人类选择 A 的概率约为：

$$
\sigma(1)=\frac{1}{1+e^{-1}}\approx 0.731
$$

这个 0.731 不是“答案有 73.1% 正确”，而是“在这一对候选里，人类更可能选 A”。

看一个玩具例子。问题是“Python 列表和元组有什么区别？”：

- 回答 A：说明“列表可变，元组不可变；列表适合频繁修改，元组适合固定结构”
- 回答 B：只说“它们都能存数据”

如果奖励模型给 A 打 2 分，给 B 打 1 分，那么：

$$
P(A \succ B)=\sigma(2-1)=\sigma(1)\approx 0.731
$$

也就是说，模型认为在这对回答里，A 更可能被人类偏好，但这仍然只是相对偏好，不是绝对质量认证。

工程上常见两条路线如下：

| 架构 | 白话解释 | 优点 | 缺点 |
|---|---|---|---|
| `value head` | 在现成 LLM 顶部接一个标量打分头 | 训练稳定、成本低、吞吐高 | 分数成因不透明，可解释性弱 |
| 生成式 reward judge | 让模型先写评审理由，再输出 verdict 或分数 | 可审计、适合多维标准 | 推理成本高、延迟大、实现复杂 |

奖励模型通常出现在 RLHF 的中间层：先有监督微调（SFT），再训练 RM，最后用 RM 去引导策略优化。问题在于，策略模型一旦开始直接优化 RM 分数，就会主动寻找奖励函数的漏洞，出现 reward hacking（奖励劫持），即“越来越会拿高分，但不一定真的更好”。

因此在强化学习目标里，通常会加入 KL 惩罚，限制新策略不要离参考策略太远：

$$
r'(x,y)=r_\theta(x,y)-\lambda \,\mathrm{KL}\!\left(\pi(\cdot \mid x)\|\pi_{\text{ref}}(\cdot \mid x)\right)
$$

其中：

- $r_\theta(x,y)$ 是 RM 给出的奖励
- $\pi$ 是当前正在优化的策略
- $\pi_{\text{ref}}$ 是参考模型，通常是 SFT 模型
- $\lambda$ 是惩罚强度

它的作用不是“让模型更聪明”，而是“别为了追分把输出分布改得过头”。

---

## 问题定义与边界

奖励模型解决的问题定义其实很窄：

给定一个 prompt $x$，以及两个候选回答 $y_a, y_b$，让人类标注者只回答一个问题：**更偏好哪一个**。

这意味着 RM 学习的是一个排序函数：

$$
r_\theta(x,y)
$$

它的目标是让“更可能被人类偏好的回答”拿到更高分，而不是给出一个覆盖事实性、安全性、完整性、风格等全部维度的终极真值分。

训练时常见的数据结构如下：

| 字段 | 含义 | 是否必需 | 典型风险 |
|---|---|---|---|
| `prompt` | 用户问题或指令 | 是 | 场景覆盖不足，分布太窄 |
| `candidate_a` | 候选回答 A | 是 | 候选质量过高或过低，区分度不足 |
| `candidate_b` | 候选回答 B | 是 | 与 A 太相似，标注难度过大 |
| `preference` | 选 A 还是 B | 是 | 单次标注噪声大 |
| `annotator_votes` | 多个标注员的投票结果 | 否，但强烈建议 | 一致性低时难判断可信度 |
| `rationale` | 标注理由 | 否 | 成本高，但有助于审计 |
| `IAA` | 标注一致性指标 | 否，但工程上重要 | 不记录则无法衡量标签可靠性 |

其中 IAA（Inter-Annotator Agreement）表示标注员之间的一致程度。它决定了偏好标签到底有多可信。偏好任务的一致性通常不会像“是否包含特定词”这种机械任务那样高，因为偏好常常同时受多个维度影响，例如：

- 准确性
- 安全性
- 简洁性
- 完整性
- 风格符合度
- 是否直接回答问题

同一个样本上，不同标注员可能依据不同标准做判断，因此偏好任务中出现 60% 到 75% 的一致性并不罕见。反过来说，也意味着 25% 到 40% 的分歧或噪声可能客观存在。

这直接带来两个边界：

1. 奖励模型学到的是“偏好排序”，不是“客观真值判定”。
2. 如果标注本身存在显著分歧，RM 也会把这些冲突标准一起学进去。

偏好学习的基本损失函数通常写成负对数似然：

$$
L(\theta)=-\log \sigma\left(r_\theta(x,y_w)-r_\theta(x,y_l)\right)
$$

也可以写成更数值稳定的形式：

$$
L(\theta)=\log\left(1+\exp\left(-(r_\theta(x,y_w)-r_\theta(x,y_l))\right)\right)
$$

直白理解：

- 如果模型给胜者的分数明显高于败者，损失就小
- 如果两者分不出高低，损失较大
- 如果败者反而分更高，损失会迅速变大

看一个更接近真实标注的例子。对于同一个回答对，5 个标注员里有 3 人选 A，2 人选 B。此时把它强行记成“绝对的 A 胜”，会高估这个样本的确定性。更稳妥的做法是至少保留投票分布：

$$
P(A)=0.6,\quad P(B)=0.4
$$

工程上常见三种处理方式：

| 处理方式 | 做法 | 优点 | 风险 |
|---|---|---|---|
| 硬标签 | 只保留多数票结果 | 简单、易训练 | 会丢失不确定性 |
| 软标签 | 用投票比例建模 | 保留样本不确定性 | 训练实现更复杂 |
| 过滤样本 | 只留高一致性样本 | 噪声更低 | 数据量会减少 |

真实工程里，客服问答、代码助手、安全审核都经常出现“没有绝对最优回答”的情况。一个回答可能更准确，另一个回答可能更简洁；一个更安全，另一个更完整。单一偏好标签只是把复杂多维质量压缩成一次二元选择，因此 RM 的分数不能被误读为“全维度总分”。

对新手来说，一个很重要的认知是：**奖励模型不是在学“世界事实”，而是在学“标注规范下的人类选择行为”**。这两者有关，但不相同。

---

## 核心机制与推导

Bradley-Terry 模型的核心假设是：两个候选回答谁更可能被选中，只取决于它们的分数差，而不是分数绝对值。也就是说，真正重要的是：

$$
\Delta r = r_\theta(x,y_w)-r_\theta(x,y_l)
$$

如果两个回答的分数同时都加 10 分，它们之间的相对偏好不应改变，因为分差没有变。

接下来，用 sigmoid 函数把分差映射成概率：

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

于是得到：

$$
P(y_w \succ y_l \mid x)=\frac{1}{1+e^{-(r_w-r_l)}}
$$

这里的逻辑可以拆成三步：

1. 先让模型输出一个可比较的分数 $r_\theta(x,y)$
2. 再取两个回答的分数差 $\Delta r$
3. 最后把分差转成“胜出的概率”

为什么这样设计合理？因为偏好任务本质上不是“预测一个类别”，而是“比较两个对象谁更优”。用分差建模，天然适合排序任务。

如果训练集包含很多偏好样本对：

$$
\{(x_i,y_i^w,y_i^l)\}_{i=1}^N
$$

那么最大似然估计对应的训练目标就是最小化整体负对数似然：

$$
\mathcal{L}_{\text{ERM}}(\theta)= - \sum_{i=1}^{N}\log \sigma\left(r_\theta(x_i,y_i^w)-r_\theta(x_i,y_i^l)\right)
$$

也可以写成平均损失：

$$
\mathcal{L}_{\text{ERM}}(\theta)= - \frac{1}{N}\sum_{i=1}^{N}\log \sigma\left(r_\theta(x_i,y_i^w)-r_\theta(x_i,y_i^l)\right)
$$

这里的 ERM（Empirical Risk Minimization）就是经验风险最小化，意思是在训练集上把平均损失压低。

这个目标为什么有效，可以从分差变化来直观看：

| 分数差 $\Delta r$ | $\sigma(\Delta r)$ | 含义 |
|---|---:|---|
| -3 | 0.047 | 模型强烈认为败者更好，方向明显错了 |
| -2 | 0.119 | 明显错边 |
| -1 | 0.269 | 倾向错边 |
| 0 | 0.500 | 完全分不出 |
| 1 | 0.731 | 有明显偏好 |
| 2 | 0.881 | 偏好很强 |
| 3 | 0.953 | 几乎确信胜者更优 |

用一个具体数值例子看更清楚。假设：

$$
r(y_1)=2,\quad r(y_2)=1
$$

那么：

$$
P(y_1 \succ y_2)=\sigma(1)\approx 0.731
$$

如果训练后变成：

$$
r(y_1)=4,\quad r(y_2)=1
$$

那么：

$$
P(y_1 \succ y_2)=\sigma(3)\approx 0.953
$$

这说明模型对“$y_1$ 更优”这件事更有把握。

不过，这种“把分差越拉越大”的趋势在监督训练阶段是合理的，到了策略优化阶段就可能变成问题。因为策略模型会反过来利用奖励模型，专门搜索能让 RM 打高分的输出模式，即 reward hacking。

这时就需要 KL 惩罚来约束策略：

$$
r'(x,y)=r_\theta(x,y)-\lambda \mathrm{KL}(\pi(\cdot|x)\|\pi_{\text{ref}}(\cdot|x))
$$

这里的 KL（Kullback-Leibler divergence）衡量两个分布之间的差异。它的直观意义是：

- 如果当前策略与参考策略很接近，惩罚较小
- 如果当前策略为了追 RM 分数而大幅偏离，惩罚会增大

需要注意的是，实际实现里经常使用 token 级近似 KL，而不是完整闭式公式。工程上更常见的写法是对生成序列逐 token 累加 log-prob 差值，再取期望。

常见符号说明如下：

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $\sigma$ | sigmoid 函数 | 把实数映射到 0 到 1 |
| $r_\theta$ | 奖励函数 | 给回答打可比较分数 |
| $\Delta r$ | 分数差 | 决定偏好概率的核心量 |
| ERM | 经验风险最小化 | 在训练样本上最小化平均损失 |
| $\pi$ | 当前策略 | 正在被优化的生成模型 |
| $\pi_{\text{ref}}$ | 参考策略 | 用来约束漂移的基线模型 |
| KL | KL 散度 | 衡量两个分布差多远 |

如果把 RM 训练和 RL 优化放在一起看，可以得到一个很实用的理解框架：

| 阶段 | 优化对象 | 主要风险 |
|---|---|---|
| RM 训练 | 让胜者分数高于败者 | 过拟合噪声偏好 |
| 策略优化 | 让策略生成高 RM 分回答 | 奖励劫持、分布漂移 |
| 加 KL 后 | 在追分和不偏离之间折中 | 仍无法修复错误奖励 |

因此，KL 的作用是“限制策略刷分的自由度”，不是“保证奖励模型正确”。

---

## 代码实现

最常见的实现方式，是在预训练语言模型顶部接一个 `value head`。语言模型负责把 `prompt + answer` 编码成隐藏状态，`value head` 再把整段回答压成一个标量分数。

先用一个不依赖深度学习框架的可运行示例，完整表达 Bradley-Terry 偏好建模：

```python
import math


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def bt_preference_prob(r_w: float, r_l: float) -> float:
    return sigmoid(r_w - r_l)


def pairwise_loss(r_w: float, r_l: float, eps: float = 1e-12) -> float:
    p = bt_preference_prob(r_w, r_l)
    p = min(max(p, eps), 1.0 - eps)
    return -math.log(p)


def demo() -> None:
    p = bt_preference_prob(2.0, 1.0)
    assert round(p, 3) == 0.731

    loss_good = pairwise_loss(2.0, 1.0)
    loss_bad = pairwise_loss(1.0, 2.0)
    assert loss_good < loss_bad

    assert bt_preference_prob(3.0, 1.0) > bt_preference_prob(2.0, 1.0)

    examples = [
        ("winner far better", 4.0, 1.0),
        ("winner slightly better", 2.0, 1.5),
        ("tie", 1.0, 1.0),
        ("winner scored lower", 1.0, 2.0),
    ]

    for name, rw, rl in examples:
        p = bt_preference_prob(rw, rl)
        loss = pairwise_loss(rw, rl)
        print(f"{name:22s} prob={p:.4f} loss={loss:.4f}")


if __name__ == "__main__":
    demo()
```

这段代码可以直接运行。输出会体现三个规律：

1. 胜者分数越高，胜出概率越大
2. 胜者分数越高于败者，损失越小
3. 如果败者被打得更高，损失会明显变大

再看一个可运行的 PyTorch 版本。它模拟“从隐藏状态抽取整段回答表示，再映射成 reward”的流程：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states: [batch, seq_len, hidden_size]
        # attention_mask: [batch, seq_len], valid token 为 1，padding 为 0
        last_index = attention_mask.sum(dim=1) - 1
        batch_index = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled = hidden_states[batch_index, last_index]
        reward = self.value_head(pooled).squeeze(-1)
        return reward


def pairwise_rm_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    diff = r_chosen - r_rejected
    return -F.logsigmoid(diff).mean()


def demo() -> None:
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 5
    hidden_size = 8

    model = RewardHead(hidden_size)

    chosen_hidden = torch.randn(batch_size, seq_len, hidden_size)
    rejected_hidden = torch.randn(batch_size, seq_len, hidden_size)

    chosen_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    rejected_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])

    r_chosen = model(chosen_hidden, chosen_mask)
    r_rejected = model(rejected_hidden, rejected_mask)

    loss = pairwise_rm_loss(r_chosen, r_rejected)
    loss.backward()

    print("chosen rewards:", r_chosen.detach())
    print("rejected rewards:", r_rejected.detach())
    print("loss:", float(loss.detach()))


if __name__ == "__main__":
    demo()
```

这段代码虽然没有接完整 LLM，但关键结构已经齐全：

- 输入是一段序列的隐藏状态
- 通过 `attention_mask` 找到最后一个有效 token
- 抽取该位置的向量作为整段回答的表示
- 线性层输出一个标量 reward
- 用 pairwise loss 训练“chosen 分数高于 rejected”

为什么很多实现会取“最后一个有效 token”而不是平均池化？原因主要有三点：

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| 最后一个有效 token | 取回答末尾对应隐藏状态 | 实现简单，与自回归 LM 相容 | 可能过度依赖末尾表示 |
| 平均池化 | 对所有 token 表示取平均 | 信息更平滑 | 可能稀释关键信号 |
| 特殊分类 token | 用专门位置承载全局信息 | 结构清晰 | 需模型结构配合 |

真实工程里，如果你训练一个代码助手 RM，输入可能是：

- `prompt`：“写一个二分查找函数，并说明边界条件”
- `candidate A`：代码正确，解释了 `left <= right`、中点取法和未找到返回值
- `candidate B`：代码能运行，但边界写错，在单元素数组时可能漏检

训练时不会要求 RM 解释“为什么 B 有 bug”，它只需要在大量样本上学到：像 A 这样的回答应该比 B 分数更高。

另一条路线是生成式 reward judge。它不是直接输出一个标量，而是先生成审查理由，再输出 verdict 或分数。例如：

| 维度 | value head | 生成式 judge |
|---|---|---|
| 输入 | prompt + 回答 | prompt + 回答 + 评分标准 |
| 输出 | 单个标量 | 文字理由 + verdict/分数 |
| 可解释性 | 弱 | 强 |
| 训练成本 | 低 | 高 |
| 推理速度 | 快 | 慢 |
| 适合场景 | 大规模排序优化 | 安全、事实、多维审计 |

judge 的优势是可以显式说明“为什么选 A 不选 B”。例如在安全审核里，它可以分维度输出：

- 是否包含危险建议
- 是否拒绝得当
- 是否有事实幻觉
- 是否风格符合规范

这比单标量更适合审计，但代价是吞吐和一致性控制更困难。

对新手来说，一个最重要的区分是：

- `value head` 更像“训练时高效打分器”
- 生成式 judge 更像“可解释评审器”

前者偏工程效率，后者偏审计能力。

---

## 工程权衡与常见坑

第一个常见坑是标注噪声。

偏好数据不是客观真值，而是“在特定规则下，人类更倾向选哪个”。如果标注定义模糊，或者标注员关注点不同，RM 最终学到的就不是稳定规律，而是混杂的偏好噪声。

例如在医学问答场景中：

- 标注员 A 更重视安全保守
- 标注员 B 更重视解释完整
- 标注员 C 更重视语言简洁

结果同一个样本可能得到 2:1 的多数票，但这不代表存在稳定共识，只代表在这三个人里，某种标准略占上风。

因此工程上需要先做数据质检，再谈模型训练。常用处理手段如下：

| 手段 | 作用 | 适用场景 |
|---|---|---|
| 多人投票筛样 | 降低单个标注员偶然偏差 | 标注预算较充足 |
| 胜率阈值过滤 | 去掉低置信度样本 | 偏好冲突明显时 |
| Krippendorff’s $\alpha$ | 衡量一致性是否足够稳定 | 数据集验收 |
| 软标签训练 | 保留投票比例而非硬标签 | 希望保留不确定性 |
| 分桶评估 | 按任务类型分别看一致性 | 多场景混合数据 |
| 主动复标 | 对争议样本进行二次确认 | 高价值数据集 |

Krippendorff’s $\alpha$ 可以粗略理解为“大家是不是在按相似标准打标”。如果它很低，继续堆更大的 RM 通常没有根本意义，因为问题首先出在任务定义和标注规范。

第二个坑是 reward hacking。

RM 优化的是“与标注偏好相关的统计模式”，策略模型一旦开始追逐这个分数，就会主动寻找高分模板，而不一定提升真实质量。典型表现包括：

- 结构看起来非常完整，但遗漏关键事实
- 拒答模板很安全，但对可回答问题也过度保守
- 语言显得礼貌专业，但内容空洞
- 格式非常工整，但真实帮助不大

例如总结模型可能总是输出：

1. 背景  
2. 要点  
3. 结论

这种格式很容易在偏好数据中占优，因为它清晰、规整、像“高质量答案”。但如果模型开始机械套模板，就可能牺牲内容密度，形成“形式优先”的高分假象。

第三个坑是 OOD（Out-of-Distribution，分布外）过优化。

训练时 RM 看到的数据分布有限，策略模型在优化过程中却会生成越来越“奇怪但高分”的回答。这些回答可能在语言表面上很像优质答案，但在隐藏空间或实际任务表现上已经偏离训练流形。

因此工程里常见的缓解方法不只一种：

| 方法 | 主要作用 | 本质 |
|---|---|---|
| KL penalty | 限制策略偏离参考模型 | 控制分布漂移 |
| Occupancy penalty | 惩罚进入低覆盖区域 | 避免过度探索陌生区域 |
| Reward clipping | 限制极端奖励值 | 防止训练不稳定 |
| Ensemble RM | 用多个 RM 看分歧 | 粗略估计不确定性 |
| Gradient regularization | 降低尖锐最优点 | 减少过优化敏感性 |
| Online human check | 对高分样本做人审抽检 | 验证是否出现骗分 |

一个简化的总目标可以写成：

$$
L_{\text{total}} = L_{\text{policy}} - \mathbb{E}[r_\theta(x,y)] + \lambda_{\text{KL}}\mathrm{KL}(\pi\|\pi_{\text{ref}}) + \lambda_{\text{occ}} L_{\text{occ}}
$$

其中：

| 项 | 含义 |
|---|---|
| $L_{\text{policy}}$ | 原始策略优化损失 |
| $-\mathbb{E}[r_\theta(x,y)]$ | 鼓励生成高奖励回答 |
| $\lambda_{\text{KL}}\mathrm{KL}$ | 约束不要偏离参考策略过远 |
| $\lambda_{\text{occ}}L_{\text{occ}}$ | 惩罚进入训练覆盖不足区域 |

对应的简化伪代码如下：

```python
def total_loss(
    policy_loss: float,
    reward_score: float,
    kl_value: float,
    occ_penalty: float,
    kl_coef: float = 0.02,
    occ_coef: float = 0.01,
) -> float:
    return (
        policy_loss
        - reward_score
        + kl_coef * kl_value
        + occ_coef * occ_penalty
    )


def demo() -> None:
    loss = total_loss(
        policy_loss=0.8,
        reward_score=1.2,
        kl_value=3.5,
        occ_penalty=0.7,
    )
    print(round(loss, 4))


if __name__ == "__main__":
    demo()
```

这段代码同样可以直接运行。它不是完整 RL 算法，只是把“奖励项”和“正则项”之间的关系表达清楚。

第四个坑是把 RM 当成绝对裁判。

RM 更接近“训练阶段的排序器”，而不是“部署阶段的最终真理源”。特别是在安全、医疗、法律、金融等高风险任务里，RM 分数只能作为一个信号，不能替代：

- 规则系统
- 检索校验
- 工具执行验证
- 人工抽检
- 专项安全模型

也就是说，RM 可以帮助模型“更像人类偏好”，但不能自动保证“符合外部事实和制度要求”。

---

## 替代方案与适用边界

`value head` 和生成式 judge 都能做奖励建模，但它们适合的问题边界不同。

先看整体对比：

| 方案 | 适合什么 | 不适合什么 |
|---|---|---|
| value head | 大规模训练、低延迟打分、批量重排序 | 需要详细解释和审计的场景 |
| 生成式 judge | 多维反馈、安全审核、事实性评审 | 高吞吐在线排序 |
| 规则奖励 + RM 混合 | 存在硬约束的任务，如格式、长度、敏感词 | 完全开放式主观质量判断 |
| 直接偏好优化 | 想减少单独训练 RM 的环节 | 需要独立可复用奖励器的场景 |

可以把它们理解为三类不同工具：

- `value head`：高效标量评分器
- 生成式 judge：可解释评审器
- 规则奖励：硬约束过滤器

在很多真实系统里，这三者并不是互斥关系，而是组合使用。例如：

| 组件 | 负责内容 |
|---|---|
| 规则系统 | 敏感词、格式、长度、工具调用约束 |
| RM / judge | 偏好质量、风格符合度、回答帮助性 |
| 人工抽检 | 审查模型是否学会骗分 |

如果偏好数据较少，通常不会从零开始训练一个复杂 judge，而是先从已有 SFT 模型初始化，再接一个 `value head` 做奖励建模。原因很简单：

- SFT 模型已经学会基本语言分布
- RM 只需要学习“哪些回答更受偏好”
- 这样样本效率更高，训练也更稳定

在 PPO 或其他 policy-gradient 方法里，常见奖励修正仍然是：

$$
r' = r_\theta(x,y) - \lambda \mathrm{KL}(\pi(\cdot|x)\|\pi_{\text{ref}}(\cdot|x))
$$

它的意义很明确：

- 回答质量高，RM 可以给正向信号
- 偏离参考分布太远，KL 会施加反向约束

但必须强调一个边界：**KL 只能限制“别偏太远”，不能保证“偏得正确”**。如果 RM 本身学偏了，KL 只会让策略在“参考模型”和“有偏奖励”之间做折中，而不会自动修复错误偏好。

如果任务需要同时考虑多个维度，例如：

| 维度 | 例子 |
|---|---|
| 安全性 | 是否包含危险建议或违规引导 |
| 事实性 | 是否与检索证据一致 |
| 风格 | 是否符合品牌语气或角色设定 |
| 完整性 | 是否覆盖关键步骤与限制条件 |
| 可执行性 | 给出的代码、命令、步骤是否可运行 |

那么生成式 judge 往往更有价值，因为它可以逐维评审，再给综合 verdict。例如它可以输出：

- 安全性通过
- 事实性存在一处证据缺失
- 风格符合要求
- 完整性不足，遗漏异常分支

这类输出对于调试奖励系统很有帮助，而单一标量难以提供这类诊断信息。

另一方面，如果任务只是大规模候选重排序，例如每个 prompt 生成 8 个候选回答，从中选前 2 个送给用户，那么 `value head` 往往更现实，因为它便宜、快、易于部署。

因此在方案选择上，可以用下面这张表快速判断：

| 条件 | 更偏向的方案 |
|---|---|
| 追求吞吐和训练效率 | value head |
| 追求可解释和可审计 | 生成式 judge |
| 存在明确硬约束 | 规则奖励混合 |
| 想减少独立 RM 训练环节 | 直接偏好优化 |

最终边界很清楚：奖励模型适合做“偏好信号压缩器”，不适合单独承担“最终质量裁决者”的角色。

---

## 参考资料

| 来源 | 内容概述 | 适用章节 |
|---|---|---|
| Christiano et al., 2017, *Deep Reinforcement Learning from Human Preferences* | 人类偏好监督、比较式反馈、奖励学习的早期代表工作 | 问题定义、核心机制 |
| Ouyang et al., 2022, *Training language models to follow instructions with human feedback* | InstructGPT 中 RM、PPO、KL 约束的经典工程链路 | 核心结论、代码实现、工程权衡 |
| Anthropic, 2022, *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback* | 偏好建模与有害性/帮助性对齐实践 | 工程权衡、替代方案 |
| Bradley and Terry, 1952, *Rank Analysis of Incomplete Block Designs* | Bradley-Terry 比较模型的原始统计基础 | 核心机制与推导 |
| Krippendorff, 2004, *Content Analysis: An Introduction to Its Methodology* | Krippendorff’s $\alpha$ 的定义与适用条件 | 问题定义、工程权衡 |
| Rafailov et al., 2023, *Direct Preference Optimization* | 不显式训练独立 RM 的替代路线 | 替代方案与适用边界 |
| RM 过优化与 reward hacking 相关实践论文与技术报告 | 讨论策略绕过奖励、分布漂移、正则化缓解 | 工程权衡与常见坑 |

这些资料之所以值得参考，是因为它们分别覆盖了三个层面：

- 统计建模层：解释 Bradley-Terry、pairwise loss 与比较概率
- 系统实现层：解释 RM、SFT、PPO、KL 约束如何接起来
- 工程风险层：解释噪声偏好、过优化、分布漂移和缓解方法

如果只想抓住可复现、最稳固的部分，优先看四类内容就够了：

1. Bradley-Terry 概率建模
2. pairwise loss 的训练目标
3. `value head` 奖励建模结构
4. KL 正则与标注一致性控制

而像 latent outlier 检测、occupancy penalty、复杂正则化这类方法，通常更依赖具体训练栈和评测设计，适合在已有基础系统后再引入，而不是一开始就上。
