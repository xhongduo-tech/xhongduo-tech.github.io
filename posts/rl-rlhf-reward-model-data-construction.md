## 核心结论

奖励模型数据构建的核心，不是给每条回答打一个“绝对分”，而是在**同一条 prompt 下**收集“哪个回答更好”的偏好对。这里的“偏好对”可以理解为：同一个问题，给人看两个候选回答，只要求选出更优的那个。奖励模型学习的是排序关系，而不是分数刻度本身。

形式上，常见训练目标是：

$$
\mathcal{L} = -\mathbb{E}_{(x,y^+,y^-)}\left[\log \sigma\left(r(x,y^+) - r(x,y^-)\right)\right]
$$

其中，$x$ 是 prompt，$y^+$ 是优选回答，$y^-$ 是劣选回答，$r(x,y)$ 是奖励模型输出，$\sigma$ 是 sigmoid 函数，白话解释就是“把分差映射成偏好概率”。

这带来三个直接结论：

1. 奖励模型首先要学会“谁比谁更好”，而不是“这条回答值 7.3 分”。
2. 数据质量通常比数据规模更重要，尤其是标注一致性、难例覆盖和候选多样性。
3. 如果候选几乎都来自同一种解码策略，模型很容易学到表面风格，比如更长、更像模板、更像安全声明，而不是真实偏好。

一个新手可理解的玩具例子是：prompt 为“解释什么是缓存穿透”。给标注员两个回答，A 定义准确、结构清楚，B 只有口语化描述但没有定义。标注员选择 A。奖励模型不需要知道 A 是 9 分还是 95 分，只需要学到“在这个 prompt 下，A 应该高于 B”。

---

## 问题定义与边界

奖励模型数据构建，解决的是这样一个问题：给定 prompt $x$，生成多个候选回答，再让人类从中比较出更优和更差的回答，最终得到 $(x,y^+,y^-)$ 形式的数据。

这个问题的边界很重要，常见误区都出在边界没有定义清楚。

第一，比较必须发生在**同一 prompt 内部**。不能把“解释 TCP 三次握手”和“写一个 Python 排序函数”的回答拿来比较，因为任务目标不同，偏好没有可比性。

第二，标签本质上是**相对偏好**，不是自动分数。即便后续会记录 confidence、difficulty 等附加字段，训练主信号仍然是“chosen vs rejected”。

第三，候选来源会决定数据的上限。如果所有候选都来自同一个模型、同一温度、同一种解码方式，那么样本之间差异很可能过于单一，最后训练出的奖励模型只会识别某种固定写法。

第四，数据不是越多越好。若标注标准不稳、题目过易、难例缺失、偏见未监控，扩容只会放大噪声。

一个实用的数据结构可以写成下面这样：

| Prompt ID | Prompt | Chosen Response | Rejected Response | Annotator ID | Confidence | Difficulty Flag |
|---|---|---|---|---|---|---|
| p_1024 | 解释 RM 训练原理 | 定义准确，给出 BT loss 与训练目标 | 只说“让模型更好”但无机制 | ann_07 | 0.92 | hard |
| p_1025 | 写 SQL 去重查询 | 使用 `ROW_NUMBER()` 并解释窗口函数 | 只给 `DISTINCT`，未满足要求 | ann_03 | 0.88 | medium |

这里的 `Difficulty Flag` 可以理解为“样本难度标签”，白话说就是这条样本是不是容易判断、是不是能区分强弱模型。难例通常比简单样本更有价值，因为简单样本只会让模型学到非常粗的排序。

一个真实工程例子是客服助手训练。prompt 是“用户要求退款但订单已超过 30 天，如何回复”。候选 A 严格按政策说明并给替代方案，候选 B 语气礼貌但直接承诺退款。标注员可能会选择 A，因为它更符合业务规则。这里奖励模型学到的不是“礼貌句式更好”，而是“在业务约束下，合规且有可执行方案的回答更优”。

---

## 核心机制与推导

奖励模型训练常用 Bradley-Terry 框架。它假设在同一个 prompt 下，回答 $y^+$ 被偏好于 $y^-$ 的概率为：

$$
P(y^+ \succ y^- \mid x) = \sigma\left(r(x,y^+) - r(x,y^-)\right)
$$

这里的“Bradley-Terry”可以理解为一种“用分差解释比较结果”的排序模型。重点是分差，不是绝对值。

如果 $r(x,y^+) - r(x,y^-)$ 很大，说明模型非常确信优选回答更好；如果差值接近 0，说明模型分不出；如果差值为负，说明模型排反了。

因此损失函数写成：

$$
\mathcal{L} = -\log \sigma\left(r(x,y^+) - r(x,y^-)\right)
$$

它只依赖于差值 $\Delta r = r^+ - r^-$。这意味着下面两组分数在训练上几乎等价：

- $r^+=2,\ r^-=1$
- $r^+=102,\ r^-=101$

因为两者差值都为 $1$。这就是为什么奖励模型的“分数范围”往往不重要，真正重要的是排序稳定性。

看一个最小数值例子。若：

$$
r^+=0.22,\quad r^-=-1.24
$$

则：

$$
\Delta r = 0.22 - (-1.24) = 1.46
$$

对应偏好概率：

$$
\sigma(1.46)\approx 0.81
$$

意思是，模型认为“优选回答胜出”的概率约为 81%。这已经足够作为训练信号，不需要再把它映射成一个绝对评分体系。

但工程上只靠这个公式还不够，因为尺度会漂移。所谓“尺度漂移”，白话说就是模型可能不断把分数整体放大或缩小，虽然排序没变，但会影响后续 RL 阶段的稳定性。常见处理方式有三类：

| 方法 | 作用 | 适用场景 |
|---|---|---|
| Reward Centering | 让奖励均值更稳定 | 训练后接 PPO/GRPO |
| Temperature Scaling | 调整分差敏感度 | 偏好概率过陡或过平 |
| Margin-aware Loss | 对“强偏好”样本拉开更大间隔 | 标注中包含强弱程度 |

例如带温度的写法可以表示为：

$$
P(y^+ \succ y^- \mid x)=\sigma\left(\frac{r^+ - r^-}{T}\right)
$$

其中 $T$ 是温度。温度可以理解为“分差放大器”或“压缩器”。$T$ 小时更敏感，$T$ 大时更平缓。

再看一个玩具例子。prompt 是“写一个二分查找解释”。两个候选都没有事实错误，但 A 给了边界条件，B 省略了 `left <= right` 的原因。对人类来说 A 稍好，但不是压倒性胜利。这类样本的价值很高，因为它逼奖励模型学习细粒度判断，而不是只分辨“正确 vs 完全错误”。

真实工程里，难点在于“什么叫更好”并不唯一。比如代码助手场景中，“更好”通常同时包含正确性、可运行性、解释清晰度、安全性、是否遵守用户约束。奖励模型实际上是在压缩一个多目标偏好空间。数据构建时如果不明确 rubric，最终模型就会学到一堆混杂信号。

---

## 代码实现

训练数据通常组织为 `prompt`、`chosen`、`rejected` 三列。实现上，本质就是对两条回答分别打分，再对分差做 BT loss。

下面给出一个可运行的最小 Python 版本。它不是深度学习框架代码，而是一个“从数据结构到损失计算”的玩具实现，便于理解训练信号到底是什么。

```python
import math

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def bt_loss(r_plus: float, r_minus: float) -> float:
    # 为数值稳定性做简单截断
    diff = max(min(r_plus - r_minus, 30.0), -30.0)
    return -math.log(sigmoid(diff))

def score_answer(prompt: str, answer: str) -> float:
    # 玩具打分器：真实工程里这里是 Transformer 奖励模型
    score = 0.0
    if "定义" in answer:
        score += 1.0
    if "公式" in answer:
        score += 1.0
    if "例子" in answer:
        score += 0.8
    if len(answer) > 20:
        score += 0.3
    if "胡说" in answer:
        score -= 2.0
    return score

dataset = [
    {
        "prompt": "解释奖励模型训练原理",
        "chosen": "先给出定义，再写公式，再给一个例子。",
        "rejected": "奖励模型就是让回答更好，没有定义，属于胡说。"
    },
    {
        "prompt": "解释缓存击穿",
        "chosen": "先下定义，再说明热点 key 失效时大量请求打到后端，并给出例子。",
        "rejected": "只说缓存有问题，但没有定义。"
    }
]

losses = []
margins = []

for row in dataset:
    r_plus = score_answer(row["prompt"], row["chosen"])
    r_minus = score_answer(row["prompt"], row["rejected"])
    loss = bt_loss(r_plus, r_minus)
    margin = r_plus - r_minus

    losses.append(loss)
    margins.append(margin)

    print(row["prompt"])
    print("r_plus =", round(r_plus, 3), "r_minus =", round(r_minus, 3))
    print("margin =", round(margin, 3), "loss =", round(loss, 3))
    print()

assert all(m > 0 for m in margins)
assert sum(losses) / len(losses) < 0.5
```

这段代码体现了三个关键点：

1. 训练目标只关心 `r_plus - r_minus`。
2. 同一条样本必须有 `chosen` 和 `rejected`。
3. 日志里最值得监控的指标之一是 `margin`，也就是平均分差。

如果改成真实工程实现，流程通常是：

| 步骤 | 输入 | 输出 | 关键检查 |
|---|---|---|---|
| 候选生成 | prompt 集合 | 多个候选回答 | 解码策略是否足够多样 |
| 偏好标注 | 同 prompt 候选对 | chosen/rejected | 标注一致性是否稳定 |
| RM 训练 | `(x, y^+, y^-)` | 奖励模型 | loss、margin、reward mean |
| 下游优化 | policy + RM | 新 policy | 是否出现 reward hacking |

真实工程例子可以看代码助手。对于 prompt“写一个 LRU 缓存实现”，系统可能从不同温度、不同 checkpoint、不同采样策略生成 4 到 8 个候选，再通过规则筛掉明显不可运行的版本，然后把接近的版本交给标注员比较。这样采集到的偏好对更有信息量，因为模型之间差距细而真实。

---

## 工程权衡与常见坑

奖励模型数据构建最常见的问题，不是“样本太少”，而是“样本信号不对”。

先看一个总结表：

| Pitfall | 后果 | 兜底措施 |
|---|---|---|
| 候选全来自单一解码策略 | 学到表面风格偏好 | 混合温度、top-p、不同模型来源 |
| 标注员标准不一致 | 排序噪声大，RM 不稳定 | 明确 rubric，做交叉复标 |
| 简单样本过多 | 对强模型区分度低 | 主动挖 hard pairs |
| 偏好受长度影响 | 学成“越长越好” | 长度归因分析，加入等长对照 |
| 只看训练 loss | 下游效果虚高 | 看 win rate、OOD 集表现 |
| synthetic pair 过于线性可分 | 下游可被投机利用 | 引入人工难例和真实失败样本 |

所谓“表面风格偏好”，白话说就是模型学到一些并不真正代表质量的外观特征，例如：

- 回答更长
- 开头先说“当然可以”
- 更像安全模板
- 用更多项目符号
- 带标准免责声明

如果候选都来自同一个解码器，优选和劣选的差异往往很单一，奖励模型自然会抓住这些最容易学的模式。

一个玩具例子是：prompt 为“解释哈希表冲突”。候选 A 内容短但准确，候选 B 内容更长，还多了“下面我详细展开”。如果标注员经常把 B 选成更好，模型会逐渐把“篇幅长”误当成“质量高”。

真实工程里，这种问题会直接变成 reward hacking。比如代码补全任务中，模型发现只要多写解释、多加警告语，奖励模型分数就会上升，即便代码本身没有更正确。最后 PPO 会沿着这个方向过度优化，产出“看起来很负责、实际上没更有用”的回答。

另一个高频坑是标注一致性差。偏好数据不是问卷投票，它要求标注员对“正确性、完整性、遵守约束、风险”有相对统一的判断。如果一部分标注员偏爱简洁，一部分偏爱详尽，而 rubric 又没定义清楚，那么训练集中会同时出现互相冲突的信号。

因此，实践中常做三件事：

1. 先写清楚判定顺序，例如正确性 > 安全性 > 遵守格式 > 简洁性。
2. 对同一批样本做复标，检查 inter-annotator agreement，也就是标注员一致率。
3. 主动采集 on-policy 难例。所谓 on-policy，白话说就是“让当前模型生成它自己最容易犯错的样本”，而不是只用历史旧样本。

---

## 替代方案与适用边界

奖励模型加 PPO 是经典路线，但不是唯一选择。若团队资源有限，或者 reward scale 很难调稳，直接偏好优化方法通常更省事。

最常见替代方案是 DPO。它可以理解为：不显式训练一个单独的奖励模型，而是直接在 preference pair 上更新策略模型。对初学者来说，可以把它看成“把偏好对直接变成 policy 的训练目标”。

两种路线的差异可以概括如下：

| 方案 | 数据形式 | 训练链路 | 稳定性 | 算力成本 | 适用边界 |
|---|---|---|---|---|---|
| RM + RLHF | prompt + chosen/rejected | 先训 RM，再 PPO/GRPO | 链路更长，调参更重 | 高 | 需要迭代优化、在线采样、精细控制 |
| DPO | prompt + chosen/rejected | 直接训 policy | 通常更稳 | 中 | 想降低工程复杂度，快速用偏好数据对齐 |
| 单纯 SFT | prompt + target | 监督学习 | 最稳 | 低 | 只有演示数据，没有成对偏好 |

DPO 并不是“比奖励模型更高级”，而是适用边界不同。如果你需要一个可复用的打分器，用来做 rerank、过滤、在线采样、红队筛查，那么显式奖励模型仍然很有价值。如果你只是想把已有 preference 数据快速转成更好的回答策略，DPO 往往更直接。

还有一种常见替代思路是大量使用 synthetic preference pair，也就是“合成偏好对”。白话说就是先让模型自己生成候选，再用规则、教师模型或自动判别器构造 chosen/rejected。它的优势是便宜、快、可扩展；劣势是容易把生成器偏见复制到训练数据里。尤其在安全任务上，过于干净、过于线性可分的 synthetic pair，可能让模型学到错误边界。

因此适用边界很明确：

- 需要高可信偏好信号时，优先人工高质量 pair。
- 需要快速扩量时，可以引入 synthetic pair，但必须混入真实难例。
- 候选分布变化快时，优先 on-policy 采样。
- 如果团队没有能力维护 RL 链路，先做 DPO 往往更现实。

---

## 参考资料

- EmergentMind, Reward Models: Foundations & Advances: https://www.emergentmind.com/topics/reward-model-rm
- EmergentMind, Reward Modeling from Human Preferences: https://www.emergentmind.com/topics/reward-modeling-from-human-preferences
- The RLHF Book, Chapter 11 Preference Data: https://rlhfbook.com/c/11-preference-data
- Hugging Face TRL RewardTrainer Documentation: https://huggingface.co/docs/trl/v0.24.0/reward_trainer
- Hugging Face Forum, Reward value range: https://discuss.huggingface.co/t/reward-model-reward-value-range/169576
