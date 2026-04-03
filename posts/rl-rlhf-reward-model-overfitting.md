## 核心结论

奖励模型过拟合，指的是模型把“人类偏好”学成了训练集里的表面模式，而不是真正学会判断回答质量。对白话解释就是：它不是学会了“什么答案更好”，而是学会了“什么样的模板更像高分答案”。

在 RLHF 里，奖励模型最常见的问题不是训练损失降不下去，而是训练分数不断升高、验证排名能力却开始下降。更具体地说，若训练阶段的 reward 持续上升，但验证集的 `pairwise accuracy` 和 $r(y^+)-r(y^-)$ 间隔开始变差，同时在线采样质量下降，这通常就是进入 reward overfit，后续再训练只会让策略更容易学会“骗分”。

因此，监控重点不能只放在 loss。真正有判别力的是两类指标：

| 现象 | 更可信的解释 | 应对措施 |
| --- | --- | --- |
| 训练 loss 下降、训练 reward 上升 | 只说明模型在记忆训练对比对 | 不能单独作为停止条件 |
| 验证 pairwise accuracy 上升 | 模型在未见样本上还能排对顺序 | 可以继续训练 |
| 验证 gap 上升后持平 | 学到稳定区分能力 | 接近最佳停止点 |
| 训练 reward 继续上升，但验证 accuracy/gap 下降 | 典型过拟合，开始记模板、长度、格式 | 早停、减容量、加正则、重做切分 |
| 在线生成越来越像固定套话，但 RM 打分更高 | policy 开始 exploit RM | 长度归一化、对抗验证、重训 RM |

工程上最有效的防线通常不是单一技巧，而是组合：按 prompt 去重切分、控制 RM 容量、加 dropout 和 L2、使用早停、处理长度偏差、做对抗验证。它们共同的目标都一样：阻止模型记住“标注偏好模板”。

---

## 问题定义与边界

奖励模型训练的输入通常是三元组 $(x, y^+, y^-)$：同一个 prompt $x$ 下，人类更喜欢的回答 $y^+$ 和较差的回答 $y^-$。这里的目标不是做“文本分类”，而是做“排序判断”。排序判断的白话解释是：模型要能在两个候选回答里把更好的那个排在前面。

过拟合的边界要先说清楚。下面几种情况不能混为一谈：

| 问题类型 | 定义 | 常见信号 | 典型 probe |
| --- | --- | --- | --- |
| 训练不足 | 模型还没学会基本偏好 | 训练和验证都低 | 再训练、查数据质量 |
| 普通过拟合 | 训练集表现好，验证变差 | train-validate gap 变大 | 早停、正则化 |
| prompt bias | 学到 prompt 模板偏好而非内容偏好 | 换个写法就掉分 | 模板改写、prompt 分组 |
| length bias | 默认更偏爱长答案 | 冗长回答总拿高分 | 长度归一化、长度匹配 |
| OOD 脆弱性 | 遇到分布外样本就误判 | 轻微改写后准确率暴跌 | 对抗样本、变换测试 |

玩具例子最容易看清边界。

玩具例子：训练集中经常出现两类 prompt。

1. `Please summarize this news in three bullets`
2. `Summarize the following article`

如果高质量答案大多出现在第一种模板下，而低质量答案大多出现在第二种模板下，奖励模型就可能把 “Please...in three bullets” 当成高分信号。此时即便回答内容普通，只要套上熟悉模板也会得高分。这不是学到了偏好，而是学到了 prompt-template bias。ACL 2025 的 Prompt Bias Calibration 工作就是在处理这类问题。

所以，“奖励模型泛化” 的边界不是它能否把训练集分开，而是它在新 prompt、新措辞、新长度、新格式下，是否仍然能维持稳定排序能力。reWordBench 这类基准之所以重要，就是因为它用改写、重排、保义变换去检验 RM 是否只会认表面形式。

---

## 核心机制与推导

奖励模型最常见的训练形式基于 Bradley-Terry 模型。Bradley-Terry 的白话解释是：把“哪个回答更好”转成“两个分数差有多大”。

设奖励模型为 $r(x,y)$，则在同一个 prompt 下，正例比负例更受偏好的概率写成：

$$
P(y^+ \succ y^- \mid x)=\sigma\big(r(x,y^+) - r(x,y^-)\big)
$$

其中 $\sigma(z)=\frac{1}{1+e^{-z}}$ 是 sigmoid。直观上，分数差越大，模型越确信正例应该排前面。

对应的训练目标通常是：

$$
\mathcal{L}=-\mathbb{E}\left[\log \sigma \big(r(x,y^+) - r(x,y^-)\big)\right]
$$

这里的核心量不是单独的 $r(x,y)$，而是 gap：

$$
\Delta = r(x,y^+) - r(x,y^-)
$$

gap 的白话解释是：模型把好答案和坏答案拉开了多远。$\Delta$ 大，排序更稳；$\Delta$ 小甚至为负，说明排序在出错。

举一个最小数值例子。若某个 prompt 上：

$$
r(x,y^+)=2.0,\quad r(x,y^-)=0.5
$$

那么 gap 为：

$$
\Delta=1.5
$$

对应偏好概率为：

$$
\sigma(1.5)\approx 0.8176
$$

也就是模型大约以 81.76% 的置信度认为正例更好。这个数值可以帮助理解为什么验证 gap 比单看 loss 更有意义：loss 只告诉你“还在优化”，gap 才告诉你“排序能力是否在变强”。

真正危险的情况是训练和验证背离。比如训练前 3 个 epoch 内，训练 gap 从 1.2 升到 4.8，看起来很好；但验证 gap 从 1.4 降到 1.1，验证 pairwise accuracy 从 74% 降到 69%。这说明模型越来越擅长在训练分布里“拉开分数”，却越来越不擅长在未见 prompt 上正确排序。若此时还继续用它做 PPO 或 best-of-n 采样，策略就会朝着 RM 的漏洞方向优化，出现 reward overoptimization。

为什么会这样？因为 RM 会利用任何稳定但错误的捷径，包括：

1. 长度更长就给更高分。
2. 某种回答格式更像“标注者喜欢的样子”。
3. 某类 prompt 模板在训练集中和高分答案强相关。
4. 来自特定模型家族的语言风格更容易被误认成高质量。

Kim 等人在 ACL 2025 的工作强调，RM 评估如果不能反映“被策略继续优化后会不会被钻空子”，那它的基准分数就和真实效用弱相关。也就是说，奖励模型不是“静态分类器”，而是“将来会被策略持续攻击的目标函数近似器”。

---

## 代码实现

下面给一个最小可运行示例，演示三个事情：

1. 如何计算 Bradley-Terry loss。
2. 如何同时监控 pairwise accuracy 和平均 gap。
3. 为什么早停条件不该只看训练 loss。

```python
import math

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def pairwise_metrics(chosen_scores, rejected_scores):
    assert len(chosen_scores) == len(rejected_scores)
    gaps = [c - r for c, r in zip(chosen_scores, rejected_scores)]
    probs = [sigmoid(g) for g in gaps]
    acc = sum(g > 0 for g in gaps) / len(gaps)
    mean_gap = sum(gaps) / len(gaps)
    bt_loss = -sum(math.log(max(sigmoid(g), 1e-12)) for g in gaps) / len(gaps)
    return {
        "accuracy": acc,
        "mean_gap": mean_gap,
        "bt_loss": bt_loss,
        "mean_prob": sum(probs) / len(probs),
    }

# 训练集越来越“好看”
train_chosen = [4.2, 3.8, 5.1, 4.7]
train_rejected = [0.5, 1.0, 1.2, 0.7]

# 验证集没有同步变好，说明泛化在恶化
valid_chosen = [1.5, 1.1, 1.7, 1.2]
valid_rejected = [0.8, 0.9, 1.0, 1.1]

train_metrics = pairwise_metrics(train_chosen, train_rejected)
valid_metrics = pairwise_metrics(valid_chosen, valid_rejected)

assert train_metrics["accuracy"] == 1.0
assert train_metrics["mean_gap"] > valid_metrics["mean_gap"]
assert valid_metrics["accuracy"] < 1.0
assert train_metrics["bt_loss"] < valid_metrics["bt_loss"]

# 一个简单的早停判据：验证 accuracy 或 gap 不再提升
best_valid_acc = 0.75
best_valid_gap = 0.55
should_early_stop = (
    valid_metrics["accuracy"] <= best_valid_acc and
    valid_metrics["mean_gap"] <= best_valid_gap
)

assert should_early_stop is True
print(train_metrics)
print(valid_metrics)
```

把它放到真实训练里，关键不是这几行公式，而是数据切分和日志结构。

一个更接近工程实践的训练框架如下：

```python
for epoch in range(max_epochs):
    train_one_epoch(
        model=rm,
        dropout=0.1,
        weight_decay=0.01,   # L2
    )

    # 核心前提：验证集必须按 prompt 组切开，不能和训练集共享模板簇
    valid_acc, valid_gap = evaluate_pairwise(valid_loader)

    # 长度归一化，避免把“更长”误当成“更好”
    valid_len_norm_acc = evaluate_pairwise(valid_loader, length_normalize=True)

    # 对抗验证：加入改写、模板变换、极端冗长样本
    adv_acc, adv_gap = evaluate_pairwise(adversarial_loader)

    if early_stop(valid_acc, valid_gap, adv_acc):
        break
```

这里每个部件都有明确作用。

`prompt 去重切分`：先按 prompt 或模板哈希分组，再划分 train/valid/test，避免同模板泄漏到验证集。它解决的是“看似泛化，其实见过近似 prompt”。

`dropout`：训练时随机屏蔽一部分神经元，白话解释是强迫模型不要过度依赖某几个记忆路径。

`L2`：限制参数过大，白话解释是防止模型把局部模式记得太死。

`长度归一化`：把分数除以长度，或显式减去长度惩罚，防止 verbosity bias，即“话越多分越高”。

真实工程例子：一个问答系统用 7B SFT 模型初始化 RM，再用偏好对训练。最初团队只看 train loss 和 held-out loss，发现 RM 分数不断改善，于是进入 PPO。上线后模型回答开始变得更长、更礼貌、更像固定模板，但事实性和任务完成度下降。排查后发现验证集和训练集共享大量 prompt 模板，RM 学会了“礼貌长回答”这个高分捷径。修复措施通常包括：按 prompt hash 重切分、把 RM 容量控制到与 policy 同级或略小、加入长度惩罚、在每个 epoch 后跑 adversarial validation。修完后，训练 reward 的绝对值可能没以前高，但线上采样质量会更稳。

---

## 工程权衡与常见坑

奖励模型防过拟合不是“加一个正则项”就结束，而是工程系统设计问题。

| 风险/坑 | 探测信号 | 缓解措施 |
| --- | --- | --- |
| 只看 training loss | loss 很漂亮，但线上质量变差 | 强制记录 valid accuracy/gap |
| train/valid 按样本随机切分 | 验证分异常高 | 改成按 prompt/template 分组切分 |
| RM 容量过大 | 很快把训练集学满 | 降模型规模、减 epoch |
| RM 容量过小 | 训练和验证都低 | 提升容量，但别超过 policy 太多 |
| 忽略长度偏差 | 回答越来越长 | 长度惩罚、长度匹配采样 |
| 不做对抗验证 | 上线后才暴露漏洞 | 加 paraphrase、格式扰动、OOD 样本 |
| 只看 RM 离线指标 | policy 优化后突然崩 | 联动观察在线采样质量 |

几个常见误区需要单独指出。

第一，把“训练 reward 持续升高”理解成“奖励模型越来越好”。这是最危险的误判。RM 是被 policy 用来优化的目标，目标函数越容易被钻空子，训练分数往往越好看。

第二，把“验证 loss”当成唯一泛化指标。对于偏好学习，排序准确率和 gap 往往更贴近下游效用，因为它们直接对应“好回答是否排在坏回答前面”。

第三，忽略模板偏差。Prompt Bias Calibration 这类工作说明，RM 不只会学长度偏差，也会学 prompt 模板偏差。模型一旦把“某种格式”误当成“高质量”，策略就会迅速学会复读这种格式。

第四，过早把精力都花在 PPO 上。很多系统的问题不是 PPO 参数，而是 RM 本身已经开始过拟合。RM 错了，后面的 RL 只会更高效地放大错误。

---

## 替代方案与适用边界

如果计算预算有限，或你只是要快速验证偏好数据是否有价值，可以考虑 DPO。DPO 的白话解释是：直接用偏好对训练策略，而不是先单独训练一个奖励模型再做 RL。

它的优势是流程短、训练更稳定、工程复杂度低。但这不代表它自动解决了过拟合。DPO 仍然可能学到模板偏差、长度偏差和数据泄漏，只是问题不再显式表现为“RM 被 exploit”，而是表现为“策略直接过拟合偏好对”。

| 维度 | 显式 RM + RLHF | DPO |
| --- | --- | --- |
| 是否有单独 reward model | 有 | 没有 |
| 工程复杂度 | 高 | 低 |
| 是否能复用 reward 评分器 | 可以 | 不可以 |
| 是否便于做 reward 诊断 | 更方便 | 较弱 |
| 是否适合快速起步 | 一般 | 更适合 |
| 多目标奖励扩展 | 更自然 | 较麻烦 |

适用边界可以这样理解。

如果你只有 20K 左右偏好对，目标是快速做出一个可用版本，DPO 常常更合适。

如果你需要独立的 reward 分数、要做 best-of-n、拒答控制、多目标权衡，或者后续要做在线 RL，那么单独训练 RM 仍然更有价值。

如果你的数据明显存在 prompt 模板重复、长度偏差或多模型风格偏差，那么不论是 RM 还是 DPO，都不能跳过去重、分组切分和鲁棒验证这一步。方法变了，泛化约束没有变。

---

## 参考资料

- Kim et al., “Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization,” ACL 2025. https://aclanthology.org/2025.acl-long.649/
- Wang et al., “Removing Prompt-template Bias in Reinforcement Learning from Human Feedback,” Findings of ACL 2025. https://aclanthology.org/2025.findings-acl.1237/
- Wu et al., “reWordBench: Benchmarking and Improving the Robustness of Reward Models with Transformed Inputs,” EMNLP 2025. https://aclanthology.org/2025.emnlp-main.167/
- Chittepu et al., “Reinforcement Learning from Human Feedback with High-Confidence Safety Guarantees,” Reinforcement Learning Journal 2025. https://rlj.cs.umass.edu/2025/papers/Paper62.html
- Sedhain, “Reward Modeling and DPO: Learning What ‘Good’ Means,” 2026. https://mesuvash.github.io/blog/2026/reward-modeling/
- Bukharin et al., “Adversarial Training of Reward Models,” COLM 2025. https://openreview.net/forum?id=H6Ae8Po6fS
- Bu et al., “Beyond Excess and Deficiency: Adaptive Length Bias Mitigation in Reward Models for RLHF,” Findings of NAACL 2025. https://aclanthology.org/2025.findings-naacl.169/
- QubitTool Team, “RLHF Complete Guide [2026].” https://qubittool.com/blog/rlhf-reinforcement-learning-human-feedback-guide
