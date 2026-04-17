## 核心结论

RAFT（Reward-ranked Fine-tuning，按奖励排序的微调）是一种轻量对齐方法。白话说，它不是让模型边生成边做复杂强化学习，而是先让模型一次生成多个答案，再用奖励模型挑出最好的那个，最后只拿这条高质量样本做普通监督微调。

它的核心优势不在“奖励”两个字，而在“解耦”。解耦的意思是把“生成候选”“打分排序”“参数更新”拆成三个独立阶段。这样做的直接结果是：训练阶段仍然是标准交叉熵，不需要像 PPO 那样同时维护 actor、critic、参考模型等复杂部件，显存压力和调试成本都更低。

一个新手版玩具例子是：同一个提示词，让模型回答 3 次，奖励模型分别打分 0.2、0.5、0.8，只留下 0.8 那条，然后把它当作“标准答案”继续训练。下一轮再用更新后的模型重复这个过程。

| 阶段 | 功能 |
|---|---|
| 生成 | 对每个 prompt 采样 $K$ 个候选答案 |
| 评价 | 用奖励模型给候选排序，找出最高分 |
| 训练 | 只用 top-1 样本做监督微调 |
| 解耦优势 | 生成和训练分开执行，节省显存，便于批处理和排错 |

适合资源受限场景的原因很直接：RAFT 把“对齐”问题改写成“高质量数据构造”问题。你不需要在线强化学习循环，只需要有一个能打分的奖励模型，以及一套稳定的 SFT（supervised fine-tuning，监督微调）训练脚本。

---

## 问题定义与边界

RAFT 解决的问题是：当你没有完整 RLHF 基础设施时，怎么让模型更偏向“高质量回答”。RLHF 是 Reinforcement Learning from Human Feedback，白话说，就是用人类偏好或其代理信号来调整模型输出。RAFT 选择了一条更便宜的路：奖励模型只负责排序，不直接参与梯度更新。

这里的“奖励模型”是一个打分器。白话说，它像一个阅卷老师，只判断多个答案里谁更好，不负责生成答案。RAFT 的关键假设是：这个老师虽然不完美，但大体能排出相对正确的顺序。

边界也很明确。

第一，只能微调一个主模型时，RAFT 很合适；如果你已经有成熟的 actor-critic 训练系统，它不一定是唯一选择。

第二，生成阶段必须对每个提示批量采样 $K$ 个候选。如果 $K=1$，就没有“排名”这一步，RAFT 退化成普通 SFT。

第三，它不适合特别依赖输出多样性的任务。因为大多数低分样本不会进入梯度，模型会越来越集中在奖励模型偏好的表达方式上。

可以把它理解成一道过滤题。你给模型 10 次作答机会，只收第 1 名，然后把第 1 名当答案再讲给模型听。这样能稳步提高平均质量，但代价是很多“次优但有价值”的样本被丢掉了。

| 资源维度 | RAFT | 传统 RLHF / PPO |
|---|---|---|
| 主训练模型数 | 通常 1 个主模型 + 1 个奖励模型 | 常见为 actor + critic + reference + reward |
| 显存压力 | 较低 | 较高 |
| 在线训练复杂度 | 低，离线构造数据即可 | 高，需要在线采样和策略更新 |
| 调试难度 | 中等，主要看采样与打分 | 高，要看奖励、KL、优势估计等 |
| 适用场景 | 资源有限、想快速做对齐增强 | 大规模预算、追求更强策略优化 |

所以，RAFT 的问题定义不是“学会最优策略”，而是“从当前策略生成的数据里筛出更好的样本，再把这些样本灌回模型”。

---

## 核心机制与推导

RAFT 的采样阶段本质上等价于 rejection sampling（拒绝采样）。白话说，就是先多生成几个候选，再把不满意的扔掉，只保留最优者。

对一个输入 $x$，当前模型 $\pi_\theta$ 生成 $K$ 个候选：
$$
y_1, y_2, \dots, y_K \sim \pi_\theta(\cdot|x)
$$

奖励模型 $r(x,y)$ 对每个候选打分后，选出：
$$
y^* = \arg\max_j r(x, y_j)
$$

然后把 $(x, y^*)$ 放入训练集 $\mathcal{D}$。更新阶段不再使用强化学习目标，而是直接做普通交叉熵最小化：
$$
\mathcal{L}_{\text{RAFT}} = -\mathbb{E}_{(x,y)\in\mathcal{D}} \log \pi_\theta(y|x)
$$

这一步很重要。它说明 RAFT 的训练目标和普通 SFT 完全同形，只是训练数据不再来自人工标注，而是来自“模型生成 + 奖励筛选”。

简化流程图可以写成：

`prompt -> 生成 K 个候选 -> 奖励模型排序 -> 取 top-1 -> 写入训练集 -> 监督微调 -> 新模型继续生成`

看一个数值玩具例子。提示词是 `Explain RAFT`，模型生成 3 个回答：

| 候选 | 奖励分数 |
|---|---|
| 回答 A：只说“RAFT 是一种训练方法” | 0.2 |
| 回答 B：解释“多候选 + 选最好”但不提训练目标 | 0.5 |
| 回答 C：同时提到 rejection sampling 与交叉熵更新 | 0.8 |

那么这轮只保留回答 C。接下来训练目标不是“让 C 比 A 好多少”，而是更直接地“让模型以后更可能生成 C 这种形式”。

这也是它和 PPO 的根本差异。PPO 会把“好多少”编码进策略梯度；RAFT 则把“谁最好”编码进数据集。前者直接优化策略，后者间接优化策略。

真实工程例子可以是安全拒答对齐。假设你有一批高风险提示，如“如何绕过权限检查”。主模型可能会给出 8 个候选，其中有的直接违规，有的空泛拒答，有的能在拒答同时给出合规替代建议。奖励模型给“拒答明确、理由充分、提供安全替代”的回答更高分，那么进入训练集的就是这类 top-1 输出。经过多轮 RAFT 后，模型更容易稳定地产生安全且有用的拒答模式。

---

## 代码实现

工程上最稳的做法是把数据生成脚本和训练脚本分开。原因有两个。

第一，资源隔离。生成阶段通常更吃推理吞吐，训练阶段更吃显存和梯度累积，把它们拆开后更容易分别调度机器。

第二，调试方便。你可以单独检查“奖励模型是不是选错了”，而不用每次都重跑训练。

最小可运行的玩具代码如下：

```python
from math import exp

def reward_fn(prompt: str, response: str) -> float:
    score = 0.0
    if "多候选" in response:
        score += 0.3
    if "奖励模型" in response:
        score += 0.3
    if "交叉熵" in response:
        score += 0.4
    return score

def raft_select(prompt: str, candidates: list[str]) -> tuple[str, float]:
    scores = [reward_fn(prompt, c) for c in candidates]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return candidates[best_idx], scores[best_idx]

def nll(prob: float) -> float:
    return -__import__("math").log(prob)

prompt = "Explain RAFT"
candidates = [
    "RAFT 是一种训练方法。",
    "RAFT 通过多候选和奖励模型选最优答案。",
    "RAFT 先做多候选采样，再用奖励模型排序，只保留最高分样本做交叉熵微调。"
]

best, score = raft_select(prompt, candidates)

assert best == candidates[2]
assert score > reward_fn(prompt, candidates[1])

# 玩具版“训练收益”检查：如果模型对 best 的概率提高，损失下降
old_prob = 0.2
new_prob = 0.6
assert nll(new_prob) < nll(old_prob)

print(best)
```

对应的伪代码流程是：

```python
for prompt in prompts:
    candidates = [model.generate(prompt) for _ in range(K)]
    scores = reward_model(prompt, candidates)
    best = candidates[argmax(scores)]
    train_dataset.append((prompt, best))

trainer.train(train_dataset)
```

如果你用 Hugging Face `Trainer`，训练部分几乎不用改。真正需要你写的是“批量生成 + 奖励打分 + top-1 落盘”的数据构造脚本。一个常见做法是：

1. 用当前 checkpoint 对全部 prompts 生成 $K$ 条候选。
2. 把候选写成 JSONL，保留 prompt、response、score、rank。
3. 只抽取 rank=1 的样本形成 SFT 数据集。
4. 调用现有 SFT 模板训练。
5. 用新 checkpoint 重复流程。

真实工程里，生成通常并行化，评分也并行化，而“写入 top-1 数据集”保持串行或做去重控制。这样可以避免重复 prompt、异常样本、脏数据混进训练集。

---

## 工程权衡与常见坑

RAFT 的最大风险不是训练不起来，而是“奖励模型把错误样本选成第一名”。这叫 reward hacking，白话说，就是模型学会了讨好打分器，但不一定真正更好。比如奖励模型偏爱“语气礼貌、结构整齐”的回答，结果一些内容空洞的模板化答案被反复选中。

另一个常见问题是 acceptance rate 过低。这里可以把 acceptance rate 理解成“每轮采样里能留下多少有效样本”的比例。虽然 top-1 总会存在，但工程上常会加阈值，例如分数低于某个门槛就整组丢弃。如果阈值过高，很多 prompt 最终没有训练样本，梯度就会变得很不稳定。

| 问题 | 现象 | 规避策略 |
|---|---|---|
| reward hacking | 输出越来越像“高分模板”，但信息价值下降 | 加 KL 或 $\lambda$ 正则，人工抽检 top 样本 |
| acceptance rate 低 | 可用训练样本太少，训练波动大 | 降低阈值、提高 $K$、调高采样温度 |
| 多样性坍塌 | 回答越来越像一个模板 | 混入不同来源数据，监控 n-gram 重复率 |
| 奖励噪声过大 | 排名不稳定，轮次间质量波动 | 提升奖励模型质量，使用成对偏好校准 |
| 数据偏置 | 某些题型被过度强化 | 按任务类型分桶采样，控制数据配比 |

为什么有时要保留少量低分样本做 validation？因为你需要确认模型是不是只学会了“像高分答案那样写”，却失去了对普通分布的覆盖能力。低分样本不一定进训练，但可以作为验证集的一部分，帮助你观察模型是否过拟合奖励偏好。

一个新手常见症状是：训练脚本跑起来了，但最终只有很少几个 prompt 进入训练集。如果你发现“只有 3 个 prompt 被保留下来”，通常不是训练器坏了，而是奖励模型太苛刻，或者阈值设得太高。优先检查三件事：评分分布、阈值、$K$ 的大小。

还要注意一个细节：如果奖励模型只看最终文本，不看生成代价，它可能会偏爱超长回答。解决办法通常是加长度惩罚、格式约束，或者在奖励模型输入中显式加入任务要求。

---

## 替代方案与适用边界

RAFT 不是 PPO 的简化版，而是另一种工程路径。PPO 是 Proximal Policy Optimization，白话说，它通过策略梯度直接推动模型朝高奖励方向移动；RAFT 则先筛数据，再做监督学习。两者都能做对齐，但资源结构和稳定性来源不同。

| 方案 | 资源成本 | 稳定性 | 对齐能力 | 适用边界 |
|---|---|---|---|---|
| 直接 SFT | 最低 | 高 | 中等偏弱 | 只有标注数据，没有奖励模型 |
| RAFT | 低到中 | 中等 | 较强 | 有奖励模型，预算有限，希望快速增强 |
| PPO / 传统 RLHF | 高 | 较高（前提是系统成熟） | 强 | 大规模训练、能承担复杂调参与多模型协同 |

RAFT 适合的典型路径是：先用 SFT 打底，让模型具备基本可用性；再用 RAFT 从模型自己生成的候选中筛出更优输出，做一轮或多轮高质量增强。这样做的好处是，基础能力来自人工监督，偏好提升来自奖励排序，两者职责清晰。

如果你已经有一套稳定的 PPO pipeline，RAFT 更像一个低成本补充。比如在某个垂直任务上快速扩充高质量样本，先跑一轮 RAFT 把模型拉到更好的初始点，再决定是否进入更贵的 RLHF 训练。

反过来，如果奖励噪声很大，或者任务本身需要探索大量不同策略，PPO 往往更稳。因为 PPO 不只看“谁第一”，还利用连续奖励信号更新策略；RAFT 只保留 top-1，会丢掉大量排序信息。

所以可以用一句话判断边界：当你的瓶颈是算力和训练系统复杂度，RAFT 往往划算；当你的瓶颈是奖励噪声和策略优化精度，RAFT 可能不够。

---

## 参考资料

- RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment. arXiv / TMLR, 2023.
- RAFT 官方 GitHub：rejection sampling fine-tuning pipeline. https://github.com/RLHFlow/RAFT
- EmergentMind: Reward-Ranked Fine-Tuning. 2025 更新。
- Hugging Face Papers: RAFT 论文索引页。https://huggingface.co/papers/2304.06767
- 关于 reward hacking 与 acceptance rate 讨论的 ResearchGate 页面。https://www.researchgate.net/publication/370058357_RAFT_Reward_rAnked_FineTuning_for_Generative_Foundation_Model_Alignment
