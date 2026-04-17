## 核心结论

拒绝采样微调，英文是 Rejection Sampling Fine-Tuning，简称 RFT。白话说法是：让模型对同一个问题多答几次，只留下“能被程序或规则证明是对的”那些答案，再把这些正确轨迹当成新教材继续做监督微调。

它解决的不是“模型不会生成文字”，而是“模型会生成很多看起来像对、实际上不可靠的推理路径”。RFT 的核心价值是把“生成”与“筛选”拆开：生成阶段允许模型自由探索，训练阶段只学习通过验证的轨迹。

先看一个新手版玩具例子。现在有一道题“37 + 58 = ?”，模型对同一个 prompt 采样 5 次，得到 95、96、85、95、94。验证器只做一件事：检查结果是否等于 95。于是两条通过，三条淘汰。接下来训练时，不去分析错在哪里，而是直接把那两条正确答案加入正例集 $D^+$，让模型更倾向输出 95 这条路径。

RFT 和 RLHF 的差别可以先抓住一句话：RFT 是“筛出好样本后做 SFT”，RLHF 是“给样本打分后做策略优化”。SFT 是监督微调，白话说法是“拿标准答案直接教模型”；RLHF 是基于奖励的强化学习，白话说法是“不给标准答案，只告诉模型这次好还是差，让它自己调整策略”。

一个简化流程图可以写成：

`采样 -> 验证 -> 追加到 D⁺ -> SFT -> 下一轮采样`

下表先给出关键术语对照。

| 维度 | RFT | RLHF |
|---|---|---|
| 学习信号 | 只保留验证通过样本 | 奖励分数或偏好排序 |
| 是否需要奖励模型 | 通常不需要 | 通常需要 |
| 训练更新 | 标准 SFT | PPO/DPO/GRPO 等策略或偏好优化 |
| 失败样本处理 | 多数直接丢弃 | 通常仍参与学习 |
| 最适合的任务 | 可自动判定对错 | 难自动验证但能做人类偏好排序 |

---

## 问题定义与边界

RFT 的目标很明确：在尽量少依赖人工标注、奖励模型和复杂强化学习的情况下，提高模型在推理、数学、代码生成这类任务上的正确率。

这里有一个前提词必须先说明。验证器，英文是 validator，白话说法是“一个能自动判断答案过不过关的裁判”。如果没有可靠裁判，RFT 就失去基础。

因此，RFT 最适合的不是所有任务，而是“结果可验证”的任务。例如：

- 数学题：检查最后结果是否满足等式
- 编程题：运行测试用例，看代码是否通过
- SQL 题：执行查询，看结果表是否正确
- 定格式抽取：检查 JSON 是否合法、字段是否齐全

相反，如果任务是“这段文案是否更有说服力”“这段回答是否更礼貌”“这段总结是否更有洞察力”，自动验证就很弱。这类任务更接近偏好学习，RFT 的优势会明显下降。

可以把任务适配性看成三个条件：可验证性、采样成本、失败信号质量。

| 任务属性 | 含义 | 对 RFT 的影响 | 适配性 |
|---|---|---|---|
| 可验证性高 | 能用程序判断对错 | 是核心前提 | 高 |
| 采样成本低 | 同题多次生成不太贵 | 能支持大规模筛选 | 高 |
| 失败信号明确 | 错误能被稳定识别 | 便于过滤脏数据 | 高 |
| 可验证性低 | 对错依赖主观判断 | 验证器容易失真 | 低 |
| 单次采样很贵 | 长链推理或长代码成本高 | 很难多次采样 | 中或低 |
| 错误类型复杂 | 很多答案“部分正确” | 二值筛选过于粗糙 | 中 |

再看一个边界例子。对一道数学题，模型可以自己生成 8 次解题过程，再用程序检查最后一步是否满足原式，只保留通过的轨迹。这很适合 RFT。可如果任务是“写一段更打动人的产品介绍”，很难写出一个可靠程序去判定“更打动人”，这时 RFT 的筛选机制就会失效。

所以，RFT 不是“更便宜的万能 RLHF”，而是一种对任务结构要求很高的方法。它依赖两个预算：

1. 验证预算：你必须能自动、稳定、低成本地判题。
2. 采样预算：你必须能承受同一问题多次生成。

---

## 核心机制与推导

RFT 的核心对象是正例集 $D^+$。正例集，白话说法是“当前已知确实通过验证的一批好样本”。训练不是直接在原始模型输出上做，而是在这个不断扩大的正例集上做。

设当前模型为 $\pi_\theta$，给定初始状态或 prompt $s_0$，对每个任务采样 $k$ 条轨迹：

$$
\tau_j \sim \pi_\theta(\cdot \mid s_0), \quad j=1,\dots,k
$$

轨迹 $\tau_j$ 可以理解成“一整条生成路径”，比如完整的思维步骤、代码输出或答案文本。然后用验证器 $R(\tau)$ 做二值判定：

$$
R(\tau)=
\begin{cases}
1, & \text{通过验证} \\
0, & \text{未通过验证}
\end{cases}
$$

本轮新得到的正样本集合是：

$$
D^+_{\text{new}} = \{\tau_j \mid R(\tau_j)=1\}
$$

总的正例集更新为：

$$
D^+ \leftarrow D^+ \cup D^+_{\text{new}}
$$

之后，不做强化学习策略梯度，而是直接在 $D^+$ 上做标准监督微调。损失函数通常写成：

$$
\mathcal{L}_{\mathrm{SFT}} = - \sum_l m_l \log \pi_\theta(t_l \mid t_{<l})
$$

这里 $t_l$ 是第 $l$ 个 token，$m_l$ 是掩码。掩码，白话说法是“告诉模型哪些 token 该计入损失，哪些不该算分”。例如只训练回答部分，不训练 prompt 部分。

这个公式的意思很直接：让模型提高“正确轨迹中每个 token 出现”的概率。因为训练数据只来自验证通过样本，模型被持续推向“更容易走到可验证正确结果”的区域。

### 数值演示

设某类题上，当前模型一次采样答对的概率是 $p=0.2$。每题采样 $k=5$ 次，那么至少采到一个正确答案的概率是：

$$
1-(1-p)^k = 1-0.8^5 \approx 0.672
$$

也就是说，虽然单次只有 20% 正确，但做 5 次采样后，约 67.2% 的题能至少找到一条正确轨迹。只要找到并留下这些轨迹，下一轮 SFT 就会提高模型在类似题上的正确概率。

继续做玩具例子。某一轮中 100 道题，每题采样 5 次，共 500 条轨迹。假设只有 40 条通过验证，那么这 40 条就进入 $D^+$。用这 40 条做一轮 SFT 后，下一轮同样设置下，单次正确率可能从 20% 提到 28%。这不是理论保证值，而是机制上的预期方向：正确轨迹被持续重放，模型更容易复现它们。

### 简化伪代码

```text
initialize D_plus = []
for round in training_rounds:
    for prompt in prompts:
        samples = sample_k_times(model, prompt, k)
        passed = [x for x in samples if validator(x)]
        D_plus.extend(passed)
    train_with_sft(model, D_plus)
```

这里最重要的逻辑不是“采样很多次”，而是“采样很多次后，只学习通过验证的轨迹”。这就是 rejection sampling 的含义。rejection sampling，白话说法是“先大量提出候选，再把不合格的拒掉”。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不训练真正的大模型，而是模拟 RFT 的数据闭环：对算术题多次采样，验证答案是否正确，把通过的样本加入 $D^+$，再统计下一轮更偏向正确答案的效果。

```python
import random
from collections import Counter

def make_prompt(a, b):
    return f"{a} + {b} = ?"

def validator(a, b, answer):
    return answer == a + b

def sample_answers(a, b, k, skill_bias):
    """
    skill_bias 越高，模型越容易采到正确答案。
    """
    outputs = []
    correct = a + b
    for _ in range(k):
        if random.random() < skill_bias:
            outputs.append(correct)
        else:
            outputs.append(correct + random.choice([-2, -1, 1, 2, 10]))
    return outputs

def rft_round(dataset, k, skill_bias):
    d_plus = []
    for a, b in dataset:
        samples = sample_answers(a, b, k=k, skill_bias=skill_bias)
        for ans in samples:
            if validator(a, b, ans):
                d_plus.append({
                    "prompt": make_prompt(a, b),
                    "answer": str(ans),
                    "passed": True,
                })
    return d_plus

def estimate_pass_rate(dataset, k, skill_bias, trials=200):
    success = 0
    for _ in range(trials):
        for a, b in dataset:
            samples = sample_answers(a, b, k=k, skill_bias=skill_bias)
            if any(validator(a, b, ans) for ans in samples):
                success += 1
    total = len(dataset) * trials
    return success / total

random.seed(42)

dataset = [(2, 3), (7, 9), (11, 8), (13, 6)]
k = 5

rate_before = estimate_pass_rate(dataset, k=k, skill_bias=0.20)
d_plus = rft_round(dataset, k=k, skill_bias=0.20)

# 用通过样本数量模拟一轮 SFT 后能力上升
improved_bias = min(0.20 + len(d_plus) * 0.01, 0.60)
rate_after = estimate_pass_rate(dataset, k=k, skill_bias=improved_bias)

assert len(d_plus) > 0
assert rate_after >= rate_before

print("accepted_samples =", len(d_plus))
print("rate_before =", round(rate_before, 3))
print("rate_after =", round(rate_after, 3))
```

这段代码对应真实系统里的四个模块：

| 模块 | 作用 | 真实工程中的实现 |
|---|---|---|
| `sample_answers` | 多次采样 | 调用当前模型，多 temperature 或多 seed 生成 |
| `validator` | 自动判题 | 单元测试、符号检查、执行环境、规则系统 |
| `d_plus` | 正例缓存 | JSONL、Parquet、训练样本池 |
| “improved_bias” | 模拟 SFT 更新 | 真正训练 LoRA 或全量参数 |

如果写成更接近生产环境的伪代码，可以是：

```python
for prompt in prompt_batch:
    trajectories = model.sample(prompt, num_samples=k, temperature=0.8)
    accepted = []
    for traj in trajectories:
        if validator(traj):
            accepted.append({"prompt": prompt, "response": traj})
    D_plus.extend(accepted)

trainer.sft_step(D_plus_recent)
```

### 数据结构怎么组织

最常见的做法是把 $D^+$ 组织成 `{prompt, response, meta}`。`meta` 一般记录：

- 采样轮次
- 模型版本
- 验证器结果
- 题目难度
- 去重哈希

是否清空 $D^+$，取决于目标。

- 累积保留：适合想不断扩大正样本池的场景。
- 滑动窗口：适合担心旧数据风格过时、分布漂移的场景。
- 分层缓存：把简单题和难题分开存，避免简单题淹没难题。

### 真实工程例子

StarCoder2-Instruct 的 SelfCodeAlign 路线，本质上就是 RFT 思路在代码任务上的工程化版本：先让代码模型自己生成指令响应或代码解答，再通过执行和测试筛选，最后把通过的数据重新拿来训练同一个模型。这里“执行验证”就是强验证器，因为代码能不能运行、测例能不能通过，通常能程序化判断。

这个例子说明 RFT 为什么在代码领域尤其有效。因为代码任务的“正确”比开放式文本更容易定义，验证器也更强，筛出来的正样本噪声更低。

---

## 工程权衡与常见坑

RFT 看起来简单，但真正难的是数据分布，而不是公式。

第一个坑是“只学会容易题”。假设每轮只有 10% 样本通过验证，而这些通过样本几乎都来自简单题，那么 $D^+$ 会越来越偏向简单模式。最后模型确实更稳定了，但只是更会做容易题，难题没有足够监督信号。

第二个坑是“失败样本全丢后，负信号缺失”。RFT 的优势是干净，但代价是信息浪费。模型不知道某条路径为什么错，只知道“不要学它”。对于一些需要细粒度纠错的任务，这会让学习效率变低。

第三个坑是“验证器过窄”。如果验证器只检查最终答案，不检查推理过程，模型可能学会投机。比如数学题里瞎写过程但最后答案碰巧对；代码题里只对公开测试集过拟合。这会把伪正例引入 $D^+$。

第四个坑是“重复样本污染”。同一个 prompt 如果总是保留几乎一样的轨迹，训练会被高频模板主导，导致表达和解法单一。

下表给出常见风险和缓解手段。

| 坑或风险 | 典型表现 | 缓解措施 |
|---|---|---|
| 简单题偏置 | 通过样本大多来自低难度任务 | adaptive sampling、课程学习、按难度配额采样 |
| 负信号缺失 | 模型知道什么对，不知道为什么错 | 混入少量偏好对比数据或错误分析数据 |
| 验证器过窄 | 伪正确样本进入 $D^+$ | 增加多样验证器，检查过程、边界条件和鲁棒性 |
| 数据重复 | 同质化严重，泛化下降 | 去重、聚类采样、限制每题保留条数 |
| 采样成本过高 | GPU 花在重复生成上 | 只对高价值题多采样，其他题低采样 |
| 分布漂移 | 旧正例不再适配新目标 | 滑动窗口或按轮次衰减旧样本权重 |

一个工程上常见的例子是：团队发现模型每轮都能筛出不少通过样本，指标也在涨，但最后只涨在基础题集上，难题集几乎不动。根因往往不是训练器坏了，而是 $D^+$ 被简单题占满了。解决办法通常不是“再训更久”，而是改变采样和入池策略，比如对难题加大 $k$，或者强制每轮纳入一定比例的中高难样本。

---

## 替代方案与适用边界

RFT 不是唯一选择。至少有三类常见路线可以比较：RFT、RLHF、SPIN。

SPIN，英文是 Self-Play Fine-Tuning，白话说法是“让模型在自我博弈中生成更强数据，再反过来训练自己”。它和 RFT 一样都走自生成数据路线，但更强调迭代对抗或自比较，而不只是“过了验证就留下”。

| 方法 | 数据来源 | 监管信号 | 优点 | 局限 |
|---|---|---|---|---|
| RFT | 模型自采样 | 验证器通过/失败 | 流程简单，训练稳定，适合可验证任务 | 强依赖验证器，负信号利用弱 |
| RLHF | 人类偏好或奖励模型 | 排序或奖励分数 | 适合开放式偏好任务 | 成本高，流程复杂 |
| SPIN | 模型自博弈生成 | 迭代对比或博弈信号 | 能持续制造更强训练对手 | 训练设计更复杂，稳定性依赖实现 |

可以这样理解三者边界：

- 如果任务能自动判题，比如代码、数学、形式推理，RFT 往往是第一候选。
- 如果任务核心是人类偏好，比如助手风格、礼貌性、帮助性，RLHF 更自然。
- 如果任务需要模型在自我对抗中不断超越当前水平，SPIN 更有吸引力。

还可以补一个对比句帮助理解：相比 RFT，RLHF 不会简单把失败样本扔掉，而是会利用奖励信号告诉模型“失败样本比成功样本差多少”；相比 RLHF，RFT 不需要额外奖励模型，但前提是你已经有一个足够强的验证器。

因此，最实用的工程判断不是“哪种方法更先进”，而是“我的任务能不能可靠验证”。一旦答案是“能”，RFT 往往会比想象中更直接有效。

---

## 参考资料

- [Rejection-sampling Fine-Tuning (RFT) 综述](https://www.emergentmind.com/topics/rejection-sampling-fine-tuning-rft-ad4c417c-416b-40b6-bf9a-4653b83ddcfb)：提供 RFT 的基本定义、公式写法和“采样-验证-SFT”闭环描述。
- [SPIN: Self-Play Fine-Tuning 项目页](https://uclaml.github.io/SPIN/)：展示自博弈式数据生成与迭代优化思路，适合对比 RFT 与自对齐方法的差异。
- [SelfCodeAlign / StarCoder2-Instruct 仓库](https://github.com/bigcode-project/selfcodealign)：给出代码领域无人工标注自对齐的工程实践，可作为 RFT 在编程任务中的真实案例参考。
