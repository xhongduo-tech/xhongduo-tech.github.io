## 核心结论

DeepSeek-R1 的关键突破，不是“模型把答案写得更长”，而是“模型在没有人工推理链监督的情况下，只凭可验证结果的奖励，自己学出了更稳的推理策略”。更直白地说，系统不告诉它标准解法，只告诉它这次答题有没有答对、输出格式是否合规；训练久了，模型就会自然偏向那些更容易拿分的生成轨迹。

在 DeepSeek-R1-Zero 的核心推理阶段，规则奖励可以写成：

$$
Reward_{rule}=Reward_{acc}+Reward_{format}
$$

其中：

- `Reward_acc`：正确性奖励，意思是“答案有没有通过验证器”。
- `Reward_format`：格式奖励，意思是“是否按要求输出推理区和最终答案区”。

这条路线的重要性在于，它把训练重点从“模仿人工示范的推理过程”转成“优化可自动验证的最终结果”。传统 SFT 可以理解成“老师先把标准解法写好，模型跟着学”；R1 的核心 RL 阶段更像“给模型一个能自动判分的考场，让它自己摸索高分写法”。这也是为什么它在数学、代码、逻辑题这类可验证任务上特别强。

最小玩具例子如下。题目是 `12 * 13 = ?`。模型可能输出：

```text
<think>
12*13 = 12*10 + 12*3 = 120 + 36 = 156
</think>
156
```

系统随后只做两件事：

1. 检查最终答案是不是 `156`。
2. 检查是否存在合法的 `<think>...</think>` 包裹，并且最终答案在标签外。

如果两项都满足，就给高奖励；如果答案错误，或者格式不合规，就给低奖励。训练足够久以后，模型会更常走“先拆解、再核验、再输出”的路径，因为这类路径更容易获得奖励。

公开结果说明，这种“只对结果和格式打分”的 RL 的确能逼出复杂推理行为。DeepSeek-R1 论文报告中，AIME 2024 达到 79.8%，MATH-500 达到 97.3%。这些数字本身不是本文重点，重点在于：它们证明了**不依赖大规模人工推理链标注，仅靠可验证奖励，也能把推理能力明显推高**。

下面这张表先把三类训练路线放在一起看：

| 训练路线 | 依赖什么数据 | 奖励信号来自哪里 | 人工成本 | 更适合什么任务 |
|---|---|---|---|---|
| 传统 SFT / CoT | 人工示范的推理链 | 拟合训练样本 | 高 | 通用问答、写作、风格控制 |
| 纯规则化 RL | 可自动判分题目 | 验证器、测试、格式检查 | 低到中 | 数学、代码、符号推导、逻辑题 |
| 混合路线 | 少量示范 + 自动奖励 + 偏好数据 | 规则奖励 + 偏好模型 | 中 | 推理任务 + 通用对话 |

---

## 问题定义与边界

DeepSeek-R1 解决的问题可以精确定义为：

**在没有人工推理链标签的前提下，让大模型仅凭“答对题会得分”这件事，逐步学会更稳定的推理策略。**

这里有两个关键词必须讲清楚。

第一，可验证。  
可验证的意思是：机器可以自动判断结果是否正确。比如数学题可以验算，代码题可以跑测试，符号化表达可以检查等价性，逻辑题可以用规则程序判分。只有当“对错”能被稳定定义时，规则奖励才可靠。

第二，规则化奖励。  
规则化奖励的意思是：奖励来自确定性程序，不来自另一个神经网络的主观打分。它的好处是目标明确、训练信号稳定；坏处是适用范围窄，因为不是每类任务都能写出可靠判题器。

任务边界因此非常清楚：

| 任务类型 | 是否适合纯规则化 RL | 原因 |
|---|---|---|
| 数学计算 | 适合 | 最终答案可直接验证 |
| 代码生成 | 适合 | 可运行测试、编译、静态检查 |
| 逻辑谜题 | 较适合 | 常可写显式判题器 |
| 定理证明的部分子任务 | 较适合 | 可借助 proof checker 或 verifier |
| 开放式摘要 | 不适合 | 很难定义唯一正确答案 |
| 文案创作 | 不适合 | 质量高度主观 |
| 情绪陪伴对话 | 不适合 | 奖励标准不稳定且依赖语境 |

这意味着 R1 不是“通吃所有任务的万能训练法”。如果任务本身没有稳定的自动判分器，那么“只靠结果奖励”就抓不住训练目标。比如让模型写一首诗，你无法像数学题那样写一个清晰的 `pass/fail` 程序，这时纯规则化 RL 就会失效。

对新手来说，可以把它理解成“只看最终卷面得分的训练”。老师不提前给你标准解法，也不逐句批改你的草稿，而是在你交卷后只告诉你“对了还是错了”。如果考试是数学、代码、选择题，这种机制可能有效；如果考试是“写一篇感人的散文”，就很难直接工作。

在最简单的实现里，奖励可以离散成 0/1：

$$
Reward_{acc}\in \{0,1\},\quad Reward_{format}\in \{0,1\}
$$

于是总奖励是：

$$
Reward_{rule}\in \{0,1,2\}
$$

也就是三档：

- `0`：答案错，格式也错
- `1`：答案对但格式错，或者答案错但格式对
- `2`：答案对且格式对

工程上也可以把奖励做成连续值，例如部分通过测试记 `0.4`，语言一致性再加 `0.1`，但原理不变：**奖励必须和可验证目标直接绑定**。

下面给一个可运行的最小 verifier 示例：

```python
import re

THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*(.+?)\s*$", re.S)

def check_format(text: str) -> int:
    match = THINK_BLOCK_RE.match(text)
    return 1 if match else 0

def extract_final_answer(text: str) -> str:
    match = THINK_BLOCK_RE.match(text)
    if not match:
        raise ValueError("format invalid: missing complete <think>...</think> block")
    return match.group(1).strip()

def check_math_answer(text: str, expected: str) -> int:
    final_answer = extract_final_answer(text)
    return 1 if final_answer == expected.strip() else 0

def reward_rule(full_output: str, expected: str) -> int:
    reward_format = check_format(full_output)
    reward_acc = check_math_answer(full_output, expected) if reward_format else 0
    return reward_acc + reward_format

if __name__ == "__main__":
    sample = "<think>12*10=120, 12*3=36, total=156</think>\n156"
    assert check_format(sample) == 1
    assert extract_final_answer(sample) == "156"
    assert check_math_answer(sample, "156") == 1
    assert reward_rule(sample, "156") == 2
    print("verifier ok")
```

这个例子虽然很小，但已经覆盖了 R1 训练最核心的边界条件：**必须存在可重复执行、可自动评分、误差较小的判题器**。

---

## 核心机制与推导

R1 的 RL 不是“每题只生成一个答案，然后按这一个答案更新”。更关键的做法是：**对同一道题采样一组候选输出，再做组内相对比较**。这里用到的算法叫 GRPO，`Group Relative Policy Optimization`，可以直白理解成“分组相对比较的策略优化”。

为什么要这样做？因为如果每题只采样一条输出，奖励信号太稀疏，更新方向很容易抖动。反过来，如果一题采样 8 条、16 条甚至更多候选，就能直接看到：**同一道题上，哪些写法更容易通过验证器，哪些写法更容易失败**。这比单条样本的绝对打分更稳定。

先看一个最小数值例子。某题采样 4 条答案，奖励如下：

| 候选 | 是否答对 | 格式是否合规 | 总奖励 |
|---|---|---|---|
| 解 1 | 1 | 1 | 2 |
| 解 2 | 0 | 1 | 1 |
| 解 3 | 1 | 1 | 2 |
| 解 4 | 0 | 0 | 0 |

组平均奖励是：

$$
\bar{r}=\frac{2+1+2+0}{4}=1.25
$$

于是每条样本的组内相对优势可以写成：

$$
A_i = r_i-\bar{r}
$$

代入后得到：

- `A_1 = 2 - 1.25 = 0.75`
- `A_2 = 1 - 1.25 = -0.25`
- `A_3 = 2 - 1.25 = 0.75`
- `A_4 = 0 - 1.25 = -1.25`

这意味着：

- 解 1、解 3 高于组内平均水平，应该提高概率。
- 解 2、解 4 低于组内平均水平，应该降低概率。
- 解 4 最差，因此应被压得最明显。

真实实现里，常见做法还会做组内标准化，减少不同题目之间奖励尺度不一致的问题：

$$
\hat{A}_i=\frac{r_i-\bar{r}}{\sigma_r+\epsilon}
$$

这里：

- `\bar{r}` 是组内平均奖励
- `\sigma_r` 是组内奖励标准差
- `\epsilon` 是防止除零的小常数

这样做的直觉是：同样是“高于平均水平 0.5”，如果整组奖励都很接近，那么这 `0.5` 就很重要；如果整组奖励分布本来就很分散，那么这 `0.5` 的意义就相对小一些。

R1 论文中的核心约束还包括参考策略的 KL 项。可以把它写成更接近直觉的形式：

$$
Objective \approx \mathbb{E}\left[\hat{A}_i \cdot \log \pi_\theta(y_i|x)\right]-\beta \cdot KL(\pi_\theta \| \pi_{ref})
$$

新手只要记住三件事就够了：

- `\hat{A}_i`：这条样本比同组平均水平好多少
- `\pi_\theta`：当前正在训练的策略
- `KL(\pi_\theta \| \pi_{ref})`：当前策略偏离参考策略有多远

为什么要加 KL 约束？因为如果只追求奖励，模型很容易走向极端，例如：

- 机械重复某种容易过格式检查的模板
- 学会钻判题器的漏洞
- 输出越来越长、越来越怪，只为换一点奖励
- 语言风格和可读性迅速恶化

这就是常说的 reward hacking，也就是“钻评分规则空子”。参考策略的 KL 惩罚相当于刹车：模型可以偏离原分布，但不能一下冲太远。

对新手来说，这一轮训练可以理解成：

1. 同一题，模型先写出一组不同版本。
2. 验证器给每个版本打分。
3. 系统比较“谁比同组平均更好”。
4. 训练提高高分轨迹的概率，压低低分轨迹的概率。
5. 同时用 KL 约束防止策略漂移过快。

长期重复后，模型会逐步偏向一些更稳定的推理模式，例如：

- 先拆分大问题
- 在中间步骤做自检
- 遇到异常结果时回退重算
- 在最终回答前再核对一次

这也是论文里提到的“反思、自我验证、动态调整策略”之类现象的来源。它们不是人工逐条硬编码进去的，而是从“高奖励轨迹更容易存活”这一机制里逐渐涌现出来的。

下面给一个可运行的 toy GRPO 代码，保留“多采样、组内比较、KL 惩罚”的骨架：

```python
from math import exp, sqrt

def verify_math(candidate):
    reward_acc = 1 if candidate["answer"] == "156" else 0
    reward_format = 1 if candidate["trace"].startswith("<think>") and candidate["trace"].endswith("</think>") else 0
    return reward_acc + reward_format

def mean(xs):
    return sum(xs) / len(xs)

def std(xs):
    mu = mean(xs)
    return sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))

def grpo_step(candidates, ref_logps, beta=0.1, eps=1e-6):
    rewards = [verify_math(c) for c in candidates]
    mu = mean(rewards)
    sigma = std(rewards)
    normalized_advantages = []
    for reward, ref_lp, cur_lp in zip(rewards, ref_logps, [c["logp"] for c in candidates]):
        advantage = (reward - mu) / (sigma + eps)
        kl_penalty = beta * (cur_lp - ref_lp)
        normalized_advantages.append(advantage - kl_penalty)
    return rewards, normalized_advantages

if __name__ == "__main__":
    candidates = [
        {"trace": "<think>12*10+12*3=156</think>", "answer": "156", "logp": -0.20},
        {"trace": "<think>12*13=154</think>", "answer": "154", "logp": -0.10},
        {"trace": "<think>12*(10+3)=156</think>", "answer": "156", "logp": -0.25},
        {"trace": "12*13=156", "answer": "156", "logp": -0.30},
    ]
    rewards, advantages = grpo_step(candidates, ref_logps=[-0.22, -0.12, -0.24, -0.28])

    assert rewards == [2, 1, 2, 1]
    assert len(advantages) == 4
    assert advantages[0] > advantages[1]
    assert advantages[2] > advantages[3]
    print("rewards =", rewards)
    print("advantages =", [round(x, 4) for x in advantages])
```

这个 toy 代码虽然不是完整训练器，但已经表达了 R1 推理 RL 的三个核心点：

1. 判题器只看可验证结果和格式，不看解释是否“像人类老师”。
2. 优势是组内相对量，不是单条样本的孤立绝对分数。
3. 策略更新时要加 KL 约束，否则很容易偏到奇怪的高奖励角落。

---

## 代码实现

从工程角度看，R1 更接近一条多阶段流水线，而不是“拿 base model 直接做纯 RL 一把梭”。如果把 R1-Zero 和最终 R1 放在一起看，更准确的理解是：

- 在推理能力增长上，**规则化 RL 是核心驱动力**
- 在可读性、语言一致性、通用对话能力上，**仍然需要后续 SFT 和混合奖励来补齐**

后续阶段的总体奖励可以概括成：

$$
Reward_{total}=Reward_{reasoning}+Reward_{general}+Reward_{language}
$$

其中：

- `Reward_reasoning`：推理正确性奖励，数学和代码能否过判题器
- `Reward_general`：通用对齐奖励，回答是否符合任务意图、是否有帮助
- `Reward_language`：语言质量奖励，输出是否清晰、是否语言一致、是否易读

一个简化的训练阶段表如下：

| 阶段 | 目标 | 主要数据 | 奖励/损失 | 作用 |
|---|---|---|---|---|
| Cold Start SFT | 建立基本输出格式和可读性 | 少量高质量推理样本 | SFT loss | 先把回答形状校正好 |
| RL Stage 1 | 强化可验证推理能力 | 数学、代码、逻辑题 | 规则奖励 | R1 系列最核心的能力增长阶段 |
| Rejection Sampling + SFT | 把高质量轨迹沉淀成数据 | RL 产出的优质样本 | SFT loss | 把 RL 的成果“压实”成稳定习惯 |
| RL Stage 2 | 泛化与通用对齐 | 推理数据 + 通用指令数据 | 混合奖励 | 补语言质量、帮助性和安全性 |

如果只看 R1-Zero，很多人会误以为“纯 RL 已经够了”。其实论文自己也明确给出了边界：**R1-Zero 的推理能力很强，但可读性差、语言混杂明显，对开放式通用任务提升有限**。这正是为什么后面还要继续做多阶段对齐。

下面给一个更完整、仍然可运行的 toy pipeline。它不训练大模型，但能把“同题多采样、判题、组内标准化、筛选高质量轨迹”这条工程链路串起来：

```python
from dataclasses import dataclass
from math import sqrt

@dataclass
class Candidate:
    trace: str
    answer: str
    logp: float

def verify(candidate: Candidate, expected: str) -> int:
    reward_acc = 1 if candidate.answer == expected else 0
    reward_format = 1 if candidate.trace.startswith("<think>") and candidate.trace.endswith("</think>") else 0
    return reward_acc + reward_format

def group_advantages(candidates, expected, ref_logps, beta=0.05, eps=1e-6):
    rewards = [verify(c, expected) for c in candidates]
    mu = sum(rewards) / len(rewards)
    sigma = sqrt(sum((r - mu) ** 2 for r in rewards) / len(rewards))
    advantages = []
    for cand, reward, ref_lp in zip(candidates, rewards, ref_logps):
        adv = (reward - mu) / (sigma + eps)
        kl_penalty = beta * (cand.logp - ref_lp)
        advantages.append(adv - kl_penalty)
    return rewards, advantages

def rejection_sampling(candidates, rewards, min_reward=2):
    return [c for c, r in zip(candidates, rewards) if r >= min_reward]

if __name__ == "__main__":
    expected = "156"
    candidates = [
        Candidate("<think>12*10=120; 12*3=36; 120+36=156</think>", "156", -0.20),
        Candidate("<think>12*13=154</think>", "154", -0.10),
        Candidate("<think>12*(10+3)=156</think>", "156", -0.25),
        Candidate("12*13=156", "156", -0.30),
    ]
    ref_logps = [-0.22, -0.11, -0.23, -0.29]

    rewards, advantages = group_advantages(candidates, expected, ref_logps)
    winners = rejection_sampling(candidates, rewards, min_reward=2)

    assert rewards == [2, 1, 2, 1]
    assert len(advantages) == 4
    assert len(winners) == 2
    assert winners[0].answer == "156"
    assert winners[1].answer == "156"

    print("rewards =", rewards)
    print("advantages =", [round(x, 4) for x in advantages])
    print("kept =", len(winners))
```

这个示例对应真实工程中的三步：

1. 同题采样多条候选轨迹
2. 用 verifier 打规则奖励
3. 保留高奖励轨迹，继续用于更新或回灌 SFT

真实场景可以换成代码生成任务。比如 prompt 是“实现 `two_sum(nums, target)`”。系统可以：

1. 对同一 prompt 采样 16 个程序
2. 运行单元测试
3. 给通过测试且格式合规的样本更高奖励
4. 用 GRPO 做组内相对优化
5. 再把最好的代码样本收集起来，做 rejection sampling + SFT

这种做法的实际意义很大：你不必为海量题目手写“标准思路版代码示范”，而是可以直接复用自动测试体系。

---

## 工程权衡与常见坑

纯规则化 RL 很强，但它并不天然优雅。最常见的问题不是“训不出分数”，而是“分数上来了，输出却越来越难用”。

第一个坑是可读性差。  
如果奖励只关心“能不能过验证器”，模型不会天然关心自己的解释是否清楚。它可能学会一套对判题器友好、但对人类不友好的表达方式。R1-Zero 的“可读性差”就是这个问题的典型例子。

第二个坑是语言混杂。  
如果底座本身是多语种，纯 RL 又没有语言一致性约束，模型很可能把中文说明、英文模板、符号片段混在一起。只要最终过判题器，这种输出就可能继续被保留。

第三个坑是 reward hacking。  
reward hacking 的意思不是模型更聪明了，而是它学会了“利用打分规则的漏洞”。例如格式检查太弱时，模型可能机械地插入 `<think>` 标签，却没有形成真正稳定的推理结构。

第四个坑是底座能力不够。  
R1 这类方法不是从很弱的 base model 上平地起高楼。如果预训练底座的数学、代码、语言能力本来就不够，RL 往往只能在低水平区域打转，很难出现真正的能力跃迁。

第五个坑是判题器本身有漏洞。  
这比很多新手想象得更常见。比如：

- 数学答案只按字符串完全匹配，导致等价表达被错判
- 代码测试覆盖不全，模型学会“只过公开测试”
- 格式检查只看标签存在，不看最终答案区是否冲突

常见工程权衡可以整理成表：

| 挑战 | 典型表现 | 常见缓解措施 |
|---|---|---|
| 可读性差 | 推理冗长、混乱、用户难读 | 先做 cold-start SFT，后续加语言奖励 |
| 语言混杂 | 中英夹杂、模板污染 | 增加语言一致性奖励，过滤差样本 |
| Reward hacking | 钻格式或测试漏洞 | 强化 verifier，补边界测试 |
| 训练发散 | 输出突然崩坏，KL 暴涨 | 提高 KL 约束，缩小更新步长 |
| 底座太弱 | RL 收益很小 | 换更强 base model，先补基础能力 |
| 判题器误判 | 好样本被错杀，坏样本漏过 | 改进等价检查与测试覆盖 |

格式奖励本身也要设计清楚。最简单的形式是：

$$
Reward_{format}=
\begin{cases}
1, & \text{存在合法的 } <think>...</think> \text{ 且有最终答案} \\
0, & \text{否则}
\end{cases}
$$

但真实工程通常还会检查更多约束：

- 是否先输出推理区，再输出最终答案
- 标签是否闭合
- 是否存在多个互相冲突的最终答案
- 是否超长到影响吞吐
- 是否在目标语言内完成主要输出

下面给一个更严格、可运行的格式检查器：

```python
import re

THINK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*(.+?)\s*$", re.S)

def format_checker(text: str) -> int:
    match = THINK_RE.match(text)
    if not match:
        return 0

    reasoning, final_answer = match.groups()
    if not reasoning.strip():
        return 0
    if not final_answer.strip():
        return 0
    if "<think>" in final_answer or "</think>" in final_answer:
        return 0
    return 1

if __name__ == "__main__":
    assert format_checker("<think>reasoning</think>\n156") == 1
    assert format_checker("<think></think>\n156") == 0
    assert format_checker("<think>reasoning\n156") == 0
    assert format_checker("156") == 0
    print("format checker ok")
```

对新手来说，最容易误解的一点是：R1 不是“完全不需要 SFT”。更准确的说法是：

- **推理能力增长的核心动力来自规则化 RL**
- **但工程上通常仍然需要 SFT 来修正输出格式、语言质量和通用对话能力**

否则，模型虽然可能更会做题，但未必更适合给人用。

---

## 替代方案与适用边界

如果任务天然可验证，纯规则化 RL 是非常强的路线，因为它把“人写示范”的成本，换成了“系统自动判题”的成本。数学、代码、符号推导、部分定理证明，这些任务都可能从中显著受益。

但如果任务不可验证，就必须使用替代方案。最常见的是“偏好模型 + SFT”或更一般的 RLHF / RLAIF 路线。这里的偏好模型可以理解为“学习人更喜欢哪个回答的打分器”，它不要求存在唯一正确答案，但需要人工偏好数据或 AI 偏好数据。

三类路线可以直接比较：

| 路线 | 适合场景 | 优点 | 局限 |
|---|---|---|---|
| 纯规则化 RL | 数学、代码、可验证推理 | 奖励明确，可大规模自动化 | 任务边界窄 |
| SFT + Reward Model | 开放问答、写作、对话 | 适用面广 | 人工数据成本高，奖励主观 |
| 混合路线 | 既要推理又要通用能力 | 兼顾可验证性与可用性 | 系统复杂度更高 |

因此，R1 更像一种“在特定任务上极强”的训练范式，而不是所有大模型训练路线的替代品。比如写诗、写广告、做情绪陪伴，这些任务很难稳定定义 `Reward_acc`，就不能直接照搬 R1 的纯规则化方案。

在非纯推理任务里，奖励往往必须组合：

$$
Reward_{total}=Reward_{reasoning}+Reward_{general}+Reward_{language}
$$

这说明当任务从“算对”转向“说得好”时，系统就必须引入更复杂的评价器，而不再能只依赖 deterministic verifier。

下面是一个可运行的混合奖励示例：

```python
def hybrid_reward(task_type, answer_ok, tests_ok, preference_score, language_score):
    if task_type in {"math", "code"}:
        return int(answer_ok) + int(tests_ok)
    return 0.6 * preference_score + 0.4 * language_score

if __name__ == "__main__":
    assert hybrid_reward("math", True, True, 0.0, 0.0) == 2
    score = hybrid_reward("writing", False, False, 0.8, 0.9)
    assert abs(score - 0.84) < 1e-9
    print("hybrid reward ok")
```

这个例子表达的边界很明确：

- 数学题、代码题：优先走规则奖励
- 开放式问答、创作任务：必须更多依赖偏好分和语言质量分
- 混合任务：往往要把多类奖励拼起来

所以，判断一个任务是否适合 R1 式训练，最直接的问题不是“它重不重要”，而是：

**它能不能被自动、稳定、低歧义地判分？**

只要这个条件成立，纯规则化 RL 就有机会放大；只要这个条件不成立，就必须回到更依赖人工信号或模型评价器的路线。

---

## 参考资料

1. DeepSeek 团队，*DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning*，Nature，2025。  
   https://www.nature.com/articles/s41586-025-09422-z  
   用途：本文关于 `Reward_rule = Reward_acc + Reward_format`、R1-Zero、语言混杂、两阶段 RL 与最终 R1 训练流程的核心来源。

2. DeepSeek 团队，*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*。  
   https://arxiv.org/abs/2501.12948  
   用途：补充 Nature 论文对应的公开技术版本，便于查训练细节、实验表格和方法描述。

3. DeepSeekMath 团队，*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*，2024。  
   https://arxiv.org/abs/2402.03300  
   用途：GRPO 的直接来源。本文关于“同题多采样、组内相对比较、KL 约束”的方法背景，主要参考这篇论文。

4. DeepSeek-AI 官方仓库，*DeepSeek-Math*。  
   https://github.com/deepseek-ai/DeepSeek-Math  
   用途：核对 GRPO 的公开说明、数学推理 RL 的任务设置，以及开源实现层面的补充信息。

5. Schulman 等，*Proximal Policy Optimization Algorithms*，2017。  
   https://arxiv.org/abs/1707.06347  
   用途：理解 GRPO 背后的 PPO 直觉，尤其是策略更新、相对优势和稳定优化约束。
