## 核心结论

自一致性解码（Self-Consistency）是一种测试时推理策略。白话说法是：不是让模型只想一次，而是让它在同一个问题上独立“想多次”，最后对答案投票。它主要用于数学题、符号推理题、步骤明确的问答题，因为这类任务的最终答案通常可以规范化成同一种形式，比如一个数字、一个选项或一个公式结果。

它的核心规则很简单。设模型采样出 $m$ 条推理路径，每条路径得到一个最终答案 $a_i$，那么最终输出是：

$$
a^*=\arg\max_a |\{a_i=a\}|
$$

意思是：哪个答案出现次数最多，就选哪个。

在数学推理上，这个方法的收益很明确。以 GSM8K 为例，单条 Chain-of-Thought（CoT，链式思维，指模型先写中间推理步骤再给答案）大约在 56% 左右，而 Self-Consistency 可提升到约 74%。提升不是因为模型“更懂数学”，而是因为多次采样把偶然走错的一条路径平均掉了。

一个最直观的玩具例子如下。对同一道小学应用题采样 5 次，模型返回答案分别是：

| 路径 | 最终答案 |
|---|---:|
| 1 | 18 |
| 2 | 18 |
| 3 | 20 |
| 4 | 18 |
| 5 | 20 |

这里 `18` 有 3 票，`20` 有 2 票，所以最终输出 `18`。即使个别路径算错，只要错误不是大多数，投票仍然能纠正。

经验上，采样数增加带来的收益呈“前快后慢”的对数递减趋势。前几条路径最值钱，后面会逐渐进入平台期。

| 采样路径数 | GSM8K 准确率 |
|---|---:|
| 1 | 56.5% |
| 5 | 68.2% |
| 10 | 72.1% |
| 20 | 74.0% |
| 40 | 74.4% |

可以把流程理解成一张简图：

`同一 prompt -> 温度采样多条 CoT -> 提取最终答案 -> 投票/加权投票 -> 最终输出`

---

## 问题定义与边界

问题定义很具体：当模型需要经过多步中间推理才能得到答案时，单次生成很容易在某一步偏航，后续即使语法通顺，最终结果也可能完全错误。数学推理就是典型场景，因为错误一旦进入中间步骤，后面往往会连锁传播。

Self-Consistency 解决的不是“模型不知道答案”，而是“模型有时会走错推理路径”。因此它假设一件事：同一个问题存在多条可行推理链，其中正确链被模型采到的概率并不低，只是单次生成不稳定。

它更适合以下任务：

| 适用场景 | 原因 |
|---|---|
| 数学题、算术题 | 最终答案通常是可比较的数字 |
| 逻辑推理、符号推导 | 可以把不同路径压缩到同一个结论 |
| 多选题、分类题 | 最终输出空间有限，投票自然有效 |

它不太适合以下任务：

| 不适用场景 | 原因 |
|---|---|
| 开放式写作 | 不存在唯一正确答案，投票意义弱 |
| 创意生成 | 多样性本身是目标，不应强行收敛 |
| 多模态复杂任务 | 路径差异可能来自感知误差，不只是推理误差 |

新手版边界可以这样理解：如果题目的最后能落到“到底是 17 还是 19”这种可判定答案，自一致性通常有用；如果任务是“写一段更有感染力的文案”，那就没有稳定的“多数正确答案”。

---

## 核心机制与推导

Self-Consistency 一般分两步。

第一步是采样。这里的采样不是每次都按最高概率 token 走，而是用温度参数 $T$ 做随机采样。温度可以理解成“放大探索”的旋钮，常见设置如 $T=0.7$。温度太低，路径都长得像 greedy decoding（贪心解码，指每一步都取当前概率最大的 token）；温度太高，路径会发散，噪声变多。数学推理里，中等温度常用于在“多样性”和“稳定性”之间找平衡。

第二步是聚合。最基础的是等权投票，也就是只看答案出现次数，不看这条路径本身有多可信。

进一步可以做加权自一致性。直观说法是：每条推理路径都有一个“模型自己给出的置信痕迹”，通常来自 token 的对数概率（log-prob，对数概率，多个 token 的概率相乘时通常改为相加，数值更稳定）。若一条路径虽然票数不多，但整体生成概率明显更高，它不应和低概率乱猜路径完全等价。

常见写法是：

$$
w(a)=\sum_{\text{path}:a_i=a}\log p(\text{path})
$$

最终选择：

$$
a^*=\arg\max_a w(a)
$$

这里要注意，这个公式表达的是“把所有通向同一答案的路径权重合并”。在工程实现里，也常见把路径概率先转成分数再累加，或者对长度做归一化，避免长链天然吃亏。

看一个玩具例子。假设采样出 4 条路径：

| 路径 | 最终答案 | $\log p(\text{path})$ |
|---|---:|---:|
| 1 | 42 | -2.0 |
| 2 | 42 | -2.4 |
| 3 | 39 | -0.8 |
| 4 | 39 | -0.9 |

等权投票下，`42` 和 `39` 都是 2 票，打平。  
加权后：

- $w(42)=-2.0+(-2.4)=-4.4$
- $w(39)=-0.8+(-0.9)=-1.7$

因为对数概率越大越好，所以 `39` 获胜。白话解释是：虽然两边人数一样，但支持 `39` 的路径明显更“像模型真心相信的输出”。

这也是为什么加权投票常在小样本时更有价值。只采 3 到 5 条路径时，单纯多数票容易受偶然性影响；而加权可以更快偏向高置信答案。

真实工程例子是批量数学评测或金融问答。假设你在 Amazon Bedrock 上对数千道题做离线评分。若每题只跑一次 greedy，速度快，但错一条就直接错。若改成每题采样 3 条或 5 条路径，再做投票，延迟和 token 成本线性上升，但整体准确率通常明显改善，尤其适合“答错代价高、吞吐量又还能接受”的离线任务。

---

## 代码实现

下面给出一个可运行的 Python 示例。它不依赖具体模型 API，而是演示“采样结果 -> 提取答案 -> 投票/加权投票”的核心逻辑。

```python
import re
from collections import Counter, defaultdict

def extract_final_answer(text: str) -> str:
    # 提取最后一个数字，真实工程里可换成更严格的解析器
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        raise ValueError(f"no answer found: {text}")
    return nums[-1]

def majority_vote(samples):
    answers = [extract_final_answer(s["text"]) for s in samples]
    counts = Counter(answers)
    return counts.most_common(1)[0][0]

def weighted_vote(samples):
    scores = defaultdict(float)
    for s in samples:
        ans = extract_final_answer(s["text"])
        scores[ans] += s["logprob"]  # 对数概率越大越好，通常是接近 0 的负数
    return max(scores.items(), key=lambda x: x[1])[0]

# 玩具样本：同一道题采样 5 次
samples = [
    {"text": "先算单价，再乘数量。最终答案：18", "logprob": -1.8},
    {"text": "分步计算后得到 18。最终答案：18", "logprob": -2.1},
    {"text": "中间一步算错了。最终答案：20", "logprob": -3.5},
    {"text": "换一种方法验证。最终答案：18", "logprob": -1.9},
    {"text": "另一条路径。最终答案：20", "logprob": -2.7},
]

assert extract_final_answer("最终答案：18") == "18"
assert majority_vote(samples) == "18"
assert weighted_vote(samples) == "18"

# 加权比等权更敏感的例子
samples_tie = [
    {"text": "最终答案：42", "logprob": -2.0},
    {"text": "最终答案：42", "logprob": -2.4},
    {"text": "最终答案：39", "logprob": -0.8},
    {"text": "最终答案：39", "logprob": -0.9},
]

assert majority_vote(samples_tie) in {"42", "39"}  # 等权打平，取决于实现细节
assert weighted_vote(samples_tie) == "39"
```

如果接入真实大模型，流程通常是：

1. 同一个 prompt 重复调用 $m$ 次。
2. 打开温度采样，如 `temperature=0.7`。
3. 要求模型在末尾输出标准格式答案，如 `Final Answer: 18`。
4. 对结果做规范化提取。
5. 执行多数投票或加权投票。
6. 返回最终答案，而不是任意一条原始路径。

一个常见伪代码如下：

```python
for _ in range(m):
    path = model.generate(prompt, temperature=0.7, top_p=1.0)
    answer = parse(path)
    save(answer, path_logprob)

final = vote(saved_answers)
```

关键点不在 prompt 花样，而在“重复采样 + 稳定解析 + 聚合规则”。

---

## 工程权衡与常见坑

Self-Consistency 的最大代价是线性成本。采样 5 条路径，几乎就是 5 次生成成本；采样 20 条路径，延迟和 token 消耗也会明显上涨。因此它本质上是在用测试时计算量换精度。

下面是一个简化的权衡表：

| 采样数 | 相对 token 成本 | 准确率提升特点 |
|---|---:|---|
| 1 | 1x | 基线，最便宜 |
| 5 | 5x | 提升最大，通常最划算 |
| 10 | 10x | 仍有明显收益 |
| 20 | 20x | 接近平台期 |
| 40 | 40x | 收益很小，更多用于高精度离线任务 |

几个常见坑：

- `采样太少`
  2 条或 3 条路径时，投票稳定性仍然有限，尤其在题目难度高时。
- `答案提取不规范`
  两条路径都算出 18，但一条写成 `18 元`，另一条写成 `十八`，若解析不统一，会被误判成不同答案。
- `把 beam search 当替代品`
  beam search（束搜索，指每一步保留若干高概率前缀）更像“在一棵概率树上保留若干最可能的续写”，不是主动制造多样解法。它保留的是相似高概率前缀，不等于独立思路，因此在数学推理上往往不如 sampling。
- `直接比较路径 log-prob，不做长度处理`
  长路径天然累计更多负数，对数概率会更低。若任务中路径长度波动大，最好考虑长度归一化。
- `忽视延迟预算`
  在线交互系统对 500ms 和 5s 的感受完全不同。自一致性更适合高价值问答、批处理评测、离线生成，而不是所有前台请求都默认打开。

为什么 beam search 不能替代 Self-Consistency，可以用一句话概括：  
Self-Consistency 依赖“多样化采样后再聚合”，beam search 依赖“按局部概率保留少数候选”。前者追求不同思路，后者追求同一高概率区域内的最优序列，两者目标不同。

---

## 替代方案与适用边界

如果目标只是最快返回一个还不错的答案，single greedy 最省钱。如果目标是数学正确率，Self-Consistency 往往更值得。如果还需要中间步骤可控、能结合外部工具，那么 ReAct 或 Tree-of-Thought 之类方法更合适。

| 方案 | 优势 | 局限 | 适用边界 |
|---|---|---|---|
| Self-Consistency | 数学推理精度高，易加到现有系统 | 成本线性上升 | 有标准答案、可投票任务 |
| Beam Search | 生成稳定，可控 | 候选缺乏真正多样性 | 确定性续写、翻译、结构化生成 |
| Single Greedy | 最快最便宜 | 易受单路径错误影响 | 草稿生成、低成本场景 |
| ReAct / ToT | 可结合工具、可做步骤反馈 | 实现复杂，延迟更高 | 需要搜索、调用工具、复杂规划 |

给新手的场景对照可以写得更直接：

| 场景 | 更合适的方法 |
|---|---|
| 写一个快速草稿 | Greedy |
| 做一批数学题评测 | Self-Consistency |
| 生成格式严格的固定文本 | Beam Search |
| 需要查资料再推理 | ReAct |

边界也要说清楚。Self-Consistency 不是万能补丁。若模型本身不会做这类题，多采样只是重复犯错；若预算极低，1 条路径改成 5 条路径的收益可能不值成本；若任务是开放式创作，投票会把多样性错误地压扁。

---

## 参考资料

- [Self-Consistency Improves Chain of Thought Reasoning in Language Models（OpenReview）](https://openreview.net/pdf/9d06013867701125040af03996c3aefddc8d58d1.pdf)
  原始论文，给出 Self-Consistency 的定义、实验设置和 GSM8K 等基准结果。

- [arXiv.gg 论文页：Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.gg/paper/2203.11171)
  适合快速浏览论文主题、任务范围和核心思路。

- [Enrico Piovano: Test-Time Compute Scaling](https://enricopiovano.com/blog/test-time-compute-scaling/)
  对 Self-Consistency、sample 数与收益曲线、log-prob 加权投票有比较清晰的工程解释。

- [Preprints 文档：关于 sample 数与精度关系的整理](https://www.preprints.org/frontend/manuscript/63b544e58c3ec05bb1fc9ab3962afc12/download_pub)
  可用于查看“1、5、10、20、40 条路径”对应的精度趋势，说明收益递减。

- [AWS Machine Learning Blog / Bedrock 实战案例转载](https://phdstudio.org/2024/03/19/enhance-performance-of-generative-language-models-with-self-consistency-prompting-on-amazon-bedrock-lucia-santamaria-aws-machine-learning-blog/)
  重点在部署视角，讨论批量任务里延迟、成本和准确率如何取舍。

- [Your AI Staff: Self-Consistency Improves Chain-of-Thought](https://your-ai-staff.com/self-consistency-improves-chain-of-thought/)
  适合补充理解 beam search 与 self-consistency 的区别，以及常见误用。
