## 核心结论

解码策略是语言模型在“已经算出下一个 token 概率分布之后，究竟怎么选”的规则。token 可以理解为模型内部处理的最小文本片段，可能是一个字、一个词，或词的一部分。

结论先说清楚：

| 策略 | 决策方式 | 确定性 | 典型目标 | 主要问题 |
|---|---|---:|---|---|
| 贪心解码 Greedy | 每步直接选概率最大 token | 高 | 最低延迟、分类式输出 | 容易落入局部最优 |
| Beam Search | 同时保留 $k$ 条最优路径 | 高 | 翻译、摘要、结构化生成 | 计算成本高，容易重复 |
| Top-k 采样 | 只在前 $k$ 个 token 中随机采样 | 中 | 开放式生成 | $k$ 固定，不适应分布变化 |
| Top-p 采样 | 取累计概率达到 $p$ 的最小集合再采样 | 中 | 对话、创作、通用生成 | 对参数敏感，行为波动更大 |
| Temperature | 先缩放 logits 再交给其他策略 | 取决于主策略 | 调整随机性强弱 | 与 Top-p/Top-k 组合时不稳定 |

这里的 logits 是模型输出但还没归一化的分数，可以把它理解为“每个 token 的原始倾向值”。

对零基础读者，最重要的判断标准只有一条：任务到底要“稳定复现一个最优答案”，还是要“在合理范围内保留创造性”。

玩具例子：如果下一个词分布是 `the:0.50, a:0.20, cat:0.15, dog:0.10, eats:0.05`，贪心一定选 `the`；Beam 会保留多条候选前缀；Top-k 可能在 `{the, a, cat}` 里抽一个；Top-p 若设 $p=0.8$，则只在 `{the, a, cat}` 中抽样，因为前两个是 0.7，还不够 0.8，加上 `cat` 后变成 0.85。

真实工程例子：多语种机器翻译通常优先 Beam Search，因为目标是准确、稳定、可复现；聊天机器人或写作助手更常用 Top-p 加 Temperature，因为目标不是唯一正确答案，而是“合理且不呆板”。

---

## 问题定义与边界

解码问题可以写成：给定上下文 $x_{<t}$，模型产生条件分布 $P(x_t \mid x_{<t})$，系统要从这个分布中选出下一个 token，并不断重复，直到生成结束。

如果追求整句概率最大，目标形式通常写成：

$$
x_{1:T}^* = \arg\max_{x_{1:T}} \sum_{t=1}^{T} \log P(x_t \mid x_{<t})
$$

这里用对数概率 $\log P$，因为多个小概率连乘会非常小，而取对数后变成求和，数值更稳定。

边界主要有两个：

1. 是否允许探索低概率 token。
2. 是否同时维护多条候选路径。

这两个边界基本决定了所有常见策略：

| 维度 | 贪心 | Beam | Top-k | Top-p |
|---|---|---|---|---|
| 是否保留多路径 | 否 | 是 | 否 | 否 |
| 是否包含随机性 | 否 | 否 | 是 | 是 |
| 是否允许动态候选集 | 否 | 否 | 否 | 是 |

一个常见误解是“概率最大就一定最好”。这只对单步成立，不对整句成立。因为语言生成是链式决策，前面一步的选择会改变后面所有步的条件分布。局部最优不等于全局最优，这正是 Beam Search 出现的原因。

再看一个玩具例子。假设模型在第一步有两个候选：

- 路径 A：`machine`，概率 0.51
- 路径 B：`deep`，概率 0.49

如果贪心选了 `machine`，第二步也许只能走向普通短语；而 `deep` 后面却可能接出高质量高概率的 `learning system`。此时整句总概率可能反而更高。Beam 的价值就是不在第一步过早丢掉 B。

但这类方法也有明确边界。若任务是代码补全、翻译、信息抽取，通常更重视一致性；若任务是故事续写、营销文案、聊天回复，则需要保留随机探索，否则文本会非常模板化。

---

## 核心机制与推导

先看 Temperature。它不直接决定选谁，而是先改写分布形状：

$$
P(x_i)=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
$$

其中 $z_i$ 是第 $i$ 个 token 的 logit，$T$ 是温度。

- $T<1$：高分 token 被进一步放大，分布更尖，结果更确定。
- $T=1$：不改原分布。
- $T>1$：分布被拉平，低概率 token 更容易被抽到。

这就是“温度控制随机性”的数学原因。它不是凭空增加创意，而是在重排概率分布的陡峭程度。

Top-k 的定义最简单。设词表为 $V$，取概率最高的 $k$ 个 token 组成集合 $V_k$，再在该集合上重归一化：

$$
P_k(x_i)=
\begin{cases}
\frac{P(x_i)}{\sum_{x_j \in V_k} P(x_j)}, & x_i \in V_k \\
0, & x_i \notin V_k
\end{cases}
$$

问题在于 $k$ 是固定的。固定，意思是不管当前分布很尖还是很平，它都只保留同样数量的候选。这会导致两个方向的错误：

- 分布极尖时，$k$ 可能过大，引入噪声。
- 分布很平时，$k$ 可能过小，截断掉有意义候选。

Top-p 也叫 Nucleus Sampling，核心思想是“不要固定数量，固定概率质量”。定义为最小集合 $V^{(p)}$，满足：

$$
\sum_{x \in V^{(p)}} P(x \mid context) \ge p
$$

并且这个集合中的 token 按概率从高到低累计后，刚刚达到阈值 $p$。

这就是为什么说 Top-p 是“动态的 Top-k”。当分布很尖时，达到 $p$ 只需要少数 token；当分布较平时，需要更多 token。

用前面的玩具例子说明：

| token | 原概率 |
|---|---:|
| the | 0.50 |
| a | 0.20 |
| cat | 0.15 |
| dog | 0.10 |
| eats | 0.05 |

若 `top_k=3`，候选固定为 `{the, a, cat}`。  
若 `top_p=0.8`，累计过程是：`the=0.50`，`the+a=0.70`，`the+a+cat=0.85`，所以 nucleus 也是 `{the, a, cat}`。  
但如果把分布改成 `the:0.75, a:0.10, cat:0.08, ...`，那么同样 `top_p=0.8` 只需要 `{the, a}`，而 `top_k=3` 依然会把 `cat` 加进来。

Beam Search 的机制不同。它不是采样，而是保留 $k$ 条前缀路径，每一轮扩展后只留下累计分数最高的 $k$ 条。路径分数通常写作：

$$
score(y_{1:t})=\sum_{i=1}^{t}\log P(y_i \mid y_{<i}, x)
$$

这里 $y_{1:t}$ 是当前生成前缀。由于每多生成一个 token 都会再乘一个小于 1 的概率，序列越长，累计对数概率越负，所以 Beam 往往偏向短句。工程上常加长度惩罚：

$$
score_{norm}(y)=\frac{\sum_{i=1}^{|y|}\log P(y_i \mid y_{<i}, x)}{|y|^\alpha}
$$

其中 $\alpha$ 是长度惩罚系数。它的作用是避免“短句天然占优”。

真实工程例子：机器翻译里，如果不用长度惩罚，Beam 可能输出过短句子，因为短句路径累计损失更少；如果长度惩罚过强，又会鼓励冗长翻译，甚至出现重复短语。

另一个常见问题是 Temperature 与 Top-p 联合使用。因为 Temperature 先改变分布陡峭程度，而 Top-p 再按累计概率切集合，所以 $T$ 的一点变化可能让 nucleus 集合突然扩张或收缩。这就是很多线上服务里“只开一个主随机参数”的原因：排障更容易，行为更稳定。

---

## 代码实现

下面给出一个可运行的简化 Python 实现。它不依赖深度学习框架，只演示“给定 logits，如何按不同策略选 token”。其中 `assert` 用来验证基本行为。

```python
import math
import random

def softmax(logits, temperature=1.0):
    assert temperature > 0
    scaled = [x / temperature for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps)
    probs = [x / s for x in exps]
    assert abs(sum(probs) - 1.0) < 1e-9
    return probs

def greedy_decode(logits):
    probs = softmax(logits, temperature=1.0)
    idx = max(range(len(probs)), key=lambda i: probs[i])
    return idx, probs

def sample_from_probs(indices, probs, rng=random):
    r = rng.random()
    acc = 0.0
    for idx, p in zip(indices, probs):
        acc += p
        if r <= acc:
            return idx
    return indices[-1]

def top_k_sample(logits, k, temperature=1.0, rng=random):
    assert k >= 1
    probs = softmax(logits, temperature=temperature)
    ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:k]
    indices = [i for i, _ in ranked]
    values = [p for _, p in ranked]
    s = sum(values)
    renorm = [p / s for p in values]
    idx = sample_from_probs(indices, renorm, rng=rng)
    return idx, renorm, indices

def top_p_sample(logits, p, temperature=1.0, rng=random):
    assert 0 < p <= 1
    probs = softmax(logits, temperature=temperature)
    ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)

    kept = []
    cumulative = 0.0
    for idx, prob in ranked:
        kept.append((idx, prob))
        cumulative += prob
        if cumulative >= p:
            break

    indices = [i for i, _ in kept]
    values = [prob for _, prob in kept]
    s = sum(values)
    renorm = [v / s for v in values]
    idx = sample_from_probs(indices, renorm, rng=rng)
    return idx, renorm, indices

# 玩具 logits，对应概率大致接近 [0.50, 0.20, 0.15, 0.10, 0.05]
logits = [2.4, 1.5, 1.2, 0.8, 0.1]

g_idx, g_probs = greedy_decode(logits)
assert g_idx == 0

k_idx, k_probs, k_indices = top_k_sample(logits, k=3, temperature=1.0, rng=random.Random(0))
assert len(k_indices) == 3
assert set(k_indices).issubset({0, 1, 2, 3, 4})

p_idx, p_probs, p_indices = top_p_sample(logits, p=0.8, temperature=1.0, rng=random.Random(0))
assert len(p_indices) >= 1
assert abs(sum(k_probs) - 1.0) < 1e-9
assert abs(sum(p_probs) - 1.0) < 1e-9

# 温度降低后，最大概率应更集中
probs_t1 = softmax(logits, temperature=1.0)
probs_t07 = softmax(logits, temperature=0.7)
assert probs_t07[0] > probs_t1[0]
```

这个实现里有几个关键点：

1. 贪心和采样流程要分离。前者取 `argmax`，后者必须先裁剪候选集，再随机采样。
2. Temperature 应该在裁剪之前应用。因为 Top-k 和 Top-p 都是按概率排序和截断，温度会直接影响候选集合。
3. Top-p 必须按“累计概率”截断，而不是“概率大于某阈值”截断，这两个概念不同。

如果要扩展到 Beam Search，接口通常不再只返回一个 token，而要返回多个候选前缀及其累计分数。也就是说，贪心和采样是“单状态推进”，Beam 是“多状态并行推进”。

真实工程例子：在推理服务中，一个常见做法是把 `temperature/top_p/top_k` 封装成统一配置对象。这样同一个模型可以复用一套推理 pipeline，只在策略层变化，而不需要改模型前向计算代码。

---

## 工程权衡与常见坑

解码策略不是“哪个更高级”，而是“哪个与目标函数一致”。

| 常见坑 | 原因 | 缓解措施 |
|---|---|---|
| Beam 输出重复短语 | 路径分数偏向高频安全表达 | 加长度惩罚、重复惩罚、多样性约束 |
| Beam size 越大越慢 | 每步扩展和排序的候选线性增加 | 通常先试 4 到 8，而不是盲目加到 16 |
| Top-k 抽到噪声词 | 固定 $k$ 无法适应分布形状 | 优先改用 Top-p，或减小 $k$ |
| Top-p 行为波动 | nucleus 集合随分布变化 | 固定温度，先调 $p$ 再调别的参数 |
| Temperature 太高 | 分布被拉平，低概率 token 被放大 | 开放式生成一般不要长期超过 1.0 |
| 贪心过于模板化 | 每步都选局部最大项 | 仅用于低延迟或确定性强任务 |

Beam 的常见坑最典型。它在翻译里常有效，但在开放式文本生成里往往会“太安全”。安全，意思是总走高频表达，导致文本同质化、重复、无信息增量。例如生成解释性文本时，Beam 可能更倾向于输出一串常见套话，而不是上下文里更具体的信息。

Top-k 的坑更隐蔽。假设某一步模型其实非常确定，第一名概率已经 0.92，后面几个都是边缘项。如果还固定 `top_k=20`，就等于强行把许多弱相关 token 放回抽样池。结果是文本偶尔会突然跳出错字、怪词或风格断裂。

Top-p 解决的是这个问题，但也引入另一个问题：可解释性更差。因为每一步的候选数量不是固定值，线上排查时你会发现“相同参数下，不同位置的随机空间完全不同”。这要求测试不能只看一两条样例，而要看批量分布。

真实工程例子：客服机器人如果 `top_p=0.95`、`temperature=1.1` 同时开启，短问题上可能表现正常，但在长上下文、多轮对话里容易出现话题漂移。漂移的意思是回复虽然语法通顺，但逐渐偏离用户意图。

---

## 替代方案与适用边界

实践里通常这样选：

| 任务 | 推荐策略 | 常见参数 | 不适用场景 |
|---|---|---|---|
| 机器翻译 | Beam Search | `beam_size=4~8`, `length_penalty=0.6~1.0` | 高创意写作 |
| 摘要生成 | Beam 或小温度采样 | `beam_size=4`, 或 `top_p=0.9, T=0.7` | 需要大量风格变化 |
| 问答抽取 | 贪心或小 Beam | `beam_size=1~3` | 需要丰富表达 |
| 聊天机器人 | Top-p + Temperature | `top_p=0.9~0.95`, `T=0.7~0.9` | 必须严格一致的输出 |
| 创意写作 | Top-p 或 Top-k | `top_p≈0.9`, 或 `top_k=40~100` | 标准答案型任务 |
| 低延迟边缘设备 | 贪心 | `beam_size=1` | 需要探索多候选时 |

这里有两个经验规则。

第一，Beam 与 Sampling 一般不要在同一层目标上混用。原因不是“技术上不能”，而是优化方向冲突。Beam 试图逼近高概率最优序列，Sampling 则故意保留随机探索空间。一个追求确定最优，一个追求受控多样性。

第二，可以分阶段组合，而不是同阶段混合。例如：

- 先用 Beam 生成多个高分候选，再用 reranker 重排。
- 先用采样生成多个答案，再用规则或奖励模型筛选。
- 代码生成里先采样出多个候选程序，再执行测试用例做选择。

这类组合不是直接把 Beam 和 Top-p 混到同一步，而是在“候选生成”和“候选筛选”两个阶段分别发挥作用。

真实工程例子：多语种翻译 pipeline 常用 `beam_size=5 + length penalty`，因为它要求结果稳定、术语一致；通用聊天机器人更常见的是 `top_p=0.92 + temperature=0.8`，因为回复既要合理，也要避免机械重复。

---

## 参考资料

| 来源 | 内容摘要 | 建议重点看什么 | 理由 |
|---|---|---|---|
| Arun Baby, Beam Search Decoding | 对比 Greedy、Beam、Sampling 的工程实践 | Beam 的使用场景、重复问题、参数权衡 | 工程经验 |
| Southbridge.ai, Entropixplained | 解释 temperature、top-k、分布形状变化 | 温度公式、重归一化逻辑 | 数学机制 |
| Wikipedia, Top-p Sampling | 给出 nucleus sampling 的定义与累计截断逻辑 | $V^{(p)}$ 的集合定义 | 概念基线 |
| 相关解码论文与框架文档 | 长度惩罚、重复惩罚、候选重排 | Beam score 的修正项 | 工程扩展 |

参考链接：

- Arun Baby: https://arunbaby.com/ml-system-design/0023-beam-search-decoding/
- Southbridge.ai: https://www.southbridge.ai/blog/entropixplained
- Wikipedia, Top-p sampling: https://en.wikipedia.org/wiki/Top-p_sampling
