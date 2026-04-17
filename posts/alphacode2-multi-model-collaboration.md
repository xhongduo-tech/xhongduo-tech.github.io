## 核心结论

AlphaCode 2 的关键不是“一个更强的单模型”，而是把代码生成拆成两个协作角色：先由策略模型大规模采样，再由评分模型做排序。策略模型可以理解为“负责多写草稿的模型”，评分模型可以理解为“负责判断哪些草稿更像正确解的模型”。这套流水线让系统在 Codeforces 这类高难度竞赛题上达到接近人类强选手的水平，公开结果对应约前 15% 参赛者。

它的核心收益来自两个方向。第一，高样本量提高“撞到正确解”的概率。竞赛题往往存在大量看似合理但实际错误的程序，若只生成 1 到 5 个候选，正确解很容易根本不出现。第二，聚类加 rerank 保证提交名额不会浪费在重复解法上。聚类就是“把行为相似的程序归成一组”，这样系统不是从 100 万份草稿里直接挑前 10，而是先分组，再从每组挑最可信的一份。

一个玩具例子是：把同一道题交给一组写代码机器人。它们先写出 100 万份 C++ 草稿，经过编译和样例过滤后只剩 5 万份；再把这些程序按输出行为分成若干类，只保留最大的 10 个簇；最后每个簇交出评分最高的 1 份，因此最多提交 10 份。这里真正重要的不是“每次都一次写对”，而是“先把正确解保留下来，再别把它在后处理阶段丢掉”。

| 阶段 | 作用 | 典型样本规模 | 输出数量 |
|---|---|---:|---:|
| 策略模型采样 | 生成大量候选程序 | $10^6$ 级 | 数十万到百万 |
| 聚类/过滤 | 剔除无效解并做多样性保留 | 数万级 | 前 10 个主要簇 |
| 评分模型 rerank | 在每个簇内挑最可信解 | 每簇若干到数千 | 最多 10 个提交 |

---

## 问题定义与边界

AlphaCode 2 解决的问题，不是“写一段能运行的小脚本”，而是“给定自然语言竞赛题描述，生成足够可靠的完整程序，并在有限提交次数内尽量做对”。这里的输入是题面、输入输出格式、样例和约束；输出不是单一程序，而是一个候选集经过筛选后的少量提交。

形式化地说，设策略集合为 $\{\pi_\theta\}$，它表示多个用于采样的代码生成策略；从这些策略得到候选集 $S$。经过编译、运行样例和基础 IO 过滤后，得到过滤集合 $F \subseteq S$。再把 $F$ 聚成若干簇 $\{C_k\}$，每个簇里候选程序的行为相近。评分模型给每个候选 $c$ 一个分数 $r_\phi(c)$，最终在每个保留簇中选择

$$
c_k^* = \arg\max_{c \in C_k} r_\phi(c)
$$

并把这些 $c_k^*$ 作为最终提交。这样做的直接目标，是在有限提交预算下最大化命中正确解的概率，而不是最大化单次生成的“平均流畅度”。

边界也很明确。第一，它主要面向竞赛编程，默认语言、测试方式和目标函数都偏向 Codeforces 这类在线判题系统。第二，它依赖大规模采样与执行过滤，因此不适合低延迟、低算力场景。第三，它优化的是“最终解题率”，不是“代码可维护性”或“企业工程风格”。

对新手可以这样理解：你把题目交给写代码机器人，它不是只交一份作业，而是先写很多草稿，再从中挑最像能过题的少数版本去提交。

---

## 核心机制与推导

策略模型阶段沿用了 AlphaCode 里的 GOLD 思路。GOLD 可以理解为“一种偏向高质量样本的训练方式”，它不是平均学习所有程序，而是对高概率样本给更大权重。其梯度可写成近似形式：

$$
\nabla L \propto - \sum_s P_\theta(s)^\alpha \nabla \log P_\theta(s), \quad \alpha=\tfrac{1}{2}, \beta=0.05
$$

这里 $P_\theta(s)$ 是程序 $s$ 在当前策略下的概率。因为权重里有 $P_\theta(s)^\alpha$，高概率样本会被进一步放大，训练会更偏向“精确命中少量高质量模式”，而不是一味追求覆盖所有可能写法。直白说，模型会更愿意重复自己更有把握的正确结构。

但仅靠这个仍然不够。原因是竞赛题的正确解很稀疏，错误代码往往也能写得很像样。于是第二阶段引入评分模型。评分模型不是继续生成代码，而是估计“这份代码有多像真正确解”。一个关键观察是，token-level likelihood，也就是“模型对代码中每个 token 的局部置信度”，和整体正确率存在相关性。token 就是代码里的最小语法片段，比如关键字、变量名、操作符。若一份程序整体由高置信 token 组成，并且这种置信模式来自针对正确性校准后的评分器，那么它更可能是真的能通过测试的解。

因此，排序目标可以写成：

$$
\text{Submit}(F) = \{ \arg\max_{c \in C_k} r_\phi(c) \mid k \in \text{TopClusters} \}
$$

其中 $r_\phi(c)$ 不是简单的语言流畅度，而是尽量对齐“通过隐藏测试的概率”。

一个玩具例子是求数组最大值。策略模型可能写出很多版本：有的用线性扫描，有的错误地初始化为 0，有的忘记处理负数。对新手看起来这些代码都“像是对的”，但评分模型若已经见过大量正确/错误样本，就会倾向给“初始化为第一个元素”的版本更高分，因为这种 token 组合与真实正确性更一致。

一个真实工程例子是 Codeforces 上的构造题或贪心题。此类题常出现大量“样例过了但系统测试挂掉”的程序。单看可读性无法区分，必须依赖大样本保留多种思路，再靠评分模型识别哪些思路更接近稳定正确的算法结构。

| 维度 | raw token likelihood | 精调后评分分布 |
|---|---|---|
| 含义 | 语言模型对代码自然度的原始置信度 | 针对正确性校准后的可信度 |
| 优点 | 便宜，直接可算 | 更接近实际过题率 |
| 风险 | 容易高估“写得像代码”的错误程序 | 需要标注和持续校准 |
| 适用位置 | 初步过滤或特征之一 | 最终 rerank 主信号 |

---

## 代码实现

工程上可以把 AlphaCode 2 理解成“采样 -> 过滤 -> 聚类 -> 打分 -> 提交”的流水线。下面给一个可运行的简化版 Python 玩具实现，它不生成真实 C++，而是模拟候选程序在样例上的行为、聚类和评分过程。

```python
from collections import defaultdict

def io_filter(samples):
    return [s for s in samples if s["compile_ok"] and s["sample_ok"]]

def cluster_by_behavior(samples):
    buckets = defaultdict(list)
    for s in samples:
        buckets[s["behavior_signature"]].append(s)
    clusters = sorted(buckets.values(), key=len, reverse=True)
    return clusters

def select_submissions(samples, top_k_clusters=2):
    filtered = io_filter(samples)
    clusters = cluster_by_behavior(filtered)[:top_k_clusters]
    picks = [max(cluster, key=lambda x: x["score"]) for cluster in clusters]
    return picks

samples = [
    {"id": "a", "compile_ok": True, "sample_ok": True, "behavior_signature": "greedy_v1", "score": 0.62},
    {"id": "b", "compile_ok": True, "sample_ok": True, "behavior_signature": "greedy_v1", "score": 0.81},
    {"id": "c", "compile_ok": True, "sample_ok": False, "behavior_signature": "dp_v1", "score": 0.95},
    {"id": "d", "compile_ok": True, "sample_ok": True, "behavior_signature": "dp_v2", "score": 0.77},
    {"id": "e", "compile_ok": False, "sample_ok": False, "behavior_signature": "broken", "score": 0.99},
]

picked = select_submissions(samples, top_k_clusters=2)
picked_ids = {x["id"] for x in picked}

assert "b" in picked_ids
assert "d" in picked_ids
assert "c" not in picked_ids
assert "e" not in picked_ids
```

如果把它扩展成更接近真实系统的伪代码，大致是：

```python
samples = collect_samples(policy_models, max_samples=1_000_000)
filtered = compile_and_io_filter(samples)
clusters = cluster(filtered)[:10]
submissions = [max(cluster, key=score_model) for cluster in clusters]
```

真实实现里，`collect_samples` 往往来自多个经过 GOLD 微调的策略模型，并配合不同温度和采样参数制造多样性；`compile_and_io_filter` 会真正编译 C++ 并跑样例与附加测试；`cluster` 可能基于输出行为、程序特征或执行轨迹；`score_model` 则是独立训练的 Gemini 评分器。

| 阶段 | 输入规模 | 输出规模 | 关键子模块 |
|---|---:|---:|---|
| 采样 | 1 道题 | $10^6$ 候选 | 多策略模型、温度采样 |
| 过滤 | $10^6$ | $10^4$ 到 $10^5$ | 编译、样例、IO 检查 |
| 聚类 | $10^4$ 到 $10^5$ | 前 10 簇 | 行为签名、去重、多样性控制 |
| rerank | 每簇若干样本 | 每簇 1 个 | 评分模型、正确性校准 |
| 提交 | 最多 10 | 最多 10 | 提交预算控制 |

---

## 工程权衡与常见坑

第一类问题是成本。百万级采样的主要压力不在“文本生成”本身，而在后续编译、运行和存储。即使 95% 候选会被很早剔除，前面的生成和后面的执行也都很贵。因此系统必须高度并行，并且把廉价过滤放在前面，把昂贵评分放在后面。

第二类问题是多样性。若只用一个风格稳定的策略模型，即使采样很多次，得到的程序也可能只是同一种思路的轻微改写。这样聚类后看似有很多候选，实际只覆盖极少算法路线。解决方法通常是多策略模型、不同温度、不同 prompt 或不同采样种子联合使用。

第三类问题是评分偏差。评分模型若只学会“像正确代码的表面形式”，就会高分选出错误解。尤其在竞赛题里，很多错误程序语法完整、变量命名合理、复杂度描述也对，但核心边界条件是错的。这要求评分器持续用真实正确性标签做校准，而不能只依赖语言模型原始似然。

第四类问题是聚类失真。若聚类依据过粗，不同错误会被混成一簇；若依据过细，同一种算法的等价实现会被拆散，浪费提交名额。聚类质量直接决定最后 10 个提交是否真正覆盖不同解法空间。

对新手可理解为：如果只让一个“写作风格很固定”的模型写很多遍，它多数时候只是换变量名；如果裁判又只喜欢“看起来像标准答案”的代码，就会把真正能过隐藏测试的解错杀掉。

| 风险/坑 | 表现 | 规避手段 |
|---|---|---|
| 采样成本过高 | 生成与执行预算爆炸 | 分层过滤、并行执行、限制后续打分规模 |
| 候选缺乏多样性 | 聚类后全是相似解法 | 多策略模型、温度扰动、多 prompt |
| 高似然但错误 | 代码看起来顺但逻辑错 | 用真实正确率数据校准评分器 |
| 聚类不合理 | 重复提交或覆盖不足 | 用行为特征和执行结果联合聚类 |
| 过滤太松 | 错误代码进入 rerank | 增加编译、样例、附加随机测试 |
| 过滤太严 | 正确但脆弱的解被误删 | 把廉价测试作为筛选而非最终裁决 |

---

## 替代方案与适用边界

最直接的替代方案是单模型 beam search。beam search 可以理解为“同时保留若干最有希望的生成分支”。它比百万级采样便宜很多，但覆盖的解空间更窄，尤其面对需要跳出局部模式的题时，容易在一批相似错误中反复搜索。

第二种替代方案是“小规模采样 + 简单 rerank”。例如只采样 5 到 50 个候选，再用样例通过数、代码长度、模型打分综合排序。这在企业内部代码补全或低时延场景更现实，但通常达不到 AlphaCode 2 在竞赛题上的效果。

第三种替代方案是“生成 + 手写验证器”。也就是让模型提议算法，再用问题特定的验证器自动检查。它对某些有强结构约束的任务非常有效，但缺点是需要额外领域知识，通用性不足。

如果只有一台笔记本，比较现实的做法是：一个策略模型生成 5 个候选，用 beam search 或低温采样控制质量，再用样例和少量补充测试选最优。AlphaCode 2 这种 million-scale 流水线更适合科研平台、竞赛系统或有明显算力预算的团队。

| 方案 | 样本数量 | 资源需求 | 预期准确率 | 适用边界 |
|---|---:|---|---|---|
| AlphaCode 2 多模型协作 | $10^5$ 到 $10^6$ | 很高 | 高，适合竞赛题 | 高预算、离线求最优 |
| 轻量单模型 | 1 到 20 | 低 | 中低 | 补全、原型、小任务 |
| 小规模采样 + rerank | 5 到 100 | 中 | 中 | 有少量预算、需平衡时延 |
| 生成 + 验证 | 视验证器而定 | 中到高 | 任务相关 | 规则强、可写验证器 |

---

## 参考资料

下面这些资料足够支撑本文各章节。若只想读一篇原文，建议先看 AlphaCode 2 technical report；它最完整地解释了“采样、过滤、聚类、评分”整条流水线。若想理解训练目标，再看 AlphaCode 论文中的 GOLD 机制。若想理解“为什么 token 级概率能辅助判断正确性”，再补充看 token probability 与 accuracy 对齐的相关研究。

| 参考名 | 来源 | 覆盖内容 |
|---|---|---|
| AlphaCode 2 Technical Report | Google DeepMind | 策略模型、评分模型、采样过滤聚类流水线、Codeforces 评估 |
| Competition-Level Code Generation with AlphaCode | AlphaCode 论文 | GOLD 训练目标、竞赛代码生成的基础方法 |
| Token Probabilities to Mitigate LLM Overconfidence in Medical QA | ScienceDirect 相关研究 | token-level probability 与实际正确率相关性的经验依据 |

1. AlphaCode 2 Technical Report, Google DeepMind. 对应本文“核心结论”“问题定义与边界”“代码实现”“工程权衡与常见坑”。
2. Competition-Level Code Generation with AlphaCode. 对应本文“核心机制与推导”中的 GOLD 思路来源。
3. Token probability 与 accuracy 对齐相关研究。对应本文“核心机制与推导”里评分模型为何能利用 token-level likelihood 近似正确性。
