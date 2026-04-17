## 核心结论

合成指令数据的“多样性控制”，本质上不是让模型“多写几种说法”，而是让数据在三个维度都形成有效覆盖：语义、词法、任务分布。语义多样性指任务意图是否真的不同；词法多样性指表面表达是否重复；任务分布指分类、抽取、改写、推理、代码、规划等任务类型是否失衡。三个维度缺一不可。

工程上最稳妥的做法不是一次把温度、`top-k`、`top-p` 调得很大，而是建立“生成-度量-回调参”的闭环。先生成一批指令，再用聚类分数、重复率、压缩率、judge-LLM 评分等指标判断重复和失真，再决定下一轮是扩种子池、提高探索度，还是收紧采样参数做质量修复。

玩具例子：有 100 条指令，如果只是把“总结下面文章”改写成“概括以下内容”，这主要是词法变化，不算新增任务；但如果加入“抽取时间线”“判断论证漏洞”“把自然语言改成 SQL”，才算语义和任务分布上的新覆盖。真实工程里，训练集是否真的变强，往往取决于后者。

| 维度 | 它在看什么 | 常用指标 | 调参方向 |
| --- | --- | --- | --- |
| 语义 | 任务意图是否不同 | LLM cluster score、主题覆盖数 | 扩种子池、增加广度演化 |
| 词法 | 说法是否总是同一套模板 | distinct-n、自重复率、压缩率 | 提高温度、改 prompt 模板 |
| 任务分布 | 不同任务类型是否失衡 | 各类任务占比、长尾覆盖率 | 定向补齐稀缺任务 |

---

## 问题定义与边界

这里的问题不是“如何让文本更花哨”，而是“如何让合成指令成为高质量、可训练、覆盖面广的数据”。可控多样性有边界：变化必须对应有效的新训练信号，而不是模糊、矛盾、不可执行的噪声。

术语首次出现时可以这样理解。语义聚类：把“意思接近”的指令分成组。词法多样性：看句子表面写法是否总在复读。任务类型分布：看数据是否过度集中在少数任务。judge-LLM：用另一个模型充当自动评审员，检查指令是否清晰、可答、无自相矛盾。

一个常见误区是把“同义改写”当成多样性。比如：

- “请总结这段文字”
- “请概括以下内容”
- “阅读后给出摘要”

这三条对模型训练的新增信息很有限，更多是浅层重复。相反，下面才是有效扩展：

- “从这段文字中抽取三条关键事实”
- “判断作者论证是否包含因果倒置”
- “把文本改写为面向儿童的解释版本”

为了把这个问题写成可计算形式，可以用一个简化分数表示每轮聚类发现的新方向密度：

$$
D=\frac{1}{N}\sum_{i=1}^{N}\frac{C_i}{S_i}
$$

其中，$N$ 是评估轮数，$C_i$ 是第 $i$ 轮识别出的聚类数，$S_i$ 是该轮样本数。直观上看，样本数固定时，聚类越多，说明方向越分散；如果样本很多但只聚成少数类，说明重复严重。这个公式不完美，但足够作为工程告警信号。

---

## 核心机制与推导

最常见的工程机制是一套 LLM Cluster-agent 循环。

1. 从当前合成指令池随机抽取 $K$ 条。
2. 让一个评估模型把它们聚成若干语义簇，输出簇数 $C$、每簇样本数、每簇代表任务。
3. 再做一次自验证，检查聚类是否明显混乱。
4. 同时计算词法指标，如 `distinct-2`、重复 n-gram 比例、压缩率。
5. 把这些结果反馈给生成器，更新种子池、温度、`top-k`、`top-p` 或采样方式。

玩具例子：抽 10 条指令，聚成 3 类。假设分别是“总结类”“抽取类”“推理类”，每类样本数分别为 5、3、2。那你立刻知道：这批数据不是“没有多样性”，而是“总结类过密，推理类偏少”。下一轮不是盲目提温度，而是定向补齐推理和长尾任务。

下面是一个简化的轮次表：

| 轮次 | 样本数 $S_i$ | 聚类数 $C_i$ | 贡献 $C_i/S_i$ | 观察 |
| --- | --- | --- | --- | --- |
| 1 | 10 | 3 | 0.30 | 总结类过密 |
| 2 | 10 | 4 | 0.40 | 新增代码类 |
| 3 | 10 | 5 | 0.50 | 主题更分散 |
| 4 | 10 | 3 | 0.30 | 又开始模板化 |

若四轮平均，则

$$
D=\frac{1}{4}(0.30+0.40+0.50+0.30)=0.375
$$

这表示当前有一定覆盖，但还没到很健康的分散状态。

词法侧还需要第二把尺子。常见表达式有：

$$
\text{Distinct-}n=\frac{\#\text{unique } n\text{-grams}}{\#\text{all } n\text{-grams}}
$$

值越大，说明表面表达越不容易复读。

压缩率也常被拿来粗略估计重复：

$$
\text{Compression Ratio}=\frac{\text{compressed size}}{\text{raw size}}
$$

如果大量文本是模板拼接，压缩后会变得更小，压缩率通常更低。它不能直接代表语义多样性，但能快速发现“句式机器复读”。

真实工程例子：做客服助理数据时，若 70% 都是“解释退款政策”，即使写法不同，模型也会偏向这个任务；加入“识别情绪升级”“抽取订单号”“判断是否需转人工”“将非结构化投诉转工单字段”，训练价值才会上来。这就是任务分布比“表面不重复”更重要的原因。

---

## 代码实现

下面给出一个可运行的 Python 玩具实现。它没有真的调用 LLM，而是用标签模拟“语义聚类”，用词统计模拟词法多样性，核心目的是展示闭环逻辑：采样、计分、调参。

```python
from collections import Counter
from typing import List, Dict, Tuple
import math

samples = [
    {"text": "总结下面文章的核心观点", "task": "summary"},
    {"text": "概括以下段落的中心思想", "task": "summary"},
    {"text": "抽取这段文本中的时间和地点", "task": "extract"},
    {"text": "从对话中识别用户情绪", "task": "classify"},
    {"text": "判断这段论证是否存在偷换概念", "task": "reason"},
    {"text": "把自然语言需求改写成SQL查询", "task": "code"},
]

def cluster_score(batch: List[Dict]) -> float:
    tasks = [x["task"] for x in batch]
    c = len(set(tasks))
    s = len(batch)
    return c / s

def distinct_2(texts: List[str]) -> float:
    all_bigrams = []
    for text in texts:
        chars = list(text)
        all_bigrams.extend(["".join(chars[i:i+2]) for i in range(len(chars) - 1)])
    if not all_bigrams:
        return 0.0
    return len(set(all_bigrams)) / len(all_bigrams)

def quality_filter(text: str) -> bool:
    bad_patterns = ["随便写", "无法回答", "不需要条件"]
    return not any(p in text for p in bad_patterns)

def update_temperature(cluster: float, lex_div: float, current_temp: float) -> float:
    # 语义和词法都偏低时，增加探索；否则收紧
    if cluster < 0.55 and lex_div < 0.80:
        return min(1.2, current_temp + 0.1)
    if cluster > 0.75 and lex_div > 0.90:
        return max(0.6, current_temp - 0.1)
    return current_temp

texts = [x["text"] for x in samples if quality_filter(x["text"])]
c_score = cluster_score(samples)
l_score = distinct_2(texts)
temp = update_temperature(c_score, l_score, current_temp=0.7)

assert round(c_score, 2) == 0.83
assert l_score > 0.5
assert 0.6 <= temp <= 1.2
print({"cluster_score": c_score, "lexical_diversity": l_score, "next_temperature": temp})
```

把它映射到真实系统，流程通常是：

- 生成器按当前参数合成一批指令。
- 评估器输出语义聚类报告、任务分布报告、词法重复报告。
- 过滤器用规则和 judge-LLM 去掉不可执行、目标不清、答案依赖外部未知信息的指令。
- 调度器根据指标更新参数。
  - 语义不足：扩充种子池、增加广度演化。
  - 词法不足：提高温度、换模板、混用不同 system prompt。
  - 质量下降：降低温度、缩小 `top-p`、加强过滤。

关键点在于把“多样性控制”写成带 hook 的反馈系统，而不是一次性离线生成脚本。

---

## 工程权衡与常见坑

多样性和质量不是同一个方向。温度升高、`top-k` 变大、`top-p` 放宽，通常会让模型探索更多表达和主题，但也更容易产生边界模糊、条件缺失、任务不可执行的数据。这个趋势常被写成一个简化关系：

$$
\mathcal{R}=-a\cdot e^{c\mathcal{D}}+b
$$

其中 $\mathcal{D}$ 表示多样性，$\mathcal{R}$ 表示可用质量或最终收益。它不是精确物理定律，而是提醒你：多样性继续增大时，收益可能先升后降，甚至快速恶化。

常见坑有五类：

| 坑 | 表现 | 根因 | 修复方式 |
| --- | --- | --- | --- |
| 只看改写数量 | 数据很多但训练没提升 | 把同义改写当新任务 | 加语义聚类和任务统计 |
| 温度过高 | 指令新奇但不可执行 | 探索过度 | 降温并加质量过滤 |
| 类别失衡 | 某类任务占绝对多数 | 种子池偏科 | 对稀缺任务定向采样 |
| judge-LLM 单点失真 | 过滤标准飘忽 | 评审 prompt 不稳 | 固定 rubric，抽样人工校验 |
| 过度追求长尾 | 噪声任务太多 | 把罕见当有效 | 设最小可答性门槛 |

新手最容易犯的错误，是看到多样性分数上升就以为系统变好了。比如把温度从 0.7 调到 1.2，可能确实冒出了更多“看起来新”的任务，但其中一部分会变成“要求不完整”“输入条件不足”“多个目标互相冲突”的坏样本。这时需要高质量低温种子做锚点，并用 judge-LLM 或规则过滤挡掉噪声。

---

## 替代方案与适用边界

如果你不想自己从零设计多样性闭环，一个成熟替代方案是 Evol-Instruct。它把演化分成两个方向。广度演化：扩展到新的主题和任务。深度演化：在同一主题上增加限制、步骤、推理深度或格式约束。白话说，前者负责“去更多地方”，后者负责“在一个地方挖更深”。

玩具例子：先拿“解释 TCP 三次握手”做广度演化，扩成“面向高中生解释”“对比 UDP”“写成面试题”“改成英文摘要”；再做深度演化，要求“必须给出状态转移”“必须指出丢包重传影响”“必须用表格总结”。这样既扩题材，也加难度。

真实工程例子：以 Dolly-15k 之类的基础指令为种子，经过多轮广度和深度演化，可以得到包含抽取、分类、改写、代码、约束推理、多步规划等混合任务的数据池。SurgeGlobal/Evol-Instruct 一类数据就是沿这个思路构造的。

| 策略 | 核心动作 | 优点 | 风险 | 适用场景 |
| --- | --- | --- | --- | --- |
| 广度演化 | 扩新主题、新任务 | 快速补覆盖面 | 容易漂出目标域 | 冷启动、补长尾 |
| 深度演化 | 加约束、加步骤、加推理 | 提升难度和监督密度 | 容易过复杂 | 强化推理、格式控制 |
| 纯模板改写 | 换表达不换任务 | 成本低 | 训练增益有限 | 数据清洗、风格统一 |
| 检索增强生成 | 先取知识再出题 | 事实性更稳 | 系统复杂度高 | 知识密集型任务 |

它的边界也要明确。若你的目标是严格事实问答，仅靠演化不够，通常还要接检索或知识库；若你的目标是统一格式抽取，过强的“广度”反而会破坏分布稳定性。多样性从来不是越大越好，而是越接近目标训练分布越好。

---

## 参考资料

| 资料 | 链接说明 | 关联章节 |
| --- | --- | --- |
| On the Diversity of Synthetic Data and its Impact on Training Large Language Models | https://www.researchgate.net/publication/385107506_On_the_Diversity_of_Synthetic_Data_and_its_Impact_on_Training_Large_Language_Models | 核心结论、问题定义、核心机制 |
| Quality-Diversity Trade-off in Synthetic Data Generation for LLMs | https://openreview.net/forum?id=gIOtRMxLxE | 工程权衡与常见坑 |
| Azure Evol-Instruct 实现说明 | https://azure.github.io/slm-innovator-lab/1_2_evolve-instruct/ | 替代方案与适用边界、入门实践 |
| SurgeGlobal/Evol-Instruct 数据集 | https://huggingface.co/datasets/SurgeGlobal/Evol-Instruct | 替代方案、真实工程例子 |
