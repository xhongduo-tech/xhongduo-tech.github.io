## 核心结论

数据质量过滤不是“把坏数据删掉”这么简单，而是把训练资源优先分配给**更值得学的样本**。这里的“样本”就是一条训练数据，例如一段文本、一组问答或一条指令回复。对大模型训练来说，算力和训练步数是稀缺资源，模型每多看一条低质、重复、偏航或有风险的数据，就少看一条真正有用的数据。

一个实用系统通常分成两层：

1. 规则过滤：先用可解释、低成本的方法去掉明显不该进训练的数据，比如太短、乱码、重复、带毒性、疑似隐私泄露、格式损坏。
2. 评分排序：再给剩余样本打分，判断哪些样本更“可学”、更符合当前训练方向，并控制它们被送入训练的频率。

对新手，可以把它理解为：先用规则把明显坏数据挑走，再给剩下的每条打一个价值分。分高的优先训练，分低的延后、降频，或者直接丢弃。否则训练就像反复听噪声，既浪费算力，也容易把模型带偏。

一个常见的统一评分形式是：

$$
s_i = \lambda \cdot L_i + (1-\lambda) \cdot Q_i
$$

其中，$s_i$ 是样本 $i$ 的最终价值分；$L_i$ 表示“可学性”，也就是模型从这条样本上还能学到多少；$Q_i$ 表示“方向一致性”，也就是这条样本的梯度方向是否和整体训练目标一致；$\lambda$ 用来控制两者权重。这个分数既能决定“保不保留”，也能决定“训练几次”。

---

## 问题定义与边界

数据质量过滤的目标，是在训练开始前或训练过程中，尽量排除以下几类样本：

| 类型 | 白话解释 | 风险 |
| --- | --- | --- |
| 低质样本 | 内容混乱、断裂、乱码、无信息量 | 浪费训练步数 |
| 重复样本 | 和别的数据基本一样 | 让模型过度记忆局部模式 |
| 有害样本 | 带毒性、违法、危险引导 | 放大安全风险 |
| 隐私敏感样本 | 包含手机号、身份证、地址等 | 合规风险高 |
| 目标不一致样本 | 和任务方向不一致 | 拉偏模型能力 |

如果完全不做过滤，直接把采集来的所有数据都送进训练，会出现两个问题。第一，训练资源被低价值样本占用，等于把昂贵算力花在垃圾上。第二，分布会被脏数据和重复数据扭曲，模型可能在某些方向上“学得很多”，但学到的是错误模式。

新手可以把“未过滤直接训练”和“先过滤再训练”的差别理解为：

- 未过滤：模型在重复听垃圾话，甚至把有毒和错误内容也当教材。
- 先过滤再训练：先把明显不合格的资料清掉，再把重点教材排在前面。

一个典型多阶段流程可以这样看：

| 阶段 | 目标 | 主要指标 |
| --- | --- | --- |
| 规则阶段 | 去噪、去重、去风险 | 长度、字符集、重复率、毒性、PII 命中 |
| 模型阶段 | 评估样本价值 | 梯度幅度变化、困惑度、分类置信度、一致性 |
| 排序阶段 | 决定训练优先级 | score、分桶比例、采样频率 |
| 监控阶段 | 防止误杀和漏检 | 精度、召回、偏差、lineage 覆盖率 |

这里的边界很重要。过滤不是越狠越好。规则阈值如果过紧，会把冷门但重要的领域数据一并清掉，比如医学、小语种、方言、代码片段、表格文本。结果不是“数据更干净”，而是“数据更窄”。所以数据质量过滤的本质不是最大化删除量，而是在**质量、安全、覆盖度、成本**之间做平衡。

---

## 核心机制与推导

多阶段过滤的核心机制可以概括成一句话：**先做便宜且高精度的排除，再对候选样本做更细粒度的价值判断**。

### 1. 规则过滤为什么先做

规则过滤成本低、吞吐高、解释性强。比如：

- 长度小于 5 个 token 的文本直接丢弃
- URL 比例过高、标点异常密集的文本判为噪声
- 命中敏感词黑名单或隐私模式的文本送人工审查或删除
- 用指纹或 MinHash 做重复检测，去掉近重复内容

这一步不追求完美，只追求把“明显不值得算”的样本尽快拦在外面。因为后面的评分模块通常更贵，可能要依赖模型前向、梯度统计或额外特征计算。

### 2. DELT 评分为什么成立

DELT 可以理解为“数据效能评分”，也就是判断一条数据对当前训练阶段到底有多大价值。它把价值拆成两个部分：

- $L_i$：学习速率或可学性。白话说，就是模型在这条样本上还有没有明显进步空间。
- $Q_i$：方向一致性。白话说，就是这条样本推动参数更新的方向，是否和整体目标大体一致。

统一公式是：

$$
s_i = \lambda \cdot L_i + (1-\lambda) \cdot Q_i
$$

其中：

- 当 $\lambda$ 更大时，系统更偏向“好学”的样本
- 当 $\lambda$ 更小时，系统更偏向“方向正确、代表性强”的样本

一个常见定义是：

$$
Q_i = \cos(g_i, \bar{g}) = \frac{g_i \cdot \bar{g}}{\|g_i\|\|\bar{g}\|}
$$

这里 $g_i$ 是样本 $i$ 的梯度，$\bar{g}$ 是全局平均梯度。余弦相似度越高，说明这条样本和整体训练方向越一致。

### 3. 玩具例子：两个样本 A/B 的比较

假设现在只看两个样本：

- 样本 A：梯度幅度从 2.0 降到 0.5，所以 $L_a \approx 1.5$；和全局梯度的余弦相似度 $Q_a = 0.9$
- 样本 B：梯度幅度只从 1.0 降到 0.8，所以 $L_b \approx 0.2$；余弦相似度 $Q_b = 0.4$

取 $\lambda = 0.6$，则：

$$
s_a = 0.6 \times 1.5 + 0.4 \times 0.9 = 1.14
$$

$$
s_b = 0.6 \times 0.2 + 0.4 \times 0.4 = 0.28
$$

结论很直接：A 的分数更高，应更早进入训练，或者在采样时被看到更多次。对新手来说，这可以理解为：A 更像“老师现在最想让模型学会的重点题”。

### 4. Folding Ordering 为什么需要重复采样

只排序一次还不够，因为训练是动态过程。今天高价值的样本，几轮之后可能已经被学会；今天中等价值的样本，可能在后续阶段变得重要。因此很多系统不会只做一次静态排序，而会采用类似 Folding Ordering 的做法：

- 高分样本重复出现，但不是无限重复
- 中分样本保留一定覆盖率，避免只盯着少数“尖子样本”
- 低分样本降频或进入回收池，等待后续重评

可以把它想成一个循环展开的训练队列：

```text
高分: A A B A C
中分: D E
低分: F
展开后: A A B A C D E A B D ...
```

它的目的不是让高分样本霸占全部训练，而是在“重点复习”和“分布覆盖”之间取得平衡，缓解遗忘和偏差。否则只训练高分样本，模型会在窄分布里越来越强，但整体泛化变差。

### 5. 真实工程例子

假设一个中文问答模型要训练客服与知识检索能力，原始数据来自工单、FAQ、网页抓取、论坛摘要和历史客服对话。真实工程里常见情况是：

- 网页抓取里有大量模板页、导航栏、版权信息
- 历史对话里混有辱骂、隐私、账号信息
- FAQ 有很多版本重复，只是时间和数字不同
- 少量高质量人工问答，数量少但很值钱

如果不做过滤，模型会被大量模板噪声和重复页淹没。做法通常是：

- 规则层删除模板页、空洞短句、PII、明显重复内容
- 模型层计算样本分数，优先学习高质量人工问答和结构完整的客服对话
- 排序层提高高分样本曝光，但保留少量长尾领域数据，避免只学“主流问题”

这就是“数据质量过滤”在真实系统中的核心价值：不是抽象的清洗，而是直接决定训练资源分配。

---

## 代码实现

下面给出一个可运行的 Python 玩具实现。它不依赖深度学习框架，但把流程结构完整保留了：读取数据、规则过滤、计算 DELT score、按分数排序、再做 Folding Ordering。

```python
from math import sqrt
from collections import Counter

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def gradient_magnitude_drop(sample):
    # 用 before - after 模拟“这条样本还让模型进步多少”
    return max(sample["grad_before"] - sample["grad_after"], 0.0)

def sample_gradient(sample):
    return sample["grad_vec"]

def contains_pii(text):
    # 极简示例：连续 11 位数字视为手机号
    digits = "".join(ch for ch in text if ch.isdigit())
    return len(digits) >= 11

def duplicate_ratio(text):
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 1.0
    most_common = Counter(chars).most_common(1)[0][1]
    return most_common / len(chars)

def passes_rules(sample):
    text = sample["text"]
    if len(text.strip()) < 8:
        return False
    if contains_pii(text):
        return False
    if duplicate_ratio(text) > 0.45:
        return False
    if sample.get("toxic", False):
        return False
    return True

def compute_score(sample, g_bar, lambda_):
    L = gradient_magnitude_drop(sample)
    Q = cosine_similarity(sample_gradient(sample), g_bar)
    return lambda_ * L + (1 - lambda_) * Q

def fold_order(scored_samples, repeat_ratio=2):
    output = []
    for rank, item in enumerate(scored_samples):
        repeats = max(1, repeat_ratio - rank // 2)
        output.extend([item["id"]] * repeats)
    return output

data = [
    {
        "id": "A",
        "text": "退款流程分为提交申请、审核和原路退回三个步骤。",
        "grad_before": 2.0,
        "grad_after": 0.5,
        "grad_vec": [0.8, 0.6],
        "toxic": False,
    },
    {
        "id": "B",
        "text": "好的好的好的好的好的",
        "grad_before": 1.0,
        "grad_after": 0.8,
        "grad_vec": [0.3, 0.4],
        "toxic": False,
    },
    {
        "id": "C",
        "text": "请联系 13800138000 获取验证码",
        "grad_before": 1.2,
        "grad_after": 0.3,
        "grad_vec": [0.7, 0.2],
        "toxic": False,
    },
]

g_bar = [1.0, 0.0]
lambda_ = 0.6

filtered = [s for s in data if passes_rules(s)]
scored = sorted(
    filtered,
    key=lambda s: compute_score(s, g_bar, lambda_),
    reverse=True
)
output = fold_order(scored, repeat_ratio=2)

score_a = compute_score(data[0], g_bar, lambda_)
assert round(score_a, 2) == 1.14
assert [s["id"] for s in filtered] == ["A"]
assert output == ["A", "A"]
```

这个例子里：

- `passes_rules` 负责规则过滤
- `compute_score` 负责算样本价值
- `fold_order` 负责根据排序结果重复输出高价值样本

如果要把它扩展成工程可用版本，通常会拆成四类模块：

| 模块 | 作用 | 常见实现 |
| --- | --- | --- |
| rules | 规则过滤 | 正则、黑名单、PII 检测、去重 |
| scorer | 打分 | 梯度统计、困惑度、分类器、奖励模型 |
| scheduler | 排序与重复 | score 降序、分桶采样、folding |
| monitor | 监控与追踪 | 命中率、抽样审计、lineage、污染检测 |

伪代码流程如下：

```python
def pipeline(data, g_bar, lambda_, threshold, repeat_ratio):
    # 1. 先做低成本规则过滤
    filtered = [s for s in data if passes_rules(s)]

    # 2. 给每条样本打分
    scored = []
    for s in filtered:
        score = compute_score(s, g_bar, lambda_)
        if score >= threshold:
            scored.append((s, score))

    # 3. 按得分从高到低排序
    scored.sort(key=lambda x: x[1], reverse=True)

    # 4. 通过 folding 控制高分样本出现次数
    ordered_samples = [item[0] for item in scored]
    training_queue = fold_order(ordered_samples, repeat_ratio=repeat_ratio)

    return training_queue
```

新手可以把这段流程记成一句话：**先删明显坏的，再给剩下的排序，最后让高分样本多上几次课。**

---

## 工程权衡与常见坑

数据质量过滤的难点不在“会不会写规则”，而在“怎么避免把系统做偏”。

常见坑和对策如下：

- 过滤过度 → 可调阈值、分域配置、定期抽样回放  
如果一套规则对所有领域一刀切，冷门合法数据会最先被误杀。典型例子是冷门医学文本，因为句式短、缩写多、符号密，容易被当作噪声踢掉，最后模型在医学问答上明显变差。

- 过滤不足 → 多层防线、风险标签回灌  
只做长度和重复过滤不够，有毒、违法、隐私泄露内容可能漏进训练。应同时部署词表、分类器和抽样人工审核，把漏检样本回流成新规则。

- 只看“干净”，不看“分布” → 监控类别占比与领域覆盖  
把数据越洗越整齐，不代表模型越强。很多系统最后的问题不是数据脏，而是数据窄。要持续看各领域、语种、任务形式的占比是否被过滤器改变。

- score 当成真理 → 定期重算、分阶段调参  
DELT 分数是相对当前训练阶段的估计，不是永恒真值。训练早期需要更多高可学样本，训练中后期可能更需要长尾和难例。$\lambda$ 和阈值都应阶段性调整。

- 没有 lineage 追踪 → 保留来源、规则命中、版本号  
lineage 可以理解为“样本履历”。一条数据来自哪里、经过哪些规则、为何被保留或删除，必须可追踪。否则一旦出现合规事故或能力退化，你无法定位是哪个过滤器造成的。

- 训练集和评测集污染 → 单独做 benchmark 泄露检测  
如果评测题在训练数据里出现过，过滤效果会被假象掩盖，看起来模型更强，其实只是背过答案。这类污染必须在训练数据集之外做独立检测。

两个典型悲剧特别值得记住。

第一个是“误杀长尾领域”。团队为了提高清洗精度，把短文本、符号密集文本大规模删掉，结果医学、法律、代码片段被清掉一大批，主指标看似提升，专业领域能力却掉得很明显。

第二个是“漏掉风险样本”。团队只看训练 loss 和 token 利用率，没有做毒性与隐私监控，结果有毒指令和联系方式混进训练，后续上线后才暴露。这种问题通常不是模型推理时临时产生，而是训练材料本身就被污染了。

---

## 替代方案与适用边界

并不是所有项目都需要多阶段 DELT pipeline。方案应和数据规模、预算、风险等级匹配。

对于几万到几十万条样本的小项目，单阶段规则过滤通常已经足够。因为：

- 数据量不大，人工抽样复核可行
- 训练预算有限，搭复杂打分系统的收益不一定覆盖成本
- 任务边界清晰时，规则往往更稳定、更容易解释

但当规模上升到千万、上亿样本，或者训练目标复杂、合规要求高时，多阶段方案才真正划算。因为这时每 1% 的低质样本都会转化成显著算力浪费或风险暴露。

下面是两类方案对比：

| 方案 | 复杂度 | 资源消耗 | 过滤效果 | 适用场景 |
| --- | --- | --- | --- | --- |
| single-stage rules | 低 | 低 | 能挡住明显坏样本，但细粒度不足 | 小规模项目、原型验证、低预算任务 |
| multi-stage DELT pipeline | 高 | 中到高 | 能同时处理去噪、排序、重复采样和动态调度 | 大规模预训练、高风险场景、长期训练系统 |

还有一种常被低估的替代思路，是把过滤前移到采集端。比如：

- 爬虫抓取时就过滤模板页和低正文比页面
- 用户输入侧先做敏感词和隐私拦截
- 日志采样阶段就去重和脱敏

这样做的好处是后端压力更小，也能减少风险数据进入下游存储。新手可以把它理解为：敏感词在入口挡掉，比先存下来、再进入训练前清洗更便宜也更稳。

所以适用边界可以总结为：

- 小规模、低风险、预算紧：先做单阶段规则过滤
- 中大规模、训练资源昂贵：加入评分排序
- 高风险、强合规行业：入口过滤 + 多阶段训练过滤 + lineage 追踪一起上

---

## 参考资料

1. Towards AI, *Data Quality and Filtering at Scale for Training Large Language Models*  
   https://towardsai.net/p/machine-learning/data-quality-and-filtering-at-scale-for-training-large-language-models  
   聚焦大模型训练中的规则过滤、去重、质量控制和规模化流程设计。

2. Emergent Mind, *Multi-Stage Data Filtering Pipeline*  
   https://www.emergentmind.com/topics/multi-stage-data-filtering-pipeline  
   聚焦多阶段过滤架构，解释规则层、模型层和排序层如何组合。

3. AIModels.fyi, *Data Efficacy for Language Model Training*  
   https://www.aimodels.fyi/papers/arxiv/data-efficacy-language-model-training  
   聚焦 DELT/LQS 一类数据效能评分方法，说明如何用梯度变化和方向一致性评估样本价值。

4. Techment, *Data Quality for AI 2026 Enterprise Guide*  
   https://www.techment.com/blogs/data-quality-for-ai-2026-enterprise-guide/  
   聚焦企业落地中的风险控制、偏差监控、lineage 追踪与合规问题。
