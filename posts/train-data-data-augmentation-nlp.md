## 核心结论

NLP 里的数据增强，本质是**在不改模型结构的前提下，主动制造更多“可学习样本”**。白话说，就是先不急着换更大的模型，而是先让模型见到更多合理表达。

当前主流做法不是机械同义词替换，而是**受控地扩大语义覆盖**：让模型看到“同一个标签的更多表达方式”，从而提升泛化能力。最常见的工程形式是“原句 + 语义等价的 paraphrase（改写句）”一起训练。

但增强不是“越多越好”。稳定增强必须同时满足两个条件：

| 维度 | 直观定义 | 工程含义 |
| --- | --- | --- |
| Affinity | 增强样本仍然像原始分布 | 不要把模型带偏，验证集准确率不能掉 |
| Diversity | 增强样本带来新信息 | 不只是重复原句，训练时要出现新的学习信号 |

如果只有 Affinity 没有 Diversity，增强集只是“原句复印件”，训练价值有限；如果只有 Diversity 没有 Affinity，模型会学到偏离任务边界的模式，出现标签漂移或决策边界崩裂。

一个新人能立刻理解的例子是情感分类。原句是“这家餐厅上菜很快，味道也稳定”，标签是“积极”。增强可以生成“出餐效率高，而且口味一直在线”这类改写。它换了表达，但没有改变情感极性，这就是有效增强。

---

## 问题定义与边界

数据增强可以定义为：**在训练过程中生成或变换样本，以扩展训练分布，同时保持任务标签语义不变。**

这里有两个关键词。

第一是“扩展训练分布”。训练分布指模型训练时看到的数据形态范围。白话说，就是模型以为“世界上用户会怎么说话”。原始训练集很小的时候，这个范围通常过窄，模型只记住少量固定表达。

第二是“标签语义不变”。标签语义就是样本应该对应的监督信号。对白话任务来说，就是“原来是正面，增强后还得是正面；原来问的是退款，增强后不能变成投诉物流”。

所以，数据增强的边界不是“能不能生成新句子”，而是“**新句子能不能合法地属于原任务**”。

下面这个边界表很重要：

| 维度 | 监控量 | 满足条件 | 违背后的问题 |
| --- | --- | --- | --- |
| Affinity | updated accuracy，相对干净验证集的准确率变化 | 增强后验证性能不下降，最好上升 | 分布偏移，模型学偏 |
| Diversity | 训练损失或新增梯度信号 | 新样本确实让模型继续学习 | 样本重复，增强无效 |
| 标签一致性 | 原标签与增强样本标签是否一致 | 语义核心保持不变 | label drift，标签漂移 |
| 流形接近性 | 增强样本是否仍靠近原始语义空间 | 不远离原任务数据流形 | 决策边界被撕裂 |

“流形”这个词可以简单理解为：**真实数据通常分布在高维空间里的一块可行区域**。增强样本如果跑出这块区域，模型就会被迫拟合一些现实里几乎不会出现的表达。

玩具例子可以看二分类短文本任务：

- 原句：`这个手机续航不错`，标签：正面
- 合法增强：`这款手机电池很耐用`，标签仍是正面
- 越界增强：`这个手机续航差，而且发热严重`，这已经变成负面，标签不能再沿用原值

所以，增强不是“文本改写技术”，而是“**带标签约束的分布扩展技术**”。

---

## 核心机制与推导

数据增强为什么会起作用，可以用三个量来理解：Affinity、Diversity、EIG。

先给出公式：

$$
\text{Affinity} = \text{Accuracy}(m, AugVal) - \text{Accuracy}(m, CleanVal)
$$

$$
\text{Diversity} = \mathbb{E}_{x \sim AugTrain}[L_{train}(m, x)]
$$

$$
\text{EIG} = \frac{\text{vol}(F) - \text{vol}(Z)}{\text{vol}(Z)}
$$

其中：

- $m$ 是模型
- $CleanVal$ 是原始验证集
- $AugVal$ 是引入增强后的验证评估
- $L_{train}$ 是训练损失
- $\text{vol}(Z)$ 可以理解为原始特征空间覆盖体积
- $\text{vol}(F)$ 是增强后特征空间覆盖体积

Affinity 衡量“增强是否仍然贴近原任务”。如果增强后模型在干净验证集上更好，说明增强样本没有破坏任务边界，反而补足了表达覆盖。

Diversity 衡量“增强是否真的带来新学习信号”。如果增强样本过于重复，模型对它们的损失很低，训练几乎学不到新东西；如果适度增加变化，模型会看到新的局部模式，训练收益更高。

EIG 是有效信息增益，可以理解为“**覆盖范围扩大了多少**”。它描述的是增强到底是在做有价值的扩展，还是只是把数据推得太远。  
当 $\text{EIG}$ 太低时，说明样本几乎没扩展；当 $\text{EIG}$ 太高时，常见情况是生成样本偏离原分布过多。

一个最小数值例子：

- 干净验证集准确率：$0.82$
- 增强后验证准确率：$0.86$
- 则 $\text{Affinity} = 0.86 - 0.82 = 0.04$

这表示增强没有伤害泛化，反而带来了 4 个百分点提升。

再看训练损失：

- 原始训练损失：$0.45$
- 增强数据参与训练后的平均损失：$0.38$

这通常说明增强样本帮助模型学到了更稳定的模式，尤其在小样本或长尾类别中常见。

再看 EIG：

- 若体积比 $\text{vol}(F)/\text{vol}(Z) = 1.2$
- 则 $\text{EIG} = (1.2 - 1.0) / 1.0 = 0.2$

这个量不算激进，通常意味着“有扩展，但没脱轨”。

这里有个关键推导逻辑：

1. 模型泛化误差高，往往不是模型不会拟合，而是训练分布覆盖不够。
2. 受控增强通过生成语义等价但表达不同的样本，扩大了局部邻域覆盖。
3. 如果扩展后的样本仍在原任务流形附近，模型会学到更平滑的决策边界。
4. 如果扩展过远，模型会把无关模式也当成有效信号，导致验证性能下降。

所以，增强成功的条件不是“生成质量高”，而是“**生成样本对决策边界的作用方向正确**”。

真实工程例子可以看客服意图分类。原始训练集里“退款申请”类只有几百条，而且多数写法集中在“我要退款”“申请退款”。线上用户真实表达会更散，比如“能把钱退回来吗”“这个订单我不想要了怎么处理”“麻烦取消并返还支付金额”。如果只靠原始样本训练，模型会把很多变体误判到“售后咨询”或“投诉”。加入受控 paraphrase 后，这类长尾表达被提前覆盖，召回率通常会明显改善。

---

## 代码实现

工程上最常见的流程是：

| 步骤 | 输入 | 输出 | 目的 |
| --- | --- | --- | --- |
| 提示构造 | 原句、标签、约束 | 生成提示词 | 明确“可改写，不可改标签” |
| 样本生成 | 提示词、LLM | 候选增强样本 | 扩大表达覆盖 |
| 自动筛选 | 原句、候选句、标签 | 合格增强样本 | 过滤 label drift |
| 合并训练 | 原始集 + 增强集 | 新训练集 | 提升泛化 |
| 监控评估 | 验证集、训练损失、特征统计 | Affinity / Diversity / EIG | 判断增强是否有效 |

一个新手可直接理解的伪代码是：

```python
generate_prompt = f"Paraphrase the sentence while keeping label = {label} unchanged."
for sample in dataset:
    candidates = llm_generate(generate_prompt, sample["text"])
    for c in candidates:
        if not label_drift(sample["text"], c, sample["label"]):
            augmented_dataset.append({"text": c, "label": sample["label"]})
```

下面给一个可运行的 Python 玩具实现。它不调用真实 LLM，而是模拟“生成 + 筛选 + 合并”的流程，重点看工程结构。

```python
from typing import List, Dict

dataset = [
    {"text": "这个手机续航不错", "label": "positive"},
    {"text": "物流太慢了", "label": "negative"},
]

candidate_map = {
    "这个手机续航不错": [
        "这款手机电池很耐用",
        "这个手机续航差",
        "手机用一天还有电",
    ],
    "物流太慢了": [
        "配送速度太慢",
        "快递到得很及时",
        "收货时间拖太久了",
    ],
}

positive_words = {"不错", "耐用", "还有电", "及时"}
negative_words = {"慢", "差", "拖"}

def simple_sentiment(text: str) -> str:
    pos = sum(w in text for w in positive_words)
    neg = sum(w in text for w in negative_words)
    return "positive" if pos >= neg else "negative"

def label_drift(original_label: str, candidate_text: str) -> bool:
    return simple_sentiment(candidate_text) != original_label

def augment(dataset: List[Dict[str, str]]) -> List[Dict[str, str]]:
    augmented = []
    for sample in dataset:
        for candidate in candidate_map[sample["text"]]:
            if not label_drift(sample["label"], candidate):
                augmented.append({"text": candidate, "label": sample["label"]})
    return augmented

augmented_dataset = augment(dataset)

assert {"text": "这款手机电池很耐用", "label": "positive"} in augmented_dataset
assert {"text": "这个手机续航差", "label": "positive"} not in augmented_dataset
assert {"text": "配送速度太慢", "label": "negative"} in augmented_dataset
assert {"text": "快递到得很及时", "label": "negative"} not in augmented_dataset
assert len(augmented_dataset) == 4

print(augmented_dataset)
```

如果落到真实工程，流程通常会再加两层筛选。

第一层是**标签一致性筛选**。可以用分类器复判、规则模板、人工抽检，或者“小模型打分 + 大模型复审”的组合。

第二层是**分布偏移筛选**。常见方式是计算 embedding 相似度，保留“与原句足够接近但不完全重复”的样本。一个简单思想是限制：

$$
\tau_{min} \leq \cos(e(x), e(\tilde{x})) \leq \tau_{max}
$$

其中 $e(\cdot)$ 是文本向量，$\tilde{x}$ 是增强样本。这个约束的意思是：不要太远，也不要太像。

真实工程例子可以写成下面这种 pipeline：

```python
def build_prompt(text: str, label: str) -> str:
    return (
        f"请改写下面的文本，保持标签不变。\n"
        f"原标签: {label}\n"
        f"要求: 不改变事实、不反转情感、不新增未出现的关键信息。\n"
        f"文本: {text}"
    )

def accept_candidate(original_text, original_label, candidate_text, sim_score, min_sim=0.75, max_sim=0.95):
    if predicted_label(candidate_text) != original_label:
        return False
    if sim_score < min_sim or sim_score > max_sim:
        return False
    return True
```

核心点不是提示词写得多花，而是整个链路里有**生成、约束、筛选、监控**四个闭环。

---

## 工程权衡与常见坑

LLM 驱动增强的最大问题，不是“生成不出来”，而是“**生成得太像真的，但其实标签已经变了**”。

最常见风险和控制措施如下：

| 风险 | 控制措施 |
| --- | --- |
| label drift | 提示中显式要求“保持结论为正面且原标签不变”，并做分类器复判 |
| 新增事实 | 限制“不得添加原文没有的实体、时间、数字、结论” |
| EIG 偏高 | 设置 embedding 距离阈值，过远样本直接丢弃 |
| EIG 偏低 | 去重、限制高相似 paraphrase，避免复读式增强 |
| 类别失衡加剧 | 对长尾类增强，对头部类限量生成 |
| 生成成本高 | 先在小样本上做离线试验，证明有效再扩量 |
| 线上效果不稳定 | 每 N 个样本人工抽检，保留审计日志 |

实践里最容易犯的三个错：

第一，只看生成文本“读起来像真话”，不看标签是否还成立。  
比如原句是“客服处理得很快”，生成成“客服终于处理了，但态度很差”。这句话通顺，但标签语义已经变化。

第二，只追求多样性，不控制靠近原分布。  
很多团队喜欢把温度调高、让 LLM 生成得更“有创造性”。对写作任务可能有用，对分类训练通常危险，因为这会把样本推离任务流形。

第三，把增强当万能手段。  
如果原始标注本身噪声大，增强只会把错误标签扩散出去。脏数据乘以 2，结果通常不是更好数据，而是更大污染面。

一个常用的抽检流程是：每生成 100 条，随机抽 10 条做人审；若标签一致率低于阈值，比如 95%，则回滚这批提示模板。这个流程很土，但比只看离线指标可靠。

---

## 替代方案与适用边界

数据增强不是唯一选择。很多时候，问题并不是“样本太少”，而是“采样策略不对”或“标签体系不稳”。

可以把常见方案放在一起比较：

| 方案 | 核心做法 | 适用场景 | 不适合场景 |
| --- | --- | --- | --- |
| 数据增强 | 生成或变换新样本 | 长尾类别、小样本、短文本分类 | 标签极敏感、事实不能改写的任务 |
| 重采样 | 提高少数类采样频率 | 类别不平衡明显 | 少数类本身太脏时 |
| 规则替换 | 用词典/模板做替换 | 行业术语稳定、可控性要求高 | 开放域表达变化太多 |
| 弱监督标注 | 用规则或教师模型自动标注 | 未标注数据很多 | 教师质量不稳时误差会扩散 |
| 常规模型正则化 | dropout、weight decay 等 | 数据量充足时稳健提效 | 长尾覆盖不足时补不了表达缺口 |

什么时候优先用数据增强？

- 标注贵，短期拿不到更多人工样本
- 长尾类别明显，线上表达比训练集更散
- 任务标签有相对稳定的语义边界，比如意图分类、情感分类、FAQ 匹配

什么时候不该先上数据增强？

- 数据已经很多，主要瓶颈在模型容量或特征设计
- 标签极度敏感，比如法律定性、医疗结论、风控拒绝原因
- 原始数据噪声高，连基准集都不稳定

一句话判断边界：**增强适合“可改写但不可改意”的任务；不适合“一个字变化都会改变标签责任”的任务。**

---

## 参考资料

1. Data Augmentation using LLMs: Data Perspectives, Learning Paradigms and Challenges  
   用途：总结 LLM 数据增强的主流策略、训练范式和可控性挑战。  
   链接：https://huggingface.co/papers/2403.02990

2. Affinity and Diversity 指标介绍（AI Scholar）  
   用途：解释增强数据为什么既要贴近原分布，又要提供新学习信号。  
   链接：https://ai-scholar.tech/en/articles/data-augmentation/Data-Augmentation-metrics

3. Tradeoffs Between Richness and Bias of Augmented Data in Long-Tail Recognition（Entropy, 2025）  
   用途：讨论增强数据的信息丰富度与分布偏移之间的权衡，并引入 EIG。  
   链接：https://www.mdpi.com/1099-4300/27/2/201

4. Grokking in the Wild（OpenReview）  
   用途：提供“合成推理链增强训练信号”的真实研究案例，说明增强不只适用于表层改写，也能用于结构化推理补全。  
   链接：https://openreview.net/forum?id=lyUJH51URt
