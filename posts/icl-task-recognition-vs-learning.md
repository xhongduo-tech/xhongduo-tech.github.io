## 核心结论

ICL，中文常译“上下文学习”，指模型不改参数，只看提示词里的示例就完成新任务。Pan 等人在 2023 年把它拆成两个相对独立的机制：

1. 任务识别（Task Recognition, TR）：模型先判断“这堆示例到底是什么任务”。白话说，就是先认出这是情感分类、自然语言推断，还是问答。
2. 任务学习（Task Learning, TL）：模型再利用示例里的输入输出对应关系，学习“这个输入该映到哪个标签”。白话说，就是真正学会映射规则。

这个拆分很重要，因为很多“few-shot 有效果”的现象，未必说明模型已经学会了新映射，可能只是先靠 TR 猜中了任务，再调用预训练里已有的常识完成预测。

一个直接结论是：`多给示例` 不等于 `更会学映射`。如果新增示例只帮助模型更快认出“这是情感分类”，而没有提供稳定、清晰、可利用的输入标签对应关系，性能提升会很快饱和。Pan 等人的控制实验说明，RANDOM 设定下只保留 TR 时，模型仍能得到非平凡表现；但这种表现不会随着示例数和模型规模持续扩大。真正会随示例数、模型规模提升而放大的，是 TL。

---

## 问题定义与边界

先把四个核心术语定清楚。

| 术语 | 定义 | 保留了什么 | 去掉了什么 |
| --- | --- | --- | --- |
| TR | 只根据示例输入分布和标签分布识别任务类型 | “这是哪类任务”的信息 | 输入和标签的一一对应关系 |
| TL | 利用示例中的输入标签配对学习映射 | 新的映射关系 | 不能脱离配对关系单独存在 |
| RANDOM | 把示例标签随机打乱 | TR 大体还在 | TL 基本失效 |
| ABSTRACT | 把真实标签替换成抽象符号如 `A/B` | TL 还在 | 标签语义先验被削弱 |
| GOLD | 使用真实标签 | TR + TL 都在 | 无 |

Pan 等人的关键形式化写法是：

$$
p_\theta(y \mid x_{\text{test}}, \{x_i,y_i\}_{i=1}^K)
=
p_\theta(y \mid x_{\text{test}}, \{x_i\}_{i=1}^K, \{y_i\}_{i=1}^K)
$$

这条式子描述的是“只有 TR 在起作用”的情形。含义是：模型预测时只用到了输入集合和标签集合的边际信息，而没有真正用到每个 $(x_i,y_i)$ 的配对关系。

边界要注意两点：

1. RANDOM 不等于完全没信息。示例的文本分布还在，标签集合也还在，所以模型仍可能认出任务。
2. ABSTRACT 不等于只有 TR。虽然标签名字变成 `A/B`，但只要配对关系保留，模型仍可进行 TL，只是不能直接借助标签词本身的语义。

一个玩具例子最容易看清这件事。

假设上下文里有 8 个电影评论示例，文本都明显带情感色彩，但你把 `positive/negative` 标签随机打乱，保持正负标签个数还是 4 比 4。测试句子是“我还挺喜欢这部片子”。这时模型即便学不到“哪句对应哪个标签”，也可能仍然判断：这里讨论的是情感任务，且当前句子语义更偏正向，于是输出一个高于瞎猜的结果。这个效果来自 TR，不是 TL。

---

## 核心机制与推导

TR 和 TL 之所以能分开，是因为它们依赖的信息不同。

TR 依赖边际分布。所谓“边际分布”，白话说，就是不管一一配对，只看“这些输入大致像什么”“这些标签大致像什么”。如果输入像商品评论，标签像 `positive/negative`，模型很容易把任务识别为情感分类。这里不要求它知道哪条评论配哪个标签。

TL 依赖联合分布。所谓“联合分布”，白话说，就是必须看成对关系：某种输入模式到底对应哪个标签。只有这一步成立，模型才算真正学会了一个新映射。

可以把两者看成两个阶段：

| 阶段 | 模型在做什么 | 典型信号 |
| --- | --- | --- |
| TR | 识别任务族别，调用预训练先验 | RANDOM 仍有非平凡准确率 |
| TL | 从配对示例里估计映射规则 | GOLD 或 ABSTRACT 随 shot 增长继续提升 |

为什么示例数增加通常更利于 TL，而不是 TR？因为 TR 更像“模式识别触发器”。一旦上下文里有足够多的线索让模型认出“这是情感分类”，继续再塞更多同类例子，收益就会迅速下降。TL 不一样，它要从有限样本中估计映射，更多配对样本往往意味着更稳的归纳。

这里还要澄清一个常见混淆：Zhao 等人 2024 提出的 PIR，全称是 Peak Inverse Rank，不是“中间表示”本身。它是一个衡量任务识别强度的指标。工程上常把同一个指标通过 logit lens 投影到各层隐藏状态后分别计算，于是得到一条“按层展开的 PIR 曲线”。因此大家会直观地说“中层点亮了”，但严格说，点亮的是“按层计算后的任务识别指标”，不是 PIR 的全称发生了变化。

如果画一条示意曲线，可以这样理解：

- 早层：主要编码词形和局部模式，任务信号弱。
- 中层：任务类型开始可分，按层计算的 PIR 明显上升，说明 TR 基本完成。
- 深层：若示例配对有效，模型开始利用更深语义和标签关系，TL 的收益才继续出现。

真实工程例子是少样本客服工单分类。你给模型 8 个“退款/发票/物流”示例，即使标签被随机打乱，模型仍可能从文本分布判断“这是工单分类”，并把“发票怎么开”归到一个和发票相关的标签上。可如果你新造一套内部编码，比如 `T17/T23/T91`，并要求模型学会哪个意图对应哪个编码，那么只有 TR 不够，必须发生 TL。

---

## 代码实现

下面先给一个可运行的玩具脚本。它不依赖外部模型，但把 TR 和 TL 的区别写成了明确逻辑：TR 只看“是不是情感任务”，TL 才看标签映射。

```python
import random
from collections import Counter

def sentiment_score(text: str) -> int:
    positive_words = {"like", "love", "great", "good", "excellent", "amazing"}
    negative_words = {"hate", "bad", "terrible", "awful", "boring", "poor"}
    words = set(text.lower().split())
    return sum(w in positive_words for w in words) - sum(w in negative_words for w in words)

def recognize_task(demos):
    # TR: 只根据输入分布判断是不是“情感分类”
    sentiment_like = 0
    for x, _ in demos:
        if sentiment_score(x) != 0:
            sentiment_like += 1
    return sentiment_like / len(demos) > 0.5

def learn_label_mapping(demos):
    # TL: 统计“正向文本更常对应哪个标签”
    pos_counter = Counter()
    neg_counter = Counter()
    for x, y in demos:
        if sentiment_score(x) >= 0:
            pos_counter[y] += 1
        else:
            neg_counter[y] += 1
    if not pos_counter or not neg_counter:
        return None
    return {
        "positive": pos_counter.most_common(1)[0][0],
        "negative": neg_counter.most_common(1)[0][0],
    }

def predict(test_x, demos):
    assert recognize_task(demos), "TR failed: demos do not look like sentiment classification"

    mapping = learn_label_mapping(demos)
    polarity = "positive" if sentiment_score(test_x) >= 0 else "negative"

    # 如果 TL 学不出来，就退化为只靠 TR 的默认猜测
    if mapping is None or mapping["positive"] == mapping["negative"]:
        return "POS" if polarity == "positive" else "NEG"
    return mapping[polarity]

gold_demos = [
    ("i love this movie", "POS"),
    ("this is amazing", "POS"),
    ("i hate this film", "NEG"),
    ("this is terrible", "NEG"),
]

random.seed(0)
random_demos = [(x, random.choice(["POS", "NEG"])) for x, _ in gold_demos]

gold_pred = predict("i like this", gold_demos)
rand_pred = predict("i like this", random_demos)

assert gold_pred in {"POS", "NEG"}
assert rand_pred in {"POS", "NEG"}
assert recognize_task(gold_demos) is True
assert recognize_task(random_demos) is True

print("gold prediction:", gold_pred)
print("random prediction:", rand_pred)
```

这段代码的意义不是复现论文数值，而是把逻辑拆开看：

- `recognize_task` 对应 TR。
- `learn_label_mapping` 对应 TL。
- `random_demos` 下 TR 仍可能成功，但 TL 会明显变差。

如果要做一个接近论文风格的实验，最小流程通常如下：

| 步骤 | 做法 | 目的 |
| --- | --- | --- |
| 1 | 准备同一批 few-shot 示例 | 保持输入分布一致 |
| 2 | 构造 RANDOM / ABSTRACT / GOLD 三组 prompt | 控制变量 |
| 3 | 在同一测试集上评估 accuracy | 比较 TR 和 TL |
| 4 | 观察 accuracy 随 shot 数变化 | 判断谁在增长 |
| 5 | 对中间层做 logit lens + PIR | 看 TR 在哪层成形 |

下面给出 Hugging Face 风格的伪实装框架，重点在流程而不是可直接运行的模型配置：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random

def build_prompt(demos, test_x):
    lines = []
    for x, y in demos:
        lines.append(f"Input: {x}\nLabel: {y}\n")
    lines.append(f"Input: {test_x}\nLabel:")
    return "\n".join(lines)

def randomize_labels(demos, label_space):
    return [(x, random.choice(label_space)) for x, _ in demos]

def abstract_labels(demos):
    uniq = sorted({y for _, y in demos})
    mapping = {y: f"L{i}" for i, y in enumerate(uniq)}
    return [(x, mapping[y]) for x, y in demos], mapping

def greedy_label_decode(model, tokenizer, prompt, label_space):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    scores = {}
    for label in label_space:
        token_id = tokenizer.encode(" " + label, add_special_tokens=False)[0]
        scores[label] = logits[token_id].item()
    return max(scores, key=scores.get), scores

def accuracy(preds, golds):
    return sum(p == g for p, g in zip(preds, golds)) / len(golds)

# 真实实验里：
# 1. 同一批 demos 构造 RANDOM / ABSTRACT / GOLD
# 2. 分别预测测试集
# 3. 比较 acc_random, acc_abstract, acc_gold
# 4. 若 acc_random 已接近 acc_gold，说明主要是 TR 在工作
assert accuracy(["a", "b"], ["a", "b"]) == 1.0
```

按论文思路，判断逻辑通常是：

- `acc_random > chance`：说明 TR 在起作用。
- `acc_abstract - acc_random` 随 shot 增大而上升：说明 TL 在增强。
- `acc_gold - acc_abstract` 很大：说明标签词本身语义先验也在帮忙。

如果你还要看“哪一层开始识别任务”，可以用中间层 hidden states 做 logit lens，再对每层计算一个任务识别指标。这里写成简化伪代码：

```python
def layerwise_task_signal(hidden_states, unembed, task_label_token_ids):
    # hidden_states: 每层最后一个位置的表示
    # unembed: LM head
    scores = []
    for h in hidden_states:
        logits = h @ unembed.T
        rank = 1
        target_logit = max(logits[i].item() for i in task_label_token_ids)
        better = sum(v.item() > target_logit for v in logits)
        rank += better
        pir = 1.0 / rank  # Zhao et al. 使用的逆排名思想
        scores.append(pir)
    return scores

assert abs(1.0 / 4 - 0.25) < 1e-9
```

这条 `scores` 序列就能画成工程上常说的“PIR 曲线”。如果中层突然抬升，通常表示 TR 已经形成；如果更深层在 GOLD 或 ABSTRACT 下继续分化，往往说明 TL 开始接管。

---

## 工程权衡与常见坑

真正落地时，最容易犯的错是把“模型答对了”直接等价成“模型学会了新任务”。这在 ICL 里经常不成立。

| 坑 | 后果 | 检测方式 | 补救措施 |
| --- | --- | --- | --- |
| 只看 GOLD 准确率 | 把 TR 误判成 TL | 加 RANDOM / ABSTRACT 对照 | 先做消融再决定是否继续堆 shot |
| 示例数只加不筛 | 上下文更长但映射更乱 | 观察 `acc_gold - acc_random` 是否变大 | 优先提高示例质量和覆盖度 |
| 标签词语义太强 | 标签本身替模型“泄题” | 比较 ABSTRACT 与 GOLD | 对新标签编码做抽象化测试 |
| 只看最终层 | 看不到任务信号何时形成 | 做 layer-wise logit lens / PIR | 判断是 TR 早饱和还是 TL 未启动 |
| 新任务和预训练差太远 | TR 能认任务但不会映射 | RANDOM 有分，GOLD 提升有限 | 改 prompt、换标签设计，必要时微调 |

一个很实用的工业流程是先跑“随机标签基线”。以少样本邮件分类为例，你可以先拿 16 个示例做 RANDOM 版本。如果 RANDOM 和 GOLD 差距很小，说明当前收益主要来自 TR，继续机械地加同类示例，效果大概率有限。此时更值得做的是：

- 让示例覆盖更清晰的边界样本；
- 避免语义过强的标签词；
- 调整 prompt 结构，把任务说明和标签定义写清楚；
- 如果任务本身是内部编码映射，尽早评估是否该转向微调或 adapter。

---

## 替代方案与适用边界

TR/TL 拆分不是唯一分析框架，但它非常适合 few-shot 分类任务。它最适合回答的问题是：模型现在到底是在“认任务”，还是在“学映射”。

常见替代策略如下：

| 方法 | 主要看什么 | 适用场景 | 局限 |
| --- | --- | --- | --- |
| RANDOM / ABSTRACT / GOLD 对照 | 拆出 TR 与 TL | 分类任务、标签清晰 | 需要精心控制变量 |
| Layer-wise PIR 或 logit lens | 任务信号在哪层出现 | 机制分析、模型诊断 | 实现复杂，解释需谨慎 |
| Finetune / Adapter | 直接把映射写进参数 | 稳定高频任务 | 成本更高，灵活性下降 |
| Human-in-the-loop 选例 | 提高示例质量 | 业务数据噪声大 | 人工成本高 |

适用边界也要明确。

1. 标签很少、任务语义接近预训练分布时，TR 往往很强。比如情感分类、主题分类。
2. 标签很多、编码抽象、标签关系复杂时，TL 更关键。比如内部工单路由、多标签医疗编码。
3. 当任务需要严格遵守新映射，而不是调用旧知识时，仅靠 ICL 往往不稳，应尽早比较微调方案。
4. 对生成任务，这个拆分仍有启发，但“标签”不再是离散类别，诊断会更复杂。

一个真实工程例子是医疗文本编码。病历摘要和 ICD 子类编码之间的关系往往不是“看到文本就能凭常识猜”，而是强依赖规范映射。此时如果 ABSTRACT 与 GOLD 差距仍然大，说明标签词语义本身在帮忙，TL 其实还没学稳；如果 RANDOM 还有不低分数，只能说明模型知道“这是医疗分类”，不能说明它学会了医院自己的编码体系。

---

## 参考资料

| 文章 | 会议/年份 | 关键词 | 贡献 |
| --- | --- | --- | --- |
| [What In-Context Learning “Learns” In-Context: Disentangling Task Recognition and Task Learning](https://aclanthology.org/2023.findings-acl.527/) | ACL Findings 2023 | TR、TL、RANDOM、ABSTRACT、GOLD | 提出把 ICL 拆成任务识别与任务学习，并做控制实验区分两者 |
| [Unveiling In-Context Learning: A Coordinate System to Understand Its Working Mechanism](https://aclanthology.org/2024.emnlp-main.689/) | EMNLP 2024 | cognition、perception、PIR | 给出二维坐标系，并提出 Peak Inverse Rank 指标衡量任务识别强度 |
| [In-context learning learns label relationships but is not conventional learning](https://openreview.net/forum?id=YPIA7bgd5y) | ICLR 2024 | label relationships、ICL dynamics | 说明 ICL 确实会利用标签关系，但方式不同于传统参数学习 |
| [Pan et al. 论文 PDF](https://aclanthology.org/2023.findings-acl.527.pdf) | 论文原文 | 公式、实验细节 | 包含 TR 只依赖边际分布的形式化定义与数据集结果 |
| [Zhao et al. 论文 PDF](https://aclanthology.org/2024.emnlp-main.689.pdf) | 论文原文 | PIR、层级分析 | 适合进一步查看任务识别指标如何定义与使用 |
