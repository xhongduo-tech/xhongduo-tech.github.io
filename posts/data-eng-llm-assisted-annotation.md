## 核心结论

LLM 辅助自动标注，本质上不是“让模型替代标注员”，而是把模型放进一条可控的数据生产链：先用 prompt 把任务定义、标签边界、输出格式和 few-shot 示例写清楚，再让模型批量产生初标，最后用置信度门控和人工复核把错误挡在入库之前。

对初级工程师最重要的结论有三条。

第一，Prompt 不是装饰，而是标注规则本身的机器可执行版本。术语“few-shot”就是“给模型看几个带答案的样例”，作用不是增加知识，而是把标签边界钉死。没有边界示例，模型往往会把“看起来相关”误当成“应该打正类”。

第二，自动标注系统的核心不是单次准确率，而是“可审计的批处理稳定性”。也就是说，你要优先关心三件事：输出是否符合 schema、模型是否给出可用置信度、低置信度样本是否能自动流入人工复核。只看整体 accuracy，通常会高估系统可用性。

第三，人机协作通常比“纯 LLM 自动标注”更值得上线。BMC 2025 的精准肿瘤学文献筛选研究里，Human-LLM 协同流程在人工工作量减少约 80% 的同时，最终 F1 达到 0.9583；而且用协同标注数据训练的 BioBERT，效果优于只用 LLM 标注数据训练的版本。这说明 LLM 最适合做“规模扩展器”，不是最终裁决者。

表 1 给出一个最直观的判断框架。

| 指标 | 人工全注 | LLM 预注 + 人工核查 | 备注 |
| --- | --- | --- | --- |
| 人工样本数 | 约 1,800 | 约 274 个正样本再核查 | 人工量下降约 80% |
| 最终 F1 | 0.9583 | 0.9583 | 协同流程接近人工质量 |
| 下游模型训练 | 可用 | 更好 | 人工纠错提升训练数据质量 |
| 扩展速度 | 慢 | 快 | 适合批量滚动生产 |

如果把一句话说透：LLM 辅助自动标注能降本增效，但前提是你把它设计成“结构化输出 + 置信度分流 + 人工闭环”的系统，而不是一个直接返回答案的聊天接口。

---

## 问题定义与边界

“LLM 辅助自动标注”指的是：把待标注样本送给大语言模型，让它按预定义标签体系输出结构化标签、理由和置信度，再根据规则决定是否直接采纳、是否进入人工复核、是否回流改 prompt。

这里有三个边界必须先讲清楚。

第一，它适合“规则能说清”的任务，不适合“规则本身还在争论”的任务。比如“是否提到肿瘤精准医学”这种二分类，边界虽有模糊区，但总体可定义；而“这段评论是否冒犯”往往受文化背景影响更大，分歧本身就是信息，此时不能只追 majority label。

第二，它更适合低阳性率、样本量大、人工成本高的场景。低阳性率的意思是正类少、负类多。因为大多数简单负样本可以被高置信度自动过滤，人力集中到少量边界样本上，系统才真正省钱。

第三，LLM 的“confidence”不能直接当概率真值。这里的置信度更接近“系统自报把握”或“由 logprob / 多模型一致性推导出的可排序分数”，它适合做路由，不适合未经校准就直接解释为“95% 一定正确”。

玩具例子可以这样理解。你要给新闻句子打标签：是否涉及“精准肿瘤治疗”。

输入句子 A：
“研究显示某肺癌患者携带 EGFR 突变后，对靶向药反应更好。”

输入句子 B：
“某医院新开设肿瘤营养门诊，提升患者饮食管理效率。”

A 应该是正类，因为它涉及基因突变与治疗匹配；B 通常应是负类，因为虽然提到肿瘤，但不属于精准医学。真正难的是介于两者之间的句子，例如只提到“生物标志物”但没有明确治疗决策关系。这个“边界样本”就是 few-shot 示例最该覆盖的部分。

工程上常用一个门控规则：
若某批样本的平均置信度低于阈值 $\tau$，就扩大人工复核范围；否则允许更多自动入库。写成公式就是：

$$
\text{expand}_i =
\begin{cases}
1, & \text{if } \operatorname{mean}_i(C) < \tau \\
0, & \text{otherwise}
\end{cases}
$$

这里的 $\tau$ 是人工介入阈值，白话说就是“系统一旦明显没把握，就别硬自动化”。

---

## 核心机制与推导

一个可上线的自动标注流程，通常由四层组成：任务定义、输出约束、置信度估计、人工升级。

先看任务定义。你不能只写“请判断是否相关”，而要写成可执行规则：

1. 标签集合是什么。
2. 每个标签的判定条件是什么。
3. 哪些情况容易误判。
4. 输出必须长成什么样。

“schema”就是“输出字段的固定结构”。例如：

```json
{"label":"yes|no","confidence":0.0,"evidence":"一句话证据"}
```

这一步的价值不在美观，而在于机器后处理能稳定解析。没有 schema，后面的批处理、监控、回查都做不稳。

再看 few-shot。它的重点不是给很多例子，而是给“决策边界上的少量高价值例子”。对初学者最实用的经验是：2 到 6 个样例通常已经能明显改善一致性，但这几个例子必须覆盖“显然正类、显然负类、最容易误判的边界类”。

再看置信度。最朴素的办法是让模型直接输出一个 0 到 1 的分数；更稳一点的办法是结合 logprob、温度设定、重复采样一致性或多模型投票。一种常见做法是用 top-1 和 top-2 标签概率差作为 margin 信心：

$$
C(x)=|P(t^\*)-P(t_{runner})|
$$

其中 $t^\*$ 是最高分标签，$t_{runner}$ 是第二高标签。白话说，如果第一名和第二名差距很大，模型更像“明确判断”；差距很小，说明它在犹豫。

当任务规模更大时，可以做多级路由。设有 $m$ 层模型，从便宜到昂贵依次处理。第 $i$ 层处理当前样本集合 $S_i$，按置信度从高到低排序，只保留前 $\alpha_i$ 比例，其余样本转发到下一层。一个常见启发式写法是：

$$
\alpha_i=\frac{m-i+1}{m}
$$

它不是唯一正确答案，而是一个简单可实现的工程近似：前层尽量多吃掉高把握样本，后层只处理更难样本。若样本 $x_j$ 在第 $i$ 层得到标签 $y_{j,i}$ 和置信度 $C_{j,i}$，则可以按排名或标准化分数选最终结果：

$$
y_j^\* = y_{j,i^\*}, \quad i^\*=\arg\max_i R_{j,i}
$$

这里 $R_{j,i}$ 可以理解成“跨层比较后的可信排名”。白话说，就是便宜模型先做大盘筛选，贵模型只接难题，最后取最可信的那次判断。

真实工程例子比玩具例子更说明问题。比如“筛选肿瘤临床试验摘要”时，便宜模型先做粗筛，只判断是否可能涉及 biomarker-guided treatment；中等模型再看边界样本；最贵模型只处理低一致性样本。若任一层输出格式错误、证据字段缺失或 confidence 低于 0.95，就强制进入人工复核。这样做的关键收益不是“单条最准”，而是“把最贵的人力留给最不确定的样本”。

---

## 代码实现

下面给一个最小可运行的 Python 示例。它不依赖真实 API，而是演示“自动接收 + 人工复核”的核心逻辑。术语“校准”指的是“让置信度分数和真实正确率更接近”，这个玩具版本只做路由，不做概率校准。

```python
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Sample:
    text: str
    gold: str

def mock_llm_predict(text: str) -> Dict[str, object]:
    text_lower = text.lower()
    if "egfr" in text_lower or "biomarker" in text_lower or "precision oncology" in text_lower:
        return {"label": "yes", "confidence": 0.97, "evidence": "mentions biomarker-guided treatment"}
    if "nutrition" in text_lower or "ward" in text_lower:
        return {"label": "no", "confidence": 0.98, "evidence": "operational care, not precision medicine"}
    return {"label": "no", "confidence": 0.72, "evidence": "insufficient precision-oncology signal"}

def annotate_batch(samples: List[Sample], threshold: float = 0.95):
    auto_labels = []
    requires_review = []

    for sample in samples:
        pred = mock_llm_predict(sample.text)
        record = {
            "text": sample.text,
            "gold": sample.gold,
            "label": pred["label"],
            "confidence": pred["confidence"],
            "evidence": pred["evidence"],
        }
        if pred["confidence"] >= threshold:
            auto_labels.append(record)
        else:
            requires_review.append(record)

    return auto_labels, requires_review

dataset = [
    Sample("EGFR mutation guides targeted therapy selection", "yes"),
    Sample("Hospital nutrition service for cancer patients", "no"),
    Sample("Cancer study reports survival benefit", "no"),
]

auto_labels, requires_review = annotate_batch(dataset, threshold=0.95)

assert len(auto_labels) == 2
assert len(requires_review) == 1
assert auto_labels[0]["label"] == "yes"
assert auto_labels[1]["label"] == "no"
assert requires_review[0]["confidence"] < 0.95
```

这个例子对应的工程化扩展一般有四步。

1. Prompt 固化。把任务定义、标签说明、few-shot、边界条件写进模板。
2. 输出校验。检查 JSON 是否可解析、字段是否齐全、标签值是否在允许集合内。
3. 置信度分流。高置信度自动入库，低置信度进入 `requires_review`。
4. 反馈回流。把人工纠错样本加入示例池，定期重写 prompt 或更新小模型。

如果把 prompt 写成真正可用于 API 的样子，结构通常如下：

- system：你是数据标注器，只能输出合法 JSON。
- user：任务定义、标签说明、反例说明、few-shot 示例、当前样本。
- output rule：只能返回 `label/confidence/evidence` 三个字段。

对新手而言，最容易忽略的一点是：`evidence` 字段不是为了可读性，而是为了审计。没有证据字段，人工复核要重新读原文，速度会明显下降。

---

## 工程权衡与常见坑

第一类坑，是把模型一致性误当成数据真值。“一致性”指不同标注者或不同模型是否给出相同标签。常用指标如 Cohen’s $\kappa$，白话说就是“扣掉随机一致后的真实一致程度”。如果任务本身高度主观，LLM 可能很擅长复现主流标签，却不擅长保留人类分歧。OpenReview 2025 关于 annotator disagreement 的研究指出，LLM 在预测“人类会不会分歧”这件事上明显更弱，因此只保留单一标签，会丢掉数据中的不确定性信息。

第二类坑，是把模型自报置信度当作真实概率。很多系统里 `confidence=0.99` 只是“模型说自己很有把握”，并不等于 99% 正确率。解决办法通常是留出一小批人工验证集，估计不同置信度桶的真实准确率，再决定阈值。

第三类坑，是只算 API 单价，不算全链路成本。真实成本包括重试、格式错误、人工复核、缓存失效、错误入库后的返工成本。下面这张表更适合工程决策。

| 项目 | 典型值 | 说明 |
| --- | --- | --- |
| 人工标签成本 | 约 \$5/条、\$50/小时 | 质量高，但扩展慢 |
| LLM 输出成本 | 约 \$3-\$75 / 百万输出 token | 高端模型贵，mini 便宜很多 |
| 置信度驱动人工量 | 可减少 25% 以上人工标注 | 取决于任务与校准质量 |
| 分歧建模风险 | 主观任务中较高 | 单一标签会抹平争议 |

第四类坑，是忽略类别分布。若阳性样本本来就很多，例如超过 50%，那“先自动打标、再重点复核正类”未必省钱，因为大量样本都会落入人工核查。LLM 辅助标注最划算的往往是“海量、稀疏、边界可定义”的筛选任务，而不是“主观、高争议、高阳性率”的判断任务。

第五类坑，是 prompt 版本失控。你一旦开始根据误判不断补样例，prompt 很容易越写越长、越写越乱。工程上必须给 prompt 做版本号、回归测试和变更记录，不然你无法解释“为什么上周通过的样本，这周又失败了”。

---

## 替代方案与适用边界

如果你的目标不是生成训练集，而是做统计推断，比如估计某类观点在语料中的比例，那么 Confidence-Driven Inference 更合适。它不是简单地“把高置信度样本直接当真”，而是用 LLM 置信度来决定应该在哪些样本上花人工预算，从而保证置信区间覆盖率。Gligorić 等人在 NAACL 2025 的结果表明，这类方法能在减少人工标注的同时，保持有效的统计结论。

如果你的任务高度主观，例如毒性、冒犯、立场、偏见，另一种更稳妥的路线是“多标注者分布建模”。也就是不只记录单标签，而是记录多个标注者的分布，必要时保留争议。此时 LLM 可以做预标注或提议理由，但不应直接替代人类分歧本身。

还有一种常见替代方案是“主动学习 + 全人工”。主动学习的意思是“优先挑最有信息量的样本给人标”。当任务定义不稳定、标签空间变化快、合规要求高时，这往往比 LLM 自动标注更稳。

| 策略 | 适用场景 | 优点 | 风险/边界 |
| --- | --- | --- | --- |
| LLM + 置信度路由 | 大样本、低阳性率、规则清楚 | 降本快、吞吐高 | 置信度可能失真 |
| Confidence-Driven Inference | 目标是统计结论 | 能控制推断有效性 | 需要信心校准 |
| 多标注者分布建模 | 主观任务、争议重要 | 保留分歧信息 | 成本更高 |
| 主动学习 + 人工 | 高风险、高变动任务 | 稳定、可解释 | 扩展速度慢 |

所以边界判断可以压缩成一句话：如果你要的是“大规模、可控、可审计的初标生产”，选 LLM 辅助自动标注；如果你要的是“保留人类分歧”或“保证统计推断有效”，就不要把它设计成单一自动分类器。

---

## 参考资料

| 参考 | 内容焦点 | 说明 |
| --- | --- | --- |
| Chen et al., 2025, BMC Medical Research Methodology | 人机协同文献筛选 | 精准肿瘤学场景下，人工量下降约 80%，最终 F1 为 0.9583。https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-025-02674-3 |
| Gligoric et al., 2025, NAACL | Confidence-Driven Inference | 用 LLM 置信度选择人工标注点，减少人工同时维持有效置信区间。https://aclanthology.org/2025.naacl-long.179/ |
| Ni et al., 2025, OpenReview | 人类分歧建模 | 指出 LLM 在捕捉 annotator disagreement 上存在明显局限。https://openreview.net/pdf?id=SfMoAphDly |
| Ye et al., 2025, ACL Workshop | Structured Outputs | 说明结构化输出能提升指令遵循与结果可控性。https://aclanthology.org/2025.wasp-main.13/ |
| Stockyard, 2026 | API 成本对比 | 汇总多家模型的输入输出 token 定价，适合做预算估算。https://stockyard.dev/blog/llm-api-pricing-2026/ |
