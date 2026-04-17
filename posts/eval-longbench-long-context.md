## 核心结论

LongBench 是长上下文能力的任务型评测集。任务型评测，意思是不用“模型声称支持多少 token”当结论，而是直接让模型做问答、摘要、检索、代码理解，再看分数。它的价值不在于测试窗口上限，而在于测试有效利用率。有效利用率，意思是上下文变长后，模型还能不能继续找到真正有用的信息。

原始 LongBench 由 21 个数据集、6 类任务组成，覆盖英文和中文，统一成可自动评测的格式。对工程师来说，它像一套“长文档体检表”：输入统一，输出统一，指标也统一，便于横向比较模型、提示词、RAG 流程和长上下文微调方案。

更重要的是，LongBench 不是只问“能不能塞进去”，而是问“塞进去之后还能不能用”。这也是它和很多只看窗口标称长度的讨论之间的本质差别。LongBench-E 与 LongBench v2 进一步把“长度分层”这件事做得更明确：同一个模型，在短样本上可能很好，但在更长的样本上会显著退化，这说明长上下文不是一个开关，而是一条随长度逐步下滑的能力曲线。

一个直接的数值例子来自 LongBench v2：`o1-preview` 在 Short、Medium、Long 三档上的 CoT 准确率分别为 62.6%、53.5%、58.1%。如果把长档相对短档的保留比例记为
$$
u=\frac{58.1}{62.6}\approx 0.93
$$
那么这个 $u$ 就可以直观理解为“上下文拉长后还保住了多少效果”。它不是论文官方指标，但很适合工程上做快速比较。

| 维度 | LongBench 关注点 | 为什么重要 |
| --- | --- | --- |
| 窗口长度 | 模型最多能接收多长输入 | 只能说明“能装下” |
| 任务得分 | F1、ROUGE-L、Accuracy | 才能说明“能不能用” |
| 长度分层 | 短、中、长分桶比较 | 能看退化曲线 |
| 位置偏置 | 开头、结尾、中间的信息利用差异 | 能暴露“lost in the middle” |

---

## 问题定义与边界

长上下文评测要回答的问题不是“模型是否支持 128k 或 1M 上下文”，而是“当文档真的很长时，模型是否还能稳定完成任务”。这里的任务包括单文档问答、多文档问答、摘要、few-shot 学习、合成检索任务和代码理解。few-shot 学习，意思是在提示里先给几个示例，再让模型按同样模式回答。

LongBench 的边界也很明确。

第一，它评测的是长文本理解，不是训练方法本身。也就是说，你可以拿它去比较原生长上下文模型、位置编码扩展、RAG、压缩摘要等不同方案，但它自己不替你决定哪个方案更好。

第二，它主要衡量“给定长上下文后的任务完成度”，不是系统级吞吐。吞吐，意思是每秒能处理多少 token、成本多少、延迟多少。这些是部署问题，不是 LongBench 的核心指标。

第三，它并不完全等于“真实生产流量”。例如 LongBench v2 已经比 v1 更接近真实复杂任务，但它依然是基准集，不会覆盖你公司所有文档格式、噪声类型和错误标签。

从工程视角看，这个边界很重要。假设你在做合规审核：一次输入里同时放入监管政策、会议纪要、内部制度、代码仓库说明。你关心的不是模型能不能接住 100k 字，而是它能不能从这些材料里找出冲突条款、证据位置和结论依据。LongBench 的多文档 QA、长对话理解、代码仓库理解，本质上都在逼近这种问题。

| 任务类型 | 白话解释 | 常见指标 | 代表场景 |
| --- | --- | --- | --- |
| Single-doc QA | 对一篇长文回答问题 | F1 | 报告、论文、政策解读 |
| Multi-doc QA | 综合多篇材料回答问题 | F1 / Accuracy | 合规审核、情报汇总 |
| Summarization | 把长材料压缩成摘要 | ROUGE-L | 会议纪要、政府报告 |
| Few-shot | 在超长提示里学例子再作答 | Accuracy | 规则抽取、格式迁移 |
| Synthetic | 人工构造定位与计数任务 | Accuracy | 测定位能力上限 |
| Code | 在长代码上下文里补全或理解 | Accuracy / Pass 类指标 | 仓库问答、代码导航 |

---

## 核心机制与推导

LongBench 的核心机制可以概括成一句话：先按任务算分，再跨数据集求平均。

设数据集集合为 $D$，每个数据集的分数是 $score_d$，那么总分可写为：
$$
\text{LongBench Score}=\frac{1}{|D|}\sum_{d\in D} score_d
$$

这里的 $score_d$ 不固定。问答常用 F1，F1 可以理解为“答案内容重合得多不多”；摘要常用 ROUGE-L，ROUGE-L 可以理解为“摘要和参考答案在最长公共序列上像不像”；分类或选择题常用 Accuracy，也就是答对比例。

这个设计有两个优点。

一是统一。不同任务可以共存于同一套评测管道。

二是可比较。你可以比较模型 A 与模型 B，也可以比较“原始输入”与“RAG 预检索输入”。

但它也带来一个问题：如果样本长度分布不均，平均分可能被某一段长度主导。LongBench-E 的思路就是把样本按长度分桶，再更均衡地统计。例如 0-4k、4-8k、8k+。这样做的目标不是发明新任务，而是把“长度退化”单独暴露出来。

一个玩具例子最容易说明问题。

假设只有 3 道题：

| 题目 | 长度桶 | 是否答对 |
| --- | --- | --- |
| A | Short | 1 |
| B | Short | 1 |
| C | Long | 0 |

如果直接平均，准确率是 $\frac{2}{3}=66.7\%$。这个结果看起来还行，但它掩盖了一个事实：模型在长样本上完全失败。如果改成按桶平均，则
$$
\text{BucketAvg}=\frac{1.0 + 0.0}{2}=50\%
$$
这时你会立刻看到模型对长上下文的有效利用率很差。

再看真实工程例子。假设你做代码仓库问答，仓库根目录说明在开头，关键实现埋在中间，测试样例在结尾。很多模型会优先利用开头和结尾的信息，忽略中间的核心函数调用链。这种现象就是位置偏置。位置偏置，意思是模型更容易使用某些位置上的信息，而不是所有位置一视同仁。`Lost in the Middle` 的结果说明，模型往往在上下文开头和末尾表现更好，在中间表现最差，这也是很多长窗口模型“看起来很长，实际不稳”的原因。

---

## 代码实现

如果你要自己实现一个 LongBench 风格的最小评测器，核心流程只有四步：

1. 读取样本：`context`、`question`、`answer`、`pred`、`metric`
2. 按任务计算单条得分
3. 按长度分桶
4. 输出总体分数和分桶分数

下面给出一个可运行的 Python 玩具实现。它不复现全部官方指标，但足够说明 LongBench 的统一评测思路。

```python
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class Record:
    context: str
    answer: str
    pred: str
    metric: str  # "accuracy" or "f1"

def normalize(text: str) -> list[str]:
    return text.lower().strip().split()

def f1_score(pred: str, answer: str) -> float:
    p = normalize(pred)
    a = normalize(answer)
    if not p and not a:
        return 1.0
    if not p or not a:
        return 0.0
    common = 0
    used = [False] * len(a)
    for token in p:
        for i, ans_token in enumerate(a):
            if not used[i] and token == ans_token:
                used[i] = True
                common += 1
                break
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(a)
    return 2 * precision * recall / (precision + recall)

def accuracy(pred: str, answer: str) -> float:
    return 1.0 if pred.strip() == answer.strip() else 0.0

def compute_metric(pred: str, answer: str, metric: str) -> float:
    if metric == "accuracy":
        return accuracy(pred, answer)
    if metric == "f1":
        return f1_score(pred, answer)
    raise ValueError(f"unknown metric: {metric}")

def length_bucket(context: str) -> str:
    n = len(context)
    if n < 4000:
        return "short"
    if n < 8000:
        return "medium"
    return "long"

def longbench_style_score(records: list[Record]):
    by_bucket = defaultdict(list)
    all_scores = []
    for r in records:
        score = compute_metric(r.pred, r.answer, r.metric)
        by_bucket[length_bucket(r.context)].append(score)
        all_scores.append(score)

    overall = sum(all_scores) / len(all_scores)
    bucket_avg = {
        k: sum(v) / len(v)
        for k, v in by_bucket.items()
    }
    return overall, bucket_avg

records = [
    Record(context="a" * 1000, answer="cache invalidation", pred="cache invalidation", metric="accuracy"),
    Record(context="b" * 5000, answer="token routing", pred="routing token", metric="f1"),
    Record(context="c" * 10000, answer="42", pred="41", metric="accuracy"),
]

overall, buckets = longbench_style_score(records)

assert round(overall, 4) == round((1.0 + 1.0 + 0.0) / 3, 4)
assert buckets["short"] == 1.0
assert buckets["medium"] == 1.0
assert buckets["long"] == 0.0
print(overall, buckets)
```

这个实现对应的工程含义很直接：先把每条样本转成“单条分数”，再聚合成“总体表现”和“长度表现”。如果你继续往前走，通常会加三层能力：

1. 任务适配器：不同数据集映射到统一字段。
2. 模型调用器：本地模型、API 模型、RAG 流程都走同一接口。
3. 结果分析器：按任务、长度、位置、语言分别统计。

真实工程例子里，你可以把这套逻辑接到 CI。每次你调整 RAG 切块大小、检索条数、上下文压缩比或模型版本，就自动回跑 LongBench 子集。如果总体分数涨了，但 `long` 桶分数跌了，说明你可能优化了短样本，却破坏了长样本的信息保留。

---

## 工程权衡与常见坑

最常见的误判是把“大窗口”当成“强长上下文能力”。这两者不是一回事。窗口大，只表示模型能接收更长输入；长上下文能力强，表示模型能稳定使用长输入里的关键信息。

第二个常见坑是只看总体平均分，不看长度分层。总体分数可能很好看，但长样本已经明显退化。上线后用户往往正好用的是最长、最乱、最接近生产的那部分数据。

第三个坑是忽略位置偏置。`lost in the middle` 不是文学描述，而是很实在的评测现象。信息放在开头或结尾，模型更容易答对；放在中间，性能更容易掉。对工程系统来说，这意味着“把所有检索结果简单拼接进去”并不稳，因为最关键的证据可能刚好落在中间。

第四个坑是把 RAG 当成万能补丁。RAG，意思是先检索，再把检索到的片段交给模型。它通常能降低输入长度、提升弱模型表现，但它解决的是“送什么进模型”，不完全解决“模型如何使用这些信息”。如果检索漏召回、切块断语义、重排把核心证据挤到中间，最后仍然会掉分。

| 方案 | 优点 | 代价 | 对中间丢失问题的缓解 |
| --- | --- | --- | --- |
| 扩展位置编码 | 保持原始长文上下文 | 训练和推理成本上升 | 有帮助，但不保证稳定 |
| 长序列继续微调 | 对目标长度更匹配 | 需要数据和算力 | 通常更有效 |
| RAG / 压缩摘要 | 降低输入负担，便于落地 | 依赖检索质量和切块策略 | 间接缓解，不是根治 |

一个真实工程例子是法务审查。你把政策条文、补充解释、过往案例、合同文本全部给模型。若系统只在首尾信息上表现好，就可能抓住合同开头定义和结尾免责条款，却漏掉正文中部的关键限制条件。这类错误比“完全答不出来”更危险，因为它会给出看似完整、实际漏证据的结论。

---

## 替代方案与适用边界

LongBench 适合做“真实任务上的长上下文基线”，但它不是唯一选择。

如果你最关心的是更长上下文、更多推理、选择题式稳定评测，那么 LongBench v2 更合适。它把任务统一成多选题，长度覆盖从 8k 到 2M 词，适合比较推理模型在不同长度上的退化。

如果你最关心的是“到底是不是在测长上下文能力，而不是模型本来就会这题”，那么 100-LongBench 的思路更值得补充。它强调长度可控和把基础能力与长上下文能力拆开分析，适合做更严谨的 ablation。ablation，意思是固定其他条件，只改一个因素，看它到底带来多大变化。

如果你最关心的是 100k 以上甚至更长窗口的极限评测，可以补充 $\infty$Bench。它更偏极长上下文边界，而不是 LongBench 这种多任务日常场景。

如果你最关心的是长文本生成，而不是检索或问答，那么 LongGenBench 这类生成型基准更贴切。因为很多模型在“从长文中找答案”上还行，但在“根据长文生成结构完整、约束一致的长输出”上会明显退化。

| 基准 | 主要目标 | 长度范围 | 任务形式 | 适用场景 |
| --- | --- | --- | --- | --- |
| LongBench | 真实多任务长文本理解 | 中长上下文 | QA / 摘要 / 代码等 | 建立通用基线 |
| LongBench-E | 看长度退化 | 分桶控制 | 与 LongBench 类似 | 分析有效利用率 |
| LongBench v2 | 更难、更长、推理更深 | 8k 到 2M 词 | 多选题 | 比较推理模型 |
| 100-LongBench | 长度可控，拆分基础能力 | 可控长度 | 真实任务变体 | 做严谨对照实验 |
| $\infty$Bench | 100k+ 极长上下文 | 超长上下文 | 多类任务 | 测窗口极限 |
| LongGenBench | 长上下文生成 | 16k-32k 等生成长度 | 长文本生成 | 测长输出质量 |

实际落地时，一个务实组合通常是：LongBench 做总体验收，LongBench-E 看长度退化，LongBench v2 看深推理，再用你自己的生产样本做最终回归。公开基准负责“可比较”，私有样本负责“真上线”。

---

## 参考资料

- Yushi Bai 等，LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding，ACL 2024  
  https://aclanthology.org/2024.acl-long.172/

- THUDM/LongBench 官方仓库，含 LongBench 与 LongBench-E 说明  
  https://github.com/THUDM/LongBench

- LongBench v2 官网与 leaderboard  
  https://longbench2.github.io/

- Nelson F. Liu 等，Lost in the Middle: How Language Models Use Long Contexts，TACL 2024  
  https://aclanthology.org/2024.tacl-1.9/

- Van Yang 等，100-LongBench: Are de facto Long-Context Benchmarks Literally Evaluating Long-Context Ability?，Findings of ACL 2025  
  https://aclanthology.org/2025.findings-acl.903/

- Xinrong Zhang 等，$\infty$Bench: Extending Long Context Evaluation Beyond 100K Tokens  
  https://arxiv.org/abs/2402.13718

- Xiang Liu 等，LongGenBench: Long-context Generation Benchmark  
  https://arxiv.org/abs/2410.04199
