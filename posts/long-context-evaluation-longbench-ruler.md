## 核心结论

LongBench 和 RULER 不在回答同一个问题。

LongBench 是**真实任务导向评测**。真实任务导向，意思是它尽量用接近实际应用的任务来测模型，比如问答、摘要、多文档检索、few-shot 学习和代码补全。它更适合回答：这个模型放进业务流程后，整体有没有工作能力。

RULER 是**诊断导向评测**。诊断导向，意思是它不先追求任务多样性，而是优先拆开“长上下文到底哪里失效”，用长度扫描和合成任务定位模型的退化边界。它更适合回答：这个模型在 4K、32K、64K、128K 输入下，检索、跟踪、聚合能力还能不能稳定保持。

可以把两者压成一句话：

$$
\text{LongBench 看综合能力，RULER 看长度退化边界}
$$

对初级工程师，最实用的判断是：

| 问题 | 先看什么 |
|---|---|
| 这个模型能不能做实际工作 | LongBench |
| 这个模型到底能撑多长输入 | RULER |
| 这个模型适不适合我的业务 | 两者都看，再加真实业务集 |

玩具例子很简单。假设你有两个模型：

- 模型 A 在 LongBench 上高分，但在 64K 长度下开始明显掉点。
- 模型 B 在 RULER 上 64K 仍稳定，但真实多文档问答回答质量一般。

这时结论不是“谁分更高谁更强”，而是“谁更适合你的问题”。一个看业务像不像，一个看上下文到底行不行。

---

## 问题定义与边界

长上下文评测不是测“模型宣称支持多少 token”，而是测“输入变长后，任务能力是否还能保持”。

这里先统一符号：

- $M$：模型
- $T$：任务集合
- $D_t$：任务 $t$ 的样本集合
- $m_t$：任务 $t$ 的得分函数
- $L$：输入长度
- $\tau$：阈值，用来判断能力是否还算可接受

很多新手会把“支持 128K 上下文”理解成“128K 内任意任务都能稳定工作”。这通常不成立。模型能接收长输入，只说明接口层面允许塞进去更多文本；不说明它还能稳定定位证据、保持指令一致性、跨段聚合信息。

下面这张表可以把边界看清楚：

| 维度 | LongBench | RULER |
|---|---|---|
| 评测对象 | 多类真实风格任务 | 长度与能力退化 |
| 输入形式 | 真实数据集、异构任务 | 可控长度的诊断任务 |
| 输出指标 | EM / F1 / ROUGE-L / Accuracy 等 | 准确率或任务得分随长度变化 |
| 主要目的 | 看综合任务能力 | 看有效上下文长度与退化点 |
| 能否直接做业务验收 | 只能部分支持 | 不适合单独验收 |

所以不能拿一个总分替代另一个。

新手版理解是：

- 把一篇 50K 文档丢给模型，不是看它“吃没吃下去”，而是看它还能不能回答问题、做总结、找证据。
- 如果模型在 4K 还行，32K 开始掉，64K 明显退化，这就是 RULER 要定位的边界。

这也是本文的边界条件：我们讨论的是**长输入下的任务能力保持**，不是模型训练方法，也不是单纯的 token 上限宣传。

---

## 核心机制与推导

LongBench 的核心机制是**任务级聚合**。任务级聚合，意思是先分别评估多个不同任务，再把这些任务结果做平均，得到综合分数。它的总分可以写成：

$$
S_{\text{LB}}(M)=\frac{1}{|T|}\sum_{t\in T} m_t(M;D_t)
$$

这里最关键的一点是：$m_t$ 不是统一指标。问答任务可能用 EM 或 F1，摘要常用 ROUGE-L，分类或选择题可能用 Accuracy。也就是说，LongBench 的“平均分”本质上是异构任务的聚合结果。

这很像期末总评。语文、数学、英语、编程都要算，最后看综合成绩。它适合做横向比较，但会天然掩盖细节：你不知道高分是因为所有科目都稳，还是某几科特别强。

LongBench-E 可以理解成 LongBench 的长度视角扩展。它不是只看一个总均值，而是把样本按长度分桶，观察不同长度段的表现变化。这个扩展非常重要，因为很多模型的平均分还可以，但一旦进入更长长度段，分数会快速下滑。

RULER 的核心机制是**长度维度扫描**。长度维度扫描，意思是同一种能力在多个输入长度上重复测量，观察退化曲线。可以把单任务得分写成：

$$
a_t(M,L,c)
$$

其中 $t$ 是任务类型，$L$ 是输入长度，$c$ 是复杂度。复杂度可以理解为任务难度，比如要检索的目标数量、需要跟踪的线索数量、聚合步骤多少。

进一步，把多个任务的平均能力记成 $\bar a(M,L)$，有效上下文长度可以定义为：

$$
L_{\text{eff}}=\max\{L:\bar a(M,L)\ge \tau\}
$$

这表示：在阈值 $\tau$ 之上，模型还能稳定工作的最大长度是多少。

看一个玩具例子。假设某模型在 RULER 上不同长度的平均分是：

| 长度 | 平均得分 |
|---|---|
| 4K | 90.0% |
| 32K | 87.0% |
| 64K | 82.0% |

如果阈值 $\tau=85.6\%$，那么：

$$
L_{\text{eff}}=32K
$$

因为 32K 还在阈值之上，64K 已经掉到阈值以下。

LongBench 和 RULER 的差异，本质上是“横向多任务聚合”和“纵向长度扫描”之间的差异。前者回答“像不像真实工作”，后者回答“长度拉长后会不会坏”。

下面这张表可以把测量重点再压缩一次：

| 评测 | 主要子任务/维度 | 常见指标 |
|---|---|---|
| LongBench | 单文档 QA、多文档 QA、摘要、few-shot、代码补全 | EM / F1 / ROUGE-L / Accuracy |
| RULER | 检索、跟踪、聚合在不同长度下的稳定性 | 准确率或分任务得分曲线 |

真实工程例子更能说明问题。假设你在做“企业知识库 + 代码库助手”：

- LongBench 可以先帮你筛模型，判断它在问答、总结、代码场景是否都不差。
- RULER 再告诉你，当上下文从 32K 拉到 128K 后，模型是不是只是“能塞进更多字”，还是“长输入下仍然可用”。

这两个视角合在一起，才接近工程决策。

---

## 代码实现

评测实现的核心不是训练，而是统一评测流水线。统一评测流水线，意思是你要用同一套脚本完成数据加载、Prompt 构造、模型调用、指标切换和结果聚合。

先看一个可以运行的简化版本：

```python
from statistics import mean

def metric_dispatch(task_type, pred, gold):
    if task_type == "qa":
        return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0
    if task_type == "summary":
        pred_tokens = set(pred.lower().split())
        gold_tokens = set(gold.lower().split())
        if not gold_tokens:
            return 0.0
        return len(pred_tokens & gold_tokens) / len(gold_tokens)
    raise ValueError(f"unknown task type: {task_type}")

def mean_group_by_task(results):
    groups = {}
    for r in results:
        groups.setdefault(r["task"], []).append(r["score"])
    return {k: mean(v) for k, v in groups.items()}

def aggregate_by_length_bucket(results, buckets):
    bucket_scores = {}
    for r in results:
        length = r["length"]
        matched = None
        for upper in buckets:
            if length <= upper:
                matched = upper
                break
        if matched is None:
            matched = "overflow"
        bucket_scores.setdefault(matched, []).append(r["score"])
    return {k: mean(v) for k, v in bucket_scores.items()}

def max_length_where(length_curve, threshold):
    valid = [k for k, v in length_curve.items() if isinstance(k, int) and v >= threshold]
    return max(valid) if valid else 0

tasks = [
    {"task": "single_doc_qa", "type": "qa", "pred": "Paris", "gold": "Paris", "length": 3800},
    {"task": "single_doc_qa", "type": "qa", "pred": "Berlin", "gold": "Paris", "length": 4200},
    {"task": "multi_doc_summary", "type": "summary", "pred": "cache invalidation consistency",
     "gold": "cache invalidation and consistency", "length": 18000},
    {"task": "multi_doc_summary", "type": "summary", "pred": "latency retry",
     "gold": "latency retry timeout", "length": 36000},
]

results = []
for item in tasks:
    score = metric_dispatch(item["type"], item["pred"], item["gold"])
    results.append({
        "task": item["task"],
        "length": item["length"],
        "score": score,
    })

task_scores = mean_group_by_task(results)
length_curve = aggregate_by_length_bucket(results, [4096, 8192, 32768, 65536])
eff_len = max_length_where(length_curve, threshold=0.6)

assert round(task_scores["single_doc_qa"], 2) == 0.5
assert 32768 in length_curve
assert eff_len == 32768
print(task_scores, length_curve, eff_len)
```

这个脚本对应了两个评测的骨架：

1. `metric_dispatch`：按任务类型切换指标。
2. `mean_group_by_task`：做 LongBench 风格的任务级聚合。
3. `aggregate_by_length_bucket`：做 LongBench-E 或 RULER 风格的长度统计。
4. `max_length_where`：根据阈值求有效上下文长度。

任务指标分发表通常要先明确：

| 任务类型 | 指标建议 | 说明 |
|---|---|---|
| 抽取式 QA | EM / F1 | 看答案是否对、是否重合 |
| 摘要 | ROUGE-L | 看关键信息覆盖 |
| 分类/选择 | Accuracy | 看判断是否正确 |
| 检索定位 | Accuracy / Recall | 看目标是否找到 |

长度分桶统计也要提前设计，否则很难对比不同模型：

| 长度桶 | 样本数 | 平均分 |
|---|---:|---:|
| 0-4K | 120 | 0.88 |
| 4K-8K | 110 | 0.84 |
| 8K-16K | 95 | 0.79 |
| 16K-32K | 80 | 0.73 |
| 32K-64K | 60 | 0.61 |
| 64K-128K | 40 | 0.49 |

对 LongBench，重点是“任务分发 + 指标汇总”。对 RULER，重点是“同类任务在不同长度上重复测试，再判断阈值前的最大长度”。

真实工程里，评测脚本还要处理 Prompt 模板、并发调用、重试、截断策略、缓存命中和结果落盘。但骨架通常就是上面这几步。

---

## 工程权衡与常见坑

最常见的错误是只看总分。

LongBench 总分高，不代表超长样本也稳。RULER 长度表现好，也不代表真实业务任务强。两者各自都只覆盖问题的一部分。

下面这张表是工程里最容易踩的坑：

| 常见坑 | 结果 | 规避方式 |
|---|---|---|
| 直接比较 LongBench 和 RULER 绝对分数 | 得出错误强弱结论 | 只在各自框架内比较 |
| 只看平均总分，不看长度分布 | 掩盖长输入退化 | LongBench 看分任务与分桶，RULER 看长度曲线 |
| 忽略任务异构指标 | 把不同比分直接混成一个感觉 | 明确每个子任务的指标定义 |
| 忽略长文本截断策略 | 结果不可复现，甚至方向错误 | 固定截断、拼接与窗口策略 |
| 把 needle in a haystack 当完整能力 | 高估模型长上下文理解 | 结合聚合、推理、多证据任务一起测 |
| 把输入上限当业务上限 | 线上效果与离线宣传不一致 | 用真实业务集做回归验证 |

这里要明确写出四条硬规则：

- 不要直接比较 LongBench 和 RULER 绝对分数。
- 不要忽略任务异构指标。
- 不要忽略长文本截断策略。
- 不要把“needle in a haystack”当成完整长上下文能力。

工程上还有一个常被忽略的问题：**上下文扩容后的回归测试**。很多团队把模型从 32K 升到 128K，就认为能力自动提升了。实际可能只是接口允许更多输入，真实任务表现反而更不稳定，比如回答更散、引用位置更错、摘要更容易漏关键段落。

真实工程例子可以这样做：你的知识库问答系统升级到更长上下文后，回归测试不该只验证“能否塞入 100 页 PDF”，还要验证：

- 多文档问题的正确率有没有掉。
- 引用证据的位置是否仍准确。
- 超长代码库检索后的回答是否更稳定。
- 响应延迟和成本是否还能接受。

长上下文是能力问题，不只是容量问题。

---

## 替代方案与适用边界

LongBench 适合做综合能力基线、模型横向对比、真实任务预筛选。RULER 适合做诊断、回归和定位退化点，但不适合单独作为业务验收标准。

可以直接看适用边界表：

| 场景 | 适合 LongBench | 适合 RULER | 是否还需要真实业务集 |
|---|---|---|---|
| 模型初筛 | 是 | 可选 | 建议需要 |
| 看长输入退化点 | 一般 | 是 | 视业务而定 |
| 业务上线验收 | 不够 | 不够 | 必须需要 |
| 上下文扩容回归 | 部分适合 | 很适合 | 建议需要 |
| 多任务综合对比 | 很适合 | 一般 | 建议需要 |

对新手最实用的组合是三层：

1. 用 LongBench 做综合筛选。
2. 用 RULER 做长度诊断。
3. 用真实业务集做最终验收。

替代方案也很明确。

第一类是**真实业务数据集**。这是最接近上线效果的方案，比如你自己的客服问答、法务检索、代码审查、研报摘要数据。

第二类是**自建长文 QA 集**。如果你的业务强依赖内部文档、规范、代码仓库，这类数据比公开基准更贴近真实难点。

第三类是**端到端回归测试**。端到端回归测试，意思是不只看模型单条输出，而是看完整流程能否稳定工作，包括检索、重排、生成、引用、前端展示和人工验收。

所以最终结论应该很明确：

- 诊断用 RULER。
- 业务验收用真实任务。
- 综合筛选用 LongBench。

如果只能选一个公开基准，选哪个要看你要解决什么问题；如果要做工程决策，只看一个公开基准通常不够。

---

## 参考资料

1. [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://aclanthology.org/2024.acl-long.172/)
2. [LongBench 官方仓库 README](https://github.com/THUDM/LongBench/blob/main/LongBench/README.md)
3. [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://huggingface.co/papers/2404.06654)
4. [RULER 官方仓库 README](https://github.com/NVIDIA/RULER)
5. [LongBench 项目主页与代码说明](https://github.com/THUDM/LongBench)
