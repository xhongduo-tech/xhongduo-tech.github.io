## 核心结论

LLM 合成数据生成，指的是用已有生成能力的模型，基于少量人工样本自动扩展出更多可训练样本。它不是“让模型随便编一些数据”，而是把模型当作数据工厂，批量生产指令数据、对话数据和推理数据，再通过筛选把能用于训练的部分留下来。

这件事成立的前提有两个。第一，人工种子样本要足够典型。种子，白话说，就是最开始那一小批人工写的示范样本。第二，任务边界要足够明确。边界，白话说，就是你提前规定“什么算对、什么不算对”，否则模型只会把错误风格批量放大。

工程上最重要的结论不是“能不能生成”，而是“怎么避免生成坏数据”。常见稳定做法是三段式流水线：

1. 先生成：Self-Instruct 扩展任务，Evol-Instruct 提高任务复杂度，对话模板扩展多轮轨迹，CoT 再生成补足推理样本。
2. 再过滤：规则过滤、分类器过滤、LLM 评判三层串联。
3. 最后混合：不要用纯合成数据直接训练，保留至少 $20\%\sim30\%$ 的真实数据作为锚点。

真实数据锚点的意思很直白：真实样本像重力，能把训练分布拉回真实世界。若没有这个锚点，模型会逐步偏向自己最熟悉、最容易生成的模式，尾部场景会越来越少，最终出现分布坍缩。分布坍缩，白话说，就是模型只会越来越像它自己，越来越不像真实用户。

下面这个表格可以先建立直观判断：

| 数据方案 | 成本 | 覆盖面 | 尾部场景保留 | 分布坍缩风险 | 适合场景 |
|---|---:|---:|---:|---:|---|
| 真实 Only | 高 | 中 | 高 | 低 | 高风险任务、数据量较小 |
| 混合 + 20%真实 | 中 | 高 | 中高 | 低到中 | 大多数微调与蒸馏 |
| 纯合成 | 低 | 表面高 | 低 | 高 | 仅限早期原型或数据冷启动 |

一个最小玩具例子是：先人工写 5 条高质量指令，再让模型按相同风格生成 200 条相似但不重复的新任务。之后先用规则过滤去掉太短、拒答、格式错乱的样本，再按 IFD 或 Judge 分数选 top 30%。这套方法的关键不在“200 条”，而在“先定风格，再扩展，再筛选”。

一个真实工程例子是：做企业客服助手微调时，真实客服日志只有 3000 条，但产品要覆盖退款、改签、催发货、发票、异常升级等 40 多种意图。此时可以用真实日志抽样出高质量种子，再生成更多同分布对话，补齐长尾问法。真正有效的不是把 3000 条扩到 30000 条，而是把“原来覆盖不到的问法”补齐，同时不把真实语气洗掉。

---

## 问题定义与边界

这类问题的正式定义可以写成一句话：在人工标注稀缺时，用 LLM 构造 instruction-response、multi-turn dialogue、chain-of-thought 等训练样本，以较低成本提升任务覆盖和模型能力。

但这个定义必须带边界，否则会误用。最重要的三个边界如下：

| 检查项 | 你必须先回答的问题 | 不能跨越的边界 |
|---|---|---|
| 问题定义 | 训练目标到底是什么？分类、生成、工具调用还是推理？ | 不能一边生成闲聊，一边想训练严肃问答 |
| 种子来源 | 种子来自人工样本、历史日志还是专家模板？ | 种子若不典型，扩出来的全是错分布 |
| 真实性保底 | 训练集里真实数据最低占比是多少？ | 一般不建议低于 $0.2$ |
| 领域覆盖 | 要覆盖哪些意图、语气、难度、异常场景？ | 不能只覆盖主流路径，忽略尾部 |
| 偏见控制 | 是否会放大固定表达、刻板角色或安全问题？ | 不能默认“模型自己会均衡” |

零基础读者最容易忽略的是：合成数据不是替代任务定义，而是放大任务定义。你原来任务定义模糊，放大后只会更模糊。

举一个新手版例子。假设你做客服意图分类，先把范围限定为 4 类：退款、物流查询、修改地址、投诉升级。你可以先为这 4 类各写几段典型对话，再让模型生成变体。这时模型是在已有意图空间内做扩展。如果你一开始不限定类别，模型可能会生成“优惠券咨询”“会员规则”“门店营业时间”等新意图，导致训练目标漂移。漂移，白话说，就是你本来想训练 A 任务，结果数据慢慢变成了 B 任务。

因此，合成数据适合“任务清楚、格式清楚、评估标准清楚”的场景，不适合“我先生成一些再看看能训练什么”的场景。

这里还要区分三类常见目标：

| 样本类型 | 目标 | 典型格式 | 风险 |
|---|---|---|---|
| 指令数据 | 学会按要求完成任务 | instruction + output | 风格单一、答案模板化 |
| 对话数据 | 学会多轮交互与状态跟踪 | messages 列表 | 轮次虚假自然、用户意图不真实 |
| 推理链数据 | 学会中间步骤或复杂求解 | question + reasoning + answer | 容易把错误推理也学进去 |

如果任务是事实敏感型，例如法务问答、医疗信息、企业知识库检索，那么不能单靠模型自由生成答案。因为这类任务不是“语言看起来像”，而是“事实必须对”。这时合成流程通常要接上检索或执行器，而不是只靠一个生成模型闭门造车。

---

## 核心机制与推导

从统计角度看，真实数据来自分布 $P$，合成数据来自分布 $Q$。分布，白话说，就是“样本在各种类型、难度、风格上的整体概率结构”。如果 $Q$ 和 $P$ 差太远，训练出来的模型自然也会偏。

常见偏移衡量方法之一是 KL 散度：

$$
D_{KL}(P\parallel Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}
$$

它可以理解为：如果真实世界是 $P$，你却用 $Q$ 去近似，会多付出多少信息代价。KL 越大，说明合成分布越不像真实分布。

在工程里不一定真的逐 token 精确计算 KL，更常见的是用一组代理特征来估计分布是否偏了，例如：

- 意图类别比例
- 问句长度分布
- 对话轮数分布
- 拒答率
- 工具调用比例
- 推理步数分布

Self-Instruct 的机制很简单：给模型少量高质量指令样本，让它模仿任务风格生成更多指令。Evol-Instruct 则是在已有指令上施加“变换算子”。算子，白话说，就是一组固定改写动作，比如增加约束、提高难度、要求多步推理、引入格式限制。这样能把简单任务一步步演化成更复杂任务。

可以把它写成一个简单过程：

1. 从人工种子集合 $S_0$ 出发。
2. 用生成器 $G$ 产生候选样本 $C=G(S_0)$。
3. 用演化算子 $T$ 得到更复杂的候选 $C'=T(C)$。
4. 用过滤器 $F$ 保留高质量子集 $S_1=F(C\cup C')$。
5. 将 $S_1$ 与真实集 $R$ 按比例混合训练。

真实比例通常写成：

$$
r_{\text{real}}=\frac{N_{\text{real}}}{N_{\text{real}}+N_{\text{synthetic}}}
$$

当 $r_{\text{real}}$ 太低时，模型容易过度吸收合成风格，削弱真实场景下的鲁棒性。鲁棒性，白话说，就是输入稍微变化时，模型还能稳定工作。

看一个最小数值例子。假设你有 10000 条合成样本和 3000 条真实样本。生成后经过规则、分类器、Judge 三层过滤，最后留下 8600 条高质量合成数据，那么：

$$
r_{\text{real}}=\frac{3000}{3000+8600}\approx 0.26
$$

这个比例处在常见推荐区间内，说明训练集里仍有足够的真实样本作为锚点。

这里还有一个关键推导：为什么纯合成数据会伤害 tail 行为？tail，白话说，就是少见但重要的长尾场景。原因在于生成模型天然偏向高概率模式。它最容易生成“看起来合理、最常见”的样本，而不是“少见但关键”的样本。多轮自举后，这种偏好会被不断放大。于是主流样本越来越多，尾部样本越来越少，训练分布不断收缩。

因此，合成数据的核心矛盾不是“量够不够”，而是“覆盖是否真实”。量只是手段，覆盖才是目标。

---

## 代码实现

下面给一个可运行的最小 Python 示例，模拟一个“Self-Instruct + Evol-Instruct + 规则过滤 + Judge 打分 + 真实数据混合”的简化流水线。它不是调用真实 LLM API，而是用假模型说明结构，便于直接改写成你的生成脚本。

```python
from dataclasses import dataclass
from typing import List, Dict
import random

random.seed(42)

SEEDS = [
    "总结一篇技术文章的核心观点",
    "解释 Python 中列表推导式的用途",
    "把一段客服对话改写成标准问答格式",
    "比较 SQL 和 NoSQL 的主要差异",
    "给出一个二分查找的边界条件说明",
]

REAL_SAMPLES = [
    {"instruction": "用户要退款但订单已发货，给出客服回复", "quality": 0.95, "source": "real"},
    {"instruction": "解释 HTTP 404 和 500 的区别", "quality": 0.92, "source": "real"},
    {"instruction": "将多轮对话整理为 FAQ", "quality": 0.90, "source": "real"},
]

OPERATORS = {
    "deepen": "要求增加中间推理步骤",
    "constraint": "要求输出必须包含表格或编号步骤",
    "realistic": "要求加入真实业务限制",
    "edge_case": "要求覆盖异常和边界情况",
}

@dataclass
class Sample:
    instruction: str
    output: str
    score: float
    source: str  # synth or real

class MockModel:
    def generate(self, prompt: str) -> str:
        return f"回答：{prompt}。请给出清晰步骤和最终结论。"

def self_instruct(seed: str, model: MockModel, n: int = 3) -> List[str]:
    results = []
    for i in range(n):
        results.append(f"{seed}（变体{i+1}，要求更具体）")
    return results

def evolve_instruction(instruction: str, operator: str, model: MockModel) -> str:
    return f"{instruction}；并且{OPERATORS[operator]}"

def rule_filter(instruction: str, output: str) -> bool:
    banned = ["抱歉，我不能", "作为AI", "无法提供"]
    if len(instruction.split()) < 1:
        return False
    if any(x in output for x in banned):
        return False
    if len(output) < 12:
        return False
    return True

def judge_score(instruction: str, output: str) -> float:
    score = 0.5
    if "边界" in instruction or "异常" in instruction:
        score += 0.1
    if "步骤" in output:
        score += 0.1
    if "结论" in output:
        score += 0.1
    score += min(len(instruction) / 80, 0.2)
    return round(min(score, 0.99), 2)

def build_synth_dataset(seeds: List[str], model: MockModel) -> List[Sample]:
    samples = []
    for seed in seeds:
        expanded = self_instruct(seed, model, n=4)
        for ins in expanded:
            op = random.choice(list(OPERATORS.keys()))
            evolved = evolve_instruction(ins, op, model)
            output = model.generate(evolved)
            if rule_filter(evolved, output):
                score = judge_score(evolved, output)
                samples.append(Sample(evolved, output, score, "synth"))
    return samples

def top_percent(samples: List[Sample], keep_ratio: float) -> List[Sample]:
    samples = sorted(samples, key=lambda x: x.score, reverse=True)
    k = max(1, int(len(samples) * keep_ratio))
    return samples[:k]

def mix_real_and_synth(real_n: int, synth: List[Sample], r_real_target: float) -> Dict[str, int]:
    max_synth = int(real_n * (1 - r_real_target) / r_real_target)
    used_synth = min(len(synth), max_synth)
    total = real_n + used_synth
    r_real = real_n / total
    return {"real": real_n, "synth": used_synth, "total": total, "r_real": round(r_real, 4)}

model = MockModel()
raw_synth = build_synth_dataset(SEEDS, model)
selected = top_percent(raw_synth, keep_ratio=0.3)
mix_stats = mix_real_and_synth(real_n=len(REAL_SAMPLES), synth=selected, r_real_target=0.25)

assert len(raw_synth) > 0
assert len(selected) <= len(raw_synth)
assert mix_stats["r_real"] >= 0.25
assert mix_stats["total"] == mix_stats["real"] + mix_stats["synth"]

print("raw_synth =", len(raw_synth))
print("selected =", len(selected))
print("mix_stats =", mix_stats)
```

这个脚本体现了四个关键点。

第一，Self-Instruct 负责横向扩展。也就是把已有任务风格铺开，生成更多同类任务。

第二，Evol-Instruct 负责纵向加深。也就是在原任务上增加约束、多步推理和边界条件。

第三，过滤必须独立于生成。生成器负责“多产”，过滤器负责“挑对”。这两个角色混在一起时，通常会造成“自己给自己放水”。

第四，混合比例最后计算，而不是一开始拍脑袋决定。你应该先看筛完还剩多少高质量合成样本，再反推可用混合量。

如果改成真实工程流水线，通常是下面这个结构：

1. 从真实日志或专家模板中抽取 50 到 500 条种子。
2. 用 few-shot prompt 扩展出 instruction-response 或多轮 messages。
3. 对一部分样本再做 evol 操作，补足复杂度。
4. 跑规则过滤。
5. 跑轻量分类器或去重模型。
6. 跑 LLM judge，得到质量分。
7. 分桶抽样，保证不同类别、难度、风格都有覆盖。
8. 按目标比例与真实样本混合。
9. 小规模训练后回看错误案例，继续补种子，而不是盲目继续扩量。

一个真实工程例子是代码推理或工具使用任务。比如你要训练一个能回答数据分析问题的模型，不只需要“问题-答案”，还需要“问题-代码-执行结果”三元组。此时可以先生成问题，再让模型生成代码，执行代码得到结果，最后由 judge 检查“问题是否可解、代码是否可运行、答案是否与执行结果一致”。这比只生成文本问答更可靠，因为执行器提供了额外监督信号。

---

## 工程权衡与常见坑

合成数据项目里，最常见的失败不是模型不够强，而是数据管线看起来完整，实际没有真正控制质量。

建议把质量控制拆成三层：

| 层级 | 做什么 | 典型规则 | 成本 | 能发现什么 |
|---|---|---|---:|---|
| 规则过滤 | 去掉明显坏样本 | 长度、格式、拒答、敏感词、空输出 | 低 | 脏数据、格式错 |
| 分类器过滤 | 批量打分或去重 | 相似度、意图分类、异常检测、IFD | 中 | 重复、离题、低信息量 |
| LLM Judge | 做语义级评估 | 是否可答、是否一致、是否完整 | 高 | 逻辑错、风格错、隐性低质 |

IFD 可以粗略理解为信息密度分数，也就是“这条样本到底有没有足够训练价值”。有些样本看起来很长，但信息量很低；有些样本很短，但表达精确。长度不是质量，信息密度才更接近质量。

常见坑和规避方式如下：

| 常见坑 | 具体表现 | 规避方式 |
|---|---|---|
| 过度自我训练 | 数据越来越像模型自己的说话方式 | 保留至少 20% 真实数据，周期性监测分布偏移 |
| 演化过度 | 指令被不断加约束，最后互相矛盾 | 加 coherence filter，限制演化深度 |
| 只追求量 | 生成很多，但训练后效果不升反降 | 先做小样本实验，观察高质量样本增益 |
| Judge 不一致 | 同一条样本多次评分差异大 | 多评一轮，做平均或投票 |
| 去重不足 | 表面不同，实质重复 | 用 embedding 相似度做聚类去重 |
| 尾部缺失 | 主流问法很多，异常场景很少 | 按类别和难度分桶采样，不按总分一刀切 |
| 事实幻觉 | 答案流畅但与资料不符 | 事实任务接 RAG 或执行器，不做闭环自由生成 |

这里给一个新手很容易上手的筛选策略：

1. 规则层先删掉少于 3 个词的指令、明显拒答、格式不完整样本。
2. 分类器层去掉重复任务和离题样本。
3. Judge 层要求“任务清晰、答案一致、可训练性高”三个维度平均分至少 4/5。

这类三段式流程看起来保守，但比“让一个更强的大模型直接生成并自评”稳定得多。原因很简单：单模型闭环容易产生相关性偏差。相关性偏差，白话说，就是生成时犯的错，评估时也可能继续同样看错。

还要注意一个经常被忽略的问题：合成数据会放大风格偏见。例如你所有种子都来自非常正式的技术说明，那么生成出的对话很可能过度书面化，导致模型在真实用户的口语输入上表现变差。因此分桶设计种子时，要把语气、长度、难度、角色都考虑进去。

---

## 替代方案与适用边界

Self-Instruct 和 Evol-Instruct 很常见，但不是所有任务都适合。不同方法解决的是不同问题。

| 合成策略 | 优势 | 适合场景 | 最小真实比例建议 |
|---|---|---|---:|
| Self-Instruct | 简单、便宜、易扩量 | 通用指令微调、冷启动 | 0.2 |
| Evol-Instruct | 能补复杂度和多步任务 | 推理、复杂问答、步骤性任务 | 0.25 |
| Persona 生成 | 能覆盖不同用户语气和角色 | 客服、助理、教育对话 | 0.25 |
| Magpie 类方法 | 适合大规模开放式任务扩展 | 广义指令、多风格探索 | 0.3 |
| RAG 驱动合成 | 事实更稳，可绑定文档 | 企业知识库、医疗、法务 | 0.3 |
| QA-code-exec 流水线 | 可执行验证强 | 代码、数学、数据分析 | 0.3 |

RAG，白话说，就是先检索相关资料，再基于资料生成答案。它适合事实敏感领域，因为模型不是凭记忆乱写，而是先“查资料再作答”。

对新手来说，一个很实用的判断标准是：

- 任务主要考表达和格式，用 Self-Instruct。
- 任务主要考复杂度和多步约束，用 Evol-Instruct。
- 任务主要考用户风格和角色差异，用 Persona。
- 任务主要考事实正确性，用 RAG 驱动合成。
- 任务主要考程序可执行性，用 QA-code-exec 或类似 LoongEnv 管线。

举一个事实任务的入门例子。假设你想生成“公司报销制度问答”数据。错误做法是直接让模型编 500 条报销问答。正确做法是先检索报销制度文档，拿到相关段落，再让模型根据文档生成问题和标准答案。之后保留“问题-文档片段-答案”三元组。这样后续即使模型答错，也能追溯依据。

再看适用边界。合成数据不适合替代以下内容：

- 高风险事实标注，例如医疗诊断结论
- 稀有异常数据，例如极少见的线上事故日志
- 需要人工价值判断的数据，例如复杂审核尺度
- 法规或制度快速变化的数据

这些场景里，模型可以辅助扩写、改写、重组，但不应成为最终真值来源。真值，白话说，就是训练时被当作标准答案的内容。

最终选型原则很简单：如果你能定义“什么样本是好样本”，合成数据就适合进入流水线；如果你连“好坏标准”都说不清，先别扩量，先补任务定义。

---

## 参考资料

- “How to Generate Synthetic Training Data for LLM Fine-Tuning (2026 Guide)” – Premai 博客，涵盖 Self-Instruct、Evol-Instruct、质量过滤、真实数据比例与模型坍缩风险。https://blog.premai.io/how-to-generate-synthetic-training-data-for-llm-fine-tuning-2026-guide/
- “Synthetic Data Generation with LLMs” – Next Electronics，讨论合成分布评估、KL/JS 等指标与实践建议。https://www.next.gr/ai/generative-vision-models/synthetic-data-generation-with-llms
- “Synthetic Data: Generation, Quality, Diversity, Distillation” – Michael Brenndoerfer，讨论多阶段质量验证、分布坍缩与偏见放大。https://mbrenndoerfer.com/writing/synthetic-data-generation-quality-diversity-distillation-llm
- “LoongEnv / LoongBench” – EmergentMind，介绍 QA-code 三元组生成、judge 验证和可执行反馈式训练。https://www.emergentmind.com/topics/loongenv
- “CoT-Self-Instruct” – OpenReview，讨论基于链式思考轨迹的再生成与训练信号扩展。https://openreview.net/pdf?id=nPEWyL8kxO
