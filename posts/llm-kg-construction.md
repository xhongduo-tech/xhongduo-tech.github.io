## 核心结论

LLM 可以成为知识图谱构建里的“抽取器”，但前提不是让它自由发挥，而是把它放进一条受约束的流水线。更准确地说，这条流水线通常包含三件事：先把长文档切成适合抽取的高密度语义块，再按固定 schema 输出实体、关系和类型，最后把每个结果强制回钉到原文位置。这里的 schema 是“输出长什么样”的明确模板，比如 JSON 字段和允许的类别；provenance 是“这条结论来自原文哪里”的证据记录。

如果没有最后一步的证据绑定，LLM 抽出的三元组很容易看起来合理、实际上无据可查。AEVS（Anchor–Extraction–Verification–Supplement）这类方法的价值就在这里：先找 anchor，也就是原文里可定位的候选片段；再抽取；再验证；最后补抽遗漏项。任何不能回溯到原文 span 的三元组，都不该直接写入图谱。

对初级工程师最重要的结论是：LLM 辅助 KG 构建的核心不是“模型多聪明”，而是“约束是否足够强”。在公开 IE 评测里，ChatGPT 在 CoNLL04 关系分类任务上的 F1 约为 65.82，而对应 SOTA 更高；这说明通用大模型可以显著减轻人工抽取负担，但不能被当成免审计的事实生成器。

表 1 给出一条最小可落地流水线：

| 阶段 | 典型输入 | 约束/输出 |
| --- | --- | --- |
| 文本精炼 | 维护日志、手册长段落 | 用 RAG/CoT 或任务切片提取核心事实，控制上下文长度 |
| 实体+关系抽取 | 精炼后的句子 | 按 schema 输出实体、关系、类型、置信度、span |
| 验证/补充 | 初步 triples + anchor list | 检查每个元素是否可回溯；补抽未覆盖 anchor；去重后入库 |

玩具例子可以这样理解：把一段机器人维护手册交给 LLM，先拆成“故障-部件-原因”信息块，再要求它输出 `{"subject":"hydraulic pump","predicate":"causes","object":"overheat","source_span":[120,145]}`。下游系统不需要再理解自然语言，只要校验 JSON 和 span 是否有效，就能决定是否入图。

---

## 问题定义与边界

这个问题的定义很明确：把非结构化文本变成知识图谱可消费的结构化事实。知识图谱最常见的基本单元是三元组 `(subject, predicate, object)`，也就是“谁”和“什么关系”和“谁/什么属性”。例如 `(液压泵, 导致, 过热)`。实体识别是先找“谁”；关系抽取是判断“它们之间是什么关系”；本体构建是给实体和关系规定统一类别，比如“部件”“故障”“因果关系”。

边界也必须明确，否则系统会把“看起来合理”的内容误当成“可以入库的事实”。

| in-scope | out-of-scope |
| --- | --- |
| 自动抽取有明确 anchor 的实体/关系 | 无 schema 的自由生成 |
| 多轮 prompt 收敛，输出可解析 JSON | 单轮随手生成、无法追踪 |
| 有 provenance 的结果进入 KG | 无证据三元组直接入库 |
| 人审兜底和补录 | 把模型输出当成最终真值 |

这里的 anchor 可以理解为“原文里能精确指出位置的证据片段”。如果模型输出 `bolt connected_to hydraulic pump`，但系统找不到这三个元素在原文中的字符区间，那么这条结果就该进入人工复核队列，而不是写进 KG。原因很简单：图谱一旦被下游搜索、推荐、诊断、问答系统消费，错误会被重复放大。

这也是为什么“可追溯”比“生成得像不像”更重要。对于知识图谱构建，系统首先是数据库写入器，其次才是语言模型应用。数据库写入器的首要要求是可验证，而不是流畅。

---

## 核心机制与推导

AEVS 的思路可以概括成三步。

第一步是 Anchor Discovery。系统先不急着抽三元组，而是先把文本中有意义的片段找出来，包括实体、关系短语、属性值，并记录字符级位置。字符级位置就是像 `[120,145)` 这样的半开区间，表示证据在原文中的起止位置。

第二步是 Grounded Extraction。模型只能从已发现的 anchor 中选择 subject、predicate、object，而不是自由造词。这样问题从“开放式生成”变成了“受限选择”。

第三步是 Verification + Supplement。对每个三元组做恢复匹配：能否把三元组元素重新映射回 anchor。如果能，就接受；如果部分能，就重试或补问；如果完全不能，就判为 hallucination。hallucination 这里不是泛指“模型胡说”，而是“输出无法由当前输入文本支持”。

可以用一个更形式化的记号表达：

$$
F1 = \frac{2PR}{P+R}
$$

其中 $P$ 是 precision，表示抽出的结果里有多少是真的；$R$ 是 recall，表示原文里该抽的事实有多少被抽到了。传统关系抽取只看三元组是否对，AEVS 额外要求 provenance 非空。设三元组为 $\tau$，元素为 $e \in \{s,p,o\}$，定义 provenance 函数为：

$$
\pi(\tau, e) \rightarrow \{\text{text spans}\}
$$

如果 $\pi(\tau, e)=\varnothing$，说明这个元素没有原文证据，那么这条三元组至少部分失真。工程上通常直接判为“不可自动入库”。

一个玩具例子：

原文：`The motor coil overheating causes abnormal current.`

系统先发现 anchor：
- `motor coil overheating`
- `causes`
- `abnormal current`

然后抽出三元组：
- `(motor coil overheating, causes, abnormal current)`

验证阶段检查这三个字符串是否都能定位到原文。如果 relation 被模型改写成 `leads_to`，但 schema 里允许把 `causes` 规范化到 `cause_of`，那可以通过 schema match 接受；如果模型输出了 `power module`，而原文根本没有这个片段，就必须拒绝。

真实工程例子更接近工业维护场景。比如机器人维护日志里有一句：“Hydraulic pump temperature rose after seal wear, resulting in repeated overheat alarms.” 抽取系统可能先把文档切成“部件状态”“故障现象”“可能原因”三类语义块，再输出：
- `(seal wear, causes, temperature rise of hydraulic pump)`
- `(temperature rise of hydraulic pump, triggers, overheat alarm)`

随后用位置证据校验每个元素。如果第二条关系只在语义上“像是成立”，但文本里没有触发关系的直接表述，就不应自动入图。

---

## 代码实现

可落地的实现通常不是一条 prompt，而是三轮控制逻辑：精炼、抽取、验证。下面给一个最小可运行示例，用 Python 演示“有证据才入库”的核心思想。这个例子没有调用真实 LLM，但把工程骨架完整保留下来了。

```python
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    source_span: Tuple[int, int]

def find_span(text: str, phrase: str):
    start = text.find(phrase)
    if start == -1:
        return None
    return (start, start + len(phrase))

def verify_provenance(text: str, triple: Triple) -> bool:
    s_ok = find_span(text, triple.subject) is not None
    o_ok = find_span(text, triple.object) is not None
    span_ok = 0 <= triple.source_span[0] < triple.source_span[1] <= len(text)
    evidence = text[triple.source_span[0]:triple.source_span[1]]
    p_ok = triple.predicate in evidence or triple.predicate in text
    return s_ok and o_ok and p_ok and span_ok

def deduplicate(triples: List[Triple]) -> List[Triple]:
    seen = set()
    out = []
    for t in triples:
        key = (t.subject, t.predicate, t.object)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out

def pipeline(text: str, raw_triples: List[Triple]) -> List[Triple]:
    verified = [t for t in raw_triples if verify_provenance(text, t)]
    return deduplicate(verified)

text = "Hydraulic pump overheat causes abnormal shutdown."
triples = [
    Triple("Hydraulic pump", "causes", "abnormal shutdown", (0, 48)),
    Triple("power module", "causes", "abnormal shutdown", (0, 48)),  # 幻觉
]

result = pipeline(text, triples)

assert len(result) == 1
assert result[0].subject == "Hydraulic pump"
assert result[0].object == "abnormal shutdown"
```

把这个骨架替换成真实服务后，典型主循环如下：

1. 文档切片：按段落、句子、主题或 token 长度切块。
2. 精炼阶段：用轻量模型先抽密集事实，减少主模型上下文负担。
3. Anchor 发现：要求模型输出实体、关系词、属性值及字符位置。
4. Grounded 抽取：按 JSON schema 生成 triples。
5. 验证：检查字段类型、类别合法性、span 可回溯性。
6. 补抽：如果还有大量 anchor 未被任何 triple 使用，再发起补充抽取。
7. 去重与规范化：别名合并、关系映射、冲突检测。
8. 入库：写入 Neo4j、RDF store 或自定义图存储。

对新手来说，最关键的不是先写复杂 prompt，而是先定 schema。比如先固定：
- `entity_type`: `Component | Fault | Symptom | Action`
- `relation_type`: `causes | located_in | triggers | solved_by`
- `source_span`: `[start, end]`
- `confidence`: `0-1`

只要 schema 稳定，后面的解析、校验、缓存、重试、监控才有基础。

---

## 工程权衡与常见坑

最大的风险仍然是幻觉。模型最常见的错不是把实体完全编造出来，而是把关系说得过头。AEVS 论文里也指出，检测到的 hallucination 以 unsupported relation 为主。这很符合工程直觉：实体通常能在文本里找到，关系更容易被模型“脑补”。

第二个现实问题是成本和延迟。多轮 prompt、长 schema、补抽重试都会抬高 token 消耗。生产环境里，账单和时延不是附属问题，而是系统边界的一部分。行业实践文章普遍建议引入缓存、模型路由和 prompt 压缩，否则高并发下 API 成本会迅速失控。

第三个问题是一致性。同一个实体在不同文档里可能被写成 `hydraulic pump`、`pump`、`main pump`。如果没有统一 schema、别名归并和位置证据，最终图里会出现大量重复点和互相冲突的边。

| 风险 | 描述 | 缓解 |
| --- | --- | --- |
| 幻觉 | 输出了无证据三元组 | provenance 校验，找不到 span 就拒收 |
| 成本/延迟 | 多轮调用拉高 token 与等待时间 | 轻重模型路由、缓存、短 schema、批处理 |
| 一致性 | 同义词、别名、关系口径不统一 | 规范化映射、去重规则、统一本体 |
| 召回不足 | 只抽显式关系，漏掉隐式事实 | coverage-aware supplement，人审回流 |
| 长文本失真 | 关键信息跨句、跨段丢失 | 任务切片、滑窗、局部摘要后再抽 |

一个真实工程例子是运输或设备维护工单系统。最初直接用大模型三轮抽取，每小时新增数百份工单时，账单可能迅速膨胀。后来把“文档切片+anchor 初筛”交给便宜模型，把“补抽与疑难验证”留给强模型，通常能在质量接近的前提下显著降本。这类分层路由本质上是把“每一段文本都用最贵模型处理”的粗糙策略改成“按复杂度付费”。

---

## 替代方案与适用边界

LLM-AEVS 并不是所有图谱任务的默认答案。它最适合“原始文本很多，想自动沉淀结构化事实，而且必须可审计”的场景。

如果你已经有一张较成熟的图谱，当前问题是“怎么让问答更可靠、更可解释”，那么 GraphRAG 往往更合适。GraphRAG 是“用图增强生成”，不是“从头抽图”。它更关注根据已有图路径做检索、解释和生成，而不是高吞吐自动抽取三元组。

如果预算敏感、语料小且领域可控，也可以先用小模型或开源模型配合 schema prompt。近年的研究表明，在知识图谱构建任务上，经过微调的 7B 到 8B 模型可以达到不错效果，而且 prompt 用自然语言还是类代码风格，影响不一定大；真正影响性能的常常是 prompt 格式是否稳定、输出是否结构化、后处理是否严格。

| 方案 | 适用边界 | 何时转向 LLM-AEVS |
| --- | --- | --- |
| GraphRAG（KG→LLM） | 已有图谱，重点是解释与问答 | 需要从海量新文本自动补图 |
| 小型/开源 LLM + schema prompts | 预算紧、领域封闭、可接受更多人工校验 | 需要更强 provenance 和自动补抽 |
| 纯 LLM QA | 只关心答案，不需要结构化沉淀 | 需要把结果长期存储并复用 |
| 传统规则/监督 IE | 文本格式稳定、标签充分 | 领域变化快、标注成本过高 |

对初级工程师的实际建议是：先别试图一步做到“全自动构图”。更稳妥的路径是先做“半自动入库系统”。
- 第一步，只允许有 span 的结果入库。
- 第二步，把低置信或无源结果送人工审核。
- 第三步，再考虑本体扩展、跨文档实体对齐和自动补图。

这样系统才会从“能演示”变成“能上线”。

---

## 参考资料

- Peng, Yang, et al. “The construction and refined extraction techniques of knowledge graph based on large language models.” *Scientific Reports*, 2026. https://www.nature.com/articles/s41598-026-38066-w
- Yang, Chen, He, Zhao. “Grounded Knowledge Graph Extraction via LLMs: An Anchor-Constrained Framework with Provenance Tracking.” *Computers*, 2026. https://www.mdpi.com/2073-431X/15/3/178
- Han et al. “Is Information Extraction Solved by ChatGPT? An Analysis of Performance, Evaluation Criteria, Robustness and Errors.” arXiv, 2023. https://doi.org/10.48550/arXiv.2305.14450
- Gajo et al. “Natural vs programming language in LLM knowledge graph construction.” *Information Processing & Management*, 2025. https://www.sciencedirect.com/science/article/pii/S0306457325001360
- Liao et al. “Large language model assisted fine-grained knowledge graph construction for robotic fault diagnosis.” *Advanced Engineering Informatics*, 2025. https://www.sciencedirect.com/science/article/pii/S1474034625000278
- “How can the integration of AI large language models and knowledge graph enhance fault diagnosis? A systematic literature review.” *Applied Soft Computing*, 2026. https://www.sciencedirect.com/science/article/pii/S156849462600356X
- “LLM Routing: Intelligent Model Selection for Cost and Performance Optimization.” Zylos Research, 2026. https://zylos.ai/research/2026-01-29-llm-routing-intelligent-model-selection
- “How to Reduce LLM Cost and Latency: A Practical Guide for Production AI.” Maxim AI, 2026. https://www.getmaxim.ai/articles/how-to-reduce-llm-cost-and-latency-a-practical-guide-for-production-ai/
- “A Graph RAG Approach to Enhance Explainability in Dataset Discovery.” *Data Science and Engineering*, 2025. https://link.springer.com/article/10.1007/s41019-025-00313-x
