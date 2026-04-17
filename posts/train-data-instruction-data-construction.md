## 核心结论

指令数据的构建，本质上不是“尽量多收集问答”，而是把一批候选的 `instruction → response` 样本，经过生成、清洗、打分、排序，变成一组适合监督微调（SFT，意思是“拿标准答案直接教模型”）的数据子集。真正决定效果的通常不是总量，而是覆盖面、可信度和训练目标是否一致。

第一条结论是，稳定可复用的做法通常是一个闭环：先准备少量高质量种子，再让强模型扩写，再用自动指标和人工复核筛掉低质量样本，最后按优先级送入训练。这个闭环成立的原因很直接：SFT 学到的是“看到什么指令，应当如何回应”的条件映射。如果数据里有大量重复、含糊、错误或偷懒回答，模型就会把这些坏模式一并学进去。

第二条结论是，小而精的数据集可以接近甚至超过大而杂的数据集。原因不是“少就是好”，而是高质量样本能更集中地覆盖技能边界。像 Instruct-SkillMix 这类工作说明，几千条精心组合和筛选过的样本，可以让基础模型获得相当强的指令跟随能力。这说明构建阶段的设计，很多时候比盲目扩充规模更重要。

第三条结论是，指令数据构建不是只做“生成”，还必须做“选择”。对每条候选样本，只问“它像不像答案”是不够的，还要问“它对训练是否有增益”。常见做法是把长度、奖励模型分数、困惑度、嵌入距离等指标合成一个质量分，再结合多样性选择和子集规模搜索，避免把大量相似样本一起塞进训练。

一个新手可执行的玩具例子是：先手写两条 expert seed，例如“解释心血管健康”和“生成财务报表摘要”，再让强能力 LLM 组合“解释 + 推理 + 表格输出”技能去扩写 500 条候选，对所有结果做清晰度、重复度、格式一致性打分，最后只保留最好的 100 条做初期 SFT。这里关键不是 100 这个数字，而是你只把“模型确实应该学”的内容送进去。

---

## 问题定义与边界

“指令数据的构建”指的是：围绕某个训练目标，产出并筛选一批结构化的 `instruction → response` 对，使它们能直接进入 SFT 或作为偏好训练前的基础样本。它不等于预训练数据清洗，也不等于在线推理时的提示词工程。它关注的是训练前的数据制造与治理。

这个问题有几个明确边界。

第一，输入来源可以多样，但输出格式要统一。输入可以是专家撰写的种子、已有高质量文案、企业知识库、FAQ、对话日志，甚至是检索增强生成（RAG，意思是“先查资料再生成”）产生的草稿。但到了训练集阶段，最好都落成统一字段，例如：`instruction`、`response`、`domain`、`difficulty`、`safety_level`。如果结构不统一，后续筛选、统计和训练都很难稳定。

第二，允许扩写，但不允许失控。扩写是指从少量样本生成更多任务变体，比如把“解释一个概念”扩成“解释 + 对比 + 举例 + 输出表格”。它的价值在于提高覆盖面。但扩写不能脱离原始任务边界，否则你以为在做金融摘要，结果模型学到的是闲聊、写诗和无关推理，训练信号会被稀释。

第三，“高质量”不是抽象赞美，而是可操作的过滤条件。至少要回答五个问题：这条样本是否语义清楚、是否与任务对齐、是否包含错误、是否和已有样本重复、是否属于目标领域。下面这张表可以把判断链条具体化。

| 维度 | 白话解释 | 在判断链条中的角色 | 常见检查方式 |
|---|---|---|---|
| 多样性 | 不要所有题都长得一样 | 防止模型只会几种模板 | embedding 距离、聚类覆盖 |
| 语义对齐 | 问题和回答是否真的匹配 | 保证训练目标正确 | 规则校验、LLM judge、人工抽查 |
| 自动评分 | 先机器粗筛 | 降低人工成本 | reward score、困惑度、格式分 |
| 人类复核 | 关键样本人工确认 | 防止自动指标漏判 | spot check、专家审稿 |
| 领域/难度标签 | 给样本贴上类别和层级 | 便于分桶训练与评估 | taxonomy 标注 |

第四，规模本身也是边界条件。不是候选池越大越好，也不是最终训练集越大越好。有研究指出，不加控制地把全部样本放入训练，可能遇到 double descent，也就是性能随数据量变化不是单调提升，而是在某些区间先变差再变好。工程上这意味着你不仅要筛样本，还要决定“到底喂多少”。

一个真实工程例子是精神科 Agent。企业可能没有大规模公开标注语料，但有内部指南、病程模板、访谈流程和安全规范。此时可以先检索相关文档，再让 LLM 生成“患者提问 → 助手回应”的候选对话，并为每条附上 `领域=精神科`、`情绪=焦虑/抑郁/稳定`、`安全等级=高/中/低` 等元数据。之后再剔除含糊回答、危险建议和重复样本，才形成可用于微调的指令集。这属于“有约束的数据生成”，而不是自由写作。

下面这张边界矩阵可以帮助理解问题范围。

| 项目 | 合理边界 |
|---|---|
| 可接收的输入形式 | 专家种子、知识文档、FAQ、历史高质量回答、RAG 草稿 |
| 允许的扩写方式 | 改写、难度升降、技能组合、多轮化、结构化输出 |
| 过滤条件 | 错误事实、低清晰度、答非所问、格式不合规、语义重复 |
| 标签维度 | 领域、难度、技能、安全等级、回答风格 |
| 上线前评估门槛 | 离线集 loss、人工通过率、安全拒答率、重复率阈值 |

---

## 核心机制与推导

为什么“生成后筛选”比“原样全收”有效？因为训练的目标不是最大化候选样本数，而是最小化目标任务上的损失。设基础模型为 $M$，用某个候选子集 $D$ 微调后得到 $M_{ft}$，在验证集 $D_{eval}$ 上的损失记为 $L(M_{ft}, D_{eval})$。那么样本子集的质量可以写成：

$$
Q_{(D \mid M,S)} \propto -L(M_{ft}, D_{eval})
$$

意思很简单：某个子集如果能让微调后的模型在验证集上损失更低，它就更“值钱”。这里的 $S$ 可以理解为评分器、选择策略或约束条件。

问题在于，真正把每个候选子集都训一遍再测 loss，成本太高。所以工程上通常先对每条样本提取一组便宜的代理指标，再拟合一个质量预测模型。常见指标包括：

1. 长度特征：太短可能信息不足，太长可能啰嗦或偏题。
2. 奖励模型分数：奖励模型可以理解为“像老师一样打主观分”的模型。
3. 困惑度 perplexity：白话讲，就是模型对这段文本“陌不陌生、顺不顺手”。异常高可能表示文本怪异。
4. 嵌入距离：把文本映射成向量后，判断它和已有样本是否过于相似。
5. 一致性指标：例如 instruction 和 response 是否真的在回答同一个任务。

这样每条样本 $x_i$ 都会得到一个特征向量 $\phi(x_i)$，再用一个回归器去预测它对最终 loss 的贡献。哪怕只是最小二乘回归，本质上也在回答一个工程问题：哪些表面特征，最可能对应训练收益。

但只有分数还不够，因为高分样本可能彼此很像。假设你选出的前 500 条全都是“总结文章并列 3 个要点”，训练时就会发生局部过拟合。为了解决这个问题，常配合多样性选择。一个典型方法是 $\text{k-center}$ 贪心选择：

$$
u = \arg\max_{i \notin s}\min_{j \in s}\Delta(x_i, x_j), \quad s \leftarrow s \cup \{u\}
$$

这里 $\Delta(x_i, x_j)$ 表示样本间距离，$s$ 是已选集合。直觉是：每次都补进“离当前集合最远”的样本，让覆盖面尽量大。它解决的是“不要把预算浪费在近似重复样本上”。

再往前一步，最终子集规模也要优化。BlendSearch 这一类方法的作用，是在不同规模上做带预算的搜索，而不是凭经验拍脑袋说“取 3000 条”。因为在真实数据上，验证损失随数据量变化可能呈现不平滑曲线，盲目增加样本不一定有好处。

一个玩具推导可以这样理解。假设你有 10 万条候选数据，每条数据先得到一个质量预测分 $q_i$。如果你只按 $q_i$ 排序取前 3000 条，你可能得到高质量但很单一的集合；如果你只按多样性选 3000 条，可能又会混入大量低质量样本。更合理的做法是先用质量分过滤到一个候选池，再在池中做多样性抽样，最后对不同规模 $k$ 做验证。最终保留的是“质量足够高、覆盖足够广、规模又不过量”的黄金子集。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不依赖真实 reward model，而是用简化规则模拟“质量分 + 去重 + 选子集”的流程，目的是把机制讲清楚。真实工程里你会把 `clarity_score`、`safety_score`、`embedding_similarity` 换成模型调用结果。

```python
from math import exp

candidates = [
    {
        "instruction": "解释心血管健康，并给出三条可执行建议",
        "response": "心血管健康指心脏和血管系统保持正常工作。建议控制盐摄入、规律运动、监测血压。",
        "domain": "health",
        "difficulty": "easy",
    },
    {
        "instruction": "解释心血管健康",
        "response": "就是健康。",
        "domain": "health",
        "difficulty": "easy",
    },
    {
        "instruction": "生成财务报表摘要，并输出表格",
        "response": "本季度收入增长8%，净利润下降2%。原因包括营销支出增加与汇率波动。",
        "domain": "finance",
        "difficulty": "medium",
    },
    {
        "instruction": "生成财务报表摘要，并输出表格",
        "response": "本季度收入增长8%，净利润下降2%。原因包括营销支出增加与汇率波动。",
        "domain": "finance",
        "difficulty": "medium",
    },
]

def clarity_score(text: str) -> float:
    # 过短回答通常信息不足
    return min(len(text) / 40.0, 1.0)

def safety_score(text: str) -> float:
    banned = ["随便", "无法确定但照做", "直接停药"]
    return 0.0 if any(word in text for word in banned) else 1.0

def perplexity_proxy(text: str) -> float:
    # 用简化指标模拟“越怪异越高”
    unique_ratio = len(set(text)) / max(len(text), 1)
    return 10.0 * unique_ratio

def quality_score(item: dict) -> float:
    reward = 0.5 * clarity_score(item["response"]) + 0.5 * safety_score(item["response"])
    ppl = perplexity_proxy(item["response"])
    return reward + 1.0 / (1.0 + ppl)

def dedup_key(item: dict) -> tuple:
    return (item["instruction"].strip(), item["response"].strip())

def select_dataset(items, threshold=0.9):
    seen = set()
    selected = []
    for item in items:
        key = dedup_key(item)
        if key in seen:
            continue
        seen.add(key)
        q = quality_score(item)
        if q > threshold:
            item = dict(item)
            item["q"] = q
            selected.append(item)
    selected.sort(key=lambda x: x["q"], reverse=True)
    return selected

selected = select_dataset(candidates)

assert len(selected) == 2
assert selected[0]["domain"] in {"health", "finance"}
assert all(x["q"] > 0.9 for x in selected)

for row in selected:
    print(row["instruction"], round(row["q"], 3))
```

这个例子做了四件事。

第一，模拟了自动评分。这里的 `quality_score` 相当于一个极简版回归器，把清晰度、安全性、困惑度组合成单一分数。真实系统会更复杂，但思想一致：先用廉价信号近似判断“这条样本值不值得训”。

第二，做了去重。这里的去重只是完全相同文本去重；真实工程会进一步做语义去重，例如比较 embedding 余弦相似度。因为两条文本不完全相同，也可能在训练意义上完全重复。

第三，做了阈值过滤。低于阈值就直接丢弃，不要抱着“也许有用”的心理全留着。坏数据对 SFT 的污染通常比很多人想的更严重。

第四，做了排序。训练资源有限时，排序意味着优先级。先训练最稳定、最关键的样本，再决定是否扩展边界。

如果把这个玩具流程升级到真实工程，大致伪代码如下：

```python
for seed in expert_seeds:
    candidates += llm_expand(seed, skill_mix=True)

for item in candidates:
    item.reward = reward_model(item)
    item.ppl = perplexity(item.response)
    item.align = semantic_alignment(item.instruction, item.response)
    item.dup = max_similarity(item, accepted_pool)
    item.q = regression_predict(
        reward=item.reward,
        ppl=item.ppl,
        align=item.align,
        dup=item.dup,
    )

pool = [x for x in candidates if x.q > threshold and x.dup < dup_threshold]
pool = semantic_dedup(pool)
final_set = blend_search_select(pool, eval_set)
```

“解释心血管健康”这个玩具例子属于概念解释任务，目标是让模型学会清楚、简洁、可靠地说明一个主题；“生成财务报表摘要”则更接近真实工程，因为它同时要求抽取关键数字、压缩信息并按结构输出。这两类任务混合在一起训练，能帮助模型学会在不同输出形式之间切换，但前提是样本质量一致、标签明确。

---

## 工程权衡与常见坑

第一类坑是低质量回答污染训练。所谓 “shirker”，可以理解为“偷懒回答”，表面像答案，实际没有完成任务。例如用户要求“给出三点原因并附风险提示”，结果 response 只写一句模糊结论。研究里已经观察到，即便这类低质量样本只占一部分，也会明显拖慢或拉低最终效果。对策不是训练时指望模型自己分辨，而是在入池前就把它们筛掉。

第二类坑是重复样本过多。重复有两种，一种是字面重复，另一种是语义重复。后者更隐蔽，比如“总结财报要点”和“概括季度业绩重点”，如果答案结构和信息密度几乎一样，对训练的边际收益就很低。大量重复会让你误以为“数据很多”，实际上模型只是在反复看同一类模式。

第三类坑是子集规模选择错误。很多团队看到 10 万条候选样本，就倾向于全部使用，因为“扔掉太可惜”。但如果这些样本中混杂了低质量、弱对齐和高度重复内容，全部加入训练可能不如只取 3000 条高质量子集。这里的核心不是节省训练费，而是防止训练信号被稀释。

第四类坑是标签体系太粗。你如果只标“医疗”这个大类，那精神科问诊、安全拒答、药物说明、安抚式沟通都被混在一起。模型能学到一些表面格式，但很难学到任务边界。标签不是装饰，它决定后续分桶评估、采样均衡和故障定位是否可做。

下面这张表把常见坑和常用对策放在一起。

| 常见坑 | 具体表现 | 对策 |
|---|---|---|
| 低质量回答 | 答非所问、偷懒、缺步骤 | 自动评分 + 人工复核 |
| 冗余数据 | 大量近似样本反复出现 | embedding 去重、聚类抽样 |
| 子集规模不当 | 数据越加越多，效果反而下降 | BlendSearch 或验证集搜索 |
| 标签粗糙 | 训练混杂、问题难定位 | 细化领域/难度/安全标签 |
| 只看生成不看评估 | 觉得样本“看起来不错”就上线 | 建立固定离线评测集 |

真实工程里，比较稳妥的做法是建立“质量护栏”。例如每一批新合成数据进入候选池前，先跑一轮 reward model、格式校验、embedding 去重和安全规则扫描；如果本批数据加入后，验证集 loss 上升、关键任务通过率下降或安全拒答率异常，就直接回退到上一版本优质池。数据构建不是一次性生产，而是带监控的迭代系统。

---

## 替代方案与适用边界

当通用指令数据不够用时，常见替代方案有两类。

第一类是 `RAG + 自有文档 + LLM 生成`。它适合领域稀缺、公开数据难拿、专业约束很强的场景，例如医疗、法律、企业客服。它的优势是能快速把内部知识转成训练样本，并天然带上下文依据。它的风险是覆盖面受文档源限制，如果知识库本身偏窄或存在陈旧内容，生成出来的数据也会一起继承这些问题。

第二类是 `skill mix 合成 + 筛选`。所谓 skill mix，就是把多个能力维度组合起来，比如“解释 + 推理 + 表格输出 + 风险提示”，再让 LLM 基于种子样本扩写。这类方案适合通用助手、办公助理、教育辅导等跨任务场景。它的优势是覆盖面广，容易系统化扩展；风险是如果种子设计不好，模型会学到一堆格式化但不真实的任务。

下面这张表可以作为选型起点。

| 方案 | 适用场景 | 资源需求 | 风险点 |
|---|---|---|---|
| RAG + 自有文档 | 医疗、法律、企业内部知识型 Agent | 需要高质量知识库与检索链路 | 覆盖受限、文档过时、幻觉引用 |
| skill mix 合成 + 筛选 | 通用助手、跨域任务、低成本起步 | 需要强基座模型和评分体系 | 容易模板化、分布不真实 |

对于零基础到初级工程师，一个实用决策规则是：

1. 如果任务高度依赖专业知识和安全边界，优先用 RAG 生成候选，再做严格过滤和标签化。
2. 如果任务更偏通用能力训练，优先从少量高质量 seed 出发做 skill mix 扩写，再通过质量预测和多样性选择缩成高质量子集。
3. 无论选哪条路线，都不要跳过验证集和人工抽查。没有评估的“高质量数据”只是感觉，不是工程结论。

所以，替代方案不是互斥关系，而是不同数据条件下的起点不同。很多成熟系统最后都会收敛到同一个基本框架：多源候选池、自动评分、人类复核、子集选择、持续回归评估。差别只在于候选池从哪里来，以及约束强度有多高。

---

## 参考资料

- EmergentMind, “High-Quality Instruction Dataset”  
  https://www.emergentmind.com/topics/high-quality-instruction-dataset
- EmergentMind, “Instruction Data Selection Method”  
  https://www.emergentmind.com/topics/instruction-data-selection-method
- ICLR 2025 Poster, “Instruct-SkillMix”  
  https://iclr.cc/virtual/2025/poster/31037
- OpenReview, “InstructMining / Instruction Data Selection”  
  https://openreview.net/forum?id=wF6k0aWjAu
- OpenReview, “Instruct-SkillMix”  
  https://openreview.net/forum?id=XGurA1H49Y
- CatalyzeX, “A New Pipeline For Generating Instruction Dataset via RAG and Self Fine-Tuning”  
  https://www.catalyzex.com/paper/a-new-pipeline-for-generating-instruction
