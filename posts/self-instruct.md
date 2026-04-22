## 核心结论

Self-Instruct 是一种自举式指令数据生成方法：用少量人工写好的种子任务，引导强模型自动生成新的指令、输入和输出，再经过过滤后组成训练集，用于监督微调目标模型。

一句话写成公式就是：

$$
\text{Self-Instruct} = \text{seed tasks} + \text{model-generated instructions} + \text{filtering} + \text{supervised fine-tuning}
$$

其中，“自举”指的是系统先依赖一小批人工样本启动，然后用模型自己的生成结果扩大数据池。它不是完全取消人工，而是把人工成本从“逐条标注几万条数据”转成“设计少量种子任务、制定过滤规则、抽检合成数据”。

新手版解释：你先写 100 条高质量问题模板，模型就能自己扩成几万条相似但不重复的新问题，再把这些数据拿去微调模型。

| 方式 | 数据来源 | 成本 | 覆盖范围 | 主要风险 |
|---|---|---:|---|---|
| 人工标注 | 人逐条写 instruction/input/output | 高 | 取决于标注团队 | 慢、贵、规模受限 |
| Self-Instruct | 少量种子 + 模型批量生成 | 中低 | 可快速扩大 | 噪声、重复、任务跑偏 |
| 直接抓真实日志 | 用户真实输入 | 中 | 接近业务 | 隐私、输出缺失、质量不稳定 |

Self-Instruct 最适合解决指令微调的数据稀缺问题。指令微调是指让模型学习“看懂人类任务描述，并按要求输出结果”的训练方式。没有足够多样的指令数据，模型就容易只会完成少数固定任务，遇到不同表达方式就不稳定。

---

## 问题定义与边界

Self-Instruct 解决的问题是：指令微调需要大量 instruction-following 数据，但高质量指令数据很贵、很慢、覆盖不足。

一条典型指令微调数据通常包含三部分：

| 字段 | 含义 | 例子 |
|---|---|---|
| `instruction` | 用户希望模型完成的任务 | “把下面这段话改写得更正式” |
| `input` | 任务输入，可以为空 | “我明天不来了，你们自己处理。” |
| `output` | 期望答案 | “我明天无法到场，相关事项请各位协助处理。” |

如果你要做客服助手，不可能把每一种用户问法都人工写一遍。Self-Instruct 的作用就是先写少量代表性问题，然后让模型自己扩出更多说法和变体。

问题可以定义为：

| 项目 | 定义 |
|---|---|
| 问题 | 指令数据稀缺，人工扩写速度慢 |
| 目标 | 构造更多高质量 instruction-following 数据 |
| 输入 | 少量人工种子任务 |
| 输出 | 大规模指令、输入、输出三元组 |
| 约束 | 不能大量重复、不能跑偏、不能包含模型无法处理的输入类型 |

这里的“种子任务”是人工写的一小批高质量任务样例，用来告诉生成模型：什么样的任务是我们想要的。论文中的原始设置使用 175 条种子任务，最终生成约 52k 条指令和 82k 条输入输出实例。

Self-Instruct 的边界也要明确。它主要面向纯文本任务，不是多模态方案，也不是检索增强方案。

适合的任务包括：

| 任务类型 | 例子 |
|---|---|
| 文本问答 | 根据一段文字回答问题 |
| 改写 | 改成正式、简短、礼貌、客服口吻 |
| 分类 | 判断评论是正面还是负面 |
| 抽取 | 从工单中抽取设备型号、故障类型 |
| 生成 | 写邮件、摘要、标题、步骤说明 |

不适合直接使用的任务包括：

| 不适合场景 | 原因 |
|---|---|
| 图像理解 | 原始 Self-Instruct 面向文本，不处理图像输入 |
| 表格复杂计算 | 需要结构化解析和计算工具 |
| 实时外部知识 | 模型可能编造过期或不存在的信息 |
| 强工具调用任务 | 需要单独设计工具轨迹和执行结果 |
| 高风险专业判断 | 医疗、法律、金融等需要强审核链路 |

---

## 核心机制与推导

Self-Instruct 的核心流程是一个自举循环：

```text
seed tasks -> prompt sampling -> instruction generation -> instance generation -> filtering -> training set -> fine-tune
```

设初始种子池为 $S_0$，当前任务池为 $P_t$，生成模型为 $M$。第 $t$ 轮时，从 $P_t$ 中抽取 $k=8$ 条已有指令作为上下文，让模型生成候选指令 $\hat{I}$ 和候选实例 $\hat{D}$：

$$
\hat{I}, \hat{D} = M(\text{sample}(P_t, k))
$$

其中：

$$
\hat{D} = \{(x_i, y_i)\}_{i=1}^{n}
$$

$x_i$ 是输入，$y_i$ 是输出。比如 instruction 是“把句子改写得更礼貌”，input 是“快点把文件发我”，output 是“麻烦你方便时把文件发给我”。

玩具例子如下。

初始种子任务有 2 条：

| 编号 | 种子任务 |
|---|---|
| 1 | 把一句话改写得更礼貌 |
| 2 | 判断一句评论是正面还是负面 |

每轮让模型围绕这 2 条种子生成 6 条候选指令：

| 候选 | 内容 | 处理结果 |
|---|---|---|
| A | 把邮件改写得更正式 | 保留 |
| B | 判断用户反馈是否表达不满 | 保留 |
| C | 把下面句子改写得更礼貌 | 与已有任务太像，丢弃 |
| D | 根据 graph 回答问题 | 包含不可处理关键词，丢弃 |
| E | 为客服回复生成更温和版本 | 保留 |
| F | 从评论中抽取投诉原因 | 保留 |

如果保留 4 条，每条再生成 2 个输入输出实例，就得到 8 条训练样本。

过滤规则的关键是去重和排除不可处理任务。论文中使用 ROUGE-L 做近重复过滤。ROUGE-L 是一种基于最长公共子序列的文本相似度指标，可以粗略判断两条指令是否太像。

过滤规则可以写成：

$$
\max_{I \in P_t} \text{ROUGE-L}(\hat{I}, I) \ge 0.7 \Rightarrow \text{discard}
$$

意思是：如果候选指令 $\hat{I}$ 和任务池中任意已有指令的 ROUGE-L 相似度达到 0.7 或以上，就丢弃。

还要过滤多模态或不可处理任务：

$$
\text{contains}(\hat{I}, \{\text{image}, \text{picture}, \text{graph}\}) \Rightarrow \text{discard}
$$

因为原始流程主要生成纯文本指令，如果混入“看图回答”“根据图表分析”这类任务，训练目标会失真。

实例生成还有一个细节：分类任务使用 `output-first`，非分类任务使用 `input-first`。

“分类任务”是指输出来自有限标签集合的任务，比如“正面/负面”“通过/拒绝”。如果先生成输入，再让模型输出标签，模型可能偏向某些常见标签。`output-first` 是先确定标签，再生成符合该标签的输入，用来平衡类别分布。

非分类任务则更自然地使用 `input-first`：先生成输入，再生成输出。例如先写一段需要改写的句子，再生成改写结果。

最终微调目标是标准监督学习。给定指令 $I$、输入 $x_i$ 和答案 $y_i$，训练目标是最大化答案的条件概率：

$$
\max_{\theta} \sum_i \log p_{\theta}(y_i \mid \text{concat}(I, x_i))
$$

这里的 $\theta$ 是目标模型参数，$\text{concat}(I, x_i)$ 表示把指令和输入拼接成模型输入。白话说，就是让模型看到“任务说明 + 输入内容”后，更可能生成正确答案。

真实工程例子：企业客服或 IT 工单助手。团队先人工写 100-200 条种子任务，覆盖故障分类、步骤排查、表单填写、回复改写、升级转人工等高频场景。然后用 Self-Instruct 扩展同义问法和边界任务，最后规则过滤明显无效样本，人工抽检高风险样本。这样可以用较低成本把小模型微调成更贴近企业话术的助手。

---

## 代码实现

工程实现的重点不是训练模型，而是“生成、过滤、入库”的数据管线。训练可以使用常见监督微调框架，但数据进入训练前必须可复查、可追踪、可复现。

最小流程可以拆成四步：

| 步骤 | 作用 |
|---|---|
| 任务生成 | 从任务池抽样，构造提示词，让强模型生成候选指令 |
| 重复度过滤 | 用 ROUGE-L 或其他相似度指标过滤近重复指令 |
| 关键词过滤 | 排除图像、图表、音频、实时网页等不可处理任务 |
| 样本落盘 | 保存 instruction/input/output 和元数据 |

一个简化版 Python 示例：

```python
from difflib import SequenceMatcher

BAD_KEYWORDS = {"image", "picture", "graph", "audio", "video"}

def similarity(a: str, b: str) -> float:
    # 这里用 SequenceMatcher 近似演示；生产环境可替换为 ROUGE-L。
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def too_similar(candidate: str, task_pool: list[str], threshold: float = 0.7) -> bool:
    return any(similarity(candidate, old) >= threshold for old in task_pool)

def contains_bad_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in BAD_KEYWORDS)

def accept_instruction(candidate: str, task_pool: list[str]) -> bool:
    if too_similar(candidate, task_pool, threshold=0.7):
        return False
    if contains_bad_keyword(candidate):
        return False
    return True

seed_tasks = [
    "Rewrite the sentence in a more polite tone.",
    "Classify the review as positive or negative.",
]

task_pool = seed_tasks[:]

candidates = [
    "Rewrite an email in a more formal tone.",
    "Rewrite the sentence in a more polite tone.",
    "Answer the question based on the graph.",
    "Extract the complaint reason from a customer message.",
]

accepted = [c for c in candidates if accept_instruction(c, task_pool)]

assert "Rewrite an email in a more formal tone." in accepted
assert "Extract the complaint reason from a customer message." in accepted
assert "Answer the question based on the graph." not in accepted
assert "Rewrite the sentence in a more polite tone." not in accepted
assert len(accepted) == 2
```

完整工程中，不建议只保存三元组，还应保存元数据：

| 字段 | 含义 |
|---|---|
| `instruction` | 任务说明 |
| `input` | 输入内容 |
| `output` | 目标输出 |
| `source_seed` | 生成时参考的种子任务 |
| `filter_reason` | 被过滤时的原因 |
| `task_type` | 分类、改写、抽取、摘要等 |
| `generator_model` | 生成数据所用模型 |
| `created_at` | 生成时间 |
| `review_status` | 未审查、自动通过、人工通过、拒绝 |

伪代码如下：

```python
seed_tasks = load_seeds()
task_pool = seed_tasks[:]

while len(task_pool) < target_size:
    context = sample(task_pool, k=8)
    candidate_instruction = model.generate_instruction(context)

    if too_similar(candidate_instruction, task_pool, threshold=0.7):
        save_rejected(candidate_instruction, reason="too_similar")
        continue

    if contains_bad_keyword(candidate_instruction):
        save_rejected(candidate_instruction, reason="bad_keyword")
        continue

    if is_classification(candidate_instruction):
        label = model.generate_label(candidate_instruction)
        input_ = model.generate_input_for_label(candidate_instruction, label)
        output = label
    else:
        input_ = model.generate_input(candidate_instruction)
        output = model.generate_output(candidate_instruction, input_)

    save_example(
        instruction=candidate_instruction,
        input=input_,
        output=output,
        task_type=infer_task_type(candidate_instruction),
    )
    task_pool.append(candidate_instruction)
```

这里的关键不是把代码写复杂，而是把每一步的中间结果保存下来。否则一旦发现训练结果变差，很难判断是种子太窄、生成模型不稳、过滤阈值不合适，还是某一批数据污染了训练集。

---

## 工程权衡与常见坑

Self-Instruct 最大的风险是“规模大但质量不稳”。合成数据可以快速变多，但数据量不等于有效信息量。如果任务重复、答案错误、格式混乱，模型会稳定地学到错误模式。

种子设计决定上限。种子任务太窄，生成数据会继承并放大这种狭窄性。比如你只给模型 10 条“翻译成更礼貌表达”的种子，它就会一直围绕这个方向扩写，最后数据看起来很多，但任务类型很单一。

过滤强度决定噪声水平。阈值太松，重复任务会大量进入训练集；阈值太严，又会误删合理变体。论文采用的近重复约束是 ROUGE-L 小于 0.7，可以作为起点，但真实业务中仍要结合抽检结果调整。

| 常见坑 | 后果 | 对策 |
|---|---|---|
| 种子太窄 | 覆盖不足，模型只会少数任务 | 扩大任务类型、语气、格式 |
| 直接全量使用 | 噪声高，训练后行为不稳定 | 分层抽检，保留人工验证集 |
| 阈值太松 | 重复任务多 | 保持类似 `ROUGE-L < 0.7` 的约束 |
| 阈值太严 | 有效变体被删 | 按任务类型分别调阈值 |
| 分类任务顺序错 | 标签分布偏 | 分类使用 `output-first` |
| 混入多模态任务 | 训练目标失真 | 关键词过滤，单独处理 |
| 没有记录来源 | 无法排查坏数据 | 保存 `source_seed` 和生成批次 |

风险可以按严重程度分级：

| 风险等级 | 问题 | 影响 |
|---|---|---|
| 低风险 | 语义重复 | 浪费训练预算，提升有限 |
| 中风险 | 格式偏差 | 模型输出格式不稳定 |
| 高风险 | 错误标签、不可执行任务、幻觉输出 | 直接损害模型能力 |

还有一个容易忽视的问题：Self-Instruct 生成的是“看起来合理”的任务，不等于“业务真实高频”的任务。模型可能生成很多语言上自然但业务中很少出现的请求。因此在真实工程中，最好把 Self-Instruct 和真实日志统计结合起来。真实日志决定任务分布，Self-Instruct 负责扩展表达方式和边界样本。

---

## 替代方案与适用边界

Self-Instruct 不是唯一的数据构造方案。它适合“有少量种子、需要快速扩数据、任务主要是纯文本”的场景。如果任务依赖外部知识、工具调用、多模态输入，通常需要别的方案配合。

如果你做的是知识问答，光靠 Self-Instruct 不够，因为模型自己编出来的知识可能不可靠；这时更适合加检索，或者用真实语料构建数据。

| 方案 | 适合场景 | 优点 | 局限 |
|---|---|---|---|
| Self-Instruct | 文本任务扩写、指令微调冷启动 | 成本低，扩展快 | 需要过滤，知识可靠性弱 |
| 人工标注 | 高风险、高精度任务 | 质量稳定 | 贵，慢 |
| 检索增强 | 知识型问答、文档问答 | 可引用外部资料 | 需要检索系统和语料库 |
| 规则模板 | 格式固定任务 | 可控、稳定 | 覆盖变化能力弱 |
| 多轮人工协作 | 法律、医疗、金融等高风险领域 | 审核链路强 | 成本最高 |

Self-Instruct 适合以下场景：

| 场景 | 为什么适合 |
|---|---|
| 客服话术 | 同一意图有大量不同说法 |
| 文本改写 | 输入输出关系清晰 |
| 分类 | 标签集合明确，可控制类别分布 |
| 摘要 | 可批量生成不同长度和风格 |
| 信息抽取 | 字段结构明确，便于自动校验 |

不适合直接作为唯一方案的场景：

| 场景 | 原因 |
|---|---|
| 图像理解 | 需要视觉输入和视觉标注 |
| 强时效知识问答 | 合成数据可能过期或编造 |
| 复杂工具链任务 | 需要执行轨迹、工具结果和错误恢复 |
| 严肃专业判断 | 需要专家审核和责任边界 |
| 高精度结构化计算 | 需要程序校验或数据库校验 |

工程上更稳妥的做法是：先用 Self-Instruct 做冷启动数据，再用真实业务数据、人工验证集和线上反馈持续修正。Self-Instruct 负责把数据从 0 扩到 1，不能替代完整的数据治理。

---

## 参考资料

1. [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://aclanthology.org/2023.acl-long.754/)
2. [Self-Instruct ACL 2023 PDF](https://aclanthology.org/2023.acl-long.754.pdf)
3. [Self-Instruct 官方代码仓库](https://github.com/yizhongw/self-instruct)
4. [Stanford Alpaca: An Instruction-following LLaMA Model](https://github.com/tatsu-lab/stanford_alpaca)
