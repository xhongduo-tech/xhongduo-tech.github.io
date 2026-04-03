## 核心结论

WizardLM 的“演进指令”本质上不是把训练数据简单扩写，而是把一个原本容易回答的指令，递归改写成更难、约束更多、步骤更长、格式更复杂的版本，再让教师模型为这些新指令生成答案，最后把不同难度的数据混合起来做微调。

这件事的重要性在于，它解决了一个常见问题：普通指令扩写往往只能增加“数量”，却不一定增加“难度”。这里的“难度”指任务对模型理解、推理、格式控制和多条件遵循能力的要求。Evol-Instruct 则显式制造难度阶梯，让模型不只会答“简单事实题”，还更擅长长指令遵循、比较分析、条件约束和多步推理。

可以把它和 Alpaca 那类平铺式扩写对比理解。平铺式扩写更像“多写几道同水平练习题”，而 Evol-Instruct 更像“先做基础题，再做带限制条件的题，再做需要分步骤证明的题”。两者都能扩充数据，但训练出来的能力侧重点不同。WizardLM 的收益主要体现在复杂问答、长指令、综合推理，而不是“1+1 等于几”这种低复杂度问题。

一个最小玩具例子如下：

| 层级 | 指令 |
|---|---|
| 原始指令 | `1+1=?` |
| 增加约束 | `只用加法解释 1+1，答案必须在 0-10 范围内。` |
| 增加步骤 | `先分两步解释 1+1 的计算过程，再给结果。` |
| 格式复杂化 | `先输出“过程”，再输出“结果”，每部分各一行。` |

这张表体现的不是“换个说法”，而是“逐轮加码”。WizardLM 的关键思路正是：从简单任务出发，逐轮演化出更高复杂度样本，并在每轮之后做筛选，避免训练集被劣质数据污染。

---

## 问题定义与边界

先把问题说清楚。WizardLM 不是要让模型“凭空发明所有任务”，而是从一个初始指令集出发，构造出更复杂但仍然可解的新任务。这里的“初始指令集”可以记作 $D^0$，它就是最早的一批 instruction-response 对。

目标不是无限增难，而是在“复杂度提升”和“可解性保留”之间找到平衡。所谓“可解性”，白话解释就是：新指令虽然更难，但教师模型仍然大概率能答对，且答案对训练有价值。如果新指令被改得过度复杂，最后只会得到错误、空洞或胡乱生成的回答，这些样本反而会伤害微调。

因此，Evol-Instruct 有明确边界：

| 边界问题 | 正确做法 | 错误做法 |
|---|---|---|
| 是否允许增加约束 | 可以，但要保留原任务核心语义 | 把原任务改成另一个完全不同的问题 |
| 是否允许增加步骤 | 可以，用于制造推理链 | 强行堆叠无关步骤，导致任务失真 |
| 是否允许改变格式 | 可以，训练输出控制能力 | 只改排版，不增加信息要求 |
| 是否允许无限变长 | 不应如此，通常限制新增词数 | 一轮加太多条件，教师模型无法完成 |

论文里强调，每次演化应尽量是“小步递进”，而不是“大幅重写”。这背后的工程逻辑很直接：如果每轮只增加 10 到 20 个词左右，就更容易保证“任务还在原问题附近”，同时难度又能稳步上升。

可以把整个流程概括成下面这条链路：

`初始任务 -> 演化器 -> 教师回答 -> 过滤器 -> 进入训练集`

其中“演化器”负责改写指令，“过滤器”负责删掉坏样本。没有过滤器，这套方法很快会退化成“自动制造垃圾数据”。

一个典型失败例子是：

`请解释 1+1 的每一步，并写一段长于 200 字且包含至少五个引用。`

这条指令看起来更复杂，但复杂度主要来自无关格式负担，不是来自问题本身。对于数学解释任务来说，这种改写会显著降低可解性，也降低样本质量，应被 eliminator 剔除。

---

## 核心机制与推导

Evol-Instruct 的核心机制可以形式化地写成多轮演化。设第 $t$ 轮的数据集为 $D^{(t)}$，其中每个样本是一个指令和响应对 $(x_i^{(t)}, y_i^{(t)})$。从上一轮到下一轮的过程是：

1. 取出上一轮指令 $x_i^{(t-1)}$
2. 用 evolver 把它改写成更复杂的新指令 $x_i^{(t)}$
3. 用教师模型为 $x_i^{(t)}$ 生成答案 $y_i^{(t)}$
4. 用 eliminator 过滤掉低质量样本
5. 保留通过筛选的样本进入 $D^{(t)}$

最终训练时使用所有轮次的并集：

$$
D_{\text{train}} = \bigcup_{t=0}^{T} D^{(t)}
$$

这个公式的含义很重要。它不是“只保留最难的数据”，而是“把所有难度层级一起训练”。原因很简单：如果只训练高难样本，模型可能失去对基础任务的稳定性；如果只训练低难样本，又学不到复杂能力。混合不同难度的数据，本质上是在做一种能力分层覆盖。

论文里常提到两类演化：

| 类型 | 含义 | 作用 |
|---|---|---|
| in-depth | 在原任务上逐步加深复杂度 | 强化推理、约束遵循、格式控制 |
| in-breadth / mutate | 生成相邻但不同的任务变体 | 扩展任务覆盖面，补充长尾场景 |

对于初学者，最容易理解的是 in-depth。它通常包含几类操作：

| 演化操作 | 白话解释 | 示例 |
|---|---|---|
| 增加约束 | 给答案增加必须遵守的条件 | “必须分三点回答” |
| 加深推理 | 要求展示中间步骤 | “先推导，再结论” |
| 增加比较 | 要求在多个方案间分析差异 | “比较 A 和 B 的优缺点” |
| 引入反事实 | 讨论“如果条件变了会怎样” | “若输入翻倍，结果会怎样” |
| 格式复杂化 | 要求结构化输出 | “输出成表格或 JSON” |

玩具例子可以写得更完整一点：

原始指令：`1+1=?`

第一轮演化：`只用加法解释 1+1，并保证答案在 0-10 范围内。`

第二轮演化：`先说明为什么 1 和 1 相加得到 2，再给最终结果，只能分两步回答。`

第三轮演化：`请按“步骤”和“结果”两个字段输出，对 1+1 进行解释。`

你会发现，原问题没有变，但任务负担逐轮提升了。这里的“负担”不是噪声，而是模型需要学会的控制能力。

真实工程例子则更接近 WizardCoder。假设原始任务是：

`写一个 Python 函数，判断字符串是否回文。`

演化后可以变成：

`写一个 Python 函数，忽略大小写和非字母数字字符判断字符串是否回文，并给出时间复杂度分析。`

再进一步：

`写一个 Python 函数，忽略大小写和非字母数字字符判断字符串是否回文；先给出双指针思路，再给实现，再写三个 assert 测试。`

这个例子说明，Evol-Instruct 适合把“单点技能题”逐步抬升成“带工程约束的任务题”。对代码模型来说，这比单纯要求“再写一个类似函数”更有训练价值。

---

## 代码实现

下面给一个可运行的简化版 Python 示例，用来模拟 Evol-Instruct 的三段式流程：演化、筛选、汇总。这个例子不是论文原始实现，但结构上和工程设计一致。

```python
from dataclasses import dataclass
from typing import List


@dataclass
class Sample:
    prompt: str
    response: str
    complexity: int
    source: str


def evolve_prompt(prompt: str, round_id: int) -> str:
    if round_id == 1:
        return f"{prompt} 请增加一个明确约束，并用两句话回答。"
    if round_id == 2:
        return f"{prompt} 请先给出推理步骤，再给结论，输出使用“步骤/结果”格式。"
    return prompt


def teacher_answer(prompt: str) -> str:
    if "1+1" in prompt and "步骤/结果" in prompt:
        return "步骤：1 和 1 相加得到 2。\n结果：2"
    if "1+1" in prompt:
        return "1 和 1 相加得到 2。答案是 2。"
    return "占位回答"


def eliminator(sample: Sample) -> bool:
    prompt = sample.prompt.strip()
    response = sample.response.strip()

    # 过滤空回答、照抄 prompt、过短回答
    if not response:
        return False
    if response == prompt:
        return False
    if len(response) < 4:
        return False
    return True


def build_dataset(seed_prompts: List[str], rounds: int = 2) -> List[Sample]:
    dataset: List[Sample] = []

    for prompt in seed_prompts:
        base = Sample(
            prompt=prompt,
            response=teacher_answer(prompt),
            complexity=0,
            source="seed"
        )
        if eliminator(base):
            dataset.append(base)

        prev_prompt = prompt
        for round_id in range(1, rounds + 1):
            new_prompt = evolve_prompt(prev_prompt, round_id)
            sample = Sample(
                prompt=new_prompt,
                response=teacher_answer(new_prompt),
                complexity=round_id,
                source=f"evol_round_{round_id}"
            )
            if eliminator(sample):
                dataset.append(sample)
                prev_prompt = new_prompt

    return dataset


data = build_dataset(["1+1=?"], rounds=2)

assert len(data) == 3
assert data[0].complexity == 0
assert data[1].complexity == 1
assert "结果：2" in data[2].response

for item in data:
    print(item)
```

这段代码有几个关键点：

1. `evolve_prompt` 模拟“演化函数”，负责把简单指令改写成更难版本。
2. `teacher_answer` 模拟“教师模型”，负责给新指令生成答案。
3. `eliminator` 模拟“过滤器”，负责删除无效样本。
4. `complexity` 字段记录复杂度等级，后续可用于采样或分析。

如果把它写成更接近实际训练管线的伪代码，大致是：

```python
dataset = seed_dataset

for round_id in evol_rounds:
    new_tasks = evolver.transform(dataset, round_id)
    answered = teacher.generate(new_tasks)
    filtered = eliminator.filter(answered)
    dataset.extend(filtered)

shuffle(dataset)
train(model, dataset)
```

工程里通常还会保存 metadata。一个常见 schema 可以是：

| 字段 | 含义 |
|---|---|
| `prompt` | 最终指令文本 |
| `response` | 教师模型生成的答案 |
| `complexity` | 当前难度等级 |
| `evol_type` | 本轮属于加约束、加步骤还是格式变化 |
| `parent_id` | 它由哪个旧样本演化而来 |
| `valid` | 是否通过过滤 |
| `teacher_model` | 由哪个教师模型生成 |

真实工程例子里，WizardCoder 把这套思路迁移到代码任务：不是只生成“更多代码题”，而是生成“更复杂、更像真实开发需求的代码题”。例如在 HumanEval 风格题目基础上，加入输入校验、复杂边界条件、性能要求、输出格式要求，这比平铺式扩充更接近真实编程工作。

---

## 工程权衡与常见坑

Evol-Instruct 看起来很优雅，但真正落地时，难点主要不在“生成”，而在“控制”。

第一个权衡是复杂度和可解性的平衡。复杂度越高，样本理论上越有训练价值；但复杂度一旦超过教师模型能力，答案质量就会快速下降。这个关系可以粗略理解成：当任务难度 $c$ 上升时，样本价值并不总是单调上升，而是可能在某个阈值之后下降，因为错误标注开始增多。

第二个权衡是数据规模和清洗成本。演化轮数越多，数据量越大；但过滤、去重、质量检查的开销也越高。没有清洗的扩量，通常只是把噪声一并放大。

下面这张表是常见工程权衡：

| 维度 | 提高一侧的收益 | 代价 |
|---|---|---|
| 复杂度 | 更强的推理和长指令能力 | 更高的答错率 |
| 数据量 | 更大覆盖面 | 更高的清洗成本 |
| 演化轮数 | 更细的难度阶梯 | 更容易语义漂移 |
| 格式要求 | 更强的输出控制能力 | 更容易生成模板噪声 |

常见坑主要有四类。

第一类是复杂度失控。比如每轮都同时加约束、加步骤、加格式、加反事实，最后生成一个长度巨大、条件互相缠绕的指令。这样的样本即使看起来“高级”，实际也可能是低质量数据。

第二类是语义漂移。这里的“语义漂移”指演化后的新任务已经不是原任务的递进版，而是变成另一个问题。例如从“总结一篇文章”演化成“比较三种算法并做实验设计”，这已经不是深度演化，而是任务跳变。

第三类是 prompt 抄写。教师模型有时会把 prompt 内容直接复述成 response，看起来格式完整，实际没有提供有用答案。这类样本必须过滤。

第四类是伪复杂。所谓“伪复杂”就是文本更长了，但认知要求没变。例如只是把一句话改写成三句话，或者附加一些无关修饰词。这种样本会增加 token 成本，却不增加能力密度。

一个简单的筛选 checklist 可以是：

| 检查项 | 是否保留 |
|---|---|
| 是否保留原始任务核心语义 | 否则丢弃 |
| 是否真的增加了约束或推理负担 | 否则丢弃 |
| 教师回答是否完整且不照抄 prompt | 否则丢弃 |
| 新样本是否与旧样本高度重复 | 否则丢弃 |
| 难度是否仍处于教师能力范围内 | 否则丢弃 |

实践上，一个保守但有效的策略是：每轮只做一种主要演化操作，限制新增词数，并在每轮后立即过滤。这比“先大规模生成、最后统一清洗”更稳。

---

## 替代方案与适用边界

Evol-Instruct 不是唯一的数据自举方法。最常见的对照方法是 Self-Instruct。两者都依赖模型生成训练数据，但优化目标不同。

Self-Instruct 的重点是“扩展广度”。白话说，就是先给少量种子任务，再让模型批量生成大量相似但不同的任务、输入和输出。它非常适合快速得到一批通用指令数据，因此 Alpaca 一类项目会采用类似思路。

Evol-Instruct 的重点则是“提升深度”。它不是优先追求任务种类越多越好，而是优先把已有任务逐步变难，制造能力阶梯。所以它更适合需要复杂推理、长指令遵循、格式控制的场景。

两者对比如下：

| 维度 | Evol-Instruct | Self-Instruct |
|---|---|---|
| 主要目标 | 提升任务复杂度 | 扩展任务覆盖面 |
| 生成方式 | 递归加码、逐轮演化 | 批量生成相邻任务 |
| 数据特点 | 难度分层明显 | 多样性更强 |
| 质量控制重点 | 复杂度是否可解 | 去重和无效样本清洗 |
| 更适合 | 复杂问答、代码、长指令 | 通用助手、基础指令覆盖 |

一个 Self-Instruct 风格的真实例子是：先人工写几十条种子 instructions，再让模型扩展成数万条数据，过滤掉重复和低质量样本，然后用于基础 instruction tuning。这条路线的优点是快，适合在冷启动阶段快速构造训练集。

但如果你的目标是“让模型更会处理复杂任务”，Self-Instruct 往往不够，因为它默认的扩展方式偏平铺，难度跃迁不够强。反过来，如果你的教师模型本身不够强，Evol-Instruct 也未必合适，因为它很依赖教师在更复杂任务上的正确回答能力。

所以它的适用边界可以总结为：

| 场景 | 是否适合 Evol-Instruct |
|---|---|
| 想提升复杂指令遵循能力 | 适合 |
| 想增强多步推理和比较分析 | 适合 |
| 教师模型质量较高 | 适合 |
| 只想快速做通用数据扩写 | 不一定划算 |
| 教师模型复杂任务错误率高 | 风险较大 |

结论是：Evol-Instruct 不是“更通用”的 Self-Instruct，而是“更强调难度控制和质量筛选”的一条专门路线。它最适合那些已经有基础数据、但希望进一步强化复杂任务表现的微调项目。

---

## 参考资料

- WizardLM: Empowering Large Language Models to Follow Complex Instructions, arXiv:2304.12244  
- WizardCoder: Empowering Code Large Language Models with Evol-Instruct, Microsoft Research / ICLR 2024  
- Self-Instruct: Aligning Language Models with Self-Generated Instructions, arXiv:2212.10560  
- ar5iv 对 WizardLM 论文的可读版本  
- IBM 对 instruction tuning 与 Self-Instruct 的综述资料
