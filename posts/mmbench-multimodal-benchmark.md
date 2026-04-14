## 核心结论

MMBench 是一个面向多模态大模型的选择题评测集。多模态大模型，白话讲，就是既能看图也能读文字，再把两者一起理解的模型。它的价值不在于“题很多”，而在于把能力拆成 20 个细分维度，并且用一套更严格的计分方法，尽量避免模型靠运气、靠选项位置偏好、靠非标准回答混过评测。

它主要解决两个老问题。

第一，模型明明知道答案，但输出成了“我认为第三个更合理”这种自由文本，程序如果只按 `A/B/C/D` 精确匹配，就会把它误判成错。第二，模型可能并不真正理解图片和问题，只是偏好固定位置，比如更常选第一个选项，或者更常选最后一个选项。普通单轮评测很难分辨“真会做题”和“碰巧押中”。

MMBench 的设计是：先用 LLM extractor 把自由回答抽取成标准选项，再用 CircularEval 把同一道题按不同选项顺序重复多轮，只要有一轮没对，这题就算没通过。LLM extractor，白话讲，就是一个“答案格式整理器”；CircularEval，白话讲，就是“把选项顺序轮着换，检查模型是不是一直都真会”。

它的核心准确率公式很简单：

$$
\text{Accuracy} = \frac{\text{全轮全对题数}}{\text{总题数}}
$$

这里“全轮全对”是重点，不是“做对过一次”就算对。

一个新手版本的理解方式是：给模型一道带图的四选一题，让它答 4 次。每次把 `A/B/C/D` 的内容整体右移一位，再让模型重新选。如果它只是偏爱某个位置，那么换几轮之后就会露出问题；如果它真的理解了题目，不管选项怎么轮换，它都应该持续指向同一个语义正确答案。

下面这张表可以直观看 MMBench 输出的能力画像。能力画像，白话讲，就是“把总分拆成很多局部得分，看到模型擅长什么、不擅长什么”。

| 能力维度 | 示例准确率 |
| --- | ---: |
| 目标识别 | 86% |
| 属性推理 | 71% |
| 场景理解 | 78% |
| 社会关系 | 62% |
| 空间关系 | 68% |
| 文本图像对齐 | 74% |

这类表的意义是：总分相同的两个模型，可能短板完全不同。一个可能视觉感知强、推理弱；另一个可能知识问答强、空间判断弱。对工程选型，这比单一排行榜更有用。

---

## 问题定义与边界

MMBench 要回答的问题不是“这个模型是不是聪明”，而是更窄也更工程化的问题：在给定图片、问题和候选答案时，这个多模态模型能否稳定地做出正确选择，并且这种正确能否分解到多个具体能力维度上。

这里的“能力维度”是评测设计里的分析单位。白话讲，就是把“会不会做题”拆成很多小能力桶，每个桶单独算分。根据公开资料，MMBench 覆盖 20 个能力维度，并归入感知、推理、知识三大类。对初级工程师来说，可以把它理解成一张更细的成绩单，而不是只有一个总分。

下面给出一个便于理解的维度总表。具体命名在不同实现或平台展示上可能略有差异，但核心思想是一致的：把视觉理解拆成足够细的子任务。

| 大类 | 能力维度 | 评测关注点 |
| --- | --- | --- |
| 感知 | 目标识别 | 图里是什么物体 |
| 感知 | 目标计数 | 有几个目标 |
| 感知 | 属性识别 | 颜色、材质、状态 |
| 感知 | 文字识别 | 图中文字内容 |
| 感知 | 场景理解 | 场景类型与布局 |
| 感知 | 动作识别 | 人或物在做什么 |
| 感知 | 空间关系 | 左右上下远近遮挡 |
| 推理 | 属性推理 | 从属性组合推出结论 |
| 推理 | 逻辑推理 | 排除、比较、蕴含 |
| 推理 | 因果判断 | 结果与原因关系 |
| 推理 | 常识推理 | 依赖常见世界知识 |
| 推理 | 社会关系 | 人物关系与互动 |
| 推理 | 时间顺序 | 先后与阶段判断 |
| 推理 | 细节比对 | 多候选之间的细微差别 |
| 知识 | 领域知识 | 专业或百科知识 |
| 知识 | 文化知识 | 地标、习俗、符号 |
| 知识 | 跨模态对齐 | 图和文是否一致 |
| 知识 | 指代表达理解 | “它”“这个人”指谁 |
| 知识 | 图表理解 | 读图表与统计图 |
| 知识 | 综合判断 | 多因素联合决策 |

边界也必须说清楚。MMBench 不是万能评测，它至少有四个边界。

第一，它主要评估选择题能力，不直接评估开放式生成质量。一个模型可能选项题分高，但写长答案时幻觉很多。幻觉，白话讲，就是模型自信地编造错误内容。

第二，它评估的是离散题库中的能力表现，不等于真实业务稳定性。业务里还会出现长上下文、多轮对话、工具调用、异常输入、图像质量差等问题。

第三，它不直接覆盖部署成本，比如推理延迟、显存占用、吞吐量。工程选型不能只看分数，还要看资源约束。

第四，它提供的是“能力切片”，不是因果解释。某个维度低分，只能说明该模型在这一类题上表现差，不能自动推出它是数据不够、视觉编码器弱，还是指令跟随差。

一个玩具例子能说明这个边界：你把“属性推理”和“社会关系”各取 20 道题，模型在前者拿到 85%，后者只有 50%。这说明它对“红色、圆形、金属”这类显式属性组合比较稳，但对“谁是老师、谁是学生、谁和谁在合作”这类隐含语义关系不稳。这个结论有用，但它仍然只是题库内结论，不等于真实课堂监控或客服审单场景里的完整效果。

---

## 核心机制与推导

MMBench 的核心机制由两部分组成：Choice Extractor 和 CircularEval。

### 1. Choice Extractor：先把自由文本变成标准选项

很多模型不会老老实实只输出 `A`。它可能输出：

- “我认为答案是第三项。”
- “The correct option should be C because...”
- “图中明显是苹果，所以选第二个。”

如果程序只做字符串精确匹配，就会出现大量假阴性。假阴性，白话讲，就是“本来答对了，但统计系统把它算成错”。

Choice Extractor 的做法是：把模型原始输出、题目和选项再交给一个更稳定的 LLM，让它抽取成标准标签 `A/B/C/D`。这一步不是替模型思考，而是统一答案格式。

转换示意可以写成：

| 原始输出 | 抽取后标签 |
| --- | --- |
| “我选第三个，因为图里只有三只猫。” | `C` |
| “The answer is option B.” | `B` |
| “应当是红色那项。” | `A` 或对应项 |
| “无法判断” | `Unknown` 或判错 |

这里要注意，Extractor 本身也可能出错，所以工程上通常要固定 prompt、保留原始响应，并在异常样本上人工抽查。

### 2. CircularEval：把选项顺序轮换，检查是否稳定命中

CircularEval 的核心思想很朴素：如果模型真的理解了题目，那么把选项顺序打乱后，它也应该稳定选择语义上正确的那个答案，而不是跟着位置跑。

以四选一为例，原始选项可能是：

- A: 2 个苹果
- B: 3 个苹果
- C: 4 个苹果
- D: 5 个苹果

假设正确答案是 `B`，也就是“3 个苹果”。CircularEval 会做 4 轮：

1. 原始顺序：A B C D
2. 右移一次：D A B C
3. 再右移一次：C D A B
4. 再右移一次：B C D A

每一轮都把“3 个苹果”放到了不同字母位置。模型必须每轮都指向“3 个苹果”对应的新标签，这题才算最终答对。

伪流程可以写成：

```text
for each question:
    passed = True
    for shift in 0..N-1:
        rotated_options = rotate_right(options, shift)
        response = model(image, question, rotated_options)
        pred = extractor(response, rotated_options)
        gold = remap_gold_label(original_gold, shift)
        if pred != gold:
            passed = False
            break
    record(passed)
```

其中 `remap_gold_label` 的作用是：原始正确答案是某个语义选项，选项右移后，它对应的字母也跟着变化，所以必须同步映射。

如果总题数是 $Q$，第 $i$ 题在全部轮次都通过时记为 $c_i = 1$，否则为 $0$，那么总准确率是：

$$
\text{Accuracy} = \frac{1}{Q}\sum_{i=1}^{Q} c_i
$$

其中：

$$
c_i = \prod_{r=1}^{N_i} \mathbb{I}(\hat{y}_{i,r} = y_{i,r})
$$

$\mathbb{I}$ 是指示函数。白话讲，相等记 1，不等记 0。因为用了乘积，所以只要某一轮错，整题就是 0。

一个玩具例子：

某题正确语义答案是“香蕉”，共有四个选项。模型四轮预测结果分别是 `A, D, B, B`。如果第三轮中“香蕉”对应的标准标签本应是 `C`，那第三轮错了，这题整体就判错。即使前两轮和第四轮都对，也不加分。

这比普通单轮评测严格得多，但正因为严格，它更能过滤掉“靠位置偏好碰巧答对”的假成绩。

### 3. 为什么这能降低位置偏差

假设一个糟糕模型根本不会做题，只是总选 `A`。在普通四选一评测中，它理论上可能拿到约 25% 的分数。如果题库选项分布不均，甚至还能更高。

但在 CircularEval 中，同一道题的正确语义答案会轮流出现在 `A/B/C/D`。如果模型始终只选 `A`，那它最多只会在某一轮碰巧命中，而不可能在所有轮次都命中，因此整题依然是错。

这就是 MMBench 里最关键的信度提升点。信度，白话讲，就是“这个分数到底靠不靠谱”。

---

## 代码实现

下面给一个最小可运行的 Python 实现，用来说明 CircularEval 的核心逻辑。这个例子不调用真实大模型，而是用一个模拟模型展示为什么“总选 A”会在单轮评测里拿分，但在 CircularEval 里失效。

```python
from dataclasses import dataclass
from collections import defaultdict

LABELS = ["A", "B", "C", "D"]

@dataclass
class Question:
    qid: str
    ability: str
    options: list[str]
    correct_index: int  # 原始正确选项下标

def rotate_right(items, shift):
    shift = shift % len(items)
    if shift == 0:
        return items[:]
    return items[-shift:] + items[:-shift]

def remap_correct_label(num_options, correct_index, shift):
    # 原始 correct_index 对应的选项右移后落到的新位置
    new_index = (correct_index + shift) % num_options
    return LABELS[new_index]

def always_choose_a_model(prompt, options):
    # 模拟一个只会选 A 的糟糕模型
    return "I choose A."

def extractor(raw_output):
    # 极简 extractor：真实系统里通常由更强的 LLM 完成
    raw_output = raw_output.strip().upper()
    for label in LABELS:
        if label in raw_output:
            return label
    return None

def vanilla_eval(dataset, model_fn):
    hit = 0
    stats = defaultdict(lambda: {"total": 0, "hit": 0})
    for q in dataset:
        pred = extractor(model_fn(q.qid, q.options))
        gold = LABELS[q.correct_index]
        ok = pred == gold
        hit += int(ok)
        stats[q.ability]["total"] += 1
        stats[q.ability]["hit"] += int(ok)
    return hit / len(dataset), stats

def circular_eval(dataset, model_fn):
    hit = 0
    stats = defaultdict(lambda: {"total": 0, "hit": 0})
    for q in dataset:
        passed = True
        for shift in range(len(q.options)):
            rotated = rotate_right(q.options, shift)
            pred = extractor(model_fn(q.qid, rotated))
            gold = remap_correct_label(len(q.options), q.correct_index, shift)
            if pred != gold:
                passed = False
                break
        hit += int(passed)
        stats[q.ability]["total"] += 1
        stats[q.ability]["hit"] += int(passed)
    return hit / len(dataset), stats

dataset = [
    Question("q1", "目标计数", ["1", "2", "3", "4"], 0),  # A
    Question("q2", "属性推理", ["red", "blue", "green", "yellow"], 1),  # B
    Question("q3", "社会关系", ["teacher", "student", "parent", "doctor"], 2),  # C
    Question("q4", "空间关系", ["left", "right", "top", "bottom"], 3),  # D
]

vanilla_acc, vanilla_stats = vanilla_eval(dataset, always_choose_a_model)
circular_acc, circular_stats = circular_eval(dataset, always_choose_a_model)

assert abs(vanilla_acc - 0.25) < 1e-9
assert abs(circular_acc - 0.0) < 1e-9
assert vanilla_stats["目标计数"]["hit"] == 1
assert circular_stats["目标计数"]["hit"] == 0

print("vanilla_acc =", vanilla_acc)
print("circular_acc =", circular_acc)
```

这段代码里有几个关键变量：

| 变量 | 含义 |
| --- | --- |
| `shift` | 当前右移轮次 |
| `rotated` | 右移后的选项列表 |
| `correct_index` | 原始正确答案所在位置 |
| `gold` | 当前轮次正确标签 |
| `passed` | 该题是否全轮通过 |
| `stats` | 按能力维度累计统计结果 |

如果要接入真实评测系统，数据结构通常至少需要三层信息：

1. 题目级：`qid`、图片路径、问题文本、选项、正确答案、语言。
2. 能力级：该题属于哪个维度，是否还带有更高层分组。
3. 结果级：原始模型输出、抽取标签、每轮是否命中、最终是否通过。

再给一个更贴近真实工程的伪代码版本：

```python
for question in dataset:
    all_correct = True
    for shift in range(len(question.options)):
        rotated = rotate(question.options, shift)
        prompt = build_prompt(question.image, question.text, rotated)
        raw_pred = run_multimodal_model(prompt)
        pred_label = gpt4_extractor(
            question=question.text,
            options=rotated,
            raw_answer=raw_pred
        )
        gold_label = remap_gold_label(question.correct_label, shift)
        if pred_label != gold_label:
            all_correct = False
            break
    ability_stats[question.ability]["total"] += 1
    ability_stats[question.ability]["hit"] += int(all_correct)
```

真实工程例子：假设你要给双语客服机器人选底座模型。候选模型有三个，都支持图文输入。你不应该只看一个总分，而应该：

- 同时跑中文和英文题集。
- 按 20 个能力维度输出准确率表。
- 重点看与业务相关的维度，比如文字识别、属性推理、社会关系、图文对齐。
- 保留每题每轮原始输出，抽查低分维度的失败样本。

这样你得到的不只是“模型 X 比模型 Y 高 3 分”，而是“模型 X 在中文 OCR 和关系理解上明显更稳，模型 Y 在英文常识题更强”。后续无论做微调还是 prompt 规划，方向都会更清晰。

---

## 工程权衡与常见坑

MMBench 的核心优点是评测更稳，但代价也是真实存在的：更贵、更慢、实现更复杂。

先看 VanillaEval 和 CircularEval 的差异。

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| VanillaEval | 实现简单、成本低、跑得快 | 容易受位置偏好影响，非标准输出难匹配 | 快速冒烟测试 |
| CircularEval | 能压制位置偏差，结果更稳 | 轮次更多，成本高，实现复杂 | 正式对比、模型选型 |
| CircularEval + LLM 抽取 | 同时解决位置偏差和自由回答匹配问题 | 额外增加抽取成本和依赖 | 高可信评测 |

一个典型坑是：把“单轮答对率”误当成“真实能力”。例如四选一题里，一个喜欢总选 `A` 的模型，即使完全不会看图，也可能拿到约 25% 的分数。很多人看到 30% 左右的基线分，会误以为模型“具备基础能力”，但其实可能只是位置偏好。

第二个坑是：忽略抽取步骤。模型回答“应该是左边穿蓝衣服的人”，如果你不做 choice extraction，这个答案无法直接匹配成 `B`，就会被误判成错。大量此类误判会让分数系统性偏低，尤其在模型偏好解释性回答时更严重。

第三个坑是：没有保存每轮日志。CircularEval 一题会跑多轮，如果只存最终 `True/False`，排查时几乎无法定位问题。至少要记录：

- 原始问题与图片 ID
- 当前轮次 `shift`
- 轮换后的选项
- 模型原始输出
- 抽取后的标签
- 当前轮金标准标签

第四个坑是：中英文混跑但不分语言汇总。MMBench 支持双语评测，但中文和英文结果往往不对称。如果你只给一个总分，会把语言差异淹没掉。对跨语言业务，这个损失很大。

下面是一个粗略成本估算表。假设题库为 3000 题，平均每题 4 个选项。

| 方案 | 模型调用次数 | 抽取调用次数 | 总轮次量级 |
| --- | ---: | ---: | ---: |
| 单轮 VanillaEval | 3000 | 0 或 3000 | 3000 |
| 4 轮 CircularEval | 12000 | 0 | 12000 |
| 4 轮 CircularEval + Extractor | 12000 | 12000 | 24000 次处理 |

这还没算失败重试、限流、日志写入和异常样本复核的开销。所以它不适合作为每次提交都跑的 CI 任务，更适合阶段性评测。

一个很实用的判断标准是：

- 你在做日常回归检查，用小样本单轮即可。
- 你在做模型采购、版本晋级、论文对比，必须上更严格的 CircularEval。
- 你在做跨语言能力评估，Extractor 和细粒度维度统计基本是必需项。

---

## 替代方案与适用边界

不是所有团队都需要完整跑一遍 MMBench。完整方案信度高，但成本也高，所以可以按目标做裁剪。

| 方案 | 成本 | 信度 | 适合什么场景 |
| --- | --- | --- | --- |
| 完整 MMBench | 高 | 高 | 正式模型选型、论文复现 |
| 采样化 MMBench | 中 | 中 | 快速筛模型、发现短板 |
| 单轮选择题评测 | 低 | 低 | 冒烟测试、开发早期 |
| 开放式生成评测 | 中到高 | 取决于指标 | 关注长答案质量、对话体验 |

### 1. 采样化版本

如果不想跑 3000+ 题，一个务实做法是：每个能力维度抽 20 题到 50 题，再保留 CircularEval。这样可以显著降低成本，同时还能看出哪些能力明显偏弱。

代价是统计波动变大。样本越少，分数越容易受个别题影响。换句话说，它更适合做方向判断，不适合做精确排名。

### 2. 少轮数近似

另一种折中是减少轮数。例如四选一题本来跑 4 轮，你只跑 2 轮或 3 轮。这样可以部分检查位置偏差，但不能像全轮一样严格。它的本质是用更低成本换更弱的偏差控制能力。

### 3. 用任务集补充，而不是替代

如果你的真实目标是做图文客服、图表问答、票据识别或教学辅助，仅靠 MMBench 不够。更合理的做法是：

- 用 MMBench 看基础能力画像。
- 再用业务任务集看端到端效果。
- 必要时加人工误例分析，看失败模式是否与业务强相关。

也就是说，MMBench 更像“体检”，而不是“上线前验收”。

### 4. 与开放式评测配合

如果你关心的是“答案是否自然、解释是否可信、对话是否连贯”，那就必须引入开放式评测。因为选择题只能验证“有没有选中”，不能充分衡量“为什么这样回答”和“表达质量如何”。

所以它的适用边界可以概括为：

- 适合：模型横向比较、能力画像、双语对比、微调方向判断。
- 不适合：单独用于上线验收、替代真实业务数据、评价开放式生成质量。

真实工程里，一个常见组合是：先用 OpenCompass 或类似平台跑 MMBench 中英版本，筛掉明显短板模型；再拿剩下的候选模型跑业务数据集，最后做人工抽样验收。这样时间和结论质量更平衡。

---

## 参考资料

| 资料 | 作用 | 说明 |
| --- | --- | --- |
| MMBench ECCV 2024 论文 | 核心定义与方法来源 | 用于确认题目规模、能力维度、评测设计 |
| 论文 Section 4.3 | CircularEval 机制来源 | “全轮一致才算命中”的直接依据 |
| OpenCompass 项目页 | 工程实现与平台集成参考 | 用于查看中英评测、结果展示与基准接入方式 |

1. Liu et al., *MMBench: Is Your Multi-modal Model an All-around Player?*, ECCV 2024. 可用于复查数据规模、20 个能力维度、choice extractor 与 CircularEval 的原始描述。
2. 同一论文的 Section 4.3。可用于核对 CircularEval 的轮换逻辑、位置偏差问题与“全轮全对”的判定方式。
3. OpenCompass / OmniMMBench 相关说明页。可用于查看工程侧如何组织多模型、多语言、多能力维度的评测展示。
