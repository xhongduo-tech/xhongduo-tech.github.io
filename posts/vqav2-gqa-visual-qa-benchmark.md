## 核心结论

VQAv2 和 GQA 都是视觉问答基准。视觉问答，指模型输入一张图像和一个自然语言问题，然后输出一个简短答案。二者的共同任务形式是：

$$
f(x, q) \rightarrow \hat{a}
$$

其中 $x$ 是图像，$q$ 是问题，$\hat{a}$ 是模型预测答案。

VQAv2 的核心价值是“削弱语言偏置后的标准视觉问答基准”。语言偏置，指模型不认真看图，只根据问题模板和训练集分布猜答案。例如看到问题 “What color is the banana?” 就倾向回答 “yellow”。VQAv2 通过“相似图片、相同问题、不同答案”的配对设计，迫使模型更多依赖图像内容。

GQA 的核心价值是“组合推理与一致性评测基准”。组合推理，指模型不能只识别单个物体，而要把颜色、位置、关系、数量等条件连起来判断。例如“左边那个红色盒子里有几个苹果”需要先定位左边盒子，再确认红色属性，再识别盒内物体，最后计数。

| 基准 | 目标 | 题型特点 | 主要指标 | 适用场景 |
|---|---|---|---|---|
| VQAv2 | 衡量模型是否真的看图，而不是只靠语言猜测 | 常规问答、属性识别、yes/no、简单计数 | VQA accuracy | 通用视觉问答、基础识别、去语言偏置评测 |
| GQA | 衡量组合推理、多步判断和答案自洽性 | 属性、关系、空间、计数、多跳问题 | accuracy、consistency、validity、plausibility | 复杂视觉推理、关系理解、一致性诊断 |

玩具例子可以这样理解：问“这个球是什么颜色”，VQAv2 更关注模型是否能从图中识别出 “red”。问“左边红色盒子里有几个苹果”，GQA 更关注模型是否完成了“找左边物体 -> 判断颜色 -> 找内部物体 -> 计数”这条推理链。

对初级工程师来说，结论很直接：VQAv2 更适合看模型的基础视觉问答能力，GQA 更适合看模型在复杂问题上的推理能力和自洽性。一个 VLM 在 VQAv2 上高分，不代表它已经能稳定解决真实业务里的多条件视觉判断。

---

## 问题定义与边界

VQAv2 和 GQA 都可以写成同一个任务：

$$
f(x, q) = \hat{a}
$$

输入是图像 $x$ 和问题 $q$，输出是预测答案 $\hat{a}$。评测时再把 $\hat{a}$ 和标准答案比较。区别不在任务接口，而在数据构造、问题复杂度和指标设计。

VQAv2 主要覆盖常规问答与去偏置识别。它关心模型是否能在相似图片之间给出不同答案。例如同样问“桌子上是什么水果”，一张图是苹果，另一张图是香蕉，模型不能只靠问题模板猜“banana”。

GQA 主要覆盖场景图驱动的组合推理。场景图，指把图像中的物体、属性和关系表示成结构化图，例如 `apple --on--> plate`、`box --left of--> cup`。GQA 的问题通常由功能程序生成。功能程序，指一组可执行的推理步骤，例如先筛选红色物体，再找它左边的对象，再判断类别。

同一张厨房图片里，问“这是苹果吗”更接近 VQAv2 风格，因为它主要测试基础识别。问“苹果在盘子左边还是右边、数量是否一致、是否被遮挡”更接近 GQA 风格，因为它测试空间关系、计数和条件组合。

| 问题类型 | 含义 | VQAv2 覆盖程度 | GQA 覆盖程度 |
|---|---|---:|---:|
| yes/no | 判断命题是否成立，例如“这是猫吗” | 高 | 高 |
| 属性 | 识别颜色、材质、形状等属性 | 高 | 高 |
| 关系 | 判断物体之间的位置或语义关系 | 中 | 高 |
| 计数 | 输出物体数量 | 中 | 高 |
| 多跳组合推理 | 多个条件连续推导，例如“左边红盒里的水果数量” | 低到中 | 高 |

边界也要明确。VQAv2 和 GQA 都不等价于开放世界对话。开放世界对话，指模型可以围绕图像进行多轮、长上下文、带常识扩展的交流。它们也不等价于 OCR 密集型文档理解。OCR，指识别图像中的文字；如果任务是读票据、合同、海报或菜单，VQAv2/GQA 只能提供部分参考，不能替代专门的文档理解评测。

因此，使用这两个基准时要先问清楚业务目标：你是在测“模型能不能看懂常见图像问题”，还是在测“模型能不能执行多步视觉推理”。前者更偏 VQAv2，后者更偏 GQA。

---

## 核心机制与推导

VQAv2 的标准准确率不是简单的 0/1 分类。它使用多人标注容错。多人标注容错，指同一道题允许多个合理人类答案存在，模型只要和足够多的人类答案一致，就可以得到满分。

常见写法是：

$$
Acc_{vqa} = avg_i \min(\frac{n_i}{3}, 1)
$$

其中 $n_i$ 是第 $i$ 道题中，10 个人工答案里与模型预测答案完全匹配的数量。`avg_i` 表示对所有题目取平均。

玩具例子：某题 10 个人工答案里有 7 个是 `red`，模型预测 `red`，则：

$$
Acc_{vqa} = \min(\frac{7}{3}, 1) = 1
$$

如果只有 1 个人回答 `red`，模型预测 `red`，则得分是：

$$
\min(\frac{1}{3}, 1) = 0.333
$$

这说明 VQAv2 看的是“模型答案和人类共识有多接近”，不是只看唯一标准答案。

GQA 除了准确率，还强调一致性。一致性，指模型答对一个源问题后，对由这个答案推出的相关问题也应该答对。例如源问题是“红色盒子在桌子上吗”，模型回答“是”。那么蕴含问题可能包括“桌子上有盒子吗”“盒子是什么颜色”“红色物体在哪里”。蕴含问题，指可以从原问题和答案中逻辑推出的问题。

GQA 的一致性可以写成：

$$
Cons = \frac{1}{|Q^+|} \sum_{q \in Q^+} \frac{1}{|E_q|} \sum_{q' \in E_q} \mathbf{1}[f(x,q') = a(q')]
$$

其中 $Q^+$ 是模型答对的源问题集合，$E_q$ 是由源问题 $q$ 可推出的蕴含问题集合，$\mathbf{1}[\cdot]$ 是指示函数：条件成立为 1，不成立为 0。

数值例子：某个源问题对应 4 个蕴含问题，模型答对 3 个，则该源问题的一致性是：

$$
\frac{3}{4} = 75\%
$$

新手版解释是：VQAv2 看你和人类答案有多接近，GQA 看你答对一个问题后，相关问题是否也自洽。

从源问题到蕴含问题的推导可以写成：

```text
源问题：左边的红色盒子里有几个苹果？
答案：2

可推出的问题：
1. 图中有盒子吗？ -> yes
2. 盒子是什么颜色？ -> red
3. 苹果在哪里？ -> in the box
4. 盒子里有几个苹果？ -> 2
```

| 指标 | 含义 | 主要回答的问题 |
|---|---|---|
| 准确率 accuracy | 预测答案是否等于标准答案 | 单题有没有答对 |
| 一致性 consistency | 答对源问题后，相关蕴含问题是否也答对 | 推理结果是否自洽 |
| 有效性 validity | 答案是否属于该问题类型允许的答案空间 | 答案格式是否合理 |
| 合理性 plausibility | 答案是否符合常识或数据分布 | 答案是否像一个可能答案 |

这几个指标不能互相替代。模型可能准确率不低，但一致性差；也可能输出格式有效，但视觉判断错误。工程评测必须把它们拆开看。

---

## 代码实现

实现评测时，应把数据加载、答案归一化、指标计算、分类型统计拆开。答案归一化，指把模型输出和标准答案转换成可比较形式，例如去空格、转小写、统一数字格式。否则 `Red`、` red ` 和 `red` 会被误判成不同答案。

下面是一个可运行的最小 Python 例子，展示 VQAv2 分数、GQA 一致性和按题型汇总的基本结构：

```python
from collections import defaultdict

def normalize(ans):
    return str(ans).strip().lower()

def vqav2_score(pred, human_answers):
    n = sum(normalize(a) == normalize(pred) for a in human_answers)
    return min(n / 3.0, 1.0)

def gqa_consistency(model, image, entailment_qs, gold_answers):
    correct = 0
    for q, gold in zip(entailment_qs, gold_answers):
        pred = model(image, q)
        correct += int(normalize(pred) == normalize(gold))
    return correct / max(len(entailment_qs), 1)

def summarize_by_type(records):
    bucket = defaultdict(list)
    for r in records:
        bucket[r["type"]].append(r["score"])
    return {k: sum(v) / len(v) for k, v in bucket.items()}

def toy_model(image, question):
    answers = {
        "what color is the box?": "red",
        "is there a box?": "yes",
        "what is in the box?": "apple",
        "how many apples are in the box?": "2",
    }
    return answers.get(normalize(question), "unknown")

# VQAv2 toy example
human_answers = ["red", "red", "red", "red", "red", "red", "red", "orange", "pink", "red"]
assert vqav2_score("red", human_answers) == 1.0
assert round(vqav2_score("orange", human_answers), 3) == 0.333

# GQA toy example
entailment_qs = [
    "Is there a box?",
    "What color is the box?",
    "What is in the box?",
    "How many apples are in the box?",
]
gold_answers = ["yes", "red", "apple", "2"]
assert gqa_consistency(toy_model, "toy_image.jpg", entailment_qs, gold_answers) == 1.0

# Type-level summary
records = [
    {"type": "attribute", "score": vqav2_score("red", human_answers)},
    {"type": "counting", "score": 0.5},
    {"type": "relation", "score": 0.25},
]
summary = summarize_by_type(records)
assert summary["attribute"] == 1.0
assert summary["relation"] == 0.25
```

实际工程中，VQAv2 和 GQA 的评测代码最好分开写，但共享同一套输入接口。这样同一个 VLM 可以被放到两类评测中比较：基础识别问题用 VQAv2 风格指标，复杂推理问题用 GQA 风格指标。

| 流水线阶段 | 输入 | 处理 | 输出 | 日志记录项 |
|---|---|---|---|---|
| 数据加载 | image、question、gold answer、type | 校验字段和文件路径 | 标准样本对象 | 样本 ID、题型、数据版本 |
| 模型推理 | image、question | 调用 VLM 生成答案 | raw prediction | prompt、模型版本、耗时 |
| 答案归一化 | raw prediction、gold answer | 小写、去空格、规则替换 | normalized answer | 归一化前后文本 |
| 指标计算 | prediction、gold answer | 计算 VQA score 或 consistency | 分数 | 指标名、题型、是否命中 |
| 汇总分析 | 多条样本分数 | 按题型、难度、场景聚合 | 报告 | 总分、分组分数、失败样本 |

核心原则是：不要让模型输出逻辑和评测逻辑混在一起。模型可以升级，评测脚本必须稳定；数据可以扩展，指标定义必须可复现。

---

## 工程权衡与常见坑

只看总准确率是最常见的误判来源。一个模型可能在 yes/no 和颜色题上表现很好，但在关系题、计数题和多跳题上明显掉分。总分会把简单题的高分和复杂题的低分平均掉，让问题不明显。

| 题型 | 示例 | 常见模型表现 | 工程风险 |
|---|---|---|---|
| yes/no | “这是苹果吗” | 容易做高 | 可能靠分布猜测 |
| 属性 | “盒子是什么颜色” | 相对稳定 | 光照、遮挡会影响结果 |
| 关系 | “苹果在盘子左边吗” | 容易掉分 | 空间判断错误 |
| 计数 | “有几个瓶子” | 容易不稳定 | 密集物体误计数 |
| 组合推理 | “左边红盒里的苹果有几个” | 最容易掉分 | 多条件任一步错都会失败 |

真实工程例子：零售货架巡检模型需要判断商品是否摆放正确。VQAv2 类问题可以测“这是什么商品”“包装是什么颜色”。GQA 类问题更接近真实需求，例如“这个商品是否在盒子左边”“是否被遮挡”“同类商品数量是否一致”。如果只报一个总准确率，模型可能看起来可上线，但上线后在遮挡、错位、数量不一致这些场景里持续失败。

一个简短失败日志可能是这样：

```text
image_id=store_1027
q1="What product is this?" pred="milk" gold="milk" score=1
q2="Is the milk carton left of the red box?" pred="yes" gold="no" score=0
q3="How many milk cartons are behind the box?" pred="3" gold="2" score=0
overall_sample_score=0.33
```

如果评测集里大量是 q1 这种简单识别题，总分会被拉高；但业务真正关心的是 q2 和 q3。

| 坑 | 后果 | 规避方法 |
|---|---|---|
| 只看总准确率 | 掩盖复杂题失败 | 按题型、场景、难度分组统计 |
| 忽略答案归一化 | 同义格式被误判 | 固定 normalize 规则并记录版本 |
| 把 VQAv2 高分等同于推理强 | 高估模型能力 | 同时报告 GQA 或自建组合题 |
| 只测单题，不测一致性 | 同图同义问题自相矛盾 | 加入改写题、蕴含题、反事实题 |
| 测试集过于接近训练分布 | 线上泛化差 | 加入真实业务图片和失败样本回放 |
| 不记录 prompt 和模型版本 | 结果不可复现 | 每次评测保留完整元数据 |

一致性评测也不能只看单题。改写题，指同一个问题换一种说法，例如“盒子是什么颜色”和“这个箱子的颜色是什么”。反事实题，指改变某个条件后重新提问，例如把“红色盒子”改成“蓝色盒子”。如果模型对这些问题给出互相矛盾的答案，说明它可能没有稳定理解图像。

---

## 替代方案与适用边界

VQAv2 更适合衡量通用视觉问答与去偏置能力，但不适合单独代表复杂推理、长链推理或强一致性要求的场景。长链推理，指问题需要多个连续步骤才能得到答案。VQAv2 能告诉你模型是否具备基础看图问答能力，但不能充分说明模型是否能可靠处理多条件业务规则。

GQA 更适合衡量组合推理和自洽性，但它也不是万能指标。如果任务更偏 OCR、文档、开放域常识问答或真实对话，单靠 GQA 仍不够。比如票据识别需要读金额、日期、商户名；菜单理解需要识别文字和版式；海报理解需要文字、图像、设计元素共同解析。这些任务应该额外使用 OCR 和文档理解类基准。

| 基准 | 更适合解决的问题 | 不适合单独代表的问题 |
|---|---|---|
| VQAv2 | 通用视觉问答、基础识别、削弱语言偏置 | 多步推理、强一致性、业务规则链 |
| GQA | 场景图关系、组合推理、一致性诊断 | OCR 密集文档、开放世界长对话 |
| VQA-CP | 检查模型是否过度依赖语言先验 | 全面视觉能力评估 |
| VQA-Rephrasings | 检查同义改写下是否稳定 | 复杂空间推理和计数能力 |

选择基准时可以按任务目标判断：

| 业务目标 | 更合适的评测方向 |
|---|---|
| 识别商品是什么 | VQAv2 风格指标更直接 |
| 判断货架上左边红盒是否被遮挡 | GQA 风格组合推理更接近需求 |
| 判断数量是否匹配 | GQA 加自建计数测试 |
| 读取票据、海报、菜单 | 需要 OCR 或文档理解基准 |
| 检查同一图多种问法是否稳定 | VQA-Rephrasings 或自建改写集 |
| 检查模型是否只靠问题猜答案 | VQAv2、VQA-CP 都有参考价值 |

工程上更稳妥的做法是组合评测：用 VQAv2 风格题检查基础识别，用 GQA 风格题检查关系和推理，用改写题检查一致性，用真实业务数据检查上线风险。单个公开基准不能替代完整验收，只能回答一部分能力问题。

---

## 参考资料

### 官方站点

- [VQAv2 官方站](https://visualqa.org/)：数据版本、任务说明、评测入口和数据规模说明。
- [GQA 官方站](https://cs.stanford.edu/people/dorarad/gqa/about.html)：场景图、功能程序、多步推理和一致性指标说明。
- [GQA 评测页](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html)：accuracy、consistency、validity、plausibility 等指标定义。

### 论文

- [VQAv2 论文：Making the V in VQA Matter](https://arxiv.org/abs/1612.00837)：提出通过配对相似图片、不同答案来削弱语言偏置。
- [GQA 论文：arXiv 1902.09506](https://arxiv.org/abs/1902.09506)：介绍基于场景图和功能程序的数据构造方法，以及细粒度诊断指标。

### 扩展评测

- [VQA-Rephrasings / Cycle-Consistency for Robust Visual Question Answering](https://arxiv.org/abs/1902.05660)：使用问题改写和循环一致性评测模型稳定性。
- VQA-CP：用于观察模型是否依赖训练集中的语言先验，适合补充分析语言偏置问题。
