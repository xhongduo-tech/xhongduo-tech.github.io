## 核心结论

VQAv2 和 GQA 都是视觉问答基准。视觉问答，指模型输入一张图像和一个自然语言问题，输出一个简短答案。两者都测“模型是否看懂图”，但侧重点不同。

VQAv2 解决的是“视觉问答会不会被语言偏置骗分”的问题。语言偏置，指模型不看图也能靠问题模式猜答案，例如看到 `What color is the sky?` 就猜 `blue`。VQAv2 通过构造相似问题和互补图像，降低这种猜题收益，因此适合衡量基础视觉理解是否稳定。

GQA 进一步把问题推进到“模型是否真的能做组合推理和多步推断”。组合推理，指把多个条件合在一起判断；多步推断，指答案不能一步得到，必须先定位对象、判断属性、分析关系，再输出结果。GQA 更适合检验图像理解是否可解释、可追踪、可约束。

玩具例子：同样问“这张图里有没有车”，VQAv2 主要看模型能不能给出符合标注分布的答案。如果问题变成“左边那个红杯子旁边有没有叉子”，GQA 会要求模型先定位“左边”“红杯子”，再判断“旁边”关系，最后确认“叉子”是否存在。

| 基准 | 任务目标 | 问题类型 | 评测重点 | 常见失真 |
|---|---|---|---|---|
| VQAv2 | 基础视觉问答 | 颜色、数量、存在性、物体类别 | 减少语言偏置后的答案准确率 | 把软评分误当 exact match |
| GQA | 诊断型视觉推理 | 关系、属性、位置、多条件组合 | 准确率、一致性、有效性、可追踪性 | 只看 overall accuracy，忽略推理链 |
| 二者结合 | 上线前基础验收 | 简单问答 + 关系链问题 | 简单题稳定性与复杂题可靠性差距 | 简单题高分被误读为复杂理解能力强 |

---

## 问题定义与边界

形式化地说，视觉问答任务输入图像 $I$ 和问题 $q$，模型输出预测答案 $\hat a$，评测系统把 $\hat a$ 与标注答案集合 $A$ 对齐后打分。这里的标注答案集合，指同一道题由多个人类标注者给出的答案列表。

| 术语 | 白话解释 |
|---|---|
| 图像 $I$ | 被提问的图片 |
| 问题 $q$ | 针对图片提出的自然语言问题 |
| 预测答案 $\hat a$ | 模型给出的答案 |
| 标注答案集合 $A$ | 多个人类标注者给出的参考答案 |
| 根问题 | GQA 中被重点评估的主问题 |
| entailed question | 由根问题语义推出的相关问题，用来检查答案是否一致 |

VQAv2 的边界是“基础视觉问答基准”。它不要求复杂推理，但要求模型答案尽量不被语言先验主导。例如 `What color is the bus?` 可以理解为“直接看公交车颜色就能答”。如果模型真正看图，应该能区分黄色公交车、红色公交车、白色公交车，而不是只记住训练集中常见颜色。

GQA 的边界是“诊断型推理基准”。它不只是测单题对错，还测根问题与推论问题之间的一致性。例如 `What is on the table next to the plate?` 可以理解为“先找桌子，再找盘子，再看盘子旁边是什么”。模型如果主问题答“fork”，但在相关问题里又说“盘子旁边没有叉子”，就暴露出内部理解不一致。

什么时候用 VQAv2：当目标是确认模型是否具备基础看图问答能力，例如颜色、数量、物体存在性、常见属性。什么时候用 GQA：当目标是确认模型能否处理关系链、组合条件和多步定位。什么时候两者都不够：当问题依赖读图中文字、外部常识、开放域知识、长链跨图推理或视频时，VQAv2 和 GQA 都只能覆盖一部分能力。

---

## 核心机制与推导

VQAv2 使用软准确率，不是严格 exact match。软准确率，指答案不是只有 0 分和 1 分，而是按有多少标注者同意来给部分分。常用简化公式是：

$$
Acc_{VQA}(q) = \min(m/3, 1)
$$

其中 $m$ 是 10 个标注答案中与模型预测 $\hat a$ 一致的个数。例子：10 个标注里 8 个写 `yes`，模型答 `yes`，则：

$$
Acc_{VQA} = \min(8/3, 1) = 1
$$

如果只有 1 个标注者写 `yes`，模型答 `yes`，分数就是 $\min(1/3, 1)=0.333$。这比普通 accuracy 更适合 VQA，因为自然语言答案存在同义表达、标注差异和轻微歧义。

GQA 的核心不是单题准确率，而是一组诊断指标。consistency 表示一致性；validity 表示答案是否属于合理答案空间；plausibility 表示答案对该问题类型是否合理；grounding 表示模型是否把问题和图像区域对应起来；distribution 表示预测分布是否异常集中。

GQA 一致性可以写成：

$$
Cons = \frac{1}{|Q|}\sum_{q \in Q}\frac{1}{|E_q|}\sum_{q' \in E_q}\mathbf{1}[\hat a(q') = a(q')]
$$

其中 $Q$ 是答对的根问题集合，$E_q$ 是根问题 $q$ 对应的 entailed question 集合。例子：根问题答对，但 2 个 entailed 问题只对 1 个，则该根问题一致性是 $1/2=50\%$。

机制流程图：

```text
图像 + 问题
    |
    v
模型预测
    |
    v
答案归一化 / 对齐
    |
    v
评分
    |
    +--> VQAv2: 与 10 个标注答案做软匹配
    |
    +--> GQA: 按根问题和 entailed questions 统计一致性
```

| 项目 | VQAv2 软评分 | GQA 一致性评分 |
|---|---|---|
| 基本单位 | 单个问题 | 根问题及其推论问题组 |
| 关键输入 | 10 个人类答案 | 根问题、推论问题、标准答案 |
| 得分逻辑 | 至少 3 人同意通常即可满分 | 根问题答对后检查推论问题是否自洽 |
| 主要目的 | 容忍自然语言标注差异 | 暴露多步推理矛盾 |

最小推导是：VQAv2 不能用普通 accuracy 直接算，因为同一道题可能有多个合理答案。比如图片里有一只小狗，有人答 `dog`，有人答 `puppy`，严格字符串匹配会把合理差异误判为错。GQA 需要按根问题分组统计，因为它关心“同一个视觉事实链条是否一致”。如果把所有 entailed questions 打散成普通题目，就看不出模型是在某个推理链上稳定失败，还是随机错了几个孤立问题。

---

## 代码实现

实现 VQAv2 评测的关键是答案归一化和与 10 个标注的软匹配，不能直接做字符串 exact match。答案归一化，指把大小写、空格、常见数字写法等处理成统一形式。真实评测应尽量使用官方脚本；下面代码只展示最小机制。

伪代码 1：VQAv2 评分流程。

```text
输入: 模型答案 pred，10 个标注答案 answers
pred_norm = normalize_answer(pred)
m = 0
for a in answers:
    if normalize_answer(a) == pred_norm:
        m += 1
return min(m / 3, 1)
```

伪代码 2：GQA consistency 评分流程。

```text
输入: 根问题列表 root_questions，预测表 pred，标准答案表 gt
total = 0
count = 0
for root in root_questions:
    if pred[root] != gt[root]:
        continue
    local = root 的 entailed questions 正确率
    total += local
    count += 1
return total / count
```

代码结构建议保持三个函数：`normalize_answer()` 负责归一化；`score_vqa()` 负责 VQAv2 单题软评分；`score_gqa_consistency()` 负责 GQA 分组一致性。

```python
import re

_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
}

def normalize_answer(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return _NUMBER_WORDS.get(text, text)

def score_vqa(pred, answers):
    pred_norm = normalize_answer(pred)
    m = sum(pred_norm == normalize_answer(a) for a in answers)
    return min(m / 3.0, 1.0)

def score_gqa_consistency(root_questions, pred, gt):
    total = 0.0
    count = 0

    for root in root_questions:
        root_id = root["id"]
        entailed = root.get("entailed", [])

        if normalize_answer(pred[root_id]) != normalize_answer(gt[root_id]):
            continue
        if not entailed:
            continue

        correct = sum(
            normalize_answer(pred[eid]) == normalize_answer(gt[eid])
            for eid in entailed
        )
        total += correct / len(entailed)
        count += 1

    return total / count if count else 0.0

answers = ["yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "no", "no"]
assert score_vqa("YES", answers) == 1.0

answers_number = ["two", "2", "two", "three", "two", "2", "two", "one", "two", "two"]
assert score_vqa("2", answers_number) == 1.0

roots = [{"id": "q_root", "entailed": ["q_e1", "q_e2"]}]
pred = {"q_root": "yes", "q_e1": "red", "q_e2": "fork"}
gt = {"q_root": "yes", "q_e1": "red", "q_e2": "knife"}
assert score_gqa_consistency(roots, pred, gt) == 0.5
```

新手版 VQAv2 流程可以概括为：先把模型输出统一成标准答案格式，再统计它和 10 个标注答案的重合次数，最后套用 $\min(m/3,1)$。新手版 GQA 流程可以概括为：先找到一组“主问题 + 推论问题”，主问题答对后，再检查推论问题是否都自洽。

真实工程例子：你在做一个电商多模态助手，用户上传货架照片并提问“红色包装旁边有几瓶饮料”。VQAv2 类测试能检查模型是否看得见颜色、数量和物体；GQA 类测试能检查模型是否稳定处理“红色包装”“旁边”“饮料”这几个约束。两类测试都过，才更接近可上线的基础视觉问答能力。

---

## 工程权衡与常见坑

VQAv2 的常见坑是把它当成严格分类题，忽略官方答案归一化和软评分，导致离线分数和官方分数不一致。比如 `two` 和 `2` 是否算同一个答案，取决于官方归一化流程，不是自己手写几条规则就能完全替代。

GQA 的常见坑是只看 overall accuracy，不看 question type、reasoning steps、validity、grounding。overall accuracy，指所有题目混在一起算一个总准确率。这个数字容易把“会猜简单题”误判成“会理解复杂图像”。如果模型在 `yes/no` 上很高，但在三步关系推理上很低，单个总分会掩盖风险。

| 坑点 | 影响 | 规避方式 |
|---|---|---|
| VQAv2 用 exact match | 离线分数偏低或偏高 | 使用官方归一化和软评分 |
| 忽略 `two` 与 `2` | 数字题评分不稳定 | 对照官方 evaluation 脚本 |
| 只看 overall accuracy | 简单题掩盖复杂题失败 | 按问题类型和推理步数分桶 |
| GQA 根问题答错仍统计 entailed | 把错误传播放大成评测噪声 | consistency 只在根问题答对样本上算 |
| 不查答案分布 | 模型可能大量输出高频答案 | 监控 distribution 与类别分布 |
| 不查 grounding | 模型可能答对但没看对区域 | 结合区域标注或可解释性检查 |

上线验收可以分两层。第一层用 VQAv2 类测试检查基础问答：颜色、数量、有没有、是什么。第二层用 GQA 类测试检查关系链和约束一致性：左边、右边、旁边、同色、包含、被遮挡、多对象组合。对于多模态助手，尤其要跑一致性测试，因为用户很容易在同一张图上连续追问。如果模型前一句说“桌上有叉子”，后一句又说“盘子旁边没有餐具”，体验上就是明显不可靠。

指标分桶建议至少包括：VQAv2 按 `yes/no`、`number`、`other` 分桶；GQA 按 `question type`、`reasoning steps`、关系类型、属性类型分桶。这样才能看到简单问题和复杂问题的性能差距。简单题高分是必要条件，不是充分条件；复杂题低分通常更接近真实产品风险。

---

## 替代方案与适用边界

如果目标是测抗语言偏置，VQAv2 是基础基线，但 VQA-CP 这类更偏向偏置控制的数据集可能更敏感。VQA-CP 的核心思路是改变训练集和测试集中的答案分布，让靠语言先验猜题的模型更容易暴露问题。

如果目标是测特定能力，GQA 之外还需要其他数据集补足短板。CLEVR 更强调合成场景下的可控组合推理；TextVQA 专门测图中文字读取；OK-VQA 更依赖外部常识；这些能力不是 VQAv2 和 GQA 的主要覆盖范围。

| 任务目标 | 推荐基准 | 适用原因 | 不足 |
|---|---|---|---|
| 看图就能答 | VQAv2 | 覆盖基础视觉问答，标注成熟 | 复杂推理诊断不足 |
| 抗语言偏置 | VQA-CP | 训练和测试答案分布变化更明显 | 可能过度强调分布迁移 |
| 多条件组合 | GQA | 有问题结构和一致性指标 | 仍不能覆盖开放域知识 |
| 可控逻辑推理 | CLEVR | 合成数据可精确控制推理链 | 与真实图像分布有差距 |
| 读图中文字 | TextVQA | 专门考察 OCR 与视觉文本理解 | 不代表一般视觉推理能力 |
| 常识问答 | OK-VQA | 需要图像信息加外部知识 | 答案空间更开放，评测更难 |

新手理解版可以这样选：想测“看图就能答”，优先 VQAv2；想测“图里多个条件能不能一起满足”，优先 GQA；想测“读图中文字”，要换 TextVQA；想测“常识推理”，要看 OK-VQA。

边界总结：VQAv2 和 GQA 都不能完整覆盖读字、常识、开放域知识、长链跨图推理、视频时序理解和真实交互式对话。它们的价值在于建立基础能力坐标：VQAv2 告诉你模型是否有稳定的基础视觉问答能力，GQA 告诉你模型在组合约束和多步推理上是否可靠。工程上应把它们当作验收矩阵的一部分，而不是唯一结论。

---

## 参考资料

1. [VQA 官方站](https://visualqa.org/)
2. [VQA evaluation](https://visualqa.org/evaluation.html)
3. [Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering](https://arxiv.org/abs/1612.00837)
4. [GQA 官方站](https://cs.stanford.edu/people/dorarad/gqa/about.html)
5. [GQA evaluate](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html)
6. [GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering](https://arxiv.org/abs/1902.09506)
