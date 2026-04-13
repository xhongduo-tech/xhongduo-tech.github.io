## 核心结论

MATH 数据集是当前最常用的高难度数学推理基准之一，核心用途不是测“会不会算”，而是测模型能不能完成**多步符号推导**。符号推导可以直白理解为：不是只给出一个数字，而是要把代数式、分式、根式、几何关系一步一步变形成目标答案。

它收集了 12,500 道高中数学竞赛题，主要来自 AMC、AIME 等题源，覆盖 7 个主题：代数、数论、几何、计数与概率、初等代数、前代数、预微积分。相比 GSM8K 这类以小学到初中应用题为主的数据集，MATH 更强调形式化推理、LaTeX 表达和最终答案规范输出，因此更适合做“高阶数学能力验收”。

MATH 的评测并不复杂，但很严格。标准流程通常是：让模型输出完整 LaTeX 解答，再从答案中提取 `\boxed{...}` 内的最终表达式，与标准答案做规范化后比较。指标一般写成：

$$
\text{Accuracy} = \frac{\#\text{Correct}}{\text{Total}}
$$

这里的 Accuracy 就是准确率，白话说就是“总共做了多少题，其中最后答对了多少题”。

一个典型玩具例子是分式化简题。假设题目要求在某个约束下化简：

$$
\frac{a^2+b^2}{a-b}
$$

如果已知 $ab=8$ 且需要围绕 $a-b$ 改写，那么常见目标形式可能是：

$$
(a-b) + \frac{16}{a-b}
$$

这类题的难点不在于背公式，而在于把表达式拆开、重组，再用 AM-GM 等不等式工具继续推导。AM-GM 是“算术平均数不小于几何平均数”，白话说就是用平均值关系给表达式找下界，是竞赛题里很常见的工具。MATH 要测的正是这种多步、结构化、不能只靠模板匹配的能力。

| 主题 | 常见内容 | AoPS 难度 | 常见所需技巧 |
|---|---|---:|---|
| 代数 | 方程、恒等变形、不等式 | 2-5 | 因式分解、配方、AM-GM |
| 数论 | 整除、同余、质数 | 3-5 | 模运算、分类讨论 |
| 几何 | 角度、相似、圆 | 3-5 | 辅助线、面积比、坐标化 |
| 计数与概率 | 排列组合、概率模型 | 2-5 | 分类计数、容斥、递推 |
| 初等代数 | 基础表达式与函数 | 1-3 | 代换、展开、整理 |
| 前代数 | 比例、整数、基础逻辑 | 1-2 | 枚举、约束分析 |
| 预微积分 | 三角、复数、序列 | 2-5 | 恒等式、极值、递推 |

---

## 问题定义与边界

MATH 的问题定义很明确：给定一道高中数学竞赛题，要求模型生成完整推理过程，并给出最终的规范答案。这里的“规范”不是写得像人话就行，而是通常要求最终结果放在 `\boxed{...}` 中，便于评测脚本自动抽取。

它的边界也很清楚：

1. 题目来源是竞赛真题或同等风格题，主要来自 AMC、AIME 等体系。
2. 题目数量是 12,500，不是开放式无限扩展题库。
3. 主题限定在 7 个高中核心领域，不覆盖大学数学的系统证明题。
4. 难度按 AoPS 1-5 分级。AoPS 是 Art of Problem Solving 社区常用的难度标注体系，白话说就是“题有多竞赛化、多绕、多需要技巧”。
5. 每题通常附带标准解，便于复现评测和误差分析。

例如题目“某函数在整数解最多多少？”如果它需要分情况讨论定义域、单调性、整点分布，再结合边界取值，就很可能被归到“计数与概率”或“代数+计数”的交叉区域，并且难度可能在 AoPS 4 左右。这里的 AoPS 4 可以简单理解为：不是基础练习，而是需要多个中间结论拼接起来才能完成的题。

从工程视角看，MATH 评估的是一种受限任务，不是全能数学系统。它不能直接说明模型“理解了所有数学”，因为它只覆盖特定题型、特定格式和特定答案判定规则。它更像一把高标准尺子，用来测模型在竞赛风格数学上的推理上限。

简化后的评测流程可以画成这样：

`题目 -> 模型生成 LaTeX 解答 -> 提取 \boxed 内容 -> 规范化字符串 -> 与标准答案对比`

这个流程说明了一个关键事实：MATH 同时测两层能力。

第一层是推理能力。模型得先把题做出来。

第二层是输出能力。模型即便中间想对了，如果最后没把答案放进 `\boxed{}`，也可能在自动评测里被判错。

---

## 核心机制与推导

MATH 的核心机制是“标准答案抽取 + 字符级对比”。这听起来很机械，但正因为机械，才适合做基准评测。只要规则固定，不同模型就能在同一标准下比较。

常见流程可以拆成 4 步：

1. 读取题目与标准解。
2. 让模型输出完整解答，末尾必须包含 `\boxed{final}`。
3. 从模型输出和标准解中分别抽取 `\boxed{...}` 内部表达式。
4. 对表达式做规范化，再比较是否一致。

规范化通常包括以下操作：

1. 去除空格。
2. 去除 `\left` 和 `\right`。
3. 去除多余的 `\ `。
4. 有时还会统一分数、括号、负号形式。

为什么这一步重要？因为 LaTeX 是表示语言，不是数学对象本身。同一个答案，可能写成 `\frac{1}{2}`，也可能写成 `\left(\frac{1}{2}\right)`。如果不清洗格式，很多本来正确的答案会被误判。

准确率定义非常直接：

$$
\text{Accuracy} = \frac{\#\text{Correct}}{\text{Total}}
$$

其中 $\#\text{Correct}$ 是判定为完全匹配的题数，$\text{Total}$ 是总题数。

下面看一个玩具例子。假设题目要求在约束 $ab=8$ 下化简：

$$
\frac{a^2+b^2}{a-b}
$$

先用恒等变形：

$$
a^2+b^2=(a-b)^2+2ab
$$

代入得到：

$$
\frac{a^2+b^2}{a-b}
=\frac{(a-b)^2+2ab}{a-b}
=\frac{(a-b)^2+16}{a-b}
=(a-b)+\frac{16}{a-b}
$$

如果后续题目还要求求最小值，就可以令 $x=a-b>0$，转化为：

$$
x+\frac{16}{x}
$$

再用 AM-GM：

$$
x+\frac{16}{x} \ge 2\sqrt{x\cdot \frac{16}{x}}=8
$$

等号在 $x=4$ 时成立。这个例子很小，但它已经体现出 MATH 的典型结构：先变形，再抽象，再套工具，最后给出规范答案。新手可以先把它理解成“把分式拆成更好处理的两项，再做不等式分析”。

再看一个真实工程例子。假设一个在线数学辅导产品要接入大模型自动解题。团队上线前会担心三件事：

1. 模型会不会只会写表面步骤，关键推导是错的。
2. 模型会不会在几何题或数论题上频繁跳步。
3. 模型最后的 LaTeX 输出是否稳定，能否进入自动批改链路。

这时 MATH 很适合做验收集。因为它不是单看“最后算对没有”，而是强迫系统在高难度、多步、竞赛风格题上稳定输出。若模型在 MATH 上只能靠模板猜测，它在真实辅导场景里通常也会在复杂题上失稳。

---

## 代码实现

一个最小可用的评测脚本只需要三部分：`extract`、`normalize`、`evaluate`。`extract` 负责从文本中取出 `\boxed{...}` 的内容；`normalize` 负责统一格式；`evaluate` 负责遍历题目并计算准确率。

下面给一个可运行的 Python 版本，逻辑简化但足够说明工程实现思路：

```python
import re

def extract_boxed(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None

    i = start + len(marker)
    depth = 1
    content = []

    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            content.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(content)
            content.append(ch)
        else:
            content.append(ch)
        i += 1

    return None

def normalize(expr: str | None) -> str | None:
    if expr is None:
        return None
    expr = expr.replace(r"\left", "")
    expr = expr.replace(r"\right", "")
    expr = expr.replace(r"\ ", "")
    expr = re.sub(r"\s+", "", expr)
    return expr

def is_correct(prediction: str, ground_truth: str) -> bool:
    pred_boxed = normalize(extract_boxed(prediction))
    gt_boxed = normalize(extract_boxed(ground_truth))
    return pred_boxed == gt_boxed

def evaluate(samples: list[dict]) -> float:
    correct = 0
    total = len(samples)
    for sample in samples:
        if is_correct(sample["prediction"], sample["ground_truth"]):
            correct += 1
    return correct / total if total else 0.0

# toy example
pred = r"先化简，最终得到 \boxed{\left((a-b)+\frac{16}{a-b}\right)}"
gt = r"标准答案是 \boxed{(a-b)+\frac{16}{a-b}}"
assert is_correct(pred, gt) is True

bad_pred = r"最终答案是 \boxed{\frac{a^2+b^2}{a-b}}"
assert is_correct(bad_pred, gt) is False

samples = [
    {"prediction": pred, "ground_truth": gt},
    {"prediction": bad_pred, "ground_truth": gt},
]
acc = evaluate(samples)
assert abs(acc - 0.5) < 1e-9
print("accuracy =", acc)
```

这段代码对应的输入输出关系可以概括如下：

| 字段 | 含义 | 示例 |
|---|---|---|
| `prediction` | 模型输出的完整解答 | `... \boxed{(a-b)+16/(a-b)}` |
| `ground_truth` | 标准解或标准答案文本 | `... \boxed{(a-b)+16/(a-b)}` |
| `extract_boxed()` | 从全文提取最终答案 | 返回盒内表达式 |
| `normalize()` | 清洗 LaTeX 形式差异 | 去空格、去 `\left/\right` |
| `evaluate()` | 统计总体准确率 | 返回 `0.0` 到 `1.0` |

如果把它写成伪代码，就是：

```python
for question in math_dataset:
    prediction = model.solve(question)
    boxed_pred = extract(prediction)
    boxed_gt = extract(question.ground_truth)
    is_correct = normalize(boxed_pred) == normalize(boxed_gt)
```

真实工程实现里，通常还会加三类约束：

1. 输出模板约束：要求模型必须按“推导过程 + `\boxed{final}`”输出。
2. 日志记录：保留原始答案、清洗后答案、判定结果，便于排查错因。
3. 分主题统计：分别统计代数、几何、数论等子集表现，而不是只看总准确率。

---

## 工程权衡与常见坑

MATH 很强，但工程上不能只看总分。因为总分高不代表每类题都稳，也不代表模型真的会“解释”。

最常见的坑是格式错判。比如模型输出：

```text
Answer: \boxed{\left(\frac{1}{2}\right)}
```

而标准答案是：

```text
\boxed{\frac{1}{2}}
```

如果没有规范化函数，系统可能判错。再比如模型明明推对了，但最后写成：

```text
Therefore the answer is 42.
```

没有 `\boxed{42}`，自动评测也拿不到答案。

第二类坑是“只会像，不会做”。有些模型能生成看起来很像数学解答的文本，但中间步骤并不成立。这在 MATH 里尤其危险，因为竞赛题往往需要跨多步推理，某一步错了，最后大概率就全错。换句话说，MATH 不适合靠表面模板刷分，泛化能力差的模型很容易暴露出来。

第三类坑是表达式等价但字符串不等价。比如 $\frac{2}{4}$ 和 $\frac{1}{2}$ 数学上等价，但简单字符串比较会判错。因此严格说，字符级评测是一个折中方案：它稳定、便宜、可复现，但不具备完整计算机代数系统的等价判断能力。

下面是常见风险与缓解手段：

| 风险 | 具体表现 | 缓解方法 |
|---|---|---|
| 格式错判 | 没有 `\boxed` 或括号风格不同 | 强制输出模板 + `normalize()` |
| 假推理 | 步骤很多但关键变形错误 | 抽样人工审查 + 分步验证 |
| 等价未识别 | `2/4` 与 `1/2` 被判不同 | 增加符号化简或 CAS 校验 |
| 主题偏科 | 代数高分，几何极差 | 按主题拆分报告 |
| 过拟合模板 | 见过类似题才会做 | 混合题源、做外部分布测试 |

上线前的验收清单通常至少包括：

1. 最终答案是否稳定放入 `\boxed{}`。
2. 规范化函数是否覆盖 `\left/\right`、空格和常见噪声。
3. 多步题是否保留清晰推理链条。
4. 几何、数论、计数等长尾主题是否单独评估。
5. 错题是否能回放原始输出，便于定位是推理错还是格式错。

真实工程里，MATH 更像“高压测试”而不是唯一测试。它能快速暴露多步推理和格式约束的问题，但不能替代面向真实用户题目的产品验收。

---

## 替代方案与适用边界

如果目标是评估基础算术和简单文字题，GSM8K 往往更合适。GSM8K 主要是小学到初中的 word problem，也就是文字应用题，白话说就是“看故事、列算式、出数字”。它的优点是贴近日常推理，缺点是难度和符号推导深度明显低于 MATH。

如果目标更偏数值推理或程序化回答，也可以考虑 BigBench 中的 numerical 类任务。它覆盖面更广，但题型一致性不如 MATH，标准答案格式也不总是围绕 LaTeX 和 `\boxed{}` 设计。

如果目标明确偏几何图形理解，一些专门的几何数据集会更直接，比如 Geometry3K 一类图文结合评测。但这类数据集通常关注图形感知和几何关系抽取，不等于 MATH 这种纯文本高阶竞赛数学。

可以用一个表格快速比较：

| 数据集 | 主要题型 | 输出要求 | 难度标签 | 适用场景 |
|---|---|---|---|---|
| MATH | 高中竞赛数学 | 常见为 LaTeX + `\boxed{}` | 有 AoPS 1-5 | 高阶数学推理验收 |
| GSM8K | 小学/初中应用题 | 普通文本答案即可 | 无统一竞赛分级 | 基础文字推理 |
| BigBench Numerical | 数值推理子任务 | 任务依赖具体定义 | 通常不统一 | 通用数值能力摸底 |
| Geometry3K 等 | 几何图文题 | 多为结构化或短答案 | 依任务而定 | 几何专项评测 |

因此，MATH 的适用边界很明确：

1. 适合测高阶、多步、竞赛风格数学。
2. 适合测 LaTeX 输出和答案格式稳定性。
3. 不适合替代基础算术评测。
4. 不适合单独代表“模型全部数学能力”。
5. 不适合在没有格式约束的自由生成场景中直接照搬结论。

如果新手只记一句话，可以这样理解：GSM8K 更像“会不会算生活题”，MATH 更像“会不会做竞赛题并且按规范写出来”。

---

## 参考资料

1. Hendrycks et al., *Measuring Mathematical Problem Solving With the MATH Dataset*, NeurIPS 2021 Datasets and Benchmarks.
用途：MATH 的原始定义、题目来源、12,500 题规模、主题与难度设定。

2. EmergentMind, *MATH Dataset* 主题综述页面，2025。
用途：补充对 7 个主题、AoPS 难度、结构化解答和工程实践的总结，便于从应用视角理解数据集。

3. Next / 相关数学推理评测介绍页面。
用途：说明基于 `\boxed{}` 抽取、LaTeX 规范化和准确率统计的常见评测流程。

4. MCPBR Benchmark 的 MATH 页面。
用途：补充工程侧常见坑，包括格式错判、规范化策略和高阶数学推理的评估关注点。
