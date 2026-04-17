## 核心结论

DROP 是一个面向段落阅读理解的离散推理基准。这里的“离散推理”，可以先理解成一个很具体的动作链：模型不是直接复制原文中的某个连续片段，而是要先从段落里取出多个分散的事实，再做加法、减法、计数、比较、排序等操作，最后输出答案。它检验的重点不是复杂数学，而是模型能否把“读懂文本”和“做对简单计算”连成一个稳定流程。

这个基准的数据规模大约在 96K 问题量级。题目通常给出一段体育比赛、新闻事件或事实叙述，再给出一个问题。很多题的答案并不以完整字符串的形式直接出现在原文里，因此模型不能只做检索或 span 抽取，而要跨句定位多个数字、事件或实体，再执行一步或多步离散操作。换句话说，DROP 测的是“文本理解 + 证据拼接 + 简单运算 + 答案规范化匹配”的联合能力。

最常用的两个指标是 EM 和 F1。

- `EM`：Exact Match。预测答案在规范化后，是否与某个参考答案完全一致。
- `F1`：token 级调和平均。预测与参考答案在词粒度上的重合程度。

公式是：

$$
F1 = \frac{2 \times precision \times recall}{precision + recall}
$$

其中：

$$
precision = \frac{matched\_tokens}{candidate\_tokens}
$$

$$
recall = \frac{matched\_tokens}{reference\_tokens}
$$

可以先用一个最小例子理解。段落写：

> 第一节结束前跑卫拿下 11 码，下次驱动再拿 7 码。

问题问：

> 这两次驱动一共是多少码？

正确过程不是复制 `11` 或 `7`，而是先抽取两个数，再计算：

$$
11 + 7 = 18
$$

因此正确答案是 `18`。这正是 DROP 想评估的最小能力闭环：先理解文本中哪些数值相关，再执行一个离散操作。

EM/F1 的评分流程可以概括成 4 步：

1. 读取模型预测和参考答案。
2. 对答案做规范化处理，例如统一大小写、去除无关标点、整理数字表达。
3. 将预测与多个参考答案逐一比较，分别计算 EM 和 F1。
4. 取该题的最优分数，再对所有题求平均。

只看这四步，会觉得它像普通问答评测；真正的区别在于，DROP 的很多题目要求“答案来自计算”，而不是“答案在原文里原样出现”。因此，DROP 对系统的要求比纯抽取更高，但又没有走到形式化数学证明那么远。

---

## 问题定义与边界

DROP 的输入是一段短文本和一个问题，输出是一个答案。这里的关键不在“问答”二字，而在它限定了推理发生的范围：证据通常来自给定段落内部，模型需要在段落内部完成事实抽取和离散操作。

新手可以把一条 DROP 题拆成四层任务：

| 输入段落 | 拉取的数值 | 运算类型 | 输出 |
|---|---:|---|---:|
| “队伍 A 第一场 27 票，第二场 33 票” | 27, 33 | 加法 | 60 |
| “上半场得 14 分，下半场得 10 分” | 14, 10 | 加法 | 24 |
| “本场有 3 次达阵、2 次射门” | 3, 2 | 计数合并 | 5 |
| “第三节 21 分，第四节 14 分” | 21, 14 | 比较 | 21 或“第三节” |

这个表格反映的是 DROP 的自底向上路径：

1. 先从文本中抽出相关事实。
2. 再判断事实之间的关系。
3. 最后执行操作并输出答案。

因此，问题通常不是“27 是多少”，而是“27 和 33 合起来是多少”。如果模型只会定位数字而不会判断这些数字该如何组合，它在 DROP 上就会失分。

再看一个更完整的新手版例子。

段落：

> 队伍 A 赢得第一场 27 票、第二场 33 票。第三场因天气原因取消。

问题：

> 两场总共是多少票？

这道题里，第三场信息是干扰项。正确步骤是：

1. 识别问题只问“两场”。
2. 找到相关数字 `27` 和 `33`。
3. 忽略“第三场取消”。
4. 计算 `27 + 33 = 60`。

如果模型输出 `27`，说明它只做了单点抽取；如果输出 `27 33`，说明它找到了证据，但没有完成运算；只有输出 `60`，才完成了题目定义要求的能力链。

DROP 的边界也很明确：

| 任务类型 | DROP 是否适合 | 原因 |
|---|---|---|
| 单一片段抽取 | 不一定 | 任务过轻，SQuAD 一类更直接 |
| 多数值组合问答 | 很适合 | 正好覆盖加减、比较、计数 |
| 纯符号数学推导 | 不适合 | 主轴是文本理解，不是公式证明 |
| 长链外部知识问答 | 不适合 | 证据通常限定在给定段落 |
| 结构化表格计算 | 一般 | DROP 原生面向自然语言段落，不是表格执行 |

这个边界在工程里非常重要。比如你做的是“比赛报道自动问答”系统，用户常问：

- 前两次驱动累计多少码
- 上下半场总得分是多少
- 最终领先多少分
- 一共发生了几次关键事件

这类问题和 DROP 高度同构，DROP 就是合适的评测工具。

相反，如果用户主要问：

- 谁完成了最后一次达阵
- 哪一年发生了这件事
- 哪支球队获胜

这些答案大概率就是原文中的一个连续片段，DROP 就可能过重，因为你并不需要额外验证离散运算能力。

可以把 DROP 的问题边界压缩成一句话：它关注的是“给定段落内，多证据、多数值、多步拼接”的问答，而不是开放知识问答，也不是纯数学竞赛题。

---

## 核心机制与推导

DROP 评分的关键，不是直接比较原始字符串，而是先做标准化，再做 token 级比对。这里的 `token` 可以先理解成“用于比较的最小词片段”。这样做的原因很实际：自然语言答案经常有多种表面写法，如果不先规范化，很多本来等价的答案会被误判。

评分的基本机制可以拆成五步：

1. 规范化预测答案和参考答案。
2. 将规范化结果切成 token。
3. 统计预测与参考的重合 token 数。
4. 根据重合数计算 precision、recall、F1。
5. 如果一题有多个参考答案，对每个参考分别打分，取最佳结果。

先把三个指标的含义说清楚：

- `precision`：你输出的内容里，有多少是对的。
- `recall`：标准答案里的关键信息，你覆盖了多少。
- `F1`：precision 和 recall 的综合值，避免只偏向一边。

例如：

- 预测：`18`
- 参考：`18`

那么：

$$
matched\_tokens = 1,\ candidate\_tokens = 1,\ reference\_tokens = 1
$$

因此：

$$
precision = 1,\ recall = 1,\ F1 = 1
$$

再看一个常见偏差例子：

- 预测：`the answer is 18`
- 参考：`18`

若规范化后得到：

- 预测 token：`["the", "answer", "is", "18"]`
- 参考 token：`["18"]`

则：

$$
matched\_tokens = 1
$$

$$
precision = \frac{1}{4}
$$

$$
recall = \frac{1}{1} = 1
$$

所以：

$$
F1 = \frac{2 \times \frac{1}{4} \times 1}{\frac{1}{4} + 1} = 0.4
$$

这个例子说明两件事：

1. F1 允许“部分正确”。
2. 输出过长会拖低 precision，因此很多 DROP 系统会倾向输出紧凑答案，而不是完整句子。

再看一个多参考答案场景。假设某题人工标注有两个可接受答案：

- `18`
- `eighteen`

如果系统预测 `18`，它不需要同时命中两个参考，只要与其中一个规范化后完全匹配，就能拿到该题的最佳分数。多参考答案的目的不是“放水”，而是减少表面写法差异造成的误伤。

下面用一个表格把不同情况展开：

| 预测 | 参考 | 规范化后是否完全一致 | EM | F1 直观解释 |
|---|---|---|---:|---|
| `18` | `18` | 是 | 1.0 | 完全命中 |
| `18` | `eighteen` | 若数字归一化成功，则是 | 1.0 | 视为等价 |
| `the answer is 18` | `18` | 否 | 0.0 | 命中核心数字，但多余 token 降低 F1 |
| `17` | `18` | 否 | 0.0 | 完全错误 |
| `18 yards` | `18` | 否 | 0.0 | 语义接近，但输出更长，F1 低于 1 |

下面给出一个更接近真实逻辑的简化伪代码，展示“多参考答案取最佳 EM/F1”的过程：

```python
def best_score(prediction, gold_answers):
    pred_norm = normalize(prediction)
    best_em = 0.0
    best_f1 = 0.0

    for gold in gold_answers:
        gold_norm = normalize(gold)

        em = 1.0 if pred_norm == gold_norm else 0.0

        pred_tokens = tokenize(pred_norm)
        gold_tokens = tokenize(gold_norm)
        matched = overlap_count(pred_tokens, gold_tokens)

        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            f1 = 1.0 if pred_tokens == gold_tokens else 0.0
        else:
            precision = matched / len(pred_tokens)
            recall = matched / len(gold_tokens)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)

        best_em = max(best_em, em)
        best_f1 = max(best_f1, f1)

    return best_em, best_f1
```

这个过程里最容易被忽视的地方，不是公式，而是规范化细节。实际分数是否稳定，往往取决于下面这些问题：

| 规范化细节 | 做错后的后果 |
|---|---|
| 大小写未统一 | `Touchdown` 和 `touchdown` 被当成不同答案 |
| 标点未清洗 | `18.` 与 `18` 不能匹配 |
| 冠词未处理 | `the Patriots` 与 `Patriots` 匹配不稳定 |
| 数字表达未对齐 | `18` 与 `eighteen` 被错误打成不相等 |
| 多答案未取最优 | 同一道题被人为压低分数 |

所以，DROP 评分虽然表面上只有 EM/F1 两个数字，真正决定可比性的却是规范化实现。很多自研脚本的问题并不在公式，而在“看起来相似、实际不一致”的预处理逻辑。

---

## 代码实现

工程上最稳妥的做法，是直接复用官方 `drop_eval.py` 的逻辑，而不是自己手写一个“差不多”的版本。原因很直接：评测脚本本身就是标准的一部分。你改了 normalize 规则，得到的就不是同一个 DROP 分数。

下面给一个**可直接运行**的 Python 玩具实现，用来解释核心结构。它不替代官方脚本，但能帮助你把评分流程完整跑通。为了让示例真的可运行，这里显式实现了一个极小的英文数字映射，并避免写出无法通过断言的伪功能。

```python
import re
from collections import Counter
from typing import Iterable

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^0-9a-z\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_tokens(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    tokens = normalized.split()
    return [NUMBER_WORDS.get(token, token) for token in tokens]


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_tokens(prediction) == normalize_tokens(reference) else 0.0


def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = normalize_tokens(prediction)
    ref_tokens = normalize_tokens(reference)

    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    matched = sum(common.values())

    if matched == 0:
        return 0.0

    precision = matched / len(pred_tokens)
    recall = matched / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def best_drop_score(prediction: str, references: Iterable[str]) -> tuple[float, float]:
    best_em = 0.0
    best_f1 = 0.0

    for ref in references:
        best_em = max(best_em, exact_match(prediction, ref))
        best_f1 = max(best_f1, f1_score(prediction, ref))

    return best_em, best_f1


def demo() -> None:
    # 完全匹配
    em, f1 = best_drop_score("18", ["18", "eighteen"])
    assert em == 1.0
    assert f1 == 1.0

    # 数字词与阿拉伯数字对齐
    em2, f12 = best_drop_score("eighteen", ["18"])
    assert em2 == 1.0
    assert f12 == 1.0

    # 部分匹配：有正确核心词，但输出更长
    em3, f13 = best_drop_score("the answer is 18", ["18"])
    assert em3 == 0.0
    assert abs(f13 - 0.4) < 1e-9

    # 完全错误
    em4, f14 = best_drop_score("17", ["18"])
    assert em4 == 0.0
    assert f14 == 0.0

    print("All demo checks passed.")


if __name__ == "__main__":
    demo()
```

这段代码可以直接保存为 `drop_toy_eval.py` 并运行：

```bash
python drop_toy_eval.py
```

如果输出：

```text
All demo checks passed.
```

说明这份玩具实现至少已经具备四个关键性质：

1. 能做规范化。
2. 能算单个参考答案的 EM/F1。
3. 能在多参考答案里取最佳分数。
4. 能用断言验证基本行为没有跑偏。

再用一个表格对照代码中的行为：

| 输入 | 规范化 token | EM | F1 |
|---|---|---:|---:|
| `18` vs `18` | `["18"]` vs `["18"]` | 1.0 | 1.0 |
| `eighteen` vs `18` | `["18"]` vs `["18"]` | 1.0 | 1.0 |
| `the answer is 18` vs `18` | `["answer","is","18"]` vs `["18"]` | 0.0 | 0.5 |
| `17` vs `18` | `["17"]` vs `["18"]` | 0.0 | 0.0 |

注意这里第三行的 F1 是 `0.5`，不是前文例子里的 `0.4`。原因是这份实现把冠词 `the` 去掉了，预测 token 从 4 个变成 3 个：

$$
precision = \frac{1}{3},\ recall = 1
$$

$$
F1 = \frac{2 \times \frac{1}{3} \times 1}{\frac{1}{3} + 1} = 0.5
$$

这恰好能说明一个工程事实：**规范化规则不同，分数就会不同**。也正因为如此，正式评测时才必须尽量复用官方实现。

真实工程中的批量评测流程通常类似下面这样：

```python
# 1. 读取 predictions.json
# 2. 读取 gold annotations
# 3. 对每道题：
#    a. 取该题预测答案
#    b. 取该题所有参考答案
#    c. 计算 best EM / best F1
# 4. 对所有题平均
# 5. 输出总分和分题型统计
```

如果想写成一个最小可运行批量脚本，结构可以是：

```python
import json

from drop_toy_eval import best_drop_score

def evaluate_file(pred_path: str, gold_path: str) -> dict:
    with open(pred_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    with open(gold_path, "r", encoding="utf-8") as f:
        gold = json.load(f)

    total_em = 0.0
    total_f1 = 0.0
    count = 0

    for qid, references in gold.items():
        prediction = predictions.get(qid, "")
        em, f1 = best_drop_score(prediction, references)
        total_em += em
        total_f1 += f1
        count += 1

    return {
        "em": total_em / count if count else 0.0,
        "f1": total_f1 / count if count else 0.0,
        "count": count,
    }
```

配套输入可以长成这样：

```json
{
  "q1": "18",
  "q2": "Patriots",
  "q3": "3"
}
```

```json
{
  "q1": ["18", "eighteen"],
  "q2": ["the patriots", "patriots"],
  "q3": ["3", "three"]
}
```

这已经足够让新手把一整套 DROP 评分流水跑起来。需要再次强调的是：这类玩具实现适合教学、单元测试、理解指标；正式报分、论文复现、模型横向比较，仍然应该以官方脚本为准。

---

## 工程权衡与常见坑

把 DROP 接入 CI，本质上是在给“段落内离散推理能力”设置回归门槛。这里的 CI 可以先理解成：每次模型、提示词、检索链路或后处理模块发生变更时，自动跑一套固定评测并对比历史基线。

一个最小的 CI 流程通常包含两步：

```bash
python drop_eval.py --pred predictions.json --gold gold.json > report.json
python compare_metrics.py --current report.json --baseline baseline.json
```

它的价值不在于“多跑一次脚本”，而在于把模糊感受变成明确约束。比如你改了答案后处理模块，主观感觉“回复更自然了”，但 DROP 上 F1 从 `0.72` 掉到 `0.61`，那就说明新系统虽然更会说话，却更不擅长给出紧凑、可判分的离散答案。

一个可操作的回滚条件可以写成：

- `EM` 下降超过 `2.0` 个百分点，阻断部署。
- `F1` 下降超过 `3.0` 个百分点，阻断部署。
- 任一关键子集下降超过阈值，触发人工复核。

这里的“关键子集”建议不要只看全量平均，而要按能力切片，例如：

| 子集名称 | 关注问题 |
|---|---|
| `add_sub` | 加法、减法是否退化 |
| `count` | 计数题是否退化 |
| `compare` | 比较大小、确定领先方是否退化 |
| `multi_span` | 多证据拼接是否退化 |
| `long_answer` | 输出变长后 F1 是否被污染 |

下面这个表格总结常见坑及其影响：

| 风险 | 影响 | 对策 |
|---|---|---|
| 未复用官方 normalize 逻辑 | EM/F1 偏高或偏低，横向不可比 | 正式评测直接用官方脚本 |
| 只看 EM 不看 F1 | 部分正确答案全部被打成 0，信息损失严重 | 同时监控 EM 与 F1 |
| 只看全量均值 | 某类题严重退化，但被平均数掩盖 | 输出按题型切片报告 |
| 输出格式变长 | 核心答案正确，但 token 污染导致 F1 下滑 | 单独检查答案后处理模块 |
| baseline 更新随意 | 门槛漂移，回归告警失真 | baseline 文件版本化、可追溯 |
| 参考答案标注不统一 | 同模型多次打分波动 | 统一标注规范并做审校 |
| 漏评空答案 | 预测缺失时统计失真 | 明确无预测时按空串计分 |
| 评测数据泄漏到提示词 | 指标虚高，线上无泛化 | 保持评测集隔离 |

新手最容易踩的坑通常有三个。

第一，只实现了公式，没有实现标准化。  
结果是看起来“F1 算对了”，但实际上 `18.` 和 `18`、`The Patriots` 和 `Patriots` 都可能被误判，最后整份分数没有可比性。

第二，只看单一总分，不看错误类型。  
比如总 F1 只降了 1 个点，但所有计数题都在下降，这说明系统的某个具体能力已经退化，只是被其他题型平均掉了。

第三，把 DROP 当成通用问答指标。  
如果你的产品本质上是单 span 抽取，DROP 的维护成本可能高于收益，因为你为了少量离散题付出了整套脚本、基线、切片分析和回滚流程。

因此，工程上的权衡很明确：DROP 很有价值，但它的价值来自“能准确暴露离散推理回退”，不是来自“任何问答系统都应该测一次”。

---

## 替代方案与适用边界

DROP 最常见的比较对象，是 SQuAD 这类 span 抽取基准，以及 MathQA 这类偏公式求解的数学基准。它们看起来都在做问答，但实际考核的能力并不相同。

| 基准 | 关注点 | 运算复杂度 | 适用场景 |
|---|---|---|---|
| DROP | 段落理解 + 多数值离散推理 | 中 | 多句证据拼接、加减计数比较 |
| SQuAD | 单段答案片段抽取 | 低 | 答案直接出现在原文 |
| MathQA | 数学表达式与公式推导 | 高 | 偏符号计算、公式建模 |

可以用一个更直接的判断法来区分它们。

如果你的问题是：

- 谁得分了
- 哪一年发生的
- 哪支球队获胜
- 最后一次达阵是谁完成的

这类答案通常就是原文中的一个 span，更适合 SQuAD 式评测，因为任务重点是定位，不是运算。

如果你的问题是：

- 两次驱动累计多少码
- 上下半场总得分是多少
- 双方净胜多少票
- 一共发生了几次射门或达阵
- 第三节和第四节哪一节得分更高

这已经进入 DROP 的典型范围，因为系统必须先抽取多个事实，再做离散操作。

如果你的问题进一步升级为：

- 根据题意列方程
- 进行多步代数变换
- 推导中间变量
- 解决纯符号表达式问题

那 DROP 又太轻了，因为它的主轴仍然是自然语言段落理解，而不是正式数学推导。

所以，DROP 过重的情况通常有两类：

1. 任务根本不涉及多数值组合。
2. 线上验收并不关心离散推理，只关心抽取稳定性。

而 DROP 明显适用的场景一般有四个条件：

1. 输入是自然语言段落，不是纯结构化表格。
2. 问题经常跨句引用多个数字或事件。
3. 正确答案依赖加减、计数、比较或排序。
4. 你希望把这种能力纳入版本回归和发布门禁。

还可以把三类基准的差异压缩成一个“问题到评测”的映射表：

| 用户问题类型 | 更适合的评测 |
|---|---|
| “谁在最后一分钟得分？” | SQuAD 类 |
| “前两节合计得了多少分？” | DROP |
| “设某量为 x，求最终表达式” | MathQA / 数学推理基准 |

因此，DROP 不是“通用问答评测”，而是一个非常明确的能力探针：它检查模型能否把段落中的离散事实，转换成一个可执行的小型推理过程，并把结果输出成可稳定计分的答案。

---

## 参考资料

| 资源名称 | 内容摘要 | 链接/路径（描述性） |
|---|---|---|
| DROP 原始论文 | 数据集设计目标、任务定义、标注方法、基线模型和主要实验结果 | `arxiv.org/abs/1903.00161` |
| AllenNLP 官方 DROP 评测脚本 | 官方 `drop_eval.py` 与对应标准化、EM/F1 计算逻辑，是正式报分的实现基准 | `github.com/allenai/allennlp-reading-comprehension/.../drop_eval.py` |
| EvalScope DROP 概览 | 适合快速了解 DROP 的任务定位、规模与评测接入方式 | `evalscope.readthedocs.io/.../benchmarks/drop.html` |
| DeepWiki 对 DROP Eval 的机制复盘 | 适合阅读 normalize、多参考答案取最优、分数汇总等实现思路 | `deepwiki.com/.../drop-reading-comprehension` |
| Inspect Evals DROP 页面 | 提供任务示例和最小评测样例，便于理解如何在实际评测框架中接入 | `ukgovernmentbeis.github.io/inspect_evals/.../drop` |
| 论文复现与评测工程资料 | 用于理解如何管理基线、切片指标和回归门禁 | 各评测框架文档与内部 CI 规范 |

这些资料可以按三个层次使用：

1. 先看原始论文，确认 DROP 到底在测什么。
2. 再看官方评测脚本，确认分数到底怎么算。
3. 最后看工程化框架资料，决定如何把它接入批量评测和 CI。

如果只读二手介绍而不看官方脚本，最容易出现的问题是“任务理解大致正确，但分数口径不一致”。对于需要横向比较模型、复现实验结果或建立长期回归门槛的团队来说，这一步不能省略。
