## 核心结论

Few-shot 评测和 Zero-shot 评测的差别，不在于“模型是不是同一个”，而在于“模型答题前看没看过任务示范”。

Zero-shot 是零样本评测，白话说就是只给任务说明，不给例题，直接看模型能不能靠预训练时学到的通用知识完成任务。Few-shot 是少样本评测，白话说就是在正式答题前先给 $k$ 个带标准答案的示例，让模型在当前上下文里临时学会题型、输出格式和推理节奏。

这两种评测都有效，但回答的是不同问题：

| 维度 | Zero-shot | Few-shot |
|---|---|---|
| 是否给示例 | 否 | 是，给 $k$ 个示例 |
| 主要考察 | 纯泛化能力 | 借助示范后的任务适应能力 |
| 分数来源 | 预训练知识 + 指令理解 | 预训练知识 + 指令理解 + 上下文示范 |
| 适合场景 | 比较基础能力、做统一基线 | 比较“上线时加模板后”的真实效果 |
| 风险 | 低估可用性 | 高估泛化能力 |

工程上最重要的指标不是只看某一个分数，而是看增益：

$$
\Delta(k) = Score_{few}(k) - Score_{zero}
$$

其中 $k$ 是 few-shot 示例数，$\Delta(k)$ 表示“加了示例后到底提升了多少”。如果一个模型 zero-shot 很强但 $\Delta$ 很小，说明它本身泛化已经不错；如果 zero-shot 一般但 few-shot 提升很大，说明它很依赖 prompt 中的示范。

最典型的玩具理解方式是：老师不做示范，直接让你做题，这是 zero-shot；老师先做 3 道同类型例题，再让你做第 4 题，这是 few-shot。两种考试都合理，但不能混着比成绩。

真实工程里，这个区别会直接影响模型选择。比如有些团队会同时跑 `MMLU 5-shot` 和 `GSM8K 8-shot`：前者更像“知识问答加少量示范”，后者更像“多步推理任务在示范后的表现”。如果你拿一个模型的 zero-shot 成绩去和另一个模型的 8-shot 成绩比较，结论通常没有可比性。

---

## 问题定义与边界

先把边界说清楚。

Zero-shot 评测的定义是：测试样本前不放任何同任务标注示例，只给任务说明。例如“把这句话翻译成英文”就是一个 zero-shot 提示。Few-shot 评测的定义是：在测试样本前放入 $k$ 个输入-输出示例，再让模型回答新样本。例如“下面给你 3 个中英翻译示例，请按同样格式翻译第 4 句”。

这个差别看起来只是 prompt 长短不同，实际上改变了评测对象。

Zero-shot 主要测的是“模型是否已经在参数里学会任务”。参数，白话说就是模型在预训练或指令微调阶段吸收进内部权重的知识和模式。Few-shot 主要测的是“模型能否利用上下文中的现场示范快速适应任务格式”。上下文，白话说就是当前这次请求里模型能看到的全部文本。

一个最小例子：

- Zero-shot：`请判断这条评论是正面还是负面：'这个电池很耐用'`
- Few-shot：先给 3 条“评论 -> 情感标签”的例子，再给 `'这个电池很耐用'`

两者都在做情感分类，但 few-shot 额外告诉了模型：
1. 标签集合是什么
2. 输出格式是什么
3. 边界样本怎么判
4. 任务语气应该偏简洁还是偏解释型

因此，few-shot 得分更高并不奇怪，它不是“作弊”，但它测的不是纯泛化。

这里有两个边界必须守住。

第一，$k$ 不能随便增大。理论上，随着示例数增加，模型获得的信息更多，表现可能继续上升；但当 $k$ 很大时，评测已经不再接近“无额外帮助的模型能力”，而更像“长 prompt 条件下的任务适配能力”。同时 token 成本也会升高。

第二，示例必须和测试任务同分布，至少要格式一致。分布，白话说就是数据的来源、风格和难度结构。如果你在数学推理任务里给的是百科问答示例，这种 few-shot 没有评测意义；如果示例和测试题几乎重复，又会污染评测。

因此 few-shot 评测不是“多放点例子看最高分”，而是“在固定、公开、可复现的示例策略下，测模型被轻量示范后的能力”。

用公式表达边界更清楚：

$$
\Delta(k) = Score_{few}(k) - Score_{zero}, \quad k \ge 0
$$

通常我们关心的不只是某个单点的 $\Delta(k)$，还关心它随 $k$ 的变化。如果 $k$ 从 0 增加到 5 时提升很大，但从 5 到 10 提升很小，说明 few-shot 的边际价值在下降。边际价值，白话说就是“每多给一个例子，带来的额外收益”。

---

## 核心机制与推导

Few-shot 为什么有效，核心不是模型“现场改权重”，而是模型利用上下文做条件生成。条件生成，白话说就是：模型根据你前面给出的示范，推断接下来该按什么模式继续写。

这个机制至少包含三层。

第一层是格式对齐。  
很多任务难点不在知识本身，而在“题目到底要我怎么答”。例如多选题是输出 `A/B/C/D`，摘要题是输出一段短文，分类题是输出一个标签。few-shot 先把输出协议固定下来，所以模型少走弯路。

第二层是决策边界提示。  
示例会告诉模型什么样的输入对应什么样的标签、答案或推理路径。对边界模糊的任务，这很重要。例如“这句评论算中性还是轻微正面”，zero-shot 时模型可能犹豫，few-shot 看到相似例子后会更稳定。

第三层是推理轨迹提示。  
在数学、多步逻辑、代码解释等任务里，few-shot 示例不只提供答案，还隐含展示了“先算什么，再算什么”。这就是为什么 GSM8K 这类数据集经常在 few-shot 下提升明显。推理轨迹，白话说就是中间思考步骤的排列方式。

可以把 score 写成一个函数：

$$
Score_{few}(k) = f(\text{model}, \text{task}, \text{template}, \text{examples}_{1:k}, \text{order})
$$

这里除了模型和任务本身，模板、示例内容、示例顺序都会影响结果。于是：

$$
\Delta(k) = f(\cdots, \text{examples}_{1:k}, \text{order}) - f(\cdots, \varnothing)
$$

这说明 $\Delta$ 不是模型的绝对属性，而是“模型在某种 prompt 设计下的增益”。

看一个玩具例子。假设是判断奇偶数：

- zero-shot 指令：`判断数字是奇数还是偶数，只输出 odd 或 even`
- few-shot 示例：
  - `2 -> even`
  - `5 -> odd`
  - `8 -> even`

对人来说这题太简单，但它说明了 few-shot 的本质：先把标签空间、映射规则、输出格式都示范一遍。

再看真实工程例子。  
在模型选型流水线中，团队常常不会只跑一个 benchmark，而是同时跑多个任务族。例如：

| Benchmark | 常见设置 | 主要能力 |
|---|---|---|
| MMLU | 5-shot | 学科知识、多选题理解 |
| GSM8K | 8-shot | 数学文字题、多步推理 |
| 某内部分类集 | 0-shot / 3-shot | 业务标签判定 |
| 某客服摘要集 | 0-shot / 5-shot | 结构化摘要与格式遵循 |

假设某模型在采样实验中的结果如下：

| 任务 | k=0 | k=5 | k=10 | 观察 |
|---|---:|---:|---:|---|
| MMLU | 72.4 | 85.6 | 86.1 | 5-shot 后已接近收敛 |
| GSM8K | 46.0 | 57.8 | 60.2 | few-shot 对多步推理更敏感 |
| 内部分类集 | 81.5 | 84.0 | 84.1 | 提升有限，zero-shot 已足够 |
| 内部摘要集 | 68.2 | 76.9 | 77.0 | 示例主要改善格式稳定性 |

从表里能看出两件事。

第一，不同任务的 $\Delta(k)$ 不一样。  
这意味着不能把“few-shot 有用”当成统一结论，而要按任务族拆开看。

第二，$\Delta(k)$ 往往有收敛趋势。  
收敛，白话说就是再继续加示例，分数也不再明显上升。工程上这很关键，因为 token 预算不是无限的。你通常想找的是“成本还能接受、分数已基本稳定”的那个 $k$，而不是最大 $k$。

如果把评测过程写成伪代码，逻辑就是：

1. 固定数据集、模板和采样种子
2. 对每个 $k \in \{0, 5, 10\}$ 抽取示例
3. 构造 prompt
4. 跑完整测试集
5. 记录 score 与 $\Delta(k)$
6. 多次重复，估计平均值和方差

方差，白话说就是“同样方法多跑几次，结果会波动多少”。few-shot 如果对示例顺序非常敏感，就会表现为方差较大。

---

## 代码实现

下面给一个可运行的 Python 玩具脚本。它不依赖真实大模型 API，而是用一个简单规则模型模拟“few-shot 通过示范学格式”的过程，重点是展示评测方法，而不是追求模型真实性。

```python
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Example:
    text: str
    label: str

def fake_zero_shot_predict(text: str) -> str:
    """
    玩具模型：zero-shot 只靠非常粗糙的关键词规则。
    """
    positive_words = {"好", "喜欢", "满意", "耐用", "值得"}
    negative_words = {"差", "失望", "坏", "退货", "卡顿"}

    pos = sum(word in text for word in positive_words)
    neg = sum(word in text for word in negative_words)

    if pos >= neg:
        return "positive"
    return "negative"

def fake_few_shot_predict(text: str, shots: List[Example]) -> str:
    """
    玩具模型：few-shot 先从示例中统计标签词倾向，再做预测。
    """
    label_bias: Dict[str, int] = {"positive": 0, "negative": 0}

    for ex in shots:
        if "耐用" in ex.text and ex.label == "positive":
            label_bias["positive"] += 2
        if "卡顿" in ex.text and ex.label == "negative":
            label_bias["negative"] += 2
        label_bias[ex.label] += 1

    base = fake_zero_shot_predict(text)

    if "耐用" in text and label_bias["positive"] >= label_bias["negative"]:
        return "positive"
    if "卡顿" in text and label_bias["negative"] >= label_bias["positive"]:
        return "negative"
    return base

def accuracy(pairs: List[Tuple[str, str]], predictor) -> float:
    correct = 0
    for text, gold in pairs:
        pred = predictor(text)
        if pred == gold:
            correct += 1
    return correct / len(pairs)

def main():
    test_set = [
        ("这个电池很耐用", "positive"),
        ("系统更新后非常卡顿", "negative"),
        ("整体还行，但物流慢", "positive"),
        ("用了两天就坏了", "negative"),
    ]

    shots = [
        Example("这款产品很耐用", "positive"),
        Example("手机打开应用很卡顿", "negative"),
        Example("我对这次购买很满意", "positive"),
    ]

    zero_score = accuracy(test_set, fake_zero_shot_predict)
    few_score = accuracy(test_set, lambda text: fake_few_shot_predict(text, shots))
    delta = few_score - zero_score

    print("zero-shot:", zero_score)
    print("few-shot :", few_score)
    print("delta    :", delta)

    assert 0.0 <= zero_score <= 1.0
    assert 0.0 <= few_score <= 1.0
    assert abs(delta - (few_score - zero_score)) < 1e-12
    assert few_score >= zero_score

if __name__ == "__main__":
    main()
```

这个脚本对应的方法论是：

| 步骤 | 目的 |
|---|---|
| 准备测试集 | 固定评测对象，避免边跑边改题 |
| 构造 zero-shot predictor | 作为无示范基线 |
| 构造 few-shot predictor | 模拟加入示例后的效果 |
| 计算 `zero_score` 与 `few_score` | 得到两个可比较分数 |
| 计算 `delta` | 衡量示例带来的净增益 |

如果你接真实 API，流程也一样，只是把 `predictor` 换成“拼 prompt 并调用模型”。一个简化版伪代码如下：

```python
def build_prompt(query, shots):
    prompt = "你是一个分类器，只输出 positive 或 negative。\n\n"
    for x, y in shots:
        prompt += f"输入: {x}\n输出: {y}\n\n"
    prompt += f"输入: {query}\n输出:"
    return prompt

def eval_dataset(dataset, shots, call_model):
    correct = 0
    for query, gold in dataset:
        prompt = build_prompt(query, shots)
        pred = call_model(prompt).strip()
        if pred == gold:
            correct += 1
    return correct / len(dataset)

zero = eval_dataset(test_data, shots=[], call_model=api_call)
few5 = eval_dataset(test_data, shots=sample_shots(5), call_model=api_call)
delta = few5 - zero
```

真实工程里，建议再加两层控制：

1. 固定随机种子，保证示例采样可复现。
2. 对每个 $k$ 重复多次，统计均值和标准差，而不是只跑一次。

例如：

$$
\bar{\Delta}(k) = \frac{1}{n} \sum_{i=1}^{n} \Delta_i(k)
$$

这里 $n$ 是重复次数。这样你看到的不是“这次正好抽中一组好例子”的偶然结果，而是更稳定的估计。

---

## 工程权衡与常见坑

few-shot 评测最容易出问题的地方，不是代码，而是实验设计。

先看最常见的坑：

| 常见坑 | 具体表现 | 规避策略 |
|---|---|---|
| 示例顺序不固定 | 同样 5-shot，换个顺序分数波动明显 | 固定顺序或多次打乱后取均值 |
| 示例分布失衡 | 5 个示例全是简单题或同一标签 | 保证类别和难度覆盖 |
| 模板不统一 | zero-shot 与 few-shot 指令风格差别太大 | 只改变“是否放示例”，其他保持一致 |
| 测试污染 | 示例与测试题过于相似甚至重复 | 去重并检查近重复样本 |
| 只报最高分 | 多次实验只挑最好的一次 | 报均值、方差和采样规则 |
| 忽略成本 | few-shot 分数高，但 prompt 太长太贵 | 同时报 token 成本和延迟 |

一个典型玩具坑是“翻译任务只改了示例，不只改了示例数量”。  
比如 zero-shot 提示是“翻译成英文”，few-shot 提示却写成“你是一位专业译者，请自然、流畅、地道地翻译，并参考下面案例”。这时分数变化不再只是 shot 数导致，而是模板整体变化导致。方法论上这是变量没控住。控变量，白话说就是一次只改一个因素，其他都保持不变。

真实工程里最常见的坑是顺序敏感和成本失控。

以 GSM8K 为例，8-shot 常见于评测和实践，因为多步算术题往往受示范影响较大。但如果这 8 个示例全是一步算术，或者全按同一种冗长 Chain-of-Thought 模板书写，那么模型可能只学到一种狭窄套路。一旦真实测试题分布变了，收益就会下降。

另一个工程问题是 token 预算。token，白话说就是模型处理文本时的计费和上下文单位。few-shot 每增加一个示例，几乎一定会增加输入 token，进而增加：
1. 调用成本
2. 响应延迟
3. 上下文占用
4. 长输入下的截断风险

所以工程上不应该只问“8-shot 是否比 0-shot 高”，而应该问：
1. $\Delta(8)$ 是否显著
2. 这部分提升是否值得额外成本
3. 这个提升在不同采样下是否稳定

如果你要写一个稳妥的 prompt template，结构通常至少包括：

- 任务说明
- 标签或输出格式约束
- few-shot 示例区
- 待预测样本
- 明确的停止条件或输出要求

例如分类任务建议固定为：

```text
任务说明
标签定义
示例1
示例2
...
待测输入
只输出标签
```

这样做的目的不是“让 prompt 看起来整齐”，而是减少模板噪声，把实验差异尽量压缩到 shot 数和示例内容本身。

---

## 替代方案与适用边界

few-shot 不是唯一办法，也不是任何时候都该优先使用。

第一种替代方案是 Zero-shot CoT。CoT 是 Chain-of-Thought，白话说就是“要求模型按步骤思考再作答”。它不提供示例，只提供推理指令，例如“请先逐步分析，再给出最终答案”。这种方法的优点是保留了 zero-shot 的简洁性，缺点是提升是否稳定，依赖任务类型和模型本身。

第二种方案是 `few-shot + CoT`。这通常在数学、逻辑、多步决策题上更强，因为它既给了示例，又给了推理轨迹。但它也最贵、最长、最容易受模板细节影响。

第三种方案是自动示例检索，也叫 retrieve-then-prompt。检索，白话说就是先从样例库里找和当前问题最像的例子，再把这些例子拼进 prompt。它不是固定 5-shot 或 8-shot，而是动态找最相关示例。这种方法在业务数据充足时常常很有效，但它测到的已不只是“语言模型本体”，还包含检索系统质量。

对比如下：

| 方案 | 是否给示例 | 优点 | 局限 |
|---|---|---|---|
| Zero-shot | 否 | 基线清晰，成本最低 | 可能低估可用性能 |
| Zero-shot CoT | 否 | 对推理题有时有效 | 对指令敏感，稳定性不一定好 |
| Fixed few-shot | 是，固定 $k$ 个 | 容易复现，适合 benchmark | 对示例选择和顺序敏感 |
| Few-shot + CoT | 是 | 多步任务常有更高上限 | 成本高，模板复杂 |
| Retrieved examples | 是，动态检索 | 更贴近真实业务场景 | 引入检索系统变量 |

玩具例子可以这样理解：

- Zero-shot：`请解这道应用题。`
- Zero-shot CoT：`请一步一步推理后给出答案。`
- Few-shot：先给 3 道同类题及答案，再给新题。
- Few-shot + CoT：先给 3 道同类题的完整推理过程，再给新题。

真实工程里怎么选，取决于你到底在回答什么问题。

如果你在做论文式对比或模型基础能力排名，zero-shot 更像统一基线。  
如果你在做“上线后会不会配模板”的产品评估，few-shot 更贴近真实部署。  
如果你在做高价值推理任务，比如财务表格问答、客服复杂工单归因、数学辅导，few-shot 或 few-shot + CoT 通常更值得测。  
如果你有成熟样例库和检索系统，retrieve-then-prompt 可能比固定 8-shot 更实用，但此时评测口径必须明确写出“模型 + 检索”的整体系统结果，不能再假装只是模型分数。

结论是：few-shot 和 zero-shot 不是谁替代谁，而是它们回答的评测问题不同。只有在任务定义、模板、示例策略和成本口径都写清楚时，分数才有解释力。

---

## 参考资料

1. `Benchmarking Zero-Shot vs. Few-Shot Performance in LLMs`，2023 年 12 月。关键贡献：明确区分 zero-shot 与 few-shot 的评测定义，并用 $\Delta = Score_{few}(k) - Score_{zero}$ 描述 few-shot 增益，同时给出 MMLU 等任务上的对比数据。

2. QLoRA / MMLU evaluation guide。关键贡献：说明 MMLU 常见的 5-shot 评测实践，帮助理解“先给若干问答示例，再评估正式题目”的 benchmark 组织方式。

3. GSM8K 相关评测实践说明。关键贡献：展示多步数学推理任务对 few-shot 和推理模板更敏感，解释为什么工程中经常单独关注 GSM8K 的 shot 设定。

4. BytePlus 关于 LLM benchmark 的工程总结。关键贡献：从工程视角讨论示例数量、顺序、模板差异、token 成本和稳定性问题，适合把 benchmark 分数转化为上线决策。

5. MMLU benchmark 公开资料。关键贡献：帮助理解知识型多选题 benchmark 为什么更强调任务格式、选项约束与示例一致性。
