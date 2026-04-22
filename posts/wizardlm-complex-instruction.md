## 核心结论

WizardLM 是一种基于 `Evol-Instruct` 的复杂指令微调方法。复杂指令微调，指的是用包含多条件、多步骤、多约束任务的数据，对语言模型做监督微调，让模型更稳定地理解并执行复杂要求。

它的核心不是“收集更多指令”，而是把简单指令逐步演化成更复杂、覆盖更广的训练样本，再用这些样本训练模型。训练重点从“题量”转向“题目的复杂度分布”。

一个新手版例子是：

```text
解释梯度下降
  -> 用例子解释梯度下降并说明学习率过大时会怎样
  -> 在指定格式下比较梯度下降、牛顿法和 Adam，并分析适用场景
```

这三条指令考察的不是同一个能力。第一条主要考察定义复述，第二条加入数值推导和现象解释，第三条加入比较、格式约束和场景判断。WizardLM 的价值在于让模型见到这种复杂度递增的训练分布。

| 维度 | 传统人工指令微调 | WizardLM 式复杂指令微调 |
|---|---|---|
| 数据来源 | 人工编写或人工整理 | 从种子指令出发，用 Evol-Instruct 演化 |
| 主要目标 | 提升基础问答和指令遵循 | 提升复杂任务的指令遵循 |
| 难度分布 | 容易集中在简单任务 | 人为构造从简单到复杂的梯度 |
| 关键风险 | 成本高、覆盖有限 | 合成数据可能重复、矛盾或失真 |

流程可以概括为：

```text
简单指令
  -> 指令演化
  -> 复杂指令集合
  -> 标准 SFT
  -> 更强的复杂任务处理能力
```

---

## 问题定义与边界

监督微调，简称 SFT，是用成对的输入和标准答案训练模型，让模型在看到类似输入时生成期望输出。传统 SFT 的常见瓶颈不是只有“数据太少”，还包括“数据难度分布不均”。模型可能在简单问答上表现不错，但面对多条件、多步骤、跨约束任务时不稳定。

简单指令和复杂指令的差别，不只是字数。复杂指令通常包含更多条件、更明确的输出格式、更长的推理路径，或者多个目标之间的约束关系。

客服场景里：

```text
怎么退款
```

这是简单指令。它只要求模型复述退款规则。

```text
订单超 7 天、已拆封、使用了优惠券、只退部分商品，请按政策判断是否可退，并给出客服回复和后台操作步骤
```

这是复杂指令。它要求模型同时处理时间条件、商品状态、优惠券影响、部分退货、政策判断、回复话术和操作流程。两者考察的不是知识量本身，而是条件判断和步骤组织能力。

| 简单指令 | 复杂指令 | 为什么复杂 | 训练收益 |
|---|---|---|---|
| 解释什么是梯度下降 | 用二次函数演示一次梯度下降更新，并说明学习率过大时的后果 | 加入数值计算和异常分析 | 提升推导和解释稳定性 |
| 怎么退款 | 超 7 天、已拆封、用了优惠券、部分退货时怎么处理 | 多条件分支和业务规则组合 | 提升复杂工单处理能力 |
| 写一个排序函数 | 写一个稳定排序函数，说明时间复杂度，并补充测试用例 | 同时要求代码、解释和验证 | 提升工程任务完成度 |
| 总结这段话 | 按“背景、问题、结论、风险”四段总结，并限制每段 50 字 | 格式和长度受约束 | 提升格式遵循能力 |

WizardLM 解决的是“如何构造更有训练价值的指令数据”。它不是替代预训练。预训练，指模型在大规模文本上学习语言和知识分布的阶段。WizardLM 也不是专门解决工具调用、长上下文问答或安全对齐的方法。工具调用需要函数 schema、调用轨迹和执行反馈；长上下文需要位置编码、检索或上下文压缩；安全对齐通常还需要偏好数据、拒答策略和红队测试。

---

## 核心机制与推导

`Evol-Instruct` 的核心机制是递进演化。演化算子，指把一条已有指令改写成更复杂或覆盖更广指令的操作。设种子指令为 $x_0$，第 $t$ 步使用的演化算子为 $E_{k_t}$，则新指令可以抽象为：

$$
x_t = E_{k_t}(x_{t-1})
$$

其中 $E_k$ 可以是增加约束、增加推理步骤、加入具体场景、要求指定输出格式、扩大任务范围等操作。目标不是让所有指令都变成最长，而是让复杂度分布更合理。可以把复杂度函数记为 $c(x)$，理想情况下希望：

$$
c(x_0) < c(x_1) < ... < c(x_T)
$$

一个玩具例子如下：

```text
x0: 解释什么是梯度下降

x1: 用函数 f(w)=w^2 举例，说明从 w=2 开始做一次梯度下降更新

x2: 在学习率分别为 0.1 和 1.2 时，比较更新结果，并解释学习率过大为什么可能震荡

x3: 比较梯度下降和牛顿法，输出表格，并说明各自适用场景
```

复杂度递增链条可以写成：

```text
定义复述
  -> 数值计算
  -> 参数影响分析
  -> 方法比较 + 表格格式 + 场景判断
```

训练阶段仍然是标准 SFT。设训练集为 $D = \{(x_i, y_i)\}$，其中 $x_i$ 是指令，$y_i$ 是期望答案，模型参数为 $\theta$，目标函数为：

$$
L(\theta) = - \sum_{(x,y)\in D} \log p_\theta(y|x)
$$

这里的关键不在损失函数创新，而在数据分布变化。普通 SFT 和 WizardLM 式 SFT 都可以使用交叉熵损失。交叉熵，白话说就是“标准答案在模型眼里越不可能，损失越大”。如果某个答案的概率从 $0.25$ 提升到 $0.50$，单样本损失从 $-\log(0.25)\approx1.39$ 降到 $-\log(0.50)\approx0.69$。训练让模型在复杂指令上也把正确答案看得更可能。

演化伪代码如下：

```text
for seed in seed_instructions:
    x = seed
    for step in range(T):
        op = choose_evolution_operator()
        x_new = evolve(x, op)
        if pass_quality_check(x_new):
            save(x_new)
            x = x_new
```

真实工程例子是企业知识库问答。普通 FAQ 数据可以覆盖“怎么退款”“怎么改地址”“发票怎么开”。但真实客服工单经常是组合问题：订单状态、商品状态、用户权益、促销规则、售后政策同时出现。把这些复杂指令加入微调后，模型更容易形成稳定输出结构：先判断条件，再给结论，再列操作步骤，最后补充例外情况。

---

## 代码实现

实现上通常拆成三段：种子指令准备、指令演化生成、SFT 训练数据构造。重点是数据管线，不是修改模型结构。

数据流如下：

```text
种子数据
  -> 演化数据
  -> 过滤与抽检
  -> SFT 训练集 D={(x_i,y_i)}
  -> 微调后的模型
```

下面是一个可运行的 Python 玩具实现。它不调用大模型，只模拟“增加约束”和“质量过滤”的最小流程，用来说明数据管线。

```python
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Sample:
    instruction: str
    answer: str


def load_seed_instructions() -> List[str]:
    return [
        "解释什么是梯度下降",
        "说明退款规则",
        "写一个 Python 排序函数",
    ]


def evolve_instruction(instruction: str, level: int) -> str:
    if level == 1:
        return f"{instruction}，并给出一个具体例子"
    if level == 2:
        return f"{instruction}，要求按步骤输出，并说明一个常见错误"
    if level == 3:
        return f"{instruction}，要求输出表格，比较至少两种方案，并说明适用场景"
    raise ValueError("level must be 1, 2, or 3")


def filter_samples(instructions: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in instructions:
        text = " ".join(item.split())
        too_short = len(text) < 8
        too_long = len(text) > 120
        duplicated = text in seen
        contradictory = "必须" in text and "不要" in text and "同一内容" in text
        if not (too_short or too_long or duplicated or contradictory):
            seen.add(text)
            result.append(text)
    return result


def build_dataset(instructions: List[str]) -> List[Sample]:
    return [
        Sample(
            instruction=x,
            answer=f"这是针对指令「{x}」的标准答案占位符，真实系统中应由人工或强模型生成并抽检。",
        )
        for x in instructions
    ]


def sft_train(dataset: List[Sample]) -> dict:
    return {
        "num_samples": len(dataset),
        "loss": 1.0 / max(len(dataset), 1),
    }


seeds = load_seed_instructions()
evolved = []
for seed in seeds:
    evolved.append(seed)
    for level in [1, 2, 3]:
        evolved.append(evolve_instruction(seed, level))

filtered = filter_samples(evolved)
dataset = build_dataset(filtered)
metrics = sft_train(dataset)

assert len(seeds) == 3
assert len(filtered) == 12
assert dataset[0].instruction == "解释什么是梯度下降"
assert metrics["num_samples"] == len(dataset)
assert metrics["loss"] > 0
```

真实实现里，`evolve_instruction` 往往由强模型完成，例如让模型按照预设演化策略改写指令。`filter_samples` 不能只检查长度，还要检查重复、矛盾、答案是否覆盖全部约束、是否出现无法验证的要求。`build_dataset` 也不是简单拼接文本，而是要构造训练框架需要的格式，例如：

```json
{
  "instruction": "订单超 7 天、已拆封、用了优惠券、只退部分商品，请按政策判断处理路径",
  "output": "先判断退货条件，再判断优惠券影响，最后给出客服回复和后台操作步骤。"
}
```

还要保留一部分简单样本。原因是模型如果只看复杂样本，可能在基础问答上变得过度展开。合理做法是混合简单、中等、复杂样本，让训练分布覆盖完整难度梯度。

---

## 工程权衡与常见坑

复杂度不是“字数更长”，而是“约束更多、步骤更多、推理链更完整”。如果把“说明退款规则”改成“把退款规则写得更长”，这不是复杂化，而是冗长化。真正的复杂化应该增加条件分支、格式约束、决策顺序和异常处理。

| 坑点 | 现象 | 原因 | 规避方法 |
|---|---|---|---|
| 复杂度变长不变难 | 模型输出越来越啰嗦 | 数据只增加字数，没有增加有效约束 | 用条件数、步骤数、格式约束衡量复杂度 |
| 合成数据自相矛盾 | 模型答非所问或同时给出冲突结论 | 指令或答案内部逻辑不一致 | 做规则校验、人工抽检和强模型复核 |
| 只看自动指标不看人工可用性 | 排行榜分数高，业务上线效果差 | 自动评测不能覆盖真实流程 | 加入业务样本回放和人工验收 |
| 复杂样本比例过高 | 简单问题也输出长篇结构化答案 | 训练分布偏离用户真实请求 | 保留简单样本和中等样本 |
| 答案质量低于指令质量 | 指令很复杂，答案漏条件 | 只演化了问题，没有认真生成答案 | 对答案做覆盖率检查 |
| 去重不足 | 模型学到模板化回答 | 演化样本表面不同，本质重复 | 使用语义去重和聚类抽样 |

还有一个常见误解是：复杂指令数据越多越好。实际工程里，数据量、质量和覆盖范围要一起看。低质量复杂样本会放大错误，因为它们通常包含更多条件，模型一旦学错，会在更长链路上持续传播错误。

比较稳妥的做法是：

```text
小规模种子集
  -> 多轮演化
  -> 自动过滤
  -> 人工抽检
  -> 小规模微调实验
  -> 业务样本回放
  -> 再扩大数据规模
```

这比一次性生成大量数据更可控。

---

## 替代方案与适用边界

WizardLM 适合提升复杂指令跟随和复杂任务处理能力，但不是所有场景都应该优先采用它。

如果业务目标只是让模型更会回答固定 FAQ，人工构造高质量指令可能已经足够。如果目标是处理多条件工单、复杂代码任务、结构化分析报告或跨约束写作，复杂指令演化更合适。如果目标是让模型更符合人类偏好，偏好优化会更直接。如果目标是让模型稳定调用工具，则需要工具调用轨迹和执行反馈，而不只是复杂自然语言指令。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| WizardLM / Evol-Instruct | 可系统生成复杂指令，覆盖多难度梯度 | 合成质量控制成本高 | 多条件、多步骤、跨约束任务 |
| 纯人工 SFT | 质量可控，贴近业务语言 | 成本高，复杂样本覆盖慢 | FAQ、固定流程、专业客服话术 |
| 课程学习式微调 | 按难度排序训练，训练过程更可控 | 难度标注成本高 | 有明确难度层级的任务 |
| 混合真实对话数据 | 贴近真实用户分布 | 噪声、隐私和标注问题明显 | 已有大量线上对话日志的业务 |
| 偏好优化 | 直接优化“哪个答案更好” | 需要成对偏好数据 | 风格、帮助性、安全性对齐 |
| 工具调用微调 | 能学习函数选择和参数填充 | 需要工具 schema 和调用结果 | 搜索、下单、查库、执行操作 |

它的边界也要明确。第一，它不能替代底座模型能力。底座模型，指微调前已经完成预训练的基础模型。如果底座模型数学能力很弱，只靠复杂指令微调很难让它获得稳定高阶数学推理。第二，它不能天然解决事实正确性。复杂指令让模型更会组织答案，但事实仍可能出错。第三，它不能替代安全对齐。复杂任务中的拒答、敏感内容边界和合规策略，需要单独设计数据和评测。

更准确的定位是：WizardLM 提供了一种构造复杂指令训练集的方法。它把“怎么让模型见到更有训练价值的题”变成一个可操作的数据工程问题。

---

## 参考资料

1. [WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions](https://openreview.net/forum?id=CfXh93NDgH)
2. [Microsoft Research: WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://www.microsoft.com/en-us/research/publication/wizardlm-empowering-large-language-models-to-follow-complex-instructions/)
3. [WizardLM GitHub Repository](https://github.com/nlpxucan/WizardLM)
4. [Evol-Instruct GitHub Repository](https://github.com/nlpxucan/evol-instruct)
5. [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)
