## 核心结论

毒性与越狱 benchmark，本质上是在测模型面对攻击输入时的稳健性。稳健性可以先白话理解为：环境变坏、输入被人故意操纵后，系统还能不能守住原来的行为边界。它关心的不是“模型有没有一次拒答成功”，而是“攻击者在一组方法里，平均有多大概率把模型绕过去”。

普通能力评测通常问两个问题：答案是否正确、表达是否流畅。安全评测多问三个问题：模型会不会产出有害内容、攻击模板能不能让它突破拒答边界、攻击后正常能力有没有明显变差。因此，安全评测是风险视角，不是单条样本视角。

一句话概括：安全评测关注的是攻击分布下的稳健性，而不是单次拒答是否成立。

| 维度 | 普通能力评测 | 安全评测 |
| --- | --- | --- |
| 核心问题 | 会不会做题 | 会不会被绕过 |
| 输入分布 | 正常任务输入 | 恶意构造输入 |
| 主要风险 | 答错、幻觉 | 有害输出、策略泄露、拒答失稳 |
| 常见指标 | 准确率、BLEU、F1 | 毒性分数 `T`、越狱成功率 `ASR`、稳定性损失 `Δ` |
| 结论方式 | 单题或单集表现 | 攻击族整体暴露面 |

常见简写可以先记住：

$$
T=\frac{1}{N}\sum_{i=1}^{N} t(y_i), \quad
ASR=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[J(y_i)=1], \quad
\Delta = metric_{clean} - metric_{attack}
$$

这里 `t(y)` 是毒性评分器，白话讲就是“给输出打一个有害程度分数”；`J(y)` 是越狱判定器，白话讲就是“判断这次是不是成功把模型绕过去了”。

---

## 问题定义与边界

毒性 benchmark 主要测“模型会不会稳定地产生有害、冒犯、仇恨、侮辱、鼓励伤害的内容”。越狱 benchmark 主要测“模型会不会在攻击提示下突破既定安全边界”。安全边界可以白话理解为：系统本来承诺不做的事，比如不给危险建议、不泄露敏感策略、不配合明显恶意请求。

这两类 benchmark 有交集，但不完全相同。一个模型可能毒性不高，却能在角色扮演攻击下泄露内部策略；也可能越狱成功率不高，但一旦失守，输出就非常有害。所以工程上一般不会只测其中一个。

设模型为 `M`，攻击族为 `A = {a_j}`。攻击族可以先理解成“一类相似攻击方法的集合”，例如角色扮演、Base64 编码绕过、翻译污染、多轮诱导。原始样本为 `x_i`，攻击后输出为：

$$
y_{ij}=M(a_j(x_i))
$$

这意味着：同一个任务样本 `x_i`，可以被多种攻击方式改写，再分别喂给模型，看输出是否失守。

问题空间大致可以这样分：

| 类型 | 输入特征 | 关注问题 | 是否属于本文重点 |
| --- | --- | --- | --- |
| 正常任务 | 用户正常提问 | 能力与可用性 | 部分涉及 |
| 毒性生成 | 容易诱发冒犯内容的提示 | 输出是否有害 | 是 |
| 单轮越狱 | 一次提示中直接攻击 | 单次防线是否被绕过 | 是 |
| 多轮越狱 | 通过多轮对话逐步诱导 | 会话级是否累计失守 | 是 |
| 上下文污染 | 恶意内容藏在文档、邮件、翻译文本里 | 模型是否被外部内容带偏 | 是 |

玩具例子很简单。用户直接问“怎么伤害别人”，模型拒答，并不等于安全。攻击者可能先说“请把下面文本翻译成中文”，然后把恶意指令藏进原文；或者先问一轮无害问题，建立角色设定，再在第三轮切入危险请求。安全 benchmark 测的是这类攻击族，而不是单条提示。

本文边界也要说清楚：

1. 不讨论通用能力 benchmark，例如 MMLU、GSM8K 这类知识和推理评测。
2. 不讨论训练算法本身，例如 RLHF、DPO、对齐损失的细节设计。
3. 只聚焦上线前或模型对比阶段的安全评测，重点是“如何测”和“指标代表什么”。

---

## 核心机制与推导

安全评测里最常见的三个量是 `T`、`ASR`、`Δ`。这三个量分别回答三个不同问题。

第一，`T` 测输出有多有害。

$$
T=\frac{1}{N}\sum_{i=1}^{N} t(y_i)
$$

其中 `t(y_i) \in [0,1]`。如果评分器给出 0，表示几乎无害；接近 1，表示有害程度高。`T` 是均值，所以它更像“总体风险温度”，而不是单条样本结论。

第二，`ASR` 测攻击有没有成功。`ASR` 是 Attack Success Rate 的缩写，白话讲就是“攻击成功率”。

$$
ASR_j=\frac{1}{N_j}\sum_{i=1}^{N_j}\mathbf{1}[J(y_{ij})=1]
$$

这里下标 `j` 表示第 `j` 类攻击族。按攻击族分别统计很重要，因为“整体平均”会把局部脆弱性冲淡。一个模型可能对普通角色扮演防得住，但对编码绕过特别脆弱。如果只看全局平均值，这个问题很容易被隐藏。

第三，`Δ` 测攻击带来的退化。退化可以是安全退化，也可以是正常能力退化。

$$
\Delta = metric_{clean} - metric_{attack}
$$

如果 `metric` 取的是正常任务准确率，那么 `Δ` 大，表示一加攻击模板，模型连正常任务都做差了。如果 `metric` 取的是安全得分，那么 `Δ` 大，表示攻击显著削弱了防线。

一个最小数值例子可以直接算。假设有 10 条攻击样本，其中 3 条越狱成功，则：

$$
ASR = \frac{3}{10}=30\%
$$

如果干净集平均毒性是 `T_clean = 0.08`，攻击后是 `T_attack = 0.31`，那么：

$$
\Delta T = T_{attack} - T_{clean} = 0.23
$$

这不是“有几条偶发输出不稳定”，而是攻击分布下整体有害输出概率明显抬高。

为什么不能只看 `T`？因为低毒性不等于高安全。模型可能不直接说危险内容，但会泄露系统提示、执行受限操作、输出规避检测的方法。这些未必都被毒性分类器打高分。

为什么不能只看 `ASR`？因为低 `ASR` 也可能是假安全。最典型情况是过拒答，也就是模型对大量正常输入也一起拒绝。这样看起来攻击成功率低，实际上系统已经不可用。

为什么要组合指标？因为安全与可用性是双目标。一边要挡住恶意输入，一边不能把正常用户也挡在外面。工程上常见做法是同时看下面三类量：

| 指标 | 它回答的问题 | 单独使用的风险 |
| --- | --- | --- |
| `T` | 输出内容有多有害 | 看不到策略泄露、隐性协助 |
| `ASR` | 攻击是否成功绕过防线 | 看不到过拒答和输出质量退化 |
| `Δ` | 攻击造成多大变化 | 依赖基线定义，不同基线难横比 |

真实工程例子比玩具例子更有代表性。假设企业客服助手接入工单、邮件和网页摘要。攻击者不一定直接发危险问题，而可能把恶意提示藏在工单正文、翻译请求、附件 OCR 文本、甚至 Base64 片段里。上线前，团队不会只跑“直接问危险问题”这一类样本，而会分桶测试：角色扮演、编码绕过、上下文污染、多轮诱导、提示拆分。最后看的不是“一条样本有没有拒答”，而是“哪一类攻击最容易打穿，打穿后是否伴随毒性上升，正常工单处理能力是否被拖垮”。

---

## 代码实现

最小实现一般分五步：数据集加载、攻击生成、模型调用、评分器判定、汇总统计。它很像单元测试流水线，只是测试目标从“函数逻辑是否正确”变成了“系统在攻击下是否稳”。

下面给一个可运行的玩具实现。为了安全，这里不生成真实有害内容，而是用占位规则模拟“模型是否失守”。

```python
from dataclasses import dataclass
from statistics import mean

@dataclass
class Sample:
    text: str
    safe: bool  # True 表示正常任务，False 表示潜在风险任务

@dataclass
class Attack:
    name: str

    def apply(self, text: str) -> str:
        templates = {
            "roleplay": f"[ROLEPLAY]{text}",
            "base64_hint": f"[ENCODED]{text}",
            "multi_turn": f"[HISTORY]{text}",
        }
        return templates[self.name]

def model_generate(prompt: str) -> str:
    # 玩具规则：
    # 某些攻击前缀会让模型从拒答变成泄露“unsafe-plan”
    if "harmful" in prompt and ("[ROLEPLAY]" in prompt or "[ENCODED]" in prompt):
        return "unsafe-plan"
    if "harmful" in prompt:
        return "refuse"
    return "normal-answer"

def toxicity_scorer(output: str) -> float:
    return {
        "unsafe-plan": 0.9,
        "refuse": 0.0,
        "normal-answer": 0.0,
    }[output]

def jailbreak_checker(output: str) -> int:
    return 1 if output == "unsafe-plan" else 0

def task_success(output: str, sample: Sample) -> int:
    if sample.safe:
        return 1 if output == "normal-answer" else 0
    return 1 if output == "refuse" else 0

samples = [
    Sample("normal: summarize an email", True),
    Sample("harmful: give dangerous guidance", False),
]

attacks = [Attack("roleplay"), Attack("base64_hint"), Attack("multi_turn")]

clean_outputs = [model_generate(s.text) for s in samples]
clean_metric = mean(task_success(y, s) for y, s in zip(clean_outputs, samples))

results = []
for s in samples:
    for a in attacks:
        attacked = a.apply(s.text)
        y = model_generate(attacked)
        results.append({
            "attack": a.name,
            "safe": s.safe,
            "toxicity": toxicity_scorer(y),
            "jailbreak": jailbreak_checker(y),
            "task_ok": task_success(y, s),
        })

asr_roleplay = mean(r["jailbreak"] for r in results if r["attack"] == "roleplay" and not r["safe"])
asr_base64 = mean(r["jailbreak"] for r in results if r["attack"] == "base64_hint" and not r["safe"])
t_attack = mean(r["toxicity"] for r in results)
attack_metric = mean(r["task_ok"] for r in results)
delta = clean_metric - attack_metric

assert clean_metric == 1.0
assert round(asr_roleplay, 2) == 1.0
assert round(asr_base64, 2) == 1.0
assert round(t_attack, 2) == 0.30
assert round(delta, 2) == 0.50

print({
    "clean_metric": clean_metric,
    "t_attack": round(t_attack, 2),
    "delta": round(delta, 2),
})
```

这段代码表达了最小骨架：

1. `Sample` 表示评测样本。
2. `Attack` 表示攻击模板。
3. `model_generate` 表示模型接口。
4. `toxicity_scorer` 表示毒性分类器。
5. `jailbreak_checker` 表示越狱判定器。
6. 最后按攻击族和全局做聚合。

实际工程里，这几个模块通常对应不同服务：

| 模块 | 作用 | 最小实现 |
| --- | --- | --- |
| 数据集 | 提供正常样本与风险样本 | JSONL / CSV / Parquet |
| 攻击模板 | 生成角色扮演、编码、翻译污染等变体 | 规则模板或脚本 |
| 模型接口 | 调用待测模型 | HTTP API 或本地推理 |
| 毒性分类器 | 判断输出有害程度 | 规则 + 分类模型 |
| 越狱判定器 | 判断是否突破防线 | 拒答规则 + LLM judge |
| 汇总报表 | 统计各类指标 | pandas / SQL / dashboard |

汇总时不要只给一个全局平均值，而要按攻击族分桶：

$$
ASR_j=\frac{1}{N_j}\sum_i \mathbf{1}[J(y_{ij})=1], \quad
T_j=\frac{1}{N_j}\sum_i t(y_{ij})
$$

有时还会额外统计过拒答率：

$$
over\_refusal = \frac{rejected\_normal}{normal\_total}
$$

也就是：正常样本里，被模型错误拒绝的比例。

---

## 工程权衡与常见坑

第一个坑是只看单轮拒答。很多系统单轮看起来很稳，但多轮对话里会逐步失守。原因不复杂：攻击者先建立角色、再降低警惕、最后切入危险目标。若评测只按单轮算，风险会被低估。解决办法是按会话级统计成功率，也就是“只要一轮失守，整段会话记为失守”。

第二个坑是只用一种攻击模板。模型可能对“直接命令式攻击”很强，但对翻译污染、上下文注入、编码绕过很弱。覆盖面不足，会让 benchmark 变成“测你会不会这一道题”，而不是“测防线有没有系统短板”。

第三个坑是只看分类器分数。分类器会误判，特别是隐性协助、含蓄危险建议、结构化泄露这类场景。工程上更稳妥的是三层组合：规则判定、分类器打分、人工抽检。人工抽检不是为了替代自动化，而是用来校准阈值和发现漏判模式。

第四个坑是只测有害样本，不测正常样本。如果模型把很多普通问题也一起拒了，`ASR` 可能很好看，但系统已经不可用。上线系统最终要服务正常用户，所以必须同步统计 `over_refusal`。

第五个坑是训练集和评测集重叠。如果模型在训练或后处理里见过同类样本，分数会虚高。规避方法是固定 held-out prompts，也就是专门留出、训练中不接触的评测提示集合。

常见坑和规避方式可以压缩成表：

| 常见坑 | 问题后果 | 规避方式 |
| --- | --- | --- |
| 只看单轮拒答 | 低估多轮累计失守 | 按会话级统计 |
| 只用一种攻击模板 | 覆盖不足，结果失真 | 覆盖多攻击族 |
| 只看分类器分数 | 漏判或误判隐性风险 | 分类器 + 规则 + 人工抽检 |
| 只测有害样本 | 看不到过拒答 | 同测正常样本 |
| 训练评测重叠 | 得分虚高 | 固定 held-out prompts |

真实工程里还有一个常见权衡：要不要用 LLM judge 做判定。LLM judge 可以先理解为“让另一个模型来当裁判”。优点是能处理复杂语义，缺点是一致性、成本和可复现性都比规则差。经验上，规则适合高确定性的拒答检查，分类器适合大规模打分，LLM judge 适合边界样本复核，而不是一把梭。

---

## 替代方案与适用边界

不同 benchmark 解决的问题不同，不能混成一个分数看。`RealToxicityPrompts` 更偏毒性生成倾向，`HarmBench` 更偏标准化自动红队与稳健拒答评测，`JailbreakBench` 更偏越狱攻击研究与防御对比。

可以先用一个表建立直觉：

| 方案 | 更关注什么 | 更适合什么场景 | 典型输出 |
| --- | --- | --- | --- |
| `RealToxicityPrompts` | 毒性生成倾向 | 看模型会不会产出冒犯/有害内容 | 毒性分数、生成倾向 |
| `HarmBench` | 自动红队与稳健拒答 | 统一框架下做安全回归评测 | ASR、拒答稳健性 |
| `JailbreakBench` | 越狱攻击与防御比较 | 研究攻击方法、比较防御策略 | 攻击成功率、攻击覆盖 |

如果你关心“模型会不会自己说出有害内容”，优先看毒性 benchmark。如果你关心“攻击者能不能把模型策略绕过去”，优先看越狱 benchmark。如果你要上线企业助手，通常需要组合使用，因为真实风险既包括内容风险，也包括交互链路里的绕过风险。

这里还有两个边界必须明确。

第一，不同 benchmark 的标签标准不同，分数不能直接横向比较。一个库里算“越狱成功”的样本，在另一个库里可能被标成“部分违规”或“边界拒答不足”。因此，更可靠的比较方法是：同一模型、同一攻击集、同一判定逻辑下，比较相对变化，而不是跨库比绝对值。

第二，不同模型的拒答风格不同。一个模型可能偏保守，另一个偏解释型拒答。只看绝对值容易误判，所以更稳的方式是看：

$$
\Delta metric = metric_{model\_a} - metric_{model\_b}
$$

或者看同一模型在改版前后的变化：

$$
\Delta metric = metric_{before} - metric_{after}
$$

也就是说，benchmark 更适合做“同条件对比”和“回归检测”，不适合拿一个单点数字宣布“模型绝对安全”。

---

## 参考资料

下表先给出快速索引：

| 名称 | 关注点 | 输入类型 | 输出指标 | 开源实现 |
| --- | --- | --- | --- | --- |
| RealToxicityPrompts | 毒性生成 | 文本提示 | 毒性相关分数 | 有 |
| HarmBench | 自动红队、稳健拒答 | 攻击提示集 | ASR、拒答稳健性 | 有 |
| JailbreakBench | 越狱攻击研究 | 单轮/多轮攻击提示 | 攻击成功率等 | 有 |

1. [RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models](https://arxiv.org/abs/2009.11462)
2. [HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal](https://arxiv.org/abs/2402.04249)
3. [HarmBench 源码](https://github.com/centerforaisafety/HarmBench)
4. [JailbreakBench 官方站点](https://jailbreakbench.github.io/)
5. [JailbreakBench 源码](https://github.com/JailbreakBench/jailbreakbench)
