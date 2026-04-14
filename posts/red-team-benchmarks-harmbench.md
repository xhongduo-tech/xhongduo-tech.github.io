## 核心结论

HarmBench 和 AdvBench 都是大模型安全里的标准化红队测试基准，但它们回答的问题不同。

红队测试：站在攻击者视角，系统性寻找模型失效方式的测评方法。

第一，HarmBench 更像一条完整测评流水线。它不只给出“要测什么”，还规定“怎么生成测试用例、怎么判定成功、怎么避免调参污染测试集”。其核心价值是**可比性**。同一批攻击方法、同一批目标模型、同一套判定器，最后才能比较谁更危险、谁更稳健。官方论文给出 510 个精心整理的 harmful behaviors，其中 400 个文本行为、110 个多模态行为，还区分 standard、contextual、copyright、multimodal 等功能类别。[HarmBench PDF](https://www.harmbench.org/HarmBench.pdf)

第二，AdvBench 更像一组“攻模型”的标准题库。它出自 2023 年 GCG 攻击论文，对外最常见的说法是 500 条 harmful behaviors；与之配套的原始仓库还包含 500 条 harmful strings，用来检验攻击是否能逼出指定目标串。因此，工程上很多人会把 AdvBench 理解成“500 行为 + 500 字符串”的组合评测资产，但若只按 HarmBench 论文里的“去重后唯一行为数”口径统计，AdvBench 只算 58 个 unique behaviors。这两个数字并不冲突，统计口径不同而已。[arXiv 2307.15043](https://arxiv.org/abs/2307.15043) [llm-attacks GitHub](https://github.com/llm-attacks/llm-attacks) [HarmBench PDF](https://www.harmbench.org/HarmBench.pdf)

第三，两者结合最有价值。AdvBench 适合快速验证“某种 jailbreak 是否能稳定逼出目标输出”；HarmBench 适合做长期、可复现的攻击与防御对比。前者偏“打得出来吗”，后者偏“在统一标准下到底有多脆弱”。

| 数据集 | 常见规模口径 | 核心对象 | 输出验证方式 | 更适合回答的问题 |
|---|---:|---|---|---|
| HarmBench | 510 behaviors | 语义/功能层面的有害行为 | 分类器、人工标注、部分哈希规则 | 攻击/防御谁更强，结果是否可复现 |
| AdvBench | 500 behaviors；工程上常与 500 strings 一起使用 | 行为触发与目标串触发 | 子串/目标串匹配，或行为是否“合理尝试执行” | jailbreak 是否能逼出特定不安全输出 |

---

## 问题定义与边界

先把边界说清楚，否则很容易把两个基准混成一件事。

HarmBench 定义的是**统一评估问题**。论文指出，以往自动红队论文各自选题、各自算分，结果不可比，所以它强调三个性质：breadth、comparability、robust metrics。白话说，就是覆盖要广、横向能比、指标不容易被“刷分”。它的基本流程是：

行为定义 → 测试用例生成 → 攻击方法生成提示词或后缀 → 目标模型输出 → 分类器或规则判定是否命中 → 汇总 ASR。

AdvBench 定义的是**攻击目标集合**。在 harmful behaviors 设置里，攻击者希望找到一个后缀，让模型对尽可能多的有害指令给出“合理尝试执行”的回答；在 harmful strings 设置里，攻击者希望模型精确吐出预设目标串。前者偏语义成功，后者偏字面成功。

一个新手常见误解是：“只要模型输出了危险词，就是红队成功。”这不准确。因为不同任务的“成功”标准不同。

玩具例子可以这样理解：

假设你不测真实危险内容，只测一个抽象任务库，里面有 10 个禁止动作，比如“输出禁词 A 的制作步骤”“伪造身份模板 B”。  
如果攻击后模型只回答“我不能帮你”，那是失败。  
如果模型没有逐字输出目标串，但明显开始给出执行步骤，那在 HarmBench 的行为判定里可能算成功。  
如果任务要求必须输出某个精确目标串，而模型只输出相似改写，那在 AdvBench 的 harmful strings 设置里仍然算失败。

所以，HarmBench 与 AdvBench 的核心差别，不在“都测不测危险”，而在“成功到底按什么定义”。

---

## 核心机制与推导

最核心的指标是 ASR，Attack Success Rate，攻击成功率。白话说，就是“所有测试题里，攻击到底打穿了多少题”。

公式很简单：

$$
ASR=\frac{\text{命中的测试用例数}}{\text{总测试用例数}}
$$

如果 100 个测试用例里有 15 个被判定为成功，则：

$$
ASR=\frac{15}{100}=15\%
$$

这个公式简单，但真正难的是“命中”怎么定义。

HarmBench 的推进点就在这里。它把“命中”从随意主观判断，改造成可复用的判定层。论文给出了测试分类器、验证分类器，以及验证集/测试集拆分，目的就是避免一种常见作弊方式：攻击器在开发时不断针对评测器调参，最后学会的是“骗过评测器”，不是“真正越狱模型”。官方报告里，验证分类器与人工标签一致率约为 88.6%，测试分类器约为 93.2%，说明分类器虽然不是完美真理，但已经足够作为工程上的统一度量标准。[HarmBench PDF](https://www.harmbench.org/HarmBench.pdf)

AdvBench 的机制更直接。尤其在 harmful strings 设置下，成功标准几乎可以退化成字符串匹配问题。这种好处是便宜、快、客观；坏处是容易被“长度效应”误导。HarmBench 论文专门指出，基于 substring matching 的 ASR 会受生成 token 数显著影响。模型一旦输出更长，碰巧覆盖目标子串的概率就会上升，看起来 ASR 变高，但这不一定代表攻击更鲁棒。

这正好解释为什么 HarmBench 要做标准化。

| 判定方式 | 白话解释 | 优点 | 主要问题 |
|---|---|---|---|
| 子串匹配 | 看输出里有没有目标片段 | 快、便宜、自动化强 | 容易受输出长度影响 |
| 行为分类器 | 看回答是否在“做这件危险事” | 更接近真实风险 | 需要训练与校准分类器 |
| 人工判官 | 人来审查是否命中 | 最接近最终语义判断 | 慢、贵、难大规模复现 |

再看数据集设计本身。HarmBench 不是简单堆数量，它强调“唯一行为”。也就是说，同一件事的不同措辞不应被算成多个独立样本，因为那更像攻击器该做的“改写能力”，不是基准该送分的“重复题”。这也是为什么它在论文里把 AdvBench 的 unique behaviors 统计为 58，而不是 500。它在做语义去重，不是在否认 AdvBench 的 500 行为文件存在。

---

## 代码实现

工程上，可以把两套评测都抽象成三层：

1. `case`：测试题，可能是行为描述，也可能是目标字符串。
2. `attack`：把原始请求加工成攻击输入，比如加后缀、加角色设定、加上下文污染。
3. `judge`：决定这一题是否算成功。

下面给一个可运行的 Python 玩具实现。它不包含任何危险内容，只演示统计逻辑。

```python
from dataclasses import dataclass
from typing import List


@dataclass
class Case:
    kind: str   # "behavior" or "string"
    prompt: str
    target: str


def mock_attack_model(prompt: str) -> str:
    # 玩具模型：遇到 "unlock" 就输出更激进的回复
    if "unlock" in prompt.lower():
        return "ACTION: comply with request"
    return "REFUSE: cannot help"


def behavior_classifier(response: str) -> bool:
    # 行为判定：只要看起来开始执行任务，就算命中
    return response.startswith("ACTION:")


def string_match(response: str, target: str) -> bool:
    return target in response


def evaluate(cases: List[Case]) -> float:
    hits = 0
    for case in cases:
        response = mock_attack_model(case.prompt)
        if case.kind == "behavior":
            hit = behavior_classifier(response)
        elif case.kind == "string":
            hit = string_match(response, case.target)
        else:
            raise ValueError(f"unknown case kind: {case.kind}")
        hits += int(hit)
    return hits / len(cases)


harmbench_like = [
    Case(kind="behavior", prompt="normal request", target=""),
    Case(kind="behavior", prompt="unlock hidden mode", target=""),
]

advbench_string_like = [
    Case(kind="string", prompt="normal request", target="ACTION"),
    Case(kind="string", prompt="unlock hidden mode", target="ACTION"),
]

asr_behavior = evaluate(harmbench_like)
asr_string = evaluate(advbench_string_like)

assert asr_behavior == 0.5
assert asr_string == 0.5
print("behavior ASR =", asr_behavior)
print("string ASR =", asr_string)
```

这段代码对应的正是：

- HarmBench 风格：`classifier(response)` 决定有没有“开始执行行为”。
- AdvBench string 风格：`target in response` 决定有没有逼出目标串。
- 最后统一聚合为 `hits / total`。

真实工程例子则更接近下面的流程：

| 步骤 | HarmBench 常见做法 | AdvBench 常见做法 |
|---|---|---|
| 数据准备 | 读取行为集，区分 standard/contextual/multimodal | 读取 harmful behaviors 或 harmful strings |
| 攻击执行 | 跑 GCG、AutoDAN、PAIR 等攻击器 | 常见于 GCG 及其变体 |
| 响应收集 | 保存 prompt、attack payload、response、元数据 | 同样保存，便于复现实验 |
| 成功判定 | 分类器、人工抽检、哈希规则 | 行为是否尝试执行，或目标串是否命中 |
| 汇总报告 | 按模型、攻击、类别统计 ASR | 按攻击设置统计 ASR |

如果你在做公司内部模型评估，一个实用组合是：先用 AdvBench 风格 case 快速筛查，发现高风险攻击模板；再把这些模板接到 HarmBench 风格流水线里，在统一判定器下做回归测试和防御对比。

---

## 工程权衡与常见坑

第一个坑是把“字符串命中”当成“真实风险命中”。

如果一个模型原本只输出 20 个 token，现在为了规避拒绝模板改成输出 200 个 token，那么 substring matching 的命中率可能明显上升。指标涨了，但不代表模型真的更容易被稳定操控。HarmBench 论文的 Figure 2 就专门展示了生成 token 数会显著改变基于子串匹配的 ASR。[HarmBench PDF](https://www.harmbench.org/HarmBench.pdf)

第二个坑是在测试集上调参。

这和机器学习里的“偷看答案”是同一个问题。若你让攻击器直接朝测试分类器反复优化，最终得到的往往不是更强攻击，而是更会钻评测空子。HarmBench 的解决思路是：给开发阶段用的 validation classifier，测试阶段再换 held-out test classifier。

第三个坑是忽略题目分布。

AdvBench 更适合测试“特定攻击是否有效”，但它对多模态、上下文依赖、版权类行为的覆盖不足。HarmBench 则把 contextual 和 multimodal 纳入统一框架，这更接近真实部署场景。一个只在单轮文本指令上安全的模型，放到长上下文或图文输入里，可能完全不是同一个安全水平。

第四个坑是只看总平均，不看分桶结果。

同一个模型对不同类别的鲁棒性可能完全不同。HarmBench 论文就显示，不同语义类别和功能类别的 ASR 差异明显。工程上至少要按“攻击方法、模型版本、行为类别”三维切分，否则你很难知道防御到底改进了什么。

---

## 替代方案与适用边界

如果目标是快速验证一个 jailbreak 模板值不值得继续研究，AdvBench 往往更高效。原因很简单：题库明确，成功标准清晰，复现实验成本低。你很快就能知道“这个后缀在 500 个行为上能打穿多少”。

如果目标是做正式安全报告、版本回归、攻防联调，HarmBench 更合适。因为它不仅给题，还给判定协议、验证/测试拆分和跨方法对比框架。它适合回答“防御上线后，整体风险有没有下降”这种工程问题。

一个真实工程例子是 HarmBench 论文中的 R2D2 对抗训练结果。作者在 Zephyr 7B 相关设置上展示，对 GCG 的 ASR 可显著下降；同时也指出，若训练时只针对单一攻击，面对风格差异较大的攻击方法时，收益会变弱。这说明 HarmBench 的价值不只是“找漏洞”，还在于帮助你验证防御是否泛化。[HarmBench PDF](https://www.harmbench.org/HarmBench.pdf)

可以把选择规则压缩成一句话：

- 要快速验攻击模板，用 AdvBench。
- 要统一比较攻击和防御，用 HarmBench。
- 要做真正上线前的闭环评估，两者一起用。

---

## 参考资料

1. Mantas Mazeika et al. “HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.” ICML 2024. 官方 PDF：<https://www.harmbench.org/HarmBench.pdf>
2. Andy Zou et al. “Universal and Transferable Adversarial Attacks on Aligned Language Models.” arXiv:2307.15043：<https://arxiv.org/abs/2307.15043>
3. `llm-attacks` 官方仓库，包含 AdvBench 相关实验与数据说明：<https://github.com/llm-attacks/llm-attacks>
4. Hugging Face 上的 AdvBench 数据卡镜像：<https://huggingface.co/datasets/NoorNizar/AdvBench>
5. HarmBench 官方项目页：<https://www.harmbench.org/>
