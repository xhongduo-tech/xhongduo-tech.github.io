## 核心结论

MMLU、HumanEval、MATH 分别测三种不同能力：知识问答、代码生成、数学推理。它们都很有用，但都不是“模型真实能力”的完整代表。更准确的说法是：它们测的是“模型在一套固定题型、固定评分规则下的表现”。

先看差异：

| 基准 | 主要考什么 | 典型输入 | 典型指标 | 最主要限制 |
| --- | --- | --- | --- | --- |
| MMLU | 多学科知识与选项判断 | 多选题 | 准确率 | 容易受提示词、记忆和选项猜测影响 |
| HumanEval | 代码生成是否能通过测试 | 函数签名 + docstring + 单测 | pass@k | 测试集小，容易被“刷题式优化” |
| MATH | 数学竞赛题求解与精确作答 | 题目文本 | 准确率 | 精确匹配严格，但未必等于真实推理能力 |

HumanEval 可以先用一个新手版直觉理解：它像一道编程练习题，系统允许你交多个候选答案，`pass@k` 表示“抽出 $k$ 个候选时，至少有一个是对的概率”。所以它测的不是“单次输出有多稳”，而是“采样多次后能不能撞到一个正确程序”。

共同弱点也要直接说清楚：

1. 数据污染。数据污染就是训练数据里混进了评测题，模型可能是在背题，不是在泛化。
2. 评测偏差。评测偏差就是评分方法本身会偏向某种回答风格，例如更长、更像标准答案、或排在前面的回答。
3. 单榜单误导。单一 leaderboard 很容易把“会做这套题”误读成“在真实工程里也可靠”。

所以，MMLU、HumanEval、MATH 适合当“能力切片”，不适合当“终局裁判”。如果目标是选模型做真实产品，必须把 benchmark 分数和真实任务测试一起看。

---

## 问题定义与边界

先把三个 benchmark 的边界划清。

MMLU 是 Massive Multitask Language Understanding 的缩写，核心形式是一套覆盖 57 个学科的大型选择题考试。题目横跨数学、法学、医学、历史、计算机等领域，难度从高中水平到专业考试水平不等。常见设置是 0-shot 和 5-shot。0-shot 就是不提供示例直接作答，5-shot 就是先给 5 个示例再答题。它主要测“模型能不能把题意映射到正确选项”。

HumanEval 是 OpenAI 提出的代码生成评测。它不是让模型解释代码，而是让模型“写出一个真的能跑过测试的函数”。每题通常给出函数签名、自然语言说明和若干隐藏测试。模型输出 Python 代码，评测系统执行测试，再统计通过情况。这里的重点不是“代码看起来像不像”，而是“代码运行后行为对不对”。

MATH 是竞赛数学题数据集，题目大量来自中学竞赛和进阶训练材料。它不是普通口算题，也不是简单代数应用题，而是包含组合、数论、几何、代数等较强推理成分的问题。模型不只要给方向，还要给出最终精确答案。这里的“精确”很关键，意思是答案通常要和标准答案在规范化后匹配，而不是“思路差不多就给分”。

可以用一个新手版类比快速区分：

- MMLU 像模拟大学考试，多数题是四选一。
- HumanEval 像“按题意写函数并通过判题”。
- MATH 像“写出数学解法并给出最终标准答案”。

它们的输入、输出和允许的反馈并不一样：

| 基准 | 输入形式 | 输出要求 | 评分方式 | 可接受反馈 |
| --- | --- | --- | --- | --- |
| MMLU | 题干 + 选项，可能附 few-shot 示例 | 选项字母或对应答案 | 是否答对 | 通常只有对错 |
| HumanEval | 函数定义 + docstring + 测试环境 | 可执行 Python 函数 | 通过测试的概率 | 单测通过/失败、异常 |
| MATH | 题目文本，常含公式 | 最终数学答案，常需规范格式 | 精确匹配准确率 | 通常只有是否匹配 |

这里要特别强调边界问题。

MMLU 的边界在于，它更接近“知识检索加判断”，而不完全等于深层推理。一个模型可能知道很多答案，但不代表它能在开放环境中持续做复杂任务。比如它可以答对“哪个器官负责胰岛素分泌”，但不代表它能在真实医疗对话里稳定处理病史、追问缺失信息、识别风险和避免幻觉。

HumanEval 的边界在于，它评的是“短小、函数级、测试驱动”的代码生成，不直接等于“改大型代码库 bug 的能力”。会做 164 道函数题，不代表会修 CI 失败、会理解十万行项目、会处理依赖和环境，也不代表会在多文件调用链里定位问题。

MATH 的边界在于，它偏向竞赛数学题。竞赛题需要形式化推理，但现实里的“推理任务”并不都长这样。很多业务推理是带噪声、不完整、还要调用工具的，例如财务对账、风控审核、工单排查，这些任务往往同时依赖外部数据、规则约束和中间验证步骤。

为了避免把三者混为一谈，可以再看一个对照：

| 任务场景 | 更接近哪个 benchmark | 原因 | 不能直接推出什么 |
| --- | --- | --- | --- |
| 学科问答助手 | MMLU | 都是知识检索与判断 | 不能推出长流程执行稳定性 |
| 写算法题函数 | HumanEval | 都是函数级代码生成 | 不能推出仓库级维护能力 |
| 数学辅导或解题器 | MATH | 都要求多步推导与最终答案 | 不能推出开放任务决策能力 |

结论是：benchmark 名字相近，目标却不同。先看输入形式、输出形式和评分机制，再谈分数高低，才不会误解它们在测什么。

---

## 核心机制与推导

MMLU 和 MATH 的主指标都比较直接，通常就是准确率：

$$
\text{Accuracy}=\frac{\text{答对题数}}{\text{总题数}}
$$

如果一套题有 100 道，模型答对 72 道，准确率就是 $72\%$。

这个指标看起来简单，但含义并不完全一样。

- 在 MMLU 里，准确率对应“最终选项是否正确”。
- 在 MATH 里，准确率对应“最终答案是否规范化后匹配标准答案”。

也就是说，同样都是准确率，背后的评测对象并不相同。MMLU 测选项判断，MATH 测最终答案匹配。

MMLU 常比较 0-shot 和 5-shot，是因为 few-shot 示例会帮助模型“进入题型”。这不是让模型突然学会新知识，而是降低它对题面格式、回答方式和任务分布的误解。

一个玩具例子可以说明 5-shot 为什么常有提升。

假设模型第一次见到一类法律题时，不确定输出格式，可能回答整句解释：

> 根据公司法相关规定，我认为更合理的答案是 B，因为……

但评测脚本只接受 `A/B/C/D`。这时模型即使“方向对”，也可能因为格式错而丢分。看过 5 个示例后，它知道应直接输出 `B`，同时知道类似题目如何抽取关键词，例如“主体责任”“举证责任”“例外条款”。于是分数提高。这个提高里，既有能力利用上下文的提升，也有“格式适配”带来的提升，二者不能混为一谈。

HumanEval 的机制更值得展开，因为 `pass@k` 不是普通准确率。

设：

- $n$ 是总共生成的候选程序数
- $c$ 是其中真正能通过测试的程序数
- $k$ 是你最终抽取来尝试的程序数

那么至少抽到一个正确程序的概率是：

$$
\text{pass@k}=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}
$$

这条式子可以这样理解：

1. 先算“抽到的 $k$ 个程序全错”的概率。
2. 全错的意思是，这 $k$ 个都从 $n-c$ 个错误程序里抽出来。
3. 再用 $1$ 减去它，就是“至少有一个对”。

这里有一个必要补充。很多论文在估计 `pass@k` 时，不是简单拿“前 $k$ 个样本是否包含正确答案”做统计，而是基于无放回抽样的组合公式来估计。如果模型生成了很多样本，其中有若干个能过测试，那么 `pass@k` 实际是在估计“随机取 $k$ 个时成功的概率”。所以它不是排序指标，而是采样指标。

用最小玩具例子算一遍。假设某题一共采样 $n=3$ 个程序，其中只有 $c=1$ 个能过测试。

那么：

$$
\text{pass@1}=1-\frac{\binom{2}{1}}{\binom{3}{1}}=1-\frac{2}{3}=\frac{1}{3}
$$

$$
\text{pass@2}=1-\frac{\binom{2}{2}}{\binom{3}{2}}=1-\frac{1}{3}=\frac{2}{3}
$$

这说明同一个模型，单次作答成功率可能只有 $33\%$，但如果允许采样两个候选，成功率就能升到 $67\%$。

所以 `pass@k` 本质上在测两件事：

- 模型能否生成正确解。
- 模型的正确解是否能在多次采样中出现。

这也是为什么 `pass@k` 往往比单次准确率更高。它奖励“多样性里包含正确解”，但不保证“第一次就稳定正确”。

真实工程里可以这样理解。假设你让模型写一个“保序去重”函数：

- 第一个候选用了 `set`，会破坏顺序。
- 第二个候选忘了处理重复元素。
- 第三个候选同时满足“去重”和“保持原始顺序”。

那么 `pass@1` 可能失败，但 `pass@3` 成功。这说明模型“有能力生成对的解”，但不说明它“足够稳定，适合直接上线”。

MATH 的难点则在另一边。它通常要求最终答案精确匹配，这能减少“说了一堆似是而非的话也得分”的问题，但也会带来另一个局限：推理过程可能是对的，只因格式不一致或最后一步笔误而记为错；反过来，也可能答案碰巧对，但过程并不可靠。

一个简单例子：

- 标准答案是 $\frac{1}{2}$。
- 模型输出 `0.5`，如果评测脚本支持等价规范化，可能算对。
- 模型输出 `1/2 `，多了空格，通常也会被规范化。
- 模型输出 `x = 1/2`，如果脚本只提取最终数值，也可能算对。
- 模型输出 `0.5000001`，即使推理过程只差一步，也通常算错。

因此，MATH 的准确率有两个来源：一部分来自真实推理质量，另一部分来自答案抽取与规范化质量。后者虽然是工程细节，但会直接影响分数。

把三者放在一起看，可以得到一个更稳妥的理解：

| benchmark | 分数上升可能意味着什么 | 也可能混入什么因素 |
| --- | --- | --- |
| MMLU | 知识覆盖更广，题型适应更好 | 提示词适配、选项猜测、记忆污染 |
| HumanEval | 更容易在采样中生成正确程序 | 测试集局限、样本数增加带来的收益 |
| MATH | 最终答案更常命中标准结果 | 答案规范化、格式抽取、题库记忆 |

所以，读 benchmark 分数时，不能只问“高了多少”，还要问“是哪里变高了”。

---

## 代码实现

下面用一个可运行的 Python 玩具实现展示 HumanEval 风格的 `pass@k` 统计。这个例子不调用真实模型，只模拟“生成多个候选程序并跑测试”的流程，并补上了异常处理、基本测试执行和一个最小可运行入口。

```python
from __future__ import annotations

from math import comb
from typing import Callable, Iterable


def pass_at_k(n: int, c: int, k: int) -> float:
    """Estimate pass@k from n samples with c correct ones."""
    if not (0 <= c <= n):
        raise ValueError("c must satisfy 0 <= c <= n")
    if not (1 <= k <= n):
        raise ValueError("k must satisfy 1 <= k <= n")

    # If there are fewer than k wrong samples, drawing k all-wrong samples is impossible.
    if n - c < k:
        return 1.0

    return 1.0 - comb(n - c, k) / comb(n, k)


def run_candidate(
    fn: Callable[[int], int],
    tests: Iterable[tuple[int, int]],
) -> bool:
    """Return True if fn passes all tests, False otherwise."""
    try:
        for x, expected in tests:
            if fn(x) != expected:
                return False
        return True
    except Exception:
        return False


def count_passing_candidates(
    candidates: Iterable[Callable[[int], int]],
    tests: Iterable[tuple[int, int]],
) -> int:
    """Count how many candidates pass all tests."""
    tests = list(tests)
    passed = 0
    for fn in candidates:
        if run_candidate(fn, tests):
            passed += 1
    return passed


def main() -> None:
    # 玩具例子：3 个候选里 1 个正确
    candidates = [
        lambda x: x + x,   # 错：平方被写成了乘 2
        lambda x: x * x,   # 对
        lambda x: abs(x),  # 错：只对负数看起来“像对”
    ]
    tests = [(2, 4), (3, 9), (-2, 4)]

    n = len(candidates)
    c = count_passing_candidates(candidates, tests)

    p1 = pass_at_k(n, c, 1)
    p2 = pass_at_k(n, c, 2)

    print(f"n={n}, c={c}")
    print(f"pass@1={p1:.6f}")
    print(f"pass@2={p2:.6f}")

    assert c == 1
    assert round(p1, 6) == round(1 / 3, 6)
    assert round(p2, 6) == round(2 / 3, 6)


if __name__ == "__main__":
    main()
```

运行输出应接近：

```text
n=3, c=1
pass@1=0.333333
pass@2=0.666667
```

如果把它映射到真实评测流程，大致是下面这样：

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class HumanEvalProblem:
    prompt: str
    hidden_tests: list[str]


@dataclass
class MMLUQuestion:
    prompt: str
    choices: list[str]
    gold: str


@dataclass
class MathProblem:
    prompt: str
    gold: str


class DummyModel:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


def run_hidden_tests(code: str, hidden_tests: Iterable[str]) -> bool:
    """
    占位函数：真实系统里需要安全执行代码、限制时间和内存、
    拦截异常，并在受限环境中运行隐藏测试。
    """
    raise NotImplementedError


def normalize_choice(pred: str) -> str:
    pred = pred.strip().upper()
    for ch in ("A", "B", "C", "D"):
        if pred == ch or pred.startswith(ch):
            return ch
    return pred


def extract_final_answer(text: str) -> str:
    """
    最小占位实现：真实数学评测常需要提取 \\boxed{}、最后一行答案、
    或特定格式中的最终结果。
    """
    return text.strip().splitlines()[-1].strip()


def normalize_math_answer(ans: str) -> str:
    return ans.replace(" ", "").lower()


def evaluate_humaneval(
    problem: HumanEvalProblem,
    model: DummyModel,
    n: int = 20,
    k: int = 1,
) -> float:
    samples = [model.generate(problem.prompt) for _ in range(n)]
    c = sum(1 for code in samples if run_hidden_tests(code, problem.hidden_tests))
    return pass_at_k(n=n, c=c, k=k)


def evaluate_mmlu(
    questions: list[MMLUQuestion],
    model: DummyModel,
) -> float:
    correct = 0
    for q in questions:
        pred = model.generate(q.prompt)
        answer = normalize_choice(pred)
        if answer == q.gold:
            correct += 1
    return correct / len(questions)


def evaluate_math(
    problems: list[MathProblem],
    model: DummyModel,
) -> float:
    correct = 0
    for p in problems:
        pred = model.generate(p.prompt)
        final_answer = extract_final_answer(pred)
        if normalize_math_answer(final_answer) == normalize_math_answer(p.gold):
            correct += 1
    return correct / len(problems)
```

这里有三个工程点很关键。

第一，HumanEval 不是“字符串相等比较”，而是“执行代码并看测试是否通过”。这意味着评测系统要处理超时、异常、资源限制和沙箱隔离。沙箱隔离就是把候选代码放在受限环境里跑，避免恶意代码读写系统、发起网络请求或卡死评测机。

第二，MATH 也不是纯文本比较。通常需要先做答案规范化，例如把分数、小数、根式、括号、空格统一到某种标准形式，否则很多本质相同的答案会被误判。常见处理包括：

| 原始答案 | 规范化后可能变成 | 目的 |
| --- | --- | --- |
| `1 / 2` | `1/2` | 去除空格差异 |
| `0.5` | `1/2` 或 `0.5` | 处理等价表示 |
| `\boxed{3}` | `3` | 去掉外层格式包装 |
| `x=7` | `7` | 提取最终数值 |

第三，MMLU 的实现虽然最简单，但也有隐藏细节。比如模型输出“我认为答案是 C，因为……”时，脚本必须先抽取选项；如果模型输出多个候选字母，如何取舍；如果采用 few-shot，不同示例顺序是否影响结果；这些都会影响最终准确率。

真实工程例子可以看得更清楚。假设你在公司里评测“自动写数据清洗函数”的模型：

- 用 HumanEval 风格，你会给输入输出约束和单测。
- 用 MMLU 风格，几乎没法测，因为这不是选择题。
- 用 MATH 风格，也不合适，因为重点不是最终符号答案，而是代码是否在异常输入下稳定运行。

这说明 benchmark 的评测机制必须和任务机制对齐，否则分数没有决策意义。不是所有任务都该转成一个统一指标，任务不一样，评法也应该不一样。

---

## 工程权衡与常见坑

真正做模型选型时，最大的坑不是“没 benchmark”，而是“把 benchmark 当现实”。

先看最常见问题：

| 问题 | 白话解释 | 后果 | 常见缓解方式 |
| --- | --- | --- | --- |
| 数据污染 | 训练时见过测试题 | 分数虚高，误以为泛化强 | 去重、时间切分、私有评测集 |
| Prompt 依赖 | 换个提示词分数就变 | 结果不稳定，不可复现 | 固定模板，多模板平均 |
| Judge bias | 评分模型偏爱某种答案 | 排名失真 | 双盲对比、多 judge、人审抽检 |
| 过拟合榜单 | 专门优化某题库 | 真实任务效果差 | 加入真实业务集和在线指标 |
| 指标错配 | 用错指标测错能力 | 得分高但无业务价值 | 按任务选指标 |

数据污染是最严重的问题。想象一下，训练语料里直接包含了评测题及其答案。模型在测试时给出高分，并不一定说明它会推理，也可能只是记住了题。对 MATH 的争议就常围绕这一点展开：如果训练过程接触过相同或高相似度题目，分数就会偏高，leaderboard 变成“记忆强度排行榜”。

MMLU 也会受到污染影响，而且它的问题更隐蔽。因为 MMLU 是选择题，模型即使没有完整记住整道题，也可能记住高频知识点或近似表述，再加上选项结构帮助，最终分数看起来仍然很高。这就会出现一种常见误判：把“熟悉题库分布”当成“稳定掌握知识”。

HumanEval 也不是天然干净。题目数量小、格式固定、Python 社区内容丰富，题目被改写、翻译、转载后混入训练语料并不难。这样一来，即使没有逐字重复，也可能出现“语义污染”。模型未必见过完全相同的题面，但可能见过等价问题和标准写法。

再看 Prompt 依赖。很多 benchmark 分数不是模型常数，而是“模型 + 提示词模板 + 采样参数 + 后处理脚本”的联合结果。比如 HumanEval 中，温度设高一点，样本多样性会上升，`pass@10` 可能提高；但 `pass@1` 可能反而下降。MMLU 中，提示词如果明确要求“只输出字母”，准确率也可能上升。也就是说，benchmark 排名往往不是单一模型能力的纯投影。

再看 LLM-as-Judge。它的意思是“用一个语言模型给另一个模型打分”。这个方法扩展性很强，因为开放式回答很难人工大规模评分。但它有明显偏差：

1. Position bias。位置偏差，意思是先出现的答案更容易被选中。
2. Verbosity bias。冗长偏差，意思是写得更长的答案容易被误判为更好。
3. Self-preference。自我偏好，意思是 judge 模型更偏爱与自己风格接近的回答。

新手可以这样理解：如果让一个强模型当裁判，它可能更喜欢“像它自己写出来的答案”，哪怕另一个答案更短、更准。

真实工程里，这会造成很具体的问题。比如你在比较两个客服回复模型：

- 模型 A：短、准、直接解决问题。
- 模型 B：长、礼貌、看起来解释更全。

如果 judge 偏好冗长文本，模型 B 可能得分更高，但真实用户可能更喜欢 A，因为更省时间、可执行性更强。

规避方式通常不是“完全不用 LLM judge”，而是把它关进流程里：

1. 固定评审模板，减少随意性。
2. 打乱回答顺序，消除位置偏差。
3. 用 pairwise comparison，即两两比较，而不是直接打绝对分。
4. 加入人工抽检，校准 judge。
5. 对关键任务保留可执行指标，例如是否修复 bug、是否通过测试，而不是只看主观评分。

还要补一个容易被忽略的坑：单榜单误导。很多团队看到某个模型在公开榜单第一，就默认它在所有任务上都更强。但实际情况通常是：

- A 模型在 MMLU 更强，说明它知识题更稳。
- B 模型在 HumanEval 更强，说明它代码采样里更容易出现正确程序。
- C 模型在真实内部任务上更强，说明它更符合你的业务分布。

所以，排行榜的正确用法不是“替代判断”，而是“缩小备选范围”。先用公开 benchmark 做初筛，再用内部任务做决策，这才是稳妥流程。

---

## 替代方案与适用边界

如果你的目标是“模型在真实工程里是否好用”，那么只看 MMLU、HumanEval、MATH 不够。需要引入更接近真实工作的评测，例如 SWE-bench。

SWE-bench 的思路和 HumanEval 很不同。它不是给一个短函数题，而是给真实 GitHub issue、真实代码仓库和真实测试套件，让模型提交补丁，再看测试是否通过。它测的不是“会不会写一段代码”，而是“能不能在真实项目里把问题修好”。

下面做一个对照：

| 评测类型 | 代表 | 更适合测什么 | 不适合测什么 | 适用边界 |
| --- | --- | --- | --- | --- |
| 固定知识题库 | MMLU | 知识覆盖、选项判断 | 长流程任务执行 | 适合早期横向比较 |
| 函数级代码题 | HumanEval | 小粒度代码生成 | 大项目维护 | 适合测代码合成能力 |
| 精确答案数学题 | MATH | 形式推理与精确作答 | 开放式决策任务 | 适合测受控推理 |
| 真实仓库修复 | SWE-bench | 工程修复、上下文理解 | 纯知识回忆 | 适合工程代理评估 |
| 对话裁判打分 | MT-Bench/LLM-as-Judge | 开放回答比较 | 高确定性可执行任务 | 适合辅助，不宜单独定胜负 |

这里的核心不是“谁更高级”，而是谁和你的任务更接近。

如果你做的是教育问答，MMLU 可能能提供一部分参考，因为你的系统确实要处理多学科知识。  
如果你做的是代码助手，HumanEval 能告诉你基础代码生成能力，但还要补上仓库级评测。  
如果你做的是数学辅导，MATH 有价值，但最好再加“过程可读性”“错误定位能力”“多步追问稳定性”等指标。  
如果你做的是软件工程代理，SWE-bench 这类真实仓库评测通常比 HumanEval 更贴近生产。

还可以把“替代方案”理解为“补足单一 benchmark 看不到的部分”。常见补法有三类：

| 补充方式 | 解决什么问题 | 适合什么场景 |
| --- | --- | --- |
| 私有业务评测集 | 公开题库与业务分布不一致 | 产品选型、内部回归 |
| 在线 A/B 或人工验收 | benchmark 高分但用户未必满意 | 面向真实用户的系统 |
| 过程指标与故障分析 | 只看最终分数看不出失败原因 | 模型迭代、系统调优 |

比如一个客服模型，公开 benchmark 分数都不错，但上线后可能仍然失败，原因可能不是知识不足，而是：

- 拒答过多，用户问题没被真正解决。
- 话太长，用户读不完。
- 不会调用内部知识库或工具。
- 在多轮对话里丢失上下文。

这些问题，MMLU、HumanEval、MATH 都很难直接测出来。

可以用一个真实工程例子收尾。假设团队要选一个模型做“自动修复后端单测失败”的助手：

- 只看 HumanEval：可能会选出一个短函数题分很高的模型。
- 加上 SWE-bench 风格内部评测：你会发现它在真实仓库里找不到正确文件、不会读调用链、修了一个 bug 又引入另一个。
- 再加人工验收：你会进一步发现它虽然能改对，但补丁风格不符合团队规范，解释也不利于 code review。
- 最终决策往往是：基础生成能力看 HumanEval，真实修复能力看仓库级评测，线上效果再看人工验收和回归测试。

所以，替代方案不是“废掉 benchmark”，而是把 benchmark 放回正确位置：  
它们适合做标准化切片，不适合代替全链路验收。

---

## 参考资料

1. Hendrycks et al., *Measuring Massive Multitask Language Understanding*. MMLU 的原始论文，定义了 57 学科、多选题设置，以及 0-shot、5-shot 的标准比较方式。
2. Chen et al., *Evaluating Large Language Models Trained on Code*. HumanEval 的原始来源，给出了 164 道 Python 函数题、隐藏测试设计和 `pass@k` 公式。
3. Hendrycks et al., *MATH: Measuring Mathematical Problem Solving With the MATH Dataset*. 说明竞赛数学题构成、答案规范化方式与精确匹配评测。
4. Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* 说明“真实 issue + 真实仓库 + 真实测试”为什么比函数级题库更接近工程场景。
5. Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. 讨论用强模型打分的可行性与偏差问题，是理解 position bias、verbosity bias、自我偏好的关键资料。
6. OpenAI HumanEval 评测代码与相关复现实验。价值在于看清 `pass@k` 的实现依赖样本数、采样策略、执行环境，而不是只看公式本身。
7. 各类数据污染分析与 benchmark 复现实验。重点不在“给一个最终污染比例”，而在提醒我们：公开 benchmark 很容易被训练语料、数据改写、后处理脚本和提示词细节影响。
8. 各类私有评测、时间切分评测和生产 A/B 方案文档。它们通常没有统一公开榜单，但在真实模型选型里往往比单个公开 benchmark 更有决策价值。
