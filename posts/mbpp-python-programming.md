## 核心结论

MBPP，全称 Mostly Basic Python Problems，可以直译为“多数是基础 Python 问题”，是一个专门评测代码生成能力的基准。基准的意思是：一套固定题目和固定打分规则，方便不同模型在同一标准下比较。它的核心价值不是考大工程，而是考模型能否稳定写出“短、小、明确”的 Python 函数。

MBPP 一共包含 974 道入门级 Python 任务。每道题通常给出四类信息：自然语言题面、目标函数、参考实现、3 条固定测试。自然语言题面就是“用文字描述要你写什么函数”；固定测试就是几条 `assert` 断言，用来自动判断答案是否正确。工程团队常把它当作代码模型的 smoke benchmark。smoke 的白话解释是：先跑一组便宜、快速、覆盖基础功能的检查，确认系统没有明显坏掉。

它适合回答两个问题。第一，模型是否还会写基础函数，比如字符串处理、列表遍历、简单数学和条件分支。第二，模型或 prompt 改动后，基础能力是否回退。它不适合回答更大的问题，比如模型是否会设计模块边界、是否能修复跨文件 bug、是否具备真实仓库中的长期推理能力。

先看 MBPP 的规模和结构：

| 维度 | 数值/说明 |
| --- | --- |
| 基准名称 | MBPP（Mostly Basic Python Problems） |
| 总题数 | 974 |
| 主要语言 | Python |
| 每题测试数 | 通常 3 条 `assert` |
| 典型输出 | 一个可执行函数 |
| 常见用途 | 代码生成评测、模型回归检测、prompt 对比 |

玩具例子最能说明它的本质。任务 1 的要求就是实现 `add_numbers(a, b)`，返回两数之和。这个题没有隐藏业务规则，也没有复杂上下文，本质是在测模型能不能把自然语言映射成最基本的函数逻辑。如果连这种题都经常失败，后续更复杂任务的结果通常也不可信。

---

## 问题定义与边界

理解 MBPP，先要明确它到底在测什么、不测什么。

它测的是“从题面到短函数实现”的能力。短函数的白话解释是：函数通常很短，几行到十几行，目标单一，不要求多文件协作。题目类型集中在列表、字符串、字典、循环、条件判断、简单数学运算这些 Python 初级主题。也就是说，MBPP 更像“统一格式的编程练习册”，而不是“真实项目开发现场”。

每个评估单元通常由若干字段组成：

| 字段 | 作用 | 白话解释 |
| --- | --- | --- |
| `task_id` | 题目唯一编号 | 用来区分第几题 |
| `text` | 自然语言描述 | 告诉模型要写什么 |
| `code` | 参考解法 | 官方提供的一个正确实现 |
| `test_list` | 测试断言列表 | 自动检查输出是否正确 |
| `challenge_test_list` | 可选附加测试 | 更严格的补充检查 |

对新手最重要的一点是：MBPP 不是让模型“解释怎么做”，而是让模型“直接写出能运行的函数”。因此，一个完整的 MBPP 评估单元至少包含三部分：题目描述、函数名或签名、测试断言。比如“写一个函数，返回两个数的和”，如果只给题意，不给函数名，测试里又调用 `add_numbers`，那模型即使理解了题目，也可能因为函数名不匹配而失败。

这就是它的边界。MBPP 测“可执行正确性”，不测“表达能力”；测“局部函数生成”，不测“系统设计”；测“有限测试下的通过率”，不等于测“真实世界中的稳健性”。

数据划分上，MBPP 通常分为训练部分和测试部分，公开资料里常见的说法是 500 道训练题、474 道测试题。一些版本还会有 sanitized 处理。sanitized 的白话解释是：对原始题目做清洗，去掉歧义、重复或不合理样本，让结果更稳定。这个动作不是为了提高难度，而是为了减少数据本身的问题干扰结论。

真实工程里，团队往往不会只看“总分”。他们更关心这些边界问题：

| 团队问题 | MBPP 能否回答 | 原因 |
| --- | --- | --- |
| 模型基础 Python 是否退化 | 能 | 题型稳定，适合比较版本 |
| prompt 改动是否影响短函数生成 | 能 | 输入输出格式统一 |
| 模型能否完成多文件重构 | 不能 | 题目太小，不含仓库上下文 |
| 模型能否处理线上复杂 bug | 不能 | 缺少日志、依赖、历史状态 |

所以，MBPP 的正确定位不是“总代码能力榜单”，而是“基础代码生成体检表”。

---

## 核心机制与推导

MBPP 的评估流程非常固定：把题目喂给模型，让模型生成函数，把官方测试拼接进去执行，看测试是否通过。这套流程简单，但它背后最关键的指标不是单次对错，而是 pass@k。

pass@k 的白话解释是：如果允许模型尝试 $k$ 次，至少有一次答对的概率有多大。这个指标适合生成式模型，因为模型同一题可能每次输出不同代码。只看一次输出，容易把“偶然抽到坏样本”误认为“模型整体很差”。

常见估计公式是：

$$
\text{pass@k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}
$$

其中：

- $n$ 是总共生成了多少个样本
- $c$ 是其中通过全部测试的样本数
- $k$ 是你实际允许挑选的候选数

这个公式的含义是：先算“挑出 $k$ 个样本且一个都没挑中正确答案”的概率，再用 1 减掉它。于是就得到“至少挑中一个正确答案”的概率。

举一个最小推导例子。假设某题一共生成了 $n=200$ 个候选，其中 $c=10$ 个是正确的。

- 对 `pass@1` 来说，直觉上就是抽 1 次答对的概率，所以近似是 $10/200 = 0.05$
- 对 `pass@10` 来说，不是简单乘 10，因为抽样之间有关联，正确做法是用组合公式：
  $$
  1 - \frac{\binom{190}{10}}{\binom{200}{10}}
  $$

这说明什么？说明模型未必“第一次就会”，但可能“多试几次就会”。这在代码生成里很常见，因为采样温度、提示词细节、上下文顺序都会影响输出。

玩具例子可以继续用 `add_numbers(a, b)`。题目要求：返回两个数之和。模型如果输出：

```python
def add_numbers(a, b):
    return a + b
```

再运行测试：

```python
assert add_numbers(2, 3) == 5
assert add_numbers(-1, 1) == 0
assert add_numbers(0, 0) == 0
```

就通过了。这说明 MBPP 的判断逻辑非常朴素：不是人工看“像不像对”，而是机器跑“过没过断言”。

真实工程例子则更有代表性。假设一个团队在做代码补全模型的日常评测，他们每晚固定跑一遍 MBPP。某天 prompt 模板从“只输出 Python 代码”改成“先解释思路，再输出代码”，结果模型大量返回解释文本，代码提取器失败，`pass@1` 立刻下降。这里分数下降不一定代表“推理能力变差”，而可能只是“输出格式偏了”。MBPP 的意义就在于它能把这种工程回归及时暴露出来。

也正因为这样，MBPP 实际测到的是一个链路的乘积结果：

$$
\text{最终得分} \approx \text{题意理解} \times \text{函数生成} \times \text{格式遵守} \times \text{测试执行成功}
$$

只要其中任一环节不稳定，分数就会掉。

---

## 代码实现

下面用一个最小可运行示例说明 MBPP 风格的自动评测。这个例子不依赖外部库，直接模拟“生成代码 -> 执行测试 -> 统计通过率”的基本流程。

```python
from math import comb

def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    assert 0 <= c <= n
    assert 1 <= k <= n
    if n - c < k:
        return 1.0
    return 1 - comb(n - c, k) / comb(n, k)

def run_mbpp_like_eval(code: str, tests: list[str]) -> bool:
    namespace = {}
    exec(code, namespace)
    for test in tests:
        exec(test, namespace)
    return True

# 玩具题：add_numbers
prompt_text = "Write a python function to find the sum of two numbers."
generated_code = """
def add_numbers(a, b):
    return a + b
"""

tests = [
    "assert add_numbers(2, 3) == 5",
    "assert add_numbers(-1, 1) == 0",
    "assert add_numbers(0, 0) == 0",
]

assert run_mbpp_like_eval(generated_code, tests) is True

# pass@k 示例
p1 = estimate_pass_at_k(200, 10, 1)
p10 = estimate_pass_at_k(200, 10, 10)

assert round(p1, 4) == 0.05
assert p10 > p1
```

这段代码体现了 MBPP pipeline 的三个核心步骤：

| 步骤 | 输入 | 输出 | 目的 |
| --- | --- | --- | --- |
| 生成 | 题面、函数要求 | 候选代码 | 让模型给出函数实现 |
| 提取 | 原始模型输出 | 可执行 Python 代码 | 去掉解释文字和格式噪声 |
| 运行测试 | 候选代码、官方断言 | 通过/失败 | 自动判断是否正确 |

如果写成更接近生产环境的伪代码，大致是这样：

```text
for task in mbpp_tasks:
    prompt = build_prompt(task.text, task.test_list, target_signature)
    samples = model.generate(prompt, n=num_samples)

    for sample in samples:
        code = extract_python_code(sample)
        program = code + "\n" + join(task.test_list)
        result = run_in_sandbox(program)
        record(result)

    compute_pass_at_k(task_results)
```

这里有两个工程点很关键。

第一，代码提取器必须稳。很多模型会输出 markdown 代码块、解释文字，甚至多个版本函数。如果提取策略不一致，评测结果会被“格式问题”污染。提取器的白话解释是：从模型大段输出里，尽量准确地拿到真正要执行的 Python 代码。

第二，执行环境必须隔离。因为评测本质是在运行模型生成的任意代码，所以通常会放进沙箱。沙箱的白话解释是：一个受限制的运行环境，避免模型代码读写系统文件、死循环占满资源、访问网络或执行危险命令。

真实工程例子通常不会只跑一题，而是整批执行。比如一个代码助手团队可能维护一套 nightly benchmark：

- 读取固定版本的 MBPP 数据
- 对每题采样 20 次
- 计算 `pass@1`、`pass@5`、`pass@10`
- 和前一版本对比
- 如果 `pass@1` 下降超过阈值，就阻止模型上线

这时 MBPP 的价值不是“绝对分数多高”，而是“相同制度下的变化是否异常”。

---

## 工程权衡与常见坑

MBPP 好用，但不能把它当成无偏、完备、无漏洞的标准答案。它有几个非常现实的坑。

第一类坑是数据污染。数据污染的白话解释是：模型在训练阶段已经见过题目或答案，测试时看起来像“会做题”，实际只是“记住了”。MBPP 公开时间较早，很多题在博客、教程、镜像站里能直接搜到。这样一来，高分并不一定代表泛化能力强，可能只是记忆命中率高。

第二类坑是测试太少。每题 3 条断言，对于基础题来说够快，但对正确性覆盖并不充分。一个实现可能碰巧通过 3 条测试，却在边界输入上失败。比如排序、去重、字符串截断这类问题，如果断言没覆盖空输入、重复元素、负数、大小写差异，模型可能拿到“假阳性”。

第三类坑是函数名和输出格式。很多失败不是算法不会，而是函数没定义对、签名不一致、返回值类型不对，或者模型输出了说明文字导致执行失败。对于自动评测平台，这些都算失败，因为平台只看“能不能运行并通过”。

下面用对照表看更清楚：

| 常见陷阱 | 具体表现 | 后果 | 规避方式 |
| --- | --- | --- | --- |
| 数据污染 | 模型背过题库 | 高分但不代表真实泛化 | 配合 sanitized 或更新基准 |
| 测试覆盖不足 | 只过公开 3 条断言 | 结果虚高 | 增加私有补充测试 |
| 函数名缺失 | 题意对了但没写目标函数名 | `NameError` 或找不到函数 | prompt 中显式给函数头 |
| 输出格式漂移 | 返回解释 + 代码混合文本 | 提取失败，执行失败 | 强制“只输出代码” |
| 环境不一致 | Python 版本或库行为不同 | 分数波动 | 固定解释器和运行参数 |

一个典型坏例子是：提示里只写“计算两个数之和”，却没有写明目标函数是 `add_numbers(a, b)`。模型可能输出 `def sum_two(x, y): ...`，逻辑完全正确，但测试执行 `add_numbers(2, 3)` 时还是失败。对人来说这是“小偏差”，对评测系统来说这是“零分”。所以在自动 benchmark 里，接口契约比“意思差不多”更重要。

还有一个容易被忽略的问题：MBPP 题目集中在数学和列表处理，这会让某些模型显得特别强，因为这些题型分布非常规律。如果团队的真实业务是 SQL 生成、Web 框架代码、异步任务编排、测试修复，那么单看 MBPP 高分几乎没有直接业务含义。它只能说明模型“基础 Python 函数生成不错”。

---

## 替代方案与适用边界

当团队开始追求更高真实性、更低污染、更强覆盖时，通常会在 MBPP 之外再加其他 benchmark。

第一类替代方案是动态题库，例如 LiveCodeBench。动态的白话解释是：题目持续更新，不是多年前固定写死的一套。这样做的主要目的是压低数据污染，因为模型更难在训练时完整见过这些新题。它更适合检验“当前真实泛化能力”，但成本也更高，因为题目维护、执行环境和版本管理都更复杂。

第二类是跨语言扩展，例如 MultiPL-E 的 MBPP 变体。跨语言的白话解释是：同一类问题不只测 Python，还会改写成 Java、C++、JavaScript 等其他语言版本。对于只做 Python 助手的团队，价值有限；但对多语言代码生成平台，能显著提高比较的完整度。

第三类是修正版或增强版，比如 MBUPP。这类版本的目标通常不是“完全换题”，而是修正原始 MBPP 中的提示歧义、测试不足、题文不一致等问题。它适合那些已经习惯 MBPP 工作流，但又希望结果更可靠的团队。

可以用一个表快速对比：

| 基准 | 主要语言 | 测试覆盖 | 更新频率 | 污染控制 | 适用场景 |
| --- | --- | --- | --- | --- | --- |
| MBPP | Python | 较少，通常 3 条断言 | 低，题库固定 | 一般 | 基础能力回归、快速 smoke |
| MBUPP | Python | 比 MBPP 更严格 | 低到中 | 更好 | 想保留 MBPP 形式但提高质量 |
| MultiPL-E MBPP | 多语言 | 依具体实现而定 | 中 | 一般 | 跨语言代码生成评测 |
| LiveCodeBench | 多语言/动态题 | 通常更强调真实性 | 高 | 更强 | 检验真实泛化与时效性 |

选择原则可以很直接：

- 如果你要一个便宜、稳定、可重复的基础 benchmark，用 MBPP。
- 如果你担心公开题库记忆效应，用 sanitized 版本或并行增加动态 benchmark。
- 如果你要评估多语言生成，不要只看 MBPP，要补跨语言基准。
- 如果你的业务是复杂仓库级任务，MBPP 只能保留为底层健康检查，不能作为主评测。

因此，MBPP 最合适的位置不是“唯一标准”，而是“第一层标准”。它像单元测试，不像全链路压测。单元测试过了，只能说明基础没坏；不能说明整套系统已经足够强。

---

## 参考资料

- LLMIndex MBPP 概览（任务数量、结构、用途）：https://llmindex.net/benchmarks/mbpp
- Michael Brenndoerfer 的 MBPP 指南（示例与测试流程）：https://mbrenndoerfer.com/writing/mbpp-mostly-basic-python-programming-benchmark
- MCPBR 对 MBPP 字段与机制的介绍：https://mcpbr.org/mbpp.html
- AI Wiki 关于 MBPP 与 pass@k 的说明：https://aiwiki.ai/wiki/mbpp
- DeepWiki 对 MBPP 执行流程的整理：https://deepwiki.com/deepseek-ai/DeepSeek-Coder/4.2-mbpp-benchmark
- Emergent Mind 关于 MBPP 污染与局限的分析：https://www.emergentmind.com/topics/mostly-basic-programming-problems-mbpp
- mgx.dev 关于 MBPP 工业用途的综述：https://mgx.dev/insights/2c57f19c4f62486ca13fbc3778262294
- EvalScope 关于 MultiPL-E MBPP 的资料：https://evalscope.readthedocs.io/en/latest/benchmarks/multiple_mbpp.html
- Emergent Mind 关于 LiveCodeBench 的介绍：https://www.emergentmind.com/topics/livecodebench
- Microsoft Research 关于改进版 MBPP 的论文页面：https://www.microsoft.com/en-us/research/uploads/prod/2024/09/Improved_MBPP_benchmark-2.pdf
