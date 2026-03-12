## 核心结论

Gemini 的代码生成与执行架构，本质上不是“模型会写 Python”这么简单，而是“模型可以把 Python 当成一个工具反复调用”。这里的工具，指的是模型在回答过程中可以主动触发的外部能力。对 Gemini 来说，这个能力叫 code execution，即一个受限的 Python 沙箱。模型先生成代码，再执行，再把执行结果放回上下文，最后继续推理。

这件事的工程意义很直接：凡是“文字推理容易飘，计算结果必须落地”的任务，比如数值计算、表格统计、画图、简单搜索、算法验证，都更适合交给“模型 + 沙箱”的闭环，而不是只让模型口头解释。官方文档给出的边界也很清楚：代码执行有时间上限，单次最多 30 秒；语言只支持 Python；库是白名单，不能自己安装；不能直接访问你机器上的本地磁盘，但 Gemini API 的较新文档支持把 CSV、文本等文件以输入内容的形式送进沙箱处理，而不是让沙箱自己去读一个本地路径。

它和 AlphaCode 2 的关系也很重要。AlphaCode 2 不是“聊天时顺手跑几段 Python”，而是把 Gemini 作为核心代码模型，叠加海量采样、执行过滤、聚类去重、评分重排，形成一条更重型的 search pipeline。search pipeline 可以直白理解为“先生成很多候选，再筛掉垃圾，再从不同思路里挑最优”。这套系统在 Codeforces 上达到约 85th percentile，说明 Gemini 的代码能力不仅能支撑交互式分析，也能作为复杂程序搜索系统的底座。

如果只看公开 benchmark，代码执行架构和静态代码 benchmark 不是一回事。前者强调“生成后可运行、可校验、可迭代”，后者强调“单次输出是否正确”。截至 2026 年 3 月，Google DeepMind 页面列出的 Gemini 2.0 Flash-Lite 在非 agentic、pass@1 设置下，HumanEval 为 90.2%，MBPP 为 75.8%。这说明基础代码生成已经不弱；而 code execution 的价值，在于把“会写”进一步推进到“会验证、会修正”。

---

## 问题定义与边界

先把问题讲窄：本文讨论的不是 Gemini 所有编码能力，而是“Gemini 在回答过程中，如何通过 code execution 工具完成代码生成、执行和回写”。

这里有三个边界必须先分清。

第一，只有显式开启工具，模型才有资格真跑代码。否则它只能描述“我会这样写”，本质上还是文本补全。对新手来说，这个区别很关键。你看到一段看起来像代码的回答，不等于结果已经算过。只有请求里配置了 `codeExecution`，并且模型真的触发了它，数值和图表才来自执行结果。

第二，沙箱不是你的电脑。沙箱，直白说，就是一台临时、隔离、权限受限的小运行环境。它能做内存里的 Python 计算，但不能直接读取你本地的 `/Users/.../report.csv`。如果你说“请分析我电脑里的报表”，正确做法不是让模型去猜路径，而是把文件作为输入上传给 Gemini，再在提示里明确让它用 Pandas 读入并分析。官方文档在这里有两个看似矛盾的表述：Vertex 的 model reference 页面写“doesn't support file I/O”，而 Gemini API 文档又说明 Gemini 2.0 Flash 起支持文件输入和图表输出。把这两句合起来理解，边界其实是明确的：它不支持任意文件系统访问，也不支持自由读写文件 URI；但它支持把文件内容以内联输入的形式送进执行环境。

第三，code execution 不是 AlphaCode 2。前者是对话里的工具调用，适合交互式小闭环；后者是竞赛级程序搜索系统，适合高难度开放题。两者共享的核心思想是“生成后执行，再根据执行反馈继续筛选”，但计算预算完全不在一个量级。

| 维度 | Gemini code execution | AlphaCode 2 |
| --- | --- | --- |
| 核心目标 | 在对话中辅助推理和计算 | 在竞赛题上搜索高质量程序 |
| 典型预算 | 单轮、秒级、多次小迭代 | 大规模采样，百万级候选 |
| 运行语言 | 沙箱执行 Python | 最终主要提交 C++ |
| 结果形态 | 文本、表格、图表、执行输出 | 排序后的候选程序集合 |
| 适用场景 | 数据分析、验证、调试 | Codeforces 这类复杂算法题 |

---

## 核心机制与推导

Gemini code execution 的核心闭环可以写成：

$$
\text{一轮推理} = \text{采样} + \text{执行} + \text{观察} + \text{再采样}
$$

这里的“采样”，就是模型生成一段候选代码；“执行”，就是把代码送进 Python 沙箱运行；“观察”，就是读取 stdout、错误信息或图表结果；“再采样”，就是基于这些反馈继续生成下一段代码或最终答案。

术语第一次出现时可以这么理解：

- 采样：从模型里“抽”出一个具体答案，不同温度和随机性会让代码方案不同。
- stdout：程序标准输出，直白说就是 `print()` 打出来的内容。
- 上下文：模型当前能看到的全部输入，包括你的问题、之前的回答、代码执行结果。

为什么这种结构有用？因为很多任务的正确性来自“可运行验证”，而不是“文字看起来像对”。例如让模型心算第 20 个斐波那契数，模型可能答对，也可能答错；但让它写 5 行 Python 跑一下，正确率会更稳定。

一个玩具例子很适合说明这点。问题是：“求第 20 个斐波那契数，再找最近的回文数。”如果只靠文本补全，模型可能把 6765 附近的回文数说错；如果进入闭环，它会先算出 6765，再程序化枚举或构造邻近回文，最后给出 6776。这里关键不是“Python 更聪明”，而是“把容易错的局部推理改成了可验证计算”。

再看 AlphaCode 2。它把这个思想推到极致：不是只生成 1 段代码，而是生成海量候选代码，然后执行样例、过滤不合格者、按行为聚类去重，再用评分模型排序。为什么海量采样有效？一个直观公式是：

$$
P(\text{至少有一个正确候选}) = 1 - (1-p)^n
$$

其中 $p$ 是单个候选命中的概率，$n$ 是候选数量。即使单个候选很弱，只要候选足够多，整体命中率也会上升。当然，真实工程不会无限增大 $n$，因为还要付执行、过滤、排序的成本。所以 AlphaCode 2 不是“多生成一点”而已，而是“多生成 + 强过滤 + 强重排”。

真实工程例子比玩具例子更能说明价值。假设你做一个周报机器人，用户上传销售 CSV，希望自动回答三件事：总销售额、异常波动、趋势图。纯文本模型容易在聚合统计或百分比变化上出错；而 code execution 可以让模型直接用 Pandas 计算，再用 Matplotlib 出图，然后把图和结论一起返回。这个架构的重点不在“图画得出来”，而在“图背后的数值和描述来自同一次执行”，所以一致性更高。

---

## 代码实现

如果你要从 API 角度理解 Gemini 的执行架构，最小心智模型只有两步：

1. 在请求里声明 `tools=[Tool(code_execution=ToolCodeExecution())]`
2. 让模型面对一个明确需要计算或验证的任务

下面这个 Python 例子不依赖 Gemini SDK，本身可以直接运行。它模拟的是“生成候选解，再用执行结果校验”的最小闭环。重点不是复刻 Gemini，而是让初学者看到：为什么执行反馈会改变最终答案。

```python
def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def is_palindrome(x: int) -> bool:
    s = str(x)
    return s == s[::-1]

def nearest_palindrome(x: int) -> int:
    offset = 0
    while True:
        lower = x - offset
        upper = x + offset
        if lower >= 0 and is_palindrome(lower):
            return lower
        if is_palindrome(upper):
            return upper
        offset += 1

value = fib(20)
pal = nearest_palindrome(value)

assert value == 6765
assert pal == 6776

print(value, pal)
```

这个例子里，`assert` 的作用是“把模型的猜测变成机器可检查的约束”。约束，直白说，就是程序必须满足的条件。真实的 code execution 也是类似思路：模型先产出代码，再由沙箱帮它检查“算出来的东西到底对不对”。

如果接 Gemini API，最关键的不是 prompt 花样，而是工具配置。官方示例的结构大致如下：

```python
from google import genai
from google.genai.types import Tool, ToolCodeExecution, GenerateContentConfig

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="Calculate 20th fibonacci number. Then find the nearest palindrome to it.",
    config=GenerateContentConfig(
        tools=[Tool(code_execution=ToolCodeExecution())],
        temperature=0,
    ),
)

for part in response.candidates[0].content.parts:
    if part.executable_code:
        print(part.executable_code.code)
    if part.code_execution_result:
        print(part.code_execution_result.output)
```

这里有两个工程细节值得注意。

一是 `temperature=0`。temperature 可以理解为“输出随机性旋钮”。做计算或验证任务时，低温度通常更稳，因为你希望模型优先走最直接、最可复现的代码路径。

二是要读 `parts`。Gemini 的返回不是只有一段最终文本，它可能包含“生成的代码”“代码执行结果”“最后总结”。这其实暴露了它的内部工作方式：回答不是一次性写完，而是中途插入过工具调用。

真实工程例子可以这样设计：用户上传 `sales.csv`，prompt 写成“请用 Pandas 读取上传文件，计算各区域销售额与环比变化，并用 Matplotlib 画柱状图和趋势图”。这类任务特别适合 Gemini 的 code execution，因为输入是结构化数据，输出又需要数值和图同时成立。

---

## 工程权衡与常见坑

最常见的坑，不是代码写错，而是架构理解错。

第一坑，是没开工具就让模型“算”。这会导致模型只进行文本推理。文本推理不是没用，但它不具备执行校验，所以一旦问题涉及多步计算、边界条件或数据清洗，误差会快速累积。

第二坑，是把沙箱当完整运行环境。官方文档明确给出 30 秒限制，而且只能用白名单库。你不能指望它 `pip install` 一个冷门依赖，也不能假设它能长期保存中间文件。对复杂任务，正确做法通常是拆小轮次：先清洗，再统计，再可视化，而不是一次 prompt 塞满所有操作。

第三坑，是混淆“文件输入”和“本地文件访问”。当前公开文档允许把 CSV、文本等内容作为输入提供给 code execution，并支持图表以内联结果形式返回；但这不等于它能自己去读你电脑或服务器上的任意路径。这个边界如果不说清，新手最容易在“帮我打开本地文件”这类需求上卡住。

第四坑，是误把 benchmark 成绩等同于产品体验。HumanEval、MBPP 反映的是模型在标准代码题上的单次生成能力；但交互式应用里，真正决定体验的是“工具触发是否稳定、错误信息是否可利用、输出是否可追踪”。也就是说，90% 的 HumanEval 不等于你的数据分析工作流就 90% 稳定。

| 坑 | 根因 | 规避方式 |
| --- | --- | --- |
| 没配置 `codeExecution` | 模型没有执行权限 | 请求里显式声明工具 |
| 任务一次塞太大 | 30 秒上限、上下文噪声增多 | 拆成多轮短任务 |
| 依赖本地路径 | 沙箱不能直接访问本地磁盘 | 通过上传或内联提供文件内容 |
| 依赖冷门三方库 | 只能使用官方支持库 | 预先核对白名单 |
| 只看最终文字 | 中间执行失败被忽略 | 检查 code/result parts |

还有一个容易被忽略的权衡：开启 code execution 不一定总是更好。官方文档明确提醒，某些非计算任务上，开启该工具可能带来其他输出质量回退。原因不神秘，模型一旦拥有工具，有时会过度倾向“写代码解决”，而不是直接给出自然语言结果。所以它更像“专用增益器”，不是“全场景默认开关”。

---

## 替代方案与适用边界

如果你的任务只是“求个数、画个图、验证一个算法边界”，Gemini 的 code execution 很合适。它的优势是闭环短、接入轻、结果能回灌上下文，用户体验接近“边想边算”。

但如果你的任务超出这些边界，就该换方案。

一种替代方案是 Function Calling 加自建服务。Function Calling，直白说，就是模型不直接执行代码，而是调用你提供的后端函数。这样你就能访问公司数据库、本地文件系统、私有模型、内部 API 和自定义依赖。代价是你要自己维护服务、安全策略、超时控制和日志链路。

另一种替代方案是 AlphaCode 2 那类 search pipeline。它适合竞赛题、复杂算法题、需要大规模候选搜索的问题。其核心不是“把 Python 沙箱开起来”，而是“给足采样预算，再用执行与评分把错误候选层层淘汰”。这条路性能更高，但成本也高得多，不适合普通业务请求。

还可以给出一个简单判断法：

- 任务结果能否在 30 秒内用 Python 内存计算完成？
- 是否只需要官方支持库？
- 是否不依赖直接访问本地或私有数据源？
- 是否需要把执行结果继续喂回模型做下一步推理？

如果四个问题大多回答“是”，用 code execution 通常合适；如果大多回答“否”，更适合函数调用或外部执行系统。

| 方案 | 适用场景 | 优势 | 局限 |
| --- | --- | --- | --- |
| 纯 code execution | 表格分析、数学计算、画图、短流程调试 | 接入简单，结果可回写上下文 | 30 秒限制，库受限 |
| Function Calling + 自建服务 | 企业数据、私有依赖、长任务 | 权限和环境完全可控 | 基础设施成本更高 |
| AlphaCode 2 式 search pipeline | 竞赛、复杂算法、开放式程序搜索 | 候选覆盖广，正确率上限更高 | 算力和工程复杂度都高 |

---

## 参考资料

| 资料 | 内容 |
| --- | --- |
| Google AI for Developers: Code execution | 工具能力、支持库、图表输出、文件输入边界 |
| Vertex AI: Execute code with the Gemini API | API 配置、30 秒限制、斐波那契与回文示例 |
| Google Developers Blog: Gemini 2.0 Deep Dive: Code Execution | AI Studio 工具开关、典型数据分析与可视化场景 |
| Google DeepMind: AlphaCode 2 Technical Report | 大规模采样、过滤、聚类、评分、85th percentile |
| Google DeepMind: Gemini Diffusion / model comparison page | Gemini 2.0 Flash-Lite 的 HumanEval 与 MBPP 公开分数 |

- Google AI for Developers, Code execution: https://ai.google.dev/gemini-api/docs/code-execution
- Vertex AI, Execute code with the Gemini API: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/code-execution-api
- Google Developers Blog, Gemini 2.0 Deep Dive: Code Execution: https://developers.googleblog.com/zh-hans/gemini-20-deep-dive-code-execution
- Google DeepMind, AlphaCode 2 Technical Report: https://deepmind.google/AlphaCode2_Tech_Report.pdf
- Google DeepMind, Gemini Diffusion: https://deepmind.google/models/gemini-diffusion/
