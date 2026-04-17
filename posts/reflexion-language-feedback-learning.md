## 核心结论

Reflexion 是一种给 Agent 增加“语言化经验记忆”的方法。它不修改模型参数，也不要求重新训练，而是在一次尝试失败后，把失败原因压缩成几句自然语言，再把这些短句放进下一轮 prompt，驱动模型继续修正。可以把它直接理解成：**把踩坑记录写成短句，让模型下一次先看“避坑提示”再行动**。

它和普通 ReAct 的关键差别，不在于“会不会推理”，而在于“会不会显式记住上次为什么错”。ReAct 主要围绕“思考 - 行动 - 观察”展开，每轮都能根据环境反馈继续推进，但失败经验通常停留在局部轨迹里；Reflexion 则在这条链路外再加一层“失败后复盘”，把失败轨迹压缩成可迁移的经验，再喂回系统。因此 Agent 不只是反复重试，而是会在重试前先读到“上一次哪里出错”。

一个最容易理解的玩具例子是实现“计算列表平均值”的函数。第一轮代码可能只写了 `sum(nums) / len(nums)`，遇到空列表时会触发除零错误。Reflexion 不需要把整段 traceback 原样带回下一轮，而是只写一句“缺少空输入判断”。下一轮模型先读到这句经验，就更容易先补出 `if not nums:` 这样的分支。这也是它常被称为“语义梯度”的原因：**不是用参数梯度更新模型，而是用自然语言告诉模型“错误的方向在哪里”，从而引导下一轮往相反方向修正**。

论文中的代表性结果来自代码生成任务 HumanEval。在“自动生成测试 + 保留少量反思记忆”的设置下，Reflexion 可把 GPT-4 的 pass@1 从约 80.1% 提升到 91.0%。这里的 pass@1 可以直接理解为：**只给模型一次最终答案机会时，直接做对的比例**。这说明在“能自动验证结果”的任务上，短小但高信号的经验记忆，足以显著提升一次命中的成功率。

为了避免把它和一般“多轮对话”混淆，可以先看下面这个对照表：

| 维度 | 普通 ReAct | Reflexion + 记忆 |
|---|---|---|
| 输入 | 任务描述 + 当前上下文 | 任务描述 + 当前上下文 + 最近反思记忆 |
| 失败后响应 | 继续下一轮尝试，容易重复犯错 | 先总结失败原因，再进入下一轮 |
| 是否写入经验 | 不写入，或只保留临时轨迹 | 写入短句式反思 |
| 下一轮变化 | 依赖模型临场发挥 | 明确提示“不要再犯上次的错” |
| 适合场景 | 短链路、低重复错误 | 可迭代、可评估、易重复踩坑的任务 |

可以把它压缩成一句判断标准：**如果任务允许反复尝试，而且每轮都能明确判断对错，那么 Reflexion 往往比“单纯多试几次”更有价值。**

---

## 问题定义与边界

Reflexion 解决的问题是：**在不做微调的前提下，让 Agent 在多轮尝试中减少重复错误。** 这里的“微调”是指更新模型参数；Reflexion 不碰参数，只调整输入提示。因此它特别适合这样的工程条件：

| 场景约束 | Reflexion 为什么合适 |
|---|---|
| 只能调用闭源 API，不能训练模型 | 不需要访问模型权重 |
| 任务允许多轮尝试 | 可以把每轮失败变成下一轮经验 |
| 有稳定反馈信号 | 可以明确触发“该不该反思” |
| 想低成本增强 Agent | 只是在现有调用链上加一层 memory 和 reflection |

它的工作边界也很明确。

第一，必须有相对可靠的 evaluator，也就是**能判断本轮结果到底对不对的模块**。在代码任务里，evaluator 通常是单元测试、静态检查或沙箱执行结果；在网页自动化里，可能是“是否成功点击按钮”“是否提取到目标字段”“是否出现确认页”；在 SQL 任务里，可能是执行结果是否匹配预期模式。如果连“这次到底成功没有”都判断不清，系统就不知道何时该生成反思，更不知道反思该指向哪里。

第二，记忆不能无限增长。因为上下文窗口有限，旧经验堆得太多会稀释新问题。实践里常见做法是只保留最近 1 到 3 条反思，或者只保留与当前任务类型最相关的几条。

这个边界可以写成一个直接的更新式：

$$
mem_{t+1} = truncate(mem_t \cup \{\phi_t\})
$$

其中，$mem_t$ 表示第 $t$ 轮开始前的记忆，$\phi_t$ 表示第 $t$ 轮失败后生成的反思，`truncate` 表示把记忆裁剪到固定长度。白话解释就是：**把新教训塞进记忆，再删掉旧的、弱相关的部分，只保留最近几条真正有指导意义的经验。**

如果想更明确一点，可以把 `truncate` 写成：

$$
truncate([m_1, m_2, \dots, m_n], k) = [m_{n-k+1}, \dots, m_n]
$$

它表达的是一个很简单的策略：**只保留最近的 $k$ 条经验。**

对初学者来说，一个常见误区是把所有报错、所有历史输出、所有中间推理都塞回 prompt，认为“信息越多越好”。这通常会适得其反。模型真正需要的不是海量日志，而是“下一轮该优先修什么”的短结论。下面这个区分很重要：

| 输入内容 | 是否推荐进入记忆 | 原因 |
|---|---|---|
| “缺少空列表判断” | 推荐 | 明确指出错误类型 |
| “索引越界来自空字符串输入” | 推荐 | 说明触发条件 |
| “返回值类型应为 `list[str]` 而非 `str`” | 推荐 | 直接指向修正方向 |
| 完整 traceback 原文 | 一般不推荐 | 冗长，噪声高 |
| 整段失败代码 + 所有日志 | 一般不推荐 | 会稀释关键信号 |
| “再认真一点” | 不推荐 | 没有可操作性 |

所以，Reflexion 的核心不是“把失败历史保存下来”，而是**把失败历史压缩成下一轮能直接利用的控制信号**。

---

## 核心机制与推导

Reflexion 可以拆成三个角色：Actor、Evaluator、Reflector。

Actor 是执行者，负责生成动作、代码或答案；可以把它理解为**实际干活的 Agent**。Evaluator 是评估器，负责判断 Actor 这轮是否成功；在代码场景里它通常就是测试器。Reflector 是反思器，负责把失败轨迹压缩成可复用经验；它可以是同一个大模型换一个提示模板，也可以是独立模块。

这三个角色对应的是三件不同的事：

| 角色 | 负责的问题 | 典型输入 | 典型输出 |
|---|---|---|---|
| Actor | “这轮怎么做” | 任务描述 + 当前记忆 | 动作、代码、答案 |
| Evaluator | “这轮做得对不对” | Actor 输出 + 环境反馈 | 成功/失败信号 |
| Reflector | “如果错了，下轮该避免什么” | 失败轨迹 + 错误日志 | 一到三句反思 |

每一轮可以形式化写成：

$$
\tau_t \sim \pi(s_t, mem_t)
$$

$$
r_t = Eval(\tau_t)
$$

$$
\phi_t = Reflect(\tau_t, r_t, mem_t)
$$

$$
mem_{t+1} = truncate(mem_t \cup \{\phi_t\})
$$

其中：

| 符号 | 含义 | 直白解释 |
|---|---|---|
| $s_t$ | 当前任务状态 | 这轮开始时系统已知的信息 |
| $\pi$ | Actor 的策略 | 模型基于输入生成答案的方式 |
| $\tau_t$ | 第 $t$ 轮轨迹 | 本轮完整动作、代码或操作记录 |
| $r_t$ | 评估结果 | 成功或失败信号 |
| $\phi_t$ | 反思文本 | 失败后提炼出的短经验 |
| $mem_t$ | 当前记忆 | 下一轮会被带入 prompt 的经验集合 |

这个公式的重点不在数学复杂度，而在控制流很清晰：

1. Actor 根据任务和记忆先做一次。
2. Evaluator 判断结果对不对。
3. 如果失败，Reflector 把“失败原因”压缩成一句或几句短文本。
4. 系统更新记忆，并在下一轮重试。

一个最小流程如下：

| 轮次变量 | 含义 | 例子 |
|---|---|---|
| $\tau_1$ | 第 1 轮轨迹 | 生成了求平均值函数 |
| $r_1$ | 评估结果 | 测试失败，空列表时报错 |
| $\phi_1$ | 反思 | “缺少空列表判断” |
| $mem_2$ | 下一轮记忆 | 只保存这句反思 |
| $\tau_2$ | 第 2 轮轨迹 | 先加 `if not nums: return 0` |

再看一个对新手更直观的例子。任务是实现 `first_char(s)`：返回字符串首字符，空字符串返回空字符串。第一轮模型很可能直接写：

```python
def first_char(s):
    return s[0]
```

这段代码在普通输入下没问题，但遇到 `""` 时会触发索引越界。Evaluator 捕获失败后，Reflector 不需要写长篇分析，只要产出一句高信号反思：

```text
边界条件遗漏：空字符串输入会导致索引越界
```

下一轮 prompt 顶部带上这句话，模型就更可能主动写成：

```python
def first_char(s):
    return "" if s == "" else s[0]
```

这不是数学意义上的梯度下降，因为没有参数更新；但它确实沿着“上轮错误方向的反方向”在修正，因此常被叫作“语义梯度式学习”。

真实工程中，HumanEval 流水线就是这个思路的典型例子。Agent 先根据题目生成代码，再自动生成或执行测试，Evaluator 根据测试结果给出反馈。如果用例暴露出“未处理空输入”“排序方向错误”“遗漏去重”“返回值类型不匹配”之类的问题，Reflector 会把这些失败压缩成短句，写进记忆。下一轮生成代码时，模型最先看到的不是整段长日志，而是这些高密度的修正提示。这样做有两个直接好处：

| 做法 | 影响 |
|---|---|
| 把长日志直接塞回 prompt | 噪声多，模型不一定抓得住关键点 |
| 先压缩成反思再塞回 prompt | 信号更集中，更像“下一轮修复清单” |

所以，Reflexion 的关键机制可以概括成一句话：**先做，再判，再复盘，再带着复盘结果重做。**

---

## 代码实现

工程上，最小可用实现只需要四部分：构造 prompt、生成候选答案、执行 evaluator、在失败时生成 reflection 并更新 memory。

下面给出一个可直接运行的 Python 玩具版本。它不依赖第三方库，也不调用真实大模型，而是用规则模拟“模型第一次犯错，第二次因记忆修正”的过程。重点不是生成质量，而是把 Reflexion 的控制流讲清楚，并确保代码本身可以运行。

```python
from __future__ import annotations

from typing import Callable, List, Tuple


def truncate(mem: List[str], k: int = 3) -> List[str]:
    """Keep only the most recent k reflections."""
    return mem[-k:]


def build_prompt(task: str, mem: List[str]) -> str:
    """Place memory before the task so the actor sees constraints first."""
    if not mem:
        return f"Task:\n{task}"

    memory_block = "\n".join(f"- {item}" for item in mem)
    return f"Past reflections:\n{memory_block}\n\nTask:\n{task}"


def mock_actor_generate(prompt: str, mem: List[str]) -> str:
    """
    Simulate an actor.
    Without relevant memory, it returns a buggy implementation.
    With the reflection about empty strings, it returns the fixed version.
    """
    joined = " ".join(mem)
    if "空字符串" in joined or "边界条件" in joined:
        return (
            "def first_char(s):\n"
            "    if s == '':\n"
            "        return ''\n"
            "    return s[0]\n"
        )

    return (
        "def first_char(s):\n"
        "    return s[0]\n"
    )


def evaluator(code: str) -> Tuple[bool, str]:
    """
    Execute generated code in an isolated namespace and run simple tests.
    Return (success, log).
    """
    scope: dict = {}
    try:
        exec(code, scope)
    except Exception as exc:  # syntax or runtime error during definition
        return False, f"code execution failed: {type(exc).__name__}: {exc}"

    fn: Callable[[str], str] | None = scope.get("first_char")
    if fn is None:
        return False, "missing function: first_char"

    test_cases = [
        ("abc", "a"),
        ("z", "z"),
        ("", ""),
    ]

    for raw, expected in test_cases:
        try:
            result = fn(raw)
        except Exception as exc:
            return False, (
                f"test failed for input {raw!r}: "
                f"{type(exc).__name__}: {exc}"
            )

        if result != expected:
            return False, (
                f"test failed for input {raw!r}: "
                f"expected {expected!r}, got {result!r}"
            )

    return True, "all tests passed"


def reflect(failure_log: str) -> str:
    """
    Compress the failure into a short, reusable reflection.
    Real systems often let an LLM do this step.
    """
    if "IndexError" in failure_log or "input ''" in failure_log:
        return "边界条件遗漏：空字符串输入会导致索引越界"
    if "missing function" in failure_log:
        return "需要先定义题目要求的目标函数"
    return "需要检查边界条件和返回值是否符合题意"


def run_with_reflexion(
    task: str,
    max_trials: int = 3,
) -> Tuple[bool, str, List[str]]:
    """
    Run the full Reflexion loop.
    Return (success, final_code, final_memory).
    """
    memory: List[str] = []
    last_code = ""

    for trial in range(1, max_trials + 1):
        prompt = build_prompt(task, memory)
        code = mock_actor_generate(prompt, memory)
        success, log = evaluator(code)

        print(f"[trial {trial}]")
        print(prompt)
        print("generated code:")
        print(code)
        print("evaluation:", log)
        print("-" * 40)

        last_code = code
        if success:
            return True, last_code, memory

        reflection = reflect(log)
        memory = truncate(memory + [reflection], k=3)

    return False, last_code, memory


if __name__ == "__main__":
    task = "实现 first_char(s)：返回字符串首字符；若为空字符串，返回空字符串。"
    success, final_code, final_memory = run_with_reflexion(task, max_trials=3)

    print("success:", success)
    print("final memory:", final_memory)
    print("final code:")
    print(final_code)

    assert success is True
    assert "if s == ''" in final_code
    assert len(final_memory) <= 3
```

这段代码体现了 Reflexion 的最小闭环：

1. `build_prompt` 把历史反思和当前任务拼起来。
2. `mock_actor_generate` 根据 memory 决定生成“错误版”还是“修正版”代码。
3. `evaluator` 执行代码并跑测试。
4. 如果失败，`reflect` 生成一句短反思。
5. `truncate` 控制 memory 长度。
6. `run_with_reflexion` 负责串起多轮尝试。

如果直接运行，第一轮会失败，因为生成的是：

```python
def first_char(s):
    return s[0]
```

失败后，memory 会新增一条：

```text
边界条件遗漏：空字符串输入会导致索引越界
```

第二轮 Actor 读到这条记忆后，会生成修正版并通过测试。

这段实现有两个关键点值得单独说明。

第一，`build_prompt` 要把 memory 放在任务之前。原因很简单：**反思是约束，任务是目标。** 约束应该先出现，让模型先知道“这次最容易错在哪里”，再去生成答案。若把反思埋在后面，模型未必会优先利用它。

第二，`truncate` 必须由系统显式控制，而不是“希望模型自己忽略旧内容”。在工程系统里，记忆管理是控制逻辑，不应依赖模型临场发挥。一个常见且稳定的策略就是只保留最近 $N$ 条反思：

$$
truncate([m_1, m_2, ..., m_n], N) = [m_{n-N+1}, ..., m_n]
$$

如果把这套思路换成真实大模型调用，流程通常长这样：

| 步骤 | 说明 |
|---|---|
| 1. Actor 生成候选 | 输入是任务描述 + 最近 memory |
| 2. Evaluator 验证结果 | 运行测试、规则检查或环境交互 |
| 3. Reflector 生成反思 | 输入是失败日志、错误样例、当前记忆 |
| 4. Memory manager 截断记忆 | 只保留最近 1 到 3 条 |
| 5. 再次调用 Actor | 带着新记忆进入下一轮 |

如果是网页 Agent，这个结构同样成立。比如让 Agent 自动填写表单并提交订单：

| 模块 | 可能实现 |
|---|---|
| Actor | 调用浏览器工具点击、输入、提交 |
| Evaluator | 检查是否出现成功页、订单号、状态字段 |
| Reflector | 生成“提交前要先勾选协议”“按钮在弹窗里而不是主页面”等反思 |
| Memory | 作为下一轮浏览器操作前的提示 |

所以，代码层面的 Reflexion 并不复杂。复杂的部分不在循环结构，而在两个地方：**Evaluator 是否可靠，Reflector 是否能把失败压缩成有用的话。**

---

## 工程权衡与常见坑

Reflexion 最大的工程价值，在于它通常不需要改模型本身；但它最大的工程风险，也正来自“系统效果高度依赖外围组件质量”。真正落地时，问题往往不出在 Actor，而出在评估、记忆和反思文本本身。

第一大风险是 evaluator 假阳性。假阳性就是**系统把错误结果误判为正确**。一旦错答案被判成成功，本轮就不会触发反思，学习链条直接断掉。论文中讨论过 MBPP 存在约 16.3% 的假阳性，这会明显削弱 Reflexion 的收益，因为系统会误以为“已经学会了”，实际上只是测试没覆盖到。

第二大风险是 memory 过长。记忆太多会带来两类问题：一是上下文膨胀，直接提高成本和延迟；二是经验相互冲突，比如上上轮的经验只适用于某个局部子任务，却还在当前 prompt 里争抢注意力。很多场景下，最近 1 到 3 条高质量反思，比保留十几条模糊经验更稳。

第三大风险是反思质量差。反思如果写成“再认真一点”“可能某处有 bug”“考虑更多情况”，这种句子几乎没有可执行性。有效反思必须尽量覆盖三件事中的至少两件：**错误类型、触发条件、修正方向**。

下面这个表更适合工程排查：

| 常见坑 | 具体表现 | 根因 | 对策 |
|---|---|---|---|
| Evaluator 假阳性 | 错代码被当成成功，无法触发反思 | 测试覆盖不足或验证逻辑过宽松 | 增加边界测试、随机测试、多评估器交叉验证 |
| Evaluator 假阴性 | 对答案被误判为失败，系统反复无效重试 | 判定规则写错或环境不稳定 | 固定环境、减少 flaky tests、记录失败样例复核 |
| Memory 过长 | Prompt 膨胀，模型被旧经验干扰 | 没有限制记忆长度 | 截断到最近 1 到 3 条，按任务类型分桶 |
| 反思过空 | 写成泛泛建议，无法指导修正 | Reflector 缺少模板约束 | 强制模板化，限制长度，要求指出错误原因 |
| 反思过细 | 把一次性日志写成长期经验 | 没区分“噪声”与“可迁移经验” | 只保留可迁移教训，不保留偶发细节 |
| 任务切换过快 | 旧经验误导新任务 | 共享 memory 但任务分布差异大 | 为 memory 加任务标签，跨任务时清空或降权 |
| 迭代成本过高 | 能学会，但每次重试太慢 | evaluator 运行代价太大 | 限制最大轮数，先跑便宜验证，再跑昂贵验证 |

对初学者来说，最容易踩的三个误区通常是下面这些：

| 误区 | 实际问题 |
|---|---|
| “失败日志越完整越好” | 长日志不等于高价值记忆 |
| “记忆越多越聪明” | 过多 memory 会污染 prompt |
| “反思只要有就行” | 没有可操作性的反思几乎等于没有 |

实际部署时，还要关注成本问题。如果 evaluator 本身很慢，比如要启动容器、编译代码、执行端到端测试，Reflexion 的每一轮重试都很贵。这时要给系统设置最大尝试次数，例如 2 到 4 轮；否则“它会逐步学会”并不等于“这件事在生产环境里划算”。

一个简单的成本估算方式是：

$$
Cost \approx N_{trial} \times (C_{actor} + C_{eval} + C_{reflect})
$$

其中：

| 符号 | 含义 |
|---|---|
| $N_{trial}$ | 最大尝试轮数 |
| $C_{actor}$ | 一次生成成本 |
| $C_{eval}$ | 一次评估成本 |
| $C_{reflect}$ | 一次反思生成成本 |

如果 `C_eval` 特别高，优化方向往往不是“写更长的反思”，而是先把 evaluator 拆成两级：先做便宜的快速验证，再做昂贵的完整验证。

---

## 替代方案与适用边界

和 ReAct 相比，Reflexion 不是替换关系，而是增强关系。ReAct 负责让 Agent 分步行动，Reflexion 负责让 Agent 把失败经验显式带入下一轮。没有记忆的 ReAct 更像“多试几次”；有记忆的 Reflexion 更像“试一次，记住为什么错，再带着这条教训去改”。

和 fine-tuning 相比，Reflexion 的最大优势是部署成本低。fine-tuning 需要准备数据、执行训练、管理模型版本；Reflexion 只要求你能构造 prompt，并且拿到稳定反馈信号。但它的上限也更受限于上下文长度、反思质量和 evaluator 可靠性，不能替代真正的参数学习。

下面这个对比更容易落到工程决策上：

| 方案 | 需要训练 | 是否更新模型 | 对 Evaluator 依赖 | 启动成本 | 适合场景 |
|---|---|---|---|---|---|
| ReAct | 否 | 否 | 低到中 | 低 | 一次性推理、多步操作 |
| Reflexion | 否 | 否 | 高 | 低到中 | 可反复试错、可自动验证的任务 |
| Fine-tuning | 是 | 是 | 训练阶段依赖标注数据 | 高 | 长期稳定分布、批量任务优化 |

如果把三者的核心差异压缩成一句话：

| 方法 | 核心思想 |
|---|---|
| ReAct | 让模型边想边做 |
| Reflexion | 让模型做完后记住为什么错 |
| Fine-tuning | 直接把能力写进参数里 |

什么时候不适合用 Reflexion？至少有三类情况。

第一，任务几乎没有可验证反馈。比如开放式写作、品牌文案、诗歌创作，这类任务的“好坏”高度主观，Evaluator 很难稳定地说出“这轮失败在哪里”，那么反思环就难以成立。

第二，任务之间差异极大，旧经验难以迁移。例如上一轮在修 SQL，下一轮在做法律摘要，再下一轮在浏览器自动化。此时共享的 memory 很容易互相污染。

第三，错误主要来自知识缺失，而不是策略失误。比如模型根本不知道某个冷门 API 的参数含义，或者缺少某个专业领域事实，这时“记住上次错在哪”并不能补齐核心知识空洞。

可以用下面这个简表快速判断：

| 问题特征 | 是否适合 Reflexion |
|---|---|
| 能明确判断对错 | 适合 |
| 可多轮重试 | 适合 |
| 错误类型会重复出现 | 适合 |
| 任务完全开放式、无客观标准 | 不太适合 |
| 每轮任务差异很大 | 不太适合 |
| 错误主要来自缺知识而非缺策略 | 收益有限 |

一个实用判断方法是：先在已有 ReAct 流程上做 A/B 测试，对比“原始 ReAct”和“加入最近 2 条反思记忆”的成功率、平均尝试次数、总成本。如果成功率上升，且重复错误显著减少，说明 evaluator 足够可靠、反思有价值；如果成功率不升反降，优先检查以下三点：

1. evaluator 是否存在假阳性或假阴性。
2. memory 是否过长或跨任务污染。
3. 反思是否写成了无操作性的空话。

换句话说，**Reflexion 不是“多加一个 prompt 模板”就会自动生效，它依赖的是一条完整、干净、可重复利用的反馈链。**

---

## 参考资料

为了便于查阅，先给出按用途整理的资料表：

| 来源 | 类型 | 重点贡献 |
|---|---|---|
| Shinn et al., 2023, Reflexion | 论文 | 提出 Actor-Evaluator-Reflector 闭环，定义 verbal reinforcement learning，并报告 HumanEval、MBPP、AlfWorld 等实验结果 |
| ar5iv 论文转写版 | 论文镜像 | 便于快速阅读算法定义、实验设置和公式 |
| Deep Paper 对 Reflexion 的解读 | 技术博客 | 强调 evaluator 质量、false positive、记忆长度控制等工程问题 |
| CSDN 关于 ReAct vs Reflexion 的文章 | 技术博客 | 用更直白的方式解释两者在反馈和记忆上的差异 |
| Reflected Intelligence 相关文章 | 技术博客 | 从“反思式智能”的角度解释语言化经验记忆的直觉 |

如果要按“阅读顺序”来安排，建议这样看：

| 阅读顺序 | 建议资料 | 目的 |
|---|---|---|
| 第 1 步 | 论文摘要和方法部分 | 搞清楚 Reflexion 的定义和整体框架 |
| 第 2 步 | HumanEval / MBPP 实验部分 | 理解它为什么在可验证任务上有效 |
| 第 3 步 | 工程解读博客 | 理解 evaluator 假阳性、memory 截断等落地问题 |
| 第 4 步 | 自己做一个玩具 demo | 把概念转成控制流和代码实现 |

下面给出参考条目：

1. Shinn, N., et al. 2023. *Reflexion: Language Agents with Verbal Reinforcement Learning*.
2. arXiv: https://arxiv.org/abs/2303.11366
3. ar5iv: https://ar5iv.org/html/2303.11366
4. Deep Paper: https://deep-paper.org/en/paper/2303.11366/
5. CSDN 解读: https://blog.csdn.net/weixin_63681863/article/details/152665862
6. Reflected Intelligence: https://reflectedintelligence.com/2025/05/03/reflective-intelligence-in-llms/
