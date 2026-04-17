## 核心结论

代码执行反馈的自动修复循环，本质上是把“程序报错后人工再改一次”的动作，封装成一个可度量、可终止、可审计的闭环。闭环的起点是一次失败执行，终点只有两个：代码通过验证，或触发停止条件。

对初级工程师最重要的结论有三条。

第一，错误分类决定修复策略。语法错误指代码连解释器或编译器都过不去，通常最容易修；运行时错误指代码能启动，但执行到某一步崩掉；逻辑错误指代码能跑完，却给出错误结果。三类错误对应的信息来源、修复手段、验证方式都不同，不能统一理解为“上次错了，那就再试一次”。

第二，自动重试有效，但收益递减。若第 $i$ 轮修复成功概率为 $p_i$，最多尝试 $R$ 次时，总成功率是：

$$
S_R = 1 - \prod_{i=1}^{R}(1-p_i)
$$

若每轮成功率近似相同，即 $p_i \approx p$，则化简为：

$$
S_R = 1-(1-p)^R
$$

新手先记住最简单的例子即可。若单轮修复成功率为 $0.3$，三次内成功率不是 $0.9$，而是：

$$
S_3 = 1-(1-0.3)^3 = 0.657
$$

原因很直接：第二次修复只会发生在第一次失败之后，第三次修复只会发生在前两次都失败之后，所以不能把每轮成功率直接相加。

第三，重试上限必须和成本一起设计。NUS 的一项程序修复研究里，GPT-4 的 Repair@1 为 74.9%，Repair@3 为 86.1%，Repair@5 为 88.5%。这说明前几次重试通常有明显收益，但后续提升会迅速变小。真实工程里如果没有总预算和断路器，循环会从“提高成功率”变成“烧钱机器”。TrackAI 记录过一个金融交易 Agent，因为循环检测只统计错误类型、没统计总迭代次数，3 天里打了 47,000 次失败调用，额外消耗约 12,000 美元。

把这三条合起来，结论就是：自动修复循环不是“让模型一直改”，而是“让系统在有限预算内，优先修复最可能修好的错误，并且在看不到进展时立即停止”。

---

## 问题定义与边界

自动修复循环的目标，不是“让模型一直改到对”，而是“在有限预算内，把可修的失败变成成功，把不可修的失败尽快暴露出来”。

一个最小定义如下：

1. Agent 生成代码并执行。
2. 执行器返回结果：成功、语法错误、运行时错误、逻辑失败。
3. 修复器基于错误类型生成下一版代码。
4. 再执行，再判断。
5. 直到成功，或达到停止条件。

这五步看上去简单，但工程上真正困难的部分不在“会不会重试”，而在“什么时候该继续，什么时候必须停”。

先看三类错误的边界。

| 错误类型 | 白话解释 | 常见信号 | 常见处理策略 | 典型跳出条件 |
| --- | --- | --- | --- | --- |
| 语法错误 | 连“语法检查”都过不去 | `SyntaxError`、括号不闭合、缩进错误 | 约束输出格式、局部补丁、静态检查后重试 | 连续 2 次相同语法错误 |
| 运行时错误 | 跑到中途崩了 | `TypeError`、`NameError`、空指针、越界、依赖缺失 | 补依赖、空值保护、边界检查、修正 API 调用 | 同一堆栈重复出现、外部资源不可用 |
| 逻辑错误 | 能跑完但答案错 | 断言失败、测试失败、结果偏差 | 增加测试、回放失败样例、要求解释推理 | 缺少可验证 oracle，或同一错误用例反复失败 |

对新手来说，可以先用一句更直白的话理解：先判断是“代码写不通”，还是“运行时炸了”，还是“算出来了但算错了”。前两类往往可以靠局部修补解决，最后一类必须依赖更强的验证。

因此，自动修复循环至少要有四个硬边界：

| 边界 | 作用 | 典型实现 |
| --- | --- | --- |
| 最大重试次数 `max_retries` | 防止无限循环 | 每类错误设置不同上限 |
| token 或调用预算 `max_tokens/max_cost` | 防止成本失控 | 记录累计调用次数、累计 token、累计费用 |
| 语义重复检测 | 防止换个措辞重复同一路径 | 比较错误签名、补丁摘要、状态哈希 |
| 断路器 `circuit_breaker` | 在外部服务异常、连续失败时强制停止 | 同类错误连续超过阈值直接熔断 |

这里要特别区分两件事。

第一，失败不等于应该继续。  
第二，继续不等于会更接近正确。

没有验证器的逻辑错误，不应长期留在循环内。因为它会把“模型看起来像修好了”和“系统真的修好了”混在一起。一个很常见的误判是：代码已经不报错了，于是系统把这轮视为成功；但业务结果仍然是错的。这类“静默失败”往往比直接报错更危险。

从系统边界看，自动修复循环适合处理的是“失败可观测、结果可验证、成本可控制”的任务。如果一个任务连“怎样算修好”都说不清，那它本质上就不适合交给纯自动循环。

---

## 核心机制与推导

一轮自动修复循环通常包含四步：检测、生成、测试、记录。

检测阶段负责分类错误。分类不是为了好看，而是为了决定“下一轮提示词长什么样、能不能继续、预算还剩多少”。生成阶段只做最小修改，尽量避免整段重写，因为大范围重写会破坏已经正确的部分。测试阶段用语法检查、单元测试、回归样例或业务断言判断是否真的进步。记录阶段把失败轨迹写入 `error_history`，用于重复路径检测。

可以用三个玩具例子理解这四步。

### 例 1：语法错误

```python
print("hello"
```

解释器直接返回 `SyntaxError`。这类错误的特点是定位清晰，修复范围小。系统通常不需要重规划任务，只需要补齐括号、引号或缩进即可。

### 例 2：运行时错误

```python
items = None
print(len(items))
```

代码能通过语法检查，但运行时报 `TypeError`。这时问题不在语法，而在状态。修复器应该考虑空值保护、默认值、输入校验，而不是继续做语法模板替换。

### 例 3：逻辑错误

```python
def is_even(n):
    return n % 2 == 1
```

代码可以运行，但测试 `is_even(4) == True` 会失败。这里真正错的是判断条件。若系统只根据“上轮失败”机械续写，很可能一直在变量名、格式、注释上打转，而不是修改逻辑本身。

这三个例子揭示了一个关键事实：错误文本只是线索，不是修复策略本身。  
真正的修复策略取决于“错误来自哪一层”。

| 层级 | 典型问题 | 主要证据 | 优先修复方式 |
| --- | --- | --- | --- |
| 语法层 | 括号、缩进、关键字拼写 | 编译器/解释器报错 | 局部补丁、格式约束 |
| 执行层 | 空值、类型、越界、依赖、权限 | 堆栈、返回码、环境日志 | 补充上下文、修改调用方式 |
| 语义层 | 结果不符合需求 | 断言、测试、业务指标 | 扩展验证、回放样例、推理解释 |

Repair@k 的意义，就是在有限轮数里衡量“覆盖到多少原始失败”。如果第 $i$ 轮成功概率是 $p_i$，那么：

$$
S_R = 1-\prod_{i=1}^{R}(1-p_i)
$$

如果每轮独立且概率近似相同，得到：

$$
S_R = 1-(1-p)^R
$$

这个公式告诉我们两件事。

第一，重试确实能提高总体成功率。  
第二，提升不是线性的，而是边际递减的。

例如 $p=0.5$ 时：

| 最大轮数 $R$ | 总成功率 $S_R$ | 相比前一轮的增量 |
| --- | --- | --- |
| 1 | 50.00% | 50.00% |
| 2 | 75.00% | 25.00% |
| 3 | 87.50% | 12.50% |
| 4 | 93.75% | 6.25% |
| 5 | 96.88% | 3.13% |

从 1 次到 2 次，提升 25 个百分点；从 4 次到 5 次，只提升 3.13 个百分点。  
所以系统设计不是“能重试几次就重试几次”，而是“多一次重试带来的成功率，值不值得它的 token、延迟和风险”。

如果进一步把成本纳入考虑，可以写成一个更工程化的判断式：

$$
\text{继续重试} \iff \Delta S_R \times V > C_R
$$

其中：

- $\Delta S_R$ 表示再多尝试一轮带来的成功率增量
- $V$ 表示一次成功的业务价值
- $C_R$ 表示第 $R+1$ 轮的边际成本，包括 token、延迟、外部调用和潜在风险

这个式子不是严格的财务模型，但它足够说明一个实践原则：重试不是免费动作，应该按收益决定。

如果采用题设给出的一组经验值作为初始先验，例如语法错误 95%、运行时错误 78%、逻辑错误 42%，就能直接得到一个实践策略：

| 错误类型 | 经验成功率 | 建议链路 | 原因 |
| --- | --- | --- | --- |
| 语法错误 | 95% | 短链路、低上下文、快速重试 | 错误定位直接，修改面小 |
| 运行时错误 | 78% | 保留堆栈、输入样例、环境信息 | 需要知道代码在哪个状态下失败 |
| 逻辑错误 | 42% | 更强验证、更详细测试、必要时人工介入 | 报错信息不足以定义“正确答案” |

因此，自动修复循环的核心机制不是“失败后继续生成”，而是“基于错误层级，选择最小代价的下一步验证与修复”。

---

## 代码实现

下面给出一个最小可运行的 Python 示例。它不依赖大模型，只模拟“分类后采用不同重试策略”的核心流程。示例包含四个模块：执行、分类、修复、验证，并补上了预算、重复路径检测和可直接运行的演示入口。

```python
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Callable


TEST_CASES: dict[str, Callable[[dict], bool]] = {
    "logic_case": lambda ns: ns["is_even"](4) is True and ns["is_even"](5) is False,
}


@dataclass
class RepairStats:
    retries: int = 0
    token_budget: int = 1200
    error_history: list[str] = field(default_factory=list)
    stop_reason: str = ""
    last_error_type: str = ""


def classify_error(result: str) -> str:
    if result.startswith("SyntaxError"):
        return "syntax"
    if result.startswith(("TypeError", "NameError", "RuntimeError", "ValueError")):
        return "runtime"
    if result.startswith("ASSERT_FAIL"):
        return "logic"
    if result == "PASS":
        return "pass"
    return "unknown"


def max_retries_for(error_type: str) -> int:
    limits = {
        "syntax": 2,
        "runtime": 3,
        "logic": 2,
        "unknown": 0,
        "pass": 0,
    }
    return limits[error_type]


def execute(code: str, scenario: str) -> str:
    if scenario == "syntax_case":
        try:
            ast.parse(code)
            return "PASS"
        except SyntaxError as exc:
            return f"SyntaxError: {exc.msg}"

    namespace: dict = {}
    try:
        exec(code, namespace, namespace)
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"

    if scenario == "runtime_case":
        return "PASS"

    if scenario == "logic_case":
        ok = TEST_CASES["logic_case"](namespace)
        return "PASS" if ok else "ASSERT_FAIL: is_even failed tests"

    return "RuntimeError: unknown scenario"


def repair(code: str, error_type: str, scenario: str) -> str:
    if scenario == "syntax_case" and error_type == "syntax":
        return "print('ok')"

    if scenario == "runtime_case" and error_type == "runtime":
        return "x = []\nprint(len(x))"

    if scenario == "logic_case" and error_type == "logic":
        return "def is_even(n):\n    return n % 2 == 0"

    return code


def auto_repair(code: str, scenario: str) -> tuple[str, RepairStats]:
    stats = RepairStats()

    while stats.token_budget > 0:
        result = execute(code, scenario)
        error_type = classify_error(result)
        stats.last_error_type = error_type

        if result == "PASS":
            stats.stop_reason = "verified_success"
            return code, stats

        signature = f"{error_type}:{result}"
        stats.error_history.append(signature)

        if len(stats.error_history) >= 2 and stats.error_history[-1] == stats.error_history[-2]:
            stats.stop_reason = "repeated_error_signature"
            return code, stats

        allowed = max_retries_for(error_type)
        if stats.retries >= allowed:
            stats.stop_reason = "retry_limit_reached"
            return code, stats

        if stats.token_budget < 200:
            stats.stop_reason = "token_budget_exhausted"
            return code, stats

        stats.retries += 1
        stats.token_budget -= 200
        code = repair(code, error_type, scenario)

    stats.stop_reason = "token_budget_exhausted"
    return code, stats


def demo() -> None:
    cases = [
        ("syntax_case", "print('ok'"),
        ("runtime_case", "x = None\nprint(len(x))"),
        ("logic_case", "def is_even(n):\n    return n % 2 == 1"),
    ]

    for scenario, bad_code in cases:
        fixed_code, stats = auto_repair(bad_code, scenario)
        final_result = execute(fixed_code, scenario)

        print(f"[{scenario}]")
        print("final_result:", final_result)
        print("retries:", stats.retries)
        print("stop_reason:", stats.stop_reason)
        print("fixed_code:")
        print(fixed_code)
        print("-" * 40)

        assert final_result == "PASS"


if __name__ == "__main__":
    demo()
```

这段代码可以直接运行，输出三类错误各自的修复结果，并通过 `assert` 验证最终状态确实为 `PASS`。

如果从新手角度拆开看，最重要的是下面四个函数。

| 函数 | 作用 | 关键点 |
| --- | --- | --- |
| `execute()` | 执行代码并返回结果 | 不是直接返回布尔值，而是返回可分类的错误文本 |
| `classify_error()` | 把执行结果映射成错误类型 | 错误分类是后续策略选择的入口 |
| `repair()` | 按错误类型生成下一版代码 | 示例里用规则替代大模型，真实系统里可接 LLM |
| `auto_repair()` | 组织整个循环 | 负责预算控制、停止条件、错误历史记录 |

这个例子故意保留了三个关键工程点。

第一，`classify_error()` 把错误转换成策略入口。  
第二，`max_retries_for()` 让不同错误类型有不同重试上限。  
第三，`error_history` 用于发现“系统没有前进”。

再看一个更接近真实工程的场景。假设一个数据分析 Agent 生成 SQL 后执行失败：

1. 若数据库返回 `syntax error near FROM`，可直接让模型最小改写 SQL。
2. 若返回 `relation not found`，问题可能是表名、schema 或权限，应补充元数据查询，而不是只改 SQL 语法。
3. 若 SQL 执行成功但报表总额错了，这是逻辑错误，必须回放对账样例，而不是继续盲改 SQL 文本。

因此工程实现通常会拆成四个模块：`executor`、`error_classifier`、`repair_generator`、`verifier/logger`。其中真正决定系统质量的，不是“会不会重试”，而是“验证器是否足够强，能不能证明这次修复比上次更接近正确”。

如果把上面的示例再提升一步，真实系统通常还会补三类能力：

| 能力 | 为什么需要 | 典型做法 |
| --- | --- | --- |
| 状态快照 | 仅看错误文本往往不够 | 保存输入、环境变量、依赖版本、失败样例 |
| 补丁对比 | 防止每轮大改 | 记录 diff 摘要，只允许局部修改 |
| 可审计日志 | 复盘失败原因 | 保存每轮提示、补丁、测试结果、停止原因 |

对新手来说，最值得记住的一句是：自动修复循环不是一个“生成器”，而是一个“带验证的状态机”。

---

## 工程权衡与常见坑

自动修复循环最常见的问题，不是修不好，而是“以为自己在修，实际在重复”。

| 常见坑 | 现象 | 根因 | 对策 |
| --- | --- | --- | --- |
| 不做错误分类 | 所有失败都走同一提示模板 | 把不同层级错误混成一种问题 | 先分语法、运行时、逻辑三类 |
| 没有总预算 | 次数虽少，但每次上下文越来越长 | 只限制轮数，不限制成本 | 同时限制次数、token、总成本 |
| 只看最近一次错误 | 相同路径反复重放 | 缺少历史轨迹比较 | 记录 `error_history` 和状态哈希 |
| 逻辑错误无验证 | 代码能跑但结果仍错 | 把“可运行”误当“正确” | 增强测试集或人工审核 |
| 大范围重写 | 修好一个点，破坏三个已正确点 | 没有最小补丁约束 | 优先局部补丁 |
| 缺少断路器 | 外部 API 挂掉时无限重试 | 把基础设施故障当成代码问题 | 识别基础设施故障并快速失败 |

对新手尤其要强调一个坑：只盯语法错误，很容易错过真正的问题。比如测试一直失败，模型每次都改缩进、改变量名、改注释，看起来“很努力”，但业务逻辑完全没动。这样的循环只会消耗上下文和时间。

还有一个常见误区，是把“重试次数”当成唯一控制杆。实际上至少有四个维度要一起管：

| 维度 | 失控表现 | 为什么单看轮数不够 |
| --- | --- | --- |
| 次数 | 无限循环 | 少量重试也可能很贵 |
| token | 上下文越来越长 | 3 次超长调用可能比 10 次短调用更贵 |
| 时间 | 单次修复等待过长 | 实时系统对延迟敏感 |
| 风险 | 错误结果进入生产 | 即使调用便宜，也可能业务代价高 |

成本问题也不能抽象化。金融 Agent 的 47,000 次失败调用案例说明，循环本身不是安全机制；没有上限的循环只是把失败自动化。系统应当至少回答三个问题：

1. 这轮还有没有新增信息？
2. 再试一次的成功概率是否高于阈值？
3. 失败的业务代价是否允许继续？

从 ROI 看，Repair@k 只在前几轮有明显收益。若你的业务是低价值、低容错的小任务，常见策略反而是 `Repair@1` 或 `Repair@2` 后直接断路；若你的业务是高价值但可验证任务，比如单元测试驱动的代码修复，可接受更高的 `k`，因为验证成本低、成功价值高。

一个实用的工程经验是，把“是否继续”写成显式规则，而不是交给模型自由发挥。例如：

| 条件 | 建议动作 |
| --- | --- |
| 同一错误签名连续 2 次出现 | 立即停止 |
| 外部依赖错误占主导 | 触发断路器，不再修代码 |
| 通过率提升但未全过 | 允许继续 1 轮 |
| 新补丁导致通过测试数下降 | 回滚上一版本并停止 |
| 缺少 oracle | 升级为人工审批 |

这些规则的价值在于：它们把“感觉应该停了”变成了“系统知道为什么停”。

---

## 替代方案与适用边界

自动修复循环不是唯一方案，它适合“错误可观测、结果可验证、失败可承受”的任务。

| 条件 | 首选策略 | 原因 |
| --- | --- | --- |
| 语法或简单运行时错误，占比高 | 错误分类 + 短重试 | 修复路径短，收益高 |
| 有可靠测试或 oracle | 自动修复循环 + 验证器 | 能客观判断是否变好 |
| 逻辑错误多，测试稀缺 | 规划器/验证器升级，减少盲修 | 仅靠报错文本不够 |
| 预算很低 | `Repair@1` + 断路器 | 控制成本优先 |
| 高风险业务 | 人工审批或 HITL | 防止错误自动放大 |

一个适合新手理解的切换路径是：

规则式速修 -> 仍失败 -> 引入更强验证或规划 -> 仍失败 -> 人工回调

这里的核心不是“模型不够强”，而是任务信息不够。

语法错误通常只需要局部上下文。  
运行时错误需要堆栈、环境、依赖信息。  
逻辑错误往往需要需求、测试、业务规则，甚至需要知道“为什么这个答案才是对的”。

所以替代方案主要有两类。

第一类是更强的验证层。比如增加单元测试、属性测试、对拍数据、静态分析器。这类方法不一定让模型更聪明，但会让循环更快停止在正确方向上。

| 验证方式 | 适合什么问题 | 优点 | 局限 |
| --- | --- | --- | --- |
| 单元测试 | 函数级逻辑错误 | 精确、易自动化 | 覆盖范围有限 |
| 属性测试 | 边界条件、随机输入 | 能发现隐蔽反例 | 需要抽象出性质 |
| 静态分析 | 语法、类型、未定义引用 | 速度快、成本低 | 很难验证业务正确性 |
| 回归样例 | 历史已知问题 | 贴近真实场景 | 样例维护成本高 |
| 人工审核 | 高风险业务 | 可处理复杂语义 | 成本高、速度慢 |

第二类是更高层的反馈层。比如把“修代码”升级为“先分析失败计划，再决定是否继续调用工具”，或者增加人工复核。它适合逻辑错误比例高、单次失败代价大的场景。

从系统设计上看，可以把替代方案理解为三个层级：

| 层级 | 目标 | 典型适用场景 |
| --- | --- | --- |
| 快速修补层 | 低成本修复明显错误 | 语法错误、简单运行时错误 |
| 强验证层 | 判断是否真的变好 | 有测试、有样例、有业务约束 |
| 决策控制层 | 决定要不要继续自动化 | 高风险、高成本、低可验证任务 |

边界也很明确：如果任务结果无法验证，或者错误会直接影响资金、权限、医疗等高风险动作，那么自动修复循环只能做建议层，不能直接做执行层。

换句话说，自动修复循环适合当“自动调试器”，不适合在没有护栏的情况下直接充当“自动决策者”。

---

## 参考资料

| 来源 | 主要信息点 | 用处 |
| --- | --- | --- |
| [ADLI: AI Agents That Learn](https://adli.dev/) | 观察-学习-注入的运行反馈闭环 | 用来说明“失败后学习并调整策略”的总体框架 |
| [Improving the Coverage of GPT for Automated Feedback on High School Programming Assignments](https://aicet.comp.nus.edu.sg/wp-content/uploads/2024/10/gaied23-gpt.pdf) | GPT-4 Repair@1=74.9%，Repair@3=86.1%，Repair@5=88.5%；多轮交互可提升修复覆盖率 | 用来支撑 Repair@k、边际收益递减、验证 oracle 的重要性 |
| [TrackAI: Loop Detection & Breaking](https://trackai.dev/tracks/observability/debugging-tracing/loop-detection/) | 金融服务 Agent 因循环检测缺陷导致 47,000 次失败调用、约 12,000 美元成本 | 用来说明预算上限、断路器、循环检测不是可选项 |
| [Chronos / Autonomous Debugging Loop](https://chronos.so/the-autonomous-debugging-loop) | 不同 bug 类型的成功率与迭代成本差异，逻辑类问题更依赖多轮调试和更强检索 | 用来说明不同错误类型不能共享同一重试策略 |
| [Advancements in automated program repair: a comprehensive review](https://link.springer.com/article/10.1007/s10115-025-02383-9) | 综述中提到语法类错误通常更容易修复，成功率普遍高于复杂语义错误 | 用来辅助解释“语法最容易、逻辑最难”的普遍规律 |

这些资料合起来，支撑的是同一个判断：代码执行反馈的自动修复循环并不是单纯的“多试几次”，而是一套依赖错误分类、验证器强度、预算控制和停止规则的系统设计问题。只要这四项中缺一项，循环就可能从增益机制退化成成本放大器。
