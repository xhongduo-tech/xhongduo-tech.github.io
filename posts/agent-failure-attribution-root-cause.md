## 核心结论

Agent 的失败归因，本质上是回答一个具体问题：这次失败，究竟是错在“选了不该选的工具”、还是“给对工具传错了参数”、还是“推理链已经偏离问题本身”。如果这个问题答不清，后续补丁通常只是“改到能过一部分测试”，而不是修到根因。

对初级工程师最重要的结论只有三条。

第一，失败分析必须沿着执行日志做，而不是沿着主观猜测做。执行日志可以理解为“系统实际做过什么”的记录，通常包含调用链、输入参数、输出结果、错误信息和时间顺序。日志是 Runtime Facts，也就是真实运行事实。只有事实能约束补丁方向。

第二，最稳定的流程不是“看到报错就改代码”，而是“三段式”：复现、定位、验证。复现是先让问题在真实入口上重新出现；定位是顺着 trace，也就是一条完整执行轨迹，找出最早把系统带偏的决策点；验证是确认修复后新轨迹与问题描述一致，且没有引入副作用。

第三，很多失败并不是“代码不会写”，而是“位置找错了”。在自动化修复场景里，Localization 可以直白地理解为“先找到应该改哪里”。相关复盘显示，失败往往集中在定位阶段，尤其是搜索策略不当：系统把注意力放到症状附近，而不是原因上游。对 Agent 来说，找错文件、找错函数、找错调用入口，比写错一行判断更常见。

| 阶段 | 要回答的问题 | 核心证据 | 没做好的后果 |
|---|---|---|---|
| 复现 | 问题是否真的按描述出现 | 真实入口、异常类型、trace | 误修不存在的问题 |
| 定位 | 最早失效点在哪里 | span、参数、上下游调用链 | 补丁落在症状层 |
| 验证 | 修复是否命中根因 | 新旧 trace 对比、测试结果 | 产生“假通过”补丁 |

---

## 问题定义与边界

Agent 失败归因，不是泛泛地说“这个 Agent 不行”，而是把一次失败映射到具体决策点。这里的“决策点”通常分三类。

工具选择错误：本来应该搜索代码、查询数据库或调用测试工具，却调用了格式化、生成补丁或无关工具。白话说，就是“手段选错了”。

参数构造错误：工具本身选对了，但传入的文件路径、函数名、查询条件、输入格式不对。白话说，就是“方向对，但问法错了”。

推理链偏离：前两步都可能没明显报错，但中间假设已经脱离 issue 描述，后面所有动作都围绕一个错误前提展开。白话说，就是“越想越远”。

边界也要说清。不是所有报错都适合做精细归因。如果连复现都不稳定，比如同一脚本有时报错有时不报错，或者 issue 说的是接口 500，但日志里根本没经过目标业务代码，那么这时谈根因没有意义。因为你分析的不是目标问题，而是环境噪声。

所以进入归因前，至少要满足三个条件：

| 检查项 | 合格标准 | 不合格时该做什么 |
|---|---|---|
| 真实入口 | 走的是用户或测试真正会走的路径 | 先修复复现脚本 |
| 问题一致性 | 异常类型、关键信息与 issue 对齐 | 重新确认问题描述 |
| 轨迹可见性 | trace 中能看到项目核心函数与参数 | 补埋点或增强日志 |

一个玩具例子很适合理解这个边界。

假设有一条调用链：`entry -> parse_order -> search_inventory -> format_response`。用户看到的现象是“下单时报库存不足”。如果你只盯着最后页面文案，就可能去改 `format_response`。但如果 trace 显示 `search_inventory` 的输入商品 ID 一直是 `None`，那问题根本不在页面文案，而在更早的参数传递。这里失败归因的目标，不是解释“为什么显示错了”，而是找出“哪个最早的步骤让后续都错了”。

---

## 核心机制与推导

理解 Agent 失败归因，最关键的是“最早破坏点”这个概念。

把一次执行轨迹记成：

$$
\tau = (a_1, a_2, \dots, a_n)
$$

其中每个 $a_i$ 表示一步动作，例如一次搜索、一次工具调用、一次函数执行或一次参数生成。我们希望找到最小的 $k$，使得从这一步开始，系统被带入失败路径。

一种常见做法是反事实分析。反事实可以直白理解为：“如果当时不是这样做，而是换一个合理动作，后面会不会通过？”把第 $k$ 步替换为候选动作 $a'_k$，构造新的轨迹：

$$
R(\tau, k, a'_k) = (a_1, \dots, a_{k-1}, a'_k, a_{k+1}', \dots, a_n')
$$

如果替换后任务通过，或至少后续状态明显更接近正确结果，那么第 $k$ 步就不是普通错误，而是决定性根因。

这里要注意，后面的 $a_{k+1}', \dots, a_n'$ 不一定与原轨迹完全一样，因为前面一步改了，后续状态也会变化。这就是为什么只看最终报错不够，必须看整条轨迹。

为了让归因落到工程上，每一步通常都要带 span 属性。span 可以理解为“某一步操作的观测单元”。常见属性有：

| 属性 | 含义 | 归因作用 |
|---|---|---|
| `tool_id` | 调用了哪个工具 | 判断是否选错工具 |
| `args/input` | 输入参数是什么 | 判断参数是否构造错 |
| `output` | 返回了什么结果 | 判断错误从哪一步开始暴露 |
| `confidence` | 模型对当前选择的置信度 | 判断是否存在高置信误判 |
| `tokens/time` | 消耗了多少资源 | 判断是否陷入无效循环 |

继续看前面的玩具例子。

轨迹是：`entry -> parse -> search_helper -> format`

日志显示：
- `parse` 成功解析请求
- `search_helper` 收到的 `query=None`
- `format` 只是把空结果包装成“未命中”

如果你只看最后一步，会以为 `format` 把空值处理错了。但做反事实分析时，把 `search_helper(query=None)` 替换成 `search_helper(query="sku_123")`，后续整个链路恢复正常。这时根因就可以归到参数构造或上游解析，而不是格式化函数。

真实工程里的例子更复杂。以 SWE-bench 失败样本为代表，很多 Agent 在收到 issue 后，会先搜索“名字最像”的文件，然后围绕这个文件生成补丁。问题是 issue 描述里的关键词，常常对应的是症状层，而不是根因层。比如错误表现出现在 `formatter.py`，但真正原因是 `db_query.py` 返回了空字段，或者 `resolver.py` 没处理某种配置组合。相关复盘里，定位阶段失败占比很高，本质就是搜索策略把注意力放错了位置。

所以推导出的工程原则是：补丁必须从 trace 逆推出来，而不是从错误文本顺推出来。顺推容易停在表面，逆推更容易找到最早破坏点。

---

## 代码实现

下面给一个最小可运行的 Python 版本，模拟“从轨迹里判断失败归因”。这个例子不依赖外部库，重点是看思路。

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Span:
    name: str
    tool_id: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    ok: bool
    confidence: float


def classify_failure(trace: List[Span]) -> Optional[str]:
    """
    返回:
    - tool_selection_error
    - parameter_construction_error
    - reasoning_drift
    - None
    """
    for span in trace:
        if not span.ok:
            if span.tool_id == "formatter" and "search" in span.name:
                return "tool_selection_error"
            if any(v is None for v in span.input.values()):
                return "parameter_construction_error"
            return "reasoning_drift"
    return None


def earliest_breakpoint(trace: List[Span]) -> Optional[int]:
    for i, span in enumerate(trace):
        if not span.ok:
            return i
    return None


toy_trace = [
    Span("entry", "router", {"path": "/order"}, {"accepted": True}, True, 0.99),
    Span("parse", "parser", {"body": {"sku": "sku_123"}}, {"query": None}, True, 0.82),
    Span("search_helper", "search", {"query": None}, {"rows": []}, False, 0.76),
    Span("format", "formatter", {"rows": []}, {"message": "not found"}, True, 0.91),
]

assert classify_failure(toy_trace) == "parameter_construction_error"
assert earliest_breakpoint(toy_trace) == 2

fixed_trace = [
    Span("entry", "router", {"path": "/order"}, {"accepted": True}, True, 0.99),
    Span("parse", "parser", {"body": {"sku": "sku_123"}}, {"query": "sku_123"}, True, 0.82),
    Span("search_helper", "search", {"query": "sku_123"}, {"rows": [{"stock": 8}]}, True, 0.76),
    Span("format", "formatter", {"rows": [{"stock": 8}]}, {"message": "ok"}, True, 0.91),
]

assert classify_failure(fixed_trace) is None
assert earliest_breakpoint(fixed_trace) is None
```

这个实现很简化，但已经体现了三件事。

第一，日志要结构化。结构化的意思是“每一步都有固定字段”，这样才能自动分析，而不是靠人肉翻长文本。

第二，归因优先找最早失败点，而不是最后报错点。最后报错点通常只是症状出口。

第三，修复后要重新跑新轨迹，对比旧轨迹。没有重跑验证，归因就没有闭环。

如果要更接近真实工程，可以把这个流程扩展成下面这样：

| 模块 | 作用 | 示例输出 |
|---|---|---|
| tracer | 记录 span 与上下文 | trace_id、parent_id、args |
| trace query | 查询链路证据 | callers、downstream、args |
| hypothesis card | 记录假设与证据 | 支持/反驳/待验证 |
| validator | 修复后重放 | 新旧 trace 差异 |

一个真实工程例子是代码修复 Agent 处理 SWE-bench issue。它先运行复现脚本，拿到异常和 trace；然后查看 issue 相关入口经过了哪些文件、哪些函数、哪些参数；如果发现搜索阶段持续把注意力放到“出错位置附近”而不是“异常来源上游”，就调整搜索策略或加约束；最后再运行测试和新 trace，确认补丁没有只掩盖报错。这个流程里，日志不是附属品，而是补丁决策的输入。

---

## 工程权衡与常见坑

工程上最常见的误区，不是“不会加日志”，而是“加了日志却不用来约束决策”。

第一个坑是信息环。信息环可以理解为“反复查同一批日志，却没有新增证据”。例如反复运行同一个复现脚本、重复查询同一调用链、重复让 Agent 总结同一段错误。结果是 token 和时间一直消耗，但定位没有前进。解决方法是加 anti-loop guard，也就是反循环保护：相同查询超过阈值就强制换方向，或者要求每次查询必须回答一个新的问题。

第二个坑是焦点漂移。焦点漂移就是 trace 明明指向 `search_helper`，补丁却改到了 `formatter` 或文案层。因为症状层通常更容易改，也更容易让局部测试通过。解决办法是做 focus alignment，即焦点对齐：限制补丁主要发生在 trace 高亮出的函数、参数构造点或其直接上游。

第三个坑是把“修复报错”当成“修复根因”。例如给空值补默认值，可以让程序不崩，但如果空值本来就不该出现，这个补丁只是在掩盖问题。真正需要验证的是：修复后，新 trace 是否回到 issue 描述中的正确路径。

| 常见坑 | 典型触发条件 | 后果 | 对应 guard |
|---|---|---|---|
| 信息环 | 重复跑同一脚本、查同一 trace | 分析停滞 | anti-loop guard |
| 焦点漂移 | 补丁落到症状层 | 假通过 | focus alignment |
| 掩盖式修复 | 给异常加兜底值 | 根因保留 | 新旧 trace 对比 |
| 错误复现 | 入口不真实、环境不一致 | 全程分析无效 | reproduce gate |

再看一个更贴近工程的例子。假设某 Agent 在大量失败样本中发现，70% 的失败都停在代码定位阶段。它的搜索模块给 `SearchHelper` 较高权重，于是总优先查“名字匹配度高”的文件。结果 issue 里提到 `format`，系统就一直围绕 `format.py` 生成补丁。但 trace 进一步显示，`format.py` 接到的输入早已是空对象，真正的异常来自更上游的 `db/api` 查询。此时正确的动作不是继续优化格式化逻辑，而是下调搜索偏好，沿调用链往上追 `query` 从哪里变成了空值。这个例子说明，归因系统不只是“解释失败”，还会反过来改进 Agent 的搜索策略。

---

## 替代方案与适用边界

日志驱动的根因分析不是唯一方案，但它是最稳的基础方案。不同场景下，可以选择不同归因路径。

| 方案 | 适用场景 | 成本 | 对 trace 质量要求 | 输出粒度 |
|---|---|---|---|---|
| Runtime Facts 三段式 | 单 Agent 修复、需要高可信补丁 | 中等 | 高 | 可定位到函数/参数 |
| 反事实回放 | 需要判定最早失效步骤 | 较高 | 很高 | 可定位到具体 step |
| Expert-Executor 协作 | trace 不完整、需要人工纠偏 | 中高 | 中等 | 可定位到策略层 |
| 统计式失败分类 | 大规模样本复盘 | 低到中 | 中等 | 多为阶段级结论 |

如果你有清晰 trace，并且能重放任务，优先用 Runtime Facts 三段式。原因很简单：它最容易建立“证据 -> 假设 -> 验证”的闭环。

如果系统是多 Agent、多工具、多轮协作，单靠人工看日志成本太高，可以引入反事实回放或像 AgenTracer 这样的框架。它的优势是能自动判断“如果第 k 步换一种做法，任务是否会恢复”，适合复杂链路归因。

如果 trace 质量一般，但你又必须尽快推进，Expert-Executor 协作更实用。Expert 可以理解为“专门负责纠偏的监督者”，它不直接执行任务，而是审查执行者的定位是否跑偏。这类方法牺牲了一部分自动化，但在复杂项目里常常更稳。

它们的共同边界也很明确：没有可靠执行证据时，归因质量不会高。没有真实入口、没有关键参数、没有上下游调用链，再强的分析模型也只能做猜测。对初学者来说，最该建立的不是“更会猜”，而是“先让系统留下足够证据”。

---

## 参考资料

- Syncause, Achieving an 83.4% Fix Rate on SWE-bench Verified with Runtime Facts
- Medium / Dev 系列文章：How We Hit 83.4% on SWE-bench Verified
- AgenTracer framework 相关介绍与论文整理
- AgentMark 文档中关于 traces、logs、span attributes 的说明
- Emergent Mind 关于自动化 issue solving 失败分类与 localization 研究
- ApX 关于 agent action sequence debugging 的方法论整理
