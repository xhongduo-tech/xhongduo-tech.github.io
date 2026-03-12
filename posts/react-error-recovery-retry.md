## 核心结论

ReAct Agent 的错误恢复，不是“失败了再试一次”这么简单，而是一套分层处理链路：先反思失败原因，再决定是否回溯重规划，最后才进入受控重试或降级。这里的 ReAct，指“边推理边行动”的代理模式；它会一边生成思考步骤，一边调用搜索、浏览器、数据库等工具。真正让它可落地的，不是会不会调用工具，而是工具失败时能不能稳定收场。

更准确地说，生产级 ReAct 的恢复流程通常可抽象为三步：

| 阶段 | 触发条件 | 输出 | 目标 |
| --- | --- | --- | --- |
| 自省 | 工具报错、结果为空、结果与约束冲突 | 一条失败总结或经验记忆 | 不重复犯同一种错 |
| 回溯 | 当前计划依赖已失效 | 新的子任务拆分或局部重做计划 | 只修复受影响链路 |
| 降级 | 超过重试预算、工具长期不稳定 | 切换备用模型、备用工具或人工兜底 | 防止系统卡死和成本失控 |

“自省”首次出现时，可以理解成“把这次失败写成一条以后可复用的经验”；“回溯重规划”可以理解成“不是整盘重来，而是退回到出错节点重新拆任务”；“降级”可以理解成“放弃最理想路径，改走更稳但可能更慢或更笨的路径”。

一个常被引用的结果是，Reflexion 这类“失败后写反思并带入下一轮”的方法，在 AlfWorld 任务上把成功率从 75% 提升到 97%。这个数字本身不是说“只要加一段反思 prompt 就够了”，而是说明：当代理能把失败经验转成下一轮决策输入时，错误恢复会从盲目重试变成有方向的修正。

对工程系统来说，核心原则只有一句：恢复逻辑必须放在 agent runtime，也就是代理运行时这一层，而不是依赖模型临场发挥。否则一个不稳定的 `browser.get` 就可能触发几十次重复调用，把一次普通查询拖成高成本事故。

---

## 问题定义与边界

本文讨论的“错误恢复”，专指 ReAct Agent 在执行任务链条时，遇到工具调用失败、工具结果无效、或中间状态被污染后的处理机制。这里的“工具”包括 API、浏览器、数据库查询、代码执行器、检索系统等；“无效结果”不只是不返回，也包括返回了但不能支撑下一步。

边界先说清楚。不是所有错误都值得重试，至少要分三类：

| 错误等级 | 定义 | 典型例子 | 恢复动作 | 待机/降级路径 |
| --- | --- | --- | --- | --- |
| `RECOVERABLE` | 临时性错误，可短期重试 | 超时、429、网络抖动 | 指数退避后重试 | 保持同计划 |
| `DEGRADABLE` | 主路径不稳，但可换路继续 | 主模型太慢、页面结构异常、次要工具不可用 | 重试有限次后切备用方案 | 备用模型、缓存、人工校验 |
| `FATAL` | 继续执行没有意义或有风险 | 权限不足、输入缺字段、业务约束冲突 | 立即停止并上报 | 人工介入或直接失败 |

“指数退避”首次出现时，可以理解成“每次失败后等待时间按倍数增长，避免瞬时打爆服务”。它常写成：

$$
delay = \min(base\_delay \times 2^{attempt},\ max\_delay)
$$

这里 $attempt$ 是第几次重试，$base\_delay$ 是初始等待时间，$max\_delay$ 是等待上限。

玩具例子可以用在线购物说明。假设代理要完成“下单并支付”：
1. 浏览器打开商品页成功。
2. 加入购物车成功。
3. 支付接口超时。

这时系统不该无脑重试 100 次。更合理的边界是：
- 如果只是一次网络超时，属于 `RECOVERABLE`，可以按 1 秒、2 秒、4 秒再试。
- 如果连续失败且风控提示支付通道异常，属于 `DEGRADABLE`，应切换备用支付通道或转人工确认。
- 如果用户余额不足或订单已关闭，属于 `FATAL`，应立即终止。

所以，错误恢复的目标不是“尽量成功”，而是“在预算、时间、正确性约束内尽量成功”。这里的“预算”首次出现时，可以理解成“系统允许这次任务最多花多少步数、多少时间、多少 token、多少钱”。没有预算上限，重试机制就会反过来变成风险源。

---

## 核心机制与推导

一个可落地的 ReAct 恢复框架，通常可以写成“自省记忆 + 局部回溯 + 分层重试/降级”。

先看自省。Reflexion 的关键不是让模型多说几句，而是把失败变成结构化反馈。比如代理调用搜索工具得到空结果，不应只记录“失败了”，而应记录更可执行的信息：

| 失败现象 | 反思记忆 | 下一轮影响 |
| --- | --- | --- |
| 搜索结果为空 | 查询词过窄，缺少同义词 | 扩展关键词再查 |
| 页面抓取成功但字段缺失 | CSS 选择器依赖页面旧结构 | 改用语义定位或备用抽取器 |
| 数据库查询超时 | 全表扫描过大 | 先缩小时间范围 |
| 模型回答与约束冲突 | 计划未读取业务规则 | 先拉取规则再回答 |

这类记忆相当于给下一轮补了“失败上下文”。如果没有它，代理会重复走同一路径；如果有它，代理下一步更像“修正搜索策略”，而不是“重复上一次动作”。

再看回溯重规划。单轮 ReAct 往往是线性展开的：想一步，做一步，再看结果。但真实任务有依赖关系。比如“抓取网页 -> 提取价格 -> 计算折扣 -> 输出推荐”。如果“提取价格”失败，问题不是最后一步输出错了，而是中间依赖断了。这时最优策略不是整条链从头再跑，而是只回到受影响节点，重新拆分子任务：

1. 判断失败是否污染后续状态。
2. 标记哪些步骤依赖失败结果。
3. 丢弃这些步骤的缓存。
4. 保留不受影响结果。
5. 重新生成局部计划。

这就是“局部重做”优于“整盘重跑”的原因。设一次任务链有 $n$ 个步骤，其中只有后 $k$ 步依赖失败节点，那么整盘重跑成本近似为 $O(n)$，局部回溯成本接近 $O(k)$。当 $k \ll n$ 时，收益非常明显。

再看重试与降级。最常用的是指数退避：

$$
delay = \min(base\_delay \times 2^{attempt},\ max\_delay)
$$

如果 `base_delay = 1`、`max_delay = 8`，前三次失败的等待时间就是：
- 第 0 次后等待 1 秒
- 第 1 次后等待 2 秒
- 第 2 次后等待 4 秒

玩具例子：代理请求天气 API，为用户生成出行建议。
- 第一次请求超时，等待 1 秒后重试。
- 第二次仍超时，等待 2 秒。
- 第三次返回 500，等待 4 秒。
- 超过 `max_retries=3` 后，不再请求主 API，而是降级到缓存天气或直接提示“当前天气数据不可确认”。

真实工程例子：企业内一个销售助手代理，需要从 CRM、邮件系统、知识库三个工具收集信息后给销售生成跟进建议。若 CRM 查询失败：
- 先写反思：“失败点在 CRM 接口，不是用户问题。”
- 再判断依赖：客户基础信息缺失，会影响建议生成，但不影响知识库查询。
- 于是保留知识库结果，只回溯 CRM 相关子链。
- 若 CRM 连续失败，则降级到昨日缓存快照，并在最终回答中标注“客户状态基于最近可用数据”。

这一套机制的本质不是“模型更聪明”，而是把失败处理拆成三层职责：
- 记忆层负责总结为什么失败。
- 规划层负责决定从哪里重新开始。
- 运行时层负责控制多久重试、何时停止、何时降级。

---

## 代码实现

下面给一个可运行的 Python 最小实现。它不依赖具体框架，但把三个关键点都放进去了：错误分级、指数退避、降级回调。

```python
from enum import Enum
from dataclasses import dataclass


class ErrorSeverity(str, Enum):
    RECOVERABLE = "recoverable"
    DEGRADABLE = "degradable"
    FATAL = "fatal"


@dataclass
class RetryPolicy:
    max_retries: int = 3
    base_delay: int = 1
    max_delay: int = 8


class TemporaryToolError(Exception):
    pass


class FatalToolError(Exception):
    pass


def backoff_delay(base_delay: int, attempt: int, max_delay: int) -> int:
    return min(base_delay * (2 ** attempt), max_delay)


class ProductionErrorHandler:
    def __init__(self, policy: RetryPolicy):
        self.policy = policy
        self.events = []

    def with_retry(self, fn, severity: ErrorSeverity, fallback=None):
        if severity == ErrorSeverity.FATAL:
            self.events.append("fatal_stop")
            raise FatalToolError("fatal error, stop immediately")

        last_error = None
        for attempt in range(self.policy.max_retries):
            try:
                result = fn(attempt)
                self.events.append(f"success_at_{attempt}")
                return result
            except TemporaryToolError as e:
                last_error = e
                delay = backoff_delay(
                    self.policy.base_delay,
                    attempt,
                    self.policy.max_delay,
                )
                self.events.append(f"retry_{attempt}_delay_{delay}")

        if severity == ErrorSeverity.DEGRADABLE and fallback is not None:
            self.events.append("fallback")
            return fallback()

        self.events.append("raise_last_error")
        raise last_error if last_error else RuntimeError("unknown failure")


# 玩具工具：前 3 次失败，第 4 次本来也不会被调用，因为 max_retries=3
def flaky_tool(attempt: int):
    raise TemporaryToolError(f"temporary failure at attempt {attempt}")


def stable_fallback():
    return {"source": "fallback_model", "answer": "cached result"}


policy = RetryPolicy(max_retries=3, base_delay=1, max_delay=8)
handler = ProductionErrorHandler(policy)

result = handler.with_retry(
    flaky_tool,
    severity=ErrorSeverity.DEGRADABLE,
    fallback=stable_fallback,
)

assert result["source"] == "fallback_model"
assert handler.events == [
    "retry_0_delay_1",
    "retry_1_delay_2",
    "retry_2_delay_4",
    "fallback",
]

assert backoff_delay(1, 0, 8) == 1
assert backoff_delay(1, 1, 8) == 2
assert backoff_delay(1, 2, 8) == 4
assert backoff_delay(1, 5, 8) == 8
```

这段代码故意保持简单，但已经体现了真实控制流：

```text
agent_call
  -> 成功: 返回结果
  -> 失败:
       判断 severity
       -> FATAL: 立即停止
       -> RECOVERABLE: 按 backoff 重试，超限后报错
       -> DEGRADABLE: 按 backoff 重试，超限后走 fallback
```

如果把它放进 ReAct runtime，一般还要再补四个组件：

| 组件 | 作用 | 为什么不能省 |
| --- | --- | --- |
| 反思记忆存储 | 保存失败原因 | 避免重复试错 |
| LoopBudget | 限制最大步数/工具调用数 | 防止无限循环 |
| Dedupe | 去重相同参数的重复调用 | 防止“同一坏请求”重复发送 |
| Circuit Breaker | 工具持续异常时短路 | 避免把不稳定服务打崩 |

真实工程例子可以这样映射：浏览器工具抓页面失败 3 次后，不再继续调用 `browser.get`，而是切换到搜索摘要结果；若页面对业务关键，则交给人工审核队列，而不是让代理在页面上反复刷新 47 次。

---

## 工程权衡与常见坑

最大的误区，是把“重试”当成“恢复”。重试只是恢复的一部分，而且是最危险的一部分，因为它最容易让系统在错误路径上投入更多成本。

一个典型事故链是这样的：

1. 浏览器工具偶发超时。
2. 代理把超时理解成“页面还没加载完”。
3. 没有去重和步数限制，于是重复 `browser.get`。
4. 每次失败又触发下一轮推理和下一次工具调用。
5. 最终形成“工具重复调用 -> token 持续消耗 -> 用户等待时间变长 -> 成本暴涨”的连锁反应。

因此，工程上要把 guardrail，也就是“护栏机制”，放在运行时，而不是 prompt。它的作用可以理解成“即使模型判断失误，系统也不会无限失控”。

下面是一张实用 checklist：

| 检查项 | 建议阈值/做法 | 触发后行为 |
| --- | --- | --- |
| 最大工具调用数 | 每任务 10 到 30 次 | 超限后停止或降级 |
| 最大总时长 | 30 到 120 秒 | 超限后返回部分结果 |
| 最大 token 成本 | 按任务类型设硬上限 | 超限后禁止继续思考链 |
| 相同工具相同参数重复次数 | 不超过 2 到 3 次 | 命中后触发 dedupe |
| 同一错误连续次数 | 不超过 3 次 | 打开 circuit breaker |
| 降级路径是否存在 | 必须有 | 无降级则直接失败，不再试 |

预算触发后的系统行为，也要提前定义。最差的做法是“超预算了但继续偷偷跑”；正确做法是：
- 先停止新增工具调用。
- 再保留已得到的可信结果。
- 最后明确告诉上层系统当前输出是部分完成、降级完成，还是失败退出。

真实工程例子：一个企业级代理要汇总竞品信息，先访问网页，再抽取表格，再生成分析。如果网页工具非常不稳定，而系统又没有 loop detection，代理可能在同一页面 URL 上来回请求十几次。即使最终拿到页面，用户也可能早就离开了。这时失败的不是模型推理，而是运行时治理。

另一个常见坑，是反思记忆污染。若代理把错误原因总结错了，比如真实问题是权限不足，却记成“查询词不准”，后续轮次会沿着错误方向持续修正。解决方法不是取消反思，而是给反思加 evaluator，也就是“评估器”。它可以是规则、单元测试、约束检查器，甚至人工审核节点。没有稳定评估信号时，反思系统会积累“看似合理、实际上错误”的经验。

---

## 替代方案与适用边界

并不是所有场景都需要“Reflexion + 回溯 + 降级”三件套。是否值得上这套机制，取决于任务复杂度、工具数量、失败成本，以及你能否提供稳定评估信号。

可以把常见方案做一个对比：

| 方案 | 适用任务规模 | 恢复能力 | 成本 | 对模型能力依赖 | 适用边界 |
| --- | --- | --- | --- | --- | --- |
| Hard-coded retries | 小任务、单工具 | 低 | 低 | 低 | 只适合临时性接口错误 |
| 单 agent + Reflexion | 中等任务、少量工具 | 中 | 中 | 高 | 适合需要从失败中学习的链路 |
| RP-ReAct / 多 agent 架构 | 大任务、多工具、多依赖 | 高 | 高 | 中到高 | 适合企业编排、局部回滚、隐私隔离 |

“Hard-coded retries”可以理解成“把重试规则写死在代码里”；优点是简单，缺点是不会分析为什么失败。它适合纯技术性瞬时错误，不适合计划级失败。

单 agent Reflexion 的优势，是能把失败经验直接带进下一轮推理。它适合：
- 错误模式可总结
- 工具数量不算太多
- 单轮上下文还能容纳历史经验

但它也有边界。如果上下文窗口很小、工具输出很长、依赖关系很多，单个 agent 会同时背负“记忆、规划、执行”三项职责，容易过载。

这时更适合 RP-ReAct 或类似多 agent 方案。所谓 RP-ReAct，可以理解成“规划者负责拆任务，执行者负责跑工具，二者分工协作”。它不是 Reflexion 的替代，而是更大的外层框架。二者完全可以组合：
- 执行 agent 在局部失败时记录反思。
- 规划 agent 在收到失败轨迹后重拆子任务。
- 长工具输出存到外部存储，只把摘要放进上下文。
- 关键步骤再由代理或人工做校验。

真实工程例子：企业内一个隐私要求高、工具又很多的场景，例如财务合规审查。规划 agent 只看任务状态和摘要，不直接接触全量敏感文本；执行 agent 去访问内部系统；Reflexion 只记录“哪类查询方式失败、哪种字段映射不稳定”，不记录完整敏感内容。这样既保留错误学习能力，也控制了上下文膨胀和隐私泄露。

最后要强调一个边界：当评估信号不稳定时，不应过度依赖反思式学习。比如代码修复任务，如果没有单元测试、没有静态检查、没有人工 review，代理可能把“这次刚好跑通”误判成“方法正确”。这类场景里，更保守的 hard-coded 校验和人工审查，往往比更复杂的自省机制更可靠。

---

## 参考资料

- Shinn 等人提出的 Reflexion 相关工作，核心思想是将语言反馈写入记忆以改进后续决策；相关复述材料提到其在 AlfWorld 上从 75% 提升到 97%。
- ReAct 相关论文与实践资料，核心思想是将推理轨迹与工具调用交替执行。
- 企业级 Reason-Plan-ReAct 讨论资料，重点说明了失败后的局部回溯与重规划，而不是整链重跑。
- 生产环境下的错误处理实践资料，常见实现包括 `ProductionErrorHandler`、指数退避、fallback 模型与降级流程。
- ReAct with budgets 一类的工程文章，重点强调 step budget、cost budget、loop detection、circuit breaker 等运行时护栏。
- https://deep-paper.org/en/papers/2025-10/2303.11366/
- https://bytetrending.com/2025/12/06/reason-plan-react-ai-agents-for-enterprise-tasks/
- https://www.kanaeru.ai/blog/2025-10-06-production-ai-agents-langchain
- https://www.agentpatterns.tech/en/agent-patterns/react-agent
