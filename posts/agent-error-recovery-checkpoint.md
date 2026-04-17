## 核心结论

Agent 错误恢复不是“报错后再试一次”，而是一条完整的控制链：先判断错误类型，再决定是重试、回退到检查点，还是直接熔断并交给人工。这里的“熔断”可以理解为先停住故障路径，避免错误继续放大。

长流程 Agent 的失败，通常不是单点问题，而是三类问题混在一起：

| 模式 | 适用场景 | 延迟开销 | 复杂度 | 注释 |
| --- | --- | --- | --- | --- |
| 简单重试 | 瞬时网络抖动 | +1-5 秒 | 低 | 只适合可安全重放的步骤 |
| 指数退避 | 429、临时超时 | +5-60 秒 | 低 | 要配合抖动，避免集体同时重试 |
| 回退到检查点 | 长链路任务 | 秒级到分钟级 | 中 | 失败后从最近安全点继续 |
| 熔断 + 人工介入 | 下游持续异常、写操作高风险 | 即刻 | 中 | 先止损，再排障 |

核心设计原则只有两条。

第一，错误要分型。工具调用失败、推理偏离、环境状态破坏，不是同一个问题，不能用同一套恢复逻辑。

第二，恢复要闭环。闭环指“检查点保存 + 状态回滚 + 回退后重规划”三个动作连起来，而不是只做其中一个。只会重试而不会回退，失败会级联；只会回退而不会重规划，Agent 很可能回到旧状态后重复犯错。

对初学者可以这样理解：如果 Agent 在“调用外部 API”这一步超时，系统不应该从头再跑，而应该读取前一个安全快照，恢复上下文、工具结果和必要的环境信息，然后从“这一步之后”重新想后续计划。

---

## 问题定义与边界

先定义“错误恢复”讨论的边界。本文讨论的是有状态 Agent，也就是执行过程中会积累上下文、工具结果、文件改动或容器状态的系统。纯问答机器人基本不需要复杂检查点，因为它没有长链路副作用。

三类错误可以这样划分：

| 故障类别 | 典型症状 | 监控信号 | 回滚边界 | 可直接重试？ |
| --- | --- | --- | --- | --- |
| 工具失败 | HTTP 502、超时、429、命令退出码非 0 | 重试计数上升、相同参数反复调用 | 当前步骤之前 | 通常可以，需限次数 |
| 推理偏离 | 计划跑偏、答非所问、逻辑自相矛盾 | 输出置信度下降、目标漂移、步骤重复 | 上一个子目标 | 不建议直接重试，应重规划 |
| 环境破坏 | 文件被误删、容器污染、缓存与磁盘不同步 | 校验和变化、文件 diff 异常、测试环境不可复现 | 最近环境快照 | 不能裸重试，应先恢复环境 |

这里“子目标”是把大任务拆成几个阶段性目标，例如“定位 bug”“生成补丁”“运行测试”。对子目标做恢复，比对每个 token 做恢复更实际。

玩具例子：一个自动回复 Agent 分三步执行。

| 步骤 | 动作 | 是否建议写检查点 |
| --- | --- | --- |
| 1 | 识别用户问题类型 | 是 |
| 2 | 调用知识库检索答案 | 是 |
| 3 | 生成最终回复并发送 | 发送前必须写 |

如果第 2 步知识库超时，应该回到第 1 步之后的检查点，保留“问题类型已判定”的结果，再重新规划第 2、3 步，而不是重新分析整段对话。

边界也要说清楚。不是所有系统都值得做细粒度检查点。任务只有 2 步、每步成本很低、无外部副作用时，直接整体重跑通常更便宜。检查点本身也有成本，包括序列化、存储、版本兼容和恢复验证。

---

## 核心机制与推导

恢复策略可以分成三层。

第一层是瞬时错误处理，用于网络抖动、429、偶发超时。常用公式是指数退避：

$$
delay_n = base \times multiplier^n + jitter
$$

其中 `jitter` 是抖动，白话解释就是“给每个实例加一点随机延迟，别让大家同一秒一起重试”。

第二层是状态回退。每完成一个关键步骤或子目标，就写入检查点。检查点至少要保存四类信息：

| 状态层 | 具体内容 | 为什么必须保存 |
| --- | --- | --- |
| 任务状态 | 当前步骤、已完成子目标、待办计划 | 否则恢复后不知道做到哪里 |
| 记忆状态 | 摘要上下文、重要中间结论 | 否则模型会“失忆” |
| 工具状态 | API 返回、游标、临时路径、缓存引用 | 否则会重复打外部系统 |
| 环境状态 | 文件 diff、容器快照标识、配置版本 | 否则恢复后环境不一致 |

第三层是回退后重规划。回滚不是回到过去继续盲跑，而是基于“最近安全状态 + 当前错误类型”重新生成后续轨迹。

数学上，重试收益是递减的。若单次调用成功率为 $a$，最多再试 $k$ 次，则单步最终成功率为：

$$
P_{step}=1-(1-a)^{k+1}
$$

若一个任务需要连续成功完成 $d$ 个相互独立的关键步骤，则总成功率近似为：

$$
P_{task}=\left(1-(1-a)^{k+1}\right)^d
$$

这个公式比“把依赖数和重试数直接相加”的简化写法更准确，因为它明确区分了“单步重试”和“整条链路都要成功”这两个层面。

举一个玩具例子。假设每个工具单次成功率 $a=0.6$，任务需要 3 个关键步骤，允许 2 次重试，即总共 3 次尝试：

$$
P_{step}=1-(1-0.6)^3=0.936
$$

$$
P_{task}=0.936^3 \approx 0.82
$$

也就是说，单步从 60% 提高到了 93.6%，整条 3 步任务能从原来的 $0.6^3=0.216$ 提高到约 0.82。提升非常大。

但继续加重试次数，边际收益会迅速下降。Nebius 在软件工程 Agent 的实验中给出过一个很直观的结果：默认提交率大约 50%，最多 3 次重试后能到 80%，增加到 9 次后只到 90%。这就是典型的非线性关系。前三次很值钱，后面越来越像“花更多钱换更少收益”。

---

## 代码实现

下面给出一个可运行的极简实现，演示三件事：保存检查点、失败后回退、回退后继续执行。这里用内存字典模拟持久化存储，用“计划步骤”模拟 Agent 的执行链。

```python
from dataclasses import dataclass, asdict
from typing import Any, Dict, List
import copy

STORE: Dict[str, Dict[str, Any]] = {}

class RetryableToolError(Exception):
    pass

@dataclass
class Checkpoint:
    session_id: str
    step_index: int
    memory: List[str]
    artifacts: Dict[str, Any]
    schema_version: int = 1

def save_checkpoint(cp: Checkpoint) -> None:
    STORE[cp.session_id] = copy.deepcopy(asdict(cp))

def load_checkpoint(session_id: str) -> Checkpoint:
    data = STORE.get(session_id)
    if not data:
        return Checkpoint(session_id=session_id, step_index=0, memory=[], artifacts={})
    return Checkpoint(**copy.deepcopy(data))

def step_classify(state: Checkpoint) -> None:
    state.memory.append("intent=refund")

def step_search_kb(state: Checkpoint, should_fail: bool) -> None:
    if should_fail:
        raise RetryableToolError("knowledge base timeout")
    state.artifacts["kb_result"] = "refund_policy_v2"

def step_reply(state: Checkpoint) -> None:
    assert state.memory[-1] == "intent=refund"
    assert state.artifacts["kb_result"] == "refund_policy_v2"
    state.memory.append("reply_sent=false")

def run_agent(session_id: str, fail_once: bool = True) -> Checkpoint:
    state = load_checkpoint(session_id)
    plans = [
        lambda s: step_classify(s),
        lambda s: step_search_kb(s, should_fail=fail_once and s.step_index == 1),
        lambda s: step_reply(s),
    ]

    while state.step_index < len(plans):
        try:
            plans[state.step_index](state)
            state.step_index += 1
            save_checkpoint(state)
        except RetryableToolError:
            # 回到最近安全点，不清空已验证结果
            state = load_checkpoint(session_id)
            fail_once = False
    return state

final_state = run_agent("demo")
assert final_state.step_index == 3
assert final_state.artifacts["kb_result"] == "refund_policy_v2"
assert "intent=refund" in final_state.memory
assert "reply_sent=false" in final_state.memory
```

这段代码故意让知识库检索第一次失败。恢复时不会重新跑“识别问题类型”，因为第 1 步已经被检查点确认过。

真实工程例子可以看软件工程 Agent。一个修复 GitHub issue 的 Agent，通常会经历“读 issue -> 浏览仓库 -> 修改文件 -> 运行测试 -> 提交补丁”。这里最贵的状态不是对话历史，而是环境状态：文件改动、测试产物、依赖安装结果。SWE-Replay 的思路就是在高价值语义节点选检查点，并尽量用文件 diff 恢复后续轨迹，而不是每次都从干净仓库重新探索。这样省的是整段搜索和执行成本，不只是几条消息。

---

## 工程权衡与常见坑

检查点粒度是最常见的架构权衡。粒度太粗，失败后回退成本高；粒度太细，存储和恢复开销高，还会让状态版本管理变复杂。

按步骤保存，优点是恢复精确；按子目标保存，优点是实现简单、状态更稳定。对大多数业务 Agent，建议先按“子目标”做，例如“检索完成”“草稿生成完成”“外部写操作前”。

常见坑如下：

| 坑 | 影响 | 规避方式 |
| --- | --- | --- |
| 对写操作直接重试 | 重复扣费、重复发消息、重复建工单 | 强制使用 idempotency key |
| 只有重试，没有熔断 | 下游雪崩、限流扩大 | 设置最大重试数、失败预算、熔断窗口 |
| 只存对话，不存环境 | 恢复后上下文在，文件状态不在 | 同时保存文件 diff 或环境快照引用 |
| 检查点无版本号 | 升级后旧状态无法恢复 | 增加 `schema_version` 并做迁移 |
| 回滚后不重规划 | 重复走原错误路径 | 把错误原因输入规划器重新拆步 |

真实工程例子：客服 Agent 调票务接口创建工单，接口连续返回 502。系统如果只做“无限重试”，而没有幂等键，就可能生成多张重复工单。这个问题不是模型推理错了，而是系统设计错了。正确做法是：

1. 写操作前生成唯一业务键。
2. 所有重试都带同一个 idempotency key。
3. 超过阈值直接熔断，不再继续打下游。
4. 从最近检查点恢复后，改走“人工审核”分支。

另一个常见误区是把“环境回滚”理解成“删掉上下文重来”。如果 Agent 刚刚编译完项目，失败发生在测试阶段，最便宜的恢复方式通常是回到“补丁已应用、依赖已安装”的安全点，而不是重新 clone 仓库、重新分析 issue、重新生成补丁。

---

## 替代方案与适用边界

不同框架对恢复的支持层级不同，选型要看状态复杂度，而不是看功能列表长短。

| 方法 | 适用场景 | 恢复方式 | 主要限制 |
| --- | --- | --- | --- |
| 手动 JSON 检查点 | 状态结构清晰的自研 Agent | 每步写状态文件 | 需要自己处理 schema 演进 |
| LangGraph 持久化 | 图结构、节点明确的 Agent | 节点前后自动 checkpoint | 状态大时要管好 TTL 和压缩 |
| Temporal Workflow | 强一致、长事务业务流程 | 基于事件历史重放 | 要求工作流代码具备确定性 |
| SWE-Replay 类方案 | 代码仓库、测试成本高的任务 | 语义关键点 + 文件差异恢复 | 实现复杂，依赖环境恢复能力 |

LangGraph 的优势是把检查点绑定在图节点上，适合“规划器 -> 执行器 -> 审核器”这类明确 DAG。Temporal 更像工作流引擎，适合支付、审批、异步回调这种强业务流程。SWE-Replay 代表的是另一条路：不追求通用工作流，而是针对软件工程任务，把“仓库状态恢复”做到极致。

SWE-agent 的多次尝试结果也能说明适用边界。重试对“有机会自己改正”的任务有效，例如模型第一次没提交、第二次换轨迹后成功；但如果失败根因是环境污染、测试机损坏、外部依赖挂掉，单纯增加重试次数很快就没有价值。此时更重要的是环境硬重置、状态恢复和子目标级重规划。

一句话概括边界：短任务靠限次重试，长任务靠检查点，强副作用任务必须加幂等和熔断，环境复杂任务要优先解决可恢复性。

---

## 参考资料

- Tencent Cloud, “How does the intelligent agent recover and roll back from task failure?” https://www.tencentcloud.com/techpedia/126216
- Fast.io, “AI Agent State Checkpointing: A Practical Guide” https://fast.io/resources/ai-agent-state-checkpointing/
- LangChain Docs, “LangGraph Persistence” https://docs.langchain.com/oss/javascript/langgraph/persistence
- Temporal Docs, “Temporal Platform Documentation” https://docs.temporal.io/
- Nebius, “Leveraging training and search for better software engineering agents” https://nebius.com/blog/posts/training-and-search-for-better-software-engineering-agents
- Emergent Mind, “SWE-Replay: Efficient Test-Time Scaling for Software Engineering Agents” https://www.emergentmind.com/papers/2601.22129
- Agent Patterns, “What Is an AI Agent? / Why Agents Fail in Production” https://www.agentpatterns.tech/en/start-here/what-is-an-agent
- SWE-agent GitHub 仓库 https://github.com/SWE-agent/SWE-agent
- SWE-agent Retry and Review System 文档索引（DeepWiki）https://deepwiki.com/SWE-agent/SWE-agent/3.3-retry-and-review-system
