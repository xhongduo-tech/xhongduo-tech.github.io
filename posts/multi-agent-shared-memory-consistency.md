## 核心结论

多 Agent 共享记忆的难点，不在“能不能把内容存下来”，而在“多个写入同时发生时，系统还能不能判断谁依赖谁、哪些更新能直接覆盖、哪些更新必须合并”。如果这一步做错，表面上看只是一个字段被改乱，实际上会连带影响任务拆分、工具调用、结果汇总和最终共识。

CRDT，中文常译为“无冲突复制数据类型”，白话讲就是一种专门为“大家同时写、最后还能收敛到同一结果”设计的数据结构。向量时钟，白话讲就是给每次写入带上一组逻辑计数，用来判断事件之间有没有先后因果关系。两者组合后，共享记忆可以从“谁最后写谁赢”的脆弱白板，变成“冲突可检测、状态可合并、最终能收敛”的并发系统。

一个新手版直觉是：把共享记忆想成所有 Agent 都能写的白板。CRDT 保证“大家同时写，最后还能合成一张一致的图”；向量时钟保证“我知道自己看到的是不是旧图，以及两笔内容到底是谁先谁后”。

下表先给出最重要的差异：

| 方案 | 并发写入结果 | 是否能判定因果 | 是否会静默丢信息 | 典型结果 |
| --- | --- | --- | --- | --- |
| 无保障共享记忆 | 依赖覆盖顺序 | 不能 | 高 | 旧值、覆写、错乱共识 |
| 仅 LWW | 最后写入覆盖前值 | 很弱 | 高 | 简单但经常丢语义 |
| CRDT + 向量时钟 | 可合并并最终收敛 | 能 | 低 | 最终一致、可审计 |

研究侧也已经给出足够明确的信号。Zylos 在 2026 年对多 Agent 记忆架构的总结指出，主流框架的冲突解决仍然偏原始，常见做法仍是 LWW 或串行调度。CodeCRDT 则给出相反方向的证据：把共享状态建成 CRDT 后，600 次试验里实现了 100% 收敛、零合并失败。另一组工程评测显示，复杂软件任务里顶级模型的成功率可能低到约 23%，这不能全部归因于记忆冲突，但足以说明协调成本会直接吞掉模型能力上限。

---

## 问题定义与边界

本文讨论的对象，是**多个 Agent 并发读写同一个共享记忆池**的场景。这里的“共享记忆”不是聊天历史，而是会被多个 Agent 当作事实源、任务板、状态表或决策缓存来读写的数据层。

问题可以具体化为一个很小的例子：3 个 Agent 同时维护同一个任务字段 `status`。

- Agent A 写入“进行中”
- Agent B 写入“暂停”
- Agent C 基于旧快照写入“已完成”

如果系统没有因果判断，最终结果可能只是“最后一次写入的值”。这会带来三类错误：

1. 互相覆写：后写入者把前面的语义直接抹掉。
2. 读旧值：某个 Agent 基于过时快照继续推理。
3. 错乱共识：汇总层把彼此不兼容的状态当成一致事实。

边界需要说清楚：

- 只讨论共享记忆，不讨论完全隔离记忆。
- 只讨论存在并发写入的任务，不讨论严格串行流水线。
- 假设 Agent 之间能传播状态，哪怕传播有延迟。
- 重点是最终一致性，不追求每一时刻都强一致。
- 只处理“状态合并”问题，不展开讨论权限系统、隐私隔离和向量检索召回。

一个玩具例子最能说明问题。3 个 Agent 共用一个进度条：

| 时间 | Agent | 读到的旧值 | 写入的新值 | 如果无保障会怎样 |
| --- | --- | --- | --- | --- |
| T1 | A | `todo` | `started` | 正常 |
| T2 | B | `started` | `paused` | 可能正常 |
| T3 | C | `todo` | `ready` | 可能把 `paused` 覆盖掉 |

这里最关键的不是谁“更晚”，而是谁“看见过谁”。如果 C 根本没看见 A 和 B 的写入，那它和 B 的写入就是并发关系，不应直接用覆盖来判定对错。

---

## 核心机制与推导

向量时钟的定义可以写成：对于 Agent $i$，维护一个向量 $VC_i$，其中 $VC_i[j]$ 表示“Agent $i$ 目前已知的、来自 Agent $j$ 的事件计数”。

写入规则很简单：

1. Agent $i$ 发生一次本地写入时，先执行 $VC_i[i] = VC_i[i] + 1$
2. 把当前值和整个向量一起写入共享记忆
3. 收到其他副本时，逐维取最大值进行合并

合并公式是：

$$
VC_{merge}[k] = \max(VC_a[k], VC_b[k])
$$

因果判定规则是核心。给定两个事件的向量时钟 $VC_a$ 和 $VC_b$：

- 如果 $VC_a \le VC_b$，说明 $a$ 发生在 $b$ 之前，或被 $b$ 观察到了
- 如果 $VC_b \le VC_a$，说明 $b$ 发生在 $a$ 之前
- 如果两者都不成立，即 $VC_a \not\le VC_b$ 且 $VC_b \not\le VC_a$，说明它们是并发冲突

判定表如下：

| 条件 | 含义 | 处理方式 |
| --- | --- | --- |
| $VC_a \le VC_b$ | `a` 早于 `b` | `b` 可覆盖 `a` |
| $VC_b \le VC_a$ | `b` 早于 `a` | `a` 可覆盖 `b` |
| 不可比 | 并发冲突 | 交给 CRDT 合并 |

玩具例子可以写成：

| 事件 | 内容 | 向量时钟 |
| --- | --- | --- |
| 初始 | `todo` | `[0,0,0]` |
| A 写 `started` | A 本地加一 | `[1,0,0]` |
| B 看到 A 后写 `paused` | B 本地加一 | `[1,1,0]` |
| C 没看到前两者，写 `ready` | C 本地加一 | `[0,0,1]` |

此时 `[1,1,0]` 和 `[0,0,1]` 不可比，因为前者在前两维更大，后者在第三维更大。它们不能直接覆盖，必须进入 CRDT 合并。

这里要注意：向量时钟只负责回答“有没有因果先后”，它**不负责定义业务语义**。真正决定怎么合并的是 CRDT 结构本身。

举两个常见设计：

| 字段类型 | 适合的 CRDT | 说明 |
| --- | --- | --- |
| 计数器 | G-Counter | 每个 Agent 只增不减，合并时逐维取 max 再求和 |
| 集合/标签 | OR-Set 或 G-Set | 并发添加可共存，不会互相抹掉 |
| 任务状态 | 多值寄存器 + 业务优先级 | 并发保留多个候选，再按规则归一 |

任务状态为什么不建议直接做成单值 LWW？因为 `paused` 和 `ready` 不是简单的新旧关系，而是不同语义分支。更稳妥的做法是先保留并发值，再在顶层规则里做归并，例如：

- 同时出现 `ready` 和 `started`，保留 `started`
- 同时出现 `paused` 和 `completed`，要求附带前置事件检查
- 同时出现多个终态，进入人工或仲裁 Agent 审核

所以共享记忆的正确分层通常是：

1. 底层：向量时钟判定因果
2. 中层：CRDT 承担可交换合并
3. 顶层：业务规则决定最终可读状态

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不是完整分布式系统，但足够展示“写入、传播、冲突判定、CRDT 合并”这四步。

```python
from dataclasses import dataclass
from typing import Dict, Set, List


def vc_leq(a: List[int], b: List[int]) -> bool:
    return all(x <= y for x, y in zip(a, b))


def vc_merge(a: List[int], b: List[int]) -> List[int]:
    return [max(x, y) for x, y in zip(a, b)]


def compare_vc(a: List[int], b: List[int]) -> str:
    if vc_leq(a, b) and a != b:
        return "before"
    if vc_leq(b, a) and a != b:
        return "after"
    if a == b:
        return "equal"
    return "concurrent"


@dataclass
class VersionedState:
    values: Set[str]
    vc: List[int]


class MultiValueRegister:
    def __init__(self, n_agents: int):
        self.state = VersionedState(values={"todo"}, vc=[0] * n_agents)

    def write(self, agent_id: int, value: str, seen_vc: List[int]) -> VersionedState:
        new_vc = seen_vc[:]
        new_vc[agent_id] += 1
        incoming = VersionedState(values={value}, vc=new_vc)
        self.state = self.merge(self.state, incoming)
        return incoming

    def merge(self, local: VersionedState, incoming: VersionedState) -> VersionedState:
        relation = compare_vc(local.vc, incoming.vc)
        if relation == "before":
            return incoming
        if relation == "after":
            return local
        merged_values = local.values | incoming.values
        merged_vc = vc_merge(local.vc, incoming.vc)
        return VersionedState(values=merged_values, vc=merged_vc)


def choose_business_state(values: Set[str]) -> str:
    priority = ["completed", "paused", "started", "ready", "todo"]
    for item in priority:
        if item in values:
            return item
    raise ValueError("empty values")


register = MultiValueRegister(3)

a = register.write(agent_id=0, value="started", seen_vc=[0, 0, 0])
assert a.vc == [1, 0, 0]

b = register.write(agent_id=1, value="paused", seen_vc=a.vc)
assert b.vc == [1, 1, 0]

c = register.write(agent_id=2, value="ready", seen_vc=[0, 0, 0])
assert c.vc == [0, 0, 1]

assert register.state.vc == [1, 1, 1]
assert register.state.values == {"paused", "ready"}
assert choose_business_state(register.state.values) == "paused"
```

这段代码展示了两个关键步骤：

1. `VC 更新`：写入前先让本 Agent 的计数加一。
2. `CRDT 合并`：若向量可比则覆盖，若不可比则保留并发值并做并集。

写入到传播的流程可以概括成：

| 步骤 | 动作 | 结果 |
| --- | --- | --- |
| 写入 | Agent 本地递增自己的 VC | 产生新事件 |
| 传播 | 把 `(value, VC)` 发给其他 Agent | 别人拿到因果信息 |
| 比较 | 判断 VC 是否可比 | 知道能否直接覆盖 |
| 合并 | 并发则交给 CRDT | 保证最终收敛 |

真实工程例子可以看多 Agent 代码协作。比如 1 个 Planner 负责拆任务，2 个 Coder 并发实现，1 个 Reviewer 汇总。如果大家共享一个“任务板 + 文件状态 + 缺陷列表”：

- `todo_claims` 适合用集合型 CRDT，避免重复抢单
- `progress_count` 适合用 G-Counter，避免计数回退
- `task_status` 不适合裸 LWW，应保留并发候选并加业务仲裁
- `audit_log` 适合追加型结构，保留完整事件链

CodeCRDT 的价值就在这里：让 Agent 通过共享状态观察彼此行为，而不是把协调全压在消息传递上。这样做的收益不是“完全没有冲突”，而是“冲突出现时能确定地收敛”。

---

## 工程权衡与常见坑

第一类坑是把 LWW 当成默认正确。LWW，白话讲就是“最后写入谁就赢”，实现最省事，但它把语义冲突硬压成时间冲突。Zylos 的总结明确指出，很多框架现在仍然停留在这个层次。问题在于，`completed` 覆盖 `paused` 并不代表流程真的合法，只代表最后一个包到了。

第二类坑是没有事件顺序和审计。MongoDB 在 2026 年的文章引用研究指出，多 Agent 失败里有 36.9% 与 agent 间错位有关。这里的“错位”本质上就是：你不知道谁基于什么上下文做出当前写入，也无法回放系统为什么走到这个状态。

第三类坑是把所有字段都强行 CRDT 化。CRDT 不是银弹。像“最终审批人是谁”“付款是否已执行”这种高语义、强约束字段，常常需要中心仲裁，而不是无脑并发合并。

常见坑和规避方式可以总结如下：

| 常见坑 | 典型后果 | 规避策略 |
| --- | --- | --- |
| 只用 LWW | 静默丢语义 | 向量时钟判因果，冲突字段交给 CRDT |
| 没有事件日志 | 无法回放和追责 | 追加式审计日志 + 状态重建 |
| 所有 Agent 都可写所有字段 | 记忆污染快速扩散 | 分层写权限 + 命名空间隔离 |
| 把文本记忆和结构化状态混存 | 合并规则混乱 | 结构化字段单独建模 |
| 向量维度无限增长 | 元数据膨胀 | 固定 Agent 集合或做压缩/分片 |
| 把最终一致性当实时一致性 | 读到短暂旧值 | 在上层逻辑做重试、确认或仲裁 |

还有一个很容易被忽略的点：**最终一致性不等于立即一致性**。如果 Agent A 刚写完，Agent B 马上读，有可能还没收到传播结果。这不是系统坏了，而是系统设计的正常代价。因此顶层逻辑要么接受短暂旧值，要么在关键路径上加确认读、重试或屏障。

---

## 替代方案与适用边界

CRDT + 向量时钟适合的是：并发高、多个 Agent 真会同时写、还希望保留因果历史的系统。若你的任务根本不满足这些条件，就不必上这套复杂度。

几种常见替代方案如下：

| 方案 | 并发能力 | 实现复杂度 | 冲突韧性 | 适用场景 |
| --- | --- | --- | --- | --- |
| CRDT + 向量时钟 | 高 | 高 | 高 | 并发协作、离线同步、最终一致 |
| 中心协调器 | 中 | 中 | 中 | 有 Supervisor，能接受瓶颈 |
| 单写代理 | 低 | 低 | 高 | 写频次低、强控制流程 |
| 数据库锁/事务串行化 | 低到中 | 中 | 高 | 结构化状态、低延迟内网系统 |
| 事件溯源 | 中 | 高 | 高 | 审计优先、可回放系统 |

新手版判断方法很简单：

- 如果只有一个“记忆管理员”负责写，别的 Agent 只读，那单写代理就够了。
- 如果多个 Agent 会同时改同一批状态，但允许短暂不同步，优先考虑 CRDT + 向量时钟。
- 如果每一次写入都必须立即全局一致，比如审批、扣费、发布，那应该用中心协调或事务，而不是最终一致性方案。
- 如果你最看重“以后能不能把事故还原出来”，事件溯源往往比单纯状态表更重要。

因此，CRDT + 向量时钟不是“更高级”的默认答案，而是“在共享并发记忆确实存在时，最系统化的答案”。

---

## 参考资料

- Zylos Research. 2026. *AI Agent Memory Architectures for Multi-Agent Systems*. 结论：共享记忆很重要，但无约束共享会带来污染、噪声和冲突，主流框架的冲突解决仍偏原始。  
  https://zylos.ai/research/2026-03-09-multi-agent-memory-architectures-shared-isolated-hierarchical

- Sergey Pugachev. 2025. *CodeCRDT: Observation-Driven Coordination for Multi-Agent LLM Code Generation*. 结论：基于 CRDT 的共享状态可实现强最终一致，600 次试验达到 100% 收敛、零合并失败。  
  https://www.researchgate.net/publication/396789696_CodeCRDT_Observation-Driven_Coordination_for_Multi-Agent_LLM_Code_Generation

- Raghu Vijaykumar. *Multi-Master Database Replication with Conflict Resolution*. 结论：向量时钟用于建立因果顺序、检测并发冲突；冲突解决可采用 LWW、合并策略或自定义逻辑。  
  https://raghu-vijaykumar.github.io/docs/docs/system-design/examples/data-replication/multi-master-database-replication-with-conflict-resolution/

- MongoDB. 2026. *Why Multi-Agent Systems Need Memory Engineering*. 结论：多 Agent 失败会系统化扩散，引用研究显示 36.9% 失败与 Agent 间错位有关，强调需要可追踪的记忆工程。  
  https://www.mongodb.com/company/blog/technical/why-multi-agent-systems-need-memory-engineering

- Scale AI. 2026. *SWE-Bench Pro*. 结论：复杂软件工程任务中，前沿模型仍存在较低成功率与明显失败模式，说明协调、工具使用和状态管理仍是主要瓶颈。  
  https://static.scale.com/uploads/654197dc94d34f66c0f5184e/SWEAP_Eval_Scale%20%289%29.pdf

- Zylos Research. 2026. *Multi-Agent Software Development: AI-Native Engineering Teams in Practice*. 结论：多 Agent 协作在代码任务中常比单 Agent 更难，协调失败会直接吞掉并行收益。  
  https://zylos.ai/research/2026-03-09-multi-agent-software-development-ai-native-teams
