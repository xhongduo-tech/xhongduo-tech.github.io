## 核心结论

多 Agent 系统中的死锁，本质是循环等待，也就是 A 等 B 的输出，B 又等 A 的输出，导致这一轮协作永远没有新进展。这里的 Agent，可以先理解为“会独立做事、也会互相请求结果的执行单元”。

判断死锁最直接的方法是构造等待图。等待图，英文是 Wait-for Graph，简称 WFG，可以先理解为“谁在等谁”的有向图：节点是 Agent，边 $A \rightarrow B$ 表示 A 正在等 B。只要图里出现环，例如 $A \rightarrow B \rightarrow A$，就说明系统已经进入死锁。

工程上通常不会只靠一种方法，而是三层一起做：

| 层次 | 目标 | 典型手段 | 适用场景 |
|---|---|---|---|
| 预防 | 尽量不形成环 | 资源有序分配、单调调度、TTL/超时 | 流程可控、资源有限 |
| 检测 | 已经卡住时识别出来 | WFG 环检测、stall 计数 | Agent 数量中等、可收集依赖关系 |
| 恢复 | 死锁后尽快继续 | 终止一条等待、回滚一个 Agent、重试 | 长任务、生产系统 |

如果系统有 $n$ 个 Agent，最坏情况下等待边数可达 $O(n^2)$，因此一次全图巡检的成本常写成 $O(n^2)$。检测越频繁，恢复越快；但 CPU 和调度开销也越高。一个常见近似是：

$$
\text{检测开销/秒} \approx f \times O(n^2)
$$

其中 $f$ 是每秒检测次数。

---

## 问题定义与边界

死锁不是“慢”，而是“永远等不到”。它成立通常需要四个条件同时满足，这四个条件叫 Coffman 条件，可以先理解为“死锁出现前必须凑齐的四张牌”。

| 条件 | 白话解释 | 在多 Agent 里的表现 | 常见防护 |
|---|---|---|---|
| 互斥 | 某资源同一时刻只能给一个人用 | 一个文档编辑锁、一个数据库写锁、一个审批令牌 | 减少独占资源 |
| 占有并等待 | 手里拿着资源，还继续等别的资源 | Agent 已持有上下文窗口或任务所有权，还去等下游结果 | 拿不到就释放、短持有 |
| 不可抢占 | 资源不能被系统强制拿走 | Agent 持有会话、锁、草稿所有权，别人不能直接收回 | TTL、租约、回滚 |
| 环形等待 | 大家排成圈互相等 | $A \rightarrow B \rightarrow C \rightarrow A$ | 资源排序、统一编排 |

这里要明确边界。WFG 最适合单实例、互斥、可重用资源的场景。也就是：

1. 同一资源一次只能被一个 Agent 占用。
2. 资源用完后会释放，不是一次性消耗品。
3. 我们能知道“谁在等谁”。

如果资源有多个实例，或者依赖并不是资源而是“权限委派”“审批链”“工具返回”，WFG 仍然能用，但语义要从“锁等待”扩展成“依赖等待”。这时检测到的更准确说法是协作停滞风险，而不是严格的操作系统锁死锁。

一个玩具例子最容易看清：

- Agent A 先拿到“数据快照锁”，再等 B 的摘要。
- Agent B 先拿到“摘要写入锁”，再等 A 的数据。
- 两边都不超时，也没有外部调度器强制回收。

于是形成：

$$
A \rightarrow B,\quad B \rightarrow A
$$

图里有环，系统停住。

如果所有 Agent 都遵守统一顺序，例如先申请资源 1，再申请资源 2，再申请资源 3，那么就不可能出现“先拿 3 再等 2”的反向边，环形等待自然被切断。

---

## 核心机制与推导

WFG 的规则很简单：

- 节点：Agent
- 边：等待关系
- 判定：有环即死锁

关键不是图本身，而是图的构造时机。系统每次发生“请求某资源但未立即获得”时，就插入一条等待边；资源释放或请求取消时，删除对应边。

三 Agent 的最小环例子：

- A 等 B 的 review
- B 等 C 的数据清洗
- C 等 A 的计划确认

于是得到：

$$
A \rightarrow B \rightarrow C \rightarrow A
$$

这时任何一个 Agent 都没有能力单独推进，必须由系统打破其中一条边，比如让 A 放弃本轮确认并重试。

检测环最常见的方法是 DFS。DFS，深度优先搜索，可以先理解为“沿着一条依赖链一直追下去，看会不会绕回自己”。核心状态有两个：

- `visited`：这个节点是否已经搜过
- `in_stack`：这个节点当前是否在递归路径里

如果从当前节点继续走，遇到了一个 `in_stack=True` 的节点，说明发现回边，也就是环。

伪代码可以写成：

```text
for 每个 agent:
    如果未访问:
        dfs(agent)

dfs(x):
    标记 x 已访问
    标记 x 在当前路径中
    for x 等待的每个 y:
        if y 未访问:
            dfs(y)
        elif y 在当前路径中:
            发现死锁环
    取消 x 的当前路径标记
```

复杂度为什么常写成 $O(n^2)$？因为在多 Agent 协作里，最坏情况下任意两个 Agent 都可能存在等待关系，边数 $E$ 最高可到 $n(n-1)$。DFS 本身是 $O(V+E)$，代入最坏边数后就是 $O(n^2)$。这也是为什么检测周期不能设得过于激进。

再看一个“检测频率”问题。假设 100 个 Agent，最坏边数接近 1 万；如果每秒检测 20 次，监控本身就可能开始抢 CPU。工程上通常会把检测触发拆成两类：

| 触发方式 | 何时用 | 优点 | 缺点 |
|---|---|---|---|
| 周期检测 | 每 1 秒或 5 秒巡检一次 | 简单稳定 | 有固定开销 |
| 事件检测 | 新增等待边时立即检测 | 延迟低 | 高频冲突时更贵 |
| 混合策略 | 事件快速筛查 + 周期兜底 | 实用性最好 | 实现更复杂 |

---

## 代码实现

下面用一个可运行的 Python 玩具实现演示三件事：

1. 维护等待图；
2. 用 DFS 检测环；
3. 用超时或主动中止一个 Agent 来恢复。

```python
from collections import defaultdict
from typing import Dict, Set, List, Optional

class WaitForGraph:
    def __init__(self) -> None:
        self.edges: Dict[str, Set[str]] = defaultdict(set)

    def wait(self, waiter: str, holder: str) -> None:
        if waiter != holder:
            self.edges[waiter].add(holder)

    def clear_wait(self, waiter: str, holder: str) -> None:
        self.edges[waiter].discard(holder)
        if not self.edges[waiter]:
            self.edges.pop(waiter, None)

    def detect_cycle(self) -> Optional[List[str]]:
        visited = set()
        in_stack = set()
        path: List[str] = []

        def dfs(node: str) -> Optional[List[str]]:
            visited.add(node)
            in_stack.add(node)
            path.append(node)

            for nxt in self.edges.get(node, set()):
                if nxt not in visited:
                    cycle = dfs(nxt)
                    if cycle:
                        return cycle
                elif nxt in in_stack:
                    idx = path.index(nxt)
                    return path[idx:] + [nxt]

            path.pop()
            in_stack.remove(node)
            return None

        for node in list(self.edges.keys()):
            if node not in visited:
                cycle = dfs(node)
                if cycle:
                    return cycle
        return None

class DeadlockManager:
    def __init__(self, timeout_seconds: int = 30) -> None:
        self.wfg = WaitForGraph()
        self.timeout_seconds = timeout_seconds
        self.wait_started: Dict[str, int] = {}

    def start_wait(self, waiter: str, holder: str, now: int) -> None:
        self.wfg.wait(waiter, holder)
        self.wait_started.setdefault(waiter, now)

    def resolve_by_abort(self, victim: str) -> None:
        self.wfg.edges.pop(victim, None)
        for node in list(self.wfg.edges.keys()):
            self.wfg.edges[node].discard(victim)
            if not self.wfg.edges[node]:
                self.wfg.edges.pop(node, None)
        self.wait_started.pop(victim, None)

    def timed_out(self, waiter: str, now: int) -> bool:
        return now - self.wait_started.get(waiter, now) >= self.timeout_seconds

# 玩具例子：A 等 B，B 等 A
m = DeadlockManager(timeout_seconds=30)
m.start_wait("A", "B", now=0)
m.start_wait("B", "A", now=1)

cycle = m.wfg.detect_cycle()
assert cycle is not None
assert cycle[0] in {"A", "B"}

# 30 秒后触发恢复，终止 A 这一轮
assert m.timed_out("A", now=30) is True
m.resolve_by_abort("A")
assert m.wfg.detect_cycle() is None
```

上面的代码有三个工程含义：

- `wait(waiter, holder)` 负责记录“谁在等谁”。
- `detect_cycle()` 负责形式化检测死锁。
- `resolve_by_abort(victim)` 负责恢复，方式是让一个 Agent 放弃本轮，释放依赖。

真实工程例子比玩具例子更接近“停滞”而不是纯锁。比如一个编排器里有 `planner`、`researcher`、`reviewer` 三个 Agent：

- `planner` 等 `reviewer` 批准计划；
- `reviewer` 等 `researcher` 补证据；
- `researcher` 又等 `planner` 明确问题范围。

这时不一定有数据库锁，但协作图已经形成环。微软在 AutoGen 迁移到 Agent Framework 的示例里，用 `MagenticBuilder(..., max_stall_count=2)` 表示“连续两轮没有进展就触发 stall 处理”。这不是严格的 WFG 判环，而是把“无进展”当作工程上更稳妥的死锁信号。

如果你在 GroupChat 类系统里实现，实践上可以这样分层：

- 消息层：每条请求带 `deadline` 或 TTL。
- 调度层：维护 `waiter -> holder` 依赖表。
- 恢复层：超过 `max_stall_count` 或超时就 re-plan、重试或人工介入。

---

## 工程权衡与常见坑

生产环境最常见的问题不是“不会检测”，而是“检测得太窄”。

| 方案 | 成本 | 能发现什么 | 容易漏掉什么 | 恢复方式 |
|---|---|---|---|---|
| 只看本地锁 | 低 | 单机互斥资源卡死 | 跨 Agent 委派环 | 超时重试 |
| 全局 WFG | 中 | 形式化循环等待 | 隐式依赖、语义停滞 | 选牺牲者打断 |
| stall 计数 | 低到中 | 连续无进展 | 短暂慢任务与真实死锁混淆 | 重规划、人工介入 |
| WFG + stall | 中到高 | 环和停滞都能看见 | 需要更完整埋点 | 自动恢复更稳 |

几个常见坑：

1. 只画资源锁，不画消息依赖。  
很多多 Agent 死锁不是“数据库锁死”，而是“审批链打圈”。如果日志里没有“Agent A 正在等 Agent B”的事件，WFG 就是不完整的。

2. 没有默认 TTL。  
HTTP 调用通常会设超时，但 Agent 间消息等待常被当作“内部流程”，结果无限挂起。等待必须有过期时间。

3. 资源申请顺序不统一。  
同一组 Agent 今天按“文档后工具”申请，明天按“工具后文档”申请，环迟早出现。顺序必须是协议，不是建议。

4. 只检测，不恢复。  
发现环但不选 victim，系统仍然卡住。检测模块和恢复模块必须一起设计。

5. 把 stall 当死锁，或把死锁当 stall。  
两者有交集，但不相等。死锁强调循环等待；stall 强调连续无进展。实际系统里最好同时看两种指标。

一个实用指标表如下：

| 指标 | 含义 | 典型阈值 |
|---|---|---|
| `deadlock_count` | 检测到的环次数 | 大于 0 就应报警 |
| `resolution_time` | 从发现到恢复耗时 | 超过 5 秒需关注 |
| `avg_wait_time` | 平均等待时长 | 接近 TTL 说明风险升高 |
| `stall_rounds` | 连续无进展轮数 | 2 到 3 轮通常可触发重规划 |

---

## 替代方案与适用边界

如果你的系统能提前知道“每个 Agent 最多还会要多少资源”，可以考虑 Banker’s Algorithm，也就是银行家算法，可以先理解为“批资源前先模拟一下，会不会把系统带进危险状态”。它比 WFG 更偏预防，但要求更多先验信息。

另一个更简单的替代方案，是不用复杂图算法，而是直接采用“资源有序 + TTL + 单调调度”：

- 资源统一编号 1 到 5；
- 所有 Agent 只能按编号递增申请；
- 任一等待超过 30 秒就释放并重试；
- 只有 orchestrator 可以重新分配任务所有权。

例如资源编号为：

| 编号 | 资源 |
|---|---|
| 1 | 原始数据快照 |
| 2 | 检索索引写权限 |
| 3 | 计划草稿编辑权 |
| 4 | 审批令牌 |
| 5 | 发布锁 |

如果所有 Agent 都必须按 1 → 2 → 3 → 4 → 5 的顺序申请，那么“先拿 4 再等 2”这种反向等待就被禁止了，环形等待无法形成。

三类方案可以这样选：

| 方案 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|
| 纯超时机制 | 最简单，容易落地 | 会误伤慢任务，不能解释依赖链 | 小系统、先保可用性 |
| WFG 检测 | 能明确找到环和责任链 | 需要完整依赖数据 | 中型多 Agent 编排 |
| Banker/安全状态 | 能在授权前预防风险 | 需要最大需求已知，维护成本高 | 资源类型固定、需求可预估 |

结论很直接：

- 流程简单时，优先用超时和统一排序。
- 依赖复杂时，上 WFG。
- 资源模型稳定且需求可估时，再考虑银行家算法类方案。

---

## 参考资料

- OneUptime, “How to Create Agent Coordination”: https://oneuptime.com/blog/post/2026-01-30-agent-coordination/view
- OneUptime, “How to Create Deadlock Detection”: https://oneuptime.com/blog/post/2026-01-30-deadlock-detection/view
- Microsoft Learn, “AutoGen to Microsoft Agent Framework Migration Guide”: https://learn.microsoft.com/en-us/agent-framework/migration-guide/from-autogen/
- Agent Patterns, “Deadlocks in Multi-Agent Systems”: https://agentpatterns.tech/en/failures/deadlocks
- GeeksforGeeks, “Wait For Graph Deadlock Detection in Distributed System”: https://www.geeksforgeeks.org/wait-for-graph-deadlock-detection-in-distributed-system/
- PyPI, “agentguard-ai”: https://pypi.org/project/agentguard-ai/
- IDC Online, “Deadlock Prevention”: https://www.idc-online.com/technical_references/pdfs/information_technology/Deadlock_Prevention.pdf
- IDC Online, “Necessary and Sufficient Deadlock Conditions”: https://www.idc-online.com/technical_references/pdfs/information_technology/Necessary_and_Sufficient_Deadlock_Conditions.pdf
