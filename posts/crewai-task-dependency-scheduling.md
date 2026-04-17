## 核心结论

CrewAI 里真正决定任务怎么排队的，不是 `Agent` 的角色描述，而是 `Task` 之间的依赖关系。`context` 字段本质上是在声明“这个任务要消费哪些上游结果”，调度器会把这些声明解析成一个 DAG（有向无环图，白话是“有方向、不能绕回来的依赖图”），只有当某个任务依赖的所有上游任务都完成后，它才会执行。

顺序流程 `Process.sequential` 和层级流程 `Process.hierarchical` 的差别，也不只是“有没有 manager”。前者的语义是“按任务列表顺序推进，默认前一个结果会进入后一个上下文”；后者的语义是“由 `manager_llm` 或 `manager_agent` 决定分配、复审和完成判定”，任务不再只是机械串联，而是先经过经理层调度。

`expected_output` 是任务的成功标准。它不是装饰字段，而是在提示词、输出校验、Guardrail 重试中都直接起作用。写得越具体，Agent 越不容易跑偏；写得越空，调度再正确，输出也可能不稳定。

一个最适合初学者理解的链路是：A 先产出关键点，B 消费 A 的 `context` 做推演，C 再把 A 和 B 融合成 Markdown。这个链路说明了三件事：数据流是显式的、依赖是可追踪的、最终汇总任务可以同时消费多个上游结果。

| Task 名 | context | 预期输出 | 依赖类型 |
|---|---|---|---|
| A: `collect_points` | 无 | 5 条关键点列表 | 根任务 |
| B: `expand_points` | `[A]` | 基于 5 条关键点的展开分析 | 单上游依赖 |
| C: `compose_markdown` | `[A, B]` | 一篇结构化 Markdown 文档 | 多上游汇总 |

---

## 问题定义与边界

Task 是 CrewAI 的基本调度单元。白话说，系统不是直接调度“一个聪明的 Agent”，而是调度“一件被定义清楚的工作”。一个 Task 至少要说明三类信息：做什么 `description`、做成什么样 `expected_output`、需要吃哪些上游结果 `context`。

如果把全部任务记为集合 $T=\{t_1,t_2,\dots,t_n\}$，那么对任意任务 $t_j$，它的 `context` 里每出现一个上游任务 $t_i$，就形成一条有向边：

$$
t_i \rightarrow t_j
$$

调度条件可以写成：

$$
ready(t_j)=\bigwedge_{t_i \in context(t_j)} done(t_i)
$$

意思是：只有当 `context(t_j)` 里的所有任务都完成，$t_j$ 才能进入可执行状态。

这里有两个边界必须说清。

第一，CrewAI 的依赖是显式依赖，不是“猜出来的依赖”。你觉得 Task3 “应该用到” Task1 的输出，不代表系统就会自动理解；只有把 Task1 放进 Task3 的 `context`，这个依赖才存在。对零基础读者来说，可以把它理解成函数参数：没传进去，就不能假定里面有。

第二，异步与同步混用时，不是任何引用都合法。文档和源码摘要都强调，系统会拒绝“未来依赖”，也会限制不安全的异步上下文引用。原因很简单：如果一个任务引用了还没开始或尚未稳定完成的结果，整个数据流就不再可重现。

看一个新手最常见的顺序流程：

- Task1：列出“5 个 API 设计原则”
- Task2：基于 Task1 写一段解释
- `tasks=[Task1, Task2]`
- `Task2.context=[Task1]`

这时语义非常直接：Task2 不是“自己找感觉写”，而是明确消费 Task1 的输出。它不能跳过 Task1，也不能去依赖列表后面的 Task3，因为那会形成未来依赖。

---

## 核心机制与推导

先看 `context` 是怎么把任务串成图的。每个任务都像一个节点，`context=[A, B]` 就表示当前任务向 A、B 各连一条入边。这样在执行前，系统就能判断：

1. 图里有没有回路。
2. 当前任务依赖的节点是否都出现在前面。
3. 异步任务之间是否存在不安全引用。

DeepWiki 对源码的总结给出两个关键校验：一是不能引用未来任务，二是异步任务不能在没有同步屏障时把另一个顺序异步任务放进 `context`。白话说，CrewAI 不允许“先声明你会用到未来结果，再赌它运行时能赶上”。

可以把调度前校验理解成下面这段简化伪代码：

```pseudo
for each task t in tasks:
  for each upstream in t.context:
    if index(upstream) >= index(t):
      raise Error("future dependency")
    if t.async_execution and upstream.async_execution and no_sync_barrier_between(upstream, t):
      raise Error("unsafe async dependency")

build DAG from all edges upstream -> t
topologically schedule tasks
```

`_get_context` 的职责可以用一句话概括：把允许被引用的上游输出收集起来，拼进当前任务的上下文提示里。它不是随便抓历史记录，而是只抓合法、已完成、可见的任务输出。

再看 `async_execution`。这个字段的语义是“当前任务可以并发跑，不阻塞后面的任务列表推进”。但这里的“不阻塞”不是“所有后续任务都能继续跑”，而是“那些不依赖它的任务可以继续排队”。只要某个后继任务在 `context` 里显式引用了它，这个后继任务仍然要等它完成。

最小玩具例子如下：

- A：异步生成 5 条模型压缩要点
- B：异步生成 5 条推理优化要点
- C：同步汇总 A 和 B，输出 Markdown

数学上可以写成：

$$
A \rightarrow C,\quad B \rightarrow C
$$

A 和 B 之间没有边，所以可以并行；C 同时依赖 A、B，所以必须等待两者都完成。调度语义不是“先看到谁完成就先把 C 跑起来”，而是必须满足：

$$
done(A)=1 \land done(B)=1 \Rightarrow ready(C)=1
$$

这就是 CrewAI 支持“A/B 并发，C 汇总”的根本原因。它不是靠任务描述里写“请等待前面任务”，而是靠依赖图和阻塞条件来保证。

再往下推一步，为什么 `expected_output` 会影响执行质量？因为对 LLM 来说，任务描述是在说“要做什么”，而 `expected_output` 是在说“什么算完成”。前者约束过程，后者约束结果。对调度器来说，Task 结束不等于结果合格；如果配置了 Guardrail，系统会拿输出去比对成功标准，不合格就重试。于是你得到一个非常工程化的闭环：

- `description` 定义操作
- `context` 定义输入
- `expected_output` 定义验收
- `guardrail` 定义失败后的反馈和重试

---

## 代码实现

先用一个可运行的 Python 小程序，把 DAG 校验和“依赖满足才能执行”的规则跑通。这里不依赖 CrewAI 本体，只模拟它的核心调度思想。

```python
from dataclasses import dataclass, field
from collections import deque

@dataclass
class Task:
    name: str
    context: list[str] = field(default_factory=list)
    async_execution: bool = False

def validate_and_toposort(tasks: list[Task]) -> list[str]:
    name_to_idx = {t.name: i for i, t in enumerate(tasks)}
    indegree = {t.name: 0 for t in tasks}
    graph = {t.name: [] for t in tasks}

    for i, task in enumerate(tasks):
        for upstream in task.context:
            assert upstream in name_to_idx, f"unknown dependency: {upstream}"
            assert name_to_idx[upstream] < i, f"future dependency: {upstream} -> {task.name}"
            graph[upstream].append(task.name)
            indegree[task.name] += 1

    q = deque([name for name, d in indegree.items() if d == 0])
    order = []

    while q:
        cur = q.popleft()
        order.append(cur)
        for nxt in graph[cur]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)

    assert len(order) == len(tasks), "cycle detected"
    return order

tasks = [
    Task("A", async_execution=True),
    Task("B", async_execution=True),
    Task("C", context=["A", "B"]),
]

order = validate_and_toposort(tasks)
assert order[0] in {"A", "B"}
assert order[1] in {"A", "B"}
assert order[2] == "C"
print(order)
```

这段代码证明两件事：

1. 依赖边只允许从前面的任务指向后面的任务。
2. 没有依赖的任务可以先进入队列，汇总任务必须最后执行。

如果把它映射回 CrewAI 风格，Task 定义通常长这样：

```javascript
const collectPoints = new Task({
  description: "列出 AI Agent 编排的 5 个关键点",
  expected_output: "恰好 5 条要点，使用项目符号，不能出现空泛表述",
  agent: researcher,
  async_execution: true
});

const deriveImplications = new Task({
  description: "基于关键点推导工程影响",
  expected_output: "按 5 条关键点分别给出影响分析",
  agent: analyst,
  context: [collectPoints]
});

const composeMarkdown = new Task({
  description: "融合上游结果，输出 Markdown 文章草稿",
  expected_output: "包含标题、三级结构、代码块的 Markdown 文本",
  agent: writer,
  context: [collectPoints, deriveImplications],
  markdown: true
});
```

对应的调度器逻辑，可以抽象成：

```javascript
async function runTasks(tasks) {
  const futures = new Map();
  const outputs = new Map();

  for (const task of tasks) {
    const contextReady = task.context.every(t => outputs.has(t.name));

    if (task.async_execution && task.context.length === 0) {
      futures.set(task.name, task.executeAsync());
      continue;
    }

    for (const dep of task.context) {
      if (!outputs.has(dep.name) && futures.has(dep.name)) {
        outputs.set(dep.name, await futures.get(dep.name));
      }
    }

    const mergedContext = task.context.map(t => outputs.get(t.name));
    const result = await task.executeSync(mergedContext);
    outputs.set(task.name, result);
  }

  return outputs;
}
```

这个伪代码里最关键的不是 `await` 语法，而是两个判断：

- 如果任务本身是异步且没有必须立刻等待的依赖，可以先发出去。
- 如果下游任务引用了异步上游，就在真正执行前把这些 `Future` 收回来。

真实工程例子可以用“市场洞察报告”理解：

- `Researcher`：抓行业综述
- `Analyst`：基于综述做结构化分析
- `Writer`：同时消费前两者，生成最终 Markdown 报告

这个设计的价值在于责任边界清楚。你能知道“是研究结论有问题，还是分析推理有问题，还是写作整合有问题”，因为每个 Task 的输入和输出都被显式记录了。

---

## 工程权衡与常见坑

先说最常见的一类坑：`description` 和 `expected_output` 不一致。比如描述写“写一段总结”，`expected_output` 却写“输出 JSON，包含 5 个字段”。这时 Agent 往往会摇摆，Guardrail 也更容易判定失败并触发重试。对工程来说，这不是模型不稳定，而是验收标准自相矛盾。

第二类坑是把异步理解成“自动并行优化”。不是。`async_execution=True` 只是在执行层允许并发；如果你没有在后继任务的 `context` 里显式列出依赖，系统就没有义务等待那个异步结果。结果往往是下游任务拿到不完整上下文，生成内容出现缺块。

第三类坑是过早切到层级流程。`hierarchical` 的优势在复杂拆解和跨 Agent 协调，但它多了一层经理决策，意味着更多 token、更多提示词路径、更多不确定性。在线性任务还没跑稳之前就上 manager，通常只会让排查变难。

第四类坑是把“顺序列表”误认为“完整依赖图”。在 `sequential` 里，列表顺序确实决定推进顺序，但这不等于所有真实数据依赖都被表达出来。尤其当某个汇总任务需要同时吃 Task1 和 Task3 的结果时，最好显式写 `context=[task1, task3]`，不要只依赖“反正它们之前执行过”。

下面这张表可以直接当检查单：

| 坑 | 表现 | 原因 | 规避措施 |
|---|---|---|---|
| `expected_output` 过空 | 输出风格漂移、字段缺失 | 成功标准不清晰 | 写清格式、条数、字段、长度、禁止项 |
| `description` 与 `expected_output` 冲突 | Guardrail 频繁驳回 | 过程要求和验收标准矛盾 | 先统一任务说明，再配置校验 |
| 忘记等待异步结果 | 汇总任务内容不全 | 后继 `context` 未显式列依赖 | 所有要消费的上游都放进 `context` |
| 未来依赖 | 启动前校验失败 | 下游引用了任务列表后面的任务 | 只依赖已经定义在前面的任务 |
| 过早使用层级流程 | 排查困难、成本变高 | manager 增加了一层决策不确定性 | 先用顺序流程验证，再引入 manager |

一个真实工程例子是市场洞察报告。很多团队会这么配：

- `research_task`：Researcher 输出行业综述
- `analysis_task`：Analyst 基于综述产出风险、机会、竞争格局
- `writing_task`：Writer 的 `context=[research_task, analysis_task]`，最终生成 Markdown 报告

这种设计好处有三点：一是责任归属明确；二是后续复盘时能追踪每步输入输出；三是如果要并发扩展，还能把“竞品研究”和“用户反馈分析”拆成并行异步任务，再让 Writer 做最终汇总。

---

## 替代方案与适用边界

如果任务天然是线性的，先用 `Process.sequential`。它最适合“上一步结果就是下一步输入”的场景，比如资料收集 → 分析 → 成文。对初学者来说，这种模式最容易观察错误，因为任务列表顺序和数据流顺序基本一致。

如果任务拆解本身不稳定，或者需要一个统一的“经理”来决定“谁该做下一步”，再考虑 `Process.hierarchical`。这里的 `manager_llm` 可以理解为调度型模型，白话说就是“专门负责分派和验收，不直接做业务细节的那层大脑”。

同一组任务，用两种模式的区别可以这样看：

| 场景/需求 | 推荐流程 | 注意事项 |
|---|---|---|
| 研究 -> 分析 -> 写作，链路固定 | `sequential` | 显式写 `context`，不要只靠默认顺序 |
| 多专家协作，任务拆解会动态变化 | `hierarchical` | 必须配置 `manager_llm` 或 `manager_agent` |
| 需要 A/B 并发收集，再由 C 汇总 | `sequential` + `async_execution` | C 必须在 `context` 里列出 A、B |
| 输出格式要求严格，易被 Guardrail 拒绝 | 先 `sequential` | 先把 `expected_output` 调稳，再增加复杂调度 |

一个新手可感知的对比是：

- 顺序模式：你已经知道 Task1 给 Researcher，Task2 给 Analyst，Task3 给 Writer，系统只是照着清单执行。
- 层级模式：你只提供任务池和 Agent 能力，manager 决定什么时候派给谁、是否重做、何时算完成。

因此，层级流程不是“更高级的默认选项”，而是“当任务拆解和复审需要动态决策时才值得引入的选项”。如果你连 `expected_output` 都没写稳，先别上 manager；否则你只是把不稳定放大了一层。

---

## 参考资料

- CrewAI `Processes` 文档：说明 `sequential` 与 `hierarchical` 的执行语义，以及 `manager_llm` / `manager_agent` 是层级流程的必需项。https://docs.crewai.com/en/concepts/processes
- CrewAI `Tasks` 文档：说明 `Task` 的核心属性，包括 `context`、`expected_output`、`async_execution`，并给出“多个异步上游 + 一个汇总任务”的官方示例。https://docs.crewai.com/en/concepts/tasks
- CrewAI `Sequential Processes` 文档：说明顺序流程中任务按列表推进，适合线性链路，并强调任务顺序与上下文传递的关系。https://docs.crewai.com/en/learn/sequential-process
- CrewAI `Hierarchical Process` 文档：说明经理模型如何负责规划、委派、复审与完成判定，支撑“层级流程适合复杂拆解”的结论。https://docs.crewai.com/en/learn/hierarchical-process
- DeepWiki `Task Management` 摘要：总结了 `_get_context`、未来依赖校验、异步上下文限制，以及 `async_execution` 通过 `Future` 和线程执行的实现方式。https://deepwiki.com/crewAIInc/crewAI/2.3-task
