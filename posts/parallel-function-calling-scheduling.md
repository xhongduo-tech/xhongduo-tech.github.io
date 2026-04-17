## 核心结论

并行函数调用的本质，是让模型在**同一轮推理**里一次性产出多个互不依赖的 `tool_call`，再由运行时真正并发执行。模型负责判断“哪些事可以同时做”，调度器负责把这些事一起发出去；二者缺一不可。

如果这些调用主要是外部 API、数据库查询、搜索这类 **I/O 密集型** 操作，端到端延迟通常会从“所有调用耗时相加”压缩成“最慢那个调用的耗时，再加上少量规划和汇总开销”。公开工程资料与论文实验里，常见加速区间大约是 1.4x 到 3.7x，对应延迟下降常在 40% 到 60% 左右，但前提是：

1. 这些工具调用之间没有数据依赖  
2. 运行时确实并发执行，而不是只让模型返回多个调用  
3. 并发层没有被限流、锁竞争、共享状态冲掉收益

并行不是默认正确。只要调用之间存在状态依赖、顺序依赖，或者共享同一份可变资源，就会出现 **竞态**。竞态可以直接理解为：两个操作同时争抢同一份状态，最终顺序不可预测，因此结果也不可预测。

工程上最稳妥的规则是：

- **只读工具优先并行**
- **写入工具默认串行**
- **同资源写入必须显式加锁或做事务保护**
- **所有可能重试的调用都要有幂等性**

**幂等性** 可以理解为：同一个请求重复执行一次或多次，最终状态一致。没有幂等性，就无法安全重试；没有安全重试，并行批次里的局部失败就很难处理。

| 指标 | 串行调度 | 并行调度 |
| --- | --- | --- |
| 工具执行时间 | $\sum_i t_i$ | $\max_i t_i$（同层） |
| 模型推理轮数 | 多轮 Think → Act → Execute | 更少轮，常可一轮发出多调用 |
| 端到端延迟 | 高，随调用数近似线性增长 | 低，受最慢调用支配 |
| 结果正确性风险 | 低 | 更高，需处理依赖、乱序、重试 |
| 调试难度 | 低 | 更高，需要追踪并发行为 |
| 状态安全性 | 高 | 需额外控制竞态与副作用 |
| 适用场景 | 依赖链、写操作、联调阶段 | 独立读请求、聚合查询、低延迟场景 |

一句话概括：**并行函数调用不是“多调几个工具”，而是“先做依赖分析，再只对独立任务并发执行”。**

---

## 问题定义与边界

问题不是“模型会不会调用工具”，而是“多个工具调用能不能在同一轮里一起发，并且一起跑”。

在传统串行模式里，一个典型循环是：

1. 模型决定调用工具 A  
2. 工具 A 执行并返回  
3. 模型读取 A 的结果，决定是否调用工具 B  
4. 重复直到结束

这套流程安全、直观、好调试，但延迟高。原因不是只有工具慢，而是每个工具调用之间还夹着一次模型重新规划。用户实际支付的是两部分时间：

1. 工具本身的执行时间  
2. 模型每一轮重新思考和生成的时间

并行函数调用只解决一种很具体的问题：**当多个调用彼此没有依赖时，不必按顺序等待。**

形式化地说，把一次工具使用过程抽象成有向无环图（DAG）：

$$
G=(V,E)
$$

其中：

- $V$ 是工具调用节点集合
- $E$ 是依赖边集合
- 若 $(u,v)\in E$，表示工具 $v$ 依赖工具 $u$ 的输出，因此 $u$ 必须先执行

**DAG** 的直白解释是：只表示“谁依赖谁”的图，而且不会绕回自己形成闭环。只要出现环，就说明“任务 A 要等 B，B 又要等 A”，这类计划本身不可执行。

进一步定义每个节点的前驱集合：

$$
\mathrm{Pred}(v)=\{u\in V \mid (u,v)\in E\}
$$

当且仅当：

$$
\mathrm{Pred}(v)=\varnothing
$$

节点 $v$ 才能进入当前执行层。也就是常说的“**入度为 0**”。它的意思很简单：这个任务没有任何前置依赖，可以立即执行。

下面这个例子比抽象定义更直观：

| 任务 | 输入来源 | 是否依赖别的任务 | 能否与其他任务并行 |
| --- | --- | --- | --- |
| `get_weather` | 城市名 | 否 | 能 |
| `get_fx_rate` | 币种对 | 否 | 能 |
| `get_hotel_price` | 城市名、日期 | 否 | 能 |
| `make_plan` | 天气、汇率、酒店价格 | 依赖前三者 | 不能与前三者并行 |

对应依赖关系可以写成：

$$
\text{weather} \rightarrow \text{plan}
$$

$$
\text{fx} \rightarrow \text{plan}
$$

$$
\text{hotel} \rightarrow \text{plan}
$$

因此可执行层是：

$$
L_1=\{\text{weather},\text{fx},\text{hotel}\},\quad
L_2=\{\text{plan}\}
$$

这就是“并行层”的基本概念：**同一层内没有相互依赖，层与层之间有先后约束。**

边界也要说清楚。下面几类任务通常**不适合并行**：

| 场景 | 为什么不能直接并行 | 典型后果 |
| --- | --- | --- |
| 工具 B 依赖工具 A 的输出 | 明确数据依赖 | B 参数缺失或错误 |
| 两个工具同时写同一张表 | 共享可变状态 | 覆盖、脏写、锁冲突 |
| 同一 API 有严格顺序语义 | 顺序改变会改变结果 | 业务语义错误 |
| 创建后立即读取 | 读取依赖新生成 ID 或新状态 | 读到旧状态或 404 |
| 调试阶段要求稳定复现 | 并发增加不确定性 | 难以复盘问题 |

新手最常见的误判是：**“有两个调用”不等于“适合并行”。**

例如：

- “先创建订单，再查询订单状态”不能并行，因为第二步依赖第一步返回的订单 ID
- “先生成 SQL，再执行 SQL”不能并行，因为执行阶段依赖生成结果
- “同时查 CRM 客户信息、订单数据仓库、工单系统历史”通常可以并行，因为三者只是最终汇总到同一个回答中

所以判断标准不是“调用数量”，而是“依赖关系”。

可以把这个判断压缩成一个简单表：

| 判断问题 | 如果答案是“是” | 调度建议 |
| --- | --- | --- |
| 后一步是否要用前一步输出？ | 有数据依赖 | 串行 |
| 多个工具是否会改同一份状态？ | 有共享写入 | 串行或加锁 |
| 工具是否只有读操作？ | 只读 | 优先并行 |
| 结果只是最终汇总，不互相喂参数？ | 无相互依赖 | 可并行 |
| 是否必须严格复现执行顺序？ | 调试或审计要求高 | 串行或混合 |

---

## 核心机制与推导

并行函数调用一般分成两层机制：

1. **模型层判断可并行性**
2. **运行时层执行并发**

### 1. 模型层：先产出计划

模型接到用户问题和工具定义后，会做一件事：判断哪些工具调用彼此独立，并在一次响应里返回多个 `tool_call`。这一步可以理解为 Planner。

**Planner** 的作用不是执行，而是输出一份“任务列表 + 参数 + 潜在依赖”的计划。模型看到的主要信息有：

- 用户目标
- 每个工具的用途
- 参数模式和约束
- 工具描述中暴露的读写语义
- 历史上下文里已有的中间结果

如果接口允许多工具同轮返回，那么模型就有机会把若干独立调用一起给出来。OpenAI 当前的函数调用文档也明确提示：一次响应里可能出现零个、一个或多个函数调用，因此应用侧必须按“可能有多个调用”的前提来设计处理流程。

### 2. 调度层：把计划变成执行层

模型返回多个 `tool_call` 后，运行时不能直接照着数组顺序 `for` 循环执行，而是要先做依赖分析。调度器的目标是把调用划分成若干层：

$$
L_1,L_2,\dots,L_m
$$

满足两个条件：

1. 同一层内部节点互不依赖  
2. 第 $\ell$ 层只依赖前面若干层的结果

这就是拓扑排序的分层版本。标准写法是：若节点 $v$ 的所有前驱都已经完成，则它进入下一层。

更形式化一点，设已完成节点集合为 $S$，则当前可执行集合是：

$$
L(S)=\{v\in V\setminus S \mid \mathrm{Pred}(v)\subseteq S\}
$$

每完成一层，就更新：

$$
S \leftarrow S \cup L(S)
$$

直到所有节点都被调度完成。如果某一步没有任何可执行节点，但仍有未完成任务，就说明图中有环，计划非法。

### 3. 延迟模型

串行执行时，假设第 $i$ 个工具调用要经历一次规划和一次执行，则总时间可近似写成：

$$
T_{\mathrm{serial}}=\sum_{i=1}^{n}\bigl(T_{\mathrm{plan},i}+T_{\mathrm{exec},i}\bigr)+T_{\mathrm{synth}}
$$

其中：

- $T_{\mathrm{plan},i}$ 是第 $i$ 次模型规划耗时
- $T_{\mathrm{exec},i}$ 是第 $i$ 个工具执行耗时
- $T_{\mathrm{synth}}$ 是最终汇总回答耗时

并行分层执行时，总时间可写成：

$$
T_{\mathrm{parallel}}=\sum_{\ell=1}^{m}\Bigl(T_{\mathrm{plan},\ell}+\max_{v\in L_\ell}T_{\mathrm{exec}}(v)\Bigr)+T_{\mathrm{synth}}
$$

关键差异在于：

- 串行是 $\sum$
- 并行同层是 $\max$

如果所有独立任务都能在第一层发出，且只有一次统一规划，则可进一步近似为：

$$
T_{\mathrm{parallel}}\approx T_{\mathrm{planner}}+\max_i T_{\mathrm{exec}}(i)+T_{\mathrm{synth}}
$$

这也是很多并行工具调用系统的核心收益来源：**把“多个独立 I/O 等待”折叠成“等最慢那个 I/O”。**

### 4. 一个可算清楚的例子

假设有三个独立查询：

- A：查客户画像，耗时 220ms
- B：查近 30 天订单，耗时 180ms
- C：查工单历史，耗时 260ms

模型规划耗时 100ms，最终综合回答耗时 80ms。

串行时：

$$
T_{\mathrm{serial}} \approx (100+220)+(100+180)+(100+260)+80=1040\text{ms}
$$

并行时：

$$
T_{\mathrm{parallel}} \approx 100+\max(220,180,260)+80=440\text{ms}
$$

理论加速比：

$$
\mathrm{Speedup}=\frac{T_{\mathrm{serial}}}{T_{\mathrm{parallel}}}
=\frac{1040}{440}\approx 2.36
$$

延迟下降比例：

$$
1-\frac{440}{1040}\approx 57.7\%
$$

这组数字说明了两个关键点：

1. 并行收益通常很可观，但不会无限增长  
2. 最慢的那个工具会决定这一层的完成时间

所以工程上不能只问“能不能并行”，还要问“这一层是否被单个慢工具拖死”。

### 5. 为什么“模型返回多个调用”还不够

很多实现只做到一半：模型确实在同一轮里返回了多个 `tool_call`，但后端仍然串行执行。这样得到的是“并行计划，串行执行”，数学上仍然是：

$$
T \approx \sum_i T_{\mathrm{exec},i}
$$

而不是：

$$
T \approx \max_i T_{\mathrm{exec},i}
$$

因此真正决定收益的，不是日志里出现几个 `tool_call`，而是运行时有没有把同层任务**真的并发派发**出去。

### 6. 一个真实场景

客服代理收到问题：“这个客户最近是否有退款、未解决工单和高价值订单？”

它需要访问三个系统：

- Salesforce：客户等级、客户主档
- Snowflake：最近订单与退款记录
- Zendesk：未关闭工单

这三个查询互不依赖，天然适合放在同一层。调度过程是：

1. 模型一次返回三个读取型工具调用
2. 调度器把三者放入同一并行层
3. 运行时并发执行三次查询
4. 结果回填给模型
5. 模型基于三份结果生成最终总结

用户等待时间不再是三段查询耗时之和，而主要是三者中最慢那个系统的响应时间，再加少量模型规划和汇总开销。

---

## 代码实现

下面给一个**可直接运行**的 Python 最小实现，演示三件事：

1. 如何从依赖图计算并行层  
2. 如何估算串行与并行延迟  
3. 如何用 `asyncio.gather` 真正并发执行同层任务

这段代码只用标准库，不依赖第三方包。

```python
import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class TaskSpec:
    name: str
    latency_ms: int
    mode: str = "read"  # "read" or "write"
    resource: str = ""  # optional shared resource name


def layered_toposort(tasks: Iterable[str], deps: Iterable[Tuple[str, str]]) -> List[List[str]]:
    """
    返回按层划分的拓扑排序结果。
    tasks: 节点列表
    deps: 依赖边 (u, v)，表示 v 依赖 u
    """
    tasks = list(tasks)
    graph: Dict[str, List[str]] = defaultdict(list)
    indegree: Dict[str, int] = {task: 0 for task in tasks}

    for u, v in deps:
        if u not in indegree or v not in indegree:
            raise KeyError(f"Unknown task in dependency: {(u, v)}")
        graph[u].append(v)
        indegree[v] += 1

    queue = deque(sorted([task for task in tasks if indegree[task] == 0]))
    layers: List[List[str]] = []
    visited = 0

    while queue:
        current_layer = list(queue)
        layers.append(current_layer)
        queue = deque()

        for u in current_layer:
            visited += 1
            for v in graph[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    queue.append(v)

        queue = deque(sorted(queue))

    if visited != len(tasks):
        raise ValueError("Dependency graph contains a cycle")

    return layers


def estimate_latency(
    layers: List[List[str]],
    exec_cost_ms: Dict[str, int],
    planner_cost_ms_per_layer: int = 100,
    synth_cost_ms: int = 80,
) -> Tuple[int, int]:
    """
    返回 (serial_ms, parallel_ms)
    serial: 每个任务都单独经历一次规划 + 执行
    parallel: 每层只经历一次规划，同层执行取最大耗时
    """
    serial_ms = 0
    for layer in layers:
        for task in layer:
            serial_ms += planner_cost_ms_per_layer + exec_cost_ms[task]
    serial_ms += synth_cost_ms

    parallel_ms = 0
    for layer in layers:
        parallel_ms += planner_cost_ms_per_layer + max(exec_cost_ms[task] for task in layer)
    parallel_ms += synth_cost_ms

    return serial_ms, parallel_ms


def validate_parallel_layer(layer: List[TaskSpec]) -> None:
    """
    简单的工程规则：
    - write 任务不允许和任何共享 resource 的任务同层
    - 同一 resource 上出现多个 write 任务也不允许同层
    """
    by_resource: Dict[str, List[TaskSpec]] = defaultdict(list)
    for task in layer:
        if task.resource:
            by_resource[task.resource].append(task)

    for resource, tasks in by_resource.items():
        writes = [t for t in tasks if t.mode == "write"]
        if len(writes) >= 2:
            raise ValueError(f"Layer has multiple write tasks on shared resource {resource!r}: {writes}")
        if writes and len(tasks) > 1:
            raise ValueError(
                f"Layer mixes write task with other tasks on shared resource {resource!r}: {tasks}"
            )


async def execute_task(task: TaskSpec) -> Dict[str, object]:
    """
    用 sleep 模拟 I/O 工具调用。
    """
    start = time.perf_counter()
    await asyncio.sleep(task.latency_ms / 1000.0)
    end = time.perf_counter()
    return {
        "task": task.name,
        "mode": task.mode,
        "resource": task.resource,
        "elapsed_ms": round((end - start) * 1000, 1),
        "output": f"result_of_{task.name}",
    }


async def execute_layers(layers: List[List[TaskSpec]]) -> List[Dict[str, object]]:
    """
    同层并发、层间串行。
    """
    results: List[Dict[str, object]] = []
    for idx, layer in enumerate(layers, start=1):
        validate_parallel_layer(layer)
        print(f"[dispatch] layer {idx}: {[task.name for task in layer]}")
        layer_results = await asyncio.gather(*(execute_task(task) for task in layer))
        # 回填时按原始层内顺序组织，避免“谁先返回就排前面”的乱序问题
        results.extend(layer_results)
    return results


async def main() -> None:
    specs = {
        "salesforce": TaskSpec("salesforce", latency_ms=220, mode="read", resource="crm"),
        "snowflake": TaskSpec("snowflake", latency_ms=180, mode="read", resource="warehouse"),
        "zendesk": TaskSpec("zendesk", latency_ms=260, mode="read", resource="ticket"),
        "summary": TaskSpec("summary", latency_ms=50, mode="read", resource=""),
    }

    deps = [
        ("salesforce", "summary"),
        ("snowflake", "summary"),
        ("zendesk", "summary"),
    ]

    task_names = list(specs.keys())
    layers_by_name = layered_toposort(task_names, deps)
    print("layers_by_name =", layers_by_name)

    exec_cost = {name: spec.latency_ms for name, spec in specs.items()}
    serial_ms, parallel_ms = estimate_latency(layers_by_name, exec_cost)
    print("estimated_serial_ms   =", serial_ms)
    print("estimated_parallel_ms =", parallel_ms)
    print("estimated_speedup     =", round(serial_ms / parallel_ms, 2), "x")

    layered_specs = [[specs[name] for name in layer] for layer in layers_by_name]

    start = time.perf_counter()
    results = await execute_layers(layered_specs)
    total_ms = round((time.perf_counter() - start) * 1000, 1)

    print("actual_results =", results)
    print("actual_elapsed_ms =", total_ms)


if __name__ == "__main__":
    asyncio.run(main())
```

这段代码的关键点有三个。

第一，`layered_toposort(...)` 把依赖图转换成执行层。  
如果图里有环，会直接报错，因为这种任务图本来就无法合法调度。

第二，`estimate_latency(...)` 把“串行是求和，并行同层是取最大”这个结论写成了代码。  
它不是精确性能模型，但足够用来做容量规划和架构判断。

第三，`execute_layers(...)` 使用 `asyncio.gather(...)` 做同层并发。  
这一句才是真正产生延迟收益的地方：

```python
layer_results = await asyncio.gather(*(execute_task(task) for task in layer))
```

如果你把它改成：

```python
for task in layer:
    await execute_task(task)
```

那就退化成了串行执行，哪怕模型已经返回了多个 `tool_call`，整体延迟也不会下降到预期水平。

### 如何把它映射到真实 LLM 调用链

真实系统通常比上面的示例多两步：解析模型输出，以及把工具结果回填给模型。主流程通常是：

1. 向模型发送消息和工具定义  
2. 允许模型返回多个 `function_call`
3. 从响应中提取 `call_id`、工具名和参数  
4. 建立依赖关系，得到分层结果  
5. 对每一层执行 `asyncio.gather(...)`
6. 用 `call_id` 把工具结果一一回填  
7. 再发一轮模型请求生成最终回答

可以用下面的伪代码概括：

```python
response = client.responses.create(
    model="gpt-4.1",
    input=user_input,
    tools=tools,
    parallel_tool_calls=True,
)

tool_calls = extract_function_calls(response)
layers = build_layers_from_dependencies(tool_calls)

tool_outputs = []
for layer in layers:
    layer_outputs = await asyncio.gather(
        *(execute_tool_call(tc) for tc in layer)
    )
    tool_outputs.extend(layer_outputs)

final = client.responses.create(
    model="gpt-4.1",
    input=[
        *original_messages,
        *serialize_tool_outputs(tool_outputs),
    ],
)
```

这里有两个容易忽略的细节。

**细节一：`parallel_tool_calls=True` 只是允许模型同轮返回多个调用，不等于你的后端已经并发。**

**细节二：结果回填必须按 `call_id` 对应，而不是按完成时间排序。**  
并发执行后，最先返回的工具不一定是模型最先发出的工具；如果你用“谁先回来就先塞回去”的方式组织结果，后续综合阶段就可能把 A 的结果错配给 B。

### 再给一个新手更容易代入的例子

用户问题是：“杭州明天适不适合去东京出差？”

系统可能要同时做三件事：

- 查杭州天气
- 查人民币兑日元汇率
- 查东京酒店均价

这三个查询都只依赖用户原始输入，不依赖彼此，因此可以同层并发。  
但“生成出差建议”这一步必须等前三者都回来，所以只能放到下一层。

这类模式的共同点是：

- 前半段是**并行收集上下文**
- 后半段是**串行综合结论**

这正是生产系统里最常见、也最稳妥的并行方式。

---

## 工程权衡与常见坑

并行调度的收益主要来自降延迟，但代价很明确：复杂度上升，而且不止上升一点。

### 1. 成本不一定同步下降

延迟下降不等于成本下降。并行后，模型可能在单轮里处理更多工具描述、更多参数、更多工具返回值，因此单次推理 token 会变多。生产环境至少要同时看这几类指标：

| 指标 | 为什么要看 |
| --- | --- |
| `latency_ms` | 验证并行是否真的降延迟 |
| `tokens_per_request` | 防止因为单轮上下文变大导致成本上升 |
| `tool_success_rate` | 并发后单点失败会不会变多 |
| `retry_rate` | 局部失败是否被放大 |
| `rate_limit_hit_rate` | 并发是否把下游 API 打到限流 |
| `slowest_tool_p95` | 批次是否总被一个慢工具拖住 |

很多团队看到平均延迟下降，就以为方案已经成功；结果上线后发现：

- token 成本显著上升
- 某个下游 API 限流更频繁
- 局部失败导致整批重试
- 用户体感并没有改善多少

原因通常是并发收益被下游系统瓶颈抵消了。

### 2. 竞态不是理论问题，而是线上事故来源

下面这张表可以直接作为初版调度规则：

| 工具类型 | 是否默认并行 | 主要风险 | 推荐策略 |
| --- | --- | --- | --- |
| 只读查询 | 是 | 限流、超时、结果乱序 | 可并行，按 `call_id` 回填 |
| 缓存写入 | 谨慎 | 覆盖旧值、重复写入 | 要求幂等键和去重 |
| 数据库写入 | 否 | 抢锁、事务冲突、覆盖 | 默认串行，必要时加锁 |
| 外部副作用调用 | 否 | 重复扣费、重复发送、重复建单 | 必须幂等，通常串行 |
| 创建后查询 | 否 | 读到旧状态或尚未提交状态 | 先写后读，显式串行 |

这套规则背后的原则很简单：

- **读操作主要怕慢**
- **写操作主要怕错**

并行系统最难修的通常不是“慢一点”，而是“偶尔错一次”。因为偶发错误更难复现。

### 3. 新手最容易踩的六个坑

**坑一：模型并行计划了，但后端没有并发执行。**  
日志里有多个 `tool_call`，总耗时却几乎没变。原因通常是代码里仍然是：

```python
for tc in tool_calls:
    result = await execute_tool(tc)
```

而不是 `asyncio.gather(...)`。

**坑二：把“多个调用”误判为“独立调用”。**  
例如两个工具都写同一张用户画像表，虽然参数不同，但底层改的是同一份状态，仍然存在冲突。

**坑三：结果乱序。**  
并发返回后，如果你按完成先后拼结果，而不是按原始 `call_id` 回填，模型后续汇总就可能错配上下文。

**坑四：重试没有幂等保护。**  
并行批次中一个工具失败后，经常需要局部重试。如果这个工具是“扣费”“发短信”“创建工单”，没有幂等键就会制造重复副作用。

**坑五：忽略限流和连接池瓶颈。**  
理论上三路并发应该更快，但实际上三路都打向同一外部服务，结果触发限流，整体反而更慢。

**坑六：把结构化约束和并行能力混为一谈。**  
OpenAI 当前文档里，函数调用支持一次返回多个函数调用；但官方也明确提示过，某些场景下严格结构化输出与并行函数调用存在兼容性限制。因此工程上要把“多调用并行”与“严格 schema 保证”分开验证，不要默认它们在所有模型和端点上都同时成立。

### 4. 真实事故类比

假设一个运维代理同时调用：

- `query_instance_status`
- `restart_instance`

如果调度器错误地把两者放进同一层并发，可能出现这样的情况：

1. `query_instance_status` 在重启发起前读取到“运行中”
2. `restart_instance` 同时触发了实例重启
3. 最终模型拿到的结果却是“实例运行正常，但已触发重启”

这不是模型“理解差”，而是调度错了。正确做法是给工具补齐元数据，例如：

| 工具 | 类型 | 资源 | 调度规则 |
| --- | --- | --- | --- |
| `query_instance_status` | `read_only` | `instance:{id}` | 可与其他只读并行 |
| `restart_instance` | `mutating` | `instance:{id}` | 不可与同资源任务同层 |

然后在调度器里落实规则：  
**同一资源上的 `mutating` 工具，不得与任何其他任务并层执行。**

### 5. 最低限度的工程防护

如果你要把并行工具调用上线，至少要有这几项：

| 防护项 | 作用 |
| --- | --- |
| 工具元数据：`read_only` / `mutating` | 决定默认调度策略 |
| 资源标识：如 `order:{id}`、`user:{id}` | 判断是否共享状态 |
| `call_id` 映射表 | 防止结果错配 |
| 幂等键 | 支持安全重试 |
| 超时与取消机制 | 避免单个慢调用拖死整批 |
| 局部失败策略 | 允许部分成功、部分降级 |
| 追踪链路 | 看清每层调度与每个工具耗时 |

没有这些防护，并行常常只是把“慢”换成“偶发错”。

---

## 替代方案与适用边界

并行调度不是唯一方案，很多时候也不是第一方案。

### 1. 纯串行

最保守，也最容易调试。适合：

- 存在明显依赖链
- 包含写操作或外部副作用
- 系统还在联调阶段
- 需要稳定复现问题

纯串行的缺点是延迟高，但优点是行为清楚。对于很多生产系统，**先正确，再提速** 比一开始就全量并行更现实。

### 2. 混合调度

这是最实用的生产折中。思路通常是：

1. 第一阶段并行收集上下文  
2. 第二阶段串行执行有状态操作  
3. 第三阶段统一汇总结果

例如一个售后代理：

- 并行读取 CRM、订单、工单、物流信息
- 串行执行“退款审批”“发补偿券”“更新工单状态”
- 最后生成给客服或用户的解释

这种模式兼顾了低延迟和状态安全。

### 3. 显式工作流

即不用模型临时判断依赖，而是由开发者预先写死执行图。常见方式有：

- 代码里显式写 DAG
- 用工作流引擎固化依赖
- 用图编排框架定义节点和边

优点是：

- 可解释性强
- 可测试性高
- 更容易做审计和回放

缺点是：

- 灵活性低
- 对开放问题适应性弱
- 新增任务分支时要改流程定义

如果任务结构稳定、合规要求高，显式工作流往往比“让模型每次临时判断依赖”更可靠。

### 4. 一个足够实用的决策表

| 判断条件 | 建议 |
| --- | --- |
| 任务是否互相依赖输出 | 依赖则串行 |
| 是否包含写操作或副作用 | 默认串行 |
| 是否主要耗时在外部 I/O | 是则优先考虑并行 |
| 是否需要极强可调试性 | 优先串行或显式工作流 |
| 是否有清晰的工具元数据与资源边界 | 没有就不要全量并行 |
| 是否能接受局部失败后降级回答 | 能则更适合并行批处理 |

### 5. 适用边界

| 类型 | 典型任务 | 是否适合并行 | 原因 |
| --- | --- | --- | --- |
| 跨系统读查询 | CRM、工单、订单、搜索 | 适合 | 彼此独立，主要是 I/O 等待 |
| 检索增强 | 多知识库、多索引并查 | 适合 | 最终只是聚合上下文 |
| 监控与观测 | 多指标、多服务状态拉取 | 适合 | 大多是只读 |
| 事务写入 | 下单、扣费、审批、退款 | 不适合 | 状态变更敏感 |
| 创建后读取 | 建订单后查状态 | 不适合 | 第二步依赖第一步结果 |
| 多步推理链 | 先分析再生成再验证 | 多数不适合 | 中间结果依赖明显 |

最稳妥的经验法则是：

- **并行适合收集信息**
- **串行适合改变状态**
- **混合适合大多数生产任务**

所以工程上更准确的表述不是“我们支持并行函数调用”，而是：

**我们能把独立读任务并行化，同时把有状态任务保护为串行。**

前者是功能，后者才是成熟度。

---

## 参考资料

1. OpenAI 官方文档，*Function calling*  
   https://platform.openai.com/docs/guides/function-calling  
   当前函数调用主文档。明确说明模型响应中可能出现零个、一个或多个函数调用，应用侧应按“可能有多个调用”来处理；同时给出 `parallel_tool_calls` 的控制方式与限制说明。

2. OpenAI 官方 API Reference，*Responses API*  
   https://platform.openai.com/docs/api-reference/responses  
   当前 Responses API 参考文档中提供 `parallel_tool_calls` 参数，用于控制是否允许模型在单轮中生成多个工具调用。工程实现时应以这里的最新接口定义为准。

3. OpenAI 官方文档，*Assistants Function Calling*  
   https://platform.openai.com/docs/assistants/tools/function-calling  
   历史接口文档。当前页面已明确标注 Assistants API 已弃用，并写明将于 **2026 年 8 月 26 日** 关闭；但该页仍保留了并行函数调用的早期示例，适合用来理解旧版行为。

4. OpenAI 官方博客，*Introducing Structured Outputs in the API*  
   https://openai.com/index/introducing-structured-outputs-in-the-api/  
   这篇文章说明了 `strict: true` 的结构化输出能力，同时也明确写到并行函数调用与严格结构化约束并非在所有场景下完全兼容。做生产方案时，应把“并行能力”和“严格 schema 约束”分别验证。

5. Sehoon Kim, *Full Stack Approach for Efficient Deep Learning Inference*，第 6 章 *LLM Compiler for Parallel Function Calling*  
   https://escholarship.org/content/qt4wf834q8/qt4wf834q8.pdf  
   论文系统性给出 Planner、Task Fetching Unit、Executor 的架构，把函数调用计划编译成依赖图，并对独立任务做并行执行，是理解“规划层”和“执行层”分离的核心资料。

6. UC Berkeley 条目页，*Full Stack Approach for Efficient Deep Learning Inference*  
   https://escholarship.org/uc/item/4wf834q8  
   用于快速定位论文条目、摘要和 PDF，便于查找第 6 章实验与实现描述。

7. Airbyte Engineering，*What Are Parallel Tool Calls in LLMs?*  
   https://airbyte.com/agentic-data/parallel-tool-calls-llm  
   工程视角较强，明确区分“模型返回多个调用”和“运行时真的并发执行”两件事，并给出大致 1.4x 到 3.7x 的收益区间以及生产环境中的限流、连接池、失败处理问题。

8. Jiarui Lu et al., *ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities*  
   https://aclanthology.org/2025.findings-naacl.65/  
   强调状态依赖、隐式依赖和交互式工具环境对 LLM 工具使用能力的挑战。它提醒我们：并行问题不只是“快不快”，还包括“在有状态环境里会不会错”。
