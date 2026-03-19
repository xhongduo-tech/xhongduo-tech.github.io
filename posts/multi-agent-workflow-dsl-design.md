## 核心结论

多 Agent 工作流 DSL，意思是“专门用来描述多个 Agent 如何协作的一套小语言”。它的核心不是先选 YAML、Python 还是图形界面，而是先把执行语义定死：**Agent 是节点，消息传递是边，条件决定分支，反馈边表示循环**。只要这四件事统一，不同语法只是不同的“书写界面”。

可以先用一个最小例子理解：

```yaml
nodes:
  - actor: ResearchAgent
    next: ReviewAgent

  - actor: ReviewAgent
    condition: quality_score >= 0.6
    on_success: PublishAgent
    on_failure: ReviseAgent
```

这段 DSL 表达的不是“代码细节”，而是“流程事实”：

1. `ResearchAgent` 先执行。
2. 输出交给 `ReviewAgent`。
3. `ReviewAgent` 根据 `quality_score` 走向发布或返工。

对零基础读者最重要的一点是：**DSL 不是为了炫技，而是为了把工作流从散落的代码里提取出来，变成可审查、可追踪、可迁移的流程定义**。  
对工程实现最重要的一点是：**语法可以变，运行时语义不能变**。如果 YAML 里的边和 Python 里的边含义不同，系统很快就会失控。

三种常见 DSL 形态的定位可以直接总结如下：

| 形态 | 核心优势 | 主要短板 | 更适合谁 |
|---|---|---|---|
| YAML 声明式 | 可读、可审查、适合固化流程 | 复杂循环和动态逻辑难写 | 产品、运营、平台工程 |
| Python 装饰器式 | 表达力强，便于扩展和复用 | 容易把流程重新写回普通代码 | 应用工程师 |
| 图形可视化 | 并发、汇聚、回路最直观 | 易与源码脱节 | 架构设计、复盘分析、跨团队沟通 |

还可以再补一句判断标准：  
**DSL 不是“让 Agent 自动变聪明”，而是“让协作关系变明确”。**  
一个 Agent 会不会回答问题，主要取决于模型、提示词、工具和上下文；多个 Agent 如何稳定协作，主要取决于编排语义是否清楚。

---

## 问题定义与边界

多 Agent 工作流要解决三个基础问题：

| 问题 | 白话解释 | DSL 中必须有的元素 |
|---|---|---|
| 角色分工 | 谁负责哪一步 | `node` / `actor` |
| 状态传递 | 上一步结果怎么交给下一步 | `edge` / `message` / `state` |
| 执行顺序 | 什么时候分支、什么时候回退、什么时候结束 | `condition` / `loop` / `terminal` |

因此可以先给出一个抽象定义。设节点集合为 $N$，边集合为 $E$，状态空间为 $S$：

$$
N=\{A_1, A_2, \dots, A_n\}, \quad E \subseteq N \times N, \quad s \in S
$$

这里：

- 节点（Node）就是“一个可执行角色”，例如 `ResearchAgent`。
- 边（Edge）就是“从一个角色到另一个角色的调用或消息路径”。
- 状态（State）就是“工作流运行过程中共享的数据”，例如 `query`、`draft`、`quality_score`。

如果再把消息体和条件加进去，一条边就不只是“连线”，而是：

$$
e=(A_i, A_j, m, c)
$$

其中：

- $m$ 表示消息体，也就是要传递的数据。
- $c(s)$ 表示条件谓词。谓词可以理解为“一个返回真或假的判断式”。

对新手更直白的理解是：  
**节点回答“谁做事”，边回答“交给谁”，状态回答“拿着什么做”，条件回答“什么时候走这条路”。**

还可以把一次执行写成状态转移函数：

$$
T: (A_i, s) \rightarrow (A_j, s')
$$

含义是：当前节点 $A_i$ 读取旧状态 $s$，运行后产出新状态 $s'$，再把控制权交给下一个节点 $A_j$。  
如果一个 DSL 连这个最基本的转移关系都说不清，它就不是工作流 DSL，只是配置文件的另一种写法。

一个客服场景的玩具例子如下：

```text
RoutingAgent -> ResearchAgent -> AnalyzerAgent -> ReplyAgent
```

含义很直接：

- `RoutingAgent` 判断用户问题属于哪个类别。
- `ResearchAgent` 检索资料。
- `AnalyzerAgent` 整理证据。
- `ReplyAgent` 生成回复。

把这个例子再展开一步，新手通常就能看出边界：

| 节点 | 输入 | 输出 | 职责 |
|---|---|---|---|
| `RoutingAgent` | `user_query` | `intent` | 判断走退款、物流还是技术支持 |
| `ResearchAgent` | `intent`, `user_query` | `docs` | 检索知识库和案例 |
| `AnalyzerAgent` | `docs` | `facts`, `constraints` | 提炼可用事实和限制条件 |
| `ReplyAgent` | `facts`, `constraints` | `reply` | 生成最终回复 |

如果每一步输出字段不固定，比如 `ResearchAgent` 有时返回 `docs`，有时返回 `knowledge`，后续节点就无法稳定消费输入。所以 DSL 的边界很明确：**它适合描述“流程结构稳定、节点职责明确、输入输出可约束”的协作系统**。

反过来说，下面这些情况不适合只靠 DSL：

- 节点内部算法极其复杂，而且经常变化。
- 路由策略依赖大量实时代码逻辑。
- 状态对象没有固定 schema，也就是字段结构不稳定。
- 流程是否继续执行，严重依赖外部异步事件，但 DSL 中没有等待、超时和恢复语义。
- 节点副作用很多，例如改数据库、发短信、扣费，但 DSL 没有补偿或幂等约束。

这也是为什么声明式 DSL 常用于“标准化流程”，而不是替代所有业务代码。  
**DSL 负责描述骨架，业务代码负责填充肌肉。**

---

## 核心机制与推导

多 Agent DSL 的关键，不是把节点画出来，而是把**节点契约、边条件、状态演化**说清楚。

先看一个带分支的定义：

$$
N = \{ResearchAgent, ReviewAgent, PublishAgent, ReviseAgent\}
$$

$$
E = \{
(ResearchAgent, ReviewAgent),
(ReviewAgent, PublishAgent)\ \text{if}\ quality\_score \ge 0.6,
(ReviewAgent, ReviseAgent)\ \text{if}\ quality\_score < 0.6,
(ReviseAgent, ReviewAgent)\ \text{if}\ revision\_count < 3
\}
$$

这里的 `quality_score` 是状态变量。状态变量就是“会被后续节点读取的运行数据”。  
分支逻辑的本质是：**边的可达性由状态决定**。

可以把它写成一个状态转移表：

| 当前节点 | 关键状态 | 条件 | 后续节点 |
|---|---|---|---|
| `ResearchAgent` | `draft` | 无 | `ReviewAgent` |
| `ReviewAgent` | `quality_score` | `quality_score >= 0.6` | `PublishAgent` |
| `ReviewAgent` | `quality_score` | `quality_score < 0.6` | `ReviseAgent` |
| `ReviseAgent` | `draft`, `revision_count` | `revision_count < 3` | `ReviewAgent` |

这张表已经暴露了 DSL 设计里最重要的三条约束。

第一，**节点输入输出必须显式化**。  
比如 `ReviewAgent` 至少要声明：输入包含 `draft`，输出包含 `quality_score`。否则下游边的条件根本无法判断。

第二，**条件表达式必须有统一求值语义**。  
如果 YAML 用 `quality_score >= 0.6`，Python 却把分数按 100 分制解释成 `60`，那么同一流程会在不同入口走出不同结果。

第三，**循环必须是可追踪的反馈，而不是隐式递归**。  
循环的本质是图中的反馈弧，也就是从后面节点连回前面节点：

$$
A_j \rightarrow A_i
$$

例如 `ReviseAgent -> ReviewAgent`。这意味着状态会被迭代更新，所以必须保留至少这些字段：

| 状态字段 | 作用 | 为什么不能省 |
|---|---|---|
| `workflow_id` | 标识一次流程实例 | 否则无法追踪一次完整运行 |
| `current_node` | 标识当前执行位置 | 否则恢复执行困难 |
| `quality_score` | 分支判断 | 否则无法复现决策 |
| `revision_count` | 控制循环次数 | 否则可能死循环 |
| `message_history` | 保留上下文 | 否则 Agent 间信息断裂 |

为了让新手更容易看懂，可以把“反馈边”理解成“返工单”。  
不是函数自己调用自己，而是流程显式地写明：**审核不过，就回到修改节点；修改后，再重新审核。**  
这和普通递归最大的区别是，工作流引擎能够在每一轮都保存状态、记录路径、限制次数、支持恢复。

玩具例子可以再具体一点。假设输入分数为 `0.55`：

1. `ResearchAgent` 产出一版草稿。
2. `ReviewAgent` 给出 `quality_score = 0.55`。
3. 因为 `0.55 < 0.6`，边 `(ReviewAgent, ReviseAgent)` 被激活。
4. `ReviseAgent` 修改草稿，并把 `revision_count` 加一。
5. 流程回到 `ReviewAgent`，直到分数达标或达到最大重试次数。

把这件事写成一次完整状态演化，会更清楚：

| 轮次 | 当前节点 | 输入状态片段 | 输出状态片段 | 下一跳 |
|---|---|---|---|---|
| 1 | `ResearchAgent` | `topic="DSL"` | `draft="v1"`, `evidence=[...]` | `ReviewAgent` |
| 2 | `ReviewAgent` | `draft="v1"` | `quality_score=0.55`, `review_comment="examples too few"` | `ReviseAgent` |
| 3 | `ReviseAgent` | `draft="v1"`, `review_comment=...` | `draft="v2"`, `revision_count=1` | `ReviewAgent` |
| 4 | `ReviewAgent` | `draft="v2"` | `quality_score=0.81` | `PublishAgent` |
| 5 | `PublishAgent` | `draft="v2"` | `published_url="..."` | 结束 |

真实工程里，这套机制经常用于“先检索、再分析、再撰写、最后审核”的内容生产或客服流程。比如一个客户支持系统：

- `RoutingAgent` 把工单路由到退款、物流或技术支持。
- `ResearchAgent` 检索知识库和历史案例。
- `AnalyzerAgent` 提炼事实与限制条件。
- `WriterAgent` 生成答复草稿。
- `ReviewerAgent` 检查合规性与准确性。
- 不合格则回退到 `WriterAgent` 或 `ResearchAgent`。

这类流程的价值不在于“Agent 数量多”，而在于**每个 Agent 只处理局部职责，DSL 负责把它们串成一个可验证的系统**。

如果进一步支持并发，还需要补上“汇聚”语义。  
例如 `FactCheckAgent`、`StyleCheckAgent`、`PolicyCheckAgent` 可以并行执行，然后统一汇聚到 `FinalReviewAgent`。此时图结构就变成：

```text
             -> FactCheckAgent  -
DraftAgent -> StyleCheckAgent   -> FinalReviewAgent
             -> PolicyCheckAgent -
```

这时单纯的“谁 next 到谁”已经不够，还要定义汇聚条件。最常见的形式是：

$$
join(A_j) = \bigwedge_{A_i \in Parents(A_j)} done(A_i)
$$

意思是：只有当 `FinalReviewAgent` 的所有上游节点都完成，汇聚节点才允许执行。  
如果 DSL 不支持这个语义，工程师就会在节点内部偷偷查询“其他分支跑完没”，依赖关系会从图上消失。

---

## 代码实现

实现上可以把 DSL 分为“定义层”和“执行层”。

定义层负责回答“流程长什么样”。  
执行层负责回答“运行时怎么走图”。

先看 YAML 声明式版本。它适合把流程结构交给人直接阅读和审查。

```yaml
workflow: article_pipeline
version: 1

state_schema:
  topic: str
  draft: str
  evidence: list
  quality_score: float
  review_comment: str
  revision_count: int
  published_url: str

nodes:
  - actor: ResearchAgent
    input: [topic]
    output: [draft, evidence]
    next: ReviewAgent

  - actor: ReviewAgent
    input: [draft, evidence, revision_count]
    output: [quality_score, review_comment]
    condition: quality_score >= 0.6
    on_success: PublishAgent
    on_failure: ReviseAgent

  - actor: ReviseAgent
    input: [draft, review_comment, revision_count]
    output: [draft, revision_count]
    next: ReviewAgent

  - actor: PublishAgent
    input: [draft]
    output: [published_url]
    terminal: true
```

它的优点是字段稳定。`actor`、`next`、`condition` 这些键一眼就能看懂。  
它的缺点也很明确：一旦条件逻辑、并发汇聚、异常恢复变复杂，YAML 会迅速膨胀。

因此一个更稳妥的工程做法是：**YAML 只负责描述结构，不负责承载复杂运行逻辑。**  
复杂逻辑应该下沉到统一 IR 或执行器里。

可以先定义一个统一 IR。IR 的意思是“中间表示”，也就是所有前端语法最终都要编译成的公共结构。

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class EdgeIR:
    source: str
    target: str
    condition: str | None = None

@dataclass
class NodeIR:
    actor: str
    inputs: list[str]
    outputs: list[str]
    terminal: bool = False
```

这样 YAML、Python 装饰器、图形界面都不直接互相转换，而是都转成 `NodeIR + EdgeIR`。  
好处是：**语义只有一套，前端可以有很多套。**

下面给出一个可以直接运行的 Python 装饰器式最小执行器。它不是演示语法花样，而是把“节点注册、条件求值、循环控制、输入输出校验”都补齐。

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

State = dict[str, Any]
NodeFn = Callable[[State], State]


@dataclass
class NodeSpec:
    actor: str
    fn: NodeFn
    inputs: list[str]
    outputs: list[str]
    next: str | None = None
    condition: str | None = None
    on_success: str | None = None
    on_failure: str | None = None
    terminal: bool = False


registry: dict[str, NodeSpec] = {}


def node(
    *,
    actor: str,
    inputs: list[str],
    outputs: list[str],
    next: str | None = None,
    condition: str | None = None,
    on_success: str | None = None,
    on_failure: str | None = None,
    terminal: bool = False,
) -> Callable[[NodeFn], NodeFn]:
    def wrap(fn: NodeFn) -> NodeFn:
        registry[actor] = NodeSpec(
            actor=actor,
            fn=fn,
            inputs=inputs,
            outputs=outputs,
            next=next,
            condition=condition,
            on_success=on_success,
            on_failure=on_failure,
            terminal=terminal,
        )
        return fn
    return wrap


def require_fields(state: State, fields: list[str], actor: str) -> None:
    missing = [field for field in fields if field not in state]
    if missing:
        raise KeyError(f"{actor} missing required inputs: {missing}")


def eval_condition(expr: str, state: State) -> bool:
    allowed_builtins = {"True": True, "False": False, "len": len, "min": min, "max": max}
    return bool(eval(expr, {"__builtins__": {}}, {**allowed_builtins, **state}))


@node(
    actor="ResearchAgent",
    inputs=["topic"],
    outputs=["draft", "evidence"],
    next="ReviewAgent",
)
def research(state: State) -> State:
    topic = state["topic"]
    state["draft"] = f"{topic} 的初稿：先定义执行语义，再选择 DSL 语法。"
    state["evidence"] = [
        "节点表示角色",
        "边表示消息流",
        "条件表示分支",
        "反馈边表示循环",
    ]
    state.setdefault("revision_count", 0)
    return state


@node(
    actor="ReviewAgent",
    inputs=["draft", "evidence", "revision_count"],
    outputs=["quality_score", "review_comment"],
    condition="quality_score >= 0.6",
    on_success="PublishAgent",
    on_failure="ReviseAgent",
)
def review(state: State) -> State:
    revision_count = state["revision_count"]
    if revision_count == 0:
        state["quality_score"] = 0.55
        state["review_comment"] = "定义有了，但例子太少，对新手不友好。"
    else:
        state["quality_score"] = 0.82
        state["review_comment"] = "通过。"
    return state


@node(
    actor="ReviseAgent",
    inputs=["draft", "review_comment", "revision_count"],
    outputs=["draft", "revision_count"],
    next="ReviewAgent",
)
def revise(state: State) -> State:
    state["revision_count"] += 1
    state["draft"] += " 已补充状态转移表、并发汇聚示例和失败处理说明。"
    return state


@node(
    actor="PublishAgent",
    inputs=["draft"],
    outputs=["published_url"],
    terminal=True,
)
def publish(state: State) -> State:
    state["published_url"] = "https://example.com/article/dsl"
    state["published"] = True
    return state


def run(start: str, state: State, *, max_steps: int = 20) -> State:
    current = start
    steps = 0
    state["trace"] = []

    while True:
        steps += 1
        if steps > max_steps:
            raise RuntimeError("workflow exceeded max_steps")

        spec = registry[current]
        require_fields(state, spec.inputs, spec.actor)

        state["current_node"] = spec.actor
        state["trace"].append(spec.actor)
        state = spec.fn(state)

        require_fields(state, spec.outputs, spec.actor)

        if spec.terminal:
            return state

        if spec.condition is None:
            if spec.next is None:
                raise RuntimeError(f"{spec.actor} has no next node")
            current = spec.next
            continue

        branch = eval_condition(spec.condition, state)
        current = spec.on_success if branch else spec.on_failure
        if current is None:
            raise RuntimeError(f"{spec.actor} branch target is missing")


if __name__ == "__main__":
    initial_state = {"topic": "多 Agent 工作流 DSL 设计"}
    result = run("ResearchAgent", initial_state)

    assert result["published"] is True
    assert result["revision_count"] == 1
    assert result["quality_score"] >= 0.6
    assert result["trace"] == [
        "ResearchAgent",
        "ReviewAgent",
        "ReviseAgent",
        "ReviewAgent",
        "PublishAgent",
    ]

    print("workflow finished")
    print(result)
```

这段代码可以直接运行，命令如下：

```bash
python workflow_dsl_demo.py
```

它演示了一个最小执行器：

- 用 `@node(...)` 注册节点。
- 用 `registry` 维护图元数据。
- 用 `run()` 按条件边推进流程。
- 用 `require_fields()` 检查输入输出契约。
- 用 `max_steps` 防止循环失控。
- 用 `trace` 记录完整执行路径，便于排障和回放。

对新手尤其要说明一件事：  
这里的 `condition="quality_score >= 0.6"` 不是“为了炫表达式引擎”，而是为了让**条件边可序列化**。  
如果条件只能写成 Python lambda，它就很难被 YAML、数据库配置或图形界面复用。

字段映射关系最好在 DSL 设计阶段就固定下来：

| 语义 | YAML field | Python decorator arg | 图形节点属性 |
|---|---|---|---|
| 节点名 | `actor` | `actor` | `label` |
| 顺序边 | `next` | `next` | `outgoing edge` |
| 条件表达式 | `condition` | `condition` | `edge predicate` |
| 成功分支 | `on_success` | `on_success` | `true edge` |
| 失败分支 | `on_failure` | `on_failure` | `false edge` |
| 输入契约 | `input` | `inputs` | `input ports` |
| 输出契约 | `output` | `outputs` | `output ports` |
| 结束节点 | `terminal` | `terminal` | `end node` |

图形 DSL 本质上不是第三套语义，而是前两者的可视化投影。  
如果图形工具里画出一个回路，但 YAML 无法表达，问题不在图，而在于**三种 DSL 没有共享同一个中间模型**。工程上更稳妥的做法通常是：先定义统一 IR，再让 YAML、Python、图形分别编译到这个 IR。

如果流程里需要并发与汇聚，IR 最好进一步补上这些字段：

| 字段 | 含义 |
|---|---|
| `edge_type` | `direct`、`conditional`、`fan_out`、`fan_in` |
| `join_key` | 哪些分支属于同一次汇聚 |
| `wait_all` | 是否等待所有上游完成 |
| `timeout_sec` | 汇聚等待超时时间 |
| `retry_policy` | 节点失败后的重试规则 |

没有这些字段时，你以为自己设计的是“工作流 DSL”，实际只是“串行调用配置”。

---

## 工程权衡与常见坑

工程里最大的风险不是“不会写 DSL”，而是“写出三套看似相同、实际不同的 DSL”。

常见坑可以直接列出来：

| 坑 | 影响 | 规避方式 |
|---|---|---|
| YAML 只写 `next`，不写状态 schema | 下游节点输入不稳定 | 强制声明 `input/output` 字段 |
| 条件表达式语义不统一 | 同一流程在不同运行器结果不同 | 定义统一表达式求值规则 |
| 循环没有最大次数 | 返工流程可能死循环 | 增加 `max_iteration` 或 `revision_count` |
| Python 装饰器掺杂业务逻辑 | 流程结构难以审查 | 节点逻辑与图元数据分离 |
| 图形工具手工改图不回写源码 | 图和代码失真 | 图形编辑后统一生成 IR 或源码 |
| 并发汇聚无显式 join | 节点提前执行，拿到半成品状态 | 引入 `join` 语义和完成条件 |
| 错误处理只靠异常 | 流程不可恢复 | 区分“节点失败”和“工作流失败” |

对初级工程师最容易踩的坑是：**把 DSL 当成“另一种代码写法”，而不是“流程契约”**。  
例如在客服流程中，YAML 很容易写出串行路径：

```yaml
RoutingAgent -> ResearchAgent -> AnalyzerAgent -> ReviewerAgent
```

但真实业务可能是：

- 一个分支检索知识库。
- 一个分支读取用户历史工单。
- 两路结果汇聚后再审核。

这时简单的 `next` 已经不够，需要显式支持并发和汇聚。否则你在 YAML 里写出来的是线性流程，运行时却偷偷在 Python 里做并发，最后系统行为就不可解释。

并发或反馈场景里，建议保留这些状态字段：

| 字段 | 用途 |
|---|---|
| `trace_id` | 贯穿一次请求的全链路追踪 |
| `branch_id` | 标识并发分支 |
| `join_status` | 标识汇聚节点是否已收齐输入 |
| `retry_count` | 控制失败重试 |
| `last_error` | 保存最近一次节点错误 |
| `updated_at` | 排查乱序与超时 |

还要再补一个经常被忽略的问题：**失败语义**。  
一个节点抛异常，不一定等于整个工作流失败。至少要区分三种情况：

| 失败类型 | 例子 | 推荐处理 |
|---|---|---|
| 可重试失败 | 调用外部 API 超时 | 节点级重试 |
| 可补偿失败 | 已写数据库但后续审核失败 | 触发补偿节点 |
| 不可恢复失败 | 输入状态缺关键字段 | 终止工作流并报警 |

真实工程例子里，内容审核流程常见一个问题：`ReviewerAgent` 要等待多个上游结果，例如事实校验、风格校验、合规校验。如果 DSL 只支持单输入边，工程师就会在节点内部偷偷拉取别的状态字段，导致依赖关系从图上消失。  
一旦图上看不到依赖，排障、重放、补偿都会变得很困难。

另一个常见坑是“节点副作用不幂等”。  
例如 `PublishAgent` 可能会发邮件、写数据库、推送消息。如果工作流恢复时重复执行这一节点，就可能重复发布。  
因此对有副作用的节点，最好额外定义：

| 约束 | 目的 |
|---|---|
| `idempotency_key` | 防止重复写入或重复发送 |
| `timeout` | 控制卡死节点 |
| `retry_policy` | 避免无限重试 |
| `compensation` | 支持撤销或补救 |
| `checkpoint` | 支持从中间状态恢复 |

一句话总结工程权衡：  
**你不是在设计“怎么写配置”，而是在设计“系统如何解释配置”。**

---

## 替代方案与适用边界

多 Agent 工作流 DSL 不是唯一方案。常见替代方案有状态机库、Airflow、Temporal 这类工作流引擎。

| 方案 | 表达力 | 易用性 | 协作性 | 适用边界 |
|---|---|---|---|---|
| 多 Agent DSL | 对 Agent 角色和消息流表达直接 | 中等 | 高 | 需要统一人机可读流程 |
| 状态机库 | 状态转移严谨 | 中等 | 中 | 规则明确、角色概念不强 |
| Airflow DAG | 批处理与调度成熟 | 高 | 中 | 数据任务、定时任务 |
| Temporal | 长时运行、补偿、重试强 | 中等偏低 | 中 | 复杂可靠性要求的业务流程 |

边界要看你要优化什么。

如果重点是“任务调度”和“可恢复执行”，Temporal 往往更强。  
如果重点是“数据管道编排”，Airflow 往往更成熟。  
如果重点是“把多个 Agent 的角色分工、消息传递、分支与循环用一套统一语义表达出来”，专门的多 Agent DSL 更自然。

举个新手容易看懂的对比：

如果用 Airflow DAG，你画出来的通常是任务节点，关注点是“哪个 task 先跑、哪个 task 后跑”。它能编排流程，但不天然强调“这个节点是 `ReviewerAgent`，那个节点是 `ResearchAgent`，它们交换的是什么语义消息”。  
而多 Agent DSL 会把“agent-node”作为一等公民，也就是系统里最基本的建模对象。

如果用 Temporal，重点则是“长时运行、故障恢复、重试和补偿”。  
它很适合承载生产级可靠性，但不会自动替你定义“多 Agent 协作语义”。  
也就是说，Temporal 更像可靠执行底座，DSL 更像协作语义层。

实际工程中，常见的稳妥做法不是二选一，而是混合：

- 用 DSL 描述 Agent 级别语义。
- 用传统工作流引擎承载重试、超时、持久化、补偿。

推荐混合使用的场景包括：

- 流程跨分钟、跨小时，需要可靠恢复。
- 一个节点要调用外部系统，失败率和超时不可忽略。
- 审批、合规、人工介入等步骤需要持久化状态。
- 团队既要图形化审阅，又要生产级调度能力。

一句话概括适用边界：**DSL 负责“把流程说清楚”，引擎负责“把流程跑稳”**。

如果进一步压缩成一个决策表，可以这样看：

| 你的核心问题 | 更优先的方案 |
|---|---|
| 我想让产品、运营、工程都能读懂 Agent 协作关系 | 多 Agent DSL |
| 我想严格控制有限状态和状态转移 | 状态机库 |
| 我想做定时任务和数据处理流水线 | Airflow |
| 我想做强恢复、强重试、强补偿的长流程 | Temporal |
| 我既要清晰语义，又要生产级可靠性 | DSL + 工作流引擎混合 |

---

## 参考资料

| 资料名称 | 用途 | 链接 |
|---|---|---|
| Strands Agents: Multi-agent Patterns | 用于理解 Graph、Swarm、Workflow 三类多 Agent 模式，以及共享状态与图式组织方式 | https://strandsagents.com/latest/documentation/docs/user-guide/concepts/multi-agent/multi-agent-patterns/ |
| Microsoft Agent Framework: Declarative Workflows | 用于参考 YAML 声明式工作流、变量命名空间、表达式语义和多 Agent 编排模式 | https://learn.microsoft.com/en-us/agent-framework/user-guide/workflows/declarative-workflows |
| Microsoft Agent Framework: Workflow with Branching Logic | 用于参考条件边、分支路由、结构化输出与条件求值 | https://learn.microsoft.com/en-us/agent-framework/tutorials/workflows/workflow-with-branching-logic |
| Microsoft Agent Framework: Sequential Workflow | 用于参考顺序编排、Agent 管道和事件流式执行 | https://learn.microsoft.com/en-us/agent-framework/tutorials/workflows/simple-sequential-workflow |
| Microsoft Agent Framework: Concurrent Workflow | 用于参考并发编排、fan-out 和结果聚合 | https://learn.microsoft.com/en-us/agent-framework/tutorials/workflows/simple-concurrent-workflow |
| Apache Airflow: DAG Concepts | 用于对比任务 DAG 与 Agent 语义 DSL 的差异，理解 DAG 更偏调度而非 Agent 角色建模 | https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html |
| Temporal Docs | 用于对比长时运行、故障恢复、重试与补偿这类执行层能力 | https://docs.temporal.io/ |

这些资料对应三类问题：

| 你想补哪块知识 | 优先看什么 |
|---|---|
| 想看多 Agent 模式本身怎么分类 | Strands Agents |
| 想看声明式 YAML 和条件边怎么落地 | Microsoft Agent Framework |
| 想看生产级执行可靠性怎么补足 | Temporal |
| 想看传统 DAG 为什么不等于 Agent DSL | Airflow |

最后把全文压缩成一句工程判断：  
**多 Agent 工作流 DSL 的设计重点，不是“选一种更好看的写法”，而是“先固定节点、边、条件、循环、并发、错误恢复这些执行语义，再决定用什么写法承载它”。**
