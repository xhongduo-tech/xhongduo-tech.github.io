## 核心结论

多 Agent 系统的负载均衡，核心不是“把请求平均分出去”，而是“把不同重量、不同能力要求、不同区域约束的请求，送到最合适的 Agent 实例”。这里的 Agent 池，是一组可被调度器统一管理的 Agent 工作单元；调度器，是站在入口处做分发决策的组件。

高并发下，系统通常不是先死在“总请求数”上，而是先死在尾延迟。尾延迟指最慢那部分请求的延迟，工程上常看 p95 或 p99。多 Agent 场景里，尾延迟主要由三类因素放大：

1. LLM API 调用慢，单次推理时间拉长。
2. 上下文窗口占用高，导致服务时间变长、显存或 KV cache 更紧。
3. 消息队列积压，请求还没开始算就已经在排队。

容量判断可以先用 M/M/c 的排队模型做一阶近似。若到达率为 $\lambda$，每个 Agent 的平均服务率为 $\mu$，并发 Agent 数为 $c$，则系统利用率为：

$$
\rho = \frac{\lambda}{c\mu}
$$

当 $\rho < 1$ 时系统才稳定；当 $\rho \to 1$ 时，平均等待时间会快速上升。常见写法是：

$$
W = W_q + \frac{1}{\mu}
$$

其中：

$$
W_q = \frac{C(c,a)}{c\mu-\lambda}, \quad a=\frac{\lambda}{\mu}, \quad \rho=\frac{\lambda}{c\mu}
$$

$C(c,a)$ 是 Erlang C 等待概率，白话讲就是“一个新请求到来时，是否必须先排队”的概率。这个公式给出一个直接结论：要压低等待时间，只能从三件事入手，降低 $\lambda$、提高 $\mu$、增加 $c$。对应到工程里，就是限流、提速、扩容。

一个可操作的结论是：外层先用“能力感知的加权分流”，内层再用“最少未完成请求”或 P2C 做负载感知，最后配合水平扩容和冷启动预热。只做轮询通常不够，只做扩容也不够。

---

## 问题定义与边界

本文讨论的对象，是“多个 Agent 共同处理请求，但底层可能共享一组或多组 LLM 实例”的系统。它常见于以下场景：

| 场景 | Agent 行为 | 真正瓶颈 |
| --- | --- | --- |
| 多角色协作问答 | Planner、Retriever、Coder、Reviewer 串联调用 | 某一步长上下文导致整条链路等待 |
| 工具调用型 Agent | 查库、调 API、执行脚本后再请求模型 | 下游工具慢，导致 Agent 长时间占位 |
| 多区域部署 | 不同区域各有 Agent 池 | 区域配额、网络抖动、冷启动 |

这里要先划清边界。

第一，负载均衡的对象不只是 HTTP 请求，还包括“推理时间”和“上下文重量”。上下文重量可以粗看成 token 数量和模型状态占用，白话讲就是“这次请求到底有多吃显存、多占队列”。

第二，多 Agent 不等于多模型。很多系统里，10 个 Agent 逻辑角色最终共享同一个模型池，这时问题会从“Agent 数量不够”转成“共享 LLM 被打爆”。

第三，真实工程必须考虑区域与额度。比如 `us-east-1` 的某家模型 API 配额先耗尽，调度器不能继续只看本地最低延迟，而要把新请求引到 `eu-west-1`，同时预热该区域的 Agent 实例。

一个真实工程例子是客服自动化平台。白天高峰时，一个用户请求会拆成“意图识别 Agent -> 工单检索 Agent -> 回复生成 Agent -> 审核 Agent”。如果回复生成用长上下文模型，审核用短上下文模型，但入口仍然简单轮询，那么某些节点会被长任务占满，短任务也被拖慢，最后 p99 明显恶化。

一个玩具例子更直观。假设有 3 个 Agent：

- A：能力强，平均每秒处理 3 个轻任务，权重 3
- B：能力一般，平均每秒处理 1 个轻任务，权重 1
- C：能力一般，平均每秒处理 1 个轻任务，权重 1

如果每轮按 3:1:1 分流，那么流量大约是 60%/20%/20%。这不是为了“平均”，而是为了“让强节点多吃一点但别被打爆”。如果再加一条规则：任何节点未完成请求数超过 5 就临时降权，那么系统会比静态 3:1:1 更稳定。

区域边界也可以表成表格：

| Region | 配额/状态 | 调度动作 |
| --- | --- | --- |
| `us-east-1` | token 配额接近上限 | 新请求转发到 `eu-west-1`，并限制重试 |
| `eu-west-1` | 容量充足 | 接收溢出流量，保留预热实例 |
| `us-west-2` | 当前无热实例 | 先冷启动，再逐步加权放量 |

---

## 核心机制与推导

先看为什么简单轮询会失效。

轮询默认假设每个请求成本差不多，但 Agent 请求明显不是这样。一个“总结 20 行日志”的请求，和一个“分析 500 页 PDF 并调用 3 个工具”的请求，耗时差一个数量级很常见。于是相同的请求数，不代表相同的负载。

所以调度至少要看三类信号：

| 信号 | 含义 | 适合做什么决策 |
| --- | --- | --- |
| `outstanding` | 当前未完成请求数 | 判断谁更空闲 |
| `queue_depth` | 等待队列深度 | 判断是否该限流或扩容 |
| `p95/p99 latency` | 最近尾延迟 | 判断某个节点是否进入不稳定状态 |
| `context_tokens` | 输入上下文长度 | 估计服务时间和显存占用 |
| `warm/cold` | 是否热实例 | 防止冷实例一上来就吃满流量 |

从排队论看，多 Agent 池可以被粗略看成一个 $M/M/c$ 系统。虽然真实系统不一定满足泊松到达与指数服务时间，但这个模型足够解释趋势：当平均到达率逼近总处理率时，等待时间不是线性变差，而是陡增。

Erlang C 的等待概率可写为：

$$
P_w = C(c,a)=\frac{a^c}{c!(1-\rho)}P_0
$$

其中 $P_0$ 是系统空闲概率，$a=\lambda/\mu$。白话讲，$P_w$ 越高，说明“新请求一来就得排队”的概率越高。

这说明一个常被忽略的事实：只要 $\rho$ 太接近 1，任何局部优化都会失效。你可以换更聪明的路由，但如果系统本身只剩很小余量，排队仍然会炸。于是工程上需要同时做三层控制：

1. 调度层：把请求送到更合适的 Agent。
2. 准入层：当队列过深时拒绝、降级或排优先级。
3. 容量层：及时扩容 Agent 实例或更多模型副本。

“最少未完成请求”适合处理长短任务混合。它不是看谁历史性能最好，而是看“谁现在最不忙”。P2C，power of two choices，意思是随机挑两个候选，再选更空闲的那个。它的优点是不用全局扫描全部节点，也能显著减少热点。

“能力路由”适合异构池。异构池指池中实例能力不同，白话讲就是“不是每台机器都一样”。例如：

- 8B 模型节点：适合短问答、分类、改写
- 32B 模型节点：适合复杂推理、长上下文审阅
- 带工具缓存的节点：适合同租户重复查询

这时最优策略通常不是统一池内抢活，而是先按能力分池，再在池内做负载感知。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它把“能力权重”“当前负载”“冷启动惩罚”合到一起，先做加权抽样，再在超载时回退到最少未完成请求。

```python
from dataclasses import dataclass
from random import Random

@dataclass
class Agent:
    name: str
    weight: int
    outstanding: int
    healthy: bool = True
    warm: bool = True
    max_outstanding: int = 5

def effective_weight(agent: Agent) -> int:
    if not agent.healthy:
        return 0
    # 冷实例先降权，避免一启动就被打满
    warm_factor = 1.0 if agent.warm else 0.3
    # outstanding 越高，动态权重越低
    load_factor = max(0.1, 1 - agent.outstanding / max(1, agent.max_outstanding))
    return max(0, int(agent.weight * warm_factor * load_factor * 10))

def select_agent(agents, rnd=None):
    rnd = rnd or Random(0)
    candidates = [a for a in agents if a.healthy and a.outstanding < a.max_outstanding]
    if not candidates:
        return min([a for a in agents if a.healthy], key=lambda x: x.outstanding)

    scored = [(a, effective_weight(a)) for a in candidates]
    total = sum(w for _, w in scored)

    if total == 0:
        return min(candidates, key=lambda x: x.outstanding)

    pick = rnd.randint(1, total)
    acc = 0
    for agent, w in scored:
        acc += w
        if pick <= acc:
            return agent

    return min(candidates, key=lambda x: x.outstanding)

# 玩具例子：A 强，B/C 弱；B 是冷实例
agents = [
    Agent(name="A", weight=3, outstanding=1, warm=True),
    Agent(name="B", weight=1, outstanding=0, warm=False),
    Agent(name="C", weight=1, outstanding=2, warm=True),
]

selected = select_agent(agents, Random(42))
assert selected.name in {"A", "C"}

# 超载回退：所有节点都满时，选 outstanding 最小者
agents2 = [
    Agent(name="A", weight=3, outstanding=5, max_outstanding=5),
    Agent(name="B", weight=1, outstanding=4, max_outstanding=4),
    Agent(name="C", weight=1, outstanding=6, max_outstanding=6),
]
selected2 = select_agent(agents2, Random(1))
assert selected2.name == "B"

print(selected.name, selected2.name)
```

这个实现故意保持简单，但已经体现三个关键点：

1. 权重不是静态常量，而是会被健康状态、冷启动状态、当前负载修正。
2. 节点一旦接近上限，不应继续按原权重接流量。
3. 没有合格候选时，必须有回退策略，而不是返回空。

真实工程例子可以这样落地：

- 第一层，入口网关按租户、任务类型、区域先路由到不同 Agent 池。
- 第二层，池内调度器读取每个 Agent 的 `outstanding`、`queue_depth`、`p95`、`context_tokens`。
- 第三层，扩缩容控制器根据过去 1 到 5 分钟的排队与延迟趋势增减副本。
- 第四层，预热器提前拉起模型、准备连接池、加载常用提示模板和缓存。

如果你在 Kubernetes 上部署，可以把 `outstanding` 和 `queue_depth` 暴露成自定义指标，让 HPA 或自定义控制器触发水平扩容。水平扩容指“增加更多同类实例”，白话讲就是“多开几个干活的副本”。

---

## 工程权衡与常见坑

最常见的错误，不是算法太差，而是观测信号太粗。

| 坏情况 | 典型信号 | 对策 |
| --- | --- | --- |
| 静态权重长期不变 | 某强节点 p99 飙升，其他节点空闲 | 用 p95、队列深度、热状态动态调权 |
| 只看 CPU 或平均延迟 | 平均值正常，但用户仍感到卡 | 盯住 p95/p99 与 `outstanding` |
| 冷实例一上线就吃满流量 | 冷启动率高、首个响应很慢 | slow start，逐步加权放量 |
| 区域故障后全量切流 | 第二个区域也被打爆 | 故障转移要配 admission control |
| 长任务和短任务混跑 | 短任务也被拖慢 | 分池或按任务重量分级队列 |

几个坑需要单独强调。

第一，平均延迟会骗人。多 Agent 系统的体验通常败在 p99，而不是平均值。一个节点挂着几个超长任务，就足以把整条链路的尾部拖死。

第二，静态轮询会天然偏向“请求数平均”，但工程真正关心的是“资源占用平均”。如果一个任务 50 token，另一个任务 5000 token，二者不能视作同一重量。

第三，冷启动不只是拉容器。对 Agent 系统来说，冷启动可能包含模型加载、网络连接建立、工具认证、缓存填充、提示模板预处理。很多系统扩容动作已经发生了，但流量一到还是慢，原因就在这里。

第四，扩容触发不要只看 CPU。对 LLM Agent，真正接近用户体验的信号往往是 `queue_wait_ms`、TTFT、KV cache 压力、上下文长度分布，而不是单纯 CPU 使用率。

第五，重试可能制造二次灾难。Agent 系统里一个上游超时，可能触发多层重试，结果把下游进一步打爆。所以重试必须有预算，且副作用操作不能随便做 hedging。hedging 指“主请求慢时，再发一个备份请求争取更快返回”。

---

## 替代方案与适用边界

如果你的系统只是几个轻量 Agent 调同一模型，简单的“加权轮询 + 最少未完成请求 + 基于队列的 HPA”通常就够了。复杂调度不是免费收益，它会带来更多状态同步、指标采集和控制面复杂度。

但当系统进入以下区间，替代方案会更有价值：

| 方案 | 适用场景 | 代价 |
| --- | --- | --- |
| 一般负载感知路由 | 中等并发，任务长短差异明显 | 实现简单，收益稳定 |
| 能力分池 + 池内最少待办 | 异构模型、多任务类型 | 需要维护分类规则 |
| Workflow-aware 调度 | 多 Agent 串联、链路步骤差异大 | 需要理解工作流剩余时延 |
| Memory-aware 调度 | 共享 LLM、显存压力大 | 需要更细粒度的资源画像 |
| Token-aware 公平队列 | 多租户共享推理池 | 配置复杂，但公平性更好 |

Kairos 这类方案更进一步。它不是只看“哪个节点更空”，而是理解“这个请求还剩多少链路延迟、会吃多少内存”，再决定优先级和落点。对多 Agent 共享 LLM 的场景，这比普通轮询更符合真实瓶颈，因此论文报告了 17.8% 到 28.4% 的端到端延迟改善。

但它的适用边界也很明确：

1. 你需要足够精细的工作流信息，知道请求处于哪一步。
2. 你需要能估计不同请求的内存需求，否则 memory-aware 只是口号。
3. 你的负载真的复杂到值得这套控制面，否则会过度设计。

所以实践上可以按层升级：

1. 先做加权轮询。
2. 再加最少未完成请求或 P2C。
3. 再加按任务重量、上下文长度、区域配额的能力路由。
4. 最后才考虑 workflow-aware 或 memory-aware 调度。

这个升级顺序的理由很简单：前两层解决 80% 的热点和积压问题，后两层解决更深的异构与共享资源问题。

---

## 参考资料

- AceCloud，*Agentic AI Load Balancing: Reduce P99 Latency And Control GPU Spend*：<https://acecloud.ai/blog/agentic-ai-load-balancing/>
- ShShell，*The Agent Traffic Jam: Load Balancing*，发布于 2026-01-05：<https://www.shshell.com/blog/ai-agents-module-18-lesson-3-load-balancing>
- Kairos 论文摘要页，*Kairos: Low-latency Multi-Agent Serving with Shared LLMs and Excessive Loads in the Public Cloud*，ArXiv 2025，编号 `2508.06948`：<https://arxivlens.com/paperview/details/kairos-low-latency-multi-agent-serving-with-shared-llms-and-excessive-loads-in-the-public-cloud-3727-4c4777ea>
- Richard E. Newman 相关排队论讲义与 M/M/c、Erlang C 公式说明，可参考教材化整理版 PDF：<https://irh.inf.unideb.hu/user/jsztrik/education/16/Queueing_Problems_Solutions_2021_Sztrik.pdf>
- M/M/c queue 条目，作为公式与符号对照入口：<https://en.wikipedia.org/wiki/M/M/c_queue>
