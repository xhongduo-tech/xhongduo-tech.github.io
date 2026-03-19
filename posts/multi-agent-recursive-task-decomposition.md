## 核心结论

多Agent任务递解的递归细化策略，本质上是把一个复杂目标表示成一棵任务树。规划器先给出较粗的主分支，再按需要继续拆分，直到每个叶子任务都满足“单个agent能独立完成”的条件。这里的“递归细化”可以理解成：如果一块任务还太大、太模糊、依赖还不清楚，就继续往下拆；如果已经小到可执行，就停止拆分并派发。

这套策略的价值不在“拆得越细越好”，而在“只把必要的部分拆细”。因此它通常结合两类搜索方式：

| 策略 | 直观解释 | 优点 | 代价 | 适合场景 |
|---|---|---|---|---|
| 深度优先分解（DFS-like） | 先沿一条分支一路拆到底 | 更快找到可执行叶子任务 | 容易忽略全局最优 | 强依赖链、需要尽快落地执行 |
| 广度优先分解（BFS-like） | 先把同层候选都看一遍 | 全局视角更稳 | token消耗更高 | 需要比较多个方案质量 |
| 自适应粒度控制 | 只细化“不够清楚”的分支 | 成本和精度更均衡 | 需要额外评价器 | 真实工程中的默认选择 |

玩具例子可以直接看旅行规划。主任务是“准备日本旅行”，规划器先生成“签证、机票、酒店”三个节点。若“机票”和“酒店”已经足够明确，可以直接交给预订agent；若“签证”仍包含材料准备、预约、时间窗口等不确定性，就只继续细化“签证”分支。粗粒度节省token，细粒度提高决策精度，这就是递归细化的核心收益。

粗粒度与细粒度的差异可以明确写成表：

| 粒度 | token成本 | 决策精度 | 记忆触发条件 |
|---|---:|---:|---|
| 粗粒度 | 低 | 中 | 结果出现冲突或信息缺失时再补拆 |
| 中粒度 | 中 | 较高 | 某分支评分下降或依赖不清时触发 |
| 细粒度 | 高 | 高 | 高风险任务、强约束依赖、单步错误代价大 |

Tree-of-Thought强调“树状搜索与状态评分”，Planner+Agents强调“规划与执行分离”。前者提供如何扩展和筛选候选分支，后者提供如何把叶子任务交给合适的执行agent。两者结合后，规划器负责“拆与选”，agent负责“做与回写”。

---

## 问题定义与边界

“任务树”是把总目标拆成若干子目标的层级结构。“前沿状态集”指当前准备继续扩展的一组节点，也就是“下一轮最值得继续拆的任务块”。记为 $S_t$，表示第 $t$ 轮递归时保留下来的候选集合。常见的筛选公式是：

$$
S_t=\arg\max_{|S|=b}\sum_{s\in S}V_t(s)
$$

其中，$V_t(s)$ 是状态 $s$ 在第 $t$ 轮的价值评分，$b$ 是本轮保留的分支数。白话解释是：每轮先生成一些候选子任务，再从中保留最有价值的 $b$ 个继续展开。

边界问题比“怎么拆”更重要，因为如果没有明确的终止条件，任务树会无限膨胀。这里通常有三个收口条件，只要满足任意一个，就停止继续分解：

| 终止条件 | 含义 | 新手理解 |
|---|---|---|
| token阈值 $ \le \tau $ | 子任务上下文足够小 | 任务已经短到一个agent看得完 |
| attention sufficiency $ \ge \sigma $ | 注意力信号足够集中 | 关键信息已经够明确，不必再拆 |
| agent可行性成立 | 某个agent能力完全匹配 | 已经能直接派给专人处理 |

可以把这件事想成搭积木。规划器每次问三个问题：这块积木是不是还太大？是不是还不够清楚？是不是已经有人能独立拿走处理？若答案分别是否、否、是，就应该终止。

形式化一点，可以把每个状态写成：

$$
s=(g, c, d, m)
$$

其中 $g$ 是当前子目标，$c$ 是上下文摘要，$d$ 是依赖集合，$m$ 是记忆摘要。是否继续递归，不只看目标本身，还要看上下文预算、依赖是否已显式建模、历史尝试是否失败过。

---

## 核心机制与推导

递归细化一般包含四步：候选生成、价值评分、前沿筛选、终止判定。

第一步是候选生成。对于一个待细化节点 $s_t$，规划器生成 $k$ 个候选拆分方案：

$$
\text{Expand}(s_t) \rightarrow \{s_t^{(1)}, s_t^{(2)}, \dots, s_t^{(k)}\}
$$

第二步是价值评分。评分函数 $V_t(s)$ 至少应综合四个因素：完成价值、依赖清晰度、预估成本、agent匹配度。一个简化写法是：

$$
V_t(s)=\alpha Q(s)+\beta D(s)+\gamma A(s)-\delta C(s)
$$

其中：
- $Q(s)$：该子任务对总目标的贡献度
- $D(s)$：依赖是否已明确
- $A(s)$：是否存在匹配agent
- $C(s)$：预计token与调用成本

第三步是筛选前沿状态。若本轮扩展后得到很多候选，就按 $V_t(s)$ 排序，只保留 top-$b$。这一步对应 Tree-of-Thought 中的“搜索剪枝”，避免所有分支都展开导致成本爆炸。

第四步是终止判定。可写成：

$$
\text{Stop}(s)=\mathbf{1}[T(s)\le\tau \;\lor\; \Sigma(s)\ge\sigma \;\lor\; F(s)=1]
$$

其中：
- $T(s)$：子任务token估计
- $\Sigma(s)$：attention sufficiency，表示当前上下文是否足以支持决策
- $F(s)$：feasibility，表示是否存在单agent可独立完成

旅行规划可以直接代入。假设第一层拆出三条路径：签证、机票、酒店。它们的token估计分别为 220、140、100，且阈值 $\tau=150$。那么机票和酒店已经小于阈值，可以直接派发；签证仍大于阈值，需要继续细化。再把签证拆成“材料清单”和“预约流程”，若分别为 90 和 80，则两者都可停止分解。

伪代码如下：

```text
refine(state):
    candidates = expand(state)
    scored = rank_by_value(candidates)
    frontier = top_b(scored)

    for sub in frontier:
        if token_estimate(sub) <= tau
           or attention_sufficient(sub) >= sigma
           or single_agent_feasible(sub):
            dispatch(sub)
        else:
            memory.write(compress(sub))
            refine(sub)
```

这里的 `memory.write` 是关键。记忆不是简单日志，而是压缩后的状态表示，例如“已确定签证需先于机票支付”“酒店仅缺地理约束”。规划器下一轮不需要重新阅读全部历史，只需读取摘要。

---

## 代码实现

下面给出一个可运行的 Python 玩具实现。它没有接入真实LLM，但把递归细化、评分、终止、记忆写回这些核心部件都保留了。

```python
from dataclasses import dataclass, field

TAU = 150
SIGMA = 0.75

@dataclass
class Task:
    name: str
    tokens: int
    attention: float
    feasible_agents: list[str]
    deps: list[str] = field(default_factory=list)
    children: list["Task"] = field(default_factory=list)

def value(task: Task) -> float:
    # 贡献度这里用 tokens 的反比近似，真实系统会接入模型评分器
    quality = 1.0 / max(task.tokens, 1)
    dep_score = 1.0 if task.deps else 0.6
    agent_score = 1.0 if task.feasible_agents else 0.0
    cost = task.tokens / 300.0
    return 2.0 * quality + 1.2 * dep_score + 1.5 * agent_score - 0.8 * cost

def stop(task: Task) -> bool:
    return (
        task.tokens <= TAU
        or task.attention >= SIGMA
        or len(task.feasible_agents) > 0
    )

def compress(task: Task) -> str:
    return f"{task.name}|tokens={task.tokens}|deps={','.join(task.deps) or 'none'}"

def refine(task: Task, memory: list[str], out: list[str]) -> None:
    if stop(task):
        out.append(f"EXEC:{task.name}")
        return

    ranked = sorted(task.children, key=value, reverse=True)[:2]
    for child in ranked:
        memory.append(compress(child))
        refine(child, memory, out)

# 玩具例子：旅行规划
visa_docs = Task("准备签证材料", 90, 0.6, ["doc_agent"], ["签证"])
visa_appointment = Task("预约递签时间", 80, 0.7, ["booking_agent"], ["签证"])
visa = Task("签证", 220, 0.4, [], [], [visa_docs, visa_appointment])

flight = Task("机票", 140, 0.8, ["flight_agent"])
hotel = Task("酒店", 100, 0.85, ["hotel_agent"])

root = Task("日本旅行", 600, 0.3, [], [], [visa, flight, hotel])

memory = []
out = []
refine(root, memory, out)

assert "EXEC:准备签证材料" in out
assert "EXEC:预约递签时间" in out
assert len(memory) >= 2
print(out)
```

这段代码体现了三个工程点。

第一，`stop(task)` 同时检查 token、attention 和 agent 可行性，符合递归终止的联合判定逻辑。

第二，`value(task)` 把“值得继续拆分还是值得直接执行”转成一个分数。真实系统里，`quality` 往往来自LLM评估器或规则评估器；`dep_score` 可由DAG约束完备度决定；`agent_score` 来自agent注册表匹配。

第三，`compress(task)` 是 planner memory 的写回点。真实工程中，写回的数据结构可能是摘要字符串、结构化JSON，或向量索引项。摘要的目的不是存档，而是为下一轮规划保留最少但足够的信息。

真实工程例子可以看旅行规划基准与任务依赖图系统。比如 ItineraryBench 这类任务中，规划器会持续拆出签证、交通、住宿、本地通勤等子任务，并根据依赖推理出“签证未确认前，不应锁定不可退款机票”。再如 Minecraft 中的 VillagerAgent，若目标是“建房并准备资源”，系统会显式记录“先建家，再挖矿，再扩建仓储”的DAG；如果不建依赖图，agent可能先去挖矿，导致后续资源、位置、安全约束全部错位。

---

## 工程权衡与常见坑

递归细化不是越复杂越好，真正困难的是控制“拆分收益”与“额外成本”的平衡。

最常见的第一个坑是过度拆分。表面上看，拆得更细会让每个agent更容易执行；实际上，每多一层就多一次规划、多一次评分、多一次记忆写回，token和调用成本会快速上升。可以把这种趋势理解成近似凸增：早期拆分提升明显，后期继续细化的收益下降，但成本持续上升。

第二个坑是依赖错位。子任务如果没有显式前后关系，系统只会把它们当作一组并行块。这在旅行规划里会表现为“还没确认签证就先支付不可退机票”，在代码代理里会表现为“测试还没补齐就先合并”，在 Minecraft agent 里会表现为“房子还没建完就先挖远程矿”。

第三个坑是记忆膨胀。每轮都把完整上下文传下去，会让后续节点越来越重；而如果摘要过度压缩，又会丢失依赖信号。因此记忆系统通常要区分“执行所需上下文”和“规划所需上下文”，前者面向agent，后者面向planner。

下表是典型风险与缓解方式：

| 风险 | 表现 | 缓解手段 |
|---|---|---|
| 过度拆分 | 调用次数爆炸、延迟升高 | 用状态触发粒度调整，只细化低置信分支 |
| 依赖错位 | 顺序颠倒、返工增多 | 用DAG显式记录前后与资源依赖 |
| 记忆膨胀 | 上下文越来越长 | 写回压缩摘要，按层裁剪历史 |
| agent误匹配 | 分给错误执行者 | 维护能力注册表和失败回退策略 |
| 局部最优 | 早早深入错误分支 | 混合BFS/DFS，保留少量探索分支 |

若把“拆分粒度”作为横轴，“token成本”作为纵轴，曲线通常不是线性，而是越往细粒度端增长越快。这意味着工程上应该优先找“足够小而不是最小”的任务单元。

---

## 替代方案与适用边界

递归细化不是唯一方案。另一类常见做法是一次性全局规划，也就是先把整棵树尽量展开，再统一排序和派发。两者的差别可以直接比较：

| 方案 | 成本 | 适配任务规模 | 依赖管理 | 并行性 | 稳定性 |
|---|---|---|---|---|---|
| 递归细化 | 按需付费，通常更省 | 中大任务 | 强，适合动态补依赖 | 高 | 较稳 |
| 单块规划 | 前期集中消耗 | 小任务 | 弱，依赖容易漏 | 中 | 取决于一次规划质量 |

对新手可以这样理解：小任务像“一次写完整个旅行计划”，直接整体展开更快；大任务像“跨国出行加预算限制加多人协同”，一次展开会很长、很乱，也不利于后续修改，这时递归拆分更合适。

适合递归细化的边界条件通常有三类：

| 边界条件 | 为什么适合递归 |
|---|---|
| 多agent并行能力存在 | 拆出来的叶子任务可同时执行 |
| token预算紧张 | 无法把全局上下文一次塞进单轮推理 |
| 依赖关系明显 | 需要逐步显式建模和校验顺序 |

相反，如果任务规模很小、依赖很弱、且实时性极强，例如“给出一个简单三日游建议”，则粗粒度方案甚至单块规划更合适。因为规划时间本身也是成本。在 token 足够但延迟敏感的窗口里，常见策略是“粗粒度规划 + 策略缓存”，而不是继续递归。

ItineraryBench 这类多agent任务，与传统单体LLM规划的核心差别就在这里：前者默认任务会继续变化、依赖会暴露、执行会回写，因此更需要动态任务树；后者更像“一次性生成答案”，适合静态、短链路问题。

---

## 参考资料

1. Tree-of-Thought 相关资料：用于理解树状搜索、BFS/DFS扩展、状态评分与剪枝公式。若想先看 $S_t$ 与 $V_t(s)$ 的来源，应先读这一类资料。
2. LLM Planner Agent 相关资料：用于理解“Planner+Agents”架构、递归细化、自适应粒度控制与记忆反馈。
3. Dynamic Context Cutoff / attention sufficiency 相关资料：用于理解何时停止继续拆分，重点看“上下文已经足够”的判据，而不是只看 token 长度。
4. Token Budgeting 相关资料：用于理解层级预算分配、不同深度节点的 token 控制方法，以及如何避免上下文窗口被单一路径耗尽。
5. TDAG / ItineraryBench 相关资料：用于看旅行规划这类真实工程任务中，如何把签证、交通、住宿等拆成可执行图，并处理依赖关系。
6. VillagerAgent 一类的任务依赖图实践：用于理解在开放世界环境中，为什么必须显式记录“先建家、再采矿、再扩建”这样的前后约束。

建议阅读顺序是：先读 Tree-of-Thought 的评分与搜索机制，再读 Planner Agent 的架构，再结合 Token Budgeting 理解预算控制，最后看 ItineraryBench 和任务依赖图案例，观察这些机制如何落到真实系统里。
