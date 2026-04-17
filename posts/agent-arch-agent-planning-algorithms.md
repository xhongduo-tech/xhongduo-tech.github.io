## 核心结论

Agent 的规划算法，本质上是在回答一个严格的问题：给定当前世界状态、可执行动作和目标，怎样构造一条既能达成目标、又不违反资源与顺序约束的行动方案。

这里的“规划”不是简单列步骤，而是构造一个**部分有序计划**。部分有序计划的意思是：只固定那些必须固定的先后关系，不必要的顺序先不锁死。这样做的好处是，计划既保留因果正确性，也保留并行执行和后续调整的自由度。

可以把经典规划问题先写成一个最小形式：

$$
P = (F, A, I, G)
$$

其中：

- $F$ 是 fluent 集合。fluent 可以理解为“会随动作变化的事实状态”。
- $A$ 是动作集合，每个动作带有前提和效果。
- $I$ 是初始状态。
- $G$ 是目标状态。

而规划器输出的，不一定是唯一线性序列，更常见的是“动作 + 约束”的结构。一个更贴近部分有序规划的写法是：

$$
\Pi = (S, \prec, L)
$$

其中：

- $S$ 是计划中的动作步骤集合。
- $\prec$ 是部分顺序关系，表示哪些动作必须在另一些动作之前。
- $L$ 是因果链集合，每条链说明“哪个动作为哪个后继动作提供了什么前提”。

所以，规划器真正维护的不是“第 1 步、第 2 步、第 3 步”这种死顺序，而是如下结构：

`开始 -> 取邮件 -> 结束`  
`开始 -> 送咖啡 -> 结束`

如果两个动作之间没有硬依赖，就不必强行规定谁先谁后。只有在因果上必须如此时，顺序才会被锁死。

一个玩具例子最容易说明这一点。假设办公室里有两个任务：给 Sam 送咖啡、去前台取邮件。一个新手常见误解是“规划就是把所有动作排成死顺序”。其实不是。规划算法只会锁定必要因果链。比如 `notify_done` 这个动作必须等到“咖啡已送达”和“邮件已取回”都成立后才能执行，那么这两条依赖必须固定；但“取邮件”和“开走廊灯”如果彼此无关，就没必要锁死顺序。

在多 Agent 架构里，规划器不是单纯的路径生成器，而是**高层意图与执行资源之间的协调器**。它要同时考虑：

- 目标能不能达成
- 成本是否可接受
- 人、机器人、工具、时间窗等资源是否冲突
- 是否需要保留后续调整空间

所以，规划算法成立的核心原因，不是“它会搜索”，而是“它把因果、约束和可执行性放进了同一个结构里”。

---

## 问题定义与边界

先把问题定义清楚。经典规划问题通常写成：

$$
P = (F, A, I, G)
$$

目标是找到动作序列或部分有序计划 $\Pi$，使得从初始状态 $I$ 出发执行后，可以到达某个状态 $s'$，并满足：

$$
G \subseteq s'
$$

也常写成状态转移函数：

$$
\Gamma(s, \pi)=s'
$$

白话解释：$\Gamma$ 表示“把计划 $\pi$ 从状态 $s$ 上执行一遍，最后会走到哪个状态”。

如果进一步写出动作语义，通常记为：

$$
a = (pre(a), add(a), del(a), cost(a))
$$

含义是：动作 $a$ 执行前需要满足 $pre(a)$，执行后会新增 $add(a)$ 中的事实，删除 $del(a)$ 中的事实，并带来某种代价 $cost(a)$。

核心元素如下表。

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $F$ | fluent 集合 | 所有可能变化的事实，比如“邮件在等待”“咖啡还没送到” |
| $A$ | 动作集合 | 系统允许执行的操作 |
| $I$ | 初始状态 | 一开始哪些事实为真 |
| $G$ | 目标集合 | 最终希望哪些事实为真 |
| $pre(a)$ | 动作前提 | 执行动作前必须满足的条件 |
| $add(a)$ | 添加效果 | 动作执行后变为真的事实 |
| $del(a)$ | 删除效果 | 动作执行后变为假的事实 |
| $cost(a)$ | 动作代价 | 时间、金钱、风险或资源占用 |

继续用“咖啡 + 邮件”这个新手版例子。

设：

- 初始状态  
  $I = \{wants\_coffee,\ mail\_waiting\}$
- 目标  
  $G = \{all\_done\}$

动作定义如下：

| 动作 | pre | add | del | cost |
|---|---|---|---|---|
| `deliver_coffee` | `{wants_coffee}` | `{coffee_delivered}` | `{wants_coffee}` | 2 |
| `fetch_mail` | `{mail_waiting}` | `{mail_fetched}` | `{mail_waiting}` | 1 |
| `notify_done` | `{coffee_delivered, mail_fetched}` | `{all_done}` | `{}` | 1 |

这个例子里：

- `deliver_coffee` 负责把“想要咖啡”变成“咖啡已送达”
- `fetch_mail` 负责把“邮件在等待”变成“邮件已取回”
- `notify_done` 依赖前两个结果都成立，才能宣布任务完成

对于新手，最容易混淆的点有两个：

| 误解 | 实际情况 |
|---|---|
| 目标必须直接等于最终状态 | 不需要，只要目标集合 $G$ 被最终状态包含即可 |
| 只要每个动作都能执行，计划就正确 | 不够，还要保证动作之间不会互相破坏前提 |

边界在哪里？主要有三类。

| 边界类型 | 含义 | 影响 |
|---|---|---|
| 顺序依赖 | 一个动作必须在另一个动作之后 | 决定哪些边必须锁死 |
| 资源冲突 | 两个动作同时争抢同一资源 | 决定哪些动作不能并行 |
| 并行能力 | 两个动作互不干扰时可并行 | 决定计划是否需要完全线性化 |

部分有序计划的价值，正是在这个边界上体现。它不会默认“所有动作都排队执行”，而是只在必要处加约束。这样做对真实系统尤其重要，因为真实系统里执行时间不稳定、外部事件会扰动，留出局部灵活性通常比一次性排死所有顺序更稳健。

---

## 核心机制与推导

规划结构不只是“动作列表”。更准确地说，它由四部分组成：

1. 动作实例集合
2. 部分顺序关系 $\prec$
3. 因果链
4. 开放前提（open preconditions）

其中，**因果链**是核心。因果链形如：

$$
a_i \xrightarrow{p} a_j
$$

意思是：动作 $a_i$ 产生事实 $p$，动作 $a_j$ 需要这个事实作为前提。白话说，就是“前者给后者供条件”。

开放前提指的是：某个动作需要的条件，现在还没人负责提供。规划搜索的过程，本质上就是不断消除开放前提。

### 1. 从开放前提开始搜索

假设当前计划里已经放入目标动作 `notify_done`，它有两个前提：

- `coffee_delivered`
- `mail_fetched`

这两个前提一开始都是开放的。于是规划器会去找谁能提供它们：

- `deliver_coffee` 提供 `coffee_delivered`
- `fetch_mail` 提供 `mail_fetched`

于是得到两条因果链：

$$
deliver\_coffee \xrightarrow{coffee\_delivered} notify\_done
$$

$$
fetch\_mail \xrightarrow{mail\_fetched} notify\_done
$$

然后再检查 `deliver_coffee` 和 `fetch_mail` 自己的前提是否满足：

- `deliver_coffee` 需要 `wants_coffee`
- `fetch_mail` 需要 `mail_waiting`

如果这两个事实已经在初始状态里成立，那么开放前提被消除；如果没有，就要继续找更早的动作来支持它们。

这一过程可以写成递归式：

$$
Open(\Pi)=\{\,p \mid \exists a\in S,\ p\in pre(a),\ p\text{ 尚未被任何因果链支持}\,\}
$$

规划的推进过程，就是不断选择一个开放前提 $p$，再选择一个能实现它的动作 $a$ 或者初始状态事实来关闭它。

### 2. 非确定性来自“谁来实现”和“顺序怎么插”

规划搜索有两个核心自由度。

第一，**选择实现者**。  
如果一个前提能由多个动作提供，规划器要选哪一个。

第二，**插入顺序**。  
动作加入后，要放在计划图中的什么位置，既满足前提，又不破坏已有因果链。

这也是为什么规划问题会迅速变难。难点不在“执行动作”，而在“为每个前提选择实现者，同时避免冲突”。

下面这个表更适合新手理解搜索难点。

| 决策点 | 要回答的问题 | 如果答错会怎样 |
|---|---|---|
| 选哪个动作支持目标 | 谁来产出所需事实 | 可能成本过高，甚至后续无解 |
| 把动作插在哪 | 动作要处于哪些前驱和后继之间 | 可能破坏已有前提 |
| 是否允许并行 | 两个动作会不会互相干扰 | 可能出现资源冲突或状态冲突 |
| 何时重规划 | 旧计划还是否可信 | 可能继续执行已经失效的计划 |

### 3. 因果链为什么需要保护

如果已经建立：

$$
fetch\_mail \xrightarrow{mail\_fetched} notify\_done
$$

那么任何位于二者之间、并且会删除 `mail_fetched` 的动作，都会构成威胁。规划器必须处理这个威胁。典型方法有三种：

| 方法 | 含义 | 白话解释 |
|---|---|---|
| Promotion | 把威胁动作放到因果链后面 | 让它来不及破坏 |
| Demotion | 把威胁动作放到因果链前面 | 让它先执行完 |
| Separation | 加资源或变量约束使其不影响该事实 | 从定义上拆开冲突 |

这一步叫**冲突消解**。如果不做，计划表面上看成立，执行时却会失效。

把这个条件写成公式会更清楚。对于一条因果链：

$$
a_i \xrightarrow{p} a_j
$$

若存在动作 $a_t$ 满足：

$$
p \in del(a_t)
$$

并且它在顺序上可能落在 $a_i$ 和 $a_j$ 中间，即：

$$
a_i \prec a_t \prec a_j
$$

那么 $a_t$ 就是这条因果链的威胁动作。规划器必须增加新的顺序约束，或者改写变量绑定，使这个威胁消失。

### 4. 玩具例子：因果链如何防止中途破坏

假设新增一个动作：

| 动作 | pre | add | del |
|---|---|---|---|
| `discard_mail` | `{mail_waiting}` | `{}` | `{mail_waiting}` |

如果系统错误地在 `fetch_mail` 之前插入 `discard_mail`，那么 `fetch_mail` 的前提 `mail_waiting` 就会被删掉，后续动作无法执行。

所以计划图上需要显式保护：

$$
Start \xrightarrow{mail\_waiting} fetch\_mail
$$

并要求所有会删除 `mail_waiting` 的动作，不能落在这条链的中间。

这正是部分有序规划的关键直觉：不是先把所有动作排成线，而是先把“谁支持谁、谁不能插进来破坏”表达清楚。

### 5. 真实工程例子：仓储中的多角色协同

在仓储系统里，一个订单可能涉及：

- 机器人搬箱
- 人工复核
- 司机装车

如果系统把这些子任务分别交给各自模块，再由上层合并计划，那么真正困难的点不在“各自会不会做”，而在“合并后有没有冲突”。

例如：

- 机器人占用了唯一升降台
- 人工复核需要货物先到缓冲区
- 司机装车有发车时间窗

这时局部计划各自都可能正确，但合并后会在共享资源上打架。于是上层规划器需要维护部分顺序与资源约束，例如：

`机器人搬运 -> 人工复核 -> 司机装车`

但如果另一条订单链与此无共享资源，则没必要锁死相对顺序，可以并行。

这就是为什么真实系统里常常先做局部规划，再做 plan merging，也就是“计划合并”。它本质上是在更高层继续做因果保护和冲突检测。

### 6. 一个新手常见问题：为什么不直接用拓扑排序

很多人第一次接触规划时会问：既然动作之间有依赖，为什么不直接建图然后做一次拓扑排序？

原因是，拓扑排序只能解决**依赖已经明确给定**的问题；而规划要先回答“依赖到底是什么、谁来满足它、有没有别的动作会破坏它”。也就是说：

| 问题 | 拓扑排序 | 规划 |
|---|---|---|
| 已知动作先后关系后排顺序 | 能做 | 能做 |
| 目标前提由谁提供 | 不能做 | 能做 |
| 动作会不会删掉别人需要的状态 | 不能做 | 能做 |
| 资源冲突和时间窗约束 | 不能直接处理 | 可以纳入约束模型 |

所以，拓扑排序往往只是规划器的一个后处理步骤，用来把“已经求出的部分顺序计划”线性化为某条可执行顺序，而不是规划问题本身。

---

## 代码实现

下面给出一个最小可运行的 Python 版本。它不是工业级规划器，但能展示四个关键点：

1. 用 `pre/add/delete` 表示动作
2. 从目标反推开放前提
3. 自动补上依赖动作
4. 线性化后验证计划确实可执行

```python
from __future__ import annotations

from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Dict, FrozenSet, Iterable, List, Set


@dataclass(frozen=True)
class Action:
    name: str
    pre: FrozenSet[str]
    add: FrozenSet[str]
    delete: FrozenSet[str]
    cost: int = 1


def apply_action(state: Set[str], action: Action) -> Set[str]:
    if not action.pre.issubset(state):
        missing = sorted(action.pre - state)
        raise ValueError(f"{action.name} precondition failed: missing {missing}")
    new_state = set(state)
    new_state.difference_update(action.delete)
    new_state.update(action.add)
    return new_state


def execute_plan(initial: Iterable[str], plan: List[Action]) -> Set[str]:
    state = set(initial)
    for action in plan:
        state = apply_action(state, action)
    return state


def choose_provider(goal: str, actions: List[Action]) -> Action:
    candidates = [a for a in actions if goal in a.add]
    if not candidates:
        raise ValueError(f"no action can achieve goal: {goal}")
    # 玩具策略：优先选 cost 更小、名字更稳定的动作
    return min(candidates, key=lambda a: (a.cost, a.name))


def backward_plan(initial: Set[str], goals: Set[str], actions: List[Action]) -> List[Action]:
    selected: Dict[str, Action] = {}
    # dependency[x] = {y1, y2} 表示动作 x 依赖 y1、y2 先执行
    dependency: Dict[str, Set[str]] = {}

    def achieve(fact: str, stack: Set[str]) -> str | None:
        if fact in initial:
            return None
        if fact in stack:
            raise ValueError(f"cyclic requirement detected on fact: {fact}")

        provider = choose_provider(fact, actions)
        selected[provider.name] = provider
        dependency.setdefault(provider.name, set())

        next_stack = set(stack)
        next_stack.add(fact)

        for pre_fact in sorted(provider.pre):
            pre_provider_name = achieve(pre_fact, next_stack)
            if pre_provider_name is not None:
                dependency[provider.name].add(pre_provider_name)

        return provider.name

    goal_actions = []
    for goal in sorted(goals):
        goal_provider = achieve(goal, set())
        if goal_provider is not None:
            goal_actions.append(goal_provider)

    # graphlib.TopologicalSorter 需要“节点 -> 其前驱集合”
    ts = TopologicalSorter()
    for name in selected:
        ts.add(name, *dependency.get(name, set()))

    ordered_names = list(ts.static_order())
    ordered_actions = [selected[name] for name in ordered_names]

    final_state = execute_plan(initial, ordered_actions)
    if not goals.issubset(final_state):
        raise ValueError(
            f"plan does not reach goals. missing goals: {sorted(goals - final_state)}"
        )
    return ordered_actions


if __name__ == "__main__":
    actions = [
        Action(
            name="deliver_coffee",
            pre=frozenset({"wants_coffee"}),
            add=frozenset({"coffee_delivered"}),
            delete=frozenset({"wants_coffee"}),
            cost=2,
        ),
        Action(
            name="fetch_mail",
            pre=frozenset({"mail_waiting"}),
            add=frozenset({"mail_fetched"}),
            delete=frozenset({"mail_waiting"}),
            cost=1,
        ),
        Action(
            name="notify_done",
            pre=frozenset({"coffee_delivered", "mail_fetched"}),
            add=frozenset({"all_done"}),
            delete=frozenset(),
            cost=1,
        ),
    ]

    initial = {"wants_coffee", "mail_waiting"}
    goals = {"all_done"}

    plan = backward_plan(initial, goals, actions)
    names = [a.name for a in plan]
    final_state = execute_plan(initial, plan)

    print("plan:", names)
    print("final_state:", sorted(final_state))

    assert names[-1] == "notify_done"
    assert "deliver_coffee" in names
    assert "fetch_mail" in names
    assert "all_done" in final_state
```

这段代码可以直接运行，预期输出类似：

```text
plan: ['deliver_coffee', 'fetch_mail', 'notify_done']
final_state: ['all_done', 'coffee_delivered', 'mail_fetched']
```

也可能输出：

```text
plan: ['fetch_mail', 'deliver_coffee', 'notify_done']
final_state: ['all_done', 'coffee_delivered', 'mail_fetched']
```

两种都正确，因为 `deliver_coffee` 和 `fetch_mail` 之间没有因果依赖，谁先谁后都可以，只要都在 `notify_done` 之前完成即可。

这个实现故意保持简单，但结构已经接近真实系统中的抽象。

| 字段/函数 | 作用 | 对应真实系统含义 |
|---|---|---|
| `action.pre` | 前提条件 | 执行前必须满足的业务状态 |
| `action.add` | 正向效果 | 动作完成后新增的事实 |
| `action.delete` | 负向效果 | 动作会破坏或消耗的事实 |
| `action.cost` | 代价 | 时间、算力、路程、风险等 |
| `choose_provider` | 为目标找实现者 | 技能选择、工具选择、子 Agent 选择 |
| `backward_plan` | 从目标反推前提 | 目标分解与依赖补全 |
| `execute_plan` | 验证计划可执行 | 仿真、沙箱执行、离线校验 |

但必须明确，这仍然只是“教学用最小版本”，它没有覆盖真实系统里的几个难点：

| 未覆盖问题 | 为什么重要 |
|---|---|
| 威胁检测 | 还没检查某个动作会不会删掉别人需要的事实 |
| 资源约束 | 没有表示“唯一叉车”“唯一 GPU”“同一时间窗” |
| 时间约束 | 没有表示动作持续时间、截止时间、并发窗口 |
| 不确定性 | 没有处理动作失败、外界扰动、部分可观测状态 |

真实系统中的规划器通常不是一次性函数，而是一个循环：

```text
plan(state, goals, resources):
  1. 感知更新世界状态
  2. 从记忆或经验中检索相关上下文
  3. 选择候选动作或技能
  4. 为开放前提寻找实现者
  5. 插入部分顺序约束
  6. 检查因果冲突与资源冲突
  7. 输出当前可执行的动作段
  8. 执行后根据反馈决定继续、修补或重规划
```

这里“输出当前可执行的动作段”很重要。很多 Agent 系统不会一次吐出完整长计划，而是输出一个可安全执行的前缀。原因很现实：环境会变，外部系统会失败，长计划在动态环境里很容易过期。

---

## 工程权衡与常见坑

工程里最常见的问题，不是“算法不会搜索”，而是“系统结构把规划器逼成瓶颈”。

### 1. 单一监督者会成为吞吐瓶颈

如果所有子任务都经由一个 supervisor 统一规划，那么任务量一上来，整个系统就像单收费站。轻载时可控，重载时排队严重。

玩具版理解：一个仓储系统把“拣货、复核、搬运、装车”全交给一个调度器。订单少时没问题，订单一多就出现两种延迟：

- 规划等待时间变长
- 因为状态变化太快，旧计划还没发出去就已经过期

更稳妥的做法通常是分层：

- 上层做目标分解和全局资源约束
- 下层各子域做局部规划
- 仲裁器只处理跨域冲突

可以把这三层理解成不同粒度的决策：

| 层次 | 主要职责 | 典型输出 |
|---|---|---|
| 上层规划 | 决定大目标如何拆分 | 任务树、全局约束 |
| 子域规划 | 在本专业域内排可执行动作 | 局部计划、资源申请 |
| 仲裁层 | 处理跨域资源冲突与优先级 | 最终可执行窗口 |

### 2. 混合反应与规划如果没仲裁好，会 thrash

**反应式策略**，可以理解为“看到条件就立刻触发的规则”。它快，但局部。  
**规划式策略**，可以理解为“先算一段全局合理路径，再执行”。它稳，但慢。

问题在于，很多系统同时需要两者。如果仲裁器设计不好，就会出现 thrash，也就是“来回打架、频繁切换，系统一直忙但不前进”。

例如移动机器人：

- 反应层检测到前方有人，要求立即避障
- 规划层坚持当前路线最优，要求继续走原路径
- 两者来回覆盖命令，导致路径抖动

所以必须提前定义三件事：

| 仲裁问题 | 必须明确的规则 |
|---|---|
| 谁优先 | 安全规则通常高于效率规则 |
| 何时回退 | 局部修补失败多少次后触发重规划 |
| 如何恢复 | 避障结束后是继续旧计划，还是重算全局路径 |

### 3. Plan-reuse 不是总能省成本

**Plan-reuse** 的意思是“复用旧计划的局部片段”，白话说就是“别每次从零开始算”。

它在以下场景非常有效：

- 任务重复率高
- 资源结构稳定
- 冲突较少

但一旦资源强耦合，比如多个 Agent 抢同一台设备、同一时间窗、同一运输通道，复用旧计划反而容易产生隐蔽冲突。因为旧计划当初成立的前提，现在未必还成立。

下面这个表更接近工程决策。

| 机制 | 优点 | 主要风险 | 何时回退到全量重规划 |
|---|---|---|---|
| 监督式统一规划 | 全局视角强 | 单点瓶颈、延迟高 | 任务爆发、状态变化过快 |
| 混合反应+规划 | 响应快且保留全局性 | 仲裁不当会 thrash | 局部规则频繁否定全局计划 |
| Plan-reuse | 计算成本低 | 旧假设失效、冲突漏检 | 共享资源冲突显著上升 |

### 4. 常见坑

| 坑 | 现象 | 原因 | 修复思路 |
|---|---|---|---|
| 把计划当固定脚本 | 一处失败全盘崩 | 没把状态反馈接入规划循环 | 改成“计划-执行-反馈-修补”闭环 |
| 只看动作前提，不看删除效果 | 计划静态可行，执行中断 | 未做因果链保护 | 引入 threat detection |
| 过早线性化 | 并行度下降，改动成本高 | 过早把局部自由度锁死 | 先保留部分顺序，最后再线性化 |
| 迷信复用 | 旧计划越修越乱 | 没有冲突检测和回退条件 | 给 plan-reuse 设失效阈值 |
| 统一调度所有细节 | 中枢过载 | 架构层次划分不合理 | 把局部规划下放到子域 |
| 动作建模过粗 | 计划“看起来对”，执行却卡住 | pre/add/del 省略了关键事实 | 拆细状态变量和资源占用 |

### 5. 一个工程判断准则：什么时候说明建模已经太粗

很多规划失败不是搜索器太弱，而是动作模型写得太粗。下面几个信号通常说明应该先修建模，而不是继续调参：

| 信号 | 常见表现 |
|---|---|
| 大量动作都能“看似”满足目标 | 事实定义过泛，缺少区分度 |
| 执行时频繁遇到“实际不能做” | 前提条件漏建模 |
| 计划总在中途互相覆盖 | 删除效果或资源占用漏建模 |
| 规划结果总是极不稳定 | 关键约束没有进入状态表示 |

这也是新手最容易忽略的一点：规划算法只在“动作模型足够准确”时才有意义。状态、前提、效果建模错了，再高级的搜索也只能在错误空间里找到“错误但自洽”的答案。

---

## 替代方案与适用边界

不是所有 Agent 都需要完整规划器。是否引入规划模块，主要取决于环境变化速度、动作依赖程度和资源冲突强度。

### 1. 反应式策略

反应式策略适合短周期、高动态环境。它的核心思想是：遇到条件直接触发动作，不显式维护长计划。

例如车间里，机器人手臂发现某个工具在最近位置，规则就直接选“最近可用工具”；而不是先生成完整多步计划再执行。这里规划器可以退到后台，只负责关键目标，比如“今天必须完成哪批订单”。

优点是快，缺点是容易局部最优，看不见长程依赖。

### 2. 部分有序规划

当任务存在明显前提依赖，但仍希望保留并行自由度时，部分有序规划最合适。它比纯规则更能处理复杂目标，比完全线性计划更有弹性。

适合：

- 多步骤任务
- 存在因果依赖
- 执行时间不确定
- 需要并行协调

不适合的情况也要讲清楚：

- 动作模型经常变化，今天的 `pre/add/del` 明天就失效
- 状态严重不可观测，连当前世界状态都很难确认
- 规划时间预算极低，来不及做显式搜索

### 3. Plan-reuse

如果系统每天都在做相似任务，例如仓库里重复的入库、拣选、装车流程，那么复用历史计划片段通常能显著减少搜索成本。

但它的适用边界是“环境相似且资源冲突稳定”。一旦进入高竞争场景，就不能把复用当默认策略。

### 4. 全局重规划

当旧计划的大量前提已经失效，继续局部修补往往比重算更贵。全局重规划的成本更高，但有时是唯一能恢复一致性的办法。

下面给出一个总表。

| 策略 | 适合边界 | 不适合边界 |
|---|---|---|
| 反应式/规则驱动 | 超短周期、环境剧烈变化、局部安全优先 | 长链依赖、多目标协调 |
| 部分有序规划 | 多步骤任务、可并行、需保护因果链 | 动作模型极不稳定、状态不可观测严重 |
| Plan-reuse | 高重复任务、低冲突、结构稳定 | 资源强耦合、共享瓶颈明显 |
| 全局重规划 | 冲突激烈、旧计划大面积失效 | 高频实时决策、算力预算极低 |

判断是否需要“全局重规划”，可以看三个信号：

1. 关键资源的冲突数是否突然升高  
2. 旧计划中的因果链是否被多次破坏  
3. 本地修补是否已经影响全局 makespan

其中 makespan 就是“整批任务完成所需的总时间”。如果局部修补不断拖长 makespan，通常说明继续补丁式修复已经不划算，应回退到全局重规划。

为了便于工程落地，可以把切换条件写成一张更实用的决策表。

| 触发信号 | 更适合的策略切换 |
|---|---|
| 外部环境每秒都在变化 | 从长计划退回反应式策略 |
| 任务依赖变多但仍可建模 | 从规则系统升级到部分有序规划 |
| 任务高度重复且状态稳定 | 从全量重算切到 plan-reuse |
| 冲突激增、局部修补失效 | 从 plan-reuse 回退到全局重规划 |

---

## 参考资料

| 资料 | 主要贡献 | 适合为什么读 |
|---|---|---|
| David L. Poole, Alan K. Mackworth, *Artificial Intelligence: Foundations of Computational Agents*, Chapter 6 | 系统讲解状态空间规划、部分顺序规划、因果支持关系 | 适合建立最标准的规划问题定义 |
| Stuart Russell, Peter Norvig, *Artificial Intelligence: A Modern Approach*, Planning 相关章节 | 把规划、搜索、调度、决策放在统一 AI 框架内 | 适合从更大背景理解规划不是孤立模块 |
| Malik Ghallab, Dana Nau, Paolo Traverso, *Automated Planning: Theory and Practice* | 深入讨论经典规划、时间规划、层次规划、资源约束 | 适合从入门走向工程与研究 |
| Penberthy, Weld, “UCPOP: A Sound, Complete, Partial Order Planner for ADL” | 经典部分有序规划器论文，完整展示开放前提和威胁消解机制 | 适合理解 POP 为什么不是“简单排序” |
| Brafman, Domshlak, et al. 关于 Multi-Agent Planning / Plan Merging 的论文脉络 | 解释多主体局部计划为什么还需要上层合并与冲突检测 | 适合理解多 Agent 规划不等于多个单体规划的拼接 |

如果需要可直接检索以下关键词：

- `Poole Mackworth Chapter 6 planning`
- `UCPOP partial order planner`
- `Automated Planning Theory and Practice Ghallab Nau Traverso`
- `multi-agent planning plan merging`
