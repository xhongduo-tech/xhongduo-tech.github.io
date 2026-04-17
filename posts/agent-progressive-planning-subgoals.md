## 核心结论

渐进式规划的核心不是“先想得更全”，而是“执行一步，再决定下一步要细化到哪个层级”。这里的“子目标粒度”指的是一次规划到底规划到多细，例如“做晚饭”是粗粒度，“去冰箱拿鸡蛋”是细粒度。长序列任务里，环境会持续变化，静态计划很容易在中途失效；因此，真正稳定的系统通常需要 `Execute -> Feedback -> Replan` 的闭环。

Plan-and-Solve、ADaPT、LATS 三类方法的差异，不在于“会不会规划”，而在于“多久改一次计划、失败后怎么改、改到多细”。Plan-and-Solve 先生成线性步骤表，默认顺序执行；ADaPT 在失败时递归拆解当前子目标；LATS 把当前决策点展开成多条候选分支，用树搜索挑更值得继续探索的路径。便于记忆的一句话是：执行完再问“接下来我该细化到哪个层级”。

下面这张表先给出最重要的区别。

| 方法 | 计划更新频率 | 子目标粒度控制 | 反馈触发条件 |
|---|---|---|---|
| Plan-and-Solve | 低，通常开头一次性生成 | 基本固定，整条计划粒度较早锁定 | 明显失败或人工要求改写 |
| ADaPT | 中，按需递归重规划 | 动态，失败时把当前目标继续拆细 | 当前步骤失败，且未超过最大深度 |
| LATS | 高，几乎每个关键决策点都可重评估 | 动态，多候选并行展开后再选择 | 每轮搜索的评估分数、回传值、反思结果 |

一个玩具例子很直观。假设任务是“给客人泡茶”。静态计划可能是：烧水、找茶叶、冲泡、端上桌。如果执行到第二步才发现茶叶罐是空的，原计划剩余步骤几乎全部作废。ADaPT 会把“找茶叶”拆成“检查厨房柜子、检查储物间、改用咖啡或提示缺货”；LATS 则会同时评估“找替代饮品”“询问用户”“继续搜索”几个分支，再选择期望收益更高的分支。

真实工程里也是同样逻辑。以 ALFWorld 这类家庭环境模拟任务为例，动作是否成功依赖当前环境状态：门有没有开、物体是否可见、容器是否已被占用。公开复述材料里常见两组数字，一组是较高的二手引用，另一组是更常见的复述值；若按题目给定研究摘要采用后者，ADaPT 在该任务上的成功率可从 ReAct 的 43.3% 提升到 71.6%。这说明收益主要来自“失败后局部重规划”，而不是一次性生成更长的提示词。

---

## 问题定义与边界

问题可以定义为：在多步骤任务中，如何让有限的 LLM 调用根据环境反馈动态调整后续子目标，而不是把整条路径在起点一次写死。这里的“环境反馈”就是动作执行后返回的状态信息，例如“门打不开”“未找到目标物体”“网页元素不存在”。“渐进式规划闭环”则是一次循环：先执行，再读取反馈，再决定是否保留原计划、局部改写、继续递归拆解，或进入更强的搜索。

这个问题不是所有任务都需要。若任务环境稳定、步骤依赖关系清晰、单步成功率高，那么线性计划已经够用。若任务存在以下特征，渐进式规划的价值就会快速上升：

| 任务类型 | Plan-and-Solve | ADaPT | LATS |
|---|---|---|---|
| 环境几乎不变的脚本任务 | 适合 | 可能过度设计 | 通常不划算 |
| 中等深度、局部失败较多的交互任务 | 勉强可用 | 很适合 | 视预算而定 |
| 高不确定性、多分支探索任务 | 容易失效 | 可部分应对 | 最适合 |
| 强预算约束、低延迟场景 | 最友好 | 需限制递归深度 | 往往过重 |

边界也必须说明清楚。第一，渐进式规划适合“可分解但反馈不确定”的任务，不适合完全不可分解、只能一把梭哈的任务。第二，它要求失败代价可控，因为系统需要试错和回退。第三，如果系统对时延特别敏感，例如在线客服必须在几百毫秒内回复，频繁递归或树搜索可能不可接受，此时必须设置停机机制，例如最大递归深度、最大搜索节点数、最大 token 预算。

ALFWorld 是一个典型边界案例。它不是开放世界到无限复杂，但也绝不是固定脚本。每个动作都会改变后续可行动作集，因此很适合作为“执行后再定下一层子目标”的实验场。ADaPT 的价值，就在于不要求一开始把所有细节都猜对，而是在局部失败处补细节。

---

## 核心机制与推导

先看最简单的 Plan-and-Solve。它先生成计划序列

$$
P = [s_1, s_2, \dots, s_n]
$$

然后按顺序执行，默认希望保持一个计划不变式：在执行到第 $k$ 步前，后续步骤 $s_{k+1:n}$ 仍与当前环境兼容。问题在于，这个不变式在真实环境里经常被打破。一旦第 $k$ 步改变了世界状态，后面的步骤可能不再成立。

因此，Plan-and-Solve 的优点是结构简单，缺点是对环境漂移非常脆弱。一个 20 步任务里，只要前 10 步中的某一步隐含前提变化，后 10 步都可能作废。它适合的是“计划本身比执行更贵，但环境很稳定”的情况。

ADaPT 的思路是按需递归拆解。可以把它写成一个判断条件：

$$
\text{Recurse if } (\text{failure}=1) \land (d < d_{\max})
$$

其中 $d$ 是当前递归深度，$d_{\max}$ 是最大递归深度。含义很直接：只有当前子目标执行失败，而且还没超过允许的细化层数，系统才继续往下拆。这样就避免了“一开始把所有事情都拆到最细”的巨大开销。

用一个玩具例子说明。目标是“把房间整理干净”。

1. 粗粒度计划：收衣服、扔垃圾、擦桌子。
2. 执行“收衣服”失败，因为找不到脏衣篮。
3. ADaPT 不会重写整条计划，而是把“收衣服”递归拆成：找衣篮、若找不到则临时指定收纳袋、继续收衣服。
4. 如果“找衣篮”继续失败，且还没到 $d_{\max}$，再往下拆。

这种“只在失败点细化”的策略，正是它比静态规划更稳的原因。题目给定摘要中的 TextCraft 例子也体现了这一点：任务深度为 3 时，ReAct 成功率只有 1.8%，ADaPT 通过把平均有效拆解层数控制在约 1.9，将成功率提升到 38.7%。本质上不是模型突然更会“想”，而是系统允许它在失败位置补充更细的动作结构。

LATS 更进一步。它不是遇错才修一条线，而是在关键节点主动保留多条候选路径。LATS 常借用 MCTS，也就是蒙特卡洛树搜索。这里的“树搜索”可以白话理解为：把当前选择点看成一棵树，先选一条看起来最值的分支试，再根据结果更新整棵树的判断。常见的 UCT 选点公式是

$$
\text{UCT}(c)=Q(c)+\alpha \sqrt{\frac{\ln N}{n_c}}
$$

其中 $Q(c)$ 是子节点 $c$ 的当前平均价值，$N$ 是父节点总访问次数，$n_c$ 是该子节点访问次数，$\alpha$ 控制探索强度。前半部分鼓励继续走高分分支，后半部分鼓励给少试过的分支机会。

把三者放进同一个流程，可以简化为：

1. 执行当前步骤。
2. 读取反馈。
3. 如果反馈正常：
继续原计划，或在 LATS 中回传正奖励。
4. 如果反馈失败：
Plan-and-Solve 倾向整体重写；
ADaPT 递归拆当前子目标；
LATS 扩展更多候选并基于评估继续搜索。

真实工程例子可以看网页自动化。目标是“完成一次机票改签”。Plan-and-Solve 会先列：登录、查订单、点击改签、选择航班、支付。若执行到第四步发现目标按钮因页面实验版本而消失，整条后续计划都失真。ADaPT 会把“选择航班”拆成“重新查询可改签入口、检查页面标签、尝试备用路径”；LATS 则可能同时尝试 DOM 选择器分支、文本匹配分支、导航回退分支，再根据结果选择最可行路线。

---

## 代码实现

工程上最重要的不是先写哪种算法，而是先抽象统一接口。执行器只负责和环境交互，规划器只负责根据反馈决定下一步策略。一个实用接口可以是：

- `execute(step)` 返回 `status`、`obs`、`failure_reason`
- `plan(goal, context)` 生成候选步骤
- `replan(...)` 根据失败原因局部调整

下面给出一个可运行的简化 Python 实现。它不是完整论文复现，但把三种控制结构的核心差异都保留下来了。

```python
from dataclasses import dataclass
from math import log, sqrt

@dataclass
class ExecResult:
    status: str
    obs: str = ""
    failure_reason: str = ""

WORLD = {
    "tea_leaves_available": False,
    "coffee_available": True,
    "kettle_ready": True,
}

def execute(step: str) -> ExecResult:
    if step == "boil_water":
        return ExecResult("ok" if WORLD["kettle_ready"] else "fail", "kettle checked", "kettle broken")
    if step == "find_tea":
        return ExecResult("ok" if WORLD["tea_leaves_available"] else "fail", "searched cabinet", "tea not found")
    if step == "use_coffee":
        return ExecResult("ok" if WORLD["coffee_available"] else "fail", "searched shelf", "coffee not found")
    if step == "serve_drink":
        return ExecResult("ok", "served")
    return ExecResult("fail", failure_reason="unknown step")

def plan_and_solve():
    plan = ["boil_water", "find_tea", "serve_drink"]
    for step in plan:
        result = execute(step)
        if result.status == "fail":
            return False, f"failed at {step}: {result.failure_reason}"
    return True, "done"

def adapt(goal: str, depth: int, dmax: int):
    # 何时触发递归：当前目标失败，且深度未超过上限
    result = execute(goal)
    if result.status == "ok":
        return True, [goal]

    if depth >= dmax:
        return False, [goal]

    if goal == "find_tea":
        subgoals = ["use_coffee"]
    else:
        subgoals = []

    path = [goal]
    for sub in subgoals:
        ok, subpath = adapt(sub, depth + 1, dmax)
        path.extend(subpath)
        if ok:
            return True, path
    return False, path

class Node:
    def __init__(self, name, q=0.0, visits=0):
        self.name = name
        self.q = q
        self.visits = visits

def uct(node: Node, parent_visits: int, c: float = 1.4) -> float:
    if node.visits == 0:
        return float("inf")
    return node.q + c * sqrt(log(parent_visits) / node.visits)

def lats_pick():
    # 何时触发树搜索：当前步骤存在多种候选动作时
    parent_visits = 10
    candidates = [
        Node("find_tea", q=0.2, visits=5),
        Node("use_coffee", q=0.8, visits=2),
    ]
    best = max(candidates, key=lambda n: uct(n, parent_visits))
    return best.name

# 基本行为断言
ok_ps, msg_ps = plan_and_solve()
assert ok_ps is False
assert "find_tea" in msg_ps

ok_adapt, path_adapt = adapt("find_tea", depth=0, dmax=2)
assert ok_adapt is True
assert path_adapt[-1] == "use_coffee"

choice = lats_pick()
assert choice in {"find_tea", "use_coffee"}
```

这段代码对应三个关键点：

| 机制 | 触发点 | 控制动作 |
|---|---|---|
| Plan-and-Solve | 初始阶段 | 生成一条线性计划并直接执行 |
| ADaPT | 单步失败后 | 递归拆当前子目标 |
| LATS | 存在多候选路径时 | 对候选分支做搜索和选择 |

如果要进一步工程化，建议把“失败原因”标准化，例如统一成 `precondition_failed`、`resource_missing`、`navigation_error`。这样 ADaPT 才能按失败类型选择不同拆解模板，LATS 才能把历史反思写成可复用的启发信息。

---

## 工程权衡与常见坑

真正上线时，三类方法都不是“越强越好”，而是“你为成功率愿意付出多少时延和 token”。

第一个常见坑是 Plan-and-Solve 的计划僵化。很多系统表面上做了规划，实际上只是把长提示词变成了长待办清单。一旦环境和计划不一致，执行器还会机械地继续跑，导致错误在后续步骤持续放大。解决方法不是把提示词写得更长，而是给计划设置校验点，例如每 3 步或每次关键状态变化后重新验证前提。

第二个常见坑是 ADaPT 递归失控。失败后不断往下拆，看起来很聪明，但如果拆解模板质量差，系统可能一直在把同一个问题换种说法重复描述，最后没有任何有效动作。这里要加两类约束：最大深度 $d_{\max}$，以及“重复失败模式熔断”，即连续多次出现同类失败时直接停止递归，改为求助外部工具或返回上层策略。

第三个常见坑是 LATS 成本飙升。树搜索最容易出现“理论更优，账单更差”。一个决策点如果展开 8 个候选，每个候选还要做评估、回传、反思，token 开销很容易变成线性规划的 10 到 100 倍。若没有节点预算和时间预算，系统会陷入“越算越慢，越慢越不敢停”。

下面这张表是工程上最常见的风险整理。

| 风险/坑 | 原因 | 规避策略 |
|---|---|---|
| 计划僵化 | 线性计划默认环境不变 | 设置中途校验点，关键状态变化时强制重写 |
| 递归空转 | 失败后只会继续细化，不会切换策略 | 设 `d_max`、失败模式去重、熔断到备用方案 |
| 成本飙升 | LATS 分支过多、评估过频 | 设 `max_nodes`、`max_rollouts`、单步时限 |
| 奖励错配 | 评估函数不反映真实成功率 | 用真实环境信号校正，避免只看语言自评分 |
| 反思污染 | 错误 reflection 被反复继承 | 只保留高置信失败原因，限制反思生命周期 |

可以记一个很短的执行提示：反馈 -> 判断 -> 预算/深度校验 -> 动作。不要在反馈到来之前预设所有细节，也不要在预算见底后还继续搜索。

一个真实工程例子是浏览器智能体做后台配置。若任务有 20 步，前 10 步中某个页面字段因权限变化被隐藏，Plan-and-Solve 若没有中途校验，后续步骤全会建立在错误前提上。LATS 虽然能在异常节点展开更多页面定位策略，但如果不给 `max_nodes`，一次字段定位失败就可能把整轮 quota 烧光。

---

## 替代方案与适用边界

如果任务复杂度低，不要急着上动态规划。很多场景里，简单方法是更好的系统设计。比如固定格式的数据填报、稳定后端 API 编排、规则清晰的文档转换，这些场景通常用 Plan-and-Solve 就够了。因为环境变化小，线性计划的失效率低，额外引入递归或树搜索只会增加成本和调试复杂度。

当任务开始出现“局部失败很多，但失败位置相对集中”时，ADaPT 通常是最划算的升级。它比完全静态规划稳，又没有 LATS 那么重。适合网页操作、家居环境交互、多工具调用编排这类任务。若系统只在失败点需要更细的动作拆解，ADaPT 往往能提供最好的性价比。

LATS 更适合“在关键节点必须探索多个备选路径”的任务，例如复杂网页导航、开放式软件调试、策略博弈、需要试探性信息收集的代理任务。它不是默认方案，而是高不确定性场景下的强化方案。

| 方法 | 适合场景 | 资源要求 | 关键限制 |
|---|---|---|
| Plan-and-Solve | 环境稳定、流程清晰、低延迟需求强 | 低 | 中途失配时恢复能力弱 |
| ADaPT | 局部失败多、任务可递归拆解 | 中 | 深度控制不好会空转 |
| LATS | 多分支探索、高不确定性决策 | 高 | token 与时延成本高 |

可以按下面的决策流程理解：

1. 任务复杂度低、环境稳定？
用 Plan-and-Solve。
2. 任务有明显失败点，且失败后能靠更细步骤恢复？
用 ADaPT。
3. 关键节点有多个可行分支，需要比较探索收益？
在这些节点上局部启用 LATS。
4. 预算紧张？
优先限制 ADaPT 深度，只在高价值节点启用小规模 LATS。

一个对话机器人就是很典型的边界例子。若对话状态简单，只需识别意图并调用固定接口，Plan-and-Solve 已足够。若用户经常提供不完整信息，系统就需要 ADaPT 式递归澄清，把“完成退款”拆成“确认订单号、确认退款原因、确认支付渠道”。若用户状态复杂、问题路径多，例如客服升级、跨系统核验、异常争议处理，则可以只在关键转折点启用有限版 LATS，展开几条候选处理路径再选择。

---

## 参考资料

- Emergent Mind, Plan-and-Solve Prompting 综述：用于理解 Plan-and-Solve 的基本框架、计划不变式和渐进式规划的总体位置。推荐先读它，建立方法谱系。
- ADaPT 相关论文与公开介绍，NAACL 2024：核心看点是 as-needed decomposition，也就是“按需分解”，适合理解递归触发条件、深度控制和 executor 交接方式。
- LATS 相关材料：重点关注树搜索、UCT、reflection 回传机制，理解为什么它在高不确定性场景里比线性规划更稳。
- Liner 上的 ADaPT 解读与实验复述：适合快速查看 TextCraft 等任务上的数值变化，尤其是不同任务深度下的成功率差异。
- Medium 上关于递归分解提升代理成功率的案例文章：适合快速理解 ALFWorld 一类环境中的工程直觉，但阅读时应优先以论文或原始报告校对数字。
- 若要复现，推荐阅读顺序是：先看 Plan-and-Solve 综述，再看 ADaPT 的递归分解，再看 LATS 的树搜索与反思机制。这样更容易把“线性计划、失败细化、多分支搜索”三者放到同一坐标系里。
