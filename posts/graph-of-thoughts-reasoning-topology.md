## 核心结论

Graph of Thoughts，简称 GoT，可以直译为“思维图”。它把大模型的推理过程从一条链或一棵树，扩展成一张**有向无环图**，也就是节点有方向、整体不能绕回来的图。

它解决的核心问题不是“让模型想得更长”，而是“让不同分支的中间结果可以复用”。这点比 Tree of Thoughts，简称 ToT，更关键。ToT 允许分叉，但不同分支通常彼此隔离；GoT 允许把多个分支的结果在后续节点中合并，因此可以减少重复推理。

形式化地说，GoT 把推理状态写成：

$$
G=(V,E)
$$

其中 $V$ 是思维节点集合，每个节点 $v \in V$ 表示一个中间结论、候选答案或子问题结果；$E$ 是依赖边集合，表示“哪个结果依赖哪个前置结果”。

GoT 的价值主要体现在三点：

| 结论 | 含义 | 直接收益 |
|---|---|---|
| 推理拓扑从树变成 DAG | DAG 是有向无环图，允许多路汇合 | 中间结果可复用 |
| 支持聚合节点 | 聚合就是把多个分支的结果整理成一个新节点 | 降低重复调用 |
| 更适合复杂任务 | 尤其是多证据、多阶段、多分支问题 | 更容易并行和解释 |

论文中的排序实验表明，GoT 相比 ToT 在该任务上能提高结果质量，并把成本降低到超过 31% 的幅度。这里的“成本”主要指 LLM 调用与相关推理开销。这个数字不能机械外推到所有任务，但它说明了一个事实：**当任务里存在可复用子结果时，图结构比树结构更省。**

---

## 问题定义与边界

GoT 适合的问题，不是所有“需要思考”的问题，而是满足下面两个条件的问题：

1. 任务可以拆成多个中间状态。
2. 不同路径之间存在共享信息，值得合并。

如果一个任务只能线性推进，比如简单问答“法国首都是什么”，把它画成图没有收益。因为没有可复用的分支，中间节点也不需要汇合，这时 Chain of Thought，简称 CoT，也就是“链式思维”，就够了。

GoT 的问题定义可以写成：给定原始问题 $x$，构造一张逐步扩展的 DAG，使每个节点保存一个中间状态，每条边保存依赖关系，再用评分函数决定哪些节点继续展开，哪些节点被剪掉。

常见节点属性如下：

| 属性 | 含义 | 是否必须 |
|---|---|---|
| `state` | 当前思维内容，例如局部排序结果、子问题答案 | 是 |
| `parents` | 前驱节点，也就是它依赖哪些结果 | 是 |
| `score` | 当前节点质量分，例如一致性、正确率估计 | 是 |
| `op` | 节点由什么操作得到，如生成、聚合、验证 | 建议 |
| `expandable` | 是否值得继续展开 | 建议 |
| `terminal` | 是否已达到终止条件 | 建议 |

边界也很明确。

第一，GoT 一般要求图保持无环。如果允许循环，系统会遇到“旧结论反复喂给自己”的问题，调度和终止条件都会变复杂。

第二，聚合必须是**有意义的合并**。不是把多段文本拼起来就算聚合。真正的聚合是把多个分支的公共信息、互补信息或投票结果整理成一个新的、更强的状态。

第三，评分函数必须可操作。评分就是“给当前节点打分”，白话讲是判断它值不值得继续算。如果没有评分和剪枝，GoT 很容易图爆炸。

一个玩具例子最容易看清边界。

假设要对 `[3,1,2]` 排序。ToT 的做法可能是生成几条不同排序路径，各自推进。GoT 的做法是把“3 和 1 的比较”“3 和 2 的比较”看成两个可复用节点，后续汇总节点直接引用它们。这样，后续节点不需要重新比较两次。

这说明 GoT 的对象不是“整题答案”，而是“可复用的中间判断”。

---

## 核心机制与推导

GoT 的标准流程可以概括为五步：

1. 初始化根节点。
2. 生成候选子节点。
3. 对候选打分并排序。
4. 对可合并的节点做聚合。
5. 剪枝并继续迭代，直到终止。

把它写成抽象更新过程：

$$
v' = \alpha(\{v_1, v_2, \dots, v_k\})
$$

这里 $\alpha$ 是聚合函数，意思是“把多个已有节点合成一个新节点”。例如，把三条候选解释做一致性总结；或者把多个局部排序结果整理成一个更完整的排序状态。

新节点的分数可以写成：

$$
s(v') = f(s(v_1), s(v_2), \dots, s(v_k), c)
$$

其中 $s(\cdot)$ 是评分函数，$c$ 表示一致性、覆盖率、约束满足度等额外指标。

一个简化的推导逻辑是：

- 若某结论只依赖单一路径，树结构足够。
- 若多个路径会导出共享中间结论，树结构会重复计算。
- 图结构允许把共享部分收敛到同一个节点。
- 因此，当任务存在共享子问题时，GoT 理论上能减少重复推理成本。

### 玩具例子：排序 `[3,1,2]`

设：

- $v_0$：原始任务“排序 `[3,1,2]`”
- $v_1$：比较 `3` 和 `1`，得出 `1 < 3`
- $v_2$：比较 `3` 和 `2`，得出 `2 < 3`
- $v_3$：聚合 $v_1$ 与 $v_2$，得出 `3` 不应在最前
- $v_4$：再判断 `1` 和 `2`
- $v_5$：输出 `[1,2,3]`

这里 $v_3$ 就是典型的聚合节点。它不重新做比较，而是复用前面两个结论。树结构的问题在于，相似的比较可能在多条路径里被各算一遍；图结构把共享部分抽出来，后面统一引用。

### 真实工程例子：RAG 多证据问答

RAG 是 Retrieval-Augmented Generation，直译是“检索增强生成”，也就是先查资料再回答。

假设问题是“某数据库故障为何导致写入延迟抖动”。你可能同时拿到三类证据：

- 监控指标：CPU、IO、队列长度
- 日志片段：锁等待、超时重试
- 配置快照：连接池、刷盘策略

如果用 CoT，模型会把这些证据串成一条长文本来推理。问题是中途很难分清“哪个结论来自哪类证据”。

如果用 ToT，可以让模型分别从“IO 瓶颈”“锁冲突”“配置错误”三个方向探索，但这些分支后面通常还是分开走。

GoT 更合适的做法是：

- 先为每类证据生成局部诊断节点；
- 再建立聚合节点，把“IO 饱和”和“刷盘频率过高”合并成“存储子系统瓶颈”；
- 再把“锁等待”与“连接池重试”合并成“放大延迟的次级机制”；
- 最后总汇成根因分析节点。

这样做的价值是：不同分支可以先并行，再在高层节点融合，而不是从头到尾只走一条路径。

---

## 代码实现

工程上，GoT 通常拆成三层：

1. `Controller`：控制器，负责调度图的扩展、评分、剪枝和终止。
2. `Prompter/Parser`：提示构造器与解析器，负责和模型交互。
3. `Operations`：操作集合，封装生成、评分、聚合、验证等步骤。

官方实现里常见的是 `GraphOfOperations`，也就是“操作图”，它定义“先生成，再评分，再校验”这样的执行流程。对于入门理解，可以先把它看成一个**推理流水线**。

下面给一个可运行的 Python 玩具实现。它不调用 LLM，只模拟 GoT 的“生成-聚合-剪枝”思想。

```python
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class Node:
    id: str
    state: Tuple[int, ...]
    parents: List[str] = field(default_factory=list)
    score: float = 0.0
    op: str = "generate"


def inversions(arr: Tuple[int, ...]) -> int:
    count = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                count += 1
    return count


def score_state(arr: Tuple[int, ...]) -> float:
    # 分数越高表示越接近有序
    return -inversions(arr)


def local_swaps(arr: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    out = []
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            b = list(arr)
            b[i], b[i + 1] = b[i + 1], b[i]
            out.append(tuple(b))
    return out


def aggregate(states: List[Tuple[int, ...]]) -> Tuple[int, ...]:
    # 简化版聚合：保留当前候选里分数最高的状态
    return max(states, key=score_state)


def got_sort(start: List[int]) -> Tuple[int, ...]:
    root = Node(id="v0", state=tuple(start), score=score_state(tuple(start)))
    frontier = [root]
    seen: Dict[Tuple[int, ...], Node] = {root.state: root}
    node_index = 1

    while frontier:
        candidates: List[Node] = []

        for node in frontier:
            for child_state in local_swaps(node.state):
                if child_state in seen:
                    # 已见过的状态视为“图中的复用节点”
                    continue
                child = Node(
                    id=f"v{node_index}",
                    state=child_state,
                    parents=[node.id],
                    score=score_state(child_state),
                    op="generate",
                )
                node_index += 1
                seen[child_state] = child
                candidates.append(child)

        if not candidates:
            break

        # top-k 剪枝
        candidates.sort(key=lambda n: n.score, reverse=True)
        topk = candidates[:2]

        # 聚合：从多个候选中选出更优的共享状态
        merged_state = aggregate([n.state for n in topk])
        merged = Node(
            id=f"v{node_index}",
            state=merged_state,
            parents=[n.id for n in topk],
            score=score_state(merged_state),
            op="aggregate",
        )
        node_index += 1

        if merged.state == tuple(sorted(start)):
            return merged.state

        frontier = [merged]

    best = max(seen.values(), key=lambda n: n.score)
    return best.state


result = got_sort([3, 1, 2])
assert result == (1, 2, 3)
assert inversions((3, 1, 2)) == 2
assert score_state((1, 2, 3)) > score_state((3, 1, 2))
```

这段代码的重点不是“实现了完整 GoT”，而是把三个核心动作落地了：

- `local_swaps`：生成候选节点
- `score_state`：对节点打分
- `aggregate`：把多个候选汇总成一个共享状态

如果切回官方框架，常见入口类似下面这种结构：

```python
from examples.sorting.sorting_032 import SortingPrompter, SortingParser, got
from graph_of_thoughts import controller, language_models

gop = got()
lm = language_models.ChatGPT("config.json", model_name="chatgpt")

ctrl = controller.Controller(
    lm,
    gop,
    SortingPrompter(),
    SortingParser(),
    {
        "original": "[3,1,2]",
        "current": "",
        "phase": 0,
        "method": "got"
    }
)

ctrl.run()
ctrl.output_graph("output_got.json")
```

这里 `method="got"` 的作用，就是让初始状态和后续流程走 GoT 方案，而不是 CoT 风格的单链展开。

---

## 工程权衡与常见坑

GoT 的最大优点，也是最大风险来源：它允许更多结构操作，因此更容易失控。

最常见的问题不是“模型不会推理”，而是“图太大”。

| 风险 | 现象 | 缓解手段 | 监控指标 |
|---|---|---|---|
| 分支爆炸 | 每轮节点数指数增长 | 评分阈值、top-k、最大深度 | 节点总数、每轮新增节点 |
| 重复推理 | 相似状态多次生成 | 状态去重、哈希缓存、子图复用 | 唯一状态占比、缓存命中率 |
| 错误聚合 | 把矛盾结论合成一个节点 | 聚合后验证、一致性检查 | 聚合后回退率 |
| 评分漂移 | 分数不能反映真实质量 | 引入规则校验或外部工具 | 高分错误样本占比 |
| 调度开销过高 | 图管理成本超过推理收益 | 简化图操作、限制入度 | 单轮调度耗时 |

有几个坑尤其容易在真实系统里出现。

第一，**把“多分支”误当成“高质量”**。分支多不代表结果好。没有评分和剪枝，GoT 会比 ToT 更贵，因为图调度本身也要成本。

第二，**把拼接当聚合**。例如把三段候选答案直接拼成一个长上下文，再让模型“总结一下”。这不是真正的图聚合，而是把问题重新塞回长上下文窗口。真正的聚合应该有明确目标，比如投票、去重、约束合并、证据归因。

第三，**节点定义过粗**。如果每个节点都是一大段长文本，图虽然存在，但复用粒度太粗，收益会被吃掉。工程上更合理的是让节点对应“一个局部判断”“一个子结论”或“一个结构化状态”。

第四，**忽略验证节点**。验证节点就是专门检查结论是否满足约束的节点。比如代码生成里检查单元测试是否通过；检索问答里检查引用是否支持结论；规划任务里检查路径是否合法。没有验证，GoT 只是“更复杂的提示工程”，不是稳定系统。

一个真实工程例子是多工具 Agent。Agent 是“能调用外部工具完成任务的智能体”。假设它要排查线上故障，可能并行调用日志查询、指标查询、配置读取三个工具。GoT 的好处在于：

- 每个工具结果先形成独立节点；
- 中间聚合节点专门做“证据对齐”；
- 最终结论节点引用的是被验证过的中间状态，而不是未经整理的原始输出。

如果不这样做，常见后果是：日志分支说“锁冲突”，指标分支说“CPU 正常”，配置分支说“连接池过小”，最后模型把三者硬写成一段看似完整、实际缺乏因果结构的总结。

---

## 替代方案与适用边界

GoT 不是 CoT 和 ToT 的简单升级，而是不同的拓扑选择。

| 方法 | 拓扑 | 能否聚合 | 并行能力 | 适用任务 |
|---|---|---|---|---|
| CoT | 链 | 否 | 弱 | 简单推导、单路径解释 |
| ToT | 树 | 通常否 | 中 | 需要搜索多个候选路径 |
| GoT | DAG | 是 | 强 | 共享子问题、多证据融合 |
| AGoT | 动态 DAG | 是 | 强 | 难度变化大、需按需扩展 |

CoT 最简单。任务本身就是线性的，或者你只需要一个可读的推导过程时，用 CoT 成本最低。

ToT 适合“需要试错搜索”的任务，例如谜题、规划、组合优化。它比 CoT 强，但它的问题是分支之间通常不能天然复用。

GoT 适合“多个分支最终会共享部分结论”的任务。例如：

- 多文档证据融合
- 复杂代码修复
- 多阶段规划
- 多模态推断
- 工具调用后的统一归因

AGoT，也就是 Adaptive Graph of Thoughts，可以译作“自适应思维图”。它进一步强调：不是每个节点都要继续展开，而是只对复杂节点递归拆分。白话讲，就是“把算力花在难的地方”。这比固定宽度、固定深度的图搜索更实用。

但 GoT 也有明确不适用的场景：

- 问题很短，答案几乎一步可得
- 中间状态难以结构化
- 分支之间几乎没有共享信息
- 系统没有足够的评分和验证机制

如果这些条件成立，强行上 GoT 往往只是增加复杂度。

---

## 参考资料

1. Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger 等. *Graph of Thoughts: Solving Elaborate Problems with Large Language Models*. AAAI 2024. 对应 arXiv:2308.09687。
2. `spcl/graph-of-thoughts` 官方实现仓库，包含 sorting、keyword counting 等示例，以及 `Controller`、`GraphOfOperations` 等核心模块。
3. Tushar Pandey, Ara Ghukasyan, Oktay Goktas, Santosh Kumar Radha. *Adaptive Graph of Thoughts: Test-Time Adaptive Reasoning Unifying Chain, Tree, and Graph Structures*. arXiv:2502.05078，2025。
4. GoT 论文摘要中的关键结果包括：在排序任务上，相比 ToT 提升结果质量，并把成本降低超过 31%。
5. 阅读顺序建议：先看 GoT 论文中的排序例子，再看官方仓库的 `sorting` 示例，最后看 AGoT 对“按需扩展”的补充。
