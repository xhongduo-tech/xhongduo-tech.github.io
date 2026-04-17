## 核心结论

TAP，中文可理解为“带剪枝的攻击树”，本质上不是某一条更巧妙的越狱提示，而是一种**搜索策略**。搜索策略就是“在很多候选方案里，按规则继续扩展有希望的、丢弃没希望的”。它把黑盒大模型的越狱提示优化，从“单条提示不断改写”升级成“多条提示并行扩展的树状搜索”。

它的核心收益有两个。第一，**覆盖面更大**。一条线性迭代路径如果走偏，整轮预算就浪费了；树搜索允许同时保留多个方向。第二，**查询效率更高**。效率在这里指“用更少的目标模型调用，找到更高成功率的路径”。TAP 不是盲目扩树，而是每一轮都先评分再剪枝，只保留最有潜力的分支。

新手可以先把它理解成一个三角色流水线：

| 角色 | 白话解释 | 职责 |
| ---- | ---- | ---- |
| 攻击者模型 | 负责“想新点子”的模型 | 基于当前提示生成多个变体 |
| 评估器模型 | 负责“先筛掉差方案”的模型 | 判断是否跑题、是否更接近成功 |
| 目标模型 | 被测试的模型 | 返回真实响应，供评估器打分 |

因此，TAP 的重点不是“神奇 prompt”，而是**Branch -> Evaluate -> Prune** 这个循环。设宽度为 $w$、分支因子为 $b$、最大深度为 $d$，粗略查询上界可以写成：

$$
Q_{\max} \approx w \times b \times d
$$

但这是悲观上界。真实运行时，大量分支会在早期因离题或低分被剪掉，所以实际查询数通常明显低于这个值。

---

## 问题定义与边界

TAP 讨论的是**黑盒大模型攻击搜索**。黑盒的白话解释是：你只能看到输入和输出，看不到模型权重、激活值、训练数据或中间推理状态。也就是说，攻击者只能像普通用户一样反复调用 API。

它解决的问题不是“证明模型一定能被越狱”，而是：**在有限调用预算内，怎样更系统地搜索高潜力攻击路径**。这里的“预算”通常指目标模型 API 的可调用次数，因为这部分最贵，也最容易被频率限制。

TAP 的三个核心超参数如下：

| 参数 | 含义 | 典型作用 |
| ---- | ---- | -------- |
| `d` | 最大深度 | 决定最多迭代多少轮 |
| `w` | 每轮保留宽度 | 决定同时保留多少条高潜力路径 |
| `b` | 每个节点扩展的子分支数 | 决定每轮探索有多发散 |

可以把它看成“先扩散，再收缩”的节奏。$b$ 大，探索更广；$w$ 大，保留更多候选；$d$ 大，允许更长的多轮演化。但三者一起增大时，成本会上升。

一个玩具例子：假设你只有 30 次目标模型调用预算，设置 `w=3`、`b=2`、`d=5`。这不表示一定会用满 $3 \times 2 \times 5 = 30$ 次，因为很多节点会在进入目标查询前就被 off-topic 检查淘汰。off-topic 的白话解释是“提示已经偏题，不再朝目标行为靠近”。

边界也必须讲清楚。TAP 是**授权红队测试方法**，适合用来评估模型守护、比较防护强弱、检验拦截链路，而不是为了生成可直接复用的现实攻击内容。它的论文价值主要在“搜索与评估机制”，不是在某个具体违规话术模板上。

---

## 核心机制与推导

TAP 的核心流程可以拆成五步：

1. 从当前保留的提示集合出发，攻击者模型为每个提示生成 $b$ 个新变体。
2. 评估器先做 Phase 1：过滤 off-topic 节点。
3. 将剩余提示发给目标模型，拿到真实响应。
4. 评估器再做 Phase 2：给“提示-响应”对打分。
5. 只保留分数最高的前 $w$ 个节点，进入下一层。

其中，Phase 2 的保留规则可以写成：

$$
S_{t+1} = \operatorname{top}_w \{ \text{JudgeScore}(x) \mid x \in \mathcal{C}_t \}
$$

这里 $\mathcal{C}_t$ 是第 $t$ 轮所有候选节点集合，`JudgeScore` 是评估器给出的分数。分数越高，表示该路径越接近“目标行为被触发”或“防护开始松动”。

为什么它比单链条迭代更有效？因为单链条方法每轮只押注一个方向，失败就只能继续在该方向局部微调。TAP 则保留多条高分支，实质上是在做一种受预算约束的**启发式搜索**。启发式的白话解释是“不是穷举全部，而是优先搜索看起来更有希望的地方”。

再看一个最小数值例子。设 `w=3`、`b=2`、`d=2`：

- 第 0 层保留 3 个提示
- 每个提示扩 2 个子分支，得到 6 个候选
- 过滤离题后，假设剩 4 个
- 查询目标并打分
- 保留其中前 3 个，进入下一层

理论上最大查询可按 $w \times b \times d = 12$ 粗略估算，但如果第一轮就筛掉一半离题节点，实际目标查询可能只有 6 到 8 次。剪枝节省的不是“搜索逻辑”，而是**把最贵的调用集中在更有价值的候选上**。

真实工程例子是模型红队评估。假设一个团队要比较“裸模型”和“带守护模型”在同一批高风险任务上的表现。若使用人工手写 prompt，通常只能覆盖少数固定模板；若使用 TAP，则可以把每个任务作为根节点，让攻击者模型自动扩展多个语义变体，再由评估器统一打分。这样得到的不是单次命中结果，而是“在固定预算下，这个防护系统会被多少高分路径穿透”。

---

## 代码实现

下面给出一个**安全的玩具实现**。它不连接真实模型，也不生成真实越狱内容，只模拟 TAP 的搜索与剪枝逻辑。重点是理解输入输出结构，而不是复现危险样本。

```python
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Node:
    prompt: str
    score: float = 0.0
    depth: int = 0


def attacker_expand(node: Node, branching_factor: int) -> List[Node]:
    # 玩具扩展器：为同一个节点生成多个“变体”
    children = []
    for i in range(branching_factor):
        children.append(Node(prompt=f"{node.prompt} | variant_{i}", depth=node.depth + 1))
    return children


def evaluator_off_topic(node: Node) -> bool:
    # 玩具规则：包含 bad_topic 的节点视为跑题
    return "bad_topic" in node.prompt


def target_query(node: Node) -> str:
    # 玩具目标：某些关键词组合更接近“成功”
    if "variant_1" in node.prompt and "seed_good" in node.prompt:
        return "high_potential_response"
    if "variant_0" in node.prompt:
        return "medium_potential_response"
    return "low_potential_response"


def evaluator_score(response: str) -> float:
    # 玩具评分器：把目标响应映射为分数
    mapping = {
        "high_potential_response": 9.5,
        "medium_potential_response": 6.0,
        "low_potential_response": 2.0,
    }
    return mapping[response]


def select_top_w(nodes: List[Node], width: int) -> List[Node]:
    return sorted(nodes, key=lambda x: x.score, reverse=True)[:width]


def tap_search(seeds: List[str], width: int, branching_factor: int, max_depth: int, threshold: float) -> Tuple[bool, List[Node]]:
    frontier = [Node(prompt=s, depth=0) for s in seeds][:width]

    for _ in range(max_depth):
        candidates: List[Node] = []

        for node in frontier:
            for child in attacker_expand(node, branching_factor):
                if evaluator_off_topic(child):
                    continue
                response = target_query(child)
                child.score = evaluator_score(response)
                candidates.append(child)

        if not candidates:
            return False, []

        best = select_top_w(candidates, width)
        if best[0].score >= threshold:
            return True, best

        frontier = best

    return False, frontier


found, nodes = tap_search(
    seeds=["seed_good", "seed_bad_topic"],
    width=2,
    branching_factor=2,
    max_depth=3,
    threshold=9.0,
)

assert found is True
assert len(nodes) <= 2
assert nodes[0].score >= nodes[-1].score
assert nodes[0].score == 9.5
```

这段代码对应 TAP 的最小骨架：

| 模块 | 输入 | 输出 | 作用 |
| ---- | ---- | ---- | ---- |
| `attacker_expand` | 当前节点 | 子节点列表 | 生成变体 |
| `evaluator_off_topic` | 子节点 | 布尔值 | 过滤跑题 |
| `target_query` | 子节点 | 目标响应 | 模拟目标模型返回 |
| `evaluator_score` | 目标响应 | 分数 | 判断潜力 |
| `select_top_w` | 候选节点 | 前 `w` 个节点 | 剪枝 |

如果要接入真实工程，通常需要把这几个函数改成 API 适配层：

- `attacker_expand` 调用攻击者模型
- `target_query` 调用被测目标模型
- `evaluator_score` 调用评估器模型或规则引擎
- 每个节点保存完整对话历史、父节点、分数来源、时间戳

这里有一个容易忽略的点：**节点不是单条字符串，而应是“状态”**。状态的白话解释是“后续搜索需要继承的全部上下文”。多轮攻击中，状态至少应包括当前 prompt、历史对话、父节点编号、是否被 phase 1 过滤、目标响应、judge 分数。

---

## 工程权衡与常见坑

TAP 在工程上最容易出问题的，不是“树不够大”，而是**评估器质量不够**。评估器如果太弱，会出现两类错误。

| 风险 | 影响 | 规避方式 |
| ---- | ---- | ---- |
| 误把安全回复判成成功 | 提前停止，漏掉真正有效路径 | 用更强评估器，或增加二次复核 |
| 误把高潜力节点判成低分 | 好分支被剪掉，成功率下降 | 保留少量探索性节点，降低过早收缩 |
| off-topic 未隔离 | 跑题历史污染后续扩展 | 在 Phase 1 就丢弃，不写入后续上下文 |
| `w` 过小 | 多样性不足，搜索早熟收敛 | 在预算允许时适度增大宽度 |
| `b` 过大 | 候选过多，评估成本暴涨 | 先小分支试探，再按分数自适应扩展 |

一个典型坑是“把 off-topic 节点也保留进对话历史”。这会导致下一轮攻击者在错误语境上继续扩展。树搜索表面上还在增长，但增长的是无效枝条。对于多轮方法，这种历史污染比单轮误判更致命，因为错误会沿着父子关系向下传播。

另一个坑是“把评分器当成真值”。TAP 的评分本质上是代理信号。代理信号的白话解释是“它不是最终目标，只是用来近似目标的替代指标”。如果评分器偏好某类表面特征，例如更长、更绕或更像角色扮演的文字，那么搜索树会被引导到“更像攻击、但未必更有效”的方向。

真实工程里，比较稳妥的做法通常是：

- Phase 1 用保守规则严格过滤离题
- Phase 2 用能力更强的模型打分
- 成功阈值附近的结果再做人工复核或第二模型复判
- 保存完整搜索轨迹，便于回放“为什么这条分支被留下”

---

## 替代方案与适用边界

TAP 不是唯一方案。它适合**预算中等、需要探索多个方向**的测试场景；但预算极小或任务结构非常固定时，未必最优。

| 方法 | 结构 | 优点 | 适用边界 |
| ---- | ---- | ---- | -------- |
| TAP | 多分支树搜索 + 剪枝 | 覆盖广，查询利用率高 | 预算中等，目标防护较强 |
| PAIR 类方法 | 单链条迭代优化 | 实现简单，状态管理轻 | 预算很小，只能押注少数方向 |
| 手工红队脚本 | 人工模板或规则库 | 可控、可解释 | 任务固定，需稳定回归测试 |

如果任务是“每周回归测试 20 条固定风险样本”，手工脚本可能更便宜，因为你要的是稳定复现，不是最大探索。如果任务是“评估新守护策略是否真的提高鲁棒性”，TAP 更合适，因为它能给出更全面的攻击覆盖。

还要注意一个适用边界：TAP 提升的是**搜索效率**，不是保证成功的魔法。若目标模型守护极强、输出审查独立、上下文隔离严格，即使树搜索做得再好，也可能没有可行路径。此时 TAP 的价值仍然存在，但价值变成了“证明在给定预算下没有明显弱点”，而不是“必须找出一条突破路径”。

从方法论上看，TAP 可以理解为把“提示工程”升级成“预算受限的启发式搜索”。这也是它最值得记住的地方：真正可复用的不是某个提示文本，而是**如何系统地分配探索、评估和剪枝预算**。

---

## 参考资料

- NeurIPS 2024 论文《Tree of Attacks: Jailbreaking Black-Box LLMs Automatically》：介绍 TAP 的攻击树、两阶段剪枝、实验成功率与查询成本。https://proceedings.neurips.cc/paper_files/paper/2024/file/70702e8cbb4890b4a467b984ae59828a-Paper-Conference.pdf
- TAP 官方 GitHub 仓库：包含 `main_TAP.py`、参数配置和整体流程说明。https://github.com/RICommunity/TAP
- Azure PyRIT 文档《Tree of Attacks with Pruning (Multi-Turn)》：展示多轮 TAP 的工程封装与宽度、深度、剪枝参数。https://azure.github.io/PyRIT/code/executor/attack/tap_attack.html
