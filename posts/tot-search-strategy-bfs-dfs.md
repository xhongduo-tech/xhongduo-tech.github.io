## 核心结论

Tree of Thoughts，简称 ToT，可以直白理解为“把推理过程当成搜索树来做”。普通 Chain of Thought，简称 CoT，本质上是沿着一条思路一直往下写；ToT 则允许模型在某一步同时保留多条候选思路，给这些思路打分，再继续扩展更有希望的分支。

这件事的意义很直接：很多问题不是“不会推”，而是“第一条思路走错后没有回头机制”。ToT 通过“生成候选思路 + 评估 + 剪枝 + 回溯”解决这个问题，因此比单路径 CoT 更不容易卡死在局部最优。

Game of 24 是最容易说明这个差异的玩具例子。给定数字 $(4, 9, 10, 13)$，目标是通过四则运算得到 24。CoT 常见失败方式是先选错组合，然后整条链路报废；ToT 会同时探索多个分支，例如保留 $(13-9)\times(10-4)$ 这样的中间思路，并丢弃明显走不通的分支。公开实验中，GPT-4 在该任务上 ToT 的成功率约为 74%，而普通 CoT 约为 4%。这里的核心不是“模型突然更会算了”，而是“搜索策略让模型有机会纠错”。

| 方法 | 推理形态 | 是否保留多路径 | 是否可回溯 | Game of 24 成功率 |
|---|---|---:|---:|---:|
| CoT | 单链路展开 | 否 | 否 | 约 4% |
| ToT | 树状搜索 | 是 | 是 | 约 74% |

因此，ToT 的核心设计问题不是“要不要搜索”，而是“搜索到什么程度、如何剪枝、何时选 BFS、何时选 DFS”。

---

## 问题定义与边界

要讨论 ToT，先统一状态表示。设输入问题为 $x$，已经生成的思路序列为 $z_1,\dots,z_i$，那么当前状态写作：

$$
s=[x,z_1,\dots,z_i]
$$

这里“状态”可以白话理解为“题目本身加上目前已经想到哪一步”。语言模型不是直接输出最终答案，而是在这个状态上继续生成下一步思路。

如果把整条推理链看成概率模型，可以写成：

$$
p_{\theta}(s)=p_{\theta}(x)\prod_{j=1}^{i}p_{\theta}(z_j\mid x,z_{1:j-1})
$$

这个式子的含义很朴素：当前这条思路有多靠谱，取决于每一步延续在上下文里的条件概率。工程里通常不会真的把这个概率精确算出来，而是用“评分器”近似它，比如让模型输出 `sure / maybe / impossible`，或者给 1 到 10 的分数，再据此排序和剪枝。

问题边界也要说清楚。ToT 不是所有任务都划算，它主要适合这类问题：

| 任务特征 | 是否适合 ToT | 原因 |
|---|---|---|
| 存在多条中间思路 | 是 | 多路径搜索有价值 |
| 中间状态可评估 | 是 | 可以及时剪枝 |
| 单条链路容易走偏 | 是 | 回溯能补救 |
| 两三步就能稳定做对 | 否或收益有限 | 搜索开销可能大于收益 |

一个简单例子：如果是固定 3 步的逻辑题，而且每一步都能立即验证约束，比如“先选 A 或 B，再检查是否违反条件，再选最终答案”，这类浅层问题更偏向 BFS。BFS 是广度优先搜索，可以白话理解为“先把这一层所有值得看的路都看一遍，再下潜到下一层”。因为问题浅、每层可验证，广度探索通常比深挖一条错误路径更稳。

---

## 核心机制与推导

ToT 的基本流程可以拆成三个操作：

1. 生成：从当前状态生成若干候选思路。
2. 评估：判断每个候选思路是否值得保留。
3. 选择：按搜索策略继续展开，或剪枝、或回溯。

这里的“思路节点”不是最终答案，而是中间推理单元。例如在 Game of 24 中，一个节点可以是“先计算 13-9=4，并保留 10 和 4”；在 Mini Crosswords 里，一个节点可以是“第 3 行候选词填 `RATE`”。

为什么这套机制有效？因为真实问题往往存在两个事实：

第一，中间步骤的错误很常见，但错误通常能在较早阶段暴露。  
第二，只要中间状态能被粗略评估，保留多个候选就比赌单条链路更稳。

BFS 与 DFS 的差别，关键在“扩展顺序”和“资源分配”。

| 维度 | BFS | DFS |
|---|---|---|
| 中文解释 | 先横向看同一层多个分支 | 先顺着一条分支往深处走 |
| 扩展时机 | 每层统一扩展 | 选最有希望的分支立即深入 |
| 剪枝时机 | 每层评估后统一裁掉低分分支 | 走到局部矛盾时立即回退 |
| 优势 | 稳定，适合浅层任务 | 省内存，适合强约束回溯任务 |
| 风险 | 分支数爆炸 | 早期误判可能漏掉正确解 |

BFS 的典型推导逻辑是：如果问题深度较浅，且每一层都能拿到较可信的评分，那么保留 top-$b$ 个分支最合理。设每层候选数为 $k$，深度为 $d$，完全展开成本约为 $O(k^d)$；BFS 通过 beam 式保留，把每层状态数压到 $b$，成本更接近 $O(d\cdot b\cdot k)$。这不是严格等价，但足够说明为什么“限制宽度”是工程上必要的。

玩具例子可以这样看。假设一道 3 步题，每一步模型都能给出 5 个候选思路，若不剪枝，总分支数是 $5^3=125$。如果采用 BFS 且每层只保留 top-5，看似还是 5，但实际已经把“每个节点继续裂变”变成“只让当前最强的 5 个节点进入下一层”，搜索树宽度被硬控住了。

真实工程例子是 Mini Crosswords。这个任务的关键不是“多写几个词”，而是“每填一个词都会改变后续所有横纵约束”。因此 DFS 很自然：先填一个候选词，再检查剩余 clue 是否仍然可能；如果某个填法导致后续格子不可能满足，就立即回溯。这里的搜索不是为了广度覆盖，而是为了尽快发现局部矛盾。

---

## 代码实现

下面给一个简化版 ToT 搜索实现。它不依赖大模型，直接用一个小型玩具状态空间演示 BFS 与评分剪枝的流程。代码里的 `expand` 相当于“生成候选思路”，`score` 相当于“评价器”，`bfs_tot` 相当于“按策略保留 frontier”。

```python
from dataclasses import dataclass

@dataclass
class Node:
    value: int
    steps: tuple[str, ...]

TARGET = 24

def expand(node: Node) -> list[Node]:
    # 玩具例子：每一步可以 +3、*2、-1
    ops = [
        ("+3", node.value + 3),
        ("*2", node.value * 2),
        ("-1", node.value - 1),
    ]
    return [Node(v, node.steps + (name,)) for name, v in ops]

def score(node: Node) -> float:
    # 分数越高表示越接近目标；真实 ToT 中这里通常由 LM 或规则评估器完成
    return -abs(TARGET - node.value)

def bfs_tot(start: int, max_depth: int = 4, breadth: int = 3):
    frontier = [Node(start, ())]
    for _ in range(max_depth):
        candidates = []
        for node in frontier:
            candidates.extend(expand(node))
        candidates.sort(key=score, reverse=True)
        frontier = candidates[:breadth]
        for node in frontier:
            if node.value == TARGET:
                return node
    return max(frontier, key=score)

result = bfs_tot(start=3, max_depth=4, breadth=3)

assert result.value == 24
assert result.steps == ("+3", "*2", "*2")
```

上面这段代码可以直接运行。它展示了 ToT 最重要的三个工程点：

1. 把“思路”存成显式节点，而不是只保留最后答案。
2. 用列表保存当前 frontier，也就是“当前还值得继续扩展的节点集合”。
3. 每一轮统一排序后再截断，这就是最基本的 BFS 剪枝。

如果要改成接近真实 LLM 系统的伪代码，结构通常是这样：

```python
def tot_search(root_state, strategy, max_depth, breadth):
    frontier = [root_state]

    for depth in range(max_depth):
        candidates = []
        for state in frontier:
            thoughts = lm_generate_thoughts(state)      # 生成候选思路
            for thought in thoughts:
                next_state = apply_thought(state, thought)
                value = lm_evaluate_state(next_state)   # 打分或分类
                if value > threshold:
                    candidates.append((next_state, value))

        if strategy == "bfs":
            frontier = select_top_k(candidates, k=breadth)
        elif strategy == "dfs":
            frontier = select_best_single_path(candidates)

        if any(is_solution(s) for s, _ in frontier):
            return best_solution(frontier)

    return best_effort(frontier)
```

真实工程里，DFS 往往还要补两个机制：

| 机制 | 作用 |
|---|---|
| 回溯栈 | 记录走过的分支，失败时退回上一个可选点 |
| 宽松剪枝 | 不把低分直接删光，而是保留少量“次优分支”防止误删 |

如果评分器只给出粗糙标签，例如 `sure / maybe / impossible`，那么不要把一次 `impossible` 当成绝对真理。更稳妥的做法是多次评估投票，或把 `impossible` 分成“强不可能”和“弱不可能”两个阈值。

---

## 工程权衡与常见坑

ToT 的核心难点不是“写出搜索循环”，而是“控制搜索成本并避免错剪”。

最常见的坑是 BFS 宽度失控。假设每个节点扩展 5 个候选，深度 4 时理论上有 $5^4=625$ 个节点；如果每个节点都要调用一次生成和一次评估，成本会很快变成 token 爆炸。工程上通常要同时限制三件事：每节点生成数、每层保留数、最大深度。

第二个坑是 DFS 误剪枝。很多系统会让模型判断“当前状态是否还能解出来”，如果这个判断不准，就会过早放弃正确路径。特别是在文字谜题、代码修复、规划任务中，早期状态往往信息不足，`impossible` 只是“我暂时看不出来”。

下面这个表可以直接当作选型速查表：

| 维度 | BFS 特点 | DFS 特点 | 规避策略 |
|---|---|---|---|
| 内存占用 | 高 | 低 | BFS 控制 breadth；DFS 控制回溯深度 |
| 漏解风险 | 相对低 | 相对高 | DFS 不要依赖单次不可能判断 |
| 适合任务 | 浅层、每步可评估 | 约束强、需要回溯 | 先按任务结构选，再调阈值 |
| 常见失败 | 状态数爆炸 | 走错深路后剪错 | 增加缓存与多轮评估 |

真实工程例子可以继续看 Mini Crosswords。这里每填一个词，都会对交叉格产生硬约束。DFS 的优势是能快速发现“不可能状态”：比如某个竖向单词第二个字母已经被横向填成 `Q`，而 clue 允许的候选库里没有任何匹配词，那么当前分支就该立即回溯。公开分析中，这类基于 DFS 和可行性判断的做法，在 word-level 指标上可达到约 60% 成功率。它的经验很明确：当约束是稠密耦合的，DFS 的“先试一条，发现矛盾立刻退”比 BFS 的“大面积铺开”更划算。

另一个常见坑是重复状态。如果两个不同推理路径达到同一个中间状态，而系统没有做去重，就会浪费大量成本。因此工程里通常要做 state hashing，也就是“把状态编码成可比较的键”，命中后直接复用评分或跳过展开。

---

## 替代方案与适用边界

ToT 不是默认最优解。很多问题其实不需要树搜索。

最简单的替代方案是 CoT。它可以白话理解为“按一条链直接讲清楚步骤”。如果问题只有 2 到 3 步，且每一步都很确定，例如“先算括号，再算乘法”，那就没有必要为了保留多路径而付出额外成本。一个两步算术题，如“某商品 80 元打 75 折后再减 10 元，最终多少钱”，CoT 直接写：$80\times0.75-10=50$，已经足够。

另一个替代方案是 Beam Search。它和 BFS 很像，但更强调“固定保留 top-k 序列”。它适合评分信号较稳定、任务结构更接近序列生成的问题。可以把它看成“受控版 BFS”，但它通常不像完整 ToT 那样强调显式的回溯和自评估。

| 方法 | 任务深度 | 中间可验证性 | 资源开销 | 适用边界 |
|---|---|---|---|---|
| CoT | 短 | 不强依赖 | 低 | 单路径足够、步骤短 |
| Beam Search | 中等 | 有稳定排序信号 | 中 | 序列生成、保留少量候选 |
| ToT | 中到深 | 强依赖 | 高 | 多路径必要、可评估、需回溯 |

因此，选择策略时可以用一个简单判断：

1. 如果问题短、直、稳定，先用 CoT。
2. 如果需要保留少量候选，但不需要复杂回溯，用 Beam Search。
3. 如果错误路径很多，且中间状态可评估，用 ToT。
4. 在 ToT 内部，如果任务浅且早期能判断优劣，优先 BFS；如果约束强、局部矛盾明显、需要回溯，优先 DFS。

从工程角度看，ToT 真正的价值不在“树”这个形式，而在“把推理变成可搜索、可打分、可丢弃的显式状态机”。一旦这个抽象成立，BFS、DFS、启发式打分、阈值剪枝、缓存复用这些经典搜索设计都可以直接接入语言模型系统。

---

## 参考资料

- Yao et al.《Tree of Thoughts: Deliberate Problem Solving with Large Language Models》  
  URL: https://arxiv.org/abs/2305.10601  
  用途：ToT 原始论文，Game of 24、Mini Crosswords 等实验结果的主要来源。

- Emergent Mind, “Tree of Thoughts (ToT) Framework”  
  URL: https://www.emergentmind.com/topics/tree-of-thoughts-tot-framework  
  用途：整理 ToT 的状态表示、搜索策略、评分与剪枝框架，适合快速建立整体结构。

- Daniel Park, “Tree of Thoughts” 相关实验解读  
  URL: https://dsdanielpark.github.io/llm/2023-12-07-TreeOfThought.html  
  用途：补充 Mini Crosswords 等任务的搜索流程与结果说明，便于理解 DFS 剪枝场景。

- Mai Van Hai, “Tree of Thoughts: Search Strategies, Heuristics, Rollout Policies”  
  URL: https://maivanhai.io.vn/tree-of-thoughts-search-strategies-heuristics-rollout-policies/  
  用途：概括 BFS、DFS、启发式评分和 rollout 策略，适合工程实现时做选型参考。
