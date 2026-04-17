## 核心结论

ToT，Tree of Thoughts，直白说就是“把一条线性的思考过程改成一棵可分叉、可回退的思路树”。它解决的不是“模型不会生成答案”，而是“模型只会顺着一条路往下写，走错后很难回头”。

把 MCTS，Monte Carlo Tree Search，直白说就是“用统计分数决定下一步优先试哪条分支”的搜索方法，引入 ToT 后，核心收益是把“继续深挖当前高分思路”和“试一试访问很少但可能更优的新思路”统一到一个公式里。最常见的选择分数是

$$
\text{UCB}_i = \frac{w_i}{n_i} + c\sqrt{\frac{\ln N}{n_i}}
$$

其中 $\frac{w_i}{n_i}$ 是利用项，表示“这条路目前平均表现怎样”；$c\sqrt{\frac{\ln N}{n_i}}$ 是探索项，表示“这条路虽然没那么稳，但还没试够，值得再看一眼”。

RAP，Reasoning via Planning，可以看成“把 ToT 风格的显式思路树，交给 MCTS 来推进”的代表性实现。它让 LLM 同时扮演世界模型和推理引擎，在 Blocksworld 规划任务上，论文报告 LLaMA-33B + RAP 在 2/4/6 步任务上的平均成功率达到 64%，并相对 GPT-4 + CoT 提升约 33%。这说明一件事：当任务需要多步规划、状态跟踪和回溯时，显式树搜索通常比单条 CoT 更稳。

---

## 问题定义与边界

先把问题说清楚。CoT，Chain of Thought，直白说就是“让模型把中间推理步骤写出来”。它适合“沿着一条链一路推下去”的题，但不擅长下面这类任务：

| 任务类型 | 为什么单条 CoT 容易失效 | ToT+MCTS 为什么更合适 |
| --- | --- | --- |
| 多步规划 | 前几步一旦选错，后面整条链都偏掉 | 可以并行保留多个部分解 |
| 需要回溯 | 线性输出天然不擅长撤销错误 | 树结构天然支持回退 |
| 状态依赖强 | 中间变量一旦丢失，后续推理崩溃 | 每个节点都显式记录状态 |
| 分支空间大 | 只试一条路，命中率低 | 用搜索预算分配给更有潜力的分支 |

这里的“边界”也必须讲清楚。ToT+MCTS 不是“任何推理都更强”的通用答案，它主要适用于：

1. 每一步都能定义“当前状态”。
2. 可以生成多个候选下一步。
3. 可以对候选步骤做相对稳定的打分。
4. 错误早期决策会严重影响最终结果。

如果题目本身只需要一条很短的推理链，比如简单问答、轻量信息抽取、模板改写，那么树搜索往往只会增加成本，不会明显提升质量。

术语先记住这一张表：

| 符号 | 含义 | 白话解释 |
| --- | --- | --- |
| $w_i$ | 累计奖励 | 这条分支历史上总共拿了多少分 |
| $n_i$ | 节点访问次数 | 这条分支被试过多少次 |
| $N$ | 父节点访问次数 | 当前决策点一共试了多少轮 |
| $c$ | 探索常数 | 决定系统有多愿意尝试“还没试够”的路线 |

一个玩具例子是 Game of 24。给定四个数，目标是通过加减乘除得到 24。普通 CoT 可能先写出一个看上去顺的算式，然后一路算下去；ToT 会在每一步保留多个候选算式；MCTS 再用 UCB 决定下一轮优先扩展哪个候选。即使某条算式当前得分略低，只要它访问次数很少，探索项仍可能把它推到前面。

---

## 核心机制与推导

把 ToT 和 MCTS 接起来，最自然的做法是把“一个部分思路”视为一个树节点。节点状态通常包含两部分：

1. 原始问题。
2. 到当前为止已经生成的思路或动作序列。

然后每轮搜索做四件事，这就是 MCTS 的标准骨架：

1. 选择，Selection：从根节点开始，沿着 UCB 最大的子节点往下走。
2. 扩展，Expansion：在叶子节点让 LLM 生成多个候选 thought 或 action。
3. 评估，Evaluation / Rollout：让 LLM 或任务奖励函数给新节点打分。
4. 回传，Backpropagation：把这次结果沿路径向上更新 $w_i$ 和 $n_i$。

为什么 UCB 能平衡探索和利用，可以直接看公式。

$$
\text{UCB}_i = \frac{w_i}{n_i} + c\sqrt{\frac{\ln N}{n_i}}
$$

第一项是平均奖励，越大说明“这条路历史表现越稳定”。第二项在 $n_i$ 很小时会更大，意思是“虽然这条路证据少，但不能太早放弃”。随着父节点总访问数 $N$ 增长，$\ln N$ 增长很慢；随着某个子节点被反复访问，分母 $n_i$ 增长更快，所以探索项最终会收敛。这正是它有效的原因：前期鼓励多试，后期逐步收敛到更优分支。

看一个最小数值例子。设父节点总访问 $N=25$，有两个候选：

- child1: $w_1=10, n_1=20$
- child2: $w_2=2, n_2=5$
- $c=\sqrt{2}\approx1.414$

则

$$
\text{UCB}_1 = \frac{10}{20} + 1.414\sqrt{\frac{\ln 25}{20}} \approx 1.14
$$

$$
\text{UCB}_2 = \frac{2}{5} + 1.414\sqrt{\frac{\ln 25}{5}} \approx 1.54
$$

虽然 child2 的平均奖励更低，但因为访问更少，探索项更大，所以会被优先继续扩展。这就是“潜力股优先再试一次”的数学表达。

真实工程例子是 Blocksworld。它是一个经典规划任务，直白说就是“在一堆积木块之间按规则搬动，最终满足目标摆放关系”。RAP 的做法不是让模型直接写完整答案，而是让模型：

1. 把当前积木状态描述出来。
2. 生成可执行动作。
3. 预测动作后的新状态。
4. 根据目标是否更接近来给奖励。
5. 用 MCTS 决定下一步优先扩哪条计划。

这里最关键的一点不是“树搜索”本身，而是“状态必须被显式维护”。如果模型只写动作，不写动作后的状态，那么回溯时父节点根本不知道哪里出了错，搜索树就退化成了胡乱枚举。

---

## 代码实现

下面给一个最小可运行版本。它不调用真实 LLM，而是用固定数据模拟“候选 thought 的奖励与访问次数”，演示 UCB 选择和回传更新。代码重点不是完整 MCTS，而是最核心的统计量更新逻辑。

```python
from math import log, sqrt

class Node:
    def __init__(self, name, reward_sum=0.0, visits=0, parent=None):
        self.name = name
        self.reward_sum = reward_sum
        self.visits = visits
        self.parent = parent
        self.children = []

    def avg_reward(self):
        return self.reward_sum / self.visits if self.visits > 0 else 0.0

    def ucb(self, c=1.414):
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        return self.avg_reward() + c * sqrt(log(parent_visits) / self.visits)

def select_best_child(node, c=1.414):
    return max(node.children, key=lambda child: child.ucb(c))

def backpropagate(node, reward):
    while node is not None:
        node.visits += 1
        node.reward_sum += reward
        node = node.parent

# 玩具树：模拟 Game of 24 某一步的三个候选 thought
root = Node("root", visits=25)

a = Node("thought_a", reward_sum=10, visits=20, parent=root)
b = Node("thought_b", reward_sum=2, visits=5, parent=root)
c = Node("thought_c", reward_sum=0, visits=0, parent=root)

root.children = [a, b, c]

# 未访问节点先被扩展，这是 MCTS 的常见约定
assert select_best_child(root).name == "thought_c"

# 假设先扩展 c，并得到一次中等奖励
backpropagate(c, reward=0.8)

assert c.visits == 1
assert root.visits == 26

# 现在 c 已访问过，再比较 UCB，通常会在 a/b/c 之间重新权衡
best = select_best_child(root)
assert best.name in {"thought_a", "thought_b", "thought_c"}

# 验证论文常见的数值示例：访问少的分支可能被优先选择
test_root = Node("test_root", visits=25)
child1 = Node("child1", reward_sum=10, visits=20, parent=test_root)
child2 = Node("child2", reward_sum=2, visits=5, parent=test_root)
test_root.children = [child1, child2]

assert child2.ucb() > child1.ucb()
print("UCB selection works.")
```

把它映射到真实系统时，流程通常是这样的：

| 步骤 | LLM 负责什么 | MCTS 负责什么 |
| --- | --- | --- |
| 生成候选 | 给当前状态生成 $k$ 个 thought/action | 把候选挂到树上 |
| 节点评估 | 对候选自评，或预测下一状态并算 reward | 保存 reward_sum |
| 选择分支 | 不负责最终决策 | 用 UCB 选下一个 child |
| 回传更新 | 不直接处理整棵树统计 | 更新沿途节点的 $w_i, n_i$ |

工程里常见的伪代码可以写成：

```python
for _ in range(num_simulations):
    path = select_by_ucb(root, c)
    leaf = path[-1]
    children = lm_generate_candidates(leaf.state, k=3)
    scored_children = lm_or_env_evaluate(children)
    expand(leaf, scored_children)
    best_new_child = select_by_ucb(leaf, c)
    reward = rollout_or_direct_score(best_new_child)
    backpropagate(best_new_child, reward)
```

如果任务像 RAP 一样有明确状态转移，那么 `lm_generate_candidates` 更像“生成动作”，`lm_or_env_evaluate` 更像“预测动作后状态并计算离目标有多近”。如果任务像数学推理，更像“生成下一步 thought，再自评这一步是否让答案更接近”。

---

## 工程权衡与常见坑

第一类坑是 $c$ 调参。它不是越大越好，也不是固定 1.414 就万事大吉。

| $c$ 的范围 | 倾向 | 常见现象 |
| --- | --- | --- |
| 过小，如 0.2 | 偏利用 | 很快锁死在早期高分分支，后面难纠错 |
| 中等，如 0.7 到 1.4 | 相对平衡 | 适合先做小规模 sweep |
| 过大，如 2.0 以上 | 偏探索 | 预算被大量耗在低价值未试节点上 |

第二类坑是 reward 设计。很多团队以为“只要让模型自己打分”就够了，实际往往不够。因为模型自评分数可能高相关但不稳定，尤其在开放文本任务里更明显。更稳的做法是把 reward 拆成两部分：

1. 模型自评，这一步是否合理。
2. 任务启发式，这一步是否更接近目标。

RAP 在 Blocksworld 上能工作，一个关键原因就是它不是只看语言流畅度，而是看状态是否真的更接近目标条件。

第三类坑是世界模型不完整。世界模型，直白说就是“系统对当前状态和动作后果的内部表示”。在 Blocksworld 里，至少要显式记录：

| 必填状态变量 | 漏掉后会怎样 |
| --- | --- |
| 每个块在哪里 | 后续动作是否合法无法判断 |
| 哪个块顶部为空 | `pick`、`stack` 可能被错误执行 |
| 当前目标条件满足了几个 | reward 无法稳定计算 |
| 已执行动作历史 | 容易重复走回头路 |

一个典型失误是：团队把 $c$ 设成 2.0，希望“多探索更安全”，但 prompt 里没有强制模型写出完整状态，于是系统不断扩展新分支，却没有可靠依据判断这些分支是不是有效。结果不是“更聪明”，而是“更贵地乱试”。通常先把 $c$ 调回 1.0 左右，再把状态变量写入模板，收益比盲目增加搜索轮数更明显。

第四类坑是预算失控。ToT+MCTS 的成本大致与下面几个量相乘：

$$
\text{总成本} \approx \text{搜索轮数} \times \text{每次扩展候选数} \times \text{每次评估调用成本}
$$

所以工程上常用三个限流手段：

1. 限制最大深度。
2. 每层只保留 top-k 候选。
3. 提前停止，找到足够高 reward 的解就返回。

---

## 替代方案与适用边界

不是所有任务都该直接上 ToT+MCTS。可以先按任务结构选策略。

| 方案 | 适用场景 | 优点 | 缺点 |
| --- | --- | --- | --- |
| CoT | 短链条、低分支推理 | 成本最低 | 一旦走错很难回头 |
| ToT + DFS | 分支不多、需要回溯 | 实现简单 | 容易钻进单一路线 |
| ToT + BFS | 浅层搜索、需要广覆盖 | 不容易漏掉浅层解 | 很快爆炸 |
| ToT + UCB/MCTS | 多步规划、状态依赖强、预算有限 | 能动态平衡探索与利用 | 系统最复杂，调参成本高 |

所以边界可以直接记成一句话：

- 如果问题只有单条短路径，CoT 就够了。
- 如果需要显式保留多个思路，但分支很少，ToT + DFS/BFS 就够了。
- 如果既要多分支搜索，又不能把预算平均撒出去，就该上 MCTS。

对初级工程师来说，最实用的判断标准不是“这方法先进不先进”，而是“错误的前两步会不会毁掉后面全部步骤”。如果答案是会，而且你还能明确写出状态、动作和奖励，那么 ToT 与 MCTS 的结合就很有价值。反过来，如果状态根本不可定义，reward 也只能靠主观印象打分，那么树搜出来的往往不是更可靠的答案，只是更昂贵的尝试记录。

---

## 参考资料

- Tree of Thoughts: Deliberate Problem Solving with Large Language Models. arXiv, 2023. https://arxiv.org/abs/2305.10601
- Reasoning with Language Model is Planning with World Model. EMNLP 2023, ACL Anthology. https://aclanthology.org/2023.emnlp-main.507/
- Finite-time Analysis of the Multiarmed Bandit Problem. Machine Learning, 2002. DOI: https://doi.org/10.1023/A:1013689704352
- Bandit Based Monte-Carlo Planning. ECML 2006. DOI: https://doi.org/10.1007/11871842_29
