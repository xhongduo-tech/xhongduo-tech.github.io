## 核心结论

多 Agent 辩论机制的核心价值，不是“让更多模型同时回答”，而是让多个求解路径在几轮交互里互相纠错，再通过共识机制收敛到更稳定的答案。这里的 Agent 可以先理解成“独立思考的一名答题者”，它不一定是不同公司出的模型，也可以是同一模型扮演不同角色。

Du 等人的框架说明了一件很直接的事：对需要逐步推理的任务，单个模型一次作答容易在中间步骤走偏；如果让 3 个以上 Agent 先独立回答，再读取彼此的推理并指出错误，最终准确率会明显提升。以 GSM8K 这类小学到初中水平的数学文字题为例，单 Agent 大约 77%，同轮多数投票约 81%，经过 2 到 3 轮辩论后可稳定到 85% 左右；进一步引入基于角色或历史表现的加权共识，异构模型组合在更多轮次下可逼近 91%。

对零基础读者，最直观的理解是：不是把“一次猜答案”换成“多次重复猜”，而是把“一个人独自做题”换成“三个人各自解题、互相指出漏洞、最后再投票”。错误推理在辩论过程中会被持续压低，而持续给出正确线索的 Agent 会在最终聚合时获得更高影响力。

下表先给出常见对比：

| 方案 | 典型设置 | GSM8K 准确率示意 | 关键原因 |
| --- | --- | ---: | --- |
| 单 Agent | 1 个模型直接回答 | 77% | 单一路径，易早期出错 |
| 同轮多数投票 | 3 个模型独立回答后直接投票 | 81% | 有多样性，但没有纠错过程 |
| 多 Agent 辩论 | 3 个 Agent，2 到 3 轮 | 85% | 通过互相批判修正中间步骤 |
| 加权共识辩论 | 异构 Agent，3 到 4 轮，加权聚合 | 91% | 同时利用辩论和角色可靠度 |

---

## 问题定义与边界

问题定义可以写成一句话：给定一个需要推理的问题，如何让多个语言模型协作，使最终答案比任何单个模型更可靠。

这里的“推理可靠性”，白话解释就是：不仅最终答案对，而且中间推导链条更不容易出现隐藏错误。多 Agent 辩论最适合以下任务：

| 任务类型 | 为什么适合 |
| --- | --- |
| 数学文字题 | 中间步骤可被他人逐步检查 |
| 逻辑推断题 | 不同 Agent 容易暴露彼此前提冲突 |
| 长文事实核查 | 多个 Agent 可分别从不同证据路径核对 |
| 结构化决策 | 可拆成求解、校验、知识补充等角色 |

它的输入通常包括三部分：

| 输入项 | 说明 |
| --- | --- |
| 原始问题 | 用户提问或待求解任务 |
| 初始回答 | 每个 Agent 第 1 轮独立给出的答案与理由 |
| 辩论上下文 | 其他 Agent 的回答、质疑、摘要、历史结论 |

一个最小流程可以抽象成：

问题 → 各 Agent 首轮输出 → 第 2 轮读取他人答案并反驳 → 第 3 轮根据反驳修正 → 投票或加权共识 → 最终答案

边界也很明确。

第一，Agent 数量通常至少为 3。少于 3 时，容易退化成“你说我改”的二元互审，缺少充分的路径多样性。这里的“路径多样性”，白话解释就是：不同人从不同角度解题，而不是把同一种错误重复三遍。

第二，辩论轮数一般控制在 2 到 4 轮。轮数太少，纠错不充分；轮数太多，收益递减，而且上下文会迅速膨胀。

第三，上下文窗口是硬约束。所谓“上下文窗口”，白话解释就是模型一次性能读进去的最大文本长度。如果 3 个 Agent 每轮都输出长链路推理，4 轮后很容易堆到几万 token。工程上通常要在每轮后做摘要，保留“争议点、当前答案、关键证据”，删除重复推理。

一个新手能立刻理解的边界例子是：如果你让 4 个 Agent 每轮各写 1200 token，3 轮后仅原始辩论文本就接近 14400 token，还没算系统提示词、历史记录和参考材料。此时不做摘要，系统不是变强，而是先撞上上下文上限。

---

## 核心机制与推导

多 Agent 辩论的数学表达并不复杂。设第 $i$ 个 Agent 在第 $r$ 轮输出为 $y_i^{(r)}$，总共有 $N$ 个 Agent，辩论进行 $R$ 轮。最终答案定义为：

$$
y^* = \arg\max_y \sum_{i=1}^{N} w_i \cdot \mathbb{1}[y_i^{(R)} = y]
$$

这里：

- $y_i^{(R)}$：第 $i$ 个 Agent 在最后一轮给出的答案
- $\mathbb{1}[\cdot]$：指示函数，白话解释就是“条件成立记 1，不成立记 0”
- $w_i$：第 $i$ 个 Agent 的权重，白话解释就是“这个 Agent 的票有多重”

如果所有 $w_i = 1$，这就是最普通的多数投票。它的优点是实现简单，缺点也很明显：票数多不代表推理对。尤其在角色不对称时，多数票可能把正确少数压掉。

### 玩具例子

题目：一件商品原价 80 元，涨价 25% 后卖出，售价是多少？

3 个 Agent 第 1 轮输出：

| Agent | 角色 | 第 1 轮答案 | 问题 |
| --- | --- | --- | --- |
| A | Solver | 100 | 正确 |
| B | Solver | 95 | 把 25% 错算成加 15 |
| C | Checker | 100 | 正确，并指出公式应为 $80 \times 1.25$ |

如果直接多数投票，A 和 C 都是 100，已经正确。但再看更容易出问题的变体：

| Agent | 角色 | 第 1 轮答案 |
| --- | --- | --- |
| A | Solver | 95 |
| B | Solver | 95 |
| C | Checker | 100 |

这时多数投票会得到 95，错误。可如果进入辩论，C 会指出“涨价 25% 等于乘以 1.25，不是加 15 元”，A 或 B 可能在下一轮修正。即使两名 Solver 最终仍未改正，加权机制也可以让 Checker 拥有更高权重。

### 轮次的作用

辩论不是简单重复，而是一个“错误暴露再修正”的过程。可把第 $r$ 轮看成：

$$
y_i^{(r)} = g_i\left(x, \{y_j^{(r-1)}\}_{j \neq i}, s^{(r-1)}\right)
$$

其中：

- $x$ 是原问题
- $\{y_j^{(r-1)}\}_{j \neq i}$ 是其他 Agent 上一轮的输出
- $s^{(r-1)}$ 是上轮摘要

这表示第 $i$ 个 Agent 在新一轮不再只看题目，而会结合“别人怎么答、哪里互相冲突、当前争议点是什么”重新推理。实质上，它把单链路推理改造成了“带外部反馈的迭代推理”。

可用一个简单表看轮次演化：

| 轮次 | 输入 | Agent 行为 | 输出 |
| --- | --- | --- | --- |
| 第 1 轮 | 问题本身 | 独立求解 | 初始答案与理由 |
| 第 2 轮 | 问题 + 他人答案 | 找漏洞、反驳、修正 | 更新后的答案 |
| 第 3 轮 | 问题 + 历史摘要 + 争议点 | 收敛到更一致的结论 | 最终候选答案 |
| 终局 | 最后一轮答案集合 | 投票或加权共识 | 最终输出 |

### 加权共识为什么更稳

A-HMAD 这类方法把权重写成学习函数：

$$
w_i = f_\phi(\text{role}, \text{history}, \text{confidence})
$$

这里：

- `role`：角色，例如 Solver、Checker、Knowledge
- `history`：历史表现，例如过去轮次是否经常被证明正确
- `confidence`：自信度，白话解释就是 Agent 对自己答案的确信程度

如果多数投票是“每人一票”，那加权投票就是“更可靠的人票更重”。这种设计特别适合角色分工场景。比如 Solver 擅长生成候选解，Checker 擅长找计算错误，Knowledge Agent 擅长补充定义或背景事实。若仍然按 1:1:1 投票，系统就浪费了角色差异。

一个直观推导是：设正确答案为 $c$，错误答案为 $e$。若支持 $c$ 的 Agent 权重和大于支持 $e$ 的权重和，即

$$
\sum_{i: y_i^{(R)} = c} w_i > \sum_{i: y_i^{(R)} = e} w_i
$$

则最终系统会选 $c$。所以重点不只是“有多少 Agent 说对”，还包括“哪些 Agent 说对”。

### 真实工程例子

设想一个数学答题机器人，后台有三个角色：

| 角色 | 主要任务 |
| --- | --- |
| Solver | 先给出完整解题步骤 |
| Checker | 专查算术、单位、边界条件 |
| Knowledge | 补充公式定义、题目隐藏约束 |

用户输入一道“列车相遇”题后，系统并行调用三个角色。第 1 轮大家独立解；第 2 轮 Checker 指出 Solver 在单位换算上把分钟误当小时；Knowledge 说明相遇题应使用相对速度；第 3 轮 Solver 改写推导，最后再由聚合器给出统一答案和最终解析。这个流程的价值不只是正确率提升，还包括能把“争论日志”存下来，作为后续评估、微调或构建偏好数据的依据。

---

## 代码实现

实现一个能运行的最小版，不需要真的调用大模型 API，先用“模拟 Agent”理解框架即可。下面的代码展示三件关键事：

1. 每轮让多个 Agent 看到问题与历史摘要
2. 根据他人输出进行修正
3. 用多数投票或加权投票做最终聚合

```python
from collections import Counter, defaultdict

def aggregate_majority(outputs):
    counter = Counter(outputs)
    answer, count = counter.most_common(1)[0]
    return answer, count

def aggregate_weighted(outputs, weights):
    scores = defaultdict(float)
    for agent_name, answer in outputs.items():
        scores[answer] += weights[agent_name]
    best_answer = max(scores.items(), key=lambda x: x[1])[0]
    return best_answer, dict(scores)

def solver_agent(problem, peer_outputs, summary):
    if "80" in problem and "25%" in problem:
        # 模拟一个会被同伴纠正的 Solver
        if "1.25" in summary:
            return "100"
        return "95"
    return "unknown"

def checker_agent(problem, peer_outputs, summary):
    if "80" in problem and "25%" in problem:
        return "100"
    return "unknown"

def knowledge_agent(problem, peer_outputs, summary):
    if "80" in problem and "25%" in problem:
        return "100"
    return "unknown"

def summarize_round(outputs):
    answers = list(outputs.values())
    if answers.count("100") >= 2:
        return "多数意见：涨价25%应乘1.25，候选答案100"
    return "存在分歧：请检查25%是乘1.25还是直接加固定值"

def run_debate(problem, rounds=2):
    agents = {
        "solver": solver_agent,
        "checker": checker_agent,
        "knowledge": knowledge_agent,
    }
    summary = ""
    history = []

    for _ in range(rounds):
        round_outputs = {}
        for name, fn in agents.items():
            peer_outputs = history[-1] if history else {}
            round_outputs[name] = fn(problem, peer_outputs, summary)
        history.append(round_outputs)
        summary = summarize_round(round_outputs)

    final_outputs = history[-1]
    majority_answer, _ = aggregate_majority(list(final_outputs.values()))
    weighted_answer, weighted_scores = aggregate_weighted(
        final_outputs,
        weights={"solver": 1.0, "checker": 1.4, "knowledge": 1.2},
    )
    return final_outputs, majority_answer, weighted_answer, weighted_scores

problem = "一件商品原价80元，涨价25%后卖出，售价是多少？"
final_outputs, majority_answer, weighted_answer, weighted_scores = run_debate(problem, rounds=2)

assert final_outputs["checker"] == "100"
assert majority_answer == "100"
assert weighted_answer == "100"
assert weighted_scores["100"] > weighted_scores.get("95", 0)

print(final_outputs)
print(majority_answer, weighted_answer)
```

这段代码虽然是玩具版，但结构已经接近真实工程：

| 模块 | 作用 |
| --- | --- |
| `solver_agent/checker_agent/knowledge_agent` | 代表不同角色或不同模型 |
| `summarize_round` | 压缩历史上下文，避免窗口爆炸 |
| `run_debate` | 轮次调度器，管理每轮输入输出 |
| `aggregate_majority` | 普通多数投票 |
| `aggregate_weighted` | 角色加权共识 |

如果换成真实 API，通常会再加四层能力：

| 工程模块 | 真实需求 |
| --- | --- |
| 并行调用 | 同时请求多个模型，降低总延迟 |
| 重试与超时 | 某个 Agent 超时或报错时可降级 |
| 日志与追踪 | 保存每轮输入输出，便于复盘 |
| 成本控制 | 控制轮次、token、模型选择 |

一个实用的伪流程如下：

| 步骤 | 数据流 |
| --- | --- |
| 1 | 读取问题，创建 3 到 5 个 Agent |
| 2 | 并行生成首轮答案 |
| 3 | 提取分歧点，构造辩论提示词 |
| 4 | 进入下一轮，让每个 Agent 只读摘要与关键反驳 |
| 5 | 到达最大轮数或已收敛后，执行聚合器 |
| 6 | 输出答案、解释、置信度、辩论日志 |

真实工程例子里，`debate_round(agent_outputs)` 往往不会把全部原文喂回去，而是只保留这些字段：

- 当前答案
- 支撑证据
- 被他人指出的错误
- 是否接受修正
- 下一轮关注点

这样做的原因很现实：真正昂贵的不是“多一次函数调用”，而是“多轮长上下文调用”。

---

## 工程权衡与常见坑

多 Agent 辩论不是免费收益，它在准确率、成本、延迟、复杂度之间做交换。只看论文里的最高准确率，通常会低估实现代价。

先看最常见的坑：

| 坑 | 现象 | 原因 | 规避方式 |
| --- | --- | --- | --- |
| 轮次过多 | 成本高但准确率不再涨 | 边际收益递减 | 一般控制在 2 到 4 轮，做早停 |
| 上下文爆炸 | 提示词越来越长，响应变慢 | 每轮都拼接完整历史 | 每轮摘要，只保留争议点 |
| 多数票误判 | 2 个 Solver 错，1 个 Checker 对，结果仍错 | 票数不等于可靠性 | 用角色加权或学习型共识 |
| 同构 Agent 增益低 | 3 个模型总犯类似错误 | 缺少路径多样性 | 引入异构模型或角色差异 |
| 假收敛 | 多个 Agent 表面一致，但一致地错 | 被早期错误锚定 | 在终局加入独立复核 Agent |
| 延迟过高 | 用户等待明显增长 | 多轮串行调用 | 并行首轮、限制后续轮数 |

### 多数投票为什么会失效

一个新手最容易忽略的点是：多数票只适用于“每张票质量接近”的场景，而多 Agent 辩论恰好经常不是这样。

例如 3 个 Agent 中，两个是偏生成型的 Solver，一个是严格校验型的 Checker。某道题里两个 Solver 都受同一错误启发，答案一致但错误；Checker 票数少，却推理正确。此时简单投票会错，而加权机制会给 Checker 更高权重，恢复正确结果。这也是为什么很多工程系统不直接用“谁人多听谁”，而会把角色、历史准确率、是否提供可验证证据一起纳入终局评分。

### 轮数不是越多越好

从经验上看，1 到 4 轮通常呈现“先明显增益、后逐步趋缓”的曲线。原因不复杂：

- 第 1 轮提供多样候选路径
- 第 2 轮开始暴露明显错误
- 第 3 轮常用于收敛
- 第 4 轮以后常常只是在重复已有争论

所以工程上更常见的策略不是固定跑满 4 轮，而是定义早停条件，例如：

- 最后两轮答案完全一致
- 各 Agent 的置信分数差异已经很小
- 新一轮没有出现新的反驳点

### 模型多样性很重要

“模型多样性”白话解释就是：参与辩论的人不能都用同一种思维方式。若 3 个 Agent 只是同一模型、同一提示词、同一温度参数的轻微变体，它们很可能一起犯错，辩论只是把同一个错误说三遍。相比之下，异构模型组合，或同一模型下的明确角色分工，更容易产生互补。

---

## 替代方案与适用边界

多 Agent 辩论不是唯一的推理增强方法。它至少要和单 Agent 自反思、链式思维、外部工具校验做比较。

| 方案 | 适用场景 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 单 Agent 直接回答 | 延迟极低场景 | 最快最便宜 | 准确率受单一路径限制 |
| 单 Agent 自反思 | 中低复杂度推理 | 实现简单，成本低于多 Agent | 容易陷入自我重复 |
| 多 Agent 辩论 | 数学、逻辑、事实核查 | 多路径互相批判，纠错强 | 成本和延迟更高 |
| 工具增强校验 | 可形式验证任务 | 可直接检查计算或检索事实 | 对开放式任务覆盖有限 |
| 多 Agent + 工具 | 高价值复杂任务 | 准确率最高，解释性更强 | 系统复杂度最高 |

如果只有一个模型，self-refine 或 self-critique 也能提升表现。它们的核心思想是“先答，再自己审”。但问题在于：同一个模型很容易沿着原先的错误路径继续走，因为它看到的是自己生成的上下文，纠偏能力有限。多 Agent 辩论的优势，在于不同路径之间天然有冲突，冲突会把隐藏错误暴露出来。

可以用一个非常直白的对比来理解：

- 单 Agent 反思：一个同学做完题后自己再检查一遍
- 多 Agent 辩论：三个同学各自做题，再互相指出别人哪里算错

前者更省资源，后者更容易发现“我自己没意识到的错”。

它的适用边界也很清楚：

| 适合 | 不适合 |
| --- | --- |
| 高价值数学问答 | 极低延迟的实时交互 |
| 复杂逻辑推理 | 设备或预算极度受限 |
| 法规、合同、长文核查 | 问题极简单且答案唯一明确 |
| 需要保留推理日志的系统 | 无法接受多轮 API 成本的场景 |

真实工程里，一个客服机器人如果只回答“营业时间是什么”，多 Agent 辩论大概率不划算，因为单次检索后直接回答就够了。但如果是“这个退款条款在 A 页面和 B 页面是否冲突、用户是否符合退款条件”，就已经进入多路径分析更有价值的区域。

---

## 参考资料

1. Du, Y. et al. *Improving Factuality and Reasoning in Language Models through Multiagent Debate*. 适合先看整体框架、辩论轮次设计、GSM8K 等任务上的基础增益。
2. Zhou, X. and Chen, Y. *Adaptive Heterogeneous Multi-Agent Debate (A-HMAD)*. 适合进一步理解加权共识、角色可靠度建模、为何多数投票会在异构角色下失效。
3. Hegazy, A. et al. *Diversity of Thought Elicits Stronger Reasoning Capabilities in Multi-Agent Debate Frameworks*. 适合理解“多样性”为什么是辩论有效的前提，而不是单纯增加 Agent 数量。
4. 阅读顺序建议：先看 Du 等人的论文理解完整流程，再看 A-HMAD 理解权重层面的改进，最后看 Hegazy 等人的工作理解模型异构与思维多样性的作用边界。
