## 核心结论

AgentBench 的价值，不在于继续问“模型会不会聊天”，而在于直接问“模型能不能完成任务”。它把大语言模型当成一个会观察环境、选择动作、执行步骤的智能体，然后放进 8 个不同的交互场景里做事，最后用任务结果打分。

这 8 个场景覆盖了多轮对话、网页浏览、网页购物、操作系统、数据库、知识图谱、卡牌博弈和家居控制等典型任务。它们共同回答一个更接近工程落地的问题：同一个模型，离开纯文本问答后，是否还能稳定地感知状态、规划步骤、调用动作并完成目标。

如果要用一句话概括 AgentBench，可以这样理解：它不是在测“像不像人”，而是在测“能不能把事做完”。想知道某模型能不能接手运维任务？就让它依次跑 OS、DB、KG 等真实任务，把成功率、F1、reward 全部跑一遍，最后直接出综合分。

下表先给出八个场景和典型指标：

| 场景 | 含义 | 典型任务 | 主要指标 |
|---|---|---|---|
| OS | 操作系统环境，指命令行或系统操作任务 | 查日志、改文件、运行命令 | SR |
| DB | 数据库环境，指结构化查询与数据操作 | 写 SQL、修查询、取结果 | SR |
| KG | 知识图谱环境，指基于实体关系的检索与推理 | 找实体、走关系链 | F1 |
| DCG | 卡牌/战斗游戏环境，指带策略和损耗的对抗任务 | 出牌、攻击、防御 | $0.7 \times WinRate + 0.3 \times DamageRate$ |
| HH | 家居控制环境，指多设备、多状态下的操作 | 开灯、调温、切换模式 | SR |
| WS | 网页购物环境，指在电商页面完成搜索和下单流程 | 搜商品、筛选、加入购物车 | reward / step-success |
| WB | 网页浏览环境，指在开放网页中查找信息 | 打开页面、点击链接、提取答案 | reward / step-success |
| Dialog | 多轮对话环境，指持续轮次下的任务推进 | 约会面、问路、任务协作 | 任务成功率或对话完成度 |

这里的 SR 是 Success Rate，白话解释就是“总任务里做成了多少个”；F1 是一种同时考虑查准率和查全率的综合指标，白话解释就是“既不能乱答，也不能漏答”；reward 是环境给出的累计奖励，白话解释就是“动作对不对，由环境即时记分”。

---

## 问题定义与边界

AgentBench 解决的问题很具体：如果一个 LLM 不只是回答问题，而是要作为“动作执行者”去操作外部环境，怎样客观评估它的能力？

传统聊天评测常看流畅度、相关性、安全性，但这类指标很难说明一个模型是否真的能执行任务。一个模型可能回答得很自然，却在真实环境里连续做出无效动作。例如，它可以清楚解释“怎样查看 Linux 磁盘占用”，但真正进入操作系统任务时，仍可能把命令写错、路径理解错，或者做了一半忘记目标。

因此，AgentBench 的边界很明确：

| 评测对象 | 测什么 | 不直接测什么 |
|---|---|---|
| LLM 作为智能体 | 感知环境、选择动作、完成任务 | 纯文本闲聊质量 |
| 多步交互流程 | 长链路执行稳定性 | 单轮问答记忆点 |
| 不同领域任务 | 跨环境泛化能力 | 某一项技能的极限上限 |
| 有反馈环境 | 错误恢复与迭代 | 只靠人工主观打分的表达效果 |

它的评估范围限定在 8 类交互环境中，而不是覆盖所有现实工作。也就是说，AgentBench 可以帮助你判断模型是否具备“执行型能力”，但不能直接替代所有业务验收。

一个玩具例子很容易说明这个边界。假设你要验证模型能否执行一个简单运维任务：

- 目标：找到 `/var/log` 下最大的日志文件。
- 模型需要做的事：列目录、看文件大小、排序、输出结果。
- 评价方式：任务完成记 1，失败记 0。

如果一共跑 10 个类似任务，模型做成 7 个，那么：

$$
SR = \frac{7}{10} = 0.7
$$

这比“它会不会解释 `du -sh` 命令”更接近真实工程需求。因为线上系统并不关心模型解释得多漂亮，只关心它是否稳定完成任务。

AgentBench 的总体流程可以抽象成：

$$
任务 \rightarrow 环境反馈 \rightarrow 环境内指标 \rightarrow 归一化 \rightarrow 加权总分
$$

也就是先在每个场景里用最合适的指标打分，再把不同场景的分数拉到同一量级，最后汇总成综合分。这样做的目的，是避免某些场景因为天然分值范围大，就在总分里“声音过大”。

---

## 核心机制与推导

AgentBench 的核心机制分两层：先做“环境内评估”，再做“跨环境汇总”。

第一层是环境内评估。不同环境的任务性质不同，所以指标也不同。如果强行用一个统一指标，会丢失关键信息。

最常见的三类公式如下。

Success Rate：

$$
SR = \frac{\text{成功任务数}}{\text{总任务数}}
$$

白话解释：给定一批任务，统计真正完成了多少。

F1：

$$
Precision = \frac{TP}{TP+FP}, \quad Recall = \frac{TP}{TP+FN}
$$

$$
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
$$

这里的 TP、FP、FN 分别是真正例、假正例、假反例。白话解释：Precision 看“答对的答案里有多少是真的”，Recall 看“该答出来的东西答出来了多少”，F1 把这两个维度合并。

DCG 场景中的复合指标：

$$
Score_{DCG} = 0.7 \times WinRate + 0.3 \times DamageRate
$$

白话解释：赢没赢最重要，所以胜率权重大；但只看输赢太粗，还要看造成的伤害水平。

第二层是跨环境汇总。难点在于，不同环境的分数量纲不同。比如 SR 的范围通常是 0 到 1，reward 可能是几十到几百，F1 也是 0 到 1，但分布特征又不同。如果直接平均，reward 大的环境会主导总分。

因此需要归一化。一个常见思路是：对每个环境，先用参与比较的一组模型得分求均值，再让当前模型的分数除以这个均值：

$$
score^{norm}_i = \frac{score_i}{\operatorname{mean}(score_i^{all\ models})}
$$

白话解释：不看绝对值，而看“你相对于该环境平均水平高多少”。

然后按预设权重做加权平均：

$$
TotalScore = \sum_{i=1}^{n} w_i \cdot score^{norm}_i
$$

其中 $w_i$ 是第 $i$ 个环境的权重，满足：

$$
\sum_{i=1}^{n} w_i = 1
$$

这样总分才有可比性。

看一个最小数值例子。假设某模型的分数如下：

- OS：10 个任务完成 7 个，$SR = 0.7$
- KG：$Precision = 0.8, Recall = 0.6$
- DCG：$WinRate = 0.5, DamageRate = 0.9$

则 KG 的 F1 为：

$$
F1 = \frac{2 \times 0.8 \times 0.6}{0.8 + 0.6}
= \frac{0.96}{1.4}
\approx 0.686
$$

DCG 得分为：

$$
Score_{DCG} = 0.7 \times 0.5 + 0.3 \times 0.9 = 0.62
$$

再假设基线模型在三个环境上的平均分分别是：

- OS 平均：0.5
- KG 平均：0.49
- DCG 平均：0.50

那么归一化后：

- OS：$0.7 / 0.5 = 1.4$
- KG：$0.686 / 0.49 \approx 1.40$
- DCG：$0.62 / 0.50 = 1.24$

如果三者等权，那么总分就是：

$$
TotalScore = \frac{1.4 + 1.40 + 1.24}{3} \approx 1.35
$$

这个 1.35 的含义不是“得了 135 分”，而是“整体表现约为基线均值的 1.35 倍”。

这里有一个关键认识：AgentBench 不追求“单一绝对真值”，而追求“跨环境、跨模型的可比较性”。这也是它适合模型选型、版本回归和工程验收的原因。

---

## 代码实现

实现一个简化版 AgentBench，至少要有两层模块：

- 任务 runner：负责把模型放进环境里，循环执行动作，直到成功、失败或超时。
- 指标模块：负责把原始执行轨迹转成 SR、F1、reward、归一化分数和最终总分。

下面先看一个可运行的简化 Python 例子。它不是完整 benchmark，而是一个最小骨架，能展示 SR、F1、归一化和加权汇总。

```python
from math import isclose

def success_rate(success_count: int, total_count: int) -> float:
    assert total_count > 0
    return success_count / total_count

def f1_score(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def dcg_score(win_rate: float, damage_rate: float) -> float:
    assert 0.0 <= win_rate <= 1.0
    assert 0.0 <= damage_rate <= 1.0
    return 0.7 * win_rate + 0.3 * damage_rate

def normalize_score(score: float, baseline_mean: float) -> float:
    assert baseline_mean > 0
    return score / baseline_mean

def weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    assert set(scores) == set(weights)
    assert isclose(sum(weights.values()), 1.0, rel_tol=1e-9)
    return sum(scores[k] * weights[k] for k in scores)

# 玩具例子
os_sr = success_rate(7, 10)               # 0.7
kg_f1 = f1_score(tp=8, fp=2, fn=4)        # precision=0.8, recall=2/3
dcg = dcg_score(win_rate=0.5, damage_rate=0.9)

assert isclose(os_sr, 0.7)
assert round(kg_f1, 3) == 0.727
assert isclose(dcg, 0.62)

normalized = {
    "os": normalize_score(os_sr, 0.5),
    "kg": normalize_score(kg_f1, 0.52),
    "dcg": normalize_score(dcg, 0.50),
}

weights = {
    "os": 0.4,
    "kg": 0.3,
    "dcg": 0.3,
}

total = weighted_score(normalized, weights)

assert normalized["os"] == 1.4
assert round(total, 3) > 1.0
print("total_score =", round(total, 3))
```

上面这个例子体现了最重要的结构：

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `success_rate` | 成功数、总数 | SR | 适合 OS/DB/HH |
| `f1_score` | TP/FP/FN | F1 | 适合 KG |
| `dcg_score` | 胜率、伤害率 | 复合分 | 适合 DCG |
| `normalize_score` | 原始分、环境均值 | 归一化分 | 拉齐量纲 |
| `weighted_score` | 多环境分、权重 | 总分 | 得到综合结果 |

如果再往前一步，任务 runner 可以写成下面这种伪代码结构：

```python
def run_tasks(env, model, tasks, max_steps=20):
    results = []
    for task in tasks:
        obs = env.reset(task)
        done = False
        total_reward = 0.0

        for _ in range(max_steps):
            action = model.act(obs)          # 模型根据当前观察选择动作
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break

        results.append({
            "task_id": task["id"],
            "success": info.get("success", False),
            "reward": total_reward,
            "meta": info,
        })
    return results
```

这里的 `obs` 是观察，白话解释就是“环境当前给模型看到的状态”；`action` 是动作，白话解释就是“模型决定下一步做什么”；`done` 表示任务是否结束。

真实工程例子可以更具体。假设你在做一个自动化运维 agent，上线前想评估它能否处理工单式任务，比如：

- 在日志目录中找到异常文件
- 检查某服务端口是否监听
- 读取配置并修复拼写错误
- 在数据库里查出最近 24 小时失败任务

这时可以把 AgentBench 思路落地成内部测试集：

1. 为 OS 环境准备一组 shell 任务，统计 SR。
2. 为 DB 环境准备一组 SQL 任务，统计 SR。
3. 为 KG 或内部知识库任务准备实体检索题，统计 F1。
4. 用历史模型均值做归一化。
5. 给每类环境设置上线阈值，比如 `OS >= 0.8`、`DB >= 0.85`、`KG F1 >= 0.7`。

这比只看“模型回答是否看起来靠谱”要严格得多，因为它直接把失败暴露成轨迹和数字。

---

## 工程权衡与常见坑

AgentBench 的工程难点不在“分数怎么加”，而在“任务怎么稳定跑通”。模型一旦从文本回答进入交互式环境，错误会立刻放大。

最常见的问题如下：

| 常见坑 | 具体表现 | 典型缓解措施 |
|---|---|---|
| 多步规划失效 | 前两步正确，后面偏离目标 | 提示中显式要求先列计划，再执行 |
| 动作误判 | 生成了语义上合理但环境不接受的动作 | 加动作候选约束与语法校验 |
| 上下文遗忘 | 做到一半忘记原始目标或中间结果 | 引入短期记忆窗口或状态摘要 |
| 错误恢复差 | 一步失败后持续重复错误动作 | 加 retry 策略与异常分支提示 |
| 奖励误导 | 为了刷局部 reward 偏离真实目标 | 结合终局成功与过程奖励 |
| 评估不稳定 | 同样任务多次跑分差异很大 | 固定温度、固定种子、重复采样 |
| 环境封装不严 | benchmark 本身泄漏答案或状态不一致 | 把评测环境和提示模板隔离 |

其中，开源模型常见短板是长链路一致性。单步看似合理，但连续 10 步后开始胡乱试探，尤其在 KG、DCG、HH 这类需要持续记住状态的场景里容易崩掉。

举一个 KG 的真实常见坑。任务要求模型从若干实体关系中找出目标节点，很多模型会直接“猜一个像是答案的实体”，而不是按路径一步步验证。结果是 Precision 看起来偶尔不差，但 Recall 很低，F1 也会很差。一个简单但有效的工程改进是，在提示里强制要求模型输出中间结构，例如：

- 先列出识别到的候选实体
- 再列出每个实体的关系路径
- 最后给出最终答案

这种做法本质上是 CoT，也就是 Chain of Thought，白话解释就是“把中间推理步骤显式写出来”。它不保证模型一定更聪明，但通常会减少跳步和漏步。

另一个常见坑在动作执行。比如网页环境中，模型说“点击购买按钮”在语义上没问题，但系统真正需要的是某个具体 DOM 元素 ID。如果动作接口不严格，评测就会混入“自然语言描述能力”，而不是“可执行动作能力”。所以工程上常把动作空间限制为：

- `click(element_id)`
- `type(element_id, text)`
- `scroll(direction)`
- `finish(answer)`

动作越结构化，评测越可控。

还有一个容易被忽略的问题是 reward 设计。如果一个网页浏览任务给“打开新页面”太高奖励，模型可能学会疯狂点链接，而不是真正完成检索目标。所以 reward 最好只作为辅助信号，核心仍然要看是否完成最终任务。

---

## 替代方案与适用边界

AgentBench 很有用，但不是所有场景都该用它。

如果你只关心对话质量，例如客服机器人、陪聊应用、FAQ 助手，那么更适合的是传统对话 benchmark。因为这些业务的关键不在于执行外部动作，而在于回答准确、安全、稳定、有上下文。

如果你关心的是明确的执行能力，比如自动化运维、网页代理、数据库助手、办公代理，那么 AgentBench 这类任务驱动 benchmark 才有意义。原因很简单：执行型系统最终交付的不是一句话，而是一个可验证的结果。

下面做一个对比：

| 方案 | 核心指标 | 典型场景 | 优点 | 局限 |
|---|---|---|---|---|
| AgentBench | SR、F1、reward、加权总分 | 智能体执行、多环境操作 | 接近真实任务，能测动作能力 | 成本高，环境搭建复杂 |
| 对话 benchmark | 流畅度、相关性、安全性 | 客服、问答、聊天机器人 | 便宜、快、易比较 | 很难反映执行能力 |
| 单任务 benchmark | 单项成功率或专属指标 | SQL、代码、检索 | 适合专项优化 | 跨环境泛化信息不足 |

对于资源有限的团队，一个实用做法不是“一次性跑完所有环境”，而是先挑关键场景做预检。

例如，一个要上线内部运维 agent 的团队，可以先只跑：

- OS：测命令执行和文件处理
- DB：测 SQL 查询与数据读写
- KG/KB：测知识检索和关系推理

这三类任务已经足够判断模型是否具备“硬执行能力”的底子。至于网页购物、家居控制、卡牌游戏等环境，可以在后续扩展时再补。

反过来，如果是客服机器人，AgentBench 的 OS、DB、HH 场景就可以暂时跳过。因为这些环境会增加测试成本，却不一定提升业务判断质量。评测体系应该服务业务目标，而不是追求“benchmark 越全越高级”。

所以 AgentBench 的适用边界可以总结为一句话：当你需要验证模型是否能在外部环境里稳定做事时，用它；当你只需要验证文本输出质量时，不必硬套它。

---

## 参考资料

1. AgentBench 论文与 THUDM 项目说明：用于理解 benchmark 的总体结构、任务设计和“LLM as agent”的评测目标。
2. EmergentMind 对 AgentBench 的分析文章：用于理解各环境指标以及归一化、加权汇总的思路。
3. TechTarget 对智能体 benchmark 的综述：用于了解 AgentBench 在行业语境中的定位，以及八类环境为什么接近现实任务。
4. 社区评测脚本与 issue 讨论：用于补充实际落地中的常见问题，例如动作格式、长链路稳定性和复现实验设置。
5. 相关任务型 benchmark 资料：用于和纯对话 benchmark、专项 benchmark 做横向比较，明确 AgentBench 的适用范围。
