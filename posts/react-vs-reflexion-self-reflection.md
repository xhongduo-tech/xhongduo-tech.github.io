## 核心结论

ReAct 和 Reflexion 都属于 Agent 推理框架。Agent 可以理解为“会自己分步决策的模型程序”。两者的核心差别不在“会不会思考”，而在“失败后会不会形成可复用经验”。

ReAct 的主循环是“思考 Thought → 行动 Action → 观察 Observation”。它把推理和行动交织在一起，适合边查边做，但每次失败后通常只是重来一遍，缺少系统性的复盘。Reflexion 在这个循环外再加一个“反思 Reflection”模块，把失败原因写成自然语言经验，存进短期记忆，再喂给下一轮决策。这个过程本质上是在做“语言化的策略修正”，论文把它描述成 verbal reinforcement，也就是“不是改模型参数，而是改下一轮提示中的经验”。

直观地说，ReAct 每轮只问“下一步做什么”；Reflexion 每轮还问“刚才错在哪，以后别再犯什么错”。

| 维度 | ReAct | Reflexion |
| --- | --- | --- |
| 基本循环 | Thought → Action → Observation | Thought → Action → Observation → Reflection |
| 是否有跨 episode 学习 | 弱 | 强 |
| `episode` 含义 | 一次完整尝试，从开始到成功或失败结束 | 同左，但会把失败经验带到下一次 |
| 反馈形式 | 当前轨迹里的即时观察 | 观察 + 评分 + 自然语言反思 |
| 失败后修正方式 | 再试一次 | 先总结错误，再试一次 |
| 适合任务 | 短流程、低成本工具调用 | 长轨迹、需要复盘的任务 |

从原始论文看，Reflexion 在 ALFWorld 上把 ReAct 的成功率从约 75% 提升到 97%，论文正文概括为绝对提升 22 个百分点；在 HotPotQA 上正文概括为提升 20 个百分点，附录里的 100 题 GPT-4 设置中，ReAct 为 39%，Reflexion 为 51%。这些数字说明一个事实：Reflexion 的价值主要不在“单步更聪明”，而在“多轮以后不再重复同一种错误”。

---

## 问题定义与边界

问题很具体：当 LLM Agent 需要执行长轨迹任务时，单次失败往往不会自动转化成下一次的改进。长轨迹任务就是“要走很多步才知道对错”的任务，比如多文档问答、网页操作、文本世界探索、复杂代码调试。

ReAct 能处理“边想边做”，但它的历史主要局限在当前轨迹里。一旦这一轮结束，上一轮失败为什么失败，很容易只剩下模糊印象。于是模型会重复检索同类错误页面、重复执行无效动作、或者继续基于错误前提推理。

Reflexion 试图解决的不是“让模型记住所有历史”，而是“把最有价值的失败经验压缩成少量可读规则”。它的边界也很明确：

1. 它依赖可靠的 Evaluator。Evaluator 就是“打分器”，负责告诉系统这轮到底成功还是失败。
2. 它更适合 episodic 任务。episodic 任务指“一轮有明确起点和终点”的任务。
3. 它的记忆通常是滑动窗口，不会无限增长。
4. 它不是参数训练，不能替代真正的强化学习或微调。

一个玩具例子很容易说明差别。

假设任务是“在三个盒子里找钥匙，再去开门”。

- ReAct 第一次失败：按顺序查盒子 A、B，没找到，就重复查 B。
- Reflexion 第一次失败后会写下反思：“我重复检查了 B，但没有覆盖 C；下次应该优先检查未探索位置。”

下一轮开始时，这句反思被拼进 prompt，策略就变成“先查 C，再决定是否回溯”。

真实工程例子是 HotPotQA 这类多跳问答。多跳指“答案需要组合多篇资料”。ReAct 可能在第一轮只搜到人物 A 的页面，遗漏人物 B 的页面，最后给出错误答案。Reflexion 会把失败写成类似“缺少支持证据：需要先检索人物 B 的生平，再比对两人的所属团体数量”，下一轮检索路径就会改变。这不是让模型“更会背知识”，而是让它“更会纠正检索计划”。

任务流可以压缩成下面三步：

1. 输入问题与工具能力。
2. ReAct 或 Reflexion 生成动作轨迹。
3. 若失败，ReAct 直接重试；Reflexion 先生成复盘，再进入下一轮。

---

## 核心机制与推导

Reflexion 的标准结构有三个部件：

- Actor：行动者，负责生成 Thought、Action 和最终答案。
- Evaluator：评估器，负责给奖励或成功标记。
- Self-Reflection：反思器，负责把失败翻译成自然语言经验。

如果把状态写成 $s_t$，动作写成 $a_t$，奖励写成 $r_t$，反思文本写成 $f_t$，记忆写成 $mem_t$，那么可以把 Actor 的决策写成：

$$
a_t \sim \pi(\cdot \mid s_t, history_t, mem_t)
$$

这里的 $\pi$ 就是策略，白话说就是“在当前状态和上下文下，下一步怎么做的规则”。

Reflexion 的关键不在单步公式，而在记忆更新：

$$
mem_{t+1}=truncate(mem_t \cup \{f_t\}, K)
$$

`truncate` 的意思是截断，只保留最多 $K$ 条反思。论文里常用 $K \approx 1 \sim 3$。这不是偶然设计，而是工程约束：反思太多会拖长 prompt，导致注意力分散、成本上升、甚至把早期低质量经验也保留太久。

触发反思通常不是每步都做，而是靠启发式函数 $h_t$ 决定。启发式就是“人工制定的触发规则”。

```text
if h_t(r_t, history_t):
    f_t = reflect(history_t, r_t)
    mem_t = update(mem_t, f_t, K)
```

常见触发条件如下：

| 触发条件 `h_t` | 白话解释 | 对应反思方向 |
| --- | --- | --- |
| 低 reward / 最终失败 | 这一轮没做对 | 总结缺失证据或错误计划 |
| 重复动作 | 在原地打转 | 强制覆盖未探索分支 |
| hallucination | 模型以为自己拿到了并不存在的信息 | 校正状态认知 |
| 轨迹过长 | 步数过多还没接近目标 | 压缩计划，减少无效搜索 |
| 工具返回冲突观察 | 前后观察对不上 | 回退到更早的可信状态 |

这里最重要的机制推导是：Reflexion 并没有把错误映射成数值梯度，而是把错误映射成语言约束。这个约束进入下一轮上下文后，实际上改变了策略分布。虽然没有显式更新参数，但它确实改变了下一轮动作采样的条件。

这也是为什么它常被称为“语言梯度”。这个说法不是数学上的梯度，而是“用自然语言提供方向性修正”。

---

## 代码实现

下面给一个可运行的简化版实现，模拟“重复搜索导致失败，然后把反思压入滑动窗口，下一轮优先搜索未探索位置”的流程。

```python
from dataclasses import dataclass, field

@dataclass
class AgentState:
    memory: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    k: int = 2

def action_generator(state: AgentState, boxes=("A", "B", "C")) -> str:
    joined_memory = " ".join(state.memory)
    tried = set(state.history)

    # 最新经验优先：如果反思提示覆盖未探索位置，就优先找没开过的盒子
    if "未探索" in joined_memory:
        for box in boxes:
            if f"open_{box}" not in tried:
                return f"open_{box}"

    # 基线策略：错误地偏好重复开 B
    if "open_B" not in tried:
        return "open_B"
    return "open_B"

def evaluator(action: str, key_box="C") -> int:
    return 1 if action == f"open_{key_box}" else 0

def should_reflect(reward: int, history: list[str]) -> bool:
    if reward == 1:
        return False
    if len(history) >= 2 and history[-1] == history[-2]:
        return True
    return False

def self_reflection(history: list[str]) -> str:
    return "重复动作且遗漏未探索位置；下轮优先检查未探索盒子。"

def update_memory(memory: list[str], reflection: str, k: int) -> list[str]:
    memory.append(reflection)
    return memory[-k:]

def run_episode(state: AgentState) -> int:
    action = action_generator(state)
    state.history.append(action)
    reward = evaluator(action)
    if should_reflect(reward, state.history):
        reflection = self_reflection(state.history)
        state.memory = update_memory(state.memory, reflection, state.k)
    return reward

state = AgentState()

# 前两轮会重复开 B，触发反思
assert run_episode(state) == 0
assert run_episode(state) == 0
assert len(state.memory) == 1
assert "未探索" in state.memory[-1]

# 第三轮读取反思后，优先开未探索盒子 A；第四轮再开 C 成功
assert run_episode(state) == 0
assert run_episode(state) == 1

# 测试滑动窗口
state.memory = update_memory(state.memory, "经验2", state.k)
state.memory = update_memory(state.memory, "经验3", state.k)
assert state.memory == ["经验2", "经验3"]
```

这个例子故意做得很小，但结构已经完整：

- `ActionGenerator(history, mem)` 负责根据轨迹和记忆生成动作。
- `Evaluator(action)` 判断这一轮是否成功。
- `should_reflect` 决定何时触发复盘。
- `self_reflection` 生成自然语言经验。
- `update_memory` 用滑动窗口保留最近的 $K$ 条反思。

真实工程里可以把它扩展成下面的调用链：

1. 用户问题进入 Actor。
2. Actor 用 ReAct 风格调用搜索、数据库、浏览器或内部 API。
3. Evaluator 用单元测试、EM/F1、业务规则、人工反馈或另一个 LLM 打分。
4. 若失败且命中启发式，Reflection 生成“下轮应该避免什么、优先找什么”的短句。
5. 记忆窗口截断后拼入下一轮 system prompt 或 scratchpad。

在 HotPotQA 一类任务中，真实反思文本往往长这样：“第一轮只检索到人物 A，缺少人物 B 的支持文档；下次先检索 B，再比较加入乐队数量。”这类句子比“reward=0”更能改变下一轮行为。

---

## 工程权衡与常见坑

Reflexion 的上限，首先取决于反馈质量。反馈质量差，不是“学得慢”，而是“学错了还越来越坚定”。

| 常见坑 | 具体表现 | 规避方式 |
| --- | --- | --- |
| Evaluator 不准 | 错误答案被判对，或正确答案被判错 | 多重评估器交叉校验；关键任务做人审抽检 |
| 过度反思 | 小错误也生成大量经验，prompt 很快膨胀 | 限制触发次数；只在失败、重复、幻觉时反思 |
| 记忆窗口过短 | 早期关键教训被挤掉 | 给反思打优先级；保留“高价值经验”摘要 |
| 记忆窗口过长 | 新旧经验冲突，模型抓不住重点 | 明确“最新经验优先”；控制在 1 到 3 条 |
| 反思太空泛 | 只写“要更仔细”这类废话 | 要求 reflection 输出“错误原因 + 下轮动作建议” |
| 错误归因 | 实际问题是检索源不全，却反思成推理错误 | 把检索失败、推理失败、工具失败分开诊断 |

一个典型工程坑出现在多文档 QA。假设 Evaluator 错把答案判成“缺少证据 B”，而真实问题是“检索接口根本没返回文档 C”。此时 Reflexion 会把错误经验写进记忆，后续每轮都执着于补证据 B，结果形成局部最优。局部最优就是“在一个错误方向上越走越稳”。

另一个常见问题是反思文本质量不稳定。强模型通常能写出可执行经验，例如“不要重复打开同一抽屉，优先搜索未检查容器”。弱模型则容易写成“需要更认真一些”。后者几乎不提供策略增量，所以 Reflexion 的收益和底模能力强相关。

---

## 替代方案与适用边界

如果任务很短、每轮成本敏感、或者根本拿不到可靠反馈，直接上 Reflexion 并不总是最优。

| 方案 | 反馈利用方式 | 是否跨 episode | 成本 | 适用边界 |
| --- | --- | --- | --- | --- |
| ReAct | 利用当前轨迹观察 | 弱 | 低 | 短任务、工具链稳定、允许多试几次 |
| Reflexion | 把失败转成语言经验 | 强 | 中 | 长轨迹、可获得清晰成败信号 |
| Replay Buffer | 存历史轨迹供检索 | 中 | 中 | 想复用经验，但不想生成反思文本 |
| 强化学习微调 | 更新模型参数 | 强 | 高 | 数据和算力充足，任务分布稳定 |

Replay Buffer 可以理解为“经验回放池”，就是把旧轨迹存起来，下次按相似任务检索。它的优点是简单，不要求模型自己总结；缺点是检索回来的是原始历史，不一定能压缩成明确规则。Reflexion 的优势正在于“压缩”，也就是把一长串失败轨迹变成一句能指导下一轮的经验。

强化学习微调则是另一条路线。它真正改模型参数，长期上限更高，但开发门槛也高得多，需要稳定奖励、训练基础设施和更多样本。对很多中小团队来说，Reflexion 是更便宜的第一步。

一个实用判断标准是：

- 如果任务单轮就能完成，优先用 ReAct。
- 如果任务经常“同一种错重复出现”，优先用 Reflexion。
- 如果只有大量历史日志、没有明确反思文本，先上 Replay Buffer。
- 如果任务长期稳定且投入足够，再考虑 RL 或 SFT。

---

## 参考资料

1. Shinn, Noah, et al. *Reflexion: Language Agents with Verbal Reinforcement Learning*. arXiv:2303.11366. 原始论文，定义了 Actor / Evaluator / Self-Reflection / Memory 的完整框架，并给出 $mem_{t+1}=truncate(mem_t \cup \{f_t\}, K)$ 的滑动窗口思想。链接：https://ar5iv.labs.arxiv.org/html/2303.11366
2. Yao, Shunyu, et al. *ReAct: Synergizing Reasoning and Acting in Language Models*. ReAct 原始工作，奠定“Thought-Action-Observation”范式，是 Reflexion 的直接基础。链接：https://arxiv.org/abs/2210.03629
3. Deep Paper 对 Reflexion 的解读。适合从工程视角理解为什么“语言化失败经验”会比单纯重试更有效，并总结了 ALFWorld 约 75% 到 97% 的提升曲线。链接：https://deep-paper.org/en/paper/2303.11366/
4. IJCAI 2025 Survey Track 相关调研。汇总了 ReAct、Reflexion-R1/R2/R3、ExpeL 等反馈机制在 HotPotQA、ALFWorld、WebShop 等任务上的对比，适合看横向定位。链接：https://www.ijcai.org/proceedings/2025/1175.pdf
