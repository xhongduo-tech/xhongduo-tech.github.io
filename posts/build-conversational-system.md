## 核心结论

多轮对话系统不是“把聊天记录拼长一点”这么简单。它的工程核心，是在每一轮都同时维护三件事：上下文、目标约束、执行策略。上下文指前面说过什么；目标约束指哪些规则必须持续成立；执行策略指这一轮应该回答、追问、调用工具还是更新结构化状态。三者只要有一个断开，系统就会看起来“能说话”，但不能稳定完成任务。

可以把它理解成接力赛。接力赛的重点不是某一棒跑得快，而是交棒时信息不能丢。用户在第 1 轮给出目标，第 2 轮补充条件，第 3 轮修改之前的参数，如果系统每轮都只看当前输入，就会把会话做成一堆孤立的单轮问答；如果系统只记历史，不校验约束，又会在中途逐渐偏离最初要求。

工程上真正需要落地的是“轮次内外同步”：

| 维度 | 单轮系统 | 多轮系统 |
|---|---|---|
| 主要目标 | 回答当前输入 | 维持整段任务流 |
| 关键状态 | 当前 prompt | 历史、记忆、工具状态、约束状态 |
| 评估重点 | 准确率、相关性 | 一致性、连续性、约束保持率 |
| 常见失败 | 当前轮答错 | 前后矛盾、忘记条件、工具选错 |
| 工程约束 | Prompt 设计 | 状态追踪、策略校验、安全联动 |

一个直接结论是：多轮能力不能只靠更强的基础模型“顺带得到”。它需要明确的状态层、约束层和校验层，否则模型在真实部署中会被后续轮次的干扰信息带偏。

---

## 问题定义与边界

多轮对话，指系统当前输出依赖历史轮次，而不是只依赖当前一句输入。这里的“依赖”不是把历史原文全部塞进上下文，而是要把历史压缩成对当前决策有用的状态。状态可以是用户目标、已确认槽位、已调用工具、剩余待办、仍然生效的格式限制等。

术语“状态追踪”首次出现时，可以把它理解成“给对话维护一份持续更新的任务记录”。没有这份记录，系统只能临时发挥；有了它，系统才知道哪些信息是已经确认的，哪些还需要追问。

边界必须划清，否则系统设计会失控。不是所有聊天都要做成重型多轮系统。例如“北京今天天气怎样”是单轮查询；但“先查北京今天天气，再把周末也补上，最后只用五句话总结并提醒我带伞”就是多轮任务，因为它跨轮累积了目标和约束。

下面这个表格能帮助判断工程边界：

| 维度 | 低复杂度场景 | 高复杂度多轮场景 |
|---|---|---|
| 输入信号 | 纯文本 | 文本、语音、按钮事件、工具返回 |
| 状态跨度 | 1 轮内 | 3 到 20 轮甚至更长 |
| 目标类型 | 单次问答 | 任务推进、参数修改、持续约束 |
| 评估指标 | 当前轮正确率 | 跨轮一致性、任务完成率、约束满足率 |
| 部署约束 | 延迟低即可 | 延迟、稳定性、可恢复性、安全都要兼顾 |

玩具例子很简单。用户说：“帮我订明天下午去浦东机场的车。”下一轮又说：“改成后天早上 8 点，从静安寺出发。”如果系统只处理当前轮，容易只记住“后天早上 8 点”，却忘了目的地还是浦东机场；如果系统错误覆盖状态，可能把出发地和目的地也一起改掉。

真实工程例子更典型。客服机器人在一次会话里可能连续做三件事：先按“最多五句话”回答政策问题，再调用天气工具判断是否需要改签，最后把变更后的日期、时间、联系人写入表单。这已经不是“聊天”，而是一个持续演进的任务流。每轮都要更新全局状态，并重新判断当前最合理的动作。

在语音或机器人场景里，边界还会进一步扩大。这里会出现“轮次切换信号”，也就是系统判断用户是否说完、自己是否该接话。它的白话解释是“别抢话，也别沉默太久”。如果文本系统只管内容正确，不管何时接话，落到语音产品里就会频繁打断用户。

---

## 核心机制与推导

多轮系统为什么需要额外机制？因为历史不是线性的字符串，而更像有关系的状态网络。某些轮次是澄清，某些轮次是纠错，某些轮次是执行结果反馈。它们对当前决策的重要性不同，不能等权处理。

一种可解释的方法，是把整段对话建模成图 $G=(V,E)$。图模型的白话解释是“把每轮对话当成一个点，再用边表示它们之间的关系”。例如，当前轮和上轮有顺序关系，当前轮和首次目标声明有约束继承关系，工具返回轮和发起调用轮有执行对应关系。

关系感知 GCN（图卷积网络）的作用，是让每个节点从相关节点收集信息后再更新表示。它不是简单平均，而是按关系类型分别变换。核心更新可写为：

$$
h_i'=\sigma\left(\sum_{\theta\in\Theta}\sum_{j\in S_i^{(\theta)}}a_{ij}c_{i,\theta}W_\theta'e_j + a_{ii}W_0'e_i\right)
$$

这里，$e_j$ 是第 $j$ 个轮次的初始表示，$W_\theta'$ 是关系类型 $\theta$ 对应的参数，$a_{ij}$ 可以理解成边权重，表示“这个历史轮次对当前轮有多重要”。$\sigma$ 是非线性激活函数，白话解释是“把线性加权后的结果做一次压缩映射”。

再做一次聚合：

$$
h_i=\sigma\left(\sum_{j\in S_i}W''h_j'+W_0''h_i'\right)
$$

这一层的意思是：当前轮不仅接收单类关系信息，还要综合邻居节点的整体上下文。这样得到的 $h_i$，比“当前句子 + 原始历史拼接”更适合用于状态更新、策略选择和风险判断。

但只理解上下文还不够，还要保证约束没丢。这里可以用加权约束满足率 WCSR（Weighted Constraint Satisfaction Rate）：

$$
WCSR=\frac{\sum_j w_j s_j}{\sum_j w_j}
$$

其中，$s_j \in \{0,1\}$ 表示第 $j$ 条约束是否满足，$w_j$ 表示约束重要性。白话说法是：不是所有规则都同等重要。“不能泄露隐私”显然比“尽量简洁”更重要，所以前者权重更高。

用一个玩具例子看就很清楚。假设系统有三条约束：

| 约束 | 是否满足 $s_j$ | 权重 $w_j$ |
|---|---:|---:|
| 回复不超过 5 句 | 1 | 2 |
| 必须包含天气结论 | 1 | 1 |
| 不得编造未查询到的数据 | 0 | 4 |

那么：

$$
WCSR=\frac{2\times1 + 1\times1 + 4\times0}{2+1+4}=\frac{3}{7}
$$

虽然表面上满足了两条约束，但最重的安全约束失败了，所以整体分数很低。这个指标比“满足了几条规则”更符合工程直觉。

真实工程中，这套机制的价值在于处理干扰轮次。用户先说“之后所有回复控制在 5 句内”，中间又插入几轮无关追问，最后再问一个新问题。单轮系统很容易忘记前面的持续约束；图表示和 WCSR 则能把“早先声明但仍生效的规则”保留下来，并在每轮输出前重新检查。

---

## 代码实现

落地时，不一定真的要在线跑一个复杂 GCN，但系统设计应当具备同样的结构思想：轮次索引、状态容器、关系更新、约束校验、动作选择。

先给一个可运行的 Python 玩具实现。它没有训练模型，而是把多轮系统里最关键的工程骨架写清楚。

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Turn:
    role: str
    text: str
    intent: str = ""
    tool: str = ""

@dataclass
class DialogueState:
    history: List[Turn] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=dict)
    constraints: List[Dict[str, Any]] = field(default_factory=list)

def add_turn(state: DialogueState, role: str, text: str, intent: str = "", tool: str = ""):
    state.history.append(Turn(role=role, text=text, intent=intent, tool=tool))

def update_memory(state: DialogueState):
    # 玩具规则：从历史中抽取持续约束和任务参数
    for turn in state.history:
        if "五句话" in turn.text or "<=5句" in turn.text:
            state.memory["max_sentences"] = 5
        if "天气" in turn.text:
            state.memory["topic"] = "weather"
        if "股票" in turn.text:
            state.memory["topic"] = "stock"
        if "明天" in turn.text:
            state.memory["date"] = "tomorrow"

def choose_tool(state: DialogueState) -> str:
    topic = state.memory.get("topic")
    if topic == "weather":
        return "WeatherAPI"
    if topic == "stock":
        return "StockAPI"
    return "None"

def check_constraints(state: DialogueState, response: str) -> float:
    checks = []
    max_sentences = state.memory.get("max_sentences")
    if max_sentences is not None:
        sentence_count = sum(response.count(p) for p in "。！？")
        checks.append({"weight": 2, "ok": sentence_count <= max_sentences})
    # 示例里的高权重约束：不能声称调用过不存在的工具结果
    if "已查询到" in response and choose_tool(state) == "None":
        checks.append({"weight": 4, "ok": False})
    else:
        checks.append({"weight": 4, "ok": True})

    total_weight = sum(x["weight"] for x in checks)
    score = sum(x["weight"] for x in checks if x["ok"]) / total_weight
    return score

state = DialogueState()
add_turn(state, "user", "之后所有回答请控制在五句话内")
add_turn(state, "user", "帮我看下明天天气", intent="ask_weather")
update_memory(state)

tool = choose_tool(state)
response = "明天天气以晴到多云为主。气温适中。建议正常出行。若晚间出门可加薄外套。"
score = check_constraints(state, response)

assert tool == "WeatherAPI"
assert state.memory["max_sentences"] == 5
assert score == 1.0
```

这段代码表达了四个工程原则。

第一，`history` 不是日志备份，而是状态更新的输入。第二，`memory` 必须从历史中提炼持续有效的信息，而不是每轮临时重新猜。第三，工具选择应是显式步骤，不能把“该不该调工具”藏在生成结果里。第四，输出前要走约束检查，而不是事后看线上事故。

如果想把上面的思想写成更接近真实系统的流程，可以用下面这个伪代码：

```python
def on_new_turn(user_text, state):
    state.history.append({"role": "user", "text": user_text})

    node_repr = encode_turn(user_text)
    graph = update_dialogue_graph(state.history, node_repr)   # 维护轮次关系图
    state_repr = gcn_update(graph)                            # 类似关系感知GCN聚合

    intent = classify_intent(state_repr, user_text)
    tool = select_tool(intent, state.memory)
    extracted_slots = extract_entities(user_text, state.memory)

    state.memory = merge_memory(state.memory, extracted_slots, intent, tool)

    draft = generate_response(state_repr, state.memory, tool)
    wcsr = check_constraints(state.memory, draft)

    if wcsr < 0.9:
        draft = repair_response(draft, state.memory)

    return draft, state
```

真实工程例子可以是一个出行助手：

1. 第 1 轮，用户要求“以后都用最多五句话回答”。
2. 第 2 轮，用户问“上海明天会下雨吗”。
3. 第 3 轮，用户追问“如果下雨，顺便把我周五去机场的预约出发时间改成早上 8 点”。

此时系统不能把“天气查询”和“预约修改”混成同一个动作。更稳妥的做法是：
1. 先维护长期约束：最多五句话。
2. 再识别当前是否为多意图输入：天气查询 + 预约变更。
3. 分开调用天气工具与订单系统。
4. 把“周五”“机场”“早上 8 点”写回结构化状态。
5. 合并成符合五句话约束的最终答复。

这就是“多轮 + 多模块”系统的最小闭环。

---

## 工程权衡与常见坑

多轮系统的难点，不在于“记住更多字”，而在于“记住对当前动作真正重要的信息”。工程上最常见的失败模式有三类。

| 失效模式 | 现象 | 根因 | 缓解方式 |
|---|---|---|---|
| 指令漂移 | 早先的持续规则逐渐失效 | 历史压缩时丢掉长期约束 | 独立约束层，每轮重检 |
| 意图混淆 | 继续沿用上轮工具或任务 | 近期上下文权重过高 | 动作选择单独建模，工具前校验 |
| 上下文覆盖 | 新信息错误覆盖旧参数 | 缺少字段级状态管理 | 结构化 memory，支持字段更新与冲突检测 |

第一个坑是指令漂移。用户一开始说“后续都用五句话以内回答”，系统前两轮还记得，后面一长段任务执行后就忘了。这个问题的本质是长期约束与临时上下文混在一起，没有独立存储。解决方法不是“再把 prompt 写长一点”，而是把持续约束显式放进状态层，并让每轮输出都经过同一套校验。

第二个坑是意图混淆。比如上一轮在查天气，这一轮用户改问股票，系统却继续调用天气工具。这是因为模型在局部上下文里形成了惯性。一个有效办法是把“意图识别”“工具选择”“最终生成”拆成三段，每段都能被单独检查。这样即使生成模型有惯性，前面的策略层也能把它拦住。

第三个坑是上下文覆盖。用户说“帮我把明天下午三点的预约改到后天上午十点”，系统如果没有字段级状态，可能把“地点、联系人、服务类型”也一起重置，或者把日期和时间只更新了一半。这里需要结构化记忆。结构化的白话解释是“把状态按字段存，而不是整段文本覆盖整段文本”。

还有一个经常被忽略的现实约束是延迟。多轮系统通常会加上分类器、记忆层、工具层、校验层。每多一层，稳定性更好，但延迟更高。对于纯文本客服，几百毫秒到 2 秒往往可接受；对实时语音助手，如果系统在用户停顿时反应过慢，就会显得迟钝；如果过快，又可能抢话。因此语音场景必须引入轮次切换检测，而不能只复用文本对话链路。

一个实用原则是：把高代价模块放到“必要时才触发”。例如，普通闲聊只做轻量状态更新；只有检测到任务型意图、参数变更或高风险输出时，才启用重校验和工具一致性检查。

---

## 替代方案与适用边界

不是所有场景都值得上完整的多轮状态机。替代方案取决于交互密度、任务复杂度和错误成本。

| 方案 | 适用场景 | 优点 | 缺点 | 何时切换 |
|---|---|---|---|---|
| 纯单轮 + 回放机制 | FAQ、低风险问答 | 简单、成本低 | 易忘历史，跨轮一致性差 | 用户开始频繁引用前文时 |
| 轻量记忆 + 工具路由 | 常规客服、助手类产品 | 成本与效果平衡 | 长任务下仍会漂移 | 出现复杂参数修改时 |
| 强化记忆 + 显式策略模块 | 高约束任务流 | 一致性高、可审计 | 实现复杂、延迟更高 | 错误代价高或需合规时 |

第一种方案是“每轮重新计算，再把上轮结果作为参考输入”。它适合知识问答、轻客服、低风险场景。优点是实现非常快，问题定位也容易。缺点是系统没有真正的会话状态，用户一旦进行纠错、追问、改参，就容易前后不一致。

第二种方案是在单轮基础上加轻量记忆和工具路由。比如保存最近确认的城市、时间、产品名，并由单独模块决定是否调用天气、搜索、订单 API。这是很多实用系统的平衡点。

第三种方案是强化记忆 + 显式策略模块。这里的“显式策略模块”，白话上就是“先决定做什么，再决定怎么说”。适合预约修改、工单流转、金融助手、医疗分诊这类高约束任务。它的优势是可以把生成与执行分离，方便审计；代价是系统更重，需要更多监控和回放数据。

玩具例子里，如果用户只是连续问“北京天气”“上海天气”“广州天气”，单轮加最近城市缓存就够了。但如果用户说“以后都用三句话回答，并且如果下雨就提醒我带伞，最后帮我把周五的行程时间同步到日历”，那就应该切到更完整的多轮系统，因为这里已经同时涉及持续约束、条件逻辑和外部执行。

因此，多轮系统不是默认答案，而是当任务具备以下特征时才值得采用：历史依赖强、参数会被多次改写、存在外部工具、副作用明显、错误成本高。

---

## 参考资料

1. Multi-Turn Evaluation Framework in Dialogue AI：总结多轮对话评估的核心流程、指标与约束建模方法，适合理解为什么多轮系统不能只看单轮准确率。来源链接：<https://www.emergentmind.com/topics/multi-turn-evaluation-framework>
2. Quantifying Conversational Reliability of Large Language Models under Multi-Turn Interaction：给出多轮交互下指令漂移、工具误选、实体覆盖等失效案例，并展示单轮到多轮性能下降的量化结果。来源链接：<https://openreview.net/pdf?id=uFlvUriRLj>
3. Turn-Taking Modelling in Conversational Systems：讨论语音和机器人场景中的轮次切换、重叠发言与打断问题，适合补足“多轮不仅是文本记忆”的工程视角。来源链接：<https://www.mdpi.com/2227-7080/13/12/591>
