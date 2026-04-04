## 核心结论

事件抽取的结构化，本质是把一句话或一篇文档里的“发生了什么”转换成统一的机器结构。这里的“结构”不是泛指整理，而是严格对齐到预先定义好的 `schema`，也就是一套事件类型、角色类型和约束规则。最常见的表示可以写成：

$$
E = (trigger, type, \{(role_i, arg_i)\}_{i=1}^{n})
$$

其中，`trigger` 是触发词，白话说就是最能说明“事件发生”的那个词；`type` 是事件类型；`role` 是论元角色，白话说就是事件里每个参与者扮演的功能位置；`arg` 是角色对应的具体文本片段或实体。

玩具例子可以直接看一句英文：`The court determined that Alice was the victim.`  
如果 `determined` 被识别为触发词，事件类型是 `Judge`，那么 `Alice` 就可能被抽成 `Victim` 角色。结构化输出不是一句自然语言，而更像：

| 字段 | 值 |
|---|---|
| trigger | determined |
| type | Judge |
| Victim | Alice |
| Agent | The court |

真正难的部分不在“抽到几个词”，而在“抽出来以后必须满足结构约束”。例如 `Victim` 通常应该是人，`Time` 应该是时间表达式，某些事件允许一个角色多值，某些事件不允许。工程上，能否保证这种一致性，比单个标签的局部准确率更重要。

主流方法大致分三类。`pipeline` 先识别触发词，再识别论元，再把它们拼成事件；优点是简单，缺点是误差会级联。`joint extraction` 联合抽取，把触发词、角色和关系一起建模，减少前一步错了后面全错的问题。`table/span` 结构化方法把输出写成表格填空或 span 图，天然更接近 schema，因此更容易做约束解码和后处理。

---

## 问题定义与边界

事件抽取的输入通常是自然语言文本，输出是一个或多个事件结构。注意，这里不是“摘要生成”，也不是“问答”。系统必须明确给出：

| 输出要素 | 含义 | 常见约束 |
|---|---|---|
| 触发词 trigger | 触发事件的关键词 | 必须与事件类型兼容 |
| 事件类型 type | 事件属于哪一类 | 来自固定 schema |
| 论元角色 role | 参与者在事件中的功能位置 | 角色集合由类型决定 |
| 论元 arg | 角色对应的实体或片段 | 类型要满足角色约束 |
| 事件关系 | 事件与事件之间的联系 | 可选，但文档级常见 |

边界要先说清楚，否则容易把任务说得过于简单。

第一，跨句论元。白话说，事件信息不一定都在一句里。比如第一句给出裁决原因，第二句给出受害者，系统仍然要把它们装配到同一个 `Judge` 事件里。  
第二，角色多值。一个事件的某个角色可能出现多个论元，比如多个受害者、多个地点。  
第三，嵌套与重叠。一个片段可能既是某事件的结果，又是另一个事件的触发环境。  
第四，高频多事件文档。新闻、金融公告、情报文档里经常一段话连续描述多个事件，简单的句级抽取会漏连或错连。

可以把边界问题整理成表：

| 边界场景 | 典型表现 | 主要挑战 |
|---|---|---|
| 跨句 | 原因在 S1，受害者在 S2 | 长距离依赖 |
| 多值角色 | 多个 Victim/Target | 漏抽或过抽 |
| 嵌套事件 | 一个事件作为另一个事件论元 | 结构冲突 |
| 重叠触发 | 同一 span 服务多个事件 | 边界不稳定 |
| 多事件文档 | 一段里多个 trigger | 事件归属混淆 |

研究里常用约束树控制角色分支扩展。可以把某个分支门控写成：

$$
\delta_b = f(h_{doc}, h_{role})
$$

其中 $h_{doc}$ 是文档摘要向量，白话说就是整篇文档压缩后的全局表示；$h_{role}$ 是当前角色表示。$\delta_b$ 可以理解为“这个角色还有没有必要继续扩展子分支”的动态阈值。它的价值在于：不是给所有角色统一设死规则，而是根据文档上下文决定当前分支应不应该继续长。

---

## 核心机制与推导

如果把 pipeline 看成“先拆零件，再拼装”，那么联合模型更像“边看图纸边装配”。它关心的不只是局部分类对不对，还关心整个事件结构是否合法。

先看 EACE 一类方法。它的核心是 `Argument Constraint Tree`，也就是论元约束树。白话说，它不是平铺地预测一堆角色，而是按层级展开角色分支。这样做的好处是，当同一角色可能有多个值时，模型可以在“继续扩展”与“停止扩展”之间做动态决策，而不是硬编码“每个角色只取一个”或“取前 k 个”。

对新手最直观的玩具例子如下：

- S1: `The court determined the attack was intentional.`
- S2: `Alice and Bob were identified as victims.`

如果事件类型是 `Judge`，触发词是 `determined`，那么 S1 更像提供 `Reason`，S2 提供两个 `Victim`。这时如果模型只看句内，很可能只能抽出 `Reason`，漏掉 `Victim`；如果强行给 `Victim` 只保留一个，又会丢失 `Bob`。约束树方法会把文档级信息汇聚起来，再决定 `Victim` 分支是否继续展开。

这个思想可以写成：

$$
\delta_b = f(h_{doc}, h_{role}), \quad
expand_b = \mathbb{I}(\delta_b > \tau)
$$

其中 $\tau$ 是阈值，$\mathbb{I}$ 是指示函数。若 `expand_b=1`，说明当前角色分支继续向下搜索更多论元；否则停止。这样同一角色多值时，不会机械地固定个数。

再看 RCEAE 一类方法。它比单纯角色树更进一步，显式建模角色之间以及事件之间的关系。这里的 `cross encoder` 可以理解为“让触发词、候选论元、角色提示一起进入编码器”，从而直接学习“某个候选片段是不是这个角色”。`role graph` 是角色图，白话说就是把角色节点连成图，再用图上的信息传播来保证结构一致。

它可以写成一个条件概率：

$$
p(role_i \mid context, graph) = g(h_{context}, h_{graph}, h_{trigger}, h_{arg_i})
$$

这个式子表达的是：某个论元属于某个角色，不只由局部上下文决定，还受整张角色图约束。比如已经有 `Court` 被判成 `Agent`，那么另一个人名更可能是 `Victim` 而不是再来一个法院。图结构的价值就在于这种“全局协调”。

两类方法可以对照看：

| 方法 | 输入组织 | 如何处理角色 | 如何保证输出一致 |
|---|---|---|
| Pipeline | 先 trigger 再 argument | 分阶段独立预测 | 依赖规则拼装 |
| EACE | 文档级表示 + 约束树 | 动态扩展角色分支 | 用树结构约束多值 |
| RCEAE | cross encoder + role graph | 角色相关性交互建模 | 用图解码保持 schema |
| Table/Span | 表格或片段图 | 按 schema 单元格填充 | 结构天然规范 |

从推导角度看，结构化事件抽取比传统序列标注多了一个核心目标：不是只最大化局部标签概率，而是同时最大化“结构合法性”。如果勉强写成一个目标，可以理解为：

$$
\mathcal{L} = \mathcal{L}_{trigger} + \mathcal{L}_{role} + \lambda \mathcal{L}_{consistency}
$$

其中 $\mathcal{L}_{consistency}$ 是一致性损失，白话说就是专门惩罚“不符合 schema 的组合”。

---

## 代码实现

工程里不一定要先上复杂论文结构。先实现一个“结构感正确”的最小版本，更容易验证 schema 是否可用。最简单的办法是表格填充：行表示事件类型，列表示角色，单元格里填候选文本 span。

下面是一个可运行的 Python 玩具实现。它不训练模型，只演示“如何把候选论元按 schema 组织起来，并校验结构约束”。

```python
from dataclasses import dataclass, field

SCHEMA = {
    "Judge": {
        "roles": ["Agent", "Victim", "Reason"],
        "multi_value_roles": {"Victim"},
        "required_roles": {"Agent"}
    }
}

@dataclass
class Event:
    trigger: str
    event_type: str
    arguments: dict = field(default_factory=dict)

def add_argument(event: Event, role: str, arg: str):
    spec = SCHEMA[event.event_type]
    assert role in spec["roles"], f"illegal role: {role}"

    if role in spec["multi_value_roles"]:
        event.arguments.setdefault(role, [])
        if arg not in event.arguments[role]:
            event.arguments[role].append(arg)
    else:
        if role in event.arguments and event.arguments[role] != arg:
            raise ValueError(f"role {role} only allows one value")
        event.arguments[role] = arg

def validate_event(event: Event):
    spec = SCHEMA[event.event_type]
    for role in spec["required_roles"]:
        assert role in event.arguments, f"missing required role: {role}"
    return True

event = Event(trigger="determined", event_type="Judge")
add_argument(event, "Agent", "The court")
add_argument(event, "Reason", "the attack was intentional")
add_argument(event, "Victim", "Alice")
add_argument(event, "Victim", "Bob")

assert validate_event(event)
assert event.arguments["Agent"] == "The court"
assert set(event.arguments["Victim"]) == {"Alice", "Bob"}
```

这个例子说明两件事。第一，事件输出最好一开始就按 schema 存储，而不是最后再“补救式整理”。第二，多值角色和单值角色要明确区分，否则后处理会非常混乱。

如果进一步写成 table-filling 伪代码，可以是：

```python
schema_table = {
    "Judge": ["trigger", "Agent", "Victim", "Reason"]
}

for event_type, columns in schema_table.items():
    trigger_candidates = detect_triggers(document, event_type)
    for trig in trigger_candidates:
        row = {"trigger": trig}
        for role in columns[1:]:
            candidates = find_candidate_spans(document, role, trig)
            row[role] = decode_with_constraints(role, candidates, row)
        emit(row)
```

这里 `decode_with_constraints` 是关键。它不能只看当前角色，还要看前面已经填了什么。例如已经填了一个组织名作为 `Agent`，则另一个组织名再去竞争 `Victim` 时，分数应该受抑制。

真实工程例子可以看金融情报系统。假设一篇公告同时包含“高管减持”“监管问询”“业绩下调”三个事件。如果用大模型逐事件自由生成，资源消耗高，而且格式难控。更现实的做法往往是：

1. 用编码器得到全文表示。
2. 用递归事件查询解码器逐步生成事件 query。
3. 对每个 query 预测 trigger 和 role-argument 配对。
4. 用双线性打分或图约束做一致性筛选。

对应的 schema 表可以长成这样：

| 事件类型 | trigger | Agent | Victim/Target | Time | Reason |
|---|---|---|---|---|---|
| Judge | determined | court | Alice/Bob | yesterday | intentional attack |
| Regulation | asked | regulator | company | Monday | disclosure issue |
| Finance | reduced | executive | shares | Q2 | risk control |

---

## 工程权衡与常见坑

工程上最常见的误区，是把事件抽取当成“更复杂的 NER”。这不准确。NER 主要识别实体边界和类型；事件抽取还要解决“谁和谁通过什么触发词构成一个合法结构”。

几个高频坑可以直接列出来：

| 常见坑 | 典型后果 | 常见对策 |
|---|---|---|
| 跨句论元漏抽 | 事件不完整 | 文档级编码、长距离摘要 |
| 角色多值过抽 | 一个角色塞进过多噪声 | 动态分支阈值 |
| 单值角色冲突 | 一个角色出现多个互斥值 | 约束解码、后处理 |
| 嵌套事件断裂 | 下游图谱无法连通 | 统一图建模 |
| 多事件混连 | 论元归到错误触发词 | 事件 query 或 role graph |

跨句问题尤其容易在真实数据里出错。原因不是模型“没见过这个角色”，而是注意力被更近的局部证据抢走了。EACE 这类方法通过文档摘要和约束树减少“只盯当前句”的偏差。RCEAE 这类方法则通过角色图把多个候选之间的兼容关系显式建模出来。

一致性损失可以用一个很简单的伪代码表示：

```python
loss = trigger_loss + role_loss

for event in predicted_events:
    if violates_schema(event):
        loss += alpha
    if has_duplicate_single_value_role(event):
        loss += beta
    if incompatible_type_role_pair(event):
        loss += gamma
```

这个思路朴素，但很重要。很多线上系统的稳定性提升，未必来自更大的模型，而来自更严格的结构约束。

再看真实工程权衡。高频多事件文档里，如果每个事件都走一次大模型提示生成，延迟和成本通常都不可接受。递归 query decoder 的优势在于：把“我要找下一个事件”的状态编码成 query，不断更新，而不是每轮重新理解全文。它的伪代码像这样：

```python
query = init_query(document)
events = []

for _ in range(max_steps):
    event = decode_one_event(document, query)
    if event is None:
        break
    events.append(event)
    query = update_query(query, event, document)

assert len(events) >= 0
```

这类方法通常比微调或调用通用 LLM 更省资源，也更容易把输出锁到固定 schema。代价是系统设计更复杂，训练数据标注也要更细。

---

## 替代方案与适用边界

没有一种方法适合所有场景。选型时最重要的不是“最新”，而是“当前数据、算力、延迟、标注质量能支撑什么”。

先给一个整体对比：

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| Pipeline | 简单、可拆分、易调试 | 误差级联明显 | 数据少、先做 MVP |
| Joint | 结构一致性更好 | 训练复杂 | 中大型标注集 |
| Table-Filling | 与 schema 对齐天然清晰 | 表过大时计算重 | schema 固定、角色稳定 |
| Span/Graph | 适合重叠与嵌套 | 推理实现复杂 | 复杂文档级抽取 |
| LLM 生成式 | 上手快、迁移强 | 格式不稳、成本高 | 原型验证、低频任务 |

如果数据量很少，先上 pipeline 往往更现实。比如先做“触发词识别 + 角色分类 + 规则合成”，哪怕存在级联误差，也能快速验证业务 schema 是否合理。等数据和需求稳定后，再升级到 joint 或 table/span 方案。

可以把这种权衡抽象成：

$$
Consistency = \alpha \cdot SchemaCoverage - \beta \cdot ErrorPropagation
$$

这里 `SchemaCoverage` 表示输出对 schema 的覆盖和遵守程度，`ErrorPropagation` 表示前序步骤错误向后扩散的程度。pipeline 的问题通常是 $\beta$ 偏大；joint 和结构化方法的目标就是压低这个项。

适用边界也要明确。  
如果文档短、事件少、角色简单，pipeline 完全够用。  
如果跨句和多值角色频繁出现，EACE 一类带约束树的方法更合适。  
如果事件之间关系复杂、角色之间互相制约明显，RCEAE 这类图交互方法更稳。  
如果系统强依赖固定格式落库，table-filling 或 span-schema 方法通常更好，因为输出天然接近数据库表结构。

---

## 参考资料

| 资料 | 涉及机制 | 解决的问题 |
|---|---|---|
| Event Extraction Overview | 事件抽取定义、schema 目标、结构化表示 | 明确任务边界与输出形式 |
| Pipeline vs Joint / Table-Span 综述 | pipeline、joint、table-filling、span graph | 比较主流建模路线 |
| EACE 相关研究 | 文档级约束树、动态分支阈值 $\delta_b$ | 跨句论元、多值角色 |
| RCEAE 相关研究 | cross encoder、multi-view role graph、interactive decoder | 角色相关性与一致性解码 |
| Recurrent Event Query Decoder | 递归事件 query、双线性 role-argument 映射 | 多事件高频文档的工程落地 |

1. Event Extraction Overview：用于理解结构化事件抽取的统一定义、触发词与论元的 schema 化目标。  
2. Pipeline vs Joint、table/span 结构化综述：用于比较 cascade pipeline、联合抽取与表格/片段图方法的优缺点。  
3. EACE：重点看文档级 Argument Constraint Tree 和动态阈值 $\delta_b$，适合理解跨句与多值角色问题。  
4. RCEAE：重点看角色相关性建模、role graph 和 role-specific interactive decoder，适合理解一致性约束。  
5. Recurrent Event Query Decoder：重点看多事件文档中的递归 query 更新机制，适合理解工程系统为何不一定直接依赖大模型自由生成。
