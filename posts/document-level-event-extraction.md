## 核心结论

文档级事件抽取的目标，不是判断一句话里“有没有事件”，而是把整篇文档中分散出现的触发词、角色和实体提及组装成一条完整事件记录。这里的“触发词”是最直接表明事件发生的词，“论元”是事件里的参与者、时间、地点等槽位，“共指”是多个提及其实指向同一个对象，比如“伤者”“该男子”“他”其实是同一人。

一句话写成公式就是：

$$
\text{事件记录} = \text{触发词} + \text{角色槽位} + \text{实体共指链}
$$

如果没有“实体共指链”，模型只能看到零散事实，难以确认它们是不是同一个事件；如果没有“角色槽位”，模型只能聚合文本片段，无法输出结构化结果。

玩具例子最能说明问题。看三句话：

1. “昨晚市中心发生袭击。”
2. “两名伤者随后被送往人民医院。”
3. “警方确认受害者均为游客。”

句级模型通常只能抽到：
- 句 1 有一个“袭击”触发词
- 句 2 有“伤者”和“人民医院”
- 句 3 有“受害者”和“游客”

但文档级模型要输出的是一条统一记录：

| 事件类型 | Trigger | Victim | Destination | Time | Location |
|---|---|---|---|---|---|
| Attack | 袭击 | 两名游客 | 人民医院 | 昨晚 | 市中心 |

这就是“跨句事件组装”的核心价值。它把“袭击发生”“有人受伤”“被送到医院”看成一件事的不同证据，而不是三段互不相干的局部事实。

当前主流结果说明这个问题仍然难。以 RAMS 这类文档级基准为例，公开系统的 F1 只有约 52.3%，这表示“能做”不等于“已经做好”。主要瓶颈不是句内识别，而是跨句对齐误差：模型要先知道谁和谁是同一个实体，再判断哪些角色属于同一事件，任何一层出错都会传到最后结果。

因此，工程上真正有效的方向通常有三类：

| 方向 | 解决的问题 | 为什么重要 |
|---|---|---|
| 事件-实体共指联合模型 | 同一实体、同一事件被拆散 | 减少跨句错配 |
| 文档图神经网络或图 Transformer | 远距离句间依赖弱 | 强化跨句信息传递 |
| 事件槽位填充 | 从“识别片段”走向“输出记录” | 更接近真实业务需求 |

结论可以压缩成一句：文档级事件抽取的关键不在“多读几句”，而在“把多句信息按同一事件模板正确归并”。

---

## 问题定义与边界

给定一篇文档 $D$，文档级事件抽取要求输出一个事件集合 $E=\{e_1,e_2,\dots,e_n\}$。每个事件 $e$ 至少包含三部分：

$$
e = (t, R, C)
$$

其中：
- $t$ 是触发词或触发片段
- $R=\{(r_1,a_1),(r_2,a_2),\dots\}$ 是角色到论元的映射
- $C$ 是论元背后的实体共指链，也就是多个提及如何归到同一实体

更展开一点，可以写成：

$$
\forall e \in E,\quad e = \left(t_e,\{a_{e,r}\}_{r \in \mathcal{R}_e}, \{\text{coref}(m)\}\right)
$$

这里 $\mathcal{R}_e$ 表示该事件类型允许的角色集合，比如攻击事件常见角色有攻击者、受害者、地点、时间、武器。

这个定义和句级事件抽取的边界不同。句级任务通常是：在一条句子里找到触发词和角色。文档级任务则要求跨句整合，输出整篇文档层面的结构化表。它不关心“每句各抽一点”，而关心“最终是不是拼成了正确事件”。

一个金融公告例子更适合理解边界。假设文档里有这些句子：

- 句 1：“公司公告控股股东办理股份质押。”
- 句 5：“质押人为张某。”
- 句 7：“本次质押开始日期为 2024-01-10。”
- 句 10：“质押股份数量为 500 万股。”

目标不是抽四条独立片段，而是得到一条 `Equity Pledge` 事件：

| 字段 | 值 | 所在句子 |
|---|---|---|
| Trigger | 股份质押 | 1 |
| Pledger | 张某 | 5 |
| Begin Date | 2024-01-10 | 7 |
| Pledged Shares | 500万股 | 10 |

这说明文档级任务有几个明确边界。

第一，只考虑能通过触发词、角色和共指链连通起来的事实。若一句话提到“公司经营正常”，但与质押事件无角色关系，它不应被并入事件表。

第二，要处理“重叠事件”。一篇文档可能同时写“质押”“解除质押”“增持”，它们共享同一个实体，但不是同一个事件。

第三，要处理“角色共享”。同一个人可以同时是多个事件的参与者。例如在安全新闻里，“警方”既可能是调查者，也可能是通报者；在金融文档里，“公司”既可能是公告主体，也可能是担保方。

第四，本文讨论的是“文档级结构化抽取”，不是开放域摘要，不是问答，也不是只做共指消解。共指只是中间步骤，最终目标仍是事件表。

实际工程里，什么时候必须上文档级？一个粗略判断标准是：当关键字段分散到多句，且跨句实体数超过 3 个，或者同一文档内共存多个同类型事件时，句级方法基本就不够用了。

---

## 核心机制与推导

文档级事件组装之所以难，本质上是因为模型要同时完成三件事：

1. 找到哪些 mention 是实体提及
2. 判断哪些 mention 彼此共指
3. 判断哪些 mention 应该被填到同一事件的同一张表里

这里的 mention 可以直译成“提及”，就是文本里一次具体出现的实体或事件短语，比如“伤者”“该男子”“袭击”。

### 1. 联合建模：把实体、事件、共指放到同一个网络里

如果把实体识别、事件识别、共指消解分开做，会出现典型误差传递。

例如：
- 第一步没认出“该男子”是人名实体
- 第二步自然无法把它填进 Victim 槽位
- 第三步也无法把“该男子”和“伤者”合并

所以联合建模的思想是：让这些任务共享文档级表示。可以把每个 mention 的表示写成 $h_i$，再同时优化多个目标：

$$
\mathcal{L} = \lambda_1 \mathcal{L}_{entity} + \lambda_2 \mathcal{L}_{event} + \lambda_3 \mathcal{L}_{coref}
$$

其中：
- $\mathcal{L}_{entity}$ 约束 mention 的实体类型
- $\mathcal{L}_{event}$ 约束触发词和角色预测
- $\mathcal{L}_{coref}$ 约束 mention 对之间是否共指

为什么这样有效？因为共指信息会反向帮助事件抽取。若模型已经学到“伤者”和“游客”是同一人，那么它更容易把两个句子的证据装配到同一条 `Attack` 事件里。反过来，如果两个 mention 都频繁出现在同一事件模板的 Victim 槽位，共指判断也会更稳定。

### 2. 文档图：让模型显式走跨句路径

只靠 Transformer 的自注意力，理论上可以看整篇文档，但在长文本里，远距离信息很容易被稀释。所以很多方法会构图。图里的节点是 mention，边表示它们之间的关系，例如：

- 共现边：出现在同一句或相邻句
- 共指边：指向同一实体
- 共类型边：都属于人名、地点、组织等
- 句法或语义边：有明确语义联系

把图记作 $G=(V,A)$，其中 $V$ 是 mention 集合，$A$ 是邻接矩阵。图传播的一步可写成：

$$
H^{(l+1)} = \sigma \left( A H^{(l)} W^{(l)} \right)
$$

其中：
- $H^{(l)}$ 是第 $l$ 层节点表示
- $W^{(l)}$ 是可学习参数
- $\sigma$ 是非线性函数

如果用图 Transformer，则 attention 不再对所有 token 平等展开，而是被图结构引导：

$$
\alpha_{ij} \propto \exp \left( \frac{(Qh_i)^\top (Kh_j)}{\sqrt{d}} + b_{ij} \right)
$$

其中 $b_{ij}$ 是结构偏置，表示节点 $i,j$ 是否存在共指、共现或同类型关系。这个偏置的作用很直接：有结构关系的 mention，更容易互相“看见”。

新手可以把它想成一张跳转地图。假设：
- 句 5 出现 “质押人张某”
- 句 7 出现 “此次质押”
- 句 10 出现 “其所持股份”

如果没有图，模型要靠纯文本上下文自己猜“其”是不是“张某”。有了图以后，可以走一条更稳定的路径：

`张某 mention -> 共指边 -> 其 -> 角色边 -> 质押事件 trigger`

于是跨句链条更容易闭合。

### 3. 事件模板：从“找片段”变成“填表”

事件抽取真正落地时，输出通常不是一串标签，而是一张模板表。以攻击事件为例：

| 角色槽位 | 说明 |
|---|---|
| Trigger | 事件触发词 |
| Attacker | 攻击者 |
| Victim | 受害者 |
| Place | 地点 |
| Time | 时间 |
| Instrument | 工具或武器 |

模型做的不是“分类一句话”，而是对每个槽位填值。这个过程可以理解为条件生成或条件搜索：

$$
P(e \mid D) = P(t \mid D) \prod_{r \in \mathcal{R}} P(a_r \mid t, D, C)
$$

含义是：先根据整篇文档找到事件触发词，再在给定触发词和共指结构的条件下，为每个角色槽位选择合适论元。

玩具例子如下：

- 句 1：“昨晚市中心发生袭击。”
- 句 2：“两名伤者被紧急送医。”
- 句 3：“警方称这两名游客暂无生命危险。”

推导过程是：
- `Trigger = 袭击`
- `Time = 昨晚`
- `Place = 市中心`
- 句 2 的“伤者”和句 3 的“游客”共指
- 所以 `Victim = 两名游客`
- `送医`不是单独攻击事件，而是受害者后续状态，不应新建 Attack 事件

这一步就是“事件归并”。它决定模型到底输出一条记录，还是错误地拆成两条。

真实工程例子是金融公告或法务通报。它们往往比新闻更长，字段更离散，同一实体还会在多处以简称、代词、职位名出现。此时只做触发词分类几乎没有业务价值，必须依赖联合建模和图结构把整篇文档的信息连起来。

---

## 代码实现

工程上可以把“文档级事件抽取”拆成三个层次：

1. mention 检测与编码
2. mention 图构建与聚合
3. 按事件模板做槽位填充

Doc2EDAG 的价值在于第三步。它把事件表转换成实体为中心的 DAG。DAG 是“有向无环图”，意思是路径只能向前扩展，不会回到原点。这样一来，事件抽取不再是“在全文里一次性找全所有字段”，而是“沿着模板路径逐步填空”。

一个简化数据结构如下：

| 组件 | 含义 |
|---|---|
| nodes | 所有 mention 节点 |
| edges | mention 之间的角色或共指关系 |
| state memory | 当前路径已经填了哪些槽位 |
| frontier | 下一步允许扩展的候选槽位 |
| record | 当前生成的事件表行 |

下面给一个可运行的玩具实现。它不依赖深度学习，只演示“跨句归并 + 槽位填充”的核心逻辑。

```python
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Mention:
    mid: str
    text: str
    sent_id: int
    mtype: str
    coref_id: Optional[str] = None

@dataclass
class EventRecord:
    trigger: str
    victim: Optional[str] = None
    place: Optional[str] = None
    time: Optional[str] = None
    destination: Optional[str] = None

def canonical_entity(mention: Mention, coref_map: Dict[str, str]) -> str:
    # 把共指链归一到统一实体名
    if mention.coref_id and mention.coref_id in coref_map:
        return coref_map[mention.coref_id]
    return mention.text

def assemble_attack_event(trigger: Mention, mentions: List[Mention], coref_map: Dict[str, str]) -> EventRecord:
    record = EventRecord(trigger=trigger.text)

    for m in mentions:
        text = canonical_entity(m, coref_map)
        if m.mtype == "victim" and record.victim is None:
            record.victim = text
        elif m.mtype == "place" and record.place is None:
            record.place = text
        elif m.mtype == "time" and record.time is None:
            record.time = text
        elif m.mtype == "hospital" and record.destination is None:
            record.destination = text

    return record

coref_map = {
    "c1": "两名游客"
}

trigger = Mention("m0", "袭击", 1, "trigger")
mentions = [
    Mention("m1", "昨晚", 1, "time"),
    Mention("m2", "市中心", 1, "place"),
    Mention("m3", "伤者", 2, "victim", coref_id="c1"),
    Mention("m4", "人民医院", 2, "hospital"),
    Mention("m5", "游客", 3, "victim", coref_id="c1"),
]

event = assemble_attack_event(trigger, mentions, coref_map)

assert event.trigger == "袭击"
assert event.time == "昨晚"
assert event.place == "市中心"
assert event.victim == "两名游客"
assert event.destination == "人民医院"
```

这个例子体现了三件事：
- 触发词和论元不必在同一句
- 多个 mention 可以通过 `coref_id` 合并成同一实体
- 事件输出是结构化记录，不是散乱标签

如果把它写成更接近 Doc2EDAG 的伪代码，流程是这样的：

```python
def expand_path(state, role, candidates):
    best = select_best_candidate(state, role, candidates)
    if best is None:
        return state
    state.memory[role] = best
    state.path.append((role, best.mid))
    return state

def update_memory(state, mention):
    entity_id = resolve_coref(mention)
    state.entity_memory.add(entity_id)
    return state

def decode_event(trigger, schema, doc_graph):
    state = init_state(trigger)
    for role in schema.roles:
        candidates = retrieve_candidates(doc_graph, trigger, role, state)
        state = expand_path(state, role, candidates)
        if role in state.memory:
            state = update_memory(state, state.memory[role])
    return to_record(state)
```

这里的输入输出很清楚：
- 输入：触发词、事件 schema、文档图
- 中间状态：已经填了哪些槽位、已经看到哪些实体
- 输出：一条事件记录

在真实系统里，`retrieve_candidates` 不会是规则，而是依赖编码器、图聚合模块和打分函数。例如候选论元分数可写成：

$$
s(a_r) = f(h_t, h_{a_r}, g_{doc}, m)
$$

其中：
- $h_t$ 是触发词表示
- $h_{a_r}$ 是候选论元表示
- $g_{doc}$ 是文档图聚合后的全局表示
- $m$ 是当前 memory，记录已经填过哪些槽位

为什么要有 memory？因为事件表的各槽位并不独立。比如你已经确定当前是“股份质押”事件，且 Pledger 是“张某”，那么后续找 Begin Date、Pledged Shares 时，就应优先寻找与“张某”相关的 mention，而不是全文任意日期或数字。memory 提供了这层条件约束。

真实工程例子可以看金融公告。假设一篇文档同时包含两次质押：
- 第一次由张某发起，开始日期 1 月 10 日
- 第二次由李某发起，开始日期 2 月 18 日

如果没有路径状态和记忆，模型很容易把“张某”与“2 月 18 日”错误配对。Doc2EDAG 这类路径扩展方法的核心贡献，就是把“成表”过程显式化，减少跨事件串槽。

---

## 工程权衡与常见坑

文档级事件抽取在论文里看起来像“再加一个图层”，但在工程落地里，真正的代价和坑比模型结构更关键。

先看主要权衡：

| 方案 | 速度 | 实现复杂度 | 跨句能力 | 典型 F1 表现 | 主要问题 |
|---|---|---|---|---|---|
| 句级模型 | 快 | 低 | 弱 | 通常下限稳定 | 漏掉跨句论元 |
| 联合模型 | 中 | 中高 | 中强 | 明显优于流水线 | 训练和标注依赖更强 |
| 图增强文档级模型 | 慢 | 高 | 强 | 上限更高 | 构图成本大、调参复杂 |
| Doc2EDAG 类填表模型 | 中慢 | 高 | 强 | 适合结构化事件 | schema 设计要求高 |

### 常见坑 1：不做联合训练，角色重复提取

最常见的错误是：同一角色被重复抽成不同事件。比如：

- 句 2：“张三受伤”
- 句 10：“该男子仍在治疗”

如果不做实体共指与事件联合训练，系统可能输出两条记录：
- Attack-1 的 Victim = 张三
- Attack-2 的 Victim = 该男子

这不是召回低，而是归并失败。业务上会直接导致事件重复计数。

联合损失可以用一个简化形式表示：

$$
\mathcal{L}_{joint} = \mathcal{L}_{trigger} + \mathcal{L}_{argument} + \alpha \mathcal{L}_{entity} + \beta \mathcal{L}_{coref}
$$

伪代码如下：

```python
def joint_loss(trigger_logits, arg_logits, entity_logits, coref_scores, labels):
    loss_trigger = ce(trigger_logits, labels["trigger"])
    loss_arg = ce(arg_logits, labels["argument"])
    loss_entity = ce(entity_logits, labels["entity"])
    loss_coref = bce(coref_scores, labels["coref"])
    return loss_trigger + loss_arg + 0.5 * loss_entity + 0.5 * loss_coref
```

关键不是公式本身，而是“同步更新”。你不能先把实体模块训好冻结，再单独训事件模块，否则共指边界和事件边界会慢慢漂移。

### 常见坑 2：没有图结构，远距离线索连不上

纯句级或纯 token attention 模型在短文本里能工作，但文档一长就会退化。尤其当：
- 触发词在前文
- 关键角色在后文
- 中间隔着多句背景说明

这时没有图，很难稳定连通。

新手常见误区是“模型都能看 4k token 了，为什么还要图”。问题不在能不能看见，而在能不能有效利用。图结构把“应该重点对齐的 mention”显式标出来，等于给模型提供了更低噪声的搜索空间。

### 常见坑 3：只做抽取，不做槽位约束

即使已经连上跨句信息，如果没有 schema 约束，系统也可能把不合法的字段填进去。比如在 `Attack` 事件里把“人民医院”填成 `Victim`，因为它和“送医”有强共现。解决方法是加入细粒度槽位约束：
- 类型约束：Victim 必须是 Person 或 Group
- 数值约束：Pledged Shares 应是数量
- 时间约束：Begin Date 必须是日期表达式
- 事件一致性约束：同一条记录内字段应服务于同一 trigger

### 常见坑 4：构图成本被低估

图模型常被宣传成“精度更高”，但工程上有真实成本：
- mention 抽取质量要足够高
- 共指边要构得准
- 图过密会拖慢训练和推理
- 长文档 batching 更麻烦

所以不是所有场景都值得上最重的图结构。如果文档普遍只有 3 到 5 句，且一个文档通常只有一个事件，复杂图模型很可能收益不大。

一个实用判断是：
- 文档短、事件少：先用句级 + merge
- 文档长、字段散：再上联合 + 图
- 需要成表输出：优先考虑模板填充或 Doc2EDAG 思路

---

## 替代方案与适用边界

不是所有业务都需要完整的文档级神经模型。替代方案通常按成本和能力分层。

### 1. 句级模型 + 后处理 merge

最轻的路线是先逐句抽触发词和论元，再用规则或聚类做归并。流程如下：
- 句级模型抽取局部事件片段
- 用实体共指或字符串匹配做聚类
- 对同类型事件按时间、地点、主体进行合并

这种方法适合：
- 文档短
- 事件数少
- 时效要求高
- 标注数据有限

新手例子：短新闻里先抽到“袭击”“伤者”“医院”，再用规则把“伤者/游客/该男子”并到同一实体链，快速得到一条简化事件表。这种方案在法律监测、舆情告警等需要先上线的业务里很常见。

伪代码可写成：

```python
def merge_sentence_events(events):
    clusters = cluster_by_type_and_entity(events)
    merged = []
    for cluster in clusters:
        merged.append(fill_missing_slots(cluster))
    return merged
```

缺点也很明确：一旦出现多个同类型事件共存，简单 merge 很容易串槽。

### 2. 规则系统 + 共指匹配

如果业务文本格式固定，比如某类金融公告、某类审计披露，可以直接上规则：
- 正则或词典抽取 trigger
- 基于模板找角色字段
- 用别名表和代词规则做共指归并

优点是快、可控、可解释；缺点是泛化差，换领域就要重写。

### 3. 文档级神经模型

这是最强但也最贵的方案。它适合：
- 长文档
- 多事件共存
- 字段跨句分散
- 结果要直接进入知识图谱、风控、检索系统

可以把三类方案并排看：

| 方案 | 适用场景 | 优点 | 缺点 |
|---|---|---|---|
| 规则 | 文本模板稳定、资源有限 | 上线快、可解释 | 泛化弱、维护重 |
| 句级 + merge | 短文档、事件少 | 实现简单、推理快 | 多事件串槽严重 |
| 文档级联合/图模型 | 长文档、多事件、跨句强 | 精度上限高 | 成本高、训练复杂 |

适用边界可以给出一个实用经验：

| 文档特征 | 推荐方案 |
|---|---|
| 关键字段大多在同一句 | 句级即可 |
| 跨句实体不超过 2 个，事件通常只有 1 个 | 句级 + merge |
| 跨句实体超过 3 个，且同类事件可能共存 | 必须考虑文档级 |
| 输出要求是结构化事件表或知识图谱入库 | 优先模板填充/Doc2EDAG |

真实工程里，常见路线不是一步到位，而是渐进升级：
1. 先上规则或句级模型，验证字段定义和业务价值
2. 再加入共指聚类，减少重复事件
3. 最后引入联合模型和图结构，解决跨句归并难题

这是因为文档级模型的真正难点，不只是训练，而是要先把事件 schema 定清楚。没有稳定 schema，再强的模型也只能产出不稳定表格。

---

## 参考资料

| 资料 | 年份 | 一句话 takeaway | 链接 |
|---|---|---|---|
| Zheng et al., Doc2EDAG | 2019 | 把文档级事件表转成实体中心 DAG，用路径扩展和记忆机制做槽位填充 | https://people.iiis.tsinghua.edu.cn/~weixu/Krvdro9c/EMNLP2019-zheng.pdf |
| Kriman & Ji, Joint Detection and Coreference Resolution of Entities and Events with Document-level Context Aggregation | 2021 | 实体、事件和共指联合训练，减少流水线误差传递 | https://aclanthology.org/2021.acl-srw.18/ |
| Zhang et al., A Semantic Mention Graph Augmented Model for Document-Level Event Argument Extraction | 2024 | 构建语义 mention 图，用图结构增强跨句事件论元抽取 | https://aclanthology.org/2024.lrec-main.139/ |
| RAMS benchmark | 持续更新 | 文档级事件抽取仍然困难，公开结果约 52.32% F1 说明跨句归并远未解决 | https://www.wizwand.com/dataset/rams |
| Papers With Code: Doc2EDAG 页面 | 持续更新 | 可快速查看任务定义、论文入口和相关实现 | https://paperswithcode.com/paper/document-level-event-argument-extraction-by |

如果按阅读顺序建议：
- 新手先看 Doc2EDAG，理解“事件表填充”这个问题定义。
- 然后看 Kriman & Ji，理解为什么共指与事件要联合建模。
- 最后看 Zhang et al.，理解为什么文档图能提升跨句聚合能力。
- RAMS 分数可以作为现实校准：文档级事件抽取不是“已解决问题”，而是“可用但误差仍大”的工程问题。
