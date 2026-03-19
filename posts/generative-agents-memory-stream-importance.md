## 核心结论

Generative Agents 的“记忆流”可以理解为一条按时间追加的事件日志。系统不会把全部历史都塞进下一次推理，而是先给每条记忆计算三个分数：

- `recency`：新近性，白话说就是“离现在越近，越容易被想起来”。
- `importance`：重要性，白话说就是“这件事本身值不值得长期记住”。
- `relevance`：相关性，白话说就是“这条记忆和当前问题像不像一回事”。

检索时不看单一维度，而看综合分数：

$$
score = \alpha \cdot recency + \beta \cdot importance + \gamma \cdot relevance
$$

其中 $\alpha,\beta,\gamma$ 是权重，用来控制系统更偏向“最近发生”“本身重要”还是“和当前问题相关”。

这套机制的核心价值不是“让模型记住更多”，而是“让模型在有限上下文里只看最该看的记忆”。如果没有这个过滤层，长对话、多轮任务、多角色模拟很快就会被无关历史挤满。

玩具例子：用户上午说“我想了解重疾险”，中午说“我午饭吃了面”，下午又问“保额怎么选”。这时系统更应该取回“想了解重疾险”，而不是“午饭吃了面”。后者虽然更新，但对当前任务几乎无帮助。

真实工程例子：在多智能体模拟里，角色 A 刚和角色 B 一起吃饭，随后又收到“晚上聚会邀请”。此时检索到“刚刚和 B 见过面”这类高相关记忆，能帮助模型生成更连贯的社交反应，而不是把“早上整理桌面”这类琐事也送进上下文。

| 维度 | 含义 | 常见评分方式 | 是否随时间变化 | 典型重置/更新频率 |
| --- | --- | --- | --- | --- |
| `recency` | 记忆有多新 | 指数衰减，如 $0.99^{hours}$ | 会 | 每次检索时重算 |
| `importance` | 记忆本身有多重要 | LLM 打 1 到 10 分，再归一化 | 通常不变 | 记忆写入时评一次，必要时重评 |
| `relevance` | 和记当前查询的语义接近程度 | 向量余弦相似度 | 会 | 每次有新查询时重算 |

---

## 问题定义与边界

问题不是“怎么存下所有历史”，而是“怎么从大量历史里稳定挑出最值得进入当前推理上下文的少量片段”。

如果一个智能体每观察到一句话就永久保留，而且每次推理都把所有历史一并发送给大模型，会立刻遇到三个问题：

1. token 成本快速上升。
2. 无关信息稀释当前任务信号。
3. 模型更容易被高频但低价值的日常事件干扰。

因此，记忆系统必须先定义“什么叫值得取回”。Generative Agents 的做法是把这个问题拆成三个可度量维度：时间、价值、语义匹配。

其边界也很明确。

第一，`recency` 不是“越近越好”的绝对规则。某些任务对长期信息更敏感。例如用户一个月前说“我对花生过敏”，哪怕时间很久，依然不应被普通衰减直接淹没。

第二，`importance` 依赖评分过程。如果评分 prompt 太宽松，模型可能把很多普通事件都打成高分，导致重要性维度失效。

第三，`relevance` 依赖 embedding。embedding 是把文本压成向量的表示方法，白话说就是“把一句话变成一个可计算相似度的坐标点”。如果向量模型切换、索引没更新、文本预处理不一致，相关性排序会明显偏移。

可以把公式边界写得更具体一些：

$$
score = \alpha \cdot recency + \beta \cdot importance + \gamma \cdot relevance
$$

通常可令：

- $recency \in (0, 1]$
- $importance \in [0.1, 1.0]$，若原始分数是 1 到 10，可除以 10
- $relevance \in [-1, 1]$，但工程上常经过截断或归一化到 $[0,1]$

参数调节方向如下：

| 参数 | 调大后的效果 | 适合场景 | 风险 |
| --- | --- | --- | --- |
| $\alpha$ | 更看重最近发生 | 实时聊天、短任务协作 | 容易忘掉关键旧信息 |
| $\beta$ | 更看重事件本身价值 | 长周期规划、人物设定 | importance 评分失真时会放大误差 |
| $\gamma$ | 更看重当前语义匹配 | 问答检索、任务回忆 | embedding 不稳定时结果抖动 |

新手版理解：跟保险中介聊天时，系统不需要反复取回“你昨天买了雪糕”，因为它既不重要，也和“保单怎么选”不相关。它更应该取回“你提过想查重疾险和预算上限”。

---

## 核心机制与推导

记忆流的最小单元是一条 observation，也就是一次观察到的事件。例如“用户提到想买保险”“角色 A 看见 B 在公园散步”。每条 observation 写入记忆后，系统会为它附加可检索属性。

### 1. 新近性：指数衰减

常见做法是：

$$
recency = \lambda^{\Delta h}
$$

其中 $\Delta h$ 是距离当前时刻的小时数，$\lambda$ 是衰减底数，常见取值如 $0.99$ 或 $0.995$。

它的含义很直接：每多过 1 小时，记忆价值按固定比例缩小一次。  
如果 $\lambda=0.99$，5 小时后的新近性是：

$$
0.99^5 \approx 0.951
$$

如果过了 100 小时：

$$
0.99^{100} \approx 0.366
$$

这说明指数衰减不是“到点清零”，而是“逐步降权”。它适合真实对话，因为很多记忆不是突然失效，而是慢慢不再优先。

### 2. 重要性：由模型打分

`importance` 反映事件对未来推理的潜在影响。论文式实现里通常让 LLM 对一句记忆打 1 到 10 分，再归一化到 0 到 1。

例如：

- “今天刷牙了”可能是 1 或 2。
- “决定下周离职”可能是 9 或 10。
- “用户说预算上限是 5000 元”在购物代理里也可能是 8，因为它会持续影响决策。

这里的关键不是绝对客观，而是相对排序。系统并不需要知道“重要性真值”，只需要能把明显琐碎的事件压到后面。

### 3. 相关性：向量相似度

`relevance` 常用余弦相似度：

$$
relevance = \cos(\theta) = \frac{x \cdot y}{\|x\|\|y\|}
$$

其中 $x$ 是当前查询的向量，$y$ 是某条记忆的向量。白话说，如果两句话语义接近，它们在向量空间里的方向也更接近，余弦值就更高。

例如：

- 查询：“怎么选重疾险保额”
- 记忆 A：“用户想了解重疾险”
- 记忆 B：“用户中午吃了面”

则 A 的 relevance 明显高于 B。

### 4. 综合排序

最终系统按：

$$
score = \alpha \cdot recency + \beta \cdot importance + \gamma \cdot relevance
$$

对全部候选记忆排序，取 top-k 进入上下文。

玩具例子：

- 记忆 M1：5 小时前发生，$recency \approx 0.95$，重要性 7 分即 $0.7$，相关性 $0.6$
- 记忆 M2：1 小时前发生，$recency \approx 0.99$，重要性 2 分即 $0.2$，相关性 $0.3$

若 $\alpha=\beta=\gamma=1$：

$$
score(M1)=0.95+0.7+0.6=2.25
$$

$$
score(M2)=0.99+0.2+0.3=1.49
$$

所以虽然 M2 更新，但 M1 更值得进入当前推理。这就是三维评分比“只看最近”更合理的原因。

再看一个真实工程例子。一个客服代理处理用户历史：

- 三天前：用户说“我给父母买医疗险”
- 两小时前：用户说“预算 8000 以内”
- 十分钟前：用户说“页面打不开”

如果当前问题是“推荐方案”，那么预算和投保对象的相关性高，页面打不开虽然更新，但更像当前故障上下文，不一定该进入方案生成的记忆池。系统可以在不同任务 query 下得到不同的 top-k，这比固定历史摘要更灵活。

---

## 代码实现

下面给出一个最小可运行版本。它省略了真实 LLM 调 importance 和真实 embedding 模型，但保留了数据结构、评分逻辑和 top-k 检索流程。

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import sqrt
from typing import List, Tuple


@dataclass
class Memory:
    text: str
    timestamp: datetime
    importance: float   # 0.0 - 1.0
    embedding: List[float]


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(v: List[float]) -> float:
    return sqrt(sum(x * x for x in v))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    na = norm(a)
    nb = norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)


def recency_score(now: datetime, ts: datetime, decay: float = 0.99) -> float:
    hours = max((now - ts).total_seconds() / 3600.0, 0.0)
    return decay ** hours


def combined_score(
    memory: Memory,
    query_embedding: List[float],
    now: datetime,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    decay: float = 0.99,
) -> Tuple[float, float, float, float]:
    rcy = recency_score(now, memory.timestamp, decay=decay)
    rel = cosine_similarity(memory.embedding, query_embedding)
    score = alpha * rcy + beta * memory.importance + gamma * rel
    return score, rcy, memory.importance, rel


def retrieve_top_k(
    memories: List[Memory],
    query_embedding: List[float],
    now: datetime,
    k: int = 3,
) -> List[Tuple[float, Memory]]:
    scored = []
    for m in memories:
        score, _, _, _ = combined_score(m, query_embedding, now)
        scored.append((score, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


now = datetime(2026, 3, 19, 18, 0, 0)

memories = [
    Memory(
        text="用户想了解重疾险和保额",
        timestamp=now - timedelta(hours=5),
        importance=0.7,
        embedding=[0.9, 0.8, 0.1],
    ),
    Memory(
        text="用户中午吃了面",
        timestamp=now - timedelta(hours=1),
        importance=0.2,
        embedding=[0.1, 0.0, 0.9],
    ),
    Memory(
        text="用户预算上限是5000元",
        timestamp=now - timedelta(hours=3),
        importance=0.8,
        embedding=[0.8, 0.7, 0.1],
    ),
]

query_embedding = [1.0, 0.9, 0.0]
top = retrieve_top_k(memories, query_embedding, now, k=2)

assert len(top) == 2
assert "重疾险" in top[0][1].text or "预算" in top[0][1].text
assert all(top[i][0] >= top[i + 1][0] for i in range(len(top) - 1))

for score, memory in top:
    print(round(score, 4), memory.text)
```

这段代码体现了四个关键点。

第一，记忆对象最少包含 `{text, timestamp, importance, embedding}`。  
第二，`recency` 在检索时根据当前时刻动态计算，而不是写死。  
第三，`relevance` 来自 query 和 memory 的余弦相似度。  
第四，最终只取 top-k，而不是把所有高分记忆都拼进 prompt。

如果接到真实工程里，一般流程是：

1. observation 到来，先写入原始文本。
2. 调 LLM 生成 `importance` 分数。
3. 调 embedding 模型生成向量。
4. 写入记忆库和向量索引。
5. 推理前根据当前 query 做一次排序检索。
6. 把 top-k 记忆拼接到系统上下文或工作记忆区。

---

## 工程权衡与常见坑

这套方案能工作，但不代表默认参数就可靠。实际落地时常见问题比公式本身多。

| 常见问题 | 现象 | 根因 | 规避手段 |
| --- | --- | --- | --- |
| importance 失效 | 什么都像重要记忆 | 评分 prompt 过宽，模型总给高分 | 强约束评分标准，加入少量标尺样例 |
| recency 衰减过慢 | 琐碎旧事长期占位 | decay 太接近 1 | 按业务设置 domain-aware 衰减 |
| relevance 索引过期 | 明明相关却检不出来 | embedding 更新后未重建索引 | 向量模型版本化，必要时批量刷新 |
| top-k 过大 | 上下文冗长、重复 | 贪心拉太多记忆 | 先小后大，配合去重和摘要 |
| top-k 过小 | 丢掉关键背景 | 检索窗口太窄 | 对高 importance 记忆加保底机制 |

最常见的坑是把 importance 当成“情绪强烈程度”。这不对。重要性判断的是“这条记忆会不会持续影响后续推理”。“今天很开心”在闲聊里可能重要，在报销机器人里则几乎无关。

另一个坑是 recency 半衰期设置脱离业务。聊天助手和项目管理代理对时间的敏感度完全不同。聊天里 48 小时前的寒暄可以快速贬值；项目管理里“三周前确认的上线日期”仍然应高权重。工程上最好按场景定衰减曲线，而不是全站一套参数。

真实工程例子：一个销售代理每天接触上百条 CRM 事件。如果 `importance` 没有压低“已读邮件”“例行打卡”这类事件，几轮对话后上下文就会被低价值记录占满。结果不是“记忆太少”，而是“噪声太多”。

还要注意 embedding drift。它是“向量语义分布随着模型或预处理变化而漂移”。如果一半记忆用旧 embedding 模型生成，另一半用新模型生成，余弦相似度可能不再可比。此时 relevance 排序会出现隐性退化，问题往往不容易第一时间被发现。

---

## 替代方案与适用边界

并不是所有系统都需要完整的 `recency + importance + relevance` 三维评分。

低资源场景下，可以只用 `recency + importance`。这相当于放弃语义检索，只保留“最近且重要”的事件。它不够聪明，但实现简单，适合端侧、小模型、无向量库环境。

例如安卓端本地代理没有 embedding 服务，可以直接设规则：“最近 2 小时内，且 importance > 0.6 的记忆参与检索。”这样 recall 会下降，但复杂度和成本显著更低。

另一种替代方案是 rule-based 清理。也就是直接设过期和保留规则，如：

- 普通事件保留 24 小时
- 用户偏好保留 30 天
- 明确约束类信息永久保留直到人工删除

这种做法的优点是可解释、可控，缺点是对语义适配能力弱，难以处理“一条旧信息忽然在当前问题里重新变得重要”的情况。

| 策略 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| `recency + importance` | 实现简单，不依赖向量库 | 语义召回弱 | 端侧、低资源、原型阶段 |
| full score | 兼顾时间、价值、语义 | 工程复杂度更高 | 通用智能体、长对话、多任务系统 |
| rule-based 清理 | 可解释性强，行为稳定 | 灵活性差，难处理隐式相关 | 合规要求强、规则明确的业务系统 |

还要看记忆规模。如果系统总记忆量只有几十条，复杂排序的收益可能不大，简单队列或手工标签就够用。相反，如果是长期运行的陪伴型智能体、办公代理、多角色仿真，三维评分几乎是基础设施，不做会很快失控。

所以边界很清楚：

- 记忆少、任务短、预算紧：先上简化版。
- 任务长、角色多、历史深：需要 full score。
- 监管强、规则先行：可加 rule-based 作为外层约束。

---

## 参考资料

- Generative Agents: Interactive Simulacra of Human Behavior 的论文总结：提供记忆流、反思、规划的整体框架，用于支撑“问题定义与边界”的系统背景。
- Memory Architectures 分析文章：给出 `score = α·recency + β·importance + γ·relevance`、指数衰减、top-k 检索等实现细节，用于支撑“核心机制与推导”和“代码实现”。
- Flow Engineering / Generative Agents 相关文章：讨论 importance 评分 prompt、检索噪声、上下文占满等工程问题，用于支撑“工程权衡与常见坑”。
