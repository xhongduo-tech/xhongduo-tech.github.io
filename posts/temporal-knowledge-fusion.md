## 核心结论

时序知识融合，是把“事实成立的时间”一起纳入融合主键，而不是只比较两个值是不是一样。这里的“融合”可以先白话理解为：把多个来源里关于同一对象的信息，整理成一套可查询、可追溯、可裁决的统一结果。

静态知识融合通常关心“某实体的某属性最终是什么”。时序知识融合关心的是“某实体的某属性在什么时间是什么，以及系统在什么时间知道这件事”。这两个时间必须分开建模：

$$
f=(e,r,a,V,T,s,p)
$$

其中，$e$ 是实体，$r$ 是关系或属性，$a$ 是属性值，$V$ 是 valid time，也就是现实世界中事实成立的时间区间，$T$ 是 transaction time，也就是系统保存这条事实的时间区间，$s$ 是来源，$p$ 是来源优先级。

一个直接结论是：时序知识融合默认允许多版本共存。因为现实世界本来就在变化，历史版本不是脏数据，而是系统必须保留的证据链。公司 CEO 会换人，药物审批状态会变，供应商关系会终止又恢复。如果只保留“当前值”，系统就无法回答“某一天当时到底是什么状态”。

玩具例子最容易说明这个点。公司 X 的 CEO 记录如下：

- Alice：`[2024-01-01, 2024-07-01)`
- Bob：`[2024-07-01, +∞)`

两条记录的值不同，但它们只是边界相接，不是冲突。因为 Alice 在 7 月 1 日之前有效，Bob 从 7 月 1 日开始有效，真实世界中的状态是在演化，不是在打架。

真实工程里更典型的是药物审批状态。某药可能先是 `approved`，后是 `suspended`，再是 `approved_with_restrictions`。如果你把后来的状态覆盖前面的状态，历史就消失了；如果你把不同状态一律判成冲突，系统就把真实变化错当成错误数据。

---

## 问题定义与边界

时序知识融合解决的问题，不是普通去重，也不是简单投票，而是“多来源、随时间变化的事实如何统一表达和裁决”。

先把边界说清楚：

| 问题类型 | 是否属于时序知识融合 | 原因 |
|---|---|---|
| 静态事实去重 | 否 | 只关心是不是同一条事实 |
| 同一事实跨时间多版本 | 是 | 关键在于事实随时间变化 |
| 仅按最新值覆盖 | 否 | 会丢失历史与审计能力 |
| 回填历史修正 | 是 | 新数据可能描述过去发生的事 |
| 只按置信度投票 | 否 | 不处理时间区间就会误判 |

这里的“回填”可以白话理解为：数据今天才到，但说的是上个月就已经发生的事实。比如某公司在 6 月 15 日完成 CEO 交接，但监管文件在 6 月 20 日才公开，系统在 6 月 21 日才抓到。事实的发生时间不是系统的入库时间。

判断一条新事实是否需要进入冲突处理，顺序不能错：

1. 先看实体是否相同。
2. 再看关系是否相同。
3. 再看值是否不同。
4. 最后看有效时间是否重叠。

只有前 4 步都成立，才进入来源裁决。这个顺序很重要，因为很多“看起来不同”的记录，实际上只是不同时间段的正常状态。

例如药物状态：

- `DrugA = approved`，有效时间 `[2024-01-01, 2024-04-01)`
- `DrugA = suspended`，有效时间 `[2024-04-01, 2024-05-15)`
- `DrugA = approved`，有效时间 `[2024-05-15, +∞)`

这三条如果只看值，会觉得前后矛盾；如果放到时间轴上，就会发现它们描述的是完整演化过程。时序知识融合的边界就是：它处理的是“时间上的真假”，不是只处理“字符串上的相等”。

---

## 核心机制与推导

时序知识融合的第一条规则是：冲突不是“值不同”就成立，而是“值不同且有效时间重叠”才成立。

冲突候选条件可以写成：

$$
e_i=e_j,\quad r_i=r_j,\quad a_i\neq a_j
$$

但这还不够。真实时间冲突还要求两个有效时间区间相交：

$$
\text{overlap}(V_i,V_j)=\max(v_i^s,v_j^s)<\min(v_i^e,v_j^e)
$$

这里使用半开区间 $[start,end)$。白话解释是：包含开始时刻，不包含结束时刻。这样做的好处是边界一致，`[2024-01-01, 2024-07-01)` 和 `[2024-07-01, +∞)` 不会被误判成重叠。

继续看 CEO 的玩具例子：

- `f1`: Alice，`V=[2024-01-01, 2024-07-01)`
- `f2`: Bob，`V=[2024-07-01, +∞)`

因为：

$$
\max(2024\text{-}01\text{-}01, 2024\text{-}07\text{-}01)=2024\text{-}07\text{-}01
$$

$$
\min(2024\text{-}07\text{-}01, +\infty)=2024\text{-}07\text{-}01
$$

所以并不满足严格小于关系，不重叠，因此不是冲突。

如果再来一条：

- `f3`: Carol，`V=[2024-06-15, 2024-07-10)`

那么 `f3` 同时与 `f1`、`f2` 重叠。这时不能直接“选最新”或“删旧的”，而是要做时间切片。所谓“切片”，可以白话理解为：把所有边界点拆开，把整段时间切成若干互不重叠的小段，然后在每一小段里单独做裁决。

例如边界点是：

- `2024-01-01`
- `2024-06-15`
- `2024-07-01`
- `2024-07-10`

切片后得到：

| 时间片 | 候选值 |
|---|---|
| `[2024-01-01, 2024-06-15)` | Alice |
| `[2024-06-15, 2024-07-01)` | Alice, Carol |
| `[2024-07-01, 2024-07-10)` | Bob, Carol |
| `[2024-07-10, +∞)` | Bob |

接下来在每个时间片里排序。常见排序因子如下：

| 规则 | 作用 |
|---|---|
| 按 `V` 切分 | 保证每次裁决只面对同一时间片 |
| 按 `p` 排序 | 来源优先级高的先保留 |
| 按 `confidence` 排序 | 证据更强的优先 |
| 按 `lag` 排序 | 处理迟到数据 |
| 版本不覆盖 | 保留可追溯历史 |

其中 `lag` 可以定义为：

$$
lag = t_{ingest} - v_s
$$

意思是：系统比事实发生晚知道了多久。`lag` 不是绝对真理标准，但在多个来源都声称自己正确时，它是一个很有用的辅助信号。一个经常晚到两周的新闻聚合源，通常不应该压过一个当天披露的监管公告源。

真实工程例子可以看药物审批。来源可能有 FDA 公告、企业新闻稿、医学数据库和媒体报道。媒体最先发，不代表它最权威；监管公告最晚入库，不代表事实最晚发生。时序知识融合的核心机制，就是把“成立时间”和“知道时间”拆开，再在重叠区间里基于来源规则裁决。

---

## 代码实现

代码层面最重要的一点是：不要把事实存成单行当前值，而要存成带双时间的版本记录。

一个最小数据结构如下：

| 字段 | 含义 |
|---|---|
| `entity` | 实体 |
| `relation` | 关系或属性 |
| `value` | 属性值 |
| `valid_start/end` | 事实在现实中成立的区间 |
| `txn_start/end` | 系统记录该事实的区间 |
| `source` | 数据来源 |
| `priority` | 来源优先级 |
| `confidence` | 证据强度 |
| `ingest_time` | 入库时间 |

下面给出一个可运行的 Python 玩具实现。它展示三件事：半开区间重叠判断、时间切片、以及按优先级和置信度裁决。

```python
from dataclasses import dataclass
from datetime import date
from typing import Optional, List, Tuple

INF = date(9999, 12, 31)

@dataclass(frozen=True)
class Fact:
    entity: str
    relation: str
    value: str
    valid_start: date
    valid_end: Optional[date]
    source: str
    priority: int
    confidence: float

    def end(self) -> date:
        return self.valid_end or INF

def overlap(a: Fact, b: Fact) -> bool:
    return max(a.valid_start, b.valid_start) < min(a.end(), b.end())

def slice_points(facts: List[Fact]) -> List[date]:
    points = set()
    for f in facts:
        points.add(f.valid_start)
        points.add(f.end())
    return sorted(points)

def winner(candidates: List[Fact]) -> Fact:
    # priority 越小越高；confidence 越大越高
    return sorted(candidates, key=lambda f: (f.priority, -f.confidence, f.source))[0]

def fuse(facts: List[Fact]) -> List[Tuple[date, date, str]]:
    points = slice_points(facts)
    result = []
    for i in range(len(points) - 1):
        start, end = points[i], points[i + 1]
        active = []
        for f in facts:
            if f.valid_start <= start and end <= f.end():
                active.append(f)
        if active:
            chosen = winner(active)
            result.append((start, end, chosen.value))
    return result

alice = Fact("CompanyX", "CEO", "Alice", date(2024, 1, 1), date(2024, 7, 1), "filing", 1, 0.98)
bob = Fact("CompanyX", "CEO", "Bob", date(2024, 7, 1), None, "filing", 1, 0.99)
carol = Fact("CompanyX", "CEO", "Carol", date(2024, 6, 15), date(2024, 7, 10), "news", 3, 0.70)

assert overlap(alice, bob) is False
assert overlap(alice, carol) is True
assert overlap(bob, carol) is True

timeline = fuse([alice, bob, carol])

assert timeline[0] == (date(2024, 1, 1), date(2024, 6, 15), "Alice")
assert timeline[1] == (date(2024, 6, 15), date(2024, 7, 1), "Alice")
assert timeline[2] == (date(2024, 7, 1), date(2024, 7, 10), "Bob")
assert timeline[3] == (date(2024, 7, 10), INF, "Bob")
```

这个例子里，`Carol` 的记录与 Alice、Bob 都重叠，但由于 `news` 来源优先级低，所以在重叠时间片中没有胜出。注意，这不等于把 Carol 彻底删除。工程上更合理的做法是保留原始版本，并把裁决结果单独存成“融合视图”或“canonical view”。

实现流程一般分三步：

1. 标准化输入。统一时间格式、来源标识、时区和区间边界。
2. 区间切分与冲突检测。先找同实体同关系的候选，再按有效时间判断是否重叠。
3. 版本裁决与落库。保留原始版本链，额外生成融合后的时间片结果。

真实工程例子里，药物状态融合通常还会加一层状态机约束。例如某药不太可能从 `withdrawn` 直接跳回 `phase_2_trial`。这种约束不是基础时序融合必须有的，但在行业场景里很常见，因为它能进一步过滤低质量来源。

---

## 工程权衡与常见坑

时序知识融合难的不是“怎么存几列时间”，而是“怎么在查询、裁决、审计之间保持一致”。

最常见的坑如下：

| 坑 | 后果 | 规避方式 |
|---|---|---|
| 只存当前值 | 历史丢失，无法回答某天状态 | 保留版本链 |
| 混淆 `V` 和 `T` | 审计失真，迟到数据被误读 | 入库时分列 |
| 只看值不同 | 把状态演化误判成冲突 | 先判时间重叠 |
| 边界处理不一致 | 产生伪重叠或时间空洞 | 统一半开区间 |
| 不保留来源 | 无法追责和复盘 | 保存 `source/priority/confidence` |
| 直接覆盖旧记录 | 回填历史时破坏证据链 | 关闭版本，不物理覆盖 |

“错误示例 vs 正确示例”可以更直观看出差异：

| 场景 | 错误做法 | 正确做法 |
|---|---|---|
| CEO 更换 | 更新一行，把 Alice 改成 Bob | 新增 Bob 版本，关闭 Alice 的有效区间 |
| 药物暂停公告晚到 | 把暂停时间记成入库时间 | 暂停发生时间写入 `valid time`，入库时间写入 `transaction time` |
| 新闻与公告不一致 | 按最后到达者覆盖 | 在重叠时间片内按来源规则裁决 |

还有一个容易被忽略的问题是查询成本。覆盖式表查询简单，因为只有当前值；双时间模型查询复杂，因为你经常要问的是：

- 某一天现实中什么为真？
- 某一天系统当时认为什么为真？
- 现在回头看，系统什么时候第一次知道这件事？

这三类查询分别对应不同语义。如果表结构和索引设计不清楚，线上查询会很痛苦。因此工程上通常会同时维护两层数据：

1. 原始事实层，完整保留版本链和来源。
2. 融合视图层，供业务快速读取当前或某时点状态。

这样做的代价是写入流程更复杂，但换来的是可追溯性和可解释性。对于医学、金融、监管、供应链这类场景，这个代价通常是值得的。

---

## 替代方案与适用边界

不是所有系统都必须上双时间模型。是否值得做，要看变化频率、审计要求和多来源冲突的强度。

| 方案 | 适用场景 | 优点 | 局限 |
|---|---|---|---|
| 覆盖式当前表 | 只看最新值 | 简单，查询快 | 无历史 |
| 单时间版本表 | 只关心一类时间 | 实现较轻 | 迟到数据难解释 |
| 双时间模型 | 需要历史与审计 | 可追溯、可回放 | 设计复杂 |
| 事件流 | 高吞吐变化场景 | 适合增量写入 | 查询与裁决复杂 |

可以用一个简单判断法：

- 低频变化、无审计要求：覆盖式当前表通常够用。
- 低频变化、要看历史：单时间版本表可能够用。
- 多来源、常迟到、要追责：双时间模型更合适。
- 高频事件、以日志驱动为主：事件流或事件溯源更合适。

例如网站首页显示“当前价格”，很多时候只要最新值，不需要回答“系统在三周前知道的价格是多少”。但药物审批、供应商合规状态、企业高管任职、法律条款生效状态这类数据，往往都需要回答两个问题：

1. 某一天现实里到底是什么状态？
2. 某一天系统当时知道的是什么状态？

只有双时间或接近双时间的设计，才能稳定回答这两个问题。

因此，时序知识融合的适用边界很明确：当“事实变化”本身就是业务对象的一部分时，就不要把变化当噪声删掉；当“系统何时知道”会影响审计、决策或责任界定时，就不要只存一个时间字段。

---

## 参考资料

1. [XTDB: Bitemporality](https://v1-docs.xtdb.com/concepts/bitemporality/)
2. [Teradata: Bitemporal Tables](https://docs.teradata.com/r/Enterprise_IntelliFlex_VMware/Temporal-Table-Support/Basic-Temporal-Concepts/Transaction-Time-and-Valid-Time/Bitemporal-Tables)
3. [Temporal databases with two-dimensional time: Modeling and implementation of multihistory](https://www.sciencedirect.com/science/article/pii/0020025594900582)
4. [Reasoning over temporal knowledge graph with temporal consistency constraints](https://journals.sagepub.com/doi/10.3233/JIFS-210064)
5. [TeCre: A Novel Temporal Conflict Resolution Method Based on Temporal Knowledge Graph Embedding](https://www.mdpi.com/2078-2489/14/3/155)
