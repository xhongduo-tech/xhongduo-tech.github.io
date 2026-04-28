## 核心结论

Schema 映射解决的不是“把字段改个名字”，而是判断两个系统里的类、属性、关系是否表达同一个语义。语义就是“这个东西到底在说什么”。输出通常不是直接生效的最终规则，而是一组带置信度的候选映射。

真正能上线的映射，必须同时看四类证据：名称证据、实例证据、结构证据、约束证据。名称证据就是字段名或关系名像不像；实例证据就是样例值和实例共现像不像；结构证据就是它在图里的上下游连接像不像；约束证据就是类型、基数、枚举域等规则能不能对上。只看名字相似，常常会把“看起来像”误判成“语义相同”。

下面这张表先给结论，再给判断标准：

| 候选映射 | 名称相似 | 实例/结构/约束 | 结论 | 原因 |
|---|---:|---:|---|---|
| `authorOf -> writes` | 高 | 高 | 接受 | 主体和客体类型一致，实例共现稳定，查询回归通过 |
| `birthDate -> date_of_birth` | 高 | 中 | 暂不接受 | 一侧是 `date`，一侧是 `string`，需先做日期归一化 |
| `date -> birthDate` | 高 | 低 | 拒绝 | 一个系统表示出版日期，另一个表示出生日期，语义冲突 |
| `paper.title -> author.name` | 低 | 低 | 拒绝 | 类型和上下文都不兼容 |

玩具例子里，`authorOf` 和 `writes` 都像“作者写了作品”。真实工程里，是否接受还要看它们的定义域和值域，也就是“这个关系从什么类型指向什么类型”。如果一个关系从 `Author -> Paper`，另一个实际是 `Editor -> Book`，名字再像也不能直接映射。

---

## 问题定义与边界

Schema 是数据模式，指系统里“允许有哪些类、属性、关系，以及它们怎么约束”。Schema 映射就是在两个 Schema 之间建立语义对应，例如类到类、属性到属性、关系到关系。

常见输入包括：

| 输入形式 | 例子 | 映射对象 |
|---|---|---|
| 两个知识图谱 | RDF/OWL Schema | 类、属性、关系 |
| 两个数据库 | 表结构、字段、外键 | 表、列、关联 |
| 图谱与数据库 | 图谱属性对关系库字段 | 属性、关系、实体类型 |
| 标准 Schema 与业务 Schema | 公共词表对内部模型 | 规范类目到业务字段 |

要把边界说清楚。Schema 映射处理的是“语义对齐”，不是所有数据问题的总称。

| 任务 | 解决什么 | 例子 | 是否等同于 Schema 映射 |
|---|---|---|---|
| Schema 映射 | 模式层语义对应 | `birthDate -> date_of_birth` | 是 |
| 实体对齐 | 两条记录是不是同一对象 | `Alice Zhang` 是否等于 `A. Zhang` | 否 |
| 字段标准化 | 值格式统一 | `2024/01/01` 转成 `2024-01-01` | 否 |
| ETL 转换 | 抽取、转换、加载 | 把库 A 导入库 B | 否 |

例如 `birthDate`、`date_of_birth`、`dob` 很可能表达同一语义，但这只是“候选同义”。如果 `dob` 在某旧系统里存成 `"Jan 3, 1998"` 的字符串，而另一边要求 `xsd:date`，那 Schema 映射本身只会说“这两个字段可能对应”，不会自动替你完成解析失败、脏数据修复、时区补齐等清洗工作。

---

## 核心机制与推导

工程上常把一个候选映射写成：

$$
m_{ij} = (e_i, f_j, s_{ij})
$$

其中 $e_i$ 是源 Schema 的一个元素，$f_j$ 是目标 Schema 的一个元素，$s_{ij}$ 是综合得分。综合得分通常来自四类证据加权：

$$
s_{ij}=\alpha \cdot sim_{name}+\beta \cdot sim_{inst}+\gamma \cdot sim_{struct}+\delta \cdot sim_{cons}, \quad \alpha+\beta+\gamma+\delta=1
$$

这里的 `sim` 就是相似度分数，范围一般取 $[0,1]$。接受规则不是“分数高就上”，而是：

$$
accept(m_{ij}) = 1 \iff s_{ij}\ge \tau \land TypeOK \land CardOK \land QueryOK
$$

`TypeOK` 表示类型兼容，`CardOK` 表示基数约束兼容，`QueryOK` 表示下游查询回归通过。回归就是“改完以后原本应该成立的结果还成立”。

先看玩具例子 `authorOf -> writes`。假设证据如下：

| 证据 | 分数 | 说明 |
|---|---:|---|
| `sim_name` | 0.78 | 名称接近，但不是完全同名 |
| `sim_inst` | 0.84 | 很多 `(作者, 论文)` 实例在两边都能对齐 |
| `sim_struct` | 0.60 | 两边都连接作者与作品，但邻接结构不完全一致 |
| `sim_cons` | 0.90 | 定义域和值域都接近，约束兼容 |

如果权重取 $\alpha=0.2,\beta=0.4,\gamma=0.1,\delta=0.3$，则：

$$
s = 0.2 \times 0.78 + 0.4 \times 0.84 + 0.1 \times 0.60 + 0.3 \times 0.90 = 0.822
$$

这说明它是强候选，但还没有结束。若阈值 $\tau=0.8$，并且 `Author -> Paper` 对 `Person -> Publication` 被判定为类型兼容，关系也都是“一位作者可对应多篇作品”，再加上查询“某作者论文数”回归前后一致，就可以接受。

再看 `birthDate -> date_of_birth`。名称相似通常更高，综合分甚至可能超过阈值，但如果一边是 `xsd:date`，另一边是自由文本字符串，那么 `TypeOK=0`。这时正确做法不是强行上线，而是先引入日期归一化，把 `"1998/1/3"`、`"Jan 3 1998"` 统一转成标准日期，然后再重新验证。

可以把接受过程理解成一个简化流程：

| 步骤 | 问题 | 不通过时处理 |
|---|---|---|
| 1 | 是否进入候选池 | 直接丢弃 |
| 2 | 综合分是否过阈值 | 降级为人工复核 |
| 3 | 类型是否兼容 | 增加转换规则或拒绝 |
| 4 | 基数是否兼容 | 改成一对多映射或拒绝 |
| 5 | 查询回归是否通过 | 回滚映射，重新分析 |

真实工程例子更复杂。假设要融合“论文图谱、作者主数据、机构图谱”三个系统。你可能先得到候选：`affiliatedWith -> works_at`、`paperCount -> publication_total`、`birthDate -> date_of_birth`。这时不能只看候选分数，还要跑回归查询，比如“某作者论文数”“某机构下的论文列表”“某人的出生日期是否唯一且可解析”。只有这些关键查询稳定，映射才能进入 ETL 或虚拟图谱层。

---

## 代码实现

最小可运行实现通常分五步：候选生成、特征提取、打分、阈值判断、约束校验。关键点不是某一种方法最强，而是把规则、统计、LLM 三类信号统一到同一个候选结构里，便于排序、审计和复核。

一个常见数据结构如下：

| 字段 | 含义 |
|---|---|
| `name_a`, `name_b` | 源端与目标端元素名 |
| `sim_name` | 名称相似度 |
| `sim_inst` | 实例相似度 |
| `sim_struct` | 结构相似度 |
| `sim_cons` | 约束相似度 |
| `score` | 综合分 |
| `accepted` | 是否接受 |

```python
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class CandidateMapping:
    name_a: str
    name_b: str
    sim_name: float
    sim_inst: float
    sim_struct: float
    sim_cons: float
    score: float = 0.0
    accepted: bool = False

WEIGHTS = {
    "name": 0.2,
    "inst": 0.4,
    "struct": 0.1,
    "cons": 0.3,
}

def score_candidate(c: CandidateMapping) -> float:
    c.score = (
        WEIGHTS["name"] * c.sim_name
        + WEIGHTS["inst"] * c.sim_inst
        + WEIGHTS["struct"] * c.sim_struct
        + WEIGHTS["cons"] * c.sim_cons
    )
    return c.score

def type_ok(type_a: str, type_b: str) -> bool:
    compatible = {
        ("Author", "Person"),
        ("Paper", "Publication"),
        ("date", "date"),
    }
    return (type_a, type_b) in compatible or type_a == type_b

def card_ok(card_a: str, card_b: str) -> bool:
    # 简化处理：都允许 1-N 或 N-N 时判兼容
    return (card_a, card_b) in {
        ("1-N", "1-N"),
        ("N-N", "N-N"),
        ("1-N", "N-N"),
        ("N-N", "1-N"),
    }

def query_ok(before_after_counts: List[Tuple[int, int]]) -> bool:
    # 用关键查询结果是否一致做一个最小回归检查
    return all(before == after for before, after in before_after_counts)

def validate_mapping(
    c: CandidateMapping,
    meta: Dict[str, str],
    cards: Dict[str, str],
    regression: List[Tuple[int, int]],
    threshold: float = 0.8,
) -> bool:
    score_candidate(c)
    c.accepted = (
        c.score >= threshold
        and type_ok(meta["src_type"], meta["dst_type"])
        and card_ok(cards["src_card"], cards["dst_card"])
        and query_ok(regression)
    )
    return c.accepted

toy = CandidateMapping(
    name_a="authorOf",
    name_b="writes",
    sim_name=0.78,
    sim_inst=0.84,
    sim_struct=0.60,
    sim_cons=0.90,
)

accepted = validate_mapping(
    toy,
    meta={"src_type": "Author", "dst_type": "Person"},
    cards={"src_card": "1-N", "dst_card": "1-N"},
    regression=[(12, 12), (3, 3)],
)
assert round(toy.score, 3) == 0.822
assert accepted is True

birth = CandidateMapping(
    name_a="birthDate",
    name_b="date_of_birth",
    sim_name=0.92,
    sim_inst=0.80,
    sim_struct=0.70,
    sim_cons=0.85,
)

rejected = validate_mapping(
    birth,
    meta={"src_type": "date", "dst_type": "string"},
    cards={"src_card": "1-1", "dst_card": "1-1"},
    regression=[(1, 1)],
)
assert birth.score >= 0.8
assert rejected is False
```

这段代码故意保持简单，但它已经体现出一个核心思想：候选生成和候选接受是两回事。LLM 很适合根据字段名、样例值、字段说明生成 `top-k` 候选，例如把 `works_at`、`affiliatedWith`、`memberOf` 都提出来；规则法和统计法再补充分数；最后由 `TypeOK`、`CardOK`、`QueryOK` 做上线前闸门。

真实工程里，`query_ok` 不会只比两个计数，而会检查一批关键查询，例如：

1. 作者论文数是否变化异常。
2. 出生日期是否出现重复或空值暴增。
3. 机构到论文的可达路径是否断掉。

这一步的价值很高，因为它直接回答“这个映射会不会把业务查询搞坏”。

---

## 工程权衡与常见坑

高分不等于正确。工程上最容易出错的，不是“完全不像”的映射，而是“看起来很合理”的映射。

| 风险 | 表现 | 后果 | 规避方式 |
|---|---|---|---|
| 名称歧义 | `date`、`title`、`name` 到处都像 | 错把不同语义合并 | 引入上下文与定义域检查 |
| 样例值太少 | 只有几条样本就很像 | 统计证据失真 | 设最小样本量门槛 |
| 类型不一致 | `date` 对 `string`，`int` 对 `float` | 查询失败或隐式截断 | 先做归一化，再重验 |
| 基数冲突 | 一对一映射到一对多 | 聚合结果错误 | 显式建桥表或关系展开 |
| 单位不一致 | `cm` 对 `m`，`USD` 对 `CNY` | 数值可比性失效 | 单位转换规则入库 |
| 时区不一致 | UTC 对本地时间 | 时间过滤错误 | 统一时区和时间语义 |
| 枚举域不一致 | `M/F` 对 `Male/Female/Unknown` | 分类统计偏差 | 建枚举映射表 |
| LLM 幻觉 | 解释看起来顺，但不可复现 | 线上不稳定 | 只让 LLM 生成候选，不直接上线 |

一个典型坑是字段名完全相同，但语义不同。比如两个系统都有 `date`，一个表示论文发布日期，另一个表示人物出生日期。只做字符串匹配会给出高分，但一旦你跑查询“1990 年以后出生的作者数”，结果就会离谱。这类错误说明：名称相同只能证明“字面像”，不能证明“语义同”。

另一个常见坑是一对多、多对一。比如 A 系统有 `author_names`，把多个作者拼在一个字符串里；B 系统有规范化的 `writes` 关系，一篇论文连多个作者。这里不是简单字段映射，而是需要拆分、归一化、再建关系。如果团队只支持“一对一字段映射”，后续查询一定会失真。

---

## 替代方案与适用边界

没有一种方法能覆盖所有 Schema 映射场景。方法选择要看数据量、规则稳定性、文档质量、可审计要求和上线风险。

| 方法 | 优点 | 缺点 | 适用边界 | 是否可自动上线 |
|---|---|---|---|---|
| 规则法 | 可解释、稳定、便于审计 | 覆盖面有限，维护成本高 | 字段少、规则稳、领域术语固定 | 可以，前提是规则完备 |
| 统计法 | 能利用值分布和实例共现 | 依赖样本质量 | 样本较多、历史数据可用 | 可以，但需回归测试 |
| 机器学习法 | 能学习复杂特征组合 | 需要标注数据与训练维护 | 大规模异构 Schema | 谨慎，需监控漂移 |
| LLM 法 | 适合读字段描述、文档、注释 | 可复现性弱，易幻觉 | 文档丰富、候选空间大 | 不建议直接自动上线 |

小型业务主数据系统通常更适合规则法。原因不是它“高级”，而是它稳。比如十几个核心字段、命名习惯固定、术语白名单明确，这时同义词词典加人工规则，往往比训练复杂模型更便宜，也更可审计。

跨多个异构知识图谱时，LLM 更适合做候选生成器。因为它能同时读字段名、样例值、注释、文档段落，快速给出 `top-k` 候选。但它只应该解决“缩小搜索空间”这个问题，而不应该越过约束校验直接生效。真正的落地动作，仍然要回到类型兼容、基数兼容和查询回归。

可以把适用边界记成一句话：数据稀少但规则清楚，用规则法；数据规模大且模式复杂，用统计法或机器学习法；文本说明丰富但候选空间太大，用 LLM 先提候选，再用传统验证收口。

---

## 参考资料

| 作者/年份 | 主题 | 贡献 | 适合读者的阅读顺序 |
|---|---|---|---|
| Rahm & Bernstein, 2001 | Schema Matching 综述 | 经典问题定义、方法分类与评价框架 | 1 |
| Paes Leme et al., 2010 | OWL Schema Matching | 把本体和语义网场景下的匹配问题系统化 | 2 |
| W3C, 2012 | R2RML 标准 | 关系库到 RDF 的映射标准，偏工程落地 | 3 |
| Rodrigues & da Silva, 2021 | 机器学习方法综述 | 总结 ML 在 Schema Matching 中的适用方式 | 4 |
| Parciak et al., 2024 | LLM 在 Schema Matching 中的实验研究 | 展示大模型作为候选生成器的最新趋势 | 5 |

1. [A Survey of Approaches to Automatic Schema Matching](https://dbs.uni-leipzig.de/research/publications/a-survey-of-approaches-to-automatic-schema-matching) - 经典综述，解决“Schema 匹配到底有哪些主流路线”这个问题。  
2. [OWL schema matching](https://link.springer.com/article/10.1007/s13173-010-0005-3) - 面向 OWL 与语义网场景，解决“本体层语义如何对齐”这个问题。  
3. [R2RML: RDB to RDF Mapping Language](https://www.w3.org/TR/r2rml/) - W3C 标准，解决“关系数据库如何规范映射到 RDF”这个问题。  
4. [A study on machine learning techniques for the schema matching network problem](https://link.springer.com/article/10.1186/s13173-021-00119-5) - 机器学习综述，解决“统计与学习方法在匹配里怎么用”这个问题。  
5. [Schema Matching with Large Language Models: an Experimental Study](https://doi.org/10.48550/arXiv.2407.11852) - 前沿实验研究，解决“LLM 在候选生成与排序上的效果如何”这个问题。
