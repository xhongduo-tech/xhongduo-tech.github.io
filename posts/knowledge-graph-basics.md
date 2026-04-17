## 核心结论

知识图谱可以先看成一个“带语义约束的图式知识库”。它不是把一句话原样存进去，而是把“对象是什么”和“对象之间是什么关系”拆成可计算的结构。最基础的表达单位是三元组 $(h,r,t)$，在 RDF 语境下也常写成 $(s,p,o)$，分别表示头实体、关系、尾实体。

如果用集合形式表示，一个知识图谱可以写成：

$$
KG=(E,R,F)
$$

其中：

$$
F=\{(h,r,t)\mid h,t\in E,\ r\in R\}
$$

这里：

- `实体` 是可被单独指代的对象，如“巴黎”“法国”“卢浮宫”
- `关系` 是实体之间的语义连接，如“首都是”“位于”“作者是”
- `事实` 是一条已经规范化的知识记录，也就是一条可入库的三元组

最小玩具例子如下：

- 实体：`Paris`、`France`
- 关系：`isCapitalOf`
- 事实：`(Paris, isCapitalOf, France)`

它表示“巴黎是法国的首都”。一旦知识被表达成这种形式，系统就可以做语义查询、路径遍历、规则校验和多跳推理，而不只是检索同时出现过哪些字符串。

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $E$ | Entity set，实体集 | 图里有哪些“对象” |
| $R$ | Relation set，关系集 | 对象之间允许有哪些语义连接 |
| $F$ | Fact set，事实集 | 真正存进去的知识记录 |
| $(h,r,t)$ | 三元组 | 一条“谁-关系-谁”的事实 |

从工程角度看，知识图谱的价值不在“图”这个外形，而在“语义约束”这件事本身。只有先把事实集合 $F$ 建干净，再用统一标识、本体和查询语言把约束立住，知识图谱才会变成可检索、可复用、可推理的基础设施。

---

## 问题定义与边界

知识图谱要解决的问题，不是“把所有文本都自动变聪明”，而是“把离散知识规范化成机器可处理的事实网络”。

常见输入可以分成两类：

| 输入类型 | 典型来源 | 处理步骤 | 输出 |
|---|---|---|---|
| 非结构化文本 | 新闻、百科、论文、客服对话 | 实体抽取 → 关系抽取 → 实体链接 → 校验 | 规范 URI 三元组 |
| 结构化数据 | 数据库表、CSV、接口返回 | 字段映射 → 本体对齐 → ID 统一 → 校验 | 规范 URI 三元组 |

几个核心术语先解释清楚：

| 术语 | 定义 | 为什么需要它 |
|---|---|---|
| 实体抽取 | 从文本里识别重要对象 | 不先找出对象，就谈不上关系 |
| 关系抽取 | 判断对象之间的语义连接 | 决定三元组里的谓词是什么 |
| 实体链接 | 把文本名字对齐到标准实体 | 解决“同名不同物”和“同物不同名” |
| 指代消解 | 判断代词或省略成分指向谁 | 防止把 `it`、`he` 之类误当新实体 |
| URI / IRI | 统一资源标识 | 给实体一个不歧义的全局 ID |

例如句子：

`巴黎是法国的首都。`

理想输出不是原句文本，而是：

```text
(Paris, isCapitalOf, France)
```

再看一个稍复杂的例子：

`巴黎市政府宣布新规。两天后 it 发布了详细解释。`

如果系统不做指代消解，就可能错误地产生：

```text
(it, published, DetailedExplanation)
```

正确做法是先结合上下文判断 `it = 巴黎市政府`，再形成：

```text
(ParisCityGovernment, published, DetailedExplanation)
```

因此，知识图谱构建更接近“规范化建模”，而不是“把句子切词后丢进数据库”。

一个极简接口可以写成：

$$
extract\_triple(text)\rightarrow (subject,predicate,object)
$$

但这个接口很容易让人误解任务复杂度。真正困难的不是返回一个三元组，而是保证这个三元组：

1. 指向的是正确对象
2. 关系名称是统一的
3. 主语和宾语方向没写反
4. 结果符合本体和类型约束
5. 能和已有知识库中的实体合并

知识图谱的边界也必须说清楚：

1. 它不自动保证事实为真。抽取出来的通常只是“候选事实”，还需要校验、溯源和置信度控制。
2. 它不等于全文检索。检索擅长找“哪些文档提到了这个词”，图谱擅长找“这个对象和哪些对象存在特定关系”。
3. 它不是所有数据系统的默认答案。如果业务只是按主键查表，关系数据库更直接、更便宜。
4. 它要求先定义实体类型和关系类型。否则同一个语义会在不同数据源里变成 `author_of`、`written_by`、`creator`，后续无法统一查询。

所以，知识图谱适合的问题非常明确：实体明确、关系明确、希望做语义查询、联结分析或多跳推理；它不适合把一切文本原样塞进去，再期待系统自动理解世界。

---

## 核心机制与推导

知识图谱可以分三层理解：集合层、表示层、查询层。

先看集合层。定义：

$$
KG=(E,R,F)
$$

其中：

- $E$ 是实体集合
- $R$ 是关系集合
- $F$ 是事实集合

事实必须由实体和关系组成，因此：

$$
F=\{(h,r,t)\mid h,t\in E,\ r\in R\}
$$

这句话的意思是：任意一条合法事实都必须满足“头尾是实体，中间是某个已定义关系”。这一步看似简单，实际已经引入了最重要的工程约束：图谱不是任意连边，而是受语义模式控制的连边。

继续看玩具例子：

$$
E=\{Paris, France, LeLouvre\}
$$

$$
R=\{isCapitalOf, locatedIn\}
$$

$$
F=\{(Paris,isCapitalOf,France),(LeLouvre,locatedIn,Paris)\}
$$

对应图结构可以画成：

```text
[LeLouvre] --locatedIn--> [Paris] --isCapitalOf--> [France]
```

从这个图里可以直接读出两跳路径：

```text
LeLouvre -> Paris -> France
```

这意味着系统即使没有显式存一条 `(LeLouvre, locatedInCountry, France)`，也已经具备沿路径回答“卢浮宫在哪个国家”的基础。

第二层是表示层。最常见的标准是 `RDF`。它可以把上面的事实写成主语、谓语、宾语三部分：

$$
(s,p,o)=(Paris,\ isCapitalOf,\ France)
$$

真实系统通常不直接用裸字符串，而是用 URI/IRI 标识实体和关系，例如：

```text
<http://example.org/entity/Paris> <http://example.org/relation/isCapitalOf> <http://example.org/entity/France>
```

这样做不是形式主义，而是为了解决歧义和跨系统对齐问题。字符串 `Paris` 可能表示巴黎市、巴黎大学、美国得州某个县；URI 表示的是“唯一对象”，不是“一个名字”。

再往前一步，宾语并不总是实体。知识图谱里常见两种尾部对象：

| 对象类型 | 例子 | 说明 |
|---|---|---|
| 实体 | `(LeLouvre, locatedIn, Paris)` | 尾部是另一个节点 |
| 字面量 | `(LeLouvre, annualVisitors, "10000000")` | 尾部是普通值，不再继续连边 |

所以更准确地说，工程实现里通常会同时处理“实体关系事实”和“实体属性事实”。

第三层是查询层。RDF 图谱常用 `SPARQL` 查询。比如查询“法国的首都是谁”：

```sparql
SELECT ?capital
WHERE {
  ?capital <http://example.org/relation/isCapitalOf> <http://example.org/entity/France> .
}
```

如果图谱里有三元组 `(Paris, isCapitalOf, France)`，就会返回 `Paris`。

再看一个多跳查询。假设图谱中有：

```text
(LeLouvre, locatedIn, Paris)
(Paris, isCapitalOf, France)
```

那么“位于法国首都的博物馆有哪些”可以写成：

```sparql
SELECT ?museum
WHERE {
  ?museum <http://example.org/relation/locatedIn> ?city .
  ?city   <http://example.org/relation/isCapitalOf> <http://example.org/entity/France> .
}
```

这和全文检索的核心差别是：

| 方法 | 依赖对象 | 本质 |
|---|---|---|
| 全文检索 | 字符串共现、关键词相关性 | 找“哪些文本像在回答问题” |
| 知识图谱 | 实体、关系、图结构 | 找“哪些结构满足语义条件” |

本体会进一步给图谱增加约束。`本体` 可以先理解成“对类型、关系和规则的正式定义”。例如：

- `CapitalCity` 是 `City` 的子类
- `isCapitalOf` 的主语必须是 `City`
- `isCapitalOf` 的宾语必须是 `Country`

这样，当有人错误写入：

```text
(France, isCapitalOf, Paris)
```

系统就能据此发现方向不对。这里的约束常写成：

$$
domain(isCapitalOf)=City,\quad range(isCapitalOf)=Country
$$

如果记实体类型函数为 $\tau(\cdot)$，那么合法事实需要满足：

$$
\tau(h)\in domain(r),\quad \tau(t)\in range(r)
$$

这一步很关键。它说明知识图谱比“随便画个图”更有工程价值的原因，不在节点和边，而在“边能不能合法存在”。

真实系统通常会在下面几个环节一起工作：

```text
数据接入 -> 实体标准化 -> 关系规范化 -> 本体校验 -> 入库 -> 查询/推理
```

以开放中文知识图谱为例，像 CN-DBpedia 这类系统，会汇总百科、结构化表格和半结构化页面，再经过抽取、对齐、消歧和融合，形成大规模实体关系网络。用户问“《红楼梦》的作者是谁”，系统不需要搜索整页文本，而是沿 `作品 -> 作者` 关系边取到 `曹雪芹`；继续问“他生活在哪个朝代”，则可以继续沿 `作者 -> 朝代` 做多跳查询。这正是知识图谱在问答、搜索增强和知识增强检索中的基础价值。

---

## 代码实现

下面给出一个可直接运行的最小 Python 示例。它覆盖四件事：

1. 定义实体、关系和类型
2. 校验三元组是否合法
3. 支持按谓词查询和多跳查询
4. 区分实体对象与字面量对象

```python
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

Node = str
Literal = Union[str, int, float]

@dataclass(frozen=True)
class Triple:
    s: Node
    p: str
    o: Union[Node, Literal]

class MiniKG:
    def __init__(self) -> None:
        self.entities: Set[Node] = set()
        self.relations: Set[str] = set()
        self.entity_types: Dict[Node, str] = {}
        self.relation_schema: Dict[str, Tuple[str, str]] = {}
        self.triples: List[Triple] = []
        self.spo_index: Dict[Tuple[Node, str], List[Union[Node, Literal]]] = defaultdict(list)
        self.pos_index: Dict[Tuple[str, Union[Node, Literal]], List[Node]] = defaultdict(list)

    def add_entity(self, entity: Node, entity_type: str) -> None:
        self.entities.add(entity)
        self.entity_types[entity] = entity_type

    def add_relation(self, relation: str, domain: str, range_: str) -> None:
        self.relations.add(relation)
        self.relation_schema[relation] = (domain, range_)

    def validate_triple(self, triple: Triple) -> bool:
        if triple.s not in self.entities:
            return False
        if triple.p not in self.relations:
            return False

        domain, range_ = self.relation_schema[triple.p]
        if self.entity_types.get(triple.s) != domain:
            return False

        # range_ == "Literal" 表示允许普通值，不要求 o 是实体
        if range_ == "Literal":
            return not isinstance(triple.o, Triple)

        if triple.o not in self.entities:
            return False
        if self.entity_types.get(triple.o) != range_:
            return False

        return True

    def add_triple(self, triple: Triple) -> None:
        if not self.validate_triple(triple):
            raise ValueError(f"invalid triple: {triple}")
        self.triples.append(triple)
        self.spo_index[(triple.s, triple.p)].append(triple.o)
        self.pos_index[(triple.p, triple.o)].append(triple.s)

    def query_object(self, subject: Node, predicate: str) -> List[Union[Node, Literal]]:
        return list(self.spo_index.get((subject, predicate), []))

    def query_subject(self, predicate: str, obj: Union[Node, Literal]) -> List[Node]:
        return list(self.pos_index.get((predicate, obj), []))

    def two_hop_query(
        self,
        start: Node,
        predicate1: str,
        predicate2: str,
    ) -> List[Union[Node, Literal]]:
        results: List[Union[Node, Literal]] = []
        middle_nodes = self.query_object(start, predicate1)
        for middle in middle_nodes:
            if isinstance(middle, str) and middle in self.entities:
                results.extend(self.query_object(middle, predicate2))
        return results


def build_demo_kg() -> MiniKG:
    kg = MiniKG()

    kg.add_entity("Paris", "City")
    kg.add_entity("France", "Country")
    kg.add_entity("LeLouvre", "Museum")
    kg.add_entity("MonaLisa", "Artwork")

    kg.add_relation("isCapitalOf", "City", "Country")
    kg.add_relation("locatedIn", "Museum", "City")
    kg.add_relation("displays", "Museum", "Artwork")
    kg.add_relation("annualVisitors", "Museum", "Literal")

    kg.add_triple(Triple("Paris", "isCapitalOf", "France"))
    kg.add_triple(Triple("LeLouvre", "locatedIn", "Paris"))
    kg.add_triple(Triple("LeLouvre", "displays", "MonaLisa"))
    kg.add_triple(Triple("LeLouvre", "annualVisitors", 10000000))

    return kg


def run_tests() -> None:
    kg = build_demo_kg()

    assert kg.query_object("Paris", "isCapitalOf") == ["France"]
    assert kg.query_subject("isCapitalOf", "France") == ["Paris"]
    assert kg.query_object("LeLouvre", "annualVisitors") == [10000000]
    assert kg.two_hop_query("LeLouvre", "locatedIn", "isCapitalOf") == ["France"]

    try:
        kg.add_triple(Triple("France", "isCapitalOf", "Paris"))
        raise AssertionError("expected validation error")
    except ValueError:
        pass

    print("all tests passed")


if __name__ == "__main__":
    run_tests()
```

这段代码可以直接运行，输出：

```text
all tests passed
```

它展示了三个最基础的工程点：

1. 三元组不是随便加的，必须经过模式校验
2. 宾语可能是实体，也可能是字面量
3. 多跳查询本质上是沿着边做受约束遍历

如果用 JavaScript 写一个浏览器可运行的最小版本，可以这样表示：

```javascript
const triples = [
  { s: "Paris", p: "isCapitalOf", o: "France" },
  { s: "LeLouvre", p: "locatedIn", o: "Paris" },
  { s: "LeLouvre", p: "annualVisitors", o: 10000000 }
];

function queryObject(subject, predicate) {
  return triples
    .filter(t => t.s === subject && t.p === predicate)
    .map(t => t.o);
}

function querySubject(predicate, object) {
  return triples
    .filter(t => t.p === predicate && t.o === object)
    .map(t => t.s);
}

console.log(queryObject("Paris", "isCapitalOf"));   // ["France"]
console.log(querySubject("isCapitalOf", "France")); // ["Paris"]
```

如果需要更接近 RDF 的文本格式，可以序列化成 Turtle：

```turtle
@prefix ex: <http://example.org/> .

ex:Paris ex:isCapitalOf ex:France .
ex:LeLouvre ex:locatedIn ex:Paris .
ex:LeLouvre ex:annualVisitors 10000000 .
```

对应的 SPARQL 查询可以写成：

```sparql
SELECT ?museum
WHERE {
  ?museum <http://example.org/locatedIn> <http://example.org/Paris> .
}
```

再看一个对新手最容易踩坑的地方。输入文本：

`Le Louvre is in Paris; it attracts 10M visitors.`

合理的入库流程不是“看见词就连边”，而是：

```text
文本
 -> 实体识别：Le Louvre, Paris
 -> 指代消解：it = Le Louvre
 -> 关系抽取：locatedIn, annualVisitors
 -> 对象分类：Paris 是实体，10M 是字面量
 -> 本体校验
 -> 入库
```

最后形成的更合理结果是：

- `(LeLouvre, locatedIn, Paris)`
- `(LeLouvre, annualVisitors, 10000000)`

而不是错误地生成：

- `(it, locatedIn, Paris)`
- `(it, attractsVisitors, 10M)`

所以，代码实现的重点从来不只是“把三元组存进去”，而是“在入库前把对象、关系和类型规范化”。

---

## 工程权衡与常见坑

知识图谱最容易被低估的部分，不是存储，而是清洗、约束和维护。很多项目失败，不是因为没有图数据库，而是因为事实质量差、命名混乱、约束缺失。

常见坑如下：

| 问题 | 表现 | 后果 | 规避策略 |
|---|---|---|---|
| 指代错误 | 把 `it`、`he`、`this company` 当成新实体 | 图谱中出现垃圾节点 | 做指代消解，优先回指最近合法实体 |
| 实体歧义 | “Apple”既可能是公司，也可能是水果 | 查询结果混乱 | 使用标准 URI，并结合上下文消歧 |
| 关系方向错 | 把 `(France, isCapitalOf, Paris)` 写反 | 推理和查询失败 | 为关系定义 domain/range 约束 |
| 关系命名失控 | `written_by`、`author_of`、`creator` 混用 | 无法统一查询 | 做关系规范化和本体对齐 |
| 事实冲突 | 两个来源给出不同出生日期 | 图谱不一致 | 保留来源、时间戳和置信度 |
| 抽取噪声高 | 从文本抽到大量伪事实 | 图谱可用性下降 | 规则校验 + 人工抽样审核 |
| 字面量误建模 | 把数值、日期都当实体节点 | 图谱膨胀且语义混乱 | 区分实体关系和属性值 |
| 模式过松 | 什么都能连，什么类型都能接 | 后期难维护 | 尽早定义最小可用本体 |
| 模式过严 | 新来源接入成本极高 | 扩展速度慢 | 核心关系严格，边缘关系渐进收敛 |

`domain/range` 可以理解成“某个关系的起点和终点应该是什么类型”。例如：

| 关系 | domain | range |
|---|---|---|
| `isCapitalOf` | `City` | `Country` |
| `locatedIn` | `Museum` | `City` |
| `authorOf` | `Person` | `Work` |
| `annualVisitors` | `Museum` | `Literal` |

继续看一个典型坑：

`Le Louvre is in Paris; it attracts 10M visitors.`

如果系统只按“最近名词”或者“最近代词”粗暴抽取，很容易得到：

- `(it, attractsVisitors, 10M)`

这条记录结构上像三元组，语义上却不可用，因为主语不是已知实体。正确流程应该是：

```text
文本
 -> 实体识别：Le Louvre, Paris
 -> 句法分析：it 指向前句主语
 -> 指代消解：it = Le Louvre
 -> 关系抽取：locatedIn, annualVisitors
 -> 对象分类：Paris 是实体，10M 是字面量
 -> 本体校验
 -> 入库
```

这里顺带澄清一个常见误区：并不是所有宾语都必须是实体。像“10000000”“2024-01-01”“3.14”这类值，更适合作为 `literal`，也就是普通值，而不是图中的可继续扩展节点。

真实工程里还要面对三类持续权衡：

1. 抽取精度与覆盖率的权衡。抽取得太保守，图谱会稀疏；太激进，噪声会迅速积累。
2. 本体统一与业务灵活性的权衡。本体太严格，接新数据源很痛苦；太松散，后续查询无法稳定复用。
3. 实时更新与一致性的权衡。实时写入延迟低，但更容易引入脏数据；离线批处理质量高，但时效性差。

因此，成熟的知识图谱系统通常不是单层流程，而是双层机制：

```text
候选事实生成层：尽量抽到
事实治理层：尽量挡错
```

只有抽取、链接、校验、溯源、版本管理一起工作，图谱才会稳定。

---

## 替代方案与适用边界

知识图谱不是关系数据库、全文检索或向量数据库的替代品。它们解决的是不同问题，常见做法也是组合使用，而不是只留一种。

对比如下：

| 方案 | 擅长场景 | 表达力 | 推理能力 | 适用边界 |
|---|---|---|---|---|
| SQL / 关系数据库 | 强结构化表数据、事务处理 | 高，但偏表结构 | 弱 | 订单、账户、库存、报表 |
| Full-text 全文检索 | 文档召回、关键词匹配 | 中 | 很弱 | 搜文章、搜页面、站内检索 |
| Vector DB 向量库 | 语义相似召回 | 中 | 弱 | RAG、相似内容推荐、召回增强 |
| 知识图谱 | 实体关系、多跳查询、规则推理 | 高，偏语义结构 | 强 | 问答、推荐、知识增强、联结分析 |

先看最简单的 SQL 例子。若只存国家和首都：

| country | capital |
|---|---|
| France | Paris |

查询法国首都可以写：

```sql
SELECT capital
FROM capitals
WHERE country = 'France';
```

这个场景下，SQL 完全足够，没必要强行上知识图谱。

但如果问题升级成：

- 法国首都有哪些著名博物馆？
- 这些博物馆收藏了哪些作品？
- 这些作品的作者属于哪个流派？
- 这些流派在哪些城市还有代表作品？

SQL 当然也能做，但会迅速演变成多表连接、模式耦合和字段命名管理问题。知识图谱在这类问题上的优势，不是“语句一定更短”，而是“问题本身就是沿关系路径展开的”。

同样的查询，用 SPARQL 更贴近语义结构：

```sparql
SELECT ?museum
WHERE {
  ?capital <http://example.org/isCapitalOf> <http://example.org/France> .
  ?museum  <http://example.org/locatedIn> ?capital .
  ?museum  a <http://example.org/Museum> .
}
```

再看一个更现实的系统组合：

| 层 | 常用技术 | 作用 |
|---|---|---|
| 文档召回层 | 全文检索 / 向量检索 | 先从大量文本里找候选内容 |
| 结构化事实层 | 知识图谱 | 表达实体与关系 |
| 事务层 | 关系数据库 | 管订单、权限、日志、配置 |
| 应用层 | API / 问答 / 推荐服务 | 把多种能力拼成产品 |

这也是为什么在 RAG、搜索增强或智能问答系统里，知识图谱通常不是单独存在，而是和检索系统一起工作。

一个典型开放图谱接入流程是：

1. 从用户问题中识别实体，如“红楼梦”
2. 查询开放图谱中的标准实体 ID
3. 按关系类型请求“作者”“朝代”“相关人物”等事实
4. 将外部事实与本地业务知识融合
5. 再决定是直接回答，还是继续做多跳推理

它的适用边界也很清楚：

- 如果业务知识主要是内部事务数据，且查询模式固定，优先考虑关系数据库
- 如果目标是从海量文档中做召回，优先考虑全文检索或向量检索
- 如果核心问题是“对象之间的关系本身就是业务对象”，知识图谱就更合适

一句话概括：知识图谱最适合处理“关系结构本身有价值”的问题，不适合把所有数据系统都替换掉。

---

## 参考资料

| 来源 | 用途 | 链接说明 |
|---|---|---|
| W3C《RDF 1.1 Concepts and Abstract Syntax》 | RDF 三元组、图模型的正式定义 | https://www.w3.org/TR/rdf11-concepts/ |
| W3C《RDF 1.1 Turtle》 | Turtle 序列化格式示例与语法 | https://www.w3.org/TR/turtle/ |
| W3C《SPARQL 1.1 Query Language》 | SPARQL 查询语法与能力边界 | https://www.w3.org/TR/sparql11-query/ |
| Wikipedia《Knowledge graph》 | 知识图谱概念概览与历史背景 | https://en.wikipedia.org/wiki/Knowledge_graph |
| Hogan et al., *Knowledge Graphs*, ACM Computing Surveys, 2021 | 系统综述，适合补充工程视角与研究脉络 | https://doi.org/10.1145/3447772 |
| CN-DBpedia 介绍页 | 中文开放知识图谱实例、规模与应用 | https://kw.fudan.edu.cn/cndbpedia/intro/ |
