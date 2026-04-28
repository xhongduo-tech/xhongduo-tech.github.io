## 核心结论

命名空间 URI 规范管理，解决的是 RDF 和知识图谱里“同一个东西到底怎么稳定命名”的问题。命名空间 URI 是一组术语的边界，白话说就是“这套字段和概念属于哪个词表”；实体 URI 是具体对象的标识，白话说就是“这个人、这个商品、这家公司到底是谁”。

最重要的结论只有一句：**稳定的是实体，变化的是词表版本。**  
如果把实体 URI 也跟着 schema 一起改，旧数据、外部链接、映射关系、去重规则都会被连带打碎。更稳妥的做法是：

$$
entityURI = stable\_base + local\_id
$$

$$
termURI = namespaceURI + local\_name
$$

其中 `local_id` 应该来自稳定主键，白话说就是“不容易变的内部编号”；`local_name` 是词表中的术语名，比如 `name`、`birthDate`、`supplierCode`。

下面这张表先把三类 URI 分清：

| 对象 | 作用 | 是否应长期稳定 | 例子 |
|---|---|---:|---|
| 实体 URI | 标识现实对象 | 是 | `https://kg.example.com/id/person/123` |
| 词表 URI | 定义字段和类型 | 通常稳定，但可按版本演进 | `https://kg.example.com/ns/2025/people#name` |
| 别名 URI | 吸收历史写法或外部写法 | 可保留，但不做主键 | `https://old.example.com/person/123` |

玩具例子可以这样理解：命名空间像字典目录，实体 URI 像身份证号。字典可以出新版，身份证号不能今天换一版、明天再换一版。

真实工程里，命名空间设计不是“美观问题”，而是后续对齐、映射、去重、迁移的成本开关。URI 规则早定清楚，后面图谱越大，收益越明显。

---

## 问题定义与边界

这篇文章讨论的不是“网页 URL 怎么写好看”，也不是“前端路由该不该带参数”，而是**知识图谱中 URI 的长期可维护性**。具体边界包括四件事：

| 问题 | 不处理的后果 | 正确做法 |
|---|---|---|
| 同一实体被多个系统生成多个 URI | 去重困难，查询结果碎片化 | 设 canonical URI，统一入图 |
| schema 升级时直接改实体 URI | 历史数据断链，外部引用失效 | 只升级命名空间或版本声明 |
| 大小写、尾斜杠、`#` 写法不统一 | 同义 URI 被当成不同主键 | 制定归一化规则 |
| 展示文案编码进 URI | 标题一改，标识就变 | URI 只承载稳定标识，不承载文案 |

这里有四个术语需要先定边界：

| 术语 | 定义 | 白话解释 |
|---|---|---|
| 命名空间 URI | 一组 RDF 术语共享的前缀空间 | 一套字段来自哪本“字典” |
| 实体 URI | 唯一标识具体资源的 URI | 这个对象的固定身份证号 |
| canonical URI | 规范主 URI | 系统最终认定的唯一正式写法 |
| alias URI | 历史或外部别名 URI | 旧编号、别称、兼容写法 |

一个典型场景是企业知识图谱接入商品主数据、供应商系统和内容标签系统。三个系统都表示“同一供应商”，但各自生成不同 URI。此时问题不是语义推理，而是最基础的标识不统一。后续做实体对齐，也就是“把多个来源中同一个对象认成一个对象”的过程，会变得很重。

不在本文讨论范围内的内容包括：

| 不讨论内容 | 原因 |
|---|---|
| 页面展示 URL 设计 | 这是 Web 导航问题，不是 RDF 标识问题 |
| 前端路由设计 | 与知识图谱标识稳定性不是一层问题 |
| 业务字段如何展示 | 这是产品和内容规则，不是 URI 规范本身 |

---

## 核心机制与推导

核心机制可以压缩成三条规则：

1. 实体 URI 稳定。
2. 词表版本可演进。
3. alias 只做映射，不做主键。

先看一个最小例子。

- 实体 URI：`https://kg.example.com/id/person/123`
- 2024 版词表：`https://kg.example.com/ns/2024/people#name`
- 2025 版词表：`https://kg.example.com/ns/2025/people#fullName`

这里 schema 变了，但人还是那个人，所以 `person/123` 不该变。变化发生在术语层，而不是实体层。这个设计背后的推导是：

$$
\Delta entityURI = 0,\quad \Delta schemaVersion = 1
$$

也就是说，版本变化应落在词表，不应落在实体标识。

为什么？因为实体 URI 的职责是“唯一定位对象”，不是“描述对象今天长什么样”。如果把 schema 变化绑到实体 URI 上，会出现三个直接问题：

| 错误做法 | 短期看起来 | 长期结果 |
|---|---|---|
| schema 改了就重发实体 URI | 实现简单 | 历史引用全部失效 |
| 把字段名写进实体 URI | 可读性高 | 字段一变，标识失稳 |
| alias 和 canonical 混着用 | 兼容方便 | 主键不唯一，聚合失真 |

玩具例子：  
你给图书馆里的书编号。如果今天按“文学/001”，明天按“现代文学/001”，后天按“长篇小说/001”，那编号就不再是编号，而是在混入分类规则。分类会变，书本身没变。知识图谱的实体 URI 也是同理。

再看 `#` 和 `/`。它们都能用，但作用方式不同。

- `#` 风格常见于词表：`https://kg.example.com/ns/people#name`
- `/` 风格常见于资源分层：`https://kg.example.com/ns/people/name`

`#` 的白话解释是“同一文档内的片段标识”；`/` 的白话解释是“路径层级”。从 RDF 角度，两者都合法，真正关键不是哪种更高级，而是**不要在同一套规则里混用**。因为混用后，解析策略、缓存策略、文档发布方式都容易不一致。

真实工程例子：  
某零售公司做供应链知识图谱，供应商来自 ERP、采购平台和外部信用库。最开始三个系统各自发 URI：

- ERP：`https://erp.example.com/vendor/8891`
- 采购：`https://proc.example.com/supplier/S-204`
- 外部信用库：`https://credit.example.org/company/91310000xxxx`

如果直接把三者都当主 URI，图里就会出现三个“供应商 A”。正确做法是先设 canonical URI，例如：

`https://kg.example.com/id/supplier/000128`

再把其他 URI 记录为 alias，并维护映射关系。这样 schema 升级时只更新词表，例如 `supplierName` 是否改为 `legalName`，不会影响这个供应商的实体 URI。

---

## 代码实现

工程里不能只靠“大家记住规则”，而要把规则落成代码。最少需要四类逻辑：生成、校验、归一化、映射。

下面是一个可运行的 Python 玩具实现：

```python
from urllib.parse import urlparse

STABLE_ENTITY_BASE = "https://kg.example.com/id"
SCHEMA_BASE = "https://kg.example.com/ns"

def build_entity_uri(entity_type: str, local_id: str) -> str:
    entity_type = entity_type.strip().lower()
    local_id = local_id.strip()
    assert entity_type and local_id
    return f"{STABLE_ENTITY_BASE}/{entity_type}/{local_id}"

def build_term_uri(version: str, vocab: str, local_name: str) -> str:
    version = version.strip()
    vocab = vocab.strip().lower()
    local_name = local_name.strip()
    assert version and vocab and local_name
    return f"{SCHEMA_BASE}/{version}/{vocab}#{local_name}"

def normalize_uri(uri: str) -> str:
    uri = uri.strip()
    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    fragment = parsed.fragment
    normalized = f"{scheme}://{netloc}{path}"
    if fragment:
        normalized += f"#{fragment}"
    return normalized

def resolve_canonical(uri: str, alias_map: dict[str, str]) -> str:
    normalized = normalize_uri(uri)
    return alias_map.get(normalized, normalized)

alias_map = {
    "https://erp.example.com/vendor/8891": "https://kg.example.com/id/supplier/000128",
    "https://proc.example.com/supplier/S-204": "https://kg.example.com/id/supplier/000128",
}

entity_uri = build_entity_uri("Supplier", "000128")
term_uri = build_term_uri("2025", "supplier", "legalName")

assert entity_uri == "https://kg.example.com/id/supplier/000128"
assert term_uri == "https://kg.example.com/ns/2025/supplier#legalName"
assert normalize_uri(" HTTPS://KG.EXAMPLE.COM/id/supplier/000128/ ") == "https://kg.example.com/id/supplier/000128"
assert resolve_canonical("https://erp.example.com/vendor/8891", alias_map) == entity_uri
```

这个实现虽然简单，但已经覆盖了核心思想：

| 输入 | 处理 | 输出 | 是否稳定 |
|---|---|---|---:|
| `entity_type + local_id` | 按固定规则拼接 | 实体 URI | 是 |
| `version + vocab + local_name` | 按词表版本生成 | 术语 URI | 版本内稳定 |
| 历史 URI | 归一化后查 alias_map | canonical URI | 是 |
| 非规范 URI | 统一大小写和尾斜杠 | 规范 URI | 是 |

真实工程里通常还会再加两层：

1. 入库前校验器  
检查 host 是否统一、路径是否符合模板、是否混入中文标题或空格。

2. namespace document  
也就是“命名空间说明文档”，明确版本策略、弃用策略、解析地址、兼容策略。没有这份文档，别人拿到 URI 也不知道规则是什么。

一个常见误区是“URI 可读性越高越好”，于是有人写出：

`https://kg.example.com/person/zhang-san-beijing-team-leader`

这类 URI 把展示文案混进了标识。名字、部门、城市都可能变化，一旦变化，URI 就不再稳定。正确做法是把这些信息放在 RDF 属性里，而不是 URI 本体里。

---

## 工程权衡与常见坑

URI 规范最容易失败的地方，不是 RDF 标准本身，而是工程实现细节没收口。

| 坑点 | 后果 | 规避措施 |
|---|---|---|
| 大小写混用 | `/Person/123` 与 `/person/123` 被当成两个资源 | 约定路径全部小写，生成器强制转换 |
| `#` 和 `/` 混用 | 词表发布和解析规则混乱 | 同一类命名空间只选一种 |
| 展示文案写进 URI | 文案一改就断链 | URI 只放稳定 ID |
| 可变业务字段做主键 | 手机号、SKU 变更后主键漂移 | 使用内部不可变 ID |
| 没有 namespace document | 外部团队无法正确接入 | 发布规范说明与版本策略 |
| 未做归一化 | 尾斜杠、空格、host 大小写导致重复 | 入库前统一 normalize |
| 同义实体多 URI 并存 | 去重和映射成本快速上升 | canonical + alias 映射 |
| 直接批量改实体 URI | 历史三元组、缓存、外链全部受影响 | 用映射层兼容历史 URI |

这里有一个工程判断要说清楚：**规范文档本身不产生稳定性，只有“规范 + 代码 + 流程”一起落地才产生稳定性。**

至少要把检查放进这三个环节：

| 环节 | 应检查内容 |
|---|---|
| 生成前 | 是否使用规定 base URI，是否使用稳定 ID |
| 入库前 | 是否完成归一化，是否命中 alias 映射 |
| 发布前 | 是否存在 namespace document，是否声明版本和弃用策略 |

如果团队已经有历史脏数据，不要幻想一次性“洗干净”。更现实的做法是：

1. 先冻结新规则。
2. 新数据全部走 canonical。
3. 老数据通过 alias 映射逐步吸收。
4. 查询层统一返回 canonical。
5. 后台异步做补写和迁移。

这样做比全量重写风险小得多。

---

## 替代方案与适用边界

URI 设计没有唯一正确答案，但有明显错误答案。替代方案可以有多种，前提是长期一致。

先看 `#` 与 `/`：

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| `#` 风格命名空间 | 词表表达紧凑，常见于 RDF 词表 | 文档片段语义更强，不适合随意路径化 | 小而稳定的词表 |
| `/` 风格命名空间 | 更像普通 Web 资源路径，层级清晰 | 需更明确处理发布和缓存规则 | 需要更强资源分层管理的团队 |

再看版本策略：

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 直接改 URI | 实现直观 | 历史断链，迁移成本高 | 几乎不推荐 |
| 版本命名空间管理 | 实体稳定，schema 可演进 | 需要维护版本文档 | 长期运行的图谱系统 |

再看多 URI 问题：

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 多 URI 并存 | 接入快，不用清洗 | 查询、去重、聚合复杂 | 临时接入阶段 |
| canonical + alias 映射 | 一致性高，后续成本低 | 初期要做映射治理 | 正式生产环境 |

适用边界也要说清楚。  
如果你做的是完整 RDF/知识图谱平台，这套规则应尽早建立。因为数据源会增加、schema 会演进、跨团队引用会变多，URI 稳定性会直接变成工程成本。

如果你做的只是一次性离线分析、内部短期脚本，稳定性要求可以降低，甚至可以只保证文件内唯一。但只要数据要跨系统流动、要被别人复用、要长期保存，就不该把 URI 当成随手拼接的字符串。

一句话概括适用边界：**越接近长期共享知识资产，URI 规范越要前置；越接近临时内部数据，规范可以适度简化。**

---

## 参考资料

| 资料 | 作用 | 对应部分 |
|---|---|---|
| RFC 3986 | URI 基础语法和规范边界 | 问题定义、归一化 |
| Cool URIs for the Semantic Web | 长期可解析 URI 的设计原则 | 核心机制、工程实践 |
| URIs for W3C Namespaces | 命名空间设计的原则说明 | 命名空间边界、版本策略 |
| RDF 1.1 Concepts and Abstract Syntax | RDF 中 IRI/资源标识的概念基础 | 整体语义背景 |

1. [RFC 3986: Uniform Resource Identifier (URI): Generic Syntax](https://www.rfc-editor.org/rfc/rfc3986)
2. [Cool URIs for the Semantic Web](https://www.w3.org/TR/cooluris/)
3. [URIs for W3C Namespaces](https://www.w3.org/guide/editor/namespaces)
4. [RDF 1.1 Concepts and Abstract Syntax](https://www.w3.org/TR/rdf11-concepts/)
