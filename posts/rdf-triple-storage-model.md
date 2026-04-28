## 核心结论

RDF 三元组存储模型的核心目标，不是把 `(subject, predicate, object)` 这三个字段放进一张表，而是让 SPARQL 查询里的不同三元组模式都能以低代价命中索引，并把后续 `join` 成本压到可控范围。`join` 可以白话理解为“把两批部分结果按共同变量拼起来”。

对工程实现来说，性能主要由四件事共同决定：

| 关键因素 | 作用 | 做不好会发生什么 |
| --- | --- | --- |
| 多排列索引 | 让不同绑定模式直接命中合适顺序 | 退化成大范围扫描 |
| 字典编码 | 把字符串项映射成整数 ID | 比较、排序、压缩都变慢 |
| 选择性估计 | 判断哪个模式更该先查 | 中间结果过大 |
| join 顺序 | 控制多跳查询的扩张路径 | 自连接爆炸 |

最短的判断标准是：如果一个存储设计只能回答“能不能查到”，而不能回答“不同查询模式会走哪种低代价路径”，那它还不算合格的 RDF 查询存储模型。

一个新手可感知的对比是同一个查询：

```sparql
SELECT ?x ?y WHERE {
  ?x type Person .
  ?x knows ?y .
}
```

如果底层只有单一 `(s,p,o)` 表，系统往往要先扫出所有 `type=Person`，再扫出所有 `knows`，最后在 `?x` 上做连接。如果系统同时维护 `POS` 和 `SPO` 之类的索引，那么第一个模式可以直接用 `(predicate, object)` 定位到所有 `Person`，第二个模式可以按 `subject` 快速扩展，代价会明显更低。

单表扫描和多索引命中的差异可以先看成下面这张表：

| 方案 | 第一步 | 第二步 | 结果 |
| --- | --- | --- | --- |
| 单一三元组表 | 扫描大量记录找 `type Person` | 再扫大量记录找 `knows` | join 成本高 |
| 多排列索引 | 用 `POS` 直接取 `(?x, type, Person)` | 用 `SPO` 或 `PSO` 扩展 `?x knows ?y` | 中间结果小 |

公式上，若数据集为 $D$，三元组模式为 $tp=(s,p,o)$，其选择率定义为：

$$
sel(tp)=\frac{|match(tp)|}{|D|}
$$

选择率越小，说明这个模式越“挑数据”，越适合作为查询起点。

---

## 问题定义与边界

RDF 是一种用三元组表示事实的数据模型。白话说，它把“谁和谁之间有什么关系”拆成最基本的三段式记录。`subject` 是主语，表示被描述的对象；`predicate` 是谓词，表示关系或属性；`object` 是宾语，表示关系指向的值或另一个实体。

SPARQL 查询不是在“查某行记录”，而是在“找所有满足某个三元组模式的边”。所谓 `triple pattern`，就是允许某些位置是常量，某些位置是变量的三元组模板。记作：

$$
tp=(s,p,o)
$$

其中 `s/p/o` 可以是常量，也可以是变量。`match(tp)` 表示图中所有能匹配该模式的三元组集合。

例如：

- `?x type Person`：找所有类型为 `Person` 的实体
- `Alice ?p ?o`：找所有从 `Alice` 出发的属性或关系
- `?s knows Bob`：找所有认识 `Bob` 的实体

不同绑定模式的含义可以直接列出来：

| 模式 | 已知部分 | 查询含义 | 常见索引起点 |
| --- | --- | --- | --- |
| `?s p o` | `p,o` 已知 | 谁满足某个固定关系和值 | `POS` |
| `s ?p ?o` | `s` 已知 | 某个实体有哪些边 | `SPO` |
| `?s ?p o` | `o` 已知 | 哪些边指向这个对象 | `OSP` 或 `OPS` |
| `s p ?o` | `s,p` 已知 | 某实体在某属性上的值 | `SPO` |
| `?s p ?o` | `p` 已知 | 所有某类关系的边 | `PSO` 或垂直分表 |
| `?s ?p ?o` | 全未知 | 全图扫描 | 任意索引都不便宜 |

这里要明确边界。本文讨论的是“面向查询优化的 RDF 存储设计”，不讨论下面三类问题：

- 不展开 RDF 语义推理，例如 RDFS/OWL 推导规则
- 不展开分布式一致性、分片复制与容灾
- 不展开全文检索、权限控制和业务 API 设计

一个玩具例子可以帮助区分“完全绑定”和“部分绑定”：

假设图里有三条边：

- `(Alice, type, Person)`
- `(Alice, knows, Bob)`
- `(Bob, type, Person)`

那么：

- `Alice knows Bob` 是完全绑定，答案只有“是否存在”
- `Alice knows ?x` 是部分绑定，答案可能有多个对象
- `?x type Person` 是更宽的模式，答案是一组实体

这正是 RDF 存储和普通键值存储不同的地方。它不是只优化“按主键取一条记录”，而是优化“按模式批量找边，并继续连接”。

---

## 核心机制与推导

为什么单一 `(s,p,o)` 表不够用？原因很直接：不同三元组模式的最佳访问起点不同，而单一排序顺序只能高效支持少数模式。

如果表按 `SPO` 排序，那么下面两类查询代价差异会很大：

- `s p ?o`：很好查，因为 `s,p` 是前缀
- `?s p o`：不好查，因为 `s` 不知道，前缀无法利用

因此工程上会维护多种排列索引。常见含义如下：

| 已知条件 | 适合索引 | 原因 |
| --- | --- | --- |
| `s` 已知 | `SPO` | 直接按主语定位 |
| `s,p` 已知 | `SPO` | 前缀最短、范围最小 |
| `p,o` 已知 | `POS` | 先按谓词再按对象收缩 |
| `o,s` 已知 | `OSP` | 先按对象定位，再看主语 |
| `p` 已知 | `PSO` | 扫某个谓词的全部边 |
| `o` 已知 | `OSP/OPS` | 反向引用查询更快 |

更完整的六种排列索引是：

| 索引 | 优势模式 |
| --- | --- |
| `SPO` | `s p ?o`、`s ?p ?o` |
| `SOP` | `s o ?p` |
| `PSO` | `p s ?o`、`?s p ?o` |
| `POS` | `p o ?s`、`?s p o` |
| `OSP` | `o s ?p`、`?s ?p o` |
| `OPS` | `o p ?s` |

查询优化器的目标，是先选最有选择性的模式，再把结果逐步扩展。选择率仍然用：

$$
sel(tp)=\frac{|match(tp)|}{|D|}
$$

估计基数时，通常先估单个模式命中数：

$$
card(tp)=|match(tp)|
$$

再估连接结果：

$$
card(tp_1 \bowtie tp_2)
$$

实际系统不会直接精确计算，而是借助谓词频次、主语去重数、对象去重数、联合分布统计做近似估计。白话说，就是先猜“这一步会吐出多少行”，再决定先做谁、后做谁。

看一个最小推导例子。数据集：

- `(Alice, type, Person)`
- `(Alice, knows, Bob)`
- `(Bob, type, Person)`
- `(Bob, name, "Bob")`

查询：

```sparql
SELECT ?x ?y WHERE {
  ?x type Person .
  ?x knows ?y .
}
```

执行过程可以写成一条简单机制链：

`triple pattern -> index scan -> intermediate result -> join -> final result`

具体到这组数据：

1. 对 `?x type Person`，因为 `p=type, o=Person` 已知，走 `POS`
2. 命中结果：`?x ∈ {Alice, Bob}`
3. 对 `?x knows ?y`，若按谓词扫描，可得到 `Alice -> Bob`
4. 在变量 `?x` 上 join
5. 只保留 `Alice`，最终结果为 `(?x=Alice, ?y=Bob)`

这个例子说明一件基础但关键的事：RDF 查询性能不是由“单条边怎么存”决定，而是由“多步模式匹配的扩张顺序”决定。只要 join 顺序选错，多跳查询就会迅速失控。

真实工程里最常见的退化场景，是三到五跳关系查询。例如商品知识图谱中查询“某商品所属品牌的上级集团的其他热门品类”，或论文图谱中查询“某作者合作者的机构及机构下相关论文”。这些模式本质上都是连续的自连接。如果每一步都产生大中间结果，后面就不是查图，而是在搬运海量候选集。

---

## 代码实现

实现层的重点不是 SPARQL 语法本身，而是数据结构。常见流程是：

| 实现步骤 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- |
| 字典编码 | 字符串 RDF 项 | 整数 ID | 降低比较与存储成本 |
| 三元组编码 | `(s,p,o)` | `(sid,pid,oid)` | 统一内部表示 |
| 索引构建 | ID 三元组数组 | 多排列有序索引 | 支持不同绑定模式 |
| 查询分解 | SPARQL | triple patterns | 把查询转成可执行单元 |
| 索引选择 | 模式 + 统计信息 | 扫描计划 | 控制扫描范围 |
| join 执行 | 局部结果集 | 最终绑定结果 | 拼出查询答案 |

字典编码是第一步。它把 `Alice`、`Person`、`knows` 这类字符串变成整数。白话说，就是先把“难比、难压缩的文本”变成“易比、易排的编号”。

下面给一个可运行的简化 Python 版本，演示字典编码、多索引构建和一个两模式查询。它不是完整 SPARQL 引擎，但核心思路是一样的。

```python
from bisect import bisect_left, bisect_right

class RDFStore:
    def __init__(self):
        self.term_to_id = {}
        self.id_to_term = {}
        self.next_id = 1
        self.triples = []
        self.spo = []
        self.pos = []
        self.osp = []

    def encode(self, term):
        if term not in self.term_to_id:
            tid = self.next_id
            self.next_id += 1
            self.term_to_id[term] = tid
            self.id_to_term[tid] = term
        return self.term_to_id[term]

    def decode(self, tid):
        return self.id_to_term[tid]

    def add(self, s, p, o):
        self.triples.append((self.encode(s), self.encode(p), self.encode(o)))

    def build_indexes(self):
        self.spo = sorted(self.triples, key=lambda t: (t[0], t[1], t[2]))
        self.pos = sorted(self.triples, key=lambda t: (t[1], t[2], t[0]))
        self.osp = sorted(self.triples, key=lambda t: (t[2], t[0], t[1]))

    def lookup_pos(self, p, o):
        pid, oid = self.encode(p), self.encode(o)
        data = self.pos
        left = bisect_left(data, (pid, oid, -1))
        right = bisect_right(data, (pid, oid, 10**18))
        return data[left:right]

    def lookup_spo_subject(self, s, p=None):
        sid = self.encode(s)
        pid = self.encode(p) if p is not None else None
        result = []
        for ts, tp, to in self.spo:
            if ts != sid:
                continue
            if pid is not None and tp != pid:
                continue
            result.append((ts, tp, to))
        return result

    def query_person_knows(self):
        persons = self.lookup_pos("type", "Person")
        person_ids = {s for _, _, s in [(p, o, s) for (p, o, s) in persons]}
        answers = []
        for sid in person_ids:
            s = self.decode(sid)
            for ts, tp, to in self.lookup_spo_subject(s, "knows"):
                answers.append((self.decode(ts), self.decode(to)))
        return sorted(answers)

store = RDFStore()
store.add("Alice", "type", "Person")
store.add("Alice", "knows", "Bob")
store.add("Bob", "type", "Person")
store.add("Bob", "name", "Bob")
store.build_indexes()

result = store.query_person_knows()
assert result == [("Alice", "Bob")]
print(result)
```

这个实现里有四个关键点：

- `dictionary`：`term_to_id` 和 `id_to_term`
- `id triples`：内部只存整数三元组
- `SPO index / POS index / OSP index`：同一批数据按不同顺序排序
- 查询先拆模式，再按模式走不同索引

如果把逻辑抽成伪代码，索引构建大致是这样：

```python
def build(triples):
    encoded = [(encode(s), encode(p), encode(o)) for s, p, o in triples]
    spo = sort(encoded, key=(s, p, o))
    pos = sort(encoded, key=(p, o, s))
    osp = sort(encoded, key=(o, s, p))
    return spo, pos, osp
```

查询执行的伪代码大致是这样：

```python
def execute(query):
    patterns = parse(query)
    ordered = choose_by_selectivity(patterns)
    partial = lookup(ordered[0])
    for tp in ordered[1:]:
        rows = lookup(tp)
        partial = join(partial, rows)
    return partial
```

如果某些谓词访问频繁且结构稳定，还会增加 `predicate partitions`，也就是按谓词垂直分表。例如把 `brand(item, brand_id)`、`category(item, category_id)` 作为独立两列表维护。这对详情页类请求很常见，因为请求通常是“给定实体，拿若干固定属性”。

真实工程例子可以看商品知识图谱详情页。一次请求可能同时拉取：

- `item brand ?b`
- `item category ?c`
- `item relatedTo ?x`
- `?x brand ?bx`

其中 `brand` 和 `category` 常常是高频低基数字段，适合单独按谓词存；`relatedTo` 可能更稀疏，适合走通用三元组索引。也就是说，一个成熟实现往往不是“只选一种物理模型”，而是“通用索引 + 重点谓词特化”并存。

---

## 工程权衡与常见坑

RDF 存储没有万能索引。索引越多，理论可覆盖的查询模式越全，但写入代价、存储体积、压缩复杂度、统计信息维护成本都会上升。因此设计时要明确：你的系统是读多写少，还是写多读多；是复杂图遍历多，还是属性读取多。

先看两个基础权衡：

| 选择 | 优点 | 代价 |
| --- | --- | --- |
| 少索引 | 写入简单、空间小 | 很多查询退化 |
| 多索引 | 查询覆盖全面 | 写放大、空间放大 |
| 字符串直接存 | 实现直观 | 排序、比较、缓存都差 |
| 整数 ID 存 | 高效、可压缩 | 需要维护字典 |

再看“索引数量 vs 写入代价 vs 查询代价”的关系：

| 策略 | 写入代价 | 查询代价 | 适合场景 |
| --- | --- | --- | --- |
| 单一 `SPO` | 低 | 高 | 很少反向查询、很少 join |
| 三索引 `SPO/POS/OSP` | 中 | 中低 | 通用中小型系统 |
| 六索引全覆盖 | 高 | 低 | 读多写少、查询复杂 |
| 垂直分表 + 通用索引 | 中高 | 对热点谓词低 | 属性型查询明显的业务 |

常见坑基本都和“中间结果失控”有关：

| 常见坑 | 后果 | 规避方式 |
| --- | --- | --- |
| 只建单一索引 | 大量模式无法高效命中 | 至少覆盖高频绑定模式 |
| 盲目堆索引不压缩 | 空间和缓存压力过大 | 先字典编码，再做块压缩或列式压缩 |
| join 顺序靠经验猜 | 先做大表 join，结果爆炸 | 用统计信息估计选择率 |
| 忽略谓词倾斜 | 热点谓词成为瓶颈 | 对高频低基数谓词做特化 |
| URI 直接存字符串 | CPU 与内存开销都高 | 统一映射整数 ID |

基数估计误差为什么危险？因为它直接影响 join 顺序。假设优化器误以为某个模式很“窄”，实际上它很“宽”，那么它会把这个模式排到前面，先产生巨大结果集，后续所有 join 都要在这个大结果上继续做。即使单步扫描不慢，总查询也会变得很慢。

可以把这个影响理解成：

$$
cost(plan) \approx \sum_i scan_i + \sum_j join_j
$$

而 `join_j` 的真实代价又高度依赖中间结果大小。中间结果一旦被错误放大，后面的成本通常不是线性增加，而是连锁放大。

一个典型错误做法是只维护 `SPO`，认为“反正都能查”。这在功能上成立，在性能上通常不成立。正确做法不是盲目上六索引，而是先看业务模式：哪些谓词最热，哪些绑定最常见，哪些查询经常多跳，再决定最小可行索引集合。

---

## 替代方案与适用边界

不同 RDF 存储方案解决的是不同问题，不能混成一类讨论。本文关注的是“查询性能优先”的存储设计，不是“最小存储体积优先”的表示法。

常见方案可以放在一张表里比较：

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 单一三元组表 | 实现最简单 | 查询易退化、自连接重 | 演示系统、很小数据集 |
| 六方向排列索引 | 覆盖绑定模式全面、join 友好 | 空间和写入代价高 | 低更新、高查询、复杂 SPARQL |
| 垂直分表 | 热点谓词读取快，列式压缩友好 | 谓词多时表数量膨胀，复杂模式仍需 join | 属性查询多、字段相对稳定 |
| HDT / 压缩交换格式 | 存储和传输成本低 | 更新能力弱，通常不以在线查询优化见长 | 发布、交换、静态分发 |
| RDF-3X 类列式多索引 | 高压缩、高性能查询 | 实现复杂 | 读密集分析型场景 |
| Hexastore 类六索引 | 模式命中强，查询直观 | 索引空间较大 | 图模式复杂、读远多于写 |

几类方案的边界要分清：

- 六方向排列索引，重点解决“不同绑定模式都能直接走索引”
- 垂直分表，重点解决“按谓词访问时更像列存系统”
- HDT，重点解决“压缩与交换”
- 单一三元组表，更多是逻辑起点，不是高性能终点

如果你的系统主要是详情页类读取，例如“查某商品的品牌、类目、几个固定关系”，按谓词垂直分表通常更稳。因为大部分请求本质上是固定字段读取，不需要通用图模式的完全自由度。

如果你的系统查询模式复杂，经常出现多跳路径、变量多位置绑定、分析型 SPARQL，那么多排列索引或 RDF-3X 这类设计更有优势。因为它们的目标就是减少不同模式之间的连接代价。

如果你的目标是把 RDF 数据高压缩发布给别人下载，或者本地离线分发，HDT 更合适。它不是典型的“在线更新友好型事务存储”。

一个简单选型表可以直接用：

| 条件 | 更合适的方案 |
| --- | --- |
| 低更新、高查询、多种绑定模式 | 多排列索引 |
| 高频谓词、字段固定、详情页读取 | 垂直分表 |
| 存储和交换优先 | HDT |
| 只是验证概念或数据量很小 | 单一三元组表 |

所以，RDF 三元组存储模型真正要回答的问题不是“把三元组放哪里”，而是“面对你的查询模式，最贵的扫描和最贵的 join 在哪里，怎么提前绕开它”。

---

## 参考资料

下表说明每篇资料主要对应本文哪个部分：

| 资料名称 | 主要贡献 | 对应本文章节 |
| --- | --- | --- |
| SPARQL 1.1 标准 | 定义查询语义与 triple pattern | 问题定义与边界 |
| RDF-3X | 多索引、压缩、RDF 查询引擎设计 | 核心机制与代码实现 |
| Hexastore | 六重索引思路 | 核心机制与替代方案 |
| Vertical Partitioning | 按谓词分表的经典做法 | 替代方案与工程权衡 |
| HDT | RDF 压缩与交换格式 | 替代方案与适用边界 |
| SPARQL join ordering | 查询优化与 join 顺序 | 核心机制与工程权衡 |

1. [SPARQL 1.1 Query Language](https://www.w3.org/TR/sparql11-query/)
2. [RDF-3X: a RISC-style Engine for RDF](https://www.vldb.org/pvldb/vol1/1453927.pdf)
3. [Hexastore: Sextuple Indexing for Semantic Web Data Management](https://www.vldb.org/pvldb/vol1/1453965.pdf)
4. [Scalable Semantic Web Data Management Using Vertical Partitioning](https://www.cs.umd.edu/~abadi/papers/abadirdf.pdf)
5. [Binary RDF representation for publication and exchange (HDT)](https://www.w3.org/submissions/HDT/)
6. [Exploiting the query structure for efficient join ordering in SPARQL queries](https://portal.fis.tum.de/en/publications/exploiting-the-query-structure-for-efficient-join-ordering-in-spa)
