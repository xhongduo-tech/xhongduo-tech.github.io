## 核心结论

RDFS 的类层次继承机制，本质上是一套“轻量推理规则”。轻量推理，意思是规则不多、实现成本低，但能把少量显式三元组扩展成一批可直接使用的新事实。

它最核心的四个部件是：

- `rdfs:subClassOf`：子类关系。白话说，某类的实例一定也是它父类的实例。
- `rdfs:subPropertyOf`：子属性关系。白话说，某个更具体的关系成立时，对应的更一般关系也成立。
- `rdfs:domain`：定义属性主语会被推成什么类型。白话说，只要某个属性被用在主语上，主语就会被自动补一个类型。
- `rdfs:range`：定义属性宾语会被推成什么类型。白话说，只要某个属性被用在宾语上，宾语就会被自动补一个类型。

这四类规则联合起来，可以形成闭包。闭包，意思是规则反复应用，直到再也推不出新三元组为止。

对工程系统来说，RDFS 往往已经够用。因为很多业务需要的不是复杂逻辑证明，而是这几类基础能力：

| 工程目标 | RDFS 是否擅长 | 典型做法 |
| --- | --- | --- |
| 类型自动补全 | 是 | `teaches domain Professor` 推出教师类型 |
| 父类继承传播 | 是 | `Professor ⊑ Person ⊑ Agent` 连续传播 |
| 父属性传播 | 是 | `hasManager ⊑ hasSupervisor` 自动补全 |
| 数据合法性校验 | 否 | 交给 SHACL 或应用层 |
| 基数、互斥、否定 | 否 | 交给 OWL 或应用层 |

最小玩具例子：

已知：

- `Professor rdfs:subClassOf Person`
- `Person rdfs:subClassOf Agent`
- `teaches rdfs:domain Professor`
- `Alice teaches CS101`

可以推出：

- `Alice rdf:type Professor`
- `Alice rdf:type Person`
- `Alice rdf:type Agent`

这里最容易误解的一点是：`domain` 和 `range` 是“蕴含规则”，不是“校验规则”。蕴含规则，意思是它告诉系统“可以推什么”，而不是“只允许什么”。

---

## 问题定义与边界

RDFS 解决的问题，不是把 RDF 图变成一个强约束数据库，而是在不引入复杂逻辑的前提下，给图增加基础语义继承能力。

更具体地说，RDFS 主要负责三件事：

1. 表达类层次。
2. 表达属性层次。
3. 根据属性的使用方式补全实体类型。

统一记号如下：

- `C1 ⊑ C2` 表示 `C1 rdfs:subClassOf C2`
- `P1 ⊑p P2` 表示 `P1 rdfs:subPropertyOf P2`
- `x : C` 表示 `x rdf:type C`

于是，RDFS 的任务可以理解为：已知一些显式事实和模式定义，系统自动推导出更多隐含事实。

例如：

`teaches rdfs:domain Professor`

这句话不表示“只有 Professor 才能出现在 teaches 的主语位置”。它真正表示的是：

只要图中出现 `x teaches y`，系统就可以推出 `x rdf:type Professor`。

这就是 RDFS 的边界关键点：它偏向“补全”，不偏向“限制”。

下面这个对比很重要：

| 维度 | RDFS | OWL | SHACL | 应用层校验 |
| --- | --- | --- | --- | --- |
| 主要目标 | 轻量语义补全 | 强表达推理 | 形状约束校验 | 业务规则校验 |
| 是否擅长继承推导 | 是 | 是 | 否 | 一般靠手写 |
| 是否擅长合法性判断 | 弱 | 部分可表达 | 强 | 强 |
| 推理成本 | 低 | 中到高 | 通常不做推理 | 取决于实现 |
| 工程落地难度 | 低 | 中到高 | 中 | 中 |

所以，如果你的目标是：

- 自动让子类继承父类
- 自动让子属性继承父属性
- 自动根据属性 usage 补类型

那么 RDFS 很合适。

但如果你的目标是：

- “一个员工最多只能有一个直属经理”
- “学生和老师互斥”
- “这个字段必须存在，否则报错”

那就不是 RDFS 的强项。

---

## 核心机制与推导

RDFS 类层次继承的核心，可以抽象成 5 条最常用规则。

1. 类层次传递性：

$$
C_1 \sqsubseteq C_2 \land C_2 \sqsubseteq C_3 \Rightarrow C_1 \sqsubseteq C_3
$$

2. 类实例继承：

$$
C_1 \sqsubseteq C_2 \land x : C_1 \Rightarrow x : C_2
$$

3. 属性继承：

$$
P_1 \sqsubseteq_p P_2 \land x\ P_1\ y \Rightarrow x\ P_2\ y
$$

4. 域推断：

$$
P\ domain\ C \land x\ P\ y \Rightarrow x : C
$$

5. 值域推断：

$$
P\ range\ C \land x\ P\ y \Rightarrow y : C
$$

这几条规则单看都不复杂，真正有价值的是它们会串起来。

看一个玩具例子。

已知：

- `Professor ⊑ Person`
- `Person ⊑ Agent`
- `teaches domain Professor`
- `teaches range Course`
- `Alice teaches CS101`

推导过程如下：

| 步骤 | 使用规则 | 新三元组 |
| --- | --- | --- |
| 1 | `domain` | `Alice rdf:type Professor` |
| 2 | 子类继承 | `Alice rdf:type Person` |
| 3 | 子类继承 | `Alice rdf:type Agent` |
| 4 | `range` | `CS101 rdf:type Course` |

这个过程说明，一条事实 `Alice teaches CS101`，并不只是一个业务记录。只要词汇表里定义了类层次和属性语义，它就会触发一串类型传播。

再看属性层次。

如果有：

- `teaches ⊑p interactsWith`
- `Alice teaches CS101`

那么可推出：

- `Alice interactsWith CS101`

属性继承常用于统一查询口径。比如底层数据里有很多具体关系，查询层只查父属性即可。

真实工程例子可以用企业知识图谱来理解。

假设有如下模式：

- `Manager ⊑ Employee`
- `FullTimeEmployee ⊑ Employee`
- `worksIn domain Employee`
- `worksIn range Department`
- `hasManager ⊑p hasSupervisor`

数据里只显式写：

- `Alice worksIn PlatformTeam`
- `Bob hasManager Alice`

那么系统可以自动补出：

- `Alice rdf:type Employee`
- `PlatformTeam rdf:type Department`
- `Bob hasSupervisor Alice`

如果再有 `Employee ⊑ Person`，还会继续推出 `Alice rdf:type Person`。

这里的重点不是“推理多高级”，而是“应用不必重复实现这些继承逻辑”。搜索、筛选、报表、权限和推荐，都可以直接消费推理后的结果。

从实现角度看，RDFS 闭包就是“重复应用规则直到收敛”。收敛，意思是本轮推导不再产生任何新三元组。

流程可以写成：

| 阶段 | 输入 | 操作 | 输出 |
| --- | --- | --- | --- |
| 初始化 | 显式三元组 | 放入集合 | 初始事实集 |
| 规则扫描 | 当前事实集 | 套用五类规则 | 候选新事实 |
| 去重 | 候选新事实 | 已存在则丢弃 | 真正新增事实 |
| 迭代 | 新增事实非空 | 继续扫描 | 闭包 |
| 结束 | 无新增事实 | 停止 | 最终闭包 |

因此，RDFS 的类层次继承不是一次性查表，而是一个迭代推导过程。

---

## 代码实现

工程里常见的实现思路很直接：用一个集合保存全部三元组，循环扫描规则，发现新事实就加入集合，直到不再增长。

下面是一个可运行的 Python 玩具实现，只覆盖本文讨论的核心规则：`subClassOf`、`subPropertyOf`、`domain`、`range` 与 `rdf:type` 传播。

```python
from collections import defaultdict

RDF_TYPE = "rdf:type"
SUBCLASS = "rdfs:subClassOf"
SUBPROPERTY = "rdfs:subPropertyOf"
DOMAIN = "rdfs:domain"
RANGE = "rdfs:range"

def rdfs_closure(triples):
    triples = set(triples)

    changed = True
    while changed:
        changed = False
        new_triples = set()

        subclass_edges = {(s, o) for s, p, o in triples if p == SUBCLASS}
        subproperty_edges = {(s, o) for s, p, o in triples if p == SUBPROPERTY}
        domain_map = {(s, o) for s, p, o in triples if p == DOMAIN}
        range_map = {(s, o) for s, p, o in triples if p == RANGE}
        type_facts = {(s, o) for s, p, o in triples if p == RDF_TYPE}
        factual_triples = {(s, p, o) for s, p, o in triples}

        # subClassOf transitivity
        for c1, c2 in subclass_edges:
            for c3_src, c3 in subclass_edges:
                if c2 == c3_src:
                    new_triples.add((c1, SUBCLASS, c3))

        # subPropertyOf transitivity
        for p1, p2 in subproperty_edges:
            for p3_src, p3 in subproperty_edges:
                if p2 == p3_src:
                    new_triples.add((p1, SUBPROPERTY, p3))

        # type inheritance
        for x, c1 in type_facts:
            for child, parent in subclass_edges:
                if c1 == child:
                    new_triples.add((x, RDF_TYPE, parent))

        # property inheritance
        for x, p1, y in factual_triples:
            for child_p, parent_p in subproperty_edges:
                if p1 == child_p:
                    new_triples.add((x, parent_p, y))

        # domain inference
        for x, p, y in factual_triples:
            for dp, c in domain_map:
                if p == dp:
                    new_triples.add((x, RDF_TYPE, c))

        # range inference
        for x, p, y in factual_triples:
            for rp, c in range_map:
                if p == rp:
                    new_triples.add((y, RDF_TYPE, c))

        before = len(triples)
        triples |= new_triples
        changed = len(triples) > before

    return triples


triples = {
    ("Professor", SUBCLASS, "Person"),
    ("Person", SUBCLASS, "Agent"),
    ("teaches", DOMAIN, "Professor"),
    ("teaches", RANGE, "Course"),
    ("teaches", SUBPROPERTY, "interactsWith"),
    ("Alice", "teaches", "CS101"),
}

closure = rdfs_closure(triples)

assert ("Alice", RDF_TYPE, "Professor") in closure
assert ("Alice", RDF_TYPE, "Person") in closure
assert ("Alice", RDF_TYPE, "Agent") in closure
assert ("CS101", RDF_TYPE, "Course") in closure
assert ("Alice", "interactsWith", "CS101") in closure
assert ("Professor", SUBCLASS, "Agent") in closure

print("closure size:", len(closure))
```

这个实现有三个工程特点。

第一，它是“物化闭包”思路。物化，意思是把推出来的事实真的存下来。优点是查询快，缺点是存储会变大。

第二，它用 `set` 去重。因为闭包推理一定会反复命中旧事实，不去重就会死循环或重复计算。

第三，它是全量扫描。玩具实现可以这样写，但数据量上来后，通常要改成增量推理，也就是“只让新事实触发相关规则”。

如果换成真实工程，常见有两种落地方式：

| 方式 | 含义 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 离线物化 | 入库后先把闭包算完 | 查询快、业务简单 | 占存储、更新成本高 |
| 查询时推理 | 查询阶段按需展开 | 存储省、更新灵活 | 查询复杂、性能波动 |

小型知识图谱、元数据目录、权限标签系统，常用离线物化。数据变化频繁、规则不稳定的系统，更适合按需推理或局部增量物化。

---

## 工程权衡与常见坑

RDFS 在工程中最大的风险，不是推不出来，而是推太多，尤其是 `domain` 和 `range` 被错误理解的时候。

最常见的坑如下：

| 常见坑 | 产生原因 | 影响 | 规避方法 |
| --- | --- | --- | --- |
| 把 `domain/range` 当校验规则 | 混淆“蕴含”和“限制” | 大量实体被误推类型 | 把校验交给 SHACL 或应用层 |
| 公共属性挂了过窄 `domain` | 词汇表设计过度乐观 | 类型污染，查询结果失真 | 公共属性尽量挂宽域或不挂 |
| 一个属性定义多个 `domain` | 误以为是“任选其一” | 实际语义接近交集，越推越窄 | 多域前先确认是否真要同时成立 |
| 全量物化前未做样本测试 | 没看闭包规模 | 三元组爆炸，报表和检索异常 | 先在小样本上跑闭包 |
| 把层次设计得过深 | 过度抽象 | 推理链长、调试难 | 控制类层次深度，保留业务可读性 |

先看一个典型误用。

有人写了：

- `name rdfs:domain Person`

原意可能只是“人通常有名字”。但在 RDFS 里，这条规则的真实效果是：

只要出现 `x name y`，就能推出 `x rdf:type Person`。

于是，组织、产品、课程、城市，只要也用了 `name` 属性，就都可能被推成 `Person`。这会直接污染搜索索引、用户画像、权限标签和统计口径。

再看多个 `domain` 的问题。

如果同时声明：

- `teaches rdfs:domain Professor`
- `teaches rdfs:domain Employee`

那么出现 `Alice teaches CS101` 时，系统会同时推出：

- `Alice rdf:type Professor`
- `Alice rdf:type Employee`

这更接近“两个类型都成立”，而不是“二选一”。

真实工程例子里，这类错误很常见。比如在企业主数据平台中，把 `owner` 的 `domain` 定成 `Employee`，但后来设备、项目、服务实例也用了 `owner`。结果所有这些资源都被误推成 `Employee`，下游权限系统再按类型发放默认权限，就会引入非常隐蔽的安全风险。

因此，RDFS 词汇表设计要遵守一个务实原则：

“越公共的属性，越不要轻易挂窄域；越稳定的继承关系，越适合放进 RDFS。”

---

## 替代方案与适用边界

RDFS 适合做“轻量语义补全”，不适合做“强约束验证”。

可以把它理解成自动补全器。自动补全器，意思是系统看到一条事实后，会帮你补出它隐含的父类、父属性和类型信息。它不会主动替你判断“这条数据是不是违法”。

下面是一个实用选型表：

| 方案 | 表达能力 | 推理成本 | 是否适合校验 | 是否适合工程落地 | 典型适用场景 |
| --- | --- | --- | --- | --- | --- |
| RDFS | 低到中 | 低 | 弱 | 很适合 | 类型补全、层次继承、元数据统一 |
| OWL | 高 | 中到高 | 部分可表达 | 视团队能力而定 | 复杂本体、等价类、互斥、基数 |
| SHACL | 中 | 通常不强调推理 | 强 | 适合治理场景 | 入库校验、质量规则、字段约束 |
| 应用层校验 | 任意 | 任意 | 强 | 最灵活 | 强业务逻辑、流程控制、权限规则 |

几个典型判断标准：

如果需求是：

- `Manager ⊑ Employee`
- `hasManager ⊑p hasSupervisor`
- `worksIn domain Employee`

这种“继承和补全”需求，用 RDFS 就够了。

如果需求是：

- “一个员工最多只有一个直属经理”
- “Student 和 Professor 互斥”
- “每个订单必须有且仅有一个付款记录”

这种“限制和判错”需求，RDFS 不够，应转向 OWL、SHACL 或应用层规则。

所以更准确的说法不是“RDFS 弱，所以不值得用”，而是“RDFS 刚好覆盖了很多工程里最常见的那部分语义需求”。只要你清楚它负责补全，不负责验错，它就非常实用。

---

## 参考资料

1. [RDF 1.2 Schema](https://www.w3.org/TR/rdf12-schema/)
2. [RDF 1.2 Semantics](https://www.w3.org/TR/rdf12-semantics/)
3. [RDF Semantics](https://www.w3.org/TR/rdf-mt/)
4. [RDF Schema 1.1](https://www.w3.org/TR/rdf-schema/)
