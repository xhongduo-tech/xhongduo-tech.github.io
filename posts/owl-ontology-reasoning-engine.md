## 核心结论

OWL 本体推理的核心，不是“把图里没写出来的边补全完”，而是在 **Direct Semantics** 下判断一个本体是否存在满足全部约束的解释。解释，白话说，就是“一个可能的世界以及其中各个类、关系、个体各自代表什么”。

形式上，给定本体 $O$，一致性的定义是：

$$
O\ \text{一致} \iff \exists I,\ I \models O
$$

这句话的意思很直接：只要存在一个解释 $I$，能让本体里的所有公理都成立，这个本体就没有逻辑冲突。

OWL 推理在工程里通常回答三类问题：

| 问题类型 | 问的是啥 | 输出是什么 |
|---|---|---|
| 一致性检测 | 本体有没有内在冲突 | `yes / no` |
| 分类推理 | 哪些类必然是哪些类的子类 | 子类层次 |
| 实例归属 | 某个个体必然属于哪些类 | 断言结论 |

另外两个核心公式也最好先记住：

$$
O \models C \sqsubseteq D \iff \forall I,\ C^I \subseteq D^I
$$

$$
O \models a:C \iff \forall I,\ a^I \in C^I
$$

第一条说的是子类关系：在所有满足本体的解释里，$C$ 的实例集合都包含在 $D$ 里。第二条说的是实例归属：在所有满足本体的解释里，个体 $a$ 都属于类 $C$。

对新手最重要的一点是区分数据库和 OWL。数据库更像“表里有没有这条记录”；OWL 更像“有没有一个世界能让这些约束都成立”。前者偏闭世界，后者偏开放世界。

---

## 问题定义与边界

Direct Semantics 是 OWL 2 的直接模型论语义。模型论，白话说，就是不用“执行规则”的视角，而用“什么样的世界算满足这些句子”的视角来定义真值。

最小玩具例子：

- `Student ⊑ Person`
- `Alice : Student`

那么推理器可以推出：

- `Alice : Person`

这里不是去“查表看 Alice 有没有 Person 标签”，而是判断：在所有满足前两条陈述的解释里，`Alice` 是否必然属于 `Person`。如果答案是“必然”，那就是逻辑蕴含。

类可满足性也有一个标准定义：

$$
C\ \text{可满足} \iff \exists I,\ I \models O \land C^I \neq \varnothing
$$

意思是：存在一个满足本体的解释，并且类 $C$ 不是空集。比如某个类被定义成“既是人又不是人”，那它就是不可满足类。

OWL 推理的边界必须说清楚，否则很容易把它误解成图数据库查询器。

| 对比项 | 闭世界数据库 | OWL 开放世界 |
|---|---|---|
| 缺失事实 | 没写通常当不存在 | 没写不代表不存在 |
| 目标 | 查已存数据 | 判断逻辑蕴含 |
| 结果语义 | 当前记录集上的答案 | 所有模型中的必然结论 |

边界可以压缩成四句：

- 不处理“唯一答案”，而处理“是否必然成立”。
- 不承诺数据完备，缺失信息默认仍可能存在。
- 不等同 SQL 查询，查询只是使用推理结果的一种方式。
- 不等同普通业务规则，OWL 受形式语义约束。

一个常见误解是：如果系统里没写 `Bob hasChild Tom`，就能推出 Bob 没有孩子。OWL 里不能这样推。因为开放世界假设下，“没写”只表示“当前不知道”。

---

## 核心机制与推导

OWL 2 DL 的核心实现路线之一是 **Tableau**。Tableau，白话说，就是“试着一步步构造一个不矛盾的候选世界；如果所有尝试都失败，就说明不可满足”。

常见展开规则可以先记成这张表：

| 构造 | 含义 | Tableau 动作 |
|---|---|---|
| $C \sqcap D$ | 交，必须同时满足 | 同时加入 `C` 和 `D` |
| $C \sqcup D$ | 并，满足其一即可 | 分成两个分支 |
| $\exists R.C$ | 存在一个 `R` 后继落在 `C` 中 | 新建后继个体 |
| $\forall R.C$ | 所有 `R` 后继都在 `C` 中 | 向所有后继传播 `C` |
| `x:C` 与 `x:\neg C` | 同一对象同时满足正反条件 | `clash` 冲突 |

可以把简化流程理解成：

1. 初始化断言集  
2. 反复展开可展开的约束  
3. 遇到析取就分支搜索  
4. 遇到冲突就关闭该分支  
5. 只要有一个分支无冲突，则可满足；所有分支都冲突，则不可满足

### 玩具例子：从存在限制推出实例归属

给出定义：

- `Parent ≡ (≥ 1 hasChild.Person)`
- `hasChild(zhangSan, liSi)`
- `liSi : Person`

新手版推导链：

- `zhangSan` 有一个孩子 `liSi`
- `liSi` 属于 `Person`
- 所以 `zhangSan` 至少有一个 `hasChild` 关系连接到一个 `Person`
- 因此 `zhangSan` 满足 `(≥ 1 hasChild.Person)`
- 又因为 `Parent` 与该条件等价，所以 `zhangSan : Parent`

形式版可以写成：若存在某个个体 $y$，使得 $(zhangSan, y) \in hasChild^I$ 且 $y \in Person^I$，则 $zhangSan \in (\ge 1\ hasChild.Person)^I$，再由等价公理得到 $zhangSan \in Parent^I$。

冲突版本再加一条：

- `zhangSan : (≤ 0 hasChild.Person)`

这条表示 `zhangSan` 不能有任何一个 `hasChild` 后继属于 `Person`。但前面已经由 `hasChild(zhangSan, liSi)` 和 `liSi : Person` 证明至少有一个。于是同一个对象同时满足“至少 1 个”和“至多 0 个”，数值限制冲突，本体不可满足。

### 为什么能用构造模型来做分类

分类推理本质上是在问某个子类关系是否必然成立。常见做法不是直接证明 $C \sqsubseteq D$，而是转成可满足性测试：检查 $C \sqcap \neg D$ 是否可满足。

- 如果可满足，说明存在某个世界里，有对象属于 `C` 但不属于 `D`，那就不能推出 `C ⊑ D`
- 如果不可满足，说明任何满足本体的世界里，都不存在“属于 `C` 但不属于 `D`”的对象，因此 `C ⊑ D`

这就是很多 reasoner 在内部把“分类”“实例检查”都统一成可满足性问题的原因。

---

## 代码实现

实现一个最小推理器时，顺序最好是：先定内部表示，再定状态推进，再写冲突检测。否则代码会变成把概念名硬塞进 if-else。

一个简化模块划分如下：

| 模块 | 作用 |
|---|---|
| 解析层 | 把 OWL 表达式转成内部结构 |
| 断言层 | 维护 TBox 和 ABox |
| 规则层 | 实现 `⊓ / ⊔ / ∃ / ∀` 展开 |
| 冲突层 | 检查 `clash` 和数值限制矛盾 |
| 查询层 | 输出一致性、分类、实例结论 |

TBox，白话说，就是“类和类之间的模式约束”；ABox 是“具体个体及其事实”。

下面这段代码不是完整 OWL reasoner，而是一个可运行的玩具实现，演示如何用最小规则检查“Parent 定义”和“≤0 hasChild.Person”是否冲突：

```python
from collections import defaultdict

def infer_parent(facts):
    children = defaultdict(list)
    persons = set(facts["types"].get("Person", []))

    for s, o in facts["roles"].get("hasChild", []):
        children[s].append(o)

    inferred_parent = set()
    for s, objs in children.items():
        if any(o in persons for o in objs):
            inferred_parent.add(s)
    return inferred_parent

def check_max_zero_conflict(facts):
    inferred_parent = infer_parent(facts)
    max_zero = set(facts["restrictions"].get("max0_hasChild_Person", []))
    return len(inferred_parent & max_zero) > 0

facts_ok = {
    "types": {"Person": {"liSi"}},
    "roles": {"hasChild": {("zhangSan", "liSi")}},
    "restrictions": {"max0_hasChild_Person": set()},
}

facts_bad = {
    "types": {"Person": {"liSi"}},
    "roles": {"hasChild": {("zhangSan", "liSi")}},
    "restrictions": {"max0_hasChild_Person": {"zhangSan"}},
}

assert infer_parent(facts_ok) == {"zhangSan"}
assert check_max_zero_conflict(facts_ok) is False
assert check_max_zero_conflict(facts_bad) is True
```

这段代码体现了三个最基本的工程思想：

- 代码不是直接“算最终答案”，而是在维护一个候选世界。
- 规则展开本质是把隐含约束显式化。
- 冲突检测一旦命中，当前分支就不能作为模型存在。

如果再往上抽象，一个简化版伪代码可以写成：

```text
function check_satisfiable(ontology):
    branch = initialize_assertions(ontology)
    stack = [branch]

    while stack not empty:
        current = stack.pop()

        if detect_clash(current):
            continue

        rule = pick_expandable_rule(current)
        if rule is None:
            return SAT

        next_states = expand(current, rule)
        push all next_states into stack

    return UNSAT
```

真实工程里，内部状态通常不是“字符串列表”，而是几种结构组合：

- 概念树或 DAG：避免重复解析复杂类表达式
- 个体图：维护对象节点
- 角色边：维护 `R(a,b)` 关系
- 约束队列：记录尚未展开的断言

真实工程例子可以看医院知识图谱。假设有类 `Disease`、`Symptom`、`Exam`、`Procedure`，以及大量子类关系。这里的核心需求通常不是复杂否定推理，而是把新增术语自动归入正确层次，并把病历实例归入合适概念。此时推理器最重要的是“分类速度、增量更新能力、结果稳定性”，而不是支持全部高表达力构造。

---

## 工程权衡与常见坑

初学者最容易踩的坑不是语法，而是语义。

第一类坑是开放世界假设。

| 错误认知 | 正确理解 |
|---|---|
| 没写 `no fever`，所以病人发烧 | 没写只能表示未知 |
| 没写父节点，所以不是子类 | 可能只是尚未声明 |
| 没查到实例，所以类为空 | 可能数据不完备 |

第二类坑是复杂构造会迅速抬高推理代价。

| 常见构造 | 复杂度影响 | 工程建议 |
|---|---|---|
| `¬` 否定 | 需要处理互补类与冲突传播 | 能不用全局否定就不用 |
| 枚举 `oneOf` | 引入具体个体语义 | 只在强需求下使用 |
| 数值限制 | 需要计数与合并判断 | 大规模数据慎用 |
| inverse role | 传播方向更复杂 | 只为查询方便时不值得引入 |
| nominal | 把类绑定到具体个体 | 会增加推理负担 |
| `sameAs` | 等价类膨胀 | 严格控制粒度 |

医院知识图谱就是典型场景。疾病、症状、检查、手术等术语层次非常大，更新频繁，核心任务是高效分类。这类场景常选 `OWL 2 EL`，因为它牺牲了一部分表达力，换来更可控的多项式时间推理。像 SNOMED CT 这样的医疗本体就是经典方向。

实操时有几条规则很管用：

- 先定 profile，再定建模方式。
- 控制 `sameAs`，不要把实体对齐问题全塞给推理器。
- 先做离线分类，再做在线查询。
- 把“本体定义”和“业务数据”分层，不要混写。

---

## 替代方案与适用边界

不是所有知识图谱场景都该上完整 OWL 2 DL。选择标准通常看四个维度：表达能力、推理复杂度、扩展性、查询模式。

| 方案 | 优势 | 代价 | 适合场景 |
|---|---|---|---|
| OWL 2 EL | 分类高效，适合大本体 | 表达能力受限 | 医疗术语、本体层次维护 |
| OWL 2 QL | 适合数据库查询下推 | 复杂表达受限 | 关系库上的本体访问 |
| OWL 2 RL | 易做规则物化 | 语义能力不如 DL 完整 | 规则执行、批处理推导 |
| OWL 2 DL | 表达力最强 | 推理代价高 | 中小规模高精度建模 |
| 规则引擎 | 易控、易解释 | 缺少完整模型论语义 | 业务规则自动化 |
| 数据库/搜索 | 查询快、运维成熟 | 不提供 OWL 蕴含 | 事务、检索、统计 |

几条选择建议可以直接落地：

- 如果目标是大规模本体分类，优先 `OWL 2 EL`。
- 如果目标是把本体查询下推到 SQL，优先 `OWL 2 QL`。
- 如果目标是规则物化和批量补全，优先 `OWL 2 RL`。
- 如果确实需要高表达力约束且规模可控，再考虑 `OWL 2 DL`。

从实现路线看，也可以粗分为四类：

- Tableau 路线：经典、通用，适合高表达力 DL 推理。
- Hypertableau 路线：减少无效分支，是 HermiT 的代表做法。
- 规则化推理路线：把部分语义编译成规则，更适合 RL。
- 数据库下推路线：把查询重写到关系数据库，更适合 QL。

何时不用完整 OWL 推理，也应该明确：

- 只是关键词检索时，不需要。
- 只是简单枚举和过滤时，不需要。
- 只关心事务一致性时，数据库更合适。
- 业务规则已经能完整表达时，规则引擎通常更直接。

---

## 参考资料

| 来源 | 用途 | 对应章节 |
|---|---|---|
| W3C Direct Semantics | 一致性、模型、蕴含定义 | 问题定义与边界、核心机制与推导 |
| W3C Profiles | EL/QL/RL 的适用边界 | 工程权衡与常见坑、替代方案与适用边界 |
| HermiT / Pellet | 推理器实现路线 | 代码实现、替代方案与适用边界 |

1. [OWL 2 Web Ontology Language Document Overview (Second Edition)](https://www.w3.org/TR/owl-overview/)
2. [OWL 2 Web Ontology Language Direct Semantics (Second Edition)](https://www.w3.org/TR/owl2-direct-semantics/)
3. [OWL 2 Web Ontology Language Profiles (Second Edition)](https://www.w3.org/TR/owl2-profiles/)
4. [HermiT: an OWL 2 reasoner](https://ora.ox.ac.uk/objects/uuid%3Ae719ebc8-ff8b-4efa-a14f-7da8478ff0ed)
5. [Pellet: A practical OWL-DL reasoner](https://www.sciencedirect.com/science/article/pii/S1570826807000169)
