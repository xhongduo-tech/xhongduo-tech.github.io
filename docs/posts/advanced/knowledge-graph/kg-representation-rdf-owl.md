---
title: 知识图谱的表示：RDF 与 OWL
date: 2026-08-11
---

# 知识图谱的表示：RDF 与 OWL

<div class="epigraph">
<p>语义网并不是与现有网络分庭抗礼的「另一个网」，而是现有网络的延伸——在其中，信息被赋予明确的含义，让计算机与人能够更有效地协作。</p>
<footer>—— 蒂姆 · 伯纳斯-李（Tim Berners-Lee），《The Semantic Web》（2001）</footer>
</div>

<div class="article-byline">
<p>第四级 · 高阶专题 · 知识图谱 ｜ 对标教材：Hogan et al. §2 ｜ 2026-08-11</p>
</div>

## 为什么从「表示」开始

任何知识系统都要回答同一个问题：**知识以什么形式存下来，才既能查询、又能推理？** 这就是表示（representation）问题，也是本专题的第一问。知识图谱这一路解决方案的答案，浓缩成三个词：**RDF、RDFS、OWL**。它们分别是数据层、轻量模式层、知识表示层的语法地基。<span class="marginnote">本专题其余九篇——实体识别、关系抽取、嵌入、推理、问答、本体——全都建立在这套语法之上。先把三元组和类层级吃透，后面每一步都会反复引用。</span>

这件事与大模型形成最鲜明的对照：大模型把知识折进亿万参数里，**知识图谱把知识一笔一划写在图上**。参数里的知识不可见、不可校验；图上的知识可见、可验证、可推理。<span class="marginnote">第四级《RAG 专题》讲的正是把两者接起来：用图的精确补偿向量的模糊，用向量的泛化补偿图的稀疏。这个「双引擎」思路的源头就在本节。</span>

## 1 RDF：一张三元组构成的大网

**RDF（Resource Description Framework，资源描述框架）**：一种用「主语—谓语—宾语」三元组描述资源的模型。每一条知识都是一个三元组 $(s, p, o)$，三部分角色各不相同：

**主语（subject）**：一个 URI 或空白节点，指代「被描述的东西」；
- **谓语（predicate）**：一个 URI，指代「属性或关系」；
- **宾语（object）**：一个 URI、空白节点或**字面量（literal）**（字符串、数字、日期等，可带数据类型）。

用 Turtle 语法写三条事实：

```
@prefix ex: <http://example.org/> .
ex:Einstein  ex:bornIn   ex:Ulm .
ex:Einstein  ex:birthYear  "1879"^^xsd:gYear .
ex:Ulm       rdf:type   ex:City .
```

它们读作三句人话：「爱因斯坦出生在乌尔姆」「爱因斯坦出生于 1879 年」「乌尔姆是一座城市」。

**核心概念**：**RDF 图谱（RDF graph）**：一张图就是一个三元组的集合。集合天然无序、无重复——这决定了 RDF 的语义是「陈述的集合」，而不是有顺序的树或表。

三个容易误解的部件：

- **URI/IRI**：全局唯一的资源标识符，是世界通用的「身份证」。谓语必须用 URI，因为关系必须有明确身份。
- **空白节点（blank node）**：以 `_:` 开头、没有全局名字的节点，用于表示「存在某个东西，但我不必给它起名」。它让 RDF 能表达存在量词（见第八篇《本体》）。
- **字面量**：数据值本身，可带类型 `^^xsd:...`，把「1879」这个字符串和「1879 这个年份」区分开。

**易错点：** RDF 是「图」而不是「树」。同一个主语可以出现在很多三元组里，同一个宾语也可以被很多主语指向，多个图谱可以并成一个大图谱——并图的操作就是三元组集合的并集。这与后面要讲的图嵌入、图查询都直接相关。

## 2 RDFS：给图加上类型与继承

**RDFS（RDF Schema）**：RDF 的轻量模式层，提供描述「类与属性之关系」的词汇，让图拥有**类型、继承、属性定义**：

- `rdfs:Class` 与 `rdf:type`：声明类、给实例打标签。`ex:Einstein rdf:type ex:Person`。
- `rdfs:subClassOf`：类继承。`ex:Person rdfs:subClassOf ex:Agent`，则「爱因斯坦是人」蕴含「爱因斯坦是行动者」。
- `rdfs:subPropertyOf`：属性继承。`ex:hasMother rdfs:subPropertyOf ex:hasParent`。
- `rdfs:domain` 与 `rdfs:range`：属性的定义域与值域。

**重点：RDFS 的 domain/range 是「推导」而不是「校验」。** 关系数据库里的 schema 会拒绝不满足约束的记录；RDFS 里若 `ex:bornIn` 声明了 domain 为 `ex:Person`，那么一出现 `ex:Einstein ex:bornIn ex:Ulm`，推理器就**自动推导**出 `ex:Einstein rdf:type ex:Person`，而不是报错。<span class="marginnote">这是 RDF 世界与关系数据库最深刻的分野之一，请务必记住：开放世界假设下没有「违例」，只有「尚未明说的知识」。第八篇《本体》会用开放世界假设专门辨析。</span>

再加上 `rdfs:label`（可读名字）与 `rdfs:comment`（注释），RDFS 就够描述绝大多数轻量知识图谱的模式了——DBpedia、Schema.org 这类轻量模式的骨架就是它。

## 3 OWL：把图的表达力推向推理

**OWL（Web Ontology Language，网络本体语言）**：在 RDFS 之上提供更丰富的逻辑词汇，让图不仅能分类，还能表达**等价、互斥、函数性、约束**：

- `owl:sameAs`：两个不同 URI 指向同一个现实对象（对齐的关键工具，见第四篇）；
- `owl:equivalentClass` / `owl:equivalentProperty`：类与属性的等价；
- `owl:inverseOf`：互逆属性，如 `hasChild` 与 `hasParent`；
- `owl:FunctionalProperty`：函数性属性，一个主语最多取一个值；
- `owl:TransitiveProperty`：传递属性；
- `owl:disjointWith`：互斥类；
- 构造类：`owl:unionOf`、`owl:intersectionOf`、`owl:Restriction`（配合 `owl:onProperty`、`owl:someValuesFrom`、`owl:allValuesFrom`、基数约束）。

这些词汇把「分类学」升级为「一阶逻辑的一角」。例如用 OWL 说「母亲都有至少一个孩子」、说「猫和狗互斥」，推理器就能据此推出原本没有显式写出的知识。

RDF、RDFS、OWL 是层层叠加的三层：**RDF 是数据层，RDFS 是轻量模式层，OWL 是知识表示层**。OWL 的精确语义建立在描述逻辑之上，我们在《本体与本体构建》一篇给出它的形式语法——本节先把这三个层次的名字和分工记住。

## 4 公式解析：RDF 图的精确定义与 RDFS 闭包

记号先行。令 $U$ 为全体 IRI 的集合，$B$ 为空白节点的集合，$L$ 为字面量的集合。

**RDF 三元组的精确定义**：

$$
(s,\ p,\ o) \in (U \cup B) \times U \times (U \cup B \cup L)
$$

**一张 RDF 图就是三元组的集合**：

$$
G \subseteq (U \cup B) \times U \times (U \cup B \cup L)
$$

对这条式子做三步拆解：

- **第一步，看主语**：主语可以是 IRI 也可以是空白节点，但不能是字面量——「一个数字不能作为陈述的主语」，这保证了世界上的「东西」与「值」被分开。
- **第二步，看谓语**：谓语只能是 IRI。这是设计哲学：关系必须有全局唯一的身份，否则两套图谱无法合并。
- **第三步，看宾语**：宾语三者皆可，字面量只出现在宾语位置——数据值永远是被断言的东西，不再往下开枝散叶。

有了图的精确定义，就能定义**推理闭包（closure）**。以 `rdfs:subClassOf` 的传递性为例，推理规则写作「分子分母」的形式：

$$
\frac{(x,\ \texttt{rdfs:subClassOf},\ y) \quad (y,\ \texttt{rdfs:subClassOf},\ z)}
{(x,\ \texttt{rdfs:subClassOf},\ z)}
$$

规则读作：**只要分子（前提）成立，就能推出分母（结论）**。把这类规则机械地反复套用，直到不再产生新的三元组，所得的集合叫闭包 $\mathrm{cl}(G)$，这个过程叫**物化（materialization）**。它的价值在于：把「每次查询都要现算的推理」提前成「存下来的新事实」——这正是第七篇《知识推理与规则》forward chaining 的雏形。

## 5 图表示与关系表、大模型

RDF 不是唯一的表示方案，与它对照着看才显出其取舍：

| 维度 | 关系数据库 | RDF 图谱 |
| --- | --- | --- |
| 模式 | 固定 schema，列名即语义 | 无固定 schema，开放世界 |
| 缺失值 | NULL（封闭世界假设） | 直接不写（开放世界假设） |
| 跨源合并 | 需要精心设计的 join | 图即并集，天然可合并 |
| 语义推理 | 不提供 | RDFS/OWL 蕴含规则 |

关系数据库擅长「事务与聚合」，RDF 擅长「开放世界下的知识合并」。这也是为什么知识图谱工程普遍的做法是：**把结构化的运营数据留在库里，把跨源的、需要语义推理的知识搬到图上**。

而与大模型的关系，一句话：**LLM 把知识存在参数里，RDF 把知识存在图上**。前者擅长联想与生成，后者擅长精确与可解释——RAG 把它们桥接起来，让模型「先查图、再作答」。

## 6 小结

- **RDF** 以三元组 $(s, p, o)$ 为基本单位，图谱 = 三元组集合；主语可为空白节点，宾语可为字面量，谓语必须是 IRI。
- **RDFS** 提供类、类型（`rdf:type`）、继承（`subClassOf`/`subPropertyOf`）与 `domain`/`range` 推导。
- **OWL** 提供 `sameAs`、逆属性、函数性、互斥、构造类与基数约束，是描述逻辑的表述层。
- **RDF 语义是开放世界、单调的**：`domain`/`range` 是推导而非校验。
- 闭包 `cl(G)` 是把推理规则反复套用得到的不动点，物化让推理结果可预先存储。

在下一节，我们将离开「语法层」，回答一个更接地气的问题：**文本里的知识如何变成三元组**——先认出「谁」，再认出「它和谁是什么关系」，这就是实体识别与实体链接。
