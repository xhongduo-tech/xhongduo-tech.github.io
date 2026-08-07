---
title: RDFS：类、子类、属性与领域值域
date: 2026-08-07
---

# RDFS：类、子类、属性与领域值域

<div class="epigraph">
<p>RDF 告诉你「存在哪些陈述」，RDFS 告诉你「这些陈述之间的类与属性如何组织、能推出什么」。</p>
<footer>—— 丹·布里克利与拉马纳坦·古哈（Dan Brickley, Ramanathan V. Guha），《RDF Schema 1.1》(2014)</footer>
</div>

<div class="article-byline">
<p>第四级 · 本体论 ｜ W3C《RDF Schema 1.1》(2014)；Brickley & Guha 编 ｜ 2026-08-07</p>
</div>

## 为什么从 RDFS 开始

上一节 RDF 只会陈述事实，却不懂「类」与「子类」。这一节给它补上
第一层语义——**RDFS（RDF Schema，RDF 模式）**。RDFS 用一组
**内置词汇**（`rdfs:Class`、`rdfs:subClassOf`、`rdfs:domain`、`rdfs:range`）
给 RDF 图加上「类—属性」的组织结构，并定义**推理规则**——
从已知三元组推出新三元组。它是语义网里最「轻」的 schema 语言，
弱得恰到好处：够表达分类层级，又简单到可以大规模执行。
理解 RDFS，你就理解了「数据如何长出 schema」，
也理解了它在描述逻辑家族里的位置——一个极简的 EL。

## 1 核心词汇：类与属性

RDFS 扩展 RDF，引入约十个核心词汇：

- **`rdfs:Class`**：声明「某个资源是一个类」——如 `:猫 rdf:type rdfs:Class`。
- **`rdfs:subClassOf`**：子类关系——`:猫 rdfs:subClassOf :宠物`。
- **`rdfs:subPropertyOf`**：子属性关系——`:有女儿 rdfs:subPropertyOf :有孩子`。
- **`rdfs:domain`**：属性的定义域——`:拥有宠物 rdfs:domain :人`。
- **`rdfs:range`**：属性的值域——`:拥有宠物 rdfs:range :动物`。
- **`rdfs:label` / `rdfs:comment`**：给资源加人类可读的名称与注释。

注意区分：`rdf:type` 连接「个体—类」（花花是猫），
`rdfs:subClassOf` 连接「类—类」（猫是宠物）。**这正是第四篇语义网络
危机的「五种 is-a」在 RDFS 里的正规化**：`rdf:type` 管实例化，
`rdfs:subClassOf` 管类包含，各司其职。<span class="marginnote">RDFS 词汇表是
「最小的本体语言」：它只承诺「类、子类、属性、子属性、领域、值域」——
没有否定、没有数量、没有逆关系。这份克制让 RDFS 的推理
（见第 3 节）简单且可大规模执行。</span>

## 2 domain 与 range：属性的「户口」

`rdfs:domain` 与 `rdfs:range` 是最容易误解、也最容易用错的两个词。

- **`rdfs:domain`**：声明「谁可以当这个属性的主语」。
  `:拥有宠物 rdfs:domain :人`——意思是：**凡出现 `?x :拥有宠物 ?y`，
  则 `?x` 是一个 `:人`**。
- **`rdfs:range`**：声明「谁可以当宾语」。
  `:拥有宠物 rdfs:range :动物`——**凡出现 `?x :拥有宠物 ?y`，则 `?y` 是一个 `:动物`**。

**重点：domain/range 不是「约束」，而是「推导」**——它们不拒绝违规数据，
而是**自动补类型**。若数据里出现 `:李四 :拥有宠物 :石头`（没有类型声明），
RDFS 推理会推出「李四是人」「石头是动物」。这与数据库的
外键约束（违规即拒绝）截然相反：**RDFS 是开放世界的，它倾向于
「推断类型」而不是「报错」**。<span class="marginnote">domain/range 的「推导而非约束」
语义常让数据库背景的人措手不及：你想用 `rdfs:domain` 做「数据校验」，
结果它悄悄给每一条数据补了类型。若真要校验，得用 SHACL（本系列
第 10 节）——那是专门为「约束」设计的语言。领域值域与约束，
是两种完全不同的需求。</span>

## 3 公式解析：RDFS 的五条推理规则

RDFS 的「语义」可以浓缩成一组推理规则。核心五条：

$$x \ \text{rdf:type} \ A, \quad A \ \text{rdfs:subClassOf} \ B \;\Longrightarrow\; x \ \text{rdf:type} \ B$$

$$P \ \text{rdfs:subPropertyOf} \ Q, \quad (a, b) \in P \;\Longrightarrow\; (a, b) \in Q$$

$$P \ \text{rdfs:domain} \ C, \quad (a, b) \in P \;\Longrightarrow\; a \ \text{rdf:type} \ C$$

$$P \ \text{rdfs:range} \ C, \quad (a, b) \in P \;\Longrightarrow\; b \ \text{rdf:type} \ C$$

$$A \ \text{rdfs:subClassOf} \ B, \quad B \ \text{rdfs:subClassOf} \ C \;\Longrightarrow\; A \ \text{rdfs:subClassOf} \ C$$

逐条拆解第一式（最常用的子类推理）：

- **前提一**：`x` 被声明为 `A` 的实例（`x rdf:type A`）。
- **前提二**：`A` 是 `B` 的子类（`A subClassOf B`）。
- **结论**：`x` 自动成为 `B` 的实例——**实例沿子类链向上传播**。

用实例走一遍：`花花 rdf:type :猫` + `:猫 subClassOf :宠物`
⇒ 推出 `花花 rdf:type :宠物`。**一条子类公理，让无数实例获得新类型**——
这就是 RDFS 推理的价值：少量 schema 公理，驱动大量数据推演。<span class="marginnote">这套
推理规则与第五篇 DL 的包含推理完全同构：`subClassOf` 就是 ⊑，
`x rdf:type A` 就是概念断言。RDFS 可以被翻译成 EL 描述逻辑——
它「恰好」是 DL 家族里最轻、最可扩展的一支。想深入，
回看第五篇的 EL 复杂度（PTIME）。</span>

## 4 RDFS 的边界：它有多「弱」

RDFS 刻意不提供 DL 的大部分能力。它**不能**表达：

- **否定**：不能说「猫不是狗」。
- **基数约束**：不能说「人至少有两个孩子」。
- **逆关系**：不能说「有孩子 ⟺ 有父母」。
- **传递角色**：不能说「祖先可传递」。
- **不相交**：不能说「猫与狗互斥」。

**重点：RDFS 的「弱」，换来了「快」与「简单」**——它的推理规则
是固定的、局部的、可在大规模图上高效执行（RDFS 蕴含可在多项式
甚至近线性时间计算）。**RDFS 是「词汇层」的 schema，OWL 才是
「逻辑层」的 schema**：当领域需要否定、数量、逆关系时，就要升级到 OWL
（第 5 节）；当只需要分类层级时，RDFS 足够且更划算。<span class="marginnote">RDFS vs OWL 的分工，
对应第五篇 EL vs ALC 的分工：RDFS ≈ EL（子类、存在、无否定），
OWL DL ≈ SROIQ（完整构造子）。「选 RDFS 还是 OWL」的本质，
是「你的领域需不需要否定与数量约束」——这个判断，决定整个建模的复杂度预算。</span>

## 5 RDFS 的工程地位

RDFS 在实践中远比它「弱」的名声更常用：

- **轻量 schema**：许多知识图谱（如 schema.org 早期、维基数据的前身）
  用 RDFS 级的类层级组织数据。
- **RDF 数据校验的辅助**：RDFS 提供基本的类型推断，是数据清洁的
  「第一道工序」。
- **OWL 的温床**：几乎每个 OWL 本体都包含 RDFS 词汇——OWL 是
  RDFS 的超集，「rdfs:subClassOf」在 OWL 里原样可用。

**易错点｜`rdfs:subClassOf` 与 `rdf:type` 是不同层的断言**：
`猫 rdfs:subClassOf 宠物` 是类到类的公理；`花花 rdf:type 猫` 是
个体到类的断言。把 `subClassOf` 写成 `rdf:type`（或反之）是新手最常见的错——
前者推不出「花花是宠物」，后者会错误地把「猫」当一个个体的名字。

## 6 小结

- RDFS 给 RDF 补上**类、子类、属性、子属性、领域、值域**的语义。
- **`rdfs:subClassOf`** 管类包含，**`rdf:type`** 管实例化，两层分明。
- **domain/range 是推导不是约束**：它们自动补类型，不拒绝数据。
- 五条推理规则：**子类传播、子属性传播、domain 补类型、range 补类型、子类传递**。
- RDFS 不能表达：**否定、基数、逆关系、传递角色、不相交**。
- RDFS 的「弱」换「快」：推理可大规模执行，适合轻量 schema。
- 工程地位：**RDFS 是词汇层 schema，OWL 是逻辑层 schema**；
  选型看「需不需要否定与数量」。

在下一节，我们将从「词汇层」升到「逻辑层」——**OWL 2 的构子**：
等价、不相交、属性链与基数约束，如何把 RDFS 升级成完整的描述逻辑。
