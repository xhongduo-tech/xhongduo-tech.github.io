---
title: SPARQL 查询语言：图模式、OPTIONAL 与 FILTER
date: 2026-08-07
---

# SPARQL 查询语言：图模式、OPTIONAL 与 FILTER

<div class="epigraph">
<p>对 RDF 而言，SPARQL 就像 SQL 对关系数据库：一套用「模式匹配」代替「表连接」的查询语言。</p>
<footer>—— 豪尔赫·佩雷斯等（Jorge Pérez, Marcelo Arenas, Claudio Gutierrez），《SPARQL 的语义与复杂度》(2009)</footer>
</div>

<div class="article-byline">
<p>第四级 · 本体论 ｜ W3C《SPARQL 1.1 查询语言》(2013)；Pérez et al.《SPARQL 的语义与复杂度》(2009) ｜ 2026-08-07</p>
</div>

## 为什么从 SPARQL 开始

前面几节我们建立了 RDF 图、RDFS/OWL 语义，但还缺最后一环——**怎么把知识查出来**。
这就是 **SPARQL**（SPARQL Protocol and RDF Query Language）的工作：
语义网标准查询语言。它的核心思想是**图模式匹配（graph pattern matching）**——
你写一个带变量的「图模板」，SPARQL 引擎在整张 RDF 图上找所有「对得上」的子图。
理解 SPARQL，你就掌握了「查 RDF」的唯一标准方式，
也为第八篇「三元组存储」与「图数据库」的对比打下基础——
SPARQL 是 RDF 世界的事实查询标准，就像 SQL 之于关系库。

## 1 基本图模式：用变量搭模板

**基本图模式（Basic Graph Pattern, BGP）**：一组带变量的三元组模式。
变量以 `?` 或 `$` 开头。一个最简单的查询：

```sparql
PREFIX : <http://example.org/>
SELECT ?宠物
WHERE {
  ?人 :拥有宠物 ?宠物 .
}
```

- **`SELECT ?宠物`**：要返回的变量。
- **`WHERE { ... }`**：图模式——`?人 :拥有宠物 ?宠物` 是一条三元组模式，
  其中 `?人`、`?宠物` 是变量，`:拥有宠物` 是具体谓词。
- **匹配语义**：引擎在图上找所有 `(资源, :拥有宠物, 资源)` 的三元组，
  把 `?人`、`?宠物` 分别绑定到对应资源——输出所有「宠物」。

**重点：BGP 匹配 = 找子图同态。** 整张 RDF 图是一张大网，你的 WHERE
是一张「带洞的模板」；匹配就是找出所有「模板塞进图里能对齐」的位置。
多个三元组模式合在一起，要求**同一变量在多处绑定一致**——这就是
「图的连接」。<span class="marginnote">SPARQL 的 `?x` 是「存在变量」：查询
「?人 拥有宠物 ?宠物」在逻辑上等于「存在某人，某人拥有某宠物」——
这与你用第五篇 ABox 的「检索服务」问的问题完全一致。
SPARQL 就是 ABox 检索的工程化语法。</span>

## 2 查询形式：四种输出

SPARQL 有四种查询形式，对应不同的输出需求：

- **`SELECT`**：返回一组变量绑定（表格）。
- **`CONSTRUCT`**：用匹配结果**构造新的 RDF 图**——用于图转换与抽取。
- **`ASK`**：返回布尔值——「有没有匹配？」（用于存在性判断）。
- **`DESCRIBE`**：返回描述某资源的一张图（实现相关的资源描述）。

**重点：`CONSTRUCT` 让 SPARQL 不只是「查」，还能「造图」**——
从一个图模式匹配的结果生成另一组三元组，这是「图重写」的引擎：
数据转换、本体对齐、视图物化全都能用 CONSTRUCT 实现。<span class="marginnote">`CONSTRUCT`
的地位相当于 SQL 的 `SELECT ... INTO` 与视图的组合：它让 SPARQL
从「只读查询语言」升级为「图转换语言」。第七篇「本体对齐」
常借 CONSTRUCT 把映射写成可执行的图变换。</span>

## 3 OPTIONAL、FILTER 与 UNION：补足图模式的表达力

单独一个 BGP 还不够，SPARQL 用三个操作符扩展模式语言：

- **`OPTIONAL`**：左连接——「若能匹配则匹配，匹配不上不报错」。
  查「所有人及其宠物」，没宠物的人也要出现，`宠物` 列为空。
- **`FILTER`**：对变量做条件过滤——数值比较、正则、语言标签。
  `FILTER(?年龄 > 18)` 筛掉未成年。
- **`UNION`**：并集——匹配「模式 A 或模式 B」。

```sparql
SELECT ?人 ?宠物
WHERE {
  ?人 a :人 .
  OPTIONAL { ?人 :拥有宠物 ?宠物 }
  FILTER NOT EXISTS { ?人 :拥有宠物 :猫 }
}
```

**重点：`OPTIONAL` 是 SPARQL 最反直觉也最常用的操作符。**
它实现的是「左连接」——左边（`?人 a :人`）的结果全部保留，
右边（`?人 :拥有宠物 ?宠物`）能匹配的补上，匹配不上的留空。
**它把「信息缺失」变成「可选的补充」**，与 RDF 的开放世界
气质天然契合。<span class="marginnote">对照关系数据库：`OPTIONAL` ≈ `LEFT JOIN`，
`UNION` ≈ `UNION`，`FILTER` ≈ `WHERE` 子句。SPARQL 与 SQL
的操作符一一对应，只是作用对象从「表」换成了「图」。
学过 SQL 的人，SPARQL 语法几乎是「平移」。</span>

## 4 公式解析：一次图模式匹配的展开

把一条带 OPTIONAL 的查询，拆成引擎内部的执行语义。

查询：「找出所有人，以及他们的宠物（如果有），且宠物不是猫。」

- **第一步，执行 BGP `?人 a :人`**：扫描图，得到所有人的绑定集合
  $\{ \langle ?人 \mapsto 张三\rangle, \langle ?人 \mapsto 李四\rangle, \ldots \}$。
- **第二步，执行 OPTIONAL 子模式 `?人 :拥有宠物 ?宠物`**：对每个
  `?人` 绑定，在图上找 `(张三, :拥有宠物, ?)`——找到就绑定 `?宠物`，
  找不到就让 `?宠物` 保持未绑定。这是**左连接**：所有 `?人` 保留。
- **第三步，FILTER 后置过滤**：剔除「`?宠物` 绑定到一只猫」的行。
- **第四步，投影**：输出 `?人` 与 `?宠物`（未绑定的输出空）。

**重点：SPARQL 的求值顺序是「BGP 匹配 → 逐级连接 → 过滤 → 投影」。**
理解这个管道，你就能预测查询结果，也能读懂「为什么有的行有空值」——
那不是错误，而是 OPTIONAL 留下的「未绑定」。<span class="marginnote">注意 SPARQL 的
`FILTER NOT EXISTS` 是<strong>封闭世界味道</strong>的构造：它判定「图上不存在匹配」。
这与 OWL 推理的开放世界形成张力——同一份数据，SPARQL 说「查无此猫」，
OWL 推理机却说「不知道」。这再次提醒：查询层与推理层，
是两套不同的世界假设。</span>

## 5 SPARQL 与蕴含：查询「推出来的知识」

SPARQL 的匹配对象可以是「原始图」，也可以是「蕴含闭包图」——

- **简单图匹配（simple entailment）**：只查显式三元组。
- **RDF/RDFS 蕴含（entailment regime）**：引擎在匹配前先做 RDFS 推理，
  于是 `subClassOf`、domain/range 推出的三元组也能被查到。
- **OWL 蕴含**：完整 OWL 推理后匹配（最贵）。

**重点：选蕴含机制 = 选「查询时要不要先推理」。** 用 RDFS 蕴含，
`SELECT ?x WHERE { ?x a :宠物 }` 能返回所有「被推理为宠物」的个体；
用简单匹配则只返回显式声明。**蕴含机制的代价是性能**——
RDFS 闭包可接受，OWL 闭包则可能指数级。工程上常用「物化」
（预先算好闭包存起来）来兼顾两者。

## 6 小结

- SPARQL 是 RDF 的**标准查询语言**，核心是**图模式匹配**。
- **BGP**：带变量的三元组模式，匹配 = 找子图同态，变量跨模式绑定一致。
- 四种查询形式：**SELECT、CONSTRUCT、ASK、DESCRIBE**——查、造、问、述。
- **OPTIONAL**（左连接）、**FILTER**（过滤）、**UNION**（并集）扩展表达力。
- 求值管道：**BGP 匹配 → 连接 → 过滤 → 投影**。
- **蕴含机制**决定「查显式数据」还是「查推理结果」。
- 工程对照：**SPARQL ≈ 图的 SQL**，操作符与 SQL 几乎一一对应。

在下一节，我们将看 SPARQL 的进阶语法——**聚合、子查询与联邦查询**：
`GROUP BY`、`COUNT`、嵌套查询，以及如何跨多个 SPARQL 端点做联邦检索。
