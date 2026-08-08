---
title: SPARQL 1.1：聚合、子查询与联邦查询
date: 2026-08-07
---

# SPARQL 1.1：聚合、子查询与联邦查询

<div class="epigraph">
<p>SPARQL 1.0 能查「图上有什么」；SPARQL 1.1 能算「图告诉了我们什么」。</p>
<footer>—— 史蒂夫·哈里斯与安迪·西博恩（Steve Harris, Andy Seaborne）主编，W3C《SPARQL 1.1 查询语言》(2013)</footer>
</div>

<div class="article-byline">
<p>第四级 · 本体论 ｜ W3C《SPARQL 1.1 查询语言》(2013)；Harris & Seaborne 编 ｜ 2026-08-07</p>
</div>

## 为什么从 SPARQL 1.1 开始

上一节的 SPARQL 1.0 能匹配图模式、能做左连接，但它有个致命短板：
**不能聚合、不能嵌套、不能跨越属性路径、不能查询远程图**。
2013 年发布的 SPARQL 1.1 把这些全部补齐：聚合/COUNT 让查询能统计，
子查询让复杂查询可以分步组合，**属性路径（property path）**让「传递闭包」
一蹴而就，SERVICE 让查询能跨数据源联邦检索。理解 SPARQL 1.1，
你就从「能查图」升级到「能分析图」——这是知识图谱从「存储」走向
「洞察」的分水岭。

## 1 聚合：把结果算成数字

SPARQL 1.1 引入 SQL 风格的聚合操作：

- **COUNT**：计数。
- **SUM、AVG、MIN、MAX**：对数值型字面量聚合。
- **GROUP BY**：按属性分组。
- **HAVING**：对分组结果过滤。

```sparql
SELECT ?person (COUNT(?pet) AS ?petCount)
WHERE {
  ?person :hasPet ?pet .
}
GROUP BY ?person
HAVING (COUNT(?pet) >= 2)
```

**重点：聚合把「图匹配」变成「图统计」。** 上面的查询返回
「每位至少养两只宠物的人及其宠物数」——图上的数据被折叠成数值。
`AS` 把聚合结果绑定给一个变量，供外层使用。<span class="marginnote">对照 SQL：
`(COUNT(?pet) AS ?petCount)` ≈ `COUNT(pet) AS cnt`，`GROUP BY ?person` ≈ `GROUP BY person`，
聚合语义完全一致。SPARQL 1.1 的聚合几乎是 SQL 聚合的
「图版平移」——学过 SQL 的人上手零成本。</span>

## 2 子查询与 BIND：分步组合复杂查询

子查询让一个查询的结果成为另一个查询的输入，实现「查询的复合」：

```sparql
SELECT ?person
WHERE {
  {
    SELECT ?person (COUNT(?pet) AS ?petCount)
    WHERE { ?person :hasPet ?pet . }
    GROUP BY ?person
  }
  FILTER (?petCount > 5)
}
```

外层查询里的 `嵌套 SELECT` 就是子查询：先算出每人宠物数，
再由外层过滤「宠物数 > 5」。**子查询是 SPARQL 的「函数组合」**
——复杂分析被拆成可读的小步骤。

**BIND** 在查询内做**表达式求值**并绑定到新变量：
`BIND (?price * ?quantity AS ?total)` 在匹配结果上计算新列。
**VALUES** 则显式给定一组解，用于「指定要查的实例」：
`VALUES ?person { :zhangsan :lisi }` 限定查询范围。

**重点：子查询、BIND、VALUES 是 SPARQL 1.1 的「计算能力三件套」**
——子查询组合查询，BIND 做行内计算，VALUES 枚举输入。它们让
SPARQL 从「模式匹配语言」进化成「图上的查询计算语言」。<span class="marginnote">子查询的
「分步」思想正是第四篇后向链的投影：先求子目标，再组合结果。
而 BIND 让你能在查询里「算数」，这让 SPARQL 具备轻量 ETL 能力——
图转换不必再导出到程序里做，查询内部就能完成。</span>

## 3 属性路径：一查到底的闭包

**属性路径（property path）** 是 SPARQL 1.1 最受用的新特性：
允许在谓词位置写**路径表达式**，一次匹配沿关系走多步。

```sparql
SELECT ?ancestor
WHERE {
  :zhangsan :hasParent+ ?ancestor .
}
```

路径运算符：

`elt1 / elt2`：连续走多步（复合路径）。
`elt*`：零步或多步（传递闭包）。
`elt+`：一步或多步。
`elt?`：零步或一步。
`^elt`：反向。

**重点：属性路径把「图的传递闭包查询」变成一行。**
「找出张三所有祖先」「找出所有间接依赖」这类「走到底」的问题，
在 SPARQL 1.0 里要靠反复 UNION 或应用层递归；1.1 用属性路径一蹴而就。
这是知识图谱查询最有威力的特性——**它让「关系图谱」真正被当图来查**。<span class="marginnote">属性路径与
第五篇的传递角色（S）遥相呼应：OWL 的 `TransitiveProperty` 用推理
表达「可传递」，SPARQL 的 `elt*` 用路径表达「可遍历」。一个由推理机
保证语义，一个由查询引擎保证效率——两种「闭包」各司其职。</span>

## 4 联邦查询：SERVICE 的跨源检索

**联邦查询（federated query）** 让一条 SPARQL 查询同时查多个端点：

```sparql
SELECT ?person
WHERE {
  ?person :bornIn ?place .
  SERVICE <http://example.org/geonames/sparql> {
    ?place rdfs:label "北京"@zh .
  }
}
```

**SERVICE**：把内部图模式**发送给远程 SPARQL 端点**执行，
  结果带回本地参与连接。
本地先匹配 `?person :bornIn ?place`，再远程确认「出生地是否叫北京」。

**重点：联邦查询把语义网「开放数据」连成一张虚拟大图。**
你不必把所有数据下载到本地——`SERVICE` 让查询引擎充当「分布式
检索代理」，跨 DBpedia、Wikidata、政府开放数据查询。**这是
「链接数据（Linked Data）」理念的查询层实现**：数据分布在世界各地，
查询却像在一张图上。<span class="marginnote">联邦查询的代价是<strong>性能与可靠性</strong>：
远程端点慢或不可用，整个查询就卡住。工程上常用「物化缓存」
把远程结果快照到本地，再用增量同步更新——这是「联邦 vs 物化」
两种分布式数据架构的经典权衡，与第八篇「图谱存储」话题相通。</span>

## 5 SPARQL 1.1 特性总览

一张表收拢 SPARQL 1.1 的主要新增：

| 特性 | 语法 | 解决什么问题 | 对应 SQL |
| --- | --- | --- | --- |
| 聚合 | COUNT/SUM/AVG | 统计与分组 | 同 SQL |
| 子查询 | 嵌套 `SELECT` | 查询复合 | 派生表 |
| BIND | `BIND` | 行内计算 | SELECT 表达式 |
| VALUES | `VALUES` | 枚举输入 | IN 列表 |
| 属性路径 | `/`、`*`、`+`、`?` | 传递闭包 | 递归 CTE |
| 联邦 | `SERVICE` | 跨源查询 | 外部数据源 |
| 否定 | `NOT EXISTS` | 不存在判断 | NOT EXISTS |

**易错点｜FILTER NOT EXISTS 与 MINUS 作用范围不同**：
`FILTER NOT EXISTS` 是对**当前行的值**做检查——每个绑定
用自己变量代入模式，若「存在匹配」则该行被滤掉。它是「按行过滤」
而非「全局判断」，理解这一点，才不会写出「滤不掉」的查询。

## 6 小结

- SPARQL 1.1 补齐 1.0 的四大短板：**聚合、子查询、属性路径、联邦**。
- **聚合**（COUNT/SUM/AVG）把图匹配变成图统计。
- **子查询 + BIND + VALUES** 提供查询的复合、计算与枚举能力。
- **属性路径**（`/`、`*`、`+`、`?`）一行实现传递闭包查询——最有威力。
- **联邦查询**（`SERVICE`）跨端点检索，把开放数据连成虚拟大图。
- 特性总览表：SPARQL 1.1 与 SQL 几乎一一对应。
- 工程提醒：**属性路径用 `*` 快在常见、慢在极端**，大图上要设限制。

在下一节，我们将把语言与算法交给真正的推理机——**Pellet、HermiT、FaCT++**：
看成熟的 OWL 推理引擎如何落地分类、一致性检查与解释服务。
