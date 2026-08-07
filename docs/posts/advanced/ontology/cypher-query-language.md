---
title: Cypher 查询语言：模式匹配的图查询
date: 2026-08-07
---

# Cypher 查询语言：模式匹配的图查询

<div class="epigraph">
<p>Cypher 让查询看起来像图画：你画出你想找的形状，数据库替你把它在数据里找出来。</p>
<footer>—— 纳丁·弗朗西斯等（Nadime Francis et al.），《Cypher 的形式语义》(2018)</footer>
</div>

<div class="article-byline">
<p>第四级 · 本体论 ｜ Neo4j Cypher 手册；Francis et al.《Cypher 形式语义》(2018) ｜ 2026-08-07</p>
</div>

## 为什么从 Cypher 开始

上一节我们看了属性图的存储（Neo4j），这一节配上它的查询语言——
**Cypher**。它是属性图世界的「SQL」：一种**声明式图模式匹配**语言，
用类似 ASCII 画的语法（`(:Person)-[:WORKS_FOR]->(:Company)`）
直接表达「我想找的形状」。Cypher 与第六篇的 SPARQL 是
「图查询」的两副面孔——一个为属性图、一个为 RDF。
理解 Cypher，你就补齐了「工程派」知识图谱的查询武器，
也为下一节「属性图 vs RDF」的正面比较备好了双方弹药。

## 1 模式语法：把查询画出来

Cypher 的核心是**图模式（graph pattern）**，用 ASCII 艺术写：

```cypher
MATCH (p:Person)-[:WORKS_FOR]->(c:Company)
WHERE p.name = '张三'
RETURN c.name
```

- **`(p:Person)`**：节点模式——变量 `p`，标签 `Person`。
- **`-[:WORKS_FOR]->`**：关系模式——类型 `WORKS_FOR`，方向向右。
- **`(c:Company)`**：另一个节点。
- **`MATCH ... WHERE ... RETURN`**：匹配 → 过滤 → 返回。

**重点：Cypher 的模式就是「图的模板」**——与 SPARQL 的 BGP 同构，
只是用 ASCII 画的箭头代替尖括号三元组。**「画出形状，找出匹配」**
是 Cypher 的心智模型，比 SQL 的 `SELECT ... FROM ... JOIN`
更贴近「关系」的直觉。<span class="marginnote">Cypher 的箭头语法并非装饰：
`->` 表示「必须这个方向」，`-[]-` 表示「方向无所谓」，
`<-[r]-` 表示「反向」。<strong>方向的显式表达</strong>是图查询语言的特权——
SQL 的 join 没有方向概念，这正是图模型优于表模型的场景之一。</span>

## 2 路径与变长匹配：多跳的天然表达

Cypher 处理「多跳」用**变长路径（variable-length path）**：

```cypher
MATCH (a:Person)-[:FRIEND*1..3]->(b:Person)
RETURN a.name, b.name
```

- **`[:FRIEND*1..3]`**：沿 `FRIEND` 关系走 1 到 3 跳。
- **`[:FRIEND*]`**：任意长度（要小心：可能指数级）。
- **`*2`**：恰好 2 跳。

**重点：变长路径把「找所有间接好友」「找供应链上 N 层」这类查询
写成一行。** 这是图查询对比 SQL 的核心优势——SQL 的递归 CTE
要写一大段，Cypher 一个 `*` 搞定。**「多跳」是 Cypher 的母语，
是 SQL 的外语。**<span class="marginnote">变长路径对应第八篇的「传递闭包查询」，
与 SPARQL 1.1 的属性路径（`*`）功能等价。两者在不同存储
上实现了同一个「闭包」直觉——<strong>闭包查询是图查询语言的
「灵魂功能」</strong>，谁缺了它谁就出局。</span>

## 3 写图：CREATE 与 MERGE

Cypher 不只读，还能写：

- **`CREATE`**：无条件新建节点/关系。
- **`MERGE`**：**有则返回、无则创建**（upsert）——幂等写入，最常用。
- **`SET`**：给节点/关系加属性。
- **`DELETE`**：删除；`DETACH DELETE` 连关系一起删。

```cypher
MERGE (p:Person {name: '张三'})
ON CREATE SET p.age = 35
MERGE (p)-[:WORKS_FOR]->(c:Company {name: '谷歌'})
```

**重点：`MERGE` 是图谱写入的「防重复」核心。** 它按模式查找，
找到了就复用，找不到才创建——天然支持幂等的增量写入。
**知识图谱构建管道（第八篇）持续写入时，`MERGE` 就是
「实体消解」在存储层的最后防线**：同一名字不会建出两个节点。<span class="marginnote">`MERGE` 的
语义与 SQL 的 `MERGE`/`UPSERT`、以及上一节实体消解的目标一致：
「同一事物只有一个节点」。它是「去重」需求在查询语言里的
制度化——<strong>不写 `MERGE` 而用 `CREATE`，图谱必然长出重复实体</strong>。</span>

## 4 公式解析：一个「找同事」查询的执行

把「张三的同事」翻译成 Cypher，拆解执行路径：

```cypher
MATCH (:Person {name:'张三'})-[:WORKS_FOR]->(:Company)<-[:WORKS_FOR]-(colleague:Person)
RETURN colleague.name
```

- **第一步，锚定**：从「名字为张三的 Person」节点出发（走索引）。
- **第二步，一跳**：沿 `WORKS_FOR` 找到他任职的公司。
- **第三步，反向一跳**：从公司反向沿所有 `WORKS_FOR` 找到其他员工。
- **第四步，去重与返回**：`colleague` 排除张三自己，返回名字。

**逐项拆解成本：**

- **锚点靠索引**：`name='张三'` 走唯一性索引，$O(\log n)$ 定位。
- **遍历靠邻接**：后续每步是 index-free adjacency 的指针跳转。
- **模式是连接**：整条模式就是一个多跳图遍历，无 join 全表。

**重点：Cypher 的执行器把「图模式」编译成「图遍历计划」。**
模式里的每个箭头 = 一次沿关系的行走，整条链 = 一次深度受控的
遍历。**理解「模式 = 遍历」，你就理解了 Cypher 的性能模型**
——快慢取决于路径长与节点度，而非全图大小。<span class="marginnote">与 SPARQL 执行
（索引连接）对照：Cypher 偏「遍历式执行」，SPARQL 偏「连接式执行」。
两者在「多跳链式」查询上都快，但在「密集节点的星形扩展」上，
遍历式有天然优势——这也是属性图在社交/风控类图谱上
常胜的原因之一。</span>

## 5 Cypher vs SPARQL：一页对照

| 维度 | Cypher | SPARQL |
| --- | --- | --- |
| 数据模型 | 属性图 | RDF 图 |
| 模式语法 | ASCII 箭头 | 三元组模式 |
| 多跳 | `*1..3` | `/`、`*` |
| 写操作 | `CREATE`/`MERGE` | CONSTRUCT（间接） |
| 推理 | 无原生 | 蕴含机制 |
| 标准 | Neo4j 主导（openCypher） | W3C 标准 |
| 擅长 | 遍历、性能 | 语义查询、推理 |

**易错点｜Cypher 与 SPARQL 不是「同义反复」**：它们服务不同
数据模型——Cypher 查属性图（节点/关系/属性），SPARQL 查 RDF
（三元组/IRI/推理）。**「把一个 RDF 数据加载进 Neo4j 就能用
Cypher 查」是对的，但会丢失 RDF 的语义**（blank node、推断、
全局 IRI 语义）。选语言先选数据模型——这个决定上一节已铺垫，
下一节正式摊牌。

## 6 小结

- Cypher 是属性图的**声明式模式匹配语言**，语法即图画。
- 核心构件：**节点模式、关系模式、变长路径**（`*1..3`）。
- 多跳是 Cypher 的母语：**变长路径一行搞定传递闭包**。
- 写图三件套：**CREATE（建）、MERGE（幂等 upsert）、SET（改）**。
- 执行本质：**模式 → 图遍历计划**，成本随路径与节点度。
- 与 SPARQL 对照：**遍历式 vs 连接式、属性图 vs RDF**。
- 定位：Cypher 是「工程派」图谱的查询武器，SPARQL 是「语义派」的。

在下一节，我们将让两个阵营正面交锋——**属性图 vs RDF**：
两种范式的全面对比，以及它们在工程里如何互操作、如何各取所长。
