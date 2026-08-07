---
title: SHACL：RDF 数据的形状约束与验证
date: 2026-08-07
---

# SHACL：RDF 数据的形状约束与验证

<div class="epigraph">
<p>OWL 告诉我们什么可以推出来，SHACL 告诉我们什么不可以发生。</p>
<footer>—— 霍尔格·克努布劳赫与迪米特里斯·孔托科斯塔斯（Holger Knublauch, Dimitris Kontokostas），W3C《SHACL 规范》(2017) 编者的共识</footer>
</div>

<div class="article-byline">
<p>第四级 · 本体论 ｜ W3C《SHACL 规范》(2017)；Knublauch & Kontokostas 编 ｜ 2026-08-07</p>
</div>

## 为什么从 SHACL 开始

前面十节我们建起了完整的语义网栈：RDF 存数据，RDFS/OWL 给语义，
SPARQL 查数据，推理机推结论。但缺了一块让工程师抓狂的能力——
**数据校验**。OWL 的开放世界说「没声明的不算假」，可现实中你需要
「这个字段必须有」「这个值必须是日期」「这个属性至多出现一次」。
2017 年，W3C 发布 **SHACL（Shapes Constraint Language，形状约束语言）**，
给 RDF 补上了「封闭世界的约束」：定义**形状（shape）**，验证数据
是否符合形状。理解 SHACL，你就掌握了语义网栈的「质检闸门」——
也理解了 OWL 的开放世界与校验的封闭世界如何互补。

## 1 为什么 OWL 做不了校验

回到第五篇的 OWA：如果知识库里没有「张三有电话」，OWL 会说「未知」，
不会说「缺电话」。而数据校验要求「**张三必须有电话**」——这是一条
**违反就失败**的硬约束。OWL 的表达是「推理导向」的，它从公理推出
新事实，却不「拒绝」不符合的数据。

**重点：OWL 是「推理语义」，SHACL 是「约束语义」。**
OWL 问「从这些公理能推出什么？」；SHACL 问「这份数据符合这些形状吗？」。
前者开放、单调、可判定；后者封闭、非单调、可校验。
**两类语义服务于两类需求：建模用 OWL，质检用 SHACL。**<span class="marginnote">用一句话
区分：OWL 的 `minCardinality` 是「推理」——它允许推出「张三至少有一个
孩子」，但不报「缺孩子」的错误；SHACL 的 `minCount` 是「校验」——
「张三必须有孩子」不满足就报告违规。一个管「能推」，一个管「该有」。</span>

## 2 形状：node shape 与 property shape

**SHACL 形状（shape）**：对一组 RDF 节点应满足的条件描述。
形状分两类：

- **节点形状（node shape）**：约束「某个节点本身」——它是什么类型、
  必须有哪些属性。用 `sh:targetClass` 指定作用于哪个类的所有实例。
- **属性形状（property shape）**：约束「节点的某个属性」——出现的次数、
  值的数据类型、值的节点类型。

```turtle
:PersonShape a sh:NodeShape ;
  sh:targetClass :Person ;
  sh:property [
    sh:path :年龄 ;
    sh:datatype xsd:integer ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
  ] .
```

**重点：形状就是「数据的体检标准」。** 上面的形状规定：每个 `Person`
必须**恰好一个**整数类型的 `年龄`。验证时，引擎扫描所有 `Person`
实例，逐条检查这个属性形状——不满足就生成一条**违规报告
（validation result）**。<span class="marginnote">形状的哲学对应：它像
数据库的「schema 约束」与「表单校验」的合体——但作用在 RDF 图上。
你可以把 SHACL 想成「RDF 世界的 JSON Schema」：JSON Schema
校验 JSON 文档，SHACL 校验 RDF 图。这个类比对现代工程师特别友好。</span>

## 3 约束组件：SHACL 的词汇表

SHACL 提供一组标准的**约束组件（constraint component）**：

| 约束 | 语法 | 作用 |
| --- | --- | --- |
| 最小/最大次数 | `sh:minCount`/`sh:maxCount` | 属性出现次数范围 |
| 数据类型 | `sh:datatype` | 值必须是某数据类型 |
| 节点类型 | `sh:nodeKind` | 值是 IRI、字面量还是空白节点 |
| 类约束 | `sh:class` | 值的类型必须是某类 |
| 值域 | `sh:in` | 值必须在给定列表里 |
| 节点约束 | `sh:node` | 值必须满足另一个形状 |
| 封闭形状 | `sh:closed` | 节点只能有列出的属性 |
| 逻辑组合 | `sh:and`/`sh:or`/`sh:not` | 形状的布尔组合 |
| 自定义 | `sh:sparql` | 用 SPARQL 写任意约束 |

**重点：约束组件覆盖了数据质量的全部常见需求。**
`minCount`/`maxCount` 管数量，`datatype`/`nodeKind` 管类型，
`in` 管枚举，`closed` 管「不允许多余字段」。**SHACL 的可扩展性
（`sh:sparql`）让它能表达任意可查询的约束**——约束语言本身
不会「不够用」。<span class="marginnote">`sh:closed` 是 SHACL 最「封闭世界」的构子：
它要求节点<strong>只</strong>拥有列出的属性。这在 OWL 里完全无法表达（OWL
从不禁止「多一个属性」），却是数据质量（防脏字段、防拼写错误）
的刚需——封闭世界假设在约束层名正言顺地回归了。</span>

## 4 公式解析：一次 SHACL 验证的完整流程

用上面的 `PersonShape` 验证一份数据，走一遍：

```turtle
:张三 a :Person ; :年龄 "abc" .
:李四 a :Person ; :年龄 25 ; :年龄 26 .
```

- **第一步，确定目标**：`sh:targetClass :Person` → 验证对象是
  张三与李四。
- **第二步，应用属性形状**：对每个人，检查 `:年龄` 是否满足
  `datatype xsd:integer`、`minCount 1`、`maxCount 1`。
- **第三步，逐条判定**：
  - 张三：`:年龄 "abc"` 是字符串不是整数 → **违反 datatype**。
  - 李四：`:年龄` 出现两次 → **违反 maxCount**。
- **第四步，生成报告**：输出违规列表（哪条数据、哪个形状、哪条约束）。

**重点：验证的结果是「违规报告」，不是「模型拒绝」**——SHACL 不修改
数据，只告诉你「哪里不合格、违反哪条形状」。这份报告可以被
人类审查、被流水线消费、被用来阻断不合格数据入库。<span class="marginnote">注意验证
与推理的执行方式不同：OWL 推理是「闭包扩展」（往图里加新三元组），
SHACL 验证是「检查判定」（扫描图并输出违规）。前者增、后者查——
这是两个完全不同的运行时，也是工程上要分别部署的原因。</span>

## 5 SHACL 与 OWL 的分工

两者常被混淆，一张表划清界限：

| 维度 | OWL | SHACL |
| --- | --- | --- |
| 世界假设 | 开放（OWA） | 封闭（校验） |
| 核心动作 | 推理（推新事实） | 验证（查违规） |
| 失败语义 | 不适用（无「失败」） | 产生违规报告 |
| 典型用途 | 建模、分类、蕴含 | 数据质量、表单、契约 |
| 数量约束 | `minCardinality`（推理） | `minCount`（校验） |
| 封闭约束 | 无法表达 | `sh:closed` |

**易错点｜不要用 SHACL 当推理机，也不要要求 OWL 做校验**：
SHACL 的 `sh:class` 检查「值的类型」但不推出新类型；OWL 的
`subClassOf` 推出类型但不报告「缺失」。**一个完整的知识系统
两样都要**：OWL 定义「知识是什么」，SHACL 保证「数据对不对」。
现代图谱实践（第八、九篇）正是这种「推理 + 校验」的双引擎架构。

## 6 小结

- SHACL 给 RDF 补上**封闭世界的约束与验证**——OWL 做不了的事。
- 形状分**节点形状**与**属性形状**，用 `sh:targetClass` 指定验证对象。
- 约束组件覆盖**数量、类型、枚举、封闭、逻辑组合**，可扩展 `sh:sparql`。
- 验证产出**违规报告**：指出哪条数据违反哪条形状，不改数据。
- 分工明确：**OWL 管推理（开放、增），SHACL 管校验（封闭、查）**。
- 工程双引擎：**知识系统 = OWL 定义语义 + SHACL 保证质量**。

到这里，第六篇《语义网技术栈》就完成了：从分层蛋糕、RDF、RDFS、OWL 2、
Profiles、SPARQL 到推理机与 SHACL——一套完整的 W3C 标准栈。
下一站进入第七篇——**本体工程**：从「怎么用语言」到「怎么建本体」，
方法论文、能力问题与上层本体的系统工程实践。
