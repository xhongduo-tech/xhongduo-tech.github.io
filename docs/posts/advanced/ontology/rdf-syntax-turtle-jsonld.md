---
title: RDF 语法：Turtle、RDF/XML 与 JSON-LD
date: 2026-08-07
---

# RDF 语法：Turtle、RDF/XML 与 JSON-LD

<div class="epigraph">
<p>RDF 的抽象数据模型只有一个，能把它写下来的语法却有很多——每种语法都是一个不完美的仆人，为一种读者服务。</p>
<footer>—— 戴维·伍德（David Wood）等，RDF 1.1 语法规范编写者的共识</footer>
</div>

<div class="article-byline">
<p>第四级 · 本体论 ｜ W3C《RDF 1.1 Turtle 语法》《JSON-LD 1.1》规范 ｜ 2026-08-07</p>
</div>

## 为什么从语法开始

上一节定义了 RDF 的抽象模型（三元组、IRI、字面量、空白节点）。
但抽象模型没法落盘——你得把它**写出来**，机器才读得进去。
这一节讲 RDF 的三种主要**序列化（serialization）**：**Turtle**（人读的紧凑语法）、
**RDF/XML**（当年最正统、如今最啰嗦）、**JSON-LD**（Web 时代的 JSON 语法）。
理解这三种语法，你就明白了一个关键道理：**同一张 RDF 图可以长成
完全不同的外表，而语义完全不变**——语法的选择，是「读者」的选择，
不是「内容」的选择。

## 1 Turtle：为人类设计的紧凑语法

**Turtle（Terse RDF Triple Language）** 是 RDF 1.1 的人读首选。
它把「同一主语的多条三元组」合并书写，用缩写消除重复。

一个完整的 Turtle 例子：

```turtle
@prefix : <http://example.org/> .

:zhangsan a :Person ;
    :hasPet :huahua ;
    :name "张三" .

:huahua a :Cat ;
    :name "花花" .
```

Turtle 的四个常用缩写：

- **@prefix**：给 IRI 起短前缀，**`ex:Person` 展开为 <http://example.org/Person>`**。
- **分号（`;`）**：分隔**同一个主语**的下一条三元组。
- **逗号（`,`）**：分隔**同一主语谓语**的多个宾语。
- **`a`**：**rdf:type** 的简写——**`:x a :Y` 等价于 `:x rdf:type :Y`**。

**重点：Turtle 的语法糖全是「消除重复」**——主语写一次、谓语写一次、
类型写一个字母 a。这让 Turtle 成为「给人写的 RDF」：同样的信息，
比 RDF/XML 短一个数量级，且直接可读。<span class="marginnote">Turtle 的 a 不是缩写，而是
RDF 词汇表里预定义好的 rdf:type 的语法别名——它专门出现，因为
「类型断言」是 RDF 里最高频的关系。一个语法糖为一种高频操作而造，
这就是 Turtle 的设计哲学。</span>

## 2 N-Triples 与 RDF/XML：极简与正统

**N-Triples** 是 RDF 的「最小公倍数」：每行一条完整三元组，无任何缩写。

```
<http://example.org/zhangsan> <http://example.org/hasPet> <http://example.org/huahua> .
<http://example.org/zhangsan> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Person> .
```

优点：**极简、逐行解析、适合流式处理与差量比较**。
缺点：**冗长**——没有前缀、没有分号，完整 IRI 重复出现。

**RDF/XML** 是 RDF 的第一种语法，把三元组装进 XML：

```xml
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:ex="http://example.org/">
  <rdf:Description rdf:about="http://example.org/zhangsan">
    <rdf:type rdf:resource="http://example.org/Person"/>
    <ex:hasPet rdf:resource="http://example.org/huahua"/>
  </rdf:Description>
</rdf:RDF>
```

**重点：RDF/XML 的荣光与包袱同在**——它是 W3C 最早标准化的 RDF 语法
（让 RDF 借 XML 生态起步），但嵌套结构反而让「扁平三元组」的表达
绕了远路。今天它主要用于**兼容旧工具**，新项目几乎不用。
**Turtle 取代 RDF/XML，是「为机器设计」让位于「为人设计」的典型一幕。**<span class="marginnote">RDF/XML 的历史
教训值得记一笔：一个标准若绑定在「当时的主流技术」（XML）上，
就可能被更贴合问题的设计（Turtle、JSON-LD）反超。RDF/XML 没有错，
只是它的「读者」——XML 解析器——已经不再是大多数场景的主角。</span>

## 3 JSON-LD：Web 时代的语义 JSON

**JSON-LD（JSON for Linking Data）** 把 RDF 图装进 JSON 对象，
让「带语义的 JSON」能直接嵌入网页、API 与前端代码。

```json
{
  "@context": {
    "拥有宠物": "http://example.org/hasPet"
  },
  "@id": "http://example.org/zhangsan",
  "@type": "http://example.org/Person",
  "拥有宠物": {
    "@id": "http://example.org/huahua"
  }
}
```

**@context**：JSON 的键（"拥有宠物"）到 IRI 的映射表——
  上下文把「普通 JSON」升级成「语义 JSON」。
**@id**：资源的 IRI。
**@type**：类型，等价于 rdf:type。

**重点：JSON-LD 让「网页里的 JSON」变成「机器可读的 RDF」**
——浏览器和前端照常用 JSON，爬虫与知识引擎却能从同一份数据
读出 RDF 语义。这是 JSON-LD 结构化标记的标准载体：
Google 等搜索引擎从网页 JSON-LD 片段
里提取语义，构成搜索结果的知识底座。<span class="marginnote">JSON-LD 的「语义分层」
是它成功的秘诀：没有 @context，它就是普通 JSON（人人都会用）；
加上 @context，同一份数据就进入 RDF 世界（机器可推理）。
「渐进增强」的语义，让 JSON-LD 成为语义网渗透进 Web 的完美载体。</span>

## 4 同一张图的三副面孔

把「张三拥有宠物花花」这一句话，用三种语法写出来对照：

| 语法 | 写法 | 读者 | 特点 |
| --- | --- | --- | --- |
| N-Triples | 完整 IRI 一行一条 | 机器 | 极简、流式 |
| Turtle | 前缀 + `;` + `a` | 人 | 紧凑、可读 |
| RDF/XML | XML 元素嵌套 | 旧工具 | 啰嗦、正统 |
| JSON-LD | JSON + @context | Web | 渐进增强 |

**重点：四种语法描述的是同一张图——它们可以在任何解析器间
无损互转。** 一张图从 Turtle 转成 JSON-LD，语义零损失。
「语法可变、语义不变」是 RDF 生态最优雅的设计：
它把「表示」与「内容」彻底分离，让不同技术栈各取所需。<span class="marginnote">「一图多形」的
哲学对应物是第二篇弗雷格的意义—指称区分：意义（抽象图）独立于
表达（具体语法）。工程上，多语法意味着「数据格式选择」不再绑架
「数据语义」——你能按读者换外表，而不改内容。</span>

## 5 怎么选语法

工程选型的小指南：

| 场景 | 推荐语法 |
| --- | --- |
| 手工写本体/样例 | Turtle |
| 大文件流式处理 | N-Triples |
| 网页结构化数据 | JSON-LD |
| 兼容旧 RDF 工具链 | RDF/XML |
| API 返回数据 | JSON-LD 或 Turtle |

**易错点｜不要用「扩展名」猜语义**：同一张图可以存成 .ttl、.nt、
.jsonld 不同文件，它们表达的语义可能完全相同。「RDF 文件」的正确读法
是「先解析语法、再看图内容」，而不是「看扩展名猜结构」。

## 6 小结

- RDF 抽象模型只有一个，**语法可以有很多**。
- **Turtle**：@prefix/@base、`;`/`,`、`a` 缩写，为人读而设计。
- **N-Triples**：每行一三元组，极简、流式、适合差量比较。
- **RDF/XML**：最早的 XML 语法，如今主要用于兼容旧工具。
- **JSON-LD**：JSON + @context，Web 时代的渐进增强语义。
- 同一张图可**无损互转**——语法可变、语义不变。
- 选型指南：**手写用 Turtle，流式用 N-Triples，网页用 JSON-LD**。

在下一节，我们将给 RDF 加上「第一层语义」——**RDFS**：
rdfs:Class、rdfs:subClassOf、rdfs:subPropertyOf/rdf:type
如何让三元组之间的类与属性关系变得可推理。
