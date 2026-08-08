---
title: OWL 2 的构子：等价、不相交、属性链与基数约束
date: 2026-08-07
---

# OWL 2 的构子：等价、不相交、属性链与基数约束

<div class="epigraph">
<p>OWL 2 是一个词汇表，更是一台推理机的规格——它把第五篇的 SROIQ 装进了 W3C 标准，让「语义」可以在全世界被共享。</p>
<footer>—— 伊恩·霍罗克斯等（Ian Horrocks et al.），OWL 2 规范编写者</footer>
</div>

<div class="article-byline">
<p>第四级 · 本体论 ｜ W3C《OWL 2 结构规范与直接语义》(2012)；Baader《描述逻辑手册》 ｜ 2026-08-07</p>
</div>

## 为什么从 OWL 2 构子开始

RDFS 只提供了「词汇层」的语义；这一节把它升级成完整的**逻辑层**——
**OWL 2（Web Ontology Language）**。OWL 2 是一套 W3C 标准词汇，
把第五篇的 SROIQ 描述逻辑包装成 RDF 语法：`owl:equivalentClass`、
`owl:disjointWith`、`owl:propertyChainAxiom`、`owl:someValuesFrom`——每个关键字
都对应一个 DL 构造子，每个构造子都有模型论语义，都能被推理机
（HermiT、Pellet）判定。**掌握 OWL 2 构子，就是掌握「可共享的
描述逻辑」**：你写下的每一条 OWL 公理，全世界的推理机都能解读。

## 1 类的构子与公理：等价与不相交

OWL 2 表达「类之间的关系」，最常用四类公理：

- **owl:equivalentClass**：声明两个类等价——「单身汉 ≡ 未婚男性」。
  等价是最强的类断言，等价类共享全部实例。
- **rdfs:subClassOf**：子类关系——RDFS 的 `subClassOf` 在 OWL 里原样继承。
- **owl:disjointWith**：声明两个类**不相交**——「猫与狗互斥」，
  没有任何个体能同时属于两者。这是 RDFS 完全缺失的能力。
- **owl:disjointUnionOf**：声明「若干个类正好把父类瓜分」——如「人
  恰好是男人与女人的不相交并」。

**重点：owl:disjointWith 是 RDFS 升 OWL 的第一块跳板。** RDFS 只能说
「猫是宠物」，OWL 还能说「猫不是狗」——有了否定性声明，推理机
才能发现**矛盾**（如果一个个体被断言既是猫又是狗，知识库就不一致）。
**「能否表达互斥」，是词汇层与逻辑层的分水岭。**<span class="marginnote">等价与不相交
对应第五篇 DL 的 ≡ 与「⊓ = ⊥」；它们把 TBox 从「分类金字塔」升级成
「可检测矛盾的理论」。这也是本体调试（用推理机找「哪条公理导致
概念为空」）的前提——没有否定，就没有矛盾可查。</span>

## 2 属性公理：链、传递、逆与函数

OWL 2 的属性公理是 SROIQ 的「角色」部分：

**owl:propertyChainAxiom**：属性链——「叔父 = 父亲的兄弟」：
  `hasParent ∘ hasBrother ⊑ hasUncle`。这是 SROIQ 的 R。
**owl:TransitiveProperty**：传递属性——`hasAncestor` 可传递。
**owl:inverseOf**：逆属性——`hasChild` 与 `hasParent` 互逆。
**owl:FunctionalProperty**：函数属性——每个个体至多一个值，
  如「亲生父亲」。对应 DL 的 ≤ 1 数限制。
**owl:SymmetricProperty**：对称属性——`friendOf` 对称。

**重点：属性公理回答「关系怎么组合、怎么走、怎么反向」**——
链描述「关系的复合」，传递描述「关系的闭包」，逆描述「关系的反向」，
函数描述「关系的一值性」。这些是 RDFS 完全触碰不到的能力，
也是「语义」真正变厚的地方。<span class="marginnote">属性链是 OWL 2 里
「表达力最贵」的构子之一（第五篇复杂度一节说 R 让 SROIQ 跳到
N2EXPTIME）。用它一次，推理机可能慢十倍——工程上要克制。
但像「叔父」这种常识关系，不用链就得手工枚举所有叔父，两难。</span>

## 3 属性限制：someValuesFrom 与基数

OWL 2 的**属性限制（restriction）**把属性与类结合，定义「拥有某属性
且满足某条件的类」：

**owl:someValuesFrom**（存在限制）：`hasPet some Cat`——「至少有一只
  宠物是猫」的人。对应 DL 的 ∃R.C。
**owl:allValuesFrom**（全称限制）：`hasChild only Human`——「所有孩子
  都是人」的人。对应 ∀R.C。
**owl:minCardinality / owl:maxCardinality / owl:cardinality**（基数）：
  `hasChild min 2`——「至少两个孩子」。对应 ≥ 2 / ≤ 2 / = 2。
**owl:hasValue**：`worksFor value Google`——「在谷歌工作」。
**owl:hasSelf**：`knows Self`——「认识自己」。

**重点：属性限制让「类」可以由「属性条件」来定义，而不只是子类堆叠。**
「有女儿的人」「至多两个孩子的人」「在谷歌工作的人」——这些
「条件类」是本体建模的主力，也是查询（「找出所有有女儿的人」）
背后的逻辑引擎。<span class="marginnote">owl:someValuesFrom 与 owl:allValuesFrom 的对偶
（∃ vs ∀）我们已在第五篇 ALC 一节反复演练。它们在 OWL 里的写法
直接对应 DL 记号，理解 DL 则 OWL 只需背语法。</span>

## 4 公式解析：一条属性链公理

把「叔父 = 父亲的兄弟」写成 OWL，逐行拆解：

```turtle
:hasUncle  owl:propertyChainAxiom  ( :hasParent :hasBrother ) .
```

**第一步，读语义**：属性链公理说——若 `hasParent(x, y)` 且
  `hasBrother(y, z)`，则 `hasUncle(x, z)`。即「我的父亲的兄弟是我的叔父」。
**第二步，对应 DL**：$hasParent \circ hasBrother \sqsubseteq hasUncle$。
**第三步，推理效果**：知识库一旦有 `hasParent(:zhangsan, :laozhang)`、
  `hasBrother(:laozhang, :laowang)`，推理机自动推出 `hasUncle(:zhangsan, :laowang)`。

**重点：属性链把「多跳关系」压缩成「单跳关系」**——它是查询优化与
知识压缩的利器：不用手工枚举「所有叔父三元组」，一条公理即可。
但记住它的复杂度代价：SROIQ 的 R 是双指数复杂度的来源之一，
大本体上慎用。<span class="marginnote">属性链的哲学对应：它表达的正是
「关系如何复合」——这让人想起范畴论的态射复合（第十篇）：
`hasParent` 与 `hasBrother` 是两条态射，链公理断言它们的复合
落在 `hasUncle` 里。范畴论的关系复合观，在这里是字面意义。</span>

## 5 一张构子对照表

把 OWL 2 构子、DL 记号与含义收进一张表，方便查阅：

| OWL 2 构子 | DL 记号 | 含义 |
| --- | --- | --- |
| owl:subClassOf | $C \sqsubseteq D$ | 子类 |
| owl:equivalentClass | $C \equiv D$ | 等价 |
| owl:disjointWith | $C \sqcap D \sqsubseteq \bot$ | 不相交 |
| owl:unionOf | $C \sqcup D$ | 并类 |
| owl:complementOf | $\neg C$ | 补类 |
| owl:someValuesFrom | $\exists R.C$ | 存在限制 |
| owl:allValuesFrom | $\forall R.C$ | 全称限制 |
| owl:minCardinality | $\geq n\,R.C$ | 至少 n |
| owl:inverseOf | $R^{-}$ | 逆角色 |
| owl:TransitiveProperty | $R$ 传递 | 传递角色 |
| owl:propertyChainAxiom | $R \circ S \sqsubseteq T$ | 属性链 |

**易错点｜owl:allValuesFrom 不是「必须有」**：owl:allValuesFrom
只约束「**若**有孩子，孩子必须是人」，并不声明「有孩子」。
要同时表达「有孩子且孩子都是人」，需 `hasChild some Human`
与 `hasChild only Human` 合写。这个「全称是条件、存在是存在」的
区别，是 OWL 建模的第一大坑。

## 6 小结

- OWL 2 = **SROIQ 的 W3C 词汇版**：每个构子都有模型论语义。
- 类公理：**等价、子类、不相交、不相交并**——owl:disjointWith 是逻辑层分水岭。
- 属性公理：**属性链、传递、逆、函数、对称**——关系如何组合与走法。
- 属性限制：**some/allValuesFrom、基数、hasValue、hasSelf**——条件类的定义。
- 属性链公理把「多跳关系」压成「单跳」，代价是复杂度剧增。
- owl:allValuesFrom 是「若有则必须」，不是「必须有」——建模第一大坑。
- 构子对照表：**OWL 2 ↔ DL ↔ 含义**，一表打通两套语言。

在下一节，我们将回答「完整 OWL 2 太贵怎么办」——**OWL 2 Profiles**：
EL、QL、RL 三个「减配」子语言如何各取所需地换低复杂度。
