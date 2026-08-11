---
title: 词汇语义与词汇资源（WordNet / FrameNet）
date: 2026-08-11
---

# 词汇语义与词汇资源（WordNet / FrameNet）

<div class="epigraph">
<p>世界不是由名词组成的，而是由事件组成的；名词只是事件的影子。</p>
<footer>—— 查尔斯 · 菲尔墨（Charles Fillmore，框架语义学）</footer>
</div>

<div class="article-byline">
<p>第四级 · 高阶专题 · NLP · 句法与语义分析 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从词汇资源开始

上一节我们靠 WordNet 的 synset 消歧，但那只是**用**资源，还没看清资源本身长什么样。这一节要系统地回答一个问题：**「词义」到底是怎样一种数据结构？** 答案的两种经典形态，各自塑造了一整代 NLP 系统：

- **WordNet**：把词义组织成**关系网**——上位/下位、整体/部分、同义/反义，像一张巨大的语义地图。
- **FrameNet**：把词义组织成**场景**——动词不是孤立词条，而是一个「事件脚本」里带槽位的演员。

对标 Manning & Schütze 第 19 章与 Jurafsky 第 21 章相关部分。这张「词汇资源地图」是连接上一节 WSD 与下一节分布语义的枢纽：WordNet 代表**人工精心构造**的词义结构，分布语义代表**数据自动涌现**的词义结构——两种哲学的对照，正是现代 NLP 最深刻的一条分水岭。<span class="marginnote">分水岭可以这样记：WordNet 是「一个词义一个 synset，关系由人写死」；词嵌入是「一个词义一个向量，关系由数据学出」。前者稳、后者活，今天的主流明显倒向后者，但 WordNet 的清单性价值（下位关系、反义）依然不可替代。</span>

## 1 WordNet：同义词集与词汇关系

**WordNet**（Princeton WordNet，1985 年起，George Miller 主持）是历史最悠久、影响最大的词汇资源。它的核心单位是**同义词集（synset）**：一组可互换的近义词共同表示一个词义。`{car, auto, automobile, machine, motorcar}` 是一个 synset，「汽车」这个义项。一个多义词词形（如 `bank`）同时属于多个 synset，一个 synset 反过来容纳多个词形——**「词形 ↔ 词义」是多对多映射**，这正是 WSD 的结构基础。

synset 之间通过**词汇关系**连成网：

**上位（hypernym）**：`car` ⊂ `vehicle`；上位词是「更一般」的类。
- **下位（hyponym）**：`car` ⊃ `sedan`；下位词是「更具体」的实例。<span class="marginnote">上位/下位构成一棵棵 <strong>IS-A 树</strong>：`car`→`vehicle`→`conveyance`→`artifact`→…→`entity` 一路爬到顶。这是信息检索「查询扩展」（搜「车」也搜「轿车」）和本体对齐的骨架。</span>
**整体/部分（meronym/holonym）**：`engine` ⊂ `car`（部分-整体关系，PART-OF 而非 IS-A）。
- **同义（synonymy）** 与**反义（antonymy）**：`fast/slow`。

词的每种词性（名词、动词、形容词、副词）各有一棵独立的 IS-A 树，名词树最深、结构最清晰，动词树则更强调**蕴含（entailment）** 关系（`walk` ⊃ `move`）。

## 2 FrameNet：框架与框架元素

**FrameNet**（Fillmore 主持，1997 年起）换了一套组织哲学：**语义的最小单位不是「词义」，而是「框架（frame）」**——一个被语言反复表征的典型场景。<span class="marginnote">「商业交易」是一个框架，它的<strong>框架元素（frame elements）</strong> 是买方、卖方、商品、价钱；`buy/sell/pay/cost/charge` 这些词都「唤起」同一框架，只是从不同视角点亮它。框架语义学主张：理解一个词，就是唤起整个场景。</span>

- **框架（frame）**：如 `Commerce_buy`、`Ingestion`、`Motion`，带有语义条件与元素列表。
- **框架元素（frame elements）**：场景中的参与者，如 `Buyer`、`Seller`、`Goods`、`Money`。
- **词元（lexical units）**：能唤起该框架的词，如 `buy.v`、`purchase.v`、`sale.n`。
- **框架间关系**：如 `Inherits_from`（`Commerce_buy` 继承 `Getting`）、`Uses`、`Precedes`。

FrameNet 的力量在于**跨动词抽象**：「我买书」与「他卖书」涉及同一场景，只是观察角度（Buyer 视角 vs Seller 视角）不同——这比 PropBank 逐动词编号更接近认知，但标注成本也更高，规模远小于 WordNet。

## 3 公式解析：WordNet 上的语义相似度

资源的价值在能「算」。如何度量两个词/两个 synset 的**语义相似度**？最自然的是走 IS-A 树上的路径：

$$
\mathrm{sim}_{\text{path}}(c_1, c_2) = \frac{1}{1 + \mathrm{dist}(c_1, c_2)}
$$

拆解三步：

- **第一步，距离**：$\mathrm{dist}(c_1, c_2)$ 是两 synset 在上位树上的最短路径边数——`car` 到 `sedan` 走 1 条边，`car` 到 `dog` 要绕到 `entity` 再下来，距离很大。
- **第二步，转换**：距离越小相似度越大；加 1 保证分母不为零且值域落在 $(0,1]$。
- **第三步，更精的变体**：路径法有盲区——不同深度的边「含金量」不同。**Wu-Palmer** 用最近公共祖先 $lcs$ 的深度加权；**Resnik** 引入**信息含量（information content）** $IC(c) = -\log P(c)$（用语料统计「上位词出现频率」），把「越罕见越具体」编码进相似度。三者权衡了「结构距离」与「数据频率」。

$$ \mathrm{sim}_{\text{resnik}}(c_1, c_2) = IC\big(\mathrm{lcs}(c_1, c_2)\big) $$

Resnik 的精髓一句话：**两个词有多像，取决于它们共同的上位信息有多「稀有」**——「狗和猫」共享的上位 `animal` 比「狗和汽车」共享的 `entity` 罕见得多，故前者更相似。这条「用信息量度量语义距离」的思路，直接预言了十几年后词嵌入里「向量夹角度量相似」的做法。

## 4 从资源到模型：WordNet 的现代命运

WordNet 的鼎盛期是统计 NLP 时代：词义消歧、查询扩展、释义生成都挂在它的 IS-A 树上。它的遗产在三个层面延续至今：

1. **作为标注标准**：SemCor 的词义标注、许多评测任务仍以 WordNet synset 为标签空间。
2. **作为监督信号**：下游模型用 WordNet 的上下位关系构造正负样本（如对比学习中「`dog` 与 `poodle` 更近」）。
3. **作为结构化知识**：知识图谱、本体工程里，IS-A 树几乎是「常识分类」的标准骨架。<span class="marginnote">有个有趣的对照：词嵌入（下节）擅长「相似」，WordNet 擅长「分类」——`dog` 和 `cat` 向量夹角很小，但只有 WordNet 告诉你 `dog` 的 IS-A 父类是 `canine`。今天的系统往往两者兼取。</span>

## 5 小结

- **WordNet** 以 synset 为单位组织词义，通过**上位/下位、整体/部分、同义/反义**关系构成 IS-A 网，是词义清单的标准答案。
- **FrameNet** 以框架（场景）为单位，跨动词抽象参与者角色，是「动词论元语义」的认知方案。
- 相似度可沿 IS-A 树计算：**路径距离、Wu-Palmer、Resnik（信息含量）**，一步步从「结构」走向「结构+数据」。
- 两种资源互补：WordNet 管「分类清单」，FrameNet 管「场景角色」；今天常与词嵌入并用。

在下一节，我们告别人工构造的词义——**分布语义与词嵌入**让机器自己从语料里长出「词义向量」。
