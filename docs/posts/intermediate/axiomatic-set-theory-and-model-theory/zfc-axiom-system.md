---
title: ZFC 公理体系（外延、配对、并集、幂集等）
date: 2026-08-07
---

# ZFC 公理体系（外延、配对、并集、幂集等）

<div class="epigraph">
<p>我们必须知道，我们必将知道。</p>
<footer>—— 大卫 · 希尔伯特（David Hilbert），1930 年柯尼斯堡广播演讲</footer>
</div>

<div class="article-byline">
<p>第二级 · 公理集合论与模型论 ｜ Jech, <em>Set Theory</em> 第1章；Kunen 第1章 ｜ 2026-08-07</p>
</div>

## 为什么从公理清单开始

在《朴素集合论悖论与 ZFC 公理系统》里，我们通过罗素悖论看到了「任何性质都造出一个集合」这条原则的危险，也看到了分离公理如何收窄它。这一篇换一个视角：把 ZFC 当作一部**形式系统**逐条解剖——每条公理的形式陈述是什么、它解决什么问题、哪些构造依赖哪一条、哪些公理其实可以被别的公理推出来。<span class="marginnote">第一级《集合的概念》给了集合的直观，本篇给这套直观配上「语法」；到了第4篇模型论，我们才讨论它的「语义」——同一组公理可以有两个不同的模型。</span>

先把「公理是什么」这件事说破：公理不是「显然为真的真理」，而是**一个形式系统的出发点**。ZFC 公理选得自然、与数学实践吻合，但它的权威不在「自明」，而在「自洽」——这正是希尔伯特那句「我们必须知道，我们必将知道」的背景。<span class="marginnote">希尔伯特 1900 年把连续统假设列为二十三个问题之首，1930 年又在柯尼斯堡说出这句名言。它的完整含义是：任何一个明确的数学问题原则上都能得到确定的回答——本专题第3篇将告诉你，这在某种意义上是对的，在某种意义上又是错的。</span>

## 1 外延公理与四条构造公理

**外延公理（extensionality）** 说：两个集合元素完全相同，则它们相等，

$$\forall x \, \forall y \, \bigl( \forall z (z \in x \leftrightarrow z \in y) \rightarrow x = y \bigr)$$

外延公理不保证任何集合存在，它只约束「相等」由元素决定。存在性由下面四条公理给出：

- **空集公理**：$\exists x \, \forall y \, (y \notin x)$。空集 $\emptyset$ 存在，且由外延公理唯一。
- **配对公理**：$\forall a \, \forall b \, \exists x \, (a \in x \wedge b \in x)$。$\{a, b\}$ 存在。
- **并集公理**：$\forall x \, \exists y \, \forall z \, \bigl(z \in y \leftrightarrow \exists w \in x (z \in w)\bigr)$。$\bigcup x$ 存在。
- **幂集公理**：$\forall x \, \exists y \, \forall z \, (z \in y \leftrightarrow z \subseteq x)$。$\mathcal{P}(x)$ 存在。

这四条的名字值得背下来，因为后面所有构造都会「点名」它们。<span class="marginnote">幂集公理是「爆炸性」的一步：$\mathcal{P}(\mathbb{N})$ 不可数，康托尔定理 $|\mathcal{P}(x)| > |x|$ 保证集合宇宙每升一层就真地变大一次——那是基数篇的主角。</span>

## 2 公式解析：从公理造出有序对与笛卡尔积

单靠外延、配对、并集、幂集，就能造出本课程以后用到的一切结构。看有序对的库拉托夫斯基（Kuratowski）定义：

$$(a, b) = \bigl\{ \{a\}, \{a, b\} \bigr\}$$

三步拆解：

- **第一步**：为什么需要绕一层？$\{a, b\}$ 是集合，记不住次序；而 $\{\{a\}, \{a,b\}\}$ 里有一个「只含 $a$ 的子元素 $\{a\}$」，据此可以唯一地读出谁是第一个坐标。
- **第二步**：验证语义性质 $(a,b) = (c,d) \leftrightarrow a = c \wedge b = d$。若两对相等，由外延公理比对四个小集合，推出 $a = c$ 且 $b = d$；反过来显然。
- **第三步**：配对公理给出 $\{a\}$ 与 $\{a,b\}$，并集与幂集给出它们的容器——于是有序对在 ZFC 内部就有了合法身份。

有序对一旦成立，**笛卡尔积**就水到渠成：

$$A \times B = \bigl\{ z \in \mathcal{P}\bigl(\mathcal{P}(A \cup B)\bigr) \;\bigm|\; \exists a \in A \, \exists b \in B \; (z = (a,b)) \bigr\}$$

这里每层括号都对应一条公理：$A \cup B$ 由并集公理，$\mathcal{P}(\mathcal{P}(A\cup B))$ 由两次幂集，最后的「筛选」由分离公理模式。**集合论里没有「凭空造」，只有「从已有集合出发、逐条公理地点名」**——这是它与朴素集合论最大的气质差别。<span class="marginnote">把同样的程序搬进编程：Python 的元组、C 的 struct、JSON 的对象都是「有序对的实现」。程序员说「类型系统是一层又一层的对与积」，和这里的公理构造是同一种递归精神。</span>

**辨析｜易错点：** 初学者常把「有序对」与「集合」对立起来，其实在 ZFC 里有序对就是一个集合。真正需要警惕的是次序：$(a,b) \ne (b,a)$ 但 $\{a,b\} = \{b,a\}$——「序」要从无序的集合里生长出来，这正是序数篇的伏笔。

## 3 分离、替换与无穷：三个公理模式

**分离公理模式（separation schema）** 我们已见过：对每条公式 $\varphi(x, w_1, \dots, w_n)$，

$$\forall z \, \forall \vec w \, \exists y \, \forall x \bigl( x \in y \leftrightarrow x \in z \wedge \varphi(x, \vec w) \bigr)$$

它取代了朴素概括原则，是罗素悖论的解药。

**替换公理模式（replacement schema）**：对每条公式 $\varphi(x, y, \vec w)$，若它对每个 $x$ 至多确定一个 $y$，则

$$\forall z \, \exists u \, \forall y \bigl( \exists x \in z \, \varphi(x, y, \vec w) \rightarrow y \in u \bigr)$$

直观地说：**把公式作用在集合 $z$ 的每个元素上，得到的「像」仍是一个集合**。<span class="marginnote">替换公理是为「序数序列」这类对象引进的：没有它，$n \mapsto \omega + n$ 的像 $\{\omega+n : n<\omega\}$ 就不是集合，$2 \cdot \omega$ 等序数算术就卡壳——序数篇的主角。</span>

**无穷公理**：$\exists x \bigl( \emptyset \in x \wedge \forall y \in x \, (y \cup \{y\} \in x) \bigr)$。它宣告自然数集 $\omega$ 存在：$0 = \emptyset$，$1 = \{\emptyset\}$，$2 = \{\emptyset, \{\emptyset\}\}$，……每一个后继都是 $n+1 = n \cup \{n\}$。<span class="marginnote">von Neumann 的这一条定义让「自然数」与「序数」住进同一个屋檐：自然数是最小的序数，序数是超限的「自然数」。到序数篇、基数篇，它会一路长成 $\omega$、$\omega_1$、$\aleph_0$、$\aleph_1$……</span>

**公式解析：替换公理为什么不可少。** 有人问：$\{\omega + n : n < \omega\}$ 不就是「从 $\omega$ 里筛出满足 $x = \omega+n$ 的元素」吗，分离公理难道不够？——不够，因为 $\omega+n \notin \omega$，它们根本不在「母集合」里。分离公理只能从已有集合中筛，而 $\omega+n$ 是「新造」的。替换公理把函数 $n \mapsto \omega+n$（一条公式）作用在 $\omega$ 上，先得到像集 $\{\omega+n : n<\omega\}$，再取并集才得到 $\omega \cdot 2$。**没有替换公理，序数就停不下来，也就没有 $\aleph_\omega$ 之后的基数层**。

## 4 良基公理、秩与累积层级

**正则（良基）公理**：每个非空集合都有 $\in$-极小元，

$$\forall x \bigl( x \ne \emptyset \rightarrow \exists y \in x \, (y \cap x = \emptyset) \bigr)$$

它排除了 $x \in x$、排除了无穷下降链 $\cdots \in x_2 \in x_1 \in x_0$，并保证每个集合都有**秩（rank）**：

$$\mathrm{rank}(x) = \sup \bigl\{ \mathrm{rank}(y) + 1 : y \in x \bigr\}$$

秩沿累积层级 $V_0 \subset V_1 \subset \cdots \subset V_\omega \subset V_{\omega+1} \subset \cdots$ 递增：$V_{\alpha+1} = \mathcal{P}(V_\alpha)$，极限步取并。**每个集合在某层 $V_\alpha$ 首次出现，秩就是它的「出生证号」**——基数篇用秩定义基数，力迫篇在「$V$ 之外造新集合」，全靠这张出生证。

## 5 核心对比表：ZFC 公理一网打尽

| 公理 | 类型 | 一句话作用 |
| --- | --- | --- |
| 外延公理 | 逻辑性 | 相等由元素决定 |
| 空集公理 | 构造性 | 保证宇宙非空 |
| 配对公理 | 构造性 | 造 $\{a,b\}$，有序对由此而来 |
| 并集公理 | 构造性 | 造 $\bigcup x$，序数加法靠它取极限 |
| 幂集公理 | 构造性 | 造 $\mathcal{P}(x)$，基数「变大」的唯一来源 |
| 无穷公理 | 构造性 | 造 $\omega$，有限与无限的界碑 |
| 分离公理（模式） | 模式 | 从母集合中筛选 |
| 替换公理（模式） | 模式 | 函数的像是集合 |
| 正则（良基）公理 | 结构性 | 排除病态自属与无穷下降链 |
| 选择公理 | 结构性 | 每个非空集族有选择函数（本篇第5篇有专文） |

这张表还要配一句行话：**ZFC 的公理不是铁板一块**。去掉正则公理、把选择公理换成较弱的依赖选择公理（DC），仍能得到自洽的系统；反过来，正则公理在其余公理的背景下是独立的。公理之间存在独立性，这正是第4篇「模型」概念要解释的现象：**两条命题之间无推论关系，当且仅当存在一个模型满足前者而不满足后者**。

## 6 动手推导：用分离公理定义差集与交集

分离公理的功能是「从母集合中筛」。用它给几个常见对象下定义，体会「公理化构造」的完整手感：

**差集**：$A \setminus B = \{ x \in A : x \notin B \}$。直接套分离公理：母集合 $A$，性质 $\varphi(x) \equiv x \notin B$，$B$ 作为参数。

**交集**：$\bigcap \mathcal{F} = \{ x \in \bigcup \mathcal{F} : \forall A \in \mathcal{F}\, (x \in A) \}$。妙处在于：先对 $\mathcal{F}$ 取并得到母集合 $\bigcup \mathcal{F}$，再用分离公理筛出「属于每个 $A$」的元素——并集公理与分离公理在这里第一次并肩作战。

**空集的唯一性**：由外延公理，任何两个空集相等；由分离公理，可从任意集合 $z$ 筛出 $\emptyset = \{x \in z : x \ne x\}$——不依赖「空」的直观。

**有序对的还原**：第一、第二坐标也能用分离定义。令 $\mathrm{fst}(z) = \bigcup \{\, x : \exists y\, (z = (x,y)) \,\}$，外层集合 $\{x : \exists y\,(z=(x,y))\}$ 由分离（母集合 $\bigcup\bigcup z$）+ 替换（当 $z$ 是「对的集合」时）筛出。这几行足以说明：**关系、函数、元组、笛卡尔积——全部是「并、幂、分离、替换」四件套的排列组合**。

**辨析｜易错点：** 差集公式里的参数 $B$ 必须是**已构造的集合**。真类（如全体序数 $\mathrm{On}$）不能充当分离公理中的参数——「类是谈资，集合才可运算」，这条界线在构造 $L$、力迫时还会反复出现。

## 7 小结

- ZFC 由十条公理构成，分**逻辑性、构造性、模式、结构性**四类；构造从空集与无穷起步，靠配对、并、幂逐层生长。
- **库拉托夫斯基有序对** $(a,b) = \{\{a\},\{a,b\}\}$ 说明「一切结构都能用集合造出来」。
- **分离公理只能筛，替换公理才能造**——没有替换公理就没有 $\{\omega+n:n<\omega\}$，序数算术卡壳。
- **正则公理**排除病态自属，给出**秩函数**与**累积层级** $V = \bigcup_{\alpha \in \mathrm{On}} V_\alpha$。
- 公理系统追求**自洽**而非自明；同一组公理可以有不同模型（第4篇）。

在下一节，我们将把自然数的「次序」推广到无穷：**序数与超限归纳、超限递归**——为什么「无限步之后的下一步」也有名字，von Neumann 如何用集合造出全部序数。
