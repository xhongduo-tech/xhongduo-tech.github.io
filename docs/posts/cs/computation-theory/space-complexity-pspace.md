---
title: 空间复杂性类与 PSPACE
date: 2026-08-07
---

# 空间复杂性类与 PSPACE

<div class="epigraph">
<p>空间很大，大到你无法想象它有多么浩瀚、无边无际。</p>
<footer>—— 道格拉斯 · 亚当斯（Douglas Adams, "The Hitchhiker's Guide to the Galaxy"）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算理论（可计算性与计算复杂性） ｜ Sipser《计算理论导引》第8章 ｜ 2026-08-07</p>
</div>

## 为什么把「空间」也当资源

时间不是唯一昂贵的资源——**内存**（纸带格子）同样有限。
一台机器可以「慢慢算」，但不能「内存不够」。
**空间复杂性（space complexity）** 度量「解决问题要占多少内存」。
它引出一系列与时间类**不对应**的美丽结果：非确定性在空间世界里几乎免费（Savitch 定理），而 PSPACE 完整问题——如量化布尔公式——难度横跨整座 NP 大厦。
<span class="marginnote">空间与时间的一个关键不对称：时间是「一次性消耗」（用完就没了），空间是「可复用」（格子能反复读写）。
这种复用性让空间类表现出与时间类迥异的封闭性质——例如非确定性空间只需平方代价即可确定化，而时间至今未找到这样的定理。
</span>

在「从极限到大模型」主线里，PSPACE 是「比 NP 更难一大截」的第一站：P ⊆ NP ⊆ PSPACE ⊆ EXPTIME，而 PSPACE 完整问题（一般化的棋类游戏）让你直观体会「指数时间都不一定够」。

## 1 空间复杂性的定义

**空间复杂性（space complexity）**：图灵机 $M$ 的空间复杂度 $s_M(n)$，是对长度为 $n$ 的输入，**读头访问过的纸带格子数的最大值**——**不计输入本身**（输入带是只读的，不算工作空间）。

**空间复杂性类**：

$$SPACE(f(n)) = \{ L \mid L \text{ 能被某台用 } O(f(n)) \text{ 空间判定的图灵机判定} \}$$

$$NSPACE(f(n)) = \{ L \mid L \text{ 能被某台用 } O(f(n)) \text{ 空间的 NTM 判定} \}$$

**重点：空间不计输入、计「额外工作区」。**
 这保证空间复杂性与「输入多长」解耦——读输入的成本不算，只算你要「额外记住多少」。
<span class="marginnote">空间计量的惯例：通常假设多带图灵机，输入带只读不计入空间。
这样像「判断输入是否为回文」这类需要常数空间的问题（两头指针往中间扫），空间复杂度是 $O(1)$，而不是 $O(n)$。
</span>

## 2 PSPACE 与其上的包含关系

**PSPACE** = 用多项式空间可判定的语言：

$$
\boxed{\;PSPACE = \bigcup_{k \ge 1} SPACE(n^k)\;}
$$

**定理（包含链）：**

$$
P \subseteq NP \subseteq PSPACE \subseteq EXPTIME = \bigcup_{k} TIME(2^{n^k})
$$

- $P \subseteq NP$：显然（判定器本身是验证器）。
- $NP \subseteq PSPACE$：NP 的暴力搜索「枚举所有证书并验证」可以用多项式空间完成——证书空间多项式，验证器空间多项式，逐条重放即可。
- $PSPACE \subseteq EXPTIME$：用 $f(n)$ 空间的机器，其**格局数**至多是 $f(n) \cdot |\Gamma|^{f(n)} = 2^{O(f(n))}$；若它超过这个步数还不停机，必然重复格局（死循环）。所以 $O(f(n))$ 空间的判定器必然在 $2^{O(f(n))}$ 时间内停机。<span class="marginnote">格局数的计算：读头位置 $f(n)$ 种 × 纸带内容 $|\Gamma|^{f(n)}$ 种 × 状态 $|Q|$ 种。空间换时间的硬约束——<strong>占多少空间，就必须在多少时间内停</strong>。这就是 PSPACE ⊆ EXPTIME 的原因。</span>

**哪些包含是严格的？**
 除了 $P \subsetneq EXPTIME$（由时间层次定理保证，见本系列第7课）外，其余（$P \stackrel{?}{=} NP$、$NP \stackrel{?}{=} PSPACE$）全都未知。
我们只知道 $P \neq EXPTIME$，而 PSPACE 卡在中间。

## 3 公式解析：Savitch 定理

空间世界里最反直觉的定理之一是 **Savitch 定理**：

$$
NSPACE(f(n)) \subseteq SPACE(f(n)^2), \qquad \text{对 } f(n) \ge \log n
$$

**非确定性的空间代价只是平方。**
 对比时间世界：NTIME 转 DTIME 的已知最好模拟仍是指数。
为什么空间如此慷慨？
因为——**格局可以复用**。

证明核心是**可到达性**：非确定机器 $M$ 用 $O(f(n))$ 空间接受 $w$，等价于「从起始格局 $c_1$ 能到达接受格局 $c_2$」。
定义函数

$$
CANREACH(c_1, c_2, t): \text{「} c_1 \xrightarrow{\le t \text{ 步}} c_2 \text{」}
$$

递归分裂：

$$
CANREACH(c_1, c_2, t) = \exists \text{ 中间格局 } c_m:\ CANREACH(c_1, c_m, t/2) \wedge CANREACH(c_m, c_2, t/2)
$$

- **第一步，读基例**：$t = 1$ 时直接查 $M$ 的转移表，一步可达。
- **第二步，读分裂**：要判断 $t$ 步内能否从 $c_1$ 到 $c_2$，只需找一个中间格局 $c_m$，让两半各在 $t/2$ 步内可达。**$c_m$ 不用存所有路径——只存一个格局。**
- **第三步，读出空间**：递归深度 $\log t = O(f(n))$（因为 $t$ 是格局总数 $2^{O(f(n))}$），每层存 $O(f(n))$ 空间的一个格局，总空间 $O(f(n)^2)$。**时间虽指数，空间只要平方。**

**重点：Savitch 定理告诉我们，对空间而言，「猜」几乎不增加能力。**
 这一结论没有任何时间类比，是空间理论独有的风景。

## 4 PSPACE 完全性：TQBF

**PSPACE 完全（PSPACE-complete）**：$B \in PSPACE$ 且一切 $A \in PSPACE$ 都有 $A \le_p B$。

最典型的 PSPACE 完全问题是**真量化布尔公式（TQBF）**：

$$TQBF = \{ \langle \phi \rangle \mid \phi \text{ 是全量化的布尔公式且为真} \}$$

其中 $\phi = \forall x_1\, \exists x_2\, \forall x_3 \cdots \psi(x_1, \dots, x_m)$，$\psi$ 是布尔公式。量词交替出现。

- **TQBF ∈ PSPACE**：递归求值——遇到 $\forall$ 分别代入 0、1 都要为真；遇到 $\exists$ 代入 0 或 1 任一为真。递归深度 $O(m)$，每层常数空间，总空间多项式。
- **TQBF 是 PSPACE 难的**：对任意 PSPACE 语言，用与 Cook-Levin 类似的**计算表编码**，把「从起始格局可达接受格局」写成量化公式——**量词用来表达「中间格局存在性」**，而 Savitch 的分裂思想让量词交替可控。<span class="marginnote">注意对比：SAT 的 $\exists$ 只有一行（存在一个赋值），TQBF 的 $\exists/\forall$ 交错出现——交替让公式能「逐层存在/任意地选择格局」，这正是 PSPACE 完整性的来源。量词交替越多，问题越难。</span>

**工程直觉**：TQBF 的实例——「给定局面，是否存在一步棋使对手无论怎么走我都能赢」——正是**完美信息博弈**的模型。PSPACE 完全性意味着**一般化的棋类游戏（可自动扩展棋盘）在多项式时间无解**，哪怕对手也认真下。

## 5 PSPACE 完整问题的味道

顺着 TQBF 的归约，一批「博弈类」问题落入 PSPACE 完全家族：

| 问题 | 描述 |
| --- | --- |
| $TQBF$ | 量化布尔公式可满足 |
| $FORMULA\text{-}GAME$ | 公式博弈（两玩家轮流赋值） |
| 广义 $GO$ / 广义国际象棋（受限版） | 棋盘任意大时的判定 |
| $GEOGRAPHY$ | 地理游戏（词首尾相接不重复） |

**重点：PSPACE 完全问题通常带「轮流」与「无论你如何应对」的味道。** 它们刻画的是「对抗性」难度，比 NP 的「存在性」更难一个量级。<span class="marginnote">有趣的是，固定棋盘大小的围棋、象棋是 PSPACE 甚至 EXPTIME 完全的判定版本，但「有限棋盘」本身可穷举——真正难的是「棋盘能变大」。这类问题提醒我们：<strong>规模任意增长，往往是把「可行」拖入「不可能」的那一刀</strong>。</span>

## 6 小结

- **空间复杂性**不计输入、只计额外工作区；$SPACE(f(n))$ 与 $NSPACE(f(n))$ 对应确定/非确定空间。
- **PSPACE** = $\bigcup_k SPACE(n^k)$；包含链 $P \subseteq NP \subseteq PSPACE \subseteq EXPTIME$。
- **Savitch 定理**：$NSPACE(f(n)) \subseteq SPACE(f(n)^2)$