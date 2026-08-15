---
title: Mal'cev 条件
date: 2026-08-07
---

# Mal'cev 条件

<div class="epigraph">
<p>簇的许多深刻性质，不取决于它的等式长什么样，只取决于某个项存不存在。</p>
<footer>—— 阿纳托利 · 马尔采夫（Anatoly Mal'cev），1954</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛代数（万有代数） ｜ Burris &amp; Sankappanavar《A Course in Universal Algebra》第III章 §7 ｜ 2026-08-07</p>
</div>

## 为什么从「Mal'cev 条件」继续

簇的语义在 HSP 定理里被算子锁定，等式逻辑的完备性又让语法与语义合一。但还有一个更深的问题悬着：**如何用「等式集」表达「元性质」？** 比如「这个簇的所有代数，同余都可换」——「可换」是关于同余格的性质，不是一条等式，怎么写进等式语言？1954 年 Mal'cev 给出了革命性的回答：某些元性质，等价于**某个三元项存在**。这类「存在一个项使得某组等式成立」的陈述，就是**Mal'cev 条件**。它让簇理论从「研究等式」跃升到「研究等式能捕捉的元性质」，是 20 世纪下半叶泛代数最重要的工作母机。

## 1 从元性质到项的存在

回忆同余格 $\operatorname{Con}\mathbf{A}$（前文《同余格》）：它是完备格，有交 $\wedge$ 与并 $\vee$。**同余可换（congruence permutable）**：代数 $\mathbf{A}$ 的同余 $\theta, \psi$ 满足

$$\theta \circ \psi = \psi \circ \theta$$

其中 $\circ$ 是关系复合：$(a,b) \in \theta \circ \psi$ 当且仅当存在 $c$ 使 $a \,\theta\, c$ 且 $c \,\psi\, b$。对等价关系，$\theta \circ \psi = \psi \circ \theta$ 恰好等价于 $\theta \circ \psi$ 仍是等价关系（即 $\theta \vee \psi = \theta \circ \psi$）。<span class="marginnote">在群、环这类「有减法」的结构里，同余自动可换——这正是为什么群论里从未需要这个概念。可换性被破坏的反常结构出现在没有「减」的语言里，Mal'cev 项的使命就是为「减」提供一种替身。</span>

Mal'cev 的观察：**「同余可换」不是一个等式的属性，而是一个项的存在性**。

**Mal'cev 项**：三元项 $p(x, y, z)$，满足

$$p(x, y, y) \approx x, \qquad p(y, y, x) \approx x$$

即「$p$ 在第二、三参数相同时消掉中间元」。群里的 $p(x,y,z) = x \cdot y^{-1} \cdot z$ 是 Mal'cev 项（$x y^{-1} y = x$，$y y^{-1} x = x$）；格里的三元多数运算 $\mathrm{m}(x,y,z) = (x \wedge y) \vee (y \wedge z) \vee (z \wedge x)$ 满足 $\mathrm{m}(x,x,z) \approx \mathrm{m}(x,z,x) \approx \mathrm{m}(z,x,x) \approx x$，但**不是** Mal'cev 项——多数运算要求三个参数中至少两个相等才消元，Mal'cev 项只要求第二、三或第一、二相等，更宽松也更强。

为什么「存在项」能等价于「每个代数都怎样」？直觉是：簇里的等式是「语言内部」的陈述，项的复合最终落在语言内部；而「每个代数的同余可换」是「关于语言外部所有结构」的陈述。Mal'cev 条件断言，在某些元性质上两者等价——这是模型论意义下「可定义性」的代数版本。

## 2 Mal'cev 定理

**定理（Mal'cev, 1954）**：设 $\mathcal{V}$ 是簇，则下列等价：

1. $\mathcal{V}$ 中存在一个 Mal'cev 项 $p$（即存在三元项使 $p(x,y,y) \approx x$ 与 $p(y,y,x) \approx x$ 在 $\mathcal{V}$ 中成立）；
2. $\mathcal{V}$ 中**每个**代数都是同余可换的；
3. $\mathcal{V}$ 中每个代数的同余格，其并运算可「显式化」：$\theta \vee \psi = \theta \circ \psi$。

这个定理开创了整整一个分支。**重点：Mal'cev 条件把「逐代数的性质」与「整个簇的性质」等价起来。** 条件 2 是关于每个成员的陈述，条件 1 是一个全局的项——从「每个」到「一个」，是谓词逻辑到代数语言的一次压缩。<span class="marginnote">这类等价常被概括为「局部 = 全局」：簇里要是每个代数都怎样，那么整个簇用一个项就能保证它。马尔采夫定理是所有 Mal'cev 条件的鼻祖，其后每个同类条件都沿袭它的名字。</span>

证明要点：$(2) \Rightarrow (1)$ 用自由代数 $\mathbf{F}_\mathcal{V}(x, y, z)$——取三个自由生成元，令 $\theta = \operatorname{Cg}(x, y)$（由 $x \sim y$ 生成的同余）、$\psi = \operatorname{Cg}(y, z)$；可换性给出 $(x, z) \in \theta \circ \psi$，即存在项 $p$ 使 $p(x,y,z)$ 同时满足两条 Mal'cev 等式。$(1) \Rightarrow (2)$ 则用 $p$ 显式构造「换位路径」：若 $a \,\theta\, c$ 且 $c \,\psi\, b$，用 $p(a, c, b)$ 验证 $(a, b) \in \psi \circ \theta$。

证明还有一个值得记住的副产品：$(1) \Rightarrow (2)$ 方向的构造是**显式**的——给定 Mal'cev 项 $p$，任意一条换位路径都可以用 $p$ 现场焊接。所以同余可换不只是「存在」，而是「可计算地存在」；这一显式性在算法化代数（如计算机代数里的同余计算）里直接可用。

## 3 同余模与同余分配

Mal'cev 条件不是孤例，而是模板。同余格上另外两个著名的元性质同样有项刻画：

**同余模（congruence modular）**：每个 $\operatorname{Con}\mathbf{A}$ 都是模格。**Day 项**：存在四元项 $m(x,y,z,w)$ 满足四组等式，刻画同余模簇。<span class="marginnote">模格的「模律」$x \leq z \Rightarrow x \vee (y \wedge z) \leq (x \vee y) \wedge z$ 是比分配律弱、但比任意格强的正则性条件。群、环、模、格——几乎所有经典结构——的同余格都模，这解释了为什么同余模条件覆盖面极广。</span>
**同余分配（congruence distributive）**：每个 $\operatorname{Con}\mathbf{A}$ 都是分配格。**Jónsson 项**：存在四元项 $j(x,y,z,w)$，其刻画是经典课题；**Gumm 项**刻画同余模 + 可换的细化。<span class="marginnote">分配性最强的著名簇是<strong>格本身</strong>：格的同余格必是分配格。反过来，「同余分配簇」里的代数某种程度上都「像格」。Mal'cev 条件因此成了连接泛代数与格论的标准桥梁。</span>

下表汇总三条经典 Mal'cev 条件：

| 元性质 | 判别项 | 项的元数 | 典型簇 |
| --- | --- | --- | --- |
| 同余可换 | Mal'cev 项 $p$ | 3 | 群、环、模 |
| 同余模 | Day 项 $m$ | 4 | 群、环、模、格 |
| 同余分配 | Jónsson 项 $j$ | 4 | 格、布尔代数、相对补格 |

## 4 公式解析：Mal'cev 项两条等式

把判别式本身拆开，它比表面更精巧：

$$
p(x, y, y) \approx x, \qquad p(y, y, x) \approx x
$$

- **第一步，看第一式**：当第二、三参数相等时，$p$ 返回第一参数。$p$ 把「第二个位置与第三个位置重合」当作信号，忽略被重复的值 $y$。
- **第二步，看第二式**：当第一、二参数相等时，$p$ 返回第三参数。两条等式合起来覆盖「三参中有一对相邻相等」的情形。
- **第三步，看缺位**：注意**没有**要求 $p(x, y, x) \approx x$（第一、三参数相等的情形）。这与多数运算形成关键对比——多数运算对「隔一个相等」也消元，Mal'cev 项不需要，这正是它更强、更通用的原因。
- **第四步，看群例**：$p = x \cdot y^{-1} \cdot z$ 时，第一式给 $x y^{-1} y = x$，第二式给 $y y^{-1} x = x$，两条都靠「元素乘自己的逆得单位元」。Mal'cev 项把「减法」压缩成一个纯等式形的替身——同余可换的本质就是「处处可以减」。

## 5 辨析｜易错点：Mal'cev 条件是存在性陈述

**辨析｜易错点：** Mal'cev 条件有几个反直觉处：

- **它不是一个等式，是「存在项使等式成立」。** 判别式写的是「存在 $p$ 使得……」，而不是直接把 $p(x,y,y) \approx x$ 当公理。簇 $\mathcal{V}$ 满足条件，指的是 $\mathcal{V}$ 的**某个项**满足这些等式——项是由 $\mathcal{V}$ 的类型生成的语法对象，不是额外加进来的运算符号。
- **项的存在与簇相关**：同样的三元项在甲簇是 Mal'cev 项、在乙簇可能不是——由两个簇各自的等式理论决定。判别式是「关于簇」的断言。
- **Mal'cev 条件不保证「唯一」**：一个簇里可能有多个 Mal'cev 项，满足条件的是存在性而非唯一性。
- **同余可换 ≠ 同余模**：可换对应 Mal'cev 项（三元），模对应 Day 项（四元）——条件不同，项不同，不可混为一谈。混淆它们会在后续读文献时立刻翻车。

## 6 数值算例：用「减法」把同余可换走通

在整数加法群 $\mathbb{Z}$ 上，Mal'cev 项就是减法 $p(x,y,z) = x - y + z$：$p(x,y,y) = x - y + y = x$，$p(y,y,x) = y - y + x = x$，两条等式都成立。

现在验证同余可换。取同余 $\theta$ 为「模 2」、$\psi$ 为「模 3」。任取 $a \,\theta\, c$ 且 $c \,\psi\, b$，例如 $a = 0, c = 2, b = 5$（$0 \equiv 2 \pmod 2$，$2 \equiv 5 \pmod 3$）。要证 $(a,b) \in \psi \circ \theta$，即找 $d$ 使 $a \,\psi\, d$ 且 $d \,\theta\, b$。取 $d = p(a,c,b) = 0 - 2 + 5 = 3$：$0 \equiv 3 \pmod 3$ 成立，$3 \equiv 5 \pmod 2$ 成立（两者都 $\equiv 1$）。于是 $(0,5) \in \psi \circ \theta$，可换性在这条路上被「算」了出来。

这就是证明要点 $(1) \Rightarrow (2)$ 的数值切片：**中间元素 $d$ 由 Mal'cev 项现场造出**，不需要任何搜索——项就是那个「万能中转站」。

把 $\mathbb{Z}$ 换成任意群、任意环，同一套「减法」都成立——这就是为什么 Mal'cev 项在群、环、模这些「有减」的结构里无处不在，而「没有减」的格类只在特殊子类里才可换。

## 7 术语速查表

| 术语 | 含义 | 一句话辨析 |
| --- | --- | --- |
| 同余可换 | $\theta \circ \psi = \psi \circ \theta$ | 可换 ⇔ $\theta \vee \psi = \theta \circ \psi$ |
| Mal'cev 项 | $p(x,y,y)\approx x$、$p(y,y,x)\approx x$ | 「减法」的等式替身 |
| Day 项 | 四元项刻画同余模 | 模律的项化 |
| Jónsson 项 / Gumm 项 | 四元项刻画同余分配等 | 分配性的项化 |
| 判别函数 $d(x,y,z)$ | $x\neq y$ 取 $x$，否则取 $z$ | 是 Mal'cev 项 |
| 多数函数 $m$ | 取三参中至少出现两次的值 | 不是 Mal'cev 项 |
| 关系复合 $\circ$ | $(a,b)\in\theta\circ\psi \iff \exists c:\ a\,\theta\,c\,\psi\,b$ | 可换性检验的基本操作 |

## 8 小结

- **Mal'cev 条件** = 用「存在某个项满足某组等式」来刻画簇的元性质。
- **Mal'cev 定理**：簇同余可换 ⟺ 存在 Mal'cev 项 $p(x,y,y) \approx x$、$p(y,y,x) \approx x$。
- **Day 项**刻画同余模簇，**Jónsson 项 / Gumm 项**刻画同余分配及相关性质。
- 同余可换的代数里，并可显式化：$\theta \vee \psi = \theta \circ \psi$。
- Mal'cev 项是「局部 = 全局」的引擎：簇里每个代数可换，就有一个全局的项把换位路径当场造出。
- Mal'cev 项的换位路径由 $p$ 现场显式焊接，「可换」不只是存在、而是可计算地存在。

在下一节，我们把镜头从「簇能用项捕捉哪些元性质」转向「单个代数能表达多少函数」——这就是**判别性（primality）引论**。