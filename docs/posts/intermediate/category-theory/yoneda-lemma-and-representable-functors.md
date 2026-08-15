---
title: Yoneda 引理与可表函子
date: 2026-08-07
---

# Yoneda 引理与可表函子

<div class="epigraph">
<p>Yoneda 引理是范畴论唯一重要的定理。</p>
<footer>—— 依据桑德斯 · 麦克莱恩（Saunders Mac Lane）</footer>
</div>

<div class="article-byline">
<p>第二级 · 范畴论 ｜ Riehl Ch. 2-3 ｜ 2026-08-07</p>
</div>

## 为什么从 Yoneda 开始

学过线性代数的人都知道「认识一个向量空间，最好认识它上面的线性泛函」。范畴论把这条经验推到极致：**认识一个对象 $a$，最好认识从任何对象到 $a$ 的所有态射**。Yoneda 引理说——这不止是「好」，而是「完全」：一个对象 $a$ 在范畴内的全部信息，等价于它到其他对象的所有态射之集合。它是整个范畴论最核心、最出人意料的定理，也是从普适性质通向「可表函子」的桥：**乘积、张量积、自由群之所以能由普适性质唯一刻画，全都因为它们是可表函子的代表对象**。<span class="marginnote">现代机器学习里「用函数（特征）认识数据点」「用表示认识对象」的思想，与 Yoneda 引理共享同一种结构：不直接看对象本身，而是看它与其他对象的关系图谱。</span>

## 1 Hom 函子与预层

对固定对象 $a$，定义**（逆变）Hom 函子**

$$\mathrm{Hom}(-, a): \mathbf{C}^{\mathrm{op}} \to \mathbf{Set}, \qquad x \mapsto \mathrm{Hom}(x, a)$$

以及协变的 $\mathrm{Hom}(a, -)$。一个（反变）函子 $\mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$ 称为**预层（presheaf）**——它是「把 $a$ 的全体关系收集起来」的泛化。<span class="marginnote">预层的名字来自几何：拓扑空间上的「开集范畴」到集合的函子就是常义预层；这里我们先在纯范畴层面理解它，下一节拓扑斯会正式登场。</span>**可表函子（representable functor）**：与某个 $\mathrm{Hom}(-, a)$ 自然同构的预层，其代表对象为 $a$。可表 = 被普适性质完全决定。

**预层与可表函子，一张对照表：**

| | 一般预层 $F$ | 可表预层 $\mathrm{Hom}(-, a)$ |
| --- | --- | --- |
| 形状 | 任意 $\mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$ | 固定的 Hom 函子 |
| 数据量 | 每个 $x$ 一个集合 | 由单个对象 $a$ 决定 |
| 可表性 | 未必 | 是（代表对象 $a$） |
| 与普适性质 | 无关 | 一一对应 |

## 2 Yoneda 引理

**Yoneda 引理**：设 $F: \mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$ 是任意预层，$a \in \mathbf{C}$，则「从 $\mathrm{Hom}(-, a)$ 到 $F$ 的自然变换全体」与「$F a$」之间存在自然同构：

$$
\mathrm{Nat}(\mathrm{Hom}(-, a), F) \cong F a
$$

它说了两件大事：

1. **Yoneda 嵌入** $y: \mathbf{C} \to \mathbf{Set}^{\mathbf{C}^{\mathrm{op}}}$，$a \mapsto \mathrm{Hom}(-, a)$，是**满忠实**的——对象 $a$ 被它的 Hom 函子完全决定。
2. **推论（保序等价）**：$\mathrm{Hom}(-, a) \cong \mathrm{Hom}(-, b)$ 当且仅当 $a \cong b$——**态射图谱完全决定对象**。

**数值算例：把双射在最小范畴里走一遍。** 取 $\mathbf{C}$ 为「含两个对象 $a, b$ 与唯一非平凡态射 $f: a \to b$」的范畴。看预层 $F = \mathrm{Hom}(-, a)$ 本身：$F a = \{1_a\}$、$F b = \emptyset$。$\mathrm{Nat}(\mathrm{Hom}(-,a), F)$ 只有一个自然变换 $\alpha$：$\alpha_a$ 把 $1_a$ 送到 $F a$ 中唯一元素，故 $\Phi(\alpha) = \alpha_a(1_a) = 1_a$——双射两侧都只有一个元素，平凡但完整。换 $F$：$F a = \{x\}$、$F b = \{y\}$、$F(f)(x) = y$，则 $\mathrm{Nat}(\mathrm{Hom}(-,a), F)$ 仍只有一个元素（由 $\alpha_a(1_a) = x$ 决定，$\alpha_b$ 被迫取 $\emptyset \to F b$ 的唯一空映射）——与 $F a = \{x\}$ 一一对应。Yoneda 在最小例子里也逐字成立。

**辨析｜易错点：** Yoneda 引理不是「$a$ 决定了 $\mathrm{Hom}(-,a)$」这种平凡方向，而是反过来的**内射性**：$\mathrm{Hom}(-,a)$ 的同构类唯一决定 $a$。另注意方向——预层是反变的，$\mathrm{Hom}(a,-)$ 的版本有同样的结论但作用于协变情形。

## 3 公式解析：Yoneda 双射怎么「造」

引理最迷人的地方在于：它的同构映射几乎是「免费」的。设 $\alpha: \mathrm{Hom}(-, a) \Rightarrow F$ 是一个自然变换，定义

$$
\Phi(\alpha) = \alpha_a(1_a) \in F a
$$

反过来，给定 $u \in F a$，对每个 $x$ 和每个 $f: x \to a$ 定义

$$
(\Psi(u))_x(f) = F f(u) \in F x
$$

- **第一步，正向来**：自然变换 $\alpha$ 在 $a$ 处作用在恒等态射 $1_a$ 上，得到一个元素 $\alpha_a(1_a) \in F a$——**「点」从自然变换中被压出来**。
- **第二步，反向来**：给定元素 $u$，对任意态射 $f$，把它沿 $F f$ 推到目标集 $F x$——**一个自然变换被「拉」出来**。
- **第三步，互逆**：$\Psi(\Phi(\alpha)) = \alpha$ 靠自然性交换图；$\Phi(\Psi(u)) = u$ 靠取 $f = 1_a$。两条方向完美闭合。
- **第四步，为什么深刻**：这个构造对任意 $F$、任意 $a$ 都自然成立，且证明只用到了定义本身——**「关系决定本质」不是信仰，而是可构造的算法**。

**为什么这个双射「免费」。** 关键在于 $\mathrm{Hom}(-, a)$ 是「被 $1_a$ 生成的」：它在 $a$ 处有且仅有一个「自持」元素 $1_a$，一切其他 $\mathrm{Hom}(x, a)$ 的元素都形如「$1_a$ 沿某个 $f$ 推出」。于是自然变换完全由「$1_a$ 去哪」决定，而「去哪」恰是 $F a$ 的元素。Yoneda 的证明之短，正因为它把「所有关系」压缩进了一个点。

## 4 可表函子与普适性质

Yoneda 引理把普适性质语言统一成了「可表性」：

- **乘积** $a \times b$：函子 $x \mapsto \mathrm{Hom}(x, a) \times \mathrm{Hom}(x, b)$ 的代表对象——存在唯一 $\langle x_1, x_2\rangle$ 正是「可表」的内容。
- **张量积** $A \otimes B$：函子「双线性映射 $A \times B \to -$」的代表对象——普适双线性映射即代表对象给出的泛态射。<span class="marginnote">上一节所有「普适构造存在且唯一」的结果，都可以翻译成「某个函子可表，且代表对象在同构意义下唯一」——这是 Yoneda 的工程价值。</span>
**自由对象**：遗忘函子有左伴随，等价于「$\mathrm{Hom}(X, U-)$ 可表」。

**把一个可表性检查做到底。** 对乘积 $a \times b$，定义函子 $\mathrm{Hom}(x, a) \times \mathrm{Hom}(x, b)$。它可表，当且仅当存在 $p$ 使 $\mathrm{Hom}(x, p) \cong \mathrm{Hom}(x,a) \times \mathrm{Hom}(x,b)$ 自然于 $x$——取 $x = p$ 并看恒等，得到两条投影 $\pi_1, \pi_2$；「存在唯一分解」正是这条自然同构对每个 $x$ 的逐点内容。**可表 = 用普适性质定义对象**在此被完整兑现。

由此，**定义可表的函子 = 提出普适性质**，这是现代数学「用普适性质定义对象」的总语法。

## 5 应用：从 Cayley 到特征嵌入

- **群论里的原型**：Cayley 定理「每个群同构于置换群的子群」就是 Yoneda 嵌入在单对象范畴（群）上的实例——群通过左乘作用认识自身。
- **范畴的稠密性（density theorem）**：每个预层都是可表预层的（加权）余极限，即「一切数据都由关系碎片拼成」——这是 Yoneda 的直接推论，也是下一节 Kan 扩张的伏笔。<span class="marginnote">机器学习里的「表征学习」把对象映到向量，本质上是把「关系图谱」（共现、相似度）压进低维表示——Yoneda 视角提醒我们：好的表示应当让「表示之间的态射结构」忠实地反映「对象之间的态射结构」。</span>
- **范畴的元素（category of elements）**：把 $F: \mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$ 的可表性检验转化为在配对范畴上找初对象——Riehl 的教材从这条路线讲可表函子，构造更显式。
- **幺半群的 Cayley**：单对象范畴 $\mathcal{B}G$ 上，Yoneda 嵌入把唯一对象映到「左乘表示」$\mathrm{Hom}(-, \ast)$——Cayley 定理「群同构于置换群的子群」正是 Yoneda 嵌入在这个范畴上的实例。

**给 ML 读者一句话。** 特征嵌入 = 把「关系图谱」压进向量；Yoneda 提醒我们：**好的表示应让「表示间的结构」忠实反映「对象间的结构」**。「代表对象唯一决定一切关系」正是「一个良好的表示包含该对象全部信息」的数学原型。

## 6 术语速查表

| 术语 | 英文 | 一句解释 |
| --- | --- | --- |
| Hom 函子 | hom-functor | $\mathrm{Hom}(-, a): \mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$ |
| 预层 | presheaf | $\mathbf{C}^{\mathrm{op}} \to \mathbf{Set}$ 的函子 |
| 可表函子 | representable functor | 与某个 Hom 函子自然同构 |
| 代表对象 | representing object | 可表函子对应的对象 $a$ |
| Yoneda 引理 | Yoneda lemma | $\mathrm{Nat}(\mathrm{Hom}(-,a), F) \cong F a$ |
| Yoneda 嵌入 | Yoneda embedding | $a \mapsto \mathrm{Hom}(-, a)$，满忠实 |
| 稠密性 | density theorem | 每个预层 = 可表预层的余极限 |
| 范畴的元素 | category of elements | 配对 $(c, x \in F c)$ 的范畴 |
| 满忠实 | fully faithful | 嵌入在每个 Hom 集上双射 |
| 保序等价 | representable criterion | $\mathrm{Hom}(-,a) \cong \mathrm{Hom}(-,b) \iff a \cong b$ |

## 7 小结

- **Yoneda 引理**：$\mathrm{Nat}(\mathrm{Hom}(-,a), F) \cong F a$，自然于 $a$ 与 $F$。
- **Yoneda 嵌入满忠实**：对象被 Hom 函子的同构类唯一决定；$\mathrm{Hom}(-,a) \cong \mathrm{Hom}(-,b) \iff a \cong b$。
- 双射由「在恒等上取值」与「沿态射推进」显式给出。
- **可表函子** = 普适性质的统一语法：乘积、张量积、自由对象都是可表。
- 稠密性：每个预层都是可表层的余极限——关系碎片足以拼出全体。

在下一节，我们将 Yoneda 的余极限视角发挥到底：把「沿嵌入拼出预层」推广成一般范畴间的「沿函子搬运」，这就是**Kan 扩张**——Mac Lane 说，一切概念都是 Kan 扩张。
