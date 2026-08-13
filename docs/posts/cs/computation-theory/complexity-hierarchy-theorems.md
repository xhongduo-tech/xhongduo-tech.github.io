---
title: 复杂性层次定理与类间关系
date: 2026-08-07
---

# 复杂性层次定理与类间关系

<div class="epigraph">
<p>底层有广阔的空间。</p>
<footer>—— 理查德 · 费曼（Richard Feynman, "There's Plenty of Room at the Bottom", 1959）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算理论（可计算性与计算复杂性） ｜ Hopcroft, Motwani & Ullman《自动机理论、语言和计算导论》第9章 ｜ 2026-08-07</p>
</div>

## 为什么需要层次定理

前面说了无数「未知」——P 是否等于 NP、NL 是否等于 L。
但计算理论**并非全是未知**：有一些类我们**能证明**不同。
**层次定理（hierarchy theorems）** 就是这样的「确定分离」：**给机器更多时间或空间，它一定能判定更多语言。**
 这是对角线法在复杂性世界的第二次胜利。
<span class="marginnote">第2篇的对角线法证明了「存在不可判定问题」；
层次定理把同一手法改造成「存在需要更多资源的可判定问题」——不是「不可能」，而是「要多花资源」。
工具的相似性正是图灵式对角线的通用性。
</span>

在课程主线上，层次定理提供了稀缺的**确定性事实**：$P \neq EXPTIME$、$L \neq PSPACE$ 等严格包含。
它们是复杂性的「已知疆域」，让 P vs NP 这类「未知疆域」显得格外醒目。

## 1 时间构造函数

层次定理要求时间/空间函数是「良性的」——能自己算出自己的值。

**时间可构造（time-constructible）**：函数 $f: \mathbb{N} \to \mathbb{N}$ 是时间可构造的，如果存在一台图灵机，对输入 $1^n$ 恰好用 $O(f(n))$ 步后停机，输出 $f(n)$ 的二进制表示。

**重点：几乎所有「正常」函数都是可构造的**——$n^k$、$2^n$、$n \log n$、$n!$ 都是。
可构造性是为了让机器能「给自己上闹钟」：跑 $f(n)$ 步后强制停机。
不可构造的函数会让「时钟」无法实现，层次定理就失效了。
<span class="marginnote">直觉：机器需要知道「$f(n)$ 是多少」才能在恰好的预算内模拟并计时。
可构造函数可以现场算出预算；
不可构造的函数（罕见构造品）连「多给资源」都无从谈起。
</span>

## 2 时间层次定理

**时间层次定理（time hierarchy theorem）：** 若 $f, g$ 是时间可构造函数，且

$$f(n) \log f(n) = o(g(n))$$

则

$$TIME(f(n)) \subsetneq TIME(g(n))$$

即：**给足够多的时间，一定能判定更多语言。**
<span class="marginnote">那个 $\log f(n)$ 因子不是技术噪声，而是<strong>模拟开销</strong>：定理用「通用机模拟任意机器」，而通用模拟要付出 $\log$ 倍代价（编码机器、解码转移）。
所以只多一点点时间不够，要多一个 $\log$ 因子才行。
</span>

**推论（重要）：$P \neq EXPTIME$。**
 因为 $TIME(n^k) \subsetneq TIME(2^n)$ 对一切 $k$ 成立（$n^k \log n = o(2^n)$），而 $EXPTIME = \bigcup_k TIME(2^{n^k})$。
**多项式时间与指数时间之间，存在严格断层。**

## 3 公式解析：层次定理的证明骨架

层次定理的证明是对角线法的「资源受限」版本。
假设要证明 $TIME(f(n)) \subsetneq TIME(g(n))$，构造一个「反着做」的机器 $D$：

$$
D(\langle M \rangle):\ \text{用 } O(g(n)) \text{ 时间模拟 } M(\langle M \rangle)；\ \text{若 } M \text{ 接受则拒绝，若 } M \text{ 拒绝则接受；超时则接受}
$$

逐项拆解：

- **第一步，读「模拟 + 时钟」**：$D$ 先模拟 $M$ 在 $\langle M \rangle$ 上的运行，但只跑 $O(g(n))$ 步——超时强制停机（这就是时间可构造性的用处：$D$ 能算出自己的预算）。$n = |\langle M \rangle|$。
- **第二步，读「反着做」**：$M$ 接受则 $D$ 拒绝，$M$ 拒绝则 $D$ 接受。若 $M$ 在 $TIME(f(n))$ 内（比 $D$ 的预算少一个 $\log$ 因子），$D$ 的模拟必然在超时前看到 $M$ 的答案。
- **第三步，读出矛盾**：假设 $D \in TIME(f(n))$，取 $M = D$。$D$ 接受 $\langle D \rangle$ 当且仅当 $D$ 拒绝 $\langle D \rangle$——矛盾。故 $D \notin TIME(f(n))$，但 $D \in TIME(g(n))$（模拟 + 反做都在预算内）。**$D$ 就是「多要时间才判得了」的那个语言。**

## 4 空间层次定理

**空间层次定理（space hierarchy theorem）：** 若 $f, g$ 是空间可构造函数，且

$$f(n) = o(g(n))$$

则

$$SPACE(f(n)) \subsetneq SPACE(g(n))$$

注意空间定理**没有 $\log$ 因子**——空间模拟不需要编码机器的 $\log$ 开销，读头位置直接记录即可。
<span class="marginnote">时间的模拟开销是 $\log$（因为要解释程序编码），空间的模拟开销是常数（因为格局直接落在纸带上）。
这解释了为什么空间层次比时间层次「更干净」——多一点点空间就严格变强。
</span>

**推论：$L \neq PSPACE$。**
 因为 $\log n = o(n)$，故 $SPACE(\log n) \subsetneq SPACE(n) \subseteq PSPACE$。
**对数空间与多项式空间之间，也存在严格断层。**

## 5 类间关系的「已知」与「未知」

把层次定理的确定结果与其他类间关系对照：

**已证明的严格包含：**

- $P \subsetneq EXPTIME$（时间层次）
- $L \subsetneq PSPACE$（空间层次）
- 更多：$TIME(n) \subsetneq TIME(n^2) \subsetneq \cdots$，$SPACE(n) \subsetneq SPACE(n^2) \subsetneq \cdots$

**未解决的包含（全部开放）：**

- $P \stackrel{?}{=} NP$，$NP \stackrel{?}{=} PSPACE$，$L \stackrel{?}{=} NL$，$NL \stackrel{?}{=} P$……

**已知的包含链（可能不严格）：**

$$
L \subseteq NL \subseteq P \subseteq NP \subseteq PSPACE \subseteq EXPTIME
$$

**重点：层次定理只分离「资源相差很大的类」，分不开「资源相近的类」。**
 $P$ 与 $NP$ 的资源上限都是「多项式」——层次定理管不着，因为从 $TIME(n^k)$ 到 $TIME(n^{k+1})$ 可以严格增长，但 $NP$ 是「非确定多项式」，不落在任一确定的 $TIME(n^k)$ 里。

**辨析｜易错点：** 层次定理证明的是「存在某个语言要多花资源」，不是「所有问题都要多花资源」。
$D$ 是精心构造的刁钻语言，不代表 SAT 就一定不在 P 里。
**层次定理分离的是「资源档位」，不是「具体问题」。**

## 6 为什么层次定理有边界

层次定理不能解决 P vs NP，有一个深刻原因：**对角化构造本质上是「模拟 + 反做」，它只利用机器的「程序性」属性，而不利用问题的「结构」**。
但 P vs NP 的困难恰恰在于——NP 完全问题的求解需要理解问题的结构（如 SAT 的赋值结构），这不是通用模拟能绕过的。
<span class="marginnote">事实上，有相对化（oracle）证据表明：存在某个 oracle $A$ 使 $P^A = NP^A$，也存在 oracle $B$ 使 $P^B \neq NP^B$（Baker-Gill-Solovay, 1975）。
这说明「纯对角化」式证明<strong>不可能</strong>解决 P vs NP——因为它对 oracle 也成立。
</span>要证明 P ≠ NP，必须用非相对化的、利用问题结构的新方法。

**重点：层次定理的证明「相对化」（对 oracle 也成立），而 P vs NP 的答案不能相对化。**
 这是理论界的经典洞见：它判了「纯对角线法」的死刑，逼着研究者寻找更深刻的工具。

## 7 小结

- **时间可构造**：$f$ 能被机器在 $O(f(n))$ 内算出；一切正常函数皆可构造。
- **时间层次定理**：$f(n)\log f(n) = o(g(n))$ ⟹ $TIME(f(n)) \subsetneq TIME(g(n))$；推论 **$P \neq EXPTIME$**。
- **空间层次定理**：$f(n) = o(g(n))$ ⟹ $SPACE(f(n)) \subsetneq SPACE(g(n))$；推论 **$L \neq PSPACE$**。
- 层次定理靠**资源受限的对角化**：模拟 + 时钟 + 反做。
- **已知**：$P \subsetneq EXPTIME$、$L \subsetneq PSPACE$