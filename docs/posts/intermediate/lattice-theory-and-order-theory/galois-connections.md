---
title: Galois 联络与反变对偶
date: 2026-08-07
---

# Galois 联络与反变对偶

<div class="epigraph">
<p>数学里最深刻的对应，几乎都是 Galois 联络：一边的极大对象，映成另一边的极大对象。</p>
<footer>—— 埃瓦里斯特·伽罗瓦（Évariste Galois）思想的现代形式</footer>
</div>

<div class="article-byline">
<p>第二级 · 格论与序理论 ｜ Davey &amp; Priestley 第7章 ｜ 2026-08-07</p>
</div>

## 为什么从 Galois 联络开始

回顾全书，一条暗线贯穿始终：**两个结构之间的「反向对应」**——
第1篇闭包算子来自反序映射的复合，第3篇 Stone 对偶是反变等价，
第4篇的 Birkhoff 表示定理也是反变翻译。现在我们把这条暗线锻造成正式理论：
**Galois 联络（Galois connection）**。它用一条优美的等价式
$f(a) \le b \iff a \le g(b)$ 概括了「一对映射互为逆态」的关系，
是范畴论里伴随函子的雏形。伽罗瓦当年发现的「中间域 ↔ 伽罗瓦群子群」的对应，
正是第一个、也是最震撼的一个 Galois 联络。本节将系统建立 Galois 联络的机制，
为最后四节（代数格、表示论、Stone 对偶、Domain 理论）提供统一语言。
<span class="marginnote">Galois 联络的重要性怎么强调都不过分：模型论的「理论 ↔ 模型」、代数几何的「理想 ↔ 代数集」、线性代数的「子空间 ↔ 正交补」、乃至程序分析里的「抽象域 ↔ 具体域」，全是它的实例。学会 Galois 联络，等于掌握了一整套「双向翻译」的语法。</span>

## 1 保序 Galois 联络：伴随对

设 $(A, \le_A)$ 与 $(B, \le_B)$ 是偏序集，$f : A \to B$ 与 $g : B \to A$
是两个保序映射。若对一切 $a \in A$，$b \in B$：

$$f(a) \le_B b \iff a \le_A g(b)$$

则称 $(f, g)$ 是 **Galois 联络**，$f$ 称为**左伴随（lower / left adjoint）**，
$g$ 称为**右伴随（upper / right adjoint）**，记 $f \dashv g$。
<span class="marginnote">直觉：$f$ 与 $g$ 是一对「上下往复」的镜子——「$f(a)$ 不越过 $b$」当且仅当「$a$ 不越过 $g(b)$」。左伴随负责「向上加」，右伴随负责「向下收」；它们互相确定，一个知道对方的一切。</span>

**辨析｜易错点：** 两个映射方向不同步：$f : A \to B$、$g : B \to A$，
但 $f \dashv g$ 用「$\le$ 在两个方向」连接。左伴随与右伴随不对称：
**$f$ 保一切上确界（并），$g$ 保一切下确界（交）**。这条「伴随保确界」定理是
Galois 联络最锋利的工具——只凭 $f \dashv g$ 就自动得到
「并的像 = 像的并」等一整套性质。

## 2 反变 Galois 联络：经典伽罗瓦对应

历史上「Galois 联络」最初指**反序映射对**：$f : A \to B$，$g : B \to A$，且

$$b \le_B f(a) \iff a \le_A g(b)$$

（等价于：$f, g$ 都反序，且 $a \le gf(a)$、$b \le fg(b)$。）
这是经典 Galois 理论的形态——伽罗瓦群与中间域之间的对应就是这种反变联络。
<span class="marginnote">反变 Galois 联络 ⇄ 保序 Galois 联络可以互相转换：把 $B$ 换成对偶序 $B^{\partial}$，反序映射就变成保序映射。两种说法是同一硬币的两面，文献中「Galois connection」常混用二者，注意上下文。</span>

经典例子（伽罗瓦理论）：设 $E/F$ 是伽罗瓦扩张，$G = \operatorname{Gal}(E/F)$。

$$A = \{\text{中间域 } K : F \subseteq K \subseteq E\}, \qquad B = \{\text{子群 } H : H \le G\}$$

定义 $f(K) = \operatorname{Gal}(E/K)$（固定 $K$ 的伽罗瓦群），
$g(H) = E^{H}$（被 $H$ 逐点固定的中间域）。则 $(f, g)$ 是反变 Galois 联络：
$K_1 \subseteq K_2 \iff f(K_2) \le f(K_1)$，且 $g \circ f(K) \supseteq K$、
$f \circ g(H) \supseteq H$。

## 3 例子：遍布数学的反向对应

- **代数几何（Zariski 对应）**：$A = \mathbb{C}[x_1, \dots, x_n]$ 的理想，
  $B = \mathbb{C}^n$ 的代数集。$V(I)$ = $I$ 的零点集，$J(S)$ = 在 $S$ 上为零的
  多项式。这是 Galois 联络；**Hilbert 零点定理**说「$J(V(I)) = \sqrt{I}$」——
  闭包正好是根理想。
  <span class="marginnote">Zariski 对应是代数几何的起点：理想 ↔ 代数集的双向对应（连同零点定理给出的精确闭包刻画）把几何翻译成代数、把代数翻译成几何。这是「Galois 联络的闭包 = 根」的经典实例。</span>
**线性代数**：$A = B = V$ 的子空间格，$f(W) = W^{\perp}$（正交补）。
  $(^{\perp}, ^{\perp})$ 是反变 Galois 联络，闭包 $W^{\perp\perp}$ 给出闭子空间
  （有限维下 $W^{\perp\perp} = W$）。
**逻辑**：$A$ = 理论集合（按包含），$B$ = 模型类集合。
  $\operatorname{Mod}(\Gamma)$ = 满足 $\Gamma$ 的模型类，
  $\operatorname{Th}(\mathcal{K})$ = $\mathcal{K}$ 中所有模型共有的定理。
  这是 Galois 联络；闭包 $\operatorname{Th}(\operatorname{Mod}(\Gamma))$ =
  逻辑后承闭包。
  <span class="marginnote">「理论 ↔ 模型」的 Galois 联络把模型论的语义与语法织在一起：紧致性、完备性都可表述为这个联络的不动点/闭包性质。闭包 = 逻辑后承，这是「伽罗瓦联络」在逻辑里的同构化身。</span>

## 4 公式解析：$f(a) \le b \iff a \le g(b)$ 的四重含义

这条等价式是整个理论的心脏，逐层拆解：

$$f(a) \le b \iff a \le g(b)$$

- **第一步，读方向**：左伴随 $f$ 在「左边」出场（作为 $f(a)$），
  右伴随 $g$ 在「右边」出场（作为 $g(b)$）。式子的语义是：
  **$f$ 与 $g$ 互为「最优上/下界」**。
- **第二步，读三角等式**：由 $f(a) \le f(a)$ 与等价式推 $a \le g(f(a))$；
  由 $g(b) \le g(b)$ 推 $f(g(b)) \le b$。这两条「三角不等式」是 Galois 联络的
  等价刻画，也是所有闭包性质的来源。
- **第三步，读闭包**：复合 $gf : A \to A$ 满足 $a \le gf(a)$ 且 $gfgf = gf$
  （幂等），因而是 $A$ 上的**闭包算子**（第1篇）。闭元 $A_{gf} = \{a : gf(a) = a\}$
  与 $B_{fg}$ 互为同构的格——**Galois 联络在闭元处建立同构，
  这正是「对应」的本质**。
  <span class="marginnote">一切 Galois 联络的最终形态：两侧各取闭包后，得到的闭元格<strong>同构</strong>。伽罗瓦基本定理说的正是：若扩张是伽罗瓦的，则两侧闭包都是恒等（所有中间域与所有子群都是闭的），于是「中间域 ↔ 子群」是一一对应。闭元 = 不漏气的对象。</span>
**第四步，读范畴**：$f \dashv g$ 正是范畴论中**伴随函子（adjoint functor）**
  在偏序集（视为小范畴）上的退化形态。$\le$ 是「单态射」，$f$ 保持并、
  $g$ 保持交，对应「左伴随保余极限、右伴随保极限」。Galois 联络是
  「伴随」最老、最清晰的先行者。

## 5 从联络到闭包：工具箱

给定反变 Galois 联络 $(f, g)$，三件套自动成立：

1. $gf$ 与 $fg$ 都是闭包算子；
2. $f$ 把「$g$ 闭的并」映为「$f$ 闭的交」（保序版本：$f$ 保并、$g$ 保交）；
3. 不动点（闭元）格 $A_{gf} \cong B_{fg}$ 同构。

这套工具箱的用法：要证明「$A$ 与 $B$ 之间存在某种对应」，
只需构造一个 Galois 联络，再在闭元上取同构。
<span class="marginnote">实操模式：伽罗瓦理论证明「子群 ↔ 中间域」用「伽罗瓦联络 + 闭元 = 全体」；代数几何证明「根理想 ↔ 代数集」用「Zariski 联络 + 零点定理」。识别出联络，就找到了证明的骨架。</span>

## 6 求解联络：给定一个映射，找出它的伴随

Galois 联络的实践中，经常遇到「已知 $f$，求 $g$」或反之。
关键定理是：**伴随互相唯一确定**，且可以由公式给出。

**左伴随决定右伴随**：若 $f \dashv g$ 且 $f$ 保一切上确界，则

$$g(b) = \bigvee \{ a \in A : f(a) \le b \}$$

**右伴随决定左伴随**：若 $g$ 保一切下确界，则

$$f(a) = \bigwedge \{ b \in B : a \le g(b) \}$$

**例（取整与嵌入）**：设 $f : \mathbb{Z} \hookrightarrow \mathbb{R}$（嵌入）。
它的右伴随 $g : \mathbb{R} \to \mathbb{Z}$ 是什么？由 $f \dashv g$：
$f(n) \le r \iff n \le g(r)$，即 $n \le r \iff n \le g(r)$ 对一切整数 $n$，
故 $g(r) = \lfloor r \rfloor$。**嵌入 $\mathbb{Z} \to \mathbb{R}$
的右伴随是「向下取整」。**
<span class="marginnote">这个例子揭示了 Galois 联络的编程直觉：「嵌入」把整数塞进实数，它的右伴随「取整」把实数拉回整数——两个方向互逆（在闭元上）。Cousot 的抽象解释里，具体域与抽象域之间的 $\alpha \dashv \gamma$ 正是这种「嵌入 ↔ 取整」的推广。</span>

**对偶地**：若 $g : \mathbb{Z} \to \mathbb{R}$ 是嵌入，它的左伴随
$f : \mathbb{R} \to \mathbb{Z}$ 是「向上取整」$\lceil r \rceil$。
**左伴随「慷慨」（向上取整），右伴随「保守」（向下取整）**——
这条直觉对理解伴随的对偶性极有帮助。

**逻辑的 Galois 联络**：设 $A$ = 理论（命题集合）按包含，
$B$ = 模型类按包含。$\operatorname{Mod}(\Gamma)$ = $\Gamma$ 的模型类，
$\operatorname{Th}(\mathcal{K})$ = $\mathcal{K}$ 的所有模型共有的命题。
则 $(\operatorname{Mod}, \operatorname{Th})$ 是反变 Galois 联络：
$\Gamma_1 \subseteq \Gamma_2 \Rightarrow \operatorname{Mod}(\Gamma_2) \subseteq
\operatorname{Mod}(\Gamma_1)$，且 $\Gamma \subseteq \operatorname{Th}(\operatorname{Mod}(\Gamma))$。
闭包 $\operatorname{Th}(\operatorname{Mod}(\Gamma))$ = **逻辑后承闭包**——
$\Gamma$ 能推出的全部命题。**完备性定理说这个闭包恰好是「语法后承」：
$\Gamma \vdash \varphi \iff \varphi \in \operatorname{Th}(\operatorname{Mod}(\Gamma))$。**
<span class="marginnote">逻辑的 Galois 联络把语义（模型）与语法（推导）接成一对伴随。「完备性」= 闭包恰好是语法闭包；「紧致性」= 这个闭包只依赖有限子集。把整个模型论放进 Galois 联络的框架，是理解「语义-语法对偶」最清晰的路径。</span>

**练习**：证明「$f$ 保一切上确界 ⟺ $f$ 有右伴随」中的「⟸」方向：由
$f \dashv g$，取 $S \subseteq A$，证 $f(\bigvee S) = \bigvee f(S)$。
（提示：用等价式两边夹。）

## 7 小结

- **保序 Galois 联络**：$f \dashv g$，$f(a) \le b \iff a \le g(b)$；
  $f$ 保并、$g$ 保交。
- **反变 Galois 联络**：反序映射对；经典伽罗瓦理论、Zariski 对应、正交补、
  理论-模型都是实例。
- 三角不等式 $a \le gf(a)$、$f(g(b)) \le b$ 等价于联络定义。
- $gf$、$fg$ 是**闭包算子**；闭元格 $A_{gf} \cong B_{fg}$