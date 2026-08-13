---
title: 分配格的判据与 Dedekind 定理
date: 2026-08-07
---

# 分配格的判据与 Dedekind 定理

<div class="epigraph">
<p>一个格是分配格，当且仅当它的定律是「等式中件件可消」的。</p>
<footer>—— 理查德·戴德金（Richard Dedekind）</footer>
</div>

<div class="article-byline">
<p>第二级 · 格论与序理论 ｜ Birkhoff 第3章 ｜ 2026-08-07</p>
</div>

## 为什么从分配格的判据开始

上一节我们认识了分配格与模格，但「如何判断一个格是不是分配的」仍然是个问题——
直接验证分配律要检查所有三元组，在无限格里更是无从下手。
格论的伟大贡献，正是给出**一组可机械验证的判据**：有的看「小陷阱子格」
（$M_3, N_5$），有的看「等式可消性」（消去律），有的看「同态到 $\mathbf{2}$
的分离性」。这一节我们把判据收拢成一套工具箱，并证明其中最深刻的
**Dedekind 消去律判据**——它把「分配」翻译成「加加减减可以消去」的代数直觉，
也为后文的表示定理与布尔代数铺路。

## 1 判据全景：四种等价说法

格 $L$ 是分配格的四个判据，全部等价：

1. **分配律**：$a \wedge (b \vee c) = (a \wedge b) \vee (a \wedge c)$
   （对一切 $a,b,c$）。
2. **禁止子格**（Birkhoff）：$L$ 不含同构于 $M_3$ 或 $N_5$ 的子格。
3. **消去律**（Dedekind）：$a \vee b = a \vee c$ 且 $a \wedge b = a \wedge c$
   ⟹ $b = c$。
4. **同态分离**：任意 $x \ne y$，存在同态 $f : L \to \mathbf{2}$
   使得 $f(x) \ne f(y)$（有限分配格的情形，即 Birkhoff 表示定理的核心）。

<span class="marginnote">四条判据各有侧重：第1条是定义、第2条是「找杂质」、第3条是「代数可消性」、第4条是「能被 0-1 赋值区分」。理解全部四条，等于从四个角度看清分配格——这正是本专题反复强调的「同一结构的多副面孔」。</span>

**辨析｜易错点：** 判据第2条说的是「**子格**同构」，不是「子偏序集同构」。
$\{0,a,b,c,1\}$ 若只是偏序上像 $M_3$ 但 $\vee, \wedge$ 运算对不上，不算数。
判断时必须在 $L$ 中找出**运算封闭**的五元素集合。

## 2 判据二：禁止子格 $M_3$ 与 $N_5$

$M_3$ 是五元素菱形：$\{0, a, b, c, 1\}$，$a,b,c$ 两两不可比且夹在 $0$ 与 $1$ 之间。
$N_5$ 是五元素五角：$\{0, a, b, c, 1\}$，其中 $0 \lt  a \lt  c \lt  1$，
$b$ 只夹在 $0$ 与 $1$ 之间且与 $a, c$ 不可比。

$M_3$ 是模格但非分配：$a \wedge (b \vee c) = a \wedge 1 = a$，
  $(a \wedge b) \vee (a \wedge c) = 0 \vee 0 = 0$。
$N_5$ 连模都不是：取 $a \le c$，$a \vee (b \wedge c) = a \vee 0 = a$，
  $(a \vee b) \wedge c = 1 \wedge c = c$。

**Birkhoff 判据**：分配 ⟺ 禁 $M_3$ 禁 $N_5$。这给出一个「查毒」算法：
扫遍所有五元子集，看有没有运算封闭的 $M_3$ 或 $N_5$。
<span class="marginnote">这条判据也可以反着用：要证明某格不分配，只需构造一个 $M_3$ 或 $N_5$ 子格。子群格不分配的最快证明，就是在 $V_4$ 的子群格里找出 $M_3$——$\{e\}, \langle a\rangle, \langle b\rangle, \langle ab\rangle, V_4$ 恰好排成菱形。</span>

## 3 判据三：Dedekind 消去律

**Dedekind 消去律（cancellation law）**：格 $L$ 中，若

$$a \vee b = a \vee c \quad \text{且} \quad a \wedge b = a \wedge c, \quad \text{则} \quad b = c$$

**定理（Dedekind）**：$L$ 是分配格 ⟺ $L$ 满足消去律。
<span class="marginnote">直觉：在分配格里，「$a$ 对 $b$ 与 $c$ 的并、交都一样」，就足以把 $b, c$ 区分开——分配律保证 $b$ 可以从「$a$ 的加加减减」中唯一还原。这像解方程时的消元：两边同加同减同乘，等式不破则解唯一。</span>

**证明思路（⟸ 方向，即消去律 ⇒ 分配）**：用反证。若 $L$ 非分配，
由 Birkhoff 判据存在 $M_3$ 或 $N_5$ 子格。在 $M_3$ 中取 $a, b, c$
（三个不可比元素）：$a \vee b = 1 = a \vee c$ 且 $a \wedge b = 0 = a \wedge c$，
但 $b \ne c$——消去律失败。在 $N_5$ 中同理构造。故消去律 ⇒ 无 $M_3, N_5$ ⇒ 分配。

**证明思路（⟹ 方向）**：设分配律成立且 $a \vee b = a \vee c$、
$a \wedge b = a \wedge c$。计算 $b$：

$$b = b \wedge (a \vee b) = b \wedge (a \vee c) = (b \wedge a) \vee (b \wedge c)$$

而 $b \wedge a = c \wedge a$（已知），故 $b = (c \wedge a) \vee (b \wedge c)$。
对称地算 $c$，可得 $b = c$。
<span class="marginnote">关键一跳是「$b = b \wedge (a \vee b)$」——这正是吸收律，再借分配律把括号拆开。分配律让「$b \wedge (a \vee c)$」能展开成可消的形式。每一步都只用定义与假设，干净利落。</span>

## 4 公式解析：消去律证明里的三步拆解

把消去律证明中最核心的等式演算单独拎出来：

$$b = b \wedge (a \vee b) \xrightarrow{a \vee b = a \vee c} b \wedge (a \vee c) \xrightarrow{\text{分配律}} (b \wedge a) \vee (b \wedge c) \xrightarrow{b \wedge a = c \wedge a} (c \wedge a) \vee (b \wedge c)$$

- **第一步，吸收**：$b = b \wedge (a \vee b)$ 是吸收律，
  把 $b$ 还原成「$b$ 与 $a \vee b$ 的交」。
- **第二步，代入**：已知 $a \vee b = a \vee c$，替换得到 $b \wedge (a \vee c)$。
- **第三步，分配**：分配律把 $\wedge$ 分配进括号：$(b \wedge a) \vee (b \wedge c)$。
- **第四步，消去**：已知 $a \wedge b = a \wedge c$，即 $b \wedge a = c \wedge a$，
  替换后 $b$ 与 $c$ 的地位完全对称。同样的演算把 $c$ 化成
  $(b \wedge a) \vee (b \wedge c)$（对称式），于是 $b = c$。
  **对称性 + 吸收 + 分配 = 消去**。

## 5 应用：分配格的判断实操

- **整除格 $(\mathbb{N}^+, \mid)$** 分配：验证消去律。若
  $\operatorname{lcm}(a,b) = \operatorname{lcm}(a,c)$ 且
  $\gcd(a,b) = \gcd(a,c)$，则 $b = c$（由素因子幂次逐一比对）。
  <span class="marginnote">这条在算术里很直观：两个数对 $a$ 的「上公倍数」与「下公约数」都相同，则这两个数相同。分配律在整除格上正是「$\gcd$ 对 $\operatorname{lcm}$ 分配」，是初等数论的格论面孔。</span>
**子群格**不分配：找 $M_3$ 或 $N_5$ 子格。
**自由分配格**：3 个生成元的自由分配格恰有 18 个元素，其哈斯图是一个
  漂亮的对称结构——它是「在分配律下互不相同的 3 变量格多项式」的全体。

**辨析｜易错点：** 消去律只在「分配格」里成立。模格满足**模消去**
（$a \vee b = a \vee c$、$a \wedge b = a \wedge c$、且 $a \le b$ 或类似条件
才推出 $b = c$），但一般格中「并交同时相等」推不出「元素相等」——
$M_3$ 里 $a,b,c$ 就是活生生的反例。

## 6 判据的应用：自由分配格的计数

分配格的判据不只是「检查工具」，还能用于**构造与计数**。
最漂亮的例子是自由分配格。

**自由分配格 $\operatorname{FDist}(n)$**：由 $n$ 个生成元、只满足分配律
（连同格律）生成的分配格。它是「所有 $n$ 变量分配格多项式按分配恒等式
模去等价」的集合。

**定理**：$\operatorname{FDist}(n)$ 同构于「$n$ 元素偏序集（互不可比的反链）
的全体下集」的格。于是 $|\operatorname{FDist}(n)|$ =「$n$ 元素集合的
全部集合族的个数」= **Dedekind 数**。

**Dedekind 数 $M(n)$**：$n$ 元素集上「对并交封闭、含空集与全集」的集族个数。
头几个是：

| $n$ | $M(n)$ |
| --- | --- |
| 0 | 2 |
| 1 | 3 |
| 2 | 6 |
| 3 | 20 |
| 4 | 168 |

**辨析｜易错点：** 自由分配格 $\operatorname{FDist}(n)$ 的大小是
**Dedekind 数 $M(n)$**，而自由布尔代数 $\operatorname{FBool}(n)$ 的大小是
$2^{2^n}$。两者不同：$\operatorname{FBool}(1)$ 有 4 个元素，
$\operatorname{FDist}(1)$ 有 3 个。**加补律（布尔）比不加（分配）
多出一倍结构**——每条额外公理都在缩小自由对象。
<span class="marginnote">准确事实：$\operatorname{FDist}(n)$ 的「并不可约元」构成 $n$ 元布尔格 $\mathbf{2}^n$，于是 $|\operatorname{FDist}(n)| = |\mathcal{O}(\mathbf{2}^n)|$ = 布尔格的下集数 = Dedekind 数 $M(n)$。$\operatorname{FDist}(3) = M(3) = 20$，$\operatorname{FDist}(2) = M(2) = 6$，$\operatorname{FDist}(1) = M(1) = 3$。Dedekind 数 $M(4) = 168$、$M(5) = 7581$，增长极快，至今没有闭式公式。</span>

**用判据验证自由格**：$\operatorname{FDist}(3)$ 的 20 个元素可以借助
$M_3/N_5$ 判据确认它确实分配——它不含 $M_3$、$N_5$ 子格，同时是所有
「3 生成元分配多项式」的完整清单。**判据在这里同时是「检查器」与「生成器」**：
既验证已给结构，也枚举自由结构。

## 7 小结

- **四个判据**：分配律 / 禁 $M_3,N_5$ / 消去律 / 同态分离，全部等价。
- **Birkhoff 判据**：分配 ⟺ 不含 $M_3$、$N_5$ 子格；模 ⟺ 只不含 $N_5$。
- **Dedekind 消去律**：$a \vee b = a \vee c \land a \wedge b = a \wedge c
  \Rightarrow b = c$；证明 = 吸收 + 代入 + 分配 + 对称。
- 应用：整除格分配、子群格不分配、自由分配格 20 元素（= Dedekind 数 $M(3)$