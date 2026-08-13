---
title: 布尔代数的基本性质
date: 2026-08-07
---

# 布尔代数的基本性质

<div class="epigraph">
<p>我完全不知道这个演算的对象是什么，也不知道符号背后是什么含义；但它是可以机械执行的，而且永远正确。</p>
<footer>—— 乔治·布尔（George Boole, 1847）</footer>
</div>

<div class="article-byline">
<p>第二级 · 格论与序理论 ｜ Birkhoff 第6章；Davey &amp; Priestley 第4章 ｜ 2026-08-07</p>
</div>

## 为什么从布尔代数开始

分配格已经把「集合的并交」抽象出来了，但还缺最关键的一招：**补元**。
有了补元，分配格升级为**布尔代数**——$\vee, \wedge, \neg$ 三件套齐备，
既是格论的主角，也是数字电路、命题逻辑、集合代数共同的语言。
布尔代数之所以是「代数」，是因为它的运算规则完全可以形式化、机械化地执行：
布尔本人坦承不知道符号的含义，却仍能算出正确结果。
这正是「代数」的精髓——**结构自足，含义自明**。
本节建立布尔代数的定义与基本运算律，下一节再给出 Stone 表示定理，
宣告布尔代数就是幂集代数。

## 1 补元：格里的「否定」

设 $L$ 是有界格（有 $0$ 与 $1$），$a \in L$。若存在 $b$ 使得

$$a \vee b = 1 \quad \text{且} \quad a \wedge b = 0$$

则 $b$ 是 $a$ 的一个**补元（complement）**，记作 $\bar{a}$ 或 $\neg a$ 或 $a'$。
<span class="marginnote">补元的直觉：$a$ 与它的补「拼成整个世界（$1$）」，且「没有公共部分（$0$）」。在幂集格中，$S$ 的补元就是补集 $X \setminus S$：并成全集、交为空集。</span>

**辨析｜易错点：** 任意格中补元**不一定唯一**。$M_3$（菱形）里，
$a, b, c$ 互不相同的三元素都是同一个中间层的成员：$a \vee b = 1$ 且
$a \wedge b = 0$，$a \vee c = 1$ 且 $a \wedge c = 0$——$a$ 有两个补元 $b, c$。
只有在**分配格**中，补元才唯一（下面证明）。
所以「每个元素有唯一补元」是布尔代数的特征性条件。

## 2 布尔代数的定义：分配 + 有补

**布尔代数（Boolean algebra）**是一个有界分配格
$(B, \vee, \wedge, 0, 1)$，其中每个元素 $a$ 都有补元 $\neg a$。

等价地：$(B, \vee, \wedge, \neg, 0, 1)$ 是满足下列公理的代数结构：

**格公理**：$\vee, \wedge$ 幂等、交换、结合、吸收；
**分配律**：$a \wedge (b \vee c) = (a \wedge b) \vee (a \wedge c)$；
**有界性**：$a \vee 0 = a$，$a \wedge 1 = a$；
**补元律**：$a \vee \neg a = 1$，$a \wedge \neg a = 0$。

**辨析｜易错点：** 有些教材用「有补格」代替分配律，但**有补格未必分配**
（$M_3$ 是有补格但不是分配格、不是布尔代数）。布尔代数的精确定义必须
**同时**要求分配与有补——分配保证补元唯一，补元保证否定总存在。
<span class="marginnote">注意：有补分配格自动是「每元素恰有一补」；反过来说，「每元素恰有一补的有界格」也自动是分配格（有定理保证）。所以布尔代数 = 有界格 + 唯一补。这两条路径殊途同归。</span>

## 3 基本性质：对合、De Morgan 与吸收

从定义可直接推出布尔代数的一整套运算律：

**对合律（involution）**：$\neg(\neg a) = a$。
**De Morgan 律**：$\neg(a \vee b) = \neg a \wedge \neg b$，
$\neg(a \wedge b) = \neg a \vee \neg b$。
**常数**：$\neg 0 = 1$，$\neg 1 = 0$。
**补元唯一**：若 $a \vee b = 1$ 且 $a \wedge b = 0$，则 $b = \neg a$。

**补元唯一的证明**（用分配律）：设 $a$ 有补元 $b$ 与 $c$，则

$$b = b \wedge 1 = b \wedge (a \vee c) = (b \wedge a) \vee (b \wedge c) = 0 \vee (b \wedge c) = b \wedge c$$

对称地 $c = b \wedge c$，故 $b = c$。
<span class="marginnote">这套演算把「唯一性」归约为分配律的一次展开：$b$ 与 $c$ 都等于 $b \wedge c$。它不需要任何直觉，纯代数推演——正是布尔「不知道含义也算得对」的样板。</span>

**辨析｜易错点：** De Morgan 律把「否定」与「并交」互换，
与数理逻辑里「$\neg(P \land Q) = \neg P \lor \neg Q$」同构。
初学者常漏掉「$\vee \leftrightarrow \wedge$ 互换」这一步，
写出 $\neg(a \vee b) = \neg a \vee \neg b$ 的错误版本。口诀：
**否定要「分配」，但符号要翻转。**

## 4 公式解析：De Morgan 律的格论验证

验证 $\neg(a \vee b) = \neg a \wedge \neg b$，只需检验右侧确实是
$a \vee b$ 的补元：

$$(a \vee b) \vee (\neg a \wedge \neg b) = 1, \qquad (a \vee b) \wedge (\neg a \wedge \neg b) = 0$$

- **第一步，验并**：$(a \vee b) \vee (\neg a \wedge \neg b)$。用分配律展开：
  $[(a \vee b) \vee \neg a] \wedge [(a \vee b) \vee \neg b] =
  [1 \vee b] \wedge [a \vee 1] = 1 \wedge 1 = 1$。
- **第二步，验交**：$(a \vee b) \wedge (\neg a \wedge \neg b)$ 同理：
  $= (a \wedge \neg a \wedge \neg b) \vee (b \wedge \neg a \wedge \neg b) = 0 \vee 0 = 0$。
- **第三步，读结构**：两条验证都只用了分配律 + 补元律 + 吸收律，
  把「$a \vee b$ 与 $\neg a \wedge \neg b$ 互补」从三个公理机械推出。
  **De Morgan 律不是新公理，是分配 + 补元的推论。**
- **第四步，读哲学**：对合律 $\neg\neg a = a$ 配合 De Morgan，
  说明 $\neg$ 是 $B$ 上的一个**反自同构**（对偶同构）——
  翻转序、交换 $\vee \wedge$。这给布尔代数一个漂亮的对称性。

## 5 布尔代数与布尔环：Stone 的桥梁

布尔代数还能「降维」成一种环。在布尔代数 $B$ 上定义：

$$a + b = (a \wedge \neg b) \vee (\neg a \wedge b) \quad \text{（对称差）}, \qquad a \cdot b = a \wedge b$$

则 $(B, +, \cdot)$ 是一个**布尔环（Boolean ring）**：交换、有单位元 $1$、
且每个元素幂等（$a^2 = a$）。反过来，任何布尔环由 $a \vee b = a + b + ab$
定义出布尔代数。**布尔代数 ⟺ 布尔环**（Stone 建立），这是一座著名的桥。
<span class="marginnote">「布尔代数 = 布尔环」这个等价，让布尔代数能借用环论的全部武器：理想、商环、素理想（就是上一节的素理想！）、极大理想。Stone 表示定理的证明正是从「极大理想 = 点」出发的。数论里的 $\mathbb{Z}/2\mathbb{Z}$ 是两元素布尔环。</span>

## 6 例子与「为什么值得学」

**幂集代数** $(\mathcal{P}(X), \cup, \cap, \complement)$：最标准的布尔代数。
**命题逻辑的 Lindenbaum 代数**：命题公式按逻辑等价分类，
  $\vee, \wedge, \neg$ 是逻辑联结词——布尔代数就是命题逻辑的代数化。
  <span class="marginnote">「布尔代数 = 命题逻辑的代数」是理解它的钥匙：$1$ = 恒真，$0$ = 恒假，$\vee$ = 或，$\wedge$ = 与，$\neg$ = 非。任何布尔恒等式都是逻辑重言式，反之亦然。这也预告了第3篇末尾的 Heyting 代数——它把「排中律」扔掉后得到的直觉主义逻辑。</span>
**开关电路**：数字电路里的 AND/OR/NOT 门就是布尔代数运算；
  「布尔代数 = 电路设计」是香农 1938 年的里程碑发现。
**集族代数**：任何「对 $\cup, \cap, \complement$ 封闭」的集族
  （如 $X$ 的子集全体、$\sigma$-代数）都是布尔代数——测度论就在这种代数上搭台。

## 7 布尔代数的运算熟练度：化简与对偶

布尔代数的学习离不开「算」——熟练掌握运算律，才能在电路设计、逻辑化简里
自如运用。这里给出四个最常用的化简套路。

**套路一：吸收律化简**。$a \vee (a \wedge b) = a$、
$a \wedge (a \vee b) = a$。凡出现「自己与自己并/交某物」，直接消去。

**套路二：De Morgan 逐层翻**。对多层否定，从外到内翻：
$\neg(a \vee \neg(b \wedge c)) = \neg a \wedge \neg\neg(b \wedge c) = \neg a \wedge (b \wedge c)$。
每翻一层，$\vee \leftrightarrow \wedge$ 互换一次。

**套路三：补元配对**。$a \vee \neg a = 1$、$a \wedge \neg a = 0$。
见到「一个元素与其补同时出现」，优先配对消去。

**例**：化简 $(a \wedge b) \vee (\neg a \wedge b)$。

- 观察：$b$ 是公因子。在布尔代数里没有「分配提取」的公理，
  但有分配律：$(a \wedge b) \vee (\neg a \wedge b) = (a \vee \neg a) \wedge b = 1 \wedge b = b$。
- **结论**：$(a \wedge b) \vee (\neg a \wedge b) = b$。
  这是数字电路里「$AB + \bar{A}B = B$」的代数来源。

**对偶原则在布尔代数中尤其好用**：把 $\vee \leftrightarrow \wedge$、
$0 \leftrightarrow 1$ 全部互换，任何成立的恒等式仍成立。
例：$a \vee 0 = a$ 的对偶是 $a \wedge 1 = a$；
$a \vee \neg a = 1$ 的对偶是 $a \wedge \neg a = 0$。**背一条，自动得一条。**

**辨析｜易错点：** 对偶原则里，$\neg$ **不变**——只有 $\vee, \wedge, 0, 1$
互换。$a \vee b$ 的对偶是 $a \wedge b$，不是 $\neg a \vee \neg b$
（那是 De Morgan）。两个原则分工明确：De Morgan 管否定，对偶原则管并交换。

**练习**（每题用套路化简）：

1. $a \vee \neg a \wedge b$——注意运算优先级：先 $\wedge$ 后 $\vee$。
   答案：$a \vee (\neg a \wedge b) = (a \vee \neg a) \wedge (a \vee b) =
   1 \wedge (a \vee b) = a \vee b$。
2. $(a \vee b) \wedge (\neg a \vee b) = (a \wedge \neg a) \vee b = b$。
   <span class="marginnote">这些化简在卡诺图（Karnaugh map）、布尔表达式最小化里都是基本功。数字电路设计的本质，就是在布尔代数里做「最小化」——用最少的与门或门实现同一个布尔函数。香农 1938 年的开创论文《继电器与开关电路的符号分析》正是把布尔代数变成电路设计的语言。</span>
3. 证明 $\neg(a \vee b) \vee \neg(a \wedge b) = \neg a$？——**陷阱**：
   $\neg(a \vee b) \vee \neg(a \wedge b) = (\neg a \wedge \neg b) \vee (\neg a \vee \neg b)$，
   这不等于 $\neg a$。用真值表核对：$a=0, b=1$ 时原式 $= \neg 1 \vee \neg 0 = 0 \vee 1 = 1$，
   而 $\neg a = 1$，$a=1,b=1$ 时原式 $= 0 \vee 0 = 0 \ne \neg a = 0$？——
   $a=1$ 时 $\neg a = 0$，原式 $=0$，碰巧相等；$a=0,b=0$ 时原式 $=1 \vee 1=1=\neg 0=1$。
   真值表逐行核对才知道等式不恒成立。**教训：化简要机械、逐步、可核验，
   别凭直觉跳步。**

## 8 小结

- **补元**：$a \vee b = 1$ 且 $a \wedge b = 0$；分配格中补元**唯一**，
  $M_3$ 中不唯一。
- **布尔代数** = 有界分配格 + 每元素有补元（或：有界格 + 唯一补）。
- 核心运算法则：对合律 $\neg\neg a = a$、De Morgan 律（否定翻转并交）、
  常数补、补元唯一。
- **布尔代数 ⟺ 布尔环**（Stone）：对称差 + 交构成环。
- 例子：幂集、命题逻辑 Lindenbaum 代数、开关电路、$\sigma$