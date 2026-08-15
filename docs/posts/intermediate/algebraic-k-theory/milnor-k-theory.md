---
title: Milnor K 理论
date: 2026-08-07
---

# Milnor K 理论

<div class="epigraph">
<p>一个方程若想要有意义，它的解必须能经受住符号的检验。</p>
<footer>—— 出自米尔诺《代数 K 理论导引》（John Milnor, Introduction to Algebraic K-Theory）</footer>
</div>

<div class="article-byline">
<p>第二级 · 代数 K 理论 ｜ Milnor《Introduction to Algebraic K-Theory》§5 ｜ 2026-08-07</p>
</div>

## 为什么在 K₂ 之后立刻讲 Milnor K 理论

上一节 Matsumoto 定理把域上的 $K_2$ 写成了张量积的商。这个表达式有一个惊人的特点：**它完全不需要矩阵**。只要域 $k$ 的乘法群 $k^\times$ 和「$1-x$ 这个减法结构」，就能造出 $K_2$。Milnor 在 1970 年抓住这个线索，问了一个自然的问题：**能不能把这种「纯乘法 + Steinberg 关系」的构造推广到任意高阶？** 答案就是 Milnor K 理论 $K_*^M(k)$。<span class="marginnote">Milnor K 理论比 Quillen 的高阶 $K$ 理论早诞生两年，是它的「堂兄弟」。两者在低阶重合（$K_0, K_1, K_2$），高阶却分道扬镳；后来的 Norm Residue 定理把 Milnor K 与 Galois 上同调缝合，使它在数论里大放异彩。</span>

Milnor K 理论在整棵知识树里的位置很像「从极限到大模型」里的递推数列：**用一个机械的规则（张量积 + 商掉关系）把已知的低阶结果一层层往上铺**。理解它，你就理解了什么叫「代数结构的无穷阶梯」。

## 1 从符号到张量积：丢弃矩阵

回顾 $K_2(k)$ 的符号语言：$\{a,b\}$ 满足双线性（$\{a_1a_2,b\} = \{a_1,b\}\{a_2,b\}$）与 Steinberg 关系（$\{a,1-a\} = 0$）。符号的乘法实际上对应「把两个符号拼成一个更长的符号」：

$$
\{a_1, \dots, a_r\} \cdot \{b_1, \dots, b_s\} = \{a_1, \dots, a_r, b_1, \dots, b_s\}
$$

一旦承认这个拼接运算，对象就从「二元符号」变成了「任意长符号」$\{a_1, \dots, a_n\}$。这正是张量积的领地：$a_1 \otimes \cdots \otimes a_n$ 天然可以拼接。**矩阵被彻底抛在身后**——Milnor K 理论是一个关于域本身的理论，不是关于环上矩阵的理论。

## 2 Milnor K 群的定义

设 $k$ 是域，$T(k^\times)$ 是乘法群 $k^\times$ 的**张量代数**：

$$
T(k^\times) = \bigoplus_{n \ge 0} \underbrace{k^\times \otimes_{\mathbb{Z}} \cdots \otimes_{\mathbb{Z}} k^\times}_{n \text{ 份}}
$$

其中第 0 项约定为 $\mathbb{Z}$。在 $T(k^\times)$ 里商掉「Steinberg 关系子模」——由所有形如「$a$ 与 $1-a$ 相邻出现的张量」生成的子模：

> **Milnor K 理论（Milnor K-theory）**：
> $$
> K_*^M(k) = \frac{T(k^\times)}{\big\langle a_1 \otimes \cdots \otimes a_n \mid a_i + a_{i+1} = 1,\ a_i \neq 0, 1 \big\rangle}
> $$
> 记 $a_1 \otimes \cdots \otimes a_n$ 在商里的像为 $\{a_1, \dots, a_n\}$，称为 **Milnor 符号**。第 $n$ 次齐次部分记作 $K_n^M(k)$。

**一个定义包含三代结果**：$K_0^M(k) = \mathbb{Z}$（张量代数第 0 项），$K_1^M(k) = k^\times$（商不掉任何东西），$K_2^M(k) = K_2(k)$（Matsumoto 定理——这正是定义里 Steinberg 关系的来源）。<span class="marginnote">注意关系里的条件是「存在某对相邻的 $a_i + a_{i+1} = 1$」——不是「全部相邻都如此」。这保证了低阶 $K_1$ 不受打扰，而高阶处处受约束。</span>

## 3 分次交换环的结构

$K_*^M(k) = \bigoplus_n K_n^M(k)$ 在符号拼接下构成一个**分次交换环（graded-commutative ring）**：

$$
x \cdot y = (-1)^{\deg x \cdot \deg y}\, y \cdot x \qquad (x \in K_p^M(k),\ y \in K_q^M(k))
$$

由此立即得到符号的**斜对称**：$\{a_1, a_2\} = -\{a_2, a_1\}$，以及任意两个相邻项交换时要乘 $(-1)$。这与 $K_2$ 里的斜对称 $\{a,b\}\{b,a\} = 1$ 完全一致（乘法记号 vs 加法记号）。

Milnor K 群还是**函子性的**：域嵌入 $k \hookrightarrow F$ 诱导 $K_n^M(k) \to K_n^M(F)$，符号原样送入；而 Galois 群的作用（若扩张是 Galois 的）使 $K_n^M$ 带上 $G$-模结构——这为下一节与 Galois 上同调的联系埋下伏笔。

**辨析｜易错点：** Milnor K 理论 $K_*^M$ 与 Quillen 的代数 $K$ 理论 $K_*$ 在 $n \le 2$ 时重合，$n \ge 3$ 时**不**重合。例如 $K_3^M(\mathbb{Q}) = 0$，而 $K_3(\mathbb{Q})$ 非零且含丰富结构。二者不是同一回事——很多教材标题里的「K 理论」指的是 Quillen 的那套，Milnor K 只是「符号世界」里独立生长的一支。

## 4 公式解析：Steinberg 关系的张量积形式

定义里最需要拆解的是那个商子模。把定义重写为「生成元 + 关系」：

$$
K_n^M(k) = \frac{\big(k^\times\big)^{\otimes n}}{\big\langle a_1 \otimes \cdots \otimes a_n :\ \text{存在 } i,\ a_i + a_{i+1} = 1 \big\rangle}
$$

**第一步，看分子**：$(k^\times)^{\otimes n}$ 是 $n$ 个乘法群的整数张量积。它把「$n$ 个非零元素摆成一串」的抽象对象全部列出：$a_1 \otimes \cdots \otimes a_n$ 有双线性——可以把任意一个位置里的乘积拆开。这保证了符号 $\{a_1,\dots,a_n\}$ 对每个槽都是「可加的」（在 $K_2$ 里就是 $\{a_1a_2,b\} = \{a_1,b\}+\{a_2,b\}$）。

**第二步，看关系**：$\langle \cdots \rangle$ 表示由「某对相邻元素凑成 $1$」的张量生成的子模。这条关系的灵魂是**「$a$ 与 $1-a$ 不共存」**：一旦相邻两槽里出现互为「$1-x$」配对的元素，整个符号归零。它把域里加减法的信息，刻进了纯乘法的世界。

**第三步，看它如何约束高阶**：$K_2^M(k)$ 里 $\{a,1-a\}=0$；到 $K_3^M(k)$，只要 $\{a_1,a_2,a_3\}$ 中有一对相邻满足 $a_i + a_{i+1} = 1$（比如 $a_2 = 1-a_3$），就为零。**关系只看相邻**——所以 $K_n^M$ 比「所有配对都要检查」的构造更宽松，也因此常常有限、可算。

**第四步，读出哲学**：张量积 + Steinberg 关系 = 「乘法世界的多线性代数」模掉「一个与减法有关的唯一公理」。这个配方轻到只有两条输入（乘法、$1-x$），重到足以承载 Milnor 猜想与 Norm Residue 定理——**最轻的定义，往往产生最深的定理**。

## 5 与 Galois 上同调：Norm Residue 定理

Milnor K 理论的巨大价值，来自它与 **Galois 上同调** 的精确对接。

> **Norm Residue 定理（Voevodsky–Rost；m=2 时即 Milnor 猜想，2011 年魏恩斯坦定稿）。** 设 $k$ 是域，$m$ 与 $\operatorname{char}(k)$ 互素，则对一切 $n \ge 0$：
> $$
> K_n^M(k) / m \ \cong\ H^n\big(k,\ \mu_m^{\otimes n}\big)
> $$
> 其中右侧是 $k$ 的（绝对）Galois 群的 $n$ 次连续上同调，系数为 $m$ 次单位根群的第 $n$ 次张量幂。

**第一步，读左侧**：$K_n^M(k)/m$ 是「模 $m$ 的 Milnor K 群」——把符号的全部 $m$ 倍信息扔掉。它是一个 $\mathbb{Z}/m$-模，携带域的乘法符号信息。

**第二步，读右侧**：$H^n(k, \mu_m^{\otimes n})$ 是 Galois 群 $G = \operatorname{Gal}(\bar k / k)$ 的上同调，系数是单位根群张量幂——它是**域的「内心秘密」（Galois 群）的 $n$ 次测量**。

**第三步，看对应**：定理说这两者**同构**。对 $n=1$ 这是 Kummer 理论 $k^\times/k^{\times m} \cong H^1(k, \mu_m)$（中学就有的「根式扩张」的上同调翻版）；对 $n=2$ 它与类域论、Hilbert 符号的互反律吻合；一般 $n$ 由 Voevodsky 用**动机上同调**证明。<span class="marginnote">2002 年 Voevodsky 因证明 Milnor 猜想（$m=2$ 情形）获菲尔兹奖。他给出一条链：$K_n^M(k)/2 \cong CH^n(\operatorname{Spec} k, n)$（higher Chow 群）$\cong H^n(k, \mathbb{Z}/2)$。动机上同调——代数 K 理论的远房近亲——在此扮演枢轴。</span>

**第四步，尝到甜头**：Milnor K 理论于是成了「域的算术不变量」最便捷的载体。对有限域 $K_n^M(\mathbb{F}_q) = 0\ (n\ge 2)$；对实数域 $K_n^M(\mathbb{R}) = \mathbb{Z}/2\ (n\ge 2)$（由 $\{-1,\dots,-1\}$ 生成）；对数域则与单位根群、Tame 符号精细纠缠——这些正是第 11 篇代数数论要展开的疆域。

### 术语速查表：Milnor K 理论

| 记号 | 名称 | 含义 |
| --- | --- | --- |
| $K_n^M(k)$ | Milnor K 群 | $T(k^\times)$ 商掉 Steinberg 关系的第 $n$ 次部分 |
| $\{a_1,\dots,a_n\}$ | Milnor 符号 | $a_1\otimes\cdots\otimes a_n$ 在商里的像 |
| $T(k^\times)$ | 张量代数 | $\bigoplus_{n\ge0}(k^\times)^{\otimes n}$ |
| Steinberg 关系 | —— | $a_i + a_{i+1} = 1$ 使相邻张量归零 |
| $K_*^M(k)$ | Milnor K 环 | 分次交换环，拼接为乘法 |
| $\mu_m^{\otimes n}$ | 单位根群张量幂 | Norm Residue 定理的系数 |
| $H^n(k,\cdot)$ | Galois 上同调 | 绝对 Galois 群的连续上同调 |

**辨析｜易错点：** Milnor 符号 $\{a_1,\dots,a_n\}$ 的槽**有序**（斜对称），交换相邻两项要乘 $(-1)$；但若 $a_i = a_{i+1}$，符号不一定为零——只归到 $\{a_i, -1\}$ 型，2-挠由此潜入高阶。初学时把「符号斜对称」误当成「相同项必零」，会在计算 $K_n^M(\mathbb{R})$ 时出错（那里 $\{-1,\dots,-1\} = \mathbb{Z}/2$ 恰恰非零）。

## 6 小结

- **Milnor K 理论**：$K_*^M(k) = T(k^\times) / \langle a \otimes (1-a)\rangle$，是纯符号（张量积 + Steinberg 关系）的构造，与矩阵无关。
- **低阶重合**：$K_0^M(k)=\mathbb{Z}$，$K_1^M(k)=k^\times$，$K_2^M(k)=K_2(k)$（Matsumoto）。
- **分次交换环**：符号拼接给出乘法，斜对称由 $(-1)^{\deg x\deg y}$ 控制。
- **关系只看相邻**：$a_i + a_{i+1} = 1$ 使整个符号归零。
- **Norm Residue 定理**：$K_n^M(k)/m \cong H^n(k, \mu_m^{\otimes n})$——Milnor 猜想（$m=2$）由 Voevodsky 证明。
- **易错**：Milnor K 与 Quillen K 在 $n \ge 3$ 不同（$K_3^M(\mathbb{Q})=0$ 而 $K_3(\mathbb{Q}) \neq 0$）。

在下一节，我们暂时离开域，回到几何与代数的交界处。Swan 定理将告诉你：**为什么环上的 $K_0$