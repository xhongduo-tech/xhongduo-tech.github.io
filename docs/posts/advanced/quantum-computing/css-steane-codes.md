---
title: CSS 码与 Steane 码
date: 2026-08-07
---

# CSS 码与 Steane 码

<div class="epigraph">
<p>经典纠错理论的宝藏，通过 CSS 构造直接搬进了量子世界。</p>
<footer>—— 考尔德班克（Robert Calderbank）与肖（Peter Shor）、斯蒂恩（Andrew Steane）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen &amp; Chuang《量子计算与量子信息》§10.4 ｜ 2026-08-07</p>
</div>

## 为什么从 CSS 码开始

稳定子给了我们语法，但还缺「怎么造码」的蓝图。**CSS（Calderbank–Shor–Steane）构造**提供了最实用的蓝图：**用两个「互为对偶的经典线性码」直接构造一个量子稳定子码**。它的优雅在于——量子纠错里最难的「同时防 $X$ 与 $Z$」，在 CSS 下变成「用经典码 $C_1$ 防 $X$、用对偶经典码 $C_2^\perp$ 防 $Z$」，两套经典理论自动分工。<span class="marginnote">CSS 构造由 Calderbank 与 Shor（1996）、Steane（1996）独立提出。它把量子码设计化归为经典码设计，让几十年经典纠错理论（Hamming、RS、LDPC 码）能直接「翻译」成量子码。Steane 码（$[[7,1,3]]$）是它最著名的产物，而今天大规模容错方案里的表面码也可视为 CSS 思想的拓扑版。</span>本节拆开 CSS 构造，再手把手造出 Steane 码。

## 1 经典线性码复习

经典 $[n, k]$ 线性码 $C$ 是 $\mathbb{F}_2^n$ 的一个 $k$ 维子空间，由生成矩阵或校验矩阵 $H$（$(n-k)\times n$）定义：$H x^T = 0$ 对 $x \in C$ 成立。**对偶码**

$$
C^\perp = \{y \in \mathbb{F}_2^n : y \cdot x = 0 \; \forall x \in C\}
$$

是「与 $C$ 所有向量正交」的向量集，维数 $n-k$。<span class="marginnote">经典纠错的奇偶校验矩阵 $H$ 在这里扮演稳定子生成元的「基因」。CSS 的关键要求是 $C_2 \subseteq C_1$（或 $C_2^\perp \subseteq C_1^\perp$），这保证 $X$ 类与 $Z$ 类稳定子互相对易。</span>

## 2 CSS 构造的公式

给定两个经典线性码 $C_1 = [n, k_1, d_1]$ 与 $C_2 = [n, k_2, d_2]$，满足 $C_2 \subseteq C_1$。**CSS 码**的稳定子群由两组生成元构成：

$$
S = \langle \{X_{\vec v} : \vec v \in C_2^\perp\}, \{Z_{\vec w} : \vec w \in C_1^\perp\} \rangle
$$

其中 $X_{\vec v}$ 是在 $\vec v$ 非零位置施加 $X$ 的算子，$Z_{\vec w}$ 同理。码字为

$$
\lvert x + C_2\rangle = \frac{1}{\sqrt{\lvert C_2\rvert}}\sum_{y \in C_2} \lvert x + y\rangle, \qquad x \in C_1
$$

- **第一步，$X$ 类稳定子**：$X_{\vec v}$（$\vec v \in C_2^\perp$）检测 $Z$ 类错误。它把码字在「按 $C_2$ 平移」的等价类上保持不变。
- **第二步，$Z$ 类稳定子**：$Z_{\vec w}$（$\vec w \in C_1^\perp$）检测 $X$ 类错误。
- **第三步，对易条件**：$C_2 \subseteq C_1$ 保证 $X_{\vec v}$ 与 $Z_{\vec w}$ 对易（因为 $\vec v \cdot \vec w = 0$ 对 $C_2^\perp \ni \vec v$、$C_1^\perp \ni \vec w$），稳定子群 Abel 条件满足。<span class="marginnote">CSS 码的参数：编码 $k = k_1 - k_2$ 个逻辑比特（码字数 $= \lvert C_1\rvert/\lvert C_2\rvert = 2^{k_1-k_2}$），距离 $d \ge \min(d_1, d_2^\perp)$。经典码「性能越好」，CSS 码越好——这就是「经典→量子」的免费午餐。</span>

## 3 公式解析：CSS 码为何能防两类错误

验证 CSS 码防 $X$ 错误的能力。设错误是 $X_{\vec e}$（权重 $w(\vec e)$）。

- **第一步，与 $Z$ 稳定子检测**：$X_{\vec e}$ 与 $Z_{\vec w}$（$\vec w \in C_1^\perp$）的交换关系由 $\vec e \cdot \vec w$ 决定：$\vec e\cdot\vec w = 1$ 时反对易（可检测），$=0$ 时对易（不可检测）。
- **第二步，不可检测条件**：$X_{\vec e}$ 不可检测当且仅当 $\vec e$ 与所有 $\vec w \in C_1^\perp$ 正交，即 $\vec e \in (C_1^\perp)^\perp = C_1$——但 $\vec e \in C_1$ 且权重 $\ge d_1$ 才「重到不可检测」。所以 $X$ 类错误可纠到权重 $\lfloor(d_1-1)/2\rfloor$。
- **第三步，对称论证**：$Z$ 类错误由 $C_2^\perp$ 的「距离 $d_2^\perp$」控制，可纠到 $\lfloor(d_2^\perp-1)/2\rfloor$。<span class="marginnote">读法：CSS 码的距离是「$C_1$ 的距离」与「$C_2^\perp$ 的距离」的较小者。经典码理论在这里直接注入——这也是为什么 CSS 码总能借用最好的经典码构造。</span>

## 4 Steane 码：$[[7,1,3]]$

**Steane 码**取 $C_1 = C_2 = C$ 为经典 **Hamming 码** $[7,4,3]$。因为 Hamming 码是**自对偶包含**的（$C^\perp \subseteq C$），满足 $C_2 \subseteq C_1$。稳定子生成元取 $C^\perp$ 的一组基：

$$
X_4X_5X_6X_7, \quad X_2X_3X_6X_7, \quad X_1X_3X_5X_7, \qquad
Z_4Z_5Z_6Z_7, \quad Z_2Z_3Z_6Z_7, \quad Z_1Z_3Z_5Z_7
$$

（三个 $X$ 型 + 三个 $Z$ 型，共 6 个生成元，$7-6=1$ 个逻辑比特）。距离 $d = 3$，可纠任意单比特错误。<span class="marginnote">Steane 码的对称之美：$X$ 型与 $Z$ 型稳定子各 3 个、结构完全相同（都是 Hamming 校验模式）。它同时具有「transversal」的性质——某些逻辑门（如 $H$、$S$）可以按比特逐位实现，天然容错。这让 Steane 码成为容错计算的经典教学样板。</span>

**辨析｜易错点：** Steane 码是 $[[7,1,3]]$，但**不是**「把 7 个比特复制 3 份」。它的 7 个物理比特通过 Hamming 码的代数结构纠缠起来，比重复码更高效（重复码 $[[9,1,3]]$ 需 9 比特）。「比特更多 = 码更强」是直觉误导——真正决定码力的是**距离**与**率**的平衡，代数结构才是核心。

## 5 CSS 码的应用版图

CSS 构造在现代量子计算里无处不在：

- **Steane 码**：教科书标准码，演示 transversal 逻辑门与容错子例程。
- **表面码（surface code）**：可看作 CSS 码的拓扑实例，用「棋盘格」上的局域稳定子实现，是当前超导平台大规模纠错的事实标准。
- **量子 LDPC 码**：用经典 LDPC 码构造 CSS 码，码率高、阈值好，是「低开销容错」的前沿方向。<span class="marginnote">CSS 之外还有更一般的稳定子码（如五比特码 $[[5,1,3]]$ 不是 CSS），但 CSS 家族覆盖了绝大多数实用构造。「经典码 → CSS → 量子码」这条流水线，让量子纠错站在了经典纠错理论的肩膀上。</span>

## 6 小结

- **CSS 构造**：给定 $C_2 \subseteq C_1$，稳定子取 $\{X_{\vec v}:\vec v\in C_2^\perp\}$ 与 $\{Z_{\vec w}:\vec w\in C_1^\perp\}$，码字是 $C_1/C_2$ 的陪集叠加。
- 防 $X$ 错误靠 $C_1$ 的距离，防 $Z$ 错误靠 $C_2^\perp$ 的距离，互不干扰。
- **Steane 码** $[[7,1,3]]$：取经典 Hamming 码，6 个稳定子、1 个逻辑比特、距离 3，支持 transversal 门。
- 应用：容错计算教学、表面码（拓扑 CSS）、量子 LDPC 码。
- **易错点**：码力由距离与率决定，不是由比特数决定。

在下一节，我们讲今天大规模纠错的明星——**表面码（surface code）初步与容错阈值**。
