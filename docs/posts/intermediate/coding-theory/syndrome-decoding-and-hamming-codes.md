---
title: 线性码解码：伴随式、标准阵列与 Hamming 码
date: 2026-08-07
---

# 线性码解码：伴随式、标准阵列与 Hamming 码

<div class="epigraph">
<p>若人们不相信数学是简单的，那只是因为他们没有意识到生活有多复杂。</p>
<footer>—— 约翰 · 冯 · 诺依曼（John von Neumann）</footer>
</div>

<div class="article-byline">
<p>第二级 · 编码理论（纠错编码） ｜ Roth 第2章；van Lint 第3章 ｜ 2026-08-07</p>
</div>

## 为什么从解码开始

编码解决了「怎么把消息变成码字」，但接收方拿到的是被噪声污染的向量 $\boldsymbol{r}$。**解码（decoding）** 回答的是更棘手的问题：怎么从 $\boldsymbol{r}$ 高效地恢复发送的码字。最朴素的最近邻译码要跟每个码字比较距离，$q^k$ 个码字逐一比对——在码字指数增长的现实里根本不可行。

线性代数再次出手相救：校验矩阵 $H$ 不仅定义了码，还提供了一种「只凭一次矩阵乘法就把错误定位到陪集」的机制。**伴随式（syndrome）解码**把译码从「全空间搜索」压缩成「查一张小表」，而 Hamming 码把这套机制推到极致——查表都省了，伴随式本身就是错误位置。<span class="marginnote">信息论里「最大似然译码」是理论最优，伴随式译码则是线性码的<strong>结构化实现</strong>：它在 BSC 上等价于最近邻译码，但复杂度从指数降到多项式。这是「理论存在性」与「工程可行性」的一次完美会师。</span>

## 1 伴随式：错误的指纹

设发送码字 $\boldsymbol{c}$，信道加了错误向量 $\boldsymbol{e}$，接收向量为 $\boldsymbol{r} = \boldsymbol{c} + \boldsymbol{e}$。<span class="marginnote">错误向量 $\boldsymbol{e}$ 的支撑（非零位置）就是出错位置；$\mathrm{wt}(\boldsymbol{e})$ 是出错个数。把加法理解为 $\mathbb{F}_2$ 上的异或，$+$ 与 $-$ 相同。</span>两边同乘校验矩阵：

$$H \boldsymbol{r}^T = H(\boldsymbol{c} + \boldsymbol{e})^T = H \boldsymbol{c}^T + H \boldsymbol{e}^T = \boldsymbol{0} + H \boldsymbol{e}^T = H \boldsymbol{e}^T$$

**伴随式（syndrome）**：$\boldsymbol{s} = H \boldsymbol{r}^T \in \mathbb{F}_q^{n-k}$，是接收向量的「校验结果」。

**重点：伴随式只依赖错误，不依赖码字。** $H \boldsymbol{c}^T = 0$ 把码字的影响清零，剩下的 $H \boldsymbol{e}^T$ 完全由错误决定。于是「$\boldsymbol{s} = 0$」等价于「$\boldsymbol{r}$ 是码字」（无错或错误向量本身是码字）；「$\boldsymbol{s} \neq 0$」说明有错，且 $\boldsymbol{s}$ 记录了错误的全部可观测信息。<span class="marginnote">这就像体检报告上的异常指标：正常人的指标是基准值，异常指标本身不告诉你具体病根，但把所有异常指标合并起来就能缩小到几种可能。伴随式把「$q^n$ 个可能的接收向量」归并成「$q^{n-k}$ 类错误指纹」。</span>

系统码还有一个直白的伴随式计算方式：若 $\boldsymbol{c} = (\boldsymbol{u} \mid \boldsymbol{p})$，收到 $\boldsymbol{r} = (\boldsymbol{r}_u \mid \boldsymbol{r}_p)$，则伴随式等于「用收到的信息位重新算一遍校验位」与「收到的校验位」之差：$\boldsymbol{s} = f(\boldsymbol{r}_u) - \boldsymbol{r}_p$。不一致的地方，就是校验没通过的地方——这更贴近工程里「重新计算奇偶校验」的直觉。

## 2 陪集与陪集首：把「谁最可能」说清楚

伴随式把 $\mathbb{F}_q^n$ 划分成若干类。**陪集（coset）**：对固定向量 $\boldsymbol{e}_0$，集合 $\boldsymbol{e}_0 + \mathcal{C} = \{\boldsymbol{e}_0 + \boldsymbol{c} \mid \boldsymbol{c} \in \mathcal{C}\}$ 称为码的一个陪集。

三个基本事实：

1. 同一个陪集里的所有向量有**相同的伴随式**（$H(\boldsymbol{e}_0 + \boldsymbol{c})^T = H \boldsymbol{e}_0^T$）；
2. 不同陪集的伴随式不同（否则两陪集合并），所以伴随式与陪集**一一对应**，共 $q^{n-k}$ 个陪集；
3. $\mathbb{F}_q^n$ 是所有陪集的**不交并**，每个陪集恰有 $q^k$ 个元素。

给定接收向量 $\boldsymbol{r}$，它的伴随式锁定一个陪集 $\boldsymbol{e} + \mathcal{C}$。发送的码字就在这个陪集里（因为 $\boldsymbol{r} = \boldsymbol{e} + \boldsymbol{c}$），译码要做的就是从陪集里挑一个码字。

**陪集首（coset leader）**：一个陪集里重量最小的向量，记作 $\boldsymbol{e}_L$。它代表「最可能出现的错误模式」。<span class="marginnote">在 BSC 上出错 $i$ 个的概率是 $p^i (1-p)^{n-i}$，$p \lt  1/2$ 时出错越少概率越大，所以「陪集里重量最小的错误」正是最大似然意义上最可能的错误——陪集首译码就是最大似然译码。</span>

## 3 标准阵列：一张表完成最近邻译码

**标准阵列（standard array）** 把整个 $\mathbb{F}_q^n$ 排成一张 $q^{n-k} \times q^k$ 的表：

1. 第一行是全部码字（以零码字开头）；
2. 对每个尚未出现过的陪集，选一个最小重量向量做陪集首 $\boldsymbol{e}_L$，作为该行首元素；
3. 每行其余元素是 $\boldsymbol{e}_L$ 加对应列码字。

| 陪集首 | 码字 | … |
| --- | --- | --- |
| $\boldsymbol{0}$ | $\boldsymbol{c}_1$ | … |
| $\boldsymbol{e}_1$ | $\boldsymbol{e}_1 + \boldsymbol{c}_1$ | … |
| $\vdots$ | $\vdots$ | $\vdots$ |
| $\boldsymbol{e}_{q^{n-k}-1}$ | $\boldsymbol{e}_{q^{n-k}-1} + \boldsymbol{c}_1$ | … |

**译码规则**：收到 $\boldsymbol{r}$，找到它所在的行（由伴随式确定），读出该行行首 $\boldsymbol{e}_L$，输出 $\boldsymbol{c} = \boldsymbol{r} - \boldsymbol{e}_L$。<span class="marginnote">标准阵列的译码正确性由「陪集首是行内最小重量」保证：如果实际错误恰是陪集首，则译码正确；如果实际错误不是陪集首，则译码错误。它能纠正的错误模式恰好是「是某个陪集首」的错误向量。</span>

**实例：$[5, 2, 3]$ 码的标准阵列。** 设码由 $G = \begin{pmatrix} 1 & 0 & 1 & 1 & 0 \\ 0 & 1 & 0 & 1 & 1 \end{pmatrix}$ 生成，4 个码字为 $00000, 10110, 01011, 11101$。$2^{5-2} = 8$ 个陪集，前 4 个陪集首可依次取 $00000, 10000, 01000, 00100$；剩下 4 个陪集已无重量 1 向量（全部单错位置被前 4 个占尽），只能取重量 2 的向量如 $00010, 00001, 11000, 10001$。<span class="marginnote">注意：$t = \lfloor (3-1)/2 \rfloor = 1$，但标准阵列实际能纠正的错误比「全部单错」多——表里有 4 个重量 2 的陪集首也被纳入了纠错范围。保证纠 1 错是下界，不是上限。</span>收到 $10011$：算得伴随式后定位到陪集首 $01000$ 那一行，译码输出 $10011 - 01000 = 11011$（恰是码字 $11101$ 的一位翻转——但注意这里的翻转是第 2 位，实际错误未必真是它，只是最可能假设）。

**辨析｜易错点：** 标准阵列能纠正的错误数是「$t = \lfloor (d-1)/2 \rfloor$」吗？严格说，标准阵列能纠正的错误模式多于 $t$ 个的集合（只要该错误向量是陪集首），但**保证**纠正的是重量 $\le t$ 的所有错误。个别陪集可能有两个重量相等的最小向量，这时任选一个都行，但会把另一个「等概率候选」漏掉——这是译码器的内在模糊性。

## 4 公式解析：从伴随式到错误位置

核心问题是：知道了伴随式 $\boldsymbol{s} = H \boldsymbol{e}^T$，怎么反解错误 $\boldsymbol{e}$？

$$\boldsymbol{s}^T = H \boldsymbol{e}^T = e_1 \boldsymbol{h}_1 + e_2 \boldsymbol{h}_2 + \cdots + e_n \boldsymbol{h}_n$$

其中 $\boldsymbol{h}_1, \dots, \boldsymbol{h}_n$ 是 $H$ 的列向量。逐项拆解：

- **第一步，线性组合视角**：伴随式是错误非零位置对应列的加权和。若只有单个错误在第 $j$ 位（$e_j = 1$，其余为 0），则 $\boldsymbol{s}^T = \boldsymbol{h}_j$——**伴随式就是出错列的编号**。
- **第二步，查表反解**：把每个可能伴随式 $\boldsymbol{s}$ 对应的最小重量错误 $\boldsymbol{e}$（陪集首）预先存成表，译码时查表。这就是「伴随式 → 陪集首」的查找表，大小 $q^{n-k}$，远小于码字总数。
- **第三步，多错误的情形**：若错误数超过 1，$\boldsymbol{s}$ 是多个列的线性组合，反解可能不唯一。选择最小重量解——这正是陪集首——等价于最大似然。

**要点：** 伴随式译码的复杂度 = 算一次 $H \boldsymbol{r}^T$（$O(n(n-k))$ 次运算）+ 查一次表。它把「指数级搜索」的译码问题，变成了「多项式时间 + 小表」的工程问题，是线性码理论对实践最重要的贡献。

## 5 Hamming 码：伴随式就是错误位置

**Hamming 码**：取 $m \ge 2$，让校验矩阵 $H$ 的列恰好是 $1$ 到 $2^m - 1$ 的所有非零 $m$ 位向量（任意顺序）。这给出参数

$$[n, k, d] = [2^m - 1, \; 2^m - 1 - m, \; 3]$$

前面已证 $d = 3$（任意两列不同故无关，存在三列相关）。$t = \lfloor (3-1)/2 \rfloor = 1$，即 Hamming 码**纠正全部单个错误**。<span class="marginnote">$\mathrm{wt}(\boldsymbol{e}) = 1$ 时，$H \boldsymbol{e}^T$ 就是出错那一列的列向量。而 $H$ 的列是 $1$ 到 $2^m-1$ 的二进制编码——所以<strong>伴随式本身就是出错位置的二进制编号</strong>，连查表都省了。</span>

**$[7,4]$ Hamming 码的译码过程：**

设校验矩阵（按二进制列）

$$H = \begin{pmatrix} 0 & 0 & 0 & 1 & 1 & 1 & 1 \\ 0 & 1 & 1 & 0 & 0 & 1 & 1 \\ 1 & 0 & 1 & 0 & 1 & 0 & 1 \end{pmatrix}$$

收到 $\boldsymbol{r} = 1011001$。算伴随式 $\boldsymbol{s}^T = H \boldsymbol{r}^T$：

- 第 1、3、4、7 位为 1，对应列 $001, 011, 100, 111$；
- 三列二进制加和（逐位异或）：$001 + 011 = 010$，$010 + 100 = 110$，$110 + 111 = 001$；
- $\boldsymbol{s} = 001$，即二进制数 1——错误在第 1 位！

翻转第 1 位，得码字 $\boldsymbol{c} = 0011001$。<span class="marginnote">整个译码是三次异或加法加一次查「$\boldsymbol{s}$ = 位置」——硬件上就是几根异或门连线。Hamming 码因此成为历史上第一个投入工程使用的纠错码，今天仍活跃在内存 ECC 里。</span>

**Hamming 码与完美性**：$[7,4]$ 码有 $2^7 = 128$ 个向量、$16$ 个码字，每个码字周围的 1 错误球（含 1 + 7 = 8 个向量）两两不交，$16 \times 8 = 128$ 恰好铺满全空间——Hamming 码是**完美码（perfect code）**，冗余被利用到极致，一个球都不浪费。

## 6 小结

- **伴随式** $\boldsymbol{s} = H\boldsymbol{r}^T = H\boldsymbol{e}^T$：只依赖错误、不依赖码字，是「错误的指纹」。
- $\mathbb{F}_q^n$ 按伴随式划分为 $q^{n-k}$ 个**陪集**；每个陪集选最小重量向量为**陪集首**。
- **标准阵列**把全空间排成一张表：先算伴随式定行，再取行首错误、翻转即得码字。
- 伴随式译码 = 最大似然译码的线性实现，复杂度从指数降到多项式。
- **Hamming 码** $[2^m-1, 2^m-1-m, 3]$：$H$ 的列是全部非零 $m$ 位向量，伴随式直接读出错误位置；$[7,4]$ 码是完美码。
- 单错误纠正 + 工程极简实现，让 Hamming 码成为内存 ECC 等场景沿用至今的基石。
- 记住「保证纠 $t$ 错」是下界：标准阵列常能额外纠正若干重量 $> t$