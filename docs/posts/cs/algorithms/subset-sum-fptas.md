---
title: 子集和问题的完全多项式时间近似方案（FPTAS）
date: 2026-08-07
---

# 子集和问题的完全多项式时间近似方案（FPTAS）

<div class="epigraph">
<p>当数字大到伪多项式跑不动，就把它变小——损失一点精度，换来对 $1/\varepsilon$ 也多项式的时间。</p>
<footer>—— 奥斯卡 · 伊巴拉 与 金基勋（Oscar Ibarra & Chul Kim）</footer>
</div>

<div class="article-byline">
<p>第三级 · 算法设计与分析 ｜ 《算法导论》（CLRS）第 35.5 节 ｜ 2026-08-07</p>
</div>

## 为什么从 FPTAS 开始

子集和是 NPC，但它有个 $O(nt)$ 的伪多项式 DP——数值 $t$ 受控时完全可行。**FPTAS（完全多项式时间近似方案）**把这个优点发挥到极致：**把数值缩放到可控范围，牺牲任意小的精度，换来对 $n$ 与 $1/\varepsilon$ 都多项式的算法**。<span class="marginnote">FPTAS 是近似算法的「黄金档」：对任意精度 $\varepsilon$，时间 $\text{poly}(n, 1/\varepsilon)$——它同时是「近似比任意接近 1」与「时间对 $1/\varepsilon$ 也多项式」。子集和是「伪多项式 DP 如何升级成 FPTAS」的教科书：把数值缩小，DP 就变快；误差被缩放比例钉住。</span>

这一课讲「列表合并 + 修剪」的精确算法、以及「缩放 + DP」的 FPTAS 构造。

## 1 子集和的精确伪多项式 DP

**子集和**：$n$ 个整数 $S = \{s_1, \dots, s_n\}$、目标 $t$，问能否选子集和为 $t$。

**精确 DP**（伪多项式）：设 `dp[w]` = 能否用前若干个数凑出 $w$。逐数更新：

```text
dp[0] = true; 其余 dp[w] = false
for each s in S:
    for w = t downto s:
        dp[w] = dp[w] OR dp[w - s]   // 不选 / 选 s
答案 = dp[t]
```

时间 $O(nt)$——$t$ 大时不可行（0-1 背包课已分析）。**FPTAS 的思路：把 $s_i$ 与 $t$ 按比例缩小，让 DP 在「缩小后的数值范围」上跑**——精度损失被控制在 $\varepsilon$ 内。

## 2 精确算法：列表合并 + 修剪

另一个精确思路：维护「所有可达和」的列表 $L$，每处理一个数 $s_i$，把 $L$ 与「$L$ 每个元素加 $s_i$」合并、去重、删掉 > $t$ 的项：

$$L \leftarrow \text{MERGE-LISTS}(L, L + s_i), \quad \text{删去 } > t \text{ 的项}$$

**问题**：列表大小指数增长（$2^n$ 个可能和）。**修剪（trim）**把「接近的项」合并成一个，控制列表规模：

**修剪规则**：若 $y \le x \le (1+\delta)y$，则删掉 $x$（用 $y$ 代表）——相对误差 ≤ $\delta$ 的项被合并。修剪后列表大小被压到 $O(\log t / \delta)$ 量级。<span class="marginnote">修剪的直觉：两个和相差不足 $\delta$ 倍，就用小的那个代表——反正目标「接近 $t$」的判断不会因 $\delta$ 级误差改变结论。这个「用 δ 容忍误差换列表规模」是近似算法的核心交易：<strong>精确性的一部分换取可计算性</strong>，误差全部可控在 $\varepsilon$ 内。</span>

## 3 FPTAS：缩放 + 精确 DP

**FPTAS 构造**（对最大化版本：找不超过 $t$ 的最大子集和）：

```text
FPTAS-SUBSET-SUM(S, t, ε)
  n = |S|,  K = max(1, ⌊ε·t / n⌋)         // 缩放因子
  S' = { ⌊s_i / K⌋ : s_i ∈ S }            // 缩放后的整数集合
  L = ⟨0⟩                                 // 可达和列表
  for i = 1 to n:
      L = MERGE-LISTS(L, L + s'_i)        // 合并：不加 / 加第 i 个
      L = TRIM(L, ε / n)                  // 修剪：δ 相对误差内合并
      从 L 中删去 > ⌊t/K⌋ 的项
  return max(L) · K                       // 按原始尺度还原
```

**关键参数**：修剪阈值 $\delta = \varepsilon / n$——把「总误差」分摊到 $n$ 步，每步误差 ≤ $\delta$，累积误差 ≤ $n\delta = \varepsilon$。<span class="marginnote">「$\delta = \varepsilon/n$」是误差分摊的经典手法：每一步修剪引入 ≤ $\delta$ 的相对误差，$n$ 步累积 ≤ $n\delta = \varepsilon$。若直接取 $\delta = \varepsilon$，$n$ 步累积会放大 $n$ 倍。把预算「按步均摊」是近似算法的通用技巧——与摊还分析的「按操作分摊」同构。</span>

## 4 公式解析：为什么时间对 $1/\varepsilon$ 是多项式

**修剪后列表大小**：$L_i$ 中相邻元素的比值 ≥ $1 + \delta$，且都在 $[1, t]$ 内，所以大小 ≤ $\log_{1+\delta} t = O(\log t / \delta)$。总时间：

$$T(n, t, \varepsilon) = O\left(n \cdot \frac{\log t}{\delta}\right) = O\left(\frac{n \log t}{\varepsilon/n}\right) = O\left(\frac{n^2 \log t}{\varepsilon}\right)$$

- **第一步，列表大小**：$\log_{1+\delta} t$——$t$ 以内「每项至少差 $1+\delta$ 倍」最多能放这么多个。
- **第二步，代入 $\delta = \varepsilon/n$**：$\log_{1+\delta} t \approx (\log t)/\delta = (n\log t)/\varepsilon$。
- **第三步，总时间**：$n$ 步 × 每步 $O(\text{列表大小})$ = $O(n^2\log t / \varepsilon)$——**对 $n$ 与 $1/\varepsilon$ 都是多项式**，且与 $t$ 只差 $\log t$。

**近似比**：输出 ≥ $(1-\varepsilon) \times$ 最优。因为每步修剪只损失 $\delta$ 相对精度，$n$ 步后仍保 $1-\varepsilon$——**FPTAS 定义（对任意 $\varepsilon$ 有 $(1\pm\varepsilon)$-近似且时间 $\text{poly}(n,1/\varepsilon)$）被满足**。<span class="marginnote">这个构造的妙处：<strong>伪多项式 DP 的「数值依赖」被缩放消除</strong>——本来 $O(nt)$ 依赖 $t$ 的大小，现在 $t$ 只以 $\log t$ 出现（因为 DP 处理的是「修剪后的稀疏列表」而非「整个 $[1,t]$ 数组」）。FPTAS 把「对数值敏感」的算法变成「几乎对数值不敏感」。这也解释了为什么子集和有 FPTAS 而 TSP 没有——子集和的数值结构允许缩放，TSP 的组合结构不允许。</span>

**辨析｜易错点：** FPTAS 的近似比是**乘法**的（$1-\varepsilon$ 相对误差），不是加法的。且它要求**时间对 $1/\varepsilon$ 多项式**——PTAS 的 $n^{O(1/\varepsilon)}$ 不算 FPTAS（$\varepsilon$ 在指数里）。**「$1/\varepsilon$ 进多项式」是 FPTAS 与 PTAS 的分界线**（近似比课的定义，这里落地）。

## 5 FPTAS 的意义与边界

**意义**：子集和/0-1 背包有 FPTAS——实践中「任意精度近似」是可达的。这是「NPC ≠ 无解」的最强形式：不是近似到常数倍，而是近似到任意 $(1+\varepsilon)$ 倍，且时间多项式。

**边界**：并非所有 NPC 问题都有 FPTAS。**TSP（一般情形）没有**——若 TSP 有 FPTAS，用 $\varepsilon$ 足够小就能精确解它（多项式），与 NPC 矛盾。**「有 FPTAS ⟺ 问题有伪多项式算法且非强 NPC」**是近似的经典刻画。<span class="marginnote">强 NPC 与 FPTAS 的关系：强 NP 完全问题（TSP 的判定版在「一元编码」下仍 NPC）<strong>没有 FPTAS</strong>（除非 P=NP）。子集和不是强 NPC（数值可以缩放），所以有 FPTAS。这个「可缩放性」区分了「能任意近似」与「只能常数近似」——近似比课已埋下伏笔，这里闭合。</span>

## 6 小结

- **精确算法**：列表合并 + 修剪（$\delta$ 相对误差内合并），$O(n\log t/\delta)$ 时间。
- **FPTAS**：$\delta = \varepsilon/n$ 把误差按步均摊，累积 ≤ $\varepsilon$。
- **复杂度** $O(n^2\log t/\varepsilon)$——对 $n$ 与 $1/\varepsilon$ 多项式，$t$ 只以 $\log t$ 出现。
- **近似比**：输出 ≥ $(1-\varepsilon)$ 最优——任意精度可达。
- 子集和有 FPTAS（非强 NPC）；TSP 没有（强 NPC）——「可缩放性」决定近似能力。

在下一课，我们进入专题的最后一个前沿——**字符串匹配进阶**的第一课：朴素匹配的缺陷分析，为 Rabin-Karp、KMP、Boyer-Moore 立靶子。
