---
title: 期望线性时间选择：RANDOMIZED-SELECT
date: 2026-08-07
---

# 期望线性时间选择：RANDOMIZED-SELECT

<div class="epigraph">
<p>快速排序只递归一边，期望就从 $n\log n$ 掉到 $n$——顺序统计量是分治的轻骑兵。</p>
<footer>—— 查尔斯 · 勒瑟森（Charles E. Leiserson）</footer>
</div>

<div class="article-byline">
<p>第三级 · 算法设计与分析 ｜ 《算法导论》（CLRS）第 9.2 节 ｜ 2026-08-07</p>
</div>

## 为什么从 RANDOMIZED-SELECT 开始

上一课解决了最值；这一课把问题一般化：**找第 $k$ 小的元素**（如中位数，$k = \lceil n/2 \rceil$）。排序显然可行（$O(n\log n)$），但直觉上「只要第 $k$ 小」不该比排序更贵。<span class="marginnote">RANDOMIZED-SELECT 是快速排序的「单臂」版本：同样随机选主元划分，但只递归包含第 $k$ 小元素的那一边。由于每层只处理一个子问题，期望时间从快排的 $n\log n$ 掉到 $n$。这是「分治 + 只走一枝」的范式教科书。</span>

这一课回答两个问题：算法如何工作，以及**期望 $O(n)$ 的证明**如何建立在「划分位置的期望」上。

## 1 算法：随机划分后只走一边

```
RANDOMIZED-SELECT(A, p, r, i)
  if p == r
    return A[p]
  q = RANDOMIZED-PARTITION(A, p, r)   // 主元最终位置 q
  k = q - p + 1                        // 主元是第 k 小的元素
  if i == k
    return A[q]
  else if i < k
    return RANDOMIZED-SELECT(A, p, q-1, i)
  else
    return RANDOMIZED-SELECT(A, q+1, r, i-k)
```

划分后主元落在 $q$，它在区间内是第 $k = q-p+1$ 小。若目标 $i = k$，主元就是答案；若 $i < k$，答案在左半；否则在右半（第 $i-k$ 小）。<span class="marginnote">与快排的对比一望而知：快排对两半都递归，选择只递归包含答案的那一半。这个「剪枝」正是期望从 $n\log n$ 变 $n$ 的原因——每层工作量为 $O(n)$，而递归深度取决于「运气好时主元接近中位数」。</span>

**辨析｜易错点：** 递归右半时下标要写 $i-k$ 而不是 $i$——右半区间的第 1 小元素整体上已经是第 $k+1$ 小。漏掉这个偏移，右半的选择结果就错了。

## 2 期望时间：最坏是 $O(n^2)$，期望是 $O(n)$

**最坏情况**：每次随机划分都选到极值主元，只削掉一个元素——$T(n) = T(n-1) + O(n) = O(n^2)$。与随机化快排一样，最坏存在但概率指数衰减。

**期望**：设 $T(n)$ 是 $n$ 个元素的期望时间。主元落在位置 $q$ 时，子问题规模为 $\max(k, n-k)$（因为只走较大的一边——目标可能在任一侧，最坏按较大侧算）。主元均匀随机，$k$ 取 $1..n$ 各概率 $1/n$，得递归式：

$$E[T(n)] \le \frac{1}{n}\sum_{k=1}^{n} E[T(\max(k-1, n-k))] + O(n)$$

化简（把对称项合并，去掉 $k$ 靠近两端时的较小项）：

$$E[T(n)] \le \frac{2}{n}\sum_{k=\lfloor n/2 \rfloor}^{n-1} E[T(k)] + O(n)$$

这一课的核心是**解这个递归式**，结论 $E[T(n)] = O(n)$。<span class="marginnote">对比快排的期望递归式：快排对<strong>每个</strong> $k$ 都付出 $T(k) + T(n-1-k)$，选择只付出 $\max(T(k), T(n-1-k))$——一个「两臂之和」，一个「单臂之大」。这一字之差，决定了 $n\log n$ 与 $n$ 的天壤之别。</span>

## 3 公式解析：代入法解出 $O(n)$

用代入法证明 $E[T(n)] \le cn$。假设对 $m < n$ 成立 $E[T(m)] \le cm$，代入：

$$E[T(n)] \le \frac{2}{n}\sum_{k=\lfloor n/2 \rfloor}^{n-1} ck + O(n) = \frac{2c}{n}\left(\sum_{k=1}^{n-1}k - \sum_{k=1}^{\lfloor n/2\rfloor - 1}k\right) + O(n)$$

三步拆解：

- **第一步，求和**：$\sum_{k=1}^{n-1}k = \frac{n(n-1)}{2}$；$\sum_{k=1}^{\lfloor n/2\rfloor-1}k = \frac{(\lfloor n/2\rfloor-1)\lfloor n/2\rfloor}{2} \le \frac{n^2}{8}$。
- **第二步，代回**：$\frac{2c}{n}\left(\frac{n(n-1)}{2} - \frac{n^2}{8}\right) = \frac{2c}{n}\cdot \frac{3n^2 - 4n}{8} = \frac{3cn}{4} - c$。
- **第三步，收紧**：要求 $\frac{3cn}{4} - c + O(n) \le cn$，即 $\frac{cn}{4} \ge O(n) - c$。选足够大的 $c$（比如 $c \ge 4$ 倍常数项）即可满足。

**结论**：$E[T(n)] = O(n)$——**期望线性时间**。<span class="marginnote">注意上界里用 $\max$ 并把较小项丢弃，都是「放大」；放大后仍能证明线性，说明真正的期望更紧。这个证明的妙处在于：即使每次划分的运气都很差（主元总偏向极端），期望仍线性——因为「差运气」本身是低概率的。</span>

**辨析｜易错点：** 不要把这个递归式与快排的搞混。快排对每个 $k$ 付 $T(k)+T(n-1-k)$，总和 $\approx 2\sum T(k)$，解得 $n\log n$；选择只付 $\max$，单臂，解得 $n$。**区别在于「分治是否只走一枝」**——这是选择问题比排序便宜的本质。

## 4 何时用 RANDOMIZED-SELECT

适用场景：需要**第 $k$ 小/中位数**，且 $n$ 很大、排序显得浪费。典型应用：

- **中位数求法**：$k = \lceil n/2\rceil$，快速获得中位数；
- **分治算法的枢纽**：快排的三数取中、划分里的中位主元、最坏线性选择的子步骤；
- **流式统计**：配合在线划分处理「随时问中位数」。
- **带权中位数**：后续「最近邮局问题」用带权中位数定位一维优化点。<span class="marginnote">RANDOMIZED-SELECT 与快速排序共享同一个 PARTITION，因此在已有快排代码的系统里，实现选择几乎零成本。相比下一课的最坏线性 SELECT，它更简单、常数更小，只是不保证最坏线性。</span>

若需要**保证最坏 $O(n)$**（如对手可控输入的库），就用下一课的 SELECT——它用「中位数的中位数」保证每次划分都不太坏。

## 5 小结

- **RANDOMIZED-SELECT**：随机划分后**只递归包含第 $k$ 小的一侧**，是快排的「单臂」变体。
- 最坏 $O(n^2)$ 但概率指数小；**期望 $O(n)$**。
- 期望递归式 $E[T(n)] \le \frac{2}{n}\sum E[T(k)] + O(n)$，代入法解得 $O(n)$。
- 与快排的本质区别：快排两臂递归得 $n\log n$，选择单臂递归得 $n$。
- 用于中位数、快排枢纽、带权中位数等「只要第 $k$ 小」的场景。

在下一课，我们去掉「期望」二字——**SELECT 与中位数的中位数**：通过巧妙的五分五组，保证每次划分至少淘汰固定比例，从而最坏情况也 $O(n)$。
