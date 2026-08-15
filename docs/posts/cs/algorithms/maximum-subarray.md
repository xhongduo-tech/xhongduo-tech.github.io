---
title: 分治经典案例：最大子数组问题
date: 2026-08-07
---

# 分治经典案例：最大子数组问题

<div class="epigraph">
<p>把问题分解为若干规模更小、性质相同的子问题，然后递归求解，最后合并——许多看似困难的优化问题在这一范式下豁然开朗。</p>
<footer>—— 托马斯 · 科尔曼 等（Thomas H. Cormen）《算法导论》</footer>
</div>

<div class="article-byline">
<p>第三级 · 算法设计与分析 ｜ 《算法导论》（CLRS）第 4.1 节 ｜ 2026-08-07</p>
</div>

## 为什么从最大子数组开始

上一课我们学会了归并排序，见识了**分治（divide and conquer）**三件套：分解、递归、合并。但归并排序的分解是「按位置切一半」，太规整了——真实世界的分治往往需要一点创造性：**最优解到底藏在左边、右边，还是跨越中线？** 最大子数组问题就是回答这个问题的经典舞台。<span class="marginnote">这个问题在金融里就是「在哪个交易日买入、哪个交易日卖出收益最大」；在信号处理里是「哪一段能量最集中」。一个问题的应用场景越多，越值得把它当成思维范式来学。</span>

同时，它也是我们第一次遇到「分治解不是朴素解」的问题：最直接的枚举需要 $O(n^2)$ 个连续段逐一检查，而分治把它降到 $O(n \log n)$。**从平方到 $n \log n$，是算法设计能带来的、肉眼可见的收益。**

## 1 问题定义与朴素思路

**最大子数组问题（maximum subarray problem）**：给定一个长度为 $n$ 的数组 $A$（元素可正可负），寻找一个连续子数组 $A[l..r]$（$1 \le l \le r \le n$），使得区间和

$$\sum_{i=l}^{r} A[i]$$

在所有连续子数组中最大。注意**子数组必须是连续的**——这一点决定了它和「子序列」完全不同。

一个常见的直觉错误是「全正数时取全部，全负数时就该取空」。教材约定：**允许子数组为空时，空子数组的和为 0，答案至少是 0**；若要求非空，则全负数时答案是最大的那个负数。<span class="marginnote">这两条规定都能自洽，关键是一开始就说清楚。考试与面试里「能否取空」是必须主动确认的细节——很多错解就死在约定不明确上。</span>

朴素方法非常直白：枚举所有可能的 $l$ 和 $r$，对每个区间求和。共 $\Theta(n^2)$ 个区间，每个区间求和若再花 $O(n)$，总代价 $O(n^3)$；用前缀和技巧可以把单区间求和压到 $O(1)$，于是是 $O(n^2)$。**平方级在 $n$ 稍大时（比如 $10^5$）就不可行了**，这就是我们需要更聪明算法的理由。

## 2 分治思路：跨过中线的三种可能

设数组为 $A[low..high]$，取中点 $mid = \lfloor (low+high)/2 \rfloor$。任何一个最大子数组 $A[i..j]$，只有三种归属：

- **完全在左半边**：$low \le i \le j \le mid$；
- **完全在右半边**：$mid < i \le j \le high$；
- **跨越中线**：$i \le mid < j$，即左端点落在左半、右端点落在右半。

前两种交给递归，第三种专门处理。于是分治策略自然成型：

$$\text{FIND-MAXIMUM-SUBARRAY}(A, low, high) = \max\begin{cases} \text{FIND-MAXIMUM-SUBARRAY}(A, low, mid) \\ \text{FIND-MAXIMUM-SUBARRAY}(A, mid+1, high) \\ \text{MAX-CROSSING-SUBARRAY}(A, low, mid, high) \end{cases}$$

关键洞察：**跨越中线的最大子数组一定由「左半以 $mid$ 结尾的最大后缀」与「右半从 $mid+1$ 开始的最大前缀」拼接而成**。<span class="marginnote">为什么可以这么拼？因为跨越中线的子数组必然包含 $A[mid]$ 与 $A[mid+1]$，把这段区间拆成以 $mid$ 结尾的左段和以 $mid+1$ 开头的右段，两段互不影响，各取最优即可。这正是最优子结构的雏形，下一课动态规划会把它系统化。</span>

计算 MAX-CROSSING-SUBARRAY 只需从 $mid$ 向左扫一遍、从 $mid+1$ 向右扫一遍，各维护一个当前最大和，代价为 $\Theta(n)$。

## 3 代码骨架与递归式

伪代码（面向读者，语言无关）：

```text
FIND-MAXIMUM-SUBARRAY(A, low, high):
    if high == low:                         // 只有一个元素
        return (low, high, A[low])
    else:
        mid = ⌊(low + high) / 2⌋
        (left_low, left_high, left_sum) = FIND-MAXIMUM-SUBARRAY(A, low, mid)
        (right_low, right_high, right_sum) = FIND-MAXIMUM-SUBARRAY(A, mid+1, high)
        (cross_low, cross_high, cross_sum) = MAX-CROSSING-SUBARRAY(A, low, mid, high)
        if left_sum >= right_sum and left_sum >= cross_sum:
            return (left_low, left_high, left_sum)
        elif right_sum >= left_sum and right_sum >= cross_sum:
            return (right_low, right_high, right_sum)
        else:
            return (cross_low, cross_high, cross_sum)
```

MAX-CROSSING-SUBARRAY 的代价是线性的：向左累加并记录最大后缀和，向右累加并记录最大前缀和，返回两边拼起来的三元组。

于是总运行时间满足递归式：

$$T(n) = 2T(n/2) + \Theta(n)$$

这正是一等分情形下的**主定理 Case 2**（$\log_b a = \log_2 2 = 1 = d$），解为 $T(n) = \Theta(n \log n)$。若左右两边不等长，比如 $T(n) = T(\lceil n/2 \rceil) + T(\lfloor n/2 \rfloor) + \Theta(n)$，结论不变。<span class="marginnote">这一步与归并排序的递归式一模一样——分治的收益来自「每层合计线性，共 $\log n$ 层」。记住这个模式：递归式长成 $2T(n/2)+\Theta(n)$，答案就是 $\Theta(n \log n)$。</span>

## 4 公式解析：为什么 $T(n) = 2T(n/2) + \Theta(n)$ 给出 $n \log n$

把递归式展开成递归树，看每一层的合计代价：

- **第一层**：规模 $n$，代价 $c n$（其中 $c$ 是常数，来自 MAX-CROSSING 的线性扫描）。
- **第二层**：两个规模 $n/2$ 的子问题，每个代价 $c(n/2)$，合计 $2 \cdot c(n/2) = c n$。
- **第 $k$ 层**：$2^k$ 个规模 $n/2^k$ 的子问题，合计 $2^k \cdot c(n/2^k) = c n$。

每层的代价都是 $cn$，而层数从 $n$ 到 $1$ 共 $\log_2 n$ 层，因此总代价

$$T(n) = cn \cdot \log_2 n = \Theta(n \log n)$$

对这条式子做三步拆解：

- **第一步，看递归项 $2T(n/2)$**：它表示「问题被对半分成两个独立子问题」，这是分治「分解 + 递归」的代价。
- **第二步，看合并项 $\Theta(n)$**：MAX-CROSSING 需要从左到右、从右到左各扫一遍，线性时间——**合并代价不能超过 $O(n)$，否则主定理会给出更大的阶**。
- **第三步，看层合计不变**：$2^k \cdot c(n/2^k) = cn$ 恰好抵消了规模缩小，每层工作量相等。这种「每层等重」的分治树，总和必然是「层重 × 层数」=$cn \cdot \log n$。

**辨析｜易错点：** 不要把最大子数组问题的分治解当成「左右各递归一次」就完事。**跨越中线的子数组不是两个子问题的解拼接那么简单——它自己需要一次 $\Theta(n)$ 的专门扫描**。遗漏这一项，递归式会变成 $T(n) = 2T(n/2)$，解只有 $\Theta(n)$，那显然不对，因为问题答案可能恰好横跨中线。

## 5 从分治到更优：前缀扫描的线性算法

分治给了我们 $O(n \log n)$，但这远不是终点。**最大子数组有一个更漂亮的线性算法（Kadane 算法）**，它不用分治，而是用「以 $A[i]$ 结尾的最大子数组和」做一维动态规划：

$$\text{best}[i] = \max\big(A[i],\; \text{best}[i-1] + A[i]\big), \qquad \text{ans} = \max_i \text{best}[i]$$

直觉是：以 $A[i]$ 结尾的最大后缀，要么只含 $A[i]$ 自己，要么把前面以 $A[i-1]$ 结尾的最大后缀接上。这比分治更快，但也更难自然想到——**分治解的价值恰恰在于它提供了一条从「朴素」到「可证明正确」的稳妥路径**。<span class="marginnote">Kadane 的递推 $\text{best}[i]$ 就是动态规划里的「状态」与「转移」，到第七篇《动态规划》你会见到它作为教科书级的入门例子反复出现。</span>

**辨析｜易错点：** 当数组全为负数时，Kadane 的递推会不断「丢弃前缀」——每一步都取 $\max(A[i], \text{best}[i-1]+A[i])$，结果 $\text{best}[i]$ 总是等于当前最大单个元素，这正是「非空子数组」约定的正确答案；若允许空子数组，需把答案初值设为 0 并允许「不取」。

## 6 小结

- **最大子数组问题**要求一段连续区间使和最大；朴素枚举是 $O(n^2)$。
- **分治的关键**是把解按位置分成三种：全左、全右、跨中线；跨中线的解 = 左最大后缀 + 右最大前缀。
- 跨越中线的扫描代价 $\Theta(n)$，递归式 $T(n)=2T(n/2)+\Theta(n)$，由主定理得 $\Theta(n \log n)$。
- **合并项决定了分治的最终复杂度**：合并若是线性，分治才有 $n \log n$ 的收益。
- Kadane 的线性算法用一维动态规划做到 $O(n)$，是同一问题更优的解法。

**数值快照**：对 $A = [-2, 1, -3, 4, -1, 2, 1, -5, 4]$，Kadane 逐位维护「以 $i$ 结尾的最大后缀」：$\text{best} = [-2, 1, -2, 4, 3, 5, 6, 1, 5]$，最大值 6——对应子数组 $[4, -1, 2, 1]$。分治在同一数组上通过「跨中线的最大后缀+前缀」也能得到 6，两条路线殊途同归。

在下一课，我们继续分治的经典舞台，但这次要挑战**矩阵乘法**——当「合并」不再是线性而牵涉三次矩阵乘积时，Strassen 用 7 次递归乘法改写递归式，把复杂度从 $n^3$ 压到 $n^{2.81}$。
