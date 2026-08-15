---
title: 树状数组的原理与区间求和
date: 2026-08-07
---

# 树状数组的原理与区间求和

<div class="epigraph">
<p>用最低位的 1，把前缀和织成一张网。</p>
<footer>—— 树状数组格言</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据结构 ｜ 严蔚敏《数据结构》 专题篇·线段树与树状数组 ｜ 2026-08-07</p>
</div>

## 为什么「单点改 + 区间查」要专门结构

朴素数组支持「单点改 $O(1)$、前缀和 $O(n)$」；前缀和数组支持「前缀和 $O(1)$、单点改 $O(n)$」——**总有一边是 $O(n)$**。**树状数组（Fenwick Tree / Binary Indexed Tree）**把两边同时压到 $O(\log n)$：它用一棵「基于最低位的 1」的隐式树，让**单点修改与区间求和都是 $O(\log n)$**。树状数组代码极短（十几行）、常数极小、空间 $O(n)$，是「单点改 + 区间查」这类问题的首选——也是线段树（下几节）的轻量替身。

## 1 树状数组的定义：C[i] 管一段

树状数组用数组 `tree` 记录「若干原数组元素的和」。关键约定：

$$
tree[i] = \sum_{j = i - lowbit(i) + 1}^{i} a[j]
$$

其中 $i$ = $i$ 的二进制中**最低位的 1 所代表的值**，即 $6 = (110)_2$。$4 = (100)_2$ 管住「以 $i$ 结尾、长度为 lowbit(i) 的一段」——**下标 $i$ 管辖的长度，由它二进制里最低位的 1 决定**。<span class="marginnote">「<strong>lowbit(i) = i & (-i)</strong>」是树状数组的灵魂：<strong>$i$ 管一段长度恰好等于 lowbit(i) 的区间</strong>。例：$6 = (110)_2$，lowbit = 2，管 $4 = (100)_2$；$4 = (100)_2$，lowbit = 4，管 $i$。<strong>下标的下位算术，直接决定管辖范围</strong>。</span>

## 2 前缀和查询：沿 lowbit 跳

求前缀和 `prefixSum(i)`（前缀和）：

累加 `tree[i]`；
`i -= lowbit(i)` ——跳到「上一段」；
重复直到 $i = 0$。

```c
int prefixSum(int i) {
    int s = 0;
    while (i > 0) {
        s += tree[i];
        i -= lowbit(i);      /* 跳到「上一段」 */
    }
    return s;
}
```

**重点：前缀和 =「沿 lowbit 链累加」，$i$ 每步至少减半，所以 $O(\log n)$。** 例：求前缀和 `prefixSum(7)`：累加 `tree[7]`（a7）→ `tree[6]`（a5,a6）→ `tree[4]`（a1..a4），三段恰好盖全。<span class="marginnote">「<strong>前缀和 = 若干个 disjoint 段拼起来</strong>」：<strong>$7 \to 6 \to 4 \to 0$ 每步跳过一整段</strong>——<strong>二进制里 $i$ 的 1 的个数，就是累加的段数</strong>（≤ $\log n$）。<strong>「跳段」是树状数组一切操作的通用动作</strong>。</span>

## 3 单点修改：沿 lowbit 上爬

修改 `a[i]`，要更新所有「管辖 $i$ 的 `tree`」：

更新 `tree[i]`；
`i += lowbit(i)` ——跳到「下一个管辖 $i$ 的结点」；
重复直到越界。

```c
void add(int i, int delta) {
    while (i <= n) {
        tree[i] += delta;
        i += lowbit(i);      /* 跳到「下一个管辖 i 的结点」 */
    }
}
```

**重点：单点修改 =「沿 lowbit 上爬」，更新的结点数 ≤ $\log n$，$O(\log n)$。** 例：改 `a[1]`，更新 `tree[1]`（a1）→ `tree[2]`（a1,a2）→ `tree[4]`（a1..a4）→ `tree[8]`（a1..a8）……<span class="marginnote">「<strong>修改上爬、查询下跳</strong>」是树状数组的双向舞步：<strong>改一个点，要通知所有「管得到它的段」；查一段和，把「盖住它的段」拼起来</strong>。<strong>两个方向都沿 lowbit 走，都是 $\log n$ 步</strong>。</span>

## 4 公式解析：区间求和 = 两个前缀和

区间 $[l, r]$ 的和 = 前缀和相减：

$$
\sum_{i=l}^{r} a[i] = \text{PrefixSum}(r) - \text{PrefixSum}(l-1)
$$

$$
T_{\text{区间求和}} = O(\log n), \qquad T_{\text{单点修改}} = O(\log n), \qquad S = O(n)
$$

- **第一步，读「前缀和相减」**：两个前缀和都是 $O(\log n)$，区间查询 $O(\log n)$。
- **第二步，读「为什么快」**：朴素前缀和数组区间查询 $O(1)$ 但单点改 $O(n)$；树状数组把两者都压到 $O(\log n)$——**没有 $O(n)$ 的操作**。
- **第三步，读「均摊无」**：树状数组的 $O(\log n)$ 是每步都有的稳定上界，无摊销、无退化——**确定性好**。<span class="marginnote">「<strong>树状数组 = 两全其美</strong>」：<strong>数组的快改、前缀和的快查，它都拿到了，代价只是两个 $O(\log n)$</strong>。<strong>「没有一个操作是 $O(n)$」</strong>——这在「既要改又要查」的场景里是质的飞跃，也是它无法被朴素结构替代的原因。</span>

## 5 树状数组 vs 朴素结构

| 维度 | 数组 | 前缀和数组 | 树状数组 |
| --- | --- | --- | --- |
| 单点修改 | $O(1)$ | $O(n)$ | $O(\log n)$ |
| 区间求和 | $O(n)$ | $O(1)$ | $O(\log n)$ |
| 代码量 | 最少 | 少 | 十几行 |
| 空间 | $O(n)$ | $O(n)$ | $O(n)$ |
| 适用 | 只查不改 | 只改不查 | **改查都要** |

**重点：树状数组是「动态前缀和」的最优轻量解——改查都 $O(\log n)$，代码比线段树短得多。** 当问题只是「单点改 + 区间查」，树状数组应优先于线段树。<span class="marginnote">「<strong>能树状数组就树状数组</strong>」是竞赛与工程的经验法则：<strong>单点改 + 区间查 → 树状数组；区间改 + 区间查、区间最值 → 线段树</strong>。<strong>杀鸡不用牛刀——结构的选型要先问「操作的组合」</strong>。</span>

**一个 lowbit 与管辖段的算例。** 设 $n=8$：

- $lowbit(1)=1$：`tree[1]` 管 `a[1]`；
- $lowbit(2)=2$：`tree[2]` 管 `a[1..2]`；
- $lowbit(3)=1$：`tree[3]` 管 `a[3]`；
- $lowbit(4)=4$：`tree[4]` 管 `a[1..4]`；
- $lowbit(8)=8$：`tree[8]` 管 `a[1..8]`。

可见**管辖段长 = 二进制里最低位的 1 对应的值**；下标为 2 的幂的结点管辖最长的前缀。前缀和 7：`tree[7]+tree[6]+tree[4]` = a7 + a5,a6 + a1..a4，恰好盖全 1..7。<span class="marginnote">「<strong>2 的幂下标 = 管最长前缀</strong>」是个速查规律：<strong>$tree[8]$ 管全数组、$tree[4]$ 管前 4、$tree[2]$ 管前 2</strong>。<strong>前缀和查询每次跳掉「最低位的 1」，所以步数 = 二进制中 1 的个数</strong>——$7=(111)_2$ 恰好 3 段。</span>

**术语速查表**

| 术语 | 含义 |
| --- | --- |
| lowbit(i) | $i \& (-i)$，最低位的 1 的值 |
| 管辖段 | $tree[i]$ 管 $[i-\text{lowbit}+1, i]$ |
| 前缀和 | 沿 lowbit 下跳累加 |
| 单点修改 | 沿 lowbit 上爬更新 |
| 区间求和 | 两个前缀和相减 |

## 6 小结

- `tree[i]` 管 $[i - \text{lowbit}(i) + 1,\ i]$ 的和；管长 = lowbit(i)。
- 前缀和：沿 lowbit 向下跳，累加各段，$O(\log n)$。
- 单点修改：沿 lowbit 向上爬，更新所有管辖段，$O(\log n)$。
- 区间求和 = `prefixSum(r) - prefixSum(l-1)`，$O(\log n)$。
- 改查都是 $O(\log n)$——动态前缀和的最优轻量解。
- 单点改 + 区间查 → 树状数组；区间改/最值 → 线段树。

在下一节，我们看树状数组的两个进阶应用——**树状数组的区间修改与逆序对计数**。
