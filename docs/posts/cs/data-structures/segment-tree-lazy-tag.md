---
title: 懒惰标记与区间修改
date: 2026-08-07
---

# 懒惰标记与区间修改

<div class="epigraph">
<p>能不做的修改就先记账，等真的要用时再结清。</p>
<footer>—— 线段树格言</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据结构 ｜ 严蔚敏《数据结构》 专题篇·线段树与树状数组 ｜ 2026-08-07</p>
</div>

## 为什么区间修改不能「真的改」

上一节的线段树只支持单点改。要「区间整体加 $v$」时，朴素做法是把这个区间拆成 $O(\log n)$ 段、逐段真的改——但每个段的子树里还有无数结点，全改就是 $O(n)$。**懒惰标记（lazy propagation）**的答案是：**区间修改只「记账」——在覆盖整个区间的结点上打一个标记，记录「这段该加 $v$」，但暂时不真正下推到子树**。等到查询要下探时，才把标记「下推」给孩子。这样区间修改与区间查询都 $O(\log n)$——**「把修改推迟到必须之时」是懒惰标记的灵魂**。

## 1 懒惰标记的思想：记账 + 结账

线段树每个结点多一个 `lazy` 字段：

- **区间修改**：若当前区间 `[l, r]` 完全在修改区间内 → `tree[node] += v * len` 并 `lazy[node] += v`，**不再下探**——「记账」；
- **下推（push down）**：当需要访问孩子时（查询或继续修改下探），先把 `lazy` 传给两个孩子并清空自己——「结账」。

**重点：懒惰标记让「完全覆盖的区间」只改自己、不下探——把区间修改从 $O(n)$ 降到 $O(\log n)$。**<span class="marginnote">「<strong>完全覆盖就打标记、不下探</strong>」是懒惰标记的全部：<strong>修改只需更新 $O(\log n)$ 个「完整覆盖段」，每个段只改 $tree$ 和 $lazy$ 两个值</strong>。<strong>「记账」的成本是 $O(1)$</strong>——区间修改因此与区间查询同阶。</span>

## 2 区间修改 + 下推的代码

```c
void PushDown(int node, int l, int r) {        /* 把 lazy 下推给孩子 */
    if (lazy[node] != 0) {
        int mid = (l + r) / 2;
        lazy[node*2] += lazy[node];            /* 左孩子记账 */
        tree[node*2] += lazy[node] * (mid - l + 1);
        lazy[node*2+1] += lazy[node];          /* 右孩子记账 */
        tree[node*2+1] += lazy[node] * (r - mid);
        lazy[node] = 0;                        /* 本结点结清 */
    }
}

void Update(int node, int l, int r, int ql, int qr, int v) {
    if (ql <= l && r <= qr) {                  /* 完全覆盖：记账不下探 */
        tree[node] += v * (r - l + 1);
        lazy[node] += v;
        return;
    }
    PushDown(node, l, r);                      /* 部分覆盖：先下推，再递归 */
    int mid = (l + r) / 2;
    if (ql <= mid) Update(node*2, l, mid, ql, qr, v);
    if (qr > mid)  Update(node*2+1, mid+1, r, ql, qr, v);
    tree[node] = tree[node*2] + tree[node*2+1];
}
```

**重点：`PushDown` 在「即将下探孩子」前必须调用**——保证孩子的 $tree$ 是最新值，否则孩子可能用了「过期」的聚合值。<span class="marginnote">「<strong>下探之前必下推</strong>」是懒惰线段树的铁律：<strong>只要还要往孩子走（查询或修改），就得先把本结点的 lazy 结清</strong>。<strong>漏一次下推，孩子读到的就是旧值</strong>——这是懒惰线段树最常见的 Bug 源。</span>

## 3 带懒惰的区间查询

区间查询也要「先下推再递归」——否则孩子可能带着未结清的 lazy，聚合值不对：

```c
int Query(int node, int l, int r, int ql, int qr) {
    if (ql <= l && r <= qr) return tree[node];   /* 完全覆盖：直接取 */
    PushDown(node, l, r);                        /* 下推，保证孩子最新 */
    int mid = (l + r) / 2, res = 0;
    if (ql <= mid) res += Query(node*2, l, mid, ql, qr);
    if (qr > mid)  res += Query(node*2+1, mid+1, r, ql, qr);
    return res;
}
```

**重点：查询与修改在「部分覆盖」时都先 `PushDown`**——懒惰标记的「推迟」到「必须精确值」时才结算。<span class="marginnote">「<strong>查询也要下推</strong>」常被忽略：<strong>如果只修改时下推、查询时不下推，查到的区间可能包含「被记账但未结算」的部分，聚合值偏小</strong>。<strong>「下推」的时机是「要读孩子的精确值之前」——修改、查询都一样</strong>。</span>

## 4 公式解析：懒惰线段树的复杂度

$$
\begin{aligned}
T_{\text{区间修改}} &= O(\log n) \quad \text{（只碰 $O(\log n)$ 个完整覆盖段）} \\
T_{\text{区间查询}} &= O(\log n) \quad \text{（拆段 + 沿途下推，仍对数）} \\
S &= O(4n) \quad \text{（多一个 lazy 数组）}
\end{aligned}
$$

- **第一步，读「为什么区间改也是 $O(\log n)$」**：区间被拆成 $O(\log n)$ 个完整段，每段只改自身 + 打标记，$O(1)$ 每段。
- **第二步，读「下推不改变阶」**：沿途的 `PushDown` 也是每层常数，总 $O(\log n)$。
- **第三步，读「记账的代价」**：多一个 `lazy` 数组、多一次 `PushDown` 调用——**常数略增，阶不变**。<span class="marginnote">「<strong>记账 $O(1)$、下推 $O(1)$、总 $O(\log n)$</strong>」是懒惰线段树的成本账：<strong>把「区间修改的真操作」推迟并分块，代价从 $O(n)$ 变 $O(\log n)$</strong>。<strong>「用一次下推，换整个子树的免改」</strong>——这笔交易让区间操作全部对数化。</span>

## 5 懒惰标记的适用与变体

- **适用**：区间加/减、区间赋值、区间乘（模）——一切「区间整体变化」的操作；
- **变体一**：区间赋值（整体设成 $v$）——lazy 存「赋值标记」，下推时直接覆盖；
- **变体二**：区间加 + 区间乘组合（需要**两个 lazy**，下推顺序有讲究：先乘后加）；
- **变体三**：历史最值、区间最大子段——lazy 配合更复杂的聚合。

**辨析｜易错点：多个 lazy 的下推顺序。** 若同时支持「加」与「乘」，下推时必须**先乘后加**——因为加法要「被乘数缩放」：`(x + add) * mul = x * mul + add * mul`。顺序错，数值全错。<span class="marginnote">「<strong>多标记下推：先乘后加</strong>」是懒惰线段树的进阶考点：<strong>乘法会缩放已有的加法，所以「乘」要先下推、把加法一起缩放</strong>。<strong>「标记的复合顺序」是懒惰传播最容易出错的地方</strong>——写前先想清楚运算的代数结构。</span>

## 6 小结

- 懒惰标记：区间修改「记账不下探」，查询时再下推结清。
- 完全覆盖：改 `tree` + 打 `lazy`，$O(1)$；部分覆盖：先 `PushDown` 再递归。
- 查询与修改在「下探孩子前」都要 `PushDown`。
- 区间修改 $O(\log n)$——把真操作推迟到 $O(\log n)$ 个完整段。
- 适用：区间加/赋值/乘；多标记下推「先乘后加」。
- 空间多一个 lazy 数组，常数略增、阶不变。

在下一节，我们看线段树的实战——**线段树的典型应用（区间最值、扫描线）**。
