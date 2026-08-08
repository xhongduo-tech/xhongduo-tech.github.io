---
title: 优先队列：最大堆、最小堆与 d 叉堆
date: 2026-08-07
---

# 优先队列：最大堆、最小堆与 d 叉堆

<div class="epigraph">
<p>优先队列是操作系统的心脏、图算法的引擎——它在「谁先谁后」之间写下秩序。</p>
<footer>—— 安德鲁 · 塔能鲍姆（Andrew S. Tanenbaum）</footer>
</div>

<div class="article-byline">
<p>第三级 · 算法设计与分析 ｜ 《算法导论》（CLRS）第 6.5 节 ｜ 2026-08-07</p>
</div>

## 为什么从优先队列开始

堆排序展示了堆的排序能力，但堆更本质的角色是**优先队列（priority queue）**——一种支持「插入」与「取最大/最小元素」的抽象数据类型。它没有固定的「先进先出」纪律，而是让**优先级最高的元素最先被服务**。<span class="marginnote">优先队列无处不在：操作系统按优先级调度进程、网络路由器按优先级转发数据包、Dijkstra 最短路与 Prim 最小生成树用最小堆选「最近的未访问节点」、A\* 搜索用它选「最可能最优的节点」。可以说，贪心与图算法的「下一步选谁」，都由它承载。</span>

这一课系统梳理优先队列的四类操作，并延伸到两个变体：**最小堆**与 **d 叉堆**。

## 1 抽象数据类型与堆实现

优先队列是一个**动态集合**，元素各带一个**关键字（key）**，支持：

**INSERT**：把 $x$ 插入集合。
**MAXIMUM**：返回关键字最大的元素（不删除）。
**EXTRACT-MAX**：取出并删除关键字最大的元素。
**INCREASE-KEY**：把元素 $x$ 的关键字增大到 $k$（前提 $k \ge$ 原值）。

用最大堆实现时，堆顶就是最大元素，四个操作都在 $O(\log n)$ 内完成（MAXIMUM 是 $O(1)$）。**「动态集合」与「静态数组」的区别在这里显现**：排序是一次性把全部元素排好；优先队列则支持「元素随时进出、随时问极值」，每次操作只花对数时间。

## 2 四个操作的实现细节

### MAXIMUM 与 EXTRACT-MAX

```text
HEAP-MAXIMUM(A):                // 只读堆顶，O(1)
    return A[1]

HEAP-EXTRACT-MAX(A):
    if A.heap-size < 1: error "堆为空"
    max ← A[1]                  // 取堆顶
    A[1] ← A[A.heap-size]       // 末尾元素搬到堆顶
    A.heap-size ← A.heap-size − 1
    MAX-HEAPIFY(A, 1)           // 下沉修复堆序
    return max
```

EXTRACT-MAX 把末尾元素搬到堆顶再下沉——**这正是堆排序里「提取」那一步的原型**。MAXIMUM 只读堆顶，$O(1)$。<span class="marginnote">把末尾元素搬到堆顶的做法有个额外好处：数组的「空洞」总出现在末尾，堆结构始终紧凑。删除任意位置元素（如任务取消）需要用更复杂的 heap-delete，这里不展开。</span>

### INCREASE-KEY 与 INSERT

```text
HEAP-INCREASE-KEY(A, i, k):     // 前提：k ≥ A[i]
    A[i] ← k
    while i > 1 and A[PARENT(i)] < A[i]:
        交换 A[i] 与 A[PARENT(i)]
        i ← PARENT(i)           // 沿父链上浮
```

这里不是下沉而是**上浮（percolate up）**：把增大的关键字不断与父节点比较，若违反堆序就交换，直到到位。INSERT 则是先把它追加到末尾（赋一个极小哨兵），再 INCREASE-KEY 到目标值：

```text
MAX-HEAP-INSERT(A, k):
    A.heap-size ← A.heap-size + 1
    A[A.heap-size] ← −∞         // 极小哨兵
    HEAP-INCREASE-KEY(A, A.heap-size, k)
```

**辨析｜易错点：** INCREASE-KEY 的前提是「新关键字不小于旧值」；若要**减小**关键字（比如 Dijkstra 里距离变短），必须用另一套「下沉或上浮」的修改，不能直接套用本操作。工程里很多 bug 来自「把减小当成增大处理」导致堆序被破坏。

## 3 最小堆与最大堆的对偶性

把最大堆的所有比较反转（`>` 换 `<`），就得到**最小堆**：堆顶是全局最小。对应的四个操作是 MINIMUM、EXTRACT-MIN、DECREASE-KEY、INSERT。<span class="marginnote">DECREASE-KEY 是 Dijkstra/Prim 里「松弛更新后把更短距离上浮」的原语——它是最小堆最重要的操作，远比最大堆场景常用。很多算法教材以最小堆为默认讨论对象，正是图算法驱动的。</span>

**最大堆 vs 最小堆**不是两种数据结构，而是**同一份代码的符号翻转**。理解了最大堆的四个操作，最小堆只是把「谁更大」改成「谁更小」。

## 4 d 叉堆：调参空间里的折中

**d 叉堆（d-ary heap）**把每个节点从 2 个孩子扩展到 $d$ 个孩子。父子下标关系变为

$$\text{PARENT}(i) = \lfloor (i-2)/d \rfloor + 1,\qquad \text{child } j \text{ of } i = d(i-1) + j + 1$$

复杂度随之改变：**EXTRACT-MIN 要比较 $d$ 个孩子中的最小者，代价 $O(d \log_d n)$**；而 INCREASE-KEY/DECREASE-KEY 只沿一条父链上浮，代价 **$O(\log_d n)$**。<span class="marginnote">于是出现了一个调参权衡：$d$ 越大，上浮越快（路径更短）但提取越慢（每个节点要比 $d$ 个孩子）。当应用「更新频繁、提取较少」时（如 Dijkstra），取 $d \approx 4$ 或 $d$ 稍大往往在常数上胜出；应用「提取频繁」时取 $d=2$ 更稳。</span>

**辨析｜易错点：** 不要以为「孩子越多越差」。$d$ 叉堆的价值在**常数优化**：虽然都是 $O(\log n)$ 量级，但 $d$ 的选择改变了实际比较次数与缓存局部性。工程实现（如 Boost.Graph 的 Dijkstra）有时用斐波那契堆或配对堆在理论上更优，但 d 叉堆因实现简单、缓存友好仍是最实用的折中。

## 5 公式解析：为什么 EXTRACT 是 $O(d \log_d n)$

最大堆的 EXTRACT-MIN 需要：把末尾搬到根（$O(1)$），然后 MAX-HEAPIFY 下沉。下沉每一步要在 $d$ 个孩子中找最小者，需要 $d$ 次比较；而树高是 $\log_d n$（每层孩子数 $d$，节点总数 $n$）。故

$$T_{\text{extract}} = d \cdot \log_d n = O(d \log_d n)$$

做三步拆解：

- **第一步，看高度 $\log_d n$**：d 叉完全二叉树中，第 $k$ 层有 $d^k$ 个节点，从根到叶的层数满足 $d^h \approx n$，即 $h = \log_d n$。
- **第二步，看每步代价 $d$**：每一步要在 $d$ 个孩子中比较出最值——若用「打擂台」，$d$ 个元素需要 $d-1$ 次比较，视为 $O(d)$。
- **第三步，相乘**：路径长 $\times$ 每步代价 $= d\log_d n$。当 $d=2$ 时退化为 $2\log_2 n = O(\log n)$，与二叉堆一致。

**要点**：d 叉堆展示了复杂度分析里的**参数化思维**——把「孩子数 $d$」当作显式参数，量级随 $d$ 变化，这为针对具体应用的调优提供了理论依据。

## 6 小结

- **优先队列** = 动态集合 + 「取极值」操作；堆是它的经典实现，四个操作都在 $O(\log n)$。
- EXTRACT-MAX：搬末尾到根 + 下沉；INCREASE-KEY：沿父链**上浮**，前提是关键字只增不减。
- INSERT 用「追加 + 极小哨兵 + INCREASE-KEY」实现。
- **最小堆**是最大堆的符号翻转，DECREASE-KEY 是图算法最重要的原语。
- **d 叉堆**：$d$ 越大上浮越快（$O(\log_d n)$）但提取越慢（$O(d\log_d n)$），常数可调。

在下一课，我们转向排序的另一座高峰——**快速排序**。先看它的核心原语 PARTITION：如何在一次线性扫描里把数组分成「小、主元、大」三段，以及它为什么正确。
