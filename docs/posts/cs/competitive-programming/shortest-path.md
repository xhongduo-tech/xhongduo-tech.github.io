---
title: 最短路：Dijkstra/Bellman-Ford/Floyd
date: 2026-08-07
---

# 最短路：Dijkstra/Bellman-Ford/Floyd

<div class="epigraph">
<p>最短路的本质不是寻找，而是放松——每一条边都在说：走我这条路，能不能再近一点？</p>
<footer>—— 松弛操作的哲学</footer>
</div>

<div class="article-byline">
<p>第三级 · 算法竞赛与编程实践 ｜ 刘汝佳《算法竞赛入门经典》第11章 ｜ 2026-08-07</p>
</div>

## 为什么从最短路开始

导航软件规划路线、网络数据包选路、社交网络算「几度人脉」，背后都是同一个问题：**从点 A 到点 B 的最短路径**。竞赛里最短路是图论第一大考点，而它的解法是一个完整的谱系——**Dijkstra** 处理非负权单源最短路，**Bellman-Ford** 能对付负权边，**Floyd** 一次算出所有点对。三个算法共享同一个核心操作——**松弛（relaxation）**——却用三种完全不同的策略调度它。这一章，我们把这套谱系一次讲透，并学会判断「这道题该用哪个」。

## 1 松弛操作：最短路的最小公分母

**核心概念：松弛（relaxation）。** 设 `dist[u]` 是当前已知的源点到 `u` 的最短距离。对边 `u → v`（权 `w`），若「经过 u 再到 v」能更近，就更新：

$$
\text{dist}[v] = \min(\text{dist}[v],\ \text{dist}[u] + w)
$$

**辨析｜易错点：** 松弛是**方向性**的——只能从已知更优的前驱出发更新，不能反向。所有最短路算法（除 BFS 这种无权特例）都是「反复松弛直到没有边能再更新」。这个「没有能再更新的边」的时刻，就是答案收敛的时刻。<span class="marginnote">把 `dist` 想成「目前已知的最佳方案」，松弛就是「发现一条更短的绕路就改主意」。算法的区别只在于<strong>以什么顺序、松弛多少次</strong>：Dijkstra 用贪心顺序松一次，Bellman-Ford 松 n-1 轮，Floyd 用 DP 枚举中间点。</span>

## 2 Dijkstra：贪心 + 优先队列

**Dijkstra 算法** 处理**非负权**的单源最短路。思想：每次从「未确定最短路的点」里挑 `dist` 最小的那个，把它标记为确定，然后松弛它的所有出边。

```cpp
vector<long long> dist(n, INF);
priority_queue<pair<long long,int>, vector<...>, greater<...>> pq;
dist[s] = 0; pq.push({0, s});
while (!pq.empty()) {
    auto [d, u] = pq.top(); pq.pop();
    if (d != dist[u]) continue;          // 惰性删除：旧条目跳过
    for (auto &[v, w] : g[u])
        if (dist[u] + w < dist[v]) {
            dist[v] = dist[u] + w;
            pq.push({dist[v], v});
        }
}
```

**重点：为什么贪心成立。** 因为边权非负，已确定的点不可能再被「更短的绕路」改善——`dist` 最小的未确定点，其值就是最终答案。这就是 Dijkstra 的「确定一个、丢一个」逻辑。<span class="marginnote">复杂度 $O(m\log n)$。实现细节三件套：`dist` 初值 `INF`、优先队列按 `(距离, 节点)` 存储、弹出时用 `d != dist[u]` 跳过旧条目。这三点漏任何一点，Dijkstra 都会出错或变慢。它与 Prim 的代码高度相似，差别只在「取最小的是到源点的距离」还是「到集合的边权」。</span>

**辨析｜易错点：负权会让 Dijkstra 失效。** 存在负权边时，已确定点可能被「绕负权路」改善——Dijkstra 的贪心前提崩塌。判断标准：题目没保证非负权，就用 Bellman-Ford 或 SPFA。

## 3 Bellman-Ford：轮次松弛，扛住负权

**Bellman-Ford 算法** 不搞贪心，而是**整轮整轮地松弛**：每一轮松弛所有边，重复 $n-1$ 轮。为什么 $n-1$ 轮就够？因为**一条最短路径最多经过 $n-1$ 条边**——第 $k$ 轮结束后，所有「最多 $k$ 条边的最短路」都已正确。

```cpp
vector<long long> dist(n, INF);
dist[s] = 0;
for (int k = 0; k < n - 1; k++) {
    bool changed = false;
    for (auto &[u, v, w] : edges)
        if (dist[u] != INF && dist[u] + w < dist[v]) {
            dist[v] = dist[u] + w;
            changed = true;
        }
    if (!changed) break;               // 提前收敛
}
```

**核心概念：负环检测。** 若第 $n$ 轮还能松弛，说明存在**负环**——一个总权为负的环，可以让最短路无限缩短。负环存在时最短路无定义，Bellman-Ford 是少数能检测负环的算法。<span class="marginnote">复杂度 $O(nm)$，比 Dijkstra 慢，但换来了「负权 + 负环检测」的能力。竞赛里若图有负权且无负环，也常用 <strong>SPFA</strong>（队列优化的 Bellman-Ford）——平均快，但最坏仍 $O(nm)$，且被卡是著名话题，谨慎使用。</span>

## 4 Floyd：所有点对最短路的 DP

**Floyd-Warshall 算法** 一次算出**任意两点间**的最短路，用动态规划枚举中间点 $k$：

$$
dp[i][j] = \min(dp[i][j],\ dp[i][k] + dp[k][j]), \qquad k = 0, 1, \ldots, n-1
$$

```cpp
vector<vector<long long>> d(n, vector<long long>(n, INF));
for (int i = 0; i < n; i++) d[i][i] = 0;
// 读边：d[u][v] = min(d[u][v], w);
for (int k = 0; k < n; k++)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
```

- **第一步，定义状态**：`d[i][j]` 是「只允许经过前 k 个点中转」的 `i → j` 最短路。
- **第二步，写转移**：要么不经过 `k`（沿用旧值），要么经过 `k`（`d[i][k] + d[k][j]`），取小者。
- **第三步，循环顺序**：`k` 必须在最外层——这是「逐个放行中间点」的 DP 顺序，放错会漏解。

**重点：$O(n^3)$ 的代价换全面。** Floyd 代码极短、实现极简，适合 $n \le 500$ 的全源最短路；稠密图它甚至比跑 $n$ 次 Dijkstra 还方便。<span class="marginnote">Floyd 还能检测负环：结束后若 `d[i][i] < 0`，说明存在经过 i 的负环。另外 Floyd 可以改造来求「传递闭包」（布尔可达性）——把 `min` 换成 `或`，把 `+` 换成 `与`，同样三重循环搞定。</span>

## 5 三兄弟对比与选型

| 算法 | 单源/全源 | 负权 | 负环 | 复杂度 | 适用 |
| --- | --- | --- | --- | --- | --- |
| Dijkstra | 单源 | 否 | — | $O(m\log n)$ | 非负权，最常用 |
| Bellman-Ford | 单源 | 是 | 能检测 | $O(nm)$ | 负权、负环检测 |
| Floyd | 全源 | 是 | 能检测 | $O(n^3)$ | 所有点对，$n \le 500$ |

**辨析｜易错点：选型口诀。** 单源 + 非负 → Dijkstra；单源 + 负权 → Bellman-Ford（或 SPFA）；全源 → Floyd。还有一个常被忽略的点——**无权图用 BFS**（$O(n+m)$），它是 Dijkstra 的退化特例，比 Dijkstra 更快。别一上来就上 Dijkstra。

## 6 公式解析：松弛的极限与最短路的不等式

为什么「松到不能再松」就是最短路？所有最短路径满足**三角不等式**：

$$
\text{dist}[v] \le \text{dist}[u] + w(u, v) \quad \forall\ \text{边 } (u, v)
$$

- **第一步，读不等式**：源点经任意路径到 `u` 再走一条边到 `v`，不可能比直接的最短路更短。
- **第二步，联系算法**：Bellman-Ford 停止时，所有边都满足此不等式——任何再松弛都不能改善，`dist` 已是最优。
- **第三步，反向应用**：这个不等式也是「差分约束系统」的核心——把约束 `x_v - x_u \le w` 建成图，最短路即为可行解。最短路与不等式的联系，是图论题里最隐蔽的高级考法。

**重点：最短路算法是「不等式求解器」。** 一旦能把问题写成 `x_v \le x_u + w` 的约束组，就可以建图跑最短路/最长路求解。这是最短路「跳出导航」的进阶用法，也是竞赛里区分高手的一道门槛。

## 7 小结

- 松弛操作 `dist[v] = min(dist[v], dist[u]+w)` 是全部最短路算法的公共内核。
- Dijkstra：非负权 + 贪心 + 优先队列，$O(m\log n)$，最常用。
- Bellman-Ford：$n-1$ 轮全边松弛，扛负权，能检测负环，$O(nm)$。
- Floyd：三重循环枚举中间点，全源最短路，$O(n^3)$