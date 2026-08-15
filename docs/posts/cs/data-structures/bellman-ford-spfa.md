---
title: Bellman-Ford 与 SPFA
date: 2026-08-07
---

# Bellman-Ford 与 SPFA

<div class="epigraph">
<p>Dijkstra 怕负权，Bellman-Ford 不怕——它用「松弛 n-1 轮」换来了鲁棒。</p>
<footer>—— 最短路格言</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据结构 ｜ 严蔚敏《数据结构》 专题篇·图算法进阶 ｜ 2026-08-07</p>
</div>

## 为什么负权边让 Dijkstra 失效

§7.6 的 Dijkstra 要求边权非负——负权边会让「已确定最短路的点」被后续的负边推翻。**Bellman-Ford 算法**是「单源最短路」的全面版本：**支持负权边**，还能**检测负权回路**。它的原理朴素而深刻：**最短路经过的边数 ≤ n-1**（无环），所以「对每条边做一次松弛」重复 $n-1$ 轮，必得正确最短路。代价是 $O(nm)$——比 Dijkstra 慢，但适用范围广。**SPFA** 是它的队列优化：只把「距离被更新的点」入队，平均快、最坏仍 $O(nm)$。

## 1 Bellman-Ford 的思想：松弛 n-1 轮

初始化 $dist[s]=0$、其余 $dist=\infty$。然后**对全部边做松弛，重复 $n-1$ 轮**：

$$
dist[v] = \min(dist[v], \ dist[u] + w(u,v)) \quad \text{对所有边}
$$

**为什么是 $n-1$ 轮？** 因为从源点到任意点的最短路径，最多经过 $n-1$ 条边（无环最短路）——每轮松弛「至少确定一条边的正确距离」，$n-1$ 轮后全部确定。

```c
void BellmanFord(Edge e[], int n, int m, int s, int dist[]) {
    int i, j;
    for (i = 1; i <= n; i++) dist[i] = INF;
    dist[s] = 0;
    for (i = 1; i <= n - 1; i++)               /* 松弛 n-1 轮 */
        for (j = 1; j <= m; j++)
            if (dist[e[j].u] != INF && dist[e[j].v] > dist[e[j].u] + e[j].w)
                dist[e[j].v] = dist[e[j].u] + e[j].w;   /* 松弛 */
    for (j = 1; j <= m; j++)                   /* 第 n 轮还能松弛 → 有负环 */
        if (dist[e[j].u] != INF && dist[e[j].v] > dist[e[j].u] + e[j].w)
            printf("存在负环\n");
}
```

**重点：Bellman-Ford 是「无差别地松弛所有边 n-1 轮」**——它不用像 Dijkstra 那样挑「最小点」，所以不怕负权；而「第 n 轮还能松弛」就说明存在负环（距离可无限减小）。<span class="marginnote">「<strong>最短路无环 → 至多 n-1 条边 → n-1 轮松弛</strong>」是 Bellman-Ford 的完整逻辑链：<strong>负权只能让「路径变短」，但不会让「最优路径变长」——最优路径依然无环</strong>。<strong>「n-1 轮」不是拍脑袋，是最短路的结构性质</strong>。</span>

## 2 公式解析：正确性与复杂度

设 $n$ 个顶点、$m$ 条边：

$$
T_{\text{Bellman-Ford}} = O(nm)
$$

- **第一步，读「$n-1$ 轮 × $m$ 条边」**：每轮松弛全部 $m$ 条边，$n-1$ 轮——$O(nm)$。
- **第二步，读「正确性」**：第 $i$ 轮后，所有「至多 $i$ 条边的最短路」都已确定——归纳成立；$n-1$ 轮后全部确定。
- **第三步，读「与 Dijkstra 对比」**：Dijkstra $O((n+e)\log n)$、要求非负；Bellman-Ford $O(nm)$、允许负权 + 检测负环——**用时间换鲁棒**。<span class="marginnote">「<strong>Dijkstra 快而挑食，Bellman-Ford 慢而全面</strong>」：<strong>非负权 → Dijkstra；有负权 → Bellman-Ford；要检测负环 → 只有 Bellman-Ford</strong>。<strong>「适用范围」与「复杂度」总是此消彼长</strong>——这是算法选型的第一定律。</span>

## 3 SPFA：队列优化

**SPFA（Shortest Path Faster Algorithm）** 是 Bellman-Ford 的队列版：

不用每轮扫所有边，只把「**被松弛成功**（dist 变小的点）」入队；
出队一个点，松弛它的所有出边；若某邻居被松弛，入队；
一个点可能多次入队（每次 dist 变小）——**用 `inQueue` 标记避免重复入队**；
**入队次数 ≥ n 说明有负环**。

```c
void SPFA(int s, int dist[]) {
    queue<int> q;
    bool inQueue[MAXN] = {false};
    int cnt[MAXN] = {0}, u;
    for (int i = 1; i <= n; i++) dist[i] = INF;
    dist[s] = 0; q.push(s); inQueue[s] = true; cnt[s]++;
    while (!q.empty()) {
        u = q.front(); q.pop(); inQueue[u] = false;
        for (每条出边 (u, v, w)) {
            if (dist[v] > dist[u] + w) {       /* 松弛成功 */
                dist[v] = dist[u] + w;
                if (!inQueue[v]) {
                    q.push(v); inQueue[v] = true;
                    if (++cnt[v] >= n) { printf("存在负环\n"); return; }
                }
            }
        }
    }
}
```

**重点：SPFA 只处理「被松弛的点」——平均快得多，但最坏仍是 $O(nm)$**（每个点可入队多次）。它是「用队列做『有选择』的松弛」——把 Bellman-Ford 的「扫全部」变成「扫有变化的」。<span class="marginnote">「<strong>SPFA = 只松弛有希望的点</strong>」：<strong>dist 没变的点，松弛它的出边必然没效果，跳过</strong>。<strong>「谁变了就处理谁」是增量思想</strong>——平均 $O(m)$，最坏 $O(nm)$。<strong>SPFA 的平均快、最坏差</strong>让它「竞赛可用、对抗场景慎用」。</span>

## 4 辨析｜易错点：Bellman-Ford vs SPFA vs Dijkstra

| 维度 | Dijkstra | Bellman-Ford | SPFA |
| --- | --- | --- | --- |
| 负权边 | 不支持 | 支持 | 支持 |
| 负环检测 | 不支持 | 第 n 轮松弛 | 入队 ≥ n 次 |
| 平均复杂度 | $O((n+e)\log n)$ | $O(nm)$ 固定 | 平均近 $O(m)$、最坏 $O(nm)$ |
| 结构 | 贪心 + 堆 | 无差别松弛 | 队列 + 松弛 |

**重点：三者的选择口诀——非负权且图稀疏用堆 Dijkstra；有负权用 Bellman-Ford/SPFA；要检测负环用 Bellman-Ford 或 SPFA 的计数。**<span class="marginnote">「<strong>看图说话选最短路</strong>」：<strong>非负 → Dijkstra；负权 → Bellman-Ford/SPFA；负环 → Bellman-Ford 第 n 轮</strong>。<strong>工程（导航、路由）几乎都是非负权 → Dijkstra 一家独大</strong>；负权多出现在「数学建模」与「约束系统」里（下节的差分约束）。</span>

## 5 负环检测的意义

**负权回路（negative cycle）**：总权为负的环——沿它绕圈，路径长无限减小，最短路无定义。Bellman-Ford 的「第 n 轮还能松弛」与 SPFA 的「入队 ≥ n 次」都是检测信号。**凡可能出现负环的问题（差分约束、汇率套利、利润模型），都必须先测负环。**<span class="marginnote">「<strong>负环 = 最短路问题的病态输入</strong>」：<strong>一旦存在负环，任何「最短距离」都没有意义（可以无限小）</strong>。<strong>所以「最短路」问题标配「负环检测」</strong>——差分约束系统（下节）正是靠它判断「不等式组有无解」。</span>

**辨析｜易错点：SPFA 的「入队次数 ≥ n」与 Bellman-Ford 的「第 n 轮还能松弛」等价。** 一个点入队超过 $n-1$ 次，说明它在松弛中反复被更新——这只有在存在负环时才可能（最短路至多 $n-1$ 条边）。两种检测是同一原理的两种表述。

**一句话**：负环检测 = 「松弛轮数超过最短路边数上限」的越界信号。

## 6 小结

- Bellman-Ford：对全部边松弛 $n-1$ 轮，支持负权、检测负环。
- 正确性根基：最短路至多 $n-1$ 条边；第 n 轮还能松弛 = 有负环。
- 复杂度 $O(nm)$：用时间换鲁棒。
- SPFA：只入队「被松弛的点」，平均近 $O(m)$、最坏 $O(nm)$。
- 选型：非负 Dijkstra、负权 Bellman-Ford/SPFA、负环检测 Bellman-Ford。
- 负环 = 最短路病态输入——差分约束等模型必须先测。

在下一节，我们看 Bellman-Ford 的经典建模应用——**差分约束系统**。
