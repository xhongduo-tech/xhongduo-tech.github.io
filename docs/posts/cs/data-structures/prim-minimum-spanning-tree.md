---
title: 最小生成树（Prim 算法）
date: 2026-08-07
---

# 最小生成树（Prim 算法）

<div class="epigraph">
<p>从一点出发，永远选择离自己最近的新大陆。</p>
<footer>—— 贪心算法格言</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据结构 ｜ 严蔚敏《数据结构》 §7.4 ｜ 2026-08-07</p>
</div>

## 为什么 Prim 先登场

上一节连通性问题留下一个待办：**最小生成树（MST）**。Prim 算法是求解 MST 的第一种思路，也是最贴近 Dijkstra 的算法——两者共用「贪心选点 + 松弛」的骨架，只差「选点标准」与「松弛对象」。Prim 从任意一个顶点出发，维护「已连集合 $U$」与「未连集合 $V-U$」，每步选**连接两边的最短边**，把新顶点并入 $U$，直到 $n$ 个顶点全部连上。朴素版 $O(n^2)$，堆优化 $O((n+e)\log n)$——**稠密图选朴素 Prim**。

## 1 Prim 的思想

设图 $G = (V, E)$ 连通带权。算法维护集合 $U$（已在生成树中的顶点），初始任选一个顶点加入。

1. 对每个未入 $U$ 的顶点 $v$，记录 $v$ = $v$ 与 $U$ 内顶点之间的**最小边权**（不在 $U$ 的邻接点为 $\infty$）；
2. **选点**：从未入 $U$ 的顶点中，选 $v$ 最小的 $v$，把它和那条边加入生成树；
3. **更新**：用新入 $U$ 的 $v$ 去刷新其他未入 $U$ 顶点的 $U$；
4. 重复直到 $U = V$。

```c
void Prim(int n, int adj[][N], int start) {
    int lowcost[N];          /* lowcost[v] = v 到集合 U 的最小边权 */
    int mst[N];              /* mst[v] = 这条最小边在 U 内的那个顶点 */
    for (int v = 1; v <= n; v++) {
        lowcost[v] = adj[start][v];
        mst[v] = start;
    }
    lowcost[start] = 0;      /* 兼作「已入 U」标记 */
    for (int i = 1; i < n; i++) {
        int min = INF, u = -1;
        for (int v = 1; v <= n; v++)          /* 选点：lowcost 最小 */
            if (lowcost[v] && lowcost[v] < min) { min = lowcost[v]; u = v; }
        lowcost[u] = 0;                       /* 并入 U，并记录边 (mst[u], u) */
        for (int v = 1; v <= n; v++)          /* 刷新：用 u 更新未入 U 顶点的 lowcost */
            if (lowcost[v] && adj[u][v] < lowcost[v]) {
                lowcost[v] = adj[u][v];
                mst[v] = u;
            }
    }
}
```

**重点：`lowcost[v] = 0` 兼作「已入 U」的标记**——把「并入」与「不可再选」合一，这是 Prim 的经典实现细节。<span class="marginnote">对照 Dijkstra 的代码：Dijkstra 选「$dist$ 最小」、松弛「$dist[u]+w$」；Prim 选「`lowcost` 最小」、刷新「$w$」。<strong>同一个循环骨架，一个算「到源点的路」，一个算「到集合的边」</strong>——两套算法的血缘一目了然。</span>

## 2 公式解析：Prim 的复杂度

设图有 $n$ 个顶点、$e$ 条边：

$$
T_{\text{Prim 朴素}} = O(n^2), \qquad T_{\text{Prim 堆优化}} = O((n+e)\log n)
$$

- **第一步，读朴素版**：外层循环 $n-1$ 次；每次「选最小」扫 $O(n)$、「刷新」扫 $O(n)$——总 $O(n^2)$。
- **第二步，读堆优化版**：用最小堆存「未入 $U$ 顶点的 `lowcost`」，选最小 $O(\log n)$；刷新时对每条边可能做一次堆更新 $O(\log n)$——总 $O((n+e)\log n)$。
- **第三步，读选型**：**稠密图 $e \sim n^2$ 时，朴素 $O(n^2)$ 优于堆优化 $O(n^2\log n)$；稀疏图才用堆优化。** 这与 Dijkstra 的选型镜像对称。<span class="marginnote">「稠密用朴素、稀疏用堆」是 Prim 与 Dijkstra 共同的分界。<strong>堆优化的 $\log n$ 因子在稠密图是负担不是福利</strong>——因为 $n^2$ 条边让「每边一次堆操作」反而更贵。理解这一点，就理解了许多「优化」只在特定数据形态下才是优化。</span>

## 3 正确性：切割性质

Prim 的贪心为什么对？靠**切割性质（cut property）**：

**对任意顶点集合 $S \subset V$，连接 $S$ 与 $V-S$ 的所有边中，权最小的那条边必然出现在某棵最小生成树里。**

Prim 每一步都在沿「$U$ / $V-U$」这个切割选最轻边，由切割性质，这些边都能进某棵 MST——贪心不会错。<span class="marginnote">切割性质的证明用「<strong>交换论证</strong>」：若最小生成树不含这条最轻边 $e$，把它加进去必产生一个含 $e$ 的环，环上有一条跨切割的边 $f$ 比 $e$ 重，换掉 $f$ 总权不增。<strong>「换不亏」是贪心正确性证明的通用句式</strong>，赫夫曼树、Dijkstra、Kruskal 的证明都是它的变体。</span>

## 4 辨析｜易错点：Prim vs Dijkstra vs Kruskal

三个「选边/选点」算法最容易混淆：

| 算法 | 目标 | 每次选 | 适用 | 复杂度 |
| --- | --- | --- | --- | --- |
| Dijkstra | 单源最短路 | 到源点 $dist$ 最小 | 非负权单源 | $O(n^2)$ / $O((n+e)\log n)$ |
| Prim | 最小生成树 | 到集合边权最小 | 稠密图 | $O(n^2)$ / 堆优化 |
| Kruskal | 最小生成树 | 全局最轻且不成环的边 | 稀疏图 | $O(e\log e)$ |

三条区分点：**Dijkstra 算「路径和」、Prim/Kruskal 算「树总权」；Prim 是「点扩展」、Kruskal 是「边扫描」；稠密图 Prim、稀疏图 Kruskal。**<span class="marginnote">最容易踩的坑是「Prim 的 lowcost 与 Dijkstra 的 dist 混用」：Prim 松弛用<strong>裸边权</strong> $w(u,v)$，Dijkstra 用<strong>累积距离</strong> $dist[u]+w$。代码长得像，语义完全不同——写前先问「我在累加路径，还是在选边」。</span>

## 5 Prim 的应用

**稠密连通图的 MST**：如城市间直达线路（边多、权密的网络铺设）；
**平面欧几里得 MST**：多点连接成网（如传感器组网）的最小总长；
**与 Dijkstra 的组合使用**：先 MST 得连通骨架，再在其上做路由。

**重点：Prim 是「点扩展式」贪心的代表作**——每一步都在「已连通区域」的边缘挑最便宜的扩展。理解 Prim，就理解了「从一个种子区域生长」这类算法（BFS 生长、区域生长图像分割）的通用模式。<span class="marginnote">「从一个区域向外长」是很多算法的形态：<strong>BFS 生长区域、Prim 生长连通骨架、图像处理的区域生长分割</strong>。它们的共同点：维护「已处理集合」，每步从边界挑一个最优扩展——把「全局最优」问题拆成「逐点局部贪心」。</span>

**一个具体的算例。** 顶点 $A, B, C, D$，边权 $AB=1$、$AC=4$、$AD=3$、$BC=2$、$CD=5$，从 $A$ 出发：

- $lowcost = [B:1, C:4, D:3]$，选 $B$（权 1），边 $AB$ 入树；
- 刷新：$BC=2 < 4$，$lowcost=[C:2, D:3]$，选 $C$（权 2），边 $BC$ 入树；
- 刷新：$CD=5 > 3$ 不变，选 $D$（权 3），边 $AD$ 入树。

MST 边集 $\{AB(1), BC(2), AD(3)\}$，总权 6。注意第 3 步「$D$ 与 $A$ 的直连 3 胜过了与 $C$ 的 5」——**Prim 永远选「到集合」的最近边，不是「到上一个点」的最近边**。<span class="marginnote">「到集合」而非「到上一点」是 Prim 与最短路式贪心最易混淆处：<strong>lowcost[D] 记录的是 D 到整个已连集合的最小边，会随新点加入被刷新</strong>——<strong>它可能是 $A$—$D$ 也可能是 $C$—$D$，取最小</strong>。这个「集合视角」正是切割性质的体现。</span>

**术语速查表**

| 术语 | 含义 |
| --- | --- |
| 已连集合 $U$ | 已在生成树中的顶点 |
| lowcost[v] | v 到 $U$ 的最小边权 |
| 切割性质 | 跨任何切割的最轻边必在某棵 MST 中 |
| 点扩展 | 每步沿已连区域的边界扩展 |
| 稠密图选朴素 | $e \sim n^2$ 时 $O(n^2)$ 优于堆优化 |

## 6 小结

- Prim：从一点出发，维护「已连/未连」集合，每步选跨切割最轻边并入。
- `lowcost[v]` 存「到集合的最小边权」，`mst[v]` 记录来源边；`lowcost[v] = 0` 兼作入 $U$ 标记。
- 复杂度：朴素 $O(n^2)$，堆优化 $O((n+e)\log n)$；稠密图选朴素。
- 正确性靠**切割性质**：跨任何切割的最轻边必在某棵 MST 中。
- 与 Dijkstra 同骨架、异语义；与 Kruskal 互补（稀疏图）。
- 应用：稠密网络铺设、欧几里得 MST、区域生长式贪心。

在下一节，我们用「边排序 + 并查集」实现 MST 的第二种——**最小生成树（Kruskal 算法）**。
