---
title: 强连通分量（Strongly Connected Components）：Kosaraju 算法与分量图
date: 2026-08-07
---

# 强连通分量（Strongly Connected Components）：Kosaraju 算法与分量图

<div class="epigraph">
<p>在有向图里，「互相到达」才是真正的一家人——把图压缩成分量图，复杂的环结构顿时清晰。</p>
<footer>—— 托马斯 · 科尔曼 等（Thomas H. Cormen）《算法导论》</footer>
</div>

<div class="article-byline">
<p>第三级 · 算法设计与分析 ｜ 《算法导论》（CLRS）第 22.5 节 ｜ 2026-08-07</p>
</div>

## 为什么从强连通分量开始

有向图里，若 $u$ 能到达 $v$ 且 $v$ 能到达 $u$，则 $u, v$ **互相可达**——这种「互达」关系是等价关系，把顶点划分成**强连通分量（SCC）**。<span class="marginnote">SCC 是理解有向图结构的钥匙：社交网络里的紧密社群、代码调用图里的模块、Web 图里的网站群落，都表现为 SCC。而且，把每个 SCC 缩成一个点，原图变成<strong>无环的「分量图」</strong>（DAG）——环全部被消化在分量内部，剩下的结构是清晰的偏序。</span>

这一课讲 Kosaraju 算法：**为什么在转置图上按「原图完成时间降序」再跑一次 DFS，就恰好分出 SCC**。

## 1 定义与分量图

**强连通分量（strongly connected component）**：极大顶点集合 $C$，其中任意两顶点 $u, v$ 互相可达（$u \rightsquigarrow v$ 且 $v \rightsquigarrow u$）。

**分量图（component graph）**：把每个 SCC 缩成超点，超点之间按原图的边连——**分量图必是 DAG**。<span class="marginnote">为什么分量图无环？若分量之间成环，环上的分量彼此可达、会合并成更大的 SCC，与「极大」矛盾。所以「SCC 的缩图是无环」是 SCC 定义的自然后果——这一事实让 SCC 问题与拓扑排序天然连接。</span>

## 2 Kosaraju 算法

```
STRONGLY-CONNECTED-COMPONENTS(G)
  1. call DFS(G) to compute finish times f[u]
  2. compute G^T (转置图：所有边反向)
  3. call DFS(G^T), 按 f[u] 降序选择源点
  4. 第二次 DFS 的每棵树的顶点构成一个 SCC
```

**三步**：原图 DFS 求完成时间 → 转置图 → 按完成时间降序做 DFS。<span class="marginnote">直觉：原图「完成最晚」的顶点位于「汇」侧的 SCC；转置图把方向翻转后，从该 SCC 出发的 DFS 只会在该 SCC 内部游走（因为跨分量的边都指向别处，反向后在转置图里「指向别处」的边变成「来自别处」）——于是每棵树恰好框出一个 SCC。</span>

**复杂度**：两次 DFS 各 $O(V+E)$，转置 $O(V+E)$，总计 $O(V+E)$。

## 3 为什么正确：三个关键引理

**引理 1（连通性对称）**：$u$ 与 $v$ 在同一个 SCC ⟺ $u$ 与 $v$ 在原图与转置图中都互相可达。因为转置图只是方向反转，互达关系不变。<span class="marginnote">这解释为什么要在 $G^T$ 上做——SCC 是「双向可达」的结构，方向反转不影响它，但改变了 DFS 的「出发次序」敏感性。</span>

**引理 2（分量之间的边方向）**：设 $C_1, C_2$ 是两个 SCC，若原图有边 $C_1 \to C_2$，则分量图里 $C_1$ 的完成时间「更大」（$f[C_1] > f[C_2]$）——原图 DFS 中，$C_1$ 中顶点的最大完成时间大于 $C_2$ 中顶点的最大完成时间。<span class="marginnote">这个「完成时间沿边递减」的方向性，与拓扑排序的引理同源：分量图是 DAG，边总是从「晚完成」指向「早完成」。它是第二次 DFS 选择起点的依据——「最晚完成」的 SCC 在转置图中没有入边（是源），从它开始的 DFS 不会跑到别的分量。</span>

**引理 3（第二次 DFS 框出分量）**：在 $G^T$ 上按 $f$ 降序出发，每次 DFS 恰好访问一个 SCC。因为起点所在的 SCC 在 $G^T$ 中没有入边（引理 2 + 降序），DFS 无法离开该分量；而分量内部互达保证能访问到全部成员。

三个引理合起来，Kosaraju 正确。<span class="marginnote">引理 2 与引理 3 是本课最难啃的部分，但值得读透：它们展示了「一次 DFS 的时间戳」如何编码「分量的拓扑序」，第二次 DFS 又如何在转置图上把每个分量「圈」出来。Tarjan 的单次 DFS 算法（用 low-link）是另一个选择，但 Kosaraju 的结构更清晰。</span>

## 4 公式解析：为什么分量图是 DAG（形式论证）

设 $C_1, C_2$ 是不同 SCC 且 $C_1$ 能到达 $C_2$。若 $C_2$ 也能到达 $C_1$，则两个分量合并成一个更大的互达集，与极大性矛盾。故分量之间的可达关系是**反对称**的：

$$C_1 \rightsquigarrow C_2 \;\wedge\; C_2 \rightsquigarrow C_1 \;\Longrightarrow\; C_1 = C_2$$

- **第一步，假设反向可达**：$C_1$ 可达 $C_2$ 且 $C_2$ 可达 $C_1$。
- **第二步，取顶点**：任取 $u \in C_1$，$v \in C_2$——$u \rightsquigarrow v$（经 $C_1\to C_2$）且 $v \rightsquigarrow u$（经反向），故 $u, v$ 互达。
- **第三步，矛盾**：$C_1, C_2$ 的任意顶点互相可达，它们其实同属一个 SCC——与「不同分量」矛盾。

**结论**：分量图无环（反对称 + 无自环即 DAG）。**SCC 的缩图是无环 DAG**，这是理解「有向图结构」的核心事实。<span class="marginnote">缩图成 DAG 的实际价值：可以在 SCC 级别上做拓扑排序、动态规划、最短路径（对 DAG 用拓扑序线性求解）。工程里「把复杂有向图压缩成 DAG」是标准预处理——SCC 消除环，剩下的无环结构好处理得多。</span>

## 5 小结

- **SCC**：极大互达顶点集；缩成超点后，**分量图是 DAG**。
- **Kosaraju 算法**：原图 DFS 求完成时间 → 转置图 → 按完成时间降序再做 DFS，每棵树一个 SCC。
- 正确性靠三个引理：互达不变性、分量间完成时间沿边递减、转置图降序起点框出分量。
- 复杂度 $O(V+E)$；Tarjan 的单遍算法是另一实现。
- 应用：社交网络社群、代码模块分析、Web 图压缩、SCC 级 DAG 上的动态规划。

在下一课，我们开始最短路专题——**单源最短路径**：从 Bellman-Ford（允许负权、检测负环）到 DAG 上的 $O(V+E)$ 线性算法。
