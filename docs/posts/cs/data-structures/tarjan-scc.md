---
title: Tarjan 算法求强连通分量
date: 2026-08-07
---

# Tarjan 算法求强连通分量

<div class="epigraph">
<p>一趟 DFS，用时间戳与回溯值，把「互相可达」的环一网打尽。</p>
<footer>—— Tarjan 格言</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据结构 ｜ 严蔚敏《数据结构》 专题篇·图算法进阶 ｜ 2026-08-07</p>
</div>

## 为什么强连通分量值得专门算法

§7.1 定义了**强连通分量（SCC）**——有向图中「两两互相可达」的极大子图。朴素求法「对每对点跑可达性」是 $O(n^3)$ 级。**Tarjan 算法**在一趟 DFS 里完成全部 SCC 划分，$O(n+e)$：它用两个数组——**$O(n+e)$（发现时间戳）**与 **low（回溯值）**——识别「环」。原理一句话：**一个 SCC 是一棵 DFS 子树上的一群「互相通过回边/横叉边回到祖先」的结点**，low[u] == dfn[u] 的结点就是某个 SCC 的「根」。Tarjan 算法同时是割点、桥（下节）的同族算法。

## 1 两个关键数组：dfn 与 low

DFS 时维护：

- **dfn**：结点 $u$ 第一次被访问的**时间戳**（发现序）；
- **low**：从 $u$ 出发，**能通过 DFS 树边/回边到达的「最小 dfn」**——即 $u$ 的子树中所有结点能「够到」的最早祖先。

$$
low[u] = \min \begin{cases} dfn[u] \\ low[v] & v \text{ 是 } u \text{ 的树儿子} \\ dfn[v] & (u,v) \text{ 是回边，} v \text{ 在栈中} \end{cases}
$$

**重点：low 回答「$u$ 的子树能回溯到多早」——若 low == dfn，说明 $u$ 的子树「飞不出去」，$u$ 是某个 SCC 的根。**<span class="marginnote">「<strong>low 是能回溯到的最早祖先</strong>」是 Tarjan 的灵魂：<strong>一个 SCC 里的所有结点，都能沿环回到这个 SCC 的入口（dfn 最小者）</strong>。<strong>low == dfn = 「$u$ 及其子树自成一圈，飞不出去了」</strong>——这里就是 SCC 的边界。</span>

## 2 Tarjan 的主流程

维护一个**栈**（存当前 DFS 路径上「尚未归属 SCC」的结点）：

```cpp
int dfn[N], low[N], timer, sccCnt;
bool inStack[N]; stack<int> st;        // 栈：存尚未归属 SCC 的结点

void tarjan(int u) {
    dfn[u] = low[u] = ++timer;         // 发现时间戳
    st.push(u); inStack[u] = true;
    for (int v : adj[u]) {
        if (!dfn[v]) {                 // 树边：儿子回溯值上收
            tarjan(v);
            low[u] = min(low[u], low[v]);
        } else if (inStack[v]) {       // 回边：够到栈中祖先
            low[u] = min(low[u], dfn[v]);
        }
        // 横叉边（访问过但不在栈中）：忽略
    }
    if (low[u] == dfn[u]) {            // u 是某个 SCC 的根
        ++sccCnt;
        while (true) {                 // 弹出栈顶到 u 的全部结点
            int v = st.top(); st.pop(); inStack[v] = false;
            belong[v] = sccCnt;
            if (v == u) break;
        }
    }
}
```

**重点：三要素——树边更新 low（儿子回溯值上收）、回边更新 low（够到更早祖先）、low[u] == dfn[u] 时弹出栈顶到 u 的全部结点为一个 SCC。**<span class="marginnote">「<strong>树边上收 low、回边压低 low、等号出环</strong>」是 Tarjan 的三句口诀：<strong>儿子能到的祖先父亲也能到（树边），环能回到的祖先要压低 low（回边）</strong>，<strong>low 不再降时（等号）栈里的就是完整的一个环</strong>。</span>

## 3 公式解析：复杂度 O(n+e)

$$
T_{\text{Tarjan}} = O(n + e) \quad \text{（每个顶点 DFS 一次、每条边检查一次、每个顶点入出栈一次）}
$$

- **第一步，读「DFS 骨架」**：Tarjan 就是一次完整的 DFS，每个顶点访问一次。
- **第二步，读「每条边一次」**：每条边在遍历时检查一次（树边/回边分支）。
- **第三步，读「栈操作线性」**：每个顶点入栈一次、出栈一次——$O(n)$。<span class="marginnote">「<strong>Tarjan 把 $O(n^3)$ 的强连通问题压到 $O(n+e)$</strong>」：<strong>一趟 DFS + 两个数组 + 一个栈</strong>——<strong>「线性」是图算法的最高荣誉</strong>。<strong>对比朴素的「每对点跑可达性」，Tarjan 省掉了所有重复遍历</strong>。</span>

## 4 辨析｜易错点：横叉边与栈

**树边**（第一次访问）：low[u] = min(low[u], low[v])——儿子能回溯到多早，父亲也能；
**回边**（访问过且在栈中）：low[u] = min(low[u], dfn[v])——能回到栈中祖先；
**横叉边**（访问过但不在栈中）：**忽略**——它指向已归属其他 SCC 的结点，不影响当前 SCC。

**重点：回边与横叉边的判断必须用「是否在栈中」而不是「是否访问过」**——访问过但已出栈的结点属于别的 SCC，横叉边对当前 low 无贡献。<span class="marginnote">「<strong>回边看栈、横叉边忽略</strong>」：<strong>栈里的是「当前还没分完 SCC 的结点」，回边够到它们才构成环</strong>。<strong>已出栈的是「别的 SCC」，横叉边不能把它们拉进来</strong>——<strong>这一条判断错误，SCC 就会并错</strong>。</span>

## 5 强连通分量的应用

**缩点（condensation）**：把每个 SCC 缩成一个点，图变成 **DAG**——复杂图问题转 DAG 问题（拓扑排序、DP 都可上）；
**求解「环上条件」**：2-SAT、博弈论的强连通判定；
**依赖分析**：模块依赖的循环检测（有 SCC = 有环依赖）；
**社交网络**：互相成就的「核心圈子」。

**重点：SCC 的最大价值是「缩点成 DAG」**——把「带环的有向图」化简为「无环图」，然后一切 DAG 算法（拓扑、关键路径、DP）全部解锁。<span class="marginnote">「<strong>缩点 = 把环变成点</strong>」：<strong>SCC 内部互相可达，对「外部关系」来说整个 SCC 是一个整体</strong>。<strong>缩点后图无环（否则会并入同一 SCC）——环消了，DAG 来了</strong>。<strong>「先缩点、再 DAG 算法」是强连通问题的标准套路</strong>。</span>

**一个 dfn/low 的算例。** 有向图边 $1 \to 2$、$2 \to 3$、$3 \to 1$、$3 \to 4$、$4 \to 4$。

从 1 开始 DFS：$dfn[1]=1, dfn[2]=2, dfn[3]=3$，3 有回边 $3 \to 1$（1 在栈中），$low[3] = \min(3, 1) = 1$；3 还有树边到 4：$dfn[4]=4$，4 有自环 $4 \to 4$（回边），$low[4]=4$。回溯：$low[3] = \min(low[3], low[4]) = \min(1, 4) = 1$；$low[2] = \min(2, low[3]) = 1$；$low[1] = \min(1, low[2]) = 1$。

此时 $dfn[4]=4=low[4]$：弹出 4，SCC $\{4\}$；$dfn[3]=3 \ne low[3]=1$、$dfn[2]=2 \ne low[2]=1$，不弹；$dfn[1]=1=low[1]$：弹出栈顶 3、2、1，SCC $\{1,2,3\}$。<span class="marginnote">「$dfn=low$ 才弹栈」在例子里看得很清楚：<strong>4 自成环路，$low[4]=dfn[4]$ 直接成 SCC；1、2、3 由回边 $3\to1$ 连成环，最终由根 1 统一弹出</strong>。<strong>回边是「把 low 拉低」的元凶，也是环存在的证据</strong>。</span>

**术语速查表**

| 术语 | 含义 |
| --- | --- |
| 强连通分量 SCC | 两两互相可达的极大子图 |
| dfn | 发现时间戳 |
| low | 子树能回溯到的最小 dfn |
| 树边 / 回边 / 横叉边 | DFS 边分类 |
| 缩点 | 把 SCC 缩成点，图变 DAG |

**一句话**：Tarjan = 一趟 DFS + dfn/low + 栈——「low == dfn 才弹栈」，环就一网打尽。

## 6 小结

- SCC：有向图中两两互相可达的极大子图。
- Tarjan 用 dfn（时间戳）+ low（回溯值）+ 栈，一趟 DFS 求全部 SCC。
- low[u] == dfn[u] 的 u 是 SCC 根，弹出栈顶到 u 即一个 SCC。
- 树边上收 low、回边压低 low、横叉边忽略（看 in_stack）。
- 复杂度 $O(n+e)$——线性，朴素法 $O(n^3)$ 的终极优化。
- 核心应用：缩点成 DAG——带环图问题转无环图算法。

在下一节，我们看 Tarjan 家族的另一半——**割点与桥**。
