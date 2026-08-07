---
pageClass: plain-doc
---

# 数据结构

学完数据结构 = 写完严蔚敏《数据结构（C语言版）》全部章节对应的博文，并补齐常用算法专题。每篇博文对应教材的一节，写完一篇勾掉一条。

## 主题规划

<ProgressGrid cat="cs/data-structures" />


### 第一篇 绪论

- [x] [什么是数据结构](./what-is-data-structure)
- [x] [基本概念和术语](./basic-concepts-and-terminology)
- [x] [抽象数据类型的表示与实现](./adt-representation-and-implementation)
- [x] [算法和算法分析](./algorithms-and-analysis)
- [x] [时间复杂度与空间复杂度的计算方法](./time-and-space-complexity)

### 第二篇 线性表

- [x] [线性表的类型定义](./linear-list-type-definition)
- [x] [线性表的顺序表示和实现](./sequential-list)
- [x] [线性表的链式表示和实现（单链表）](./linked-list)
- [x] [循环链表与双向链表](./circular-and-doubly-linked-list)
- [x] [一元多项式的表示及相加](./polynomial-representation-addition)

### 第三篇 栈和队列

- [x] [栈的定义与顺序栈的实现](./stack-definition-sequential-stack)
- [x] [栈的链式表示](./linked-stack)
- [x] [栈与递归（汉诺塔）](./stack-recursion-hanoi)
- [x] [表达式求值](./expression-evaluation)
- [x] [队列的顺序表示与循环队列](./sequential-queue-circular-queue)
- [x] [链队列](./linked-queue)
- [x] [双端队列](./deque)

### 第四篇 串

- [x] [串类型的定义](./string-type-definition)
- [x] [串的表示和实现（定长顺序存储、堆分配存储、块链存储）](./string-storage-representation)
- [x] [串的模式匹配算法（朴素匹配）](./string-naive-pattern-matching)
- [x] [串操作应用举例（文本编辑）](./text-editing-string-application)

### 第五篇 数组和广义表

- [x] [数组的定义与顺序表示](./array-definition-sequential-representation)
- [x] [矩阵的压缩存储（特殊矩阵）](./special-matrix-compression)
- [x] [稀疏矩阵的三元组顺序表表示](./sparse-matrix-triple-representation)
- [x] [稀疏矩阵的十字链表表示](./sparse-matrix-cross-linked-list)
- [x] [广义表的定义](./generalized-list-definition)
- [x] [广义表的存储结构](./generalized-list-storage)
- [x] [m元多项式的表示与广义表的递归算法](./m-degree-polynomial-generalized-list)

### 第六篇 树和二叉树

- [x] [树的定义和基本术语](./tree-definition-terminology)
- [x] [二叉树的定义与性质](./binary-tree-definition-properties)
- [x] [二叉树的存储结构](./binary-tree-storage)
- [x] [遍历二叉树（先序、中序、后序、层序）](./binary-tree-traversal)
- [x] [线索二叉树](./threaded-binary-tree)
- [x] [树和森林（存储结构及与二叉树的转换）](./tree-forest-storage-conversion)
- [x] [树与森林的遍历](./tree-forest-traversal)
- [x] [赫夫曼树及其应用](./huffman-tree)
- [x] [赫夫曼编码](./huffman-coding)
- [x] [回溯法与树的遍历（八皇后问题）](./backtracking-eight-queens)
- [x] [树的计数](./tree-counting)

### 第七篇 图

- [x] [图的定义和术语](./graph-definition-terminology)
- [x] [图的存储结构（数组表示法、邻接表）](./graph-storage)
- [x] [十字链表与邻接多重表](./cross-linked-list-adjacency-multilist)
- [x] [图的遍历（深度优先搜索）](./graph-dfs)
- [x] [图的遍历（广度优先搜索）](./graph-bfs)
- [x] [图的连通性问题（无向图的连通分量和生成树）](./graph-connectivity-spanning-tree)
- [x] [有向无环图及其应用（拓扑排序）](./dag-topological-sort)
- [x] [关键路径](./critical-path)
- [x] [最短路径（Dijkstra 算法）](./dijkstra-shortest-path)
- [x] [最短路径（Floyd 算法）](./floyd-shortest-path)
- [x] [最小生成树（Prim 算法）](./prim-minimum-spanning-tree)
- [x] [最小生成树（Kruskal 算法）](./kruskal-minimum-spanning-tree)

### 第八篇 动态存储管理

- [x] [动态存储管理概述](./dynamic-storage-management-overview)
- [x] [可利用空间表及分配方法](./available-space-list)
- [x] [边界标识法](./boundary-tag-method)
- [x] [伙伴系统](./buddy-system)
- [x] [无用单元收集](./garbage-collection)
- [x] [存储紧缩](./storage-compaction)

### 第九篇 查找

- [x] [静态查找表（顺序查找）](./static-search-table-sequential)
- [x] [有序表的折半查找](./binary-search)
- [x] [索引顺序表（分块查找）](./index-sequential-search)
- [x] [二叉排序树](./binary-search-tree)
- [x] [平衡二叉树](./balanced-binary-tree)
- [x] [B-树和B+树](./b-tree-bplus-tree)
- [x] [键树](./key-trie)
- [x] [哈希表的基本概念与哈希函数构造](./hash-table-basic-concepts)
- [x] [处理哈希冲突的方法](./hash-collision-resolution)
- [x] [哈希表的查找及其分析](./hash-table-search-analysis)

### 第十篇 内部排序

- [x] [排序概述与稳定性](./sorting-overview-stability)
- [x] [直接插入排序与折半插入排序](./insertion-sort)
- [x] [希尔排序](./shell-sort)
- [x] [起泡排序](./bubble-sort)
- [x] [快速排序](./quick-sort)
- [x] [简单选择排序](./simple-selection-sort)
- [x] [树形选择排序与堆排序](./heap-sort)
- [x] [归并排序](./merge-sort)
- [x] [基数排序](./radix-sort)
- [x] [各种内部排序方法的比较讨论](./internal-sorting-comparison)

### 第十一篇 外部排序

- [x] [外存信息的存取](./external-storage-access)
- [x] [外部排序的方法](./external-sorting-methods)
- [x] [多路平衡归并的实现](./multiway-balanced-merge)
- [x] [置换-选择排序](./replacement-selection-sort)
- [x] [最佳归并树](./optimal-merge-tree)

### 第十二篇 文件

- [x] [有关文件的基本概念](./file-basic-concepts)
- [x] [顺序文件](./sequential-file)
- [x] [索引文件](./index-file)
- [x] [ISAM 文件和 VSAM 文件](./isam-vsam-file)
- [x] [直接存取文件（散列文件）](./hash-file)
- [x] [多关键字文件（多重表文件、倒排文件）](./multi-keyword-file)

### 专题篇 哈希

- [x] [一致性哈希与分布式场景应用](./consistent-hashing)
- [x] [布隆过滤器](./bloom-filter)
- [x] [哈希表工程实现（开放寻址 vs 链地址、扩容与装载因子）](./hash-table-engineering)

### 专题篇 并查集

- [x] [并查集的基本实现与路径压缩](./union-find-path-compression)
- [x] [按秩合并与复杂度分析](./union-find-rank-complexity)
- [x] [带权并查集与种类并查集](./weighted-union-find)
- [x] [并查集经典应用（连通性判定、Kruskal 中的应用）](./union-find-applications)

### 专题篇 堆与优先队列

- [x] [二叉堆的上浮与下沉](./binary-heap-sift)
- [x] [堆排序与建堆的复杂度分析](./heap-sort-complexity)
- [x] [优先队列的典型应用（Top K、合并K个有序链表）](./priority-queue-applications)
- [x] [对顶堆求动态中位数](./double-heap-median)

### 专题篇 平衡树

- [x] [AVL 树的旋转与平衡维护](./avl-tree-rotation)
- [x] [红黑树的性质与插入调整](./red-black-tree-insertion)
- [x] [红黑树的删除调整](./red-black-tree-deletion)
- [x] [红黑树的工程应用（TreeMap、epoll、进程调度）](./red-black-tree-applications)

### 专题篇 跳表

- [x] [跳表的原理与随机层数](./skip-list-principle)
- [x] [跳表的实现与复杂度分析](./skip-list-implementation)
- [x] [跳表 vs 平衡树（Redis 为什么选跳表）](./skip-list-vs-balanced-tree)

### 专题篇 B/B+树

- [x] [B树的定义与插入分裂](./b-tree-insert-split)
- [x] [B树的删除与合并](./b-tree-delete-merge)
- [x] [B+树与数据库索引](./bplus-tree-database-index)
- [x] [B+树在 MySQL InnoDB 中的落地](./bplus-tree-innodb)

### 专题篇 线段树与树状数组

- [x] [树状数组的原理与区间求和](./fenwick-tree)
- [x] [树状数组的区间修改与逆序对计数](./fenwick-tree-range-update)
- [x] [线段树的建树与单点修改、区间查询](./segment-tree-basic)
- [x] [懒惰标记与区间修改](./segment-tree-lazy-tag)
- [x] [线段树的典型应用（区间最值、扫描线）](./segment-tree-applications)

### 专题篇 Trie 与字符串匹配

- [x] [Trie 树的构建与查询](./trie-build-query)
- [x] [Trie 的应用（前缀统计、异或最大对）](./trie-applications)
- [x] [KMP 算法与 next 数组](./kmp-next-array)
- [x] [KMP 的正确性证明与复杂度分析](./kmp-correctness-complexity)
- [x] [字符串哈希与 Rabin-Karp](./string-hash-rabin-karp)

### 专题篇 单调栈与单调队列

- [x] [单调栈与下一个更大元素](./monotonic-stack-next-greater)
- [x] [单调栈应用（柱状图最大矩形、接雨水）](./monotonic-stack-applications)
- [x] [单调队列与滑动窗口最大值](./monotonic-queue-sliding-window)
- [x] [单调队列优化动态规划初步](./monotonic-queue-dp-optimization)

### 专题篇 图算法进阶

- [x] [Bellman-Ford 与 SPFA](./bellman-ford-spfa)
- [x] [差分约束系统](./difference-constraints)
- [x] [次短路与第K短路](./k-shortest-path)
- [x] [Tarjan 算法求强连通分量](./tarjan-scc)
- [x] [割点与桥](./articulation-point-bridge)
- [x] [二分图匹配（匈牙利算法）](./hungarian-bipartite-matching)
- [x] [网络流初步（最大流与 Ford-Fulkerson）](./max-flow-ford-fulkerson)
- [x] [Dinic 算法与最小割](./dinic-min-cut)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
