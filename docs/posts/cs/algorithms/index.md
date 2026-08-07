---
pageClass: plain-doc
---

# 算法设计与分析

以《算法导论》（CLRS）为纲，覆盖算法设计与分析的经典内容：从渐进记号、分治与随机化，到动态规划、贪心、图算法、NP 完全性，再到近似算法、字符串匹配、计算几何、数论算法、并行与外部存储算法。

## 主题规划

<ProgressGrid cat="cs/algorithms" />


### 第一篇：算法基础与渐进记号

- [x] [算法的定义、性质与作为一门技术的地位](./what-is-an-algorithm)
- [x] [插入排序（Insertion Sort）：循环不变式与正确性证明](./insertion-sort)
- [x] [算法分析框架：最坏情况、平均情况与渐近效率](./algorithm-analysis)
- [x] [渐进记号（Asymptotic Notation）：Θ、O、Ω、o、ω 的严格定义与性质](./asymptotic-notation)
- [x] [常用函数的增长率比较与求和技巧（级数、积分近似、调和数）](./growth-functions-summations)

### 第二篇：分治与递归

- [x] [归并排序（Merge Sort）：分治范式与递归树分析](./merge-sort)
- [x] [求解递归式的代入法（Substitution Method）](./substitution-method)
- [x] [递归树方法（Recursion Tree）：展开与求和](./recursion-tree)
- [x] [主定理（Master Theorem）：三种情形与间隙讨论](./master-theorem)
- [x] [分治经典案例：最大子数组问题](./maximum-subarray)
- [x] [分治经典案例：矩阵乘法（Strassen 算法）](./strassen-matrix-multiplication)

### 第三篇：随机化算法

- [x] [随机化算法的基本概念：雇佣问题与指示器随机变量](./randomized-hiring-problem)
- [x] [随机排列算法：按优先级排序与原地洗牌（Fisher–Yates）](./random-permutation-fisher-yates)
- [x] [概率分析进阶：球与箱子问题（Balls and Bins）](./balls-and-bins)
- [x] [生日悖论（Birthday Paradox）及其算法含义](./birthday-paradox)
- [x] [随机化快速排序的期望时间分析](./randomized-quicksort-analysis)

### 第四篇：堆排序与快速排序

- [x] [二叉堆（Binary Heap）：堆结构维护与 HEAPIFY](./binary-heap)
- [x] [堆排序（Heapsort）：建堆的 O(n) 分析](./heapsort)
- [x] [优先队列：最大堆、最小堆与 d 叉堆](./priority-queue)
- [x] [快速排序（Quicksort）：划分过程与正确性](./quicksort-partition)
- [x] [快速排序的性能分析：最坏情况与平衡划分](./quicksort-performance)
- [x] [随机化快速排序的期望运行时间证明](./randomized-quicksort-expected)
- [x] [快速排序的工程优化：三数取中、三向划分、小数组插排](./quicksort-optimization)

### 第五篇：线性时间排序

- [x] [比较排序下界：决策树模型与 Ω(n log n) 证明](./comparison-sort-lower-bound)
- [x] [计数排序（Counting Sort）：稳定性与适用条件](./counting-sort)
- [x] [基数排序（Radix Sort）：低位优先与正确性归纳](./radix-sort)
- [x] [桶排序（Bucket Sort）：均匀分布假设下的期望分析](./bucket-sort)

### 第六篇：中位数与顺序统计量

- [x] [最小值与最大值：比较次数的精确下界](./min-max-lower-bound)
- [x] [期望线性时间选择：RANDOMIZED-SELECT](./randomized-select)
- [x] [最坏情况线性时间选择：SELECT 与中位数的中位数（Median of Medians）](./select-median-of-medians)
- [x] [顺序统计量的应用：带权中位数与最近邮局问题](./order-statistics-applications)

### 第七篇：动态规划

- [x] [动态规划原理：最优子结构与重叠子问题](./dynamic-programming-principles)
- [x] [钢条切割问题（Rod Cutting）：自顶向下备忘录与自底向上](./rod-cutting)
- [x] [矩阵链乘法（Matrix-Chain Multiplication）：括号化的最优次序](./matrix-chain-multiplication)
- [x] [最长公共子序列（Longest Common Subsequence）：递推式与回溯构造](./longest-common-subsequence)
- [x] [最长公共子序列的变体：编辑距离（Edit Distance）](./edit-distance)
- [x] [最优二叉搜索树（Optimal BST）：期望搜索代价最小化](./optimal-bst)
- [x] [0-1 背包问题（0-1 Knapsack）：伪多项式时间与完全多项式的区别](./knapsack-01)
- [x] [0-1 背包的空间优化与方案回溯](./knapsack-space-optimization)
- [x] [分数背包（Fractional Knapsack）：贪心性质及其与 0-1 背包的本质差异](./fractional-knapsack)
- [x] [最长递增子序列（LIS）：O(n²) 动态规划与 O(n log n) 优化](./longest-increasing-subsequence)

### 第八篇：贪心算法

- [x] [贪心算法的理论基础：贪心选择性质与最优子结构](./greedy-principles)
- [x] [活动选择问题（Activity-Selection）：按结束时间排序的贪心证明](./activity-selection)
- [x] [哈夫曼编码（Huffman Coding）：最优前缀码的构造与正确性](./huffman-coding)
- [x] [拟阵（Matroid）：贪心理论的抽象框架](./matroid)
- [x] [拟阵视角下的任务调度问题：带截止时间单位任务的调度](./matroid-task-scheduling)

### 第九篇：摊还分析

- [x] [摊还分析导论：聚合分析（Aggregate Analysis）与栈操作](./amortized-aggregate)
- [x] [核算法（Accounting Method）：二进制计数器的信用分配](./accounting-method)
- [x] [势能法（Potential Method）：动态表的扩缩容分析](./potential-method)
- [x] [动态表（Dynamic Table）：插入与删除的摊还代价](./dynamic-table)
- [x] [摊还分析与平均情况分析的区别](./amortized-vs-average)

### 第十篇：图算法深入

- [x] [图的表示与遍历复习：BFS、DFS 及其性质](./graph-traversal-bfs-dfs)
- [x] [拓扑排序（Topological Sort）：DFS 完成时间逆序的证明](./topological-sort)
- [x] [强连通分量（Strongly Connected Components）：Kosaraju 算法与分量图](./strongly-connected-components)
- [x] [单源最短路径：Bellman-Ford 与 DAG 上的最短路径](./bellman-ford-dag-shortest-path)
- [x] [全源最短路径：Floyd-Warshall 与 Johnson 算法](./all-pairs-shortest-paths)
- [x] [最大流（Maximum Flow）：流网络、残量网络与增广路](./max-flow-networks)
- [x] [最大流最小割定理（Max-Flow Min-Cut Theorem）及其推论](./max-flow-min-cut)
- [x] [Ford-Fulkerson 方法及其变体：Edmonds-Karp 算法](./ford-fulkerson-edmonds-karp)
- [x] [二分图最大匹配（Maximum Bipartite Matching）：归约到最大流](./bipartite-matching-max-flow)
- [x] [推送-重贴标签算法（Push-Relabel）初步](./push-relabel)

### 第十一篇：NP 完全性

- [x] [多项式时间：P 类问题与编码方式的影响](./polynomial-time-p)
- [x] [多项式时间验证：NP 类问题与证书（Certificate）](./np-verification-certificate)
- [x] [归约（Reduction）：多项式时间归约与引理证明](./polynomial-time-reduction)
- [x] [NP 完全性（NP-Complete）：定义、性质与 Cook-Levin 定理](./np-complete-cook-levin)
- [x] [NPC 证明范式：以顶点覆盖（Vertex Cover）为例](./vertex-cover-npc-proof)
- [x] [经典 NPC 问题：哈密顿回路（Hamiltonian Cycle）](./hamiltonian-cycle-npc)
- [x] [经典 NPC 问题：旅行商问题（TSP）](./tsp-npc)
- [x] [经典 NPC 问题：子集和（Subset-Sum）与 3-CNF 可满足性](./subset-sum-3cnf-sat)

### 第十二篇：近似算法

- [x] [近似算法的性能比：绝对近似比与相对误差界](./approximation-ratio)
- [x] [顶点覆盖问题的 2-近似算法](./vertex-cover-2-approx)
- [x] [旅行商问题的近似算法：满足三角不等式时的 2-近似](./tsp-2-approx)
- [x] [集合覆盖（Set Cover）：贪心算法的 ln n 近似比](./set-cover-greedy)
- [x] [子集和问题的完全多项式时间近似方案（FPTAS）](./subset-sum-fptas)

### 第十三篇：字符串匹配进阶

- [x] [朴素字符串匹配及其缺陷分析](./naive-string-matching)
- [x] [Rabin-Karp 算法：滚动哈希（Rolling Hash）](./rabin-karp)
- [x] [有限自动机（DFA）字符串匹配：转移函数的构造](./string-matching-finite-automata)
- [x] [KMP 算法：前缀函数与线性时间匹配](./kmp)
- [x] [Boyer-Moore 算法：坏字符规则与好后缀规则](./boyer-moore)

### 第十四篇：计算几何初步

- [x] [线段的几何性质：叉积（Cross Product）与方向判定](./cross-product)
- [x] [线段相交判定：跨立实验（Straddling Test）](./segment-intersection)
- [x] [任意线段对相交检测：扫描线算法（Sweep Line）](./sweep-line-segment-intersection)
- [x] [凸包（Convex Hull）：Graham 扫描法](./graham-scan)
- [x] [凸包（Convex Hull）：Jarvis 步进法（礼品包装）](./jarvis-march)
- [x] [最近点对问题：分治法与 O(n log n) 分析](./closest-pair)

### 第十五篇：数论算法

- [x] [初等数论概念：整除性、最大公约数与欧几里得算法](./euclidean-algorithm)
- [x] [模运算与扩展欧几里得算法：线性同余方程求解](./extended-euclid)
- [x] [中国剩余定理（Chinese Remainder Theorem）与模幂运算](./crt-modular-exponentiation)
- [x] [RSA 公钥密码体制：密钥生成、加密与正确性](./rsa)
- [x] [素性测试：费马小定理、Miller-Rabin 素性测试](./miller-rabin)
- [x] [整数因子分解：Pollard 的 rho 启发式算法](./pollard-rho)

### 第十六篇：并行算法初步

- [x] [多线程计算模型：动态多线程、工作（Work）与跨度（Span）](./multithreaded-model-work-span)
- [x] [并行斐波那契计算：性能度量与调度](./parallel-fibonacci)
- [x] [并行矩阵乘法与并行归并](./parallel-matrix-merge)
- [x] [贪心调度定理与并行算法的竞态条件](./greedy-scheduling-race)

### 第十七篇：外部存储算法

- [x] [外部存储模型：I/O 复杂度与磁盘块访问代价](./external-memory-model)
- [x] [外部排序：多路归并排序（External Merge Sort）](./external-merge-sort)
- [x] [B 树：定义、性质与磁盘友好的查找结构](./b-tree)
- [x] [B 树的插入：节点分裂（Split）操作](./b-tree-insert)
- [x] [B 树的删除：合并与借位操作](./b-tree-delete)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
