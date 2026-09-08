import type { Outline } from './schema'

/** 算法：分析、排序、图、范式与难解。 */
export const csAlgo: Outline = [
  '算法',
  [
    [
      '正确与代价',
      [
        [
          '分析',
          [
            '循环不变式|loop-invariant',
            '主定理|master-theorem',
            'Akra–Bazzi 对照|akra-bazzi',
            '分治|divide-conquer',
            '二分查找|binary-search',
          ],
        ],
      ],
    ],
    [
      '排序与选择',
      [
        [
          '比较模型',
          [
            '插入与归并|insertion-merge',
            '堆排序|heapsort',
            '快排与期望|quicksort-expected',
            '三路快排|quicksort-3way',
            '下界 Ω(n log n)|sort-lower-bound',
            '线性时间排序|linear-time-sort',
            '基数与桶|radix-bucket',
            '选择第 k 小|select-kth',
            '中位数的中位数|median-of-medians',
          ],
        ],
      ],
    ],
    [
      '图算法',
      [
        [
          '遍历到流',
          [
            'BFS 与无权最短路|bfs-unweighted',
            'DFS 与边分类|dfs-edge-types',
            '连通分量与桥|bcc-bridge',
            '拓扑排序|topo-sort',
            'Dijkstra|dijkstra',
            'Dial 与堆变体|dijkstra-heap',
            'Bellman–Ford|bellman-ford',
            '差分约束|diff-constraint',
            'Floyd–Warshall|floyd-warshall',
            'Johnson 全源|johnson-apsp',
            'Kruskal 与 Prim|mst-kruskal-prim',
            '最大流 Ford–Fulkerson|max-flow-ff',
            'Edmonds–Karp 与 Dinic|dinic',
            '二分图匹配|bipartite-match',
            'KMP 串匹配|kmp',
            'Boyer–Moore|boyer-moore',
            'Aho–Corasick|aho-corasick',
          ],
        ],
      ],
    ],
    [
      '设计范式与难解',
      [
        [
          '范式',
          [
            '贪心正确性|greedy-correct',
            '交换论证|exchange-argument',
            '动态规划|dynamic-programming',
            '最优子结构与重叠|opt-substructure',
            '背包|knapsack',
            '区间 DP 直觉|interval-dp',
            'P 与 NP|p-vs-np',
            '多项式归约|np-reduction',
            'NPC 典型问题|npc-canonical',
            'SAT 与 3-SAT|sat-3sat',
            '近似比|approximation-ratio',
            '随机化算法直觉|randomized-algo',
          ],
        ],
      ],
    ],
  ],
]
