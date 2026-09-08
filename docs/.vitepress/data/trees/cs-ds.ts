import type { Outline } from './schema'

/** 数据结构：接口与代价 → 树/散列 → 图表示。 */
export const csDs: Outline = [
  '数据结构',
  [
    [
      '抽象',
      [
        [
          '接口与代价',
          [
            '抽象数据类型|adt-cost',
            '数组与随机访问|array-random-access',
            '动态数组扩容|dynamic-array',
            '链表与局部性代价|linked-list-locality',
            '双向与循环链表|dll-circular',
            '栈与调用|stack-adt',
            '队列与缓冲|queue-buffer',
            '双端队列|deque',
            '环形缓冲|ring-buffer',
            '摊还分析|amortized-analysis',
          ],
        ],
      ],
    ],
    [
      '树与散列',
      [
        [
          '对数查找',
          [
            '二叉查找树|bst',
            'BST 删除与后继|bst-delete',
            'AVL 与旋转|avl-rotate',
            '红黑树直觉|rbtree-intuition',
            'B 树与外存|btree-external',
            'B+ 树作为结构|bplus-as-ds',
            '堆与优先队列|heap-priority',
            'd 叉堆与索引堆|dary-heap',
            '散列函数|hash-function',
            '链地址与开放寻址|chaining-open-address',
            'Robin Hood 与布谷|robin-cuckoo',
            '布隆过滤器|bloom-filter',
            '跳表|skip-list',
            'Trie|trie',
            '基数树|radix-tree',
            '并查集|union-find',
            '按秩与路径压缩|uf-rank-compress',
          ],
        ],
      ],
    ],
    [
      '图结构',
      [
        [
          '表示',
          [
            '邻接表与邻接矩阵|adj-list-matrix',
            '稀疏图|sparse-graph',
            '边表与 CSR|csr-graph',
            'DAG 与拓扑序预备|dag-repr',
            '平面图直觉|planar-graph',
          ],
        ],
      ],
    ],
  ],
]
