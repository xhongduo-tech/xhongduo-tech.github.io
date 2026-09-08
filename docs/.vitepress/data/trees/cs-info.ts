import type { Outline } from './schema'

/** 信息与离散：从比特到后课要用的数学。 */
export const csInfo: Outline = [
  '信息与离散',
  [
    [
      '表示',
      [
        [
          '从比特到编码',
          [
            '比特作为区分|bit-as-distinction',
            '进制与位权|positional-notation',
            'Gray 码|gray-code',
            '补码与溢出|twos-complement',
            '符号幅度与反码对照|sign-magnitude',
            '定点与浮点|fixed-and-float',
            'IEEE 754|ieee-754',
            '舍入模式|rounding-modes',
            '非规格化与 NaN|denormal-nan',
            '字符与 Unicode|char-unicode',
            'UTF-8 变长|utf8',
            '信息量与熵|entropy-bits',
            '联合熵与互信息|mutual-information',
            '纠错码直觉|error-correcting-intuition',
            '汉明距离|hamming-distance',
            '奇偶与 CRC|parity-crc',
          ],
        ],
      ],
    ],
    [
      '布尔',
      [
        [
          '开关代数',
          [
            '布尔代数公理|boolean-algebra',
            '德摩根与对偶|de-morgan',
            '最小项与最大项|minterm-maxterm',
            '卡诺图与化简|karnaugh-map',
            'Quine–McCluskey|quine-mccluskey',
            '不完全指定与险象|dont-care-hazard',
            '静态与动态险象|static-dynamic-hazard',
          ],
        ],
      ],
    ],
    [
      '离散结构',
      [
        [
          '后课要用的数学',
          [
            '集合与关系|sets-relations',
            '等价关系与划分|equivalence-partition',
            '偏序与格直觉|poset-lattice',
            '函数与可数|functions-countable',
            '数学归纳|mathematical-induction',
            '强归纳与结构归纳|strong-induction',
            '鸽笼原理|pigeonhole',
            '容斥|inclusion-exclusion',
            '渐近记号|asymptotic-notation',
            '常见和式与递推|sum-recurrence',
            '图的定义|graph-definition',
            '度、连通与割|degree-cut',
            '树作为无环连通图|tree-as-acyclic',
            '组合计数|counting-combinatorics',
            '离散概率够用的那一层|discrete-probability',
            '期望与线性性|expectation-linearity',
            '条件概率与贝叶斯|conditional-bayes',
          ],
        ],
      ],
    ],
  ],
]
