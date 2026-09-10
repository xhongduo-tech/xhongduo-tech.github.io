import type { Outline } from './schema'

/** 金融栏第一课：选择、均衡与动态规划要用的数学。 */
export const econFoundations: Outline = [
  '数学与优化基础',
  [
    [
      '集合与凸',
      [
        [
          '空间',
          [
            '欧氏空间与开集|euclidean-open-set',
            '凸集与凸组合|convex-set-combo',
            '凸函数与凹函数|convex-concave-fn',
            '上境图与下水平集|epigraph-sublevel',
            '分离超平面|separating-hyperplane',
          ],
        ],
      ],
    ],
    [
      '优化',
      [
        [
          '约束',
          [
            '无约束一阶与二阶条件|unconstrained-fonc',
            '拉格朗日乘子|lagrange-multiplier',
            'KKT 条件|kkt-conditions',
            '包络定理|envelope-theorem',
            '隐函数定理|implicit-function',
            '比较静态|comparative-statics',
          ],
        ],
      ],
    ],
    [
      '不动点与动态',
      [
        [
          '存在性',
          [
            '压缩映射|contraction-mapping',
            'Brouwer 不动点|brouwer-fixed-point',
            'Kakutani 不动点|kakutani-fixed-point',
            '上半连续对应|uhc-correspondence',
            '贝尔曼方程|bellman-equation',
            '值函数迭代|value-function-iteration',
          ],
        ],
      ],
    ],
    [
      '概率够用的一层',
      [
        [
          '测度与期望',
          [
            '样本空间与 σ-代数|sigma-algebra',
            '条件期望|conditional-expectation-econ',
            '大数定律与中心极限|lln-clt',
            '随机占优预备|stochastic-order-prep',
          ],
        ],
      ],
    ],
  ],
]

/** 领域课：不并入微观/宏观主干，接在贸易之后。 */
export const econFields: Outline[] = [
  [
    '产业组织',
    [
      [
        '结构与策略',
        [
          [
            '从市场结构往下',
            [
              'SCP 与新实证 IO|scp-neio',
              '差异化产品定价|differentiated-pricing',
              '进入、退出与沉没成本|entry-exit-sunk',
              '纵向约束|vertical-restraints',
              '网络效应与标准|network-standards-io',
              '平台双边定价续|two-sided-pricing-io',
              '合并模拟|merger-simulation',
              '掠夺与排他|predation-exclusion',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '公共财政',
    [
      [
        '税与支出',
        [
          [
            '政府做什么',
            [
              '公共品自愿供给失败|public-goods-free-rider',
              '最优商品税 Ramsey|ramsey-optimal-commodity-tax',
              '所得税与激励|income-tax-incentives',
              '再分配与效率代价|redistribution-efficiency',
              '社会保障与年金|social-security-annuity',
              '财政乘数争议|fiscal-multiplier-debate',
              '主权债务可持续|sovereign-debt-sustain',
              '地方财政与转移支付|local-fiscal-transfers',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '经济史',
    [
      [
        '危机与制度',
        [
          [
            '对照',
            [
              '金本位与大萧条|gold-standard-depression',
              '布雷顿森林与崩溃|bretton-woods-collapse',
              '1970 年代滞胀|stagflation-1970s',
              '东亚危机|east-asian-crisis',
              '2008 与影子银行|gfc-shadow-banking',
              '欧债危机|eurozone-debt-crisis',
              '央行独立性简史|cb-independence-history',
            ],
          ],
        ],
      ],
    ],
  ],
]
