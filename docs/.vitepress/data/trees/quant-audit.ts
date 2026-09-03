import { fromOutline, type Outline } from './schema'

/** 专业审核补层：微观结构/因子/套利/执行/加密 + 波动率/XVA/组合风险/回测纪律 */
const extra: Outline[] = [
  [
    '微观结构续',
    [
      [
        '成交方向与采样',
        [
          [
            '方向分类',
            [
              'Lee-Ready 算法|lee-ready',
              'Tick Rule / Quote Rule|tick-quote-rule',
              "Ellis-Michaely-O'Hara 改进|emo-trade-sign",
              'Bulk Volume Classification|bulk-volume-classification',
              '逆向选择成本度量|adverse-selection-cost',
            ],
          ],
          [
            '信息驱动 Bars',
            [
              'Tick bars|tick-bars',
              'Volume bars|volume-bars',
              'Dollar bars|dollar-bars',
              'Tick Imbalance Bars|tick-imbalance-bars',
              'Volume Imbalance Bars|volume-imbalance-bars',
              'Dollar Imbalance Bars|dollar-imbalance-bars',
            ],
          ],
        ],
      ],
      [
        '簿动力学与点过程',
        [
          [
            '队列与强度',
            [
              'Queue-reactive 模型|queue-reactive',
              'Cont-Kukanov-Stoikov 最优挂单|cks-optimal-placement',
              'Hawkes 订单簿强度|hawkes-lob',
              '多元 Hawkes 互激|multivariate-hawkes',
              'ACD 自回归条件久期|acd-duration',
              'LOBSTER / ITCH 行情|lobster-itch-feed',
              'Bouchaud 传播核|bouchaud-propagator',
              '平方根冲击律|square-root-impact-law',
              'Madhavan-Richardson-Roomans|mrr-decomposition',
              '延迟套利|latency-arbitrage',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '因子与横截面续',
    [
      [
        '经典扩展与异象',
        [
          [
            '定价因子',
            [
              'Carhart 四因子|carhart4',
              '特质波动 IVOL|ivol-anomaly',
              'MAX 极端收益效应|max-effect',
              'Baker-Wurgler 情绪|baker-wurgler',
              '彩票型需求因子|lottery-demand',
              '动量崩溃|momentum-crash',
              '时序 vs 截面动量|ts-vs-cs-momentum',
              '隔夜 / 日内 alpha|overnight-intraday-alpha',
              '行业动量|industry-momentum',
              '52 周高点动量|fiftytwo-week-high',
              'GRS 检验|grs-test',
              'Harvey-Liu-Zhu 多重检验|harvey-liu-zhu',
            ],
          ],
          [
            'A 股截面补充',
            [
              '壳溢价 / 小市值|cn-shell-premium',
              '换手率因子|cn-turnover-factor',
              '北向持股因子|cn-northbound-hold',
              '涨跌停邻近效应|cn-limit-proximity',
              '龙虎榜异常|cn-lhb-anomaly',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '统计套利与高频续',
    [
      [
        '协整、体制与事件',
        [
          [
            '多元与断裂',
            [
              'VECM 向量误差修正|vecm',
              'Gregory-Hansen 协整破|gregory-hansen',
              'Bai-Perron 结构断点|bai-perron',
              'CUSUM / MOSUM|cusum-mosum',
              'HMM 隐马尔可夫体制|hmm-regime',
              '门限协整|threshold-cointegration',
              'OU 最优停时|ou-optimal-stopping',
              'Hayashi-Yoshida 相关|hayashi-yoshida',
              'Epps 效应|epps-effect',
              '并购套利|merger-arbitrage',
              '可转债套利|convertible-arb',
              '指数调仓套利|index-rebalance-arb',
            ],
          ],
        ],
      ],
      [
        '回测验证补全',
        [
          [
            '过拟合与重要性',
            [
              'Probability of Backtest Overfitting|pbo',
              '嵌套交叉验证|nested-cv',
              'MDA 特征重要性|mda-importance',
              'White Reality Check|white-reality-check',
              'Hansen SPA 检验|hansen-spa',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '执行与市场冲击续',
    [
      [
        '费用、做市与冲击',
        [
          [
            '微观费用与最优执行',
            [
              'Maker-taker 费用|maker-taker',
              'Guéant-Lehalle-Fernandez-Tapia|glft-market-making',
              'Cartea-Jaimungal 执行|cartea-jaimungal',
              'Bertsimas-Lo 动态执行|bertsimas-lo',
              'Huberman-Stanzl 无套利冲击|huberman-stanzl',
              '瞬时冲击传播核|transient-impact-kernel',
              '自适应 POV|adaptive-pov',
              '子单切片|child-order-slicing',
              '滑点归因|slippage-attribution',
              '收盘竞价不平衡|close-auction-imbalance',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '另类与加密续',
    [
      [
        '加密微观与链上',
        [
          [
            '永续、AMM 与 MEV',
            [
              '永续合约资金费率|crypto-perp-funding',
              'Funding rate 套利|funding-rate-arb',
              '加密期现基差交易|crypto-basis-trade',
              '清算连锁|liquidation-cascade',
              'Uniswap 恒定乘积 AMM|uniswap-cpmm',
              'Uniswap v3 集中流动性|uniswap-v3-clmm',
              'MEV sandwich|mev-sandwich',
              'CEX-DEX 套利|cex-dex-arb',
              '稳定币脱锚风险|stablecoin-depeg',
              '电话会语气 NLP|earnings-call-tone',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '波动率与期权续',
    [
      [
        '局部与随机波动',
        [
          [
            '曲面、产品与对冲',
            [
              'Heston Feller 条件|heston-feller',
              'SABR 翼部外推|sabr-wing-extrap',
              'SVI 无套利参数化|svi-arb-params',
              'SSVI 期限结构一致性|ssvi-term-consistency',
              'Rough Bergomi|rough-bergomi',
              '局部-随机混合校准|lsv-hybrid-calib',
              '方差互换 vs 波动互换|var-vs-vol-swap',
              'Gamma scalping PnL|gamma-scalping-pnl',
              'Dispersion 相关溢价|dispersion-corr-prem',
              'Sticky delta / sticky strike|sticky-delta-strike',
              'VIX 期限结构交易|vix-term-structure-trade',
              'Risk reversal 偏度交易|risk-reversal-skew',
              '模型无关隐含矩|model-free-moments',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '利率信用与 XVA',
    [
      [
        '利率与对手方',
        [
          [
            'HW / LMM / CMS',
            [
              'Hull-White 树校准|hw-tree-calib',
              'Hull-White 两因子|hw-two-factor',
              'LMM 漂移与测度变换|lmm-drift-measure',
              'CMS 复制与凸性调整|cms-replication',
              '利率-信用混合定价|rates-credit-hybrid',
            ],
          ],
          [
            '信用与 XVA',
            [
              'CDS 生存曲线自举|cds-survival-bootstrap',
              'KMV 违约距离 DD|kmv-dd',
              'CVA 暴露剖面 EPE|cva-exposure-profile',
              'DVA 自身信用调整|dva-own-credit',
              'FVA 融资估值调整|fva-funding',
              'MVA 初始保证金调整|mva-initial-margin',
              'SA-CCR EAD|sa-ccr-ead',
              'IMM EEPE 内部模型|imm-eepe',
              'Wrong-way risk|wrong-way-risk',
              'SIMM 敏感度保证金|simm-delta-vega',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '组合与风险续',
    [
      [
        '配置、尾部与压力',
        [
          [
            '仓位与尾部',
            [
              'Black-Litterman 先验收缩|bl-prior-shrink',
              'ERC 与边际风险贡献|erc-mrc',
              'HRP 联结与准对角化|hrp-linkage',
              '分数 Kelly 边界|fractional-kelly-bound',
              'CVaR Rockafellar-Uryasev|cvar-ru-opt',
              '回撤约束优化|drawdown-constrained-opt',
              '组合保险 CPPI / TIPP|cppi-tipp',
              'HAR-RV 多尺度预测|har-rv-forecast',
              'Barndorff-Nielsen 跳跃检验|bn-jump-test',
              'Copula-CoVaR|copula-covar',
              '反向压力测试|reverse-stress-test',
              '相关性崩溃情景|corr-break-scenario',
              '拥挤度与容量指标|crowding-capacity-metric',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '回测与研究设计续',
    [
      [
        '多重检验与稳健性',
        [
          [
            '检验族与纪律',
            [
              'Nested CPCV|nested-cpcv',
              'FDR / Romano-Wolf|fdr-romano-wolf',
              'CSCV 组合对称交叉验证|cscv',
              '策略容量与冲击衰减|strategy-capacity-decay',
              '拥挤 alpha 衰减|crowded-alpha-decay',
              '费用滑点蒙特卡洛|cost-mc-sensitivity',
              '经济显著性 vs 统计显著性|econ-vs-stat-sig',
              '实盘漂移监测|live-drift-monitor',
            ],
          ],
        ],
      ],
    ],
  ],
]

export const quantAudit = fromOutline(extra)
