import { fromOutline, type Outline } from './schema'
import { quantExtra } from './quant-extra'
import { quantAudit } from './quant-audit'
import { quantFrontier } from './quant-frontier'

const outline: Outline[] = [
  [
    '市场微观结构',
    [
      [
        '订单簿',
        [
          [
            '簿与事件',
            [
              'Limit order book 结构|lob-structure',
              '限价、市价、冰山与暗池|order-types',
              'Level-2 / Level-3 数据|l2-l3-data',
              '事件时间 vs 时钟时间|event-time',
              '撤单、改单与队列位置|cancel-queue-position',
              '盘口不平衡|order-imbalance',
              '买卖价差的分解|spread-decomposition',
            ],
          ],
          [
            '价格形成',
            [
              'Roll 模型|roll-model',
              'Glosten-Milgrom|glosten-milgrom',
              'Kyle 模型|kyle-model',
              '信息交易概率 PIN|pin',
              'VPIN|vpin',
              '有效价差与实现价差|effective-realized-spread',
            ],
          ],
        ],
      ],
      [
        '高频数据',
        [
          [
            '采样与噪声',
            [
              'Tick 数据清洗|tick-cleaning',
              '异常成交与错价|outlier-trades',
              '微观结构噪声|microstructure-noise',
              '已实现波动与噪声修正|rv-noise',
              '不等间隔采样|irregular-sampling',
              '日历效应与隔夜|calendar-overnight',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '资产定价与因子',
    [
      [
        '截面因子',
        [
          [
            '经典因子',
            [
              'CAPM 与市场因子|capm',
              'Fama-French 三因子|ff3',
              'Fama-French 五因子|ff5',
              '动量 WML|momentum-wml',
              '反转|short-term-reversal',
              '价值：BM / EP / CF|value-factors',
              '质量与盈利|quality-profitability',
              '低波动 / BAB|low-vol-bab',
              '流动性因子|liquidity-factor',
              '投资与资产增长|investment-factor',
            ],
          ],
          [
            '构建与检验',
            [
              '单因子排序与多空|long-short-sort',
              '行业中性|industry-neutral',
              '市值中性|size-neutral',
              '正交化与纯化|factor-orthogonal',
              'Barra / 基本面风险模型|fundamental-risk-model',
              '统计风险模型 PCA|stat-risk-pca',
              '因子拥挤与拥挤崩|factor-crowding',
              '因子衰减与换手|factor-decay-turnover',
              '多重检验与 p-hacking|multiple-testing-factors',
            ],
          ],
        ],
      ],
      [
        '条件定价',
        [
          [
            '时变风险溢价',
            [
              '条件 CAPM|conditional-capm',
              '宏观状态变量|macro-state-variables',
              'IC / IR 与因子择时|ic-ir-timing',
              '体制切换|regime-switching-premia',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '时间序列',
    [
      [
        '收益与波动',
        [
          [
            '均值',
            [
              'ARMA / ARIMA|arma',
              '单位根与协整预备|unit-root',
              'HAR 已实现波动|har-rv',
              '已实现核估计|realized-kernel',
            ],
          ],
          [
            '波动',
            [
              'GARCH|garch',
              'EGARCH / GJR|egarch-gjr',
              '随机波动 SV|stochastic-volatility',
              '已实现 GARCH|realized-garch',
              '隐含 vs 已实现波动|iv-vs-rv',
            ],
          ],
        ],
      ],
      [
        '依赖与极值',
        [
          [
            '尾部',
            [
              '极值理论 EVT|evt',
              'Copula 相关结构|copula',
              '动态条件相关 DCC|dcc',
              '跳跃检验|jump-tests',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '统计套利',
    [
      [
        '配对与篮子',
        [
          [
            '协整',
            [
              'Engle-Granger|engle-granger',
              'Johansen|johansen',
              '残差的 Ornstein-Uhlenbeck|ou-spread',
              '半衰期与开平阈值|half-life-bands',
              'Kalman 时变对冲比|kalman-hedge',
              '协整破裂|cointegration-break',
            ],
          ],
          [
            '多资产',
            [
              '距离法配对|distance-pairs',
              'PCA / SVD 统计套利|pca-stat-arb',
              '残差动量|residual-momentum',
              '指数套利|index-arb',
              'ETF 与成分股|etf-arb',
            ],
          ],
        ],
      ],
      [
        '均值回复交易',
        [
          [
            '信号',
            [
              'Z-score 开仓|zscore-entry',
              '仓位与 Kelly / 分数 Kelly|kelly-sizing',
              '止损与时间止损|statarb-stops',
              '交易成本后的可交易性|net-edge',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '衍生品与波动率',
    [
      [
        '定价',
        [
          [
            '股权衍生品',
            [
              'Black-Scholes-Merton|bsm',
              '二叉树 / 三叉树|binomial-tree',
              '风险中性定价|risk-neutral-pricing',
              '美式期权与提前行权|american-exercise',
              '蒙特卡洛定价|mc-pricing',
              '对偶、控制变量、重要性采样|mc-variance-reduction',
              'PDE 与有限差分|option-pde',
            ],
          ],
          [
            '波动率',
            [
              '隐含波动率曲面|vol-surface',
              'Skew / Smile|vol-skew',
              'Dupire 局部波动|dupire',
              'Heston|heston',
              'SABR|sabr',
              '方差互换与 VIX|variance-swap-vix',
              'Delta / Gamma / Vega 对冲|greeks-hedge',
              '离散对冲误差|discrete-hedge-error',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '利率与固定收益',
    [
      [
        '曲线',
        [
          [
            '构建',
            [
              '即期、远期与折现|spot-forward-df',
              'Bootstrap 曲线|curve-bootstrap',
              'OIS 与多曲线|multi-curve-ois',
              '久期、凸性|duration-convexity',
              '关键利率久期|key-rate-duration',
            ],
          ],
        ],
      ],
      [
        '模型',
        [
          [
            '期限结构',
            [
              'Vasicek / CIR|vasicek-cir',
              'Hull-White|hull-white',
              'HJM|hjm',
              'LMM|lmm',
              '可转债与信用混合|convertible-credit',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '投资组合',
    [
      [
        '配置',
        [
          [
            '均值方差',
            [
              'Markowitz|markowitz',
              '估计误差与收缩|cov-shrinkage',
              'Black-Litterman|black-litterman',
              '风险平价|risk-parity',
              '最大分散 / 风险预算|risk-budgeting',
              '约束：换手、行业、杠杆|portfolio-constraints',
              '再平衡规则|rebalance-rules',
            ],
          ],
        ],
      ],
      [
        '业绩',
        [
          [
            '归因',
            [
              'Brinson 归因|brinson',
              '因子归因|factor-attribution',
              'IR / Sharpe / Sortino|ir-sharpe',
              '最大回撤与 Calmar|drawdown-calmar',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '执行与交易成本',
    [
      [
        '冲击',
        [
          [
            '模型',
            [
              '买卖价差成本|spread-cost',
              '线性 / 平方根冲击|sqrt-impact',
              'Almgren-Chriss 最优执行|almgren-chriss',
              '瞬时 vs 永久冲击|temp-perm-impact',
              'TWAP / VWAP / POV|twap-vwap-pov',
              '实施缺口 Implementation Shortfall|implementation-shortfall',
            ],
          ],
        ],
      ],
      [
        '做市',
        [
          [
            '库存',
            [
              'Avellaneda-Stoikov|avellaneda-stoikov',
              '库存风险与偏度报价|inventory-skew',
              '排队与成交概率|fill-probability',
              '毒性流与停报价|toxic-flow',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '风险',
    [
      [
        '度量',
        [
          [
            '市场风险',
            [
              'VaR：历史、参数、MC|var-methods',
              'Expected Shortfall|expected-shortfall',
              '回测 VaR：Kupiec / Christoffersen|var-backtest',
              '杠杆、保证金与强平|leverage-liquidation',
              '流动性风险与变现时间|liquidity-horizon',
              '对手方与信用风险要点|counterparty-credit',
            ],
          ],
        ],
      ],
      [
        '模型风险',
        [
          [
            '失效',
            [
              '过拟合作为风险|overfit-as-risk',
              '体制切换失效|regime-break-risk',
              '相关性崩溃|correlation-breakdown',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '研究与回测',
    [
      [
        '偏差',
        [
          [
            '常见坑',
            [
              '幸存者偏差|survivorship-bias',
              '前视偏差|look-ahead-bias',
              '点-in-time 基本面|point-in-time',
              '停牌、涨跌停与不可交易|trading-halts',
              '分红、拆股与复权|corporate-actions',
              '样本外与滚动检验|walk-forward',
              '交叉验证在金融中的泄漏|cv-leakage-finance',
            ],
          ],
        ],
      ],
      [
        '实验设计',
        [
          [
            '稳健',
            [
              'Deflated Sharpe|deflated-sharpe',
              '组合过拟合 CPCV|cpcv',
              '费用、滑点敏感性|cost-sensitivity',
              '容量与参与率|capacity-participation',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '交易系统',
    [
      [
        '链路',
        [
          [
            '工程',
            [
              '研究、模拟、实盘隔离|research-sim-prod',
              '事件驱动回测引擎|event-driven-backtest',
              '时钟、迟到数据与对账|late-data-recon',
              '订单状态机|order-state-machine',
              '风控闸门与熔断|trading-kill-switch',
              '监控：PnL、敞口、拒单|trading-telemetry',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '另类数据',
    [
      [
        '来源',
        [
          [
            '类型',
            [
              '卫星与地理|satellite-geo',
              '信用卡与消费|card-spending',
              '供应链与货运|supply-chain-alt',
              '舆情与文本|news-nlp-alpha',
              '期权链作为现货信号|options-as-spot-signal',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '机器学习方法',
    [
      [
        '监督',
        [
          [
            '表格与序列',
            [
              '标签、预测期与重叠|label-horizon-overlap',
              '特征：滞后、截面 rank|cs-rank-features',
              '树模型：GBDT|gbdt-alpha',
              '正则线性模型|regularized-linear-alpha',
              '防止未来函数|no-future-function',
            ],
          ],
        ],
      ],
      [
        '强化学习交易',
        [
          [
            '设定',
            [
              '交易作为 MDP 的陷阱|trading-mdp-pitfalls',
              '模拟器偏差|sim-to-real-trading',
              '执行 RL 与冲击|rl-execution',
            ],
          ],
        ],
      ],
    ],
  ],
]

export const quantTree = [...fromOutline(outline), ...quantExtra, ...quantAudit, ...quantFrontier]
