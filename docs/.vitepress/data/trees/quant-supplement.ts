import { fromOutline, type Outline } from './schema'

/**
 * 量化补层（2026-09 审视）：接在 quantExtra 之后、附录之前。
 * 金融微观结构与定价，不是权重量化。默认已读完主干与既有补层；叶子均为待撰写。
 */
const extra: Outline[] = [
  [
    '信息与库存之后的微观结构',
    [
      [
        '信息与库存模型',
        [
          [
            'Kyle 之后',
            [
              'Kyle 多期与连续时间|kyle-multiperiod',
              'Back 连续时间内幕交易|back-continuous-insider',
              'Easley–O’Hara 事件不确定|easley-ohara-event',
              'Ho–Stoll 库存做市|ho-stoll-inventory',
              'Grossman–Miller 流动性提供|grossman-miller',
              'Amihud–Mendelson 价差与预期收益|amihud-mendelson',
              'Stoll 价差三成分|stoll-three-components',
              'Hasbrouck 报价-成交 VAR|hasbrouck-var',
              'Gonzalo–Granger 永久成分|gonzalo-granger',
              '价格效率度量：方差比与自相关|price-efficiency-measures',
              '策略交易者与噪声交易者|strategic-noise-traders',
              '透明度与匿名|transparency-anonymity',
            ],
          ],
        ],
      ],
      [
        '限价簿理论',
        [
          [
            '排队、竞争与形状',
            [
              'Parlour 限价簿排队|parlour-lob',
              'Foucault 限价簿竞争|foucault-lob',
              'Roşu 动态限价簿|rosu-dynamic-lob',
              'Glosten 电子限价簿|glosten-electronic-lob',
              '限价单对市价单的选择|limit-vs-market-choice',
              '逆向选择对非执行风险|adverse-selection-vs-nonexecution',
              '最优限价单放置理论|optimal-limit-placement',
              '价格-时间对 pro-rata|price-time-pro-rata',
              '隐藏流动性理论|hidden-liquidity-theory',
              '报价与撤单博弈|quote-cancellation-game',
              '簿的宏观形状|lob-shape-macro',
              '簿的均场极限|lob-mean-field',
            ],
          ],
        ],
      ],
      [
        '高频交易理论',
        [
          [
            '速度的经济学',
            [
              'Budish–Cramton–Shim 频繁批量拍卖|frequent-batch-auction',
              '速度竞赛与狙击|speed-race-sniping',
              'Menkveld 高频做市实证|menkveld-hft-mm',
              'Brogaard 高频与价格发现|brogaard-hft-discovery',
              '2010 闪电崩盘|flash-crash-2010',
              '报价填塞|quote-stuffing',
              '幌骚与分层|spoofing-layering',
              '做市商义务与撤退|mm-obligations-withdrawal',
              '波动率-流动性反馈|vol-liquidity-feedback',
              '微观结构不变量|microstructure-invariance',
              '交易速度与市场质量|speed-market-quality',
            ],
          ],
        ],
      ],
      [
        '市场设计续',
        [
          [
            '规则与场所',
            [
              'peg 与 D-limit 订单|peg-d-limit-orders',
              'IEX 速度颠簸|speed-bump-iex',
              '收盘竞价机制对比|closing-auction-design',
              '期货交易所匹配算法|futures-matching-algo',
              '期权多腿与复杂单|options-complex-orders',
              '零售流与批发商|retail-wholesaler',
              'Reg NMS 与订单保护|reg-nms-order-protection',
              'tick size pilot 实证|tick-size-pilot',
              '熔断机制设计|circuit-breaker-design',
              '盘前盘后交易|pre-post-market',
              '场外与经销商市场|otc-dealer-market',
              'RFQ 平台|rfq-platforms',
              '债券电子化与全对全|bond-electronification',
              '外汇市场结构|fx-market-structure',
              'FX last look|fx-last-look',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '金融计量与高频统计',
    [
      [
        '基础推断',
        [
          [
            '在金融数据上做回归',
            [
              'OLS 与稳健标准误|ols-robust-se',
              'Newey–West HAC|newey-west',
              '聚类标准误|clustered-se',
              'Bootstrap 在金融|bootstrap-finance',
              '面板与固定效应|panel-fe-finance',
              '工具变量在金融|iv-finance',
              '事件研究法|event-study',
              'Stambaugh 偏差|stambaugh-bias',
              '长期收益与重叠观测|long-horizon-overlap',
              'GMM 与资产定价检验|gmm-asset-pricing',
              'HJ 距离检验|hj-distance-test',
              '因子模型的贝叶斯比较|bayesian-factor-comparison',
              '收缩与实证贝叶斯|shrinkage-empirical-bayes',
              '稳健回归与离群|robust-regression-outliers',
              '分位回归|quantile-regression-finance',
            ],
          ],
        ],
      ],
      [
        '时间序列续',
        [
          [
            '多元与长记忆',
            [
              'VAR 与脉冲响应|var-irf',
              'Granger 因果|granger-causality',
              '状态空间与 Kalman 平滑|state-space-kalman-smoother',
              'Chow 断点检验|chow-test',
              'ARFIMA 与长记忆|arfima-long-memory',
              'QLIKE 与波动预测评估|qlike-vol-forecast-eval',
              'BEKK 多元 GARCH|bekk-mgarch',
              '波动率因子模型|vol-factor-model',
              '日内季节性|intraday-seasonality',
              '已实现半方差与符号跳跃|realized-semivariance',
            ],
          ],
        ],
      ],
      [
        '高频计量',
        [
          [
            '噪声下的估计',
            [
              '预平均|pre-averaging',
              '双尺度 TSRV|tsrv',
              '多元已实现协方差|realized-covariance',
              '已实现 beta|realized-beta',
              '刷新时间与异步|refresh-time',
              '噪声方差估计|noise-variance-estimate',
              '跳跃与新闻事件|jumps-news',
              '点过程与久期建模|point-process-durations',
              '交易时间对成交量时间|business-time-clock',
              '高频预测的损失函数|hf-forecast-loss',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '奇异期权与非股权衍生',
    [
      [
        '奇异与结构',
        [
          [
            '路径依赖与多资产',
            [
              '亚式期权|asian-options',
              '回望期权|lookback-options',
              '数字与二元|digital-options',
              '障碍期权解析解|barrier-analytics',
              '篮子与彩虹|basket-rainbow',
              'quanto|quanto-options',
              '远期起始与 cliquet|cliquet-forward-start',
              '自动赎回 autocallable|autocallable',
              'Gamma swap|gamma-swap',
              '相关互换|correlation-swap',
              '复合与选择者期权|compound-chooser',
              '期权组合策略|option-strategies',
              '权证与雇员期权|warrants-eso',
              '结构性票据拆解|structured-notes',
            ],
          ],
        ],
      ],
      [
        '希腊与模型风险',
        [
          [
            '敏感度怎么算',
            [
              '伴随算法微分 AAD|aad-adjoint',
              '路径导数与似然比|pathwise-likelihood-ratio',
              'Vega 桶与曲面风险|vega-buckets',
              '隐波曲面动态与 PCA|iv-surface-dynamics',
              'gamma-theta 权衡|gamma-theta-tradeoff',
              '衍生品模型风险|derivative-model-risk',
              '校准正则化|calibration-regularization',
              '全局优化在校准|calibration-global-opt',
              '报价惯例|quoting-conventions',
              '到 XVA 的桥|xva-bridge',
            ],
          ],
        ],
      ],
      [
        'FX 与商品衍生',
        [
          [
            '股权之外',
            [
              'Garman–Kohlhagen|garman-kohlhagen',
              'FX 波动率报价 ATM / RR / BF|fx-vol-quotes',
              'FX 障碍与触碰|fx-barrier-touch',
              '交叉货币互换定价|xccy-swap-pricing',
              '商品期权与季节性|commodity-options-seasonality',
              '天气与电力衍生|weather-power-derivatives',
              '通胀挂钩|inflation-linked',
              'caps / floors / swaptions|caps-floors-swaptions',
              'SABR 对 LMM 校准|sabr-lmm-calib',
              '负利率与移位模型|shifted-lognormal-negative-rates',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '多期组合与策略族',
    [
      [
        '多期与动态',
        [
          [
            '单期之外',
            [
              'Merton 组合：实现与校准|merton-portfolio-implementation',
              '动态规划与再平衡|dynamic-programming-rebalance',
              'Gârleanu–Pedersen 动态交易|garleanu-pedersen',
              '贝叶斯组合与参数不确定|bayesian-portfolio',
              '稳健优化|robust-portfolio-opt',
              '税务感知组合|tax-aware-portfolio',
              '多期风险预算|multi-period-risk-budget',
              '目标日期与下滑路径|target-date-glidepath',
              '杠杆与波动拖累|leverage-volatility-drag',
              '再平衡溢价|rebalancing-premium',
            ],
          ],
        ],
      ],
      [
        '策略族',
        [
          [
            '策略地图',
            [
              '多空股票|long-short-equity',
              '130/30|one-thirty-thirty',
              '事件驱动全景|event-driven-overview',
              '全球宏观|global-macro',
              '固定收益相对价值|fi-relative-value',
              '资本结构套利|capital-structure-arb',
              '波动率卖方与尾部|short-vol-tail',
              '另类风险溢价 ARP|alternative-risk-premia',
              '因子择时的可行性|factor-timing-feasibility',
              '季节性与日历异象|seasonality-anomalies',
              '财报期策略|earnings-season-strategies',
              '指数纳入与被动流|index-inclusion-passive-flow',
              '回购流|buyback-flows',
              'ESG 因子争论|esg-factor-debate',
              '主题与叙事|thematic-narrative',
              '策略生命周期|strategy-lifecycle',
            ],
          ],
        ],
      ],
      [
        '业绩与费用',
        [
          [
            '算清楚赚了多少',
            [
              '业绩费与高水位|performance-fee-hwm',
              '费用对复合收益|fees-compounding',
              '业绩持续性|performance-persistence',
              '基金流与回报追逐|fund-flows-chasing',
              '基准与跟踪误差|benchmark-tracking-error',
              '主动份额|active-share',
              '时间加权对资金加权|twr-vs-mwr',
              '交易层级归因|trade-level-attribution',
              '风险调整比率的抽样误差|ratio-sampling-error',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '交易所、数据与工程',
    [
      [
        '交易所技术',
        [
          [
            '从撮合到清算',
            [
              '撮合引擎架构|matching-engine-arch',
              '订单簿重建|orderbook-reconstruction',
              'FIX 协议|fix-protocol',
              '二进制协议 OUCH / SBE|binary-protocols-ouch-sbe',
              '快照与增量行情|snapshot-incremental-feed',
              '主机托管与微波|colocation-microwave',
              '低延迟工程：内核旁路与 FPGA|low-latency-engineering',
              '时间戳精度与单调时钟|timestamp-precision',
              '交易所故障|exchange-outages',
              '中央对手方与违约瀑布|ccp-default-waterfall',
              '结算周期与失败|settlement-fails',
              '证券借贷与做空成本|securities-lending',
              '回购市场与融资|repo-funding',
              '期权到期流程|options-expiration-process',
            ],
          ],
        ],
      ],
      [
        '数据与研究栈',
        [
          [
            '研究基础设施',
            [
              'CRSP / Compustat 对齐|crsp-compustat',
              'TAQ 数据处理|taq-processing',
              '基本面数据与重述|fundamentals-restatements',
              '一致预期与分析师|analyst-consensus',
              '指数成分历史|index-constituents-history',
              '复权的实现|adjustment-implementation',
              '数据供应商对照|data-vendors',
              '特征商店与研究平台|feature-store-research',
              '回测向量化|backtest-vectorization',
              '研究日志与纪律|research-log-discipline',
              '生产监控与告警|production-monitoring-alerts',
              '实盘容量估计|live-capacity-estimate',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    'A 股与债市制度细节',
    [
      [
        '产品与制度',
        [
          [
            'A 股与债市续',
            [
              '公募量化与指增|cn-public-quant',
              '私募与中性策略|cn-private-neutral',
              '融券与转融通|cn-securities-lending',
              '股指期货基差与对冲成本|cn-index-basis-hedge-cost',
              'ETF 期权市场|cn-etf-options',
              '商品期权与波动率|cn-commodity-options',
              '可转债市场结构|cn-convertible-market',
              '北交所与新三板|cn-bse-neeq',
              '注册制与新股定价|cn-ipo-registration',
              '涨跌停与流动性黑洞|cn-limit-liquidity-hole',
              '交易费用与印花税|cn-fees-stamp-duty',
              '量化监管演进|cn-quant-regulation-history',
              '情绪指标：两融与换手|cn-sentiment-indicators',
              '利率债市场|cn-rates-market',
              '银行间与交易所债市|cn-interbank-exchange-bond',
            ],
          ],
        ],
      ],
    ],
  ],
]

export const quantSupplement = fromOutline(extra)
