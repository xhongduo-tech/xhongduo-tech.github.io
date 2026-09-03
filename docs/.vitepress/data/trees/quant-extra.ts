import { fromOutline, type Outline } from './schema'

const extra: Outline[] = [
  [
    '市场与产品',
    [
      [
        '中国市场微观结构',
        [
          [
            '交易制度',
            [
              'T+1 与回转交易约束|cn-tplus1',
              '涨跌停与集合竞价|cn-limit-auction',
              '融券与做空约束|cn-short-constraint',
              '北向资金与沪深港通|stock-connect',
              '科创板 / 创业板制度差异|star-chinext',
              '转债与正股联立|cb-equity-link',
              'ETF 申赎与 IOPV|cn-etf-creation',
              '融资融券保证金与强平|cn-margin-trading',
              '大宗交易与折价|cn-block-trade',
              'ST / 退市风险|cn-st-delist',
              '停牌复牌与信息窗口|cn-halt-resume',
              '可转债打新与赎回|cb-call-put',
              '股指期货保证金与贴水|cn-index-fut-basis',
              '国债期货 CTD 与基差|cn-tf-ctd',
              '商品期货夜盘|cn-night-session',
            ],
          ],
        ],
      ],
      [
        '期货与基差',
        [
          [
            '期限结构',
            [
              'Contango / Backwardation|contango-backwardation',
              '展期收益|roll-yield',
              '期现套利|cash-and-carry',
              '跨期套利|calendar-spread-arb',
              '仓单与交割|futures-delivery',
              '保证金与盯市|futures-margin',
              '最便宜可交割 CTD|ctd-cheapest',
              '隐含回购利率|implied-repo',
              '展期成交量与持仓|roll-volume-oi',
              '库存报告与曲线|inventory-curve',
              '季节性展期|seasonal-roll',
              '跨品种价差|intercommodity-spread',
            ],
          ],
        ],
      ],
      [
        '外汇与商品',
        [
          [
            '定价',
            [
              '利率平价|irp',
              '购买力平价作为锚|ppp-anchor',
              '交叉汇率三角|fx-triangle',
              '商品便利收益|convenience-yield',
              '库存与曲线|commodity-inventory',
              'NDF 与不可兑换货币|ndf',
              '交叉货币基差|xccy-basis',
              '商品展期 alpha|commodity-roll-alpha',
              '能源裂解价差|crack-spread',
              '农产品天气升水|ag-weather-premium',
            ],
          ],
        ],
      ],
      [
        '期权做市细节',
        [
          [
            '簿与希腊',
            [
              '隐波插值 arbitrage-free|arb-free-iv',
              '蝶式与日历无套利|butterfly-calendar-arb',
              'Delta 对冲频率|delta-hedge-freq',
              'Pin risk|pin-risk',
              '隔夜跳空对冲|overnight-gap-hedge',
              '隐波曲面 SVI / SSVI|svi-ssvi',
              'Vanna / Volga 对冲|vanna-volga',
              'Charm / Color 高阶希腊|higher-greeks',
              '离散股息对美式期权|discrete-dividend-am',
              '障碍期权监控频率|barrier-monitoring',
            ],
          ],
        ],
      ],
      [
        '高频因子',
        [
          [
            '信号',
            [
              '订单流毒性|ofi-toxicity',
              '队列不平衡因子|queue-imbalance-alpha',
              '成交到达强度|hawkes-trades',
              '微观价格|microprice',
              '深度加权中间价|depth-mid',
              'Hasbrouck 信息份额|hasbrouck-is',
              'Kyle lambda 冲击系数|kyle-lambda',
              'Amihud 非流动性|amihud-illiquidity',
              'Pastor-Stambaugh 流动性|pastor-stambaugh',
              'Corwin-Schultz 价差估计|corwin-schultz',
              'Roll 有效价差估计|roll-effective-spread',
              'VPIN 实时毒性|vpin-realtime',
              '订单流不平衡 OFI|ofi-cont',
              '撤单速率|cancel-rate',
              '冰山探测|iceberg-detection',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '因子与资产定价续',
    [
      [
        '因子动物园',
        [
          [
            '经典扩展',
            [
              'Hou-Xue-Zhang q-因子|q-factor-hxz',
              'Stambaugh-Yuan 错误定价|sy-mispricing',
              'Fama-French 六因子|ff6',
              'Quality Minus Junk|qmj',
              'Betting Against Beta 原文|frazzini-bab',
              'HML Devil|hml-devil',
              '时间序列动量|tsmom',
              '跨资产价值与动量|value-momentum-everywhere',
              'Carry 跨资产|carry-everywhere',
              '盈利意外 PEAD / SUE|pead-sue',
              '应计 anomalous accruals|accruals-anomaly',
              '净股票发行|net-issuance',
              '资产增长异常|asset-growth',
              '毛利率 Novy-Marx|gross-profitability',
              '投资因子 CMA|cma-investment',
            ],
          ],
          [
            '条件与宏观',
            [
              'Fama-MacBeth 回归|fama-macbeth',
              'Giglio-Xiu 潜因子|giglio-xiu',
              'IPCA 工具主成分|ipca',
              '风险溢价时变|time-varying-rp',
              '宏观因子：增长、通胀、流动性|macro-factor-set',
              '收益率曲线因子|yield-curve-factors',
              '波动率风险溢价|variance-risk-premium',
              '偏度风险溢价|skew-risk-premium',
              '尾部风险因子|tail-risk-factor',
              '资金流动性因子|funding-liquidity',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '衍生品数值与模型',
    [
      [
        '数值方法',
        [
          [
            '定价引擎',
            [
              'Carr-Madan FFT 定价|carr-madan',
              'COS 方法|cos-method',
              'Longstaff-Schwartz 美式 MC|longstaff-schwartz',
              '最小二乘蒙特卡洛 LSM|lsm-american',
              'PDE Crank-Nicolson|crank-nicolson',
              'ADI 高维 PDE|adi-pde',
              '树图与美式提前行权|tree-early-exercise',
              '控制变量对欧式|cv-european',
              '重要性采样对障碍|is-barrier',
              '拟随机数 Sobol|sobol-qmc',
            ],
          ],
        ],
      ],
      [
        '随机波动与跳跃',
        [
          [
            '模型',
            [
              'Heston 特征函数|heston-cf',
              'Bates 跳扩散|bates',
              'Merton 跳扩散|merton-jump',
              'Variance Gamma|variance-gamma',
              'CGMY|cgmy',
              'SABR 校准|sabr-calib',
              'Local-stochastic vol|local-stoch-vol',
              'Bergomi 方差曲线|bergomi',
              'Rough volatility|rough-vol',
              'Gatheral 无套利曲面|gatheral-arb-free',
              'Dupire 局部波动校准|dupire-calib',
              'VIX 期货定价|vix-futures',
              '方差互换复制|var-swap-replication',
              '波动率套利|vol-arb',
              'Dispersion 交易|dispersion-trade',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '利率信用与曲线',
    [
      [
        '收益率曲线',
        [
          [
            '拟合',
            [
              'Nelson-Siegel|nelson-siegel',
              'Svensson 扩展|svensson',
              '曲线 PCA：水平、斜率、曲率|curve-pca',
              '蝶式交易|curve-butterfly',
              '互换价差|swap-spread',
              '基差互换|basis-swap',
              'SOFR 过渡与后备|sofr-transition',
              'CMS 凸性调整|cms-convexity',
              '百慕大互换期权|bermudan-swaption',
              '关键期限 DV01|key-tenor-dv01',
            ],
          ],
        ],
      ],
      [
        '信用',
        [
          [
            '结构与简化',
            [
              'Merton 结构模型|merton-structural',
              'Black-Cox 首达|black-cox',
              '简化形式强度模型|reduced-form-intensity',
              'CDS 定价与升水|cds-pricing',
              'CDS-债券基差|cds-bond-basis',
              '指数 vs 单名|cdx-vs-single',
              '高斯 Copula 与批判|gaussian-copula-cdo',
              '回收率假设|recovery-assumption',
              '评级迁移矩阵|rating-migration',
              '主权 CDS|sovereign-cds',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '组合优化进阶',
    [
      [
        '估计与约束',
        [
          [
            '协方差',
            [
              'Ledoit-Wolf 收缩|ledoit-wolf',
              '图形 Lasso 精度矩阵|glasso',
              '因子协方差 vs 样本|factor-vs-sample-cov',
              'EWMA 协方差|ewma-cov',
              'DCC 动态相关入组合|dcc-portfolio',
            ],
          ],
          [
            '配置方法',
            [
              '层次风险平价 HRP|hrp',
              '等风险贡献 ERC|erc',
              'CVaR 优化|cvar-opt',
              '交易成本感知优化|tcost-opt',
              '换手惩罚|turnover-penalty',
              '多空与多头约束|long-short-constraints',
              '行业/国家中性组合|industry-country-neutral',
              'Black-Litterman 观点|bl-views',
              '风险预算再平衡|rb-rebalance',
              '波动目标 vol targeting|vol-targeting',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '执行算法续',
    [
      [
        '最优执行',
        [
          [
            '模型',
            [
              'Obizhaeva-Wang|obizhaeva-wang',
              'Gatheral-Schied 无漂移|gatheral-schied',
              'Almgren 临时冲击校准|almgren-temp-calib',
              '到达价格算法|arrival-price',
              '暗池路由|dark-pool-routing',
              '参与率封顶|participation-cap',
              '开盘/收盘竞价执行|auction-execution',
              '冰山与隐藏单|iceberg-algo',
              '智能订单路由 SOR|smart-order-routing',
              '成交后分析 TCA|tca',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '统计套利续',
    [
      [
        '配对交易',
        [
          [
            '方法',
            [
              'Gatev-Goetzmann-Rouwenhorst|ggr-pairs',
              'Avellaneda-Lee 统计套利|avellaneda-lee',
              'Copula 配对|copula-pairs',
              'Kalman 价差滤波|kalman-spread',
              '协整秩选择|coint-rank',
              '时变对冲比断裂|hedge-ratio-break',
              '篮子 PCA 残差|pca-residual-basket',
              '行业内相对价值|sector-rv',
              'ADR / 两地上市溢价|adr-premium',
              'ETF 创建赎回套利|etf-create-redeem',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '回测方法学续',
    [
      [
        '金融机器学习',
        [
          [
            '标签与验证',
            [
              'Triple Barrier 标签|triple-barrier',
              'Meta-labeling|meta-labeling',
              '分数阶差分|fracdiff',
              '信息驱动 bar：tick/volume/dollar|imbalance-bars',
              'Combinatorial Purged CV|cpcv-lopez',
              'Embargo 与 purge|purge-embargo',
              'Deflated Sharpe 原文|bailey-dsr',
              '概率夏普 PSR|probabilistic-sharpe',
              '过拟合次数与试错|backtest-overfitting',
              '点-in-time 基本面对齐|pit-alignment',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '风险计量续',
    [
      [
        '市场与流动性',
        [
          [
            '度量',
            [
              'Cornish-Fisher VaR|cornish-fisher-var',
              'Filtered Historical Simulation|fhs-var',
              'Copula VaR|copula-var',
              '压力 VaR 与 FRTB|stressed-var-frtb',
              '流动性调整 VaR|liquidity-adjusted-var',
              '成分 ES / Euler 分配|component-es',
              '保证金模型 SPAN / SIMM|span-simm',
              '缺口风险 gap risk|gap-risk',
              '拥挤度与清盘螺旋|crowding-spiral',
              '对手方 CVA 要点|cva-lite',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '宏观与 CTA',
    [
      [
        '趋势与 Carry',
        [
          [
            '策略',
            [
              '时间序列动量 CTA|cta-tsmom',
              '期限结构 Carry|ts-carry',
              '突破与均线|breakout-ma',
              '波动缩放仓位|vol-scale-cta',
              '跨资产趋势相关|xasset-trend-corr',
              '危机 alpha|crisis-alpha',
              '管理期货风险平价|mfd-risk-parity',
              '商品趋势 vs 展期|trend-vs-roll',
              '外汇 Carry 与动量|fx-carry-mom',
              '债券趋势|bond-trend',
            ],
          ],
        ],
      ],
    ],
  ],
]

export const quantExtra = fromOutline(extra)
