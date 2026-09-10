import type { Outline } from './schema'

/**
 * 金融补层（2026-09 审视）：接在主干「到限价簿」之后、附录之前。
 * 微观、宏观、货币银行与公司金融的理论续篇；不重写 LOB / CAPM 实证。叶子均为待撰写。
 */
export const econSupplement: Outline[] = [
  [
    '跨期、行为与家庭金融',
    [
      [
        '跨期选择',
        [
          [
            '时间上的偏好',
            [
              '指数贴现与时间一致|exponential-discounting',
              '双曲贴现与现时偏差|hyperbolic-present-bias',
              '承诺装置|commitment-devices',
              '生命周期假说|life-cycle-hypothesis',
              '永久收入假说|permanent-income',
              '预防性储蓄与缓冲存量|precautionary-buffer-stock',
              '流动性约束与过度敏感|liquidity-constraints-excess-sensitivity',
              '遗赠动机|bequest-motive',
              '年金之谜|annuity-puzzle',
              '住房：资产与消费|housing-asset-consumption',
              '家庭组合与参与之谜|household-portfolio-participation',
              '金融素养与默认选项|financial-literacy-defaults',
            ],
          ],
        ],
      ],
      [
        '行为经济学',
        [
          [
            '偏离理性的规律',
            [
              '前景理论|prospect-theory',
              '损失厌恶与参考点|loss-aversion-reference',
              '概率加权|probability-weighting',
              '心理账户|mental-accounting',
              '框架效应|framing-effects',
              '禀赋效应|endowment-effect',
              '过度自信|overconfidence',
              '启发式与偏差|heuristics-biases',
              '有限理性与满意|bounded-rationality-satisficing',
              '社会偏好与公平|social-preferences',
              '助推与选择架构|nudge-choice-architecture',
              '显著性与注意力|salience-attention',
              '信念更新的偏差|belief-updating-biases',
              '实验经济学方法|experimental-methods',
            ],
          ],
        ],
      ],
      [
        '行为金融理论',
        [
          [
            '噪声、情绪与泡沫',
            [
              '噪声交易者风险 DSSW|noise-trader-risk',
              '套利限制 Shleifer–Vishny|limits-to-arbitrage',
              'BSV 情绪模型|bsv-sentiment-model',
              'DHS 过度自信模型|dhs-overconfidence',
              'Hong–Stein 信息扩散|hong-stein-diffusion',
              '前景理论资产定价|prospect-theory-asset-pricing',
              '处置效应|disposition-effect',
              'Harrison–Kreps 异质信念|harrison-kreps-speculation',
              'Scheinkman–Xiong 换手与泡沫|scheinkman-xiong',
              '理性泡沫|rational-bubbles',
              'Abreu–Brunnermeier 同步失败|abreu-brunnermeier',
              '羊群与信息级联|herding-cascades',
              '有限注意力与公告漂移|limited-attention-pead',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '信息、流动性与无套利',
    [
      [
        '信息与价格',
        [
          [
            '价格能揭示多少',
            [
              'Grossman–Stiglitz 悖论|grossman-stiglitz',
              'Hellwig 竞争性 REE|hellwig-ree',
              'Admati 多资产 REE|admati-multi-asset-ree',
              'REE 的存在与揭示|ree-revelation',
              '信息获取与互补|information-acquisition',
              '美人竞赛与高阶信念|beauty-contest-higher-order',
              'Morris–Shin 公共信息|morris-shin-public-info',
              'Kyle 在均衡框架中的位置|kyle-in-equilibrium',
              '不对称信息与流动性理论|asymmetric-info-liquidity',
              '披露理论|disclosure-theory',
              '信息中介|information-intermediaries',
              '微观结构与资产定价的接口|microstructure-asset-pricing-interface',
            ],
          ],
        ],
      ],
      [
        '无套利与鞅',
        [
          [
            '定价的数学骨架',
            [
              '一价定律与无套利|law-of-one-price',
              '资产定价基本定理|ftap',
              '等价鞅测度|equivalent-martingale-measure',
              '动态完备与复制|dynamic-completeness',
              'Harrison–Kreps 鞅定价|harrison-kreps-martingale',
              '状态价格到 SDF|state-price-sdf-bridge',
              'Lucas 树|lucas-tree',
              'Campbell–Shiller 分解|campbell-shiller-decomposition',
              '现值恒等式与可预测性|present-value-identity',
              'Shiller 过度波动|shiller-excess-volatility',
              '股利-价格比预测|dp-ratio-predictability',
              '风险的期限结构|term-structure-of-risk',
              '生产基础的资产定价|production-based-asset-pricing',
            ],
          ],
          [
            '摩擦与中介',
            [
              '中介资产定价 He–Krishnamurthy|intermediary-asset-pricing',
              '市场与融资流动性 Brunnermeier–Pedersen|market-funding-liquidity',
              '杠杆周期 Geanakoplos|leverage-cycle',
              '资金约束与 CAPM 偏离|funding-constraints-capm',
              '安全资产与便利收益|safe-asset-convenience',
              '需求系统资产定价|demand-system-asset-pricing',
            ],
          ],
        ],
      ],
      [
        '连续时间与组合',
        [
          [
            '从伊藤到均衡',
            [
              '布朗运动与伊藤引理|brownian-ito',
              '连续时间预算约束|continuous-time-budget',
              'Merton 组合问题|merton-portfolio-theory',
              '对冲需求的来源|hedging-demand-origin',
              '连续时间 CAPM|continuous-time-capm',
              'CIR 一般均衡利率|cir-general-equilibrium',
              'Black–Scholes 作为均衡结果|bs-as-equilibrium',
              '不完全市场的定价界|incomplete-market-pricing-bounds',
              '交易成本下的组合 Constantinides|transaction-cost-portfolio-theory',
              '多期消费组合|multi-period-consumption-portfolio',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '动态宏观与异质性',
    [
      [
        '动态方法',
        [
          [
            '求解与识别',
            [
              '贝尔曼方程|bellman-dp',
              '值函数迭代与策略迭代|vfi-pi',
              '欧拉方程与横截条件|euler-transversality',
              '对数线性化|log-linearization',
              'Blanchard–Kahn 条件|blanchard-kahn',
              '扰动法与高阶|perturbation-methods',
              '投影法|projection-methods',
              '校准与矩匹配|calibration-moments',
              'DSGE 贝叶斯估计|dsge-bayesian-estimation',
              'HP 滤波与周期分离|hp-filter-cycles',
              '商业周期事实|business-cycle-facts',
              'SVAR 识别|svar-identification',
              '局部投影|local-projections',
              '叙事与高频识别|narrative-hf-identification',
            ],
          ],
        ],
      ],
      [
        '异质性',
        [
          [
            '代表性个体之外',
            [
              'Bewley–Huggett–Aiyagari|bewley-aiyagari',
              'Krusell–Smith|krusell-smith',
              '财富分布与帕累托尾|wealth-distribution-pareto',
              'HANK|hank',
              '边际消费倾向异质|heterogeneous-mpc',
              '财政刺激与 HANK|fiscal-hank',
              '不平等与 r > g|inequality-r-g',
              '代际流动与人力资本|intergenerational-mobility',
              '劳动收入风险与保险|income-risk-insurance',
              '失业保险与道德风险|ui-moral-hazard',
              '家庭债务与周期|household-debt-cycles',
              '企业异质性与投资|firm-heterogeneity-investment',
            ],
          ],
        ],
      ],
      [
        '预期与信息摩擦',
        [
          [
            '人们怎么形成预期',
            [
              '适应性预期与学习|adaptive-learning',
              '粘性信息|sticky-information',
              '理性疏忽|rational-inattention',
              '诊断性预期|diagnostic-expectations',
              '新闻冲击与信心|news-shocks',
              '不确定性冲击|uncertainty-shocks',
              '调查预期数据|survey-expectations',
              '预期管理与沟通|expectations-management',
              '通胀预期锚定|inflation-anchoring',
              '有限理性 NK|bounded-rationality-nk',
            ],
          ],
        ],
      ],
      [
        '金融摩擦宏观',
        [
          [
            '信贷、主权与央行',
            [
              'Kiyotaki–Moore 信贷周期|kiyotaki-moore',
              'Gertler–Kiyotaki 银行中介|gertler-kiyotaki',
              '抵押约束与房价|collateral-house-prices',
              '债务通缩|debt-deflation',
              'Eggertsson–Krugman 去杠杆|eggertsson-krugman',
              '宏观审慎政策|macroprudential',
              '全球金融周期|global-financial-cycle',
              'Eaton–Gersovitz 主权违约|eaton-gersovitz',
              'Arellano 主权债务|arellano-sovereign',
              '债务可持续与 r − g|debt-sustainability',
              '财政-货币主导|fiscal-monetary-dominance',
              '银行危机史与模式|banking-crises-history',
              '影子银行挤兑|shadow-run',
              '央行资产负债表|central-bank-balance-sheet',
              '回购市场与货币市场基金|repo-mmf',
              'CBDC|cbdc',
              '美元主导与国际货币|dollar-dominance',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '估值、资本结构与治理',
    [
      [
        '资本预算与估值',
        [
          [
            '把项目折成今天的钱',
            [
              'NPV 与 IRR 的陷阱|npv-irr-pitfalls',
              '资本成本与 WACC|wacc-cost-of-capital',
              'CAPM 用于项目贴现|capm-project-discount',
              'APV|apv',
              '自由现金流预测|fcf-forecasting',
              'DCF 与终值|dcf-terminal-value',
              '可比与乘数|multiples-valuation',
              'EVA 与剩余收益|eva-residual-income',
              '实物期权估值|real-options-valuation',
              '租赁对购买|lease-vs-buy',
              '营运资本|working-capital',
              '通胀与估值|inflation-valuation',
              '国家风险溢价|country-risk-premium',
            ],
          ],
        ],
      ],
      [
        '资本结构动态',
        [
          [
            '债务的时间维度',
            [
              'Leland 动态资本结构|leland-model',
              '动态权衡与调整|dynamic-tradeoff',
              '市场择时理论|market-timing-theory',
              'Miller 均衡与个人税|miller-equilibrium',
              '债务期限与展期|debt-maturity-rollover',
              '契约条款|covenants',
              '银行债对公开债|bank-vs-public-debt',
              '关系借贷|relationship-lending',
              '信用评级机构|credit-rating-agencies',
              '证券化与发起-分销|securitization-otd',
              '财务困境与重组|financial-distress-restructuring',
              '第 11 章破产|chapter-11',
              '债权人协调与空壳债权人|creditor-coordination-empty-creditor',
            ],
          ],
        ],
      ],
      [
        '治理与控制',
        [
          [
            '谁说了算',
            [
              '董事会与监督|boards-monitoring',
              '高管薪酬与激励|executive-compensation',
              '股东激进主义|shareholder-activism',
              '接管与免费搭车 Grossman–Hart|takeover-free-rider',
              '反收购防御|takeover-defenses',
              '并购协同与支付方式|ma-synergy-payment',
              '杠杆收购与私募股权|lbo-private-equity',
              '风险投资：分阶段与合同|venture-capital-staging',
              '双层股权|dual-class-shares',
              '法与金融 LLSV|law-and-finance',
              '内部人交易理论|insider-trading-theory',
              '公司对冲的理由|corporate-hedging-rationale',
              '现金持有|cash-holdings',
              'ESG 理论|esg-theory',
              '家族企业与金字塔|family-firms-pyramids',
              '公司金融的实证识别|corporate-finance-identification',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '计量经济学',
    [
      [
        '识别与回归',
        [
          [
            '从相关到因果',
            [
              '潜在结果|potential-outcomes',
              'OLS 与 Gauss–Markov|ols-gauss-markov',
              '异方差与 HAC|hac-heteroskedasticity',
              '聚类标准误|clustered-se',
              '遗漏变量与测量误差|ovb-measurement-error',
              '工具变量与弱工具|iv-weak-instruments',
              '2SLS 与过度识别|2sls-overid',
              'GMM|gmm-econ',
            ],
          ],
          [
            '准实验',
            [
              '双重差分|difference-in-differences',
              '交错 DiD|staggered-did',
              '断点回归|rdd',
              '合成控制|synthetic-control',
              '匹配与倾向得分|matching-propensity',
              '事件研究|event-study-econ',
            ],
          ],
        ],
      ],
      [
        '结构与预测',
        [
          [
            '模型驱动的估计',
            [
              '结构对约化形式|structural-vs-reduced',
              '离散选择 logit / probit|discrete-choice',
              'BLP 需求估计|blp-demand',
              '生产函数估计|production-function-estimation',
              '拍卖的结构估计|auction-structural-estimation',
              '预测评估 Diebold–Mariano|forecast-evaluation-dm',
              '极大似然与贝叶斯|mle-bayesian-econ',
            ],
          ],
        ],
      ],
      [
        '面板、推断与稳健',
        [
          [
            '重复观测',
            [
              '固定效应与随机效应|fe-re',
              '动态面板 Arellano–Bond|arellano-bond',
              '异质处理效应|heterogeneous-effects',
              'Bootstrap|bootstrap-econ',
              '多重检验|multiple-testing-econ',
              '稳健、聚类与多重检验|inference-robust-cluster',
              '机器学习与因果|ml-causal-econ',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '贸易与国际金融',
    [
      [
        '贸易理论',
        [
          [
            '为什么交换',
            [
              '李嘉图比较优势|ricardian-comparative-advantage',
              'Heckscher–Ohlin|heckscher-ohlin',
              'Stolper–Samuelson|stolper-samuelson',
              'Krugman 新贸易理论|krugman-new-trade',
              'Melitz 异质企业|melitz-heterogeneous-firms',
              '引力模型|gravity-model',
              'Eaton–Kortum|eaton-kortum',
              '关税与最优关税|tariffs-optimal',
              '贸易协定与 WTO|trade-agreements',
              '全球价值链|global-value-chains',
              'China shock|china-shock',
              '离岸外包与任务|offshoring-tasks',
              '贸易与增长|trade-growth',
            ],
          ],
        ],
      ],
      [
        '国际金融续',
        [
          [
            '资本为何这样流',
            [
              'Lucas 悖论|lucas-paradox',
              '全球失衡|global-imbalances',
              '外汇干预与储备|fx-intervention-reserves',
              '汇率脱节之谜|exchange-rate-disconnect',
              '远期溢价之谜|forward-premium-puzzle',
              '国际风险分担|international-risk-sharing',
              '货币危机模型|currency-crisis-models',
              '三代危机模型|crisis-generations',
              '资本管制|capital-controls',
              '主权财富基金|sovereign-wealth-funds',
              '欧元区危机|eurozone-crisis',
              '美元融资与互换线|dollar-swap-lines',
            ],
          ],
        ],
      ],
    ],
  ],
]
