import { fromOutline, markAppendix, type Outline } from './schema'
import { econSupplement } from './econ-supplement'
import { econFoundations, econFields } from './econ-foundations'

const outline: Outline[] = [
  [
    '微观：选择与需求',
    [
      [
        '偏好',
        [
          [
            '偏好到效用',
            [
              '偏好、完备与传递|preference-choice',
              '连续性|continuity-preference',
              '单调与局部非饱和|monotonicity-lns',
              '凸偏好|convex-preference',
              '效用函数何时存在|utility-representation',
              '拟线性与位似|quasilinear-homothetic',
              'CES 与 Cobb–Douglas|ces-cobb-douglas',
            ],
          ],
          [
            '需求与对偶',
            [
              '预算集与马歇尔需求|marshallian-demand',
              '内点与角点|interior-corner',
              '间接效用与支出函数|indirect-utility',
              'Roy 恒等式|roy-identity',
              '希克斯需求|hicksian-demand',
              '斯勒茨基方程|slutsky-equation',
              '斯勒茨基对称与负半定|slutsky-symmetry',
              '吉芬与劣等|giffen-inferior',
            ],
          ],
          [
            '加总与跨期',
            [
              '恩格尔曲线|engel-curves',
              '等价变化与补偿变化|ev-cv',
              '显示偏好 WARP|revealed-preference',
              'SARP 与 GARP|sarp-garp',
              '可积性与恢复偏好|integrability',
              'Gorman 加总|gorman-polar',
              '劳动–闲暇|labor-leisure',
              '两期消费|two-period-micro',
            ],
          ],
        ],
      ],
      [
        '风险',
        [
          [
            '期望效用',
            [
              '冯·诺依曼–Morgenstern 期望效用|expected-utility',
              'vNM 公理怎么用|vnm-axioms',
              '阿莱悖论与独立性|allais-independence',
              '风险厌恶与阿罗–普拉特|risk-aversion',
              'CARA / CRRA / DARA|cara-crra',
              '谨慎与 Kimball|prudence-kimball',
              '背景风险|background-risk',
              '均值方差何时等于 EU|mean-variance-eu',
              '两基金分离|two-fund-separation',
            ],
          ],
          [
            '占优、状态与模糊',
            [
              '随机占优|stochastic-dominance',
              '二阶占优与均值保持展开|sosd-mps',
              '三阶占优|third-order-sd',
              '状态依存与或有要求权|state-contingent',
              '资产张成与完全市场|complete-span',
              '风险分担|risk-sharing',
              '状态价格|state-prices',
              '埃尔斯伯格与模糊|ellsberg-ambiguity',
              'Maxmin 期望效用|maxmin-eu',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '微观：厂商与市场',
    [
      [
        '生产',
        [
          [
            '技术与成本',
            [
              '技术、规模报酬与替代|production-technology',
              '自由处置与 Inada|free-disposal-inada',
              '齐次生产与欧拉定理|homothetic-production',
              '替代弹性|elasticity-of-substitution',
              '列昂惕夫技术|leontief-technology',
              '成本最小化|cost-minimization',
              '成本函数性质|cost-fn-properties',
              '谢泼德引理与霍特林引理|shephard-hotelling',
              '条件要素需求|conditional-factor-demand',
              '利润最大化与供给|profit-supply',
              '短长期与沉没成本|sunk-cost',
              '规模报酬对规模经济|returns-vs-economies',
            ],
          ],
        ],
      ],
      [
        '市场结构',
        [
          [
            '价格形成',
            [
              '完全竞争的短期与长期|perfect-competition',
              '长期进入与零利润|long-run-entry',
              '可竞争市场|contestable-markets',
              '垄断与勒纳指数|monopoly-lerner',
              '自然垄断与拉姆齐定价|natural-monopoly-ramsey',
              '价格歧视三级|price-discrimination',
              '两部收费与捆绑|two-part-bundling',
              '高峰负荷定价|peak-load-pricing',
              '古诺数量竞争|cournot-oligopoly',
              '伯川德与产能约束|bertrand-capacity',
              '合谋与卡特尔|collusion-cartel',
              '斯塔克尔伯格与领导者|stackelberg',
              '限制进入定价|limit-pricing',
              '掠夺性定价|predatory-pricing',
              '产品差异与霍特林|hotelling',
              '垄断竞争 Dixit–Stiglitz|monopolistic-dixit-stiglitz',
              '垂直差异|vertical-differentiation',
            ],
          ],
          [
            '平台与锁定',
            [
              '网络外部性|network-externalities',
              '双边市场|two-sided-platform',
              '转换成本|switching-costs',
              '后市场锁定|aftermarket-lockin',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '一般均衡与福利',
    [
      [
        '交换与生产',
        [
          [
            '配置',
            [
              '埃奇沃思盒与帕累托|edgeworth-pareto',
              '契约曲线与 MRS|mrs-contract-curve',
              '瓦尔拉斯均衡|walrasian-equilibrium',
              '瓦尔拉斯定律与计价物|walras-law-numeraire',
              '福利经济学两定理|welfare-theorems',
              '带生产的一般均衡|production-economy-ge',
              '存在性与不动点|ge-existence',
              'Sonnenschein–Mantel–Debreu|smd-excess-demand',
              'Negishi 方法|negishi-map',
              '唯一性与试错|uniqueness-tatonnement',
              '核与埃奇沃思猜想|core-equivalence',
              '不完全市场 GEI|incomplete-markets-gei',
              'Radner 均衡|radner-plans',
              '太阳黑子均衡|sunspot-equilibrium',
            ],
          ],
        ],
      ],
      [
        '市场失灵',
        [
          [
            '外部性与公共',
            [
              '外部性与庇古税|externality-pigou',
              '缺失市场|missing-markets',
              '科斯定理与交易成本|coase-theorem',
              '总量管制与许可证|cap-and-trade',
              '公共物品与免费搭车|public-goods',
              '俱乐部物品|club-goods',
              '公地悲剧|common-pool',
              '林达尔与萨缪尔森条件|lindahl-samuelson',
              '税收归宿|tax-incidence',
              '无谓损失|deadweight-loss',
              '次优理论|theory-of-second-best',
              '拉姆齐商品税|ramsey-commodity-tax',
              'Diamond–Mirrlees 生产效率|diamond-mirrlees',
              '最优所得税|mirrlees-income',
              '阿罗不可能定理|arrow-impossibility',
              '中位选民|median-voter',
              '孔多塞循环|condorcet-cycles',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '博弈与信息',
    [
      [
        '博弈',
        [
          [
            '均衡',
            [
              '优势与反复剔除|dominance-iesds',
              '可理性化|rationalizability',
              '策略式与纳什|nash-equilibrium',
              '混合策略与存在性|mixed-strategy',
              '相关均衡|correlated-equilibrium',
              '扩展式与子博弈完美|subgame-perfect',
              '颤抖手完美|trembling-hand',
              '逆向归纳的限度|backward-induction-limits',
              '重复博弈与无名氏定理|folk-theorem',
              '纳什讨价还价|nash-bargaining',
              '鲁宾斯坦轮流出价|rubinstein-bargaining',
              '廉价交谈|cheap-talk',
              '消耗战|war-of-attrition',
              '不完全信息与贝叶斯纳什|bayesian-nash',
              '完美贝叶斯与信号精炼|perfect-bayesian',
              '直观标准与 D1|intuitive-criterion',
            ],
          ],
        ],
      ],
      [
        '拍卖与匹配',
        [
          [
            '机制',
            [
              '独立私人价值与一价|ipv-first-price',
              '英式与荷式拍卖|english-dutch-auction',
              '二价与维克里|second-price-vickrey',
              '保留价|auction-reserve',
              '收入等价|revenue-equivalence',
              '共同价值与赢家诅咒|winner-curse',
              '全支付拍卖|all-pay-auction',
              '双向拍卖|double-auction',
              'Myerson–Satterthwaite|myerson-satterthwaite',
              'AGV 期望外部性|agv-mechanism',
              'Gale–Shapley 稳定匹配|gale-shapley',
              '策略证明与延迟接受|matching-strategy-proof',
              '多对一匹配|many-to-one-matching',
              '学校选择|school-choice',
            ],
          ],
        ],
      ],
      [
        '信息经济学',
        [
          [
            '逆向选择与信号',
            [
              '逆向选择与柠檬市场|akerlof-lemons',
              '逐级退出|unraveling-lemons',
              '信号发送|spence-signaling',
              '混同与分离|pooling-separating',
              '筛选与信息租金|screening-rent',
              'Rothschild–Stiglitz 崩溃|rs-unraveling',
            ],
          ],
          [
            '道德风险与合同',
            [
              '道德风险与隐藏行动|moral-hazard',
              '霍姆斯特罗姆充足统计|holmstrom-informativeness',
              '有限责任下的激励|limited-liability-mh',
              '委托–代理与激励相容|principal-agent',
              '多任务代理|multitask-holmstrom',
              '职业关注|career-concerns',
              '套牢与专用性|hold-up-gh',
              '不完全合同|incomplete-contracts',
            ],
          ],
          [
            '机制可实施',
            [
              '显示原理|revelation-principle',
              '可实施性与包络|implementability-envelope',
              'VCG 与枢轴|vcg-pivot',
              '迈尔森虚拟价值|myerson-virtual-value',
              'Gibbard–Satterthwaite|gibbard-satterthwaite',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '宏观：核算与增长',
    [
      [
        '核算',
        [
          [
            '国民账户',
            [
              'GDP 三种算法|gdp-accounts',
              'GDP、GNI 与 NNP|gdp-gni-nnp',
              '链式指数与替代偏差|chain-index-bias',
              '储蓄–投资恒等式|saving-investment-id',
              'Feldstein–Horioka|feldstein-horioka',
              '价格指数与实际变量|price-index-real',
              'CPI 与 PCE|cpi-vs-pce',
              '失业与自然率|natural-unemployment',
              '奥肯定律|okun-law',
              '劳动参与|labor-participation',
              '贝弗里奇曲线|beveridge-curve',
            ],
          ],
        ],
      ],
      [
        '增长',
        [
          [
            '资本与技术',
            [
              '索洛模型|solow-growth',
              '索洛残差|solow-residual',
              '黄金律与动态无效率|golden-rule',
              'Ramsey–Cass–Koopmans|ramsey-cass-koopmans',
              'OLG 与动态无效率再访|olg-diamond',
              '内生增长：人力资本与 R&D|endogenous-growth',
              'AK、种类与熊彼特|ak-variety-schumpeter',
              '有偏技术进步|directed-technical-change',
              '制度与增长|institutions-acemoglu',
              '马尔萨斯到现代增长|malthus-to-modern',
              '错配与 TFP|misallocation-tfp',
              '收敛与发展核算|convergence-accounting',
              '增长核算对发展核算|growth-vs-development-acct',
            ],
          ],
        ],
      ],
      [
        '搜寻与投资',
        [
          [
            '摩擦',
            [
              'DMP 搜寻匹配|search-matching-dmp',
              '效率工资|efficiency-wage',
              '托宾 q|tobin-q',
              '投资调整成本|adjustment-cost-invest',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '宏观：货币、周期与政策',
    [
      [
        '货币',
        [
          [
            '需求与通胀',
            [
              '货币层次与支付|money-as-medium',
              '现金先行与货币入效用|cia-miu',
              'Lagos–Wright 新货币主义|lagos-wright',
              '数量方程与货币中性|quantity-theory',
              '费雪方程|fisher-equation',
              '通胀税与铸币税|seigniorage',
              '菲利普斯曲线与预期|phillips-expectations',
              '自然率假说|natural-rate-u-star',
            ],
          ],
        ],
      ],
      [
        '周期',
        [
          [
            '需求与粘性',
            [
              'IS–LM 作为会计|is-lm',
              '跨期消费与欧拉方程|consumption-euler',
              'Hall 随机游走|hall-random-walk',
              '新凯恩斯粘性价格|new-keynesian',
              'Calvo 与 Rotemberg|calvo-rotemberg',
              '菜单成本|menu-cost',
              '粘性工资|sticky-wages',
              'Taylor 交错合同|taylor-contracts',
              '新凯恩斯菲利普斯曲线|nk-phillips',
              '混合 NKPC|hybrid-nkpc',
              '从欧拉到动态 IS|dynamic-is',
              '金融加速器|financial-accelerator',
              '真实经济周期对照|rbc-contrast',
              '卢卡斯批判|lucas-critique',
              '自然利率 r*|r-star',
            ],
          ],
        ],
      ],
      [
        '政策',
        [
          [
            '规则',
            [
              '泰勒规则|taylor-rule',
              '泰勒原理|taylor-principle',
              '时间不一致与承诺|time-inconsistency',
              '权变对承诺|discretion-commitment',
              '神圣巧合|divine-coincidence',
              '最优货币政策|optimal-nk-policy',
              '通胀目标制|inflation-targeting',
              '准备金利息与走廊|ior-corridor-floor',
              '零下限与非常规政策|zlb-unconventional',
              '前瞻指引之谜|fg-puzzle',
              'QE 与前瞻指引|qe-forward-guidance',
              '财政乘数与李嘉图|fiscal-multiplier',
              'Barro 税收平滑|tax-smoothing-barro',
              '物价水平的财政理论|ftpl',
              '自动稳定器|automatic-stabilizers',
            ],
          ],
        ],
      ],
      [
        '开放',
        [
          [
            '两国',
            [
              '开放经济会计与经常账户|open-ca',
              '双赤字|twin-deficits',
              '购买力平价与一价定律|ppp-loop',
              '巴拉萨–萨缪尔森|balassa-samuelson',
              'UIP 与 CIP|uip-and-cip',
              '马歇尔–勒纳|marshall-lerner',
              'J 曲线|j-curve',
              '蒙代尔–弗莱明|mundell-fleming',
              '多恩布什超调|dornbusch-overshoot',
              '汇率制度|exchange-regimes',
              '货币同盟|monetary-union',
              '不可能三角|impossible-trinity',
              '突然停止|sudden-stop',
              '原罪与货币错配|original-sin',
              'Backus–Smith|backus-smith',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '货币银行与公司金融',
    [
      [
        '中介',
        [
          [
            '银行为何存在',
            [
              '金融中介的功能|why-intermediaries',
              '信贷配给 Stiglitz–Weiss|stiglitz-weiss',
              'Diamond–Dybvig 挤兑|diamond-dybvig',
              '传染与系统性|bank-contagion',
              '火线出售|fire-sales',
              '存款保险|deposit-insurance',
              '准备金、乘数与内生货币|reserve-endogenous-money',
              '支付系统与 RTGS|rtgs-payments',
              '最后贷款人|lender-of-last-resort',
              '太大而不能倒|too-big-to-fail',
              '资本监管与巴塞尔|basel-capital',
              'LCR 与 NSFR|lcr-nsfr',
              '影子银行|shadow-banking',
            ],
          ],
        ],
      ],
      [
        '公司',
        [
          [
            '资本结构',
            [
              '莫迪利亚尼–米勒|modigliani-miller',
              'MM 与公司税盾|mm-corporate-tax',
              '权衡理论与破产成本|tradeoff-capital',
              '直接与间接破产成本|bankruptcy-direct-indirect',
              '债务积压|debt-overhang',
              '风险转移|risk-shifting',
              '优序融资|pecking-order',
              '自由现金流代理成本|agency-free-cash-flow',
              '股利与自由现金流|dividend-fcf',
              '股利信号|dividend-signaling',
              '股票回购|share-repurchase',
              '可转债|convertible-security',
              'IPO 抑价作为理论|ipo-underpricing-theory',
              '控制权与治理|control-rights',
              '实物期权|real-options-theory',
            ],
          ],
        ],
      ],
      [
        '资产定价理论',
        [
          [
            '折现因子与谜题',
            [
              '有效市场假说|emh',
              'EMH 三档|emh-three-forms',
              '随机折现因子|stochastic-discount-factor',
              'Hansen–Jagannathan 界|hansen-jagannathan',
              '股权溢价之谜|equity-premium-puzzle',
              '无风险利率之谜|risk-free-rate-puzzle',
              '递归效用 Epstein–Zin|recursive-utility',
              '习惯形成|habit-formation',
              '长期风险|long-run-risk',
              '稀有灾难|rare-disasters',
            ],
          ],
          [
            '均衡定价到限价簿',
            [
              '消费 CAPM|ccapm',
              'Merton ICAPM|icapm-merton',
              'CAPM 作为均衡陈述|capm-theory',
              'APT 作为均衡|apt-as-equilibrium',
              '期限结构预期假说|eh-term-structure',
              '流动性溢价与栖息地|liquidity-habitat-term',
              '仿射期限结构|affine-yield-curve',
              '到限价簿：理论在此停|to-limit-order-book',
            ],
          ],
        ],
      ],
    ],
  ],
]

const papers: Outline[] = [
  [
    '经典论文对照',
    [
      [
        '微观与均衡',
        [
          [
            '文献',
            [
              'Arrow–Debreu 或有商品|arrow-debreu-paper',
              'Debreu 价值理论|debreu-theory-of-value',
              'Akerlof 柠檬市场原文|akerlof-1970',
              'Spence 市场信号原文|spence-1973',
              'Rothschild–Stiglitz 保险筛选|rothschild-stiglitz',
              'Nash 均衡原文|nash-1950-paper',
              'Vickrey 反拍卖|vickrey-1961-paper',
              'Myerson 最优拍卖|myerson-1981-paper',
              'Gale–Shapley 原文|gale-shapley-paper',
              'Mirrlees 最优税原文|mirrlees-1971-paper',
              'Myerson–Satterthwaite 原文|ms-1983-paper',
            ],
          ],
        ],
      ],
      [
        '宏观与货币',
        [
          [
            '文献',
            [
              'Keynes《通论》问题设定|keynes-general-theory',
              'Friedman 货币数量重述|friedman-quantity',
              'Kydland–Prescott 规则优于权变|kydland-prescott',
              'Woodford 利息与价格|woodford-interest',
              'Diamond–Dybvig 原文|diamond-dybvig-paper',
              'Solow 1956 增长原文|solow-1956-paper',
              'Lucas 政策评价原文|lucas-1976-paper',
              'Mundell 1963 资本流动|mundell-1963-paper',
              'Mortensen–Pissarides 搜寻|mp-1994-paper',
              'Mehra–Prescott 股权溢价原文|mehra-prescott-paper',
            ],
          ],
        ],
      ],
    ],
  ],
]

const [behavior, infoLiq, dynMacro, corpAdv, econometrics, trade] = econSupplement

export const econTree = [
  ...fromOutline([
    econFoundations,
    ...outline.slice(0, 4),
    econometrics,
    ...outline.slice(4, 6),
    dynMacro,
    outline[6],
    corpAdv,
    behavior,
    infoLiq,
    trade,
    ...econFields,
  ]),
  ...markAppendix(fromOutline(papers)),
]
