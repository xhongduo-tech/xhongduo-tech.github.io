import { fromOutline, type Outline } from './schema'

/** 前沿补层（2025H2–2026 审计）：LLM×量化 / 期权流与 GEX / 中国新规与产品 / ML 资产定价 / 深度微观结构与仿真 / 深度对冲 / 数据栈 / 零售流与预测市场 / 稳健与治理 */
const extra: Outline[] = [
  [
    'LLM × 量化',
    [
      [
        '因子挖掘智能体',
        [
          [
            '闭环系统',
            [
              'Alpha-GPT 人类启发式挖掘|alpha-gpt-mining',
              'Hubble：AST 沙箱与三重门|hubble-alpha-agent',
              'XAlpha：假设到代码|xalpha-hypothesis-code',
              'AlphaSchema 领域本体|alphaschema-ontology',
              'QuantGPT 与 BRAIN 自动化|quantgpt-brain',
              '101 Formulaic Alphas|kakushadze-101-alphas',
            ],
          ],
        ],
      ],
      [
        '金融基础模型',
        [
          [
            '模型与基准',
            [
              'Kronos K 线基础模型|kronos-kline-fm',
              'FinGPT|fingpt',
              'BloombergGPT|bloomberggpt',
              'CTBench 中文金融基准|ctbench',
              'EDINET-Bench 金融长文档|edinet-bench',
              'FinMMDocR 多模态文档基准|finmmdocr',
              'FinSearchComp 检索基准|finsearchcomp',
            ],
          ],
        ],
      ],
      [
        '交易智能体',
        [
          [
            '系统',
            [
              'TiMi：离线研发 + 分钟级执行|timi-trading-agent',
              'TradingAgents 多智能体辩论|tradingagents-multiagent',
              'LLM 情绪 alpha（Lopez-Lira-Tang）|llm-sentiment-alpha',
              'LLM embedding 因子|llm-embedding-alpha',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '期权流与经销商定位',
    [
      [
        'GEX 与 gamma 体制',
        [
          [
            '机制',
            [
              'GEX 计算与聚合|gex-calculation',
              'Gamma Flip 零 gamma 位|gamma-flip-level',
              '经销商正负 gamma 体制|dealer-gamma-regime',
              '0DTE 期权微观结构|zero-dte-microstructure',
              'Strike Magnetism / Max Pain|strike-magnetism-max-pain',
              '经销商 Vanna / Charm 到期流|dealer-vanna-charm-flows',
              'JPM Collar 季度护盘|jpm-collar-flow',
              'Vol ETP 每日再平衡流|vol-etp-rebalance-flow',
            ],
          ],
        ],
      ],
      [
        '学术锚点',
        [
          [
            '论文',
            [
              'Ni et al. 期权需求压力|ni-option-demand-pressure',
              'Amaya et al. 0DTE 与波动|amaya-0dte-volatility',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '中国监管与产品结构',
    [
      [
        '程序化交易新规',
        [
          [
            '监管要点',
            [
              '程序化交易管理规定与实施细则|cn-program-trading-rules',
              '高频交易认定标准|cn-hft-threshold',
              '差异化收费与撤单率监控|cn-fee-cancel-ratio',
              '融券 T+0 封堵|cn-margin-short-t0',
              '主机托管清退与交易单元整改|cn-colocation-rectify',
              '中基协程序化权威解读|cn-amac-program-trading',
            ],
          ],
        ],
      ],
      [
        '雪球与杠杆产品',
        [
          [
            '产品',
            [
              '雪球结构与定价|snowball-pricing',
              '敲入事件与对冲踩踏|snowball-knockin-cascade',
              '对冲商的障碍期权堆积与 gamma|snowball-dealer-hedging',
              'DMA 杠杆与微盘流动性危机|dma-microcap-crisis',
              '指增产品超额衰减|index-enhancement-decay',
              'T0 底仓增强|t0-basis-enhancement',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '机器学习资产定价',
    [
      [
        '深度截面模型',
        [
          [
            '模型族',
            [
              'Gu-Kelly-Xiu 与 NN3|gkx-nn3',
              'Autoencoder 条件因子模型|autoencoder-factor-model',
              'GNN 资产定价|gnn-asset-pricing',
              '供应链网络因子|supply-chain-graph-factor',
              'Transformer 截面收益预测|transformer-return-prediction',
              '频域与时频模型|frequency-domain-models',
            ],
          ],
        ],
      ],
      [
        '因果与可解释',
        [
          [
            '方法',
            [
              '双重机器学习 DML|double-machine-learning',
              '因果树与异质处理效应|causal-tree-heterogeneity',
              'SHAP 因子归因|shap-factor-attribution',
              '跨市场迁移学习|cross-market-transfer-learning',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '深度学习微观结构与市场仿真',
    [
      [
        '深度订单簿',
        [
          [
            '模型与陷阱',
            [
              'DeepLOB|deeplob-cnn-lstm',
              'TransLOB / AxialLOB|translob-axiallob',
              'Neural Hawkes 订单流强度|neural-hawkes-orderflow',
              '深度 LOB 模型的泄漏陷阱|deep-lob-leakage',
            ],
          ],
        ],
      ],
      [
        '生成式市场仿真',
        [
          [
            '系统与方法',
            [
              'ABIDES 事件驱动仿真|abides-simulation',
              'MarS 生成式订单流|mars-generative-sim',
              'Market Generator（VAE）|market-generator-vae',
              'Sig-Wasserstein GAN|sig-wasserstein-gan',
              'LLM 多智能体市场仿真|llm-multiagent-market-sim',
              '生成式回测的评估|generative-backtest-eval',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '深度对冲与学习定价',
    [
      [
        '深度方法',
        [
          [
            '模型',
            [
              'Deep Hedging|deep-hedging-buehler',
              'Deep BSDE 求解器|deep-bsde',
              'Neural SDE|neural-sde',
              'PINN 期权定价|pinn-option-pricing',
              'Path Signature 特征|path-signature-features',
              'Signature Trading|signature-trading',
              'Sig-SDE|sig-sde',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '数据与研究基础设施',
    [
      [
        '数据栈',
        [
          [
            '工具',
            [
              'kdb+ / q|kdb-plus',
              'DolphinDB|dolphindb',
              'ClickHouse 金融时序|clickhouse-finance-ts',
              'Apache Arrow 零拷贝|apache-arrow',
              'Microsoft Qlib|qlib-quant',
              'vn.py 交易框架|vnpy-framework',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '零售流与预测市场',
    [
      [
        '零售与注意力',
        [
          [
            '流与信号',
            [
              'Boehmer 零售流识别|boehmer-retail-flow',
              'PFOF 订单流支付|pfof-payment-flow',
              'meme 事件与轧空|meme-short-squeeze',
              'Da-Engel-Gao 投资者关注度|investor-attention-deg',
              'WSB 舆情因子|wsb-sentiment-factor',
            ],
          ],
        ],
      ],
      [
        '预测市场与加密',
        [
          [
            '新市场',
            [
              'Polymarket / Kalshi 事件合约|prediction-market-venues',
              '预测市场跨市场套利|prediction-market-arbitrage',
              '比特币现货 ETF 基差|btc-etf-basis',
              '链上交易所流入流出|onchain-exchange-flows',
              'Deribit DVOL 隐含波动|deribit-dvol',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '稳健方法与模型治理',
    [
      [
        '鲁棒与在线',
        [
          [
            '方法',
            [
              'Wasserstein DRO 组合|wasserstein-dro-portfolio',
              'BOCPD 在线变点检测|bocpd-online-changepoint',
              'Martingale Optimal Transport|martingale-optimal-transport',
              '强化学习做市（Ganesh-Cartea）|rl-market-making',
            ],
          ],
        ],
      ],
      [
        '合规与验证',
        [
          [
            '制度',
            [
              '美国 T+1 结算|us-t1-settlement',
              'SR 11-7 模型风险管理|sr-11-7-model-risk',
              'MiFID II RTS 6 算法交易合规|mifid-ii-rts6',
              'EU AI Act 金融条款|eu-ai-act-finance',
            ],
          ],
        ],
      ],
    ],
  ],
]

export const quantFrontier = fromOutline(extra)
