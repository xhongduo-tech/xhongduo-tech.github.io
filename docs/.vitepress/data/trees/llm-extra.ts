import { fromOutline, type Outline } from './schema'

/** 补层：采样、安全、CUDA、数据工程、评测、对齐、服务、多模态 */
const extra: Outline[] = [
  [
    '采样与解码策略',
    [
      [
        '随机与约束',
        [
          [
            '采样',
            [
              'Temperature / Top-k / Top-p|sampling-temperature-topp',
              'Min-p / Typical / η-sampling|minp-typical',
              'Repetition / frequency penalty|repetition-penalty',
              'CFG 用于语言模型|cfg-llm',
              '停用词与 stop sequences|stop-sequences',
              '投机采样与采样一致性|speculative-sampling',
              'Mirostat 困惑度控制|mirostat',
              'Contrastive Search|contrastive-search',
              'Locally Typical Sampling|locally-typical',
              'Epsilon sampling|epsilon-sampling',
              'Top-a / quadratic sampling|top-a-sampling',
              'Grammar-constrained decoding|grammar-decode',
              'JSON schema 约束解码|json-constrained-decode',
              '正则约束解码|regex-constrained-decode',
              'FIM / fill-in-the-middle 解码|fim-decode',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '安全与滥用',
    [
      [
        '对齐失效',
        [
          [
            '攻击与防御',
            [
              '越狱与越狱评测|jailbreak',
              '提示注入|prompt-injection',
              '数据投毒与后门|data-poisoning',
              '成员推断与训练数据提取|membership-extraction',
              '输出过滤与分类器|output-filter',
              '间接提示注入与检索投毒|indirect-injection',
              '多模态越狱|multimodal-jailbreak',
              'GCG / 自动对抗后缀|gcg-suffix',
              'Many-shot jailbreak|many-shot-jailbreak',
              'Prefilling 越狱|prefill-jailbreak',
              'Llama Guard / 安全分类器|llama-guard',
              'WildGuard / HarmBench|wildguard-harmbench',
              'Constitutional classifiers|constitutional-classifiers',
              'Circuit Breaker 消融有害方向|circuit-breaker',
              '表示工程抑制有害行为|repeng-refusal',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    'CUDA 与内核实现',
    [
      [
        '编程模型',
        [
          [
            '基础',
            [
              'Warp / CTA / 占用率|cuda-occupancy',
              'Shared memory 与 bank conflict|shared-memory-banks',
              'Tiling 与融合|kernel-fusion-tiling',
              'Warp specialization|warp-specialization',
              'CUTLASS 层次|cutlass',
              'cuBLAS / cuDNN 边界|cublas-cudnn',
              'CUDA Graph 捕获推理|cuda-graph-infer',
              'TMA 与 Hopper 异步拷贝|hopper-tma',
              'WGMMA / wgmma 指令|wgmma',
              'Persistent kernel|persistent-kernel',
              '软件流水与 double buffering|sw-pipeline-buffer',
              'FlashInfer 内核库|flashinfer',
              'ThunderKittens|thunderkittens',
              'Triton 写注意力核|triton-attention-kernel',
              'cuTeDSL / CUTLASS 3|cutlass3-cute',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '数据工程',
    [
      [
        '流水线',
        [
          [
            '预处理',
            [
              '分布式清洗|distributed-clean',
              '语言识别与安全过滤|langid-safety-filter',
              'PII 去除|pii-redact',
              'Packing 与文档边界|sequence-packing',
              'Checkpoint 与 resume|pretrain-ckpt',
              'Common Crawl WARC 解析|cc-warc',
              'trafilatura / resiliparse 抽取|html-extract',
              'CCNet 管道|ccnet',
              'Dolma / FineWeb 配方|dolma-fineweb',
              'DCLM 数据过滤|dclm-filter',
              '语义去重 SemDeDup|semdedup',
              'Cross-lingual 去重|xl-dedup',
              '数学数据抽取与校对|math-extract',
              '代码许可与质量门|code-license-gate',
              'PDF / 扫描件进预训练|pdf-ocr-pretrain',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '评测方法学',
    [
      [
        '基准设计',
        [
          [
            '泄漏与污染',
            [
              '训练-评测污染检测|eval-contamination',
              '动态基准与题库轮换|live-benchmarks',
              'n-gram 重叠与嵌入近邻|contamination-ngram',
              'Canary 字符串|canary-strings',
            ],
          ],
          [
            '能力基准',
            [
              'MMLU-Pro / GPQA|mmlu-pro-gpqa',
              'BIG-bench / BBH|bigbench-bbh',
              'HELM 多维度|helm',
              'LiveCodeBench|livecodebench',
              'SWE-bench Verified|swebench-verified',
              'AIME / 竞赛数学|aime-contest-math',
              'SimpleQA / FreshQA|simpleqa-freshqa',
              'MMMU / MathVista|mmmu-mathvista',
              'LongBench / RULER / ∞Bench|longbench-ruler',
              'Needle-in-haystack 变体|niah-variants',
              'IFEval / IFBench|ifbench',
              'BFCL 工具调用评测|bfcl',
              'τ-bench / AgentBench|taubench-agentbench',
              'Arena-Hard / AlpacaEval 2|arena-hard',
              'MT-Bench 多轮|mt-bench',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '对齐数据与偏好',
    [
      [
        '偏好收集',
        [
          [
            '标注',
            [
              '成对比较协议|pairwise-label-protocol',
              'Likert 与绝对评分|absolute-rating-prefs',
              '多轴偏好：有用/诚实/无害|hhh-axes',
              '专家 vs 众包标注|expert-vs-crowd',
              'AI 反馈作为偏好|aif-as-preference',
              '隐式反馈：点选与停留|implicit-feedback-prefs',
            ],
          ],
          [
            '目标冲突',
            [
              '过度拒绝 over-refusal|over-refusal',
              '谄媚 sycophancy|sycophancy',
              '奖励黑客|reward-hacking-llm',
              '风格黑客与长度偏置|length-bias-rm',
              '虚构引用与诚实性|honesty-citations',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '服务工程',
    [
      [
        '多 LoRA 与适配',
        [
          [
            '服务形态',
            [
              'S-LoRA 分页适配器|slora',
              'Punica 背景批|punica',
              'dLoRA 动态加载|dlora',
              'LoRAX|lorax-serving',
              '适配器热更新|adapter-hot-swap',
            ],
          ],
        ],
      ],
      [
        '网关与可观测',
        [
          [
            '生产',
            [
              'token 计量与限流|token-ratelimit',
              '提示缓存计费|prompt-cache-billing',
              '追踪：TTFT 分位|ttft-percentiles',
              '内容安全同步卡点|moderation-inline',
              '多区域故障转移|multi-region-failover',
              '模型路由与级联|model-cascade-routing',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '推理链与提示',
    [
      [
        '提示范式',
        [
          [
            '链式方法',
            [
              'Chain-of-Thought|chain-of-thought',
              'Zero-shot CoT|zero-shot-cot',
              'Self-Consistency|self-consistency',
              'Tree of Thoughts|tree-of-thoughts',
              'Graph of Thoughts|graph-of-thoughts',
              'Least-to-Most|least-to-most',
              'PAL / Program-of-Thoughts|pal-pot',
              'ReAct 提示形态|react-prompting',
              'Reflexion|reflexion',
              'Self-Ask|self-ask',
              'Plan-and-Solve|plan-and-solve',
              'Analogical prompting|analogical-prompting',
            ],
          ],
        ],
      ],
    ],
  ],
]

export const llmExtra = fromOutline(extra)
