import { fromOutline, type Outline } from './schema'

/** 前沿补层（2025H2–2026 审计）：可解释性 / Agentic RL / 上下文工程 / 新架构与型号 / 系统与硬件 / 评测与智能体 / 多模态与具身 */
const extra: Outline[] = [
  [
    '机制可解释性',
    [
      [
        '特征与字典学习',
        [
          [
            'SAE 路线',
            [
              'Superposition 叠加假说|superposition-hypothesis',
              'Toy Models of Superposition|toy-models-superposition',
              '稀疏自编码器 SAE|sae-sparse-autoencoder',
              'Dictionary Learning 超完备字典|dictionary-learning-interp',
              'Towards Monosemanticity|towards-monosemanticity',
              'Scaling Monosemanticity 与特征转向|scaling-monosemanticity',
              'Gemma Scope 开放 SAE 生态|gemma-scope',
            ],
          ],
        ],
      ],
      [
        '电路与干预',
        [
          [
            '机制分析',
            [
              'Induction Heads 归纳头|induction-heads',
              'In-context Learning 的诱导回路|icl-induction-circuit',
              'Attribution Graphs 归因图|attribution-graphs',
              'Circuit Tracing 电路追踪|circuit-tracing',
              'Logit Lens / Tuned Lens|logit-lens',
              'Probing 线性探针|linear-probing',
              'Activation Patching 激活修补|activation-patching',
              'Steering Vectors 表示操纵向量|steering-vectors',
              '线性表示假设|linear-representation-hypothesis',
              'Grokking 与突然泛化|grokking',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    'Agentic RL 与训练框架',
    [
      [
        '训练框架',
        [
          [
            '主流框架',
            [
              'verl 与 HybridFlow|verl-hybridflow',
              'OpenRLHF|openrlhf',
              'TRL|trl-framework',
              'NeMo-RL|nemo-rl',
              'slime 轻量 RL 框架|slime-rl',
              'AReaL 异步强化学习|areal-async-rl',
              'VERLTool 工具 RL 集成|verltool',
              'Agent Lightning 训练器注入 harness|agent-lightning',
            ],
          ],
        ],
      ],
      [
        '异步与多轮工程',
        [
          [
            '工程要点',
            [
              '异步 rollout 架构|async-rollout-arch',
              'server-based AgentLoop 与 token 级 API|agentloop-server',
              '多轮对话 loss mask|multiturn-loss-mask',
              'delta tokenization 与边界 token|delta-tokenization',
              'GiGPO 多轮步级优势|gigpo-credit',
              'rollout 与训练资源复用|rollout-resource-sharing',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '上下文工程与记忆',
    [
      [
        '上下文管理',
        [
          [
            '原理与策略',
            [
              'Context Rot 上下文腐烂|context-rot',
              'Lost in the Middle|lost-in-middle',
              'Compaction 摘要压缩|compaction-summarize',
              '结构化记忆压缩|structured-memory-compaction',
              '检索式上下文压缩|retrieval-based-compaction',
              'Just-in-Time Context Retrieval|jit-context-retrieval',
              '渐进式披露|progressive-disclosure',
              'AGENTS.md 项目规范|agents-md-spec',
              '多智能体上下文隔离|multi-agent-context-isolation',
            ],
          ],
        ],
      ],
      [
        '压缩与记忆系统',
        [
          [
            '论文与产品',
            [
              'ACM Agentic Context Management|acm-context',
              'AgentFold 上下文折叠|agentfold',
              'Mem1 递归上下文重写|mem1-context',
              'ReSum 递归摘要|resum-context',
              'ACON 上下文导航|acon-context',
              'SUPO 上下文优化|supo-context',
              'Mem0 记忆层|mem0-layer',
              'Zep / Graphiti 时序知识图谱记忆|zep-graphiti',
              'MemOS 记忆操作系统|memos-os',
              'Letta 记忆代理框架|letta-framework',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '前沿架构与型号 2025-2026',
    [
      [
        '架构',
        [
          [
            '线性与混合',
            [
              'Kimi Linear 与 KDA|kimi-linear-kda',
              'Qwen3-Next 混合架构|qwen3-next-hybrid',
              'RWKV-7 Goose|rwkv-7-goose',
              'xLSTM 复兴|xlstm',
              'Titans 测试时记忆|titans-test-time-memory',
            ],
          ],
          [
            '目标、表示与内核',
            [
              'COCONUT 潜空间思维链|coconut-latent-cot',
              'Byte Latent Transformer|byte-latent-transformer',
              'MoBA 块注意力混合|moba-block-attention',
              'FlashKDA 关联记忆内核|flashkda-kernel',
              'Sparse MLA|sparse-mla',
              '原生 MXFP4 预训练权重|native-mxfp4-weights',
              'Mercury 商用扩散 LM|mercury-diffusion-lm',
              'Gemini Diffuse 扩散推理|gemini-diffuse',
            ],
          ],
        ],
      ],
      [
        '型号',
        [
          [
            '闭源前沿',
            [
              'GPT-5.5（Sol / Terra / Luna）|gpt-5-5',
              'Claude Opus 4.6|claude-opus-4-6',
              'Claude Sonnet 5|claude-sonnet-5',
              'Claude Opus 5|claude-opus-5',
              'Claude Fable 5|claude-fable-5',
              'Gemini 3 / Deep Think|gemini-3',
              'Grok 4.5 / 4.6|grok-4-5',
            ],
          ],
          [
            '开源前沿',
            [
              'Kimi K3|kimi-k3',
              'DeepSeek V4|deepseek-v4',
              'GLM-5|glm-5',
              'Qwen3.5（397B-A17B）|qwen-3-5',
              'Qwen3.8-Max|qwen-3-8-max',
              'MiniMax M-2.5|minimax-m-2-5',
              'Inkling（Thinking Machines）|inkling-tml',
              'Meta Muse|meta-muse',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '推理系统与硬件 2026',
    [
      [
        '推理系统',
        [
          [
            '新组件',
            [
              'DeepSeek 3FS 文件系统|deepseek-3fs',
              'KTransformers CPU/GPU 混合推理|ktransformers-hybrid',
              'AMX 内核与至强加速|amx-kernel',
              'vLLM 0.28：KV 分层与 DCP|vllm-0-28',
              'Ray 2.58 KV-aware 路由|ray-kv-aware-routing',
              'NVIDIA Inference Context Memory|nvidia-inference-context-memory',
            ],
          ],
        ],
      ],
      [
        '硬件',
        [
          [
            'Google TPU 第八代',
            [
              'TPU 8t（Sunfish）训练|tpu-8t-sunfish',
              'TPU 8i（Zebrafish）推理|tpu-8i-zebrafish',
              'Virgo ICI 互连|virgo-ici',
              'Boardfly 芯片间互连|boardfly',
              'CAE 集合加速引擎|tpu-cae',
            ],
          ],
          [
            'AWS / AMD / NVIDIA 下代',
            [
              'Trainium 3 与 Trn3 UltraServer|trainium-3',
              'Project Rainier 超级集群|project-rainier',
              'AMD MI350|amd-mi350',
              'AMD MI400 与 Helios 机架|mi400-helios',
              'NVIDIA Feynman 架构|feynman-arch',
              'Feynman A16 原型|feynman-a16',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '评测与智能体 2026',
    [
      [
        '评测',
        [
          [
            '前沿基准',
            [
              'Terminal-Bench 2|terminal-bench-2',
              'SWE-Bench Pro|swe-bench-pro',
              'ARC-AGI-2|arc-agi-2',
              'METR Time Horizon|metr-time-horizon',
              'METR CoT 监控与 reward hacking|metr-cot-monitoring',
              'Vending-Bench 2|vending-bench-2',
              'Artificial Analysis Intelligence Index|artificial-analysis-index',
            ],
          ],
        ],
      ],
      [
        '智能体工程',
        [
          [
            'CLI 与 SDK',
            [
              'Claude Code|claude-code-cli',
              'Codex CLI|codex-cli',
              'Gemini CLI|gemini-cli',
              'Anthropic Agent Skills|agent-skills',
              'Claude Agent SDK|claude-agent-sdk',
              'AI Scientist 自动化研究|ai-scientist',
              'Agent Harness 设计|agent-harness-design',
            ],
          ],
        ],
      ],
      [
        '智能体安全',
        [
          [
            '沙箱与注入防御',
            [
              'E2B / Docker agent 沙箱|agent-sandbox-e2b',
              'CaMeL 间接注入防御|camel-injection-defense',
              'Spotlighting 输入标记|spotlighting-defense',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '多模态生成与具身 2026',
    [
      [
        '视频与世界模型',
        [
          [
            '产品',
            [
              'Sora 2 与 Sora App|sora-2',
              'Genie 3 交互世界模型|genie-3',
              'Seedance 2.0|seedance-2',
              '实时视频交互|realtime-video-interaction',
            ],
          ],
        ],
      ],
      [
        '图像生成',
        [
          [
            '模型',
            [
              'Gemini 原生图像生成|gemini-native-image',
              'Nano Banana 2|nano-banana-2',
              'FLUX.2|flux-2',
            ],
          ],
        ],
      ],
      [
        '具身智能',
        [
          [
            'VLA 与机器人基座',
            [
              'π0 / π0.5 VLA|pi-0-vla',
              'Figure Helix 双系统|figure-helix',
              'NVIDIA GR00T N1|groot-n1',
              'V-JEPA 2 世界模型|vjepa-2',
              'Open X-Embodiment|open-x-embodiment',
            ],
          ],
        ],
      ],
    ],
  ],
]

export const llmFrontier = fromOutline(extra)
