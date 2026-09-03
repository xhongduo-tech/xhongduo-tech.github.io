import { fromOutline, type Outline } from './schema'

/** 专业审核补层：架构/RL/智能体/生成 + 推理系统/MoE/数值/封装/评测 */
const extra: Outline[] = [
  [
    '前沿架构续',
    [
      [
        '注意力与序列模型',
        [
          [
            '稀疏与内核',
            [
              'DeepSeek Sparse Attention|deepseek-sparse-attention',
              'Differential Attention|differential-attention',
              'FlexAttention|flex-attention',
              'Lightning Attention|lightning-attention',
              'Gated DeltaNet|gated-delta-net',
              'DeltaNet|delta-net',
              'Mixture-of-Depths|mixture-of-depths',
              'Mixture-of-Recursions|mixture-of-recursions',
              'EAGLE-3|eagle-3',
            ],
          ],
          [
            '训练结构细节',
            [
              'QK-Norm|qk-norm-pretrain',
              'xIELU 激活|xielu-activation',
              'Cut Cross-Entropy|cut-cross-entropy',
              'MoE z-loss|z-loss-moe',
              'Multi-Token Prediction 训练目标|multi-token-prediction-training',
              '跨文档 packing 掩码|packing-cross-doc-mask',
              'Document attention mask|document-masking',
            ],
          ],
        ],
      ],
      [
        '预训练配方续',
        [
          [
            '阶段与数据',
            [
              'Midtraining|midtraining',
              'Annealing 退火阶段|annealing-phase',
              'Continued Pretraining|continued-pretraining',
              'OLMo 2 midtrain 配方|olmo2-midtrain-recipe',
            ],
          ],
        ],
      ],
      [
        '前沿型号续',
        [
          [
            '闭源与开源型号',
            [
              'GPT-5|gpt-5',
              'Claude 4 Opus|claude-4-opus',
              'Claude 4 Sonnet|claude-4-sonnet',
              'gpt-oss-120B|gpt-oss-120b',
              'Llama 4 Scout|llama-4-scout',
              'Llama 4 Maverick|llama-4-maverick',
              'Qwen3-235B-A22B|qwen3-235b-a22b',
              'DeepSeek-R1-0528|deepseek-r1-0528',
              'MiniMax-M1|minimax-m1',
              'Seed-1.6|seed-1-6',
              'Hunyuan-TurboS|hunyuan-turbos',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '推理模型与强化学习续',
    [
      [
        '策略优化',
        [
          [
            '群体与稳定化',
            [
              'DAPO|dapo',
              'GSPO|gspo',
              'Dr. GRPO|dr-grpo',
              'REINFORCE++|reinforce-plusplus',
              'RLVR|rlvr',
              'On-policy Distillation|on-policy-distillation',
              'OpenR1 训练配方|openr1-recipe',
            ],
          ],
        ],
      ],
      [
        '过程奖励与算力',
        [
          [
            'PRM 与预算',
            [
              '过程奖励模型训练|process-reward-training',
              '结果奖励模型|outcome-reward-model',
              'Math-Shepherd PRM|math-shepherd-prm',
              'Skywork-o1 PRM|skywork-o1-prm',
              'Budget Forcing / s1|budget-forcing-s1',
              'Deliberative Alignment|deliberative-alignment',
              'CoT Monitorability|cot-monitorability',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '智能体系统续',
    [
      [
        '协议与计算机使用',
        [
          [
            '编排',
            [
              'A2A 协议|a2a-protocol',
              'Anthropic Computer Use|anthropic-computer-use-api',
              'OpenAI Operator|openai-operator',
              'Gemini Computer Use|gemini-computer-use',
              'Browser-use Agent|browser-use-agent',
              '多代理编排|multi-agent-orchestration',
            ],
          ],
        ],
      ],
      [
        '记忆与检索增强',
        [
          [
            '记忆与向量',
            [
              'MemGPT|memgpt',
              'A-MEM 代理记忆|amem-agent-memory',
              'BGE-M3|bge-m3',
              'BGE Reranker v2|bge-reranker-v2',
              'ColBERTv2|colbert-v2',
              'GraphRAG|graphrag',
              'HippoRAG|hipporag',
              'LightRAG|light-rag',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '生成与多模态续',
    [
      [
        '扩散语言模型',
        [
          [
            '离散扩散与流',
            [
              'Diffusion LM|diffusion-lm',
              'SEDD|sedd-diffusion-lm',
              'MDLM|mdlm',
              'LLaDA|llada',
              'Discrete Flow Matching|discrete-flow-matching',
            ],
          ],
        ],
      ],
      [
        '视频与图像生成',
        [
          [
            'DiT 与产品',
            [
              'DiT 架构|dit-architecture',
              'Sora|sora',
              'Veo 3|veo-3',
              'Wan 2.1|wan-2-1',
              'Kling|kling-video',
              'HunyuanVideo|hunyuan-video',
              'FLUX.1|flux-dev',
              'NVIDIA Cosmos 世界模型|cosmos-world-model',
            ],
          ],
        ],
      ],
      [
        '语音与水印',
        [
          [
            '通用语音与出处',
            [
              'Whisper large-v3|whisper-large-v3',
              'CosyVoice 2|cosyvoice-2',
              'SenseVoice|sensevoice-small',
              'Fish-Speech|fish-speech',
              'SynthID|synthid-watermark',
              'Kirchenbauer 水印|kirchenbauer-watermark',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '推理系统续',
    [
      [
        '引擎与编排',
        [
          [
            '运行时',
            [
              'NVIDIA Dynamo|nvidia-dynamo',
              'Triton Inference Server|triton-inference-server',
              'Ray Serve LLM|ray-serve-llm',
              'vLLM V1 架构|vllm-v1',
              'SGLang Router|sglang-router',
              'LMCache|lmcache',
            ],
          ],
        ],
      ],
      [
        'PD 分离与 KV 池',
        [
          [
            '分离服务',
            [
              'Mooncake Transfer Engine|mooncake-transfer-engine',
              'Mooncake Store|mooncake-store',
              '分离 Prefill 资源池|disagg-prefill-pool',
              '分层 KV：HBM→CPU→Disk|kv-tiered-storage',
              'KV RDMA 远端拉取|kv-rdma-fetch',
              '前缀感知扩缩容|prefix-aware-autoscaling',
            ],
          ],
        ],
      ],
      [
        '投机与多 Token 服务',
        [
          [
            '解码加速',
            [
              'Hydragen 共享前缀投机|hydragen',
              'MTP 验证闭环|mtp-verify-loop',
              '接受长度调度|acceptance-length-sched',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    'MoE 推理与并行',
    [
      [
        '通信与内核',
        [
          [
            'DeepSeek 栈与专家并行',
            [
              'DeepEP|deepep',
              'DeepGEMM|deepgemm',
              'FlashMLA|flashmla',
              'EP All-to-All|ep-all2all',
              'Token Dispatch / Combine|token-dispatch-combine',
              'EPLB 专家负载均衡|eplb',
              'Decode 侧 Expert Parallel|expert-parallel-decode',
              'MegaBlocks Grouped GEMM|megablocks',
              '序列并行推理|infer-sp',
              'Context Parallel 推理|infer-cp',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '数值与量化续',
    [
      [
        '内核与低精度',
        [
          [
            'FP4 / FP8 / INT4',
            [
              'TileLang|tilelang',
              'CUTLASS Grouped GEMM|cutlass-grouped-gemm',
              'Blackwell SM100 MMA|blackwell-sm100-mma',
              'NVFP4 Tensor Core 路径|nvfp4-tc',
              'MXFP4 microscaling|mxfp4-microscale',
              'MXFP8|mxfp8',
              'FP8 训练（Transformer Engine）|fp8-training',
              'Marlin INT4 内核|marlin',
              'ExLlamaV2 内核|exllamav2',
              'KV FP4 量化|kv-fp4',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '硬件与封装续',
    [
      [
        '异构与先进封装',
        [
          [
            '加速器与互连',
            [
              'TPU v5e / v6e 推理|tpu-v5e-infer',
              'Trainium2 / Inferentia2|trainium2-inferentia2',
              'Gaudi 3|gaudi3',
              '寒武纪 MLU|cambricon-mlu',
              'CXL 内存池化|cxl-memory-pool',
              'UCIe die-to-die|ucie-d2d',
              'CoWoS-L / CoWoS-R|cowos-l-r',
              'JEDEC HBM4|jedec-hbm4',
              'HBM3E|hbm3e',
              'OCS 光交换 AI 网络|ocs-ai-fabric',
              'NVSHMEM / NVLS|nvshmem-nvls',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '评测与基准续',
    [
      [
        '系统与能力',
        [
          [
            '服务与前沿基准',
            [
              'GenAI-Perf|genai-perf',
              'MLPerf Inference LLM|mlperf-inference-llm',
              '延迟-吞吐帕累托|latency-throughput-pareto',
              '百万 token 成本|cost-per-mtok',
              "Humanity's Last Exam|humanitys-last-exam",
              'FrontierMath|frontiermath',
              'LiveBench|livebench',
              'BrowseComp|browsecomp',
            ],
          ],
        ],
      ],
    ],
  ],
]

export const llmAudit = fromOutline(extra)
