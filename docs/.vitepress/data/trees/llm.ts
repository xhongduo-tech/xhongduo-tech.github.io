import { fromOutline, markAppendix, type Outline } from './schema'
import {
  extraSampling,
  extraSafety,
  extraCuda,
  extraDataEng,
  extraEvalMethod,
  extraAlignData,
  extraServing,
  extraPrompting,
} from './llm-extra'
import { llmPapers } from './llm-papers'
import { llmAudit } from './llm-audit'
import { llmFrontier } from './llm-frontier'
import { llmSideline } from './llm-sideline'
import { llmSupplement } from './llm-supplement'
import { llmFoundations } from './llm-foundations'

const representation: Outline = [
  '表示与 Transformer 块',
  [
    [
      '分词',
      [
        [
          '从字符到子词',
          [
            '离散符号与词表|token-as-discrete-unit',
            'BPE 合并规则|bpe-merge-rule',
            'Unigram 切分|unigram-segmentation',
            '词表大小与 UNK|vocab-size-unk',
            'Detokenize 与往返|detokenize-roundtrip',
          ],
        ],
      ],
    ],
    [
      '嵌入',
      [
        [
          '查找表',
          [
            'Token 嵌入查找|embedding-lookup',
            'Tied vs untied 嵌入|tied-untied-embedding',
            '词表宽度与 d_model|embedding-dmodel',
          ],
        ],
      ],
    ],
    [
      '残差块',
      [
        [
          '一块的组成',
          [
            '残差流|residual-stream',
            'Transformer 块：注意力加前馈|transformer-block-compose',
            'FFN 扩展比|ffn-expansion-ratio',
            '块内信息路径|block-information-path',
          ],
        ],
      ],
    ],
  ],
]

const trunk: Outline[] = [
  [
    '模型架构',
    [
      [
        '注意力',
        [
          [
            '点积与多头',
            [
              'Scaled Dot-Product Attention|sdpa',
              'Multi-Head Attention|mha',
              'Self-Attention 与因果掩码|causal-self-attention',
              'Cross-Attention|cross-attention',
              'Encoder-Decoder Attention|encoder-decoder-attention',
              'Attention 中的缩放、温度与数值稳定|attention-scale-stability',
              'Softmax 注意力的二次复杂度|attention-quadratic-cost',
            ],
          ],
          [
            'KV 头压缩',
            [
              'Multi-Query Attention|mqa',
              'Grouped-Query Attention|gqa',
              'Multi-head Latent Attention|mla',
              'KV 头数、质量与吞吐的权衡|kv-head-tradeoff',
              '共享 KV 下的训练稳定性|shared-kv-training',
            ],
          ],
          [
            '稀疏、局部与线性',
            [
              'Sliding Window Attention|sliding-window-attention',
              'Dilated Attention|dilated-attention',
              'Local-Global Hybrid Attention|local-global-attention',
              'BigBird / Longformer 稀疏模式|sparse-attention-patterns',
              'Linear Attention|linear-attention',
              'Performer / FAVOR+|performer-favor',
              'Attention Sink|attention-sink',
              'Native Sparse Attention|native-sparse-attention',
            ],
          ],
          [
            '状态空间与混合',
            [
              'Mamba|mamba',
              'Mamba-2|mamba-2',
              'Griffin / Recurrent Hybrids|griffin-hybrid',
              'Jamba 注意力-SSM 混合|jamba',
              '线性 RNN 与注意力的分工|linear-rnn-vs-attention',
            ],
          ],
        ],
      ],
      [
        '位置与结构',
        [
          [
            '位置编码',
            [
              '绝对位置编码|absolute-position',
              '正弦位置编码|sinusoidal-pe',
              '可学习位置嵌入|learned-pe',
              'RoPE|rope',
              'RoPE 的频率、基数与长上下文|rope-frequency',
              'ALiBi|alibi',
              'YaRN|yarn',
              'NTK-aware 插值|ntk-aware-interpolation',
              '位置插值与外推失败模式|position-extrapolation',
              'NoPE|nope',
            ],
          ],
          [
            '块结构',
            [
              'Pre-LN 与 Post-LN|pre-ln-post-ln',
              'RMSNorm|rmsnorm',
              'LayerNorm|layernorm',
              '残差连接与 DeepNet|residual-scaling',
              '并行 Attention-FFN|parallel-attn-ffn',
              'Sandwich / Sandwich-LN|sandwich-ln',
            ],
          ],
          [
            '前馈与稀疏专家',
            [
              'ReLU FFN|relu-ffn',
              'GeLU / SwiGLU|swiglu',
              'Gated Linear Unit|glu',
              'Mixture of Experts 路由|moe-routing',
              'Switch Transformer|switch-transformer',
              'Expert Parallelism|expert-parallelism',
              '负载均衡损失|moe-load-balance',
              '共享专家与细粒度专家|shared-expert-moe',
              'DeepSeek MoE|deepseek-moe',
              'MoE 推理时的专家缓存|moe-inference-cache',
              'Capacity factor|moe-capacity-factor',
              'Dropless MoE|moe-dropless',
              'Expert-choice 路由|moe-expert-choice',
              '路由器梯度|moe-router-gradient',
              '共享专家与路由专家的配比|moe-shared-routed-size',
            ],
          ],
        ],
      ],
      [
        '长上下文',
        [
          [
            '外推与窗口',
            [
              '上下文长度与损失缩放|context-length-scaling',
              '位置外推评测|long-context-eval',
              'StreamingLLM|streaming-llm',
              'LM-Infinite|lm-infinite',
              'Self-Extend|self-extend',
              'Dual Chunk Attention|dual-chunk-attention',
            ],
          ],
          [
            '记忆与压缩',
            [
              '压缩记忆 / Compressive Transformer|compressive-transformer',
              'Infini-Attention|infini-attention',
              'Activation Beacon|activation-beacon',
              'Landmark Attention|landmark-attention',
              'KV 压缩作为长上下文手段|kv-as-long-context',
            ],
          ],
        ],
      ],
      [
        '多模态',
        [
          [
            '视觉语言',
            [
              'CLIP 对齐|clip',
              'ViT 作为视觉编码器|vit-as-encoder',
              'LLaVA 投影器|llava-projector',
              'Qwen-VL / InternVL 连接器|vl-connector',
              '动态分辨率与 AnyRes|anyres',
              '视觉 token 压缩|vision-token-compression',
              'OCR 与文档 VLM|ocr-vlm',
              'Early vs late fusion|early-late-fusion',
            ],
          ],
          [
            '语音与视频',
            [
              '语音 tokenizer|speech-tokenizer',
              '音频语言模型|audio-lm',
              '音频 codec 与离散 token|audio-codec-lm',
              '视频 token 与时间采样|video-tokens',
              '视频时序融合|video-temporal-pool',
              '多模态交错训练|interleaved-multimodal',
              '离散扩散语言模型|discrete-diffusion-lm',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '预训练',
    [
      [
        '目标与数据',
        [
          [
            '训练目标',
            [
              'Causal LM|causal-lm',
              'Masked LM|masked-lm',
              'Prefix LM|prefix-lm',
              'UL2 / 混合去噪|ul2',
              '下一 token 之外的预训练目标|beyond-ntp',
            ],
          ],
          [
            '数据',
            [
              '网页清洗与去重|web-clean-dedup',
              'MinHash / 精确去重|minhash-dedup',
              '质量过滤与分类器|data-quality-filter',
              '配比定律与领域配比|data-mixture-laws',
              '代码、数学、多语言配比|code-math-multilingual-mix',
              '合成数据在预训练中的位置|synthetic-pretrain',
              'Tokenizer：BPE / Unigram / Byte|tokenizer-design',
              '词表大小与压缩率|vocab-compression',
            ],
          ],
        ],
      ],
      [
        '优化与并行',
        [
          [
            '优化器',
            [
              'AdamW|adamw',
              'Muon / 矩阵优化器|muon',
              '学习率：warmup、余弦、WSD|lr-schedule',
              '梯度裁剪与损失尖峰|grad-clip-loss-spike',
              '混合精度 BF16 / FP8 训练|pretrain-mixed-precision',
              '权重衰减与 μP|weight-decay-mup',
              'z-loss 与 logit 稳定|z-loss',
              'Cut cross-entropy|cut-cross-entropy',
              'NaN skip batch|nan-skip-batch',
              '批次与学习率|batch-vs-lr',
            ],
          ],
          [
            '并行',
            [
              '数据并行与 ZeRO|data-parallel-zero',
              'ZeRO-1 / 2 / 3|zero-stages',
              '张量并行|tensor-parallel',
              '流水线并行|pipeline-parallel',
              '序列并行 / 上下文并行|context-parallel',
              '专家并行|ep-pretrain',
              '3D / 5D 并行组合|nd-parallel',
              '激活重计算|activation-checkpointing',
              '通信：NCCL、NVLink、跨节点|pretrain-comm',
            ],
          ],
        ],
      ],
      [
        '扩展律',
        [
          [
            '定律与计算预算',
            [
              'Kaplan 扩展律|kaplan-scaling',
              'Chinchilla 计算最优|chinchilla',
              '过度训练与推理成本|overtraining-for-inference',
              'MoE 的扩展律|moe-scaling-laws',
              '数据约束下的扩展|data-constrained-scaling',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '微调',
    [
      [
        '参数高效',
        [
          [
            '低秩与适配器',
            [
              'LoRA|lora',
              'LoRA 秩、α 与学习率|lora-rank-alpha',
              'QLoRA|qlora',
              'DoRA|dora',
              'AdaLoRA|adalora',
              'LoRA+ / LoRA-FA|lora-plus',
              'Prefix / Prompt Tuning|prefix-tuning',
              'Adapter 层|adapters',
              'IA3|ia3',
            ],
          ],
        ],
      ],
      [
        '全参与数据',
        [
          [
            '全参 SFT',
            [
              '全参微调的学习率与批次|full-sft-hparams',
              '指令模板与 chat template|chat-template',
              '多轮对话数据格式|multiturn-format',
              '遗忘与原能力保持|sft-forgetting',
            ],
          ],
          [
            '数据配比',
            [
              '指令数据来源与清洗|sft-data-sources',
              '人写 vs 合成指令|human-vs-synthetic-instruct',
              '难度分层与课程|instruction-curriculum',
              '安全数据在 SFT 中的比例|sft-safety-mix',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '后训练与对齐',
    [
      [
        '偏好优化',
        [
          [
            '成对方法',
            [
              '奖励模型|reward-model',
              'Bradley-Terry 与成对比较|bradley-terry',
              'DPO|dpo',
              'IPO|ipo',
              'KTO|kto',
              'ORPO|orpo',
              'SimPO|simpo',
              'SLiC|slic',
              '参考模型与 β|dpo-beta-ref',
            ],
          ],
          [
            '过程与结果',
            [
              '结果监督|outcome-supervision',
              '过程监督 / PRM|process-supervision',
              '逐步验证与搜索|step-level-verify',
              '自我批判与修订|self-critique-revise',
            ],
          ],
        ],
      ],
      [
        '蒸馏与合成',
        [
          [
            '蒸馏',
            [
              '对数概率蒸馏|logit-distill',
              '序列级蒸馏|sequence-distill',
              '推理链蒸馏|cot-distill',
              '在线 vs 离线蒸馏|online-offline-distill',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '强化学习',
    [
      [
        'RLHF 经典路径',
        [
          [
            'PPO 系',
            [
              'RLHF 流程|rlhf-pipeline',
              'PPO 在语言模型中的实现|ppo-llm',
              'KL 惩罚与价值函数|ppo-kl-value',
              'Actor-Critic 稳定性|ac-stability',
              'Reject Sampling / RFT|rejection-sampling-rft',
            ],
          ],
        ],
      ],
      [
        '无奖励模型与群体方法',
        [
          [
            'GRPO 与变体',
            [
              'GRPO|grpo',
              'RLOO|rloo',
              'REINFORCE / R3|reinforce-llm',
              '优势估计不依赖 Critic|critic-free-advantage',
              '群体相对基线|group-relative-baseline',
            ],
          ],
        ],
      ],
      [
        '推理时',
        [
          [
            '搜索与计算',
            [
              'Best-of-N|best-of-n',
              'Beam Search vs Sampling|beam-vs-sample',
              'MCTS 用于语言模型|mcts-llm',
              'Test-time compute scaling|test-time-scaling',
              '过程奖励引导的搜索|prm-guided-search',
            ],
          ],
        ],
      ],
      [
        '推理模型训练',
        [
          [
            '长推理与可验证奖励',
            [
              'Long CoT SFT|long-cot-sft',
              '可验证奖励|verifiable-reward',
              '长度预算与截断|length-budget-rl',
              '推理 RL 闭环|reasoning-rl-loop',
              '过程奖励进训练环|prm-in-rl-loop',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '推理算法',
    [
      [
        'Prefill 与 Decode',
        [
          [
            '两阶段',
            [
              'Prefill 计算特征|prefill-compute',
              'Decode 的显存墙|decode-memory-wall',
              '首 token 延迟 TTFT|ttft',
              '交错延迟 TPOT / ITL|tpot-itl',
              'chunked prefill|chunked-prefill',
              'splitfuse / 混合批次|splitfuse',
            ],
          ],
        ],
      ],
      [
        '注意力内核',
        [
          [
            'Flash 系',
            [
              'FlashAttention|flashattention',
              'FlashAttention-2|flashattention-2',
              'FlashAttention-3|flashattention-3',
              'FlashDecoding|flashdecoding',
              'SageAttention|sageattention',
              'PagedAttention|paged-attention',
              'RadixAttention|radix-attention',
              'vAttention|vattention',
            ],
          ],
        ],
      ],
      [
        'KV Cache',
        [
          [
            '布局与管理',
            [
              'KV Cache 布局：BSHD / HND|kv-layout',
              '分页 KV 与块大小|paged-kv-block-size',
              '前缀缓存 / 自动前缀共享|prefix-caching',
              '多轮前缀命中|multi-turn-prefix',
              'KV 卸载到 CPU / 远端|kv-offload',
              '窗口淘汰与 sink|kv-eviction',
            ],
          ],
          [
            '压缩 KV',
            [
              'KV INT8 / FP8|kv-int8-fp8',
              'KIVI / 逐通道 KV 量化|kivi',
              'StreamingLLM 作 KV 策略|streaming-kv',
              'MLA 对 KV 的压缩|mla-kv',
            ],
          ],
        ],
      ],
      [
        '调度',
        [
          [
            '批与抢占',
            [
              '连续批处理|continuous-batching',
              '静态批 vs 动态批|static-vs-dynamic-batch',
              '抢占与公平性|preemption-fairness',
              '优先级与 SLA|priority-sla',
              'iteration-level 调度|iteration-scheduling',
            ],
          ],
        ],
      ],
      [
        '投机解码',
        [
          [
            '草稿与验证',
            [
              '投机解码原理|speculative-decoding',
              '草稿模型选择|draft-model',
              '树状验证|speculative-tree',
              'Medusa|medusa',
              'EAGLE / EAGLE-2|eagle',
              'Lookahead Decoding|lookahead-decoding',
              'MTP 多 Token 预测|mtp',
              '接受率与加速比|spec-acceptance-rate',
            ],
          ],
        ],
      ],
      [
        '并行推理',
        [
          [
            '切分',
            [
              '推理张量并行|infer-tp',
              '流水线并行推理|infer-pp',
              '专家并行推理|infer-ep',
              'PD 分离|pd-disaggregation',
              'PD 分离的 KV 传输|pd-kv-transfer',
              'Decode 实例的亲和与缓存|decode-affinity',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '推理系统',
    [
      [
        '引擎',
        [
          [
            'vLLM / SGLang',
            [
              'vLLM 架构|vllm-architecture',
              'vLLM 调度器|vllm-scheduler',
              'vLLM 与 PagedAttention|vllm-paged',
              'SGLang 与 RadixAttention|sglang',
              'SGLang 前缀树|sglang-radix-tree',
              'Structured output / 约束解码|constrained-decoding',
            ],
          ],
          [
            '其他运行时',
            [
              'TensorRT-LLM|tensorrt-llm',
              'llama.cpp / ggml|llamacpp',
              'MLC / TVM 编译|mlc-tvm',
              'Hugging Face TGI|tgi',
              'lmdeploy|lmdeploy',
              'MindIE / 昇腾推理|mindie',
            ],
          ],
        ],
      ],
      [
        '服务化',
        [
          [
            'API 与路由',
            [
              'OpenAI 兼容协议|openai-compat-api',
              '流式输出与取消|sse-cancel',
              'KV 感知路由|kv-aware-routing',
              '多 LoRA 服务|multi-lora-serving',
              'Tokenizer 与 detokenize 开销|serving-tokenizer-cost',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '硬件与集群',
    [
      [
        '超节点',
        [
          [
            '形态',
            [
              'Scale-Up 超节点 vs Scale-Out 集群|scale-up-vs-scale-out',
              '机柜作为一块逻辑加速器|rack-as-accelerator',
              'GB200 / GB300 NVL72 超节点|gb200-nvl72',
              '铜缆 spine 短距 vs 光模块长距|copper-spine-vs-optics',
              '机内全互连 vs 分层 Clos|all-to-all-vs-clos',
              '超节点内内存语义与集合通信|supernode-memory-collectives',
            ],
          ],
        ],
      ],
      [
        'GPU 与 NVIDIA',
        [
          [
            '体系',
            [
              'HBM 与算力墙|hbm-roofline',
              'A100 / H100 / Blackwell 差异|nvidia-gpu-gen',
              'Tensor Core 与 MMA|tensor-core',
              'CUDA Graph|cuda-graph',
              'MPS 与 MIG|mps-mig',
              'NVLink / NVSwitch|nvlink',
              'InfiniBand 与 GPUDirect|infiniband-gpudirect',
            ],
          ],
          [
            'Vera Rubin 超节点',
            [
              'Vera Rubin NVL72：72 GPU 一块机柜加速器|vera-rubin-nvl72',
              '六芯片共设计：GPU/CPU/交换/网卡/DPU/以太|rubin-six-chips',
              'Rubin GPU：HBM4、Transformer Engine、NVFP4|rubin-gpu-hbm4',
              'Tensor Memory Accelerator 与本地化访存|rubin-tma',
              'NVLink 6：3.6 TB/s 卡间、机柜全互连|nvlink-6',
              'NVLink counted writes 与核内融合通信|nvlink-counted-writes',
              'SHARP 在交换内做集合规约|nvlink-sharp',
              'Vera CPU：Olympus 核与 Spatial Multithreading|vera-cpu-olympus',
              'NVLink-C2C：CPU–GPU 内存一致性超芯|nvlink-c2c-superchip',
              'ConnectX-9 SuperNIC 与 Spectrum-X 向外扩展|connectx-9-spectrum-x',
              'BlueField-4：Grace + 网卡卸载基础设施|bluefield-4',
              'Spectrum-6 共封装光学以太网|spectrum-6-cpo',
              '第三代 MGX：无缆托盘与可热插拔交换|mgx-nvl72-tray',
              '45°C 温水液冷与 Intelligent Power Smoothing|rubin-liquid-power-smoothing',
              'Groq 3 LPX 低延迟推理加速卡|groq-3-lpx',
            ],
          ],
        ],
      ],
      [
        'OpenAI 自研芯片',
        [
          [
            'Jalapeño 推理加速器',
            [
              '专为 LLM 推理的空白设计（非通用 GPU）|jalapeno-inference-only',
              'OpenAI 架构 + Broadcom 实现 + TSMC 3nm|jalapeno-openai-broadcom-tsmc',
              '减少数据搬运、让利用率贴近峰值|jalapeno-data-movement',
              '脉动阵列 / 权重驻留矩阵核|jalapeno-systolic',
              '切片化：每核本地 HBM 视图 + 集合网络|jalapeno-sliced-hbm',
              'HBM4 与 2.5D 中介层近存封装|jalapeno-hbm4-2p5d',
              'MXFP4 / 低精度推理数值|jalapeno-mxfp4',
              'Tomahawk 以太：scale-up 与 MoE scale-out|jalapeno-tomahawk',
              'Celestica 板卡与机柜系统化|jalapeno-celestica',
              '用自家模型加速 ASIC 设计闭环|jalapeno-ai-assisted-design',
            ],
          ],
        ],
      ],
      [
        '昇腾与异构',
        [
          [
            'CANN',
            [
              '昇腾 910 架构要点|ascend-910',
              '达芬奇 Cube / Vector / Scalar|davinci-cube-vector',
              '910C 双 Die 共封装与片上互连|ascend-910c-dual-die',
              'CANN 图编译|cann-graph',
              '昇腾算子与落差|cann-op-gap',
              'vLLM-Ascend|vllm-ascend',
              '异构集群调度|hetero-cluster',
            ],
          ],
          [
            'CloudMatrix 超节点',
            [
              'CloudMatrix 384：384×910C + 192 鲲鹏一块超节点|cloudmatrix-384',
              '统一总线 UB 灵衢：内存语义 + 消息语义|ub-lingqu',
              'HCCS 节点内一致性到 UB 多柜 Scale-Up|hccs-to-ub',
              'UB-Mesh：递归 nD 全互连、短距电直连优先|ub-mesh',
              'L1 板载交换 + L2 通信柜、七子平面|ub-l1-l2-planes',
              '跨柜光模块把多柜收成一个逻辑节点|ub-optical-cabinets',
              '节点间带宽衰减与微秒级时延|ub-near-local-perf',
              '计算/内存/网络池化与统一编址|cloudmatrix-resource-pool',
              '超节点内 MoE 专家并行与分布式 KV|cloudmatrix-moe-kv',
              'RoCE/RDMA 做超节点之间 Scale-Out|cloudmatrix-roce-scaleout',
              '青田 DPU 与 VPC 控制面|qingtian-dpu',
              'CloudMatrix-Infer：算子融合与 AIC/AIV/SDMA 重叠|cloudmatrix-infer',
            ],
          ],
        ],
      ],
      [
        '集群调度',
        [
          [
            '编排',
            [
              'K8s 推理工作负载|k8s-inference',
              '弹性与缩容时的 KV|scale-down-kv',
              '显存碎片与装箱|gpu-packing',
              '多租户隔离|multi-tenant-gpu',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '压缩与数值',
    [
      [
        '权重量化',
        [
          [
            'PTQ',
            [
              'GPTQ|gptq',
              'AWQ|awq',
              'SmoothQuant|smoothquant',
              'GGUF / k-quants|gguf',
              'AutoRound / 其他 PTQ|other-ptq',
            ],
          ],
        ],
      ],
      [
        '激活、KV 与训练',
        [
          [
            '低精度计算',
            [
              'W8A8|w8a8',
              'W4A8 / W4A16|w4a16',
              'FP8 推理|fp8-inference',
              'QAT|qat',
              'QLoRA 作为压缩微调|qlora-as-compression',
            ],
          ],
        ],
      ],
      [
        '结构压缩',
        [
          [
            '剪枝与蒸馏',
            [
              '结构化剪枝|structured-prune',
              '非结构化稀疏|unstructured-sparsity',
              '层剪与深度压缩|layer-prune',
              '投机草稿作为压缩|draft-as-compression',
              '激活离群值为何出现|why-activation-outliers',
              '量化误差与剪枝的分工|quantization-vs-prune',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '评测',
    [
      [
        '能力',
        [
          [
            '基准',
            [
              'MMLU / CMMLU|mmlu',
              'GSM8K / MATH|math-bench',
              'HumanEval / SWE-bench|code-bench',
              '长上下文针测|niah',
              'IFEval 指令遵循|ifeval',
              'Arena / 人工偏好|lmsys-arena',
              '人工评测 vs 自动评测|human-vs-auto-eval',
              '智能体与工具评测协议|agent-eval-protocol',
              '评测方差与随机种子|eval-variance-seed',
            ],
          ],
        ],
      ],
      [
        '系统',
        [
          [
            '性能指标',
            [
              'TTFT / TPOT / 吞吐|serving-metrics',
              '有效吞吐 vs 请求吞吐|goodput',
              '显存占用分解|memory-breakdown',
              'SLO 与尾延迟|serving-slo',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '智能体与工具',
    [
      [
        '工具调用',
        [
          [
            '协议与编排',
            [
              'Function calling|function-calling',
              'JSON / 结构化输出|json-schema-decode',
              'MCP 与工具生态|mcp',
              '多步 ReAct|react',
              '规划 vs 反应式循环|plan-vs-react',
            ],
          ],
          [
            '学习与环境',
            [
              '工具 schema 训练|tool-schema-sft',
              '多步信用分配|multi-step-credit-assign',
              '工具沙箱|tool-sandbox',
              '记忆写入与检索|agent-memory-write',
              'Computer-use 循环|computer-use-loop',
            ],
          ],
        ],
      ],
      [
        '检索',
        [
          [
            'RAG',
            [
              '向量检索与切分|rag-chunking',
              '混合检索|hybrid-retrieval',
              '重排序|rerank',
              '长上下文是否取代 RAG|long-context-vs-rag',
            ],
          ],
        ],
      ],
    ],
  ],
]

const [
  trainBasics,
  archDepth,
  pretrainEng,
  finetuneEdit,
  rlDepth,
  decodeDepth,
  inferSysDepth,
  commCluster,
  compressDepth,
  evalDepth,
  alignMonitor,
  retrievalEng,
  multimodalGen,
] = llmSupplement

export const llmTree = [
  ...fromOutline([
    llmFoundations,
    representation,
    trunk[0],
    archDepth,
    multimodalGen,
    trainBasics,
    trunk[1],
    pretrainEng,
    extraDataEng,
    trunk[2],
    finetuneEdit,
    trunk[3],
    extraAlignData,
    alignMonitor,
    trunk[4],
    rlDepth,
    extraSampling,
    decodeDepth,
    trunk[5],
    trunk[6],
    inferSysDepth,
    extraCuda,
    extraServing,
    trunk[7],
    commCluster,
    trunk[8],
    compressDepth,
    trunk[9],
    extraEvalMethod,
    evalDepth,
    extraSafety,
    trunk[10],
    extraPrompting,
    retrievalEng,
  ]),
  ...markAppendix(llmSideline),
  ...markAppendix(llmPapers),
  ...markAppendix(llmAudit),
  ...markAppendix(llmFrontier),
]

