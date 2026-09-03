import { fromOutline, type Outline } from './schema'
import { llmExtra } from './llm-extra'
import { llmPapers } from './llm-papers'
import { llmAudit } from './llm-audit'
import { llmFrontier } from './llm-frontier'

const outline: Outline[] = [
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
            ],
          ],
          [
            'Qwen OCR 与文档解析',
            [
              'Naive Dynamic Resolution 原生分辨率切块|qwen-vl-naive-dynamic-res',
              '2×2 patch merge 控制视觉 token 数|qwen-vl-patch-merge',
              '窗口注意力与周期性全局注意力交替|qwen-vl-window-full-attn',
              'MRoPE / Interleaved MRoPE 时空位置|qwen3-vl-interleaved-mrope',
              'DeepStack：多层 ViT 特征注入 LLM|qwen3-vl-deepstack',
              'SigLIP-2 视觉骨干|qwen3-vl-siglip2',
              'Qwen HTML 版式感知文档解析|qwen-html-document-parse',
              '文字定位与 2D grounding|qwen-ocr-text-grounding',
              '表格、公式与卡证关键信息抽取|qwen-ocr-kie',
              '粗到细伪标注 OCR 数据管线|qwen-ocr-coarse-to-fine',
              '多页 PDF 合成与跨页文档 VQA|qwen-ocr-long-pdf',
              '图像旋转矫正|qwen-ocr-rotation',
              'Qwen-VL-OCR 内置任务模板|qwen-vl-ocr-tasks',
              'Qwen3.5-OCR：原生 PDF 与多轮抽取|qwen35-ocr',
            ],
          ],
          [
            '语音与视频',
            [
              '语音 tokenizer|speech-tokenizer',
              '音频语言模型|audio-lm',
              '视频 token 与时间采样|video-tokens',
              '多模态交错训练|interleaved-multimodal',
            ],
          ],
          [
            'Qwen ASR',
            [
              'LALM：先理解音频再生成转写|qwen3-asr-lalm',
              'Qwen3-Omni 作为语音理解基座|qwen3-omni-speech-base',
              'AuT：AED 音频 Transformer 编码器|qwen3-asr-aut',
              '128 维 Fbank 与 Conv2D 8× 下采样|qwen3-asr-fbank-downsample',
              '12.5 Hz 音频 token 率|qwen3-asr-token-rate',
              '动态 FlashAttention 窗口 1s–8s|qwen3-asr-dynamic-window',
              '分块 Conv2D（约 100 帧 → 13 token）|qwen3-asr-chunked-conv',
              '学习型 projector 对齐 AuT 与 Qwen3|qwen3-asr-projector',
              'Qwen3 解码器：GQA、RoPE、QK-Norm|qwen3-asr-decoder',
              '流式与离线统一推理|qwen3-asr-streaming-offline',
              '语言识别与 52 语种/方言|qwen3-asr-lid',
              'Qwen3-ForcedAligner 非自回归时间戳|qwen3-forced-aligner',
              '伪标注大规模语音预训练|qwen3-asr-pseudo-label',
              'vLLM 批推理与流式 ASR 服务|qwen3-asr-vllm',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '世界模型与空间智能',
    [
      [
        '李飞飞 / World Labs',
        [
          [
            '空间智能纲领',
            [
              '空间智能：感知、推理、在三维中行动|spatial-intelligence',
              '世界模型四件事：重建、生成、仿真、交互|world-model-four-roles',
              '持久三维世界 vs 边走边生成帧|persistent-3d-vs-streaming-frames',
            ],
          ],
          [
            'Marble 生成式世界',
            [
              '多模态提示到三维世界（文/图/视频/布局）|marble-multimodal-prompt',
              '多图与视频的视角拼接成一致场景|marble-multi-view-stitch',
              'Chisel：粗几何定结构、文本定风格|marble-chisel',
              '区域扩展与 Composer 拼世界大图|marble-expand-compose',
              '三维高斯溅射作为高保真表示|marble-gaussian-splats',
              '碰撞网格与视觉网格双导出|marble-dual-mesh',
              'Spark：浏览器高斯溅射渲染|marble-spark',
              '结构保持的视频增强与动态元素|marble-video-enhance',
              'AI 原生局部编辑与风格改写|marble-world-edit',
            ],
          ],
          [
            'RTFM 实时帧模型',
            [
              'RTFM：探索时实时出帧而非导出场景|worldlabs-rtfm',
              '实时世界模型的形变与不一致性|rtfm-morphing',
            ],
          ],
          [
            'Atlas Omni 世界模型',
            [
              '多模态自回归扩散 Transformer|atlas-ardt',
              '共享空间上下文：图像锚定在三维位姿|atlas-spatial-context',
              '相机位姿作为原生输入而非文本描述|atlas-native-camera',
              '视频作为带位姿的图像序列|atlas-video-as-frames',
              'Rectified flow 潜空间扩散|atlas-rectified-flow',
              '深度图、点云与高斯溅射写出|atlas-3d-writeout',
              '稀疏视角新视角合成与三维重建|atlas-sparse-view-recon',
              '相机可控长视频（至 1440p / 1 分钟）|atlas-camera-controlled-video',
              '多机位 reframing 与子弹时间|atlas-video-reframe',
              'Real-to-Sim：重建场景并生成机器人传感器视图|atlas-real-to-sim',
              '沿用 LLM 的 KV cache 与分离式服务|atlas-llm-serving-tricks',
              '扩散蒸馏、CFG 与 VAE 潜空间|atlas-diffusion-stack',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '模型族',
    [
      [
        'Dense 开源',
        [
          [
            'Llama 系',
            [
              'Llama 1 架构选择|llama-1',
              'Llama 2 GQA 与对话|llama-2',
              'Llama 3 数据与 tokenizer|llama-3',
              'Llama 3.1 长上下文|llama-3-1',
              'Llama 4 与 MoE 方向|llama-4',
              'Code Llama|code-llama',
            ],
          ],
          [
            'Qwen / GLM / Gemma / Mistral',
            [
              'Qwen 1.5 / 2 / 2.5 演进|qwen-evolution',
              'Qwen3|qwen3',
              'Qwen2-Audio / Qwen2.5-Omni 语音|qwen-audio-omni',
              'Qwen3-ASR-1.7B / 0.6B|qwen3-asr',
              'Qwen3-VL 与文档 OCR|qwen3-vl',
              'Qwen3.5-OCR|qwen35-ocr-model',
              'GLM 与 ChatGLM|glm',
              'GLM-4|glm-4',
              'Gemma / Gemma 2|gemma',
              'Mistral 与 Mixtral|mistral-mixtral',
              'Phi 小模型路线|phi',
            ],
          ],
        ],
      ],
      [
        'MoE 与推理导向',
        [
          [
            'DeepSeek 系',
            [
              'DeepSeek-V2 MLA 与 MoE|deepseek-v2',
              'DeepSeek-V3|deepseek-v3',
              'DeepSeek-R1 与推理时行为|deepseek-r1',
              'DeepSeek 开源栈与部署约束|deepseek-serving',
            ],
          ],
          [
            '其他 MoE',
            [
              'Mixtral 8x7B / 8x22B|mixtral',
              'DBRX|dbrx',
              'Grok MoE 公开信息|grok-moe',
              'OLMoE|olmoe',
            ],
          ],
        ],
      ],
      [
        '小模型与端侧',
        [
          [
            '结构与蒸馏',
            [
              'SLM 的能力边界|slm-capability',
              '端侧上下文与 KV 预算|on-device-kv',
              'NPU 友好算子|npu-friendly-ops',
              '蒸馏到端侧的数据与温度|on-device-distill',
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
    '半导体与光刻',
    [
      [
        '光刻物理',
        [
          [
            '分辨率',
            [
              '瑞利判据：CD = k₁ λ / NA|rayleigh-litho',
              '波长台阶：g/i 线、KrF 248、ArF 193、EUV 13.5|litho-wavelengths',
              '掩模、光刻胶、曝光、显影、刻蚀转印|litho-process-flow',
              '套刻 Overlay 与对准|litho-overlay',
              '双工件台 Twinscan 提高产能|twinscan-dual-stage',
            ],
          ],
        ],
      ],
      [
        'DUV 与多重曝光',
        [
          [
            '浸没与分解',
            [
              'ArF 浸没：水作介质抬高 NA|arf-immersion',
              '离轴照明与偶极/四极光瞳|off-axis-illumination',
              '相移掩模 PSM|phase-shift-mask',
              'LELE 多次曝光套刻|lele-multipattern',
              'SADP / SAQP 自对准双重/四重图形|sadp-saqp',
              '浸没 DUV + 多重曝光走到 7/5 nm 的代价|duv-multipattern-cost',
            ],
          ],
        ],
      ],
      [
        'EUV 与 ASML',
        [
          [
            '光源与光学',
            [
              '13.5 nm 真空全反射：Mo/Si 多层膜镜|euv-multilayer-mirror',
              'LPP：CO₂ 激光打锡滴产生等离子体|euv-lpp-tin',
              '预脉冲 + 主脉冲提高转换效率|euv-prepulse',
              '蔡司投影物镜与收集镜|zeiss-euv-optics',
              '氢气流 Dynamic Gas Lock 防污染|euv-hydrogen-dgl',
              'EUV 薄膜 Pellicle|euv-pellicle',
              'NXE：NA 0.33 量产 5/3 nm|asml-nxe',
              'High-NA 0.55 EXE：变形光学与半场|asml-high-na',
              '真空磁浮工件台|euv-maglev-stage',
              '随机效应与光子散粒噪声|euv-stochastics',
              '金属氧化物胶 vs 化学放大胶|euv-resist',
            ],
          ],
        ],
      ],
      [
        '计算光刻',
        [
          [
            '图形修正',
            [
              '光学邻近修正 OPC|opc',
              '光源掩模协同优化 SMO|smo',
              '逆光刻 ILT 与曲线掩模|ilt-curvilinear',
              '多束电子束写掩模|multibeam-mask-writer',
              'GPU / AI 加速 OPC（cuLitho 等）|computational-litho-gpu',
            ],
          ],
        ],
      ],
      [
        '制程与封装',
        [
          [
            '前后道',
            [
              'FinFET 到 GAA / nanosheet|finfet-gaa',
              '原子层沉积 ALD 与原子层刻蚀|ald-ale',
              'HBM 堆叠与混合键合|hbm-hybrid-bonding',
              'CoWoS / 2.5D 中介层|cowos-2p5d',
              'Chiplet 与先进封装补光刻极限|chiplet-packaging',
            ],
          ],
        ],
      ],
      [
        '国产与管制',
        [
          [
            '设备与供应链',
            [
              'EUV 出口管制卡住先进逻辑|euv-export-control',
              '国产浸没 DUV 与 28 nm 单次曝光|china-immersion-duv',
              '多重曝光把国产 DUV 往更先进节点推|china-duv-multipattern',
              '国产 EUV 仍处原型、光学与光源是瓶颈|china-euv-prototype',
              '光刻胶、光源、镜头的国产替代|china-litho-supply-chain',
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

export const llmTree = [...fromOutline(outline), ...llmExtra, ...llmPapers, ...llmAudit, ...llmFrontier]

