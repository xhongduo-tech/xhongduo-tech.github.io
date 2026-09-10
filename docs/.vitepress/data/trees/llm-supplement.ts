import type { Outline } from './schema'

/**
 * 大模型加深课：在 llm.ts 里按先修插进主干，不再整块挂在末尾。
 */
export const llmSupplement: Outline[] = [
  [
    '训练基础与稳定性',
    [
      [
        '反传与初始化',
        [
          [
            '训练一个 Transformer 之前',
            [
              '注意力的反向传播|attention-backward',
              '自动微分与计算图|autograd-graph',
              'Xavier / Kaiming 初始化|xavier-kaiming-init',
              '深度缩放初始化|depth-scaled-init',
              '嵌入初始化与尺度|embedding-init-scale',
              '输出层初始化与 logit 尺度|output-init-logit-scale',
              'Dropout 在 Transformer|dropout-transformer',
              '注意力 dropout 与 drop-path|attention-dropout-droppath',
              '标签平滑|label-smoothing-lm',
              '困惑度与 bits-per-byte|perplexity-bpb',
              '损失曲线的阶段|loss-curve-phases',
            ],
          ],
        ],
      ],
      [
        '稳定性',
        [
          [
            '训练为什么会炸',
            [
              '注意力 logit 增长|attention-logit-growth',
              'logit soft-capping|logit-soft-capping',
              '输出 logit 发散|output-logit-divergence',
              '梯度范数监控|grad-norm-monitoring',
              '激活尺度漂移|activation-scale-drift',
              'Post-LN 预热|post-ln-warmup',
              'Adam ε 与更新尺度|adam-epsilon-update-scale',
              'β₂ 与损失尖峰|adam-beta2-spikes',
              '学习率敏感度与 μP 实践|lr-sensitivity-mup-practice',
              '权重衰减的豁免：嵌入与 norm|weight-decay-exemptions',
              'EMA 与检查点平均|ema-checkpoint-averaging',
              '模型汤与权重平均|model-soups-averaging',
              '训练确定性|training-determinism',
              '静默数据损坏 SDC|silent-data-corruption-training',
            ],
          ],
        ],
      ],
      [
        '缩放实践',
        [
          [
            '形状、批次与生长',
            [
              '深度对宽度|depth-vs-width',
              '词表规模缩放律|vocab-scaling-law',
              '学习率与批次缩放律|lr-batch-scaling-law',
              '梯度噪声尺度与临界批次|gradient-noise-scale',
              '序列长度预热|sequence-length-warmup',
              '多 epoch 与重复数据|multi-epoch-repetition',
              '拟合缩放律的方法|scaling-law-fitting',
              '涌现能力争论|emergence-debate',
              '推理算力最优|inference-compute-optimal',
              '小模型代理实验|proxy-model-experiments',
              '模型生长与深度扩展|model-growth-depth-upscaling',
              '稠密升级到 MoE|dense-to-moe-upcycling',
              '模型合并：SLERP / TIES / DARE|model-merging',
              '任务向量算术|task-arithmetic',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '编码器、状态空间与专家',
    [
      [
        '编码器与编解码',
        [
          [
            '纯解码器之外',
            [
              'BERT 编码器结构|bert-encoder',
              'T5 相对位置偏置|t5-relative-bias',
              'BART 与去噪编解码|bart-denoising',
              '编码器-解码器对纯解码器|encdec-vs-decoder-only',
              '双向注意力与前缀掩码|bidirectional-prefix-mask',
              '嵌入模型的池化|embedding-pooling',
            ],
          ],
        ],
      ],
      [
        '注意力续',
        [
          [
            '门、共享与深度',
            [
              '门控注意力|gated-attention',
              'sigmoid 注意力|sigmoid-attention',
              'softmax 替代与 softpick|softmax-alternatives',
              '跨层 KV 共享 CLA|cross-layer-kv-sharing',
              'YOCO|yoco',
              '多 token 注意力|multi-token-attention',
              '头维度对头数|head-dim-vs-heads',
              '注意力长度缩放|attention-length-scaling',
              '注意力头的功能分化|attention-head-roles',
              'Universal Transformer 与循环深度|universal-transformer-recurrence',
              '循环 Transformer|looped-transformer',
              '早退与自适应计算|early-exit-adaptive',
              'ALBERT 参数共享|albert-param-sharing',
            ],
          ],
        ],
      ],
      [
        'SSM 与线性续',
        [
          [
            '状态、扫描与召回',
            [
              'HiPPO 与 S4|hippo-s4',
              '选择性扫描|selective-scan',
              '硬件感知扫描|hardware-aware-scan',
              '分块并行形式|chunked-parallel-form',
              '状态空间对偶 SSD|state-space-duality',
              '门控线性注意力 GLA|gated-linear-attention',
              'HGRN|hgrn',
              '增量规则与关联记忆|delta-rule-memory',
              'TTT 测试时训练层|ttt-layers',
              '长卷积|long-convolution',
              '混合比与层排布|hybrid-layer-ratio',
              '线性模型的召回瓶颈|linear-recall-bottleneck',
              '状态大小对上下文|state-size-vs-context',
            ],
          ],
        ],
      ],
      [
        'MoE 进阶',
        [
          [
            '路由与专家',
            [
              '哈希路由|hash-routing',
              'Soft MoE|soft-moe',
              '无辅助损失均衡|aux-loss-free-balancing',
              '节点受限路由|node-limited-routing',
              '序列级均衡|sequence-level-balance',
              '专家专精化分析|expert-specialization',
              'MoE 微调|moe-finetuning',
              '专家剪枝与合并|expert-pruning-merging',
              'PEER 百万专家|peer-million-experts',
              '层次 MoE|hierarchical-moe',
              'MoE 的通信-计算比|moe-comm-compute-ratio',
              '路由抖动|routing-flapping',
              '注意力 MoE|attention-moe',
              '专家粒度与数量律|expert-granularity-law',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '词表、配比与并行调度',
    [
      [
        '分词器工程',
        [
          [
            '词表的细节',
            [
              '预分词正则|pretokenization-regex',
              '字节回退|byte-fallback',
              '数字切分|digit-splitting',
              '特殊 token 与聊天 token|special-tokens',
              '词表扩展与多语言|vocab-extension-multilingual',
              '分词器训练语料|tokenizer-training-corpus',
              '分词 fertility 与公平|tokenizer-fertility',
              'glitch token|glitch-tokens',
            ],
          ],
        ],
      ],
      [
        '数据配比与选择',
        [
          [
            '选什么、按什么比',
            [
              'DoReMi|doremi',
              'RegMix|regmix',
              '影响函数选数据|influence-data-selection',
              '多语言 α 采样|multilingual-alpha-sampling',
              '仓库级代码排布|repo-level-code-data',
              'FIM 训练目标|fim-training',
              '长文档上采样|long-doc-upsampling',
              '指令数据进预训练|instruction-in-pretrain',
              '教科书式合成|textbook-synthetic',
              '评测去污染流水线|decontamination-pipeline',
              '数据排序与课程|data-ordering-curriculum',
              '退火阶段的配比|anneal-mixture',
            ],
          ],
        ],
      ],
      [
        '并行进阶',
        [
          [
            '调度、重叠与容错',
            [
              '1F1B 与交错流水|1f1b-interleaved',
              '零气泡流水|zero-bubble-pipeline',
              'DualPipe|dualpipe',
              '计算通信重叠|comp-comm-overlap',
              'DeepSpeed Ulysses|ulysses-sp',
              '梯度累积与微批|grad-accum-microbatch',
              'MoE 训练 all-to-all|moe-training-all2all',
              '异步检查点|async-checkpointing',
              '分布式检查点格式|distributed-ckpt-format',
              '弹性训练与掉卡|elastic-training',
              '落后者缓解|straggler-mitigation',
              '训练重放与可复现|training-replay',
              '训练中评测与早期信号|in-training-eval',
              '数据加载流水线|dataloader-pipeline',
              '拓扑感知放置|topology-aware-placement',
              '万卡训练的故障率|large-scale-failure-rate',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    'SFT、适配与编辑',
    [
      [
        'SFT 工程',
        [
          [
            '数据、损失与预算',
            [
              'FLAN / Alpaca / ShareGPT 谱系|instruction-data-lineage',
              '仅回复损失|response-only-loss',
              'SFT 打包与掩码|sft-packing-mask',
              'NEFTune|neftune',
              '学习率：LoRA 对全参|lora-vs-full-lr',
              '长上下文微调|long-context-finetune',
              '领域适配微调|domain-adaptation-ft',
              'SFT 对 RL 的数据效率|sft-vs-rl-efficiency',
              '拒绝采样数据循环|rft-data-loop',
            ],
          ],
        ],
      ],
      [
        'PEFT 续',
        [
          [
            '低秩之外',
            [
              'rsLoRA|rslora',
              'VeRA|vera',
              'GaLore|galore',
              'LISA|lisa-layerwise',
              'ReLoRA|relora',
              'BitFit|bitfit',
              'P-tuning v2|p-tuning-v2',
              'LoRA 合并与冲突|lora-merge-conflict',
              'MoE 上的 LoRA|lora-on-moe',
              'LoRA 的内在秩|lora-intrinsic-rank',
            ],
          ],
        ],
      ],
      [
        '编辑与遗忘',
        [
          [
            '改一处而不动其他',
            [
              'ROME / MEMIT|rome-memit',
              '知识注入的限度|knowledge-injection-limits',
              '机器遗忘|machine-unlearning',
              'RMU 与 WMDP|rmu-wmdp',
              '持续学习与回放|continual-replay',
              '遗忘的度量|forgetting-metrics',
              '负任务向量|negative-task-vector',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '奖励、自博弈与推理行为',
    [
      [
        '奖励模型进阶',
        [
          [
            '奖励从哪来',
            [
              'RM 集成与不确定性|rm-ensembles',
              'RM 校准|rm-calibration',
              'RM 过优化与 Goodhart|rm-overoptimization',
              '生成式奖励模型|generative-rm',
              'LLM 裁判作奖励|llm-judge-reward',
              '评分细则奖励|rubric-reward',
              '代码验证器：单元测试|code-verifier-unittests',
              '数学验证器与等价|math-verifier-equivalence',
              '多目标奖励加权|multi-objective-reward',
              'token 级奖励塑形|token-level-reward',
            ],
          ],
        ],
      ],
      [
        'PPO / GRPO 细节',
        [
          [
            '超参与偏差',
            [
              'GAE 与 λ|gae-lambda',
              'clip 与 clip-higher|ppo-clip-higher',
              '价值模型初始化|value-model-init',
              '奖励归一化与白化|reward-whitening',
              '长度归一化偏差|length-normalization-bias',
              '熵塌缩与熵正则|entropy-collapse',
              '过长过滤|overlong-filtering',
              '动态采样|dynamic-sampling-rl',
              '无 KL 的 RL|kl-free-rl',
              '截断重要性采样|truncated-importance-sampling',
              '离策略校正|off-policy-correction',
              'rollout 引擎与权重同步|rollout-weight-sync',
              '采样温度与探索|rl-sampling-temperature',
              '难度课程|rl-difficulty-curriculum',
              '训练-推理精度不匹配|train-infer-mismatch',
            ],
          ],
        ],
      ],
      [
        '自博弈与监督扩展',
        [
          [
            '超越人类标注',
            [
              '迭代与在线 DPO|iterative-online-dpo',
              '自奖励语言模型|self-rewarding-lm',
              'Nash 学习|nash-learning-hf',
              '自博弈|self-play-lm',
              '辩论|debate-oversight',
              '弱到强泛化|weak-to-strong',
              '可扩展监督|scalable-oversight',
              '宪法式 RL 实践|constitutional-rl-practice',
              '拒绝训练与边界|refusal-training',
              '诚实与校准训练|honesty-calibration-training',
              '多模态 RLHF|multimodal-rlhf',
              '代码 RL 与 SWE 环境|swe-rl-env',
              '工具整合推理 RL|tool-integrated-rl',
              'RL 缩放律|rl-scaling-laws',
            ],
          ],
        ],
      ],
      [
        '推理模型行为',
        [
          [
            '想多久、想什么',
            [
              '思考预算与自适应思考|thinking-budget-adaptive',
              '混合思考模式|hybrid-thinking-mode',
              '过度思考|overthinking',
              '推理长度控制|reasoning-length-control',
              '自我验证与回溯|self-verification-backtrack',
              '并行思考与聚合|parallel-thinking',
              '推理蒸馏到小模型|reasoning-distill-small',
              'aha moment|aha-moment',
              '语言混杂|language-mixing-reasoning',
              '推理的忠实性|reasoning-faithfulness',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '多样与约束解码',
    [
      [
        '多样与质量',
        [
          [
            '采样之外',
            [
              '长度惩罚与退化|length-penalty-degeneration',
              '多样束搜索|diverse-beam-search',
              'MBR 解码|mbr-decoding',
              'DoLa 层对比|dola-decoding',
              '上下文感知解码|context-aware-decoding',
              'n-gram 阻断|ngram-blocking',
              'logit bias|logit-bias',
              '温度对推理的影响|temperature-reasoning',
              '自投机解码|self-speculative-decoding',
              'Jacobi 并行解码|jacobi-decoding',
              '流式 detokenize 边界|streaming-detokenize',
              '采样器内核|sampler-kernel',
              '结构化输出的开销|structured-output-overhead',
              '多样本聚合与投票|multi-sample-aggregation',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '性能会计与编译',
    [
      [
        '性能会计',
        [
          [
            '算清楚再优化',
            [
              'KV cache 大小计算|kv-cache-size-math',
              '解码的算术强度|arithmetic-intensity-decode',
              '批大小与 roofline 拐点|batch-roofline-knee',
              '每 token 能耗|energy-per-token',
              '服务成本模型|serving-cost-model',
              '长上下文的显存曲线|long-context-memory-curve',
            ],
          ],
        ],
      ],
      [
        '编译与内核',
        [
          [
            '引擎里的细节',
            [
              'torch.compile 与 Inductor|torch-compile-inductor',
              '内核自动调优|kernel-autotuning',
              '自定义 allreduce|custom-allreduce',
              'TP 通信重叠|tp-comm-overlap',
              'NCCL 调优|nccl-tuning',
              'MoE 推理批处理|moe-inference-batching',
              '分词并行与预处理|tokenizer-parallel',
              '多模态编码器流水|vision-encoder-pipeline',
              '嵌入与重排模型服务|embedding-serving',
              '权重加载与流式|weight-loading-streaming',
              'safetensors|safetensors-format',
            ],
          ],
        ],
      ],
      [
        '端侧与替代运行时',
        [
          [
            'GPU 之外',
            [
              'ONNX Runtime|onnx-runtime',
              'OpenVINO|openvino',
              'Apple MLX|mlx-apple',
              'WebGPU 推理|webgpu-inference',
              '移动端 NPU 部署|mobile-npu-deploy',
              'CPU 推理与量化|cpu-inference-quant',
              '端云协同|edge-cloud-split',
              '端侧投机解码|on-device-speculative',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '集合通信与异构加速',
    [
      [
        '集合通信',
        [
          [
            '算法与拓扑',
            [
              'Ring / Tree allreduce|ring-tree-allreduce',
              'reduce-scatter 与 all-gather|reduce-scatter-allgather',
              'all-to-all 实现|all-to-all-impl',
              'NCCL 算法选择|nccl-algorithms',
              '集合通信带宽模型|collective-bandwidth-model',
              'rail-optimized 拓扑|rail-optimized-topology',
              'Dragonfly 与 torus 对照|dragonfly-torus',
              '训练网络的拥塞控制|training-network-congestion',
              '梯度压缩|gradient-compression',
              '参数服务器对照|parameter-server-contrast',
            ],
          ],
          [
            '集群运行',
            [
              '检查点 I/O 带宽|checkpoint-io-bandwidth',
              '数据集存储与流式加载|dataset-storage-streaming',
              '机架功率密度|rack-power-density',
              '液冷与热设计|liquid-cooling-thermal',
              'GPU 故障模式|gpu-failure-modes',
              '机群公平调度|cluster-fair-scheduling',
            ],
          ],
        ],
      ],
      [
        '其他加速器',
        [
          [
            'NVIDIA 之外',
            [
              'TPU 架构基本|tpu-architecture-basics',
              'Cerebras 晶圆级|cerebras-wafer-scale',
              'Groq LPU 确定性|groq-lpu-deterministic',
              'SambaNova 数据流|sambanova-dataflow',
              'Tenstorrent|tenstorrent',
              'AMD ROCm 栈|amd-rocm-stack',
              '国产加速器谱系|chinese-accelerators',
              '加速器对比方法|accelerator-comparison-method',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '量化基础与稀疏',
    [
      [
        '量化基础',
        [
          [
            'PTQ 之前',
            [
              '均匀与非均匀量化|uniform-nonuniform-quant',
              '对称与非对称|symmetric-asymmetric-quant',
              'per-tensor / channel / group|quant-granularity',
              '校准集|calibration-set',
              'AdaRound|adaround',
              'Hessian 感知量化|hessian-aware-quant',
              '混合精度搜索|mixed-precision-search',
              'NF4 与分位量化|nf4-quantile',
              'E4M3 / E5M2|fp8-formats',
              '量化误差度量|quant-error-metrics',
              '量化后退化模式|quant-eval-degradation',
            ],
          ],
        ],
      ],
      [
        '稀疏与结构',
        [
          [
            '剪、分解与压缩输入',
            [
              'SparseGPT|sparsegpt',
              'Wanda|wanda',
              '2:4 结构稀疏|two-four-sparsity',
              'Sheared LLaMA|sheared-llama',
              '激活稀疏与 ReLU²|activation-sparsity',
              'Deja Vu 上下文稀疏|deja-vu-sparsity',
              '注意力头剪枝|attention-head-pruning',
              '低秩分解|low-rank-factorization',
              '张量分解|tensor-decomposition',
              '提示压缩 LLMLingua|prompt-compression',
              '词表裁剪|vocab-pruning',
              '权重共享压缩|weight-sharing-compress',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '评测协议与裁判',
    [
      [
        '协议',
        [
          [
            '数字怎么来',
            [
              '困惑度评测的陷阱|perplexity-eval-pitfalls',
              '对数似然对生成式评测|loglik-vs-generative-eval',
              '提示格式敏感|prompt-format-sensitivity',
              '答案抽取与归一化|answer-extraction',
              'pass@k|pass-at-k',
              '校准与 ECE|calibration-ece',
              'TruthfulQA 与幻觉基准|truthfulqa-hallucination',
              '多语言基准|multilingual-benchmarks',
              '评测框架 lm-eval / OpenCompass|eval-harness',
              '基准饱和|benchmark-saturation',
              '模型卡与报告|model-cards',
            ],
          ],
        ],
      ],
      [
        '裁判与人评',
        [
          [
            '谁来判',
            [
              'LLM 裁判偏差|llm-judge-bias',
              '成对对逐点|pairwise-vs-pointwise',
              'Elo 与 BT 评分|elo-bt-rating',
              '风格控制|style-control',
              '自我偏好|self-preference-bias',
              '人类评测协议|human-eval-protocol',
              '危险能力评测|dangerous-capability-eval',
              '红队评测|red-team-eval',
              '偏见与毒性基准|bias-toxicity-benchmarks',
              '能力引出|capability-elicitation',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '对齐概念与监控',
    [
      [
        '概念与监控',
        [
          [
            '对齐失效的名字',
            [
              '规约博弈|specification-gaming',
              '目标误泛化|goal-misgeneralization',
              '欺骗性对齐|deceptive-alignment',
              '谋划评测|scheming-evals',
              'sandbagging|sandbagging',
              '情境意识|situational-awareness',
              '潜伏代理|sleeper-agents',
              '忠实思维链|faithful-cot',
              '幻觉分类|hallucination-taxonomy',
              '弃权与校准拒答|abstention-calibrated-refusal',
              '记忆化与版权|memorization-copyright',
              'DP-SGD|dp-sgd',
              '模型提取攻击|model-extraction-attack',
              '模型指纹|model-fingerprinting',
              '安全等级与 RSP|safety-levels-rsp',
              '部署后监控|deployment-monitoring',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '嵌入训练与检索工程',
    [
      [
        '嵌入训练',
        [
          [
            '把文本变成向量',
            [
              '对比学习与 InfoNCE|contrastive-infonce',
              '难负例挖掘|hard-negatives',
              'Matryoshka 表示|matryoshka-embeddings',
              '指令嵌入|instruction-embeddings',
              '多向量与晚交互|multi-vector-late-interaction',
              'BM25 与稀疏检索|bm25-sparse',
              'SPLADE|splade',
              'MTEB|mteb',
              '长文档嵌入|long-doc-embedding',
              '多模态嵌入|multimodal-embeddings',
            ],
          ],
        ],
      ],
      [
        'RAG 进阶',
        [
          [
            '检索的工程',
            [
              '查询改写与 HyDE|query-rewrite-hyde',
              '多跳检索|multi-hop-retrieval',
              '引用与归因|citation-attribution',
              'RAG 评测 RAGAS|rag-evaluation',
              '自适应 RAG|adaptive-rag',
              'RAG 中的上下文压缩|rag-context-compression',
              'HNSW / IVF|hnsw-ivf',
              'PQ 向量索引|pq-vector-index',
              'RAG 对微调|rag-vs-finetune',
              '检索投毒防御|retrieval-poisoning-defense',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '多模态生成与统一模型',
    [
      [
        '视觉 tokenizer 与理解',
        [
          [
            '像素怎么进模型',
            [
              'VQ-VAE|vq-vae',
              'VQGAN|vqgan',
              'FSQ / LFQ|fsq-lfq',
              '连续对离散视觉 token|continuous-vs-discrete-vision-tokens',
              '原生分辨率 ViT|native-resolution-vit',
              '2D RoPE|2d-rope',
              '视觉指令数据|visual-instruction-data',
              'VLM 幻觉与 POPE|vlm-hallucination',
              '定位与指代|grounding-referring',
              'GUI 定位|gui-grounding',
              '图表与表格理解|chart-table-understanding',
              '长视频理解|long-video-understanding',
              '多模态推理|multimodal-reasoning',
            ],
          ],
        ],
      ],
      [
        '生成基础',
        [
          [
            '扩散、流与自回归',
            [
              'DDPM|ddpm',
              '流匹配|flow-matching',
              '潜空间与 VAE|latent-vae',
              'CFG 在图像生成|cfg-image',
              '自回归图像生成|autoregressive-image-gen',
              'VAR 尺度自回归|var-next-scale',
              'MAR 掩码自回归|mar-masked-ar',
              '统一理解与生成 Janus / Show-o|unified-understanding-generation',
              'Transfusion|transfusion',
              '扩散蒸馏与少步|diffusion-distillation',
              '视频生成的时空建模|video-gen-spatiotemporal',
            ],
          ],
        ],
      ],
      [
        '语音与全模态',
        [
          [
            '说与听',
            [
              'VALL-E 式 TTS|valle-tts',
              '全双工语音 Moshi|full-duplex-speech',
              '端到端语音对话|e2e-speech-dialogue',
              '音乐生成|music-generation',
              'Omni 模型对齐|omni-model-alignment',
              '音画同步|audio-visual-sync',
            ],
          ],
        ],
      ],
    ],
  ],
]
