#!/usr/bin/env python3

import argparse
import json
import re
import unicodedata
from collections import defaultdict, deque
from pathlib import Path


BATCH_PRIORITY = {
    "orig": 0,
    "c": 1,
    "d": 1,
    "ab": 2,
    "f": 3,
    "e": 4,
    "g": 5,
}

DIRECT_BLOG_CATEGORIES = {
    "理论基础",
    "前沿追踪",
    "模型训练",
    "模型微调",
    "模型部署",
    "智能体",
    "系统基础",
    "工程实践",
}

QUEUE_TO_BLOG_CATEGORY = {
    "数学基础": "理论基础",
    "NLP基础": "理论基础",
    "模型架构": "理论基础",
    "多模态": "前沿追踪",
    "数据工程": "工程实践",
    "评测与基准": "工程实践",
    "推荐系统": "工程实践",
    "知识图谱": "工程实践",
    "强化学习": "模型训练",
}

GENERIC_BRIEF_PATTERNS = [
    "围绕“",
    "围绕'",
    "主题要求给出可验证技术结论",
    "主题要求输出可验证技术信息",
    "重点分析它改变了哪些计算路径",
    "不只需要会调用，还要能从基础算子实现并验证正确性",
]

GENERIC_DEPTH_PATTERNS = [
    "执行“推导 + 实现 + 对比 + 压测”",
    "执行“推导-实现-对比-压测”",
    "执行“推导 + 实现 + 对比 + 压测”闭环",
    "先推导",
    "围绕“",
    "围绕 ",
]

BRIEF_TEMPLATE = {
    "理论基础": "{core} 需要从形式化定义、关键公式与可证明性质三层讲清，说明它如何影响模型表达能力、误差界或信息流，并比较相近方法在假设条件和适用范围上的差异。",
    "前沿追踪": "{core} 需要交代它相对主流方案到底改动了哪条技术路径，并用公开结果或可复现实验比较收益、成本、数据需求和能力边界，而不是只列名词。",
    "模型训练": "{core} 要写清核心更新公式、关键超参数和稳定性信号，比较不同配置对收敛速度、最终指标、显存与吞吐的影响，并说明何时会出现发散或退化。",
    "模型微调": "{core} 需要围绕参数更新路径、数据格式和效果-成本权衡展开，说明它在 SFT、偏好优化或 PEFT 场景中的适用条件，以及最常见的性能退化来源。",
    "模型部署": "{core} 应围绕延迟、吞吐、显存和实现复杂度展开分析，说明它改变了哪一段推理链路、收益来自哪里，以及在哪些负载和上下文长度下不再划算。",
    "智能体": "{core} 需要拆成规划、工具调用、状态管理和失败恢复几个环节讨论，并给出成功率、调用延迟、token 成本或人工接管率等可验证指标。",
    "系统基础": "{core} 需要说明数据路径、通信或存储成本如何变化，并比较不同系统方案在一致性、吞吐、尾延迟和故障恢复上的边界条件。",
    "工程实践": "{core} 要放进完整工程流程中讨论，写清输入输出、评测口径、回归风险与上线门槛，避免只停留在工具清单或经验口号。",
}

DEPTH_TEMPLATE = {
    "理论基础": "先统一符号、定义与假设，再完成核心推导或证明草图；随后用最小数值实验、可视化或反例验证结论，最后比较至少两种相近方法在复杂度、误差界或稳定性上的差异。",
    "前沿追踪": "先说明与基线模型的关键差分，再复现一组可对照实验或公开结果，最后从性能、成本、数据需求和工程可落地性四个维度总结它是否值得采用。",
    "模型训练": "实现最小训练脚本并记录 loss、梯度范数、学习率、显存和 tokens/s；比较两组关键超参数或调度策略，定位最容易导致训练不稳或泛化退化的触发条件。",
    "模型微调": "构造统一数据格式和评测集，对比全参数、LoRA 或偏好优化等两到三种微调路径，记录显存、训练时长和任务指标，并分析退化样例与过拟合信号。",
    "模型部署": "从请求路径拆出预填充、解码、缓存和调度四段，做最小可复现实验；记录 P50/P95 延迟、tokens/s、显存占用和单位成本，并解释收益来自算力、带宽还是调度优化。",
    "智能体": "搭建可重复的任务环境，记录成功率、步数、工具调用数、平均延迟和失败类别；至少比较两种规划或记忆策略，并总结最值得上线前守护的回退条件。",
    "系统基础": "画出组件交互和数据流，量化关键路径的网络、磁盘或内存开销；随后对比至少两种架构或参数配置在吞吐、尾延迟和容错恢复上的差异。",
    "工程实践": "给出最小可运行工程或配置，再建立一组自动化检查与回归用例；最后补齐指标看板、故障演练和上线准入条件，确保方案可验证、可回滚、可维护。",
}

CATEGORY_SLUG_TOKENS = {
    "理论基础": "theory",
    "前沿追踪": "frontier",
    "模型训练": "model-training",
    "模型微调": "model-finetuning",
    "模型部署": "model-deployment",
    "智能体": "agent",
    "系统基础": "systems",
    "工程实践": "engineering",
    "数学基础": "math",
    "NLP基础": "nlp",
    "模型架构": "architecture",
    "多模态": "multimodal",
    "数据工程": "data-engineering",
    "评测与基准": "evaluation",
    "推荐系统": "recsys",
    "知识图谱": "knowledge-graph",
    "强化学习": "reinforcement-learning",
    "安全与对齐": "alignment",
}

BRIEF_SUFFIX = {
    "理论基础": "正文最好至少落到一个公式、复杂度边界、反例或数值验证上，避免只停留在概念解释。",
    "前沿追踪": "正文最好补齐公开基准、训练或推理成本，以及为什么它值得替换现有方案的证据。",
    "模型训练": "正文最好补齐训练曲线、关键超参数和异常信号，说明如何定位训练失败。",
    "模型微调": "正文最好补齐数据格式、参数量变化和退化样例，说明效果提升是否真的划算。",
    "模型部署": "正文最好补齐 P50/P95 延迟、tokens/s、显存峰值和单位请求成本。",
    "智能体": "正文最好补齐成功率、调用延迟、步数分布和失败恢复策略。",
    "系统基础": "正文最好补齐关键路径指标、容量上界和故障恢复信号。",
    "工程实践": "正文最好补齐自动化检查、核心指标和可回滚条件。",
}

KEYWORD_OVERRIDES = [
    (re.compile(r"RoPE|位置编码|LongRoPE|YaRN|Attention Sink", re.I), {
        "brief": "重点说明相位编码、插值或缩放如何改变长上下文中的注意力分布，并比较不同外推策略在困惑度、稳定性和可扩展长度上的差异。",
        "hint": "建议推导相位变换或插值公式，在 4K、8K、32K 等不同长度上比较 PPL、长答稳定性和显存变化。",
    }),
    (re.compile(r"FlashAttention|Chunked Attention|分块|IO", re.I), {
        "brief": "要把标准注意力与分块实现的 IO、HBM 读写和峰值显存成本算清楚，解释 online softmax 或 tile 调度为何能把吞吐提升出来。",
        "hint": "复现不同 tile 大小对 HBM 读写量、tokens/s 和峰值显存的影响，并解释性能拐点来自算力饱和还是内存带宽。",
    }),
    (re.compile(r"KV Cache|PagedAttention|连续批处理|Speculative|解码|量化|GQA|MQA", re.I), {
        "brief": "要把预填充与解码阶段拆开分析，比较缓存布局、调度方式或低精度策略如何影响延迟、吞吐、显存和输出一致性。",
        "hint": "设计 prefill/decode 分离的压测，记录 batch、context length 与并发变化下的延迟、tokens/s、显存占用和接受率或精度损失。",
    }),
    (re.compile(r"LoRA|QLoRA|PEFT|偏好|DPO|PPO|RLHF|奖励模型|SFT|指令", re.I), {
        "brief": "需要把参数更新路径、数据组织方式和目标函数联系起来讨论，比较 rank、beta、KL 约束或样本质量对最终效果和训练成本的影响。",
        "hint": "在统一数据集上对比至少两种微调或偏好优化配置，记录显存、训练时长、任务指标和退化样例，并分析最敏感的超参数。",
    }),
    (re.compile(r"Agent|智能体|MCP|ReAct|Plan|Function Calling|工具调用|多智能体|记忆|RAG", re.I), {
        "brief": "要把规划、检索、工具路由、状态持久化和失败恢复串起来分析，说明成功率提升来自哪一环，以及额外 token 与延迟成本是否值得。",
        "hint": "搭建统一任务集，比较两种规划或记忆策略在成功率、平均步数、工具调用数、P95 延迟和人工接管率上的差异。",
    }),
    (re.compile(r"分词|Tokenizer|BPE|WordPiece|SentencePiece|token", re.I), {
        "brief": "需要说明词表构建规则如何影响 token 长度分布、OOV 退化、训练吞吐和跨语言兼容性，并比较不同分词器的代价与收益。",
        "hint": "选一份中英混合语料，统计 token 长度、词表覆盖率、训练速度和下游困惑度，比较 BPE、WordPiece 或 Unigram 的差异。",
    }),
    (re.compile(r"矩阵|张量|特征值|SVD|QR|LU|谱半径|Jacobian|Hessian", re.I), {
        "brief": "建议从线性算子或高维数组视角讲清核心对象的几何含义、谱性质和数值稳定性，并说明它们在训练或表示学习中的直接作用。",
        "hint": "补一个小规模数值例子或 NumPy/PyTorch 验证脚本，展示公式推导、维度变化和误差来源如何对应到实际计算。",
    }),
    (re.compile(r"概率|熵|KL|尾界|贝叶斯|MLE|随机变量", re.I), {
        "brief": "需要写清随机变量、分布、似然或散度的定义差异，并说明这些量如何进入损失函数、泛化界或优化目标。",
        "hint": "至少推导一个关键公式，再用仿真或可视化验证参数变化对概率质量、误差界或损失曲线的影响。",
    }),
    (re.compile(r"ZeRO|FSDP|NCCL|RDMA|并行|分布式|All-to-All|Sequence Parallel|Expert Parallel", re.I), {
        "brief": "要把参数、梯度、优化器状态或激活在节点间如何切分讲清楚，并比较通信量、重计算和显存节省之间的真实平衡。",
        "hint": "给出集群拓扑和并行配置，记录 step time、通信占比、显存峰值和扩展效率，并用通信模型解释瓶颈来自哪里。",
    }),
]

SLUG_TRANSLATIONS = {
    "Test-Time Compute Scaling": "test-time-compute-scaling",
    "Function Calling": "function-calling",
    "OpenAI Agents SDK": "openai-agents-sdk",
    "Computer Use": "computer-use",
    "GUI": "gui",
    "MCTS": "mcts",
    "MCP": "mcp",
    "ReAct": "react",
    "Plan-and-Execute": "plan-and-execute",
    "Plan-Execute": "plan-execute",
    "RoPE": "rope",
    "LongRoPE": "longrope",
    "YaRN": "yarn",
    "QK-Norm": "qk-norm",
    "GQA": "gqa",
    "MQA": "mqa",
    "KV Cache": "kv-cache",
    "PagedAttention": "paged-attention",
    "FlashAttention": "flash-attention",
    "Speculative": "speculative",
    "Speculative Decoding": "speculative-decoding",
    "LoRA": "lora",
    "QLoRA": "qlora",
    "SwiGLU": "swiglu",
    "MoE": "moe",
    "Mamba-2": "mamba-2",
    "Ring Attention": "ring-attention",
    "Chunked Attention": "chunked-attention",
    "Pre-LN": "pre-ln",
    "Post-LN": "post-ln",
    "Attention Sink": "attention-sink",
    "Hybrid Transformer-SSM": "hybrid-transformer-ssm",
    "SentencePiece": "sentencepiece",
    "WordPiece": "wordpiece",
    "BPE": "bpe",
    "Tokenizer": "tokenizer",
    "BM25": "bm25",
    "RAG": "rag",
    "Tiktoken": "tiktoken",
    "LlamaTokenizer": "llama-tokenizer",
    "JSON Schema": "json-schema",
    "UCB1": "ucb1",
    "AlphaCode": "alphacode",
    "Game-of-24": "game-of-24",
    "The Pile": "the-pile",
    "CI": "ci",
    "CI/CD": "cicd",
    "SLO": "slo",
    "BF16": "bf16",
    "FP8": "fp8",
    "FP16": "fp16",
    "WSD": "wsd",
    "ZeRO": "zero",
    "FSDP": "fsdp",
    "NCCL": "nccl",
    "RDMA": "rdma",
    "ICL": "icl",
    "Scaling Law": "scaling-law",
    "Chinchilla": "chinchilla",
    "Double Descent": "double-descent",
    "GPT": "gpt",
    "BERT": "bert",
    "SVD": "svd",
    "QR": "qr",
    "LU": "lu",
}

ZH_TRANSLATIONS = {
    "多智能体": "multi-agent",
    "智能体": "agent",
    "工具调用": "tool-calling",
    "工具路由": "tool-routing",
    "工具": "tool",
    "规划": "planning",
    "重规划": "replanning",
    "记忆检索": "memory-retrieval",
    "记忆": "memory",
    "检索": "retrieval",
    "压缩": "compression",
    "衰减": "decay",
    "轨迹": "trajectory",
    "校验": "verification",
    "协议协同": "protocol-coordination",
    "协议": "protocol",
    "角色协商": "role-negotiation",
    "失败安全": "fail-safe",
    "状态机": "state-machine",
    "状态": "state",
    "向量数据库": "vector-database",
    "客户服务": "customer-service",
    "创意内容生成": "creative-content-generation",
    "神经网络": "neural-network",
    "万能近似定理": "universal-approximation-theorem",
    "梯度下降": "gradient-descent",
    "收敛性分析": "convergence-analysis",
    "损失函数": "loss-function",
    "规模定律": "scaling-laws",
    "最优训练": "optimal-training",
    "数据规模": "data-scale",
    "模型规模": "model-scale",
    "权衡": "tradeoff",
    "涌现能力": "emergent-abilities",
    "统计假象": "statistical-artifact",
    "真实验证": "real-validation",
    "注意力": "attention",
    "多头": "multi-head",
    "自注意力": "self-attention",
    "稀疏性": "sparsity",
    "上下文学习": "in-context-learning",
    "隐式梯度下降": "implicit-gradient-descent",
    "标签噪声鲁棒性": "label-noise-robustness",
    "格式敏感性": "format-sensitivity",
    "样例顺序影响": "example-order-effect",
    "指令微调": "instruction-tuning",
    "能力": "capability",
    "相互作用": "interaction",
    "外推相位漂移": "extrapolation-phase-drift",
    "外推缩放": "extrapolation-scaling",
    "相位漂移": "phase-drift",
    "频率插值": "frequency-interpolation",
    "超长外推退化": "ultra-long-extrapolation-degradation",
    "分组注意力头配置": "grouped-attention-head-configuration",
    "头部共享": "head-sharing",
    "共享": "shared",
    "路由负载均衡": "routing-load-balancing",
    "分块调度": "tiled-scheduling",
    "分块策略": "tiling-strategy",
    "稳定化设计": "stabilization-design",
    "动态缩放策略": "dynamic-scaling-strategy",
    "宽度配比": "width-ratio",
    "动态深度早退门控": "dynamic-depth-early-exit-gating",
    "混合位置编码拼接策略": "hybrid-positional-encoding-stitching",
    "接口约束": "interface-constraints",
    "置信阈值": "confidence-threshold",
    "任务分解": "task-decomposition",
    "故障恢复": "failure-recovery",
    "恢复策略": "recovery-strategy",
    "策略": "strategy",
    "梯度裁剪阈值": "gradient-clipping-threshold",
    "数据配比": "data-mixture",
    "配比策略": "mixture-strategy",
    "跨模态": "cross-modal",
    "对齐损失": "alignment-loss",
    "语音文本联合建模": "speech-text-joint-modeling",
    "安全评测": "safety-evaluation",
    "随机变量尾界": "random-variable-tail-bound",
    "尾界": "tail-bound",
    "泛化误差分解": "generalization-error-decomposition",
    "双下降机理": "double-descent-mechanism",
    "机械可解释性": "mechanistic-interpretability",
    "架构收益": "architecture-tradeoffs",
    "秩分配": "rank-allocation",
    "量化误差": "quantization-error",
    "偏好优化目标": "preference-optimization-objective",
    "指令数据构建": "instruction-data-construction",
    "多轮对话对齐": "multi-turn-dialog-alignment",
    "信息论": "information-theory",
    "应用": "application",
    "构建": "construction",
    "误差": "error",
    "目标": "objective",
    "发布": "release",
    "窗口": "window",
    "治理": "governance",
    "审计": "audit",
    "模板": "template",
    "流程": "workflow",
    "设计": "design",
    "更新": "update",
    "方法": "methods",
    "架构": "architecture",
    "基础": "fundamentals",
    "最新进展": "latest-progress",
    "生态": "ecosystem",
    "突破": "breakthrough",
    "统一": "unified",
    "具身智能": "embodied-intelligence",
    "长上下文": "long-context",
    "原生融合": "native-fusion",
    "测试时计算扩展": "test-time-compute-scaling",
    "开源闭源差距评估": "open-vs-closed-model-gap",
    "分片通信": "sharding-communication",
    "合并规则学习": "merge-rule-learning",
    "词表退化": "vocabulary-degradation",
    "最大匹配搜索复杂度": "max-match-search-complexity",
    "概率建模": "probabilistic-modeling",
    "字节级分词": "byte-level-tokenization",
    "多语言": "multilingual",
    "稳定性": "stability",
    "聊天模板": "chat-template",
    "特殊": "special",
    "词表大小": "vocabulary-size",
    "训练吞吐": "training-throughput",
    "耦合关系": "coupling",
    "退化路径": "degradation-path",
    "子词回退": "subword-fallback",
    "一致性": "consistency",
    "切片召回": "chunk-recall",
    "版本漂移": "version-drift",
    "线上兼容": "online-compatibility",
    "正则预分词": "regex-pretokenization",
    "归一化": "normalization",
    "收益曲线建模": "benefit-curve-modeling",
    "思维链长度": "chain-of-thought-length",
    "正确率饱和点": "accuracy-saturation-point",
    "并行采样": "parallel-sampling",
    "风险收益": "risk-reward",
    "自一致性投票": "self-consistency-voting",
    "方差收敛分析": "variance-convergence-analysis",
    "反思": "reflection",
    "重试": "retry",
    "计算预算分配": "compute-budget-allocation",
    "验证器引导解码": "verifier-guided-decoding",
    "误差传播": "error-propagation",
    "树搜索推理": "tree-search-reasoning",
    "引用对齐": "citation-alignment",
    "答案可追溯性": "answer-traceability",
    "宽深度权衡": "width-depth-tradeoff",
    "预算约束": "budget-constraints",
    "最优停止准则": "optimal-stopping-rule",
    "推理时温度调度": "inference-temperature-schedule",
    "不确定性估计": "uncertainty-estimation",
    "全链路": "end-to-end",
    "切片策略": "chunking-strategy",
    "召回上界": "recall-upper-bound",
    "稠密检索": "dense-retrieval",
    "负样本构造": "negative-sample-construction",
    "难例挖掘": "hard-negative-mining",
    "融合排序": "fusion-ranking",
    "分块并行": "tiled-parallelism",
    "下界": "lower-bound",
    "跨节点": "cross-node",
    "序列并行调度": "sequence-parallel-scheduling",
    "路由温度": "routing-temperature",
    "负载均衡损失": "load-balancing-loss",
    "共享专家": "shared-expert",
    "私有专家": "private-expert",
    "比例优化": "ratio-optimization",
    "动态跳层预算": "dynamic-layer-skipping-budget",
    "共享比": "sharing-ratio",
    "精度损失": "accuracy-loss",
    "稳态行为": "steady-state-behavior",
    "深层网络": "deep-network",
    "稳定条件": "stability-conditions",
    "梯度流": "gradient-flow",
    "深度上限": "depth-limit",
    "多尺度": "multi-scale",
    "混合注入策略": "hybrid-injection-strategy",
    "并行解码": "parallel-decoding",
    "验证接口": "verification-interface",
    "分块误差界": "chunking-error-bound",
    "稀疏注意力": "sparse-attention",
    "全局 token 选择准则": "global-token-selection-rule",
    "参数化": "parameterization",
    "宽度迁移": "width-transfer",
    "可复用性": "reusability",
    "调度": "schedule",
    "收敛差异": "convergence-difference",
    "混精训练": "mixed-precision-training",
    "分片粒度": "shard-granularity",
    "通信开销曲线": "communication-overhead-curve",
    "激活分片": "activation-sharding",
    "重计算": "recomputation",
    "热点缓解": "hotspot-mitigation",
    "学习率": "learning-rate",
    "混合精度": "mixed-precision",
    "断点续训容错": "checkpoint-resume-fault-tolerance",
    "质量门禁": "quality-gates",
    "灰度发布治理": "canary-release-governance",
    "告警分级": "alert-prioritization",
    "配置漂移检测": "config-drift-detection",
    "管道可观测性": "pipeline-observability",
    "视觉编码": "vision-encoding",
    "图像生成": "image-generation",
    "矩阵谱半径分析": "matrix-spectral-radius-analysis",
    "几何解释": "geometric-interpretation",
    "凸优化": "convex-optimization",
    "条件": "conditions",
    "截断误差": "truncation-error",
    "高效加载": "high-efficiency-loading",
    "敏感信息过滤": "sensitive-information-filtering",
    "失败快返": "fast-failure-return",
    "发布窗口": "release-window",
    "指标分层": "metric-layering",
    "复盘": "postmortem",
    "图谱": "graph",
    "实体对齐消歧": "entity-alignment-and-disambiguation",
    "检索重排融合": "retrieval-reranking-fusion",
    "覆盖度": "coverage",
    "动态题库": "dynamic-benchmark",
    "多样性": "diversity",
    "实时特征": "real-time-features",
    "对话式推荐": "conversational-recommendation",
    "镜像构建与优化": "image-build-optimization",
    "代码质量标准": "code-quality-standards",
    "测试驱动开发": "test-driven-development",
    "代码整洁": "clean-code",
    "技术债务": "technical-debt",
    "知识分享与传承": "knowledge-sharing",
    "本体论与语义网": "ontology-and-semantic-web",
    "实体识别与链接": "entity-recognition-and-linking",
    "关系抽取": "relation-extraction",
    "事件抽取": "event-extraction",
    "评测指标": "evaluation-metrics",
    "基准测试": "benchmarking",
    "红队评测": "red-team-evaluation",
    "向量空间": "vector-space",
    "线性变换": "linear-transform",
    "概率基础": "probability-fundamentals",
    "约束优化": "constrained-optimization",
    "随机优化": "stochastic-optimization",
    "熵与互信息": "entropy-and-mutual-information",
    "链式法则与反向传播": "chain-rule-and-backpropagation",
    "泰勒展开": "taylor-expansion",
    "函数分析": "functional-analysis",
    "变分法": "calculus-of-variations",
    "分类": "taxonomy",
    "深度 Q 网络": "deep-q-network",
    "批量强化学习": "batch-reinforcement-learning",
    "离线强化学习": "offline-reinforcement-learning",
    "推荐系统": "recommendation-system",
    "协同过滤": "collaborative-filtering",
    "内容推荐": "content-based-recommendation",
    "提示工程": "prompt-engineering",
    "社交推荐": "social-recommendation",
    "广告推荐": "ad-recommendation",
    "企业应用推荐": "enterprise-recommendation",
    "思维链推理": "chain-of-thought-reasoning",
}

EXACT_PREREQUISITES = {
    "梯度与链式法则": ["向量与矩阵"],
    "概率论基础": ["向量与矩阵"],
    "信息熵与交叉熵": ["概率论基础"],
    "Softmax 函数": ["向量与矩阵", "信息熵与交叉熵"],
    "感知机到多层感知机": ["向量与矩阵"],
    "激活函数全景": ["感知机到多层感知机", "梯度与链式法则"],
    "批归一化（BatchNorm）": ["梯度与链式法则", "激活函数全景"],
    "梯度下降三形态": ["梯度与链式法则"],
    "Adam 优化器": ["梯度下降三形态"],
    "权重初始化": ["感知机到多层感知机", "激活函数全景"],
    "学习率调度": ["梯度下降三形态", "Adam 优化器"],
    "Word2Vec CBOW": ["词袋模型与 TF-IDF"],
    "Word2Vec Skip-gram": ["词袋模型与 TF-IDF"],
    "BPE 分词": ["词袋模型与 TF-IDF"],
    "RNN 与梯度消失": ["梯度与链式法则", "激活函数全景"],
    "LSTM": ["RNN 与梯度消失"],
    "Seq2Seq 架构": ["LSTM"],
    "Bahdanau 注意力": ["Seq2Seq 架构", "Softmax 函数"],
    "Self-Attention": ["向量与矩阵", "Softmax 函数", "Bahdanau 注意力"],
    "多头注意力": ["Self-Attention"],
    "位置编码": ["Self-Attention"],
    "Transformer Encoder": ["多头注意力", "位置编码", "激活函数全景"],
    "GPT 架构": ["Transformer Encoder"],
    "BERT": ["Transformer Encoder"],
    "Transformer 的 FLOP 分析": ["Transformer Encoder"],
    "缩放定律（Scaling Laws）": ["GPT 架构", "预训练数据"],
    "预训练数据": ["BPE 分词"],
    "混合精度训练": ["Adam 优化器"],
    "KV Cache": ["GPT 架构", "Self-Attention"],
    "RoPE": ["Self-Attention", "位置编码"],
    "Flash Attention": ["Self-Attention", "Transformer 的 FLOP 分析"],
    "量化基础": ["GPT 架构"],
    "投机采样（Speculative Decoding）": ["GPT 架构", "解码策略"],
    "PagedAttention": ["KV Cache"],
    "指令微调（Instruction Tuning）": ["GPT 架构"],
    "RLHF 奖励模型": ["指令微调（Instruction Tuning）"],
    "PPO 在 LLM 对齐中的应用": ["RLHF 奖励模型"],
    "DPO": ["RLHF 奖励模型"],
    "LoRA": ["指令微调（Instruction Tuning）"],
    "QLoRA": ["LoRA", "量化基础"],
    "LayerNorm": ["梯度与链式法则", "Transformer Encoder"],
    "GQA 与 MQA": ["多头注意力", "KV Cache"],
    "连续批处理（Continuous Batching）": ["KV Cache", "PagedAttention"],
    "解码策略": ["Softmax 函数", "GPT 架构"],
    "困惑度（Perplexity）": ["概率论基础", "信息熵与交叉熵"],
    "涌现能力（Emergent Abilities）": ["缩放定律（Scaling Laws）"],
}

TITLE_PREFIX_ALIASES = {
    "LoRA": "LoRA",
    "QLoRA": "QLoRA",
    "DPO": "DPO",
    "PPO": "PPO 在 LLM 对齐中的应用",
    "RLHF": "RLHF 奖励模型",
    "BERT": "BERT",
    "GPT": "GPT 架构",
    "RoPE": "RoPE",
    "KV Cache": "KV Cache",
    "Self-Attention": "Self-Attention",
    "FlashAttention": "Flash Attention",
    "Flash Attention": "Flash Attention",
    "BPE": "BPE 分词",
    "Word2Vec": "Word2Vec Skip-gram",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Deduplicate and finalize topic queue.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def detect_batch(slug: str) -> str:
    if slug.startswith("expert-2026q1g"):
        return "g"
    if slug.startswith("expert-2026q1f"):
        return "f"
    if slug.startswith("expert-2026q1e"):
        return "e"
    if slug.startswith("expert-2026q1d"):
        return "d"
    if slug.startswith("expert-2026q1c"):
        return "c"
    if slug.startswith("expert-2026q1"):
        return "ab"
    return "orig"


def title_prefix(title: str) -> str:
    parts = re.split(r"[：:]", title, maxsplit=1)
    return parts[0].strip()


def first_tag(topic: dict) -> str:
    return (topic.get("tags") or ["工程实践"])[0]


def map_blog_category(queue_category: str, title: str, tags: list[str]) -> str:
    if queue_category in DIRECT_BLOG_CATEGORIES:
        return queue_category
    if queue_category == "安全与对齐":
        if re.search(r"PPO|DPO|RLHF|奖励|偏好|SFT|LoRA|QLoRA|对齐", title, re.I):
            return "模型微调"
        return "智能体"
    return QUEUE_TO_BLOG_CATEGORY.get(queue_category, "工程实践")


def is_generic(text: str, patterns: list[str]) -> bool:
    return any(p in text for p in patterns)


def score_item(topic: dict) -> tuple:
    batch = detect_batch(topic["slug"])
    brief = topic.get("brief", "")
    depth = topic.get("depth_hint", "")
    return (
        BATCH_PRIORITY[batch],
        1 if is_generic(brief, GENERIC_BRIEF_PATTERNS) else 0,
        1 if is_generic(depth, GENERIC_DEPTH_PATTERNS) else 0,
        -len(brief),
        -len(depth),
        topic["id"],
    )


def unique_preserve(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def join_sentences(*parts: str) -> str:
    cleaned = []
    for part in parts:
        part = (part or "").strip()
        if not part:
            continue
        cleaned.append(part.rstrip("。"))
    return "。".join(cleaned) + ("。" if cleaned else "")


def merge_tags(items: list[dict]) -> list[str]:
    merged = []
    for item in items:
        merged.extend(item.get("tags") or [])
    return unique_preserve(merged)


def build_brief(core: str, blog_category: str, existing: str = "", allow_existing: bool = True) -> str:
    if allow_existing and existing and len(existing) >= 55 and not is_generic(existing, GENERIC_BRIEF_PATTERNS):
        if len(existing) >= 80:
            return existing
        return join_sentences(existing, BRIEF_SUFFIX[blog_category])
    brief = BRIEF_TEMPLATE[blog_category].format(core=core)
    for pattern, payload in KEYWORD_OVERRIDES:
        if pattern.search(core):
            brief = join_sentences(brief, payload["brief"])
            break
    if len(brief) < 80:
        brief = join_sentences(brief, BRIEF_SUFFIX[blog_category])
    return brief


def build_depth_hint(core: str, blog_category: str, existing: str = "", allow_existing: bool = True) -> str:
    if allow_existing and existing and len(existing) >= 35 and not is_generic(existing, GENERIC_DEPTH_PATTERNS):
        return existing
    hint = DEPTH_TEMPLATE[blog_category]
    for pattern, payload in KEYWORD_OVERRIDES:
        if pattern.search(core):
            hint = join_sentences(hint, payload["hint"])
            break
    return hint


def needs_rewrite(primary: dict, group_size: int) -> bool:
    batch = detect_batch(primary["slug"])
    if batch in {"ab", "c", "d", "e", "f"}:
        return True
    if group_size > 1 and batch != "orig":
        return True
    if len(primary.get("brief", "")) < 55 or is_generic(primary.get("brief", ""), GENERIC_BRIEF_PATTERNS):
        return True
    if len(primary.get("depth_hint", "")) < 35 or is_generic(primary.get("depth_hint", ""), GENERIC_DEPTH_PATTERNS):
        return True
    return False


def translate_for_slug(text: str) -> str:
    for src, dst in sorted(SLUG_TRANSLATIONS.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(src, f" {dst} ")
    for src, dst in sorted(ZH_TRANSLATIONS.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(src, f" {dst} ")
    text = re.sub(r"[（()）【】\[\]{}]", " ", text)
    text = text.replace("/", " ")
    text = text.replace("&", " and ")
    text = text.replace("+", " plus ")
    text = text.replace("·", " ")
    text = text.replace("–", " ")
    text = text.replace("—", " ")
    text = text.replace("：", " ")
    text = text.replace(":", " ")
    text = text.replace("’", "")
    text = text.replace("'", "")
    text = text.replace("与", " and ")
    text = text.replace("及", " and ")
    text = text.replace("的", " ")
    text = text.replace("在", " ")
    text = text.replace("对", " ")
    text = text.replace("从", " ")
    text = text.replace("到", " ")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text


def make_slug(title: str, tags: list[str], fallback_slug: str, used: set[str]) -> str:
    slug = translate_for_slug(title)
    category_token = CATEGORY_SLUG_TOKENS.get(tags[0], "topic") if tags else "topic"
    if slug and (len(slug) < 6 or slug.count("-") == 0):
        slug = f"{slug}-{category_token}"
    if len(slug) < 6:
        queue_category = tags[0] if tags else "topic"
        slug = translate_for_slug(queue_category + " " + title_prefix(title))
    if len(slug) < 6:
        fallback_parts = [
            part
            for part in fallback_slug.split("-")
            if part
            and not part.startswith("expert")
            and not re.fullmatch(r"2026q1[a-z]*", part)
            and not re.fullmatch(r"\d+", part)
        ]
        fallback_token = "-".join(fallback_parts[-2:]) if fallback_parts else category_token
        slug = f"{category_token}-{fallback_token}".strip("-")
    original = slug
    index = 2
    while slug in used:
        slug = f"{original}-{index}"
        index += 1
    used.add(slug)
    return slug


def infer_prereq_candidates(title: str, queue_category: str, blog_category: str, tags: list[str]) -> list[str]:
    core = title_prefix(title)
    candidates = []

    def add(*values: str):
        for value in values:
            if value:
                candidates.append(value)

    for key, values in EXACT_PREREQUISITES.items():
        if core.startswith(key):
            add(*values)
            break

    if re.search(r"矩阵|张量|特征值|SVD|QR|LU|谱半径|Jacobian|Hessian", core, re.I):
        add("向量与矩阵")
    if re.search(r"概率|熵|KL|尾界|贝叶斯|MLE|随机变量", core, re.I):
        add("概率论基础")
    if re.search(r"Softmax|注意力|Transformer|RoPE|FlashAttention|GQA|MQA", core, re.I):
        add("Softmax 函数", "Self-Attention")
    if re.search(r"位置编码|RoPE|LongRoPE|YaRN|Attention Sink|长上下文", core, re.I):
        add("位置编码", "Self-Attention")
    if re.search(r"KV Cache|PagedAttention|连续批处理|Speculative|量化|解码|推理", core, re.I):
        add("GPT 架构", "KV Cache")
    if re.search(r"LoRA|QLoRA|PEFT|DPO|PPO|RLHF|奖励模型|偏好|指令", core, re.I):
        add("指令微调（Instruction Tuning）")
    if re.search(r"QLoRA", core, re.I):
        add("LoRA")
    if re.search(r"Agent|智能体|MCP|ReAct|Plan|Function Calling|工具调用|多智能体|记忆|RAG", core, re.I):
        add("GPT 架构", "指令微调（Instruction Tuning）")
    if re.search(r"RAG|检索|向量数据库|BM25|重排序", core, re.I):
        add("词袋模型与 TF-IDF", "Word2Vec Skip-gram")
    if re.search(r"ZeRO|FSDP|NCCL|RDMA|并行|分布式|All-to-All|Sequence Parallel|Expert Parallel", core, re.I):
        add("混合精度训练", "学习率调度")
    if re.search(r"梯度|优化器|学习率|归一化|BatchNorm|LayerNorm|混合精度|warmup|Adam", core, re.I):
        add("梯度与链式法则")

    if not candidates:
        if blog_category == "模型微调":
            add("GPT 架构", "指令微调（Instruction Tuning）")
        elif blog_category == "模型部署":
            add("GPT 架构")
        elif blog_category == "模型训练":
            add("梯度与链式法则")
        elif blog_category == "理论基础" and queue_category == "数学基础":
            add("向量与矩阵")
        elif blog_category == "智能体":
            add("GPT 架构")

    return unique_preserve(candidates)


def resolve_title_to_slug(title: str, prefix_index: dict[str, str], title_index: dict[str, str]) -> str | None:
    if title in title_index:
        return title_index[title]
    alias = TITLE_PREFIX_ALIASES.get(title)
    if alias and alias in prefix_index:
        return prefix_index[alias]
    prefix = title_prefix(title)
    if prefix in prefix_index:
        return prefix_index[prefix]
    if title in prefix_index:
        return prefix_index[title]
    return None


def stable_toposort(items: list[dict]) -> list[dict]:
    order = {item["slug"]: item["_order"] for item in items}
    graph = defaultdict(list)
    indegree = {item["slug"]: 0 for item in items}
    for item in items:
        for prereq in item["prerequisites"]:
            if prereq in indegree and prereq != item["slug"]:
                graph[prereq].append(item["slug"])
                indegree[item["slug"]] += 1

    ready = deque(sorted((slug for slug, deg in indegree.items() if deg == 0), key=lambda slug: order[slug]))
    by_slug = {item["slug"]: item for item in items}
    result = []
    while ready:
        slug = ready.popleft()
        result.append(by_slug[slug])
        for nxt in sorted(graph[slug], key=lambda key: order[key]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
        ready = deque(sorted(ready, key=lambda key: order[key]))

    if len(result) != len(items):
        return sorted(items, key=lambda item: item["_order"])
    return result


def validate(items: list[dict]):
    slugs = [item["slug"] for item in items]
    assert len(slugs) == len(set(slugs)), "duplicate slugs after rewrite"
    for item in items:
        assert item["blog_category"] in DIRECT_BLOG_CATEGORIES
        assert item["status"]
        for prereq in item["prerequisites"]:
            assert prereq != item["slug"]


def main():
    args = parse_args()
    data = json.loads(args.input.read_text())

    groups = defaultdict(list)
    group_order = {}
    for index, topic in enumerate(data):
        prefix = title_prefix(topic["title"])
        groups[prefix].append(topic)
        group_order.setdefault(prefix, index)

    consolidated = []
    for prefix in sorted(groups, key=lambda key: group_order[key]):
        items = groups[prefix]
        primary = sorted(items, key=score_item)[0]
        queue_category = first_tag(primary)
        tags = merge_tags(items)
        blog_category = map_blog_category(queue_category, prefix, tags)
        rewrite = needs_rewrite(primary, len(items))
        allow_existing = detect_batch(primary["slug"]) == "orig" and len(items) == 1
        title = prefix if len(items) > 1 else primary["title"]
        brief = build_brief(prefix, blog_category, primary.get("brief", ""), allow_existing=allow_existing) if rewrite else primary["brief"]
        depth_hint = build_depth_hint(prefix, blog_category, primary.get("depth_hint", ""), allow_existing=allow_existing) if rewrite else primary["depth_hint"]

        consolidated.append({
            "_order": group_order[prefix],
            "_queue_category": queue_category,
            "_fallback_slug": primary["slug"],
            "_keep_slug": detect_batch(primary["slug"]) == "orig",
            "title": title,
            "tags": tags,
            "blog_category": blog_category,
            "brief": brief,
            "depth_hint": depth_hint,
            "status": primary.get("status", "pending"),
        })

    used_slugs = set()
    for item in consolidated:
        if item["_keep_slug"]:
            slug = item["_fallback_slug"]
            if slug in used_slugs:
                slug = make_slug(item["title"], item["tags"], item["_fallback_slug"], used_slugs)
            else:
                used_slugs.add(slug)
        else:
            slug = make_slug(item["title"], item["tags"], item["_fallback_slug"], used_slugs)
        item["slug"] = slug

    title_index = {item["title"]: item["slug"] for item in consolidated}
    prefix_index = {title_prefix(item["title"]): item["slug"] for item in consolidated}

    for item in consolidated:
        candidates = infer_prereq_candidates(item["title"], item["_queue_category"], item["blog_category"], item["tags"])
        prereqs = []
        for candidate in candidates:
            slug = resolve_title_to_slug(candidate, prefix_index, title_index)
            if slug and slug != item["slug"]:
                prereqs.append(slug)
        item["prerequisites"] = unique_preserve(prereqs)

    consolidated = stable_toposort(consolidated)
    for index, item in enumerate(consolidated, start=1):
        item["id"] = index

    final_items = []
    for item in consolidated:
        final_items.append({
            "id": item["id"],
            "title": item["title"],
            "slug": item["slug"],
            "blog_category": item["blog_category"],
            "tags": item["tags"],
            "prerequisites": item["prerequisites"],
            "brief": item["brief"],
            "depth_hint": item["depth_hint"],
            "status": item["status"],
        })

    validate(final_items)
    args.output.write_text(json.dumps(final_items, ensure_ascii=False, indent=2) + "\n")

    print(f"input={len(data)} output={len(final_items)} removed={len(data) - len(final_items)}")


if __name__ == "__main__":
    main()
