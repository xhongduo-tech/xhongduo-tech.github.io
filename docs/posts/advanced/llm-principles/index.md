---
pageClass: plain-doc
---

# 大模型原理

系统性拆解大语言模型的完整知识树：从发展脉络、Tokenizer、架构细节、预训练，到长上下文、多模态、推理、RAG、Agent、幻觉安全与评测。

## 主题规划

<ProgressGrid cat="advanced/llm-principles" />


### 第一篇 发展脉络与缩放定律

- [x] [从统计语言模型到神经语言模型：N-gram、Word2Vec 与 RNNLM](./from-statistical-to-neural-lm)
- [x] [Transformer 与预训练范式的确立：GPT-1 与 BERT](./transformer-and-pretraining-paradigm)
- [x] [GPT-2 与 GPT-3：规模即能力，上下文学习的出现](./gpt2-gpt3-scale-and-in-context-learning)
- [x] [T5 与统一文本到文本（Text-to-Text）范式](./t5-unified-text-to-text)
- [x] [InstructGPT 与 ChatGPT：指令微调与人类反馈对齐](./instructgpt-and-chatgpt-alignment)
- [x] [GPT-4 之后与开源生态：LLaMA 系谱与社区爆发](./gpt4-and-open-source-llama)
- [x] [Kaplan 缩放定律：损失与模型、数据、算力的幂律关系](./kaplan-scaling-laws)
- [ ] Chinchilla 缩放定律：计算最优下的参数-数据配比
- [ ] 涌现能力：现象描述、机理解释与"度量假象"之争

### 第二篇 Tokenizer 与分词

- [ ] 分词的基本问题：字符、词与子词粒度的权衡
- [ ] BPE（字节对编码）：训练算法与编码过程
- [ ] WordPiece：与 BPE 的目标函数差异
- [ ] Unigram 语言模型分词：概率视角的子词切分
- [ ] SentencePiece 与字节级 BPE：无空格语言与 OOV 终结
- [ ] 词表设计：词表大小、多语言覆盖与中文的"分词税"
- [ ] 分词的副作用：数字切分、代码、拼写与安全问题
- [ ] 动手实践：用 BPE 从零训练一个 Tokenizer

### 第三篇 GPT 架构逐层解析

- [ ] Decoder-only 架构总览：从 Token 到 Logits 的完整数据流
- [ ] 词嵌入层：查表、初始化与权重共享
- [ ] 缩放点积注意力：Q/K/V 与 Softmax 的数学推导
- [ ] 因果掩码与多头注意力：并行的"多个视角"
- [ ] 前馈网络：逐位置两层 MLP 与参数占比分析
- [ ] 残差连接与归一化的位置：Pre-LN 与 Post-LN 之争
- [ ] 输出头、Softmax 与解码采样：Temperature、Top-k 与 Top-p
- [ ] KV Cache：自回归推理的加速核心与显存代价
- [ ] 参数量与 FLOPs 估算：以 GPT-2 为例手算一遍
- [ ] 动手实现：用 PyTorch 从零写一个 mini-GPT

### 第四篇 主流开源架构对比

- [ ] LLaMA 架构剖析：RMSNorm、RoPE 与 SwiGLU 的"三件套"
- [ ] Qwen 架构演进：从 Qwen1 到 Qwen3 的设计变化
- [ ] DeepSeek 架构演进：MLA、DeepSeekMoE 与 V3 的系统级创新
- [ ] Mistral、Gemma 与其他架构：滑窗注意力与设计选型谱系
- [ ] 从 config.json 读懂一个模型：超参背后的架构选择

### 第五篇 归一化与激活函数

- [ ] LayerNorm 与 BatchNorm：为什么 NLP 选择前者
- [ ] RMSNorm：去掉均值平移的简化与加速
- [ ] Pre-LN 与 Post-LN：训练稳定性与最终性能的权衡
- [ ] 激活函数演进：ReLU、GELU 到 SwiGLU
- [ ] 进阶技巧：QK-Norm、Sandwich-LN 与注意力 logits 软裁剪

### 第六篇 位置编码

- [ ] 为什么需要位置编码：自注意力的置换不变性
- [ ] 正弦绝对位置编码：构造与性质
- [ ] 相对位置编码：T5 Bias 与相对位置表示思想
- [ ] RoPE 的数学原理：旋转矩阵、复数视角与远程衰减
- [ ] RoPE 代码精读：HuggingFace 实现逐行解析
- [ ] ALiBi：线性注意力偏置与训练-推理长度解耦
- [ ] 位置插值与外推：PI、NTK-aware 与 YaRN

### 第七篇 注意力机制变体与高效注意力

- [ ] 标准 MHA 的瓶颈：KV Cache 显存占用分析
- [ ] MQA（多查询注意力）：共享 K/V 的极致压缩
- [ ] GQA（分组查询注意力）：质量与效率的折中
- [ ] MLA（多头潜在注意力）：DeepSeek 的低秩联合压缩
- [ ] 滑动窗口注意力：局部感受野与信息逐层传播
- [ ] 稀疏注意力：Sparse Transformer、Longformer 与 BigBird
- [ ] FlashAttention：IO 感知的分块精确注意力
- [ ] 线性注意力与状态空间模型：Mamba 能否替代注意力

### 第八篇 混合专家模型（MoE）

- [ ] MoE 的核心思想：稀疏激活与条件计算
- [ ] 门控与路由：Top-k 路由器的实现
- [ ] 负载均衡：辅助损失与专家坍塌问题
- [ ] Switch Transformer：Top-1 路由与专家容量因子
- [ ] DeepSeekMoE：细粒度专家切分与共享专家
- [ ] MoE 的工程挑战：通信开销、显存与推理部署

### 第九篇 预训练数据工程

- [ ] 数据来源全景：网页、书籍、代码、论文与百科
- [ ] 数据清洗流水线：去重、过滤与个人信息处理
- [ ] 质量筛选：启发式规则、分类器与困惑度过滤
- [ ] 数据配比：领域混合比例的实验方法
- [ ] 课程学习与退火阶段：训练后期的数据调度
- [ ] 合成数据：Self-Instruct、教科书式数据与边界

### 第十篇 预训练目标与训练技术

- [ ] 因果语言建模（CLM）：下一个词预测的功与过
- [ ] MLM 与 Prefix-LM：理解向目标的复兴
- [ ] UL2 与混合去噪目标：统一不同预训练范式
- [ ] FIM（中间填充）：代码模型的双向上下文
- [ ] 学习率调度：Warmup、Cosine 衰减与 WSD
- [ ] 训练稳定性：Loss Spike、梯度爆炸与应对手段
- [ ] μP（最大更新参数化）：超参数跨规模迁移
- [ ] 混合精度训练：FP16、BF16 与 FP8
- [ ] 分布式训练：数据/张量/流水线并行与 ZeRO

### 第十一篇 长上下文

- [ ] 长上下文的价值与瓶颈：二次复杂度与位置外推失效
- [ ] 上下文窗口扩展：插值、外推与持续预训练
- [ ] 高效长文本注意力：稀疏化与线性化的取舍
- [ ] 长文本数据构造：长文档筛选与合成任务
- [ ] KV Cache 压缩：量化、驱逐与选择性保留
- [ ] 长上下文评测：大海捞针（NIAH）与 RULER

### 第十二篇 多模态大模型

- [ ] 多模态的总体范式：编码器-对齐-大模型三段式
- [ ] 视觉编码器：ViT 与 SigLIP
- [ ] 视觉-语言对齐：CLIP 的对比学习目标
- [ ] VLM 架构（一）：LLaVA 式投影器与视觉指令微调
- [ ] VLM 架构（二）：Qwen-VL 与动态分辨率处理
- [ ] 音频模态：Whisper 编码器与端到端语音语言模型
- [ ] 视频理解：帧采样、时空压缩与长视频建模
- [ ] 原生多模态：统一理解与生成的 Next-Token 预测

### 第十三篇 推理能力

- [ ] 思维链（CoT）提示：让模型"分步思考"
- [ ] 自洽性（Self-Consistency）：多数投票的路径聚合
- [ ] 思维树（ToT）与搜索：在推理空间中探索
- [ ] 过程奖励模型（PRM）与结果奖励模型（ORM）
- [ ] 测试时计算扩展：用更多推理换更高准确率
- [ ] OpenAI o1 范式：强化学习驱动的长思维链
- [ ] DeepSeek-R1：GRPO、纯 RL 涌现与"顿悟时刻"
- [ ] 推理能力蒸馏：把长思维链迁移到小模型

### 第十四篇 检索增强生成（RAG）

- [ ] RAG 总体框架：为什么检索能缓解幻觉与知识过期
- [ ] 文档切分策略：固定窗口、语义切分与结构感知
- [ ] 文本向量化：Embedding 模型与对比学习训练
- [ ] 索引与向量数据库：HNSW、IVF 与量化
- [ ] 检索策略：稠密检索、稀疏检索（BM25）与混合检索
- [ ] 重排序（Rerank）：交叉编码器的精排
- [ ] 高级 RAG：查询改写、HyDE 与迭代式检索

### 第十五篇 Agent

- [ ] Agent 范式总览：ReAct 的"推理-行动"循环
- [ ] 工具调用：Function Calling 的协议与实现
- [ ] 规划与反思：Plan-and-Execute 与 Reflexion
- [ ] 记忆机制：短期上下文与长期外部记忆
- [ ] MCP 与工具生态：标准化的模型-工具接口
- [ ] 多智能体系统：角色分工、辩论与协作
- [ ] Agent 的失败模式与评测：成功率之外的维度

### 第十六篇 幻觉与安全

- [ ] 幻觉的类型与成因：事实性/忠实性幻觉，数据、目标与解码的视角
- [ ] 幻觉的检测与缓解：检索 grounding、不确定性估计与解码约束
- [ ] 越狱攻击与提示注入：攻击面全景
- [ ] 安全对齐：SFT、RLHF 中的安全奖励与宪法 AI
- [ ] 红队测试：方法论与自动化红队
- [ ] 隐私与偏见：训练数据泄露、公平性与价值观对齐

### 第十七篇 评测

- [ ] 评测体系总览：能力维度与评测方式的分类
- [ ] 知识与学科评测：MMLU、C-Eval 与 GPQA
- [ ] 数学与代码评测：GSM8K、MATH、HumanEval 与 SWE-bench
- [ ] 指令遵循与对话评测：MT-Bench、AlpacaEval、Arena 与 LLM-as-a-Judge 的可靠性
- [ ] 数据污染：Benchmark 过拟合与评测失真

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
