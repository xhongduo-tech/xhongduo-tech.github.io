---
pageClass: plain-doc
---

# 大模型微调

让通用模型适配特定任务与价值观：从继续预训练、指令微调到偏好对齐的完整知识树。

## 主题规划

<ProgressGrid cat="advanced/llm-finetuning" />


### 第一篇 微调范式

- [x] [微调范式总览：从预训练、指令微调到对齐的三阶段](./finetuning-paradigms)
- [x] [继续预训练（Continual Pre-Training）：语料选择、混合配比与训练策略](./continual-pre-training)
- [x] [领域自适应预训练与通用能力保持的权衡](./domain-adaptive-pretraining)
- [x] [有监督微调（SFT）：指令微调的任务定义与训练目标](./supervised-finetuning)
- [x] [指令泛化：任务多样性、规模效应与指令演化（Evol-Instruct）](./instruction-generalization)
- [x] [对齐范式总览：RLHF 与直接偏好优化两大路线](./alignment-paradigms)
- [x] [微调的缩放定律：数据量、模型规模与下游收益](./finetuning-scaling-laws)
- [x] [小样本高效微调：LIMA「表面对齐假说」及其争议](./lima-superficial-alignment)

### 第二篇 数据工程

- [x] [指令数据的三大来源：人工标注、模型蒸馏与自生成](./instruction-data-sources)
- [x] [Self-Instruct：自动指令生成流水线与过滤策略](./self-instruct)
- [x] [WizardLM 的指令演化：深度演化与广度演化](./wizardlm-evol-instruct)
- [x] [数据质量筛选：困惑度、IFD（指令遵循难度）与去重去污染](./data-quality-filtering)
- [x] [数据多样性与配比：任务混合、长度分布与采样课程](./data-diversity-mixing)
- [x] [多轮对话数据的构造、组织与质量控制](./multi-turn-dialogue-data)
- [x] [对话模板：ChatML、Llama 系列模板与特殊 token 的设计](./chat-templates)
- [x] [loss mask：只对回答部分计算损失的实现细节与常见错误](./loss-mask)
- [x] [packing：样本拼接、跨样本注意力隔离与位置编码处理](./packing)
- [x] [数据集格式与工具链：Alpaca、ShareGPT 格式的互转与校验](./dataset-format-tooling)

### 第三篇 全参数微调

- [x] [显存账本：参数、梯度、优化器状态与激活值的精确计算](./memory-budget)
- [x] [混合精度训练：BF16/FP32 的数值稳定性与 AdamW 的显存开销](./mixed-precision)
- [x] [梯度检查点：以计算换显存的激活重计算策略](./gradient-checkpointing)
- [x] [DeepSpeed ZeRO：ZeRO-1/2/3 的优化器状态、梯度与参数分片](./deepspeed-zero)
- [x] [FSDP：PyTorch 原生全分片数据并行的原理与配置](./fsdp)
- [x] [张量并行与流水线并行在微调场景中的取舍](./tensor-pipeline-parallelism)
- [x] [卸载技术：CPU/NVMe offload 与分页优化器](./offloading)
- [x] [训练吞吐调优：从单卡到多卡的 MFU 分析](./training-throughput)
- [x] [训练稳定性：loss 尖峰、梯度裁剪与学习率调度](./training-stability)

### 第四篇 参数高效微调（PEFT）

- [x] [PEFT 总览：加性方法、选择性方法与重参数化方法](./peft-overview)
- [x] [Adapter：瓶颈结构、放置位置及其变体](./adapter)
- [x] [Prefix-Tuning 与 Prompt-Tuning：可学习的连续提示](./prefix-prompt-tuning)
- [x] [P-Tuning v2：面向全规模模型的深层提示调优](./p-tuning-v2)
- [x] [LoRA 原理：低秩假设、秩的选择与目标模块](./lora-principles)
- [x] [LoRA 的工程细节：初始化、缩放系数与推理时的权重合并](./lora-engineering)
- [x] [QLoRA：NF4 量化、双重量化与分页优化器的协同](./qlora)
- [x] [DoRA：权重分解的方向-幅度解耦微调](./dora)
- [x] [PiSSA：主奇异成分初始化及其收敛加速原理](./pissa)
- [x] [LoRA+ 与 rsLoRA：非对称学习率与秩稳定缩放](./lora-plus-rslora)
- [x] [AdaLoRA：基于重要性评分的秩自适应分配](./adalora)
- [x] [LoRA 的模块化组合：多 LoRA 融合与 MoE 化 PEFT](./lora-composition)

### 第五篇 长序列微调

- [x] [长上下文微调的挑战：注意力显存、位置外推与数据稀缺](./long-context-challenges)
- [x] [位置编码扩展：位置插值（PI）、NTK 与 YaRN](./position-encoding-extension)
- [x] [序列并行：DeepSpeed Ulysses 的注意力头切分](./sequence-parallelism-ulysses)
- [x] [Ring Attention：块级注意力的环形通信与负载均衡](./ring-attention)
- [x] [长序列训练数据构造：长文档续写、长对话与合成长依赖任务](./long-sequence-data)
- [x] [长上下文能力评估：大海捞针（NIAH）与 RULER 基准](./long-context-evaluation)

### 第六篇 偏好优化

- [x] [RLHF 全流程总览：SFT → 奖励模型 → 强化学习三阶段](./rlhf-overview)
- [x] [偏好数据收集：标注协议、一致性检验与 Bradley-Terry 模型](./preference-data-collection)
- [x] [奖励模型训练：排序损失、模型集成与奖励校准](./reward-model-training)
- [x] [PPO 在 LLM 上的实现：KL 惩罚、价值网络与工程调参技巧](./ppo-for-llm)
- [x] [拒绝采样微调：Best-of-N 蒸馏、RFT 与 RAFT](./rejection-sampling)
- [x] [ReST 与迭代式拒绝采样：数据-训练循环的自提升](./rest-iterative-rejection)
- [x] [DPO 推导：从 RLHF 目标到闭式最优解的完整数学链条](./dpo-derivation)
- [x] [DPO 实践：参考模型、温度系数 β 与常见训练陷阱](./dpo-practice)
- [x] [IPO 与 KTO：非配对偏好数据与前景理论的引入](./ipo-kto)
- [x] [ORPO 与 SimPO：无参考模型的直接偏好优化](./orpo-simpo)
- [x] [GRPO：组相对优势估计与对 PPO 的简化](./grpo)
- [x] [RLVR：基于可验证奖励的强化学习（数学与代码推理）](./rlvr)
- [x] [RLAIF 与 Constitutional AI：用 AI 反馈替代人类标注](./rlaif-constitutional)
- [x] [过程奖励模型（PRM）：逐步监督与推理链对齐](./process-reward-model)
- [x] [在线与离线偏好优化的取舍：迭代 DPO 与分布偏移问题](./online-offline-preference)

### 第七篇 领域与多模态微调

- [x] [领域微调方法论：语料配比、能力保持与评测基线](./domain-finetuning-methodology)
- [x] [医疗大模型微调：语料来源、隐私合规与幻觉控制](./medical-finetuning)
- [x] [法律与金融领域微调：术语对齐与事实性约束](./legal-finance-finetuning)
- [x] [代码模型微调：CodeLlama/StarCoder 的数据配方与 Fill-in-the-Middle](./code-model-finetuning)
- [x] [数学推理微调：解题数据合成、验证器与拒绝采样](./math-reasoning-finetuning)
- [x] [LLaVA 式视觉指令微调：视觉-语言对齐预训练与指令数据构造](./llava-vlm-finetuning)
- [x] [多模态对话数据：视觉 grounding、OCR 与图文交错数据](./multimodal-dialogue-data)

### 第八篇 框架实战

- [x] [微调框架选型：LLaMA-Factory、Axolotl、ms-swift 与 TRL 横向对比](./framework-selection)
- [x] [LLaMA-Factory 实战：配置文件、WebUI 与 LoRA/QLoRA 全流程](./llama-factory-practice)
- [x] [Axolotl 实战：YAML 声明式配置与多卡训练](./axolotl-practice)
- [x] [ms-swift 实战：国产模型适配与轻量化部署衔接](./ms-swift-practice)
- [x] [TRL 实战：SFTTrainer、DPOTrainer 与 PPOTrainer 的编程接口](./trl-practice)
- [x] [训练流水线工程化：数据校验、断点续训与实验跟踪](./training-pipeline-engineering)

### 第九篇 评估与诊断

- [x] [微调效果评估体系：能力、对齐与安全三个维度](./evaluation-framework)
- [x] [通用能力基准的正确使用：MMLU、C-Eval 与 GSM8K 的评测细节](./general-benchmarks)
- [x] [指令遵循评估：IFEval 与 FollowBench](./instruction-following-eval)
- [x] [对齐质量评估：MT-Bench、AlpacaEval 与 LLM-as-Judge 的偏差分析](./alignment-eval)
- [x] [灾难性遗忘：度量方法与回放、正则化、模块隔离等缓解手段](./catastrophic-forgetting)
- [x] [过拟合与记忆化：训练集泄漏与逐字复述检测](./overfitting-memorization)
- [x] [reward hacking：奖励模型失配的识别、度量与缓解](./reward-hacking)
- [x] [评估中的混淆因素：长度偏差、格式偏差与采样随机性](./evaluation-confounders)
- [x] [安全性与越狱鲁棒性评估](./safety-jailbreak-eval)

> 注：vLLM、SGLang 属于推理引擎，归入 [大模型部署](/posts/advanced/llm-deployment/) 模块。

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
