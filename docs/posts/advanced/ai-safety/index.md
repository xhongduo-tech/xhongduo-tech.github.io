---
pageClass: plain-doc
---

# AI 安全与对齐

AI 安全与对齐研究如何确保能力不断增强的人工智能系统可靠地服务于人类意图与价值。本篇按对齐问题、可解释性、鲁棒性、对齐技术、评估监测与 AI 治理的经典课程体系梳理全部选题。

## 主题规划

<ProgressGrid cat="advanced/ai-safety" />


### 第一篇 对齐问题（The Alignment Problem）

#### 第 1 章 对齐问题总论
- [x] [什么是对齐问题：意图对齐与价值学习的概念框架](./what-is-alignment)
- [x] [工具性趋同（Instrumental Convergence）：为何足够智能的系统会争夺资源](./instrumental-convergence)
- [x] [正交性论点（Orthogonality Thesis）：智能与目标无关性](./orthogonality-thesis)
- [x] [规范博弈（Specification Gaming）与奖励黑客（Reward Hacking）实例分析](./specification-gaming)

#### 第 2 章 外部对齐与内部对齐
- [x] [外部对齐（Outer Alignment）：奖励函数能否正确表达人类意图](./outer-alignment)
- [ ] 内部对齐（Inner Alignment）：目标错位的mesa优化器（Mesa-Optimizer）
- [ ] 欺骗性对齐（Deceptive Alignment）：模型为何可能伪装对齐
- [ ] 古德哈特定律（Goodhart's Law）在奖励建模中的四种形态

### 第二篇 可解释性（Interpretability）

#### 第 3 章 特征可视化与探针
- [ ] 神经元激活可视化与特征可视化（Feature Visualization）
- [ ] 探针方法（Probing）：线性探针与网络内部表征诊断
- [ ] 显著性图与归因方法：Integrated Gradients 与 Grad-CAM

#### 第 4 章 机制可解释性
- [ ] 机制可解释性（Mechanistic Interpretability）导论：电路（Circuits）视角
- [ ] Transformer 回路分析：归纳头（Induction Heads）与上下文学习
- [ ] 激活修补（Activation Patching）与因果干预方法
- [ ] 叠加假说（Superposition）与多义神经元（Polysemantic Neurons）
- [ ] 稀疏自编码器（Sparse Autoencoders）：从叠加中提取单义特征

### 第三篇 鲁棒性（Robustness）

#### 第 5 章 对抗样本
- [ ] 对抗样本（Adversarial Examples）现象与线性假说
- [ ] 攻击方法：FGSM、PGD 与 C&W 攻击
- [ ] 对抗训练（Adversarial Training）与认证鲁棒性（Certified Robustness）
- [ ] 大语言模型上的通用对抗后缀攻击（Universal Adversarial Suffixes）

#### 第 6 章 分布外泛化
- [ ] 分布偏移（Distribution Shift）与分布外泛化（OOD Generalization）
- [ ] 分布外检测（OOD Detection）：最大 softmax、能量分数与异常值暴露
- [ ] 虚假相关（Spurious Correlations）与捷径学习（Shortcut Learning）

### 第四篇 红队、越狱与 Agent 安全

#### 第 7 章 红队与越狱攻击
- [ ] 红队测试（Red Teaming）方法论：从人工红队到自动化红队
- [ ] 越狱攻击（Jailbreaking）分类学：角色扮演、编码绕过与多轮诱导
- [ ] 多模态模型的越狱攻击与视觉提示注入

#### 第 8 章 提示注入与 Agent 安全
- [ ] 提示注入（Prompt Injection）：直接注入与间接注入
- [ ] 间接提示注入的现实威胁：恶意网页、邮件与文档
- [ ] Agent 安全：工具调用权限、最小权限原则与动作确认机制
- [ ] Agent 沙箱（Sandboxing）与能力限制（Capability Restriction）设计

### 第五篇 幻觉与事实性

#### 第 9 章 幻觉（Hallucination）
- [ ] 幻觉现象的定义与分类：内在幻觉与外在幻觉
- [ ] 幻觉的成因：训练目标、数据噪声与知识边界
- [ ] 缓解方法：检索增强生成（RAG）、引用生成与不确定性校准
- [ ] 事实性评估基准：TruthfulQA、FEVER 与事实核查流水线

### 第六篇 对齐技术（Alignment Techniques）

#### 第 10 章 从人类反馈中学习
- [ ] RLHF 全流程：奖励模型训练、PPO 与拒绝采样
- [ ] 奖励模型失效模式：过优化（Overoptimization）与奖励篡改（Reward Tampering）
- [ ] DPO 及其变体：绕过显式奖励模型的偏好优化
- [ ] RLHF 之外的偏好学习：KTO、ORPO 与过程奖励模型（PRM）

#### 第 11 章 宪法 AI 与可扩展监督
- [ ] 宪法 AI（Constitutional AI）：原则驱动的自我批评与 RLAIF
- [ ] 可扩展监督（Scalable Supervision）问题：人类无法评估时如何监督
- [ ] 辩论（Debate）与递归奖励建模（Recursive Reward Modeling）
- [ ] 弱到强泛化（Weak-to-Strong Generalization）：弱监督者能否引导强模型

### 第七篇 评估与监测（Evaluations & Monitoring）

#### 第 12 章 能力评估
- [ ] 能力评估（Capability Evaluations）方法论与基准污染问题
- [ ] 危险能力评估（Dangerous Capability Evals）：生物、网络与自主复制
- [ ] 前沿模型安全框架：负责任扩展政策（RSP）与评估触发阈值

#### 第 13 章 对齐评估与欺骗行为
- [ ] 对齐评估（Alignment Evals）：谄媚（Sycophancy）与权力寻求倾向测量
- [ ] 欺骗行为评估：模型在什么条件下会隐藏真实意图
- [ ] 思维链监测（CoT Monitoring）与其可靠性边界
- [ ] 蜜罐测试与卧底评估（Undercover Evaluations）设计

### 第八篇 AI 治理（AI Governance）

#### 第 14 章 监管框架与国际协调
- [ ] 欧盟《人工智能法案》（EU AI Act）：风险分级监管框架
- [ ] 美国行政命令与自愿承诺：行政监管的演进
- [ ] 中国生成式人工智能管理办法与算法备案制度
- [ ] 国际协议与协调机制：AI 安全峰会与前沿AI安全国际科学报告

#### 第 15 章 算力治理与技术治理工具
- [ ] 算力治理（Compute Governance）：芯片出口管制与训练算力阈值
- [ ] 模型权重开放与封闭的治理权衡
- [ ] 结构化访问（Structured Access）与 API 分级开放

### 第九篇 长期风险讨论

#### 第 16 章 AGI 风险论证
- [ ] 工具 AI 与 Agent AI：两条技术路线的安全性对比
- [ ] AGI 风险的核心论证：从对齐失败到失控情景
- [ ] 渐进式风险与突发式风险：能力涌现对安全时间表的影响
- [ ] 对 AGI 风险论证的主要批评与反驳

### 第十篇 安全工程实践

#### 第 17 章 内容审核与护栏系统
- [ ] 内容审核（Content Moderation）流水线：分类器、审核 API 与人机协同
- [ ] 护栏系统（Guardrails）：输入过滤、输出过滤与宪法分类器
- [ ] 拒绝机制（Refusal）设计：过度拒绝与拒绝率校准

#### 第 18 章 水印与溯源
- [ ] 文本水印（Text Watermarking）：统计水印与嵌入水印
- [ ] 图像生成内容溯源：C2PA 标准与内容凭证
- [ ] 水印的鲁棒性攻击与去除方法评估

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
