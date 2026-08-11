---
pageClass: plain-doc
---

# 对齐技术（RLHF / DPO）

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Ian Goodfellow et al., "Deep Learning" (2016)
- Long Ouyang et al., "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT, 2022)
- Rafael Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (NeurIPS 2023)

## 主题规划

<ProgressGrid cat="advanced/llm-alignment" />

### 第1篇

- [x] [SFT 监督微调 (Ouyang et al., InstructGPT 2022 §3)](./sft-supervised-fine-tuning)
- [x] [RLHF 框架流程 (Ouyang et al., InstructGPT 2022 §2)](./rlhf-framework)
- [x] [偏好数据构建与标注 (Ouyang et al., InstructGPT 2022 §2)](./preference-data-construction)
- [x] [奖励模型训练 (Ouyang et al., InstructGPT 2022 §4)](./reward-model-training)
- [x] [奖励黑客与过度优化（reward hacking） (Gao et al., Reward Model Overoptimization 2023)](./reward-hacking-overoptimization)
- [x] [PPO 强化学习优化 (Ouyang et al., InstructGPT 2022 §5)](./ppo-rl-optimization)
- [x] [DPO 直接偏好优化 (Rafailov et al., DPO 2023 §3)](./dpo-direct-preference-optimization)
- [x] [宪法 AI 与 RLAIF (Bai et al., Constitutional AI 2022)](./constitutional-ai-rlaif)

### 第2篇

- [x] [对齐税与平衡 (Ouyang et al., InstructGPT 2022 §6)](./alignment-tax-balance)
- [x] [安全对齐与红队 (Ouyang et al., InstructGPT 2022 §7)](./safety-alignment-red-teaming)
