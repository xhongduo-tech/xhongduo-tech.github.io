---
title: 训练与后训练
description: 把数据、目标、优化与模型能力变化对应起来。
kind: overview
status: published
---

# 训练与后训练

这一条路线逐步建立数据、监督微调、偏好优化与强化学习的实践。每个阶段既解释目标与梯度，也检查模型变化。

## 主线

| 阶段 | 需要理解 | 实践与证据 |
| --- | --- | --- |
| 预训练基础 | next-token目标、数据、计算预算与训练稳定性 | 小模型训练机制与数据检查 |
| 数据与SFT | 模板、标签、loss mask、训练与保留集 | 合理基线、训练日志、错误归因 |
| 参数高效适配 | LoRA等方法怎样改变可训练参数 | adapter训练、保存、重载与质量对照 |
| 偏好优化 | 偏好数据、目标、参考模型与偏差 | SFT与DPO固定协议比较 |
| RL进阶 | 采样、奖励、策略更新与奖励作弊 | 条件具备后的小规模可验证任务 |

SFT属于后训练，LoRA是实现适配的方式，DPO是偏好优化方法。它们按实际问题组织，不按术语数量计算项目成果。

## 对应项目

- [数据与监督微调](/projects/finetuning-lab)
- [偏好优化与后训练](/projects/post-training-lab)

资料入口：[Hugging Face LLM课程](https://huggingface.co/learn/llm-course/chapter1/1)、[PEFT LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)、[TRL DPO](https://huggingface.co/docs/trl/dpo_trainer)、[DeepSeek R1](https://github.com/deepseek-ai/DeepSeek-R1)。
