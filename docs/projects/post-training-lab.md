---
title: 偏好优化与后训练实验
description: 以SFT基线为起点，检查偏好数据、优化效果和退化。
kind: project
status: planned
---

# 偏好优化与后训练实验

## 问题

以可复现的SFT模型为基础，准备有明确依据的偏好对，用DPO研究效果、偏差与副作用。

## 计划交付

- prompt、chosen、rejected数据及标注规则与切分。
- 模型、参考设置、训练配置和检查点。
- SFT与DPO在固定协议下的比较，检查正确性、长度与格式偏差。
- 失败样例、能力退化和运行成本说明。

完成DPO不等于完成在线RL。奖励、采样、策略更新与预算条件满足后，再安排RL实验。当前数据与训练均待实践。

资料：[TRL DPO](https://huggingface.co/docs/trl/dpo_trainer)、[DPO原论文](https://arxiv.org/abs/2305.18290)。
