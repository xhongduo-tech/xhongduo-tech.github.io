---
title: 数据与监督微调实验
description: 从标签与训练目标进入真实SFT、重载、消融和独立评测。
kind: project
status: planned
---

# 数据与监督微调实验

## 问题与前置

选择一个有明确验证规则的小任务，研究数据质量、chat template、loss mask或LoRA配置中的一项因素。前置为[梯度与训练机制](/foundations/programming-math)、[数据隔离](/training/data-pretraining)和[SFT目标](/training/sft)。

## 实验步骤

| 步骤 | 必须检查 | 交付 |
| --- | --- | --- |
| 数据与标签 | 来源、近重复分组、模板、截断/packing、目标移位、padding/assistant mask | 逐token标注、数据校验与train/dev/test边界 |
| 合理基线 | 同一模型的可用提示设置，内容/格式/调用错误分别评分 | 原始输出与固定评分器 |
| 真实SFT | 冻结权重、可训练参数、rank/目标模块、学习率/有效batch、种子 | 参数/梯度检查、训练日志、环境与adapter |
| 新进程重载 | base/tokenizer revision、模板、精度与adapter匹配 | 固定输入重载结果与差异解释 |
| 一项消融 | 数据质量或mask等一项有依据的变化，其他协议可比 | 配对结果、错误切片、训练用量与不确定性 |
| 能力与运行回归 | 任务质量之外检查长度、格式和小型保留能力 | 改善、无变化、退化和部署代价 |

LoRA是适配方法，SFT是监督信号与目标的安排；QLoRA等技术组合另检查量化基座、计算精度与可训练参数。不能把三者写成互不相关的训练目标。

## 常见失败

标签全被屏蔽、模板重复特殊token、截断删掉答案、按token ID误屏蔽有效EOS、训练/测试近重复、adapter加载错base版本，都可能造成误导结果。训练loss下降与保留任务提升分别判断。

先用极小batch验证前向、反向、保存、加载，再做有界训练。没有预算或尚未跑通时记录阻塞，不以一次toy更新替代真实SFT，也不直接宣布需要更大模型。

## 验收

能解释参与loss的token与可训练参数；有真实adapter和重载记录、合理基线及受控变更；能定位一次未知数据/训练故障。报告包含独立评测与退化，不提前设定必须提升的百分比。

当前全部待实践。下一步：[偏好优化项目](/projects/post-training-lab)，或将模型部署回[推理项目](/projects/serving-lab)做联合回归。
