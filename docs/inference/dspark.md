---
title: DSpark的草稿生成与验证调度
description: 从半自回归草稿到置信度与负载相关的验证决策。
kind: topic
status: planned
---

# DSpark的草稿生成与验证调度

::: info 专题提纲
目前只建立研究路径，CPU实验与真实硬件复现都待完成。
:::

## 核心问题

DSpark将半自回归草稿生成与置信度调度结合：草稿需要建模token之间的依赖，验证长度还要考虑前缀存活概率与引擎代价。本文拟先验证这些决策的机制，再讨论性能。[DSpark论文](https://arxiv.org/abs/2607.05147)

## 两组需要分开研究的问题

第一组是效率：什么时候长草稿值得验证，什么时候会浪费批处理容量？第二组是正确性：如果先看完整草稿再决定验证长度，是否会产生选择偏差？后者需要阅读论文第3.2节与附录A，检查early-stopping及可见信息的约束。[论文正文](https://arxiv.org/html/2607.05147v1)

## 计划实验

1. 用合成的条件接受概率构造前缀存活概率：各位置的概率以此前位置均被接受为条件，累积乘积得到整个前缀被接受的概率。
2. 阅读官方PyTorch reference，检查给定`total_q`的排序与预算分配；它是调度的一部分，不包含完整的硬件代价搜索与置信度校准。
3. 复现小词表的选择偏差反例，说明为什么正确性条件重要。
4. 有匹配模型和硬件时，再测接受长度、草稿与验证耗时、单用户速度和吞吐。

CPU模拟不等于DeepInfer硬件性能复现，论文成绩也不写成本人的成绩。

## 源码与后续

[DeepInfer调度reference](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py) · [DeepInfer系统专题](/inference/deepinfer)
