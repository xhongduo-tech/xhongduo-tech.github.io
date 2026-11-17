---
title: MTP的训练目标与推理作用
description: 分清多token预测模块与投机解码算法。
kind: topic
status: planned
---

# MTP的训练目标与推理作用

::: info 专题提纲
计划进行原理与源码研究，尚无本人的训练或推理实测。
:::

## 研究问题

在DeepSeek V3中，MTP涉及辅助训练目标和预测模块；推理时可以舍弃这些模块，也可以将它们用于投机草稿。需要分别解释训练路径与推理路径，不能把MTP和投机解码当成同一个概念。[V3技术报告第2.2节](https://arxiv.org/html/2412.19437v2#S2.SS2)

## 需要讲清的内容

- 多个预测目标如何与序列位置对齐，辅助损失怎样组织。
- 草稿来自什么模块，目标模型怎样验证。
- 草稿质量、额外计算与接受长度怎样共同影响结果。
- 模型是否公开相应权重，所选引擎是否实际支持。

## 可先完成的小实验

用短序列画出不同预测深度的标签对齐；对照普通next-token目标，检查哪些token进入损失。随后用小词表草稿示意验证过程。这个实验只验证机制，不称为复训V3或复现生产性能。

## 阅读入口

[DeepSeek V3官方仓库](https://github.com/deepseek-ai/DeepSeek-V3) · [投机解码的先修](/inference/speculative-decoding) · [DSpark](/inference/dspark)
