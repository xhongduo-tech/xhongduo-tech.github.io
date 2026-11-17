---
title: 监督微调与参数高效适配
description: 把chat template、label mask、LoRA和QLoRA落实到训练目标与质量对照。
kind: topic
status: planned
---

# 监督微调与参数高效适配

::: info 待研究专题
以下是数据、目标与实现的研究路径。真实SFT、adapter重载与效果对照尚未执行，不预设改善。
:::

## 问题与前置依赖

一条对话经过模板后，哪些token参与训练？LoRA实际更新哪些参数？QLoRA量化了什么？loss下降能否转成任务收益？

前置是[数据与预训练](/training/data-pretraining)、tokenizer、因果预测、交叉熵、梯度和小模型生成。资源不够时先做mask和低秩线性层验证，不先追求大模型。

## 核心机制

SFT使用示例目标。对prompt `x`、回答`y`及目标mask `m`，一个completion-only形式为：

$$
\mathcal L_{\rm SFT}=-\frac{1}{\sum_t m_t}\sum_t m_t\log\pi_\theta(y_t\mid x,y_{<t}).
$$

prompt不计入该目标的直接损失，但仍影响回答的条件概率。全序列、completion-only与assistant-only各自有用途，必须按任务和模板明确选择；不能只看数据有`role`字段就认定已经正确屏蔽。[TRL SFT文档](https://huggingface.co/docs/trl/en/sft_trainer)区分这些模式。

LoRA在冻结基矩阵上学习低秩更新：

$$
W'=W+\frac{\alpha}{r}BA,\qquad A\in\mathbb R^{r\times d_{\rm in}},\quad B\in\mathbb R^{d_{\rm out}\times r}.
$$

这里展示常见缩放形式；实现变体另核对。每个目标矩阵新增参数约`r(d_in+d_out)`，仍需激活、梯度和优化器内存。[LoRA原论文](https://arxiv.org/abs/2106.09685v2)是机制来源，不据论文结果承诺当前任务质量。

QLoRA将冻结基座以低比特表示，并训练LoRA适配器；NF4、double quantization和paged optimizer是原方案的重要部分。计算dtype与可训练参数通常仍使用浮点，不能写成“所有参数和反向传播都是4-bit”。[QLoRA论文](https://arxiv.org/abs/2305.14314v1)的硬件实验条件也不能直接套到任意设备。

## 数据与mask格式

合成教学对话：

```json
{"messages":[{"role":"user","content":"将温度12摄氏度转成固定JSON字段。"},{"role":"assistant","content":"{\"temperature_c\":12}"}]}
```

下列ID仅演示prompt/answer/EOS的位置，不对应特定tokenizer：

```json
{"input_ids":[11,12,21,22,2],"attention_mask":[1,1,1,1,1],"labels":[-100,-100,21,22,2]}
```

保存原messages、渲染文本、token IDs、角色边界、labels与有效目标数。核对因果shift由模型还是collator执行，避免偏移两次。`attention_mask`控制可见/有效位置的含义与`labels=-100`忽略损失不同；不能互相替代。

chat template、BOS/EOS、截断和生成时的assistant前缀都要与模型匹配。先模板化后再次分词时检查是否重复添加特殊token。[官方模板文档](https://huggingface.co/docs/transformers/en/chat_templating)说明这些边界。assistant-only模式还依赖所选模板能提供相应mask，不能把一个配置布尔值当作验证。

## 实现与源码阅读切入

- [TRL collator与SFTTrainer固定快照](https://github.com/huggingface/trl/blob/f27a906d39cc60aeea576e616f9c829dc4a6ce44/trl/trainer/sft_trainer.py#L89)：沿原始样本→模板→截断/packing→labels→loss检查。
- [PEFT LoRA层固定快照](https://github.com/huggingface/peft/blob/28719900254e5bba0099a834bdca5c8c085f1527/src/peft/tuners/lora/layer.py#L112)：区分基座权重、适配参数、缩放和保存/加载；实际target modules随模型变化。

运行时锁定TRL/Transformers/PEFT、量化后端、模型与tokenizer revision。选定框架支持的稳定组合，先检查一批样本与可训练参数，不直接复制文档的大训练配方。

## 分级实验

| 层级 | 拟执行实验 | 对照 |
| --- | --- | --- |
| CPU目标 | 手算小logits的masked CE，改变padding与prompt标签，打印每token贡献 | 正确mask与故意错误mask；说明prompt虽不直接计分仍参与条件 |
| CPU低秩层 | 小线性层接入BA，做一次更新并保存/重载 | 检查W冻结、适配器梯度与重载输出；这是机制验证，不是完整LLM微调 |
| 可选GPU | 一个小模型、一份有切分的数据和有界step，先LoRA SFT | 与同基座合理提示基线比较；只改变一种数据或rank因素 |
| 资源受限分支 | 后端支持时再选择QLoRA，与匹配精度方案比较 | 检查量化配置、内存和质量；不要求同时做全参数、LoRA和QLoRA全部实验 |

## 指标与验收产物

交付数据/模板清单、逐token mask、trainable parameters、训练配置与日志、adapter及重载说明。评估保留集任务正确性、格式、长度、退化和运行用量，区分训练loss与生成质量；调参使用开发集，最终测试协议冻结。

保存基座revision、适配器配置、tokenizer与模板；如果合并权重，另记录合并与量化顺序。完成脚本运行、理解机制和证明改进分别验收。

## 常见陷阱

模板不匹配、重复特殊token、答案被截掉、全部labels变成忽略值；pad与EOS相同后误删所有EOS目标；packing使不该互看的样本共享上下文；只统计adapter大小却漏基座/激活；加载adapter时换了基座或target modules。量化降低存储不保证在每种硬件上更快。

## 第一方资料

核验日期：2026-11-16。

- [TRL SFT Trainer](https://huggingface.co/docs/trl/en/sft_trainer)：数据格式、mask与PEFT集成。
- [Transformers chat templates](https://huggingface.co/docs/transformers/en/chat_templating)：模板和特殊token。
- [LoRA v2](https://arxiv.org/abs/2106.09685v2)、[QLoRA v1](https://arxiv.org/abs/2305.14314v1)：原始方法与实验条件。
- [PEFT LoRA文档](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora)：配置、参数与适配器参考。

关联：[训练地图](/training/) · [SFT项目](/projects/finetuning-lab) · [偏好优化](/training/preference-optimization)。
