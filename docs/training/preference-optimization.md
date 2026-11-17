---
title: 偏好数据与直接偏好优化
description: 从成对回答、序列概率和参考策略，理解DPO目标及质量边界。
kind: topic
status: planned
---

# 偏好数据与直接偏好优化

::: info 待研究专题
先验证成对数据与CPU目标，再按条件开展真实DPO。当前没有偏好标注、训练或胜率成绩。
:::

## 问题与前置依赖

chosen为什么优于rejected，依据是否可靠？目标如何利用策略与参考模型的回答概率？偏好判别更好，是否代表生成更正确？

学习依赖是[SFT](/training/sft)、response mask、序列logprob、sigmoid和固定协议评测。本路线选择可复现SFT检查点与独立测试作为起点；这不是DPO目标对所有策略初始化的必要限制。

本页选择离线DPO：从固定偏好对优化策略，不要求训练时在线rollout或单独训练奖励模型。DPO有其他变体和数据流程，不能将本页的离线设定推广为所有偏好学习。[原论文第4节](https://arxiv.org/html/2305.18290v3)给出这里采用的基础目标。

## 核心机制

只对回答位置求和：

$$
\ell_\pi(y\mid x)=\sum_t m_t\log\pi(y_t\mid x,y_{<t}).
$$

对同一prompt的优选回答`y+`与较差回答`y-`，标准sigmoid DPO为：

$$
\mathcal L_{\rm DPO}=-\mathbb E\log\sigma\!\left(\beta\left[(\ell_\theta^+-\ell_{\rm ref}^+)-(\ell_\theta^--\ell_{\rm ref}^-)\right]\right).
$$

`πref`为冻结参考策略，常以SFT模型初始化策略和参考；具体参考设置需记录。`β`是目标尺度及参考约束解释中的参数，不能只由loss数值判断其质量作用。这里使用回答token的logprob之和；换成长度平均会改变目标，必须显式说明。

prompt不直接计入completion目标，但仍影响条件概率。仅比较chosen/rejected的原始概率，或忘掉参考项，都不是这个DPO目标。[TRL文档](https://huggingface.co/docs/trl/en/dpo_trainer)同时提供其他损失形式，实验需要明确选用哪一种。

## 数据格式与标注依据

以下为合成教学对，不是实际标注成果：

```json
{"prompt":[{"role":"user","content":"2+2等于多少？"}],"chosen":[{"role":"assistant","content":"4。"}],"rejected":[{"role":"assistant","content":"5。"}]}
```

另记录样本ID、来源/生成模型、偏好维度、标注者或验证规则、置信与分歧、长度、模板版本和split。按prompt族/来源分组切分，避免同一问题轻微改写跨训练测试。

偏好可以来自人、规则或模型判断，可靠性与适用范围不同。不要让“更长、格式漂亮、迎合”在未声明的情况下替代正确性；保留难判断、平局和分歧，不强行制造明确偏好。

## 实现与源码阅读切入

固定TRL提交`f27a906d39cc60aeea576e616f9c829dc4a6ce44`：从[DPO loss与completion logprob](https://github.com/huggingface/trl/blob/f27a906d39cc60aeea576e616f9c829dc4a6ce44/trl/trainer/dpo_trainer.py#L1377)追踪模板、padding、causal shift、mask、策略/参考前向和损失分支。

配置中明确基础sigmoid目标、参考模型或reference adapter、β、截断与是否预计算参考logprob。缓存必须绑定数据、模板和参考revision；LoRA只是参数化方式，不替代这些语义检查。

## 分级实验

| 层级 | 拟执行实验 | 对照 |
| --- | --- | --- |
| CPU概率 | tiny logits计算回答序列logprob，加入padding和被mask的prompt位置 | 检查mask、shift及padding不变性，直接比较逐token与求和结果 |
| CPU目标 | 令策略初始等于参考，计算loss，再交换chosen/rejected并做一次小更新 | 标准目标初始loss应为log 2；不据此声称训练了LLM或提升偏好 |
| CPU数据审计 | 故意构造长度偏置、重复prompt和错误chosen | 检查标注规则及能否发现反例 |
| 可选GPU | 同一SFT起点、固定偏好数据与有界预算，完成一次DPO | 与未优化SFT在同一生成/评测协议下比较；保存参考配置、成本与退化 |

## 指标与验收产物

交付数据card、偏好规则和分歧记录、CPU目标校验、模型/参考manifest、原始生成、评测与检查点。除reward margin或偏好判别正确率外，还检查独立任务正确性、格式、长度和能力退化。

人工或模型成对评测应盲化身份并交换位置；记录样本量与不确定性。判别数据上的高accuracy不能代替真实生成测试，也不能把标注模型的偏好直接当成人类总体偏好。

## 常见陷阱

chosen/rejected反置；把prompt复制进completion或截断后仅剩相同前缀；序列概率偷偷改成平均值；参考缓存未随模板更新；用同一批偏好反复调参并宣称泛化；只看训练loss或长度增长。完成DPO不等于完成PPO/GRPO，也不等于拥有在线RL流水线。

## 第一方资料

核验日期：2026-11-16。

- [DPO原论文 v3](https://arxiv.org/html/2305.18290v3)：目标、参考策略及其推导。
- [TRL DPO Trainer](https://huggingface.co/docs/trl/en/dpo_trainer)：数据格式、损失配置与实现边界。
- [固定DPO源码](https://github.com/huggingface/trl/blob/f27a906d39cc60aeea576e616f9c829dc4a6ce44/trl/trainer/dpo_trainer.py#L1468)：对应基础sigmoid分支；实际运行另锁依赖。

关联：[训练地图](/training/) · [后训练项目](/projects/post-training-lab) · [RL训练](/training/rl-training) · [评测](/evaluation/)。
