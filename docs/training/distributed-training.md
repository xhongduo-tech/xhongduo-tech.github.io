---
title: 分布式训练与检查点
description: 区分数据、状态、张量、流水线和专家切分，验证梯度、内存与恢复。
kind: topic
status: planned
---

# 分布式训练与检查点

::: info 待研究专题
先做CPU数学与状态验证，再按真实硬件选择一种分布式方案。本文没有多卡性能或大模型训练成绩。
:::

## 问题与前置依赖

训练内存花在哪里？多卡切分了数据、参数、梯度还是优化器？增加并行为什么可能减慢一步？保存权重后，是否真的能够恢复训练？

前置是PyTorch反向传播、优化器、[数据与预训练](/training/data-pretraining)，以及进程、通信与设备同步基础。先建立单进程参考，再逐个改变执行策略。

## 核心机制与内存账本

| 方法 | 分布的对象 | 必须追踪的成本 |
| --- | --- | --- |
| DDP | 数据分到不同rank，通常每rank保留完整模型与优化器状态 | 梯度同步、重复模型状态、数据采样与不同有效token数 |
| FSDP | 在数据并行组内分片模型状态，按策略聚合参数并归约梯度 | all-gather/reduce-scatter、参数暂存、激活和峰值 |
| ZeRO | Stage 1分片优化器状态，Stage 2再分片梯度，Stage 3再分片参数 | 具体阶段、通信、offload位置及传输成本 |
| TP | 层内张量与计算 | 切分方式、频繁集合通信及拓扑 |
| PP | 不同层/阶段和microbatch流水 | 激活传输、流水气泡、调度与检查点 |
| EP | 不同专家及token路由 | dispatch/combine、专家负载与容量 |

这些维度可以组合，不是一条必须逐级升级的阶梯。FSDP/ZeRO的实际配置决定何时重分片与聚合，不能只用名称预测峰值内存。[DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)、[FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)与[ZeRO说明](https://www.deepspeed.ai/tutorials/zero/)分别作为入口。

对参数量`P`，先写未分片模型状态的简化账本：

$$
M_{\rm states}\approx P(b_{\rm weight}+b_{\rm grad}+b_{\rm optim}).
$$

再加激活、通信buffer、临时聚合、缓存和allocator开销；混合精度主权重也要计入。模型状态可分片不代表整张卡内存按卡数等比例缩小。

数据并行的有效batch常写为：

$$
B_{\rm global}=B_{\rm micro}\times N_{\rm DP}\times K_{\rm accum}.
$$

这里`N_DP`是独立数据副本数，不把TP/PP/EP进程数再次乘入。可变长度语言建模还要检查全局有效token加权，避免各rank局部均值再平均造成目标变化。

## 配置与检查点格式

一个训练manifest应记录模型/数据revision、精度、microbatch、累积步数、DP/TP/PP/EP组、分片与offload策略、拓扑和后端。不同策略的合法组合由实现验证，不能任意相乘。

检查点至少核对模型、优化器、scheduler、AMP scaler（若使用）、随机状态、训练step、数据采样/读取位置及相关配置。仅有模型权重可用于推理，不等于能接续同一优化轨迹。分布式检查点还涉及分片metadata和恢复拓扑。[官方检查点文档](https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html)说明API与支持边界。

## 实现与源码阅读切入

以PyTorch固定提交`d29e9dd26cb86e5d7ecd8a66b801450fb1bf15b2`为阅读快照：

- [DistributedDataParallel](https://github.com/pytorch/pytorch/blob/d29e9dd26cb86e5d7ecd8a66b801450fb1bf15b2/torch/nn/parallel/distributed.py#L465)：构建副本、梯度同步与训练循环边界。
- [fully_shard](https://github.com/pytorch/pytorch/blob/d29e9dd26cb86e5d7ecd8a66b801450fb1bf15b2/torch/distributed/fsdp/_fully_shard/_fully_shard.py#L108)：参数状态、reshard与通信时机。
- [checkpoint save](https://github.com/pytorch/pytorch/blob/d29e9dd26cb86e5d7ecd8a66b801450fb1bf15b2/torch/distributed/checkpoint/state_dict_saver.py#L89)：state_dict、planner与存储职责。

TP与PP分别沿[官方TP教程](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)和[流水线教程](https://docs.pytorch.org/tutorials/intermediate/pipelining_tutorial.html)阅读；EP先用[DeepSeek V3报告](https://arxiv.org/abs/2412.19437v2)理解专家并行及计算通信关系。源码快照不是已经测试过的依赖组合。

## 分级实验

| 层级 | 拟执行实验 | 对照 |
| --- | --- | --- |
| CPU梯度 | 在后端支持时，用两个Gloo进程训练小线性模型 | 相同全局样本与目标，对照单进程梯度/更新；再加入不等长度样本检查权重 |
| CPU恢复 | 小模型保存完整状态，恢复后继续下一步 | 与不中断路径比较下一批数据、loss和参数；刻意漏存优化器或RNG观察差异 |
| CPU账本 | 列举各状态的shape、dtype与理论字节 | 分清理论容量、通信估算与实测；不推导真实集群吞吐 |
| 可选多GPU | 固定小模型、精度、有效batch和数据，只比较DDP与一种FSDP/ZeRO配置 | 测峰值内存、有效token/s、step时间、通信与恢复；不同时试遍全部并行组合 |

## 指标与验收产物

交付组拓扑与状态分布图、内存账本、梯度对照、完整检查点清单和恢复记录。硬件实验保留预热/同步方法、峰值口径、原始记录、失败rank及恢复边界。

梯度接近、结果可恢复和性能更好分别验收。不同硬件、精度、通信顺序或world size下不默认保证逐bit一致；先解释允许的误差与实际支持范围。

## 常见陷阱

把DDP当作参数分片；只计算权重而漏激活与临时all-gather；重复计算数据副本；梯度重复归一化；多rank各自写同一文件；保存成功但未演练加载；恢复了模型却改变数据位置或优化器。用CPU通信模拟宣称真实GPU互联性能也不成立。

## 第一方资料

核验日期：2026-11-16。

- [PyTorch DDP教程](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)、[FSDP2教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)：执行与状态切分。
- [ZeRO论文 v3](https://arxiv.org/abs/1910.02054v3)、[DeepSpeed ZeRO教程](https://www.deepspeed.ai/tutorials/zero/)：阶段与内存优化。
- [PyTorch TP](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)、[PP](https://docs.pytorch.org/tutorials/intermediate/pipelining_tutorial.html)、[Distributed Checkpoint 2.15](https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html)：实现与恢复入口。
- [DeepSeek V3 v2](https://arxiv.org/abs/2412.19437v2)：EP/PP与计算通信的案例。

关联：[训练地图](/training/) · [SFT](/training/sft) · [推理与系统](/inference/)。
