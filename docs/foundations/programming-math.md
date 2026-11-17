---
title: 编程、数学与训练机制
description: 围绕真实模型调用与小张量，把概率、梯度、数值和排错连接起来。
kind: topic
status: planned
---

# 编程、数学与训练机制

## 要解决的问题

给出一条输入，能够追踪它如何进入模型、怎样产生概率与损失、哪些操作改变权重，以及错误应在哪一层定位。本页是待执行的学习与实验规格，不是已完成的教程复现。

前置只要求能读写文件、调用函数与运行Python程序。先独立尝试小问题，再用AI协作；验收时需要自己解释关键逻辑、修改需求并定位故障。

## 必须掌握的最小集合

| 主题 | 学到什么程度 | 检查方式 |
| --- | --- | --- |
| Python与工程基础 | 容器、函数、迭代、异常、类型边界、环境与依赖、Git差异 | 处理坏JSONL、重复ID与缺字段；给出可重现的错误 |
| 张量与线性代数 | shape、batch/sequence/head轴、矩阵乘法、广播、transpose/reshape、dtype/device | 写出每步输入输出维度，区分矩阵乘法与逐元素乘法 |
| 概率与数值 | 条件概率、softmax/log-softmax、交叉熵、KL的方向、采样与期望 | 手算小词表分布；解释极大logit为何需要稳定计算 |
| 微积分与优化 | 链式法则、梯度、梯度累积、学习率、SGD与Adam状态的作用 | 对小函数做有限差分检查，解释backward与step的不同 |
| 实验与统计 | 配对比较、随机种子、样本划分、方差、失败计入分母 | 同一组样本比较两个配置，不把少量随机差异直接归因给算法 |
| 运行系统 | 进程、I/O、并发、HTTP、超时、资源回收、CPU/GPU同步 | 区分排队、模型计算与传输时间；复现一次超时并解释状态 |

张量的存储、维度与设备行为以[PyTorch Tensor教程](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)为入口。对高级主题先建立可计算的直观认识，再逐渐补证明与更大规模实现。

## 从logits到参数更新

对词表中的logit $z_i$，稳定softmax可写为

$$p_i=\frac{\exp(z_i-m)}{\sum_j\exp(z_j-m)},\qquad m=\max_j z_j.$$

对正确标签 $y$，单位置负对数似然为 $-\log p_y$。模型训练还需要定义哪些位置参与平均、是否包含padding或提示词，以及目标如何与输入错位。缩小loss数值可能来自分母或mask变化，不能自动解释为模型变好。

一次训练更新包括前向、目标与loss、反向、优化器更新、梯度清理。梯度会累积；应根据预期的梯度累积方案决定何时清理。`eval()`切换部分模块行为，`no_grad()`控制梯度记录，两者用途不同。参见[Autograd官方教程](https://docs.pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html)与[优化步骤教程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)。

## 四个CPU实验

| 实验 | 输入与操作 | 应留下什么 |
| --- | --- | --- |
| 文本与token记录 | 同一tokenizer编码中文、英文、代码；保留特殊token与截断设置 | 文本、ID、长度、decode、模型/tokenizer revision与不符合直觉的案例 |
| 概率与loss | 构造3至5个logit，比较普通与稳定softmax；更换正确标签 | 概率和、手算loss、数值误差；不使用未经解释的打印值代替结论 |
| 梯度与更新 | 小线性分类器完成前向/反向/一次更新；再故意漏清梯度 | 参数与梯度前后值、有限差分误差、故障解释；只记机制实验 |
| 排错与复现 | 注入错维度、错误dtype、标签越界、全被mask中的一种 | 最小复现、堆栈、假设、修复和能阻止回归的检查 |

不需要GPU。真实小模型短生成用于连接调用链，不承担业务质量比较。Tokenizer接口参考[Transformers官方文档](https://huggingface.co/docs/transformers/main_classes/tokenizer)，实际运行锁定与模型兼容的版本。

## 向性能与源码延伸

先用[PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)找出算子、CPU等待与内存活动，再考虑编译、融合或自定义kernel。GPU计时要处理异步执行与预热；CPU小矩阵不能证明某个CUDA优化提速。Triton/CUDA、算子融合、通信重叠作为后续按需专项，进入前应能用profile解释瓶颈。

## 验收标准与误区

交付一个小目录，包含环境、输入、命令、原始结果、错误复现和解释。本人能在已有程序上改一项要求，例如变更字段或mask规则，再预测并验证结果。

不会把“运行成功”“一次参数更新”“依赖AI生成代码”分别写成“模型有效”“SFT已完成”“已经掌握编程”。对暂时解释不清的部分单独记录。

资料核验日期：2026-11-16。教程页面可更新；实验需另锁定实际运行版本。下一步：[架构与Attention](/foundations/transformer-architectures)、[数据与训练目标](/training/data-pretraining)。
