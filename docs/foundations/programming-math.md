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

### 大 logits 的稳定计算与梯度

把 $-\log p_y$ 写成对数归一化常数减去目标 logit，再将两项同时平移。对 $z=(1001,1000,999)$，第二个位置为标签时，代入 $m=1001$ 可得

$$L=(m-z_y)+\log\!\sum_j e^{z_j-m}=1+\log(1+e^{-1}+e^{-2})\approx 1.407606.$$

Softmax 概率约为 $(0.665241,0.244728,0.090031)$。直接对上千的 logits 做指数运算在常用浮点格式中会溢出；先求 softmax 再对很小的概率取对数也可能得到无穷大。最小对照同时计算稳定的 log-softmax 和朴素表达式，并将三个 logits 都加上同一个有限常数。精确算术中分布和 loss 均保持，浮点验证应选择不会把原有 logit 差值舍掉的偏移量。

第一次求导只看单个位置，固定标签 $y$。对归一化项求导得到 $p_i$，对 $-z_y$ 求导得到负的标签指示量，于是

$$\frac{\partial L}{\partial z_i}=p_i-\mathbf{1}_{i=y}.$$

本例梯度约为 $(0.665241,-0.755272,0.090031)$，和为零，与平移不变性一致。如果 logits 是线性投影 $z=Wh+b$，这一位置对 $W$ 的梯度为 $(p-\mathrm{onehot}(y))h^\top$。它同时解释了更新的方向和张量形状。用 float64 中心差分核对中间 logit 的导数，再在同一批数据上错移一个标签；梯度依然有值、形状也正确，只有检查目标位置才会暴露错位。本节数字是解析例子，监督位置的选择见 [SFT 的标签与掩码](/training/sft)。

### 从梯度到真实更新

一次训练更新包括前向、目标与loss、反向、优化器更新、梯度清理。梯度会累积；应根据预期的梯度累积方案决定何时清理。`eval()`切换部分模块行为，`no_grad()`控制梯度记录，两者用途不同。参见[Autograd官方教程](https://docs.pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html)与[优化步骤教程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)。

标量对照采用 $L(w)=(w-3)^2/2$、学习率 $0.1$、普通 SGD 与初值 $w=0$。第一次反向梯度 $-3$，更新得到 $0.3$。第二次正确梯度是 $-2.7$，清理旧梯度后更新得到 $0.57$；漏清梯度时缓冲区是 $-5.7$，更新为 $0.87$。故障版本在这两步内 loss 也下降，仅看 loss 无法定位问题。逐步记下梯度缓冲区和参数值，可以区分反向、累积、步长与优化器状态。使用 Adam 时，恢复训练还要验证动量、二阶矩估计和步号；这组 SGD 数字仅用来检查累积语义。

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
