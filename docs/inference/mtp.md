---
title: MTP的训练目标与推理作用
description: 从标签对齐和因果链理解多token预测，区分辅助训练、草稿与验证。
kind: topic
status: planned
---

# MTP 的训练目标与推理作用

::: info 专题计划 · 待研究
以 DeepSeek-V3 报告的具体设计为教学锚点。计划完成标签与损失检查，尚无本人的模型训练或推理实测。
:::

## 要解决的问题

普通 next-token prediction 在每个位置预测下一 token；multi-token prediction 扩展训练目标到后续多个位置。不同论文的 MTP 实现不完全相同：[Gloeckle 等的论文](https://arxiv.org/abs/2404.19737v1)采用共享主干上的独立输出头，而 [V3 报告 §2.2](https://arxiv.org/html/2412.19437v2#S2.SS2)使用顺序连接的额外模块，维持各预测深度的因果链。

本页要回答：多步目标怎样对齐，训练时输入了哪些已知 token，推理时预测模块如何变成草稿，为什么训练采用 MTP 不代表部署后自动加速。

## 前置依赖

需要会读张量 shape、embedding、交叉熵和 loss mask，区分 teacher forcing 与生成时的历史输入。先看[基础地图](/foundations/)，再理解[投机解码](/inference/speculative-decoding)中的目标验证与修正；第一次做标签检查无需训练大模型。

## V3 中的模块和训练目标

按报告的索引：主模型在位置 $i$ 预测 $t_{i+1}$；深度 $k$ 的 MTP 模块把前一深度表示与 $t_{i+k}$ 的 embedding 组合，再预测 $t_{i+k+1}$。额外模块包含投影和 Transformer block，并与主模型共享 embedding 与输出头。[V3 §2.2，式 21–25](https://arxiv.org/html/2412.19437v2#S2.SS2)

将总训练目标示意为：

$$
\mathcal{L}_{\mathrm{total}}
=\mathcal{L}_{\mathrm{NTP}}
+\frac{\lambda}{D}
\sum_{k=1}^{D}\mathcal{L}^{(k)}_{\mathrm{MTP}}.
$$

$D$ 是额外预测深度，$\lambda$ 控制辅助损失权重。这里省略其他训练项，位置范围、归一化及序列边界以具体实现为准；不能把一个简化目标当作完整训练配方。

| 路径 | 在位置 $i$ 的新增 token 输入 | 预测目标 |
| --- | --- | --- |
| 主模型 | 因果主干已有历史 | $t_{i+1}$ |
| MTP 深度 1 | $t_{i+1}$ 的 embedding 与主干表示 | $t_{i+2}$ |
| MTP 深度 2 | $t_{i+2}$ 的 embedding 与深度 1 表示 | $t_{i+3}$ |

训练时这些后续 token 来自已知训练序列；生成时尚未确定的未来真值不可用，必须使用相应生成历史。更深预测层不意味着可以绕过因果约束。

## 从训练走到推理，还缺什么

V3 报告明确给出两种用法：推理可移除 MTP 模块，让主模型独立运行；也可利用这些模块提出投机草稿。[MTP in Inference](https://arxiv.org/html/2412.19437v2#S2.SS2)

后者还要求对应权重公开且可加载、引擎实现草稿路径、目标验证、采样与状态回退。收益取决于额外计算、草稿匹配程度、硬件和并发；MTP 的训练目标本身不构成加速证明。后续的 [DSpark](/inference/dspark)还研究专门的草稿模型、置信度估计和验证预算，不能只用“MTP”三个字概括。

## CPU 小实验：标签、mask 与损失

计划构造含 BOS、EOS、padding 的短序列，不下载大权重：

1. 输出主目标、深度 1 和深度 2 的位置对齐表，用可读 token 与 token ID 双重检查。
2. 给每个位置写出允许看到的 token 和目标，检查尾部越界、跨文档边界和 padding 是否按约定屏蔽。
3. 用手工 logits 计算各深度交叉熵，分别打印有效位置数、归一化方法和加权总损失。
4. 以随机小张量检查投影和共享输出头的 shape；若做最小可训练示例，明确它是教学结构，不是 V3 复训。

标签移位正确是必要检查，不能单凭 loss 下降判断模型实现正确。

## 后续 GPU 实测

先确认所选小模型是否真的提供可用 MTP 权重，以及引擎支持的准确版本。若进行辅助目标训练对比，须控制训练数据、token 预算与评价方法；若研究推理，则比较同一主模型开启／关闭受支持草稿路径的行为与成本。这是两类独立实验，不用一个结果替代另一个。

保存草稿长度、接受长度、验证耗时、总延迟和质量检查。V3 是本页固定教学对象，不是当前模型或当前引擎支持现状的代称。

## 验收产物

- 可核对的三层标签表、mask 断言和损失计算脚本。
- 一张分开的训练图与推理图，标清共享参数、输入历史、草稿及验证。
- 兼容性清单：具体模型 revision、MTP 权重、引擎 commit、采样模式；未支持项写明。

## 常见错误

- 把多步辅助预测等同于一次无条件产出多个可信 token。
- 在推理演示中输入未来真值，却当作真实生成结果。
- 将所有 MTP 论文画成独立并行头，或把 V3 的设计推广到全部后续架构。
- 用主模型示例源码证明全部训练模块已公开。源码中找不到的实现应标为未核验。

## 一手资料与版本

核验日期：**2026-11-16**。

- [DeepSeek-V3 Technical Report，arXiv:2412.19437v2](https://arxiv.org/html/2412.19437v2#S2.SS2)：2025-02-18 修订，§2.2 是本页模块与目标的依据。
- [Better & Faster Large Language Models via Multi-token Prediction，arXiv:2404.19737v1](https://arxiv.org/abs/2404.19737v1)：2024-04-30，作为不同 MTP 结构的对照。
- [V3 官方推理示例固定快照](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py)：用于对照主干与 attention 实现；不作为完整训练/MTP实现已公开的证明。
- [DeepInfer 固定快照](https://github.com/deepseek-ai/DeepInfer/tree/a679bb86ce36e883a98a97733492b686e5b9a08e)：另一代模型的运行时阅读对象，具体支持范围见 [DeepInfer 页](/inference/deepinfer)。
