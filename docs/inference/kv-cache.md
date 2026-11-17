---
title: KV Cache与注意力执行
description: 从因果复用到分页、前缀缓存与内存预算，分清存储布局和注意力内核。
kind: topic
status: planned
---

# KV Cache 与注意力执行

::: info 专题计划 · 待研究
计划先实现小张量 reference 和内存估算，再测受支持的模型与引擎。尚无本人的缓存性能实测。
:::

## 要解决的问题与前置依赖

在因果自注意力中，已有 token 的表示不依赖未来 token。生成新 token 时可以复用之前计算的 K、V，避免重复投影；但缓存随上下文和并发增长，存储、分配、命中和回收会成为服务约束。

先会矩阵乘法、softmax、因果 mask、head shape 和位置编码，再区分 prefill 的批量输入计算与 decode 的逐步追加。[基础地图](/foundations/)是先修入口；这里只讨论机制，不将任意模型都假定成相同的 attention 结构。

## 机制与容量公式

省略 head 索引，新位置的注意力可写为：

$$
o_t=\operatorname{softmax}
\left(\frac{q_tK_{\le t}^{\mathsf T}}{\sqrt{d_h}}\right)V_{\le t}.
$$

保留历史 K、V 后，当前 query 仍需访问有效历史；KV Cache 不会让长上下文注意力自动变成常数成本。位置编码、因果边界、dtype 与缓存追加顺序必须一致。

对每层 K/V head 数及维度相同、K 与 V 同精度的普通 MHA／GQA，未计额外开销的估算为：

$$
M_{\mathrm{KV}}\approx
2L H_{\mathrm{KV}}d_h
\left(\sum_{i=1}^{B}T_i\right)\frac{b}{8}
\quad\text{bytes}.
$$

$L$ 为层数，$H_{\mathrm{KV}}$ 为 KV head 数，$d_h$ 为每个 head 的维度，$T_i$ 为各请求当前缓存长度，$b$ 为每元素位数。它不含权重、临时区、量化 scale、块表、预留空间或设备副本；并行分片还须按实际布局计算。

**适用边界：** MLA 的压缩状态、不同 K/V 维度、混合 attention 层、滑动窗口和后续架构都需要重新建模。不能把这个式子直接套到 [V3 的 MLA](https://arxiv.org/html/2412.19437v2#S2.SS1.SSS1)或 [V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)；必须检查准确模型 revision 和执行实现。

## 五种不同的优化对象

| 机制 | 改变什么 | 不应混淆的边界 |
| --- | --- | --- |
| KV Cache | 保存已有 token 的 K、V，复用计算 | 本身不规定分页、淘汰或跨请求共享 |
| PagedAttention | 以块表将逻辑 token 映射到物理 KV 块，并配套访问内核 | 不是把上下文内容压缩成更少 token |
| FlashAttention | 分块计算精确 attention，减少 HBM 与片上存储间的数据移动 | 不是分页分配器；dense attention 的算术量并未因此变成线性 |
| prefix caching | 在有效键匹配时复用既有前缀计算 | 文本看起来相同，不保证 token、adapter、位置或多模态输入相同 |
| offload／迁移 | 将状态放到其他存储层，或在设备间移动 | 增加传输、同步和失效管理成本，不保证更快 |

分页机制依据 [PagedAttention 论文](https://arxiv.org/abs/2309.06180v1)，IO 分块依据 [FlashAttention 论文](https://arxiv.org/abs/2205.14135v2)。[vLLM 的 prefix caching 设计](https://docs.vllm.ai/en/latest/design/prefix_caching/)使用前缀关联的块键，并考虑 LoRA、多模态输入和隔离盐等额外标识；跨配置复用需满足对应实现的有效性条件。

若不共享且只考虑最后一块填不满，块长为 $S$ 时，浪费的 token 槽位可用下式检查：

$$
W_{\mathrm{slots}}
=S\sum_i\left\lceil\frac{T_i}{S}\right\rceil-\sum_iT_i.
$$

这是教学分配模型，不包含外部碎片、引用计数、copy-on-write、块预留或真实 allocator 的全部开销。

## CPU 小实验：结果不随布局改变

计划使用 NumPy 小张量，先固定随机种子和浮点容差：

1. 每一步从完整历史重算 attention，作为 reference；再实现连续 KV 追加，比较各步输出。
2. 给相同 K、V 建立块表并打乱物理块位置，通过逻辑索引 gather 后计算；结果应在容差内一致。
3. 用不同长度请求检查槽位估算，加入越界块号、错误有效长度和请求复用后的脏状态，确认断言能发现错误。
4. 为 prefix key 制造“相同文字但不同 token／adapter 标识”的反例，验证错误复用为何不成立。

这里的 gather 示例不是 GPU PagedAttention 内核复现；小张量运行时间也不能推断真实访存性能。

## 后续 GPU 实测

选一个引擎明确支持的小模型，冻结模型、tokenizer、KV dtype 和引擎版本。先验证生成行为，再在固定输入长度与并发下记录实际缓存占用、峰值显存、抢占／重算、TTFT 和吞吐。

prefix cache 实验分别报告冷缓存和暖缓存，说明命中 token 数、请求构成与隔离配置。测试 offload 时另记录传输路径和带宽；每轮只改一项，避免把不同机制的作用混在一起。

## 验收产物与常见错误

交付容量估算表、三种布局的输出对照脚本、块表／前缀键示意、有效长度和回收边界测试，以及待实测或实测的独立记录。

常见错误包括用 query head 数代替 KV head 数、把字节和 bit 混用、只算权重就估并发、把 prefix 命中归因于解码算法，或以 FlashAttention 的 IO 优化解释缓存分配收益。

## 一手资料与版本

核验日期：**2026-11-16**。

- [PagedAttention，arXiv:2309.06180v1](https://arxiv.org/abs/2309.06180v1)：2023-09-12，分页与 KV 共享的原始论文。
- [FlashAttention，arXiv:2205.14135v2](https://arxiv.org/abs/2205.14135v2)：2022-06-23 修订，精确注意力的 IO 优化。
- [vLLM Automatic Prefix Caching 设计](https://docs.vllm.ai/en/latest/design/prefix_caching/)：滚动文档，按核验当日页面理解，实际实验另记安装版本。
- [V3 报告 v2，MLA 小节](https://arxiv.org/html/2412.19437v2#S2.SS1.SSS1)、[V3 attention 示例固定快照](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L396)：用于说明通用 KV 估算的架构边界。
