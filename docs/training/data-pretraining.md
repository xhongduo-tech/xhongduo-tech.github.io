---
title: 数据与预训练
description: 用数据清单、去重与污染审计连接next-token目标、CPT和独立评测。
kind: topic
status: planned
---

# 数据与预训练

::: info 待研究专题
下列阅读与实验均待执行。CPU机制验证、小模型训练和前沿规模预训练分别验收，不预设训练收益。
:::

## 问题与前置依赖

为什么更多文本不一定带来更好的任务能力？如何发现重复、来源偏差和训练测试污染？继续预训练（CPT）究竟延续了什么，又可能改变什么？

前置是[编程与数学](/foundations/programming-math)、tokenizer、因果语言模型和[评测](/evaluation/)。先能解释一次token预测与参数更新，再讨论数据配比和预算，不以海量语料作为起点。

## 核心机制

本页以decoder-only的next-token语言建模为例：

$$
\mathcal L_{\rm LM}=-\frac{1}{M}\sum_{d,t}m_{d,t}\log p_\theta(x_{d,t}\mid x_{d,<t}),\qquad M=\sum_{d,t}m_{d,t}.
$$

`m`标识实际参与目标的token。样本拼接、文档边界、截断和padding决定模型能看到什么、在哪些位置学习；“原始字节相同”和“token数相同”也不是同一预算。

CPT是在既有检查点上继续预训练。这里仍使用语言建模目标，以观察领域数据适配；不能把它直接称为SFT，也不能由领域loss下降推断通用能力提升。[领域适配预训练论文](https://arxiv.org/abs/2004.10964v3)是概念入口，其原始实验设定与当前因果模型实验需分别说明。

去重分为文档级精确匹配、近重复以及长重复子串。污染检查关注训练内容与验证/测试内容的重叠和衍生关系；仅去掉完全相同的字符串不足以排除改写、同源文档和答案泄漏。[去重研究](https://arxiv.org/abs/2107.06499v2)给出这些问题的实验依据，论文成绩不当作本页实验结果。

## 规模规律与预算

[Chinchilla研究](https://arxiv.org/abs/2203.15556v1)把给定训练计算预算下的模型参数量与训练token数共同作为选择对象。学习时先区分参数、有效训练token、step、epoch与实测计算时间；只增加模型或只增加重复数据，不自动得到计算最优方案。

公开规模规律是特定模型、数据和预算区间的经验研究，不能将某个参数/token比例直接当作所有MoE、多模态或后训练的普遍配方。资源账本还要另记激活、优化器、通信以及后续推理成本。本路线先做token预算与数据分布检查，不要求用有限算力重做大规模scaling研究。

## 数据格式与审计

以下是合成教学记录，不是真实训练集：

```json
{"id":"doc-001","source_group":"manual-a","text":"合成教学文本。","license":"synthetic","split":"train","sha256":"由预处理计算"}
```

数据清单至少保存来源、许可或使用依据、语言/领域、时间范围、文档族、规范化规则、内容哈希和版本。先按来源或同源簇设计切分，再检查簇内及跨集合重复；清洗规则和阈值也要版本化，不能让测试结果反向决定训练样本选择。

预处理后额外保存token数、文档边界、EOS策略、packing方法和有效label数。测试答案不进入训练样本或生成提示；对公开基准做污染审计时只报告检测规则和处理流程，不把测试集改造成调参集。

## 实现与源码阅读切入

从读取、规范化、生成候选重复、判定重复、选择保留项这条数据流开始。先查看[官方去重仓库的固定快照](https://github.com/google-research/deduplicate-text-datasets/tree/4e9888ac3f95dc4f6169867a04c4c19df02dafe3)及其中README、`scripts/make_suffix_array.py`和`src/main.rs`，理解长子串去重与近似文档相似度的不同对象，不直接复制全量处理配置。

再对照[DeepSeek V3报告](https://arxiv.org/abs/2412.19437v2)的数据构造、训练与评测章节，记录其方法、披露边界和自己的待验证假设；不据此宣称获得了同样的数据或训练配方。

## 分级实验

| 层级 | 拟执行实验 | 对照与限制 |
| --- | --- | --- |
| CPU数据检查 | 构造少量精确重复、轻微改写和同源文档；比较规范化哈希与字符n-gram相似度 | 人工记录真重复/误判/漏判；小规模成对比较不是大规模MinHash或后缀数组性能复现 |
| CPU目标检查 | 用短token序列计算标签偏移、文档边界与padding mask；完成tiny模型一次更新 | 对比“边界正确”和一个故意错误的配置，检查被计分的token，不由单步loss推泛化 |
| 可选GPU | 一个小检查点、固定token/step预算，比较CPT前后或一种数据配比变化 | 其余配置固定；同时保留领域任务与通用回归集，记录资源和失败，不预设提升 |

## 指标与验收产物

- 数据清单、切分图、重复簇和污染审计报告，保留去除理由及人工抽查。
- token/label示意、有效token计数、原始训练和验证记录。
- 相同tokenizer与协议下的loss或perplexity、固定任务质量和能力退化；不同分词器的PPL不能直接排行。
- 对照报告写清预算、数据比例、随机性和仍无法排除的解释；无提升也保存结果。

## 常见陷阱

全局按行随机切分导致同一文档的近重复分散到测试；把相似但独立的有效样本全部删掉；清洗后不核对数据分布；以epoch而非实际token比较不等量语料；把低验证loss当作没有污染的证明。合成数据也需要来源、生成模型和筛选规则记录。

## 第一方资料

核验日期：2026-11-16。仓库链接固定commit；实验时另锁定实际依赖、数据与模型。

- [Deduplicating Training Data Makes Language Models Better v2](https://arxiv.org/abs/2107.06499v2)：重复、记忆与训练测试重叠。
- [作者官方去重实现](https://github.com/google-research/deduplicate-text-datasets/blob/4e9888ac3f95dc4f6169867a04c4c19df02dafe3/README.md)：处理对象与运行入口。
- [Don't Stop Pretraining v3](https://arxiv.org/abs/2004.10964v3)：领域/任务适配的研究问题。
- [DeepSeek V3报告 v2](https://arxiv.org/abs/2412.19437v2)：训练数据、预算与评测的系统案例。

关联：[训练地图](/training/) · [SFT](/training/sft) · [分布式训练](/training/distributed-training)。
