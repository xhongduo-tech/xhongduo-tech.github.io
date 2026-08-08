---
title: FSDP：PyTorch 原生全分片数据并行的原理与配置
date: 2026-08-07
---

# FSDP：PyTorch 原生全分片数据并行的原理与配置

<div class="epigraph">
<p>最好的框架，是长在原生生态里的框架。</p>
<footer>—— 引意自工程选型常谚</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型微调 ｜ 大模型微调知识树 第三章 ｜ 2026-08-07</p>
</div>

## 为什么从 FSDP 开始

上一节的 DeepSpeed ZeRO 证明了「分片冗余」的威力。但 ZeRO 有个现实门槛：它是 DeepSpeed 框架的一部分，要与 PyTorch 的训练循环做额外集成。PyTorch 自己推出了 **FSDP（Fully Sharded Data Parallel，全分片数据并行）**，把 ZeRO-3 的「参数、梯度、优化器全分片」思想做成了**原生 API**。

FSDP 不是 ZeRO 的复刻，而是在模块级别重新设计的一套运行时。理解了 FSDP，你就同时理解了现代 PyTorch 生态里「多卡训练」的默认形态，也为后面 PEFT（LoRA + FSDP 的微妙关系）打好了基础。<span class="marginnote">一句话定位：<strong>ZeRO-3 是「框架级」的全分片方案，FSDP 是「框架原生」的全分片方案</strong>。两者数学等价（都是全分片数据并行），工程形态不同——FSDP 活在 torch.distributed 生态里，训练循环、DDP、torch.compile 无缝衔接。</span>

## 1 FSDP 的核心思想：模块级全分片

ZeRO-3 把整份参数切碎分到各卡；FSDP 做的是同一件事，但粒度更细——**以「FSDP 单元（module）」为分片单位**。

一个模型通常被切成许多 FSDP 单元（典型是**每层一个**单元，由 `auto_wrap_policy` 自动决定）。每个单元的参数被展平成一个一维张量（flat parameter），再均匀切分成 $N$ 份分到 $N$ 张卡。于是：

**任何一张卡都只持有每个单元参数的 $1/N$**；
前向传播到某个单元时，通过 **all-gather** 把完整参数临时拼齐，算完这一层立即释放——「用毕即弃」；
反向传播同理：需要梯度时再次 all-gather 参数，反向完 **reduce-scatter** 梯度，各自保留 $1/N$ 片段；
**优化器状态天然只存在于参数片段上**：因为优化器只更新本卡拥有的那 $1/N$ 参数，所以 $m$、$v$ 也就只存 $1/N$——不需要单独分片优化器。

这个「每层循环：all-gather 参数 → 前向 → 释放；all-gather 参数 → 反向 → reduce-scatter 梯度 → 更新片段」的节奏，就是 FSDP 的全部运行时。<span class="marginnote">与 DDP 的对照能加深印象：DDP 每卡存完整参数，只对梯度 all-reduce；FSDP 参数就是分片的，梯度 reduce-scatter 后各自更新片段。<strong>DDP 是「每卡一份模型、梯度汇合」，FSDP 是「N 卡拼一个模型、谁用谁取」</strong>。</span>

把 FSDP 的「层级循环」用伪代码摊开，运行时一目了然（以单层单元为例）：

```text
# 以单个 FSDP 单元（如一层 transformer）为例
1. 前向：对当前单元的参数做 all-gather，拼齐完整 W
2.       用完整 W 计算该层前向，得到激活输出
3.       立即释放拼齐的 W，只保留本卡持有的 1/N 分片
4. 反向：再次 all-gather 参数 W
5.       用完整 W 计算本层梯度
6.       reduce-scatter 梯度，每卡保留自己那 1/N 片段
7.       用本卡分片梯度更新本卡分片参数（优化器状态自然也是 1/N）
```

注意第 2 步与第 4 步之间的对应：**参数被拼齐两次**（前向一次、反向一次），这是 FSDP 通信「3Ψ」里前两次 all-gather 的来源。工程实现里这两次都被与相邻层的计算重叠，所以对吞吐的实际伤害远小于直觉。

## 2 公式解析：FSDP 的显存与通信模型

FSDP 的显存公式与 ZeRO-3 完全一致——因为它就是同一套分片逻辑：

$$
M_{\text{gpu}} = \underbrace{\frac{P + G + O}{N}}_{\text{分片后的静态状态}} + \underbrace{M_{\text{act}}}_{\text{激活}} + \underbrace{\text{临时拼齐的 }P}_{\text{前向/反向的瞬时峰值}}
$$

三部分逐项拆解：

- $\frac{P+G+O}{N}$：参数、梯度、优化器状态各分 $1/N$，常驻显存——这是 FSDP 省显存的主体；
- $M_{\text{act}}$：激活值，与 ZeRO 一样不参与分片，仍靠梯度检查点去压；
- **瞬时拼齐的 $P$**：前向/反向时 all-gather 出的完整参数是**临时存在**的，算完即释放——但它在那一瞬间会顶起峰值显存。这意味着 FSDP 的**峰值显存**是「分片状态 + 一层完整参数 + 激活」，而非「分片状态 + 激活」那么干净。

通信量方面，FSDP 每个 batch 大约：前向 all-gather 参数 + 反向 all-gather 参数 + reduce-scatter 梯度 ≈ **3Ψ**——与 ZeRO-3 同量级。它的三个通信阶段都能与计算**重叠**（先算前一层、同时通信后一层），实际吞吐损耗远小于数字给人的印象。<span class="marginnote">FSDP 的通信-计算重叠是它的工程精髓：PyTorch 把「通信前一层参数」与「计算当前层」流水线化，让 all-gather 的等待被计算掩盖。这也是 FSDP 在 8 卡 NVLink 环境里吞吐接近 DDP 的原因。</span>

## 3 FSDP 的三种分片策略

FSDP 通过 **`sharding_strategy`** 暴露分片程度的旋钮，恰好对应 ZeRO 的档位：

| 策略 | 分片对象 | 等价 ZeRO | 适用场景 |
| --- | --- | --- | --- |
| `FULL_SHARD` | 参数 + 梯度 + 优化器 | ZeRO-3 | 单卡装不下模型，默认最省显存 |
| `SHARD_GRAD_OP` | 梯度 + 优化器 | ZeRO-2 | 单卡装得下参数，只想省优化器 |
| `NO_SHARD` | 不分片 | 普通 DP | 兼容旧代码、需要完整参数 |

选型逻辑与 ZeRO 一致：**先看单卡能不能装下「参数 + 激活」**。能装下 → **`SHARD_GRAD_OP`** 足够且通信少；装不下 → **`FULL_SHARD`**。多数 7B 全参微调默认 **`FULL_SHARD`**，13B 以上几乎必须它。

## 4 FSDP 与 ZeRO-3 的对比与选型

两者数学等价，工程差异集中在「生态」与「细节控制」：

| 维度 | FSDP | DeepSpeed ZeRO-3 |
| --- | --- | --- |
| 归属 | PyTorch 原生 | DeepSpeed 独立框架 |
| 与训练循环集成 | 无缝（就是 torch API） | 需要 DeepSpeed engine |
| 分片单位 | 模块级（auto-wrap 逐层） | 整模型/算子级 |
| 通信重叠 | 内置、成熟 | 内置、成熟 |
| CPU offload | 支持 | 支持（更早、更成熟） |
| 生态兼容 | 与 `torch.compile`、DDP 无缝 | 有独立加速器封装 |
| LoRA 适配 | 需要 `use_orig_params=True` 等细节 | 相对更早适配 |

**选型建议**：项目已用 PyTorch 原生生态 → **FSDP**；项目依赖 DeepSpeed 的高级特性（如更细粒度的 offload、激活分区 activation partitioning）→ **ZeRO-3**。两者在 HF Trainer 里都只是几行配置的差别，迁移成本不高——与其纠结，不如用同一份数据做一次 2 卡小实验对比吞吐与显存。<span class="marginnote">一个实践偏见值得破除：<strong>FSDP 与 ZeRO-3 的显存收益几乎相同，吞吐差距通常小于 10%</strong>。选型的主要依据是「哪个与你现有的代码、生态、团队经验更兼容」，而不是性能数值的微小差异。</span>

## 5 配置与实践：auto-wrap 与 use_orig_params

FSDP 的工程细节里，最值得懂的两个概念是 **auto-wrap** 与 **`use_orig_params`**。

**auto-wrap 决定分片粒度**。模型太大时不能整模型当一个大 FSDP 单元——那样前向要 all-gather 整个模型，显存与通信都爆。默认策略按 transformer 层的结构自动包裹每一层（**`transformer_auto_wrap_policy`**），让「拼齐—计算—释放」的粒度与层对齐。**粒度越细，峰值显存越低，但通信次数越多**——这是 FSDP 的第二个权衡旋钮。

**use_orig_params 决定参数形态**。FSDP 默认把参数展平成 flat tensor，你「看不见」原始的参数对象。**`use_orig_params=True`** 时参数保持原始形态，每个参数仍是独立可访问的 tensor。这对 LoRA 尤其关键：**LoRA 需要「只训练新增的低秩参数、冻结其他」**，若参数被展平，框架难以精确定位哪些该训哪些该冻——这也是为什么 LoRA + FSDP 组合绕不开 **`use_orig_params`**（第四篇 PEFT 会深入）。

一段最小 FSDP 开启方式（HF Trainer）：

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="out",
    fsdp="full_shard auto_wrap",          # 等价 ZeRO-3 + 逐层自动包裹
    fsdp_config={
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "fsdp_transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
        "fsdp_use_orig_params": True,      # 保留原始参数形态，LoRA 必需
    },
)
```

## 6 小结

- **FSDP 是 ZeRO-3 的 PyTorch 原生实现**：模块级参数展平分片，前向/反向按需 all-gather、算完即弃。
- 显存公式 $M_{\text{gpu}} = (P+G+O)/N + M_{\text{act}} + \text{瞬时完整参数}$；通信约 3Ψ，靠流水重叠掩盖。
- 三种策略：**`FULL_SHARD`**（ZeRO-3）、**`SHARD_GRAD_OP`**（ZeRO-2）、**`NO_SHARD`**（DP）——选型看「单卡装不装得下参数」。
- 与 ZeRO-3 显存收益几乎相同、吞吐差异 <10%，选型看生态兼容性。
- **auto-wrap** 管分片粒度，**`use_orig_params`** 管参数形态——**LoRA 组合必须理解后者**。

在下一节，我们把视野从「数据维度」扩到「模型维度」：**张量并行与流水线并行在微调场景中的取舍**——为什么微调很少用它们，又为什么长序列与超大模型逃不开。
