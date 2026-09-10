---
title: 领域适配微调
date: 2026-09-08
section: llm
---

# 领域适配微调

<div class="epigraph">
<p>先用领域无标注文本把词与语篇分布拉近，再用少量领域指令对齐接口；把说明书直接当 ShareGPT 来 SFT，术语没进先验，格式倒先过拟合。</p>
<footer>—— Gururangan 等，Don't Stop Pretraining（EMNLP 2020）；对照指令谱系与 LIMA 的「接口对齐」分工</footer>
</div>

[长上下文微调](/llm/long-context-finetune)处理的是位置与长度。领域适配处理的是**词表与语篇不在预训练里**：生物医学缩写、内部 API、法条结构。Gururangan 等人 2020 年的 Don't Stop Pretraining 把这条写成两段：领域自适应预训练（DAPT）与任务自适应预训练（TAPT），然后再上任务头。指令时代的缺口是有人用[Alpaca 谱系](/llm/instruction-data-lineage)一步代替 DAPT，期望模型从 2 万条「解释一下 ICD 码」里长出医学语言模型。本课把继续预训练与领域 SFT 拆开，不把 SFT 对 RL 的样本效率（[下一课](/llm/sft-vs-rl-efficiency)）提前写完。

## 问题

通用基座在领域文本上的困惑度高，是因为 n-gram 与语篇结构 OOD，不是因为不会[chat template](/llm/chat-template)。SFT 的[仅回复损失](/llm/response-only-loss)只更新回答侧，提示里的术语作为条件出现，但模型没有在领域分布上做过「下一词」的密集练习，生成时仍会退回网页腔、杜撰术语。反过来，只做领域继续预训练不做指令 SFT，模型会续写论文，不会按用户祈使句停止。

两条目标正交：DAPT 改的是 $p_{\mathrm{domain}}(x)$，领域 SFT 改的是 $p(\mathrm{assistant}\mid \mathrm{domain\ prefix})$。用后者代替前者，等于用稀疏监督去改语言模型先验，样本效率极差，还容易遗忘通用能力。

### 领域数据几乎总是全文目标

DAPT 对领域语料做因果（或掩码）语言建模，**不是**仅回复。这与 SFT 课的掩码契约相反，必须在配方里分成两个阶段，而不是共用一个 `train_on_inputs=False`。TAPT 更窄：在即将做的任务的无标注文本上再继续预训练（如目标论文集），再 SFT。

<span class="marginnote">Gururangan 等人的实验在 BERT 族与分类任务上；解码器 LLM 的对应物是：领域 corpus 上短学习率的继续预训练，再接指令。主张迁移：先适应分布，再适应接口。</span>

## 方法

阶段 A：收集领域无标注文本（手册、论文、清洗后的工单正文），按预训练方式打包，全文损失，学习率低于原预训练、高于激进全参 SFT，epoch 很少，监控通用基准与领域困惑度。可用 [LoRA](/llm/lora) 做便宜 DAPT，但术语若要求进嵌入，应考虑解冻 embedding 或全文。

阶段 B：少量领域指令，经该基座的[模板](/llm/chat-template)渲染，[仅回复](/llm/response-only-loss)，$\eta$ 按[LoRA 对全参](/llm/lora-vs-full-lr)选。示范要像真实接口（病历问答、条款抽取），不要把教科书章节伪造成用户问题除非产品就是那样。

```mermaid
flowchart LR
  G["通用预训练"] --> A["DAPT：领域全文 LM"]
  A --> B["领域指令 SFT"]
  B --> P["产品接口"]
```

配比：阶段 B 可混少量通用指令，减少格式遗忘。不要把阶段 A 的论文 PDF 未经清洗塞进对话字段。许可与隐私：领域语料常含 PII，这是数据工程，不是学习率。

## 机制

继续预训练把嵌入与注意力重新对准领域共现：缩写展开、公式上下文、引用语类。之后 SFT 只需把已经较低的领域困惑度接到助手完成上，示范可以少——这与 [LIMA](/llm/lima)「能力在预训练里」同构，只是「预训练」多了一段领域。跳过 DAPT 时，SFT 梯度既要教格式又要教词，低秩适配器往往只记住示范里出现过的术语，一出表就幻觉。

遗忘：阶段 A 全参会冲刷通用网页特征；LoRA DAPT 忘得少，领域先验也可能学不足。应用 Biderman 等人的「学得少、忘得少」判断：术语密集任务倾向更多参数或解冻底层；通用助手 + 薄领域倾向 LoRA。

<span class="marginnote">检索增强可以代替一部分 DAPT：把领域知识留在索引里。本课不展开 RAG。选择是产品策略：要内化流程与口吻，用适配；要时效条文，用检索。</span>

## 边界与工程取舍

领域 SFT 不能修复错误的教师谱系：用通用 ChatGPT 导出的「医学」ShareGPT，会注入自信的错误。领域示范应专家写或专家滤。长上下文领域（法规全书）要叠加[上一课](/llm/long-context-finetune)的位置方案，但先有领域先验再拉窗口。

评测：领域任务 + 通用保持 + 幻觉抽检。只报领域准确率会选出背了指南全文、不会闲聊也不会拒答的检查点。NEFTune 在术语精确匹配任务上要保守。

## 小结

- 领域适配先 DAPT（全文 LM）再领域指令 SFT（仅回复），不要用对话表代替分布适应。
- 两阶段的损失与学习率契约不同，不可共用一份掩码配置。
- 术语要进先验；接口对齐用少量真示范。
- LoRA 减少遗忘也限制内化；嵌入是否解冻单独决策。
- 评测保留通用能力与幻觉，不只报领域准确率。
- 出处：Gururangan 等，Don't Stop Pretraining，EMNLP 2020；指令阶段承接 FLAN / Alpaca / ShareGPT 谱系与 LIMA 的接口假说。
