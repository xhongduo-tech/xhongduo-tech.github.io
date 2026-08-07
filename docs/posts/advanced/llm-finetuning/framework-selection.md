---
title: 微调框架选型：LLaMA-Factory、Axolotl、ms-swift 与 TRL 横向对比
date: 2026-08-07
---

# 微调框架选型：LLaMA-Factory、Axolotl、ms-swift 与 TRL 横向对比

<div class="epigraph">
<p>框架是脚手架，不是目标——选它，是为了更快地把模型训练出来。</p>
<footer>—— 引意自工程选型共识</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型微调 ｜ 大模型微调知识树 第八章 ｜ 2026-08-07</p>
</div>

## 为什么从框架选型开始

前七篇讲完了微调的「理论」——范式、数据、显存、PEFT、长序列、对齐、领域。但理论到落地之间，隔着一个现实问题：**用什么框架跑**？微调框架决定了「你写多少代码、踩多少坑、能不能复现别人的配方」。

本节先做**选型**——四个主流框架（**LLaMA-Factory、Axolotl、ms-swift、TRL**）的横向对比。选型不是「哪个最好」，而是「哪个匹配你的场景」：有人要 WebUI 点点点、有人要 YAML 声明式、有人要编程接口、有人要国产模型适配。把四者的定位、优劣、适用场景讲清，后面的五篇「实战」就能各就各位。<span class="marginnote">一句话定位：<strong>TRL 是「编程库」、Axolotl 是「配置驱动」、LLaMA-Factory 是「全家桶 + WebUI」、ms-swift 是「国产模型生态」</strong>——它们不是替代关系，而是「同一件事的四种打开方式」。</span>

## 1 四个框架的「出身」与定位

先看每个框架从哪来、为什么存在：

**TRL（Transformers Reinforcement Learning）**——HuggingFace 出品，与 Transformers 生态同源。定位是**编程接口库**：它给 SFT、DPO、PPO 提供 Trainer（`SFTTrainer`、`DPOTrainer`、`PPOTrainer`），但你要**写代码**来用。适合「要深度定制、想控制每一步」的开发者。

**Axolotl**——社区开源，定位是**YAML 声明式配置**：写一个 YAML 文件描述「用什么模型、什么数据、什么超参」，一条命令跑训练。适合「快速复现配方、不想写训练循环」的团队，尤其擅长多卡与各种模型架构。

**LLaMA-Factory**——国内开源，定位是**全家桶 + 低门槛**：内置 WebUI（图形界面）、支持几十种模型、几十种训练方法、一键 LoRA/QLoRA 全流程。适合「想最快上手、包括非工程师」的用户。

**ms-swift**——魔搭（ModelScope）出品，定位是**国产模型生态**：对 Qwen、GLM、DeepSeek 等国产模型适配最好，且与「魔搭平台的部署/评测」无缝衔接。适合「用国产模型 + 想一并解决部署」的团队。

## 2 横向对比：四个维度

把四个框架放在四个关键维度上对比：

| 维度 | TRL | Axolotl | LLaMA-Factory | ms-swift |
| --- | --- | --- | --- | --- |
| 使用方式 | 编程接口 | YAML 配置 | WebUI / CLI / 代码 | CLI / 代码 |
| 入门门槛 | 中高（要写代码） | 中（学 YAML） | 低（点点点） | 中 |
| 可定制性 | 最高（全代码） | 中（配置 + 钩子） | 中（配置为主） | 中高 |
| 训练方法覆盖 | SFT/DPO/PPO 为主 | 广（SFT/DPO/ORPO/…） | 最广（几十种） | 广 |
| 国产模型适配 | 一般 | 一般 | 好 | **最好** |
| 多卡/并行 | 需配 DeepSpeed/FSDP | **内置多卡友好** | 支持 | 支持 |
| WebUI | 无 | 无 | **有** | 无 |
| 部署衔接 | 需自接 | 需自接 | 一般 | **最好**（魔搭生态） |

**读表的三个结论**：

1. **想写代码控制一切 → TRL**；不想写代码 → LLaMA-Factory / Axolotl；
2. **用国产模型 → ms-swift**；要开箱即用 → LLaMA-Factory；
3. **多卡生产级配置 → Axolotl**（它的 YAML 对多卡并行支持最成熟）。

## 3 按场景选型：三个典型用户

把「选哪个」落到三个具体场景：

**场景 A：研究者 / 工程师，要深度实验**。要调 loss、改采样、加自定义组件——**选 TRL**。它是唯一「把训练循环暴露给你」的框架，虽然要写代码，但可定制性最高。前七篇讲的每个技术（DPO、GRPO、LoRA）在 TRL 里都有对应的 Trainer 或扩展点。

**场景 B：团队快速交付，复现社区配方**。要在几天内训出一个可用模型、且要复现别人的配置——**选 Axolotl 或 LLaMA-Factory**。Axolotl 的 YAML 适合「改参数跑」；LLaMA-Factory 的 WebUI 适合「非工程师也能操作」。

**场景 C：国产模型 + 端到端**。模型用 Qwen/GLM，还要一并解决部署与评测——**选 ms-swift**。它与魔搭平台的衔接最顺，训练完直接对接部署/评测工具链。

**一句话**：**要代码控制选 TRL，要配置驱动选 Axolotl，要低门槛选 LLaMA-Factory，要国产生态选 ms-swift**。

## 4 选择框架时还要看什么

除了「定位」，选型还要看四个「隐藏维度」：

**① 训练方法覆盖**。你要用的方法（DPO？ORPO？GRPO？）框架支持吗？**LLaMA-Factory 的方法覆盖最全**，TRL 紧随其后，Axolotl 也广——选之前先查「方法支持矩阵」。

**② 模型兼容性**。你的模型（Llama？Qwen？GLM？）在框架里开箱即用吗？国产模型框架（LLaMA-Factory、ms-swift）对国产模型适配更好。

**③ 数据格式支持**。框架内置的「数据预处理」（对话模板、loss mask、packing）与你手头的数据格式是否匹配？**格式不匹配，是配置阶段最常见的翻车点**（第二篇《数据格式与工具链》）。

**④ 社区活跃度**。GitHub star、issue 响应、教程数量——决定「遇到坑时有没有人踩过」。**LLaMA-Factory 与 TRL 的社区最大**，坑基本都被填平了。

一个务实建议：**不要「只认一个框架」**——用 TRL 写「需要定制的部分」，用 LLaMA-Factory 或 Axolotl 跑「标准配方」，用 ms-swift 做「国产模型部署」——**框架是工具，混着用很正常**。<span class="marginnote">框架选型最容易被忽略的坑是「<strong>版本漂移</strong>」：框架更新很快，教程/配方常基于旧版本——<strong>YAML 的字段、Trainer 的参数、数据格式的约定会随版本变</strong>。复现配方时，先确认「配方基于哪个框架版本」，再决定是否升级——「照抄最新版教程跑旧配方」是新手最常踩的坑。</span>

### 同一个 LoRA 任务，四个框架的「打开方式」

把「对 7B 模型做一次 LoRA 微调」这个相同任务，在四个框架里各写一遍，差异立刻具象：

```yaml
# Axolotl：声明式，一段 YAML
base_model: Qwen/Qwen2-7B
datasets: [{path: alpaca.json, type: alpaca}]
adapter: lora
lora_r: 16
```

```python
# TRL：编程式，代码控制每一步
trainer = SFTTrainer(model=model, args=args,
                     train_dataset=dataset,
                     peft_config=LoraConfig(r=16))
trainer.train()
```

```bash
# LLaMA-Factory：CLI 一行（或用 WebUI 点点点）
llamafactory-cli train \
  --model_name_or_path Qwen/Qwen2-7B \
  --stage sft --finetuning_type lora \
  --dataset alpaca --lora_rank 16
```

```bash
# ms-swift：CLI 一行，国产生态
swift sft --model Qwen/Qwen2-7B \
  --dataset alpaca.json --lora_rank 16
```

同一件事，四种姿势：**Axolotl 写 YAML、TRL 写代码、LLaMA-Factory 与 ms-swift 敲命令行**。选型就是问自己「你更愿意跟哪种姿势相处」。## 5 选型决策清单

收成一张「选型前的自问清单」：

1. **我要不要写训练代码？** 要 → TRL；不要 → LLaMA-Factory / Axolotl；
2. **我的模型是国产还是开源通用？** 国产 → ms-swift；通用 → 其他；
3. **我要不要 WebUI？** 要 → LLaMA-Factory；
4. **我要不要复现社区配方？** 要 → Axolotl / LLaMA-Factory（配方多）；
5. **我要不要一并部署？** 要 → ms-swift（魔搭生态）；
6. **我的方法框架支持吗？** 选之前查「方法支持矩阵」。

六个问题答完，选型基本清晰。**没有「最好」的框架，只有「最匹配你场景」的框架**——记住这点，选型就不焦虑。

## 6 小结

- **四个框架四定位**：TRL（编程库）、Axolotl（YAML 配置）、LLaMA-Factory（全家桶 + WebUI）、ms-swift（国产生态）。
- **四维对比**：使用方式、可定制性、方法覆盖、国产适配——没有全能冠军，各有取舍。
- **三场景三选**：研究者选 TRL、快速交付选 Axolotl/LLaMA-Factory、国产端到端选 ms-swift。
- **隐藏维度**：方法支持矩阵、模型兼容、数据格式、社区活跃度——选型不能只看定位。
- **框架是工具**：混着用很正常；小心「版本漂移」导致的配方失效。

在下一节，我们进入第一个实战：**LLaMA-Factory 实战——配置文件、WebUI 与 LoRA/QLoRA 全流程**。
