---
title: LLaMA-Factory 实战：配置文件、WebUI 与 LoRA/QLoRA 全流程
date: 2026-08-07
---

# LLaMA-Factory 实战：配置文件、WebUI 与 LoRA/QLoRA 全流程

<div class="epigraph">
<p>把「微调」从「写代码」变成「填表单」——这是 LLaMA-Factory 的承诺。</p>
<footer>—— 引意自 LLaMA-Factory 设计理念</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型微调 ｜ 大模型微调知识树 第八章 ｜ 2026-08-07</p>
</div>

## 为什么从 LLaMA-Factory 实战开始

上一节选型，LLaMA-Factory 的定位是「低门槛全家桶」——**WebUI 点点点、几十种方法、一键 LoRA/QLoRA**。本节把它从「听说过」变成「会用了」：数据怎么注册、配置怎么写、WebUI 怎么点、LoRA/QLoRA 全流程怎么走、训完怎么导出。

LLaMA-Factory 是最适合「第一次上手微调」的框架——因为它的每一步都有 WebUI 兜底。但**「会用按钮」不等于「懂配置」**：本节在教流程的同时，会标注「这个配置对应前七篇的哪个概念」——让点按钮的你，心里清楚每个选项在干什么。<span class="marginnote">LLaMA-Factory 的中文社区文档完善、模型支持广，是「中文用户做微调」的事实首选。但它的「开箱即用」也是一把双刃剑：<strong>容易让人「不问为什么就点」</strong>——本节的原则是「流程照走、概念要懂」，避免「按教程跑通了但说不清发生了什么」。</span>

## 1 LLaMA-Factory 是什么：先看清「全家桶」的组成

LLaMA-Factory 的核心组件与「前七篇的对应关系」：

| LLaMA-Factory 组件 | 对应前文概念 | 作用 |
| --- | --- | --- |
| `dataset_info.json` | 数据工程（第二篇） | 注册数据集 |
| `stage`（pt/sft/rm/ppo/dpo/orpo…） | 微调范式（一、六篇） | 选择训练阶段 |
| `finetuning_type`（lora/qlora/full/dora/pissa…） | PEFT（第四篇） | 选择参数高效方法 |
| `template` | 对话模板（第二篇） | 选择 chat 模板 |
| `quantization_bit` | QLoRA（第四篇） | 基座量化 |
| WebUI / YAML | — | 配置入口 |

它的设计哲学是「**一个入口、全流程覆盖**」：从数据到训练到导出，都在同一个框架里完成，不必在多个工具间切换。安装也简单——`pip install llama-factory` 即可（官方推荐用 git 克隆 + conda 环境）。

## 2 数据准备：dataset_info.json 与自定义数据集

LLaMA-Factory 的数据入口是 `dataset_info.json`——一个「数据集注册表」：每一条记录声明「数据集文件 + 格式类型」：

```json
{
  "my_custom": {
    "file_name": "data/my_custom.json",
    "format": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

要点：

**format**：数据格式类型——**alpaca**（三字段）、**sharegpt**（多轮）、**messages**（messages）等，对应第二篇《数据格式与工具链》；
**columns**：把自定义数据集的字段名映射到框架约定的字段——**字段名不统一是配置阶段最常见的报错**；
注册后，在 WebUI/CLI 里用数据集名（如 `my_custom`）引用即可。

**数据准备的四步**：① 把数据整理成标准格式（Alpaca/ShareGPT）→ ② 放进 `data/` 目录 → ③ 在 `dataset_info.json` 注册 → ④ 在训练配置里引用。**格式不对，训练会直接报错**——所以先跑通「1 条数据的小样本」再上全量，是最稳的上手方式。

## 3 训练配置：YAML 与 WebUI 两条路

LLaMA-Factory 支持两种配置方式：

**方式一：WebUI（图形界面）**。执行 `llamafactory-webui` 启动，在网页里选「模型、方法、数据、超参」，点「开始训练」。适合**第一次上手、或非工程师**——每个选项都有下拉框与说明。

**方式二：YAML 配置文件**。把训练参数写进 YAML，命令行执行——适合**复现与批量**：

```yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
stage: sft                    # 范式：sft
finetuning_type: lora         # PEFT：lora
dataset: my_custom            # 数据集名（上一步注册的）
template: qwen                # 对话模板
lora_rank: 8                  # LoRA 的秩
lora_alpha: 16
learning_rate: 2.0e-5
num_train_epochs: 3.0
output_dir: output/qwen-sft-lora
```

**YAML 里每个字段都是「前七篇的概念落点」**：`stage` 是范式、`finetuning_type` 是 PEFT、`template` 是对话模板、`lora_rank` 是 LoRA 的秩……**读懂一个配置文件，等于把前七篇的配置项串了一遍**。<span class="marginnote">WebUI 与 YAML 不是二选一：<strong>WebUI 适合「探索与调试」，YAML 适合「记录与复现」</strong>。一个成熟的工作流是「先在 WebUI 里点通一条配置 → 导出成 YAML → 用 YAML 跑批量/重跑」。WebUI 生成的配置也可以「另存为 YAML」，两条路无缝衔接。</span>

## 4 全流程：从数据到导出的 LoRA/QLoRA

一次 LLaMA-Factory 的 LoRA 微调全流程，五步：

**① 准备数据**：注册数据集（第 2 节）。

**② 配置训练**：WebUI 或 YAML 选好「模型 + stage=sft + finetuning_type=lora + 数据 + 超参」。

**③ 开始训练**：WebUI 点「开始」或 CLI 执行 `llamafactory-cli train config.yaml`。训练日志、loss 曲线在 WebUI 实时可见。

**④ 测试效果**：训练完成后，在 WebUI 的「Chat」页加载「基座 + 训练好的 LoRA」，直接对话看效果——**不导出也能先验证**。

**⑤ 导出/合并**：LoRA 只是「小适配器」，部署通常要**合并进基座**：

```yaml
# export_config.yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
adapter_name_or_path: output/qwen-sft-lora
template: qwen
finetuning_type: lora
export_dir: output/qwen-merged
export_size: 2
export_legacy_format: false
```

**QLoRA 的差别**：只需在配置里加一行 `quantization_bit: 4`（4-bit 基座）——训练流程完全一样，只是**导出前要把量化基座反量化**（`export_quantization_bit` 不设即反量化，或按需导出），这正是 QLoRA 篇讲的「合并时先反量化」。

## 5 实战要点与常见问题

LLaMA-Factory 实战的几个高频经验：

**① 先跑小样本**：用 1–10 条数据把「数据格式 → 训练 → 导出」全链路跑通，确认格式无误，再上全量——**省下「训了 10 小时后才发现格式错」的惨剧**。

**② template 必须选对**：对话模板（`template`）必须与模型的 chat 模板一致——选错，模型训练后「不听话」（第二篇《对话模板》）。**模型名自带模板匹配**（如 Qwen 模型自动选 qwen 模板），自定义模型要手动确认。

**③ 断点续训**：训练中断后，加上 `resume_from_checkpoint: true` 从最近 checkpoint 续训——对应第八篇《训练流水线工程化》的「断点续训」。

**④ 显存不够就 QLoRA**：7B 全参 LoRA 要 24GB+，显存紧张时切 `quantization_bit: 4`（QLoRA）——**同样的数据与流程，显存骤降**（第四篇《QLoRA》）。

**⑤ 看日志别只盯 loss**：WebUI 的日志除了 loss，还有「学习率、显存、吞吐」——**显存接近上限、吞吐骤降**都是「配置需要调」的信号（第三篇《训练吞吐》）。<span class="marginnote">一个常见困惑：「训练完的 LoRA 在哪、怎么部署？」——<strong>输出目录里是 adapter 权重（`adapter_model.safetensors` + `adapter_config.json`），很小（几十 MB）</strong>。部署有两种姿势：合并成完整模型（`llamafactory-cli export`），或「基座 + adapter」动态加载（第四篇《LoRA 工程细节》的多适配器模式）。选哪种，取决于你要「一个干净模型」还是「多任务共存」。</span>

## 6 小结

- **LLaMA-Factory 是「全家桶」**：一个入口覆盖「数据 → 训练 → 导出」，WebUI 让微调变成「填表单」。
- **数据入口**：`dataset_info.json` 注册表——`format` 定格式、`columns` 做字段映射；格式错是最常见报错。
- **两条配置路**：WebUI（探索/调试）与 YAML（记录/复现）；字段都是前七篇概念的落点。
- **LoRA/QLoRA 全流程五步**：准备数据 → 配置 → 训练 → Chat 测试 → export 合并；QLoRA 只是加 `quantization_bit: 4`。
- **实战五经验**：先跑小样本、template 选对、断点续训、显存不够切 QLoRA、看日志不止盯 loss。

在下一节，我们看配置驱动的另一员：**Axolotl 实战——YAML 声明式配置与多卡训练**。
