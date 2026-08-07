---
title: Axolotl 实战：YAML 声明式配置与多卡训练
date: 2026-08-07
---

# Axolotl 实战：YAML 声明式配置与多卡训练

<div class="epigraph">
<p>一份 YAML，讲清「训什么、怎么训、用几张卡」——剩下的交给框架。</p>
<footer>—— 引意自 Axolotl 设计理念</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型微调 ｜ 大模型微调知识树 第八章 ｜ 2026-08-07</p>
</div>

## 为什么从 Axolotl 实战开始

上一节的 LLaMA-Factory 是「低门槛全家桶」，而 **Axolotl** 走的是另一条路：**YAML 声明式配置 + 多卡生产级训练**。它没有 WebUI，但它的 YAML 配置极其强大——一份配置文件就能描述「模型、数据、PEFT、RL、并行、日志」的全部细节，且**对多卡（accelerate/DeepSpeed）与各种新架构的支持非常及时**。

Axolotl 在「社区配方（recipe）」文化里地位很高——OpenAccess 等组织发布的大量微调配方都是 Axolotl 格式。**会读 Axolotl 的 YAML，等于会读社区一半的微调配方**。本节把 YAML 解剖透、把多卡配置讲清、把「从 YAML 到合并」的全流程走完。<span class="marginnote">一句话定位：<strong>Axolotl = 「配置即配方」</strong>——它把「训练」抽象成一个声明式问题：「这份 YAML 描述的配置」，而不是「这段 Python 写的流程」。它比 LLaMA-Factory 更「硬核」（要懂 YAML、懂并行），但换来的是「对多卡与复杂配方的掌控力」。</span>

## 1 YAML 配置解剖：一份完整的 Axolotl 配方

Axolotl 的配置是一个「大而全」的 YAML——但别被字段数量吓到，按「块」读就能拆解。一个典型的 LoRA 配方：

```yaml
# —— 模型块：训什么 ——
base_model: Qwen/Qwen2-7B
model_type: AutoModelForCausalLM
sequence_len: 2048              # 上下文长度（第五篇）

# —— 数据块：用什么训 ——
datasets:
  - path: alpaca.json
    type: alpaca               # 数据格式（第二篇）
sample_packing: true           # packing（第二篇）

# —— PEFT 块：怎么省参数 ——
adapter: lora                  # 或 qlora
lora_model_dir: null
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_linear: true       # 目标模块（第四篇）

# —— 训练块：超参 ——
learning_rate: 2e-4
num_epochs: 3
micro_batch_size: 2
gradient_accumulation_steps: 8

# —— 日志/保存块 ——
wandb_mode: online
output_dir: ./output
```

**读 YAML 的框架**：每个块都对应前文的一个主题——`sequence_len` 是长序列、`type: alpaca` 是数据格式、`lora_r/alpha/target` 是 LoRA、`sample_packing` 是 packing。**能「按块读」一份 YAML，就说明你对微调的知识是「结构化」的，而非「记配置项」**。

## 2 数据与 RL 配置：Axolotl 的「格式库」

Axolotl 的数据配置比 LLaMA-Factory 更「显式」——`datasets` 数组里，每条数据都声明 `path` 与 `type`：

```yaml
datasets:
  - path: sft_data.json
    type: alpaca                # 单轮三字段
  - path: chat_data.json
    type: sharegpt              # 多轮对话
    conversation: llama3        # 对话模板（第二篇）
  - path: messages.jsonl
    type: chatml                # messages 格式
```

**`type` 决定了解析器**：`alpaca` 按「instruction/input/output」解析，`sharegpt` 按「conversations」解析，`chatml` 按「messages」解析——**格式声明对了，字段映射就对了**。Axolotl 还支持 `rl` 块配置偏好优化：

```yaml
rl: dpo                        # 或 orpo / kto 等（第六篇）
dpo_beta: 0.1                  # DPO 的温度系数（第六篇）
```

一个 YAML 就能把「SFT → DPO」的整条偏好优化流水线配置出来——**这是 Axolotl 比 LLaMA-Factory 更「对齐方法覆盖广」的地方**。

## 3 多卡与并行配置：Axolotl 的「主场」

Axolotl 对多卡的支持是它的招牌。多卡训练走 **accelerate** 或 **DeepSpeed**，YAML 里对应：

```yaml
# —— 并行方式 ——
fsdp:                           # 或 deepspeed 配置
  fsdp_transformer_layer_cls_to_wrap: [Qwen2DecoderLayer]
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_use_orig_params: true    # FSDP + LoRA 关键（第三篇）

# 或用 accelerate + DeepSpeed
deepspeed: configs/zero3.yaml   # ZeRO-3 配置（第三篇）
```

**要点**：

- **`fsdp_use_orig_params: true` 是 LoRA + FSDP 的关键**——FSDP 默认展平参数，LoRA 需要「原始参数」才能精确控制「哪些训、哪些冻」（第三篇《FSDP》）；
- **`fsdp_transformer_layer_cls_to_wrap`** 告诉 FSDP「把哪些层包成 FSDP 单元」——要填模型的 decoder 层类名；
- **DeepSpeed 配置独立成文件**（ZeRO stage、offload）——对应第三篇《ZeRO》与《卸载》。

**启动命令**（accelerate 多卡）：

```bash
accelerate launch -m axolotl.cli.train config.yaml
```

Axolotl 会读取 YAML 里的并行配置，交给 accelerate/DeepSpeed 执行——**你只管写 YAML，并行细节框架接管**。<span class="marginnote">多卡踩坑的高频点：<strong>「单卡能跑的配置，多卡不一定」</strong>——FSDP 的 wrap 层类名写错、DeepSpeed 的 ZeRO stage 与 LoRA 不兼容、梯度累积与并行交互异常……Axolotl 社区把很多常见并行配置做成了「可直接套用的模板」，先抄模板、再改参数，比从零写稳得多。</span>

## 4 实战流程：从 YAML 到合并

一次 Axolotl 微调的完整流程：

**① 写 YAML**：按「模型/数据/PEFT/训练/并行」五块写好配置。

**② 启动训练**：`accelerate launch -m axolotl.cli.train config.yaml`——日志与 wandb 指标实时可见。

**③ 测试**：训练完，用 `axolotl.cli.inference` 加载「基座 + adapter」做推理测试（类似 LLaMA-Factory 的 Chat 页）。

**④ 合并/导出**：LoRA 适配器合并进基座：

```bash
python -m axolotl.cli.merge_sharded_safetensors --base_model Qwen/Qwen2-7B \
  --lora_model_dir ./output/checkpoint-xxx --output_dir ./merged
```

**⑤ 评估**：用评测脚本（或后续框架）验证效果。

**与 LLaMA-Factory 的流程对比**：几乎一致（准备 → 配置 → 训练 → 测试 → 导出），差别在「配置的形式」——**LLaMA-Factory 有 WebUI 兜底，Axolotl 全靠 YAML 与命令行**。选择的标准是：**你更愿意「点按钮」还是「写配置」**。

## 5 实战要点与常见问题

Axolotl 实战的高频经验：

**① 模板要配 `conversation`**：多轮数据（sharegpt 类型）要声明 `conversation: <模板名>`——不声明，模型训练后「不认识对话格式」（第二篇）。

**② `sequence_len` 决定显存**：2048 与 4096 的显存差距很大——**显存不够先降 `sequence_len`，而不是先降 batch**（第三篇的激活公式）。

**③ `sample_packing` 是双刃剑**：开它省显存提吞吐，但**packing 的注意力隔离要框架处理对**（第二篇《packing》）——Axolotl 已内置正确实现，但「开/关」要按数据特性决定。

**④ 断点续训**：`--resume_from_checkpoint` 或 YAML 里的 `resume_from_checkpoint`——对应断点续训的通用机制。

**⑤ 复现配方先查版本**：社区配方常基于特定 Axolotl 版本——**YAML 字段随版本变**，复现前先对齐版本（上一节的「版本漂移」提醒）。<span class="marginnote">Axolotl 的一个隐藏优势是「<strong>钩子（hooks/plugins）</strong>」：它允许你在训练流程的特定阶段注入 Python 自定义逻辑（如自定义评估、自定义数据变换）。这让「YAML 声明式」与「代码可定制」兼得——<strong>「配置管常规、钩子管特殊」是 Axolotl 进阶用户的常态</strong>。</span>

## 6 小结

- **Axolotl = 配置即配方**：一份 YAML 讲清「训什么、怎么训、用几张卡」，多卡是它的主场。
- **YAML 按块读**：模型块 / 数据块（`type` 定格式）/ PEFT 块（`lora_r` 等）/ 训练块 / 并行块——每个块都是前文概念。
- **数据格式显式**：`datasets[].type`（alpaca/sharegpt/chatml）决定解析器；`rl` 块配置 DPO/ORPO 等。
- **多卡**：FSDP（`use_orig_params` 是 LoRA 关键）与 DeepSpeed（ZeRO 配置独立文件）；`accelerate launch` 启动。
- **全流程**：写 YAML → 启动 → inference 测试 → merge 合并 → 评估——与 LLaMA-Factory 同构，形式是配置。
- **实战要点**：模板要配、`sequence_len` 管显存、packing 要框架处理对、复现先对齐版本。

在下一节，我们看国产模型适配的选手：**ms-swift 实战——国产模型适配与轻量化部署衔接**。
