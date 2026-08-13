---
title: LLaMA-Factory 集成与 CPU/GPU 异构 LoRA 微调
date: 2026-08-07
---

# LLaMA-Factory 集成与 CPU/GPU 异构 LoRA 微调

<div class="epigraph">
<p>训练与推理的边界，在异构计算里第一次变得模糊。</p>
<footer>—— 面向微调工程的观察</footer>
</div>

<div class="article-byline">
<p>第四级 · ktransformers（消费级 MoE 推理引擎） ｜ 官方文档 SFT User-Guide ｜ 2026-08-07</p>
</div>

## 为什么「消费级微调」是新的战场

推理能跑只是第一步；**微调（fine-tuning）**才是让模型真正适配自己数据的环节。传统观念里，微调 671B 模型需要 8×A100。ktransformers 的 `kt-sft` 模块把 **LLaMA-Factory** 作为上层编排、ktransformers 作为异构后端，让 **LoRA 微调**在 2–4 张 RTX 4090 上成为可能——比 ZeRO-Offload 快 6–12 倍。<span class="marginnote">`kt-sft` 是 ktransformers 的微调子模块：<strong>LLaMA-Factory 负责数据、训练循环、LoRA 注入；ktransformers 接管 Attention 与 MoE 的底层计算</strong>。这就是「上层框架 + 异构内核」的又一次分工。</span>

## 1 架构：把异构放置搬进训练

训练与推理共享同一套「算子放置」哲学，但方向相反——训练要**反向传播**，所以每个算子不仅要能前向，还要能反向。ktransformers 的微调架构用三个定制类实现：

**KTrainer**：继承 `transformers.Trainer`。它阻止默认的「把整个模型 `.to(device)` 拷到单张 GPU」，改为**按注入规则把各层显式分配到 `cuda:0`、`cuda:1` 或 `cpu`**——让模型「躺」在异构设备上训练。

「阻止 `.to(device)`」这个动作是 ktransformers 训练架构的基石：Transformers 默认假设「模型在一台设备上」，训练器会调用 `.to(device)` 把整个模型拷过去——对 671B 模型，这一步直接 OOM。KTrainer **覆写**这个行为：不让模型整体移动，而是按放置规则逐层分配。**「覆写框架的默认假设」是适配超大模型的关键手法**——框架的默认行为建立在「模型装得下」的假设上，而超大模型恰好打破这个假设，必须显式接管。

**KTransformersLinearLora**：双重继承 `KTransformersLinear`（推理内核）与 `LoraLayer`（LoRA 适配器）。既保留 ktransformers 的高性能 prefill/generate 内核，又可训练 `lora_A` / `lora_B` 参数——**LoRA 参数在 GPU 上训，专家权重留在 CPU 只读**。<span class="marginnote">LoRA 的精髓：<strong>冻结原有权重、只训低秩增量</strong>。于是「671B 权重的微调」变成「几百 MB LoRA 参数的训练」——GPU 只需装下 LoRA 增量与激活，671B 冻结权重可继续躺在 CPU。这就是消费级微调能成立的第一性原因。</span>

**KSFTExpertsCPU**：把 MoE 层封装成**可微黑盒**——自定义 Autograd Function，前向通过 pybind11 调用 C++ 内核（AMX-BF16/INT8 或 llamafile），反向时**预计算并缓存权重转置 $W^\top$ 与中间激活**以加速梯度计算。

「缓存 $W^\top$」是反向加速的关键：反向传播要算「对权重的梯度」，本质要复用「前向时的输入激活」并做「转置权重的乘法」——**若每次反向都重新转置权重（$W$ 是 671B 级的大矩阵），代价巨大**。提前缓存 $W^\top$ 与中间激活，反向时直接读缓存，省掉重复计算。**「缓存可复用的中间结果」是深度学习的通用加速手段**——从训练框架的激活缓存到推理的 KV Cache，都是同一思路：把「重复要用的东西」提前算好、存好。

## 2 双层 YAML：主配置 + 优化规则

微调用**两层 YAML**：

**主训练配置**（LLaMA-Factory 兼容）：模型路径（须为 **BF16 格式**）、数据集、超参数、LoRA 设置，外加 KT 开关：

```yaml
use_kt: true                      # 启用 ktransformers 后端
kt_optimize_rule: kt_rule.yaml    # 算子放置规则
cpu_infer: 32                     # CPU 线程数
finetuning_type: lora             # LoRA 微调
```

**优化规则配置**（`kt_optimize_rule`）：声明算子放置与后端，例如 Attention → `KDeepseekV2Attention`（cuda）、专家 → `KTransformersExperts`（cpu，AMX-BF16）、线性层 → `KTransformersLinear`（cuda）。

**关键点**：训练用的规则与推理**可以复用同一份**——因为「哪个算子放哪个设备」在前后向里一致。

「双层 YAML」的设计让「训练配置」与「放置配置」**解耦**：训练超参（lr、batch、epoch）归主配置，设备放置归规则配置——**改超参不动放置，改放置不动超参**。这种「关注点分离」让两种配置可以独立演进、独立复用：换数据集只需改主配置，换硬件只需改规则配置。**「把可变的东西分开」是配置设计的黄金法则**——耦合的配置一旦变多，牵一发动全身，改谁都怕坏。<span class="marginnote">「训练完直接能推理」的工程红利：<strong>同一个 `kt_optimize_rule` 加载 LoRA adapter 即可做推理（chat / OpenAI 兼容 API）</strong>——训练与推理共用一份放置蓝图，少配一套环境。</span>

## 3 启动命令与数据流

一条命令即可启动：

```bash
USE_KT=1 llamafactory-cli train examples/train_lora/kimik2_lora_sft_kt.yaml
```

训练的数据流：LLaMA-Factory 加载数据集 → 构建 batch → KTrainer 按放置规则把每层前向/反向派到对应设备 → CPU 侧专家经 `KSFTExpertsCPU` 计算（权重冻结、只算激活的梯度）→ GPU 侧更新 LoRA 参数。

「CPU 侧只算激活的梯度」是训练版异构的精髓：**冻结权重不需要梯度，所以 CPU 侧只需前向 + 对激活的反向，权重本身纹丝不动**。这让 CPU 侧的计算量与推理几乎一样（只多了「激活梯度」），而不像全量训练那样要存权重梯度与优化器状态。**「冻结 + LoRA」让「训练成本」从「整个模型」降到「一小撮参数」**——这正是消费级微调能成立的数学基础，也是 ktransformers 训练架构与推理架构「同构」的原因。

**辨析｜易错点：** 不是「把整个模型放进显存训练」。**冻结的 671B 权重大部分在 CPU 内存**，GPU 只装：LoRA 增量、激活、Attention 的参与梯度、以及热专家的前向。若误以为「要多卡才能装下模型」，就误解了 LoRA + 异构的本质——**你训练的是「增量」，不是「全量」**。

顺着「训练增量」再澄清一个边界：**LoRA 只更新注入层的低秩增量，但前向/反向仍要跑全模型**——671B 的前向计算一步都不能少（只是权重不更新、不存梯度）。所以「训练 671B」不等于「只算几百 MB」：**算的是全模型，存的是小增量**。这个区分解释了为什么「训练仍需要异构前向」——ktransformers 的推理内核在训练里继续派上用场，而不是被「训练专用」取代。<span class="marginnote">这也解释了为什么「2–4 张 4090」就够：<strong>显存装的是 LoRA + 激活 + KV，不是 671B 权重</strong>。卡多主要换「batch 更大、训练更快」，而不是「装得下」。</span>

## 4 公式解析：LoRA 的参数成本

LoRA 的显存红利可以量化。设被微调层的权重矩阵为 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$，LoRA 用低秩分解 $W + BA$，其中 $B \in \mathbb{R}^{d_{\text{out}} \times r}$、$A \in \mathbb{R}^{r \times d_{\text{in}}}$：

$$
\text{LoRA 参数} = r(d_{\text{out}} + d_{\text{in}}), \qquad \text{全量参数} = d_{\text{out}} \times d_{\text{in}}
$$

逐项拆解：

- **第一步**：全量微调要存 $d_{\text{out}} \times d_{\text{in}}$ 个梯度与优化器状态。
- **第二步**：LoRA 只存 $r(d_{\text{out}}+d_{\text{in}})$ 个可训参数——当 $r \ll \min(d_{\text{in}}, d_{\text{out}})$ 时，缩减到千分之一量级。
- **第三步**：对 671B 模型，LoRA（如 $r=64$）只带来几百 MB 可训参数——**这就是「消费级能训 671B」的数学根源**。

**一句话**：LoRA 把「训 671B 的梯度/优化器」降成「训几百 MB 增量」，配合 ktransformers 的异构前反向，消费级微调成为现实。

「低秩」为什么够用？因为**权重更新往往存在于一个低维子空间**——训练中真正「有意义的方向」远少于参数维度，用秩 $r$ 的低秩矩阵足以捕获大部分更新。这是 LoRA 的「科学依据」：**不是「碰巧够用」，而是「权重更新的内在维度就是低的」**。理解这一点，你就能回答「为什么 LoRA 效果不错」——它不是「偷工减料」，而是「认准了更新的有效维度」。这也解释了为什么 $r$ 的选择很重要：$r$ 太小捕获不充分，$r$ 太大又失去「省参数」的意义。<span class="marginnote">对比基线：官方数据 DeepSeek-V2-Lite 14B 上，HF 原生 303 token/s、Unsloth 455 token/s、KTransformers 530 token/s（同硬件同配置）——异构内核不只让推理快，<strong>训练吞吐同样受益</strong>，这正是「快 6–12 倍 vs ZeRO-Offload」的构成部分。</span>

## 5 微调后的使用闭环

训练不是终点——「训练完能立刻用」才是完整闭环。ktransformers 的微调后使用流程：

1. **保留配置**：训练用的 `kt_optimize_rule` 原样保留——它与推理共用同一份放置蓝图。
2. **加载 LoRA**：用推理配置加载模型 + LoRA adapter（`llamafactory-cli chat` / `api`）。
3. **验证效果**：拿训练前的样本对比训练后的输出，确认「学会了」。

| 环节 | 命令/配置 | 要点 |
| --- | --- | --- |
| 交互对话 | `llamafactory-cli chat <推理YAML>` | 设 `infer_backend: ktransformers` |
| API 服务 | `llamafactory-cli api <推理YAML>` | OpenAI 兼容接口 |
| 效果验证 | 前后样本对照 | 「可度量」才算真学会 |

「前后样本对照」是验证微调效果的最低成本方法：**同一组 prompt，分别喂给「微调前」与「微调后」的模型，逐条对比输出**。若微调确实生效，在目标任务上应有可见差异；若看不出差异，先怀疑「训练是否真的收敛」（loss 是否下降、LoRA 是否加载）。**「对照实验」是判断「训练是否有效」的黄金标准**——凭「感觉变强了」不可靠，两组输出摆在一起才有说服力。

**闭环的价值**：训练与推理共用一套放置配置，意味着「训练完 → 直接推理」**不需要重配环境、不需要改代码**——「train → infer」的无缝衔接，是「框架统一」给用户的最大红利。

这套「train → infer」闭环，与主流微调工具（如 Unsloth、peft）的「训完另起炉灶推理」形成了对比：**ktransformers 把两段放进同一个框架、同一套放置逻辑**——少一层转换、少一次环境切换，就少一批「换框架导致的坑」。**「一体化」的价值不在炫技，而在「减少衔接处的出错面」**——系统里出 bug 最多的地方，往往正是两个组件拼接的接口处。

## 6 小结

把异构 LoRA 微调浓缩成「一个继承、一个冻结、一个复用」：**KTrainer 继承并覆写** `.to(device)`、**671B 权重冻结在 CPU**、**放置规则训练/推理复用**——三者合起来，消费级微调才成为可能。

再补一个「心法」：微调是「最后一公里」——前面的部署（第 5 篇）解决了「能用」，微调解决「用得合适」。**「能用」与「用得合适」是消费级 AI 的两大步**：前者靠推理引擎，后者靠微调工具链。ktransformers 把两步都做了，这正是它「推理与微调双支柱」定位的完整含义。

- `kt-sft` = **LLaMA-Factory 编排 + ktransformers 异构内核**，让 671B 级模型在 2–4 张 4090 上可微调。
- 三个定制类：**KTrainer**（阻止全量 `.to(device)`，按规则放置）、**KTransformersLinearLora**（内核 + LoRA 双继承）、**KSFTExpertsCPU**（专家可微黑盒 + 反向缓存 $W^\top$）。
- 双层 YAML：主配置（`use_kt`、`kt_optimize_rule`）+ 算子放置规则，**训练与推理可共用**。
- 本质是训「增量」不是「全量」：**冻结权重在 CPU，LoRA 增量与激活在 GPU**。
- LoRA 参数 = $r(d_{\text{out}} + d_{\text{in}})$