---
title: DeepSpeed 架构：ZeRO 实现、配置文件与训练流程
date: 2026-08-07
---

# DeepSpeed 架构：ZeRO 实现、配置文件与训练流程

<div class="epigraph">
<p>优化的目标不是更快的单算子，而是更高的系统吞吐。</p>
<footer>—— 沙姆苏丁 · 拉杰班达里（Shamsuddin Rajbhandari，DeepSpeed 团队）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ DeepSpeed 论文与官方文档 · 训练框架篇 ｜ 2026-08-07</p>
</div>

## 为什么从 DeepSpeed 开始

Megatron 来自 NVIDIA，DeepSpeed 来自微软——两者是 2020 年代大模型训练框架的双雄。如果说 Megatron 的金字招牌是**张量并行与 3D 混合并行**，那么 DeepSpeed 的金字招牌就是 **ZeRO 与配套的显存优化全家桶**：ZeRO 分级、Offload、稀疏注意力、通信压缩、自动调参。

DeepSpeed 还有一个人人称赞的设计：**用一份 JSON 配置文件描述几乎所有训练选项**，用户甚至不用改训练代码——这是它当年快速铺开的关键。理解 DeepSpeed 的架构，是理解「ZeRO 家族 + 训练流程 + 配置驱动」三者如何组织成一个完整框架的样板。

## 1 DeepSpeed 的定位与组成

DeepSpeed 是一个「训练优化库」而非「模型库」：它不提供模型结构，而是在 PyTorch 训练循环之上提供一组**注入式优化**。主要组成：

- **ZeRO**：内存优化的核心，分 ZeRO-1/2/3 与 Offload（CPU/NVMe）。
- **优化器（Adam 变体、Lion、混合精度）**：与 ZeRO 深度绑定，分片参数更新。
- **通信优化**：梯度压缩（top-k 稀疏化）、梯度累积、通信计算重叠。
- **调度**：pipeline 并行、MoE（DeepSpeed-MoE）、稀疏训练。
- **工具链**：`autotuning`（自动找最优配置）、`monitor`（性能监控）、`flops profiler`。<span class="marginnote">DeepSpeed 的哲学是「不侵入」：它主张用户只需把 `optimizer` 换成 `DeepSpeedEngine`，剩下的交给配置与框架。这种「引擎式」设计与 Megatron 的「整个训练脚本都用我的」形成鲜明对比——前者融入你的代码，后者让你进入它的体系。</span>

## 2 ZeRO 在 DeepSpeed 中的实现

ZeRO 不是另一个并行策略，而是对**数据并行**的内存改造。DeepSpeed 的实现要点：

- **分片单位**：把参数、梯度、优化器状态沿 DP 维切成 $N$ 片，每卡 $1/N$。
- **通信原语**：前向/后向时 AllGather 参数，后向末尾 ReduceScatter 梯度（与 FSDP 语义一致）。
- **Stage 划分**：`zero_optimization.stage` 字段选 1/2/3，stage=3 时连参数也切。
- **Offload 配合**：`offload_optimizer`、`offload_param` 可以把优化器状态、参数卸载到 CPU 或 NVMe，见第五篇。

一个易被忽略的实现细节：DeepSpeed 的 ZeRO 是**在优化器层做文章**的——它把 `optimizer.step()` 改造成「分片状态更新」，因此**你必须用 DeepSpeed 提供的优化器（或注册自定义优化器）**才能享受 ZeRO 收益。<span class="marginnote">这解释了为什么 DeepSpeed 的配置里总有一整段 `"optimizer"`：ZeRO 的更新逻辑与优化器状态布局深度耦合。反过来，这也意味着「换优化器」在 DeepSpeed 里要小心，不是所有第三方优化器都兼容分片更新。</span>

## 3 配置文件：DS_Config 的力量

DeepSpeed 用一个 JSON 文件（常叫 `ds_config.json`）统管配置，结构大致：

```json
{
  "train_batch_size": 1024,
  "gradient_accumulation_steps": 16,
  "fp16": { "enabled": true, "loss_scale": 0, "initial_scale_power": 16 },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param": { "device": "cpu" }
  },
  "optimizer": { "type": "AdamW", "params": { "lr": 3e-4 } },
  "scheduler": { "type": "WarmupLR", "params": { "warmup_min_lr": 0 } }
}
```

拆解几个关键字段：

- **`train_batch_size`**：DeepSpeed 会按 DP 规模与梯度累积自动算出每个 step 的实际 batch。
- **`fp16.loss_scale`**：`0` 表示动态 loss scaling；`initial_scale_power` 是初始缩放指数。
- **`zero_optimization`**：ZeRO 的全部开关，stage、offload、通信优化都在这里。
- **`optimizer`/`scheduler`**：DeepSpeed 内置的优化器与学习率调度，用 JSON 描述而非代码。<span class="marginnote">这份配置的价值在于「实验可复制」：整个训练设置被固化成一个文件，换机器、换卡数、试不同 ZeRO stage 都只改文件不改代码。这也是 DeepSpeed 在学术圈流行的原因之一——写实验配置比写代码门槛低。</span>

## 4 训练流程：DeepSpeedEngine 注入循环

DeepSpeed 的使用流程极简，三步：

```python
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, config="ds_config.json"
)
for step in range(total_steps):
    loss = model_engine(batch)          # 前向
    model_engine.backward(loss)         # 后向 + 梯度分片 + ReduceScatter
    model_engine.step()                 # 分片参数更新 + 梯度清零
```

`deepspeed.initialize` 返回一个 `DeepSpeedEngine`，它把前向/后向/更新全部接管。**用户自己的训练循环几乎不用改**——这正是「引擎式注入」的体现。

DeepSpeedEngine 内部做了这些事：把模型包装进 ZeRO 分片、替换优化器、注册学习率调度、启动通信组、把 fp16 转换与 loss scaling 挂进 backward。<span class="marginnote">这个「initialize 之后一切自动」的体验，与 FSDP 的 `wrap` 风格、与 Megatron 的「整脚本都用框架」风格都不同。三种框架各有各的抽象粒度，选型时看的是你想让框架接管多少。</span>

## 5 公式解析：ZeRO stage 与 offload 的显存梯度

DeepSpeed 的显存收益可以用一个递进公式表达。设参数量 $\Psi$、DP 规模 $N_d$、CPU offload 开启与否，每卡模型状态：

$$\text{Mem}_{\text{stage3}} = \frac{16\Psi}{N_d} \xrightarrow{\text{+offload optimizer}} \frac{2\Psi}{N_d} + \frac{14\Psi}{N_d}\Big|_{\text{CPU}} \xrightarrow{\text{+offload param}} \frac{2\Psi}{N_d}\Big|_{\text{GPU}} + \frac{14\Psi}{N_d}\Big|_{\text{CPU/NVMe}}$$

- **$\frac{16\Psi}{N_d}$**：ZeRO-3 把 16Ψ（参数+梯度+Adam 状态）全摊到 DP 维。
- **offload optimizer**：把 $14\Psi$（梯度 $2$ + 优化器状态 $12$）的驻留从 GPU 移到 CPU，GPU 只留参数 $2\Psi/N_d$。
- **offload param**：连参数也挪到 CPU/NVMe，GPU 显存进一步逼近零（剩激活与临时 buffer）。<span class="marginnote">代价当然存在：CPU↔GPU 传输带宽远低于显存带宽，offload 之后「每次 AllGather 参数」都要过 PCIe，吞吐明显下降。offload 是「显存实在不够」的最后手段，不是默认选项。</span>

这条链路就是 DeepSpeed「从 ZeRO-1 到 offload」的完整显存工程图谱。直觉上记住一句话：**ZeRO 把状态摊到卡上，offload 把状态摊到卡外**。

## 6 辨析｜易错点：DeepSpeed 的常见误区

**辨析｜易错点：**
- **DeepSpeed ≠ ZeRO**：ZeRO 只是 DeepSpeed 的内存优化模块，DeepSpeed 还含调度、MoE、通信优化等。别把两者画等号。
- **DeepSpeed 不是并行策略框架（传统意义）**：它的 pipeline 并行支持相对弱，主要战场是 ZeRO/offload/单机多卡。
- **`train_batch_size` 是全局语义**：DeepSpeed 会根据 DP 与梯度累积自动分摊，别在脚本里再手动乘 DP 规模。
- **`fp16.enabled: true` 不代表自动 loss scaling 最优**：动态 scaling 需要配 `initial_scale_power`，开错会精度爆炸或收敛慢。
- **offload 会显著掉速**：它是显存不足的兜底，不是加速手段。优先考虑重计算与减小 batch。

## 7 小结

- **DeepSpeed 的定位**：训练优化库，注入式提升 PyTorch 训练吞吐与显存容量。
- **ZeRO 实现**：分片参数/梯度/优化器状态，AllGather 参数 + ReduceScatter 梯度，stage 1/2/3 递进。
- **配置驱动**：一份 `ds_config.json` 管住 ZeRO、fp16、优化器、调度与 offload。
- **训练流程**：`deepspeed.initialize` → `backward` → `step`，用户循环几乎不改。
- **显存梯度**：ZeRO 摊到卡上（$16\Psi/N_d$），offload 摊到卡外（GPU 逼近零）。
- **框架对比**：DeepSpeed（ZeRO/offload 强）、Megatron（TP/PP 强）、FSDP（PyTorch 原生）。

## 8 进阶与延伸

**动手配一份 `ds_config.json`**：给你的模型配一个 ZeRO-3 + offload_optimizer 的配置，跑起来后对比 `nvidia-smi` 与 `free -h`——你会直观看到「模型状态从 GPU 挪到 CPU」的效果，以及每步变慢的幅度。

**几个值得进一步挖的方向**：

- **DeepSpeed 与 Megatron 的互补**：DeepSpeed 的 ZeRO 可以「嵌入」Megatron 的 TP/PP 并行里（DP 维用 ZeRO）——两者不是互斥而是互补，组合起来才是完整的 3D + 分片方案。
- **`deepspeed.initialize` 黑盒里有什么**：它接管了模型包装、优化器替换、fp16 转换、通信组初始化——对照源码理解「引擎式注入」的每一步。
- **autotuning 工具**：DeepSpeed 的 `autotuning` 自动搜索最优 ZeRO 配置——它搜的是什么空间？为什么能自动找到「比手调好」的配置？

**自测题**：DeepSpeed 要求「用它的优化器」才能享受 ZeRO——为什么？如果你能说清「分片更新要动优化器内部布局」，就理解了 ZeRO 与优化器深度绑定的原因。

在下一节，我们从框架回到 PyTorch 原生，解剖 **DDP 的内部机制**——bucket 分桶如何让梯度 AllReduce 又快又稳。
