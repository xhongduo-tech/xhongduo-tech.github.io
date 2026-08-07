---
title: ZeRO-Offload：优化器状态与梯度卸载到 CPU/NVMe
date: 2026-08-07
---

# ZeRO-Offload：优化器状态与梯度卸载到 CPU/NVMe

<div class="epigraph">
<p>当显存告急时，整台机器都是你的显存。</p>
<footer>—— 任浩（Jie Ren），ZeRO-Offload 论文第一作者</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ ZeRO-Offload 论文（Ren et al., 2021）· 显存优化篇 ｜ 2026-08-07</p>
</div>

## 为什么从 ZeRO-Offload 开始

并行切分（ZeRO/FSDP）把模型状态摊到多张 GPU 上，但摊分有下限——**当你只有 1–2 张卡时，摊无可摊**。此时最后一招是：把 GPU 放不下的状态**挪出 GPU**，放到 CPU 内存，甚至放到 NVMe 固态盘。这就是 **ZeRO-Offload**。

它利用了当代机器一个被忽视的「空闲资源」：**CPU 内存普遍比显存大 4–8 倍**（服务器常配 512GB–1TB CPU 内存，GPU 显存单卡只有 80–112GB）。ZeRO-Offload 把「GPU 放不下的那部分」卸载到 CPU，让单机也能训练单卡放不下的模型。理解它，就理解了「显存的最后一公里怎么走」。

## 1 ZeRO-Offload 的核心思想：卸载 14Ψ，留下 2Ψ

回顾显存账本：模型状态 = 参数 $2\Psi$ + 梯度 $2\Psi$ + 优化器状态 $12\Psi$。ZeRO-Offload 的分配是：

- **留在 GPU**：参数 $2\Psi$（前向/后向需要它）。
- **卸载到 CPU**：梯度 $2\Psi$ 与优化器状态 $12\Psi$，共 $14\Psi$。

为什么这么分？因为**梯度与优化器状态只在「更新」时用到，而更新可以发生在 CPU 上**。每步流程变成：

1. GPU 算前向/后向，得到梯度。
2. 梯度拷贝到 CPU。
3. CPU 上执行 Adam 更新（用 CPU 内存里的 FP32 主权重与动量）。
4. 更新后的参数拷回 GPU。

**CPU 分担了「优化器」这个角色**——Adam 更新本就是数值运算，CPU 完全能跑，只是比 GPU 慢。这是「用 CPU 算力换 GPU 显存」的买卖。<span class="marginnote">ZeRO-Offload 论文里有个「分工」原则：GPU 做它擅长的（前向/后向，算力密集），CPU 做它擅长的（内存大、更新逻辑简单）。把「计算密集」与「存储密集」分离到不同硬件，是异构计算的经典套路。</span>

## 2 卸载的通信成本：PCIe 是唯一通道

GPU 与 CPU 之间只有 **PCIe** 这一条路（或 NVLink-C2C，但普通服务器是 PCIe）。PCIe 4.0 x16 的带宽约 32 GB/s，而显存带宽是 1–3 TB/s——**差了两个数量级**。于是卸载的每一步传输都是瓶颈：

每步的传输量 = 梯度拷出（$2\Psi$）+ 参数拷回（$2\Psi$）= $4\Psi$ 字节。<span class="marginnote">以 7B 模型为例，每步要过 PCIe 传 $4 \times 7\text{B} = 28\text{GB}$，按 32GB/s 算约 1 秒——相比 GPU 上一步几毫秒的前向后向，这 1 秒直接主导了步长。这就是「offload 会显著掉速」的量化来源。</span>

## 3 公式解析：卸载的吞吐损失

设参数量 $\Psi$，PCIe 有效带宽 $B_{\text{pcie}}$，GPU 上前向后向耗时 $T_{\text{gpu}}$。卸载模式下每步耗时：

$$T_{\text{step}} = \max\left(T_{\text{gpu}},\ \frac{4\Psi}{B_{\text{pcie}}}\right) + T_{\text{cpu-update}}$$

- **$\frac{4\Psi}{B_{\text{pcie}}}$（传输时间）**：每步搬 $4\Psi$ 字节（梯度出 + 参数回）。模型越大，传输越久。
- **$T_{\text{gpu}}$（GPU 计算）**：可与传输**重叠**（先拷梯度、同时算下步前向），但总步长至少是两者的较大者。
- **$T_{\text{cpu-update}}$（CPU 更新）**：Adam 更新在 CPU 上执行，通常与传输部分重叠，但会成为新的短板。

**关键结论**：当 $\frac{4\Psi}{B_{\text{pcie}}} \gg T_{\text{gpu}}$ 时（大模型 + 慢 PCIe），吞吐被**传输**主导，offload 的模型越大、掉速越狠。7B 模型 offload 后每步约 1 秒级，相比纯 GPU 训练慢一个数量级——**offload 是「能跑」而非「跑快」的方案**。<span class="marginnote">ZeRO-Offload 论文给出的经验是：卸载优化器状态+梯度（而不卸载参数）是「性价比最高」的一档——参数留在 GPU 让前向后向保持高速，只把不常用的 14Ψ 移出去。全参数卸载（连参数都挪走）虽然显存更省，但每步都要 AllGather 参数，通常更慢。</span>

## 4 NVMe 卸载：把「内存」也变大

当 CPU 内存也不够时，还有最后一级：**NVMe 卸载**。把优化器状态甚至参数存到固态盘上。NVMe 的顺序读写在 3–7 GB/s，比 PCIe 还慢一个数量级，但容量可以到 TB 级——**用「更慢」换「更大」**。

DeepSpeed 的 `offload_param.device: "nvme"` 支持把参数分片存到 NVMe。适用场景：

- 单机想训几十 B 的模型。
- 训练速度完全不重要（如调试、低优先级任务）。
- 只想「跑起来验证」而 GPU/内存都紧张。<span class="marginnote">NVMe 卸载是显存工程的「退无可退」：CPU 内存用完再上 NVMe。它的每一步传输都慢到毫秒秒级，所以几乎只用于「能跑就行」的场景——真正的训练不会用它，但「验证代码正确性」时它很香。</span>

## 5 ZeRO-Offload 与并行切分的配合

Offload 不是孤立技术，它常与 ZeRO/FSDP 组合：

- **ZeRO-2 + Offload**：梯度与优化器状态先沿 DP 维摊薄（$14\Psi/N_d$），再把每卡那份卸载到 CPU。
- **ZeRO-3 + Offload**：连参数也分片，GPU 只留「当前层」的临时参数——显存极限逼近「仅激活」。
- **FSDP + CPU offload**：PyTorch FSDP 也支持 `cpu_offload=True`，语义与 ZeRO-3 Offload 等价。

组合后的显存收益是**乘法**的：分片把 16Ψ 摊到 $N$ 卡，卸载再把每卡的份额挪出 GPU。**分片 × 卸载 = 显存问题的两个自由度的同时利用**。<span class="marginnote">DeepSpeed 的配置里 `stage=3 + offload_optimizer + offload_param` 一起开，就是这套组合的完整形态——它能用一张 A100 训练 70B 模型（虽然慢到每分钟一步）。知道有这条路，至少不会被「显存不够」卡死。</span>

## 6 辨析｜易错点：Offload 的常见误区

**辨析｜易错点：**
- **「offload 是免费的」是错觉**：它每步搬 $4\Psi$ 字节过 PCIe，掉速一个数量级是常态，只用于显存实在不够。
- **「卸载得越多越好」是错觉**：全量卸载（参数也搬）比「只卸优化器+梯度」更慢，因为每步都要 AllGather 参数。最优是「参数留 GPU、其余卸走」。
- **offload 与重计算不冲突**：两者砍不同的项（offload 砍模型状态、重计算砍激活），可同时开。
- **CPU 更新不是免费的**：CPU 算 Adam 也占 CPU 时间，但通常被传输掩盖；若 CPU 太弱会成新瓶颈。
- **NVMe 是最后手段**：它比 CPU 内存还慢一个数量级，正常训练绝不主动用它。

## 7 小结

- **核心思想**：把梯度与优化器状态（$14\Psi$）卸载到 CPU，参数（$2\Psi$）留 GPU，CPU 执行 Adam 更新。
- **每步传输**：$4\Psi$ 字节过 PCIe，是吞吐瓶颈；掉速一个数量级是常态。
- **NVMe 卸载**：CPU 内存不够时的最后一级，用更慢换更大。
- **与切分组合**：分片（× $N$）+ 卸载（× CPU），两个自由度同时利用。
- **定位**：显存不够的终极手段，能跑而非跑快——「最后一公里」。

## 8 进阶与延伸

**动手试一次 CPU offload**：在 DeepSpeed 里开 `offload_optimizer: {device: "cpu"}` 训一个小模型，对比开关前后的每步耗时——你会直观看到「吞吐掉一个数量级」的代价，以及「显存省一大块」的收益。这就是「能跑 vs 跑快」的活教材。

**几个值得进一步挖的方向**：

- **Offload 与重计算的取舍顺序**：显存不够时，先开重算还是先开 offload？重算只多花算力、offload 掉吞吐——「先重算、后 offload」的顺序为什么是合理的？
- **PCIe 带宽的现实测量**：`cudaMemcpy` 的 D2H/H2D 实测带宽往往只有理论峰值的一半——用 `cudaMemcpy` 基准测一下你的机器，重算 offload 的「真实吞吐损失」。
- **NVMe 卸载的适用边界**：DeepSpeed 的 `offload_param: {device: "nvme"}` 什么时候值得用？「验证代码正确性」之外的场景，它还有价值吗？

**自测题**：为什么 ZeRO-Offload 卸载「优化器状态 + 梯度」而不是「参数」？如果你能说清「参数留 GPU 让前向后向保持高速」，就理解了 Offload 的「性价比最高」之选。

## 9 动手实践清单

- 在 DeepSpeed 开 `offload_optimizer`，对比开关前后的每步耗时与显存。
- 观察 `free -h` 里 CPU 内存的变化，验证「14Ψ 挪到 CPU」。
- 用 `nvidia-smi` 与 `free` 画一张「GPU/CPU 内存迁移」的时间线。
- 实测 `cudaMemcpy` 的 D2H/H2D 带宽，重算 offload 的「真实吞吐损失」。
- 对比「只卸优化器」与「连参数也卸」两档的显存与速度。
- 试 NVMe 卸载，验证「能跑但慢到只适合验证」。
- 计算「4Ψ 每步过 PCIe」在 7B 模型上的传输时间。
- 对比「offload + 重算」组合与「纯 offload」的显存与吞吐。
- 验证「offload 是能跑而非跑快」的定位。
- 用「4Ψ 过 PCIe」算 70B 模型的每步传输时间。
- 对比「只卸优化器」与「全量卸载」的显存与速度。

在下一节，我们换一个视角：当显存「看起来够」却还是 OOM 时，问题往往出在 **显存碎片与 PyTorch 的 caching allocator**。
