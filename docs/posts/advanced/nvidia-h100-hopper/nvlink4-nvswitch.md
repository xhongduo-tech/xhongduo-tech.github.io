---
title: NVLink 4 与 NVSwitch 互联
date: 2026-08-07
---

# NVLink 4 与 NVSwitch 互联

<div class="epigraph">
<p>一颗 GPU 的孤独，是它把数据交给另一颗 GPU 时等待的那几微秒。</p>
<footer>—— 分布式计算的一种读法</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 硬件：NVIDIA H100/Hopper ｜ H100 白皮书 §2 ｜ 2026-08-07</p>
</div>

## 为什么从互连讲起

前两节我们讲完了「一颗 H100 内部」：制程、封装、内存。但大模型从来不是跑在一颗 GPU 上——一个 70B 模型要切开放在几十上百颗 GPU 上并行训练，GPU 与 GPU 之间必须高速通信。**GPU 之间的通道，就是互连（interconnect）。** NVIDIA 的答案是自研的 **NVLink** 与 **NVSwitch**：前者是一条「GPU 对 GPU 的专用高速公路」，后者是把 8 条高速路织成一张「全互连网络」的立交桥。这一对组合决定了「多 GPU 能不能像一台大 GPU 一样工作」——它是张量并行、数据并行的物理地基。<span class="marginnote">互连是「从极限到大模型」主线上的隐形主角：我们学分布式训练时，模型切分的通信量、并行度选择，最后都要落到「NVLink 有多快、网络有多宽」这些硬数字上。本节先把硬数字给足，后面第 4 篇《多节点集群》与《大规模 LLM 训练》会反复引用。</span>

## 1 NVLink：GPU 之间的专用通道

**NVLink（NVIDIA 高速互连）**：NVIDIA 自有的 GPU 间高速点对点互连技术，直接面向 GPU 显存地址空间，让一颗 GPU 能直接读写另一颗 GPU 的显存，而不需要经过 CPU 或 PCIe。

为什么不能用 PCIe？PCIe 5.0 的单卡带宽约 64 GB/s（双向），而 H100 的 FP16 算力需要约 1–2 TB/s 的数据吞吐才喂得饱——差了一个数量级。于是 NVIDIA 为 GPU 之间的「邻居通信」专门修了一条捷径：NVLink。

NVLink 与 PCIe 的根本区别有三点：

**带宽高**：NVLink 4 给每颗 H100 提供 900 GB/s 的双向带宽，是 PCIe 5.0 的十倍以上。
**延迟低**：NVLink 的往返延迟在微秒量级，比经过 PCIe 根联合体（root complex）的路径低一个量级。
**语义强**：NVLink 支持直接读写对方显存、原子操作、以及 NVLink 多播（multicast）——一条指令把数据同时发给多颗 GPU，这对 all-reduce 这类集合通信是巨大的加速。

一句话：**PCIe 是「快递」——要经中转站，慢但通用；NVLink 是「直达专线」——快，但只在同一台机器的 GPU 之间。**<span class="marginnote">NVLink 的历史：2016 年 Pascal 架构的 P100 首发第一代（160 GB/s），Ampere 的 A100 是第三代（600 GB/s），Hopper 的 H100 是第四代（900 GB/s）。每一代翻倍，NVIDIA 一直在用自研互连把「多 GPU 当单 GPU」的体验推向极限——这也是它相对 AMD、Intel 的护城河之一。</span>

## 2 NVLink 4 的规格：900 GB/s 从哪来

H100 的 900 GB/s 不是靠单条线提速，而是**靠数量堆出来**：

每颗 H100 有 **18 条 NVLink 4 连接**，每条双向带宽 **50 GB/s**，合计：

$$
B_{\text{NVLink}} = 18 \times 50\ \mathrm{GB/s} = 900\ \mathrm{GB/s}
$$

相比 A100 的 12 条 × 50 GB/s = 600 GB/s，**提升来自「连接的条数」（18 vs 12）而非每条的速度**。这是互连设计的一个经典规律：提高单线速率会急剧增加功耗与信号完整性难度，而「加几条线」代价更小。

这 18 条连接不是死板的并排，而是**成组布线**：每 4 条（或按端口布局）捆成一束，连接到一台 NVSwitch 上。H100 有 18 条 NVLink 连接，全部指向本机的 NVSwitch 网络。<span class="marginnote">对比 CPU 世界：Intel 的 UPI、AMD 的 Infinity Fabric 也做「多 die 互连」，但它们的规模是「几个计算 die 之间」，NVLink 面对的是「8–10 颗完整 GPU 之间」——带宽与规模都大得多。多 GPU 互连是 NVIDIA 独有的战场。</span>

## 3 NVSwitch：把点对点变成全互连

18 条 NVLink 把 GPU 连成了「蜘蛛网」，但谁来做「转发」？答案是 **NVSwitch（NVLink 交换机）**。

**NVSwitch**：一颗专门做 NVLink 数据转发的交换机芯片，作用类似于数据中心里的以太网交换机——它接收来自多颗 GPU 的数据，按目的地转发到正确的 GPU。

以 **DGX H100** 为例：8 颗 H100 各自把 18 条 NVLink 连接到 **4 颗第四代 NVSwitch**，每颗 NVSwitch 提供 64 个 NVLink 端口。交换机把 8 颗 GPU 织成一个**无阻塞全互连（non-blocking full-mesh）**：

- 任意两颗 GPU 之间都有**专用的 NVLink 带宽**，不会因为其他 GPU 同时通信而被挤占；
- 一颗 GPU 可以同时与所有其他 GPU 全速通信——这正是张量并行最需要的特性。

为什么叫「无阻塞」？想象以太网交换机遇到冲突要排队，而 NVSwitch 的交叉开关（crossbar）架构让「任意入端口 → 任意出端口」同时建立通路，互不干扰。

NVLink 4 还有一项面向集合通信的特性：**NVLink 多播（multicast）**。它允许一条写指令把数据同时发送给多颗 GPU 的显存地址。这直接加速了张量并行的关键路径——比如权重广播：一份权重一次性广播给 8 颗 GPU，不再需要 8 次点对点拷贝。配合 `reduce` 操作（在互连中归约），all-reduce 的一部分工作可以「在互连里顺路完成」，省去先搬到 SM 再归约的往返。**NVLink 不再只是「传输数据的管子」，它还参与「加工数据」**——这是它相对纯网络（InfiniBand）的又一优势。<span class="marginnote">这个「8 卡无阻塞」是有代价的：4 颗 NVSwitch + 大量 NVLink 走线把 DGX H100 变成一台 10.2 kW 的庞然大物。但它换来的是「节点内的 8 颗 GPU 像一颗 8 倍大的 GPU 那样协同」——这是张量并行能成立的前提。</span>

## 4 核心对比表：NVLink 四代演进

把 NVLink 的演进放在一起看，规律一目了然：

| 代际 | 架构 | 每 GPU 总带宽 | 连接数 | 每连接带宽 |
| --- | --- | --- | --- | --- |
| NVLink 1（P100） | Pascal | 160 GB/s | 4 | 40 GB/s |
| NVLink 2（V100） | Volta | 300 GB/s | 6 | 50 GB/s |
| NVLink 3（A100） | Ampere | 600 GB/s | 12 | 50 GB/s |
| NVLink 4（H100） | Hopper | 900 GB/s | 18 | 50 GB/s |

读这张表注意两个事实：**每连接带宽从二代起就没涨过（50 GB/s），增长全靠「加连接数」**；而连接数受限于 GPU 封装与功耗，这反过来解释了为什么单节点「8 卡」多年不变——每颗 GPU 能容纳的 NVLink 端口是有限的。

**辨析｜易错点：** 常有人把 NVLink 与 InfiniBand 混为一谈。它们的区别是本质性的：**NVLink 是「机器内部的芯片级互连」，InfiniBand 是「机器之间的网络级互连」**。NVLink 延迟微秒级、带宽 900 GB/s，InfiniBand 延迟十几微秒、每端口 400 Gb/s（约 50 GB/s）。前者贵而快，用于张量并行；后者便宜而广，用于跨节点数据并行——两者是「上下级」关系，不是「二选一」关系。

NVLink 家族还有一个容易混淆的成员：**NVLink-C2C（chip-to-chip）**。它用在 Grace-Hopper 超级芯片上，以 900 GB/s 的**相干（coherent）**链路把 NVIDIA Grace CPU 与 Hopper GPU 连成一体——CPU 与 GPU 共享一个一致的内存视图，彼此直接读对方的地址空间。它和「GPU 对 GPU」的 NVLink 4 名字相似、都是片间高速互连，但目标是「CPU+GPU 组成统一系统」而非「GPU 集群」——理解这个区别，后面看《统一内存与页迁移》和 Blackwell 的 GB200 时会更顺。

## 5 公式解析：一次 all-reduce 在 NVLink 上有多快

互连的价值要用「实际通信任务」来衡量。以数据并行每步的 **all-reduce**（全体求梯度均值）为例，通信量正比于模型大小。

设模型 $N$ 个参数、梯度用 FP32（4 字节），一次 all-reduce 的数据量：

$$
D_{\text{allreduce}} = N \times 4\ \mathrm{Byte}
$$

代入一个 70B 模型：$70 \times 10^9 \times 4 \approx 280$ GB。在 NVLink 4 的 900 GB/s 下，单对 GPU 的裸传时间：

$$
t = \frac{280\ \mathrm{GB}}{900\ \mathrm{GB/s}} \approx 0.31\ \mathrm{s}
$$

三步拆解：

- **第一步，定通信量**：all-reduce 的数据量 = 参数 × 精度字节数，与计算无关、只与模型有关。
- **第二步，看带宽**：900 GB/s 的 NVLink 把「传一遍全模型」压到 0.3 秒量级——这就是为什么张量并行能容忍「每层两次 all-reduce」。
- **第三步，比网络**：同样的 280 GB 走 InfiniBand（400 Gb/s ≈ 50 GB/s）要 5.6 秒——**差 18 倍**。这解释了为什么「通信密集的并行必须放 NVLink 域内」。

这条式子把抽象的「900 GB/s」翻译成了可感知的「传完整模型 0.3 秒」，也是后面《多节点集群》《大规模 LLM 训练》反复出现的算术。

再强调一个容易被忽略的点：**带宽是「双向」的**。NVLink 4 的 900 GB/s 是指「进出合计」——每颗 GPU 同时读与写，各占一半通道。所以设计通信模式时，要么让「读」与「写」自然对称（如 all-reduce 每节点一边发一边收），要么预留足够的单向余量。**把双向带宽误当成单向带宽，是性能预算里最常犯的错误之一**——它会让你的通信时间低估 2 倍。

最后回到系统视角：NVLink 4 与 NVSwitch 不是「可选的加速器」，而是 NVIDIA 把「多 GPU 当单 GPU」这条路线的基础设施。它的演进方向也预示了行业趋势——**互连带宽的增长速度正在超过单卡算力的增长速度**，未来 AI 芯片的竞争，很大一部分会在互连上分出高下。

## 6 小结

- **NVLink** 是 GPU 间专用高速互连，延迟微秒级、带宽远超 PCIe，支持直接读写对方显存与多播。
- **NVLink 4** 每颗 H100 有 18 条连接、合计 900 GB/s，增长来自「加连接数」而非「提速率」。
- **NVSwitch** 把 8 颗 GPU 织成无阻塞全互连，让「任意 GPU 全速访问任意 GPU」成为可能——张量并行的物理前提。
- NVLink 四代演进：**160 → 300 → 600 → 900 GB/s**，每连接 50 GB/s 十年未变。
- **NVLink（机内）与 InfiniBand（机间）是上下级关系**：前者张量并行，后者跨节点并行。
- **NVLink 多播**让一条写指令同时写多颗 GPU，配合互连内归约，直接加速张量并行的广播与 all-reduce。
- **NVLink-C2C** 是 CPU–GPU 相干互连（900 GB/s），与「GPU 对 GPU」的 NVLink 4 是两种用途；它是 Grace-Hopper 统一内存池的物理基础。
- all-reduce 传完整 70B 模型：NVLink 约 0.3 秒，InfiniBand 约 5.6 秒，相差 18 倍。

在下一节，我们把互连从「芯片对芯片」升到「系统对系统」——**DGX H100 与数据中心系统**，看 8 颗 H100 是怎么被装进一台 10.2 kW 的机器，再组成一个上千卡的集群。
