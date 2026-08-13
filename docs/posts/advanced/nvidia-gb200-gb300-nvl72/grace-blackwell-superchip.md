---
title: Grace Blackwell 超级芯片与 B200/GB300 GPU
date: 2026-08-07
---

# Grace Blackwell 超级芯片与 B200/GB300 GPU

<div class="epigraph">
<p>这不是一台计算机，而是一座数据中心。</p>
<footer>—— 黄仁勋（Jensen Huang），2024 年 GTC 大会介绍 Blackwell 平台时</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 硬件：NVIDIA GB200/GB300-NVL72 ｜ GB200 NVL72 白皮书 第1章 ｜ 2026-08-07</p>
</div>

## 为什么从超级芯片开始

上一站我们停在 NVIDIA H100/Hopper，看到一颗 GPU 怎样通过 Tensor Core 与 NVLink 变成一台「能塞进服务器的超算」<span class="marginnote">H100 时代的基本单元是「GPU 卡 + 外部 CPU 主机」，CPU 与 GPU 之间隔着 PCIe 或 NVLink，见《AI 硬件：NVIDIA H100/Hopper》。</span>。但 2024 年之后，NVIDIA 回答「如何训练万亿参数模型」的答案变了：**不再是单颗 GPU 有多强，而是把 CPU、GPU、内存、互连、供电、散热全部重新焊进同一块基板，先造出「超级芯片」，再把超级芯片组装成整个机架**。GB200 NVL72 就是这条路线上的旗舰：一座 72 颗 GPU、功耗约 120 kW 的液冷机架，被称为「机架级超级计算机」。

## 1 为什么要发明「超级芯片」

先回到一个朴素的问题：H100 不是已经在训练大模型了吗，为什么还要把 CPU 和 GPU 封在一起？

因为**大模型训练是一个「吃互连」的负载**。Transformer 的每一层都有两次全局通信——一次前向的 all-reduce、一次反向的 all-reduce，通信量与模型并行度直接挂钩。当模型从百亿参数涨到万亿参数，节点之间的数据搬运时间开始反超计算时间，系统便不再「算得快」，而是「等得久」。<span class="marginnote">更精确地说：每步训练的总时间 ≈ 计算时间 + 通信时间，二者必须同时被压下去，见《大模型预训练》与《AI 基础设施》中的并行策略。</span>

NVIDIA 的解法是双管齐下：

**横向**：把 GPU 之间的互连带宽从 PCIe 的几十 GB/s 一路推到 NVLink 的 TB/s 级——这一条线演化出本专题第 2 篇的 NVLink5 与 NVSwitch。
**纵向**：把「喂数据」的 CPU 直接焊到 GPU 旁边，用超高带宽的一致性互连 NVLink-C2C 替代 PCIe，让 CPU 不再成为瓶颈——这就是**超级芯片（Superchip）**。

「喂数据不够快」到底有多严重，把互连带宽排成一列就能看见差距：

| 互连 | 双向带宽 | 典型角色 |
| --- | --- | --- |
| PCIe Gen5 ×16 | 128 GB/s | 传统服务器 CPU-GPU |
| NVLink4（H100 时代） | 900 GB/s | Hopper GPU 之间 |
| NVLink-C2C（Grace-Blackwell） | 900 GB/s | 超级芯片内 CPU-GPU |
| NVLink5（Blackwell） | 1800 GB/s | Blackwell GPU 之间 |

PCIe 是通用总线，走的是「拷贝 → 发送 → 接收 → 拷贝」的流程；而 NVLink 系列是 NVIDIA 为 GPU 生态量身定制的私有互连，把带宽提高了整整一个数量级。**超级芯片的诞生，本质上是把「GPU 等数据」这件最贵的事，从软件层搬到了物理层解决。**

「超级芯片」这个词不是营销噱头，它对应一个具体的硬件事实：**原本分处两块电路板的 CPU 与 GPU，现在被封装到同一块基板上，用芯片间互连（chip-to-chip）直连**。GB200 超级芯片 = 1 颗 Grace CPU + 2 颗 B200 GPU。

## 2 GB200 超级芯片的三个成员

拆开一块 GB200 超级芯片，里面住着三个主角：

**Grace CPU**：72 核 Arm Neoverse V2 处理器，搭配 480 GB LPDDR5X 内存，内存带宽约 1 TB/s。它不承担矩阵运算，只负责数据搬运、调度与通信。<span class="marginnote">用 Arm 而非 x86 做数据中心 CPU，是 NVIDIA 在 GH200 上第一次的大胆试验；省电与高内存带宽让它更适合 AI 训练这类「内存墙」问题突出的负载。</span>

**两颗 B200 GPU**：每一颗都是一块独立的 Blackwell 加速器——2080 亿晶体管、192 GB HBM3e、8 TB/s 显存带宽，FP4 稠密算力约 10 PFLOPS。两颗 GPU 通过 NVLink-C2C 与 Grace CPU 相连，每颗 GPU 获得 900 GB/s 的 CPU 直连带宽。

**NVLink-C2C 互连**：这是超级芯片的灵魂。C2C（Chip-to-Chip）是一类**片间一致性互连**：两颗芯片共享同一份内存地址空间，CPU 可以直接读写 GPU 内存，GPU 也能直接访问 CPU 内存，无需像 PCIe 那样把数据先拷贝到宿主内存再搬运。<span class="marginnote">一致性（coherency）意味着「CPU 写、GPU 读」不再需要显式拷贝与同步——这把传统异构计算的「搬数据」改成了「共享数据」，是编程模型层面的一次简化。</span>

一个值得追问的细节是：**为什么是「一颗 CPU 配两颗 GPU」这个 2:1 的配比？** 答案藏在带宽里。Grace CPU 提供约 1 TB/s 的内存带宽，而两颗 B200 各自需要 900 GB/s 的 CPU 直连带宽，合计 1.8 TB/s——CPU 的内存带宽恰好是两颗 GPU「入口带宽」的一半，而 GPU 的算力增长又比 CPU 快得多。让一颗 CPU 同时喂养两颗 GPU，既把 CPU 侧的总线喂满，又不至于让昂贵的 CPU 核闲置。<span class="marginnote">这个配比不是拍脑袋：NVIDIA 在 GH200 上验证过 1:1，发现 CPU 侧带宽有富余，于是在 GB200 上改成 1:2，用同样的 CPU 支撑两倍的 GPU 算力，摊薄了单颗 GPU 的 CPU 成本。</span>

把三者放在一起，GB200 超级芯片提供的是一套**完整的、可独立训练的节点**：Grace CPU 负责喂数据与调度，两颗 B200 负责计算，C2C 让它们像一台机器那样协作。而 36 个这样的超级芯片，再通过 NVLink5 全部互连，就组成了 GB200 NVL72。

**辨析｜超级芯片 ≠ 传统主机：** 传统服务器里，CPU 是「主人」，GPU 是插在 PCIe 槽里的「外设」，CPU 通过驱动把数据写进 GPU 显存；而在超级芯片里，Grace 与 Blackwell 是**对等伙伴**，共享同一份一致性地址空间，CPU 可以直接访存 GPU 的显存，反之亦然。前者是「把数据搬过去」，后者是「大家读写同一块内存」——这是两代异构计算在编程体验上最本质的区别。

## 3 公式解析：1.4 exaflops 是怎么算出来的

GB200 NVL72 白皮书反复出现一个数字：**FP4 精度下最高 1.4 exaflops 的算力**。这个数字不是测得的目标值，而是从「单卡 × 数量」推出来的上限，拆开看只有三步：

$$
P_{\mathrm{NVL72}} = 72 \times P_{\mathrm{B200}} = 72 \times 20\ \mathrm{PFLOPS} \approx 1.44\ \mathrm{exaFLOPS}
$$

- **第一步，读懂 $P_{\mathrm{B200}} = 20\ \mathrm{PFLOPS}$**：这是单颗 B200 在 FP4 精度、启用 **2:4 结构化稀疏**时的峰值。2:4 稀疏是一种硬件加速技巧——权重矩阵每 4 个元素里只保留 2 个非零，Tensor Core 就能跳过一半的乘法，稠密算力近似翻倍。没有这个前提，B200 的 FP4 稠密算力是约 10 PFLOPS。
- **第二步，乘以 72**：GB200 NVL72 一共装 72 颗 B200 GPU（36 个超级芯片 × 2 颗）。所有 GPU 的峰值相加，就得到机架的聚合峰值。
- **第三步，单位换算**：$72 \times 20 = 1440\ \mathrm{PFLOPS}$，而 $1\ \mathrm{exaFLOPS} = 1000\ \mathrm{PFLOPS}$