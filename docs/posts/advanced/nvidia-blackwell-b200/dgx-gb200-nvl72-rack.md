---
title: DGX GB200 NVL72 整柜系统
date: 2026-08-07
---

# DGX GB200 NVL72 整柜系统

<div class="epigraph">
<p>我们要造的不是一台更快的服务器，而是一台更大的 GPU——大到 72 颗 GPU 装进一个机柜，对外仍是一颗。</p>
<footer>—— 黄仁勋（Jensen Huang）在 GTC 2024 介绍 GB200 NVL72</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 硬件：NVIDIA Blackwell/B200 ｜ NVIDIA Blackwell 白皮书 §5 ｜ 2026-08-07</p>
</div>

## 为什么从整柜系统开始

前两篇把镜头从「两颗裸片」拉到了「超级芯片」。但 GB200 的设计终点不是超级芯片，而是**整个机柜**：DGX GB200 NVL72。NVIDIA 明确说过，**NVL72 才是「一颗完整的产品」，超级芯片只是它的零件**。把视角从芯片抬到机柜，你会发现量纲变了：算力以 EFLOPS 计，带宽以 TB/s 计，功耗以 kW 计。这一篇讲清楚这个「机柜级 GPU」是怎么拼出来、又为什么值得这么拼。

## 1 NVL72：一柜子的超级芯片与交换机

DGX GB200 NVL72 的物理构成可以数出来：

- **36 个计算托盘（compute tray）**，每个托盘放 1 颗 GB200 超级芯片（1 Grace + 2 B200），合计 **36 颗 Grace CPU + 72 颗 B200 GPU**；
- **9 个 NVLink 交换托盘**，每个托盘放 2 颗 NVSwitch 5，合计 **18 颗 NVSwitch**；
- 全部部件采用**液冷**，整柜功耗约 **120 kW**。

这 72 颗 B200 通过 NVSwitch 织成一个**全连接的 NVLink 域（NVLink domain）**：任意两颗 GPU 之间都有直达通路，带宽 1.8 TB/s。把 72 颗 GPU 的全部 NVLink 带宽加总，得到 NVIDIA 反复宣传的数字——**130 TB/s**。<span class="marginnote">NVL72 的「72」指的是 72 颗 Blackwell GPU；整柜 1.4 EFLOPS FP4 推理性能与 720 PFLOPS FP8 训练性能，都是「72 颗相加」得到的系统级数字。为什么要凑到 72 而不是 16、32？因为 1.8 万亿参数（1.8T）的 MoE 模型，恰好需要约 72 颗 × 192GB 的显存容量与带宽才能实时推理。</span>

与之相对的是传统做法：8 张 H100 装一节点，节点间走 InfiniBand 组网。NVL72 把「网络」从节点间搬进了机柜内，并且用比网络快一个量级的 NVLink 来代替网卡。

## 2 单逻辑 GPU：全 NVLink 域的意义

「72 颗 GPU 织成一个域」不是物理上连起来就完事，它带来三个软件层面的质变：

**其一，张量并行可以做到 72 路**。训练 1.8T MoE 或数百亿稠密模型时，把一层权重切成 72 份分别放在 72 颗 GPU 上，每次前向都要做一次全域 AllReduce。这个通信在 NVL72 内全部走 NVLink，而不是经过网卡与以太网/IB 交换机。**通信带宽翻了几个量级，张量并行的扩展效率因此逼近线性**。<span class="marginnote">对照《大模型预训练》里的并行策略：张量并行对带宽最敏感、通常只在单节点 8 卡内使用；NVL72 把「节点」扩大成「机柜」，等于把张量并行的适用范围放大了 9 倍。</span>

**其二，NVLink 域内不需要「网络栈」**。传统的多机训练里，数据要经历「GPU → PCIe → 网卡 → 交换机 → 网卡 → PCIe → GPU」的漫长路径，延迟以微秒计；NVLink 域内则近得几乎可以忽略。这对通信频繁的小消息尤其重要。

**其三，对外呈现为单一设备**。CUDA 程序可以把它当作一颗「有 72 倍算力的 GPU」来写（借助 cooperative groups、NVLink 域感知的库），系统软件负责把计算映射到 72 颗物理 GPU 上。这是「机柜即 GPU」的产品宣言：**开发者思考的是模型，而不是机柜里的拓扑**。<span class="marginnote">当然，「当作单颗」是一种可编程性上的便利，物理上仍是 72 颗独立 SM 集群；需要显式分布的场景（数据并行、专家并行）依然存在。这一点在《大模型部署》的并行策略篇有展开。</span>

## 3 公式解析：130 TB/s 与对分带宽

NVL72 的两个带宽数字值得各自算一遍账。

**第一笔：总聚合带宽**。72 颗 GPU，每颗 1.8 TB/s：

$$
B_{\text{agg}} = 72 \times 1.8\ \text{TB/s} \approx 130\ \text{TB/s}
$$

**第二笔：对分带宽（bisection bandwidth）**。对分带宽定义为：把网络切成两个各含一半节点的半边，跨越切面的总带宽。在全连接无阻塞拓扑里，任一半的 36 颗 GPU 各以 1.8 TB/s 与对侧通信，因此：

$$
B_{\text{bisect}} = \frac{N}{2} \times 1.8\ \text{TB/s} = \frac{72}{2} \times 1.8\ \text{TB/s} \approx 64.8\ \text{TB/s}
$$

- **$N/2$**：半边节点数，36。
- **$1.8\ \text{TB/s}$**：每颗 GPU 的 NVLink 出口带宽。
- **$64.8\ \text{TB/s}$