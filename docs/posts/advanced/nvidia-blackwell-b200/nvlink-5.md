---
title: 第五代 NVLink 互联
date: 2026-08-07
---

# 第五代 NVLink 互联

<div class="epigraph">
<p>当模型大到一块芯片装不下，芯片之间的带宽就决定了一切。</p>
<footer>—— 多 GPU 系统设计的共识，见于 NVIDIA 互联技术文档</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 硬件：NVIDIA Blackwell/B200 ｜ NVIDIA Blackwell 白皮书 §6 ｜ 2026-08-07</p>
</div>

## 为什么从 NVLink 开始

前面的篇章已经反复提到 1.8 TB/s 的 NVLink 5 与 130 TB/s 的机柜带宽。这一篇正面回答：**NVLink 到底是什么、5 代演进做了什么、1.8 TB/s 这个数字是怎么凑出来的。** 多 GPU 是大模型的宿命——模型大到单卡装不下，训练与推理都要跨卡通信。**通信带宽决定多卡系统的扩展效率**：NVLink 就是 NVIDIA 为「GPU 之间说话」专门修的通道，它不是锦上添花，而是「机柜即 GPU」这条产品路线的地基。

## 1 NVLink 的谱系：从 P100 到 Blackwell

NVLink 是 NVIDIA 自 2016 年起为 GPU 间通信专门设计的**点对点高速互连**，每一代大致把带宽翻倍：

| 世代 | 代表 GPU | 每卡带宽 | 形态 |
| --- | --- | --- | --- |
| NVLink 1 | P100（2016） | 160 GB/s | 卡间专用链路 |
| NVLink 2 | V100 | 300 GB/s | 链路数增加 |
| NVLink 3 | A100 | 600 GB/s | 12 链路 × 50 GB/s |
| NVLink 4 | H100 | 900 GB/s | 18 链路 × 50 GB/s |
| NVLink 5 | B200 | 1.8 TB/s | 18 链路 × 100 GB/s |

这条谱系读出来一个规律：**每一代都把「每链路的数据率」或「链路数」往上顶，总带宽保持翻倍节奏。** NVLink 4 到 5 是「每链路数据率翻倍」——从 50 GB/s 提到 100 GB/s，链路数保持 18。<span class="marginnote">为什么一直用「链路 × 每链路带宽」的结构而不直接做一条巨宽的通道？因为高速串行互连（SerDes）的单通道带宽受物理极限约束，工程做法是「多通道并行」：18 条链路并列，每条跑它的物理极限。这与内存用「多通道」是一个思路。</span>

## 2 NVLink 5 的物理与逻辑：1.8 TB/s 从哪来

NVLink 5 的 1.8 TB/s 是**双向总带宽**：同一时刻既能发送也能接收，各 900 GB/s。从物理到逻辑，它的构成是：

**链路（link）**：每颗 B200 有 18 条 NVLink 链路；
**每链路带宽**：每条链路双向 100 GB/s（单方向 50 GB/s）；
- **串行数据率**：靠高速 SerDes 把数据编码成高速串行信号，在封装与 PCB 上传输；
- **协议**：NVLink 提供**低延迟、缓存一致或非一致**的访存语义，软件层看到的是「可以直接读邻居显存」的地址空间。

**NVLink 与 PCIe 的本质差异在「专」**：PCIe 是通用总线，要兼容硬盘、网卡、显卡等一切外设，协议开销大；NVLink 只为「GPU 访问 GPU」设计，牺牲通用性换取带宽与延迟。<span class="marginnote">体系结构里这叫「domain-specific interconnect」：为特定工作负载定制的互连。NVLink 的出现说明一个趋势——通用 PCIe 已经满足不了「多 GPU 高强度通信」的需求，专用互连成为数据中心 GPU 的标配。这个判断在《计算机体系结构》的「数据中心级架构」一章有更系统的讨论。</span>

## 3 NVLink 与 PCIe：两种互连的分工

一个容易混淆的问题是：「既然有 NVLink，为什么还要 PCIe？」答案是**分工不同**：

| 维度 | NVLink 5 | PCIe Gen6（×16） |
| --- | --- | --- |
| 典型带宽 | 1.8 TB/s | 约 256 GB/s |
| 用途 | GPU ↔ GPU 数据面 | 外设连接、启动、管理面 |
| 延迟 | 极低，微秒以下 | 相对较高 |
| 拓扑 | 点对点 / 经交换机 | 树形共享总线 |
| 协议开销 | 低 | 高（兼容一切设备） |

**NVLink 承担「数据面」**：训练时的 AllReduce、推理时的张量并行通信，全走 NVLink，因为量大、要求快；
**PCIe 承担「管理面」**：驱动加载、错误上报、固件更新、外设枚举，量小、不要求极致带宽。

NVIDIA 还提供 **NVLink-C2C（Chip-to-Chip）** 作为 NVLink 的变体，用于 CPU-GPU 之间（GB200 超级芯片里 Grace 与 B200 的 900 GB/s 走的就是它）。一个产品家族覆盖「片内、片间、卡间」三个尺度，这是只有同时做芯片与系统的厂商才能给出的统一方案。<span class="marginnote">对比 CPU 界的 CXL（Compute Express Link）：CXL 也想统一「缓存一致互连」，走的是 PCIe 物理层；NVLink 则完全自建物理层。两条技术路线在数据中心互连上的竞争，是《AI 基础设施》专题「互连技术」一节的话题。</span>

## 4 公式解析：18 条链路 × 100 GB/s

NVLink 5 的总带宽是小学算术，但把量纲理清有收获：

$$
B_{\text{NVLink5}} = N_{\text{links}} \times b_{\text{link}} = 18 \times 100\ \text{GB/s} = 1.8\ \text{TB/s}
$$

逐项拆解：

- **$N_{\text{links}} = 18$**：每颗 B200 的 NVLink 链路数。18 条链路对称地分布在 GPU 四周，与 NVSwitch 或邻居 GPU 相连。
- **$b_{\text{link}} = 100\ \text{GB/s}$