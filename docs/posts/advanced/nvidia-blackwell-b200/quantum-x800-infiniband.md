---
title: 网络 Quantum-X800 InfiniBand
date: 2026-08-07
---

# 网络 Quantum-X800 InfiniBand

<div class="epigraph">
<p>机柜之内，NVLink 说了算；机柜之间，InfiniBand 说了算。</p>
<footer>—— NVIDIA 数据中心互连架构的口头禅，见于 Blackwell 网络白皮书</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 硬件：NVIDIA Blackwell/B200 ｜ NVIDIA Blackwell 白皮书 §7 ｜ 2026-08-07</p>
</div>

## 为什么从 InfiniBand 开始

NVL72 把 72 颗 GPU 变成一个域，但一个数据中心不止一个机柜。**把许多个 NVLink 域连起来、让它们作为一个整体训练同一个模型，靠的是高速数据中心网络**。NVIDIA 给出的首选方案是 **InfiniBand**——一种为高性能计算而生、支持 RDMA 的互连技术。Blackwell 配套的 Quantum-X800 把端口带宽推到 **800 Gb/s**，并首次把「归约计算」做进了交换机。这一篇讲清楚：InfiniBand 是什么、X800 快在哪、以及它和 NVLink 如何分工。

## 1 InfiniBand 与 RDMA：为「机器间传数据」而生

以太网是互联网的通用语言，但它从设计之初就没考虑「让两台服务器的内存直接互拷」。**InfiniBand（IB）** 则是为高性能计算定制的互连：低延迟、高带宽、硬件卸载，核心能力是 **RDMA（Remote Direct Memory Access，远程直接内存访问）**。

**RDMA 的意义在于「绕过操作系统」**。普通网络发送数据要经过「用户态 → 内核 → 协议栈 → 网卡」的漫长旅程；RDMA 让网卡直接从用户态内存把数据搬走，几乎不打扰 CPU、延迟低一个量级。<span class="marginnote">为什么训练大模型必须 RDMA？分布式训练里每几步就要做一次 AllReduce，每次通信的数据量以 GB 计；如果每次都要 CPU 参与拷贝与协议处理，CPU 会先累垮。RDMA 把「搬运」完全卸载到网卡，CPU 专心算数——这是《AI 基础设施》集合通信一节的硬件前提。</span>

在 Blackwell 时代，NVIDIA 的网卡角色由 **ConnectX-8 SuperNIC** 承担：每端口 800 Gb/s，同时支持 RDMA 与网内计算。

## 2 Quantum-X800：800G 与网内计算

Quantum-X800 是 NVIDIA 的 InfiniBand **交换机 + 网卡**产品家族，相对上一代 Quantum-2 的升级点是：

| 项目 | Quantum-2（NDR） | Quantum-X800（XDR） |
| --- | --- | --- |
| 端口速率 | 400 Gb/s | 800 Gb/s |
| 速率世代 | NDR | XDR（下一代） |
| 网内计算 | 支持 SHARP v2 | 支持 SHARP v3 |
| 每端口带宽 | 50 GB/s | 100 GB/s |

**XDR 800G 的「800」是比特率（Gb/s），换成字节率要除以 8：800 ÷ 8 = 100 GB/s。** 这个量级意味着「读一个 10GB 的 checkpoint，只要 0.1 秒」。

更值得一提的技术是 **SHARP（可扩展分层聚合与归约协议）**——**把通信的「计算」做进交换机**。分布式训练里最贵的操作是 AllReduce（所有 GPU 的结果求和再广播回去）。传统做法是数据在 GPU 间来回传、由某张卡算；SHARP 让**交换机在数据经过时顺手做求和**，数据「过一遍网」就完成了归约，通信量从「2N 次传输」降到「N 次传输」。<span class="marginnote">这是「在网计算（In-Network Computing）」的经典案例：让网络设备不只搬数据，还参与计算。类比：快递网点不只是中转包裹，还顺手把包裹内容加总。NVIDIA 把这一招用在 AllReduce 上，直接砍掉一半通信量——对通信密集的梯度同步是巨大的收益。</span>

## 3 从 NVLink 域到数据中心：两段网络的接力

理解 Blackwell 网络架构，关键是记住**两段接力**：

**第一段：机柜内**。72 颗 GPU 走 NVLink 5（1.8 TB/s），由 NVSwitch 织成全连接域。它快、但它只在机柜内。
**第二段：机柜间**。每个机柜作为「超级节点」，通过 ConnectX-8 SuperNIC 的 800 Gb/s 端口连到 Quantum-X800 交换机，再织成大规模网络（胖树等拓扑）。

为什么需要两段而不是一段？因为**带宽与规模的权衡**：NVLink 的物理距离与端口成本决定了它做不大；InfiniBand 用相对低的每端口成本换取可扩展到成千上万个节点。**训练超大规模模型时，通信被分级：能待在 NVLink 域内的通信（张量并行、专家并行）绝不出去；必须跨域的通信（数据并行梯度同步）才走 IB。**<span class="marginnote">这套「分级通信」与《大模型预训练》的并行策略一一对应：模型并行（tensor/expert）放在 NVLink 域内，因为它们每次迭代都通信、对带宽最敏感；数据并行（DP）的梯度同步频率低、量可控，可以承受 IB 的延迟。软硬件两层设计是咬合的。</span>

NVIDIA 对 IB 网络的定位不止「管道」：**自适应路由（adaptive routing）** 让数据避开拥塞路径、**拥塞控制**防止某几条链路打爆全局，加上 SHARP 的网内归约，共同构成「AI 专用网络」的完整叙事。

## 4 公式解析：800 Gb/s 到底多快

比特率与字节率的换算是网络话题里最常见的坑，值得单独拆一遍：

$$
B_{\text{GB/s}} = \frac{R_{\text{Gb/s}}}{8} = \frac{800}{8} = 100\ \text{GB/s}
$$

逐项拆解：

- **$R_{\text{Gb/s}} = 800$**：端口速率，单位是「吉比特每秒」。
- **$\div 8$**：8 bit = 1 byte，把比特换成字节。
- **$= 100\ \text{GB/s}$**：单端口字节率。

把 100 GB/s 放进上下文感受量级——与 NVLink 5 的 1.8 TB/s 相比：

$$
\frac{B_{\text{NVLink5}}}{B_{\text{IB XDR}}} = \frac{1800\ \text{GB/s}}{100\ \text{GB/s}} = 18
$$