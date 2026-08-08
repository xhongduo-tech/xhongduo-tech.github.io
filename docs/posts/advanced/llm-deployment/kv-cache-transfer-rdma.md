---
title: 跨节点 KV Cache 传输与 RDMA
date: 2026-08-07
---

# 跨节点 KV Cache 传输与 RDMA

<div class="epigraph">
<p>数据搬家，最贵的不是网线，而是 CPU 的每一次插手。</p>
<footer>—— RDMA 设计哲学（源自 InfiniBand 社区）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ RDMA 与 KV 传输实践论文 ｜ 2026-08-07</p>
</div>

## 为什么从 KV 传输与 RDMA 开始

PD 分离和 Mooncake 都绕不开一个物理现实：**KV Cache 要从 prefill 节点搬到 decode 节点**，跨机器、跨网络。一个长 prompt 的 KV Cache 可达数 GB，传输它要用到「把 CPU 踢出数据通路」的高速网络技术——**RDMA（Remote Direct Memory Access，远程直接内存访问）**。不理解 RDMA，就无法理解「为什么 KV 传输能这么快」以及「传输到底卡在哪」。<span class="marginnote">本专题《PD 分离》里 KV 传输是「新问题」，这篇把它挖到底：<strong>传输的瓶颈不是带宽，是 CPU 参与带来的延迟</strong>。RDMA 让数据从一块 GPU 的显存直达另一块 GPU 的显存，绕过 CPU 与内核。</span>

本篇讲 KV 传输的现实约束、RDMA 的原理（与传统 TCP/IP 的对比）、以及 GPU 间直达（GPUDirect）如何让它更快。

## 1 KV 传输的现实：量级与约束

先给 KV 传输定个量级。70B 模型、8k 隐藏维、单头 d=128 的 128k 上下文，KV Cache 体积（见《KV Cache 显存占用估算》）约为：

$$V_{\text{KV}} = 2 \times L \times H \times d_h \times \text{bytes} \approx 128 \text{ GB}$$

这么大的数据跨节点传输，传输时间 $V_{\text{KV}} / BW$ 在 100 Gbps 网络下约 10 秒——**比生成还慢，完全不可接受**。<span class="marginnote">所以工程上 KV 传输从不传「全量」：要么<strong>量化压缩</strong>（见 KV Cache 量化篇，减到 1/4–1/2），要么<strong>只传增量</strong>（prefix 复用时只传新 token 的部分），要么<strong>重叠传输</strong>（边生成边传）。RDMA 负责把「必须传的」传得快。</span>

KV 传输的另一个约束是**布局一致性**：prefill 节点用 TP=8 切分 KV，decode 节点也必须 TP=8 且切法一致，否则 KV 块无法拼接。这意味着 KV 传输不是「搬一个文件」，而是「在多卡间按并行布局对齐地搬」。

## 2 RDMA 与传统网络的本质区别

传统 TCP/IP 网络传输数据要经过一串 CPU 环节：应用 → 内核 → 网卡 → 对端内核 → 对端应用。每次收发都触发 CPU 中断、上下文切换、数据拷贝——**CPU 是数据通路上最慢的一环**。

RDMA 的颠覆：**网卡直接访问内存**，数据从源内存直达目标内存，中间不经过两端 CPU：

网卡有 DMA 能力，直接读写宿主内存（或 GPU 显存）；
应用在**用户态**直接下发「发到哪、读到哪」的指令，不进内核；
对端网卡直接写入目标内存后，用硬件事件通知应用「数据到了」。

**RDMA 消除了「CPU 转发数据」这个环节**，把传输延迟从几十微秒压到几微秒，同时 CPU 占用几乎为零——这让 CPU 可以专心做调度，让数据传输几乎免费。

**辨析｜易错点：RDMA 快不是「网线快」，而是「少绕路」。** 同样一根 100 Gbps 的链路，TCP/IP 能跑出 20–30 Gbps 的有效吞吐（协议开销 + CPU 瓶颈），RDMA 能跑出 90 Gbps 以上。**差距来自协议栈与 CPU 参与，不是物理带宽**。这也是为什么「InfiniBand vs 以太网」之争的本质是「RDMA 能力之争」而不是带宽之争。

## 3 GPUDirect：GPU 到 GPU 的直达

KV Cache 存在 GPU 显存里，要把它搬到另一台机器的 GPU 显存。如果走「GPU → CPU 内存 → 网卡 → 对端 CPU → 对端 GPU」，要经过两段 PCIe 拷贝。**GPUDirect RDMA** 让网卡直接读写 GPU 显存：

**GPUDirect RDMA（写显存）**：源网卡从源 GPU 显存直接读取数据，目标网卡直接写入目标 GPU 显存——PCIe 只走一次，CPU 全程不碰数据。
需要 GPU 显存固定（pinned）与 CUDA 的 peer 机制配合，工程上是「分配锁页内存 + 注册 GPU 内存给 RDMA」。

这让 KV 传输的路径变成「显存 → 网卡 → 网络 → 网卡 → 显存」，**端到端延迟达到微秒级、吞吐接近网卡线速**。<span class="marginnote">GPU 与网卡之间的 PCIe/NVLink-C2C 链路带宽决定了单机吞吐上限，<strong>跨机部署时「网卡数 × 每网卡带宽」要与 KV 传输需求匹配</strong>——H100 机通常配多张 400 Gbps InfiniBand。</span>

## 4 公式解析：传输延迟的组成

一次 KV 传输的端到端延迟分解：

$$T_{\text{transfer}} = T_{\text{latency}} + \frac{V_{\text{KV}}}{BW} + T_{\text{overhead}}$$

- **第一步，读 $T_{\text{latency}}$**：首字节延迟（round-trip time 的一半量级）。RDMA 下约 1–5 微秒；TCP 下受 CPU 中断与协议栈影响，可达 10–100 微秒。**数据量小（短 KV）时延迟项主导**。
- **第二步，读 $V_{\text{KV}}/BW$**：传输吞吐项。$V_{\text{KV}}$ 量化压缩后变小，$BW$ 靠 RDMA 逼近线速。**数据量大（长 KV）时吞吐项主导**。
- **第三步，读 $T_{\text{overhead}}$**：两端 CPU 的「注册、通知、回收」开销。RDMA 的零拷贝让这项趋近于零；TCP 的拷贝与中断让这项随数据量增长。**这就是 RDMA 在 KV 传输上不可替代的原因：延迟项、吞吐项、开销项三项都被优化**。

工程经验：把 KV 传输与 decode 生成**重叠**（先传前缀、边生成边传增量），能让传输时间「藏」在生成时间之下——传输不再成为端到端延迟的瓶颈项。

## 5 小结

- **KV Cache 跨节点传输量巨大**：128k 上下文的 KV 可达 128 GB，必须靠压缩、增量传输与重叠调度控制。
- **RDMA 绕开 CPU**：网卡直读直写内存，用户态下发指令，传输延迟从几十微秒降到几微秒、吞吐接近线速。
- **快的原因是少绕路**：协议栈与 CPU 参与才是瓶颈，不是物理带宽。
- **GPUDirect 让显存直达**：网卡直接读写 GPU 显存，KV 从一块 GPU 直达另一块 GPU。
- **延迟分解**：首字节延迟 + 吞吐项 + 开销项，RDMA 三项俱优；与生成重叠可让传输几乎免费。

在下一节，我们系统化「多机推理到底贵在哪」——**多机推理的通信开销分析**。
