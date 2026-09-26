---
title: RDMA 在 OS 中的位置
date: 2026-09-08
section: cs
---

# RDMA 在 OS 中的位置

<div class="epigraph">
<p>RDMA 把注册过的用户内存直接交给网卡 DMA，绕过 TCP 与套接字缓冲；内核只做连接、注册与完成队列的映射。</p>
<footer>—— 据 InfiniBand/RDMA 规范直觉；Linux rdma-core 与 ibverbs 文档；RFC 5040 对 iWARP 的背景</footer>
</div>

[零拷贝](/cs/zero-copy-network) 仍常走 TCP。[DPDK](/cs/dpdk-kernel-bypass) 旁路的是整栈。缺口是 **RDMA 在 OS 里的位置**：verbs、MR、QP，以及它不是文件系统的 RDMA 导出百科。

## 问题

HPC/存储要微秒级、少拷贝。RDMA：进程注册虚址，卡拿到翻译（或 on-demand paging）；QP 成对，post send/recv 或 RDMA WRITE 不经过对端 CPU。缺口：内核模块（ib_core）管理设备，用户 libibverbs mmap 门铃；与 socket 并行存在，应用要改 API。本课不把 RoCE 拥塞控制写成传输课全文。

<span class="marginnote">直觉类比：注册内存（MR）像给快递员（网卡）办的一张门禁卡——只对你登记过的那几间房（虚址区间）有效，房间里的东西怎么搬都不用你亲手经手（CPU 不碰字节）。QP 则是两栋楼之间拉的一条专用传送带，发送就是把货放上带，完成情况写在门口的完成队列板上。</span>

<span class="marginnote">rping 一类工具只为证明路径。NVMe-oF RDMA 把块命令放在这套 QP 上，存储栈与网络在此汇合。</span>

## 方法

打开设备 → PD → 注册 MR → 建 CQ/QP → 交换 QP 号（TCP 带外）→ post。完成：poll CQ，与 [NVMe](/cs/nvme-driver) 同构。对照 TCP：无 sk_buff 数据路径。对照 XDP：XDP 仍以太网；RDMA 是另一种 verbs 设备。

<span class="marginnote">数字实例：一次小消息往返，走内核 TCP 大约几十微秒、数据要经内核缓冲各拷一遍；走 RDMA 通常一两微秒、字节不经过任何 CPU。差出的几十微秒正来自省掉的系统调用、协议处理与两次拷贝——这就是 HPC 与 NVMe-oF 愿意重写 API 的全部理由。</span>

```mermaid
flowchart TD
  APP["ibv_post_send"] --> QP["队列对"]
  MR["注册内存"] --> NIC["RNIC DMA"]
  QP --> NIC
  NIC --> CQ["完成队列"]
  CQ --> POLL["用户 poll"]
```

## 机制

OS 把「安全 DMA 窗口」当成一等资源（MR），把网络变成类似块设备的队列。内核不再逐字节解释流。不要写成量化超低延迟撮合。错误：pin 过多内存会伤页回收——[rmap](/cs/rmap) 后课会再遇钉页。

与 netns：RDMA 设备的 ns 支持不完整于普通网卡，部署要查清。

<span class="marginnote">常见误区：初学者容易以为「RDMA 绕过内核」等于「完全不需要内核」。数据路径确实不进内核，但控制路径全程在内核：打开设备、建保护域、注册内存、创建 QP、故障恢复。把控制路径也写成用户态轮询，既得不到安全（MR 的授权本来就是内核发的），也救不了错误处理。</span>

```mermaid
flowchart TD
  APP["进程注册虚址区间"] --> MR["内核建 MR：钉页并把翻译给网卡"]
  MR --> DMA["RNIC 直接 DMA 该区间"]
  APP --> PIN["被钉的页不可回收"]
  PIN --> PRESS["长期注册过多：可迁移内存变少"]
  PRESS --> FRAG["页回收与 compaction 失败风险上升"]
```


实现上：MR 钉页对抗回收，注册过多会让 compaction 失败。QP 状态错误只在 CQ 上看见，应用要把完成当 ABI。RoCE 丢包行为与 InfiniBand 可靠服务不同。 读法上只引用[上一课](/cs/zero-copy-network)的结论，不把对象换成训练推理或限价簿。

本课在操作系统进阶的「网络栈 / 收发路径」课序里，对象是 **RDMA 在 OS 中的位置**。

- 先修只引用，不重导：上一课的结论当公理，本课只补差。
- 五栏不吞并：不把本课写成大模型训练/推理，也不写成限价簿或权重量化。
- 文献用 OSTEP、McKusick、内核文档与具名会议论文；不发明 arXiv 编号。

## 边界

本课不引入全部 QP 状态图。不保证 Soft-RoCE 的性能。网络课序收口于「还有套接字选项这层旋钮」。下一课把 POSIX 套接字选项钉住，作为应用仍走内核 TCP 时的接口。


版本字段会变，课序钉的是机制对象「RDMA 在 OS 中的位置」，不是某一主线内核的结构体名。
后课默认：RDMA 用 verbs 与注册内存，不经 TCP 数据路径。普通套接字的选项如何改内核行为，下一课。

## 小结

- RDMA：内核管注册与 QP，数据 DMA 进用户 MR。
- 与 socket/DPDK/XDP 是不同接头。
- 套接字选项是下一课。
- 出处：rdma-core；InfiniBand 直觉；Linux RDMA。
