---
title: 零拷贝网络
date: 2026-09-08
section: cs
---

# 零拷贝网络

<div class="epigraph">
<p>零拷贝让包数据只被 DMA 一次：MSG_ZEROCOPY、sendfile、AF_XDP umem 或包头仅线性拷贝，负载留在页上。</p>
<footer>—— 据 Linux MSG_ZEROCOPY 文档；sendfile 实践；AF_XDP 对 umem 的说明</footer>
</div>

[sendfile](/cs/sendfile-splice) 已从文件侧减拷。[sk_buff](/cs/skbuff) 的 frags 是载体。[RSS](/cs/rss-multiqueue) 解决 CPU。缺口是 **网络零拷贝** 的几条路径与完成通知——不是绝对零。

## 问题

`send` 默认拷进 skb 线性区或内核页。`MSG_ZEROCOPY`：用户页钉住，协议用，完成用错误队列通知可改页。接收：`recvmmsg` 仍常拷；真正少拷是 mmap 包环、AF_XDP、或 GRO 后直接 splice 到文件。缺口：页必须对齐、不能立即重写；校验卸载与加密可能迫使拷贝。本课不把 TLS 内核卸载细节写完。

<span class="marginnote">数字实例：发 1 GB 静态文件，普通路径要在用户缓冲与内核之间 memcpy 一到两次，纯 CPU 拷贝按 10 GB/s 算约耗 0.1–0.2 秒的核时间；sendfile/零拷贝把这段归零，只剩 DMA 与协议开销——带宽没变，省的是 CPU。</span>

<span class="marginnote">「零拷贝」统计上常是「少一次」。调试要用完成通知，否则数据竞争。与 O_DIRECT 同构：应用承担生命周期。</span>

## 方法

发送静态文件：sendfile。发送用户缓冲：MSG_ZEROCOPY + 等 completion。接收旁路：AF_XDP 把 umem 填给驱动。对照 DPDK：全程用户 mbuf。对照 [mmap 文件](/cs/fs-mmap-coherence)：文件零拷贝与网络零拷贝可在 sendfile 汇合。

```mermaid
flowchart TD
  U["用户页或文件页"] --> PIN["钉页"]
  PIN --> DMA["网卡 DMA"]
  CMP["完成通知"] --> UNPIN["才可改页"]
  COPY["普通 send"] --> SKB["内核拷贝"]
```

## 机制

零拷贝把 CPU memcpy 从数据面拿掉，瓶颈回到 DMA 与协议状态。它不取消 [套接字缓冲](/cs/socket-buffers) 的会计——记账仍按字节。不要写成网卡广告。与 XDP：XDP 改的是早处理，数据仍可在同一页上 TX。

<span class="marginnote">直觉类比：零拷贝像把包裹"原箱直发"——你不拆开重新装箱，只在箱面贴张新运单（包头），交给快递（网卡）直接拉走。代价是箱子在签收（完成通知）前不许动；提前改箱内东西，寄出去的就是被改过的货。</span>

<span class="marginnote">常见误区：初学者容易以为换了 sendfile/MSG_ZEROCOPY 就"绝对零拷贝"。实际 TLS 加密、校验和卸载失败、页不对齐等都会让内核静默回退成拷贝——省没省成，要靠完成通知与计数验证，不能只看 API 名字。</span>

```mermaid
flowchart TD
  APP["应用提交发送"] --> D{"走哪条发送路径"}
  D --> NZ["MSG_ZEROCOPY：页钉住"]
  D --> SF["sendfile：文件页直发"]
  D --> CP["普通 send：拷进内核"]
  NZ --> DMA["网卡 DMA 负载页"]
  SF --> DMA
  CP --> CP2["CPU 搬运后再 DMA"]
  DMA --> ACK["完成通知回错误队列"]
  ACK --> REUSE["此后才可重用缓冲"]
  CP2 --> REUSE
```

失败回退到拷贝必须正确，否则静默损坏。


实现上：MSG_ZEROCOPY 的完成通知走错误队列，应用必须读，否则不能重用缓冲。页必须 page-aligned 且生命周期覆盖 DMA。校验与加密卸载失败会静默拷贝。 读法上只引用[上一课](/cs/rss-multiqueue)的结论，不把对象换成训练推理或限价簿。

本课在操作系统进阶的「网络栈 / 收发路径」课序里，对象是 **零拷贝网络**。

- 先修只引用，不重导：上一课的结论当公理，本课只补差。
- 五栏不吞并：不把本课写成大模型训练/推理，也不写成限价簿或权重量化。
- 文献用 OSTEP、McKusick、内核文档与具名会议论文；不发明 arXiv 编号。

## 边界

本课不引入所有 NIC 的 header-data split。不保证虚拟机 vhost 零拷贝与客户页的 IOMMU 映射总成功。下一课把「绕过 TCP 的传输」接到 OS：RDMA。


版本字段会变，课序钉的是机制对象「零拷贝网络」，不是某一主线内核的结构体名。
后课默认：发送路径可钉用户/文件页 DMA。RDMA 在 OS 中的队列对与内存注册，下一课。

## 小结

- 零拷贝靠钉页与 DMA，完成前不能改缓冲。
- sendfile、MSG_ZEROCOPY、AF_XDP 是不同接头。
- RDMA 是下一课。
- 出处：Linux `MSG_ZEROCOPY`；AF_XDP；sendfile。
