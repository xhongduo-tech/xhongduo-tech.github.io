---
title: 伪共享
date: 2026-09-08
section: cs
---

# 伪共享

<div class="epigraph">
<p>两个核写的是不同变量，只要落在同一 cache 行，协议仍按真共享来回作废；程序员看见的是无锁计数器变慢，不是数据竞争。</p>
<footer>—— 据 Bolosky and Scott；Hennessy and Patterson, CA:AQA 整理</footer>
</div>

[上一课](/cs/directory-scalability) 与 [MESI](/cs/mesi-protocol) 都按**行**维护副本。[行大小](/cs/line-size-sector) 越大，两个无关字段同行的概率越高。本课不重做目录 ack。缺口是把**伪共享（false sharing）**从「协议正确但性能崩」讲清楚：诊断是行颠簸，修复是对齐与填充，不是加目录。

## 问题

`stats[tid]++` 若 `stats` 是紧挨着的 4B 元素，核 0 与核 1 的计数器可能同行。每次 `++` 要 M，于是行在两核 L1 之间按 [MOESI](/cs/moesi-mesif) 飞。没有数据竞争：算法正确。缺口不是 fence，而是**共享粒度是行，不是 C 变量。**

<span class="marginnote">真共享：同一地址被多核读写，语义就需要通信。伪共享：地址不同、行相同。填充到行对齐把它们拆开。</span><span class="marginnote">直觉类比：像合租的两个人各自记自己的账，但账本（cache 行）只有一本——谁要写一笔都得先把整本抢到手，另一个人手里的立刻作废。争执不在账目本身，在「行」这个粒度。</span>

## 方法

诊断：PMU 上 LLC 写回/Snoop 命中对某数据段异常高，而代码无锁。修复：按行对齐每个核的热写字段；或线程局部再归约。结构体末尾 pad 到 64B。分区无法消除：配额再公平，inv 仍然发生。

```mermaid
flowchart TD
  V0["核0 写 x"] --> LINE["同一 cache 行"]
  V1["核1 写 y"] --> LINE
  LINE --> INV["来回作废"]
  PAD["x、y 分到不同行"] --> OK["各持 M"]
```

## 机制

CPI 的一致性项可以压过 DRAM：cache-to-cache 延迟虽低于内存，频率极高时核空等。与 [存储集](/cs/store-set-predict) 无关：那是单核内 load/store 地址。伪共享是多核行所有权。压缩与扇区：扇区可以减少传输字节，但标签/目录仍可能按整行作废，假共享未必消失。

数组下标按 tid 紧挨递增是教科书陷阱；改成 `tid * (64/sizeof)` 或用 per-cpu 再归约。C++ 的 `alignas(64)` 只保证起始对齐，结构体后面的字段仍可能与别人同行。

```mermaid
flowchart TD
  S["无锁多线程程序却随核数变慢"] --> Q1{"热点是无锁的频繁写吗？"}
  Q1 -->|"是"| Q2{"PMU：写回与嗅探计数异常高？"}
  Q2 -->|"是"| Q3{"热写变量的地址只差几十字节？"}
  Q3 -->|"是"| DX["判定伪共享：64B 对齐或填充拆行"]
  Q3 -->|"否"| TS["可能是真共享：需要原子或锁"]
  Q2 -->|"否"| ALG["往算法或内存带宽瓶颈找"]
```

<span class="marginnote">数字实例：一行 64 字节能装下 16 个 4 字节 int——`stats[16]` 整个数组挤在同一行，16 个线程各写各的下标也照样互相作废。把下标间隔拉到 $64/\text{sizeof(int)}=16$ 个元素，才真正「分家」。</span>

## 边界

本课不把无锁算法的 ABA 当伪共享。也不把 OS 的 per-cpu 数据展开，只点名那是同一药方。下一课原子操作：真共享时硬件如何在行上做 RMW。

后课默认：无逻辑共享仍可能行共享。真共享的 `swap`/`cas` 要在 cache 协议上实现，不是只靠 ALU。<span class="marginnote">常见误区：初学者容易以为「改成原子操作就能解决伪共享」；恰恰相反，atomic 让行所有权的切换更频繁，颠簸更凶。药方是让不同核的热写字段物理上离开同一行，一致性协议本身一行都不用改。</span>

## 小结

- 伪共享是行粒度一致性对无关变量的惩罚；对齐/填充拆行。
- 协议无需修改，软件布局是主药。
- 原子 RMW 如何占住一行，是下一课。
- 出处：Bolosky and Scott；Hennessy and Patterson, *CA:AQA*。
