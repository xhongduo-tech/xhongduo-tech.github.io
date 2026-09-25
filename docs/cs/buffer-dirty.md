---
title: 缓冲与脏页
date: 2026-09-08
section: cs
---

# 缓冲与脏页

<div class="epigraph">
<p>磁盘块在内存里有一份缓存；写先标脏，稍后才回写。崩溃时内存与介质可以不一致。</p>
<footer>—— 据 Silberschatz et al., Operating System Concepts；McKusick 等对 Unix 缓冲缓存的整理</footer>
</div>

[上一课](/cs/mount-super)让读写落到块号。[mmap](/cs/mmap) 与 [按需调页](/cs/demand-paging)已经把文件页放进帧。缺口是统一承认：**页 Cache / 缓冲缓存**是磁盘的 Cache，脏位表示比介质新。数据库栏的 [WAL](/cs/wal) 正是在这个前提下谈日志先行。本课只钉缓冲与脏，不选磁盘臂算法。

## 问题

每次 `write` 若都同步落盘，管道式小写会把吞吐打穿。若只改内存，断电丢数据。缺口不是新的 inode 字段，而是策略：写进帧并标脏；读命中则不访盘；回写可由定时器、内存压力（[置换](/cs/page-replace)）或 `fsync` 触发。同一物理块不应在缓冲里有两份互相不知道的副本。

本课不把 write-back 与 write-through 的微架构 Cache 课再推一遍；对象换成磁盘块。

<span class="marginnote">直觉类比：脏页像收银台的暂记账本——顾客买东西先记账（写内存），钱还没进保险柜（磁盘）。记账快所以收银快，但断电时没誊清的条目就丢了；`fsync` 就是「立刻誊进保险柜并当面点清」的强制命令。</span>

<span class="marginnote">脏页是还没到稳定介质的字节。`fsync` 要求该文件相关脏页与必要元数据到达持久存储。组提交与延迟分配是优化。</span>

## 方法

读：查页 Cache，无则分配帧、读盘、插入。写：在 Cache 中修改，标脏。置换选到脏页必须先写盘。元数据（inode、目录）同样走缓冲，崩溃可留下「目录项在、inode 未分」或相反，这是 fsck 与日志要补的，本课只暴露不一致可能。

```mermaid
flowchart TD
  RW["read/write/mmap"] --> CACHE["页 Cache"]
  CACHE --> HIT["命中: 不访盘"]
  CACHE --> DIRTY["写则脏"]
  DIRTY --> WB["回写 / fsync / 置换"]
```

## 机制

缓冲把磁盘的高延迟从大多数 `read` 里拿掉，让[调度指标](/cs/scheduling-metrics)里的 I/O 等待下降。它也让 mmap 与 `read` 看见同一帧成为可能：统一 Cache。脏页是正确性与性能的交界：OS 允许短暂不一致；数据库用 WAL 在更高层再加一条顺序。本栏不把关系库的 REDO 写进文件系统课。

与关中断锁的关系：Cache 索引是共享结构，下半部完成 DMA 后要把页标有效，锁规则沿用同步课。

下图回答一个具体问题：一页从干净到脏再回到干净，会经过哪些状态、谁触发转移。

```mermaid
flowchart TD
  LOAD["首次读：帧装入，页干净"] --> W["write 或 mmap 写入：标脏"]
  W --> D["脏：内存比介质新"]
  D -->|"定时器到点"| WB["回写"]
  D -->|"内存压力：置换选中"| WB
  D -->|"应用调 fsync"| WB
  WB --> C["介质已新：页变干净，仍留 Cache"]
```

<span class="marginnote">数字实例：顺序写 1 GB 文件，`write` 只是把数据拷进页 Cache，按内存速度微秒级完成；真正落盘由后台按磁盘速度（每秒几十 MB 到几 GB 不等）慢慢进行。若一直不 `fsync`，断电最多丢掉最后一个回写周期（常见约 30 秒）内的修改。</span>

## 边界

本课不引入 O_DIRECT 绕过 Cache 的全部语义，不把电池后备写缓存当默认硬件。也不保证 `write` 返回即持久——那正是脏页的含义。下一课才问：许多脏页要写时，请求以什么顺序送向设备。

写回顺序若先数据后 inode 大小，崩溃可能泄露旧数据；日志文件系统用事务把元数据捆在一起，本课只承认问题存在。

后课默认：内存可以比磁盘新。回写与读请求如何排队到磁头或闪存，下一课磁盘调度。

<span class="marginnote">常见误区：初学者以为 `write` 正常返回就等于数据已在磁盘上。它只保证进了页 Cache 并标脏；掉电或内核崩溃就可能丢。要持久必须 `fsync`，且数据与元数据的落盘顺序还要对——这正是数据库强调 WAL 先行的原因。</span>

## 小结

- 文件页经统一 Cache；写标脏，稍后回写。
- 崩溃时允许不一致；fsync 是显式持久。
- WAL 假定本课这个缺口已经在。
- 出处：Silberschatz et al., *OSC*；McKusick, Bostic, Karels, Quarterman, *The Design and Implementation of the 4.4BSD Operating System*。
