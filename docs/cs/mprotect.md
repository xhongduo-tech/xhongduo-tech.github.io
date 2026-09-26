---
title: mprotect
date: 2026-09-08
section: cs
---

# mprotect

<div class="epigraph">
<p>已存在的虚区间可以改权限：读、写、执行的组合变了，页表项与 TLB 必须跟着变。</p>
<footer>—— 据 POSIX mprotect；Silberschatz 对保护域的整理</footer>
</div>

[上一课](/cs/mmap)已经能把文件或匿名页接到 VMA。[分页](/cs/paging-vm)的权限位在缺页时检查。[用户/内核分裂](/cs/user-kernel-split)禁止用户给内核页加写。缺口是运行中**改自己的用户页权限**：JIT 要先写后执行，栈要 NX，调试器要临时可写。不是新的映射，是改已有映射。

## 问题

若只能在 mmap 时定权限，代码生成器必须unmap 再 map，丢失内容或窗口不安全。`mprotect` 按页对齐改一段 VA 的 r/w/x。缺口：拆 VMA、改 PTE、[TLB shootdown](/cs/tlb-shootdown)。把可写改成不可写可能触发后课 COW 已处理的逻辑；把不可执行改成可执行要受安全策略约束——本课只要求内核拒绝无意义组合，不讲攻击步骤。

本课不把 `pkey` 与内存保护键的全部 ISA 写完。

<span class="marginnote">`madvise` 提示内核预读或可丢页，不改权限。`mincore` 查询哪些页在内存。本课合并：它们操作同一段 VMA，对象仍是区间，不是新的页表格式。</span>

## 方法

用户给出区间与新 prot。内核按页对齐，必要时把原 VMA 劈成三段，中间段新权限。已存在的页改 PTE；未分配的页只改 VMA，下次[缺页路径](/cs/page-fault-path)按新权限填。他核 TLB 必须 shootdown，否则仍按旧写位存储。失败返回 EACCES/ENOMEM，不留下半改的权限。

<span class="marginnote">页对齐用数字看：4 KB 页、起始地址 $0x401023$ 时，mprotect 实际作用的是从 $0x401000$ 开始的整页——低 12 位被抹掉上取整。改一页只改了这 4096 字节，邻页权限原封不动，这就是要拆 VMA 的原因。</span>

```mermaid
flowchart TD
  MP["mprotect"] --> SPLIT["拆合 VMA"]
  SPLIT --> PTE["改已填 PTE"]
  PTE --> SD["TLB shootdown"]
```

## 机制

mprotect 让保护变成动态的：同一物理页在不同时刻对同一进程可写或不可写。这与文件模式位不同——那是后课 inode 上的 rwx，对象是文件；这里是虚存。W^X 策略（写与执行不同时）由用户或安全模块在调用时拒绝，不是 MMU 新周期。

```mermaid
flowchart TD
  JIT["JIT 生成机器码"] --> W["阶段一: mprotect RW"]
  W --> GEN["往页里写指令字节"]
  GEN --> X["阶段二: mprotect RX"]
  X --> RUN["CPU 执行这页: 不可写"]
  RUN --> FIX{"要补写代码?"}
  FIX -->|"是"| W2["先改回 RW 再写"]
  W2 --> X
  FIX -->|"否"| SAFE["注入攻击写不进这页"]
```

<span class="marginnote">W^X 就是「同一页要么可写、要么可执行，永不同时」的纪律：写着的页不让 CPU 执行，执行中的页不让任何人改。JIT 被迫来回 mprotect 正是代价——每次切换都是一次系统调用加一轮 TLB 失效，所以 JIT 会攒一大批代码一次性切换。</span>

不要重导页表 walker。只改权限位与 VMA 标志。

## 边界

本课不引入 `userfaultfd` 与保护键的组合拳。不保证所有架构支持对文件共享映射随意加执行。区间必须页对齐；未对齐由内核上取整或报错，POSIX 有规定。下一课离开用户 VMA，问内核自己的物理页从哪来。

<span class="marginnote">初学者容易把 mprotect 当成「内存版的 chmod」。实际上文件 rwx 管的是「谁能打开这个文件」，改的是 inode 元数据；mprotect 改的是 CPU 每次访存时硬件实时检查的页表权限位——一个是门禁名单，一个是车间里的急停闸，层级完全不同。</span>

后课默认：用户可改已有映射的权限。内核分配 2^n 页块给页表与缓冲，下一课 buddy。

## 小结

- mprotect 改 VMA 与 PTE 权限，并 shootdown。
- 未触碰的页只改区间描述，缺页时再生效。
- 内核页框分配器是 buddy 的缺口。
- 出处：POSIX.1 `mprotect`；Silberschatz et al., *OSC*；Love, *LKD*。
