---
title: 热迁移
date: 2026-09-08
section: cs
---

# 热迁移

<div class="epigraph">
<p>热迁移迭代拷贝客户内存，再短暂停机拷剩余脏页与设备状态；postcopy 用 [userfaultfd](/cs/userfaultfd) 在目标上按缺页拉。 </p>
<footer>—— 据 Clark et al., Live Migration of Virtual Machines, NSDI 2005；KVM 迁移文档</footer>
</div>

[气球](/cs/memory-balloon) 改变内存集合。[时钟](/cs/clock-virtualization) 必须在切瞬间对齐。缺口是 **live migration** 算法：预拷 vs 后拷。

## 问题

precopoy：循环拷脏，直到脏率可接受，downtime 拷 vCPU。postcopy：先切，缺页从源拉。缺口：设备状态、[SR-IOV](/cs/sriov-passthrough) 几乎不能迁；与 [NFS](/cs/nfs-semantics) 上的磁盘。本课不把 RDMA 迁移实现写完。

<span class="marginnote">术语翻译：脏页就是「拷过去之后又被客户机改写过的页」——目标上那份立刻过期，只能下一轮重拷。预拷的全部技巧，就是让这个集合一轮比一轮小，直到一轮就能拷完。</span>

<span class="marginnote">脏日志靠 EPT 写保护或 log dirty。对象是 VM 状态机，不是容器 runc 迁——容器后课。</span>

## 方法

建目标 VM → 迭代内存 → 停 → 切网络 → 跑。对照 [fs 快照](/cs/fs-snapshots)：一个钉存储根，一个搬运行态。对照 sendfile：块在宿主之间，不是文件语义保证。对照 kexec：同机换核，不是跨机。

```mermaid
flowchart TD
  PRE["预拷脏页"] --> DIRTY["收敛"]
  DIRTY --> STOP["短暂停机"]
  STOP --> RUN["目标继续"]
  POST["postcopy"] --> UFFD["缺页从源拉"]
```

## 机制

热迁移把「机器」收成可复制的内存+设备检查点，使维护不停服务。收敛失败（脏太快）是经典坑。不要写成云产品 SLA。与 [dm-crypt](/cs/dm-crypt)：盘可共享存储，只迁 RAM。

<span class="marginnote">数字实例：内存 8 GB、链路 1 GB/s。客户机每秒只写 0.5 GB 时，两三轮就收敛；若每秒写超过 1 GB，每轮新产生的脏页比拷走的还多，停机时刻永远等不来——这就是收敛失败的判据。</span>

安全：迁移通道要加密，否则内存明文在网上。


实现上：脏页率高于带宽则 precopy 永不收敛，要限 vCPU 或改 postcopy。磁盘必须是共享存储或一并拷。VF 直通要先解绑。 读法上只引用[上一课](/cs/memory-balloon)的结论，不把对象换成训练推理或限价簿。

```mermaid
flowchart TD
  R1["第 1 轮：拷全部内存"] --> W["客户机继续跑，产生新脏页"]
  W --> R2["第 2 轮：只拷脏日志里的页"]
  R2 --> CHK{"本轮脏页量 ≤ 一轮能拷的量？"}
  CHK -->|"是"| STOP["收敛：停机拷最后一批并切换"]
  CHK -->|"否"| SLOW["降脏率：限 vCPU 或改 postcopy"]
  SLOW --> W
```

<span class="marginnote">常见误区：以为迁移只是「把内存拷过去」。VF 直通设备的寄存器与队列状态几乎搬不走，要先解绑回落到软件接口；存储要么共享（只迁 RAM），要么连盘一起拷，否则目标机一跑就找错盘。</span>

本课在操作系统进阶的「虚拟化与隔离进阶 / Hypervisor」课序里，对象是 **热迁移**。

- 先修只引用，不重导：上一课的结论当公理，本课只补差。
- 五栏不吞并：不把本课写成大模型训练/推理，也不写成限价簿或权重量化。
- 文献用 OSTEP、McKusick、内核文档与具名会议论文；不发明 arXiv 编号。

## 边界

本课不引入所有 downtime 调参。不保证 GPU 直通可迁。下一课套娃：嵌套虚拟化。


版本字段会变，课序钉的是机制对象「热迁移」，不是某一主线内核的结构体名。
后课默认：VM 可迭代拷内存后切换。L1 上再跑 L2，下一课嵌套。

## 小结

- 预拷收敛脏页；postcopy 用缺页拉。
- 直通设备是迁移敌人。
- 嵌套虚拟化是下一课。
- 出处：Clark et al., NSDI 2005；KVM migration。
