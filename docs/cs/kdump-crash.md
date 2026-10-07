---
title: kdump 与 crash
date: 2026-09-08
section: cs
---

# kdump 与 crash

<div class="epigraph">
<p>kdump 用 kexec 进第二内核，把崩溃瞬间的内存当 ram 盘转出；crash 工具在事后解析 vmcore。</p>
<footer>—— 据 Linux kdump 文档；kexec；[printk](/cs/printk-logging) 环可能不够事后分析</footer>
</div>

[eBPF](/cs/ebpf-observability) 看活体。机器已经 oops。缺口是 **保留内存 + 第二内核** 抓 dump。

## 问题

崩溃时锁可能死、文件系统可能不可信。kexec 跳到预留的 capture kernel，旧内存只读。缺口：预留大小；加密盘；与 watchdog 的配合。本课不把 crash 命令当教程全文。

<span class="marginnote">pstore/ramoops 是更小的紧急日志。对象是整机内存镜像，不是 coredump 用户进程——那是另一条。</span>

## 方法

启动预留 crashkernel → panic 时 kexec → 写 vmcore。对照 [hibernate](/cs/suspend-resume)：一个有意写镜像，一个意外。对照 [fsck](/cs/fsck)：dump 后再修盘。对照 makedumpfile 过滤页。

<span class="marginnote">直觉类比：crashkernel 预留区像救生艇——平时占着甲板空间没人用；船一进水（panic），你不可能现场造船，只能跳上早已挂好的那艘。「不能崩溃后再装」说的就是这个。</span>

```mermaid
flowchart TD
  PAN["panic"] --> KX["kexec 捕获内核"]
  KX --> VM["写 vmcore"]
  VM --> CR["crash 解析"]
```

## 机制

kdump 把「死内核的 RAM」变成可分析文件，是生产根因的最后手段。它需要事先预留，不能崩溃后再装。不要写成云监控套餐。与 [KPTI](/cs/kpti-os)：解析要懂页表。

失败：预留不够、设备驱动在第二内核缺失。


实现上：crashkernel= 预留必须在启动时做，崩溃后再留来不及。第二内核驱动要能写盘或网，常用精简 config。过滤掉缓存页可缩小 dump。 读法上只引用[上一课](/cs/ebpf-observability)的结论，不把对象换成训练推理或限价簿。

<span class="marginnote">数字实例：一台 256 GB 内存的机器，通常只用启动参数预留几百 MB（如 crashkernel=512M）给第二内核；事后用 makedumpfile 过滤掉缓存页，几百 GB 的内存现场常能压成几 GB 的 vmcore 文件。</span>

```mermaid
flowchart TD
  B["启动时：划出 crashkernel 预留区，平时闲置"] --> C["运行期：正常内核用其余内存"]
  C --> P["某天 panic：原内核已不可信"]
  P --> K["kexec 把 capture kernel 装进预留区并跳过去"]
  K --> R["旧内核内存原封不动，只读"]
  R --> W["capture kernel 把它转成 vmcore"]
  W --> A["事后用 crash 分析根因"]
  B -->|"没预留"| X["崩溃后无干净内存放第二内核，dump 直接失败"]
```

本课在操作系统进阶的「安全、启动与调试 / 观测与调试」课序里，对象是 **kdump 与 crash**。

- 先修只引用，不重导：上一课的结论当公理，本课只补差。
- 五栏不吞并：不把本课写成大模型训练/推理，也不写成限价簿或权重量化。
- 文献用 OSTEP、McKusick、内核文档与具名会议论文；不发明 arXiv 编号。

## 边界

本课不引入所有过滤策略。不保证虚拟机的 pvpanic 路径。下一课崩溃语义：oops 与 panic。


版本字段会变，课序钉的是机制对象「kdump 与 crash」，不是某一主线内核的结构体名。
后课默认：可 kexec 出 vmcore。oops 与 panic 如何决定活还是死，下一课。

<span class="marginnote">常见误区：把 kdump 当成「出了事装个工具就能抓」。实际上它依赖崩溃前就位的三件事——启动参数预留、第二内核的精简 initramfs、能写盘或传网的驱动；缺一件，现场就只剩重启一条路。</span>

## 小结

- kdump：预留内核 + kexec + vmcore。
- 分析用 crash；依赖事先配置。
- oops/panic 是下一课。
- 出处：kdump；kexec；crash 工具。
