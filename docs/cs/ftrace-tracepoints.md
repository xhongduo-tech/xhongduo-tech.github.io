---
title: ftrace 与 tracepoints
date: 2026-09-08
section: cs
---

# ftrace 与 tracepoints

<div class="epigraph">
<p>tracepoint 是编译进内核的钩，默认几乎为零成本；ftrace 把它们接到环形缓冲，并可做函数图与过滤。</p>
<footer>—— 据 Rostedt 对 ftrace 的论述；Linux tracepoint 文档；[printk](/cs/printk-logging) 为重日志对照</footer>
</div>

[printk](/cs/printk-logging) 太重。[NAPI](/cs/napi) 路径不能每包打印。缺口是 **ftrace**：tracepoint、function tracer、ring。

## 问题

`TRACE_EVENT` 定义字段。启用才把探针插入。缺口：buffer per-cpu；过滤 pid/comm；function tracer 用 mcount。本课不把每个事件名列出。

<span class="marginnote">tracefs 是接口。对象是事件流，不是 gdb。与 eBPF 后课可共用 tracepoint。</span>

## 方法

`echo 1 > events/.../enable` → 事件写 per-cpu 缓冲 → `trace_pipe` 读。对照 [perf](/cs/perf-sampling) 后课：perf 可消费同一点。对照 [inotify](/cs/inotify)：一个文件树，一个内核执行点。对照 ASan：一个找空间错，一个找时间线。

<span class="marginnote">per-CPU 缓冲就是「每个核各记各的账本」：事件在哪个核上发生，就写进那个核私有的环形缓冲，读的时候再按核拼接时间线，避免了多核抢同一块内存的锁。</span>

```mermaid
flowchart TD
  TP["tracepoint"] --> FILT["过滤器"]
  FILT --> BUF["per-CPU 环"]
  BUF --> USER["trace_pipe / perf"]
```

## 机制

ftrace 把「生产内核可观测」收成零开销默认 + 按需启用，是调试延迟与调度的主工具。不要写成 APM 产品。与 [PREEMPT_RT](/cs/preempt-rt)：tracer 本身扰动，要测开/关。

错误过滤仍可让缓冲溢出丢事件。

tracepoint 为什么默认零开销？关键在「编译进去但默认关闭」：探针位置在二进制里只是一条空指令，启用时才被改写成真正的调用。

```mermaid
flowchart TD
  NOP["未启用：探针处是一条 nop 空指令"] --> ECHO["echo 1 > enable"]
  ECHO --> PATCH["探针位改写为调用"]
  PATCH --> FIRE["事件路径执行到这里"]
  FIRE --> WRITE["写本核 per-CPU 缓冲"]
  WRITE --> FULL{"缓冲满？"}
  FULL -- "否" --> KEEP["记录保留，等用户读"]
  FULL -- "是" --> DROP["丢弃新事件"]
```

<span class="marginnote">数字实例：把 buffer_size_kb 设成 4096（即每个核 4 MB），在一台 32 核机器上，环形缓冲总共占 32 × 4 MB = 128 MB 内核内存——跟踪是有账单的，核越多越贵。</span>

<span class="marginnote">常见误区：初学者容易把 tracepoint 想成 gdb 那种会让程序停下来的断点。实际上它只往缓冲里记一条数据就立刻继续跑，内核几乎感觉不到；真正会拖慢系统的是 function tracer 这类高频探针。</span>


实现上：per-cpu 缓冲避免全局锁，读的时候可能看到撕裂，工具用页头同步。function tracer 用 mcount/patchable 入口，和 livepatch 抢同一套改代码机制。 读法上只引用[上一课](/cs/printk-logging)的结论，不把对象换成训练推理或限价簿。

本课在操作系统进阶的「安全、启动与调试 / 观测与调试」课序里，对象是 **ftrace 与 tracepoints**。

- 先修只引用，不重导：上一课的结论当公理，本课只补差。
- 五栏不吞并：不把本课写成大模型训练/推理，也不写成限价簿或权重量化。
- 文献用 OSTEP、McKusick、内核文档与具名会议论文；不发明 arXiv 编号。

## 边界

本课不引入所有 tracer 插件。不保证模块事件的稳定性。下一课动态打点：kprobes。


版本字段会变，课序钉的是机制对象「ftrace 与 tracepoints」，不是某一主线内核的结构体名。
后课默认：静态 tracepoint 可低成本启用。动态探针 kprobe/uprobe，下一课。

## 小结

- tracepoint 默认便宜；ftrace 收集到 per-CPU 环。
- 函数跟踪更重，适合短时。
- kprobes 是下一课。
- 出处：ftrace 文档；Rostedt；tracepoint。
