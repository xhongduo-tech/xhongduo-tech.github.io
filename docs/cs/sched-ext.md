---
title: sched_ext 与 BPF 调度器
date: 2026-09-08
section: cs
---

# sched_ext 与 BPF 调度器

<div class="epigraph">
<p>sched_ext 把调度类的入队、选下一个、时间片交给 BPF 程序，使自定义策略不必改内核 C 代码。</p>
<footer>—— 据 Linux sched_ext 文档；eBPF 验证器约束；[XDP](/cs/xdp-ebpf) 为可编程数据面先修</footer>
</div>

[cgroup CPU](/cs/cgroup-cpu-sched) 仍是固定算法。[eBPF](/cs/xdp-ebpf) 已在网络早路径出现。缺口是 **把调度策略变成可加载 BPF**：SCX。

## 问题

数据中心想按作业类型选核，主线 CFS 不够。sched_ext：新调度类，回调 `enqueue`/`dispatch` 在 BPF 中。缺口：verifier 保证不能死循环太久；错误程序踢回 CFS；与 RT/DL 的优先级关系。本课不把每个示例调度器（scx_simple）当产品。<span class="marginnote">BPF verifier 相当于上线前的「逐路径推演审查员」：确认循环有界、不越界访问、不锁死内核——调度回调要是能死循环，整台机器就无药可救。SCX 程序的表达力，就是被「可证明会停」这条约束框住的。</span>

<span class="marginnote">仍在内核态跑 BPF，不是用户态 M:N。失败安全是硬要求。</span>

## 方法

加载 scx 程序 → 任务可选 SCX 策略 → BPF 维护自己的队列（map）。对照 XDP：一个选包命运，一个选任务。对照 [FUSE](/cs/fuse)：FUSE 出核；SCX 仍在核内 JIT。对照 EEVDF：可在 BPF 里实现别的公平。<span class="marginnote">与 FUSE 的对照可以类比「外包给外人」与「雇临时工」：FUSE 把文件系统逻辑搬到用户态，每次操作都要出内核一趟；SCX 把策略写成 BPF，但仍在内核态 JIT 运行——灵活性与 FUSE 同源，热路径却不出门。</span>

```mermaid
flowchart TD
  ENQ["任务入队"] --> BPF["BPF enqueue"]
  TICK["需要下一个"] --> DISP["BPF dispatch"]
  BAD["verifier 或运行失败"] --> CFS["回退默认类"]
```

## 机制

sched_ext 把调度器从编译进内核的算法变成可热更新策略，同时用 verifier 限制危害。它不自动正确。不要写成 AI 调度器营销。与 [PREEMPT_RT](/cs/preempt-rt)：RT 路径通常不交给随意 BPF。

```mermaid
flowchart TD
  LOAD["bpftool 加载 SCX 程序"] --> VF{"verifier 通过?"}
  VF -->|"否"| REJ["加载被拒：内核默认类原地不动"]
  VF -->|"是"| ATT["接管：任务可迁入 SCX 类"]
  ATT --> RUN["enqueue / dispatch 由 BPF 决定"]
  RUN -->|"死循环或触发看门狗"| FB["自动踢回 CFS / EEVDF"]
  RUN -->|"正常"| KEEP["策略持续生效，可热更新替换"]
```

调试：BPF 统计 map；错误会导致抖动回退。<span class="marginnote">这一步如果做错了——回退路径没有兜住——一次 BPF 崩溃会让部分 CPU 上没有可运行的调度器，表现为整机卡死；「踢回默认类」是内核看门狗的设计内行为。排查时先看 dmesg 里的 SCX 回退记录，再怀疑自己的策略逻辑。</span>


实现上：BPF 调度器崩溃或 verifier 失败必须回退，否则机器不可调度。map 里自己维护的队列要处理迁移和热插 CPU。与 RT 类的优先级关系由内核固定。 读法上只引用[上一课](/cs/cgroup-cpu-sched)的结论，不把对象换成训练推理或限价簿。

本课在操作系统进阶的「调度进阶 / 公平、实时与能耗」课序里，对象是 **sched_ext 与 BPF 调度器**。

- 先修只引用，不重导：上一课的结论当公理，本课只补差。
- 五栏不吞并：不把本课写成大模型训练/推理，也不写成限价簿或权重量化。
- 文献用 OSTEP、McKusick、内核文档与具名会议论文；不发明 arXiv 编号。

## 边界

本课不引入所有 kfunc。不保证稳定 ABI。下一课用户态 M:N：协程调度与内核线程的关系。


版本字段会变，课序钉的是机制对象「sched_ext 与 BPF 调度器」，不是某一主线内核的结构体名。
后课默认：公平类策略可 BPF 化。用户态调度与 M:N，下一课。

## 小结

- sched_ext 用 BPF 实现调度回调。
- 失败回退；verifier 限制程序。
- 用户态调度是下一课。
- 出处：Linux sched_ext；eBPF；XDP 对照。
